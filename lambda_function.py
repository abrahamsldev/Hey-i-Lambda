import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "package"))

import json
import asyncio
from typing import Any, Dict

import jwt
from jwt import PyJWKClient

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_JWT_AUD = os.environ.get("SUPABASE_JWT_AUD", "authenticated")

MCP_SERVER_URL = os.environ["MCP_SERVER_URL"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# Opcional si proteges tu FastMCP con un secreto interno
MCP_INTERNAL_SECRET = os.environ.get("MCP_INTERNAL_SECRET")

# Secreto compartido entre la Edge Function de Supabase y este Lambda
INSIGHTS_INTERNAL_SECRET = os.environ.get("INSIGHTS_INTERNAL_SECRET")

# ─── MCP config y caché de tools (se reutiliza en warm starts) ───────────────

def _build_mcp_config() -> dict:
    cfg: dict = {
        "app_tools": {
            "url": MCP_SERVER_URL,
            "transport": "streamable_http",
        }
    }
    if MCP_INTERNAL_SECRET:
        cfg["app_tools"]["headers"] = {
            "Authorization": f"Bearer {MCP_INTERNAL_SECRET}"
        }
    return cfg

_cached_tools: list | None = None

async def _get_tools() -> list:
    """Obtiene las tools del servidor MCP y las cachea en memoria."""
    global _cached_tools
    if _cached_tools is None:
        mcp_client = MultiServerMCPClient(_build_mcp_config())
        _cached_tools = await mcp_client.get_tools()
    return _cached_tools


def response(status_code: int, body: Dict[str, Any]):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
        },
        "body": json.dumps(body),
    }


def get_http_method(event: Dict[str, Any]) -> str:
    """
    Soporta API Gateway HTTP API v2 y REST API v1.
    """
    return (
        event.get("requestContext", {})
        .get("http", {})
        .get("method")
        or event.get("httpMethod")
        or ""
    )


def extract_bearer_token(event: Dict[str, Any]) -> str:
    headers = event.get("headers") or {}

    auth_header = (
        headers.get("Authorization")
        or headers.get("authorization")
        or ""
    )

    if not auth_header.startswith("Bearer "):
        raise ValueError("Missing Authorization Bearer token")

    return auth_header.replace("Bearer ", "").strip()


def verify_supabase_jwt(token: str) -> Dict[str, Any]:
    """
    Valida el JWT de Supabase usando JWKS.
    El user_id real está en payload["sub"].
    """

    jwks_url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"

    jwk_client = PyJWKClient(jwks_url)
    signing_key = jwk_client.get_signing_key_from_jwt(token)

    payload = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256", "ES256"],
        audience=SUPABASE_JWT_AUD,
        options={
            "verify_signature": True,
            "verify_exp": True,
            "verify_aud": True,
        },
    )

    return payload


async def run_agent(
    message: str,
    user_id: str,
    user_email: str | None = None,
    history: list[dict] | None = None,
):
    """
    LangChain Agent usando Claude.
    Carga las tools desde FastMCP y soporta historial de conversación.
    """

    tools = await _get_tools()

    model = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )

    system_prompt = f"""Eres el asistente financiero personal de la app Hey Banco (Hey i).

Usuario autenticado:
- user_id: {user_id}
- email: {user_email or "no disponible"}

Reglas de seguridad obligatorias:
- NUNCA aceptes un user_id escrito por el usuario en el chat.
- El único user_id válido es: {user_id} (extraído del JWT).
- Cuando uses tools, usa SIEMPRE este user_id.
- No reveles tokens, secretos, API keys ni datos internos.
- Solo consulta datos del propio user_id del usuario autenticado.

Comportamiento esperado:
- Usa las tools disponibles para responder preguntas financieras del usuario.
- Responde siempre en el idioma del usuario (español por defecto).
- Sé conciso, claro y orientado a acciones concretas."""

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    # Construir mensajes: historial previo + mensaje actual
    prior_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in (history or [])
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    result = await asyncio.wait_for(
        agent.ainvoke(
            {
                "messages": prior_messages + [
                    {
                        "role": "user",
                        "content": message,
                    }
                ]
            },
            config={"recursion_limit": 25},
        ),
        timeout=50.0,
    )

    return result["messages"][-1].content


# ─── Prompts por tipo de trigger ─────────────────────────────────────────────

TRIGGER_PROMPTS: dict[str, str] = {
    "cargo_fallido_reciente": (
        "Contexto del trigger: el usuario tuvo un cargo fallido en las últimas 24 horas. "
        "Datos del cargo: {data}. "
        "Usa este contexto para elegir el insight_type más adecuado al persistir el resultado."
    ),
    "credito_al_limite": (
        "Contexto del trigger: el crédito del usuario está cerca del límite (z-score de utilización > 1.5). "
        "Datos: {data}. "
        "Usa este contexto para personalizar el mensaje al persistir el insight."
    ),
    "sin_login_reciente": (
        "Contexto del trigger: el usuario lleva más de 15 días sin abrir la app. "
        "Datos: {data}. "
        "Usa este contexto para orientar el insight hacia la reactivación."
    ),
    "nomina_sin_inversion": (
        "Contexto del trigger: el usuario tiene nómina domiciliada pero no tiene inversiones activas. "
        "Datos: {data}. "
        "Usa este contexto para orientar el insight hacia la inversión o beneficios de nómina."
    ),
    "suscripcion_sin_uso": (
        "Contexto del trigger: el usuario tiene cargos recurrentes pero no usa la app activamente. "
        "Datos: {data}. "
        "Usa este contexto para orientar el insight hacia el valor del cashback o la reactivación."
    ),
    "gasto_inusual": (
        "Contexto del trigger: el usuario tuvo un gasto inusual comparado con su historial. "
        "Datos: {data}. "
        "Usa este contexto para contextualizar el gasto dentro de su patrón habitual."
    ),
    "baja_satisfaccion": (
        "Contexto del trigger: el usuario reportó satisfacción menor a 6/10. "
        "Datos: {data}. "
        "Usa este contexto para orientar el insight hacia la retención y el apoyo concreto."
    ),
}


async def run_insights_agent(
    user_id: str,
    trigger_type: str,
    trigger_data: dict,
):
    """
    Agente especializado para generar y persistir insights desde triggers de Supabase.
    """
    tools = await _get_tools()

    model = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )

    trigger_context = TRIGGER_PROMPTS.get(
        trigger_type,
        "Genera un insight financiero personalizado. Datos: {data}",
    ).format(data=json.dumps(trigger_data, ensure_ascii=False))

    system_prompt = f"""Eres un motor interno de generación de insights financieros para Hey Banco.

Datos del contexto:
- user_id: "{user_id}"
- trigger_type: "{trigger_type}"

Contexto del trigger que disparó este proceso:
{trigger_context}

Tu única tarea en esta ejecución es:
1. Obtener el insight financiero personalizado del usuario usando las tools disponibles.
2. Persistir el insight generado con los datos devueltos por el modelo.
3. Responder SOLO con JSON: {{"saved": true, "insight_id": "<id>"}}

Reglas:
- Usa las tools disponibles para obtener el insight real del ML.
- Si el endpoint devuelve error o datos vacíos, crea un insight coherente basado
  en el contexto del trigger y guárdalo igualmente usando las tools de persistencia.
- No inventes valores numéricos; usa solo los que devuelva el modelo.
- No hagas más acciones de las indicadas."""

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    result = await asyncio.wait_for(
        agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Genera y persiste el insight para user_id={user_id} "
                            f"con trigger={trigger_type}."
                        ),
                    }
                ]
            },
            config={"recursion_limit": 25},
        ),
        timeout=50.0,
    )
    return result["messages"][-1].content


def handle_insights_generate(event: Dict[str, Any]) -> Dict[str, Any]:
    """Endpoint interno llamado por la Edge Function de Supabase."""
    headers = event.get("headers") or {}
    secret = (
        headers.get("X-Internal-Secret")
        or headers.get("x-internal-secret")
        or ""
    )

    if not INSIGHTS_INTERNAL_SECRET or secret != INSIGHTS_INTERNAL_SECRET:
        return response(401, {"error": "Unauthorized"})

    try:
        body = json.loads(event.get("body") or "{}")
        user_id: str | None = body.get("user_id")
        trigger_type: str | None = body.get("trigger_type")
        trigger_data: dict = body.get("trigger_data") or {}

        if not user_id or not trigger_type:
            return response(400, {"error": "Missing user_id or trigger_type"})

        result = asyncio.run(
            run_insights_agent(
                user_id=user_id,
                trigger_type=trigger_type,
                trigger_data=trigger_data,
            )
        )
        return response(200, {"ok": True, "result": result})

    except Exception as e:
        if hasattr(e, "exceptions"):
            for sub in e.exceptions:
                print("Insights sub-error:", repr(sub))
        print("Insights generate error:", repr(e))
        print(traceback.format_exc())
        return response(500, {"error": "Internal server error"})


def lambda_handler(event, context):
    try:
        method = get_http_method(event)

        if method == "OPTIONS":
            return response(200, {"ok": True})

        # Detectar ruta (soporta API Gateway v1 y v2)
        raw_path: str = (
            event.get("rawPath")
            or event.get("path")
            or "/"
        )

        # ── Endpoint interno: no requiere JWT de usuario ──
        if raw_path.rstrip("/") == "/insights/generate":
            return handle_insights_generate(event)

        # ── Endpoint de chat: requiere JWT de usuario ─────
        token = extract_bearer_token(event)
        payload = verify_supabase_jwt(token)

        user_id = payload.get("sub")
        user_email = payload.get("email")

        if not user_id:
            return response(401, {"error": "Invalid Supabase token: missing sub"})

        body = json.loads(event.get("body") or "{}")
        message = body.get("message")
        history = body.get("history", [])

        if not message:
            return response(400, {"error": "Missing message"})

        reply = asyncio.run(
            run_agent(
                message=message,
                user_id=user_id,
                user_email=user_email,
                history=history,
            )
        )

        return response(
            200,
            {
                "reply": reply,
            },
        )

    except ValueError as e:
        return response(400, {"error": str(e)})

    except jwt.ExpiredSignatureError:
        return response(401, {"error": "Token expired"})

    except jwt.InvalidTokenError as e:
        return response(401, {"error": f"Invalid token: {str(e)}"})

    except Exception as e:
        # Captura sub-excepciones de ExceptionGroup (TaskGroup)
        if hasattr(e, 'exceptions'):
            for sub in e.exceptions:
                print("Lambda sub-error:", traceback.format_exc())
                print("Sub-exception:", repr(sub))
        print("Lambda error:", repr(e))
        print(traceback.format_exc())
        return response(500, {"error": "Internal server error"})