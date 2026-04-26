import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "package"))

import json
import asyncio
from typing import Any, Dict

import anthropic as anthropic_sdk
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


# ─── Prompts para generación de insight_text por tipo de trigger ────────────
# Solo se usa el contexto del usuario + el trigger para generar el párrafo.

INSIGHT_SYSTEM_PROMPT = """Eres el motor de recomendaciones de Hey Banco.
Escribe exactamente 2-3 oraciones en español mexicano, tono directo y cercano.

Reglas:
- Usa los números concretos del usuario: montos, porcentajes, días, productos activos.
- Sin saludos ni cierres. Solo el texto de recomendación.
- Máximo 55 palabras.
- La última oración es un CTA directo y específico."""

TRIGGER_USER_PROMPTS: dict[str, str] = {
    "cargo_fallido_reciente": (
        "Trigger: cargo fallido reciente.\n"
        "Datos: {profile}\n"
        "Escribe el texto: menciona el impacto en su cuenta, sugiere una acción concreta (revisar saldo o actualizar tarjeta), "
        "cierra invitando a resolver desde la app."
    ),
    "credito_al_limite": (
        "Trigger: crédito cerca del límite.\n"
        "Datos: {profile}\n"
        "Escribe el texto: menciona el porcentaje exacto de utilización, sugiere un monto de pago realista con base en su ingreso mensual, "
        "cierra con CTA para ver plan de pago."
    ),
    "sin_login_reciente": (
        "Trigger: usuario sin actividad en la app.\n"
        "Datos: {profile}\n"
        "Escribe el texto: menciona los días exactos sin actividad, nombra un beneficio concreto que tiene sin aprovechar, "
        "cierra invitándolo a entrar hoy."
    ),
    "nomina_sin_inversion": (
        "Trigger: tiene nómina pero no invierte.\n"
        "Datos: {profile}\n"
        "Escribe el texto: menciona su ingreso mensual y sugiere un porcentaje o monto a invertir, "
        "cierra con CTA para activar Hey Inversión."
    ),
    "suscripcion_sin_uso": (
        "Trigger: suscripciones activas, app sin uso.\n"
        "Datos: {profile}\n"
        "Escribe el texto: menciona el gasto anual o mensual en digital, indica el cashback que podría recuperar, "
        "cierra invitando a activar el beneficio."
    ),
    "gasto_inusual": (
        "Trigger: gasto inusual detectado.\n"
        "Datos: {profile}\n"
        "Escribe el texto: alerta de forma tranquila sobre el movimiento, ofrece reportarlo, "
        "cierra sugiriendo activar alertas de gasto."
    ),
    "baja_satisfaccion": (
        "Trigger: satisfacción menor a 6/10.\n"
        "Datos: {profile}\n"
        "Escribe el texto: reconoce sin ser defensivo, ofrece un canal concreto (chat en la app), "
        "cierra con mensaje de compromiso de Hey."
    ),
}

# ─── Pipeline directo: tipo de trigger → insight_type y texto ────────────────

TRIGGER_TO_INSIGHT_TYPE: dict[str, str] = {
    "cargo_fallido_reciente":   "retention_churn_risk",
    "credito_al_limite":        "financial_stress_relief",
    "sin_login_reciente":       "retention_reactivation",
    "nomina_sin_inversion":     "upsell_investment",
    "suscripcion_sin_uso":      "upsell_digital",
    "gasto_inusual":            "financial_stress_relief",
    "baja_satisfaccion":        "retention_churn_risk",
}

TRIGGER_INSIGHT_TEXTS: dict[str, str] = {
    "cargo_fallido_reciente": (
        "Detectamos que uno de tus pagos no pudo procesarse recientemente. "
        "Asegúrate de que tu cuenta tenga saldo suficiente o que tu tarjeta esté activa. "
        "Si necesitas ayuda, escríbenos en el chat y lo resolvemos juntos."
    ),
    "credito_al_limite": (
        "Tu crédito está cerca del límite. Reducir tu saldo puede mejorar tu salud financiera "
        "y abrirte nuevas oportunidades. Podemos ayudarte a crear un plan de pago."
    ),
    "sin_login_reciente": (
        "¡Te extrañamos! Tienes beneficios activos que no has aprovechado. "
        "Entra a la app para ver tus cashbacks, inversiones disponibles y más."
    ),
    "nomina_sin_inversion": (
        "Recibes tu nómina en Hey pero aún no tienes una inversión activa. "
        "Con Hey Inversión puedes poner a trabajar tu dinero desde el primer peso. ¿Empezamos?"
    ),
    "suscripcion_sin_uso": (
        "Tienes suscripciones activas que califican para cashback. "
        "Activa tus beneficios digitales y recupera hasta el 2% en cada cargo automático."
    ),
    "gasto_inusual": (
        "Detectamos un movimiento diferente a tu patrón habitual. "
        "Si no reconoces este cargo, puedes reportarlo desde la app en segundos."
    ),
    "baja_satisfaccion": (
        "Queremos que tengas la mejor experiencia. Si algo no funcionó como esperabas, "
        "cuéntanos en el chat y lo resolvemos juntos. Tu opinión mejora Hey para todos."
    ),
}


async def _generate_insight_text(
    trigger_type: str,
    profile: dict,
) -> str:
    """
    Llama a Claude directamente (sin agente) para generar un párrafo
    personalizado. Timeout de 20s. Si falla, devuelve el texto fallback.
    """
    client = anthropic_sdk.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    prompt_template = TRIGGER_USER_PROMPTS.get(trigger_type)
    if not prompt_template:
        return TRIGGER_INSIGHT_TEXTS.get(
            trigger_type,
            "Tienes una recomendación personalizada disponible en tu app.",
        )

    # Construir resumen legible del perfil con las métricas clave
    profile_summary = (
        f"Segmento: {profile.get('segment_name', 'Sin segmento')}. "
        f"Ingreso mensual: ${profile.get('ingreso_mensual_mxn', 0):,.0f} MXN. "
        f"Gasto anual: ${profile.get('gasto_total_anual_mxn', 0):,.0f} MXN. "
        f"Utilización de crédito: {profile.get('utilizacion_credito_pct', 0):.1f}%. "
        f"Tasa de fallos: {profile.get('tasa_fallos_pct', 0):.1f}%. "
        f"Días sin abrir la app: {profile.get('dias_desde_ultimo_login', 0)}. "
        f"Productos activos: {profile.get('num_productos_activos', 0)}. "
        f"Nómina domiciliada: {'Sí' if profile.get('nomina_domiciliada') else 'No'}. "
        f"Hey Pro: {'Sí' if profile.get('es_hey_pro') else 'No'}. "
        f"Ocupación: {profile.get('ocupacion', 'No disponible')}."
    )

    user_prompt = prompt_template.format(profile=profile_summary)

    try:
        msg = await asyncio.wait_for(
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=180,
                system=INSIGHT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            ),
            timeout=20.0,
        )
        generated = msg.content[0].text.strip()
        return generated if generated else TRIGGER_INSIGHT_TEXTS.get(trigger_type, "")
    except Exception as gen_err:
        print(f"[generate_insight_text] Claude call failed: {gen_err}")
        return TRIGGER_INSIGHT_TEXTS.get(
            trigger_type,
            "Tienes una recomendación personalizada disponible en tu app.",
        )


async def run_insights_direct(
    user_id: str,
    trigger_type: str,
    trigger_data: dict,
) -> dict:
    """
    Pipeline directo sin LLM: classify_user_segment → plantilla → save_user_insight.
    Evita timeouts y rate limits de Anthropic.
    """
    # 1. Obtener segmento existente (sin forzar reclasificación)
    segment_name: str | None = None
    try:
        seg = await call_mcp_tool("classify_user_segment", {
            "user_id": user_id,
            "force_reclassify": False,
        })
        if seg.get("ok"):
            segment_name = (
                (seg.get("existing_segment") or {}).get("segmento")
                or seg.get("segmento")
            )
    except Exception as seg_err:
        print(f"[insights_direct] segment lookup failed: {seg_err}")

    # 2. Generar insight_text personalizado con Claude (solo el párrafo)
    insight_type = TRIGGER_TO_INSIGHT_TYPE.get(trigger_type)

    # Construir perfil mínimo con los datos disponibles
    profile: dict = {"segment_name": segment_name or "Sin segmento"}
    profile.update(trigger_data)  # incluye datos del trigger (ingreso, días, etc.)

    insight_text = await _generate_insight_text(trigger_type, profile)

    # 3. Persistir via MCP (sin LLM)
    save_result = await call_mcp_tool("save_user_insight", {
        "user_id": user_id,
        "trigger_type": trigger_type,
        "insight_text": insight_text,
        "segment_name": segment_name,
        "insight_type": insight_type,
    })
    return save_result


# run_insights_agent fue reemplazado por run_insights_direct (pipeline sin LLM)
# para evitar timeouts y rate limits de Anthropic. Ver run_insights_direct más abajo.

async def call_mcp_tool(tool_name: str, tool_input: dict):
    """Invoca una tool del servidor MCP directamente, sin pasar por el agente LLM."""
    tools = await _get_tools()
    tool = next((t for t in tools if t.name == tool_name), None)
    if tool is None:
        raise ValueError(f"MCP tool '{tool_name}' no encontrada")
    result = await tool.ainvoke(tool_input)
    # langchain_mcp_adapters puede devolver el resultado como JSON string
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            pass
    return result


def handle_spending_dashboard(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /get_spending_dashboard — requiere JWT de usuario."""
    try:
        token = extract_bearer_token(event)
        payload = verify_supabase_jwt(token)
        user_id = payload.get("sub")
        if not user_id:
            return response(401, {"error": "Token inválido: falta sub"})

        result = asyncio.run(
            call_mcp_tool("get_spending_dashboard", {"user_id": user_id})
        )
        return response(200, {"structuredContent": result})
    except ValueError as e:
        return response(400, {"error": str(e)})
    except jwt.ExpiredSignatureError:
        return response(401, {"error": "Token expirado"})
    except jwt.InvalidTokenError as e:
        return response(401, {"error": f"Token inválido: {str(e)}"})
    except Exception as e:
        print("handle_spending_dashboard error:", repr(e))
        print(traceback.format_exc())
        return response(500, {"error": "Internal server error"})


def handle_savings_dashboard(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /get_savings_dashboard — requiere JWT de usuario."""
    try:
        token = extract_bearer_token(event)
        payload = verify_supabase_jwt(token)
        user_id = payload.get("sub")
        if not user_id:
            return response(401, {"error": "Token inválido: falta sub"})

        result = asyncio.run(
            call_mcp_tool("get_savings_dashboard", {"user_id": user_id})
        )
        return response(200, {"structuredContent": result})
    except ValueError as e:
        return response(400, {"error": str(e)})
    except jwt.ExpiredSignatureError:
        return response(401, {"error": "Token expirado"})
    except jwt.InvalidTokenError as e:
        return response(401, {"error": f"Token inválido: {str(e)}"})
    except Exception as e:
        print("handle_savings_dashboard error:", repr(e))
        print(traceback.format_exc())
        return response(500, {"error": "Internal server error"})


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
            run_insights_direct(
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

        # ── Dashboards: requieren JWT ──────────────────────
        if raw_path.rstrip("/") == "/get_spending_dashboard":
            return handle_spending_dashboard(event)

        if raw_path.rstrip("/") == "/get_savings_dashboard":
            return handle_savings_dashboard(event)

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