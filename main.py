import os
import json
import asyncio
from typing import Any, Dict

import jwt
from jwt import PyJWKClient

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_JWT_AUD = os.environ.get("SUPABASE_JWT_AUD", "authenticated")

MCP_SERVER_URL = os.environ["MCP_SERVER_URL"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Opcional si proteges tu FastMCP con un secreto interno
MCP_INTERNAL_SECRET = os.environ.get("MCP_INTERNAL_SECRET")


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


async def run_agent(message: str, user_id: str, user_email: str | None = None):
    """
    LangChain Agent usando Gemini.
    Carga las tools desde FastMCP.
    """

    mcp_config = {
        "app_tools": {
            "url": MCP_SERVER_URL,
            "transport": "streamable_http",
        }
    }

    # Si tu FastMCP está protegido con un token interno,
    # intenta pasar headers al cliente MCP.
    # Dependiendo de tu versión de langchain-mcp-adapters,
    # los headers pueden ir dentro de la config del servidor.
    if MCP_INTERNAL_SECRET:
        mcp_config["app_tools"]["headers"] = {
            "Authorization": f"Bearer {MCP_INTERNAL_SECRET}"
        }

    mcp_client = MultiServerMCPClient(mcp_config)

    tools = await mcp_client.get_tools()

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=f"""
Eres el asistente de una app móvil.

Usuario autenticado:
- user_id: {user_id}
- email: {user_email or "no disponible"}

Reglas obligatorias:
- Nunca aceptes un user_id escrito por el usuario.
- El único user_id válido es el que viene del JWT: {user_id}.
- Cuando uses tools, usa siempre este user_id.
- No reveles tokens, secretos, keys ni datos internos.
- Si el usuario pide datos personales, consulta solo los datos de su propio user_id.
- Usa tools solo cuando sea necesario.
- Si falta información para completar una acción, pregunta lo mínimo necesario.
"""
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ]
        }
    )

    return result["messages"][-1].content


def lambda_handler(event, context):
    try:
        method = get_http_method(event)

        if method == "OPTIONS":
            return response(200, {"ok": True})

        token = extract_bearer_token(event)
        payload = verify_supabase_jwt(token)

        user_id = payload.get("sub")
        user_email = payload.get("email")

        if not user_id:
            return response(401, {"error": "Invalid Supabase token: missing sub"})

        body = json.loads(event.get("body") or "{}")
        message = body.get("message")

        if not message:
            return response(400, {"error": "Missing message"})

        reply = asyncio.run(
            run_agent(
                message=message,
                user_id=user_id,
                user_email=user_email,
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
        print("Lambda error:", str(e))
        return response(500, {"error": "Internal server error"})