"""
Cliente de Amazon Bedrock para IA generativa con Claude.

Encapsula las llamadas a Anthropic Claude vía boto3 para generar
recomendaciones ejecutivas, explicaciones de modelos y análisis
gerenciales a partir de los resultados analíticos.
"""

from __future__ import annotations

import json
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger

from src.config import settings


class BedrockClaudeClient:
    """Cliente liviano para invocar Claude en Amazon Bedrock."""

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.model_id = model_id or settings.bedrock_model_id
        self.region = region or settings.aws_region
        self.max_tokens = max_tokens or settings.bedrock_max_tokens
        self.temperature = (
            temperature if temperature is not None else settings.bedrock_temperature
        )
        self._client = self._crear_cliente()

    def _crear_cliente(self) -> Any:
        """Inicializa el cliente boto3 para Bedrock Runtime."""
        kwargs: dict[str, Any] = {"region_name": self.region}
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
            if settings.aws_session_token:
                kwargs["aws_session_token"] = settings.aws_session_token

        return boto3.client("bedrock-runtime", **kwargs)

    # ------------------------------------------------------------------
    # Invocación principal
    # ------------------------------------------------------------------
    def generar_texto(
        self,
        prompt: str,
        contexto_sistema: str | None = None,
    ) -> str:
        """
        Envía un prompt a Claude y devuelve la respuesta en texto plano.

        Parameters
        ----------
        prompt : str
            Pregunta o instrucción del usuario.
        contexto_sistema : str | None
            Mensaje de sistema con el rol/expertise esperado.
        """
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if contexto_sistema:
            body["system"] = contexto_sistema

        try:
            respuesta = self._client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            payload = json.loads(respuesta["body"].read())
            return payload["content"][0]["text"]
        except (BotoCoreError, ClientError) as exc:
            logger.error(f"Error invocando Bedrock: {exc}")
            raise

    # ------------------------------------------------------------------
    # Casos de uso del negocio
    # ------------------------------------------------------------------
    def recomendacion_ejecutiva(self, contexto_negocio: dict) -> str:
        """Genera una recomendación gerencial a partir de KPIs del negocio."""
        prompt = (
            "Eres un consultor senior en logística. Con base en los siguientes "
            "indicadores del negocio, entrega:\n"
            "1. Diagnóstico ejecutivo (3 bullets)\n"
            "2. Riesgos críticos identificados\n"
            "3. Recomendaciones priorizadas con impacto estimado\n\n"
            f"Indicadores:\n{json.dumps(contexto_negocio, indent=2, ensure_ascii=False)}"
        )
        sistema = (
            "Eres un experto en analítica logística, optimización de rutas y "
            "rentabilidad operativa. Respondes en español, con tono ejecutivo, "
            "claro y orientado a la acción."
        )
        return self.generar_texto(prompt, contexto_sistema=sistema)

    def explicar_prediccion(self, datos_guia: dict, prediccion: dict) -> str:
        """Explica en lenguaje natural por qué una guía tiene cierto riesgo."""
        prompt = (
            "Explica de forma clara y breve a un gerente de operaciones por qué "
            "esta guía tiene este nivel de riesgo y qué acciones inmediatas tomar.\n\n"
            f"Datos de la guía:\n{json.dumps(datos_guia, indent=2, ensure_ascii=False)}\n\n"
            f"Predicción del modelo:\n{json.dumps(prediccion, indent=2, ensure_ascii=False)}"
        )
        return self.generar_texto(prompt)


# ----------------------------------------------------------------------------
# Singleton de conveniencia
# ----------------------------------------------------------------------------
_cliente_global: BedrockClaudeClient | None = None


def get_bedrock_client() -> BedrockClaudeClient:
    """Retorna una instancia única (lazy) del cliente de Bedrock."""
    global _cliente_global
    if _cliente_global is None:
        _cliente_global = BedrockClaudeClient()
    return _cliente_global
