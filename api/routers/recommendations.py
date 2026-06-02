"""
Endpoints de IA generativa (Claude vía Amazon Bedrock).

Genera recomendaciones ejecutivas y explicaciones de modelo en
lenguaje natural a partir del contexto de negocio.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.shipment import ContextoNegocio, RecomendacionEjecutiva
from src.bedrock_client import get_bedrock_client
from src.config import settings
from src.utils import ahora_iso


router = APIRouter(prefix="/recomendaciones", tags=["ia-generativa"])


@router.post(
    "/ejecutiva",
    response_model=RecomendacionEjecutiva,
    summary="Genera una recomendación ejecutiva con Claude",
)
def recomendacion_ejecutiva(contexto: ContextoNegocio) -> RecomendacionEjecutiva:
    """Invoca Claude vía Bedrock para producir una recomendación gerencial."""
    try:
        cliente = get_bedrock_client()
        texto = cliente.recomendacion_ejecutiva(contexto.model_dump())
        return RecomendacionEjecutiva(
            recomendacion=texto,
            modelo_llm=settings.bedrock_model_id,
            timestamp=ahora_iso(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error invocando el modelo generativo: {exc}",
        )
