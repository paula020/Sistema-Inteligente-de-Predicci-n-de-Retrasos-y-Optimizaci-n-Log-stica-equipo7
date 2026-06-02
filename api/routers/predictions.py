"""
Endpoints de predicción de retrasos.

Expone predicciones individuales y por lote, además del costo
financiero esperado del riesgo predicho.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import GuiaInput, LoteGuias, PrediccionRetraso
from src.financial_impact import costo_esperado_por_probabilidad
from src.predict import predecir_retraso
from src.utils import ahora_iso


router = APIRouter(prefix="/predicciones", tags=["predicciones"])


def _guia_a_features(guia: GuiaInput) -> dict:
    """Convierte el schema de entrada al formato esperado por el modelo."""
    return guia.model_dump()


@router.post(
    "/retraso",
    response_model=PrediccionRetraso,
    summary="Predice el riesgo de retraso para una guía",
)
def predecir_retraso_guia(guia: GuiaInput) -> PrediccionRetraso:
    """Predicción individual."""
    try:
        resultado = predecir_retraso(_guia_a_features(guia))
        prob = resultado["probabilidad_retraso"][0]
        costo = costo_esperado_por_probabilidad(
            probabilidad_retraso=prob,
            valor_flete=guia.valor_flete,
            dias_estimados=max(guia.dias_transito_estimado, 1),
        )
        return PrediccionRetraso(
            probabilidad_retraso=prob,
            prediccion=resultado["prediccion"][0],
            nivel_riesgo=resultado["nivel_riesgo"][0],
            costo_esperado_retraso=costo,
            timestamp=ahora_iso(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible: {exc}",
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/retraso/lote",
    summary="Predice riesgos para un lote de guías",
)
def predecir_lote_guias(lote: LoteGuias) -> dict:
    """Predicción por lote (batch)."""
    try:
        df = pd.DataFrame([_guia_a_features(g) for g in lote.guias])
        resultado = predecir_retraso(df)
        return {
            "total_guias": len(df),
            "resultados": resultado,
            "timestamp": ahora_iso(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {exc}")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))
