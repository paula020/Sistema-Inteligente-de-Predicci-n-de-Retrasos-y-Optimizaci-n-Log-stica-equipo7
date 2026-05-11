"""
Esquemas Pydantic para los endpoints de la API.

Definen el contrato de entrada/salida para predicción de retrasos,
rentabilidad y recomendaciones ejecutivas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------------
# Entrada: datos de una guía
# ----------------------------------------------------------------------------
class GuiaInput(BaseModel):
    """Información mínima de una guía para predecir riesgo de retraso."""

    distancia_km: float = Field(..., gt=0, description="Distancia de la ruta en km")
    peso_kg: float = Field(..., ge=0, description="Peso del envío en kg")
    valor_flete: float = Field(..., ge=0, description="Valor pagado por el flete")
    dias_transito_estimado: int = Field(..., ge=0, description="Días estimados de tránsito")
    ciudad_origen: str = Field(..., description="Ciudad de origen")
    ciudad_destino: str = Field(..., description="Ciudad de destino")
    tipo_servicio: str = Field(..., description="Estándar, express, refrigerado, etc.")
    cliente: str | None = Field(default=None, description="Identificador del cliente")

    model_config = {
        "json_schema_extra": {
            "example": {
                "distancia_km": 450.5,
                "peso_kg": 120.0,
                "valor_flete": 850_000,
                "dias_transito_estimado": 2,
                "ciudad_origen": "Bogota",
                "ciudad_destino": "Medellin",
                "tipo_servicio": "estandar",
                "cliente": "CLI-001",
            }
        }
    }


class LoteGuias(BaseModel):
    """Lote de guías para predicción en bloque."""

    guias: list[GuiaInput]


# ----------------------------------------------------------------------------
# Salida: predicción de retraso
# ----------------------------------------------------------------------------
class PrediccionRetraso(BaseModel):
    """Resultado de la predicción de retraso para una guía."""

    probabilidad_retraso: float = Field(..., ge=0, le=1)
    prediccion: int = Field(..., description="0 = a tiempo, 1 = retrasado")
    nivel_riesgo: str = Field(..., description="BAJO | MEDIO | ALTO | CRITICO")
    costo_esperado_retraso: float = Field(
        ..., description="Costo financiero esperado del retraso (COP)"
    )
    timestamp: str


# ----------------------------------------------------------------------------
# Recomendación ejecutiva (IA generativa)
# ----------------------------------------------------------------------------
class ContextoNegocio(BaseModel):
    """KPIs de negocio que alimentan la recomendación de Claude."""

    tasa_retraso_actual: float
    margen_promedio_pct: float
    rutas_no_rentables: int
    top_ciudades: list[str]
    impacto_financiero_mensual: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "tasa_retraso_actual": 0.18,
                "margen_promedio_pct": 0.12,
                "rutas_no_rentables": 7,
                "top_ciudades": ["Bogota", "Medellin", "Cali"],
                "impacto_financiero_mensual": 145_000_000,
            }
        }
    }


class RecomendacionEjecutiva(BaseModel):
    """Respuesta generada por Claude para el dashboard ejecutivo."""

    recomendacion: str
    modelo_llm: str
    timestamp: str
