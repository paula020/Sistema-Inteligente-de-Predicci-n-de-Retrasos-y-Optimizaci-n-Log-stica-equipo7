"""Schemas Pydantic compartidos por todos los routers."""
from api.schemas.shipment import (
    GuiaInput,
    PrediccionRetraso,
    LoteGuias,
    RecomendacionEjecutiva,
)

__all__ = ["GuiaInput", "PrediccionRetraso", "LoteGuias", "RecomendacionEjecutiva"]
