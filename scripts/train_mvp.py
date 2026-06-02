"""
Script CLI para entrenar el modelo MVP de predicción de retrasos.

Uso:
    python scripts/train_mvp.py
    python scripts/train_mvp.py --raw            # parte del CSV crudo
    python scripts/train_mvp.py --modelos xgboost lightgbm

Salida esperada:
    - models/delay_model.pkl
    - models/feature_columns.json

Pre-requisitos:
    - Tener `data/raw/DataCoSupplyChainDataset.csv` (si usas `--raw`)
      o `data/processed/dataco_clean.parquet` (modo por defecto).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permitir importar `src` cuando se ejecuta desde la raíz del proyecto
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import pandas as pd
from loguru import logger

from src.preprocessing import limpiar_dataco
from src.train_model import entrenar_modelo_delay_mvp
from src.utils import configurar_logger


def _cargar_dataset(usar_raw: bool) -> pd.DataFrame:
    """Carga el dataset desde processed (default) o desde raw."""
    if usar_raw:
        logger.info("Modo --raw: ejecutando pipeline de limpieza desde data/raw/")
        return limpiar_dataco(guardar=True)

    ruta = RAIZ / "data" / "processed" / "dataco_clean.parquet"
    if not ruta.exists():
        logger.warning(
            f"No existe {ruta}. Generándolo desde data/raw/ (--raw implícito)..."
        )
        return limpiar_dataco(guardar=True)

    logger.info(f"Cargando dataset limpio desde {ruta}")
    return pd.read_parquet(ruta)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Entrena el modelo MVP de predicción de retrasos logísticos."
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Forzar limpieza desde data/raw/DataCoSupplyChainDataset.csv",
    )
    parser.add_argument(
        "--modelos",
        nargs="+",
        choices=["logistic_regression", "random_forest", "xgboost", "lightgbm"],
        default=None,
        help="Modelos a probar (por defecto: los 4).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fracción del dataset para test (default: 0.2)",
    )
    args = parser.parse_args()

    configurar_logger("train_mvp")

    try:
        df = _cargar_dataset(usar_raw=args.raw)
    except FileNotFoundError as exc:
        logger.error(f"No se pudo cargar el dataset: {exc}")
        logger.error(
            "Asegúrate de tener 'data/raw/DataCoSupplyChainDataset.csv' "
            "descargado desde Kaggle."
        )
        return 1

    resultado = entrenar_modelo_delay_mvp(
        df,
        modelos_a_probar=args.modelos,
        test_size=args.test_size,
        guardar=True,
    )

    logger.info("\nComparativa de modelos:")
    logger.info("\n" + resultado["comparativa"].to_string(index=False))

    logger.success(
        f"\nModelo ganador: '{resultado['nombre_modelo']}'\n"
        f"Pipeline   -> {resultado['ruta_modelo']}\n"
        f"Metadata   -> {resultado['ruta_columnas']}\n"
        f"Métricas   -> {resultado['metricas']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
