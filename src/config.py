"""
Configuración centralizada del proyecto SmartDelay AI 360.

Carga variables de entorno desde `.env` y expone una instancia única
de `Settings` consumida por todos los módulos (src, api, app).
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Raíz del proyecto (dos niveles arriba de este archivo: src/config.py)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Configuración global tipada del proyecto."""

    # --- Entorno ---------------------------------------------------------------
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- API -------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_title: str = Field(default="SmartDelay AI 360 API", alias="API_TITLE")
    api_version: str = Field(default="1.0.0", alias="API_VERSION")

    # --- Streamlit -------------------------------------------------------------
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")

    # --- MLflow ----------------------------------------------------------------
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="smartdelay-ai-360", alias="MLFLOW_EXPERIMENT_NAME"
    )

    # --- AWS / Bedrock ---------------------------------------------------------
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_session_token: str | None = Field(default=None, alias="AWS_SESSION_TOKEN")

    bedrock_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20240620-v1:0",
        alias="BEDROCK_MODEL_ID",
    )
    bedrock_max_tokens: int = Field(default=2048, alias="BEDROCK_MAX_TOKENS")
    bedrock_temperature: float = Field(default=0.3, alias="BEDROCK_TEMPERATURE")

    # --- Rutas de datos --------------------------------------------------------
    data_raw_path: Path = Field(default=Path("data/raw"), alias="DATA_RAW_PATH")
    data_processed_path: Path = Field(
        default=Path("data/processed"), alias="DATA_PROCESSED_PATH"
    )
    data_external_path: Path = Field(
        default=Path("data/external"), alias="DATA_EXTERNAL_PATH"
    )
    models_path: Path = Field(default=Path("models"), alias="MODELS_PATH")
    reports_path: Path = Field(default=Path("reports"), alias="REPORTS_PATH")

    # --- Configuración de carga ------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def resolve_path(self, relative: Path) -> Path:
        """Resuelve una ruta relativa contra la raíz del proyecto."""
        path = Path(relative)
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna la instancia única (singleton) de configuración."""
    return Settings()


# Instancia global lista para importar
settings = get_settings()
