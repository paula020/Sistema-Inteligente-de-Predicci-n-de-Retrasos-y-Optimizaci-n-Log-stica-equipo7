# SmartDelay AI 360

### Plataforma Inteligente para Predicci├│n de Retrasos, Rentabilidad Log├¡stica y Toma de Decisiones Empresariales

---

## Descripci├│n del Proyecto

**SmartDelay AI 360** es una plataforma de inteligencia artificial y ciencia de datos dise├▒ada para transformar la operaci├│n log├¡stica de empresas de transporte, permitiendo pasar de un modelo reactivo a uno **predictivo, rentable y estrat├®gico**.

El sistema integra:

- Predicci├│n de retrasos con Machine Learning (XGBoost)
- An├ílisis de costos directos e indirectos por ruta
- C├ílculo de rentabilidad, margen y ROI
- Proyecci├│n de ventas (forecast con Prophet)
- Identificaci├│n de mercados potenciales
- Simulaci├│n de impacto financiero
- Recomendaciones ejecutivas con IA generativa (Claude v├¡a Amazon Bedrock)

---

## Objetivos

### Objetivo General

Desarrollar un sistema inteligente que permita anticipar retrasos, optimizar la operaci├│n log├¡stica y mejorar la toma de decisiones empresariales mediante anal├¡tica avanzada.

### Objetivos Espec├¡ficos

- Predecir la probabilidad de retraso en env├¡os
- Calcular el impacto financiero de los retrasos
- Determinar la rentabilidad de rutas log├¡sticas
- Proyectar ventas futuras por ciudad, cliente y ruta
- Identificar mercados potenciales para expansi├│n
- Generar recomendaciones ejecutivas con IA

---

## Stack Tecnol├│gico

| Capa             | Tecnolog├¡as                                                    |
|------------------|----------------------------------------------------------------|
| Lenguaje         | Python 3.11                                                    |
| Datos / ML       | Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Prophet        |
| MLOps            | MLflow                                                         |
| API              | FastAPI, Uvicorn, Pydantic                                     |
| Dashboard        | Streamlit, Plotly                                              |
| IA Generativa    | Claude (Anthropic) v├¡a Amazon Bedrock                          |
| Infraestructura  | Docker, Docker Compose, AWS                                    |
| Calidad          | Pytest, Black, Ruff, Mypy                                      |

---

## Componentes del Sistema

### 1. Predicci├│n de Retrasos
Modelo XGBoost que clasifica gu├¡as seg├║n riesgo (BAJO / MEDIO / ALTO / CRITICO).
M├®tricas clave: **Recall**, **F1 Score**, **ROC-AUC**.

### 2. Rentabilidad de Rutas
C├ílculo de costos directos (combustible, peajes, conductor, mantenimiento), costos indirectos (administraci├│n, reprocesos, devoluciones), margen absoluto, margen porcentual y ROI.

### 3. Forecast de Ventas
Predicci├│n de demanda futura con Prophet (series de tiempo) y baseline de media m├│vil.

### 4. Mercados Potenciales
Scoring compuesto por ciudad: demanda, rentabilidad, crecimiento y competencia.

### 5. IA Ejecutiva (Claude / Bedrock)
Diagn├│sticos, riesgos cr├¡ticos y recomendaciones priorizadas con impacto estimado.

---

## Estructura del Proyecto

```bash
smartdelayai360/
Ôöé
Ôö£ÔöÇÔöÇ data/
Ôöé   Ôö£ÔöÇÔöÇ raw/                       # Datos crudos (no versionados)
Ôöé   Ôö£ÔöÇÔöÇ processed/                 # Datos limpios y con features
Ôöé   ÔööÔöÇÔöÇ external/                  # Datos externos (geogr├íficos, econ├│micos)
Ôöé
Ôö£ÔöÇÔöÇ notebooks/
Ôöé   Ôö£ÔöÇÔöÇ 01_eda.ipynb               # An├ílisis exploratorio
Ôöé   Ôö£ÔöÇÔöÇ 02_feature_engineering.ipynb
Ôöé   Ôö£ÔöÇÔöÇ 03_training_mlflow.ipynb
Ôöé   ÔööÔöÇÔöÇ 04_financial_impact.ipynb
Ôöé
Ôö£ÔöÇÔöÇ src/                           # L├│gica de negocio (m├│dulos reutilizables)
Ôöé   Ôö£ÔöÇÔöÇ __init__.py
Ôöé   Ôö£ÔöÇÔöÇ config.py                  # Configuraci├│n tipada (Pydantic Settings)
Ôöé   Ôö£ÔöÇÔöÇ utils.py                   # Logging, IO de datos y modelos
Ôöé   Ôö£ÔöÇÔöÇ preprocessing.py           # Limpieza, nulos, outliers
Ôöé   Ôö£ÔöÇÔöÇ features.py                # Ingenier├¡a de variables
Ôöé   Ôö£ÔöÇÔöÇ train_model.py             # Entrenamiento XGBoost + MLflow
Ôöé   Ôö£ÔöÇÔöÇ predict.py                 # Inferencia individual y por lote
Ôöé   Ôö£ÔöÇÔöÇ financial_impact.py        # Costo financiero de retrasos
Ôöé   Ôö£ÔöÇÔöÇ route_profitability.py     # Costos, margen y ROI por ruta
Ôöé   Ôö£ÔöÇÔöÇ sales_forecast.py          # Forecast Prophet / Baseline
Ôöé   Ôö£ÔöÇÔöÇ market_scoring.py          # Score de mercados potenciales
Ôöé   ÔööÔöÇÔöÇ bedrock_client.py          # Cliente Claude v├¡a Amazon Bedrock
Ôöé
Ôö£ÔöÇÔöÇ api/                           # API REST FastAPI
Ôöé   Ôö£ÔöÇÔöÇ main.py                    # Punto de entrada
Ôöé   Ôö£ÔöÇÔöÇ routers/                   # Endpoints por dominio
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ health.py
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ predictions.py
Ôöé   Ôöé   ÔööÔöÇÔöÇ recommendations.py
Ôöé   ÔööÔöÇÔöÇ schemas/                   # Contratos Pydantic
Ôöé       ÔööÔöÇÔöÇ shipment.py
Ôöé
Ôö£ÔöÇÔöÇ app/                           # Dashboard Streamlit
Ôöé   Ôö£ÔöÇÔöÇ streamlit_app.py           # P├ígina principal con KPIs
Ôöé   ÔööÔöÇÔöÇ pages/
Ôöé       Ôö£ÔöÇÔöÇ 1_Prediccion_Retrasos.py
Ôöé       Ôö£ÔöÇÔöÇ 2_Rentabilidad_Rutas.py
Ôöé       Ôö£ÔöÇÔöÇ 3_Forecast_Ventas.py
Ôöé       Ôö£ÔöÇÔöÇ 4_Mercados_Potenciales.py
Ôöé       ÔööÔöÇÔöÇ 5_Recomendaciones_IA.py
Ôöé
Ôö£ÔöÇÔöÇ models/                        # Modelos serializados (.joblib)
Ôö£ÔöÇÔöÇ reports/                       # Reportes generados (.json, .html)
Ôö£ÔöÇÔöÇ mlruns/                        # Tracking de MLflow
Ôö£ÔöÇÔöÇ tests/                         # Pruebas pytest
Ôöé
Ôö£ÔöÇÔöÇ Dockerfile
Ôö£ÔöÇÔöÇ docker-compose.yml
Ôö£ÔöÇÔöÇ requirements.txt
Ôö£ÔöÇÔöÇ .env.example
Ôö£ÔöÇÔöÇ .gitignore
ÔööÔöÇÔöÇ README.md
```

---

## Instalaci├│n

### Opci├│n A ÔÇö Local

```bash
git clone <repo-url> smartdelayai360
cd smartdelayai360

python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/Mac

pip install -r requirements.txt
copy .env.example .env           # Windows
# cp .env.example .env           # Linux/Mac
```

### Opci├│n B ÔÇö Docker

```bash
docker-compose up --build
```

Servicios disponibles:

| Servicio   | URL                          | Descripci├│n                  |
|------------|------------------------------|------------------------------|
| API        | http://localhost:8000/docs   | OpenAPI (Swagger)            |
| Dashboard  | http://localhost:8501        | Streamlit ejecutivo          |
| MLflow     | http://localhost:5000        | Tracking de experimentos     |

---

## Uso

### Entrenar el modelo de retrasos

```python
from src.utils import cargar_dataset
from src.train_model import entrenar_modelo_retrasos

df = cargar_dataset("guias_features.parquet", capa="processed")
resultado = entrenar_modelo_retrasos(df)
print(resultado["metricas"])
```

### Predecir el riesgo de una gu├¡a v├¡a API

```bash
curl -X POST http://localhost:8000/predicciones/retraso \
  -H "Content-Type: application/json" \
  -d '{
    "distancia_km": 450.5,
    "peso_kg": 120,
    "valor_flete": 850000,
    "dias_transito_estimado": 2,
    "ciudad_origen": "Bogota",
    "ciudad_destino": "Medellin",
    "tipo_servicio": "estandar"
  }'
```

### Recomendaci├│n ejecutiva con Claude

```bash
curl -X POST http://localhost:8000/recomendaciones/ejecutiva \
  -H "Content-Type: application/json" \
  -d '{
    "tasa_retraso_actual": 0.18,
    "margen_promedio_pct": 0.12,
    "rutas_no_rentables": 7,
    "top_ciudades": ["Bogota", "Medellin", "Cali"],
    "impacto_financiero_mensual": 145000000
  }'
```

### Ejecutar pruebas

```bash
pytest -v
```

### Complemento Fase 3 (Pipeline con Prefect)

Ejecuta un flujo de orquestacion para limpieza + entrenamiento MVP + reporte:

```bash
python flows/prefect_training_flow.py
```

Nota:
- En Python 3.13 el script corre en modo local fallback si Prefect no esta instalado.
- Para usar Prefect completo, usar Python 3.11/3.12 o un entorno con wheels compatibles.

Salida esperada:

- Reporte JSON en `reports/prefect_training_run_*.json`
- Modelo MVP actualizado en `models/`

### Complemento Fase 5 (Monitoreo inicial)

Genera un reporte de monitoreo con:

- Calidad de datos (nulos, duplicados, tasa target)
- Drift por feature numerica usando PSI

```bash
python scripts/run_monitoring.py
```

Salida esperada:

- Reporte JSON en `reports/monitoring_report_*.json`

---

## Configuraci├│n (.env)

Las credenciales y rutas se gestionan desde `.env`. Copia `.env.example` y completa:

- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- `BEDROCK_MODEL_ID` (por defecto Claude 3.5 Sonnet)
- `MLFLOW_TRACKING_URI` (local o servidor remoto)

> **Importante:** nunca subir `.env` al repositorio. Est├í bloqueado en `.gitignore`.

---

## Roadmap

- [ ] Pipeline ETL programado (Airflow / AWS Step Functions)
- [ ] Despliegue en AWS ECS / Fargate
- [ ] Monitoreo de drift del modelo
- [ ] Autenticaci├│n OAuth2 / JWT en la API
- [ ] Tests de integraci├│n end-to-end

---

## Licencia

Proyecto interno SmartDelay AI 360. Todos los derechos reservados.
