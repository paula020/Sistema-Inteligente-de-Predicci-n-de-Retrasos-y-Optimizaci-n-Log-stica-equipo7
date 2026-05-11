# SmartDelay AI 360

### Plataforma Inteligente para Predicción de Retrasos, Rentabilidad Logística y Toma de Decisiones Empresariales

---

## Descripción del Proyecto

**SmartDelay AI 360** es una plataforma de inteligencia artificial y ciencia de datos diseñada para transformar la operación logística de empresas de transporte, permitiendo pasar de un modelo reactivo a uno **predictivo, rentable y estratégico**.

El sistema integra:

- Predicción de retrasos con Machine Learning (XGBoost)
- Análisis de costos directos e indirectos por ruta
- Cálculo de rentabilidad, margen y ROI
- Proyección de ventas (forecast con Prophet)
- Identificación de mercados potenciales
- Simulación de impacto financiero
- Recomendaciones ejecutivas con IA generativa (Claude vía Amazon Bedrock)

---

## Objetivos

### Objetivo General

Desarrollar un sistema inteligente que permita anticipar retrasos, optimizar la operación logística y mejorar la toma de decisiones empresariales mediante analítica avanzada.

### Objetivos Específicos

- Predecir la probabilidad de retraso en envíos
- Calcular el impacto financiero de los retrasos
- Determinar la rentabilidad de rutas logísticas
- Proyectar ventas futuras por ciudad, cliente y ruta
- Identificar mercados potenciales para expansión
- Generar recomendaciones ejecutivas con IA

---

## Stack Tecnológico

| Capa             | Tecnologías                                                    |
|------------------|----------------------------------------------------------------|
| Lenguaje         | Python 3.11                                                    |
| Datos / ML       | Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Prophet        |
| MLOps            | MLflow                                                         |
| API              | FastAPI, Uvicorn, Pydantic                                     |
| Dashboard        | Streamlit, Plotly                                              |
| IA Generativa    | Claude (Anthropic) vía Amazon Bedrock                          |
| Infraestructura  | Docker, Docker Compose, AWS                                    |
| Calidad          | Pytest, Black, Ruff, Mypy                                      |

---

## Componentes del Sistema

### 1. Predicción de Retrasos
Modelo XGBoost que clasifica guías según riesgo (BAJO / MEDIO / ALTO / CRITICO).
Métricas clave: **Recall**, **F1 Score**, **ROC-AUC**.

### 2. Rentabilidad de Rutas
Cálculo de costos directos (combustible, peajes, conductor, mantenimiento), costos indirectos (administración, reprocesos, devoluciones), margen absoluto, margen porcentual y ROI.

### 3. Forecast de Ventas
Predicción de demanda futura con Prophet (series de tiempo) y baseline de media móvil.

### 4. Mercados Potenciales
Scoring compuesto por ciudad: demanda, rentabilidad, crecimiento y competencia.

### 5. IA Ejecutiva (Claude / Bedrock)
Diagnósticos, riesgos críticos y recomendaciones priorizadas con impacto estimado.

---

## Estructura del Proyecto

```bash
smartdelayai360/
│
├── data/
│   ├── raw/                       # Datos crudos (no versionados)
│   ├── processed/                 # Datos limpios y con features
│   └── external/                  # Datos externos (geográficos, económicos)
│
├── notebooks/
│   ├── 01_eda.ipynb               # Análisis exploratorio
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training_mlflow.ipynb
│   └── 04_financial_impact.ipynb
│
├── src/                           # Lógica de negocio (módulos reutilizables)
│   ├── __init__.py
│   ├── config.py                  # Configuración tipada (Pydantic Settings)
│   ├── utils.py                   # Logging, IO de datos y modelos
│   ├── preprocessing.py           # Limpieza, nulos, outliers
│   ├── features.py                # Ingeniería de variables
│   ├── train_model.py             # Entrenamiento XGBoost + MLflow
│   ├── predict.py                 # Inferencia individual y por lote
│   ├── financial_impact.py        # Costo financiero de retrasos
│   ├── route_profitability.py     # Costos, margen y ROI por ruta
│   ├── sales_forecast.py          # Forecast Prophet / Baseline
│   ├── market_scoring.py          # Score de mercados potenciales
│   └── bedrock_client.py          # Cliente Claude vía Amazon Bedrock
│
├── api/                           # API REST FastAPI
│   ├── main.py                    # Punto de entrada
│   ├── routers/                   # Endpoints por dominio
│   │   ├── health.py
│   │   ├── predictions.py
│   │   └── recommendations.py
│   └── schemas/                   # Contratos Pydantic
│       └── shipment.py
│
├── app/                           # Dashboard Streamlit
│   ├── streamlit_app.py           # Página principal con KPIs
│   └── pages/
│       ├── 1_Prediccion_Retrasos.py
│       ├── 2_Rentabilidad_Rutas.py
│       ├── 3_Forecast_Ventas.py
│       ├── 4_Mercados_Potenciales.py
│       └── 5_Recomendaciones_IA.py
│
├── models/                        # Modelos serializados (.joblib)
├── reports/                       # Reportes generados (.json, .html)
├── mlruns/                        # Tracking de MLflow
├── tests/                         # Pruebas pytest
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Instalación

### Opción A — Local

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

### Opción B — Docker

```bash
docker-compose up --build
```

Servicios disponibles:

| Servicio   | URL                          | Descripción                  |
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

### Predecir el riesgo de una guía vía API

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

### Recomendación ejecutiva con Claude

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

---

## Configuración (.env)

Las credenciales y rutas se gestionan desde `.env`. Copia `.env.example` y completa:

- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- `BEDROCK_MODEL_ID` (por defecto Claude 3.5 Sonnet)
- `MLFLOW_TRACKING_URI` (local o servidor remoto)

> **Importante:** nunca subir `.env` al repositorio. Está bloqueado en `.gitignore`.

---

## Roadmap

- [ ] Pipeline ETL programado (Airflow / AWS Step Functions)
- [ ] Despliegue en AWS ECS / Fargate
- [ ] Monitoreo de drift del modelo
- [ ] Autenticación OAuth2 / JWT en la API
- [ ] Tests de integración end-to-end

---

## Licencia

Proyecto interno SmartDelay AI 360. Todos los derechos reservados.
