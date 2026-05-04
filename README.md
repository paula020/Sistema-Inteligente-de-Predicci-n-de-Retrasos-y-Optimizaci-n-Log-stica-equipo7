# 🚚 SmartDelay AI 360
### Plataforma Inteligente para Predicción de Retrasos, Rentabilidad Logística y Toma de Decisiones Empresariales

---

## 📌 Descripción del Proyecto

**SmartDelay AI 360** es una plataforma de inteligencia artificial y ciencia de datos diseñada para transformar la operación logística de empresas de transporte, permitiendo pasar de un modelo reactivo a uno **predictivo, rentable y estratégico**.

El sistema integra:

- Predicción de retrasos con Machine Learning
- Análisis de costos y rentabilidad por ruta
- Proyección de ventas (forecast)
- Identificación de mercados potenciales
- Simulación de impacto financiero
- Recomendaciones ejecutivas con IA generativa

---

## 🎯 Objetivos

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

## 🧠 Componentes del Sistema

### 🚚 1. Predicción de Retrasos
Modelo de Machine Learning que clasifica guías según riesgo de retraso.

**Métricas clave:**
- Recall
- F1 Score
- ROC-AUC

---

### 💰 2. Rentabilidad de Rutas
Cálculo de:

- Costos directos (combustible, peajes, conductor)
- Costos indirectos (administración, reprocesos, devoluciones)
- Margen
- ROI

---

### 📈 3. Forecast de Ventas
Predicción de demanda futura usando:

- Series de tiempo
- Modelos como Prophet, ARIMA o XGBoost

---

### 🌎 4. Mercados Potenciales
Scoring de ciudades basado en:

- Demanda
- Rentabilidad
- Crecimiento
- Competencia

---

### 🤖 5. IA Ejecutiva (Claude / LLM)
Generación de:

- Explicaciones del modelo
- Recomendaciones de negocio
- Análisis gerencial

---

## 🗂️ Estructura del Proyecto

```bash
smartdelay-ai-360/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training_mlflow.ipynb
│   ├── 04_financial_impact.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train_model.py
│   ├── predict.py
│   ├── financial_impact.py
│   ├── route_profitability.py
│   ├── sales_forecast.py
│   ├── market_scoring.py
│
├── app/
│   └── streamlit_app.py
│
├── api/
│   └── main.py
│
├── models/
├── reports/
├── mlruns/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md