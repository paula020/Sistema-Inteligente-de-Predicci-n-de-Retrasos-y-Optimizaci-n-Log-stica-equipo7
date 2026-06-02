"""
Módulo de ingeniería de variables (Feature Engineering) para SmartDelay AI 360.

Especializado en el dataset DataCo Smart Supply Chain para predecir
`late_delivery_risk`.

Buenas prácticas aplicadas
--------------------------
- **Anti data-leakage**: se distinguen explícitamente las features
  *ex-ante* (conocidas antes del envío) de las *ex-post* (solo útiles
  para EDA o como variable derivada del target).
- **Target encoding seguro**: el cálculo de `city_risk_score`,
  `route_risk_score` y `customer_delay_rate` se entrena solo sobre el
  conjunto de entrenamiento, con suavizado bayesiano (`m-estimator`)
  para grupos con pocas observaciones y un *fallback* a la media
  global para valores nunca vistos.
- **Split temporal**: la división train/test respeta la cronología,
  evitando que el modelo "vea el futuro".
- **Codificación categórica** parametrizable: one-hot o frequency.
- **Pipeline orquestador** (`construir_features_dataco`) que aplica
  todos los pasos de forma reproducible.

Columnas esperadas (post-`limpiar_dataco`)
------------------------------------------
order_date_dateorders, shipping_date_dateorders,
days_for_shipping_real, days_for_shipment_scheduled,
sales, order_profit_per_order, order_item_profit_ratio,
customer_id, order_city, order_region, market, shipping_mode,
customer_segment, category_name, department_name, late_delivery_risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================================
# Constantes — distinción ex-ante / ex-post (clave para evitar leakage)
# ============================================================================

# Variable objetivo del modelo
TARGET = "late_delivery_risk"

# Features POST-envío que NO deben entrar al modelo predictivo:
# se conocen únicamente DESPUÉS de que el envío se realiza.
COLUMNAS_LEAKAGE = (
    "days_for_shipping_real",       # tiempo real de envío (post)
    "delivery_status",              # estado final de entrega (post)
    "delay_days",                   # derivada de la real vs scheduled (post)
    "is_late",                      # equivalente al target (post)
    "shipping_efficiency",          # usa días reales (post)
    "shipping_date_dateorders",     # fecha real del envío (post en strict mode)
    "order_status",                 # estado final de la orden
    TARGET,                         # la propia variable objetivo
)

# Identificadores con alta cardinalidad — se EXCLUYEN del modelo
# (no aportan señal generalizable y pueden causar overfitting).
COLUMNAS_ID = (
    "order_id", "order_item_id", "order_customer_id",
    "customer_id", "product_card_id", "order_item_cardprod_id",
    "product_category_id", "category_id", "department_id",
)

# Columnas de texto/URL/coordenadas que no usaremos en este módulo base
COLUMNAS_DESCARTAR = (
    "customer_city", "customer_country", "customer_state",
    "product_name", "product_image", "product_description",
    "latitude", "longitude", "type",
)


# ============================================================================
# 1. VARIABLES TEMPORALES (ex-ante)
# ============================================================================
def variables_temporales(
    df: pd.DataFrame,
    columna_fecha: str = "order_date_dateorders",
) -> pd.DataFrame:
    """
    Genera variables derivadas de la fecha de la orden.

    Variables creadas:
        - order_year, order_month, order_day, order_dayofweek
        - order_quarter, order_weekofyear
        - order_is_weekend (1/0)
        - order_hour (si la fecha tiene componente horaria)

    Todas son **ex-ante**: conocidas al momento de hacer la orden.
    """
    df = df.copy()
    if columna_fecha not in df.columns:
        raise KeyError(f"No existe la columna de fecha '{columna_fecha}'")

    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors="coerce")
    fecha = df[columna_fecha]

    df["order_year"] = fecha.dt.year
    df["order_month"] = fecha.dt.month
    df["order_day"] = fecha.dt.day
    df["order_dayofweek"] = fecha.dt.dayofweek
    df["order_quarter"] = fecha.dt.quarter
    df["order_weekofyear"] = fecha.dt.isocalendar().week.astype("int64")
    df["order_is_weekend"] = (fecha.dt.dayofweek >= 5).astype(int)
    df["order_hour"] = fecha.dt.hour.fillna(0).astype(int)

    logger.info(f"Variables temporales generadas a partir de '{columna_fecha}'")
    return df


# ============================================================================
# 2. VARIABLES DE RETRASO (ex-post — solo EDA o target alternativo)
# ============================================================================
def variables_retraso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las variables ligadas al retraso real.

    ⚠️  EX-POST: no deben usarse como features predictivas.
    Sirven para:
        - Análisis exploratorio (visualizar retrasos por segmento)
        - Construir un target alternativo (`is_late`) o continuo
          (`delay_days`) si se quiere regresión en vez de clasificación.
    """
    df = df.copy()
    if not {"days_for_shipping_real", "days_for_shipment_scheduled"}.issubset(df.columns):
        logger.warning("Faltan columnas de días real/programado, se omite variables_retraso")
        return df

    df["delay_days"] = (
        df["days_for_shipping_real"] - df["days_for_shipment_scheduled"]
    ).astype(int)
    df["is_late"] = (df["delay_days"] > 0).astype(int)
    logger.info(
        f"delay_days e is_late generadas | tasa is_late = {df['is_late'].mean():.2%}"
    )
    return df


# ============================================================================
# 3. VARIABLES ECONÓMICAS
# ============================================================================
def variables_economicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables económicas que pueden alimentar el modelo.

    - `profit_margin`: margen sobre ventas (ex-ante: usa
      `order_item_profit_ratio` o `order_profit_per_order/sales`).
    - `shipping_efficiency`: ratio scheduled/real (⚠️ ex-post).
    - `discount_pct`: tasa de descuento (ex-ante).
    """
    df = df.copy()

    # profit_margin (ex-ante si se usa el ratio nativo de DataCo)
    if "order_item_profit_ratio" in df.columns:
        df["profit_margin"] = df["order_item_profit_ratio"]
    elif {"order_profit_per_order", "sales"}.issubset(df.columns):
        df["profit_margin"] = np.where(
            df["sales"] > 0, df["order_profit_per_order"] / df["sales"], 0.0
        )
    else:
        logger.warning("No se pudo calcular profit_margin (faltan columnas)")

    # shipping_efficiency (ex-post — solo EDA)
    if {"days_for_shipment_scheduled", "days_for_shipping_real"}.issubset(df.columns):
        df["shipping_efficiency"] = np.where(
            df["days_for_shipping_real"] > 0,
            df["days_for_shipment_scheduled"] / df["days_for_shipping_real"],
            np.nan,
        )

    # discount_pct (ex-ante)
    if "order_item_discount_rate" in df.columns:
        df["discount_pct"] = df["order_item_discount_rate"]

    logger.info("Variables económicas generadas")
    return df


# ============================================================================
# 4. TARGET ENCODING SEGURO (anti data-leakage)
# ============================================================================
@dataclass
class TargetEncoderSuavizado:
    """
    Target encoder con **suavizado bayesiano** (m-estimator).

    score(grupo) = (n_grupo * media_grupo + m * media_global) / (n_grupo + m)

    - `m` actúa como pseudo-cuenta: grupos con `n_grupo << m` se acercan
      a la media global (evita sobreajuste a clases raras).
    - Se entrena (`fit`) SOLO sobre el conjunto de train y se aplica
      (`transform`) al train y al test.
    - Para categorías no vistas en train, retorna `media_global_`.

    Esto evita el data leakage típico del target encoding mal hecho
    (calcularlo sobre todo el dataset, incluyendo el test).
    """

    m: int = 30
    columnas: list[str] = field(default_factory=list)
    mapeos_: dict[str, pd.Series] = field(default_factory=dict, init=False)
    media_global_: float = field(default=0.5, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoderSuavizado":
        """Aprende el mapeo grupo -> score sobre el train set únicamente."""
        self.media_global_ = float(y.mean())
        self.mapeos_ = {}

        for col in self.columnas:
            if col not in X.columns:
                logger.warning(f"TargetEncoder: columna '{col}' no encontrada, se omite")
                continue
            grupo = pd.DataFrame({col: X[col].values, "_y": y.values})
            agg = grupo.groupby(col)["_y"].agg(["count", "mean"])
            score = (
                agg["count"] * agg["mean"] + self.m * self.media_global_
            ) / (agg["count"] + self.m)
            self.mapeos_[col] = score
            logger.debug(
                f"TargetEncoder fit '{col}': {len(score)} grupos, "
                f"media_global={self.media_global_:.4f}"
            )
        return self

    def transform(self, X: pd.DataFrame, sufijo: str = "_risk_score") -> pd.DataFrame:
        """Aplica los mapeos aprendidos. Devuelve un DataFrame con nuevas columnas."""
        X = X.copy()
        for col, mapping in self.mapeos_.items():
            X[f"{col}{sufijo}"] = (
                X[col].map(mapping).fillna(self.media_global_)
            )
        return X

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, sufijo: str = "_risk_score"
    ) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X, sufijo=sufijo)


def crear_risk_scores(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None = None,
    target: str = TARGET,
    columnas_grupo: tuple[str, ...] | None = None,
    m: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame | None, TargetEncoderSuavizado]:
    """
    Genera scores de riesgo por ciudad, ruta y cliente sin data leakage.

    El encoder se entrena en train y se aplica idénticamente a train y test.

    Crea las columnas (sufijo `_risk_score`):
        - order_city_risk_score        → city_risk_score
        - route_risk_score             (combinación origen-destino)
        - customer_id_risk_score       → customer_delay_rate

    Parameters
    ----------
    df_train, df_test : pd.DataFrame
        Datasets ya particionados temporalmente.
    target : str
        Nombre del target binario (`late_delivery_risk` por defecto).
    columnas_grupo : tuple[str]
        Columnas categóricas a usar como agrupadores.
    m : int
        Suavizado bayesiano. Valores mayores → más regularización.

    Returns
    -------
    (df_train_out, df_test_out, encoder)
    """
    df_train = df_train.copy()
    if df_test is not None:
        df_test = df_test.copy()

    # Construir variable `route` si no existe
    for d in (df_train, df_test):
        if d is None:
            continue
        if "route" not in d.columns and {"market", "order_region"}.issubset(d.columns):
            d["route"] = d["market"].astype(str) + "__" + d["order_region"].astype(str)

    if columnas_grupo is None:
        columnas_grupo = tuple(
            c for c in ("order_city", "route", "customer_id") if c in df_train.columns
        )

    encoder = TargetEncoderSuavizado(m=m, columnas=list(columnas_grupo))
    encoder.fit(df_train, df_train[target])

    df_train_out = encoder.transform(df_train)
    df_test_out = encoder.transform(df_test) if df_test is not None else None

    # Renombre amigable: <col>_risk_score -> nombres del briefing
    renombres = {
        "order_city_risk_score": "city_risk_score",
        "route_risk_score": "route_risk_score",  # ya tiene buen nombre
        "customer_id_risk_score": "customer_delay_rate",
    }
    for df_ in (df_train_out, df_test_out):
        if df_ is None:
            continue
        df_.rename(columns=renombres, inplace=True)

    logger.success(
        f"Risk scores generados sobre {len(columnas_grupo)} dimensiones | "
        f"m={m} | media_global={encoder.media_global_:.4f}"
    )
    return df_train_out, df_test_out, encoder


# ============================================================================
# 5. CODIFICACIÓN DE CATEGÓRICAS
# ============================================================================
def codificar_categoricas(
    df: pd.DataFrame,
    columnas: list[str],
    metodo: str = "onehot",
    max_categorias: int = 30,
) -> pd.DataFrame:
    """
    Codifica variables categóricas a numéricas.

    Métodos:
        - 'onehot': pd.get_dummies con `drop_first=True` (evita colinealidad)
        - 'frequency': reemplaza cada categoría por su frecuencia relativa
        - 'label': codificación ordinal entera

    `max_categorias` agrupa en 'OTRO' las categorías poco frecuentes
    para evitar explosión dimensional en columnas con alta cardinalidad.
    """
    df = df.copy()
    columnas = [c for c in columnas if c in df.columns]

    for col in columnas:
        if df[col].nunique() > max_categorias:
            top = df[col].value_counts().nlargest(max_categorias - 1).index
            df[col] = df[col].where(df[col].isin(top), other="OTRO")
            logger.info(f"'{col}' agrupada: > {max_categorias} categorías → 'OTRO'")

    if metodo == "onehot":
        df = pd.get_dummies(df, columns=columnas, drop_first=True, dtype=int)
    elif metodo == "frequency":
        for col in columnas:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq)
        df = df.drop(columns=columnas)
    elif metodo == "label":
        for col in columnas:
            df[col] = df[col].astype("category").cat.codes
    else:
        raise ValueError(f"Método de codificación no soportado: {metodo}")

    logger.info(f"Codificación '{metodo}' aplicada a {len(columnas)} columnas")
    return df


# ============================================================================
# 6. SPLIT TEMPORAL (para entrenamientos honestos)
# ============================================================================
def split_temporal(
    df: pd.DataFrame,
    columna_fecha: str = "order_date_dateorders",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset por orden cronológico (NO aleatorio).

    Garantiza que el test set contiene SOLO órdenes posteriores al
    train, replicando el escenario real de producción.
    """
    if columna_fecha not in df.columns:
        raise KeyError(f"No existe la columna '{columna_fecha}' para split temporal")

    df_ord = df.sort_values(columna_fecha).reset_index(drop=True)
    corte = int(len(df_ord) * (1 - test_size))
    train, test = df_ord.iloc[:corte].copy(), df_ord.iloc[corte:].copy()

    logger.info(
        f"Split temporal | train: {train.shape} "
        f"({train[columna_fecha].min().date()} → {train[columna_fecha].max().date()}) | "
        f"test: {test.shape} "
        f"({test[columna_fecha].min().date()} → {test[columna_fecha].max().date()})"
    )
    return train, test


# ============================================================================
# 7. SELECCIÓN DE FEATURES PARA ML (anti-leakage)
# ============================================================================
def eliminar_features_leakage(
    df: pd.DataFrame, target: str = TARGET, columnas_extra: list[str] | None = None
) -> pd.DataFrame:
    """
    Elimina columnas que conducen a data leakage o que son IDs sin valor.

    Conserva el target si está en el dataframe (se devuelve, pero
    se aísla más adelante con `seleccionar_X_y`).
    """
    a_eliminar = [
        c for c in (*COLUMNAS_LEAKAGE, *COLUMNAS_ID, *COLUMNAS_DESCARTAR)
        if c in df.columns and c != target
    ]
    if columnas_extra:
        a_eliminar.extend(c for c in columnas_extra if c in df.columns)

    if a_eliminar:
        logger.warning(f"Eliminando {len(a_eliminar)} columnas (leakage/ID/descarte)")
        logger.debug(f"  -> {a_eliminar}")
        df = df.drop(columns=a_eliminar)
    return df


def seleccionar_X_y(
    df: pd.DataFrame, target: str = TARGET
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) del target (y) y elimina columnas no numéricas residuales."""
    if target not in df.columns:
        raise KeyError(f"El target '{target}' no está en el dataframe")

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Conservar solo numéricas y booleanas (lo demás debe haberse codificado)
    columnas_no_numericas = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    if columnas_no_numericas:
        logger.warning(
            f"Eliminando {len(columnas_no_numericas)} columnas no numéricas residuales: "
            f"{columnas_no_numericas[:8]}{'...' if len(columnas_no_numericas) > 8 else ''}"
        )
        X = X.drop(columns=columnas_no_numericas)

    # Reemplazar inf/-inf por NaN y luego imputar con 0 (post target-encoding seguro)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y


# ============================================================================
# 8. PIPELINE ORQUESTADOR
# ============================================================================
def construir_features_dataco(
    df: pd.DataFrame,
    columnas_categoricas: list[str] | None = None,
    metodo_categoricas: str = "onehot",
    test_size: float = 0.2,
    m_suavizado: int = 30,
) -> dict:
    """
    Pipeline orquestador: limpieza → variables temporales/económicas/retraso
    → split temporal → risk scores → encoding → selección de features.

    Returns
    -------
    dict con claves:
        - 'X_train', 'y_train', 'X_test', 'y_test'   (listos para ML)
        - 'df_train', 'df_test'                       (con todas las columnas)
        - 'encoder'                                   (TargetEncoderSuavizado)
        - 'features_finales' (list[str])
    """
    logger.info("=" * 60)
    logger.info("INICIO PIPELINE DE FEATURE ENGINEERING — DATACO")
    logger.info("=" * 60)

    if columnas_categoricas is None:
        columnas_categoricas = [
            c
            for c in (
                "shipping_mode",
                "customer_segment",
                "market",
                "order_region",
                "category_name",
                "department_name",
            )
            if c in df.columns
        ]

    # 1. Variables temporales
    df = variables_temporales(df, "order_date_dateorders")

    # 2. Variables de retraso (ex-post) — generadas para EDA pero se eliminarán
    df = variables_retraso(df)

    # 3. Variables económicas
    df = variables_economicas(df)

    # 4. Split temporal ANTES del target encoding (clave para no leak)
    df_train, df_test = split_temporal(df, "order_date_dateorders", test_size)

    # 5. Risk scores (target encoding) entrenado solo en train
    df_train, df_test, encoder = crear_risk_scores(
        df_train, df_test, target=TARGET, m=m_suavizado
    )

    # 6. Codificación de categóricas (usando ambos sets, las dummies se alinean luego)
    df_train = codificar_categoricas(df_train, columnas_categoricas, metodo_categoricas)
    df_test = codificar_categoricas(df_test, columnas_categoricas, metodo_categoricas)

    # 7. Eliminar columnas con leakage
    df_train = eliminar_features_leakage(df_train)
    df_test = eliminar_features_leakage(df_test)

    # 8. Alinear columnas (si en test faltan dummies que sí están en train, rellenar con 0)
    df_train, df_test = _alinear_columnas(df_train, df_test, target=TARGET)

    # 9. Separar X / y
    X_train, y_train = seleccionar_X_y(df_train)
    X_test, y_test = seleccionar_X_y(df_test)

    logger.success(
        f"Pipeline completado | X_train: {X_train.shape} | X_test: {X_test.shape} | "
        f"features finales: {X_train.shape[1]}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "df_train": df_train,
        "df_test": df_test,
        "encoder": encoder,
        "features_finales": X_train.columns.tolist(),
    }


def _alinear_columnas(
    df_train: pd.DataFrame, df_test: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asegura que train y test tengan exactamente las mismas columnas (rellena con 0)."""
    cols_train = set(df_train.columns)
    cols_test = set(df_test.columns)

    faltan_en_test = cols_train - cols_test
    faltan_en_train = cols_test - cols_train

    for col in faltan_en_test:
        df_test[col] = 0
    for col in faltan_en_train:
        df_train[col] = 0

    cols_ordenadas = sorted(df_train.columns)
    df_train = df_train[cols_ordenadas]
    df_test = df_test[cols_ordenadas]

    if faltan_en_test or faltan_en_train:
        logger.info(
            f"Alineación de columnas | añadidas en test: {len(faltan_en_test)} | "
            f"añadidas en train: {len(faltan_en_train)}"
        )
    return df_train, df_test


# ============================================================================
# =========  FEATURE ENGINEERING MVP — versión simplificada y robusta  =======
# ============================================================================
# Esta sección expone una API minimalista para el MVP de Streamlit.
#
# A diferencia del pipeline avanzado de arriba (que usa target encoding y
# split temporal), aquí se construye un Pipeline sklearn estándar con
# `ColumnTransformer` que encapsula TODO el preprocesamiento dentro del
# propio modelo. Esto garantiza que la inferencia desde Streamlit sea
# idéntica al entrenamiento sin necesidad de transformar las features a mano.
#
# Variables EX-ANTE únicamente — nada que el operador no conozca al despachar.
# ============================================================================

# Columnas numéricas del MVP
FEATURES_MVP_NUMERICAS: list[str] = [
    "sales",
    "order_item_quantity",
    "order_item_discount",
    "order_item_product_price",
    "days_for_shipment_scheduled",
    "order_month",
    "order_dayofweek",
]

# Columnas categóricas del MVP
FEATURES_MVP_CATEGORICAS: list[str] = [
    "shipping_mode",
    "customer_segment",
    "market",
    "order_region",
    "order_country",
    "category_name",
]

# Listado completo (orden = el que verá el pipeline)
FEATURES_MVP_TODAS: list[str] = FEATURES_MVP_CATEGORICAS + FEATURES_MVP_NUMERICAS

# Columnas PROHIBIDAS — nunca deben aparecer en el set de entrenamiento del MVP
COLUMNAS_PROHIBIDAS_MVP: tuple[str, ...] = (
    "days_for_shipping_real",        # se conoce solo después del envío
    "delivery_status",               # estado final
    "delay_days",                    # derivada del retraso real
    "is_late",                       # equivalente al target
    "shipping_efficiency",           # usa días reales
    "shipping_date_dateorders",      # fecha real del envío
    "order_status",                  # estado final de la orden
)


def preparar_features_mvp(
    df: pd.DataFrame,
    columna_fecha: str = "order_date_dateorders",
) -> pd.DataFrame:
    """
    Extrae únicamente las 13 features ex-ante del MVP a partir del
    dataset DataCo limpio.

    - Si `order_month` y `order_dayofweek` no existen, se derivan
      automáticamente de `columna_fecha`.
    - Aborta con error claro si falta cualquier columna requerida.
    - Lanza una advertencia si detecta columnas prohibidas.
    """
    df = df.copy()

    # Derivar variables temporales si no existen
    if "order_month" not in df.columns or "order_dayofweek" not in df.columns:
        if columna_fecha not in df.columns:
            raise KeyError(
                f"Falta '{columna_fecha}' para derivar order_month/dayofweek. "
                f"Proporciona la fecha o las columnas derivadas explícitamente."
            )
        fecha = pd.to_datetime(df[columna_fecha], errors="coerce")
        df["order_month"] = fecha.dt.month
        df["order_dayofweek"] = fecha.dt.dayofweek

    # Verificar columnas requeridas
    faltantes = [c for c in FEATURES_MVP_TODAS if c not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas del MVP: {faltantes}")

    # Alertar si hay columnas prohibidas presentes (no se incluyen, pero avisamos)
    prohibidas_presentes = [c for c in COLUMNAS_PROHIBIDAS_MVP if c in df.columns]
    if prohibidas_presentes:
        logger.warning(
            f"Columnas prohibidas detectadas en el dataset (NO se usarán para "
            f"entrenar): {prohibidas_presentes}"
        )

    logger.info(
        f"Features MVP preparadas: {len(FEATURES_MVP_CATEGORICAS)} categóricas + "
        f"{len(FEATURES_MVP_NUMERICAS)} numéricas = {len(FEATURES_MVP_TODAS)} columnas"
    )
    return df[FEATURES_MVP_TODAS].copy()


def construir_preprocesador_mvp():
    """
    Construye el `ColumnTransformer` del MVP, listo para integrarse en un
    Pipeline sklearn:

        - Numéricas:   SimpleImputer(median) + StandardScaler
        - Categóricas: SimpleImputer(constant='desconocido') + OneHotEncoder

    El OneHotEncoder usa `handle_unknown='ignore'`, lo que evita errores
    si en producción aparece una categoría que no se vio durante el
    entrenamiento (Streamlit puede mandar un país poco común, por ejemplo).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pipeline_num = SKPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pipeline_cat = SKPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="desconocido")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", pipeline_num, FEATURES_MVP_NUMERICAS),
            ("cat", pipeline_cat, FEATURES_MVP_CATEGORICAS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
