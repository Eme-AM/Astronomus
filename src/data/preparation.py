import os
import logging
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Configuración del Logger Profesional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# ==========================================
# CONFIGURACIÓN GLOBALES Y CONSTANTES
# ==========================================
RUTA_SILVER = "data/silver/data_lake_consolidado.csv"
RUTA_GOLD_DIR = "data/gold"
ARCHIVO_PREPARADO = f"{RUTA_GOLD_DIR}/dataset_preparado_ml.csv"

# Parámetros Científicos de Control
UMBRAL_ESI_GRIAL = 0.80
MINIMO_FILAS_VIABLE = 500
MIN_MUESTRAS_POR_CLASE = 10  # Umbral de seguridad estadística para el Split

# Lista explícita y versionada de Features (Evita selección por exclusión )
# En caso de agregar columnas al datalake, la lista tiene que ser actualizada
FEATURES_MODELO = [
    'pl_rade', 'pl_bmasse', 'pl_dens', 'pl_eqt', 'pl_insol',
    'pl_orbeccen', 'pl_orbper', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_age'
]

@dataclass
class ArtefactosML:
    """Contenedor fuertemente tipado para los artefactos de la Capa Oro (Critico #3)."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    escalador: RobustScaler
    features: list[str]

# ==========================================
# VALIDACIONES DEFENSIVAS
# ==========================================

def _validar_distribucion_target(y: pd.Series) -> None:
    """Valida que las clases tengan suficientes muestras para un split seguro"""
    conteo = y.value_counts()
    clases_criticas = conteo[conteo < MIN_MUESTRAS_POR_CLASE]
    
    if not clases_criticas.empty:
        logging.warning("RESTRICCIÓN DE SPLIT: Existen clases por debajo del mínimo robusto (%d muestras):\n%s", 
                        MIN_MUESTRAS_POR_CLASE, clases_criticas.to_string())
        logging.warning("Se procederá con el split estratificado debido al tamaño crítico del catálogo, "
                        "pero es obligatorio usar Class Weights en el modelo predictivo.")

def _validar_no_nan_inf(df: pd.DataFrame, nombre_split: str) -> None:
    """Asegura que el escalado matemático no haya introducido valores inválidos para PyTorch"""
    if not np.isfinite(df.values).all():
        cols_problematicas = df.columns[~np.isfinite(df.values).all(axis=0)].tolist()
        raise ValueError(
            f"Fallo de Escalado: {nombre_split} contiene valores NaN o Infinitos en: {cols_problematicas}. "
            f"Verificá si alguna columna tiene varianza cero tras la imputación."
        )

# ==========================================
# FUNCIONES DEL PIPELINE
# ==========================================

def cargar_capa_plata(ruta: str = RUTA_SILVER) -> pd.DataFrame:
    """Carga el dataset consolidado desde la subcarpeta Silver"""
    if not os.path.exists(ruta):
        logging.error("No se encontró el archivo de la Capa Plata en %s. Ejecutá processing.py primero.", ruta)
        raise FileNotFoundError(f"Archivo faltante: {ruta}")
    return pd.read_csv(ruta)

def codificar_super_target(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica Target Engineering combinando la física posicional y estructural."""
    logging.info("Iniciando la construcción del Súper Target (Target Engineering)...")
    
    # Sanitización de etiquetas vacías de Arecibo
    df_ml = df.dropna(subset=['phl_esi', 'phl_habitable']).copy()
    filas_eliminadas = len(df) - len(df_ml)
    cobertura_pct = (len(df_ml) / len(df)) * 100

    # Reporte completo de cobertura (Mejora #4 - Lazy evaluation en logs #5)
    logging.info("Target Engineering: %d/%d filas con etiquetas PHL (%.1f%% cobertura). %d filas descartadas por NaN.",
                 len(df_ml), len(df), cobertura_pct, filas_eliminadas)

    if len(df_ml) < MINIMO_FILAS_VIABLE:
        raise ValueError(f"Dataset resultante ({len(df_ml)} filas) por debajo del mínimo viable. Revisá el Entity Resolution.")

    # Inyección de las reglas lógicas vectorizadas del Súper Target
    df_ml['target_class'] = 0
    
    condicion_exotico = (df_ml['phl_habitable'] > 0) & (df_ml['phl_esi'] < UMBRAL_ESI_GRIAL)
    df_ml.loc[condicion_exotico, 'target_class'] = 1

    # Clase 2: El Grial (Tierra 2.0)
    condicion_grial = (df_ml['phl_habitable'] > 0) & (df_ml['phl_esi'] >= UMBRAL_ESI_GRIAL)
    df_ml.loc[condicion_grial, 'target_class'] = 2

    return df_ml

def auditar_desbalanceo(df: pd.DataFrame) -> None:
    """Audita la distribución final de las clases en la consola."""
    conteo_clases = df['target_class'].value_counts().sort_index()
    porcentajes = df['target_class'].value_counts(normalize=True).sort_index() * 100
    mapa_nombres = {0: "Mundo Inhóspito", 1: "Mundo Exótico", 2: "Tierra 2.0 (Grial)"}
    
    print("\n" + "="*60)
    print(" AUDITORÍA DE DESBALANCEO DE CLASES (TARGET CONFIGURATION):")
    for clase, nombre in mapa_nombres.items():
        total = conteo_clases.get(clase, 0)
        porcentaje = porcentajes.get(clase, 0.0)
        print(f"  - Clase {clase} [{nombre.ljust(20)}]: total = {total:4d} planetas ({porcentaje:6.2f}%)")
    print("="*60 + "\n")

def escalar_y_dividir_datos(df_ml: pd.DataFrame) -> ArtefactosML:
    """Realiza la partición estratificada de datos, aplica RobustScaler y exporta artefactos."""
    logging.info("Iniciando partición de datos (Train/Test Split) y escalado robusto...")
    
    # Validación estructural de características
    faltantes = [f for f in FEATURES_MODELO if f not in df_ml.columns]
    if faltantes:
        raise ValueError(f"Features requeridas no encontradas en el dataset de entrada: {faltantes}")

    X = df_ml[FEATURES_MODELO]
    y = df_ml['target_class']

    # Verificación de salud de las clases antes del split
    _validar_distribucion_target(y)

    # Split Estratificado (Preserva la proporción 98/1/0.15 de forma exacta)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Escalado Robusto insensible a Outliers (Gigantes gaseosos extremos)
    escalador = RobustScaler()
    X_train_scaled = pd.DataFrame(escalador.fit_transform(X_train), columns=FEATURES_MODELO)
    X_test_scaled = pd.DataFrame(escalador.transform(X_test), columns=FEATURES_MODELO)

    # Validación de sanidad post-escalado (Evita NaN/Inf en tensores de PyTorch)
    _validar_no_nan_inf(X_train_scaled, "X_train_scaled")
    _validar_no_nan_inf(X_test_scaled, "X_test_scaled")

    # Persistencia física en la Capa Oro (data/gold/)
    os.makedirs(RUTA_GOLD_DIR, exist_ok=True)
    X_train_scaled.to_csv(f"{RUTA_GOLD_DIR}/X_train.csv", index=False)
    X_test_scaled.to_csv(f"{RUTA_GOLD_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{RUTA_GOLD_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{RUTA_GOLD_DIR}/y_test.csv", index=False)

    # Persistencia de artefactos de serialización para producción (FastAPI / Frontend)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(escalador, "artifacts/robust_scaler.pkl")
    joblib.dump(FEATURES_MODELO, "artifacts/feature_cols.pkl")

    logging.info("Artefactos Oro guardados. Entrenamiento: %d filas | Testeo: %d filas | Características: %d",
                 len(X_train_scaled), len(X_test_scaled), len(FEATURES_MODELO))

    # Retornamos el contenedor de datos listo para uso en memoria (Critico #3)
    return ArtefactosML(X_train_scaled, X_test_scaled, y_train, y_test, escalador, FEATURES_MODELO)

def ejecutar_preparacion_target() -> ArtefactosML:
    """Orquestador principal de la Capa Oro."""
    logging.info("Iniciando Módulo 3: Preparación del Dataset de Machine Learning...")
    
    df_plata = cargar_capa_plata()
    df_preparado = codificar_super_target(df_plata)
    
    auditar_desbalanceo(df_preparado)
    os.makedirs(RUTA_GOLD_DIR, exist_ok=True)
    df_preparado.to_csv(ARCHIVO_PREPARADO, index=False)
    
    artefactos = escalar_y_dividir_datos(df_preparado)
    logging.info("¡Capa Oro (Estructura de Tensores) construida y verificada con éxito!")
    
    return artefactos

if __name__ == "__main__":
    ejecutar_preparacion_target()