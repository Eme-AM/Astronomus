import os
import logging
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

RUTA_SILVER = "backend/data/silver/data_lake_consolidado.csv"
RUTA_GOLD_DIR = "backend/data/gold"
ARCHIVO_PREPARADO = f"{RUTA_GOLD_DIR}/dataset_preparado_ml.csv"

UMBRAL_ESI_GRIAL = 0.80
MINIMO_FILAS_VIABLE = 500

FEATURES_MODELO = [
    'pl_rade', 'pl_bmasse', 'pl_dens', 'pl_eqt', 'pl_insol',
    'pl_orbeccen', 'pl_orbper', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_age'
]

FEATURES_LOG_TRANSFORM = ['pl_orbper', 'pl_bmasse', 'pl_insol', 'pl_dens']

@dataclass
class ArtefactosML:
    X: pd.DataFrame
    y: pd.Series
    escalador: RobustScaler
    features: list[str]

def cargar_capa_plata(ruta: str = RUTA_SILVER) -> pd.DataFrame:
    if not os.path.exists(ruta):
        logging.error("No se encontró el archivo de la Capa Plata.")
        raise FileNotFoundError(f"Archivo faltante: {ruta}")
    return pd.read_csv(ruta)

def codificar_super_target(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Iniciando la construcción del Súper Target (Target Engineering)...")
    df_ml = df.copy()
    
    df_ml['target_class'] = -1
    mask_etiquetados = df_ml['phl_esi'].notna() & df_ml['phl_habitable'].notna()
    
    df_ml.loc[mask_etiquetados, 'target_class'] = 0
    df_ml.loc[mask_etiquetados & (df_ml['phl_habitable'] > 0) & (df_ml['phl_esi'] < UMBRAL_ESI_GRIAL), 'target_class'] = 1
    df_ml.loc[mask_etiquetados & (df_ml['phl_habitable'] > 0) & (df_ml['phl_esi'] >= UMBRAL_ESI_GRIAL), 'target_class'] = 2

    if len(df_ml) < MINIMO_FILAS_VIABLE:
        raise ValueError(f"Dataset resultante ({len(df_ml)} filas) por debajo del mínimo viable.")

    return df_ml

def auditar_desbalanceo(df: pd.DataFrame) -> None:
    conteo_clases = df['target_class'].value_counts().sort_index()
    porcentajes = df['target_class'].value_counts(normalize=True).sort_index() * 100
    mapa_nombres = {-1: "Huérfanos", 0: "Inhóspito", 1: "Exótico", 2: "Tierra 2.0 (Grial)"}
    
    print("\n" + "="*60)
    print(" AUDITORÍA DE CLASES (SUPER-TARGET):")
    for clase, nombre in mapa_nombres.items():
        total = conteo_clases.get(clase, 0)
        porcentaje = porcentajes.get(clase, 0.0)
        print(f"  - Clase {clase:2d} [{nombre.ljust(22)}]: {total:4d} planetas ({porcentaje:6.2f}%)")
    print("="*60 + "\n")

def _aplicar_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    df_transformado = df.copy()
    for col in FEATURES_LOG_TRANSFORM:
        if col in df_transformado.columns:
            df_transformado[col] = np.log1p(df_transformado[col].clip(lower=0))
    return df_transformado

def escalar_y_exportar(df_ml: pd.DataFrame) -> ArtefactosML:
    logging.info("Iniciando transformación matemática y escalado...")
    
    X = df_ml[FEATURES_MODELO]
    y = df_ml['target_class']
    X_log = _aplicar_log_transform(X)

    # Entrenar el escalador SOLO con datos normales (evita sesgo por Griales)
    mask_para_scaler = y != 2
    escalador = RobustScaler()
    escalador.fit(X_log[mask_para_scaler])
    
    X_scaled = pd.DataFrame(escalador.transform(X_log), columns=FEATURES_MODELO)

    if not np.isfinite(X_scaled.values).all():
        raise ValueError("El escalado introdujo valores NaN o Infinitos.")

    # Inyectar pl_name para mantener alineación segura
    X_scaled_con_id = X_scaled.copy()
    X_scaled_con_id.insert(0, 'pl_name', df_ml['pl_name'].values)
    
    os.makedirs(RUTA_GOLD_DIR, exist_ok=True)
    X_scaled_con_id.to_csv(f"{RUTA_GOLD_DIR}/X_scaled.csv", index=False)
    y.to_csv(f"{RUTA_GOLD_DIR}/y.csv", index=False)

    os.makedirs("backend/artifacts", exist_ok=True)
    joblib.dump(escalador, "backend/artifacts/robust_scaler.pkl")
    joblib.dump(FEATURES_MODELO, "backend/artifacts/feature_cols.pkl")

    logging.info("Artefactos Oro guardados de forma segura. Total: %d planetas.", len(X_scaled))
    return ArtefactosML(X_scaled, y, escalador, FEATURES_MODELO)

def ejecutar_preparacion_target():
    logging.info("Iniciando Capa Oro: Preparación del Dataset...")
    df_plata = cargar_capa_plata()
    df_preparado = codificar_super_target(df_plata)
    auditar_desbalanceo(df_preparado)
    
    os.makedirs(RUTA_GOLD_DIR, exist_ok=True)
    df_preparado.to_csv(ARCHIVO_PREPARADO, index=False)
    
    escalar_y_exportar(df_preparado)
    logging.info("¡Capa Oro construida exitosamente!")

if __name__ == "__main__":
    ejecutar_preparacion_target()