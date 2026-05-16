import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import IterativeImputer


# Silenciamos la advertencia de MICE
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Configuración del Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Permitimos importar desde el core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.core.physics import calc_stefan_boltzmann_temp, calc_stefan_boltzmann_insol, calc_densidad_planetaria

# Constantes (Evitamos Magic Numbers)
RUTA_RAW = "data/raw"
RUTA_PROCESSED = "data/processed"
ARCHIVO_SALIDA = f"{RUTA_PROCESSED}/data_lake_consolidado.csv"
TOLERANCIA_GRADOS = 0.01
MICE_ITERACIONES = 10
SEMILLA_ALEATORIA = 42

# ==========================================
# FUNCIONES DEL PIPELINE
# ==========================================

def extract_bronce_data(ruta_raw: str = RUTA_RAW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los datasets crudos desde la Capa Bronce."""
    try:
        df_nasa = pd.read_csv(f"{ruta_raw}/nasa_exoplanets.csv", low_memory=False)
        df_eu = pd.read_csv(f"{ruta_raw}/eu_exoplanets.csv", low_memory=False)
        df_phl = pd.read_csv(f"{ruta_raw}/phl_exoplanets.csv", low_memory=False)
        return df_nasa, df_eu, df_phl
    except FileNotFoundError as e:
        logging.error(f"Falta un archivo en la Capa Bronce. Detalle: {e}")
        raise

def limpiar_y_filtrar(df_nasa: pd.DataFrame, df_eu: pd.DataFrame, df_phl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplica filtrado de columnas y normalización de unidades por fuente."""
    
    # NASA (Base Maestra)
    cols_nasa = ['pl_name', 'ra', 'dec', 'pl_rade', 'pl_bmasse', 'pl_dens', 'pl_eqt', 'pl_insol', 
                 'pl_orbeccen', 'pl_orbper', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_age']
    master_df = df_nasa[cols_nasa].copy()
    master_df = master_df.dropna(subset=['ra', 'dec']).reset_index(drop=True)
    
    # EUROPA (Extraemos también parámetros orbitales y estelares)
    cols_eu = ['name', 'ra', 'dec', 'mass', 'radius', 'temp_calculated', 
               'orbital_period', 'eccentricity', 'star_teff', 'star_mass']
    df_eu_clean = df_eu[cols_eu].copy()
    df_eu_clean = df_eu_clean.dropna(subset=['ra', 'dec'])
    
    # Normalización matemática y renombramiento
    df_eu_clean['eu_mass_earth'] = df_eu_clean['mass'] * 317.8       # Júpiter a Tierra
    df_eu_clean['eu_radius_earth'] = df_eu_clean['radius'] * 11.209  # Júpiter a Tierra
    
    df_eu_clean.rename(columns={
        'temp_calculated': 'eu_temp',
        'orbital_period': 'eu_orbital_period',
        'eccentricity': 'eu_eccentricity',
        'star_teff': 'eu_star_teff',
        'star_mass': 'eu_star_mass'
    }, inplace=True)
    
    # PHL (Arecibo)
    df_phl_clean = df_phl[['P_NAME', 'S_RA', 'S_DEC', 'P_ESI', 'P_HABITABLE']].copy()
    df_phl_clean = df_phl_clean.dropna(subset=['S_RA', 'S_DEC'])
    
    return master_df, df_eu_clean, df_phl_clean

def ejecutar_entity_resolution(master_df: pd.DataFrame, df_eu_clean: pd.DataFrame, df_phl_clean: pd.DataFrame) -> pd.DataFrame:
    """Cruza los catálogos utilizando K-D Trees (Fuzzy Matching espacial)."""
    arbol_nasa = cKDTree(master_df[['ra', 'dec']].values)
    
    # Cruce Europa
    distancias_eu, indices_eu = arbol_nasa.query(df_eu_clean[['ra', 'dec']].values, distance_upper_bound=TOLERANCIA_GRADOS)
    matches_eu = distancias_eu != np.inf
    idx_validos_eu = indices_eu[matches_eu]
    
    # Transferimos todas las columnas rescatadas de Europa
    cols_a_transferir = ['eu_mass_earth', 'eu_radius_earth', 'eu_temp', 
                         'eu_orbital_period', 'eu_eccentricity', 'eu_star_teff', 'eu_star_mass']
    for col in cols_a_transferir:
        master_df.loc[idx_validos_eu, col] = df_eu_clean.loc[matches_eu, col].values
    
    # Cruce PHL
    distancias_phl, indices_phl = arbol_nasa.query(df_phl_clean[['S_RA', 'S_DEC']].values, distance_upper_bound=TOLERANCIA_GRADOS)
    matches_phl = distancias_phl != np.inf
    idx_validos_phl = indices_phl[matches_phl]
    master_df.loc[idx_validos_phl, 'phl_esi'] = df_phl_clean.loc[matches_phl, 'P_ESI'].values
    master_df.loc[idx_validos_phl, 'phl_habitable'] = df_phl_clean.loc[matches_phl, 'P_HABITABLE'].values
    
    return master_df

def imputar_datos(master_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ejecuta los 3 niveles de imputación y retorna reportes de linaje."""
    
    cols_seguimiento = [col for col in master_df.columns if 'phl' not in col and 'eu_' not in col]
    nulos_inicio = master_df[cols_seguimiento].isna().sum().sum()

    # --- NIVEL 1: Fusión (Cross-Filling) ---
    master_df['pl_bmasse'] = master_df['pl_bmasse'].fillna(master_df['eu_mass_earth'])
    master_df['pl_rade'] = master_df['pl_rade'].fillna(master_df['eu_radius_earth'])
    master_df['pl_eqt'] = master_df['pl_eqt'].fillna(master_df['eu_temp'])
    master_df['pl_orbper'] = master_df['pl_orbper'].fillna(master_df['eu_orbital_period'])
    master_df['pl_orbeccen'] = master_df['pl_orbeccen'].fillna(master_df['eu_eccentricity'])
    master_df['st_teff'] = master_df['st_teff'].fillna(master_df['eu_star_teff'])
    master_df['st_mass'] = master_df['st_mass'].fillna(master_df['eu_star_mass'])
    
    # Eliminamos las columnas auxiliares
    cols_a_borrar = ['eu_mass_earth', 'eu_radius_earth', 'eu_temp', 
                     'eu_orbital_period', 'eu_eccentricity', 'eu_star_teff', 'eu_star_mass']
    master_df = master_df.drop(columns=cols_a_borrar)
    
    nulos_fase_1 = master_df[cols_seguimiento].isna().sum().sum()
    imputados_fusion = nulos_inicio - nulos_fase_1

    # --- NIVEL 2: Física (Usando src/core/physics.py) ---
    idx_no_temp = master_df['pl_eqt'].isna() & master_df['pl_insol'].notna()
    master_df.loc[idx_no_temp, 'pl_eqt'] = calc_stefan_boltzmann_temp(master_df.loc[idx_no_temp, 'pl_insol'])

    idx_no_insol = master_df['pl_insol'].isna() & master_df['pl_eqt'].notna()
    master_df.loc[idx_no_insol, 'pl_insol'] = calc_stefan_boltzmann_insol(master_df.loc[idx_no_insol, 'pl_eqt'])

    idx_no_dens = master_df['pl_dens'].isna() & master_df['pl_bmasse'].notna() & master_df['pl_rade'].notna()
    master_df.loc[idx_no_dens, 'pl_dens'] = calc_densidad_planetaria(master_df.loc[idx_no_dens, 'pl_bmasse'], master_df.loc[idx_no_dens, 'pl_rade'])
    
    nulos_fase_2 = master_df[cols_seguimiento].isna().sum().sum()
    imputados_fisica = nulos_fase_1 - nulos_fase_2

    # --- NIVEL 3: Machine Learning (MICE) ---
    cols_para_mice = ['ra', 'dec', 'pl_rade', 'pl_bmasse', 'pl_dens', 'pl_eqt', 'pl_insol', 
                      'pl_orbeccen', 'pl_orbper', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_age']
    
    estimador = ExtraTreesRegressor(n_estimators=30, random_state=SEMILLA_ALEATORIA)
    imputador_mice = IterativeImputer(estimator=estimador, max_iter=MICE_ITERACIONES, random_state=SEMILLA_ALEATORIA)
    
    master_df[cols_para_mice] = imputador_mice.fit_transform(master_df[cols_para_mice])
    
    nulos_fase_3 = master_df[cols_seguimiento].isna().sum().sum()
    imputados_mice = nulos_fase_2 - nulos_fase_3

    reporte_linaje = {
        "Fusion": imputados_fusion,
        "Fisica": imputados_fisica,
        "MICE": imputados_mice,
        "Total_Nulos_Originales": nulos_inicio,
        "Total_Rescatados": imputados_fusion + imputados_fisica + imputados_mice
    }

    return master_df, reporte_linaje

def ejecutar_procesamiento() -> pd.DataFrame:
    """Función principal (Orquestador) que ejecuta el pipeline de la Capa Plata."""
    logging.info("Iniciando Procesamiento (Capa Plata)...")
    
    logging.info("Extrayendo datos crudos (Capa Bronce)...")
    df_nasa, df_eu, df_phl = extract_bronce_data()
    
    logging.info("Limpiando y estandarizando columnas...")
    master_df, df_eu_clean, df_phl_clean = limpiar_y_filtrar(df_nasa, df_eu, df_phl)
    
    logging.info("Ejecutando Fuzzy Matching Espacial (K-D Tree)...")
    master_df = ejecutar_entity_resolution(master_df, df_eu_clean, df_phl_clean)
    
    logging.info("Aplicando Imputación Híbrida (Fusión -> Física -> MICE)...")
    master_df, linaje = imputar_datos(master_df)
    
    # Reporte final por consola
    print("\n" + "="*50)
    print(" REPORTE DE LINAJE DE IMPUTACIÓN:")
    print(f"  - Rescatados empíricamente (Europa) : {linaje['Fusion']} datos.")
    print(f"  - Deducidos matemáticamente (Física): {linaje['Fisica']} datos.")
    print(f"  - Inferidos algorítmicamente (MICE) : {linaje['MICE']} datos.")
    print(f"  - TOTAL RESCATADO                   : {linaje['Total_Rescatados']} nulos eliminados.")
    print("="*50 + "\n")
    
    os.makedirs(RUTA_PROCESSED, exist_ok=True)
    master_df.to_csv(ARCHIVO_SALIDA, index=False)
    
    logging.info(f"Capa Plata construida con éxito. Archivo: {ARCHIVO_SALIDA}")
    return master_df

if __name__ == "__main__":
    df_final = ejecutar_procesamiento()