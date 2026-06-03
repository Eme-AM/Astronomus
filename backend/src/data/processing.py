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

# ==========================================
# CONFIGURACIÓN GLOBALES Y CONSTANTES
# ==========================================
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Permitimos importar desde el core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.core.physics import calc_stefan_boltzmann_temp, calc_stefan_boltzmann_insol, calc_densidad_planetaria

# Constantes de Rutas y MICE
RUTA_RAW = "data/raw"
RUTA_SILVER_DIR = "data/silver"
ARCHIVO_SALIDA = f"{RUTA_SILVER_DIR}/data_lake_consolidado.csv"

MICE_ITERACIONES = 10
SEMILLA_ALEATORIA = 42

# Geometría Celeste: 0.01 grados a distancia de cuerda (Chord Distance)
TOLERANCIA_GRADOS = 0.01
TOLERANCIA_CHORD = 2 * np.sin(np.radians(TOLERANCIA_GRADOS) / 2)

# Variables exclusivamente físicas para el entrenamiento algorítmico (excluye RA/Dec)
COLS_FISICAS_MICE = [
    'pl_rade', 'pl_bmasse', 'pl_dens', 'pl_eqt', 'pl_insol',
    'pl_orbeccen', 'pl_orbper', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_age'
]

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

# Función de conversión a coordenadas cartesianas para KD-Tree
def _to_cartesian(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """
    Convierte coordenadas ecuatoriales (RA, Dec) a vectores unitarios 3D.
    Permite usar cKDTree con distancia de cuerda como aproximación 
    exacta de la distancia angular sobre la esfera celeste.
    """
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])

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
        logging.error(f"Falta un archivo en Bronce. Ejecutá ingestion.py primero. Detalle: {e}")
        raise

def limpiar_y_filtrar(df_nasa: pd.DataFrame, df_eu: pd.DataFrame, df_phl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplica filtrado de columnas y normalización, reseteando índices de forma segura."""
    
    # NASA (Base Maestra)
    cols_nasa = ['pl_name', 'ra', 'dec'] + COLS_FISICAS_MICE
    master_df = df_nasa[cols_nasa].copy()
    master_df = master_df.dropna(subset=['ra', 'dec']).reset_index(drop=True)
    
    # EUROPA
    cols_eu = ['name', 'ra', 'dec', 'mass', 'radius', 'temp_calculated', 
               'orbital_period', 'eccentricity', 'star_teff', 'star_mass']
    df_eu_clean = df_eu[cols_eu].copy()
    # Reset explícito para evitar bugs latentes en asignación de índices
    df_eu_clean = df_eu_clean.dropna(subset=['ra', 'dec']).reset_index(drop=True)
    
    df_eu_clean['eu_mass_earth'] = df_eu_clean['mass'] * 317.8
    df_eu_clean['eu_radius_earth'] = df_eu_clean['radius'] * 11.209
    df_eu_clean.rename(columns={
        'temp_calculated': 'eu_temp',
        'orbital_period': 'eu_orbital_period',
        'eccentricity': 'eu_eccentricity',
        'star_teff': 'eu_star_teff',
        'star_mass': 'eu_star_mass'
    }, inplace=True)
    
    # PHL
    df_phl_clean = df_phl[['P_NAME', 'S_RA', 'S_DEC', 'P_ESI', 'P_HABITABLE']].copy()
    df_phl_clean = df_phl_clean.dropna(subset=['S_RA', 'S_DEC']).reset_index(drop=True)
    
    return master_df, df_eu_clean, df_phl_clean

def ejecutar_entity_resolution(master_df: pd.DataFrame, df_eu_clean: pd.DataFrame, df_phl_clean: pd.DataFrame) -> pd.DataFrame:
    """Cruza los catálogos usando KD-Tree 3D con política Closest-Wins."""

    # Árbol de la NASA en 3D
    coords_nasa = _to_cartesian(master_df['ra'].values, master_df['dec'].values)
    arbol_nasa = cKDTree(coords_nasa)
    
    # --- CRUCE EUROPA ---
    coords_eu = _to_cartesian(df_eu_clean['ra'].values, df_eu_clean['dec'].values)
    distancias_eu, indices_eu = arbol_nasa.query(coords_eu, distance_upper_bound=TOLERANCIA_CHORD)
    matches_eu = distancias_eu != np.inf
    
    # Tabla de correspondencia para resolver colisiones (closest-wins)
    corr_eu = pd.DataFrame({
        'idx_nasa': indices_eu[matches_eu],
        'idx_eu': np.where(matches_eu)[0],
        'distancia': distancias_eu[matches_eu],
    }).sort_values('distancia').drop_duplicates(subset='idx_nasa', keep='first')
    
    colisiones_eu = matches_eu.sum() - len(corr_eu)
    if colisiones_eu > 0:
        logging.warning("Entity Resolution EU: %d colisión/es resuelta/s (política: closest-wins).", colisiones_eu)

    cols_a_transferir = ['eu_mass_earth', 'eu_radius_earth', 'eu_temp', 'eu_orbital_period', 'eu_eccentricity', 'eu_star_teff', 'eu_star_mass']
    eu_subset = df_eu_clean.loc[corr_eu['idx_eu'], cols_a_transferir]
    eu_subset.index = corr_eu['idx_nasa'].values
    master_df.loc[eu_subset.index, cols_a_transferir] = eu_subset
    
    # --- CRUCE PHL ---
    coords_phl = _to_cartesian(df_phl_clean['S_RA'].values, df_phl_clean['S_DEC'].values)
    distancias_phl, indices_phl = arbol_nasa.query(coords_phl, distance_upper_bound=TOLERANCIA_CHORD)
    matches_phl = distancias_phl != np.inf
    
    corr_phl = pd.DataFrame({
        'idx_nasa': indices_phl[matches_phl],
        'idx_phl': np.where(matches_phl)[0],
        'distancia': distancias_phl[matches_phl],
    }).sort_values('distancia').drop_duplicates(subset='idx_nasa', keep='first')
    
    colisiones_phl = matches_phl.sum() - len(corr_phl)
    if colisiones_phl > 0:
        logging.warning("Entity Resolution PHL: %d colisión/es resuelta/s (política: closest-wins).", colisiones_phl)

    phl_subset = df_phl_clean.loc[corr_phl['idx_phl'], ['P_ESI', 'P_HABITABLE']]
    phl_subset.index = corr_phl['idx_nasa'].values
    master_df.loc[phl_subset.index, ['P_ESI', 'P_HABITABLE']] = phl_subset.values
    
    # Renombrar para consistencia interna
    master_df.rename(columns={'P_ESI': 'phl_esi', 'P_HABITABLE': 'phl_habitable'}, inplace=True)
    
    return master_df

# Función de imputación híbrida con reporte de linaje
def imputar_datos(master_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ejecuta los 3 niveles de imputación y retorna reportes de linaje."""
    
    # Auditamos solo las columnas de la NASA imputables
    nulos_inicio = master_df[COLS_FISICAS_MICE].isna().sum().sum()

    # --- NIVEL 1: Fusión (Cross-Filling) ---
    master_df['pl_bmasse'] = master_df['pl_bmasse'].fillna(master_df['eu_mass_earth'])
    master_df['pl_rade'] = master_df['pl_rade'].fillna(master_df['eu_radius_earth'])
    master_df['pl_eqt'] = master_df['pl_eqt'].fillna(master_df['eu_temp'])
    master_df['pl_orbper'] = master_df['pl_orbper'].fillna(master_df['eu_orbital_period'])
    master_df['pl_orbeccen'] = master_df['pl_orbeccen'].fillna(master_df['eu_eccentricity'])
    master_df['st_teff'] = master_df['st_teff'].fillna(master_df['eu_star_teff'])
    master_df['st_mass'] = master_df['st_mass'].fillna(master_df['eu_star_mass'])
    
    cols_a_borrar = ['eu_mass_earth', 'eu_radius_earth', 'eu_temp', 'eu_orbital_period', 'eu_eccentricity', 'eu_star_teff', 'eu_star_mass']
    master_df = master_df.drop(columns=cols_a_borrar)
    
    nulos_fase_1 = master_df[COLS_FISICAS_MICE].isna().sum().sum()
    imputados_fusion = nulos_inicio - nulos_fase_1

    # --- NIVEL 2: Física ---
    idx_no_temp = master_df['pl_eqt'].isna() & master_df['pl_insol'].notna()
    master_df.loc[idx_no_temp, 'pl_eqt'] = calc_stefan_boltzmann_temp(master_df.loc[idx_no_temp, 'pl_insol'])

    idx_no_insol = master_df['pl_insol'].isna() & master_df['pl_eqt'].notna()
    master_df.loc[idx_no_insol, 'pl_insol'] = calc_stefan_boltzmann_insol(master_df.loc[idx_no_insol, 'pl_eqt'])

    idx_no_dens = master_df['pl_dens'].isna() & master_df['pl_bmasse'].notna() & master_df['pl_rade'].notna()
    master_df.loc[idx_no_dens, 'pl_dens'] = calc_densidad_planetaria(master_df.loc[idx_no_dens, 'pl_bmasse'], master_df.loc[idx_no_dens, 'pl_rade'])
    
    nulos_fase_2 = master_df[COLS_FISICAS_MICE].isna().sum().sum()
    imputados_fisica = nulos_fase_1 - nulos_fase_2

    # --- NIVEL 3: Machine Learning (MICE) ---
    estimador = ExtraTreesRegressor(n_estimators=10, random_state=SEMILLA_ALEATORIA)
    imputador_mice = IterativeImputer(estimator=estimador, max_iter=MICE_ITERACIONES, random_state=SEMILLA_ALEATORIA)
    
    master_df[COLS_FISICAS_MICE] = imputador_mice.fit_transform(master_df[COLS_FISICAS_MICE])
    
    nulos_fase_3 = master_df[COLS_FISICAS_MICE].isna().sum().sum()
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
    """Orquestador de la Capa Plata."""
    logging.info("Iniciando Procesamiento (Capa Plata)...")
    
    df_nasa, df_eu, df_phl = extract_bronce_data()
    master_df, df_eu_clean, df_phl_clean = limpiar_y_filtrar(df_nasa, df_eu, df_phl)
    
    logging.info("Ejecutando Fuzzy Matching Espacial (Geometría 3D + Closest-Wins)...")
    master_df = ejecutar_entity_resolution(master_df, df_eu_clean, df_phl_clean)
    
    logging.info("Aplicando Imputación Híbrida (Fusión -> Física -> MICE)...")
    master_df, linaje = imputar_datos(master_df)
    
    # Nuevo Reporte de Auditoría
    print("\n" + "="*50)
    print(" REPORTE DE LINAJE Y COBERTURA:")
    print(f"  - Planetas NASA sin etiqueta PHL  : {(master_df['phl_esi'].isna()).sum()}")
    print(f"  - Cobertura PHL (Dataset Viable)  : {(master_df['phl_esi'].notna().mean()*100):.1f}%")
    print("-" * 50)
    print(f"  - Rescatados empíricamente (Europa) : {linaje['Fusion']}")
    print(f"  - Deducidos matemáticamente (Física): {linaje['Fisica']}")
    print(f"  - Inferidos algorítmicamente (MICE) : {linaje['MICE']}")
    print(f"  - TOTAL NULOS ELIMINADOS            : {linaje['Total_Rescatados']}")
    print("="*50 + "\n")
    
    os.makedirs(RUTA_SILVER_DIR, exist_ok=True)
    master_df.to_csv(ARCHIVO_SALIDA, index=False)
    
    logging.info(f"Capa Plata construida con éxito. Archivo: {ARCHIVO_SALIDA}")
    return master_df

if __name__ == "__main__":
    df_final = ejecutar_procesamiento()