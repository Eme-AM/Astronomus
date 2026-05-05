# src/data/processing.py
import pandas as pd
import numpy as np
import os
import sys
import warnings
from scipy.spatial import cKDTree

from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import IterativeImputer

# Silenciamos la advertencia específica de MICE para mantener limpia la consola
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Esto permite poder importar las funciones de physics.py en la línea 21
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Importamos nuestro módulo de astrofísica
from src.core.physics import calc_stefan_boltzmann_temp, calc_stefan_boltzmann_insol, calc_densidad_planetaria

def extract_bronce_data(ruta_raw="data/raw"):
    """Carga los datasets crudos desde la Capa Bronce."""
    try:
        df_nasa = pd.read_csv(f"{ruta_raw}/nasa_exoplanets.csv", low_memory=False)
        df_eu = pd.read_csv(f"{ruta_raw}/eu_exoplanets.csv", low_memory=False)
        df_phl = pd.read_csv(f"{ruta_raw}/phl_exoplanets.csv", low_memory=False)
        return df_nasa, df_eu, df_phl
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Falta un archivo en la Capa Bronce. Ejecutá la ingesta primero. Detalle: {e}")

def limpiar_y_filtrar(df_nasa, df_eu, df_phl):
    """Aplica filtrado de columnas y normalización de unidades por fuente."""
    
    # NASA
    cols_nasa = ['pl_name', 'ra', 'dec', 'pl_rade', 'pl_bmasse', 'pl_dens', 'pl_eqt', 'pl_insol', 
                 'pl_orbeccen', 'pl_orbper', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_age']
    master_df = df_nasa[cols_nasa].copy()
    master_df = master_df.dropna(subset=['ra', 'dec']).reset_index(drop=True)
    
    # EUROPA
    df_eu_clean = df_eu[['name', 'ra', 'dec', 'mass', 'radius', 'temp_calculated']].copy()
    df_eu_clean = df_eu_clean.dropna(subset=['ra', 'dec'])
    df_eu_clean['eu_mass_earth'] = df_eu_clean['mass'] * 317.8    # Júpiter a Tierra
    df_eu_clean['eu_radius_earth'] = df_eu_clean['radius'] * 11.209 # Júpiter a Tierra
    df_eu_clean.rename(columns={'temp_calculated': 'eu_temp'}, inplace=True)
    
    # PHL (Arecibo)
    df_phl_clean = df_phl[['P_NAME', 'S_RA', 'S_DEC', 'P_ESI', 'P_HABITABLE']].copy()
    df_phl_clean = df_phl_clean.dropna(subset=['S_RA', 'S_DEC'])
    
    return master_df, df_eu_clean, df_phl_clean

def ejecutar_entity_resolution(master_df, df_eu_clean, df_phl_clean, tolerancia_grados=0.01):
    """Cruza los catálogos utilizando K-D Trees (Fuzzy Matching espacial)."""
    arbol_nasa = cKDTree(master_df[['ra', 'dec']].values)
    
    # Cruce Europa
    distancias_eu, indices_eu = arbol_nasa.query(df_eu_clean[['ra', 'dec']].values, distance_upper_bound=tolerancia_grados)
    matches_eu = distancias_eu != np.inf
    idx_validos_eu = indices_eu[matches_eu]
    master_df.loc[idx_validos_eu, 'eu_mass_earth'] = df_eu_clean.loc[matches_eu, 'eu_mass_earth'].values
    master_df.loc[idx_validos_eu, 'eu_radius_earth'] = df_eu_clean.loc[matches_eu, 'eu_radius_earth'].values
    master_df.loc[idx_validos_eu, 'eu_temp'] = df_eu_clean.loc[matches_eu, 'eu_temp'].values
    
    # Cruce PHL
    distancias_phl, indices_phl = arbol_nasa.query(df_phl_clean[['S_RA', 'S_DEC']].values, distance_upper_bound=tolerancia_grados)
    matches_phl = distancias_phl != np.inf
    idx_validos_phl = indices_phl[matches_phl]
    master_df.loc[idx_validos_phl, 'phl_esi'] = df_phl_clean.loc[matches_phl, 'P_ESI'].values
    master_df.loc[idx_validos_phl, 'phl_habitable'] = df_phl_clean.loc[matches_phl, 'P_HABITABLE'].values
    
    return master_df

def imputar_datos(master_df):
    """Ejecuta los 3 niveles de imputación y retorna reportes de linaje."""
    
    # Excluimos los targets (phl) y las columnas temporales de Europa (eu_)
    # Así el conteo de NaNs se hace estrictamente sobre nuestras variables astrofísicas finales.
    cols_seguimiento = [col for col in master_df.columns if 'phl' not in col and 'eu_' not in col]
    nulos_inicio = master_df[cols_seguimiento].isna().sum().sum()

    # --- NIVEL 1: Fusión ---
    master_df['pl_bmasse'] = master_df['pl_bmasse'].fillna(master_df['eu_mass_earth'])
    master_df['pl_rade'] = master_df['pl_rade'].fillna(master_df['eu_radius_earth'])
    master_df['pl_eqt'] = master_df['pl_eqt'].fillna(master_df['eu_temp'])
    
    # Eliminamos las columnas auxiliares
    master_df = master_df.drop(columns=['eu_mass_earth', 'eu_radius_earth', 'eu_temp'])
    
    # Foto 1
    nulos_fase_1 = master_df[cols_seguimiento].isna().sum().sum()
    imputados_fusion = nulos_inicio - nulos_fase_1

    # --- NIVEL 2: Física (Usando src/core/physics.py) ---
    idx_no_temp = master_df['pl_eqt'].isna() & master_df['pl_insol'].notna()
    master_df.loc[idx_no_temp, 'pl_eqt'] = calc_stefan_boltzmann_temp(master_df.loc[idx_no_temp, 'pl_insol'])

    idx_no_insol = master_df['pl_insol'].isna() & master_df['pl_eqt'].notna()
    master_df.loc[idx_no_insol, 'pl_insol'] = calc_stefan_boltzmann_insol(master_df.loc[idx_no_insol, 'pl_eqt'])

    idx_no_dens = master_df['pl_dens'].isna() & master_df['pl_bmasse'].notna() & master_df['pl_rade'].notna()
    master_df.loc[idx_no_dens, 'pl_dens'] = calc_densidad_planetaria(master_df.loc[idx_no_dens, 'pl_bmasse'], master_df.loc[idx_no_dens, 'pl_rade'])
    
    # Foto 2
    nulos_fase_2 = master_df[cols_seguimiento].isna().sum().sum()
    imputados_fisica = nulos_fase_1 - nulos_fase_2

    # --- NIVEL 3: Machine Learning (MICE) ---
    cols_para_mice = ['ra', 'dec', 'pl_rade', 'pl_bmasse', 'pl_dens', 'pl_eqt', 'pl_insol', 
                      'pl_orbeccen', 'pl_orbper', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_age']
    
    imputador_mice = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), max_iter=10, random_state=42)
    master_df[cols_para_mice] = imputador_mice.fit_transform(master_df[cols_para_mice])
    
    # Foto 3
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

def ejecutar_procesamiento():  # Main
    """Función principal (Orquestador) que ejecuta todo el pipeline de la Capa Plata."""
    print("Iniciando Procesamiento (Capa Plata)...")
    
    # 1. Extracción
    print("   Extrayendo datos raw...")
    df_nasa, df_eu, df_phl = extract_bronce_data()
    
    # 2. Transformación inicial
    print("   Limpiando y estandarizando...")
    master_df, df_eu_clean, df_phl_clean = limpiar_y_filtrar(df_nasa, df_eu, df_phl)
    
    # 3. Entity Resolution
    print("   Ejecutando Fuzzy Matching Espacial (K-D Tree)...")
    master_df = ejecutar_entity_resolution(master_df, df_eu_clean, df_phl_clean)
    
    # 4. Imputación Híbrida
    print("   Aplicando Imputación Híbrida (Fusión -> Física -> MICE)...")
    master_df, linaje = imputar_datos(master_df)
    
    # Imprimimos Linaje
    print("\n   REPORTE DE LINAJE DE IMPUTACIÓN:")
    print(f"      - Rescatados por cruce de catálogos : {linaje['Fusion']} datos.")
    print(f"      - Deducidos por Leyes Físicas       : {linaje['Fisica']} datos.")
    print(f"      - Inferidos mediante ML (MICE)      : {linaje['MICE']} datos.")
    print(f"      - Total rescatado                   : {linaje['Total_Rescatados']} de {linaje['Total_Nulos_Originales']} nulos iniciales.\n")
    
    # 5. Carga a Capa Plata
    os.makedirs("data/processed", exist_ok=True)
    archivo_salida = "data/processed/data_lake_consolidado.csv"
    master_df.to_csv(archivo_salida, index=False)
    
    #print(f"Capa Plata construida con éxito. Guardado en: {archivo_salida}")
    return master_df

# Bloque de ejecución principal (Para poder correrlo directamente o importarlo)
if __name__ == "__main__":
    df_final = ejecutar_procesamiento()