from sklearn.experimental import enable_iterative_imputer # Necesario para activar MICE
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import IterativeImputer
from scipy.spatial import cKDTree

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore", category=ConvergenceWarning)

print("Iniciando Módulo 2: Procesamiento y Entity Resolution...")

# =====================================================================
# 1. EXTRACCIÓN (Leyendo la Capa Bronce)
# =====================================================================
print("   Leyendo datos crudos (Capa Bronce)...")
df_nasa = pd.read_csv("data/raw/nasa_exoplanets.csv", low_memory=False)
df_eu = pd.read_csv("data/raw/eu_exoplanets.csv", low_memory=False)
df_phl = pd.read_csv("data/raw/phl_exoplanets.csv", low_memory=False)

# =====================================================================
# 2. TRANSFORMACIÓN: Filtrado y Normalización
# =====================================================================
print("   Limpiando y filtrando columnas clave...")

# --- NASA (Nuestra base principal) ---
cols_nasa = [
    'pl_name', 'ra', 'dec', 'pl_rade', 'pl_bmasse', 'pl_dens', 'pl_eqt', 'pl_insol', 
    'pl_orbeccen', 'pl_orbper',  # Nuevas joyas dinámicas
    'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_age' # Nueva joya estelar
]
# Descartamos el ruido y nos quedamos con lo físico
master_df = df_nasa[cols_nasa].copy()
# Borramos filas que no tengan coordenadas (es imposible ubicarlas)
master_df = master_df.dropna(subset=['ra', 'dec']).reset_index(drop=True)


# --- EXOPLANET.EU (Europa) ---
# Extraemos y convertimos unidades: de Júpiter a Tierra
df_eu_clean = df_eu[['name', 'ra', 'dec', 'mass', 'radius', 'temp_calculated']].copy()
df_eu_clean = df_eu_clean.dropna(subset=['ra', 'dec'])

df_eu_clean['eu_mass_earth'] = df_eu_clean['mass'] * 317.8
df_eu_clean['eu_radius_earth'] = df_eu_clean['radius'] * 11.209
df_eu_clean.rename(columns={'temp_calculated': 'eu_temp'}, inplace=True)


# --- PHL (Arecibo) ---
# Extraemos el Target validado (ESI y Habitable)
df_phl_clean = df_phl[['P_NAME', 'S_RA', 'S_DEC', 'P_ESI', 'P_HABITABLE']].copy()
df_phl_clean = df_phl_clean.dropna(subset=['S_RA', 'S_DEC'])


# =====================================================================
# 3. ENTITY RESOLUTION (El Cruce Espacial con K-D Tree)
# =====================================================================
print("   Ejecutando emparejamiento espacial (Fuzzy Matching por coordenadas)...")

# Tolerancia en grados (0.01 grados en el cielo es una ventana microscópica, ideal para match)
TOLERANCIA_GRADOS = 0.01

# Construimos el índice de búsqueda espacial basándonos en la NASA
arbol_nasa = cKDTree(master_df[['ra', 'dec']].values)

# --- CRUCE CON EUROPA ---
distancias_eu, indices_nasa_eu = arbol_nasa.query(df_eu_clean[['ra', 'dec']].values, distance_upper_bound=TOLERANCIA_GRADOS)
# Filtramos solo los que hicieron "Match" (distancia no infinita)
matches_eu = distancias_eu != np.inf
indices_validos_eu = indices_nasa_eu[matches_eu]

# Inyectamos los datos de Europa en el Master DataFrame de la NASA
master_df.loc[indices_validos_eu, 'eu_mass_earth'] = df_eu_clean.loc[matches_eu, 'eu_mass_earth'].values
master_df.loc[indices_validos_eu, 'eu_radius_earth'] = df_eu_clean.loc[matches_eu, 'eu_radius_earth'].values
master_df.loc[indices_validos_eu, 'eu_temp'] = df_eu_clean.loc[matches_eu, 'eu_temp'].values


# --- CRUCE CON PHL (Arecibo) ---
distancias_phl, indices_nasa_phl = arbol_nasa.query(df_phl_clean[['S_RA', 'S_DEC']].values, distance_upper_bound=TOLERANCIA_GRADOS)
matches_phl = distancias_phl != np.inf
indices_validos_phl = indices_nasa_phl[matches_phl]

master_df.loc[indices_validos_phl, 'phl_esi'] = df_phl_clean.loc[matches_phl, 'P_ESI'].values
master_df.loc[indices_validos_phl, 'phl_habitable'] = df_phl_clean.loc[matches_phl, 'P_HABITABLE'].values

print(f"      ✓ Planetas de Europa emparejados: {matches_eu.sum()}")
print(f"      ✓ Planetas de PHL emparejados: {matches_phl.sum()}")

# =====================================================================
# 4. IMPUTACIÓN (Fusión, Física y Estadística)
# =====================================================================

print("\n   Diagnosticando la completitud de los datos (Pre-Imputación)...")

# --- DIAGNÓSTICO DE COMPLETITUD ---
# Calculamos el porcentaje de valores nulos por columna
missing_pct = (master_df.isna().sum() / len(master_df)) * 100

# Filtramos solo las columnas que tienen faltantes y las ordenamos de mayor a menor
missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)

print("      Reporte de Datos Faltantes:")
for col, pct in missing_cols.items():
    print(f"         - {col:<15}: {pct:>5.2f}%")

print("\n   Iniciando rescate matemático y estadístico...")

# FOTO 0: Total de nulos al inicio (ignorando los targets del PHL)
columnas_features = [col for col in master_df.columns if 'phl' not in col]
nulos_fase_0 = master_df[columnas_features].isna().sum().sum()

# --- NIVEL 1: Cross-Filling (Fusión de fuentes) ---
# Rellenamos NaNs de NASA con datos de Europa si existen
master_df['pl_bmasse'] = master_df['pl_bmasse'].fillna(master_df['eu_mass_earth'])
master_df['pl_rade'] = master_df['pl_rade'].fillna(master_df['eu_radius_earth'])
master_df['pl_eqt'] = master_df['pl_eqt'].fillna(master_df['eu_temp'])

# Una vez fusionados, borramos las columnas de Europa para no duplicar datos
master_df = master_df.drop(columns=['eu_mass_earth', 'eu_radius_earth', 'eu_temp'])

# FOTO 1: Nulos post-fusión
columnas_features = [col for col in master_df.columns if 'phl' not in col]
nulos_fase_1 = master_df[columnas_features].isna().sum().sum()
imputados_fusion = nulos_fase_0 - nulos_fase_1

# --- NIVEL 2: Imputación Física Pura ---
# A) Si falta Temperatura, la deducimos de la Insolación
idx_no_temp = master_df['pl_eqt'].isna() & master_df['pl_insol'].notna()
master_df.loc[idx_no_temp, 'pl_eqt'] = 255 * (master_df.loc[idx_no_temp, 'pl_insol'] ** 0.25)

# B) Si falta Insolación, la deducimos de la Temperatura
idx_no_insol = master_df['pl_insol'].isna() & master_df['pl_eqt'].notna()
master_df.loc[idx_no_insol, 'pl_insol'] = (master_df.loc[idx_no_insol, 'pl_eqt'] / 255) ** 4

# C) Si falta Densidad, la deducimos de Masa y Radio
idx_no_dens = master_df['pl_dens'].isna() & master_df['pl_bmasse'].notna() & master_df['pl_rade'].notna()
master_df.loc[idx_no_dens, 'pl_dens'] = 5.51 * (master_df.loc[idx_no_dens, 'pl_bmasse'] / (master_df.loc[idx_no_dens, 'pl_rade'] ** 3))

# FOTO 2: Nulos post-física
nulos_fase_2 = master_df[columnas_features].isna().sum().sum()
imputados_fisica = nulos_fase_1 - nulos_fase_2

# --- NIVEL 3: Imputación Algorítmica con MICE ---
print("      Iniciando Imputación MICE (IterativeImputer)...     Aproximadamente 30 segundos.")

# 1. Definimos las columnas que queremos que MICE observe y arregle. 
# IMPORTANTE: Excluimos 'pl_name', 'phl_esi' y 'phl_habitable'. 
# Los nombres son texto (MICE solo usa números) y los targets del PHL no se tocan.
cols_para_mice = [
    'ra', 'dec', 'pl_rade', 'pl_bmasse', 'pl_dens', 'pl_eqt', 'pl_insol', 
    'pl_orbeccen', 'pl_orbper', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_age'
]

# 2. Aislamos solo los datos numéricos físicos para el algoritmo
df_para_imputar = master_df[cols_para_mice].copy()

# 3. Configuramos el imputador MICE. 
# Usamos ExtraTreesRegressor porque es muy robusto para relaciones no lineales (como la física)
imputador_mice = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), 
    max_iter=20, 
    random_state=42
)

# 4. Entrenamos y aplicamos la magia
df_imputado_numpy = imputador_mice.fit_transform(df_para_imputar)

# 5. Volcamos los datos limpios de vuelta al DataFrame Maestro
master_df[cols_para_mice] = df_imputado_numpy

# FOTO 3: Nulos post-MICE
nulos_fase_3 = master_df[columnas_features].isna().sum().sum()
imputados_mice = nulos_fase_2 - nulos_fase_3

# --- REPORTE DE LINAJE ---
print("\n      REPORTE DE LINAJE DE IMPUTACIÓN:")
print(f"         - Rescatados por cruce de catálogos (Exoplanet.eu) : {imputados_fusion} datos.")
print(f"         - Deducidos por Leyes Físicas (Termodinámica)      : {imputados_fisica} datos.")
print(f"         - Inferidos mediante Machine Learning (MICE)       : {imputados_mice} datos.")
print(f"         - Total de valores nulos eliminados                : {imputados_fusion + imputados_fisica + imputados_mice}")

# =====================================================================
# 5. CARGA (Guardando la Capa Plata)
# =====================================================================

os.makedirs("data/processed", exist_ok=True)
archivo_salida = "data/processed/data_lake_consolidado.csv"
master_df.to_csv(archivo_salida, index=False)

print(f"\n¡Capa Plata construida con éxito!")
print(f"   Catálogo Base (NASA): {master_df.shape[0]} planetas 100% confirmados.")
print(f"   Estructura final: {master_df.shape[0]} filas y {master_df.shape[1]} columnas.")
print(f"   Guardado en: {archivo_salida}")
