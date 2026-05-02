# test_imputation.py
import pandas as pd
import numpy as np

print("1. Descargando datos de la NASA...")
url = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+pl_name,pl_rade,pl_bmasse,pl_eqt,pl_orbsmax,"
    "pl_insol,st_teff,st_rad,st_mass,st_met,pl_dens"
    "+from+pscomppars&format=csv"
)
df = pd.read_csv(url)

# Recreamos la categorización del Gap de Fulton rápidamente
condiciones = [
    (df['pl_rade'] >= 0.0) & (df['pl_rade'] < 1.5),
    (df['pl_rade'] >= 1.5) & (df['pl_rade'] < 2.0),
    (df['pl_rade'] >= 2.0) & (df['pl_rade'] < 4.0),
    (df['pl_rade'] >= 4.0) & (df['pl_rade'] < 10.0),
    (df['pl_rade'] >= 10.0)
]
categorias = ["Terrestre", "Super-Tierra", "Sub-Neptuno", "Neptuniano", "Joviano"]
df['tipo_planeta'] = np.select(condiciones, categorias, default='Desconocido/Sin datos')

features_criticas = ['pl_rade', 'pl_bmasse', 'pl_dens', 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met']

print("\nNaNs iniciales (Antes de la imputación):")
faltantes_antes = df[features_criticas].isnull().sum()
print(faltantes_antes[faltantes_antes > 0])

print("\nIniciando Imputación (Capa de Servicio)...")

# ---------------------------------------------------------
# NIVEL 1: IMPUTACIÓN FÍSICA (LEYES TERMODINÁMICAS)
# ---------------------------------------------------------
print(" [Nivel 1] Aplicando reglas astrofísicas:")

# A. Temperatura desde la insolación (T_eq = 278.5 * S^0.25)
mask_eqt = df['pl_eqt'].isna() & df['pl_insol'].notna()
df.loc[mask_eqt, 'pl_eqt'] = 278.5 * (df.loc[mask_eqt, 'pl_insol'] ** 0.25)
print(f"   ✓ Rescatadas {mask_eqt.sum()} temperaturas usando la Ley de Stefan-Boltzmann.")

# B. Densidad desde la masa y el radio (Densidad = Masa / Volumen)
mask_dens = df['pl_dens'].isna() & df['pl_bmasse'].notna() & df['pl_rade'].notna()
# 5.51 g/cm³ es la densidad de la Tierra. Escalamos relativo a masas y radios terrestres.
df.loc[mask_dens, 'pl_dens'] = 5.51 * df.loc[mask_dens, 'pl_bmasse'] / (df.loc[mask_dens, 'pl_rade'] ** 3)
print(f"   ✓ Rescatadas {mask_dens.sum()} densidades usando fórmulas volumétricas.")

# ---------------------------------------------------------
# NIVEL 2: IMPUTACIÓN ESTADÍSTICA (MEDIANA POR GRUPO)
# ---------------------------------------------------------
print("\n  [Nivel 2] Aplicando medianas por tipo de planeta:")
print("   (Ej: Un Neptuniano sin masa recibe la masa promedio de otros Neptunianos)")

for group in df['tipo_planeta'].unique():
    if group == 'Desconocido/Sin datos': 
        continue # No imputamos a los que ni siquiera sabemos qué son
        
    mask = df['tipo_planeta'] == group
    for col in features_criticas:
        median_val = df.loc[mask, col].median()
        # Si el valor mediano existe (no es NaN), tapamos los huecos de ese grupo
        if pd.notna(median_val):
            df.loc[mask & df[col].isna(), col] = median_val

# ---------------------------------------------------------
# RESULTADO FINAL
# ---------------------------------------------------------
print("\n Imputación finalizada. NaNs restantes en variables críticas:")
faltantes_despues = df[features_criticas].isnull().sum()
print(faltantes_despues)

if faltantes_despues.sum() < 500:
    print("\n¡Data limpia! Salvamos casi todo el dataset. Listo para inyectar a la Red Neuronal.")
else:
    print("\Todavía quedan nulos, pero reducimos drásticamente la basura.")