import pandas as pd
import numpy as np

# 1. Traer los datos (Repository)
print("Conectando con la API de la NASA...")
url = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+pl_name,pl_rade,pl_bmasse,pl_eqt,pl_orbsmax,"
    "pl_insol,st_teff,st_rad,st_mass,st_met,pl_dens"
    "+from+pscomppars&format=csv"
)

try:
    df = pd.read_csv(url)
    print(f"Datos descargados. Total de planetas: {len(df)}")

    # 2. Ver la "basura" (Missing Values). #Filtramos para mostrar TODAS las columnas que tengan algún dato faltante.
    print("\nColumnas con datos faltantes (NaNs):")
    missing = (df.isnull().sum() / len(df) * 100).round(2).sort_values(ascending=False)
    print(missing[missing > 0].astype(str) + " %")

    # 3. Categorizar usando el Gap de Fulton (Service / Lógica)
    print("\nCategorizando planetas según Gap de Fulton...")
    condiciones = [
        (df['pl_rade'] >= 0.0) & (df['pl_rade'] < 1.5),
        (df['pl_rade'] >= 1.5) & (df['pl_rade'] < 2.0),
        (df['pl_rade'] >= 2.0) & (df['pl_rade'] < 4.0),
        (df['pl_rade'] >= 4.0) & (df['pl_rade'] < 10.0),
        (df['pl_rade'] >= 10.0)
    ]
    categorias = ["Terrestre", "Super-Tierra", "Sub-Neptuno", "Neptuniano", "Joviano"]

    # select de numpy es como un gran switch-case
    df['tipo_planeta'] = np.select(condiciones, categorias, default='Desconocido/Sin datos')

    print("\nResultados de la clasificación inicial:")
    print(df['tipo_planeta'].value_counts())

    # Imprimir una muestra real
    print("\nMuestra de 3 planetas aleatorios:")
    print(df[['pl_name', 'pl_rade', 'tipo_planeta']].dropna().sample(3).to_string())

except Exception as e:
    print(f"Error durante la ejecución: {e}")