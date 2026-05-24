import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DIR_SILVER = "data/silver"
DIR_FIGURAS = "reports/figures"
os.makedirs(DIR_FIGURAS, exist_ok=True)

sns.set_theme(style="white", context="paper", font_scale=1.2)

def generar_heatmap():
    print("Cargando Capa Plata para análisis de correlación...")

    ruta_datos = f"{DIR_SILVER}/data_lake_consolidado.csv"
    
    if not os.path.exists(ruta_datos):
        print(f"Error: No se encontró el dataset en {ruta_datos}.")
        print("  Por favor, ejecutá primero el pipeline de datos (ingestion.py y processing.py) para generar este archivo localmente.")
        return

    v4 = pd.read_csv(f"{DIR_SILVER}/data_lake_consolidado.csv", low_memory=False)
    
    columnas_fisicas = [
        'pl_orbper', 'pl_orbeccen', 'pl_bmasse', 'pl_rade', 
        'pl_dens', 'pl_eqt', 'pl_insol', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_age'
    ]
    df_corr = v4[columnas_fisicas]
    
    # Spearman para mitigar el impacto distorsivo de outliers extremos de la física
    matriz_correlacion = df_corr.corr(method='spearman')

    plt.figure(figsize=(12, 10))
    
    # Máscara triangular superior para evitar redundancia en espejo
    mascara = np.triu(np.ones_like(matriz_correlacion, dtype=bool))
    
    sns.heatmap(matriz_correlacion, mask=mascara, cmap='coolwarm', 
                vmax=1, vmin=-1, center=0, annot=True, fmt=".2f", 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})

    plt.title('Matriz de Correlación de Atributos Exoplanetarios\n(Dataset Silver)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    ruta_salida = f'{DIR_FIGURAS}/eda_01_matriz_correlacion.png'
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" ✓ Gráfico guardado en: {ruta_salida}")

if __name__ == "__main__":
    generar_heatmap()