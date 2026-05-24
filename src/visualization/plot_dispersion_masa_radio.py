import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DIR_SILVER = "data/silver"
DIR_FIGURAS = "reports/figures"
os.makedirs(DIR_FIGURAS, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def generar_scatter_astrofisico():
    print("Cargando Capa Plata para Scatter Plot...")

    ruta_datos = f"{DIR_SILVER}/data_lake_consolidado.csv"
    
    if not os.path.exists(ruta_datos):
        print(f"Error: No se encontró el dataset en {ruta_datos}.")
        print("  Por favor, ejecutá primero el pipeline de datos (ingestion.py y processing.py) para generar este archivo localmente.")
        return

    v4 = pd.read_csv(f"{DIR_SILVER}/data_lake_consolidado.csv", low_memory=False)
    
    # Filtro de seguridad para evitar quiebres en la escala logarítmica
    df_plot = v4[(v4['pl_bmasse'] > 0) & (v4['pl_rade'] > 0)].copy()

    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=df_plot, x='pl_bmasse', y='pl_rade', hue='pl_eqt',
        palette='Spectral_r', s=45, alpha=0.8, edgecolor='black', linewidth=0.2
    )

    # Transformación a escala logarítmica espacial
    plt.xscale('log')
    plt.yscale('log')

    # Líneas de referencia del Sistema Solar
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.6)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.6)
    plt.text(1.1, 1.1, 'Tierra', color='#555555', fontsize=11, fontweight='bold')

    plt.axvline(x=317.8, color='gray', linestyle='-.', alpha=0.6)
    plt.axhline(y=11.2, color='gray', linestyle='-.', alpha=0.6)
    plt.text(350, 12, 'Júpiter', color='#555555', fontsize=11, fontweight='bold')

    plt.title('Morfología Exoplanetaria: Masa vs. Radio', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Masa Planetaria (Masas Terrestres) [Escala Log]', fontsize=13)
    plt.ylabel('Radio Planetario (Radios Terrestres) [Escala Log]', fontsize=13)
    
    plt.legend(title='Temperatura (K)', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()

    ruta_salida = f'{DIR_FIGURAS}/eda_02_masa_radio.png'
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" ✓ Gráfico guardado en: {ruta_salida}")

if __name__ == "__main__":
    generar_scatter_astrofisico()