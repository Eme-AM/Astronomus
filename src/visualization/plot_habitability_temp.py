import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DIR_SILVER = "data/silver"
DIR_FIGURAS = "reports/figures"
os.makedirs(DIR_FIGURAS, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def generar_violin_habitabilidad():
    print("Cargando datos para análisis de habitabilidad...")

    ruta_datos = f"{DIR_SILVER}/data_lake_consolidado.csv"
    
    if not os.path.exists(ruta_datos):
        print(f"Error: No se encontró el dataset en {ruta_datos}.")
        print("  Por favor, ejecutá primero el pipeline de datos (ingestion.py y processing.py) para generar este archivo localmente.")
        return

    v4 = pd.read_csv(f"{DIR_SILVER}/data_lake_consolidado.csv", low_memory=False)
    df_plot = v4.dropna(subset=['phl_habitable', 'pl_eqt']).copy()

    mapeo_habitabilidad = {0.0: '0 - Inhóspito', 1.0: '1 - Conservador', 2.0: '2 - Optimista'}
    df_plot['phl_habitable_label'] = df_plot['phl_habitable'].map(mapeo_habitabilidad).fillna(df_plot['phl_habitable'].astype(str))
    df_plot = df_plot.sort_values('phl_habitable')

    plt.figure(figsize=(10, 7))
    sns.violinplot(
        data=df_plot, x='phl_habitable_label', y='pl_eqt',
        palette='viridis', inner='quartile', linewidth=1.5
    )

    # Regiones térmicas del agua líquida (Ricitos de Oro)
    plt.axhspan(250, 350, color='#2ecc71', alpha=0.15, label='Zona Ricitos de Oro (~250K - 350K)')
    plt.axhline(273.15, color='#3498db', linestyle='--', alpha=0.6, label='Punto de Congelación (0°C)')
    plt.axhline(373.15, color='#e74c3c', linestyle='--', alpha=0.6, label='Punto de Ebullición (100°C)')

    plt.title('Perfil Térmico según Clasificación de Habitabilidad PHL', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Categoría de Habitabilidad', fontsize=12)
    plt.ylabel('Temperatura de Equilibrio (K)', fontsize=12)
    plt.ylim(0, 1500) # Evita que los planetas extremos aplasten el violín
    plt.legend(loc='upper right', fontsize=10)
    sns.despine()

    ruta_salida = f'{DIR_FIGURAS}/eda_03_habitabilidad.png'
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" ✓ Gráfico guardado en: {ruta_salida}")

if __name__ == "__main__":
    generar_violin_habitabilidad()