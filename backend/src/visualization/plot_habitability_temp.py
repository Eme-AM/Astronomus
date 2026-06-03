# src/visualization/plot_habitabilidad_temperatura.py

import matplotlib.pyplot as plt
import seaborn as sns
from backend.src.visualization.config_plots import configurar_estilo, cargar_silver, guardar_figura

def generar_violin_habitabilidad():
    print("Cargando datos para análisis de habitabilidad...")
    configurar_estilo()
    
    df = cargar_silver()
    df_plot = df.dropna(subset=['phl_habitable', 'pl_eqt']).copy()

    # FIX AUDITORÍA: Claves enteras y cast explícito
    df_plot['phl_habitable'] = df_plot['phl_habitable'].astype(int)
    mapeo_habitabilidad = {0: '0 - Inhóspito', 1: '1 - Conservador', 2: '2 - Optimista'}
    df_plot['phl_habitable_label'] = df_plot['phl_habitable'].map(mapeo_habitabilidad)
    df_plot = df_plot.sort_values('phl_habitable')

    plt.figure(figsize=(10, 7))
    sns.violinplot(data=df_plot, x='phl_habitable_label', y='pl_eqt', palette='viridis', inner='quartile', linewidth=1.5)

    plt.axhspan(250, 350, color='#2ecc71', alpha=0.15, label='Zona Ricitos de Oro (~250K - 350K)')
    plt.axhline(273.15, color='#3498db', linestyle='--', alpha=0.6, label='Congelación (0°C)')
    plt.axhline(373.15, color='#e74c3c', linestyle='--', alpha=0.6, label='Ebullición (100°C)')

    plt.title('Perfil Térmico según Clasificación PHL', fontweight='bold', pad=20)
    plt.xlabel('Categoría PHL')
    plt.ylabel('Temperatura de Equilibrio (K)')
    plt.ylim(0, 1500)
    plt.legend(loc='upper right')

    guardar_figura('eda_03_habitabilidad.png')

if __name__ == "__main__":
    generar_violin_habitabilidad()