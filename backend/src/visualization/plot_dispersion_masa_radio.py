import matplotlib.pyplot as plt
import seaborn as sns
from config_plots import configurar_estilo, cargar_silver, guardar_figura

def generar_scatter_astrofisico():
    print("Cargando Capa Plata para Scatter Plot...")
    configurar_estilo()

    # Delegamos la carga y validación a config_plots
    v4 = cargar_silver("data_lake_consolidado.csv")
    
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
    plt.xlabel('Masa (Masas Terrestres) [Escala Log]', fontsize=13)
    plt.ylabel('Radio (Radios Terrestres) [Escala Log]', fontsize=13)

    # Delegamos el guardado (despine, savefig, close)
    guardar_figura('eda_02_dispersion_masa_radio.png')
    print(" ✓ Gráfico guardado en: reports/figures/eda_02_dispersion_masa_radio.png")

if __name__ == "__main__":
    generar_scatter_astrofisico()