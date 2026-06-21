import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config_plots import configurar_estilo, cargar_silver, guardar_figura

def generar_heatmap():
    print("Cargando Capa Plata para análisis de correlación...")
    configurar_estilo()

    v4 = cargar_silver("data_lake_consolidado.csv")
    
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

    guardar_figura('eda_01_heatmap_correlacion.png')

if __name__ == "__main__":
    generar_heatmap()