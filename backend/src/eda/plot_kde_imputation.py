import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config_plots import configurar_estilo, cargar_silver, guardar_figura, DIR_ARCHIVE

def generar_graficos_kde():
    print("Cargando versiones del Data Lake para análisis KDE...")
    configurar_estilo()

    v1 = pd.read_csv(DIR_ARCHIVE / "V1_data_lake_consolidado.csv", low_memory=False)
    v2 = pd.read_csv(DIR_ARCHIVE / "V2_data_lake_consolidado.csv", low_memory=False)
    v3 = pd.read_csv(DIR_ARCHIVE / "V3_data_lake_consolidado.csv", low_memory=False)
    v4 = cargar_silver("data_lake_consolidado.csv") # Versión actual (V4)

    variables_config = {
        'pl_insol': ('Insolación Planetaria', -200, 2000),
        'pl_eqt': ('Temperatura de Equilibrio (K)', -500, 3500),
        'st_age': ('Edad Estelar (Giga-años)', -2, 15)
    }

    print("Generando curvas de densidad (Kernel Density Estimation)...")
    for col, (titulo, x_min, x_max) in variables_config.items():
        plt.figure(figsize=(9, 6))
        
        # Graficado de las 4 eras del dataset
        sns.kdeplot(data=v1[col].dropna(), label='V1: Original (Ignorando NaNs)', color='#3498db', linewidth=2)
        sns.kdeplot(data=v2[col], label='V2: Fuerza Bruta (Media)', color='#e74c3c', linestyle='--', linewidth=2)
        sns.kdeplot(data=v3[col], label='V3: ML (MICE Base)', color='#2ecc71', linestyle='-.', linewidth=2)
        sns.kdeplot(data=v4[col], label='V4: ML (MICE + Geometría 3D)', color='#2c3e50', linewidth=2.5)

        # Formateo estético de nivel de publicación
        plt.title(titulo, fontsize=14, fontweight='bold', pad=15)
        plt.ylabel('Densidad', fontsize=12)
        plt.xlabel('Valor', fontsize=12)
        plt.xlim(x_min, x_max)
        plt.legend(loc='upper right', fontsize=10)

        guardar_figura(f'kde_imputacion_{col}.png')

if __name__ == "__main__":
    generar_graficos_kde()