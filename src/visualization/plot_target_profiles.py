import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DIR_GOLD = "data/gold"
DIR_FIGURAS = "reports/figures"
os.makedirs(DIR_FIGURAS, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def generar_perfiles_clases():
    print("Cargando dataset pre-escalado con etiquetas de clase...")

    ruta_datos = f"{DIR_GOLD}/dataset_preparado_ml.csv"
    
    if not os.path.exists(ruta_datos):
        print(f"Error: No se encontró el dataset en {ruta_datos}.")
        print("  Por favor, ejecutá primero el pipeline de datos (ingestion.py, processing.py y preparation.py) para generar este archivo localmente.")
        return

    df = pd.read_csv(f"{DIR_GOLD}/dataset_preparado_ml.csv", low_memory=False)

    # Forzamos conversión a int para evitar fallos de mapeo si se leyó como float
    df['target_class'] = df['target_class'].astype(int)
    df['Clase'] = df['target_class'].map({
        0: '0 - Inhóspito', 
        1: '1 - Exótico', 
        2: '2 - Tierra 2.0'
    })
    
    variables = {
        'pl_dens': ('Densidad Planetaria', 'g/cm³', (0, 15)),
        'pl_eqt': ('Temp. de Equilibrio', 'K', (0, 1500)),
        'pl_rade': ('Radio Planetario', 'Radios Terrestres', (0, 5)),
        'pl_insol': ('Insolación Estelar', 'Flujo Terrestre', (0, 5))
    }

    colores = {'0 - Inhóspito': '#95a5a6', '1 - Exótico': '#3498db', '2 - Tierra 2.0': '#f1c40f'}
    orden_clases = ['0 - Inhóspito', '1 - Exótico', '2 - Tierra 2.0']

    print("Generando gráficos individuales de perfilado...")
    for col, (titulo, unidad, limites) in variables.items():
        plt.figure(figsize=(8, 6))
        
        # Boxplot sin outliers para marcar la distribución central
        sns.boxplot(
            data=df, x='Clase', y=col, order=orden_clases,
            palette=colores, showfliers=False, width=0.5
        )
        
        # Stripplot superpuesto para visualizar las escasas anomalías (Clase 2)
        sns.stripplot(
            data=df, x='Clase', y=col, order=orden_clases,
            color='black', alpha=0.4, size=5, jitter=True
        )

        plt.title(f'Perfilado Físico: {titulo}', fontweight='bold', fontsize=15, pad=15)
        plt.ylabel(unidad, fontsize=12)
        plt.xlabel('Clase Target', fontsize=12)
        plt.ylim(limites)
        
        sns.despine()

        ruta_salida = f'{DIR_FIGURAS}/eda_06_perfil_{col}.png'
        plt.tight_layout()
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" ✓ Gráfico guardado en: {ruta_salida}")

if __name__ == "__main__":
    generar_perfiles_clases()