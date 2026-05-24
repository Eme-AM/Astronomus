import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

DIR_GOLD = "data/gold"
DIR_FIGURAS = "reports/figures"
os.makedirs(DIR_FIGURAS, exist_ok=True)

sns.set_theme(style="white", context="paper", font_scale=1.2)

def generar_pca():
    print("Cargando tensores de la Capa Oro para reducción de dimensionalidad...")

    ruta_datos = f"{DIR_GOLD}/X_train.csv"
    
    if not os.path.exists(ruta_datos):
        print(f"Error: No se encontró el dataset en {ruta_datos}.")
        print("  Por favor, ejecutá primero el pipeline de datos (ingestion.py, processing.py y preparation.py) para generar este archivo localmente.")
        return

    X_train = pd.read_csv(f"{DIR_GOLD}/X_train.csv")
    y_train = pd.read_csv(f"{DIR_GOLD}/y_train.csv")

    print("Calculando Componentes Principales (12D -> 2D)...")
    pca = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(X_train)
    
    df_pca = pd.DataFrame(data=componentes, columns=['Componente 1', 'Componente 2'])
    y_train['target_class'] = y_train['target_class'].astype(int)
    df_pca['Clase'] = y_train['target_class'].map({
        0: '0 - Inhóspito', 
        1: '1 - Exótico', 
        2: '2 - Tierra 2.0 (Grial)'
    })

    # Orden para el pintado (los últimos se dibujan arriba de todo)
    df_pca = df_pca.sort_values('Clase')

    plt.figure(figsize=(11, 8))
    colores = {'0 - Inhóspito': '#e0e0e0', '1 - Exótico': '#3498db', '2 - Tierra 2.0 (Grial)': '#f1c40f'}
    orden_clases = ['0 - Inhóspito', '1 - Exótico', '2 - Tierra 2.0 (Grial)']

    sns.scatterplot(
        data=df_pca, x='Componente 1', y='Componente 2', hue='Clase',
        palette=colores, hue_order=orden_clases, s=65, 
        alpha=0.85, edgecolor='black', linewidth=0.3
    )

    varianza_explicada = sum(pca.explained_variance_ratio_) * 100
    plt.title(f'Proyección 2D del Espacio Paramétrico (PCA)\nVarianza retenida: {varianza_explicada:.1f}%', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    
    # Leyenda anclada fuera del gráfico
    plt.legend(title='Target Class', bbox_to_anchor=(1.02, 1), loc='upper left')
    sns.despine()

    ruta_salida = f'{DIR_FIGURAS}/eda_04_pca_projection.png'
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" ✓ Gráfico guardado en: {ruta_salida}")

if __name__ == "__main__":
    generar_pca()