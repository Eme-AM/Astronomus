# reports/plot_pca_projection.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from backend.src.visualization.config_plots import configurar_estilo, cargar_gold, guardar_figura

# Estilo aplicado a nivel de módulo — consistente con el resto de la suite
configurar_estilo()

MAPA_CLASES = {
    0: '0 - Inhóspito',
    1: '1 - Exótico',
    2: '2 - Tierra 2.0 (Grial)',
}
COLORES_CLASES = {
    '0 - Inhóspito':          '#e0e0e0',
    '1 - Exótico':            '#3498db',
    '2 - Tierra 2.0 (Grial)': '#f1c40f',
}


def generar_pca() -> None:
    """
    Proyecta el espacio paramétrico de la Capa Oro a 2D mediante PCA.
    Colorea cada punto según su clase target (Inhóspito / Exótico / Grial).
    """
    X_train = cargar_gold("X_train.csv")
    y_train = cargar_gold("y_train.csv")

    n_features = X_train.shape[1]
    print(f"Calculando Componentes Principales ({n_features}D → 2D)...")

    pca         = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(X_train)
    var         = pca.explained_variance_ratio_

    # FIX AUDITORÍA: reset_index evita NaN por misalignment entre
    # el índice no-consecutivo de y_train (producto del train_test_split)
    # y el índice 0-based del DataFrame de componentes PCA.
    y_train = y_train.reset_index(drop=True)
    y_train['target_class'] = y_train['target_class'].astype(int)

    df_pca          = pd.DataFrame(componentes, columns=['Componente 1', 'Componente 2'])
    df_pca['Clase'] = y_train['target_class'].map(MAPA_CLASES)

    # Validación: detectar valores de clase no contemplados en el mapa
    nans_post_map = df_pca['Clase'].isna().sum()
    if nans_post_map > 0:
        clases_huerfanas = y_train.loc[df_pca['Clase'].isna(), 'target_class'].unique()
        raise ValueError(
            f"El mapa de clases no cubre {nans_post_map} filas. "
            f"Valores sin mapear: {clases_huerfanas}. "
            f"Actualizá MAPA_CLASES en {__file__}."
        )

    # Las clases raras (Grial) se dibujan al final para no quedar tapadas
    df_pca = df_pca.sort_values('Clase')

    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=df_pca,
        x='Componente 1', y='Componente 2',
        hue='Clase', palette=COLORES_CLASES,
        hue_order=list(MAPA_CLASES.values()),
        s=65, alpha=0.85, edgecolor='black', linewidth=0.3,
    )

    plt.title(
        f'Proyección 2D del Espacio Paramétrico (PCA)\n'
        f'Varianza retenida: {sum(var)*100:.1f}%',
        fontweight='bold', pad=20,
    )
    plt.xlabel(f'Componente Principal 1 ({var[0]*100:.1f}%)')
    plt.ylabel(f'Componente Principal 2 ({var[1]*100:.1f}%)')
    plt.legend(title='Target Class', bbox_to_anchor=(1.02, 1), loc='upper left')
    sns.despine()  # ← consistente con toda la suite

    guardar_figura('eda_04_pca_projection.png')


if __name__ == "__main__":
    generar_pca()