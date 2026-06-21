import matplotlib.pyplot as plt
import seaborn as sns
from config_plots import configurar_estilo, cargar_gold, guardar_figura

def generar_perfiles_clases():
    print("Cargando dataset pre-escalado con etiquetas de clase...")
    configurar_estilo()
    
    df = cargar_gold("dataset_preparado_ml.csv")
    df['target_class'] = df['target_class'].astype(int)
    df['Clase'] = df['target_class'].map({0: '0 - Inhóspito', 1: '1 - Tierra 2.0'})

    variables = {
        'pl_dens': ('Densidad Planetaria', 'g/cm³', (0, 15)),
        'pl_eqt': ('Temp. de Equilibrio', 'K', (0, 1500)),
        'pl_rade': ('Radio Planetario', 'Radios Terrestres', (0, 5)),
        'pl_insol': ('Insolación Estelar', 'Flujo Terrestre', (0, 5))
    }

    colores = {'0 - Inhóspito': '#95a5a6', '1 - Tierra 2.0': '#f1c40f'}
    orden = ['0 - Inhóspito', '1 - Tierra 2.0']

    print("Generando gráficos individuales de perfilado...")
    for col, (titulo, unidad, limites) in variables.items():
        plt.figure(figsize=(8, 6))
        
        sns.boxplot(data=df, x='Clase', y=col, order=orden, palette=colores, showfliers=False, width=0.5)
        sns.stripplot(data=df, x='Clase', y=col, order=orden, color='black', alpha=0.4, size=5, jitter=True)

        plt.title(f'Perfilado Físico: {titulo}', fontweight='bold', pad=15)
        plt.ylabel(unidad)
        plt.xlabel('Clase Target')
        plt.ylim(limites)

        guardar_figura(f'eda_06_perfil_{col}.png')

if __name__ == "__main__":
    generar_perfiles_clases()