import matplotlib.pyplot as plt
import seaborn as sns
from backend.src.visualization.config_plots import configurar_estilo, cargar_silver, guardar_figura

def generar_auditoria_orbital():
    print("Cargando Capa Plata para análisis de Mecánica Orbital...")
    configurar_estilo()

    df = cargar_silver("data_lake_consolidado.csv")
    
    # Filtro de seguridad: excluir períodos anómalos <= 0 para la escala logarítmica
    df_plot = df[df['pl_orbper'] > 0].copy()

    plt.figure(figsize=(12, 8))
    
    # Se añade la temperatura como color para aportar contexto astrofísico extra
    sns.scatterplot(
        data=df_plot, x='pl_orbper', y='pl_orbeccen', hue='pl_eqt',
        palette='magma', s=35, alpha=0.7, edgecolor='black', linewidth=0.2
    )

    plt.xscale('log')
    plt.ylim(-0.05, 1.05) 

    # Corrección de renderizado logarítmico: Inicio en 0.1 en lugar de 0
    plt.axvspan(0.1, 5, ymin=0.5, ymax=1, color='red', alpha=0.15)
    plt.text(0.7, 0.8, 'Zona de Circularización\n(Debería estar vacía)', 
             color='red', fontsize=11, fontweight='bold', ha='center')

    plt.title('Auditoría de Mecánica Orbital: Período vs. Excentricidad\n(Verificación de Leyes de Kepler)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Período Orbital (Días) [Escala Log]', fontsize=13)
    plt.ylabel('Excentricidad de la Órbita', fontsize=13)
    
    guardar_figura('eda_05_mecanica_orbital.png')
    print(" ✓ Gráfico guardado en: reports/figures/eda_05_mecanica_orbital.png")

if __name__ == "__main__":
    generar_auditoria_orbital()