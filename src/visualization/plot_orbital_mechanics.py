import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DIR_SILVER = "data/silver"
DIR_FIGURAS = "reports/figures"
os.makedirs(DIR_FIGURAS, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def generar_auditoria_orbital():
    print("Cargando Capa Plata para análisis de Mecánica Orbital...")

    ruta_datos = f"{DIR_SILVER}/data_lake_consolidado.csv"
    
    if not os.path.exists(ruta_datos):
        print(f"Error: No se encontró el dataset en {ruta_datos}.")
        print("  Por favor, ejecutá primero el pipeline de datos (ingestion.py y processing.py) para generar este archivo localmente.")
        return

    df = pd.read_csv(f"{DIR_SILVER}/data_lake_consolidado.csv", low_memory=False)
    
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
    plt.ylabel('Excentricidad de la Órbita (0 = Circular, 1 = Extrema)', fontsize=13)
    
    # Reubicar leyenda térmica
    plt.legend(title='Temp (K)', bbox_to_anchor=(1.02, 1), loc='upper left')
    sns.despine()

    ruta_salida = f'{DIR_FIGURAS}/eda_05_mecanica_orbital.png'
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" ✓ Gráfico guardado en: {ruta_salida}")

if __name__ == "__main__":
    generar_auditoria_orbital()