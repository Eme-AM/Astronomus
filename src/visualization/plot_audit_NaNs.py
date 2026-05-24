import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

DIR_FIGURAS = "reports/figures"
os.makedirs(DIR_FIGURAS, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def generar_graficos_auditoria():
    print("Generando auditoría visual de datos faltantes y linaje...")
    
    # 1. Gráfico de Barras Horizontales
    data_faltantes = {
        'eu_temp': 73.37, 'eu_mass_earth': 72.75, 'phl_esi': 56.37, 'phl_habitable': 52.82,
        'eu_radius_earth': 44.39, 'pl_insol': 29.91, 'pl_eqt': 25.50, 'st_age': 22.70,
        'pl_orbeccen': 16.74, 'st_met': 10.21, 'pl_orbper': 5.42, 'st_rad': 5.07,
        'st_teff': 4.68, 'pl_dens': 2.23, 'pl_rade': 0.80, 'pl_bmasse': 0.49, 'st_mass': 0.14
    }
    df_faltantes = pd.DataFrame(list(data_faltantes.items()), columns=['Variable', 'Porcentaje'])
    df_faltantes['Color'] = df_faltantes['Variable'].apply(
        lambda x: '#95a5a6' if x.startswith('eu_') else ('#f39c12' if x.startswith('phl_') else '#3498db')
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(data=df_faltantes, x='Porcentaje', y='Variable', palette=df_faltantes['Color'].tolist(), hue='Variable', legend=False)
    plt.title('Porcentaje de Datos Faltantes (Pre-Limpieza)', fontweight='bold', fontsize=14, pad=15)
    plt.xlabel('Porcentaje Faltante (%)', fontsize=12)
    plt.ylabel('')
    plt.xlim(0, 80)
    
    leyenda_elementos = [
        Patch(facecolor='#95a5a6', label='Apoyo (Cruce Espacial)'),
        Patch(facecolor='#f39c12', label='Targets (Preservadas)'),
        Patch(facecolor='#3498db', label='Features (Imputadas)')
    ]
    plt.legend(handles=leyenda_elementos, loc='lower right', fontsize=11)
    sns.despine()
    
    ruta_barras = f'{DIR_FIGURAS}/auditoria_barras.png'
    plt.tight_layout()
    plt.savefig(ruta_barras, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" ✓ Gráfico guardado en: {ruta_barras}")

    # 2. Gráfico de Dona (Linaje de Rescate Corregido de la Capa Plata V4)
    data_linaje = {'ML (MICE)': 7089, 'Leyes Físicas': 576, 'Cruce de Catálogos': 113}
    
    plt.figure(figsize=(8, 8))
    plt.pie(data_linaje.values(), labels=data_linaje.keys(), autopct='%1.1f%%', 
            startangle=140, colors=['#9b59b6', '#f1c40f', '#2ecc71'],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}, textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    centro_blanco = plt.Circle((0,0), 0.60, fc='white')
    plt.gcf().gca().add_artist(centro_blanco)
    
    plt.title(f'Linaje de Rescate de Datos\n(Total: {sum(data_linaje.values()):,} valores recuperados)'.replace(',', '.'), 
              fontweight='bold', fontsize=14, pad=20)
    
    ruta_dona = f'{DIR_FIGURAS}/auditoria_linaje.png'
    plt.tight_layout()
    plt.savefig(ruta_dona, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" ✓ Gráfico guardado en: {ruta_dona}")

if __name__ == "__main__":
    generar_graficos_auditoria()