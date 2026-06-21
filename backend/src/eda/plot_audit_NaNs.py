import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from config_plots import configurar_estilo, guardar_figura, DIR_ARCHIVE, DIR_FIGURAS

def obtener_datos_faltantes() -> dict:
    """
    Calcula los porcentajes de nulos dinámicamente leyendo la V1 del Data Lake.
    Si el archivo no existe, utiliza los valores hardcodeados como respaldo.
    """

    ruta_v1 = DIR_ARCHIVE / "V1_data_lake_consolidado.csv"
    
    # Las columnas que queremos auditar en el gráfico
    columnas_objetivo = [
        'eu_temp', 'eu_mass_earth', 'phl_esi', 'phl_habitable',
        'eu_radius_earth', 'pl_insol', 'pl_eqt', 'st_age',
        'pl_orbeccen', 'st_met', 'pl_orbper', 'st_rad',
        'st_teff', 'pl_dens', 'pl_rade', 'pl_bmasse', 'st_mass'
    ]

    if ruta_v1.exists():
        print(f"Calculando porcentajes de nulos dinámicamente desde {ruta_v1}...")
        df_v1 = pd.read_csv(ruta_v1, low_memory=False)
        
        missing_pct = (df_v1.isna().sum() / len(df_v1)) * 100

        # Filtramos solo las columnas que tienen faltantes y las ordenamos de mayor a menor
        porcentajes = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        return porcentajes.to_dict()
    else:
        # Bloque hardcoded con datos veridicos obtenidos en una prueba anterior
        print(f"Archivo {ruta_v1} no encontrado. Usando valores de respaldo (Fallback)...")
        return {
            'eu_temp': 73.37, 'eu_mass_earth': 72.75, 'phl_esi': 56.37, 'phl_habitable': 52.82,
            'eu_radius_earth': 44.39, 'pl_insol': 29.91, 'pl_eqt': 25.50, 'st_age': 22.70,
            'pl_orbeccen': 16.74, 'st_met': 10.21, 'pl_orbper': 5.42, 'st_rad': 5.07,
            'st_teff': 4.68, 'pl_dens': 2.23, 'pl_rade': 0.80, 'pl_bmasse': 0.49, 'st_mass': 0.14
        }

def generar_graficos_auditoria():
    print("Generando auditoría visual de datos faltantes y linaje...")
    configurar_estilo()
    
    # ==========================================
    # 1. Gráfico de Barras Horizontales (Dinámico)
    # ==========================================
    data_faltantes = obtener_datos_faltantes()
    df_faltantes = pd.DataFrame(list(data_faltantes.items()), columns=['Variable', 'Porcentaje'])
    
    # Asignación de colores semánticos
    df_faltantes['Color'] = df_faltantes['Variable'].apply(
        lambda x: '#95a5a6' if x.startswith('eu_') else ('#f39c12' if x.startswith('phl_') else '#3498db')
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=df_faltantes, x='Porcentaje', y='Variable', 
        palette=df_faltantes['Color'].tolist(), hue='Variable', legend=False
    )
    
    plt.title('Porcentaje de Datos Faltantes (Pre-Limpieza)', fontweight='bold', fontsize=14, pad=15)
    plt.xlabel('Porcentaje Faltante (%)', fontsize=12)
    plt.ylabel('')
    
    # Ajuste dinámico del límite X (por si un porcentaje supera el 80% en el futuro)
    limite_x = max(df_faltantes['Porcentaje'].max() + 5, 80)
    plt.xlim(0, limite_x)
    
    leyenda_elementos = [
        Patch(facecolor='#95a5a6', label='Apoyo (Cruce Espacial)'),
        Patch(facecolor='#f39c12', label='Targets (Preservadas)'),
        Patch(facecolor='#3498db', label='Features (Imputadas)')
    ]
    plt.legend(handles=leyenda_elementos, loc='lower right', fontsize=11)
    sns.despine()
    
    plt.tight_layout()
    guardar_figura('auditoria_barras.png')

    # ==========================================
    # 2. Gráfico de Dona (Linaje de Rescate)
    # ==========================================
    # TODO: data_linaje está hardcodeado con valores de una ejecución pasada del pipeline.
    # Si el dataset crece, el gráfico de dona refleja números incorrectos. Idealmente estos
    # totales deberían leerse de un artefacto generado por processing.py (ej. un JSON de métricas).
    data_linaje = {'ML (MICE)': 7089, 'Leyes Físicas': 576, 'Cruce de Catálogos': 113}
    
    plt.figure(figsize=(8, 8))
    plt.pie(data_linaje.values(), labels=data_linaje.keys(), autopct='%1.1f%%', 
            startangle=140, colors=['#9b59b6', '#f1c40f', '#2ecc71'],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}, textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    centro_blanco = plt.Circle((0,0), 0.60, fc='white')
    plt.gcf().gca().add_artist(centro_blanco)
    
    plt.title(f'Linaje de Rescate de Datos\n(Total: {sum(data_linaje.values()):,} valores recuperados)'.replace(',', '.'), 
              fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    guardar_figura('auditoria_linaje.png')

if __name__ == "__main__":
    generar_graficos_auditoria()