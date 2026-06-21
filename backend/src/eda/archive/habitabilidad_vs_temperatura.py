import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Cargando el Data Lake V3 (MICE)...")
v3 = pd.read_csv("data/processed/V3_data_lake_consolidado.csv", low_memory=False)

# Filtramos los nulos de la columna objetivo (los que no han sido evaluados por PHL)
df_plot = v3.dropna(subset=['phl_habitable', 'pl_eqt']).copy()

# Opcional: Convertimos la categoría numérica a texto para que el gráfico sea más claro
# (Asumiendo la nomenclatura estándar de PHL donde 0=No Habitable, 1=Conservador, 2=Optimista)
mapeo_habitabilidad = {
    0.0: '0 - No Habitable',
    1.0: '1 - Habitable (Conservador)',
    2.0: '2 - Habitable (Optimista)'
}
# Si tu columna tiene otros valores, el .map() los ignorará o dejará igual, 
# pero es una buena práctica intentarlo.
if df_plot['phl_habitable'].isin([0, 1, 2]).any():
    df_plot['phl_habitable_label'] = df_plot['phl_habitable'].map(mapeo_habitabilidad).fillna(df_plot['phl_habitable'].astype(str))
else:
    df_plot['phl_habitable_label'] = df_plot['phl_habitable'].astype(str)

# Ordenamos las etiquetas para que aparezcan lógicamente en el eje X
df_plot = df_plot.sort_values('phl_habitable')

print("Generando el Violin Plot de Habitabilidad...")
plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid")

# Dibujamos el gráfico de violín
sns.violinplot(
    data=df_plot, 
    x='phl_habitable_label', 
    y='pl_eqt', 
    palette='viridis', 
    inner='quartile', # Dibuja las líneas de los cuartiles dentro del violín
    linewidth=1.5
)

# --- AGREGANDO LA ZONA DE HABITABILIDAD (ZONA VERDE) ---
# Sombreado entre 250K y 350K (Rango aproximado para agua líquida)
plt.axhspan(250, 350, color='green', alpha=0.15, label='Zona Ricitos de Oro (~250K - 350K)')
plt.axhline(273.15, color='blue', linestyle='--', alpha=0.5, label='Punto de Congelación (0°C)')
plt.axhline(373.15, color='red', linestyle='--', alpha=0.5, label='Punto de Ebullición (100°C)')

# --- FORMATEO Y ESTÉTICA ---
plt.title('Distribución de Temperatura según Clasificación de Habitabilidad (PHL)\n(Dataset V3)', 
          fontsize=15, fontweight='bold', pad=20)
plt.xlabel('Clasificación de Habitabilidad', fontsize=12)
plt.ylabel('Temperatura de Equilibrio (K)', fontsize=12)

# Ajustamos el límite Y para que los gigantes de fuego no aplasten la visualización de la zona verde
plt.ylim(0, 1500) 

# Agregamos la leyenda para la zona sombreada y las líneas
plt.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('eda_03_habitabilidad.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ ¡Gráfico generado exitosamente y guardado como 'eda_03_habitabilidad.png'!")