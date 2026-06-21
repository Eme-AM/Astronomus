import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Cargando el Data Lake V3 (MICE)...")
v3 = pd.read_csv("data/processed/V3_data_lake_consolidado.csv", low_memory=False)

print("Procesando datos numéricos para la matriz...")
# Seleccionamos explícitamente las características astrofísicas más relevantes
# (Excluimos identificadores, nombres o variables con muchos nulos residuales)
columnas_fisicas = [
    'pl_orbper', 'pl_orbeccen', 'pl_bmasse', 'pl_rade', 
    'pl_dens', 'pl_eqt', 'pl_insol', 
    'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_age'
]

# Filtramos el dataset para quedarnos solo con esas columnas
df_corr = v3[columnas_fisicas]

# Calculamos la matriz de correlación (Usamos Spearman porque los datos astrofísicos 
# suelen tener valores extremos que rompen la correlación lineal de Pearson)
matriz_correlacion = df_corr.corr(method='spearman')

print("Generando el Heatmap de Correlación...")
# Configuramos el tamaño del lienzo
plt.figure(figsize=(12, 10))
sns.set_theme(style="white")

# Generamos una máscara para el triángulo superior (para no repetir datos espejo)
mascara = np.triu(np.ones_like(matriz_correlacion, dtype=bool))

# Generamos el mapa de calor
heatmap = sns.heatmap(matriz_correlacion, 
                      mask=mascara, 
                      cmap='coolwarm', 
                      vmax=1, vmin=-1, # Fijamos los límites de los colores
                      center=0, 
                      annot=True,      # Agregamos los números adentro de los cuadritos
                      fmt=".2f",       # Formateamos a 2 decimales
                      square=True, 
                      linewidths=.5, 
                      cbar_kws={"shrink": .8})

# Ajustes estéticos de títulos y etiquetas
plt.title('Matriz de Correlación de Atributos Exoplanetarios\n(Dataset V3 - Imputación MICE)', 
          fontsize=16, fontweight='bold', pad=20)

# Rotamos las etiquetas para que sean legibles
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()
plt.savefig('eda_01_matriz_correlacion.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ ¡Gráfico generado exitosamente y guardado como 'eda_01_matriz_correlacion.png'!")