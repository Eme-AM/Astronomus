import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------------------------------------------------------
# DATOS DE LA CONSOLA (Actualizados)
# ---------------------------------------------------------
# Estos números se mantienen iguales porque es la foto ANTES de imputar
data_faltantes = {
    'eu_temp': 73.37, 'eu_mass_earth': 72.75, 'phl_esi': 56.37, 'phl_habitable': 52.82,
    'eu_radius_earth': 44.39, 'pl_insol': 29.91, 'pl_eqt': 25.50, 'st_age': 22.70,
    'pl_orbeccen': 16.74, 'st_met': 10.21, 'pl_orbper': 5.42, 'st_rad': 5.07,
    'st_teff': 4.68, 'pl_dens': 2.23, 'pl_rade': 0.80, 'pl_bmasse': 0.49, 'st_mass': 0.14
}
df_faltantes = pd.DataFrame(list(data_faltantes.items()), columns=['Variable', 'Porcentaje'])

# ACTUALIZADO: Los números reales del reporte de linaje honesto
data_linaje = {
    'ML (MICE)': 7157,
    'Leyes Físicas': 568,
    'Cruce de Catálogos': 53  # Este es el valor real corregido
}

# ---------------------------------------------------------
# GRÁFICO 1: BARRAS HORIZONTALES (ESTADO Y DESTINO)
# ---------------------------------------------------------
# Asignamos colores según el rol real de la variable en tu pipeline
def asignar_color(var):
    if var.startswith('eu_'):
        return '#95a5a6'  # Gris: Variables de apoyo (Cruce de catálogos)
    elif var.startswith('phl_'):
        return '#f39c12'  # Naranja: Targets/Labels preservados
    else:
        return '#3498db'  # Azul: Variables imputadas y limpiadas (MICE)

df_faltantes['Color'] = df_faltantes['Variable'].apply(asignar_color)

plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid")

# Dibujamos las barras
ax_bar = sns.barplot(data=df_faltantes, x='Porcentaje', y='Variable', 
                     palette=df_faltantes['Color'].tolist(), hue='Variable', legend=False)

# Títulos y etiquetas
plt.title('Porcentaje de Datos Faltantes (Pre-Limpieza)', fontweight='bold', fontsize=14, pad=15)
plt.xlabel('Porcentaje Faltante (%)', fontsize=12)
plt.ylabel('')
plt.xlim(0, 80)

# Creamos una leyenda manual para explicar los colores
from matplotlib.patches import Patch
leyenda_elementos = [
    Patch(facecolor='#95a5a6', label='Apoyo (Usadas para cruce)'),
    Patch(facecolor='#f39c12', label='Targets (Preservadas)'),
    Patch(facecolor='#3498db', label='Features (Imputadas)')
]
plt.legend(handles=leyenda_elementos, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('auditoria_barras.png', dpi=300, bbox_inches='tight')
plt.close() # Cerramos el lienzo para no superponer con el siguiente gráfico
print("✓ Generado: auditoria_barras.png")

# ---------------------------------------------------------
# GRÁFICO 2: DONA DE LINAJE DE IMPUTACIÓN
# ---------------------------------------------------------
plt.figure(figsize=(8, 8))

# Extraemos valores y etiquetas
valores = list(data_linaje.values())
etiquetas = list(data_linaje.keys())
colores_dona = ['#9b59b6', '#f1c40f', '#2ecc71'] # Morado (MICE), Amarillo (Física), Verde (Cruce) - Reordenados

# Creamos el gráfico de pastel
plt.pie(valores, labels=etiquetas, autopct='%1.1f%%', 
        startangle=140, colors=colores_dona,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}, 
        textprops={'fontsize': 12, 'fontweight': 'bold'})

# Agregamos el círculo blanco en el centro para hacerlo Dona
centro_blanco = plt.Circle((0,0), 0.60, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centro_blanco)

# ACTUALIZADO: El título refleja el nuevo total real
total_rescatado = sum(valores)
plt.title(f'Linaje de Rescate de Datos\n(Total: {total_rescatado:,} valores recuperados)'.replace(',', '.'), 
          fontweight='bold', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig('auditoria_linaje.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Generado: auditoria_linaje.png")