import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Cargando el Data Lake V3 (MICE)...")
v3 = pd.read_csv("data/processed/V3_data_lake_consolidado.csv", low_memory=False)

# Filtramos filas donde masa o radio sean nulos o menores a 0 
# (La escala logarítmica se rompe con valores en cero o negativos)
df_plot = v3[(v3['pl_bmasse'] > 0) & (v3['pl_rade'] > 0)].copy()

print("Generando el Scatter Plot de Masa vs. Radio...")
# Configuramos el tamaño y estilo
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid", palette="muted")

# Generamos el diagrama de dispersión
# x = Masa, y = Radio, hue = Temperatura (para el color)
scatter = sns.scatterplot(
    data=df_plot, 
    x='pl_bmasse', 
    y='pl_rade', 
    hue='pl_eqt',          # El color representa la temperatura
    palette='Spectral_r',  # Paleta que va de frío (azul) a caliente (rojo)
    s=50,                  # Tamaño de los puntos
    alpha=0.7,             # Transparencia para ver cuando se superponen
    edgecolor='black',
    linewidth=0.3
)

# Convertimos los ejes a escala logarítmica
plt.xscale('log')
plt.yscale('log')

# --- AGREGANDO REFERENCIAS ASTROFÍSICAS CLAVE ---
# 1. La Tierra (Masa=1, Radio=1)
plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
plt.text(1.1, 1.1, 'Tierra', color='gray', fontsize=11, fontweight='bold')

# 2. Júpiter (Masa aprox 318, Radio aprox 11.2)
plt.axvline(x=317.8, color='gray', linestyle='-.', alpha=0.5)
plt.axhline(y=11.2, color='gray', linestyle='-.', alpha=0.5)
plt.text(350, 12, 'Júpiter', color='gray', fontsize=11, fontweight='bold')

# --- FORMATEO Y ESTÉTICA ---
plt.title('Clasificación Física de Exoplanetas: Masa vs. Radio\n(Dataset V3 - Coloreado por Temperatura)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Masa Planetaria (Masas Terrestres) [Escala Log]', fontsize=13)
plt.ylabel('Radio Planetario (Radios Terrestres) [Escala Log]', fontsize=13)

# Ajustamos la leyenda del color (Temperatura)
plt.legend(title='Temperatura (K)', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('eda_02_masa_radio.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ ¡Gráfico generado exitosamente y guardado como 'eda_02_masa_radio.png'!")