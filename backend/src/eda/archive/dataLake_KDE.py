import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Cargando los Data Lakes...")
# Cargamos las tres versiones
v1 = pd.read_csv("data/archive/V1_data_lake_consolidado.csv", low_memory=False)
v2 = pd.read_csv("data/archive/V2_data_lake_consolidado.csv", low_memory=False)
v3 = pd.read_csv("data/archive/V3_data_lake_consolidado.csv", low_memory=False)
v4 = pd.read_csv("data/archive/V4_data_lake_consolidado.csv", low_memory=False)

# Usamos un diccionario para empaquetar la variable, su título y sus límites X
variables_config = {
    'pl_insol': ('Insolación Planetaria', -200, 2000),
    'pl_eqt': ('Temperatura de Equilibrio', -500, 3500),
    'st_age': ('Edad Estelar', -2, 15)
}

sns.set_theme(style="whitegrid")
print("\nGenerando los gráficos individuales...")

# Iteramos sobre cada variable para crear y guardar su propio gráfico
for col, (titulo, x_min, x_max) in variables_config.items():
    
    # Abrimos un lienzo individual
    plt.figure(figsize=(9, 6))
    
    # 1. V1 (Azul sólido)
    sns.kdeplot(data=v1[col].dropna(), label='V1: Original (Ignorando NaNs)', 
                color='blue', linewidth=2)
    
    # 2. V2 (Rojo punteado) - El "crimen estadístico"
    sns.kdeplot(data=v2[col], label='V2: Fuerza Bruta (Media)', 
                color='red', linestyle='--', linewidth=2)
    
    # 3. V3 (Verde dash-dot) - La magia de MICE
    sns.kdeplot(data=v3[col], label='V3: Machine Learning (MICE)', 
                color='green', linestyle='-.', linewidth=2)

    # 4. V4 (Naranja sólido) - La versión más reciente, geometría 3D y Closest Wins
    sns.kdeplot(data=v4[col], label='V4: Versión Actualizada (MICE)', 
                color='orange', linewidth=2)

    # Formato individual del gráfico
    plt.title(f'{titulo}\n({col})', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Densidad (Frecuencia)', fontsize=12)
    plt.xlabel('Valor', fontsize=12)
    
    # Ajustamos los límites del eje X
    plt.xlim(x_min, x_max)
    
    # Agregamos la leyenda a cada gráfico individual
    plt.legend(fontsize=11)
    
    # Guardamos la imagen con un nombre dinámico
    nombre_archivo = f'kde_{col}.png'
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    
    # MUY IMPORTANTE: Cerramos la figura actual para no superponer los datos del siguiente ciclo
    plt.close() 
    
    print(f" ✓ Generado: {nombre_archivo}")

print("\n¡Todos los gráficos fueron guardados exitosamente!")