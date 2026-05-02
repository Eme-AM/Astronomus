import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import urllib.error  # Importamos la librería para capturar errores de red HTTP
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("1. Cargando y limpiando datos (Fast-forward del Módulo 1 y 2)...")

# Patrón de Caché Local
archivo_local = "nasa_exoplanets.csv"

if os.path.exists(archivo_local):
    print("   ✓ Archivo local encontrado. Leyendo desde el disco duro...")
    df = pd.read_csv(archivo_local)
else:
    print("Archivo no encontrado. Conectando con la API de la NASA...")
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+pl_name,pl_rade,pl_bmasse,pl_eqt,pl_orbsmax,"
        "pl_insol,st_teff,st_rad,st_mass,st_met,pl_dens"
        "+from+pscomppars&format=csv"
    )
    
    # Manejo de Excepciones (Try-Catch)
    try:
        df = pd.read_csv(url)
        df.to_csv(archivo_local, index=False)
        print(f"   ✓ Datos descargados y guardados en {archivo_local} para futuros usos.")
        
    except urllib.error.HTTPError as e:
        # Error 502, 400, 404, etc. (El servidor respondió pero con un error)
        raise Exception(
            f"\nERROR DE API: La NASA rechazó la conexión (Código {e.code}).\n"
            "El servidor de exoplanetas puede estar saturado o en mantenimiento.\n"
            "Solución: Esperá un par de minutos y volvé a ejecutar el script."
        )
    except urllib.error.URLError as e:
        # No hay internet o el DNS no resuelve
        raise Exception(
            f"\nERROR DE RED: No se pudo alcanzar el servidor de la NASA.\n"
            f"Verificá tu conexión a internet. Detalles técnicos: {e.reason}"
        )
    except Exception as e:
        # Cualquier otro error (ej: fallo al escribir en el disco)
        raise Exception(f"\nERROR INESPERADO: Falló la ingesta de datos. Detalles: {str(e)}")


# Imputación rápida global
features_criticas = ['pl_rade', 'pl_bmasse', 'pl_dens', 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met']
for col in features_criticas:
    df[col] = df[col].fillna(df[col].median())

print("\n2. Construyendo la Variable Objetivo (El Target)...")
crit_size = df['pl_rade'].between(0.5, 1.8)     
crit_temp = df['pl_eqt'].between(200, 320)      
crit_insol = df['pl_insol'].between(0.2, 1.8)   

df['habitable'] = np.where(crit_size & crit_temp & crit_insol, 1, 0)

print("Distribución de Clases:")
print(df['habitable'].value_counts())
hab_pct = (df['habitable'].sum() / len(df)) * 100
print(f"Porcentaje de planetas habitables: {hab_pct:.2f}%")

print("\n3. Feature Engineering (Creando nuevos sensores)...")
df['pl_surf_gravity'] = df['pl_bmasse'] / (df['pl_rade'] ** 2 + 1e-8)

print("\n4. Selección de Features y División...")
X = df[['pl_rade', 'pl_bmasse', 'pl_dens', 'pl_insol', 'pl_eqt', 'st_teff', 'st_met', 'pl_surf_gravity']]
y = df['habitable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print("\n5. Preparando Artefactos para la Red Neuronal...")
# El Deep Learning necesita datos escalados (entre -1 y 1 idealmente)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

# Guardar matriz de features y labels (los datos limpios)
np.save(ARTIFACTS / "X_clean.npy", X_scaled)
np.save(ARTIFACTS / "y_labels.npy", y)

# Guardar el scaler entrenado y los nombres de las columnas (Vital para FastAPI)
joblib.dump(scaler, ARTIFACTS / "robust_scaler.pkl")
feature_cols = list(X.columns)
joblib.dump(feature_cols, ARTIFACTS / "feature_cols.pkl")

print(f"   ✓ Artefactos serializados y guardados en la carpeta {ARTIFACTS}/")

print("\n6. Entrenando el Baseline (Árbol de Decisión)...")
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n7. Evaluación del Modelo Clásico (La trampa del Accuracy)...")
print(f"Precisión Global (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=["No Habitable", "Habitable"], zero_division=0))

print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
print(f"[{cm[0][0]}  {cm[0][1]}] <- Reales No Habitables")
print(f"[{cm[1][0]}    {cm[1][1]}] <- Reales Habitables")