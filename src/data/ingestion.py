import pandas as pd
import os
import urllib.request
import urllib.error
import shutil
import glob
import kagglehub

# Creamos el directorio data/ cruda si no existe para mantener el orden
os.makedirs("data/raw", exist_ok=True)

def mostrar_estructura_fuente(nombre_fuente, df, filas_head=5):
    """
    Muestra columnas y head completo de una fuente sin truncar.
    """
    if df is None:
        print(f"\n[{nombre_fuente}] Sin datos disponibles para inspección.")
        return

    print(f"\n{'=' * 80}")
    print(f"[{nombre_fuente}] Estructura de datos")
    print(f"Total de columnas: {len(df.columns)}")
    print("Columnas:")
    print(df.columns.tolist())

    print(f"\nHead ({filas_head} filas) con todas las columnas:")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 300,
        "display.max_colwidth", 120,
    ):
        print(df.head(filas_head))

def descargar_catalogo(nombre_fuente, archivo_salida, url=None, kaggle_dataset=None):
    """
    Función de ingesta unificada. Descarga eficiente directo a disco duro 
    (vía URL o Kaggle) antes de cargar en Pandas para optimizar memoria.
    """
    print(f"\n Iniciando ingesta: {nombre_fuente}")
    
    # 1. Caché Local (Escudo defensivo)
    if os.path.exists(archivo_salida):
        print(f"   ✓ Caché local encontrado. Leyendo {archivo_salida}...")
        return pd.read_csv(archivo_salida, low_memory=False)
    
    print(f"  Archivo no encontrado en disco. Iniciando descarga...")
    try:
        # 2. Descarga eficiente a disco (Patrón ELT puro)
        if kaggle_dataset:
            # Rescate vía KaggleHub
            path_kaggle = kagglehub.dataset_download(kaggle_dataset)
            archivos_csv = glob.glob(os.path.join(path_kaggle, "*.csv"))
            if not archivos_csv:
                raise FileNotFoundError("No se encontró ningún archivo CSV en Kaggle.")
            shutil.copy(archivos_csv[0], archivo_salida)
            print("   ✓ Descarga de Kaggle completada exitosamente.")
            
        elif url:
            # Descarga HTTP directa a disco (No colapsa la RAM)
            urllib.request.urlretrieve(url, archivo_salida)
            print("   ✓ Descarga HTTP completada exitosamente.")
            
        else:
            raise ValueError("Se debe proveer una 'url' o un 'kaggle_dataset'.")

        # 3. Lectura a Pandas exclusivamente desde el archivo local
        df = pd.read_csv(archivo_salida, low_memory=False)
        print(f"   ✓ ¡Éxito! Datos ingestados al Data Lake: {len(df)} planetas.")
        return df
        
    except urllib.error.HTTPError as e:
        print(f"   ERROR HTTP {e.code}: El servidor de {nombre_fuente} rechazó la conexión.")
    except ImportError:
        print("   ERROR: Falta instalar kagglehub (pip install kagglehub)")
    except Exception as e:
        print(f"   ERROR INESPERADO al procesar {nombre_fuente}: {str(e)}")
    
    return None

# =====================================================================
# 1. FUENTE PRINCIPAL: NASA Exoplanet Archive (70 MB RAW DATA)
# =====================================================================
url_nasa = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv"
df_nasa = descargar_catalogo(
    nombre_fuente="NASA", 
    archivo_salida="data/raw/nasa_exoplanets.csv", 
    url=url_nasa
)

# =====================================================================
# 2. FUENTE SECUNDARIA: The Extrasolar Planets Encyclopaedia (Europa)
# =====================================================================
url_eu = "http://exoplanet.eu/catalog/csv"
df_eu = descargar_catalogo(
    nombre_fuente="Exoplanet.eu", 
    archivo_salida="data/raw/eu_exoplanets.csv", 
    url=url_eu
)

# =====================================================================
# 3. FUENTE TERCIARIA: PHL - Habitable Exoplanets Catalog
# =====================================================================
df_phl = descargar_catalogo(
    nombre_fuente="PHL (Arecibo)", 
    archivo_salida="data/raw/phl_exoplanets.csv", 
    kaggle_dataset="chandrimad31/phl-exoplanet-catalog"
)

# =====================================================================
# REPORTE DE INGESTA
# =====================================================================
print("\n--- REPORTE DE DATA LAKE ---")
if df_nasa is not None: print(f"NASA:           {df_nasa.shape[0]} filas, {df_nasa.shape[1]} columnas")
if df_eu is not None:   print(f"Exoplanet.eu:   {df_eu.shape[0]} filas, {df_eu.shape[1]} columnas")
if df_phl is not None:  print(f"PHL (Arecibo):  {df_phl.shape[0]} filas, {df_phl.shape[1]} columnas")

# =====================================================================
# INSPECCIÓN DETALLADA POR FUENTE (HEAD + TODAS LAS COLUMNAS)
# =====================================================================
# mostrar_estructura_fuente("NASA", df_nasa)
# mostrar_estructura_fuente("Exoplanet.eu", df_eu)
# mostrar_estructura_fuente("PHL (Arecibo)", df_phl)