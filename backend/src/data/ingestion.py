import os
import urllib.request
import urllib.error
import shutil
import glob
import logging
import pandas as pd
import kagglehub

# 1. Configuración del Logger a nivel de móduloZ
logger = logging.getLogger(__name__)
#==========================
# CONSTANTES GLOBALES
# ==========================
RUTA_RAW = "backend/data/raw"
NOMBRE_CSV_PHL = "phl_exoplanet_catalog_2019.csv"

# Función de descarga con manejo de errores y patrón Caché-Primero
def descargar_catalogo(
    nombre_fuente: str, 
    archivo_salida: str, 
    url: str | None = None, 
    kaggle_dataset: str | None = None
) -> pd.DataFrame | None:
    """
    Descarga un catálogo astronómico desde una URL directa o desde Kaggle.
    Aplica patrón Caché-Primero para evitar descargas redundantes.
    """
    # Patrón Caché-Primero
    if os.path.exists(archivo_salida):
        logger.info("✓ Caché local encontrado. Leyendo %s...", archivo_salida)
        try:
            return pd.read_csv(archivo_salida, low_memory=False)
        except pd.errors.ParserError as e:
            logger.error("El CSV en caché de %s está corrupto: %s", nombre_fuente, e)
            return None

    logger.info("Descargando catálogo de %s...", nombre_fuente)
    os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)

    try:
        # Estrategia 1: Descarga por URL directa (ELT Puro)
        if url:
            urllib.request.urlretrieve(url, archivo_salida)
            logger.info("✓ %s descargado exitosamente por HTTP.", nombre_fuente)
            
        # Estrategia 2: Descarga por API de Kaggle
        elif kaggle_dataset:
            
            # Rescate vía KaggleHub
            path_kaggle = kagglehub.dataset_download(kaggle_dataset)
            archivos_csv = glob.glob(os.path.join(path_kaggle, "*.csv"))
            
            if not archivos_csv:
                raise FileNotFoundError(f"No se encontró ningún archivo CSV en Kaggle: {path_kaggle}")
            
            # Mantenemos la auditoría: Selección de CSV determinista y segura
            csv_objetivo = next(
                (f for f in archivos_csv if NOMBRE_CSV_PHL in os.path.basename(f).lower()), 
                None
            )
            
            if csv_objetivo is None:
                csvs_disp = [os.path.basename(f) for f in archivos_csv]
                raise FileNotFoundError(
                    f"No se encontró '{NOMBRE_CSV_PHL}' en Kaggle. Archivos disponibles: {csvs_disp}"
                )
                
            shutil.copy(csv_objetivo, archivo_salida)
            shutil.rmtree(path_kaggle)  # Limpieza
            logger.info("✓ %s descargado y extraído exitosamente de Kaggle.", nombre_fuente)
            
        else:
            raise ValueError("Se debe proveer una 'url' o un 'kaggle_dataset'.")

        return pd.read_csv(archivo_salida, low_memory=False)

    # 2. Manejo de Errores Específicos e Informativos
    except urllib.error.HTTPError as e:
        logger.error("HTTP %s al descargar %s. ¿Está caído el servidor?", e.code, nombre_fuente)
    except urllib.error.URLError as e:
        logger.error("Sin conectividad de red para %s: %s", nombre_fuente, e.reason)
    except pd.errors.ParserError as e:
        logger.error("El CSV descargado de %s está corrupto: %s", nombre_fuente, e)
    except OSError as e:
        logger.error("Error de disco al manipular %s: %s", nombre_fuente, e)
    except Exception as e:
        # Fallback solo para errores verdaderamente inesperados
        logger.critical("Error crítico e inesperado al procesar %s: %s", nombre_fuente, e)
        
    return None

# Función orquestadora de la ingesta de datos
def ejecutar_ingesta():
    """Función orquestadora de la ingesta de datos."""
    logger.info("Iniciando Capa Bronce: Ingesta de catálogos raw...")
    
    # 1. NASA Exoplanet Archive
    df_nasa = descargar_catalogo(
        nombre_fuente="NASA",
        archivo_salida=f"{RUTA_RAW}/nasa_exoplanets.csv",
        url="https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv"
    )
    
    # 2. Exoplanet.eu (París)
    df_eu = descargar_catalogo(
        nombre_fuente="Exoplanet.eu",
        archivo_salida=f"{RUTA_RAW}/eu_exoplanets.csv",
        url="https://exoplanet.eu/catalog/csv"
    )
    
    # 3. PHL (Kaggle)
    df_phl = descargar_catalogo(
        nombre_fuente="PHL (Arecibo)",
        archivo_salida=f"{RUTA_RAW}/phl_exoplanets.csv",
        kaggle_dataset="chandrimad31/phl-exoplanet-catalog"
    )
    
    logger.info("Capa Bronce finalizada con éxito.")

if __name__ == "__main__":
    # Configuramos el formato visual del logger SOLO si se ejecuta este script directamente
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    ejecutar_ingesta()