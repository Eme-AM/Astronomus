import argparse
import logging
import sys

from src.data.ingestion import ejecutar_ingesta
from src.data.processing import ejecutar_procesamiento
from src.data.preparation import ejecutar_preparacion_target

'''
HOLA
Si estás leyendo esto es porque acabás de clonar el repo e intuitivamente te metiste al primer Main que viste.
Este código se encarga de ejecutar el pipeline de la manera más amigable posible para que no tengas que ir archivo por archivo.
Hasta ahora la DNN no está implementada, si querés podés ir viendo eso.
'''

# Configuración del logger principal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Astronomus-Pipeline")

def ejecutar_pipeline_completo():
    """Ejecuta toda la tubería de datos de principio a fin."""
    logger.info("=== INICIANDO PIPELINE DE DATOS ASTRONOMUS ===")
    
    try:
        # 1. Capa Bronce
        ejecutar_ingesta()
        
        # 2. Capa Plata
        ejecutar_procesamiento()
        
        # 3. Capa Oro
        ejecutar_preparacion_target()
        
        logger.info("=== PIPELINE FINALIZADO CON ÉXITO ===")
        logger.info("Los tensores están listos en data/gold/ para entrenar la Red Neuronal.")
        
    except Exception as e:
        logger.error(f"El pipeline falló de manera crítica: {e}")
        sys.exit(1)

def mostrar_menu_interactivo():
    """Muestra un menú amigable si el usuario corre el script sin argumentos."""
    print("\n" + "="*50)
    print(" BIENVENIDO A ASTRONOMUS - GESTOR DE PIPELINE ")
    print("="*50)
    print("Seleccioná qué parte del proyecto querés ejecutar:")
    print("  1. Ejecutar Pipeline de Datos Completo (Ingesta -> Plata -> Oro)")
    print("  2. Ejecutar solo Ingesta (Capa Bronce)")
    print("  3. Ejecutar solo Procesamiento (Capa Plata)")
    print("  4. Ejecutar solo Preparación (Capa Oro)")
    print("  0. Salir")
    print("="*50)
    
    while True:
        
        opcion = input("Ingresá un número (0-4): ")

        if opcion == '1':
            print()
            ejecutar_pipeline_completo()
            break
        elif opcion == '2':
            print()
            ejecutar_ingesta()
            break
        elif opcion == '3':
            print()
            ejecutar_procesamiento()
            break
        elif opcion == '4':
            print()
            ejecutar_preparacion_target()
            break
        elif opcion == '0':
            print("Saliendo...")
            sys.exit(0)
            break
        else:
            print("Opción no válida. DALE NO ES TAN DIFICIL ELEGIR UN NUMERO DEL 0 AL 4. ")
            continue

if __name__ == "__main__":
    # Configuramos argparse para que los desarrolladores avanzados puedan usar flags
    parser = argparse.ArgumentParser(description="Orquestador principal del proyecto Astronomus.")
    parser.add_argument('--all', action='store_true', help="Ejecuta todo el pipeline de datos.")
    parser.add_argument('--ingest', action='store_true', help="Ejecuta solo la ingesta de datos.")
    parser.add_argument('--process', action='store_true', help="Ejecuta solo el procesamiento (Capa Plata).")
    parser.add_argument('--prepare', action='store_true', help="Ejecuta solo la preparación (Capa Oro).")
    
    args = parser.parse_args()
    
    # Lógica de ejecución basada en los argumentos
    if args.all:
        ejecutar_pipeline_completo()
    elif args.ingest:
        ejecutar_ingesta()
    elif args.process:
        ejecutar_procesamiento()
    elif args.prepare:
        ejecutar_preparacion_target()
    else:
        mostrar_menu_interactivo()