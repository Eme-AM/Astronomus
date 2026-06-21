from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ===============================
# RUTAS CANÓNICAS DEL PROYECTO
# ===============================
DIR_ARCHIVE = Path("backend/data/archive")
DIR_SILVER  = Path("backend/data/silver")
DIR_GOLD    = Path("backend/data/gold")
DIR_FIGURAS = Path("backend/reports/figures")

# ==============================
# ESTILO ÚNICO DE PUBLICACIÓN
# ==============================
PLOT_CONTEXT = "paper"
FONT_SCALE   = 1.2
STYLE        = "whitegrid"

def configurar_estilo() -> None:
    """Crea la carpeta de salida y aplica el tema científico estándar."""
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style=STYLE, context=PLOT_CONTEXT, font_scale=FONT_SCALE)

def cargar_silver(nombre: str = "data_lake_consolidado.csv") -> "pd.DataFrame":
    """Carga la Capa Plata con validación estricta de existencia."""
    import pandas as pd
    ruta = DIR_SILVER / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"Error: Silver layer no encontrada en {ruta}. Ejecutá processing.py primero.")
    return pd.read_csv(ruta, low_memory=False)

def cargar_gold(nombre: str) -> "pd.DataFrame":
    """Carga un artefacto de la Capa Oro con validación estricta."""
    import pandas as pd
    ruta = DIR_GOLD / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"Error: Gold layer no encontrada en {ruta}. Ejecutá preparation.py primero.")
    return pd.read_csv(ruta, low_memory=False)

def guardar_figura(nombre: str, fig=None) -> Path:
    """Aplica estética final, guarda la figura y libera memoria."""
    sns.despine(fig=fig) # Remueve bordes redundantes a nivel global
    ruta = DIR_FIGURAS / nombre
    (fig or plt).savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close('all') # Limpieza absoluta de memoria
    print(f" ✓ Gráfico guardado en: {ruta}")
    return ruta