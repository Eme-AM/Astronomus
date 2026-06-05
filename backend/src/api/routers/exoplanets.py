# src/api/routers/exoplanets.py

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["exoplanets"])

# ==========================================
# CONFIGURACIÓN Y RUTAS (Arquitectura Medallón)
# ==========================================
# Path(__file__) apunta a exoplanets.py. 
# Sus 'parents' son: 1(routers) -> 2(api) -> 3(src) -> 4(backend)
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent

SILVER_PATH = BACKEND_DIR / "data" / "silver" / "data_lake_consolidado.csv"
GOLD_PATH = BACKEND_DIR / "data" / "gold" / "dataset_preparado_ml.csv"
RANKING_PATH = BACKEND_DIR / "artifacts" / "models" / "ranking_anomalias.csv"

# Escala de la bóveda celeste en unidades de Three.js
SPHERE_RADIUS = 100.0   

# ==========================================
# GEOMETRÍA ESPACIAL
# ==========================================
def _to_cartesian(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convierte coordenadas ecuatoriales (RA, Dec) a vectores 3D.
    Se encapsula aquí para mantener la API completamente independiente 
    del pipeline de procesamiento de datos subyacente.
    """
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return (
        np.cos(dec) * np.cos(ra) * SPHERE_RADIUS,
        np.cos(dec) * np.sin(ra) * SPHERE_RADIUS,
        np.sin(dec) * SPHERE_RADIUS,
    )

# ==========================================
# ENDPOINT PRINCIPAL (El Catálogo 3D)
# ==========================================
@router.get("/exoplanets", summary="Catálogo 3D de exoplanetas para el visor WebGL")
def get_exoplanets() -> JSONResponse:
    """
    Retorna un payload columnar optimizado. En lugar de una lista de miles de 
    objetos JSON, enviamos arrays paralelos (Float32Array) que el motor 
    Three.js puede inyectar directamente en la GPU sin iteraciones costosas.
    """
    # 1. CARGA INTELIGENTE (Priorizamos el modelo de Machine Learning)
    if RANKING_PATH.exists():
        silver = pd.read_csv(RANKING_PATH, low_memory=False)
        silver = silver.dropna(subset=["ra", "dec"]).reset_index(drop=True)
        logger.info("Catálogo rankeado por IA cargado: %d planetas.", len(silver))
    else:
        # Fallback de seguridad si aún no entrenaron el modelo
        if not SILVER_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"No se encontró ranking ni Capa Plata. Ejecutá processing.py primero."
            )
        
        silver = pd.read_csv(SILVER_PATH, low_memory=False)
        silver = silver.dropna(subset=["ra", "dec"]).reset_index(drop=True)
        logger.warning("Cargando Silver Layer (Fallback). IA no disponible.")

        # Enriquecimiento con Capa Oro para el fallback
        if GOLD_PATH.exists():
            gold = pd.read_csv(GOLD_PATH, usecols=["pl_name", "target_class"])
            silver = silver.merge(gold, on="pl_name", how="left")
        else:
            silver["target_class"] = np.nan

    # Sanitización de clases: -1 significa "No etiquetado / Desconocido"
    silver["target_class"] = silver["target_class"].fillna(-1).astype(int)

    # 3. Transformación Geométrica y Estética
    x, y, z = _to_cartesian(silver["ra"].values, silver["dec"].values)
    
    # Clip para evitar que planetas extremos rompan la visualización en el frontend
    silver["pl_rade"] = silver["pl_rade"].fillna(1.0).clip(lower=0.1).round(3)
    silver["st_teff"] = silver["st_teff"].fillna(5778.0).clip(lower=2500, upper=50000).round(1)

    # 4. Construcción del Payload Columnar
    payload = {
        "meta": {
            "total":          len(silver),
            "labeled":        int((silver["target_class"] >= 0).sum()),
            "griales":        int((silver["target_class"] == 2).sum()),
            "ia_candidates":  int((silver["target_class"] == 3).sum()), # <-- NUEVO
            "schema_version": "1.0",
        },
        "positions":      np.column_stack([x, y, z]).flatten().round(4).tolist(),
        "temperatures":   silver["st_teff"].tolist(),
        "radii":          silver["pl_rade"].tolist(),
        "target_classes": silver["target_class"].tolist(),
        
        # Enviamos el IHP en lugar del anomaly_score crudo
        "ihp":            silver["ihp"].tolist() if "ihp" in silver.columns else [0] * len(silver),
        
        "names":          silver["pl_name"].astype(str).tolist(),
    }

    return JSONResponse(payload)