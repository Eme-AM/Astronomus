# src/api/app.py

from __future__ import annotations
import logging
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .routers.exoplanets import router as exoplanets_router

# ==========================================
# CONFIGURACIÓN DEL SISTEMA
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="Astronomus API",
    description="Motor de Inferencia y Catálogo Espacial de Exoplanetas Habitables.",
    version="1.0.0"
)

# Configuración CORS — restringe al origen definido por CORS_ORIGIN.
# En desarrollo local la variable no es necesaria (default: localhost:8000).
# En producción: export CORS_ORIGIN=https://mi-dominio.com
_cors_origins = os.getenv("CORS_ORIGIN", "http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

# Registro de Rutas
app.include_router(exoplanets_router)

# ==========================================
# SERVIDOR DEL FRONTEND INTERACTIVO
# ==========================================
FRONTEND_DIR = Path("frontend")

@app.get("/", include_in_schema=False, response_model=None)
def serve_viewer() -> FileResponse | JSONResponse:
    """Sirve el visor 3D nativo de Astronomus."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index, media_type="text/html")
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "Interfaz gráfica no encontrada.",
            "hint": "Verificá que el archivo 'frontend/index.html' exista en la raíz del proyecto.",
        },
    )

# ==========================================
# MONITOREO DE SALUD (Health Check)
# ==========================================
@app.get("/health", tags=["ops"])
def health_check() -> dict:
    """Endpoint de salud para monitoreo de infraestructura."""
    return {"status": "operativo", "service": "Astronomus API"}

# Para ejecutar: uvicorn backend/src.api.app:app --reload --port 8000