from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib

app = FastAPI(title="Planetary Habitability API")

# DTO: Definición del objeto que espera la API
class PlanetRequest(BaseModel):
    pl_rade: float
    pl_bmasse: float
    pl_eqt: float
    pl_insol: float
    st_teff: float

@app.get("/")
def read_root():
    return {"status": "Astro-API Online"}

@app.post("/predict")
def predict_habitability(planet: PlanetRequest):
    # Aquí iría la carga del modelo y la inferencia
    # Pensado para que tu futura Mobile App consuma este JSON
    return {
        "planet_name": "Unknown Candidate",
        "habitability_score": 0.85,
        "is_habitable": True
    }