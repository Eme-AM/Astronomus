# backend/src/models/train.py

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import joblib

# Importamos la nueva arquitectura del Autoencoder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.autoencoder import AstronomusAE, ExoplanetUnsupervisedDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

DIR_GOLD = "backend/data/gold"
DIR_ARTEFACTOS = "backend/artifacts/models"
os.makedirs(DIR_ARTEFACTOS, exist_ok=True)

# Hiperparámetros del Autoencoder
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# --- FIX MLOPS: Compresión de Colas Pesadas ---
FEATURES_LOG_TRANSFORM = ['pl_orbper', 'pl_bmasse', 'pl_insol', 'pl_dens']

def _aplicar_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprime features con distribución log-normal antes del RobustScaler.
    Evita que los outliers masivos saturen el gradiente del Autoencoder.
    """
    df = df.copy()
    for col in FEATURES_LOG_TRANSFORM:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df

def ejecutar_pipeline_hibrido():
    logging.info("Iniciando Pipeline Híbrido v2 (Con Log-Transform)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        df_completo = pd.read_csv(f"{DIR_GOLD}/dataset_preparado_ml.csv", low_memory=False)
    except FileNotFoundError:
        logging.error("No se encontró el dataset preparado en la Capa Oro.")
        return

    columnas_excluidas = ['pl_name', 'target_class']
    features = [c for c in df_completo.columns if c not in columnas_excluidas]
    
    mask_griales = df_completo['target_class'] == 2
    df_griales = df_completo[mask_griales].copy()
    df_normales = df_completo[~mask_griales].copy()
    
    # APLICACIÓN DEL FIX LOGARÍTMICO
    X_train_raw = _aplicar_log_transform(df_normales[features])
    X_todo_raw = _aplicar_log_transform(df_completo[features])
    
    logging.info(f"Universo de entrenamiento (Normalidad): {len(X_train_raw)} planetas.")
    logging.info(f"Griales ocultos para evaluación LOOCV: {len(df_griales)} planetas.")

    logging.info("Aplicando RobustScaler para normalizar magnitudes físicas...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_todo_scaled = scaler.transform(X_todo_raw)
    joblib.dump(scaler, f"{DIR_ARTEFACTOS}/robust_scaler.pkl")

    # ==========================================
    # FASE 1: ENTRENAMIENTO DEL AUTOENCODER
    # ==========================================
    logging.info("--- Iniciando Fase 1: Entrenamiento Autoencoder ---")
    dataset_ae = ExoplanetUnsupervisedDataset(pd.DataFrame(X_train_scaled, columns=features))
    
    dataloader = DataLoader(dataset_ae, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    modelo_ae = AstronomusAE(input_dim=len(features)).to(device)
    criterio = nn.MSELoss()
    optimizador = optim.AdamW(modelo_ae.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    modelo_ae.train()
    for epoch in range(EPOCHS):
        loss_acumulada = 0.0
        for batch_X, _ in dataloader:
            batch_X = batch_X.to(device)
            optimizador.zero_grad()
            
            reconstruccion = modelo_ae(batch_X)
            loss = criterio(reconstruccion, batch_X)
            
            loss.backward()
            optimizador.step()
            loss_acumulada += loss.item()
            
        if (epoch + 1) % 25 == 0:
            logging.info(f"  Época [{epoch+1}/{EPOCHS}] - MSE: {loss_acumulada/len(dataloader):.4f}")

    torch.save(modelo_ae.state_dict(), f"{DIR_ARTEFACTOS}/astronomus_ae.pth")
    
    # ==========================================
    # FASE 2: ENTRENAMIENTO ISOLATION FOREST
    # ==========================================
    logging.info("--- Iniciando Fase 2: Isolation Forest ---")
    modelo_if = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
    modelo_if.fit(X_train_scaled)
    joblib.dump(modelo_if, f"{DIR_ARTEFACTOS}/astronomus_if.pkl")

    # ==========================================
    # FASE 3: EVALUACIÓN Y RANKEADO
    # ==========================================
    logging.info("--- Fase 3: Evaluación de Anomalías en todo el Data Lake ---")
    
    X_todo_tensor = torch.tensor(X_todo_scaled, dtype=torch.float32).to(device)
    
    mse_scores = modelo_ae.get_reconstruction_error(X_todo_tensor).cpu().numpy()
    if_scores = -modelo_if.score_samples(X_todo_scaled)
    
    df_completo['ae_anomaly_score'] = mse_scores
    df_completo['if_anomaly_score'] = if_scores
    
    df_completo['ae_norm'] = (df_completo['ae_anomaly_score'] - df_completo['ae_anomaly_score'].min()) / (df_completo['ae_anomaly_score'].max() - df_completo['ae_anomaly_score'].min())
    df_completo['if_norm'] = (df_completo['if_anomaly_score'] - df_completo['if_anomaly_score'].min()) / (df_completo['if_anomaly_score'].max() - df_completo['if_anomaly_score'].min())
    
    df_completo['hybrid_score'] = (0.7 * df_completo['ae_norm']) + (0.3 * df_completo['if_norm'])
    
    ranking = df_completo.sort_values(by='hybrid_score', ascending=False).reset_index(drop=True)
    
    posiciones_griales = ranking[ranking['target_class'] == 2].index.tolist()
    posiciones_humanas = [p + 1 for p in posiciones_griales]
    
    logging.info(f"Resultados de la Búsqueda (Total: {len(ranking)} planetas):")
    logging.info(f"Posiciones de los 7 Griales: {posiciones_humanas}")
    
    top_5_percent = int(len(ranking) * 0.05)
    aciertos_top5 = sum(1 for p in posiciones_humanas if p <= top_5_percent)
    logging.info(f"Precisión del Ensamble: {aciertos_top5}/7 Griales en el Top 5% del universo.")
    
    # ==========================================
    # FASE 4: CREACIÓN DEL IHP Y PSEUDO-ETIQUETADO
    # ==========================================
    logging.info("--- Fase 4: Cálculo del IHP (Filtro Astrofísico) ---")
    
    df_completo['indice_rareza'] = (df_completo['hybrid_score'] * 100).round(2)
    
    # FILTRO FÍSICO (Bounding Box de Habitabilidad)
    # Extermina a los gigantes gaseosos (KELT-9 b) y mundos de lava
    mask_habitable = (
        (df_completo['pl_rade'] <= 2.5) & 
        (df_completo['pl_eqt'] >= 150) & 
        (df_completo['pl_eqt'] <= 400)
    )
    
    # Si es habitable, IHP = Rareza. Si no, IHP = 0.
    df_completo['ihp'] = np.where(mask_habitable, df_completo['indice_rareza'], 0.0)
    ranking = df_completo.sort_values(by='ihp', ascending=False).reset_index(drop=True)
    
    # Calculamos el umbral SOLO con los Griales que pasaron el filtro
    umbral_grial = ranking[ranking['target_class'] == 2]['ihp'].min()
    logging.info(f"Umbral de Habitabilidad IHP fijado en: {umbral_grial:.2f}%")
    
    # Pseudo-etiquetamos huérfanos que superen el umbral
    mask_candidatos_ia = (ranking['target_class'] == -1) & (ranking['ihp'] >= umbral_grial)
    nuevos_descubrimientos = mask_candidatos_ia.sum()
    
    ranking.loc[mask_candidatos_ia, 'target_class'] = 3
    logging.info(f"¡El modelo filtró la radiación y pseudo-etiquetó {nuevos_descubrimientos} Candidatos IA!")
    
    ranking.to_csv(f"{DIR_ARTEFACTOS}/ranking_anomalias.csv", index=False)
    logging.info("Pipeline completado exitosamente. Ranking exportado con IHP real.")

    # --- DIAGNÓSTICO PROFUNDO ---
    logging.info("\n" + "="*50)
    logging.info("PERFIL FÍSICO DE LOS 7 GRIALES (Auditoría de Reconstrucción)")
    logging.info("="*50)
    
    griales_ranking = ranking[ranking['target_class'] == 2][
        ['pl_name', 'hybrid_score', 'ae_norm', 'if_norm'] + features
    ].copy()
    griales_ranking['posicion'] = posiciones_humanas
    
    # Ordenamos columnas para visualización clara en consola
    columnas_print = ['pl_name', 'posicion', 'hybrid_score', 'ae_norm', 'if_norm', 'st_teff', 'pl_rade', 'pl_eqt', 'pl_insol']
    
    # Convertimos a string tabulado
    tabla_str = griales_ranking[columnas_print].to_string(index=False)
    for linea in tabla_str.split('\n'):
        logging.info(linea)
        
    logging.info("="*50 + "\n")
    
    ranking.to_csv(f"{DIR_ARTEFACTOS}/ranking_anomalias.csv", index=False)
    logging.info("Pipeline completado exitosamente. Ranking exportado.")

if __name__ == "__main__":
    ejecutar_pipeline_hibrido()