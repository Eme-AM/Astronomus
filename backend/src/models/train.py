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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.autoencoder import AstronomusAE, ExoplanetUnsupervisedDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

DIR_GOLD = "backend/data/gold"
DIR_SILVER = "backend/data/silver"
DIR_ARTEFACTOS = "backend/artifacts/models"
os.makedirs(DIR_ARTEFACTOS, exist_ok=True)

EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def ejecutar_pipeline_hibrido():
    logging.info("Iniciando Pipeline Híbrido (Weighted AE + Isolation Forest)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        df_scaled = pd.read_csv(f"{DIR_GOLD}/X_scaled.csv")
        y_df = pd.read_csv(f"{DIR_GOLD}/y.csv")
        df_completo_plata = pd.read_csv(f"{DIR_SILVER}/data_lake_consolidado.csv", low_memory=False)
        df_completo_oro = pd.read_csv(f"{DIR_GOLD}/dataset_preparado_ml.csv", low_memory=False)
    except FileNotFoundError as e:
        logging.error(f"Falta archivo de datos: {e}")
        return

    # Separar IDs para alineación segura
    pl_names = df_scaled['pl_name']
    X_scaled_df = df_scaled.drop(columns=['pl_name'])
    features = X_scaled_df.columns.tolist()
    
    # Estrategia Conservadora: Entrenar SOLO con planetas etiquetados como no-habitables
    mask_normales = (y_df['target_class'] == 0) | (y_df['target_class'] == 1)
    X_train_normales = X_scaled_df[mask_normales]

    logging.info(f"Universo de entrenamiento (Normalidad etiquetada): {len(X_train_normales)} planetas.")

    # ==========================================
    # FASE 1: ENTRENAMIENTO AUTOENCODER
    # ==========================================
    logging.info("--- Fase 1: Entrenamiento Autoencoder ---")
    dataset_ae = ExoplanetUnsupervisedDataset(X_train_normales)
    dataloader = DataLoader(
        dataset_ae,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    
    modelo_ae = AstronomusAE(input_dim=len(features)).to(device)
    criterio = nn.MSELoss()
    optimizador = optim.AdamW(modelo_ae.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Scheduler y clipping para estabilidad
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=EPOCHS, eta_min=1e-5)
    
    modelo_ae.train()
    for epoch in range(EPOCHS):
        loss_acumulada = 0.0
        for batch_X, _ in dataloader:
            batch_X = batch_X.to(device)
            optimizador.zero_grad()
            loss = criterio(modelo_ae(batch_X), batch_X)
            loss.backward()
            nn.utils.clip_grad_norm_(modelo_ae.parameters(), max_norm=1.0)
            optimizador.step()
            loss_acumulada += loss.item()
            
        scheduler.step()
        
        if (epoch + 1) % 25 == 0:
            logging.info("  Época [%d/%d] - MSE: %.4f | LR: %.6f", 
                         epoch+1, EPOCHS, loss_acumulada/len(dataloader), scheduler.get_last_lr()[0])

    torch.save(modelo_ae.state_dict(), f"{DIR_ARTEFACTOS}/astronomus_ae.pth")
    
    # ==========================================
    # FASE 2: ISOLATION FOREST
    # ==========================================
    logging.info("--- Fase 2: Isolation Forest ---")
    modelo_if = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
    modelo_if.fit(X_train_normales.values)

    # ==========================================
    # FASE 3: EVALUACIÓN Y RANKEADO
    # ==========================================
    logging.info("--- Fase 3: Evaluación de Anomalías Globales ---")
    X_todo_tensor = torch.tensor(X_scaled_df.values, dtype=torch.float32).to(device)
    
    # Aplicar MSE ponderado por astrofísica
    mse_scores = modelo_ae.get_reconstruction_error(X_todo_tensor, feature_names=features).cpu().numpy()
    if_scores = -modelo_if.score_samples(X_scaled_df.values)
    
    # Alineación segura mediante pl_name
    df_scores = pd.DataFrame({
        'pl_name': pl_names.values,
        'ae_score': mse_scores,
        'if_score': if_scores,
    })
    
    df_ranking = df_completo_oro.merge(df_scores, on='pl_name', how='left')
    
    ae_denom = df_ranking['ae_score'].max() - df_ranking['ae_score'].min()
    if_denom = df_ranking['if_score'].max() - df_ranking['if_score'].min()
    ae_norm = (df_ranking['ae_score'] - df_ranking['ae_score'].min()) / (ae_denom + 1e-8)
    if_norm = (df_ranking['if_score'] - df_ranking['if_score'].min()) / (if_denom + 1e-8)
    
    df_ranking['hybrid_score'] = (0.7 * ae_norm) + (0.3 * if_norm)

    # ==========================================
    # FASE 4: ÍNDICE DE HABITABILIDAD PLANETARIA (Modelo Heller)
    # ==========================================
    logging.info("--- Fase 4: Cálculo del IHP (Rareza IA + Súper-Habitabilidad Heller) ---")
    
    # 1. Conservamos la métrica de Excepcionalidad de la IA
    df_ranking['excepcionalidad_ia'] = df_ranking['hybrid_score']
    
    # 2. Óptimos Súper-Habitables (René Heller & John Armstrong)
    IDEAL_RADE = 1.25   # Mayor retención atmosférica y tectónica activa
    IDEAL_TEFF = 4200.0 # Estrella Enana K (Larga vida, baja radiación estelar)
    IDEAL_EQT  = 265.0  # Clima global ligeramente más cálido que la Tierra
    
    # 3. Cálculo de Similitud de Heller (Fórmula tipo ESI Euclidiana ponderada)
    eps = 1e-8 # Previene divisiones por cero
    
    # Utilizamos pesos empíricos (Radio y Temperatura Eq son críticos)
    sim_rade = (1 - abs((df_ranking['pl_rade'] - IDEAL_RADE) / (df_ranking['pl_rade'] + IDEAL_RADE + eps))) ** 4.0
    sim_teff = (1 - abs((df_ranking['st_teff'] - IDEAL_TEFF) / (df_ranking['st_teff'] + IDEAL_TEFF + eps))) ** 2.0
    sim_eqt  = (1 - abs((df_ranking['pl_eqt']  - IDEAL_EQT)  / (df_ranking['pl_eqt']  + IDEAL_EQT  + eps))) ** 4.0
    
    df_ranking['similitud_heller'] = (sim_rade * sim_teff * sim_eqt) ** (1/3)
    
    # 4. El Bounding Box Físico Duro (La guillotina de los gigantes gaseosos)
    mask_habitable = (
        (df_ranking['pl_rade'] <= 2.5) & 
        (df_ranking['pl_eqt'] >= 150) & 
        (df_ranking['pl_eqt'] <= 500)
    )
    
    # 5. EL NUEVO IHP: Confluencia de Excepcionalidad Estadística y Perfección Biológica
    # Un 50% de peso a que la IA lo aísle de la monotonía galáctica
    # Un 50% de peso a que la física se acerque al paraíso de Heller
    
    # 1. Conservamos los sub-scores desglosados
    df_ranking['score_ia'] = (df_ranking['excepcionalidad_ia'] * 100).round(2)
    df_ranking['score_heller'] = (df_ranking['similitud_heller'] * 100).round(2)
    
    # 2. IHP ponderado (Mantenemos la lógica de 50/50, pero ahora es auditable)
    df_ranking['ihp'] = np.where(mask_habitable,(df_ranking['score_ia'] * 0.5) + (df_ranking['score_heller'] * 0.5), 0.0).round(2)
    #ihp_calculado = (df_ranking['excepcionalidad_ia'] * 0.5) + (df_ranking['similitud_heller'] * 0.5)
    
    # Escalamos a porcentaje y fulminamos a los no habitables
    #df_ranking['ihp'] = np.where(mask_habitable, ihp_calculado * 100, 0.0).round(2)
    df_ranking = df_ranking.sort_values(by='ihp', ascending=False).reset_index(drop=True)

    # 6. Umbral Estadístico y Pseudo-etiquetado
    umbral_ia = df_ranking[df_ranking['ihp'] > 0]['ihp'].quantile(0.96)
    mask_candidatos = (df_ranking['target_class'] == -1) & (df_ranking['ihp'] >= umbral_ia)
    nuevos_hallazgos = mask_candidatos.sum()
    df_ranking.loc[mask_candidatos, 'target_class'] = 3
    
    logging.info(f"Umbral IA fijado en IHP: {umbral_ia:.2f}%. Hallazgos IA etiquetados: {nuevos_hallazgos}")
    
    df_final = pd.merge(
        df_ranking[['pl_name', 'target_class', 'ihp', 'score_ia', 'score_heller']],
        df_completo_plata, 
        on='pl_name', 
        how='left'
    )
    
    df_final.to_csv(f"{DIR_ARTEFACTOS}/ranking_anomalias.csv", index=False)
    
    # ==========================================
    # DIAGNÓSTICO FINAL
    # ==========================================
    cols_diagnostico = ['pl_name', 'ihp', 'pl_rade', 'pl_eqt', 'pl_insol', 'st_teff', 'st_mass']
    cols_presentes = [c for c in cols_diagnostico if c in df_ranking.columns]
    
    nuevos = df_ranking[df_ranking['target_class'] == 3]
    logging.info("\n── PERFIL DE LOS %d HALLAZGOS IA ──", len(nuevos))
    print(nuevos[cols_presentes].sort_values('ihp', ascending=False).to_string(index=False))

    griales = df_ranking[df_ranking['target_class'] == 2]
    logging.info("\n── GRIALES CONOCIDOS (Validación) ──")
    print(griales[cols_presentes].sort_values('ihp', ascending=False).to_string(index=False))
    
    logging.info("\nPipeline completado exitosamente. Ranking exportado con IHP real.")

if __name__ == "__main__":
    ejecutar_pipeline_hibrido()