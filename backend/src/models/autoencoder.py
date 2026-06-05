# backend/src/models/autoencoder.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ExoplanetUnsupervisedDataset(Dataset):
    """
    Dataset para aprendizaje no supervisado (Autoencoder).
    En un AE la etiqueta es la propia entrada (X -> X'),
    por lo que __getitem__ retorna (X, X).
    """
    def __init__(self, features_df: pd.DataFrame):
        # Validación defensiva: NaN silencioso mata el entrenamiento
        nans = features_df.isnull().sum().sum()
        if nans > 0:
            cols_nan = features_df.columns[features_df.isnull().any()].tolist()
            raise ValueError(
                f"ExoplanetUnsupervisedDataset recibió {nans} NaN en: {cols_nan}. "
                "Ejecutá preparation.py para regenerar la Capa Oro."
            )
        # Reemplazamos torch.tensor por torch.from_numpy para evitar copias de memoria
        self.X = torch.from_numpy(features_df.values.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.X[idx]


class AstronomusAE(nn.Module):
    """
    Autoencoder simétrico para detección de anomalías astrofísicas.

    Topología: input_dim -> 8 -> 4 -> 2 (latente) -> 4 -> 8 -> input_dim

    El cuello de botella de 2 dimensiones permite:
    - Visualización directa del espacio latente (scatter 2D)
    - Compresión agresiva que fuerza al modelo a aprender solo
      los patrones más comunes, dejando a los Griales con
      alto error de reconstrucción.
    """
    def __init__(self, input_dim: int):       
        super().__init__()                    

        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.GELU(),
            nn.Linear(4, 2),
            # Sin BN ni activación para que el espacio latente retenga magnitudes absolutas
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.BatchNorm1d(4),
            nn.GELU(),
            nn.Linear(4, 8),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Linear(8, input_dim),
            # Salida lineal: RobustScaler ya centró y escaló los datos.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Comprime x al espacio latente y lo reconstruye."""
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def get_latent_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna la representación 2D del espacio latente.
        """
        self.eval()
        return self.encoder(x)

    @torch.no_grad()
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula el error de reconstrucción MSE por muestra.
        Este es el anomaly score: valores altos = planeta físicamente atípico.

        Returns:
            Tensor de shape (N,) con el MSE de cada planeta.
        """
        self.eval()
        x_rec = self.forward(x)
        # mean(dim=1): promedio sobre las 12 features, no sobre el batch
        return ((x_rec - x) ** 2).mean(dim=1)


if __name__ == "__main__":
    print("Sanity check — AstronomusAE")

    FEATURES_REALES = 12   
    BATCH = 16

    modelo = AstronomusAE(input_dim=FEATURES_REALES)
    modelo.eval()

    x_normal = torch.randn(BATCH, FEATURES_REALES)
    x_grial  = torch.randn(BATCH, FEATURES_REALES) * 3   

    x_rec    = modelo(x_normal)
    err_norm = modelo.get_reconstruction_error(x_normal)
    err_gria = modelo.get_reconstruction_error(x_grial)
    latente  = modelo.get_latent_features(x_normal)

    assert x_rec.shape == x_normal.shape,  "Shape de reconstrucción incorrecta"
    assert err_norm.shape == (BATCH,),      "get_reconstruction_error debe ser (N,)"
    assert latente.shape  == (BATCH, 2),    "Espacio latente debe ser (N, 2)"

    params = sum(p.numel() for p in modelo.parameters())

    print(f"  Input:           {x_normal.shape}")
    print(f"  Latente:         {latente.shape}   (2D -> visualizable directamente)")
    print(f"  Reconstrucción:  {x_rec.shape}")
    print(f"  Error normales:  {err_norm.mean().item():.4f}  (sin entrenar -> ruido)")
    print(f"  Error 'Griales': {err_gria.mean().item():.4f}  (esperado mayor post-entrenamiento)")
    print(f"  Parámetros:      {params:,}  ({params/5200:.2f}x muestras — OK)")
    print("\n✓ Todas las aserciones pasaron.")