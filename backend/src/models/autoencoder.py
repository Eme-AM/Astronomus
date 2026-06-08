import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Pesos astrofísicos para el cálculo de anomalías
FEATURE_HABITABILITY_WEIGHTS = {
    'pl_rade':    4.0,   # Discriminador rocoso/gaseoso (crítico)
    'pl_eqt':     4.0,   # Temperatura equilibrio (crítico)
    'pl_insol':   3.0,   # Zona habitable estelar
    'pl_dens':    2.0,   # Composición interna
    'pl_bmasse':  1.5,   # Retención atmosférica
    'st_teff':    1.0,   # Tipo estelar
    'pl_orbeccen':0.8,   # Estabilidad orbital
    'st_rad':     0.5,   # Parámetro estelar secundario
    'st_mass':    0.5,
    'st_met':     0.5,   # Metalicidad
    'pl_orbper':  0.3,   # Período orbital
    'st_age':     0.3,   # Tiempo para desarrollo de vida
}

class ExoplanetUnsupervisedDataset(Dataset):
    """Dataset para aprendizaje no supervisado (Autoencoder)."""
    def __init__(self, features_df: pd.DataFrame):
        nans = features_df.isnull().sum().sum()
        if nans > 0:
            cols_nan = features_df.columns[features_df.isnull().any()].tolist()
            raise ValueError(
                f"ExoplanetUnsupervisedDataset recibió {nans} NaN en: {cols_nan}. "
                "Ejecutá preparation.py para regenerar la Capa Oro."
            )
        self.X = torch.from_numpy(features_df.values.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.X[idx]

class AstronomusAE(nn.Module):
    """Autoencoder simétrico para detección de anomalías astrofísicas."""
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
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.BatchNorm1d(4),
            nn.GELU(),
            nn.Linear(4, 8),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Comprime x al espacio latente y lo reconstruye."""
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def get_latent_features(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna la representación 2D del espacio latente."""
        self.eval()
        return self.encoder(x)

    @torch.no_grad()
    def get_reconstruction_error(
        self, 
        x: torch.Tensor, 
        feature_names: list[str] | None = None
    ) -> torch.Tensor:
        """
        Error de reconstrucción MSE ponderado por relevancia astrofísica.
        """
        self.eval()
        x_rec = self.forward(x)
        sq_err = (x_rec - x) ** 2

        if feature_names is not None:
            weights = torch.tensor(
                [FEATURE_HABITABILITY_WEIGHTS.get(f, 1.0) for f in feature_names],
                dtype=torch.float32,
                device=x.device,
            )
            # Normalizar los pesos para mantener la escala
            weights = weights / weights.mean()
            return (sq_err * weights).mean(dim=1)

        return sq_err.mean(dim=1)

if __name__ == "__main__":
    print("Sanity check — AstronomusAE")
    FEATURES_REALES = 12   
    BATCH = 16
    modelo = AstronomusAE(input_dim=FEATURES_REALES)
    modelo.eval()
    x_normal = torch.randn(BATCH, FEATURES_REALES)
    x_grial  = torch.randn(BATCH, FEATURES_REALES) * 3   
    
    # Check con y sin pesos
    test_features = ['pl_rade', 'pl_eqt'] + ['feat']*10
    err_norm = modelo.get_reconstruction_error(x_normal)
    err_weighted = modelo.get_reconstruction_error(x_normal, test_features)
    
    assert err_norm.shape == (BATCH,)
    assert err_weighted.shape == (BATCH,)
    print("\n✓ Todas las aserciones pasaron.")