import torch
import torch.nn as nn

class ExoplanetDNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(ExoplanetDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    