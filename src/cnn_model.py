"""Tiny CNN for 28x28 digits (kept separate so the app imports without torch installed)."""
import torch.nn as nn


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(3))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(96 * 9, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.head(self.features(x))
