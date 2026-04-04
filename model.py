"""
Churn Prediction - Deep Learning Model
PyTorch neural network for binary churn classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChurnNet(nn.Module):
    """
    Deep neural network for churn prediction.
    Architecture: Input -> BatchNorm -> [Dense -> BN -> ReLU -> Dropout] x3 -> Output
    """

    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], dropout: float = 0.3):
        super(ChurnNet, self).__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.hidden(x)
        return self.output(x)  # raw logits — BCEWithLogitsLoss handles sigmoid


class ChurnDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for churn data."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
