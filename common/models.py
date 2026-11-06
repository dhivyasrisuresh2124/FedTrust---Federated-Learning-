"""Heterogeneous model zoo.

Campuses may run *different architectures*. Because only predictions on the anchor
set are exchanged (never weights), the architectures do not have to match.
"""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_shape, n_classes, hidden, dropout=0.1):
        super().__init__()
        d = int(torch.tensor(in_shape).prod())
        layers: list[nn.Module] = [nn.Flatten()]
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNSmall(nn.Module):
    def __init__(self, in_shape, n_classes):
        super().__init__()
        c = in_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 9, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class CNNDeep(nn.Module):
    def __init__(self, in_shape, n_classes):
        super().__init__()
        c = in_shape[0]

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

        self.features = nn.Sequential(block(c, 16), block(16, 32), nn.MaxPool2d(2), block(32, 48),
                                      nn.AdaptiveAvgPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(48 * 4, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str, in_shape, n_classes: int) -> nn.Module:
    if name == "mlp_small":
        return MLP(in_shape, n_classes, [64])
    if name == "mlp_wide":
        return MLP(in_shape, n_classes, [256, 128])
    if name == "linear":
        return MLP(in_shape, n_classes, [], dropout=0.0)
    if name == "cnn_small":
        return CNNSmall(in_shape, n_classes)
    if name == "cnn_deep":
        return CNNDeep(in_shape, n_classes)
    raise ValueError(f"unknown model {name!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
