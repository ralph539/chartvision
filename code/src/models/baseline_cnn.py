"""Small from-scratch CNN baseline (~200k params), designed to train on CPU.

Three Conv-BN-ReLU-MaxPool blocks, global average pooling, then a linear classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """A compact CNN for 5-class candlestick pattern classification."""

    def __init__(self, num_classes: int = 5, in_channels: int = 3) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 16),   # 128 -> 64
            block(16, 32),            # 64 -> 32
            block(32, 64),            # 32 -> 16
            block(64, 64),            # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_baseline(num_classes: int = 5) -> BaselineCNN:
    """Factory used by the training script."""
    return BaselineCNN(num_classes=num_classes)


if __name__ == "__main__":
    m = build_baseline()
    n_params = sum(p.numel() for p in m.parameters())
    out = m(torch.randn(2, 3, 128, 128))
    print(f"BaselineCNN params={n_params:,}  output={tuple(out.shape)}")
