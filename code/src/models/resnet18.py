"""ResNet18 transfer-learning model: ImageNet backbone, new 5-class head.

Used in the final stage (on Colab GPU). For the 50% checkpoint we focus on the baseline CNN,
but the builder is here so `train.py --model resnet18` works end-to-end.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 5, mode: str = "finetune") -> nn.Module:
    """Load ImageNet-pretrained ResNet18 and replace the FC layer for `num_classes`.

    mode:
      - 'finetune' : freeze all except the last residual block (layer4) + new FC head
                     (~8.4M trainable params). Original setting.
      - 'linear'   : freeze the ENTIRE backbone, train only the new FC head
                     (~2.5k trainable params). Better for small datasets.
    """
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if mode in ("finetune", "linear"):
        for p in net.parameters():
            p.requires_grad = False
        if mode == "finetune":
            for p in net.layer4.parameters():
                p.requires_grad = True
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net
