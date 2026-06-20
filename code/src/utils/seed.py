"""Reproducibility helper: seed Python, NumPy (and PyTorch later) from one call."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python's random and NumPy. PyTorch is seeded too if it is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is deferred until the training stage; seed it only if present.
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
