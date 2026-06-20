"""Grad-CAM analysis: per-class heatmap grids + 'confidently wrong' failure grids.

Two modes:
  --mode correct  : for each class, n_per_class CORRECT predictions, input | heatmap.
  --mode failures : for each class, the n_per_class MOST CONFIDENTLY WRONG predictions
                    (true class = c, predicted != c with highest predicted prob).

Usage:
    python -m src.training.gradcam --checkpoint checkpoints/baseline_best.pt --mode correct
    python -m src.training.gradcam --checkpoint checkpoints/resnet18_best.pt --mode failures
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import make_loaders
from src.models.baseline_cnn import build_baseline
from src.models.resnet18 import build_resnet18
from src.utils.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def last_conv_layer(model, name: str):
    """Return the last convolutional layer to hook for Grad-CAM."""
    if name == "baseline":
        return model.features[-1][0]
    return model.layer4[-1].conv2


class GradCAM:
    """Standard Grad-CAM: weight activation maps by mean gradient of the target class."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.activations = out.detach()

    def _bwd(self, _m, _gi, gout):
        self.gradients = gout[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(x)
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze(0)
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def _upsample(heat: np.ndarray, shape) -> np.ndarray:
    """Bilinearly upsample a small heatmap to the input-image resolution."""
    t = torch.from_numpy(heat).float()[None, None]
    up = F.interpolate(t, size=shape, mode="bilinear", align_corners=False)
    return up.squeeze().numpy()


def collect_correct(model, cam, loader, classes, n_per_class: int):
    """For each class, collect up to n_per_class correctly-classified examples."""
    picked = {c: [] for c in range(len(classes))}
    for x, y in loader:
        c = int(y.item())
        if len(picked[c]) >= n_per_class:
            continue
        with torch.no_grad():
            pred = model(x).argmax(1).item()
        if pred == c:
            heat = cam(x, c)
            picked[c].append((x.squeeze(0).permute(1, 2, 0).numpy(), heat,
                              f"pred={classes[c]} ✓"))
        if all(len(v) >= n_per_class for v in picked.values()):
            break
    return picked


def collect_failures(model, cam, loader, classes, n_per_class: int):
    """For each true class, find the n_per_class WRONG predictions with the highest confidence."""
    by_true: dict[int, list] = {c: [] for c in range(len(classes))}
    for x, y in loader:
        c = int(y.item())
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
            pred = int(probs.argmax().item())
            conf = float(probs[pred].item())
        if pred != c:
            heat = cam(x, pred)  # heatmap for the *predicted* wrong class
            by_true[c].append((conf, x.squeeze(0).permute(1, 2, 0).numpy(), heat,
                               f"pred={classes[pred]} ({conf:.0%}) ✗"))
    return {c: [tup[1:] for tup in sorted(v, key=lambda t: -t[0])[:n_per_class]]
            for c, v in by_true.items()}


def render_grid(picked, classes, model_name: str, mode: str, n_per_class: int, out_path):
    ncol = n_per_class
    fig, axes = plt.subplots(len(classes), ncol * 2, figsize=(ncol * 3.3, len(classes) * 1.9))
    for r, c in enumerate(range(len(classes))):
        for j in range(ncol):
            ax_img = axes[r, 2 * j]
            ax_cam = axes[r, 2 * j + 1]
            if j < len(picked[c]):
                img, heat, label = picked[c][j]
                heat_up = _upsample(heat, img.shape[:2])
                ax_img.imshow(img)
                ax_cam.imshow(img); ax_cam.imshow(heat_up, cmap="jet", alpha=0.5)
                ax_cam.set_title(label, fontsize=7)
            ax_img.axis("off"); ax_cam.axis("off")
        axes[r, 0].set_ylabel(classes[c], rotation=0, ha="right", va="center", fontsize=9, labelpad=42)
    title = f"Grad-CAM - {model_name} ({mode})"
    if mode == "failures":
        title += "  ·  most confidently WRONG per true class"
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-per-class", type=int, default=3)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--mode", default="correct", choices=["correct", "failures"])
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("cpu")
    ckpt = torch.load(resolve_path(args.checkpoint), map_location=device, weights_only=False)
    classes, image_size, name = ckpt["classes"], ckpt["image_size"], ckpt["model"]

    model = build_baseline(len(classes)) if name == "baseline" else build_resnet18(len(classes))
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    cam = GradCAM(model, last_conv_layer(model, name))

    _, va, te, _ = make_loaders(cfg, image_size, batch_size=1, num_workers=0)
    loader = {"val": va, "test": te}[args.split]

    if args.mode == "correct":
        picked = collect_correct(model, cam, loader, classes, args.n_per_class)
    else:
        picked = collect_failures(model, cam, loader, classes, args.n_per_class)

    out = resolve_path(cfg["paths"]["figures_dir"]) / f"gradcam_{name}_{args.split}_{args.mode}.png"
    render_grid(picked, classes, name, args.mode, args.n_per_class, out)
    logger.info("Saved Grad-CAM grid -> %s", out)


if __name__ == "__main__":
    main()
