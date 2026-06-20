"""Load the YAML config and resolve paths relative to the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/utils/config.py -> chartvision/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    """Turn a config path (relative to the project root) into an absolute Path."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
