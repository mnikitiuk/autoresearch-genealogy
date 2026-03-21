"""Configuration loader with environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_config(path: str | Path = "config/settings.yaml") -> dict:
    """Load YAML config and apply any matching environment variable overrides."""
    with open(path) as f:
        config = yaml.safe_load(f)

    # Simple env-var override: GAS_ANALYZER__SECTION__KEY=value
    prefix = "GAS_ANALYZER__"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            parts = key[len(prefix):].lower().split("__")
            obj = config
            for part in parts[:-1]:
                obj = obj.setdefault(part, {})
            obj[parts[-1]] = _cast(value)

    return config


def _cast(value: str) -> bool | int | float | str:
    """Attempt to cast string env var to the most specific type."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
