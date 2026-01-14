"""Utility helpers for shared configuration lookups."""

from __future__ import annotations
from typing import Any, Dict


def resolve_fps_from_config(cfg: Dict[str, Any] | None, default: float = 30.0) -> float:
    """Return the global FPS value from configuration."""
    if not isinstance(cfg, dict):
        return float(default)

    fps_value = cfg.get("fps")
    if fps_value is not None:
        try:
            fps_val = float(fps_value)
            if fps_val > 0.0:
                return fps_val
        except (TypeError, ValueError):
            pass

    return float(default)
