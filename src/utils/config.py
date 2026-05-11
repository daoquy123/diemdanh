"""Tiny YAML config helper built on OmegaConf for clean overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file as an OmegaConf DictConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Config root must be a mapping, got {type(cfg)}")
    return cfg


def dump_config(cfg: DictConfig, path: str | Path) -> None:
    """Persist resolved config next to checkpoints (for reproducibility)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def merge_overrides(cfg: DictConfig, overrides: Iterable[str]) -> DictConfig:
    """Apply CLI-style overrides like ``train.batch_size=64``.

    Returns a new merged DictConfig.
    """
    if not overrides:
        return cfg
    override_cfg = OmegaConf.from_dotlist(list(overrides))
    return OmegaConf.merge(cfg, override_cfg)  # type: ignore[return-value]


def to_container(cfg: DictConfig) -> dict[str, Any]:
    """Convert config to a plain Python dict (for json/pickle)."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
