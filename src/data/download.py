"""Dataset download helpers.

Currently only LFW is auto-downloadable (CASIA-WebFace requires manual license).
"""
from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger()

_LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
_LFW_PAIRS_URL = "http://vis-www.cs.umass.edu/lfw/pairs.txt"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"Already exists, skipping: {dest}")
        return
    logger.info(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def download_lfw(root: str | Path = "data/raw/lfw") -> Path:
    """Download + extract LFW into ``root`` (returns the path)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    archive = root / "lfw.tgz"
    pairs_file = root / "pairs.txt"

    _download(_LFW_URL, archive)
    _download(_LFW_PAIRS_URL, pairs_file)

    # Extract if not already done
    extracted_marker = root / "lfw"
    if not extracted_marker.exists():
        logger.info(f"Extracting {archive} ...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(root)

    # The tar usually extracts to <root>/lfw/<id>/...; flatten if so so that
    # <root>/<id>/... is consistent with the LFWPairs reader.
    nested = root / "lfw"
    if nested.exists() and nested.is_dir() and any(p.is_dir() for p in nested.iterdir()):
        for sub in nested.iterdir():
            target = root / sub.name
            if not target.exists():
                shutil.move(str(sub), str(target))
        try:
            nested.rmdir()
        except OSError:
            pass

    logger.success(f"LFW ready at {root}")
    return root
