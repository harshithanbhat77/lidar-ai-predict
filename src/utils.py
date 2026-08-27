from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any


LOGGER_NAME = "lidar_ai_predict"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    return logger


def ensure_parent_dir(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()
