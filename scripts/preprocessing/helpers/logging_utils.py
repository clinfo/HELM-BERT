"""Shared logging setup for pipeline scripts.

Each script gets a timestamped log directory under ``<HELM_DATASETS>/logs/``
plus stdout streaming. The same logger is returned everywhere so scripts
don't accidentally split output across multiple handlers.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from helpers.paths import LOG_DIR

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logger(stage: str, name: str = "pipeline") -> tuple[logging.Logger, Path]:
    """Create a stage-specific logger writing to both stdout and a file.

    Returns (logger, log_dir). Calling twice in the same process replaces
    handlers — this is intentional so scripts can be re-imported / re-run
    in notebooks without log duplication.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_DIR / f"{stage}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{stage}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("Log file: %s", log_file)
    return logger, log_dir
