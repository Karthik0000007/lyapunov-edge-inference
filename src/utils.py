"""
src/utils.py
------------
Shared utilities used across the entire pipeline:

    - YAML config loading with deep-defaults merging
    - CUDA device setup and validation
    - Logging configuration (structured, colourised)
    - Checkpoint path resolution
    - SHA-256 file hash verification
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

# ── Types ─────────────────────────────────────────────────────────────────────

ConfigDict = Dict[str, Any]

# ── YAML / Config ─────────────────────────────────────────────────────────────

def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """
    Recursively merge *override* into a deep copy of *base*.
    Scalars and lists in *override* replace those in *base*;
    nested dicts are merged depth-first.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_path: str | Path,
    defaults_path: Optional[str | Path] = None,
) -> ConfigDict:
    """
    Load a YAML config file, optionally merging it on top of *defaults_path*.

    Parameters
    ----------
    config_path:
        Path to the primary YAML file (e.g., ``config/pipeline.yaml``).
    defaults_path:
        Optional path to a base-defaults YAML.  Values in *config_path*
        take priority; missing keys fall back to the defaults.

    Returns
    -------
    dict
        Fully-merged configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    yaml.YAMLError
        If either YAML file is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        primary: ConfigDict = yaml.safe_load(fh) or {}

    if defaults_path is not None:
        defaults_path = Path(defaults_path)
        if not defaults_path.exists():
            raise FileNotFoundError(f"Defaults file not found: {defaults_path}")
        with defaults_path.open("r", encoding="utf-8") as fh:
            defaults: ConfigDict = yaml.safe_load(fh) or {}
        return _deep_merge(defaults, primary)

    return primary


def load_all_configs(
    pipeline_path: str | Path = "config/pipeline.yaml",
    controller_path: str | Path = "config/controller.yaml",
    deployment_path: Optional[str | Path] = None,
) -> ConfigDict:
    """
    Load pipeline + controller configs and (optionally) apply deployment
    overrides.  Returns a single merged dict with top-level keys from all files.

    Parameters
    ----------
    pipeline_path:
        Path to ``config/pipeline.yaml``.
    controller_path:
        Path to ``config/controller.yaml``.
    deployment_path:
        Optional path to ``config/deployment.yaml``.

    Returns
    -------
    dict
        Merged configuration containing all sections.
    """
    pipeline_cfg  = load_config(pipeline_path)
    controller_cfg = load_config(controller_path)
    merged = _deep_merge(pipeline_cfg, controller_cfg)

    if deployment_path is not None:
        deploy_cfg = load_config(deployment_path)
        merged = _deep_merge(merged, deploy_cfg)

    return merged


# ── Device Setup ──────────────────────────────────────────────────────────────

def setup_device(device_str: Optional[str] = None) -> torch.device:
    """
    Resolve and validate a compute device.

    If *device_str* is ``None``, auto-selects ``cuda:0`` when CUDA is
    available, otherwise falls back to ``cpu``.

    Parameters
    ----------
    device_str:
        Optional explicit device string, e.g. ``"cuda:0"``, ``"cpu"``.

    Returns
    -------
    torch.device
        The validated device.

    Raises
    ------
    RuntimeError
        If a CUDA device was requested but CUDA is not available.
    """
    if device_str is None:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False. "
                "Check your CUDA installation or use device='cpu'."
            )
        device_index = device.index if device.index is not None else 0
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device_index} out of range "
                f"(only {torch.cuda.device_count()} device(s) available)."
            )
        # Warm-up: ensure context is created early so first-frame latency is not inflated.
        torch.cuda.set_device(device)
        _ = torch.zeros(1, device=device)

    return device


# ── Logging ───────────────────────────────────────────────────────────────────

class _ColourFormatter(logging.Formatter):
    """ANSI-colour formatter for console output."""

    _GREY    = "\x1b[38;5;245m"
    _CYAN    = "\x1b[36m"
    _YELLOW  = "\x1b[33m"
    _RED     = "\x1b[31m"
    _BOLD_RED = "\x1b[1;31m"
    _RESET   = "\x1b[0m"

    _LEVEL_COLOURS = {
        logging.DEBUG:    _GREY,
        logging.INFO:     _CYAN,
        logging.WARNING:  _YELLOW,
        logging.ERROR:    _RED,
        logging.CRITICAL: _BOLD_RED,
    }

    _FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _DATE_FMT = "%Y-%m-%dT%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = self._LEVEL_COLOURS.get(record.levelno, self._RESET)
        formatter = logging.Formatter(
            f"{colour}{self._FMT}{self._RESET}",
            datefmt=self._DATE_FMT,
        )
        return formatter.format(record)


def setup_logging(
    level: str | int = "INFO",
    log_file: Optional[str | Path] = None,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Configure the root (or named) logger with a console handler and
    an optional file handler.

    Parameters
    ----------
    level:
        Log level as string (``"DEBUG"``, ``"INFO"``, …) or ``int``.
    log_file:
        If provided, attach a plain-text ``FileHandler`` at this path.
    logger_name:
        Name for a named logger; ``None`` configures the root logger.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        use_colour = sys.stdout.isatty() and os.name != "nt"
        if use_colour:
            console_handler.setFormatter(_ColourFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
        logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


# ── Checkpoint Path Resolution ────────────────────────────────────────────────

def resolve_checkpoint(
    checkpoint_dir: str | Path,
    filename: str = "checkpoint.pt",
) -> Path:
    """
    Resolve the path to a checkpoint file inside *checkpoint_dir*.

    If *checkpoint_dir* is itself an existing ``.pt`` / ``.pth`` file the
    path is returned directly.  Otherwise the function joins *checkpoint_dir*
    with *filename* and verifies the result exists.

    Parameters
    ----------
    checkpoint_dir:
        Directory that contains checkpoint files, or a direct path to a
        ``.pt`` / ``.pth`` file.
    filename:
        Default filename when *checkpoint_dir* is a directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the checkpoint file.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    """
    path = Path(checkpoint_dir)
    if path.is_file():
        return path.resolve()

    candidate = path / filename
    if not candidate.exists():
        # Try common alternative filenames
        for alt in ("model.pt", "policy.pt", "best.pt", "latest.pt"):
            alt_candidate = path / alt
            if alt_candidate.exists():
                return alt_candidate.resolve()
        raise FileNotFoundError(
            f"No checkpoint found in '{checkpoint_dir}' "
            f"(tried '{filename}' and common alternatives)."
        )
    return candidate.resolve()


# ── SHA-256 Hash Verification ─────────────────────────────────────────────────

def compute_sha256(file_path: str | Path) -> str:
    """
    Compute the SHA-256 hex-digest of a file.

    Parameters
    ----------
    file_path:
        Path to the file to hash.

    Returns
    -------
    str
        Lowercase hex SHA-256 digest (64 characters).

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found for hashing: {file_path}")

    sha256 = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_checkpoint(
    file_path: str | Path,
    expected_hash: str,
) -> bool:
    """
    Verify a checkpoint file against a known SHA-256 hash.

    Parameters
    ----------
    file_path:
        Path to the checkpoint file.
    expected_hash:
        Expected SHA-256 hex digest (case-insensitive).

    Returns
    -------
    bool
        ``True`` if the hashes match, ``False`` otherwise.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    """
    actual = compute_sha256(file_path)
    return actual == expected_hash.lower()
