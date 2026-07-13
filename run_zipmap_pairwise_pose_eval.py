#!/usr/bin/env python3
"""Evaluate stateless pairwise ZipMap pose estimation on a stereo sequence.

Two diagnostic modes are supported:

1) mono_pair
   Run ZipMap independently on [L_t, L_{t+1}], accumulate the predicted
   left-camera relative poses, and report both SE(3)- and Sim(3)-aligned
   trajectory errors. No stereo scale is used.

2) stereo_pair
   Run ZipMap independently on [L_t, R_t, L_{t+1}, R_{t+1}]. Estimate one
   metric scale factor from the two predicted stereo baselines, apply it only
   to the temporal left-camera translation, accumulate the trajectory, and
   report SE(3)- and Sim(3)-aligned errors.

The model is called separately for every adjacent pair and no returned TTT
state is reused between calls. No Gaussian generation, fusion, or plots are
performed.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "t"}:
        return True
    if value in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def abs_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_images(root: Path, recursive: bool) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    images = sorted(
        path for path in iterator if path.is_file()