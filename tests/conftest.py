"""Shared fixtures for the test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.landmarks import LandmarkResult  # noqa: E402
from pipeline.regions import RegionPatch  # noqa: E402


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Tests must never reach the LLM API, even with a key configured."""
    from engine import humanize
    monkeypatch.setattr(humanize, "refine", lambda suggestions: None)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point db.models at a throwaway SQLite file and reset the engine."""
    from db import models, seed
    data_dir = tmp_path / "data"
    monkeypatch.setattr(models, "DATA_DIR", data_dir)
    monkeypatch.setattr(models, "PHOTO_DIR", data_dir / "photos")
    monkeypatch.setattr(models, "DB_PATH", data_dir / "clinic.db")
    monkeypatch.setattr(models, "_engine", None)
    monkeypatch.setattr(models, "_SessionFactory", None)
    # seed binds PHOTO_DIR at import time; keep its writes in the tmp dir
    # too (PROJECT_ROOT stays real so sample photos are still found)
    monkeypatch.setattr(seed, "PHOTO_DIR", data_dir / "photos")
    yield models
    if models._engine is not None:
        models._engine.dispose()


def make_image(h: int = 200, w: int = 200,
               bgr: tuple[int, int, int] = (165, 170, 180),
               noise: float = 0.0, seed: int = 7) -> np.ndarray:
    """A uniform BGR image, optionally with additive Gaussian noise."""
    img = np.full((h, w, 3), bgr, dtype=np.float32)
    if noise > 0:
        rng = np.random.default_rng(seed)
        img += rng.normal(0.0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def center_region(h: int = 200, w: int = 200) -> dict[str, RegionPatch]:
    """One elliptical test region in the middle of the frame."""
    return {"test": RegionPatch("test", (w / 2, h / 2), w * 0.3, h * 0.3)}


def fake_landmarks(image: np.ndarray, *, face_ratio: float = 0.6,
                   yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                   native_face_width: float | None = None,
                   face_box: tuple[int, int, int, int] | None = None,
                   ) -> LandmarkResult:
    """A LandmarkResult without running MediaPipe (quality-gate tests)."""
    h, w = image.shape[:2]
    box = face_box or (int(w * 0.2), int(h * 0.15),
                       int(w * 0.6), int(h * 0.7))
    return LandmarkResult(
        landmarks_px=np.zeros((478, 2)),
        image_shape=(h, w),
        face_box=box,
        face_ratio=face_ratio,
        yaw=yaw, pitch=pitch, roll=roll,
        native_face_width=(native_face_width
                           if native_face_width is not None
                           else float(box[2])),
    )
