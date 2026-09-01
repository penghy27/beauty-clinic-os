"""Measurement-reliability harness for the imaging pipeline.

Two modes, both offline CLI (never imported by the UI):

* ``burst`` — run the pipeline over a folder of photos of the same face
  taken in one session; report per (region, metric) spread. This is the
  true test-retest number when real burst data is available.
* ``sensitivity`` — apply controlled capture perturbations (brightness,
  colour cast, downsampling, JPEG recompression) to reference photos and
  report how far each metric drifts from the unperturbed baseline. The
  perturbations model variation the quality gate still accepts, so the
  observed drift is the noise floor a between-visit delta must exceed
  before it means anything.

Usage:
    python scripts/repeatability.py sensitivity [--out docs/REPEATABILITY.md]
    python scripts/repeatability.py burst PATH/TO/FOLDER [--out ...]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import METHOD_VERSION  # noqa: E402
from pipeline.metrics import METRICS, by_metric  # noqa: E402
from pipeline.process import process_photo  # noqa: E402

DEFAULT_IMAGES = sorted(
    (Path(__file__).resolve().parent.parent / "sample_photos").glob("hero_v*.jpg")
)


# --- perturbations -----------------------------------------------------------

def _brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _cast(img: np.ndarray, gain_b: float, gain_r: float) -> np.ndarray:
    out = img.astype(np.float32)
    out[:, :, 0] *= gain_b
    out[:, :, 2] *= gain_r
    return np.clip(out, 0, 255).astype(np.uint8)


def _downsample(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return img.copy()
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img.copy()
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# Variation the quality gate would still accept — mixed capture conditions
# a clinic will realistically produce between two visits.
PERTURBATIONS: list[tuple[str, callable]] = [
    ("亮度 ×0.8", lambda im: _brightness(im, 0.8)),
    ("亮度 ×1.2", lambda im: _brightness(im, 1.2)),
    ("暖色偏 (+6% R, −6% B)", lambda im: _cast(im, 0.94, 1.06)),
    ("冷色偏 (−6% R, +6% B)", lambda im: _cast(im, 1.06, 0.94)),
    ("降採樣 長邊 1280px", lambda im: _downsample(im, 1280)),
    ("降採樣 長邊 640px (webcam)", lambda im: _downsample(im, 640)),
    ("JPEG 品質 70", lambda im: _jpeg(im, 70)),
]


# --- runners -----------------------------------------------------------------

def _profile_by_metric(img: np.ndarray) -> tuple[dict[str, float] | None, float]:
    """Run the pipeline; return (per-metric sub-score averages, quality score)."""
    result = process_photo(img)
    if not result.ok or not result.profile:
        return None, 0.0
    q = result.quality.score if result.quality else 0.0
    return by_metric(result.profile), q


def run_sensitivity(images: list[Path]) -> list[str]:
    lines = [
        f"# 量測敏感度報告（method {METHOD_VERSION}）",
        "",
        f"*產生日期 {date.today()}；每格為該指標平均膚質分相對未擾動基準的漂移"
        "（sub-score 分，正負皆列）。*",
        "",
    ]
    # metric -> list of |drift| across all images & perturbations
    drift_pool: dict[str, list[float]] = {m: [] for m in METRICS}

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            lines.append(f"**{path.name}**：讀取失敗，跳過。")
            continue
        base, base_q = _profile_by_metric(img)
        if base is None:
            lines.append(f"**{path.name}**：基準照未偵測到人臉，跳過。")
            continue

        lines += [f"## {path.name}（基準品質分 {base_q:.0f}）", ""]
        header = "| 擾動 | 品質分 | " + " | ".join(METRICS) + " |"
        lines += [header,
                  "|" + "---|" * (len(METRICS) + 2)]
        base_row = " | ".join(f"{base.get(m, float('nan')):.1f}" for m in METRICS)
        lines.append(f"| 基準（絕對分數） | {base_q:.0f} | {base_row} |")

        for name, fn in PERTURBATIONS:
            got, q = _profile_by_metric(fn(img))
            if got is None:
                lines.append(f"| {name} | — | 未偵測到人臉 |")
                continue
            cells = []
            for m in METRICS:
                if m in got and m in base:
                    d = got[m] - base[m]
                    drift_pool[m].append(abs(d))
                    cells.append(f"{d:+.1f}")
                else:
                    cells.append("—")
            lines.append(f"| {name} | {q:.0f} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += _summary(drift_pool,
                      "各指標漂移彙總（所有照片 × 所有擾動）",
                      "建議容差帶 = 最大漂移向上取整；跨次變化小於此值視為量測雜訊。")
    return lines


def run_burst(folder: Path) -> list[str]:
    paths = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    lines = [
        f"# 重測穩定性報告（method {METHOD_VERSION}）",
        "",
        f"*產生日期 {date.today()}；資料夾 `{folder}`（{len(paths)} 張，"
        "應為同人同場景連拍）。*",
        "",
    ]
    per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
    rows = []
    for path in paths:
        img = cv2.imread(str(path))
        got, q = (None, 0.0) if img is None else _profile_by_metric(img)
        if got is None:
            rows.append(f"| {path.name} | — | 讀取失敗或未偵測到人臉 |")
            continue
        for m in METRICS:
            if m in got:
                per_metric[m].append(got[m])
        cells = " | ".join(f"{got.get(m, float('nan')):.1f}" for m in METRICS)
        rows.append(f"| {path.name} | {q:.0f} | {cells} |")

    lines += ["| 照片 | 品質分 | " + " | ".join(METRICS) + " |",
              "|" + "---|" * (len(METRICS) + 2)]
    lines += rows + [""]

    spread = {m: [abs(v - statistics.mean(vals)) for v in vals]
              for m, vals in per_metric.items() if len(vals) >= 2}
    lines += _summary(spread,
                      "各指標離散度彙總（|與平均值的差|）",
                      "std 與最大偏差直接反映連拍下的量測雜訊。")
    return lines


def _summary(pool: dict[str, list[float]], title: str, note: str) -> list[str]:
    lines = [f"## {title}", "",
             "| 指標 | 樣本數 | 平均漂移 | 最大漂移 | std |",
             "|---|---|---|---|---|"]
    for m in METRICS:
        vals = pool.get(m, [])
        if not vals:
            lines.append(f"| {m} | 0 | — | — | — |")
            continue
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        lines.append(
            f"| {m} | {len(vals)} | {statistics.mean(vals):.1f} "
            f"| {max(vals):.1f} | {std:.1f} |"
        )
    lines += ["", f"*{note}*", ""]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("sensitivity", help="合成擾動敏感度分析")
    s.add_argument("images", nargs="*", type=Path,
                   default=None, help="參考照（預設 sample_photos/hero_v*.jpg）")
    s.add_argument("--out", type=Path, default=None)

    b = sub.add_parser("burst", help="同場景連拍重測穩定性")
    b.add_argument("folder", type=Path)
    b.add_argument("--out", type=Path, default=None)

    args = ap.parse_args()
    if args.mode == "sensitivity":
        images = args.images or DEFAULT_IMAGES
        lines = run_sensitivity(list(images))
    else:
        lines = run_burst(args.folder)

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"報告已寫入 {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
