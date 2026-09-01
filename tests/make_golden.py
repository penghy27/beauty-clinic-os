"""Regenerate tests/fixtures/golden_profiles.json from the sample photos.

Run only when a pipeline change is intentional (and bump METHOD_VERSION):
    python tests/make_golden.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.process import process_photo  # noqa: E402

OUT = Path(__file__).resolve().parent / "fixtures" / "golden_profiles.json"


def main() -> None:
    golden = {}
    for path in sorted((ROOT / "sample_photos").glob("hero_v*.jpg")):
        result = process_photo(cv2.imread(str(path)))
        if not result.ok:
            raise SystemExit(f"{path.name}: no face detected")
        golden[path.name] = {
            "quality_passed": result.quality.passed,
            "quality_score": result.quality.score,
            "overall": result.overall,
            "profile": {f"{e.region}.{e.metric}": e.sub_score
                        for e in result.profile},
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(golden, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT} ({len(golden)} photos)")


if __name__ == "__main__":
    main()
