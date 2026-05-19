# Technical Design Document — Beauty Clinic OS

A self-built, AI-native CRM and clinical decision-support prototype for Taiwanese aesthetic clinics. Built in Python/Streamlit.

## 1. System Overview

Beauty Clinic OS is a closed-loop prototype with four components and no external skin-analysis dependency:

- **UI** — Streamlit, pure-Python, Mandarin-labelled (`app.py`, `ui/*.py`).
- **Data** — SQLite via SQLAlchemy 2.0 ORM, a six-table model (`db/models.py`).
- **Imaging pipeline** — classical computer vision orchestrated by `pipeline/process.py`.
- **Suggestion engine** — a rules-based, outcome-aware engine reading `rules/treatments.yaml` (`engine/suggest.py`).

**No external skin API — a deliberate design decision.** Taiwanese customers' face photos are sensitive personal data under the Personal Data Protection Act; transmitting them to overseas (notably China-based) skin-analysis APIs is a compliance and reviewer-optics risk, and a thin wrapper over such an API is not panel-ready. Therefore the entire skin-analysis core is 100% self-built from OpenCV, MediaPipe, scikit-image and NumPy. Every number is explainable and reproducible from the pixels, and all processing stays on the host. The CV stack is heavier to build than an API call — and that is exactly the point: the build effort is the evidence that this is not a wrapper.

## 2. Imaging Pipeline Architecture

`process_photo()` in `pipeline/process.py` runs one fixed chain; the UI only ever calls this entry point. It returns a `PipelineResult` carrying landmarks, quality report, normalized image, white-balance info, region patches and the Skin Profile.

1. **Face detection + 478 landmarks** (`landmarks.py`) — the MediaPipe Face Landmarker (Tasks API, `face_landmarker.task`, single-image mode, `num_faces=1`) returns 478 mesh landmarks in pixel coordinates plus a 4×4 facial transformation matrix. Head pose (yaw/pitch/roll, degrees, 0 = frontal) is read directly from the matrix' rotation block and folded into [-90°, 90°]. If no face is detected the pipeline stops and returns `ok=False` — no skin metric is ever computed on a photo without a confirmed single face.
2. **Quality gate** (`quality.py`) — `assess()` scores capture quality and explains every check (see §3).
3. **Gray-world white balance** (`normalize.py`) — `white_balance()` removes lighting/device colour cast (see §3). Both raw and normalized images are stored.
4. **Landmark-anchored region patches** (`regions.py`) — `build_regions()` builds six fixed elliptical patches anchored to stable mesh landmarks (see §3).
5. **Per-region metric extraction** (`metrics.py`) — `analyze()` computes four metrics inside each region mask (see §4).
6. **Skin Profile** — the flattened list of `ProfileEntry(region, metric, value, sub_score)` rows, plus an overall Skin Health Index (`overall_score()`, the mean sub-score).

## 3. Depth Area — Photo-Consistency Control

The technical depth area is **photo-consistency control**: the foundation that makes cross-visit longitudinal comparison clinically meaningful. A redness or evenness delta between two visits is only valid if both photos were captured and measured under comparable conditions. Three mechanisms enforce this.

**Capture Quality Score** (`quality.py`) — `assess()` runs five explainable checks and produces a 0–100 score (start at 100, subtract 26 per fail / 9 per warn; a photo passes at score ≥ 60 with no fail):

- **Sharpness** — variance of the Laplacian over the face crop; fail < 55, warn < 110.
- **Face size** — face-box height ÷ image height; fail < 0.22, warn < 0.38.
- **Head pose** — `max(|yaw|, |pitch|)` from the transformation matrix; fail > 20°, warn > 12° (roll warn > 12°).
- **Exposure** — mean face-crop brightness with blown-pixel clip fraction; fail outside ~55–230, warn outside ~85–205.
- **Lighting symmetry** — absolute left/right mean-brightness gap, detecting side lighting; fail > 42, warn > 22.

Each check yields a pass/warn/fail status and a plain-language Mandarin message, so the consultant sees exactly why a photo was rejected and re-shoots fast. The gate turns a risk (unreliable input) into a feature (standardized intake).

**Colour normalization** (`normalize.py`) — gray-world white balance assumes the scene average is neutral gray and rescales each BGR channel so its mean matches the global mean. The per-channel gains and a plain description of the original cast (warm/cool/neutral) are recorded. Skin-colour metrics (redness, evenness) are otherwise meaningless across visits when lighting or device colour shifts.

**Fixed landmark-anchored region masks** (`regions.py`) — six regions (forehead, left/right cheek, nose/T-zone, chin, under-eye) are each an ellipse centred on the centroid of a fixed set of mesh landmark indices, with radii scaled as a fixed fraction of the face-box width. Because anchors and scaling are identical every visit, the *same skin area* is sampled regardless of how large the face appears in frame.

Together: standardized intake + neutral colour + identical sampling regions mean that a between-visit metric difference reflects the skin, not the photo. This is why photo-consistency is the prerequisite for the outcome-tracking loop.

## 4. Explainable Metrics

`metrics.py` computes four classical-CV metrics per region. Each has a documented formula and a 0–100 sub-score (100 = best). No black box, no model weights — every value is reproducible from the pixels.

- **Redness** — mean CIELAB a\* offset from neutral 128 inside the region. Higher = redder. Sub-score range: 8.0 (good) → 34.0 (bad).
- **Evenness** — standard deviation of CIELAB L\* inside the region; lower std = more uniform tone. Range: 6.0 → 30.0.
- **Texture** — variance of the Laplacian over the grayscale region; a surface-roughness proxy. Range: 15.0 → 260.0.
- **Spots** — count of dark blemish blobs found by black-hat morphology (15×15 elliptical kernel) followed by thresholding and connected-component analysis (blobs ≥ 6 px). Range: 0 → 14.

`_sub_score()` linearly maps each raw value to 0–100 against its `(good, bad)` range, clamped. `by_metric()` averages sub-scores per metric across regions; `overall_score()` is the mean across all entries (the Skin Health Index). Choosing transparent, reproducible classical-CV formulas over an opaque model is itself a reliability decision.

## 5. Suggestion Engine

`engine/suggest.py` generates ranked treatment suggestions and is explainable, editable and outcome-aware.

- **Rules** — `rules/treatments.yaml` holds eight rules across the four metrics, each with `metric`, `max_sub_score` (severity tier), `priority`, `treatment`, `detail` and `caution`. A rule fires when a metric's average sub-score falls at or below `max_sub_score`; only the most severe matching rule per metric fires. Treatments are generic and non-branded — decision support, not a prescription — and clinic staff can edit the YAML.
- **Explainable** — each `Suggestion` records the triggering `metric`, `region`, `sub_score` and a data-grounded `reason` naming the metric average and the worst region with its score (e.g. "泛紅平均膚質分 58，最弱區為右頰（51 分）"). Every suggestion points to the exact data that triggered it, so it never feels like an upsell — the wedge that fixes the inconsistent-consultation pain point.
- **Editable** — suggestions serialise via `to_json()`/`from_json()`. The `recommendations` table stores `generated_json` (engine output) and `edited_json` (consultant-approved copy) separately, with a `draft`/`approved` status — a visible human-in-the-loop edit.
- **Outcome-aware** — when a previous visit is supplied, `_trend_note()` compares the current per-metric average to the last visit's and appends a trend note: improvement ≥ 5 sub-score points ("療程有效，建議延續同方案"), regression ≤ −5 ("改善有限，建議調整療程強度或方式"), or roughly flat ("建議維持並持續追蹤"). `positive_notes()` surfaces improved metrics for the customer-facing progress report. This closes the loop: measure → suggest → treat → re-measure → outcome-aware next round.

## 6. Data Model & Workflow Integration

`db/models.py` defines six tables carrying the closed loop:
`customers → packages / visits → photos → skin_profiles → recommendations`.

- **customers** — name, sex, birth year, phone, skin note, consent timestamp.
- **packages** — pre-paid treatment package: `total_sessions`, `sessions_used`, `price`, `is_paid`; a `sessions_left` property replaces the clinic's paper session tally.
- **visits** — per visit: customer, optional package, `visit_date`, `consultant`, `notes`, `next_revisit_date` (drives the revisit-overdue CRM filter).
- **photos** — `raw_path`, `normalized_path`, `quality_score`, `quality_json`, `wb_method`.
- **skin_profiles** — one quantified metric for one region in **long/tidy format** (`region`, `metric`, `value`, `sub_score`, `method_version`).
- **recommendations** — `generated_json` vs `edited_json`, `status`, `consultant_notes`.

The long-format Skin Profile is the key design choice for longitudinal comparison: filtering by `(region, metric)` lines values up across visits automatically, making cross-visit diffing and charting trivial with no schema change. `method_version` (`"v1"`) tags the metric algorithm so only comparable profiles are diffed. CRM integration is native: packages track session counts and payment, visits track revisit dates and consultant, and the same `customer → visits → photos → profiles` chain feeds both the CRM timeline and the outcome-aware engine.

## 7. Reliability & Explainability Posture

The prototype reports **relative trends, not absolute diagnosis**. It is framed as decision support for the front-line consultant, not an automated medical verdict. Concretely:

- Metrics are presented as 0–100 sub-scores and between-visit deltas, contextualised by the Capture Quality Score; comparisons are only valid when capture quality is comparable.
- Every metric formula is documented and reproducible — no opaque model, no external API call, all processing on-host.
- Treatment copy is generic and conservative, every `caution` field flags contraindications, and the consultant must review and approve suggestions (`generated_json` → `edited_json`) before they reach the customer.
- The quality gate fails fast and explains itself, so unreliable inputs never silently become clinical records.

This posture — explainable, reproducible, conservative, human-in-the-loop, privacy-preserving — is the technical answer to why this approach fits clinic operations and is panel-ready rather than a thin wrapper.
