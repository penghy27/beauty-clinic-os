# Technical Design Document — Beauty Clinic OS

A self-built, AI-native CRM and clinical decision-support prototype for Taiwanese aesthetic clinics. Built in Python/Streamlit.

## 1. System Overview

Beauty Clinic OS is a closed-loop prototype with four components and no external skin-analysis dependency:

- **UI** — Streamlit, pure-Python, Mandarin-labelled (`app.py`, `ui/*.py`). The consultation page accepts file upload and live in-browser webcam capture (`st.camera_input`).
- **Data** — SQLite via SQLAlchemy 2.0 ORM, six tables (`db/models.py`).
- **Imaging pipeline** — classical CV orchestrated by `pipeline/process.py`.
- **Suggestion engine** — rules-based, outcome-aware, reading `rules/treatments.yaml` (`engine/suggest.py`).

**No external skin API — by design.** Customer face photos are PDPA-sensitive, so the skin-analysis core is self-built from OpenCV, MediaPipe, scikit-image and NumPy — every number reproducible from the pixels, all processing on-host.

## 2. Imaging Pipeline Architecture

`process_photo()` in `pipeline/process.py` runs one fixed chain; the UI only calls this entry point. It returns a `PipelineResult` carrying landmarks, quality report, normalized image, region patches and the Skin Profile.

1. **Face detection + 478 landmarks** (`landmarks.py`) — MediaPipe Face Landmarker (Tasks API, `face_landmarker.task`) returns 478 mesh landmarks plus a 4×4 transformation matrix; head pose (yaw/pitch/roll) is read from its rotation block. No face → `ok=False`; no skin metric runs without a confirmed single face.
2. **Canonical face scale** — the image is resized so the face box is a fixed 384 px wide before anything is measured. Texture kernels and blob-area thresholds therefore sample the same physical skin patch whether the photo came from a phone or a 640×480 webcam — without this, resolution alone shifted texture sub-scores by 25 points.
3. **Quality gate → background-anchored white balance + face-luminance anchor → six landmark-anchored region patches → per-region metric extraction** (`quality.py`, `normalize.py`, `regions.py`, `metrics.py`) — see §3 and §4. Raw and normalized images are both stored.
4. **Skin Profile** — flattened `ProfileEntry(region, metric, value, sub_score)` rows plus `overall_score()` (mean sub-score, the Skin Health Index), stamped with `method_version`.

## 3. Depth Area — Photo-Consistency Control

The technical depth area is **photo-consistency control** — the foundation for clinically meaningful cross-visit comparison. A delta between visits is only valid if both photos were captured and measured under comparable conditions. Four mechanisms enforce this, and a repeatability harness (`scripts/repeatability.py`) measures how well they work: it perturbs reference photos with brightness, colour-cast, resolution and JPEG changes and reports per-metric sub-score drift. Worst-case drift under gate-accepted variation fell from **14–25 points to under 5** between method v1 and v2.

**Canonical face scale** (`process.py`) — all measurement happens with the face box resized to a fixed 384 px width (§2), removing the analysis-scale dependence of texture and blob metrics.

**Capture Quality Score** (`quality.py`) — six explainable checks, 0–100 (−26 per fail, −9 per warn; passes at ≥ 60 with no fail):

- **Sharpness** — Laplacian variance over the canonical face crop; fail < 45, warn < 100.
- **Face size** — face-box height ÷ image height; fail < 0.22, warn < 0.38.
- **Capture resolution** — native face-box width; fail < 220 px, warn < 340 px. Upscaling cannot restore detail the sensor never recorded, so a too-small face is flagged as a resolution problem instead of being mislabelled "blurry".
- **Head pose** — `max(|yaw|, |pitch|)`; fail > 20°, warn > 12° (roll warn > 12°).
- **Exposure** — mean face-crop brightness with blown-pixel clip; fail outside ~55–230, warn outside ~85–205.
- **Lighting symmetry** — left/right mean-brightness gap; fail > 42, warn > 22.

Each check yields pass/warn/fail and a plain-Mandarin message, so the consultant sees why a photo was rejected and re-shoots fast.

**Colour and exposure normalization** (`normalize.py`) — chromatic gray-world gains are estimated from the **background** (pixels outside the face box): the background carries the ambient cast, is constant for a given clinic room, and — unlike the face — is not the signal being measured (gray-world on face pixels would force mean skin tone to neutral and erase redness itself). Luminance is then anchored so mean face brightness lands on a fixed target, making L\*-based metrics exposure-invariant. Gains and the original cast are recorded.

**Fixed landmark-anchored region masks** (`regions.py`) — six regions (forehead, left/right cheek, nose/T-zone, chin, under-eye) are ellipses centred on fixed mesh-landmark centroids, with radii scaled to face-box width. Identical anchoring every visit means the *same skin area* is sampled regardless of face size in frame.

## 4. Explainable Metrics

`metrics.py` computes four classical-CV metrics per region at the canonical scale; each has a documented formula and a 0–100 sub-score (100 = best). No black box, no model weights.

- **Redness** — mean CIELAB a\* offset from neutral 128, excluding pixels that were saturated in the capture (a clipped channel has already destroyed the hue there). Sub-score range 7.0 → 30.0.
- **Evenness** — coefficient of variation of CIELAB L\* (std ÷ region mean; lower = more uniform). Self-normalising per region, so exposure and lighting geometry cancel out. Range 5.0 → 30.0.
- **Texture** — band-pass Laplacian variance (Gaussian σ=1 pre-blur), contrast-normalised to the region's brightness. The pre-blur suppresses pixel-level frequencies that encode capture sharpness rather than skin. Range 2.0 → 22.0.
- **Spots** — count of dark blobs from black-hat morphology (9×9 at canonical scale) + threshold + connected components (blobs ≥ 2 px). Range 0 → 14.

`_sub_score()` linearly maps each raw value to 0–100 against its `(good, bad)` range; `overall_score()` is the mean of all entries. `NOISE_BAND` records each metric's empirically measured noise floor (±4/±3/±6/±5 sub-score points) — the single source of truth for what counts as real change.

## 5. Suggestion Engine

`engine/suggest.py` generates ranked treatment suggestions — explainable, editable, outcome-aware.

- **Rules** — `rules/treatments.yaml` holds eight rules across the four metrics (`metric`, `max_sub_score`, `priority`, `treatment`, `detail`, `caution`). A rule fires when a metric's average ≤ `max_sub_score`; only the most severe matching rule per metric fires. Treatments are generic, non-branded, clinic-editable.
- **Explainable** — each `Suggestion` records the triggering `metric`, `region`, `sub_score` and a data-grounded `reason` naming the metric average and the worst region — evidence, not an upsell.
- **Editable** — suggestions serialise via `to_json()`/`from_json()`; the `recommendations` table stores `generated_json` vs `edited_json` separately with a `draft`/`approved` status — visible human-in-the-loop edit.
- **Outcome-aware** — when a prior visit is supplied, `_trend_note()` compares per-metric averages against that metric's measured noise band: improvement beyond the band → continue; regression beyond it → adjust; inside it → maintain (noise is never sold as progress). `positive_notes()` surfaces improved metrics for the customer-facing progress report.
- **Optional LLM polishing** — `engine/humanize.py` populates `reason_humanized`; the UI displays `reason_humanized or reason`. The rules engine still chooses every treatment and every number — the model only rewrites display copy. On any failure the field stays empty and the UI falls back to the template. Inputs are numeric only — no images, names or medical history leave the host.

## 6. Data Model & Workflow Integration

`db/models.py` defines six tables: `customers → packages / visits → photos → skin_profiles → recommendations`.

- **customers** — name, sex, birth year, phone, skin note, consent timestamp.
- **packages** — `total_sessions`, `sessions_used`, `price`, `is_paid`; `sessions_left` replaces the paper session tally.
- **visits** — customer, optional package, `visit_date`, `consultant`, `notes`, `next_revisit_date` (drives the overdue-revisit filter).
- **photos** — `raw_path`, `normalized_path`, `quality_score`, `quality_json`, `wb_method`.
- **skin_profiles** — one metric for one region in **long/tidy format** (`region`, `metric`, `value`, `sub_score`, `method_version`).
- **recommendations** — `generated_json` vs `edited_json`, `status`, `consultant_notes`.

Long-format `skin_profiles` is the key design choice for longitudinal comparison: filtering by `(region, metric)` lines values up across visits with no schema change. `method_version` is enforced at every comparison point: the compare view withholds improve/regress verdicts across versions, and the suggestion engine starts a fresh baseline instead of trend-comparing against an older method's scale.

## 7. Reliability & Explainability Posture

The prototype reports **relative trends, not absolute diagnosis** — decision support, not a medical verdict.

- Metrics are 0–100 sub-scores and between-visit deltas, contextualised by the Capture Quality Score; comparisons are only valid when capture quality is comparable.
- **Measured, not asserted**: the repeatability harness quantifies per-metric drift under capture variation, and those numbers — not guesses — set the tolerance bands used everywhere a delta is judged. Redness/evenness/spots stay within ±3.2/±1.7/±4.8 points across brightness ±20%, ±6% colour casts, 1280 px downsampling and JPEG recompression; low-resolution captures that would degrade texture are rejected or flagged by the gate rather than silently compared.
- Every metric formula is documented and reproducible — no opaque model, no external skin API, all processing on-host. The optional LLM polishing layer (§5) sees only numeric inputs and never alters treatments or scores.
- Treatment copy is generic and conservative; `caution` fields flag contraindications; the consultant must approve (`generated_json` → `edited_json`) before suggestions reach the customer.

## 8. Tests & CI

A pytest suite (44 tests, GitHub Actions on every push) covers: metric monotonicity on synthetic images (redder → worse redness, blobs → more spots), quality-gate rejection of blurred/dark/side-lit/turned/low-resolution captures, background-anchored normalisation properties, rule firing and noise-band trend logic, LLM-fallback contract (an API failure can never block a consultation), a golden-profile regression pinning the full pipeline's output on the sample photos (method drift turns CI red), the canonical-scale invariant itself (a 1280 px re-capture must stay inside every noise band), and a closed-loop integration test from seeding through outcome-aware recommendations, plus Streamlit `AppTest` smoke tests for every page.
