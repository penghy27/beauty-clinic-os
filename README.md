# Beauty Clinic OS

**An AI-native CRM and clinical decision-support prototype for Taiwanese aesthetic clinics.**

**Live demo:** https://beauty-clinic-os.streamlit.app

Beauty Clinic OS gives first-line consultants one standardized, evidence-based consultation flow. It takes a quality-checked face photo, scores the skin, suggests treatments with the reason for each, and tracks packages, payments and revisits. On later visits it measures again and shows the customer what has actually improved.

> Built as a working prototype in Python / Streamlit. The UI is in Traditional Chinese (Taiwan) by design.

---

## Why

The project starts from three problems clinic owners described first-hand:

1. **Inconsistent, sales-driven consultations.** Consultant skill varies and turnover is high, so advice depends on who is on shift.
2. **Paper-based CRM.** Revisits, payments and pre-paid package sessions are tracked by hand.
3. **No record of what improved.** Customers finish a course of treatments with no objective evidence of progress, so they have little reason to renew.

The core promise: **junior consultants can deliver senior-level, standardized consultations.**

## The closed loop

```
photo intake → quality gate → skin profile → explainable suggestions
     ↑                                                   ↓
outcome-aware next suggestions ← progress report ← CRM (packages, payments, revisits)
```

| Step | What happens |
|------|--------------|
| **Photo intake** | Upload a front-face photo or capture one live from the webcam in the browser. |
| **Capture quality gate** | Six checks (sharpness, face size, capture resolution, head pose, exposure, lighting symmetry) give a 0–100 score. Poor photos are rejected with a plain-language reason so they can be re-shot right away. |
| **Skin profile** | Four metrics (**redness, evenness, texture, spots**) are scored 0–100 in six landmark-anchored face regions. |
| **Explainable suggestions** | A rules engine ranks treatments, and each suggestion names the metric, region and score that triggered it (e.g. "right cheek redness 72"). The consultant can edit, reorder, add or remove items before approving. |
| **CRM** | Customer records, visit timeline, package sessions (bought / used / remaining), payment status, and an overdue-revisit filter. |
| **Re-measure & progress report** | Visits are compared region by region and metric by metric, with each metric's measured noise band, so capture noise is never reported as progress. |
| **Outcome-aware next round** | Improving metrics → *continue*; flat → *maintain*; regressing → *adjust*. |

## Design principles

- **No external skin API.** Face photos are sensitive personal data (PDPA), so all analysis runs on the host using classical CV (OpenCV, MediaPipe, scikit-image, NumPy). Every number can be reproduced from the pixels.
- **Photo consistency first.** Faces are resized to a fixed scale before anything is measured. White balance is taken from the background and exposure is anchored to the face, with the same landmark-anchored regions every visit. A repeatability harness measures how much the scores drift: worst-case drift under capture variation the gate still accepts fell from 14–25 points (method v1) to under 5 (v2).
- **The LLM never decides.** Treatments and scores come from the rules engine and the CV pipeline. An *optional* Claude layer only rewrites the reason text into more natural Mandarin. It receives numbers and labels only, never images, names or medical history. With no API key, or if the call fails, the app uses the template text and works the same way.
- **Decision support, not diagnosis.** The app reports relative trends, keeps treatment copy generic and non-branded, flags cautions, and needs the consultant's approval before suggestions reach the customer.

## Tech stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| Imaging | MediaPipe Face Landmarker (478 landmarks), OpenCV, scikit-image, NumPy |
| Data | SQLite via SQLAlchemy 2.0 |
| Rules | YAML (`rules/treatments.yaml`), editable by clinic staff |
| Optional LLM | Anthropic Claude (reason-text polishing only) |
| Quality | pytest, ruff, GitHub Actions |

## Getting started

The quickest way to try it is the [live demo](https://beauty-clinic-os.streamlit.app). It is preloaded with demo data. To run it locally:

**Requirements:** Python 3.12. On Linux, also install the native libraries MediaPipe needs (listed in `packages.txt`):

```bash
sudo apt-get install -y $(cat packages.txt)
```

**Install and run:**

```bash
git clone https://github.com/penghy27/beauty-clinic-os.git
cd beauty-clinic-os
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. On first launch the app creates `data/clinic.db` and loads demo data, including a sample customer with three visits and before/after photos, so the whole loop can be explored right away.

A Dev Container config (`.devcontainer/`) is also included for GitHub Codespaces / VS Code.

**Optional: LLM polishing.** Add a key to `.streamlit/secrets.toml` (git-ignored):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

## Project structure

```
app.py               Streamlit entry point and page routing
ui/                  Pages: customers, consultation, compare / progress report
pipeline/            Imaging pipeline
  process.py           process_photo(): the single entry point
  landmarks.py         face detection + 478 landmarks + head pose
  quality.py           capture quality gate
  normalize.py         background-anchored white balance, exposure anchor
  regions.py           six landmark-anchored region masks
  metrics.py           redness / evenness / texture / spots + noise bands
  viz.py               skin-map overlays
engine/
  suggest.py           rules-based, outcome-aware suggestion engine
  humanize.py          optional LLM rewrite of reason text
rules/treatments.yaml  treatment rules
db/                  SQLAlchemy models (6 tables) and demo seed data
models/              MediaPipe face_landmarker.task
sample_photos/       demo photos for the sample customer
scripts/repeatability.py  measurement-reliability harness
tests/               pytest suite
docs/                PRD.md, TDD.md
```

## Testing

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The suite covers metric behaviour on synthetic images, quality-gate rejections, normalization properties, rule firing and noise-band trend logic, the LLM fallback path, a closed-loop integration test, and Streamlit `AppTest` smoke tests for every page. A golden-profile regression test fixes the full pipeline's output on the sample photos. It runs only on macOS, because native-library differences between platforms shift some count-based scores. CI runs ruff and pytest on every push to `main` and on every pull request.

To measure how much the scores drift under capture variation:

```bash
python scripts/repeatability.py sensitivity
```

## Documentation

- [Product Requirements Document](docs/PRD.md): problem, users, closed loop, wedge and retention flywheel
- [Technical Design Document](docs/TDD.md): pipeline architecture, metric formulas, quality gate, data model, reliability

## Disclaimer

This is a prototype for clinical **decision support**. It assists a trained consultant's judgment and does not replace it. It is not a medical device and does not provide a diagnosis.
