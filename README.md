# Beauty Clinic OS

AI-native consultation + CRM prototype for Taiwanese aesthetic clinics.
醫美診所 AI 諮詢 + CRM。

**Live demo:** https://beauty-clinic-os.streamlit.app

## Why

Three pain points heard first-hand from clinic owners:

1. The first-line consultation is inconsistent and feels sales-driven.
2. The CRM is paper-based (revisits, packages, payments).
3. Customers have no record of what actually improved.

## The closed loop

Standardized photo intake → quality gate → quantified skin profile →
explainable recommendations → CRM → re-measure → progress report →
outcome-aware next recommendations.

- **Capture quality gate** – blur, pose, exposure/side-lighting, face ratio,
  resolution; poor photos are rejected with a reason.
- **Skin profile** – redness, evenness, texture, spots, scored 0–100 per
  facial region, each with a measured noise band.
- **Explainable suggestions** – a rules engine names the metric, region and
  score behind every recommendation. Consultants can edit before saving.
- **CRM** – customers, visit timeline, package sessions, payments,
  overdue-revisit filter.
- **Progress report** – region-by-region comparison across visits that
  ignores changes inside the noise band.

## Try the demo

The app seeds fictional demo data on first run. The UI is in Traditional Chinese.

1. **顧客管理 (Customers)** – open 陳怡君: three visits and a 6-session package.
2. **療程比較 / 膚質進步報告 (Compare / Progress report)** – see how her skin
   metrics change across visits.
3. **新增諮詢 (New consultation)** – upload or capture a front-face photo to
   see the quality gate, skin profile and explainable suggestions.

## Design principles

- **The LLM never decides.** Treatments and scores come from the rules engine
  and the on-host CV pipeline. An optional LLM only rewrites the reason text
  into natural Mandarin, and falls back to the template on any failure.
  Only numeric inputs are sent; no images, names or medical history.
- **Mandarin-first.** Traditional Chinese UI and Taiwanese clinical terms.
- **Decision support, not diagnosis.** It assists a trained consultant.

## Quick start

Requires Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Debian/Ubuntu, first install the native libraries in `packages.txt`.
Demo data is seeded into a local SQLite DB on first run.

**Optional LLM polish:** add to `.streamlit/secrets.toml`
```toml
ANTHROPIC_API_KEY = "sk-..."
```
Without it the app runs identically using template text.

## Project layout

```
app.py        Streamlit entry + navigation
ui/           pages: customers, consultation, compare / progress report
pipeline/     landmarks, normalization, quality gate, regions, metrics
engine/       rules-based suggestions (+ optional LLM humanizer)
rules/        treatments.yaml – editable suggestion rules
db/           SQLAlchemy models + demo seed
tests/        pytest suite
scripts/      repeatability.py – measurement-noise harness
```

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

CI runs both on every push. The golden-profile regression test is skipped
off macOS because pipeline scores are platform-sensitive.

## Docs

- [Product Requirements](docs/PRD.md)
- [Technical Design](docs/TDD.md)

## Status

Working prototype. Treatments are generic and non-branded.
