# Beauty Clinic OS — Product Requirements Document

**An AI-native CRM and clinical decision-support system for Taiwanese aesthetic clinics.**

*Status: working prototype.*

---

## 1. Problem — three first-hand pain points

Clinic owners described three problems the existing software market does not solve:

1. **The first-line consultation is inconsistent and feels sales-driven.** Consultant quality varies and turnover is high, so the same customer can get different advice on different shifts. Because consultants are paid on commission, the experience often feels like a pushed package rather than a real diagnosis. Owners want this touchpoint **standardized and made scientific** — it is what drives retention and revenue.
2. **The CRM is paper-based.** Revisits, payments and the session count of pre-paid packages are tracked by hand, giving the owner no operational visibility.
3. **Customers have no record of what actually improved.** A finished course of treatments produces no objective evidence of progress, so there is no compelling reason to renew.

These are voice-of-customer, not assumptions, and they are the answer to *why an AI-native CRM fits clinic operations*: the work of a clinic **is** consultation, scheduling, packages and outcomes — each pain point above is one of those workflows breaking down.

## 2. Target users

| Role | Who | What they get |
|------|-----|---------------|
| **Primary user** | First-line **consultant** | A guided, standardized consultation flow. A junior runs the exact same evidence-based process as a senior — quality stops depending on the individual and survives turnover. |
| **Economic buyer** | **Clinic owner** | Consistent first-line experience, digital operational visibility (revisits, packages, payments), and an outcome record that lifts retention and renewal. |
| **Beneficiary** | **Customer** | A consultation that is transparent rather than pushy, and a personal, visible record of how their skin has improved. |

The core promise to the buyer: **the product lets junior consultants deliver senior-level, standardized consultations.**

## 3. The closed loop (core workflow)

The product is one **closed loop**, not a set of separate features: *standardized photo intake → quality gate → quantified skin profile → explainable recommendations → consultation → CRM (packages, payments, revisits) → re-measure → progress report → outcome-aware next recommendations → next consultation.*

What the prototype implements today:

- **Standardized photo intake.** A front-face photo — uploaded or captured live in-browser via the consultant's webcam — runs through a **Capture Quality gate** (blur, pose, exposure/side-lighting, face ratio, capture resolution) producing a 0–100 score and a check-list. Poor photos are rejected with an explanation, so longitudinal comparison stays trustworthy.
- **Quantified skin profile.** Every photo is measured at a fixed face scale with colour and exposure normalized, so a webcam capture and a phone photo are scored on the same footing; landmark-anchored regions are sampled identically every visit; four metrics — **redness, evenness, texture, spots** — are scored 0–100 per region, each with a measured noise band so real change is never confused with capture noise.
- **Explainable recommendations.** A rules-based engine produces ranked treatment suggestions; each one names the exact metric, region and score that triggered it (e.g. "right cheek redness 72"), so it reads as evidence, not an upsell. The consultant can edit, reorder, add or remove items before saving.
- **Consultation view.** Consultant and customer look at the same screen: skin map plus the data behind each suggestion.
- **CRM tracking.** Customer records, visit timeline, package session counts (bought / used / remaining), payment status, and a next-revisit date with an **overdue-revisit** filter — replacing the paper book.
- **Re-measure and progress report.** Subsequent visits are compared region-by-region and metric-by-metric, with a tolerance band so noise is not mistaken for change. A customer-facing progress report turns this into something the customer can see.
- **Outcome-aware next recommendations.** When a prior visit exists, the engine reads the delta: an improving metric ("redness 72 → 58") is flagged as *treatment working — continue*, a flat metric as *adjust*. The loop closes.

## 4. Wedge, platform, and the retention flywheel

**Wedge — standardize the first-line consultation.** The single most painful problem and the easiest adoption entry: no back-office change required, value on day one, and it visibly fixes the "pushy salesperson" perception owners care about most. This is what gets a clinic to try the product at all.

**Platform — the AI-native CRM.** Every consultation already produces structured data: a quality-gated photo, a quantified profile, an approved plan. That data accretes into a full CRM with **longitudinal customer context** and **treatment-outcome analytics** — the foundation for clinic-wide operations and, over time, cross-clinic benchmarking of what actually works. The wedge gets the product in; the accumulated longitudinal data is what makes it hard to remove.

**Retention flywheel.** The loop *is* the retention engine: standardized measurement makes the recommendation credible; the credible recommendation drives the treatment; the treatment produces an outcome; the visible outcome drives renewal (pain point 3 fixed); renewal generates the next round of data, which makes the next recommendation smarter and more personalized. Retention and recommendation quality compound together.

## 5. Design and safety stance

**Mandarin-first by design.** Traditional Chinese throughout, using Taiwanese clinical and aesthetic terminology. The primary user is a Taiwanese consultant working with Taiwanese customers, so building Traditional-Chinese-native from the start — rather than translating an English product — is what makes the consultation feel professional and local. This is a deliberate product decision, treated as one of the brief's optional depth areas.

**LLM never decides.** Treatments, metric scores and the data-grounded reason that ties them together are all produced by the rules engine and the on-host CV pipeline. An optional LLM layer rewrites the template reason into more natural Mandarin **without changing the treatment or the numbers**. With no key configured, or on any failure, the UI falls back to the template — behaviour is identical to a key-free run. Inputs sent to the model are numeric only (metric label, region label, sub-score, template reason, treatment name, trend note); no images, names or medical history leave the host.

---

*Scope note: this prototype keeps skin metrics deliberately narrow (four solid, explainable metrics) and treatments generic and non-branded. It is positioned as clinical **decision support** for a trained consultant — it assists judgment, it does not replace it.*
