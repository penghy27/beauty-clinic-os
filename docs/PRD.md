# Beauty Clinic OS — Product Requirements Document

**An AI-native CRM and clinical decision-support system for Taiwanese aesthetic clinics**

*Status: working prototype.*

---

## 1. Problem — three pain points heard first-hand

Clinic owners described, in their own words, three problems that the existing
software market does not solve:

1. **The first-line consultation is inconsistent and feels sales-driven.** Every
   customer who walks in is met first by a consultant. But consultant quality
   varies widely and turnover is high, so the same customer can get very
   different advice depending on who is on shift that day. Worse, because
   consultants are paid on commission, customers often feel they are being
   *pushed a package* rather than having their actual skin needs addressed.
   Owners want this first touchpoint **standardized and made scientific** —
   because first-line experience is what drives retention and revenue.

2. **The CRM is paper-based.** Clinics still track, by hand on paper, whether a
   customer is due for a revisit, whether they have paid, and how many sessions
   of a purchased package they have used. This is error-prone and gives the
   owner no operational visibility.

3. **Customers have no record of what actually improved.** A customer completes
   a course of treatments but has no objective way to see whether their skin
   got better. Without visible proof of progress, there is no compelling reason
   to renew.

These pain points are not assumptions — they are first-hand voice-of-customer.
They are also the answer to *why an AI-native CRM fits clinic operations*: the
work of a clinic **is** consultation, scheduling, packages and outcomes, and
each pain point above is one of those workflows breaking down.

## 2. Target users

| Role | Who | What they get |
|------|-----|---------------|
| **Primary user** | First-line **consultant** | A guided, standardized consultation flow. A junior or newly-hired consultant runs the exact same evidence-based process as a senior one — quality no longer depends on the individual or survives staff turnover. |
| **Economic buyer** | **Clinic owner** | Consistent first-line experience, digital operational visibility (revisits, packages, payments), and an outcome record that lifts retention and renewal. |
| **Beneficiary** | **Customer** | A consultation that is transparent rather than pushy, and a personal, visible record of how their skin has improved. |

The core promise to the buyer: **the product lets junior consultants deliver
senior-level, standardized consultations.**

## 3. The closed loop (core workflow)

The product is deliberately built as one **closed loop**, not a set of separate
features. Each step feeds the next:

```
Standardized photo intake  →  Quantified skin profile  →  Explainable
recommendations  →  Treatment  →  CRM session / payment tracking  →  Revisit
→  Re-measure  →  Customer progress report  →  Outcome-aware next
recommendations  →  (back to the next consultation)
```

**What the prototype implements today:**

- **Standardized photo intake.** A front-face photo — either uploaded or
  captured live in-browser via the consultant's webcam — is run through an
  imaging pipeline with a **Capture Quality gate** (blur, pose,
  exposure/side-lighting, face ratio) producing a 0–100 score and a
  check-list. Poor photos are rejected with an explanation, so longitudinal
  comparison stays trustworthy.
- **Quantified skin profile.** Color is normalized (gray-world white balance);
  landmark-anchored regions are sampled identically every visit; four metrics —
  **redness, evenness, texture, spots** — are scored 0–100 per region.
- **Explainable recommendations.** A rules-based engine (`rules/treatments.yaml`)
  produces ranked treatment suggestions. Every suggestion names the exact
  metric, region and score that triggered it (e.g. "right cheek redness 72"),
  so it reads as evidence, not an upsell. The consultant can edit, reorder, add
  or remove items before saving — the engine assists, it does not dictate.
- **Consultation view.** Consultant and customer look at the same screen:
  visualized skin map plus the data behind each suggestion.
- **CRM tracking.** Customer records, visit timeline, package session counts
  (bought / used / remaining), payment status, and a next-revisit date with an
  **"overdue revisit"** filter — replacing the paper book.
- **Re-measure and progress report.** Subsequent visits are compared region-by-
  region and metric-by-metric against earlier ones, with a tolerance band so
  measurement noise is not mistaken for change. A customer-facing **progress
  report** turns this into something the customer can see.
- **Outcome-aware next recommendations.** When a prior visit exists, the engine
  reads the delta and adjusts: an improving metric ("redness 72 → 58") is
  flagged as *treatment working — continue*, a flat metric as *adjust the
  approach*. The loop closes.

## 4. Wedge → platform

**Wedge — standardize the first-line consultation.** This is the single most
painful problem and the easiest adoption entry: it needs no back-office change,
delivers value on day one, and visibly fixes the "pushy salesperson" perception
that owners care about most. It is the reason a clinic agrees to try the
product at all.

**Platform — the AI-native CRM.** Once the consultation runs through the
system, every consultation already produces structured data: a quality-gated
photo, a quantified skin profile, an approved treatment plan. That data
naturally accretes into a full CRM with **longitudinal customer context** and
**treatment-outcome analytics** — the foundation for clinic-wide operations,
retention management, and, over time, cross-clinic benchmarking of what
treatments actually work. The wedge is what gets the product in the door; the
accumulated longitudinal data is what makes it hard to remove.

## 5. The closed-loop flywheel — the retention engine

The loop is not just a workflow; it is the retention mechanism:

> **measure → recommend → treat → re-measure → visible improvement →
> retention / renewal → outcome-aware next recommendation → measure …**

Each turn of the loop makes the next turn stronger. A re-measurement that shows
improvement gives the customer a concrete reason to renew (solving pain point
3), and simultaneously feeds the engine the outcome signal it needs to make the
*next* recommendation smarter and more personalized. Standardized measurement
makes the recommendation credible; the credible recommendation drives the
treatment; the treatment produces an outcome; the visible outcome drives
renewal; renewal generates the next round of data. Retention and recommendation
quality compound together.

## 6. Design note — Mandarin-first by design

The product is **Mandarin-first**: Traditional Chinese throughout, using
Taiwanese clinical and aesthetic terminology. This is a deliberate product
decision, not a localization afterthought. The primary user is a Taiwanese
first-line consultant working with Taiwanese customers; the interface,
treatment vocabulary, recommendation copy and customer-facing progress report
all need to read naturally to that consultant and that customer. Building
Traditional-Chinese-native from the start — rather than translating an
English product — is what makes the consultation feel professional and local,
and is treated here as one of the brief's optional depth areas.

## 7. Optional copy polishing — LLM never decides

The treatment recommendation, the metric scores and the data-grounded reason
that ties one to the other are all produced by the rules engine and the
on-host CV pipeline. An optional LLM layer rewrites the template reason into
more natural Mandarin **without changing the treatment or the numbers**. If
no key is configured, or the call fails, the UI falls back to the template
text — behaviour is identical to a key-free run. Inputs sent to the model
are numeric only: metric label, region label, sub-score, the template
reason, the treatment name and the trend note. No images, names or medical
history leave the host.

---

*Scope note: this prototype keeps skin metrics deliberately narrow (four solid,
explainable metrics) and treatments generic and non-branded. It is positioned
as clinical **decision support** for a trained consultant — it assists
judgment, it does not replace it.*
