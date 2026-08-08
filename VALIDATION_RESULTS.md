# Validation results — VARS/MVFoul test set

Evaluated on `mvfoul_extracted/test`: **285 clips** (251 clear, 34 hard), 0 pipeline
failures. The "hard" set is the clips where annotators hedged on severity — the
borderline card codes 2.0 (No-card/Yellow) and 4.0 (Yellow/Red), which
`dataset_builder` routes to `hard_cases`. All sweeps run with the Gemini judgment
layer disabled (`validator.py` default), so the numbers reflect the CV pipeline's own
outputs and confidence, not the LLM's.

## Summary

Wiring the fine-tuned dual-head classifier into the ruling produced large accuracy
gains. The project's novel calibration claim did **not** hold: the model's confidence
does not predict which clips humans found hard, and neither does test-time epistemic
uncertainty.

| Claim (CLAUDE.md) | Result |
|---|---|
| Perception accuracy | Action head 50% foul-type (tackle 76%, rare classes weak); severity head 59% card accuracy |
| Ruling accuracy | Foul-detected **80.7%** (was 18.2%) |
| Calibration (novel) | **Not supported** — confidence is at chance for predicting human disagreement |

## What changed in the pipeline

The base pipeline gated foul detection on a CLIP zero-shot contact detector and took
severity from a Farneback optical-flow heuristic. Both are unreliable on these clips.
The fine-tuned `DualHeadVideoMAE` (9-way action head + 4-way offence/severity head,
combined macro-F1 0.2685) was activated in `foul_classifier.py` and then wired into
the ruling:

- **Detection** now comes from the severity head's offence signal (`severity != "No
  offence"`), not the contact gate. (`ruling_engine`, commit `10ee0b5`)
- **Card/severity** now comes from the severity head's card prediction, not optical
  flow. (`ruling_engine`, commit `7efcd83`)
- `--skip-judgment-layer` / `--with-judgment-layer` flags added; the calibration sweep
  skips the judgment layer by default. (`8e174ef`)
- `validator.py` enumerates clips and ground truth via `dataset_builder` for the MVFoul
  split layout. (`8e174ef`)

The two heuristic modules (CLIP contact, optical-flow severity) still run but their
outputs no longer feed the decision.

## Accuracy: three sweeps

| Config | Foul detected | Foul type | Severity | Calibration F1 |
|---|---|---|---|---|
| Baseline (CLIP contact gate) | 18.2% | 12.3% | 60.0% * | 0.201 |
| + severity head → detection | 80.7% | 49.8% | 13.0% † | 0.200 |
| + severity head → card severity | **80.7%** | **49.8%** | **58.7%** | 0.200 |

\* Artifact: with detection failing, 86% of clips defaulted to the "careless" floor,
which happened to match the careless-heavy ground truth.
† Real but poor: once detection worked, the ruling used the optical-flow severity,
which predicts "excessive_force" on 240/245 detected clips (ground truth has 5).

Root causes of the 18% baseline detection, confirmed by measurement:

- **CLIP contact detector** returns `contact=True` on only 2/25 real-foul clips (mean
  "contact" probability 0.168). `make_ruling` correctly requires contact for a
  non-handball foul, so detection collapsed.
- **Optical-flow severity** is miscalibrated toward "excessive_force", exposed once
  detection started using real (non-default) severity.

Per-foul-type accuracy is carried by the majority class: tackle **76%**, with
obstruction/push/simulation in single digits — a rare-class recognition limit of the
action head, traceable to the small training set (and the Red-card class, 27 train / 3
valid clips, is not learnable at all).

## Calibration: the negative result

The novel claim is that pipeline confidence below 0.65 should coincide with the
human-disagreement clips. It does not. We tested every plausible confidence and
uncertainty measure over all 285 test clips and asked, for each: does it separate the
34 hard clips from the 251 clear ones, and what is the best achievable calibration F1
(F1 of "low confidence / high uncertainty" as a predictor of disagreement)?

**Softmax confidence** (`mean_clear` vs `mean_hard`; best F1 over all thresholds):

| Measure | mean clear | mean hard | best F1 | precision | recall |
|---|---|---|---|---|---|
| action max-prob | 0.365 | 0.382 | 0.214 | 0.120 | 1.00 |
| severity max-prob | 0.504 | 0.483 | 0.236 | 0.136 | 0.88 |
| severity margin | 0.205 | 0.186 | 0.233 | 0.133 | 0.97 |
| action margin | 0.135 | 0.140 | 0.241 | 0.204 | 0.29 |

**MC-dropout epistemic uncertainty** (dropout p=0.3 on pooled features, 30 passes;
higher = more uncertain):

| Measure | mean clear | mean hard | best F1 | precision | recall |
|---|---|---|---|---|---|
| severity BALD | 0.021 | 0.020 | 0.214 | 0.120 | 1.00 |
| severity pred-entropy | 1.067 | 1.100 | 0.252 | 0.146 | 0.91 |
| severity variation-ratio | 0.183 | 0.178 | 0.208 | 0.123 | 0.68 |
| action variation-ratio | 0.216 | 0.254 | **0.286** | 0.228 | 0.38 |

For essentially every measure the hard and clear distributions are the same
(`mean_hard ≈ mean_clear`, differences within noise), and the best achievable
calibration F1 — around 0.24 for softmax, 0.29 for the single best MC-dropout measure —
sits at or barely above the random base rate (34/285 = 0.12 precision). The severity
head's uncertainty, which should be most informative since the hard clips are severity-
borderline, is flat at chance.

**Interpretation.** The severity head was trained with hard one-hot labels; it learned
to predict a card class, not to be uncertain where humans hedged. That disagreement
signal was never in the training target, so no test-time extraction (softmax or
MC-dropout) recovers it. This is a robust negative result across two independent
uncertainty methods.

## Conclusions

- **Task accuracy improved substantially** by driving detection and severity from the
  trained heads rather than the heuristic modules.
- **The calibration claim is not supported by this model.** Its confidence does not
  predict human disagreement.
- **Foul-type accuracy is capped by rare-class recognition**, a data-scarcity limit.

## Future work

- **Calibrated-uncertainty retrain** (the only principled path to the calibration
  claim): train the severity head on soft/distributional targets so it learns to hedge
  where annotators did — e.g. map the borderline codes to soft labels (2.0 → half
  No-card / half Yellow; 4.0 → half Yellow / half Red) or apply label smoothing. Still
  constrained by the small hard-case count.
- **More data / better rare-class handling** for the action head and the Red-card
  severity class.

## Reproduction

```bash
# accuracy sweep (judgment layer skipped by default), MVFoul split auto-detected
python generated_pipeline/validator.py --dataset-path ./mvfoul_extracted/test --output results.json

# single clip with full judgment-layer reasoning
python generated_pipeline/pipeline.py <clip.mp4> --with-judgment-layer
```

Result JSONs: `results_test_skipjudgment.json` (baseline), `results_test_offencedetect.json`
(severity-head detection), `results_test_headseverity.json` (final).
