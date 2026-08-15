# Soccer AI Referee

An automated VAR (Video Assistant Referee) pipeline: give it a broadcast soccer clip, it tells you whether a foul occurred, what type, how severe, and what the punishment should be — with a calibrated confidence score and an escalation path to an LLM "judgment layer" for the cases it isn't sure about.

The project has two halves:

1. **A working local pipeline** — frame extraction, shot-cut filtering, contact detection, foul classification, severity assessment, and a deterministic IFAB rules engine, all running on a single consumer GPU.
2. **A calibration argument** — the claim that a model's confidence score should predict when human VAR annotators would themselves disagree. This is tested directly against the SoccerNet MVFoul dataset, and the result is reported below.

## Architecture

```
video clip
    │
    ▼
frame_extractor.py       OpenCV, 16 evenly-spaced frames
    │
    ▼
shot_boundary_filter.py  PySceneDetect (MSE fallback), drops frames that straddle a camera cut
    │
    ├──► contact_detector.py    CLIP zero-shot — player-to-player contact
    ├──► foul_classifier.py     fine-tuned dual-head VideoMAE — foul type + severity
    ├──► severity_assessor.py   Farneback optical flow (legacy, superseded — see below)
    └──► location_detector.py   CLIP zero-shot — inside the penalty box?
    │
    ▼
confidence routing        any module confidence < 0.65?
    │                             │
    │ no                          │ yes
    ▼                             ▼
ruling_engine.py  ◄──── judgment_layer.py (Gemini, ambiguous cases only)
    │
    ▼
{foul_detected, foul_type, severity, punishment, confidence, human_review_recommended, ...}
```

Every ML stage is swappable via a `MODEL_ID` constant and a marked `SWAP HOOK` block in its file. `ruling_engine.py` is the one stage deliberately *not* swappable for a model — the final call stays deterministic, pure-Python IFAB rules, so it can be audited line by line.

## What's actually trained here

`foul_classifier.py` runs a fine-tuned `DualHeadVideoMAE` — a VideoMAE-base backbone with two classification heads (9-class action type, 4-class offence severity), fine-tuned via `train_foul_classifier.py` on SoccerNet's MVFoul clips. Its severity head has replaced the original optical-flow heuristic as the pipeline's actual severity signal; `severity_assessor.py` is kept in the repo for reference but is no longer load-bearing.

`judgment_layer.py` calls Gemini only when a module's confidence drops below the routing threshold — most clips never leave the GPU.

## Results (VARS / MVFoul dataset, 285 clips)

| Metric | Score |
|---|---|
| Foul detected accuracy | 80.7% |
| Foul type accuracy | 49.8% |
| Severity accuracy (on 230 scored offence clips) | 58.7% |
| Per-type accuracy: tackle | 75.7% |
| Per-type accuracy: push / obstruction / simulation | 0–13% |

Tackle detection — the dominant class in the dataset — is solid. The rarer classes (push, obstruction, simulation, and especially red-card-level severity) are held back by how little labeled data MVFoul provides for them; oversampling and class reweighting were tried and did not move the needle (see [Known limitations](#known-limitations)).

### The calibration claim — a negative result

The project set out to test whether the pipeline's own confidence score predicts the clips human VAR annotators disagreed about most — the idea being that an automated system's self-reported uncertainty could do something a human referee's gut feeling can't: be checked. Measured directly:

| | |
|---|---|
| Low-confidence clips (< 0.65) | 266 of 285 |
| Human-disagreement clips | 34 |
| Overlap | 30 |
| Calibration F1 | **0.20** |

The pipeline's confidence is uninformative about which clips humans found hard — the overlap is close to what you'd get by chance, and the model is broadly under-confident everywhere rather than selectively uncertain on the ambiguous cases. This is reported as-is rather than reframed: getting a calibration signal that actually tracks human disagreement looks like it needs either retraining with an uncertainty objective (e.g. MC-dropout, deep ensembles) or a narrower, better-scoped version of the claim — not a tuning fix on top of the current model.

## Setup

```bash
cd generated_pipeline
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Models download from HuggingFace on first run and are cached locally after that. The fine-tuned checkpoint (`checkpoints/videomae-foul-best.pt`) is produced by `train_foul_classifier.py`; the judgment layer needs a `GEMINI_API_KEY` in a local `.env` file (optional — the pipeline runs fully offline with `--skip-judgment-layer`).

## Usage

```bash
# Run the full pipeline on a clip
python pipeline.py clip.mp4
python pipeline.py clip.mp4 --skip-judgment-layer   # no API calls, fully local

# Validate against a downloaded MVFoul split
python validator.py --dataset-path ./mvfoul_dataset --output results.json
python validator.py --smoke-test                    # no dataset or models needed

# Every stage module also self-tests standalone
python frame_extractor.py
python shot_boundary_filter.py
python contact_detector.py
python foul_classifier.py
python location_detector.py
python ruling_engine.py
```

`pipeline.py` prints per-stage progress to stderr and the final ruling as JSON to stdout, so it's pipeable:

```json
{
  "foul_detected": true,
  "foul_type": "tackle",
  "severity": "reckless",
  "in_penalty_box": false,
  "punishment": "free kick + yellow card",
  "confidence": 0.71,
  "human_review_recommended": false,
  "judgment_layer_used": false,
  "low_confidence_modules": []
}
```

## Known limitations

- **Rare-class data scarcity**: MVFoul's harder severity classes (particularly red-card-level "excessive force") have on the order of dozens of training clips. Resampling and class-weighted loss were tried and did not improve red-card F1 — this looks like a data-scale ceiling, not a training bug.
- **`foul_classifier`'s action head can't predict `handball`**: the fine-tuned 9-class head doesn't currently include a handball label (deferred during training), so that foul type is unreachable from the trained model regardless of input.
- **Calibration is not yet a validated claim** — see above.
- **`location_detector` reads scene composition, not pitch geometry** — a calibration-based model (e.g. SoccerNet field-registration) would be the principled upgrade over CLIP zero-shot.

## Repo layout

```
generated_pipeline/    the pipeline itself — see its own README for module-level detail
checkpoints*/          trained model weights (gitignored, produced locally)
```

Each module in `generated_pipeline/` documents its own public API contract, model choice, and swap hook in its file-level docstring — start there for implementation detail beyond what's summarized here.
