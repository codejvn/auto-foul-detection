# Soccer AI Referee — Pipeline

A soccer foul detection pipeline: local GPU inference for perception, an LLM judgment layer for ambiguous cases, and a deterministic rules engine for the final call. See the [repo root README](../README.md) for the project's results and the calibration research claim.

**How it works:** OpenCV extracts 16 evenly-spaced frames → a shot-boundary filter drops frames that straddle a camera cut → CLIP (zero-shot) detects player contact and penalty-box location → a fine-tuned dual-head VideoMAE classifies foul type and severity in one forward pass → if any module's confidence is below 0.65, Gemini reviews the ambiguous case → a deterministic pure-Python ruling engine produces the final call.

## Setup

```bash
cd generated_pipeline
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Models (~1 GB total: CLIP + base VideoMAE) download from HuggingFace automatically on first run and are cached locally after that. `~4.5 GB` VRAM with everything loaded; falls back to CPU if no GPU is present.

The fine-tuned foul/severity checkpoint is **not** downloaded automatically — `foul_classifier.py` expects it at `../checkpoints/videomae-foul-best.pt`, produced by running `train_foul_classifier.py` (see below). Without it, `classify_foul()` raises `FileNotFoundError`.

The judgment layer needs a Gemini API key: create a `.env` file in the repo root with `GEMINI_API_KEY=...`. It's optional — pass `--skip-judgment-layer` to `pipeline.py` (or `--dataset-path` alone to `validator.py`, which defaults to skipping it) to run fully offline with zero API calls.

## Run on a video

```bash
python pipeline.py clip.mp4
python pipeline.py clip.mp4 --num-frames 16
python pipeline.py clip.mp4 --skip-judgment-layer   # no Gemini calls, fully local
python pipeline.py clip.mp4 --with-judgment-layer   # force it on (default)
```

Progress goes to stderr; only the final ruling JSON goes to stdout, so output is pipeable:

```json
{
  "foul_detected": true,
  "foul_type": "tackle",
  "severity": "reckless",
  "in_penalty_box": false,
  "punishment": "free kick + yellow card",
  "confidence": 0.82,
  "human_review_recommended": false,
  "judgment_layer_used": false,
  "judgment_layer_reasoning": null,
  "low_confidence_modules": []
}
```

- `foul_type`: tackle | handball | obstruction | simulation | push | none (note: `handball` is not currently reachable — see [Known limitations](#known-limitations))
- `severity`: careless | reckless | excessive_force
- `punishment`: IFAB-style ruling (free kick / penalty kick, yellow/red card, or "play on")
- `human_review_recommended`: true when overall confidence < 0.60
- `low_confidence_modules`: which of contact/foul/severity/location fell below the 0.65 routing threshold and triggered the judgment layer

## Validate against the VARS / MVFoul dataset

```bash
python validator.py --dataset-path ./mvfoul_dataset --output results.json
python validator.py --smoke-test                     # 5 hardcoded examples, no dataset or models
python validator.py --dataset-path ./mvfoul_dataset --with-judgment-layer   # spend Gemini credits on a spot-check
```

Auto-detects an MVFoul split (`annotations.json` with an `"Actions"` key, via `dataset_builder.py`) or the older `<clip_dir>/annotations.json` + one video per directory layout. Scores foul-detected / foul-type / severity accuracy against consensus ground truth, plus the calibration F1 of "our low confidence" against "human annotators disagreed." See the root README for current numbers.

## Smoke tests (no video needed)

Every module self-tests with synthetic input:

```bash
python frame_extractor.py        # no models needed
python severity_assessor.py      # no models needed
python ruling_engine.py          # pure Python, no ML deps at all
python shot_boundary_filter.py   # PySceneDetect, falls back to MSE differencing if absent
python contact_detector.py       # downloads CLIP on first run
python location_detector.py      # reuses cached CLIP
python foul_classifier.py        # needs the fine-tuned checkpoint at ../checkpoints/videomae-foul-best.pt
python validator.py --smoke-test # scoring logic only, no models or dataset
```

## Fine-tuning `foul_classifier`'s checkpoint

```bash
python train_foul_classifier.py --dataset-path ./mvfoul_dataset --output-dir ../checkpoints
```

Trains `DualHeadVideoMAE` (VideoMAE-base backbone + a 9-class action head + a 4-class offence-severity head) on MVFoul clips via `dataset_builder.py`, and writes `videomae-foul-best.pt` with the state dict plus the checkpoint's class-name lists. `foul_classifier.py` loads this checkpoint directly — no keyword-mapping stopgap over a generic Kinetics label set.

## Module reference

| File | Role | Notes |
|---|---|---|
| `frame_extractor.py` | 16 evenly-spaced frames via OpenCV | no ML deps |
| `shot_boundary_filter.py` | drop frames straddling a camera cut | PySceneDetect, MSE fallback |
| `contact_detector.py` | player-to-player contact | CLIP zero-shot |
| `foul_classifier.py` | foul type + severity, one forward pass | fine-tuned dual-head VideoMAE |
| `severity_assessor.py` | optical-flow severity heuristic | **superseded** by `foul_classifier`'s trained severity head; kept for reference, not wired into `pipeline.py` |
| `location_detector.py` | inside the penalty box? | CLIP zero-shot |
| `judgment_layer.py` | reviews ambiguous cases (any module confidence < 0.65) | Gemini API, can override foul type/severity, cannot override contact/location |
| `ruling_engine.py` | final IFAB-style ruling | pure Python, deterministic, intentionally not swappable for a model |
| `pipeline.py` | orchestrates all stages, CLI entry point | single GPU model resident at a time |
| `dataset_builder.py` | MVFoul split → training/eval examples | used by `train_foul_classifier.py` and `validator.py` |
| `train_foul_classifier.py` | fine-tunes `DualHeadVideoMAE` | produces the checkpoint `foul_classifier.py` loads |
| `validator.py` | accuracy + calibration scoring against MVFoul | see root README for results |

## Swapping in a different model

Every ML module has a `MODEL_ID` (or `CHECKPOINT_PATH`) constant near the top of the file and a marked `SWAP HOOK` comment block:

| File | Current model |
|---|---|
| `contact_detector.py` | `openai/clip-vit-base-patch32` |
| `foul_classifier.py` | fine-tuned `DualHeadVideoMAE` on `MCG-NJU/videomae-base-finetuned-kinetics` |
| `location_detector.py` | `openai/clip-vit-base-patch32` |
| `severity_assessor.py` | Farneback optical-flow heuristic (unwired, see above) |
| `judgment_layer.py` | `gemini-2.5-flash` |
| `ruling_engine.py` | — intentionally not swappable: rules stay deterministic and auditable |
| `shot_boundary_filter.py` | PySceneDetect `ContentDetector` (MSE fallback) |

Change the constant, keep the return dict shape the same, and `pipeline.py` needs no changes.

## Known limitations

- **`foul_classifier`'s action head can't produce `handball`** — the 9-class head was trained without a handball label (deferred during training); `classify_foul()` can never return `"handball"` regardless of input.
- **Rare foul types and high severities are weak.** Push/obstruction/simulation and red-card-level severity are held back by dataset scale, not the modeling approach — see the root README's Known Limitations for the oversampling result.
- **`severity_assessor.py`'s optical-flow heuristic is no longer load-bearing** — kept in the repo as a reference implementation and swap target, not because it's competitive with the trained head.
- **`location_detector.py` reads scene composition, not pitch geometry** — a calibration-based model (SoccerNet-calibration) is the principled upgrade over CLIP zero-shot.
