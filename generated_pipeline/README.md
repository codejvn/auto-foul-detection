# Soccer AI Referee — Generated Pipeline

A fully local soccer foul detection pipeline. Input: a video clip. Output: foul type, severity, location, and punishment ruling. Runs on an NVIDIA GPU (~4.5 GB VRAM with all models loaded), falls back to CPU gracefully. Zero API calls at inference time.

**How it works:** OpenCV extracts 16 evenly-spaced frames → CLIP (zero-shot) detects player contact and penalty-box location → VideoMAE classifies foul type → Farneback optical flow drives severity → a deterministic pure-Python ruling engine produces the final call.

## Setup

```bash
cd generated_pipeline
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Models (~1 GB total) download from HuggingFace automatically on first run and are cached locally after that.

## Run on a video

```bash
python pipeline.py clip.mp4
# optional: python pipeline.py clip.mp4 --num-frames 16
```

Progress goes to stderr; only the final ruling JSON goes to stdout, so output is pipeable:

```json
{
  "foul_detected": true,
  "foul_type": "tackle",
  "severity": "reckless",
  "in_penalty_box": false,
  "punishment": "free kick + yellow card",
  "confidence": 0.82
}
```

- `foul_type`: tackle | handball | obstruction | simulation | push | none
- `severity`: careless | reckless | excessive_force
- `punishment`: IFAB-style ruling (free kick / penalty kick, yellow/red card, or "play on")

## Smoke tests (no video needed)

Every module self-tests with synthetic input:

```bash
python frame_extractor.py     # no models needed
python severity_assessor.py   # no models needed
python ruling_engine.py       # pure Python, no ML deps at all
python contact_detector.py    # downloads CLIP on first run
python location_detector.py   # reuses cached CLIP
python foul_classifier.py     # downloads VideoMAE on first run
```

## Swapping in a fine-tuned model

Every ML module has a `MODEL_ID` string constant at the top of the file and a clearly marked `SWAP HOOK` comment block showing the exact lines to change:

| File | Default model | Swap to |
|---|---|---|
| `contact_detector.py` | `openai/clip-vit-base-patch32` | fine-tuned CLIP contact checkpoint |
| `foul_classifier.py` | `MCG-NJU/videomae-base-finetuned-kinetics` | VideoMAE fine-tuned on SoccerNet fouls (6-class head) |
| `location_detector.py` | `openai/clip-vit-base-patch32` | pitch-localization model (SoccerNet-calibration) |
| `severity_assessor.py` | optical-flow heuristic | fine-tuned video severity model (see SWAP HOOK) |
| `ruling_engine.py` | — | intentionally not swappable: rules stay deterministic and auditable |

Change the `MODEL_ID` string, keep the return dict the same, and `pipeline.py` needs no changes.

## Fine-tuning on SoccerNet (the accuracy upgrade)

The pre-trained pipeline is a working scaffold; the biggest accuracy win is fine-tuning VideoMAE on real foul clips. The stock Kinetics head can't distinguish handball or simulation — those become reliable only after this step.

1. Download [SoccerNet-v2](https://www.soccer-net.org/) foul/action-spotting clips
2. Fine-tune following the [HuggingFace video classification guide](https://huggingface.co/docs/transformers/tasks/video_classification) (VideoMAE section) with the 6 foul labels
3. Push the checkpoint to the HuggingFace Hub (or save locally)
4. Point `MODEL_ID` in `foul_classifier.py` at it and delete the Kinetics keyword-mapping function (see the SWAP HOOK block)
5. Repeat for contact/location/severity as needed

## Known limitations

- Foul-type classification from the stock Kinetics head is a keyword-mapping stopgap — expect weak accuracy until the SoccerNet fine-tune
- Severity thresholds are motion heuristics tuned for broadcast-style footage; recalibrate the constants in `severity_assessor.py` for other camera setups
- CLIP location detection reads the scene, not pitch geometry — a calibration-based model is the proper upgrade
