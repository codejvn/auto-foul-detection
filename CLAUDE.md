# Soccer AI Referee — Project Context

## What this project is
An AI soccer referee pipeline. Input: a video clip. Output: foul type, severity, location, and punishment ruling. Runs entirely locally on NVIDIA GPU — zero API calls at inference time.

## How the code gets built
`codegen.py` is a one-time code generator. It calls Fable 5 (architect) and Sonnet subagents (module writers) to produce a complete local pipeline in `./generated_pipeline/`. You run codegen once, then run the generated pipeline forever.

## Project structure
```
codegen.py                  ← run this once to generate everything
generated_pipeline/
  frame_extractor.py        ← OpenCV frame extraction
  contact_detector.py       ← CLIP-based contact detection
  foul_classifier.py        ← VideoMAE foul type classification
  severity_assessor.py      ← optical flow severity assessment
  location_detector.py      ← penalty box location detection
  ruling_engine.py          ← pure Python rule logic, no ML
  pipeline.py               ← end-to-end orchestrator
  requirements.txt
  README.md
architecture_plan.json      ← Fable 5's architecture decisions
```

## Key design decisions
- Pre-trained HuggingFace models, no training required to get started
- Every module has a SWAP HOOK comment showing exactly how to replace with a fine-tuned model (change MODEL_ID string at top of file)
- Target: NVIDIA GPU, falls back to CPU gracefully
- `ruling_engine.py` is pure Python rule logic — no ML, deterministic

## Commands
```bash
# Generate the pipeline (one time)
python codegen.py

# Run on a video
python generated_pipeline/pipeline.py clip.mp4

# Smoke test individual modules (no real video needed)
python generated_pipeline/frame_extractor.py
python generated_pipeline/contact_detector.py
python generated_pipeline/foul_classifier.py
python generated_pipeline/severity_assessor.py
python generated_pipeline/location_detector.py
python generated_pipeline/ruling_engine.py

# Install deps (after codegen runs)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r generated_pipeline/requirements.txt
```

## What good output looks like
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

## Upgrade path
1. Get pipeline working with pre-trained models
2. Download SoccerNet-v2 dataset
3. Fine-tune VideoMAE on SoccerNet foul clips
4. Change MODEL_ID in foul_classifier.py to your fine-tuned model
5. Repeat for other modules as needed

## Known issues / things to watch for
- Sonnet sometimes wraps code output in markdown fences — codegen.py strips these
- Fable 5 refusals return stop_reason="refusal" not an exception — codegen handles this
- If a generated module is incomplete, ask Claude Code to rewrite just that file
- GPU memory: expect ~4-6GB VRAM with all models loaded simultaneously