"""
codegen.py — uses Fable 5 + Sonnet subagents to generate a complete
local soccer referee pipeline. Runs once. Produces Python files you
run locally with no ongoing API costs.
"""

import os
import json
import anthropic
from pathlib import Path

client = anthropic.Anthropic()
FABLE  = "claude-fable-5"
SONNET = "claude-sonnet-4-6"

OUTPUT_DIR = Path("./generated_pipeline")
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Step 1: Fable 5 produces the architecture plan
# ──────────────────────────────────────────────

FABLE_ARCHITECT_PROMPT = """You are a senior ML engineer. Design a complete local soccer foul detection pipeline.

CONSTRAINTS:
- Runs entirely locally on an NVIDIA GPU — zero API calls at inference time
- Uses pre-trained HuggingFace models with clean hooks to swap in fine-tuned versions later
- Input: path to a video clip (.mp4)
- Output: structured ruling dict with foul_type, severity, in_penalty_box, punishment
- Python 3.10+, PyTorch, HuggingFace transformers

PIPELINE MODULES to design (one Python file each):
1. frame_extractor.py — extract N evenly-spaced frames from video, return tensors
2. contact_detector.py — detect whether player contact occurred (binary), return bool + confidence
3. foul_classifier.py — classify foul type from frames (tackle/handball/obstruction/simulation/push/none)
4. severity_assessor.py — assess severity (careless/reckless/excessive_force) from frames + foul type
5. location_detector.py — detect if foul occurred inside penalty box (binary + confidence)
6. ruling_engine.py — pure Python rule logic: takes all module outputs, returns final ruling dict
7. pipeline.py — orchestrates all modules end to end, single entry point
8. requirements.txt — all pip dependencies with pinned versions

For each file, specify:
- Exact HuggingFace model to use (model ID string)
- Input/output types as Python type hints
- Any special preprocessing needed
- The swap hook: a commented section showing exactly how to replace with a fine-tuned model

Be specific. Give model IDs, not vague descriptions. For contact_detector and location_detector
you can use a vision-language model like CLIP or a ViT. For foul_classifier use VideoMAE.
For severity_assessor, reason from motion features — optical flow or frame differencing is fine.

Respond with a JSON object:
{
  "architecture_summary": "2-3 sentence overview",
  "modules": [
    {
      "filename": "frame_extractor.py",
      "model": "none — pure OpenCV",
      "inputs": "video_path: str, num_frames: int = 16",
      "outputs": "frames: list[np.ndarray], timestamps: list[float]",
      "key_decisions": ["decision 1", "decision 2"],
      "swap_hook_description": "how to swap in fine-tuned version"
    },
    ... (one entry per file)
  ],
  "gpu_memory_estimate_gb": 4.5,
  "recommended_num_frames": 16
}
"""

print("="*60)
print("Step 1: Fable 5 designing architecture...")
print("="*60)

arch_response = client.messages.create(
    model=FABLE,
    max_tokens=4096,
    messages=[{"role": "user", "content": FABLE_ARCHITECT_PROMPT}]
)

arch_text = arch_response.content[0].text

# Parse JSON from Fable 5's response
import re
json_match = re.search(r'\{.*\}', arch_text, re.DOTALL)
if not json_match:
    print("ERROR: Fable 5 didn't return valid JSON")
    print(arch_text[:500])
    exit(1)

architecture = json.loads(json_match.group())
print(f"\nArchitecture summary: {architecture['architecture_summary']}")
print(f"GPU estimate: {architecture.get('gpu_memory_estimate_gb', 'unknown')} GB")
print(f"Modules to generate: {[m['filename'] for m in architecture['modules']]}\n")

# Save architecture plan
with open(OUTPUT_DIR / "architecture_plan.json", "w") as f:
    json.dump(architecture, f, indent=2)
print(f"Saved architecture plan to {OUTPUT_DIR}/architecture_plan.json")

# ──────────────────────────────────────────────
# Step 2: Sonnet subagents write each module
# Each module is written independently and completely
# ──────────────────────────────────────────────

def sonnet_write_module(module_spec: dict, architecture_summary: str, num_frames: int) -> str:
    """Call Sonnet to write one module file."""

    prompt = f"""You are an expert ML engineer. Write a complete, production-quality Python file.

ARCHITECTURE CONTEXT:
{architecture_summary}

THIS MODULE: {module_spec['filename']}
Model to use: {module_spec['model']}
Inputs: {module_spec['inputs']}
Outputs: {module_spec['outputs']}
Key decisions: {json.dumps(module_spec['key_decisions'], indent=2)}
Swap hook: {module_spec['swap_hook_description']}
Default num_frames: {num_frames}

REQUIREMENTS:
- Complete, runnable Python file — no placeholders, no TODOs
- All imports at the top
- GPU-aware: use torch.cuda if available, fall back to CPU gracefully
- Clean type hints throughout
- Docstrings on every class and function
- The swap hook must be a clearly commented block showing the EXACT lines to change
  to plug in a fine-tuned model, with the HuggingFace model ID as a string constant
  at the top of the file (e.g. MODEL_ID = "microsoft/videomae-base") so it's trivial
  to change
- For model loading: cache the model as a module-level singleton so it only loads once
- Error handling for corrupt frames, missing GPU, model download failures
- A self-contained __main__ block that runs a quick smoke test with a dummy input
  (synthesized tensor, not requiring a real video file) so the user can verify
  the file works with: python {module_spec['filename']}

Write ONLY the Python code. No explanation before or after. Start with the imports."""

    response = client.messages.create(
        model=SONNET,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


print("="*60)
print("Step 2: Sonnet subagents writing each module in parallel...")
print("="*60)

import threading

results = {}
errors = {}
num_frames = architecture.get("recommended_num_frames", 16)
arch_summary = architecture["architecture_summary"]

def write_module(module_spec):
    filename = module_spec["filename"]
    try:
        print(f"  [{filename}] Sonnet writing...")
        code = sonnet_write_module(module_spec, arch_summary, num_frames)
        # Strip markdown fences if Sonnet wrapped output
        code = re.sub(r'^```python\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\n?', '', code, flags=re.MULTILINE)
        results[filename] = code
        print(f"  [{filename}] Done ({len(code)} chars)")
    except Exception as e:
        errors[filename] = str(e)
        print(f"  [{filename}] ERROR: {e}")

threads = []
for module in architecture["modules"]:
    if module["filename"] == "requirements.txt":
        continue  # handle separately
    t = threading.Thread(target=write_module, args=(module,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# ──────────────────────────────────────────────
# Step 3: Sonnet writes requirements.txt
# Done after modules so it can reference what was actually used
# ──────────────────────────────────────────────

print("\nWriting requirements.txt...")
req_module = next((m for m in architecture["modules"] if m["filename"] == "requirements.txt"), None)
models_used = [m["model"] for m in architecture["modules"] if m["model"] != "none — pure OpenCV"]

req_prompt = f"""Write a requirements.txt for a local soccer foul detection pipeline.

Models used: {json.dumps(models_used)}
Architecture: {arch_summary}

Include:
- torch (with CUDA 11.8 index URL as a comment)
- transformers
- opencv-python
- numpy
- Pillow
- decord (fast video loading)
- accelerate (HuggingFace model loading)
- any other packages the models above require

Format: one package per line with pinned versions that are stable as of mid-2025.
Include a comment at the top with the torch CUDA install command.
Write ONLY the requirements.txt content."""

req_response = client.messages.create(
    model=SONNET,
    max_tokens=1024,
    messages=[{"role": "user", "content": req_prompt}]
)
results["requirements.txt"] = req_response.content[0].text

# ──────────────────────────────────────────────
# Step 4: Write all files to disk
# ──────────────────────────────────────────────

print("\n" + "="*60)
print("Step 3: Writing files to disk...")
print("="*60)

for filename, code in results.items():
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        f.write(code)
    print(f"  Wrote {path} ({len(code):,} chars)")

# ──────────────────────────────────────────────
# Step 5: Fable 5 writes a README for the generated code
# ──────────────────────────────────────────────

print("\nFable 5 writing setup README...")

readme_prompt = f"""Write a README.md for a local soccer foul detection pipeline that was just generated.

Architecture: {arch_summary}
Files generated: {list(results.keys())}
GPU memory needed: {architecture.get('gpu_memory_estimate_gb', 'unknown')} GB

Include:
1. One-command setup (pip install)
2. How to run on a video: python pipeline.py clip.mp4
3. How to swap in a fine-tuned model (reference the MODEL_ID constant in each file)
4. How to fine-tune on SoccerNet (high level — just point to the right HuggingFace docs)
5. Expected output format

Be concise and practical. This is for a Princeton CS student building a portfolio project."""

readme_response = client.messages.create(
    model=FABLE,
    max_tokens=2048,
    messages=[{"role": "user", "content": readme_prompt}]
)

readme_text = readme_response.content[0].text
with open(OUTPUT_DIR / "README.md", "w") as f:
    f.write(readme_text)
print(f"  Wrote {OUTPUT_DIR}/README.md")

# ──────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
print(f"\nGenerated pipeline in: ./{OUTPUT_DIR}/")
print("\nFiles:")
for p in sorted(OUTPUT_DIR.iterdir()):
    size = p.stat().st_size
    print(f"  {p.name:30s} {size:>8,} bytes")

print(f"\nErrors: {list(errors.keys()) if errors else 'none'}")
print("\nNext steps:")
print("  1. cd generated_pipeline")
print("  2. pip install torch --index-url https://download.pytorch.org/whl/cu118")
print("  3. pip install -r requirements.txt")
print("  4. python pipeline.py your_clip.mp4")
print("  5. Smoke test each module: python frame_extractor.py")