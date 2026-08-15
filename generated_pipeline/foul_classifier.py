"""
foul_classifier.py
===================

Foul-type classification module for the auto-foul-detection pipeline.

This module wraps a pre-trained VideoMAE video classification model
(fine-tuned on Kinetics-400) and maps its action-recognition predictions
onto soccer foul categories using a keyword mapping over the predicted
label strings. It is one signal among several (CLIP contact/box-location
heads, optical-flow severity, ruling engine) that are combined elsewhere
in the pipeline.

Public API
----------
classify_foul(frames: list[np.ndarray]) -> dict
    frames: list of RGB uint8 HxWx3 numpy arrays (any length >= 1)
    returns: {"foul_type": str, "confidence": float}
    where foul_type is one of: tackle, handball, obstruction, simulation,
    push, none.

Design notes
------------
- VideoMAE requires exactly 16 frames per clip. Shorter sequences are
  padded by repeating the final frame; longer sequences are subsampled
  evenly across the clip.
- The pre-trained Kinetics-400 head does NOT know about soccer fouls.
  We take its top-5 predicted action labels and map them onto foul
  categories via keyword matching (see KINETICS_LABEL_TO_FOUL_KEYWORDS).
  Handball and simulation are essentially unreachable through this
  mapping because Kinetics-400 has no closely related action classes;
  they will only become reliably detectable after fine-tuning on
  SoccerNet-style foul-labeled clips (see SWAP HOOK below).
- The model and image processor are cached as module-level singletons
  and lazily loaded on first use to avoid paying the load cost at
  import time.
- Uses CUDA when available, otherwise falls back to CPU automatically.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SWAP HOOK
# ---------------------------------------------------------------------------
# MODEL_ID controls which checkpoint this module loads. Today it points at
# a generic Kinetics-400 action-recognition checkpoint, whose predictions
# are translated into foul categories via keyword matching further below.
#
# `train_foul_classifier.py` fine-tunes this exact checkpoint into a direct
# 6-class head and saves it to `./checkpoints/videomae-foul-best.pt`. To
# activate that fine-tuned checkpoint here, do EXACTLY this:
#
#   1. Load the checkpoint dict from './checkpoints/videomae-foul-best.pt'
#      via `torch.load(...)`. It has keys: 'model_state_dict',
#      'class_names', 'epoch', 'val_f1'.
#   2. Before loading `model_state_dict`, replace the model's classification
#      head with `torch.nn.Linear(768, len(CLASS_NAMES))` (768 = VideoMAE-
#      base hidden size; this must match the head shape
#      `train_foul_classifier.py` trains).
#   3. Use the checkpoint's 'class_names' list (index order: tackle,
#      handball, obstruction, simulation, push, none) to map the argmax
#      output index directly to a foul type string — no keyword matching
#      needed, e.g.:
#          probs = F.softmax(logits, dim=-1)[0]
#          idx = int(torch.argmax(probs).item())
#          label = checkpoint["class_names"][idx]
#          confidence = float(probs[idx].item())
#          return {"foul_type": label, "confidence": confidence}
#   4. In `classify_foul`, replace the call to `_map_kinetics_logits_to_foul`
#      with the direct softmax + argmax over the 6-class logits shown above.
#   5. DELETE the `_map_kinetics_logits_to_foul` function and the
#      `KINETICS_LABEL_TO_FOUL_KEYWORDS` mapping dict entirely — they are
#      Kinetics-400-specific compatibility shims that no longer apply once
#      the model natively outputs foul classes.
# ---------------------------------------------------------------------------
MODEL_ID: str = "MCG-NJU/videomae-base-finetuned-kinetics"

# Number of frames VideoMAE expects per clip.
NUM_FRAMES: int = 16

# Number of top predicted Kinetics-400 labels to inspect when mapping to
# foul categories.
TOP_K: int = 5

# The complete set of foul categories this module can emit.
FOUL_TYPES = ("tackle", "handball", "obstruction", "simulation", "push", "none")

# Keyword -> foul type mapping applied (in order) against lowercased
# Kinetics-400 label strings. The first matching keyword for a label wins.
# NOTE: 'handball' and 'simulation' have no reliable Kinetics-400 analogues
# and are effectively unreachable via this mapping today; they will become
# detectable once a SoccerNet fine-tuned head is swapped in (see SWAP HOOK).
KINETICS_LABEL_TO_FOUL_KEYWORDS: dict[str, str] = {
    "tackl": "tackle",
    "wrestling": "push",
    "push": "push",
    "slapping": "push",
    "punching": "push",
    "headbutting": "push",
    "shoving": "push",
    "grappling": "obstruction",
    "wrestl": "obstruction",
    "capoeira": "obstruction",
    "sparring": "obstruction",
}

# Module-level singleton cache for the model and processor.
_model: Optional[VideoMAEForVideoClassification] = None
_processor: Optional[VideoMAEImageProcessor] = None
_device: Optional[torch.device] = None


def _get_device() -> torch.device:
    """Return the best available torch device, preferring CUDA when present."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> tuple[VideoMAEForVideoClassification, VideoMAEImageProcessor, torch.device]:
    """
    Lazily load and cache the VideoMAE model and its image processor.

    Returns the cached singletons on subsequent calls. Raises a
    RuntimeError with a helpful message if the model fails to download
    or load (e.g. due to network issues or a bad MODEL_ID).
    """
    global _model, _processor, _device

    if _model is not None and _processor is not None and _device is not None:
        return _model, _processor, _device

    _device = _get_device()
    logger.info("Loading VideoMAE model '%s' onto device '%s'...", MODEL_ID, _device)

    try:
        processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
        model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - we want to wrap any load failure
        raise RuntimeError(
            f"Failed to load VideoMAE model/processor for MODEL_ID='{MODEL_ID}'. "
            "This usually means either (1) there is no internet connection to "
            "download the checkpoint from the Hugging Face Hub, (2) the "
            "'transformers' cache is corrupted, or (3) MODEL_ID is invalid. "
            "Try running `huggingface-cli download " + MODEL_ID + "` manually "
            "to diagnose, or check your network connection."
        ) from exc

    model.to(_device)
    model.eval()

    _model = model
    _processor = processor
    logger.info("VideoMAE model loaded successfully.")
    return _model, _processor, _device


def _prepare_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Validate and normalize a list of frames to exactly NUM_FRAMES frames.

    - Drops None or otherwise invalid (non-3D / wrong-dtype) frames.
    - Pads by repeating the last valid frame if fewer than NUM_FRAMES remain.
    - Subsamples evenly if more than NUM_FRAMES are provided.

    Raises ValueError if no valid frames remain after filtering.
    """
    valid_frames: list[np.ndarray] = []
    for frame in frames:
        if frame is None:
            continue
        if not isinstance(frame, np.ndarray):
            continue
        if frame.ndim != 3 or frame.shape[2] != 3:
            continue
        valid_frames.append(frame)

    if len(valid_frames) == 0:
        raise ValueError(
            "classify_foul received no valid frames (all frames were None, "
            "corrupt, or not HxWx3 RGB uint8 arrays)."
        )

    n = len(valid_frames)

    if n == NUM_FRAMES:
        return valid_frames

    if n < NUM_FRAMES:
        padded = list(valid_frames)
        last_frame = valid_frames[-1]
        while len(padded) < NUM_FRAMES:
            padded.append(last_frame)
        return padded

    # n > NUM_FRAMES: subsample evenly across the clip.
    indices = np.linspace(0, n - 1, NUM_FRAMES).round().astype(int)
    return [valid_frames[i] for i in indices]


def _map_kinetics_logits_to_foul(logits: torch.Tensor, id2label: dict[int, str]) -> dict[str, float]:
    """
    Map raw Kinetics-400 classification logits to a foul-type prediction.

    Applies softmax, inspects the top-TOP_K predicted labels, and matches
    each label string against KINETICS_LABEL_TO_FOUL_KEYWORDS (case
    insensitive substring match). The foul type with the highest summed
    probability mass among top-K predictions wins.

    If no foul-mapped keyword is found among the top-K labels, the result
    is 'none' with confidence = 1.0 minus the summed probability of all
    foul-mapped labels found in the top-K (clamped to [0.05, 0.99]).

    Returns {"foul_type": str, "confidence": float}.
    """
    probs = F.softmax(logits, dim=-1)[0]

    debug_topk = torch.topk(probs, k=min(10, probs.shape[-1]))
    print("[foul_classifier] top-10 Kinetics predictions:", file=sys.stderr)
    for idx, prob in zip(debug_topk.indices.tolist(), debug_topk.values.tolist()):
        print(f"  {prob:.3f}  {id2label.get(idx, '<unknown>')}", file=sys.stderr)

    topk = torch.topk(probs, k=min(TOP_K, probs.shape[-1]))
    top_indices = topk.indices.tolist()
    top_probs = topk.values.tolist()

    foul_type_scores: dict[str, float] = {}
    total_foul_mass = 0.0

    for idx, prob in zip(top_indices, top_probs):
        label = id2label.get(idx, "").lower()
        matched_type: Optional[str] = None
        for keyword, foul_type in KINETICS_LABEL_TO_FOUL_KEYWORDS.items():
            if keyword in label:
                matched_type = foul_type
                break
        if matched_type is not None:
            foul_type_scores[matched_type] = foul_type_scores.get(matched_type, 0.0) + prob
            total_foul_mass += prob

    if not foul_type_scores:
        confidence = max(0.05, min(0.99, 1.0 - total_foul_mass))
        return {"foul_type": "none", "confidence": confidence}

    winning_type = max(foul_type_scores, key=foul_type_scores.get)
    confidence = max(0.05, min(0.99, foul_type_scores[winning_type]))
    return {"foul_type": winning_type, "confidence": confidence}


def classify_foul(frames: list[np.ndarray]) -> dict:
    """
    Classify the type of foul depicted in a short clip of RGB frames.

    Parameters
    ----------
    frames : list[np.ndarray]
        A list of RGB uint8 HxWx3 numpy arrays. May contain fewer or more
        than 16 frames; will be padded/subsampled internally. None or
        corrupt entries are skipped.

    Returns
    -------
    dict
        {"foul_type": str, "confidence": float}
        foul_type is one of: tackle, handball, obstruction, simulation,
        push, none.

    Raises
    ------
    ValueError
        If no valid frames are provided.
    RuntimeError
        If the underlying model fails to load (e.g. network failure).
    """
    prepared_frames = _prepare_frames(frames)
    model, processor, device = _load_model()

    inputs = processor(list(prepared_frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()

    id2label = model.config.id2label
    result = _map_kinetics_logits_to_foul(logits, id2label)

    assert result["foul_type"] in FOUL_TYPES, (
        f"Internal error: produced invalid foul_type '{result['foul_type']}'"
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"Running smoke test for foul_classifier.py (MODEL_ID='{MODEL_ID}')...")
    rng = np.random.default_rng(seed=42)
    dummy_frames = [
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8) for _ in range(16)
    ]

    result = classify_foul(dummy_frames)
    print("Smoke test result:", result)
