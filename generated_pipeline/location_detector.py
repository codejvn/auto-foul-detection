"""
location_detector.py

Zero-shot penalty-box localization for the auto-foul-detection pipeline.

Uses CLIP (openai/clip-vit-base-patch32) as a frozen zero-shot classifier over
a bank of contrastive natural-language prompts: prompts that describe soccer
play happening inside/near the penalty area (goal mouth, goalkeeper, box lines
visible) versus prompts that describe midfield/open-field play far from either
goal. The mean positive-prompt probability across the middle portion of the
clip (the actual foul moment) is used as a location confidence score.

Model is cached as a module-level singleton so repeated calls (and calls from
other modules in the same process) do not repeatedly pay model load cost.

Public API:
    detect_location(frames: list[np.ndarray]) -> dict
        frames: RGB uint8 HxWx3 numpy arrays
        returns: {"in_penalty_box": bool, "confidence": float}
"""

from __future__ import annotations

import sys
import threading
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# MODEL_ID: swap-hook constant
# ---------------------------------------------------------------------------
# Current model: general-purpose CLIP zero-shot vision-language model.
#
# SWAP HOOK
# ---------
# To use a fine-tuned pitch-localization / camera-calibration model (e.g. a
# model trained on SoccerNet-calibration that directly regresses camera pose
# or field-line homography instead of doing zero-shot prompt matching),
# change the line below:
#
#     MODEL_ID = "openai/clip-vit-base-patch32"
#
# to, for example:
#
#     MODEL_ID = "your-org/soccernet-calibration-clip"
#
# As long as the replacement model exposes a CLIP-compatible
# `get_image_features` / `get_text_features` interface (or you adapt
# `_load_model` and `_score_frames` below to the new model's forward-pass
# signature), the rest of this module -- including the public
# `detect_location` return contract -- does not need to change.
# ---------------------------------------------------------------------------
MODEL_ID = "openai/clip-vit-base-patch32"

# Contrastive zero-shot prompts.
POSITIVE_PROMPTS: list[str] = [
    "a photo of a soccer match inside the penalty box near the goal",
    "a soccer player challenging near the goal mouth with the goalkeeper close by",
    "a football goalkeeper diving to save a shot inside the penalty area",
    "players contesting the ball right in front of the goal, penalty box lines visible",
    "a close-up soccer scene near the six-yard box and goalpost",
    "a penalty area incident with the goalkeeper and defenders near their own goal",
]

NEGATIVE_PROMPTS: list[str] = [
    "a photo of a soccer match in the midfield far from either goal",
    "soccer players passing the ball in the center circle of the pitch",
    "a wide shot of a football field with no goal or penalty box visible",
    "players running in open midfield space during a soccer match",
    "a soccer throw-in near the sideline far from the goal",
    "a general view of a football pitch during open play in midfield",
]

_ALL_PROMPTS: list[str] = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
_NUM_POSITIVE: int = len(POSITIVE_PROMPTS)

# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------
_model: Optional[CLIPModel] = None
_processor: Optional[CLIPProcessor] = None
_device: Optional[torch.device] = None
_load_lock = threading.Lock()


def _get_device() -> torch.device:
    """Return the best available torch device, preferring CUDA when present."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> tuple[CLIPModel, CLIPProcessor, torch.device]:
    """
    Load (or return the cached) CLIP model and processor singleton.

    Thread-safe lazy initialization. Raises a RuntimeError with a helpful
    message if the model cannot be downloaded/loaded (e.g. no internet
    connection on first run and no local HuggingFace cache present).

    Returns:
        Tuple of (model, processor, device).
    """
    global _model, _processor, _device

    if _model is not None and _processor is not None and _device is not None:
        return _model, _processor, _device

    with _load_lock:
        if _model is not None and _processor is not None and _device is not None:
            return _model, _processor, _device

        device = _get_device()
        try:
            model = CLIPModel.from_pretrained(MODEL_ID)
            processor = CLIPProcessor.from_pretrained(MODEL_ID)
        except Exception as exc:  # noqa: BLE001 - surface a clear actionable error
            raise RuntimeError(
                f"Failed to load CLIP model '{MODEL_ID}'. This usually means "
                "there is no internet connection for the first-time download "
                "and no local HuggingFace cache is available, or the "
                "'transformers'/'torch' packages are missing or incompatible. "
                f"Original error: {exc}"
            ) from exc

        model.to(device)
        model.eval()

        _model, _processor, _device = model, processor, device
        return _model, _processor, _device


def _select_middle_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Select the middle 50% of a list of frames (the foul moment).

    For a list of length N, returns the contiguous slice covering indices
    [N/4, 3N/4). Guarantees at least one frame is returned when the input
    list is non-empty.

    Args:
        frames: List of valid (non-None) frames, in temporal order.

    Returns:
        The middle-50% sub-list of frames.
    """
    n = len(frames)
    if n == 0:
        return []
    if n <= 2:
        # Too few frames to meaningfully trim; use all of them.
        return frames

    start = n // 4
    end = n - (n // 4)
    if end <= start:
        end = start + 1
    return frames[start:end]


def _filter_valid_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Filter out None or malformed entries from a list of candidate frames.

    A valid frame is a non-None numpy array with 3 dimensions and 3 channels
    (RGB), and non-zero spatial size.

    Args:
        frames: Raw list of frames, possibly containing None or corrupt data.

    Returns:
        List of frames that passed validation, in original order.
    """
    valid: list[np.ndarray] = []
    for frame in frames:
        if frame is None:
            continue
        if not isinstance(frame, np.ndarray):
            continue
        if frame.ndim != 3 or frame.shape[2] != 3:
            continue
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            continue
        valid.append(frame)
    return valid


@torch.inference_mode()
def _score_frames(
    frames: list[np.ndarray],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> float:
    """
    Compute the mean positive-prompt softmax probability across frames.

    For each frame, computes CLIP image-text similarity logits against the
    full prompt bank (positive + negative prompts), applies softmax over the
    prompt axis, sums the probability mass assigned to the positive prompts,
    then averages that positive-mass score across all input frames.

    Args:
        frames: Non-empty list of valid RGB uint8 frames.
        model: Loaded CLIPModel.
        processor: Loaded CLIPProcessor.
        device: Torch device the model is on.

    Returns:
        Mean positive-prompt probability mass, in [0.0, 1.0].
    """
    pil_images = [Image.fromarray(frame.astype(np.uint8), mode="RGB") for frame in frames]

    inputs = processor(
        text=_ALL_PROMPTS,
        images=pil_images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    # logits_per_image: (num_frames, num_prompts)
    logits_per_image = outputs.logits_per_image
    probs = torch.softmax(logits_per_image, dim=-1)

    positive_mass = probs[:, :_NUM_POSITIVE].sum(dim=-1)  # (num_frames,)
    mean_positive_mass = positive_mass.mean().item()

    return float(mean_positive_mass)


def detect_location(frames: list[np.ndarray]) -> dict:
    """
    Detect whether the foul incident occurs inside the penalty box.

    Runs zero-shot CLIP classification against contrastive location prompts
    on the middle 50% of the supplied frames (the foul moment), and averages
    the positive-prompt probability across those frames to produce a stable
    confidence score.

    Args:
        frames: List of RGB uint8 HxWx3 numpy arrays sampled from the clip.
            None or malformed entries are skipped.

    Returns:
        A dict with keys:
            "in_penalty_box": bool, True if confidence >= 0.5
            "confidence": float, mean positive-prompt probability in [0, 1]

    Raises:
        ValueError: If no valid frames remain after filtering.
    """
    valid_frames = _filter_valid_frames(frames)
    if not valid_frames:
        raise ValueError(
            "detect_location received no valid frames (all entries were None, "
            "non-numpy, or malformed). Expected a list of RGB uint8 HxWx3 "
            "numpy arrays."
        )

    middle_frames = _select_middle_frames(valid_frames)
    if not middle_frames:
        raise ValueError(
            "detect_location: middle-frame selection produced an empty list "
            "from a non-empty valid-frame list. This should not happen."
        )

    model, processor, device = _load_model()
    confidence = _score_frames(middle_frames, model, processor, device)

    return {
        "in_penalty_box": bool(confidence >= 0.5),
        "confidence": confidence,
    }


if __name__ == "__main__":
    print(f"location_detector.py smoke test — MODEL_ID = {MODEL_ID}")
    print("Loading CLIP model (first run will download weights)...")

    rng = np.random.default_rng(seed=42)
    dummy_frames = [
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8) for _ in range(8)
    ]

    try:
        result = detect_location(dummy_frames)
    except Exception as exc:  # noqa: BLE001
        print(f"Smoke test FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Smoke test result:")
    print(result)
    print("Smoke test passed.")
