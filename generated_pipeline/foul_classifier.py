"""
foul_classifier.py
===================

Foul-type AND severity classification module for the auto-foul-detection
pipeline.

This module loads the fine-tuned `DualHeadVideoMAE` checkpoint produced by
`train_foul_classifier.py` (VideoMAE backbone + fc_norm + a 9-way action
head + a 4-way offence-severity head) and forwards prepared clips through
it directly -- no keyword mapping over a generic Kinetics-400 label set is
involved anymore. The 9-class action prediction is collapsed onto the
original 6-term foul_type vocabulary pipeline.py's callers expect via
ACTION_TO_FOUL_TYPE. It is one signal among several (CLIP contact/box-
location heads, optical-flow severity, ruling engine) that are combined
elsewhere in the pipeline.

Public API
----------
classify_foul(frames: list[np.ndarray]) -> dict
    frames: list of RGB uint8 HxWx3 numpy arrays (any length >= 1)
    returns: {
        "foul_type": str,               # 6-vocab, for pipeline.py callers
        "confidence": float,            # max action-head softmax prob
        "severity": str,                # 4-class label from the severity head
        "severity_confidence": float,
        "action_class_probs": dict[str, float],    # full 9-class softmax
        "severity_class_probs": dict[str, float],  # full 4-class softmax
    }
    where foul_type is one of: tackle, handball, obstruction, simulation,
    push, none. NOTE: "handball" is not currently reachable -- see
    ACTION_TO_FOUL_TYPE below.

Design notes
------------
- VideoMAE requires exactly 16 frames per clip. Shorter sequences are
  padded by repeating the final frame; longer sequences are subsampled
  evenly across the clip.
- Preprocessing replicates `train_foul_classifier.py`'s `FoulClipDataset`
  EXACTLY: torchvision `Resize((224, 224))` (squash, ignores aspect ratio)
  + `ToTensor()` + `Normalize(processor.image_mean, processor.image_std)`.
  We deliberately do NOT use `VideoMAEImageProcessor.__call__`, which does
  shortest-edge-resize + center-crop -- a different framing that would
  mis-feed a model trained on the squashed preprocessing above.
- The model, processor-derived transform, device, and class-name lists are
  cached as module-level singletons and lazily loaded on first use.
- Uses CUDA when available, otherwise falls back to CPU automatically.

REDUNDANCY NOTE (severity_assessor.py)
---------------------------------------
This module now emits a direct, trained severity signal (`severity` /
`severity_class_probs`) from the checkpoint's severity_head, which makes
`severity_assessor.py`'s Farneback optical-flow heuristic redundant as a
severity estimator. `severity_assessor.py` is intentionally left in place
and unwired here pending validation of the trained head's accuracy against
the VARS dataset (see validator.py) -- this module does not import from or
modify severity_assessor.py or pipeline.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import VideoMAEImageProcessor

logger = logging.getLogger(__name__)

# The base VideoMAE checkpoint the fine-tuned dual-head model is built on
# (backbone architecture + image processor mean/std source).
MODEL_ID: str = "MCG-NJU/videomae-base-finetuned-kinetics"

# Fine-tuned dual-head checkpoint produced by train_foul_classifier.py.
# Resolved relative to the repo root (parent of generated_pipeline/) so it
# is robust to the working directory the pipeline is invoked from.
CHECKPOINT_PATH: Path = (
    Path(__file__).resolve().parent.parent / "checkpoints" / "videomae-foul-best.pt"
)

# Number of frames VideoMAE expects per clip.
NUM_FRAMES: int = 16

# Square side length frames are resized to before feeding the model (must
# match training's torchvision Resize((224, 224)) squash).
IMG_SIZE: int = 224

# The complete set of foul categories this module can emit.
FOUL_TYPES = ("tackle", "handball", "obstruction", "simulation", "push", "none")

# Maps the fine-tuned 9-class action label -> the original 6-term foul_type
# vocabulary pipeline.py's callers expect. This COLLAPSES distinctions:
# Tackling, Standing tackling, High leg, and Challenge all -> "tackle";
# Elbowing -> "push". NOTE: "handball" is NOT reachable from this head --
# the action head has no handball class (handball was a separate label
# deferred during training), so classify_foul can never return "handball"
# now.
ACTION_TO_FOUL_TYPE: dict[str, str] = {
    "Tackling": "tackle",
    "Standing tackling": "tackle",
    "High leg": "tackle",
    "Holding": "obstruction",
    "Pushing": "push",
    "Elbowing": "push",
    "Challenge": "tackle",
    "Dive": "simulation",
    "none": "none",
}

# Module-level singleton cache for the model, preprocessing transform,
# device, and the checkpoint's class-name lists.
_model = None
_transform: Optional[transforms.Compose] = None
_device: Optional[torch.device] = None
_action_classes: Optional[list[str]] = None
_severity_classes: Optional[list[str]] = None

# Ensures the severity_assessor redundancy warning is only logged once.
_redundancy_warning_emitted: bool = False


def _get_device() -> torch.device:
    """Return the best available torch device, preferring CUDA when present."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    """
    Lazily load and cache the fine-tuned DualHeadVideoMAE model, its
    preprocessing transform, device, and class-name lists.

    Returns (model, transform, device, action_classes, severity_classes) on
    subsequent calls from cache. Raises FileNotFoundError if the checkpoint
    is missing, or RuntimeError if loading fails (network/cache issues or
    an architecture/checkpoint mismatch).
    """
    global _model, _transform, _device, _action_classes, _severity_classes
    global _redundancy_warning_emitted

    if (
        _model is not None
        and _transform is not None
        and _device is not None
        and _action_classes is not None
        and _severity_classes is not None
    ):
        return _model, _transform, _device, _action_classes, _severity_classes

    # Lazy import: keeps this module's import light and reuses the exact
    # architecture train_foul_classifier.py trained, guaranteeing
    # load_state_dict(strict=True) succeeds.
    from train_foul_classifier import DualHeadVideoMAE

    _device = _get_device()
    logger.info(
        "Loading fine-tuned DualHeadVideoMAE from '%s' onto device '%s'...",
        CHECKPOINT_PATH,
        _device,
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found at '{CHECKPOINT_PATH}'. Run "
            "train_foul_classifier.py to produce it before using "
            "foul_classifier.classify_foul()."
        )

    try:
        processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=processor.image_mean, std=processor.image_std
                ),
            ]
        )
        model = DualHeadVideoMAE(MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - we want to wrap any load failure
        raise RuntimeError(
            f"Failed to load base VideoMAE model/processor for MODEL_ID='{MODEL_ID}'. "
            "This usually means either (1) there is no internet connection to "
            "download the checkpoint from the Hugging Face Hub, (2) the "
            "'transformers' cache is corrupted, or (3) MODEL_ID is invalid. "
            "Try running `huggingface-cli download " + MODEL_ID + "` manually "
            "to diagnose, or check your network connection."
        ) from exc

    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except Exception as exc:  # noqa: BLE001 - wrap architecture/checkpoint mismatches
        raise RuntimeError(
            f"Failed to load state dict from checkpoint '{CHECKPOINT_PATH}' into "
            "DualHeadVideoMAE. This usually means the checkpoint was trained "
            "with a different architecture than train_foul_classifier.py "
            "currently defines. Re-run training to regenerate a compatible "
            "checkpoint, or check that train_foul_classifier.py hasn't "
            "diverged from the checkpoint's architecture."
        ) from exc

    model.to(_device)
    model.eval()

    action_classes = list(ckpt["action_classes"])
    severity_classes = list(ckpt["offence_severity_classes"])

    if not _redundancy_warning_emitted:
        logger.warning(
            "foul_classifier now emits a direct trained severity signal "
            "(severity/severity_class_probs) from the checkpoint's "
            "severity_head, making severity_assessor.py's Farneback "
            "optical-flow heuristic redundant as a severity estimator. "
            "severity_assessor.py is intentionally left in place / unwired "
            "pending validation."
        )
        _redundancy_warning_emitted = True

    _model = model
    _transform = transform
    _action_classes = action_classes
    _severity_classes = severity_classes
    logger.info("Fine-tuned DualHeadVideoMAE loaded successfully.")
    return _model, _transform, _device, _action_classes, _severity_classes


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


def classify_foul(frames: list[np.ndarray]) -> dict:
    """
    Classify the foul type AND severity depicted in a short clip of RGB
    frames, via a direct forward pass through the fine-tuned dual-head
    VideoMAE model.

    Parameters
    ----------
    frames : list[np.ndarray]
        A list of RGB uint8 HxWx3 numpy arrays. May contain fewer or more
        than 16 frames; will be padded/subsampled internally. None or
        corrupt entries are skipped.

    Returns
    -------
    dict
        {
            "foul_type": str,
            "confidence": float,
            "severity": str,
            "severity_confidence": float,
            "action_class_probs": dict[str, float],
            "severity_class_probs": dict[str, float],
        }
        foul_type is one of: tackle, handball, obstruction, simulation,
        push, none.

    Raises
    ------
    ValueError
        If no valid frames are provided.
    FileNotFoundError
        If the fine-tuned checkpoint is missing.
    RuntimeError
        If the underlying model fails to load (e.g. network failure or
        architecture/checkpoint mismatch).
    """
    prepared_frames = _prepare_frames(frames)
    model, transform, device, action_classes, severity_classes = _load_model()

    pixel_values = torch.stack(
        [transform(frame) for frame in prepared_frames]
    ).unsqueeze(0).to(device)  # (1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        action_logits, severity_logits = model(pixel_values)

    action_probs = F.softmax(action_logits, dim=-1)[0].cpu()
    severity_probs = F.softmax(severity_logits, dim=-1)[0].cpu()

    action_idx = int(torch.argmax(action_probs).item())
    action_label = action_classes[action_idx]
    confidence = float(action_probs[action_idx])

    severity_idx = int(torch.argmax(severity_probs).item())
    severity = severity_classes[severity_idx]
    severity_confidence = float(severity_probs[severity_idx])

    if action_label not in ACTION_TO_FOUL_TYPE:
        logger.warning(
            "Unexpected action label '%s' not present in ACTION_TO_FOUL_TYPE; "
            "falling back to foul_type='none'.",
            action_label,
        )
    foul_type = ACTION_TO_FOUL_TYPE.get(action_label, "none")

    action_class_probs = {
        action_classes[i]: float(action_probs[i]) for i in range(len(action_classes))
    }
    severity_class_probs = {
        severity_classes[i]: float(severity_probs[i])
        for i in range(len(severity_classes))
    }

    result = {
        "foul_type": foul_type,
        "confidence": confidence,
        "severity": severity,
        "severity_confidence": severity_confidence,
        "action_class_probs": action_class_probs,
        "severity_class_probs": severity_class_probs,
    }

    if result["foul_type"] not in FOUL_TYPES:
        raise RuntimeError(
            f"Internal error: produced invalid foul_type '{result['foul_type']}'"
        )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"Running smoke test for foul_classifier.py (checkpoint='{CHECKPOINT_PATH}')...")
    rng = np.random.default_rng(seed=42)
    dummy_frames = [
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8) for _ in range(16)
    ]

    result = classify_foul(dummy_frames)
    print("Smoke test result:", result)

    if result["foul_type"] not in FOUL_TYPES:
        raise RuntimeError(
            f"FAIL: foul_type '{result['foul_type']}' not in FOUL_TYPES {FOUL_TYPES}"
        )

    if len(result["action_class_probs"]) != 9:
        raise RuntimeError(
            f"FAIL: expected 9 action classes, got {len(result['action_class_probs'])}"
        )

    if len(result["severity_class_probs"]) != 4:
        raise RuntimeError(
            f"FAIL: expected 4 severity classes, got {len(result['severity_class_probs'])}"
        )

    for name, probs in (
        ("action_class_probs", result["action_class_probs"]),
        ("severity_class_probs", result["severity_class_probs"]),
    ):
        values = list(probs.values())
        if any(np.isnan(v) or np.isinf(v) for v in values):
            raise RuntimeError(f"FAIL: {name} contains NaN/inf: {probs}")
        total = sum(values)
        if abs(total - 1.0) > 1e-4:
            raise RuntimeError(f"FAIL: {name} sums to {total}, expected 1.0 (+/- 1e-4)")

    if not (0.0 <= result["confidence"] <= 1.0):
        raise RuntimeError(f"FAIL: confidence {result['confidence']} not in [0, 1]")

    if not (0.0 <= result["severity_confidence"] <= 1.0):
        raise RuntimeError(
            f"FAIL: severity_confidence {result['severity_confidence']} not in [0, 1]"
        )

    print(
        "PASS: foul_type valid, both action_class_probs (9) and "
        "severity_class_probs (4) are valid probability distributions "
        "(sum to 1.0, no NaN/inf), confidences in [0, 1]."
    )
