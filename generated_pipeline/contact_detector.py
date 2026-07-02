"""
contact_detector.py

Zero-shot player-contact detection for the auto-foul-detection pipeline.

Uses a frozen CLIP (Contrastive Language-Image Pre-training) model to score
each extracted video frame against a set of contrastive text prompts:
prompts that describe two soccer players physically colliding / tackling
each other (positive) versus prompts that describe players simply running
with no contact (negative). Because contact between players is typically a
momentary event within a clip, the pipeline takes the MAX positive-prompt
probability across all sampled frames as the overall contact confidence,
rather than averaging.

Public API:
    detect_contact(frames: list[np.ndarray]) -> dict

This module is fully local and GPU-aware: the CLIP model runs on CUDA when
available and falls back to CPU automatically. The model and processor are
loaded lazily on first use and cached as module-level singletons so repeated
calls (e.g. once per video in a batch job) do not pay the model-load cost
more than once.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SWAP HOOK
# ---------------------------------------------------------------------------
# To swap in a fine-tuned checkpoint (e.g. one fine-tuned specifically on
# soccer broadcast footage for contact detection), change the line below:
#
#     MODEL_ID = "openai/clip-vit-base-patch32"
#
# to:
#
#     MODEL_ID = "your-org/clip-soccer-contact"
#
# The rest of this module (processor loading, forward pass, softmax logic)
# requires no changes as long as the replacement checkpoint is a standard
# HuggingFace `CLIPModel` / `CLIPProcessor` compatible repo.
#
# The CONTACT_PROMPTS / NO_CONTACT_PROMPTS text lists below are also tunable
# independently of the checkpoint swap -- refining prompt wording is often
# the cheapest way to improve zero-shot accuracy before investing in
# fine-tuning.
# ---------------------------------------------------------------------------
MODEL_ID = "openai/clip-vit-base-patch32"

#: Default number of frames the upstream frame-extraction stage samples per
#: clip. Provided here for documentation / downstream reference; this module
#: itself operates on whatever list of frames it is given.
DEFAULT_NUM_FRAMES = 16

#: Contrastive text prompts describing two soccer players in physical
#: contact, colliding, or tackling each other.
CONTACT_PROMPTS: list[str] = [
    "two soccer players colliding into each other",
    "a soccer player tackling another player",
    "two football players physically contacting each other during a challenge",
    "a defender pulling or grabbing an attacker's shirt",
    "two players pushing each other while contesting the ball",
    "a soccer player being tripped by an opponent",
    "a player shoving an opposing player during a match",
    "two soccer players clashing bodies in a physical duel for the ball",
]

#: Contrastive text prompts describing soccer players with no contact
#: between them.
NO_CONTACT_PROMPTS: list[str] = [
    "soccer players running on the field with no contact between them",
    "a soccer player dribbling the ball alone with no opponent nearby",
    "players spread out across the pitch with clear space between them",
    "a soccer player passing the ball with no other player close by",
    "two soccer players standing apart on the field",
    "a football player jogging without any nearby opponent",
    "an empty stretch of the soccer pitch with players far apart",
    "a soccer player taking a shot on goal with no defender close by",
]

#: Confidence threshold above which contact is declared to have occurred.
CONTACT_THRESHOLD = 0.5

# Module-level singleton cache for the lazily-loaded model and processor.
_model: Optional[CLIPModel] = None
_processor: Optional[CLIPProcessor] = None
_device: Optional[torch.device] = None


def _get_device() -> torch.device:
    """Return the torch device to run inference on (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> tuple[CLIPModel, CLIPProcessor, torch.device]:
    """
    Lazily load and cache the CLIP model and processor as module-level
    singletons.

    Returns:
        A tuple of (model, processor, device). The model is already moved
        to the target device and set to eval mode.

    Raises:
        RuntimeError: If the model or processor fails to download or load
            (e.g. due to no network access on first run, or a corrupted
            local cache), with a message pointing the user at the likely
            cause.
    """
    global _model, _processor, _device

    if _model is not None and _processor is not None and _device is not None:
        return _model, _processor, _device

    device = _get_device()
    logger.info("Loading CLIP model %s on device %s", MODEL_ID, device)

    try:
        model = CLIPModel.from_pretrained(MODEL_ID)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - we intentionally re-wrap any failure
        raise RuntimeError(
            f"Failed to load CLIP model/processor '{MODEL_ID}'. This is usually "
            "caused by no internet connection on first run (weights must be "
            "downloaded from the HuggingFace Hub and cached locally), an "
            "incomplete/corrupted local cache, or insufficient disk space. "
            "Verify connectivity and try again, or pre-download the model with "
            "`huggingface-cli download openai/clip-vit-base-patch32`."
        ) from exc

    model.to(device)
    model.eval()

    _model, _processor, _device = model, processor, device
    return _model, _processor, _device


def _filter_valid_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Filter out corrupt, None, or malformed entries from a list of frames.

    A valid frame is a non-None numpy array with shape (H, W, 3) and a
    non-zero number of pixels.

    Args:
        frames: Candidate list of RGB uint8 HxWx3 numpy arrays, possibly
            containing None or malformed entries.

    Returns:
        The subset of `frames` that are well-formed usable images.
    """
    valid: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        if frame is None:
            logger.warning("Skipping frame %d: frame is None", i)
            continue
        if not isinstance(frame, np.ndarray):
            logger.warning("Skipping frame %d: not a numpy array (got %s)", i, type(frame))
            continue
        if frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning("Skipping frame %d: unexpected shape %s", i, frame.shape)
            continue
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            logger.warning("Skipping frame %d: empty frame", i)
            continue
        valid.append(frame)
    return valid


def detect_contact(frames: list[np.ndarray]) -> dict:
    """
    Detect whether two soccer players are in physical contact in a sequence
    of video frames using zero-shot CLIP classification.

    Each frame is scored against a set of contrastive text prompts (contact
    vs. no-contact). A softmax over the combined prompt set yields, per
    frame, a probability mass assigned to the "contact" prompt group. Since
    contact is typically a brief, momentary event within a clip, the max
    such probability across all frames is used as the overall confidence
    rather than the mean.

    Args:
        frames: A list of RGB uint8 HxWx3 numpy arrays sampled evenly from
            the source video clip (16 frames by default upstream, but any
            non-empty list is accepted). Entries that are None or malformed
            are skipped.

    Returns:
        A dict with keys:
            "contact": bool, True if confidence >= CONTACT_THRESHOLD (0.5).
            "confidence": float, the max positive-prompt softmax probability
                across all valid frames, in [0.0, 1.0].

    Raises:
        ValueError: If `frames` is empty or contains no valid frames after
            filtering out corrupt/None entries.
        RuntimeError: If the underlying CLIP model fails to load (e.g. due
            to a network/download failure on first run).
    """
    if not frames:
        raise ValueError("detect_contact received an empty frame list.")

    valid_frames = _filter_valid_frames(frames)
    if not valid_frames:
        raise ValueError(
            "detect_contact received no valid frames after filtering out "
            "None/corrupt entries. Cannot run contact detection."
        )

    model, processor, device = _load_model()

    text_prompts = CONTACT_PROMPTS + NO_CONTACT_PROMPTS
    num_contact_prompts = len(CONTACT_PROMPTS)

    pil_images = [Image.fromarray(frame.astype(np.uint8), mode="RGB") for frame in valid_frames]

    inputs = processor(
        text=text_prompts,
        images=pil_images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image: (num_frames, num_prompts)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1)

    # Sum the probability mass assigned to the contact-prompt group for each
    # frame, then take the max across frames (contact is momentary).
    contact_mass_per_frame = probs[:, :num_contact_prompts].sum(dim=-1)
    confidence = float(contact_mass_per_frame.max().item())

    return {
        "contact": confidence >= CONTACT_THRESHOLD,
        "confidence": confidence,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"Running smoke test for contact_detector.py using model '{MODEL_ID}'...")
    rng = np.random.default_rng(seed=42)
    dummy_frames = [
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8) for _ in range(4)
    ]

    result = detect_contact(dummy_frames)
    print("detect_contact result on random dummy frames:")
    print(result)
