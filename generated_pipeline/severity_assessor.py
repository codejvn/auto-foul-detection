"""
severity_assessor.py

Motion-based foul severity assessment for the local soccer foul detection
pipeline.

This module deliberately avoids any machine-learning model. It estimates how
"severe" a foul was purely from dense optical flow computed between the RGB
frames extracted from the clip. The intuition: a high-speed, high-impact
challenge (excessive force) produces a much larger peak in relative pixel
motion between consecutive frames than a mistimed but controlled tackle
(careless), with reckless challenges falling in between.

Pipeline position:
    OpenCV frame extraction -> CLIP contact/box heads -> VideoMAE foul-type
    classifier -> [THIS MODULE: optical-flow severity] -> ruling engine

Public API:
    assess_severity(frames: list[np.ndarray], foul_type: str) -> dict
"""

from __future__ import annotations

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Module-level constants
# --------------------------------------------------------------------------

#: Severity labels, in increasing order of seriousness.
SEVERITY_CARELESS = "careless"
SEVERITY_RECKLESS = "reckless"
SEVERITY_EXCESSIVE_FORCE = "excessive_force"

#: Foul types recognized by upstream classifiers (VideoMAE head).
VALID_FOUL_TYPES = frozenset(
    {"tackle", "handball", "obstruction", "simulation", "push", "none"}
)

#: Width (pixels) frames are downscaled to before optical flow is computed.
#: Aspect ratio is preserved. Smaller frames make Farneback flow much faster
#: without materially harming the reliability of the 95th-percentile motion
#: statistic.
FLOW_TARGET_WIDTH = 320

#: Normalized peak motion thresholds. "Normalized" means the raw 95th
#: percentile flow magnitude (in downscaled pixels) has been divided by the
#: diagonal length of the downscaled frame, making the statistic roughly
#: scale-invariant across different source resolutions.
CARELESS_MAX = 0.010
RECKLESS_MAX = 0.025

#: Percentile used to summarize the flow-magnitude field for a single frame
#: pair. The 95th percentile is used (rather than the max) to stay robust to
#: single-pixel noise spikes in the flow field.
FLOW_MAGNITUDE_PERCENTILE = 95.0

#: Farneback optical flow parameters. These are reasonable general-purpose
#: defaults for short, low-resolution sports clips.
_FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def _confidence_from_motion(normalized_motion: float) -> float:
    """Derive a confidence score from proximity to a threshold boundary.

    The heuristic decision boundaries (``CARELESS_MAX`` and ``RECKLESS_MAX``)
    are the points of maximum ambiguity: a value sitting exactly on a
    boundary is a coin flip between the two adjacent severities. A value far
    from any boundary (e.g. deep inside the "excessive_force" band, or
    exactly zero motion) is a confident call.

    The distance to the nearest boundary is normalized by half the width of
    the "reckless" band (the smaller of the two finite bands), then mapped
    linearly into the confidence range [0.5, 0.9] and clamped.

    Args:
        normalized_motion: The normalized peak motion statistic.

    Returns:
        A confidence value in the closed interval [0.5, 0.9].
    """
    boundaries = (CARELESS_MAX, RECKLESS_MAX)
    distance_to_nearest_boundary = min(
        abs(normalized_motion - boundary) for boundary in boundaries
    )

    # Half the width of the reckless band sets the scale over which
    # confidence ramps from "coin flip" to "confident".
    band_half_width = (RECKLESS_MAX - CARELESS_MAX) / 2.0
    scale = band_half_width if band_half_width > 0 else 1e-6

    ramp = distance_to_nearest_boundary / scale  # 0 at boundary, grows outward
    confidence = 0.5 + 0.4 * min(ramp, 1.0)
    return float(min(max(confidence, 0.5), 0.9))


def _to_grayscale_downscaled(frame: np.ndarray) -> np.ndarray | None:
    """Convert an RGB frame to a downscaled grayscale frame for flow.

    Args:
        frame: An RGB uint8 HxWx3 numpy array, or possibly a malformed /
            None value coming from an upstream frame extractor.

    Returns:
        A single-channel uint8 numpy array downscaled to width
        ``FLOW_TARGET_WIDTH`` (aspect-preserved), or ``None`` if the input
        frame is missing or malformed and should be skipped.
    """
    if frame is None:
        return None
    if not isinstance(frame, np.ndarray):
        return None
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return None

    scale = FLOW_TARGET_WIDTH / float(width)
    target_height = max(1, int(round(height * scale)))
    target_width = FLOW_TARGET_WIDTH

    try:
        resized = cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        return None

    return gray


def _compute_peak_normalized_motion(frames: list[np.ndarray]) -> float:
    """Compute the peak normalized 95th-percentile flow magnitude.

    Dense Farneback optical flow is computed between every pair of
    consecutive valid (non-corrupt, non-None) grayscale frames. For each
    pair, the 95th percentile of the flow-magnitude field is taken as that
    pair's motion score. The maximum such score across all pairs is the
    "peak motion" for the clip, normalized by the diagonal of the
    downscaled frame so the statistic is roughly resolution-independent.

    Args:
        frames: List of RGB uint8 frames (may include None / corrupt
            entries, which are skipped).

    Returns:
        The normalized peak motion as a float >= 0.0. Returns 0.0 if fewer
        than two valid frames are available.
    """
    valid_gray_frames = [
        gray
        for gray in (_to_grayscale_downscaled(frame) for frame in frames)
        if gray is not None
    ]

    if len(valid_gray_frames) < 2:
        return 0.0

    height, width = valid_gray_frames[0].shape[:2]
    diagonal = float(np.hypot(height, width))
    if diagonal <= 0:
        return 0.0

    peak_motion = 0.0
    for prev_gray, next_gray in zip(valid_gray_frames[:-1], valid_gray_frames[1:]):
        if prev_gray.shape != next_gray.shape:
            continue

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None, **_FARNEBACK_PARAMS
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        pair_score = float(np.percentile(magnitude, FLOW_MAGNITUDE_PERCENTILE))
        normalized_score = pair_score / diagonal
        peak_motion = max(peak_motion, normalized_score)

    return peak_motion


def _severity_from_motion(normalized_motion: float) -> str:
    """Map a normalized peak motion value to a severity label.

    Args:
        normalized_motion: The normalized peak motion statistic.

    Returns:
        One of ``"careless"``, ``"reckless"``, ``"excessive_force"``.
    """
    if normalized_motion < CARELESS_MAX:
        return SEVERITY_CARELESS
    if normalized_motion <= RECKLESS_MAX:
        return SEVERITY_RECKLESS
    return SEVERITY_EXCESSIVE_FORCE


def _apply_foul_type_override(severity: str, foul_type: str) -> str:
    """Apply foul-type-aware overrides to a motion-derived severity.

    Two overrides are enforced regardless of measured motion:

    - ``simulation``: there is no real contact by definition, so the
      severity is always downgraded to ``"careless"``.
    - ``handball``: capped at ``"reckless"`` since handballs, even
      forceful ones, are not physically assessed the way tackles are
      under the "excessive force" standard.

    Args:
        severity: The motion-derived severity label.
        foul_type: The foul type classified upstream.

    Returns:
        The (possibly overridden) severity label.
    """
    if foul_type == "simulation":
        return SEVERITY_CARELESS
    if foul_type == "handball" and severity == SEVERITY_EXCESSIVE_FORCE:
        return SEVERITY_RECKLESS
    return severity


def assess_severity(frames: list[np.ndarray], foul_type: str) -> dict:
    """Assess foul severity from dense optical flow motion in the clip.

    Computes dense Farneback optical flow between consecutive grayscale,
    downscaled frames, uses the peak (across frame pairs) 95th-percentile
    flow magnitude — normalized by the downscaled frame diagonal — as a
    proxy for physical intensity of contact, then maps that value to a
    discrete severity label via fixed thresholds, with foul-type-aware
    overrides for ``simulation`` and ``handball``.

    Args:
        frames: List of RGB uint8 HxWx3 numpy arrays sampled evenly from
            the foul clip. Entries may be ``None`` or malformed; such
            entries are skipped.
        foul_type: One of ``"tackle"``, ``"handball"``, ``"obstruction"``,
            ``"simulation"``, ``"push"``, ``"none"``.

    Returns:
        A dict with keys:
            - ``"severity"``: one of ``"careless"``, ``"reckless"``,
              ``"excessive_force"``.
            - ``"confidence"``: float in [0.5, 0.9].
            - ``"peak_motion"``: the normalized peak motion float (>= 0.0)
              used to derive the severity.
    """
    if foul_type not in VALID_FOUL_TYPES:
        raise ValueError(
            f"Unknown foul_type {foul_type!r}; expected one of "
            f"{sorted(VALID_FOUL_TYPES)}"
        )

    valid_frame_count = sum(
        1 for frame in frames if _to_grayscale_downscaled(frame) is not None
    )
    if valid_frame_count < 2:
        return {
            "severity": SEVERITY_CARELESS,
            "confidence": 0.5,
            "peak_motion": 0.0,
        }

    peak_motion = _compute_peak_normalized_motion(frames)
    severity = _severity_from_motion(peak_motion)
    severity = _apply_foul_type_override(severity, foul_type)
    confidence = _confidence_from_motion(peak_motion)

    return {
        "severity": severity,
        "confidence": confidence,
        "peak_motion": peak_motion,
    }


# --------------------------------------------------------------------------
# SWAP HOOK: replacing the optical-flow heuristic with a fine-tuned model
# --------------------------------------------------------------------------
#
# The heuristic above is a fast, zero-training baseline. If a labeled
# dataset of foul clips with severity annotations (careless / reckless /
# excessive_force) becomes available, this module can be swapped for a
# fine-tuned video classifier while keeping every downstream module
# (the ruling engine, API layer, etc.) completely unchanged, because they
# only ever depend on the `assess_severity(frames, foul_type) -> dict`
# signature and its three output keys.
#
# MODEL_ID = "your-org/videomae-severity"  # placeholder — not loaded by default
#
# To reimplement `assess_severity` around a fine-tuned model:
#
#   1. Add `torch`, `transformers` (VideoMAEForVideoClassification +
#      VideoMAEImageProcessor) imports at the top of this file.
#   2. At module load time (or lazily, on first call, to keep import-time
#      side effects minimal), load the processor and model once:
#          _processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
#          _model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
#          _model.eval().to(_device)
#      Cache both in module-level globals so repeated calls to
#      `assess_severity` don't reload the model.
#   3. Inside `assess_severity`, replace the call to
#      `_compute_peak_normalized_motion` with:
#        a. Filter `frames` to valid (non-None, well-formed) entries, same
#           as today; if fewer than 2 remain, keep the existing early-return
#           behavior for parity with the heuristic.
#        b. Run `inputs = _processor(list(valid_frames), return_tensors="pt")`
#           and move tensors to `_device`.
#        c. `with torch.no_grad(): logits = _model(**inputs).logits`
#        d. `probs = logits.softmax(dim=-1)[0]`; take
#           `pred_idx = int(probs.argmax())` and map it through the model's
#           `id2label` (fine-tuned with 3 classes: careless / reckless /
#           excessive_force) to get `severity`.
#        e. Set `confidence = float(probs[pred_idx])`, clamped to whatever
#           range downstream consumers expect (or leave unclamped if the
#           [0.5, 0.9] convention is dropped along with the heuristic).
#        f. Still compute `peak_motion` via `_compute_peak_normalized_motion`
#           (cheap, useful as an auxiliary diagnostic signal / feature for
#           the ruling engine) and include it unchanged in the returned dict.
#        g. Keep applying `_apply_foul_type_override` to the model's
#           severity prediction, since the simulation/handball business
#           rules are policy decisions independent of the perception model.
#   4. Return the same dict shape: {"severity": ..., "confidence": ...,
#      "peak_motion": ...}. No other module needs to change.
#
# --------------------------------------------------------------------------


if __name__ == "__main__":
    def _make_synthetic_frames(
        num_frames: int, height: int, width: int, step: int
    ) -> list[np.ndarray]:
        """Build synthetic RGB frames with a bright square moving each frame.

        Args:
            num_frames: Number of frames to generate.
            height: Frame height in pixels.
            width: Frame width in pixels.
            step: Pixels the square moves to the right per frame.

        Returns:
            List of RGB uint8 HxWx3 numpy arrays.
        """
        frames: list[np.ndarray] = []
        # Sized so the square covers a large enough fraction of the frame
        # (well above the 5% implied by the 95th-percentile flow statistic)
        # that its motion reliably shows up in the peak-motion measurement.
        square_size = min(height, width) // 3
        top = height // 2 - square_size // 2

        # Farneback flow is gradient-based, so a flat-colored square only
        # yields a trackable signal at its thin edge border (interior pixels
        # have no local texture to match frame-to-frame). A checkerboard
        # texture inside the square gives every pixel a gradient, so the
        # whole square area — not just its edges — contributes to the flow
        # field, producing a realistic, clearly nonzero motion signal.
        checker = np.indices((square_size, square_size)).sum(axis=0) % 16 < 8
        square_pattern = np.where(checker, 255, 160).astype(np.uint8)

        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, :] = 20  # dim background so the square stands out
            left = min(width - square_size, 10 + i * step)
            for channel in range(3):
                frame[
                    top : top + square_size, left : left + square_size, channel
                ] = square_pattern
            frames.append(frame)
        return frames

    print("=== Smoke test: tackle with moving bright square ===")
    tackle_frames = _make_synthetic_frames(
        num_frames=8, height=240, width=320, step=6
    )
    tackle_result = assess_severity(tackle_frames, foul_type="tackle")
    print(tackle_result)
    assert tackle_result["severity"] in {
        SEVERITY_CARELESS,
        SEVERITY_RECKLESS,
        SEVERITY_EXCESSIVE_FORCE,
    }
    assert 0.5 <= tackle_result["confidence"] <= 0.9
    assert tackle_result["peak_motion"] >= 0.0
    print("OK: tackle result well-formed.\n")

    print("=== Smoke test: simulation override (must be careless) ===")
    simulation_frames = _make_synthetic_frames(
        num_frames=8, height=240, width=320, step=30
    )
    simulation_result = assess_severity(simulation_frames, foul_type="simulation")
    print(simulation_result)
    assert simulation_result["severity"] == SEVERITY_CARELESS
    print("OK: simulation forced to careless regardless of motion.\n")

    print("=== Smoke test: handball cap (must not exceed reckless) ===")
    handball_frames = _make_synthetic_frames(
        num_frames=8, height=240, width=320, step=30
    )
    handball_result = assess_severity(handball_frames, foul_type="handball")
    print(handball_result)
    assert handball_result["severity"] in {SEVERITY_CARELESS, SEVERITY_RECKLESS}
    print("OK: handball capped at reckless.\n")

    print("=== Edge case: fewer than 2 valid frames ===")
    edge_result = assess_severity([tackle_frames[0], None], foul_type="push")
    print(edge_result)
    assert edge_result == {
        "severity": SEVERITY_CARELESS,
        "confidence": 0.5,
        "peak_motion": 0.0,
    }
    print("OK: edge case handled.\n")

    print("All smoke tests passed.")
