"""
shot_boundary_filter.py
=======================

Shot-boundary (camera cut) filter for the auto-foul-detection pipeline.

Broadcast soccer footage frequently cuts between camera angles (wide shot,
close-up, replay). The 16 evenly-spaced frames returned by
``frame_extractor.extract_frames`` can therefore straddle one or more camera
cuts, which corrupts every downstream module: CLIP sees unrelated
compositions, VideoMAE sees an impossible "motion" across the cut, and the
Farneback optical-flow severity estimator registers the cut itself as a huge
motion spike.

This module sits directly after ``extract_frames`` in the pipeline. It takes
the ``(frames, timestamps)`` tuple, detects camera cuts *within the sampled
frame sequence*, and returns only the frames belonging to the longest
continuous shot (with their timestamps), so downstream modules always see a
single consistent camera angle.

Detection backends
------------------
1. **PySceneDetect ContentDetector** (preferred): HSV-space content delta,
   the same algorithm PySceneDetect uses for full videos, driven here
   frame-by-frame over the sparse sampled sequence. ``min_scene_len`` is set
   to 1 because our 16 sampled frames are already far apart in time -- the
   default of 15 (meant for consecutive video frames) would suppress every
   cut.
2. **MSE frame differencing** (fallback): mean squared error between
   consecutive frames; a cut is declared when MSE exceeds
   ``MSE_CUT_THRESHOLD`` (25.0). Used automatically when PySceneDetect is
   not installed, or when its API raises unexpectedly.

SWAP HOOK (detection backend)
-----------------------------
``filter_shot_boundaries`` accepts ``detector="auto" | "pyscenedetect" |
"mse"``. To plug in a different cut detector entirely (e.g. a learned
shot-boundary model such as TransNetV2), implement a function with the
signature ``(frames: list[np.ndarray]) -> list[int]`` returning the indices
where a new shot *starts*, and reassign the module-level ``CUT_DETECTOR``
callable:

    import shot_boundary_filter
    shot_boundary_filter.CUT_DETECTOR = my_detect_cuts

Everything downstream only ever sees the filtered ``(frames, timestamps)``
tuple, so no other module needs to change. This mirrors the FRAME_SELECTOR
convention in ``frame_extractor.py``.

GPU note: this module is CPU-only (adds ~0.2 GB peak host memory for
PySceneDetect's HSV buffers on full-resolution frames, zero GPU).
"""

from __future__ import annotations

import sys
from typing import Callable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration / swap hooks
# ---------------------------------------------------------------------------

#: MSE threshold for the frame-differencing fallback. Consecutive sampled
#: frames from the same shot (even ~0.3 s apart) typically score well below
#: this; a hard camera cut scores orders of magnitude above it.
MSE_CUT_THRESHOLD: float = 25.0

#: Content threshold passed to PySceneDetect's ContentDetector. Raised from
#: PySceneDetect's default (27.0) to reduce false positives on normal in-shot
#: scene changes (e.g., players moving within the same camera angle).
PYSCENEDETECT_THRESHOLD: float = 40.0

#: Minimum number of frames to keep after shot filtering. If the longest
#: continuous shot has fewer than this many frames, combine the top N longest
#: shots instead, trading some shot-cut contamination for robustness (downstream
#: optical flow and classifiers need sufficient frame count).
MIN_FRAMES_FLOOR: int = 8

#: SWAP HOOK: reassign this to a callable with the signature
#: ``(frames: list[np.ndarray]) -> list[int]`` (returning sorted indices at
#: which a new shot starts) to replace the cut-detection backend used by
#: ``detector="auto"``. See module docstring above.
CUT_DETECTOR: Callable[[list[np.ndarray]], list[int]]

try:  # PySceneDetect is optional; the MSE fallback covers its absence.
    from scenedetect.detectors import ContentDetector

    _PYSCENEDETECT_AVAILABLE = True
except ImportError:
    ContentDetector = None  # type: ignore[assignment]
    _PYSCENEDETECT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_shot_boundaries(
    frames: list[np.ndarray],
    timestamps: list[float],
    detector: str = "auto",
) -> tuple[list[np.ndarray], list[float]]:
    """Remove frames that fall outside the longest continuous camera shot.

    Detects camera cuts within the sampled frame sequence, splits the
    sequence into shots at those cuts, and returns the frames (and their
    timestamps) of the longest shot. "Longest" is measured in number of
    sampled frames; on a tie the earliest shot wins. If no cut is detected,
    the input is returned unchanged.

    Args:
        frames: RGB uint8 numpy arrays (HxWx3), as returned by
            ``frame_extractor.extract_frames``.
        timestamps: Per-frame timestamps in seconds, same length as
            ``frames``.
        detector: Cut-detection backend. ``"auto"`` (default) uses
            PySceneDetect's ContentDetector when installed and silently
            falls back to MSE frame differencing otherwise;
            ``"pyscenedetect"`` forces ContentDetector (raises if
            unavailable); ``"mse"`` forces the frame-differencing fallback.

    Returns:
        A tuple ``(kept_frames, kept_timestamps)`` containing only the
        frames of the longest continuous shot, in their original order.

    Raises:
        ValueError: If ``frames`` and ``timestamps`` differ in length, or
            if ``detector`` is not one of the recognized values.
        ImportError: If ``detector="pyscenedetect"`` is forced but
            PySceneDetect is not installed.
    """
    if len(frames) != len(timestamps):
        raise ValueError(
            f"frames ({len(frames)}) and timestamps ({len(timestamps)}) "
            "must have the same length."
        )

    # Zero or one frame: nothing to segment.
    if len(frames) < 2:
        return list(frames), list(timestamps)

    if detector == "auto":
        cuts = CUT_DETECTOR(frames)
    elif detector == "pyscenedetect":
        if not _PYSCENEDETECT_AVAILABLE:
            raise ImportError(
                "detector='pyscenedetect' was requested but PySceneDetect "
                "is not installed. Install it via `pip install scenedetect` "
                "or use detector='auto' / 'mse'."
            )
        cuts = _detect_cuts_pyscenedetect(frames)
    elif detector == "mse":
        cuts = _detect_cuts_mse(frames)
    else:
        raise ValueError(
            f"Unknown detector '{detector}'. "
            "Expected 'auto', 'pyscenedetect', or 'mse'."
        )

    start, end = _longest_shot_span(len(frames), cuts)
    kept_frames = frames[start:end]
    kept_timestamps = timestamps[start:end]
    print(
        f"[shot_boundary_filter] threshold={PYSCENEDETECT_THRESHOLD}, "
        f"detected_cuts={len(cuts)}, kept_frames={len(kept_frames)}/{len(frames)}",
        file=sys.stderr,
    )
    return kept_frames, kept_timestamps


# ---------------------------------------------------------------------------
# Cut-detection backends
# ---------------------------------------------------------------------------


def _detect_cuts_auto(frames: list[np.ndarray]) -> list[int]:
    """Detect cuts with ContentDetector when available, else MSE fallback.

    Also falls back to MSE if the PySceneDetect call raises (e.g. an API
    change in a future PySceneDetect release), so the pipeline never dies
    on the optional dependency.

    Args:
        frames: RGB uint8 numpy arrays (HxWx3).

    Returns:
        Sorted list of frame indices at which a new shot starts.
    """
    if _PYSCENEDETECT_AVAILABLE:
        try:
            return _detect_cuts_pyscenedetect(frames)
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash.
            print(
                "[shot_boundary_filter] PySceneDetect backend failed "
                f"({exc!r}); falling back to MSE frame differencing."
            )
    return _detect_cuts_mse(frames)


def _detect_cuts_pyscenedetect(frames: list[np.ndarray]) -> list[int]:
    """Detect cuts using PySceneDetect's ContentDetector, frame by frame.

    ContentDetector expects BGR frames (PySceneDetect is OpenCV-based), so
    each RGB frame is converted before being fed in. Frame "numbers" are
    simply the indices within the sampled sequence -- ContentDetector only
    uses them to label where cuts occur.

    Args:
        frames: RGB uint8 numpy arrays (HxWx3).

    Returns:
        Sorted list of frame indices at which a new shot starts.
    """
    # min_scene_len=1: our frames are sparsely sampled from the clip, so a
    # "shot" may legitimately span a single sampled frame. The default of
    # 15 assumes consecutive video frames and would mask every cut here.
    detector = ContentDetector(
        threshold=PYSCENEDETECT_THRESHOLD, min_scene_len=1
    )

    cuts: set[int] = set()
    for index, rgb_frame in enumerate(frames):
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cuts.update(detector.process_frame(index, bgr_frame))

    return sorted(cut for cut in cuts if 0 < cut < len(frames))


def _detect_cuts_mse(frames: list[np.ndarray]) -> list[int]:
    """Detect cuts via mean squared error between consecutive frames.

    A cut is declared between frames ``i-1`` and ``i`` when the per-pixel
    MSE (across all RGB channels, computed in float64) exceeds
    ``MSE_CUT_THRESHOLD``.

    Args:
        frames: RGB uint8 numpy arrays (HxWx3).

    Returns:
        Sorted list of frame indices at which a new shot starts.
    """
    cuts: list[int] = []
    for index in range(1, len(frames)):
        previous = frames[index - 1].astype(np.float64)
        current = frames[index].astype(np.float64)
        if previous.shape != current.shape:
            # Differing resolutions can only mean a source switch (e.g. a
            # replay wipe to different-sized content): treat as a cut.
            cuts.append(index)
            continue
        mse = float(np.mean((previous - current) ** 2))
        if mse > MSE_CUT_THRESHOLD:
            cuts.append(index)
    return cuts


CUT_DETECTOR = _detect_cuts_auto


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _longest_shot_span(num_frames: int, cuts: list[int]) -> tuple[int, int]:
    """Find the ``[start, end)`` span of the longest continuous shot.

    Splits ``range(num_frames)`` into contiguous segments at each cut index
    (a cut index marks the first frame of a *new* shot), returns the longest
    segment, or combines the top 2 longest if the longest is shorter than
    ``MIN_FRAMES_FLOOR`` (a safety floor to prevent filtering out too much).

    Args:
        num_frames: Total number of frames in the sequence.
        cuts: Sorted indices at which a new shot starts; values outside
            ``(0, num_frames)`` are ignored.

    Returns:
        A ``(start, end)`` tuple suitable for slicing the frame list.
    """
    boundaries = (
        [0]
        + [cut for cut in sorted(set(cuts)) if 0 < cut < num_frames]
        + [num_frames]
    )

    shots = [
        (start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    if not shots:
        return 0, num_frames

    shots_sorted_by_length = sorted(shots, key=lambda span: span[1] - span[0], reverse=True)
    longest_start, longest_end = shots_sorted_by_length[0]
    longest_length = longest_end - longest_start

    if longest_length >= MIN_FRAMES_FLOOR:
        return longest_start, longest_end

    if len(shots_sorted_by_length) >= 2:
        top_two = shots_sorted_by_length[:2]
        combined_start = min(s[0] for s in top_two)
        combined_end = max(s[1] for s in top_two)
        return combined_start, combined_end

    return longest_start, longest_end


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _synthesize_shot(
    base_color: tuple[int, int, int],
    num_frames: int,
    width: int = 64,
    height: int = 48,
    drift_per_frame: int = 2,
) -> list[np.ndarray]:
    """Create frames of a fake continuous shot: a solid color with drift.

    Each successive frame brightens slightly (``drift_per_frame`` per
    channel), mimicking gradual in-shot change while staying far below any
    cut threshold between consecutive frames.

    Args:
        base_color: Starting (R, G, B) fill color.
        num_frames: Number of frames to synthesize.
        width: Frame width in pixels.
        height: Frame height in pixels.
        drift_per_frame: Per-channel brightness increase per frame.

    Returns:
        List of RGB uint8 numpy arrays (HxWx3).
    """
    frames: list[np.ndarray] = []
    for i in range(num_frames):
        color = np.clip(
            np.array(base_color, dtype=np.int32) + i * drift_per_frame,
            0,
            255,
        ).astype(np.uint8)
        frames.append(np.full((height, width, 3), color, dtype=np.uint8))
    return frames


def _run_smoke_test() -> None:
    """Self-contained smoke test: synthesize a sequence with one obvious
    camera cut and verify the filter keeps only the longest shot."""
    # Shot A: 10 dark-teal frames; Shot B: 6 bright-red frames. The cut sits
    # between indices 9 and 10 -- large in both HSV content delta
    # (PySceneDetect) and raw MSE (fallback).
    shot_a = _synthesize_shot(base_color=(20, 60, 60), num_frames=10)
    shot_b = _synthesize_shot(base_color=(200, 40, 40), num_frames=6)
    frames = shot_a + shot_b
    timestamps = [round(i * 0.25, 2) for i in range(len(frames))]

    backends = ["mse"]
    if _PYSCENEDETECT_AVAILABLE:
        backends.append("pyscenedetect")
    else:
        print(
            "[smoke test] PySceneDetect not installed; "
            "testing MSE fallback only."
        )
    backends.append("auto")

    for backend in backends:
        print(f"[smoke test] detector='{backend}': filtering "
              f"{len(frames)} frames with a cut at index 10...")
        kept_frames, kept_timestamps = filter_shot_boundaries(
            frames, timestamps, detector=backend
        )
        print(
            f"[smoke test]   kept {len(kept_frames)} frames, "
            f"timestamps {kept_timestamps[0]:.2f}s..{kept_timestamps[-1]:.2f}s"
        )

        assert len(kept_frames) == len(kept_timestamps), (
            "frames/timestamps length mismatch after filtering"
        )
        assert len(kept_frames) == 10, (
            f"expected the 10-frame shot to survive, got {len(kept_frames)}"
        )
        assert kept_timestamps == timestamps[:10], (
            "expected timestamps of the first (longest) shot"
        )
        assert all(
            np.array_equal(kept, original)
            for kept, original in zip(kept_frames, shot_a)
        ), "kept frames should be exactly the first shot's frames"

    # No-cut sequence: everything survives.
    steady_frames = _synthesize_shot(base_color=(90, 90, 90), num_frames=8)
    steady_timestamps = [i * 0.25 for i in range(8)]
    kept_frames, kept_timestamps = filter_shot_boundaries(
        steady_frames, steady_timestamps
    )
    assert len(kept_frames) == 8, "no-cut sequence should pass through intact"
    assert kept_timestamps == steady_timestamps
    print("[smoke test] No-cut sequence passed through unchanged.")

    # Degenerate inputs.
    assert filter_shot_boundaries([], []) == ([], [])
    single_frame = _synthesize_shot(base_color=(10, 10, 10), num_frames=1)
    kept_frames, kept_timestamps = filter_shot_boundaries(single_frame, [0.0])
    assert len(kept_frames) == 1 and kept_timestamps == [0.0]
    print("[smoke test] Degenerate inputs (empty / single frame) handled.")

    # Length-mismatch error path.
    try:
        filter_shot_boundaries(steady_frames, steady_timestamps[:-1])
    except ValueError as exc:
        print(f"[smoke test] Correctly raised ValueError: {exc}")
    else:
        raise AssertionError("Expected ValueError for mismatched lengths")

    print("[smoke test] PASSED.")


if __name__ == "__main__":
    _run_smoke_test()
