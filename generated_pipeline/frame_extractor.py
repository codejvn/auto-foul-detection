"""
frame_extractor.py
===================

Pure-OpenCV frame extraction module for the auto-foul-detection pipeline.

This module is responsible for turning a raw video clip into a fixed-size
set of evenly-spaced RGB frames (plus their timestamps in seconds) that
downstream ML modules (CLIP zero-shot heads, VideoMAE foul classifier,
dense optical flow severity estimator) can consume directly.

No ML model is used here -- only OpenCV (cv2) for video I/O and frame
decoding. Frames are returned as RGB uint8 numpy arrays (HxWx3) since most
HuggingFace vision processors (CLIP, VideoMAE) expect RGB input, while
OpenCV natively decodes frames as BGR.

SWAP HOOK (uniform sampling -> learned keyframe selector)
-----------------------------------------------------------
Today, frame *selection* is purely index-based: ``_select_frame_indices``
picks ``num_frames`` indices evenly spaced across ``[0, frame_count - 1]``.

To swap in a learned keyframe selector (e.g. a lightweight model that
scores frames by "eventfulness" -- motion spikes, player-contact
likelihood, whistle-adjacent audio cues, etc.) without touching the rest
of the pipeline:

    1. Implement a new function with the same signature as
       ``_select_frame_indices``:

           def _select_frame_indices_learned(
               frame_count: int, num_frames: int, video_path: str
           ) -> list[int]:
               ...

    2. Point ``extract_frames`` at it by reassigning the module-level
       ``FRAME_SELECTOR`` callable (see near the top of this file), e.g.:

           import frame_extractor
           frame_extractor.FRAME_SELECTOR = _select_frame_indices_learned

    3. Everything downstream (CLIP, VideoMAE, optical flow, ruling engine)
       is agnostic to *how* indices were chosen -- they only ever see the
       resulting ``(frames, timestamps)`` tuple, so no other module needs
       to change.

This mirrors the MODEL_ID swap-hook convention used by the other modules
in this pipeline, just applied to a selection *strategy* instead of a
model checkpoint.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Callable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration / swap hooks
# ---------------------------------------------------------------------------

#: Default number of frames to extract per clip. Kept as a module constant
#: so callers and tests can reference the same default used by the public
#: API's ``num_frames`` parameter default.
DEFAULT_NUM_FRAMES: int = 16

#: SWAP HOOK: reassign this to a different callable with the signature
#: ``(frame_count: int, num_frames: int, video_path: str) -> list[int]``
#: to change how frame indices are chosen (e.g. a learned keyframe
#: selector instead of uniform spacing). See module docstring above.
FRAME_SELECTOR: Callable[[int, int, str], list[int]]


def _select_frame_indices(frame_count: int, num_frames: int, video_path: str) -> list[int]:
    """Select ``num_frames`` evenly-spaced frame indices across a clip.

    Args:
        frame_count: Total number of frames in the video, as reported by
            ``cv2.CAP_PROP_FRAME_COUNT``. Must be a positive integer.
        num_frames: Desired number of frames to sample.
        video_path: Path to the source video (unused by this uniform
            strategy, but part of the stable signature so a learned
            selector can inspect the file if needed -- see SWAP HOOK).

    Returns:
        A sorted list of unique frame indices, each in
        ``[0, frame_count - 1]``, of length at most ``num_frames``.
    """
    del video_path  # Unused in the uniform-sampling strategy.

    if frame_count <= 0:
        return []

    if num_frames <= 0:
        return []

    if frame_count <= num_frames:
        # Not enough frames to satisfy the request; return every frame.
        return list(range(frame_count))

    # np.linspace over the valid index range, evenly spaced, inclusive of
    # both endpoints, then rounded to the nearest integer index.
    raw_indices = np.linspace(0, frame_count - 1, num=num_frames)
    indices = sorted(set(int(round(idx)) for idx in raw_indices))
    return indices


FRAME_SELECTOR = _select_frame_indices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_frames(
    video_path: str, num_frames: int = DEFAULT_NUM_FRAMES
) -> tuple[list[np.ndarray], list[float]]:
    """Extract evenly-spaced RGB frames (and their timestamps) from a video.

    Frames are sampled uniformly by frame index across the full clip
    (using ``cv2.CAP_PROP_FRAME_COUNT``), decoded via OpenCV, converted
    from BGR to RGB, and returned alongside per-frame timestamps in
    seconds.

    If an individual frame fails to decode (corrupt frame), the most
    recently successfully decoded frame is reused in its place and
    extraction continues. A ``ValueError`` is only raised if *zero*
    frames could be decoded from the entire clip.

    Args:
        video_path: Path to the input video file.
        num_frames: Number of frames to extract, evenly spaced across the
            clip. Defaults to 16.

    Returns:
        A tuple ``(frames, timestamps)`` where:
            - ``frames`` is a list of RGB uint8 numpy arrays, each shaped
              ``(H, W, 3)``.
            - ``timestamps`` is a list of floats (seconds) of the same
              length as ``frames``, one per extracted frame.

    Raises:
        FileNotFoundError: If ``video_path`` does not exist on disk.
        ValueError: If the video cannot be opened, reports zero/invalid
            frame count, or if zero frames could be successfully decoded.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video file not found: '{video_path}'. Please provide a valid path "
            "to an existing video file."
        )

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise ValueError(
                f"Failed to open video file: '{video_path}'. The file may be "
                "corrupt or in an unsupported format."
            )

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)

        if frame_count <= 0:
            # Some containers/codecs misreport frame count; fall back to a
            # manual scan to establish an upper bound.
            frame_count = _count_frames_by_scanning(capture)

        if frame_count <= 0:
            raise ValueError(
                f"Video '{video_path}' reports zero decodable frames; "
                "cannot extract frames from an empty or unreadable clip."
            )

        indices = FRAME_SELECTOR(frame_count, num_frames, video_path)
        if not indices:
            raise ValueError(
                f"No frame indices were selected for '{video_path}' "
                f"(frame_count={frame_count}, num_frames={num_frames})."
            )

        frames: list[np.ndarray] = []
        timestamps: list[float] = []
        last_good_frame: np.ndarray | None = None

        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, bgr_frame = capture.read()

            if success and bgr_frame is not None:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                last_good_frame = rgb_frame
            elif last_good_frame is not None:
                # Corrupt/unreadable frame: reuse the last good decode.
                rgb_frame = last_good_frame
            else:
                # No successful decode yet; skip this index and try the
                # next one. It will still be reused later once we have a
                # good frame, or the clip will fail overall if none decode.
                continue

            frames.append(rgb_frame)
            timestamps.append(_compute_timestamp(idx, fps, frame_count))

        if not frames:
            raise ValueError(
                f"Failed to decode any frames from '{video_path}'. The file "
                "may be corrupt or use an unsupported codec."
            )

        # Backfill any leading gap: if the very first selected indices
        # failed to decode (so `last_good_frame` was None at the time),
        # they were skipped above. Pad the front with the first
        # successfully decoded frame so frames/timestamps stay aligned
        # in length with the requested sampling where possible.
        missing = len(indices) - len(frames)
        if missing > 0 and frames:
            pad_frame = frames[0]
            pad_indices = indices[:missing]
            pad_frames = [pad_frame] * missing
            pad_timestamps = [
                _compute_timestamp(idx, fps, frame_count) for idx in pad_indices
            ]
            frames = pad_frames + frames
            timestamps = pad_timestamps + timestamps

        return frames, timestamps
    finally:
        capture.release()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_timestamp(frame_index: int, fps: float, frame_count: int) -> float:
    """Compute the timestamp (seconds) for a given frame index.

    Uses ``frame_index / fps`` when valid FPS metadata is available.
    Falls back to an index-based estimate (assuming a nominal 30 fps)
    when FPS metadata is missing, zero, or nonsensical.

    Args:
        frame_index: Zero-based index of the frame within the clip.
        fps: Frames-per-second reported by the video capture, may be
            zero/NaN/negative for malformed metadata.
        frame_count: Total frame count of the clip, used only to guard
            against degenerate inputs.

    Returns:
        The estimated timestamp in seconds as a float.
    """
    if fps and fps > 0 and not np.isnan(fps) and not np.isinf(fps):
        return float(frame_index) / float(fps)

    # Fallback: assume a nominal frame rate when metadata is unusable.
    nominal_fps = 30.0
    return float(frame_index) / nominal_fps


def _count_frames_by_scanning(capture: cv2.VideoCapture) -> int:
    """Manually count decodable frames by scanning the video.

    Used as a fallback when ``cv2.CAP_PROP_FRAME_COUNT`` reports an
    invalid (zero or negative) value, which can happen with certain
    containers/codecs that don't populate frame-count metadata.

    Args:
        capture: An already-opened ``cv2.VideoCapture`` instance.

    Returns:
        The number of frames successfully scanned. The capture's
        position is reset to the start (frame 0) before returning.
    """
    count = 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        success, frame = capture.read()
        if not success or frame is None:
            break
        count += 1

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return count


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _synthesize_dummy_video(path: str, width: int = 64, height: int = 48, fps: float = 24.0, num_frames: int = 40) -> None:
    """Write a small synthetic video of random RGB frames to ``path``.

    Used only by the ``__main__`` smoke test to avoid depending on any
    real video asset.

    Args:
        path: Output path for the synthesized ``.mp4`` file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames-per-second to encode the video at.
        num_frames: Total number of random frames to write.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for path: '{path}'")

    rng = np.random.default_rng(seed=42)
    try:
        for _ in range(num_frames):
            random_frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            writer.write(random_frame)
    finally:
        writer.release()


def _run_smoke_test() -> None:
    """Run a self-contained smoke test: synthesize a video, extract frames,
    print the results, and clean up temporary files."""
    temp_dir = tempfile.mkdtemp(prefix="frame_extractor_smoke_")
    video_path = os.path.join(temp_dir, "dummy_clip.mp4")

    try:
        print(f"[smoke test] Synthesizing dummy video at: {video_path}")
        _synthesize_dummy_video(video_path)

        print(f"[smoke test] Extracting {DEFAULT_NUM_FRAMES} frames...")
        frames, timestamps = extract_frames(video_path, num_frames=DEFAULT_NUM_FRAMES)

        print(f"[smoke test] Extracted {len(frames)} frames.")
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            print(
                f"  frame[{i:02d}] shape={frame.shape} dtype={frame.dtype} "
                f"timestamp={ts:.4f}s"
            )

        assert len(frames) == len(timestamps), "frames/timestamps length mismatch"
        assert all(f.ndim == 3 and f.shape[2] == 3 for f in frames), "expected HxWx3 frames"
        assert all(f.dtype == np.uint8 for f in frames), "expected uint8 frames"

        # Missing-file error path check.
        try:
            extract_frames(os.path.join(temp_dir, "does_not_exist.mp4"))
        except FileNotFoundError as exc:
            print(f"[smoke test] Correctly raised FileNotFoundError: {exc}")
        else:
            raise AssertionError("Expected FileNotFoundError for missing video path")

        print("[smoke test] PASSED.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[smoke test] Cleaned up temp dir: {temp_dir}")


if __name__ == "__main__":
    _run_smoke_test()
