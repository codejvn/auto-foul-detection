"""End-to-end orchestrator for the local, GPU-accelerated soccer foul detection pipeline.

This module is the single CLI entry point for the pipeline. It wires together the
sibling stage modules in strict sequential order so that only one model needs to be
resident on the GPU at a time, bounding peak VRAM usage:

    extract_frames -> detect_contact -> classify_foul -> assess_severity
        -> detect_location -> make_ruling

Usage:
    python pipeline.py clip.mp4 [--num-frames 16]

Programmatic usage:
    from pipeline import run_pipeline
    ruling = run_pipeline("clip.mp4", num_frames=16)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# The sibling stage modules live alongside this file. Insert this file's directory
# into sys.path BEFORE importing them so the pipeline works correctly regardless of
# the current working directory it is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_detector import detect_contact  # noqa: E402
from foul_classifier import classify_foul  # noqa: E402
from frame_extractor import extract_frames  # noqa: E402
from location_detector import detect_location  # noqa: E402
from ruling_engine import make_ruling  # noqa: E402
from severity_assessor import assess_severity  # noqa: E402

# ---------------------------------------------------------------------------
# SWAP HOOK
# ---------------------------------------------------------------------------
# Model swapping (e.g. changing checkpoints, quantization, or backends) happens
# INSIDE each sibling module via their own MODEL_ID (or equivalent) constants:
#   frame_extractor.py, contact_detector.py, foul_classifier.py,
#   severity_assessor.py, location_detector.py, ruling_engine.py.
# This orchestrator does not import, reference, or hardcode any model identifiers.
# It depends ONLY on the public function signatures listed in each module's
# docstring above. To swap a model, edit the constant in the relevant sibling
# module; no changes to this file are required.
# ---------------------------------------------------------------------------


class PipelineStageError(RuntimeError):
    """Raised when a pipeline stage fails, carrying the stage name for reporting."""

    def __init__(self, stage_name: str, original_exception: Exception) -> None:
        """Store the failing stage name and wrap the original exception.

        Args:
            stage_name: Human-readable name of the stage that failed.
            original_exception: The exception raised by the stage implementation.
        """
        self.stage_name = stage_name
        self.original_exception = original_exception
        super().__init__(f"Stage '{stage_name}' failed: {original_exception}")


def _log(message: str) -> None:
    """Write a progress/log message to stderr, keeping stdout clean for JSON output.

    Args:
        message: The message to print.
    """
    print(message, file=sys.stderr, flush=True)


def run_pipeline(video_path: str, num_frames: int = 16) -> dict[str, Any]:
    """Run the full foul detection pipeline on a single video clip.

    Executes each stage strictly in sequence (never in parallel) so that at most
    one model is loaded on the GPU at a time, bounding peak VRAM usage. Progress
    and timing for each stage are logged to stderr.

    Args:
        video_path: Path to the input video clip.
        num_frames: Number of frames to sample from the clip for analysis.

    Returns:
        The final ruling dictionary produced by ``ruling_engine.make_ruling``.

    Raises:
        PipelineStageError: If any stage raises an exception. The original
            exception is available via ``PipelineStageError.original_exception``.
    """
    pipeline_start = time.perf_counter()
    frame_count = 0

    # Stage 1: Frame extraction
    _log("[1/6] Extracting frames...")
    stage_start = time.perf_counter()
    try:
        frames, timestamps = extract_frames(video_path, num_frames=num_frames)
    except Exception as exc:  # noqa: BLE001 - intentionally broad to report stage
        raise PipelineStageError("extract_frames", exc) from exc
    frame_count = len(frames)
    _log(
        f"[1/6] Extracted {frame_count} frames "
        f"({time.perf_counter() - stage_start:.2f}s)"
    )
    del timestamps  # not consumed downstream in this orchestrator

    # Stage 2: Contact detection
    _log("[2/6] Detecting contact...")
    stage_start = time.perf_counter()
    try:
        contact = detect_contact(frames)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("detect_contact", exc) from exc
    _log(f"[2/6] Contact detection done ({time.perf_counter() - stage_start:.2f}s)")

    # Stage 3: Foul classification
    _log("[3/6] Classifying foul type...")
    stage_start = time.perf_counter()
    try:
        foul = classify_foul(frames)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("classify_foul", exc) from exc
    _log(f"[3/6] Foul classification done ({time.perf_counter() - stage_start:.2f}s)")

    # Stage 4: Severity assessment (depends on classified foul type)
    _log("[4/6] Assessing severity...")
    stage_start = time.perf_counter()
    try:
        severity = assess_severity(frames, foul.get("foul_type", ""))
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("assess_severity", exc) from exc
    _log(f"[4/6] Severity assessment done ({time.perf_counter() - stage_start:.2f}s)")

    # Stage 5: Location detection
    _log("[5/6] Detecting location...")
    stage_start = time.perf_counter()
    try:
        location = detect_location(frames)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("detect_location", exc) from exc
    _log(f"[5/6] Location detection done ({time.perf_counter() - stage_start:.2f}s)")

    # Stage 6: Final ruling
    _log("[6/6] Making ruling...")
    stage_start = time.perf_counter()
    try:
        ruling = make_ruling(contact, foul, severity, location)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("make_ruling", exc) from exc
    _log(f"[6/6] Ruling complete ({time.perf_counter() - stage_start:.2f}s)")

    total_elapsed = time.perf_counter() - pipeline_start
    _log(
        f"Summary: {frame_count} frames extracted, "
        f"total elapsed {total_elapsed:.2f}s"
    )

    return ruling


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the pipeline CLI.

    Returns:
        A configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description=(
            "Run the fully local, GPU-accelerated soccer foul detection pipeline "
            "on a single video clip and print the final ruling as JSON."
        ),
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video clip (e.g. clip.mp4).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to sample from the clip for analysis (default: 16).",
    )
    return parser


def main() -> None:
    """Parse CLI arguments, run the pipeline, and print the resulting ruling.

    Prints per-stage progress to stderr and the final ruling dict as pretty-printed
    JSON to stdout so the output can be piped into other tools. On missing input
    file or stage failure, prints a readable error to stderr and exits with
    status 1. When invoked with no arguments, prints usage to stderr and exits
    with status 2.
    """
    if len(sys.argv) == 1:
        _log("Usage: python pipeline.py <video_path> [--num-frames 16]")
        sys.exit(2)

    parser = _build_arg_parser()
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.is_file():
        _log(f"Error: video file not found: {video_path}")
        sys.exit(1)

    try:
        ruling = run_pipeline(str(video_path), num_frames=args.num_frames)
    except PipelineStageError as exc:
        _log(f"Error: pipeline failed at stage '{exc.stage_name}': {exc.original_exception}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - top-level safety net for CLI use
        _log(f"Error: unexpected failure: {exc}")
        sys.exit(1)

    print(json.dumps(ruling, indent=2))


if __name__ == "__main__":
    main()
