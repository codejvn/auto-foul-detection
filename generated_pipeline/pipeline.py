"""End-to-end orchestrator for the local, GPU-accelerated soccer foul detection pipeline.

This module is the single CLI entry point for the pipeline. It wires together the
sibling stage modules in strict sequential order so that only one model needs to be
resident on the GPU at a time, bounding peak VRAM usage:

    extract_frames -> filter_shot_boundaries -> detect_contact -> classify_foul
        -> assess_severity -> detect_location
        -> [evaluate_ambiguous_case, only if any module confidence < 0.65]
        -> make_ruling

The shot boundary filter removes frames that straddle a camera cut so every
downstream module sees a single continuous shot. The judgment layer (Gemini
API) is consulted only for ambiguous cases -- when any of the four analysis
modules reports confidence below the routing threshold (0.65); unambiguous
cases go straight to the deterministic ruling engine.

Usage:
    python pipeline.py clip.mp4 [--num-frames 16] [--skip-judgment-layer | --with-judgment-layer]

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
from judgment_layer import evaluate_ambiguous_case  # noqa: E402
from location_detector import detect_location  # noqa: E402
from ruling_engine import LOW_CONFIDENCE_THRESHOLD, make_ruling  # noqa: E402
from severity_assessor import assess_severity  # noqa: E402
from shot_boundary_filter import filter_shot_boundaries  # noqa: E402

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


def run_pipeline(
    video_path: str, num_frames: int = 16, use_judgment_layer: bool = True
) -> dict[str, Any]:
    """Run the full foul detection pipeline on a single video clip.

    Executes each stage strictly in sequence (never in parallel) so that at most
    one model is loaded on the GPU at a time, bounding peak VRAM usage. Progress
    and timing for each stage are logged to stderr.

    Args:
        video_path: Path to the input video clip.
        num_frames: Number of frames to sample from the clip for analysis.
        use_judgment_layer: When True (default), Stage 7 calls the Gemini
            judgment layer for ambiguous cases (any module confidence below
            ``LOW_CONFIDENCE_THRESHOLD``). When False, Stage 7 is bypassed
            entirely -- the four module outputs go straight to
            ``ruling_engine`` with no escalation, and ``low_confidence_modules``
            is still computed/flagged (it is needed by the ruling and by
            calibration scoring) even though no Gemini call is made.

    Returns:
        The final ruling dictionary produced by ``ruling_engine.make_ruling``.

    Raises:
        PipelineStageError: If any stage raises an exception. The original
            exception is available via ``PipelineStageError.original_exception``.
    """
    pipeline_start = time.perf_counter()
    frame_count = 0

    # Stage 1: Frame extraction
    _log("[1/8] Extracting frames...")
    stage_start = time.perf_counter()
    try:
        frames, timestamps = extract_frames(video_path, num_frames=num_frames)
    except Exception as exc:  # noqa: BLE001 - intentionally broad to report stage
        raise PipelineStageError("extract_frames", exc) from exc
    extracted_count = len(frames)
    _log(
        f"[1/8] Extracted {extracted_count} frames "
        f"({time.perf_counter() - stage_start:.2f}s)"
    )

    # Stage 2: Shot boundary filtering (keep only the longest continuous shot)
    _log("[2/8] Filtering shot boundaries...")
    stage_start = time.perf_counter()
    try:
        frames, timestamps = filter_shot_boundaries(frames, timestamps)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("filter_shot_boundaries", exc) from exc
    frame_count = len(frames)
    _log(
        f"[2/8] Kept {frame_count} of {extracted_count} frames from the "
        f"longest continuous shot ({time.perf_counter() - stage_start:.2f}s)"
    )
    del timestamps  # not consumed downstream in this orchestrator

    # Stage 3: Contact detection
    _log("[3/8] Detecting contact...")
    stage_start = time.perf_counter()
    try:
        contact = detect_contact(frames)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("detect_contact", exc) from exc
    _log(f"[3/8] Contact detection done ({time.perf_counter() - stage_start:.2f}s)")

    # Stage 4: Foul classification
    _log("[4/8] Classifying foul type...")
    stage_start = time.perf_counter()
    try:
        foul = classify_foul(frames)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("classify_foul", exc) from exc
    _log(f"[4/8] Foul classification done ({time.perf_counter() - stage_start:.2f}s)")

    # Stage 5: Severity assessment (depends on classified foul type)
    _log("[5/8] Assessing severity...")
    stage_start = time.perf_counter()
    try:
        severity = assess_severity(frames, foul.get("foul_type", ""))
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("assess_severity", exc) from exc
    _log(f"[5/8] Severity assessment done ({time.perf_counter() - stage_start:.2f}s)")

    # Stage 6: Location detection
    _log("[6/8] Detecting location...")
    stage_start = time.perf_counter()
    try:
        location = detect_location(frames)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("detect_location", exc) from exc
    _log(f"[6/8] Location detection done ({time.perf_counter() - stage_start:.2f}s)")

    # Stage 7: Confidence routing -> judgment layer for ambiguous cases only
    named_outputs = [
        ("contact_detector", contact),
        ("foul_classifier", foul),
        ("severity_assessor", severity),
        ("location_detector", location),
    ]
    low_confidence_modules = [
        name
        for name, output in named_outputs
        if float(output["confidence"]) < LOW_CONFIDENCE_THRESHOLD
    ]

    judgment: dict[str, Any] | None = None
    if use_judgment_layer and low_confidence_modules:
        _log(
            "[7/8] Judgment layer INVOKED: low-confidence modules "
            f"(< {LOW_CONFIDENCE_THRESHOLD}): {', '.join(low_confidence_modules)}"
        )
        stage_start = time.perf_counter()
        try:
            judgment = evaluate_ambiguous_case(
                {
                    "contact": contact,
                    "foul": foul,
                    "severity": severity,
                    "location": location,
                    "low_confidence_modules": low_confidence_modules,
                }
            )
            _log(
                f"[7/8] Judgment layer done ({time.perf_counter() - stage_start:.2f}s)"
            )
        except Exception as exc:  # noqa: BLE001 - degrade, don't kill the run
            judgment = None
            _log(
                f"[7/8] Warning: judgment layer failed ({exc}); proceeding "
                "with module outputs only. The ruling will still flag the "
                "low-confidence modules."
            )
    elif low_confidence_modules:
        _log(
            "[7/8] Judgment layer BYPASSED (--skip-judgment-layer): "
            f"{len(low_confidence_modules)} low-confidence module(s) "
            f"({', '.join(low_confidence_modules)}) routed straight to ruling_engine "
            "with no escalation."
        )
    else:
        _log(
            "[7/8] Judgment layer SKIPPED: all module confidences >= "
            f"{LOW_CONFIDENCE_THRESHOLD}"
        )

    # Stage 8: Final ruling
    _log("[8/8] Making ruling...")
    stage_start = time.perf_counter()
    try:
        ruling = make_ruling(
            contact,
            foul,
            severity,
            location,
            judgment=judgment,
            low_confidence_modules=low_confidence_modules,
        )
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("make_ruling", exc) from exc
    _log(f"[8/8] Ruling complete ({time.perf_counter() - stage_start:.2f}s)")

    total_elapsed = time.perf_counter() - pipeline_start
    _log(
        f"Summary: {extracted_count} frames extracted, {frame_count} kept "
        f"after shot filtering, total elapsed {total_elapsed:.2f}s"
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
    jl_group = parser.add_mutually_exclusive_group()
    jl_group.add_argument(
        "--skip-judgment-layer", action="store_true",
        help="Bypass the Gemini judgment layer entirely; route module outputs straight to "
             "ruling_engine even when confidence is low (no API call, no escalation).",
    )
    jl_group.add_argument(
        "--with-judgment-layer", action="store_true",
        help="Force the judgment layer on for ambiguous cases (the default). Explicit opt-in for "
             "spot-checking individual clips with full judgment-layer reasoning.",
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
        _log(
            "Usage: python pipeline.py <video_path> [--num-frames 16] "
            "[--skip-judgment-layer | --with-judgment-layer]"
        )
        sys.exit(2)

    parser = _build_arg_parser()
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.is_file():
        _log(f"Error: video file not found: {video_path}")
        sys.exit(1)

    use_judgment_layer = not args.skip_judgment_layer

    try:
        ruling = run_pipeline(
            str(video_path),
            num_frames=args.num_frames,
            use_judgment_layer=use_judgment_layer,
        )
    except PipelineStageError as exc:
        _log(f"Error: pipeline failed at stage '{exc.stage_name}': {exc.original_exception}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - top-level safety net for CLI use
        _log(f"Error: unexpected failure: {exc}")
        sys.exit(1)

    print(json.dumps(ruling, indent=2))


if __name__ == "__main__":
    main()
