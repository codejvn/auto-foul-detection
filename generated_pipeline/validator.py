"""
validator.py
============

Accuracy + calibration validation of the auto-foul-detection pipeline
against the VARS / SoccerNet MVFoul dataset (https://github.com/SoccerNet/sn-mvfoul).

This script backs the project's three research claims:
    1. Perception accuracy  -- foul_type accuracy vs. human VAR labels.
    2. Ruling accuracy      -- foul detected / severity accuracy vs. labels.
    3. Calibration (novel)  -- clips where OUR confidence < 0.65 should be
       the clips where HUMAN annotators disagreed with each other.

Expected dataset layout (--dataset-path)
----------------------------------------
    <dataset_path>/
        <clip_id>/
            *.mp4 | *.avi | *.mkv        (exactly one video per clip dir)
            annotations.json

``annotations.json`` may be any of:
    - a single label object:      {"action_class": ..., "severity_class": ..., "offence": bool}
    - a bare list of label objects (one per annotator)
    - {"annotations": [ ...label objects... ]}

Clips with multiple annotator labels that are not unanimous are the
"human disagreement" cases used for the calibration claim. Ground truth
for scoring is the per-field majority vote across annotators.

Label mapping
-------------
VARS-style labels are normalized onto this pipeline's vocabularies via
``ACTION_CLASS_KEYWORDS`` and ``SEVERITY_CLASS_MAP`` below (e.g.
"Standing tackling" -> "tackle", severity "3.0" / "Yellow card" ->
"reckless"). Edit those tables to adapt to other annotation dialects --
they are the swap hook of this module.

Calibration score
-----------------
Let L = clips with overall ruling confidence < 0.65, D = clips with human
disagreement, O = |L ∩ D|. The calibration score is the F1 of "low
confidence" as a predictor of "human disagreement":
    precision = O / |L|, recall = O / |D|, score = harmonic mean.
A score near 1 means our uncertainty flags exactly the clips humans found
hard; near 0 means our confidence is uninformative about human difficulty.

Usage
-----
    python validator.py --dataset-path ./mvfoul_dataset --output results.json
    python validator.py --smoke-test

The pipeline (and its GPU models) is imported lazily, so --smoke-test runs
with no dataset, no models, and no API key.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

# Sibling modules live alongside this file; make imports work from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ruling_engine import LOW_CONFIDENCE_THRESHOLD  # noqa: E402

# ---------------------------------------------------------------------------
# Label normalization tables (swap hook -- edit to match annotation dialects)
# ---------------------------------------------------------------------------

#: Substring -> pipeline foul type. Checked in order against the lowercased
#: action_class; first match wins. Covers VARS action classes and our own.
ACTION_CLASS_KEYWORDS: list[tuple[str, str]] = [
    ("tackl", "tackle"),          # "Tackling", "Standing tackling"
    ("high leg", "tackle"),
    ("push", "push"),
    ("elbow", "push"),            # "Elbowing" -- upper-body force
    ("hold", "obstruction"),      # "Holding"
    ("challenge", "obstruction"),
    ("obstruct", "obstruction"),
    ("block", "obstruction"),
    ("dive", "simulation"),
    ("simulat", "simulation"),
    ("hand", "handball"),
    ("none", "none"),
    ("no offence", "none"),
]

#: Severity label -> pipeline severity. Keys are lowercased/stripped. Covers
#: VARS numeric severities (1.0-5.0), card names, and our own labels.
SEVERITY_CLASS_MAP: dict[str, str] = {
    "1": "careless", "1.0": "careless",
    "2": "careless", "2.0": "careless",
    "3": "reckless", "3.0": "reckless",
    "4": "reckless", "4.0": "reckless",
    "5": "excessive_force", "5.0": "excessive_force",
    "no card": "careless",
    "offence + no card": "careless",
    "yellow card": "reckless",
    "offence + yellow card": "reckless",
    "red card": "excessive_force",
    "offence + red card": "excessive_force",
    "careless": "careless",
    "reckless": "reckless",
    "excessive_force": "excessive_force",
    "excessive force": "excessive_force",
}

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv")


# ---------------------------------------------------------------------------
# Label normalization / consensus
# ---------------------------------------------------------------------------


def normalize_foul_type(action_class: Any) -> str:
    """Map a raw action_class label onto this pipeline's foul-type vocabulary.

    Args:
        action_class: Raw annotation value (any VARS-style action class or a
            native pipeline foul type).

    Returns:
        One of the pipeline's foul types; unrecognized labels map to "none".
    """
    text = str(action_class).strip().lower()
    for keyword, foul_type in ACTION_CLASS_KEYWORDS:
        if keyword in text:
            return foul_type
    return "none"


def normalize_severity(severity_class: Any) -> Optional[str]:
    """Map a raw severity_class label onto the pipeline's severity vocabulary.

    Args:
        severity_class: Raw annotation value (VARS numeric severity, card
            name, or a native pipeline severity).

    Returns:
        One of "careless" / "reckless" / "excessive_force", or None when the
        label is missing/unrecognized (severity is then skipped for scoring).
    """
    if severity_class is None:
        return None
    return SEVERITY_CLASS_MAP.get(str(severity_class).strip().lower())


def build_ground_truth(labels: list[dict]) -> dict:
    """Reduce one clip's annotator labels to consensus ground truth.

    Consensus is the per-field majority vote (ties resolved by first-seen
    order, as ``collections.Counter`` does). The clip counts as a human
    disagreement case when the normalized labels are not unanimous across
    annotators on any of offence / foul type / severity.

    Args:
        labels: One or more annotation dicts, each with keys
            ``action_class``, ``severity_class``, and ``offence``.

    Returns:
        Dict with keys:
            'offence' (bool), 'foul_type' (str), 'severity' (str | None),
            'human_disagreement' (bool), 'num_annotators' (int).

    Raises:
        ValueError: If ``labels`` is empty.
    """
    if not labels:
        raise ValueError("Cannot build ground truth from zero annotator labels")

    offences = [bool(label.get("offence", False)) for label in labels]
    foul_types = [normalize_foul_type(label.get("action_class")) for label in labels]
    severities = [normalize_severity(label.get("severity_class")) for label in labels]

    disagreement = (
        len(set(offences)) > 1
        or len(set(foul_types)) > 1
        or len(set(severities)) > 1
    )

    known_severities = [s for s in severities if s is not None]
    consensus_severity = (
        Counter(known_severities).most_common(1)[0][0] if known_severities else None
    )

    return {
        "offence": Counter(offences).most_common(1)[0][0],
        "foul_type": Counter(foul_types).most_common(1)[0][0],
        "severity": consensus_severity,
        "human_disagreement": disagreement,
        "num_annotators": len(labels),
    }


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def discover_clips(dataset_path: Path) -> list[dict]:
    """Find (video, labels) pairs under the dataset root.

    Every directory containing an ``annotations.json`` and exactly one video
    file is treated as a clip. Directories missing either piece are skipped
    with a warning on stderr.

    Args:
        dataset_path: Root directory of the (extracted) MVFoul dataset.

    Returns:
        List of dicts with keys 'clip_id', 'video_path' (str), and
        'labels' (list of annotation dicts), sorted by clip_id.

    Raises:
        FileNotFoundError: If ``dataset_path`` does not exist.
    """
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    clips: list[dict] = []
    for annotation_file in sorted(dataset_path.rglob("annotations.json")):
        clip_dir = annotation_file.parent
        clip_id = str(clip_dir.relative_to(dataset_path))

        videos = [
            p for p in sorted(clip_dir.iterdir())
            if p.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if len(videos) != 1:
            print(
                f"[validator] Skipping '{clip_id}': expected exactly 1 video, "
                f"found {len(videos)}.",
                file=sys.stderr,
            )
            continue

        try:
            raw = json.loads(annotation_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[validator] Skipping '{clip_id}': unreadable annotations "
                f"({exc}).",
                file=sys.stderr,
            )
            continue

        if isinstance(raw, dict) and "annotations" in raw:
            labels = raw["annotations"]
        elif isinstance(raw, dict):
            labels = [raw]
        elif isinstance(raw, list):
            labels = raw
        else:
            print(
                f"[validator] Skipping '{clip_id}': unrecognized annotation "
                f"shape ({type(raw).__name__}).",
                file=sys.stderr,
            )
            continue

        if not labels:
            print(
                f"[validator] Skipping '{clip_id}': empty annotation list.",
                file=sys.stderr,
            )
            continue

        clips.append(
            {"clip_id": clip_id, "video_path": str(videos[0]), "labels": labels}
        )

    return clips


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(evaluated: list[dict], failed: int = 0) -> dict:
    """Aggregate per-clip results into the results JSON structure.

    Args:
        evaluated: Per-clip dicts, each with keys 'clip_id',
            'ground_truth' (from ``build_ground_truth``), and 'ruling'
            (the pipeline's ruling dict).
        failed: Number of clips where the pipeline raised and was skipped.

    Returns:
        Dict with keys 'overall_accuracy', 'calibration',
        'per_foul_type_accuracy', and 'per_clip'.
    """
    detected_correct = 0
    foul_type_correct = 0
    severity_total = 0
    severity_correct = 0
    per_type_totals: Counter = Counter()
    per_type_correct: Counter = Counter()
    low_confidence_ids: set[str] = set()
    disagreement_ids: set[str] = set()
    per_clip: list[dict] = []

    for record in evaluated:
        gt = record["ground_truth"]
        ruling = record["ruling"]
        clip_id = record["clip_id"]

        if bool(ruling["foul_detected"]) == bool(gt["offence"]):
            detected_correct += 1

        gt_foul_type = gt["foul_type"] if gt["offence"] else "none"
        foul_type_match = ruling["foul_type"] == gt_foul_type
        if foul_type_match:
            foul_type_correct += 1
        per_type_totals[gt_foul_type] += 1
        if foul_type_match:
            per_type_correct[gt_foul_type] += 1

        # Severity is only meaningful on true offences with a known label.
        severity_match = None
        if gt["offence"] and gt["severity"] is not None:
            severity_total += 1
            severity_match = ruling["severity"] == gt["severity"]
            if severity_match:
                severity_correct += 1

        is_low_confidence = ruling["confidence"] < LOW_CONFIDENCE_THRESHOLD
        if is_low_confidence:
            low_confidence_ids.add(clip_id)
        if gt["human_disagreement"]:
            disagreement_ids.add(clip_id)

        per_clip.append(
            {
                "clip_id": clip_id,
                "predicted_foul_type": ruling["foul_type"],
                "predicted_severity": ruling["severity"],
                "foul_detected": ruling["foul_detected"],
                "confidence": ruling["confidence"],
                "low_confidence": is_low_confidence,
                "judgment_layer_used": ruling.get("judgment_layer_used", False),
                "gt_foul_type": gt_foul_type,
                "gt_severity": gt["severity"],
                "gt_offence": gt["offence"],
                "human_disagreement": gt["human_disagreement"],
                "num_annotators": gt["num_annotators"],
            }
        )

    total = len(evaluated)
    overlap = len(low_confidence_ids & disagreement_ids)
    precision = overlap / len(low_confidence_ids) if low_confidence_ids else 0.0
    recall = overlap / len(disagreement_ids) if disagreement_ids else 0.0
    calibration_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "overall_accuracy": {
            "clips_evaluated": total,
            "clips_failed": failed,
            "foul_detected": detected_correct / total if total else 0.0,
            "foul_type": foul_type_correct / total if total else 0.0,
            "severity": severity_correct / severity_total if severity_total else 0.0,
            "severity_clips_scored": severity_total,
        },
        "calibration": {
            "low_confidence_clips": len(low_confidence_ids),
            "human_disagreement_clips": len(disagreement_ids),
            "overlap": overlap,
            "calibration_score": calibration_score,
        },
        "per_foul_type_accuracy": {
            foul_type: per_type_correct[foul_type] / count
            for foul_type, count in sorted(per_type_totals.items())
        },
        "per_clip": per_clip,
    }


def print_summary(results: dict) -> None:
    """Print a human-readable summary of the results to stdout.

    Args:
        results: The dict produced by ``compute_metrics``.
    """
    overall = results["overall_accuracy"]
    calibration = results["calibration"]

    print("=" * 62)
    print("VARS / MVFoul validation summary")
    print("=" * 62)
    print(f"Clips evaluated:        {overall['clips_evaluated']}"
          + (f"  (+{overall['clips_failed']} failed)" if overall["clips_failed"] else ""))
    print(f"Foul detected accuracy: {overall['foul_detected']:.1%}")
    print(f"Foul type accuracy:     {overall['foul_type']:.1%}")
    print(f"Severity accuracy:      {overall['severity']:.1%} "
          f"(on {overall['severity_clips_scored']} offence clips)")
    print("-" * 62)
    print("Per-foul-type accuracy:")
    for foul_type, accuracy in results["per_foul_type_accuracy"].items():
        print(f"    {foul_type:<16} {accuracy:.1%}")
    print("-" * 62)
    print("Calibration (does low confidence predict human disagreement?)")
    print(f"    low-confidence clips (< {LOW_CONFIDENCE_THRESHOLD}): "
          f"{calibration['low_confidence_clips']}")
    print(f"    human-disagreement clips:      "
          f"{calibration['human_disagreement_clips']}")
    print(f"    overlap:                       {calibration['overlap']}")
    print(f"    calibration score (F1):        "
          f"{calibration['calibration_score']:.3f}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Evaluation drivers
# ---------------------------------------------------------------------------


def evaluate_dataset(dataset_path: Path) -> dict:
    """Run the full pipeline over every clip in the dataset and score it.

    The pipeline module (and its GPU models) is imported here, lazily, so
    that --smoke-test never touches the heavy dependencies.

    Args:
        dataset_path: Root directory of the dataset.

    Returns:
        The results dict from ``compute_metrics``.

    Raises:
        FileNotFoundError: If the dataset path does not exist.
        ValueError: If no usable clips are found under it.
    """
    clips = discover_clips(dataset_path)
    if not clips:
        raise ValueError(
            f"No usable clips found under '{dataset_path}'. Expected "
            "<clip_dir>/annotations.json plus one video file per clip."
        )

    # Lazy import AFTER dataset discovery, so path/annotation problems
    # surface immediately instead of after minutes of model loading.
    from pipeline import run_pipeline

    print(f"[validator] Evaluating {len(clips)} clips...", file=sys.stderr)
    evaluated: list[dict] = []
    failed = 0
    start = time.perf_counter()

    for index, clip in enumerate(clips, start=1):
        print(
            f"[validator] ({index}/{len(clips)}) {clip['clip_id']}",
            file=sys.stderr,
        )
        try:
            ruling = run_pipeline(clip["video_path"])
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the run
            failed += 1
            print(
                f"[validator]   FAILED: {exc}",
                file=sys.stderr,
            )
            continue

        evaluated.append(
            {
                "clip_id": clip["clip_id"],
                "ground_truth": build_ground_truth(clip["labels"]),
                "ruling": ruling,
            }
        )

    elapsed = time.perf_counter() - start
    print(
        f"[validator] Done: {len(evaluated)} evaluated, {failed} failed "
        f"({elapsed:.1f}s).",
        file=sys.stderr,
    )
    return compute_metrics(evaluated, failed=failed)


# ---------------------------------------------------------------------------
# Smoke test (no dataset, no models, no API key)
# ---------------------------------------------------------------------------


def _fake_ruling(foul_detected: bool, foul_type: str, severity: str,
                 confidence: float) -> dict:
    """Build a minimal fake ruling dict shaped like make_ruling's output."""
    return {
        "foul_detected": foul_detected,
        "foul_type": foul_type,
        "severity": severity,
        "in_penalty_box": False,
        "punishment": "free kick" if foul_detected else "play on",
        "confidence": confidence,
        "human_review_recommended": confidence < 0.60,
        "judgment_layer_used": confidence < LOW_CONFIDENCE_THRESHOLD,
        "judgment_layer_reasoning": None,
        "low_confidence_modules": [],
    }


def run_smoke_test() -> dict:
    """Score 5 hardcoded fake examples to verify the script end to end.

    The examples cover: a confident correct prediction, a human-disagreement
    clip our low confidence correctly flags, a disagreement clip we were
    (over)confident on, a confident correct no-foul, and a low-confidence
    miss without disagreement. Expected metric values are asserted so the
    scoring logic itself is verified, not just executed.

    Returns:
        The results dict from ``compute_metrics``.
    """
    single = lambda action, sev, off: [  # noqa: E731 - tiny local helper
        {"action_class": action, "severity_class": sev, "offence": off}
    ]

    examples = [
        # 1. Confident, fully correct tackle (VARS-style labels).
        {
            "clip_id": "smoke/confident_correct_tackle",
            "labels": single("Tackling", "3.0", True),
            "ruling": _fake_ruling(True, "tackle", "reckless", 0.86),
        },
        # 2. Annotators disagree on severity; our confidence is low. This is
        #    the calibration claim working: overlap of L and D.
        {
            "clip_id": "smoke/disagreement_flagged",
            "labels": [
                {"action_class": "Tackling", "severity_class": "3.0", "offence": True},
                {"action_class": "Tackling", "severity_class": "3.0", "offence": True},
                {"action_class": "Tackling", "severity_class": "5.0", "offence": True},
            ],
            "ruling": _fake_ruling(True, "tackle", "reckless", 0.52),
        },
        # 3. Annotators disagree (offence vs. dive) but we were confident:
        #    a calibration recall miss.
        {
            "clip_id": "smoke/disagreement_missed",
            "labels": [
                {"action_class": "Pushing", "severity_class": "1.0", "offence": True},
                {"action_class": "Pushing", "severity_class": "1.0", "offence": True},
                {"action_class": "Dive", "severity_class": None, "offence": False},
            ],
            "ruling": _fake_ruling(True, "push", "careless", 0.83),
        },
        # 4. Confident correct no-foul.
        {
            "clip_id": "smoke/confident_no_foul",
            "labels": single("None", None, False),
            "ruling": _fake_ruling(False, "none", "careless", 0.90),
        },
        # 5. Low confidence without human disagreement, and both foul type
        #    and severity are wrong: a calibration precision miss + a
        #    per-type accuracy miss + a severity miss.
        {
            "clip_id": "smoke/low_conf_no_disagreement",
            "labels": single("Holding", "1.0", True),
            "ruling": _fake_ruling(True, "push", "reckless", 0.58),
        },
    ]

    evaluated = [
        {
            "clip_id": example["clip_id"],
            "ground_truth": build_ground_truth(example["labels"]),
            "ruling": example["ruling"],
        }
        for example in examples
    ]
    results = compute_metrics(evaluated)

    overall = results["overall_accuracy"]
    calibration = results["calibration"]
    assert overall["clips_evaluated"] == 5
    assert overall["foul_detected"] == 1.0, "all 5 detect decisions are correct"
    assert overall["foul_type"] == 0.8, "4 of 5 foul types are correct"
    assert overall["severity_clips_scored"] == 4
    assert overall["severity"] == 0.75, "3 of 4 scored severities are correct"
    assert calibration["low_confidence_clips"] == 2, "examples 2 and 5"
    assert calibration["human_disagreement_clips"] == 2, "examples 2 and 3"
    assert calibration["overlap"] == 1, "only example 2 overlaps"
    assert abs(calibration["calibration_score"] - 0.5) < 1e-9, "F1 of P=R=0.5"
    assert results["per_foul_type_accuracy"]["obstruction"] == 0.0
    assert results["per_foul_type_accuracy"]["tackle"] == 1.0

    # Ground-truth reduction sanity checks.
    gt_disagree = build_ground_truth(examples[1]["labels"])
    assert gt_disagree["human_disagreement"] is True
    assert gt_disagree["severity"] == "reckless", "majority vote: 2x 3.0 vs 1x 5.0"
    gt_single = build_ground_truth(examples[0]["labels"])
    assert gt_single["human_disagreement"] is False

    print("[smoke test] All metric assertions passed on 5 fake examples.\n",
          file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the validator CLI."""
    parser = argparse.ArgumentParser(
        prog="validator.py",
        description=(
            "Validate the foul-detection pipeline against the VARS/MVFoul "
            "dataset: accuracy vs. human VAR labels, plus calibration of our "
            "confidence against human annotator disagreement."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Root directory of the extracted MVFoul dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the results JSON (optional).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on 5 hardcoded fake examples; needs no dataset or models.",
    )
    return parser


def main() -> None:
    """Parse arguments, run validation (or the smoke test), report results."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.smoke_test:
        results = run_smoke_test()
    elif args.dataset_path:
        try:
            results = evaluate_dataset(Path(args.dataset_path))
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.error("either --dataset-path or --smoke-test is required")
        return  # unreachable; parser.error exits

    print_summary(results)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(f"[validator] Results written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
