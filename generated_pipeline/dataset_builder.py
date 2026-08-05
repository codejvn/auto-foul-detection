"""
dataset_builder.py
==================

Parses a SoccerNet-MVFoul ``annotations.json`` into training-ready labels.

The filter/coercion logic replicates ``reference_repo/VARS model/data_loader.py``
(``label2vectormerge``) exactly, with one deliberate addition: rows VARS drops
because their severity is empty or *borderline* ('2.0' = borderline No/Yellow,
'4.0' = borderline Yellow/Red) are not thrown away here. They are captured in a
separate ``hard_cases`` list with their raw annotation fields preserved.

Those borderline rows are precisely the clips where human annotators hedged, so
they are the evaluation set for this project's calibration claim (see CLAUDE.md,
"Key research claims" #3): our confidence scores should be low exactly where the
humans could not commit to a label.

Usage
-----
    python dataset_builder.py
    python dataset_builder.py --annotations ../mvfoul_extracted/valid/annotations.json

Public API
----------
    build_dataset(annotations_path) -> (train_examples, hard_case_examples)

Each training example is::

    {
        "action_id": "0",
        "video_path": "/abs/path/to/train/action_0/clip_0.mp4",
        "action_class_label": 6,        # index into ACTION_CLASSES
        "offence_severity_class": 1,    # index into OFFENCE_SEVERITY_CLASSES
        "handball_label": 0,            # 1 = Handball, 0 = No handball
    }

``action_class_label`` is always the real annotated action class -- it is never
overridden to 'none' based on the card outcome. A no-offence standing tackle is
still labelled as a standing tackle; index 8 ('none') is reserved and never
produced by any real row (see ``NONE_ACTION_INDEX``).

Hard-case examples carry the same keys plus ``raw`` (the untouched annotation
dict) and ``reason``. Their ``offence_severity_class`` is ``None`` -- the whole
point is that the severity could not be resolved to a card decision.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label vocabularies
# ---------------------------------------------------------------------------

#: 8-class MVFoul action vocabulary, index 0-7, plus a reserved index 8 ('none')
#: that is never assigned to any real row (see NONE_ACTION_INDEX below).
#:
#: Ordering is copied verbatim from ``reference_repo/VARS model/config/classes.py``,
#: ``EVENT_DICTIONARY_action_class`` (VARS reserves its own index 8 for
#: "Dont know"; those rows are filtered out before labelling here, so we reuse
#: slot 8 for 'none' instead). This ordering is deliberately identical to VARS
#: so our labels are directly comparable against a VARS checkpoint.
ACTION_CLASSES: list[str] = [
    "Tackling",           # 0
    "Standing tackling",  # 1
    "High leg",           # 2
    "Holding",            # 3
    "Pushing",            # 4
    "Elbowing",           # 5
    "Challenge",          # 6
    "Dive",               # 7
    "none",               # 8 -- reserved, never produced by any real row
]

ACTION_CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(ACTION_CLASSES[:8])}

NONE_ACTION_INDEX: int = 8

#: Combined offence + card decision, index 0-3.
OFFENCE_SEVERITY_CLASSES: list[str] = [
    "No offence",           # 0
    "Offence + No card",    # 1
    "Offence + Yellow card",  # 2
    "Offence + Red card",   # 3
]

#: Offence spellings that both appear in the annotations for "not a foul".
NO_OFFENCE_VALUES: frozenset[str] = frozenset({"No offence", "No Offence"})

#: Severity values VARS treats as unusable: empty, or the two borderline codes.
BORDERLINE_SEVERITY_VALUES: frozenset[str] = frozenset({"", "2.0", "4.0"})

#: Camera view preferred when an action has several clips.
PREFERRED_CAMERA_TYPE: str = "Main camera center"


# ---------------------------------------------------------------------------
# Class-imbalance handling
# ---------------------------------------------------------------------------


def compute_class_weights(counts: Sequence[int], num_classes: int) -> list[float]:
    """Standard inverse-frequency class weights for ``CrossEntropyLoss(weight=...)``.

    ``weight_i = total / (num_classes * count_i)`` where ``total`` is the sum of
    ``counts``.

    ``num_classes`` should be the number of classes that actually appear (i.e.
    it should NOT count reserved, zero-count classes) -- with that convention,
    the COUNT-WEIGHTED mean of the weights is 1.0: ``sum(weight_i * count_i for
    all i) / total == 1.0``. (The plain, unweighted arithmetic mean of the
    weight values is NOT 1.0 -- rare classes get large weights that pull it up.)
    The count-weighted identity is what matters: it means
    ``CrossEntropyLoss(weight=...)`` on the real, imbalanced batch distribution
    keeps the same overall loss scale as unweighted cross-entropy would.

    A ``count_i`` of 0 (e.g. the reserved 'none' action index, which is never
    produced by any real row -- see ``NONE_ACTION_INDEX``) would divide by zero
    under the naive formula. Since ``torch.nn.CrossEntropyLoss(weight=...)`` is
    poisoned by an ``inf`` entry, zero-count classes get an explicit weight of
    ``0.0`` instead -- they never appear in training, so their weight is moot
    aside from not corrupting the vector.
    """
    total = sum(counts)
    weights: list[float] = []
    for count in counts:
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))
    return weights


#: Class weights below are derived from the TRAIN split
#: (``mvfoul_extracted/train/annotations.json``) only. They are baked as literal
#: lists so importing this module does no file I/O. To regenerate: run
#: ``python dataset_builder.py`` (the default --annotations points at the train
#: split) and copy the "recomputed" values it prints, rounded to 4 decimal
#: places. The smoke test recomputes these from whatever split it is pointed at
#: and warns on stderr if they drift from the constants below by more than
#: 1e-3 -- drift is EXPECTED (and not a bug) when running against valid/test,
#: since those constants are train-derived by definition.
CLASS_WEIGHTS_ACTION: list[float] = [
    0.8120, 0.2793, 3.2208, 1.0618, 3.9709, 2.3377, 0.8627, 10.3527, 0.0,
]
CLASS_WEIGHTS_OFFENCE_SEVERITY: list[float] = [1.9071, 0.4449, 0.8464, 21.4722]


# ---------------------------------------------------------------------------
# Clip selection
# ---------------------------------------------------------------------------


def select_clip(clips: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Pick which camera view to train on.

    Prefers the ``Main camera center`` view, else falls back to ``clip_0``, else
    the first clip listed. Returns ``None`` for an action with no clips at all.
    """
    if not clips:
        return None

    for clip in clips:
        if clip.get("Camera type") == PREFERRED_CAMERA_TYPE:
            return clip

    for clip in clips:
        url = clip.get("Url", "")
        if url.replace("\\", "/").rstrip("/").endswith("clip_0"):
            return clip

    return clips[0]


def clip_to_video_path(clip: dict[str, Any], action_id: str, split_dir: Path) -> Path:
    """Resolve a clip's annotation ``Url`` to a local ``.mp4`` path.

    The ``Url`` field is dataset-relative and Windows-flavoured (e.g.
    ``Dataset/Train/action_0/clip_0``), so only its last component is trusted;
    the rest of the path comes from where ``annotations.json`` actually lives.
    """
    url = clip.get("Url", "").replace("\\", "/").rstrip("/")
    clip_name = url.rsplit("/", 1)[-1] if url else "clip_0"
    return split_dir / f"action_{action_id}" / f"{clip_name}.mp4"


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------


def offence_severity_to_index(offence: str, severity: str) -> Optional[int]:
    """Map a (coerced) Offence/Severity pair to a 0-3 class index.

    Returns ``None`` for any combination outside the four valid ones -- callers
    must skip those rows rather than guess.
    """
    if offence in NO_OFFENCE_VALUES:
        return 0
    if offence == "Offence":
        if severity == "1.0":
            return 1
        if severity == "3.0":
            return 2
        if severity == "5.0":
            return 3
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_dataset(
    annotations_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse ``annotations.json`` into (train_examples, hard_case_examples).

    Filter order follows VARS' ``label2vectormerge`` exactly:

    1. drop Action class in ('', 'Dont know')
    2. drop Offence in ('', 'Between') unless Action class == 'Dive'
    3. otherwise coerce that Offence to 'Offence'
    4. borderline/empty Severity on a non-Dive offence -> hard_cases (VARS drops)
    5. otherwise coerce that Severity to '1.0'
    6. map (Offence, Severity) -> 0-3; anything unmapped is skipped with a warning
    """
    annotations_path = Path(annotations_path).resolve()
    split_dir = annotations_path.parent

    with open(annotations_path, encoding="utf-8") as f:
        data = json.load(f)

    actions: dict[str, dict[str, Any]] = data["Actions"]

    train_examples: list[dict[str, Any]] = []
    hard_case_examples: list[dict[str, Any]] = []

    for action_id, action in actions.items():
        action_class = action.get("Action class", "")
        offence = action.get("Offence", "")
        severity = action.get("Severity", "")

        # (1) unusable action label
        if action_class in ("", "Dont know"):
            continue

        # (2) unusable offence label -- except for dives, where 'Between' still
        # means the dive happened
        if offence in ("", "Between") and action_class != "Dive":
            continue

        # (3) coerce the surviving ambiguous offences
        if offence in ("", "Between"):
            offence = "Offence"

        clip = select_clip(action.get("Clips", []))
        if clip is None:
            logger.warning("Action %s has no clips; skipping.", action_id)
            continue
        video_path = clip_to_video_path(clip, action_id, split_dir)

        # (4) borderline severity: VARS discards these, we keep them aside.
        # Both 2.0 (No card / Yellow) and 4.0 (Yellow / Red) are cases where the
        # annotators would not commit -- our calibration target.
        if (
            severity in BORDERLINE_SEVERITY_VALUES
            and action_class != "Dive"
            and offence not in NO_OFFENCE_VALUES
        ):
            hard_case_examples.append(
                {
                    "action_id": action_id,
                    "video_path": str(video_path),
                    "action_class_label": ACTION_CLASS_TO_INDEX.get(action_class),
                    "offence_severity_class": None,
                    "handball_label": 1 if action.get("Handball") == "Handball" else 0,
                    "reason": (
                        "empty severity" if severity == ""
                        else f"borderline severity {severity}"
                    ),
                    "raw": action,
                }
            )
            continue

        # (5) remaining borderline severities are dives or no-offence rows, where
        # severity carries no card meaning
        if severity in BORDERLINE_SEVERITY_VALUES:
            severity = "1.0"

        # (6) combined offence+card class
        offence_severity_class = offence_severity_to_index(offence, severity)
        if offence_severity_class is None:
            logger.warning(
                "Action %s has unmappable Offence=%r / Severity=%r; skipping.",
                action_id,
                offence,
                severity,
            )
            continue

        # (7) action class label -- always the real annotated action, regardless
        # of the card outcome. A no-offence standing tackle is still a standing
        # tackle; index 8 ('none') is reserved and unused by any real row.
        action_class_label = ACTION_CLASS_TO_INDEX.get(action_class)
        if action_class_label is None:
            logger.warning(
                "Action %s has unknown Action class %r; skipping.",
                action_id,
                action_class,
            )
            continue

        # (8) handball is an independent binary property, not an action class
        handball_label = 1 if action.get("Handball") == "Handball" else 0

        train_examples.append(
            {
                "action_id": action_id,
                "video_path": str(video_path),
                "action_class_label": action_class_label,
                "offence_severity_class": offence_severity_class,
                "handball_label": handball_label,
            }
        )

    return train_examples, hard_case_examples


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

DEFAULT_ANNOTATIONS = (
    Path(__file__).resolve().parent.parent / "mvfoul_extracted" / "train" / "annotations.json"
)


def _count_drops(annotations_path: Path) -> tuple[int, int]:
    """Re-walk the annotations to report how many rows steps 1 and 2 removed."""
    with open(annotations_path, encoding="utf-8") as f:
        actions = json.load(f)["Actions"]

    dropped_action_class = 0
    dropped_offence = 0
    for action in actions.values():
        action_class = action.get("Action class", "")
        offence = action.get("Offence", "")
        if action_class in ("", "Dont know"):
            dropped_action_class += 1
        elif offence in ("", "Between") and action_class != "Dive":
            dropped_offence += 1
    return dropped_action_class, dropped_offence


def _print_distribution(title: str, counts: Counter, names: list[str], total: int) -> None:
    print(f"\n{title}")
    for index, name in enumerate(names):
        n = counts.get(index, 0)
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {index}  {name:<24} {n:>6}  ({pct:5.1f}%)")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help=f"Path to annotations.json (default: {DEFAULT_ANNOTATIONS})",
    )
    args = parser.parse_args()

    if not args.annotations.exists():
        print(f"annotations.json not found at: {args.annotations}")
        return 1

    with open(args.annotations, encoding="utf-8") as f:
        total_actions = len(json.load(f)["Actions"])

    train_examples, hard_cases = build_dataset(args.annotations)
    dropped_action_class, dropped_offence = _count_drops(args.annotations)

    print(f"Annotations : {args.annotations}")
    print(f"Total actions              : {total_actions}")
    print(f"Kept for training          : {len(train_examples)}")
    print(f"Dropped (Dont know/empty)  : {dropped_action_class}")
    print(f"Dropped (offence ''/Between): {dropped_offence}")
    print(f"Moved to hard_cases        : {len(hard_cases)}")
    accounted = (
        len(train_examples) + dropped_action_class + dropped_offence + len(hard_cases)
    )
    print(f"Unaccounted (warnings)     : {total_actions - accounted}")

    # index 8 ('none') must never be produced -- a no-offence action keeps its
    # real action class label (see change #2 in task-1-brief.md). A regression
    # here must fail loudly and visibly: an explicit check (not a bare `assert`,
    # which `python -O` strips out entirely) that always prints its result, pass
    # or fail, rather than relying on the absence of a traceback.
    label_8_action_ids = [
        e["action_id"] for e in train_examples if e["action_class_label"] == NONE_ACTION_INDEX
    ]
    if label_8_action_ids:
        print(
            f"no-label-8 assertion: FAIL "
            f"({len(label_8_action_ids)} rows with action_class_label == 8 / "
            f"{len(train_examples)})"
        )
        raise RuntimeError(
            "action_class_label == 8 ('none') was assigned to real row(s) -- "
            "index 8 is reserved and must stay unused. Offending action_ids: "
            f"{label_8_action_ids}"
        )
    print(
        f"no-label-8 assertion: PASS "
        f"(0 rows with action_class_label == 8 / {len(train_examples)})"
    )

    action_counts = Counter(e["action_class_label"] for e in train_examples)
    severity_counts = Counter(e["offence_severity_class"] for e in train_examples)
    handball_counts = Counter(e["handball_label"] for e in train_examples)

    _print_distribution(
        "action_class_label distribution:", action_counts, ACTION_CLASSES, len(train_examples)
    )
    _print_distribution(
        "offence_severity_class distribution:",
        severity_counts,
        OFFENCE_SEVERITY_CLASSES,
        len(train_examples),
    )
    print(
        f"\nhandball_label: 1 (Handball) = {handball_counts.get(1, 0)}, "
        f"0 (No handball) = {handball_counts.get(0, 0)}"
    )

    # --- class weights: recompute from this run's parsed data and compare
    # against the baked module-level constants, which are train-split literals.
    action_count_list = [action_counts.get(i, 0) for i in range(len(ACTION_CLASSES))]
    severity_count_list = [
        severity_counts.get(i, 0) for i in range(len(OFFENCE_SEVERITY_CLASSES))
    ]
    action_num_classes = sum(1 for c in action_count_list if c > 0)
    severity_num_classes = sum(1 for c in severity_count_list if c > 0)

    recomputed_action_weights = compute_class_weights(action_count_list, action_num_classes)
    recomputed_severity_weights = compute_class_weights(
        severity_count_list, severity_num_classes
    )

    print("\naction_class_label class weights (recomputed from this run):")
    for index, name in enumerate(ACTION_CLASSES):
        print(
            f"  {index}  {name:<24} count={action_count_list[index]:>6}  "
            f"weight={recomputed_action_weights[index]:.4f}"
        )
    print("\noffence_severity_class class weights (recomputed from this run):")
    for index, name in enumerate(OFFENCE_SEVERITY_CLASSES):
        print(
            f"  {index}  {name:<24} count={severity_count_list[index]:>6}  "
            f"weight={recomputed_severity_weights[index]:.4f}"
        )

    def _warn_on_drift(label: str, recomputed: list[float], baked: list[float]) -> None:
        for index, (r, b) in enumerate(zip(recomputed, baked)):
            if abs(r - b) > 1e-3:
                logging.warning(
                    "%s[%d] recomputed weight %.4f differs from baked constant %.4f "
                    "by more than 1e-3. CLASS_WEIGHTS_* are literals derived from the "
                    "TRAIN split; drift is EXPECTED (not a bug) when running this "
                    "smoke test against any other split (valid/test). If this drift "
                    "is on the TRAIN split itself, the baked constants are stale -- "
                    "regenerate them.",
                    label,
                    index,
                    r,
                    b,
                )

    _warn_on_drift("CLASS_WEIGHTS_ACTION", recomputed_action_weights, CLASS_WEIGHTS_ACTION)
    _warn_on_drift(
        "CLASS_WEIGHTS_OFFENCE_SEVERITY",
        recomputed_severity_weights,
        CLASS_WEIGHTS_OFFENCE_SEVERITY,
    )

    # --- handball imbalance warning: only fires when the positive rate is
    # genuinely low, and the counts come from whatever split was just parsed.
    handball_positive = handball_counts.get(1, 0)
    handball_total = len(train_examples)
    if handball_total > 0 and (handball_positive / handball_total) < 0.05:
        logging.warning(
            "handball_label is severely imbalanced (%d/%d positive) -- consider "
            "excluding from initial training and revisiting after evaluating "
            "action_class performance.",
            handball_positive,
            handball_total,
        )

    if hard_cases:
        reason_counts = Counter(h["reason"] for h in hard_cases)
        print("\nhard_cases by reason:")
        for reason, n in reason_counts.most_common():
            print(f"  {reason:<26} {n:>6}")
        print(f"\nExample hard case: {json.dumps(hard_cases[0]['raw'], indent=2)[:400]} ...")

    missing = sum(1 for e in train_examples if not Path(e["video_path"]).exists())
    print(f"\nTraining clips missing on disk: {missing} / {len(train_examples)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
