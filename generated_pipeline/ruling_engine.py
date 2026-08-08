"""
ruling_engine.py

Final deterministic decision layer for the local soccer foul-detection
pipeline. This module takes the outputs of upstream ML modules (contact
detection, foul classification, severity estimation, and location
detection) and produces a single, auditable ruling.

===========================================================================
SWAP HOOK
===========================================================================
This module is intentionally NOT swappable for an ML model. Every upstream
stage (contact, foul type, severity, location) is allowed to be a learned
model, but the final decision -- turning those signals into a punishment --
must remain deterministic, explainable, and auditable (IFAB-style rules).
Referee-facing / compliance-facing systems need a decision layer that can
be inspected line-by-line, not a black box.

To change ruling behavior, DO NOT swap in a model here. Instead edit the
module-level `PUNISHMENT_TABLE` constant (and the special-case handling in
`make_ruling` for simulation / penalty-box upgrades) to reflect the desired
rule changes.
===========================================================================
"""

import math
from typing import Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid foul type labels produced by the upstream foul classification module.
VALID_FOUL_TYPES = frozenset({
    "tackle",
    "handball",
    "obstruction",
    "simulation",
    "push",
    "none",
})

#: Valid severity labels produced by the upstream severity estimation module.
VALID_SEVERITIES = frozenset({
    "careless",
    "reckless",
    "excessive_force",
})

#: Foul types that do NOT require player-to-player contact to be ruled a foul.
#: Handballs are contact-with-ball (not player) events, and simulation is by
#: definition an absence of genuine contact.
CONTACT_EXEMPT_FOUL_TYPES = frozenset({"handball", "simulation"})

#: Base punishment for each severity level, per IFAB-style disciplinary
#: guidance. This is the single point of control for changing punishment
#: outcomes -- see the SWAP HOOK note above.
PUNISHMENT_TABLE: Dict[str, str] = {
    "careless": "free kick",
    "reckless": "free kick + yellow card",
    "excessive_force": "free kick + red card",
}

#: Confidence values are clamped to this range before use in the geometric
#: mean calculation, to guard against zero, negative, or otherwise malformed
#: upstream confidence scores.
MIN_CONFIDENCE = 0.01
MAX_CONFIDENCE = 1.0

#: Punishment issued for a foul detected while officially "no foul" applies.
PLAY_ON = "play on"

#: Fixed punishment for simulation, regardless of severity or location.
SIMULATION_PUNISHMENT = "free kick + yellow card (simulation)"

#: Per-module confidence threshold: any module reporting confidence below
#: this is flagged in the ruling's `low_confidence_modules` list. Must stay
#: in sync with the pipeline's judgment-layer routing threshold.
LOW_CONFIDENCE_THRESHOLD = 0.65

#: Overall-confidence threshold below which the ruling recommends human
#: review (`human_review_recommended` = True).
HUMAN_REVIEW_THRESHOLD = 0.60

#: The fine-tuned severity head's "no offence" label (OFFENCE_SEVERITY_CLASSES[0] in
#: dataset_builder / foul_classifier). When the foul dict carries this severity-head
#: signal, foul_detected is driven by "is this an offence?" from that trained head
#: rather than the (unreliable) zero-shot CLIP contact gate.
SEVERITY_HEAD_NO_OFFENCE = "No offence"

#: Maps the fine-tuned severity head's card-decision labels
#: (OFFENCE_SEVERITY_CLASSES[1..3] from dataset_builder / foul_classifier) onto the ruling
#: engine's internal careless/reckless/excessive_force severity vocab (PUNISHMENT_TABLE keys).
#: The trained head replaces the optical-flow severity_assessor for the punishment decision;
#: severity_assessor was measured to predict "excessive_force" on ~98% of detected clips.
SEVERITY_HEAD_TO_SEVERITY = {
    "Offence + No card": "careless",
    "Offence + Yellow card": "reckless",
    "Offence + Red card": "excessive_force",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_confidence(value: Union[int, float]) -> float:
    """Clamp a confidence score into the valid [MIN_CONFIDENCE, MAX_CONFIDENCE] range.

    Args:
        value: A raw confidence score, potentially zero, negative, or
            greater than 1.0 due to upstream noise.

    Returns:
        The confidence clamped into [MIN_CONFIDENCE, MAX_CONFIDENCE].
    """
    return min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, float(value)))


def _geometric_mean(confidences: list) -> float:
    """Compute the geometric mean of a list of confidence scores.

    Each input is clamped to [MIN_CONFIDENCE, MAX_CONFIDENCE] before the
    product is taken, guaranteeing the product is always positive and the
    n-th root is well defined.

    Args:
        confidences: A non-empty list of raw confidence scores.

    Returns:
        The geometric mean of the clamped confidence scores.

    Raises:
        ValueError: If `confidences` is empty.
    """
    if not confidences:
        raise ValueError("Cannot compute geometric mean of an empty confidence list")
    clamped = [_clamp_confidence(c) for c in confidences]
    product = math.prod(clamped)
    return math.pow(product, 1.0 / len(clamped))


def _validate_foul_type(foul_type: str) -> None:
    """Validate that a foul type string is recognized.

    Args:
        foul_type: The foul type string to validate.

    Raises:
        ValueError: If `foul_type` is not one of VALID_FOUL_TYPES.
    """
    if foul_type not in VALID_FOUL_TYPES:
        raise ValueError(
            f"Unknown foul_type {foul_type!r}; expected one of "
            f"{sorted(VALID_FOUL_TYPES)}"
        )


def _validate_severity(severity: str) -> None:
    """Validate that a severity string is recognized.

    Args:
        severity: The severity string to validate.

    Raises:
        ValueError: If `severity` is not one of VALID_SEVERITIES.
    """
    if severity not in VALID_SEVERITIES:
        raise ValueError(
            f"Unknown severity {severity!r}; expected one of "
            f"{sorted(VALID_SEVERITIES)}"
        )


def _find_low_confidence_modules(
    contact: dict, foul: dict, severity: dict, location: dict
) -> List[str]:
    """List the modules whose confidence falls below LOW_CONFIDENCE_THRESHOLD.

    Args:
        contact: Output dict of the contact detection module.
        foul: Output dict of the foul classification module.
        severity: Output dict of the severity estimation module.
        location: Output dict of the location detection module.

    Returns:
        Module names (matching the pipeline's module naming) whose reported
        confidence is below the threshold, in fixed pipeline order.
    """
    named_outputs = [
        ("contact_detector", contact),
        ("foul_classifier", foul),
        ("severity_assessor", severity),
        ("location_detector", location),
    ]
    return [
        name
        for name, output in named_outputs
        if float(output["confidence"]) < LOW_CONFIDENCE_THRESHOLD
    ]


def _build_punishment(foul_type: str, severity: str, in_penalty_box: bool) -> str:
    """Construct the punishment string for a confirmed foul.

    Args:
        foul_type: The foul type of the confirmed foul (must not be 'none').
        severity: The severity level of the foul.
        in_penalty_box: Whether the foul occurred inside the penalty box.

    Returns:
        The punishment string, with 'free kick' upgraded to 'penalty kick'
        when the foul occurred in the penalty box (except for simulation,
        which always keeps its fixed punishment).
    """
    if foul_type == "simulation":
        # Simulation is punished on the simulating player regardless of
        # location; it never becomes a penalty kick.
        return SIMULATION_PUNISHMENT

    punishment = PUNISHMENT_TABLE[severity]
    if in_penalty_box:
        punishment = punishment.replace("free kick", "penalty kick")
    return punishment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_ruling(
    contact: dict,
    foul: dict,
    severity: dict,
    location: dict,
    judgment: Optional[dict] = None,
    low_confidence_modules: Optional[List[str]] = None,
) -> dict:
    """Produce the final deterministic ruling for a candidate foul event.

    This is the sole public entry point of the ruling engine. It combines
    the outputs of the contact detection, foul classification, severity
    estimation, and location detection modules into a single ruling dict
    using fixed, auditable IFAB-style rules. When the (optional) judgment
    layer was consulted for an ambiguous case, its foul_type/severity
    override the corresponding module outputs -- but the IFAB rules applied
    to those values are exactly the same.

    Rules applied:
        - Foul detection follows three branches, in priority order: (a) when
          a judgment layer verdict is provided, foul_detected is simply
          judgment['foul_type'] != 'none'; (b) otherwise, when `foul` carries
          the fine-tuned severity head's signal (a 'severity' key), a foul is
          detected when that head predicts an offence (severity != 'No
          offence') AND foul_type != 'none' -- this trained signal is far
          more reliable on soccer-foul clips than the zero-shot CLIP contact
          detector; (c) otherwise (legacy foul dicts without the severity-head
          signal), the original contact-gated rule applies: foul_type !=
          'none' AND (contact['contact'] is True OR foul_type is in
          {'handball', 'simulation'}), since handballs and simulation do
          not require player-to-player contact.
        - Severity (absent a judgment override) prefers the fine-tuned severity
          head's card decision: when `foul` carries a 'severity' key that is one
          of SEVERITY_HEAD_TO_SEVERITY's keys ('Offence + No card' / 'Offence +
          Yellow card' / 'Offence + Red card'), it is mapped via that table to
          the internal careless/reckless/excessive_force vocab and used as
          severity_level. Otherwise (legacy foul dicts without the severity-head
          signal), severity_level falls back to `severity['severity']` from the
          optical-flow severity_assessor.
        - Punishment is looked up from PUNISHMENT_TABLE by severity.
        - Simulation always yields 'free kick + yellow card (simulation)'
          and is never upgraded to a penalty kick.
        - Any other foul committed inside the penalty box has 'free kick'
          upgraded to 'penalty kick' in its punishment string.
        - If no foul is detected, punishment is 'play on', foul_type is
          normalized to 'none', severity is floored to 'careless', and
          in_penalty_box is passed through unchanged.
        - Overall confidence is the geometric mean of the confidences of
          only the modules that actually contributed to the decision.

    Args:
        contact: Dict with keys 'contact' (bool) and 'confidence' (float),
            from the contact detection module.
        foul: Dict with keys 'foul_type' (str, one of VALID_FOUL_TYPES) and
            'confidence' (float), from the foul classification module.
        severity: Dict with keys 'severity' (str, one of VALID_SEVERITIES),
            'confidence' (float), and 'peak_motion' (float), from the
            severity estimation module.
        location: Dict with keys 'in_penalty_box' (bool) and 'confidence'
            (float), from the location detection module.
        judgment: Optional dict from the judgment layer with keys
            'foul_type' (str), 'severity' (str), 'reasoning' (str), and
            'confidence' (float). When provided, its foul_type and severity
            REPLACE the foul/severity module outputs in the ruling logic
            (the judgment layer may override those two judgment calls, but
            never contact or location). None for unambiguous cases.
        low_confidence_modules: Optional list of module names the caller
            (the pipeline's confidence router) flagged as below the routing
            threshold. When None, the list is derived here from the module
            confidences using LOW_CONFIDENCE_THRESHOLD -- both paths yield
            the same result for well-formed inputs.

    Returns:
        A dict with keys:
            'foul_detected' (bool): Whether a punishable foul occurred.
            'foul_type' (str): The ruled foul type ('none' if no foul).
            'severity' (str): The ruled severity ('careless' if no foul).
            'in_penalty_box' (bool): Passed through from `location`.
            'punishment' (str): The final punishment string.
            'confidence' (float): Geometric mean confidence of the
                contributing modules.
            'human_review_recommended' (bool): True when the overall
                confidence is below HUMAN_REVIEW_THRESHOLD (0.60).
            'judgment_layer_used' (bool): True when `judgment` was provided.
            'judgment_layer_reasoning' (str | None): The judgment layer's
                reasoning, or None when it was not consulted.
            'low_confidence_modules' (list[str]): Names of modules whose
                confidence fell below LOW_CONFIDENCE_THRESHOLD (0.65).

    Raises:
        ValueError: If the effective foul_type or severity (from the
            modules, or from `judgment` when provided) is not a recognized
            value.
    """
    if judgment is not None:
        foul_type = judgment["foul_type"]
        severity_level = judgment["severity"]
    else:
        foul_type = foul["foul_type"]
        # Prefer the trained severity head's card decision (mapped to the internal
        # careless/reckless/excessive_force vocab) over the optical-flow severity_assessor,
        # which was measured badly miscalibrated (excessive_force on ~98% of detected clips).
        # Fall back to severity_assessor for legacy foul dicts without the severity-head signal.
        if foul.get("severity") in SEVERITY_HEAD_TO_SEVERITY:
            severity_level = SEVERITY_HEAD_TO_SEVERITY[foul["severity"]]
        else:
            severity_level = severity["severity"]

    _validate_foul_type(foul_type)
    _validate_severity(severity_level)

    in_penalty_box = bool(location["in_penalty_box"])

    if judgment is not None:
        # The judgment layer is the resolver for ambiguous cases; trust its
        # foul_type assertion directly for detection.
        foul_detected = foul_type != "none"
    elif "severity" in foul and foul.get("severity") is not None:
        # Preferred path: the fine-tuned severity head predicts offence vs
        # "No offence" directly. It is far more reliable on soccer-foul clips
        # than the zero-shot CLIP contact detector, which systematically
        # under-detects contact. A foul is detected when the severity head
        # predicts an offence AND the action head predicts a real foul type.
        foul_detected = (foul["severity"] != SEVERITY_HEAD_NO_OFFENCE) and (foul_type != "none")
    else:
        # Legacy fallback for foul dicts WITHOUT the dual-head severity signal
        # (e.g. this module's own smoke test, or any caller passing the old
        # {foul_type, confidence} shape): the original contact-gated IFAB rule.
        has_contact = bool(contact["contact"])
        foul_detected = (foul_type != "none") and (
            has_contact or foul_type in CONTACT_EXEMPT_FOUL_TYPES
        )

    if not foul_detected:
        ruling_foul_type = "none"
        ruling_severity = "careless"
        punishment = PLAY_ON
        contributing_confidences = [foul["confidence"], contact["confidence"]]
    else:
        ruling_foul_type = foul_type
        ruling_severity = severity_level
        punishment = _build_punishment(foul_type, severity_level, in_penalty_box)
        contributing_confidences = [
            foul["confidence"],
            contact["confidence"],
            severity["confidence"],
            location["confidence"],
        ]

    overall_confidence = _geometric_mean(contributing_confidences)

    return {
        "foul_detected": foul_detected,
        "foul_type": ruling_foul_type,
        "severity": ruling_severity,
        "in_penalty_box": in_penalty_box,
        "punishment": punishment,
        "confidence": overall_confidence,
        "human_review_recommended": overall_confidence < HUMAN_REVIEW_THRESHOLD,
        "judgment_layer_used": judgment is not None,
        "judgment_layer_reasoning": (
            judgment["reasoning"] if judgment is not None else None
        ),
        "low_confidence_modules": (
            list(low_confidence_modules)
            if low_confidence_modules is not None
            else _find_low_confidence_modules(contact, foul, severity, location)
        ),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenarios = []

    # 1. Reckless tackle outside the box -> free kick + yellow card.
    scenarios.append((
        "reckless tackle outside box",
        dict(contact={"contact": True, "confidence": 0.92}, foul={"foul_type": "tackle", "confidence": 0.88},
             severity={"severity": "reckless", "confidence": 0.81, "peak_motion": 0.55},
             location={"in_penalty_box": False, "confidence": 0.95}),
        "free kick + yellow card",
    ))

    # 2. Excessive-force tackle inside the box -> penalty kick + red card.
    scenarios.append((
        "excessive-force tackle inside box",
        dict(contact={"contact": True, "confidence": 0.97}, foul={"foul_type": "tackle", "confidence": 0.93},
             severity={"severity": "excessive_force", "confidence": 0.89, "peak_motion": 0.97},
             location={"in_penalty_box": True, "confidence": 0.90}),
        "penalty kick + red card",
    ))

    # 3. Simulation inside the box -> stays a yellow card, not a penalty.
    scenarios.append((
        "simulation inside box",
        dict(contact={"contact": False, "confidence": 0.60}, foul={"foul_type": "simulation", "confidence": 0.77},
             severity={"severity": "careless", "confidence": 0.70, "peak_motion": 0.10},
             location={"in_penalty_box": True, "confidence": 0.85}),
        "free kick + yellow card (simulation)",
    ))

    # 4. Handball without contact -> still a foul (contact-exempt), free kick.
    scenarios.append((
        "handball without contact",
        dict(contact={"contact": False, "confidence": 0.55}, foul={"foul_type": "handball", "confidence": 0.80},
             severity={"severity": "careless", "confidence": 0.65, "peak_motion": 0.05},
             location={"in_penalty_box": False, "confidence": 0.88}),
        "free kick",
    ))

    # 5. No foul -> play on.
    scenarios.append((
        "no foul",
        dict(contact={"contact": True, "confidence": 0.70}, foul={"foul_type": "none", "confidence": 0.91},
             severity={"severity": "careless", "confidence": 0.50, "peak_motion": 0.02},
             location={"in_penalty_box": False, "confidence": 0.60}),
        "play on",
    ))

    for name, kwargs, expected_punishment in scenarios:
        result = make_ruling(**kwargs)
        print(f"[{name}] -> {result}")
        assert result["punishment"] == expected_punishment, (
            f"Scenario '{name}' expected punishment {expected_punishment!r}, "
            f"got {result['punishment']!r}"
        )

    # Sanity check: invalid foul_type / severity should raise ValueError.
    try:
        make_ruling(
            contact={"contact": True, "confidence": 0.9},
            foul={"foul_type": "dive", "confidence": 0.9},
            severity={"severity": "careless", "confidence": 0.9, "peak_motion": 0.1},
            location={"in_penalty_box": False, "confidence": 0.9},
        )
        raise AssertionError("Expected ValueError for invalid foul_type")
    except ValueError:
        pass

    try:
        make_ruling(
            contact={"contact": True, "confidence": 0.9},
            foul={"foul_type": "tackle", "confidence": 0.9},
            severity={"severity": "brutal", "confidence": 0.9, "peak_motion": 0.1},
            location={"in_penalty_box": False, "confidence": 0.9},
        )
        raise AssertionError("Expected ValueError for invalid severity")
    except ValueError:
        pass

    # ------------------------------------------------------------------
    # Confidence-routing fields (judgment layer + human review + flags)
    # ------------------------------------------------------------------

    # 6. No judgment: new fields present with their default values.
    result = make_ruling(
        contact={"contact": True, "confidence": 0.92},
        foul={"foul_type": "tackle", "confidence": 0.88},
        severity={"severity": "reckless", "confidence": 0.81, "peak_motion": 0.55},
        location={"in_penalty_box": False, "confidence": 0.95},
    )
    assert result["judgment_layer_used"] is False
    assert result["judgment_layer_reasoning"] is None
    assert result["human_review_recommended"] is False
    assert result["low_confidence_modules"] == []
    print(f"[no judgment, all confident] -> {result}")

    # 7. Judgment override: judgment's foul_type/severity drive the ruling
    #    (module said reckless push; judge ruled excessive-force tackle).
    judgment = {
        "foul_type": "tackle",
        "severity": "excessive_force",
        "reasoning": "Studs-up contact at speed endangers the opponent's safety.",
        "confidence": 0.72,
    }
    result = make_ruling(
        contact={"contact": True, "confidence": 0.90},
        foul={"foul_type": "push", "confidence": 0.52},
        severity={"severity": "reckless", "confidence": 0.61, "peak_motion": 0.70},
        location={"in_penalty_box": True, "confidence": 0.93},
        judgment=judgment,
    )
    assert result["foul_type"] == "tackle", "judgment foul_type must win"
    assert result["severity"] == "excessive_force", "judgment severity must win"
    assert result["punishment"] == "penalty kick + red card", (
        "IFAB logic (penalty-box upgrade) must apply to the judgment values"
    )
    assert result["judgment_layer_used"] is True
    assert result["judgment_layer_reasoning"] == judgment["reasoning"]
    assert result["low_confidence_modules"] == ["foul_classifier", "severity_assessor"]
    print(f"[judgment override in box] -> {result}")

    # 8. Low overall confidence -> human review recommended.
    result = make_ruling(
        contact={"contact": True, "confidence": 0.55},
        foul={"foul_type": "tackle", "confidence": 0.50},
        severity={"severity": "careless", "confidence": 0.52, "peak_motion": 0.20},
        location={"in_penalty_box": False, "confidence": 0.58},
    )
    assert result["confidence"] < HUMAN_REVIEW_THRESHOLD
    assert result["human_review_recommended"] is True
    assert result["low_confidence_modules"] == [
        "contact_detector", "foul_classifier", "severity_assessor", "location_detector",
    ]
    print(f"[all modules uncertain] -> {result}")

    # 9. Invalid judgment values go through the same validation.
    try:
        make_ruling(
            contact={"contact": True, "confidence": 0.9},
            foul={"foul_type": "tackle", "confidence": 0.9},
            severity={"severity": "careless", "confidence": 0.9, "peak_motion": 0.1},
            location={"in_penalty_box": False, "confidence": 0.9},
            judgment={"foul_type": "tackle", "severity": "brutal",
                      "reasoning": "x", "confidence": 0.9},
        )
        raise AssertionError("Expected ValueError for invalid judgment severity")
    except ValueError:
        pass

    # ------------------------------------------------------------------
    # Severity-head-driven detection (dual-head foul_classifier output)
    # ------------------------------------------------------------------

    # 10. Severity head says offence, contact detector says False -> the
    #     offence signal from the trained severity head overrides the
    #     absent/unreliable contact gate. foul_detected must be True.
    result = make_ruling(
        contact={"contact": False, "confidence": 0.30},
        foul={"foul_type": "tackle", "confidence": 0.85, "severity": "Offence + No card"},
        severity={"severity": "careless", "confidence": 0.80, "peak_motion": 0.40},
        location={"in_penalty_box": False, "confidence": 0.90},
    )
    assert result["foul_detected"] is True, (
        "severity head 'Offence + No card' must drive foul_detected True "
        "even with contact=False"
    )
    print(f"[severity-head offence, no contact] -> {result}")

    # 11. Severity head says no offence, contact detector says True -> the
    #     trained head's 'no offence' verdict overrides contact=True.
    #     foul_detected must be False and punishment must be PLAY_ON.
    result = make_ruling(
        contact={"contact": True, "confidence": 0.90},
        foul={"foul_type": "tackle", "confidence": 0.85, "severity": "No offence"},
        severity={"severity": "careless", "confidence": 0.80, "peak_motion": 0.10},
        location={"in_penalty_box": False, "confidence": 0.90},
    )
    assert result["foul_detected"] is False, (
        "severity head 'No offence' must drive foul_detected False "
        "even with contact=True"
    )
    assert result["punishment"] == PLAY_ON
    print(f"[severity-head no offence, contact True] -> {result}")

    # 12. Severity head says offence (red card), in the penalty box ->
    #     penalty kick + red card, foul_detected True.
    result = make_ruling(
        contact={"contact": True, "confidence": 0.90},
        foul={"foul_type": "tackle", "confidence": 0.88, "severity": "Offence + Red card"},
        severity={"severity": "excessive_force", "confidence": 0.85, "peak_motion": 0.95},
        location={"in_penalty_box": True, "confidence": 0.92},
    )
    assert result["foul_detected"] is True
    assert result["punishment"] == "penalty kick + red card"
    print(f"[severity-head offence red card, in box] -> {result}")

    # ------------------------------------------------------------------
    # Severity-head-driven punishment (severity head wins over severity_assessor)
    # ------------------------------------------------------------------

    # 13. Severity head says "Offence + Yellow card" -> mapped to "reckless",
    #     which wins over the severity_assessor param's "careless".
    result = make_ruling(
        contact={"contact": True, "confidence": 0.90},
        foul={"foul_type": "tackle", "confidence": 0.5, "severity": "Offence + Yellow card"},
        severity={"severity": "careless", "confidence": 0.80, "peak_motion": 0.40},
        location={"in_penalty_box": False, "confidence": 0.90},
    )
    assert result["severity"] == "reckless", (
        "severity head 'Offence + Yellow card' must map to 'reckless', "
        "overriding the severity_assessor param"
    )
    assert result["punishment"] == "free kick + yellow card"
    print(f"[severity-head yellow card wins over assessor] -> {result}")

    # 14. Severity head says "Offence + Red card", in the penalty box ->
    #     penalty kick + red card.
    result = make_ruling(
        contact={"contact": True, "confidence": 0.90},
        foul={"foul_type": "tackle", "confidence": 0.5, "severity": "Offence + Red card"},
        severity={"severity": "careless", "confidence": 0.80, "peak_motion": 0.40},
        location={"in_penalty_box": True, "confidence": 0.90},
    )
    assert result["severity"] == "excessive_force"
    assert result["punishment"] == "penalty kick + red card"
    print(f"[severity-head red card in box] -> {result}")

    # 15. Severity head says "Offence + No card" -> mapped to "careless",
    #     free kick.
    result = make_ruling(
        contact={"contact": True, "confidence": 0.90},
        foul={"foul_type": "tackle", "confidence": 0.5, "severity": "Offence + No card"},
        severity={"severity": "reckless", "confidence": 0.80, "peak_motion": 0.40},
        location={"in_penalty_box": False, "confidence": 0.90},
    )
    assert result["severity"] == "careless"
    assert result["punishment"] == "free kick"
    print(f"[severity-head no card] -> {result}")

    # 16. Legacy foul dict WITHOUT "severity" key -> still uses the
    #     severity_assessor param (unchanged behavior).
    result = make_ruling(
        contact={"contact": True, "confidence": 0.90},
        foul={"foul_type": "tackle", "confidence": 0.85},
        severity={"severity": "excessive_force", "confidence": 0.80, "peak_motion": 0.95},
        location={"in_penalty_box": False, "confidence": 0.90},
    )
    assert result["severity"] == "excessive_force", (
        "legacy foul dict without 'severity' key must fall back to severity_assessor"
    )
    assert result["punishment"] == "free kick + red card"
    print(f"[legacy foul dict, no severity-head key] -> {result}")

    print("\nAll smoke tests passed.")
