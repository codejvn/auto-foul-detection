"""
judgment_layer.py
=================

Claude (Sonnet 5) judgment layer for the auto-foul-detection pipeline.

This module is ONLY invoked for ambiguous cases: when any upstream CV
module reports confidence below the routing threshold (0.65), the pipeline
hands the full set of structured module outputs to Claude Sonnet 5, which
acts as a referee judgment layer. Unambiguous cases never reach this
module -- they go straight from the CV modules to the deterministic
``ruling_engine``.

Division of authority
---------------------
The judgment layer may OVERRIDE:
    - ``foul_type``  (from foul_classifier -- a judgment call)
    - ``severity``   (from severity_assessor -- a judgment call)

It may NEVER override:
    - ``contact``          (from contact_detector -- a factual observation)
    - ``in_penalty_box``   (from location_detector -- a factual observation)

This split is enforced structurally: the model's output schema simply has
no fields for contact or location, so there is nothing for it to override.

Input contract (from pipeline.py)
---------------------------------
    {
      "contact":  {"contact": bool, "confidence": float},
      "foul":     {"foul_type": str, "confidence": float},
      "severity": {"severity": str, "confidence": float, "peak_motion": float},
      "location": {"in_penalty_box": bool, "confidence": float},
      "low_confidence_modules": list[str]
    }

Output contract (consumed by ruling_engine.make_ruling(judgment=...))
---------------------------------------------------------------------
    {"foul_type": str, "severity": str, "reasoning": str, "confidence": float}

API usage
---------
Uses the official ``anthropic`` Python SDK against ``claude-sonnet-5``
with structured JSON output (``output_config.format`` with a
``json_schema``), so the response is guaranteed to be valid JSON matching
the output contract's shape. Requires the ``ANTHROPIC_API_KEY``
environment variable; a clear ``RuntimeError`` is raised when it is
missing.

SWAP HOOK (model / caller)
--------------------------
``MODEL_ID`` is a module constant -- point it at a different Claude model
to swap the judge. For testing (or an offline fallback), reassign the
module-level ``CLAUDE_CALLER`` callable, which has the signature
``(module_outputs: dict) -> str`` and returns the model's raw JSON text:

    import judgment_layer
    judgment_layer.CLAUDE_CALLER = my_fake_caller

This mirrors the FRAME_SELECTOR / CUT_DETECTOR conventions used elsewhere
in this pipeline. The ``__main__`` smoke test uses exactly this hook to
mock the API call, so it runs without a key and without network access.

GPU note: this module is API-only -- zero local GPU memory.
"""

from __future__ import annotations

import json
import os
from typing import Callable

# ---------------------------------------------------------------------------
# Configuration / swap hooks
# ---------------------------------------------------------------------------

#: SWAP HOOK: the Claude model used as the judge. Sonnet 5 per the project
#: plan (strong judgment quality at Sonnet cost; ambiguous cases only).
MODEL_ID: str = "claude-sonnet-5"

#: Max output tokens for the judgment call. Sonnet 5 runs adaptive thinking
#: by default and thinking tokens count against this limit, so it is sized
#: well above the (small) JSON answer itself.
MAX_TOKENS: int = 8192

#: Valid foul types -- must stay in sync with foul_classifier.FOUL_TYPES.
FOUL_TYPES: tuple[str, ...] = (
    "tackle", "handball", "obstruction", "simulation", "push", "none",
)

#: Valid severities -- must stay in sync with severity_assessor's labels.
SEVERITIES: tuple[str, ...] = ("careless", "reckless", "excessive_force")

#: Required keys in the input dict, with the sub-keys each must contain.
_REQUIRED_INPUT_KEYS: dict[str, tuple[str, ...]] = {
    "contact": ("contact", "confidence"),
    "foul": ("foul_type", "confidence"),
    "severity": ("severity", "confidence", "peak_motion"),
    "location": ("in_penalty_box", "confidence"),
}

SYSTEM_PROMPT: str = """\
You are the judgment layer of an automated soccer refereeing (VAR) system.

You receive structured visual evidence produced by computer-vision modules
that analyzed a short video clip of a potential foul:
- contact_detector: whether player-to-player contact occurred (CLIP zero-shot)
- foul_classifier: the type of foul (VideoMAE action recognition)
- severity_assessor: severity estimated from optical-flow motion intensity,
  plus the raw peak_motion value that drove it
- location_detector: whether the incident is inside the penalty box (CLIP)

Each module reports a confidence in [0, 1]. You are consulted ONLY because
at least one module's confidence fell below 0.65; those modules are listed
in low_confidence_modules. Weigh low-confidence outputs skeptically and
high-confidence outputs heavily.

Your authority is strictly limited:
- You MAY override foul_type and severity -- these are judgment calls.
- You MUST NOT dispute or reinterpret contact or in_penalty_box -- these are
  factual observations outside your authority. Treat them as ground truth
  and make your foul_type/severity judgment consistent with them (e.g. if
  contact is false, the plausible foul types are "none" or "simulation").

Apply the IFAB Laws of the Game definitions when judging severity:
- careless: the player showed a lack of attention or consideration when
  making a challenge, or acted without precaution. No disciplinary sanction
  is needed beyond the free kick.
- reckless: the player acted with disregard to the danger to, or
  consequences for, an opponent. A reckless challenge is cautioned
  (yellow card).
- excessive_force: the player exceeded the necessary use of force and/or
  endangered the safety of an opponent. A challenge using excessive force
  is sanctioned with a sending-off (red card).

Respond with your best judgment of foul_type and severity, a brief
reasoning grounded in the evidence and the IFAB definitions above, and
your own confidence in [0, 1] reflecting how decisive the evidence is.
Valid foul_type values: tackle, handball, obstruction, simulation, push,
none. Valid severity values: careless, reckless, excessive_force. If
foul_type is "none" or "simulation", report severity "careless" (it is
ignored downstream)."""

#: JSON schema enforced on the model's response via structured outputs.
#: Note: numeric range constraints aren't supported by the API's schema
#: subset, so confidence is clamped to [0, 1] in code after parsing.
JUDGMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "foul_type": {
            "type": "string",
            "enum": list(FOUL_TYPES),
            "description": "The judged foul type, possibly overriding the classifier.",
        },
        "severity": {
            "type": "string",
            "enum": list(SEVERITIES),
            "description": "The judged severity under IFAB definitions.",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief justification grounded in the module evidence.",
        },
        "confidence": {
            "type": "number",
            "description": "Judgment confidence in [0, 1].",
        },
    },
    "required": ["foul_type", "severity", "reasoning", "confidence"],
    "additionalProperties": False,
}

#: SWAP HOOK: the callable that actually talks to Claude. Signature:
#: ``(module_outputs: dict) -> str`` returning the raw JSON response text.
#: The smoke test replaces this with a mock; an offline deployment could
#: point it at a local model. See module docstring.
CLAUDE_CALLER: Callable[[dict], str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_ambiguous_case(module_outputs: dict) -> dict:
    """Ask the Claude judgment layer to adjudicate an ambiguous case.

    Args:
        module_outputs: Dict with keys ``"contact"``, ``"foul"``,
            ``"severity"``, ``"location"`` (each the raw output dict of the
            corresponding CV module) and ``"low_confidence_modules"`` (list
            of module names whose confidence fell below 0.65).

    Returns:
        ``{"foul_type": str, "severity": str, "reasoning": str,
        "confidence": float}`` where ``foul_type``/``severity`` may differ
        from the upstream classifier/assessor outputs (an override), and
        ``confidence`` is clamped to ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``module_outputs`` is missing required keys, or the
            model response fails validation against the output contract.
        RuntimeError: If ``ANTHROPIC_API_KEY`` is not set, the ``anthropic``
            package is not installed, or the API call fails.
    """
    _validate_input(module_outputs)
    raw_response = CLAUDE_CALLER(module_outputs)
    return _parse_and_validate_judgment(raw_response)


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------


def _call_claude_api(module_outputs: dict) -> str:
    """Send the evidence to claude-sonnet-5 and return its raw JSON text.

    Uses structured outputs (``output_config.format`` with
    ``JUDGMENT_SCHEMA``) so the returned text block is guaranteed to be
    valid JSON matching the judgment contract's shape.

    Args:
        module_outputs: The validated module-output dict.

    Returns:
        The model's response text (a JSON object string).

    Raises:
        RuntimeError: If ``ANTHROPIC_API_KEY`` is unset, the SDK is not
            installed, or the API call fails (with the failure category --
            auth, rate limit, connection, API error -- in the message).
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "The judgment layer requires an Anthropic API key: the "
            "ANTHROPIC_API_KEY environment variable is not set. Ambiguous "
            "cases (any module confidence < 0.65) are routed to Claude "
            f"({MODEL_ID}) for adjudication and cannot be processed "
            "without it. Set ANTHROPIC_API_KEY, or route the case to "
            "human review instead."
        )

    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The judgment layer requires the 'anthropic' package "
            "(pip install anthropic); it is listed in requirements.txt."
        ) from exc

    client = anthropic.Anthropic()

    user_message = (
        "Adjudicate this potential foul. Structured evidence from the CV "
        "modules:\n\n"
        + json.dumps(module_outputs, indent=2, sort_keys=True)
    )

    try:
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            output_config={
                "format": {"type": "json_schema", "schema": JUDGMENT_SCHEMA}
            },
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError as exc:
        raise RuntimeError(
            "Judgment layer authentication failed: the ANTHROPIC_API_KEY "
            "is set but was rejected by the API. Check that the key is "
            "valid and active."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise RuntimeError(
            "Judgment layer rate-limited by the Anthropic API. The SDK "
            "already retried with backoff; wait and retry this clip, or "
            "reduce concurrent validator workers."
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError(
            "Judgment layer could not reach the Anthropic API (network "
            "error). Check connectivity and retry."
        ) from exc
    except anthropic.APIStatusError as exc:
        raise RuntimeError(
            f"Judgment layer API call failed with HTTP {exc.status_code}: "
            f"{exc.message}"
        ) from exc

    if response.stop_reason == "refusal":
        raise RuntimeError(
            "Judgment layer request was refused by the model; route this "
            "clip to human review."
        )

    text = next(
        (block.text for block in response.content if block.type == "text"),
        None,
    )
    if text is None:
        raise RuntimeError(
            "Judgment layer response contained no text block "
            f"(stop_reason={response.stop_reason!r})."
        )
    return text


CLAUDE_CALLER = _call_claude_api


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_input(module_outputs: dict) -> None:
    """Check that the input dict matches the judgment-layer input contract.

    Args:
        module_outputs: Candidate input dict.

    Raises:
        ValueError: On any missing top-level key, missing sub-key, or a
            missing/invalid ``low_confidence_modules`` list.
    """
    if not isinstance(module_outputs, dict):
        raise ValueError(
            f"module_outputs must be a dict, got {type(module_outputs).__name__}."
        )

    for key, sub_keys in _REQUIRED_INPUT_KEYS.items():
        if key not in module_outputs:
            raise ValueError(f"module_outputs is missing required key '{key}'.")
        section = module_outputs[key]
        if not isinstance(section, dict):
            raise ValueError(f"module_outputs['{key}'] must be a dict.")
        for sub_key in sub_keys:
            if sub_key not in section:
                raise ValueError(
                    f"module_outputs['{key}'] is missing required "
                    f"sub-key '{sub_key}'."
                )

    if not isinstance(module_outputs.get("low_confidence_modules"), list):
        raise ValueError(
            "module_outputs must include 'low_confidence_modules' as a list "
            "of module names with confidence < 0.65."
        )


def _parse_and_validate_judgment(raw_response: str) -> dict:
    """Parse the model's JSON text and enforce the output contract.

    Structured outputs already guarantee the shape when the real API is
    used, but the caller is swappable (CLAUDE_CALLER), so the contract is
    re-validated here regardless of backend.

    Args:
        raw_response: The raw JSON text returned by the Claude caller.

    Returns:
        The validated judgment dict, with ``confidence`` coerced to float
        and clamped to ``[0.0, 1.0]``.

    Raises:
        ValueError: If the text is not valid JSON or violates the contract.
    """
    try:
        judgment = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Judgment layer returned invalid JSON: {exc}. "
            f"Raw response: {raw_response[:200]!r}"
        ) from exc

    if not isinstance(judgment, dict):
        raise ValueError(
            f"Judgment must be a JSON object, got {type(judgment).__name__}."
        )

    if judgment.get("foul_type") not in FOUL_TYPES:
        raise ValueError(
            f"Judgment foul_type {judgment.get('foul_type')!r} is not one "
            f"of {FOUL_TYPES}."
        )
    if judgment.get("severity") not in SEVERITIES:
        raise ValueError(
            f"Judgment severity {judgment.get('severity')!r} is not one "
            f"of {SEVERITIES}."
        )
    if not isinstance(judgment.get("reasoning"), str) or not judgment["reasoning"]:
        raise ValueError("Judgment must include a non-empty 'reasoning' string.")
    if not isinstance(judgment.get("confidence"), (int, float)):
        raise ValueError("Judgment must include a numeric 'confidence'.")

    return {
        "foul_type": judgment["foul_type"],
        "severity": judgment["severity"],
        "reasoning": judgment["reasoning"],
        "confidence": max(0.0, min(1.0, float(judgment["confidence"]))),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _example_ambiguous_input() -> dict:
    """Build a representative ambiguous-case input for the smoke test."""
    return {
        "contact": {"contact": True, "confidence": 0.82},
        "foul": {"foul_type": "tackle", "confidence": 0.51},
        "severity": {
            "severity": "reckless",
            "confidence": 0.58,
            "peak_motion": 14.3,
        },
        "location": {"in_penalty_box": False, "confidence": 0.91},
        "low_confidence_modules": ["foul_classifier", "severity_assessor"],
    }


def _run_smoke_test() -> None:
    """Self-contained smoke test: mocks the Claude API call via the
    CLAUDE_CALLER swap hook, so it needs no API key and no network."""
    original_caller = CLAUDE_CALLER
    original_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    module = __import__(__name__)

    try:
        # 1. Happy path with a mocked API response.
        mocked_judgment = {
            "foul_type": "tackle",
            "severity": "careless",
            "reasoning": (
                "Contact is confirmed with high confidence, but peak motion "
                "is modest and the classifier is uncertain; under IFAB this "
                "reads as a careless challenge, not a reckless one."
            ),
            "confidence": 0.74,
        }
        captured_inputs: list[dict] = []

        def _mock_caller(module_outputs: dict) -> str:
            captured_inputs.append(module_outputs)
            return json.dumps(mocked_judgment)

        module.CLAUDE_CALLER = _mock_caller
        print("[smoke test] Evaluating ambiguous case with mocked API...")
        result = evaluate_ambiguous_case(_example_ambiguous_input())
        print(f"[smoke test]   judgment: {json.dumps(result, indent=2)}")

        assert result == mocked_judgment, "judgment should round-trip the mock"
        assert captured_inputs and captured_inputs[0]["foul"]["foul_type"] == "tackle", (
            "mock caller should receive the module outputs"
        )
        assert set(result) == {"foul_type", "severity", "reasoning", "confidence"}

        # 2. Severity override: judgment may disagree with severity_assessor.
        assert result["severity"] != _example_ambiguous_input()["severity"]["severity"], (
            "smoke case intentionally exercises a severity override"
        )
        print("[smoke test] Severity override path exercised "
              "(reckless -> careless).")

        # 3. Confidence clamping.
        module.CLAUDE_CALLER = lambda _: json.dumps(
            {**mocked_judgment, "confidence": 1.7}
        )
        clamped = evaluate_ambiguous_case(_example_ambiguous_input())
        assert clamped["confidence"] == 1.0, "confidence must clamp to [0, 1]"
        print("[smoke test] Out-of-range confidence clamped to 1.0.")

        # 4. Contract violations from the (mocked) model are rejected.
        for bad_response, label in [
            ("not json at all", "invalid JSON"),
            (json.dumps({**mocked_judgment, "severity": "brutal"}),
             "invalid severity"),
            (json.dumps({**mocked_judgment, "foul_type": "headbutt"}),
             "invalid foul_type"),
        ]:
            module.CLAUDE_CALLER = lambda _, r=bad_response: r
            try:
                evaluate_ambiguous_case(_example_ambiguous_input())
            except ValueError as exc:
                print(f"[smoke test] Correctly rejected {label}: {exc}")
            else:
                raise AssertionError(f"Expected ValueError for {label}")

        # 5. Malformed input is rejected before any API call.
        module.CLAUDE_CALLER = _mock_caller
        try:
            evaluate_ambiguous_case({"contact": {"contact": True}})
        except ValueError as exc:
            print(f"[smoke test] Correctly rejected malformed input: {exc}")
        else:
            raise AssertionError("Expected ValueError for malformed input")

        # 6. Missing API key raises a clear RuntimeError on the real path.
        module.CLAUDE_CALLER = original_caller
        assert "ANTHROPIC_API_KEY" not in os.environ
        try:
            evaluate_ambiguous_case(_example_ambiguous_input())
        except RuntimeError as exc:
            assert "ANTHROPIC_API_KEY" in str(exc)
            print(f"[smoke test] Correctly raised RuntimeError without "
                  f"API key: {exc}")
        else:
            raise AssertionError("Expected RuntimeError when API key is unset")

        print("[smoke test] PASSED.")
    finally:
        module.CLAUDE_CALLER = original_caller
        if original_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = original_key


if __name__ == "__main__":
    _run_smoke_test()
