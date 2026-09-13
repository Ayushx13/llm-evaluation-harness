"""
Support ticket classifier - the system under test.

Design decision:
The classifier returns an `evidence_span` along with the predicted
`category` and `priority`. The evidence span must be copied verbatim
from the input ticket.

This gives the evaluation harness an objective way to test whether the
model's supporting evidence is grounded in the input. The harness checks
whether the returned span exists exactly in the original input. An invalid
or missing span is treated as unsupported evidence.

Task correctness and evidence validity are evaluated separately:
- `category` and `priority` are compared against the ground-truth labels.
- `evidence_span` is checked against the original input.

The classifier intentionally does not modify or repair an invalid evidence
span. The evaluation harness must observe the model's raw output so that
failures remain measurable.

Reproducibility:
The model is configured with temperature 0 and a fixed seed to reduce
randomness. Because model APIs may still produce different responses after
serving-stack or model changes, responses are cached on disk.

The cache key includes the model, prompt, input, generation configuration,
and output schema. Therefore, changing any of these produces a new cache
entry rather than silently reusing an outdated response.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import time
from typing import Any, Iterable, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from app.prompts import SYSTEM_PROMPT_V1, get_prompt
from app.schema import TicketOutput

load_dotenv()

# A "-latest" alias re-points to a new checkpoint without warning, which would
# invalidate every number in the report the day it moves. Pin a real version.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
CACHE = pathlib.Path(os.getenv("CLASSIFIER_CACHE", ".cache"))
CACHE.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_OUTPUT_TOKENS = 1024
# Gemini 3 models cannot turn thinking off; "minimal" is the floor the API
# accepts. thinking_budget=0 is a Gemini 2.x knob and is rejected with a bare
# 400 INVALID_ARGUMENT here.
THINKING_LEVEL = "minimal"
MAX_ATTEMPTS = 5
RETRY_BASE_DELAY = 1.0
# Transient on the server side; anything else is our bug and should surface.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

# Client-side pacing. The free tier allows 15 requests per minute per model;
# the default stays one under that so clock skew cannot trip the limit.
GEMINI_RPM = float(os.getenv("GEMINI_RPM", "14"))
MIN_INTERVAL = 60.0 / GEMINI_RPM
# If the server asks us to wait longer than this, give up rather than hang.
MAX_RETRY_DELAY = 300.0

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Retry jitter draws from a private RNG. Using the global `random` module here
# would advance the shared stream and change the output of the seeded generator
# in evals/adversarial.py, making adversarial inputs depend on how many times
# the API happened to rate-limit us.
_rng = random.Random(SEED)

_STATS = {"cache_hits": 0, "api_calls": 0, "retries": 0}


class ClassifierError(RuntimeError):
    """The classifier could not produce a valid TicketOutput for an input."""


def _config(prompt: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=prompt,
        temperature=0.0,
        top_p=1.0,
        seed=SEED,
        candidate_count=1,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=TicketOutput,
        # Reasoning tokens are sampled too, so every extra one is another source
        # of run-to-run drift on what is only a single-label task.
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        # Nothing here declares tools, and the SDK warns on every call unless
        # automatic function calling is explicitly off.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def _cache_key(text: str, prompt: str) -> str:
    """Hash everything that could change the answer, not just the text."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "text": text,
        "config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": SEED,
            "candidate_count": 1,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "thinking_level": THINKING_LEVEL,
        },
        # Renaming a category or widening an enum changes what a correct answer
        # is, so it has to bust the cache.
        "schema": TicketOutput.model_json_schema(),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _refusal_reason(resp: Any) -> str:
    """Explain an empty response. Adversarial inputs trip safety filters."""
    feedback = getattr(resp, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        return f"prompt blocked: {feedback.block_reason}"
    for candidate in getattr(resp, "candidates", None) or []:
        finish = getattr(candidate, "finish_reason", None)
        if finish:
            return f"finish_reason: {finish}"
    return "empty response with no reason given"


def _write_atomic(path: pathlib.Path, obj: dict) -> None:
    """Write via a temp file so an interrupted run cannot leave half a JSON."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


_last_call = 0.0


def _throttle() -> None:
    """Space live calls MIN_INTERVAL apart. Cache hits never reach this."""
    global _last_call
    wait = _last_call + MIN_INTERVAL - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _quota_info(exc: genai_errors.APIError) -> tuple[float | None, bool]:
    """Return (server-requested retry delay in seconds, whether a per-day quota was hit).

    A 429 body carries RetryInfo ("22s") and QuotaFailure entries. Our own
    backoff tops out well under a typical retryDelay, so ignoring it just
    burns every attempt on another 429.
    """
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None, False
    delay, daily = None, False
    for item in (details.get("error") or {}).get("details") or []:
        kind = item.get("@type", "")
        if kind.endswith("RetryInfo"):
            try:
                delay = float(str(item.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                pass
        elif kind.endswith("QuotaFailure"):
            daily = any("PerDay" in v.get("quotaId", "") for v in item.get("violations") or [])
    return delay, daily


def _generate(text: str, config: types.GenerateContentConfig) -> str:
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        _throttle()
        try:
            _STATS["api_calls"] += 1
            resp = client.models.generate_content(
                model=MODEL,
                contents=text,
                config=config,
            )
        except genai_errors.APIError as exc:
            status = getattr(exc, "code", None)
            if status not in RETRYABLE_STATUS or attempt == MAX_ATTEMPTS - 1:
                raise ClassifierError(f"{MODEL} call failed ({status}): {exc}") from exc
            last_error = exc

            wait = RETRY_BASE_DELAY * (2**attempt)
            if status == 429:
                server_delay, daily = _quota_info(exc)
                if daily:
                    # Retrying cannot help until the quota resets; stop now so
                    # the run fails fast instead of recording hundreds of errors.
                    raise ClassifierError(
                        f"{MODEL} daily quota exhausted. Rerun after it resets; "
                        "finished responses are cached and will not be re-requested."
                    ) from exc
                if server_delay is not None:
                    if server_delay > MAX_RETRY_DELAY:
                        raise ClassifierError(
                            f"{MODEL} asked for a {server_delay:.0f}s wait, over the "
                            f"{MAX_RETRY_DELAY:.0f}s limit: {exc}"
                        ) from exc
                    wait = max(wait, server_delay)
        else:
            raw = (resp.text or "").strip()
            if raw:
                return raw
            # A refusal is a real result about the input, not a flaky call, so
            # it must not be retried into looking like a network problem.
            raise ClassifierError(f"{MODEL} returned no text - {_refusal_reason(resp)}")

        _STATS["retries"] += 1
        time.sleep(wait + _rng.uniform(0, 0.25))

    raise ClassifierError(f"{MODEL} call failed after {MAX_ATTEMPTS} attempts: {last_error}")


def classify(
    text: str,
    prompt: str = SYSTEM_PROMPT_V1,
    *,
    prompt_version: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Classify one ticket. Returns a dict matching TicketOutput.

    prompt_version ("v1", "v2", ...) overrides `prompt` with the named version
    from app.prompts, which is how the eval runner selects a prompt.

    Raises ClassifierError if the model refuses, keeps failing, or returns
    something that does not validate against the schema.
    """
    if prompt_version is not None:
        prompt = get_prompt(prompt_version)

    path = CACHE / f"{_cache_key(text, prompt)}.json"

    if use_cache and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            _STATS["cache_hits"] += 1
            return cached
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)  # truncated by an older interrupted run

    raw = _generate(text, _config(prompt))

    try:
        out = TicketOutput.model_validate_json(raw).model_dump(mode="json")
    except ValidationError as exc:
        # response_schema makes this rare, but it still happens when the output
        # hits the token cap mid-object. Never cache it.
        raise ClassifierError(f"invalid output: {exc}\nraw: {raw[:500]}") from exc

    _write_atomic(path, out)
    return out


def classify_many(
    texts: Iterable[str],
    prompt: str = SYSTEM_PROMPT_V1,
    *,
    prompt_version: str | None = None,
    use_cache: bool = True,
    on_error: Literal["raise", "record"] = "raise",
) -> list[dict]:
    """Classify a batch, preserving input order.

    on_error="record" replaces a failed case with {"error": "..."} instead of
    aborting the run. Use it for the adversarial suite, where a refusal is a
    finding worth counting rather than a crash worth debugging.
    """
    results: list[dict] = []
    for text in texts:
        try:
            results.append(
                classify(text, prompt, prompt_version=prompt_version, use_cache=use_cache)
            )
        except ClassifierError as exc:
            if on_error == "raise":
                raise
            results.append({"error": str(exc)})
    return results


PARAPHRASE_PROMPT = """Rewrite the customer support message you are given using different words.
Keep every fact, number, name, and the customer's request exactly the same.
Do not add or remove information, do not answer the message, and do not change its tone.
Return only the rewritten message."""


def paraphrase(text: str, *, use_cache: bool = True) -> str:
    """Reword a ticket for the adversarial suite. Cached like classify().

    The label is never taken from this call; evals/adversarial.py inherits it
    from the base case.
    """
    payload = {
        "task": "paraphrase",
        "model": MODEL,
        "prompt": PARAPHRASE_PROMPT,
        "text": text,
        "config": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": SEED,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "thinking_level": THINKING_LEVEL,
        },
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    path = CACHE / f"paraphrase_{hashlib.sha256(blob.encode('utf-8')).hexdigest()}.json"

    if use_cache and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))["text"]
            _STATS["cache_hits"] += 1
            return cached
        except (json.JSONDecodeError, KeyError):
            path.unlink(missing_ok=True)

    config = types.GenerateContentConfig(
        system_instruction=PARAPHRASE_PROMPT,
        temperature=0.0,
        top_p=1.0,
        seed=SEED,
        candidate_count=1,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    out = _generate(text, config)
    _write_atomic(path, {"text": out})
    return out


def stats() -> dict:
    """Cache hits vs live calls, so the report can say how much was replayed."""
    return dict(_STATS)


if __name__ == "__main__":
    import sys

    sample = " ".join(sys.argv[1:]) or (
        "I was charged twice for order #4471 and the second charge is still pending."
    )
    print(json.dumps(classify(sample), indent=2))
    print(json.dumps(stats(), indent=2), file=sys.stderr)
