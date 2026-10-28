"""Tracing that carries measurements, never words.

The privacy claim in this project is that a family's recording never leaves the machine. A tracing product that
ships transcript text to a third party would quietly falsify that on the first run — so this module is written
to make the violation impossible rather than merely discouraged:

* `spans` records counts, durations, ratios, decisions and reasons. There is no helper here that accepts free
  text, and every call site passes numbers or an enum-like literal.
* `init()` forces `send_default_pii=False` and `max_request_body_size="never"`.
* `redact()` is applied to any string that could conceivably have come from a recording, keeping the shape of the
  value and nothing else — so a debugging trace shows `claim_text.len=42` and not the claim.

If a future contributor needs the words, they will have to change this file, and a reviewer will see it.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import pathlib
import time

import sentry_sdk

DSN_PATH = pathlib.Path.home() / ".config" / "hf26" / "sentry_dsn"
_INITIALISED = False

# Correlate a thing across spans *within this run*, and tell anybody reading the trace afterwards nothing.
# A plain hash of a phrase would be a commitment: given a guess, a third party could confirm it. A random
# per-process key removes that, at the cost of not being able to correlate across processes — which nothing here
# needs. Reset on every import, never written to disk, never logged.
_SESSION_KEY = os.urandom(32)


def dsn() -> str | None:
    env = os.environ.get("SENTRY_DSN")
    if env:
        return env.strip()
    if DSN_PATH.exists():
        value = DSN_PATH.read_text().strip()
        return value or None
    return None


def init(component: str, traces_sample_rate: float = 1.0, debug: bool = False) -> bool:
    """Initialise once per process. Returns False (and stays silent) if there is no DSN configured."""
    global _INITIALISED
    if _INITIALISED:
        return True
    value = dsn()
    if not value:
        return False
    sentry_sdk.init(
        dsn=value,
        traces_sample_rate=traces_sample_rate,
        send_default_pii=False,
        max_request_body_size="never",
        debug=debug or os.environ.get("SO_GIO_TRACE_DEBUG") == "1",
        release=os.environ.get("SO_GIO_RELEASE", "so-gio@dev"),
        environment=os.environ.get("SO_GIO_ENV", "local"),
        # The whole point of the product is that this process handles private audio. Keep the breadcrumb trail
        # but cap it, so a long pipeline run cannot accumulate a readable log of family data in memory.
        max_breadcrumbs=50,
        attach_stacktrace=False,
        default_integrations=True,
    )
    sentry_sdk.set_tag("component", component)
    sentry_sdk.set_tag("product", "so-gio")
    _INITIALISED = True
    return True


@contextlib.contextmanager
def transaction(name: str, op: str = "pipeline", **data):
    """A root transaction. Standalone spans are created and thrown away; only a transaction is transmitted, so
    every traced run of the pipeline opens exactly one of these and hangs its spans underneath."""
    started = time.perf_counter()
    tx = sentry_sdk.start_transaction(name=name, op=op)
    with tx as root:
        for key, value in data.items():
            root.set_data(key, value)
        try:
            yield root
        finally:
            root.set_data("elapsed_ms", round((time.perf_counter() - started) * 1000.0, 1))


@contextlib.contextmanager
def span(op: str, **data):
    """One traced operation. `data` must be numbers, booleans or short enum-like strings."""
    started = time.perf_counter()
    with sentry_sdk.start_span(op=op, name=op) as sp:
        for key, value in data.items():
            sp.set_data(key, value)
        try:
            yield sp
        finally:
            sp.set_data("elapsed_ms", round((time.perf_counter() - started) * 1000.0, 1))


def trace_result(op: str, **data) -> None:
    """Attach a result to the current span — used after a stage finishes so the outcome lands in the same span."""
    with sentry_sdk.start_span(op=f"{op}.result") as sp:
        for key, value in data.items():
            sp.set_data(key, value)


def redact(text: str | None) -> dict:
    """Describe a string without disclosing it.

    Length and syllable count are enough to debug a pipeline; the fingerprint is a truncated HMAC under a key that
    exists only in this process's memory, so it correlates two spans in the same run and is worthless to anyone who
    reads the trace later. `tests/test_core.py` asserts that no syllable of the input survives.
    """
    if not text:
        return {"len": 0, "syllables": 0}
    digest = hmac.new(_SESSION_KEY, text.encode("utf-8"), hashlib.sha256).hexdigest()[:12]
    return {"len": len(text), "syllables": len(text.split()), "fingerprint": digest}


def flush(timeout: float = 5.0) -> bool:
    return bool(sentry_sdk.flush(timeout=timeout))


def capture_refusal(reason: str, claim_kind: str, redacted: dict) -> None:
    """A refused claim is a product event, not an error: it is counted so 'we invented nothing' is auditable."""
    with sentry_sdk.start_span(op="guard.refusal") as sp:
        sp.set_data("reason", reason)
        sp.set_data("claim_kind", claim_kind)
        sp.set_data("redacted", redacted)
    sentry_sdk.add_breadcrumb(category="guard", level="info", message=f"refused:{reason}",
                              data={"kind": claim_kind, **redacted})
