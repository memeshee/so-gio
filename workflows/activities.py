"""Side effects for :class:`workflows.gio.GioWorkflow`.

Activities are where the parts that can fail live: wall-clock time, the filesystem, and (in
production) the Zalo/SMS/email client. Keeping them out of the workflow is what lets the workflow
itself be replayed after a crash without sending the same reminder twice.

The demo outbox is a JSONL file, one line per reminder, under `scripts/state/` (git-ignored).
Every line carries the pid of the worker that wrote it — that is the fingerprint the durability
proof uses to show a *different* process finished the job.
"""
from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from temporalio import activity

from workflows.gio import Delivery, Reminder


def state_dir() -> Path:
    """Where the demo outbox and ledgers live. Overridable so tests do not share a directory."""
    raw = os.environ.get("GIO_STATE_DIR")
    path = Path(raw) if raw else Path(__file__).resolve().parents[1] / "scripts" / "state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())  # the proof kills processes; do not let a reminder sit in a buffer


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn line from a process killed mid-write is not a reason to lose the rest
    return rows


@activity.defn
async def send_reminder(rem: Reminder) -> dict:
    """Deliver one giỗ reminder, or fail so the workflow's retry policy retries it.

    Idempotency: if this (workflow_id, cycle) already left the building, return the existing
    receipt instead of sending a second message. Temporal guarantees at-least-once activity
    execution, so the sender must be the one that makes "at least once" look like "once".
    """
    info = activity.info()
    outbox = state_dir() / "reminders.jsonl"
    key = f"{rem.workflow_id}#{rem.cycle}"

    for row in _read_jsonl(outbox):
        if row.get("key") == key:
            return {
                "message_id": row["message_id"],
                "attempt": row["attempt"],
                "worker_pid": row["worker_pid"],
                "sent_at_utc": row["sent_at_utc"],
                "duplicate_suppressed": True,
            }

    # Retry demo: fail the first N attempts. `activity.info().attempt` is durable server-side state,
    # so the count keeps rising across worker restarts — the retries are real, not a local counter.
    if info.attempt <= rem.fail_first:
        raise ConnectionError(
            f"simulated gateway failure {info.attempt}/{rem.fail_first} for {rem.person} "
            f"({rem.lunar_label}): SMTP 421, try again later"
        )

    sent_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    message_id = f"<{rem.workflow_id}-c{rem.cycle}@so-gio.local>"
    _append_jsonl(
        outbox,
        {
            "key": key,
            "kind": "gio-reminder",
            "workflow_id": rem.workflow_id,
            "cycle": rem.cycle,
            "person": rem.person,
            "lunar": rem.lunar_label,
            "solar_date": rem.solar_date,
            "scheduled_for_local": rem.fire_at_local,
            "sent_at_utc": sent_at,
            "attempt": info.attempt,
            "message_id": message_id,
            "body_vi": rem.body_vi,
            "channel": "file-outbox (demo); production: Zalo ZNS / SMS",
            "worker_pid": os.getpid(),
            "worker_host": socket.gethostname(),
            "activity_id": info.activity_id,
        },
    )
    return {
        "message_id": message_id,
        "attempt": info.attempt,
        "worker_pid": os.getpid(),
        "sent_at_utc": sent_at,
        "duplicate_suppressed": False,
    }


@activity.defn
async def record_delivery(delivery: Delivery) -> dict:
    """Write the immutable `delivered` ledger row for a cycle, once.

    The ledger is what the family page reads ("đã nhắc ngày giỗ 2027"), and it is keyed by
    workflow_id#cycle so a retried activity cannot double-count a year.
    """
    ledger = state_dir() / "delivered.jsonl"
    key = f"{delivery.workflow_id}#{delivery.cycle}"

    for row in _read_jsonl(ledger):
        if row.get("key") == key:
            return {"recorded": False, "duplicate": True, "key": key}

    _append_jsonl(
        ledger,
        {
            "key": key,
            "workflow_id": delivery.workflow_id,
            "person": delivery.person,
            "cycle": delivery.cycle,
            "lunar": delivery.lunar_label,
            "solar_date": delivery.solar_date,
            "message_id": delivery.message_id,
            "attempts_used": delivery.attempts_used,
            "sent_at_utc": delivery.sent_at_utc,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "recorded_by_pid": os.getpid(),
        },
    )
    return {"recorded": True, "duplicate": False, "key": key}
