"""One durable Temporal workflow per deceased relative: the giỗ that never gets forgotten.

The promise is small and absolute — on the morning of a giỗ, the family is reminded, every year, for
as long as the account exists. A cron job on somebody's laptop cannot make that promise: laptops
sleep, containers are replaced, a `deploy` kills the process that was holding the timer in memory.
A durable timer can, because the deadline lives in the server's event history and the worker is
disposable. This module is that timer.

Design notes a reader should check:

* :func:`next_fire` is the **only** place that decides when a reminder is due. Production (a real
  giỗ, 07:00 Vietnam time, computed from `core.lunar`) and the demo (`wait_seconds`) differ only in
  the timestamp that function returns. Everything downstream — durable timer, activity, retry
  policy, acknowledgement, re-arm — is the production path, exercised unchanged by
  `scripts/prove_durability.sh`.
* Every wait is `workflow.sleep(...)`, i.e. a server-side timer. `asyncio.sleep` would die with the
  process, which is precisely the failure this module exists to rule out.
* The workflow does not exit after one reminder. A giỗ repeats, so the workflow re-arms and waits
  for the next occurrence; that is the whole point of modelling a ritual instead of an alarm.
* Delivery is an activity with a retry policy, because the part that actually reaches a person
  (SMTP, Zalo ZNS, SMS) is the least reliable component in the system.
* `wait_seconds`, `fail_first` and `max_cycles` are demo knobs. They default to production values
  (None / 0 / None) and never change the shape of the execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

# `core.lunar` is pure stdlib arithmetic (no IO, no clock), so it is safe to run inside the workflow
# sandbox; passing it through keeps the workflow deterministic without a second implementation.
with workflow.unsafe.imports_passed_through():
    from core import lunar

TASK_QUEUE = "gio-reminders"

# Vietnam is UTC+7 all year (no DST since 1975). A fixed offset is the honest model here: a tz
# database lookup inside a workflow would be a determinism hazard for no benefit.
VN_TZ = timezone(timedelta(hours=7), "UTC+07")

# 07:00 local: the family is awake, the market is open, the altar can still be prepared today.
REMINDER_HOUR = 7

# Retry the flaky part (the gateway) generously, but not forever: an undelivered reminder that
# retries for a week is worse than one that fails loudly and shows up on the dashboard.
DELIVERY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=8,
)

# Recording a delivery must be idempotent (the activity de-duplicates on workflow_id#cycle), so a
# short retry window is enough.
RECORD_RETRY = RetryPolicy(
    initial_interval=timedelta(milliseconds=200),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=5),
    maximum_attempts=5,
)


# --------------------------------------------------------------------------- data carried around

@dataclass
class GioRequest:
    """What the family tells us once: who, and which lunar day their giỗ falls on."""

    person: str
    lunar_month: int
    lunar_day: int
    leap: bool = False
    # Demo pacing. None = production: sleep until 07:00 on the next real giỗ.
    wait_seconds: float | None = None
    # How long to wait for a family member to acknowledge before re-arming anyway (2 days default).
    ack_window_seconds: float = 2 * 86400.0
    # Demo knob: stop after N cycles so the proof terminates. None = run forever, like a real giỗ.
    max_cycles: int | None = None
    # Demo knob: make the delivery activity fail this many times to show the retry policy working.
    fail_first: int = 0
    note: str = ""


@dataclass
class ReminderTarget:
    """When the next reminder fires, and what the calendar says the real giỗ is."""

    fire_at_utc: datetime
    fire_at_local: datetime
    production_fire_at_local: datetime
    solar_date: str
    lunar_label: str
    demo_override: bool


@dataclass
class Reminder:
    """Activity input: everything the sender needs, no workflow context required."""

    workflow_id: str
    person: str
    cycle: int
    lunar_label: str
    solar_date: str
    fire_at_local: str
    body_vi: str
    fail_first: int = 0


@dataclass
class Delivery:
    """Activity input: proof that a reminder left the building, to be written to the ledger."""

    workflow_id: str
    person: str
    cycle: int
    lunar_label: str
    solar_date: str
    message_id: str
    attempts_used: int
    worker_pid: int
    sent_at_utc: str


# --------------------------------------------------------------------------- calendar → timestamp

def lunar_label(req: GioRequest) -> str:
    return f"{req.lunar_day:02d}/{req.lunar_month:02d}{' nhuận' if req.leap else ''} âm lịch"


def _next_gio_local(req: GioRequest, now_local: datetime) -> datetime:
    """07:00 Vietnam time on the next occurrence of this lunar date, strictly after `now_local`.

    `after` is yesterday, not today, so a giỗ that falls *today* but whose 07:00 reminder has not
    passed yet is still fired today instead of being skipped for a year.
    """
    yesterday = now_local.date() - timedelta(days=1)
    try:
        occurrences = lunar.next_occurrence(
            req.lunar_month,
            req.lunar_day,
            after=(yesterday.year, yesterday.month, yesterday.day),
            years_ahead=5,
            leap=req.leap,
        )
    except Exception as exc:  # a calendar we cannot resolve must fail loudly, never silently skip
        raise ApplicationError(
            f"cannot resolve lunar date {lunar_label(req)}: {exc}", non_retryable=True
        ) from exc

    for year, month, day in occurrences:
        fire_local = datetime(year, month, day, REMINDER_HOUR, 0, tzinfo=VN_TZ)
        if fire_local > now_local:
            return fire_local
    raise ApplicationError(
        f"no occurrence of {lunar_label(req)} in the next 5 years after {now_local.date()}",
        non_retryable=True,
    )


def next_fire(req: GioRequest, now_utc: datetime) -> ReminderTarget:
    """The single decision point: when does the next reminder fire?

    Production returns 07:00 local on the next giỗ. The demo override returns `now + wait_seconds`.
    Only the timestamp differs — the durable timer, activity, retries and re-arm below are identical.
    """
    now_local = now_utc.astimezone(VN_TZ)
    production = _next_gio_local(req, now_local)

    if req.wait_seconds is not None:
        fire_utc = now_utc + timedelta(seconds=req.wait_seconds)
        return ReminderTarget(
            fire_at_utc=fire_utc,
            fire_at_local=fire_utc.astimezone(VN_TZ),
            production_fire_at_local=production,
            solar_date=production.date().isoformat(),
            lunar_label=lunar_label(req),
            demo_override=True,
        )

    return ReminderTarget(
        fire_at_utc=production.astimezone(timezone.utc),
        fire_at_local=production,
        production_fire_at_local=production,
        solar_date=production.date().isoformat(),
        lunar_label=lunar_label(req),
        demo_override=False,
    )


def reminder_body(person: str, label: str, solar_date: str) -> str:
    return (
        f"Hôm nay là {label} — ngày giỗ của {person} ({solar_date}). "
        "Con cháu chuẩn bị hương, hoa và mâm cơm cúng."
    )


# --------------------------------------------------------------------------- the workflow

@workflow.defn
class GioWorkflow:
    """A repeating ritual, durably scheduled. One instance per person.

    Timeline of one cycle:

        compute target → workflow.sleep(until target) → send_reminder (retried) → record_delivery
        → wait for `acknowledge` signal (with timeout) → loop for the next giỗ

    The loop is the product: an annual reminder that fires once is a notification, not a ritual.
    """

    def __init__(self) -> None:
        self._person = ""
        self._label = ""
        self._phase = "starting"
        self._cycle = 0
        self._target: ReminderTarget | None = None
        self._awaiting_ack = False
        self._acked = False
        self._ack_note: str | None = None
        self._deliveries: list[dict[str, Any]] = []
        self._acks: list[dict[str, Any]] = []

    @workflow.run
    async def run(self, req: GioRequest) -> dict[str, Any]:
        self._person = req.person
        self._label = lunar_label(req)
        workflow.logger.info(
            "giỗ workflow for %s (%s) starting on queue %s", req.person, self._label, TASK_QUEUE
        )

        while req.max_cycles is None or self._cycle < req.max_cycles:
            self._cycle += 1
            now = workflow.now()
            target = next_fire(req, now)
            self._target = target
            self._acked = False
            self._awaiting_ack = False
            self._ack_note = None

            delay = (target.fire_at_utc - now).total_seconds()
            self._phase = f"sleeping until cycle {self._cycle} reminder"
            workflow.logger.info(
                "cycle %d: %s → fire %s (real giỗ %s, %.0fs away)%s",
                self._cycle,
                self._label,
                target.fire_at_local.isoformat(timespec="seconds"),
                target.production_fire_at_local.isoformat(timespec="seconds"),
                delay,
                " [demo override]" if target.demo_override else "",
            )

            if delay > 0:
                # A durable timer: the deadline is written to the server's history before we wait,
                # so it survives kill -9 of this worker, a redeploy, or a machine reboot.
                await workflow.sleep(timedelta(seconds=delay))

            self._phase = f"sending cycle {self._cycle} reminder"
            reminder = Reminder(
                workflow_id=workflow.info().workflow_id,
                person=req.person,
                cycle=self._cycle,
                lunar_label=self._label,
                solar_date=target.solar_date,
                fire_at_local=target.fire_at_local.isoformat(timespec="seconds"),
                body_vi=reminder_body(req.person, self._label, target.solar_date),
                fail_first=req.fail_first,
            )
            receipt = await workflow.execute_activity(
                "send_reminder",
                reminder,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=DELIVERY_RETRY,
            )
            receipt = dict(receipt or {})

            delivery = Delivery(
                workflow_id=reminder.workflow_id,
                person=req.person,
                cycle=self._cycle,
                lunar_label=self._label,
                solar_date=target.solar_date,
                message_id=str(receipt.get("message_id", "")),
                attempts_used=int(receipt.get("attempt", 0)),
                worker_pid=int(receipt.get("worker_pid", -1)),
                sent_at_utc=str(receipt.get("sent_at_utc", "")),
            )
            ledger = await workflow.execute_activity(
                "record_delivery",
                delivery,
                start_to_close_timeout=timedelta(seconds=15),
                retry_policy=RECORD_RETRY,
            )

            self._deliveries.append(
                {
                    "cycle": self._cycle,
                    "lunar": self._label,
                    "solar_date": target.solar_date,
                    "scheduled_for_local": target.fire_at_local.isoformat(timespec="seconds"),
                    "production_gio_local": target.production_fire_at_local.isoformat(timespec="seconds"),
                    "message_id": delivery.message_id,
                    "attempts_used": delivery.attempts_used,
                    "worker_pid": delivery.worker_pid,
                    "sent_at_utc": delivery.sent_at_utc,
                    "ledger": dict(ledger or {}),
                }
            )
            workflow.logger.info(
                "cycle %d delivered by pid %s after %d attempt(s): %s",
                self._cycle,
                delivery.worker_pid,
                delivery.attempts_used,
                delivery.message_id,
            )

            self._phase = f"awaiting acknowledgement of cycle {self._cycle}"
            self._awaiting_ack = True
            try:
                await workflow.wait_condition(
                    lambda: self._acked, timeout=timedelta(seconds=req.ack_window_seconds)
                )
                ack = {
                    "cycle": self._cycle,
                    "note": self._ack_note or "",
                    "acked_at_utc": workflow.now().isoformat(timespec="seconds"),
                }
                workflow.logger.info("cycle %d acknowledged: %s", self._cycle, ack["note"])
            except TimeoutError:
                # Silence is information, not a reason to stop: the ritual is owed either way.
                ack = {
                    "cycle": self._cycle,
                    "note": "(no acknowledgement — timed out)",
                    "acked_at_utc": None,
                }
                workflow.logger.warning(
                    "cycle %d not acknowledged within %.0fs; re-arming anyway",
                    self._cycle,
                    req.ack_window_seconds,
                )
            self._acks.append(ack)
            self._awaiting_ack = False
            self._phase = f"re-arming after cycle {self._cycle}"

        self._phase = "completed"
        return {
            "person": self._person,
            "lunar": self._label,
            "cycles_delivered": len(self._deliveries),
            "deliveries": self._deliveries,
            "acks": self._acks,
        }

    # ------------------------------------------------------------------ signals & queries

    @workflow.signal
    def acknowledge(self, note: str = "") -> None:
        """A family member confirms the reminder was seen.

        Signals are persisted in the event history, so an acknowledgement sent while the worker is
        down is delivered when it comes back — no lost messages, no polling.
        """
        self._acked = True
        self._ack_note = note

    @workflow.query
    def state(self) -> dict[str, Any]:
        """Read-only view for the dashboard, the CLI and the durability proof."""
        return {
            "person": self._person,
            "lunar": self._label,
            "phase": self._phase,
            "cycle": self._cycle,
            "delivered_cycles": len(self._deliveries),
            "awaiting_ack": self._awaiting_ack,
            "next_fire_utc": self._target.fire_at_utc.isoformat(timespec="seconds") if self._target else None,
            "next_fire_local": self._target.fire_at_local.isoformat(timespec="seconds") if self._target else None,
            "production_gio_local": self._target.production_fire_at_local.isoformat(timespec="seconds") if self._target else None,
            "demo_override": self._target.demo_override if self._target else None,
            "deliveries": self._deliveries,
            "acks": self._acks,
        }
