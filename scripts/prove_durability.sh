#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Proof that a giỗ reminder survives a hard kill of the worker that owns it.
#
#   ./scripts/prove_durability.sh            # transcript on stdout
#   KEEP_SERVER=1 ./scripts/prove_durability.sh
#
# Sequence (all against a real Temporal dev server, not a time-skipping test):
#   1. reuse the dev server on :7233, or start one and stop it again at the end
#   2. start a worker, start GioWorkflow with wait_seconds=20 (demo pacing only)
#   3. kill -9 the worker while the durable timer is still pending
#   4. restart the worker under a NEW pid
#   5. show the timer firing, the retried activity, and the reminder written by the new pid
#
# Re-runnable: the workflow id is unique per run, the state files are per-run, and the EXIT trap
# terminates the workflow, kills the worker and (if it started it) stops the dev server.
# ---------------------------------------------------------------------------
set -Eeuo pipefail
export LC_ALL=C.UTF-8

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-/home/kiter/.pyenv/versions/3.12.6/bin/python}
TEMPORAL=${TEMPORAL_CLI:-$HOME/.temporalio/bin/temporal}
STATE_DIR=${GIO_STATE_DIR:-$ROOT/scripts/state}
TASK_QUEUE=${TASK_QUEUE:-gio-reminders}

WAIT_SECONDS=${WAIT_SECONDS:-20}   # the demo's "one year" — the timer machinery is identical
ACK_WINDOW=${ACK_WINDOW:-180}
FAIL_FIRST=${FAIL_FIRST:-2}        # intentional activity failures, to show the retry policy
MAX_CYCLES=${MAX_CYCLES:-2}        # 1 = the reminder we killed for, 2 = proof it re-armed
PERSON=${PERSON:-"Cụ Nguyễn Thị Đào"}
LUNAR_MONTH=${LUNAR_MONTH:-3}
LUNAR_DAY=${LUNAR_DAY:-10}
WF_ID=${WF_ID:-gio-proof-$(date +%Y%m%d-%H%M%S)}

export GIO_STATE_DIR="$STATE_DIR"
mkdir -p "$STATE_DIR"

SERVER_PID=""
WORKER_PID=""
STARTED_SERVER=0

say() { printf '\n=== %s ===\n' "$*"; }
ts() { date '+%H:%M:%S'; }
note() { printf '[%s] %s\n' "$(ts)" "$*"; }

cleanup() {
  local code=$?
  set +e
  if [[ -n "$WORKER_PID" ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
    kill -9 "$WORKER_PID" 2>/dev/null
    wait "$WORKER_PID" 2>/dev/null || true
    note "cleanup: killed worker pid $WORKER_PID"
  fi
  "$TEMPORAL" workflow terminate -w "$WF_ID" --reason "proof run finished (cleanup)" >/dev/null 2>&1
  if [[ "$STARTED_SERVER" == "1" && -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    note "cleanup: stopped the dev server this script started (pid $SERVER_PID)"
  fi
  exit $code
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- 0. environment
say "0. environment"
echo "repo              $ROOT"
echo "python            $($PY --version 2>&1)  ($PY)"
echo "temporal cli      $("$TEMPORAL" --version 2>&1 | head -1)"
echo "wall clock        $(date -Is)"
echo "workflow id       $WF_ID"
echo "task queue        $TASK_QUEUE"
echo "demo pacing       wait_seconds=${WAIT_SECONDS}s, fail_first=${FAIL_FIRST} (real failures), max_cycles=${MAX_CYCLES}"
echo "person / giỗ      ${PERSON} — ${LUNAR_DAY}/${LUNAR_MONTH} âm lịch"

# --------------------------------------------------------------------------- 1. dev server
say "1. dev server on :7233"
if (exec 3<>/dev/tcp/127.0.0.1/7233) 2>/dev/null; then
  echo "already listening — reusing it (this script will leave it running)"
else
  echo "not listening — starting: temporal server start-dev --headless --port 7233"
  "$TEMPORAL" server start-dev --headless --port 7233 --log-level error \
    > "$STATE_DIR/temporal-server.log" 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1
  for _ in $(seq 1 150); do
    (exec 3<>/dev/tcp/127.0.0.1/7233) 2>/dev/null && break
    sleep 0.1
  done
  if ! (exec 3<>/dev/tcp/127.0.0.1/7233) 2>/dev/null; then
    echo "FATAL: dev server did not come up; last log lines:"
    tail -20 "$STATE_DIR/temporal-server.log" || true
    exit 1
  fi
  echo "dev server pid $SERVER_PID (stopped again on exit)"
fi
"$TEMPORAL" operator cluster health

# --------------------------------------------------------------------------- worker helpers
start_worker() {
  local tag=$1
  ( cd "$ROOT" && exec "$PY" -m workflows.worker ) > "$STATE_DIR/worker-$tag.log" 2>&1 &
  WORKER_PID=$!
  for _ in $(seq 1 150); do
    if grep -q "worker ready" "$STATE_DIR/worker-$tag.log" 2>/dev/null; then
      printf 'worker[%s] pid=%s — %s\n' "$tag" "$WORKER_PID" \
        "$(grep -m1 'worker ready' "$STATE_DIR/worker-$tag.log")"
      printf 'worker[%s] cmd   %s\n' "$tag" "$(ps -o cmd= -p "$WORKER_PID")"
      # registered poller, as seen by the server (identity is "<pid>@<host>").
      # Only this pid's lines: the server keeps dead pollers listed for a few minutes, and printing
      # them would imply the killed worker is still polling.
      for _ in $(seq 1 100); do
        if "$TEMPORAL" task-queue describe --task-queue "$TASK_QUEUE" 2>/dev/null | grep -q "${WORKER_PID}@"; then
          "$TEMPORAL" task-queue describe --task-queue "$TASK_QUEUE" \
            | grep -E "Identity|UNVERSIONED +[a-z]+ +${WORKER_PID}@" | sed 's/^/    /'
          return 0
        fi
        sleep 0.1
      done
      echo "worker[$tag] never showed up as a poller"; return 1
    fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
      echo "worker[$tag] died during startup:"; tail -30 "$STATE_DIR/worker-$tag.log"; return 1
    fi
    sleep 0.1
  done
  echo "worker[$tag] did not report ready"; tail -30 "$STATE_DIR/worker-$tag.log"; return 1
}

status_line() {
  "$TEMPORAL" workflow list --query "WorkflowId='$WF_ID'" 2>/dev/null | sed 's/^/  /'
}

describe_brief() {
  "$TEMPORAL" workflow describe -w "$WF_ID" 2>/dev/null \
    | grep -E "WorkflowId|Type |TaskQueue|Status|RunTime|HistoryLength" \
    | sed 's/^  //' || true
}

# --------------------------------------------------------------------------- 2. worker + workflow
say "2. start the worker"
rm -f "$STATE_DIR/reminders.jsonl" "$STATE_DIR/delivered.jsonl"
start_worker 1
KILLED_PID=$WORKER_PID

say "3. start GioWorkflow (durable timer pending, nothing sent yet)"
"$PY" "$ROOT/scripts/gio_demo.py" start \
  --workflow-id "$WF_ID" --person "$PERSON" --month "$LUNAR_MONTH" --day "$LUNAR_DAY" \
  --wait-seconds "$WAIT_SECONDS" --ack-window "$ACK_WINDOW" --max-cycles "$MAX_CYCLES" \
  --fail-first "$FAIL_FIRST"

sleep 5
note "5s in: the workflow is asleep on a server-side timer"
status_line
describe_brief
"$TEMPORAL" workflow show -w "$WF_ID" > "$STATE_DIR/history-1-armed.txt"
echo "--- history so far (TimerStarted must exist, TimerFired must not) ---"
sed -n '1,12p' "$STATE_DIR/history-1-armed.txt"
grep -q "TimerStarted" "$STATE_DIR/history-1-armed.txt" || { echo "FATAL: no TimerStarted"; exit 1; }
if grep -q "TimerFired" "$STATE_DIR/history-1-armed.txt"; then
  echo "FATAL: the timer already fired before the kill; shorten the wait or kill sooner"; exit 1
fi

# --------------------------------------------------------------------------- 3. kill -9
say "4. kill -9 the worker (pid $KILLED_PID) mid-sleep"
note "kill -9 $KILLED_PID"
kill -9 "$KILLED_PID"
wait "$KILLED_PID" 2>/dev/null || true
WORKER_PID=""
sleep 0.5
if kill -0 "$KILLED_PID" 2>/dev/null; then
  echo "FATAL: pid $KILLED_PID survived kill -9"; exit 1
fi
printf 'ps -p %s -> %s\n' "$KILLED_PID" "$(ps -o pid=,stat=,cmd= -p "$KILLED_PID" 2>/dev/null || echo 'no such process (gone)')"
printf 'worker log tail (its last words before dying):\n'
tail -3 "$STATE_DIR/worker-1.log" | sed 's/^/    /' || true

note "no worker is polling; the deadline is only in the server's history now"
status_line
describe_brief
"$TEMPORAL" workflow show -w "$WF_ID" > "$STATE_DIR/history-2-dead.txt"
if grep -q "ActivityTaskScheduled" "$STATE_DIR/history-2-dead.txt"; then
  echo "FATAL: an activity was scheduled before the kill; the timer was not what was pending"; exit 1
fi
echo "no ActivityTaskScheduled yet — the workflow is asleep on a timer, nothing is queued to a worker"

# let real time pass with no worker alive, so the timer provably expires while nobody is watching
sleep 3
note "3s with zero workers. History still ends at TimerStarted:"
tail -4 "$STATE_DIR/history-2-dead.txt"

# --------------------------------------------------------------------------- 4. restart worker
say "5. restart the worker under a new pid"
start_worker 2
NEW_PID=$WORKER_PID
if [[ "$NEW_PID" == "$KILLED_PID" ]]; then echo "FATAL: pid was reused; proof would be meaningless"; exit 1; fi
note "worker restarted: $KILLED_PID -> $NEW_PID"

# --------------------------------------------------------------------------- 5. the timer fires
say "6. wait for the durable timer to fire and the flaky activity to succeed"
"$PY" "$ROOT/scripts/gio_demo.py" wait-delivered --workflow-id "$WF_ID" --count 1 --timeout 120

say "7. acknowledge the reminder (signal) — the workflow then re-arms for next year"
"$PY" "$ROOT/scripts/gio_demo.py" ack --workflow-id "$WF_ID" --note "Con cháu đã chuẩn bị mâm cúng"
"$PY" "$ROOT/scripts/gio_demo.py" wait-delivered --workflow-id "$WF_ID" --count 2 --timeout 120
"$PY" "$ROOT/scripts/gio_demo.py" ack --workflow-id "$WF_ID" --note "Xong, hẹn năm sau"

say "8. workflow result"
"$PY" "$ROOT/scripts/gio_demo.py" result --workflow-id "$WF_ID" --timeout 60

# --------------------------------------------------------------------------- 6. evidence
"$TEMPORAL" workflow show -w "$WF_ID" > "$STATE_DIR/history-3-final.txt"

say "9. final status"
status_line
describe_brief

say "10. full event history (Progress view)"
cat "$STATE_DIR/history-3-final.txt"

say "11. the timer, across the kill"
grep -E "TimerStarted|TimerFired|WorkflowExecutionSignaled|WorkflowExecutionCompleted|ActivityTaskScheduled|ActivityTaskCompleted" "$STATE_DIR/history-3-final.txt" | sed 's/  */ /g'

say "12. the flaky activity: attempts, backoff and which worker ran them"
"$TEMPORAL" workflow show -w "$WF_ID" --output json > "$STATE_DIR/history-3-final.json"
"$PY" - "$STATE_DIR/history-3-final.json" <<'PY'
import json
import sys
from datetime import datetime

history = json.load(open(sys.argv[1], encoding="utf-8"))
scheduled: dict[str, tuple[str, datetime]] = {}
for event in history["events"]:
    if event["eventType"] == "EVENT_TYPE_ACTIVITY_TASK_SCHEDULED":
        attrs = event.get("activityTaskScheduledEventAttributes", {})
        scheduled[str(event["eventId"])] = (
            attrs.get("activityType", {}).get("name", ""),
            datetime.fromisoformat(event["eventTime"].replace("Z", "+00:00")),
        )

print(f"{'id':>4}  {'time(UTC)':<10} {'event':<22} {'activity':<15} {'attempt':>7}  identity")
for event in history["events"]:
    kind = event["eventType"].removeprefix("EVENT_TYPE_")
    if not kind.startswith("ACTIVITY_TASK_"):
        continue
    attrs = next((v for k, v in event.items() if k.endswith("EventAttributes") and isinstance(v, dict)), {})
    name = attrs.get("activityType", {}).get("name", "")
    when = datetime.fromisoformat(event["eventTime"].replace("Z", "+00:00"))
    if not name:
        name, _ = scheduled.get(str(attrs.get("scheduledEventId")), ("", when))
    if name != "send_reminder":
        continue
    print(f"{event['eventId']:>4}  {event['eventTime'][11:19]:<10} {kind.title().replace('_',''):<22} {name:<15} {str(attrs.get('attempt','-')):>7}  {attrs.get('identity','')}")
    if kind == "ACTIVITY_TASK_STARTED":
        _, sched_at = scheduled.get(str(attrs.get("scheduledEventId")), ("", when))
        gap = (when - sched_at).total_seconds()
        failure = attrs.get("lastFailure") or {}
        extra = f" — {failure.get('applicationFailureInfo', {}).get('type', '?')}: {failure['message']}" if failure.get("message") else ""
        print(f"      └─ attempt {attrs.get('attempt')} started {gap:+.1f}s after scheduling (retry backoff){extra}")
PY

say "13. artifacts written to disk (the 'sent' reminder and the delivered ledger)"
echo "--- $STATE_DIR/reminders.jsonl ---"
"$PY" -c "import json,sys;[print(json.dumps(json.loads(l),ensure_ascii=False,indent=2)) for l in open(sys.argv[1],encoding='utf-8') if l.strip()]" "$STATE_DIR/reminders.jsonl"
echo "--- $STATE_DIR/delivered.jsonl ---"
"$PY" -c "import json,sys;[print(json.dumps(json.loads(l),ensure_ascii=False,indent=2)) for l in open(sys.argv[1],encoding='utf-8') if l.strip()]" "$STATE_DIR/delivered.jsonl"

say "14. verdict"
fail=0
if ! grep -q "TimerFired" "$STATE_DIR/history-3-final.txt"; then
  echo "FAIL: no TimerFired in history — the timer did not survive"; fail=1
fi
if ! grep -q "ActivityTaskCompleted" "$STATE_DIR/history-3-final.txt"; then
  echo "FAIL: no activity ever completed — nothing was sent"; fail=1
fi
if ! grep -q "\"worker_pid\": $NEW_PID" "$STATE_DIR/reminders.jsonl"; then
  echo "FAIL: no reminder was written by the restarted worker (pid $NEW_PID)"; fail=1
fi
if grep -q "\"worker_pid\": $KILLED_PID" "$STATE_DIR/reminders.jsonl"; then
  echo "FAIL: a reminder was written by the killed worker (pid $KILLED_PID)"; fail=1
fi
[[ "$fail" == "0" ]] || exit 1

echo "killed worker pid $KILLED_PID with SIGKILL while the timer was pending;"
echo "a different process (pid $NEW_PID) woke up and delivered the reminder:"
grep -h '"worker_pid"' "$STATE_DIR/reminders.jsonl" | sed 's/^/    /'
echo "PASS — the timer lived in the server, not in the killed process."
