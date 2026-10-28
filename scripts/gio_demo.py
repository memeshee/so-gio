"""Temporal client used by `scripts/prove_durability.sh`.

It lives apart from the worker on purpose: the proof needs to query and signal the workflow while
*no worker process is alive*, which is only possible because the workflow's state is in the server.

    python scripts/gio_demo.py start  --workflow-id ID --person "Cụ Nguyễn Thị Đào" \
        --month 3 --day 10 --wait-seconds 20 --max-cycles 2 --fail-first 2
    python scripts/gio_demo.py state  --workflow-id ID
    python scripts/gio_demo.py ack    --workflow-id ID --note "..."
    python scripts/gio_demo.py wait-delivered --workflow-id ID --count 1 --timeout 90
    python scripts/gio_demo.py result --workflow-id ID --timeout 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporalio.client import Client  # noqa: E402

from workflows.gio import TASK_QUEUE, GioRequest, GioWorkflow  # noqa: E402

ADDRESS = "localhost:7233"


def emit(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str), flush=True)


def clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


async def connect() -> Client:
    last: Exception | None = None
    for _ in range(100):
        try:
            return await Client.connect(ADDRESS)
        except Exception as exc:  # server may still be binding the port
            last = exc
            await asyncio.sleep(0.2)
    raise SystemExit(f"cannot reach Temporal at {ADDRESS}: {last}")


async def cmd_start(args: argparse.Namespace) -> None:
    client = await connect()
    req = GioRequest(
        person=args.person,
        lunar_month=args.month,
        lunar_day=args.day,
        leap=args.leap,
        wait_seconds=args.wait_seconds,
        ack_window_seconds=args.ack_window,
        max_cycles=args.max_cycles,
        fail_first=args.fail_first,
        note=args.note,
    )
    handle = await client.start_workflow(
        GioWorkflow.run, req, id=args.workflow_id, task_queue=TASK_QUEUE
    )
    print(f"started workflow_id={handle.id} run_id={handle.result_run_id}", flush=True)
    state = await asyncio.wait_for(handle.query(GioWorkflow.state), timeout=30)
    emit(state)


async def cmd_state(args: argparse.Namespace) -> None:
    client = await connect()
    handle = client.get_workflow_handle(args.workflow_id)
    state = await asyncio.wait_for(handle.query(GioWorkflow.state), timeout=args.timeout)
    emit(state)


async def cmd_ack(args: argparse.Namespace) -> None:
    client = await connect()
    handle = client.get_workflow_handle(args.workflow_id)
    await handle.signal(GioWorkflow.acknowledge, args.note)
    print(f"[{clock()}] sent signal acknowledge note={args.note!r}", flush=True)


async def cmd_wait_delivered(args: argparse.Namespace) -> None:
    client = await connect()
    handle = client.get_workflow_handle(args.workflow_id)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + args.timeout
    last: tuple[int, str] | None = None

    while True:
        state = None
        try:
            state = await asyncio.wait_for(handle.query(GioWorkflow.state), timeout=10)
        except Exception as exc:  # no worker yet, or workflow task in flight
            print(f"[{clock()}] query not answered yet ({type(exc).__name__}); retrying", flush=True)
        if state is not None:
            delivered = len(state.get("deliveries") or [])
            key = (delivered, str(state.get("phase")))
            if key != last:
                print(
                    f"[{clock()}] phase={state.get('phase')!r} "
                    f"delivered={delivered} awaiting_ack={state.get('awaiting_ack')}",
                    flush=True,
                )
                last = key
            if delivered >= args.count:
                emit(state)
                return
        if loop.time() > deadline:
            raise SystemExit(
                f"timed out after {args.timeout}s waiting for {args.count} delivery(ies)"
            )
        await asyncio.sleep(0.25)


async def cmd_result(args: argparse.Namespace) -> None:
    client = await connect()
    handle = client.get_workflow_handle(args.workflow_id)
    result = await asyncio.wait_for(handle.result(), timeout=args.timeout)
    print(f"[{clock()}] workflow returned:", flush=True)
    emit(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="start a GioWorkflow and print its first state")
    start.add_argument("--workflow-id", required=True)
    start.add_argument("--person", required=True)
    start.add_argument("--month", type=int, required=True)
    start.add_argument("--day", type=int, required=True)
    start.add_argument("--leap", action="store_true")
    start.add_argument("--wait-seconds", type=float, default=None)
    start.add_argument("--ack-window", type=float, default=180.0)
    start.add_argument("--max-cycles", type=int, default=None)
    start.add_argument("--fail-first", type=int, default=0)
    start.add_argument("--note", default="")
    start.set_defaults(func=cmd_start)

    state = sub.add_parser("state", help="query GioWorkflow.state")
    state.add_argument("--workflow-id", required=True)
    state.add_argument("--timeout", type=float, default=30.0)
    state.set_defaults(func=cmd_state)

    ack = sub.add_parser("ack", help="send the acknowledge signal")
    ack.add_argument("--workflow-id", required=True)
    ack.add_argument("--note", default="")
    ack.set_defaults(func=cmd_ack)

    wait = sub.add_parser("wait-delivered", help="poll until N cycles have been delivered")
    wait.add_argument("--workflow-id", required=True)
    wait.add_argument("--count", type=int, default=1)
    wait.add_argument("--timeout", type=float, default=90.0)
    wait.set_defaults(func=cmd_wait_delivered)

    result = sub.add_parser("result", help="wait for the workflow result")
    result.add_argument("--workflow-id", required=True)
    result.add_argument("--timeout", type=float, default=60.0)
    result.set_defaults(func=cmd_result)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
