"""Worker entrypoint: `python -m workflows.worker`.

Run it from the repository root, or set `TEMPORAL_ADDRESS` / `TEMPORAL_NAMESPACE` to point at
somewhere other than a local dev server. The worker holds no state that matters — that is the
whole design — so it can be killed, restarted, or run three times at once.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporalio.client import Client  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from workflows.activities import record_delivery, send_reminder  # noqa: E402
from workflows.gio import TASK_QUEUE, GioWorkflow  # noqa: E402


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("GIO_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    client = await Client.connect(address, namespace=namespace)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[GioWorkflow],
        activities=[send_reminder, record_delivery],
        max_cached_workflows=200,
        # Fail fast instead of silently holding the queue when the server goes away.
        max_concurrent_workflow_tasks=50,
        max_concurrent_activities=40,
    )
    print(
        f"worker ready: pid={os.getpid()} task_queue={TASK_QUEUE!r} address={address} "
        f"workflows=[GioWorkflow] activities=[send_reminder, record_delivery]",
        flush=True,
    )
    await worker.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
