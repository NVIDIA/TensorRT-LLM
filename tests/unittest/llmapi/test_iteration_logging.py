# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise logging with real IPC in killable processes, including shutdown hangs."""

import ast
import asyncio
import multiprocessing
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import zmq
import zmq.asyncio

from tensorrt_llm.bench.benchmark.utils import asynchronous, processes

pytestmark = pytest.mark.cpu_only


class _Stats:
    def __init__(self, count: int) -> None:
        self.remaining = count
        self.sent = asyncio.Event()

    async def get_stats_async(self, timeout: int):
        while self.remaining:
            self.remaining -= 1
            yield {"iteration": self.remaining}
        self.sent.set()
        await asyncio.sleep(0)


async def _produce(address: str, count: int, request_seen: bool = True) -> None:
    manager = asynchronous.LlmManager.__new__(asynchronous.LlmManager)
    manager.llm = _Stats(count)
    manager._stop = asyncio.Event()
    manager.request_seen = asyncio.Event()
    if request_seen:
        manager.request_seen.set()
    manager._backend_task = asyncio.create_task(asyncio.sleep(0))
    manager._iteration_log_task = asyncio.create_task(manager.iteration_worker(address))
    # Give the producer time to reach a blocked send, if the consumer is absent.
    await asyncio.sleep(0.2)
    await manager.stop()
    assert manager._backend_task.done()
    assert manager._iteration_log_task.done()


def _scenario(case: str, directory: str) -> None:
    root = Path(directory)
    # Shorten drain/join deadlines, but retain real sockets and finite linger.
    asynchronous._ITERATION_LOG_DRAIN_TIMEOUT = 0.5
    processes._ITERATION_WRITER_JOIN_TIMEOUT = 1.0
    if case in ("dead_small", "dead_full", "no_requests"):
        count = 2000 if case == "dead_full" else 5
        asyncio.run(_produce(f"ipc://{root / 'absent.sock'}", count, case != "no_requests"))
    elif case in ("healthy", "missing_parent", "empty", "body_error"):
        log = (
            root / "nested" / "iterations.log"
            if case == "missing_parent"
            else root / "iterations.log"
        )
        writer = processes.IterationWriter(log)
        if case == "body_error":
            with pytest.raises(ValueError, match="benchmark failed"):
                with writer.capture():
                    raise ValueError("benchmark failed")
        else:
            with writer.capture():
                if case != "empty":
                    asyncio.run(_produce(writer.full_address, 2000))
            expected = [] if case == "empty" else [{"iteration": i} for i in reversed(range(2000))]
            assert [ast.literal_eval(line) for line in log.read_text().splitlines()] == expected
    elif case == "disabled":
        writer = processes.IterationWriter()
        with writer.capture():
            assert writer.full_address is None
    elif case == "stalled_writer":
        with patch.object(
            processes.IterationWriter, "run", side_effect=lambda *args: time.sleep(60)
        ):
            with patch.object(processes.logger, "warning") as warning:
                with processes.IterationWriter(root / "iterations.log").capture():
                    pass
                warning.assert_called_once()
                assert "timed out" in warning.call_args.args[0]
    elif case == "failed_writer":
        with patch.object(processes.IterationWriter, "run", side_effect=OSError("disk failed")):
            with patch.object(processes.logger, "warning") as warning:
                with processes.IterationWriter(root / "iterations.log").capture():
                    pass
                warning.assert_called_once()
                assert "failed" in warning.call_args.args[0]
    elif case == "invalid_parent":
        parent = root / "not_a_directory"
        parent.write_text("existing file")
        with pytest.raises(FileExistsError):
            with processes.IterationWriter(parent / "iterations.log").capture():
                pytest.fail("invalid path entered benchmark body")
        assert parent.read_text() == "existing file"
    elif case == "invalid_file":
        with pytest.raises(IsADirectoryError):
            with processes.IterationWriter(root).capture():
                pytest.fail("invalid path entered benchmark body")
    elif case == "shared_context":

        async def check() -> None:
            context = zmq.asyncio.Context.instance()
            unrelated = context.socket(zmq.PULL)
            unrelated.bind(f"ipc://{root / 'unrelated.sock'}")
            try:
                await _produce(f"ipc://{root / 'absent.sock'}", 5)
                assert not context.closed
                assert not unrelated.closed
            finally:
                unrelated.close(linger=0)
                context.term()

        asyncio.run(check())
    elif case == "setup_error":

        async def check() -> None:
            manager = asynchronous.LlmManager.__new__(asynchronous.LlmManager)
            with patch.object(asynchronous, "Context", side_effect=OSError("setup failed")):
                with pytest.raises(OSError, match="setup failed"):
                    await manager.iteration_worker("ipc://unused")

        asyncio.run(check())
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "dead_small",
        "dead_full",
        "no_requests",
        "healthy",
        "missing_parent",
        "empty",
        "body_error",
        "invalid_parent",
        "invalid_file",
        "shared_context",
        "setup_error",
        "disabled",
        "stalled_writer",
        "failed_writer",
    ],
)
def test_iteration_logging_shutdown(case: str, tmp_path: Path) -> None:
    # fork matches trtllm-bench and keeps each hang outside pytest's event loop.
    process = multiprocessing.get_context("fork").Process(
        target=_scenario, args=(case, str(tmp_path))
    )
    process.start()
    try:
        process.join(timeout=8)
        assert not process.is_alive(), f"iteration logging hung: {case}"
        assert process.exitcode == 0, f"iteration logging failed: {case}"
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=3)
        process.close()
