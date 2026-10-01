# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise logging with real IPC in killable processes, including shutdown hangs."""

import ast
import asyncio
import multiprocessing
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import zmq
import zmq.asyncio

from tensorrt_llm.bench import benchmark as benchmark_module
from tensorrt_llm.bench.benchmark import low_latency
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
    asynchronous._ITERATION_LOG_DRAIN_TIMEOUT_SEC = 0.5
    processes._ITERATION_WRITER_JOIN_TIMEOUT_SEC = 1.0
    if case in ("dead_small", "dead_full", "no_requests"):
        count = 2000 if case == "dead_full" else 5
        if case == "no_requests":
            with patch.object(asynchronous.logger, "warning") as warning:
                asyncio.run(_produce(f"ipc://{root / 'absent.sock'}", count, False))
            warning.assert_called_once_with(
                "Iteration logging timed out; the iteration log may be incomplete."
            )
        else:
            asyncio.run(_produce(f"ipc://{root / 'absent.sock'}", count))
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
    elif case == "missing_end":
        writer = processes.IterationWriter(root / "iterations.log")
        stop = multiprocessing.Event()
        stop.set()
        with patch.object(processes.logger, "warning") as warning:
            processes.IterationWriter.run(writer.full_address, writer.log_path, stop)
            warning.assert_called_once()
            assert "without receiving the end marker" in warning.call_args.args[0]
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
        "missing_end",
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


@pytest.mark.parametrize("enabled", [True, False])
def test_latency_command_forwards_iteration_log(enabled: bool, tmp_path: Path) -> None:
    """Latency configuration enables iteration statistics only when requested."""
    iteration_log = tmp_path / "iterations.log" if enabled else None
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")

    iteration_writer = MagicMock()
    iteration_writer.capture.return_value = nullcontext()
    iteration_writer.full_address = "ipc://iteration-log" if enabled else None
    options = SimpleNamespace(
        backend="pytorch",
        beam_width=1,
        checkpoint_path=tmp_path,
        concurrency=1,
        dataset_path=dataset,
        duration=None,
        iteration_log=iteration_log,
        iteration_writer=iteration_writer,
        kv_cache_percent=0.5,
        max_input_len=128,
        max_seq_len=128,
        modality=None,
        model="test-model",
        model_type="decoder",
        num_requests=1,
        output_json=None,
        report_json=None,
        request_json=None,
        warmup=0,
    )
    bench_env = SimpleNamespace(
        checkpoint_path=tmp_path,
        model="test-model",
        revision=None,
        telemetry_config=None,
    )
    metadata = MagicMock(max_sequence_length=128)
    tokenizer = MagicMock(eos_token_id=0, pad_token_id=0)
    runtime_config = MagicMock(
        backend="pytorch",
        iteration_log=iteration_log,
    )
    runtime_config.get_llm_args.return_value = {}
    runtime_config.decoding_config.decoding_mode = low_latency.SpeculativeDecodingMode.NONE
    llm = MagicMock(startup_metrics={})
    settings = {
        "settings_config": {
            "max_num_tokens": 128,
        },
        "performance_options": {},
    }

    with (
        patch.object(low_latency, "get_general_cli_options", return_value=options),
        patch.object(low_latency, "initialize_tokenizer", return_value=tokenizer),
        patch.object(low_latency, "create_dataset_from_stream", return_value=(metadata, [])),
        patch.object(low_latency, "get_settings", return_value=settings),
        patch.object(low_latency, "collect_explicit_cli_keys", return_value=set()),
        patch.object(
            low_latency, "RuntimeConfig", return_value=runtime_config
        ) as runtime_config_cls,
        patch.object(benchmark_module, "PyTorchLLM", return_value=llm) as llm_cls,
        patch.object(low_latency, "async_benchmark", new=AsyncMock(return_value=[])),
        patch.object(low_latency, "SamplingParams"),
        patch.object(low_latency, "ReportUtility"),
        patch.object(low_latency, "generate_json_report"),
    ):
        low_latency.latency_command.callback.__wrapped__(bench_env, sampler_options=None)

    assert runtime_config_cls.call_args.kwargs["iteration_log"] == iteration_log
    llm_kwargs = llm_cls.call_args.kwargs
    assert llm_kwargs.get("enable_iter_perf_stats", False) is enabled
