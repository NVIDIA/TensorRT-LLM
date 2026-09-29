# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tensorrt_llm.usage as usage
from tensorrt_llm.usage import usage_lib
from tensorrt_llm.visual_gen.args import VisualGenArgs
from tensorrt_llm.visual_gen.visual_gen import VisualGen

pytestmark = pytest.mark.cpu_only


def test_completion_and_timeout_metrics_are_counted_once(monkeypatch):
    from tensorrt_llm._torch.visual_gen import executor
    from tensorrt_llm.usage.visual_gen import VisualGenMetrics

    metrics = VisualGenMetrics()
    monkeypatch.setattr(
        executor, "_record_visual_gen", lambda action, *args: getattr(metrics, action)(*args)
    )

    async def exercise():
        tracker = executor._IterationStatsTracker()
        client = SimpleNamespace(
            lock=asyncio.Lock(),
            pending_requests=queue.Queue(),
            _abandoned_request_ids=set(),
            completed_responses={},
            _iter_stats=tracker,
            response_event=asyncio.Event(),
        )
        tracker.record_telemetry_enqueue()
        tracker.record_request_started(1, 0)
        response = executor.DiffusionResponse(
            request_id=1,
            generation=2.0,
            telemetry_shape={"resolution": "le_1024"},
        )
        await executor.DiffusionRemoteClient._store_response(client, response)
        await executor.DiffusionRemoteClient._store_response(client, response)
        tracker.record_telemetry_enqueue()
        tracker.record_request_started(2, 0)
        await executor.DiffusionRemoteClient.abandon_request_id(client, 2)
        await executor.DiffusionRemoteClient.abandon_request_id(client, 2)
        response.request_id = 2
        await executor.DiffusionRemoteClient._store_response(client, response)
        assert not tracker._active_request_ids

    asyncio.run(exercise())
    assert metrics.counters["resolution"] == {"le_1024": 1}
    assert metrics.counters["errors"] == {"timeout": 1}
    assert metrics.peak_queued == 1
    assert metrics.peak_active == 1
    assert sum(metrics.latencies["generation"]) == 1


def test_worker_collects_resolved_shapes_without_request_content(monkeypatch):
    from tensorrt_llm._torch.visual_gen import executor
    from tensorrt_llm._torch.visual_gen.output import PipelineOutput

    monkeypatch.setattr(executor, "_cuda_memory_logging_enabled", lambda: False)
    monkeypatch.setattr(usage, "is_usage_stats_enabled", lambda disabled=False: not disabled)
    params = SimpleNamespace(
        width=None,
        height=None,
        num_frames=32,
        num_inference_steps=20,
        image_reference=["private"],
        video_reference=None,
    )
    request = executor.DiffusionRequest(
        123,
        ["private prompt"],
        params,
        telemetry_extra_keys=("stg_scale", "private-key"),
    )

    def prepare(req):
        req.params.width, req.params.height = 768, 1024

    worker = SimpleNamespace(
        rank=0,
        device_id=0,
        in_client_process=False,
        response_queue=queue.Queue(),
        visual_gen_args=SimpleNamespace(telemetry_config=SimpleNamespace(disabled=False)),
        _merge_defaults=lambda req: None,
        pipeline=SimpleNamespace(
            prepare_request=prepare,
            request_warmup_cache_key=lambda req: (1024, 768),
            _warmed_up_shapes=set(),
            run_inference=lambda req: PipelineOutput(),
            classify_request_failure=lambda error: None,
        ),
    )
    executor.DiffusionExecutor.process_request(worker, request)
    response = worker.response_queue.get_nowait()
    assert response.error_msg is None
    assert response.telemetry_shape["resolution"] == "le_1024"
    assert response.telemetry_shape["numFrames"] == "le_64"
    assert response.telemetry_shape["inputReferenceKind"] == "image"
    assert response.telemetry_shape["extraParamsKeysUsed"] == ["stg_scale"]
    assert "private" not in str(response.telemetry_shape)
    worker.visual_gen_args.telemetry_config.disabled = True
    executor.DiffusionExecutor.process_request(worker, request)
    assert worker.response_queue.get_nowait().telemetry_shape == {}


def _executor():
    return SimpleNamespace(
        telemetry_metadata={"model_id": "other", "modality": "image"},
        launch_mode="local_spawn",
        node_count=1,
        n_workers=1,
        shutdown=MagicMock(),
    )


def test_visual_gen_reports_initialized_runtime_and_shutdown_once():
    executor = _executor()
    args = VisualGenArgs(model="/private/model")

    with (
        patch(
            "tensorrt_llm.visual_gen.visual_gen.DiffusionRemoteClient",
            return_value=executor,
        ),
        patch.object(usage, "record_visual_gen_initialization_attempt", return_value=True),
        patch.object(usage, "record_visual_gen_initialized", return_value=True),
        patch.object(usage, "record_visual_gen_shutdown") as record_shutdown,
        patch.object(usage_lib, "report_visual_gen_usage") as report_usage,
    ):
        visual_gen = VisualGen(model=args.model, args=args)
        visual_gen.shutdown()
        visual_gen.shutdown()

    metadata = report_usage.call_args.args[1]
    assert metadata["launch_mode"] == "local_spawn"
    assert metadata["node_count"] == 1
    assert metadata["n_workers"] == 1
    assert report_usage.call_args.args[2].usage_context is usage.UsageContext.VISUAL_GEN_CLASS
    executor.shutdown.assert_called_once()
    record_shutdown.assert_called_once()


def test_visual_gen_records_initialization_failure():
    args = VisualGenArgs(model="/private/model")

    with (
        patch(
            "tensorrt_llm.visual_gen.visual_gen.DiffusionRemoteClient",
            side_effect=RuntimeError("worker failed"),
        ),
        patch.object(usage, "record_visual_gen_initialization_attempt", return_value=True),
        patch.object(usage, "record_visual_gen_initialization_failure") as record_failure,
    ):
        with pytest.raises(RuntimeError, match="worker failed"):
            VisualGen(model=args.model, args=args)

    record_failure.assert_called_once()


def test_visual_gen_records_external_world_size_failure():
    args = VisualGenArgs(model="/private/model")

    with (
        patch(
            "tensorrt_llm.visual_gen.visual_gen._detect_external_launch",
            return_value=(0, 0, 2, "localhost", 1234),
        ),
        patch.object(usage, "record_visual_gen_initialization_attempt", return_value=True),
        patch.object(usage, "record_visual_gen_initialization_failure") as record_failure,
        patch("tensorrt_llm.visual_gen.visual_gen.DiffusionRemoteClient") as executor,
    ):
        with pytest.raises(ValueError, match=r"world_size \(2\) does not match n_workers \(1\)"):
            VisualGen(model=args.model, args=args)

    record_failure.assert_called_once()
    executor.assert_not_called()


def test_visual_gen_shutdown_failure_can_be_retried():
    visual_gen = VisualGen.__new__(VisualGen)
    visual_gen.executor = _executor()
    visual_gen.executor.shutdown.side_effect = [RuntimeError("shutdown failed"), None]
    visual_gen._usage_lifecycle_active = True
    visual_gen._usage_lifecycle_lock = threading.Lock()

    with (
        patch.object(usage, "record_visual_gen_shutdown") as record_shutdown,
        pytest.raises(RuntimeError, match="shutdown failed"),
    ):
        visual_gen.shutdown()

    assert visual_gen.executor is not None
    assert visual_gen._usage_lifecycle_active is True
    record_shutdown.assert_not_called()

    with patch.object(usage, "record_visual_gen_shutdown") as record_shutdown:
        visual_gen.shutdown()

    assert visual_gen.executor is None
    record_shutdown.assert_called_once()
