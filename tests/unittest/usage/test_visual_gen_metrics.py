# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded VisualGen collection, privacy, and final-snapshot coverage."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from tensorrt_llm.usage import schema, usage_lib
from tensorrt_llm.usage.visual_gen import (
    LATENCY_BOUNDS,
    UINT_MAX,
    VisualGenMetrics,
    VisualGenTelemetryMiddleware,
    record,
    request_shape,
    supplied_extra_keys,
    track_submission_errors,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def session(monkeypatch, enable_telemetry):
    monkeypatch.setattr(usage_lib, "_SESSION", None)
    monkeypatch.setattr(usage_lib, "_SESSION_DISABLED", False)
    monkeypatch.setattr(usage_lib, "_REPORTER_STARTED", False)
    monkeypatch.setattr(usage_lib, "_REPORTER_ACTIVE", False)
    monkeypatch.setattr(usage_lib, "_REPORTER_STOP", threading.Event())
    monkeypatch.setattr(usage_lib, "_PENDING_TERMINAL", None)
    assert usage_lib.record_visual_gen_initialization_attempt()
    assert usage_lib.record_visual_gen_initialized()
    return usage_lib._SESSION


def test_request_histograms_and_percentiles_retain_no_content():
    params = SimpleNamespace(
        width=777,
        height=1000,
        num_frames=33,
        num_inference_steps=29,
        image_reference=["/customer/private-image"],
        video_reference=None,
        extra_params={"stg_scale": "private-value", "customer-key": "secret"},
        prompt="customer prompt",
        negative_prompt="customer negative prompt",
        seed=7654321,
    )
    metrics = VisualGenMetrics()
    shape = request_shape(params, 2, supplied_extra_keys(params))
    for duration in range(1, 101):
        metrics.request("video")
        metrics.complete(shape, {"generation": duration, "denoise": duration / 2}, None)
    metrics.complete(shape, {"generation": 10000}, "private-exception-message")
    metrics.queue(7, 2)
    metrics.queue(3, 1)
    snapshot = metrics.snapshot()
    summary = json.loads(snapshot["visualGenMetricsJson"])
    assert summary["requestsByModality"] == {"video": 100}
    assert summary["resolution"] == {"le_1024": 101}
    assert summary["numFrames"] == {"le_64": 101}
    assert summary["numInferenceSteps"] == {"le_30": 101}
    assert summary["batchSize"] == {"le_2": 101}
    assert summary["inputReferenceKind"] == {"image": 101}
    assert summary["extraParamsKeysUsed"] == {"stg_scale": 101}
    assert summary["errors"] == {"unclassified": 1}
    latency = summary["latencySec"]["generation"]
    assert latency["count"] == 100
    assert 50 <= latency["p50"] <= 55
    assert 95 <= latency["p95"] <= 104.5
    assert snapshot["peakNumQueuedRequests"] == 7
    assert snapshot["peakNumActiveRequests"] == 2
    assert all(len(bins) == len(LATENCY_BOUNDS) + 1 for bins in metrics.latencies.values())
    serialized = json.dumps(snapshot)
    for forbidden in ("customer", "private", "secret", "7654321", "777", "1000"):
        assert forbidden not in serialized


def test_invalid_values_and_saturation_stay_bounded():
    metrics = VisualGenMetrics()
    metrics.request("private-modality")
    metrics.counters["requestsByModality"]["unknown"] = UINT_MAX
    metrics.request("private-modality")
    metrics.complete(
        {"resolution": "/private/path", "extraParamsKeysUsed": ["customer-key"]},
        {"generation": float("inf"), "denoise": float("nan"), "pre_denoise": -1},
        None,
    )
    summary = json.loads(metrics.snapshot()["visualGenMetricsJson"])
    assert summary["requestsByModality"] == {"unknown": UINT_MAX}
    assert "resolution" not in summary
    assert summary["latencySec"] == {}
    assert len(metrics.snapshot()["visualGenMetricsJson"]) < 8192


def test_endpoint_allowlist_and_sync_alias(session):
    called = []

    async def app(scope, receive, send):
        called.append(scope["path"])

    middleware = VisualGenTelemetryMiddleware(app)
    for path in (
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/videos",
        "/v1/videos/sync",
        "/v1/videos/generations",
        "/v1/videos/private-id",
    ):
        asyncio.run(middleware({"type": "http", "method": "POST", "path": path}, None, None))
    asyncio.run(middleware({"type": "http", "method": "GET", "path": "/v1/videos"}, None, None))
    summary = json.loads(session.snapshot()["visualGenMetricsJson"])
    assert summary["endpointRequests"] == {
        "/v1/images/generations": 1,
        "/v1/images/edits": 1,
        "/v1/videos": 1,
        "/v1/videos/generations": 2,
    }
    assert len(called) == 7


def test_short_session_exit_contains_final_cumulative_metrics(session, monkeypatch):
    sent = []
    monkeypatch.setattr(usage_lib, "_send_to_gxt", sent.append)
    record("request", "image")
    record("complete", {"resolution": "le_1024"}, {"generation": 2.0}, None)
    heartbeat = schema.TrtllmVisualGenHeartbeat(
        seq=0, **usage_lib._visual_gen_session_event_fields(session.snapshot())
    )
    record("request", "video")
    record("error", "timeout")
    assert usage_lib.report_exit(
        usage_lib.TerminalOutcome(
            termination_kind="clean",
            component="visual_gen",
            exit_code_known=True,
            exit_code=0,
        )
    )
    final = sent[0]["events"][0]["parameters"]
    assert json.loads(heartbeat.visual_gen_metrics_json)["requestsByModality"] == {"image": 1}
    assert json.loads(final["visualGenMetricsJson"])["requestsByModality"] == {
        "image": 1,
        "video": 1,
    }
    assert json.loads(final["visualGenMetricsJson"])["errors"] == {"timeout": 1}
    assert final["sessionDurationSec"] >= 0
    record("request", "video")
    assert session.snapshot()["visualGenMetricsJson"] == final["visualGenMetricsJson"]


def test_metrics_optout_and_fail_silent(session, monkeypatch):
    monkeypatch.setenv("TRTLLM_NO_USAGE_STATS", "1")
    record("request", "image")
    assert session.visual_gen_metrics.counters == {}
    monkeypatch.delenv("TRTLLM_NO_USAGE_STATS")
    session.disabled = True
    record("request", "image")
    assert session.visual_gen_metrics.counters == {}
    session.disabled = False

    def broken(*args):
        raise RuntimeError("optional telemetry failed")

    monkeypatch.setattr(session.visual_gen_metrics, "request", broken)
    record("request", "image")


def test_submission_errors_preserve_exceptions_without_their_details(session):
    @track_submission_errors
    def submit(error):
        raise error

    for error in (
        ValueError("private content"),
        MemoryError("private path"),
        RuntimeError("private exception"),
    ):
        with pytest.raises(type(error)) as raised:
            submit(error)
        assert raised.value is error
    assert session.visual_gen_metrics.counters["errors"] == {
        "client": 1,
        "capacity": 1,
        "unclassified": 1,
    }


def test_static_fields_follow_explicit_inventory():
    args = SimpleNamespace(
        parallel_config=SimpleNamespace(
            cfg_size=2,
            ulysses_size=2,
            ring_size=1,
            attn2d_size=(2, 1),
            tp_size=4,
            parallel_vae_size=2,
            parallel_vae_split_dim="height",
        ),
        attention_config=SimpleNamespace(
            backend="CUDNN",
            sparse_attention_config=SimpleNamespace(
                algorithm="vsa", vsa_sparsity=0.62, target_sparsity=0.82
            ),
            quant_attention_config=SimpleNamespace(
                qk_dtype="fp8",
                v_dtype="nvfp4",
                q_block_size=128,
                k_block_size=64,
                v_block_size=32,
                clamp_val=1.23456,
            ),
        ),
        cache_config=SimpleNamespace(cache_backend="teacache", coefficients=[0.123456]),
        cuda_graph_config=SimpleNamespace(enable=True),
        torch_compile_config=SimpleNamespace(
            enable=True,
            enable_fullgraph=True,
            enable_autotune=False,
            compilation_resolutions=[(777, 999)],
        ),
        model="/private/model",
        private_label="customer",
    )
    fields = usage_lib._visual_gen_initial_fields(args, {})
    event = schema.TrtllmVisualGenInitialReport(**fields)
    assert fields["attentionBackend"] == "other"
    assert fields["vsaSparsity"] == "le_0.75"
    assert fields["targetSparsity"] == "le_1.0"
    assert fields["qkDtype"] == "fp8" and fields["qBlockSize"] == 128
    assert fields["torchCompileEnable"] and fields["enableFullgraph"]
    assert not fields["enableAutotune"]
    assert fields["cudaGraphEnable"]
    assert json.loads(fields["featuresJson"]) == {
        "parallelVae": True,
        "sparseAttention": True,
        "quantAttention": True,
        "quantizedWeights": False,
    }
    payload = schema.build_gxt_payload(event, session_id="random-session", trtllm_version="1.3.0")
    serialized = json.dumps(payload)
    for forbidden in (
        "private",
        "customer",
        "coefficients",
        "clamp",
        "compilation_resolutions",
        "visualGenConfigJson",
        "cudaGraphs",
    ):
        assert forbidden not in serialized
