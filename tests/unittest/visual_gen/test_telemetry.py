# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tensorrt_llm.usage as usage
from tensorrt_llm.usage import usage_lib
from tensorrt_llm.visual_gen.args import VisualGenArgs
from tensorrt_llm.visual_gen.visual_gen import VisualGen

pytestmark = pytest.mark.cpu_only


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
        patch("tensorrt_llm.visual_gen.visual_gen.atexit.register"),
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


def test_visual_gen_shutdown_tolerates_missing_telemetry_state():
    visual_gen = VisualGen.__new__(VisualGen)
    visual_gen.executor = _executor()

    visual_gen.shutdown()

    assert visual_gen.executor is None


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
