# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _torch.pyexecutor.engine.runners.utils import _config, _make_runner, _model

from tensorrt_llm._torch.pyexecutor.engine.runners.interface import PreparedInputs, ScheduledInputs
from tensorrt_llm._torch.pyexecutor.engine.runners.pooling import PoolingRunner
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only


def test_pooling_runner_owns_forward_output_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.arange(12).reshape(3, 4)
    model_forward = Mock(return_value=logits)
    runner = _make_runner(
        PoolingRunner,
        _model(is_generation=False),
        model_caller=model_forward,
    )
    attn_metadata = SimpleNamespace(on_update_kv_lens=Mock())
    prepared = PreparedInputs(
        {"attn_metadata": attn_metadata, "input_ids": torch.tensor([1])},
    )
    monkeypatch.setattr(runner, "_prepare_inputs", Mock(return_value=prepared))

    outputs = runner.forward(
        ScheduledInputs(batch=ScheduledRequests()),
        resource_manager=SimpleNamespace(get_resource_manager=Mock(return_value=None)),
    )

    torch.testing.assert_close(outputs["logits"], logits)
    attn_metadata.on_update_kv_lens.assert_called_once_with()
    model_forward.assert_called_once_with(
        attn_metadata=attn_metadata,
        input_ids=prepared.kwargs["input_ids"],
        return_context_logits=False,
    )


def test_pooling_runner_returns_raw_model_outputs_when_logits_are_disabled() -> None:
    model_outputs = {"hidden_states": object()}
    model_forward = Mock(return_value=model_outputs)
    runner = _make_runner(
        PoolingRunner,
        _model(is_generation=False),
        replace(_config(), without_logits=True),
        model_caller=model_forward,
    )

    outputs = runner._forward_step({}, SimpleNamespace())

    assert outputs is model_outputs
