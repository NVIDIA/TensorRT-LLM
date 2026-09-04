# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners import (
    apply_position_id_offset,
    get_padding_params,
    get_top_level_model,
    resolve_runner,
)
from tensorrt_llm._torch.pyexecutor.engine.runners import mm_encoder as mm_encoder_module
from tensorrt_llm._torch.pyexecutor.engine.runners import pooling as pooling_module
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import PreparedInputs, RunnerDeps
from tensorrt_llm._torch.pyexecutor.engine.runners.mm_encoder import MultimodalEncoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.pooling import PoolingRunner
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

pytestmark = pytest.mark.cpu_only


def _deps() -> RunnerDeps:
    return RunnerDeps(
        dist=SimpleNamespace(),
        mapping=SimpleNamespace(),
        enable_attention_dp=False,
        max_num_tokens=16,
        max_seq_len=16,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
        prefill_cuda_graph_num_tokens=[],
        mm_encoder_cache_enabled=False,
        input_ids_cuda=torch.empty(16, dtype=torch.int),
        position_ids_cuda=torch.empty(16, dtype=torch.int),
        gather_ids_cuda=None,
        draft_tokens_cuda=None,
        lora=SimpleNamespace(build=Mock(return_value=None)),
        forward_step=Mock(return_value={"logits": torch.empty(0)}),
    )


def _model(*, is_generation: bool, is_encoder_decoder: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_generation=is_generation,
            is_encoder_decoder=is_encoder_decoder,
        )
    )


@pytest.mark.parametrize(
    ("encode_only", "mm_encoder_only", "is_generation", "is_encoder_decoder", "runner_type"),
    [
        (False, True, False, False, MultimodalEncoderRunner),
        (False, False, False, False, PoolingRunner),
        (True, False, False, False, None),
        (False, False, True, True, None),
        (False, False, True, False, None),
    ],
)
def test_resolve_runner_dispatches_startup_families(
    encode_only: bool,
    mm_encoder_only: bool,
    is_generation: bool,
    is_encoder_decoder: bool,
    runner_type: type[Any] | None,
) -> None:
    args = SimpleNamespace(encode_only=encode_only, mm_encoder_only=mm_encoder_only)

    runner = resolve_runner(
        _deps(),
        _model(is_generation=is_generation, is_encoder_decoder=is_encoder_decoder),
        args,
    )

    if runner_type is None:
        assert runner is None
    else:
        assert isinstance(runner, runner_type)


def test_resolve_runner_checks_encode_only_before_pooling() -> None:
    args = SimpleNamespace(encode_only=True, mm_encoder_only=False)

    assert resolve_runner(_deps(), _model(is_generation=False), args) is None


def test_resolve_runner_checks_mm_encoder_before_non_generation() -> None:
    args = SimpleNamespace(encode_only=False, mm_encoder_only=True)

    assert isinstance(
        resolve_runner(_deps(), _model(is_generation=False), args),
        MultimodalEncoderRunner,
    )


def test_prepared_inputs_is_frozen_and_preserves_kwargs_identity() -> None:
    kwargs = {"input_ids": torch.tensor([1])}
    prepared = PreparedInputs(kwargs)

    assert prepared.kwargs is kwargs
    with pytest.raises(FrozenInstanceError):
        prepared.gather_ids = torch.tensor([0])


@pytest.mark.parametrize(
    ("runner_type", "module"),
    [
        (PoolingRunner, pooling_module),
        (MultimodalEncoderRunner, mm_encoder_module),
    ],
)
def test_runner_wraps_prepared_input_dict_without_copying(
    runner_type: type[PoolingRunner] | type[MultimodalEncoderRunner],
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = {"input_ids": torch.tensor([1])}
    gather_ids = torch.tensor([0])
    monkeypatch.setattr(module, "prepare_no_cache_inputs", Mock(return_value=(kwargs, gather_ids)))
    deps = _deps()
    runner = runner_type(_model(is_generation=False), deps)
    lora_manager = object()

    prepared = runner.prepare_inputs(
        SimpleNamespace(),
        SimpleNamespace(),
        spec_metadata=None,
        resource_manager=SimpleNamespace(),
        enable_spec_decode=False,
        cuda_graph_lora_manager=lora_manager,
        runtime_draft_len=3,
    )

    assert prepared.kwargs is kwargs
    assert prepared.gather_ids is gather_ids
    call_kwargs = module.prepare_no_cache_inputs.call_args.kwargs
    assert call_kwargs["lora"] is deps.lora
    assert call_kwargs["cuda_graph_lora_manager"] is lora_manager
    assert call_kwargs["runtime_draft_len"] == 3


@pytest.mark.parametrize("runner_type", [PoolingRunner, MultimodalEncoderRunner])
def test_no_cache_runner_warmup_and_capture_are_noops(
    runner_type: type[PoolingRunner] | type[MultimodalEncoderRunner],
) -> None:
    runner = runner_type(_model(is_generation=False), _deps())
    resource_manager = SimpleNamespace()

    assert runner.warmup(resource_manager) is None
    assert runner.capture_graphs(resource_manager) is None


def test_mm_encoder_runner_forward_step_returns_empty_result_without_multimodal_params() -> None:
    runner = MultimodalEncoderRunner(SimpleNamespace(), _deps())

    result = runner._forward_step({}, SimpleNamespace(context_requests=[]))

    assert result == {
        "mm_embeddings": [],
        "mm_embedding_request_indices": [],
        "mm_embedding_lengths": [],
    }


def test_mm_encoder_runner_forward_step_rejects_payload_request_count_mismatch() -> None:
    request = SimpleNamespace(py_multimodal_data={"image": object()})
    runner = MultimodalEncoderRunner(SimpleNamespace(), _deps())

    with pytest.raises(ValueError, match="one multimodal payload per context"):
        runner._forward_step(
            {"multimodal_params": [SimpleNamespace(), SimpleNamespace()]},
            SimpleNamespace(context_requests=[request]),
        )


def test_mm_encoder_runner_forward_step_skips_metadata_only_and_missing_length_requests() -> None:
    requests = [
        SimpleNamespace(py_multimodal_data={"mrope_config": {}}),
        SimpleNamespace(py_multimodal_data={"image": object()}),
        SimpleNamespace(
            py_multimodal_data={
                "image": object(),
                "multimodal_embedding_lengths": [2],
            },
            multimodal_lengths=[2],
        ),
    ]
    params = [
        SimpleNamespace(multimodal_data={}),
        SimpleNamespace(multimodal_data={}),
        SimpleNamespace(multimodal_data={}),
    ]
    model = SimpleNamespace(forward=Mock(return_value=[torch.arange(4).reshape(2, 2)]))
    runner = MultimodalEncoderRunner(model, _deps())

    result = runner._forward_step(
        {"multimodal_params": params},
        SimpleNamespace(context_requests=requests),
    )

    model.forward.assert_called_once_with([params[2]])
    assert result["logits"] is None
    assert result["mm_embedding_request_indices"] == [2]
    assert result["mm_embedding_lengths"] == [[2]]
    torch.testing.assert_close(result["mm_embeddings"][0], torch.arange(4).reshape(2, 2))


def test_padding_params_preserve_existing_cases() -> None:
    assert get_padding_params(
        129,
        1,
        None,
        dist=None,
        enable_attention_dp=False,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE,
        prefill_cuda_graph_num_tokens=[128, 256, 512],
    ) == (256, True, None)


def test_padding_params_requires_dist_for_attention_dp() -> None:
    with pytest.raises(AssertionError, match="attention DP requires"):
        get_padding_params(
            129,
            1,
            [129],
            dist=None,
            enable_attention_dp=True,
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE,
            prefill_cuda_graph_num_tokens=[128, 256, 512],
        )


def test_position_offset_helpers_preserve_identity_and_unwrap_models() -> None:
    position_ids = [0, 1]
    model_without_offset = SimpleNamespace()
    top_level = SimpleNamespace(position_id_offset=2)
    wrapped = SimpleNamespace(_orig_mod=SimpleNamespace(model=SimpleNamespace(_orig_mod=top_level)))

    assert apply_position_id_offset(position_ids, model=model_without_offset) is position_ids
    assert get_top_level_model(wrapped) is top_level
    assert apply_position_id_offset(position_ids, model=wrapped) == [2, 3]
