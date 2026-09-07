# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.pyexecutor.engine.runners import (
    apply_position_id_offset,
    get_padding_params,
    get_top_level_model,
    resolve_runner_type,
)
from tensorrt_llm._torch.pyexecutor.engine.runners import no_cache as no_cache_module
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import (
    PreparedInputs,
    RunnerConfig,
    RunnerDeps,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.mm_encoder import MultimodalEncoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.no_cache import (
    NoCacheRunner,
    NoCacheRunnerConfig,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.pooling import PoolingRunner
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

pytestmark = pytest.mark.cpu_only


class _AttentionMetadata:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self.block_ids_per_seq = object()
        self.kv_block_ids_per_seq = object()
        self.on_update_kv_lens = Mock()


class _AttentionBackend:
    Metadata = _AttentionMetadata


def _make_runner(
    runner_type: type[PoolingRunner] | type[MultimodalEncoderRunner],
    model: Any,
    deps: RunnerDeps | None = None,
    config: NoCacheRunnerConfig | None = None,
) -> PoolingRunner | MultimodalEncoderRunner:
    return runner_type(model, deps or _deps(), config or _config())


def _deps(*, model_forward: Mock | None = None) -> RunnerDeps:
    if model_forward is None:
        model_forward = Mock(return_value={"logits": torch.empty(0)})
    return RunnerDeps(
        dist=SimpleNamespace(),
        mapping=SimpleNamespace(),
        input_ids_cuda=torch.empty(16, dtype=torch.int),
        position_ids_cuda=torch.empty(16, dtype=torch.int),
        gather_ids_cuda=None,
        draft_tokens_cuda=None,
        cache_indirection=None,
        lora=SimpleNamespace(build=Mock(return_value=None)),
        model_forward=model_forward,
    )


def _config() -> NoCacheRunnerConfig:
    return NoCacheRunnerConfig(
        max_batch_size=4,
        max_num_tokens=16,
        max_seq_len=16,
        max_beam_width=1,
        without_logits=False,
        attention_backend=_AttentionBackend,
        attention_runtime_features=AttentionRuntimeFeatures(),
        enable_attention_dp=False,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
        prefill_cuda_graph_num_tokens=[],
        mm_encoder_cache_enabled=False,
        spec_config=None,
        is_draft_model=False,
        num_seq_slots=None,
        original_max_draft_len=0,
        original_max_total_draft_tokens=0,
        spec_dec_max_total_draft_tokens=0,
    )


def _model(*, is_generation: bool, is_encoder_decoder: bool = False) -> SimpleNamespace:
    pretrained_config = SimpleNamespace(
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_generation=is_generation,
            is_encoder_decoder=is_encoder_decoder,
            pretrained_config=pretrained_config,
            sparse_attention_config=None,
            enable_flash_mla=False,
        ),
        config=pretrained_config,
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

    actual_type = resolve_runner_type(
        _model(is_generation=is_generation, is_encoder_decoder=is_encoder_decoder),
        args,
    )

    assert actual_type is runner_type


def test_resolve_runner_type_does_not_require_runtime_dependencies() -> None:
    args = SimpleNamespace(encode_only=False, mm_encoder_only=False)

    assert resolve_runner_type(_model(is_generation=False), args) is PoolingRunner


def test_resolve_runner_checks_encode_only_before_pooling() -> None:
    args = SimpleNamespace(encode_only=True, mm_encoder_only=False)

    assert resolve_runner_type(_model(is_generation=False), args) is None


def test_resolve_runner_checks_mm_encoder_before_non_generation() -> None:
    args = SimpleNamespace(encode_only=False, mm_encoder_only=True)

    assert resolve_runner_type(_model(is_generation=False), args) is MultimodalEncoderRunner


def test_prepared_inputs_is_frozen_and_preserves_kwargs_identity() -> None:
    kwargs = {"input_ids": torch.tensor([1])}
    prepared = PreparedInputs(kwargs)

    assert prepared.kwargs is kwargs
    with pytest.raises(FrozenInstanceError):
        prepared.gather_ids = torch.tensor([0])


def test_no_cache_config_extends_common_runner_config() -> None:
    assert isinstance(_config(), RunnerConfig)


def test_no_cache_runner_exposes_only_forward_step_as_abstract() -> None:
    assert NoCacheRunner.__abstractmethods__ == frozenset({"_forward_step"})


@pytest.mark.parametrize("runner_type", [PoolingRunner, MultimodalEncoderRunner])
def test_concrete_runners_implement_no_cache_forward_step(
    runner_type: type[NoCacheRunner],
) -> None:
    assert issubclass(runner_type, NoCacheRunner)
    assert runner_type._forward_step is not NoCacheRunner._forward_step


@pytest.mark.parametrize("runner_type", [PoolingRunner, MultimodalEncoderRunner])
def test_runner_wraps_prepared_input_dict_without_copying(
    runner_type: type[PoolingRunner] | type[MultimodalEncoderRunner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = {"input_ids": torch.tensor([1])}
    gather_ids = torch.tensor([0])
    monkeypatch.setattr(
        no_cache_module,
        "prepare_no_cache_inputs",
        Mock(return_value=(kwargs, gather_ids)),
    )
    deps = _deps()
    runner = _make_runner(runner_type, _model(is_generation=False), deps)
    runner._attn_metadata = SimpleNamespace()
    lora_manager = object()

    prepared = runner.prepare_inputs(
        SimpleNamespace(),
        resource_manager=SimpleNamespace(),
        cuda_graph_lora_manager=lora_manager,
        runtime_draft_len=3,
    )

    assert prepared.kwargs is kwargs
    assert prepared.gather_ids is gather_ids
    call_kwargs = no_cache_module.prepare_no_cache_inputs.call_args.kwargs
    assert call_kwargs["lora"] is deps.lora
    assert call_kwargs["cuda_graph_lora_manager"] is lora_manager
    assert call_kwargs["runtime_draft_len"] == 3


def test_no_cache_runner_owns_and_reuses_attention_metadata() -> None:
    runner = _make_runner(PoolingRunner, _model(is_generation=False))

    first = runner.setup_attn_metadata()
    second = runner.setup_attn_metadata()

    assert first is second
    assert first.kv_cache_manager is None
    assert first.max_num_requests == 4
    assert first.max_num_sequences == 4
    assert first.num_heads_per_kv == 4
    assert first.block_ids_per_seq is None
    assert first.kv_block_ids_per_seq is None


def test_no_cache_runner_owns_spec_metadata_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_mode = SimpleNamespace(
        attention_need_spec_dec_mode=Mock(return_value=True),
        is_parallel_draft=Mock(return_value=False),
    )
    spec_metadata = SimpleNamespace(
        spec_dec_mode=spec_mode,
        is_spec_dec_tree=True,
        is_spec_dec_dynamic_tree=False,
    )
    get_spec_metadata = Mock(return_value=spec_metadata)
    monkeypatch.setattr(no_cache_module, "get_spec_metadata", get_spec_metadata)
    deps = _deps()
    runner = _make_runner(
        PoolingRunner,
        _model(is_generation=False),
        deps,
        replace(
            _config(),
            spec_config=SimpleNamespace(
                get_runtime_tokens_per_gen_step=lambda runtime_draft_len: (runtime_draft_len + 1),
            ),
            original_max_draft_len=2,
            spec_dec_max_total_draft_tokens=3,
        ),
    )
    attn_metadata = SimpleNamespace(update_spec_dec_param=Mock())
    scheduled_requests = SimpleNamespace(
        batch_size=2,
        num_context_requests=2,
        context_requests=[object(), object()],
        generation_requests=[],
    )
    resource_manager = SimpleNamespace(get_resource_manager=Mock(return_value=None))

    result = runner.setup_spec_metadata(
        scheduled_requests,
        resource_manager,
        attn_metadata,
        runtime_draft_len=1,
    )

    assert result is spec_metadata
    assert spec_metadata.runtime_draft_len == 1
    assert spec_metadata.runtime_tokens_per_gen_step == 2
    attn_metadata.update_spec_dec_param.assert_called_once_with(
        batch_size=2,
        is_spec_decoding_enabled=True,
        is_spec_dec_tree=True,
        is_spec_dec_dynamic_tree=False,
        max_draft_len=2,
        max_total_draft_tokens=3,
        spec_metadata=spec_metadata,
        spec_tree_manager=None,
        num_contexts=2,
    )


def test_pooling_runner_owns_forward_output_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.arange(12).reshape(3, 4)
    model_forward = Mock(return_value=logits)
    runner = _make_runner(
        PoolingRunner,
        _model(is_generation=False),
        _deps(model_forward=model_forward),
    )
    attn_metadata = SimpleNamespace(on_update_kv_lens=Mock())
    prepared = PreparedInputs(
        {"attn_metadata": attn_metadata, "input_ids": torch.tensor([1])},
        gather_ids=torch.tensor([2, 0]),
    )
    monkeypatch.setattr(runner, "prepare_inputs", Mock(return_value=prepared))

    outputs = runner.forward(
        SimpleNamespace(),
        resource_manager=SimpleNamespace(),
        cuda_graph_lora_manager=None,
        runtime_draft_len=0,
        moe_load_balancer=None,
        gather_context_logits=False,
    )

    torch.testing.assert_close(outputs["logits"], logits[[2, 0]])
    attn_metadata.on_update_kv_lens.assert_called_once_with()
    model_forward.assert_called_once_with(
        attn_metadata=attn_metadata,
        input_ids=prepared.kwargs["input_ids"],
        return_context_logits=True,
    )


@pytest.mark.parametrize("runner_type", [PoolingRunner, MultimodalEncoderRunner])
def test_pooling_runner_warmup_and_capture_are_noops(
    runner_type: type[PoolingRunner] | type[MultimodalEncoderRunner],
) -> None:
    runner = _make_runner(runner_type, _model(is_generation=False))
    resource_manager = SimpleNamespace()

    assert runner.warmup(resource_manager) is None
    assert runner.capture_graphs(resource_manager) is None


def test_mm_encoder_runner_forward_step_returns_empty_result_without_multimodal_params() -> None:
    runner = _make_runner(MultimodalEncoderRunner, SimpleNamespace())

    result = runner._forward_step({}, SimpleNamespace(context_requests=[]))

    assert result == {
        "mm_embeddings": [],
        "mm_embedding_request_indices": [],
        "mm_embedding_lengths": [],
    }


def test_mm_encoder_runner_forward_step_rejects_payload_request_count_mismatch() -> None:
    request = SimpleNamespace(py_multimodal_data={"image": object()})
    runner = _make_runner(MultimodalEncoderRunner, SimpleNamespace())

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
    runner = _make_runner(MultimodalEncoderRunner, model)

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
