# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from _torch.pyexecutor.engine.runners.utils import _config, _make_runner, _model

from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.pyexecutor.engine.input_buffers import InputBuffers
from tensorrt_llm._torch.pyexecutor.engine.runners import common as common_module
from tensorrt_llm._torch.pyexecutor.engine.runners import no_kv_cache as no_kv_cache_module
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import (
    PreparedInputs,
    RunnerConfig,
    ScheduledInputs,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.mm_encoder import MultimodalEncoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.no_kv_cache import (
    NoKVCacheRunner,
    NoKVCacheRunnerConfig,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.pooling import PoolingRunner
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import get_model_extra_attrs
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

pytestmark = pytest.mark.cpu_only


class _AttentionMetadata:
    def __init__(self, num_tokens: int) -> None:
        self.num_tokens = num_tokens
        self.is_cuda_graph = False
        self.kv_cache_manager = None
        self.prepare = Mock()

    def get_adp_token_counts(self) -> list[int]:
        return [self.num_tokens]

    def set_adp_token_counts(self, counts: list[list[int]]) -> list[int]:
        return counts[0]


class _AttentionBackend:
    Metadata = _AttentionMetadata


def _request(
    request_id: int,
    tokens: list[int],
    *,
    position_ids: list[int] | None = None,
    multimodal_data: dict | None = None,
) -> SimpleNamespace:
    params = (
        MultimodalParams(multimodal_data=multimodal_data) if multimodal_data is not None else None
    )
    return SimpleNamespace(
        py_multimodal_params=params,
        get_tokens=Mock(return_value=tokens),
        py_request_id=request_id,
        py_position_ids=position_ids,
        py_multimodal_data=multimodal_data,
        py_seq_slot=request_id,
    )


def _prepare(
    monkeypatch: pytest.MonkeyPatch,
    requests: list[SimpleNamespace],
    *,
    model: object | None = None,
    dist: object | None = None,
    enable_attention_dp: bool = False,
) -> tuple[
    dict,
    _AttentionMetadata,
    torch.Tensor,
    torch.Tensor,
]:
    monkeypatch.setattr(no_kv_cache_module, "prefer_pinned", lambda: False)
    monkeypatch.setattr(no_kv_cache_module, "VanillaAttentionMetadata", _AttentionMetadata)
    attn_metadata = _AttentionMetadata(sum(len(request.get_tokens(0)) for request in requests))
    scheduled_requests = SimpleNamespace(
        context_requests=requests,
        generation_requests=[],
        num_context_requests=len(requests),
    )
    input_ids_cuda = torch.full((16,), -1, dtype=torch.int)
    position_ids_cuda = torch.full((16,), -1, dtype=torch.int)
    buffers = InputBuffers(
        input_ids_cuda=input_ids_cuda,
        position_ids_cuda=position_ids_cuda,
    )
    with patch.object(InputBuffers, "allocate", return_value=buffers):
        runner = PoolingRunner(
            model or SimpleNamespace(),
            NoKVCacheRunnerConfig(
                max_batch_size=4,
                max_num_tokens=16,
                max_seq_len=8,
                max_beam_width=1,
                without_logits=False,
                attention_backend=_AttentionBackend,
                enable_attention_dp=enable_attention_dp,
                prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
                prefill_cuda_graph_num_tokens=[],
                mm_encoder_cache_enabled=True,
            ),
            mapping=SimpleNamespace(has_cp_helix=lambda: False),
            dist=dist,
            moe_load_balancer=None,
            model_caller=Mock(),
            attention_runtime_features=AttentionRuntimeFeatures(),
        )
    monkeypatch.setattr(
        runner,
        "_setup_attn_metadata",
        Mock(return_value=attn_metadata),
    )
    prepared = runner._prepare_inputs(
        scheduled_requests,
        resource_manager=SimpleNamespace(name="resources"),
    )
    return (
        prepared.kwargs,
        attn_metadata,
        input_ids_cuda,
        position_ids_cuda,
    )


def test_no_kv_cache_runner_prepare_inputs_packs_context_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_prefill_flag = Mock()
    monkeypatch.setattr(
        common_module,
        "set_per_request_prefill_cuda_graph_flag",
        set_prefill_flag,
    )
    first = _request(2, [11, 12])
    second = _request(5, [21], position_ids=[7])

    (
        inputs,
        attn_metadata,
        input_ids_cuda,
        position_ids_cuda,
    ) = _prepare(monkeypatch, [first, second])

    torch.testing.assert_close(inputs["input_ids"], torch.tensor([11, 12, 21], dtype=torch.int))
    torch.testing.assert_close(inputs["position_ids"], torch.tensor([[0, 1, 7]], dtype=torch.int))
    assert inputs["attn_metadata"] is attn_metadata
    assert inputs["inputs_embeds"] is None
    assert inputs["multimodal_params"] == []
    assert first.py_batch_idx == 2
    assert second.py_batch_idx == 5
    assert attn_metadata.num_contexts == 2
    assert attn_metadata.max_seq_len == 8
    assert attn_metadata.request_ids == [2, 5]
    torch.testing.assert_close(attn_metadata.seq_lens, torch.tensor([2, 1], dtype=torch.int))
    attn_metadata.prepare.assert_called_once_with()
    set_prefill_flag.assert_called_once_with(False)
    torch.testing.assert_close(input_ids_cuda[:3], torch.tensor([11, 12, 21], dtype=torch.int))
    torch.testing.assert_close(position_ids_cuda[:3], torch.tensor([0, 1, 7], dtype=torch.int))


def test_no_kv_cache_runner_prepare_inputs_builds_and_ships_multimodal_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(3, [1, 101], multimodal_data={"image": object()})
    multimodal_params = Mock()
    multimodal_params_factory = Mock(return_value=multimodal_params)
    multimodal_input = object()
    build_multimodal_input = Mock(return_value=multimodal_input)
    text_indices = torch.tensor([0])
    mm_indices = torch.tensor([1])
    prepare_indices = Mock(return_value=(text_indices, mm_indices))
    ship_indices = Mock()
    monkeypatch.setattr(no_kv_cache_module, "MultimodalParams", multimodal_params_factory)
    monkeypatch.setattr(
        no_kv_cache_module,
        "_build_request_multimodal_input",
        build_multimodal_input,
    )
    monkeypatch.setattr(no_kv_cache_module, "prepare_multimodal_indices", prepare_indices)
    monkeypatch.setattr(no_kv_cache_module, "ship_multimodal_indices", ship_indices)
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=100))

    inputs, _, _, _ = _prepare(
        monkeypatch,
        [request],
        model=model,
    )

    build_multimodal_input.assert_called_once_with(request, True)
    multimodal_params_factory.assert_called_once_with(
        multimodal_input=multimodal_input,
        multimodal_data=request.py_multimodal_data,
        input_ids_start_offset=0,
    )
    multimodal_params.to_device.assert_called_once_with(
        "multimodal_data",
        "cuda",
        pin_memory=False,
    )
    prepare_indices.assert_called_once_with([1, 101], model=model)
    assert inputs["multimodal_params"] == [multimodal_params]
    ship_indices.assert_called_once_with(
        inputs,
        mm_token_indices_cpu=mm_indices,
        text_token_indices_cpu=text_indices,
        num_ctx_tokens=2,
        total_num_tokens=2,
    )


def test_no_kv_cache_runner_prepare_inputs_gathers_attention_dp_token_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dist = SimpleNamespace(
        tp_allgather_int64=Mock(return_value=torch.tensor([[2], [3]])),
        tp_cp_allgather_int64=Mock(return_value=torch.tensor([[2], [5]])),
    )

    _, attn_metadata, _, _ = _prepare(
        monkeypatch,
        [_request(4, [31, 32])],
        dist=dist,
        enable_attention_dp=True,
    )

    assert attn_metadata.all_rank_num_tokens == [2, 5]
    dist.tp_allgather_int64.assert_called_once_with([2])
    dist.tp_cp_allgather_int64.assert_called_once_with([2])


def test_no_kv_cache_runner_rejects_kv_manager_before_preparation() -> None:
    runner = _make_runner(PoolingRunner, _model(is_generation=False))
    runner._prepare_inputs = Mock()
    resource_manager = Mock()
    resource_manager.get_resource_manager.return_value = object()

    with pytest.raises(
        AssertionError,
        match="no-KV-cache runner was initialized, but a KV cache manager was allocated",
    ):
        runner.forward(
            ScheduledInputs(batch=ScheduledRequests()), resource_manager=resource_manager
        )

    runner._prepare_inputs.assert_not_called()


def test_no_kv_cache_config_extends_common_runner_config() -> None:
    assert isinstance(_config(), RunnerConfig)


def test_no_kv_cache_runner_exposes_only_forward_step_as_abstract() -> None:
    assert NoKVCacheRunner.__abstractmethods__ == frozenset({"_forward_step"})


@pytest.mark.parametrize("runner_type", [PoolingRunner, MultimodalEncoderRunner])
def test_concrete_runners_implement_no_kv_cache_forward_step(
    runner_type: type[NoKVCacheRunner],
) -> None:
    assert issubclass(runner_type, NoKVCacheRunner)
    assert runner_type._forward_step is not NoKVCacheRunner._forward_step


def test_no_kv_cache_runner_owns_and_reuses_attention_metadata() -> None:
    runner = _make_runner(PoolingRunner, _model(is_generation=False))

    first = runner._setup_attn_metadata()
    second = runner._setup_attn_metadata()

    assert first is second
    assert first.kv_cache_manager is None
    assert first.max_num_requests == 4
    assert first.max_num_sequences == 4
    assert first.max_num_heads_per_kv == 4
    assert first.block_ids_per_seq is None
    assert first.kv_block_ids_per_seq is None


@pytest.mark.parametrize("runner_type", [PoolingRunner, MultimodalEncoderRunner])
def test_no_kv_cache_runner_default_lifecycle_with_no_kv_resources(
    runner_type: type[PoolingRunner] | type[MultimodalEncoderRunner],
) -> None:
    runner = _make_runner(runner_type, _model(is_generation=False))
    resource_manager = SimpleNamespace(get_resource_manager=Mock(return_value=None))

    assert runner.warmup(resource_manager) is None
    assert runner.release_graphs() is None
    assert runner.wait_for_input_copy() is None


def test_no_kv_forward_enters_model_context() -> None:
    model = _model(is_generation=False)

    def model_forward(**kwargs: object) -> dict[str, object]:
        assert torch.is_inference_mode_enabled()
        assert get_model_extra_attrs() is model.extra_attrs
        return {"hidden_states": object()}

    caller = Mock(side_effect=model_forward)
    runner = _make_runner(PoolingRunner, model, model_caller=caller)
    runner._prepare_inputs = Mock(return_value=PreparedInputs({}))
    resources = SimpleNamespace(get_resource_manager=Mock(return_value=None))

    runner.forward(ScheduledInputs(batch=ScheduledRequests()), resource_manager=resources)

    caller.assert_called_once()
    assert get_model_extra_attrs() is None


def test_no_kv_forward_preserves_model_output_dictionary_without_length_update() -> None:
    outputs = {"hidden_states": object()}
    caller = Mock(return_value=outputs)
    runner = _make_runner(PoolingRunner, _model(is_generation=False), model_caller=caller)
    runner._prepare_inputs = Mock(return_value=PreparedInputs({}))
    inputs = ScheduledInputs(batch=ScheduledRequests(), runtime_draft_len=3)
    resources = SimpleNamespace(get_resource_manager=Mock(return_value=None))

    actual = runner.forward(inputs, resource_manager=resources)

    assert actual is outputs
    assert "runtime_draft_len" not in outputs
    assert inputs.runtime_draft_len == 3
