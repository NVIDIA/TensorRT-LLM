# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.pyexecutor.engine.runners import no_kv_cache as no_kv_cache_module
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import RunnerDeps
from tensorrt_llm._torch.pyexecutor.engine.runners.no_kv_cache import NoKVCacheRunnerConfig
from tensorrt_llm._torch.pyexecutor.engine.runners.pooling import PoolingRunner
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

pytestmark = pytest.mark.cpu_only


class _AttentionMetadata:
    def __init__(self, num_tokens: int) -> None:
        self.num_tokens = num_tokens
        self.is_cuda_graph = False
        self.kv_cache_manager = None
        self.prepare = Mock()


class _AttentionBackend:
    Metadata = _AttentionMetadata


def _request(
    request_id: int,
    tokens: list[int],
    *,
    position_ids: list[int] | None = None,
    multimodal_data: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        get_tokens=Mock(return_value=tokens),
        py_request_id=request_id,
        position_ids=position_ids,
        py_multimodal_data=multimodal_data,
        py_mm_item_order=None,
        py_seq_slot=request_id,
    )


def _prepare(
    monkeypatch: pytest.MonkeyPatch,
    requests: list[SimpleNamespace],
    *,
    model: object | None = None,
    spec_metadata: object | None = None,
    dist: object | None = None,
    enable_attention_dp: bool = False,
    enable_spec_decode: bool = False,
    generation_requests: list[object] | None = None,
    lora_params: dict | None = None,
) -> tuple[
    dict,
    torch.Tensor | None,
    _AttentionMetadata,
    SimpleNamespace,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    monkeypatch.setattr(no_kv_cache_module, "prefer_pinned", lambda: False)
    monkeypatch.setattr(no_kv_cache_module, "VanillaAttentionMetadata", _AttentionMetadata)
    attn_metadata = _AttentionMetadata(sum(len(request.get_tokens(0)) for request in requests))
    scheduled_requests = SimpleNamespace(
        context_requests=requests,
        generation_requests=generation_requests or [],
        num_context_requests=len(requests),
    )
    input_ids_cuda = torch.full((16,), -1, dtype=torch.int)
    position_ids_cuda = torch.full((16,), -1, dtype=torch.int)
    gather_ids_cuda = torch.full((8,), -1, dtype=torch.int)
    draft_tokens_cuda = torch.empty(8, dtype=torch.int)
    lora = SimpleNamespace(build=Mock(return_value=lora_params))
    runner = PoolingRunner(
        model or SimpleNamespace(),
        RunnerDeps(
            dist=dist,
            mapping=SimpleNamespace(has_cp_helix=lambda: False),
            input_ids_cuda=input_ids_cuda,
            position_ids_cuda=position_ids_cuda,
            gather_ids_cuda=gather_ids_cuda,
            draft_tokens_cuda=draft_tokens_cuda,
            cache_indirection=None,
            lora=lora,
            moe_load_balancer=None,
            model_forward=Mock(),
        ),
        NoKVCacheRunnerConfig(
            max_batch_size=4,
            max_num_tokens=16,
            max_seq_len=8,
            max_beam_width=1,
            without_logits=False,
            attention_backend=_AttentionBackend,
            attention_runtime_features=AttentionRuntimeFeatures(),
            enable_attention_dp=enable_attention_dp,
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
            prefill_cuda_graph_num_tokens=[],
            mm_encoder_cache_enabled=True,
            spec_config=object() if enable_spec_decode else None,
            is_draft_model=False,
            num_seq_slots=None,
            original_max_draft_len=0,
            original_max_total_draft_tokens=0,
            spec_dec_max_total_draft_tokens=0,
        ),
    )
    monkeypatch.setattr(
        runner,
        "setup_attn_metadata",
        Mock(return_value=attn_metadata),
    )
    monkeypatch.setattr(
        runner,
        "setup_spec_metadata",
        Mock(return_value=spec_metadata),
    )
    prepared = runner.prepare_inputs(
        scheduled_requests,
        resource_manager=SimpleNamespace(name="resources"),
        cuda_graph_lora_manager=None,
        runtime_draft_len=2,
    )
    return (
        prepared.kwargs,
        prepared.gather_ids,
        attn_metadata,
        lora,
        input_ids_cuda,
        position_ids_cuda,
        gather_ids_cuda,
        draft_tokens_cuda,
    )


def test_no_kv_cache_runner_rejects_unhandled_model_inputs() -> None:
    runner = PoolingRunner.__new__(PoolingRunner)

    with pytest.raises(NotImplementedError, match="token_type_ids"):
        runner.prepare_inputs(
            SimpleNamespace(),
            resource_manager=None,
            cuda_graph_lora_manager=None,
            runtime_draft_len=0,
            token_type_ids=object(),
        )


def test_no_kv_cache_runner_prepare_inputs_packs_context_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_prefill_flag = Mock()
    monkeypatch.setattr(
        no_kv_cache_module,
        "set_per_request_prefill_cuda_graph_flag",
        set_prefill_flag,
    )
    first = _request(2, [11, 12])
    second = _request(5, [21], position_ids=[7])

    (
        inputs,
        gather_ids,
        attn_metadata,
        lora,
        input_ids_cuda,
        position_ids_cuda,
        _,
        _,
    ) = _prepare(monkeypatch, [first, second])

    assert gather_ids is None
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
    lora.build.assert_called_once()
    assert lora.build.call_args.kwargs["enable_spec_decode"] is False
    assert lora.build.call_args.kwargs["runtime_draft_len"] == 2
    torch.testing.assert_close(input_ids_cuda[:3], torch.tensor([11, 12, 21], dtype=torch.int))
    torch.testing.assert_close(position_ids_cuda[:3], torch.tensor([0, 1, 7], dtype=torch.int))


def test_no_kv_cache_runner_prepare_inputs_builds_and_ships_multimodal_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(3, [1, 101], multimodal_data={"image": object()})
    request.py_mm_item_order = ["image"]
    multimodal_params = Mock()
    multimodal_params_factory = Mock(return_value=multimodal_params)
    multimodal_input = object()
    build_multimodal_input = Mock(return_value=multimodal_input)
    text_indices = torch.tensor([0])
    mm_indices = torch.tensor([1])
    prepare_indices = Mock(return_value=(text_indices, mm_indices))
    ship_indices = Mock()
    lora_value = object()
    monkeypatch.setattr(no_kv_cache_module, "MultimodalParams", multimodal_params_factory)
    monkeypatch.setattr(
        no_kv_cache_module,
        "_build_request_multimodal_input",
        build_multimodal_input,
    )
    monkeypatch.setattr(no_kv_cache_module, "prepare_multimodal_indices", prepare_indices)
    monkeypatch.setattr(no_kv_cache_module, "ship_multimodal_indices", ship_indices)
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=100))

    inputs, _, _, _, _, _, _, _ = _prepare(
        monkeypatch,
        [request],
        model=model,
        lora_params={"layer": lora_value},
    )

    build_multimodal_input.assert_called_once_with(request, True)
    multimodal_params_factory.assert_called_once_with(
        multimodal_input=multimodal_input,
        multimodal_data=request.py_multimodal_data,
        mm_item_order=request.py_mm_item_order,
        input_ids_start_offset=0,
    )
    multimodal_params.to_device.assert_called_once_with(
        "multimodal_data",
        "cuda",
        pin_memory=False,
    )
    prepare_indices.assert_called_once_with([1, 101], model=model)
    assert inputs["multimodal_params"] == [multimodal_params]
    assert inputs["lora_params"] == {"layer": lora_value}
    ship_indices.assert_called_once_with(
        inputs,
        mm_token_indices_cpu=mm_indices,
        text_token_indices_cpu=text_indices,
        num_ctx_tokens=2,
        total_num_tokens=2,
    )


def test_no_kv_cache_runner_prepare_inputs_populates_spec_and_attention_dp_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_metadata = SimpleNamespace(
        prepare=Mock(),
        spec_dec_mode=SimpleNamespace(
            is_mtp_eagle_one_model=Mock(return_value=False),
            is_eagle3_one_model=Mock(return_value=False),
        ),
    )
    dist = SimpleNamespace(
        tp_allgather_int64=Mock(return_value=torch.tensor([[2], [3]])),
        tp_cp_allgather_int64=Mock(return_value=torch.tensor([[2, 2, 1, 1], [3, 3, 2, 1]])),
    )
    request = _request(4, [31, 32])
    generation_request = object()

    (
        inputs,
        _,
        attn_metadata,
        lora,
        _,
        _,
        gather_ids_cuda,
        _,
    ) = _prepare(
        monkeypatch,
        [request],
        spec_metadata=spec_metadata,
        dist=dist,
        enable_attention_dp=True,
        enable_spec_decode=True,
        generation_requests=[generation_request],
    )

    assert inputs["spec_metadata"] is spec_metadata
    assert spec_metadata.draft_tokens.shape == (0,)
    torch.testing.assert_close(spec_metadata.gather_ids, gather_ids_cuda[:1])
    assert spec_metadata.request_ids == [4]
    assert spec_metadata.num_generations == 1
    assert spec_metadata.num_tokens == 2
    assert spec_metadata.seq_lens == [2]
    assert spec_metadata.all_rank_num_tokens == [2, 3]
    assert spec_metadata.all_rank_num_seqs == [1, 2]
    assert spec_metadata.all_rank_num_gens == [1, 1]
    assert attn_metadata.all_rank_num_tokens == [2, 3]
    spec_metadata.prepare.assert_called_once_with()
    dist.tp_allgather_int64.assert_called_once_with([2])
    dist.tp_cp_allgather_int64.assert_called_once_with([2, 2, 1, 1])
    assert lora.build.call_args.kwargs["enable_spec_decode"] is True
