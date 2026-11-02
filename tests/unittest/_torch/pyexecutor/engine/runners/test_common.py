# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.pyexecutor.engine.runners import (
    DecoderRunner,
    EncoderDecoderRunner,
    EncoderRunner,
    common,
    resolve_runner_type,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.common import (
    apply_position_id_offset,
    get_all_rank_num_tokens,
    get_padding_params,
    get_top_level_model,
    make_scheduled_inputs,
    pad_input_tokens,
    prepare_multimodal_indices,
    ship_multimodal_indices,
    token_model_inputs,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import ModelRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.mm_encoder import MultimodalEncoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.no_kv_cache import NoKVCacheRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.pooling import PoolingRunner
from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("encode_only", "mm_encoder_only", "is_generation", "is_encoder_decoder", "runner_type"),
    [
        (True, False, False, False, EncoderRunner),
        (True, True, False, False, MultimodalEncoderRunner),
        (False, True, False, False, MultimodalEncoderRunner),
        (False, False, False, True, PoolingRunner),
        (False, False, True, True, EncoderDecoderRunner),
        (False, False, True, False, DecoderRunner),
    ],
)
def test_resolve_runner_dispatches_startup_families(
    encode_only: bool,
    mm_encoder_only: bool,
    is_generation: bool,
    is_encoder_decoder: bool,
    runner_type: type[ModelRunner],
) -> None:
    args = SimpleNamespace(encode_only=encode_only, mm_encoder_only=mm_encoder_only)
    model = SimpleNamespace(
        model_config=SimpleNamespace(
            is_generation=is_generation, is_encoder_decoder=is_encoder_decoder
        )
    )

    assert resolve_runner_type(model, args) is runner_type


@pytest.mark.parametrize(
    ("runner_type", "initializer_name"),
    [
        (EncoderRunner, "_initialize_encoder_runner"),
        (EncoderDecoderRunner, "_initialize_encoder_decoder_runner"),
        (NoKVCacheRunner, "_initialize_no_kv_cache_runner"),
        (DecoderRunner, "_initialize_decoder_runner"),
        (ModelRunner, None),
    ],
)
def test_model_engine_initializes_runner_by_family(
    runner_type: type[ModelRunner], initializer_name: str | None
) -> None:
    engine = object.__new__(ModelEngine)
    engine._model_caller = Mock()
    if initializer_name is None:
        with pytest.raises(TypeError, match="No runner initializer registered"):
            engine._initialize_runner(runner_type)
        return

    expected = Mock(spec=runner_type)
    initializer = Mock(return_value=expected)
    setattr(engine, initializer_name, initializer)

    assert engine._initialize_runner(runner_type) is expected
    initializer.assert_called_once_with(runner_type)


@pytest.mark.parametrize("token_id_attribute", ["multimodal_token_ids", "mm_token_ids"])
def test_prepare_multimodal_indices_uses_model_token_ids(token_id_attribute: str) -> None:
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=100))
    setattr(model, token_id_attribute, torch.tensor([90, 91], dtype=torch.int32))

    text_indices, multimodal_indices = prepare_multimodal_indices(
        [1, 90, 2, 91, 3],
        model=model,
    )

    torch.testing.assert_close(text_indices, torch.tensor([0, 2, 4]))
    torch.testing.assert_close(multimodal_indices, torch.tensor([1, 3]))


def test_prepare_multimodal_indices_falls_back_to_out_of_vocab_tokens() -> None:
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=100))

    text_indices, multimodal_indices = prepare_multimodal_indices(
        [1, 100, 2, 101],
        model=model,
    )

    torch.testing.assert_close(text_indices, torch.tensor([0, 2]))
    torch.testing.assert_close(multimodal_indices, torch.tensor([1, 3]))


def test_get_all_rank_num_tokens_uses_tp_collective() -> None:
    dist = SimpleNamespace(
        tp_allgather_int64=Mock(return_value=torch.tensor([[5], [7]])),
    )
    mapping = SimpleNamespace(has_cp_helix=lambda: False)

    result = get_all_rank_num_tokens(
        SimpleNamespace(
            get_adp_token_counts=lambda: [5],
            set_adp_token_counts=lambda counts: AttentionMetadata.set_adp_token_counts(
                None, counts
            ),
        ),
        enable_attention_dp=True,
        mapping=mapping,
        dist=dist,
    )

    assert result == [5, 7]
    dist.tp_allgather_int64.assert_called_once_with([5])


def test_get_all_rank_num_tokens_reports_post_reduce_scatter_helix_count() -> None:
    dist = SimpleNamespace(
        tp_cp_allgather_int64=Mock(return_value=torch.tensor([[3], [4]])),
    )
    mapping = SimpleNamespace(cp_size=2, has_cp_helix=lambda: True)

    result = get_all_rank_num_tokens(
        SimpleNamespace(
            get_adp_token_counts=lambda: [5],
            set_adp_token_counts=lambda counts: AttentionMetadata.set_adp_token_counts(
                None, counts
            ),
        ),
        enable_attention_dp=True,
        mapping=mapping,
        dist=dist,
    )

    assert result == [3, 4]
    dist.tp_cp_allgather_int64.assert_called_once_with([3])


@pytest.mark.parametrize(
    "backend",
    [PrefillCudaGraphBackend.PIECEWISE, PrefillCudaGraphBackend.BREAKABLE],
)
def test_padding_params_round_up_to_capture_bucket(
    backend: PrefillCudaGraphBackend,
) -> None:
    assert get_padding_params(
        129,
        1,
        None,
        dist=None,
        enable_attention_dp=False,
        prefill_cuda_graph_backend=backend,
        prefill_cuda_graph_num_tokens=[128, 256, 512],
    ) == (256, True, None)


@pytest.mark.parametrize(
    ("context_counts", "token_counts", "expected"),
    [
        ([0, 1, 0, 0], [1, 129, 1, 1], (256, True, [256] * 4)),
        ([0, 0, 0, 0], [1, 129, 1, 1], (1, False, [1, 129, 1, 1])),
        ([0, 1, 0, 0], [1, 513, 1, 1], (1, False, [1, 513, 1, 1])),
    ],
)
def test_attention_dp_padding_uses_all_rank_context_and_token_counts(
    context_counts: list[int],
    token_counts: list[int],
    expected: tuple[int, bool, list[int]],
) -> None:
    dist = SimpleNamespace(
        tp_allgather_int64=lambda values: torch.tensor(context_counts).unsqueeze(1),
    )

    assert (
        get_padding_params(
            1,
            0,
            token_counts,
            dist=dist,
            enable_attention_dp=True,
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
            prefill_cuda_graph_num_tokens=[128, 256, 512],
        )
        == expected
    )


@pytest.mark.parametrize("position_shape", [(8,), (3, 1, 8)], ids=["flat", "mrope"])
def test_padded_token_inputs_zero_padding_rows_and_keep_position_layout(
    position_shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    set_prefill_flag = Mock()
    monkeypatch.setattr(common, "set_per_request_prefill_cuda_graph_flag", set_prefill_flag)
    attn_metadata = SimpleNamespace(padded_num_tokens=None, all_rank_num_tokens=None)
    input_ids = torch.full((8,), 7, dtype=torch.int)
    position_ids = torch.full(position_shape, 7, dtype=torch.int)

    num_tokens = pad_input_tokens(
        attn_metadata,
        3,
        1,
        None,
        input_ids=input_ids,
        position_ids=position_ids,
        dist=None,
        enable_attention_dp=False,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE,
        prefill_cuda_graph_num_tokens=[4, 8],
    )
    inputs = token_model_inputs(
        attn_metadata,
        "resources",
        input_ids=input_ids,
        position_ids=position_ids,
        num_tokens=num_tokens,
        multimodal_params=[],
    )

    assert num_tokens == 4
    assert attn_metadata.padded_num_tokens == 4
    set_prefill_flag.assert_called_once_with(True)
    assert input_ids.tolist() == [7, 7, 7, 0, 7, 7, 7, 7]
    assert position_ids[..., 3].eq(0).all() and position_ids[..., 4:].eq(7).all()
    assert inputs["input_ids"].tolist() == [7, 7, 7, 0]
    expected_positions = (1, 4) if len(position_shape) == 1 else (3, 1, 4)
    assert tuple(inputs["position_ids"].shape) == expected_positions
    assert inputs["resource_manager"] == "resources"


def test_unpadded_token_inputs_leave_buffers_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "set_per_request_prefill_cuda_graph_flag", Mock())
    attn_metadata = SimpleNamespace(padded_num_tokens=5, all_rank_num_tokens=None)
    input_ids = torch.full((8,), 7, dtype=torch.int)

    num_tokens = pad_input_tokens(
        attn_metadata,
        3,
        0,
        None,
        input_ids=input_ids,
        position_ids=torch.full((8,), 7, dtype=torch.int),
        dist=None,
        enable_attention_dp=False,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
        prefill_cuda_graph_num_tokens=[],
    )

    assert num_tokens == 3
    assert attn_metadata.padded_num_tokens is None
    assert input_ids.eq(7).all()


def test_position_offset_helpers_preserve_identity_and_unwrap_models() -> None:
    position_ids = [0, 1]
    model_without_offset = SimpleNamespace()
    top_level = SimpleNamespace(position_id_offset=2)
    wrapped = SimpleNamespace(_orig_mod=SimpleNamespace(model=SimpleNamespace(_orig_mod=top_level)))

    assert apply_position_id_offset(position_ids, model=model_without_offset) is position_ids
    assert get_top_level_model(wrapped) is top_level
    assert apply_position_id_offset(position_ids, model=wrapped) == [2, 3]


class _FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.dtype = torch.int64
        self.cuda_value = object()
        self.to = Mock(return_value=self.cuda_value)


@pytest.mark.parametrize("total_num_tokens", [4, 6])
def test_ship_multimodal_indices_copies_and_extends_text_indices(
    total_num_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mm_indices = _FakeTensor([1, 3])
    text_indices = _FakeTensor([0, 2])
    extra_text = _FakeTensor([4, 5])
    extended_text = _FakeTensor([0, 2, 4, 5])
    arange = Mock(return_value=extra_text)
    cat = Mock(return_value=extended_text)
    monkeypatch.setattr(common, "maybe_pin_memory", lambda tensor: tensor)
    monkeypatch.setattr(common.torch, "arange", arange)
    monkeypatch.setattr(common.torch, "cat", cat)
    inputs = {}

    ship_multimodal_indices(
        inputs,
        mm_token_indices_cpu=mm_indices,
        text_token_indices_cpu=text_indices,
        num_ctx_tokens=4,
        total_num_tokens=total_num_tokens,
    )

    assert inputs["mm_token_indices"] is mm_indices.cuda_value
    expected_text = extended_text if total_num_tokens > 4 else text_indices
    assert inputs["text_token_indices"] is expected_text.cuda_value
    mm_indices.to.assert_called_once_with("cuda", non_blocking=True)
    expected_text.to.assert_called_once_with("cuda", non_blocking=True)
    if total_num_tokens > 4:
        arange.assert_called_once_with(4, 6, dtype=text_indices.dtype)
        cat.assert_called_once_with([text_indices, extra_text])
    else:
        arange.assert_not_called()
        cat.assert_not_called()


@pytest.mark.parametrize("returns_context_logits", [False, True])
def test_make_scheduled_inputs_gathers_context_logits_from_context_requests(
    returns_context_logits: bool,
) -> None:
    batch = ScheduledRequests()
    batch.context_requests_chunking = [SimpleNamespace(py_return_context_logits=False)]
    batch.context_requests_last_chunk = [
        SimpleNamespace(py_return_context_logits=returns_context_logits)
    ]
    batch.generation_requests = [SimpleNamespace(py_return_context_logits=True)]
    new_tensors_device = object()

    inputs = make_scheduled_inputs(
        batch, new_tensors_device, None, enable_spec_decode=True, runtime_draft_len=3
    )

    assert inputs.batch is batch
    assert inputs.new_tensors_device is new_tensors_device
    assert inputs.cache_indirection_buffer is None
    assert inputs.gather_context_logits is returns_context_logits
    assert (inputs.enable_spec_decode, inputs.runtime_draft_len) == (True, 3)
