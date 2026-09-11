# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners import common
from tensorrt_llm._torch.pyexecutor.engine.runners.common import (
    apply_position_id_offset,
    get_all_rank_num_tokens,
    get_padding_params,
    get_top_level_model,
    prepare_multimodal_indices,
    set_spec_metadata_all_rank_num_tokens,
    ship_multimodal_indices,
)
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

pytestmark = pytest.mark.cpu_only


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
        SimpleNamespace(num_tokens=5),
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
        SimpleNamespace(num_tokens=5),
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


def test_set_spec_metadata_all_rank_counts_for_one_model() -> None:
    mode = SimpleNamespace(
        is_mtp_eagle_one_model=Mock(return_value=True),
        is_eagle3_one_model=Mock(return_value=False),
    )
    spec_metadata = SimpleNamespace(spec_dec_mode=mode)

    set_spec_metadata_all_rank_num_tokens(
        spec_metadata,
        [8, 9],
        [2, 3],
        [1, 2],
    )

    assert spec_metadata.all_rank_num_tokens == [8, 9]
    assert spec_metadata.all_rank_num_seqs == [2, 3]
    assert spec_metadata.all_rank_num_gens == [1, 2]
    assert spec_metadata.subseq_all_rank_num_tokens == [2, 3]


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
