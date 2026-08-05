# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the reusable sparse index-selection Top-K module."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.modules import top_k as top_k_module
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation


def test_prefill_torch_masks_dirty_scores_and_pads_output() -> None:
    scores = torch.tensor(
        [
            [1000.0, 9.0, 8.0, 7.0, 1000.0, 1000.0],
            [1000.0, 1000.0, 3.0, 1000.0, 1000.0, 1000.0],
            [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        ]
    )
    row_starts = torch.tensor([1, 2, 4], dtype=torch.int32)
    row_ends = torch.tensor([4, 3, 4], dtype=torch.int32)
    output = torch.full((3, 4), 77, dtype=torch.int32)

    result = TopK(4, prefill_implementation=TopKImplementation.TORCH)(
        scores,
        output,
        is_prefill=True,
        row_starts=row_starts,
        row_ends=row_ends,
    )

    assert result is output
    assert output.tolist() == [[0, 1, 2, -1], [0, -1, -1, -1], [-1, -1, -1, -1]]


def test_decode_torch_uses_scan_lengths() -> None:
    scores = torch.tensor(
        [
            [1.0, 2.0, 3.0, 1000.0, 1000.0, 1000.0],
            [1.0, 2.0, 3.0, 4.0, 1000.0, 1000.0],
        ]
    )
    logical_lengths = torch.tensor([16], dtype=torch.int32)
    scan_lengths = torch.tensor([3], dtype=torch.int32)
    output = torch.full((2, 4), 77, dtype=torch.int32)

    result = TopK(
        4,
        decode_implementation=TopKImplementation.TORCH,
        compress_ratio=4,
    )(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=logical_lengths,
        scan_lengths=scan_lengths,
        next_n=2,
    )

    assert result is output
    assert output.tolist() == [[1, 0, -1, -1], [2, 1, 0, -1]]


def test_one_module_dispatches_prefill_and_decode() -> None:
    top_k = TopK(
        1,
        prefill_implementation=TopKImplementation.TORCH,
        decode_implementation=TopKImplementation.TORCH,
    )
    scores = torch.tensor([[1.0, 3.0, 2.0]])
    output = torch.empty(1, 1, dtype=torch.int32)

    top_k(
        scores,
        output,
        is_prefill=True,
        row_starts=torch.tensor([1], dtype=torch.int32),
        row_ends=torch.tensor([3], dtype=torch.int32),
    )
    assert output.item() == 0

    top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=torch.tensor([3], dtype=torch.int32),
        scan_lengths=torch.tensor([3], dtype=torch.int32),
    )
    assert output.item() == 1


def test_cute_dsl_preferred_preserves_compressed_mtp_fallback(monkeypatch) -> None:
    cute_dsl = Mock()
    trtllm = Mock()
    monkeypatch.setattr(torch.ops.trtllm, "cute_dsl_indexer_topk_decode", cute_dsl)
    monkeypatch.setattr(torch.ops.trtllm, "indexer_topk_decode", trtllm)

    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_PREFERRED,
        compress_ratio=4,
    )
    logical_lengths = torch.tensor([16], dtype=torch.int32)
    scan_lengths = torch.tensor([4], dtype=torch.int32)

    scores = torch.randn(1, 4)
    output = torch.empty(1, 2, dtype=torch.int32)
    top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=logical_lengths,
        scan_lengths=scan_lengths,
        next_n=1,
    )
    cute_dsl.assert_called_once_with(scores, scan_lengths, output, 2, 1)
    trtllm.assert_not_called()

    cute_dsl.reset_mock()
    scores = torch.randn(2, 4)
    output = torch.empty(2, 2, dtype=torch.int32)
    radix_indices = torch.empty(2, 10, 2, dtype=torch.int32)
    radix_values = torch.empty(2, 10, 2)
    top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=logical_lengths,
        scan_lengths=scan_lengths,
        next_n=2,
        radix_indices=radix_indices,
        radix_values=radix_values,
    )
    cute_dsl.assert_not_called()
    trtllm.assert_called_once_with(
        scores,
        logical_lengths,
        output,
        2,
        2,
        pre_idx=None,
        heuristic_scratch=None,
        compress_ratio=4,
        radix_aux_indices=radix_indices,
        radix_aux_logits=radix_values,
    )


def test_gvr_routes_logical_lengths_and_workspace(monkeypatch) -> None:
    gvr = Mock()
    monkeypatch.setattr(torch.ops.trtllm, "cute_dsl_gvr_topk_decode", gvr)
    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
        compress_ratio=4,
    )
    scores = torch.randn(1, 8)
    logical_lengths = torch.tensor([32], dtype=torch.int32)
    scan_lengths = torch.tensor([8], dtype=torch.int32)
    output = torch.empty(1, 2, dtype=torch.int32)
    prior = torch.zeros(1, 2, dtype=torch.int32)
    row_order = torch.zeros(1, dtype=torch.int32)

    top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=logical_lengths,
        scan_lengths=scan_lengths,
        next_n=1,
        prior_indices=prior,
        max_num_columns=8,
        row_order=row_order,
    )

    gvr.assert_called_once_with(
        scores,
        prior,
        logical_lengths,
        output,
        2,
        next_n=1,
        compress_ratio=4,
        max_seq_len=8,
        order_row=row_order,
    )


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_must_be_positive(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        TopK(top_k, prefill_implementation=TopKImplementation.TORCH)


def test_top_k_requires_an_implementation() -> None:
    with pytest.raises(ValueError, match="at least one Top-K implementation"):
        TopK(1)


def test_decode_validates_workspace_pairs() -> None:
    top_k = TopK(2, decode_implementation=TopKImplementation.TRTLLM)
    with pytest.raises(ValueError, match="must be provided together"):
        top_k(
            torch.randn(1, 4),
            torch.empty(1, 2, dtype=torch.int32),
            is_prefill=False,
            sequence_lengths=torch.tensor([4], dtype=torch.int32),
            scan_lengths=torch.tensor([4], dtype=torch.int32),
            next_n=1,
            radix_indices=torch.empty(1, 10, 2, dtype=torch.int32),
        )


def test_prepare_deduplicates_success_for_full_key(monkeypatch) -> None:
    prepared: set[tuple[object, ...]] = set()
    monkeypatch.setattr(top_k_module, "_PREPARED_DECODE_TOP_K", prepared)
    monkeypatch.setattr(top_k_module, "_cuda_device", lambda _: torch.device("cuda:3"))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())

    top_k = TopK(32, decode_implementation=TopKImplementation.TRTLLM_HEURISTIC)
    warmup = Mock()
    monkeypatch.setattr(top_k, "_warmup_trtllm_heuristic", warmup)

    prepare_args = dict(
        device=torch.device("cuda:3"),
        max_num_columns=4096,
        next_n=1,
        input_dtype=torch.bfloat16,
        num_sms=148,
    )
    top_k.prepare(**prepare_args)
    top_k.prepare(**prepare_args)

    warmup.assert_called_once_with(torch.bfloat16)
    assert len(prepared) == 1


def test_prepare_does_not_cache_failure(monkeypatch) -> None:
    prepared: set[tuple[object, ...]] = set()
    monkeypatch.setattr(top_k_module, "_PREPARED_DECODE_TOP_K", prepared)
    monkeypatch.setattr(top_k_module, "_cuda_device", lambda _: torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())

    top_k = TopK(16, decode_implementation=TopKImplementation.TRTLLM_HEURISTIC)
    warmup = Mock(side_effect=RuntimeError("warmup failed"))
    monkeypatch.setattr(top_k, "_warmup_trtllm_heuristic", warmup)
    prepare_args = dict(
        device=torch.device("cuda:0"),
        max_num_columns=1024,
        next_n=1,
        input_dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        top_k.prepare(**prepare_args)
    assert not prepared

    warmup.side_effect = None
    top_k.prepare(**prepare_args)
    assert warmup.call_count == 2
    assert len(prepared) == 1
