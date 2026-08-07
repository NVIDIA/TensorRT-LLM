# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the reusable sparse index-selection Top-K module."""

from unittest.mock import Mock

import torch

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


def test_cute_dsl_radix_preserves_compressed_mtp_fallback(monkeypatch) -> None:
    cute_dsl = Mock()
    trtllm = Mock()
    monkeypatch.setattr(torch.ops.trtllm, "cute_dsl_indexer_topk_decode", cute_dsl)
    monkeypatch.setattr(torch.ops.trtllm, "indexer_topk_decode", trtllm)

    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_RADIX,
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


def test_gvr_owns_prior_state_and_updates_it(monkeypatch) -> None:
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
    gvr.side_effect = lambda *args, **kwargs: output.copy_(torch.tensor([[5, 3]]))

    top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=logical_lengths,
        scan_lengths=scan_lengths,
        next_n=1,
    )

    args, kwargs = gvr.call_args
    assert args[0] is scores
    assert args[1].data_ptr() == top_k._gvr_prior_indices.data_ptr()
    assert args[2] is logical_lengths
    assert args[3] is output
    assert args[4] == 2
    assert kwargs == {
        "next_n": 1,
        "compress_ratio": 4,
        "max_seq_len": 8,
        "order_row": None,
    }
    assert top_k._gvr_prior_indices.tolist() == [[5, 3]]


def test_gvr_prepares_row_order_at_threshold(monkeypatch) -> None:
    gvr = Mock(side_effect=lambda *args, **kwargs: args[3].zero_())
    monkeypatch.setattr(torch.ops.trtllm, "cute_dsl_gvr_topk_decode", gvr)
    top_k = TopK(2, decode_implementation=TopKImplementation.CUTE_DSL_GVR)
    next_n = 2
    lengths = torch.tensor([4, 1, 8, 2], dtype=torch.int32)

    top_k(
        torch.randn(lengths.shape[0] * next_n, 8),
        torch.empty(lengths.shape[0] * next_n, 2, dtype=torch.int32),
        is_prefill=False,
        sequence_lengths=lengths,
        scan_lengths=lengths,
        next_n=next_n,
    )

    row_order = gvr.call_args.kwargs["order_row"]
    assert row_order is not None
    assert row_order.tolist() == [2, 0, 3, 1]


def test_seed_from_prefill_uses_last_request_rows() -> None:
    top_k = TopK(2, decode_implementation=TopKImplementation.CUTE_DSL_GVR)
    prefill_indices = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)

    top_k.seed_from_prefill(
        prefill_indices,
        torch.tensor([2, 1], dtype=torch.int32),
        request_offset=1,
    )

    assert top_k._gvr_prior_indices.tolist() == [[0, 0], [2, 3], [4, 5]]


def test_gvr_state_buffers_are_registered_during_init() -> None:
    top_k = TopK(2, decode_implementation=TopKImplementation.CUDA_GVR)

    buffers = dict(top_k.named_buffers())
    assert set(buffers) == {
        "_gvr_prior_indices",
        "_cuda_gvr_scratch",
        "_gvr_row_order",
    }
    assert all(buffer.numel() == 0 for buffer in buffers.values())


def test_cuda_gvr_warmup_calls_cpp_op(monkeypatch) -> None:
    from tensorrt_llm._torch.custom_ops import cpp_custom_ops

    decode = Mock()
    synchronize = Mock()
    torch_empty = torch.empty
    torch_tensor = torch.tensor
    torch_zeros = torch.zeros
    monkeypatch.setattr(torch.ops.trtllm, "indexer_topk_decode", decode)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(
        torch,
        "empty",
        lambda shape, *, dtype, device: torch_empty(shape, dtype=dtype),
    )
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda data, *, dtype, device: torch_tensor(data, dtype=dtype),
    )
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda shape, *, dtype, device: torch_zeros(shape, dtype=dtype),
    )

    cpp_custom_ops.warmup_cuda_gvr_topk_decode(top_k=512)

    args = decode.call_args.args
    kwargs = decode.call_args.kwargs
    assert args[0].shape == (1, 4096)
    assert args[2].shape == (1, 512)
    assert kwargs["pre_idx"].shape == (1, 512)
    assert kwargs["heuristic_scratch"].shape == (1, 512)
    assert kwargs["radix_aux_indices"].shape == (1, 10, 512)
    assert kwargs["radix_aux_logits"].shape == (1, 10, 512)
    synchronize.assert_called_once_with()


def test_implementations_are_named_by_backend_and_algorithm() -> None:
    assert {implementation.value for implementation in TopKImplementation} == {
        "torch",
        "cuda_radix",
        "cute_dsl_radix",
        "cuda_gvr",
        "cute_dsl_gvr",
    }


def test_none_implementations_use_cuda_radix_defaults() -> None:
    top_k = TopK(1)

    assert top_k.prefill_implementation == TopKImplementation.CUDA_RADIX
    assert top_k.decode_implementation == TopKImplementation.CUDA_RADIX


def test_cuda_gvr_owns_scratch_and_updates_prior(monkeypatch) -> None:
    decode = Mock(side_effect=lambda *args, **kwargs: args[2].copy_(torch.tensor([[3, 1]])))
    monkeypatch.setattr(torch.ops.trtllm, "indexer_topk_decode", decode)

    top_k = TopK(2, decode_implementation=TopKImplementation.CUDA_GVR)
    scores = torch.randn(1, 8)
    lengths = torch.tensor([8], dtype=torch.int32)
    output = torch.empty(1, 2, dtype=torch.int32)
    radix_indices = torch.empty(2, 10, 2, dtype=torch.int32)
    radix_values = torch.empty(2, 10, 2)

    top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=lengths,
        scan_lengths=lengths,
        radix_indices=radix_indices,
        radix_values=radix_values,
    )

    runtime_call = decode.call_args_list[-1]
    assert runtime_call.kwargs["pre_idx"].data_ptr() == top_k._gvr_prior_indices.data_ptr()
    assert runtime_call.kwargs["heuristic_scratch"].data_ptr() == top_k._cuda_gvr_scratch.data_ptr()
    assert top_k._gvr_prior_indices.tolist() == [[3, 1], [0, 0]]
