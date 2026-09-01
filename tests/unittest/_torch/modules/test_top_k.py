# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the reusable sparse index-selection Top-K module."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
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
    buffers = Mock()
    buffers.get_buffer.side_effect = [radix_indices, radix_values]
    monkeypatch.setattr(TopK, "_memory_buffers", buffers)
    top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=logical_lengths,
        scan_lengths=scan_lengths,
        next_n=2,
    )
    cute_dsl.assert_not_called()
    assert buffers.get_buffer.call_args_list == [
        call(
            (2, 10, 2),
            dtype=torch.int32,
            buffer_name="top_k_radix_indices_workspace_cpu",
            reserve_buffer=False,
        ),
        call(
            (2, 10, 2),
            dtype=torch.float32,
            buffer_name="top_k_radix_values_workspace_cpu",
            reserve_buffer=False,
        ),
    ]
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


def test_gvr_uses_caller_prior_state(monkeypatch) -> None:
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
    prior_indices = torch.zeros(1, 2, dtype=torch.int32)
    gvr.side_effect = lambda *args, **kwargs: output.copy_(torch.tensor([[5, 3]]))

    top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=logical_lengths,
        scan_lengths=scan_lengths,
        next_n=1,
        max_seq_len=16,
        gvr_ext_kwargs={"gvr_prior_indices": prior_indices},
    )

    args, kwargs = gvr.call_args
    assert args[0] is scores
    assert args[1] is prior_indices
    assert args[2] is logical_lengths
    assert args[3] is output
    assert args[4] == 2
    assert kwargs == {
        "next_n": 1,
        "compress_ratio": 4,
        "max_seq_len": 16,
        "order_row": None,
    }
    assert prior_indices.tolist() == [[0, 0]]


def test_gvr_uses_caller_prepared_row_order(monkeypatch) -> None:
    gvr = Mock(side_effect=lambda *args, **kwargs: args[3].zero_())
    monkeypatch.setattr(torch.ops.trtllm, "cute_dsl_gvr_topk_decode", gvr)
    top_k = TopK(2, decode_implementation=TopKImplementation.CUTE_DSL_GVR)
    next_n = 2
    lengths = torch.tensor([4, 1, 8, 2], dtype=torch.int32)
    row_order = torch.tensor([2, 0, 3, 1], dtype=torch.int32)

    top_k(
        torch.randn(lengths.shape[0] * next_n, 8),
        torch.empty(lengths.shape[0] * next_n, 2, dtype=torch.int32),
        is_prefill=False,
        sequence_lengths=lengths,
        scan_lengths=lengths,
        next_n=next_n,
        max_seq_len=8,
        gvr_ext_kwargs={
            "gvr_prior_indices": torch.zeros(lengths.shape[0], 2, dtype=torch.int32),
            "gvr_row_order": row_order,
        },
    )

    assert gvr.call_args.kwargs["order_row"] is row_order


def _install_fake_selfsampling_runner(monkeypatch) -> Mock:
    """Replace the lazily imported self-sampling varlen entry with a Mock."""
    runner = Mock()
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k",
        SimpleNamespace(selfsampling_topk_run_varlen=runner),
    )
    return runner


def _run_gvr_v2_decode(top_k: TopK, out_width: int = 2) -> None:
    scores = torch.randn(1, 8)  # satisfies the V2 hardware-format gate
    top_k(
        scores,
        torch.empty(1, out_width, dtype=torch.int32),
        is_prefill=False,
        sequence_lengths=torch.tensor([32], dtype=torch.int32),
        scan_lengths=torch.tensor([8], dtype=torch.int32),
        next_n=1,
        max_seq_len=16,
    )


def test_gvr_v2_decode_is_hint_free(monkeypatch) -> None:
    runner = _install_fake_selfsampling_runner(monkeypatch)
    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR_V2,
        compress_ratio=4,
    )

    _run_gvr_v2_decode(top_k)

    args, kwargs = runner.call_args
    assert len(args) == 3
    assert args[0].shape == (1, 8)
    assert args[1].tolist() == [32]
    assert args[2].shape == (1, 2)
    assert kwargs == {"next_n": 1, "compress_ratio": 4, "max_seq_len": 64}
    assert not top_k.needs_gvr_prior


def test_gvr_v2_hardware_gate_falls_back_without_prior(monkeypatch) -> None:
    runner = _install_fake_selfsampling_runner(monkeypatch)
    decode = Mock()
    monkeypatch.setattr(torch.ops.trtllm, "indexer_topk_decode", decode)
    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR_V2,
        compress_ratio=4,
    )
    scores = torch.randn(1, 8, dtype=torch.bfloat16)
    lengths = torch.tensor([32], dtype=torch.int32)
    output = torch.empty(1, 2, dtype=torch.int32)

    top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=lengths,
        scan_lengths=torch.tensor([8], dtype=torch.int32),
        max_seq_len=16,
    )

    runner.assert_not_called()
    decode.assert_called_once_with(
        scores,
        lengths,
        output,
        1,
        2,
        pre_idx=None,
        heuristic_scratch=None,
        compress_ratio=4,
        radix_aux_indices=None,
        radix_aux_logits=None,
    )


def test_gvr_v2_decode_rejects_output_width_mismatch(monkeypatch) -> None:
    """Hint-free k derives from the output width; a scratch wider than
    top_k must be rejected before launch, not silently become the k."""
    runner = _install_fake_selfsampling_runner(monkeypatch)
    top_k = TopK(
        2,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR_V2,
        compress_ratio=4,
    )
    with pytest.raises(AssertionError):
        _run_gvr_v2_decode(top_k, out_width=3)

    runner.assert_not_called()


def test_update_gvr_prior_from_prefill_uses_last_request_rows() -> None:
    top_k = TopK(2, decode_implementation=TopKImplementation.CUTE_DSL_GVR)
    prefill_indices = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
    prior_indices = torch.zeros(3, 2, dtype=torch.int32)

    top_k.update_gvr_prior_from_prefill(
        prefill_indices,
        torch.tensor([2, 1], dtype=torch.int32),
        prior_indices,
        request_offset=1,
    )

    assert prior_indices.tolist() == [[0, 0], [2, 3], [4, 5]]
    assert top_k.needs_gvr_prior


def test_gvr_v2_does_not_update_prior_from_prefill() -> None:
    top_k = TopK(2, decode_implementation=TopKImplementation.CUTE_DSL_GVR_V2)
    prior_indices = torch.zeros(1, 2, dtype=torch.int32)

    top_k.update_gvr_prior_from_prefill(
        torch.tensor([[4, 5]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        prior_indices,
    )

    assert prior_indices.tolist() == [[0, 0]]
    assert not top_k.needs_gvr_prior


def test_cuda_radix_defaults_dispatch_to_cpp(monkeypatch) -> None:
    decode = Mock()
    monkeypatch.setattr(torch.ops.trtllm, "indexer_topk_decode", decode)
    top_k = TopK(1)
    scores = torch.randn(1, 8)
    lengths = torch.tensor([8], dtype=torch.int32)
    output = torch.empty((1, 1), dtype=torch.int32)
    radix_indices = torch.empty(1, 10, 1, dtype=torch.int32)
    radix_values = torch.empty(1, 10, 1)
    buffers = Mock()
    buffers.get_buffer.side_effect = [radix_indices, radix_values]
    monkeypatch.setattr(TopK, "_memory_buffers", buffers)

    result = top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=lengths,
        scan_lengths=lengths,
    )

    assert top_k.prefill_implementation == TopKImplementation.CUDA_RADIX
    assert top_k.decode_implementation == TopKImplementation.CUDA_RADIX
    assert result is output
    assert buffers.get_buffer.call_count == 2
    decode.assert_called_once_with(
        scores,
        lengths,
        output,
        1,
        1,
        pre_idx=None,
        heuristic_scratch=None,
        compress_ratio=1,
        radix_aux_indices=radix_indices,
        radix_aux_logits=radix_values,
    )


def test_cuda_gvr_reserves_workspace_during_capture(monkeypatch) -> None:
    decode = Mock(side_effect=lambda *args, **kwargs: args[2].copy_(torch.tensor([[3, 1]])))
    monkeypatch.setattr(torch.ops.trtllm, "indexer_topk_decode", decode)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", Mock(return_value=True))
    device_context = Mock(side_effect=lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "device", device_context)

    top_k = TopK(2, decode_implementation=TopKImplementation.CUDA_GVR)
    scores = Mock(
        shape=(1, 8),
        dtype=torch.float32,
        is_cuda=True,
        device=torch.device("cuda", 3),
    )
    lengths = torch.tensor([8], dtype=torch.int32)
    output = torch.empty(1, 2, dtype=torch.int32)
    radix_indices = torch.empty(1, 10, 2, dtype=torch.int32)
    radix_values = torch.empty(1, 10, 2)
    workspace = torch.empty(1, 2)
    prior_indices = torch.zeros(1, 2, dtype=torch.int32)
    buffers = Mock()
    buffers.get_buffer.side_effect = [workspace, radix_indices, radix_values]
    monkeypatch.setattr(TopK, "_memory_buffers", buffers)

    top_k(
        scores,
        output,
        is_prefill=False,
        sequence_lengths=lengths,
        scan_lengths=lengths,
        gvr_ext_kwargs={"gvr_prior_indices": prior_indices},
    )

    assert buffers.get_buffer.call_args_list == [
        call(
            (scores.shape[0], 2),
            dtype=scores.dtype,
            buffer_name="top_k_cuda_gvr_workspace_cuda:3",
            reserve_buffer=True,
        ),
        call(
            (scores.shape[0], 10, 2),
            dtype=torch.int32,
            buffer_name="top_k_radix_indices_workspace_cuda:3",
            reserve_buffer=True,
        ),
        call(
            (scores.shape[0], 10, 2),
            dtype=torch.float32,
            buffer_name="top_k_radix_values_workspace_cuda:3",
            reserve_buffer=True,
        ),
    ]
    assert device_context.call_args_list == [call(scores.device)] * 3
    runtime_call = decode.call_args_list[-1]
    assert runtime_call.kwargs["pre_idx"] is prior_indices
    assert runtime_call.kwargs["heuristic_scratch"].data_ptr() == workspace.data_ptr()
    assert runtime_call.kwargs["radix_aux_indices"] is radix_indices
    assert runtime_call.kwargs["radix_aux_logits"] is radix_values
    assert prior_indices.tolist() == [[0, 0]]


def test_unsupported_prefill_implementation_raises() -> None:
    top_k = TopK(1, prefill_implementation=TopKImplementation.CUTE_DSL_RADIX)

    with pytest.raises(NotImplementedError, match="does not support prefill Top-K"):
        top_k(
            torch.ones(1, 1),
            torch.empty(1, 1, dtype=torch.int32),
            is_prefill=True,
            row_starts=torch.zeros(1, dtype=torch.int32),
            row_ends=torch.ones(1, dtype=torch.int32),
        )
