# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side contract tests for the CuTe DSL MLA FMHA library (FP8 scales, split-KV sizing)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import CuteDslMlaFmha
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE


def test_fp8_mla_scales_remain_tensors() -> None:
    """FP8 scales reach the kernel as the same device tensors, never as host floats."""
    q = torch.empty(1)
    bmm1_scale = torch.tensor([0.25, 0.25 * 1.442695], dtype=torch.float32)
    bmm2_scale = torch.tensor([4.0], dtype=torch.float32)

    actual_bmm1_scale, actual_bmm2_scale = CuteDslMlaFmha._get_fp8_scales(
        q,
        SimpleNamespace(mla_bmm1_scale=bmm1_scale, mla_bmm2_scale=bmm2_scale),
    )

    assert actual_bmm1_scale is bmm1_scale
    assert actual_bmm2_scale is bmm2_scale


@pytest.mark.parametrize(
    "bmm1_scale, bmm2_scale, match",
    [
        (None, torch.tensor([1.0]), "requires mla_bmm1_scale and mla_bmm2_scale"),
        (torch.tensor([1.0], dtype=torch.float16), torch.tensor([1.0]), "float32"),
        (torch.tensor([1.0]), torch.empty(0), "float32"),
    ],
)
def test_fp8_mla_scales_are_validated(bmm1_scale, bmm2_scale, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        CuteDslMlaFmha._get_fp8_scales(
            torch.empty(1),
            SimpleNamespace(mla_bmm1_scale=bmm1_scale, mla_bmm2_scale=bmm2_scale),
        )


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="nvidia-cutlass-dsl is not installed")
def test_fp8_mla_custom_op_accepts_tensor_scales() -> None:
    # Importing the module registers the CuTe DSL custom ops.
    import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops  # noqa: F401

    schema = torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell.default._schema
    argument_types = {argument.name: str(argument.type) for argument in schema.arguments}

    assert argument_types["softmax_scale"] == "Tensor"
    assert argument_types["output_scale"] == "Tensor"


def _mla_decode_runner_cls():
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    return CuteDSLNVMlaDecodeBlackwellRunner


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="nvidia-cutlass-dsl is not installed")
def test_mla_decode_split_kv_is_a_bounded_power_of_two() -> None:
    runner_cls = _mla_decode_runner_cls()
    max_active_blocks = 296
    previous = None
    for batch_size in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        split_kv = runner_cls.get_default_split_kv(batch_size, 1, max_active_blocks)
        assert 1 <= split_kv <= runner_cls._MAX_SPLIT_KV
        assert split_kv & (split_kv - 1) == 0
        # More requests already fill more CTAs, so the split never grows.
        if previous is not None:
            assert split_kv <= previous
        previous = split_kv
    # Persistent round-robin is never a candidate: skewed decode KV balances
    # better on the hardware scheduler.
    assert runner_cls.get_is_persistent_candidates() == [False]


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="nvidia-cutlass-dsl is not installed")
@pytest.mark.parametrize("num_heads", [8, 128])
def test_mla_decode_split_kv_workspace_rows_cover_every_bucket(num_heads: int) -> None:
    """The CUDA-graph workspace is sized once, so its row bound must cover every
    batch bucket and query length below the configured maxima, and must not
    shrink when either maximum grows."""
    runner_cls = _mla_decode_runner_cls()
    max_active_blocks = 296
    max_seq_len_q, max_batch_size = 4, 64

    rows = runner_cls._max_split_kv_rows(
        num_heads, max_seq_len_q, max_batch_size, max_active_blocks
    )
    for seq_len_q in range(1, max_seq_len_q + 1):
        folded = runner_cls.get_folded_seq_len_q(num_heads, seq_len_q)
        assert 1 <= folded <= seq_len_q
        bucket = 1
        while bucket <= max_batch_size:
            split_kv = runner_cls.get_default_split_kv(bucket, folded, max_active_blocks)
            if split_kv > 1:
                assert bucket * seq_len_q * split_kv <= rows
            bucket *= 2

    assert (
        runner_cls._max_split_kv_rows(
            num_heads, max_seq_len_q + 1, max_batch_size, max_active_blocks
        )
        >= rows
    )
    assert (
        runner_cls._max_split_kv_rows(
            num_heads, max_seq_len_q, 2 * max_batch_size, max_active_blocks
        )
        >= rows
    )
