# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the optimized Kimi K3 KDA decode op."""

import copy

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import (  # noqa: E402
    KimiKDACachedState,
    KimiKDALinearAttention,
)

# 73: deliberately odd and > 64 to cover non-power-of-two batched decode.
BATCH_SIZE = 73
NUM_HEADS = 96
HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
HIDDEN_SIZE = 7168


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in {(10, 0), (10, 3)}


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 is supported only on Blackwell (SM100/SM103)",
)


def _make_attention_pair() -> tuple[KimiKDALinearAttention, KimiKDALinearAttention]:
    common = {
        "hidden_size": HIDDEN_SIZE,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
        "rms_norm_eps": 1e-5,
        "dtype": torch.bfloat16,
    }
    optimized = KimiKDALinearAttention(**common).to("cuda")
    reference = KimiKDALinearAttention(**common, use_optimized_decode=False).to("cuda")
    reference.load_state_dict(optimized.state_dict())

    assert optimized.decode_kernel_path == "optimized"
    assert reference.decode_kernel_path == "fla"
    assert reference.prefill_kernel_path == optimized.prefill_kernel_path
    return optimized, reference


def _make_cache() -> KimiKDACachedState:
    projection_size = NUM_HEADS * HEAD_DIM
    return KimiKDACachedState(
        conv_state_q=(
            torch.randn(
                BATCH_SIZE,
                projection_size,
                CONV_KERNEL_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        ),
        conv_state_k=(
            torch.randn(
                BATCH_SIZE,
                projection_size,
                CONV_KERNEL_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        ),
        conv_state_v=(
            torch.randn(
                BATCH_SIZE,
                projection_size,
                CONV_KERNEL_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        ),
        recurrent_state=(
            torch.randn(
                BATCH_SIZE,
                NUM_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                dtype=torch.float32,
                device="cuda",
            )
            * 0.05
        ),
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_float = actual.float()
    expected_float = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    ).item()
    relative_l2 = ((actual_float - expected_float).norm() / (expected_float.norm() + 1e-12)).item()
    assert cosine > 0.999
    assert relative_l2 < 3e-2


@torch.no_grad()
def test_optimized_decode_matches_fla_reference() -> None:
    torch.manual_seed(0)
    optimized, reference = _make_attention_pair()
    hidden_states = (
        torch.randn(
            BATCH_SIZE,
            1,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    initial_cache = _make_cache()

    actual_output, actual_cache = optimized.forward_decode(
        hidden_states, copy.deepcopy(initial_cache)
    )
    expected_output, expected_cache = reference.forward_decode(
        hidden_states, copy.deepcopy(initial_cache)
    )

    _assert_close(actual_output, expected_output)
    _assert_close(actual_cache.recurrent_state, expected_cache.recurrent_state)
    _assert_close(actual_cache.conv_state_q, expected_cache.conv_state_q)
    _assert_close(actual_cache.conv_state_k, expected_cache.conv_state_k)
    _assert_close(actual_cache.conv_state_v, expected_cache.conv_state_v)
    assert optimized.decode_kernel_source()
