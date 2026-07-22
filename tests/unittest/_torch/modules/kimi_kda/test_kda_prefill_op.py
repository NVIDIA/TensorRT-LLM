# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the optimized Kimi K3 KDA prefill op."""

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention  # noqa: E402

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
    reference = KimiKDALinearAttention(**common, use_optimized_prefill=False).to("cuda")
    reference.load_state_dict(optimized.state_dict())

    assert optimized.prefill_kernel_path == "optimized"
    assert reference.prefill_kernel_path == "fla"
    assert reference.decode_kernel_path == optimized.decode_kernel_path
    return optimized, reference


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
def test_optimized_prefill_matches_fla_reference() -> None:
    torch.manual_seed(0)
    optimized, reference = _make_attention_pair()

    for batch_size, sequence_length in [(2, 256), (1, 1024)]:
        hidden_states = (
            torch.randn(
                batch_size,
                sequence_length,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        actual = optimized.forward_prefill(hidden_states)
        expected = reference.forward_prefill(hidden_states)
        _assert_close(actual, expected)

    hidden_states = torch.randn(1, 300, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    actual = optimized.forward_prefill(hidden_states)
    expected = reference.forward_prefill(hidden_states)
    _assert_close(actual, expected)

    sequence_lengths = [128, 256, 192]
    cumulative_lengths = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.long,
        device="cuda",
    )
    hidden_states = (
        torch.randn(
            1,
            sum(sequence_lengths),
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    actual = optimized.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
    expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
    _assert_close(actual, expected)
    assert optimized.prefill_kernel_source()
