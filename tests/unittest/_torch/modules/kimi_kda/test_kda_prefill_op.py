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


@torch.no_grad()
def test_kda_prefill_op_empty_token_batch():
    """T=0 call: no output rows, recurrent state passes through unchanged.

    The runtime can emit a context batch with an empty token payload
    (observed under the overlap scheduler + logprobs flows). The op used
    to raise ``RuntimeError: step must be nonzero`` from its
    ``arange(step=T)`` buffer setup; the FLA fallback tolerates the call.
    """
    num_heads, head_k, head_v = 4, 128, 128
    q = torch.empty(1, 0, num_heads, head_k, dtype=torch.bfloat16, device="cuda")
    k = torch.empty_like(q)
    g = torch.empty(1, 0, num_heads, head_k, dtype=torch.float32, device="cuda")
    v = torch.empty(1, 0, num_heads, head_v, dtype=torch.bfloat16, device="cuda")
    beta = torch.empty(1, 0, num_heads, dtype=torch.float32, device="cuda")
    initial_state = torch.randn(
        1, num_heads, head_k, head_v, dtype=torch.float32, device="cuda")

    output, final_state = torch.ops.trtllm.kda_prefill(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=head_k**-0.5,
        initial_state=initial_state,
        output_final_state=True,
    )
    assert output.shape == (1, 0, num_heads, head_v)
    torch.testing.assert_close(final_state, initial_state)
    # State must not alias the input (the caller copies it back into the
    # pool the initial state may be a view of).
    assert final_state.data_ptr() != initial_state.data_ptr()

    _, final_state_zero = torch.ops.trtllm.kda_prefill(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=head_k**-0.5,
        initial_state=None,
        output_final_state=True,
    )
    torch.testing.assert_close(final_state_zero, torch.zeros_like(initial_state))


@torch.no_grad()
def test_kda_prefill_op_empty_token_batch_variants():
    """Regression coverage for the T=0 guard's other entry shapes.

    - varlen (cu_seqlens present): n_seqs derives from cu_seqlens, not B
    - output_final_state=False: the op's empty-tensor final-state fallback
    - use_fused_k1234=True: the guard must precede the fused-path branch
    """
    num_heads, head_k, head_v = 4, 128, 128
    q = torch.empty(1, 0, num_heads, head_k, dtype=torch.bfloat16, device="cuda")
    k = torch.empty_like(q)
    g = torch.empty(1, 0, num_heads, head_k, dtype=torch.float32, device="cuda")
    v = torch.empty(1, 0, num_heads, head_v, dtype=torch.bfloat16, device="cuda")
    beta = torch.empty(1, 0, num_heads, dtype=torch.float32, device="cuda")
    common = dict(q=q, k=k, v=v, g=g, beta=beta, scale=head_k**-0.5)

    # Varlen: two zero-length sequences -> final_state per sequence.
    cu_seqlens = torch.tensor([0, 0, 0], dtype=torch.long, device="cuda")
    _, final_state = torch.ops.trtllm.kda_prefill(
        **common, initial_state=None, output_final_state=True,
        cu_seqlens=cu_seqlens)
    assert final_state.shape == (2, num_heads, head_k, head_v)
    assert (final_state == 0).all()

    # output_final_state=False: op returns the empty placeholder tensor.
    output, final_state = torch.ops.trtllm.kda_prefill(
        **common, initial_state=None, output_final_state=False)
    assert output.shape == (1, 0, num_heads, head_v)
    assert final_state.numel() == 0

    # Fused path: the guard must fire before _launch_fused_k1234.
    output, final_state = torch.ops.trtllm.kda_prefill(
        **common, initial_state=None, output_final_state=True,
        use_fused_k1234=True)
    assert output.shape == (1, 0, num_heads, head_v)
    assert final_state.shape == (1, num_heads, head_k, head_v)


@torch.no_grad()
def test_kda_mixer_empty_prefill():
    """Runtime-shaped regression: the mixer dispatch with an empty token
    payload (the crashing call shape from jobs 2654867/2655260) must run
    end-to-end on the optimized path."""
    optimized, _ = _make_attention_pair()
    hidden_states = torch.empty(1, 0, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    out = optimized.forward_prefill(hidden_states)
    assert out.shape == (1, 0, HIDDEN_SIZE)


@torch.no_grad()
def test_kda_prefill_op_small_varlen_batch():
    """Small varlen batches (short-prompt contexts) through the dispatch.

    The persistent K123 scheduler needs >= 4 total chunks, so the dispatch
    routes NT < 4 batches to the FLA path ([6,12], [1,2,3], [30] here);
    the [6,12,20,25] case carries exactly NT=4 with total T < 64, running
    the optimized masked path where building the eqlen chunk-offset
    scratch used to raise ``step must be nonzero`` (arange step
    ``T // 64 == 0``) — its output is parity-checked against FLA.
    """
    optimized, reference = _make_attention_pair()
    for sequence_lengths in ([6, 12], [1, 2, 3], [30], [6, 12, 20, 25]):
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
