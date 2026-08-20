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

    # Keep B=2 across a T transition: eqlen mBeta/mAqk/mAkk batch strides
    # depend on T and therefore require distinct compiled kernel variants.
    for batch_size, sequence_length in [(2, 256), (2, 512), (1, 1024)]:
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
    initial_state = torch.randn(1, num_heads, head_k, head_v, dtype=torch.float32, device="cuda")

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
        **common, initial_state=None, output_final_state=True, cu_seqlens=cu_seqlens
    )
    assert final_state.shape == (2, num_heads, head_k, head_v)
    assert (final_state == 0).all()

    # output_final_state=False: op returns the empty placeholder tensor.
    output, final_state = torch.ops.trtllm.kda_prefill(
        **common, initial_state=None, output_final_state=False
    )
    assert output.shape == (1, 0, num_heads, head_v)
    assert final_state.numel() == 0

    # Fused path: the guard must fire before _launch_fused_k1234.
    output, final_state = torch.ops.trtllm.kda_prefill(
        **common, initial_state=None, output_final_state=True, use_fused_k1234=True
    )
    assert output.shape == (1, 0, num_heads, head_v)
    assert final_state.shape == (1, num_heads, head_k, head_v)


@torch.no_grad()
def test_kda_mixer_empty_prefill():
    """Runtime-shaped regression: the mixer dispatch with an empty token
    payload (a crashing call shape observed at runtime) must run
    end-to-end on the optimized path."""
    optimized, _ = _make_attention_pair()
    hidden_states = torch.empty(1, 0, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    out = optimized.forward_prefill(hidden_states)
    assert out.shape == (1, 0, HIDDEN_SIZE)


@torch.no_grad()
def test_kda_prefill_op_partial_final_chunk_large_batch():
    """Regression: varlen batches whose FINAL chunk is partial.

    The chunk-tile kernels access the full 64-row tile of every chunk and
    neutralize invalid rows only after the access, so the batch's final
    partial chunk touches up to 63 rows past the logical packed length —
    OOB reads on the beta input (now bounds-checked in fused_k123) and on
    the A_kk/A_qk scratch (now allocated with one chunk of slack). The
    runtime's autotuner-warmup shape [max_seq_len - 1, 1] = [8191, 1] hit
    this as CUDA_ERROR_ILLEGAL_ADDRESS whenever the following page was
    unmapped.

    - [8191, 1]: the exact autotuner-warmup composition (one max_seq_len-1
      context plus a 1-token remainder).
    - [8000, 150, 42]: interior partial chunks (cross-sequence rows) plus
      a partial final chunk, at eval-like scale.
    """
    optimized, reference = _make_attention_pair()
    for sequence_lengths in ([8191, 1], [8000, 150, 42]):
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


@torch.no_grad()
def test_kda_prefill_op_shape_growth_and_cu_dtype_transitions():
    """Cross-call transitions through one process's compile caches.

    Regression for the cu/ci-dtype cache-key bug: the K123/akk_inv compile
    caches were keyed shape-independently but NOT on the cu_seqlens /
    chunk_indices dtype, while the compiled kernels bake the element type
    (int64 reads use stride 8, int32 stride 4). Reusing an int64-compiled
    kernel on int32 cu/ci misaddressed every cu/ci element — garbage seq
    ids / chunk starts -> cudaErrorIllegalAddress on the first call after
    the flip (memcheck: 4-byte read one element past the 2-entry int32 cu);
    the reverse direction (int32-compiled, int64 passed) corrupted
    silently. Shape growth alone (same dtype) was already sound.

    The sequence below covers, in one process: buffer-cache growth
    (T 1171 -> 8191), int64 -> int32 flip on the grown shape, shrink with
    a flip back, and a multi-seq int32 batch. Every call is parity-checked
    against FLA (catches the silent-corruption direction too).
    """
    torch.manual_seed(0)
    optimized, reference = _make_attention_pair()
    cases = [
        ([517, 654], torch.long),  # small batch, int64 cu (dump-replay-like)
        ([8191], torch.int32),  # buffer growth + dtype flip (crashed pre-fix)
        ([1171], torch.long),  # shrink + flip back (silent corruption pre-fix)
        ([150, 900, 333, 640], torch.int32),  # multi-seq int32
    ]
    for sequence_lengths, cu_dtype in cases:
        cumulative_lengths = torch.tensor(
            [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
            dtype=cu_dtype,
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
