# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the optimized Kimi K3 KDA prefill op."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda import (
    KimiKDALinearAttention,  # noqa: E402
    _kda_kernels,  # noqa: E402
)
from tests.unittest._torch.modules.kimi_kda.kimi_kda_test_utils import (  # noqa: E402
    KimiKDAReference,
    get_production_prefill_kernel_path,
)

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


def _make_attention_pair(
    expected_prefill_kernel_path: str = "optimized",
) -> tuple[KimiKDALinearAttention, KimiKDAReference]:
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
    cfg = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        rms_norm_eps=common["rms_norm_eps"],
        linear_attn_config={
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "short_conv_kernel_size": CONV_KERNEL_SIZE,
            "use_full_rank_gate": common["use_full_rank_gate"],
            "gate_lower_bound": common["gate_lower_bound"],
        },
    )
    optimized = KimiKDALinearAttention(cfg, layer_idx=0).to("cuda")
    with torch.no_grad():
        optimized.dt_bias.zero_()
    reference = KimiKDAReference(**common).to("cuda")
    reference.load_state_dict(optimized.state_dict())
    optimized.finalize_decode_weights()

    assert get_production_prefill_kernel_path(optimized) == expected_prefill_kernel_path
    return optimized, reference


def _run_production_prefill(
    attention: KimiKDALinearAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    *,
    conv_pool: torch.Tensor | None = None,
    state_pool: torch.Tensor | None = None,
    slot_indices: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, sequence_length, _ = hidden_states.shape
    if cu_seqlens is None:
        cu_seqlens = torch.arange(batch_size + 1, device="cuda", dtype=torch.long) * sequence_length
    else:
        batch_size = cu_seqlens.numel() - 1

    projection_size = NUM_HEADS * HEAD_DIM
    if conv_pool is None:
        conv_pool = torch.zeros(
            batch_size,
            3 * projection_size,
            CONV_KERNEL_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
    if state_pool is None:
        state_pool = torch.zeros(
            batch_size,
            NUM_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
    if slot_indices is None:
        slot_indices = torch.arange(batch_size, device="cuda", dtype=torch.long)
    use_initial_states = has_initial_states is not None
    if has_initial_states is None:
        has_initial_states = torch.zeros(batch_size, device="cuda", dtype=torch.bool)
    metadata = SimpleNamespace(
        use_initial_states=use_initial_states,
        has_initial_states=has_initial_states,
    )
    output = attention.forward_prefill(
        hidden_states.reshape(-1, HIDDEN_SIZE),
        cu_seqlens,
        metadata,
        batch_size,
        conv_pool,
        state_pool,
        slot_indices,
    )
    return output.reshape_as(hidden_states)


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
        actual = _run_production_prefill(optimized, hidden_states)
        expected = reference.forward_prefill(hidden_states)
        _assert_close(actual, expected)

    hidden_states = torch.randn(1, 300, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    actual = _run_production_prefill(optimized, hidden_states)
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
    actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
    expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
    _assert_close(actual, expected)


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
    """The production mixer handles an empty token payload without raising."""
    optimized, _ = _make_attention_pair()
    hidden_states = torch.empty(1, 0, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    out = _run_production_prefill(optimized, hidden_states)
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
        actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
        expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
        _assert_close(actual, expected)


@torch.no_grad()
@pytest.mark.parametrize(
    "sequence_lengths",
    ([30], [65], [6, 12], [1, 2, 3], [64, 64, 63], [6, 12, 20, 25]),
    ids=(
        "one-chunk",
        "two-chunk-single-seq",
        "two-chunk-varlen",
        "three-short-seqs",
        "three-chunk-boundary",
        "four-chunk-optimized-boundary",
    ),
)
def test_kda_prefill_small_varlen_dispatch_matches_fla_reference(sequence_lengths):
    """Cover every small-varlen fallback chunk count and the optimized boundary.

    The persistent K123 scheduler needs at least four total chunks. The
    one-, two-, and three-chunk configurations must use FLA; the final case
    has exactly four chunks and verifies the optimized boundary.
    """
    optimized, reference = _make_attention_pair()
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
    actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
    expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
    _assert_close(actual, expected)


@torch.no_grad()
def test_kda_prefill_unavailable_kernel_fallback_preserves_mixed_initial_states(monkeypatch):
    """The FLA core preserves prefix and continuation pool states."""
    torch.manual_seed(3)
    optimized, _ = _make_attention_pair()
    monkeypatch.setattr(_kda_kernels, "is_intree_prefill_available", lambda: False)
    fallback, _ = _make_attention_pair(expected_prefill_kernel_path="fla")
    fallback.load_state_dict(optimized.state_dict())

    sequence_lengths = [64, 64, 64, 64]
    cumulative_lengths = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.long,
        device="cuda",
    )
    hidden_states = (
        torch.randn(1, sum(sequence_lengths), HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        * 0.05
    )
    slots = 5
    slot_indices = torch.tensor([3, 0, 4, 1], dtype=torch.long, device="cuda")
    has_initial_states = torch.tensor([True, False, True, False], device="cuda")
    projection_size = NUM_HEADS * HEAD_DIM
    conv_seed = (
        torch.randn(
            slots,
            3 * projection_size,
            CONV_KERNEL_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    state_seed = (
        torch.randn(
            slots,
            NUM_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.05
    )

    optimized_conv, optimized_state = conv_seed.clone(), state_seed.clone()
    optimized_output = _run_production_prefill(
        optimized,
        hidden_states,
        cumulative_lengths,
        conv_pool=optimized_conv,
        state_pool=optimized_state,
        slot_indices=slot_indices,
        has_initial_states=has_initial_states,
    )
    fallback_conv, fallback_state = conv_seed.clone(), state_seed.clone()
    fallback_output = _run_production_prefill(
        fallback,
        hidden_states,
        cumulative_lengths,
        conv_pool=fallback_conv,
        state_pool=fallback_state,
        slot_indices=slot_indices,
        has_initial_states=has_initial_states,
    )

    _assert_close(fallback_output, optimized_output)
    _assert_close(
        fallback_conv.index_select(0, slot_indices),
        optimized_conv.index_select(0, slot_indices),
    )
    _assert_close(
        fallback_state.index_select(0, slot_indices),
        optimized_state.index_select(0, slot_indices),
    )


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
        actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
        expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
        _assert_close(actual, expected)
