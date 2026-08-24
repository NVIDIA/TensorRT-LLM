# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the optimized Kimi K3 KDA prefill op."""

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fla")

from fla.modules import ShortConvolution  # noqa: E402

from tensorrt_llm._torch.modules.kimi_kda import (
    KimiKDALinearAttention,  # noqa: E402
    _kda_kernels,  # noqa: E402
)
from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (  # noqa: E402
    copy_kda_replay_conv_window,
    fused_kda_post_conv,
)
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn  # noqa: E402
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


def _make_kda(
    expected_prefill_kernel_path: str = "optimized",
    source_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> KimiKDALinearAttention:
    cfg = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        rms_norm_eps=1e-5,
        linear_attn_config={
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "short_conv_kernel_size": CONV_KERNEL_SIZE,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    kda = KimiKDALinearAttention(cfg, layer_idx=0).to("cuda")
    with torch.no_grad():
        kda.dt_bias.zero_()
    if source_state_dict is not None:
        kda.load_state_dict(source_state_dict)
    kda.finalize_decode_weights()

    assert get_production_prefill_kernel_path(kda) == expected_prefill_kernel_path
    return kda


def _make_reference(source_state_dict: Mapping[str, torch.Tensor]) -> KimiKDAReference:
    reference = KimiKDAReference(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL_SIZE,
        use_full_rank_gate=True,
        gate_lower_bound=-5.0,
        rms_norm_eps=1e-5,
        dtype=torch.bfloat16,
    ).to("cuda")
    reference.load_state_dict(source_state_dict)
    return reference


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
            CONV_KERNEL_SIZE - 1,
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
        state_indices=slot_indices.to(torch.int32),
        query_start_loc=cu_seqlens.to(torch.int32),
    )
    core = attention.forward_prefill(
        hidden_states.reshape(-1, HIDDEN_SIZE),
        cu_seqlens,
        metadata,
        batch_size,
        conv_pool,
        state_pool,
        slot_indices,
    )
    output = attention._project_output(core)
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


@pytest.mark.parametrize("sequence_length", [1, 127, 256])
@torch.no_grad()
def test_fused_kda_post_conv_matches_reference(sequence_length: int) -> None:
    """Packed convolution output is normalized and transposed correctly."""
    torch.manual_seed(1)
    num_heads, head_dim = 8, 128
    projection_size = num_heads * head_dim
    packed = torch.randn(
        3 * projection_size,
        sequence_length,
        dtype=torch.bfloat16,
        device="cuda",
    )

    actual_q, actual_k, actual_v = fused_kda_post_conv(packed, num_heads, head_dim)
    unpacked = packed.view(3, projection_size, sequence_length)
    expected_q = unpacked[0].transpose(0, 1).reshape(1, sequence_length, num_heads, head_dim)
    expected_k = unpacked[1].transpose(0, 1).reshape(1, sequence_length, num_heads, head_dim)
    expected_v = unpacked[2].transpose(0, 1).reshape(1, sequence_length, num_heads, head_dim)
    expected_q = expected_q.float()
    expected_k = expected_k.float()
    expected_q *= torch.rsqrt((expected_q * expected_q).sum(dim=-1, keepdim=True) + 1e-6)
    expected_k *= torch.rsqrt((expected_k * expected_k).sum(dim=-1, keepdim=True) + 1e-6)

    assert actual_q.is_contiguous()
    assert actual_k.is_contiguous()
    assert actual_v.is_contiguous()
    torch.testing.assert_close(actual_q, expected_q.to(packed.dtype), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_k, expected_k.to(packed.dtype), rtol=1e-2, atol=1e-2)
    assert torch.equal(actual_v, expected_v)


@torch.no_grad()
def test_packed_gdn_convolution_matches_three_fla_convolutions() -> None:
    """The reused GDN kernel preserves KDA convolution and cache semantics."""
    torch.manual_seed(2)
    dim, width, slots = 256, 4, 5
    sequence_lengths = [7, 5]
    num_tokens = sum(sequence_lengths)
    state_indices = torch.tensor([4, 1], dtype=torch.int32, device="cuda")
    state_indices_long = state_indices.long()
    has_initial_state = torch.tensor([True, False], device="cuda")
    cu_seqlens = torch.tensor(
        [0, sequence_lengths[0], num_tokens], dtype=torch.int32, device="cuda"
    )
    cu_seqlens_long = cu_seqlens.long()

    projected = [
        torch.randn(1, num_tokens, dim, dtype=torch.bfloat16, device="cuda") for _ in range(3)
    ]
    convolutions = [
        ShortConvolution(dim, width, activation="silu").to(device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    packed_weight = torch.cat(
        [conv.weight.detach().squeeze(1) for conv in convolutions], dim=0
    ).contiguous()
    conv_pool = torch.randn(slots, 3 * dim, width - 1, dtype=torch.bfloat16, device="cuda")
    conv_pool_before = conv_pool.clone()

    packed = torch.cat(projected, dim=-1).squeeze(0).transpose(0, 1).contiguous()
    causal_conv1d_fn(
        packed,
        packed_weight,
        query_start_loc=cu_seqlens,
        cache_indices=state_indices,
        has_initial_state=has_initial_state,
        conv_states=conv_pool,
        activation="silu",
    )

    selected = conv_pool_before.index_select(0, state_indices_long)
    selected[~has_initial_state] = 0
    expected_outputs = []
    expected_caches = []
    for section, (projected_section, convolution) in enumerate(
        zip(projected, convolutions, strict=True)
    ):
        cache = torch.nn.functional.pad(selected[:, section * dim : (section + 1) * dim], (1, 0))
        output, final_cache = convolution(
            projected_section,
            cache=cache,
            output_final_state=True,
            cu_seqlens=cu_seqlens_long,
        )
        expected_outputs.append(output)
        expected_caches.append(final_cache[:, :, 1:])

    actual_output = packed.transpose(0, 1).reshape(1, num_tokens, 3 * dim)
    torch.testing.assert_close(
        actual_output,
        torch.cat(expected_outputs, dim=-1),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        conv_pool.index_select(0, state_indices_long),
        torch.cat(expected_caches, dim=1),
        rtol=0,
        atol=0,
    )
    untouched = torch.ones(slots, dtype=torch.bool, device="cuda")
    untouched[state_indices_long] = False
    assert torch.equal(conv_pool[untouched], conv_pool_before[untouched])


@torch.no_grad()
def test_copy_kda_replay_conv_window_preserves_slot_padding() -> None:
    """Committed Q/K/V rows copy across both strided production layouts."""
    slots, dim, committed, num_spec = 6, 11, 3, 2
    slot_stride = 3 * dim * committed + 17
    storage = torch.arange(slots * slot_stride, dtype=torch.float32, device="cuda").to(
        torch.bfloat16
    )
    conv_pool = torch.as_strided(
        storage,
        size=(slots, 3 * dim, committed),
        stride=(slot_stride, committed, 1),
    )

    def replay_cache() -> torch.Tensor:
        return torch.full(
            (slots, committed + num_spec, dim),
            -1,
            dtype=torch.float32,
            device="cuda",
        ).transpose(-1, -2)

    q_cache, k_cache, v_cache = replay_cache(), replay_cache(), replay_cache()
    state_indices = torch.tensor([4, 1], dtype=torch.int32, device="cuda")
    copy_kda_replay_conv_window(
        conv_pool,
        q_cache,
        k_cache,
        v_cache,
        state_indices,
    )

    state_indices_long = state_indices.long()
    for section, cache in enumerate((q_cache, k_cache, v_cache)):
        expected = conv_pool.index_select(0, state_indices_long)[
            :, section * dim : (section + 1) * dim
        ].float()
        torch.testing.assert_close(
            cache.index_select(0, state_indices_long)[:, :, :committed],
            expected,
        )
        assert torch.equal(
            cache.index_select(0, state_indices_long)[:, :, committed:],
            torch.full(
                (state_indices.numel(), dim, num_spec),
                -1,
                dtype=cache.dtype,
                device=cache.device,
            ),
        )
    untouched = torch.ones(slots, dtype=torch.bool, device="cuda")
    untouched[state_indices_long] = False
    assert torch.equal(q_cache[untouched], replay_cache()[untouched])
    assert torch.equal(k_cache[untouched], replay_cache()[untouched])
    assert torch.equal(v_cache[untouched], replay_cache()[untouched])


@torch.no_grad()
def test_optimized_prefill_matches_fla_reference() -> None:
    torch.manual_seed(0)
    optimized = _make_kda()
    reference = _make_reference(optimized.state_dict())

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
@pytest.mark.parametrize(
    "sequence_lengths",
    [None, [100, 257, 63]],
    ids=["padded_eqlen", "varlen"],
)
def test_fused_prefill_beta_sigmoid_matches_unfused_kernel(
    sequence_lengths: list[int] | None,
) -> None:
    """In-kernel beta sigmoid matches the legacy host launch."""
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.common.gate import fused_beta_sigmoid

    torch.manual_seed(10)
    batch_size = 1
    if sequence_lengths is None:
        sequence_length = 300
        num_sequences = batch_size
        cu_seqlens = None
    else:
        sequence_length = sum(sequence_lengths)
        num_sequences = len(sequence_lengths)
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
            dtype=torch.long,
            device="cuda",
        )
    A_log = torch.randn(NUM_HEADS, dtype=torch.float32, device="cuda") * 0.1
    dt_bias = torch.randn(NUM_HEADS * HEAD_DIM, dtype=torch.float32, device="cuda") * 0.1

    for seed in (11, 12):
        generator = torch.Generator(device="cuda").manual_seed(seed)

        def randn(
            *shape: int, dtype: torch.dtype = torch.bfloat16, scale: float = 0.05
        ) -> torch.Tensor:
            return (
                torch.randn(*shape, generator=generator, dtype=torch.float32, device="cuda") * scale
            ).to(dtype)

        q = randn(batch_size, sequence_length, NUM_HEADS, HEAD_DIM)
        k = randn(batch_size, sequence_length, NUM_HEADS, HEAD_DIM)
        v = randn(batch_size, sequence_length, NUM_HEADS, HEAD_DIM)
        g = randn(batch_size, sequence_length, NUM_HEADS, HEAD_DIM)
        if seed == 11:
            beta = randn(batch_size, sequence_length, NUM_HEADS, dtype=torch.float32)
        else:
            # Exercise the L2 epsilon path and saturated sigmoid tails.
            q[:, ::17].zero_()
            k[:, ::19].zero_()
            beta = torch.linspace(
                -20.0,
                20.0,
                batch_size * sequence_length * NUM_HEADS,
                dtype=torch.float32,
                device="cuda",
            ).reshape(batch_size, sequence_length, NUM_HEADS)
        initial_state = randn(
            num_sequences, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, scale=0.01
        )
        state_indices = torch.arange(num_sequences, dtype=torch.int32, device="cuda")

        common = dict(
            v=v,
            g=g,
            scale=HEAD_DIM**-0.5,
            state_indices=state_indices,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
        )

        q_unfused, _ = l2norm_fwd(q)
        k_unfused, _ = l2norm_fwd(k)
        beta_unfused = fused_beta_sigmoid(beta, scale=1.0).to(torch.bfloat16)
        state_unfused = initial_state.clone()
        output_unfused = torch.ops.trtllm.kda_prefill(
            q=q_unfused,
            k=k_unfused,
            beta=beta_unfused,
            state_pool=state_unfused,
            use_beta_sigmoid_in_kernel=False,
            **common,
        )
        # The op reuses shape-keyed output buffers; retain the first result
        # before the fused call overwrites it.
        output_unfused = output_unfused.clone()

        state_fused = initial_state.clone()
        output_fused = torch.ops.trtllm.kda_prefill(
            q=q_unfused,
            k=k_unfused,
            beta=beta,
            state_pool=state_fused,
            use_beta_sigmoid_in_kernel=True,
            **common,
        )

        _assert_close(output_fused, output_unfused)
        _assert_close(state_fused, state_unfused)


@torch.no_grad()
def test_kda_mixer_empty_prefill():
    """The production mixer handles an empty token payload without raising."""
    optimized = _make_kda()
    hidden_states = torch.empty(1, 0, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    out = _run_production_prefill(optimized, hidden_states)
    assert out.shape == (1, 0, HIDDEN_SIZE)


@torch.no_grad()
def test_indexed_prefill_routing_validates_equal_length_state_indices() -> None:
    dispatch = _kda_kernels.KDAKernelDispatch(
        use_optimized_prefill=True,
        use_optimized_decode=False,
    )
    assert dispatch.prefill_kernel_path == "optimized"
    state_pool = torch.zeros(3, 1, 2, 2, dtype=torch.float32, device="cuda")
    state_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    has_initial_states = torch.zeros(2, dtype=torch.bool, device="cuda")
    common = {
        "state_pool": state_pool,
        "has_initial_states": has_initial_states,
        "cu_seqlens": None,
        "num_tokens": 256,
    }

    assert dispatch.can_use_indexed_prefill(
        state_indices=state_indices,
        num_sequences=2,
        **common,
    )
    assert not dispatch.can_use_indexed_prefill(
        state_indices=state_indices,
        num_sequences=1,
        **common,
    )

    misaligned_storage = torch.tensor([-1, 0, 1], dtype=torch.int32, device="cuda")
    misaligned_indices = misaligned_storage[1:]
    assert misaligned_indices.is_contiguous()
    assert misaligned_indices.data_ptr() % 16 != 0
    assert not dispatch.can_use_indexed_prefill(
        state_indices=misaligned_indices,
        num_sequences=2,
        **common,
    )


@torch.no_grad()
def test_kda_prefill_op_rejects_misaligned_state_indices() -> None:
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_custom_ops  # noqa: F401

    q = torch.zeros(1, 256, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    beta = torch.zeros(1, 256, 1, dtype=torch.float32, device="cuda")
    state_pool = torch.zeros(1, 1, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda")
    index_storage = torch.tensor([-1, 0], dtype=torch.int32, device="cuda")
    state_indices = index_storage[1:]
    assert state_indices.is_contiguous()
    assert state_indices.data_ptr() % 16 != 0

    with pytest.raises(ValueError, match="16-byte-aligned state_indices"):
        torch.ops.trtllm.kda_prefill(
            q=q,
            k=q,
            v=q,
            g=q,
            beta=beta,
            state_pool=state_pool,
            state_indices=state_indices,
            scale=HEAD_DIM**-0.5,
        )


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
    optimized = _make_kda()
    reference = _make_reference(optimized.state_dict())
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
    optimized = _make_kda()
    reference = _make_reference(optimized.state_dict())
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


@pytest.mark.parametrize("misaligned_indices", [False, True], ids=["indexed", "fallback"])
@torch.no_grad()
def test_kda_prefill_state_pool_matches_fallback_with_mixed_initial_states(
    monkeypatch: pytest.MonkeyPatch,
    misaligned_indices: bool,
) -> None:
    """Indexed and unsupported-index paths match FLA state handling."""
    torch.manual_seed(3)
    optimized = _make_kda()
    monkeypatch.setattr(_kda_kernels, "is_intree_prefill_available", lambda: False)
    fallback = _make_kda(
        expected_prefill_kernel_path="fla",
        source_state_dict=optimized.state_dict(),
    )

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
    slot_indices = torch.tensor([3, 0, 4, 1], dtype=torch.int32, device="cuda")
    if misaligned_indices:
        slot_storage = torch.tensor([-1, 3, 0, 4, 1], dtype=torch.int32, device="cuda")
        slot_indices = slot_storage[1:]
        assert slot_indices.is_contiguous()
        assert slot_indices.data_ptr() % 16 != 0
    has_initial_states = torch.tensor([True, False, True, False], device="cuda")
    projection_size = NUM_HEADS * HEAD_DIM
    conv_seed = (
        torch.randn(
            slots,
            3 * projection_size,
            CONV_KERNEL_SIZE - 1,
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
    optimized = _make_kda()
    reference = _make_reference(optimized.state_dict())
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
