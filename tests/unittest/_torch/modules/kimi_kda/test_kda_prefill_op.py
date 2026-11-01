# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the optimized Kimi K3 KDA prefill op."""

from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("fla")

from fla.modules import ShortConvolution  # noqa: E402
from kimi_kda_test_utils import KimiKDAReference, get_production_prefill_kernel_path

from tensorrt_llm._torch.modules.kimi_kda import (
    KimiKDALinearAttention,  # noqa: E402
    _kda_kernels,  # noqa: E402
)
from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (  # noqa: E402
    copy_kda_replay_conv_window,
    fused_kda_post_conv,
    is_kda_optimized_supported,
)
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn  # noqa: E402

NUM_HEADS = 96
HEAD_DIM = 128
CONV_KERNEL_SIZE = 4
HIDDEN_SIZE = 7168


def _has_supported_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    return is_kda_optimized_supported()


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="FlashInfer KDA prefill requires SM100/SM103/SM107",
)


def _make_kda(
    expected_prefill_kernel_path: str = "optimized",
    source_state_dict: Mapping[str, torch.Tensor] | None = None,
    *,
    num_heads: int = NUM_HEADS,
    use_full_rank_gate: bool = True,
) -> KimiKDALinearAttention:
    cfg = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        rms_norm_eps=1e-5,
        linear_attn_config={
            "num_heads": num_heads,
            "head_dim": HEAD_DIM,
            "short_conv_kernel_size": CONV_KERNEL_SIZE,
            "use_full_rank_gate": use_full_rank_gate,
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


def _make_reference(
    source_state_dict: Mapping[str, torch.Tensor],
    *,
    num_heads: int = NUM_HEADS,
    use_full_rank_gate: bool = True,
) -> KimiKDAReference:
    reference = KimiKDAReference(
        hidden_size=HIDDEN_SIZE,
        num_heads=num_heads,
        head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL_SIZE,
        use_full_rank_gate=use_full_rank_gate,
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

    projection_size = attention.proj_size
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
            attention.num_heads,
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
        # causal_conv1d wants int32 offsets, unlike the int64 cu_seqlens above.
        query_start_loc=cu_seqlens.to(torch.int32),
    )
    core = attention.forward_prefill(
        hidden_states.reshape(-1, HIDDEN_SIZE),
        cu_seqlens.to(torch.int64),
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


@pytest.mark.parametrize("token_major", [False, True], ids=["channel_major", "token_major"])
@pytest.mark.parametrize("sequence_length", [1, 127, 256])
@torch.no_grad()
def test_fused_kda_post_conv_matches_reference(sequence_length: int, token_major: bool) -> None:
    """The unfused convolution path produces contiguous raw Q/K/V without normalization."""
    torch.manual_seed(1)
    num_heads, head_dim = 8, 128
    projection_size = num_heads * head_dim
    shape = (3 * projection_size, sequence_length)
    if token_major:
        shape = shape[::-1]
    packed = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    if token_major:
        packed = packed.t()

    actual_q, actual_k, actual_v = fused_kda_post_conv(
        packed, num_heads, head_dim, normalize_qk=False
    )
    unpacked = packed.reshape(3, projection_size, sequence_length)
    expected_q = unpacked[0].transpose(0, 1).reshape(1, sequence_length, num_heads, head_dim)
    expected_k = unpacked[1].transpose(0, 1).reshape(1, sequence_length, num_heads, head_dim)
    expected_v = unpacked[2].transpose(0, 1).reshape(1, sequence_length, num_heads, head_dim)
    assert actual_q.is_contiguous()
    assert actual_k.is_contiguous()
    assert actual_v.is_contiguous()
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    assert torch.equal(actual_v, expected_v)


@torch.no_grad()
def test_fused_kda_post_conv_rejects_layout_without_unit_stride() -> None:
    """Input with no unit stride along tokens or features is rejected."""
    projection_size = 2 * 128
    packed = torch.randn(3 * projection_size, 8, 2, dtype=torch.bfloat16, device="cuda")[..., 0]
    with pytest.raises(ValueError, match="unit stride"):
        fused_kda_post_conv(packed, 2, 128)


@torch.no_grad()
@pytest.mark.parametrize(
    ("num_tokens", "dtype", "channel_major"),
    [
        (256, torch.bfloat16, True),
        (301, torch.bfloat16, False),
        (300, torch.bfloat16, False),
        (300, torch.float32, True),
    ],
    ids=["bf16-aligned", "bf16-odd", "bf16-even-unaligned", "fp32-aligned"],
)
def test_project_packed_conv_input_layout(
    num_tokens: int, dtype: torch.dtype, channel_major: bool
) -> None:
    """The transposed GEMM is used only when its output rows are 16-byte aligned.

    Other token counts keep the token-major GEMM and return its transposed view,
    so the alignment rule follows the element size rather than a token count.
    """
    torch.manual_seed(3)
    # Only the projection layout is checked, so two heads keep the weights small.
    kda = _make_kda(num_heads=2).to(dtype)
    kda._fuse_prefill_conv = False
    kda.finalize_decode_weights()
    assert kda._qkvg_proj_weight is not None
    x2d = torch.randn(num_tokens, HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.05

    packed_conv, _ = kda._project_packed_conv_input(x2d.unsqueeze(0), x2d)

    assert packed_conv.shape == (3 * kda.proj_size, num_tokens)
    assert packed_conv.stride(1 if channel_major else 0) == 1
    expected = torch.nn.functional.linear(
        x2d.float(), kda._qkvg_proj_weight[: 3 * kda.proj_size].float()
    ).t()
    torch.testing.assert_close(packed_conv.float(), expected, rtol=2e-2, atol=2e-2)


@torch.no_grad()
def test_prefill_output_matches_across_projection_layouts() -> None:
    """Aligned and unaligned token counts produce the same prefill output.

    The odd-length request is the aligned one plus one trailing token, so its
    shared prefix must match the aligned request token for token.
    """
    torch.manual_seed(4)
    kda = _make_kda()
    hidden_states = torch.randn(1, 257, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05

    aligned = _run_production_prefill(kda, hidden_states[:, :256].contiguous())
    unaligned = _run_production_prefill(kda, hidden_states)

    torch.testing.assert_close(unaligned[:, :256], aligned, rtol=2e-2, atol=2e-2)


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
@pytest.mark.parametrize(
    ("num_heads", "use_full_rank_gate"),
    [(NUM_HEADS, True), (16, False), (64, False)],
    ids=["full-rank", "low-rank-h16", "low-rank-h64"],
)
def test_optimized_prefill_matches_fla_reference(num_heads: int, use_full_rank_gate: bool) -> None:
    torch.manual_seed(0)
    optimized = _make_kda(num_heads=num_heads, use_full_rank_gate=use_full_rank_gate)
    reference = _make_reference(
        optimized.state_dict(), num_heads=num_heads, use_full_rank_gate=use_full_rank_gate
    )

    # Reuse the same module across changing token counts and batch sizes.
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
def test_kda_mixer_empty_prefill():
    """An empty token payload still publishes the conv state for its slot."""
    optimized = _make_kda()
    hidden_states = torch.empty(1, 0, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    # Seed the pool with a sentinel: the output shape alone would still match if the
    # zero-token state write-back were skipped, so only the pool can catch that.
    conv_pool = torch.full(
        (1, 3 * NUM_HEADS * HEAD_DIM, CONV_KERNEL_SIZE - 1),
        7.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    out = _run_production_prefill(optimized, hidden_states, conv_pool=conv_pool)
    assert out.shape == (1, 0, HIDDEN_SIZE)
    # Without an initial state a zero-token sequence leaves an all-zero causal window.
    assert torch.equal(conv_pool, torch.zeros_like(conv_pool))


@torch.no_grad()
def test_kda_mixer_empty_prefill_preserves_initial_conv_state():
    """A zero-token payload carrying an initial state leaves that state untouched."""
    optimized = _make_kda()
    hidden_states = torch.empty(1, 0, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    conv_seed = (
        torch.randn(
            1,
            3 * NUM_HEADS * HEAD_DIM,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    conv_pool = conv_seed.clone()
    out = _run_production_prefill(
        optimized,
        hidden_states,
        conv_pool=conv_pool,
        has_initial_states=torch.ones(1, dtype=torch.bool, device="cuda"),
    )
    assert out.shape == (1, 0, HIDDEN_SIZE)
    # No token shifts into the window, so the write-back must republish it verbatim.
    assert torch.equal(conv_pool, conv_seed)


@torch.no_grad()
@pytest.mark.parametrize(
    ("state_dtype", "mode"),
    [(torch.bfloat16, "small-bh"), (torch.float32, "auto")],
    ids=["bf16-small-bh", "fp32-auto"],
)
@pytest.mark.parametrize(
    "sequence_lengths",
    [[640, 100, 1000, 37], [1024]],
    ids=["varlen", "single"],
)
def test_kda_prefill_op_sm_reserve_keeps_results(sequence_lengths, state_dtype, mode) -> None:
    """Persistent grids of every size compute bitwise the same output and states.

    ``sm_reserve`` only shrinks the Small-BH persistent grid, so each work
    unit is computed the same way by another block.
    """
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_custom_ops  # noqa: F401

    heads = 8
    tokens = sum(sequence_lengths)
    rows = len(sequence_lengths)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.int64,
        device="cuda",
    )
    gen = torch.Generator(device="cuda").manual_seed(0)
    shape = (1, tokens, heads, HEAD_DIM)

    def bf16(scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(torch.bfloat16)

    inputs = dict(
        q=bf16(),
        k=bf16(),
        v=bf16(0.5),
        g=bf16(),
        beta=torch.randn(1, tokens, heads, generator=gen, device="cuda").to(torch.bfloat16),
    )
    state_seed = (
        torch.randn(rows + 1, heads, HEAD_DIM, HEAD_DIM, generator=gen, device="cuda") * 0.1
    ).to(state_dtype)
    state_indices = torch.arange(1, rows + 1, dtype=torch.int32, device="cuda")

    def call(sm_reserve: int) -> tuple[torch.Tensor, torch.Tensor]:
        state_pool = state_seed.clone()
        output = torch.ops.trtllm.kda_prefill(
            **inputs,
            state_pool=state_pool,
            state_indices=state_indices,
            cu_seqlens=cu_seqlens,
            A_log=torch.zeros(heads, device="cuda"),
            dt_bias=torch.zeros(heads * HEAD_DIM, device="cuda"),
            scale=HEAD_DIM**-0.5,
            lower_bound=-5.0,
            mode=mode,
            sm_reserve=sm_reserve,
        )
        return output, state_pool

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    expected = call(0)
    for sm_reserve in (1, num_sms // 2, num_sms - 3):
        for got, want in zip(call(sm_reserve), expected, strict=True):
            assert torch.equal(got, want), f"sm_reserve={sm_reserve}"
    with pytest.raises(ValueError, match="sm_reserve"):
        call(num_sms)


@torch.no_grad()
def test_kda_mixer_prefill_passes_the_reserve_set_before_construction() -> None:
    """The mixer passes the SM reserve that was set before it was built to the prefill."""
    from tensorrt_llm._torch.cute_dsl_utils import set_sm_reserve

    set_sm_reserve(16)
    try:
        kda = _make_kda(num_heads=2)
    finally:
        set_sm_reserve(0)
    reserves = []
    prefill_chunk_kda = kda._dispatch.prefill_chunk_kda

    def spy(**kwargs):
        reserves.append(kwargs["sm_reserve"])
        return prefill_chunk_kda(**kwargs)

    kda._dispatch.prefill_chunk_kda = spy
    hidden_states = torch.randn(1, 64, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.05
    _run_production_prefill(kda, hidden_states)
    assert reserves == [16]


@torch.no_grad()
def test_kda_prefill_op_partial_final_chunk_large_batch():
    """Large packed batches with short final sequences match the FLA reference.

    Cover the autotuner warmup split [8191, 1] and a mixed-length 8192-token batch.
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
    ([1], [30], [65], [6, 12], [64, 64], [1, 2, 3], [64, 64, 63], [6, 12, 20, 25]),
    ids=(
        "single-token",
        "short-single-seq",
        "partial-single-seq",
        "two-short-seqs",
        "two-aligned-seqs",
        "three-short-seqs",
        "three-seqs-partial-tail",
        "four-short-seqs",
    ),
)
def test_kda_prefill_small_varlen_stays_on_optimized_op(
    monkeypatch: pytest.MonkeyPatch,
    sequence_lengths: list[int],
) -> None:
    """Short packed prompts stay on the optimized prefill path.

    Routing these short-prompt shapes to FLA instead would make a real serving
    run pay FLA's cold Triton autotune on its first short context.
    """
    import fla.ops.kda

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

    # The FLA fallback resolves ``chunk_kda`` at call time, so this counts it;
    # the reference binds it at import time and is unaffected.
    fla_spy = MagicMock(side_effect=fla.ops.kda.chunk_kda)
    monkeypatch.setattr(fla.ops.kda, "chunk_kda", fla_spy)
    actual = _run_production_prefill(optimized, hidden_states, cumulative_lengths)
    assert fla_spy.call_count == 0, (
        f"prefill fell back to FLA for sequence lengths {sequence_lengths}; "
        "the optimized op must handle short packed prompts"
    )

    expected = reference.forward_prefill(hidden_states, cu_seqlens=cumulative_lengths)
    _assert_close(actual, expected)


@pytest.mark.parametrize(
    "misaligned_indices", [False, True], ids=["aligned-indices", "offset-indices"]
)
@torch.no_grad()
def test_kda_prefill_state_pool_matches_fallback_with_mixed_initial_states(
    monkeypatch: pytest.MonkeyPatch,
    misaligned_indices: bool,
) -> None:
    """Aligned and offset int32 slot views both match the explicit FLA fallback."""
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
@pytest.mark.parametrize(
    "sequence_lengths",
    [[100, 200], None],
    ids=["varlen", "single-sequence"],
)
def test_kda_prefill_op_output_survives_the_next_call(sequence_lengths) -> None:
    """The op's output belongs to the caller, as its fake (a new tensor) declares.

    A second call with the same shapes and other inputs must leave the first
    call's output as it was. ``None`` runs one packed 300-token sequence.
    """
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_custom_ops  # noqa: F401

    heads = 2
    tokens = sum(sequence_lengths) if sequence_lengths else 300
    rows = len(sequence_lengths) if sequence_lengths else 1
    state_pool = torch.zeros(rows, heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda")
    state_indices = torch.arange(rows, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int64, device="cuda")
    if sequence_lengths:
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
            dtype=torch.long,
            device="cuda",
        )

    def call(seed: int) -> torch.Tensor:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        shape = (1, tokens, heads, HEAD_DIM)

        def unit() -> torch.Tensor:
            t = torch.randn(*shape, generator=gen, device="cuda")
            return (t / t.norm(dim=-1, keepdim=True)).to(torch.bfloat16)

        state_pool.zero_()
        return torch.ops.trtllm.kda_prefill(
            q=unit(),
            k=unit(),
            v=(torch.randn(*shape, generator=gen, device="cuda") * 0.5).to(torch.bfloat16),
            g=torch.randn(*shape, generator=gen, device="cuda").to(torch.bfloat16),
            beta=torch.randn(1, tokens, heads, generator=gen, device="cuda", dtype=torch.bfloat16),
            state_pool=state_pool,
            state_indices=state_indices,
            scale=HEAD_DIM**-0.5,
            cu_seqlens=cu_seqlens,
            lower_bound=-5.0,
            A_log=torch.zeros(heads, device="cuda"),
            dt_bias=torch.zeros(heads * HEAD_DIM, device="cuda"),
        )

    first = call(seed=0)
    kept = first.clone()
    second = call(seed=1)
    torch.cuda.synchronize()
    assert first.shape == second.shape == (1, tokens, heads, HEAD_DIM)
    assert not torch.equal(second, kept), "the two calls must compute different outputs"
    assert torch.equal(first, kept), "a later call wrote an earlier call's output"


@torch.no_grad()
def test_token_major_prefill_matches_channel_major_with_initial_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The token-major projection layout carries continuation state like channel-major.

    An unaligned token count takes the token-major projection and the
    channel-last convolution. Forcing the same projection channel-major must
    give the same outputs and pool updates for mixed initial states,
    permuted slots and an unused slot.
    """
    torch.manual_seed(5)
    kda = _make_kda()
    kda._fuse_prefill_conv = False
    sequence_lengths = [64, 64, 64, 63]
    assert sum(sequence_lengths) * torch.bfloat16.itemsize % 16 != 0
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
    has_initial_states = torch.tensor([True, False, True, False], device="cuda")
    conv_seed = (
        torch.randn(
            slots,
            3 * kda.proj_size,
            CONV_KERNEL_SIZE - 1,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.05
    )
    state_seed = (
        torch.randn(slots, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda") * 0.05
    )

    def run():
        """Prefill from fresh copies of the seeded pools; return output and pools."""
        conv_pool, state_pool = conv_seed.clone(), state_seed.clone()
        output = _run_production_prefill(
            kda,
            hidden_states,
            cumulative_lengths,
            conv_pool=conv_pool,
            state_pool=state_pool,
            slot_indices=slot_indices,
            has_initial_states=has_initial_states,
        )
        return output, conv_pool, state_pool

    project = KimiKDALinearAttention._project_packed_conv_input
    layouts = []

    def recording_project(self, x, x2d):
        """Record whether the projection came back token-major."""
        packed_conv, onorm_g = project(self, x, x2d)
        layouts.append(packed_conv.stride(0) == 1)
        return packed_conv, onorm_g

    monkeypatch.setattr(KimiKDALinearAttention, "_project_packed_conv_input", recording_project)
    token_major = run()
    assert layouts == [True]

    def channel_major_project(self, x, x2d):
        """Force the channel-major layout regardless of alignment."""
        packed_conv, onorm_g = project(self, x, x2d)
        return packed_conv.contiguous(), onorm_g

    monkeypatch.setattr(KimiKDALinearAttention, "_project_packed_conv_input", channel_major_project)
    channel_major = run()

    _assert_close(token_major[0], channel_major[0])
    # Convolution states are copied verbatim out of the projection.
    torch.testing.assert_close(token_major[1], channel_major[1], rtol=0, atol=0)
    _assert_close(
        token_major[2].index_select(0, slot_indices),
        channel_major[2].index_select(0, slot_indices),
    )
    unused_slot = 2
    assert torch.equal(token_major[1][unused_slot], conv_seed[unused_slot])
    assert torch.equal(token_major[2][unused_slot], state_seed[unused_slot])


@torch.no_grad()
def test_kda_prefill_op_shape_growth_and_cu_dtype_transitions():
    """Changing token counts and caller offset dtypes preserve reference parity.

    The production adapter prepares int64 packed offsets for FlashInfer from
    either caller dtype. Repeated growth and shrinkage exercise cached kernels.
    """
    torch.manual_seed(0)
    optimized = _make_kda()
    reference = _make_reference(optimized.state_dict())
    cases = [
        ([517, 654], torch.long),  # small batch, int64 cu (dump-replay-like)
        ([8191], torch.int32),  # growth + dtype conversion
        ([1171], torch.long),  # shrink + flip back
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
