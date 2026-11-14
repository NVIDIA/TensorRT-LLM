# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA prefill kernels the SM107 glue runs, on their own (no layer).

* ``fused_kda_post_conv`` (the channel-major convolution's Q/K/V producer, chunk padding with
  beta) and the packed GDN convolution (``causal_conv1d_fn``) against three FLA convolutions.
* ``copy_kda_replay_conv_window`` (the fused-verify replay caches every glue seeds after prefill).
* The ``kda_prefill`` op: in-kernel beta sigmoid against the host launch, and bitwise independence
  from the K1 warps' timing.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("fla")

from fla.modules import ShortConvolution  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_kda._kda_kernels import (  # noqa: E402, E501
    copy_kda_replay_conv_window,
    fused_kda_post_conv,
    is_kda_optimized_supported,
)
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn  # noqa: E402

NUM_HEADS = 96
HEAD_DIM = 128


def _has_supported_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    return is_kda_optimized_supported()


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="FlashInfer KDA prefill requires SM100/SM103/SM107",
)


def _kda_prefill_entry():
    """The ``ops/ssm/kda_prefill`` entry (its import needs CuTe DSL and FlashInfer)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.k3r_kda_prefill import (  # noqa: E501
        kda_prefill,
    )

    return kda_prefill


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


@pytest.mark.parametrize("sequence_length", [1, 63, 65, 193])
@pytest.mark.parametrize("beta_layout", ["contiguous", "fused_projection"])
@torch.no_grad()
def test_fused_kda_post_conv_prepad_matches_separate_padding(
    sequence_length: int, beta_layout: str
) -> None:
    """The producer writes the same Q/K/V/beta prefix and exact zero tail as F.pad."""
    torch.manual_seed(1)
    heads, head_dim = 6, 128
    packed = torch.randn(3 * heads * head_dim, sequence_length, device="cuda", dtype=torch.bfloat16)
    if beta_layout == "fused_projection":
        projection = torch.randn(
            1, sequence_length, head_dim + heads, device="cuda", dtype=torch.bfloat16
        )
        beta = projection[..., head_dim:]
        assert beta.stride(-2) > heads
    else:
        beta = torch.randn(1, sequence_length, heads, device="cuda", dtype=torch.bfloat16)

    padded_tokens = ((sequence_length + 63) // 64) * 64
    baseline_qkv = fused_kda_post_conv(packed, heads, head_dim)
    actual = fused_kda_post_conv(packed, heads, head_dim, padded_tokens=padded_tokens, beta=beta)
    expected = tuple(
        torch.nn.functional.pad(value, (0, 0, 0, 0, 0, padded_tokens - sequence_length))
        for value in baseline_qkv
    ) + (torch.nn.functional.pad(beta.float(), (0, 0, 0, padded_tokens - sequence_length)),)

    for name, got, want in zip(("q", "k", "v", "beta"), actual, expected, strict=True):
        context = f"sequence_length={sequence_length} beta_layout={beta_layout} tensor={name}"
        assert got.shape == want.shape and got.is_contiguous(), context
        got_bits = got.view(torch.int16 if got.element_size() == 2 else torch.int32)
        want_bits = want.view(torch.int16 if want.element_size() == 2 else torch.int32)
        assert torch.equal(got_bits, want_bits), context


@pytest.mark.parametrize("invalid", ["shape", "head_stride", "device"])
def test_fused_kda_post_conv_rejects_invalid_beta(invalid: str) -> None:
    """Reject inputs that would silently copy beta or read from another device."""
    packed = torch.empty(3 * 6 * 128, 65, device="cuda", dtype=torch.bfloat16)
    beta = torch.empty(1, 65, 6, device="cuda", dtype=torch.bfloat16)
    if invalid == "shape":
        beta = beta[:, :-1]
    elif invalid == "head_stride":
        beta = torch.empty(1, 65, 12, device="cuda", dtype=torch.bfloat16)[..., ::2]
    else:
        beta = beta.cpu()
    with pytest.raises(ValueError, match="beta must be"):
        fused_kda_post_conv(packed, 6, 128, padded_tokens=128, beta=beta)


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
        output_unfused = _kda_prefill_entry()(
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
        state_unfused = state_unfused.clone()

        state_fused = initial_state.clone()
        output_fused = _kda_prefill_entry()(
            q=q_unfused,
            k=k_unfused,
            beta=beta,
            state_pool=state_fused,
            use_beta_sigmoid_in_kernel=True,
            **common,
        )

        _assert_close(output_fused, output_unfused)
        _assert_close(state_fused, state_unfused)


# fused_k123's K1 warps meet at this barrier after the cross-warp scan, then
# read the scanned prefix rows and the chunk total from sPartialLast.
_K1_SCAN_BARRIER = "                k1_internal_barrier()\n"


_K1_SCAN_READS = (
    _K1_SCAN_BARRIER
    + "\n"
    + "                for vi in cutlass.range_constexpr(VEC):\n"
    + "                    rGkLast[vi] = sPartialLast[K1_ROW_GROUPS - 1, col_base + vi]\n"
)


# K1 warp 1 reads the prefix row 0 and the chunk total.
_LATE_K1_WARP = (
    "                if k1_warp == 1:\n"
    "                    late_t = _test_globaltimer()\n"
    "                    late_end = late_t + cutlass.Int64(20000)\n"
    "                    while late_t < late_end:\n"
    "                        late_t = _test_globaltimer()\n"
)


_GLOBALTIMER = """

@dsl_user_op
def _test_globaltimer(*, loc=None, ip=None):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
"""


@torch.no_grad()
@pytest.mark.parametrize(
    "sequence_lengths",
    [None, [100, 257, 63]],
    ids=["padded_eqlen", "varlen"],
)
def test_fused_prefill_does_not_depend_on_k1_warp_timing(
    sequence_lengths: list[int] | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A K1 warp that reads the scanned partial sums late gets the same bits.

    A copy of fused_k123 stalls K1 warp 1 for 20 us right after the scan
    barrier, so the other K1 warps run ahead into the next chunk before it
    reads sPartialLast. Outputs and states must equal the unpatched kernel's.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_kda import (  # noqa: E501
        fused_k123,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm import (
        _kda_prefill as kimi_k3_ops,
    )

    source = Path(fused_k123.__file__).read_text()
    assert source.count(_K1_SCAN_READS) == 1, "fused_k123's K1 scan reads moved"
    late_source = source.replace(
        _K1_SCAN_READS, _K1_SCAN_BARRIER + _LATE_K1_WARP + _K1_SCAN_READS[len(_K1_SCAN_BARRIER) :]
    )
    late_path = tmp_path / "fused_k123_late_k1_warp.py"
    late_path.write_text(late_source + _GLOBALTIMER)
    spec = importlib.util.spec_from_file_location(late_path.stem, late_path)
    late_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, late_path.stem, late_module)
    spec.loader.exec_module(late_module)

    torch.manual_seed(10)
    if sequence_lengths is None:
        sequence_length = 300
        num_sequences = 1
        cu_seqlens = None
    else:
        sequence_length = sum(sequence_lengths)
        num_sequences = len(sequence_lengths)
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
            dtype=torch.long,
            device="cuda",
        )

    def randn(*shape: int, dtype: torch.dtype = torch.bfloat16, scale: float = 0.05):
        return (torch.randn(*shape, dtype=torch.float32, device="cuda") * scale).to(dtype)

    def l2norm(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x.float(), dim=-1).to(torch.bfloat16)

    shape = (1, sequence_length, NUM_HEADS, HEAD_DIM)
    inputs = dict(
        q=l2norm(randn(*shape)),
        k=l2norm(randn(*shape)),
        v=randn(*shape),
        g=randn(*shape),
        beta=randn(1, sequence_length, NUM_HEADS, dtype=torch.float32),
        scale=HEAD_DIM**-0.5,
        state_indices=torch.arange(num_sequences, dtype=torch.int32, device="cuda"),
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=torch.randn(NUM_HEADS, dtype=torch.float32, device="cuda") * 0.1,
        dt_bias=torch.randn(NUM_HEADS * HEAD_DIM, dtype=torch.float32, device="cuda") * 0.1,
        cu_seqlens=cu_seqlens,
        use_beta_sigmoid_in_kernel=True,
    )
    initial_state = randn(
        num_sequences, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, scale=0.01
    )

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        state = initial_state.clone()
        output = _kda_prefill_entry()(state_pool=state, **inputs).clone()
        return output, state

    expected_output, expected_state = run()
    monkeypatch.setattr(kimi_k3_ops, "_fused_make_host", late_module.make_host_function)
    monkeypatch.setattr(kimi_k3_ops, "_fused_k123_cache", {})
    output, state = run()

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
