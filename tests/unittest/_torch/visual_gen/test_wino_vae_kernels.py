# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WinoVae kernels against their eager references: the runtime-compiled norm kernels, the
temporal / spatial Winograd pieces, the compiled frame ops and the sm100 pipeline."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (
    TEMPORAL_WINO,
    add_rms_norm_silu_cuda,
    apply_in_bias,
    can_use_cuda_rms_norm_silu,
    cuda_op_available,
    rms_norm_silu_cuda,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")


# ---------------------------------------------------------------------------
# Eager references: what each kernel must compute, in plain PyTorch ops
# ---------------------------------------------------------------------------


def rms_norm_silu_reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | float,
    scale: float,
    in_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """The eager path, verbatim (WanRMSNorm.forward followed by F.silu)."""
    x = apply_in_bias(x, in_bias)
    needs_fp32 = x.dtype in (torch.float16, torch.bfloat16)
    normalized = F.normalize(x.float() if needs_fp32 else x, dim=1).to(x.dtype)
    return F.silu(normalized * scale * gamma + bias)


def add_rms_norm_silu_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | float,
    scale: float,
    in_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    s = apply_in_bias(x, in_bias) + residual
    return s, rms_norm_silu_reference(s, gamma, bias, scale)


def rms_norm_reference(x: torch.Tensor, gamma: torch.Tensor, scale: float) -> torch.Tensor:
    """The eager WanRMSNorm.forward alone (no activation, no bias)."""
    needs_fp32 = x.dtype in (torch.float16, torch.bfloat16)
    normalized = F.normalize(x.float() if needs_fp32 else x, dim=1).to(x.dtype)
    return normalized * scale * gamma


def interleave_bias_reference(u: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    t, p, c2 = u.shape
    c = c2 // 2
    ub = u + bias.reshape(1, 1, -1) if bias is not None else u
    return torch.stack([ub[:, :, :c], ub[:, :, c:]], dim=1).reshape(2 * t, p, c)


def wino_combine_reference(m: list[torch.Tensor]) -> torch.Tensor:
    """``m[k] [groups, P, C]`` (``order + 2`` of them) -> bf16 frames ``[order*groups, P, C]``:
    frame j of a group is ``sum_k AT[j][k] m_k`` in index order with bf16 rounding after every
    add (the products are exact: powers of two)."""
    order = len(m) - 2
    at = TEMPORAL_WINO[order]["AT"]
    frames = []
    for row in at:
        terms = [(c, m[k]) for k, c in enumerate(row) if c != 0]
        acc = (terms[0][0] * terms[0][1].float()).to(torch.bfloat16)
        for c, mk in terms[1:]:
            acc = (acc.float() + c * mk.float()).to(torch.bfloat16)
        frames.append(acc)
    return torch.stack(frames, dim=1).reshape(order * m[0].shape[0], *m[0].shape[1:])


def wino_norm_silu_reference(
    x: torch.Tensor | None,
    m: list[torch.Tensor] | None,
    residual: torch.Tensor | None,
    in_bias: torch.Tensor | None,
    cache: torch.Tensor | None,
    gamma: torch.Tensor,
    bias: torch.Tensor | float,
    scale: float,
    order: int = 2,
    winograd_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor | None, torch.Tensor, list[torch.Tensor]]:
    """Eager form of the kernel, same rounding points (bf16 after every add of the combine,
    the Winograd inputs formed in fp32 and rounded once to ``winograd_dtype``)."""
    if x is None:
        assert m is not None
        x = wino_combine_reference(m)
    s = None
    if in_bias is not None:
        x = s = x + in_bias.reshape(1, 1, -1)
    if residual is not None:
        x = s = x + residual
    if m is not None and s is None:
        s = x
    t, p, c = x.shape
    y = rms_norm_silu_reference(x.permute(0, 2, 1), gamma.reshape(1, -1, 1), bias, scale).permute(
        0, 2, 1
    )
    cache_frames = cache.shape[0] if cache is not None else 0
    pad = torch.zeros(2 - cache_frames, p, c, dtype=x.dtype, device=x.device)
    hist = torch.cat([pad, cache], 0) if cache is not None and cache_frames else pad
    f = torch.cat([hist, y], 0).float()  # frames -2 .. T-1
    bt = torch.tensor(TEMPORAL_WINO[order]["BT"], dtype=torch.float32, device=x.device)
    groups = f.unfold(0, order + 2, order)  # [groups, P, C, order+2]
    d = torch.einsum("ki,gpci->kgpc", bt, groups).to(winograd_dtype)
    return s, y[-2:].contiguous(), [d[k] for k in range(order + 2)]


def upsample_gather_reference(
    temp: torch.Tensor,
    batch: int,
    frames: int,
    shortcut_expanded: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager phase gather: ``out[.., 2i+p, 2j+q] = temp[.., phase, i+p, j+q]`` (+ bias, + shortcut)."""
    bt, c4, hp1, wp1 = temp.shape
    cout, h, w = c4 // 4, hp1 - 1, wp1 - 1
    t4 = temp.view(bt, 4, cout, hp1, wp1)
    out = temp.new_empty(bt, cout, 2 * h, 2 * w)
    for p in (0, 1):
        for q in (0, 1):
            out[:, :, p::2, q::2] = t4[:, 2 * p + q, :, p : p + h, q : q + w]
    out = out.view(batch, frames, cout, 2 * h, 2 * w).permute(0, 2, 1, 3, 4)
    out = apply_in_bias(out, bias)
    if shortcut_expanded is not None:
        out = out + shortcut_expanded
    return out.contiguous(memory_format=torch.channels_last_3d)


# Decoder/encoder channel counts of the Cosmos3 VAE plus a few small ones.
CHANNELS = [48, 96, 160, 256, 320, 512, 640, 1024, 16, 24, 32]


def _inputs(
    C: int, frames: int = 3, hw: tuple[int, int] = (6, 10), dtype=torch.bfloat16, bias: bool = False
):
    gen = torch.Generator(device="cuda").manual_seed(C)
    x = torch.randn(2, C, frames, *hw, generator=gen, device="cuda").to(dtype)
    x = x.to(memory_format=torch.channels_last_3d)
    gamma = (1 + 0.1 * torch.randn(C, 1, 1, 1, generator=gen, device="cuda")).to(dtype)
    b = (0.05 * torch.randn(C, 1, 1, 1, generator=gen, device="cuda")).to(dtype) if bias else None
    return x, gamma, b


def _built(available: bool, what: str) -> pytest.MarkDecorator:
    """Gate for a first-party op or a declared dependency: on a CUDA machine it must be there
    (a missing build fails the file at collection instead of skipping it green); without CUDA
    the tests skip as hardware-gated."""
    if torch.cuda.is_available() and not available:
        pytest.fail(f"{what} unavailable on a CUDA machine", pytrace=False)
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")


def _require_sm100() -> None:
    """Blackwell is a hardware gate (skip); the CuTe DSL import on Blackwell is not (fail)."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import sm100_wino_available

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("sm100 kernels need Blackwell")
    assert sm100_wino_available(), "nvidia-cutlass-dsl (a declared requirement) does not import"


cuda_op = _built(cuda_op_available(), "runtime compilation of the norm kernels (nvcc, cuda-python)")


@cuda_op
@pytest.mark.parametrize("C", CHANNELS)
def test_cuda_fp32_mode_is_closer_to_fp32_than_eager(C: int) -> None:
    x, gamma, b = _inputs(C, bias=True)
    scale = C**0.5
    truth = rms_norm_silu_reference(x.float(), gamma.float(), b.float(), scale)
    eager = rms_norm_silu_reference(x, gamma, b, scale).float()
    fused = rms_norm_silu_cuda(x, gamma, b, scale).float()
    err_eager = (eager - truth).norm() / truth.norm()
    err_fused = (fused - truth).norm() / truth.norm()
    assert torch.isfinite(fused).all()
    assert err_fused <= err_eager * 1.05, f"C={C}: fused {err_fused:.3e} vs eager {err_eager:.3e}"
    assert err_fused < 1e-2


def test_cuda_norm_eligibility() -> None:
    x = torch.randn(1, 32, 2, 4, 4, device="cuda", dtype=torch.bfloat16)  # NCTHW, not channels-last
    assert not can_use_cuda_rms_norm_silu(x)
    assert can_use_cuda_rms_norm_silu(x.to(memory_format=torch.channels_last_3d))
    assert not can_use_cuda_rms_norm_silu(x.float().to(memory_format=torch.channels_last_3d))
    assert not can_use_cuda_rms_norm_silu(x.cpu())
    odd = torch.randn(1, 12, 2, 4, 4, device="cuda", dtype=torch.bfloat16)
    assert not can_use_cuda_rms_norm_silu(odd.to(memory_format=torch.channels_last_3d))


@cuda_op
def test_cuda_4d_images_layout() -> None:
    C = 96
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = (
        torch.randn(3, C, 8, 8, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last)
    )
    gamma = torch.ones(C, 1, 1, device="cuda", dtype=torch.bfloat16)
    expected = rms_norm_silu_reference(x, gamma, 0.0, C**0.5)
    actual = rms_norm_silu_cuda(x, gamma, None, C**0.5, reference_rounding=True)
    assert (actual.float() - expected.float()).abs().max().item() < 2e-2


@cuda_op
@pytest.mark.parametrize("C", CHANNELS)
@pytest.mark.parametrize("bias", [False, True])
def test_cuda_reference_mode_matches_eager(C: int, bias: bool) -> None:
    x, gamma, b = _inputs(C, frames=5, hw=(9, 13), bias=bias)
    assert can_use_cuda_rms_norm_silu(x)
    scale = C**0.5
    expected = rms_norm_silu_reference(x, gamma, b if b is not None else 0.0, scale)
    actual = rms_norm_silu_cuda(x, gamma, b, scale, reference_rounding=True)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    diff = (actual.float() - expected.float()).abs()
    ulp = expected.float().abs().clamp_min(1e-3) * 2**-7
    mismatched = (diff > ulp).float().mean().item()
    assert mismatched < 1e-3, f"C={C}: {mismatched:.2e} of elements differ by more than 1 bf16 ulp"


@cuda_op
def test_cuda_tail_rows_and_zero_rows() -> None:
    C = 256
    gen = torch.Generator(device="cuda").manual_seed(7)
    # 1 batch x 3 frames x 7 x 11 = 231 rows: not a multiple of any block size.
    x = (
        torch.randn(1, C, 3, 7, 11, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    x[0, :, 0, 0, 0] = 0
    gamma = torch.ones(C, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    for reference in (False, True):
        out = rms_norm_silu_cuda(x, gamma, None, C**0.5, reference_rounding=reference)
        assert torch.isfinite(out.float()).all()
        assert out[0, :, 0, 0, 0].abs().max().item() == 0.0
        expected = rms_norm_silu_reference(x, gamma, 0.0, C**0.5)
        assert (out.float() - expected.float()).abs().max().item() < 2e-2


@cuda_op
@pytest.mark.parametrize("C", [256, 512, 160, 48])
def test_cuda_large_tensor_matches_eager(C: int) -> None:
    # ~300k rows: every persistent block streams many shared-memory stages.
    gen = torch.Generator(device="cuda").manual_seed(C + 1)
    x = (
        torch.randn(1, C, 3, 320, 320, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    gamma = (1 + 0.1 * torch.randn(C, 1, 1, 1, generator=gen, device="cuda")).to(torch.bfloat16)
    scale = C**0.5
    ref = rms_norm_silu_cuda(x, gamma, None, scale, reference_rounding=True)
    expected = rms_norm_silu_reference(x, gamma, 0.0, scale)
    ulp = expected.float().abs().clamp_min(1e-3) * 2**-7
    assert ((ref.float() - expected.float()).abs() > ulp).float().mean().item() < 1e-3


@cuda_op
@pytest.mark.parametrize("C", [256, 512, 160, 48])
@pytest.mark.parametrize("reference", [True, False])
def test_cuda_residual_variant(C: int, reference: bool) -> None:
    gen = torch.Generator(device="cuda").manual_seed(C + 7)
    x = (
        torch.randn(1, C, 3, 160, 160, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    r = (
        torch.randn(1, C, 3, 160, 160, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    gamma = (1 + 0.1 * torch.randn(C, 1, 1, 1, generator=gen, device="cuda")).to(torch.bfloat16)
    scale = C**0.5
    s_ref, a_ref = add_rms_norm_silu_reference(x, r, gamma, 0.0, scale)
    s, a = add_rms_norm_silu_cuda(x, r, gamma, None, scale, reference_rounding=reference)
    assert torch.equal(s, s_ref), "the returned sum must be bit-identical to the eager bf16 add"
    assert a.is_contiguous(memory_format=torch.channels_last_3d)
    ulp = a_ref.float().abs().clamp_min(1e-3) * 2**-7
    mismatched = ((a.float() - a_ref.float()).abs() > ulp).float().mean().item()
    assert mismatched < (1e-3 if reference else 5e-2)
    # fp32 mode must also agree with the standalone norm applied to the sum
    if not reference:
        a2 = rms_norm_silu_cuda(s, gamma, None, scale)
        assert torch.equal(a, a2)


@cuda_op
@pytest.mark.parametrize("C", [256, 512, 160, 48, 16])
@pytest.mark.parametrize("reference", [True, False])
@pytest.mark.parametrize("with_residual", [False, True])
def test_cuda_in_bias(C: int, reference: bool, with_residual: bool) -> None:
    """A deferred conv bias is added first as a bf16 add; the sum stays bit-identical to eager."""
    gen = torch.Generator(device="cuda").manual_seed(C + 11)
    shape = (1, C, 2, 40, 40)
    x = (
        torch.randn(*shape, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    r = (
        torch.randn(*shape, generator=gen, device="cuda")
        .to(torch.bfloat16)
        .to(memory_format=torch.channels_last_3d)
    )
    gamma = (1 + 0.1 * torch.randn(C, 1, 1, 1, generator=gen, device="cuda")).to(torch.bfloat16)
    in_bias = (0.3 * torch.randn(C, generator=gen, device="cuda")).to(torch.bfloat16)
    scale = C**0.5
    if with_residual:
        s_ref, a_ref = add_rms_norm_silu_reference(x, r, gamma, 0.0, scale, in_bias=in_bias)
        s, a = add_rms_norm_silu_cuda(
            x, r, gamma, None, scale, reference_rounding=reference, in_bias=in_bias
        )
        assert torch.equal(s, s_ref)
    else:
        a_ref = rms_norm_silu_reference(x, gamma, 0.0, scale, in_bias=in_bias)
        a = rms_norm_silu_cuda(x, gamma, None, scale, reference_rounding=reference, in_bias=in_bias)
    ulp = a_ref.float().abs().clamp_min(1e-3) * 2**-7
    mismatched = ((a.float() - a_ref.float()).abs() > ulp).float().mean().item()
    assert mismatched < (1e-3 if reference else 5e-2)
    # The in-bias path must equal the standalone kernel applied to the eagerly biased input.
    xb = (x + in_bias.view(1, -1, 1, 1, 1)).contiguous(memory_format=torch.channels_last_3d)
    if with_residual:
        _, a2 = add_rms_norm_silu_cuda(xb, r, gamma, None, scale, reference_rounding=reference)
    else:
        a2 = rms_norm_silu_cuda(xb, gamma, None, scale, reference_rounding=reference)
    assert torch.equal(a, a2)


# ---------------------------------------------------------------------------
# Upsample: packed 2x2 conv + phase gather vs nearest-2x + 3x3 conv (+ DupUp3D shortcut)
# ---------------------------------------------------------------------------


from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import DupUp3D  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (  # noqa: E402
    cuda_upsample_available,
    pack_upsample_weights,
    upsample_gather_cuda,
)

upsample_op = _built(cuda_upsample_available(), "trtllm::vae_upsample_gather")


def _eager_upsample(x4: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    up = F.interpolate(x4.float(), scale_factor=2.0, mode="nearest-exact").to(x4.dtype)
    return F.conv2d(up, weight, bias, padding=1)


@pytest.mark.parametrize("shape", [(3, 16, 5, 7), (2, 8, 1, 1), (1, 24, 4, 6)])
def test_packed_upsample_algebra_fp32(shape: tuple[int, int, int, int]) -> None:
    bt, cin, h, w = shape
    cout = 16
    gen = torch.Generator(device="cuda").manual_seed(h * 31 + w)
    x = torch.randn(bt, cin, h, w, generator=gen, device="cuda")
    weight = torch.randn(cout, cin, 3, 3, generator=gen, device="cuda")
    bias = torch.randn(cout, generator=gen, device="cuda")
    # fp32 convolutions default to TF32 on Ampere+, which would hide the algebra behind
    # ~1e-2 of conv rounding; this test is about the exact identity.
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        expected = _eager_upsample(x, weight, bias)
        packed, packed_bias = pack_upsample_weights(weight, bias)
        temp = F.conv2d(x, packed, packed_bias, padding=1)
    finally:
        torch.backends.cudnn.allow_tf32 = prev
    got = upsample_gather_reference(temp.contiguous(memory_format=torch.channels_last), bt, 1)
    got = got[:, :, 0]  # frames=1 -> [bt, cout, 2h, 2w]
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)


@upsample_op
@pytest.mark.parametrize("factor_t,first_chunk", [(1, False), (2, False), (2, True)])
# repeats = cout*factor/cin: equal to the factor (64->64), below it (64->32, strided
# span), above it (16->64 and 32->64, several output channels per source channel).
@pytest.mark.parametrize("cin_out", [(64, 64), (64, 32), (16, 64), (32, 64)])
def test_cuda_upsample_gather_with_shortcut(
    factor_t: int, first_chunk: bool, cin_out: tuple[int, int]
) -> None:
    cin, cout = cin_out
    b, tr, h, w = 1, 3, 9, 13
    frames = tr * factor_t - (factor_t - 1 if first_chunk else 0)
    gen = torch.Generator(device="cuda").manual_seed(cout + factor_t)
    # upsampler input after the temporal part: [b, cout_in=cin, frames, h, w]
    x5 = torch.randn(b, cin, frames, h, w, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (0.1 * torch.randn(cout, cin, 3, 3, generator=gen, device="cuda")).to(torch.bfloat16)
    bias = (0.1 * torch.randn(cout, generator=gen, device="cuda")).to(torch.bfloat16)
    residual = torch.randn(b, cin, tr, h, w, generator=gen, device="cuda").to(torch.bfloat16)
    residual = residual.to(memory_format=torch.channels_last_3d)
    dup = DupUp3D(cin, cout, factor_t=factor_t, factor_s=2)

    x4 = (
        x5.permute(0, 2, 1, 3, 4)
        .reshape(b * frames, cin, h, w)
        .contiguous(memory_format=torch.channels_last)
    )
    eager = _eager_upsample(x4, weight, bias)
    eager = eager.view(b, frames, cout, 2 * h, 2 * w).permute(0, 2, 1, 3, 4)
    eager = (eager + dup(residual, first_chunk=first_chunk)).contiguous(
        memory_format=torch.channels_last_3d
    )

    packed, packed_bias = pack_upsample_weights(weight, bias)
    temp = F.conv2d(x4, packed, packed_bias, padding=1).contiguous(
        memory_format=torch.channels_last
    )
    crop = factor_t - 1 if first_chunk else 0
    got = upsample_gather_cuda(temp, residual, b, frames, factor_t, dup.repeats, crop)
    assert got.shape == eager.shape and got.is_contiguous(memory_format=torch.channels_last_3d)

    # The kernel's gather + shortcut add must be bit-identical to the reference gather
    # followed by the eager DupUp3D add (same bf16 inputs, one bf16 rounding).
    via_reference = upsample_gather_reference(
        temp, b, frames, dup(residual, first_chunk=first_chunk)
    )
    assert torch.equal(got, via_reference)
    plain = upsample_gather_cuda(temp, None, b, frames)
    assert torch.equal(plain, upsample_gather_reference(temp, b, frames))

    # Deferred conv bias: the gather adds it before the shortcut, bit-identical to the
    # reference gather of the unbiased conv output followed by the eager bias add.
    temp_nb = F.conv2d(x4, packed, None, padding=1).contiguous(memory_format=torch.channels_last)
    got_nb = upsample_gather_cuda(
        temp_nb, residual, b, frames, factor_t, dup.repeats, crop, bias=bias
    )
    ref_nb = upsample_gather_reference(
        temp_nb, b, frames, dup(residual, first_chunk=first_chunk), bias=bias
    )
    assert torch.equal(got_nb, ref_nb)
    assert torch.equal(
        upsample_gather_cuda(temp_nb, None, b, frames, bias=bias),
        upsample_gather_reference(temp_nb, b, frames, bias=bias),
    )

    # Against the original upsample + 3x3 conv the only difference is the packed weights
    # rounding the summed taps once more: last-bit noise, never a mapping-sized error.
    diff = (got.float() - eager.float()).abs()
    assert (
        diff.norm() / eager.float().norm()
    ).item() < 5e-3  # one extra bf16 rounding of the summed taps
    assert diff.max().item() <= 2**-6 * eager.float().abs().max().item()


# ---------------------------------------------------------------------------
# Temporal Winograd norm kernel vs its eager reference
# ---------------------------------------------------------------------------

from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (  # noqa: E402
    wino_norm_silu_cuda,
)

wino_op = cuda_op


@wino_op
@pytest.mark.parametrize("C", [256, 512, 1024, 160, 48])
@pytest.mark.parametrize("cache_frames", [0, 1, 2])
@pytest.mark.parametrize("source", ["x", "m"])
@pytest.mark.parametrize("extras", ["plain", "residual_bias"])
@pytest.mark.parametrize("order_dtype", [(2, "bf16"), (2, "fp16"), (4, "fp16")])
def test_wino_norm_silu_reference_mode_is_bitwise(
    C: int, cache_frames: int, source: str, extras: str, order_dtype: tuple[int, str]
) -> None:
    order, dtype_name = order_dtype
    winograd_dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    gen = torch.Generator(device="cuda").manual_seed(C + cache_frames)
    T, P = (4 if order == 2 else 8), 9 * 13
    bf = torch.bfloat16
    rnd = lambda *s: torch.randn(*s, generator=gen, device="cuda").to(bf)  # noqa: E731
    x = rnd(T, P, C) if source == "x" else None
    m = (
        [rnd(T // order, P, C).to(winograd_dtype) for _ in range(order + 2)]
        if source == "m"
        else None
    )
    residual = rnd(T, P, C) if extras == "residual_bias" else None
    in_bias = (
        (0.3 * torch.randn(C, generator=gen, device="cuda")).to(bf)
        if extras == "residual_bias"
        else None
    )
    cache = rnd(cache_frames, P, C) if cache_frames else None
    gamma = (1 + 0.1 * torch.randn(C, generator=gen, device="cuda")).to(bf)
    scale = C**0.5
    s_ref, tail_ref, d_ref = wino_norm_silu_reference(
        x, m, residual, in_bias, cache, gamma, 0.0, scale, order, winograd_dtype
    )
    s, tail, d = wino_norm_silu_cuda(
        x,
        m,
        residual,
        in_bias,
        cache,
        gamma,
        None,
        scale,
        T,
        reference_rounding=True,
        order=order,
        fp16_inputs=dtype_name == "fp16",
    )
    assert d.dtype == winograd_dtype and d.shape[0] == order + 2
    if residual is not None:
        assert torch.equal(s, s_ref), "sum must be bit-identical to the eager adds"
    # The normalized frames (cache) and the Winograd inputs match eager up to the fp32 reduction
    # order of the sum of squares: at most 1 bf16 ulp on a tiny fraction of elements.
    for name, a, b in [
        ("cache", tail, tail_ref),
        *((f"D{k}", d[k], d_ref[k]) for k in range(order + 2)),
    ]:
        diff = (a.float() - b.float()).abs()
        ulp = b.float().abs().clamp_min(1e-3) * 2**-7
        assert (diff > ulp).float().mean().item() < 1e-3, f"{name} differs from the eager form"


@wino_op
@pytest.mark.parametrize("C", [256, 160])
@pytest.mark.parametrize("order", [2, 4])
def test_wino_norm_silu_fp32_mode_close(C: int, order: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(C)
    T, P = (6 if order == 2 else 8), 20 * 20
    bf = torch.bfloat16
    x = torch.randn(T, P, C, generator=gen, device="cuda").to(bf)
    cache = torch.randn(2, P, C, generator=gen, device="cuda").to(bf)
    gamma = (1 + 0.1 * torch.randn(C, generator=gen, device="cuda")).to(bf)
    winograd_dtype = torch.float16 if order == 4 else torch.bfloat16
    _, tail_ref, d_ref = wino_norm_silu_reference(
        x, None, None, None, cache, gamma, 0.0, C**0.5, order, winograd_dtype
    )
    _, tail, d = wino_norm_silu_cuda(
        x,
        None,
        None,
        None,
        cache,
        gamma,
        None,
        C**0.5,
        T,
        order=order,
        fp16_inputs=order == 4,
    )
    for a, b in [(tail, tail_ref), *zip(d, d_ref)]:
        rel = ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)).item()
        assert rel < 1e-2, rel


@wino_op
@pytest.mark.parametrize("order", [2, 4])
def test_wino_norm_inputs_reproduce_conv3d(order: int) -> None:
    """Winograd inputs from the kernel, one conv2d per component filter, combine: the causal 3x3x3 conv
    of the normalized frames (fp32 reference on the same 16-bit inputs)."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (
        TEMPORAL_WINO,
        winograd_weights,
    )

    gen = torch.Generator(device="cuda").manual_seed(order)
    C, Co, H, W = 64, 48, 6, 10
    T = 2 * order
    x = torch.randn(T, H * W, C, generator=gen, device="cuda").to(torch.bfloat16)
    cache = torch.randn(2, H * W, C, generator=gen, device="cuda").to(torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(C, generator=gen, device="cuda")).to(torch.bfloat16)
    weight = (0.1 * torch.randn(Co, C, 3, 3, 3, generator=gen, device="cuda")).to(torch.bfloat16)
    _, _, d = wino_norm_silu_cuda(
        x, None, None, None, cache, gamma, None, C**0.5, T, order=order, fp16_inputs=order == 4
    )
    g = winograd_weights(weight, order, torch.float32)
    comps = order + 2
    m = torch.stack(
        [
            F.conv2d(d[k].float().view(T // order, H, W, C).permute(0, 3, 1, 2), g[k], None, 1, 1)
            for k in range(comps)
        ]
    )  # [comps, groups, Co, H, W]
    at = torch.tensor(TEMPORAL_WINO[order]["AT"], dtype=torch.float32, device="cuda")
    y = torch.einsum("jk,kgchw->gjchw", at, m).reshape(T, Co, H, W)
    # direct conv of the normalized frames (bf16, as the kernel's ring holds them) with the cache
    y_norm = rms_norm_silu_reference(x.permute(0, 2, 1), gamma.reshape(1, -1, 1), 0.0, C**0.5)
    frames = torch.cat([cache.float(), y_norm.permute(0, 2, 1).float()], 0)  # [T+2, P, C]
    ref = F.conv3d(
        frames.view(T + 2, H, W, C).permute(3, 0, 1, 2).unsqueeze(0),
        weight.float(),
        None,
        1,
        (0, 1, 1),
    )[0].permute(1, 0, 2, 3)  # [T, Co, H, W]
    rel = ((y - ref).norm() / ref.norm()).item()
    assert rel < 2e-2, f"rel_l2={rel:.2e}"


# ---------------------------------------------------------------------------
# Frame ops: Winograd combine (+ bias + residual) and the time interleave (+ bias)
# ---------------------------------------------------------------------------

from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (  # noqa: E402
    cuda_frame_ops_available,
    interleave_bias_cuda,
    wino_combine_cuda,
)

frame_ops = _built(cuda_frame_ops_available(), "trtllm::vae_interleave_bias")


@cuda_op
@pytest.mark.parametrize("C", [256, 160, 48])
@pytest.mark.parametrize("source", ["x", "m2", "m2h", "m4h"])
@pytest.mark.parametrize("extras", ["none", "bias", "residual", "bias_residual"])
def test_wino_combine_is_bitwise(C: int, source: str, extras: str) -> None:
    gen = torch.Generator(device="cuda").manual_seed(C)
    order = 4 if source == "m4h" else 2
    T, P = 2 * order, 7 * 11
    bf = torch.bfloat16
    winograd_dtype = torch.float16 if source.endswith("h") else bf
    rnd = lambda *s: torch.randn(*s, generator=gen, device="cuda").to(bf)  # noqa: E731
    x = rnd(T, P, C) if source == "x" else None
    m = (
        [rnd(T // order, P, C).to(winograd_dtype) for _ in range(order + 2)]
        if source != "x"
        else None
    )
    bias = (0.3 * torch.randn(C, generator=gen, device="cuda")).to(bf) if "bias" in extras else None
    residual = rnd(T, P, C) if "residual" in extras else None
    ref = x if x is not None else wino_combine_reference(m)
    if bias is not None:
        ref = ref + bias.reshape(1, 1, -1)
    if residual is not None:
        ref = ref + residual
    got = wino_combine_cuda(x, m, bias, residual, T, 7, 11)
    assert torch.equal(got, ref)
    # zero-padded output (the downsampler's ZeroPad2d((0, 1, 0, 1))) in the same pass
    padded = wino_combine_cuda(x, m, bias, residual, T, 7, 11, 1, 1).view(T, 8, 12, C)
    ref_img = F.pad(ref.view(T, 7, 11, C).permute(0, 3, 1, 2), (0, 1, 0, 1)).permute(0, 2, 3, 1)
    assert torch.equal(padded, ref_img)


@frame_ops
@pytest.mark.parametrize("C", [1024, 640, 48])
@pytest.mark.parametrize("with_bias", [False, True])
def test_interleave_bias_is_bitwise(C: int, with_bias: bool) -> None:
    gen = torch.Generator(device="cuda").manual_seed(C)
    T, P = 3, 9 * 13
    u = torch.randn(T, P, 2 * C, generator=gen, device="cuda").to(torch.bfloat16)
    bias = (
        (0.3 * torch.randn(2 * C, generator=gen, device="cuda")).to(torch.bfloat16)
        if with_bias
        else None
    )
    assert torch.equal(interleave_bias_cuda(u, bias), interleave_bias_reference(u, bias))


from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import AvgDown3D  # noqa: E402
from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (  # noqa: E402
    avg_down_add_cuda,
)


@frame_ops
@pytest.mark.parametrize("cin_cout_ft", [(160, 160, 1), (160, 320, 2), (320, 640, 2)])
@pytest.mark.parametrize("frames", [4, 1])
@pytest.mark.parametrize("with_bias", [False, True])
def test_avg_down_add_matches_eager(
    cin_cout_ft: tuple[int, int, int], frames: int, with_bias: bool
) -> None:
    cin, cout, ft = cin_cout_ft
    gen = torch.Generator(device="cuda").manual_seed(cin + frames)
    h, w = 12, 20
    bf = torch.bfloat16
    r = (
        torch.randn(1, cin, frames, h, w, generator=gen, device="cuda")
        .to(bf)
        .to(memory_format=torch.channels_last_3d)
    )
    dup = AvgDown3D(cin, cout, factor_t=ft, factor_s=2)
    ref_avg = dup(r)
    x = (
        torch.randn(ref_avg.shape, generator=gen, device="cuda")
        .to(bf)
        .to(memory_format=torch.channels_last_3d)
    )
    bias = torch.randn(cout, generator=gen, device="cuda").to(bf) if with_bias else None
    ref = (
        x.float() + (bias.float().view(1, -1, 1, 1, 1) if with_bias else 0.0) + ref_avg.float()
    ).to(bf)
    pad_t = (ft - frames % ft) % ft
    got = avg_down_add_cuda(
        x.permute(0, 2, 3, 4, 1).reshape(x.shape[2], h // 2 * (w // 2), cout),
        r.permute(0, 2, 3, 4, 1).reshape(frames, h * w, cin),
        h // 2,
        w // 2,
        w,
        ft,
        2,
        dup.group_size,
        pad_t,
        bias,
    )
    got = got.view(1, x.shape[2], h // 2, w // 2, cout).permute(0, 4, 1, 2, 3)
    assert torch.equal(got, ref), "group size 4 means the mean is exact: bitwise with eager"


@cuda_op
@pytest.mark.parametrize("C", [1024, 640, 256])
@pytest.mark.parametrize("reference", [True, False])
def test_cuda_norm_without_activation(C: int, reference: bool) -> None:
    x, gamma, _ = _inputs(C, frames=2, hw=(9, 13))
    expected = rms_norm_reference(x, gamma, C**0.5)
    actual = rms_norm_silu_cuda(x, gamma, None, C**0.5, reference_rounding=reference, silu=False)
    ulp = expected.float().abs().clamp_min(1e-3) * 2**-7
    mismatched = ((actual.float() - expected.float()).abs() > ulp).float().mean().item()
    assert mismatched < (1e-3 if reference else 5e-2)


# --- spatial Winograd F(2x2, 3x3) transforms -------------------------------------------

spatial_op = _built(
    hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "vae_wino_spatial_in"),
    "trtllm::vae_wino_spatial_in",
)
_BT = torch.tensor([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=torch.float32)
_AT = torch.tensor([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=torch.float32)


# F(4x4, 3x3) on points (0, +-3/2, +-5/8, inf), as in the CUDA op and the sm100 path.
_BT4 = torch.tensor(
    [
        [225 / 256, 0, -169 / 64, 0, 1, 0],
        [0, -75 / 128, -25 / 64, 3 / 2, 1, 0],
        [0, 75 / 128, -25 / 64, -3 / 2, 1, 0],
        [0, -45 / 32, -9 / 4, 5 / 8, 1, 0],
        [0, 45 / 32, -9 / 4, -5 / 8, 1, 0],
        [0, 225 / 256, 0, -169 / 64, 0, 1],
    ],
    dtype=torch.float32,
)
_AT4 = torch.tensor(
    [
        [1, 1, 1, 1, 1, 0],
        [0, 3 / 2, -3 / 2, 5 / 8, -5 / 8, 0],
        [0, 9 / 4, 9 / 4, 25 / 64, 25 / 64, 0],
        [0, 27 / 8, -27 / 8, 125 / 512, -125 / 512, 1],
    ],
    dtype=torch.float32,
)
_SPATIAL_MATS = {2: (_BT, _AT), 4: (_BT4, _AT4)}


def _ref_spatial_in(
    frames: torch.Tensor, components: int, row0: int, row1: int, tile: int = 2
) -> torch.Tensor:
    """frames [N, H, W, C] -> V [components*(tile+2)^2, (N/components) * strip tiles, C] in fp32."""
    n, h, w, c = frames.shape
    a = tile + 2
    th, tw = -(-h // tile), -(-w // tile)
    x = frames.permute(0, 3, 1, 2).float()
    tiles = F.unfold(
        F.pad(
            x,
            (1, tw * tile + 1 - w, 1, th * tile + 1 - h),
        ),
        kernel_size=a,
        stride=tile,
    )  # [n, c*a*a, L]
    tiles = tiles.view(n, c, a, a, th, tw)[..., row0:row1, :].reshape(n, c, a, a, -1)
    bt = _SPATIAL_MATS[tile][0].to(frames.device)
    v = torch.einsum("xa,ncabl,yb->nxylc", bt, tiles, bt)
    pairs = n // components
    return (
        v.reshape(components, pairs, a * a, -1, c)
        .permute(0, 2, 1, 3, 4)
        .reshape(components * a * a, -1, c)
    )


def _ref_spatial_out(
    m: torch.Tensor, n: int, h: int, w: int, components: int, row0: int, row1: int, tile: int = 2
) -> torch.Tensor:
    """M [components*(tile+2)^2, (N/components) * strip tiles, C] -> y [N, tile*(row1-row0), W, C]
    in fp32."""
    _, _, c = m.shape
    a = tile + 2
    pairs = n // components
    rows, tw = row1 - row0, -(-w // tile)
    mm = m.float().view(components, a, a, pairs, rows, tw, c)
    at = _SPATIAL_MATS[tile][1].to(m.device)
    y = torch.einsum("rx,nxyphwc,sy->nphrwsc", at, mm, at)
    return y.reshape(n, rows * tile, tw * tile, c)[:, : min(rows * tile, h - tile * row0), :w]


@spatial_op
@pytest.mark.parametrize("C", [8, 160, 256])
@pytest.mark.parametrize("hw", [(6, 10), (5, 9), (12, 20), (9, 13)])
@pytest.mark.parametrize("strip", [(0, 3), (1, 2), (2, 3)])
@pytest.mark.parametrize("tile", [2, 4])
def test_spatial_winograd_transforms(
    C: int, strip: tuple[int, int], hw: tuple[int, int], tile: int
) -> None:
    torch.manual_seed(0)
    n, components = 8, 4
    (h, w), (row0, row1) = hw, strip
    if row1 > -(-h // tile):
        pytest.skip("strip past the last tile row")
    frames = torch.randn(n, h, w, C, device="cuda").to(torch.bfloat16)
    v = torch.ops.trtllm.vae_wino_spatial_in(frames, components, row0, row1, tile)
    v_ref = _ref_spatial_in(frames, components, row0, row1, tile)
    assert v.shape == v_ref.shape
    assert v.dtype == torch.float16
    if tile == 2:
        # bf16 inputs times integer transform entries are exact in fp32: bitwise after rounding.
        assert torch.equal(v.float(), v_ref.to(torch.float16).float())
    else:
        # F(4x4): the separable kernel and the einsum reference sum the six terms in a different
        # order, so a handful of elements land one fp16 ulp apart.
        torch.testing.assert_close(v.float(), v_ref.to(torch.float16).float(), rtol=1e-3, atol=1e-2)

    coords = (tile + 2) ** 2
    m = torch.randn(components * coords, v.shape[1], C, device="cuda").to(torch.float16)
    out = torch.full((n, h, w, C), float("nan"), device="cuda", dtype=torch.bfloat16)
    torch.ops.trtllm.vae_wino_spatial_out(m, out, components, row0, row1, tile)
    got = out[:, tile * row0 : min(tile * row1, h)].float()
    ref = _ref_spatial_out(m, n, h, w, components, row0, row1, tile)
    if tile == 2:
        assert torch.equal(got, ref.to(torch.bfloat16).float())
    else:
        # fp16 products with entries up to 8 summed over six terms: fp32 order-dependent.
        torch.testing.assert_close(got, ref.to(torch.bfloat16).float(), rtol=2e-2, atol=1e-2)
    untouched = torch.cat([out[:, : tile * row0], out[:, tile * row1 :]], dim=1)
    assert torch.isnan(untouched.float()).all()


@spatial_op
@pytest.mark.parametrize("C", [8, 64])
@pytest.mark.parametrize("hw_tile", [((720, 1280), 2), ((1080, 1920), 4)])
def test_spatial_winograd_transforms_large_frames(
    C: int, hw_tile: tuple[tuple[int, int], int]
) -> None:
    """Frames whose tile count exceeds a 65535-wide grid dimension (720x1280 F(2x2): 460,800
    tiles; 1080x1920 F(4x4): 129,600 tiles) with the whole frame as one strip, both channel
    paths (plain for C = 8, staged for C % 64 == 0)."""
    torch.manual_seed(0)
    (h, w), tile = hw_tile
    n, components = 2, 1
    row1 = -(-h // tile)
    frames = torch.randn(n, h, w, C, device="cuda").to(torch.bfloat16)
    v = torch.ops.trtllm.vae_wino_spatial_in(frames, components, 0, row1, tile)
    v_ref = _ref_spatial_in(frames, components, 0, row1, tile).to(torch.float16)
    assert v.shape == v_ref.shape
    if tile == 2:
        assert torch.equal(v.float(), v_ref.float())
    else:
        # F(4x4): the separable kernel and the einsum reference sum the six terms in a different
        # order; with the dyadic (0, +-3/2, +-5/8) coefficients about 6e-5 of the 7e7 elements
        # land one fp16 ulp apart.
        torch.testing.assert_close(v.float(), v_ref.float(), rtol=1e-3, atol=1e-2)
        assert (v != v_ref).float().mean().item() < 2e-4
    del v_ref
    out = torch.full((n, h, w, C), float("nan"), device="cuda", dtype=torch.bfloat16)
    torch.ops.trtllm.vae_wino_spatial_out(v, out, components, 0, row1, tile)
    ref = _ref_spatial_out(v, n, h, w, components, 0, row1, tile)
    assert not torch.isnan(out.float()).any()
    if tile == 2:
        assert torch.equal(out.float(), ref.to(torch.bfloat16).float())
    else:
        torch.testing.assert_close(
            out.float(), ref.to(torch.bfloat16).float(), rtol=2e-2, atol=1e-2
        )


@spatial_op
@pytest.mark.parametrize("cin_cout", [(16, 24), (64, 32)])
@pytest.mark.parametrize("hw", [(6, 10), (5, 9)])
@pytest.mark.parametrize("tile", [2, 4])
def test_spatial_winograd_conv_matches_conv2d(
    cin_cout: tuple[int, int], hw: tuple[int, int], tile: int
) -> None:
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (
        wino_spatial_conv_cuda,
        winograd_spatial_weights,
        winograd_weights,
    )

    torch.manual_seed(1)
    cin, cout = cin_cout
    h, w = hw
    pairs = 3
    weight = (0.1 * torch.randn(cout, cin, 3, 3, 3, device="cuda")).to(torch.bfloat16)
    d = torch.randn(4, pairs, h * w, cin, device="cuda").to(torch.bfloat16)
    got = wino_spatial_conv_cuda(d, winograd_spatial_weights(weight, tile), h, w)
    g = winograd_weights(weight)
    for k in range(4):
        ref = F.conv2d(
            d[k].view(pairs, h, w, cin).permute(0, 3, 1, 2).float(), g[k].float(), None, 1, 1
        )
        ref = ref.permute(0, 2, 3, 1).reshape(pairs, h * w, cout)
        rel = ((got[k].float() - ref).norm() / ref.norm()).item()
        assert rel < 1e-2, f"component {k}: rel_l2={rel:.2e}"


@spatial_op
@pytest.mark.parametrize("cin_cout", [(64, 128), (128, 64)])
@pytest.mark.parametrize("hw", [(6, 10), (9, 13)])
@pytest.mark.parametrize("tile", [2, 4])
@pytest.mark.parametrize("taps", [1, 3])
@pytest.mark.parametrize("with_bias", [False, True])
def test_spatial_winograd_frames_matches_conv3d(
    cin_cout: tuple[int, int], hw: tuple[int, int], tile: int, taps: int, with_bias: bool
) -> None:
    """One output frame of a causal 3x3x3 conv by spatial Winograd alone: the current frame
    (zero history) or the three tap frames summed inside the GEMM."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (
        wino_spatial_conv_frames,
        winograd_spatial_taps_weights,
    )

    torch.manual_seed(3)
    cin, cout = cin_cout
    h, w = hw
    weight = (0.1 * torch.randn(cout, cin, 3, 3, 3, device="cuda")).to(torch.bfloat16)
    bias = torch.randn(cout, device="cuda").to(torch.bfloat16) if with_bias else None
    frames = torch.randn(taps, h * w, cin, device="cuda").to(torch.bfloat16)
    u = winograd_spatial_taps_weights(weight, tile, taps)
    assert u.shape == ((tile + 2) ** 2, taps * cin, cout)
    got = wino_spatial_conv_frames(frames, u, h, w, bias)
    x = frames.view(taps, h, w, cin).permute(3, 0, 1, 2).unsqueeze(0).float()
    x = F.pad(x, (0, 0, 0, 0, 3 - taps, 0))  # zero history before the first frame
    ref = F.conv3d(x, weight.float(), bias.float() if with_bias else None, 1, (0, 1, 1))
    ref = ref[0, :, 0].permute(1, 2, 0).reshape(h * w, cout)
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"rel_l2={rel:.2e}"


@spatial_op
@pytest.mark.parametrize("cin_cout", [(64, 128), (128, 256)])
@pytest.mark.parametrize("hw", [(6, 10), (9, 13)])
@pytest.mark.parametrize("history", [0, 1, 2])
@pytest.mark.parametrize("with_bias", [False, True])
def test_spatial_winograd_taps_matches_conv3d(
    cin_cout: tuple[int, int], hw: tuple[int, int], history: int, with_bias: bool
) -> None:
    """The single-frame path with the history frames in their own buffer (``history`` of them,
    zero frames before): the cuBLAS GEMM + output transform."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (
        wino_spatial_conv_taps,
        winograd_spatial_taps_weights,
    )

    torch.manual_seed(4)
    cin, cout = cin_cout
    h, w = hw
    weight = (0.1 * torch.randn(cout, cin, 3, 3, 3, device="cuda")).to(torch.bfloat16)
    bias = torch.randn(cout, device="cuda").to(torch.bfloat16) if with_bias else None
    frames = torch.randn(3, h * w, cin, device="cuda").to(torch.bfloat16)
    frames[: 2 - history].zero_()
    taps = ([frames[2 - history : 2]] if history else []) + [frames[2:]]
    u = winograd_spatial_taps_weights(weight, 2, 3)
    got = wino_spatial_conv_taps(taps, 2 - history, u, h, w, bias)
    x = frames.view(3, h, w, cin).permute(3, 0, 1, 2).unsqueeze(0).float()
    ref = F.conv3d(x, weight.float(), bias.float() if with_bias else None, 1, (0, 1, 1))
    ref = ref[0, :, 0].permute(1, 2, 0).reshape(h * w, cout)
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"rel_l2={rel:.2e}"


@spatial_op
@pytest.mark.parametrize("frame_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("comps", [4, 6])
@pytest.mark.parametrize("cin_cout_hw_pairs", [(64, 128, 6, 10, 3), (128, 256, 9, 13, 2)])
def test_spatial_winograd_conv_frame_dtypes(
    frame_dtype: torch.dtype, comps: int, cin_cout_hw_pairs: tuple[int, int, int, int, int]
) -> None:
    """The portable path on bf16 and fp16 frames with four or six temporal components."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (
        wino_spatial_conv_cuda,
        winograd_spatial_weights,
        winograd_weights,
    )

    torch.manual_seed(comps)
    cin, cout, h, w, pairs = cin_cout_hw_pairs
    order = comps - 2
    weight = (0.1 * torch.randn(cout, cin, 3, 3, 3, device="cuda")).to(torch.bfloat16)
    d = torch.randn(comps, pairs, h * w, cin, device="cuda").to(frame_dtype)
    u = winograd_spatial_weights(weight, 4, order)
    got = wino_spatial_conv_cuda(d, u, h, w)
    assert got.dtype == frame_dtype
    g = winograd_weights(weight, order, torch.float32)
    ref = torch.stack(
        [
            F.conv2d(d[k].float().view(pairs, h, w, cin).permute(0, 3, 1, 2), g[k], None, 1, 1)
            .permute(0, 2, 3, 1)
            .reshape(pairs, h * w, cout)
            for k in range(comps)
        ]
    )
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"rel_l2={rel:.2e}"


@spatial_op
@pytest.mark.parametrize(
    "cin_cout_hw_pairs", [(64, 256, 6, 12, 4), (128, 256, 9, 16, 2), (512, 512, 24, 40, 2)]
)
def test_spatial_winograd_sm100_matches_reference(
    cin_cout_hw_pairs: tuple[int, int, int, int, int],
) -> None:
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (
        sm100_spatial_filters,
        wino_spatial_conv_sm100,
        winograd_weights,
    )

    _require_sm100()
    torch.manual_seed(3)
    cin, cout, h, w, pairs = cin_cout_hw_pairs
    weight = (0.1 * torch.randn(cout, cin, 3, 3, 3, device="cuda")).to(torch.bfloat16)
    d = torch.randn(4, pairs, h * w, cin, device="cuda").to(torch.bfloat16)
    g = winograd_weights(weight)
    ref = torch.stack(
        [
            F.conv2d(
                d[k].view(pairs, h, w, cin).permute(0, 3, 1, 2).float(), g[k].float(), None, 1, 1
            )
            .permute(0, 2, 3, 1)
            .reshape(pairs, h * w, cout)
            for k in range(4)
        ]
    )
    got = wino_spatial_conv_sm100(d, sm100_spatial_filters(g), h, w).float()
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"rel_l2={rel:.2e}"


@spatial_op
def test_spatial_winograd_sm100_fp16_six_components() -> None:
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (
        sm100_spatial_filters,
        wino_spatial_conv_sm100,
        winograd_weights,
    )

    _require_sm100()
    torch.manual_seed(6)
    cin, cout, h, w, pairs = 128, 256, 12, 16, 1
    weight = (0.1 * torch.randn(cout, cin, 3, 3, 3, device="cuda")).to(torch.bfloat16)
    d = torch.randn(6, pairs, h * w, cin, device="cuda").to(torch.float16)
    g = winograd_weights(weight, 4, torch.float16)
    ref = torch.stack(
        [
            F.conv2d(
                d[k].view(pairs, h, w, cin).permute(0, 3, 1, 2).float(), g[k].float(), None, 1, 1
            )
            .permute(0, 2, 3, 1)
            .reshape(pairs, h * w, cout)
            for k in range(6)
        ]
    )
    got = wino_spatial_conv_sm100(d, sm100_spatial_filters(g), h, w)
    assert got.dtype == torch.float16
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"rel_l2={rel:.2e}"


def _sm100_tile_v(x: torch.Tensor, n: int, th: int, tw: int, c0: int, c1: int) -> torch.Tensor:
    """V[:, tile (n, th, tw), c0:c1] of the F(4x4, 3x3) input transform in fp32."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100 import transforms as wino

    _, h, w, _ = x.shape
    d = torch.zeros(6, 6, c1 - c0, dtype=torch.float32, device=x.device)
    for a in range(6):
        for b in range(6):
            hh, ww = 4 * th - 1 + a, 4 * tw - 1 + b
            if 0 <= hh < h and 0 <= ww < w:
                d[a, b] = x[n, hh, ww, c0:c1].float()
    bt = torch.tensor(wino.WINO[4]["BT"], dtype=torch.float32, device=x.device)
    return torch.einsum("ia,abc,jb->ijc", bt, d, bt).reshape(36, c1 - c0)


def test_spatial_winograd_sm100_in_xform_above_2_31_elements() -> None:
    """Frames whose stride (H*W*C = 1.06e9) puts the third frame past 2^31 elements and a V of
    7.2e9 elements: the transform's frame and tile coordinates must be 64-bit.  Checks the far
    corners of V against the definition.  Needs ~21 GiB of device memory."""

    _require_sm100()
    if torch.cuda.get_device_properties(0).total_memory < 40 * 2**30:
        pytest.skip("needs a 40 GiB GPU")
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100 import transforms as wino

    n, h, w, c = 3, 1080, 1920, 512
    th, tw = -(-h // 4), w // 4
    tiles = n * th * tw
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(n, h, w, c, device="cuda", generator=g).to(torch.float16)
    v = torch.empty(36, tiles, c, dtype=torch.float16, device="cuda")
    assert x.numel() > 2**31 and v.numel() > 2**31
    args = (from_dlpack(x, assumed_align=16), from_dlpack(v, assumed_align=16))
    fn = cute.compile(wino.in_xform, *args, n, h, w, c, 4, 4, 0, make_fake_stream())
    fn(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    for fn_, th_, tw_ in (
        (0, 0, 0),
        (2, th - 1, tw - 1),
        (2, 0, 0),
        (1, th // 2, tw // 2),
        (0, th - 1, tw - 1),
    ):
        t = (fn_ * th + th_) * tw + tw_
        for c0 in (0, c - 64):
            got = v[:, t, c0 : c0 + 64].float()
            ref = _sm100_tile_v(x, fn_, th_, tw_, c0, c0 + 64)
            torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-2)
    del x, v
    torch.cuda.empty_cache()


def test_spatial_winograd_sm100_out_xform_above_2_31_elements() -> None:
    """The output transform at the same frame geometry: y's third frame lies past 2^31
    elements; checks pixels of every frame against the definition.  Needs ~21 GiB."""

    _require_sm100()
    if torch.cuda.get_device_properties(0).total_memory < 40 * 2**30:
        pytest.skip("needs a 40 GiB GPU")
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    from tensorrt_llm._torch.visual_gen.models.wan.wino_sm100 import transforms as wino

    n, h, w, c = 3, 1080, 1920, 512
    th, tw = -(-h // 4), w // 4
    tiles = n * th * tw
    g = torch.Generator(device="cuda").manual_seed(3)
    m = torch.randn(36, tiles, c, device="cuda", generator=g).to(torch.float16)
    y = torch.full((n, h, w, c), float("nan"), dtype=torch.float16, device="cuda")
    args = (from_dlpack(m, assumed_align=16), from_dlpack(y, assumed_align=16))
    fn = cute.compile(wino.out_xform, *args, n, h, w, c, 4, 4, 0, make_fake_stream())
    fn(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    at = torch.tensor(wino.WINO[4]["AT"], dtype=torch.float32, device="cuda")
    for fn_, th_, tw_ in ((0, 0, 0), (2, th - 1, tw - 1), (2, 0, 0), (1, th // 2, tw // 2)):
        t = (fn_ * th + th_) * tw + tw_
        mt = m[:, t, :].float().view(6, 6, c)
        ref = torch.einsum("ri,ijc,sj->rsc", at, mt, at)  # [4, 4, c]
        for r in range(4):
            if 4 * th_ + r < h:
                got = y[fn_, 4 * th_ + r, 4 * tw_ : 4 * tw_ + 4, :].float()
                torch.testing.assert_close(got, ref[r], rtol=2e-2, atol=1e-2)
    assert not torch.isnan(y).any()
    del m, y
    torch.cuda.empty_cache()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [((1, 64, 6, 12, 20), 48, True), ((1, 32, 5, 9, 11), 32, False)])
def test_cudnn_frontend_conv3d_matches_torch(
    dtype: torch.dtype, shape: tuple[tuple[int, ...], int, bool]
) -> None:
    from tensorrt_llm._torch.visual_gen.models.wan.cudnn_conv import (
        conv_frontend,
        cudnn_frontend_available,
    )

    assert cudnn_frontend_available(), (
        "nvidia-cudnn-frontend (a declared requirement) does not import"
    )
    torch.manual_seed(1)
    xshape, cout, with_bias = shape
    cin = xshape[1]
    x = (
        torch.randn(*xshape, device="cuda")
        .to(dtype)
        .contiguous(memory_format=torch.channels_last_3d)
    )
    w = (
        (0.1 * torch.randn(cout, cin, 3, 3, 3, device="cuda"))
        .to(dtype)
        .contiguous(memory_format=torch.channels_last_3d)
    )
    b = (0.3 * torch.randn(cout, device="cuda")).to(dtype) if with_bias else None
    ref = F.conv3d(x, w, b, 1, (0, 1, 1))
    got = conv_frontend(x, w, b, (1, 1, 1), (0, 1, 1), (1, 1, 1))
    assert got.shape == ref.shape and got.dtype == dtype
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 5e-3, f"rel_l2={rel:.2e}"
    # second call reuses the selected plan
    got2 = conv_frontend(x, w, b, (1, 1, 1), (0, 1, 1), (1, 1, 1))
    assert torch.equal(got, got2)


@spatial_op
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("cin_comps", [(256, 4), (512, 6), (64, 4)])
def test_spatial_winograd_fused_matches_reference(
    dtype: torch.dtype, cin_comps: tuple[int, int]
) -> None:
    """The fused F(2,3)-horizontal / direct-vertical 256-channel conv against fp32 conv2d."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import (
        sm100_fused_eligible,
        sm100_fused_filters,
        wino_spatial_conv_fused,
        winograd_weights,
    )

    _require_sm100()
    torch.manual_seed(5)
    cin, comps = cin_comps
    cout, h, w, pairs = 256, 16, 64, 2
    assert sm100_fused_eligible(cin, cout, h, w)
    order = comps - 2
    weight = (0.1 * torch.randn(cout, cin, 3, 3, 3, device="cuda")).to(torch.bfloat16)
    d = torch.randn(comps, pairs, h * w, cin, device="cuda").to(dtype)
    g = winograd_weights(weight, order, torch.float32)
    ref = torch.stack(
        [
            F.conv2d(d[k].float().view(pairs, h, w, cin).permute(0, 3, 1, 2), g[k], None, 1, 1)
            .permute(0, 2, 3, 1)
            .reshape(pairs, h * w, cout)
            for k in range(comps)
        ]
    )
    got = wino_spatial_conv_fused(d, sm100_fused_filters(winograd_weights(weight, order)), h, w)
    assert got.dtype == dtype and got.shape == (comps, pairs, h * w, cout)
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"rel_l2={rel:.2e}"


def test_sm100_fused_needs_two_cin_chunks() -> None:
    """The fused conv's MMA schedule peels a first and a last 32-channel chunk per pass, so
    Cin = 32 (one chunk) is outside its contract and must fall to the unfused path."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import sm100_fused_eligible

    assert not sm100_fused_eligible(32, 256, 8, 32)
    assert sm100_fused_eligible(64, 256, 8, 32)
    assert not sm100_fused_eligible(96, 128, 8, 32)


frame_ops_op = _built(
    hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "vae_unpatchify_clamp"),
    "trtllm::vae_unpatchify_clamp",
)


@frame_ops_op
@pytest.mark.parametrize("patch", [1, 2])
@pytest.mark.parametrize("shape", [(3, 5, 7, 3), (4, 6, 10, 3), (2, 3, 4, 5)])
def test_unpatchify_clamp_into_matches_eager(patch: int, shape: tuple[int, int, int, int]) -> None:
    from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import unpatchify
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_kernels import unpatchify_clamp_into

    t, h, w, c = shape
    torch.manual_seed(4)
    chunks = [
        (torch.randn(1, c * patch * patch, n, h, w, device="cuda") * 1.5)
        .to(torch.bfloat16)
        .contiguous(memory_format=torch.channels_last_3d)
        for n in (1, t)
    ]
    out = torch.full(
        (1, c, 1 + t, h * patch, w * patch), float("nan"), device="cuda", dtype=torch.bfloat16
    ).contiguous(memory_format=torch.channels_last_3d)
    unpatchify_clamp_into(chunks[0], out, 0, patch)
    unpatchify_clamp_into(chunks[1], out, 1, patch)
    ref = torch.clamp(unpatchify(torch.cat(chunks, dim=2), patch), -1.0, 1.0)
    assert torch.equal(out.float(), ref.float())


# ---------------------------------------------------------------------------
# Frame geometry: products read from, and Winograd inputs / rows written to, frames with halo rows
# or columns (the spatially sharded model exchanges the halos)
# ---------------------------------------------------------------------------


def _halo_geometry(h: int, w: int, axis: str):
    from tensorrt_llm._torch.visual_gen.models.wan.jit_norm import FrameGeometry

    if axis == "h":
        return FrameGeometry(h, w, h + 2, w, 1, 0)
    return FrameGeometry(h, w, h, w + 2, 0, 1)


def _embed(frames: torch.Tensor, geo) -> torch.Tensor:
    """Frames ``[..., H*W, C]`` placed inside physical frames ``[..., rows*cols, C]`` whose
    halo positions hold a sentinel the kernel must never read."""
    lead = frames.shape[:-2]
    out = torch.full(
        (*lead, geo.rows, geo.cols, frames.shape[-1]), 7.0, dtype=frames.dtype, device=frames.device
    )
    out[..., geo.row0 : geo.row0 + geo.height, geo.col0 : geo.col0 + geo.width, :] = frames.view(
        *lead, geo.height, geo.width, -1
    )
    return out.view(*lead, geo.frame, frames.shape[-1])


def _interior(phys: torch.Tensor, geo) -> torch.Tensor:
    lead = phys.shape[:-2]
    v = phys.view(*lead, geo.rows, geo.cols, -1)
    return v[..., geo.row0 : geo.row0 + geo.height, geo.col0 : geo.col0 + geo.width, :].reshape(
        *lead, geo.height * geo.width, -1
    )


@cuda_op
@pytest.mark.parametrize("C", [256, 160, 48, 24, 16])
@pytest.mark.parametrize("axis", ["h", "w"])
@pytest.mark.parametrize("order", [2, 4])
def test_wino_norm_geometry_matches_plain(C: int, axis: str, order: int) -> None:
    """Producer: Winograd inputs into halo frames; consumer: products from halo frames; both bitwise."""
    from tensorrt_llm._torch.visual_gen.models.wan import jit_norm

    gen = torch.Generator(device="cuda").manual_seed(C)
    H, W, T = 6, 10, 2 * order
    P = H * W
    bf = torch.bfloat16
    rnd = lambda *s: torch.randn(*s, generator=gen, device="cuda").to(bf)  # noqa: E731
    x = rnd(T, P, C)
    residual = rnd(T, P, C)
    in_bias = (0.3 * torch.randn(C, generator=gen, device="cuda")).to(bf)
    cache = rnd(2, P, C)
    gamma = (1 + 0.1 * torch.randn(C, generator=gen, device="cuda")).to(bf)
    geo = _halo_geometry(H, W, axis)

    # Producer with a Winograd-input geometry.
    s_ref, tail_ref, d_ref = jit_norm.wino_norm_silu(
        x, None, residual, in_bias, cache, gamma, None, C**0.5, T, order=order, fp16_inputs=True
    )
    d_phys = torch.full(
        (order + 2, T // order, geo.frame, C), 7.0, dtype=torch.float16, device="cuda"
    )
    s, tail, d = jit_norm.wino_norm_silu(
        x,
        None,
        residual,
        in_bias,
        cache,
        gamma,
        None,
        C**0.5,
        T,
        order=order,
        fp16_inputs=True,
        inputs_geometry=geo,
    )
    assert d.shape == d_phys.shape
    assert torch.equal(s, s_ref) and torch.equal(tail, tail_ref)
    assert torch.equal(_interior(d, geo), d_ref)

    # Consumer with a products geometry (halo positions hold a sentinel).
    m_ref = [rnd(T // order, P, C).to(torch.float16) for _ in range(order + 2)]
    s_ref, tail_ref, d_ref = jit_norm.wino_norm_silu(
        None, m_ref, residual, in_bias, cache, gamma, None, C**0.5, T, order=order, fp16_inputs=True
    )
    m_phys = list(_embed(torch.stack(m_ref), geo).unbind(0))
    s, tail, d = jit_norm.wino_norm_silu(
        None,
        m_phys,
        residual,
        in_bias,
        cache,
        gamma,
        None,
        C**0.5,
        T,
        order=order,
        fp16_inputs=True,
        products_geometry=geo,
    )
    assert torch.equal(s, s_ref) and torch.equal(tail, tail_ref) and torch.equal(d, d_ref)

    # Both at once: products with halos in, Winograd inputs with halos out.
    s, tail, d = jit_norm.wino_norm_silu(
        None,
        m_phys,
        residual,
        in_bias,
        cache,
        gamma,
        None,
        C**0.5,
        T,
        order=order,
        fp16_inputs=True,
        products_geometry=geo,
        inputs_geometry=geo,
    )
    assert torch.equal(s, s_ref) and torch.equal(_interior(d, geo), d_ref)


@cuda_op
@pytest.mark.parametrize("C", [256, 160, 48, 24, 16])
@pytest.mark.parametrize("axis", ["h", "w"])
def test_rows_and_combine_geometry_match_plain(C: int, axis: str) -> None:
    from tensorrt_llm._torch.visual_gen.models.wan import jit_norm

    gen = torch.Generator(device="cuda").manual_seed(C)
    H, W, T = 6, 10, 3
    P = H * W
    bf = torch.bfloat16
    rnd = lambda *s: torch.randn(*s, generator=gen, device="cuda").to(bf)  # noqa: E731
    geo = _halo_geometry(H, W, axis)
    gamma = (1 + 0.1 * torch.randn(C, generator=gen, device="cuda")).to(bf)
    bias = (0.1 * torch.randn(C, generator=gen, device="cuda")).to(bf)

    # Row norm into halo frames.
    x = rnd(T * P, C)
    residual = rnd(T * P, C)
    s_ref, y_ref = jit_norm.rms_norm_silu_rows(x, gamma, bias, C**0.5, residual=residual)
    out = torch.full((T * geo.frame, C), 7.0, dtype=bf, device="cuda")
    s, y = jit_norm.rms_norm_silu_rows(
        x, gamma, bias, C**0.5, residual=residual, out=out, frames=T, out_geometry=geo
    )
    assert torch.equal(s, s_ref)
    assert torch.equal(_interior(y.view(T, geo.frame, C), geo), y_ref.view(T, P, C))
    halo = y.view(T, geo.rows, geo.cols, C)
    edge = halo[:, 0] if axis == "h" else halo[:, :, 0]
    assert torch.all(edge == 7.0), "halo positions must stay untouched"

    # Combine from halo products (plain and zero-padded output).
    m_ref = [rnd(T, P, C).to(torch.float16) for _ in range(6)]
    T4 = 4
    m_ref = [mk[:1] for mk in m_ref]  # one group of four frames
    bias_c = (0.3 * torch.randn(C, generator=gen, device="cuda")).to(bf)
    res = rnd(T4, P, C)
    for pad in (0, 1):
        ref = jit_norm.combine(None, list(m_ref), bias_c, res, T4, H, W, pad, pad)
        m_phys = list(_embed(torch.stack(m_ref), geo).unbind(0))
        got = jit_norm.combine(None, m_phys, bias_c, res, T4, H, W, pad, pad, products_geometry=geo)
        assert torch.equal(got, ref)


def _peer(comps: int, groups: int, h: int, w: int, axis: str, C: int, dtype, lower: bool):
    """A neighbour's frame buffer (one more line along the split axis than ours, so its
    strides differ) and the view of the halo line that must receive our border."""
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_shard import PeerFrames

    g = _halo_geometry(h + 1, w, axis) if axis == "h" else _halo_geometry(h, w + 1, axis)
    buf = torch.full((comps, groups, g.rows, g.cols, C), 7.0, dtype=dtype, device="cuda")
    if axis == "h":
        line = g.row0 + g.height if lower else g.row0 - 1
        peer = PeerFrames(buf, line, g.col0, g.width)
        got = lambda: buf[:, :, line, g.col0 : g.col0 + g.width, :]  # noqa: E731
    else:
        line = g.col0 + g.width if lower else g.col0 - 1
        peer = PeerFrames(buf, line, g.row0, g.height)
        got = lambda: buf[:, :, g.row0 : g.row0 + g.height, line, :]  # noqa: E731
    return buf, peer, got, line


def _untouched(buf: torch.Tensor, axis: str, line: int) -> bool:
    """Everything but the halo line still holds the sentinel."""
    other = buf.clone()
    if axis == "h":
        other[:, :, line] = 7.0
    else:
        other[:, :, :, line] = 7.0
    return bool(torch.all(other == 7.0))


@cuda_op
@pytest.mark.parametrize("C", [256, 48])
@pytest.mark.parametrize("axis", ["h", "w"])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("one_line", [False, True])
def test_halo_push_lands_in_peer_buffers(C: int, axis: str, order: int, one_line: bool) -> None:
    """The kernel stores the border lines of its Winograd / row output into the neighbours'
    buffers (here: local stand-ins with their own layout) and nothing else; the host-side
    copy writes the same lines. ``one_line``: the shard is a single line along the split
    axis, so the same line goes to both neighbours."""
    from dataclasses import replace

    from tensorrt_llm._torch.visual_gen.models.wan import jit_norm
    from tensorrt_llm._torch.visual_gen.models.wan.wino_vae_shard import DirectFrames, push_border

    gen = torch.Generator(device="cuda").manual_seed(C + order)
    T = 2 * order
    H, W = (6, 10) if not one_line else (1, 10) if axis == "h" else (6, 1)
    P = H * W
    bf = torch.bfloat16
    rnd = lambda *s: torch.randn(*s, generator=gen, device="cuda").to(bf)  # noqa: E731
    geo = _halo_geometry(H, W, axis)
    gamma = (1 + 0.1 * torch.randn(C, generator=gen, device="cuda")).to(bf)
    x = rnd(T, P, C)
    cache = rnd(2, P, C)

    # Winograd inputs: [order + 2, T / order, positions, C] fp16.
    comps, groups = order + 2, T // order
    bl, left, got_l, line_l = _peer(comps, groups, H, W, axis, C, torch.float16, lower=True)
    br, right, got_r, line_r = _peer(comps, groups, H, W, axis, C, torch.float16, lower=False)
    mine = torch.empty((comps, groups, geo.rows, geo.cols, C), dtype=torch.float16, device="cuda")
    push = DirectFrames(mine, None, axis, left, right).push()
    _, _, d = jit_norm.wino_norm_silu(
        x,
        None,
        None,
        None,
        cache,
        gamma,
        None,
        C**0.5,
        T,
        order=order,
        fp16_inputs=True,
        inputs_geometry=replace(geo, push=push),
    )
    phys = d.view(comps, groups, geo.rows, geo.cols, C)
    if axis == "h":
        first = phys[:, :, geo.row0, geo.col0 : geo.col0 + W]
        last = phys[:, :, geo.row0 + H - 1, geo.col0 : geo.col0 + W]
    else:
        first = phys[:, :, geo.row0 : geo.row0 + H, geo.col0]
        last = phys[:, :, geo.row0 : geo.row0 + H, geo.col0 + W - 1]
    assert torch.equal(got_l(), first) and torch.equal(got_r(), last)
    assert _untouched(bl, axis, line_l) and _untouched(br, axis, line_r)
    # The host-side push (one component at a time) writes the same lines.
    bl.fill_(7.0)
    br.fill_(7.0)
    for k in range(comps):
        one = replace(push, left=push.left[k : k + 1], right=push.right[k : k + 1])
        push_border(one, phys[k], geo)
    assert torch.equal(got_l(), first) and torch.equal(got_r(), last)

    # Rows: [T, positions, C] bf16 into a frame-buffer window of T frames.
    bl, left, got_l, line_l = _peer(1, T, H, W, axis, C, bf, lower=True)
    br, right, got_r, line_r = _peer(1, T, H, W, axis, C, bf, lower=False)
    out = torch.full((T * geo.frame, C), 7.0, dtype=bf, device="cuda")
    push = DirectFrames(out.view(1, T, geo.rows, geo.cols, C), None, axis, left, right).push()
    jit_norm.rms_norm_silu_rows(
        x.view(T * P, C),
        gamma,
        None,
        C**0.5,
        out=out,
        frames=T,
        out_geometry=replace(geo, push=push),
    )
    phys = out.view(1, T, geo.rows, geo.cols, C)
    if axis == "h":
        first = phys[:, :, geo.row0, geo.col0 : geo.col0 + W]
        last = phys[:, :, geo.row0 + H - 1, geo.col0 : geo.col0 + W]
    else:
        first = phys[:, :, geo.row0 : geo.row0 + H, geo.col0]
        last = phys[:, :, geo.row0 : geo.row0 + H, geo.col0 + W - 1]
    assert torch.equal(got_l(), first) and torch.equal(got_r(), last)
    assert _untouched(bl, axis, line_l) and _untouched(br, axis, line_r)
    # A window in the middle of a deeper buffer: only its groups are addressed.
    bl.fill_(7.0)
    br.fill_(7.0)
    deep = DirectFrames(out.view(1, T, geo.rows, geo.cols, C), None, axis, left, right)
    narrow = deep.push(1, T - 2)
    assert narrow.left.shape[1] == T - 2 and narrow.left.data_ptr() == push.left[:, 1].data_ptr()
