# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_rmsnorm catalog entry.

The reference is built here from native torch only, in the same algebra the
DeepSeek-V4.1-Flash checkpoint's own `inference/model.py` `RMSNorm.forward`
uses: upcast to fp32, square-mean over the last dim, `rsqrt(mean + eps)`, scale
by weight, downcast. It shares nothing with the op under test.

Tolerance is `torch.testing.assert_close`'s default for the dtype, with **no
loosening anywhere in this file**. That is measured, not assumed: the domain
probe applied the default elementwise bound to every V4.1 width at M=1 and
M=4096 in bf16 and fp32 and found 0 elements outside in all 16, while a
sum-instead-of-mean variant put 660,410 of 660,480 elements outside at
(5120, M=129) — essentially all of them, with the 70 survivors being positions
whose correct value is already near zero.

The V4.1 cases below exist because this checkpoint sets `rms_norm_eps = 1e-20`
-- fourteen orders below anything this entry had previously been certified at,
and epsilon is the one term a kernel can silently drop, clamp or downcast
without any ordinary activation noticing.
"""

import os

import pytest
import torch

from tensorrt_llm._torch.staircase.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm

assert torch.cuda.is_available(), "flashinfer_rmsnorm requires a CUDA device"

#: DeepSeek-V4.1-Flash `config.json` -> `text_config.rms_norm_eps`.
V41_EPS = 1e-20

#: Every distinct RMSNorm width in the V4.1 text backbone, derived from the
#: checkpoint's `inference/model.py`, not assumed:
#:   RMSNorm(args.dim)        -> attn_norm, ffn_norm (per layer), final norm
#:   RMSNorm(q_lora_rank)     -> q_norm
#:   RMSNorm(head_dim)        -> kv_norm and the per-layer compressor norm
#:   RMSNorm(index_head_dim)  -> indexer k_norm
V41_WIDTHS = [
    pytest.param(5120, id="attn_ffn_final_5120"),
    pytest.param(1280, id="q_norm_1280"),
    pytest.param(512, id="kv_and_compressor_norm_512"),
    pytest.param(128, id="indexer_k_norm_128"),
]

#: Row counts a served target reaches: single decode, captured decode batches
#: and prefill chunks. A norm runs at every one of them.
ROW_BUCKETS = [1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096]


def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32-accumulated reference: x / sqrt(mean(x^2, -1) + eps) * weight."""
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


def _check(x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    out = flashinfer_rmsnorm(x, weight, eps)
    ref = _ref_rmsnorm(x, weight, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    torch.testing.assert_close(out, ref)


def test_bf16_2d() -> None:
    torch.manual_seed(0)
    # decode-like (few tokens) and prefill-like (many tokens) shapes
    for num_tokens, hidden in [(1, 4096), (4, 5120), (2048, 4096)]:
        x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
        _check(x, w, 1e-6)


def test_bf16_3d_qk_norm_shape() -> None:
    torch.manual_seed(1)
    x = torch.randn(16, 32, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    _check(x, w, 1e-5)


def test_bf16_unaligned_hidden() -> None:
    # hidden sizes not divisible by the 128-bit vector width
    torch.manual_seed(2)
    for hidden in [111, 1152]:
        x = torch.randn(16, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
        _check(x, w, 1e-6)


def test_bf16_strided_rows() -> None:
    # last-dim contiguous slice of a wider buffer (row stride != hidden)
    torch.manual_seed(3)
    buf = torch.randn(8, 8192, dtype=torch.bfloat16, device="cuda")
    x = buf[:, :4096]
    assert not x.is_contiguous() and x.stride(-1) == 1
    w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    _check(x, w, 1e-6)


def test_fp16_2d() -> None:
    torch.manual_seed(4)
    for num_tokens, hidden in [(2, 4096), (1024, 2048)]:
        x = torch.randn(num_tokens, hidden, dtype=torch.float16, device="cuda")
        w = torch.randn(hidden, dtype=torch.float16, device="cuda")
        _check(x, w, 1e-6)


def test_fp32_2d() -> None:
    torch.manual_seed(5)
    for num_tokens, hidden in [(2, 4096), (1024, 2048)]:
        x = torch.randn(num_tokens, hidden, dtype=torch.float32, device="cuda")
        w = torch.randn(hidden, dtype=torch.float32, device="cuda")
        _check(x, w, 1e-6)


# ── DeepSeek-V4.1-Flash column: four widths, every engine row bucket, eps 1e-20 ──


def _seeded(m: int, width: int, seed: int, dtype=torch.bfloat16):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, width, generator=gen, device="cuda", dtype=dtype)
    w = torch.randn(width, generator=gen, device="cuda", dtype=dtype)
    return x, w


@pytest.mark.parametrize("width", V41_WIDTHS)
@pytest.mark.parametrize("m", ROW_BUCKETS)
def test_v41_width_by_row_bucket(width: int, m: int) -> None:
    """Every V4.1 norm width at every row count the engine serves, at eps 1e-20.

    The cross-product rather than a sample: a norm runs at every token count, so
    certifying the widths at one row count and the rows at one width would leave
    exactly the combinations the target uses uncertified.
    """
    x, w = _seeded(m, width, width * 8191 + m)
    _check(x, w, V41_EPS)


@pytest.mark.parametrize("width", V41_WIDTHS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_v41_dtypes(width: int, dtype: torch.dtype) -> None:
    """bf16 is the ONLY dtype V4.1 feeds this op; fp16 and fp32 are extra coverage.

    An earlier revision of this test claimed fp32 was a real V4.1 path because
    `Compressor.__init__` builds `wkv` with `dtype=torch.float32` when
    `compress_ratio > 1`. That reads the declaration and not the call site.
    `Compressor.forward` captures `dtype = x.dtype` on entry, pools in fp32, and
    returns `self.norm(kv.to(dtype))`; the `ratio == 1` branch returns
    `self.norm(self.wkv(x))` on a bf16 `wkv`. The norm sees **bf16** on both
    branches, and every other V4.1 placement is on the bf16 text path too.

    The fp16/fp32 cases stay because they cost nothing and the op supports them,
    but they are coverage of the entry, not of this checkpoint's semantics.
    """
    x, w = _seeded(64, width, width + 17, dtype)
    _check(x, w, V41_EPS)


@pytest.mark.parametrize("width", V41_WIDTHS)
@pytest.mark.parametrize("c", [1e-9, 1e-10, 1e-11, 1e-12])
def test_v41_epsilon_is_used_at_its_stated_magnitude(width: int, c: float, capfd) -> None:
    """eps 1e-20 is honoured, and this is what proves it rather than assuming it.

    On ordinary activations `mean(x^2)` is O(1) and an epsilon of 1e-20 changes
    nothing, so every other case in this file would pass identically against a
    kernel that dropped epsilon entirely. Here each row is a constant `c`, so
    `mean(x^2) = c^2` exactly, and `c` is walked down through and past 1e-10
    where `c^2` meets eps. In that regime an eps-honouring kernel and an
    eps-dropping one diverge by up to 0.99 on a unit-scale output.

    The assertion is two-sided: the kernel must match the eps-honouring
    reference, AND the eps-dropping reference must be far outside the gate --
    otherwise the case has not discriminated anything and it must not be allowed
    to report a pass.
    """
    x = torch.full((8, width), c, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(width, device="cuda", dtype=torch.bfloat16)
    got = flashinfer_rmsnorm(x, w, V41_EPS).float()
    with_eps = _ref_rmsnorm(x, w, V41_EPS).float()
    without_eps = _ref_rmsnorm(x, w, 0.0).float()

    separation = (with_eps - without_eps).abs().max().item()
    d_honoured = (got - with_eps).abs().max().item()
    d_dropped = (got - without_eps).abs().max().item()
    with capfd.disabled():
        print(
            f"\n    width {width} c={c:.0e} mean(x^2)={c * c:.0e}: "
            f"|kernel - ref(eps=1e-20)|={d_honoured:.3e} "
            f"|kernel - ref(eps=0)|={d_dropped:.3e} "
            f"(the two references differ by {separation:.3e})"
        )
    assert separation > 1e-3, (
        f"the eps-honouring and eps-dropping references differ by only {separation:.3e} at "
        f"c={c:.0e}, so this case cannot tell them apart and proves nothing about epsilon"
    )
    torch.testing.assert_close(got, with_eps)
    assert d_dropped > separation / 2, (
        f"the kernel is as close to the eps-DROPPING reference ({d_dropped:.3e}) as to the "
        f"eps-honouring one; epsilon is not being applied as the contract claims"
    )


@pytest.mark.parametrize("width", V41_WIDTHS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_v41_all_zero_row_is_zero_not_nan(width: int, dtype: torch.dtype) -> None:
    """A zero row is the degenerate case epsilon exists to make finite.

    With epsilon the row normalizes to 0; with `eps = 0` it is `0 * rsqrt(0)` =
    `0 * inf` = NaN. So this is a second, independent witness that epsilon
    reaches the kernel -- and it is the one that matters operationally, because
    a NaN row propagates through the whole forward.
    """
    x = torch.zeros(4, width, device="cuda", dtype=dtype)
    w = torch.ones(width, device="cuda", dtype=dtype)
    got = flashinfer_rmsnorm(x, w, V41_EPS)
    assert not got.isnan().any(), "a zero row produced NaN; epsilon was not applied"
    assert not got.isinf().any(), "a zero row produced inf; epsilon was not applied"
    torch.testing.assert_close(got, _ref_rmsnorm(x, w, V41_EPS))


def test_v41_zero_row_does_not_disturb_its_neighbours() -> None:
    """Normalization is per row, so one degenerate row must not change the others."""
    x, w = _seeded(8, 5120, 4242)
    x[3] = 0.0
    got = flashinfer_rmsnorm(x, w, V41_EPS)
    ref = _ref_rmsnorm(x, w, V41_EPS)
    assert not got[3].any(), "the zeroed row did not normalize to zero"
    others = [r for r in range(8) if r != 3]
    torch.testing.assert_close(got[others], ref[others])


@pytest.mark.parametrize("width", V41_WIDTHS)
def test_v41_pdl_does_not_change_the_result(width: int) -> None:
    """PDL is on by default in the trtllm op, so the target runs the PDL path.

    The contract says it is a scheduling choice. Asserted here rather than
    inherited: both settings must be bit-identical, at this checkpoint's eps.
    """
    x, w = _seeded(129, width, width + 99)
    previous = os.environ.get("TRTLLM_ENABLE_PDL")
    try:
        os.environ["TRTLLM_ENABLE_PDL"] = "1"
        on = flashinfer_rmsnorm(x, w, V41_EPS)
        os.environ["TRTLLM_ENABLE_PDL"] = "0"
        off = flashinfer_rmsnorm(x, w, V41_EPS)
    finally:
        if previous is None:
            os.environ.pop("TRTLLM_ENABLE_PDL", None)
        else:
            os.environ["TRTLLM_ENABLE_PDL"] = previous
    assert torch.equal(on, off), (
        f"PDL changed the result by {(on.float() - off.float()).abs().max().item():.3e} at "
        f"width {width}; it is documented as a scheduling choice only"
    )


def test_v41_strided_rows_at_checkpoint_width() -> None:
    """The q/kv norms read column slices of wider buffers, so row stride != width."""
    gen = torch.Generator(device="cuda").manual_seed(777)
    buf = torch.randn(129, 8192, generator=gen, device="cuda", dtype=torch.bfloat16)
    x = buf[:, :1280]
    assert not x.is_contiguous() and x.stride(-1) == 1
    w = torch.randn(1280, generator=gen, device="cuda", dtype=torch.bfloat16)
    _check(x, w, V41_EPS)


#: Every input dtype the contract makes a claim about, and the claim.
#: `None` means "accepted"; a string is the exact `KeyError` text observed.
#: `float8_e5m2` is here because the inherited contract said it raised and the
#: probe measured otherwise -- the claim that turned out to be false is exactly
#: the one that has to become a test.
DTYPE_DOMAIN = [
    pytest.param(torch.bfloat16, None, id="bfloat16_accepted"),
    pytest.param(torch.float16, None, id="float16_accepted"),
    pytest.param(torch.float32, None, id="float32_accepted"),
    pytest.param(torch.float64, "torch.float64", id="float64_KeyError"),
    pytest.param(torch.int8, "torch.int8", id="int8_KeyError"),
    pytest.param(torch.uint8, "torch.uint8", id="uint8_KeyError"),
    pytest.param(torch.float8_e5m2, None, id="float8_e5m2_ACCEPTED_not_rejected"),
    pytest.param(torch.float8_e4m3fn, None, id="float8_e4m3fn_ACCEPTED_not_rejected"),
]


@pytest.mark.parametrize("dtype,key_error", DTYPE_DOMAIN)
def test_dtype_domain(dtype: torch.dtype, key_error: str | None) -> None:
    """Each dtype the contract describes, driven, with the exact error text asserted.

    A contract sentence saying "this is rejected" promises a guard the caller
    then does not write, so it is the expensive kind to get wrong -- and one of
    these was. The inherited bullet claimed `float8_e5m2` raises `KeyError`; it
    is accepted. Both fp8 types are accepted and normalize the stored fp8 values
    as if they were the real numbers, so a caller who forgets to dequantize gets
    a plausible answer rather than an error.
    """
    width = 512
    gen = torch.Generator(device="cuda").manual_seed(2026)
    if dtype in (torch.int8, torch.uint8):
        x = torch.ones(4, width, device="cuda", dtype=dtype)
        w = torch.ones(width, device="cuda", dtype=dtype)
    else:
        x = torch.randn(4, width, generator=gen, device="cuda").to(dtype)
        w = torch.randn(width, generator=gen, device="cuda").to(dtype)

    if key_error is not None:
        with pytest.raises(KeyError, match=key_error.replace(".", r"\.")):
            flashinfer_rmsnorm(x, w, V41_EPS)
        return

    got = flashinfer_rmsnorm(x, w, V41_EPS)
    torch.cuda.synchronize()
    assert got.dtype == dtype, "an accepted dtype must come back as itself"
    assert not got.float().isnan().any()


def test_fp8_input_is_accepted_and_is_a_caller_hazard(capfd) -> None:
    """The fp8 acceptance is documented rather than guarded, so it is tested.

    No wrapper guard was added for this: two already-certified targets
    (`deepseek_v3/r1_0528_nvfp4/sm_103/dep4` and `gpt_oss/gpt_oss_120b/sm_103/tp1`)
    call this wrapper on hot paths, and an assert in a shared wrapper would
    change behaviour for artifacts whose gates are hour-scale accuracy runs. The
    hazard is pinned here instead, so that if a future flashinfer starts
    rejecting fp8 this test fails and the contract gets corrected rather than
    silently rotting.
    """
    width = 512
    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        gen = torch.Generator(device="cuda").manual_seed(2026)
        x = torch.randn(4, width, generator=gen, device="cuda").to(dtype)
        w = torch.randn(width, generator=gen, device="cuda").to(dtype)
        got = flashinfer_rmsnorm(x, w, V41_EPS)
        torch.cuda.synchronize()
        ref = _ref_rmsnorm(x.float(), w.float(), V41_EPS)
        d = (got.float() - ref).abs().max().item()
        with capfd.disabled():
            print(
                f"\n    {str(dtype).replace('torch.', '')}: ACCEPTED, returned "
                f"{str(got.dtype).replace('torch.', '')}, {d:.3e} from an fp32 reference "
                f"on a scale of {ref.abs().max().item():.3e}"
            )
        assert got.dtype == dtype
        assert not got.float().isnan().any(), (
            "an fp8 input produced NaN; the contract describes it as returning a plausible result"
        )


def test_v41_default_gate_sees_a_sum_instead_of_mean_error(capfd) -> None:
    """The gate is the dtype default, so show it is tight enough to catch a real slip.

    Reducing with `sum` instead of `mean` is the transcription error this op is
    most likely to meet: same shape, same values, same epsilon, and off by
    exactly `sqrt(width)`. It must be outside the default bound essentially
    everywhere, otherwise the unloosened gate is not doing any work.
    """
    width, m = 5120, 129
    x, w = _seeded(m, width, 31337)
    got = flashinfer_rmsnorm(x, w, V41_EPS).float()
    ref = _ref_rmsnorm(x, w, V41_EPS).float()
    xf = x.float()
    wrong = xf * torch.rsqrt(xf.square().sum(-1, keepdim=True) + V41_EPS)
    wrong = (wrong * w.float()).to(x.dtype).float()

    bound = 1e-5 + 1.6e-2 * ref.abs()
    over_correct = int(((got - ref).abs() > bound).sum().item())
    over_wrong = int(((wrong - ref).abs() > bound).sum().item())
    ratio = (wrong - ref).abs().max().item() / max((got - ref).abs().max().item(), 1e-30)
    with capfd.disabled():
        print(
            f"\n    width {width} M={m}: correct puts {over_correct} of {ref.numel()} elements "
            f"over the DEFAULT gate | sum-not-mean puts {over_wrong} over | "
            f"wrong/correct max_abs = {ratio:.1f}x"
        )
    assert over_correct == 0, f"the correct result put {over_correct} elements over the gate"
    # Essentially all, not literally all: the survivors are positions whose
    # correct value is already near zero, where a sqrt(width) scaling error is
    # still small in absolute terms. 660,410 of 660,480 measured on sm_103.
    assert over_wrong > ref.numel() * 0.99, (
        f"the sum-instead-of-mean control put only {over_wrong} of {ref.numel()} elements over "
        f"the default gate; the gate cannot see a reduction error"
    )
