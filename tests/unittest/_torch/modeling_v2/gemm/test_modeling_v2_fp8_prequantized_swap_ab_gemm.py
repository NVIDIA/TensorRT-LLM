# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_prequantized_swap_ab_gemm catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.fp8_prequantized_swap_ab_gemm import (  # noqa: E501
    fp8_prequantized_swap_ab_gemm as op,
)
from tensorrt_llm.quantization.utils.fp8_utils import (
    resmooth_to_fp8_e8m0,
    transform_sf_into_required_layout,
)

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — needs the path declared above
    ULP,
    assert_within_ulp,
    certifying,
)

assert torch.cuda.is_available(), "fp8_prequantized_swap_ab_gemm requires a CUDA device"

#: The band, in bf16 ulp of the row's largest magnitude; derived in `_compare`.
ELEMENT_ULP, RMS_ULP = 4.0, 2.0

#: Fills the gaps of an MN-major scale's storage: every byte 0xFF, a UE8M0 NaN, so a kernel
#: that read a gap for a valid row would put NaN in that row.
_GAP = -1

#: Rows per slice of a sweep's band: an fp32 slice is at most 4096 x 49152 x 4 B = 0.75 GiB.
_BAND_ROWS = 4096


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference, in the output's own ulp.

    Kernel and reference consume bit-identical operands, and every product is exact in
    fp32 on both sides: a dequantized operand is an e4m3 code (4 significant bits) times a
    power of two, so a product has at most 8. What differs is how the 6144 (8192, 24576,
    3072, 2048, 1536, 1024, 768) products are summed in fp32: DeepGEMM's block-scaled
    tensor-core MMA accumulating in tensor memory, against cuBLAS SGEMM with TF32 off. The
    two fp32 sums agree to within about 3e-5 of the row's scale (a float32 output measured at
    most 92 fp32 ulp off at K 6144, 102 at K 8192, 245 at K 24576, 50 at K 3072, 27 at
    K 1024, 20 at K 768), and each side then rounds once to
    bf16. Where the two sums straddle a bf16 rounding boundary they land one bf16 spacing
    apart, which is between 1 and 2 ulp of the row's scale as `ULP` counts it, so both gates
    are against the row's scale
    (`assert_within_ulp`): 4 ulp element-wise, twice one spacing; 2 ulp of relative RMS,
    every element one spacing off.

    Measured worst case over every m in 1..9216 at the sixteen N x K, on the eager cells' own
    operands: 1.39 to 1.94 ulp element-wise and 0.0095 to 0.062 RMS; over the 32768-row calls
    of the windowed cells, 1.64 to 1.98 and 0.010 to 0.057. Wrong operands land
    far outside, at M 1024: activation scale bytes reversed in each int32, 480-768 ulp
    element-wise and 303-308 RMS; the weight scale shifted one 128-row block, 371-491 and
    284-298; one K block's activation scale doubled, 131-305 and 15.6-123; the weight's two K
    halves swapped, 524-659 and 362 (at 2048x6144, 6144x1536, 6144x6144, 8192x6144,
    6144x8192, 49152x6144, 6144x24576 and the five tep8 shards; bytes reversed not at K 768,
    whose six blocks fill no second int32). At the five tep8 shards also: adjacent K blocks'
    activation scales swapped, 657-746 and 304-309; the last K block's activation scale
    doubled, 179-277 and 31-91.
    `test_the_band_refuses_misplaced_scales` pins three of them and
    `test_the_band_refuses_k768s_last_block_scale_doubled` one at K 768.
    """
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


def _ue8m0_exponent(x: torch.Tensor) -> torch.Tensor:
    """Biased exponent of the smallest power of two >= x, for positive normal fp32 x.

    Read off the bits rather than through `log2`: `torch.log2` on CUDA is inexact at exact
    powers of two, and `ceil` then moves the scale up a binade.
    """
    bits = x.float().contiguous().view(torch.int32)
    return ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    return (exponent.to(torch.int32) << 23).view(torch.float32)


def _activation(m: int, k: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 rows quantized the way `fp8_quantize_1x128_packed_ue8m0(x, False)` quantizes them.

    Per 1x128 block: the fp32 amax, floored at 1e-10; the scale 2^ceil(log2(amax / 448));
    the codes e4m3(x / scale). The codes and the unpacked scales are bitwise that op's at
    K 6144, 2048 and 1536 and eleven M from 1 to 9216, 1..5, 17 and 9215 among them, and at
    K 8192 and 24576 and thirteen M from 1 to 32768, and at K 768, 1024, 3072 and 6144 and
    fourteen M from 1 to 32768 (measured; the op is not called here), and `_pack` gives its
    strides; there, the GEMM on the op's own (codes, scale) was bitwise the GEMM on these.
    Returns the codes and the [M, K/128] biased exponents; `_pack` lays
    the exponents out. Block magnitudes span 2^-4..2^4, so neighbouring blocks carry
    different scales and a scale applied to the wrong block shows.
    """
    x = torch.randn(m, k, generator=gen, device="cuda")
    spread = torch.randint(-4, 5, (m, k // 128, 1), generator=gen, device="cuda")
    x = (x.view(m, k // 128, 128) * torch.exp2(spread.float())).to(torch.bfloat16).float()
    exponent = _ue8m0_exponent(x.abs().amax(dim=2).clamp(min=1e-10) / 448.0)
    codes = (x / _pow2(exponent)[..., None]).to(torch.float8_e4m3fn).view(m, k)
    return codes, exponent


def _pack(exponent: torch.Tensor, empty: int = _GAP & 0xFF) -> torch.Tensor:
    """[M, K/128] exponents -> int32 [M, ceil(K/512)] with strides (1, align(M, 4)).

    The quantizer's layout and allocation: the storage ends at row M of the last column,
    and every gap between columns holds `_GAP`. Where K/128 is not a multiple of four (K 768,
    six blocks), the last int32 of each row has empty byte slots past the last block; they
    hold `empty`, by default 0xFF like the gaps (the quantizer writes 0 there).
    """
    m, blocks = exponent.shape
    cols = -(-blocks // 4)
    m_aligned = (m + 3) // 4 * 4
    storage = torch.full(((cols - 1) * m_aligned + m,), _GAP, dtype=torch.int32, device="cuda")
    packed = storage.as_strided((m, cols), (1, m_aligned))
    padded = torch.full((m, cols * 4), empty, dtype=torch.uint8, device="cuda")
    padded[:, :blocks] = exponent
    packed.copy_(padded.view(torch.int32))
    return packed


def _weight(n: int, k: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """A checkpoint-like FP8 weight, prepared the way the target's loader prepares it.

    The checkpoint holds e4m3 [N, K] and `.weight_scale`, one fp32 power-of-two scale per
    128 x 128 block shaped [N/128, 1, K/128, 1], which the loader views [N/128, K/128]; the
    load path is then `FP8BlockScalesLinearMethod.transform_weights`' DeepGEMM branch,
    `resmooth_to_fp8_e8m0` and `transform_sf_into_required_layout(recipe=(1, 128, 128),
    is_sfa=False)`. Block magnitudes span 2^-3..2^4 and the draw is cut at 5 sigma, so a
    block's scale is 2^(spread - 12) and the scales cover 2^-15..2^-8, the range the
    checkpoint's 404 block scales span; asserted.
    """
    w = torch.randn(n, k, generator=gen, device="cuda").clamp(-5.0, 5.0) * 0.02
    spread = torch.randint(-3, 5, (n // 128, 1, k // 128, 1), generator=gen, device="cuda")
    blocks = w.view(n // 128, 128, k // 128, 128) * torch.exp2(spread.float())
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
    checkpoint_scale = _pow2(_ue8m0_exponent(amax / 448.0))
    codes = (blocks / checkpoint_scale).to(torch.float8_e4m3fn).view(n, k)
    codes, scale = resmooth_to_fp8_e8m0(codes, checkpoint_scale.view(n // 128, k // 128))
    drawn = _ue8m0_exponent(scale) - 127
    assert (drawn.min().item(), drawn.max().item()) == (-15, -8), "block scales off range"
    weight_scale = transform_sf_into_required_layout(
        scale, mn=n, k=k, recipe=(1, 128, 128), is_sfa=False
    )
    assert weight_scale.dtype == torch.int32 and weight_scale.stride() == (1, n)
    return codes, weight_scale


def _flags(spec) -> tuple[torch.dtype, bool]:
    return spec["output_dtype"], spec["disable_ue8m0_cast"]


def _rows(spec) -> list[int]:
    """The m a sweep drives: every m of `m_range`, or, with `m_window` w, three in each w-row
    window [w j + 1, w j + w] of the range: its two edges and the interior row
    w j + 2 + j % (w - 2), so that across windows the rows take every residue mod w.
    """
    lo, hi = spec["m_range"]
    w = spec.get("m_window")
    if w is None:
        return list(range(lo, hi + 1))
    assert (lo - 1) % w == 0 and hi % w == 0, "a windowed range is whole windows"
    return [
        m
        for j in range((lo - 1) // w, hi // w)
        for m in (w * j + 1, w * j + 2 + j % (w - 2), w * j + w)
    ]


def _sweep(spec, seed: int) -> None:
    """Each m of `_rows`, its own call, on the leading m rows of one operand set.

    The call at the top of the range is gated against the reference, and every other m has
    to return bitwise the leading rows of that call. Row i of the product reads only row i
    of the activation and of its scale, so a correct kernel does that whatever tile layout
    it picks for m, and it carries the gate to every m exactly: the per-m gates below are
    `assert_within_ulp`'s two over the leading m rows, from running maxima and sums. Each
    m's scale is packed for m rows, strides (1, align(m, 4)), as the quantizer returns it.

    The band is taken `_BAND_ROWS` rows at a time, `_compare` on each slice (its RMS gate per
    slice is at least as strict as over the whole call), so the fp32 and fp64 copies stay a
    slice's size at N 49152. The bitwise check compares int64 words and reads the verdicts
    once, at the end.
    """
    lo, hi = spec["m_range"]
    rows = _rows(spec)
    assert rows[-1] == hi
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight, weight_scale = _weight(spec["n"], spec["k"], gen)
    codes, exponent = _activation(hi, spec["k"], gen)
    scale = _pack(exponent)
    full = op(codes, scale, weight, weight_scale, *_flags(spec))
    assert full.shape == (hi, spec["n"]) and full.dtype is spec["output_dtype"]
    ref = op.reference(codes, scale, weight, weight_scale, *_flags(spec))

    element = torch.empty(hi, dtype=torch.float32, device="cuda")
    err, mag = (torch.empty(hi, dtype=torch.float64, device="cuda") for _ in range(2))
    for i in range(0, hi, _BAND_ROWS):
        out_i, ref_i = full[i : i + _BAND_ROWS], ref[i : i + _BAND_ROWS]
        _compare(out_i, ref_i)
        diff, ref_i = out_i.float() - ref_i.float(), ref_i.float()
        element[i : i + _BAND_ROWS] = diff.abs().amax(dim=1) / ref_i.abs().amax(dim=1).clamp_min(
            1e-9
        )
        err[i : i + _BAND_ROWS] = diff.double().pow(2).sum(dim=1)
        mag[i : i + _BAND_ROWS] = ref_i.double().pow(2).sum(dim=1)
    del ref, diff, ref_i
    ulp = ULP[torch.bfloat16]
    element = element.cummax(0).values
    rms = (err.cumsum(0) / mag.cumsum(0).clamp_min(1e-18)).sqrt()
    worst = int(element[lo - 1 : hi].argmax()) + lo
    assert element[worst - 1] <= ELEMENT_ULP * ulp, (
        f"m={worst}: {element[worst - 1].item() / ulp:.3f} ulp element-wise exceeds {ELEMENT_ULP}"
    )
    worst = int(rms[lo - 1 : hi].argmax()) + lo
    assert rms[worst - 1] <= RMS_ULP * ulp, (
        f"m={worst}: relative RMS {rms[worst - 1].item() / ulp:.3f} ulp exceeds {RMS_ULP}"
    )

    words = full.view(torch.int64)
    differs = torch.empty(len(rows), dtype=torch.bool, device="cuda")
    for i, m in enumerate(rows):
        out = op(codes[:m], _pack(exponent[:m]), weight, weight_scale, *_flags(spec))
        assert out.shape == (m, spec["n"])
        differs[i] = torch.ne(out.view(torch.int64), words[:m]).any()
    differ = [rows[i] for i in differs.nonzero().flatten().tolist()]
    assert not differ, (
        f"m={differ[:8]}{'...' if len(differ) > 8 else ''}: rows differ from the leading rows "
        f"of the {hi}-row call; the kernel is no longer batch-invariant, and these m were not "
        "gated against the reference"
    )


def _replays(spec, seed: int) -> None:
    """Each decode batch captured in its own CUDA graph, then replayed on new operands.

    The new codes and scales are written into the captured inputs -- the scale through its
    logical elements, its gaps untouched -- as into an engine's static buffers. The replay
    has to equal an eager call on the same operands bit for bit (a graph that froze
    something of the capture would differ), and is gated against the reference. Each
    capture follows one call on the capture stream, as the engine's warm-up does: DeepGEMM
    compiles a tile layout on its first call, which a capture cannot contain.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    k = spec["k"]
    weight, weight_scale = _weight(spec["n"], k, gen)
    stream = torch.cuda.Stream()
    for m in spec["graph_m"]:
        codes, exponent = _activation(m, k, gen)
        a_in, s_in = codes.clone(), _pack(exponent)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(a_in, s_in, weight, weight_scale, *_flags(spec))
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = op(a_in, s_in, weight, weight_scale, *_flags(spec))
        codes, exponent = _activation(m, k, gen)
        a_in.copy_(codes)
        s_in.copy_(_pack(exponent))
        graph.replay()
        eager = op(codes, _pack(exponent), weight, weight_scale, *_flags(spec))
        assert torch.equal(out, eager), f"m={m}: graph replay differs from eager"
        _compare(out, op.reference(codes, s_in, weight, weight_scale, *_flags(spec)))


def _points(spec, seed: int) -> None:
    """A few row counts at the cell's flag: gated, and bitwise the other flag's result."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight, weight_scale = _weight(spec["n"], spec["k"], gen)
    output_dtype, flag = _flags(spec)
    for m in spec["m_points"]:
        codes, exponent = _activation(m, spec["k"], gen)
        scale = _pack(exponent)
        out = op(codes, scale, weight, weight_scale, output_dtype, flag)
        other = op(codes, scale, weight, weight_scale, output_dtype, not flag)
        assert torch.equal(out, other), f"m={m}: disable_ue8m0_cast={flag} differs from {not flag}"
        _compare(out, op.reference(codes, scale, weight, weight_scale, output_dtype, flag))


def _cell_params():
    """`op.CELLS`, a cell whose spec says `large` under pytest's `large` marker: pre-merge
    deselects it, and the stage that runs the rest drives it."""
    return [
        pytest.param(c, marks=pytest.mark.large) if c.spec.get("large") else c for c in op.CELLS
    ]


@pytest.mark.parametrize("cell", _cell_params(), ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    if "m_range" in cell.spec:
        _sweep(cell.spec, seed)
    elif "graph_m" in cell.spec:
        _replays(cell.spec, seed)
    else:
        _points(cell.spec, seed)


def _operands(m=64, n=2048, k=6144, seed=11):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight, weight_scale = _weight(n, k, gen)
    codes, exponent = _activation(m, k, gen)
    return codes, exponent, weight, weight_scale


def _rolled_rows(weight_scale: torch.Tensor) -> torch.Tensor:
    """The weight scale shifted one 128-row block, back in its (1, N) layout."""
    return weight_scale.roll(128, 0).t().contiguous().t()


@pytest.mark.parametrize(
    "wrong",
    ["activation scale bytes reversed", "weight scale one block off", "one K block doubled"],
)
def test_the_band_refuses_misplaced_scales(wrong) -> None:
    """The gate sees a scale applied to the wrong block, not only a wrong product."""
    codes, exponent, weight, weight_scale = _operands(m=1024)
    ref = op.reference(codes, _pack(exponent), weight, weight_scale, torch.bfloat16, True)
    scale = _pack(exponent)
    if wrong == "activation scale bytes reversed":
        scale = _pack(exponent.view(1024, -1, 4).flip(-1).reshape(1024, -1))
    elif wrong == "weight scale one block off":
        weight_scale = _rolled_rows(weight_scale)
    else:
        scale = _pack(torch.cat([exponent[:, :1] + 1, exponent[:, 1:]], dim=1))
    out = op(codes, scale, weight, weight_scale, torch.bfloat16, True)
    with pytest.raises(AssertionError):
        _compare(out, ref)


def test_the_band_refuses_k768s_last_block_scale_doubled() -> None:
    """At K 768 the last K block's scale shares a half-empty int32 with block 4's only; the
    gate sees that one scale doubled."""
    codes, exponent, weight, weight_scale = _operands(m=1024, n=6144, k=768)
    ref = op.reference(codes, _pack(exponent), weight, weight_scale, torch.bfloat16, True)
    doubled = _pack(torch.cat([exponent[:, :5], exponent[:, 5:] + 1], dim=1))
    out = op(codes, doubled, weight, weight_scale, torch.bfloat16, True)
    with pytest.raises(AssertionError):
        _compare(out, ref)


@pytest.mark.parametrize("m", [1, 301, 4097])
def test_k768s_empty_scale_slots_are_never_read(m) -> None:
    """At K 768 (six K blocks, two int32 per scale row) bytes 2 and 3 of each row's second
    int32 are empty. The quantizer and the weight transform write 0 there; 0xFF there, in
    the activation's scale or the weight's, gives bitwise the same output, while 0xFF in a
    live slot (block 5's) turns the rows into NaN.
    """
    codes, exponent, weight, weight_scale = _operands(m=m, n=6144, k=768)
    assert weight_scale.shape == (6144, 2) and bool((weight_scale[:, 1] >> 16 == 0).all())
    filled = weight_scale.clone()
    filled[:, 1] |= -65536  # 0xFFFF0000: the two empty bytes
    assert filled.stride() == weight_scale.stride()
    out = op(codes, _pack(exponent, empty=0), weight, weight_scale, torch.bfloat16, True)
    assert not out.isnan().any()
    for a_empty, w_scale in ((0xFF, weight_scale), (0, filled), (0xFF, filled)):
        again = op(codes, _pack(exponent, empty=a_empty), weight, w_scale, torch.bfloat16, True)
        assert torch.equal(again, out), f"activation empty 0x{a_empty:02X}, weight empty set"
    live = _pack(torch.cat([exponent[:, :5], torch.full_like(exponent[:, 5:], 255)], dim=1))
    assert op(codes, live, weight, weight_scale, torch.bfloat16, True).isnan().any(dim=1).all()


def test_a_row_major_activation_scale_is_restrided_inside_the_call() -> None:
    """Any strides are accepted: the runner copies the scale into (1, align(M, 4)) first."""
    codes, exponent, weight, weight_scale = _operands(m=301)
    scale = _pack(exponent)
    row_major = scale.contiguous()
    assert row_major.stride() != scale.stride()
    assert torch.equal(
        op(codes, row_major, weight, weight_scale, torch.bfloat16, True),
        op(codes, scale, weight, weight_scale, torch.bfloat16, True),
    )


def test_a_row_strided_activation_reads_its_stride() -> None:
    """Codes that are a column slice of a wider buffer: the TMA descriptor takes the stride."""
    codes, exponent, weight, weight_scale = _operands(m=301)
    wide = torch.zeros(301, codes.shape[1] + 128, dtype=codes.dtype, device="cuda")
    wide[:, : codes.shape[1]] = codes
    view = wide[:, : codes.shape[1]]
    assert not view.is_contiguous()
    assert torch.equal(
        op(view, _pack(exponent), weight, weight_scale, torch.bfloat16, True),
        op(codes, _pack(exponent), weight, weight_scale, torch.bfloat16, True),
    )


def test_the_r128c4_scale_layout_is_refused() -> None:
    """`use_r128c4_layout=True`, the quantizer's default, returns a 1-D uint8 scale of
    align(M, 128) * align(K/32, 4) bytes; the runner's re-stride raises on it.
    """
    codes, _, weight, weight_scale = _operands(m=16)
    r128c4 = torch.zeros(128 * 192, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match=r"dimensionality of sizes \(1\) must match"):
        op(codes, r128c4, weight, weight_scale, torch.bfloat16, True)


def test_an_fp32_activation_scale_is_refused_at_the_call_sites_flag() -> None:
    """With disable_ue8m0_cast=True DeepGEMM takes int32 scales only."""
    codes, exponent, weight, weight_scale = _operands(m=16)
    with pytest.raises(RuntimeError, match="Unsupported architecture or scaling factor types"):
        op(codes, _pow2(exponent), weight, weight_scale, torch.bfloat16, True)


def test_a_weight_scale_that_is_not_mn_major_is_refused() -> None:
    codes, exponent, weight, weight_scale = _operands(m=16)
    with pytest.raises(RuntimeError, match=r"sf\.stride\(-2\) == 1 or mn == 1"):
        op(codes, _pack(exponent), weight, weight_scale.contiguous(), torch.bfloat16, True)
