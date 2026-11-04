# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the layer_norm catalog entry.

The entry owns what is certified: `CELLS` is the list and `reference` is what
the kernel is gated against. This file owns what a cell cannot carry --
turning a spec into real tensors, and the band -- plus the measurements the
`note` states.

Weight and bias are drawn over the call site's own checkpoint tensors'
ranges. At the indexer site (44 q_input_norm tensors) the affine is near unit,
and the input carries the spread.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.layer_norm import layer_norm as op

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "layer_norm requires a CUDA device"

DEVICE = torch.device("cuda")
HIDDEN = 6144
EPS = 1e-6

#: Over layers.*.attention.indexer.q_input_norm.{weight,bias}, all 44 indexer layers (bf16).
Q_WEIGHT_LO, Q_WEIGHT_HI = 0.890625, 1.140625
Q_BIAS_LO, Q_BIAS_HI = -0.08935547, 0.08935547
#: The same 44 tensors on the NVFP4 checkpoint, another training snapshot: the
#: weights sit inside the range above, the bias reaches one bf16 step past it.
#: Its attention_norm weights (the hidden-state model below) measured the same
#: range and percentiles as the FP8 checkpoint's.
NVFP4_Q_WEIGHT_LO, NVFP4_Q_WEIGHT_HI = 0.89453125, 1.1328125
NVFP4_Q_BIAS_LO, NVFP4_Q_BIAS_HI = -0.08740234, 0.08984375

#: The indexer's input is the layer's attention-normed hidden state: an
#: RMS-normalized row times attention_norm's weight, whose magnitudes over the
#: checkpoint run from about 0.01 to 2.4 on nearly every column and reach 14.3
#: on a handful. Row r is further scaled by _ROW_SCALES[r % 4], so its
#: variance is about 1e-6, 1, 1e-2 or 1e-4 times the hidden state's own (which
#: is 1.9-3.1 here). At the hidden state's own variance eps moves the result
#: by about a millionth: a reference at eps 1e-5, 1e-7 or 0 was measured inside
#: the gate over 9216 such rows. At 1e-6 of it eps decides the answer. Row 0
#: is the small one, so a one-row call sees eps too.
_ROW_SCALES = (1e-3, 1.0, 1e-1, 1e-2)

#: Significand bits (the implicit one included) and the exponent of the
#: smallest subnormal, per output dtype.
_FORMAT = {torch.bfloat16: (8, -133), torch.float16: (11, -24), torch.float32: (24, -149)}


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _affine(gen, width=HIDDEN, dtype=torch.bfloat16, checkpoint="fp8"):
    if checkpoint == "nvfp4":
        (w_lo, w_hi) = (NVFP4_Q_WEIGHT_LO, NVFP4_Q_WEIGHT_HI)
        (b_lo, b_hi) = (NVFP4_Q_BIAS_LO, NVFP4_Q_BIAS_HI)
    else:
        (w_lo, w_hi), (b_lo, b_hi) = (Q_WEIGHT_LO, Q_WEIGHT_HI), (Q_BIAS_LO, Q_BIAS_HI)
    weight = torch.rand(width, generator=gen, device=DEVICE) * (w_hi - w_lo) + w_lo
    bias = torch.rand(width, generator=gen, device=DEVICE) * (b_hi - b_lo) + b_lo
    return weight.to(dtype), bias.to(dtype)


def _hidden_rows(rows, width, gen):
    """Rows shaped like the attention-normed hidden state, row r scaled by `_ROW_SCALES[r % 4]`."""
    columns = torch.rand(width, generator=gen, device=DEVICE) * (2.5 - 0.01) + 0.01
    columns[torch.randperm(width, generator=gen, device=DEVICE)[:8]] = 14.3125
    scale = torch.tensor(_ROW_SCALES, device=DEVICE).repeat((rows + 3) // 4)[:rows, None]
    return torch.randn(rows, width, generator=gen, device=DEVICE) * columns * scale


def _build(spec, seed, rows=None):
    """One cell's operands, at `rows` rows (the cell's own by default)."""
    gen = _gen(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    if rows is not None:
        shape = (rows, *shape[1:])
    assert spec["site"] == "indexer_q", spec["site"]
    x = _hidden_rows(shape[0], shape[-1], gen).to(dtype)
    weight, bias = _affine(
        gen, width=shape[-1], dtype=dtype, checkpoint=spec.get("checkpoint", "fp8")
    )
    return x, weight, bias, spec.get("eps", EPS)


def _spacing(t):
    """The gap from each element of `t` to the next representable value of its dtype, as fp64."""
    bits, tiny = _FORMAT[t.dtype]
    _, exponent = torch.frexp(t.double())
    exponent = torch.where(
        t == 0, torch.full_like(exponent, tiny), (exponent - bits).clamp_min(tiny)
    )
    return torch.ldexp(torch.ones_like(exponent, dtype=torch.float64), exponent)


def _terms(x, weight, bias, eps):
    """Per element, the magnitudes the kernel's fp32 arithmetic rounds (fp64, from the inputs)."""
    xd = x.double()
    mean = xd.mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt((xd - mean).square().mean(dim=-1, keepdim=True) + eps)
    terms = ((xd - mean).abs() + xd.abs().mean(dim=-1, keepdim=True)) * rstd * weight.double().abs()
    return terms + bias.double().abs() if bias is not None else terms


def _compare(out, ref, x, weight, bias, eps):
    """Within one output step of the reference plus 2^-19 of the rounded terms, and rarely off.

    Kernel and reference evaluate the same expression and round it once: the
    kernel in fp32, the reference in fp64. They differ by the kernel's fp32
    evaluation error plus at most one step of the output dtype (both round to
    nearest, so to the same value or to neighbours). That fp32 error is bounded
    by the magnitudes it rounds: the row mean is a sum of N values and carries
    about log2(N) = 13 roundings of mean|x| into every `x - mean`; the
    variance's sum about as many more, halved through the sqrt; the
    approximate sqrt and division of rstd a few; and the closing multiply-add
    one of `|x_hat * w| + |b|`. That is about 25 roundings of 2^-24, so 2^-19
    (32 of them) of S = (|x - mean| + mean|x|) * rstd * |w| + |b| bounds it.
    Where `x_hat * w` and `b` cancel, the result is far below S and can sit
    many output steps from the reference inside that allowance: on the sweep
    cell's 9216 x 6144 input, 21 elements sit more than one step off (up to
    26; 92 in another draw), each a result within 2e-6 of zero, off by at most
    1.2e-8 and using under 1% of its allowance. Elsewhere the allowance is one
    step. The fp32 error itself measured at most 2.1 x 2^-24 of S there. On
    the 32768 x 6144 cell's input (the NVFP4 checkpoint's ranges): 117 more
    than one step off (up to 965 steps), each within 1.9e-6 of zero, off by at
    most 1.5e-8 and using at most 1.1% of its allowance; the fp32 error (the
    op on fp32 operands against the exact value) at most 4.3 x 2^-24 of S.
    An element one step off reads just under one allowance by construction,
    `spacing / (spacing + 2^-19 S)`: the worst such reading there was 0.9997,
    the same as `F.layer_norm`'s.

    An element differs from the correctly rounded value only when a rounding
    boundary falls inside the fp32 error, about 2^-24 / 2^-p of the time for a
    p-bit significand: measured 1.6e-5 to 3.0e-5 at bf16. The count allows 64
    times that, 2^(p - 18): 2^-10 at bf16 and 2^-7 at fp16. At fp32 the step
    is the error's own size and the count does not apply.

    What it rejects, measured over the indexer's 9216 x 6144 sweep rows: eps
    1e-5, 1e-7 or 0 for 1e-6 lands 12,000-32,000 allowances off on the
    small-variance rows (7,000-21,000 in a single row); a variance
    over N - 1 puts 1.8% of the elements off; rounding toward zero 50%;
    `x_hat` rounded to bf16 before the affine 27% and 177 allowances off; the
    weight or bias shifted one column, a dropped bias or an RMS norm 89-98%
    and 16,000-128,000 allowances off. Each holds in a single 6144-wide row.
    On the 32768-row cell's output, rows past 9215 read wrapped at 9216 or one
    row early, or rows from 16384 left at zero, fall outside the gate in every
    such row (up to 4.4e6, 7.3e6 and 255 allowances off).
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    allowance = _spacing(ref) + 2.0**-19 * _terms(x, weight, bias, eps)
    excess = ((out.double() - ref.double()).abs() / allowance).max().item()
    assert excess <= 1.0, f"{excess:.4g} allowances from the reference; the gate is 1"
    bits, _ = _FORMAT[out.dtype]
    if bits < 24:
        off = int((out != ref).sum())
        assert off * 2 ** (18 - bits) <= out.numel(), (
            f"{off} of {out.numel()} elements differ from the correctly rounded value; "
            f"at most 2^{bits - 18} of them may"
        )


def _check(x, weight, bias, eps):
    """One call, gated; the operands are left as they were."""
    before = [t.clone() for t in (x, weight, bias) if t is not None]
    out = op(x, weight, bias, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.is_contiguous() and out.data_ptr() != x.data_ptr()
    after = [t for t in (x, weight, bias) if t is not None]
    assert all(torch.equal(a, b) for a, b in zip(before, after)), "inputs must not be mutated"
    assert torch.equal(out, op(x, weight, bias, eps)), "two launches must agree bitwise"
    _compare(out, op.reference(x, weight, bias, eps), x, weight, bias, eps)
    return out


def _words(t):
    """`t`'s bits as int64 words over whole contiguous rows, for the sweep's compare."""
    assert t.is_contiguous() and (t.shape[-1] * t.element_size()) % 8 == 0
    return t.view(torch.int64)


def _sweep(spec, seed):
    """Every row count of `sweep_rows`, each its own call on the leading rows of one input.

    The call at the top of the range is gated against the reference; each
    smaller one has to equal the leading rows of that result bit for bit. The
    kernel computes each row in its own program, so a row count that changed
    any row's answer would be a grid or specialization defect, not rounding.

    Each call is compared as whole contiguous rows of int64 words (the same
    bits, four bf16 to a word) and reduced with `any()`, which is still exact:
    over 1..32768 on sm_103 the sweep took 6.0 s this way, against 18.6 s
    with bf16 `!=` and `.sum()` and 3.2 s for the op calls alone. Each
    verdict stays on the device, so the sweep synchronizes once; the first
    failing row count is re-run to count its elements.
    """
    lo, hi = spec["sweep_rows"]
    assert spec["shape"][0] == hi, "a sweep's shape is its largest call"
    x, weight, bias, eps = _build(spec, seed)
    full = _check(x, weight, bias, eps)
    full_words = _words(full)
    differs = [
        torch.ne(_words(op(x[:rows], weight, bias, eps)), full_words[:rows]).any()
        for rows in range(lo, hi + 1)
    ]
    failing = torch.nonzero(torch.stack(differs)).flatten()
    if failing.numel():
        rows = lo + int(failing[0])
        count = int((op(x[:rows], weight, bias, eps) != full[:rows]).sum())
        raise AssertionError(
            f"rows={rows}: {count} elements differ from the leading rows of the {hi}-row call"
        )


def _replays(spec, seed):
    """Each row count of `graph_rows` in its own CUDA graph, replayed on new data.

    Captured and replayed under `inference_mode`, as the engine runs its
    decode graphs, after one warm-up call on the capture stream, as the
    engine's warm-up does: that call is where the Triton kernel compiles. The
    replay has to equal an eager call on the same data bit for bit -- a graph
    that froze anything of its capture would differ -- and is gated against
    the reference like any other output.
    """
    stream = torch.cuda.Stream()
    with torch.inference_mode():
        for rows in spec["graph_rows"]:
            x, weight, bias, eps = _build(spec, seed + rows, rows=rows)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                op(x, weight, bias, eps)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = op(x, weight, bias, eps)
            x.copy_(_build(spec, seed + rows + 1, rows=rows)[0])
            graph.replay()
            eager = op(x, weight, bias, eps)
            assert torch.equal(out, eager), f"rows={rows}: graph replay differs from eager"
            _compare(out, op.reference(x, weight, bias, eps), x, weight, bias, eps)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    if "sweep_rows" in cell.spec:
        _sweep(cell.spec, seed)
    elif "graph_rows" in cell.spec:
        _replays(cell.spec, seed)
    else:
        _check(*_build(cell.spec, seed))


def test_the_gate_sees_eps():
    """The scaled rows are what make eps a value the cells certify, not a passenger."""
    for rows in (1, 16):
        x, weight, bias, eps = _build(
            dict(shape=(rows, HIDDEN), dtype=torch.bfloat16, site="indexer_q"), seed=11 + rows
        )
        out = op(x, weight, bias, eps)
        _compare(out, op.reference(x, weight, bias, eps), x, weight, bias, eps)
        for wrong in (1e-5, 1e-7, 0.0):
            with pytest.raises(AssertionError, match="allowances from the reference"):
                _compare(out, op.reference(x, weight, bias, wrong), x, weight, bias, eps)


def _fp64(x, weight, bias, eps, ddof=0):
    xd = x.double()
    mean = xd.mean(dim=-1, keepdim=True)
    var = (xd - mean).square().sum(dim=-1, keepdim=True) / (xd.shape[-1] - ddof)
    out = (xd - mean) / torch.sqrt(var + eps) * weight.double()
    return out + bias.double() if bias is not None else out


def _toward_zero(value):
    """fp64 -> bf16 rounding toward zero instead of to nearest."""
    bits = value.float().contiguous().view(torch.int32) & ~0xFFFF
    return bits.view(torch.float32).to(torch.bfloat16)


def test_wrong_variants_land_outside_the_gate():
    """The cells' gate separates this computation from its neighbours, at the indexer's values."""
    spec = dict(shape=(256, HIDDEN), dtype=torch.bfloat16, site="indexer_q")
    x, weight, bias, eps = _build(spec, seed=3)

    xd, wd, bd = x.double(), weight.double(), bias.double()
    mean = xd.mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt((xd - mean).square().mean(dim=-1, keepdim=True) + eps)
    wrong = {
        "rmsnorm": xd * torch.rsqrt(xd.square().mean(dim=-1, keepdim=True) + eps) * wd + bd,
        "bias_dropped": _fp64(x, weight, None, eps),
        "weight_as_delta": (xd - mean) * rstd * (wd + 1.0) + bd,
        "weight_shifted": _fp64(x, torch.roll(weight, 1), bias, eps),
        "variance_over_n_minus_1": _fp64(x, weight, bias, eps, ddof=1),
        "x_hat_rounded_to_bf16": ((xd - mean) * rstd).to(torch.bfloat16).double() * wd + bd,
    }
    wrong = {name: value.to(torch.bfloat16) for name, value in wrong.items()}
    wrong["rounded_toward_zero"] = _toward_zero(_fp64(x, weight, bias, eps))

    out = op(x, weight, bias, eps)
    _compare(out, op.reference(x, weight, bias, eps), x, weight, bias, eps)
    for name, value in wrong.items():
        with pytest.raises(AssertionError):
            _compare(value, op.reference(x, weight, bias, eps), x, weight, bias, eps)


def test_the_reference_models_layer_norm_is_inside_the_same_gate():
    """The reference model computes this step as torch's `F.layer_norm` on the
    bf16 operands (fp32 inside, one rounding). It sits as close to the exact
    value as this op does, so the two are interchangeable at this gate."""
    x, weight, bias, eps = _build(
        dict(shape=(1024, HIDDEN), dtype=torch.bfloat16, site="indexer_q"), seed=21
    )
    ref = op.reference(x, weight, bias, eps)
    _compare(op(x, weight, bias, eps), ref, x, weight, bias, eps)
    _compare(F.layer_norm(x, (HIDDEN,), weight, bias, eps), ref, x, weight, bias, eps)


def test_eps_is_inside_the_square_root():
    """Only observable where `var` is comparable to `eps`: a near-constant fp32
    row. The kernel must sit orders of magnitude nearer `1/sqrt(var + eps)`
    than `1/(sqrt(var) + eps)`."""
    gen = _gen(17)
    x = torch.full((4, HIDDEN), 1.0, device=DEVICE, dtype=torch.float32)
    x[:, ::2] += 1e-3
    weight, bias = _affine(gen, dtype=torch.float32)

    out = op(x, weight, bias, EPS).double()
    xd, wd, bd = x.double(), weight.double(), bias.double()
    mean = xd.mean(dim=-1, keepdim=True)
    var = (xd - mean).square().mean(dim=-1, keepdim=True)
    inside = (xd - mean) / torch.sqrt(var + EPS) * wd + bd
    outside = (xd - mean) / (torch.sqrt(var) + EPS) * wd + bd
    to_inside = (out - inside).abs().max().item()
    to_outside = (out - outside).abs().max().item()
    assert to_outside > 100.0 * to_inside, (to_inside, to_outside)


def test_constant_rows_return_the_bias_exactly():
    """Zero variance: `x - mean` is exactly 0, so the row is `bias` bit for bit
    (a zero row included, the likeliest content of a padded graph row)."""
    gen = _gen(31)
    weight, bias = _affine(gen)
    for value in (2.5, 0.0):
        x = torch.full((4, HIDDEN), value, device=DEVICE, dtype=torch.bfloat16)
        out = op(x, weight, bias, EPS)
        assert torch.isfinite(out.float()).all()
        assert torch.equal(out, bias.expand_as(out).contiguous()), value


def test_runs_unchanged_under_inference_mode():
    gen = _gen(55)
    x = torch.randn(256, HIDDEN, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    eager = op(x, weight, bias, EPS)
    with torch.inference_mode():
        inferred = op(x, weight, bias, EPS).clone()
    assert torch.equal(inferred, eager)


def test_exactly_one_kernel_is_launched():
    """The atomicity claim: one `_layer_norm_fwd_1pass_kernel`, no repacking copy."""
    from torch.profiler import ProfilerActivity, profile

    gen = _gen(66)
    x = torch.randn(1024, HIDDEN, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    op(x, weight, bias, EPS)  # warm the Triton JIT
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        op(x, weight, bias, EPS)
        torch.cuda.synchronize()

    launched = [
        event.name
        for event in prof.events()
        if str(event.device_type) == "DeviceType.CUDA" and event.device_time_total > 0
    ]
    assert launched == ["_layer_norm_fwd_1pass_kernel"], launched


def test_capturable_in_a_cuda_graph():
    gen = _gen(77)
    x = torch.randn(512, HIDDEN, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    static_x = x.clone()
    op(static_x, weight, bias, EPS)  # compile before capture
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = op(static_x, weight, bias, EPS)

    static_x.copy_(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, op(x, weight, bias, EPS))


def test_rejects_a_weight_whose_length_is_not_the_normalized_width():
    """Loud: `_layer_norm_fwd` asserts the shapes, with no message."""
    gen = _gen(88)
    x = torch.randn(8, HIDDEN, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    with pytest.raises(AssertionError):
        op(x, weight[:-1].contiguous(), bias, EPS)
    with pytest.raises(AssertionError):
        op(x, weight, bias[:-1].contiguous(), EPS)


def test_rejects_a_feature_dim_that_does_not_fit_the_fused_kernel():
    width = 65536
    x = torch.randn(2, width, device=DEVICE).to(torch.bfloat16)
    weight = torch.ones(width, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.zeros(width, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="doesn't support feature dim"):
        op(x, weight, bias, EPS)


def test_accepts_an_fp8_activation_and_returns_fp8():
    """Measured, not certified: `empty_like(x)` makes the output e4m3, the
    arithmetic is still fp32, and the store lands within one e4m3 ulp (2^-3
    relative, 2^-9 smallest subnormal) of the exact value."""
    gen = _gen(111)
    x = torch.randn(64, HIDDEN, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight, bias = _affine(gen)
    x8 = x.to(torch.float8_e4m3fn)

    out = op(x8, weight, bias, EPS)

    assert out.dtype == torch.float8_e4m3fn
    assert torch.isfinite(out.float()).all()
    expected = _fp64(x8.float(), weight, bias, EPS)
    deviation = (out.float().double() - expected).abs()
    assert (deviation <= 2.0**-3 * expected.abs() + 2.0**-9).all(), deviation.max().item()
