# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the default target's ple_gate_conv_inject_norm variant.

Cells are gated against the float64 reference that rounds where the op rounds:
the stream (`hc_out`) and the normed stream in units of their row's scale, the
published conv state per element (see `_compare`); a verify cell's staged conv
input is checked bitwise against the state it published. Beyond the cells: the
staged conv input is what the conv consumed, `out` writes only its rows of a
larger buffer, the operand-width guard, and graph capture at the 461-slot pool and
over a rank's whole pool (the tep4 decode and MTP7 verify graphs).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ple.ple_gate_conv_inject_norm import (  # noqa: E501
    CKPT_CLAMP_MIN,
    CKPT_DILATION,
    CKPT_HC_COUNT,
    CKPT_HIDDEN,
    ENGINE_SLOTS,
    TEP4_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ple.ple_gate_conv_inject_norm import (  # noqa: E501
    ple_gate_conv_inject_norm as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "ple_gate_conv_inject_norm requires a CUDA device"

DEV = "cuda"
EPS = 1e-6
PAD = -1
GAMMA_RMS = 0.477


def _ulp_ratio(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-element distance from `ref` in units of one ulp of `|ref|`."""
    eps = torch.finfo(ref.dtype).eps
    tiny = torch.finfo(ref.dtype).tiny
    ref64 = ref.double()
    return (out.double() - ref64).abs() / (eps * ref64.abs() + tiny)


def _stats(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    ratio = _ulp_ratio(out, ref)
    return ratio.max().item(), (out != ref).double().mean().item()


#: The band. Every rounding point of the composition is kept, so the op and the
#: float64 reference differ only where an fp32 reduction (three sums of squares
#: and the gate's dot per stream) or fp32 `exp` lands a value across a bf16
#: midpoint -- measured on 2e-5..2.3e-4 of the elements. A flipped upstream
#: element then propagates, and through `hc + ple` it meets cancellation
#: (`hc ~ -ple`), so a per-element ulp is unbounded there: 73 ulp of a
#: 0.003-sized element in a row whose largest is ~10. The stream and its norm
#: are therefore gated in units of their row's scale (`assert_within_ulp`):
#: 2 ulp of the row maximum per element and 0.1 ulp relative RMS. The same
#: steps unfused (one bf16 kernel per step) sit at 0.87 / 0.33 of a row-max ulp
#: against the same reference (measured: normed / hc_out at 128 rows), the op
#: at 0.72 / 0.38. The published conv state is one rounding of the conv-input
#: norm, whose only perturbation is a one-ulp flip of a gated element: 2 ulp
#: per element (measured worst 1.78, the unfused steps 1.56).
ROW_ULP, RMS_ULP, STATE_ULP = 2.0, 0.1, 2.0
#: The per-element row-max gate is a tail statistic, and a context-shaped cell
#: draws ~100x more elements than a 128-row one. Against the same float64
#: reference over three seeds at 16384 rows, the unfused steps reach 2.91
#: row-max ulp (normed) and the op 2.94; at 8192 rows 1.89 / 1.89.
#: Cells above 1024 rows are therefore gated at 4; the RMS gate is unchanged.
#: The default target's per-rank pools (1874 / 850 slots) sit inside the same gates; a
#: reference reading each sequence's neighbouring slot falls far outside the RMS gate
#: (measured).
CONTEXT_ROWS, CONTEXT_ROW_ULP = 1024, 4.0


def _compare(out, ref) -> None:
    row_ulp = CONTEXT_ROW_ULP if out[0].shape[0] > CONTEXT_ROWS else ROW_ULP
    for name, got, want in zip(("hc_out", "normed", "conv_states"), out, ref):
        assert got.dtype == want.dtype and got.shape == want.shape, name
        if not got.numel():
            continue
        if name == "conv_states":
            worst = _ulp_ratio(got, want).max().item()
            assert worst <= STATE_ULP, f"{name}: {worst:.4g} ulp"
        else:
            assert_within_ulp(got, want, row_ulp, RMS_ULP)


def _build(spec, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    gc = torch.Generator().manual_seed(seed)
    lengths = spec["lengths"]
    c, d, dtype = CKPT_HC_COUNT, CKPT_HIDDEN, torch.bfloat16
    w = c * d
    rows = sum(lengths)
    num_seqs = len(lengths)
    pool_slots = spec.get("pool_slots", num_seqs + 2)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device=DEV) * scale).to(dtype)

    # Streams at different scales, as the trained residual's are.
    scales = torch.tensor([0.5 * 2.0 ** (i % 4) for i in range(c)], device=DEV)
    hc = (rand(rows, w, scale=3.0).float() * scales.repeat_interleave(d)).to(dtype)
    key = rand(rows, w)
    value = rand(rows, d, scale=0.5)
    norms = [rand(w, scale=GAMMA_RMS) for _ in range(4)]
    conv_weight = rand(w, 1, 4, scale=0.3)
    conv_states = rand(pool_slots, w, 3 * CKPT_DILATION)
    order = (torch.randperm(pool_slots - 2, generator=gc) + 1).tolist()
    slots = ([0, pool_slots - 1] + order)[:num_seqs] if pool_slots >= 64 else list(range(num_seqs))
    slots = torch.tensor(slots, dtype=torch.int32, device=DEV)
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    qsl = torch.tensor(starts, dtype=torch.int64, device=DEV)
    max_rows = spec.get("max_rows", max(lengths))
    carries = spec.get("carries")
    carries = None if carries is None else torch.tensor(carries, dtype=torch.bool, device=DEV)
    return [
        hc, key, value, norms[0], norms[1], norms[2], conv_weight, conv_states, slots, qsl,
        norms[3], EPS, c, CKPT_CLAMP_MIN, CKPT_DILATION, PAD, max_rows, carries,
    ]  # fmt: skip


def _run(args, conv_input=None):
    """`(hc_out, normed, conv_states)` from the op, on a copy of the pool."""
    args = list(args)
    args[7] = args[7].clone()
    hc_out, normed = op(*args, conv_input=conv_input)
    return hc_out, normed, args[7]


def _assert_staged_conv_input(xc, conv_states, slots, lengths) -> None:
    """With `L <= state_len` rows in every sequence, the published state's last `L`
    columns are those rows' conv inputs: the staged buffer must hold exactly them."""
    rows_per_seq = lengths[0]
    assert set(lengths) == {rows_per_seq} and rows_per_seq <= conv_states.shape[-1]
    tail = conv_states[slots.long()][:, :, -rows_per_seq:]
    assert torch.equal(torch.reshape(torch.transpose(tail, 1, 2), xc.shape), xc), (
        "the staged conv input is not what the conv consumed"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    hc_before = args[0].clone()
    ref = op.reference(*args)
    xc = torch.empty_like(args[0]) if cell.spec.get("conv_input") else None
    out = _run(args, conv_input=xc)
    assert torch.equal(args[0], hc_before), "hc is read-only"
    assert out[0].is_contiguous() and out[1].is_contiguous()
    _compare(out, ref)
    if xc is not None:
        _assert_staged_conv_input(xc, out[2], args[8], cell.spec["lengths"])
    for name, got, want in zip(("hc_out", "normed", "conv_states"), out, ref):
        worst, frac = _stats(got, want)
        print(
            f"\n  {cell.why[:40]:<40} {name:<12} worst {worst:.3g} ulp, differ {frac:.2e}", end=""
        )


def test_conv_input_is_what_the_conv_consumed() -> None:
    """`conv_input` receives the conv input of every row, and changes nothing else: with
    four rows a sequence, the published state's last four columns are those rows."""
    args = _build(dict(lengths=(4,) * 16, pool_slots=461), seed=11)
    plain = _run(args)
    xc = torch.empty_like(args[0])
    args[7] = args[7].clone()
    hc_out, normed = op(*args, conv_input=xc)
    assert torch.equal(hc_out, plain[0]) and torch.equal(normed, plain[1])
    assert torch.equal(args[7], plain[2])
    tail = args[7][args[8].long()][:, :, -4:]
    assert torch.equal(torch.reshape(torch.transpose(tail, 1, 2), xc.shape), xc)


def test_out_writes_the_rows_of_larger_buffers() -> None:
    """`out` takes the leading rows of two padded `[T + 8, C*D]` buffers: the same bits as
    the fresh outputs, the padding rows untouched -- the prefill target's in-place write."""
    args = _build(dict(lengths=(3001, 5000) + (1,) * 64, carries=(False, True) + (True,) * 64,
                       pool_slots=461), seed=12)  # fmt: skip
    plain = _run(args)
    rows = args[0].shape[0]
    bufs = [torch.full((rows + 8, args[0].shape[1]), 7.0, dtype=args[0].dtype, device=DEV)
            for _ in range(2)]  # fmt: skip
    args[7] = args[7].clone()
    hc_out, normed = op(*args, out=(bufs[0][:rows], bufs[1][:rows]))
    assert hc_out.data_ptr() == bufs[0].data_ptr() and normed.data_ptr() == bufs[1].data_ptr()
    assert torch.equal(bufs[0][:rows], plain[0]) and torch.equal(bufs[1][:rows], plain[1])
    assert torch.equal(args[7], plain[2])
    assert bool((bufs[0][rows:] == 7.0).all()) and bool((bufs[1][rows:] == 7.0).all())


def test_guard_refuses_a_short_value() -> None:
    args = _build(dict(lengths=(1, 1)), seed=4)
    args[2] = args[2][:, :-1]
    with validating(op), pytest.raises(AssertionError):
        op(*args)


@pytest.mark.parametrize(
    "spec",
    [
        dict(lengths=(1,) * 8, pool_slots=ENGINE_SLOTS),
        # The tep4 decode graph and the MTP7 verify graph (packed-row conv, staged
        # conv input) over a rank's whole pool.
        dict(lengths=(1,) * 128, pool_slots=TEP4_SLOTS),
        dict(lengths=(8,) * 128, pool_slots=TEP4_SLOTS, conv_input=True),
    ],
    ids=lambda s: f"{len(s['lengths'])}x{s['lengths'][0]}-{s['pool_slots']}slots",
)
def test_cuda_graph_capture_and_replay(spec) -> None:
    args = _build(spec, seed=8)
    xc = torch.empty_like(args[0]) if spec.get("conv_input") else None
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    pool = args[7].clone()
    with torch.cuda.stream(warmup):
        for _ in range(2):
            op(*args, conv_input=xc)
    torch.cuda.current_stream().wait_stream(warmup)
    args[7].copy_(pool)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_hc, graph_normed = op(*args, conv_input=xc)
    fresh = _build(spec, seed=18)
    args[0].copy_(fresh[0])
    args[1].copy_(fresh[1])
    args[2].copy_(fresh[2])
    args[7].copy_(pool)
    eager_xc = None if xc is None else torch.empty_like(xc)
    eager = _run(args, conv_input=eager_xc)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_hc, eager[0])
    assert torch.equal(graph_normed, eager[1])
    assert torch.equal(args[7], eager[2])
    if xc is not None:
        assert torch.equal(xc, eager_xc)


@pytest.mark.parametrize("rows_per_seq", (1, 3, 4))
def test_inject_block_is_the_injection_then_the_op(rows_per_seq) -> None:
    """`inject_block` is bitwise `qwen4_exp_hc_inject_norm` then the op over its `hc_out`, the
    pool too; the gate is a strided column, as the target hands it."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_norm import (  # noqa: E501
        qwen4_exp_hc_inject_norm,
    )

    args = _build(dict(lengths=(rows_per_seq,) * 2), seed=20 + rows_per_seq)
    hc = args[0]
    rows, width = hc.shape
    g = torch.Generator(device=DEV).manual_seed(5)
    routed = (0.5 * torch.randn(rows, CKPT_HIDDEN, device=DEV, generator=g)).to(hc.dtype)
    shared = (0.5 * torch.randn(rows, CKPT_HIDDEN, device=DEV, generator=g)).to(hc.dtype)
    gug = (2.0 * torch.randn(rows, 1288, device=DEV, generator=g)).to(hc.dtype)
    gate = gug[:, 1280:1281]
    inject = (2.0 * torch.sigmoid(torch.randn(rows, CKPT_HC_COUNT, device=DEV, generator=g))).to(
        hc.dtype
    )
    hc_inj, _ = qwen4_exp_hc_inject_norm(
        hc, routed, inject, args[10], EPS, CKPT_HC_COUNT, shared, gate
    )
    want = _run([hc_inj] + list(args[1:]))
    args[7] = args[7].clone()
    hc_out, normed = op(*args, inject_block=(routed, inject, shared, gate))
    assert torch.equal(hc_out, want[0]) and torch.equal(normed, want[1])
    assert torch.equal(args[7], want[2])
    # Without a shared expert (a fresh pool: the call above advanced this one).
    args = _build(dict(lengths=(rows_per_seq,) * 2), seed=20 + rows_per_seq)
    hc_inj, _ = qwen4_exp_hc_inject_norm(args[0], routed, inject, args[10], EPS, CKPT_HC_COUNT)
    want = _run([hc_inj] + list(args[1:]))
    args[7] = args[7].clone()
    hc_out, normed = op(*args, inject_block=(routed, inject, None, None))
    assert torch.equal(hc_out, want[0]) and torch.equal(normed, want[1])
    assert torch.equal(args[7], want[2])


@pytest.mark.parametrize("rows_per_seq", (1, 3))
def test_pdl_launches_are_the_plain_launches(rows_per_seq) -> None:
    """`pdl=True` is bitwise the plain launches, eagerly and replayed from a CUDA graph."""
    args = _build(dict(lengths=(rows_per_seq,) * 4), seed=30 + rows_per_seq)
    want = _run(args)
    args[7] = args[7].clone()
    hc_out, normed = op(*args, pdl=True)
    assert torch.equal(hc_out, want[0]) and torch.equal(normed, want[1])
    assert torch.equal(args[7], want[2])
    fresh = _build(dict(lengths=(rows_per_seq,) * 4), seed=30 + rows_per_seq)
    pool = fresh[7].clone()
    fresh[7] = pool
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out, g_normed = op(*fresh, pdl=True)
    pool.copy_(_build(dict(lengths=(rows_per_seq,) * 4), seed=30 + rows_per_seq)[7])
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(g_out, want[0]) and torch.equal(g_normed, want[1])
    assert torch.equal(pool, want[2])


def test_rows_per_sequence_variants_compile_at_the_first_call() -> None:
    """Every `MAX_P` variant and the rows kernel compile at a geometry's first call: later row
    counts add no variant, and outputs match the reference."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ple import (
        ple_gate_conv_inject_norm as mod,
    )

    def variants():
        dev = torch.cuda.current_device()
        return [
            len(k.device_caches[dev][0]) for k in (mod._ple_conv_kernel, mod._ple_conv_rows_kernel)
        ]

    for carries in (None, True):
        first = _build(dict(lengths=[3], carries=None if carries is None else [True]), seed=1)
        _run(first)
        warmed = variants()
        for lengths in ([1], [2], [4], [1] * 5, [2, 4], [17], [40, 3]):
            spec = dict(lengths=lengths, carries=None if carries is None else [True] * len(lengths))
            args = _build(spec, seed=len(lengths))
            out = _run(args)
            _compare(out, op.reference(*args))
        assert variants() == warmed, (carries, warmed, variants())
