# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gemm/m3_mx_gemv entry of the MiniMax-M3.1-Flash sm_107 tp2, tp4 and tp8 targets (the decode MXFP8
GEMV on a producer-quantized activation), and for the tp4 and tp8 targets' gemm/m3_mx_gemv_bf16 entry (the same GEMV
quantizing a bf16 activation in its prologue).

Operands as the targets hold them: the weight through mxfp8_quantize with the swizzled scale layout (the MXFP8 linear
method's storage), the activation through mxfp8_quantize with the swizzled layout (what the boundary all-reduce and
the attention merge publish). The bf16 entry's prologue quantization is also held to mxfp8_quantize itself: the GEMV
on a bf16 x must equal, bit for bit, the prequantized GEMV on mxfp8_quantize(x), the claim the targets rely on when
they mix the two paths.

On tp2 every projection (sparse QKV + index 4992x6144, dense QKV 4608x6144, o_proj 6144x4096, shared gate_up / down
3072x6144 / 6144x1536, dense gate_up / down 12288x6144 / 6144x6144, the DSpark drafter's QKV 5120x6144) also runs on
operands whose UE8M0 block scales span 2^-12..2^+12 (the block-scale stress: dense, narrow-scale data passes a split
/ config claim falsely), for the in-band, deterministic and graph-replay claims.
"""

import importlib
import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.mxfp8_quantize
from tensorrt_llm._torch.modeling_v2._native import NoTarget

__extra_import_path__ = ["../../../..", "../.."]
from _m3_mxfp8_operands import mxfp8  # noqa: E402
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

_PACKAGE = "tensorrt_llm._torch.modeling_v2.models.minimax_m3"
TP2 = "minimax_m3_1_flash_nvfp4__sm_107__tp2"
TP4 = "minimax_m3_1_flash_nvfp4__sm_107__tp4"
TP8 = "minimax_m3_1_flash_nvfp4__sm_107__tp8"


def _module(target: str, name: str):
    try:
        return importlib.import_module(f"{_PACKAGE}.{target}.{name}")
    except NoTarget:
        return None


def _entry(target: str, name: str):
    module = _module(target, f"ops.gemm.{name}")
    return None if module is None else getattr(module, name)


OPS = {target: _entry(target, "m3_mx_gemv") for target in (TP2, TP4, TP8)}
BF16_OPS = {target: _entry(target, "m3_mx_gemv_bf16") for target in (TP4, TP8)}


def _cases(targets, values=lambda target: [None], ids=lambda value: ""):
    """(target, value) params over ``targets``; a target this build does not carry is one skipped param."""
    params = []
    for target in targets:
        short = target.rsplit("__", 1)[1]
        if OPS[target] is None:
            params.append(
                pytest.param(target, None, marks=pytest.mark.skip(reason=f"{target} is not built"))
            )
            continue
        params.extend(
            pytest.param(target, v, id=f"{short}-{ids(v)}" if ids(v) else short)
            for v in values(target)
        )
    return params


# Element band, in ULP units of the row scale: see compare.
ELEMENT_ULP = 1.5


def compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Element ELEMENT_ULP of the row scale, relative RMS 0.1 ulp.

    Kernel and reference multiply bit-identical dequantized operands; the kernel sums in fp32 (tensor-core blocks,
    then the split-K partials), the reference in fp64. Over K <= 6144 that order difference is ~1e-6 relative, so the
    two bf16 roundings disagree only where the sum sits within that of a rounding boundary: one bf16 ulp of that
    element, which for an element in its row maximum's binade is up to two of ``assert_within_ulp``'s units of the row
    scale, and a few in ten thousand elements in aggregate. Any fp32 sum shows it: torch's IEEE fp32 matmul of the
    same dequantized operands measured at most 1.22 units against the fp64 reference over every row count 1..48 on
    the o_proj and the QKV (6 seeds each, GB300), the kernel at most 1.25, both at most 0.009 ulp RMS. The band
    leaves the margin between that and a full extra ulp. The kernel is the same on every target (each target's table
    only picks split and ring).
    """
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=0.1)


def build(n: int, k: int, m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True)
    return x, xq.view(m, k), xsf, wq.view(n, k), wsf


@pytest.mark.parametrize("target,cell", _cases(OPS, lambda t: OPS[t].CELLS, lambda c: c.why[:44]))
def test_certified_cells(target, cell) -> None:
    op = OPS[target]
    (n, k), m = cell.spec["nk"], cell.spec["m"]
    _, xq, xsf, w, wsf = build(n, k, m, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(xq, xsf, w, wsf, use_pdl=True)  # the targets launch with PDL
    assert out is not None, "a certified shape must be served"
    assert out.shape == (m, n) and out.dtype is torch.bfloat16
    compare(out, op.reference(xq, xsf, w, wsf))


@pytest.mark.parametrize("target,_", _cases((TP4, TP8)))
def test_an_unserved_shape_returns_none(target, _) -> None:
    """Shapes outside the tuned table fall back at the call site; the op says so rather than computing."""
    op = OPS[target]
    _x, xq, xsf, w, wsf = build(256, 512, 1, seed=1)
    assert op(xq, xsf, w, wsf) is None


@pytest.mark.parametrize("target,_", _cases((TP2, TP8)))
def test_guard_refuses_linear_activation_scales(target, _) -> None:
    """A linear [M, K / 32] scale buffer is read as the swizzled one."""
    op = OPS[target]
    x, _xq, _xsf, w, wsf = build(6144, 4096, 4, seed=2)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, False)
    with pytest.raises(AssertionError, match="swizzled"):
        with validating(op):
            op(xq.view(4, 4096), xsf, w, wsf)


@pytest.mark.parametrize("target,_", _cases((TP4,)))
def test_guard_refuses_a_strided_weight(target, _) -> None:
    op = OPS[target]
    _x, xq, xsf, w, wsf = build(6144, 2048, 1, seed=3)
    wide = torch.zeros(6144, 4096, dtype=w.dtype, device="cuda")
    wide[:, :2048] = w
    with pytest.raises(AssertionError, match="contiguous e4m3"):
        with validating(op):
            op(xq, xsf, wide[:, :2048], wsf)


# Each target's sparse (QKV + index) and dense QKV widths per rank.
_QKV = {TP2: ((4992, 6144), (4608, 6144)), TP4: ((2560, 6144), (2304, 6144))}


@pytest.mark.parametrize("m", [1, 4, 8])
@pytest.mark.parametrize(
    "target,nk",
    _cases(
        _QKV,
        lambda t: _QKV[t],
        lambda nk: "qkv_sparse" if nk in (_QKV[TP2][0], _QKV[TP4][0]) else "qkv_dense",
    ),
)
def test_qkv_split8_config(target, nk, m) -> None:
    """The dispatch's (split, ring) override (the TP4 measurement, kept at the TP2 widths) for the QKV at decode rows:
    within the band of the fp64 reference, and the graph replay bit-identical to the eager call (one split-K order per
    config)."""
    op = OPS[target]
    cfg = _module(target, "modeling").mx_gemv_cfg(nk[0], nk[1], m)
    assert cfg == (8, 8)
    _, xq, xsf, wq, wsf = build(nk[0], nk[1], m, zlib.crc32(f"qkv8-{nk}-{m}".encode()))
    eager = op(xq, xsf, wq, wsf, cfg=cfg)
    assert eager is not None, "the GEMV declined the split-8 config"
    compare(eager, op.reference(xq, xsf, wq, wsf))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(xq, xsf, wq, wsf, cfg=cfg)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))


@pytest.mark.parametrize("m", [1, 4, 8])
@pytest.mark.parametrize("target,_", _cases((TP4,)))
def test_oproj_partial_tmem_a_is_bitwise(target, _, m) -> None:
    """OPROJ_TMEM256: the o_proj with only its first 6 k-tiles' A in TMEM (the rest read from the smem ring) gives
    the same bits as all 8 in TMEM, and as A all in smem; graph replay equal to eager."""
    op = OPS[target]
    mx_gemv = _module(target, "kernels.mx_gemv.op")
    n, k = 6144, 2048
    _x, xq, xsf, wq, wsf = build(n, k, m, zlib.crc32(f"oproj-tmem256-{m}".encode()))
    resolved = mx_gemv._resolve(m, n, k, False, True, (2, 8, 0, 1, 1, 1, 6), True, None)
    assert resolved is not None and resolved[0][18] == 6, (
        "the partial-TMEM config must resolve as such"
    )
    outs = [
        op(xq, xsf, wq, wsf, cfg=cfg) for cfg in ((2, 8, 0), (2, 8, 0, 1, 1, 1, 6), (2, 8, 0, 1, 0))
    ]
    assert all(o is not None for o in outs)
    for o in outs[1:]:
        assert torch.equal(o.view(torch.int16), outs[0].view(torch.int16))
    compare(outs[1], op.reference(xq, xsf, wq, wsf))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(xq, xsf, wq, wsf, cfg=(2, 8, 0, 1, 1, 1, 6))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), outs[0].view(torch.int16))


@pytest.mark.parametrize(
    "target,cell", _cases(BF16_OPS, lambda t: BF16_OPS[t].CELLS, lambda c: c.why[:44])
)
def test_bf16_prologue_certified_cells(target, cell) -> None:
    op, bf16_op = OPS[target], BF16_OPS[target]
    (n, k), m = cell.spec["nk"], cell.spec["m"]
    x, xq, xsf, w, wsf = build(n, k, m, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(bf16_op):
        out = bf16_op(x, w, wsf, use_pdl=True)  # the targets launch with PDL
    assert out is not None, "a certified shape must be served"
    assert out.shape == (m, n) and out.dtype is torch.bfloat16
    # The band of compare: the quantized operands are the reference's exactly (checked below), the sums differ only
    # in order.
    compare(out, bf16_op.reference(x, w, wsf))
    pre = op(xq, xsf, w, wsf, use_pdl=True)
    assert pre is not None and torch.equal(out.view(torch.int16), pre.view(torch.int16)), (
        "the prologue's MXFP8 quantization differs from mxfp8_quantize"
    )


# Every TP2 projection (and the drafter's QKV) at one row count each, together spanning the rows the builds differ in:
# the MMA widths 8 / 16 / 32 / 48, the decode override rows (<= 8) and the c1..c4 K7 steps (8 / 16 / 24 / 32).
TP2_SHAPES = {
    "qkv_sparse": (4992, 6144),
    "qkv_dense": (4608, 6144),
    "o_proj": (6144, 4096),
    "shared_gate_up": (3072, 6144),
    "shared_down": (6144, 1536),
    "dense_gate_up": (12288, 6144),
    "dense_down": (6144, 6144),
    "drafter_qkv": (5120, 6144),
}
STRESS_ROWS = {
    "qkv_sparse": 1,
    "qkv_dense": 8,
    "o_proj": 7,
    "shared_gate_up": 48,
    "shared_down": 16,
    "dense_gate_up": 32,
    "dense_down": 24,
    "drafter_qkv": 16,
}
# UE8M0 codes 115..139: block scales 2^-12 .. 2^+12 on both operands.
STRESS_LO, STRESS_HI = 115, 140


@pytest.mark.parametrize(
    "target,case", _cases((TP2,), lambda t: list(STRESS_ROWS.items()), lambda c: f"{c[0]}-m{c[1]}")
)
def test_tp2_block_scale_stress(target, case) -> None:
    """Block scales 2^-12..2^+12 on the weight and the activation, at every config the dispatch can pick (the table's,
    and the decode override where it applies): within the fp64 reference's band, the same bits on 3 eager calls and on
    a CUDA-graph replay. Rows the GEMV does not serve for a projection (the shared down at 17..32) skip."""
    op = OPS[target]
    dispatch = _module(target, "modeling")
    mx_gemv = _module(target, "kernels.mx_gemv.op")
    name, m = case
    n, k = TP2_SHAPES[name]
    if not mx_gemv.serves(n, k, m, False):
        pytest.skip(f"{name} is not served at {m} rows (the dispatch runs the MXFP8 GEMM)")
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"stress-{name}-{m}".encode()))
    xq, xsf = mxfp8(m, k, g, STRESS_LO, STRESS_HI)
    w, wsf = mxfp8(n, k, g, STRESS_LO, STRESS_HI)
    ref = op.reference(xq, xsf, w, wsf)
    for cfg in dict.fromkeys((None, dispatch.mx_gemv_cfg(n, k, m))):
        outs = [op(xq, xsf, w, wsf, True, cfg, True, None) for _ in range(3)]
        assert all(o is not None for o in outs), (name, m, cfg)
        compare(outs[0], ref)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op(xq, xsf, w, wsf, True, cfg, True, None)
        graph.replay()
        torch.cuda.synchronize()
        for o in outs[1:] + [captured]:
            assert torch.equal(outs[0].view(torch.int16), o.view(torch.int16)), (
                f"{name} m={m} cfg {cfg}: not deterministic (eager / graph)"
            )
