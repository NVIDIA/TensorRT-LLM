# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gemm/m3_mx_gemv entry: the decode MXFP8 GEMV on a producer-quantized activation.

Operands as the target holds them: the weight through mxfp8_quantize with the swizzled scale layout (the MXFP8 linear
method's storage), the activation through mxfp8_quantize with the swizzled layout (what the boundary all-reduce and
the attention merge publish). The tp2 target's per-rank projections: sparse QKV + index 4992x6144, dense QKV
4608x6144, o_proj 6144x4096, shared gate_up / down 3072x6144 / 6144x1536, dense gate_up / down 12288x6144 / 6144x6144,
and the DSpark drafter's QKV 5120x6144 (MXDRAFT). Besides the dense operands, every TP2 projection runs on operands
whose UE8M0 block scales span 2^-12..2^+12 (the block-scale stress: dense, narrow-scale data passes a split / config
claim falsely), for the in-band, deterministic and graph-replay claims.
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.mxfp8_quantize
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.gemm.m3_mx_gemv import (  # noqa: E501
    m3_mx_gemv as op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_mxfp8_operands import mxfp8  # noqa: E402
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)

assert torch.cuda.is_available(), "m3_mx_gemv requires a CUDA device"


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
    leaves the margin between that and a full extra ulp.
    """
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=0.1)


def build(n: int, k: int, m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True)
    return x, xq.view(m, k), xsf, wq.view(n, k), wsf


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    (n, k), m = cell.spec["nk"], cell.spec["m"]
    _, xq, xsf, w, wsf = build(n, k, m, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(xq, xsf, w, wsf, use_pdl=True)  # the target launches with PDL
    assert out is not None, "a certified shape must be served"
    assert out.shape == (m, n) and out.dtype is torch.bfloat16
    compare(out, op.reference(xq, xsf, w, wsf))


def test_an_unserved_shape_returns_none() -> None:
    """Shapes outside the tuned table fall back at the call site; the op says so rather than computing."""
    _, xq, xsf, w, wsf = build(256, 512, 1, seed=1)
    assert op(xq, xsf, w, wsf) is None


def test_guard_refuses_linear_activation_scales() -> None:
    """A linear [M, K / 32] scale buffer is read as the swizzled one."""
    x, _, _, w, wsf = build(6144, 4096, 4, seed=2)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, False)
    with pytest.raises(AssertionError, match="swizzled"):
        with validating(op):
            op(xq.view(4, 4096), xsf, w, wsf)


def test_guard_refuses_a_strided_weight() -> None:
    _, xq, xsf, w, wsf = build(6144, 4096, 1, seed=3)
    wide = torch.zeros(6144, 8192, dtype=w.dtype, device="cuda")
    wide[:, :4096] = w
    with pytest.raises(AssertionError, match="contiguous e4m3"):
        with validating(op):
            op(xq, xsf, wide[:, :4096], wsf)


@pytest.mark.parametrize("nk", [(4992, 6144), (4608, 6144)], ids=["qkv_sparse", "qkv_dense"])
@pytest.mark.parametrize("m", range(1, 9))
def test_qkv_split8_config(nk, m) -> None:
    """The dispatch's (split, ring) override (the TP4 measurement, kept at the TP2 widths) for the QKV at decode rows:
    within the band of the fp64 reference, and the graph replay bit-identical to the eager call (one split-K order per
    config)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
        modeling as dispatch,
    )

    cfg = dispatch.mx_gemv_cfg(nk[0], nk[1], m)
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


def test_qkv_split8_config_is_bounded(monkeypatch) -> None:
    """The overrides cover only the two QKV projections (split 8) and the o_proj (strided k-tiles, rule 1b) at decode
    rows; every other call, and every call with its kill knob off, keeps the shared table."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
        modeling as dispatch,
    )

    assert dispatch.mx_gemv_cfg(4992, 6144, 9) is None
    assert dispatch.mx_gemv_cfg(6144, 4096, 4) == (2, 8, 0)
    assert dispatch.mx_gemv_cfg(6144, 4096, 9) is None
    assert dispatch.mx_gemv_cfg(6144, 6144, 4) is None
    assert dispatch.mx_gemv_cfg(5120, 6144, 4) is None  # the drafter's QKV keeps the shared table
    monkeypatch.setattr(dispatch, "GEMV_OPROJ_STRIDED_TP4", False)
    assert dispatch.mx_gemv_cfg(6144, 4096, 4) is None
    assert dispatch.mx_gemv_cfg(4992, 6144, 4) == (8, 8)
    monkeypatch.setattr(dispatch, "GEMV_QKV_SPLIT8_TP4", False)
    assert dispatch.mx_gemv_cfg(4992, 6144, 4) is None


@pytest.mark.parametrize(
    "nk",
    [(4992, 6144), (4608, 6144), (6144, 4096)],
    ids=["qkv_sparse", "qkv_dense", "o_proj_strided"],
)
def test_prebuild_covers_the_split8_config_while_frozen(nk) -> None:
    """The core's prebuild (``prebuild_kernels``: ``_gemv_variants`` + ``_precompile_cfg``) compiles every (rows, cfg)
    ``_linear`` can pick for this projection, the (split, ring) override included: with compiles frozen afterwards
    (as the target freezes them before warmup), every decode-row MXFP8 call at the override is served, none
    declines."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
        modeling as dispatch,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels.mx_gemv import (  # noqa: E501
        op as mx_gemv,
    )

    n, k = nk
    _, _, _, wq, wsf = build(n, k, 1, zlib.crc32(f"prebuild-{nk}".encode()))
    mx_gemv.freeze(False)
    mx_gemv.ensure_compiled(wq, wsf, True)
    for rows, x_bf16, ar in dispatch._gemv_variants(n, k, None, 2):
        cfg = None if x_bf16 else dispatch.mx_gemv_cfg(n, k, rows)
        if cfg is not None:
            dispatch._precompile_cfg(wq, wsf, rows, cfg, True, ar=ar)
        else:
            mx_gemv.precompile(wq, wsf, rows, x_bf16, use_pdl=True, ar=ar)
    picked = [
        (rows, dispatch.mx_gemv_cfg(n, k, rows))
        for rows in sorted(dispatch.MX_GEMV_ROWS[nk])
        if dispatch.mx_gemv_cfg(n, k, rows) is not None
    ]
    assert picked, "the override covers decode rows of this projection"
    for rows, cfg in picked:
        resolved = mx_gemv._resolve(rows, n, k, False, True, cfg, True, None)
        assert resolved is not None and mx_gemv.is_compiled(resolved[0]), (rows, cfg)
    mx_gemv.freeze(True)
    try:
        for rows, cfg in picked:
            _, xq, xsf, _, _ = build(n, k, rows, rows)
            out = mx_gemv.m3_mx_gemv(xq, xsf, wq, wsf, cfg=cfg, use_pdl=True)
            assert out is not None, (rows, cfg, mx_gemv.last_uncompiled)
    finally:
        mx_gemv.freeze(False)


# Every TP2 projection (and the drafter's QKV) at the rows each build differs in: the MMA widths 8 / 16 / 32 / 48, the
# decode override rows (<= 8) and the c1..c4 K7 steps (8 / 16 / 24 / 32).
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
STRESS_ROWS = (1, 7, 8, 16, 24, 32, 48)
STRESS_CASES = [(name, m) for name in TP2_SHAPES for m in STRESS_ROWS]
# UE8M0 codes 115..139: block scales 2^-12 .. 2^+12 on both operands.
STRESS_LO, STRESS_HI = 115, 140


def test_tp2_projections_are_served() -> None:
    """The TP2 table serves every TP2 projection at one token (an unserved one would decline to the GEMM silently)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels.mx_gemv import (  # noqa: E501
        op as mx_gemv,
    )

    for name, (n, k) in TP2_SHAPES.items():
        assert mx_gemv.serves(n, k, 1, False), name


@pytest.mark.parametrize("name, m", STRESS_CASES, ids=[f"{n}-m{m}" for n, m in STRESS_CASES])
def test_tp2_block_scale_stress(name, m) -> None:
    """Block scales 2^-12..2^+12 on the weight and the activation, at every config the dispatch can pick (the table's,
    and the decode override where it applies): within the fp64 reference's band, the same bits on 3 eager calls and on
    a CUDA-graph replay. Rows the GEMV does not serve for a projection (the shared down at 17..32) skip."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
        modeling as dispatch,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels.mx_gemv import (  # noqa: E501
        op as mx_gemv,
    )

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


def test_the_mxfp8_row_table_is_served_at_tp2() -> None:
    """Every (projection, rows) ``MX_GEMV_ROWS`` routes to the MXFP8-in GEMV is one the GEMV serves at the TP2 widths
    (a declined row is a ``missed`` fast path)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
        modeling as dispatch,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels.mx_gemv import (  # noqa: E501
        op as mx_gemv,
    )

    declined = [
        (nk, m)
        for nk, rows in sorted(dispatch.MX_GEMV_ROWS.items())
        for m in sorted(rows)
        if not mx_gemv.serves(nk[0], nk[1], m, False)
    ]
    assert not declined, f"MXFP8-in rows the dispatch routes but the GEMV declines: {declined}"
