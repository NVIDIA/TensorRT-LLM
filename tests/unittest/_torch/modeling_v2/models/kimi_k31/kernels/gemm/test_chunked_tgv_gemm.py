# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 chunked TGV GEMM kernel (K summed in given chunks: the bits of a split-K GEMM).

Every cell runs against the reference (FP32 chunk products summed in order) within one rounding; the chunk maps a
caller holds as plan data are certified bit for bit against the GEMM they replace (cuBLAS through the op the target
runs there) on the bit battery (``_bit_battery``), which includes exact-cancellation inputs whose outputs are the
summation order's own rounding.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm.chunked_tgv_gemm import (
    kind_tiles,
    run_chunked_tgv,
    run_chunked_tgv_split,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.fused_a_gemm import (
    k31_fused_a_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.dsv3_router_gemm_op import (
    dsv3_router_gemm_op,
)

__extra_import_path__ = ["../../../..", "."]
from _bit_battery import bit_battery  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "the chunked TGV GEMM requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 3), reason="certified on GB300 (sm_103)"
)

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
#: K tiles are this wide; a chunk bound is a K-tile index.
K_TILE = 128


def chunked_tgv(x, kinds, cta_n=16, stages=12, pdl=True):
    """For each kind ``(weight [N, K] BF16, bounds, out_dtype)``: the kernel's ``[M, N]`` output, a column view of an
    output padded to whole 128-row weight tiles."""
    outs = [
        torch.empty(x.shape[0], kind_tiles(w.shape[0]) * 64, dtype=dtype, device=x.device)
        for w, _, dtype in kinds
    ]
    run_chunked_tgv(
        x,
        [(w, out, tuple(bounds)) for (w, bounds, _), out in zip(kinds, outs)],
        cta_n,
        stages,
        pdl,
    )
    return [out[:, : w.shape[0]] for (w, _, _), out in zip(kinds, outs)]


def reference(x, kinds):
    """FP32 products of each chunk (true FP32 matmul), added in chunk order, rounded once."""
    out = []
    with true_fp32_matmul():
        for w, bounds, dtype in kinds:
            edges = [0, *[b * K_TILE for b in bounds], w.shape[1]]
            acc = None
            for lo, hi in zip(edges[:-1], edges[1:]):
                part = x[:, lo:hi].float() @ w[:, lo:hi].float().t()
                acc = part if acc is None else acc + part
            out.append(acc.to(dtype))
    return out


# (why, rows, kinds, cta_n, stages); kinds: ((weight rows, K, chunk bounds, out dtype name), ...).
CELLS = (
    ("one kind, no bound: whole K in order", 8, ((512, 7168, (), "bfloat16"),), 16, 12),
    ("one kind, two equal halves", 40, ((512, 7168, (28,), "bfloat16"),), 16, 12),
    ("one kind, FP32 out, three chunks", 40, ((128, 7168, (19, 38), "float32"),), 16, 12),
    (
        "seven chunks at 32 columns (the TMEM bound)",
        4,
        ((256, 7168, (8, 16, 24, 32, 40, 48), "bfloat16"),),
        32,
        10,
    ),
    (
        "K 12288 (96 tiles), four chunks, 80-token tile",
        72,
        ((256, 12288, (24, 48, 72), "bfloat16"),),
        64,
        8,
    ),
    ("two chunks at an 80-token tile", 80, ((256, 12288, (48,), "bfloat16"),), 80, 8),
    (
        "two kinds: rows past each weight padded, different chunk maps and dtypes",
        35,
        ((224, 7168, (28,), "bfloat16"), (56, 7168, (19, 38), "float32")),
        16,
        12,
    ),
)


def _cell_inputs(rows, cell_kinds, seed):
    """``x`` and, per kind, a row view (offset by 64 rows) of a larger K-contiguous weight."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = cell_kinds[0][1]
    x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
    kinds = []
    for n, kk, bounds, dtype in cell_kinds:
        full = (torch.randn(n + 128, kk, generator=g, device="cuda") / kk**0.5).to(torch.bfloat16)
        kinds.append((full[64 : 64 + n], tuple(bounds), DTYPES[dtype]))
    return x, kinds


@pytest.mark.parametrize("why,rows,cell_kinds,cta_n,stages", CELLS, ids=[c[0][:48] for c in CELLS])
def test_certified_cells(why, rows, cell_kinds, cta_n, stages) -> None:
    x, kinds = _cell_inputs(rows, cell_kinds, zlib.crc32(why.encode()) % 2**31)
    outs = chunked_tgv(x, kinds, cta_n, stages)
    refs = reference(x, kinds)
    for out, ref, (w, _, dtype) in zip(outs, refs, kinds):
        assert out.shape == (x.shape[0], w.shape[0]) and out.dtype == dtype
        # The same chunks, each accumulated in another order: one BF16 rounding apart; FP32 outputs keep the
        # order's own error, a random walk over the chunk's ~2.4k products (measured on GB300: relative RMS
        # 2.5e-6 = 21 FP32 ulps of the row scale), so the FP32 band is set at about twice that.
        if dtype == torch.bfloat16:
            assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)
        else:
            assert_within_ulp(out, ref, element_ulp=512.0, rms_ulp=48.0)
    again = chunked_tgv(x, kinds, cta_n, stages)
    assert all(torch.equal(a, b) for a, b in zip(outs, again)), "not deterministic"


# Chunk maps certified against the GEMM they replace: (name, N, K, rows, bounds, out dtype, cta_n, stages, reference).
# The replicated Kimi K3.1 MoE head (router: dsv3_router_gemm_op, cuBLAS split-K with FP32 output; latent down:
# k31_fused_a_gemm, cuBLAS) and the DEP16 KDA / MLA o_proj (F.linear): cuBLAS's split at these rows on GB300.
def _router(x, w):
    return dsv3_router_gemm_op(x, w.t(), None, torch.float32)


def _latent(x, w):
    return k31_fused_a_gemm(x, w.t())


CERTIFIED = (
    *(("router", 896, 7168, m, (19, 38), "float32", 16, 12, _router) for m in (33, 40, 48, 49, 64)),
    *(("router", 896, 7168, m, (28,), "float32", 64, 9, _router) for m in (65, 96)),
    *(("latent", 3584, 7168, m, (28,), "bfloat16", 16, 12, _latent) for m in (33, 40, 48)),
    *(("latent", 3584, 7168, m, (), "bfloat16", 16, 12, _latent) for m in (49, 56, 64)),
    *(("latent", 3584, 7168, m, (28,), "bfloat16", 64, 9, _latent) for m in (65, 96)),
    *(
        ("o_proj", 7168, 12288, m, (24, 48, 72), "bfloat16", cn, st, F.linear)
        for m, cn, st in ((8, 16, 10), (16, 16, 10), (32, 32, 10), (40, 64, 8), (64, 64, 8))
    ),
    *(("o_proj", 7168, 12288, m, (48,), "bfloat16", 80, 8, F.linear) for m in (72, 80)),
)
BATTERY = 20


@pytest.mark.parametrize(
    "name,n,k,m,bounds,dtype,cta_n,stages,ref_fn",
    CERTIFIED,
    ids=[f"{c[0]}-m{c[3]}-{len(c[4]) + 1}chunks" for c in CERTIFIED],
)
def test_certified_maps_are_bitwise(name, n, k, m, bounds, dtype, cta_n, stages, ref_fn) -> None:
    """The map's output equals the replaced GEMM's bit for bit on the whole battery, for the full weight and for
    TEP16 rank slices of it (the sharded MoE head's views)."""
    dt = DTYPES[dtype]
    bits = torch.int32 if dt == torch.float32 else torch.int16
    for i, (x, w) in enumerate(bit_battery(m, n, k, seed=m * 131 + n, count=BATTERY)):
        ref = ref_fn(x, w)
        views = [(0, n)] + (
            [(r * n // 16, (r + 1) * n // 16) for r in (0, 7, 15)] if name != "o_proj" else []
        )
        for lo, hi in views:
            (out,) = chunked_tgv(x, [(w[lo:hi], bounds, dt)], cta_n, stages)
            diff = int((out.view(bits) != ref[:, lo:hi].view(bits)).sum())
            assert diff == 0, (
                f"{name} m={m} rows {lo}-{hi} battery input {i}: {diff} outputs differ"
            )


# The DEP16 shared-expert gate_up on 128-row weight tiles (the low-M plan's tgvchunkm rows): cuBLAS splits K in 7 even
# chunks at 1-8 rows and in 3 front-heavy ones at 9-32 (F.linear on GB300).
M128 = tuple(
    (m, (8, 16, 24, 32, 40, 48) if m <= 8 else (19, 38), 16 if m <= 16 else 32)
    for m in (1, 5, 8, 9, 16, 17, 25, 32)
)


@pytest.mark.parametrize("m,bounds,cta_n", M128, ids=[f"gate_up-m{c[0]}" for c in M128])
def test_128_row_tiles_are_bitwise(m, bounds, cta_n) -> None:
    """On 128-row weight tiles the chunked TGV with cuBLAS's chunk map gives cuBLAS's bits on the whole battery."""
    n, k = 12288, 7168
    for i, (x, w) in enumerate(bit_battery(m, n, k, seed=m * 131 + n, count=BATTERY)):
        out = torch.empty(m, kind_tiles(n, 128) * 128, dtype=torch.bfloat16, device="cuda")
        run_chunked_tgv(x, [(w, out, bounds)], cta_n, 6, True, 128)
        diff = int((out[:, :n].view(torch.int16) != F.linear(x, w).view(torch.int16)).sum())
        assert diff == 0, f"m={m} battery input {i}: {diff} outputs differ"


def test_two_kinds_in_one_launch() -> None:
    """Latent and router rank slices in one launch equal the two kinds run alone and the replaced GEMMs."""
    m, rank = 40, 5
    for x, w in bit_battery(m, 896 + 3584, 7168, seed=3, count=BATTERY):
        wl, wr = w[896 + rank * 224 : 896 + (rank + 1) * 224], w[rank * 56 : (rank + 1) * 56]
        lat, rt = chunked_tgv(
            x, [(wl, (28,), torch.bfloat16), (wr, (19, 38), torch.float32)], 16, 12
        )
        assert torch.equal(
            lat.view(torch.int16),
            _latent(x, w[896:])[:, rank * 224 : (rank + 1) * 224].view(torch.int16),
        )
        assert torch.equal(
            rt.view(torch.int32),
            _router(x, w[:896])[:, rank * 56 : (rank + 1) * 56].view(torch.int32),
        )


SPLIT_ROWS = (33, 40, 48, 49, 64)


@pytest.mark.parametrize("m", SPLIT_ROWS)
def test_split_planes_equal_multi_accumulator(m) -> None:
    """One CTA per K chunk: the planes added in order in FP32 (and rounded once) are the multi-accumulator launch's
    bits, for the MoE head's TP16 / TP8 rank slices (router FP32 in 3 chunks, latent BF16 in 2 or 1)."""
    latent_bounds = (28,) if m <= 48 else ()
    for x, w in bit_battery(m, 896 + 3584, 7168, seed=m * 29, count=12):
        for world, rank in ((16, 9), (8, 3)):
            rs, ls = 896 // world, 3584 // world
            wl, wr = w[896 + rank * ls : 896 + (rank + 1) * ls], w[rank * rs : (rank + 1) * rs]
            lat, rt = chunked_tgv(
                x, [(wl, latent_bounds, torch.bfloat16), (wr, (19, 38), torch.float32)], 16, 12
            )
            pl = torch.empty(
                1 + len(latent_bounds),
                m,
                kind_tiles(ls) * 64,
                dtype=torch.float32 if latent_bounds else torch.bfloat16,
                device="cuda",
            )
            pr = torch.empty(3, m, kind_tiles(rs) * 64, dtype=torch.float32, device="cuda")
            run_chunked_tgv_split(x, [(wl, pl, latent_bounds), (wr, pr, (19, 38))], 16, 12, True)
            rsum = (pr[0] + pr[1]) + pr[2]
            lsum = (pl[0] + pl[1]).to(torch.bfloat16) if latent_bounds else pl[0]
            assert torch.equal(rsum[:, :rs].view(torch.int32), rt.view(torch.int32))
            assert torch.equal(lsum[:, :ls].view(torch.int16), lat.view(torch.int16))


def test_graph_capture_after_compile() -> None:
    """Compiled before capture, the launch replays under a CUDA graph with the same bits."""
    x = torch.randn(48, 7168, device="cuda").to(torch.bfloat16)
    w = (torch.randn(256, 7168, device="cuda") / 7168**0.5).to(torch.bfloat16)
    out = torch.empty(48, kind_tiles(256) * 64, dtype=torch.bfloat16, device="cuda")
    kinds = [(w, out, (28,))]
    run_chunked_tgv(x, kinds, 16, 12, True)
    eager = out.clone()
    out.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_chunked_tgv(x, kinds, 16, 12, True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
