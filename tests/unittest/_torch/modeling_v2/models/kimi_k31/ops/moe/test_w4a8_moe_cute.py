# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 CuTe DSL W4A8 routed-expert entry: every cell bitwise against the
TRTLLM-Gen runner and within the FP32 band of the torch reference, the pre-routed (front / route tail) sort and the
unfinalized output, re-armed workspace counters and CUDA-graph replay."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe.moe_finalize import (  # noqa: E501
    run_moe_finalize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe.w4a8_moe_gemm import (  # noqa: E501
    num_sms,
    workspace_layout,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.w4a8_moe_cute import (  # noqa: E501
    k31_w4a8_moe_cute,
    max_tiles_for,
    moe_workspace_bytes,
    pick_n_tile,
    pick_tiling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.w4a8_moe_runner import (  # noqa: E501
    HIDDEN,
    NUM_EXPERTS,
    TOP_K,
    k31_w4a8_moe_runner,
)
from tensorrt_llm._utils import is_sm_100f

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

op = k31_w4a8_moe_cute

if not torch.cuda.is_available() or not is_sm_100f():
    pytest.skip("the W4A8 CuTe DSL MoE needs an SM10x GPU", allow_module_level=True)

# The rows the experts see per rank at the targets that call them: the TEP target at MoE TP4 x EP8 (TEP32: 8 rows) and
# MoE TP4 x EP4 (TEP16: 6 / 14 / 24 / 72 rows, unfinalized above the MoE front's rows), the DEP16 target at EP16
# (96 / 168 / 224 rows per rank x 16 ranks).
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "kimi_k31/kimi_k31_pro_mxfp4__sm_103__tep": (
        Cell(
            why="TEP32 rows 8, tile 8",
            spec=dict(rows=8, local=112, inter=768, n_tile=8),
        ),
        *(
            Cell(
                why=f"TEP16 rows {r}, tile 8",
                spec=dict(rows=r, local=224, inter=768, n_tile=8),
            )
            for r in (6, 14, 24)
        ),
        Cell(
            why="TEP16 rows 72, tile 8",
            spec=dict(rows=72, local=224, inter=768, n_tile=8),
        ),
        Cell(
            why="TEP16 rows 72, tile 16 (pick_n_tile)",
            spec=dict(rows=72, local=224, inter=768, n_tile=16),
        ),
        *(
            Cell(
                why=f"TEP16 rows {r}, tile {t}, unfinalized (the fused tail folds the top-k sum)",
                spec=dict(rows=r, local=224, inter=768, n_tile=t, finalize=False),
            )
            for r, t in ((33, 8), (72, 16), (120, 16))
        ),
    ),
    "kimi_k31/kimi_k31_pro_mxfp4__sm_103__dep": (
        *(
            Cell(
                why=f"DEP16 96 rows/rank x 16, tile {t}",
                spec=dict(rows=1536, local=56, inter=3072, n_tile=t),
            )
            for t in (64, 128)
        ),
        *(
            Cell(
                why=f"DEP16 {r // 16} rows/rank x 16 on CTA pairs, tile 128 (pick_tiling)",
                spec=dict(rows=r, local=56, inter=3072, n_tile=128, pair=True),
            )
            for r in (1536, 2688, 3584)
        ),
        Cell(
            why="DEP16 96 rows/rank on CTA pairs, tile 64",
            spec=dict(rows=1536, local=56, inter=3072, n_tile=64, pair=True),
        ),
        Cell(
            why="DEP16 168 rows/rank on CTA pairs, unfinalized",
            spec=dict(rows=2688, local=56, inter=3072, n_tile=128, pair=True, finalize=False),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


def _build(spec, seed):
    """Random prepared operands of one rank (any bytes are a valid MXFP4 / UE8M0 operand; the scale
    exponents stay in 2^-9..2^-4 so the outputs are finite) and random distinct top-16 ids over the
    896 experts, the rank owning local experts [0, local)."""
    rows, local, inter, n_tile = spec["rows"], spec["local"], spec["inter"], spec["n_tile"]
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)

    def nib(*shape):
        return torch.randint(0, 256, shape, generator=g, device=dev, dtype=torch.uint8)

    def sf(*shape):
        return torch.randint(118, 124, shape, generator=g, device=dev, dtype=torch.uint8)

    x = (torch.randn(rows, HIDDEN, generator=g, device=dev) * 2).to(torch.bfloat16)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=32)
    ids = torch.argsort(torch.rand(rows, NUM_EXPERTS, generator=g, device=dev), dim=1)[:, :TOP_K]
    w = torch.rand(rows, TOP_K, generator=g, device=dev) + 0.5
    w = (w / w.sum(1, keepdim=True)).to(torch.bfloat16)
    ws = torch.zeros(moe_workspace_bytes(max_tiles_for(rows, local, n_tile), inter, n_tile), dtype=torch.uint8,
                     device=dev)  # fmt: skip
    return dict(
        hidden_states=xq,
        hidden_states_scale=xsf.view(rows, -1),
        topk_ids=ids.to(torch.int32).contiguous(),
        topk_weights=w.contiguous(),
        gemm1_weights=nib(local, 2 * inter, HIDDEN // 2),
        gemm1_weights_scale=sf(local, 2 * inter, HIDDEN // 32),
        gemm2_weights=nib(local, HIDDEN, inter // 2),
        gemm2_weights_scale=sf(local, HIDDEN, inter // 32),
        local_expert_offset=0,
        local_num_experts=local,
        n_tile=n_tile,
        workspace=ws,
        finalize=bool(spec.get("finalize", True)),
        pair=bool(spec.get("pair", False)),
    )


def _runner(a):
    """The TRTLLM-Gen runner on the same ids, weights and MXFP8 input."""
    local = a["local_num_experts"]
    inter = a["gemm1_weights"].shape[1] // 2
    dev = a["hidden_states"].device
    return k31_w4a8_moe_runner(
        a["hidden_states"], a["hidden_states_scale"], a["topk_ids"], a["topk_weights"], a["gemm1_weights"],
        a["gemm1_weights_scale"], torch.full((local,), 4.0, dtype=torch.float32, device=dev),
        torch.full((local,), 25.0, dtype=torch.float32, device=dev), a["gemm2_weights"], a["gemm2_weights_scale"],
        inter, inter, a["local_expert_offset"], local, None,
    )  # fmt: skip


def _counters_zero(a, n_tile):
    rows, local, inter = (
        a["hidden_states"].shape[0],
        a["local_num_experts"],
        a["gemm1_weights"].shape[1] // 2,
    )
    lay = workspace_layout(max_tiles_for(rows, local, n_tile), inter, HIDDEN, n_tile, num_sms())
    return int(a["workspace"][: lay["off_parts"]].sum()) == 0


def _band(out, ref):
    """FP32-accumulated reference vs the kernel: cosine and the max error relative to the row max."""
    o, r = out.double(), ref.double()
    cos = float((o * r).sum() / (o.norm() * r.norm()).clamp_min(1e-300))
    rel = float(((o - r).abs().amax(-1) / r.abs().amax(-1).clamp_min(1e-30)).max())
    return cos, rel


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    a = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(**a)
    ref = op.reference(**a)
    if a["finalize"]:
        assert out.shape == (cell.spec["rows"], HIDDEN) and out.dtype is torch.bfloat16
        assert torch.isfinite(ref.float()).all()
        # the same arithmetic as the runner (K order, SiTU, requant, finalize order): bitwise
        assert torch.equal(out.view(torch.int16), _runner(a).view(torch.int16)), (
            "not bitwise vs the runner"
        )
        cos, rel = _band(out, ref)
    else:
        y, w, e2p = out
        yr, _, e2p_r = ref
        # the C++ sort orders an expert's rows by atomics: compare the rows of each routed slot
        assert torch.equal(e2p >= 0, e2p_r >= 0)
        sel = e2p >= 0
        cos, rel = _band(y[e2p[sel].long()], yr[e2p_r[sel].long()])
        fin = torch.empty(cell.spec["rows"], HIDDEN, dtype=torch.bfloat16, device=y.device)
        run_moe_finalize(y, e2p, w, fin)
        assert torch.equal(fin.view(torch.int16), _runner(a).view(torch.int16)), (
            "finalized rows not bitwise"
        )
    # MXFP8 requant of an FP32-accumulated intermediate: a flipped E4M3 rounding moves one output by
    # about an ulp of its block's scale
    assert cos > 0.99999 and rel < 0.02, (cos, rel)
    assert _counters_zero(a, cell.spec["n_tile"])


def _torch_sort(ids, local, n_tile):
    """The runner sort with tokens in p2t (the front / route tail layout)."""
    rows = ids.shape[0]
    flat = ids.reshape(-1).long()
    cnt = torch.bincount(flat[flat < local], minlength=local).cpu()
    tiles = (cnt + n_tile - 1) // n_tile
    start = torch.cumsum(tiles, 0) - tiles
    max_tiles = max_tiles_for(rows, local, n_tile)
    t2e = torch.zeros(max_tiles, dtype=torch.int32)
    t2l = torch.zeros(max_tiles, dtype=torch.int32)
    p2t = torch.zeros(max_tiles * n_tile, dtype=torch.int32)
    e2p = torch.full((rows * TOP_K,), -1, dtype=torch.int32)
    seen = torch.zeros(local, dtype=torch.long)
    for e in range(local):
        for t in range(int(tiles[e])):
            t2e[start[e] + t] = e
            t2l[start[e] + t] = min(
                int(start[e] + t + 1) * n_tile, int(start[e]) * n_tile + int(cnt[e])
            )
    for x, e in enumerate(flat.cpu().tolist()):
        if e < local:
            p = int(start[e]) * n_tile + int(seen[e])
            seen[e] += 1
            p2t[p] = x // TOP_K
            e2p[x] = p
    dev = ids.device
    nt = int(tiles.sum())
    return [e2p.to(dev), p2t.to(dev), torch.tensor([nt * n_tile], dtype=torch.int32, device=dev), t2e.to(dev),
            t2l.to(dev), torch.tensor([nt], dtype=torch.int32, device=dev)]  # fmt: skip


def test_prerouted_and_unfinalized() -> None:
    spec = dict(rows=8, local=112, inter=768, n_tile=8)
    a = _build(spec, seed=11)
    base = op(**a)
    pre = _torch_sort(a["topk_ids"], spec["local"], spec["n_tile"])
    b = dict(a, prerouted=pre)
    assert torch.equal(op(**b), base)
    y, w, e2p = op(**dict(b, finalize=False))
    assert torch.equal(e2p, pre[0])
    fin = torch.empty_like(base)
    run_moe_finalize(y, e2p, w, fin)
    assert torch.equal(fin, base)
    assert _counters_zero(a, spec["n_tile"])


def test_output_buffer_and_graph_replay() -> None:
    spec = dict(rows=24, local=224, inter=768, n_tile=8)
    a = _build(spec, seed=5)
    two = op(**a)
    out = torch.empty_like(two)
    fa = dict(a, output=out)
    op(**fa)
    assert torch.equal(out, two)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op(**fa)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        op(**fa)
    for _ in range(3):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, two)
    assert _counters_zero(a, spec["n_tile"])


@pytest.mark.parametrize("rows,local", [(8, 112), (14, 224), (24, 224)])
def test_early_release_is_bitwise(rows, local) -> None:
    """FC1 releasing FC2 right after its grid dependency (decode rows) gives the bits of the release at
    its end, eagerly and in a CUDA graph of three back-to-back calls."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
        w4a8_moe_cute as _mc,
    )

    spec = dict(rows=rows, local=local, inter=768, n_tile=8)
    a = _build(spec, seed=rows + local)
    saved = _mc.EARLY_RELEASE_MAX_ROWS
    outs = {}
    try:
        for cap in (0, 32):
            _mc.EARLY_RELEASE_MAX_ROWS = cap
            eager = op(**a)
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                op(**a)
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                chain = [op(**a) for _ in range(3)]
            g.replay()
            torch.cuda.synchronize()
            outs[cap] = [eager.clone()] + [t.clone() for t in chain]
    finally:
        _mc.EARLY_RELEASE_MAX_ROWS = saved
    for u, v in zip(outs[0], outs[32]):
        assert torch.equal(u.view(torch.int16), v.view(torch.int16))
    assert _counters_zero(a, spec["n_tile"])


@pytest.mark.parametrize("rows,local", [(33, 112), (48, 112), (64, 224)])
def test_sort_early_release_is_bitwise(rows, local) -> None:
    """The self-sorting call's runner sort releasing the 1-CTA kernel right after its own grid wait gives the
    bits of the release after its stores (above the decode rows, unfinalized like the fused tail's calls),
    eagerly and in a CUDA graph of three back-to-back calls."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
        w4a8_moe_cute as _mc,
    )

    spec = dict(rows=rows, local=local, inter=768, n_tile=pick_n_tile(rows), finalize=False)
    a = _build(spec, seed=rows * 7 + local)
    saved = _mc._SORT_EARLY_RELEASE
    outs = {}

    def rows_of(out):
        y, _, e2p = out
        sel = e2p >= 0
        return [e2p.clone(), y[e2p[sel].long()].clone()]

    try:
        for early in (False, True):
            _mc._SORT_EARLY_RELEASE = early
            eager = rows_of(op(**a))
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                op(**a)
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                chain = [op(**a) for _ in range(3)]
            g.replay()
            torch.cuda.synchronize()
            outs[early] = eager + [t for c in chain for t in rows_of(c)]
    finally:
        _mc._SORT_EARLY_RELEASE = saved
    for u, v in zip(outs[False], outs[True]):
        assert torch.equal(u.view(torch.int16), v.view(torch.int16))
    assert _counters_zero(a, spec["n_tile"])


@pytest.mark.parametrize("pair", [False, True])
def test_runner_weights_untouched(pair) -> None:
    """The op reads the runner's prepared weight and scale tensors in place (no repack, no copy): their
    bytes are unchanged after the op ran (and after the runner on the same tensors)."""
    spec = dict(rows=1536 if pair else 24, local=56 if pair else 224, inter=3072 if pair else 768,
                n_tile=128 if pair else 8, pair=pair)  # fmt: skip
    a = _build(spec, seed=21)
    keys = ("gemm1_weights", "gemm1_weights_scale", "gemm2_weights", "gemm2_weights_scale")
    before = {k: a[k].clone() for k in keys}
    ptrs = {k: a[k].data_ptr() for k in keys}
    out = op(**a)
    assert torch.equal(out.view(torch.int16), _runner(a).view(torch.int16))
    torch.cuda.synchronize()
    for k in keys:
        assert a[k].data_ptr() == ptrs[k] and torch.equal(a[k], before[k]), k


@pytest.mark.parametrize("pair", [False, True])
def test_zero_rows(pair) -> None:
    spec = dict(rows=1, local=56, inter=3072, n_tile=128 if pair else 8, pair=pair)
    a = _build(spec, seed=4)
    for k in ("hidden_states", "hidden_states_scale", "topk_ids", "topk_weights"):
        a[k] = a[k][:0]
    out = op(**a)
    assert out.shape == (0, HIDDEN)
    y, w, e2p = op(**dict(a, finalize=False))
    assert y.shape == (0, HIDDEN) and e2p.numel() == 0


def test_pick_n_tile() -> None:
    assert [pick_n_tile(r) for r in (1, 8, 24, 47, 48, 72, 511, 512, 1536, 1537, 3584)] == [
        8,
        8,
        8,
        8,
        16,
        16,
        16,
        64,
        64,
        128,
        128,
    ]


def test_pick_tiling() -> None:
    assert [pick_tiling(r, 768) for r in (8, 72, 511, 512, 1536, 65536)] == [
        (8, False),
        (16, False),
        (16, False),
        (128, True),
        (128, True),
        (128, True),
    ]
    assert [pick_tiling(r, 3072) for r in (8, 384, 1536, 65536)] == [(128, True)] * 4


def test_pair_graph_replay() -> None:
    spec = dict(rows=1536, local=56, inter=3072, n_tile=128, pair=True)
    a = _build(spec, seed=9)
    ref = op(**a)
    out = torch.empty_like(ref)
    pa = dict(a, output=out)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op(**pa)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        op(**pa)
    for _ in range(2):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


def test_is_valid_rejects() -> None:
    a = _build(dict(rows=8, local=112, inter=768, n_tile=8), seed=1)
    with pytest.raises(ValueError, match="n_tile"):
        op.is_valid(**dict(a, n_tile=12))
    with pytest.raises(ValueError, match="workspace"):
        op.is_valid(**dict(a, workspace=a["workspace"][:64]))
    with pytest.raises(ValueError, match="local experts"):
        op.is_valid(**dict(a, local_num_experts=100))
    with pytest.raises(ValueError, match="n_tile"):
        op.is_valid(**dict(a, pair=True))  # tile 8 is not a pair tile
