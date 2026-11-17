# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the default target's qwen4_exp_attn_decode_prologue entry: every cell against
the entry's fp32 `reference` on every allocation either writes -- the packed QKV, the index
query, the K/V pool, the index plane, the position cache -- and a stale-read check under
graph replay with PDL.

The rotated values (the packed q and k, the index query, the K pool's k and the compressed
keys) are judged in units of their row's scale: the kernel and the reference evaluate the
same fp32 formulas and differ only in the rms's reduction order, which moves a value across
a bf16 rounding boundary at most once -- one bf16 spacing of an element no larger than the
row's largest, at most 2^-7 of the row scale (2 ulp in `assert_within_ulp`'s 2^-8 units).
Everything else -- v in the packed buffer and the V pool, the raw index keys, the
coordinates and every byte neither side writes -- is compared bitwise, and the K pool's
rows are bitwise the kernel's own packed k converted to the pool's dtype.

The pools are laid out as the engine serves them: K/V the two role planes of a
`[pages, 2, Hkv, tpb, D]` pool, requests on random pages."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.attention.qwen4_exp_attn_decode_prologue__decode import (  # noqa: E501
    qwen4_exp_attn_decode_prologue as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qwen4_exp_attn_decode_prologue requires a CUDA device"

DEV = "cuda"
D, IH, ID, TPB, CR, ROT, WIDTH, SLOTS = 256, 4, 128, 32, 4, 64, 8196, 8
#: Per-rank geometry: (q heads, KV heads); the fused row is
#: `[q | k | v | gate | index q 4 x 128, k 128 | v slot 128]`.
GEOMS = {"tp4": (6, 1), "dep4": (24, 2)}
#: One bf16 spacing of an element no larger than its row's largest, in 2^-8 units of the
#: row scale; the relative RMS bound is measured, see the module docstring.
ELEMENT_ULP, RMS_ULP = 2.0, 0.05


def _geometry(geom):
    hq, hk = GEOMS[geom]
    qkv = (hq + 2 * hk) * D
    iq_col = qkv + hq * D
    return hq, hk, qkv, iq_col, iq_col + (IH + 2) * ID


def _edge_rows(proj, hq, hk) -> None:
    """Zero, -0.0 and outlier blocks in the k head (past the rotary prefix, where the norm
    keeps zeros zero) and in the v head, over the first rows."""
    k0, v0 = hq * D, (hq + hk) * D
    edges = (
        (0, slice(k0 + 128, k0 + 144), 0.0),
        (1, slice(k0 + 160, k0 + 176), -0.0),
        (2, slice(k0 + 200, k0 + 201), 900.0),
        (0, slice(v0 + 16, v0 + 32), 0.0),
        (1, slice(v0 + 48, v0 + 64), torch.tensor([-0.0] * 8 + [0.0] * 8)),
        (3, slice(v0 + 40, v0 + 41), -900.0),
        (2, slice(v0 + 64, v0 + 80), 1e-30),
    )
    for r, cols, value in edges:
        if r < proj.shape[0]:
            proj[r, cols] = value


def _rows(spec):
    """(request, position) per row: one row a request, or `rows_per_request` consecutive
    positions of each request from its first."""
    per = spec.get("rows_per_request", 1)
    return [(r, p + i) for r, p in enumerate(spec["pos"]) for i in range(per)]


def _build(spec, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    hq, hk, qkv_width, iq_col, row_width = _geometry(spec["geom"])
    layout = _rows(spec)
    pos = [p for _, p in layout]
    rows = len(pos)
    last = {}
    for req, p in layout:
        last[req] = max(last.get(req, 0), p)
    slots = max(SLOTS, len(last) + 3)
    pages = sum(p // TPB + 1 for p in last.values()) + 64
    perm = torch.randperm(pages, generator=g, device=DEV).to(torch.int32)
    block_table = torch.full((slots, WIDTH), -1, dtype=torch.int32, device=DEV)
    used = 0
    for req, p in last.items():
        n = p // TPB + 1
        block_table[req, :n] = perm[used : used + n]
        used += n
    request = torch.tensor([req for req, _ in layout], dtype=torch.int32, device=DEV)
    if spec.get("unmapped") is not None:
        request[spec["unmapped"]] = slots + 5
    logical = torch.tensor(pos, dtype=torch.int64, device=DEV)
    ids = torch.stack([logical, logical + 3, logical + 7]).to(torch.int32).contiguous()
    buf = torch.empty(3, 1, rows + 5, dtype=torch.int32, device=DEV)
    buf[:, 0, :rows] = ids
    coords = torch.transpose(buf[:, 0, :rows], 0, 1)
    table_rows = max(pos) + 8
    angles = torch.rand(table_rows, ROT // 2, generator=g, device=DEV) * 6.28
    cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).contiguous()
    proj = (torch.randn(rows, row_width, generator=g, device=DEV) * 2.0).to(torch.bfloat16)
    if spec.get("edge"):
        _edge_rows(proj, hq, hk)
    out = torch.randn(rows, qkv_width + 64, generator=g, device=DEV).to(torch.bfloat16)
    index_scratch = torch.randn(rows, IH + 2, ID, generator=g, device=DEV).to(torch.bfloat16)
    weights = [
        (0.3 * torch.randn(n, generator=g, device=DEV)).to(torch.bfloat16) for n in (D, D, ID, ID)
    ]
    kv_dtype = torch.float8_e4m3fn if spec.get("fp8") else torch.bfloat16
    pool = torch.randn(pages, 2, hk, TPB, D, generator=g, device=DEV).to(kv_dtype)
    index_cache = torch.randn(pages, TPB, 1, ID, generator=g, device=DEV).to(torch.bfloat16)
    position_cache = torch.randint(
        0, table_rows, (pages, TPB, 3), generator=g, device=DEV, dtype=torch.int32
    )
    return dict(
        proj=proj, out=out[:, :qkv_width], index_q=index_scratch[:, :IH], position_ids=ids,
        coords=coords, cos_sin=cos_sin, q_weight=weights[0], k_weight=weights[1],
        index_q_weight=weights[2], index_k_weight=weights[3], k_cache=pool[:, 0],
        v_cache=pool[:, 1], index_cache=index_cache, position_cache=position_cache,
        request_indices=request, logical_positions=logical, block_table=block_table,
        tokens_per_block=TPB, compress_ratio=CR, num_heads_q=hq, num_heads_k=hk, head_dim=D,
        index_heads=IH, index_head_dim=ID, index_q_col=iq_col, rotary_dim=ROT, eps=1e-6,
        mrope_section1=11, mrope_section2=10, rows_per_request=spec.get("rows_per_request", 1),
    )  # fmt: skip


def _bytes(storage, device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8, device=device).set_(storage)


def _clone(args):
    """The arguments over fresh copies of every allocation they view, each view rebuilt at
    its offset and strides; returns them and the copies, keyed by the first argument
    viewing each allocation."""
    out, copies, keys = dict(args), {}, {}
    for name, t in args.items():
        if not isinstance(t, torch.Tensor):
            continue
        key = keys.setdefault(t.untyped_storage().data_ptr(), name)
        if key not in copies:
            copies[key] = _bytes(t.untyped_storage(), t.device).clone()
        view = torch.empty(0, dtype=t.dtype, device=t.device)
        view.set_(copies[key].untyped_storage(), t.storage_offset(), t.shape, t.stride())
        out[name] = view
    return out, copies


def _written_slots(args):
    """(row, page, slot) of every row whose page lookup holds, from the host's view."""
    table = args["block_table"].cpu()
    pages = args["k_cache"].shape[0]
    found = []
    for t, (req, p) in enumerate(
        zip(args["request_indices"].tolist(), args["logical_positions"].tolist())
    ):
        column = p // TPB
        if 0 <= req < table.shape[0] and column < table.shape[1]:
            page = int(table[req, column])
            if 0 <= page < pages:
                found.append((t, page, p % TPB))
    return found


def _check(got, want, where="") -> dict:
    """Judge the rotated values within the band, then neutralize them in `got` and require
    every byte of every allocation bitwise. Returns the measured worst errors."""
    g, w = got[0], want[0]
    hq, hk = g["num_heads_q"], g["num_heads_k"]
    width = (hq + hk) * D
    measured = {}

    def banded(name, a, b):
        a, b = a.float().reshape(-1, a.shape[-1]), b.float().reshape(-1, b.shape[-1])
        scale = b.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
        measured[name] = ((a - b).abs() / scale).max().item() / 2**-8
        assert_within_ulp(a.to(torch.bfloat16), b.to(torch.bfloat16), ELEMENT_ULP, RMS_ULP)

    banded("packed q/k", g["out"][:, :width], w["out"][:, :width])
    banded("index query", g["index_q"], w["index_q"])
    slots = _written_slots(g)
    k_got = g["out"][:, hq * D : width].reshape(-1, hk, D)
    for t, page, slot in slots:
        own = k_got[t].to(g["k_cache"].dtype)
        assert torch.equal(g["k_cache"][page, :, slot].view(torch.uint8), own.view(torch.uint8)), (
            f"{where}row {t}: the K pool's row is not the packed k converted"
        )
        g["k_cache"][page, :, slot] = w["k_cache"][page, :, slot]
    ends = [(t, page, slot) for t, page, slot in slots if (slot + 1) % CR == 0]
    if ends:
        a = torch.stack([g["index_cache"][page, slot, 0] for _, page, slot in ends])
        b = torch.stack([w["index_cache"][page, slot, 0] for _, page, slot in ends])
        banded("compressed keys", a, b)
        for _, page, slot in ends:
            g["index_cache"][page, slot, 0] = w["index_cache"][page, slot, 0]
    g["out"][:, :width] = w["out"][:, :width]
    g["index_q"].copy_(w["index_q"])
    assert got[1].keys() == want[1].keys()
    for key, copy in got[1].items():
        differ = (copy != want[1][key]).nonzero()
        assert differ.numel() == 0, f"{where}{key}: {differ.shape[0]} bytes differ"
    return measured


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    want = _clone(args)
    op.reference(**want[0])
    got = _clone(args)
    op(**got[0])
    torch.cuda.synchronize()
    measured = _check(got, want)
    print(f"{cell.why[:48]}: worst element / row scale in 2^-8 units {measured}")


@pytest.mark.parametrize(
    "spec",
    [
        dict(geom="tp4", pos=(8699, 8700)),
        dict(geom="tp4", pos=(84000,), rows_per_request=4),
        dict(geom="dep4", pos=(8697, 8700), rows_per_request=4, fp8=True),
    ],
    ids=["tp4_decode", "tp4_verify", "dep4_fp8_verify"],
)
def test_reads_after_the_wait_under_graph_replay(spec) -> None:
    """The projection is rewritten inside the graph right before the op; with PDL a read of
    it before the wait would see the previous replay's row (in a verify step, also the raw
    index keys of a group's earlier rows)."""
    args = _build(spec, seed=23)
    src = args["proj"].clone()
    work, work_copies = _clone(args)

    def step():
        work["proj"].copy_(src * 1.0)
        op(**work)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    # The replays' projections from a seeded generator: the global RNG's state depends on the
    # tests run before this one in the process.
    g = torch.Generator(device=DEV).manual_seed(29)
    for trial in range(3):
        src.copy_((torch.randn(src.shape, generator=g, device=DEV) * 2.0).to(src.dtype))
        before = _clone(work)
        graph.replay()
        torch.cuda.synchronize()
        want = _clone(before[0])
        want[0]["proj"].copy_(src)
        op.reference(**want[0])
        _check((work, work_copies), want, where=f"replay {trial}: ")
