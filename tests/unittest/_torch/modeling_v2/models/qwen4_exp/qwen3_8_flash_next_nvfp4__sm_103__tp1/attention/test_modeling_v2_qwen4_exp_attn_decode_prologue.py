# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_attn_decode_prologue entry: every cell bitwise the
composition it replaces (`qwen4_exp_qk_norm_rope`, `qsa_paged_kv_store` or
`qsa_paged_kv_store_nvfp4`, `qsa_paged_index_store`, `qsa_prefill_compress`) on every
allocation either writes -- the projection, the index query, the K/V pool (with its block
scales over NVFP4), the index plane, the position cache; a stale-read check under graph
replay with PDL over both pool kinds.

The NVFP4 cells lay the pools out as the engine serves them: K/V the two role planes of a
`[pages, 2, Hkv, tpb, D/2]` e2m1 pool and its `[pages, 2, Hkv, tpb, D/16]` scale pool,
and the index plane and the position cache two views of one padded page (page stride
12 x 4096 + 192 bf16 elements, the position page at its tail)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qwen4_exp_attn_decode_prologue import (  # noqa: E501
    qwen4_exp_attn_decode_prologue as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_attn_decode_prologue requires a CUDA device"

DEV = "cuda"
HQ, HK, D, IH, ID, IQ_COL, W = 24, 2, 256, 4, 128, 13312, 14080
TPB, CR, ROT, TABLE, SLOTS, WIDTH = 32, 4, 64, 16384, 8, 300
#: The NVFP4 engine's index page: 12 index buffers of tpb x 128, then the position page.
INDEX_PAGE, POSITION_PAD = 12 * TPB * ID + 192, 192
GEOM = dict(
    tokens_per_block=TPB, compress_ratio=CR, num_heads_q=HQ, num_heads_k=HK, head_dim=D,
    index_heads=IH, index_head_dim=ID, index_q_col=IQ_COL, rotary_dim=ROT, eps=1e-6,
    mrope_section1=11, mrope_section2=10,
)  # fmt: skip


def _edge_rows(proj) -> None:
    """Zero, -0.0 and outlier blocks in the k heads (past the rotary prefix, where the
    norm keeps zeros zero) and in the v heads, over the first rows (up to four)."""
    k0, v0 = HQ * D, (HQ + HK) * D
    edges = (
        (0, slice(k0 + 128, k0 + 144), 0.0),
        (1, slice(k0 + D + 160, k0 + D + 176), -0.0),
        (2, slice(k0 + 200, k0 + 201), 900.0),
        (0, slice(v0 + 16, v0 + 32), 0.0),
        (1, slice(v0 + D, v0 + D + 16), torch.tensor([-0.0] * 8 + [0.0] * 8)),
        (3, slice(v0 + 40, v0 + 41), -900.0),
        (2, slice(v0 + D + 64, v0 + D + 80), 1e-30),
    )
    for r, cols, value in edges:
        if r < proj.shape[0]:
            proj[r, cols] = value


def _rows(spec):
    """(request slot, position) per row: one row a request, or under `verify` the
    `rows_per_request` consecutive positions of each request from its first."""
    per = spec.get("rows_per_request", 1)
    return [(1 + r, p + i) for r, p in enumerate(spec["pos"]) for i in range(per)]


def _build(spec, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    layout = _rows(spec)
    pos = [p for _, p in layout]
    rows = len(pos)
    last = {}
    for req, p in layout:
        last[req] = max(last.get(req, 0), p)
    slots = max(SLOTS, max(last) + 2)
    pages = max(SLOTS * WIDTH // 2, sum(p // TPB + 1 for p in last.values()) + 64)
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
    angles = torch.rand(TABLE, ROT // 2, generator=g, device=DEV) * 6.28
    cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).contiguous()
    proj = (torch.randn(rows, W, generator=g, device=DEV) * 2.0).to(torch.bfloat16)
    if spec.get("edge"):
        _edge_rows(proj)
    index_q = torch.zeros(rows, IH + 2, ID, dtype=torch.bfloat16, device=DEV)
    weights = [
        (0.3 * torch.randn(n, generator=g, device=DEV)).to(torch.bfloat16) for n in (D, D, ID, ID)
    ]
    if spec.get("nvfp4"):
        data = torch.randint(
            0, 256, (pages, 2, HK, TPB, D // 2), generator=g, device=DEV, dtype=torch.uint8
        )
        scales = torch.randint(
            0, 256, (pages, 2, HK, TPB, D // 16), generator=g, device=DEV, dtype=torch.uint8
        )
        plane = torch.randn(pages, INDEX_PAGE, generator=g, device=DEV).to(torch.bfloat16)
        index_cache = plane[:, : TPB * ID].view(pages, TPB, 1, ID)
        position_cache = plane[:, INDEX_PAGE - POSITION_PAD :].view(torch.int32)
        position_cache = position_cache.view(pages, TPB, 3)
        position_cache.copy_(
            torch.randint(0, TABLE, (pages, TPB, 3), generator=g, device=DEV, dtype=torch.int32)
        )
        global_scales = spec.get("global_scales", (1.0, 1.0, 1.0))
        kv = dict(
            k_cache=data[:, 0], v_cache=data[:, 1], k_scale_cache=scales[:, 0],
            v_scale_cache=scales[:, 1],
            kv_scale_orig_quant=torch.tensor(global_scales, dtype=torch.float32, device=DEV),
        )  # fmt: skip
    else:
        pool = torch.randn(pages, 2, HK, TPB, D, generator=g, device=DEV).to(torch.bfloat16)
        index_cache = torch.randn(pages, TPB, 1, ID, generator=g, device=DEV).to(torch.bfloat16)
        position_cache = torch.randint(
            0, TABLE, (pages, TPB, 3), generator=g, device=DEV, dtype=torch.int32
        )
        kv = dict(k_cache=pool[:, 0], v_cache=pool[:, 1])
    if spec.get("scores"):
        # Columns past a row's visible blocks are not written: a sentinel shows a stray one.
        kv["index_scores"] = torch.full(
            (rows, WIDTH * TPB // CR), -7.0, dtype=torch.float32, device=DEV
        )
    return dict(
        proj=proj, index_q=index_q[:, :IH], position_ids=ids, coords=coords, cos_sin=cos_sin,
        q_weight=weights[0], k_weight=weights[1], index_q_weight=weights[2], index_k_weight=weights[3],
        index_cache=index_cache, position_cache=position_cache, request_indices=request,
        logical_positions=logical, block_table=block_table,
        rows_per_request=spec.get("rows_per_request", 1), **kv, **GEOM,
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


def _assert_same(got, want, where="") -> None:
    """Every allocation bitwise equal: what the op writes and everything it must not."""
    assert got[1].keys() == want[1].keys()
    for key, copy in got[1].items():
        differ = (copy != want[1][key]).nonzero()
        assert differ.numel() == 0, f"{where}{key}: {differ.shape[0]} bytes differ"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    want = _clone(args)
    op.reference(**want[0])
    got = _clone(args)
    op(**got[0])
    torch.cuda.synchronize()
    _assert_same(got, want)


@pytest.mark.parametrize(
    "spec",
    [
        dict(pos=(8699, 8700)),
        dict(pos=(8699, 8700), nvfp4=True),
        dict(pos=(8697, 8700), rows_per_request=4, nvfp4=True),
        dict(pos=(8699, 8700), nvfp4=True, scores=True),
    ],
    ids=["bf16_pool", "nvfp4_pool", "nvfp4_pool_verify", "nvfp4_pool_scores"],
)
def test_reads_after_the_wait_under_graph_replay(spec) -> None:
    """The projection is rewritten inside the graph right before the op; with PDL a read of
    it before the wait would see the previous replay's row (in a verify step, also the
    raw index keys of a group's earlier rows)."""
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
    for trial in range(3):
        src.copy_((torch.randn(src.shape, device=DEV) * 2.0).to(src.dtype))
        before = _clone(work)
        graph.replay()
        torch.cuda.synchronize()
        want = _clone(before[0])
        want[0]["proj"].copy_(src)
        op.reference(**want[0])
        _assert_same((work, work_copies), want, where=f"replay {trial}: ")
