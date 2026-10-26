# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_attn_decode_prologue entry: every cell bitwise the
composition it replaces (`qwen4_exp_qk_norm_rope`, `qsa_paged_kv_store`,
`qsa_paged_index_store`, `qsa_prefill_compress`) on every buffer either writes -- the
projection, the index query, the K/V pool, the index plane, the position cache; a
stale-read check under graph replay with PDL."""

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
GEOM = dict(
    tokens_per_block=TPB, compress_ratio=CR, num_heads_q=HQ, num_heads_k=HK, head_dim=D,
    index_heads=IH, index_head_dim=ID, index_q_col=IQ_COL, rotary_dim=ROT, eps=1e-6,
    mrope_section1=11, mrope_section2=10,
)  # fmt: skip


def _build(spec, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    pos = list(spec["pos"])
    rows = len(pos)
    pages = SLOTS * WIDTH // 2
    perm = torch.randperm(pages, generator=g, device=DEV).to(torch.int32)
    block_table = torch.full((SLOTS, WIDTH), -1, dtype=torch.int32, device=DEV)
    used = 0
    for r, p in enumerate(pos):
        n = p // TPB + 1
        block_table[r + 1, :n] = perm[used : used + n]
        used += n
    kv = torch.randn(pages, 2, HK, TPB, D, generator=g, device=DEV).to(torch.bfloat16)
    index_cache = torch.randn(pages, TPB, 1, ID, generator=g, device=DEV).to(torch.bfloat16)
    position_cache = torch.randint(
        0, TABLE, (pages, TPB, 3), generator=g, device=DEV, dtype=torch.int32
    )
    request = torch.arange(1, rows + 1, dtype=torch.int32, device=DEV)
    if spec.get("unmapped") is not None:
        request[spec["unmapped"]] = SLOTS + 5
    logical = torch.tensor(pos, dtype=torch.int64, device=DEV)
    ids = torch.stack([logical, logical + 3, logical + 7]).to(torch.int32).contiguous()
    buf = torch.empty(3, 1, rows + 5, dtype=torch.int32, device=DEV)
    buf[:, 0, :rows] = ids
    coords = torch.transpose(buf[:, 0, :rows], 0, 1)
    angles = torch.rand(TABLE, ROT // 2, generator=g, device=DEV) * 6.28
    cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).contiguous()
    proj = (torch.randn(rows, W, generator=g, device=DEV) * 2.0).to(torch.bfloat16)
    index_q = torch.zeros(rows, IH + 2, ID, dtype=torch.bfloat16, device=DEV)
    weights = [
        (0.3 * torch.randn(n, generator=g, device=DEV)).to(torch.bfloat16) for n in (D, D, ID, ID)
    ]
    return dict(
        proj=proj, index_q=index_q[:, :IH], position_ids=ids, coords=coords, cos_sin=cos_sin,
        q_weight=weights[0], k_weight=weights[1], index_q_weight=weights[2], index_k_weight=weights[3],
        k_cache=kv[:, 0], v_cache=kv[:, 1], index_cache=index_cache, position_cache=position_cache,
        request_indices=request, logical_positions=logical, block_table=block_table, **GEOM,
    )  # fmt: skip


_WRITTEN = ("proj", "index_q", "k_cache", "v_cache", "index_cache", "position_cache")


def _clone(args):
    out = dict(args)
    kv = args["k_cache"]._base if args["k_cache"]._base is not None else None
    if kv is not None:
        kv2 = kv.clone()
        out["k_cache"], out["v_cache"] = kv2[:, 0], kv2[:, 1]
    iq = args["index_q"]._base if args["index_q"]._base is not None else args["index_q"]
    iq2 = iq.clone()
    out["index_q"] = iq2[:, : args["index_q"].shape[1]]
    for name in ("proj", "index_cache", "position_cache"):
        out[name] = args[name].clone()
    return out


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    want = _clone(args)
    op.reference(**want)
    got = _clone(args)
    op(**got)
    torch.cuda.synchronize()
    for name in _WRITTEN:
        assert torch.equal(got[name], want[name]), name


def test_reads_after_the_wait_under_graph_replay() -> None:
    """The projection is rewritten inside the graph right before the op; with PDL a read of
    it before the wait would see the previous replay's row."""
    args = _build(dict(pos=(8699, 8700)), seed=23)
    src = args["proj"].clone()
    work = _clone(args)

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
        want = _clone(before)
        want["proj"].copy_(src)
        op.reference(**want)
        for name in _WRITTEN:
            assert torch.equal(work[name], want[name]), f"replay {trial}: {name}"
