# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the moe_sort entry of the Qwen3.8-Flash-Next target.

The entry owns the contract. This file turns a cell into expert ids and holds
the claims no cell carries: CUDA-graph replay at the largest captured row
count, and the contiguity guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.moe_sort import (  # noqa: E501
    moe_sort as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "moe_sort requires a CUDA device"

DEV = "cuda"
E, K = 512, 10

_NAMES = (
    "tile_idx_to_expert_idx",
    "tile_idx_to_mn_limit",
    "expanded_idx_to_permuted_idx",
    "permuted_idx_to_expanded_idx",
    "total_num_padded_tokens",
    "num_non_exiting_tiles",
)


def _compare(out: list[torch.Tensor], ref: list[torch.Tensor]) -> None:
    """Integer outputs, so exact -- up to the one freedom the op has.

    Counts and the tile arrays' written prefix are bitwise. The order of an
    expert's rows is the kernel's atomic order (it varies between identical calls
    from T = 5), so `permuted_idx_to_expanded_idx` is gated per expert run as a
    set: each real row's value sorted within its run must equal the reference's
    run. `expanded_idx_to_permuted_idx` must then be the exact inverse: -1
    exactly where the expert is not local, and otherwise a real row of the right
    run that maps back. Padding rows and tiles past `num_non_exiting_tiles` are
    not written by the op and are not compared.
    """
    assert len(out) == len(ref) == len(_NAMES), (len(out), len(ref))
    for name, o, r in zip(_NAMES, out, ref):
        assert o.dtype == r.dtype == torch.int32, (name, o.dtype, r.dtype)
        assert o.shape == r.shape, (name, tuple(o.shape), tuple(r.shape))
    t2e, t2mn, e2p, p2e, total, n_tiles = out
    r_t2e, r_t2mn, r_e2p, r_p2e, r_total, r_n_tiles = ref
    assert torch.equal(n_tiles.cpu(), r_n_tiles.cpu()), (n_tiles, r_n_tiles)
    assert torch.equal(total.cpu(), r_total.cpu()), (total, r_total)
    n = int(r_n_tiles)
    assert torch.equal(t2e[:n], r_t2e[:n]), "tile_idx_to_expert_idx differs"
    assert torch.equal(t2mn[:n], r_t2mn[:n]), "tile_idx_to_mn_limit differs"

    tile = p2e.numel() // t2e.numel()
    num_expanded = e2p.numel()
    real = r_p2e >= 0
    rows = torch.nonzero(real).squeeze(1)
    run = r_t2e[rows // tile].long()
    got = p2e[real].long()
    assert bool(((got >= 0) & (got < num_expanded)).all()), (
        "permuted_idx_to_expanded_idx holds an out-of-range value on a real row"
    )
    assert torch.equal(
        torch.sort(run * num_expanded + got).values, run * num_expanded + r_p2e[real].long()
    ), "an expert's run of permuted_idx_to_expanded_idx holds the wrong pairs"

    flat, r_flat = e2p.reshape(-1).long(), r_e2p.reshape(-1).long()
    not_local = r_flat < 0
    assert bool((flat[not_local] == -1).all()), "a non-local pair is not -1"
    p = flat[~not_local]
    assert bool(((p >= 0) & (p < p2e.numel())).all()), "a local pair points outside the rows"
    assert bool(real[p].all()), "a local pair points at a padding row"
    assert torch.equal(p2e[p].long(), torch.arange(num_expanded, device=p.device)[~not_local]), (
        "expanded_idx_to_permuted_idx is not the inverse of permuted_idx_to_expanded_idx"
    )


def _ids(dist, tokens, tile, seed, num_experts=E, top_k=K):
    """Expert ids, `[tokens, top_k]` int32, every row naming distinct experts."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    if dist == "router":
        scores = torch.rand(tokens, num_experts, device=DEV, generator=g)
        return scores.topk(top_k, dim=-1).indices.to(torch.int32)
    if dist == "tile_edges":
        edge = (7, 300, 511)
        scores = torch.rand(tokens, num_experts, device=DEV, generator=g)
        scores[:, list(edge)] = -1.0
        ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
        for slot, (expert, rows) in enumerate(zip(edge, (tile - 1, tile, tile + 1))):
            ids[:rows, slot] = expert
        return ids
    raise ValueError(dist)


def _args(ids, tile, seed=0, scales_dtype=torch.float32):
    """The CuteDSL call site's keyword arguments at the checkpoint's values."""
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    scales = torch.rand(ids.shape, device=DEV, generator=g).to(scales_dtype)
    return dict(
        token_selected_experts=ids,
        token_final_scales=scales,
        num_experts=E,
        top_k=K,
        local_expert_offset=0,
        local_num_experts=E,
        tile_tokens_dim=tile,
    )


def _check(args):
    """Drive the op and gate it through the entry's own reference and `_compare`."""
    out = op(**args)
    ref = op.reference(**args)
    _compare(out, ref)
    return out, ref


def _cell_id(cell):
    s = cell.spec
    return f"t{s['tile_tokens_dim']}-T{s['tokens']}-{s['dist']}"


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[_cell_id(c) for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    s = cell.spec
    seed = zlib.crc32(cell.why.encode())
    ids = _ids(s["dist"], s["tokens"], s["tile_tokens_dim"], seed)
    args = _args(ids, s["tile_tokens_dim"], seed, s["scales_dtype"])
    assert (s["num_experts"], s["top_k"]) == (args["num_experts"], args["top_k"])
    assert (s["local_expert_offset"], s["local_num_experts"]) == (0, E)
    ids_before = ids.clone()
    scales_before = args["token_final_scales"].clone()
    out, ref = _check(args)
    assert torch.equal(ids, ids_before), "token_selected_experts was written"
    assert torch.equal(args["token_final_scales"], scales_before), "token_final_scales was written"
    if s["dist"] == "tile_edges":
        tile = s["tile_tokens_dim"]
        n = int(ref[5])
        limits = out[1][:n].long()
        starts = torch.arange(n, device=DEV) * tile
        fill = (limits - starts).tolist()
        experts = out[0][:n].tolist()
        per_expert = {e: [] for e in (7, 300, 511)}
        for e, f in zip(experts, fill):
            if e in per_expert:
                per_expert[e].append(f)
        assert per_expert == {7: [tile - 1], 300: [tile], 511: [tile, 1]}, per_expert


# ── CUDA graphs: captured row counts, on new ids ──────────────────────────

#: The largest decode capture: 32 requests at MTP verify width 4, cluster kernel.
_CAPTURED_ROWS = 128


def _capture(args):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            op(**args)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = op(**args)
    return graph, static_out


def test_captured_rows_replay_on_new_ids(certifying):  # noqa: F811
    """The graph is replayed on two fresh id sets and gated against the reference,
    and against an eager call on the same ids: counts and tile arrays bitwise. At
    the target's tile, 128."""
    certifying(op)
    tile, rows = 128, _CAPTURED_ROWS
    args = _args(_ids("router", rows, tile, seed=rows), tile, seed=rows)
    graph, static_out = _capture(args)
    for trial in (1, 2):
        fresh = _ids("router", rows, tile, seed=1000 * trial + rows)
        args["token_selected_experts"].copy_(fresh)
        graph.replay()
        eager = op(**dict(args, token_selected_experts=fresh.clone()))
        ref = op.reference(**args)
        _compare(static_out, ref)
        _compare(eager, ref)
        n = int(ref[5])
        for i in (0, 1):
            assert torch.equal(static_out[i][:n], eager[i][:n]), (trial, i)
        for i in (4, 5):
            assert torch.equal(static_out[i], eager[i]), (trial, i)
    del graph, static_out
    torch.cuda.synchronize()


# ── the guard ─────────────────────────────────────────────────────────────


def test_guard_refuses_noncontiguous_ids():
    """What `is_valid` exists for: the op reads a strided view as the dense buffer
    that starts at its data pointer, and answers without raising."""
    wide = _ids("router", 300, 128, seed=51, top_k=16)
    sc = torch.rand(300, K, device=DEV)
    with validating(op), pytest.raises(AssertionError, match="contiguous"):
        op(wide[:, :K], sc, E, K, 0, E, 128)
