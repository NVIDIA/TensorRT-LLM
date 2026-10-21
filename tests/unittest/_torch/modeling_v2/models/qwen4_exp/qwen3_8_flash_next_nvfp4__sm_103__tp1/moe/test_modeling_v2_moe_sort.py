# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the moe_sort entry of the Qwen3.8-Flash-Next target.

The entry owns the contract. This file turns a cell into expert ids and holds
the claims no cell carries: CUDA-graph replay at the captured row counts, the
eager row sweep, which regions the op leaves unwritten, the token-order layout
of the block kernel, the scales being inert, the data hazards, the loud
rejections and the guards.
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
SENTINEL = 0x5A5A5A5A

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
    if dist == "hot":
        hot = torch.randperm(num_experts, device=DEV, generator=g)[:top_k]
        slots = torch.rand(tokens, top_k, device=DEV, generator=g).argsort(dim=-1)
        return hot[slots].to(torch.int32)
    if dist == "tile_edges":
        edge = (7, 300, 511)
        scores = torch.rand(tokens, num_experts, device=DEV, generator=g)
        scores[:, list(edge)] = -1.0
        ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
        for slot, (expert, rows) in enumerate(zip(edge, (tile - 1, tile, tile + 1))):
            ids[:rows, slot] = expert
        return ids
    if dist == "round_robin":
        base = torch.arange(tokens, device=DEV).unsqueeze(1) * top_k
        return ((base + torch.arange(top_k, device=DEV)) % num_experts).to(torch.int32)
    if dist == "strided":
        h = torch.randint(0, num_experts, (tokens, 1), device=DEV, generator=g)
        return ((h + torch.arange(top_k, device=DEV) * 51) % num_experts).to(torch.int32)
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
    if s["dist"] == "round_robin":
        # every pair on its own expert: the shape bound is reached exactly
        assert int(out[5]) == out[0].numel() == s["tokens"] * s["top_k"]
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

#: Row counts of a decode capture (requests x MTP verify width 1..4, up to 32
#: requests): both ends of the block kernel (T <= 4), the first cluster-kernel T,
#: and the largest capture.
_CAPTURED_ROWS = (1, 4, 5, 128)


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
    """Each graph is replayed on two fresh id sets and gated against the reference,
    and against an eager call on the same ids: counts and tile arrays bitwise, the
    whole output bitwise where the kernel is deterministic (T <= 4). At the
    target's tile, 128."""
    certifying(op)
    tile = 128
    for rows in _CAPTURED_ROWS:
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
                assert torch.equal(static_out[i][:n], eager[i][:n]), (rows, trial, i)
            for i in (4, 5):
                assert torch.equal(static_out[i], eager[i]), (rows, trial, i)
            if rows <= 4:
                real = ref[3] >= 0
                assert torch.equal(static_out[2], eager[2]), (rows, trial)
                assert torch.equal(static_out[3][real], eager[3][real]), (rows, trial)
        del graph, static_out
    torch.cuda.synchronize()


@pytest.mark.parametrize("rows", (3, 300))
def test_padding_rows_and_spare_tiles_are_not_written(rows):
    """A sentinel planted in a captured graph's outputs survives replay in exactly
    the regions `_compare` excludes; under a graph those regions keep the previous
    replay's indices. Block kernel (3 rows) and cluster kernel (300), tile 128."""
    tile = 128
    args = _args(_ids("router", rows, tile, seed=5), tile, seed=5)
    graph, static_out = _capture(args)
    for trial in (1, 2):
        args["token_selected_experts"].copy_(_ids("router", rows, tile, seed=50 + trial))
        for t in static_out[:4]:
            t.fill_(SENTINEL)
        graph.replay()
        ref = op.reference(**args)
        _compare(static_out, ref)
        n = int(ref[5])
        real = ref[3] >= 0
        assert bool((static_out[0][n:] == SENTINEL).all()) and bool(
            (static_out[1][n:] == SENTINEL).all()
        )
        assert bool((static_out[3][~real] == SENTINEL).all())
        assert not bool((static_out[3][real] == SENTINEL).any())
        assert not bool((static_out[2] == SENTINEL).any()), (
            "expanded_idx_to_permuted_idx is fully written"
        )
    # replay on other ids without re-planting: the unwritten rows keep the last replay's values
    before = static_out[3].clone()
    args["token_selected_experts"].copy_(_ids("router", rows, tile, seed=99))
    graph.replay()
    real = op.reference(**args)[3] >= 0
    assert torch.equal(static_out[3][~real], before[~real])
    if rows == 300:
        stale = before[~real]
        assert int(((stale >= 0) & (stale < rows * K)).sum()) > 0, "no valid-looking stale row"
    del graph, static_out


# ── eager: the row sweep ──────────────────────────────────────────────────

#: Both sides of every kernel-path edge: block / cluster at 4 / 5, the cluster
#: kernel's per-block work edges at 102 / 103 and 204 / 205 rows, its fast / slow
#: path edges at 3276 / 3277 and 6553 / 6554, and the 8192-row ceiling.
_SWEEP = (4, 5, 102, 103, 204, 205, 3276, 3277, 6553, 6554, 8192)


def test_eager_row_sweep(certifying):  # noqa: F811
    certifying(op)
    tile = 128
    for tokens in _SWEEP:
        _check(_args(_ids("router", tokens, tile, seed=tokens), tile, seed=tokens))


# ── layout facts ──────────────────────────────────────────────────────────


def test_block_kernel_rows_follow_token_order():
    """T <= 4: each expert's rows in token order, bitwise the reference's."""
    for tile in (128, 256):
        for tokens in (1, 4):
            for seed in range(2):
                args = _args(_ids("router", tokens, tile, seed=seed), tile, seed=seed)
                out, ref = _check(args)
                real = ref[3] >= 0
                assert torch.equal(out[2], ref[2]) and torch.equal(out[3][real], ref[3][real])


def test_scales_are_inert_and_untouched():
    """The scales' values and dtype never reach an output, and are not written."""
    for tokens in (3, 300):
        ids = _ids("router", tokens, 128, seed=21)
        outs = []
        for scales in (
            torch.rand(tokens, K, device=DEV),
            torch.rand(tokens, K, device=DEV).to(torch.bfloat16),
            torch.zeros(tokens, K, device=DEV),
            torch.full((tokens, K), float("nan"), device=DEV),
        ):
            bits = scales.view(
                torch.int32 if scales.dtype == torch.float32 else torch.int16
            ).clone()
            args = dict(_args(ids, 128), token_final_scales=scales)
            out, ref = _check(args)
            assert torch.equal(scales.view(bits.dtype), bits), "token_final_scales was written"
            outs.append(out)
        if tokens <= 4:  # deterministic: every written element bitwise across the four
            n, real = int(ref[5]), ref[3] >= 0
            for out in outs[1:]:
                assert torch.equal(out[0][:n], outs[0][0][:n]) and torch.equal(
                    out[1][:n], outs[0][1][:n]
                )
                assert torch.equal(out[2], outs[0][2]) and torch.equal(
                    out[3][real], outs[0][3][real]
                )
                assert torch.equal(out[4], outs[0][4]) and torch.equal(out[5], outs[0][5])


# ── data hazards is_valid cannot see ──────────────────────────────────────


def test_duplicate_expert_in_a_row():
    """Block kernel (T <= 4) keeps one of two slots naming the same expert;
    the other paths keep both."""
    ids = _ids("router", 2, 128, seed=31)
    ids[:, 1] = ids[:, 0]
    args = _args(ids, 128)
    out = op(**args)
    ref = op.reference(**args)
    for row in range(2):
        pair = out[2][row, :2].tolist()
        assert pair.count(-1) == 1, pair
    assert int((out[2] >= 0).sum()) == 2 * K - 2 and int((ref[2] >= 0).sum()) == 2 * K
    with pytest.raises(AssertionError):
        _compare(out, ref)

    ids = _ids("router", 300, 128, seed=32)
    ids[:, 1] = ids[:, 0]
    _check(_args(ids, 128))


def test_ids_past_int16_alias_off_the_block_kernel():
    """An id of 65536 + e is not local, and the block kernel says so; the
    cluster kernel reads ids through int16 and sorts it as expert e."""
    for tokens in (2, 300):
        ids = _ids("router", tokens, 128, seed=33)
        ids[0, 0] = 40000  # negative as int16: non-local everywhere
        out, _ = _check(_args(ids, 128))
        assert int(out[2][0, 0]) == -1
        ids[0, 0] = 65536 + 3
        out = op(**_args(ids, 128))
        p = int(out[2][0, 0])
        if tokens <= 4:
            assert p == -1
            _compare(out, op.reference(**_args(ids, 128)))
        else:
            assert p >= 0 and int(out[0][p // 128]) == 3
            with pytest.raises(AssertionError):
                _compare(out, op.reference(**_args(ids, 128)))


# ── loud rejections ───────────────────────────────────────────────────────


def test_op_rejects_loudly():
    """Rejected by the op itself; none of these needs a guard."""
    ids = _ids("router", 64, 128, seed=41)
    sc = torch.rand(64, K, device=DEV)
    wide33 = _ids("router", 8, 128, seed=42, top_k=33)
    cases = [
        ("int64 ids", lambda: op(ids.long(), sc, E, K, 0, E, 128), "expected scalar type Int"),
        ("1-D ids", lambda: op(ids.reshape(-1), sc, E, K, 0, E, 128), "must be 2D"),
        ("top_k vs width", lambda: op(ids, sc, E, 9, 0, E, 128), r"size\(1\) must be top_k"),
        ("scales width", lambda: op(ids, sc[:, :9], E, K, 0, E, 128), r"size\(1\) must be top_k"),
        ("scales rows", lambda: op(ids, sc[:63], E, K, 0, E, 128), "must be num_tokens"),
        ("top_k 33", lambda: op(wide33, torch.rand(8, 33, device=DEV), E, 33, 0, E, 128), "<= 32"),
        ("num_experts 2049", lambda: op(ids, sc, 2049, K, 0, 2049, 128), "no more than 2048"),
        ("T = 0", lambda: op(ids[:0], sc[:0], E, K, 0, E, 128), "mPtrTopKWeights must also"),
    ]
    for tag, call, match in cases:
        with pytest.raises(RuntimeError, match=match):
            call()
    with pytest.raises(NotImplementedError):
        op(ids.cpu(), sc.cpu(), E, K, 0, E, 128)


# ── the guards ────────────────────────────────────────────────────────────


def test_unguarded_noncontiguous_ids_answer_wrongly():
    """What `is_valid` exists for: the op reads a strided view as the dense buffer
    that starts at its data pointer, and answers without raising."""
    wide = _ids("router", 300, 128, seed=51, top_k=16)
    view = wide[:, :K]
    sc = torch.rand(300, K, device=DEV)
    out = op(view, sc, E, K, 0, E, 128)
    dense = wide.reshape(-1)[: 300 * K].reshape(300, K)
    _compare(out, op.reference(dense, sc, E, K, 0, E, 128))
    with pytest.raises(AssertionError):
        _compare(out, op.reference(view.contiguous(), sc, E, K, 0, E, 128))


def test_guards_refuse():
    """The 2^31-element operands are expanded views of one row: the guard reads
    only their shape, so nothing is allocated."""
    wide = _ids("router", 300, 128, seed=51, top_k=16)
    sc = torch.rand(300, K, device=DEV)
    ids = _ids("router", 300, 128, seed=53)
    base = ids[:1]
    with validating(op):
        with pytest.raises(AssertionError, match="contiguous"):
            op(wide[:, :K], sc, E, K, 0, E, 128)
        with pytest.raises(AssertionError, match="contiguous"):
            op(ids.t().contiguous().t(), sc, E, K, 0, E, 128)
        for tokens in (214748365, 429496730):
            with pytest.raises(AssertionError, match=r"2\^31"):
                op(base.expand(tokens, K), sc[:1].expand(tokens, K), E, K, 0, E, 128)
        # tile_tokens_dim = 0 is refused but not driven unguarded: the host divides
        # by it, which returns garbage on aarch64 and traps on x86.
        with pytest.raises(AssertionError, match="tile_tokens_dim must be positive"):
            op(ids, sc, E, K, 0, E, 0)
        out = op(wide[:, :K].contiguous(), sc, E, K, 0, E, 128)
        _compare(out, op.reference(wide[:, :K].contiguous(), sc, E, K, 0, E, 128))
