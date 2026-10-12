# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the candidate_block_select entry.

The entry owns what is certified: `CELLS` is the list, `reference` and `compare` are the gate.
The reference is the checkpoint's own ``select_candidate_blocks`` (DeepSeek-V4.1
``inference/model.py``) rebuilt from torch ops with the entry's tie rule: -inf-pad to whole
blocks, block amax, pin the block holding the newest visible position to +inf, top-
``topk_blocks`` blocks, drop picks whose value is -inf, expand to positions. Candidate masking
only binds once a row has more than ``topk_blocks`` blocks, i.e. above 16384 visible positions at
the checkpoint's 2048 x 8, so every selecting case here runs above that.

This file turns a cell's spec into tensors and keeps the checks a cell cannot carry: the wrapper
guards, CUDA-graph capture and replay, both launch configurations agreeing, the deliberately-wrong
controls, the kernel-variant maps, the seeded random packed steps, the row-prefix and
row-independence checks, the decode calls at row offsets and the far end of the maximal cell.
"""

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.candidate_block_select import (  # noqa: E501
    candidate_block_select as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "candidate_block_select requires a CUDA device"

DEV = "cuda"
TOPK_BLOCKS = 2048
BLOCK = 8
BINDS_ABOVE = TOPK_BLOCKS * BLOCK  # 16384 visible positions


def _distinct(n: int, w: int, seed: int) -> torch.Tensor:
    """fp32 [n, w], each row a random permutation of distinct values of both signs.

    Distinct scores make every block max distinct, so the kept set is unique and the
    comparison with torch.topk is exact.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    perm = torch.argsort(torch.rand(n, w, generator=g, device=DEV), dim=1)
    return (perm.float() - w / 2) * 0.03125


def _distinct_from(g: torch.Generator, n: int, w: int) -> torch.Tensor:
    """``_distinct``'s rows drawn from an existing generator (for row-chunked fills)."""
    perm = torch.argsort(torch.rand(n, w, generator=g, device=DEV), dim=1)
    return (perm.float() - w / 2) * 0.03125


def _distinct_any(n: int, w: int, seed: int) -> torch.Tensor:
    """``_distinct``, row-chunked (at most 2^27 elements a chunk) above 2^29 elements, where one
    argsort would need tens of GB; every row is still a random permutation of distinct values
    (|value| < 2^23 / 32, exact in fp32). Shapes up to 2^29 elements get ``_distinct``'s values."""
    if n * w <= 1 << 29:
        return _distinct(n, w, seed)
    out = torch.empty(n, w, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    step = max(1, (1 << 27) // w)
    for r0 in range(0, n, step):
        out[r0 : r0 + step] = _distinct_from(g, min(step, n - r0), w)
    return out


def _ref_select(logits, compress_lens, topk_blocks=TOPK_BLOCKS, block_size=BLOCK, pin=True):
    """The checkpoint's select_candidate_blocks as it stands, ``torch.topk`` and all: the source
    of the controls (``pin=False``, another ``block_size``) and of the kept-multiset check on tied
    scores. The expected mask itself is always ``op.reference``."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    if pin:
        last = (compress_lens - 1) // block_size
        scores = scores.masked_fill(
            torch.arange(num_blocks, device=logits.device) == last, torch.inf
        )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _ref_top_positions(logits, k=TOPK_BLOCKS * BLOCK):
    """Control: keep the top-k *positions* instead of blocks (same position budget)."""
    top = logits.topk(min(k, logits.size(-1)), dim=-1)
    return torch.zeros_like(logits, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )


def _prefill_reference(logits, first_pos):
    """Reference mask for one request's causal prefill rows (ratio 1: query p sees p + 1).

    ``logits`` [q, W] holds the request's columns from 0; row r is the query at position
    ``first_pos + r``, so it sees columns [0, first_pos + r + 1). Returns (mask, visible lengths).
    """
    q, _ = logits.shape
    lens = torch.arange(first_pos + 1, first_pos + q + 1, device=DEV)
    return op.reference(logits, None, lens.int(), TOPK_BLOCKS, BLOCK, 1), lens


def _check_decode_rows(logits, lens, mask):
    """Row i sees columns [0, lens[i]) (decode, next_n 1): the whole mask against the reference."""
    op.compare(mask, op.reference(logits, None, lens, TOPK_BLOCKS, BLOCK, 1))


# ---------------------------------------------------------------------------------------------
# the certified cells (op.CELLS): prefill, decode, ties, -inf blocks, empty rows, strided logits,
# and the packed prefill steps untiled and in the target's row tiles
# ---------------------------------------------------------------------------------------------


def _kept_block_scores(logits_row, keep_row):
    width = logits_row.numel()
    s = F.pad(logits_row.float(), (0, -width % BLOCK), value=-torch.inf).view(-1, BLOCK).amax(-1)
    k = F.pad(keep_row, (0, -width % BLOCK), value=False).view(-1, BLOCK).any(-1)
    return torch.sort(s[k]).values


def _check_kept_multiset(logits, lens, got):
    for i, n in enumerate(lens.tolist()):
        ref = _ref_select(logits[i : i + 1, :n], n)[0]
        assert torch.equal(
            _kept_block_scores(logits[i, :n], got[i, :n]), _kept_block_scores(logits[i, :n], ref)
        ), f"row {i}: kept block scores differ from the checkpoint's selection"


def _build(spec):
    """The call's arguments for a non-packed cell, from the spec (seeds fixed per kind)."""
    kind = spec["kind"]
    i32 = dict(dtype=torch.int32, device=DEV)
    if kind == "prefill_one":
        seq = spec["seq"]
        starts = torch.zeros(seq, **i32) if spec["explicit_starts"] else None
        lens = torch.arange(1, seq + 1, **i32)
        return _distinct(seq, seq, seed=seq), starts, lens, TOPK_BLOCKS, BLOCK, 1
    if kind == "prefill_chunk":
        first, q = spec["first"], spec["q"]
        w = first + q
        logits = _distinct(q, w + spec["pad"], seed=7)[:, :w]
        assert logits.stride(0) == w + spec["pad"]
        lens = torch.arange(first + 1, first + q + 1, **i32)
        return logits, torch.zeros(q, **i32), lens, TOPK_BLOCKS, BLOCK, 1
    if kind == "prefill_ragged":
        keys, queries = spec["keys"], spec["queries"]
        w = sum(keys) + spec["spare"]
        logits = _distinct(sum(q for _, q in queries), w, seed=11)
        starts, ends, base = [], [], 0
        for k, (first, q) in zip(keys, queries):
            starts.append(torch.full((q,), base, **i32))
            ends.append(base + torch.arange(first + 1, first + q + 1, **i32))
            base += k
        return logits, torch.cat(starts), torch.cat(ends), TOPK_BLOCKS, BLOCK, 1
    if kind == "decode":
        lens = torch.tensor(spec["lens"], **i32)
        w = int(lens.max()) + 3
        logits = _distinct(lens.numel(), w, seed=3)
        # columns past each row's visible range hold NaN / +inf: they must be neither read nor kept
        logits[torch.arange(w, device=DEV) >= lens[:, None]] = float("nan")
        logits[:, -1] = float("inf")
        return logits, None, lens, TOPK_BLOCKS, BLOCK, 1
    if kind == "decode_next_n":
        visible, next_n = torch.tensor(spec["visible"], **i32), spec["next_n"]
        logits = _distinct(visible.numel() * next_n, int(visible.max()), seed=5)
        return logits, None, visible, TOPK_BLOCKS, BLOCK, next_n
    if kind == "bf16":
        lens = torch.tensor(spec["lens"], **i32)
        g = torch.Generator(device=DEV).manual_seed(9)
        logits = torch.randn(lens.numel(), int(lens.max()), generator=g, device=DEV)
        return logits.to(torch.bfloat16), None, lens, TOPK_BLOCKS, BLOCK, 1
    if kind == "ties":
        lens = torch.tensor(spec["lens"], **i32)
        w = int(lens.max())
        g = torch.Generator(device=DEV).manual_seed(13)
        choices = torch.tensor([-1.0, -0.0, 0.0, 1.0, 2.0], device=DEV)
        logits = choices[torch.randint(0, 5, (lens.numel(), w), generator=g, device=DEV)]
        # last row: almost every block scores -0.0 or +0.0, which rank equal (the threshold tie)
        logits[3] = choices[torch.randint(0, 3, (w,), generator=g, device=DEV)]
        return logits, None, lens, TOPK_BLOCKS, BLOCK, 1
    if kind == "minus_inf":
        lens = torch.tensor(spec["lens"], **i32)
        logits = _distinct(lens.numel(), int(lens.max()), seed=17)
        blocks = logits.view(lens.numel(), -1, BLOCK)
        # row 0: 1000 all -inf blocks of 2500 -> 1500 finite blocks, fewer than 2048: all kept
        blocks[0, 100:1100] = float("-inf")
        # row 1: same, and the pinned (newest) block itself all -inf: kept anyway
        blocks[1, 100:1100] = float("-inf")
        blocks[1, (20000 - 1) // BLOCK] = float("-inf")
        # row 2: scattered -inf positions inside finite blocks do not change a block's score
        logits[2, ::3] = float("-inf")
        return logits, None, lens, TOPK_BLOCKS, BLOCK, 1
    if kind == "empty_rows":
        lens = torch.tensor(spec["lens"], **i32)
        return _distinct(lens.numel(), spec["width"], seed=19), None, lens, TOPK_BLOCKS, BLOCK, 1
    if kind == "no_rows":
        logits = _distinct(3, spec["width"], seed=19)[:0]
        return logits, None, torch.zeros(0, **i32), TOPK_BLOCKS, BLOCK, 1
    if kind == "col_strided":
        lens = torch.tensor(spec["lens"], **i32)
        logits = _distinct(lens.numel(), int(lens.max()), seed=23).t().contiguous().t()
        assert logits.stride(1) == lens.numel()
        return logits, None, lens, TOPK_BLOCKS, BLOCK, 1
    raise AssertionError(kind)


def _drive_packed(spec):
    """A packed prefill step called as the target calls it: whole, or one call per row tile on
    the step's logits rows [a:b] with the views ks[a:b] / ke[a:b]. Every mask lands on a block
    pre-filled with 1s, so every byte of it has to be written."""
    q, kv = list(spec["q_lens"]), list(spec["kv_lens"])
    seed = 8 * (sum(kv) % 100_000) + len(q) if spec["tiled"] else sum(kv) + len(q)
    try:
        step = _packed_step(q, kv, seed=seed, with_want=False)
        T, W = step["logits"].shape
        tiles = _tiles(T, W) if spec["tiled"] else [(0, T)]
        reused = 0
        for a, b in tiles:
            torch.cuda.empty_cache()
            junk = torch.ones((b - a) * W, dtype=torch.uint8, device=DEV)
            junk_ptr = junk.data_ptr()
            del junk
            args = (step["logits"][a:b], step["ks"][a:b], step["ke"][a:b], TOPK_BLOCKS, BLOCK, 1)
            got = op(*args)
            reused += got.data_ptr() == junk_ptr
            op.compare(got, op.reference(*args))
            del got
        vis = step["ke"] - step["ks"]
        print(
            f"\n  T={T} W={W} k={len(q)} tiles {[b - a for a, b in tiles][:6]}"
            f"{'...' if len(tiles) > 6 else ''} ({len(tiles)}): exact; "
            f"{int((vis > BINDS_ABOVE).sum())} binding rows, longest {int(vis.max())}; "
            f"{reused} of {len(tiles)} masks on a poisoned block"
        )
        del step
    finally:
        _release_big_logits()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    if spec["kind"] == "packed":
        _drive_packed(spec)
        return
    args = _build(spec)
    logits, _, lens, _, _, _ = args
    got = op(*args)
    assert got.dtype == torch.bool and got.shape == logits.shape and got.is_contiguous()
    op.compare(got, op.reference(*args))

    kind = spec["kind"]
    if kind == "prefill_one":
        # masking is active exactly on the rows that see more than 2048 blocks
        seq = spec["seq"]
        reachable = torch.arange(seq, device=DEV) < lens[:, None]
        narrowed = (got != reachable).any(dim=1)
        binding = lens > BINDS_ABOVE
        assert torch.equal(narrowed, binding)
        if binding.any():
            kept_blocks = got[binding].view(-1, seq // BLOCK, BLOCK).any(-1).sum(-1)
            assert bool((kept_blocks == TOPK_BLOCKS).all())
    elif kind == "prefill_chunk":
        assert int((lens > BINDS_ABOVE).sum()) == spec["first"] + spec["q"] - BINDS_ABOVE
    elif kind in ("bf16", "ties"):
        # ties are frequent: the kept block-score multiset equals the checkpoint's own selection
        _check_kept_multiset(logits, lens, got)
    elif kind == "minus_inf":
        assert int(got[0].sum()) == (2500 - 1000) * BLOCK
        assert bool(got[1, 19992:20000].all())
    elif kind == "empty_rows":
        assert not got[0].any() and not got[2].any() and bool(got[1].all())
    elif kind == "no_rows":
        assert got.shape == (0, spec["width"])


def test_both_launch_configs_agree():
    """Calls of <= 256 rows and of more rows take different launch configurations."""
    rows = 512
    g = torch.Generator(device=DEV).manual_seed(59)
    lens = torch.randint(15000, 40000, (rows,), generator=g, device=DEV, dtype=torch.int32)
    logits = _distinct(rows, 40000, seed=61)
    whole = op(logits, None, lens, TOPK_BLOCKS, BLOCK, 1)
    halves = torch.cat(
        [
            op(logits[:256], None, lens[:256], TOPK_BLOCKS, BLOCK, 1),
            op(logits[256:], None, lens[256:], TOPK_BLOCKS, BLOCK, 1),
        ]
    )
    assert torch.equal(whole, halves)
    _check_decode_rows(logits, lens, whole)


def test_is_valid_rejects_mismatched_length_tensors():
    """The kernel would read these wrongly without an error; the entry's guard refuses them."""
    logits = _distinct(8, 64, seed=67)
    lens = torch.full((8,), 64, dtype=torch.int32, device=DEV)
    zeros = torch.zeros(8, dtype=torch.int32, device=DEV)
    rejected = (
        ("row_ends", (logits, None, lens[:4], TOPK_BLOCKS, BLOCK, 1)),
        ("row_ends", (logits, None, lens, TOPK_BLOCKS, BLOCK, 6)),
        ("row_ends", (logits, None, torch.stack([lens, lens], 1)[:, 0], TOPK_BLOCKS, BLOCK, 1)),
        ("row_starts", (logits, zeros[:4], lens, TOPK_BLOCKS, BLOCK, 1)),
        ("row_starts", (logits, torch.stack([zeros, zeros], 1)[:, 0], lens, TOPK_BLOCKS, BLOCK, 1)),
    )
    for match, args in rejected:
        with pytest.raises(AssertionError, match=match):
            with validating(op):
                op(*args)
    # the guard admits the well-formed call
    with validating(op):
        op(logits, zeros, lens, TOPK_BLOCKS, BLOCK, 1)


# ---------------------------------------------------------------------------------------------
# CUDA graph
# ---------------------------------------------------------------------------------------------


def test_cuda_graph_capture_and_replay():
    rows, w = 8, 40000
    logits = torch.empty(rows, w, device=DEV)
    lens = torch.empty(rows, dtype=torch.int32, device=DEV)
    logits.copy_(_distinct(rows, w, seed=29))
    lens.copy_(
        torch.tensor([16385, 20000, 40000, 30000, 17000, 100, 16384, 39999], dtype=torch.int32)
    )
    op(logits, None, lens, TOPK_BLOCKS, BLOCK, 1)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(logits, None, lens, TOPK_BLOCKS, BLOCK, 1)
    for seed, new_lens in (
        (31, [40000] * 8),
        (37, [16400, 16393, 20001, 1, 35000, 16385, 39000, 22222]),
    ):
        logits.copy_(_distinct(rows, w, seed=seed))
        lens.copy_(torch.tensor(new_lens, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        _check_decode_rows(logits, lens, out)


# ---------------------------------------------------------------------------------------------
# deliberately-wrong controls: each matches above 16384 visible positions only if the test
# cannot see the difference; each must match below it (where masking is a no-op)
# ---------------------------------------------------------------------------------------------

CONTROL_LENS = [
    4096,
    12000,
    16384,
    16385,
    16400,
    17000,
    18000,
    20000,
    24000,
    28000,
    32768,
    40000,
    50000,
    65536,
    100000,
    131072,
]


def _control_masks(logits, lens):
    """(entry mask, {control: mask}) per row, all over each row's own visible width."""
    got = op(logits, None, lens, TOPK_BLOCKS, BLOCK, 1)
    got16 = op(logits, None, lens, TOPK_BLOCKS, 16, 1)
    rows = []
    for i, n in enumerate(lens.tolist()):
        row = logits[i : i + 1, :n]
        rows.append(
            (
                n,
                got[i, :n],
                {
                    "no pin of the newest block": _ref_select(row, n, pin=False)[0],
                    "block size 16": _ref_select(row, n, block_size=16)[0],
                    "top-k of positions, not blocks": _ref_top_positions(row)[0],
                    "entry kernel run with block_size 16": got16[i, :n],
                },
            )
        )
    return rows


def test_controls_differ_above_16384_only():
    lens = torch.tensor(CONTROL_LENS, dtype=torch.int32, device=DEV)
    logits = _distinct(lens.numel(), int(lens.max()), seed=41)
    consumer = _distinct(lens.numel(), int(lens.max()), seed=43)  # a Reindex layer's own scores
    rows = _control_masks(logits, lens)
    for _, _, controls in rows:  # the kernel honours block_size: its 16 run is the 16 reference
        assert torch.equal(
            controls["entry kernel run with block_size 16"], controls["block size 16"]
        )
    wrong_rows = {}
    wrong_top512 = {}
    for i, (n, entry, controls) in enumerate(rows):
        op.compare(
            entry[None],
            op.reference(logits[i : i + 1, :n], None, lens[i : i + 1], TOPK_BLOCKS, BLOCK, 1),
        )
        entry_top = consumer[i, :n].masked_fill(~entry, -torch.inf).topk(512).indices.sort().values
        for name, ctl in controls.items():
            differs = not torch.equal(ctl, entry)
            if n <= BINDS_ABOVE:
                assert not differs, f"control '{name}' differs at visible {n} <= 16384"
                continue
            wrong_rows[name] = wrong_rows.get(name, 0) + int(differs)
            ctl_top = consumer[i, :n].masked_fill(~ctl, -torch.inf).topk(512).indices.sort().values
            wrong_top512[name] = wrong_top512.get(name, 0) + int(
                not torch.equal(ctl_top, entry_top)
            )
    binding_rows = sum(1 for n in CONTROL_LENS if n > BINDS_ABOVE)
    print(f"\ncontrols over {binding_rows} rows with more than 16384 visible positions:")
    for name in wrong_rows:
        print(
            f"  {name}: mask differs on {wrong_rows[name]} rows, consumer top-512 differs on {wrong_top512[name]} rows"
        )
    for name, n_wrong in wrong_rows.items():
        assert n_wrong > 0, f"control '{name}' was not detected above 16384"


def test_controls_on_a_20480_token_prefill():
    """The same controls over every query row of a 20480-token prefill (visible 1 .. 20480)."""
    seq = 20480
    logits = _distinct(seq, seq, seed=47)
    consumer = _distinct(seq, seq, seed=53)
    compress_lens = torch.arange(1, seq + 1, device=DEV).unsqueeze(-1)
    unreachable = torch.arange(seq, device=DEV) >= compress_lens
    masked = logits.masked_fill(unreachable, -torch.inf)
    got = op(logits, None, compress_lens.squeeze(-1).int(), TOPK_BLOCKS, BLOCK, 1)
    op.compare(
        got, op.reference(logits, None, compress_lens.squeeze(-1).int(), TOPK_BLOCKS, BLOCK, 1)
    )
    controls = {
        "no pin of the newest block": _ref_select(masked, compress_lens, pin=False) & ~unreachable,
        "block size 16": _ref_select(masked, compress_lens, block_size=16) & ~unreachable,
        "top-k of positions, not blocks": _ref_top_positions(masked) & ~unreachable,
    }
    binding = compress_lens.squeeze(-1) > BINDS_ABOVE
    cons = consumer.masked_fill(unreachable, -torch.inf)[binding]
    entry_top = (
        cons.masked_fill(~got[binding], -torch.inf).topk(512, dim=-1).indices.sort(dim=-1).values
    )
    print(
        f"\ncontrols over the {int(binding.sum())} prefill rows that see more than 16384 positions:"
    )
    for name, ctl in controls.items():
        differs = (ctl != got).any(dim=1)
        assert not bool(differs[~binding].any()), f"control '{name}' differs at or below 16384"
        ctl_top = (
            cons.masked_fill(~ctl[binding], -torch.inf)
            .topk(512, dim=-1)
            .indices.sort(dim=-1)
            .values
        )
        top_differs = (ctl_top != entry_top).any(dim=1)
        print(
            f"  {name}: mask differs on {int(differs.sum())} rows, "
            f"consumer top-512 differs on {int(top_differs.sum())} rows"
        )
        assert int(differs[binding].sum()) > 0, f"control '{name}' was not detected above 16384"


# =============================================================================================
# The DeepSeek-V4.1 core target's first envelope: layer 20 (ratio 1) calls this entry
#
# - prefill, whenever the step's packed context key count S_1 = sum_b kv_len_b exceeds 16384:
#   logits = deepgemm_fp8_fp4_mqa_logits(...) fp32 [tc, S_1] in DeepGEMM's full-width layout (row
#   stride align(S_1 + 256, 256)), row_starts = ks, row_ends = ke (fresh int32 [tc]), next_n 1.
#   Envelope: tc <= 8192 query rows (max_num_tokens), k <= 32 context requests (max_batch_size),
#   kv_len_b <= 16384 (max_seq_len), so S_1 <= 32 x 16384 = 524288 and every row sees <= 16384
#   positions (the pre-filter never binds there; the mask is the row's range minus non-pinned
#   all -inf blocks).
# - decode, in every step (captured ones included) of an engine whose static width
#   max_ctx = max_seq_len + max_draft_tokens exceeds 16384: logits =
#   deepgemm_fp8_fp4_paged_mqa_logits(...) fp32 [G * next_n, max_ctx] (row stride ceil256(max_ctx)),
#   row_starts None, row_ends = compressed_visible_lens[tc:T] (a row-offset view of a
#   [max_num_tokens] buffer), next_n argument 1 (the verify rows carry their own lengths), rows
#   G * next_n <= 32 x 6. The runtime dumps show max_ctx 16398 / 16403 (DSpark engines); an engine
#   without drafts has max_ctx 16384 and never calls it.
# =============================================================================================

ENV_ROWS = 8192
ENV_REQS = 32
ENV_KV = 16384
ENV_W = ENV_REQS * ENV_KV  # 524288
KERNEL_NAME = "_candidate_block_select_kernel"
# max_ctx = (max_seq_len + max_draft_tokens) // 1 at max_seq_len 16384: 0 / 5 / 6 draft tokens, the
# two DSpark widths of the runtime dumps, and a W % 16 == 0 width past them
DECODE_WIDTHS = (16384, 16389, 16390, 16398, 16403, 32768)
CAPTURE_BATCHES = (1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32)  # boot 1..8, the accuracy padded set
_NAN = float("nan")
_ENV_STORE = {}


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _ri(g: torch.Generator, lo: int, hi: int) -> int:
    """Uniform int in [lo, hi]."""
    return lo + int(torch.randint(0, hi - lo + 1, (1,), generator=g))


_DEFAULT_LOGITS_ELEMS = ENV_ROWS * _align(ENV_W + 256, 256)


def _prefill_logits(rows: int, width: int) -> torch.Tensor:
    """A [rows, width] view in DeepGEMM's full-width prefill layout (row stride
    align(width + 256, 256)), carved from one reusable buffer (17.2 GB, grown for a larger step
    and shrunk back by ``_release_big_logits``); its whole span is NaN."""
    stride = _align(width + 256, 256)
    buf = _ENV_STORE.get("logits")
    if buf is None or buf.numel() < rows * stride:
        _ENV_STORE.pop("logits", None)
        del buf
        torch.cuda.empty_cache()
        buf = torch.empty(max(_DEFAULT_LOGITS_ELEMS, rows * stride), device=DEV)
        _ENV_STORE["logits"] = buf
    view = buf[: rows * stride].view(rows, stride)
    view.fill_(_NAN)
    return view[:, :width]


def _release_big_logits() -> None:
    """Drop the prefill buffer if a step grew it past its default size."""
    buf = _ENV_STORE.get("logits")
    if buf is not None and buf.numel() > _DEFAULT_LOGITS_ELEMS:
        _ENV_STORE.pop("logits")
        del buf
        torch.cuda.empty_cache()


def _dead_blocks(g: torch.Generator, rows: int, width: int, rate: float = 1 / 32) -> torch.Tensor:
    """Random whole blocks of BLOCK columns (block j = columns [8j, 8j + 8) of the row's range)."""
    nblk = -(-width // BLOCK)
    return (torch.rand(rows, nblk, generator=g, device=DEV) < rate).repeat_interleave(BLOCK, dim=1)[
        :, :width
    ]


def _packed_step(q_lens, kv_lens, seed: int, with_want: bool = True) -> dict:
    """One packed prefill call as layer 20 makes it.

    Request b holds kv_lens[b] keys at columns [base_b, base_b + kv_lens[b]) (base_b = the keys
    of the requests before it) and its last q_lens[b] positions are this step's query rows; the
    row at position p sees [base_b, base_b + p + 1). Scores in a row's visible range are distinct
    (a random permutation), with whole blocks set to -inf at random: those must be dropped unless
    pinned, which makes every row's logits addressing observable even where the pre-filter does
    not bind. Every other element (unreachable columns, other requests' columns, the stride
    padding) is NaN and must never be read. A request with q_lens[b] = 0 only holds keys (its rows
    are not in the call, as for the decoder-replay tail). ``want`` is ``op.reference`` over each
    request's own rows and keys, independent of ``ks`` / ``ke`` (skipped with ``with_want=False``).
    """
    T, W = sum(q_lens), sum(kv_lens)
    logits = _prefill_logits(T, W)
    want = torch.zeros(T, W, dtype=torch.bool, device=DEV) if with_want else None
    g = torch.Generator(device=DEV).manual_seed(seed)
    ks, ke, bases = [], [], []
    r0 = base = 0
    for b, (q, kv) in enumerate(zip(q_lens, kv_lens)):
        first = kv - q
        if (
            q == 0
        ):  # keys only: the request's keys sit in the gather, none of its rows is in the call
            bases.append(base)
            base += kv
            continue
        if q * kv <= 1 << 29:
            vals = _distinct(q, kv, seed=seed * 64 + b)
            vals.masked_fill_(_dead_blocks(g, q, kv), -torch.inf)
            lens = torch.arange(first + 1, kv + 1, device=DEV)
            if with_want:
                req_want, _ = _prefill_reference(vals, first)
                want[r0 : r0 + q, base : base + kv] = req_want
            logits[r0 : r0 + q, base : base + kv] = vals.masked_fill_(
                torch.arange(kv, device=DEV) >= lens[:, None], _NAN
            )
        else:  # a long request: the same construction in row chunks of at most 2^27 elements
            gq = torch.Generator(device=DEV).manual_seed(seed * 64 + b)
            step = max(1, (1 << 27) // kv)
            for c0 in range(0, q, step):
                c1 = min(q, c0 + step)
                vals = _distinct_from(gq, c1 - c0, kv)
                vals.masked_fill_(_dead_blocks(g, c1 - c0, kv), -torch.inf)
                if with_want:
                    req_want, _ = _prefill_reference(vals, first + c0)
                    want[r0 + c0 : r0 + c1, base : base + kv] = req_want
                    del req_want
                ctx_lens = torch.arange(first + c0 + 1, first + c1 + 1, device=DEV)
                logits[r0 + c0 : r0 + c1, base : base + kv] = vals.masked_fill_(
                    torch.arange(kv, device=DEV) >= ctx_lens[:, None], _NAN
                )
                del vals
            lens = torch.arange(first + 1, kv + 1, device=DEV)
        ks.append(torch.full((q,), base, dtype=torch.int32, device=DEV))
        ke.append((base + lens).to(torch.int32))
        bases.append(base)
        r0 += q
        base += kv
    return {
        "logits": logits,
        "ks": torch.cat(ks),
        "ke": torch.cat(ke),
        "want": want,
        "bases": bases,
        "q_lens": list(q_lens),
        "kv_lens": list(kv_lens),
    }


def _envelope_partition(g: torch.Generator, k: int):
    """Random (q_lens, kv_lens) of a context step inside the envelope that reaches the call:
    k >= 2 requests, T = sum q in [k, 8192], q_b <= kv_b <= 16384, S_1 = sum kv in (16384, 524288].
    Fresh prompts (kv = q), cached prefixes / later chunks and requests at max_seq_len are mixed."""
    mode = _ri(g, 0, 3)
    T = [
        _ri(g, k, max(k, 64 * k)),
        _ri(g, k, ENV_ROWS),
        ENV_ROWS - _ri(g, 0, min(7, ENV_ROWS - k)),
        _ri(g, k, 300),
    ][mode]
    cuts = sorted((torch.randperm(T - 1, generator=g)[: k - 1] + 1).tolist()) if k > 1 else []
    q = [b - a for a, b in zip([0] + cuts, cuts + [T])]
    kv = []
    for qb in q:
        c = _ri(g, 0, 3)
        kv.append([qb, ENV_KV, _ri(g, qb, ENV_KV), _ri(g, max(qb, ENV_KV - 64), ENV_KV)][c])
    while sum(kv) <= ENV_KV:
        kv[_ri(g, 0, k - 1)] = ENV_KV
    return q, kv


def _binding_partition(g: torch.Generator, k: int):
    """Outside the gates' envelope (a request longer than 16384 keys, the long-context shape):
    k in 1..32, T <= 8192, at least one request with more than 16384 keys whose rows bind, and
    S_1 <= 524288."""
    T = _ri(g, k, ENV_ROWS)
    cuts = sorted((torch.randperm(T - 1, generator=g)[: k - 1] + 1).tolist()) if k > 1 else []
    q = [b - a for a, b in zip([0] + cuts, cuts + [T])]
    kv = [_ri(g, qb, ENV_KV) for qb in q]
    for b in range(k):
        if b == 0 or _ri(g, 0, 3) == 0:
            kv[b] = max(kv[b], _ri(g, ENV_KV + 1, 40960))
    while sum(kv) > ENV_W:
        b = _ri(g, 1, k - 1)
        kv[b] = q[b]
    return q, kv


def _call_prefill(step, ks=None, ke=None):
    return op(
        step["logits"],
        step["ks"] if ks is None else ks,
        step["ke"] if ke is None else ke,
        TOPK_BLOCKS,
        BLOCK,
        1,
    )


def _first_bad_row(got, want):
    bad = (got != want).any(dim=1).nonzero()
    return int(bad[0]) if bad.numel() else None


def _fill_decode(
    logits: torch.Tensor, vis: torch.Tensor, next_n: int, seed: int, first_end: Optional[int] = None
) -> torch.Tensor:
    """Fill a decode call's logits [G * next_n, W] and visible lengths in place; return the
    reference mask. Request b's last row sees e_b and its row j sees e_b - (next_n - 1 - j): e_b
    past 16384 (binding), around 16384, anywhere in [1, W], or 0 / 1 (a capture padding row);
    ``first_end`` pins e_0 (e.g. W: the far-end column visible). Columns past a row's length are
    NaN (never read)."""
    rows, W = logits.shape
    G = rows // next_n
    g = torch.Generator().manual_seed(seed)
    ends = []
    for _ in range(G):
        c = _ri(g, 0, 3)
        past = _ri(g, ENV_KV + 1, W) if W > ENV_KV else _ri(g, 1, W)
        ends.append([past, _ri(g, ENV_KV - 40, min(W, ENV_KV + 8)), _ri(g, 1, W), _ri(g, 0, 1)][c])
    if first_end is not None:
        ends[0] = first_end
    e = torch.tensor(ends, dtype=torch.int32, device=DEV).repeat_interleave(next_n)
    j = torch.arange(next_n, dtype=torch.int32, device=DEV).repeat(G)
    lens = (e - (next_n - 1 - j)).clamp(min=0).to(torch.int32)
    vals = _distinct_any(rows, W, seed=seed)
    vals.masked_fill_(
        _dead_blocks(torch.Generator(device=DEV).manual_seed(seed), rows, W), -torch.inf
    )
    reach = torch.arange(W, device=DEV) < lens[:, None]
    want = op.reference(vals, None, lens, TOPK_BLOCKS, BLOCK, 1)
    logits.copy_(vals.masked_fill_(~reach, _NAN))
    vis.copy_(lens)
    return want


def _decode_buffers(rows: int, W: int):
    """Static decode inputs in the paged logits layout (row stride ceil256(W), padding +inf) and
    a [max_num_tokens] visible-length buffer, as the step metadata holds it."""
    buf = torch.full((rows, _align(W, 256)), float("inf"), device=DEV)
    return buf[:, :W], torch.zeros(ENV_ROWS, dtype=torch.int32, device=DEV)


def _jit_keys(fn) -> set:
    """The compiled-variant keys Triton holds for ``fn`` (Triton >= 3.2: ``device_caches``)."""
    dc = getattr(fn, "device_caches", None)
    if dc is not None:
        return {(str(d), k) for d, v in dc.items() for k in v[0].keys()}
    return {(str(d), k) for d, v in fn.cache.items() for k in v.keys()}


def _jit_clear(fn) -> None:
    dc = getattr(fn, "device_caches", None)
    if dc is not None:
        for v in dc.values():
            v[0].clear()
    else:
        for v in fn.cache.values():
            v.clear()


def _profiled(call):
    """Run ``call`` under the profiler; return (result, names of the CUDA kernels it launched)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = call()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if str(e.device_type).endswith("CUDA")]
    return out, names


# ─── (a) the kernel-variant map over the envelope ─────────────────────────────────────────────
#
# One call = one launch of _candidate_block_select_kernel[(rows,)]. Triton specializes on the
# constexprs (HAS_ROW_STARTS: prefill / decode; CHUNK_BLOCKS and num_warps: rows <= 256 or more),
# integer arguments divisible by 16 (num_cols and the mask row stride are both W; the logits row
# strides are multiples of 256 in both layouts) and 16-byte pointer alignment (the decode lengths
# are a view at row tc of the step's buffer). The map below compiles each class once, checks
# every call against the reference, and pins that nothing else in the class compiles.

_PREFILL_CLASSES = {
    "prefill rows<=256 W%16==0": [
        ([1, 1], [16384, 16384]),
        ([100, 156], [16384, 4096]),
        ([8] * 32, [ENV_KV] * 32),
    ],
    "prefill rows<=256 W%16!=0": [
        ([1, 1], [8193, 8192]),
        ([200, 56], [16000, 3001]),
        ([8] * 32, [ENV_KV] * 31 + [16383]),
    ],
    "prefill rows>256 W%16==0": [
        ([257, 1], [16384, 16384]),
        ([4000, 4192], [16384, 8000]),
        ([256] * 32, [ENV_KV] * 32),
    ],
    "prefill rows>256 W%16!=0": [
        ([200, 57], [8193, 8192]),
        ([4096, 4096], [16383, 16384]),
        ([256] * 32, [ENV_KV] * 31 + [16383]),
    ],
}
# (W, next_n, G, tc): tc is the row offset of the lengths view (captured steps: 0)
_DECODE_CLASSES = {
    "decode W%16!=0 lengths 16B-aligned": [
        (16398, 6, 1, 0),
        (16403, 6, 32, 0),
        (16403, 1, 32, 4),
        (16398, 6, 5, 7672),
    ],
    "decode W%16!=0 lengths misaligned": [
        (16398, 6, 1, 1),
        (16403, 6, 32, 2),
        (16403, 1, 7, 3),
        (16398, 6, 24, 573),
    ],
    "decode W%16==0 lengths 16B-aligned": [(32768, 1, 8, 0), (32768, 6, 32, 0), (16400, 6, 3, 8)],
    "decode W%16==0 lengths misaligned": [(32768, 1, 8, 1), (32768, 6, 32, 7674), (16400, 6, 3, 5)],
}


def _decode_call(W, next_n, G, tc, seed):
    logits, vis_buf = _decode_buffers(G * next_n, W)
    vis = vis_buf[tc : tc + G * next_n]
    want = _fill_decode(logits, vis, next_n, seed)
    return logits, vis, want


def test_envelope_kernel_variant_map():
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.candidate_block_select import (  # noqa: E501
        _candidate_block_select_kernel as fn,
    )

    variant_of, calls, bad = {}, 0, []
    _jit_clear(fn)  # the map starts from no variant, so each class compiles its own
    assert not _jit_keys(fn)

    def run(cls, i, desc, call, want):
        nonlocal calls
        before = _jit_keys(fn)
        got, names = _profiled(call)
        new = _jit_keys(fn) - before
        calls += 1
        assert names == [KERNEL_NAME], f"{cls} {desc}: launched {names}"
        if i == 0:
            assert len(new) == 1, f"{cls}: its first call compiled {len(new)} variants"
            variant_of[next(iter(new))] = f"{cls} (first at {desc})"
        else:
            assert not new, f"{cls} {desc}: compiled {len(new)} new variant(s)"
        if not torch.equal(got, want):
            bad.append((cls, desc, _first_bad_row(got, want)))

    for cls, cells in _PREFILL_CLASSES.items():
        for i, (q, kv) in enumerate(cells):
            step = _packed_step(q, kv, seed=1000 + 17 * i + len(cls))
            run(
                cls,
                i,
                f"T={sum(q)} W={sum(kv)} k={len(q)}",
                lambda: _call_prefill(step),
                step["want"],
            )
    for cls, cells in _DECODE_CLASSES.items():
        for i, (W, next_n, G, tc) in enumerate(cells):
            logits, vis, want = _decode_call(W, next_n, G, tc, seed=2000 + 31 * i + len(cls))
            run(
                cls,
                i,
                f"W={W} next_n={next_n} G={G} tc={tc}",
                lambda: op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1),
                want,
            )

    # closure: random envelope calls in the target's forms compile nothing further
    before = _jit_keys(fn)
    g = torch.Generator().manual_seed(4242)
    for n in range(24):
        if n % 2:
            step = _packed_step(*_envelope_partition(g, _ri(g, 2, ENV_REQS)), seed=3000 + n)
            got, names = _profiled(lambda: _call_prefill(step))
            want = step["want"]
        else:
            W, next_n = DECODE_WIDTHS[_ri(g, 0, len(DECODE_WIDTHS) - 1)], (1, 6)[_ri(g, 0, 1)]
            G = CAPTURE_BATCHES[_ri(g, 0, len(CAPTURE_BATCHES) - 1)]
            logits, vis, want = _decode_call(
                W, next_n, G, _ri(g, 0, ENV_ROWS - G * next_n), seed=3000 + n
            )
            got, names = _profiled(lambda: op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1))
        calls += 1
        assert names == [KERNEL_NAME]
        if not torch.equal(got, want):
            bad.append(("closure", n, _first_bad_row(got, want)))
    after = _jit_keys(fn)
    print(
        f"\n  variant map: {len(variant_of)} compiled variant(s) over {calls} profiled calls "
        f"(one {KERNEL_NAME} launch each):"
    )
    for key, cls in sorted(variant_of.items(), key=lambda kv: kv[1]):
        print(f"    {cls}: {key[1][:220]}")
    print(
        f"  24 random envelope calls after the map compiled {len(after - before)} further variant(s); "
        f"{len(bad)} calls differ from the reference"
    )
    assert not bad, bad[:5]
    assert after == before
    assert len(variant_of) == len(_PREFILL_CLASSES) + len(_DECODE_CLASSES)


# ─── (b) >= 200 seeded random partitions of the packed prefill call, plus binding extras ─────


def test_envelope_random_packed_partitions():
    g = torch.Generator().manual_seed(2323)
    stats = {
        "calls": 0,
        "rows": 0,
        "max_T": 0,
        "max_W": 0,
        "min_W": 1 << 30,
        "k": set(),
        "binding_rows": 0,
    }
    ctl_rows = {
        "visible end one short (ke - 1)": 0,
        "start of the neighbouring request": 0,
        "no -inf block drop": 0,
    }
    ctl_checked = 0
    bad = []
    cases = [("envelope", 2 + i % (ENV_REQS - 1)) for i in range(200)] + [
        ("binding", 1 + i % ENV_REQS) for i in range(24)
    ]
    for n, (kind, k) in enumerate(cases):
        q, kv = (_envelope_partition if kind == "envelope" else _binding_partition)(g, k)
        T, W = sum(q), sum(kv)
        if kind == "envelope":
            assert k <= ENV_REQS and T <= ENV_ROWS and max(kv) <= ENV_KV and ENV_KV < W <= ENV_W
        step = _packed_step(q, kv, seed=5000 + n)
        got = _call_prefill(step)
        if not torch.equal(got, step["want"]):
            bad.append((kind, n, q, kv, _first_bad_row(got, step["want"])))
        stats["calls"] += 1
        stats["rows"] += T
        stats["max_T"] = max(stats["max_T"], T)
        stats["max_W"] = max(stats["max_W"], W)
        stats["min_W"] = min(stats["min_W"], W)
        stats["k"].add(k)
        stats["binding_rows"] += int((step["ke"] - step["ks"] > BINDS_ABOVE).sum())
        if kind == "envelope":
            assert int((step["ke"] - step["ks"]).max()) <= BINDS_ABOVE
        if n % 10 == 0 and k > 1:  # deliberately-wrong controls on the same data
            want = step["want"]
            ks, ke = step["ks"], step["ke"]
            req = torch.repeat_interleave(torch.arange(k, device=DEV), torch.tensor(q, device=DEV))
            bases = torch.tensor(step["bases"], dtype=torch.int32, device=DEV)
            col = torch.arange(W, device=DEV)
            ctls = {
                "visible end one short (ke - 1)": _call_prefill(step, ke=ke - 1),
                "start of the neighbouring request": _call_prefill(
                    step, ks=bases[(req - 1) % k].contiguous()
                ),
                "no -inf block drop": (col >= ks[:, None]) & (col < ke[:, None]),
            }
            for name, ctl in ctls.items():
                ctl_rows[name] += int((ctl != want).any(dim=1).sum())
            ctl_checked += T
    print(
        f"\n  {stats['calls']} packed calls (200 inside the envelope, 24 binding): {stats['rows']} rows, "
        f"T <= {stats['max_T']}, S_1 {stats['min_W']}..{stats['max_W']}, k in {sorted(stats['k'])}, "
        f"{stats['binding_rows']} binding rows; {len(bad)} calls differ from the reference"
    )
    print(f"  controls over {ctl_checked} rows of every 10th call:")
    for name, rows in ctl_rows.items():
        print(f"    {name}: {rows} rows differ")
    assert not bad, bad[:3]
    assert stats["k"] == set(range(1, ENV_REQS + 1))
    for name, rows in ctl_rows.items():
        assert rows > 0, f"control '{name}' was not detected"


# ─── the maximal cell's far end (the cells themselves are in op.CELLS) ──────────────────────


def test_envelope_maximal_cell_far_end():
    """8192 x 524288: logits rows 8189..8191 start past element 2^32 (rows 4095.. past 2^31) and
    the mask spans 2^32 bytes. Controls: the same kernel run on the rows a 32-bit (signed or
    unsigned) row offset would read."""
    q, kv = [256] * 32, [ENV_KV] * 32
    step = _packed_step(q, kv, seed=777)
    logits, want = step["logits"], step["want"]
    T, W = logits.shape
    stride = logits.stride(0)
    assert (T - 1) * stride > 2**32 and T * W == 2**32
    torch.cuda.empty_cache()
    junk = torch.ones(T * W, dtype=torch.uint8, device=DEV)
    junk_ptr = junk.data_ptr()
    del junk
    got = _call_prefill(step)
    assert got.data_ptr() == junk_ptr, "the mask did not land on the poisoned block"
    assert torch.equal(got, want) and int(got.view(torch.uint8).max()) <= 1
    del got
    flat = logits.as_strided((T * stride,), (1,), logits.storage_offset())
    for bits in (31, 32):
        wrapped = torch.empty(T, W, device=DEV)
        moved = 0
        for i in range(T):
            off = (i * stride) % (1 << bits)
            moved += off != i * stride
            wrapped[i] = flat[off : off + W]
        ctl = op(wrapped, step["ks"], step["ke"], TOPK_BLOCKS, BLOCK, 1)
        differ = int((ctl != want).any(dim=1).sum())
        print(f"\n  row offset mod 2^{bits}: {moved} rows read elsewhere, {differ} rows differ")
        assert differ == moved > 0
        del wrapped, ctl


# ─── (c) the row axis: every prefix 1..8192 bitwise ────────────────────────────────────────────


@pytest.mark.parametrize(
    "q,kv",
    [([256] * 32, [ENV_KV] * 32), ([4096, 4096], [24576, 20480]), ([8192, 8192], [24576, 20480])],
    ids=["maximal 8192 x 524288", "binding 8192 x 45056", "binding 16384 x 45056"],
)
def test_envelope_rows_prefix_bitwise(q, kv):
    """Each row is one program: the r-row call (rows <= 256 and more take different launch
    configurations) must equal the first r rows of the full call, bitwise, for every r."""
    step = _packed_step(q, kv, seed=sum(kv))
    full = _call_prefill(step)
    assert torch.equal(full, step["want"])
    logits, ks, ke = step["logits"], step["ks"], step["ke"]
    T, W = logits.shape
    bad, reused = [], 0
    for r in range(T, 0, -1):  # largest first: every smaller mask reuses the cached block
        junk = torch.ones(r * W, dtype=torch.uint8, device=DEV)
        ptr = junk.data_ptr()
        del junk
        part = op(logits[:r], ks[:r], ke[:r], TOPK_BLOCKS, BLOCK, 1)
        reused += part.data_ptr() == ptr
        if not torch.equal(part, full[:r]) or int(part.view(torch.uint8).max()) > 1:
            bad.append(r)
        del part
    print(
        f"\n  every r in 1..{T} at W {W}: {len(bad)} prefixes differ; {reused} of {T} masks on a poisoned block"
    )
    assert not bad, bad[:10]


# ─── decode rows (next_n 1 and 6) under CUDA-graph capture at the capture set ────────────────


@pytest.mark.parametrize("W", DECODE_WIDTHS)
def test_envelope_decode_rows_under_capture(W):
    bad, graphs = [], 0
    for next_n in (1, 6):
        for G in CAPTURE_BATCHES:
            rows = G * next_n
            logits, vis_buf = _decode_buffers(rows, W)
            vis = vis_buf[:rows]
            want = _fill_decode(logits, vis, next_n, seed=W + 97 * G + next_n)
            eager = op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1)
            if not torch.equal(eager, want):
                bad.append(("eager", next_n, G, _first_bad_row(eager, want)))
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1)
            graphs += 1
            for rep in (1, 2):
                want = _fill_decode(logits, vis, next_n, seed=W + 97 * G + next_n + 1000 * rep)
                graph.replay()
                torch.cuda.synchronize()
                ref_eager = op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1)
                if not (torch.equal(out, ref_eager) and torch.equal(out, want)):
                    bad.append(("replay", next_n, G, rep, _first_bad_row(out, want)))
            del graph
    print(
        f"\n  W {W}: {graphs} graphs (next_n 1 and 6 x batches {CAPTURE_BATCHES}), 2 replays each; "
        f"{len(bad)} mismatches"
    )
    assert not bad, bad[:5]


def test_envelope_decode_rows_at_row_offsets():
    """Mixed steps: the lengths are the view [tc:T] of the step's buffer at any tc."""
    bad, calls = [], 0
    for W in (16389, 16398, 16403):
        for next_n, G in ((6, 1), (6, 32), (1, 32), (6, 7)):
            rows = G * next_n
            for tc in (1, 2, 3, 4, 5, 573, 7674, ENV_ROWS - rows):
                logits, vis_buf = _decode_buffers(rows, W)
                vis = vis_buf[tc : tc + rows]
                want = _fill_decode(logits, vis, next_n, seed=W + tc + rows)
                vis_buf[:tc] = -7
                vis_buf[tc + rows :] = 1 << 20
                got = op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1)
                same = op(logits, None, vis.clone(), TOPK_BLOCKS, BLOCK, 1)
                calls += 1
                if not (torch.equal(got, want) and torch.equal(got, same)):
                    bad.append((W, next_n, G, tc, _first_bad_row(got, want)))
    print(f"\n  {calls} decode calls at row offsets: {len(bad)} differ")
    assert not bad, bad[:5]


# =============================================================================================
# The long-context envelope: max_num_tokens <= 16384 per rank, max_batch_size <= 256, max_seq_len L
# up to 1,048,576, CUDA graphs at every G in GSET
#
# - prefill: layer 20 runs the step's context rows in row tiles (`_index_row_tiles` in the target):
#   ranges [a, b) whose fp32 logits (b - a) x align(S_1 + 256, 256) stay within 2^28 elements, one
#   range when the whole pass fits, otherwise a multiple of 64 rows per range where that is at
#   least 64. Each tile is one call on a fresh [b - a, S_1] logits tile with ks[a:b] / ke[a:b]
#   (views at row a). A request holds up to Lm keys (Lm = L + 14 with DSpark, the cache manager's
#   length; L without), so rows see up to Lm positions and the pre-filter binds. S_1 = sum_b kv_b
#   is the step's gathered key count: Lm or less for one request, more when several long requests
#   share a step; past S_1 = 4,194,048 a tile holds fewer than 64 rows and ks[a:b] is no longer
#   16-byte aligned.
# - tail (decoder bounded replay): the same call on 128 rows per tail request, tiled the same way,
#   ks / ke at the tail requests' own key ranges inside the step's gather.
# - decode: G * next_n rows, G <= 256 (any G eager, G in GSET captured), next_n 1 or 6 (DSpark
#   verify), so up to 1536 rows; calls above 256 rows take the many-rows launch configuration.
#   The width is md.max_compressed_kv_lens[1] = Lm + max_draft_tokens: L + 19 under the graph
#   metadata with DSpark, L + 14 on eager steps (the dumps show both, e.g. 1048595 / 1048590), and
#   L without DSpark (an engine at L 16384 without drafts never calls). The lengths are the view
#   [tc:T] of the step's [max_num_tokens] (<= 16384) visible-length buffer.
# =============================================================================================

GSET = tuple(range(1, 33)) + (40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256)
LONG_ROWS = 16384
LONG_REQS = 256
LONG_L = (16384, 32768, 65536, 131072, 262144, 1048576)
LONG_LM = LONG_L[-1] + 14  # the longest request with DSpark: 1,048,590 keys
LONG_WIDTHS = (
    tuple(L + 14 for L in LONG_L)  # DSpark, eager steps: Lm
    + tuple(L + 19 for L in LONG_L)  # DSpark, graph metadata: Lm + 5 draft tokens
    + tuple(L for L in LONG_L if L > BINDS_ABOVE)  # DSpark off: L
)
LONG_TILE_ELEMS = 1 << 28
LONG_BUDGET = (1 << 32) - (
    1 << 24
)  # elements of a random step's whole fp32 [T, S_1] logits (< 16 GiB)


def _tiles(rows: int, s_len: int) -> list:
    """The target's prefill row tiles, restated: [a, b) ranges whose logits rows x
    align(s_len + 256, 256) fit 2^28 elements; one range when all rows fit, else ranges of a
    multiple of 64 rows where that is at least 64."""
    width = _align(s_len + 256, 256)
    per = max(1, LONG_TILE_ELEMS // width)
    if per >= rows:
        return [(0, rows)]
    if per >= 64:
        per -= per % 64
    return [(a, min(rows, a + per)) for a in range(0, rows, per)]


def _tile_masks(step):
    """The target's tiled call: one launch per tile on the step's logits rows [a:b] (DeepGEMM's
    row stride, a 16-byte-aligned tile start) with the views ks[a:b] / ke[a:b]. Returns the tile
    masks and the tiles."""
    T, W = step["logits"].shape
    tiles = _tiles(T, W)
    masks = [
        op(step["logits"][a:b], step["ks"][a:b], step["ke"][a:b], TOPK_BLOCKS, BLOCK, 1)
        for a, b in tiles
    ]
    return masks, tiles


def _call_tiled(step):
    """``_tile_masks``, the tile masks joined into the step's [T, S_1] mask."""
    masks, tiles = _tile_masks(step)
    return (masks[0] if len(masks) == 1 else torch.cat(masks)), tiles


def _long_decode_call(W, next_n, G, tc, seed, first_end=None):
    """A decode call in the paged layout (row stride ceil256(W), padding +inf) whose lengths are the
    view [tc:tc + G * next_n] of a [16384] buffer."""
    rows = G * next_n
    buf = torch.full((rows, _align(W, 256)), float("inf"), device=DEV)
    logits, vis_buf = buf[:, :W], torch.zeros(LONG_ROWS, dtype=torch.int32, device=DEV)
    vis = vis_buf[tc : tc + rows]
    want = _fill_decode(logits, vis, next_n, seed, first_end=first_end)
    return logits, vis, vis_buf, want


def _long_partition(g: torch.Generator, k: int, budget: int = LONG_BUDGET):
    """Random (q_lens, kv_lens) of a context step of k requests that reaches the call: T = sum q in
    [k, 16384], q_b <= kv_b <= Lm, S_1 = sum kv > 16384, T * S_1 <= budget (the test builds the
    whole step's logits to take its tiles from). Request lengths mix fresh prompts (kv = q), long
    cached prefixes up to Lm, requests at Lm, and short ones; long requests are cut back at random
    until the step fits the budget."""
    mode = _ri(g, 0, 3)
    T = [
        _ri(g, k, max(k, 64 * k)),
        _ri(g, k, LONG_ROWS),
        LONG_ROWS - _ri(g, 0, min(7, LONG_ROWS - k)),
        _ri(g, k, max(k, 300)),
    ][mode]
    T = min(T, LONG_ROWS)
    cuts = sorted((torch.randperm(T - 1, generator=g)[: k - 1] + 1).tolist()) if k > 1 else []
    q = [b - a for a, b in zip([0] + cuts, cuts + [T])]
    kv = []
    for qb in q:
        c = _ri(g, 0, 4)
        kv.append(
            [
                qb,
                _ri(g, qb, LONG_LM),
                _ri(g, max(qb, LONG_LM - 64), LONG_LM),
                _ri(g, qb, max(qb, 32768)),
                _ri(g, qb, max(qb, 4096)),
            ][c]
        )
    while T * sum(kv) > budget:
        b = _ri(g, 0, k - 1)
        kv[b] = q[b] + (kv[b] - q[b]) // 4
    while sum(kv) <= BINDS_ABOVE:
        b = _ri(g, 0, k - 1)
        kv[b] = min(LONG_LM, kv[b] + BINDS_ABOVE + 1 - sum(kv) + _ri(g, 0, 4096))
    return q, kv


# ─── (a) the kernel-variant map over the long-context envelope ────────────────────────────────
#
# Fourteen classes. Prefill tiles: (rows <= 256 or more) x (W % 16), with ks[a:b] 16-byte aligned,
# plus the tiles of fewer than 64 rows past S_1 = 4,194,048 (rows <= 256, ks misaligned) x (W % 16).
# Decode: (rows <= 256 or more) x (W % 16) x (lengths view 16-byte aligned or not: tc % 4). The
# "decode rows > 256" classes are new at this envelope (decode had at most 32 x 6 rows); the two
# at W % 16 == 0 are not reachable from the target (without DSpark next_n is 1, so rows = G <= 256,
# and every DSpark width L + 14 / L + 19 has W % 16 != 0) but run in the capture sweep below, which
# captures 6G at the DSpark-off widths as well. Each prefill cell is a step whose every tile lands
# in the class (a misaligned cell's first tile starts at row 0, in an aligned class mapped before).

_LONG_PREFILL_CLASSES = {
    "prefill rows<=256 W%16==0": [
        ([1], [32768]),
        ([128, 64], [1048576, 4096]),
        ([1] * 192, [4096] * 192),
    ],
    "prefill rows<=256 W%16!=0": [
        ([1], [LONG_LM]),
        ([128, 128], [16393, 128]),
        ([1] * 192, [4097] + [4096] * 191),
    ],
    "prefill rows>256 W%16==0": [
        ([15872], [16400]),
        ([256, 192], [524288, 2048]),
        ([4] * 256, [512] * 256),
    ],
    "prefill rows>256 W%16!=0": [
        ([10441, 5851], [10441, 6184]),
        ([300], [262163]),
        ([10451, 5927], [10451, 6244]),
    ],
    "prefill tiles<64 rows W%16==0 ks misaligned": [
        ([32] * 4, [1048576] * 4),
        ([16] * 5, [1048576] * 5),
        ([8] * 8, [1048576] * 8),
    ],
    "prefill tiles<64 rows W%16!=0 ks misaligned": [
        ([32] * 4, [LONG_LM] * 4),
        ([16] * 5, [LONG_LM] * 5),
        ([8] * 8, [LONG_LM - 1] * 8),
    ],
}
# (W, next_n, G, tc)
_LONG_DECODE_CLASSES = {
    "decode rows<=256 W%16!=0 lengths 16B-aligned": [
        (1048595, 6, 1, 0),
        (32787, 6, 42, 0),
        (65550, 1, 256, 4),
    ],
    "decode rows<=256 W%16!=0 lengths misaligned": [
        (16403, 6, 5, 1),
        (131091, 1, 200, 2),
        (1048590, 6, 3, 16365),
    ],
    "decode rows<=256 W%16==0 lengths 16B-aligned": [
        (32768, 1, 256, 0),
        (1048576, 1, 1, 0),
        (262144, 1, 17, 8192),
    ],
    "decode rows<=256 W%16==0 lengths misaligned": [
        (65536, 1, 3, 1),
        (1048576, 1, 256, 16127),
        (131072, 1, 40, 5),
    ],
    "decode rows>256 W%16!=0 lengths 16B-aligned": [
        (16403, 6, 43, 0),
        (1048595, 6, 256, 0),
        (32782, 6, 100, 8),
    ],
    "decode rows>256 W%16!=0 lengths misaligned": [
        (32787, 6, 48, 1),
        (1048590, 6, 256, 14847),
        (262163, 6, 200, 2),
    ],
    "decode rows>256 W%16==0 lengths 16B-aligned": [
        (32768, 6, 43, 0),
        (1048576, 6, 256, 0),
        (131072, 6, 64, 4096),
    ],
    "decode rows>256 W%16==0 lengths misaligned": [
        (65536, 6, 48, 3),
        (1048576, 6, 256, 14845),
        (262144, 6, 100, 7),
    ],
}


def test_long_context_kernel_variant_map():
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.candidate_block_select import (  # noqa: E501
        _candidate_block_select_kernel as fn,
    )

    variant_of, calls, bad = {}, 0, []
    _jit_clear(fn)
    assert not _jit_keys(fn)

    def run(cls, i, desc, call, want, launches=1):
        nonlocal calls
        before = _jit_keys(fn)
        got, names = _profiled(call)  # the tile masks are joined outside the profiled region
        got = torch.cat(got) if isinstance(got, list) else got
        new = _jit_keys(fn) - before
        calls += launches
        assert names == [KERNEL_NAME] * launches, f"{cls} {desc}: launched {names}"
        if i == 0:
            assert len(new) == 1, f"{cls}: its first cell compiled {len(new)} variants"
            variant_of[next(iter(new))] = f"{cls} (first at {desc})"
        else:
            assert not new, f"{cls} {desc}: compiled {len(new)} new variant(s)"
        if not torch.equal(got, want):
            bad.append((cls, desc, _first_bad_row(got, want)))

    for cls, cells in _LONG_PREFILL_CLASSES.items():
        for i, (q, kv) in enumerate(cells):
            step = _packed_step(q, kv, seed=7000 + 16 * i + len(cls))
            tiles = _tiles(sum(q), sum(kv))
            run(
                cls,
                i,
                f"T={sum(q)} W={sum(kv)} k={len(q)} tiles={[b - a for a, b in tiles]}",
                lambda: _tile_masks(step)[0],
                step["want"],
                launches=len(tiles),
            )
            del step
    for cls, cells in _LONG_DECODE_CLASSES.items():
        for i, (W, next_n, G, tc) in enumerate(cells):
            logits, vis, _, want = _long_decode_call(
                W, next_n, G, tc, seed=8000 + 31 * i + len(cls)
            )
            run(
                cls,
                i,
                f"W={W} next_n={next_n} G={G} tc={tc}",
                lambda: op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1),
                want,
            )
            del logits, vis, want

    # closure: random calls in the target's forms over the new envelope compile nothing further
    before = _jit_keys(fn)
    g = torch.Generator().manual_seed(4343)
    for n in range(32):
        if n % 2:
            q, kv = _long_partition(g, _ri(g, 1, LONG_REQS), budget=1 << 30)
            step = _packed_step(q, kv, seed=9000 + 8 * n)
            (masks, tiles), names = _profiled(lambda: _tile_masks(step))
            got, want, launches = torch.cat(masks), step["want"], len(tiles)
        else:
            W, next_n = LONG_WIDTHS[_ri(g, 0, len(LONG_WIDTHS) - 1)], (1, 6)[_ri(g, 0, 1)]
            G = _ri(g, 1, LONG_REQS)
            logits, vis, _, want = _long_decode_call(
                W, next_n, G, _ri(g, 0, LONG_ROWS - G * next_n), seed=9000 + n
            )
            got, names = _profiled(lambda: op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1))
            launches = 1
        calls += launches
        assert names == [KERNEL_NAME] * launches
        if not torch.equal(got, want):
            bad.append(("closure", n, _first_bad_row(got, want)))
    after = _jit_keys(fn)
    print(
        f"\n  long-context variant map: {len(variant_of)} compiled variant(s) over {calls} profiled launches "
        f"(one {KERNEL_NAME} launch per call / tile):"
    )
    for key, cls in sorted(variant_of.items(), key=lambda kv: kv[1]):
        print(f"    {cls}: {key[1][:220]}")
    print(
        f"  32 random calls after the map compiled {len(after - before)} further variant(s); "
        f"{len(bad)} calls differ from the reference"
    )
    assert not bad, bad[:5]
    assert after == before
    assert len(variant_of) == len(_LONG_PREFILL_CLASSES) + len(_LONG_DECODE_CLASSES)


# ─── (b) >= 200 seeded random packed prefill steps over the new envelope ──────────


def test_long_context_random_packed_partitions():
    """Each step is called as the target calls it (one call per row tile) and, as a superset, in
    one untiled call; both are compared with the reference on every row."""
    g = torch.Generator().manual_seed(2424)
    fixed_k = [1, 2, 3, 128, 255, 256]
    ks_list = fixed_k + [_ri(g, 1, LONG_REQS) for _ in range(200 - len(fixed_k))]
    stats = {
        "rows": 0,
        "max_T": 0,
        "max_W": 0,
        "min_W": 1 << 40,
        "k": set(),
        "binding_rows": 0,
        "max_vis": 0,
        "vis_over_512k": 0,
        "T_16384": 0,
        "tiles": 0,
        "multi_tile": 0,
        "min_tile": 1 << 30,
        "max_tile": 0,
        "small_tiles": 0,
    }
    ctl_rows = {
        "visible end one short (ke - 1)": 0,
        "start of the neighbouring request": 0,
        "no -inf block drop": 0,
    }
    ctl_checked, bad = 0, []
    for n, k in enumerate(ks_list):
        q, kv = _long_partition(g, k)
        T, W = sum(q), sum(kv)
        assert (
            len(q) == k <= LONG_REQS
            and k <= T <= LONG_ROWS
            and max(kv) <= LONG_LM
            and BINDS_ABOVE < W
        )
        assert T * W <= LONG_BUDGET
        step = _packed_step(q, kv, seed=1_000_000 + 8 * n)
        tiled, tiles = _call_tiled(step)
        whole = _call_prefill(step)
        if not (torch.equal(tiled, step["want"]) and torch.equal(whole, step["want"])):
            bad.append(
                (
                    n,
                    k,
                    T,
                    W,
                    _first_bad_row(tiled, step["want"]),
                    _first_bad_row(whole, step["want"]),
                )
            )
        del tiled, whole
        vis = step["ke"] - step["ks"]
        stats["rows"] += T
        stats["max_T"] = max(stats["max_T"], T)
        stats["max_W"] = max(stats["max_W"], W)
        stats["min_W"] = min(stats["min_W"], W)
        stats["k"].add(k)
        stats["binding_rows"] += int((vis > BINDS_ABOVE).sum())
        stats["max_vis"] = max(stats["max_vis"], int(vis.max()))
        stats["vis_over_512k"] += int(int(vis.max()) > 1 << 19)
        stats["T_16384"] += int(T == LONG_ROWS)
        stats["tiles"] += len(tiles)
        stats["multi_tile"] += int(len(tiles) > 1)
        stats["min_tile"] = min(stats["min_tile"], min(b - a for a, b in tiles))
        stats["max_tile"] = max(stats["max_tile"], max(b - a for a, b in tiles))
        stats["small_tiles"] += int(len(tiles) > 1 and tiles[0][1] < 64)
        if n % 10 == 0 and k > 1:  # deliberately-wrong controls on the same data
            want, ks, ke = step["want"], step["ks"], step["ke"]
            req = torch.repeat_interleave(torch.arange(k, device=DEV), torch.tensor(q, device=DEV))
            bases = torch.tensor(step["bases"], dtype=torch.int32, device=DEV)
            col = torch.arange(W, device=DEV)
            ctls = {
                "visible end one short (ke - 1)": _call_prefill(step, ke=ke - 1),
                "start of the neighbouring request": _call_prefill(
                    step, ks=bases[(req - 1) % k].contiguous()
                ),
                "no -inf block drop": (col >= ks[:, None]) & (col < ke[:, None]),
            }
            for name, ctl in ctls.items():
                ctl_rows[name] += int((ctl != want).any(dim=1).sum())
            ctl_checked += T
            del ctls
        del step
    print(
        f"\n  {len(ks_list)} packed steps: {stats['rows']} rows, T <= {stats['max_T']} ({stats['T_16384']} at 16384), "
        f"S_1 {stats['min_W']}..{stats['max_W']}, k {min(stats['k'])}..{max(stats['k'])} ({len(stats['k'])} distinct), "
        f"{stats['binding_rows']} binding rows, longest row {stats['max_vis']} positions "
        f"({stats['vis_over_512k']} steps with a row past 524288)"
    )
    print(
        f"  {stats['tiles']} tiles ({stats['multi_tile']} steps tiled, {stats['small_tiles']} with tiles under 64 "
        "rows), "
        f"tile rows {stats['min_tile']}..{stats['max_tile']}; {len(bad)} steps differ from the reference "
        f"(tiled or whole)"
    )
    print(f"  controls over {ctl_checked} rows of every 10th step:")
    for name, rows in ctl_rows.items():
        print(f"    {name}: {rows} rows differ")
    assert not bad, bad[:3]
    assert set(fixed_k) <= stats["k"]
    for name, rows in ctl_rows.items():
        assert rows > 0, f"control '{name}' was not detected"


# ─── the tail's premise: a row's mask depends only on its own scores over [ks, ke) ────────────


def _deepgemm_layout(rows: torch.Tensor) -> torch.Tensor:
    """A copy of ``rows`` [n, W] in DeepGEMM's prefill layout (row stride align(W + 256, 256)),
    the layout (and the compiled variant) the target's calls use."""
    n, w = rows.shape
    out = torch.full((n, _align(w + 256, 256)), _NAN, device=DEV)[:, :w]
    out.copy_(rows)
    return out


def test_long_context_prefill_cells_row_independence_and_key_offset():
    """Decoder replay scores the tail rows' masks against the step's gathered keys, and the target
    tiles rows: so a row must get the same mask alone, at any position inside a larger call, and
    with its request's keys at any offset of the gather. Two tail requests of 128 rows with other
    requests' keys around them (S_1 1,048,219: one 256-row tile)."""
    q, kv = [0, 128, 0, 128], [300001, 400000, 48218, 300000]
    step = _packed_step(q, kv, seed=5_000_000)
    T, W = step["logits"].shape
    assert W == 1048219 and _tiles(T, W) == [(0, T)]
    logits = _deepgemm_layout(
        step["logits"]
    )  # our own copy: the next _packed_step reuses the buffer
    ks, ke, want, bases = step["ks"], step["ke"], step["want"], step["bases"]
    full = op(logits, ks, ke, TOPK_BLOCKS, BLOCK, 1)
    assert torch.equal(full, want)
    assert bool((ke - ks > BINDS_ABOVE).all())
    # (1) every row alone
    alone_bad = [
        i
        for i in range(T)
        if not torch.equal(
            op(logits[i : i + 1], ks[i : i + 1], ke[i : i + 1], TOPK_BLOCKS, BLOCK, 1)[0],
            full[i],
        )
    ]
    # (2) the rows in reverse order
    perm = torch.arange(T - 1, -1, -1, device=DEV)
    rev = op(
        _deepgemm_layout(logits[perm]),
        ks[perm].contiguous(),
        ke[perm].contiguous(),
        TOPK_BLOCKS,
        BLOCK,
        1,
    )
    rev_ok = torch.equal(rev, full[perm])
    del rev
    # (3) inside a larger call: 100 rows of another step before ours and 60 after (416 rows, the
    #     many-rows launch configuration where ours alone took the few-rows one)
    other = _packed_step([160], [W], seed=5_000_001)
    big = _deepgemm_layout(torch.cat([other["logits"][:100], logits, other["logits"][100:]]))
    big_ks = torch.cat([other["ks"][:100], ks, other["ks"][100:]])
    big_ke = torch.cat([other["ke"][:100], ke, other["ke"][100:]])
    in_big = op(big, big_ks, big_ke, TOPK_BLOCKS, BLOCK, 1)
    big_ok = (
        torch.equal(in_big[100 : 100 + T], full)
        and torch.equal(in_big[:100], other["want"][:100])
        and torch.equal(in_big[100 + T :], other["want"][100:])
    )
    del other, big, in_big
    # (4) each tail request's keys moved to other offsets of a gather: the mask moves with them
    shift_bad, shifts = [], 0
    r0 = 0
    for b, (qb, kvb) in enumerate(zip(q, kv)):
        if qb == 0:
            continue
        rows, base = slice(r0, r0 + qb), bases[b]
        mine = logits[rows, base : base + kvb]
        for d in (0, 1, 7, 8, 13, 4097, 524289, base + 3):
            buf = torch.full((qb, _align(d + kvb + 5 + 256, 256)), _NAN, device=DEV)
            moved = buf[:, : d + kvb + 5]
            moved[:, d : d + kvb] = mine
            got = op(
                moved,
                (ks[rows] - base + d).contiguous(),
                (ke[rows] - base + d).contiguous(),
                TOPK_BLOCKS,
                BLOCK,
                1,
            )
            shifts += 1
            if not (
                torch.equal(got[:, d : d + kvb], full[rows, base : base + kvb])
                and not got[:, :d].any()
                and not got[:, d + kvb :].any()
            ):
                shift_bad.append((b, d))
            del buf, moved, got
        r0 += qb
    print(
        f"\n  {T} tail rows at S_1 {W}: {len(alone_bad)} rows differ alone; reversed order equal: {rev_ok}; "
        f"inside a 416-row call equal: {big_ok}; {len(shift_bad)} of {shifts} key-offset moves differ"
    )
    assert not alone_bad and rev_ok and big_ok and not shift_bad, (alone_bad[:5], shift_bad[:5])


# ─── the row axis at the widest tiles: every r in 1..tile rows, bitwise ──────────────────────


@pytest.mark.parametrize(
    "q,kv",
    [([256], [1048219]), ([192], [LONG_LM]), ([0, 256], [48219, 1000000])],
    ids=[
        "256-row tile at S_1 1048219",
        "192-row tile at Lm 1048590",
        "256 tail rows past other keys",
    ],
)
def test_long_context_maximal_tile_rows_prefix_bitwise(q, kv):
    """The widest tiles hold 256 rows (S_1 up to 1,048,320) or 192 (up to Lm); every row binds.
    Every r-row call equals the first r rows of the whole tile, bitwise, on a poisoned block."""
    step = _packed_step(q, kv, seed=6_000_000 + sum(kv) % 1000)
    logits, ks, ke = step["logits"], step["ks"], step["ke"]
    T, W = logits.shape
    assert _tiles(T, W) == [(0, T)] and bool((ke - ks > BINDS_ABOVE).all())
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    full = _call_prefill(step)
    stop.record()
    torch.cuda.synchronize()
    assert torch.equal(full, step["want"])
    bad, reused = [], 0
    for r in range(T, 0, -1):
        junk = torch.ones(r * W, dtype=torch.uint8, device=DEV)
        ptr = junk.data_ptr()
        del junk
        part = op(logits[:r], ks[:r], ke[:r], TOPK_BLOCKS, BLOCK, 1)
        reused += part.data_ptr() == ptr
        if not torch.equal(part, full[:r]) or int(part.view(torch.uint8).max()) > 1:
            bad.append(r)
        del part
    print(
        f"\n  {T} x {W} (one call {start.elapsed_time(stop):.2f} ms): every r in 1..{T}: {len(bad)} prefixes differ; "
        f"{reused} of {T} masks on a poisoned block"
    )
    assert not bad, bad[:10]


# ─── decode: capture at every G in GSET, rows G and 6G, at every ratio-1 width ────────────────


@pytest.mark.parametrize("W", LONG_WIDTHS)
def test_long_context_decode_capture_at_gset(W):
    """Each (next_n, G) is captured after an eager call at its shape, then replayed twice on new
    scores and lengths. Request 0 sees the whole width (column W - 1) at the eager call and the
    second replay, and a shorter binding length at the first, so every replay's expected mask
    differs from the previous step's and a stale replay is seen."""
    bad, graphs, stale_min = [], 0, None
    mid = BINDS_ABOVE + 1 + (W - BINDS_ABOVE - 1) // 2
    for next_n in (1, 6):
        for G in GSET:
            seed = 10 * W + 977 * G + next_n
            logits, vis, _, want = _long_decode_call(W, next_n, G, 0, seed, first_end=W)
            eager = op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1)
            if not torch.equal(eager, want):
                bad.append(("eager", next_n, G, _first_bad_row(eager, want)))
            del eager
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1)
            graphs += 1
            prev = want
            for rep, end in ((1, mid), (2, W)):
                want = _fill_decode(logits, vis, next_n, seed=seed + 100_000 * rep, first_end=end)
                changed = int((want != prev).any(dim=1).sum())
                stale_min = changed if stale_min is None else min(stale_min, changed)
                graph.replay()
                torch.cuda.synchronize()
                ref_eager = op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1)
                if not (torch.equal(out, ref_eager) and torch.equal(out, want)):
                    bad.append(("replay", next_n, G, rep, _first_bad_row(out, want)))
                del ref_eager
                prev = want
            del graph, out, logits, vis, want, prev
    print(
        f"\n  W {W}: {graphs} graphs (next_n 1 and 6 x {len(GSET)} batches, rows up to {6 * GSET[-1]}), "
        f"2 replays each; {len(bad)} mismatches; every replay's mask differs from the previous step's "
        f"on >= {stale_min} row(s)"
    )
    assert not bad, bad[:5]
    assert stale_min >= 1


# ─── decode rows on the row axis: every r in 1..1536, bitwise ─────────────────────────────────


@pytest.mark.parametrize("W", (32782, 1048595))
def test_long_context_decode_rows_prefix_bitwise(W):
    """Eager steps hold any G <= 256 at next_n 1 or 6, so a decode call has any row count up to
    1536. The 1536-row call (256 DSpark requests, request 0 at the far end) against the reference,
    then every r in 1..1536 bitwise equal to its first r rows, across the 256-row launch switch."""
    rows = 6 * LONG_REQS
    logits, vis, _, want = _long_decode_call(W, 6, LONG_REQS, 0, seed=W + 11, first_end=W)
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    full = op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1)
    torch.cuda.synchronize()
    start.record()
    for _ in range(3):
        op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1)
    stop.record()
    torch.cuda.synchronize()
    assert torch.equal(full, want)
    print(
        f"\n  {rows} x {W}: exact; {int((vis > BINDS_ABOVE).sum())} binding rows; "
        f"eager call {start.elapsed_time(stop) / 3:.2f} ms"
    )
    bad, reused = [], 0
    for r in range(rows, 0, -1):
        junk = torch.ones(r * W, dtype=torch.uint8, device=DEV)
        ptr = junk.data_ptr()
        del junk
        part = op(logits[:r], None, vis[:r], TOPK_BLOCKS, BLOCK, 1)
        reused += part.data_ptr() == ptr
        if not torch.equal(part, full[:r]) or int(part.view(torch.uint8).max()) > 1:
            bad.append(r)
        del part
    print(
        f"  every r in 1..{rows} at W {W}: {len(bad)} prefixes differ; {reused} of {rows} masks on a poisoned block"
    )
    assert not bad, bad[:10]


# ─── decode in mixed (eager) steps: the lengths view at row offsets of the [16384] buffer ─────


def test_long_context_decode_rows_at_row_offsets():
    bad, calls = [], 0
    for W in (16398, 32782, 1048590, 1048576):
        for next_n, G in ((6, 1), (1, 7), (6, 43), (1, 256), (6, 256)):
            rows = G * next_n
            for tc in (1, 2, 3, 5, 8193, LONG_ROWS - rows):
                logits, vis, vis_buf, want = _long_decode_call(
                    W, next_n, G, tc, seed=W + tc + rows, first_end=W
                )
                vis_buf[:tc] = -7
                vis_buf[tc + rows :] = 1 << 22
                got = op(logits, None, vis, TOPK_BLOCKS, BLOCK, 1)
                same = op(logits, None, vis.clone(), TOPK_BLOCKS, BLOCK, 1)
                calls += 1
                if not (torch.equal(got, want) and torch.equal(got, same)):
                    bad.append((W, next_n, G, tc, _first_bad_row(got, want)))
                del logits, vis, vis_buf, want, got, same
    print(f"\n  {calls} decode calls at row offsets of the 16384 buffer: {len(bad)} differ")
    assert not bad, bad[:5]
