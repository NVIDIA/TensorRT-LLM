# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the convert_req_index_to_global entry.

The cells are the Mistral Large 4 FP8 target's at its 9216 envelope, the
Mistral Large 4 NVFP4 targets' at their 32768 envelope (the tep8, tp8 and dep8 ones
on the fp8_ds_mla pool with page ids from a 147456-block pool), the NVFP4 tep4,
tp4 and dep4 deployments' at max_seq_len 1048576 with chunked prefill, and those
three targets' and the tep8 deployment's on the 63-owner fp8_ds_mla pool at that
envelope (dep4's under attention DP, tep8's from the 147456-block pool). Every
value is integer address arithmetic, so each output is gated bitwise against
the entry's reference.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.mistral_large4.ops.attention.convert_req_index_to_global import (  # noqa: E501
    _ML4_1M_DEP4_DS,
    _ML4_1M_DS,
    _ML4_1M_TEP8_DS,
    _ML4_32K_DS,
    _ML4_32K_DS_8GPU,
    _ML4_DS_STEP,
    _ML4_OWNERS,
    _ML4_STEP,
)
from tensorrt_llm._torch.modeling_v2.models.mistral_large4.ops.attention.convert_req_index_to_global import (  # noqa: E501
    convert_req_index_to_global as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "convert_req_index_to_global requires a CUDA device"

#: Fills output memory the op is expected to leave unwritten.
_SENTINEL = 0x7EADBEEF

#: The dep4 envelope's max_seq_len: the boot captures each decode graph on a longest dummy of
#: one key less, and the engine's table is ceil((max_seq_len + 1) / 32) columns, padded to 4.
_DEP4_MAX_SEQ_LEN = 1048576


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Exact: this is integer address arithmetic, and an off-by-one row is a
    different token's KV."""
    assert out.dtype == ref.dtype == torch.int32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert torch.equal(out, ref), (
        f"{int((out != ref).sum())} of {out.numel()} pool rows differ from the reference"
    )


def _layer_ids(spec) -> tuple:
    if spec.get("layer_ids") == "all":
        return tuple(range(spec["pool_layers"]))
    if spec.get("layer_ids") == "owner_rows_0_62":
        # Every owner row of the 63-owner fp8_ds_mla page.
        assert spec["pool_layers"] == len(_ML4_OWNERS) == 63, spec["pool_layers"]
        return tuple(range(63))
    return tuple(spec["layer_ids"])


def _table(spec, g: torch.Generator) -> torch.Tensor:
    """The block table, and in it each request's pages.

    It is the cache manager's: `table_rows` x `table_width`, pages drawn from a
    `pool_blocks` pool (its top ids with `far_end`), and every entry no request
    uses holds a stale page id, so reading the wrong row or column shows.

    A step of several 1M contexts names more pages than the pool has (no rank's
    pool holds them at once; the cell bounds what the table can name, not what
    the scheduler admits). Its pages then run through fresh permutations of the
    whole pool, so the top ids are in use and ids repeat across requests.

    With `first_seq`, request `s` holds table row `first_seq + s`, as a mixed
    step's generation requests follow its context requests. With `pad`, the
    last `pad` requests are the CUDA-graph padding dummy, one request repeated:
    each keeps a row of its own, and every one of those rows holds the first's
    pages.
    """
    bs = spec["block_size"]
    need = [math.ceil(n / bs) for n in spec["kv_lens"]]
    first = spec.get("first_seq", 0)
    rows = spec["table_rows"]
    assert first + len(need) <= rows, "every request needs a row of the table"
    width = spec["table_width"]
    pool = spec["pool_blocks"]
    used = [min(n, width) for n in need]
    if sum(used) > pool:
        cycles = math.ceil(sum(used) / pool)
        pages = torch.cat([torch.randperm(pool, generator=g) for _ in range(cycles)])
    elif spec.get("far_end"):
        pages = pool - 1 - torch.randperm(sum(used), generator=g)
    else:
        pages = torch.randperm(pool, generator=g)[: sum(used)]
    table = torch.randint(0, pool, (rows, width), generator=g)
    start = 0
    for s, n in enumerate(used):
        table[first + s, :n] = pages[start : start + n]
        start += n
    dummies = spec.get("pad", 0)
    if dummies:
        assert spec["kv_lens"][-dummies:] == [1] * dummies, "the padding dummy holds one key"
        dummy = first + len(need) - dummies
        table[dummy + 1 : dummy + dummies] = table[dummy]
    return table.to(torch.int32)


def _positions(
    ends: torch.Tensor,
    topk: int,
    order: str,
    seed: int,
    last: bool = False,
    starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row `i`: `min(ends[i], topk)` distinct positions of `[0, ends[i])`, then -1.

    The shape the indexer's top-k hands over: in no order, or in the ascending
    order of the engine's dense causal index when the indexer is skipped
    (`order="causal"`). With `last`, each row's newest position `ends[i] - 1`
    is among its selections (first), so a context that fills the block table
    reads its last column. `order="chunk"` draws by `_chunk`, which also reads
    each row's cached prefix, `starts`.

    A random draw is one key per position of every row; past 2**30 keys (a
    32768-row step of 1M contexts would take 2**35) the rows are drawn by
    `_strata` instead.
    """
    if order == "chunk":
        return _chunk(ends, starts, topk, seed, last)
    width = int(ends.max())
    if ends.numel() * width > 2**30:
        assert order == "random", "only the random order is drawn by strata"
        return _strata(ends, topk, seed, last)
    col = torch.arange(width, device="cuda").unsqueeze(0)
    if order == "causal":
        keys = col.expand(ends.numel(), width).float()
    else:
        g = torch.Generator(device="cuda").manual_seed(seed)
        keys = torch.rand(ends.numel(), width, device="cuda", generator=g)
    if last:
        keys.scatter_(1, (ends - 1).unsqueeze(1), -1.0)
    keys = keys.masked_fill(col >= ends.unsqueeze(1), float("inf"))
    k = min(topk, width)
    pick = keys.topk(k, dim=1, largest=False).indices
    out = torch.full((ends.numel(), topk), -1, dtype=torch.int32, device="cuda")
    out[:, :k] = torch.where(col[:, :k] < ends.unsqueeze(1), pick, -1).to(torch.int32)
    return out


def _strata(ends: torch.Tensor, topk: int, seed: int, last: bool) -> torch.Tensor:
    """`_positions` in random order without a key per position.

    A row with `ends[i] <= topk` holds every position of `[0, ends[i])`, then
    -1, as `_positions` gives it. A longer row cuts `[0, ends[i])` into `topk`
    strata `[s * ends[i] // topk, (s + 1) * ends[i] // topk)`, each at least one
    position wide, and takes one position at random from each: `topk` distinct
    positions over the whole context. Columns are shuffled, the -1 padding
    last; with `last`, the last stratum gives `ends[i] - 1`, placed first.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    end = ends.long().unsqueeze(1)
    k = torch.arange(topk, device="cuda").unsqueeze(0)
    lo, hi = k * end // topk, (k + 1) * end // topk
    u = torch.rand(ends.numel(), topk, device="cuda", generator=g)
    pick = torch.minimum(lo + (u * (hi - lo)).long(), hi - 1)
    wide = end > topk
    if last:
        pick[:, -1] = end[:, 0] - 1
    valid = k < end
    pos = torch.where(valid, torch.where(wide, pick, k), -1)
    keys = torch.rand(ends.numel(), topk, device="cuda", generator=g)
    keys = keys.masked_fill(~valid, float("inf"))
    if last:
        keys.scatter_(1, torch.where(wide, topk - 1, end - 1), -1.0)
    return pos.gather(1, keys.argsort(dim=1)).to(torch.int32)


def _chunk(ends: torch.Tensor, starts: torch.Tensor, topk: int, seed: int, last: bool):
    """`_positions` for `order="chunk"`: a chunk row's selections on both sides of
    its request's cached prefix.

    Row `i` sees `[0, ends[i])`, and `[starts[i], ends[i])` is its chunk's own rows
    up to it (`starts[i]` is the request's cached prefix). It takes
    `min(ends[i] - starts[i], max(topk // 2, topk - starts[i]))` positions from the
    chunk and `min(starts[i], topk - that)` from `[0, starts[i])`. Each part is cut
    into that many strata, with one position drawn from each, which gives
    `min(ends[i], topk)` distinct positions. Columns are shuffled and the -1
    padding comes last; with `last`, the chunk part's last stratum gives
    `ends[i] - 1`, placed first.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    end, start = ends.long().unsqueeze(1), starts.long().unsqueeze(1)
    new = torch.minimum(end - start, torch.clamp(topk - start, min=topk // 2))
    old = torch.minimum(start, topk - new)
    k = torch.arange(topk, device="cuda").unsqueeze(0)
    in_new = k < new
    # Stratum `s` of `count` over `[base, base + span)`; a part is never narrower
    # than its count, so every stratum is at least one position wide.
    s = torch.where(in_new, k, k - new)
    count = torch.where(in_new, new, old).clamp(min=1)
    base = torch.where(in_new, start, 0)
    span = torch.where(in_new, end - start, start)
    lo, hi = base + s * span // count, base + (s + 1) * span // count
    u = torch.rand(ends.numel(), topk, device="cuda", generator=g)
    pick = torch.minimum(lo + (u * (hi - lo)).long(), hi - 1)
    newest = k == new - 1
    if last:
        pick = torch.where(newest, end - 1, pick)
    valid = k < new + old
    pos = torch.where(valid, pick, -1)
    keys = torch.rand(ends.numel(), topk, device="cuda", generator=g)
    keys = keys.masked_fill(~valid, float("inf"))
    if last:
        keys = keys.masked_fill(newest, -1.0)
    return pos.gather(1, keys.argsort(dim=1)).to(torch.int32)


def _build(spec, seed: int):
    """`(req_id, block_table, token_indices, block_size, topk, stride_factor)`.

    Request `s` contributes `seq_rows[s]` query rows; row `j` of it sees
    positions `[0, kv_lens[s] - seq_rows[s] + j + 1)`, and `req_id` names the
    request's row of the table (`first_seq + s`), as the engine's per-token
    request index does.
    """
    assert len(spec["seq_rows"]) == len(spec["kv_lens"]), "one seq_rows entry per request"
    first = spec.get("first_seq", 0)
    req, ends, starts = [], [], []
    for s, (r_count, kv) in enumerate(zip(spec["seq_rows"], spec["kv_lens"])):
        assert 1 <= r_count <= kv, f"request {s}: {r_count} rows over {kv} keys"
        req += [first + s] * r_count
        ends += [kv - r_count + j + 1 for j in range(r_count)]
        starts += [kv - r_count] * r_count
    g = torch.Generator().manual_seed(seed)
    bs, topk = spec["block_size"], spec["topk"]
    return (
        torch.tensor(req, dtype=torch.int32, device="cuda"),
        _table(spec, g).cuda(),
        _positions(
            torch.tensor(ends, device="cuda"),
            topk,
            spec.get("order", "random"),
            seed,
            spec.get("last_position", False),
            starts=torch.tensor(starts, device="cuda"),
        ),
        bs,
        topk,
        spec["pool_layers"] * bs,
    )


def _check_padding(out: torch.Tensor, tok: torch.Tensor) -> None:
    # The padding has to survive as -1, not as a row computed from -1.
    assert bool((out[tok < 0] == -1).all())


def _check_reach(spec, outs) -> None:
    """A cell drawn from a cache manager's pool names rows past 2**24 + 1, where an
    index carried through fp32 stops being exact; the note's fp32 row rests on it."""
    if "pool_blocks" in spec:
        top = max(int(o.max()) for o in outs)
        assert top > 2**24 + 1, f"the cell's highest pool row is {top}, not past 2**24 + 1"


def _check_newest(spec, tok: torch.Tensor, out: torch.Tensor) -> None:
    """With `last_position`, the longest context's newest position is selected and
    resolves to a pool row: the cell reads the last table column that context reaches,
    which a `why` naming the last column rests on."""
    if spec.get("last_position"):
        newest = tok == max(spec["kv_lens"]) - 1
        assert bool(newest.any()), "the longest context's newest position is not selected"
        assert bool((out[newest] >= 0).all()), "the newest position resolved to -1"


def _check_resolved(spec, tok: torch.Tensor, out: torch.Tensor) -> None:
    """In a cell on a cache manager's pool every selected position has a page in the
    table, so each one resolves to a pool row. A cell whose pages went negative would
    read -1 in the kernel and in the reference alike, and so pass on nothing; this
    check fails it."""
    if "pool_blocks" in spec:
        assert max(spec["kv_lens"]) <= spec["table_width"] * spec["block_size"]
        assert not bool(((tok >= 0) & (out < 0)).any()), "a selected position resolved to -1"


def _check_dummy_rows(spec, out: torch.Tensor) -> None:
    """The padding dummy's rows (the last `pad`, one query row each) are one
    request repeated, so each maps to the same pool rows as the first of them."""
    dummies = spec.get("pad", 0)
    if dummies > 1:
        rows = out[-dummies:]
        assert torch.equal(rows, rows[:1].expand_as(rows)), "padding rows differ"


def _check_boot(spec) -> None:
    """A `boot_capture` cell captures what the dep4 boot captures for the long
    sequence family: the warm-up batch's longest dummy, max_seq_len - 1 keys,
    first and one-key fillers, each a request of its own, on the engine's table
    width; one of its replays is attention DP's single live row at max_seq_len
    behind the padding dummy."""
    if not spec.get("boot_capture"):
        return
    b, bs = len(spec["kv_lens"]), spec["block_size"]
    assert spec["kv_lens"] == [_DEP4_MAX_SEQ_LEN - 1] + [1] * (b - 1), spec["kv_lens"]
    assert spec["seq_rows"] == [1] * b and not spec.get("pad"), "fillers are requests of their own"
    assert spec["table_width"] == -(-math.ceil((_DEP4_MAX_SEQ_LEN + 1) / bs) // 4) * 4
    replays = zip(spec["replay_kv_lens"], spec["replay_pad"])
    assert any(pad == b - 1 and kv[0] == _DEP4_MAX_SEQ_LEN for kv, pad in replays)


def _drive_eager(spec, seed: int) -> None:
    args = _build(spec, seed)
    snap = [a.clone() for a in args[:3]]
    for layer in _layer_ids(spec):
        out = op(*args, layer)
        _compare(out, op.reference(*args, layer))
        _check_padding(out, args[2])
        _check_resolved(spec, args[2], out)
        _check_dummy_rows(spec, out)
    _check_reach(spec, [out])
    _check_newest(spec, args[2], out)
    for a, s in zip(args[:3], snap):
        assert torch.equal(a, s), "an input tensor was mutated"
    if spec.get("prefix_sweep"):
        # The grid is one block row per query row, so a shorter call is a
        # different launch; each must be bitwise the leading rows of the long one.
        # Compared as int64 pairs with `any`: at 32768 rows a count over int32
        # elements cost about as much as the 32768 calls themselves.
        req, table, tok, bs, topk, stride = args
        full = out.view(torch.int64)
        bad = torch.zeros((), dtype=torch.bool, device="cuda")
        for m in range(1, req.shape[0] + 1):
            short = op(req[:m], table, tok[:m], bs, topk, stride, layer)
            bad |= torch.ne(short.view(torch.int64), full[:m]).any()
        if bool(bad):
            first = next(
                (
                    m
                    for m in range(1, req.shape[0] + 1)
                    if not torch.equal(
                        op(req[:m], table, tok[:m], bs, topk, stride, layer), out[:m]
                    )
                ),
                None,
            )
            raise AssertionError(
                f"a shorter call differs from the full call's leading rows (re-run: first at {first})"
            )


def _drive_captured(spec, seed: int) -> None:
    """Capture the step's calls, then replay them on new batch data.

    The replay data reverses the batch's contexts and draws new pages and
    positions; it is written into the captured inputs, as the engine writes a
    step's metadata into its graph buffers. Each replayed output must be
    bitwise an eager call on that data, and the reference. With
    `replay_kv_lens` the graph is replayed once per entry, on those contexts,
    each drawn anew, and `replay_pad` names how many of each replay's trailing
    requests are the padding dummy. A cell with `replay_kv_lens` reaches past
    2**24 + 1 over the capture data and every replay together, any other in its
    replay alone.
    """
    _check_boot(spec)
    layers = _layer_ids(spec)
    req, table, tok, bs, topk, stride = _build(spec, seed)
    peaks, replay_peaks = [], []
    for layer in layers:  # eager on the capture data, validated in full
        out = op(req, table, tok, bs, topk, stride, layer)
        _compare(out, op.reference(req, table, tok, bs, topk, stride, layer))
        _check_newest(spec, tok, out)
        _check_resolved(spec, tok, out)
        _check_dummy_rows(spec, out)
        peaks.append(out.max())
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(req, table, tok, bs, topk, stride, layers[0])
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [op(req, table, tok, bs, topk, stride, layer) for layer in layers]

    replays = spec.get("replay_kv_lens", (spec["kv_lens"][::-1],))
    pads = spec.get("replay_pad", (0,) * len(replays))
    assert len(pads) == len(replays), "one replay_pad entry per replay"
    for k, (kv_lens, pad) in enumerate(zip(replays, pads)):
        again = dict(spec, seq_rows=spec["seq_rows"][::-1], kv_lens=list(kv_lens), pad=pad)
        new_req, new_table, new_tok, *_ = _build(again, seed + 1 + k)
        assert new_tok.shape == tok.shape and not torch.equal(new_tok, tok)
        req.copy_(new_req)
        table.copy_(new_table)
        tok.copy_(new_tok)
        graph.replay()
        for layer, out in zip(layers, outs):
            eager = op(req, table, tok, bs, topk, stride, layer)
            assert torch.equal(out, eager), f"layer_id {layer}: graph replay differs from eager"
            _compare(out, op.reference(req, table, tok, bs, topk, stride, layer))
            _check_padding(out, tok)
            _check_newest(again, tok, out)
            _check_resolved(again, tok, out)
            _check_dummy_rows(again, out)
            peaks.append(out.max())
            replay_peaks.append(out.max())
    _check_reach(spec, peaks if "replay_kv_lens" in spec else replay_peaks)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    if cell.spec.get("graph"):
        _drive_captured(cell.spec, seed)
    else:
        _drive_eager(cell.spec, seed)


def _small():
    req = torch.tensor([0, 1, 1, 0], dtype=torch.int32, device="cuda")
    table = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32, device="cuda")
    tok = torch.tensor(
        [[0, 33, -1, 5], [1, 2, 63, -1], [40, 0, -1, -1], [31, 32, 7, 8]],
        dtype=torch.int32,
        device="cuda",
    )
    return req, table, tok


def test_hand_worked_rows() -> None:
    """Four rows worked out by hand, independent of `reference`."""
    req, table, tok = _small()
    out = op(req, table, tok, 32, 4, 64, 1)
    # row 0, req 0: p=0 -> page 3: 3*64 + 0 + 32 = 224; p=33 -> page 1: 64 + 1 + 32 = 97
    assert out[0].tolist() == [224, 97, -1, 229]
    # row 1, req 1: pages [0, 2]; p=63 -> page 2, offset 31: 128 + 31 + 32 = 191
    assert out[1].tolist() == [33, 34, 191, -1]
    _compare(out, op.reference(req, table, tok, 32, 4, 64, 1))


def _replayed_into_sentinel(*args) -> torch.Tensor:
    """One unguarded call, captured, its output filled with `_SENTINEL` and
    replayed: what the call leaves unwritten keeps the sentinel. The output is
    the graph's own, so this owns every byte it inspects."""
    op(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    out.fill_(_SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    return out


def test_guard_refuses_a_column_count_mismatch() -> None:
    """Narrower leaves each row's trailing columns unwritten, shown below.
    Wider writes each row's surplus over the next row and the last row's past
    the output: refused, never driven unguarded."""
    req, table, tok = _small()
    ref = op.reference(req, table, tok, 32, 4, 64, 1)
    out = _replayed_into_sentinel(req, table, tok, 32, 2, 64, 1)
    assert torch.equal(out[:, :2], ref[:, :2])
    assert bool((out[:, 2:] == _SENTINEL).all()), "the op wrote past num_topk_tokens"
    for bad in (2, 8):
        with pytest.raises(AssertionError, match="num_topk_tokens must equal"):
            with validating(op):
                op(req, table, tok, 32, bad, 64, 0)


def test_guard_refuses_a_req_id_length_mismatch() -> None:
    """The kernel's row count is req_id's, not token_indices'. Shorter leaves
    the trailing output rows unwritten, shown below; longer reads past
    token_indices and writes past the output: refused, never driven unguarded."""
    req, table, tok = _small()
    ref = op.reference(req, table, tok, 32, 4, 64, 1)
    out = _replayed_into_sentinel(req[:2].contiguous(), table, tok, 32, 4, 64, 1)
    assert torch.equal(out[:2], ref[:2])
    assert bool((out[2:] == _SENTINEL).all()), "the op wrote rows past req_id"
    for bad in (req[:2].contiguous(), torch.cat([req, req])):
        with pytest.raises(AssertionError, match="exactly one entry per row"):
            with validating(op):
                op(bad, table, tok, 32, 4, 64, 0)


def test_guard_refuses_a_req_id_outside_the_table() -> None:
    """`block_table[req_id[i]]` is read unchecked. Both calls read rows of
    `parent`, which the test owns: past a sliced table they are the parent's
    next row, and -1 is the row before the slice."""
    parent = torch.tensor([[1, 2], [3, 4], [7, 8]], dtype=torch.int32, device="cuda")
    tok = torch.tensor([[0, 33]], dtype=torch.int32, device="cuda")
    past = op(torch.tensor([2], dtype=torch.int32, device="cuda"), parent[:2], tok, 32, 2, 2304, 0)
    assert past.tolist() == [[7 * 2304, 8 * 2304 + 1]]
    before = op(
        torch.tensor([-1], dtype=torch.int32, device="cuda"), parent[1:], tok, 32, 2, 2304, 0
    )
    assert before.tolist() == [[1 * 2304, 2 * 2304 + 1]]
    for req, table in ((2, parent[:2]), (-1, parent[1:])):
        with pytest.raises(AssertionError, match="must be rows of block_table"):
            with validating(op):
                op(
                    torch.tensor([req], dtype=torch.int32, device="cuda"),
                    table,
                    tok,
                    32,
                    2,
                    2304,
                    0,
                )


def test_guard_refuses_a_layer_outside_the_page() -> None:
    """At stride 2304 a page holds layers 0..71. Layer 72 lands on the next
    page's layer 0, and -1 on the previous page's layer 71: pages this request
    was never given."""
    req = torch.zeros(1, dtype=torch.int32, device="cuda")
    table = torch.tensor([[5, 9]], dtype=torch.int32, device="cuda")
    tok = torch.tensor([[0, 1, 31, 32]], dtype=torch.int32, device="cuda")
    beyond = op(req, table, tok, 32, 4, 2304, 72)
    assert beyond.tolist() == [[6 * 2304, 6 * 2304 + 1, 6 * 2304 + 31, 10 * 2304]]
    before = op(req, table, tok, 32, 4, 2304, -1)
    assert before.tolist() == [
        [4 * 2304 + 71 * 32, 4 * 2304 + 71 * 32 + 1, 5 * 2304 - 1, 8 * 2304 + 71 * 32]
    ]
    with validating(op):
        for layer in (0, 71):
            _compare(
                op(req, table, tok, 32, 4, 2304, layer),
                op.reference(req, table, tok, 32, 4, 2304, layer),
            )
    for layer in (72, -1):
        with pytest.raises(AssertionError, match="outside a page of 72 layers"):
            with validating(op):
                op(req, table, tok, 32, 4, 2304, layer)


def test_guard_refuses_an_int32_overflow() -> None:
    """The pool row is int32 on the device. At stride 2304, page 932067's rows
    start 1280 below 2**31: layer 39 ends exactly at 2**31 - 1, and layer 40
    starts at 2**31, which the kernel wraps to -2**31 without raising."""
    req = torch.zeros(1, dtype=torch.int32, device="cuda")
    table = torch.tensor([[932067]], dtype=torch.int32, device="cuda")
    tok = torch.arange(32, dtype=torch.int32, device="cuda").view(1, 32)
    offsets = torch.arange(32, dtype=torch.int64, device="cuda")
    with validating(op):
        edge = op(req, table, tok, 32, 32, 2304, 39)
    assert torch.equal(edge.long()[0], 932067 * 2304 + 39 * 32 + offsets)
    assert int(edge.max()) == 2**31 - 1
    wrapped = op(req, table, tok, 32, 32, 2304, 40)
    assert torch.equal(wrapped.long()[0], -(2**31) + offsets), "the op no longer wraps"
    for layer, block in ((40, 932067), (0, 932068)):
        with pytest.raises(AssertionError, match="does not fit int32"):
            with validating(op):
                op(req, torch.full_like(table, block), tok, 32, 32, 2304, layer)
    small_req, small_table, small_tok = _small()
    with pytest.raises(AssertionError, match="does not fit int32"):
        with validating(op):
            op(small_req, small_table, small_tok, 32, 4, 2**30, 0)


def test_owner_row_map() -> None:
    """The fp8_ds_mla cells' `layer_ids`: each layer's source's index among the 63 KV
    owners. The closed form the entry writes is the owner-index form, every owner row
    0..62 is some layer's, and each sharer 39, 43, ..., 71 reads owner row 35."""
    assert _ML4_DS_STEP == tuple(_ML4_OWNERS.index(_ML4_STEP[i]) for i in range(72))
    assert len(_ML4_OWNERS) == _ML4_1M_DS["pool_layers"] == _ML4_32K_DS["pool_layers"] == 63
    assert _ML4_1M_DEP4_DS["pool_layers"] == _ML4_1M_TEP8_DS["pool_layers"] == 63
    assert _ML4_32K_DS_8GPU["pool_layers"] == 63
    assert sorted(set(_ML4_DS_STEP)) == list(range(63))
    sharers = [i for i in range(72) if _ML4_STEP[i] != i]
    assert sharers == list(range(39, 72, 4)) and {_ML4_DS_STEP[i] for i in sharers} == {35}
    assert _ML4_1M_DS["pool_layers"] * _ML4_1M_DS["block_size"] == 2016
    assert _ML4_1M_DEP4_DS["pool_layers"] * _ML4_1M_DEP4_DS["block_size"] == 2016
    assert _ML4_1M_TEP8_DS["pool_layers"] * _ML4_1M_TEP8_DS["block_size"] == 2016
