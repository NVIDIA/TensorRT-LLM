# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the indexer_k_cache_gather_op catalog entry.

The "dep4 1M" cells are the ones the Mistral Large 4 NVFP4 dep4 target passes at
max_seq_len 1048576 with chunked prefill, where a chunk gathers its requests'
whole KV (`cached` + new keys) from the step's full-KV mapping.
"""

import math
import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.indexer_k_cache_gather_op import (  # noqa: E501
    indexer_k_cache_gather_op as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "indexer_k_cache_gather_op requires a CUDA device"

#: Mistral Large 4's index_sharing (params.json's DSA sharing), one letter per layer: layers 0..16
#: run their own indexer, then each 4 layers are 2 'shared' and 2 'full', and
#: layer 71 is full. 44 'F' in all.
_ML4_PATTERN = "F" * 17 + "SSFF" * 13 + "SSF"
assert len(_ML4_PATTERN) == 72 and _ML4_PATTERN.count("F") == 44

#: The engine's token budget per step at the 1M envelope (the chunk size); a
#: manager at max_seq_len 32768 or below gets its max_seq_len.
MAX_NUM_TOKENS = 32768
#: The tep4 and tp4 targets' TP ranks: from q_split_threshold context rows on, the
#: engine splits a 32768-row chunk 8192 rows per rank.
TP_RANKS = 4
#: The tep8 target's: 4096 rows of a 32768-row chunk per rank (a cell's
#: `qsplit_ranks`).
TEP8_RANKS = 8
#: The block-table width a V2 manager hands out (`max_blocks_per_seq`), measured
#: at the max_seq_len the 1M cells build it at.
_V2_TABLE_WIDTH = {524288: 16388, 1048576: 32772}


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bit-exact: a gather moves bytes and computes nothing. Call once for
    the data and once for the scale."""
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert torch.equal(out.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)), (
        "gathered bytes differ from the reference"
    )


def _slot_mappings(pages_per_seq, positions_per_seq, head_dim, tpb, page_scale=1):
    """Flat byte offsets of each listed token's data and scale, in sequence order.

    The same arithmetic the DSA indexer uses: a page is `tpb * (head_dim + 4)`
    bytes, all data first, then all scales, and a block's page index is its id
    times the manager's page scale.
    """
    page_bytes = tpb * (head_dim + 4)
    fp8, scale = [], []
    for pages, pos in zip(pages_per_seq, positions_per_seq):
        page = pages[pos // tpb].long() * page_scale
        fp8.append(page * page_bytes + (pos % tpb) * head_dim)
        scale.append(page * page_bytes + tpb * head_dim + (pos % tpb) * 4)
    return torch.cat(fp8).cuda(), torch.cat(scale).cuda()


class _Built(SimpleNamespace):
    """A cell's arguments, the memory the test owns behind `k_cache`, and how to
    release it. A manager's views must be gone before the manager shuts down."""

    def close(self) -> None:
        manager = self.manager
        self.args = self.owned = self.manager = None
        if manager is not None:
            manager.shutdown()


def _build_interleaved(spec, seed) -> _Built:
    """The cache is one layer's view of a pool that interleaves `pool_layers`
    layers page by page, so dim 0 is not contiguous when there is more than
    one -- the layout the K cache manager hands out."""
    g = torch.Generator().manual_seed(seed)
    hd, tpb = spec["head_dim"], spec["tokens_per_block"]
    pages = [math.ceil(n / tpb) for n in spec["seq_lens"]]
    num_pages = 2 * sum(pages) + 1
    pool = torch.randperm(num_pages, generator=g).to(torch.int32)
    table = torch.zeros(len(pages), max(pages), dtype=torch.int32)
    used = 0
    for s, n in enumerate(pages):
        table[s, :n] = pool[used : used + n]
        used += n
    layers = spec["pool_layers"]
    parent = torch.randint(
        0, 256, (num_pages, layers, tpb * (hd + 4)), generator=g, dtype=torch.uint8
    ).cuda()
    k_cache = parent[:, layers // 2].view(num_pages, tpb, 1, hd + 4)
    assert layers == 1 or not k_cache.is_contiguous()
    fp8, scale = _slot_mappings(list(table), [torch.arange(n) for n in spec["seq_lens"]], hd, tpb)
    args = (k_cache, fp8, scale, spec["k_token_start"], spec["num_tokens"], hd)
    return _Built(args=args, owned=parent, manager=None)


def _build_view(spec, seed) -> _Built:
    """Mistral Large 4's rebuilt views, the geometry both managers were measured to
    hand out: "v1" is row `pool_row` of a [num_blocks, pool_layers, page]
    pool, "v2" a contiguous view starting at that row's first page, with page
    indices scaled by `pool_layers`. The prompts' tokens come first in the
    mapping, then one new-token slot per decode row: `decode_pos` lists their
    positions, `decode_rows` draws that many below 4096. With `cycle` the
    step's pages may outnumber the pool: they are consecutive fresh
    permutations of it, so a page repeats but no stretch of the mapping
    repeats another. With `cached` a prompt's keys are its cached prefix and
    its new keys, positions 0..cached + new - 1, as the full-KV mapping lists
    them, and a decode row owns only the page its new key sits on (its earlier
    pages are never read).

    `cycle` (only the 1M, 1M tp4 and 1M tep8 cells carry it) and `cached` (only
    the dep4 1M cells) are never set together; a cell without `cached` takes
    prefix 0, so its positions, page counts, draws and mapping are those of
    the helper without the `cached` lines."""
    g = torch.Generator().manual_seed(seed)
    hd, tpb = spec["head_dim"], spec["tokens_per_block"]
    page_bytes = tpb * (hd + 4)
    rows, row, blocks = spec["pool_layers"], spec["pool_row"], spec["num_blocks"]
    decode = torch.randint(1, 4096, (spec.get("decode_rows", 0),), generator=g).tolist()
    decode += spec.get("decode_pos", [])
    prefix = spec.get("cached", [0] * len(spec["seq_lens"]))
    prompts = [torch.arange(c + n) for c, n in zip(prefix, spec["seq_lens"])]
    positions = prompts + [torch.tensor([p]) for p in decode]
    need = [int(p[-1]) // tpb + 1 for p in positions]
    if "cached" in spec:
        need[len(prompts) :] = [1] * len(decode)
    assert not (spec.get("cycle") and "cached" in spec), "cycle and cached are never combined"
    if spec.get("cycle"):
        perms = math.ceil(sum(need) / blocks)
        ids = torch.cat([torch.randperm(blocks, generator=g) for _ in range(perms)])[: sum(need)]
    else:
        assert sum(need) <= blocks, (sum(need), blocks)
        if spec.get("tail"):
            ids = torch.arange(blocks - sum(need), blocks)[torch.randperm(sum(need), generator=g)]
        else:
            ids = torch.randperm(blocks, generator=g)[: sum(need)]
    pages = list(torch.split(ids, need))
    if "cached" in spec:
        for i, p in enumerate(decode, start=len(prompts)):
            # Only the page holding position p is real; the earlier entries are never read.
            pages[i] = torch.cat([torch.full((p // tpb,), -1, dtype=ids.dtype), pages[i]])
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if spec["layout"] == "v1":
        parent = torch.randint(
            0, 256, (blocks, rows, page_bytes), dtype=torch.uint8, device="cuda", generator=gen
        )
        k_cache = parent[:, row].view(blocks, tpb, 1, hd + 4)
        assert k_cache.stride(0) == rows * page_bytes
        page_scale = 1
    else:
        parent = torch.randint(
            0, 256, (blocks * rows * page_bytes,), dtype=torch.uint8, device="cuda", generator=gen
        )
        k_cache = parent[row * page_bytes :].view(blocks * rows - row, tpb, 1, hd + 4)
        page_scale = rows
    fp8, scale = _slot_mappings(pages, positions, hd, tpb, page_scale)
    assert fp8.numel() == sum(prefix) + sum(spec["seq_lens"]) + len(decode)
    if "cached" in spec:
        print(
            f"\n[view] {spec['layout']} row {row} of {blocks} blocks x {rows} rows: mapping "
            f"{fp8.numel()} keys, data slots {int(fp8.min())}..{int(fp8.max())}"
        )
    args = (k_cache, fp8, scale, spec["k_token_start"], spec["num_tokens"], hd)
    return _Built(args=args, owned=parent, manager=None)


def _build_manager(spec, seed) -> _Built:
    """A real DSA cache manager at Mistral Large 4's geometry (72 layers, latent 576
    fp8, the 44-row indexer mask): the view `get_indexer_k_cache_buffers` hands
    out, the block ids `get_pool_block_indices` reads for the metadata, and the
    indexer's own slot arithmetic. The pages the window reads are filled with
    seeded random bytes through the view. `max_seq_len` (default 9216, the FP8
    target's) is also the manager's max_num_tokens up to 32768, the 1M
    envelope's chunk. With `cached` each prompt holds its cached and new keys,
    and the mapping lists them all, as the full-KV mapping does; the manager
    must then keep the cell's max_seq_len (V2 lowers it to what a small quota
    holds). The pool is allocated outside torch's caching allocator, so what
    torch still caches from earlier cells is returned to the device first.

    Only the dep4 1M cells carry `cached`; every `cached` line is skipped
    without it (the prompts' lengths are then `seq_lens`). The dep4 run
    emptied torch's cache before building the config objects, the tp4, tep4
    and tep8 runs after; the configs allocate no device memory, so the manager
    is built on the same device state either way."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
        DSACacheManager,
        DSACacheManagerV2,
    )
    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import _compute_slot_mappings
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.executor import KvCacheConfig as BindingKvCacheConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    hd, tpb = spec["head_dim"], spec["tokens_per_block"]
    max_seq_len = spec.get("max_seq_len", 9216)
    v2 = spec["layout"] == "manager_v2"
    if v2:
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False, max_tokens=spec["max_tokens"], host_cache_size=0
        )
    else:
        kv_cache_config = BindingKvCacheConfig(
            enable_block_reuse=False, max_tokens=spec["max_tokens"]
        )
    torch.cuda.empty_cache()
    manager = (DSACacheManagerV2 if v2 else DSACacheManager)(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=72,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=tpb,
        max_seq_len=max_seq_len,
        max_batch_size=16,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        sparse_attention_config=DeepSeekSparseAttentionConfig(index_head_dim=hd),
        pretrained_config=SimpleNamespace(
            num_hidden_layers=72, qk_rope_head_dim=64, index_topk_pattern=list(_ML4_PATTERN)
        ),
        max_num_tokens=min(max_seq_len, MAX_NUM_TOKENS),
    )
    try:
        assert sum(manager.indexer_k_cache_local_layer_mask) == spec["pool_layers"]
        assert manager.indexer_k_cache_page_scale == (spec["pool_layers"] if v2 else 1)
        lens = [
            c + n for c, n in zip(spec.get("cached", [0] * len(spec["seq_lens"])), spec["seq_lens"])
        ]
        if "cached" in spec:
            assert manager.max_seq_len == max_seq_len, (manager.max_seq_len, max_seq_len)
        ids = list(range(len(lens)))
        added = manager.add_dummy_requests(ids, lens, is_gen=False, prepare_resource=True)
        assert added is not None, "the manager's quota refused the cell's prompts"
        if not v2:
            # V1 reads the block ids back from the offsets the attention metadata copies.
            manager.copy_batch_block_offsets(
                torch.zeros_like(manager.host_kv_cache_block_offsets, device="cuda"),
                ids,
                1,
                len(ids),
                len(ids),
            )
        blocks = manager.get_pool_block_indices(len(ids), request_ids=ids, num_contexts=len(ids))
        if "cached" in spec:
            assert blocks.shape[1] == _V2_TABLE_WIDTH[max_seq_len], tuple(blocks.shape)
        page_table = blocks.long() * manager.indexer_k_cache_page_scale
        req = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))
        pos = torch.cat([torch.arange(n) for n in lens])
        fp8, scale = _compute_slot_mappings(pos, page_table, req, hd, tpb, 128)
        fp8, scale = fp8.cuda(), scale.cuda()
        k_cache = manager.get_indexer_k_cache_buffers(spec["layer"])
        assert k_cache.is_contiguous() == v2
        print(
            f"\n[manager] {type(manager).__name__} quota {spec['max_tokens']} tokens, "
            f"max_seq_len {manager.max_seq_len}, table {tuple(blocks.shape)}; layer "
            f"{spec['layer']} view {tuple(k_cache.shape)}, data slots "
            f"{int(fp8.min())}..{int(fp8.max())}"
        )
        pages = torch.unique(fp8 // (tpb * (hd + 4)))
        gen = torch.Generator(device="cuda").manual_seed(seed)
        k_cache[pages] = torch.randint(
            0,
            256,
            (pages.numel(), *k_cache.shape[1:]),
            dtype=torch.uint8,
            device="cuda",
            generator=gen,
        )
        if "cached" in spec:
            first = [int(blocks[i, 0]) for i in ids]
            last = [int(blocks[i, (n - 1) // tpb]) for i, n in zip(ids, lens)]
            print(
                f"\n[manager] {type(manager).__name__} quota {spec['max_tokens']} tokens, "
                f"{manager.num_blocks} slots, max_seq_len {manager.max_seq_len}, table "
                f"{tuple(blocks.shape)}; prompts' slots first {first}, last {last}; layer "
                f"{spec['layer']} view {tuple(k_cache.shape)}, mapping {fp8.numel()} keys, data "
                f"slots {int(fp8.min())}..{int(fp8.max())}"
            )
    except BaseException:
        manager.shutdown()
        raise
    args = (k_cache, fp8, scale, spec["k_token_start"], spec["num_tokens"], hd)
    return _Built(args=args, owned=k_cache, manager=manager)


def _build(spec, seed) -> _Built:
    if spec["layout"].startswith("manager"):
        return _build_manager(spec, seed)
    return _build_view(spec, seed)


def _gather_once(args) -> None:
    data, scale = op(*args)
    assert data.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert data.is_contiguous() and scale.is_contiguous()
    ref_data, ref_scale = op.reference(*args)
    _compare(data, ref_data)
    _compare(scale, ref_scale)


#: Key counts `_sweep` compares at once: near 32768 keys a group's outputs and
#: the rows they must equal take about 0.5 GiB.
_SWEEP_GROUP = 32


def _sweep(args) -> None:
    """Every key count n of the cell's window, read twice: as its first n keys
    and as its last n. The kernel's grid is one block per key, so this is every
    grid the window can launch; the expected rows are the entry's reference of
    the whole window, sliced.

    The loop is launch-bound. On a GB300 at the 32768-key window a call is
    dominated by host time, and the guard's host read and two per-call
    compares add more than the call itself. So the outputs of `_SWEEP_GROUP`
    key counts are concatenated and compared bitwise with the rows they must
    equal at once, and only a group that differs is read again window by
    window to name the windows. The groups run from the largest key count
    down, so the caching allocator reuses each group's blocks for the next:
    in rising order it reserved 32.8 GiB for a 1.5 GiB peak."""
    k_cache, fp8, scale, start, total, hd = args
    ref_data, ref_scale = op.reference(*args)

    def windows(counts):
        return [(n, side, first) for n in counts for side, first in enumerate((0, total - n))]

    def gathered(first, n):
        return op(k_cache, fp8, scale, start + first, n, hd)

    def expected(first, n):
        return ref_data[first : first + n], ref_scale[first : first + n]

    def same(got, want) -> bool:
        return all(
            torch.equal(torch.cat(g).view(torch.int32), torch.cat(w).view(torch.int32))
            for g, w in zip(got, want)
        )

    wrong = []
    for high in range(total, 0, -_SWEEP_GROUP):
        group = windows(range(high, max(high - _SWEEP_GROUP, 0), -1))
        got = zip(*(gathered(first, n) for n, _, first in group))
        if not same(got, zip(*(expected(first, n) for n, _, first in group))):
            wrong += [
                (n, side)
                for n, side, first in group
                if not same(zip(gathered(first, n)), zip(expected(first, n)))
            ]
    assert not wrong, f"(key count, prefix=0/suffix=1) windows that differ: {sorted(wrong)[:8]}"


def _qsplit_shares(seq_lens, ranks=TP_RANKS) -> list[tuple[int, int]]:
    """`(start, count)` of each TP rank's gather when the engine's query split cuts
    the last 32768-key chunk of a request of that many keys or more over `ranks`
    ranks: rank r gathers the request's cached keys and its share's rows, cached +
    8192 (r + 1) keys at 4 ranks and cached + 4096 (r + 1) at 8, from the
    request's offset in the mapping."""
    shares, offset = [], 0
    for n in seq_lens:
        if n >= MAX_NUM_TOKENS:
            cached, share = n - MAX_NUM_TOKENS, MAX_NUM_TOKENS // ranks
            shares += [(offset, cached + share * (r + 1)) for r in range(ranks)]
        offset += n
    return shares


def _edge_windows(total: int) -> list[tuple[int, int]]:
    """`(start, count)` windows across the integer edges a 1M mapping reaches: 2^k - 1,
    2^k and 2^k + 1 keys (k = 15..21) and the whole mapping, at its start and at
    its end; and 1, 2049 and every remaining key from keys 2^20 and 2^21, each +-1."""
    counts = [2**k + d for k in range(15, 22) for d in (-1, 0, 1) if 2**k + d < total] + [total]
    starts = [2**k + d for k in (20, 21) for d in (-1, 0, 1) if 2**k + d < total]
    return (
        [(0, n) for n in counts]
        + [(total - n, n) for n in counts]
        + [(s, n) for s in starts for n in (1, 2049, total - s)]
    )


def _wrong_windows(args, windows) -> list[tuple[int, int]]:
    """The `(start, count)` windows whose gather differs from the rows of the
    entry's reference over the whole mapping. Each window is one call and one
    compare (int32 words: a scale's random bytes may read as NaN)."""
    k_cache, fp8, scale, _, _, hd = args
    ref_data, ref_scale = op.reference(k_cache, fp8, scale, 0, fp8.numel(), hd)
    wrong = []
    for first, n in windows:
        data, sc = op(k_cache, fp8, scale, first, n, hd)
        if not (
            torch.equal(data.view(torch.int32), ref_data[first : first + n].view(torch.int32))
            and torch.equal(sc.view(torch.int32), ref_scale[first : first + n].view(torch.int32))
        ):
            wrong.append((first, n))
    return wrong


def _windows(args, seq_lens, ranks=TP_RANKS) -> None:
    """The cell's window, each q-split share of its requests over `ranks` ranks and
    the integer-edge windows of its mapping (`_qsplit_shares`, `_edge_windows`)."""
    _, fp8, _, start, total, _ = args
    windows = [(start, total), *_qsplit_shares(seq_lens, ranks), *_edge_windows(fp8.numel())]
    wrong = _wrong_windows(args, windows)
    assert not wrong, f"(start, count) windows that differ: {wrong}"


def _wrong_counts(args, counts) -> list[tuple[int, int]]:
    """The `(key count, prefix=0/suffix=1)` windows of the listed counts whose
    gather differs from the rows of the entry's reference over the cell's whole
    window. One call and one compare per window (int32 words: a scale's random
    bytes may read as NaN)."""
    k_cache, fp8, scale, start, total, hd = args
    ref_data, ref_scale = op.reference(*args)
    wrong = []
    for n in counts:
        for side, first in enumerate((0, total - n)):
            data, sc = op(k_cache, fp8, scale, start + first, n, hd)
            if not (
                torch.equal(data.view(torch.int32), ref_data[first : first + n].view(torch.int32))
                and torch.equal(
                    sc.view(torch.int32), ref_scale[first : first + n].view(torch.int32)
                )
            ):
                wrong.append((n, side))
    return wrong


def _sweep_counts(args, counts) -> None:
    """Each listed key count n of the cell's window, read twice: as its first n
    keys and as its last n. A 1M window's every count is about 2M calls, so the
    cells name the counts at its edges instead (one key, index_topk, the 32768
    chunk and one past it, one past each power of two, the window less one key,
    the whole window)."""
    wrong = _wrong_counts(args, counts)
    assert not wrong, f"(key count, prefix=0/suffix=1) windows that differ: {wrong}"


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """`torch.equal`, compared 8 bytes at a time where the buffer allows it: its
    elementwise temporary is then an eighth of a 5 GB pool, not the pool's size."""
    if a.is_contiguous() and b.is_contiguous() and a.numel() % 8 == 0:
        a, b = a.reshape(-1).view(torch.int64), b.reshape(-1).view(torch.int64)
    return torch.equal(a, b)


@pytest.mark.parametrize(
    "cell",
    [pytest.param(c, marks=pytest.mark.large) if c.spec.get("large") else c for c in op.CELLS],
    ids=[c.why[:44] for c in op.CELLS],
)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    built = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    try:
        before = built.owned.clone()
        # `sweep="windows"` is only on the 1M cells (their q-split shares over
        # `qsplit_ranks` ranks, 4 unless given), `sweep_counts` only on a dep4 1M
        # cell, and no cell carries both: each reaches its own member's branch.
        if cell.spec.get("sweep") == "windows":
            _windows(built.args, cell.spec["seq_lens"], cell.spec.get("qsplit_ranks", TP_RANKS))
        elif cell.spec.get("sweep_counts"):
            _sweep_counts(built.args, cell.spec["sweep_counts"])
        elif cell.spec.get("sweep"):
            _sweep(built.args)
        else:
            _gather_once(built.args)
        assert _same_bytes(built.owned, before), "the gather wrote into memory behind the cache"
        del before
    finally:
        built.close()


def _ok(seed=5, seq_lens=(64,), pool_layers=1):
    return _build_interleaved(
        dict(
            head_dim=128,
            tokens_per_block=32,
            seq_lens=list(seq_lens),
            k_token_start=0,
            num_tokens=sum(seq_lens),
            pool_layers=pool_layers,
        ),
        seed=seed,
    ).args


def test_zero_tokens_returns_empty() -> None:
    """Nothing but head_dim is read: a CPU int32 mapping and a 2-D one pass."""
    k_cache, fp8, scale, _, _, hd = _ok(seed=3, seq_lens=(8,))
    data, sc = op(k_cache, fp8.cpu().int(), scale[:1].view(1, 1), 0, 0, hd)
    assert tuple(data.shape) == (0, hd) and data.dtype == torch.float8_e4m3fn
    assert tuple(sc.shape) == (0, 1) and sc.dtype == torch.float32
    with pytest.raises(RuntimeError, match="head_dim must be 128 \\(FP8\\) or 64"):
        op(k_cache, fp8, scale, 0, 0, 96)


@pytest.mark.parametrize(
    "wrong, message",
    [
        ("window past the mapping", "slot_mapping_fp8 too short for k_token_start \\+ num_tokens"),
        ("int32 mapping", "slot_mapping_fp8 must be int64"),
        ("strided mapping", "slot_mapping_fp8 must be contiguous"),
        ("fp8-typed cache", "expected scalar type Byte but found Float8_e4m3fn"),
        ("3-D cache", "k_cache must be a 4D Tensor"),
        ("head_dim 96", "head_dim must be 128 \\(FP8\\) or 64"),
    ],
)
def test_the_op_refuses_loudly(wrong, message) -> None:
    """What the op validates itself, so `is_valid` does not restate it."""
    k_cache, fp8, scale, start, n, hd = _ok(seed=13)
    call = {
        "window past the mapping": lambda: op(k_cache, fp8, scale, 1, n, hd),
        "int32 mapping": lambda: op(k_cache, fp8.int(), scale, start, n, hd),
        "strided mapping": lambda: op(k_cache, torch.stack([fp8, fp8], 1)[:, 0], scale, 0, n, hd),
        "fp8-typed cache": lambda: op(k_cache.view(torch.float8_e4m3fn), fp8, scale, 0, n, hd),
        "3-D cache": lambda: op(k_cache.reshape(k_cache.shape[0], 32, -1), fp8, scale, 0, n, hd),
        "head_dim 96": lambda: op(k_cache, fp8, scale, start, n, 96),
    }[wrong]
    with validating(op), pytest.raises(RuntimeError, match=message):
        call()


def test_guard_refuses_a_negative_start() -> None:
    """The op checks only the upper end of the window. Unguarded, a start of -8
    reads the eight int64 words in front of the mapping as slots."""
    k_cache, fp8, scale, _, _, hd = _ok(seed=17, seq_lens=(200,))
    # The words in front belong to the test: tokens 100..107's slots.
    fp8_buf, scale_buf = torch.cat([fp8[100:108], fp8]), torch.cat([scale[100:108], scale])
    fp8_view, scale_view = fp8_buf[8:], scale_buf[8:]
    data, _ = op(k_cache, fp8_view, scale_view, -8, 16, hd)
    front, _ = op.reference(k_cache, fp8, scale, 100, 8, hd)
    assert torch.equal(data[:8].view(torch.uint8), front.view(torch.uint8)), (
        "a negative start no longer reads the words in front of the mapping"
    )
    with pytest.raises(AssertionError, match="k_token_start must be >= 0"):
        with validating(op):
            op(k_cache, fp8_view, scale_view, -8, 16, hd)


def test_guard_refuses_a_negative_slot() -> None:
    """The kernel skips the row and leaves torch.empty's contents in it."""
    k_cache, fp8, scale, _, _, hd = _ok(seed=19)
    bad = fp8.clone()
    bad[3] = -1
    data, _ = op(k_cache, bad, scale, 0, 64, hd)
    ref, _ = op.reference(k_cache, fp8, scale, 0, 64, hd)
    rows, ref_rows = data.view(torch.uint8), ref.view(torch.uint8)
    assert torch.equal(torch.cat([rows[:3], rows[4:]]), torch.cat([ref_rows[:3], ref_rows[4:]]))
    assert not torch.equal(rows[3], ref_rows[3]), "a negative slot's row came back correct"
    with pytest.raises(AssertionError, match="must be >= 0"):
        with validating(op):
            op(k_cache, bad, scale, 0, 64, hd)


def test_guard_refuses_a_slot_past_the_view() -> None:
    """Nothing bounds a slot from above. Unguarded, a slot one page past a view
    that is the leading slice of a larger buffer reads that buffer's tail."""
    k_cache, fp8, scale, _, _, hd = _ok(seed=23)
    blocks = k_cache.shape[0]
    page_bytes = k_cache[0].numel()
    buf = torch.full((blocks + 4, page_bytes), 0xEE, dtype=torch.uint8, device="cuda")
    buf[:blocks] = k_cache.reshape(blocks, page_bytes)
    view = buf[:blocks].view(k_cache.shape)
    fp8 = fp8.clone()
    fp8[0] = blocks * page_bytes + 128  # token 1 of the first page past the view
    data, _ = op(view, fp8, scale, 0, 64, hd)
    assert bool((data[0].view(torch.uint8) == 0xEE).all()), "the row no longer reads past the view"
    with pytest.raises(AssertionError, match="must address bytes inside k_cache"):
        with validating(op):
            op(view, fp8, scale, 0, 64, hd)


def test_guard_refuses_a_head_dim_the_cache_rows_do_not_hold() -> None:
    """head_dim is the gather width and the op admits 64 (the FP4 cache's) on a
    132-byte FP8 row: unguarded it returns the first half of every key."""
    k_cache, fp8, scale, _, n, _ = _ok(seed=29)
    data, sc = op(k_cache, fp8, scale, 0, n, 64)
    ref, ref_sc = op.reference(k_cache, fp8, scale, 0, n, 128)
    assert tuple(data.shape) == (n, 64)
    assert torch.equal(data.view(torch.uint8), ref.view(torch.uint8)[:, :64])
    assert torch.equal(sc, ref_sc)
    with pytest.raises(AssertionError, match="does not match the cache's 132-byte rows"):
        with validating(op):
            op(k_cache, fp8, scale, 0, n, 64)


def test_the_compare_sees_a_wrong_layer_or_window() -> None:
    """The harness is not blind: the neighbouring layer's view, or a window one
    key off, fails the bit-exact compare."""
    spec = dict(
        head_dim=128,
        tokens_per_block=32,
        seq_lens=[300],
        k_token_start=0,
        num_tokens=300,
        pool_layers=3,
    )
    built = _build_interleaved(spec, seed=31)
    k_cache, fp8, scale, _, n, hd = built.args
    ref_data, _ = op.reference(k_cache, fp8, scale, 0, n, hd)
    neighbour = built.owned[:, 0].view(k_cache.shape)
    with pytest.raises(AssertionError, match="gathered bytes differ"):
        _compare(op(neighbour, fp8, scale, 0, n, hd)[0], ref_data)
    with pytest.raises(AssertionError, match="gathered bytes differ"):
        _compare(op(k_cache, fp8, scale, 1, n - 1, hd)[0], ref_data[: n - 1])


@pytest.mark.parametrize(
    "part, row, first_wrong",
    [
        ("data", 100, "[(101, 0), (102, 0), (103, 0)"),  # prefixes from 101 keys
        ("scale", 250, "[(50, 1), (51, 1), (52, 1)"),  # suffixes from 300 - 250 keys
    ],
)
def test_the_sweep_sees_one_wrong_bit(part, row, first_wrong, monkeypatch) -> None:
    """The sweep is not blind: against a reference one bit off in one key's
    data, or in its scale, exactly the windows holding that key fail."""
    spec = dict(
        head_dim=128,
        tokens_per_block=32,
        seq_lens=[300],
        k_token_start=0,
        num_tokens=300,
        pool_layers=1,
    )
    built = _build_interleaved(spec, seed=37)
    reference = type(op).reference

    def one_bit_off(self, *args):
        data, scale = reference(self, *args)
        (data if part == "data" else scale).view(torch.uint8)[row, 0] ^= 1
        return data, scale

    monkeypatch.setattr(type(op), "reference", one_bit_off)
    with pytest.raises(AssertionError, match="windows that differ") as failed:
        _sweep(built.args)
    assert first_wrong in str(failed.value), str(failed.value)


#: A 1M packed mapping on a pool its pages outnumber 400 times over: the controls'
#: cells reach the 1M cells' offsets in 12 MB.
_CYCLED_1M = dict(
    head_dim=128,
    tokens_per_block=32,
    layout="v2",
    pool_layers=44,
    pool_row=43,
    num_blocks=64,
    cycle=True,
    seq_lens=[1048576, 1048576, 275872],
    k_token_start=0,
    num_tokens=2373024,
)


@pytest.mark.parametrize("part, key", [("data", 2**20 + 5), ("scale", 2**21 - 1)])
def test_the_windows_see_one_wrong_bit(part, key, monkeypatch) -> None:
    """The 1M windows are not blind: against a reference one bit off in one key's
    data, or in its scale, exactly the windows holding that key fail."""
    built = _build_view(_CYCLED_1M, seed=41)
    reference = type(op).reference

    def one_bit_off(self, *args):
        data, scale = reference(self, *args)
        (data if part == "data" else scale).view(torch.uint8)[key, 0] ^= 1
        return data, scale

    windows = [*_qsplit_shares(_CYCLED_1M["seq_lens"]), *_edge_windows(2373024)]
    holding = [(s, n) for s, n in windows if s <= key < s + n]
    assert 0 < len(holding) < len(windows), (len(holding), len(windows))
    monkeypatch.setattr(type(op), "reference", one_bit_off)
    assert _wrong_windows(built.args, windows) == holding
    built.close()


def test_the_compare_sees_a_1m_start_one_early_or_wrapped() -> None:
    """Pages that outnumber the pool do not make a 1M window read like another: the
    second request's first 2049 keys, gathered from key 2^20 - 1 or from key 0 (the
    start wrapped at 2^20), fail the bit-exact compare against those from 2^20."""
    built = _build_view(_CYCLED_1M, seed=43)
    k_cache, fp8, scale, _, _, hd = built.args
    ref_data, ref_scale = op.reference(k_cache, fp8, scale, 2**20, 2049, hd)
    _compare(op(k_cache, fp8, scale, 2**20, 2049, hd)[0], ref_data)
    for start in (2**20 - 1, 0):
        data, sc = op(k_cache, fp8, scale, start, 2049, hd)
        with pytest.raises(AssertionError, match="gathered bytes differ"):
            _compare(data, ref_data)
        with pytest.raises(AssertionError, match="gathered bytes differ"):
            _compare(sc, ref_scale)
    built.close()


#: The 1M windows on a one-row pool: every key its own slot, the operands in 0.2 GB.
_ONE_ROW_1M = dict(
    head_dim=128,
    tokens_per_block=32,
    layout="v2",
    pool_layers=1,
    pool_row=0,
    num_blocks=36864,
    seq_lens=[32768],
    cached=[1015808],
    k_token_start=0,
    num_tokens=1048576,
)
#: The edge counts of the "dep4 1M edge key counts" cell.
_EDGE_COUNTS = next(c.spec["sweep_counts"] for c in op.CELLS if c.spec.get("sweep_counts"))


@pytest.mark.parametrize("part, key", [("data", 2048), ("scale", 1015807)])
def test_the_count_sweep_sees_one_wrong_bit(part, key, monkeypatch) -> None:
    """The edge-count sweep is not blind: against a reference one bit off in one
    key's data, or in its scale, exactly the listed windows holding that key fail.
    Key 1015807 is the last cached key of a 1M prompt's last chunk: the 32768-key
    suffix (the chunk's new keys) misses it, the 32769-key one holds it."""
    built = _build_view(_ONE_ROW_1M, seed=47)
    reference = type(op).reference

    def one_bit_off(self, *args):
        data, scale = reference(self, *args)
        (data if part == "data" else scale).view(torch.uint8)[key, 0] ^= 1
        return data, scale

    total = _ONE_ROW_1M["num_tokens"]
    holding = [
        (n, side)
        for n in _EDGE_COUNTS
        for side, first in enumerate((0, total - n))
        if first <= key < first + n
    ]
    assert 0 < len(holding) < 2 * len(_EDGE_COUNTS), holding
    monkeypatch.setattr(type(op), "reference", one_bit_off)
    assert _wrong_counts(built.args, _EDGE_COUNTS) == holding
    built.close()


def test_the_compare_sees_a_split_window_one_key_off_or_its_start_lost() -> None:
    """The split packed chunk's second window is not read like another: its first
    2049 keys gathered from key 983045 match the reference, and from one key
    early or late, from key 0 (the start dropped: the first request's keys) or
    from key 5 (983045 narrowed to 16 bits) fail the bit-exact compare."""
    spec = dict(_ONE_ROW_1M, num_blocks=61442, seq_lens=[5, 5], cached=[983040, 983040])
    built = _build_view(dict(spec, k_token_start=983045, num_tokens=983045), seed=53)
    k_cache, fp8, scale, start, _, hd = built.args
    assert fp8.numel() == 1966090 and start == 983045
    ref_data, ref_scale = op.reference(k_cache, fp8, scale, start, 2049, hd)
    data, sc = op(k_cache, fp8, scale, start, 2049, hd)
    _compare(data, ref_data)
    _compare(sc, ref_scale)
    for wrong in (start - 1, start + 1, 0, start % 2**16):
        data, sc = op(k_cache, fp8, scale, wrong, 2049, hd)
        with pytest.raises(AssertionError, match="gathered bytes differ"):
            _compare(data, ref_data)
        with pytest.raises(AssertionError, match="gathered bytes differ"):
            _compare(sc, ref_scale)
    built.close()


def test_tep8_windows_are_the_engines_chunks() -> None:
    """The "1M tep8" windows are the chunks the engine plans. Under its query split
    over eight ranks (`_split_prefill_queries`, at q_split_threshold's default 8192),
    rank r's chunk of a 1M prompt's last 32768 rows (`plan_mla_chunked_prefill_chunks`,
    `Indexer.prepare_one_prefill_chunk`) is `_qsplit_shares`' window: 1019904 keys on
    rank 0, 1032192 on rank 3. Two 983045-key requests' 5-token tails pack into one
    chunk of 1966090 keys, past max_seq_len, which the tep8 target gathers per
    request, the second from key 983045."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
        Indexer,
        _packed_logits_elem_budget,
        _split_prefill_queries,
        plan_mla_chunked_prefill_chunks,
    )

    max_seq_len = 1048576
    budget = _packed_logits_elem_budget(
        SimpleNamespace(
            max_num_tokens=MAX_NUM_TOKENS,
            kv_cache_manager=SimpleNamespace(max_seq_len=max_seq_len),
        )
    )

    def windows(new, cached, q_begin=0, q_end=None):
        kv = [c + n for c, n in zip(cached, new)]
        md = SimpleNamespace(
            cu_seqlen_ks=torch.empty(0, dtype=torch.int32),
            compress_ratios=[],
            host_ctx_cached_token_indptr=torch.tensor([0, *cached]).cumsum(0),
            host_ctx_kv_indptr=torch.tensor([0, *kv]).cumsum(0),
        )
        planned = plan_mla_chunked_prefill_chunks(new, kv, budget, q_begin, q_end)
        chunks = [Indexer.prepare_one_prefill_chunk(md, specs) for specs in planned]
        return [(c.k_token_start, c.k_token_end - c.k_token_start) for c in chunks]

    shares = []
    for rank in range(TEP8_RANKS):
        mapping = SimpleNamespace(tp_size=TEP8_RANKS, tp_rank=rank, enable_attention_dp=False)
        q_begin, q_end, sizes = _split_prefill_queries(MAX_NUM_TOKENS, mapping, 8192)
        assert sizes == [MAX_NUM_TOKENS // TEP8_RANKS] * TEP8_RANKS, sizes
        shares += windows([MAX_NUM_TOKENS], [max_seq_len - MAX_NUM_TOKENS], q_begin, q_end)
    assert shares == _qsplit_shares([max_seq_len], TEP8_RANKS), shares
    assert shares[0] == (0, 1019904) and shares[3] == (0, 1032192), shares
    packed = windows([5, 5], [983040, 983040])
    assert packed == [(0, 1966090)] and packed[0][1] > max_seq_len, packed
