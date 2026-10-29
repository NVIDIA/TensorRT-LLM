# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cute_dsl_fp8_paged_mqa_logits catalog entry.

The cells are the Mistral Large 4 FP8 target's at its 9216 envelope, the
Mistral Large 4 NVFP4 target's at its 32768 envelope (on 32- and 64-token pages)
and the NVFP4 tp4, tep4 and dep4 targets' at max_seq_len 1048576 with chunked
prefill.
"""

import math
import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm import deep_gemm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_fp8_paged_mqa_logits import (  # noqa: E501
    cute_dsl_fp8_paged_mqa_logits as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "cute_dsl_fp8_paged_mqa_logits requires a CUDA device"

E4M3_MAX = 448.0
#: The schedule block the DSA indexer builds its decode schedule with.
SCHEDULE_BLOCK_KV = 64
#: Every byte of a Mistral Large 4 cell's pool that the step does not own: 0xFF is
#: an e4m3 NaN code and, four at a time, an fp32 NaN scale, so a score read
#: from a page, slot or layer the call has no business reading comes out NaN.
POISON = 0xFF
#: A cell whose sequences need more pages than its pool has (lengths no rank's
#: pool holds at once) keeps this many of the pool's last blocks free, so the
#: stale block-table entries still name poisoned pages.
SHARED_POOL_FREE = 1024

# Gate, in fp32 ulps of the term magnitude. See `_compare`.
ULP = 2.0**-23
ELEMENT_ULP = 8
RMS_ULP = 8

#: Mistral Large 4's index_sharing (params.json's DSA sharing), one letter per layer: layers 0..16
#: run their own indexer, then each 4 layers are 2 'shared' and 2 'full', and
#: layer 71 is full. 44 'F' in all.
_ML4_PATTERN = "F" * 17 + "SSFF" * 13 + "SSF"
assert len(_ML4_PATTERN) == 72 and _ML4_PATTERN.count("F") == 44


def _keys(kv_fused, block_table, b: int, n: int, head_dim: int):
    """Sequence `b`'s `n` keys as fp64 `(k [n, D], k_scale [n])`."""
    page = kv_fused.shape[1]
    pages = block_table[b, : (n + page - 1) // page].long()
    codes, scale = op.unpack_cache(kv_fused[pages], head_dim)
    k = codes.reshape(-1, head_dim)[:n].contiguous().view(torch.float8_e4m3fn).double()
    return k, scale.reshape(-1)[:n].double()


def _magnitude(args) -> torch.Tensor:
    """`M[row, j] = |k_scale[j]| * sum_h |weights[row, h]| * sum_d |q[b, t, h, d] * k[j, d]|`.

    The size of the terms logit `[row, j]` sums, without the cancellation
    between them: what any evaluation's rounding error scales with. Every term
    is non-negative and every fp8 code is exact in fp64. 1 outside the keys.
    """
    q, kv_fused, weights, context_lens, block_table, _, max_context_len = args[:7]
    batch, next_n, _, dim = q.shape
    m = torch.ones(batch * next_n, max_context_len, dtype=torch.float64, device=q.device)
    for b, n in enumerate(context_lens.tolist()):
        k, s = _keys(kv_fused, block_table, b, min(n, max_context_len), dim)
        for t in range(next_n):
            row = b * next_n + t
            terms = q[b, t].float().double().abs() @ k.abs().t()
            weighted = terms * weights[row].double().abs().unsqueeze(1)
            m[row, : k.shape[0]] = weighted.sum(0) * s.abs()
    return m


def _compare(out: torch.Tensor, ref: torch.Tensor, magnitude: torch.Tensor):
    """8 fp32 ulps of each logit's term magnitude `M`, and 8 ulps relative RMS,
    over the causal part of each row only. Returns both measured, in ulps.

    fp8 x fp8 products are exact in fp32, so kernel and fp64 reference differ
    only where the kernel rounds: its tensor-core fp32 sum of 128 products per
    head, relu, its fp32 FMA chain over the 32 weighted heads (four partial
    sums), and the product with the key scale. Each rounding is bounded by the
    magnitude of what it sums, so the element gate is stated in ulps of `M`
    (`_magnitude`) rather than of the logit, whose own size can be one
    cancelling sum. Measured on sm_103 over every cell, rows of 32768 and
    1048576 keys included: at most 0.212 ulp of `M` per element and 0.686 ulp
    relative RMS (about what rounding the output to fp32 costs); the gate is 8
    and 8, 38x and 12x above. A wrong
    evaluation lands far outside, in ulps of `M` per element on a 16-sequence
    batch of every edge length: the op's own fp16 epilogue
    (`epi_dtype=torch.float16`) at 758 and head weights merely rounded to fp16
    at 169 (`test_the_band_rejects_an_fp16_epilogue` keeps both); relu after
    the head sum at 7.6e5, a neighbouring row's head weights at 9.3e5 and no
    key scale at 1.4e8 (probed once, on the same lengths).

    Columns past the causal end are not compared: NaN in `ref` means
    unspecified (`test_columns_past_the_causal_end` pins what they hold).
    """
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert out.shape == ref.shape == magnitude.shape, (out.shape, ref.shape, magnitude.shape)
    live = ~torch.isnan(ref)
    assert bool(torch.isfinite(out[live]).all()), "a causal entry is not finite"
    o = torch.where(live, out, torch.zeros_like(out)).double()
    r = torch.where(live, ref, torch.zeros_like(ref)).double()
    err = float(((o - r).abs() / magnitude.clamp_min(1e-300)).max()) / ULP
    assert err <= ELEMENT_ULP, (
        f"max error {err:.3g} ulp of the term magnitude exceeds {ELEMENT_ULP}"
    )
    rms = float(((o - r) ** 2).sum().sqrt() / (r**2).sum().sqrt().clamp_min(1e-300)) / ULP
    assert rms <= RMS_ULP, f"relative RMS {rms:.3g} ulp exceeds {RMS_ULP}"
    return err, rms


def _schedule(context_lens: torch.Tensor) -> torch.Tensor:
    """The decode schedule, built the way the DSA indexer builds it."""
    return deep_gemm.get_paged_mqa_logits_metadata(
        context_lens.view(-1, 1), SCHEDULE_BLOCK_KV, deep_gemm.get_num_sms()
    )


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """`torch.equal`, 8 bytes at a time where the buffer allows it: its
    elementwise temporary is then an eighth of a 5 GB pool, not the pool."""
    if a.is_contiguous() and b.is_contiguous() and a.numel() % 8 == 0:
        a, b = a.reshape(-1).view(torch.int64), b.reshape(-1).view(torch.int64)
    return torch.equal(a, b)


class _Built(SimpleNamespace):
    """A cell's arguments, the memory behind `kv_fused` the test owns, and how
    to release it. A manager's views must be gone before it shuts down."""

    def close(self) -> None:
        manager = self.manager
        self.args = self.owned = self.manager = None
        if manager is not None:
            manager.shutdown()


def _build_contiguous(spec, seed) -> _Built:
    """`(q, kv_fused, weights, context_lens, block_table, schedule, max_context_len)`
    over a contiguous single-layer cache.

    Every key is quantized per token to e4m3 with an fp32 scale and written to
    a random page, data first and scales after, as the indexer cache stores
    them. Unused block-table entries point at page 0.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    heads, dim, tpb = spec["heads"], spec["head_dim"], spec["tokens_per_block"]
    next_n = spec["next_n"]
    lens = spec["context_lens"]
    batch = len(lens)
    pages = [math.ceil(n / tpb) for n in lens]
    num_pages = sum(pages) + 1
    order = torch.randperm(num_pages, generator=cpu).to(torch.int32)
    table = torch.zeros(batch, max(pages), dtype=torch.int32)
    used = 0
    for b, n in enumerate(pages):
        table[b, :n] = order[used : used + n]
        used += n

    flat = torch.zeros(num_pages, tpb * (dim + 4), dtype=torch.uint8, device="cuda")
    keys = torch.randn(num_pages, tpb, dim, generator=g, device="cuda")
    scale = (keys.abs().amax(-1) / E4M3_MAX).clamp_min(1e-12)
    codes = (keys / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    flat[:, : tpb * dim] = codes.view(torch.uint8).reshape(num_pages, -1)
    flat[:, tpb * dim :] = scale.contiguous().view(torch.uint8).reshape(num_pages, -1)
    kv_fused = flat.view(num_pages, tpb, 1, dim + 4)

    q = torch.randn(batch, next_n, heads, dim, generator=g, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(batch * next_n, heads, generator=g, device="cuda") * (dim**-0.5 * heads**-0.5)
    context_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    args = (
        q,
        kv_fused,
        w.contiguous(),
        context_lens,
        table.cuda(),
        _schedule(context_lens),
        spec["max_context_len"],
    )
    return _Built(args=args, owned=flat, manager=None)


def _ml4_pool(spec):
    """`(parent, kv_fused, page_scale)`: the cell's per-layer view of a pool
    whose every byte is POISON, laid out as the cell's manager lays it out."""
    tpb, dim = spec["tokens_per_block"], spec["head_dim"]
    page = tpb * (dim + 4)
    rows, row, blocks = spec["pool_layers"], spec["pool_row"], spec["num_blocks"]
    if spec["layout"] == "v1":
        parent = torch.full((blocks, rows, page), POISON, dtype=torch.uint8, device="cuda")
        kv_fused = parent[:, row].view(blocks, tpb, 1, dim + 4)
        assert kv_fused.stride(0) == rows * page
        return parent, kv_fused, 1
    parent = torch.full((blocks * rows * page,), POISON, dtype=torch.uint8, device="cuda")
    kv_fused = parent[row * page :].view(blocks * rows - row, tpb, 1, dim + 4)
    return parent, kv_fused, rows


def _ml4_manager(spec):
    """A real DSA cache manager at Mistral Large 4's geometry (72 layers, latent 576
    fp8, the 44-row indexer mask): `(manager, kv_fused, page_table)` with the
    view `get_indexer_k_cache_buffers` hands out and the page indices the DSA
    metadata copies into `indexer_k_cache_block_offsets` (pool slots from
    `get_pool_block_indices`, times `indexer_k_cache_page_scale`). The manager
    is built for `max_seq_len`, the row width unless the cell says."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
        DSACacheManager,
        DSACacheManagerV2,
    )
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.executor import KvCacheConfig as BindingKvCacheConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    v2 = spec["layout"] == "manager_v2"
    # The manager's pool is allocated outside the caching allocator: hand it back
    # what earlier cells' pools left cached (the 1M manager's pool is ~100 GiB
    # in tep4's cell, ~50 GiB in dep4's).
    torch.cuda.empty_cache()
    if v2:
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False, max_tokens=spec["max_tokens"], host_cache_size=0
        )
    else:
        kv_cache_config = BindingKvCacheConfig(
            enable_block_reuse=False, max_tokens=spec["max_tokens"]
        )
    max_seq_len = spec.get("max_seq_len", spec["max_context_len"])
    manager = (DSACacheManagerV2 if v2 else DSACacheManager)(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=72,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=spec["tokens_per_block"],
        max_seq_len=max_seq_len,
        max_batch_size=16,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        sparse_attention_config=DeepSeekSparseAttentionConfig(index_head_dim=spec["head_dim"]),
        pretrained_config=SimpleNamespace(
            num_hidden_layers=72, qk_rope_head_dim=64, index_topk_pattern=list(_ML4_PATTERN)
        ),
        max_num_tokens=spec.get("max_num_tokens", 9216),
    )
    try:
        assert manager.max_seq_len == max_seq_len, (manager.max_seq_len, max_seq_len)
        assert sum(manager.indexer_k_cache_local_layer_mask) == spec["pool_layers"]
        assert manager.indexer_k_cache_page_scale == (spec["pool_layers"] if v2 else 1)
        assert manager.max_blocks_per_seq == spec["table_width"], manager.max_blocks_per_seq
        lens = spec["context_lens"]
        ids = list(range(len(lens)))
        # A manager refuses a population past its `max_tokens` quota by returning None.
        added = manager.add_dummy_requests(ids, lens, is_gen=False, prepare_resource=True)
        assert added is not None, f"the manager refused {lens} at max_tokens {spec['max_tokens']}"
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
        page_table = (blocks.long() * manager.indexer_k_cache_page_scale).to(torch.int32)
        kv_fused = manager.get_indexer_k_cache_buffers(spec["layer"])
        assert kv_fused.is_contiguous() == v2
        print(
            f"\n[{spec['layout']}] bytes per token {manager.get_cache_bytes_per_token()}, "
            f"view {tuple(kv_fused.shape)} stride {kv_fused.stride()}, "
            f"max_seq_len {manager.max_seq_len}, table {tuple(page_table.shape)}",
            flush=True,
        )
    except BaseException:
        manager.shutdown()
        raise
    return manager, kv_fused, page_table


def _write_keys(kv_fused, pages, n: int, gen) -> None:
    """`n` random keys, quantized per token to e4m3 with an fp32 scale, written
    token `i` to page `pages[i // page]`, slot `i % page`: a page holds its
    tokens' data first, then their scales, as the indexer cache stores them."""
    tpb, dim = kv_fused.shape[1], kv_fused.shape[3] - 4
    keys = torch.randn(n, dim, generator=gen, device="cuda")
    scale = (keys.abs().amax(-1) / E4M3_MAX).clamp_min(1e-12)
    codes = (keys / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8)
    flat = kv_fused.view(kv_fused.shape[0], -1)
    tok = torch.arange(n, device="cuda")
    page = torch.as_tensor(pages, device="cuda").long()[tok // tpb].unsqueeze(1)
    slot = (tok % tpb).unsqueeze(1)
    flat[page, slot * dim + torch.arange(dim, device="cuda")] = codes
    flat[page, tpb * dim + slot * 4 + torch.arange(4, device="cuda")] = scale.view(
        torch.uint8
    ).view(n, 4)


def _fill_sw(built: _Built, spec, lens, seed: int) -> None:
    """Write one decode step into the cell's buffers in place: keys into a
    poisoned pool, the block table's rows, the lengths, the query and weight
    rows, and the schedule. Block-table entries past a sequence's pages name
    pages of blocks the step leaves free (poisoned), as the engine leaves stale
    in-range entries there; padding rows repeat the dummy sequence's row and
    length. A manager cell's table is the manager's, as the metadata copies it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    tpb, heads, dim = spec["tokens_per_block"], spec["heads"], spec["head_dim"]
    q, kv_fused, weights, context_lens, block_table, schedule, _ = built.args
    pad = spec.get("padding", 0)
    live = len(lens) - pad
    seqs = list(lens[:live]) + list(lens[live : live + 1])
    need = [math.ceil(n / tpb) for n in seqs]
    if built.manager is not None:
        page_lists = [built.page_table[b, :k] for b, k in enumerate(need)]
        kv_fused.fill_(POISON)
        table = torch.zeros_like(built.table)
        table[: built.page_table.shape[0], : built.page_table.shape[1]] = built.page_table
        table.clamp_(min=0)
    else:
        blocks, scale = spec["num_blocks"], built.page_scale
        if sum(need) <= blocks:
            if spec.get("tail"):
                ids = torch.arange(blocks - sum(need), blocks)
                ids = ids[torch.randperm(sum(need), generator=cpu)]
            else:
                ids = torch.randperm(blocks, generator=cpu)[: sum(need)]
        else:
            # Each sequence draws distinct pages from all but the pool's last
            # SHARED_POOL_FREE blocks, so sequences share pages; a shared page
            # holds the later sequence's keys, which the call and the reference
            # both read.
            usable = blocks - SHARED_POOL_FREE
            assert not spec.get("tail") and max(need) <= usable, (need, blocks)
            ids = torch.cat([torch.randperm(usable, generator=cpu)[:k] for k in need])
        page_lists = [(p * scale).to(torch.int32) for p in torch.split(ids, need)]
        built.owned.fill_(POISON)
        free = torch.ones(blocks, dtype=torch.bool)
        free[ids] = False
        free = free.nonzero().flatten()
        assert free.numel(), "no free block left for the stale entries"
        stale = torch.randint(0, free.numel(), tuple(built.table.shape), generator=cpu)
        table = (free[stale] * scale).to(torch.int32)
    first = spec.get("first_seq", 0)
    for b in range(len(lens)):
        pages = page_lists[min(b, live)]
        table[first + b, : pages.numel()] = pages
    built.table.copy_(table)
    for pages, n in zip(page_lists, seqs):
        _write_keys(kv_fused, pages.tolist(), n, g)
    context_lens.copy_(torch.tensor(lens, dtype=torch.int32))
    q.copy_(torch.randn(q.shape, generator=g, device="cuda").to(torch.float8_e4m3fn))
    weights.copy_(
        torch.randn(weights.shape, generator=g, device="cuda") * (dim**-0.5 * heads**-0.5)
    )
    schedule.copy_(_schedule(context_lens))


def _rows(spec, lens) -> list:
    """The call's rows: `lens`, then under `padded_to` the cell's `padding` rows
    of a 1-key dummy, as a padded graph batch carries them after its live rows.
    Without `padded_to`, `lens` already ends in its padding rows."""
    if "padded_to" not in spec:
        return list(lens)
    assert len(lens) + spec["padding"] == spec["padded_to"], (len(lens), spec["padded_to"])
    return list(lens) + [1] * spec["padding"]


def _build_sw(spec, seed) -> _Built:
    """Mistral Large 4's cells, in buffers shaped the way the engine hands them
    over: block-table, length, query and weight rows sliced out of larger
    buffers (`table_rows`, `first_seq`, `q_offset`), the table the manager's
    `max_blocks_per_seq` wide (`table_width`), and the cache a per-layer view
    of a pool."""
    heads, dim, tpb, next_n = (
        spec["heads"],
        spec["head_dim"],
        spec["tokens_per_block"],
        spec["next_n"],
    )
    batch = len(_rows(spec, spec["context_lens"]))
    rows = spec.get("table_rows", batch)
    first = spec.get("first_seq", 0)
    q_off = spec.get("q_offset", 0)
    manager = page_table = None
    page_scale = 1
    if spec["layout"].startswith("manager"):
        manager, kv_fused, page_table = _ml4_manager(spec)
        owned = kv_fused
    else:
        owned, kv_fused, page_scale = _ml4_pool(spec)
    assert spec["table_width"] * tpb >= spec["max_context_len"]
    table = torch.zeros(rows, spec["table_width"], dtype=torch.int32, device="cuda")
    lens = torch.zeros(rows, dtype=torch.int32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(seed ^ 0x5A5A)
    q_rows = torch.randn(q_off + batch * next_n, heads, dim, generator=g, device="cuda")
    q_rows = q_rows.to(torch.float8_e4m3fn)
    w_rows = torch.randn(q_off + batch * next_n, heads, generator=g, device="cuda")
    schedule = torch.zeros(deep_gemm.get_num_sms() + 1, 2, dtype=torch.int32, device="cuda")
    args = (
        q_rows[q_off:].view(batch, next_n, heads, dim),
        kv_fused,
        w_rows[q_off:],
        lens[first : first + batch],
        table[first : first + batch],
        schedule,
        spec["max_context_len"],
    )
    built = _Built(
        args=args,
        owned=owned,
        manager=manager,
        page_table=page_table,
        page_scale=page_scale,
        table=table,
    )
    try:
        _fill_sw(built, spec, _rows(spec, spec["context_lens"]), seed)
    except BaseException:
        built.close()
        raise
    return built


def _last_page_byte(built: _Built) -> int:
    """The byte offset, into the pool, of the last byte of the highest page the
    call's block-table rows name for its sequences' keys."""
    q, kv_fused, _, context_lens, block_table, *_ = built.args
    tpb = kv_fused.shape[1]
    top = max(
        int(block_table[b, : (n + tpb - 1) // tpb].max())
        for b, n in enumerate(context_lens.tolist())
    )
    base = kv_fused.data_ptr() - built.owned.data_ptr()
    return base + top * kv_fused.stride(0) + tpb * kv_fused.shape[3] - 1


def _cell_ids(cells) -> list:
    """Each cell's test id, `why[:44]`; a cell whose 44-character prefix an
    earlier cell already took (three of tep4's 1M cells share one with tp4's)
    is named by its whole `why`, so no id repeats and no earlier id changes.
    The seed is `crc32(why)` either way."""
    ids = []
    for c in cells:
        ids.append(c.why if c.why[:44] in ids else c.why[:44])
    assert len(set(ids)) == len(ids), ids
    return ids


@pytest.mark.parametrize(
    "cell",
    [pytest.param(c, marks=pytest.mark.large) if c.spec.get("large") else c for c in op.CELLS],
    ids=_cell_ids(op.CELLS),
)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    built = _build_sw(spec, seed)
    graph = captured = None
    try:
        args = built.args
        if "capture_lens" in spec:
            # Captured where the engine captures it, at other lengths, tables
            # and keys; the replay then runs on this cell's step.
            _fill_sw(built, spec, _rows(spec, spec["capture_lens"]), seed + 1)
            op(*args)  # compile outside the capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = op(*args)
            _fill_sw(built, spec, _rows(spec, spec["context_lens"]), seed)
        before = built.owned.clone()
        if graph is not None:
            graph.replay()
            out = captured.clone()
            eager = op(*args)
            assert eager.data_ptr() == captured.data_ptr(), "the replay and eager call share it"
            eager = eager.clone()
        else:
            out = op(*args)
        assert _same_bytes(built.owned, before), "scoring modified the cache"
        del before
        batch, next_n = args[0].shape[:2]
        assert tuple(out.shape) == (batch * next_n, args[6])
        ref = op.reference(*args)
        if graph is not None:
            live = ~torch.isnan(ref)
            assert torch.equal(out.view(torch.int32)[live], eager.view(torch.int32)[live]), (
                "the replay's causal logits are not bitwise those of an eager call"
            )
        err, rms = _compare(out, ref, _magnitude(args))
        print(f"\n[{cell.why[:44]}] max {err:.3f} ulp of M, RMS {rms:.3f} ulp", flush=True)
        if "past" in spec:
            last = _last_page_byte(built)
            print(f"last byte of the highest page read: {last}", flush=True)
            assert last >= spec["past"], (last, spec["past"])
    finally:
        # The cache view goes before a manager shuts down.
        args = graph = captured = None
        built.close()


def test_the_result_is_a_shared_buffer() -> None:
    """Two calls of one shape return the same storage, and the second call
    overwrites the first result. A caller that keeps a result across calls
    reads the later scores without an error."""
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=1,
        context_lens=[500, 300],
        max_context_len=1024,
    )
    first_args = _build_contiguous(spec, seed=21).args
    second_args = _build_contiguous(spec, seed=22).args
    first = op(*first_args)
    _compare(first, op.reference(*first_args), _magnitude(first_args))
    second = op(*second_args)
    assert first.data_ptr() == second.data_ptr()
    _compare(first, op.reference(*second_args), _magnitude(second_args))


@pytest.mark.parametrize(
    "max_context_len, table_width, num_blocks, context_lens",
    [
        (9216, 292, 1024, [1, 33, 255, 256, 257, 8960, 8961, 9216]),
        (32768, 1028, 8192, [1, 257, 16384, 32512, 32513, 32767, 32768]),
        (1048576, 32772, 74183, [1, 257, 1048320, 1048321]),
        (1048576, 32772, 74157, [1, 257, 32513, 1048321, 1048576]),
    ],
    ids=["9216", "32768", "1048576_tp4", "1048576"],
)
def test_columns_past_the_causal_end(max_context_len, table_width, num_blocks, context_lens):
    """For sequence b the kernel writes every column below
    ceil(context_lens[b] / 256) * 256 and none after it: the scores of the
    poisoned slots and pages past the last key come out NaN, and the rest of
    the row keeps what the buffer held. The decode top-k reads none of it;
    this pins what the note says about it, on rows of every Mistral Large 4
    width (the 1048576 row on tp4's 74183-block boot pool, `1048576_tp4`, and
    on the delivered 74157-block pool)."""
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=1,
        max_context_len=max_context_len,
        layout="v2",
        pool_layers=44,
        pool_row=5,
        num_blocks=num_blocks,
        table_width=table_width,
        context_lens=context_lens,
    )
    built = _build_sw(spec, seed=26)
    try:
        sentinel = 12345.0
        op(*built.args).fill_(sentinel)
        out = op(*built.args)
        for b, n in enumerate(spec["context_lens"]):
            written = out[b] != sentinel
            end = -(-n // 256) * 256
            assert bool(written[:end].all()) and not bool(written[end:].any()), (n, end)
            assert bool(torch.isfinite(out[b, :n]).all())
            assert n == end or bool(torch.isnan(out[b, n]))
    finally:
        built.close()


@pytest.mark.parametrize("next_n,width", [(1, 1048576), (4, 1048584)])
def test_a_1m_row_reads_its_last_page_through_the_table(next_n, width) -> None:
    """Each causal row reads its last page through the table. Redirecting that
    page to poisoned or unrelated keys must fail the same accuracy gate."""
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=next_n,
        max_context_len=width,
        layout="v2",
        pool_layers=44,
        pool_row=43,
        num_blocks=33792,
        table_width=32772,
        context_lens=[width],
    )
    built = _build_sw(spec, seed=28)
    try:
        args = built.args
        table = args[4]
        ref, mag = op.reference(*args), _magnitude(args)
        good = op(*args).clone()
        print(f"\nas built: {_compare(good, ref, mag)}", flush=True)
        last_page = (width - 1) // 32
        page_start = last_page * 32
        for column, refusal in ((last_page + 1, "not finite"), (0, "ulp")):
            table[0, last_page] = table[0, column]
            out = op(*args).clone()
            assert torch.equal(out[:, :page_start], good[:, :page_start]), (
                "only the last page moved"
            )
            for row in range(next_n):
                end = width - next_n + row + 1
                assert bool(torch.isnan(out[row, page_start:end]).all()) == (
                    refusal == "not finite"
                )
            with pytest.raises(AssertionError, match=refusal) as refused:
                _compare(out, ref, mag)
            print(f"last page from column {column}: {refused.value}", flush=True)
    finally:
        built.close()


def test_the_band_rejects_an_fp16_epilogue() -> None:
    """The gate's discrimination, kept: on the 16-sequence edge batch the op's
    own fp16 epilogue and head weights rounded to fp16 each land outside the
    band that the fp32 call sits well inside."""
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=1,
        max_context_len=9216,
        layout="v2",
        pool_layers=44,
        pool_row=3,
        num_blocks=2048,
        table_width=292,
        context_lens=[1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 2049, 9215, 9216],
    )
    built = _build_sw(spec, seed=27)
    try:
        args = built.args
        ref, mag = op.reference(*args), _magnitude(args)
        print(f"\nfp32 epilogue: {_compare(op(*args).clone(), ref, mag)}", flush=True)
        fp16_epilogue = op(*args, 1, torch.float16).clone()
        with pytest.raises(AssertionError, match="ulp") as refused:
            _compare(fp16_epilogue, ref, mag)
        print(f"fp16 epilogue: {refused.value}", flush=True)
        q, kv, w, *rest = args
        fp16_weights = op(q, kv, w.half().float(), *rest).clone()
        with pytest.raises(AssertionError, match="ulp") as refused:
            _compare(fp16_weights, ref, mag)
        print(f"fp16 head weights: {refused.value}", flush=True)
    finally:
        built.close()


def _ok():
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=1,
        context_lens=[100, 64],
        max_context_len=256,
    )
    return _build_contiguous(spec, seed=25).args


def test_guard_refuses_a_schedule_for_another_sm_count() -> None:
    """Shown first without the guard: 16 sequences of 9216 keys scored with a
    schedule built for twice the SMs. The op launches one CTA per SM and each
    takes one schedule partition, so the partitions past the SM count are never
    scored: those logits keep what the buffer held, and nothing raises. A
    schedule for fewer SMs makes the last CTAs read rows past its end and
    faults the context (an illegal memory access, measured in a separate
    process), so it is only refused here."""
    spec = dict(
        heads=32,
        head_dim=128,
        tokens_per_block=32,
        next_n=1,
        context_lens=[9216] * 16,
        max_context_len=9216,
    )
    q, kv, w, lens, table, schedule, max_len = _build_contiguous(spec, seed=24).args
    wide = deep_gemm.get_paged_mqa_logits_metadata(
        lens.view(-1, 1), SCHEDULE_BLOCK_KV, 2 * deep_gemm.get_num_sms()
    )
    op(q, kv, w, lens, table, schedule, max_len).fill_(float("nan"))
    out = op(q, kv, w, lens, table, wide, max_len)
    unwritten = int(torch.isnan(out).sum())
    print(f"\nunwritten causal logits with a {tuple(wide.shape)} schedule: {unwritten}", flush=True)
    assert 0 < unwritten < out.numel(), unwritten

    for rows in (schedule.shape[0] - 1, schedule.shape[0] + 1, wide.shape[0]):
        bad = torch.zeros(rows, 2, dtype=torch.int32, device="cuda")
        with pytest.raises(AssertionError, match="SM count \\+ 1 rows"):
            with validating(op):
                op(q, kv, w, lens, table, bad, max_len)


def test_guard_refuses_a_context_past_max_context_len() -> None:
    """Shown first without the guard: a 100-key context scored into a row of
    64 columns. The 64 columns hold keys 0..63's scores, right to the band;
    keys 64..99 were written into the row's padding (rows are 256 columns wide
    here) and are gone, so a top-k over the row can never select them."""
    args = _ok()
    q, kv, w, lens, table, schedule, _ = args
    full = op.reference(*args)
    out = op(q, kv, w, lens, table, schedule, 64)
    assert tuple(out.shape) == (2, 64)
    _compare(out, full[:, :64], _magnitude(args)[:, :64])
    assert not bool(torch.isnan(full[0, 64:100]).any()), "keys 64..99 have scores"

    with pytest.raises(AssertionError, match="exceeds max_context_len"):
        with validating(op):
            op(q, kv, w, lens, table, schedule, 64)
