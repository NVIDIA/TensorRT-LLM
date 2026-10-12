# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the indexer_k_cache_scatter catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and `compare` are the gate.
This file turns a cell's spec into tensors and keeps the checks a cell cannot carry: the
reader-layout placement and decode, a live cache manager's pools and offsets, the wrong-layout
controls, the op's rejections and the guards, the engine-size pools, every row count, the CUDA-graph
capture sets, and pools up to the largest any DeepSeek-V4.1 cache manager can have.

The DeepSeek-V4.1 index-K write path: packed MXFP4 keys go into the owner layer's native
INDEXER_COMPRESS pool, uint8 ``[pages, block_kv, 1, 68]``, whose page holds ``block_kv x 64``
code bytes followed by ``block_kv x 4`` scale bytes (the layout the DeepGEMM paged logits kernel
reads). The inputs are produced by the target's own quantizer, ``torch.ops.trtllm.fp4_quantize(x,
None, 32, True, False)``, whose linear scale output is viewed as int32 ``[T, 1]``. Beside the
entry's flat-offset reference, the expected pool is also built from page and slot with native torch
indexing (the pre-call snapshot with exactly those bytes placed at the paged offsets), and a
reader-layout decode of the pool must give back the quantizer's values.
"""

import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.indexer_k_cache_scatter import (  # noqa: E501
    indexer_k_cache_scatter as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "indexer_k_cache_scatter requires a CUDA device"

DEV = "cuda"
DIM = 128  # index_head_dim
CODE_BYTES = 64  # two E2M1 codes per byte
SCALE_BYTES = 4  # one UE8M0 byte per 32 elements
ENTRY_BYTES = CODE_BYTES + SCALE_BYTES
# The pool's page count is the cache manager's number, never the batch's: each case allocates
# POOL_FACTOR x the pages its batch needs (+ POOL_EXTRA) and hands its requests pages drawn by a
# random permutation. Chosen here, not taken from a runtime dump.
POOL_FACTOR = 3
POOL_EXTRA = 8

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64,
    device=DEV,
)


def _keys(n: int, seed: int):
    """bf16 [n, 128] with a different power-of-two magnitude per 32-block, quantized by the
    target's quantizer -> (codes uint8 [n, 64], scale bytes uint8 [n, 4])."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(n, DIM, device=DEV, generator=g)
    mag = torch.randint(-8, 9, (n, DIM // 32, 1), device=DEV, generator=g).float()
    x = (x.view(n, DIM // 32, 32) * torch.exp2(mag)).view(n, DIM).to(torch.bfloat16)
    codes, sf = torch.ops.trtllm.fp4_quantize(x, None, 32, True, False)
    assert codes.shape == (n, CODE_BYTES) and codes.dtype == torch.uint8
    assert sf.shape == (n * SCALE_BYTES,) and sf.dtype == torch.uint8
    return codes, sf.view(n, SCALE_BYTES)


def _as_scale_word(scale_bytes: torch.Tensor) -> torch.Tensor:
    """uint8 [n, 4] -> int32 [n, 1], a reinterpretation (byte g of each row = byte g of the word)."""
    return scale_bytes.contiguous().view(torch.int32).view(-1, 1)


def _decode(codes: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
    """(uint8 [n, 64], uint8 [n, 4]) -> fp64 [n, 128]; element 2i is the low nibble of byte i,
    scale byte g multiplies elements 32g .. 32g+31 by 2^(byte - 127)."""
    v = torch.stack([_E2M1[(codes & 0xF).long()], _E2M1[(codes >> 4).long()]], -1).flatten(-2)
    sc = torch.exp2(scale_bytes.double() - 127.0)
    return (v.unflatten(-1, (4, 32)) * sc.unsqueeze(-1)).flatten(-2)


class _Batch:
    """Requests with ``cached[b]`` entries already in the pool and ``new[b]`` entries to write."""

    def __init__(self, block_kv: int, cached, new, seed: int, num_pages=None):
        g = torch.Generator().manual_seed(seed)
        self.block_kv = block_kv
        cached = torch.tensor(cached, dtype=torch.int64)
        new = torch.tensor(new, dtype=torch.int64)
        pages = ((cached + new + block_kv - 1) // block_kv).clamp_min(1)
        need = int(pages.sum())
        self.num_pages = POOL_FACTOR * need + POOL_EXTRA if num_pages is None else num_pages
        assert self.num_pages >= need
        perm = torch.randperm(self.num_pages, generator=g)
        self.table = torch.full((len(cached), int(pages.max())), -1, dtype=torch.int64)
        o = 0
        for b, p in enumerate(pages.tolist()):
            self.table[b, :p] = perm[o : o + p]
            o += p
        self.n = int(new.sum())
        self.req = torch.repeat_interleave(torch.arange(len(cached)), new)
        start = torch.cumsum(new, 0) - new
        self.pos = cached[self.req] + torch.arange(self.n) - start[self.req]
        self.table = self.table.to(DEV)
        self.req = self.req.to(DEV)
        self.pos = self.pos.to(DEV)

    def pages_slots(self, table=None, block_kv=None):
        table = self.table if table is None else table
        block_kv = self.block_kv if block_kv is None else block_kv
        page = table[self.req, self.pos // block_kv]
        return page, self.pos % block_kv

    def slot_mappings(self, table=None, block_kv=None):
        """The in-tree formula (``_compute_slot_mappings`` in the DSA indexer)."""
        block_kv = self.block_kv if block_kv is None else block_kv
        page, slot = self.pages_slots(table, block_kv)
        base = page * block_kv * ENTRY_BYTES
        return base + slot * CODE_BYTES, base + block_kv * CODE_BYTES + slot * SCALE_BYTES

    def pool(self, seed: int, extra_pages: int = 0) -> torch.Tensor:
        g = torch.Generator(device=DEV).manual_seed(seed)
        return torch.randint(
            0,
            256,
            (self.num_pages + extra_pages, self.block_kv, 1, ENTRY_BYTES),
            dtype=torch.uint8,
            device=DEV,
            generator=g,
        )

    def expected(self, before, codes, scale_bytes, rows=None):
        """Native-torch placement: the snapshot with exactly these rows' bytes at their offsets."""
        rows = torch.arange(self.n, device=DEV) if rows is None else rows
        page, slot = self.pages_slots()
        page, slot = page[rows], slot[rows]
        exp = before.clone().view(before.size(0), -1)
        ar_c = torch.arange(CODE_BYTES, device=DEV)
        ar_s = torch.arange(SCALE_BYTES, device=DEV)
        exp[page.unsqueeze(1), (slot * CODE_BYTES).unsqueeze(1) + ar_c] = codes[rows]
        exp[
            page.unsqueeze(1), (self.block_kv * CODE_BYTES + slot * SCALE_BYTES).unsqueeze(1) + ar_s
        ] = scale_bytes[rows]
        return exp.view_as(before)

    def read(self, pool: torch.Tensor):
        """The DeepGEMM paged reader's view of every token: its page's code bytes, then scales."""
        page, slot = self.pages_slots()
        flat = pool.reshape(pool.size(0), -1) if pool.is_contiguous() else None
        if flat is None:  # page-strided view: address pages through stride(0)
            flat = torch.as_strided(
                pool, (pool.size(0), self.block_kv * ENTRY_BYTES), (pool.stride(0), 1)
            )
        codes = flat[
            page.unsqueeze(1),
            (slot * CODE_BYTES).unsqueeze(1) + torch.arange(CODE_BYTES, device=DEV),
        ]
        sb = flat[
            page.unsqueeze(1),
            (self.block_kv * CODE_BYTES + slot * SCALE_BYTES).unsqueeze(1)
            + torch.arange(SCALE_BYTES, device=DEV),
        ]
        return codes, sb


def _row_mismatch(batch: _Batch, pool, codes, scale_bytes) -> float:
    rc, rs = batch.read(pool)
    bad = (_decode(rc, rs) != _decode(codes, scale_bytes)).any(-1)
    return bad.float().mean().item()


def _cell_batch(spec, seed: int) -> _Batch:
    """A cell's requests: explicit (cached, new) lists, or `requests` drawn below `cached_below` /
    up to `new_upto` entries (the random decode and DSpark-verify batches)."""
    g = torch.Generator().manual_seed(seed)
    reqs = spec.get("requests")
    cached = (
        spec["cached"]
        if "cached" in spec
        else torch.randint(0, spec["cached_below"], (reqs,), generator=g).tolist()
    )
    new = (
        spec["new"]
        if "new" in spec
        else torch.randint(0, spec["new_upto"] + 1, (reqs,), generator=g).tolist()
    )
    return _Batch(spec["block_kv"], cached, new, seed=seed)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode())
    b = _cell_batch(spec, seed)
    bk = b.block_kv
    codes, sb = _keys(b.n, seed=seed + 1)
    d_map, s_map = b.slot_mappings()
    skip = torch.zeros(b.n, dtype=torch.bool, device=DEV)
    if spec.get("skip"):
        skip[::5] = True
        if spec["skip"] in ("both", "data_only"):
            d_map = torch.where(skip, -1, d_map)
        if spec["skip"] in ("both", "scale_only"):
            s_map = torch.where(skip, -1, s_map)
    pad = spec.get("page_pad", 0)
    page_bytes = bk * ENTRY_BYTES
    if pad:
        # A view into a wider allocation: page stride block_kv * 68 + pad.
        backing = torch.randint(
            0, 256, (b.num_pages, page_bytes + pad), dtype=torch.uint8, device=DEV
        )
        pool = backing[:, :page_bytes].view(b.num_pages, bk, 1, ENTRY_BYTES)
        assert not pool.is_contiguous() and pool.stride(0) == page_bytes + pad
        backing_before = backing.clone()
    else:
        pool = b.pool(seed=seed + 2)
    n = b.n if spec.get("num_tokens") is None else spec["num_tokens"]
    word = _as_scale_word(sb)
    before = pool.clone(memory_format=torch.contiguous_format)

    out = op(codes, word, pool, d_map, s_map, n)
    torch.cuda.synchronize()
    assert out is pool, "the op writes the pool in place and returns it"
    op.compare(out, op.reference(codes, word, before, d_map, s_map, n))

    # The paged layout the reader expects: the snapshot with exactly the written rows' bytes placed
    # at their page and slot (every other byte of the pool untouched).
    rows = (~skip[:n]).nonzero().flatten()
    assert torch.equal(pool, b.expected(before, codes, sb, rows=rows))
    if pad:
        assert torch.equal(backing[:, page_bytes:], backing_before[:, page_bytes:])
    if rows.numel() == b.n:
        # the reader-layout decode of the pool is the quantizer's value set, row for row
        rc, rs = b.read(pool)
        assert torch.equal(_decode(rc, rs), _decode(codes, sb))
    print(f"[scatter {cell.why}] tokens={b.n} n={n} pages={b.num_pages} bit-exact")


def test_cuda_graph_replay():
    """Capture once, then replay with fresh keys and fresh slot offsets written into the static
    buffers (the decode / DSpark-verify step is graph-captured); padding rows carry -1."""
    n = 48
    pages = 200  # the pool outlives every batch replayed into it
    b0 = _Batch(64, [0] * 16, [3] * 16, seed=12, num_pages=pages)
    pool = b0.pool(seed=13)
    codes_s = torch.zeros(n, CODE_BYTES, dtype=torch.uint8, device=DEV)
    scale_s = torch.zeros(n, 1, dtype=torch.int32, device=DEV)
    d_s = torch.full((n,), -1, dtype=torch.int64, device=DEV)
    s_s = torch.full((n,), -1, dtype=torch.int64, device=DEV)
    op(codes_s, scale_s, pool, d_s, s_s, n)  # warm-up (all rows skipped)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(codes_s, scale_s, pool, d_s, s_s, n)
    torch.cuda.synchronize()
    for trial, (cached, new) in enumerate(
        (([5 * t for t in range(16)], [3] * 16), ([64 * t + 63 for t in range(12)], [2] * 12))
    ):
        b = _Batch(64, cached, new, seed=20 + trial, num_pages=pages)
        codes, sb = _keys(b.n, seed=30 + trial)
        d_map, s_map = b.slot_mappings()
        codes_s.zero_()
        scale_s.zero_()
        d_s.fill_(-1)
        s_s.fill_(-1)
        codes_s[: b.n] = codes
        scale_s[: b.n] = _as_scale_word(sb)
        d_s[: b.n] = d_map
        s_s[: b.n] = s_map
        before = pool.clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(pool, b.expected(before, codes, sb)), f"replay {trial}"


# ─── the real cache manager's pools and the metadata's own offsets ───────────────────────────

# (cached, new) per request, context requests first; every request here feeds its step as the
# target sees it. prefill: fresh prompts of odd and even length (1 emits no ratio-2 entry);
# mixed: a fresh and a chunked context (chunk starting mid-page at 513) ahead of four decode rows
# of both parities, one ending on a page edge; decode_b32: the target's widest decode graph batch.
_V41_DECODE32 = torch.randint(
    1, 400, (28,), generator=torch.Generator().manual_seed(77)
).tolist() + [127, 128, 2047, 2048]
_V41_STEPS = {
    "prefill": ([(0, 1), (0, 130), (0, 257), (0, 700), (0, 512)], 5),
    "mixed": ([(0, 300), (513, 388), (129, 1), (2048, 1), (63, 1), (1000, 1)], 2),
    "decode_b32": ([(c, 1) for c in _V41_DECODE32], 0),
}
_V41_MAX_BATCH = 40
_V41_MAX_TOKENS = 4096 * _V41_MAX_BATCH * 2


@pytest.mark.parametrize("step", list(_V41_STEPS))
def test_v41_cache_manager_pools(step):
    """The target's call on the pools a live DeepseekV4CacheManager allocates (V4.1 mode,
    tokens_per_block 128, max_batch_size 40, max_seq_len 4096, max_tokens 327680, pools
    uint8 [6720, 64, 1, 68] for ratio 2 and [2240, 128, 1, 68] for ratio 1), with the offsets
    of the prepared DeepseekV4TrtllmAttentionMetadata: ``v41_index_k_slot_data[r][:n]`` /
    ``v41_index_k_slot_scale[r][:n]`` and ``n = num_total_compressed_tokens[r]``, reserved rows
    included. Owners 2, 8 and 14 share page ids and interleave their pages in one allocation
    (each owner's view starts one page after the previous one's), so every ratio-2 call is
    checked over the whole allocation. The expected bytes are placed from the reader's table,
    ``indexer_compress_block_tables[r]``, and the packed-row order (context entries, then
    generation entries, in request order) with native torch indexing; every other byte of the
    allocation, and the same owner's COMPRESS pool, must be unchanged."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
        DeepseekV4CacheManager,
        DeepseekV4TrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as T,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    batch, nc = _V41_STEPS[step]
    text = dict(
        model_type="deepseek_v41_text",
        num_hidden_layers=40,
        num_nextn_predict_layers=3,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        sliding_window=128,
        head_dim=512,
        index_head_dim=DIM,
        index_topk=512,
        kv_lora_rank=448,
        qk_rope_head_dim=64,
    )
    config = SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        text_config=SimpleNamespace(**text),
        **{k: v for k, v in text.items() if k != "model_type"},
    )
    sparse = DeepSeekV4SparseAttentionConfig(
        index_head_dim=DIM, window_size=128, compress_ratios=[1] * 43
    )
    manager = DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False, max_tokens=_V41_MAX_TOKENS, event_buffer_max_size=0
        ),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=40,
        num_kv_heads=1,
        head_dim=512,
        tokens_per_block=128,
        max_seq_len=4096,
        max_batch_size=_V41_MAX_BATCH,
        max_input_len=4096,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        compressor_dtype=DataType.FLOAT,
        vocab_size=129280,
        max_num_tokens=8192,
        sparse_attn_config=sparse,
        pretrained_config=config,
    )
    requests = []
    try:
        cached = [c for c, _ in batch]
        new = [n for _, n in batch]
        B = len(batch)
        for i, (c, n) in enumerate(batch):
            req = LlmRequest(
                request_id=i,
                max_new_tokens=64,
                input_tokens=list(range(c + n)),
                sampling_config=SamplingConfig(),
                is_streaming=False,
            )
            assert manager.prepare_context(req), f"manager could not hold request {i}"
            assert manager.resize_context(req, req.context_chunk_size)
            requests.append(req)
        md = DeepseekV4TrtllmAttentionMetadata(
            seq_lens=torch.tensor(new, dtype=torch.int),
            request_ids=list(range(B)),
            max_num_requests=manager.max_batch_size,
            num_contexts=nc,
            prompt_lens=[c + n for c, n in batch],
            max_num_tokens=8192,
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(cached)),
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            sparse_attention_config=sparse,
        )
        md.prepare()
        torch.cuda.synchronize()
        assert (md.num_contexts, md.num_seqs) == (nc, B)

        for r, owners in ((2, (2, 8, 14)), (1, (20,))):
            block_kv = 128 // r
            n = md.num_total_compressed_tokens[r]
            # Packed rows: every (request, entry) the step completes, then reserved rows (-1).
            rows = [(b, k) for b, (c, m) in enumerate(batch) for k in range(c // r, (c + m) // r)]
            assert 0 < len(rows) <= n, (step, r, len(rows), n)
            d_map = md.v41_index_k_slot_data[r][:n]
            s_map = md.v41_index_k_slot_scale[r][:n]
            assert (d_map[len(rows) :] == -1).all() and (s_map[len(rows) :] == -1).all()
            assert (d_map[: len(rows)] >= 0).all() and (s_map[: len(rows)] >= 0).all()
            table = md.indexer_compress_block_tables[r][:B].long()
            req = torch.tensor([b for b, _ in rows], device=DEV)
            ent = torch.tensor([k for _, k in rows], device=DEV)
            page = table[req, ent // block_kv]
            slot = ent % block_kv
            assert (page >= 0).all()
            # The whole allocation the ratio's owners interleave in: the first owner's view reaches
            # the same end address as every other owner's.
            first = manager.get_indexer_k_cache_buffers(owners[0])
            ends = {
                manager.get_indexer_k_cache_buffers(o).data_ptr()
                + manager.get_indexer_k_cache_buffers(o).numel()
                for o in owners
            }
            assert ends == {first.data_ptr() + first.numel()}
            first.copy_(
                torch.randint(
                    0,
                    256,
                    first.shape,
                    dtype=torch.uint8,
                    device=DEV,
                    generator=torch.Generator(device=DEV).manual_seed(r),
                )
            )
            for owner in owners:
                pool = manager.get_indexer_k_cache_buffers(owner)
                assert pool.dtype == torch.uint8 and tuple(pool.shape[1:]) == (
                    block_kv,
                    1,
                    ENTRY_BYTES,
                )
                assert pool.is_contiguous()
                page_bytes = block_kv * ENTRY_BYTES
                shift, rem = divmod(pool.data_ptr() - first.data_ptr(), page_bytes)
                assert rem == 0 and 0 <= shift < len(owners), (owner, shift, rem)
                comp = manager.get_buffers(owner, T.COMPRESS)
                comp_before = comp.clone()
                before = first.clone()
                # What the target passes: fp4_quantize's data as is, its linear scales as int32 [n, 1].
                codes, sb = _keys(n, seed=1000 * r + owner)
                word = sb.reshape(-1).view(torch.int32).reshape(n, 1)
                op(codes, word, pool, d_map, s_map, n)
                torch.cuda.synchronize()
                exp = before.clone().view(first.size(0), -1)
                idx = torch.arange(len(rows), device=DEV)
                ar_c = torch.arange(CODE_BYTES, device=DEV)
                ar_s = torch.arange(SCALE_BYTES, device=DEV)
                exp[(page + shift).unsqueeze(1), (slot * CODE_BYTES).unsqueeze(1) + ar_c] = codes[
                    idx
                ]
                exp[
                    (page + shift).unsqueeze(1),
                    (block_kv * CODE_BYTES + slot * SCALE_BYTES).unsqueeze(1) + ar_s,
                ] = sb[idx]
                assert torch.equal(first.view(first.size(0), -1), exp), (
                    f"{step} r={r} owner {owner}: pool bytes"
                )
                assert torch.equal(comp, comp_before), (
                    f"{step} r={r} owner {owner}: COMPRESS pool changed"
                )
                # The reader's view of the owner's pool decodes to the quantizer's values.
                flat = pool.view(pool.size(0), -1)
                rc = flat[page.unsqueeze(1), (slot * CODE_BYTES).unsqueeze(1) + ar_c]
                rs = flat[
                    page.unsqueeze(1),
                    (block_kv * CODE_BYTES + slot * SCALE_BYTES).unsqueeze(1) + ar_s,
                ]
                assert torch.equal(_decode(rc, rs), _decode(codes[idx], sb[idx]))
                print(
                    f"[v41 {step} r={r} owner {owner}] n={n} entries={len(rows)} reserved={n - len(rows)} "
                    f"pool={tuple(pool.shape)} page_shift={shift} bit-exact"
                )
    finally:
        for req in requests:
            manager.free_resources(req)
        manager.shutdown()


# ─── deliberately wrong inputs: the checks above must see each one ───────────────────────────


def _control_batch():
    b = _Batch(64, [0, 100, 250, 7, 64, 500], [300, 90, 120, 33, 64, 200], seed=40)
    codes, sb = _keys(b.n, seed=41)
    return b, codes, sb


def test_control_wrong_slot_mapping():
    """Every request writes with its neighbour's page-table row."""
    b, codes, sb = _control_batch()
    d_map, s_map = b.slot_mappings(table=b.table.roll(1, 0))
    pool = b.pool(seed=42)
    op(codes, _as_scale_word(sb), pool, d_map, s_map, b.n)
    torch.cuda.synchronize()
    frac = _row_mismatch(b, pool, codes, sb)
    print(f"[control] wrong slot mapping (rolled table): {frac:.4f} of rows wrong")
    assert frac > 0.9


def test_control_key_interleaved_offsets():
    """Offsets for a [64 codes, 4 scales]-per-key layout instead of the page-blocked one."""
    b, codes, sb = _control_batch()
    page, slot = b.pages_slots()
    base = page * 64 * ENTRY_BYTES + slot * ENTRY_BYTES
    pool = b.pool(seed=43)
    op(codes, _as_scale_word(sb), pool, base, base + CODE_BYTES, b.n)
    torch.cuda.synchronize()
    frac = _row_mismatch(b, pool, codes, sb)
    print(f"[control] key-interleaved offsets: {frac:.4f} of rows wrong")
    assert frac > 0.9


def test_control_page_size():
    """A 64-entry pool addressed as 128-entry pages."""
    b, codes, sb = _control_batch()
    d_map, s_map = b.slot_mappings(table=b.table // 2, block_kv=128)
    pool = b.pool(seed=44, extra_pages=2)  # a 128-entry page at the last odd page overruns by < 2
    op(codes, _as_scale_word(sb), pool, d_map, s_map, b.n)
    torch.cuda.synchronize()
    frac = _row_mismatch(b, pool, codes, sb)
    print(f"[control] 64-entry pages addressed as 128: {frac:.4f} of rows wrong")
    assert frac > 0.9


@pytest.mark.parametrize("how", ["byte_reversed", "group_major"])
def test_control_scale_bytes_misinterleaved(how):
    """The four UE8M0 bytes packed into the word in the wrong order: big-endian, or the linear
    [T*4] scale output read as if it were group-major [4, T]."""
    b, codes, sb = _control_batch()
    if how == "byte_reversed":
        wrong = sb.flip(1)
    else:
        wrong = sb.reshape(-1).view(SCALE_BYTES, b.n).t()
    d_map, s_map = b.slot_mappings()
    pool = b.pool(seed=45)
    op(codes, _as_scale_word(wrong), pool, d_map, s_map, b.n)
    torch.cuda.synchronize()
    frac = _row_mismatch(b, pool, codes, sb)
    print(f"[control] scale bytes {how}: {frac:.4f} of rows wrong")
    assert frac > 0.9


# ─── rejections: the op's own (loud) and the entry's guards (run under `validating`) ──────────


def _reject_inputs():
    b = _Batch(64, [0], [8], seed=50)
    codes, sb = _keys(8, seed=51)
    d_map, s_map = b.slot_mappings()
    return b, codes, sb, d_map, s_map


_REJECTIONS = {
    # the quantizer's raw scale output, not viewed as int32 [T, 1]
    "scale_uint8_1d": (RuntimeError, "k_scale must be 2D [num_tokens, scale_elements]"),
    "scale_uint8_2d": (RuntimeError, "k_scale must have 4-byte elements (e.g. float32), got %d1"),
    "scale_int32_1d": (RuntimeError, "k_scale must be 2D [num_tokens, scale_elements]"),
    "slot_int32": (RuntimeError, "slot_mapping_fp8 must be int64"),
    "pool_int8": (RuntimeError, "expected scalar type Byte but found Char"),
    "pool_3d": (
        RuntimeError,
        "k_cache must be 4D [num_blocks, block_size, 1, per_token_size], got %d dims3",
    ),
    "pool_dim2": (RuntimeError, "k_cache dimension 2 must be 1, got %d2"),
    "data_noncontig": (RuntimeError, "k_fp8 must be contiguous"),
    "slot_cpu": (RuntimeError, "All tensors must be CUDA tensors"),
    # the entry's is_valid guards: accepted silently by the op (FP8-width rows overwrite
    # neighbouring entries; num_tokens past the supplied rows / slot entries reads past those
    # tensors), so never driven outside `validating`
    "data_fp8_width": (AssertionError, "k_data row bytes + 4 scale bytes"),
    "num_tokens_past_rows": (AssertionError, "num_tokens must not exceed"),
    "num_tokens_past_slots": (AssertionError, "num_tokens must not exceed"),
}


@pytest.mark.parametrize("case", list(_REJECTIONS))
def test_rejections(case):
    b, codes, sb, d_map, s_map = _reject_inputs()
    word = _as_scale_word(sb)
    pool = b.pool(seed=52)
    args = dict(
        k_data=codes,
        k_scale=word,
        k_cache=pool,
        slot_mapping_data=d_map,
        slot_mapping_scale=s_map,
        num_tokens=8,
    )
    if case == "scale_uint8_1d":
        args["k_scale"] = sb.reshape(-1)
    elif case == "scale_uint8_2d":
        args["k_scale"] = sb
    elif case == "scale_int32_1d":
        args["k_scale"] = word.view(-1)
    elif case == "slot_int32":
        args["slot_mapping_data"], args["slot_mapping_scale"] = d_map.int(), s_map.int()
    elif case == "pool_int8":
        args["k_cache"] = pool.view(torch.int8)
    elif case == "pool_3d":
        args["k_cache"] = pool.view(pool.size(0), 64, ENTRY_BYTES)
    elif case == "pool_dim2":
        args["k_cache"] = pool.view(pool.size(0), 32, 2, ENTRY_BYTES)
    elif case == "data_noncontig":
        args["k_data"] = torch.cat([codes, codes], 1)[:, ::2]
    elif case == "slot_cpu":
        args["slot_mapping_data"], args["slot_mapping_scale"] = d_map.cpu(), s_map.cpu()
    elif case == "data_fp8_width":
        args["k_data"] = torch.zeros(8, 128, dtype=torch.uint8, device=DEV)
    elif case == "num_tokens_past_rows":
        args["k_data"] = codes[:4]
    elif case == "num_tokens_past_slots":
        args["slot_mapping_data"], args["slot_mapping_scale"] = d_map[:4], s_map[:4]
    exc, text = _REJECTIONS[case]
    before = pool.clone()
    # Every case runs under `validating`: the op's own rejections pass the guard and are raised by
    # the op; the guard cases raise AssertionError before anything is launched.
    with pytest.raises(exc) as info:
        with validating(op):
            op(**args)
    torch.cuda.synchronize()
    assert text in str(info.value)
    assert torch.equal(pool, before)


# ─── the engine's pools, every row count, the capture sets ───────────────────────────────────
#
# The core target makes this call once per KV-source layer per step (`ModelingV2Core._compress`)
# with n = num_total_compressed_tokens[r] rows of v41_index_k_slot_data[r] / _scale[r]: every
# entry the step completes, then reserved rows (-1). At ratio 1 n = T (up to max_num_tokens, at
# most 16384 in the envelope); at ratio 2 n <= (T + the step's requests) / 2, below 16384 too, and
# the metadata's slot buffers hold max_num_tokens rows. N_MAX below is 16384 for both.
#
# The pools are sized from the core runtime dumps. The cache manager allocates B blocks; the
# ratio-2 INDEXER_COMPRESS allocation holds 3 pages of 64 entries per block, which owners 2, 8 and
# 14 see as contiguous views starting 0, 1 and 2 pages into it ([3B - shift, 64, 1, 68]; their page
# ids are 3 x block), and the ratio-1 pool (owner 20) holds one page of 128 entries per block
# ([B, 128, 1, 68], page id = block). B = 137386 / 142336 (boot dumps) and 151040 (the largest
# pool of the runtime dumps: 453120 x 64 and 151040 x 128, about 1.97e9 and 1.31e9 bytes).
ENGINE_BLOCKS = (137386, 142336, 151040)
OWNER_SHIFT = {2: 0, 8: 1, 14: 2}
N_MAX = 16384
# The longest request of the envelope: max_seq_len 1,048,576 with DSpark (manager length L + 14),
# whose last compressed entry is Lm // r - 1.
LM_1M = 1_048_576 + 14
_ENGINE_CACHE = {}


def _engine_alloc(ratio: int, blocks: int) -> torch.Tensor:
    """The ratio's whole INDEXER_COMPRESS allocation at engine size, random background bytes. One
    at a time per process; tests snapshot it before writing and compare the whole allocation."""
    key = (ratio, blocks)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE.clear()
        torch.cuda.empty_cache()
        pages = 3 * blocks if ratio == 2 else blocks
        g = torch.Generator(device=DEV).manual_seed(blocks + ratio)
        _ENGINE_CACHE[key] = torch.randint(
            0,
            256,
            (pages, 128 // ratio, 1, ENTRY_BYTES),
            dtype=torch.uint8,
            device=DEV,
            generator=g,
        )
    return _ENGINE_CACHE[key]


def _owner_view(alloc: torch.Tensor, ratio: int, owner: int) -> torch.Tensor:
    """What get_indexer_k_cache_buffers(owner) returns: a contiguous view `shift` pages in."""
    view = alloc[OWNER_SHIFT[owner] :] if ratio == 2 else alloc
    assert view.is_contiguous()
    return view


def _engine_batch(ratio: int, blocks: int, cached, new, seed: int) -> _Batch:
    """A step's requests paged over the whole engine pool. Their pages are distinct blocks drawn
    at random, except that request 0's first page is block 0 and the last request's last page is
    block `blocks - 1` (the far end); page id = 3 x block at ratio 2, = block at ratio 1."""
    block_kv = 128 // ratio
    pages = [max(1, -(-(c + m) // block_kv)) for c, m in zip(cached, new)]
    total = sum(pages)
    b = _Batch(block_kv, cached, new, seed=seed, num_pages=total)
    b.num_pages = None  # the pool is the engine allocation, not _Batch.pool()
    g = torch.Generator(device="cpu").manual_seed(seed)
    mid = torch.randperm(blocks - 2, generator=g)[: max(total - 2, 0)] + 1
    chosen = (
        torch.cat([torch.tensor([0]), mid, torch.tensor([blocks - 1])])[:total]
        if total > 1
        else torch.tensor([0])
    )
    table = torch.full((len(cached), max(pages)), -1, dtype=torch.int64)
    off = 0
    for r, p in enumerate(pages):
        table[r, :p] = chosen[off : off + p]
        off += p
    b.table = (table * (3 if ratio == 2 else 1)).to(DEV)
    return b


def _place(dst_view: torch.Tensor, b: _Batch, codes, scale_bytes, rows):
    """Native-torch placement of `rows` (a device index tensor) into `dst_view` in the reader's
    layout (writes through to the allocation the view belongs to)."""
    page, slot = b.pages_slots()
    page, slot = page[rows], slot[rows]
    flat = dst_view.view(dst_view.size(0), -1)
    ar_c = torch.arange(CODE_BYTES, device=DEV)
    ar_s = torch.arange(SCALE_BYTES, device=DEV)
    flat[page.unsqueeze(1), (slot * CODE_BYTES).unsqueeze(1) + ar_c] = codes[rows]
    flat[page.unsqueeze(1), (b.block_kv * CODE_BYTES + slot * SCALE_BYTES).unsqueeze(1) + ar_s] = (
        scale_bytes[rows]
    )


def _far_step(block_kv: int, nreq: int, per_req: int, seed: int):
    """(cached, new) for nreq requests writing per_req entries each: request 0 starts inside its
    first page, the last request's last entry is its last page's last slot, the rest random."""
    g = torch.Generator().manual_seed(seed)
    cached = torch.randint(0, 40 * block_kv, (nreq,), generator=g).tolist()
    cached[0] = (
        int(torch.randint(0, block_kv - per_req % block_kv, (1,), generator=g))
        if per_req < block_kv
        else 0
    )
    last_end = (int(torch.randint(1, 40, (1,), generator=g)) + (per_req // block_kv) + 1) * block_kv
    cached[-1] = last_end - per_req
    return cached, [per_req] * nreq


@pytest.mark.parametrize("blocks", ENGINE_BLOCKS)
@pytest.mark.parametrize("ratio,owner", [(2, 2), (2, 8), (2, 14), (1, 20)])
def test_engine_pool_far_end(ratio, owner, blocks):
    """Every owner's view of the engine-size pool. 40 requests write 6 entries each over pages drawn
    from the whole pool; the last one's last entry is the last slot of block `blocks - 1` (for owner
    14 at ratio 2 and owner 20, the allocation's last page). The whole allocation equals the
    snapshot with exactly those bytes placed; the reader-layout decode is the quantizer's values."""
    block_kv = 128 // ratio
    alloc = _engine_alloc(ratio, blocks)
    view = _owner_view(alloc, ratio, owner)
    cached, new = _far_step(block_kv, 40, 6, seed=blocks + owner)
    b = _engine_batch(ratio, blocks, cached, new, seed=blocks + owner)
    page, slot = b.pages_slots()
    assert (
        int(page[-1]) == (blocks - 1) * (3 if ratio == 2 else 1) and int(slot[-1]) == block_kv - 1
    )
    codes, sb = _keys(b.n, seed=b.n + owner)
    d_map, s_map = b.slot_mappings()
    before = alloc.clone()
    op(codes, _as_scale_word(sb), view, d_map, s_map, b.n)
    torch.cuda.synchronize()
    _place(_owner_view(before, ratio, owner), b, codes, sb, torch.arange(b.n, device=DEV))
    assert torch.equal(alloc, before), "allocation differs from the native placement"
    rc, rs = b.read(view)
    assert torch.equal(_decode(rc, rs), _decode(codes, sb))
    shift = OWNER_SHIFT.get(owner, 0)
    last = (
        (shift + int(page[-1])) * block_kv * ENTRY_BYTES
        + block_kv * CODE_BYTES
        + int(slot[-1]) * SCALE_BYTES
        + 3
    )
    print(
        f"[engine pool r={ratio} owner={owner}] view {tuple(view.shape)}, allocation {alloc.numel()} B, "
        f"last byte written {last}: bit-exact, nothing else changed"
    )
    assert last == alloc.numel() - 1 or shift < 2 and ratio == 2
    del before


@pytest.mark.parametrize("ratio", [1, 2])
def test_every_row_count_prefix_bitwise(ratio):
    """Every row count on the target's call form, over the engine-size pool (owner 14 at ratio 2, owner
    20 at ratio 1). One 16384-row step: 128 requests x 128 entries over pages drawn from the whole
    pool (far end included), request 1 writing the last 128 compressed entries of a 1,048,590-token
    request (entries up to Lm // r - 1, on its 8193rd page), with every 97th row reserved (-1). For
    every n in 1..16384 the call writes rows 0..n-1 and nothing else: the whole allocation equals
    the snapshot with exactly the first n rows of the 16384-row placement (native torch). The sweep
    alternates two key sets, so a row the call failed to write would still hold the other set's
    bytes, and a row >= n it wrote would differ from the background. The 16384-row result also
    decodes to the quantizer's values."""
    owner = 14 if ratio == 2 else 20
    blocks = ENGINE_BLOCKS[-1]
    block_kv = 128 // ratio
    alloc = _engine_alloc(ratio, blocks)
    view = _owner_view(alloc, ratio, owner)
    per_req = N_MAX // 128
    cached, new = _far_step(block_kv, 128, per_req, seed=ratio)
    cached[1] = LM_1M // ratio - per_req
    b = _engine_batch(ratio, blocks, cached, new, seed=ratio)
    assert b.n == N_MAX
    long_rows = b.pos[per_req : 2 * per_req]
    assert int(long_rows[-1]) == LM_1M // ratio - 1 and int(long_rows[-1]) // block_kv == 8192
    d_map, s_map = b.slot_mappings()
    reserved = torch.arange(N_MAX, device=DEV) % 97 == 96
    d_map = torch.where(reserved, -1, d_map).contiguous()
    s_map = torch.where(reserved, -1, s_map).contiguous()
    keys = [_keys(N_MAX, seed=100 + ratio), _keys(N_MAX, seed=200 + ratio)]
    words = [_as_scale_word(sb) for _, sb in keys]
    assert (keys[0][0] != keys[1][0]).any(1).float().mean().item() > 0.99
    # flat byte indices of every row's code bytes and scale word, in the whole allocation
    page, slot = b.pages_slots()
    base = (OWNER_SHIFT.get(owner, 0) + page) * block_kv * ENTRY_BYTES
    code_idx = (
        base.unsqueeze(1) + (slot * CODE_BYTES).unsqueeze(1) + torch.arange(CODE_BYTES, device=DEV)
    )
    scale_idx = (
        base.unsqueeze(1)
        + (block_kv * CODE_BYTES + slot * SCALE_BYTES).unsqueeze(1)
        + torch.arange(SCALE_BYTES, device=DEV)
    )
    kept = (~reserved).nonzero().flatten()
    kept_below = torch.cumsum((~reserved).long(), 0).tolist()
    exp = alloc.clone()
    exp_flat = exp.view(-1)
    bad = []
    for n in range(1, N_MAX + 1):
        v = n & 1
        op(keys[v][0], words[v], view, d_map, s_map, n)
        rows = kept[: kept_below[n - 1]]
        exp_flat[code_idx[rows].reshape(-1)] = keys[v][0][rows].reshape(-1)
        exp_flat[scale_idx[rows].reshape(-1)] = keys[v][1][rows].reshape(-1)
        if not torch.equal(alloc, exp):
            bad.append(n)
            exp.copy_(alloc)  # keep sweeping from the op's state
    print(
        f"[rows r={ratio}] every n in 1..{N_MAX} on {tuple(view.shape)}: mismatching n {bad[:10]} "
        f"({len(bad)} of {N_MAX})"
    )
    assert not bad
    rc, rs = b.read(view)
    ok = ~reserved
    v = N_MAX & 1
    assert torch.equal(_decode(rc[ok], rs[ok]), _decode(keys[v][0][ok], keys[v][1][ok]))
    del exp


def test_one_kernel_at_every_row_count():
    """An ahead-of-time compiled kernel whose row count is only its grid: the same single kernel
    launches at 1 row and at 16384 (no per-shape variant)."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    b = _Batch(128, [0] * 8, [2048] * 8, seed=3)
    codes, sb = _keys(b.n, seed=4)
    d_map, s_map = b.slot_mappings()
    pool = b.pool(seed=5)
    seen = set()
    for n in (1, 2, 97, 1935, 8192, 8193, 1536, 16384):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            op(codes, _as_scale_word(sb), pool, d_map, s_map, n)
            torch.cuda.synchronize()
        kernels = [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]
        print(f"[kernels n={n}] {kernels}")
        assert len(kernels) == 1 and "indexerKCacheScatterUnifiedKernel" in kernels[0], kernels
        seen.add(kernels[0])
    assert len(seen) == 1


# Graph batches: Gset, every G an engine of the envelope captures. That is every integer 1..32
# plus the padded default list for cuda_graph_config {enable_padding: true, max_batch_size: 256}
# (1, 2, 4, 8..128 step 8, 192, 256); the padding-off list at 256 (1..32, 64, 128, 256) and the
# gates' earlier sets (1..8 and {1, 2, 4, 8, 16, 24, 32}) are subsets. Both steps are captured at
# every G: next_n 1 without DSpark and next_n 6 (the DSpark verify step) with it. A next_n-1
# generation row reserves one row at either ratio (at ratio 2 it holds an entry only when the row
# completes a pair), and a DSpark verify request reserves ceil(6 / r) rows: 6 at ratio 1, 3 at
# ratio 2. So n = G (next_n 1) and n = 6G / 3G (next_n 6) over Gset: up to 1536 rows at ratio 1
# and 768 at ratio 2.
_GSET = tuple(range(1, 33)) + (40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256)
assert len(_GSET) == 46
_DECODE_G = _GSET
_DSPARK_G = _GSET
_CAPTURE = [(r, "decode", g) for r in (1, 2) for g in _DECODE_G] + [
    (r, "dspark", g) for r in (1, 2) for g in _DSPARK_G
]


def _graph_step(ratio: int, kind: str, G: int, blocks: int, seed: int, padding: bool):
    """One captured step's rows: valid entries request by request, then reserved rows (-1), n in
    total. Pages over the whole engine pool, the last valid entry on the far end's last slot. With
    `padding` the last request is a graph padding request (no entries)."""
    block_kv = 128 // ratio
    per = 1 if kind == "decode" else -(-6 // ratio)
    n = G * per
    g = torch.Generator().manual_seed(seed)
    pos = torch.randint(0, 30 * block_kv, (G,), generator=g)
    if kind == "decode":
        emits = torch.ones(G, dtype=torch.int64) if ratio == 1 else (pos % 2 == 1).long()
    else:
        emits = torch.full((G,), per, dtype=torch.int64)
    if padding and G > 1:
        emits[-1] = 0
    live = [i for i in range(G) if emits[i] > 0]
    if not live:  # keep at least one entry so the replay writes something
        emits[0] = 1
        live = [0]
    cached = [int(p) // ratio for p in pos]
    # the last live request's last entry sits on its last page's last slot
    j = live[-1]
    cached[j] = (int(torch.randint(1, 30, (1,), generator=g)) + 1) * block_kv - int(emits[j])
    reqs = [(cached[i], int(emits[i])) for i in live]
    b = _engine_batch(ratio, blocks, [c for c, _ in reqs], [m for _, m in reqs], seed=seed)
    d_map, s_map = b.slot_mappings()
    d = torch.full((n,), -1, dtype=torch.int64, device=DEV)
    s = torch.full((n,), -1, dtype=torch.int64, device=DEV)
    d[: b.n] = d_map
    s[: b.n] = s_map
    return b, d, s, n


@pytest.mark.parametrize("ratio,kind,G", _CAPTURE, ids=[f"r{r}-{k}-G{g}" for r, k, g in _CAPTURE])
def test_cuda_graph_capture_sets(ratio, kind, G):
    """The call captured at every graph batch G of Gset, for the next_n-1 step and the DSpark
    verify step, at both ratios, on the engine-size pool (owner 14 / owner 20). After an eager
    warm-up call, one capture; then 3 replays, each on new keys and new offsets (pages anywhere
    in the pool, the far end included; reserved rows -1; a padding request on replays 1 and 3)
    written into the captured buffers. Each replay leaves the whole allocation bitwise equal to
    an eager call on the same inputs from the same snapshot, and equal to the native placement."""
    owner = 14 if ratio == 2 else 20
    blocks = ENGINE_BLOCKS[-1]
    alloc = _engine_alloc(ratio, blocks)
    view = _owner_view(alloc, ratio, owner)
    per = 1 if kind == "decode" else -(-6 // ratio)
    n = G * per
    codes_s = torch.zeros(n, CODE_BYTES, dtype=torch.uint8, device=DEV)
    word_s = torch.zeros(n, 1, dtype=torch.int32, device=DEV)
    d_s = torch.full((n,), -1, dtype=torch.int64, device=DEV)
    s_s = torch.full((n,), -1, dtype=torch.int64, device=DEV)
    op(codes_s, word_s, view, d_s, s_s, n)  # warm-up, every row skipped
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(codes_s, word_s, view, d_s, s_s, n)
    torch.cuda.synchronize()
    for rep in range(3):
        seed = 100_000 * ratio + 100 * G + 10 * rep + (kind == "dspark")
        b, d, s, n2 = _graph_step(ratio, kind, G, blocks, seed, padding=rep % 2 == 0)
        assert n2 == n
        codes, sb = _keys(n, seed=seed)
        codes_s.copy_(codes)
        word_s.copy_(_as_scale_word(sb))
        d_s.copy_(d)
        s_s.copy_(s)
        snap = alloc.clone()
        graph.replay()
        torch.cuda.synchronize()
        got = alloc.clone()
        alloc.copy_(snap)
        op(codes_s, word_s, view, d_s, s_s, n)
        torch.cuda.synchronize()
        assert torch.equal(alloc, got), f"replay {rep} differs from eager"
        assert not torch.equal(got, snap), f"replay {rep} wrote nothing"
        _place(_owner_view(snap, ratio, owner), b, codes, sb, torch.arange(b.n, device=DEV))
        assert torch.equal(alloc, snap), f"replay {rep} differs from the native placement"
        del snap, got
    print(
        f"[graph r={ratio} {kind} G={G}] n={n}: 3 replays bitwise equal to eager and to the placement"
    )


# Beyond the engine's pools: an allocation past 2^32 bytes, rows on the pages that straddle the
# 2^31- and 2^32-byte offsets and on the last page. The kernel forms int64 byte offsets; a caller
# that computed the offsets in 32 bits would wrap them (the control).
MARGIN_PAGES = {64: 1_000_000, 128: 500_000}  # 4.352e9 bytes either way


@pytest.mark.parametrize("block_kv", [64, 128])
def test_past_2_to_the_32_bytes_and_wrap_control(block_kv):
    page_bytes = block_kv * ENTRY_BYTES
    pages = MARGIN_PAGES[block_kv]
    _ENGINE_CACHE.clear()
    torch.cuda.empty_cache()
    g = torch.Generator(device=DEV).manual_seed(block_kv)
    pool = torch.randint(
        0, 256, (pages, block_kv, 1, ENTRY_BYTES), dtype=torch.uint8, device=DEV, generator=g
    )
    assert pool.numel() > 2**32
    far = [2**31 // page_bytes, 2**32 // page_bytes, pages - 1]
    for p, bound in zip(far[:2], (2**31, 2**32)):
        assert p * page_bytes < bound < (p + 1) * page_bytes  # the page straddles the boundary
    b = _Batch(block_kv, [0, 0, 0], [block_kv] * 3, seed=block_kv, num_pages=3)
    b.table = torch.tensor(far, dtype=torch.int64, device=DEV).view(3, 1)
    b.num_pages = pages
    codes, sb = _keys(b.n, seed=block_kv)
    d_map, s_map = b.slot_mappings()
    snap = pool.clone()
    op(codes, _as_scale_word(sb), pool, d_map, s_map, b.n)
    torch.cuda.synchronize()
    exp = snap.clone()
    _place(exp, b, codes, sb, torch.arange(b.n, device=DEV))
    assert torch.equal(pool, exp), "pool differs from the native placement"
    del exp
    # control: the same offsets wrapped to signed 32 bits (negative ones are skipped, the others
    # land in the pool's first 2^31 bytes)
    pool.copy_(snap)

    def wrap(t):
        return ((t + 2**31) % 2**32) - 2**31

    op(codes, _as_scale_word(sb), pool, wrap(d_map), wrap(s_map), b.n)
    torch.cuda.synchronize()
    frac = _row_mismatch(b, pool, codes, sb)
    past = (d_map >= 2**31).float().mean().item()
    print(
        f"[margin block_kv={block_kv}] pool {pool.numel()} B, pages {far}: bit-exact; int32-wrapped "
        f"offsets: {frac:.4f} of rows wrong ({past:.4f} of rows start past 2^31)"
    )
    assert frac >= past > 0.5
    del pool, snap


def test_control_wrong_owner_view():
    """Owner 14's offsets written through owner 2's view (the ratio-2 views overlap, shifted by one
    page each): the bytes land two pages early, and owner 14's reader sees none of them."""
    ratio, blocks = 2, ENGINE_BLOCKS[-1]
    alloc = _engine_alloc(ratio, blocks)
    cached, new = _far_step(64, 12, 20, seed=9)
    b = _engine_batch(ratio, blocks, cached, new, seed=9)
    codes, sb = _keys(b.n, seed=10)
    d_map, s_map = b.slot_mappings()
    snap = alloc.clone()
    op(codes, _as_scale_word(sb), _owner_view(alloc, ratio, 2), d_map, s_map, b.n)
    torch.cuda.synchronize()
    frac = _row_mismatch(b, _owner_view(alloc, ratio, 14), codes, sb)
    alloc.copy_(snap)
    print(f"[control] owner 14's offsets through owner 2's view: {frac:.4f} of rows wrong")
    assert frac > 0.9
    del snap


# Pool-size invariance. Every V4.1 persistent role of one cache manager shares one block
# count B, so B is capped by the whole GPU (284208 MiB = 298,013,687,808 B) divided by the manager's
# persistent bytes per block. The cheapest manager that holds an index-K pool is the PP2 stage-1
# manager in native storage (layers 20-39: 356 B/token, 45,568 B per block, measured); an unsplit
# native manager costs 113,920 B per block and a stage-0 one 68,352. So no V4.1 manager on a GB300
# holds more than floor(298,013,687,808 / 45,568) = 6,539,977 blocks. The largest pool measured (a
# stage-1 manager built alone at half the GPU) is 3,245,959 blocks, and the unsplit ceiling 2,615,990; both
# are run as intermediate sizes. One B_MAX serves both ratios (the ratio-2 owners live only in
# unsplit and stage-0 managers, whose own ceilings are lower), so the ratio-2 allocation at B_MAX,
# 3 x 6,539,977 pages of 64 entries (85,385,939,712 B), is a superset of any real one.
# The same step, at the same page ids, runs on exact-size allocations (the pool ends where the
# allocation ends) from B_min, the smallest pool that holds its pages, up to that bound.
B_MAX = 6_539_977
INV_BLOCKS = (481, 1493, 151040, 852832, 2_615_990, 3_245_959, B_MAX)
B_OTHER = 852832  # the control's wrong page count
SENT = 0xA5
FAR_BYTES = (2**31, 2**32, 2**33, 2**34, 2**35, 2**36)  # those below the allocation's size


def _sentinel_alloc(ratio: int, blocks: int) -> torch.Tensor:
    """The ratio's whole INDEXER_COMPRESS allocation for `blocks` blocks, every byte SENT."""
    _ENGINE_CACHE.clear()
    torch.cuda.empty_cache()
    pages = 3 * blocks if ratio == 2 else blocks
    return torch.full((pages, 128 // ratio, 1, ENTRY_BYTES), SENT, dtype=torch.uint8, device=DEV)


def _others_are_sentinel(alloc: torch.Tensor, touched: torch.Tensor) -> bool:
    """Whether every byte of the allocation outside the pages `touched` still holds SENT."""
    saved = alloc[touched].clone()
    alloc[touched] = SENT
    word = torch.full((8,), SENT, dtype=torch.uint8, device=DEV).view(torch.int64)
    ok = not bool((alloc.view(-1).view(torch.int64) != word).any())
    alloc[touched] = saved
    return ok


def _offsets(page: torch.Tensor, slot: torch.Tensor, block_kv: int):
    base = page * block_kv * ENTRY_BYTES
    return (base + slot * CODE_BYTES).contiguous(), (
        base + block_kv * CODE_BYTES + slot * SCALE_BYTES
    ).contiguous()


def _wrap32(t: torch.Tensor) -> torch.Tensor:
    return (((t + 2**31) % 2**32) - 2**31).contiguous()


def _scatter_rows(alloc: torch.Tensor, ratio: int, rows, wrap: bool = False, page_fn=None):
    """One call per owner, through that owner's view; rows = [(owner, view page ids, slots, codes, scale bytes)]."""
    block_kv = 128 // ratio
    for owner, page, slot, codes, sb in rows:
        d, s = _offsets(page if page_fn is None else page_fn(page), slot, block_kv)
        if wrap:
            d, s = _wrap32(d), _wrap32(s)
        op(codes, _as_scale_word(sb), _owner_view(alloc, ratio, owner), d, s, codes.size(0))
    torch.cuda.synchronize()


def _expected_pages(ratio: int, rows):
    """Native-torch placement: the sorted allocation pages the rows touch, each starting as SENT,
    with every row's code bytes and scale word placed in the reader's layout."""
    block_kv = 128 // ratio
    touched = torch.unique(torch.cat([OWNER_SHIFT.get(o, 0) + p for o, p, *_ in rows]))
    exp = torch.full((touched.numel(), block_kv * ENTRY_BYTES), SENT, dtype=torch.uint8, device=DEV)
    ar_c = torch.arange(CODE_BYTES, device=DEV)
    ar_s = torch.arange(SCALE_BYTES, device=DEV)
    for owner, page, slot, codes, sb in rows:
        i = torch.searchsorted(touched, OWNER_SHIFT.get(owner, 0) + page)
        exp[i.unsqueeze(1), (slot * CODE_BYTES).unsqueeze(1) + ar_c] = codes
        exp[i.unsqueeze(1), (block_kv * CODE_BYTES + slot * SCALE_BYTES).unsqueeze(1) + ar_s] = sb
    return touched, exp


def _matches_placement(alloc: torch.Tensor, touched: torch.Tensor, exp: torch.Tensor) -> bool:
    got = alloc[touched].view(touched.numel(), -1)
    return torch.equal(got, exp) and _others_are_sentinel(alloc, touched)


def _invariance_rows(ratio: int):
    """The invariance step: 10 requests x 20 new entries (request 0 starting mid-page, pages crossed,
    the last request ending on the last slot of block B_min - 1), over B_min blocks, each block used
    once, page ids 3 x block at ratio 2. Every owner of the ratio writes it with its own keys.
    Returns (B_min, rows)."""
    block_kv = 128 // ratio
    cached, new = _far_step(block_kv, 10, 20, seed=40 + ratio)
    b_min = sum(max(1, -(-(c + m) // block_kv)) for c, m in zip(cached, new))
    b = _engine_batch(ratio, b_min, cached, new, seed=40 + ratio)
    page, slot = b.pages_slots()
    assert int(page[-1]) == (b_min - 1) * (3 if ratio == 2 else 1) and int(slot[-1]) == block_kv - 1
    owners = (2, 8, 14) if ratio == 2 else (20,)
    rows = []
    for owner in owners:
        codes, sb = _keys(b.n, seed=500 + owner)
        rows.append((owner, page, slot, codes, sb))
    return b_min, rows


@pytest.mark.parametrize("ratio", [1, 2])
def test_pool_size_invariance(ratio):
    """The same step at the same page ids through every owner's view of the manager
    layout (ratio 2: one 3B-page allocation, owners 2 / 8 / 14 at 0 / 1 / 2 pages in; ratio 1:
    [B, 128, 1, 68]) at B = B_min, 481, 1493, 151040, 852832 and B_MAX, each an exact-size SENT-filled
    allocation. At every B the pages the step touches are bitwise identical to B_min's and to the
    native placement, and every other byte of the allocation is still SENT. Control: at B_MAX the
    step's page ids counted from the end of a B_OTHER-block pool (a B-dependent addressing) fails
    the same check."""
    b_min, rows = _invariance_rows(ratio)
    touched, exp = _expected_pages(ratio, rows)
    sizes = (b_min,) + INV_BLOCKS
    assert list(sizes) == sorted(sizes)
    first = None
    for blocks in sizes:
        alloc = _sentinel_alloc(ratio, blocks)
        _scatter_rows(alloc, ratio, rows)
        got = alloc[touched].view(touched.numel(), -1).clone()
        first = got if first is None else first
        same, placed, rest = (
            torch.equal(got, first),
            torch.equal(got, exp),
            _others_are_sentinel(alloc, touched),
        )
        print(
            f"[invariance r={ratio}] B={blocks}: allocation {tuple(alloc.shape)} ({alloc.numel()} B), "
            f"{sum(r[3].size(0) for r in rows)} rows on {touched.numel()} pages: equal to B_min {same}, "
            f"to the native placement {placed}, rest untouched {rest}"
        )
        assert same and placed and rest, f"B={blocks}: result depends on the pool size"
        if blocks != sizes[-1]:
            del alloc
    k = 3 if ratio == 2 else 1
    alloc.fill_(SENT)
    _scatter_rows(alloc, ratio, rows, page_fn=lambda p: k * (B_OTHER - 1) - p)
    wrong = _matches_placement(alloc, touched, exp)
    print(
        f"[invariance control r={ratio}] page ids counted from the end of a {B_OTHER}-block pool on "
        f"B={B_MAX}: check passes {wrong}"
    )
    assert not wrong
    del alloc


def _far_pages(ratio: int, blocks: int):
    """Allocation pages that straddle each FAR_BYTES offset, plus the last page of block blocks - 1
    of every owner (ratio 2: allocation pages 3B-3, 3B-2, 3B-1; ratio 1: B-1)."""
    page_bytes = (128 // ratio) * ENTRY_BYTES
    alloc_bytes = (3 * blocks if ratio == 2 else blocks) * page_bytes
    marks = [x for x in FAR_BYTES if x < alloc_bytes]
    pages = [x // page_bytes for x in marks]
    for x, p in zip(marks, pages):
        assert p * page_bytes < x < (p + 1) * page_bytes
    return pages + (
        [3 * blocks - 3, 3 * blocks - 2, 3 * blocks - 1] if ratio == 2 else [blocks - 1]
    )


def _far_rows(ratio: int, blocks: int, seed: int):
    """Every slot of each far page, written through the owner whose view holds the page (allocation
    page p belongs to owner shift p % 3 at ratio 2, view page id p - shift = 3 x block)."""
    block_kv = 128 // ratio
    pages = _far_pages(ratio, blocks)
    rows = []
    for owner in (2, 8, 14) if ratio == 2 else (20,):
        shift = OWNER_SHIFT.get(owner, 0)
        mine = [p - shift for p in pages if ratio == 1 or p % 3 == shift]
        if not mine:
            continue
        assert all(m % (3 if ratio == 2 else 1) == 0 for m in mine)
        page = torch.tensor(mine, dtype=torch.int64, device=DEV).repeat_interleave(block_kv)
        slot = torch.arange(block_kv, device=DEV).repeat(len(mine))
        codes, sb = _keys(page.numel(), seed=seed + owner)
        rows.append((owner, page, slot, codes, sb))
    return rows


@pytest.mark.parametrize("ratio", [1, 2])
def test_far_end_at_b_max(ratio):
    """Far-end addressing on the B_MAX allocation (ratio 2: 19,619,931 pages, 85,385,939,712 B;
    ratio 1: 6,539,977 pages, 56,923,959,808 B): full pages of rows on the pages straddling bytes
    2^31 .. 2^35 (and 2^36 at ratio 2) and on the last page of block B_MAX - 1 of every owner (for owners 14
    and 20 the allocation's last byte). The whole allocation equals the SENT background with exactly
    the native placement. Control: the same offsets wrapped to signed 32 bits fail the check."""
    rows = _far_rows(ratio, B_MAX, seed=900 + ratio)
    touched, exp = _expected_pages(ratio, rows)
    alloc = _sentinel_alloc(ratio, B_MAX)
    assert alloc.numel() > 2**35
    _scatter_rows(alloc, ratio, rows)
    ok = _matches_placement(alloc, touched, exp)
    last = rows[-1]
    tail = torch.equal(alloc.view(-1)[-SCALE_BYTES:], last[4][-1])
    print(
        f"[far end r={ratio}] B={B_MAX}, allocation {alloc.numel()} B, pages {touched.tolist()}: native "
        f"placement {ok}, owner {last[0]}'s last scale word is the allocation's last 4 bytes {tail}"
    )
    assert ok and tail
    alloc.fill_(SENT)
    _scatter_rows(alloc, ratio, rows, wrap=True)
    past = (
        torch.cat([_offsets(p, s, 128 // ratio)[0] for _, p, s, _, _ in rows])
        .ge(2**31)
        .float()
        .mean()
        .item()
    )
    wrong = _matches_placement(alloc, touched, exp)
    print(
        f"[far end control r={ratio}] offsets wrapped to signed 32 bits ({past:.4f} of rows start past "
        f"2^31): check passes {wrong}"
    )
    assert past > 0.5 and not wrong
    del alloc
