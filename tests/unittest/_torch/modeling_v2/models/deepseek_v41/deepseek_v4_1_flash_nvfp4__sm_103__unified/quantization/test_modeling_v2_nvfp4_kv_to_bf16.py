# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nvfp4_kv_to_bf16 entry: the bitwise twin of FlashInfer's nvfp4_kv_dequantize_paged.

The pool is DeepSeek-V4.1's native compressed-KV layout: one 288-byte row per entry, 256 packed E2M1
bytes (even element in the low nibble) then 32 E4M3 scale bytes, read as two strided views. Every gate
is bit-exact on bf16 bit patterns, against BOTH FlashInfer's `nvfp4_kv_dequantize_paged` on the same
inputs (`_twin`, called as the DeepSeek-V4.1 target's FlashInfer form calls it) and the native-torch E2M1
lookup-table reference (`op.reference`); entries the op must not write are checked against a NaN sentinel
that both ops must leave in place.

Shapes (`op.CELLS`): the target's per-query (top-k) form -- page_size 1 over the flattened pool and an
[R, 512] row table with -1 padding, out-of-range ids and short rows, R from 1 to 8192 -- and its
per-request form -- page_size 64 / 128 page tables of whole contexts up to 16384 entries, table holes,
request lengths 0 / 1 / page +- 1 / L, the compact [B, 132 * 128 / r, 1, 512] views and the engine's
pool sizes. This file keeps what a cell cannot carry: the quantizer round trip (-0.0 kept), a global
scale != 1, CUDA-graph capture and replay, wrong variants that must fail, the op's own rejections, and
every case on the live DeepSeek-V4.1 cache manager's native views.
"""

import pytest
import torch
from flashinfer import nvfp4_kv_dequantize_paged

import tensorrt_llm._torch.custom_ops.nvfp4_fake_quantize  # noqa: F401 -- registers torch.ops.trtllm.nvfp4_fake_quantize
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.quantization.nvfp4_kv_to_bf16 import (  # noqa: E501
    nvfp4_kv_to_bf16 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "nvfp4_kv_to_bf16 requires a CUDA device"

DEV = "cuda"
W = 512
BLOCK = 16
CODE_BYTES = W // 2
ROW = CODE_BYTES + W // BLOCK  # 288
NUM_SLOTS = 16384
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SENTINEL = -1  # int16 view of a bf16 NaN (0xFFFF)
_TABLE_DTYPE = {"int32": torch.int32, "int64": torch.int64}


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _random_pool(page_size: int, seed: int, slots: int = NUM_SLOTS) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    codes = torch.randint(0, 256, (slots, CODE_BYTES), generator=g, device=DEV, dtype=torch.int32)
    scales = torch.randint(0, 0x7F, (slots, W // BLOCK), generator=g, device=DEV, dtype=torch.int32)
    flat = torch.cat([codes, scales], dim=1).to(torch.uint8)
    return flat.view(slots // page_size, page_size, ROW)


def _page_views(pool: torch.Tensor):
    return pool[..., :CODE_BYTES].unsqueeze(2), pool[..., CODE_BYTES:].unsqueeze(2)


def _row_views(pool: torch.Tensor):
    flat = pool.view(-1, ROW)
    return flat[:, :CODE_BYTES][:, None, None], flat[:, CODE_BYTES:][:, None, None]


def _gs(v: float = 1.0) -> torch.Tensor:
    return torch.full((1,), v, dtype=torch.float32, device=DEV)


def _sentinel_out(*shape) -> torch.Tensor:
    return torch.full(shape, SENTINEL, dtype=torch.int16, device=DEV).view(torch.bfloat16)


def _ref_rows(flat: torch.Tensor, rows: torch.Tensor, global_scale: float = 1.0) -> torch.Tensor:
    """bf16 [n, W] of pool rows `rows` (int64 [n], all inside the pool) of the flat [S, 288] pool:
    `op.reference` of a one-request row gather over pages of one row."""
    n = rows.numel()
    out = torch.empty(1, n, 1, W, dtype=torch.bfloat16, device=flat.device)
    ref = op.reference(
        flat[:, :CODE_BYTES][:, None, None],
        flat[:, CODE_BYTES:][:, None, None],
        rows.reshape(1, n),
        torch.full((1,), n, dtype=torch.int32, device=flat.device),
        _gs(global_scale),
        out,
    )
    return ref.view(n, W)


def _twin(kv_cache, kv_cache_sf, block_tables, seq_lens, global_scale, out):
    """FlashInfer's paged NVFP4 dequantize on the same operands, one call made the way the target's
    FlashInfer form makes it: the latent has no V, so a 16-element view of the same rows (8 code bytes,
    1 scale byte) is the V operand, into a scratch that is discarded. Writes `out`, returns it."""
    v_scratch = torch.empty(out.shape[:-1] + (16,), dtype=out.dtype, device=out.device)
    nvfp4_kv_dequantize_paged(
        (kv_cache, kv_cache[..., :8]),
        (kv_cache_sf, kv_cache_sf[..., :1]),
        block_tables,
        seq_lens,
        global_scale,
        global_scale,
        out,
        v_scratch,
        "NHD",
    )
    return out


def _both(kv, sf, table, lens, gs, shape):
    """Run the op and the FlashInfer twin on identical sentinel-filled outputs."""
    a, b = _sentinel_out(*shape), _sentinel_out(*shape)
    op(kv, sf, table, lens, gs, a)
    _twin(kv, sf, table, lens, gs, b)
    torch.cuda.synchronize()
    return a, b


def _row_table(R: int, K: int, seed: int, slots: int, table_dtype=torch.int32):
    g = torch.Generator(device=DEV).manual_seed(seed)
    tab = torch.randint(0, slots, (R, K), generator=g, device=DEV, dtype=torch.int64)
    if R > 1:
        tab[1, K // 2 :] = -1
    tab[0, 3] = -1
    tab[0, 5] = slots
    tab[0, 7] = slots + 1000
    sl = torch.full((R,), K, dtype=torch.int32, device=DEV)
    if R > 2:
        sl[2] = K - 37
    valid = (tab >= 0) & (tab < slots) & (torch.arange(K, device=DEV)[None] < sl[:, None].long())
    return tab.to(table_dtype), sl, valid, tab.clamp(0, slots - 1)


def _i32(*values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device=DEV)


def _build(spec):
    """(kv_cache, kv_cache_sf, block_tables, seq_lens, global_scale, out) for a cell; `out` is
    sentinel-filled, so `compare` also proves every entry the op must skip was left alone."""
    kind = spec["kind"]
    if kind == "topk":
        R, K = spec["R"], 512
        kv, sf = _row_views(_random_pool(64, seed=R))
        dtype = _TABLE_DTYPE[spec["table_dtype"]]
        tab, sl, _, _ = _row_table(R, K, seed=100 + R, slots=NUM_SLOTS, table_dtype=dtype)
        return kv, sf, tab, sl, _gs(), _sentinel_out(R, K, 1, W)
    if kind == "request":
        page_size = spec["page_size"]
        kv, sf = _page_views(_random_pool(page_size, seed=page_size))
        pages = NUM_SLOTS // page_size
        L = 8192
        lens = [0, 1, page_size - 1, page_size + 1, 700, 4097, L]
        B = len(lens)
        g = torch.Generator(device=DEV).manual_seed(7)
        bt = torch.stack(
            [torch.randperm(pages, generator=g, device=DEV)[: L // page_size] for _ in range(B)]
        )
        bt[4, 2] = -1
        bt[5, 1] = pages + 3
        bt = bt.to(_TABLE_DTYPE[spec["table_dtype"]])
        return kv, sf, bt, _i32(*lens), _gs(), _sentinel_out(B, L, 1, W)
    if kind == "prefill_16k":
        kv, sf = _page_views(_random_pool(128, seed=11))
        bt = torch.randperm(NUM_SLOTS // 128, device=DEV).view(1, -1).to(torch.int32)
        return kv, sf, bt, _i32(NUM_SLOTS), _gs(), _sentinel_out(1, NUM_SLOTS, 1, W)
    if kind == "compact":
        r, B = spec["ratio"], spec["B"]
        page_size = 128 // r
        kv, sf = _page_views(_random_pool(page_size, seed=20 + r))
        L = 132 * 128 // r
        g = torch.Generator(device=DEV).manual_seed(B)
        bt = torch.randint(
            0, NUM_SLOTS // page_size, (B, 132), generator=g, device=DEV, dtype=torch.int32
        )
        sl = torch.randint(0, L + 1, (B,), generator=g, device=DEV, dtype=torch.int32)
        return kv, sf, bt, sl, _gs(), _sentinel_out(B, L, 1, W)
    assert kind == "engine_pool", kind
    rows, page_size = spec["rows"], spec["page_size"]
    pages = rows // page_size
    pool = _random_pool(page_size, seed=pages, slots=rows)
    if spec["form"] == "per_request":
        kv, sf = _page_views(pool)
        bt = _i32(0, pages - 1, pages // 2, 1).view(1, 4)
        L = 4 * page_size
        return kv, sf, bt, _i32(L), _gs(), _sentinel_out(1, L, 1, W)
    kv, sf = _row_views(pool)
    return kv, sf, _i32(0, rows - 1, 12345).view(1, 3), _i32(3), _gs(), _sentinel_out(1, 3, 1, W)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec)
    out = args[5]
    expected = op.reference(*args)
    assert op(*args) is out, "the op writes in place and returns what it wrote"
    twin = _twin(*args[:5], _sentinel_out(*out.shape))
    torch.cuda.synchronize()
    op.compare(out, expected)
    assert torch.equal(_bits(out), _bits(twin)), "the op and FlashInfer's twin differ"
    written = int((_bits(out) != SENTINEL).any(-1).sum())
    assert written > 0, "the cell wrote nothing, so it certifies nothing"


def test_quantizer_round_trip_and_negative_zero():
    """Rows written by nvfp4_fake_quantize's native mode read back equal to its bf16 output."""
    n = 4096
    g = torch.Generator(device=DEV).manual_seed(3)
    x = (
        torch.randn(n, W, generator=g, device=DEV) * torch.logspace(-4, 4, n, device=DEV)[:, None]
    ).to(torch.bfloat16)
    x[:8] = 0.0
    x[8:16] = -0.0
    x[16, :BLOCK] = 3000.0
    slots = torch.randperm(NUM_SLOTS, device=DEV)[:n]
    pool = torch.full((NUM_SLOTS // 64, 64, ROW), 0xAB, dtype=torch.uint8, device=DEV)
    torch.ops.trtllm.nvfp4_fake_quantize(x, pool[..., :CODE_BYTES], slots, pool[..., CODE_BYTES:])
    core = torch.full((n, W), float("nan"), dtype=torch.bfloat16, device=DEV)
    torch.ops.trtllm.nvfp4_fake_quantize(x, core, None, None)
    kv, sf = _row_views(pool)
    a, b = _both(kv, sf, slots.view(1, n).to(torch.int32), _i32(n), _gs(), (1, n, 1, W))
    assert torch.equal(_bits(a), _bits(b))
    assert torch.equal(_bits(a[0, :, 0]), _bits(core))
    assert int((_bits(core) == -32768).sum()) > 0


@pytest.mark.parametrize("gs", [0.5, 2.0, 1.0 / 3.0])
def test_global_scale_matches_the_flashinfer_twin(gs):
    """A global scale other than 1 (the target passes 1.0): the same bits as FlashInfer's twin."""
    pool = _random_pool(64, seed=5)
    kv, sf = _row_views(pool)
    tab, sl, _, _ = _row_table(64, 512, seed=9, slots=NUM_SLOTS)
    a, b = _both(kv, sf, tab, sl, _gs(gs), (64, 512, 1, W))
    assert torch.equal(_bits(a), _bits(b))


@pytest.mark.parametrize("R", [1, 6, 64, 384, 1536])
def test_cuda_graph_capture_replay(R):
    """Captured once, replayed on new tables and lengths: equal to eager FlashInfer each time."""
    K = 512
    pool = _random_pool(64, seed=R + 1)
    kv, sf = _row_views(pool)
    tab = torch.zeros((R, K), dtype=torch.int32, device=DEV)
    sl = torch.zeros((R,), dtype=torch.int32, device=DEV)
    out = _sentinel_out(R, K, 1, W)
    gs = _gs()
    op(kv, sf, tab, sl, gs, out)  # warm-up (JIT) outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(kv, sf, tab, sl, gs, out)
    for rep in range(3):
        t, seq_lens, _, _ = _row_table(R, K, seed=1000 * rep + R, slots=NUM_SLOTS)
        tab.copy_(t)
        sl.copy_(seq_lens)
        out.view(torch.int16).fill_(SENTINEL)
        graph.replay()
        ref = _sentinel_out(R, K, 1, W)
        _twin(kv, sf, tab, sl, gs, ref)
        torch.cuda.synchronize()
        assert torch.equal(_bits(out), _bits(ref)), rep


def test_wrong_variants_fail_the_gate():
    """Nibble order swapped, one scale per 32: each must differ from the op's output."""
    pool = _random_pool(64, seed=31)
    flat = pool.view(-1, ROW)
    kv, sf = _row_views(pool)
    rows = torch.randint(0, NUM_SLOTS, (256,), device=DEV)
    a, _ = _both(kv, sf, rows.view(1, -1).to(torch.int32), _i32(256), _gs(), (1, 256, 1, W))
    good = _bits(a[0, :, 0])
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float32, device=DEV)
    r = flat[rows]
    codes = r[:, :CODE_BYTES]
    s16 = r[:, CODE_BYTES:].contiguous().view(torch.float8_e4m3fn).float()
    swapped = torch.stack([lut[(codes >> 4).long()], lut[(codes & 0xF).long()]], -1).flatten(-2)
    per32 = s16[:, ::2].repeat_interleave(32, -1)
    straight = torch.stack([lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], -1).flatten(-2)
    for bad in (
        (swapped * s16.repeat_interleave(BLOCK, -1)).to(torch.bfloat16),
        (straight * per32).to(torch.bfloat16),
    ):
        assert (good != _bits(bad)).float().mean() > 0.3


def test_rejects_wrong_operands():
    pool = _random_pool(64, seed=2)
    kv, sf = _row_views(pool)
    tab = torch.zeros((2, 8), dtype=torch.int32, device=DEV)
    sl = torch.full((2,), 8, dtype=torch.int32, device=DEV)
    with pytest.raises(ValueError):
        op(kv, sf, tab, sl, _gs(), torch.empty((2, 8, 1, W), dtype=torch.float16, device=DEV))
    with pytest.raises(ValueError):
        op(kv, sf, tab, sl.long(), _gs(), _sentinel_out(2, 8, 1, W))
    with pytest.raises(ValueError):
        op(kv, sf, tab, sl, _gs().double(), _sentinel_out(2, 8, 1, W))


# ─── native storage on the live DeepSeek-V4.1 cache manager, at the engine envelope ─────
#
# The target calls this entry at two sites (unified sm_103 modeling, native KV storage):
#   per request:  nvfp4_kv_to_bf16(codes4, scales4, md.compress_block_tables[r][:B], md.compressed_kv_lens_cuda[r][:B],
#                                  one, comp_out[r][:B])
#                 codes4 / scales4 the KV source's native COMPRESS views [pages, 128 / r, 1, 256] / [.., 32];
#                 comp_out[r] the [seqs, cols * 128 / r, 1, 512] view of one buffer (seqs = max_batch_size,
#                 cols = the manager's max_blocks_per_seq);
#   per query:    nvfp4_kv_to_bf16(codes1, scales1, topk_slots[:T], topk_cols[:T], one, topk_out[:T])
#                 codes1 / scales1 the same views re-viewed with page size 1 ([pages * 128 / r, 1, 1, 256]),
#                 topk_slots the index op's [T, 512] int32 pool slots (-1 past each query's valid entries),
#                 topk_cols 512 per query, topk_out the [T, 512, 1, 512] view of a [max_num_tokens * 512, 512]
#                 buffer.
# The tests below are the FlashInfer form's native-storage and envelope tests run on this op; the reference is
# the writer's own core output or the LUT decode of the pool bytes, and the full calls are also held bitwise to
# FlashInfer's nvfp4_kv_dequantize_paged (the twin, `_twin`). The Triton kernel specializes on integer
# arguments; _jit_variants counts its compiled variants so the tests can show that a row count, a batch or
# a capture never compiles a new one.

import os  # noqa: E402
import random  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import tensorrt_llm._torch.custom_ops.nvfp4_kv_to_bf16 as _kvb_op  # noqa: E402 (the Triton kernel, variant probe)


def _jit_variants() -> int:
    """Number of compiled variants Triton holds for the entry's kernel, over all devices."""
    fn = _kvb_op._nvfp4_kv_to_bf16_kernel
    dc = getattr(fn, "device_caches", None)
    if dc is not None:
        return sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in dc.values())
    return sum(len(v) for v in fn.cache.values())


_STORAGE_ENV = "TRTLLM_DSV41_KV_STORAGE"
BYTE_SENTINEL = 0xFF  # never a scale byte the writer produces (E4M3 NaN)
TOPK = 512  # the target's compressed top-k width (index_topk)
_BMAX_DEVICE_TOTAL = 297_020_948_480
# ratio -> (B_max pages over the gates' axis {2048, 8192} x max_batch_size 1..32 x DSpark {0, 5} at max_seq_len
# 16384, page size, gates' largest, owner whose view spans the ratio's buffer)
NATIVE_COMPRESS_AXIS = {
    "ratio2": (2_606_019, 64, 1_954_512, 2),
    "ratio1": (868_673, 128, 651_504, 20),
}
# ratio -> (ceiling pages, page size, gates' largest, owner). The ceiling is the measured maximum of the native
# COMPRESS views over the engine envelope's scanned tuples at 0.5 x the device's memory, PP2 stage managers
# included (stage alone): max_num_tokens {2048, 4096, 8192, 16384} x max_batch_size {1..8, 12, 16, 24, 32, 48,
# 64, 96, 128, 160, 192, 224, 256} x DSpark {0, 5} x max_seq_len {16K, 32K, 64K, 128K, 256K, 1M}, unsplit
# and per PP2 stage. A stage holds half the layers, so its pools are the largest: ratio 2 on stage 0 (layers
# 0-19), ratio 1 on stage 1 (layers 20-39), both at CEILING_KEY. Quotas above 0.5 x and unsampled
# max_batch_size values are not covered.
NATIVE_COMPRESS = {"ratio2": (6_480_780, 64, 1_954_512, 2), "ratio1": (3_245_959, 128, 651_504, 20)}
CEILING_KEY = (
    2048,
    256,
    0,
    1_048_576,
)  # (max_num_tokens, max_batch_size, DSpark max_draft_len, max_seq_len)
CEILING_STAGE = {
    "ratio2": 0,
    "ratio1": 1,
}  # the PP2 stage whose manager holds the ratio's ceiling view
NATIVE_COMPRESS_UNSPLIT = {
    "ratio2": 3_886_647,
    "ratio1": 1_295_549,
}  # pp_size 1, at (2048, 128, 0, 1M)
# Pool sizes the invariance test also runs: the gates' axis B_max, envelope managers' views (the engine dump's
# (16384, 256, 5, 32768) views, the (16384, 256, 5, 1M) manager's), the unsplit ceiling, stage 0's largest at
# max_seq_len 16384 and stage 1's largest in a coupled PP2 engine.
_EXTRA_B = {
    "ratio2": (2_606_019, 1_807_701, 2_974_605, 3_869_352, 3_886_647, 4_747_719),
    "ratio1": (868_673, 602_567, 991_535, 1_289_784, 1_295_549, 2_159_957),
}


def _write(x, pool, slots):
    """The writer's native form into a [.., 288] pool (codes then scales), rows at `slots`."""
    torch.ops.trtllm.nvfp4_fake_quantize(x, pool[..., :CODE_BYTES], slots, pool[..., CODE_BYTES:])


def _core(x):
    """The writer's own core (bf16) output of x, through a fixed-size out (one compiled variant)."""
    out = torch.empty(max(8192, x.shape[0]), W, dtype=torch.bfloat16, device=DEV)
    torch.ops.trtllm.nvfp4_fake_quantize(x, out, None, None)
    return out[: x.shape[0]]


def _latents(n: int, seed: int) -> torch.Tensor:
    """bf16 [n, 512], one magnitude 2^-16..2^6 per 16-block (below the amax floor up to large), a few -0.0."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    mag = torch.randint(-16, 7, (n, W // BLOCK, 1), device=DEV, generator=g).float()
    x = (torch.randn(n, W // BLOCK, BLOCK, device=DEV, generator=g) * torch.exp2(mag)).view(n, W)
    x[:, 5] = -0.0
    return x.to(torch.bfloat16)


@contextmanager
def _native_storage():
    saved = os.environ.get(_STORAGE_ENV)
    os.environ[_STORAGE_ENV] = "native"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(_STORAGE_ENV, None)
        else:
            os.environ[_STORAGE_ENV] = saved


def _v41_native_manager(
    max_num_tokens, max_batch_size, max_draft_len, max_seq_len=None, quota_div=2, pp_stage=None
):
    """A V4.1 manager with native KV storage at the gate settings and total / quota_div of the device's memory
    (0.5 by default). max_seq_len defaults to the gates' 16384, as the engine passes it: + 14 with DSpark."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekV4SparseAttentionConfig,
        DSparkDecodingConfig,
        KvCacheConfig,
    )
    from tensorrt_llm.mapping import Mapping

    total = torch.cuda.mem_get_info()[1]
    assert total == _BMAX_DEVICE_TOTAL, (
        f"B_max constants are for a {_BMAX_DEVICE_TOTAL}-byte GB300, got {total}"
    )
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
        index_head_dim=128,
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
    with _native_storage():
        mgr = DeepseekV4CacheManager(
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                max_gpu_total_bytes=total // quota_div,
                host_cache_size=0,
                event_buffer_max_size=0,
            ),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=40,
            num_kv_heads=1,
            head_dim=512,
            tokens_per_block=128,
            max_seq_len=max_seq_len or (16398 if max_draft_len else 16384),
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
            if pp_stage is None
            else Mapping(world_size=2, rank=pp_stage, tp_size=1, pp_size=2, gpus_per_node=4),
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            vocab_size=129280,
            max_num_tokens=max_num_tokens,
            sparse_attn_config=DeepSeekV4SparseAttentionConfig(
                index_head_dim=128, window_size=128, compress_ratios=[1] * 43
            ),
            pretrained_config=config,
            spec_config=DSparkDecodingConfig(max_draft_len=max_draft_len, block_size=max_draft_len)
            if max_draft_len
            else None,
            max_cuda_graph_batch_size=max_batch_size,
            disable_overlap_scheduler=False,
        )
    assert mgr.v41_layout.kv_storage == "native"
    return mgr


def _ceiling_manager(ratio):
    """The PP2 stage manager whose native COMPRESS view of `ratio` is the envelope's ceiling (NATIVE_COMPRESS),
    built as the stage probe builds it (Mapping(world_size 2, rank = stage, tp_size 1, pp_size 2)); the page
    count is asserted."""
    t, s, d, L = CEILING_KEY
    pages, _, _, owner = NATIVE_COMPRESS[ratio]
    mgr = _v41_native_manager(t, s, d, max_seq_len=L, pp_stage=CEILING_STAGE[ratio])
    try:
        got = _native_view(mgr, owner).shape[0]
        assert got == pages, (ratio, got, pages)
    except BaseException:
        mgr.shutdown()
        raise
    return mgr


def _native_view(mgr, owner):
    """The owner's native COMPRESS view [B, page, 288] uint8, geometry and native views asserted."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    page_size = 128 if owner == 20 else 64
    pool = mgr.get_buffers(owner, AT.COMPRESS)
    codes, scales = mgr.get_v41_native_kv_views(owner, AT.COMPRESS)
    b = pool.shape[0]
    assert pool.dtype == torch.uint8 and tuple(pool.shape) == (b, page_size, ROW), (
        owner,
        pool.shape,
    )
    assert pool.stride() == (page_size * ROW, ROW, 1), (owner, pool.stride())
    assert codes.data_ptr() == pool.data_ptr() and tuple(codes.shape) == (b, page_size, CODE_BYTES)
    assert scales.data_ptr() == pool.data_ptr() + CODE_BYTES and tuple(scales.shape) == (
        b,
        page_size,
        W // BLOCK,
    )
    return pool


def _n_ne(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Device count of differing elements of two same-shape tensors, in chunks (a bool reduction
    materializes an int64 copy, 8 bytes per element)."""
    fa, fb = a.reshape(-1), b.reshape(-1)
    n = torch.zeros((), dtype=torch.int64, device=DEV)
    for i in range(0, fa.numel(), 1 << 26):
        n += (fa[i : i + (1 << 26)] != fb[i : i + (1 << 26)]).sum()
    return n


def _count_not(buf: torch.Tensor, value: int) -> torch.Tensor:
    whole = buf.view(-1)
    step = 1 << 26
    n = torch.zeros((), dtype=torch.int64, device=DEV)
    for i in range(0, whole.numel(), step):
        n += (whole[i : i + step] != value).sum()
    return n


def _hazard_rows(num_rows: int, per_page: int) -> list[int]:
    """Rows of a 288-byte-row pool on both sides of every byte offset 2^31..2^36, both ends, the last page."""
    cands = {0, 1, per_page - 1, per_page}
    for limit in (2**31, 2**32, 2**33, 2**34, 2**35, 2**36):
        s = limit // ROW
        cands |= {s - 1, s, s + 1}
    cands |= set(range(num_rows - per_page - 1, num_rows))
    return sorted(c for c in cands if 0 <= c < num_rows)


def _hazard_pages(pages: int, per_page: int) -> list[int]:
    """Pages on both sides of every byte offset 2^31..2^36 of the [pages, per_page, 288] view, and the ends."""
    cands = {0, 1, pages - 2, pages - 1}
    for limit in (2**31, 2**32, 2**33, 2**34, 2**35, 2**36):
        p = limit // (per_page * ROW)
        cands |= {p - 1, p, p + 1}
    return sorted(c for c in cands if 0 <= c < pages)


def _check_gather(out, rows_table, seq_lens, expected_of_row):
    """out [R, k, 1, W]: entry (r, j) with j < seq_lens[r] and a valid table row equals expected_of_row(row);
    every other entry is still the sentinel."""
    R, k = rows_table.shape
    j = torch.arange(k, device=DEV)[None]
    valid = (rows_table >= 0) & (j < seq_lens[:, None].long())
    got = _bits(out[:, :, 0])
    rows = rows_table[valid].long()
    exp = _bits(expected_of_row(rows))
    n_bad = int(_n_ne(got[valid], exp))
    assert n_bad == 0, f"{n_bad} of {exp.numel()} gathered elements differ bitwise"
    assert bool((got[~valid] == SENTINEL).all()), (
        "an entry outside the table's valid rows was written"
    )
    return int(valid.sum())


@pytest.mark.parametrize("table_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("owner", [2, 14, 20])
def test_native_round_trip_far_end_on_the_manager(owner, table_dtype):
    """On the live PP2 stage managers' ceiling views (CEILING_KEY; stage 0: owner 2 [6480780, 64, 288], owner
    14 two pages shorter; stage 1: owner 20 [3245959, 128, 288]): the writer's native form puts rows at the far end
    (both sides of every 2^31..2^36 byte offset, the ends, the last page) and fills whole pages there
    (the last pages, pages across every byte-offset line); this op reads them back

    - row gather: a [6, 512] table of those rows with -1 padding, ids one past the pool and far past it,
      and one row whose length ends early;
    - whole context: 3 requests of 8192 entries each through [3, n] page tables of those pages, lengths
      8192 / 8191 / 1;

    every read bitwise equal to the writer's core output of the row (and to the LUT reference of the pool
    bytes), unwritten entries untouched, and the pool left exactly as the writer left it.
    """
    mgr = _ceiling_manager("ratio1" if owner == 20 else "ratio2")
    try:
        pool = _native_view(mgr, owner)
        spans = (
            _native_view(mgr, 2) if owner != 20 else pool
        )  # owner 2's view spans the ratio-2 buffer
        spans.fill_(BYTE_SENTINEL)
        pages, per_page = pool.shape[0], pool.shape[1]
        num_rows = pages * per_page
        flat = pool.view(-1, ROW)
        g = torch.Generator(device=DEV).manual_seed(26070 + owner)
        # Whole pages for the context form: the last pages and pages across every byte-offset line.
        L = 8192
        n_pages = L // per_page
        tail = list(range(pages - n_pages, pages))
        haz = _hazard_pages(pages, per_page)
        mid = [
            p
            for p in torch.randperm(pages, device=DEV, generator=g)[: 4 * n_pages].tolist()
            if p not in tail and p not in haz
        ][: n_pages - len(haz)]
        page_lists = [tail, haz + mid, [0] + mid[: n_pages - 1]]
        page_rows = torch.unique(
            torch.cat(
                [
                    torch.tensor(p, device=DEV)[:, None] * per_page
                    + torch.arange(per_page, device=DEV)[None]
                    for p in page_lists
                ]
            ).view(-1)
        )
        rows = torch.unique(
            torch.cat([page_rows, torch.tensor(_hazard_rows(num_rows, per_page), device=DEV)])
        )
        x = _latents(rows.numel(), seed=26071 + owner)
        _write(x, pool, rows)
        y = _core(x)

        def expected_of_row(r):
            idx = torch.searchsorted(rows, r)
            assert torch.equal(rows[idx], r)
            return y[idx]

        gs = _gs()
        # Row gather: far-end rows, -1, past-the-end ids.
        hz = torch.tensor(_hazard_rows(num_rows, per_page), device=DEV)
        R = 6
        tab = torch.full((R, TOPK), -1, dtype=torch.int64, device=DEV)
        pick = torch.cat(
            [hz, rows[torch.randperm(rows.numel(), device=DEV, generator=g)][: R * TOPK]]
        )
        tab.view(-1)[: R * TOPK] = pick[: R * TOPK]
        tab[0, 3], tab[0, 5], tab[0, 7] = -1, num_rows, num_rows + 1000
        tab[1, TOPK // 2 :] = -1
        sl = torch.full((R,), TOPK, dtype=torch.int32, device=DEV)
        sl[2] = TOPK - 37
        kv, sf = _row_views(pool)
        out = _sentinel_out(R, TOPK, 1, W)
        op(kv, sf, tab.to(table_dtype), sl, gs, out)
        tab_ok = torch.where((tab >= 0) & (tab < num_rows), tab, -1)
        n_rg = _check_gather(out, tab_ok, sl, expected_of_row)
        twin = _sentinel_out(R, TOPK, 1, W)
        _twin(kv, sf, tab.to(table_dtype), sl, gs, twin)
        assert torch.equal(_bits(out), _bits(twin)), "row gather differs from the FlashInfer twin"
        assert torch.equal(
            _bits(
                out[:, :, 0][(tab_ok >= 0) & (torch.arange(TOPK, device=DEV)[None] < sl[:, None])]
            ),
            _bits(
                _ref_rows(
                    flat,
                    tab_ok[
                        (tab_ok >= 0) & (torch.arange(TOPK, device=DEV)[None] < sl[:, None])
                    ].long(),
                )
            ),
        )
        # Whole context through page tables of those pages.
        bt = torch.tensor(
            [p[:n_pages] + [-1] * (n_pages - len(p[:n_pages])) for p in page_lists], device=DEV
        )
        lens = torch.tensor([L, L - 1, 1], dtype=torch.int32, device=DEV)
        kvp, sfp = _page_views(pool)
        out2 = _sentinel_out(3, L, 1, W)
        op(kvp, sfp, bt.to(table_dtype), lens, gs, out2)
        t = torch.arange(L, device=DEV)
        ctx_rows = torch.where(
            bt[:, t // per_page] >= 0, bt[:, t // per_page] * per_page + t % per_page, -1
        )
        n_ctx = _check_gather(out2, ctx_rows, lens, expected_of_row)
        twin = _sentinel_out(3, L, 1, W)
        _twin(kvp, sfp, bt.to(table_dtype), lens, gs, twin)
        assert torch.equal(_bits(out2), _bits(twin)), (
            "whole context differs from the FlashInfer twin"
        )
        # The pool is exactly as the writer left it.
        _, packed, s_bytes = _reference_native(x)
        assert torch.equal(flat[rows], torch.cat([packed, s_bytes], 1))
        flat[rows] = BYTE_SENTINEL
        assert int(_count_not(spans, BYTE_SENTINEL)) == 0, "a read changed the native pool"
        print(
            f"\n[owner {owner} view {tuple(pool.shape)}, {table_dtype}] {rows.numel()} far-end rows written; "
            f"row gather {n_rg} entries and whole context {n_ctx} entries == the writer's core output"
        )
    finally:
        mgr.shutdown()
        del mgr
        torch.cuda.empty_cache()


def _reference_native(x):
    """(bf16 core values, packed codes [n, 256], E4M3 bytes [n, 32]) of the checkpoint's NVFP4 recipe
    (amax floor 6 * 2^-9, scale e4m3(amax / 6), E2M1 round-to-nearest-even), in native torch."""
    xf = x.float().unflatten(-1, (-1, BLOCK))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0**-9)
    s8 = (amax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    s = s8.float()
    y = (xf / s).clamp(-6.0, 6.0)
    a = y.abs()
    r = torch.where(
        a < 2, torch.round(a * 2) / 2, torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2)
    )
    deq = (torch.copysign(r, y) * s).flatten(-2).to(torch.bfloat16)
    code = torch.where(r < 2, r * 2, torch.where(r == 6, 7.0, r + 2)).to(torch.int32)
    nib = (code | (torch.signbit(y).to(torch.int32) << 3)).flatten(-2)
    packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8)
    return deq, packed, s8.view(torch.uint8).squeeze(-1)


def test_native_harness_sees_a_wrapped_offset():
    """Control: tables naming the rows / pages where a mod-2^32 byte offset would land read other rows, and
    the gate sees it in both forms."""
    pages, per_page, _, owner = NATIVE_COMPRESS["ratio2"]
    buf = torch.full((pages, per_page, ROW), BYTE_SENTINEL, dtype=torch.uint8, device=DEV)
    num_rows = pages * per_page
    true = torch.tensor(
        [s for s in _hazard_rows(num_rows, per_page) if s * ROW >= 2**32], device=DEV
    )
    wrapped = (true * ROW % 2**32) // ROW
    assert true.numel() >= 9 and not torch.isin(wrapped, true).any()
    x = _latents(2 * true.numel(), seed=26072)
    _write(x, buf, torch.cat([true, wrapped]))
    y = _core(x)[: true.numel()]
    kv, sf = _row_views(buf)
    out = _sentinel_out(1, true.numel(), 1, W)
    op(
        kv,
        sf,
        wrapped.view(1, -1).to(torch.int32),
        torch.tensor([true.numel()], dtype=torch.int32, device=DEV),
        _gs(),
        out,
    )
    n_bad = int((_bits(out[0, :, 0]) != _bits(y)).sum())
    tp = torch.tensor(
        [p for p in _hazard_pages(pages, per_page) if p * per_page * ROW >= 2**32], device=DEV
    )
    wp = (tp * per_page * ROW % 2**32) // (per_page * ROW)
    xp = _latents(2 * tp.numel() * per_page, seed=26073)
    prow = torch.cat([tp, wp])[:, None] * per_page + torch.arange(per_page, device=DEV)[None]
    _write(xp, buf, prow.view(-1))
    yp = _core(xp)[: tp.numel() * per_page]
    kvp, sfp = _page_views(buf)
    out2 = _sentinel_out(1, tp.numel() * per_page, 1, W)
    op(
        kvp,
        sfp,
        wp.view(1, -1).to(torch.int32),
        torch.tensor([tp.numel() * per_page], dtype=torch.int32, device=DEV),
        _gs(),
        out2,
    )
    n_bad2 = int((_bits(out2[0, :, 0]) != _bits(yp)).sum())
    print(
        f"\n[wrapped control] row gather {n_bad} of {out[0, :, 0].numel()}, whole context {n_bad2} of "
        f"{out2[0, :, 0].numel()} elements differ from the true rows"
    )
    assert n_bad > out[0, :, 0].numel() // 2 and n_bad2 > out2[0, :, 0].numel() // 2
    del buf
    torch.cuda.empty_cache()


@pytest.mark.parametrize("n", [32, 8192])
@pytest.mark.parametrize("ratio", ["ratio2", "ratio1"])
def test_native_pool_size_invariance(ratio, n):
    """The same rows at the same page indices read through views [:B] of the live PP2 stage manager's ceiling
    view (CEILING_KEY on CEILING_STAGE): the same bits at every B (row gather over pages of one row --
    the per-query form's page-size-1 view -- and whole context over the real pages, the per-request
    form) and equal to the writer's core output; then from a separate pool of exactly the smallest size.

    B = the smallest view holding the rows, the page counts whose byte count first passes 2^31..2^36
    (where below the ceiling), a mid size, the gates' largest, _EXTRA_B (the gates' axis B_max and envelope
    managers' views) and the ceiling.
    """
    pages, per_page, gate_max, owner = NATIVE_COMPRESS[ratio]
    page_bytes = per_page * ROW
    mgr = _ceiling_manager(ratio)
    try:
        pool = _native_view(mgr, owner)
        pool.fill_(BYTE_SENTINEL)
        small = -(-n // per_page)
        g = torch.Generator(device=DEV).manual_seed(26074 + n + per_page)
        order = torch.randperm(small * per_page, device=DEV, generator=g)
        rows = torch.cat(
            [torch.tensor([small * per_page - 1], device=DEV), order[order != small * per_page - 1]]
        )[:n]
        x = _latents(n, seed=26074 + n)
        _write(x, pool, rows)
        y = _core(x)
        tab = rows.view(1, n).to(torch.int32)
        sl = torch.tensor([n], dtype=torch.int32, device=DEV)
        bt = torch.randperm(small, device=DEV, generator=g).view(1, -1).to(torch.int32)
        L = small * per_page
        t = torch.arange(L, device=DEV)
        ctx_rows = bt[0].long()[t // per_page] * per_page + t % per_page
        sizes = sorted(
            {small, pages // 2 + 1, gate_max, pages}
            | set(_EXTRA_B[ratio])
            | {
                lim // page_bytes + 1
                for lim in (2**31, 2**32, 2**33, 2**34, 2**35, 2**36)
                if lim // page_bytes + 1 < pages
            }
        )
        first = None
        for b in sizes:
            kv, sf = _row_views(pool[:b])
            out = _sentinel_out(1, n, 1, W)
            op(kv, sf, tab, sl, _gs(), out)
            kvp, sfp = _page_views(pool[:b])
            out2 = _sentinel_out(1, L, 1, W)
            op(kvp, sfp, bt, torch.tensor([L], dtype=torch.int32, device=DEV), _gs(), out2)
            got = torch.cat([_bits(out[0, :, 0]), _bits(out2[0, :, 0])])
            assert torch.equal(_bits(out[0, :, 0]), _bits(y))
            assert torch.equal(_bits(out2[0, :, 0]), _bits(_ref_rows(pool.view(-1, ROW), ctx_rows)))
            first = got if first is None else first
            assert torch.equal(got, first), f"B={b} differs from B={sizes[0]}"
        exact = pool.view(-1, ROW)[: small * per_page].clone().view(small, per_page, ROW)
    finally:
        mgr.shutdown()
        del mgr
        torch.cuda.empty_cache()
    kv, sf = _row_views(exact)
    out = _sentinel_out(1, n, 1, W)
    op(kv, sf, tab, sl, _gs(), out)
    kvp, sfp = _page_views(exact)
    out2 = _sentinel_out(1, L, 1, W)
    op(kvp, sfp, bt, torch.tensor([L], dtype=torch.int32, device=DEV), _gs(), out2)
    assert torch.equal(torch.cat([_bits(out[0, :, 0]), _bits(out2[0, :, 0])]), first)
    print(
        f"\n[native {ratio} n={n}] B = {sizes} (views of owner {owner}'s view) and a separate [{small}, "
        f"{per_page}, 288]: both forms bitwise identical and == the writer's core output"
    )


def test_native_every_row_count_1_to_16384_row_gather():
    """Over the query rows of the row-gather form (the per-query form's T, up to the envelope's
    max_num_tokens 16,384): every R in 1..16384 at the target's width 512 reads exactly the first R rows of
    the 16384-row call.

    The pool: 16,384 native rows written by the writer. The table: [16384, 512] random rows with -1
    padding (every 5th row shortened), one row per query; seq_lens 512 (every 7th row shorter). Before
    each call out[:R] is reset to the sentinel; after it out[:R] must equal the first R rows of the full
    call (which equals the writer's core output where the table names a row and the sentinel elsewhere).
    """
    g = torch.Generator(device=DEV).manual_seed(26075)
    pool = torch.full((NUM_SLOTS // 64, 64, ROW), BYTE_SENTINEL, dtype=torch.uint8, device=DEV)
    x = _latents(NUM_SLOTS, seed=26075)
    _write(x, pool, torch.arange(NUM_SLOTS, device=DEV))
    y = _core(x)
    rmax = 16384
    tab = torch.randint(0, NUM_SLOTS, (rmax, TOPK), device=DEV, generator=g, dtype=torch.int32)
    cut = torch.randint(0, TOPK, (rmax,), device=DEV, generator=g)
    short = (torch.arange(rmax, device=DEV) % 5 == 0)[:, None] & (
        torch.arange(TOPK, device=DEV)[None] >= cut[:, None]
    )
    tab[short] = -1
    sl = torch.full((rmax,), TOPK, dtype=torch.int32, device=DEV)
    sl[::7] = torch.randint(1, TOPK, (sl[::7].numel(),), device=DEV, generator=g, dtype=torch.int32)
    kv, sf = _row_views(pool)
    out = _sentinel_out(rmax, TOPK, 1, W)
    op(kv, sf, tab, sl, _gs(), out)
    _check_gather(out, tab, sl, lambda r: y[r])
    full = out.clone()
    out.view(torch.int16).fill_(SENTINEL)
    _twin(kv, sf, tab, sl, _gs(), out)
    assert torch.equal(_bits(out), _bits(full)), (
        "the 16384-row call differs from the FlashInfer twin"
    )
    n1 = _jit_variants()
    o16, f16 = out.view(torch.int16), full.view(torch.int16)
    mism = torch.zeros((), dtype=torch.int64, device=DEV)
    for r in range(1, rmax + 1):
        o16[:r] = SENTINEL
        op(kv, sf, tab[:r], sl[:r], _gs(), out[:r])
        mism += _n_ne(o16[:r], f16[:r])
    n2 = _jit_variants()
    print(
        f"\n[row gather R = 1..{rmax} x {TOPK}] prefix mismatches {int(mism)}; compiled variants {n1} -> {n2}"
    )
    assert int(mism) == 0 and n2 == n1


_ENVELOPE_GSET = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_ENVELOPE_GRAPH_R = tuple(sorted(set(_ENVELOPE_GSET) | {6 * b for b in _ENVELOPE_GSET}))


@pytest.mark.parametrize("ratio", ["ratio2", "ratio1"])
def test_native_cuda_graph_every_graph_batch_on_the_bmax_view(ratio):
    """Row gather (the per-query form) at every query-row count a captured step makes over the engine
    envelope, R in G and 6 G for every G of _ENVELOPE_GSET (1..1536; this holds the gates' {1..8, 16, 24, 32} and 6 x
    those), width 512, on the live PP2 stage manager's ceiling view: per R an eager warm-up, the capture, then 2 replays
    after the writer has put new rows at new far-end rows and the table (-1 entries included), the lengths
    and the pool changed in place; every replay equals the writer's core output of the named rows, leaves
    the rest of out (the -1 entries' rows included) untouched, and equals the eager call. The view is the
    CEILING_KEY manager's."""
    pages, per_page, _, owner = NATIVE_COMPRESS[ratio]
    mgr = _ceiling_manager(ratio)
    try:
        pool = _native_view(mgr, owner)
        num_rows = pages * per_page
        g = torch.Generator(device=DEV).manual_seed(26077 + per_page)
        cand = torch.unique(
            torch.cat(
                [
                    torch.tensor(_hazard_rows(num_rows, per_page), device=DEV),
                    num_rows - 1 - torch.arange(64 * per_page * 8, device=DEV),
                ]
            )
        )
        kv, sf = _row_views(pool)
        gs = _gs()

        def fresh(R):
            rows = cand[torch.randperm(cand.numel(), device=DEV, generator=g)][: R * TOPK // 4]
            x = _latents(
                rows.numel(), seed=int(torch.randint(0, 2**30, (1,), generator=g, device=DEV))
            )
            _write(x, pool, rows)
            y = _core(x)
            srt, idx = torch.sort(rows)
            tab = rows[torch.randint(0, rows.numel(), (R, TOPK), device=DEV, generator=g)]
            tab[:, TOPK - 13 :] = -1
            sl = torch.full((R,), TOPK, dtype=torch.int32, device=DEV)
            sl[0] = TOPK - 100
            return tab.to(torch.int32), sl, lambda r: y[idx[torch.searchsorted(srt, r)]]

        n1 = None
        for R in _ENVELOPE_GRAPH_R:
            tab, sl, _ = fresh(R)
            out = _sentinel_out(R, TOPK, 1, W)
            op(kv, sf, tab, sl, gs, out)
            torch.cuda.synchronize()
            n1 = _jit_variants() if n1 is None else n1
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op(kv, sf, tab, sl, gs, out)
            for rep in (1, 2):
                new_tab, new_sl, exp = fresh(R)
                tab.copy_(new_tab)
                sl.copy_(new_sl)
                out.view(torch.int16).fill_(SENTINEL)
                graph.replay()
                torch.cuda.synchronize()
                _check_gather(out, tab, sl, exp)
                replayed = out.clone()
                out.view(torch.int16).fill_(SENTINEL)
                op(kv, sf, tab, sl, gs, out)
                assert torch.equal(_bits(out), _bits(replayed)), (
                    f"R={R} replay {rep} differs from eager"
                )
            del graph
        n2 = _jit_variants()
        print(
            f"\n[{ratio} graphs, owner {owner} view {tuple(pool.shape)}] R = {list(_ENVELOPE_GRAPH_R)} x {TOPK} "
            f"({len(_ENVELOPE_GRAPH_R)} graphs): every replay == eager == the writer's core output; compiled variants "
            f"after the first eager call {n1} -> {n2}"
        )
        assert n2 == n1, "a row count or a capture compiled a new variant"
    finally:
        mgr.shutdown()
        del mgr
        torch.cuda.empty_cache()


_NUM_RANDOM_STEPS = 200


def _run_step(mgr, reqs, num_contexts, gen_len, uid, max_num_tokens=8192):
    """Allocate the step's requests on the manager and build its DeepSeek-V4.1 attention metadata as the
    engine builds it (max_num_requests = the manager's max_batch_size, the engine's max_num_tokens; each
    request's first token unique over the test, so block reuse shares no page across steps)."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
        DeepseekV4TrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import SamplingConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig
    from tensorrt_llm.mapping import Mapping

    requests = []
    for rid, (c, n) in enumerate(reqs):
        r = LlmRequest(
            request_id=rid,
            max_new_tokens=64,
            input_tokens=[uid + rid] + list(range(1, c + n)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert mgr.prepare_context(r) and mgr.resize_context(r, r.context_chunk_size), (rid, c, n)
        requests.append(r)
    md = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=torch.tensor([n for _, n in reqs], dtype=torch.int),
        request_ids=[r.py_request_id for r in requests],
        max_num_requests=mgr.max_batch_size,
        num_contexts=num_contexts,
        prompt_lens=[c + n for c, n in reqs],
        max_num_tokens=max_num_tokens,
        kv_cache_manager=mgr,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[c for c, _ in reqs]
        ),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        sparse_attention_config=DeepSeekV4SparseAttentionConfig(
            index_head_dim=128, window_size=128, compress_ratios=[1] * 43
        ),
    )
    if num_contexts < len(reqs) and gen_len > 1:
        md.update_spec_dec_param(
            batch_size=len(reqs),
            is_spec_decoding_enabled=False,
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=gen_len - 1,
            max_total_draft_tokens=gen_len - 1,
        )
    md.prepare()
    return requests, md


_KVQ_COLS = (
    132  # md.compress_block_tables[r].shape[1] (max_blocks_per_seq) in every envelope configuration
)
_KVQ_L = {2: _KVQ_COLS * 64, 1: _KVQ_COLS * 128}  # 8,448 / 16,896 entries per request in the buffer
_KVQ_OWNERS = (
    (2, 2),
    (8, 2),
    (14, 2),
    (20, 1),
)  # the KV sources in forward order, with their ratio
_KVQ_DSPARK_ENTRIES = {
    2: 16398 // 2,
    1: 16398,
}  # entries at the engine's max_seq_len 16398 (DSpark)
# The max_batch_size axis. The compressed read buffer is max_batch_size * 132 * 128 rows
# (16,896 per request), the only native-storage object whose size follows max_batch_size; every value 1..32 the
# engine configuration permits is run, the heavier sweeps at the listed values, and the native-storage witness's
# own configuration (max_num_tokens 2048, max_batch_size 1, DSpark) explicitly.


def _kvq_random_step(rng, max_tokens=8192, max_requests=32, seq_len=16384):
    """_random_step over the engine's envelope with DSpark: contexts end at <= seq_len tokens, decode
    requests (next_n 1) at <= seq_len, DSpark-verify requests (next_n 6) at <= seq_len + 14 (16398 at the
    gates' 16384)."""
    k = rng.randint(1, max_requests)
    gen_len = rng.choice([1, 6])
    g = rng.randint(0, k) if rng.random() < 0.7 else 0
    c = k - g
    budget = max_tokens - g * gen_len
    t_ctx = rng.randint(c, budget) if c else 0
    cuts = sorted(rng.sample(range(1, t_ctx), c - 1)) if c > 1 else []
    news = [b - a for a, b in zip([0] + cuts, cuts + [t_ctx])] if c else []
    reqs = [(0 if rng.random() < 0.4 else rng.randint(0, seq_len - n), n) for n in news]
    rng.shuffle(reqs)
    end = seq_len + 14 if gen_len == 6 else seq_len
    reqs += [(rng.randint(0, end - gen_len), gen_len) for _ in range(g)]
    return reqs, c, gen_len


def _kvq_codes_scales(mgr, owner):
    """The target's operands: the owner's native (codes, scales) COMPRESS views reshaped to
    [pages, 128 / r, 1, 256] and [pages, 128 / r, 1, 32] (views, asserted)."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as AT,
    )

    codes, scales = mgr.get_v41_native_kv_views(owner, AT.COMPRESS)
    c4 = codes.reshape(codes.shape[0], codes.shape[1], 1, CODE_BYTES)
    s4 = scales.reshape(scales.shape[0], scales.shape[1], 1, W // BLOCK)
    assert c4.data_ptr() == codes.data_ptr() and s4.data_ptr() == scales.data_ptr()
    return c4, s4


def _kvq_page_candidates(pages, per_page, n, g):
    """n distinct pages of a [pages, per_page, 288] view: every page across a 2^31..2^35 byte line and both
    ends in random order, then the view's last pages and random pages alternately."""
    hz = torch.tensor(_hazard_pages(pages, per_page), device=DEV)
    hz = hz[torch.randperm(hz.numel(), device=DEV, generator=g)]
    last = pages - 1 - torch.arange(n, device=DEV)
    last = last[~torch.isin(last, hz)]
    rnd = torch.randint(0, pages, (4 * n + 64,), device=DEV, generator=g).unique()
    rnd = rnd[~torch.isin(rnd, torch.cat([hz, last]))]
    rnd = rnd[torch.randperm(rnd.numel(), device=DEV, generator=g)][: last.numel()]
    tail = torch.stack([last[: rnd.numel()], rnd], 1).reshape(-1)
    cand = torch.cat([hz, tail, last[rnd.numel() :]])[:n]
    assert cand.numel() == n and torch.unique(cand).numel() == n
    return cand.to(torch.int64)


def _mbs_edge_steps(max_t, seqs):
    """Edge steps of at most seqs requests: a full-T context, 1 token at 0 and at 16383, one DSpark verify
    at kv_len 16398, then full batches of seqs requests (contexts ending at 16384 with T = seqs *
    (max_t // seqs), decodes at 16384, verifies at 16398) and, when seqs >= 2, one context filling T with
    seqs - 1 verifies at 16398."""
    n = max_t // seqs
    steps = [
        ([(0, max_t)], 1, 1),
        ([(0, 1)], 1, 1),
        ([(16383, 1)], 1, 1),
        ([(16392, 6)], 0, 6),
        ([(16384 - n, n)] * seqs, seqs, 1),
        ([(16383, 1)] * seqs, 0, 1),
        ([(16392, 6)] * seqs, 0, 6),
    ]
    if seqs > 1:
        c = max_t - 6 * (seqs - 1)
        steps.append(([(16384 - c, c)] + [(16392, 6)] * (seqs - 1), 1, 6))
    return steps


def _kvq_per_query_ops(mgr, owner):
    """The per-query form's operands: the owner's native (codes, scales) COMPRESS views re-viewed with page
    size 1, [pages * (128 / r), 1, 1, 256] and [.., 1, 1, 32], so a pool slot page * (128 / r) + e is its
    own page index (views, asserted)."""
    c4, s4 = _kvq_codes_scales(mgr, owner)
    c1 = c4.view(c4.shape[0] * c4.shape[1], 1, 1, CODE_BYTES)
    s1 = s4.view(s4.shape[0] * s4.shape[1], 1, 1, W // BLOCK)
    assert c1.data_ptr() == c4.data_ptr() and s1.data_ptr() == s4.data_ptr() and c1.stride(0) == ROW
    return c1, s1


def _kvq_renumbered_steps(mgr, steps, max_t, seqs, seed, cols=_KVQ_COLS, per_query=False):
    """The body of test_kvq_compressed_read_renumbered_over_random_steps for one manager: per step and per
    source in forward order, the writer's native writes, the target's read into ONE buffer sized from the
    step's metadata (asserted [seqs, cols]; cols 132 at the gates' max_seq_len) and kept across sources and
    steps, the buffer checked against the bf16 container and the renumbered top-k against the bf16 path, and
    the three controls. With per_query, each source is also read in the per-query form (see
    test_envelope_compressed_read_renumbered_over_random_steps_256_requests). Returns (stats, controls
    [source-steps detected, applicable, elements], buffer shape)."""
    Lr = {r: cols * 128 // r for r in (1, 2)}
    comp_c = q_out = None
    views = {o: _native_view(mgr, o) for o, _ in _KVQ_OWNERS}
    ops = {o: _kvq_codes_scales(mgr, o) for o, _ in _KVQ_OWNERS}
    ops1 = {o: _kvq_per_query_ops(mgr, o) for o, _ in _KVQ_OWNERS} if per_query else None
    one = _gs()
    stats = dict(steps=0, calls=0, max_t=0, max_k=0, entries=0, topk_rows=0, max_len={1: 0, 2: 0})
    ctrl = {
        "renumbered off by one": [0, 0, 0],
        "other ratio's row stride": [0, 0, 0],
        "stale buffer row": [0, 0, 0],
    }
    if per_query:
        q_out = torch.zeros((max_t, TOPK, 1, W), dtype=torch.bfloat16, device=DEV)
        q_cols = torch.full((max_t,), TOPK, dtype=torch.int32, device=DEV)
        stats.update(pq_calls=0, pq_rows=0, pq_minus1=0, pq_max_t=0)
        ctrl.update({"per-query row off by one": [0, 0, 0], "per-query stale row": [0, 0, 0]})
    chunk = 1 << 20
    try:
        for i, (reqs, nctx, gen_len) in enumerate(steps):
            requests, md = _run_step(
                mgr, reqs, nctx, gen_len, uid=1_000_000 + 64 * i, max_num_tokens=max_t
            )
            try:
                s_, c_ = md.compress_block_tables[1].shape
                assert (s_, c_) == (seqs, cols) and tuple(md.compress_block_tables[2].shape) == (
                    s_,
                    c_,
                )
                if comp_c is None:
                    comp_c = torch.zeros((s_ * cols * 128, W), dtype=torch.bfloat16, device=DEV)
                    comp_out = {r: comp_c[: s_ * Lr[r]].view(s_, Lr[r], 1, W) for r in (2, 1)}
                    comp_tbl = torch.arange(s_ * cols, dtype=torch.int32, device=DEV).view(s_, cols)
                t, k = sum(n for _, n in reqs), len(reqs)
                B = md.num_seqs
                assert B == k <= seqs and t <= max_t
                req = md.req_idx_per_token[:t].long()
                gsd = torch.Generator(device=DEV).manual_seed(seed + i)
                for o, r in _KVQ_OWNERS:
                    bk, L = 128 // r, Lr[r]
                    ent = [(c + n) // r for c, n in reqs]
                    lens = md.compressed_kv_lens_cuda[r][:B]
                    assert lens.tolist() == ent, (i, r)
                    cbt = md.compress_block_tables[r]
                    lmax = max(max(ent), 1)
                    e = torch.arange(lmax, device=DEV)
                    page = cbt[:B, (e // bk).clamp(max=cols - 1)].long()
                    row_of = page * bk + e % bk
                    has = e[None] < torch.tensor(ent, device=DEV)[:, None]
                    assert bool((page[has] >= 0).all()), f"step {i} r={r}: an entry without a page"
                    rows = row_of[has]
                    y = torch.empty((0, W), dtype=torch.bfloat16, device=DEV)
                    if rows.numel():
                        x = _latents(rows.numel(), seed=seed + 1_000_000 + 4 * i + o)
                        _write(x, views[o], rows)
                        y = _core(x)
                    srt, idx = torch.sort(rows)
                    assert rows.numel() == 0 or bool((srt[1:] != srt[:-1]).all())
                    before = comp_c.clone()
                    op(*ops[o], cbt[:B], lens, one, comp_out[r][:B])
                    expected = before.clone()
                    expected[(torch.arange(B, device=DEV)[:, None] * L + e[None])[has]] = y
                    n_bad = int(_n_ne(_bits(comp_c), _bits(expected)))
                    assert n_bad == 0, f"step {i} source {o}: {n_bad} buffer elements differ"
                    del expected
                    vis = md.compressed_visible_lens_cuda[r][:t].long()
                    kk = min(TOPK, lmax)
                    scores = torch.rand(t, lmax, device=DEV, generator=gsd)
                    scores[e[None] >= vis[:, None]] = -1.0
                    top_s, top_e = torch.topk(scores, kk, dim=1)
                    ok = top_s >= 0
                    ren = comp_tbl[req[:, None], top_e // bk].long() * bk + top_e % bk
                    assert torch.equal(ren[ok], (req[:, None] * L + top_e)[ok])
                    ren_ok, eb = ren[ok], req[:, None].expand_as(top_e)[ok]
                    counts = {
                        name: 0
                        for name in (
                            "renumbered off by one",
                            "other ratio's row stride",
                            "stale buffer row",
                        )
                    }
                    if ren_ok.numel():
                        ref_idx = idx[torch.searchsorted(srt, row_of[eb, top_e[ok]])]
                        other = eb * Lr[3 - r] + top_e[ok]
                    for s0 in range(0, ren_ok.numel(), chunk):
                        sl = slice(s0, s0 + chunk)
                        ref = _bits(y[ref_idx[sl]])
                        n_bad = int(_n_ne(_bits(comp_c[ren_ok[sl]]), ref))
                        assert n_bad == 0, (
                            f"step {i} source {o}: {n_bad} top-k elements differ from the bf16 path"
                        )
                        counts["renumbered off by one"] += int(
                            _n_ne(_bits(comp_c[ren_ok[sl] + 1]), ref)
                        )
                        counts["other ratio's row stride"] += int(
                            _n_ne(_bits(comp_c[other[sl].clamp(max=comp_c.shape[0] - 1)]), ref)
                        )
                        counts["stale buffer row"] += int(_n_ne(_bits(before[ren_ok[sl]]), ref))
                    applicable = {
                        "renumbered off by one": ren_ok.numel() > 0,
                        "other ratio's row stride": bool((eb > 0).any()),
                        "stale buffer row": ren_ok.numel() > 0,
                    }
                    for name, c in counts.items():
                        ctrl[name][0] += int(applicable[name] and c > 0)
                        ctrl[name][1] += int(applicable[name])
                        ctrl[name][2] += c
                    del before
                    if per_query:
                        # The per-query form: the index op's pool slots, -1 where a query has no entry, as the
                        # page table of the page-size-1 view, 512 per query, into ONE persistent zero-filled
                        # [max_t, 512, 1, 512] buffer; query q's column j lands at row q * 512 + j.
                        slots = torch.full((t, TOPK), -1, dtype=torch.int32, device=DEV)
                        slots[:, :kk] = torch.where(ok, row_of[req[:, None], top_e], -1).to(
                            torch.int32
                        )
                        qf = q_out.view(-1, W)
                        q_before = qf.clone()
                        op(*ops1[o], slots, q_cols[:t], one, q_out[:t])
                        valid = torch.zeros((t, TOPK), dtype=torch.bool, device=DEV)
                        valid[:, :kk] = ok
                        qi = torch.arange(t * TOPK, device=DEV).view(t, TOPK)[valid]
                        pcnt = {"per-query row off by one": 0, "per-query stale row": 0}
                        for s0 in range(0, qi.numel(), chunk):
                            sl = slice(s0, s0 + chunk)
                            ref = _bits(y[ref_idx[sl]])
                            n_bad = int(_n_ne(_bits(qf[qi[sl]]), ref))
                            assert n_bad == 0, (
                                f"step {i} source {o}: {n_bad} per-query elements differ from the bf16 path"
                            )
                            pcnt["per-query row off by one"] += int(
                                _n_ne(_bits(qf[(qi[sl] + 1).clamp(max=qf.shape[0] - 1)]), ref)
                            )
                            pcnt["per-query stale row"] += int(_n_ne(_bits(q_before[qi[sl]]), ref))
                        minus = torch.arange(t * TOPK, device=DEV).view(t, TOPK)[~valid]
                        for s0 in range(0, minus.numel(), chunk):
                            sl = slice(s0, s0 + chunk)
                            n_bad = int(_n_ne(_bits(qf[minus[sl]]), _bits(q_before[minus[sl]])))
                            assert n_bad == 0, (
                                f"step {i} source {o}: {n_bad} elements of -1 entries' rows written"
                            )
                        n_tail = int(_n_ne(_bits(qf[t * TOPK :]), _bits(q_before[t * TOPK :])))
                        assert n_tail == 0, (
                            f"step {i} source {o}: {n_tail} elements past query {t} written"
                        )
                        for name, c in pcnt.items():
                            ctrl[name][0] += int(qi.numel() > 0 and c > 0)
                            ctrl[name][1] += int(qi.numel() > 0)
                            ctrl[name][2] += c
                        stats["pq_calls"] += 1
                        stats["pq_rows"] += qi.numel()
                        stats["pq_minus1"] += minus.numel()
                        stats["pq_max_t"] = max(stats["pq_max_t"], t)
                        del q_before, qi, minus, valid, slots
                    stats["calls"] += 1
                    stats["entries"] += rows.numel()
                    stats["topk_rows"] += ren_ok.numel()
                    stats["max_len"][r] = max(stats["max_len"][r], max(ent))
                stats["steps"] += 1
                stats["max_t"] = max(stats["max_t"], t)
                stats["max_k"] = max(stats["max_k"], k)
            finally:
                for r_ in requests:
                    mgr.free_resources(r_)
                torch.cuda.empty_cache()
        return stats, ctrl, tuple(comp_c.shape)
    finally:
        comp_c = q_out = None


# ─── the engine envelope: max_seq_len to 1,048,576, max_batch_size and requests per step to 256 ───
#
# The target's call is the one above, unchanged:
#   nvfp4_kv_to_bf16(codes4, scales4, md.compress_block_tables[r][:B], md.compressed_kv_lens_cuda[r][:B], one,
#                       comp_out[r][:B])
# comp_out[r] is the [seqs, cols * 128 / r, 1, 512] view of one buffer of seqs * cols * 128 rows, with
# (seqs, cols) = md.compress_block_tables[r].shape. The envelope moves three of its numbers: seqs (the engine's
# max_batch_size) and B (the step's requests; under capture every G of the padded graph-batch list) up to 256,
# and cols (the manager's max_blocks_per_seq) with max_seq_len. The buffer is seqs * cols * 128 KiB: 256 GiB at
# (256, 8196), more than a GB300 holds beside a pool, so the 1M-width tests stop at 128 requests.

_ENVELOPE_L = (
    16384,
    20480,
    32768,
    65536,
    131072,
    262144,
    1048576,
)  # max_seq_len; 20480 is the decoder-replay guard run's
_ENVELOPE_COLS = {
    16384: 132,
    20480: 164,
    32768: 260,
    65536: 516,
    131072: 1028,
    262144: 2052,
    1048576: 8196,
}
_ENVELOPE_SEQS = {
    L: 128 if L == 1048576 else 256 for L in _ENVELOPE_L
}  # the buffer's requests: <= 128.1 GiB
assert len(_ENVELOPE_GSET) == 46
_ENVELOPE_CAND = 16384  # distinct pool pages a width test draws its tables from (every request a new permutation)
# Widths whose every request length l in 1..L_r is run at B = 1: all of them. The call at l = L_r, B = 1 stays
# under 1 ms at every width, so the sampled length set does not apply (the time is printed).
_ENVELOPE_EVERY_LENGTH = set(_ENVELOPE_L)
# Pages of the native COMPRESS views (owner 2 at ratio 2, owner 20 at ratio 1) of the manager at 0.5 x this
# GB300's total memory, per (DSpark max_draft_len, max_seq_len) at max_num_tokens 16384, max_batch_size 256.
_ENVELOPE_PAGES = {
    (0, 16384): (2_703_360, 901_120),
    (5, 16384): (2_408_448, 802_816),
    (0, 20480): (2_880_852, 960_284),
    (5, 20480): (2_606_019, 868_673),
    (0, 32768): (3_192_147, 1_064_049),
    (5, 32768): (2_974_605, 991_535),
    (0, 65536): (3_515_277, 1_171_759),
    (5, 65536): (3_371_463, 1_123_821),
    (0, 131072): (3_700_854, 1_233_618),
    (5, 131072): (3_619_899, 1_206_633),
    (0, 262144): (3_802_908, 1_267_636),
    (5, 262144): (3_759_216, 1_253_072),
    (0, 1048576): (3_877_545, 1_292_515),
    (5, 1048576): (3_869_352, 1_289_784),
}


def test_envelope_read_buffer_widths_from_the_metadata():
    """The table width of the target's call at every max_seq_len of the envelope, DSpark off and on. On the
    native manager at (max_num_tokens 16384, max_batch_size 256, max_draft_len 0 or 5) with max_seq_len L
    (L + 14 with DSpark, as the engine passes it): max_blocks_per_seq is cols(L), and metadata built as the
    engine builds it -- the step's, and its CUDA-graph copies at G = 1 and G = 256 -- has
    compress_block_tables[1] and [2] of [256, cols] int32 (contiguous) and compressed_kv_lens_cuda[r] of [256]
    int32. So out is [B, cols * 128 / r, 1, 512] with the same cols DSpark on and off. The views' page counts
    at 0.5 x the device's memory are pinned in _ENVELOPE_PAGES."""
    got, pages = {}, {}
    for L in _ENVELOPE_L:
        for draft in (0, 5):
            lm = L + 14 if draft else L
            mgr = _v41_native_manager(16384, 256, draft, max_seq_len=lm)
            try:
                assert mgr.max_seq_len == lm and mgr.max_batch_size == 256, (
                    L,
                    draft,
                    mgr.max_seq_len,
                )
                pages[(draft, L)] = (_native_view(mgr, 2).shape[0], _native_view(mgr, 20).shape[0])
                requests, md = _run_step(mgr, [(0, 100)], 1, 1, uid=970_000, max_num_tokens=16384)
                try:
                    forms = {"step": md}
                    for G in (1, 256):
                        forms[f"graph G={G}"] = md.create_cuda_graph_metadata(G, False, draft)
                    for name, m in forms.items():
                        for r in (1, 2):
                            tbl, lens = m.compress_block_tables[r], m.compressed_kv_lens_cuda[r]
                            assert tuple(tbl.shape) == (256, mgr.max_blocks_per_seq), (
                                L,
                                draft,
                                name,
                                r,
                                tbl.shape,
                            )
                            assert tbl.dtype == torch.int32 and tbl.is_contiguous(), (
                                L,
                                draft,
                                name,
                                r,
                            )
                            assert tuple(lens.shape) == (256,) and lens.dtype == torch.int32, (
                                L,
                                draft,
                                name,
                                r,
                            )
                    got[(draft, L)] = mgr.max_blocks_per_seq
                finally:
                    for r_ in requests:
                        mgr.free_resources(r_)
            finally:
                mgr.shutdown()
                del mgr
                torch.cuda.empty_cache()
    widths = {L: (got[(0, L)] * 128, got[(0, L)] * 64) for L in _ENVELOPE_L}
    print(
        f"\n[envelope widths] cols per (draft, max_seq_len) {got}; out widths (ratio 1, ratio 2) {widths}; "
        f"COMPRESS pages at 0.5 x total {pages}"
    )
    assert got == {(d, L): _ENVELOPE_COLS[L] for L in _ENVELOPE_L for d in (0, 5)}
    assert pages == _ENVELOPE_PAGES


def _envelope_fill(pool, pages, g):
    """Random native rows into `pages` of the [P, bk, 288] view: every code byte, scale bytes 0x00..0x7E."""
    bk = pool.shape[1]
    for i in range(0, pages.numel(), 2048):
        p = pages[i : i + 2048]
        codes = torch.randint(
            0, 256, (p.numel(), bk, CODE_BYTES), generator=g, device=DEV, dtype=torch.int32
        )
        scales = torch.randint(
            0, 0x7F, (p.numel(), bk, W // BLOCK), generator=g, device=DEV, dtype=torch.int32
        )
        pool[p] = torch.cat([codes, scales], -1).to(torch.uint8)


def _envelope_ref(pool, pages):
    """bf16 [n, bk, 512]: the LUT reference of every row of `pages` of the [P, bk, 288] view."""
    bk = pool.shape[1]
    flat = pool.view(-1, ROW)
    out = torch.empty((pages.numel(), bk, W), dtype=torch.bfloat16, device=DEV)
    for i in range(0, pages.numel(), 1024):
        p = pages[i : i + 1024]
        rows = (p[:, None] * bk + torch.arange(bk, device=DEV)[None]).view(-1)
        out[i : i + 1024] = _ref_rows(flat, rows).view(-1, bk, W)
    return out


def _envelope_dirty(buf):
    """Device count of the 8-byte words of the contiguous bf16 `buf` that are not four sentinels."""
    w = buf.view(-1).view(torch.int64)
    n = torch.zeros((), dtype=torch.int64, device=DEV)
    for i in range(0, w.numel(), 1 << 26):
        n += (w[i : i + (1 << 26)] != -1).sum()
    return int(n)


def _envelope_check(out, ci, lens, ref, bk):
    """Entries (b, e < lens[b]) of out [seqs, L_r, 1, 512] must be bitwise ref[ci[b, e // bk], e % bk] (the
    LUT reference of the row the table names); each is then reset to the sentinel, so _envelope_dirty(buffer) == 0
    afterwards means the call wrote nothing else. Returns the entries checked."""
    n_all = 0
    for b, n in enumerate(lens):
        if n:
            exp = _bits(ref[ci[b, : -(-n // bk)]].view(-1, W)[:n])
            got = _bits(out[b, :n, 0])
            if not torch.equal(got, exp):
                raise AssertionError(
                    f"request {b} (length {n}): {int(_n_ne(got, exp))} elements differ"
                )
            out[b, :n].view(torch.int16).fill_(SENTINEL)
            n_all += n
    return n_all


def _envelope_far(L, r):
    """Far-end lengths of a request at max_seq_len L, ratio r: the out's width L_r and L_r - 1, the DSpark
    manager's maximum (L + 14) // r and the graph metadata's (L + 19) // r, L // r and (L - 1) // r."""
    Lr = _ENVELOPE_COLS[L] * 128 // r
    return sorted({Lr, Lr - 1, (L + 14) // r, (L + 19) // r, L // r, (L - 1) // r})


def _envelope_lengths(L, r):
    """The request lengths run at B = 1: every l in 1..L_r where the call stays under 1 ms, otherwise
    every l in 1..256, every l = {0, 1, -1} mod the page size 128 / r, and the far-end lengths."""
    bk, Lr = 128 // r, _ENVELOPE_COLS[L] * 128 // r
    if L in _ENVELOPE_EVERY_LENGTH:
        return list(range(1, Lr + 1))
    ls = set(range(1, 257)) | set(_envelope_far(L, r))
    for k in range(1, Lr // bk + 1):
        ls |= {k * bk - 1, k * bk, k * bk + 1}
    return sorted(v for v in ls if 1 <= v <= Lr)


@pytest.mark.parametrize("L", _ENVELOPE_L)
def test_envelope_fixed_out_every_width(L):
    """Every length and the envelope's widths for the target's call, per max_seq_len L. Operands: the native views
    of a live manager (16384, 256, DSpark 5; owner 2 at ratio 2, owner 20 at ratio 1) built at 1/8 of the
    device's memory so the widest buffer fits beside it; out the ratio's [seqs, L_r, 1, 512] view (L_r = cols *
    128 / r) of one sentinel-filled buffer of seqs * cols * 128 rows (seqs 256, 128 at the 1M width) with a
    guard past it; the table [seqs, cols] int32 over a 16,384-page candidate set (the view's hazard and last
    pages, random pages; every request a different permutation) whose rows hold random native bytes; the
    reference the LUT decode of those bytes. Every call's named entries are bitwise the reference and nothing
    else in the buffer or the guard is written:

    - the full call, every request at L_r (every row of the buffer), one launch of the entry's Triton
      kernel; the FlashInfer twin on the same operands writes the same bits;
    - B = seqs with ragged lengths: the far-end lengths (_envelope_far), the page edges, 0, 1, random (twin too);
    - every B in 1..seqs, the last request at a far-end length and the others short (0..2 pages);
    - every request length l in 1..L_r at B = 1: each call's first l entries bitwise the full call's, entry l
      still the sentinel;
    - no compiled variant is added after the full call.
    """
    from torch.profiler import ProfilerActivity, profile

    cols, seqs = _ENVELOPE_COLS[L], _ENVELOPE_SEQS[L]
    mgr = _v41_native_manager(16384, 256, 5, quota_div=8)
    big = None
    try:
        rows_total = seqs * cols * 128
        big = torch.full((rows_total + 128, W), SENTINEL, dtype=torch.int16, device=DEV).view(
            torch.bfloat16
        )
        one = _gs()
        report = {}
        for owner, r in ((2, 2), (20, 1)):
            bk = 128 // r
            Lr = cols * bk
            out = big[: seqs * Lr].view(seqs, Lr, 1, W)
            pool = _native_view(mgr, owner)
            c4, s4 = _kvq_codes_scales(mgr, owner)
            g = torch.Generator(device=DEV).manual_seed(16101 + L + r)
            rng = random.Random(16101 + L + r)
            cand = _kvq_page_candidates(pool.shape[0], bk, _ENVELOPE_CAND, g)
            _envelope_fill(pool, cand, g)
            ref = _envelope_ref(pool, cand)
            ci = torch.stack(
                [
                    torch.randperm(_ENVELOPE_CAND, device=DEV, generator=g)[:cols]
                    for _ in range(seqs)
                ]
            )
            tbl = cand[ci].to(torch.int32)
            far = _envelope_far(L, r)
            # the full call
            full = torch.full((seqs,), Lr, dtype=torch.int32, device=DEV)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                op(c4, s4, tbl, full, one, out)
                torch.cuda.synchronize()
            kernels = [
                e.name
                for e in prof.events()
                if e.device_type.name == "CUDA" and "nvfp4_kv_to_bf16" in e.name
            ]
            assert len(kernels) == 1, kernels
            n1 = _jit_variants()
            n_full = _envelope_check(out, ci, [Lr] * seqs, ref, bk)
            assert _envelope_dirty(big) == 0, "the full call wrote outside its entries"
            _twin(c4, s4, tbl, full, one, out)
            assert (
                _envelope_check(out, ci, [Lr] * seqs, ref, bk) == n_full
                and _envelope_dirty(big) == 0
            ), "twin differs"
            # ragged lengths at B = seqs
            ragged = (far + [0, 1, bk - 1, bk, bk + 1] + [rng.randint(0, Lr) for _ in range(seqs)])[
                :seqs
            ]
            op(c4, s4, tbl, torch.tensor(ragged, dtype=torch.int32, device=DEV), one, out)
            n_rag = _envelope_check(out, ci, ragged, ref, bk)
            assert _envelope_dirty(big) == 0, "the ragged call wrote outside its entries"
            _twin(c4, s4, tbl, torch.tensor(ragged, dtype=torch.int32, device=DEV), one, out)
            assert (
                _envelope_check(out, ci, ragged, ref, bk) == n_rag and _envelope_dirty(big) == 0
            ), "twin differs (ragged)"
            # every B, the last request at the far end
            n_b = 0
            for B in range(1, seqs + 1):
                lens_b = [rng.randint(0, 2 * bk) for _ in range(B - 1)] + [far[B % len(far)]]
                op(
                    c4,
                    s4,
                    tbl[:B],
                    torch.tensor(lens_b, dtype=torch.int32, device=DEV),
                    one,
                    out[:B],
                )
                n_b += _envelope_check(out, ci, lens_b, ref, bk)
                d = _envelope_dirty(big)
                assert d == 0, f"B={B}: {d} words written outside the named entries"
            # request lengths at B = 1; the call at l = L_r timed first (median of 5 CUDA-event timings)
            ls = _envelope_lengths(L, r)
            e64 = _bits(ref[ci[0]].view(-1, W)[:Lr]).view(torch.int64)
            o64 = out[0, :, 0].view(torch.int16).view(torch.int64)
            neq = torch.empty(e64.shape, dtype=torch.bool, device=DEV)
            mism = torch.zeros((), dtype=torch.int64, device=DEV)
            past = torch.zeros((), dtype=torch.int64, device=DEV)
            lens1 = torch.full((1,), Lr, dtype=torch.int32, device=DEV)
            times = []
            for _ in range(5):
                ev0, ev1 = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                ev0.record()
                op(c4, s4, tbl[:1], lens1, one, out[:1])
                ev1.record()
                torch.cuda.synchronize()
                times.append(ev0.elapsed_time(ev1))
            assert torch.equal(o64, e64), "the B = 1 full-length call differs from the reference"
            assert L in _ENVELOPE_EVERY_LENGTH or sorted(times)[2] > 1.0, (
                f"a sampled width's call takes {times} ms"
            )
            o64.fill_(-1)
            for length in ls:
                lens1.fill_(length)
                op(c4, s4, tbl[:1], lens1, one, out[:1])
                torch.ne(o64[:length], e64[:length], out=neq[:length])
                mism += neq[:length].any()
                if length < Lr:
                    past += (o64[length] != -1).any()
                o64[:length].fill_(-1)
            assert int(mism) == 0 and int(past) == 0, (int(mism), int(past))
            assert _envelope_dirty(big) == 0
            n2 = _jit_variants()
            assert n2 == n1, f"a batch or a length compiled a new variant ({n1} -> {n2})"
            report[f"ratio {r}"] = dict(
                L_r=Lr,
                pages=pool.shape[0],
                full=n_full,
                ragged=n_rag,
                every_B=n_b,
                call_ms_B1_full=round(sorted(times)[2], 3),
                lengths=f"{len(ls)} ({'every l' if L in _ENVELOPE_EVERY_LENGTH else 'sampled'})",
                far=far,
            )
            del ref, e64, neq
        print(
            f"\n[envelope width, max_seq_len {L}: cols {cols}, buffer [{rows_total}, 512] ({seqs} requests)] "
            f"{report}"
        )
    finally:
        big = None
        mgr.shutdown()
        del mgr
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("L", _ENVELOPE_L)
@pytest.mark.parametrize("ratio", ["ratio2", "ratio1"])
def test_envelope_cuda_graph_every_gset_batch_every_width(ratio, L):
    """The target's call under CUDA-graph capture at every graph batch G of the envelope (_ENVELOPE_GSET: 1..32 and
    40..128 step 8, 192, 256; up to 128 at the 1M width) x next_n {1, 6}, at the table width the graph metadata
    uses at max_seq_len L. Operands as test_envelope_fixed_out_every_width: the ratio's owner view of the live
    (16384, 256, 5) manager at 1/8 of the device's memory, out the [seqs, L_r, 1, 512] view of one
    sentinel-filled seqs * cols * 128-row buffer (a guard past it), a persistent [seqs, cols] table and [seqs]
    lengths. First one eager call at the buffer's largest shape (seqs requests of L_r entries): the only
    warm-up. Then per (G, next_n): capture on the [:G] views, and 2 replays after the table rows, the lengths
    and the bytes of pages request 0 names all change in place. Lengths are kv_len // r: request 0 and
    request G - 1 at the far end (kv_len L for next_n 1, L + 14 for next_n 6), request 1 at next_n, the rest
    random up to kv_len 4096. Every replay writes exactly the named entries, bitwise the LUT reference, and
    nothing else in the buffer (requests G.. and the guard included); the same call run eagerly after it
    gives the same bits.
    """
    cols, seqs = _ENVELOPE_COLS[L], _ENVELOPE_SEQS[L]
    owner, r = (2, 2) if ratio == "ratio2" else (20, 1)
    bk = 128 // r
    Lr = cols * bk
    mgr = _v41_native_manager(16384, 256, 5, quota_div=8)
    big = None
    try:
        pool = _native_view(mgr, owner)
        c4, s4 = _kvq_codes_scales(mgr, owner)
        rows_total = seqs * cols * 128
        big = torch.full((rows_total + 128, W), SENTINEL, dtype=torch.int16, device=DEV).view(
            torch.bfloat16
        )
        out = big[: seqs * Lr].view(seqs, Lr, 1, W)
        one = _gs()
        g = torch.Generator(device=DEV).manual_seed(16201 + L + r)
        rng = random.Random(16201 + L + r)
        cand = _kvq_page_candidates(pool.shape[0], bk, _ENVELOPE_CAND, g)
        _envelope_fill(pool, cand, g)
        ref = _envelope_ref(pool, cand)
        ci = torch.zeros((seqs, cols), dtype=torch.int64, device=DEV)
        tbl = torch.full((seqs, cols), -1, dtype=torch.int32, device=DEV)
        lens = torch.zeros((seqs,), dtype=torch.int32, device=DEV)

        def fresh(entries):
            """New table rows for the first len(entries) requests, new bytes on up to 16 pages request 0
            names (its first and last 8), the lengths in place."""
            B = len(entries)
            ci[:B] = torch.stack(
                [torch.randperm(_ENVELOPE_CAND, device=DEV, generator=g)[:cols] for _ in range(B)]
            )
            tbl[:B] = cand[ci[:B]].to(torch.int32)
            n0 = -(-entries[0] // bk)
            if n0:
                touch = torch.unique(torch.cat([ci[0, : min(n0, 8)], ci[0, max(n0 - 8, 0) : n0]]))
                _envelope_fill(pool, cand[touch], g)
                ref[touch] = _envelope_ref(pool, cand[touch])
            lens[:B] = torch.tensor(entries, dtype=torch.int32, device=DEV)
            return entries

        entries = fresh([Lr] * seqs)
        op(c4, s4, tbl, lens, one, out)
        n_warm = _envelope_check(out, ci, entries, ref, bk)
        assert _envelope_dirty(big) == 0
        n1 = _jit_variants()
        graphs, named = 0, 0
        for G in (v for v in _ENVELOPE_GSET if v <= seqs):
            for next_n in (1, 6):
                end = L + 14 if next_n == 6 else L

                def kv_lens():
                    kv = [rng.randint(next_n, min(end, 4096)) for _ in range(G)]
                    kv[0] = end
                    if G > 1:
                        kv[1], kv[-1] = next_n, end
                    return [v // r for v in kv]

                fresh(kv_lens())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    op(c4, s4, tbl[:G], lens[:G], one, out[:G])
                graphs += 1
                assert _envelope_dirty(big) == 0, "capture wrote into the buffer"
                for rep in (1, 2):
                    entries = fresh(kv_lens())
                    graph.replay()
                    torch.cuda.synchronize()
                    named += _envelope_check(out, ci, entries, ref, bk)
                    d = _envelope_dirty(big)
                    assert d == 0, (
                        f"G={G} next_n={next_n} replay {rep}: {d} words written outside the named entries"
                    )
                    op(c4, s4, tbl[:G], lens[:G], one, out[:G])
                    _envelope_check(out, ci, entries, ref, bk)
                    assert _envelope_dirty(big) == 0, (
                        f"G={G} next_n={next_n} replay {rep}: eager differs"
                    )
                del graph
        print(
            f"\n[envelope graphs {ratio}, max_seq_len {L}: owner {owner} view {tuple(pool.shape)}, out "
            f"[{seqs}, {Lr}, 1, 512]] eager warm-up {n_warm} entries; {graphs} graphs (G in the envelope's set up "
            f"to {seqs} x next_n 1/6), {named} replayed entries: every replay == eager == the LUT reference, "
            f"nothing else written"
        )
        assert graphs == 2 * len([v for v in _ENVELOPE_GSET if v <= seqs])
        assert _jit_variants() == n1, "a capture compiled a new variant"
    finally:
        big = None
        mgr.shutdown()
        del mgr
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# KV tokens one step of the 256-request tests may hold in total. The native manager at (16384, 256, DSpark
# 5, max_seq_len 32,782) held 22 requests of 32,768 tokens and refused the 23rd (about 720K tokens), so a
# step of 256 requests cannot put every request at the far end; steps keep their total under this bound.
_ENVELOPE_STEP_TOKENS = 560_000


def _envelope_edge_steps(max_t, seqs, L):
    """Edge steps of a (max_num_tokens, max_batch_size) configuration at max_seq_len L with DSpark, inside
    _ENVELOPE_STEP_TOKENS: a full-T context from position 0 and one ending at L, one token at L - 1, one DSpark
    verify at kv_len L + 14; full batches of seqs requests with 8 at the far end and the rest at positions
    below 1024 -- contexts (8 ending at L), decodes (8 at L - 1), verifies (8 at kv_len L + 14) -- and a
    context filling T beside seqs - 1 verifies (7 at kv_len L + 14)."""
    n, lm = max_t // seqs, L + 14
    rest = seqs - 8
    steps = [
        ([(0, max_t)], 1, 1),
        ([(L - max_t, max_t)], 1, 1),
        ([(L - 1, 1)], 1, 1),
        ([(lm - 6, 6)], 0, 6),
        ([(L - n, n)] * 8 + [(0, n)] * rest, seqs, 1),
        ([(L - 1, 1)] * 8 + [(127 + (61 * i) % 1024, 1) for i in range(rest)], 0, 1),
        ([(lm - 6, 6)] * 8 + [(100 + (37 * i) % 1024, 6) for i in range(rest)], 0, 6),
    ]
    c = max_t - 6 * (seqs - 1)
    steps.append(
        ([(L - c, c)] + [(lm - 6, 6)] * 7 + [(100 + (37 * i) % 1024, 6) for i in range(rest)], 1, 6)
    )
    for reqs, _, _ in steps:
        assert sum(a + b for a, b in reqs) <= _ENVELOPE_STEP_TOKENS, reqs[:3]
    return steps


def _envelope_fit(step, rng):
    """A random step with its total KV tokens capped at _ENVELOPE_STEP_TOKENS: its requests, in a random order,
    keep their cached prefix while the rest can still fit; the others get a random prefix below 1024. The
    request order (contexts first) and every request's new tokens are unchanged."""
    reqs, nctx, gen_len = step
    if sum(a + n for a, n in reqs) <= _ENVELOPE_STEP_TOKENS:
        return step
    out, used = list(reqs), 0
    reserve = sum(n + 1024 for _, n in reqs)
    order = list(range(len(reqs)))
    rng.shuffle(order)
    for i in order:
        a, n = reqs[i]
        reserve -= n + 1024
        if used + a + n + reserve > _ENVELOPE_STEP_TOKENS:
            a = rng.randint(0, 1023)
        out[i] = (a, n)
        used += a + n
    assert used <= _ENVELOPE_STEP_TOKENS
    return out, nctx, gen_len


def test_envelope_compressed_read_renumbered_over_random_steps_256_requests():
    """At the envelope's step axes, in the target's exact call, on the live native manager of the
    engine configuration max_num_tokens 16384, max_batch_size 256, DSpark 5, max_seq_len 32768 + 14 (cols
    260: out widths 33,280 / 16,640). Steps: _envelope_edge_steps (a 16,384-token context, 256 requests of a full
    batch, verifies at kv_len 32,782) and 200 seeded random partitions of T <= 16,384 into k <= 256 requests
    (contexts first with random cached prefixes, then decode next_n 1 at kv_len <= 32,768 or DSpark verify
    next_n 6 at <= 32,782), each step's KV tokens capped at _ENVELOPE_STEP_TOKENS (_envelope_fit).
    Per step and per KV source in forward order (2, 8, 14 at ratio 2, 20 at ratio 1),
    exactly as test_kvq_compressed_read_renumbered_over_random_steps: the writer fills every compressed entry
    at its row of md.compress_block_tables[r]; the call reads md.compress_block_tables[r][:B] and
    md.compressed_kv_lens_cuda[r][:B] into the ratio's view of ONE persistent zero-filled [256 * 260 * 128,
    512] buffer; every entry (b, e < length) equals the writer's bf16 container and nothing else changes; the
    renumbered top-k is bitwise the bf16 path; the three controls differ wherever they apply.

    The same steps and writes also drive the per-query form (the target's _native_topk, used when
    max_num_tokens * 512 rows is the smaller buffer): op(codes1, scales1, slots, cols,
    one, out) with (codes1, scales1) the owner's views re-viewed with page size 1 ([pages * 128 / r, 1, 1,
    256] / [.., 32]), slots the [T, 512] int32 pool slots of each query's top-k (up to 512 random visible
    entries, -1 after them, as the index op packs them), cols 512 per query, and out the [T, 512, 1, 512]
    view of ONE persistent zero-filled [16384, 512, 1, 512] buffer. After each call every valid (q, j) row
    q * 512 + j is bitwise the bf16-container row of that slot, every -1 entry's row and every row past query
    T is exactly as before the call, and two controls (the row off by one, the buffer before the call)
    differ wherever a row is named."""
    L, max_t, seqs = 32768, 16384, 256
    rng = random.Random(16301)
    edges = _envelope_edge_steps(max_t, seqs, L)
    steps = edges + [
        _envelope_fit(_kvq_random_step(rng, max_tokens=max_t, max_requests=seqs, seq_len=L), rng)
        for _ in range(_NUM_RANDOM_STEPS)
    ]
    mgr = _v41_native_manager(max_t, seqs, 5, max_seq_len=L + 14)
    try:
        stats, ctrl, shape = _kvq_renumbered_steps(
            mgr, steps, max_t, seqs, seed=16_302_000, cols=_ENVELOPE_COLS[L], per_query=True
        )
        print(
            f"\n[envelope renumbered read, (16384, 256, 5) at max_seq_len {L + 14}, COMPRESS "
            f"{tuple(_native_view(mgr, 2).shape)} / {tuple(_native_view(mgr, 20).shape)}, buffer {shape}] "
            f"{len(edges)} edge + {_NUM_RANDOM_STEPS} random steps: {stats}; controls [detected, applicable, "
            f"elements]: {ctrl}"
        )
        assert shape == (seqs * _ENVELOPE_COLS[L] * 128, W)
        assert stats["steps"] == len(steps) and stats["max_t"] == max_t and stats["max_k"] == seqs
        assert stats["max_len"] == {2: (L + 14) // 2, 1: L + 14}
        assert (
            stats["pq_calls"] == stats["calls"]
            and stats["pq_max_t"] == max_t
            and stats["pq_minus1"] > 0
        )
        for name, (hit, applicable, _) in ctrl.items():
            assert applicable > 0 and hit == applicable, f"control '{name}': {hit} of {applicable}"
    finally:
        mgr.shutdown()
        del mgr
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def test_kvq_compressed_read_renumbered_over_random_steps_gate_config():
    """In the target's two calls on the live accuracy-gate manager (8192, 32, DSpark 5; cols 132):
    _mbs_edge_steps(8192, 32) (a full-T context, a DSpark verify at kv_len 16,398, full batches of 32
    requests) and 200 seeded random partitions of T <= 8192 into k <= 32 requests (contexts with cached
    prefixes, decode next_n 1, DSpark verify next_n 6). Per step and KV source in forward order, as
    test_envelope_compressed_read_renumbered_over_random_steps_256_requests: the per-request read into ONE
    [32 * 132 * 128, 512] buffer and the per-query read into ONE [8192, 512, 1, 512] buffer, both bitwise the
    writer's bf16 container, nothing else written, and every control fires where it applies."""
    max_t, seqs = 8192, 32
    rng = random.Random(17301)
    edges = _mbs_edge_steps(max_t, seqs)
    steps = edges + [
        _kvq_random_step(rng, max_tokens=max_t, max_requests=seqs) for _ in range(_NUM_RANDOM_STEPS)
    ]
    mgr = _v41_native_manager(max_t, seqs, 5)
    try:
        stats, ctrl, shape = _kvq_renumbered_steps(
            mgr, steps, max_t, seqs, seed=17_302_000, per_query=True
        )
        print(
            f"\n[renumbered read, gate manager (8192, 32, 5), COMPRESS {tuple(_native_view(mgr, 2).shape)} / "
            f"{tuple(_native_view(mgr, 20).shape)}, buffer {shape}] {len(edges)} edge + {_NUM_RANDOM_STEPS} random "
            f"steps: {stats}; controls [detected, applicable, elements]: {ctrl}"
        )
        assert shape == (seqs * _KVQ_COLS * 128, W)
        assert stats["steps"] == len(steps) and stats["max_t"] == max_t and stats["max_k"] == seqs
        assert stats["max_len"] == _KVQ_DSPARK_ENTRIES
        assert (
            stats["pq_calls"] == stats["calls"]
            and stats["pq_max_t"] == max_t
            and stats["pq_minus1"] > 0
        )
        for name, (hit, applicable, _) in ctrl.items():
            assert applicable > 0 and hit == applicable, f"control '{name}': {hit} of {applicable}"
    finally:
        mgr.shutdown()
        del mgr
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
