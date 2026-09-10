# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mla_rope_append_paged_kv_assign_q catalog entry.

The op reads paged-KV-cache addressing tensors and cumulative-length
tensors that the runtime normally derives from a KVCacheManager and a
TrtllmAttentionMetadata prepared with enable_context_mla_with_cached_kv=True.
The test builds that state for real, pre-writes known cached-prefix rows
into the paged pool, then checks the three kernel effects against a torch
fp32 reference: GPT-J RoPE of each new context token's q_pe written back
into q at the token's absolute position, the same RoPE of k_pe written back
into latent_cache, and [compressed_kv | rope(k_pe)] appended into the paged
latent cache at that position. Cached-prefix rows, q's nope slices, and
latent_cache's compressed_kv slice must be bitwise untouched.

Two surfaces:

1. Matching-dtype latent pool (`quant_mode=0`, `kv_scale_orig_quant=None`),
   H = 16, in bf16 (production MLA dtype) and fp16, at both pool page
   sizes: 64, and 32 (the engine default), where the append walks twice as
   many pages and the cached prefixes place the first new token at every
   32-slot alignment.

2. fp8-e4m3 latent pool (`quant_mode=128`) at the DeepSeek-R1-0528 cell —
   H = 128, tokens_per_block = 32, C/R/nope = 512/64/128, beam_width = 1,
   single-layer pool, bf16 activations. Here q and latent_cache are still
   roped in place in bf16 (neither is quantized), and only the appended
   pool row lands as e4m3, scaled by kv_scale_orig_quant. Swept over that
   scaling factor (omitted = the production call, and explicit
   1.0 / 1/1.5 / 0.5 / 0.25 / 2.0), over fresh-prefill and cached-prefix
   context calls in the same batch, and over a mixed batch whose trailing
   generation sequences must be ignored.
"""

from typing import List, NamedTuple, Optional, Sequence, Union

import torch

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.staircase.catalog.attention.mla_rope_append_paged_kv_assign_q import (
    mla_rope_append_paged_kv_assign_q,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

assert torch.cuda.is_available(), "mla_rope_append_paged_kv_assign_q requires a CUDA device"

# DeepSeek-V3 MLA head geometry (num_heads reduced to a TP-slice-like 16).
NUM_HEADS = 16
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
LATENT_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
# Pool page size. 32 is the engine default (KvCacheConfig.tokens_per_block),
# 64 the value a tuned MLA target opts into; both are covered below.
TOKENS_PER_BLOCK = 64
PAGE32 = 32
MAX_SEQ_LEN = 1024

# ─── fp8-e4m3 latent pool surface ─────────────────────────────────────
# DeepSeek-R1-0528 cell: full 128-head MLA over an fp8 latent pool.
NUM_HEADS_R1 = 128
QUANT_MODE_FP8_KV_CACHE = 128  # QuantMode.FP8_KV_CACHE
QUANT_MODE_INT8_KV_CACHE = 64  # QuantMode.INT8_KV_CACHE — rejected, see below

# One e4m3 ulp. e4m3 keeps 3 mantissa bits, so 2**-3 relative is one ulp — the
# gate for the appended row's k_pe half, the only quantized output that passes
# through the in-kernel RoPE, where the kernel and a torch fp32 reference can
# evaluate x*cos -+ y*sin in different orders. Paired with a bit-exact-fraction
# floor, because in practice the coarse e4m3 rounding absorbs that difference
# completely: every appended k_pe byte measured on sm_100 under quant_mode 128
# (all scales, all cases below) was bit-exact, so neither half of the gate has
# ever been approached. That both halves still discriminate is measured, not
# assumed — test_fp8_kv_explicit_unit_scale runs four wrong mirrors of the same
# bytes and asserts each blows the gate it is able to blow.
E4M3_ULP_RTOL = 2**-3
E4M3_MAX_INEXACT_FRACTION = 1e-3

# The roped q must come back as an ordinary bf16 rotation, NOT quantized: this
# op leaves q to the context FMHA, which does its own e4m3 quantization. A bf16
# value drawn from a continuous distribution survives an e4m3 round trip only
# when its mantissa already fits in 3 bits, which happens for ~6% of elements;
# a quantized q would sit at 100%. The gate is 25% — 4x the measured 6.2%, and
# 4x below a quantized tensor — and every case also compares its own untouched
# nope slice as an in-run baseline.
MAX_E4M3_ROUNDTRIP_FRACTION = 0.25

_TRTLLM_TO_TORCH_DTYPE = {
    DataType.BF16: torch.bfloat16,
    DataType.HALF: torch.float16,
}


class _MlaCtxEnv:
    """Real op state: MLA (kv_factor=1) paged KV cache manager + duplicated-
    layout RoPE table + context-MLA-with-cached-KV metadata.

    With fp8_pool the manager allocates an e4m3 latent pool (one byte per
    element) while the activations (`q`, `latent_cache`) stay `cache_dtype`,
    and `orig_quant` becomes the op's write-side KV scaling factor.
    """

    def __init__(
        self,
        cache_dtype: DataType = DataType.BF16,
        num_layers: int = 1,
        max_batch_size: int = 8,
        tokens_per_block: int = TOKENS_PER_BLOCK,
        num_heads: int = NUM_HEADS,
        fp8_pool: bool = False,
        orig_quant: Optional[float] = None,
    ) -> None:
        self.torch_dtype = _TRTLLM_TO_TORCH_DTYPE[cache_dtype]
        self.max_batch_size = max_batch_size
        self.tokens_per_block = tokens_per_block
        self.num_heads = num_heads
        self.fp8_pool = fp8_pool
        self.pool_dtype = torch.float8_e4m3fn if fp8_pool else self.torch_dtype
        self.quant_mode = QUANT_MODE_FP8_KV_CACHE if fp8_pool else 0
        self.kv_cache_manager = KVCacheManager(
            KvCacheConfig(max_tokens=131072, enable_block_reuse=False),
            CacheType.SELFKONLY,  # MLA latent cache: kv_factor=1, one kv head
            num_layers=num_layers,
            num_kv_heads=1,
            head_dim=LATENT_SIZE,
            tokens_per_block=tokens_per_block,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.FP8 if fp8_pool else cache_dtype,
        )
        # The pool must really be paged at the size and element type the case
        # claims: the op sizes its page slabs from tokens_per_block and
        # quant_mode, not from anything the manager tells it.
        assert self.kv_cache_manager.tokens_per_block == tokens_per_block
        pool = self.kv_cache_manager.get_buffers(0)
        assert pool is not None and pool.dtype == self.pool_dtype
        # KV scaling factor. None is what the engine's own call site passes
        # (TrtllmAttention.mla_rope_append_paged_kv_assign_q hard-codes it),
        # and the op then behaves as if it were 1.0.
        self.kv_scale_orig_quant = (
            None
            if orig_quant is None
            else torch.full((1,), orig_quant, dtype=torch.float32, device="cuda")
        )
        # The exact fp32 number the op sees, as a 0-dim tensor, so the mirror
        # cannot disagree with the kernel in the last bit.
        self.write_scale = torch.tensor(
            1.0 if orig_quant is None else orig_quant, dtype=torch.float32
        ).cuda()
        # Duplicated-layout fp32 (cos, sin) table, as the MLA backend builds
        # it (RopeParams.from_config sets duplicate_data=True for MLA models).
        rope = RopeParams(
            dim=QK_ROPE_HEAD_DIM,
            theta=10000.0,
            max_positions=MAX_SEQ_LEN,
            duplicate_data=True,
        )
        _, self.rotary_cos_sin = rope.create_rope_const_params()

    def prepare_metadata(
        self,
        request_ids: List[int],
        seq_lens: List[int],
        num_contexts: int,
        cached_lens: List[int],
    ) -> TrtllmAttentionMetadata:
        metadata = TrtllmAttentionMetadata(
            max_num_requests=self.max_batch_size,
            max_num_tokens=8192,
            kv_cache_manager=self.kv_cache_manager,
            enable_context_mla_with_cached_kv=True,
        )
        metadata.seq_lens = torch.tensor(seq_lens, dtype=torch.int)
        metadata.num_contexts = num_contexts
        metadata.request_ids = request_ids
        metadata.prompt_lens = [c + s for c, s in zip(cached_lens, seq_lens)]
        metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=cached_lens,
        )
        metadata.prepare()
        return metadata

    def rope_fp32(self, x: torch.Tensor, positions: Union[int, Sequence[int]]) -> torch.Tensor:
        """GPT-J interleaved rotation of the last dim in fp32, unrounded.
        x is [n, ..., R] with positions of length n (or a single position for
        x of shape [..., R])."""
        half = QK_ROPE_HEAD_DIM // 2
        table = self.rotary_cos_sin.view(-1, QK_ROPE_HEAD_DIM, 2)
        if isinstance(positions, int):
            cos = table[positions, :half, 0]
            sin = table[positions, :half, 1]
        else:
            pos = torch.tensor(list(positions), dtype=torch.long, device="cuda")
            # [n, half] broadcast over x's middle dims.
            shape = [len(positions)] + [1] * (x.dim() - 2) + [half]
            cos = table[pos, :half, 0].view(shape)
            sin = table[pos, :half, 1].view(shape)
        pairs = x.float().reshape(*x.shape[:-1], half, 2)
        out = torch.empty_like(pairs)
        out[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        out[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        return out.reshape(x.shape)

    def rope_ref(self, x: torch.Tensor, positions: Union[int, Sequence[int]]) -> torch.Tensor:
        """rope_fp32 with one rounding back to x's dtype — what the kernel
        writes into q and latent_cache."""
        return self.rope_fp32(x, positions).to(x.dtype)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """e4m3(x * kv_scale_orig_quant) — the op's write-side quantization of
        the appended latent row."""
        return (x.float() * self.write_scale).to(torch.float8_e4m3fn)

    def pool_tensor(self, layer_idx: int) -> torch.Tensor:
        pool = self.kv_cache_manager.get_buffers(layer_idx)
        assert pool is not None
        return pool

    def blocks(self, request_id: int) -> List[int]:
        return list(self.kv_cache_manager.get_batch_cache_indices([request_id])[0])

    def slot(self, request_id: int, position: int) -> tuple[int, int]:
        tpb = self.tokens_per_block
        return self.blocks(request_id)[position // tpb], position % tpb

    def fill_rows(self, layer_idx: int, request_id: int, length: int) -> torch.Tensor:
        """Write known random latent rows at positions [0, length) of one
        sequence's pages; return the written rows [length, LATENT_SIZE]."""
        draw_dtype = torch.float32 if self.fp8_pool else self.torch_dtype
        rows = torch.randn(length, LATENT_SIZE, dtype=draw_dtype, device="cuda").to(self.pool_dtype)
        pool = self.pool_tensor(layer_idx)  # [pages,1,tpb,1,head]
        tpb = self.tokens_per_block
        blocks = self.blocks(request_id)
        for start in range(0, length, tpb):
            page = blocks[start // tpb]
            n = min(tpb, length - start)
            pool[page, 0, :n, 0].copy_(rows[start : start + n])
        return rows

    def read_rows(self, layer_idx: int, request_id: int, start: int, end: int) -> torch.Tensor:
        """Read latent rows at positions [start, end) of one sequence back
        from the paged pool as a dense [end-start, LATENT_SIZE] tensor."""
        pool = self.pool_tensor(layer_idx)
        tpb = self.tokens_per_block
        blocks = self.blocks(request_id)
        out = torch.empty(end - start, LATENT_SIZE, dtype=self.pool_dtype, device="cuda")
        for pos in range(start, end):
            page = blocks[pos // tpb]
            out[pos - start] = pool[page, 0, pos % tpb, 0]
        return out

    def shutdown(self) -> None:
        self.kv_cache_manager.shutdown()


def _assert_bytes_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """Bit-exact comparison of two tensors, signed zeros included."""
    torch.testing.assert_close(
        got.reshape(-1).view(torch.uint8),
        want.reshape(-1).view(torch.uint8),
        rtol=0.0,
        atol=0.0,
        msg=lambda m: f"{what} not bit-exact\n{m}",
    )


def _inexact_bytes(got: torch.Tensor, want: torch.Tensor) -> int:
    return int((got.reshape(-1).view(torch.uint8) != want.reshape(-1).view(torch.uint8)).sum())


def _assert_e4m3_roped(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """One-e4m3-ulp gate plus a bit-exact-fraction floor, for the part of the
    fp8 output that passes through the in-kernel RoPE."""
    torch.testing.assert_close(got.float(), want.float(), rtol=E4M3_ULP_RTOL, atol=0.0)
    inexact = _inexact_bytes(got, want)
    allowed = max(1, int(E4M3_MAX_INEXACT_FRACTION * got.numel()))
    assert inexact <= allowed, (
        f"{what} not bit-exact enough: {inexact}/{got.numel()} bytes differ (allowed {allowed})"
    )


def _fails_e4m3_tolerance(got: torch.Tensor, mirror: torch.Tensor) -> bool:
    """Whether the tolerance half of _assert_e4m3_roped rejects this mirror."""
    try:
        torch.testing.assert_close(got.float(), mirror.float(), rtol=E4M3_ULP_RTOL, atol=0.0)
    except AssertionError:
        return True
    return False


def _e4m3_roundtrip_fraction(x: torch.Tensor) -> float:
    """Fraction of elements of a bf16/fp16 tensor that survive an e4m3 round
    trip unchanged — ~6% for continuous data, 100% for a quantized tensor."""
    rt = x.float().to(torch.float8_e4m3fn).to(x.dtype)
    return float((x.reshape(-1) == rt.reshape(-1)).float().mean())


class _Run(NamedTuple):
    """What one op call was driven with and produced."""

    q: torch.Tensor  # post-call q, [T, H*(N+R)]
    latent_cache: torch.Tensor  # post-call latent_cache, [T, C+R]
    latent_orig: torch.Tensor  # pre-call latent_cache
    ref_k_pe: torch.Tensor  # torch reference for rope(k_pe), activation dtype
    ref_k_pe_fp32: torch.Tensor  # the same reference before rounding
    rows: torch.Tensor  # appended pool rows in row order, [T, C+R]
    positions: List[int]  # absolute position of every row


def _run_and_check(
    env: _MlaCtxEnv,
    request_ids: List[int],
    seq_lens: List[int],
    num_contexts: int,
    cached_lens: List[int],
    layer_idx: int = 0,
) -> _Run:
    """Pre-fill every sequence's cached prefix, run the op over the context
    sequences' new tokens, and verify every kernel effect."""
    kv_lens = [c + s for c, s in zip(cached_lens, seq_lens)]
    metadata = env.prepare_metadata(request_ids, seq_lens, num_contexts, cached_lens)
    # Known cached-prefix rows (all sequences) to check the op leaves them be.
    prefix_rows = [env.fill_rows(layer_idx, rid, c) for rid, c in zip(request_ids, cached_lens)]

    num_heads = env.num_heads
    ctx_new = seq_lens[:num_contexts]
    num_tokens = sum(ctx_new)
    assert int(metadata.num_ctx_tokens) == num_tokens
    assert int(metadata.max_ctx_seq_len) == max(ctx_new)
    # Absolute position of every new context token, in q/latent row order.
    positions = [cached_lens[s] + i for s in range(num_contexts) for i in range(ctx_new[s])]

    q = torch.randn(num_tokens, num_heads * QK_HEAD_DIM, dtype=env.torch_dtype, device="cuda")
    # The op rejects other ranks (thop checks latent_cache.dim() == 2).
    latent_cache = torch.randn(num_tokens, LATENT_SIZE, dtype=env.torch_dtype, device="cuda")
    q_orig = q.clone()
    latent_orig = latent_cache.clone()

    block_offsets = metadata.kv_cache_block_offsets
    pool_pointers = env.kv_cache_manager.kv_cache_pool_pointers
    pool_mapping = env.kv_cache_manager.kv_cache_pool_mapping
    assert block_offsets is not None
    assert pool_pointers is not None and pool_mapping is not None
    pool = env.pool_tensor(layer_idx)
    pool_before = pool.clone()

    mla_rope_append_paged_kv_assign_q(
        q,
        latent_cache,
        num_contexts,
        metadata.ctx_cached_token_indptr,
        metadata.ctx_kv_indptr,
        int(metadata.max_ctx_seq_len),
        env.rotary_cos_sin,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        KV_LORA_RANK,
        block_offsets,
        pool_pointers,
        pool_mapping,
        env.kv_scale_orig_quant,  # None outside the fp8-KV-cache path
        0,  # residual_dim: 0 or rope_size; non-zero needs an FP4 KV pool
        layer_idx,
        env.tokens_per_block,
        MAX_SEQ_LEN,  # attention_window_size
        1,  # beam_width
        env.quant_mode,
    )
    torch.cuda.synchronize()

    # 1. q: per head, the rope slice is rotated at the token's absolute
    # position; the nope slice is bitwise untouched. This holds on the fp8
    # path too — q stays an unquantized activation tensor there.
    q_heads = q.view(num_tokens, num_heads, QK_HEAD_DIM)
    q_orig_heads = q_orig.view(num_tokens, num_heads, QK_HEAD_DIM)
    ref_q_pe = env.rope_ref(q_orig_heads[..., QK_NOPE_HEAD_DIM:], positions)
    torch.testing.assert_close(q_heads[..., QK_NOPE_HEAD_DIM:], ref_q_pe)
    torch.testing.assert_close(
        q_heads[..., :QK_NOPE_HEAD_DIM],
        q_orig_heads[..., :QK_NOPE_HEAD_DIM],
        rtol=0.0,
        atol=0.0,  # caller-owned region: must be bitwise untouched
    )

    # 2. latent_cache: k_pe rotated in place; compressed_kv bitwise untouched.
    ref_k_pe_fp32 = env.rope_fp32(latent_orig[:, KV_LORA_RANK:], positions)
    ref_k_pe = ref_k_pe_fp32.to(env.torch_dtype)
    torch.testing.assert_close(latent_cache[:, KV_LORA_RANK:], ref_k_pe)
    torch.testing.assert_close(
        latent_cache[:, :KV_LORA_RANK],
        latent_orig[:, :KV_LORA_RANK],
        rtol=0.0,
        atol=0.0,  # read-only region: must be bitwise untouched
    )

    if env.fp8_pool:
        # Neither in-place rotation is quantized, so the pair of this op and
        # the context FMHA (which quantizes q/k/v itself) is not a double
        # quantization. Compared against the untouched nope slice of the same
        # tensor, which is the in-run baseline for un-quantized bf16 data.
        for name, values in (
            ("roped q", q_heads[..., QK_NOPE_HEAD_DIM:]),
            ("roped k_pe", latent_cache[:, KV_LORA_RANK:]),
        ):
            frac = _e4m3_roundtrip_fraction(values)
            assert frac < MAX_E4M3_ROUNDTRIP_FRACTION, (
                f"{name} looks e4m3-quantized: {frac:.4f} of elements survive "
                "an e4m3 round trip unchanged"
            )
        baseline = _e4m3_roundtrip_fraction(q_orig_heads[..., :QK_NOPE_HEAD_DIM])
        assert baseline < MAX_E4M3_ROUNDTRIP_FRACTION, (
            f"un-quantized baseline is already {baseline:.4f} — the "
            "round-trip check cannot discriminate at this input distribution"
        )

    # 3. Paged cache: rows [cached_s, kv_s) hold [compressed_kv | rope(k_pe)],
    # quantized by kv_scale_orig_quant on the fp8 path; the pre-filled cached
    # prefix [0, cached_s) is bitwise untouched.
    appended = []
    row_start = 0
    for s in range(num_contexts):
        rid = request_ids[s]
        rows = env.read_rows(layer_idx, rid, cached_lens[s], kv_lens[s])
        appended.append(rows)
        row_end = row_start + ctx_new[s]
        want_ckv = latent_orig[row_start:row_end, :KV_LORA_RANK]
        want_k_pe = ref_k_pe[row_start:row_end]
        if env.fp8_pool:
            _assert_bytes_equal(
                rows[:, :KV_LORA_RANK],
                env.quantize(want_ckv),
                f"appended compressed_kv (request {rid})",
            )
            _assert_e4m3_roped(
                rows[:, KV_LORA_RANK:],
                env.quantize(want_k_pe),
                f"appended k_pe (request {rid})",
            )
        else:
            torch.testing.assert_close(
                rows[:, :KV_LORA_RANK],
                want_ckv,
                rtol=0.0,
                atol=0.0,  # dtype-preserving copy: must be bitwise equal
            )
            torch.testing.assert_close(rows[:, KV_LORA_RANK:], want_k_pe)
        row_start = row_end
    for s, rid in enumerate(request_ids):
        if cached_lens[s] == 0:
            continue
        kept = env.read_rows(layer_idx, rid, 0, cached_lens[s])
        # Cached prefix (and every generation sequence's rows): bitwise kept.
        _assert_bytes_equal(kept, prefix_rows[s], f"cached prefix (request {rid})")

    # 4. Nothing else in the pool moved: exactly one (C+R)-element row per new
    # context token, at the slot its position addresses. This is also what
    # pins the page-slab geometry, whose element width the op derives from
    # quant_mode alone (1 byte under quant_mode=128).
    changed = pool.reshape(-1).view(torch.uint8) != pool_before.reshape(-1).view(torch.uint8)
    row_bytes = pool.element_size() * LATENT_SIZE
    idx = changed.nonzero().reshape(-1) // row_bytes
    got = set(
        zip(
            (idx // env.tokens_per_block).tolist(),
            (idx % env.tokens_per_block).tolist(),
        )
    )
    expected = {
        env.slot(request_ids[s], cached_lens[s] + i)
        for s in range(num_contexts)
        for i in range(ctx_new[s])
    }
    assert got <= expected, (
        f"op touched pool slots {sorted(got - expected)} outside the "
        f"{len(expected)} (page, slot) pairs its inputs address"
    )
    # Upper bound, not equality: a written byte that happens to match the byte
    # already there is invisible to a snapshot diff. The rows' contents are
    # pinned above.
    assert idx.numel() <= num_tokens * row_bytes, (
        f"op wrote {idx.numel()} pool bytes, at most {num_tokens * row_bytes} rows' worth expected"
    )

    return _Run(
        q=q,
        latent_cache=latent_cache,
        latent_orig=latent_orig,
        ref_k_pe=ref_k_pe,
        ref_k_pe_fp32=ref_k_pe_fp32,
        rows=torch.cat(appended, dim=0),
        positions=positions,
    )


def test_bf16_mixed_cached_lengths_layer1() -> None:
    """Prefill-like batch (~400 new tokens): cached prefixes of 0 (fresh
    prefill), exactly one block, mid-block, and near max_seq_len; addressed
    through pool-mapping row 1 of a two-layer pool."""
    torch.manual_seed(0)
    env = _MlaCtxEnv(num_layers=2)
    try:
        cached = [0, 64, 100, 511]
        new = [37, 64, 300, 1]
        rids = [0, 1, 2, 3]
        env.kv_cache_manager.add_dummy_requests(
            rids, token_nums=[c + n for c, n in zip(cached, new)]
        )
        _run_and_check(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=4,
            cached_lens=cached,
            layer_idx=1,
        )
    finally:
        env.shutdown()


def test_bf16_trailing_generation_seqs_ignored() -> None:
    """Mixed batch: two context sequences followed by two generation
    sequences. The op must touch only the context sequences' new rows and
    index per-seq tensors over [0, num_contexts)."""
    torch.manual_seed(1)
    env = _MlaCtxEnv()
    try:
        cached = [128, 3, 200, 77]
        new = [40, 60, 1, 1]
        rids = [0, 1, 2, 3]
        env.kv_cache_manager.add_dummy_requests(
            rids, token_nums=[c + n for c, n in zip(cached, new)]
        )
        _run_and_check(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=2,
            cached_lens=cached,
        )
    finally:
        env.shutdown()


def test_bf16_single_token_append() -> None:
    """Smallest cached-context case: one sequence, one cached token plus one
    new token (decode-like single-row call, appended at position 1)."""
    torch.manual_seed(2)
    env = _MlaCtxEnv()
    try:
        env.kv_cache_manager.add_dummy_requests([0], token_nums=[2])
        _run_and_check(
            env,
            request_ids=[0],
            seq_lens=[1],
            num_contexts=1,
            cached_lens=[1],
        )
    finally:
        env.shutdown()


def test_fp16_mixed_cached_lengths() -> None:
    """fp16 activations with an fp16 latent cache over block-crossing
    cached/new lengths."""
    torch.manual_seed(3)
    env = _MlaCtxEnv(cache_dtype=DataType.HALF)
    try:
        cached = [65, 640, 1]
        new = [63, 128, 6]
        rids = [0, 1, 2]
        env.kv_cache_manager.add_dummy_requests(
            rids, token_nums=[c + n for c, n in zip(cached, new)]
        )
        _run_and_check(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=3,
            cached_lens=cached,
        )
    finally:
        env.shutdown()


def test_bf16_page32_mixed_cached_lengths_layer1() -> None:
    """Page size 32 (the engine default). The four cached prefixes place the
    first new token at every alignment a 32-token page has: 0 (fresh
    prefill), 32 (page 1 slot 0, an exact boundary), 100 (page 3 slot 4,
    mid-page) and 511 (page 15 slot 31, a page's last slot). The 300-token
    sequence then walks ten pages in one call. Addressed through
    pool-mapping row 1 of a two-layer pool."""
    torch.manual_seed(4)
    env = _MlaCtxEnv(num_layers=2, tokens_per_block=PAGE32)
    try:
        cached = [0, 32, 100, 511]
        new = [37, 64, 300, 1]
        rids = [0, 1, 2, 3]
        env.kv_cache_manager.add_dummy_requests(
            rids, token_nums=[c + n for c, n in zip(cached, new)]
        )
        _run_and_check(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=4,
            cached_lens=cached,
            layer_idx=1,
        )
    finally:
        env.shutdown()


def test_bf16_page32_trailing_generation_seqs_ignored() -> None:
    """Page size 32, mixed batch: two context sequences followed by two
    generation sequences. Sequence 0's new tokens fill page 3 exactly
    (96..127); sequence 1 starts on page 0's last slot and crosses on its
    very first token (31..63). The op must still touch only the context
    rows and index per-seq tensors over [0, num_contexts)."""
    torch.manual_seed(5)
    env = _MlaCtxEnv(tokens_per_block=PAGE32)
    try:
        cached = [96, 31, 200, 77]
        new = [32, 33, 1, 1]
        rids = [0, 1, 2, 3]
        env.kv_cache_manager.add_dummy_requests(
            rids, token_nums=[c + n for c, n in zip(cached, new)]
        )
        _run_and_check(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=2,
            cached_lens=cached,
        )
    finally:
        env.shutdown()


def test_fp16_page32_mixed_cached_lengths() -> None:
    """fp16 activations and fp16 latent cache at page size 32: a prefix
    ending mid-page (33 + 31 closes page 1 exactly), a 16-page prefix
    followed by three full pages of new tokens (512 + 96), and a
    single-page tail."""
    torch.manual_seed(6)
    env = _MlaCtxEnv(cache_dtype=DataType.HALF, tokens_per_block=PAGE32)
    try:
        cached = [33, 512, 1]
        new = [31, 96, 6]
        rids = [0, 1, 2]
        env.kv_cache_manager.add_dummy_requests(
            rids, token_nums=[c + n for c, n in zip(cached, new)]
        )
        _run_and_check(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=3,
            cached_lens=cached,
        )
    finally:
        env.shutdown()


def _fp8_env(max_batch_size: int = 8, orig_quant: Optional[float] = None) -> _MlaCtxEnv:
    """DeepSeek-R1-0528 cell over an fp8-e4m3 latent pool: H = 128, page 32,
    bf16 activations, single-layer pool."""
    return _MlaCtxEnv(
        max_batch_size=max_batch_size,
        tokens_per_block=PAGE32,
        num_heads=NUM_HEADS_R1,
        fp8_pool=True,
        orig_quant=orig_quant,
    )


def test_fp8_kv_production_config() -> None:
    """The production fp8 context call: the KV scaling factor omitted (what
    TrtllmAttention.mla_rope_append_paged_kv_assign_q passes) at H = 128 and
    page size 32. Four context sequences append T = 402 rows in one call, with
    cached prefixes placing the first new token at every alignment a 32-slot
    page has — 0 (fresh prefill), 32 (page 1 slot 0), 100 (page 3 slot 4) and
    511 (page 15's last slot) — and the 300-token sequence walking ten
    pages."""
    torch.manual_seed(20)
    env = _fp8_env()
    try:
        cached = [0, 32, 100, 511]
        new = [37, 64, 300, 1]
        rids = [0, 1, 2, 3]
        env.kv_cache_manager.add_dummy_requests(
            rids, token_nums=[c + n for c, n in zip(cached, new)]
        )
        _run_and_check(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=4,
            cached_lens=cached,
        )
    finally:
        env.shutdown()


def test_fp8_kv_explicit_unit_scale() -> None:
    """Explicit scaling factor 1.0 (a checkpoint whose k_scale/v_scale are the
    production 1.0, passed as a real tensor), then the wrong-variant controls
    that measure what the e4m3 gate on the appended k_pe half discriminates.
    Every variant below is a plausible mis-derivation of the same bytes."""
    torch.manual_seed(21)
    env = _fp8_env(orig_quant=1.0)
    try:
        cached = [0, 31]
        new = [33, 40]
        rids = [0, 1]
        env.kv_cache_manager.add_dummy_requests(
            rids, token_nums=[c + n for c, n in zip(cached, new)]
        )
        run = _run_and_check(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=2,
            cached_lens=cached,
        )
        # Controls. Every appended byte above came back bit-exact, which is
        # only meaningful if the same comparison moves for a wrong mirror.
        # Measured on sm_100 at this case — the whole appended k_pe block
        # (73 rows x 64 elements = 4672 bytes, bit-exact allowance 4) against
        # four plausible mis-derivations, counting bytes differing and the
        # elements the one-ulp tolerance itself rejects:
        #
        #   mirror                       inexact bytes    outside tolerance
        #   un-roped k_pe                 3278 (819x)      2609
        #   rope at position + 1          1799 (450x)      1133
        #   the previous token's row      4607 (1152x)     4469
        #   no bf16 rounding before e4m3    152 (38x)         1
        #   (the correct mirror)              0                0
        #
        # The first three blow both halves of the gate. The fourth —
        # quantizing the unrounded fp32 RoPE result instead of the bf16 one
        # the kernel rounds to first — differs from the truth by a single
        # rounding step, so it is invisible to the tolerance by construction
        # (1 element out of 4672 crosses it, which is not something to rely
        # on) and only the bit-exact-fraction floor catches it, at 38x the
        # allowance. Both halves are asserted below to actually fire.
        got = run.rows[:, KV_LORA_RANK:]
        allowed = max(1, int(E4M3_MAX_INEXACT_FRACTION * got.numel()))
        shifted = [p + 1 for p in run.positions]
        blunt = {
            "un-roped k_pe": env.quantize(run.latent_orig[:, KV_LORA_RANK:]),
            "rope at position + 1": env.quantize(
                env.rope_ref(run.latent_orig[:, KV_LORA_RANK:], shifted)
            ),
            "the previous token's row": env.quantize(run.ref_k_pe.roll(1, dims=0)),
        }
        for name, mirror in blunt.items():
            inexact = _inexact_bytes(got, mirror)
            assert inexact > 100 * allowed, (
                f"control '{name}' moved only {inexact}/{got.numel()} bytes — "
                "the bit-exact-fraction floor would not have caught it"
            )
            assert _fails_e4m3_tolerance(got, mirror), (
                f"control '{name}' stayed inside the one-ulp tolerance — that "
                "half of the gate would not have caught it"
            )
        # The rounding-level control: the fraction floor has to carry it alone.
        no_bf16 = (run.ref_k_pe_fp32 * env.write_scale).to(torch.float8_e4m3fn)
        inexact = _inexact_bytes(got, no_bf16)
        assert inexact > 8 * allowed, (
            f"control 'no bf16 rounding before e4m3' moved only {inexact}/"
            f"{got.numel()} bytes — the bit-exact-fraction floor would not "
            "have caught it"
        )
    finally:
        env.shutdown()


def test_fp8_kv_scale_factor_sweep() -> None:
    """KV scaling factors other than the production 1.0. 1/1.5 is not a power
    of two, so the write-side multiply is a real rescale rather than an
    exponent shift; 2.0 covers a factor above 1. At 1/1.5 the sweep also runs
    the control that pins the order of the two operations — the kernel rounds
    the fp32 RoPE result to bf16 and *then* multiplies by the scale in fp32,
    so e4m3(bf16(rope * w)) is a different tensor."""
    for scale in (1.0 / 1.5, 0.5, 0.25, 2.0):
        torch.manual_seed(22)
        env = _fp8_env(orig_quant=scale)
        try:
            cached = [0, 31]
            new = [33, 40]
            rids = [0, 1]
            env.kv_cache_manager.add_dummy_requests(
                rids, token_nums=[c + n for c, n in zip(cached, new)]
            )
            run = _run_and_check(
                env,
                request_ids=rids,
                seq_lens=new,
                num_contexts=2,
                cached_lens=cached,
            )
            got = run.rows[:, KV_LORA_RANK:]
            allowed = max(1, int(E4M3_MAX_INEXACT_FRACTION * got.numel()))
            if scale == 1.0 / 1.5:
                # Scaling before the bf16 rounding instead of after is exact
                # for a power of two, so only the non-power-of-two scale
                # separates the two orders. Measured: 49/4672 bytes differ
                # (1.05%), against a bit-exact allowance of 4. Exact at
                # 0.5 / 0.25 / 2.0, which is why the control runs only here.
                scale_first = (
                    (run.ref_k_pe_fp32 * env.write_scale).to(torch.bfloat16).to(torch.float8_e4m3fn)
                )
                inexact = _inexact_bytes(got, scale_first)
                assert inexact > 4 * allowed, (
                    f"control 'scale before the bf16 rounding' moved only "
                    f"{inexact}/{got.numel()} bytes — the bit-exact-fraction "
                    "floor would not have caught it"
                )
            # And the scale is really applied: a mirror that ignores it (the
            # w = 1.0 quantization) blows both halves of the gate at every
            # scale swept here.
            unscaled = run.ref_k_pe.float().to(torch.float8_e4m3fn)
            inexact = _inexact_bytes(got, unscaled)
            assert inexact > 100 * allowed and _fails_e4m3_tolerance(got, unscaled), (
                f"a mirror ignoring kv_scale_orig_quant={scale} moved only "
                f"{inexact}/{got.numel()} bytes or stayed inside the one-ulp "
                "tolerance — the scale is not discriminated"
            )
        finally:
            env.shutdown()


def test_fp8_kv_trailing_generation_seqs_ignored() -> None:
    """fp8 mixed batch: two context sequences followed by two generation
    sequences, all with cached prefixes. Sequence 0's new tokens fill page 3
    exactly (96..127); sequence 1 starts on page 0's last slot and crosses on
    its very first token (31..63). The op must touch only the context rows,
    index per-seq tensors over [0, num_contexts), and leave both generation
    sequences' pages bitwise untouched."""
    torch.manual_seed(23)
    env = _fp8_env(orig_quant=1.0 / 1.5)
    try:
        cached = [96, 31, 200, 77]
        new = [32, 33, 1, 1]
        rids = [0, 1, 2, 3]
        env.kv_cache_manager.add_dummy_requests(
            rids, token_nums=[c + n for c, n in zip(cached, new)]
        )
        _run_and_check(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=2,
            cached_lens=cached,
        )
    finally:
        env.shutdown()


def test_fp8_kv_scale_does_not_reach_q() -> None:
    """The KV scaling factor is a write-side quantization scale for the pool
    alone. Two runs over identical inputs (same seed, same batch) at
    kv_scale_orig_quant 1.0 and 0.25 must produce bitwise-identical q and
    latent_cache, and appended rows that differ."""
    runs = []
    for scale in (1.0, 0.25):
        torch.manual_seed(24)
        env = _fp8_env(orig_quant=scale)
        try:
            cached = [17, 0]
            new = [20, 45]
            rids = [0, 1]
            env.kv_cache_manager.add_dummy_requests(
                rids, token_nums=[c + n for c, n in zip(cached, new)]
            )
            run = _run_and_check(
                env,
                request_ids=rids,
                seq_lens=new,
                num_contexts=2,
                cached_lens=cached,
            )
            runs.append((run.q.clone(), run.latent_cache.clone(), run.rows.clone()))
        finally:
            env.shutdown()
    (q_a, latent_a, rows_a), (q_b, latent_b, rows_b) = runs
    _assert_bytes_equal(q_a, q_b, "roped q across kv_scale_orig_quant")
    _assert_bytes_equal(latent_a, latent_b, "roped latent_cache across kv_scale_orig_quant")
    # And the appended rows really did move, so the comparison above is not
    # vacuously true of a call that ignored the scale everywhere.
    assert _inexact_bytes(rows_a, rows_b) > rows_a.numel() // 2, (
        "appended rows barely changed between scale 1.0 and 0.25 — the scale "
        "does not reach the pool either"
    )


def test_fp8_kv_rejects_int8_kv_cache_quant_mode() -> None:
    """A quant_mode asking for a non-fp8 quantized KV cache is rejected, not
    silently treated as fp8 or as high precision."""
    torch.manual_seed(25)
    env = _fp8_env(orig_quant=1.0)
    try:
        env.kv_cache_manager.add_dummy_requests([0], token_nums=[8])
        metadata = env.prepare_metadata([0], [8], 1, [0])
        q = torch.randn(8, NUM_HEADS_R1 * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        latent_cache = torch.randn(8, LATENT_SIZE, dtype=torch.bfloat16, device="cuda")
        block_offsets = metadata.kv_cache_block_offsets
        pool_pointers = env.kv_cache_manager.kv_cache_pool_pointers
        pool_mapping = env.kv_cache_manager.kv_cache_pool_mapping
        assert block_offsets is not None
        assert pool_pointers is not None and pool_mapping is not None
        raised = ""
        try:
            mla_rope_append_paged_kv_assign_q(
                q,
                latent_cache,
                1,
                metadata.ctx_cached_token_indptr,
                metadata.ctx_kv_indptr,
                int(metadata.max_ctx_seq_len),
                env.rotary_cos_sin,
                NUM_HEADS_R1,
                QK_NOPE_HEAD_DIM,
                QK_ROPE_HEAD_DIM,
                KV_LORA_RANK,
                block_offsets,
                pool_pointers,
                pool_mapping,
                env.kv_scale_orig_quant,
                0,  # residual_dim
                0,  # layer_idx
                env.tokens_per_block,
                MAX_SEQ_LEN,
                1,  # beam_width
                QUANT_MODE_INT8_KV_CACHE,
            )
            torch.cuda.synchronize()
        except RuntimeError as exc:
            raised = str(exc)
        # rc26 added NVFP4 latent pools, so the rejection message now
        # enumerates two accepted formats. int8 is still rejected --
        # what this test certifies -- only the wording widened.
        assert "Only FP8 and NVFP4 KV caches are supported for now" in raised, (
            f"quant_mode={QUANT_MODE_INT8_KV_CACHE} was not rejected as expected; got: {raised!r}"
        )
    finally:
        env.shutdown()
