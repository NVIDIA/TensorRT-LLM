# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the load_paged_kv_cache_for_mla catalog entry.

The op reads paged-KV-cache addressing tensors that the runtime normally
derives from a KVCacheManager and a prepared TrtllmAttentionMetadata. The
test builds that state for real — an actual MLA (SELFKONLY, kv_factor=1)
KVCacheManager and a TrtllmAttentionMetadata prepared with
enable_context_mla_with_cached_kv=True — writes known random latent rows
directly into the paged pool at every position of every context sequence,
then checks that the op gathers them back into two contiguous tensors
(compressed_kv, k_pe), in batch order, ignoring trailing generation
sequences.

Two configurations are covered:

1. Matching-dtype pool (`quant_mode=0`, `kv_scale_quant_orig=None`), where
   the gather is a bitwise copy. Covered at both pool page sizes: 64, and
   32 (the engine default), where every sequence spans twice as many pages
   and the exact-fill boundaries fall elsewhere.
2. fp8-e4m3 latent pool (`quant_mode=128`) at the DeepSeek-R1-0528 cell —
   `kv_lora_rank=512`, `qk_rope_head_dim=64`, `tokens_per_block=32`,
   `beam_width=1` — where the pool holds one byte per element and the
   gather dequantizes to bf16 / fp16 / fp32 by multiplying with
   `kv_scale_quant_orig`. The pool is filled with `e4m3(row * w)` for an
   explicit write-side scale `w` (the write-side op's certified formula) so
   that a deliberately inconsistent `(w, r)` pair can be driven through the
   read side.
"""

from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.staircase.catalog.attention.load_paged_kv_cache_for_mla import (
    load_paged_kv_cache_for_mla,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

assert torch.cuda.is_available(), "load_paged_kv_cache_for_mla requires a CUDA device"

# DeepSeek-V3 MLA latent geometry.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
# Pool page size. 32 is the engine default (KvCacheConfig.tokens_per_block),
# 64 the value a tuned MLA target opts into; both are covered below.
TOKENS_PER_BLOCK = 64
PAGE32 = 32
MAX_SEQ_LEN = 1024

# quant_mode bits (tensorrt_llm.quantization.mode.QuantMode).
QM_INT8_KV_CACHE = 64
QM_FP8_KV_CACHE = 128
QM_FP8_QDQ = 256
QM_FP8_1X128_128X128 = 1024
QM_NVFP4_KV_CACHE = 8192

# e4m3 constants used to derive tolerances. The format carries 3 explicit
# mantissa bits, so a round-to-nearest quantization is within half an ulp =
# 2**-4 relative; the smallest subnormal is 2**-9, so the absolute error near
# zero is at most 2**-10.
E4M3_HALF_ULP_REL = 2.0**-4
E4M3_MIN_SUBNORMAL = 2.0**-9

_TRTLLM_TO_TORCH_DTYPE = {
    DataType.BF16: torch.bfloat16,
    DataType.HALF: torch.float16,
}


class _MlaCacheEnv:
    """Real op state: MLA (kv_factor=1) paged KV cache manager.

    With fp8_pool the manager allocates an e4m3 latent pool (one byte per
    element) while the gather's out_dtype stays a high-precision type.
    """

    def __init__(
        self,
        cache_dtype: DataType,
        num_layers: int = 1,
        max_batch_size: int = 8,
        tokens_per_block: int = TOKENS_PER_BLOCK,
        fp8_pool: bool = False,
        max_tokens: int = 131072,
    ) -> None:
        self.torch_dtype = _TRTLLM_TO_TORCH_DTYPE[cache_dtype]
        self.max_batch_size = max_batch_size
        self.tokens_per_block = tokens_per_block
        self.fp8_pool = fp8_pool
        self.pool_dtype = torch.float8_e4m3fn if fp8_pool else self.torch_dtype
        self.quant_mode = QM_FP8_KV_CACHE if fp8_pool else 0
        self.kv_cache_manager = KVCacheManager(
            KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
            CacheType.SELFKONLY,  # MLA latent cache: kv_factor=1, one kv head
            num_layers=num_layers,
            num_kv_heads=1,
            head_dim=HEAD_SIZE,
            tokens_per_block=tokens_per_block,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.FP8 if fp8_pool else cache_dtype,
        )
        # The pool must really be paged at the size and element type the case
        # claims: the op derives both from tokens_per_block and quant_mode,
        # not from anything the manager tells it.
        assert self.kv_cache_manager.tokens_per_block == tokens_per_block
        pool = self.kv_cache_manager.get_buffers(0)
        assert pool is not None and pool.dtype == self.pool_dtype

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

    def pool_tensor(self, layer_idx: int) -> torch.Tensor:
        pool = self.kv_cache_manager.get_buffers(layer_idx)
        assert pool is not None
        return pool

    def blocks(self, request_id: int) -> List[int]:
        return list(self.kv_cache_manager.get_batch_cache_indices([request_id])[0])

    def fill_layer(
        self,
        layer_idx: int,
        request_ids: List[int],
        kv_lens: List[int],
        write_scale: float = 1.0,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Write known random latent rows at positions [0, L_s) of each
        sequence's pages.

        Returns (stored, source): `stored` is what physically sits in the
        pool, `source` the unquantized fp32 rows it came from. On the fp8
        pool `stored = e4m3(source * write_scale)` — the write-side op's
        certified formula, mirrored in torch. Every drawn value stays well
        inside e4m3's +-448 range, which is where the torch cast and the
        kernel's saturating cast agree.
        """
        pool = self.pool_tensor(layer_idx)  # [pages,1,tpb,1,head]
        tpb = self.tokens_per_block
        block_ids = self.kv_cache_manager.get_batch_cache_indices(request_ids)
        stored_per_seq, source_per_seq = [], []
        for blocks, length in zip(block_ids, kv_lens):
            if self.fp8_pool:
                source = torch.randn(length, HEAD_SIZE, dtype=torch.float32, device="cuda")
                stored = (source * write_scale).to(torch.float8_e4m3fn)
            else:
                stored = torch.randn(length, HEAD_SIZE, dtype=self.torch_dtype, device="cuda")
                source = stored.float()
            for start in range(0, length, tpb):
                page = blocks[start // tpb]
                n = min(tpb, length - start)
                pool[page, 0, :n, 0].copy_(stored[start : start + n])
            stored_per_seq.append(stored)
            source_per_seq.append(source)
        return stored_per_seq, source_per_seq

    def shutdown(self) -> None:
        self.kv_cache_manager.shutdown()


def _scale_tensor(value: Optional[float]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return torch.full((1,), value, dtype=torch.float32, device="cuda")


def _call(
    env: _MlaCacheEnv,
    metadata: TrtllmAttentionMetadata,
    num_contexts: int,
    out_dtype: torch.dtype,
    kv_scale_quant_orig: Optional[torch.Tensor],
    layer_idx: int = 0,
    quant_mode: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One wrapper call driven straight off prepared metadata."""
    total_ctx_kv = int(metadata.num_ctx_cached_tokens + metadata.num_ctx_tokens)
    block_offsets = metadata.kv_cache_block_offsets
    pool_pointers = env.kv_cache_manager.kv_cache_pool_pointers
    pool_mapping = env.kv_cache_manager.kv_cache_pool_mapping
    assert block_offsets is not None
    assert pool_pointers is not None and pool_mapping is not None
    compressed_kv, k_pe = load_paged_kv_cache_for_mla(
        out_dtype,
        num_contexts,
        total_ctx_kv,
        int(metadata.max_ctx_kv_len),
        metadata.ctx_kv_indptr,
        block_offsets,
        pool_pointers,
        pool_mapping,
        kv_scale_quant_orig,
        layer_idx,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        env.tokens_per_block,
        MAX_SEQ_LEN,  # attention_window_size
        1,  # beam_width
        env.quant_mode if quant_mode is None else quant_mode,
    )
    torch.cuda.synchronize()
    return compressed_kv, k_pe


def _check_outputs(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    total_ctx_kv: int,
    out_dtype: torch.dtype,
) -> None:
    assert compressed_kv.shape == (total_ctx_kv, KV_LORA_RANK), compressed_kv.shape
    assert k_pe.shape == (total_ctx_kv, QK_ROPE_HEAD_DIM), k_pe.shape
    assert compressed_kv.dtype == out_dtype and k_pe.dtype == out_dtype
    assert compressed_kv.is_cuda and k_pe.is_cuda
    assert compressed_kv.is_contiguous() and k_pe.is_contiguous()


def _run_and_check(
    env: _MlaCacheEnv,
    request_ids: List[int],
    seq_lens: List[int],
    num_contexts: int,
    cached_lens: List[int],
    layer_idx: int = 0,
) -> None:
    """Fill the context sequences' cache rows, run the op, and verify the
    gathered compressed_kv / k_pe against the written rows."""
    kv_lens = [c + s for c, s in zip(cached_lens, seq_lens)]
    metadata = env.prepare_metadata(request_ids, seq_lens, num_contexts, cached_lens)
    ctx_kv_lens = kv_lens[:num_contexts]
    stored, _ = env.fill_layer(layer_idx, request_ids[:num_contexts], ctx_kv_lens)

    total_ctx_kv = int(metadata.num_ctx_cached_tokens + metadata.num_ctx_tokens)
    assert total_ctx_kv == sum(ctx_kv_lens)
    assert int(metadata.max_ctx_kv_len) == max(ctx_kv_lens)

    compressed_kv, k_pe = _call(
        env,
        metadata,
        num_contexts,
        env.torch_dtype,
        None,  # kv_scale_quant_orig: fp8-KV-cache path only
        layer_idx=layer_idx,
    )

    expected = torch.cat(stored, dim=0)  # [total_ctx_kv, HEAD_SIZE]
    _check_outputs(compressed_kv, k_pe, total_ctx_kv, env.torch_dtype)
    # Dtype-preserving gather: outputs must be bitwise equal to the cache rows.
    torch.testing.assert_close(compressed_kv, expected[:, :KV_LORA_RANK], rtol=0.0, atol=0.0)
    torch.testing.assert_close(k_pe, expected[:, KV_LORA_RANK:], rtol=0.0, atol=0.0)


def _dequant_mirror(
    stored: torch.Tensor,
    read_scale: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """float(e4m3 byte) * kv_scale_quant_orig[0] in fp32, rounded once to
    out_dtype — the fp8 gather's reference, built from torch alone."""
    wide = stored.float()
    if read_scale is not None:
        wide = wide * read_scale[0]
    return wide.to(out_dtype)


def _run_fp8(
    env: _MlaCacheEnv,
    request_ids: List[int],
    seq_lens: List[int],
    num_contexts: int,
    cached_lens: List[int],
    out_dtype: torch.dtype,
    read_scale: Optional[float],
    write_scale: float = 1.0,
    layer_idx: int = 0,
    extra_slots: int = 0,
    add_requests: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fill an fp8 pool with e4m3(row * write_scale) and gather it back.

    Returns (compressed_kv, k_pe, stored, source, read_scale_tensor) with
    `stored` / `source` concatenated over the context sequences.
    """
    assert env.fp8_pool
    kv_lens = [c + s for c, s in zip(cached_lens, seq_lens)]
    if add_requests:
        env.kv_cache_manager.add_dummy_requests(
            request_ids, token_nums=[n + extra_slots for n in kv_lens]
        )
    metadata = env.prepare_metadata(request_ids, seq_lens, num_contexts, cached_lens)
    ctx_kv_lens = kv_lens[:num_contexts]
    stored, source = env.fill_layer(layer_idx, request_ids[:num_contexts], ctx_kv_lens, write_scale)
    total_ctx_kv = int(metadata.num_ctx_cached_tokens + metadata.num_ctx_tokens)
    assert total_ctx_kv == sum(ctx_kv_lens)
    scale_t = _scale_tensor(read_scale)
    compressed_kv, k_pe = _call(
        env, metadata, num_contexts, out_dtype, scale_t, layer_idx=layer_idx
    )
    _check_outputs(compressed_kv, k_pe, total_ctx_kv, out_dtype)
    return (
        compressed_kv,
        k_pe,
        torch.cat(stored, dim=0),
        torch.cat(source, dim=0),
        scale_t,
    )


def _assert_fp8_exact(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    stored: torch.Tensor,
    read_scale: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> None:
    """Both halves bitwise equal to the fp32-multiply mirror."""
    mirror = _dequant_mirror(stored, read_scale, out_dtype)
    torch.testing.assert_close(compressed_kv, mirror[:, :KV_LORA_RANK], rtol=0.0, atol=0.0)
    torch.testing.assert_close(k_pe, mirror[:, KV_LORA_RANK:], rtol=0.0, atol=0.0)


def _raises(fn) -> str:
    try:
        fn()
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("expected the call to raise RuntimeError")


# ───────────────────────── matching-dtype pool ──────────────────────────


def test_bf16_mixed_lengths_layer1() -> None:
    """Prefill-like batch (~1k gathered tokens): cached lengths of 0, exactly
    one block, and multi-block; addressed through pool-mapping row 1 of a
    two-layer pool."""
    torch.manual_seed(0)
    env = _MlaCacheEnv(DataType.BF16, num_layers=2)
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


def test_bf16_single_tiny_sequence() -> None:
    """Smallest cached-context case: one sequence, one cached token plus one
    new token (two gathered rows)."""
    torch.manual_seed(1)
    env = _MlaCacheEnv(DataType.BF16)
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


def test_bf16_trailing_generation_seqs_ignored() -> None:
    """Mixed batch: two context sequences followed by two generation
    sequences. The op must gather exactly the context sequences' rows and
    index per-seq tensors over [0, num_contexts) only."""
    torch.manual_seed(2)
    env = _MlaCacheEnv(DataType.BF16)
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


def test_fp16_mixed_lengths() -> None:
    """fp16 latent cache with fp16 out_dtype over block-crossing lengths."""
    torch.manual_seed(3)
    env = _MlaCacheEnv(DataType.HALF)
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


def test_bf16_page32_mixed_lengths_layer1() -> None:
    """Page size 32 (the engine default). Gathered ranges of 37, 96, 400 and
    512 rows: 96 and 512 fill 3 and 16 pages exactly at 32 (neither is a
    page multiple at 64), 400 spans 12 full pages plus 16 rows, 37 spans two
    — every sequence crosses at least one boundary and the deepest one walks
    16 offsets-row entries. Addressed through pool-mapping row 1 of a
    two-layer pool."""
    torch.manual_seed(4)
    env = _MlaCacheEnv(DataType.BF16, num_layers=2, tokens_per_block=PAGE32)
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
    generation sequences. The gathered lengths (128 = 4 exact pages, 63 =
    two pages minus one row) sit either side of a page boundary, and the
    ignored generation sequences own pages of their own."""
    torch.manual_seed(5)
    env = _MlaCacheEnv(DataType.BF16, tokens_per_block=PAGE32)
    try:
        cached = [96, 32, 200, 77]
        new = [32, 31, 1, 1]
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


def test_fp16_page32_mixed_lengths() -> None:
    """fp16 latent cache at page size 32: gathered ranges of 64 (2 exact
    pages), 608 (19 exact pages) and 7 (a partial first page)."""
    torch.manual_seed(6)
    env = _MlaCacheEnv(DataType.HALF, tokens_per_block=PAGE32)
    try:
        cached = [33, 512, 0]
        new = [31, 96, 7]
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


# ───────────────────────── fp8-e4m3 latent pool ─────────────────────────


def _fp8_env(
    num_layers: int = 1,
    tokens_per_block: int = PAGE32,
    max_tokens: int = 131072,
) -> _MlaCacheEnv:
    """DeepSeek-R1-0528 cell: e4m3 latent pool, page 32, C/R = 512/64."""
    return _MlaCacheEnv(
        DataType.BF16,
        num_layers=num_layers,
        tokens_per_block=tokens_per_block,
        fp8_pool=True,
        max_tokens=max_tokens,
    )


def test_fp8_production_config() -> None:
    """The production fp8 context call: quant_mode=128 with
    kv_scale_quant_orig omitted (what the engine's own call site passes), a
    unit-scale pool and bf16 output. Gathered ranges of 37 / 96 / 400 / 512
    rows over page 32. Also locks the three things a pure gather owes: the
    pool is not modified, the slot one past each sequence's range is not
    read, and a second identical call reproduces the first bitwise."""
    torch.manual_seed(40)
    env = _fp8_env()
    try:
        cached = [0, 32, 100, 511]
        new = [37, 64, 300, 1]
        rids = [0, 1, 2, 3]
        # One spare slot per sequence so a sentinel can sit just past L_s.
        ckv, kpe, stored, _, scale_t = _run_fp8(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=4,
            cached_lens=cached,
            out_dtype=torch.bfloat16,
            read_scale=None,
            extra_slots=1,
        )
        assert ckv.shape[0] == 1045, ckv.shape
        _assert_fp8_exact(ckv, kpe, stored, scale_t, torch.bfloat16)

        # Sentinel just past each context sequence's gathered range, plus a
        # whole-pool byte snapshot: the call must read neither and write
        # nothing.
        pool = env.pool_tensor(0)
        sentinel = torch.full((HEAD_SIZE,), 240.0, dtype=torch.float32, device="cuda").to(
            torch.float8_e4m3fn
        )
        for rid, length in zip(rids, [c + n for c, n in zip(cached, new)]):
            page = env.blocks(rid)[length // env.tokens_per_block]
            pool[page, 0, length % env.tokens_per_block, 0].copy_(sentinel)
        before = pool.view(torch.uint8).clone()
        metadata = env.prepare_metadata(rids, new, 4, cached)
        ckv2, kpe2 = _call(env, metadata, 4, torch.bfloat16, None)
        assert int((before != pool.view(torch.uint8)).sum().item()) == 0
        assert not bool((ckv2.float() == 240.0).any().item())
        assert not bool((kpe2.float() == 240.0).any().item())
        # Re-running the same prepared step reproduces the gather bitwise.
        torch.testing.assert_close(ckv2, ckv, rtol=0.0, atol=0.0)
        torch.testing.assert_close(kpe2, kpe, rtol=0.0, atol=0.0)
    finally:
        env.shutdown()


def test_fp8_scale_and_out_dtype_matrix() -> None:
    """kv_scale_quant_orig sweep x out_dtype. The gather is
    `float(e4m3 byte) * r` evaluated in fp32 and rounded once to out_dtype,
    bit for bit, for every combination — including r omitted, which behaves
    exactly as 1.0."""
    for out_dtype in (torch.bfloat16, torch.float16, torch.float32):
        torch.manual_seed(41)
        env = _fp8_env()
        try:
            for i, read_scale in enumerate([None, 1.0, 1.0 / 1.5, 0.5, 0.25, 2.0]):
                ckv, kpe, stored, _, scale_t = _run_fp8(
                    env,
                    request_ids=[i],
                    seq_lens=[40],
                    num_contexts=1,
                    cached_lens=[24],
                    out_dtype=out_dtype,
                    read_scale=read_scale,
                )
                _assert_fp8_exact(ckv, kpe, stored, scale_t, out_dtype)
        finally:
            env.shutdown()


def test_fp8_wrong_mirrors_are_discriminated() -> None:
    """The fp8 gate is bitwise (rtol=atol=0) and is never approached, so the
    same comparison is run in the same process against five wrong mirrors of
    the same 36864-element block. Every one of them must fail it.

    The interesting control is `multiply in out_dtype`: it differs from the
    truth only by where the scale is rounded, so it is invisible to any
    tolerance-based gate and only a bitwise one catches it — and only when
    the scale is not exactly representable in out_dtype (at r = 1.5 or 0.25
    the two mirrors coincide exactly, measured)."""
    torch.manual_seed(42)
    env = _fp8_env()
    try:
        out_dtype = torch.bfloat16
        ckv, kpe, stored, source, scale_t = _run_fp8(
            env,
            request_ids=[0],
            seq_lens=[40],
            num_contexts=1,
            cached_lens=[24],
            out_dtype=out_dtype,
            read_scale=1.0 / 1.5,
            write_scale=1.5,
        )
        assert scale_t is not None
        got = torch.cat([ckv, kpe], dim=1)
        correct = _dequant_mirror(stored, scale_t, out_dtype)
        torch.testing.assert_close(got, correct, rtol=0.0, atol=0.0)

        n = got.numel()
        wrong = {
            # measured: 36843 / 36864 elements differ (99.94%)
            "scale ignored": stored.float().to(out_dtype),
            # measured: 36843 / 36864 (99.94%)
            "reciprocal scale": (stored.float() / scale_t[0]).to(out_dtype),
            # measured: 9113 / 36864 (24.72%) — the weakest control
            "multiply in out_dtype": (stored.to(out_dtype) * scale_t[0].to(out_dtype)).to(
                out_dtype
            ),
            # measured: 36358 / 36864 (98.63%)
            "rows shifted by one token": torch.roll(correct, 1, 0),
            # measured: 34362 / 36864 (93.21%)
            "quantization skipped": (source * scale_t[0] * 1.5).to(out_dtype),
        }
        for name, mirror in wrong.items():
            differing = int((got != mirror).sum().item())
            # An allowance of 1% of the block: the correct mirror scores 0 and
            # the weakest wrong one 24.8x this, the rest 93-100x.
            assert differing > n // 100, (name, differing, n)
    finally:
        env.shutdown()


def test_fp8_inconsistent_write_read_scale_pair() -> None:
    """No layer in the chain relates the write scale to the read scale. A
    pool written at `w` and read at `r` comes back as
    `float(e4m3(row * w)) * r` — exactly, with no error anywhere — so only
    `r == 1/w` recovers the original rows and every other pair is a silently
    mis-scaled prefill."""
    torch.manual_seed(43)
    env = _fp8_env()
    try:
        # (write scale, read scale, product): three inconsistent pairs and
        # one consistent one.
        cases = [(2.0, 2.0, 4.0), (4.0, 1.0, 4.0), (1.0, 3.0, 3.0), (0.25, 4.0, 1.0)]
        for i, (w, r, product) in enumerate(cases):
            ckv, kpe, stored, source, scale_t = _run_fp8(
                env,
                request_ids=[i],
                seq_lens=[40],
                num_contexts=1,
                cached_lens=[24],
                out_dtype=torch.bfloat16,
                read_scale=r,
                write_scale=w,
            )
            # Accepted, and exactly the mis-scaled value.
            _assert_fp8_exact(ckv, kpe, stored, scale_t, torch.bfloat16)
            got = torch.cat([ckv, kpe], dim=1).float()
            # Tolerance derived from the e4m3 format: one round-to-nearest
            # quantization at scale w, undone by r.
            rtol = E4M3_HALF_ULP_REL
            atol = 0.5 * E4M3_MIN_SUBNORMAL * r
            torch.testing.assert_close(got, source * product, rtol=rtol, atol=atol)
            if product != 1.0:
                # ...and therefore not the original rows.
                try:
                    torch.testing.assert_close(got, source, rtol=rtol, atol=atol)
                except AssertionError:
                    pass
                else:
                    raise AssertionError(
                        f"w={w} r={r} recovered the source rows; the op cannot be "
                        "applying both scales independently"
                    )
    finally:
        env.shutdown()


def test_fp8_trailing_generation_seqs_ignored() -> None:
    """fp8 mixed batch: two context sequences followed by two generation
    sequences whose pages are filled too. Gathered lengths 128 (four exact
    pages) and 63 straddle a page boundary."""
    torch.manual_seed(44)
    env = _fp8_env()
    try:
        cached = [96, 32, 200, 77]
        new = [32, 31, 1, 1]
        rids = [0, 1, 2, 3]
        ckv, kpe, stored, _, scale_t = _run_fp8(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=2,
            cached_lens=cached,
            out_dtype=torch.bfloat16,
            read_scale=2.0,
            write_scale=0.5,
        )
        assert ckv.shape[0] == 191, ckv.shape
        _assert_fp8_exact(ckv, kpe, stored, scale_t, torch.bfloat16)
    finally:
        env.shutdown()


def test_fp8_two_layer_pool_layer1() -> None:
    """fp8 pool-mapping row 1 of a two-layer pool, with decoy rows written
    at the same positions of layer 0."""
    torch.manual_seed(45)
    env = _fp8_env(num_layers=2)
    try:
        cached = [0, 96]
        new = [64, 33]
        rids = [0, 1]
        env.kv_cache_manager.add_dummy_requests(
            rids, token_nums=[c + n for c, n in zip(cached, new)]
        )
        env.fill_layer(0, rids, [c + n for c, n in zip(cached, new)], 1.0)
        ckv, kpe, stored, _, scale_t = _run_fp8(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=2,
            cached_lens=cached,
            out_dtype=torch.bfloat16,
            read_scale=1.0,
            layer_idx=1,
            add_requests=False,
        )
        _assert_fp8_exact(ckv, kpe, stored, scale_t, torch.bfloat16)
        # Layer 0 holds different rows at the same positions; reading it must
        # return them, not layer 1's.
        metadata = env.prepare_metadata(rids, new, 2, cached)
        ckv0, _ = _call(env, metadata, 2, torch.bfloat16, _scale_tensor(1.0))
        assert not torch.equal(ckv0, ckv)
    finally:
        env.shutdown()


def test_fp8_page64_prefill_scale() -> None:
    """fp8 at page size 64 and prefill scale: 2183 gathered rows across
    three sequences (16 exact pages, 16 exact pages, three partial), fp32
    output, write scale 0.5 undone by read scale 2.0."""
    torch.manual_seed(46)
    env = _fp8_env(tokens_per_block=TOKENS_PER_BLOCK)
    try:
        cached = [512, 0, 128]
        new = [512, 1024, 7]
        rids = [0, 1, 2]
        ckv, kpe, stored, source, scale_t = _run_fp8(
            env,
            request_ids=rids,
            seq_lens=new,
            num_contexts=3,
            cached_lens=cached,
            out_dtype=torch.float32,
            read_scale=2.0,
            write_scale=0.5,
        )
        assert ckv.shape[0] == 2183, ckv.shape
        _assert_fp8_exact(ckv, kpe, stored, scale_t, torch.float32)
        # The consistent pair recovers the pre-quantization rows to within
        # one e4m3 rounding.
        got = torch.cat([ckv, kpe], dim=1)
        torch.testing.assert_close(
            got,
            source,
            rtol=E4M3_HALF_ULP_REL,
            atol=0.5 * E4M3_MIN_SUBNORMAL * 2.0,
        )
    finally:
        env.shutdown()


def test_fp8_byte_domain_and_out_dtype_range() -> None:
    """The dequantization is a plain conversion over the whole e4m3 domain:
    all 256 byte values (including +-448, +-0, subnormals and both NaN
    bytes) come back exactly at r = 1.0. The product is *not* clamped to
    out_dtype's range either — 448 * 256 overflows fp16 to +-inf while bf16
    and fp32 hold 114688."""
    torch.manual_seed(47)
    env = _fp8_env()
    try:
        rids = [0]
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=[32])
        metadata = env.prepare_metadata(rids, [16], 1, [16])
        pool = env.pool_tensor(0)
        page = env.blocks(0)[0]
        all_bytes = torch.arange(256, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
        pool[page, 0, 0, 0, :256].copy_(all_bytes)
        # Row 1 carries the extremes the range check needs.
        extremes = torch.zeros(HEAD_SIZE, dtype=torch.float32, device="cuda")
        extremes[:4] = torch.tensor([448.0, -448.0, E4M3_MIN_SUBNORMAL, 0.0])
        pool[page, 0, 1, 0].copy_(extremes.to(torch.float8_e4m3fn))

        for out_dtype in (torch.float32, torch.bfloat16, torch.float16):
            ckv, _ = _call(env, metadata, 1, out_dtype, None)
            torch.testing.assert_close(
                ckv[0, :256],
                all_bytes.float().to(out_dtype),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
            assert ckv[0, :256].isnan().sum().item() == 2  # 0x7f and 0xff
            torch.testing.assert_close(
                ckv[1, :4],
                extremes[:4].to(out_dtype),
                rtol=0.0,
                atol=0.0,
            )

        big = _scale_tensor(256.0)
        ckv, _ = _call(env, metadata, 1, torch.float16, big)
        assert ckv[1, 0].isinf().item() and ckv[1, 0] > 0
        assert ckv[1, 1].isinf().item() and ckv[1, 1] < 0
        for out_dtype in (torch.bfloat16, torch.float32):
            ckv, _ = _call(env, metadata, 1, out_dtype, big)
            torch.testing.assert_close(
                ckv[1, :2],
                torch.tensor([114688.0, -114688.0], device="cuda").to(out_dtype),
                rtol=0.0,
                atol=0.0,
            )
    finally:
        env.shutdown()


def test_fp8_scale_tensor_forms() -> None:
    """Only element [0] of kv_scale_quant_orig is read, and only its dtype is
    validated: a `[2]` tensor, a `[1]` tensor and a 0-dim scalar carrying the
    same leading value are interchangeable, while a non-fp32 tensor raises."""
    torch.manual_seed(51)
    env = _fp8_env()
    try:
        rids = [0]
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=[32])
        metadata = env.prepare_metadata(rids, [16], 1, [16])
        stored, _ = env.fill_layer(0, rids, [32], 1.0)
        reference = _dequant_mirror(stored[0], _scale_tensor(2.0), torch.bfloat16)
        forms = [
            torch.full((1,), 2.0, dtype=torch.float32, device="cuda"),
            torch.tensor([2.0, 99.0], dtype=torch.float32, device="cuda"),
            torch.tensor(2.0, dtype=torch.float32, device="cuda"),
        ]
        for scale in forms:
            ckv, kpe = _call(env, metadata, 1, torch.bfloat16, scale)
            torch.testing.assert_close(torch.cat([ckv, kpe], dim=1), reference, rtol=0.0, atol=0.0)
        for dtype in (torch.float16, torch.float64):
            bad = torch.full((1,), 2.0, dtype=dtype, device="cuda")
            message = _raises(lambda s=bad: _call(env, metadata, 1, torch.bfloat16, s))
            assert "expected scalar type Float but found" in message, message
    finally:
        env.shutdown()


def test_fp8_quant_mode_and_out_dtype_rejections() -> None:
    """quant_mode: any KV-cache quantization other than fp8 is rejected;
    extra quantization bits alongside the fp8 one are not. out_dtype: only
    fp16 / fp32 / bf16 are accepted."""
    torch.manual_seed(48)
    env = _fp8_env()
    try:
        rids = [0]
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=[32])
        metadata = env.prepare_metadata(rids, [16], 1, [16])
        env.fill_layer(0, rids, [32], 1.0)
        scale_t = _scale_tensor(1.0)

        message = _raises(
            lambda: _call(env, metadata, 1, torch.bfloat16, scale_t, quant_mode=QM_INT8_KV_CACHE)
        )
        assert "Only FP8 KV cache is supported for now" in message, message
        # NVFP4_KV_CACHE fails earlier and for a different reason: the nvfp4
        # pool-pointer layout, checked before the quant_mode branch.
        message = _raises(
            lambda: _call(env, metadata, 1, torch.bfloat16, scale_t, quant_mode=QM_NVFP4_KV_CACHE)
        )
        assert "hostKvCachePoolPointers.dim() == 3" in message, message

        for out_dtype in (torch.float8_e4m3fn, torch.int8, torch.float64):
            message = _raises(lambda od=out_dtype: _call(env, metadata, 1, od, scale_t))
            assert "out_dtype only support float16, float32, bfloat16" in message, message

        # The fp8 bit is what selects the path; other quantization bits ride
        # along unread. 1152 is what an fp8-KV checkpoint's quant config
        # produces (FP8_KV_CACHE | FP8_1x128_128x128).
        base = torch.cat(_call(env, metadata, 1, torch.bfloat16, scale_t), dim=1)
        for quant_mode in (
            QM_FP8_KV_CACHE | QM_FP8_1X128_128X128,
            QM_FP8_KV_CACHE | QM_FP8_QDQ,
        ):
            other = torch.cat(
                _call(env, metadata, 1, torch.bfloat16, scale_t, quant_mode=quant_mode),
                dim=1,
            )
            torch.testing.assert_close(other, base, rtol=0.0, atol=0.0)
    finally:
        env.shutdown()


def test_quant_mode_selects_cache_element_width() -> None:
    """The pool arrives as a raw pointer, so quant_mode alone decides how
    wide a cache element is — one byte with the fp8 bit set, sizeof(out_dtype)
    without it. Both mismatches are silent and both are gated here, which is
    also this file's control that a wrong read is visible to its comparison:
    the same bitwise check that passes on every certified case fails on
    these."""
    torch.manual_seed(49)
    # An fp8 pool read with quant_mode=0: 2-byte elements over a 1-byte pool,
    # so block b is read from byte 2 * b * tokens_per_block * (C+R) and each
    # token consumes 2 * (C+R) bytes.
    env = _fp8_env()
    try:
        rids = [0, 1]
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=[32, 32])
        metadata = env.prepare_metadata(rids, [16, 16], 2, [16, 16])
        stored, _ = env.fill_layer(0, rids, [32, 32], 1.0)
        truth = torch.cat(stored, dim=0).float().to(torch.bfloat16)
        ckv, kpe = _call(env, metadata, 2, torch.bfloat16, None, quant_mode=0)
        got = torch.cat([ckv, kpe], dim=1)
        assert not torch.equal(got, truth)
        pool_bytes = env.pool_tensor(0).view(torch.uint8).reshape(-1)
        stride = 2 * env.tokens_per_block * HEAD_SIZE
        for out_row, rid in ((0, 0), (32, 1)):
            base = env.blocks(rid)[0] * stride
            expected = pool_bytes[base : base + 2 * HEAD_SIZE].view(torch.bfloat16)
            torch.testing.assert_close(got[out_row], expected, rtol=0.0, atol=0.0, equal_nan=True)
    finally:
        env.shutdown()

    torch.manual_seed(50)
    # A bf16 pool read with quant_mode=128: 1-byte elements over a 2-byte
    # pool, so block b is read from byte b * tokens_per_block * (C+R) — half
    # the intended stride, always in bounds and always wrong.
    env = _MlaCacheEnv(DataType.BF16, tokens_per_block=PAGE32, max_tokens=16384)
    try:
        rids = [0, 1]
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=[32, 32])
        metadata = env.prepare_metadata(rids, [16, 16], 2, [16, 16])
        stored, _ = env.fill_layer(0, rids, [32, 32])
        truth = torch.cat(stored, dim=0)
        ckv, kpe = _call(env, metadata, 2, torch.bfloat16, None, quant_mode=QM_FP8_KV_CACHE)
        got = torch.cat([ckv, kpe], dim=1)
        assert not torch.equal(got, truth)
        pool_bytes = env.pool_tensor(0).view(torch.uint8).reshape(-1)
        stride = env.tokens_per_block * HEAD_SIZE
        for out_row, rid in ((0, 0), (32, 1)):
            base = env.blocks(rid)[0] * stride
            expected = (
                pool_bytes[base : base + HEAD_SIZE]
                .view(torch.float8_e4m3fn)
                .float()
                .to(torch.bfloat16)
            )
            torch.testing.assert_close(got[out_row], expected, rtol=0.0, atol=0.0, equal_nan=True)

        # And off the fp8 path kv_scale_quant_orig is read by nothing: the
        # same call with a scale of 4.0 returns byte-identical output.
        unscaled = torch.cat(_call(env, metadata, 2, torch.bfloat16, None), dim=1)
        scaled = torch.cat(_call(env, metadata, 2, torch.bfloat16, _scale_tensor(4.0)), dim=1)
        torch.testing.assert_close(scaled, unscaled, rtol=0.0, atol=0.0)
        torch.testing.assert_close(unscaled, truth, rtol=0.0, atol=0.0)
    finally:
        env.shutdown()
