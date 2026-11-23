# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Hy4-Flash target's mla_rope_append_paged_kv_assign_q entry.

Builds a real MLA KVCacheManagerV2 and context-MLA-with-cached-KV metadata, pre-writes cached-prefix
rows, and checks q's and latent_cache's rope tails rotated in place, the appended paged rows against
the entry's `reference`, and every other byte (nope slices, compressed_kv, cached prefix, pool) untouched.
"""

import zlib
from typing import List, Optional

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.attention.mla_rope_append_paged_kv_assign_q__r14 import (  # noqa: E501
    mla_rope_append_paged_kv_assign_q as op,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["../../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mla_rope_append_paged_kv_assign_q requires a CUDA device"

NUM_HEADS_SWA = 64
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 192
TOKENS_PER_BLOCK = 32
#: KvCacheConfig.max_num_tokens default: the largest row count of one context call.
MAX_SEQ_LEN = 8192
ROPE_THETA_SWA = 10_000.0
QUANT_MODE_FP8_KV_CACHE = 128

#: One e4m3 ulp, plus a floor on how many appended bytes may miss bit-exactness.
E4M3_ULP_RTOL = 2**-3
E4M3_MAX_INEXACT_FRACTION = 1e-3
#: q stays unquantized on the fp8 path: ~6% of continuous bf16 values survive an e4m3 round trip, 100% if quantized.
MAX_E4M3_ROUNDTRIP_FRACTION = 0.25


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Within one e4m3 ulp, and at most `E4M3_MAX_INEXACT_FRACTION` of bytes not bit-exact."""
    assert out.shape == ref.shape, (out.shape, ref.shape)
    torch.testing.assert_close(out.float(), ref.float(), rtol=E4M3_ULP_RTOL, atol=0.0)
    inexact = _inexact_bytes(out, ref)
    allowed = max(1, int(E4M3_MAX_INEXACT_FRACTION * out.numel()))
    assert inexact <= allowed, f"{inexact}/{out.numel()} bytes differ (allowed {allowed})"


def _inexact_bytes(got: torch.Tensor, want: torch.Tensor) -> int:
    return int((got.reshape(-1).view(torch.uint8) != want.reshape(-1).view(torch.uint8)).sum())


def _assert_bytes_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert _inexact_bytes(got, want) == 0, f"{what} not bit-exact"


def _e4m3_roundtrip_fraction(x: torch.Tensor) -> float:
    rt = x.float().to(torch.float8_e4m3fn).to(x.dtype)
    return float((x.reshape(-1) == rt.reshape(-1)).float().mean())


class _MlaCtxEnv:
    """One-layer MLA (kv_factor=1) paged KV cache, the duplicated RoPE table, and context metadata."""

    def __init__(
        self,
        num_heads: int,
        *,
        fp8_pool: bool = False,
        orig_quant: Optional[float] = None,
        tokens_per_block: int = TOKENS_PER_BLOCK,
        theta: float = ROPE_THETA_SWA,
        max_seq_len: int = MAX_SEQ_LEN,
        max_batch_size: int = 8,
        nope_size: int = QK_NOPE_HEAD_DIM,
        rope_size: int = QK_ROPE_HEAD_DIM,
        lora_size: int = KV_LORA_RANK,
    ) -> None:
        self.torch_dtype = torch.bfloat16
        self.num_heads = num_heads
        self.nope_size = nope_size
        self.rope_size = rope_size
        self.lora_size = lora_size
        self.q_head_size = nope_size + rope_size
        self.latent_size = lora_size + rope_size
        self.max_batch_size = max_batch_size
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.fp8_pool = fp8_pool
        self.pool_dtype = torch.float8_e4m3fn if fp8_pool else self.torch_dtype
        self.quant_mode = QUANT_MODE_FP8_KV_CACHE if fp8_pool else 0
        self.kv_cache_manager = KVCacheManagerV2(
            KvCacheConfig(max_tokens=131072, enable_block_reuse=False),
            CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=self.latent_size,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.FP8 if fp8_pool else DataType.BF16,
        )
        assert self.kv_cache_manager.tokens_per_block == tokens_per_block
        pool = self.kv_cache_manager.get_buffers(0)
        assert pool is not None and pool.dtype == self.pool_dtype
        self.kv_scale_orig_quant = (
            None
            if orig_quant is None
            else torch.full((1,), orig_quant, dtype=torch.float32, device="cuda")
        )
        # A private copy: RopeParams caches the tensor by params.
        rope = RopeParams(
            dim=rope_size, theta=theta, max_positions=max_seq_len, duplicate_data=True
        )
        _, cos_sin = rope.create_rope_const_params()
        self.rotary_cos_sin = cos_sin.clone()

    def prepare_metadata(
        self,
        request_ids: List[int],
        seq_lens: List[int],
        num_contexts: int,
        cached_lens: List[int],
    ) -> TrtllmAttentionMetadata:
        metadata = TrtllmAttentionMetadata(
            max_num_requests=self.max_batch_size,
            max_num_tokens=MAX_SEQ_LEN,
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

    def rope_ref(self, x: torch.Tensor, positions: List[int]) -> torch.Tensor:
        """GPT-J interleaved rotation of the last dim in fp32, rounded once to x's dtype."""
        half = self.rope_size // 2
        table = self.rotary_cos_sin.view(-1, self.rope_size, 2)
        pos = torch.tensor(positions, dtype=torch.long, device="cuda")
        shape = [len(positions)] + [1] * (x.dim() - 2) + [half]
        cos = table[pos, :half, 0].view(shape)
        sin = table[pos, :half, 1].view(shape)
        pairs = x.float().reshape(*x.shape[:-1], half, 2)
        out = torch.empty_like(pairs)
        out[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        out[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        return out.reshape(x.shape).to(x.dtype)

    def pool_tensor(self) -> torch.Tensor:
        return self.kv_cache_manager.get_buffers(0)

    def blocks(self, request_id: int) -> List[int]:
        return list(self.kv_cache_manager.get_batch_cache_indices([request_id])[0])

    def fill_rows(self, request_id: int, length: int) -> torch.Tensor:
        """Write random latent rows at positions [0, length) of one sequence; return them."""
        draw_dtype = torch.float32 if self.fp8_pool else self.torch_dtype
        rows = torch.randn(length, self.latent_size, dtype=draw_dtype, device="cuda").to(
            self.pool_dtype
        )
        pool = self.pool_tensor()  # [pages, 1, tpb, 1, head]
        tpb = self.tokens_per_block
        blocks = self.blocks(request_id)
        for start in range(0, length, tpb):
            n = min(tpb, length - start)
            pool[blocks[start // tpb], 0, :n, 0].copy_(rows[start : start + n])
        return rows

    def read_rows(self, request_id: int, start: int, end: int) -> torch.Tensor:
        """Latent rows at positions [start, end) of one sequence, dense."""
        pool = self.pool_tensor()
        tpb = self.tokens_per_block
        blocks = self.blocks(request_id)
        out = torch.empty(end - start, self.latent_size, dtype=self.pool_dtype, device="cuda")
        pos = start
        while pos < end:
            page, offset = blocks[pos // tpb], pos % tpb
            n = min(tpb - offset, end - pos)
            out[pos - start : pos - start + n] = pool[page, 0, offset : offset + n, 0]
            pos += n
        return out

    def shutdown(self) -> None:
        self.kv_cache_manager.shutdown()


def _drive(
    env: _MlaCtxEnv, cached: List[int], new: List[int], num_contexts: Optional[int] = None
) -> None:
    """One request per sequence; run the op over the context sequences and verify every effect."""
    request_ids = list(range(len(new)))
    num_contexts = len(request_ids) if num_contexts is None else num_contexts
    env.kv_cache_manager.add_dummy_requests(
        request_ids, token_nums=[c + n for c, n in zip(cached, new)]
    )
    kv_lens = [c + s for c, s in zip(cached, new)]
    metadata = env.prepare_metadata(request_ids, new, num_contexts, cached)
    prefix_rows = [env.fill_rows(rid, c) for rid, c in zip(request_ids, cached)]

    num_heads, nope, lora = env.num_heads, env.nope_size, env.lora_size
    ctx_new = new[:num_contexts]
    num_tokens = sum(ctx_new)
    assert int(metadata.num_ctx_tokens) == num_tokens
    assert int(metadata.max_ctx_seq_len) == max(ctx_new)
    positions = [cached[s] + i for s in range(num_contexts) for i in range(ctx_new[s])]

    q = torch.randn(num_tokens, num_heads * env.q_head_size, dtype=env.torch_dtype, device="cuda")
    latent_cache = torch.randn(num_tokens, env.latent_size, dtype=env.torch_dtype, device="cuda")
    q_orig = q.clone()
    latent_orig = latent_cache.clone()
    pool = env.pool_tensor()
    pool_before = pool.clone()

    op(
        q,
        latent_cache,
        num_contexts,
        metadata.ctx_cached_token_indptr,
        metadata.ctx_kv_indptr,
        int(metadata.max_ctx_seq_len),
        env.rotary_cos_sin,
        env.num_heads,
        env.nope_size,
        env.rope_size,
        env.lora_size,
        metadata.kv_cache_block_offsets,
        env.kv_cache_manager.kv_cache_pool_pointers,
        env.kv_cache_manager.kv_cache_pool_mapping,
        env.kv_scale_orig_quant,
        0,  # residual_dim
        0,  # layer_idx
        env.tokens_per_block,
        env.max_seq_len,  # attention_window_size
        1,  # beam_width
        env.quant_mode,
    )
    torch.cuda.synchronize()

    # q and latent_cache: rope tails rotated in place, the rest bitwise untouched.
    q_heads = q.view(num_tokens, num_heads, env.q_head_size)
    q_orig_heads = q_orig.view(num_tokens, num_heads, env.q_head_size)
    torch.testing.assert_close(
        q_heads[..., nope:], env.rope_ref(q_orig_heads[..., nope:], positions)
    )
    _assert_bytes_equal(q_heads[..., :nope], q_orig_heads[..., :nope], "q nope")
    torch.testing.assert_close(
        latent_cache[:, lora:], env.rope_ref(latent_orig[:, lora:], positions)
    )
    _assert_bytes_equal(latent_cache[:, :lora], latent_orig[:, :lora], "compressed_kv")
    if env.fp8_pool:
        # The in-place rotations stay unquantized; the context FMHA quantizes q itself.
        for values in (q_heads[..., nope:], latent_cache[:, lora:], q_orig_heads[..., :nope]):
            assert _e4m3_roundtrip_fraction(values) < MAX_E4M3_ROUNDTRIP_FRACTION

    # Appended rows match the entry's reference; cached prefixes are bitwise kept.
    table = env.rotary_cos_sin.view(-1, env.rope_size, 2)
    pos = torch.tensor(positions, dtype=torch.long, device="cuda")
    cos_rows = table[pos, : env.rope_size // 2, 0]
    sin_rows = table[pos, : env.rope_size // 2, 1]
    row_start = 0
    for s in range(num_contexts):
        rid = request_ids[s]
        rows = env.read_rows(rid, cached[s], kv_lens[s])
        row_end = row_start + ctx_new[s]
        want = op.reference(
            latent_orig[row_start:row_end],
            cos_rows[row_start:row_end],
            sin_rows[row_start:row_end],
            env.rope_size,
            env.kv_scale_orig_quant,
            env.pool_dtype,
        )
        _assert_bytes_equal(
            rows[:, :lora], want[:, :lora], f"appended compressed_kv (request {rid})"
        )
        if env.fp8_pool:
            _compare(rows[:, lora:], want[:, lora:])
        else:
            torch.testing.assert_close(rows[:, lora:], want[:, lora:])
        row_start = row_end
    for s, rid in enumerate(request_ids):
        if cached[s]:
            _assert_bytes_equal(
                env.read_rows(rid, 0, cached[s]), prefix_rows[s], f"cached prefix ({rid})"
            )

    # Nothing else in the pool moved: only the (page, slot) pairs the new context tokens address.
    changed = pool.reshape(-1).view(torch.uint8) != pool_before.reshape(-1).view(torch.uint8)
    row_bytes = pool.element_size() * env.latent_size
    idx = changed.nonzero().reshape(-1) // row_bytes
    tpb = env.tokens_per_block
    got = set(zip((idx // tpb).tolist(), (idx % tpb).tolist()))
    expected = set()
    for s in range(num_contexts):
        blocks = env.blocks(request_ids[s])
        expected |= {(blocks[p // tpb], p % tpb) for p in range(cached[s], kv_lens[s])}
    assert got <= expected, f"op touched pool slots {sorted(got - expected)[:8]} outside its rows"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    spec = cell.spec
    assert spec["residual_dim"] == 0 and spec["quant_mode"] in (0, QUANT_MODE_FP8_KV_CACHE), spec
    torch.manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    fp8_pool = spec["quant_mode"] == QUANT_MODE_FP8_KV_CACHE
    env = _MlaCtxEnv(
        spec["head_num"],
        fp8_pool=fp8_pool,
        orig_quant=1.0 if fp8_pool else None,
        tokens_per_block=spec["tokens_per_block"],
        nope_size=spec["nope_size"],
        rope_size=spec["rope_size"],
        lora_size=spec["lora_size"],
    )
    certifying(op)
    try:
        _drive(env, spec["cached"], spec["new"])
    finally:
        env.shutdown()


def test_fp8_kv_trailing_generation_seqs_ignored() -> None:
    """Two context sequences (one filling a page exactly, one starting on a page's last slot) followed by
    two generation sequences: only the context rows may be written."""
    torch.manual_seed(23)
    env = _MlaCtxEnv(NUM_HEADS_SWA, fp8_pool=True, orig_quant=1.0 / 1.5)
    try:
        _drive(env, cached=[96, 31, 200, 77], new=[32, 33, 1, 1], num_contexts=2)
    finally:
        env.shutdown()
