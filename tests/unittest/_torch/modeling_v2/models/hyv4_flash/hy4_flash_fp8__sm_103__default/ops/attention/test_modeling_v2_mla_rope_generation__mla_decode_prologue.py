# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's mla_rope_generation variant.

Builds a real KVCacheManagerV2 bf16 latent pool and prepared TrtllmAttentionMetadata, and an interleaved
mRoPE table gathered from the engine's duplicated table at `position + offset(stream(pair))`. Roped halves
are gated in bf16 ulps (the kernel may contract `x*cos -+ y*sin` to FMA); everything else is bit-exact.
"""

import zlib
from typing import List, Optional

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.attention.mla_rope_generation__mla_decode_prologue import (  # noqa: E501
    mla_rope_generation as op,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["../../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mla_rope_generation requires a CUDA device"

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 192
V_HEAD_DIM = 256
GEN_HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
HALF = QK_ROPE_HEAD_DIM // 2
MAX_SEQ_LEN = 1024
#: A KV length past the 512-token window.
LONG_KV = 700
#: Prompt length of a context row leading a mixed batch.
CTX_PROMPT = 128
#: Every roped element within 4 bf16 ulps of the tensor's largest value; at most 1e-3 of elements
#: (never fewer than 2) past 4 ulps of their own magnitude (cancellation residues).
BF16_MAX_ULP = 4
BF16_MAX_MULTI_ULP_FRACTION = 1e-3


def _bytes(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.uint8).reshape(-1)


def _assert_bytes_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.shape == want.shape, (got.shape, want.shape)
    assert torch.equal(_bytes(got), _bytes(want)), f"{what} not bit-exact"


def _assert_bf16_roped(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.shape == want.shape, (got.shape, want.shape)
    dist = ((got.view(torch.int16).int() & 0x7FFF) - (want.view(torch.int16).int() & 0x7FFF)).abs()
    worst = float((got.float() - want.float()).abs().max())
    tol = BF16_MAX_ULP * 2.0**-8 * float(want.float().abs().max())
    assert worst <= tol, f"{what}: worst element differs by {worst:.4g} (allowed {tol:.4g})"
    past = int((dist > BF16_MAX_ULP).sum())
    allowed = max(2, int(BF16_MAX_MULTI_ULP_FRACTION * got.numel()))
    assert past <= allowed, f"{what}: {past}/{got.numel()} elements past {BF16_MAX_ULP} ulps"


def _mrope_streams(section) -> List[int]:
    """Interleaved mRoPE: pair `i` reads stream 1 if `i % 3 == 1 and i < 3 * section[1]`, 2 if
    `i % 3 == 2 and i < 3 * section[2]`, else 0."""
    streams = []
    for i in range(HALF):
        if i % 3 == 1 and i < 3 * section[1]:
            streams.append(1)
        elif i % 3 == 2 and i < 3 * section[2]:
            streams.append(2)
        else:
            streams.append(0)
    return streams


class _MlaEnv:
    """An MLA (kv_factor=1) bf16 KV cache manager and the mRoPE table."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        tokens_per_block: int,
        num_heads: int,
        rope_theta: float,
        mrope_section,
        mrope_offsets,
        num_layers: int,
        max_seq_len: int,
        max_tokens: int,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.tokens_per_block = tokens_per_block
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.kv_cache_manager = KVCacheManagerV2(
            KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
            CacheType.SELFKONLY,
            num_layers=num_layers,
            num_kv_heads=1,
            head_dim=GEN_HEAD_SIZE,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.BF16,
        )
        assert self.kv_cache_manager.tokens_per_block == tokens_per_block
        pool = self.kv_cache_manager.get_buffers(0)
        assert pool is not None and pool.dtype == torch.bfloat16
        rope = RopeParams(
            dim=QK_ROPE_HEAD_DIM,
            theta=rope_theta,
            max_positions=max_seq_len + max(mrope_offsets),
            duplicate_data=True,
        )
        _, base = rope.create_rope_const_params()
        base = base.view(-1, QK_ROPE_HEAD_DIM, 2)
        # Gather each pair at `position + offsets[stream]`; the duplicated layout repeats the 32 pairs.
        per_pair = torch.tensor(
            [mrope_offsets[s] for s in _mrope_streams(mrope_section)],
            dtype=torch.long,
            device=base.device,
        )
        rows = torch.arange(max_seq_len, device=base.device)
        first = base[
            rows[:, None] + per_pair[None, :], torch.arange(HALF, device=base.device)[None, :]
        ]
        self.rotary_cos_sin = torch.cat([first, first], dim=1).reshape(1, -1).contiguous()

    def prepare_metadata(
        self, request_ids: List[int], seq_lens: List[int], num_contexts: int, cached_lens: List[int]
    ) -> TrtllmAttentionMetadata:
        metadata = TrtllmAttentionMetadata(
            max_num_requests=self.max_batch_size,
            max_num_tokens=8192,
            kv_cache_manager=self.kv_cache_manager,
        )
        metadata.seq_lens = torch.tensor(seq_lens, dtype=torch.int)
        metadata.num_contexts = num_contexts
        metadata.request_ids = request_ids
        metadata.prompt_lens = cached_lens
        metadata.kv_cache_params = KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=cached_lens
        )
        metadata.prepare()
        return metadata

    def cos_sin(self, position: int):
        rows = self.rotary_cos_sin.view(-1, QK_ROPE_HEAD_DIM, 2)
        return rows[position, :HALF, 0], rows[position, :HALF, 1]

    def pool_tensor(self, layer_idx: int) -> torch.Tensor:
        return self.kv_cache_manager.get_buffers(layer_idx)

    def slot(self, request_id: int, position: int, layer_idx: int) -> tuple[int, int]:
        blocks = self.kv_cache_manager.get_batch_cache_indices([request_id], layer_idx=layer_idx)[0]
        return blocks[position // self.tokens_per_block], position % self.tokens_per_block

    def pool_page(self, layer_idx: int, page: int) -> int:
        """Rows of one pool interleave page by page: row `L`'s page `p` is the pool's page `mapping[L][1] + p`."""
        return int(self.kv_cache_manager.kv_cache_pool_mapping[layer_idx][1]) + page

    def shutdown(self) -> None:
        self.kv_cache_manager.shutdown()


def _cached_lens(
    tokens_per_block: int, p: int, num_seqs: int, long_kv: int, max_seq_len: int
) -> List[int]:
    """Sequence 0 past the window; the next three put the P rows two slots before a page's end, on its last
    slot and on the next page's first; the rest walk one slot at a time past the boundary."""
    cached = [long_kv, tokens_per_block - 2, tokens_per_block - 1, tokens_per_block]
    cached += [tokens_per_block + 1 + i for i in range(num_seqs - 4)]
    assert len(set(cached)) == num_seqs and max(cached) + p <= max_seq_len, cached
    return cached


class _Step:
    """One prepared generation step: `num_contexts` context rows, then the generation sequences."""

    def __init__(
        self,
        env: _MlaEnv,
        request_ids: List[int],
        cached_lens: List[int],
        p: int,
        *,
        q_scaling: float,
        q_lora_rank: int,
        window: int,
        num_contexts: int,
        layer_idx: int,
    ) -> None:
        self.env = env
        self.p = p
        self.q_scaling = q_scaling
        self.q_lora_rank = q_lora_rank
        self.window = window
        self.num_contexts = num_contexts
        self.layer_idx = layer_idx
        self.gen_ids = request_ids[num_contexts:]
        num_gen = len(self.gen_ids)
        for rid in self.gen_ids:
            cache = env.kv_cache_manager.kv_cache_map[rid]
            assert cache.resize(cache.capacity + p)
        seq_lens = [CTX_PROMPT] * num_contexts + [p] * num_gen
        self.gen_kv_lens = [c + p for c in cached_lens[num_contexts:]]
        self.metadata = env.prepare_metadata(request_ids, seq_lens, num_contexts, list(cached_lens))
        self.positions = [kv - p + t for kv in self.gen_kv_lens for t in range(p)]
        self.rows = num_gen * p
        heads = env.num_heads
        self.fused_q = torch.randn(
            self.rows, heads, GEN_HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        # The model's strided q_pe: sliced out of a packed [nope | rope] q.
        q = torch.randn(
            self.rows,
            heads,
            QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.q_pe = q.split([QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], dim=-1)[1]
        self.latent_cache = torch.randn(
            self.rows, GEN_HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        self.fused_q_orig = self.fused_q.clone()
        self.q_pe_orig = self.q_pe.clone()
        self.latent_orig = self.latent_cache.clone()
        self.cu_q_seqlens = torch.full((num_gen + 1,), -1, dtype=torch.int32, device="cuda")
        self.cu_kv_seqlens = torch.full((num_gen + 1,), -1, dtype=torch.int32, device="cuda")
        self.fmha_scheduler_counter = torch.full((1,), 7, dtype=torch.uint32, device="cuda")

    def call(self) -> torch.Tensor:
        env, md = self.env, self.metadata
        out = op(
            self.fused_q,
            self.q_pe,
            self.latent_cache,
            env.rotary_cos_sin,
            self.cu_q_seqlens,
            self.cu_kv_seqlens,
            self.fmha_scheduler_counter,
            None,  # mla_bmm1_scale
            None,  # mla_bmm2_scale
            None,  # quant_q_buffer
            md.kv_lens_cuda_runtime,
            md.kv_lens_runtime,
            md.prompt_lens_cpu_runtime,
            self.num_contexts,
            md.kv_cache_block_offsets,
            env.kv_cache_manager.kv_cache_pool_pointers,
            env.kv_cache_manager.kv_cache_pool_mapping,
            None,  # kv_scale_orig_quant
            None,  # kv_scale_quant_orig
            None,  # kv_cache_scale_orig_quant
            None,  # out_scale
            None,  # block_ids_per_seq
            [None, None],  # helix_tensor_params
            self.p,
            self.layer_idx,
            env.num_heads,
            1,  # num_kv_heads
            GEN_HEAD_SIZE,
            0,  # residual_dim
            env.tokens_per_block,
            self.window,
            1,  # beam_width
            0,  # quant_mode: bf16 latent pool
            self.q_scaling,
            self.q_lora_rank,
            KV_LORA_RANK,
            QK_NOPE_HEAD_DIM,
            QK_ROPE_HEAD_DIM,
            V_HEAD_DIM,
            True,  # rope_append
        )
        torch.cuda.synchronize()
        return out


def _verify(step: _Step, pool_before: torch.Tensor, all_before: Optional[torch.Tensor]) -> None:
    """Roped q tail and appended rows against `op.reference`; inputs, absorbed-q head and every other pool
    slot/page untouched; scheduler buffers against `op.reference_scheduler_buffers`."""
    env = step.env
    table = env.rotary_cos_sin.view(-1, QK_ROPE_HEAD_DIM, 2)
    pos = torch.tensor(step.positions, dtype=torch.long, device="cuda")
    q_rows = torch.cat([step.fused_q_orig[..., :KV_LORA_RANK], step.q_pe_orig], dim=-1)
    want_q = op.reference(
        q_rows,
        table[pos, None, :HALF, 0],
        table[pos, None, :HALF, 1],
        QK_ROPE_HEAD_DIM,
        None,
        torch.bfloat16,
    )
    _assert_bytes_equal(
        step.fused_q[..., :KV_LORA_RANK], want_q[..., :KV_LORA_RANK], "absorbed-q head"
    )
    _assert_bf16_roped(
        step.fused_q[..., KV_LORA_RANK:], want_q[..., KV_LORA_RANK:], "fused_q roped tail"
    )
    for r in range(step.rows):
        cos, sin = env.cos_sin(step.positions[r])
        page, offset = env.slot(step.gen_ids[r // step.p], step.positions[r], step.layer_idx)
        got_k = env.pool_tensor(step.layer_idx)[page, 0, offset, 0]
        want_k = op.reference(step.latent_orig[r], cos, sin, QK_ROPE_HEAD_DIM, None, torch.bfloat16)
        _assert_bytes_equal(got_k[:KV_LORA_RANK], want_k[:KV_LORA_RANK], f"compressed_kv, row {r}")
        _assert_bf16_roped(got_k[KV_LORA_RANK:], want_k[KV_LORA_RANK:], f"k_pe, row {r}")
    _assert_bytes_equal(step.q_pe, step.q_pe_orig, "q_pe")
    _assert_bytes_equal(step.latent_cache, step.latent_orig, "latent_cache")

    pool = env.pool_tensor(step.layer_idx)
    changed = (pool.view(torch.int16) != pool_before.view(torch.int16)).nonzero().cpu()
    got_slots = set(zip(changed[:, 0].tolist(), changed[:, 2].tolist()))
    expected = {
        env.slot(step.gen_ids[r // step.p], step.positions[r], step.layer_idx)
        for r in range(step.rows)
    }
    assert got_slots <= expected, f"op touched pool slots {sorted(got_slots - expected)[:8]}"
    if all_before is not None:
        # Row 0's view spans the whole pool: a call naming another row must move only that row's pages.
        moved = (env.pool_tensor(0).view(torch.int16) != all_before.view(torch.int16)).any(
            dim=(1, 2, 3, 4)
        )
        got_pages = set(moved.nonzero().flatten().tolist())
        want_pages = {env.pool_page(step.layer_idx, page) for page, _ in expected}
        assert got_pages <= want_pages, f"moved pool pages {sorted(got_pages - want_pages)[:8]}"

    want_cu_q, want_cu_kv = op.reference_scheduler_buffers(step.gen_kv_lens, env.num_heads, step.p)
    assert step.cu_q_seqlens.tolist() == want_cu_q.tolist()
    assert step.cu_kv_seqlens.tolist() == want_cu_kv.tolist()
    assert step.fmha_scheduler_counter.item() == 0


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    spec = cell.spec
    p = spec["predicted_tokens_per_seq"]
    layer_idx = spec.get("layer_idx", 0)
    num_contexts = spec.get("num_contexts", 0)
    kv_len = spec.get("kv_len")
    max_seq_len = MAX_SEQ_LEN if kv_len is None else kv_len
    assert spec["quant_mode"] == 0, "the driver builds a bf16 latent pool only"

    torch.manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    env = _MlaEnv(
        max_batch_size=spec["num_seqs"] + num_contexts,
        tokens_per_block=spec["tokens_per_block"],
        num_heads=spec["num_heads"],
        rope_theta=spec["rope_theta"],
        mrope_section=spec["mrope_section"],
        mrope_offsets=spec["mrope_offsets"],
        num_layers=spec.get("num_layers", 1),
        max_seq_len=max_seq_len,
        max_tokens=spec.get("max_tokens", 131072),
    )
    certifying(op)
    try:
        request_ids = list(range(spec["num_seqs"] + num_contexts))
        long_kv = LONG_KV if kv_len is None else kv_len - p
        cached = _cached_lens(spec["tokens_per_block"], p, spec["num_seqs"], long_kv, max_seq_len)
        env.kv_cache_manager.add_dummy_requests(
            request_ids, token_nums=[CTX_PROMPT] * num_contexts + cached
        )
        step = _Step(
            env,
            request_ids,
            [0] * num_contexts + cached,
            p,
            q_scaling=spec["q_scaling"],
            q_lora_rank=spec["q_lora_rank"],
            window=spec["attention_window_size"],
            num_contexts=num_contexts,
            layer_idx=layer_idx,
        )
        pool = env.pool_tensor(layer_idx)
        pool_before = pool.clone()
        all_before = None if layer_idx == 0 else env.pool_tensor(0).clone()
        assert step.call() is step.fused_q
        _verify(step, pool_before, all_before)
        # Re-running a prepared step rewrites the same slots rather than double-appending.
        pool_ref, q_ref = pool.clone(), step.fused_q.clone()
        step.call()
        assert torch.equal(pool.view(torch.int16), pool_ref.view(torch.int16))
        assert torch.equal(step.fused_q, q_ref)
    finally:
        env.shutdown()
