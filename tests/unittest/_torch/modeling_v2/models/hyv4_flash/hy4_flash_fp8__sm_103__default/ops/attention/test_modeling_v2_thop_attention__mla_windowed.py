# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for Hy4-Flash's `mla_windowed` thop_attention variant.

Windowed cells drive a caller-owned paged latent pool (a per-request ring of the window's pages) through a
context call and a decode; sparse and DFly2-block cells drive one generation call whose `sparse_attn_indices`
name pool rows. Each output is gated against `reference` over the operands the kernel reads.
"""

import math
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.cache_manager import (  # noqa: E501
    NO_PICK,
    DraftPicks,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.attention.thop_attention__mla_windowed import (  # noqa: E501
    MLA_CONTEXT,
    MLA_GENERATION,
    QM_FP8_KV_CACHE,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.attention.thop_attention__mla_windowed import (  # noqa: E501
    thop_attention as op,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    BLOCK_WIDTH,
    PAD_SLOT_ID,
    TARGET_TOPK,
    dfly_block_indices,
)

__extra_import_path__ = ["../../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "the mla_windowed entry requires a CUDA device"

NUM_HEADS_SWA = 64
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 256
V_HEAD_DIM = 256
LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
Q_LORA_RANK = 1024
TOKENS_PER_BLOCK = 32
# text_config.sliding_window, the SWA layers' rope base, and the interleaved mRoPE sections (t, h, w).
WINDOW = 512
ROPE_THETA_SWA = 10_000.0
MROPE_SECTION = (12, 10, 10)
MASK_CAUSAL = 1
QUANT_MODE_FP8_KV_CACHE = QM_FP8_KV_CACHE
# PositionEmbeddingType.mrope, as the MLA backend passes it for Hy4.
POSITION_EMBEDDING_TYPE_MROPE = 9
# KvCacheConfig.max_num_tokens default; bounds every position the cells rope at.
MAX_SEQ_LEN = 8192

BLOCK_ROWS = BLOCK_WIDTH
DRAFT_LAYERS = (25, 29, 33, 37, 41)
#: The caller's per-row selection width: `round_up(TARGET_TOPK + BLOCK_ROWS, 128)` = 2176.
INDEX_WIDTH = -(-(TARGET_TOPK + BLOCK_ROWS) // 128) * 128
#: A block request's history: longer than the selection, whole pages.
DRAFT_HISTORY = 2304
#: Sparse decode histories, one longer and one shorter than the selection's width.
SPARSE_HISTORY_LONG = 2560
SPARSE_HISTORY_SHORT = 1024

# bf16 pool: 8 bf16 ulp per element of the row scale and 4 of relative RMS. fp8 pool: atol 2**-3,
# rtol 2**-4, 4 bf16 ulp RMS, with the reference rounding probabilities to e4m3.
BF16_ELEMENT_ULP = 8.0
BF16_RMS_ULP = 4.0
FP8_MLA_ATOL = 2**-3
FP8_MLA_RTOL = 2**-4
ULP_BF16 = 2.0**-8

# The append's roped k_pe half: one ulp of the pool dtype plus a bit-exact-fraction floor.
LATENT_ROPE_ULP_RTOL = 2**-7
LATENT_ROPE_E4M3_ULP_RTOL = 2**-3
LATENT_ROPE_MAX_INEXACT_FRACTION = 1e-3

_CAPABILITY_TO_ARCH = {
    (10, 0): Arch.SM_100,
    (10, 3): Arch.SM_103,
    (10, 7): Arch.SM_107,
}


def _arch_of_this_device() -> Arch:
    cap = torch.cuda.get_device_capability()
    arch = _CAPABILITY_TO_ARCH.get(cap)
    assert arch is not None, f"device capability {cap} has no Arch entry"
    assert arch in op.ARCHS, (
        f"this device is {arch.name}, and the variant certifies {sorted(a.name for a in op.ARCHS)}"
    )
    return arch


def _seed_of(cell) -> int:
    """Stable across processes (str hash is not)."""
    return zlib.crc32(cell.why.encode())


def _mrope_table(
    inv_freq: torch.Tensor,
    plane_positions: torch.Tensor,
    sections: Tuple[int, ...] = MROPE_SECTION,
) -> torch.Tensor:
    """Interleaved three-plane mRoPE table in the op's duplicated `(cos, sin)` layout: the time plane's
    angles with the stride-3 slots of the h/w sections taken from those planes. `plane_positions` is [rows, 3]."""
    angles = plane_positions.to(torch.float32).unsqueeze(-1) * inv_freq  # [P, 3, half]
    freq = angles[:, 0].clone()
    for plane, offset in enumerate((1, 2), start=1):
        length = sections[plane] * 3
        idx = slice(offset, length, 3)
        freq[:, idx] = angles[:, plane, idx]
    pairs = torch.stack((freq.cos(), freq.sin()), dim=-1)  # [P, half, 2]
    return torch.cat((pairs, pairs), dim=1).reshape(1, -1).contiguous()


def _rope_const(theta: float, max_positions: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The engine's frequency vector and plain duplicated table; rebuilds `inv_freq` if the backend cached it."""
    rope = RopeParams(
        dim=QK_ROPE_HEAD_DIM, theta=theta, max_positions=max_positions, duplicate_data=True
    )
    inv_freq, plain = rope.create_rope_const_params()
    if inv_freq is None:
        inv_freq = torch.tensor(
            1.0
            / (theta ** (np.arange(0, QK_ROPE_HEAD_DIM, 2) / QK_ROPE_HEAD_DIM)).astype(np.float32),
            device="cuda",
        )
    return inv_freq, plain


def _rope_ref(x: torch.Tensor, table: torch.Tensor, position: int) -> torch.Tensor:
    """GPT-J interleaved rotation of the last dim (the table's first `half` pairs), fp32 math, input dtype out."""
    half = QK_ROPE_HEAD_DIM // 2
    view = table.view(-1, QK_ROPE_HEAD_DIM, 2)
    cos = view[position, :half, 0]
    sin = view[position, :half, 1]
    pairs = x.float().reshape(*x.shape[:-1], half, 2)
    out = torch.empty_like(pairs)
    out[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
    out[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
    return out.reshape(x.shape).to(x.dtype)


def _random_context_inputs(
    num_tokens: int, num_heads: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MLA context inputs: q, k (rope slice garbage; the op fills it from latent_cache), v as a split view of a
    packed kv_b_proj-style buffer, latent [ckv | k_pe]."""
    q = torch.randn(num_tokens, num_heads * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(
        num_tokens,
        num_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_nope, v = kv.split([num_heads * QK_NOPE_HEAD_DIM, num_heads * V_HEAD_DIM], dim=-1)
    k = torch.empty_like(q).view(num_tokens, num_heads, QK_HEAD_DIM)
    k[..., :QK_NOPE_HEAD_DIM] = k_nope.view(num_tokens, num_heads, QK_NOPE_HEAD_DIM)
    k[..., QK_NOPE_HEAD_DIM:] = torch.randn(
        num_tokens, num_heads, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    k = k.view(num_tokens, num_heads * QK_HEAD_DIM)
    latent = torch.randn(num_tokens, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    return q, k, v, latent


def _random_explicit_kv(num_tokens: int, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit K and a packed [T, H*(nope+v)] buffer for the latent_cache=None (chunked) context path."""
    k = torch.randn(num_tokens, num_heads * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    packed_kv = torch.randn(
        num_tokens,
        num_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    return k, packed_kv


def _v_split_view(packed_kv: torch.Tensor, num_heads: int) -> torch.Tensor:
    """V as a split view: the context FMHA hard-codes V's row stride to H*(nope+v_head_dim)."""
    return packed_kv.split([num_heads * QK_NOPE_HEAD_DIM, num_heads * V_HEAD_DIM], dim=-1)[1]


def _sinks(num_heads: int, seed: int) -> torch.Tensor:
    """Per-head fp32 sinks in -2..2, around the checkpoint's 0.0 initialisation."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.rand(num_heads, generator=generator, dtype=torch.float32, device="cuda") * 4.0 - 2.0
    )


class _Hy4MlaEnv:
    """A caller-owned paged latent pool where each request owns `ring_pages` pages as a ring (absolute page
    `j` at `base + j % ring_pages`), so a history longer than the ring overwrites aged-out slots. Every
    block-offset entry carries the ring mapping. Latent rows are mirrored per request for the references.
    """

    def __init__(
        self,
        num_heads: int,
        *,
        max_batch: int,
        max_history: int,
        pool_dtype: torch.dtype = torch.bfloat16,
        quant_mode: int = 0,
        q_scaling: float = 1.0,
        use_paged_context_fmha: bool = False,
        window: int = WINDOW,
        ring_tokens: int = WINDOW,
        head_size: int = LATENT_DIM,
    ) -> None:
        self.num_heads = num_heads
        self.max_batch = max_batch
        self.window = window
        # `head_size` only tells a generation MLA call (576) from a context one; K/V widths come elsewhere.
        self.head_size = head_size
        self.tokens_per_block = TOKENS_PER_BLOCK
        self.quant_mode = quant_mode
        self.q_scaling = q_scaling
        self.q_lora_rank = Q_LORA_RANK
        self.use_paged_context_fmha = use_paged_context_fmha
        self.max_seq_len = MAX_SEQ_LEN
        self.max_blocks_per_seq = -(-max_history // TOKENS_PER_BLOCK)
        self.ring_pages = ring_tokens // TOKENS_PER_BLOCK
        self.softmax_scale = 1.0 / (q_scaling * math.sqrt(QK_HEAD_DIM))
        self.sinks: Optional[torch.Tensor] = None

        self.pool = torch.zeros(
            max_batch * self.ring_pages,
            TOKENS_PER_BLOCK,
            LATENT_DIM,
            dtype=pool_dtype,
            device="cuda",
        )
        self.pool_pointers = torch.zeros(1, 2, dtype=torch.int64, device="cpu")
        self.pool_pointers[0, 0] = self.pool.data_ptr()
        self.pool_mapping = torch.zeros(1, 2, dtype=torch.int32, device="cpu")
        self.block_offsets = torch.zeros(
            1, max_batch, 2, self.max_blocks_per_seq, dtype=torch.int32, device="cuda"
        )
        self.workspace = torch.empty(0, dtype=torch.int8, device="cuda")

        # The interleaved-mRoPE table at text-only positions, passed as `rotary_cos_sin`.
        self.rotary_inv_freq, self.rotary_cos_sin = self._mrope_const_params()

        self.base: Dict[int, int] = {}
        self.latent_rows: Dict[int, List[torch.Tensor]] = {}

    def _mrope_const_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq, _plain = _rope_const(ROPE_THETA_SWA, MAX_SEQ_LEN)
        positions = torch.arange(MAX_SEQ_LEN, device="cuda").unsqueeze(-1).repeat(1, 3)
        return inv_freq, _mrope_table(inv_freq, positions)

    def add_request(self, request_id: int) -> None:
        self.base[request_id] = request_id * self.ring_pages
        self.latent_rows[request_id] = []

    def cached_len(self, request_id: int) -> int:
        return sum(t.shape[0] for t in self.latent_rows[request_id])

    @property
    def fp8_pool(self) -> bool:
        return self.pool.dtype == torch.float8_e4m3fn

    def quantize_e4m3(self, x: torch.Tensor) -> torch.Tensor:
        """e4m3(x) at kv scale 1.0: the op's write-side quantization and the decode query's."""
        return x.float().to(torch.float8_e4m3fn)

    def to_pool(self, rows: torch.Tensor) -> torch.Tensor:
        return self.quantize_e4m3(rows) if self.fp8_pool else rows

    def rope_ref(self, x: torch.Tensor, position: int) -> torch.Tensor:
        return _rope_ref(x, self.rotary_cos_sin, position)

    def rope_latent_rows(self, rows: torch.Tensor, first_position: int) -> torch.Tensor:
        """[ckv | rope(k_pe)] at each row's absolute position, as the append kernel lands them."""
        out = rows.clone()
        count = rows.shape[0]
        half = QK_ROPE_HEAD_DIM // 2
        positions = torch.arange(first_position, first_position + count, device=rows.device)
        view = self.rotary_cos_sin.view(-1, QK_ROPE_HEAD_DIM, 2)
        cos = view[positions, :half, 0]
        sin = view[positions, :half, 1]
        pairs = out[:, KV_LORA_RANK:].float().reshape(count, half, 2)
        rotated = torch.empty_like(pairs)
        rotated[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        rotated[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        out[:, KV_LORA_RANK:] = rotated.reshape(count, QK_ROPE_HEAD_DIM).to(rows.dtype)
        return out

    def write_latent_rows(self, request_id: int, rows: torch.Tensor) -> None:
        """Write rows into the request's ring (pool form) and mirror them: the caller-side append."""
        start = self.cached_len(request_id)
        pooled = self.to_pool(rows)
        tpb = self.tokens_per_block
        pos = torch.arange(start, start + rows.shape[0], device=rows.device)
        pages = self.base[request_id] + (pos // tpb) % self.ring_pages
        self.pool[pages, pos % tpb] = pooled
        self.latent_rows[request_id].append(rows)

    def rows_of(self, request_id: int, positions: torch.Tensor) -> torch.Tensor:
        """Flat pool row ids (`page * tokens_per_block + offset`) of the request's logical positions."""
        tpb = self.tokens_per_block
        pos = positions.long()
        page = self.base[request_id] + (pos // tpb) % self.ring_pages
        return (page * tpb + pos % tpb).to(torch.int32)

    def rows_at(self, row_ids: torch.Tensor) -> torch.Tensor:
        """The pool rows a selection's flat row ids name."""
        return self.pool.view(-1, LATENT_DIM)[row_ids.long()]

    def append_latent_row(self, request_id: int, row: torch.Tensor) -> None:
        """One decode token's latent row, k_pe already roped (as `mla_rope_generation` lands it)."""
        self.write_latent_rows(request_id, row.unsqueeze(0))

    def gather_ring(self, request_id: int) -> torch.Tensor:
        """The request's last min(history, ring) rows in logical order."""
        tpb = self.tokens_per_block
        total = self.cached_len(request_id)
        start = max(0, total - self.ring_pages * tpb)
        pos = torch.arange(start, total, device=self.pool.device)
        page = self.base[request_id] + (pos // tpb) % self.ring_pages
        return self.pool[page, pos % tpb]

    def check_ring(self, request_ids: List[int]) -> None:
        """The ring holds exactly the last rows: ckv bitwise, roped k_pe within one ulp plus a bit-exact floor."""
        tpb = self.tokens_per_block
        for rid in request_ids:
            expected = torch.cat(self.latent_rows[rid])
            start = max(0, expected.shape[0] - self.ring_pages * tpb)
            expected = self.to_pool(expected[start:])
            got = self.gather_ring(rid)
            ckv, exp_ckv = got[:, :KV_LORA_RANK], expected[:, :KV_LORA_RANK]
            if self.fp8_pool:
                same = torch.equal(ckv.view(torch.uint8), exp_ckv.view(torch.uint8))
            else:
                same = torch.equal(ckv, exp_ckv)
            assert same, f"ring compressed_kv mismatch: request {rid}, {got.shape[0]} rows"
            a = got[:, KV_LORA_RANK:].float()
            b = expected[:, KV_LORA_RANK:].float()
            torch.testing.assert_close(
                a,
                b,
                rtol=LATENT_ROPE_E4M3_ULP_RTOL if self.fp8_pool else LATENT_ROPE_ULP_RTOL,
                atol=0.0,
                msg=lambda m: f"ring rope half, request {rid}\n{m}",
            )
            inexact = int(
                (
                    got[:, KV_LORA_RANK:].view(torch.uint8)
                    != expected[:, KV_LORA_RANK:].view(torch.uint8)
                ).sum()
            )
            allowed = max(1, int(LATENT_ROPE_MAX_INEXACT_FRACTION * a.numel()))
            assert inexact <= allowed, (
                f"ring rope half not bit-exact enough: request {rid}, "
                f"{inexact}/{a.numel()} elements differ (allowed {allowed})"
            )

    def check_unwritten_pool_zero(self, request_ids: List[int]) -> None:
        """Every page outside the requests' rings is still all-zero (pins the slab geometry)."""
        used = set()
        for rid in request_ids:
            used.update(range(self.base[rid], self.base[rid] + self.ring_pages))
        rest = [p for p in range(self.pool.shape[0]) if p not in used]
        assert bool((self.pool[rest].float() == 0).all()), (
            "the append wrote outside the requests' pages"
        )

    def fp8_decode_buffers(
        self, fused_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """quant_q_buffer = e4m3(fused_q), mla_bmm1_scale = [x, x * log2(e)] with x = softmax_scale, and
        mla_bmm2_scale = [1.0], as the generation prologue fills them at kv scale 1.0."""
        num_gen = fused_q.shape[0]
        quant_q = self.quantize_e4m3(fused_q.view(num_gen, self.num_heads, LATENT_DIM)).contiguous()
        bmm1 = self.softmax_scale
        return (
            quant_q,
            torch.tensor([bmm1, bmm1 * math.log2(math.e)], dtype=torch.float32, device="cuda"),
            torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        )

    def _fill_offsets(self, request_ids: List[int], totals: Optional[List[int]] = None) -> None:
        # kv_factor-1 pool: the same block id in the K and V rows; every entry carries the ring mapping.
        for s, rid in enumerate(request_ids):
            total = self.cached_len(rid) if totals is None else totals[s]
            n_pages = -(-total // self.tokens_per_block)
            assert n_pages <= self.max_blocks_per_seq, (n_pages, self.max_blocks_per_seq)
            row = self.block_offsets[0, s]
            for j in range(self.max_blocks_per_seq):
                row[0, j] = self.base[rid] + (j % self.ring_pages)
                row[1, j] = row[0, j]

    def _call_op(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        output: torch.Tensor,
        kv_lens: List[int],
        ctx_lens: List[int],
        num_contexts: int,
        num_ctx_tokens: int,
        attention_input_type: int,
        head_size: int,
        num_kv_heads: int,
        v_head_dim: int,
        latent_cache: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        cu_q_seqlens: Optional[torch.Tensor],
        cu_kv_seqlens: Optional[torch.Tensor],
        fmha_scheduler_counter: Optional[torch.Tensor],
        mla_bmm1_scale: Optional[torch.Tensor] = None,
        mla_bmm2_scale: Optional[torch.Tensor] = None,
        quant_q_buffer: Optional[torch.Tensor] = None,
        predicted_tokens_per_seq: int = 1,
        sparse_attn_indices: Optional[torch.Tensor] = None,
        num_sparse_topk: Optional[int] = None,
        attention_window_size: Optional[int] = None,
        kv_bound: Optional[int] = None,
    ) -> None:
        ns = len(kv_lens)
        req_types = [0 if i < num_contexts else 1 for i in range(ns)]
        host_kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
        if kv_bound is not None:
            # On a sparse layer the core hands the selection's width as the host KV length.
            host_kv_lens = torch.full_like(host_kv_lens, kv_bound)
        with validating(op):
            op(
                q=q,
                k=k,
                v=v,
                output=output,
                output_sf=None,
                workspace=self.workspace,
                sequence_length=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
                host_past_key_value_lengths=host_kv_lens,
                host_total_kv_lens=torch.tensor(
                    [sum(kv_lens[:num_contexts]), sum(kv_lens[num_contexts:])],
                    dtype=torch.int32,
                ),
                context_lengths=torch.tensor(ctx_lens, dtype=torch.int32, device="cuda"),
                host_context_lengths=torch.tensor(ctx_lens, dtype=torch.int32),
                host_request_types=torch.tensor(req_types, dtype=torch.int32),
                max_context_q_len_override=None,
                kv_cache_block_offsets=self.block_offsets,
                host_kv_cache_pool_pointers=self.pool_pointers,
                host_kv_cache_pool_mapping=self.pool_mapping,
                cache_indirection=None,
                kv_scale_orig_quant=None,
                kv_scale_quant_orig=None,
                out_scale=None,
                rotary_inv_freq=self.rotary_inv_freq,
                rotary_cos_sin=self.rotary_cos_sin,
                latent_cache=latent_cache,
                q_pe=q_pe,
                block_ids_per_seq=None,
                attention_sinks=self.sinks,
                is_fused_qkv=k is None,
                update_kv_cache=True,
                predicted_tokens_per_seq=predicted_tokens_per_seq,
                local_layer_idx=0,
                num_heads=self.num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                tokens_per_block=self.tokens_per_block,
                max_num_requests=self.max_batch,
                max_context_length=self.max_seq_len,
                max_seq_len=self.max_seq_len,
                attention_window_size=self.window
                if attention_window_size is None
                else attention_window_size,
                beam_width=1,
                mask_type=MASK_CAUSAL,
                quant_mode=self.quant_mode,
                q_scaling=self.q_scaling,
                position_embedding_type=POSITION_EMBEDDING_TYPE_MROPE,
                rope_dim=QK_ROPE_HEAD_DIM,
                rope_base=ROPE_THETA_SWA,
                rope_scale_type=0,
                rope_scale=1.0,
                rope_short_m_scale=1.0,
                rope_long_m_scale=1.0,
                rope_max_positions=MAX_SEQ_LEN,
                rope_original_max_positions=MAX_SEQ_LEN,
                use_paged_context_fmha=self.use_paged_context_fmha,
                attention_input_type=attention_input_type,
                is_mla_enable=True,
                chunked_prefill_buffer_batch_size=1,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=KV_LORA_RANK,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                v_head_dim=v_head_dim,
                rope_append=True,
                mrope_rotary_cos_sin=None,
                mrope_position_deltas=None,
                helix_position_offsets=None,
                helix_is_inactive_rank=None,
                attention_chunk_size=None,
                softmax_stats_tensor=None,
                is_spec_decoding_enabled=False,
                use_spec_decoding=False,
                is_spec_dec_tree=False,
                spec_decoding_generation_lengths=None,
                spec_decoding_position_offsets=None,
                spec_decoding_packed_mask=None,
                spec_decoding_bl_tree_mask_offset=None,
                spec_decoding_bl_tree_mask=None,
                spec_bl_tree_first_sparse_mask_offset_kv=None,
                sparse_kv_indices=None,
                sparse_kv_offsets=None,
                sparse_attn_indices=sparse_attn_indices,
                sparse_attn_offsets=None,
                sparse_attn_indices_block_size=0,
                num_sparse_topk=num_sparse_topk,
                cu_q_seqlens=cu_q_seqlens,
                cu_kv_seqlens=cu_kv_seqlens,
                fmha_scheduler_counter=fmha_scheduler_counter,
                mla_bmm1_scale=mla_bmm1_scale,
                mla_bmm2_scale=mla_bmm2_scale,
                quant_q_buffer=quant_q_buffer,
                num_contexts=num_contexts,
                num_ctx_tokens=num_ctx_tokens,
            )
        torch.cuda.synchronize()

    def call_context(
        self,
        request_ids: List[int],
        seq_lens: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        latent_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Fresh-sequence context call: the op ropes q/k in place and appends [ckv | rope(k_pe)] to the pool."""
        num_contexts = len(request_ids)
        num_ctx_tokens = sum(seq_lens)
        self._fill_offsets(request_ids, totals=list(seq_lens))
        output = torch.empty(
            num_ctx_tokens, self.num_heads * V_HEAD_DIM, dtype=q.dtype, device=q.device
        )
        self._call_op(
            q=q,
            k=k,
            v=v,
            output=output,
            kv_lens=list(seq_lens),
            ctx_lens=list(seq_lens),
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            attention_input_type=MLA_CONTEXT,
            head_size=self.head_size,
            num_kv_heads=self.num_heads,  # context runs as MHA
            v_head_dim=V_HEAD_DIM,
            latent_cache=latent_cache,
            q_pe=None,
            cu_q_seqlens=None,
            cu_kv_seqlens=None,
            fmha_scheduler_counter=None,
        )

        start = 0
        for rid, ln in zip(request_ids, seq_lens):
            rows = latent_cache[start : start + ln].clone()
            for i in range(ln):
                rows[i, KV_LORA_RANK:] = self.rope_ref(rows[i, KV_LORA_RANK:], i)
            self.latent_rows[rid].append(rows)
            start += ln
        return output

    def call_context_chunk(
        self,
        request_ids: List[int],
        new_lens: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Context call with latent_cache=None (chunked prefill over a cached prefix): no RoPE, no append."""
        num_contexts = len(request_ids)
        num_ctx_tokens = sum(new_lens)
        totals = [self.cached_len(rid) for rid in request_ids]
        self._fill_offsets(request_ids)
        output = torch.empty(
            num_ctx_tokens, self.num_heads * V_HEAD_DIM, dtype=q.dtype, device=q.device
        )
        self._call_op(
            q=q,
            k=k,
            v=v,
            output=output,
            kv_lens=totals,
            ctx_lens=list(new_lens),
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            attention_input_type=MLA_CONTEXT,
            head_size=self.head_size,
            num_kv_heads=self.num_heads,  # context runs as MHA
            v_head_dim=V_HEAD_DIM,
            latent_cache=None,  # selects the no-RoPE / no-append context path
            q_pe=None,
            cu_q_seqlens=None,
            cu_kv_seqlens=None,
            fmha_scheduler_counter=None,
        )
        return output

    def call_generation(self, request_ids: List[int], fused_q: torch.Tensor) -> torch.Tensor:
        """Latent MQA decode over the pool; latent_cache/q_pe are presence-checked only, so garbage."""
        rows = len(request_ids)
        totals = [self.cached_len(rid) for rid in request_ids]
        self._fill_offsets(request_ids)
        cu_q = torch.arange(rows + 1, dtype=torch.int32) * self.num_heads
        cu_kv = torch.zeros(rows + 1, dtype=torch.int32)
        cu_kv[1:] = torch.tensor(totals, dtype=torch.int32).cumsum(0)
        counter = torch.zeros(1, dtype=torch.uint32, device="cuda")
        garbage_latent = torch.randn(rows, LATENT_DIM, dtype=fused_q.dtype, device="cuda")
        garbage_q_pe = torch.randn(
            rows, self.num_heads, QK_ROPE_HEAD_DIM, dtype=fused_q.dtype, device="cuda"
        )
        output = torch.empty(
            rows, self.num_heads * KV_LORA_RANK, dtype=fused_q.dtype, device="cuda"
        )
        fp8_buffers = self.fp8_decode_buffers(fused_q) if self.fp8_pool else (None, None, None)
        quant_q, bmm1, bmm2 = fp8_buffers
        self._call_op(
            q=fused_q,
            k=None,
            v=None,
            output=output,
            kv_lens=totals,
            ctx_lens=totals,
            num_contexts=0,
            num_ctx_tokens=0,
            attention_input_type=MLA_GENERATION,
            head_size=LATENT_DIM,
            num_kv_heads=1,  # latent MQA
            v_head_dim=KV_LORA_RANK,
            latent_cache=garbage_latent,
            q_pe=garbage_q_pe,
            cu_q_seqlens=cu_q.cuda(),
            cu_kv_seqlens=cu_kv.cuda(),
            fmha_scheduler_counter=counter,
            mla_bmm1_scale=bmm1,
            mla_bmm2_scale=bmm2,
            quant_q_buffer=quant_q,
        )
        return output

    def call_block_generation(
        self,
        request_ids: List[int],
        fused_q: torch.Tensor,
        selection: torch.Tensor,
        lengths: torch.Tensor,
        width: int,
        *,
        num_contexts: int = 0,
        ctx_lens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """The DFly2 drafter's block call: `width` rows per request, each reading its own `selection` row.

        `num_sparse_topk` and the host KV length are the selection's width; each generating request's device
        length is the writer's count through its block's last row (the kernel uses `bound - width + 1 + j`).
        `num_contexts` context sequences of `ctx_lens` may lead the batch.
        """
        gens = len(request_ids)
        rows = gens * width
        writer = [int(lengths[g * width + width - 1]) for g in range(gens)]
        assert torch.equal(
            torch.tensor(writer, dtype=torch.int32, device=selection.device),
            (selection[width - 1 :: width] >= 0).sum(dim=1, dtype=torch.int32),
        ), "the bound is not the block's last row's live count"
        totals = writer
        context = [int(v) for v in (ctx_lens or [])]
        assert len(context) == num_contexts, (len(context), num_contexts)
        batch_ids = list(range(num_contexts)) + list(request_ids)
        kv_lens = context + totals
        if num_contexts:
            self._fill_offsets(batch_ids, totals=kv_lens)
        else:
            self._fill_offsets(request_ids)
        # Scheduler buffers are per generation sequence: cu_q in q rows (num_heads * P), cu_kv in tokens.
        cu_q = torch.arange(gens + 1, dtype=torch.int32) * (self.num_heads * width)
        cu_kv = torch.zeros(gens + 1, dtype=torch.int32)
        cu_kv[1:] = torch.tensor(totals, dtype=torch.int32).cumsum(0)
        counter = torch.zeros(1, dtype=torch.uint32, device="cuda")
        garbage_latent = torch.randn(rows, LATENT_DIM, dtype=fused_q.dtype, device="cuda")
        garbage_q_pe = torch.randn(
            rows, self.num_heads, QK_ROPE_HEAD_DIM, dtype=fused_q.dtype, device="cuda"
        )
        output = torch.empty(
            rows, self.num_heads * KV_LORA_RANK, dtype=fused_q.dtype, device="cuda"
        )
        self._call_op(
            q=fused_q,
            k=None,
            v=None,
            output=output,
            kv_lens=kv_lens,
            ctx_lens=kv_lens,
            num_contexts=num_contexts,
            num_ctx_tokens=0,
            attention_input_type=MLA_GENERATION,
            head_size=LATENT_DIM,
            num_kv_heads=1,  # latent MQA
            v_head_dim=KV_LORA_RANK,
            latent_cache=garbage_latent,
            q_pe=garbage_q_pe,
            cu_q_seqlens=cu_q.cuda(),
            cu_kv_seqlens=cu_kv.cuda(),
            fmha_scheduler_counter=counter,
            predicted_tokens_per_seq=width,
            sparse_attn_indices=selection,
            num_sparse_topk=selection.shape[1],
            attention_window_size=MAX_SEQ_LEN,
            kv_bound=selection.shape[1],
        )
        return output

    def call_sparse_generation(
        self,
        request_ids: List[int],
        fused_q: torch.Tensor,
        selection: torch.Tensor,
        kv_lens: List[int],
    ) -> torch.Tensor:
        """The sparse layers' own decode: one row per request reading its `sparse_attn_indices` row.

        Host KV bound = selection width; the device lengths stay the requests' own, so a short history reads
        exactly its live prefix.
        """
        rows = len(request_ids)
        self._fill_offsets(request_ids)
        # Scheduler buffers are per sequence: cu_q in q rows (num_heads per row), cu_kv in tokens.
        cu_q = torch.arange(rows + 1, dtype=torch.int32) * self.num_heads
        cu_kv = torch.zeros(rows + 1, dtype=torch.int32)
        cu_kv[1:] = torch.tensor(kv_lens, dtype=torch.int32).cumsum(0)
        counter = torch.zeros(1, dtype=torch.uint32, device="cuda")
        garbage_latent = torch.randn(rows, LATENT_DIM, dtype=fused_q.dtype, device="cuda")
        garbage_q_pe = torch.randn(
            rows, self.num_heads, QK_ROPE_HEAD_DIM, dtype=fused_q.dtype, device="cuda"
        )
        output = torch.empty(
            rows, self.num_heads * KV_LORA_RANK, dtype=fused_q.dtype, device="cuda"
        )
        self._call_op(
            q=fused_q,
            k=None,
            v=None,
            output=output,
            kv_lens=list(kv_lens),
            ctx_lens=list(kv_lens),
            num_contexts=0,
            num_ctx_tokens=0,
            attention_input_type=MLA_GENERATION,
            head_size=LATENT_DIM,
            num_kv_heads=1,  # latent MQA
            v_head_dim=KV_LORA_RANK,
            latent_cache=garbage_latent,
            q_pe=garbage_q_pe,
            cu_q_seqlens=cu_q.cuda(),
            cu_kv_seqlens=cu_kv.cuda(),
            fmha_scheduler_counter=counter,
            sparse_attn_indices=selection,
            num_sparse_topk=selection.shape[1],
            attention_window_size=MAX_SEQ_LEN,
            kv_bound=selection.shape[1],
        )
        return output


def _append_operands(
    env: _Hy4MlaEnv,
    q: torch.Tensor,
    k: torch.Tensor,
    latent: torch.Tensor,
    q_lens: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q/k as the context FMHA reads them after the in-place rope, rebuilt from the pre-call tensors: q's rope
    slice and latent_cache's k_pe rotated at each token's position."""
    q_out, k_out = q.clone(), k.clone()
    q_view = q.view(-1, env.num_heads, QK_HEAD_DIM)
    qo = q_out.view(-1, env.num_heads, QK_HEAD_DIM)
    ko = k_out.view(-1, env.num_heads, QK_HEAD_DIM)
    start = 0
    for ln in q_lens:
        for i in range(ln):
            t = start + i
            qo[t, :, QK_NOPE_HEAD_DIM:] = env.rope_ref(q_view[t, :, QK_NOPE_HEAD_DIM:], i)
            ko[t, :, QK_NOPE_HEAD_DIM:] = env.rope_ref(latent[t, KV_LORA_RANK:], i)
        start += ln
    return q_out, k_out


def _context_reference(
    env: _Hy4MlaEnv,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: List[int],
    kv_lens: List[int],
) -> torch.Tensor:
    return op.reference(
        q,
        k,
        v,
        q_lens,
        kv_lens,
        num_heads=env.num_heads,
        num_kv_heads=env.num_heads,  # context runs as MHA
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        softmax_scale=env.softmax_scale,
        causal=True,
        attention_sinks=env.sinks,
        window=env.window,
        e4m3_inputs=env.fp8_pool,
        e4m3_probabilities=env.fp8_pool,
    )


def _generation_reference(
    env: _Hy4MlaEnv, request_ids: List[int], fused_q: torch.Tensor
) -> torch.Tensor:
    """Latent MQA over each request's rows (K = [ckv | roped k_pe], V = K[:, :512]) under the window."""
    k_all = torch.cat([torch.cat(env.latent_rows[rid]) for rid in request_ids])
    v_all = k_all[:, :KV_LORA_RANK].contiguous()
    return op.reference(
        fused_q,
        k_all,
        v_all,
        [1] * len(request_ids),
        [env.cached_len(rid) for rid in request_ids],
        num_heads=env.num_heads,
        num_kv_heads=1,
        qk_head_dim=LATENT_DIM,
        v_head_dim=KV_LORA_RANK,
        softmax_scale=env.softmax_scale,
        causal=True,
        attention_sinks=env.sinks,
        window=env.window,
        e4m3_inputs=env.fp8_pool,
        e4m3_probabilities=env.fp8_pool,
    )


def _publish_picks(
    env: _Hy4MlaEnv,
    request_ids: List[int],
    counts: List[int],
    history: int = DRAFT_HISTORY,
) -> DraftPicks:
    """The core's picks state, written through `DraftPicks.write`: per draft layer, `counts[g]` random rows of
    the request's `history` as pool rows, `NO_PICK` past them."""
    gens, device = len(request_ids), env.pool.device
    picks = DraftPicks(DRAFT_LAYERS, gens, TARGET_TOPK, device)
    rows = torch.arange(gens, device=device)
    for layer in DRAFT_LAYERS:
        packed = torch.full((gens, TARGET_TOPK), NO_PICK, dtype=torch.int32, device=device)
        for g, rid in enumerate(request_ids):
            chosen = torch.randperm(history, device=device)[: counts[g]]
            packed[g, : counts[g]] = env.rows_of(rid, chosen)
        picks.write(layer, packed, rows)
    return picks


def _block_indices(
    env: _Hy4MlaEnv,
    picks: DraftPicks,
    request_ids: List[int],
    state_rows: torch.Tensor,
    history: int = DRAFT_HISTORY,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The k29 entry's selection `[layers, rows, width]` and `lengths`: the anchor's compacted picks, then the
    block's causal prefix over its rows at `history ..`, `-1` past it."""
    gens, width, device = len(request_ids), BLOCK_ROWS, env.pool.device
    first = torch.arange(history, history + width, device=device)
    block_slots = torch.stack([env.rows_of(rid, first) for rid in request_ids])
    active = torch.ones(gens, dtype=torch.bool, device=device)
    packed = torch.full(
        (len(DRAFT_LAYERS), gens * width, INDEX_WIDTH),
        PAD_SLOT_ID,
        dtype=torch.int32,
        device=device,
    )
    lengths = torch.zeros((len(DRAFT_LAYERS), gens * width), dtype=torch.int32, device=device)
    dfly_block_indices(picks.history, state_rows, block_slots, active, packed, lengths)
    return packed, lengths


def _sparse_selection(env: _Hy4MlaEnv, request_id: int, total: int) -> torch.Tensor:
    """One decode row's selection as the core writes it: the row's position and the `TARGET_TOPK - 1` before
    it, descending, resolved to pool rows; `-1` before the document start."""
    pos = total - 1 - torch.arange(TARGET_TOPK, dtype=torch.int64, device=env.pool.device)
    rows = env.rows_of(request_id, pos.clamp(min=0))
    rows[pos < 0] = NO_PICK
    return rows


def _sparse_decode_reference(
    env: _Hy4MlaEnv,
    selection: torch.Tensor,
    kv_lens: List[int],
    fused_q: torch.Tensor,
) -> torch.Tensor:
    """Each row attends its selection's live prefix (the pool rows it names); no window."""
    keys = [env.rows_at(sel[sel >= 0]) for sel in selection]
    k = torch.cat(keys)
    v = k[:, :KV_LORA_RANK].contiguous()
    return op.reference(
        fused_q,
        k,
        v,
        [1] * len(kv_lens),
        list(kv_lens),
        num_heads=env.num_heads,
        num_kv_heads=1,
        qk_head_dim=LATENT_DIM,
        v_head_dim=KV_LORA_RANK,
        softmax_scale=env.softmax_scale,
        causal=True,
        attention_sinks=env.sinks,
        window=None,
        e4m3_inputs=env.fp8_pool,
        e4m3_probabilities=env.fp8_pool,
    )


def _block_reference(
    env: _Hy4MlaEnv,
    selection: torch.Tensor,
    counts: List[int],
    fused_q: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Each request's keys are its last block row's live prefix (`counts + width` rows); bottom-right causal
    gives row `t` its `counts + t + 1`. No window."""
    gens = len(counts)
    rows = gens * width
    q = fused_q.view(rows, env.num_heads, LATENT_DIM)
    keys = []
    for g in range(gens):
        last = selection[g * width + width - 1]
        keys.append(env.rows_at(last[: counts[g] + width]))
    k = torch.cat(keys)
    v = k[:, :KV_LORA_RANK].contiguous()
    return op.reference(
        q.reshape(rows, env.num_heads * LATENT_DIM),
        k,
        v,
        [width] * gens,
        [counts[g] + width for g in range(gens)],
        num_heads=env.num_heads,
        num_kv_heads=1,
        qk_head_dim=LATENT_DIM,
        v_head_dim=KV_LORA_RANK,
        softmax_scale=env.softmax_scale,
        causal=True,
        attention_sinks=env.sinks,
        window=None,
        e4m3_inputs=env.fp8_pool,
        e4m3_probabilities=env.fp8_pool,
    )


def _compare(out: torch.Tensor, ref: torch.Tensor, fp8: bool) -> None:
    """The bf16 or fp8 band (see the constants above)."""
    if fp8:
        torch.testing.assert_close(out, ref, rtol=FP8_MLA_RTOL, atol=FP8_MLA_ATOL)
        err = (out.float() - ref.float()).pow(2).mean().sqrt()
        rel_rms = (err / ref.float().pow(2).mean().sqrt().clamp_min(1e-9)).item()
        assert rel_rms <= BF16_RMS_ULP * ULP_BF16, (
            f"relative RMS {rel_rms:.3e} exceeds {BF16_RMS_ULP} bf16 ulp"
        )
    else:
        assert_within_ulp(out, ref, element_ulp=BF16_ELEMENT_ULP, rms_ulp=BF16_RMS_ULP)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


def _run_cell(cell) -> None:
    """Windowed cells: context then decode. `block` cells: the sparse decode (P = 1) or the DFly2 block."""
    if cell.spec.get("block"):
        if cell.spec["predicted_tokens_per_seq"] == 1:
            _run_sparse_cell(cell)
        else:
            _run_block_cell(cell)
        return
    spec = cell.spec
    seed = _seed_of(cell)
    torch.manual_seed(seed)
    heads = spec["num_heads"]
    # The harness hard-codes these.
    assert spec["num_kv_heads"] == heads and spec["tokens_per_block"] == TOKENS_PER_BLOCK
    assert spec["window"] == WINDOW and spec["v_head_dim"] == V_HEAD_DIM
    quant_mode = spec["quant_mode"]
    assert quant_mode in (0, QUANT_MODE_FP8_KV_CACHE), quant_mode
    fp8 = quant_mode == QUANT_MODE_FP8_KV_CACHE
    chunked = spec["use_paged_context_fmha"]
    max_batch = spec["max_num_requests"]
    decode = spec["decode"]
    # Context is `prefill` (fresh) or `cached` + `new` (a chunk over a prefix).
    cached = spec.get("cached", 0)
    new = spec.get("new", spec.get("prefill", 0) - cached)
    prefill = cached + new
    assert prefill == spec.get("prefill", prefill), (cached, new, spec.get("prefill"))
    assert (cached == 0) or chunked, "a cached prefix needs the paged-context path"
    history = prefill + decode

    env = _Hy4MlaEnv(
        heads,
        max_batch=max_batch,
        max_history=history,
        # The ring holds the whole prefill so its early queries' rows are not overwritten.
        ring_tokens=max(WINDOW, -(-prefill // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK),
        pool_dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
        quant_mode=quant_mode,
        use_paged_context_fmha=chunked,
        head_size=spec["head_size"],
    )
    env.sinks = _sinks(heads, seed)
    request_ids = list(range(max_batch))
    for rid in request_ids:
        env.add_request(rid)

    # Context phase.
    if chunked:
        rows = torch.randn(max_batch, prefill, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        for rid in request_ids:
            env.write_latent_rows(rid, env.rope_latent_rows(rows[rid], 0))
        k, packed_kv = _random_explicit_kv(max_batch * prefill, heads)
        v = _v_split_view(packed_kv, heads)
        q = torch.randn(max_batch * new, heads * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        out = env.call_context_chunk(request_ids, [new] * max_batch, q, k, v)
        ref = _context_reference(env, q, k, v, [new] * max_batch, [prefill] * max_batch)
    else:
        q, k, v, latent = _random_context_inputs(max_batch * prefill, heads)
        q_pre, k_pre, latent_pre = q.clone(), k.clone(), latent.clone()
        out = env.call_context(request_ids, [prefill] * max_batch, q, k, v, latent)
        env.check_ring(request_ids)
        q_ref, k_ref = _append_operands(env, q_pre, k_pre, latent_pre, [prefill] * max_batch)
        ref = _context_reference(env, q_ref, k_ref, v, [prefill] * max_batch, [prefill] * max_batch)
    _compare(out, ref, fp8)

    # Decode phase: one row at the history's end, then one generation call.
    decode_rows = torch.randn(max_batch, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    for rid in request_ids:
        env.append_latent_row(rid, decode_rows[rid])
    fused_q = torch.randn(max_batch, heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    pool_before = env.pool.clone()
    out_gen = env.call_generation(request_ids, fused_q)
    assert _bitwise_equal(pool_before, env.pool), "the generation call wrote to the pool"
    env.check_ring(request_ids)
    ref_gen = _generation_reference(env, request_ids, fused_q)
    _compare(out_gen, ref_gen, fp8)
    env.check_unwritten_pool_zero(request_ids)


def _run_block_cell(cell) -> None:
    """One DFly2 block call over the sparse layers' bf16 pool, selections built by `DraftPicks` and
    `dfly_block_indices`. A `kv_len` cell puts the block after a `kv_len - width`-row history; `num_contexts`
    context sequences of that length lead the batch."""
    spec = cell.spec
    seed = _seed_of(cell)
    torch.manual_seed(seed)
    heads = spec["num_heads"]
    width = spec["predicted_tokens_per_seq"]
    # The harness hard-codes these.
    assert (
        spec["num_kv_heads"] == 1 and spec["v_head_dim"] == KV_LORA_RANK and spec["quant_mode"] == 0
    )
    assert spec["tokens_per_block"] == TOKENS_PER_BLOCK and width == BLOCK_ROWS
    batch = spec["max_num_requests"]
    num_contexts = int(spec.get("num_contexts", 0))
    gens = batch - num_contexts
    kv_len = spec.get("kv_len")
    history = DRAFT_HISTORY if kv_len is None else kv_len - width

    env = _Hy4MlaEnv(
        heads,
        max_batch=batch,
        max_history=history + width,
        # A sparse layer is bounded by its selection: the ring holds the whole history and the block.
        window=MAX_SEQ_LEN,
        ring_tokens=-(-(history + width) // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK,
        pool_dtype=torch.bfloat16,
        quant_mode=0,
        head_size=spec["head_size"],
    )
    env.sinks = _sinks(heads, seed)
    request_ids = list(range(num_contexts, batch))
    short_history = (
        torch.randn(gens, DRAFT_HISTORY, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        if kv_len is None
        else None
    )
    block = torch.randn(gens, width, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    for g, rid in enumerate(request_ids):
        env.add_request(rid)
        rows = (
            short_history[g]
            if short_history is not None
            else torch.randn(history, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        )
        env.write_latent_rows(rid, env.rope_latent_rows(rows, 0))
        env.write_latent_rows(rid, env.rope_latent_rows(block[g], history))
    for rid in range(num_contexts):
        env.add_request(rid)
    env.check_ring(request_ids)

    # Saturated picks, with shorter lists at the short-history batch's tail.
    counts = [TARGET_TOPK] * gens
    if kv_len is None:
        for g in range(max(0, gens - 4), gens):
            counts[g] = TARGET_TOPK // (2 ** (gens - g))
    picks = _publish_picks(env, request_ids, counts, history=history)
    state_rows = torch.arange(gens, dtype=torch.int64, device="cuda")
    packed, lengths = _block_indices(env, picks, request_ids, state_rows, history=history)
    slot = len(DRAFT_LAYERS) - 1
    selection = packed[slot]
    # Row `j` of request `g` is live for `counts[g] + j + 1` columns; `_block_reference` relies on it.
    want = torch.tensor(
        [counts[g] + j + 1 for g in range(gens) for j in range(width)],
        dtype=torch.int32,
        device="cuda",
    )
    assert torch.equal(lengths[slot], want), "the k29 entry's live counts moved"

    fused_q = torch.randn(gens * width, heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    pool_before = env.pool.clone()
    ctx_lens = [kv_len] * num_contexts if kv_len is not None else None
    out = env.call_block_generation(
        request_ids,
        fused_q,
        selection,
        lengths[slot],
        width,
        num_contexts=num_contexts,
        ctx_lens=ctx_lens,
    )
    assert _bitwise_equal(pool_before, env.pool), "the generation call wrote to the pool"
    env.check_ring(request_ids)
    env.check_unwritten_pool_zero(request_ids)
    ref = _block_reference(env, selection, counts, fused_q, width)
    _compare(out, ref, fp8=False)


def _run_sparse_cell(cell) -> None:
    """The sparse layers' own decode: one row per request over histories longer and shorter than the
    selection's width, gated against `reference` over each row's live prefix."""
    spec = cell.spec
    seed = _seed_of(cell)
    torch.manual_seed(seed)
    heads = spec["num_heads"]
    # The harness hard-codes these.
    assert (
        spec["num_kv_heads"] == 1 and spec["v_head_dim"] == KV_LORA_RANK and spec["quant_mode"] == 0
    )
    assert spec["tokens_per_block"] == TOKENS_PER_BLOCK and spec["predicted_tokens_per_seq"] == 1
    gens = spec["max_num_requests"]
    histories = [SPARSE_HISTORY_LONG if g % 4 != 3 else SPARSE_HISTORY_SHORT for g in range(gens)]

    env = _Hy4MlaEnv(
        heads,
        max_batch=gens,
        max_history=SPARSE_HISTORY_LONG + 1,
        # The ring holds each request's whole history and its decode row: nothing wraps.
        window=MAX_SEQ_LEN,
        ring_tokens=-(-(SPARSE_HISTORY_LONG + 1) // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK,
        pool_dtype=torch.bfloat16,
        quant_mode=0,
        head_size=spec["head_size"],
    )
    env.sinks = _sinks(heads, seed)
    request_ids = list(range(gens))
    for g, rid in enumerate(request_ids):
        env.add_request(rid)
        history = torch.randn(histories[g], LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        env.write_latent_rows(rid, env.rope_latent_rows(history, 0))
        decode_row = torch.randn(1, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        env.write_latent_rows(rid, env.rope_latent_rows(decode_row, histories[g]))
    env.check_ring(request_ids)

    totals = [h + 1 for h in histories]
    selection = torch.stack(
        [_sparse_selection(env, rid, total) for rid, total in zip(request_ids, totals)]
    )
    # A live prefix of the row's own keys, then `-1`; `_sparse_decode_reference` relies on it.
    for g, total in enumerate(totals):
        live = min(total, TARGET_TOPK)
        assert bool((selection[g, :live] >= 0).all()) and bool((selection[g, live:] < 0).all())
    kv_lens = [min(total, TARGET_TOPK) for total in totals]

    fused_q = torch.randn(gens, heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    pool_before = env.pool.clone()
    out = env.call_sparse_generation(request_ids, fused_q, selection, totals)
    assert _bitwise_equal(pool_before, env.pool), "the generation call wrote to the pool"
    env.check_ring(request_ids)
    ref = _sparse_decode_reference(env, selection, kv_lens, fused_q)
    _compare(out, ref, fp8=False)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    """Every cell through its phases over one pool: the append and the masked softmax together."""
    _arch_of_this_device()
    certifying(op)
    _run_cell(cell)


def test_context_accepts_the_qk_width(certifying) -> None:  # noqa: F811
    """The model's own context call passes `head_size` = qk width 256, not the cells' 576."""
    _arch_of_this_device()
    torch.manual_seed(5)
    heads = NUM_HEADS_SWA
    env = _Hy4MlaEnv(heads, max_batch=1, max_history=64, head_size=QK_HEAD_DIM)
    certifying(op)
    env.sinks = _sinks(heads, 5)
    env.add_request(0)
    q, k, v, latent = _random_context_inputs(64, heads)
    q_pre, k_pre, latent_pre = q.clone(), k.clone(), latent.clone()
    out = env.call_context([0], [64], q, k, v, latent)
    q_ref, k_ref = _append_operands(env, q_pre, k_pre, latent_pre, [64])
    ref = _context_reference(env, q_ref, k_ref, v, [64], [64])
    _compare(out, ref, fp8=False)
