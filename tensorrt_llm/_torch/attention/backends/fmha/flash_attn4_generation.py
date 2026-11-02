# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generation-phase attention through FlashAttention-4 (CuTe DSL) on SM90.

Serves the generation rows of a batch, plain decode (one query token per row) and
linear speculative-decoding verification (``q_len = 1 + K`` tokens per row), for dense
GQA/MHA layers with an unquantized bf16/fp16 paged KV cache (KV cache manager V2).
Per layer it

1. applies the op's NeoX RoPE to the packed Q/K heads in place (``rope_qkv_inplace``);
2. writes the step's K/V at ``kv_len - q_len + j`` of every row into the paged pool;
3. runs FA4's paged forward over the whole cache (``seqused_k`` = device KV lengths,
   causal, bottom-right aligned), so the step's tokens see each other causally.

The page table is derived on the device from the attention op's block offsets, and every
launch reads only fixed-address tensors, so the layer can be captured in a CUDA graph.
Draft layers (they run on a draft KV-cache manager) are declined.

Opt in with ``TLLM_FMHA_LIBS=+flash_attn4_generation``.
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Callable, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha
from .sampling_shared_prefix import _fused_rope_supported, _rotary_dim
from .sampling_shared_prefix_kernels import rope_qkv_inplace

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

LIB_NAME = "flash_attn4_generation"

CHECK_ENV = "TLLM_FLASH_ATTN4_GENERATION_CHECK"
"""Set to 1 to compare the device page table with the host block-offset table on eager steps.
Debugging aid: it synchronizes the device and must stay off for measurements."""

FA4_TILE_MN = (64, 128)
FA4_NUM_THREADS = 256
"""FA4 SM90 tile for generation rows. With GQA packing a row holds ``q_len * heads_per_kv_head``
query rows (28 for Qwen2.5-7B at q_len 4), so a 64-row tile wastes less than the default 128-row
one: 193 vs 282 us per layer for 120 rows over 4.5k cached tokens (Qwen2.5-7B shapes, H100)."""


@functools.lru_cache(maxsize=1)
def _fa4_forward() -> Optional[Callable[..., tuple]]:
    """FA4's forward, or None when the CuTe DSL interface cannot be imported."""
    try:
        from tensorrt_llm._torch.cute_dsl_utils import install_cutlass_dsl_compatibility

        install_cutlass_dsl_compatibility()
        from flash_attn.cute.interface import _flash_attn_fwd
    except (ImportError, OSError, AttributeError) as error:
        logger.info(f"{LIB_NAME}: FlashAttention-4 is unavailable ({error}).")
        return None
    return _flash_attn_fwd


@triton.jit
def _write_kv_paged_kernel(
    k_ptr,
    v_ptr,
    kv_row_stride,
    pool_ptr,
    pool_page_stride,
    pool_kv_stride,
    pool_head_stride,
    pool_slot_stride,
    page_table_ptr,
    page_table_stride,
    kv_lens_ptr,
    Q_LEN: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV_HEADS: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
):
    # Program t writes token t = row * Q_LEN + j at position kv_len - Q_LEN + j of its row.
    t = tl.program_id(0)
    row = t // Q_LEN
    j = t % Q_LEN
    pos = tl.load(kv_lens_ptr + row).to(tl.int64) - Q_LEN + j
    valid = pos >= 0
    pos = tl.maximum(pos, 0)
    page = tl.load(page_table_ptr + row.to(tl.int64) * page_table_stride + pos // TOKENS_PER_BLOCK)
    heads = tl.arange(0, BLOCK_KV_HEADS)
    dims = tl.arange(0, HEAD_DIM)
    mask = (heads[:, None] < NUM_KV_HEADS) & valid
    src = t.to(tl.int64) * kv_row_stride + heads[:, None] * HEAD_DIM + dims[None, :]
    dst = (
        pool_ptr
        + page.to(tl.int64) * pool_page_stride
        + heads[:, None].to(tl.int64) * pool_head_stride
        + (pos % TOKENS_PER_BLOCK) * pool_slot_stride
        + dims[None, :]
    )
    tl.store(dst, tl.load(k_ptr + src, mask=mask), mask=mask)
    tl.store(dst + pool_kv_stride, tl.load(v_ptr + src, mask=mask), mask=mask)


def write_kv_paged(
    pool: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_table: torch.Tensor,
    kv_lens: torch.Tensor,
    q_len: int,
) -> None:
    """Write the step's K/V of every row into its pages.

    Args:
        pool: [pages, 2, kv_heads, tokens_per_block, head_dim] layer pool (HND).
        k, v: [rows * q_len, kv_heads, head_dim], row major; each token's heads contiguous.
        page_table: [rows, max_pages] int32 pool pages of every row.
        kv_lens: [rows] int32 device KV lengths including the step's ``q_len`` tokens.
    """
    tokens, num_kv_heads, head_dim = k.shape
    if tokens == 0:
        return
    if k.stride(2) != 1 or k.stride(1) != head_dim or v.stride() != k.stride():
        raise ValueError(f"k and v must share contiguous-head strides, got {k.stride()} / {v.stride()}.")
    if pool.stride(4) != 1:
        raise ValueError(f"pool must be contiguous in head_dim, got strides {pool.stride()}.")
    _write_kv_paged_kernel[(tokens,)](
        k,
        v,
        k.stride(0),
        pool,
        pool.stride(0),
        pool.stride(1),
        pool.stride(2),
        pool.stride(3),
        page_table,
        page_table.stride(0),
        kv_lens,
        Q_LEN=q_len,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_KV_HEADS=triton.next_power_of_2(num_kv_heads),
        TOKENS_PER_BLOCK=pool.shape[3],
    )


def fa4_paged_generation(
    pool: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    page_table: torch.Tensor,
    kv_lens: torch.Tensor,
    q_len: int,
    sm_scale: float,
) -> None:
    """Persist the step's K/V and attend every row's cache with FA4.

    Args:
        pool: [pages, 2, kv_heads, tokens_per_block, head_dim] layer pool (HND), bf16/fp16.
        q: [rows * q_len, heads, head_dim]; k, v: [rows * q_len, kv_heads, head_dim], row major
            (rotary embedding already applied).
        out: [rows * q_len, heads, head_dim] output.
        page_table: [rows, max_pages] int32 pool pages of every row.
        kv_lens: [rows] int32 device KV lengths including the step's ``q_len`` tokens.
    """
    fwd = _fa4_forward()
    if fwd is None:
        raise RuntimeError(f"{LIB_NAME}: FlashAttention-4 is unavailable.")
    tokens, num_heads, head_dim = q.shape
    rows = tokens // q_len
    write_kv_paged(pool, k, v, page_table, kv_lens, q_len)
    # FA4 takes NHD pages [pages, page_size, kv_heads, head_dim]; the HND pool transposes without a copy.
    k_pages = pool[:, 0].transpose(1, 2)
    v_pages = pool[:, 1].transpose(1, 2)
    fwd(
        q.view(rows, q_len, num_heads, head_dim),
        k_pages,
        v_pages,
        seqused_k=kv_lens,
        page_table=page_table,
        softmax_scale=sm_scale,
        causal=True,
        tile_mn=FA4_TILE_MN,
        num_threads=FA4_NUM_THREADS,
        num_splits=1,
        pack_gqa=None,
        out=out.view(rows, q_len, num_heads, head_dim),
        return_lse=False,
    )


class FlashAttn4GenerationFmha(PhasedFmha):
    """Generation-phase library on FlashAttention-4; see the module docstring."""

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self._pool_views: dict[int, torch.Tensor] = {}
        self._pool_manager: Optional[object] = None
        self._logged = False

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        if get_sm_version() != 90 or _fa4_forward() is None:
            return False
        if getattr(attn, "is_mla_enable", False) or getattr(attn, "sparse_params", None) is not None:
            return False
        if getattr(attn, "predicted_tokens_per_seq", 1) != 1 or getattr(attn, "attention_chunk_size", None):
            return False
        head_dim = getattr(attn, "head_dim", 0)
        num_heads = getattr(attn, "num_heads", 0)
        num_kv_heads = getattr(attn, "num_kv_heads", 0) or 0
        if head_dim not in (64, 128) or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            return False
        if QuantMode(getattr(attn, "quant_mode", 0)).has_kv_cache_quant():
            return False
        position_type = PositionEmbeddingType(getattr(attn, "position_embedding_type", 0))
        if position_type.is_rope():
            return _fused_rope_supported(attn)
        # ALiBi, relative and deferred embeddings are applied inside the attention op.
        return position_type == PositionEmbeddingType.learned_absolute

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        if phase == FmhaPhase.CONTEXT:
            return False
        if forward_args.attention_input_type == AttentionInputType.context_only:
            return False
        if metadata.num_generations == 0:
            return False
        if phase is None and metadata.num_contexts > 0:
            # Mixed batches are served phase by phase (CombinedFmha).
            return False
        manager = metadata.kv_cache_manager
        if (
            manager is None
            or not hasattr(manager, "index_mapper")
            or getattr(manager, "is_draft", False)
            or metadata.kv_cache_block_offsets is None
        ):
            return False
        if (
            metadata.beam_width != 1
            or metadata.is_cross
            or metadata.enable_helix
            or metadata.is_spec_dec_tree
            or metadata.is_spec_dec_dynamic_tree
        ):
            return False
        if not (forward_args.is_fused_qkv and k is None and v is None and forward_args.update_kv_cache):
            return False
        if forward_args.attention_mask != PredefinedAttentionMask.CAUSAL:
            return False
        if (
            forward_args.attention_mask_data is not None
            or forward_args.attention_sinks is not None
            or forward_args.relative_attention_bias is not None
            or forward_args.out_scale is not None
            or forward_args.output_sf is not None
        ):
            return False
        if q.dtype not in (torch.bfloat16, torch.float16):
            return False
        window = forward_args.attention_window_size
        if window is not None and metadata.max_seq_len is not None and window < metadata.max_seq_len:
            return False
        if PositionEmbeddingType(getattr(self.attn, "position_embedding_type", 0)).is_rope():
            max_seq_len = metadata.max_seq_len
            if max_seq_len is None or max_seq_len > (
                self.attn.rotary_cos_sin.numel() // _rotary_dim(self.attn)
            ):
                return False
        return True

    def _pool_view(self, metadata: "TrtllmAttentionMetadata") -> torch.Tensor:
        manager = metadata.kv_cache_manager
        if self._pool_manager is not manager:
            # The executor replaces the manager after memory estimation.
            self._pool_views.clear()
            self._pool_manager = manager
        layer_idx = self.attn.layer_idx
        pool = self._pool_views.get(layer_idx)
        if pool is None:
            pool = manager.get_buffers(layer_idx, kv_layout="HND")
            if pool is None:
                raise RuntimeError(f"{LIB_NAME}: layer {layer_idx} has no KV pool.")
            self._pool_views[layer_idx] = pool
        return pool

    def _page_table(self, params: FmhaParams) -> torch.Tensor:
        """[rows, max_pages] int32 pool pages of the phase's rows, derived on the device.

        The attention op's K block offset of a page is ``page * kv_factor`` in the
        layer's pool view, which folds K and V into one page.
        """
        metadata = params.meta
        manager = metadata.kv_cache_manager
        pool_index = int(manager.layer_to_pool_mapping_dict[manager.layer_offsets[self.attn.layer_idx]])
        rows = slice(params.seq_offset, params.seq_offset + params.batch_size)
        offsets = metadata.kv_cache_block_offsets[pool_index, rows, 0, :]
        table = torch.div(offsets, int(manager.kv_factor), rounding_mode="floor")
        if os.environ.get(CHECK_ENV, "0") == "1" and not torch.cuda.is_current_stream_capturing():
            self._check_page_table(params, table)
        return table

    def _check_page_table(self, params: FmhaParams, table: torch.Tensor) -> None:
        """Compare with the host table read the way the n-sample library reads it."""
        from .sampling_shared_prefix import _page_layout

        metadata = params.meta
        manager = metadata.kv_cache_manager
        pool_index, scale, divisor = _page_layout(metadata, self.attn.layer_idx)
        request_ids = metadata.request_ids
        beam_stride = int(manager.max_beam_width)
        lens = metadata.kv_lens_runtime[params.seq_offset : params.seq_offset + params.batch_size]
        host = metadata.host_kv_cache_block_offsets
        device = table.cpu()
        tpb = int(metadata.tokens_per_block)
        for i in range(params.batch_size):
            slot = manager.index_mapper.get_index(request_ids[params.seq_offset + i]) * beam_stride
            pages = (int(lens[i]) + tpb - 1) // tpb
            expect = host[pool_index, slot, 0, :pages].long() * scale // divisor
            if not torch.equal(device[i, :pages].long(), expect):
                raise RuntimeError(
                    f"{LIB_NAME} check: device page table of row {i} disagrees with the host table "
                    f"(layer {self.attn.layer_idx}): {device[i, :pages].tolist()[:8]} vs {expect.tolist()[:8]}"
                )

    def run_generation(self, params: FmhaParams) -> None:
        attn = params.attn
        qkv = params.qkv_input
        output = params.output
        if qkv is None or output is None:
            raise RuntimeError(f"{LIB_NAME} requires packed QKV input and an output tensor.")
        rows = params.batch_size
        tokens = params.num_tokens
        if rows <= 0 or tokens % rows:
            raise RuntimeError(f"{LIB_NAME}: {tokens} generation tokens are not uniform over {rows} rows.")
        q_len = tokens // rows
        if not self._logged:
            self._logged = True
            logger.info_once(
                f"{LIB_NAME}: first generation call ({rows} rows x {q_len} tokens)",
                key=f"{LIB_NAME}_first_call",
            )
        num_heads, num_kv_heads, head_dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
        kv_lens = params.meta.kv_lens_cuda_runtime[params.seq_offset : params.seq_offset + rows]
        if PositionEmbeddingType(attn.position_embedding_type).is_rope():
            rope_qkv_inplace(
                qkv,
                attn.rotary_cos_sin,
                kv_lens,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_dim=_rotary_dim(attn),
                q_len=q_len,
            )
        q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
        q = qkv[:, :q_size].reshape(tokens, num_heads, head_dim)
        k = qkv[:, q_size : q_size + kv_size].view(tokens, num_kv_heads, head_dim)
        v = qkv[:, q_size + kv_size : q_size + 2 * kv_size].view(tokens, num_kv_heads, head_dim)
        fa4_paged_generation(
            self._pool_view(params.meta),
            q,
            k,
            v,
            output.view(tokens, num_heads, head_dim),
            self._page_table(params),
            kv_lens,
            q_len,
            1.0 / (head_dim**0.5 * attn.q_scaling),
        )


__all__ = ["FlashAttn4GenerationFmha", "LIB_NAME", "fa4_paged_generation", "write_kv_paged"]
