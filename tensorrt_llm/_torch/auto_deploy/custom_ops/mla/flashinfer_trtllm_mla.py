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

"""FlashInfer TRTLLM-gen MLA backend for paged MLA decode on Blackwell.

This backend uses FlashInfer Path 2 (`trtllm_batch_decode_with_kv_cache_mla`) when:
- the batch is generate-only,
- the GPU is Blackwell-class (SM100+),
- the MLA shape matches the FlashInfer Path 2 contract.

For prefill or mixed batches, and for unsupported hardware, the op falls back to
an internal torch reference implementation over the same paged cache layout.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import flashinfer
import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ...utils.logger import ad_logger
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BatchInfo,
    Constant,
    MHACallable,
    PrepareMetadataHostCallable,
    ResourceHandler,
    ResourceHandlerDict,
    SequenceInfo,
)
from .triton_mla_cache_append import append_paged_mla_cache


@dataclass
class _WorkspaceState:
    buffer: Optional[torch.Tensor] = None


_WORKSPACES: Dict[Tuple[torch.device, torch.dtype], _WorkspaceState] = {}


class _CombinedMLAPagedResourceHandler(ResourceHandler):
    """Allocate a combined MLA paged cache for Path 2."""

    @property
    def is_paged(self) -> bool:
        return True

    def __init__(self, *token_shape: int, dtype: torch.dtype) -> None:
        self.token_shape = token_shape
        self.dtype = dtype

    def _get_bytes_per_token(self) -> int:
        return math.prod(self.token_shape) * self.dtype.itemsize

    def allocate(self, sequence_info: SequenceInfo) -> torch.Tensor:
        return torch.empty(
            sequence_info.num_blocks,
            sequence_info.tokens_per_block,
            *self.token_shape,
            device=sequence_info.device,
            dtype=self.dtype,
        )


def _get_workspace(device: torch.device) -> torch.Tensor:
    key = (device, torch.uint8)
    state = _WORKSPACES.get(key)
    if state is None or state.buffer is None or state.buffer.device != device:
        state = _WorkspaceState(
            buffer=torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        )
        _WORKSPACES[key] = state
    return state.buffer


def _is_blackwell_decode_supported(
    query: torch.Tensor,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
) -> bool:
    capability = torch.cuda.get_device_capability(query.device)
    return (
        capability >= (10, 0)
        and kv_lora_rank in (256, 512)
        and qk_nope_head_dim in (64, 128)
        and qk_rope_head_dim == 64
    )


def _append_to_paged_cache(
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    input_pos: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_lora_rank: int,
) -> None:
    num_seq = input_pos.shape[0]
    kpe = kpe.squeeze(1)
    cache_dtype = kv_cache.dtype

    if compressed_kv.dtype != cache_dtype:
        compressed_kv = compressed_kv.to(cache_dtype)
    if kpe.dtype != cache_dtype:
        kpe = kpe.to(cache_dtype)

    num_tokens = int(cu_seqlen_host[num_seq].item())
    device = kv_cache.device
    cu_seqlen_device = cu_seqlen_host[: num_seq + 1].to(device=device, dtype=torch.int32)
    cu_pages_device = cu_num_pages[: num_seq + 1].to(device=device, dtype=torch.int32)
    input_pos_device = input_pos.to(device=device, dtype=torch.int32)

    append_paged_mla_cache(
        compressed_kv,
        kpe,
        cu_seqlen_device,
        cu_pages_device,
        cache_loc,
        input_pos_device,
        kv_cache,
        kv_lora_rank,
        num_tokens,
        num_seq,
    )


def _gather_seq_cache(
    kv_cache: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_idx: int,
    seq_len_with_cache: int,
) -> torch.Tensor:
    page_size = kv_cache.shape[1]
    if seq_len_with_cache == 0:
        return kv_cache.new_empty((0, kv_cache.shape[-1]))

    page_start = int(cu_num_pages[seq_idx].item())
    page_end = int(cu_num_pages[seq_idx + 1].item())
    remaining = seq_len_with_cache
    chunks = []
    for flat_idx in range(page_start, page_end):
        page_idx = int(cache_loc[flat_idx].item())
        take = min(page_size, remaining)
        if take <= 0:
            break
        chunks.append(kv_cache[page_idx, :take])
        remaining -= take
    return torch.cat(chunks, dim=0)


def _compute_reference_prefill_or_mixed(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    input_pos: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
    out: torch.Tensor,
) -> None:
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    qk_rope_head_dim = q_pe.shape[3]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim
    num_seq = input_pos.shape[0]
    q_nope_flat = q_nope.view(-1, num_heads, qk_nope_head_dim)
    q_pe_flat = q_pe.view(-1, num_heads, qk_rope_head_dim)

    for seq_idx in range(num_seq):
        seq_start = int(cu_seqlen_host[seq_idx].item())
        seq_end = int(cu_seqlen_host[seq_idx + 1].item())
        q_nope_seq = q_nope_flat[seq_start:seq_end]
        q_pe_seq = q_pe_flat[seq_start:seq_end]
        kv_seq_len = int(seq_len_with_cache_host[seq_idx].item())
        cached = _gather_seq_cache(kv_cache, cu_num_pages, cache_loc, seq_idx, kv_seq_len)
        compressed_kv_cached = cached[:, :kv_lora_rank].to(q_nope.dtype)
        kpe_cached = cached[:, kv_lora_rank:].to(q_nope.dtype)

        kv_expanded = torch.matmul(compressed_kv_cached, kv_b_proj_weight.t())
        kv_expanded = kv_expanded.view(kv_seq_len, num_heads, qk_nope_head_dim + v_head_dim)
        k_nope = kv_expanded[:, :, :qk_nope_head_dim]
        v = kv_expanded[:, :, qk_nope_head_dim:]
        kpe_expanded = kpe_cached.unsqueeze(1).expand(-1, num_heads, -1)

        q_full = torch.cat([q_nope_seq, q_pe_seq], dim=-1)
        k_full = torch.cat([k_nope, kpe_expanded], dim=-1)
        q_t = q_full.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v.transpose(0, 1).unsqueeze(0)

        scores = torch.matmul(q_t.float(), k_t.float().transpose(-2, -1)) * scale
        seq_len = seq_end - seq_start
        causal_mask = torch.triu(
            torch.ones(seq_len, kv_seq_len, device=q_nope.device, dtype=torch.bool),
            diagonal=kv_seq_len - seq_len + 1,
        )
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1).to(q_nope.dtype)
        out_seq = torch.matmul(weights, v_t)[0].transpose(0, 1)
        out[seq_start:seq_end] = out_seq


def _compute_reference_decode(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    _input_pos: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
    out: torch.Tensor,
) -> None:
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    weight_reshaped = kv_b_proj_weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    w_k_nope = weight_reshaped[:, :qk_nope_head_dim, :]
    w_v = weight_reshaped[:, qk_nope_head_dim:, :]

    for seq_idx in range(_input_pos.shape[0]):
        kv_seq_len = int(seq_len_with_cache_host[seq_idx].item())
        cached = _gather_seq_cache(kv_cache, cu_num_pages, cache_loc, seq_idx, kv_seq_len)
        compressed_kv_cached = cached[:, :kv_lora_rank].to(q_nope.dtype)
        kpe_cached = cached[:, kv_lora_rank:].to(q_nope.dtype)

        q_nope_i = q_nope[seq_idx, 0]
        q_pe_i = q_pe[seq_idx, 0]
        q_absorbed = torch.einsum("nd,ndk->nk", q_nope_i, w_k_nope)
        scores_nope = torch.matmul(q_absorbed.float(), compressed_kv_cached.float().t())
        scores_pe = torch.matmul(q_pe_i.float(), kpe_cached.float().t())
        attn_scores = (scores_nope + scores_pe) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1).to(q_nope.dtype)
        weighted_kv = torch.matmul(attn_weights, compressed_kv_cached)
        out[seq_idx] = torch.einsum("nk,nvk->nv", weighted_kv, w_v)


@torch.library.custom_op(
    "auto_deploy::flashinfer_trtllm_mla_with_cache", mutates_args=("mla_paged_cache",)
)
def flashinfer_trtllm_mla_with_cache(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    mla_paged_cache: torch.Tensor,
    scale: Optional[float],
    kv_lora_rank: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    b, s = q_nope.shape[:2]
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    qk_rope_head_dim = q_pe.shape[3]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    num_total_tokens = num_prefill_tokens + num_decode

    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    q_nope_flat = q_nope.contiguous().view(b * s, num_heads, qk_nope_head_dim)
    q_pe_flat = q_pe.contiguous().view(b * s, num_heads, qk_rope_head_dim)
    compressed_kv_flat = compressed_kv.contiguous().view(b * s, kv_lora_rank)
    kpe_flat = kpe.contiguous().view(b * s, 1, qk_rope_head_dim)

    input_pos = seq_len_with_cache_host[:num_seq] - (
        cu_seqlen_host[1 : num_seq + 1] - cu_seqlen_host[:num_seq]
    )

    _append_to_paged_cache(
        compressed_kv_flat[:num_total_tokens],
        kpe_flat[:num_total_tokens],
        cu_seqlen_host[: num_seq + 1],
        cu_num_pages[: num_seq + 1],
        cache_loc,
        input_pos,
        mla_paged_cache,
        kv_lora_rank,
    )

    if out is not None:
        y = out.view(b * s, num_heads, v_head_dim)
    else:
        y = q_nope.new_zeros((b * s, num_heads, v_head_dim))

    if (
        num_prefill == 0
        and num_decode > 0
        and _is_blackwell_decode_supported(q_nope, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim)
    ):
        page_size = mla_paged_cache.shape[1]
        max_pages_per_seq = int(
            (cu_num_pages_host[1 : num_decode + 1] - cu_num_pages_host[:num_decode]).max().item()
        )
        # FlashInfer requires block_num % (128 / block_size) == 0
        alignment = max(1, 128 // page_size)
        max_pages_per_seq = ((max_pages_per_seq + alignment - 1) // alignment) * alignment
        # Block tables are zero-padded for alignment. Padding entries point to page 0
        # but are never accessed because seq_lens constrains the kernel's read bounds.
        pages_per_seq = cu_num_pages_host[1 : num_decode + 1] - cu_num_pages_host[:num_decode]
        seq_starts = cu_num_pages_host[:num_decode].to(device=q_nope.device)
        page_slots = torch.arange(max_pages_per_seq, device=q_nope.device, dtype=torch.int32)
        flat_idx = (seq_starts.unsqueeze(1) + page_slots.unsqueeze(0)).clamp(
            max=cache_loc.shape[0] - 1
        )
        block_tables = cache_loc[flat_idx.long()]
        mask = page_slots.unsqueeze(0) < pages_per_seq.to(device=q_nope.device).unsqueeze(1)
        block_tables = block_tables * mask.int()

        w_k_nope = kv_b_proj_weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)[
            :, :qk_nope_head_dim, :
        ]
        w_v = kv_b_proj_weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)[
            :, qk_nope_head_dim:, :
        ]
        q_nope_absorbed = torch.einsum(
            "bnd,ndk->bnk", q_nope_flat[:num_decode], w_k_nope
        ).contiguous()
        query = torch.cat([q_nope_absorbed, q_pe_flat[:num_decode]], dim=-1).unsqueeze(1)
        latent_out = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=mla_paged_cache,
            workspace_buffer=_get_workspace(q_nope.device),
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_len_with_cache_host[:num_decode].to(
                device=q_nope.device, dtype=torch.int32
            ),
            max_seq_len=int(seq_len_with_cache_host[:num_decode].max().item()),
            bmm1_scale=scale,
            bmm2_scale=1.0,
        ).squeeze(1)
        y[:num_decode] = torch.einsum("bnk,nvk->bnv", latent_out, w_v)
    else:
        if num_prefill > 0:
            _compute_reference_prefill_or_mixed(
                q_nope,
                q_pe,
                compressed_kv,
                kpe,
                kv_b_proj_weight,
                cu_seqlen_host[: num_seq + 1],
                input_pos,
                cu_num_pages[: num_seq + 1],
                cache_loc,
                seq_len_with_cache_host[:num_seq],
                mla_paged_cache,
                scale,
                kv_lora_rank,
                y,
            )
        if num_decode > 0:
            decode_slice = slice(num_prefill_tokens, num_prefill_tokens + num_decode)
            _compute_reference_decode(
                q_nope[:, :1]
                if s == 1
                else q_nope_flat[decode_slice].view(num_decode, 1, num_heads, qk_nope_head_dim),
                q_pe[:, :1]
                if s == 1
                else q_pe_flat[decode_slice].view(num_decode, 1, num_heads, qk_rope_head_dim),
                kv_b_proj_weight,
                input_pos[num_prefill:num_seq],
                cu_num_pages[num_prefill : num_seq + 1] - cu_num_pages[num_prefill],
                cache_loc[
                    int(cu_num_pages[num_prefill].item()) : int(cu_num_pages[num_seq].item())
                ],
                seq_len_with_cache_host[num_prefill:num_seq],
                mla_paged_cache,
                scale,
                kv_lora_rank,
                y[decode_slice].view(num_decode, num_heads, v_head_dim),
            )

    if out is not None:
        if num_total_tokens < b * s:
            y[num_total_tokens:].zero_()
        return out.new_empty(0)

    return y.view(b, s, num_heads, v_head_dim)


@flashinfer_trtllm_mla_with_cache.register_fake
def flashinfer_trtllm_mla_with_cache_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    mla_paged_cache: torch.Tensor,
    scale: Optional[float],
    kv_lora_rank: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is not None:
        return out.new_empty(0)

    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[-1]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim
    return q_nope.new_empty(q_nope.shape[0], q_nope.shape[1], num_heads, v_head_dim)


@AttentionRegistry.register("flashinfer_trtllm_mla")
class FlashInferTrtllmMLAAttention(AttentionDescriptor):
    """Attention descriptor for Blackwell-oriented FlashInfer Path 2 MLA."""

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        return 5

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_mla

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.flashinfer_trtllm_mla_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return [
            "batch_info_host",
            "cu_seqlen_host",
            "cu_num_pages",
            "cu_num_pages_host",
            "cache_loc",
            "last_page_len",
            "last_page_len_host",
            "seq_len_with_cache_host",
        ]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kpe_fake = source_attn_node.args[3].meta["val"]
        kv_lora_rank = compressed_kv_fake.shape[-1]
        qk_rope_head_dim = kpe_fake.shape[-1]
        model_dtype = compressed_kv_fake.dtype
        cache_dtype = cls.resolve_cache_dtype(cache_config.dtype, model_dtype)

        if cache_dtype != torch.bfloat16:
            ad_logger.warning(
                "flashinfer_trtllm_mla requires BF16 KV cache; overriding %s to %s.",
                cache_dtype,
                torch.bfloat16,
            )
            cache_dtype = torch.bfloat16

        return {
            "mla_paged_cache": _CombinedMLAPagedResourceHandler(
                kv_lora_rank + qk_rope_head_dim,
                dtype=cache_dtype,
            )
        }

    @classmethod
    def get_host_prepare_metadata_function(cls) -> Optional[PrepareMetadataHostCallable]:
        return None

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kv_lora_rank = compressed_kv_fake.shape[-1]
        scale = source_attn_node.kwargs.get("scale", None)
        return [scale, kv_lora_rank]
