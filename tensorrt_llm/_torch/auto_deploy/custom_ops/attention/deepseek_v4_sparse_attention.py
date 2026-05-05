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

"""DeepSeek V4 sparse attention source and cached reference ops."""

from typing import Optional

import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from ..._compat import KvCacheConfig
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BatchInfo,
    Constant,
    MHACallable,
    ResourceHandlerDict,
    UnpagedResourceHandler,
)

__all__ = [
    "DeepSeekV4SparseAttention",
    "torch_deepseek_v4_sparse_attention",
    "torch_deepseek_v4_sparse_attention_with_cache",
]

_SPARSE_ATTENTION_CHUNK_TARGET_BYTES = 512 * 1024 * 1024
_SPARSE_ATTENTION_MAX_CHUNK_TOKENS = 64


def _validate_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.dim() != rank:
        raise ValueError(f"{name} must have rank {rank}, got rank {tensor.dim()}")


def _validate_deepseek_v4_sparse_attention_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
) -> None:
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)

    if not q.is_floating_point():
        raise TypeError(f"q must be floating point, got {q.dtype}")
    if not kv.is_floating_point():
        raise TypeError(f"kv must be floating point, got {kv.dtype}")
    if not attn_sink.is_floating_point():
        raise TypeError(f"attn_sink must be floating point, got {attn_sink.dtype}")
    if q.dtype != kv.dtype:
        raise TypeError(f"q and kv must have the same dtype, got {q.dtype} and {kv.dtype}")
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_idxs must be int32 or int64, got {topk_idxs.dtype}")

    if kv.device != q.device:
        raise ValueError(f"kv must be on {q.device}, got {kv.device}")
    if attn_sink.device != q.device:
        raise ValueError(f"attn_sink must be on {q.device}, got {attn_sink.device}")
    if topk_idxs.device != q.device:
        raise ValueError(f"topk_idxs must be on {q.device}, got {topk_idxs.device}")

    batch_size, seq_len, num_heads, head_dim = q.shape
    kv_batch_size, _, kv_head_dim = kv.shape
    topk_batch_size, topk_seq_len, _ = topk_idxs.shape

    if kv_batch_size != batch_size:
        raise ValueError(f"kv batch dimension must be {batch_size}, got {kv_batch_size}")
    if topk_batch_size != batch_size:
        raise ValueError(f"topk_idxs batch dimension must be {batch_size}, got {topk_batch_size}")
    if topk_seq_len != seq_len:
        raise ValueError(f"topk_idxs sequence dimension must be {seq_len}, got {topk_seq_len}")
    if kv_head_dim != head_dim:
        raise ValueError(f"kv head dimension must be {head_dim}, got {kv_head_dim}")
    if attn_sink.shape[0] != num_heads:
        raise ValueError(f"attn_sink length must be {num_heads}, got {attn_sink.shape[0]}")


def _validate_topk_idx_bounds(topk_idxs: torch.Tensor, kv_rows: int) -> None:
    valid_topk_idxs = topk_idxs[topk_idxs >= 0]
    if valid_topk_idxs.numel() == 0:
        return

    max_topk_idx = int(valid_topk_idxs.max().item())
    if max_topk_idx >= kv_rows:
        raise ValueError(
            f"topk_idxs entries must be less than kv rows {kv_rows}, got {max_topk_idx}"
        )


def _validate_swa_cache_inputs(q: torch.Tensor, kv: torch.Tensor, swa_cache: torch.Tensor) -> None:
    _validate_rank("swa_cache", swa_cache, 3)
    if not swa_cache.is_floating_point():
        raise TypeError(f"swa_cache must be floating point, got {swa_cache.dtype}")
    if swa_cache.device != q.device:
        raise ValueError(f"swa_cache must be on {q.device}, got {swa_cache.device}")
    if swa_cache.shape[-1] != kv.shape[-1]:
        raise ValueError(
            f"swa_cache head dimension must be {kv.shape[-1]}, got {swa_cache.shape[-1]}"
        )


def _gather_selected_kv(
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    batch_idxs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    gather_topk_idxs = topk_idxs.to(torch.long).clamp(min=0)
    if batch_idxs is not None:
        return kv[batch_idxs.to(torch.long).unsqueeze(1), gather_topk_idxs]

    batch_size, seq_len, k_select = topk_idxs.shape
    head_dim = kv.shape[-1]
    gather_idx = gather_topk_idxs.unsqueeze(-1).expand(batch_size, seq_len, k_select, head_dim)
    expanded_kv = kv.unsqueeze(1).expand(batch_size, seq_len, kv.shape[1], head_dim)
    return torch.gather(expanded_kv, dim=2, index=gather_idx)


def _to_host_long(name: str, tensor: torch.Tensor, length: int) -> torch.Tensor:
    flat = tensor.detach().cpu().to(torch.long).flatten()
    if flat.numel() < length:
        raise ValueError(f"{name} must have at least {length} elements, got {flat.numel()}")
    return flat[:length]


def _write_swa_cache(
    kv_seq: torch.Tensor,
    swa_cache: torch.Tensor,
    slot_idx: int,
    input_pos: int,
) -> None:
    if input_pos < 0:
        raise ValueError(f"input_pos must be non-negative, got {input_pos}")
    end_pos = input_pos + kv_seq.shape[0]
    if end_pos > swa_cache.shape[1]:
        raise ValueError(
            f"Sequence write [{input_pos}, {end_pos}) exceeds SWA cache length {swa_cache.shape[1]}"
        )
    swa_cache[slot_idx, input_pos:end_pos].copy_(kv_seq.to(swa_cache.dtype))


def _slice_sequence_tokens(
    tensor: torch.Tensor,
    seq_idx: int,
    flat_start: int,
    seq_len: int,
) -> torch.Tensor:
    if tensor.shape[0] > seq_idx and tensor.shape[0] != 1:
        return tensor[seq_idx, :seq_len]
    return tensor.reshape(-1, *tensor.shape[2:])[flat_start : flat_start + seq_len]


def _prefill_kv_source(
    kv: torch.Tensor,
    kv_seq: torch.Tensor,
    seq_idx: int,
    num_seq: int,
) -> torch.Tensor:
    if kv.shape[0] > seq_idx and kv.shape[0] != 1:
        return kv[seq_idx : seq_idx + 1]
    if num_seq == 1:
        return kv
    return kv_seq.unsqueeze(0)


def _cached_sparse_attention_from_positions(
    q_token: torch.Tensor,
    attn_sink: torch.Tensor,
    swa_cache: torch.Tensor,
    slot_idx: int,
    positions: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    positions_long = positions.to(torch.long)
    gather_positions = positions_long.clamp(min=0)
    if gather_positions.numel() > 0 and int(gather_positions.max().item()) >= swa_cache.shape[1]:
        raise ValueError(
            f"topk/cache position {int(gather_positions.max().item())} exceeds SWA cache "
            f"length {swa_cache.shape[1]}"
        )

    selected_kv = swa_cache[slot_idx, gather_positions].to(q_token.dtype).unsqueeze(0)
    local_topk = torch.arange(positions_long.numel(), dtype=torch.long, device=q_token.device).view(
        1, 1, -1
    )
    local_topk = torch.where(positions_long.view(1, 1, -1) < 0, -1, local_topk)
    return torch_deepseek_v4_sparse_attention(
        q_token.view(1, 1, *q_token.shape),
        selected_kv,
        attn_sink,
        local_topk,
        softmax_scale,
    ).view(*q_token.shape)


def _cached_local_window_attention(
    q_seq: torch.Tensor,
    attn_sink: torch.Tensor,
    swa_cache: torch.Tensor,
    slot_idx: int,
    input_pos: int,
    window_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    outputs = []
    for token_offset in range(q_seq.shape[0]):
        query_pos = input_pos + token_offset
        start_pos = max(0, query_pos - window_size + 1)
        positions = torch.arange(start_pos, query_pos + 1, device=q_seq.device)
        outputs.append(
            _cached_sparse_attention_from_positions(
                q_seq[token_offset],
                attn_sink,
                swa_cache,
                slot_idx,
                positions,
                softmax_scale,
            )
        )
    if not outputs:
        return q_seq.new_empty(q_seq.shape)
    return torch.stack(outputs, dim=0)


def _cached_topk_attention(
    q_seq: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_seq: torch.Tensor,
    swa_cache: torch.Tensor,
    slot_idx: int,
    softmax_scale: float,
) -> torch.Tensor:
    outputs = []
    for token_offset in range(q_seq.shape[0]):
        outputs.append(
            _cached_sparse_attention_from_positions(
                q_seq[token_offset],
                attn_sink,
                swa_cache,
                slot_idx,
                topk_seq[token_offset],
                softmax_scale,
            )
        )
    if not outputs:
        return q_seq.new_empty(q_seq.shape)
    return torch.stack(outputs, dim=0)


def _sparse_attention_query_chunk_size(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    k_select: int,
    compute_dtype: torch.dtype,
) -> int:
    compute_element_size = torch.empty((), dtype=compute_dtype).element_size()
    logits_element_size = torch.empty((), dtype=torch.float32).element_size()
    bytes_per_token = (
        k_select * head_dim * compute_element_size
        + 3 * num_heads * (k_select + 1) * logits_element_size
        + num_heads * head_dim * compute_element_size
    )
    if bytes_per_token <= 0:
        return 1
    chunk_size = _SPARSE_ATTENTION_CHUNK_TARGET_BYTES // bytes_per_token
    chunk_size = max(1, int(chunk_size))
    chunk_size = min(chunk_size, _SPARSE_ATTENTION_MAX_CHUNK_TOKENS)
    return min(num_tokens, chunk_size)


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_sparse_attention", mutates_args=())
def torch_deepseek_v4_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    enable_sharding: bool = False,
    layer_type: str = "mha_sparse",
    layer_idx: Optional[int] = None,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_dim: Optional[int] = None,
) -> torch.Tensor:
    """Reference DeepSeek V4 sparse attention source op.

    Args:
        q: Query states with shape ``[batch, seq_len, num_heads, head_dim]``.
        kv: Shared sparse key/value rows with shape ``[batch, kv_rows, head_dim]``.
        attn_sink: Per-head sink logits with shape ``[num_heads]``.
        topk_idxs: Selected row indices into ``kv`` with shape
            ``[batch, seq_len, k_select]``. Duplicate indices are preserved and
            receive independent probability mass. Negative indices are masked
            slots and receive zero probability.
        softmax_scale: Scale applied to query/key logits before adding the sink
            logit.

    Returns:
        Attention output with shape ``[batch, seq_len, num_heads, head_dim]``.
        The sink participates in softmax normalization but contributes no value
        vector.
    """
    del (
        enable_sharding,
        layer_type,
        layer_idx,
        window_size,
        compress_ratio,
        max_compressed_len,
        head_dim,
        rope_dim,
    )

    _validate_deepseek_v4_sparse_attention_inputs(q, kv, attn_sink, topk_idxs)
    _validate_topk_idx_bounds(topk_idxs, kv.shape[1])

    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    batch_size, seq_len, num_heads, q_head_dim = q.shape
    _, _, k_select = topk_idxs.shape
    num_tokens = batch_size * seq_len
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if num_tokens == 0:
        return output

    chunk_size = _sparse_attention_query_chunk_size(
        num_tokens, num_heads, q_head_dim, k_select, compute_dtype
    )

    q_flat = q.reshape(num_tokens, num_heads, q_head_dim)
    topk_flat = topk_idxs.reshape(num_tokens, k_select)
    batch_idxs = torch.arange(batch_size, device=q.device).view(batch_size, 1)
    batch_idxs = batch_idxs.expand(batch_size, seq_len).reshape(num_tokens)
    output_flat = output.reshape(num_tokens, num_heads, q_head_dim)
    sink_logits = attn_sink.to(dtype=compute_dtype).reshape(1, num_heads, 1)

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        topk_chunk = topk_flat[start:end]
        selected_kv_compute = _gather_selected_kv(kv, topk_chunk, batch_idxs[start:end]).to(
            compute_dtype
        )
        q_compute = q_flat[start:end].to(compute_dtype)

        logits = torch.einsum("thd,tkd->thk", q_compute, selected_kv_compute)
        logits = logits * softmax_scale
        logits = logits.masked_fill((topk_chunk < 0).unsqueeze(1), float("-inf"))
        chunk_sink_logits = sink_logits.expand(end - start, num_heads, 1)
        logits_with_sink = torch.cat([logits, chunk_sink_logits], dim=-1)

        weights_with_sink = torch.softmax(logits_with_sink, dim=-1, dtype=torch.float32)
        weights = weights_with_sink[..., :-1].to(compute_dtype)
        chunk_output = torch.einsum("thk,tkd->thd", weights, selected_kv_compute)
        output_flat[start:end].copy_(chunk_output.to(q.dtype))

    return output


@torch_deepseek_v4_sparse_attention.register_fake
def torch_deepseek_v4_sparse_attention_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    enable_sharding: bool = False,
    layer_type: str = "mha_sparse",
    layer_idx: Optional[int] = None,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_dim: Optional[int] = None,
) -> torch.Tensor:
    """Fake implementation for torch.export tracing."""
    del (
        softmax_scale,
        enable_sharding,
        layer_type,
        layer_idx,
        window_size,
        compress_ratio,
        max_compressed_len,
        head_dim,
        rope_dim,
    )
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)
    return q.new_empty(q.shape).contiguous()


@torch.library.custom_op(
    "auto_deploy::torch_deepseek_v4_sparse_attention_with_cache",
    mutates_args=("swa_cache",),
)
def torch_deepseek_v4_sparse_attention_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    swa_cache: torch.Tensor,
    softmax_scale: float,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
) -> torch.Tensor:
    """Reference cached DeepSeek V4 sparse attention.

    This PR3 validation path supports uncompressed ratio-0 sparse/SWA layers.
    Compressed ratio-4 and ratio-128 decode require compressor/indexer cache
    state that is intentionally outside this minimal reference op.
    """
    _validate_deepseek_v4_sparse_attention_inputs(q, kv, attn_sink, topk_idxs)
    _validate_topk_idx_bounds(topk_idxs, kv.shape[1])
    _validate_swa_cache_inputs(q, kv, swa_cache)

    if compress_ratio != 0:
        raise NotImplementedError(
            "DeepSeek V4 compressed sparse-attention cache is not implemented in the "
            "minimal PR3 reference path. Only compress_ratio=0 SWA/local-window cache "
            "validation is supported; ratio-4 and ratio-128 decode need compressor/indexer state."
        )
    if window_size is not None and window_size <= 0:
        raise ValueError(f"window_size must be positive when provided, got {window_size}")

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode
    q_flat = q.reshape(-1, *q.shape[2:])
    if active_tokens > q_flat.shape[0]:
        raise ValueError(
            f"active token count {active_tokens} exceeds flattened q tokens {q_flat.shape[0]}"
        )

    seq_len_host = _to_host_long("seq_len", seq_len, num_seq)
    input_pos_host = _to_host_long("input_pos", input_pos, num_seq)
    slot_idx_host = _to_host_long("slot_idx", slot_idx, num_seq)
    cu_seqlen_host = _to_host_long("cu_seqlen", cu_seqlen, num_seq + 1)

    output_flat = torch.zeros_like(q_flat)

    for seq_idx in range(num_seq):
        seq_len_i = int(seq_len_host[seq_idx].item())
        if seq_len_i == 0:
            continue
        flat_start = int(cu_seqlen_host[seq_idx].item())
        input_pos_i = int(input_pos_host[seq_idx].item())
        slot_idx_i = int(slot_idx_host[seq_idx].item())
        if slot_idx_i < 0 or slot_idx_i >= swa_cache.shape[0]:
            raise ValueError(f"slot_idx must be in [0, {swa_cache.shape[0]}), got {slot_idx_i}")
        if window_size is None and input_pos_i > 0:
            raise NotImplementedError(
                "DeepSeek V4 cached sparse decode without window_size is not supported "
                "by the minimal PR3 reference path because topk_idxs has no unambiguous "
                "cache-position namespace after prior cached tokens."
            )

        q_seq = q_flat[flat_start : flat_start + seq_len_i]
        kv_seq = _slice_sequence_tokens(kv, seq_idx, flat_start, seq_len_i)
        topk_seq = _slice_sequence_tokens(topk_idxs, seq_idx, flat_start, seq_len_i)
        if q_seq.shape[0] != seq_len_i:
            raise ValueError(
                f"Sequence {seq_idx} q slice has length {q_seq.shape[0]}, expected {seq_len_i}"
            )
        if kv_seq.shape[0] != seq_len_i:
            raise ValueError(
                f"Sequence {seq_idx} kv slice has length {kv_seq.shape[0]}, expected {seq_len_i}"
            )
        if topk_seq.shape[0] != seq_len_i:
            raise ValueError(
                f"Sequence {seq_idx} topk_idxs slice has length {topk_seq.shape[0]}, "
                f"expected {seq_len_i}"
            )
        _write_swa_cache(kv_seq, swa_cache, slot_idx_i, input_pos_i)

        if input_pos_i == 0:
            kv_source = _prefill_kv_source(kv, kv_seq, seq_idx, num_seq)
            output_flat[flat_start : flat_start + seq_len_i] = torch_deepseek_v4_sparse_attention(
                q_seq.unsqueeze(0),
                kv_source,
                attn_sink,
                topk_seq.unsqueeze(0),
                softmax_scale,
            ).squeeze(0)
        elif window_size is not None:
            output_flat[flat_start : flat_start + seq_len_i] = _cached_local_window_attention(
                q_seq,
                attn_sink,
                swa_cache,
                slot_idx_i,
                input_pos_i,
                window_size,
                softmax_scale,
            )
        else:
            output_flat[flat_start : flat_start + seq_len_i] = _cached_topk_attention(
                q_seq,
                attn_sink,
                topk_seq,
                swa_cache,
                slot_idx_i,
                softmax_scale,
            )

    if active_tokens < q_flat.shape[0]:
        output_flat[active_tokens:].zero_()
    return output_flat.view_as(q)


@torch_deepseek_v4_sparse_attention_with_cache.register_fake
def torch_deepseek_v4_sparse_attention_with_cache_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    swa_cache: torch.Tensor,
    softmax_scale: float,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
) -> torch.Tensor:
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)
    _validate_rank("swa_cache", swa_cache, 3)
    del (
        kv,
        attn_sink,
        topk_idxs,
        batch_info_host,
        seq_len,
        input_pos,
        slot_idx,
        cu_seqlen,
        swa_cache,
        softmax_scale,
        window_size,
        compress_ratio,
    )
    return q.new_empty(q.shape).contiguous()


@AttentionRegistry.register("deepseek_v4_sparse")
class DeepSeekV4SparseAttention(AttentionDescriptor):
    """Cached DeepSeek V4 sparse attention descriptor for reference validation."""

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        return 4

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> list[str]:
        return ["batch_info_host", "seq_len", "input_pos", "slot_idx", "cu_seqlen"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        kv_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        head_dim = int(kv_fake.shape[-1])
        return {
            "swa_cache": UnpagedResourceHandler(
                head_dim,
                dtype=cls.resolve_cache_dtype(cache_config.dtype, kv_fake.dtype),
            )
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> list[Constant]:
        softmax_scale, window_size, compress_ratio = extract_op_args(
            source_attn_node,
            "softmax_scale",
            "window_size",
            "compress_ratio",
        )
        if not isinstance(softmax_scale, float):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"float softmax_scale, got {softmax_scale!r}."
            )
        if window_size is not None and not isinstance(window_size, int):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"int window_size or None, got {window_size!r}."
            )
        if not isinstance(compress_ratio, int):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"int compress_ratio, got {compress_ratio!r}."
            )
        if compress_ratio != 0:
            raise RuntimeError(
                "DeepSeek V4 compressed sparse attention cache insertion is not supported "
                "by the minimal PR3 reference path. Only compress_ratio=0 SWA/local-window "
                f"validation is supported, got compress_ratio={compress_ratio}."
            )
        return [softmax_scale, window_size, compress_ratio]
