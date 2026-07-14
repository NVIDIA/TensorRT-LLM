from typing import Optional, Sequence, Type

import torch

from tensorrt_llm.logger import logger

from ...models.modeling_utils import QuantConfig
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from .interface import AttentionBackend, MLAParams, PositionalEmbeddingParams
from .sparse import (get_flashinfer_sparse_attn_attention_backend,
                     get_trtllm_sparse_attn_attention_backend,
                     get_vanilla_sparse_attn_attention_backend)
from .sparse.params import SparseParams
from .trtllm import TrtllmAttention
from .vanilla import VanillaAttention


def get_attention_backend(
    backend_name: str,
    sparse_params: Optional[SparseParams] = None,
) -> Type[AttentionBackend]:
    backend_name = backend_name.upper()
    if backend_name == "VANILLA":
        if sparse_params is not None:
            return get_vanilla_sparse_attn_attention_backend(sparse_params)
        return VanillaAttention
    elif backend_name == "TRTLLM":
        if sparse_params is not None:
            return get_trtllm_sparse_attn_attention_backend(sparse_params)
        return TrtllmAttention
    elif backend_name == "FLASHINFER" and IS_FLASHINFER_AVAILABLE:
        from .flashinfer import FlashInferAttention
        if sparse_params is not None:
            return get_flashinfer_sparse_attn_attention_backend(sparse_params)
        return FlashInferAttention
    elif backend_name == "FLASHINFER_STAR_ATTENTION" and IS_FLASHINFER_AVAILABLE:
        from .star_flashinfer import StarAttention
        return StarAttention

    logger.warning("Falling back to TRTLLM attention backend")
    return TrtllmAttention


def create_attention(
    backend_name: str,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: Optional[int] = None,
    pos_embd_params: Optional[PositionalEmbeddingParams] = None,
    quant_config: Optional[QuantConfig] = None,
    q_scaling: Optional[float] = None,
    is_mla_enable: bool = False,
    q_lora_rank: Optional[int] = None,
    kv_lora_rank: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
    qk_nope_head_dim: Optional[int] = None,
    v_head_dim: Optional[int] = None,
    rope_append: Optional[bool] = None,
    hidden_size: Optional[int] = None,
    predicted_tokens_per_seq: Optional[int] = 1,
    skip_create_weights_in_init: bool = False,
    attention_chunk_size: Optional[int] = None,
    sparse_params: Optional[SparseParams] = None,
    dtype: Optional[torch.dtype] = None,
    aux_stream: Optional[torch.cuda.Stream] = None,
):
    if attention_chunk_size is not None and backend_name.upper() != "TRTLLM":
        raise ValueError(
            f"Backend {backend_name} does not support chunked attention.")
    attn_cls = get_attention_backend(backend_name, sparse_params=sparse_params)

    if is_mla_enable:
        assert attn_cls.support_mla(
        ), f"MLA is not supported for {backend_name} backend"
        assert (q_lora_rank > 0 and kv_lora_rank > 0 and qk_rope_head_dim > 0
                and qk_nope_head_dim > 0 and v_head_dim > 0)
        mla_params = MLAParams(
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            rope_append=True if rope_append is None else rope_append,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            hidden_size=hidden_size,
        )
    else:
        mla_params = None

    kwargs = dict(
        quant_config=quant_config,
        q_scaling=q_scaling,
        pos_embd_params=pos_embd_params,
        mla_params=mla_params,
        skip_create_weights_in_init=skip_create_weights_in_init,
        attention_chunk_size=attention_chunk_size,
        dtype=dtype,
        aux_stream=aux_stream,
        sparse_params=sparse_params,
    )

    return attn_cls(
        layer_idx,
        num_heads,
        head_dim,
        num_kv_heads,
        **kwargs,
    )


def append_mla_latent_cache(
    kv_cache_manager,
    layer_idx: int,
    request_ids: Sequence[int],
    seq_lens: Sequence[int],
    num_cached_tokens: Sequence[int],
    latent_cache: torch.Tensor,
    *,
    kv_layout: str = "NHD",
    seq_start: int = 0,
) -> torch.Tensor:
    """Append packed MLA latent tokens into a paged KV cache.

    The MLA cache has one latent head with ``kv_factor=1`` and stores
    ``[compressed_kv | k_pe]``. ``latent_cache`` is packed by request for the
    sequences in ``seq_lens[seq_start:]``.
    """
    kv_cache = kv_cache_manager.get_buffers(layer_idx, kv_layout=kv_layout)
    if kv_layout == "NHD":
        tokens_per_block = kv_cache.shape[2]
    elif kv_layout == "HND":
        tokens_per_block = kv_cache.shape[3]
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")

    blocks_per_seq = kv_cache_manager.get_batch_cache_indices(
        list(request_ids), layer_idx)

    offset = 0
    for i in range(seq_start, len(seq_lens)):
        q_len = int(seq_lens[i])
        new = latent_cache[offset:offset + q_len].to(kv_cache.dtype)
        start = int(num_cached_tokens[i])
        blocks = [b for b in blocks_per_seq[i] if b != -1]
        written = 0
        while written < q_len:
            pos = start + written
            block = blocks[pos // tokens_per_block]
            block_offset = pos % tokens_per_block
            n = min(tokens_per_block - block_offset, q_len - written)
            if kv_layout == "NHD":
                kv_cache[block, 0, block_offset:block_offset + n,
                         0, :].copy_(new[written:written + n])
            else:
                kv_cache[block, 0, 0, block_offset:block_offset + n, :].copy_(
                    new[written:written + n])
            written += n
        offset += q_len

    return kv_cache
