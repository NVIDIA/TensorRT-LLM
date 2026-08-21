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
    kv_cache_dtype: str = "auto",
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
        kv_cache_dtype=kv_cache_dtype,
    )

    return attn_cls(
        layer_idx,
        num_heads,
        head_dim,
        num_kv_heads,
        **kwargs,
    )


def append_mla_latent_cache_generation_cuda_graph_safe(
    metadata,
    layer_idx: int,
    latent_cache: torch.Tensor,
) -> None:
    """Append generation-phase MLA latent tokens, safe under CUDA graphs.

    :func:`append_mla_latent_cache` computes every write location on the host
    (request ids, per-request block lists, cached-token counts), so a CUDA
    graph captures its copy kernels with those positions frozen and replays
    them against stale slots, silently corrupting the cache. This variant
    derives the destination purely from device tensors living in graph-stable
    buffers that ``metadata.prepare()`` refreshes every step:

    - ``kv_lens_cuda_runtime`` holds each request's total KV length (cached +
      new), so the ``q_len`` new tokens sit at positions
      ``kv_len - q_len .. kv_len - 1`` (clamped to 0 so graph-warmup passes
      with zeroed lengths stay in bounds).
    - ``kv_cache_block_offsets[pool_idx, slot, 0]`` is the C++
      ``setOffsets``-encoded block table: entries hold
      ``pool_block_index * num_pool_layers * kv_factor`` (+ the K/V field
      index, always 0 for the single-plane MLA cache), so the raw block index
      for the per-layer ``get_buffers`` view is recovered by integer division.

    Handles any uniform ``q_len >= 1`` per generation request: plain decode
    (``q_len == 1``) and speculative-verification batches (``q_len ==
    1 + draft_len``; the spec workers pad drafts to the static max, so the
    per-request token count is uniform). This matters for spec-dec under
    CUDA graphs: the previous ``q_len == 1``-only version silently fell back
    to the host-side loop for verification batches, whose capture-time write
    positions (dummy-request block tables) were frozen into the graph — real
    requests' generation-token latents were never appended on replay,
    corrupting decode accuracy (Kimi K3 SA GSM8K 88.2 with graphs vs 96.2
    eager).

    Falls back to the host-side loop for eager forwards (numerics identical
    to the non-graph baseline) and for ragged generation batches, which
    cannot occur under CUDA graphs.
    """
    kv_cache_manager = metadata.kv_cache_manager
    num_ctx = metadata.num_contexts
    n_gen = metadata.num_generations
    # Tensor shapes are static under CUDA graphs, so this host-side check is
    # stable across replays: generation-only graph batches carry a uniform
    # per-request token count (1 for plain decode, 1 + draft_len for
    # padded speculative verification).
    q_len_is_uniform = n_gen > 0 and latent_cache.shape[0] % n_gen == 0
    if not metadata.is_cuda_graph or not q_len_is_uniform:
        append_mla_latent_cache(
            kv_cache_manager,
            layer_idx,
            metadata.request_ids,
            metadata.seq_lens.tolist(),
            metadata.kv_cache_params.num_cached_tokens_per_seq,
            latent_cache,
            kv_layout=metadata.kv_layout,
            seq_start=num_ctx,
        )
        return

    kv_layout = metadata.kv_layout
    kv_cache = kv_cache_manager.get_buffers(layer_idx, kv_layout=kv_layout)

    # Static per-layer facts: plain ints baked into the kernel launches, and
    # they never change between replays. kv_cache_pool_mapping exists on both
    # V1 and V2 managers, including hybrid subclasses whose KV manager covers
    # a masked layer subset (layer_offsets maps the global layer index).
    layer_offset = kv_cache_manager.layer_offsets[layer_idx]
    pool_mapping = kv_cache_manager.kv_cache_pool_mapping
    pool_idx = int(pool_mapping[layer_offset, 0])
    num_pool_layers = int((pool_mapping[:, 0] == pool_idx).sum())
    kv_factor = kv_cache_manager.kv_factor
    tokens_per_block = kv_cache_manager.tokens_per_block

    # Everything below only reads graph-stable device buffers. ``q_len`` is
    # derived from static tensor shapes, so it is a stable host constant per
    # captured graph (1 for plain decode, 1 + draft_len for spec verify).
    q_len = latent_cache.shape[0] // n_gen
    kv_lens = metadata.kv_lens_cuda_runtime[num_ctx:num_ctx + n_gen]
    # ``kv_len`` includes the new tokens, so they occupy positions
    # ``kv_len - q_len .. kv_len - 1``. pos: [n_gen, q_len].
    pos = ((kv_lens.to(torch.int64) - q_len).clamp_(min=0).unsqueeze(1) +
           torch.arange(q_len, dtype=torch.int64, device=kv_lens.device))
    block_slot = pos // tokens_per_block
    block_offset = pos % tokens_per_block
    # [num_pools, max_num_sequences, 2, max_blocks_per_seq]; the two K/V
    # entries are identical for the kv_factor=1 MLA cache, take field 0.
    block_table = metadata.kv_cache_block_offsets[pool_idx,
                                                  num_ctx:num_ctx + n_gen, 0]
    encoded = block_table.gather(1, block_slot)  # [n_gen, q_len]
    # Placeholder entries are negative; clamp so warmup rows stay in bounds.
    # TODO(TRTLLM-15199): clamping to block 0 means a padded/warmup row
    # scatters into a real request's block 0. Exclude invalid rows (or
    # reserve a scratch block) instead of clamping.
    dest_block = encoded.to(
        torch.int64).clamp_(min=0) // (num_pool_layers * kv_factor)
    src = latent_cache.to(kv_cache.dtype).reshape(n_gen, q_len,
                                                  latent_cache.shape[-1])
    if kv_layout == "NHD":
        kv_cache[dest_block, 0, block_offset, 0, :] = src
    elif kv_layout == "HND":
        kv_cache[dest_block, 0, 0, block_offset, :] = src
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")


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
