from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm.models.modeling_utils import QuantConfig

from .indexer import Indexer, transform_local_topk_and_prepare_pool_view
from .metadata import DSAtrtllmAttentionMetadata

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


class DSATrtllmAttention(TrtllmAttention):
    Metadata = DSAtrtllmAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        mla_params: Optional[MLAParams] = None,
        skip_create_weights_in_init: bool = False,
        attention_chunk_size: Optional[int] = None,
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        dtype: Optional[torch.dtype] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ):
        if sparse_attention_config is None:
            raise ValueError(
                "sparse_attention_config is required for DSATrtllmAttention and cannot be None"
            )
        TrtllmAttention.__init__(
            self,
            layer_idx,
            num_heads,
            head_dim,
            sparse_attention_config=sparse_attention_config,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            skip_create_weights_in_init=skip_create_weights_in_init,
            attention_chunk_size=attention_chunk_size,
            **kwargs,
        )

        self.indexer = Indexer(
            quant_config,
            pos_embd_params,
            mla_params,
            skip_create_weights_in_init,
            sparse_attention_config,
            dtype,
            layer_idx,
            aux_stream,
        )

    def predict_sparse_indices(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Run the DSA indexer to predict sparse attention indices.

        Requires ``qr`` (compressed query before q_b_proj), ``hidden_states``,
        and ``position_ids`` in kwargs.  Returns topk_indices shaped
        ``[num_tokens, index_topk]``.
        """
        qr = kwargs.get("qr")
        hidden_states = kwargs.get("hidden_states")
        position_ids = kwargs.get("position_ids")
        if qr is None or hidden_states is None or position_ids is None:
            return None
        return self.indexer.forward(qr, hidden_states, metadata, position_ids)

    def predict_sparse_indices(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Run the DSA indexer to predict sparse attention indices.

        Requires ``qr`` (compressed query before q_b_proj), ``hidden_states``,
        and ``position_ids`` in kwargs.  Returns topk_indices shaped
        ``[num_tokens, index_topk]``.
        """
        qr = kwargs.get('qr')
        hidden_states = kwargs.get('hidden_states')
        position_ids = kwargs.get('position_ids')
        if qr is None or hidden_states is None or position_ids is None:
            return None
        return self.indexer.forward(qr, hidden_states, metadata, position_ids)

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        is_generation: bool = True,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Transform the local topk indices to global topk indices in paged kv cache
        topk_indices_global, _ = transform_local_topk_and_prepare_pool_view(
            topk_indices, metadata, self.get_local_layer_idx(metadata), is_generation
        )

        return topk_indices_global, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None

    def mla_rope_append_paged_kv_assign_q(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        is_generation: bool = False,
        **kwargs,
    ) -> None:
        if is_generation:
            cached_token_indptr = metadata.gen_cached_token_indptr
            kv_indptr = metadata.gen_kv_indptr
            num_seqs = metadata.num_generations
            max_seq_len = metadata.max_gen_seq_len
            block_offsets = metadata.kv_cache_block_offsets[:, metadata.num_contexts :]
        else:
            cached_token_indptr = metadata.ctx_cached_token_indptr
            kv_indptr = metadata.ctx_kv_indptr
            num_seqs = metadata.num_contexts
            max_seq_len = metadata.max_ctx_seq_len
            block_offsets = metadata.kv_cache_block_offsets
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        sink_token_length = 0
        beam_width = 1

        torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
            q,
            latent_cache,
            num_seqs,
            cached_token_indptr,
            kv_indptr,
            max_seq_len,
            self.wrapper.rotary_cos_sin,
            self.num_heads,
            self.mla_params.qk_nope_head_dim,
            self.mla_params.qk_rope_head_dim,
            self.mla_params.kv_lora_rank,
            block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            self.kv_scale_orig_quant,
            self.kv_scale_quant_orig,
            self.get_local_layer_idx(metadata),
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            sink_token_length,
            beam_width,
            self.wrapper.quant_mode,
        )

