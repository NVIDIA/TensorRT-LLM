from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import MLAParams, PositionalEmbeddingParams
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig

from .flash_mla import forward_sparse_mla_kvcache_bf16, should_use_short_mha
from .indexer import Indexer, transform_local_topk_and_prepare_pool_view
from .metadata import DSAtrtllmAttentionMetadata

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig

    from ....modules.attention import MLA


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

    # ------------------------------------------------------------------
    # Sparse attention lifecycle methods called by MLA
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def pre_attn_process(
        self,
        attn_metadata: DSAtrtllmAttentionMetadata,
        hidden_states: torch.Tensor,
        qr: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run pre_indexer_proj and _update_k_cache on all tokens.

        Called once on ALL tokens before the ctx/gen split.  Returns
        intermediates that will be sliced per ctx/gen and forwarded to
        forward_sparse_context / forward_sparse_generation.
        """
        q_fp8, k_fp8, k_scale, weights = self.indexer.pre_indexer_proj(
            qr,
            hidden_states,
            position_ids,
        )
        self.indexer._update_k_cache(k_fp8, k_scale, attn_metadata)
        return {
            "q_fp8": q_fp8,
            "k_fp8": k_fp8,
            "k_scale": k_scale,
            "weights": weights,
        }

    def forward_sparse_context(
        self,
        mla: MLA,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: DSAtrtllmAttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        **kwargs,
    ) -> None:
        """Dispatch sparse context-phase attention for DSA.

        SM >= 100: absorption path (topk_indices produced inside
        sparse_attn_predict, fed to the C++ kernel via wrapper.plan).
        SM < 100: FlashMLA path (topk_indices produced here via
        sparse_attn_indexer, fed to the Python FlashMLA kernel).
        """
        if should_use_short_mha(mla, attn_metadata, position_ids):
            mla.forward_context(
                q, compressed_kv, k_pe, position_ids, attn_metadata, output, latent_cache
            )
        elif get_sm_version() >= 100:
            mla.forward_absorption_context(
                q,
                compressed_kv,
                k_pe,
                attn_metadata,
                output,
                latent_cache=latent_cache,
                **kwargs,
            )
        else:
            topk_indices = self._run_sparse_indexer(q, attn_metadata, is_generation=False, **kwargs)
            forward_sparse_mla_kvcache_bf16(
                mla,
                q,
                latent_cache,
                attn_metadata,
                output,
                topk_indices,
                is_generation=False,
            )

    def forward_sparse_generation(
        self,
        mla: MLA,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: DSAtrtllmAttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor],
        **kwargs,
    ) -> None:
        """Dispatch sparse generation-phase attention for DSA."""
        if get_sm_version() >= 100:
            mla.forward_absorption_generation(
                q,
                compressed_kv,
                k_pe,
                attn_metadata,
                output,
                latent_cache=latent_cache,
                **kwargs,
            )
        else:
            topk_indices = self._run_sparse_indexer(q, attn_metadata, is_generation=True, **kwargs)
            forward_sparse_mla_kvcache_bf16(
                mla,
                q,
                latent_cache,
                attn_metadata,
                output,
                topk_indices,
                is_generation=True,
            )

    def _run_sparse_indexer(
        self,
        q: torch.Tensor,
        attn_metadata: DSAtrtllmAttentionMetadata,
        is_generation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Run sparse_attn_indexer using intermediates from pre_attn_process."""
        return self.indexer.sparse_attn_indexer(
            attn_metadata,
            q,
            kwargs.get("q_fp8"),
            kwargs.get("k_fp8"),
            kwargs.get("k_scale"),
            kwargs.get("weights"),
            is_generation=is_generation,
        )

    # ------------------------------------------------------------------
    # Sparse parameter preparation for wrapper.plan()
    # ------------------------------------------------------------------

    def prepare_sparse_params(self, q, k, metadata, **kwargs):
        """Prepare SparseParams for DSA.

        Calls sparse_attn_predict to run the indexer and transform indices.
        """
        from ..params import SparseParams

        sparse_attn_indices, sparse_attn_offsets = self.sparse_attn_predict(
            q, k, metadata, **kwargs
        )
        return SparseParams(
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=sparse_attn_offsets,
            sparse_attn_indices_block_size=(self.sparse_attention_config.get_indices_block_size()),
            sparse_mla_topk=(
                metadata.sparse_mla_topk if hasattr(metadata, "sparse_mla_topk") else 0
            ),
        )

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        is_generation: bool = True,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run sparse indexing and transform indices for the C++ kernel."""
        q_fp8 = kwargs.get("q_fp8")
        weights = kwargs.get("weights")
        if q_fp8 is None or weights is None:
            return None, None

        topk_indices = self.indexer.sparse_attn_indexer(
            metadata,
            q,
            q_fp8,
            kwargs.get("k_fp8"),
            kwargs.get("k_scale"),
            weights,
            is_generation=is_generation,
        )

        topk_indices_global, _ = transform_local_topk_and_prepare_pool_view(
            topk_indices, metadata, self.get_local_layer_idx(metadata), is_generation
        )

        return topk_indices_global, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None

    # ------------------------------------------------------------------
    # MLA RoPE + paged KV cache ops
    # ------------------------------------------------------------------

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
