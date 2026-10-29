# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention modules of the glm_moe_dsa sm_103 tp8 target: the MLA, its DSA hooks, its DSA backend and
its indexer.

``GlmMLA.adopt`` re-classes a layer's built-in attention modules in place -- the
``DeepseekV32Attention`` as ``GlmMLA``, its ``DSAHooks`` as ``GlmDsaHooks``, its
``DSATrtllmAttention`` as ``GlmDsaBackend`` and its ``Indexer`` as ``GlmIndexer`` -- and attaches
the attention fast paths the installed plan enables: the attention output tail and, with
``sparse_mla_decode``, that FMHA library ahead of the registry's on the layer's attention
backends. The backends build their FMHA libraries later, in ``create_weights``, so the library is
created when and where the built-in model creates its own. The classes add methods and
class-level defaults only, so the module tree and the weights stay the built-in model's.

Each forward override runs the target's decode composition (``attention_forward``) when the
step is a decode step (``dispatch.Phase.DECODE``) and the built-in method otherwise.
"""

from typing import Optional, Tuple

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.attention import _helix_cp_allgather_input
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.dsa.module import DSAHooks
from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention
from tensorrt_llm._torch.utils import Fp4QuantizedTensor

from . import attention_forward, dispatch
from .kernels.indexer_topk import region


def reclass(module, builtin: type, cls: type) -> None:
    """Re-class ``module``, which must be exactly the built-in class ``builtin``, as ``cls``.

    Raises:
        TypeError: if ``module`` is not exactly ``builtin`` (a subclass of it is refused too).
    """
    if type(module) is not builtin:
        raise TypeError(
            f"{cls.__name__} adopts the built-in {builtin.__name__}, got {type(module).__name__}"
        )
    module.__class__ = cls


def _deferred_norm_view(hidden_states) -> Optional[torch.Tensor]:
    """The tensor view of a deferred-norm packet, else None."""
    if dispatch.is_enabled("deferred_norm") and dispatch.get_impl("deferred_norm", "is_deferred")(
        hidden_states
    ):
        return hidden_states.residual
    return None


class GlmMLA(DeepseekV32Attention):
    """A layer's MLA: the built-in attention, with the target's decode composition."""

    # The attention output tail (``attention_output_tail``) and whether it also prefetches its
    # weights into L2 (``attention_output_tail_prefetch``); None keeps the built-in output.
    _attn_tail = None
    _tail_prefetch = False

    @classmethod
    def adopt(cls, mla: DeepseekV32Attention) -> "GlmMLA":
        """Re-class a layer's built-in attention modules and attach its attention fast paths.

        Raises:
            TypeError: if ``mla``, its DSA hooks, its DSA backend or its indexer is not exactly
                the built-in class.
        """
        reclass(mla, DeepseekV32Attention, cls)
        mla._adopt_attention_tail()
        GlmDsaHooks.adopt(mla.sparse_attn_hooks)
        GlmDsaBackend.adopt(mla.mqa)
        if mla.mqa.indexer is not None:
            GlmIndexer.adopt(mla.mqa.indexer)
        # The sparse-MLA decode library goes ahead of the registry's libraries on both attention
        # backends, which build their FMHA libraries in ``create_weights``.
        if dispatch.is_enabled("sparse_mla_decode"):
            libraries = (dispatch.get_impl("sparse_mla_decode", "fmha_lib"),)
            for backend in (mla.mha, mla.mqa):
                if backend is not None:
                    backend.prepended_fmha_lib_classes = libraries
        return mla

    def _adopt_attention_tail(self) -> None:
        """The attention output tail: the fused generation W_UV and the local o_proj."""
        # attention_output_tail_prefetch uses the same tail object to stage the tail weights
        # into L2 off the critical path while the attention runs.
        self._tail_prefetch = dispatch.is_enabled("attention_output_tail_prefetch")
        self._attn_tail = None
        # attention_output_into_hop publishes the tail's partial into the post-attention hop's
        # transport.
        if (
            dispatch.is_enabled("attention_output_tail")
            or self._tail_prefetch
            or dispatch.is_enabled("attention_output_into_hop")
        ):
            self._attn_tail = dispatch.get_impl("attention_output_tail", "make_attn_tail")(self)
            # oproj_weight_staging: the tail's o_proj stages the W_o rows it streams after its
            # programmatic-dependent-launch wait before that wait.
            if dispatch.is_enabled("oproj_weight_staging"):
                self._attn_tail = dispatch.get_impl("oproj_weight_staging", "wrap_attn_tail")(
                    self, self._attn_tail
                )

    def forward_absorption_generation(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
        absorbed_fused_q: Optional[torch.Tensor] = None,
        deferred_indexer_k=None,
        **kwargs,
    ) -> torch.Tensor:
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super().forward_absorption_generation(
                q=q,
                compressed_kv=compressed_kv,
                k_pe=k_pe,
                attn_metadata=attn_metadata,
                output=output,
                position_ids=position_ids,
                latent_cache=latent_cache,
                **kwargs,
            )
        return attention_forward.forward_absorption_generation(
            self,
            q,
            compressed_kv,
            k_pe,
            attn_metadata,
            output,
            position_ids=position_ids,
            latent_cache=latent_cache,
            absorbed_fused_q=absorbed_fused_q,
            deferred_indexer_k=deferred_indexer_k,
            **kwargs,
        )

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super().forward(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                all_reduce_params=all_reduce_params,
                latent_cache_gen=latent_cache_gen,
            )
        hidden_states = _helix_cp_allgather_input(
            hidden_states, attn_metadata, self.mapping, self.layer_idx
        )

        # Unquantized view of the module input, used by create_mla_outputs and
        # by _apply_output_gate.
        output_hidden_states = hidden_states
        deferred_view = _deferred_norm_view(hidden_states)
        if deferred_view is not None:
            # A deferred-norm packet: only the DSA query path consumes it; the outputs are
            # allocated on its tensor view.
            output_hidden_states = deferred_view
        elif isinstance(hidden_states, Fp4QuantizedTensor):
            assert hidden_states.unquantized_hidden_states is not None, (
                "MLA.forward received an Fp4QuantizedTensor without a "
                "unquantized_hidden_states view"
            )
            output_hidden_states = hidden_states.unquantized_hidden_states

        # The attention tail decides here, before the attention op, so the W_UV step inside the
        # op and the projection below see one decision. arm() returns False under
        # torch.compile and inside breakable CUDA graphs, where the op runs apart from the
        # traced or captured projection.
        attn_tail = getattr(self, "_attn_tail", None)
        tail_armed = attn_tail is not None and attn_tail.arm(self, attn_metadata)

        if self.register_to_config:
            attn_output = [
                torch.ops.trtllm.create_mla_outputs(output_hidden_states, self.layer_idx_str)
            ]
            self._forward_custom_op(
                hidden_states,
                position_ids,
                attn_output,
                latent_cache_gen,
            )
        else:
            attn_output = self._create_outputs(
                hidden_states if deferred_view is None else deferred_view, attn_metadata
            )
            self.forward_impl(
                position_ids,
                hidden_states,
                attn_metadata,
                attn_output=attn_output,
                latent_cache_gen=latent_cache_gen,
            )

        attn_output[0] = self._apply_output_gate(output_hidden_states, attn_output[0])
        if tail_armed:
            partial = attn_tail.project(self, attn_output[0])
            if partial is not None:
                # The tail returns o_proj's rank-local GEMM output; reduce it as the
                # ROW-parallel Linear.forward does for this o_proj (the tail takes only an
                # o_proj without bias, LoRA or fused GEMM+all-reduce).
                if self.o_proj.reduce_output:
                    partial = self.o_proj.all_reduce(partial, all_reduce_params=all_reduce_params)
                return partial
        return self._project_output(attn_output, position_ids, attn_metadata, all_reduce_params)


class GlmDsaHooks(DSAHooks):
    """A layer's DSA hooks: the built-in sparse attention, with the target's decode forward."""

    @classmethod
    def adopt(cls, hooks: DSAHooks) -> "GlmDsaHooks":
        """Re-class a layer's built-in DSA hooks.

        Raises:
            TypeError: if ``hooks`` is not exactly the built-in ``DSAHooks``.
        """
        reclass(hooks, DSAHooks, cls)
        return hooks

    def forward(
        self,
        mla,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attn_output: list[torch.Tensor],
    ) -> None:
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super().forward(
                mla=mla,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                attn_output=attn_output,
            )
        attention_forward.forward_sparse_attn(
            mla, position_ids, hidden_states, attn_metadata, attn_output
        )

    def forward_custom_op(
        self,
        mla,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_output: list[torch.Tensor],
        latent_cache_gen: Optional[torch.Tensor],
    ) -> bool:
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super().forward_custom_op(
                mla=mla,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attn_output=attn_output,
                latent_cache_gen=latent_cache_gen,
            )
        attention_forward.forward_sparse_attn_custom_op(
            mla,
            hidden_states,
            position_ids,
            attn_output,
            latent_cache_gen,
        )
        return True


class GlmDsaBackend(DSATrtllmAttention):
    """A layer's DSA attention backend, with the target's top-k bridge on decode steps."""

    @classmethod
    def adopt(cls, backend: DSATrtllmAttention) -> "GlmDsaBackend":
        """Re-class a layer's built-in DSA attention backend.

        Raises:
            TypeError: if ``backend`` is not exactly the built-in ``DSATrtllmAttention``.
        """
        reclass(backend, DSATrtllmAttention, cls)
        return backend

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the DSA indexer and convert its local TopK to paged indices."""
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super().sparse_attn_predict(
                q=q, k=k, metadata=metadata, forward_args=forward_args
            )
        is_generation = forward_args.attention_input_type == AttentionInputType.generation_only
        sparse_backend_args = forward_args.sparse_backend_args
        phase_start = metadata.num_ctx_tokens if is_generation else 0
        phase_end = metadata.num_tokens if is_generation else metadata.num_ctx_tokens

        shared_topk_indices = metadata.shared_topk_indices
        # rope_kv_append: one fused bridge kernel performs the native top-k copy of this call
        # and the local -> global remap. With cross_rank_topk_global_emit the cross-rank top-k
        # writes them and records it on its packet.
        bridge = None
        bridge_path = (
            "cross_rank_topk_global_emit"
            if dispatch.is_enabled("cross_rank_topk_global_emit")
            else "rope_kv_append"
        )
        if dispatch.is_enabled("rope_kv_append"):
            bridge = dispatch.get_impl(bridge_path, "new_bridge_packet")(metadata, is_generation)
        if self.indexer is None:
            topk_indices = shared_topk_indices[phase_start:phase_end]
        else:
            indexer_intermediates = sparse_backend_args.indexer_intermediates
            # The indexer's top-k region reads the bridge of this call only.
            self.indexer._topk_bridge = bridge
            try:
                topk_indices = self.indexer.forward_from_projected(
                    metadata,
                    q,
                    indexer_intermediates,
                    is_generation=is_generation,
                )
            finally:
                self.indexer._topk_bridge = None
            preserve_mtp_topk = metadata.in_mtp_draft_loop and self.indexer.mtp_index_share
            if shared_topk_indices is not None and not preserve_mtp_topk:
                if bridge is not None:
                    dispatch.get_impl("rope_kv_append", "defer_shared_copy")(
                        bridge, shared_topk_indices, phase_start, topk_indices
                    )
                else:
                    shared_topk_indices[
                        phase_start : phase_start + topk_indices.shape[0],
                        : topk_indices.shape[1],
                    ].copy_(topk_indices)

        local_layer_idx = self.get_local_layer_idx(metadata)
        kv_cache_dtype = getattr(metadata.kv_cache_manager, "dtype", None)
        if kv_cache_dtype == tensorrt_llm.bindings.DataType.NVFP4 and not is_generation:
            metadata._ensure_pool_view_cached()
            page_index_scale, layer_offset = (
                metadata.kv_cache_manager.get_primary_pool_page_index_params(local_layer_idx)
            )
            total_kv_tokens = metadata.num_ctx_mla_kv_tokens
            if total_kv_tokens <= 0:
                raise RuntimeError("NVFP4 DSA context has no active KV tokens")
            head_dim = metadata.kv_cache_manager.head_dim
            # Static sparse FMHA may prefetch up to TopK rows even when a
            # short context has fewer valid KV tokens. The compaction itself
            # can select at most min(total KV tokens, query/TopK pairs), so a
            # late chunk with a long cached prefix does not need one fp8 row
            # for every token in that prefix.
            max_selected_tokens = min(total_kv_tokens, topk_indices.numel())
            scratch_capacity = max(max_selected_tokens, topk_indices.shape[1])
            # The previous layer's attention has already been enqueued on this
            # stream. Drop its Python reference before allocating the next
            # layer's scratch so the caching allocator can reuse that storage
            # in stream order instead of transiently keeping two full buffers.
            metadata.nvfp4_mla_context_fp8_scratch = None
            scratch = torch.empty(
                (scratch_capacity, 1, head_dim),
                dtype=torch.float8_e4m3fn,
                device=topk_indices.device,
            )
            compact_indices = torch.empty_like(topk_indices)
            torch.ops.trtllm.nvfp4_mla_context_kv_cache_gather(
                metadata.host_kv_cache_pool_pointers,
                metadata.host_kv_cache_pool_mapping,
                topk_indices,
                metadata._cached_req_idx_ctx,
                metadata._cached_block_table_ctx,
                metadata.ctx_kv_indptr[: metadata.num_contexts + 1],
                scratch,
                compact_indices,
                self._nvfp4_mla_kv_scale_quant_orig,
                local_layer_idx,
                total_kv_tokens,
                metadata._cached_tokens_per_block,
                page_index_scale * metadata._cached_tokens_per_block,
                layer_offset,
                metadata.kv_cache_manager.mla_kv_cache_residual_dim,
                metadata._cached_num_pool_tokens,
            )
            metadata.nvfp4_mla_context_fp8_scratch = scratch
            forward_args.sparse_runtime_params.aux_kv_cache_pool_ptr = scratch.data_ptr()
            return compact_indices, None

        if bridge is not None:
            topk_indices_global = dispatch.get_impl(bridge_path, "remap_topk_to_global")(
                self, topk_indices, metadata, bridge
            )
        else:
            topk_indices_global = self._remap_topk_to_global(topk_indices, metadata, is_generation)

        if kv_cache_dtype == tensorrt_llm.bindings.DataType.NVFP4:
            scratch = metadata.nvfp4_mla_fp8_scratch
            compact_indices = metadata.nvfp4_mla_compact_indices
            if scratch is None or compact_indices is None:
                raise RuntimeError("NVFP4 MLA scratch buffers were not allocated")
            num_rows = topk_indices_global.shape[0]
            if num_rows > scratch.shape[0] or num_rows > compact_indices.shape[0]:
                raise RuntimeError(
                    "NVFP4 MLA generation rows exceed the allocated MTP "
                    f"workspace: requested {num_rows}, capacity {scratch.shape[0]}"
                )
            scratch = scratch[:num_rows]
            compact_indices = compact_indices[:num_rows]
            torch.ops.trtllm.nvfp4_mla_kv_cache_gather(
                metadata.host_kv_cache_pool_pointers,
                metadata.host_kv_cache_pool_mapping,
                topk_indices_global,
                scratch,
                compact_indices,
                self._nvfp4_mla_kv_scale_quant_orig,
                local_layer_idx,
                metadata.kv_cache_manager.mla_kv_cache_residual_dim,
                metadata._cached_num_pool_tokens,
            )
            # Static sparse MLA treats indices as offsets from kvPtr. Feed it
            # compact offsets and the FP8 scratch base; TRTLLMGen is unchanged.
            forward_args.sparse_runtime_params.aux_kv_cache_pool_ptr = scratch.data_ptr()
            return compact_indices, None

        return topk_indices_global, None


class GlmIndexer(Indexer):
    """A full-indexer layer's DSA indexer, with the target's decode top-k region."""

    # The top-k bridge of the current ``GlmDsaBackend.sparse_attn_predict`` call (set only for
    # that call, on a decode step); None keeps the built-in copies.
    _topk_bridge = None

    @classmethod
    def adopt(cls, indexer: Indexer) -> "GlmIndexer":
        """Re-class a layer's built-in indexer.

        Raises:
            TypeError: if ``indexer`` is not exactly the built-in ``Indexer``.
        """
        reclass(indexer, Indexer, cls)
        return indexer

    def _copy_reused_topk(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        topk_indices_buffer: torch.Tensor,
        token_offset: int,
        num_gen_tokens: int,
        num_generations: int,
        has_prefill: bool,
        gvr_prior_indices: Optional[torch.Tensor],
    ) -> None:
        # rope_kv_append: the fused bridge reads the shared rows and writes this copy itself.
        bridge = self._topk_bridge
        if (
            bridge is not None
            and not has_prefill
            and num_gen_tokens == num_generations
            and dispatch.get_impl("rope_kv_append", "defer_reuse_copy")(
                bridge,
                topk_indices_buffer,
                token_offset,
                metadata.shared_topk_indices[:num_generations, :],
                gvr_prior_indices is not None,
            )
        ):
            return
        super()._copy_reused_topk(
            metadata,
            topk_indices_buffer,
            token_offset,
            num_gen_tokens,
            num_generations,
            has_prefill,
            gvr_prior_indices,
        )

    def _fill_decode_topk(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        topk_indices_buffer: torch.Tensor,
        token_offset: int,
        num_gen_tokens: int,
    ) -> bool:
        if dispatch.phase() is not dispatch.Phase.DECODE:
            return super()._fill_decode_topk(
                metadata=metadata,
                q_fp8=q_fp8,
                weights=weights,
                topk_indices_buffer=topk_indices_buffer,
                token_offset=token_offset,
                num_gen_tokens=num_gen_tokens,
            )
        out = topk_indices_buffer[token_offset : token_offset + num_gen_tokens, :]
        # The context-partitioned top-k reproduces the default selection only, so a
        # deterministic top-k goes to the indexer_topk entry.
        if not self.top_k.gvr_deterministic and region.cross_rank_topk_decode(
            self, metadata, q_fp8, weights, token_offset, num_gen_tokens, out, self._topk_bridge
        ):
            return True
        return region.indexer_topk_decode(
            self, metadata, q_fp8, weights, token_offset, num_gen_tokens, out
        )

    def _store_mtp_shared_rows(
        self,
        topk_indices_buffer: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        num_contexts: int,
        num_generations: int,
        num_gen_tokens: int,
        token_offset: int,
        has_prefill: bool,
        has_decode: bool,
        is_generation: Optional[bool],
    ) -> None:
        # rope_kv_append: the fused bridge stashes each request's last accepted row (the same
        # selection as ``_mtp_last_accepted_rows``).
        bridge = self._topk_bridge
        if bridge is not None and has_decode and not has_prefill and is_generation:
            num_accepted = metadata.mtp_num_accepted
            if num_accepted is not None:
                num_accepted = num_accepted[num_contexts : num_contexts + num_generations]
            if dispatch.get_impl("rope_kv_append", "defer_mtp_row_stash")(
                bridge,
                metadata.shared_topk_indices,
                num_contexts,
                num_generations,
                num_gen_tokens // num_generations,
                num_accepted,
            ):
                return
        super()._store_mtp_shared_rows(
            topk_indices_buffer,
            metadata,
            num_contexts,
            num_generations,
            num_gen_tokens,
            token_offset,
            has_prefill,
            has_decode,
            is_generation,
        )
