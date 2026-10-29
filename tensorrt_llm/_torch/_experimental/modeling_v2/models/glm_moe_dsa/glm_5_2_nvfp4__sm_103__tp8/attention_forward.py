# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MLA generation and DSA sparse-attention forward of the glm_moe_dsa sm_103 tp8 target on a decode step.

These are the built-in functions of ``attention/mla.py`` and ``attention/backends/sparse/dsa``
with the target's fast paths composed in:

- the absorbed-query packet: one producer runs the DSA projections and the absorbed query
  (``q_nope @ W_UK``) for a layer, so the generation skips its absorption BMM;
- the deferred-norm packet: the hidden state a post-MoE hop hands over before its RMSNorm,
  normalized inside the next layer's projection (or materialized when no producer serves);
- the fused generation rope, which also performs the deferred indexer K-cache scatter;
- the attention output tail, which takes over the W_UV BMM when ``GlmMLA.forward`` armed it.

``attention.GlmMLA`` and ``attention.GlmDsaHooks`` call them only on a decode step; every other
step runs the built-in functions. ``trtllm::glm_moe_dsa_mla_dsa_attn_inplace`` is the target's
copy of the built-in DSA op 2 (``trtllm::mla_dsa_attn_inplace``): the same op on this module's
``_forward_dsa_attn``.
"""

from typing import List, Optional

import torch

from tensorrt_llm._torch.attention.attention import extract_extra_attrs
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType, AttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.dsa.module import (
    forward_context_sparse_attn,
    forward_dsa_proj,
    forward_sparse_mla_kvcache_bf16,
    should_use_short_mha,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSABackendForwardArgs
from tensorrt_llm._torch.attention.mla import fp8_block_scaling_bmm_out
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version

from . import dispatch


def _absorbed_query_packet(self, position_ids, hidden_states, attn_metadata):
    """Return the absorbed-query packet of this layer's DSA projections, or None.

    The producers are tried in order; each decides its eligibility before any device work and
    returns None outside its domain, in which case the next one, and finally the built-in
    projection, runs:

    - a deferred-norm packet goes to the deferred-norm projection (the evict-hints variant when
      selected), which normalizes it inside the projection;
    - ``fused_kv_indexer_projection``: one projection launch for the fused [q_a | kv_a] and the
      indexer's wk | weights_proj outputs;
    - ``indexer_layer_front``: the query kernel of the full-indexer target layers and the draft
      layer (the same packet bytes);
    - ``absorbed_query`` (with ``shared_indexer_absorbed_query`` first on the shared-indexer
      layers);
    - ``draft_absorbed_query`` on the draft layer.
    """
    if dispatch.is_enabled("deferred_norm") and dispatch.get_impl("deferred_norm", "is_deferred")(
        hidden_states
    ):
        # The deferred-norm projection feeds the absorbed-query kernels; None makes the caller
        # materialize the packet.
        entry = (
            "deferred_norm_evict_hints"
            if dispatch.is_enabled("deferred_norm_evict_hints")
            else "deferred_norm"
        )
        return dispatch.get_impl(entry, "dsa_proj")(
            self, position_ids, hidden_states, attn_metadata
        )
    if dispatch.is_enabled("fused_kv_indexer_projection"):
        packet = dispatch.get_impl("fused_kv_indexer_projection", "dsa_proj")(
            self, position_ids, hidden_states, attn_metadata
        )
        if packet is not None:
            return packet
    if dispatch.is_enabled("indexer_layer_front"):
        # The indexer-front query kernel on the full-indexer target layers and the draft layer
        # (the same packet bytes); None runs the absorbed-query entries below.
        packet = dispatch.get_impl("indexer_layer_front", "dsa_proj")(
            self, position_ids, hidden_states, attn_metadata
        )
        if packet is not None:
            return packet
    if dispatch.is_enabled("absorbed_query"):
        if dispatch.is_enabled("shared_indexer_absorbed_query"):
            packet = dispatch.get_impl("absorbed_query", "shared_dsa_proj")(
                self, position_ids, hidden_states, attn_metadata
            )
            if packet is not None:
                return packet
        packet = dispatch.get_impl("absorbed_query", "dsa_proj")(
            self, position_ids, hidden_states, attn_metadata
        )
        if packet is not None:
            return packet
    if dispatch.is_enabled("draft_absorbed_query"):
        return dispatch.get_impl("absorbed_query", "mtp_dsa_proj")(
            self, position_ids, hidden_states, attn_metadata
        )
    return None


def _deferred_norm_materialize(hidden_states):
    """The normalized hidden state of a deferred-norm packet (bit-identical to the post-MoE
    hop's normalized output); any other input passes through."""
    if dispatch.is_enabled("deferred_norm") and dispatch.get_impl("deferred_norm", "is_deferred")(
        hidden_states
    ):
        return hidden_states.materialize()
    return hidden_states


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
    """Absorbed-MLA generation (``MLA.forward_absorption_generation``) on a decode step.

    ``absorbed_fused_q`` is the absorbed-query packet: a
    ``[num_tokens, num_heads_tp, kv_lora_rank + qk_rope_head_dim]`` buffer whose
    ``[..., :kv_lora_rank]`` already holds ``q_nope @ W_UK``. When given, only the absorption
    BMM is skipped; RoPE, cache append, FP8 query quantization and scheduler buffers run as
    usual. ``deferred_indexer_k`` is the indexer K-cache scatter ``_forward_dsa_attn`` deferred
    into this call's rope.
    """
    num_tokens = q.shape[0]
    q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
        [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
    # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
    num_seqs = attn_metadata.num_seqs

    # The rope kernel writes seqQOffset[batch_idx + 1] for every
    # batch_idx < batch_size, and CUB-scans total_s_len / seq_len entries
    # into seqKVOffsets (mlaKernels.cu:579, :650). `batch_size` there is the
    # count of presented rows, which equals num_seqs only while every
    # request contributes the same number of query tokens. Under the
    # token-major presentation a row is a token, so size these by the token
    # count. Uniform verification keeps the established request-major
    # allocation rather than scaling these temporary buffers by K.
    cu_seqlens_rows = (
        max(num_seqs, num_tokens) + 1
        if getattr(attn_metadata, "is_ragged_verify", False)
        else num_seqs + 1
    )
    cu_q_seqlens = torch.empty(cu_seqlens_rows, dtype=torch.int32, device=q.device)
    cu_kv_seqlens = torch.empty(cu_seqlens_rows, dtype=torch.int32, device=q.device)
    fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=q.device)
    use_fp8_mla = getattr(self.mqa, "has_fp8_kv_cache", False) or getattr(
        self.mqa, "has_fp4_kv_cache", False
    )

    mla_bmm1_scale = None
    mla_bmm2_scale = None
    quant_q_buffer = None
    if use_fp8_mla:
        mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=q.device)
        mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=q.device)
        quant_q_buffer = torch.empty(
            num_tokens,
            self.num_heads_tp,
            (self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=torch.uint8,
            device=q.device,
        )

    if hasattr(self, "k_b_proj_trans"):
        if absorbed_fused_q is not None:
            fused_q = absorbed_fused_q
        else:
            fused_q = torch.empty(
                [
                    num_tokens,
                    self.num_heads_tp,
                    (self.kv_lora_rank + self.qk_rope_head_dim),
                ],
                dtype=q.dtype,
                device=q.device,
            )

        fp4_mla = (
            self.sparse_params is None
            and self.mqa.support_fp4_kv_cache()
            and self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_fp4_kv_cache()
        )
        if fp4_mla and latent_cache is None:
            raise RuntimeError(
                "FP4 MLA generation requires a latent cache for fused "
                "Q quantization, RoPE, and cache update."
            )
        fuse_fp4_q_quant = fp4_mla
        fp4_rope_kwargs = {"fuse_fp4_q_quant": True} if fuse_fp4_q_quant else {}

        def _mla_gen_rope():
            # rope_kv_append: one fused launch writes the bytes of mla_rope_generation plus the
            # deferred indexer K-cache scatter. Every check runs before any device work;
            # otherwise the deferred scatter runs natively first and the native path follows.
            if deferred_indexer_k is not None or dispatch.is_enabled("rope_kv_append"):
                rope_fused = (
                    not self.apply_rotary_emb
                    and self.kv_cache_dtype != "fp8_ds_mla"
                    and not fp4_rope_kwargs
                    and dispatch.get_impl("rope_kv_append", "rope_eligible")(
                        self.mqa,
                        attn_metadata,
                        fused_q,
                        q_pe,
                        latent_cache,
                        quant_q_buffer,
                        mla_bmm1_scale,
                        mla_bmm2_scale,
                    )
                )
                if rope_fused:
                    # indexer_layer_front: the same fused rope, forked to trigger its
                    # dependents after its last read (the same bytes).
                    rope_owner = "rope_kv_append"
                    if dispatch.is_enabled("indexer_layer_front"):
                        front_rope = dispatch.get_impl("indexer_layer_front", "rope_available")()
                        dispatch.log_capability(
                            "indexer_layer_front",
                            "rope",
                            front_rope,
                            "" if front_rope else "indexer_layer_front rope op not loaded",
                        )
                        if front_rope:
                            rope_owner = "indexer_layer_front"
                    dispatch.get_impl(rope_owner, "mla_rope_generation")(
                        self.mqa,
                        fused_q,
                        q_pe,
                        latent_cache,
                        attn_metadata,
                        cu_q_seqlens,
                        cu_kv_seqlens,
                        fmha_scheduler_counter,
                        mla_bmm1_scale,
                        mla_bmm2_scale,
                        quant_q_buffer,
                        indexer_k=deferred_indexer_k,
                    )
                    return
                if deferred_indexer_k is not None:
                    dispatch.get_impl("rope_kv_append", "run_indexer_k_scatter_native")(
                        deferred_indexer_k
                    )
            if self.apply_rotary_emb:
                # Non-fused backends (Vanilla / FlashInfer) do not fuse RoPE
                # in the attention kernel. Reuse apply_rope, which rotates
                # q's q_pe slice in place, then copy rotated k_pe into the
                # latent cache that the backend appends.
                assert position_ids is not None
                assert latent_cache is not None
                gen_position_ids = position_ids[..., attn_metadata.num_ctx_tokens :]
                k_pe_rope = self.apply_rope(q, k_pe, gen_position_ids)
                fused_q[..., self.kv_lora_rank :] = q_pe
                latent_cache[..., self.kv_lora_rank :] = k_pe_rope
            else:
                # Fused backend (TRTLLM): RoPE, latent-cache append and the
                # trtllm-gen scheduler buffers are all produced in-kernel.
                if self.kv_cache_dtype == "fp8_ds_mla":
                    fused_q[..., self.kv_lora_rank :] = q_pe
                else:
                    self.mqa.mla_rope_generation(
                        fused_q,
                        q_pe,
                        latent_cache,
                        attn_metadata,
                        cu_q_seqlens,
                        cu_kv_seqlens,
                        fmha_scheduler_counter,
                        mla_bmm1_scale,
                        mla_bmm2_scale,
                        quant_q_buffer,
                        **fp4_rope_kwargs,
                    )

        if fuse_fp4_q_quant or use_fp8_mla:
            rope_stream = None
        else:
            rope_stream = self.aux_stream
        if absorbed_fused_q is not None:
            _mla_gen_rope()
        elif self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            maybe_execute_in_parallel(
                lambda: self._bmm_bf16_out(
                    q_nope_t,
                    self.k_b_proj_trans,
                    self.k_b_proj_trans.transpose(1, 2),
                    q_nope_out,
                    self._locality_domain_k_b_proj_trans_shards,
                ),
                _mla_gen_rope,
                self.ln_events[0],
                self.ln_events[1],
                rope_stream,
            )

        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

            maybe_execute_in_parallel(
                lambda: fp8_block_scaling_bmm_out(
                    q_nope,
                    self.k_b_proj_trans,
                    self.k_b_proj_trans_scale,
                    q_nope_out,
                    self.k_b_proj_trans_dequant,
                    self.use_cute_dsl_blockscaling_bmm,
                ),
                _mla_gen_rope,
                self.ln_events[0],
                self.ln_events[1],
                rope_stream,
            )
        else:
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        fused_q = fused_q.view(
            [
                num_tokens,
                self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
            ]
        )
    else:
        raise RuntimeError("MLA absorption requires k_b_proj_trans")

    # attention_output_tail_prefetch: on a call that GlmMLA.forward armed, fork the L2
    # prefetch of the tail weights onto a private stream while the attention runs; the
    # projection step joins it. No-op otherwise (decided before any device work).
    attn_tail = getattr(self, "_attn_tail", None)
    if getattr(self, "_tail_prefetch", False) and attn_tail is not None:
        attn_tail.before_attention(self, point="attn")

    attn_out_latent = self._attn_forward_gen(
        self.mqa,
        fused_q,
        None,
        None,
        position_ids,
        attn_metadata,
        attention_input_type=AttentionInputType.generation_only,
        out_scale=self.out_scale,
        latent_cache=latent_cache,  # kvcache and k_pe
        q_pe=q_pe,  # used by `invokeMLARopeGeneration`
        cu_q_seqlens=cu_q_seqlens,  # used by `mlaGeneration`
        cu_kv_seqlens=cu_kv_seqlens,  # used by `mlaGeneration`
        fmha_scheduler_counter=fmha_scheduler_counter,  # used by `mlaGeneration`
        mla_bmm1_scale=mla_bmm1_scale,  # used by `mlaGeneration`
        mla_bmm2_scale=mla_bmm2_scale,  # used by `mlaGeneration`
        quant_q_buffer=quant_q_buffer,  # used by `mlaGeneration`
        **kwargs,
    )
    fused_q = None

    # note: if we do not have CP, then num_heads_tp_cp == num_heads_tp
    assert (
        attn_out_latent.shape[0] == q.shape[0]
        and attn_out_latent.shape[1] == self.num_heads_tp_cp * self.kv_lora_rank
    )

    # [seq, num_heads, kv_lora_rank]
    attn_out_latent = attn_out_latent.view([-1, self.num_heads_tp_cp, self.kv_lora_rank])

    # Acts only on a call that GlmMLA.forward armed; when the attention tail takes the call,
    # its projection step consumes the latent and the native W_UV BMM is skipped. The latent
    # shape is checked before any device work.
    attn_tail = getattr(self, "_attn_tail", None)
    if attn_tail is not None and attn_tail.defer_w_uv(self, attn_metadata, attn_out_latent, output):
        return output

    attn_output = output.view([num_tokens, self.num_heads_tp_cp, self.v_head_dim])

    if self.v_b_proj.dtype == torch.bfloat16:
        # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
        # -> [num_heads, seq, v_head_dim]
        self._bmm_bf16_out(
            attn_out_latent.transpose(0, 1),
            self.v_b_proj,
            self.v_b_proj.transpose(1, 2),
            attn_output.transpose(0, 1),
            self._locality_domain_v_b_proj_shards,
        )
    elif self.v_b_proj.dtype == torch.float8_e4m3fn:
        fp8_block_scaling_bmm_out(
            attn_out_latent,
            self.v_b_proj,
            self.v_b_proj_scale,
            attn_output.transpose(0, 1),
            self.v_b_proj_dequant,
            self.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

    return output


def forward_generation_sparse_attn(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    latent_cache: Optional[torch.Tensor] = None,
    indexer_intermediates: Optional[List[torch.Tensor]] = None,
    absorbed_fused_q: Optional[torch.Tensor] = None,
    deferred_indexer_k=None,
) -> torch.Tensor:
    if get_sm_version() >= 100:
        return self.forward_absorption_generation(
            q,
            compressed_kv,
            k_pe,
            attn_metadata,
            output,
            position_ids=position_ids,
            latent_cache=latent_cache,
            absorbed_fused_q=absorbed_fused_q,
            deferred_indexer_k=deferred_indexer_k,
            sparse_backend_args=DSABackendForwardArgs(indexer_intermediates=indexer_intermediates),
        )
    assert absorbed_fused_q is None, "absorbed_query preabsorbed queries require SM100+"
    assert deferred_indexer_k is None
    return forward_sparse_mla_kvcache_bf16(
        self,
        q,
        latent_cache,
        attn_metadata,
        output,
        indexer_intermediates,
        is_generation=True,
    )


def forward_sparse_attn(
    self,
    position_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    attn_metadata: AttentionMetadata,
    attn_output: list[torch.Tensor],
) -> None:
    """
    Forward pass for the MLA module with DSA (always in MQA mode).
    Writes result into output tensor in-place.

    Delegates to forward_dsa_proj (token-wise projections) followed by
    _forward_dsa_attn (batch-dependent attention dispatch). An absorbed-query packet replaces
    forward_dsa_proj; a deferred-norm packet that no producer takes is materialized first.

    Args:
        position_ids (Optional[torch.IntTensor]): The position IDs.
        hidden_states (torch.Tensor): The hidden states.
        attn_metadata (AttentionMetadata): The attention metadata.
        attn_output: Algorithm-defined output tensors. DSA writes to the
            standard output at index zero.
    """
    output = attn_output[0]
    query_packet = _absorbed_query_packet(self, position_ids, hidden_states, attn_metadata)
    if query_packet is None:
        hidden_states = _deferred_norm_materialize(hidden_states)
        proj_outputs = forward_dsa_proj(self, position_ids, hidden_states, attn_metadata)
        absorbed_fused_q = None
    else:
        proj_outputs, absorbed_fused_q = query_packet.proj_outputs, query_packet.fused_q
    q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
    indexer_intermediates = proj_outputs[4:]
    _forward_dsa_attn(
        self,
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        indexer_intermediates,
        position_ids,
        attn_metadata,
        output,
        absorbed_fused_q=absorbed_fused_q,
    )


def forward_sparse_attn_custom_op(
    self,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    attn_output: list[torch.Tensor],
    latent_cache_gen: Optional[torch.Tensor],
) -> None:
    """Run DSA's two-stage custom ops for piecewise CUDA graph capture.

    The second op is the target's ``trtllm::glm_moe_dsa_mla_dsa_attn_inplace``. With an
    absorbed-query packet the attention runs directly, without the ops.
    """
    del latent_cache_gen
    from tensorrt_llm._torch.attention.backends.sparse.dsa import custom_ops  # noqa: F401

    if (
        dispatch.is_enabled("absorbed_query") or dispatch.is_enabled("draft_absorbed_query")
    ) and not isinstance(hidden_states, Fp4QuantizedTensor):
        metadata, _ = extract_extra_attrs(self.layer_idx_str, "mla")
        query_packet = _absorbed_query_packet(self, position_ids, hidden_states, metadata)
        if query_packet is not None:
            # The packet producers decline under torch.compile and breakable CUDA graphs, so
            # the two-op split is not needed on this path.
            proj_outputs = query_packet.proj_outputs
            _forward_dsa_attn(
                self,
                *proj_outputs[:4],
                proj_outputs[4:],
                position_ids,
                metadata,
                attn_output[0],
                absorbed_fused_q=query_packet.fused_q,
            )
            return
        hidden_states = _deferred_norm_materialize(hidden_states)

    if isinstance(hidden_states, Fp4QuantizedTensor):
        proj_outputs = torch.ops.trtllm.mla_dsa_proj(
            hidden_states.unquantized_hidden_states,
            position_ids,
            self.layer_idx_str,
            hidden_states.fp4_tensor,
            hidden_states.scaling_factor,
        )
    else:
        proj_outputs = torch.ops.trtllm.mla_dsa_proj(
            hidden_states, position_ids, self.layer_idx_str
        )
    q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
    indexer_intermediates = proj_outputs[4:]
    maybe_bcg_glm_moe_dsa_mla_dsa_attn_inplace(
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        indexer_intermediates,
        position_ids,
        self.layer_idx_str,
        attn_output[0],
    )


def _forward_dsa_attn(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    absorbed_fused_q: Optional[torch.Tensor] = None,
) -> None:
    """Batch-structure-dependent attention for DSA MLA (Op 2, not graph-captured).

    indexer_intermediates is [q_fp8, k_fp8, k_scale, weights, q_scale]
    when the indexer ran in Op 1, or [] when short-MHA handled all tokens.

    All num_tokens slicing happens here (not in Op 1) because
    num_tokens comes from batch-specific metadata and must not be
    baked into CUDA graph capture.
    """
    num_contexts = attn_metadata.num_contexts
    num_generations = attn_metadata.num_generations
    num_ctx_tokens = attn_metadata.num_ctx_tokens
    num_tokens = attn_metadata.num_tokens

    # Slice Op 1 outputs to actual num_tokens (Op 1 operates on the
    # full padded tensor for CUDA graph compatibility).
    q = q[:num_tokens, ...]
    compressed_kv = compressed_kv[:num_tokens, ...]
    k_pe = k_pe[:num_tokens, ...]
    latent_cache = latent_cache[:num_tokens, ...]
    if position_ids is not None:
        position_ids = position_ids[..., :num_tokens]

    use_short_mha_for_ctx = num_contexts > 0 and should_use_short_mha(
        self, attn_metadata, position_ids
    )

    deferred_indexer_k = None
    if not (use_short_mha_for_ctx and num_generations == 0) and self.mqa.indexer is not None:
        q_fp8, k_fp8, k_scale, weights, q_scale = indexer_intermediates
        # Slice indexer intermediates to actual num_tokens (they were
        # computed on the full padded tensor in Op 1).
        q_fp8 = q_fp8[:num_tokens, ...]
        k_fp8 = k_fp8[:num_tokens, ...]
        k_scale = k_scale[:num_tokens, ...]
        weights = weights[:num_tokens, ...]
        q_scale = q_scale[:num_tokens, ...]
        indexer_intermediates = [q_fp8, k_fp8, k_scale, weights, q_scale]
        # Update the full mixed batch once before context/generation prediction
        # splits. Context chunk gathers must observe the newly appended keys.
        # rope_kv_append: a pure-generation batch folds this scatter into the same layer's
        # fused generation rope, which still runs before the indexer reads the K cache. The
        # deferred scatter must reach the generation call below, which is checked right after
        # that call.
        if dispatch.is_enabled("rope_kv_append"):
            deferred_indexer_k = dispatch.get_impl("rope_kv_append", "defer_indexer_k_scatter")(
                self, k_fp8, k_scale, attn_metadata, forwards_packet=True
            )
        if deferred_indexer_k is None:
            self.mqa.indexer._update_k_cache(k_fp8, k_scale, attn_metadata)
    else:
        indexer_intermediates = None

    assert output is not None, "output must be provided"

    if num_contexts > 0:
        q_ctx = q[:num_ctx_tokens, ...]
        compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
        k_pe_ctx = k_pe[:num_ctx_tokens, ...]
        latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
        ctx_position_ids = position_ids[..., :num_ctx_tokens] if position_ids is not None else None
        if self.apply_rotary_emb:
            assert ctx_position_ids is not None
            k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, ctx_position_ids)

        forward_context_sparse_attn(
            self,
            q_ctx,
            compressed_kv_ctx,
            k_pe_ctx,
            attn_metadata,
            output[:num_ctx_tokens, :],
            latent_cache_ctx,
            position_ids=ctx_position_ids,
            indexer_intermediates=indexer_intermediates,
        )

    if num_generations > 0:
        q_gen = q[num_ctx_tokens:, ...]
        compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
        k_pe_gen = k_pe[num_ctx_tokens:, ...]
        latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
        gen_position_ids = (
            position_ids[..., num_ctx_tokens:num_tokens] if position_ids is not None else None
        )
        if self.apply_rotary_emb:
            assert gen_position_ids is not None
            k_pe_gen = self.apply_rope(q_gen, k_pe_gen, gen_position_ids)

        forward_generation_sparse_attn(
            self,
            q_gen,
            compressed_kv_gen,
            k_pe_gen,
            attn_metadata,
            output[num_ctx_tokens:num_tokens, :],
            latent_cache=latent_cache_gen,
            position_ids=gen_position_ids,
            indexer_intermediates=indexer_intermediates,
            absorbed_fused_q=(
                absorbed_fused_q[num_ctx_tokens:num_tokens]
                if absorbed_fused_q is not None
                else None
            ),
            deferred_indexer_k=deferred_indexer_k,
        )
    if deferred_indexer_k is not None:
        # Raises unless the generation rope (fused or the native fallback) executed the
        # deferred indexer K-cache scatter.
        dispatch.get_impl("rope_kv_append", "check_indexer_k_scatter_consumed")(deferred_indexer_k)

    # Join the prev_topk copy forked in sparse_attn_indexer; CUDA graph
    # capture requires the join within the same layer's forward.
    if self.mqa.indexer is not None:
        self.mqa.indexer.maybe_join_prev_topk_copy()


@torch.library.custom_op("trtllm::glm_moe_dsa_mla_dsa_attn_inplace", mutates_args=("output",))
def glm_moe_dsa_mla_dsa_attn_inplace(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Batch-structure-dependent attention dispatch for DSA MLA, on ``_forward_dsa_attn`` above.

    The built-in ``trtllm::mla_dsa_attn_inplace`` with the target's op-2 function:
    indexer_intermediates is [q_fp8, k_fp8, k_scale, weights, q_scale] when the indexer ran in
    Op 1, or [] when short-MHA handled all tokens. This op is excluded from CUDA graph capture.
    """
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    _forward_dsa_attn(
        mla_layer,
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        indexer_intermediates,
        position_ids,
        metadata,
        output,
    )


@glm_moe_dsa_mla_dsa_attn_inplace.register_fake
def _glm_moe_dsa_mla_dsa_attn_inplace_fake(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Model the in-place output mutation during fake-tensor propagation."""


maybe_bcg_glm_moe_dsa_mla_dsa_attn_inplace = eager_on_graph(glm_moe_dsa_mla_dsa_attn_inplace)
