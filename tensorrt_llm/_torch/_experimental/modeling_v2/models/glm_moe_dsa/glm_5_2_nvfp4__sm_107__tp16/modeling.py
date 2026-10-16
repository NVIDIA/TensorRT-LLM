# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2-NVFP4 on 16 Rubin GPUs (sm107), tp16: model construction and forward.

This target serves the generation side of a disaggregated deployment. Its
context server runs the built-in GLM-5.2 implementation (modeling_deepseekv3)
and hands over the MLA KV cache and the DSA indexer K cache it wrote, so this
target builds the same modules and loads the same weights: the caches it reads
are laid out exactly as they were written. Forwards that carry context rows
(engine warm-up, an aggregated deployment) run that same decoder stack in
``PrefillTarget``. Generation-only forwards -- every serving step of the
generation server -- run ``DecodeTarget``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.module import (
    forward_generation_sparse_attn,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSABackendForwardArgs
from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3ForCausalLM, DeepseekV3Model
from tensorrt_llm._torch.models.modeling_speculative import (
    SpecDecOneEngineForCausalLM,
    _slice_spec_position_ids,
)
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.kernel_inputs import (
    extract_routing_params,
    get_data_or_none,
    to_trtllm_gen_act_type,
)
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._torch.utils import EventType

from . import weights as _weights
from .ops.attention.mla_kv_half_append import mla_kv_half_append
from .ops.attention.sparse_mla_flash_decode import sparse_mla_flash_decode
from .ops.comm._mnnvl_workspace import (
    LamportExchangeWorkspace,
    hop_supported,
    hop_workspace,
    publish_target,
)
from .ops.comm.mnnvl_allgather import mnnvl_allgather
from .ops.comm.mnnvl_hop import mnnvl_hop
from .ops.comm.mnnvl_moe_finalize_hop import mnnvl_moe_finalize_hop
from .ops.dsa._fp8_paged_mqa_logits import num_sms
from .ops.dsa._hist_topk import (
    MAX_SHARD,
    TOPK,
    get_hist_topk_workspace,
    hist_topk_supported,
    shard_capacity,
)
from .ops.dsa.fp8_paged_mqa_logits import fp8_paged_mqa_logits
from .ops.dsa.hist_shard_meta import hist_shard_meta
from .ops.dsa.hist_topk import hist_topk
from .ops.dsa.indexer_k_tail import indexer_k_tail
from .ops.dsa.paged_mqa_logits_schedule import paged_mqa_logits_schedule
from .ops.embedding.fused_embedding import fused_embedding
from .ops.gemm.dense_mlp_small_m import INTER_PER_RANK, dense_mlp_small_m
from .ops.gemm.fused_a_gemm import INDEXER_OUT_FEATURES as FUSED_A_INDEXER_OUT_FEATURES
from .ops.gemm.fused_a_gemm import MAX_TOKENS as FUSED_A_MAX_TOKENS
from .ops.gemm.fused_a_gemm import OUT_FEATURES as FUSED_A_OUT_FEATURES
from .ops.gemm.fused_a_gemm import fused_a_gemm
from .ops.gemm.indexer_q_tail import indexer_q_tail
from .ops.gemm.low_m_gemm import low_m_gemm
from .ops.gemm.o_proj_publish import o_proj_publish
from .ops.gemm.q_b_fp8_rope import q_b_fp8_rope
from .ops.moe.routed_experts import (
    FUSED_ROUTER_MAX_TOKENS,
    make_fused_router_workspace,
    routed_experts,
)

# The deployment this target is written for; route() selects it on the same
# facts, and the constructor states them again for a direct construction.
_COMPUTE_CAPABILITY = (10, 7)
_TP = 16
# One Lamport buffer of the verify logits' all-gather: every rank's BF16 vocabulary shard of
# up to 8 rows, with room to spare.
_LOGITS_EXCHANGE_BYTES = 16 << 20


class PrefillTarget:
    """Forwards that carry context rows: the decoder stack the context server runs."""

    def __init__(self, core: "Glm52Nvfp4Sm107Tp16Core") -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        return DeepseekV3Model.forward(
            self.core,
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )


class DecodeTarget:
    """Generation-only forwards: the serving step of the generation server.

    Routed to only when the batch has no context rows, which the core decides
    per step; nothing in this class reads the phase back.

    A step of up to 8 rows (one request and its 7 draft tokens at c1) runs on
    the target's kernels, layer by layer:

    - The embedding is one lookup in the whole BF16 table every rank holds.
    - Attention: q_a, the compressed KV and its rope part come from one GEMM over
      the stored kv_a_proj_with_mqa weight. q_b, with k_b absorbed and the q_a
      norm folded, writes the RoPE'd FP8 query in its epilogue, while on the aux
      stream one kernel normalizes, ropes and appends the KV half to the FP8
      cache. The flash decode attends over the 2048 selected rows at 4 heads with
      v_b folded in, and o_proj's epilogue publishes this rank's shard into the
      all-reduce workspace.
    - The DSA indexer (on the layers that run one) shares fused-A: its K and
      weight rows are fused-A's last 160 columns. Its Q comes from wq_b (q_a norm
      folded) with an epilogue that ropes, quantizes and scales; its K tail
      normalizes, ropes, quantizes and writes the indexer K cache on a side
      stream. Each rank scores its 1/16 shard of the context's keys, and the ranks
      agree on the exact top-2048 by exchanging histograms. The layers without an
      indexer reuse the last selection, which the flash decode maps to cache rows
      itself and starts gathering before its wait.
    - Every all-reduce is a one-shot MNNVL hop that adds the residual and applies
      the following RMSNorm: post_attention_layernorm after attention, and after
      the MLP or MoE the next layer's input_layernorm (the model's final norm
      after the last layer).
    - The dense layers 0-2 run this rank's 768 intermediate columns as two
      weight-streaming kernels.
    - The MoE layers run the shared expert as two split-K GEMMs around SwiGLU on
      the aux stream, and the routed experts as one trtllm-gen call that runs the
      router GEMV in its routing kernel, joins the NVFP4 input (quantized on its
      own stream) right before FC1, runs FC2 on the K = 128 kernel and leaves the
      finalize step to the post-MoE hop.

    The verify logits come from the LM head on this rank's vocabulary shard
    (split-K GEMM) and one MNNVL all-gather that widens them to FP32.

    A larger batch runs the built-in modules in the same order.
    """

    def __init__(self, core: "Glm52Nvfp4Sm107Tp16Core") -> None:
        self.core = core
        config = core.model_config.pretrained_config
        self.mapping = core.model_config.mapping
        self.hidden_size = config.hidden_size
        self.index_topk = config.index_topk
        layers = core.layers[: core.num_hidden_layers]
        for layer in layers:
            assert layer.allreduce is not None and layer.next_layer_layernorm is not None
            self._check_attention(layer)
            if isinstance(layer.mlp, GatedMLP):
                assert (
                    layer.mlp_tp_size == _TP
                    and not layer.disable_attn_allreduce
                    and not layer.fusion_config.PRE_MLP_FUSION
                    and layer.fusion_config.POST_MLP_FUSION
                    and layer.post_attention_layernorm.nvfp4_scale is None
                ), f"layer {layer.layer_idx}: unexpected dense-layer configuration"
            else:
                assert (
                    layer.disable_attn_allreduce
                    and layer.fusion_config.PRE_MOE_FUSION
                    and layer.fusion_config.POST_MOE_FUSION
                    and layer.mapping.is_multi_node()
                ), f"layer {layer.layer_idx}: unexpected MoE-layer configuration"
                self._check_shared_expert(layer)
                backend = getattr(layer.mlp.experts, "backend", layer.mlp.experts)
                assert (
                    backend.has_nvfp4
                    and backend.scaling_vector_size == 16
                    and getattr(backend, "fc31_act_scale", None) is None
                    and not backend.bias
                    and backend.w3_w1_weight.shape[-1] * 2 == config.hidden_size
                ), f"layer {layer.layer_idx}: routed experts are not unpadded TRTLLM-Gen NVFP4"

        # The attention operands the kernels read, per layer index (_fold_attention_weights).
        self._q_b_absorbed = {}
        self._fused_a_indexer_weight = {}
        self._wq_b_folded = {}
        self._l2_after_fused_a = {}
        for layer in layers:
            self._fold_attention_weights(layer)
        # The key-sharded top-k exchanges histograms over the TP group's MNNVL workspace
        # (created here, on every rank in the same order).
        self._hist_workspace = (
            get_hist_topk_workspace(self.mapping) if hist_topk_supported(self.mapping) else None
        )
        self._hist_buffers = None
        self._logits_workspace = LamportExchangeWorkspace(self.mapping, _LOGITS_EXCHANGE_BYTES)
        self._k_tail_stream = torch.cuda.Stream()
        self._k_tail_fork = torch.cuda.Event()
        self._k_tail_done = torch.cuda.Event()

        device = core.embed_tokens.weight.device
        # The attention scales the KV-half kernel writes and the flash decode reads.
        self._bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
        self._bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
        self._unit_scale = torch.ones(1, dtype=torch.float32, device=device)
        self._kv_pool = None

        moe_layers = [layer for layer in layers if not isinstance(layer.mlp, GatedMLP)]
        # One exchange workspace per router call site: its launches must run in stream order.
        self._router_workspace = {
            layer.layer_idx: make_fused_router_workspace(device) for layer in moe_layers
        }
        # The routing kernel runs the router GEMV, so the logits it is handed only give the
        # shape; autotuner profiling routes from them, so they hold random values.
        self._router_logits = torch.randn(
            FUSED_ROUTER_MAX_TOKENS, config.n_routed_experts, device=device
        )
        self._quant_stream = torch.cuda.Stream()
        self._quant_fork = torch.cuda.Event()
        self._quant_ready = torch.cuda.Event()
        self._fast_rows = {}

    def _check_attention(self, layer) -> None:
        """Assert the layer's MLA weights and KV cache are the layout the kernels take."""
        mla = layer.self_attn
        assert not mla.apply_rotary_emb and not mla.mapping.has_cp_helix()
        assert tuple(mla.kv_a_proj_with_mqa.weight.shape) == (
            FUSED_A_OUT_FEATURES,
            self.hidden_size,
        ) and mla.kv_a_proj_with_mqa.weight.dtype == torch.bfloat16, (
            f"layer {layer.layer_idx}: kv_a_proj_with_mqa is not the BF16 [q_a; kv_a] weight"
        )
        assert (
            getattr(mla.mqa, "has_fp8_kv_cache", False)
            and mla.kv_cache_dtype != "fp8_ds_mla"
            and mla.k_b_proj_trans.dtype == torch.bfloat16
            and mla.v_b_proj.dtype == torch.bfloat16
            and mla.v_b_proj.is_contiguous()
            and mla.q_b_proj.weight.dtype == torch.bfloat16
            and mla.o_proj.weight.dtype == torch.bfloat16
        ), f"layer {layer.layer_idx}: unexpected MLA weights or KV cache layout"
        assert layer.input_layernorm.nvfp4_scale is None, "kv_a_proj is not NVFP4"

    @staticmethod
    def _check_shared_expert(layer) -> None:
        """Assert the MoE layer's shared expert is the plain BF16 SwiGLU MLP the kernels run."""
        shared = layer.mlp.shared_experts
        assert (
            shared is not None
            and layer.mlp.shared_output_scale is None
            and shared._is_plain_swiglu()
            and shared.swiglu_limit is None
            and shared.gate_up_proj.weight.dtype == torch.bfloat16
            and shared.down_proj.weight.dtype == torch.bfloat16
        ), f"layer {layer.layer_idx}: the shared expert is not a plain BF16 SwiGLU MLP"

    def _fold_attention_weights(self, layer) -> None:
        """The layer's attention operands in the forms the kernels read, by layer index."""
        mla = layer.self_attn
        index = layer.layer_idx
        # q_b with k_b absorbed and the q_a norm weight folded: for head h, rows
        # [h * 576, h * 576 + 512) are k_b_trans[h] @ W_q_b[h, :qk_nope] and the next 64 rows
        # W_q_b[h, qk_nope:], every column times gamma_q_a; FP64, rounded to BF16 once.
        qk_head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim
        w = mla.q_b_proj.weight.view(mla.num_heads_tp, qk_head_dim, mla.q_lora_rank).double()
        nope = torch.bmm(mla.k_b_proj_trans.double(), w[:, : mla.qk_nope_head_dim])
        absorbed = torch.cat([nope, w[:, mla.qk_nope_head_dim :]], 1)
        absorbed = absorbed * mla.q_a_layernorm.weight.double()
        self._q_b_absorbed[index] = (
            absorbed.to(torch.bfloat16).reshape(-1, mla.q_lora_rank).contiguous()
        )

        # A layer that runs the indexer gets its fused-A weight: kv_a_proj_with_mqa's rows,
        # then the indexer's wk and weights_proj rows (BF16), in one tensor whose first rows
        # the built-in kv_a_proj and the indexer's projections then view, so nothing is stored
        # twice. Its indexer Q comes from wq_b with the q_a norm weight folded into its columns.
        indexer = mla.mqa.indexer
        if indexer is not None:
            assert (
                indexer.wk.weight.dtype == torch.bfloat16
                and indexer.weights_proj.weight.dtype == torch.bfloat16
            ), "the indexer's wk and weights_proj must be BF16 (TRTLLM_DSA_INDEXER_BF16=1)"
            kv_a = mla.kv_a_proj_with_mqa.weight
            wk = indexer.wk.weight
            extended = torch.cat([kv_a, wk, indexer.weights_proj.weight]).contiguous()
            assert extended.shape[0] == FUSED_A_INDEXER_OUT_FEATURES
            mla.kv_a_proj_with_mqa.weight.data = extended[: kv_a.shape[0]]
            indexer.wk.weight.data = extended[kv_a.shape[0] : kv_a.shape[0] + wk.shape[0]]
            indexer.weights_proj.weight.data = extended[kv_a.shape[0] + wk.shape[0] :]
            self._fused_a_indexer_weight[index] = extended
            self._wq_b_folded[index] = (
                indexer.wq_b.weight.double() * mla.q_a_layernorm.weight.double()
            ).to(torch.bfloat16)

        # What fused-A prefetches into L2 while it waits for its input: the weights of the
        # attention GEMMs that follow it.
        prefetch = [self._q_b_absorbed[index], mla.v_b_proj, mla.o_proj.weight]
        if index in self._wq_b_folded:
            prefetch.insert(1, self._wq_b_folded[index])
        self._l2_after_fused_a[index] = [t for t in prefetch if t.is_contiguous()]

    def _fast(self, rows: int) -> bool:
        """Whether every kernel of the step serves this many rows (decided once per shape)."""
        fast = self._fast_rows.get(rows)
        if fast is None:
            fast = (
                rows <= FUSED_A_MAX_TOKENS
                and rows <= FUSED_ROUTER_MAX_TOKENS
                and bool(
                    torch.ops.trtllm.glm_dense_mlp_small_m_supported(
                        rows, self.hidden_size, INTER_PER_RANK
                    )
                )
                and hop_supported(rows, self.hidden_size, self.mapping.tp_size)
            )
            self._fast_rows[rows] = fast
        return fast

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        layers = core.layers[: core.num_hidden_layers]
        if inputs_embeds is None:
            inputs_embeds = fused_embedding(input_ids, core.embed_tokens.weight)
        residual = inputs_embeds
        hidden_states = layers[0].input_layernorm(inputs_embeds)
        if not self._fast(hidden_states.shape[0]):
            for layer in layers:
                hidden_states, residual = self._builtin_layer(
                    layer, position_ids, hidden_states, residual, attn_metadata
                )
            return hidden_states
        sharded = self._shard_indexer_keys(attn_metadata)
        for layer in layers:
            attn_out = self._attention(layer, position_ids, hidden_states, attn_metadata, sharded)
            hidden_states, residual = self._post_attention_hop(layer, attn_out, residual)
            if isinstance(layer.mlp, GatedMLP):
                mlp = layer.mlp
                # This rank's partial sum; the hop adds the 16 ranks'.
                partial = dense_mlp_small_m(
                    hidden_states, mlp.gate_up_proj.weight, mlp.down_proj.weight
                )
                hidden_states, residual = self._hop(layer.next_layer_layernorm, partial, residual)
            else:
                hidden_states, residual = self._moe(layer, hidden_states, residual)
        return hidden_states

    def logits(self, lm_head, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """FP32 logits over the whole vocabulary of the rows, or None for a batch the kernels
        do not serve: this rank's vocabulary shard from the LM head, then the all-gather."""
        rows = hidden_states.shape[0]
        weight = lm_head.weight
        if not self._fast(rows) or weight.shape[0] * self.mapping.tp_size != lm_head.num_embeddings:
            return None
        shard = low_m_gemm(
            hidden_states.contiguous(), weight, hidden_states.new_empty(rows, weight.shape[0])
        )
        return mnnvl_allgather(
            shard,
            self._logits_workspace.comm_buffer,
            self._logits_workspace.buffer_flags,
            self.mapping.tp_size,
        )

    def _shard_indexer_keys(self, attn_metadata) -> bool:
        """Point this rank's indexer at its 1/16 shard of the request's keys for this step:
        the shard's length and block table, and the logits kernel's schedule over it.

        Applies to one generation request whose shard fits the top-k kernel; returns whether it
        did, and otherwise every indexer layer scores the whole context with the built-in
        indexer."""
        manager = attn_metadata.kv_cache_manager
        if self._hist_workspace is None or attn_metadata.num_seqs != 1:
            return False
        if self._hist_buffers is None:
            capacity = shard_capacity(manager.max_seq_len, self.mapping.tp_size)
            if capacity > MAX_SHARD:
                return False
            device = self._unit_scale.device
            tokens_per_block = manager.tokens_per_block
            rows = FUSED_A_MAX_TOKENS
            sms = num_sms(device.index)
            self._hist_buffers = dict(
                capacity=capacity,
                shard_ctx=torch.zeros(1, dtype=torch.int32, device=device),
                shard_block_table=torch.zeros(
                    (1, -(-capacity // tokens_per_block)), dtype=torch.int32, device=device
                ),
                schedule=torch.zeros((sms + 1, 2), dtype=torch.int32, device=device),
                logits=torch.empty(
                    (rows, -(-capacity // 256) * 256), dtype=torch.float32, device=device
                ),
                topk=torch.empty((rows, TOPK), dtype=torch.int32, device=device),
                k_fp8=torch.empty((rows, 128), dtype=torch.float8_e4m3fn, device=device),
                k_scale=torch.empty((rows, 1), dtype=torch.float32, device=device),
                sms=sms,
            )
        buffers = self._hist_buffers
        hist_shard_meta(
            attn_metadata.gen_indexer_kv_lens_cuda_runtime,
            attn_metadata.indexer_k_cache_block_offsets[0],
            buffers["shard_ctx"],
            buffers["shard_block_table"],
            self.mapping.tp_size,
            self.mapping.tp_rank,
            manager.tokens_per_block,
        )
        paged_mqa_logits_schedule(
            buffers["shard_ctx"], manager.tokens_per_block, buffers["sms"], buffers["schedule"]
        )
        return True

    def _attention(self, layer, position_ids, hidden_states, attn_metadata, sharded):
        """MLA with the DSA indexer over the generation rows: the attention output ahead of
        o_proj, [rows, heads * v_head_dim]."""
        mla = layer.self_attn
        mqa = mla.mqa
        indexer = mqa.indexer
        manager = attn_metadata.kv_cache_manager
        num_tokens = attn_metadata.num_tokens
        seq_len_q = num_tokens // attn_metadata.num_seqs
        head_dim = mla.kv_lora_rank + mla.qk_rope_head_dim
        mqa._ensure_rope_table_size(attn_metadata.max_seq_len)
        cos_sin = mqa.rotary_cos_sin
        kv_lens = attn_metadata.kv_lens_cuda_runtime
        local_layer = mqa.get_local_layer_idx(attn_metadata)
        if self._kv_pool is None:
            self._kv_pool = manager.get_unique_primary_pool()
        own_indexer = sharded and indexer is not None

        weight = (
            self._fused_a_indexer_weight[layer.layer_idx]
            if own_indexer
            else mla.kv_a_proj_with_mqa.weight
        )
        kv_a = fused_a_gemm(hidden_states, weight, self._l2_after_fused_a[layer.layer_idx])
        kv_a = kv_a[:num_tokens]
        q_a = kv_a[:, : mla.q_lora_rank]
        latent_end = mla.q_lora_rank + head_dim
        quant_q = hidden_states.new_empty(
            (num_tokens, mla.num_heads_tp * head_dim), dtype=torch.uint8
        )

        def q_b():
            q_b_fp8_rope(
                q_a,
                self._q_b_absorbed[layer.layer_idx],
                quant_q,
                kv_lens,
                cos_sin.view(-1, mla.qk_rope_head_dim, 2),
                self._unit_scale,
                mla.q_a_layernorm.variance_epsilon,
                head_dim,
                mla.qk_rope_head_dim,
                seq_len_q,
                "regular",
            )

        def kv_half():
            mla_kv_half_append(
                kv_a[:, mla.q_lora_rank : latent_end],
                mla.kv_a_layernorm.weight,
                mla.kv_a_layernorm.variance_epsilon,
                cos_sin,
                kv_lens,
                0,
                seq_len_q,
                attn_metadata.kv_cache_block_offsets,
                manager.kv_cache_pool_pointers,
                manager.kv_cache_pool_mapping,
                local_layer,
                manager.tokens_per_block,
                attn_metadata.max_seq_len,
                self._bmm1_scale,
                self._bmm2_scale,
                mla.softmax_scale,
            )

        output = hidden_states.new_empty(
            (hidden_states.shape[0], mla.num_heads_tp * mla.v_head_dim)
        )
        q_heads = quant_q.view(num_tokens, mla.num_heads_tp, head_dim)
        if sharded:

            def select_then_q_b():
                if own_indexer:
                    self._select_rows(
                        indexer, kv_a, latent_end, q_a, position_ids, attn_metadata, num_tokens, mla
                    )
                q_b()

            # The KV half on the aux stream beside the indexer and q_b.
            maybe_execute_in_parallel(
                select_then_q_b, kv_half, mla.ln_events[0], mla.ln_events[1], mla.aux_stream
            )
            page_index_scale, layer_offset = manager.get_primary_pool_page_index_params(
                local_layer
            )
            sparse_mla_flash_decode(
                q_heads,
                self._kv_pool,
                self._hist_buffers["topk"][:num_tokens],
                kv_lens,
                self._bmm1_scale,
                self._bmm2_scale,
                mla.v_b_proj,
                output,
                seq_len_q,
                self.index_topk,
                attn_metadata.block_table[: attn_metadata.num_seqs],
                manager.tokens_per_block,
                page_index_scale,
                layer_offset,
                # The selection of an earlier layer: its rows can load before the wait.
                not own_indexer,
            )
            return output

        # The built-in indexer over the whole context, then the selection as cache rows.
        maybe_execute_in_parallel(
            q_b, kv_half, mla.ln_events[0], mla.ln_events[1], mla.aux_stream
        )
        indexer_intermediates = None
        if indexer is not None:
            qr = mla.q_a_layernorm(q_a)
            indexer_intermediates = [
                t[:num_tokens]
                for t in indexer.pre_indexer_proj(
                    qr, hidden_states[:num_tokens], position_ids[..., :num_tokens]
                )
            ]
            indexer._update_k_cache(
                indexer_intermediates[1], indexer_intermediates[2], attn_metadata
            )
        indices, _ = mqa.sparse_attn_predict(
            hidden_states[:num_tokens],
            None,
            attn_metadata,
            AttentionForwardArgs(
                attention_input_type=AttentionInputType.generation_only,
                sparse_backend_args=DSABackendForwardArgs(
                    indexer_intermediates=indexer_intermediates
                ),
            ),
        )
        sparse_mla_flash_decode(
            q_heads,
            self._kv_pool,
            indices,
            kv_lens,
            self._bmm1_scale,
            self._bmm2_scale,
            mla.v_b_proj,
            output,
            seq_len_q,
            self.index_topk,
        )
        if indexer is not None:
            indexer.maybe_join_prev_topk_copy()
        return output

    def _select_rows(self, indexer, kv_a, latent_end, q_a, position_ids, attn_metadata, rows, mla):
        """The indexer of a layer that runs one, on this rank's key shard: writes the exact
        top-2048 cache positions of every row into the shared selection buffer."""
        buffers = self._hist_buffers
        manager = attn_metadata.kv_cache_manager
        rope = indexer.rotary_emb
        cos_sin = rope.rotary_cos_sin.view(rope.max_positions, -1)
        interleave = not rope.is_neox
        positions = position_ids.view(-1)[:rows]
        index_k = kv_a[:, latent_end : latent_end + indexer.head_dim]
        index_weights = kv_a[:, latent_end + indexer.head_dim :]
        k_cache = manager.get_indexer_k_cache_buffers(indexer.layer_idx)

        # The K tail on its own stream, beside the Q tail: k_norm, RoPE, FP8 and the K cache.
        self._k_tail_fork.record()
        with torch.cuda.stream(self._k_tail_stream):
            self._k_tail_stream.wait_event(self._k_tail_fork)
            indexer_k_tail(
                index_k,
                positions,
                cos_sin,
                indexer.k_norm.weight,
                indexer.k_norm.bias,
                indexer.k_norm.variance_epsilon,
                indexer.rope_dim,
                interleave,
                indexer.scale_fmt == "ue8m0",
                buffers["k_fp8"][:rows],
                buffers["k_scale"][:rows],
                k_cache,
                attn_metadata.slot_mapping_fp8[:rows],
                attn_metadata.slot_mapping_scale[:rows],
                rows,
                True,
                True,
                False,
            )
            self._k_tail_done.record()

        q_fp8 = q_a.new_empty((rows, indexer.n_heads, indexer.head_dim), dtype=torch.float8_e4m3fn)
        q_scale = q_a.new_empty((rows, indexer.n_heads, 1), dtype=torch.float32)
        weights = q_a.new_empty((rows, indexer.n_heads), dtype=torch.float32)
        indexer_q_tail(
            q_a,
            self._wq_b_folded[indexer.layer_idx],
            positions,
            cos_sin,
            index_weights,
            q_fp8,
            q_scale,
            weights,
            mla.q_a_layernorm.variance_epsilon,
            indexer.n_heads,
            indexer.head_dim,
            indexer.rope_dim,
            interleave,
            indexer.scale_fmt == "ue8m0",
            indexer.weight_scale_factor,
        )
        torch.cuda.current_stream().wait_event(self._k_tail_done)
        logits = fp8_paged_mqa_logits(
            q_fp8.view(1, rows, indexer.n_heads, indexer.head_dim),
            k_cache,
            weights,
            buffers["shard_ctx"],
            buffers["shard_block_table"],
            buffers["schedule"],
            buffers["logits"][:rows],
            buffers["capacity"],
            True,
        )
        hist_topk(
            logits,
            attn_metadata.gen_indexer_kv_lens_cuda_runtime,
            buffers["topk"][:rows],
            self._hist_workspace.comm_buffer,
            self._hist_workspace.state,
            rows,
            self.mapping.tp_size,
        )

    def _post_attention_hop(self, layer, attn_out: torch.Tensor, residual: torch.Tensor):
        """o_proj published into the all-reduce workspace, then the hop with the residual add
        and post_attention_layernorm: (normed, residual)."""
        rows = attn_out.shape[0]
        lamport, flags = publish_target(self.mapping, rows, self.hidden_size)
        comm_buffer, buffer_flags = hop_workspace(self.mapping, rows, self.hidden_size)
        shard = attn_out.new_empty((rows, self.hidden_size))
        o_proj_publish(
            attn_out,
            layer.self_attn.o_proj.weight,
            shard,
            lamport,
            flags,
            self.mapping.tp_size,
            self.mapping.tp_rank,
        )
        norm = layer.post_attention_layernorm
        return mnnvl_hop(
            shard, residual, norm.weight, norm.variance_epsilon, comm_buffer, buffer_flags, True
        )

    def _hop(self, norm, partial: torch.Tensor, residual: torch.Tensor):
        """The hop of a rank-partial sum with the residual add and `norm`: (normed, residual)."""
        comm_buffer, buffer_flags = hop_workspace(self.mapping, partial.shape[0], self.hidden_size)
        return mnnvl_hop(
            partial, residual, norm.weight, norm.variance_epsilon, comm_buffer, buffer_flags, False
        )

    def _moe(self, layer, hidden_states: torch.Tensor, residual: torch.Tensor):
        """The MoE block and the post-MoE hop that finalizes it: (normed, residual)."""
        mlp = layer.mlp
        backend = getattr(mlp.experts, "backend", mlp.experts)
        rows = hidden_states.shape[0]

        self._quant_fork.record()
        with torch.cuda.stream(self._quant_stream):
            self._quant_stream.wait_event(self._quant_fork)
            x_fp4, x_sf = fp4_quantize(
                hidden_states, backend.fc31_input_scale, backend.scaling_vector_size, False, False
            )
            self._quant_ready.record()
        routing = extract_routing_params(backend.routing_method)

        def routed():
            # Unfinalized: [FC2 rows, expert weights, expanded index -> FC2 row].
            return routed_experts(
                self._router_logits[:rows],
                routing.routing_bias,
                x_fp4,
                x_sf.flatten().view(torch.float8_e4m3fn),
                backend.w3_w1_weight,
                backend.w3_w1_weight_scale.view(torch.float8_e4m3fn),
                None,
                backend.act_alpha,
                backend.act_beta,
                backend.act_clamp,
                backend.w2_weight,
                backend.w2_weight_scale.view(torch.float8_e4m3fn),
                None,
                get_data_or_none(backend, "fc31_scale_c"),
                get_data_or_none(backend, "fc31_alpha"),
                get_data_or_none(backend, "fc2_alpha"),
                backend.num_slots,
                routing.top_k,
                routing.n_group,
                routing.topk_group,
                backend.w3_w1_weight.shape[-2] // 2,
                backend.slot_start,
                backend.expert_size_per_partition,
                routing.routed_scaling_factor,
                backend.routing_method.routing_method_type,
                False,
                to_trtllm_gen_act_type(backend.activation_type),
                None,
                None,
                None,
                backend.max_num_tokens,
                False,
                self._quant_ready.cuda_event,
                hidden_states,
                mlp.gate.weight,
                self._router_workspace[layer.layer_idx],
                True,
            )

        (fc2_output, expert_weights, permuted_rows), shared_output = maybe_execute_in_parallel(
            routed,
            lambda: self._shared_expert(mlp, hidden_states),
            mlp.event_dict[EventType.Main],
            mlp.event_dict[EventType.MoeShared],
            mlp.aux_stream,
        )
        norm = layer.next_layer_layernorm
        comm_buffer, buffer_flags = hop_workspace(self.mapping, rows, self.hidden_size)
        return mnnvl_moe_finalize_hop(
            fc2_output,
            expert_weights,
            permuted_rows,
            shared_output,
            residual,
            norm.weight,
            norm.variance_epsilon,
            comm_buffer,
            buffer_flags,
        )

    def _shared_expert(self, mlp, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's partial sum of the shared expert: two split-K GEMMs around SwiGLU."""
        expert = mlp.shared_experts
        rows = hidden_states.shape[0]
        gate_up = expert.gate_up_proj.weight
        h = low_m_gemm(hidden_states, gate_up, hidden_states.new_empty(rows, gate_up.shape[0]))
        h = swiglu(h)
        return low_m_gemm(
            h, expert.down_proj.weight, hidden_states.new_empty(rows, self.hidden_size)
        )

    @staticmethod
    def _builtin_layer(layer, position_ids, hidden_states, residual, attn_metadata):
        """One decoder layer on the built-in modules, for a batch the kernels do not serve."""
        dense = isinstance(layer.mlp, GatedMLP)
        hidden_states = DecodeTarget._attention_builtin(
            layer,
            position_ids,
            hidden_states,
            attn_metadata,
            AllReduceParams(enable_allreduce=dense),
        )
        if dense:
            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
            hidden_states = layer.mlp(
                hidden_states, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
            )
        else:
            hidden_states, residual = layer.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=layer.post_attention_layernorm.weight,
                    eps=layer.post_attention_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ),
            )
            hidden_states = layer.mlp(
                hidden_states,
                None,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(enable_allreduce=False),
                do_finalize=True,
            )
        return layer.allreduce(
            hidden_states,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=layer.next_layer_layernorm.weight,
                eps=layer.next_layer_layernorm.variance_epsilon,
                trigger_completion_at_end=dense,
            ),
        )

    @staticmethod
    def _attention_builtin(layer, position_ids, hidden_states, attn_metadata, all_reduce_params):
        """The built-in modules' MLA with the DSA indexer, for a batch the kernels do not serve."""
        mla = layer.self_attn
        num_tokens = attn_metadata.num_tokens
        kv_a = mla.kv_a_proj_with_mqa(hidden_states)
        q, compressed_kv, k_pe = kv_a.split(
            [mla.q_lora_rank, mla.kv_lora_rank, mla.qk_rope_head_dim], -1
        )
        (q, qr), compressed_kv = maybe_execute_in_parallel(
            lambda: mla._q_a_layernorm_maybe_fused(q, return_norm_out=True),
            lambda: mla.kv_a_layernorm(compressed_kv),
            mla.ln_events[0],
            mla.ln_events[1],
            mla.aux_stream,
        )
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)
        q = mla.q_b_proj(q)
        indexer = mla.mqa.indexer
        indexer_intermediates = None
        if indexer is not None:
            indexer_intermediates = [
                t[:num_tokens]
                for t in indexer.pre_indexer_proj(qr, hidden_states, position_ids)
            ]
            indexer._update_k_cache(indexer_intermediates[1], indexer_intermediates[2], attn_metadata)
        output = hidden_states.new_empty([hidden_states.shape[0], mla.attention_output_hidden_size])
        forward_generation_sparse_attn(
            mla,
            q[:num_tokens],
            compressed_kv[:num_tokens],
            k_pe[:num_tokens],
            attn_metadata,
            output[:num_tokens],
            position_ids=position_ids[..., :num_tokens],
            latent_cache=latent_cache[:num_tokens],
            indexer_intermediates=indexer_intermediates,
        )
        if indexer is not None:
            indexer.maybe_join_prev_topk_copy()
        return mla.o_proj(output, all_reduce_params=all_reduce_params, layer_idx=mla.layer_idx)


class Glm52Nvfp4Sm107Tp16Core(DeepseekV3Model):
    """The decoder stack, with the per-step choice between the two targets."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__(model_config)
        self._prefill: Optional[PrefillTarget] = None
        self._decode: Optional[DecodeTarget] = None

    def post_load_weights(self) -> None:
        """Build the targets once the weights are final. Reached more than once
        (the shell and the loader's module walk), so it only builds once."""
        if self._decode is not None:
            return
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _select_target(self, attn_metadata: AttentionMetadata) -> "PrefillTarget | DecodeTarget":
        if self._decode is None:
            self.post_load_weights()
        return self._prefill if attn_metadata.num_contexts > 0 else self._decode

    def decode_target(self, attn_metadata: AttentionMetadata) -> Optional[DecodeTarget]:
        """The decode target when this step runs it, else None."""
        target = self._select_target(attn_metadata)
        return target if target is self._decode else None

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._select_target(attn_metadata).forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )


class ModelingV2Glm52Nvfp4Sm107Tp16(
    SpecDecOneEngineForCausalLM[Glm52Nvfp4Sm107Tp16Core, PretrainedConfig]
):
    """GLM-5.2-NVFP4, sm107, tp16."""

    @classmethod
    def get_preferred_kv_cache_manager_version(
        cls, pretrained_config: Any = None
    ) -> Literal["V2"]:
        """The KV cache manager the context server's built-in implementation prefers."""
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls, pretrained_config: Any = None
    ) -> Optional[Literal["CPP", "PYTHON"]]:
        """The transceiver runtime for the masked DSA indexer K-cache transfer."""
        return "PYTHON"

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        mapping = model_config.mapping
        assert torch.cuda.get_device_capability() == _COMPUTE_CAPABILITY, (
            f"this target runs on sm107, not {torch.cuda.get_device_capability()}"
        )
        assert (
            mapping.world_size == _TP
            and mapping.tp_size == _TP
            and mapping.moe_ep_size == 1
            and not mapping.enable_attention_dp
            and mapping.pp_size == 1
        ), f"this target is tp16 without attention DP or expert parallelism, not {mapping}"
        model_config = _weights.normalize_model_config(model_config)
        super().__init__(model=Glm52Nvfp4Sm107Tp16Core(model_config), model_config=model_config)
        if (
            model_config.spec_config is not None
            and model_config.spec_config.spec_dec_mode.is_mtp_one_model()
        ):
            self.num_hidden_layers = self.config.num_hidden_layers
            self.model.layers.extend(self.draft_model.mtp_layers)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        decode = self.model.decode_target(attn_metadata)
        if decode is None or self.spec_worker is None:
            return super().forward(
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_context_logits=return_context_logits,
                spec_metadata=spec_metadata,
                resource_manager=resource_manager,
                **kwargs,
            )
        # A generation step with speculative decoding: the verify logits on the decode
        # target's kernels where they serve the rows, then the spec worker.
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if attn_metadata.padded_num_tokens is not None:
            num_tokens = attn_metadata.num_tokens
            hidden_states = hidden_states[:num_tokens]
            if input_ids is not None:
                input_ids = input_ids[:num_tokens]
            position_ids = _slice_spec_position_ids(position_ids, num_tokens)
        rows = hidden_states[spec_metadata.gather_ids]
        logits = decode.logits(self.lm_head, rows)
        if logits is None:
            logits = self.logits_processor.forward(rows, self.lm_head, attn_metadata, True)
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=resource_manager,
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.load_weights(self, weights)

    def setup_aliases(self) -> None:
        DeepseekV3ForCausalLM.setup_aliases(self)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
