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
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
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
from .ops.embedding.fused_embedding import fused_embedding
from .ops.gemm.dense_mlp_small_m import INTER_PER_RANK, dense_mlp_small_m
from .ops.gemm.fused_a_gemm import MAX_TOKENS as FUSED_A_MAX_TOKENS
from .ops.gemm.fused_a_gemm import OUT_FEATURES as FUSED_A_OUT_FEATURES
from .ops.gemm.fused_a_gemm import fused_a_gemm
from .ops.gemm.low_m_gemm import low_m_gemm
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

    The step is written out layer by layer for this deployment's configuration,
    which ``__init__`` checks once:

    - Layers 0-2 (dense, unquantized): the attention all-reduces its own output,
      the residual add and ``post_attention_layernorm`` follow, and the MLP's
      all-reduce carries the residual add and the next layer's input norm.
    - Layers 3-77 (MoE): both all-reduces carry the residual add and the
      following norm; the experts finalize before theirs (16 ranks span nodes).
    - The last all-reduce applies the model's final norm (``next_layer_layernorm``).

    Attention is MLA with the DSA indexer on the generation rows. For up to 8
    rows: q_a, the compressed KV and its rope part come from one GEMM over the
    stored kv_a_proj_with_mqa weight; q_b with k_b absorbed and the q_a norm
    folded writes the RoPE'd FP8 query in its epilogue, while on the aux stream
    one kernel normalizes, ropes and appends the KV half to the FP8 cache; the
    built-in indexer selects the 2048 rows; and the flash decode attends over
    them at 4 heads with v_b folded in, ahead of o_proj. A larger batch takes the
    built-in modules' path.

    The MoE block over up to 8 rows runs its shared expert as two split-K
    GEMMs around SwiGLU on the aux stream, and its routed experts in one
    trtllm-gen call on the main stream: the routing kernel runs the router GEMV
    itself, the NVFP4 input quantizes on its own stream and joins right before
    FC1, and FC2 runs on the K = 128 kernel. A larger batch takes the built-in
    MoE module.

    The embedding table is whole on every rank, so the lookup is one local
    kernel. The dense MLP runs as two weight-streaming kernels on this rank's
    768 intermediate columns for up to 8 rows (every c1 step: one request and
    its 7 drafts); a larger batch takes the generic gated MLP.
    """

    def __init__(self, core: "Glm52Nvfp4Sm107Tp16Core") -> None:
        self.core = core
        config = core.model_config.pretrained_config
        self.hidden_size = config.hidden_size
        for layer in core.layers[: core.num_hidden_layers]:
            mla = layer.self_attn
            assert layer.allreduce is not None and layer.next_layer_layernorm is not None
            assert not mla.apply_rotary_emb and not mla.mapping.has_cp_helix()
            assert tuple(mla.kv_a_proj_with_mqa.weight.shape) == (
                FUSED_A_OUT_FEATURES,
                config.hidden_size,
            ) and mla.kv_a_proj_with_mqa.weight.dtype == torch.bfloat16, (
                f"layer {layer.layer_idx}: kv_a_proj_with_mqa is not the BF16 [q_a; kv_a] weight"
            )
            assert layer.input_layernorm.nvfp4_scale is None, "kv_a_proj is not NVFP4"
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
                shared = layer.mlp.shared_experts
                assert (
                    shared is not None
                    and layer.mlp.shared_output_scale is None
                    and shared._is_plain_swiglu()
                    and shared.swiglu_limit is None
                    and shared.gate_up_proj.weight.dtype == torch.bfloat16
                    and shared.down_proj.weight.dtype == torch.bfloat16
                ), f"layer {layer.layer_idx}: the shared expert is not a plain BF16 SwiGLU MLP"
                backend = getattr(layer.mlp.experts, "backend", layer.mlp.experts)
                assert (
                    backend.has_nvfp4
                    and backend.scaling_vector_size == 16
                    and getattr(backend, "fc31_act_scale", None) is None
                    and not backend.bias
                    and backend.w3_w1_weight.shape[-1] * 2 == config.hidden_size
                ), f"layer {layer.layer_idx}: routed experts are not unpadded TRTLLM-Gen NVFP4"
        self.index_topk = config.index_topk
        # q_b with k_b absorbed and the q_a norm weight folded, per layer: for head h, rows
        # [h * 576, h * 576 + 512) are k_b_trans[h] @ W_q_b[h, :qk_nope] and the next 64 rows
        # W_q_b[h, qk_nope:], every column times gamma_q_a; FP64, rounded to BF16 once.
        self._q_b_absorbed = {}
        for layer in core.layers[: core.num_hidden_layers]:
            mla = layer.self_attn
            assert (
                getattr(mla.mqa, "has_fp8_kv_cache", False)
                and mla.kv_cache_dtype != "fp8_ds_mla"
                and mla.k_b_proj_trans.dtype == torch.bfloat16
                and mla.v_b_proj.dtype == torch.bfloat16
                and mla.v_b_proj.is_contiguous()
                and mla.q_b_proj.weight.dtype == torch.bfloat16
            ), f"layer {layer.layer_idx}: unexpected MLA weights or KV cache layout"
            qk_head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim
            w = mla.q_b_proj.weight.view(mla.num_heads_tp, qk_head_dim, mla.q_lora_rank).double()
            nope = torch.bmm(mla.k_b_proj_trans.double(), w[:, : mla.qk_nope_head_dim])
            absorbed = torch.cat([nope, w[:, mla.qk_nope_head_dim :]], 1)
            absorbed = absorbed * mla.q_a_layernorm.weight.double()
            self._q_b_absorbed[layer.layer_idx] = (
                absorbed.to(torch.bfloat16).reshape(-1, mla.q_lora_rank).contiguous()
            )
        device = core.embed_tokens.weight.device
        # The attention scales the KV-half kernel writes and the flash decode reads.
        self._bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
        self._bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
        self._unit_scale = torch.ones(1, dtype=torch.float32, device=device)
        self._kv_pool = None

        moe_layers = [
            layer
            for layer in core.layers[: core.num_hidden_layers]
            if not isinstance(layer.mlp, GatedMLP)
        ]
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
        if inputs_embeds is None:
            inputs_embeds = fused_embedding(input_ids, core.embed_tokens.weight)
        residual = inputs_embeds
        hidden_states = core.layers[0].input_layernorm(inputs_embeds)
        rows = hidden_states.shape[0]
        small_m = torch.ops.trtllm.glm_dense_mlp_small_m_supported(
            rows, self.hidden_size, INTER_PER_RANK
        )
        fused_a = rows <= FUSED_A_MAX_TOKENS
        for layer in core.layers[: core.num_hidden_layers]:
            if isinstance(layer.mlp, GatedMLP):
                hidden_states, residual = self._dense_layer(
                    layer, position_ids, hidden_states, residual, attn_metadata, fused_a, small_m
                )
            else:
                hidden_states, residual = self._moe_layer(
                    layer, position_ids, hidden_states, residual, attn_metadata, fused_a
                )
        return hidden_states

    def _experts(self, layer, hidden_states: torch.Tensor) -> torch.Tensor:
        """The MoE block's output for up to 8 rows: shared expert plus routed experts."""
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

        def routed() -> torch.Tensor:
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
                True,
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
            )[0]

        def shared() -> torch.Tensor:
            expert = mlp.shared_experts
            gate_up = expert.gate_up_proj.weight
            h = low_m_gemm(hidden_states, gate_up, hidden_states.new_empty(rows, gate_up.shape[0]))
            h = swiglu(h)
            return low_m_gemm(
                h, expert.down_proj.weight, hidden_states.new_empty(rows, hidden_states.shape[1])
            )

        routed_output, shared_output = maybe_execute_in_parallel(
            routed,
            shared,
            mlp.event_dict[EventType.Main],
            mlp.event_dict[EventType.MoeShared],
            mlp.aux_stream,
        )
        return shared_output.add_(routed_output)

    def _attention(self, layer, position_ids, hidden_states, attn_metadata, all_reduce_params, fused):
        """MLA with the DSA indexer over generation rows, through o_proj."""
        if not fused:
            return self._attention_builtin(
                layer, position_ids, hidden_states, attn_metadata, all_reduce_params
            )
        mla = layer.self_attn
        mqa = mla.mqa
        manager = attn_metadata.kv_cache_manager
        num_tokens = attn_metadata.num_tokens
        seq_len_q = num_tokens // attn_metadata.num_seqs
        head_dim = mla.kv_lora_rank + mla.qk_rope_head_dim
        mqa._ensure_rope_table_size(attn_metadata.max_seq_len)
        cos_sin = mqa.rotary_cos_sin
        kv_lens = attn_metadata.kv_lens_cuda_runtime
        if self._kv_pool is None:
            self._kv_pool = manager.get_unique_primary_pool()

        kv_a = fused_a_gemm(hidden_states, mla.kv_a_proj_with_mqa.weight, [])[:num_tokens]
        q_a = kv_a[:, : mla.q_lora_rank]
        quant_q = hidden_states.new_empty((num_tokens, mla.num_heads_tp * head_dim), dtype=torch.uint8)

        def q_half():
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
                kv_a[:, mla.q_lora_rank :],
                mla.kv_a_layernorm.weight,
                mla.kv_a_layernorm.variance_epsilon,
                cos_sin,
                kv_lens,
                0,
                seq_len_q,
                attn_metadata.kv_cache_block_offsets,
                manager.kv_cache_pool_pointers,
                manager.kv_cache_pool_mapping,
                mqa.get_local_layer_idx(attn_metadata),
                manager.tokens_per_block,
                attn_metadata.max_seq_len,
                self._bmm1_scale,
                self._bmm2_scale,
                mla.softmax_scale,
            )

        maybe_execute_in_parallel(
            q_half, kv_half, mla.ln_events[0], mla.ln_events[1], mla.aux_stream
        )
        indexer = mqa.indexer
        indexer_intermediates = None
        if indexer is not None:
            qr = mla.q_a_layernorm(q_a)
            indexer_intermediates = [
                t[:num_tokens]
                for t in indexer.pre_indexer_proj(
                    qr, hidden_states[:num_tokens], position_ids[..., :num_tokens]
                )
            ]
            indexer._update_k_cache(indexer_intermediates[1], indexer_intermediates[2], attn_metadata)
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
        output = hidden_states.new_empty(
            (hidden_states.shape[0], mla.num_heads_tp * mla.v_head_dim)
        )
        sparse_mla_flash_decode(
            quant_q.view(num_tokens, mla.num_heads_tp, head_dim),
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
        return mla.o_proj(output, all_reduce_params=all_reduce_params, layer_idx=mla.layer_idx)

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

    def _dense_layer(
        self, layer, position_ids, hidden_states, residual, attn_metadata, fused_a, small_m
    ):
        hidden_states = self._attention(
            layer,
            position_ids,
            hidden_states,
            attn_metadata,
            AllReduceParams(enable_allreduce=True),
            fused_a,
        )
        hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
        if small_m:
            # The rank partial; the all-reduce below sums it over the 16 ranks.
            hidden_states = dense_mlp_small_m(
                hidden_states, layer.mlp.gate_up_proj.weight, layer.mlp.down_proj.weight
            )
        else:
            hidden_states = layer.mlp(
                hidden_states, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
            )
        return layer.allreduce(
            hidden_states,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=layer.next_layer_layernorm.weight,
                eps=layer.next_layer_layernorm.variance_epsilon,
            ),
        )

    def _moe_layer(self, layer, position_ids, hidden_states, residual, attn_metadata, fused_a):
        hidden_states = self._attention(
            layer,
            position_ids,
            hidden_states,
            attn_metadata,
            AllReduceParams(enable_allreduce=False),
            fused_a,
        )
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
        if hidden_states.shape[0] <= FUSED_ROUTER_MAX_TOKENS:
            hidden_states = self._experts(layer, hidden_states)
        else:
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
                trigger_completion_at_end=False,
            ),
        )


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

    def load_weights(self, weights, *args, **kwargs):
        _weights.load_weights(self, weights)

    def setup_aliases(self) -> None:
        DeepseekV3ForCausalLM.setup_aliases(self)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
