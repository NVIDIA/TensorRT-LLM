import copy
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
import triton
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN as HF_ACT2FN

# TODO: Need to change this once we have a proper transformers package
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5VisionPatchEmbed as HFQwen3_5VisionPatchEmbed,
    )
except (ImportError, ModuleNotFoundError):
    # Qwen3_5VisionPatchEmbed is same as Qwen3VLVisionPatchEmbed
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionPatchEmbed as HFQwen3_5VisionPatchEmbed,
    )
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5VisionRotaryEmbedding as HFQwen3_5VisionRotaryEmbedding,
    )
except (ImportError, ModuleNotFoundError):
    # Qwen3_5VisionRotaryEmbedding is same as Qwen3VLVisionRotaryEmbedding
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionRotaryEmbedding as HFQwen3_5VisionRotaryEmbedding,
    )

from tensorrt_llm._torch.models.modeling_multimodal_utils import _is_disagg
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import \
    fused_sigmoid_gating_delta_rule_update
from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import \
    use_cpp_mamba_cache_manager
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping

from ..._utils import nvtx_range, nvtx_range_debug
from ...inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    register_input_processor,
    support_multimodal_disaggregated,
)
from ...inputs.multimodal import MultimodalParams
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..attention_backend.utils import get_attention_backend
from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
                           MoEAllReduce, MoEAllReduceParams)
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear, TensorParallelMode
from ..modules.mlp import MLP
from ..modules.gated_mlp import GatedMLP
from ..modules.rotary_embedding import MRotaryEmbedding
from ..modules.decoder_layer import DecoderLayer
from ..modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from ..modules.mamba.layernorm_gated import RMSNorm as RMSNormGated
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..modules.embedding import Embedding
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5HfWeightMapper, Qwen3_5TextConfig
from ..utils import EventType
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)
from .modeling_qwen2vl import Qwen2_5_VLVisionAttention
from .modeling_qwen3 import Qwen3Attention
from .modeling_qwen3_next import divide, fused_gdn_gating, fused_qkvzba_split_reshape_cat
from .modeling_utils import (
    ModelConfig,
    QuantConfig,
    _load_weights_impl,
    filter_weights,
    register_auto_model,
    register_vision_encoder,
    EagerFusionConfig,
    DecoderModel,
    SpecMetadata,
)
from .modeling_speculative import SpecDecOneEngineForCausalLM


class Qwen3_5GatedDeltaNet(nn.Module):

    def __init__(self,
                 model_config: ModelConfig[Qwen3_5TextConfig],
                 aux_stream: torch.cuda.Stream,
                 layer_idx: Optional[int] = None):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.pretrained_config = config

        # tensor parallel
        tp_size = model_config.mapping.tp_size
        pp_size = model_config.mapping.pp_size
        if model_config.mapping.enable_attention_dp:
            tp_size = 1

        mapping = Mapping(
            world_size=tp_size * pp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            rank=model_config.mapping.rank,
            gpus_per_node=model_config.mapping.gpus_per_node,
            enable_attention_dp=model_config.mapping.enable_attention_dp,
        )
        self.mapping = mapping

        self.attn_tp_rank = mapping.tp_rank
        self.attn_tp_size = mapping.tp_size
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = Linear(
            self.conv_kernel_size,
            self.conv_dim,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.
            skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False)

        self.in_proj_qkv = Linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.
            skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False)
        self.in_proj_z = Linear(
            self.hidden_size,
            self.value_dim,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.
            skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False)
        self.in_proj_b = Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.
            skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False)
        self.in_proj_a = Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.
            skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False)

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(
                (self.num_v_heads // self.attn_tp_size),
                dtype=config.torch_dtype,
            ),
            requires_grad=False,
        )

        A = torch.empty(divide(self.num_v_heads, self.attn_tp_size),
                        dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(
            torch.log(A),
            requires_grad=False,
        )
        self.A_log._no_weight_decay = True

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.cuda.current_device(),
            dtype=torch.float32,
        )

        # gemmaNorm is not supported in fused_all_reduce kernel.
        # So, we need to do allReduce in Linear and do gemmaNorm in separate kernel.
        self.out_proj = Linear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=model_config.get_quant_config(),
            reduce_output=True,
            skip_create_weights_in_init=model_config.
            skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False)

        self.event_dict = {
            key: torch.cuda.Event()
            for key in [EventType.Main, EventType.Attention]
        }
        self.aux_stream = aux_stream

    def forward_decode(
        self,
        conv_states,
        ssm_states,
        query_start_loc_long,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        a = kwargs["a"]
        b = kwargs["b"]
        cache_indices = kwargs["cache_indices"]

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            self.conv1d.weight,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=cache_indices,
        )

        # Direct slicing instead of torch.split for better performance
        key_size = self.key_dim // self.attn_tp_size
        query = mixed_qkv[..., :key_size]
        key = mixed_qkv[..., key_size:key_size * 2]
        value = mixed_qkv[..., key_size * 2:]
        # Reshape from [l, h*d] to [1, l, h, d]
        seq_len = query.shape[0]
        num_heads = query.shape[1] // self.head_k_dim
        query = query.view(1, seq_len, num_heads, self.head_k_dim)
        key = key.view(1, seq_len, num_heads, self.head_k_dim)
        value = value.view(1, seq_len, value.shape[1] // self.head_v_dim,
                           self.head_v_dim)

        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc_long,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

        return core_attn_out

    def forward_extend(
        self,
        conv_states,
        ssm_states,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        a = kwargs["a"]
        b = kwargs["b"]
        batch_size = kwargs["batch_size"]
        has_initial_states = kwargs["has_initial_states"][:batch_size]
        cache_indices = kwargs["cache_indices"]
        query_start_loc = kwargs["query_start_loc"]
        query_start_loc_long = kwargs["query_start_loc_long"]
        num_prefill_tokens = kwargs["num_prefill_tokens"]
        num_decode_tokens = kwargs["num_decode_tokens"]
        state_indices_p = kwargs["state_indices_p"]
        state_indices_d = kwargs["state_indices_d"]
        num_prefill = kwargs["num_prefill"]

        conv_states_to_use = conv_states

        seqlen_split_size = [num_prefill_tokens, num_decode_tokens]
        if num_decode_tokens > 0:
            mixed_qkv_p, mixed_qkv_d = torch.split(mixed_qkv,
                                                   seqlen_split_size,
                                                   dim=0)
            query_start_loc_p = query_start_loc[:num_prefill + 1]
            has_initial_states_p = has_initial_states[:num_prefill]

            mixed_qkv_p = causal_conv1d_fn(
                mixed_qkv_p.transpose(0, 1),
                self.conv1d.weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_states_to_use,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_p,
                query_start_loc=query_start_loc_p,
            ).transpose(0, 1)

            mixed_qkv_d = causal_conv1d_update(
                mixed_qkv_d,
                conv_states_to_use,
                self.conv1d.weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_state_indices=state_indices_d,
            )
            mixed_qkv = torch.cat((mixed_qkv_p, mixed_qkv_d), dim=0)
        else:
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                self.conv1d.weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_states_to_use,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc).transpose(0, 1)

        key_split_dim = self.key_dim // self.attn_tp_size
        value_split_dim = self.value_dim // self.attn_tp_size

        query, key, value = torch.split(
            mixed_qkv,
            [key_split_dim, key_split_dim, value_split_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        num_heads = query.shape[1] // self.head_k_dim
        num_value_heads = value.shape[1] // self.head_v_dim

        query = query.view(1, actual_seq_len, num_heads, self.head_k_dim)
        key = key.view(1, actual_seq_len, num_heads, self.head_k_dim)
        value = value.view(1, actual_seq_len, num_value_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = fused_gdn_gating(self.A_log, a, self.dt_bias)

        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)

        recurrent_state = ssm_states[cache_indices]

        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            cu_seqlens=query_start_loc_long,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        last_recurrent_state = last_recurrent_state.to(ssm_states.dtype,
                                                       copy=False)
        ssm_states[cache_indices] = last_recurrent_state

        return core_attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
        all_reduce_params: Optional[AllReduceParams] = None,
    ):
        ### sglang linear attn
        # has_initial_states = None
        # if forward_batch.extend_prefix_lens is not None:
        #     has_initial_states = forward_batch.extend_prefix_lens > 0

        # # Set up dimensions for reshapes later
        seq_len, _ = hidden_states.shape
        conv_state, recurrent_state = None, None

        ### mamba2_mixer layer
        # calculate split size
        num_prefills = attn_metadata.num_contexts
        num_decodes = attn_metadata.seq_lens.shape[0] - num_prefills
        num_prefill_tokens = attn_metadata.num_ctx_tokens
        num_decode_tokens = attn_metadata.num_tokens - num_prefill_tokens
        batch_split_size = [num_prefills, num_decodes]
        has_initial_states = mamba_metadata.has_initial_states

        batch_size = num_prefills + num_decodes
        if use_cpp_mamba_cache_manager():
            state_indices = mamba_metadata.state_indices[:batch_size]
        else:
            state_indices = attn_metadata.kv_cache_manager.get_state_indices(
            )[:batch_size]

        state_indices_p, state_indices_d = torch.split(state_indices,
                                                       batch_split_size)
        conv_states = attn_metadata.kv_cache_manager.get_conv_states(
            self.layer_idx)
        ssm_states = attn_metadata.kv_cache_manager.get_ssm_states(
            self.layer_idx)
        if num_prefills > 0:
            ssm_states[state_indices_p] = torch.zeros((),
                                                      dtype=ssm_states.dtype,
                                                      device=ssm_states.device)
        
        def _compute_projected_states_qkv():
            return self.in_proj_qkv(hidden_states)

        def _compute_projected_states_z():
            return self.in_proj_z(hidden_states)
        
        def _compute_projected_states_b():
            return self.in_proj_b(hidden_states)

        def _compute_projected_states_a():
            return self.in_proj_a(hidden_states)
        
        mixed_qkv, z = maybe_execute_in_parallel(
            _compute_projected_states_qkv,
            _compute_projected_states_z,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.Attention],
            self.aux_stream,
        )
        b, a = maybe_execute_in_parallel(
            _compute_projected_states_b,
            _compute_projected_states_a,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.Attention],
            self.aux_stream,
        )
        z = z.view(z.size(0), -1, self.head_v_dim)

        kwargs = {
            "mixed_qkv": mixed_qkv,
            "a": a,
            "b": b,
            "z": z,
            "has_initial_states": has_initial_states,
            "cache_indices": state_indices,
            "query_start_loc": mamba_metadata.query_start_loc,
            "query_start_loc_long": mamba_metadata.query_start_loc_long,
            "batch_size": attn_metadata.seq_lens.shape[0],
            "num_prefill_tokens": num_prefill_tokens,
            "num_decode_tokens": num_decode_tokens,
            "state_indices_p": state_indices_p,
            "state_indices_d": state_indices_d,
            "num_prefill": num_prefills,
        }
        if num_prefills > 0:
            attn_out = self.forward_extend(conv_states, ssm_states, **kwargs)
        else:
            attn_out = self.forward_decode(conv_states, ssm_states, **kwargs)

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        attn_out = attn_out.reshape(-1, attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        attn_out = self.norm(attn_out, z)
        attn_out = attn_out.reshape(z_shape_og)
        attn_out = attn_out.reshape(*attn_out.shape[:-2], -1)

        output = self.out_proj(attn_out, all_reduce_params=all_reduce_params)
        return output


class Qwen3_5LinearDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3_5TextConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config
        self.linear_attn = Qwen3_5GatedDeltaNet(model_config, aux_stream,
                                                  layer_idx)

        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias if hasattr(config, "mlp_bias") else False,
            dtype=config.torch_dtype,
            overridden_tp_size=1 if self.enable_attention_dp else None,
            config=model_config,
        )

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype,
                                       use_gemma=True)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype,
                                                use_gemma=True)
        self.layer_idx = layer_idx

        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy)
        self.next_layer_layernorm: RMSNorm = None

        self.fusion_config = EagerFusionConfig()
        ### TODO: enable eager_fusion by default
        self.enable_fusion = os.environ.get(
            "TRTLLM_QWEN3_5_EAGER_FUSION_DISABLED", "1") == "0"
        self.enable_fusion &= not self.enable_attention_dp

        # has_tp = self.mapping.has_tp()
        has_pp = self.mapping.has_pp()

        # self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
        self.fusion_config.PRE_MOE_FUSION = False  # the fusion kernel does not support gemmaNorm yet
        self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION and not has_pp
        self.disable_attn_allreduce = (self.fusion_config.PRE_MOE_FUSION
                                       or self.mapping.tp_size == 1
                                       or self.enable_attention_dp)
        self.moe_allreduce = MoEAllReduce(mapping=model_config.mapping)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            self.fusion_config.POST_MOE_FUSION = False
        # Linear Attention
        ### FIXME: 1. forward_batch; 2. allreduce
        if hidden_states.shape[0] != 0:
            hidden_states = self.linear_attn(
                hidden_states,
                attn_metadata,
                all_reduce_params=AllReduceParams(
                    enable_allreduce=not (self.fusion_config.PRE_MOE_FUSION
                                          or self.mapping.tp_size == 1)),
                **kwargs)
        if self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    enable_allreduce=not (self.fusion_config.PRE_MOE_FUSION
                                          or self.mapping.tp_size == 1),
                ))
        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        # Note: this fusion pattern is only supported for TRTLLM-nvfp4 backend now
        do_finalize = not (hidden_states.shape[0]
                           <= self.moe_allreduce.max_token
                           and self.fusion_config.POST_MOE_FUSION
                           and self.model_config.moe_backend == 'TRTLLM'
                           and self.mlp.experts.has_nvfp4)

        hidden_states = self.mlp(
            hidden_states,
            attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.POST_MOE_FUSION
                                      or self.mapping.tp_size == 1)),
            do_finalize=do_finalize,
        )
        if self.fusion_config.POST_MOE_FUSION:
            if do_finalize:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
            else:
                assert len(
                    hidden_states
                ) == 3, f"hidden_states must have 3 elements, but got {len(hidden_states)}"

                fc2_output = hidden_states[0]
                expert_scale_factor = hidden_states[1]
                expanded_idx_to_permuted_idx = hidden_states[2]

                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=None,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                hidden_states, residual = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params)

        else:
            if spec_metadata and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(
                    self.layer_idx, hidden_states, residual)
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(
                    hidden_states, residual)
        return hidden_states, residual


class Qwen3_5Attention(Qwen3Attention):

    def __init__(self, model_config: ModelConfig[Qwen3_5TextConfig],
                 layer_idx: int, fuse_qk_norm_rope: bool):
        super().__init__(model_config,
                         layer_idx,
                         fuse_qk_norm_rope=fuse_qk_norm_rope,
                         attn_output_gate=True,
                         use_gemma_rms_norm=True)


class Qwen3_5FullAttentionDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[Qwen3_5TextConfig],
                 layer_idx: int, aux_stream: torch.cuda.Stream):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config

        self.self_attn = Qwen3_5Attention(
            model_config,
            layer_idx=layer_idx,
            fuse_qk_norm_rope=False,
        )
        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias if hasattr(config, "mlp_bias") else False,
            dtype=config.torch_dtype,
            overridden_tp_size=1 if self.enable_attention_dp else None,
            config=model_config,
        )

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype,
                                       use_gemma=True)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype,
                                                use_gemma=True)
        self.layer_idx = layer_idx

        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy)
        self.next_layer_layernorm: RMSNorm = None

        self.fusion_config = EagerFusionConfig()
        self.enable_fusion = os.environ.get(
            "TRTLLM_QWEN3_EAGER_FUSION_DISABLED", "0") == "0"
        self.enable_fusion &= not self.enable_attention_dp

        # has_tp = self.mapping.has_tp()
        has_pp = self.mapping.has_pp()

        # self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
        self.fusion_config.PRE_MOE_FUSION = False
        self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION and not has_pp
        self.disable_attn_allreduce = (self.fusion_config.PRE_MOE_FUSION
                                       or self.mapping.tp_size == 1
                                       or self.enable_attention_dp)
        self.moe_allreduce = MoEAllReduce(mapping=model_config.mapping)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            self.fusion_config.POST_MOE_FUSION = False

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not self.disable_attn_allreduce),
            **kwargs,
        )

        if self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ))
        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        # Note: this fusion pattern is only supported for TRTLLM-nvfp4 backend now
        do_finalize = not (hidden_states.shape[0]
                           <= self.moe_allreduce.max_token
                           and self.fusion_config.POST_MOE_FUSION
                           and self.model_config.moe_backend == 'TRTLLM'
                           and self.mlp.experts.has_nvfp4)

        hidden_states = self.mlp(
            hidden_states,
            attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.POST_MOE_FUSION
                                      or self.mapping.tp_size == 1)),
            do_finalize=do_finalize,
        )

        if self.fusion_config.POST_MOE_FUSION:
            if do_finalize:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
            else:
                assert len(
                    hidden_states
                ) == 3, f"hidden_states must have 3 elements, but got {len(hidden_states)}"

                fc2_output = hidden_states[0]
                expert_scale_factor = hidden_states[1]
                expanded_idx_to_permuted_idx = hidden_states[2]

                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=None,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                hidden_states, residual = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params)

        else:
            if spec_metadata and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(
                    self.layer_idx, hidden_states, residual)
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(
                    hidden_states, residual)

        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "full_attention": Qwen3_5FullAttentionDecoderLayer,
    "linear_attention": Qwen3_5LinearDecoderLayer,
}


class Qwen3_5TextModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[Qwen3_5TextConfig]):
        super().__init__(model_config)
        config = self.model_config
        pretrained_config = self.model_config.pretrained_config
        self.aux_stream = torch.cuda.Stream()
        self.preload_weight_modules = []
        if config.moe_backend == "TRTLLM":
            self.preload_weight_modules = [
                "experts",
                "routing_method",
                "all_reduce",
            ]

        if model_config.mapping.enable_attention_dp:
            # When attention_dp is enabled, we cannot do all_reduce since
            # the problem size of different ranks are different.
            # So, we don't do parallelism here.
            self.embed_tokens = Embedding(pretrained_config.vocab_size,
                                          pretrained_config.hidden_size,
                                          dtype=pretrained_config.torch_dtype)
        else:
            self.embed_tokens = Embedding(
                pretrained_config.vocab_size,
                pretrained_config.hidden_size,
                dtype=pretrained_config.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )

        self.layers = nn.ModuleList([
            ALL_DECODER_LAYER_TYPES[pretrained_config.layer_types[layer_idx]](
                model_config,
                layer_idx,
                self.aux_stream,
            ) for layer_idx in range(pretrained_config.num_hidden_layers)
        ])

        self.norm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype,
            use_gemma=True,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        mamba_metadata = attn_metadata.mamba_metadata
        if mamba_metadata.max_batch_size != attn_metadata.max_num_requests:
            attn_metadata.mamba_metadata = Mamba2Metadata(
                attn_metadata.max_num_requests, chunk_size=128)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                mamba_metadata=mamba_metadata)
        return hidden_states


@register_auto_model("Qwen3_5ForCausalLM")
class Qwen3_5ForCausalLM(SpecDecOneEngineForCausalLM[Qwen3_5TextModel,
                                                       Qwen3_5TextConfig]):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3_5TextConfig],
    ):
        super().__init__(
            Qwen3_5TextModel(model_config),
            model_config,
        )
        self.preload_weight_modules = self.model.preload_weight_modules

    def load_weights(self, weights: dict, weight_mapper: BaseWeightMapper, params_map: Optional[Dict[str, str]] = None):
        new_weights = weight_mapper.preprocess_weights(weights)
        super().load_weights(new_weights, weight_mapper, params_map=params_map)

    def post_load_weights(self):
        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm

class Qwen3_5InputProcessorBase(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: AutoTokenizer,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self._dtype = self.config.text_config.dtype
        self._tokenizer = (
            tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_path)
        )
        self._model_path = model_path
        self._processor = AutoProcessor.from_pretrained(
            model_path, use_fast=True, trust_remote_code=trust_remote_code
        )
        self.tllm_multimodal_token_id = self.get_vocab_size() + 1
        # temporal patch size for video frames
        self.temporal_patch_size = getattr(self.config.vision_config, "temporal_patch_size", 1)

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def get_vocab_size(self) -> int:
        """Return the vocab size of the model."""
        return self.config.text_config.vocab_size

    @classmethod
    def get_rope_index(
        cls,
        model_config: PretrainedConfig,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3_5 use timestamps rather than absolute time position ids."""

        # Since we use timestamps to separate videos, like <t1> <vision_start> <frame1> <vision_end> <t2>
        # <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = model_config.vision_config.spatial_merge_size
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode
                    # the temporal information for videos)
                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _preprocess(
        self, text: Dict[str, Any], mm_data: Dict[str, Any], mm_processor_kwargs: Dict[str, Any]
    ):
        images = mm_data.get("image")
        video_datas = mm_data.get("video")
        if video_datas is not None:
            videos = [video_data.frames for video_data in video_datas]
        else:
            videos = None
        do_rescale = True
        if images and isinstance(images[0], torch.Tensor):
            do_rescale = False
        if videos and isinstance(videos[0][0], torch.Tensor):
            do_rescale = False
        return self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            do_rescale=do_rescale,
            return_tensors="pt",
            **mm_processor_kwargs,
        )

    def _postprocess(self, input_ids: torch.IntTensor) -> torch.IntTensor:
        masks = (input_ids == self.config.image_token_id) | (
            input_ids == self.config.video_token_id
        )
        input_ids[masks] = self.tllm_multimodal_token_id
        return input_ids

    def get_mrope_config(
        self,
        input_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor,
        video_grid_thw: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mrope_position_ids, mrope_position_deltas = Qwen3_5InputProcessorBase.get_rope_index(
            self.config, input_ids, image_grid_thw, video_grid_thw, attention_mask
        )

        mrope_config = {}
        mrope_config["mrope_position_ids"] = mrope_position_ids.to("cpu").clone()
        mrope_config["mrope_position_deltas"] = (
            mrope_position_deltas.to("cpu").to(torch.int32).clone()
        )

        return mrope_config

    @nvtx_range("Qwen3_5InputProcessorBase forward()")
    @torch.inference_mode()
    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = (
            inputs.get("prompt"),
            inputs.get("multi_modal_data", {}),
            inputs.get("mm_processor_kwargs", {}),
        )
        with nvtx_range_debug("transformers input preprocess"):
            processed_inputs = self._preprocess(text_prompt, mm_data, mm_processor_kwargs)

        multimodal_data = {}
        pixel_values = processed_inputs.get("pixel_values", None)
        if pixel_values is not None:
            multimodal_data["image"] = {
                "pixel_values": pixel_values.to(self.dtype),
                "image_grid_thw": processed_inputs.get("image_grid_thw"),
            }

        pixel_values_videos = processed_inputs.get("pixel_values_videos", None)
        if pixel_values_videos is not None:
            multimodal_data["video"] = {
                "pixel_values_videos": pixel_values_videos.to(self.dtype),
                "video_grid_thw": processed_inputs.get("video_grid_thw"),
            }

        # NOTE: Even on the text-only prompts, we still need 'mrope_position_ids'.
        mrope_config = self.get_mrope_config(
            processed_inputs["input_ids"],
            processed_inputs.get("image_grid_thw", None),
            processed_inputs.get("video_grid_thw", None),
            processed_inputs.get("attention_mask", None),
        )
        multimodal_data["mrope_config"] = mrope_config

        fused_input_ids = processed_inputs["input_ids"][0]
        if mm_data:
            fused_input_ids = self._postprocess(fused_input_ids)

        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }

    def get_prompt_token_ids(
        self, inputs: TextPrompt, mm_handles: List[Dict[str, Any]]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Build input token ids with multimodal placeholders expanded to the number of MM tokens.

        Args:
            inputs: Text prompt input container. Must contain a non-empty prompt string.
            mm_handles: List of multimodal embedding handles.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                - expanded_ids: token ids with each image token expanded to a placeholder repeated per MM token
                - mm_token_length: per-image MM token lengths
                - mm_token_offsets: start offsets (positions) for each image's MM tokens within expanded_ids
        """
        # TODO: Move this function to the base input processor class when extending for more models
        text_prompt = inputs.get("prompt")
        if not text_prompt:
            raise ValueError("Text prompt is required but not provided")

        if not isinstance(mm_handles, list):
            raise TypeError("mm_handles must be a list")

        num_deepstack_levels = len(self.config.vision_config.deepstack_visual_indexes)
        # This is because, unlike previous Qwen VL models, the embeddings are concatenated with
        # feature maps from deepstack layers.
        expected_size = self.config.text_config.hidden_size * (1 + num_deepstack_levels)
        for i, mm_handle in enumerate(mm_handles):
            hidden_size = mm_handle["tensor_size"][1]
            if hidden_size != expected_size:
                raise RuntimeError(
                    f"Expected multimodal embedding {i} to have hidden size {expected_size}, got {hidden_size}."
                )

        input_ids = self.tokenizer(text_prompt, return_tensors="pt").input_ids[0]

        # TODO: what about `video_token_id`?
        image_token_index = self.config.image_token_id

        image_mask = input_ids == image_token_index
        image_positions = torch.where(image_mask)[0]
        num_images = len(image_positions)
        assert num_images == len(mm_handles), "Number of images must match number of mm_handles"
        total_mm_tokens = sum(mm_handle["tensor_size"][0] for mm_handle in mm_handles)
        final_length = len(input_ids) - num_images + total_mm_tokens
        # Create output tensor
        expanded_ids = torch.empty(final_length, dtype=input_ids.dtype)
        placeholder_id = self.tllm_multimodal_token_id

        # Fill the expanded sequence
        write_pos = 0
        image_cnt = 0
        mm_token_length = []
        mm_token_offsets = []
        for read_pos in range(len(input_ids)):
            if input_ids[read_pos] == image_token_index:
                # Replace with placeholder id
                mm_token_num = mm_handles[image_cnt]["tensor_size"][0]
                expanded_ids[write_pos : write_pos + mm_token_num] = placeholder_id
                mm_token_offsets.append(write_pos)
                mm_token_length.append(mm_token_num)
                write_pos += mm_token_num
                image_cnt += 1
            else:
                # Copy text token as-is
                expanded_ids[write_pos] = input_ids[read_pos]
                write_pos += 1

        assert write_pos == final_length, f"Write position mismatch: {write_pos} != {final_length}"
        assert mm_token_length[-1] + mm_token_offsets[-1] <= final_length, (
            f"mm_token_length[-1] + mm_token_offsets[-1] ({mm_token_length[-1] + mm_token_offsets[-1]}) should be less "
            f"than or equal to final_length ({final_length})"
        )
        return expanded_ids.to(torch.int32).tolist(), mm_token_length, mm_token_offsets


class Qwen3_5VisionAttention(Qwen2_5_VLVisionAttention):
    def __init__(self, model_config, layer_idx):
        model_config.pretrained_config.max_position_embeddings = (
            model_config.pretrained_config.text_config.max_position_embeddings
        )
        model_config.pretrained_config.vision_config.torch_dtype = (
            model_config.pretrained_config.text_config.dtype
        )
        super().__init__(
            model_config,
            layer_idx=layer_idx,
            reduce_output=(
                not model_config.mapping.enable_attention_dp and model_config.mapping.tp_size > 1
            ),
        )


class Qwen3_5VisionMLP(MLP):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], layer_idx: int):
        config = model_config.pretrained_config.vision_config
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=True,
            activation=HF_ACT2FN[config.hidden_act],
            dtype=model_config.pretrained_config.text_config.dtype,
            config=model_config,
            layer_idx=layer_idx,
            overridden_tp_size=1 if model_config.mapping.enable_attention_dp else None,
        )


class Qwen3_5VisionBlock(torch.nn.Module):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], layer_idx: int):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config.vision_config

        self.norm1 = LayerNorm(
            hidden_size=config.hidden_size,
            eps=model_config.pretrained_config.text_config.rms_norm_eps,
            dtype=model_config.pretrained_config.text_config.dtype,
        )
        self.norm2 = LayerNorm(
            hidden_size=config.hidden_size,
            eps=model_config.pretrained_config.text_config.rms_norm_eps,
            dtype=model_config.pretrained_config.text_config.dtype,
        )
        self.attn = Qwen3_5VisionAttention(model_config, layer_idx)
        self.mlp = Qwen3_5VisionMLP(model_config, layer_idx)

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = residual + self.attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Qwen3_5VisionPatchMerger(torch.nn.Module):
    def __init__(
        self, model_config: ModelConfig[PretrainedConfig], use_postshuffle_norm: bool = False
    ) -> None:
        super().__init__()
        config = model_config.pretrained_config.vision_config
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = LayerNorm(
            hidden_size=self.hidden_size if use_postshuffle_norm else config.hidden_size,
            eps=model_config.pretrained_config.text_config.rms_norm_eps,
            dtype=model_config.pretrained_config.text_config.dtype,
        )

        self.mapping = model_config.mapping
        overridden_tp_size = 1 if model_config.mapping.enable_attention_dp else None
        if overridden_tp_size is not None:
            assert self.mapping.tp_size % overridden_tp_size == 0
            tp_size = overridden_tp_size
            # "Misuse" pp_size here to perform all-reduce within smaller groups
            pp_size = self.mapping.pp_size * self.mapping.tp_size // overridden_tp_size
            mapping = Mapping(
                world_size=tp_size * pp_size,
                rank=self.mapping.rank,
                gpus_per_node=self.mapping.gpus_per_node,
                tp_size=tp_size,
                pp_size=pp_size,
            )
        else:
            mapping = self.mapping

        self.linear_fc1 = Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=True,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = Linear(
            in_features=self.hidden_size,
            out_features=config.out_hidden_size,
            bias=True,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            allreduce_strategy=model_config.allreduce_strategy,
        )

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.hidden_size)

        hidden_states = self.norm(hidden_states).view(-1, self.hidden_size)
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_fc2(hidden_states)
        return hidden_states


class Qwen3_5VisionModel(torch.nn.Module):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        self.model_config = model_config
        self.config = self.model_config.pretrained_config.vision_config

        self.spatial_merge_size = self.config.spatial_merge_size
        self.patch_size = self.config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = HFQwen3_5VisionPatchEmbed(
            config=self.config,
        )

        self.pos_embed = nn.Embedding(self.config.num_position_embeddings, self.config.hidden_size)
        self.num_grid_per_side = int(self.config.num_position_embeddings**0.5)

        head_dim = self.config.hidden_size // self.config.num_heads
        self.rotary_pos_emb = HFQwen3_5VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen3_5VisionBlock(model_config, layer_idx=layer_idx)
                for layer_idx in range(self.config.depth)
            ]
        )
        self.merger = Qwen3_5VisionPatchMerger(
            model_config=model_config,
            use_postshuffle_norm=False,
        )
        self.deepstack_visual_indexes = self.config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3_5VisionPatchMerger(
                    model_config=model_config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )
        self.metadata_cls = get_attention_backend(self.model_config.attn_backend).Metadata

        self.attn_metadata = self.metadata_cls(
            max_num_requests=8192,  # TODO: Make this dynamic
            max_num_tokens=8192,  # TODO: Make this dynamic
            kv_cache_manager=None,
        )

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def prepare_attn_metadata(self, seq_lens, attn_metadata: AttentionMetadata):
        # NOTE: The single prompt is divided into multiple seq_lens, so pretending have many batch_sizes.
        batch_size = len(seq_lens)
        prompt_lens = seq_lens
        seq_lens = torch.tensor(seq_lens, dtype=torch.int, pin_memory=True)
        request_ids = list(range(1, batch_size + 1))

        attn_metadata.num_contexts = batch_size
        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = prompt_lens
        attn_metadata.seq_lens = seq_lens
        attn_metadata.max_seq_len = seq_lens.max().item()
        attn_metadata.prepare()
        return attn_metadata

    @torch.inference_mode()
    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        seq_lens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).tolist()
        attn_metadata = self.prepare_attn_metadata(seq_lens, self.attn_metadata)

        # Getting positional embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

        # From this point, pure GPU operation
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + pos_embeds
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                attn_metadata=attn_metadata,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists


class Qwen3_5VisionModelBase(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        model_class: Union[type[PreTrainedModel], type[torch.nn.Module]],
    ):
        super().__init__()
        self.model_config = model_config
        self.model_dtype = self.model_config.pretrained_config.text_config.dtype

        # NOTE: Re-setting QuantConfig to exclude vision encoder weights from quantization load.
        self.model_config.quant_config = QuantConfig(
            kv_cache_quant_algo=self.model_config.quant_config.kv_cache_quant_algo
        )

        self.visual = model_class(self.model_config).to(self.model_dtype)

        self.post_config()

    def post_config(self):
        self.config = self.model_config.pretrained_config.vision_config

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        visual_weights = filter_weights("model.visual", weights)
        converted_weights = {}

        qkv_pattern = re.compile(r"(.*?)attn\.qkv\.(.*)")
        for name in visual_weights:
            # Handle with weights and bias for vision transformer's qkv projection.
            match = qkv_pattern.match(name)
            if match:
                prefix, suffix = match.groups()
                q_name = f"{prefix}attn.q_proj.{suffix}"
                k_name = f"{prefix}attn.k_proj.{suffix}"
                v_name = f"{prefix}attn.v_proj.{suffix}"
                dim_shape = visual_weights[name].shape[0] // 3
                converted_weights[q_name] = visual_weights[name][:dim_shape]
                converted_weights[k_name] = visual_weights[name][dim_shape : 2 * dim_shape]
                converted_weights[v_name] = visual_weights[name][2 * dim_shape :]
            else:
                converted_weights[name] = visual_weights[name]
        pattern_mapping = {
            r"(.*?)attn.proj.(.*)": r"\1attn.o_proj.\2",
            r"(.*?)mlp.linear_fc1.(.*)": r"\1mlp.up_proj.\2",
            r"(.*?)mlp.linear_fc2.(.*)": r"\1mlp.down_proj.\2",
        }
        self.visual.config.num_attention_heads = self.visual.config.num_heads
        _load_weights_impl(self.visual, converted_weights, params_map=pattern_mapping)

    def _parse_and_batch_multimodal_data(
        self, multimodal_params: List[MultimodalParams]
    ) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
        pixel_values_list = []
        pixel_values_videos_list = []
        image_grid_thw_list = []
        video_grid_thw_list = []

        for multimodal_param in multimodal_params:
            multimodal_data = multimodal_param.multimodal_data
            # Process images if present
            if multimodal_data.get("image") is not None:
                pixel_values_list.append(multimodal_data["image"]["pixel_values"])
                image_grid_thw_list.append(multimodal_data["image"]["image_grid_thw"])

            # Process videos if present
            if multimodal_data.get("video") is not None:
                pixel_values_videos_list.append(multimodal_data["video"]["pixel_values_videos"])
                video_grid_thw_list.append(multimodal_data["video"]["video_grid_thw"])

        # Concatenate tensors
        mm_content_dict = {}
        if pixel_values_list:
            mm_content_dict["pixel_values"] = (
                torch.cat(pixel_values_list, dim=0)
                if len(pixel_values_list) > 1
                else pixel_values_list[0]
            )
        if pixel_values_videos_list:
            mm_content_dict["pixel_values_videos"] = (
                torch.cat(pixel_values_videos_list, dim=0)
                if len(pixel_values_videos_list) > 1
                else pixel_values_videos_list[0]
            )

        # Prepare extra data
        mm_extra_data = {}
        if image_grid_thw_list:
            mm_extra_data["image_grid_thw"] = (
                torch.cat(image_grid_thw_list, dim=0)
                if len(image_grid_thw_list) > 1
                else image_grid_thw_list[0]
            )
        if video_grid_thw_list:
            mm_extra_data["video_grid_thw"] = (
                torch.cat(video_grid_thw_list, dim=0)
                if len(video_grid_thw_list) > 1
                else video_grid_thw_list[0]
            )

        return mm_content_dict, mm_extra_data

    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]) -> List[torch.Tensor]:
        mm_content_data, mm_extra_data = self._parse_and_batch_multimodal_data(multimodal_params)
        pixel_values = mm_content_data.get("pixel_values", None)
        pixel_values_videos = mm_content_data.get("pixel_values_videos", None)

        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("Currently only support single modality per request")

        image_grid_thw = mm_extra_data.get("image_grid_thw", None)
        video_grid_thw = mm_extra_data.get("video_grid_thw", None)

        embeds = []
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.model_dtype)
            image_embeds, deepstack_image_embeds = self.visual(
                pixel_values, grid_thw=image_grid_thw
            )
            # NOTE: We concatenate deepstack_embeds to mm_embeds
            # The shape will be [seq_len, hidden_dim * (num_deepstack_layers + 1)]
            mixed_image_embeds = torch.cat([image_embeds] + deepstack_image_embeds, dim=1)
            embeds.append(mixed_image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(self.model_dtype)
            video_embeds, deepstack_video_embeds = self.visual(
                pixel_values_videos, grid_thw=video_grid_thw
            )
            # NOTE: We concatenate deepstack_embeds to mm_embeds
            # The shape will be [seq_len, hidden_dim * (num_deepstack_layers + 1)]
            mixed_video_embeds = torch.cat([video_embeds] + deepstack_video_embeds, dim=1)
            embeds.append(mixed_video_embeds)
        return embeds


class Qwen3_5ModelBase(PreTrainedModel):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        self.original_arch = model_config.pretrained_config.architectures[0]

        disable_fuse_rope = kwargs.get("disable_fuse_rope", False)
        model_config.pretrained_config.text_config.disable_fuse_rope = disable_fuse_rope
        # Be compatible with transformers version 4.xx, need to set rope_scaling
        # TODO: May need to remove this once we have a proper transformers package, and correectly deal with rope_scaling & rope_parameters
        if not hasattr(model_config.pretrained_config.text_config, "rope_scaling"):
            if not hasattr(model_config.pretrained_config.text_config, "rope_parameters"):
                raise ValueError("rope_scaling or rope_parameters must be set")
            model_config.pretrained_config.text_config.rope_scaling = model_config.pretrained_config.text_config.rope_parameters
        model_config.pretrained_config.text_config.rope_scaling["type"] = "mrope"
        config = model_config.pretrained_config

        self._supports_sdpa = True
        self._supports_flash_attn = True
        super().__init__(config)
        if not disable_fuse_rope:
            self.init_mrope_embedding(model_config)

        self.model_config = model_config

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = config.text_config
        if self.original_arch == "Qwen3_5ForConditionalGeneration":
            llm_model_config.pretrained_config.architectures = ["Qwen3_5ForCausalLM"]
        elif self.original_arch == "Qwen3_5MoeForConditionalGeneration":
            llm_model_config.pretrained_config.architectures = ["Qwen3_5MoeForCausalLM"]
        else:
            raise ValueError(f"Unsupported architecture: {self.original_arch}")
        # Qwen3_5ForCausalLM.
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        if not _is_disagg():
            self.mm_encoder = Qwen3_5VisionModelBase(
                model_config, kwargs.get("vision_model_class", None)
            ).eval()

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        )

        self.post_config()

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.model_config.pretrained_config = self.llm.config
        self.config = self.model_config.pretrained_config

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def init_mrope_embedding(self, model_config: ModelConfig[PretrainedConfig]):
        config = model_config.pretrained_config.text_config
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.from_string(config.rope_scaling["type"]),
            rope=RopeParams.from_config(config),
            mrope_section=config.rope_scaling.get("mrope_section", None),
            mrope_interleaved=config.rope_scaling.get("mrope_interleaved", False),
        )
        self.rotary_emb = MRotaryEmbedding(
            pos_embd_params.rope,
            head_dim=config.hidden_size // config.num_attention_heads,
            is_neox=pos_embd_params.is_neox,
            mrope_section=pos_embd_params.mrope_section,
            mrope_interleaved=pos_embd_params.mrope_interleaved,
        ).to("cuda")
        self.mrope_position_ids_padding_cuda = torch.zeros(
            (
                3,
                1,
                config.max_position_embeddings,
            ),
            dtype=torch.int32,
            device="cuda",
        )

    @nvtx_range("Qwen3_5 prepare_mrope_config")
    def prepare_mrope_config(
        self, multimodal_params: List[MultimodalParams], num_context_requests: int
    ):
        mrope_config = {}
        mrope_rotary_cos_sin = []
        mrope_position_deltas = []
        for multimodal_param in multimodal_params[:num_context_requests]:
            if multimodal_param.multimodal_data.get("mrope_config") is not None:
                with nvtx_range("Qwen3_5 get_cos_sin"):
                    if (
                        multimodal_param.multimodal_data["mrope_config"].get("mrope_position_ids")
                        is not None
                    ):
                        mrope_position_ids = multimodal_param.multimodal_data["mrope_config"][
                            "mrope_position_ids"
                        ]

                        self.mrope_position_ids_padding_cuda[
                            :, :, : mrope_position_ids.shape[-1]
                        ] = mrope_position_ids
                        self.mrope_position_ids_padding_cuda[
                            :, :, mrope_position_ids.shape[-1] :
                        ] = 0
                        cos, sin = self.rotary_emb.get_cos_sin(self.mrope_position_ids_padding_cuda)
                        concat_cos_sin = torch.stack((cos, sin), dim=-1)
                        concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0], -1)
                        mrope_rotary_cos_sin.append(concat_cos_sin)

        for multimodal_param in multimodal_params[num_context_requests:]:
            if multimodal_param.multimodal_data.get("mrope_config") is not None:
                if (
                    multimodal_param.multimodal_data["mrope_config"].get("mrope_position_deltas")
                    is not None
                ):
                    mrope_position_deltas.append(
                        multimodal_param.multimodal_data["mrope_config"]["mrope_position_deltas"]
                    )

        with nvtx_range("Qwen3_5 concat mrope_rotary_cos_sin"):
            if mrope_rotary_cos_sin:
                mrope_config["mrope_rotary_cos_sin"] = torch.cat(mrope_rotary_cos_sin, dim=0)
        with nvtx_range("Qwen3_5 concat mrope_position_deltas"):
            if mrope_position_deltas:
                mrope_config["mrope_position_deltas"] = torch.cat(mrope_position_deltas, dim=0)

        return mrope_config

    def split_mm_embeds(self, mm_embed, deepstack_num_level):
        num_elements = mm_embed.shape[1] // (deepstack_num_level + 1)
        mm_embed_chunks = torch.split(mm_embed, [num_elements] * (deepstack_num_level + 1), dim=1)
        return mm_embed_chunks[0], list(mm_embed_chunks[1:])

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        VLM forward logic with inflight batching support.
        """
        num_context_requests, num_generation_requests = (
            attn_metadata.num_contexts,
            attn_metadata.num_generations,
        )
        logger.debug(
            f"num_context_requests: {num_context_requests}, num_generation_requests: {num_generation_requests}"
        )

        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embeds = []
        mrope_config = {}
        deepstack_embeds = []

        # NOTE: Qwen*-VL series has mrope_config even on the text-only prompts,
        # so we need to separate the mm_multimodal_params from the text-only prompts.
        mm_multimodal_params = self._get_requests_with_mm_data(multimodal_params)
        if len(mm_multimodal_params) > 0:
            if not _is_disagg():
                mm_embeds = get_multimodal_embeddings(
                    encoder_forward_fn=self.mm_encoder.forward,
                    multimodal_params=mm_multimodal_params,
                )
            elif not getattr(self, "support_mm_disagg", False):
                raise NotImplementedError(
                    f"{type(self)} does not support disaggregated inference yet. Please unset "
                    "the TLLM_MULTIMODAL_DISAGGREGATED environment variable, or set it to '0'."
                )
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_multimodal_params)

            if self.use_deepstack:
                for i, mm_embed in enumerate(mm_embeds):
                    mm_embed, deepstack_embed = self.split_mm_embeds(
                        mm_embed, self.deepstack_num_level
                    )
                    mm_embeds[i] = mm_embed
                    deepstack_embeds.extend(deepstack_embed)

        if not self.model_config.pretrained_config.disable_fuse_rope:
            mrope_config = self.prepare_mrope_config(multimodal_params, num_context_requests)

        result = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embeds,
            extra_embeds=deepstack_embeds,
            **kwargs,
        )
        if len(deepstack_embeds) > 0:
            input_ids, input_embeds, deepstack_embeds = result
        else:
            input_ids, input_embeds = result

        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits,
            deepstack_embeds=deepstack_embeds,
            mrope_config=mrope_config,
        )
        logger.debug(f"output shape: {output_prob.shape}")
        return output_prob

    def _get_requests_with_mm_data(self, multimodal_params):
        mm_multimodal_params = []
        for multimodal_param in multimodal_params:
            data = multimodal_param.multimodal_data
            if (
                # The first 2 conditions check whether there is input on which inference should be run.
                data.get("image", {}).get("pixel_values") is not None
                or data.get("video", {}).get("pixel_values_videos") is not None
                # This condition corresponds to when the embeddings are already populated, as is e.g.
                # the case in EPD disagg in the prefill worker.
                or data.get("multimodal_embedding")
            ):
                mm_multimodal_params.append(multimodal_param)

        return mm_multimodal_params


@support_multimodal_disaggregated
@register_vision_encoder(Qwen3_5VisionModelBase, vlm_base_model=Qwen3_5VisionModel)
@register_auto_model("Qwen3_5ForConditionalGeneration")
@register_input_processor(
    Qwen3_5InputProcessorBase,
    model_type="qwen3_5",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ),
)
class Qwen3_5Model(Qwen3_5ModelBase):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        # NOTE: HF implementation.
        kwargs["vision_model_class"] = Qwen3_5VisionModel
        kwargs["disable_fuse_rope"] = kwargs.get(
            "disable_fuse_rope", False
        )  # TODO: Make this ModelConfig's argument
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return ["image.pixel_values", "video.pixel_values_videos", "multimodal_embedding"]

    def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeightMapper):
        if not _is_disagg():
            self.mm_encoder.load_weights(weights)

        weight_mapper = Qwen3_5HfWeightMapper()
        weight_mapper.init_model_and_config(self.llm, self.model_config)
        filtered_weights = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
        params_map = {
            r"^model\.language_model\.(.*)$": r"model.\1",
        }
        self.llm.load_weights(filtered_weights, weight_mapper, params_map=params_map)
