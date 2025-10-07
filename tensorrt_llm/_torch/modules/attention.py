import math
import weakref
from typing import Optional, Union, cast

import torch
from torch import nn

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import (AttentionInputType, AttentionMetadata,
                                 FlashInferAttentionMetadata, TrtllmAttention,
                                 TrtllmAttentionMetadata)
from ..attention_backend.interface import (AttentionMask,
                                           PositionalEmbeddingParams,
                                           PredefinedAttentionMask)
from ..attention_backend.utils import create_attention, get_attention_backend
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..peft.lora.layer import LoraLayer, LoraModuleType
from ..utils import (Fp4QuantizedTensor, get_model_extra_attrs,
                     is_torch_compiling)
from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig
from .multi_stream_utils import maybe_execute_in_parallel
from .rms_norm import RMSNorm
from .rotary_embedding import MRotaryEmbedding, RotaryEmbedding


def extract_extra_attrs(layer_idx: str, attn_type: str):
    assert attn_type in ["mla", "attn"], "Invalid attention type"
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    if attn_type == "mla":
        assert isinstance(
            metadata,
            TrtllmAttentionMetadata,
        )
    else:
        assert isinstance(
            metadata,
            FlashInferAttentionMetadata,
        ) or isinstance(
            metadata,
            TrtllmAttentionMetadata,
        )

    attn_layers = extra_attrs.get(attn_type + "_layers", None)
    assert attn_layers is not None, "Attention layer is not registered"
    attn_layer_ref = attn_layers.get(layer_idx, None)
    assert attn_layer_ref is not None, f"Cannot find attention layer for layer {layer_idx}"
    attn_layer = attn_layer_ref()

    if attn_type == "mla":
        assert isinstance(
            attn_layer,
            MLA), "MLA layer must be a subclass of MLA or an instance of MLA"
    elif attn_type == "attn":
        assert isinstance(
            attn_layer, Attention
        ), "Attention layer must be a subclass of Attention or an instance of Attention"

    return metadata, attn_layer


@torch.compile
def compiled_copy_(dst, src):
    dst.copy_(src)


@torch.compile
def compiled_cat(tensors, dim):
    return torch.cat(tensors, dim)


@torch.library.custom_op("trtllm::attn_custom_op_inplace",
                         mutates_args=("output", ))
def attn_custom_op_inplace(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    attention_mask: str,
    mrope_rotary_cos_sin: Optional[torch.Tensor],
    mrope_position_deltas: Optional[torch.Tensor],
    attention_window_size: Optional[int],
    attention_mask_data: Optional[torch.Tensor],
    attention_sinks: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    metadata, attn_layer = extract_extra_attrs(layer_idx, "attn")
    # NVFP4 output cannot be supported by torch compile for TRTLLM backend.
    attn_layer._attn_impl(q,
                          k,
                          v,
                          metadata,
                          PredefinedAttentionMask(attention_mask),
                          mrope_rotary_cos_sin,
                          mrope_position_deltas,
                          attention_window_size,
                          attention_mask_data,
                          enable_attn_nvfp4_output=False,
                          output=output,
                          attention_sinks=attention_sinks)


class Attention(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        bias: bool,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        rope_fusion: Optional[bool] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        q_scaling: float = 1.0,
        attention_chunk_size: Optional[int] = None,
        disable_deep_gemm: bool = False,
        attn_output_gate: Optional[bool] = None,
        use_custom_cublas_mm: bool = False,
    ):
        """
        Initialize the Attention module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            pos_embd_params (Optional[PositionalEmbeddingParams]): The positional embedding parameters.
            rope_fusion (Optional[bool]): Whether to fuse RoPE into the attention OP and skip applying unfused RoPE. If None, whether to fuse is decided by the capability of the attention backend.
            layer_idx (Optional[int]): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (Optional[bool]): Whether to use bias in the output projection layer.
            config (Optional[ModelConfig]): The model configuration.
            q_scaling (float): The scaling factor for the qk_scale. The definition is $O = softmax(QK^T * qk_scale) * V, qk_scale = 1 / (sqrt(head_dim) * q_scaling)$. The default value is 1.0.
            attention_chunk_size (Optional[int]): See [Chunked Attention] below.
            disable_deep_gemm (bool): Whether to disable the use of DeepGEMM in Linear layers (currently only matters on SM100 + FP8).
            attn_output_gate (Optional[bool]): Determines whether to use an output gate in the attention Op. If False, the decision is automatically handled by the attention backend based on its capabilities.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)

        self.register_to_config = False
        # We only register TRTLLM attention layers to config.
        if config is not None:
            if "attn_layers" not in config.extra_attrs:
                config.extra_attrs["attn_layers"] = {}
            config.extra_attrs["attn_layers"][self.layer_idx_str] = weakref.ref(
                self)
            self.register_to_config = True

        config = config or ModelConfig()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = getattr(config.pretrained_config, 'head_dim', None)
        if not isinstance(self.head_dim, int):
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        self.q_scaling = q_scaling
        self.attn_output_gate = attn_output_gate

        if self.attn_output_gate:
            logger.info_once("using attn output gate!", key="attn_output_gate")

        # [Chunked Attention]
        # Chunked attention is applied to context requests only. Chunked attention will be
        # applied when this field is specified and mMaskType == CAUSAL.
        #
        # In chunked attention, we break context requests into chunks of a specified size. Tokens can only
        # attend to tokens in the same chunk. So, for example, if the chunk size is 3, we might have a mask
        # that looks like this:
        #
        # 1 0 0 0 0 0
        # 1 1 0 0 0 0
        # 1 1 1 0 0 0
        # 0 0 0 1 0 0
        # 0 0 0 1 1 0
        # 0 0 0 1 1 1
        self.attention_chunk_size = attention_chunk_size

        if dense_bias is None:
            self.dense_bias = bias

        # tensor parallel
        tp_size = config.mapping.tp_size
        pp_size = config.mapping.pp_size
        if config.mapping.enable_attention_dp:
            tp_size = 1

        mapping = Mapping(
            world_size=tp_size * pp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            rank=config.mapping.rank,
            gpus_per_node=config.mapping.gpus_per_node,
            enable_attention_dp=config.mapping.enable_attention_dp,
        )
        self.tp_size = tp_size
        self.tp_rank = mapping.tp_rank
        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.qkv_proj = Linear(
            self.hidden_size,
            tp_size * self.q_size * (2 if self.attn_output_gate else 1) +
            2 * tp_size * self.kv_size,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            disable_deep_gemm=disable_deep_gemm,
            use_custom_cublas_mm=use_custom_cublas_mm)

        self.o_lora = LoraLayer([LoraModuleType.ATTENTION_DENSE],
                                [self.hidden_size])

        self.o_proj = Linear(
            tp_size * self.q_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            lora=self.o_lora,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            disable_deep_gemm=disable_deep_gemm,
            use_custom_cublas_mm=use_custom_cublas_mm)

        self.quant_config = config.get_quant_config()
        self.attn_backend = config.attn_backend
        attn_cls = get_attention_backend(
            self.attn_backend,
            sparse_attn_config=config.sparse_attention_config)

        # These two modules are mutually exclusive - either splitted_qkv_lora or fused_qkv_lora will be used,
        # but never both at the same time. splitted_qkv_lora handles Q,K,V separately while fused_qkv_lora
        # handles them as a single fused operation.
        self.splitted_qkv_lora = LoraLayer([
            LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K,
            LoraModuleType.ATTENTION_V
        ], [self.q_size, self.kv_size, self.kv_size])
        self.fused_qkv_lora = LoraLayer([LoraModuleType.ATTENTION_QKV],
                                        [self.q_size + 2 * self.kv_size])

        self.o_lora = LoraLayer([LoraModuleType.ATTENTION_DENSE],
                                [self.hidden_size])

        # Whether to fuse RoPE into the attention OP.
        # If true, RoPE will be applied in self.attn.forward.
        # If false, RoPE will be applied in self.apply_rope.
        if config.sparse_attention_config is not None:
            logger.warning("disable rope_fusion for sparse attention.")
            rope_fusion = False
        self.rope_fusion = rope_fusion
        if self.rope_fusion and not attn_cls.support_fused_rope():
            logger.warning(
                "rope_fusion is true but the attention backend does not support it. Will disable rope_fusion."
            )
            self.rope_fusion = False
        # If rope_fusion is not specified, enable if the attention backend supports it.
        if self.rope_fusion is None:
            self.rope_fusion = attn_cls.support_fused_rope()

        self.rotary_emb = None
        if not self.rope_fusion and self.pos_embd_params is not None:
            if self.pos_embd_params.type.is_mrope():
                self.rotary_emb = MRotaryEmbedding(
                    self.pos_embd_params.rope,
                    head_dim=self.head_dim,
                    is_neox=self.pos_embd_params.is_neox,
                    mrope_section=self.pos_embd_params.mrope_section,
                )
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.pos_embd_params.rope,
                    head_dim=self.head_dim,
                    is_neox=self.pos_embd_params.is_neox,
                )

        self.attn = create_attention(
            self.attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            pos_embd_params=self.pos_embd_params if self.rope_fusion else None,
            quant_config=self.quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            q_scaling=self.q_scaling,
            attention_chunk_size=self.attention_chunk_size,
            sparse_attention_config=config.sparse_attention_config,
        )

        self.support_fused_qkv = self.attn.support_fused_qkv()
        self.support_nvfp4_output = self.attn.support_nvfp4_output()

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.attn has no weights but has states that are related to quant_config,
        # which could be modified after __init__
        self.attn.update_quant_config(self.quant_config)

        self.o_proj.create_weights()
        self.has_quant_scale = (self.o_proj.has_fp8_qdq or self.o_proj.has_nvfp4
                                or self.o_proj.has_fp8_block_scales
                                or self.o_proj.has_fp8_rowwise
                                or self.o_proj.has_w4a8_nvfp4_fp8)

    def split_qkv(self, q, k=None, v=None):
        if k is None and v is None:
            q, k, v = q.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def convert_qkv(self, q, k, v):
        if k is None and v is None and not self.support_fused_qkv:
            q, k, v = self.split_qkv(q)
        elif k is not None and v is not None and self.support_fused_qkv:
            qkv = torch.concat([q, k, v], dim=-1)
            q, k, v = qkv, None, None
        return q, k, v

    def create_output(self, q: torch.Tensor):
        num_tokens = q.shape[0]
        hidden_size = self.o_proj.in_features
        out_dtype = q.dtype

        if self.attn_backend == "TRTLLM":
            if self.has_quant_scale and (self.attn.has_fp8_kv_cache
                                         or self.attn.has_fp4_kv_cache):
                out_dtype = torch.float8_e4m3fn
        output = q.new_empty([num_tokens, hidden_size], dtype=out_dtype)
        return output

    def _attn_impl(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask,
        mrope_rotary_cos_sin: Optional[torch.Tensor],
        mrope_position_deltas: Optional[torch.Tensor],
        attention_window_size: Optional[int],
        attention_mask_data: Optional[torch.Tensor],
        enable_attn_nvfp4_output: bool = True,
        output: Optional[torch.Tensor] = None,
        output_sf: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
    ):
        num_tokens = attn_metadata.num_tokens

        q = q[:num_tokens, :]
        if k is not None:
            k = k[:num_tokens, :]
        if v is not None:
            v = v[:num_tokens, :]

        out_scale = None
        out_scale_sf = None
        if self.has_quant_scale:
            out_scale = self.o_proj.inv_input_scale
        if self.o_proj.has_nvfp4 and self.support_nvfp4_output and enable_attn_nvfp4_output:
            out_scale_sf = self.o_proj.input_scale

        kv_scales_sf = None
        kv_scales_sf_inv = None
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp4_kv_cache(
        ):
            kv_scales_sf = self.qkv_proj.kv_scales
            kv_scales_sf_inv = self.qkv_proj.inv_kv_scales

        mrope_config = None
        if mrope_rotary_cos_sin is not None or mrope_position_deltas is not None:
            mrope_config = dict()
            if mrope_rotary_cos_sin is not None:
                mrope_config["mrope_rotary_cos_sin"] = mrope_rotary_cos_sin
            if mrope_position_deltas is not None:
                mrope_config["mrope_position_deltas"] = mrope_position_deltas

        attn_output = self.attn.forward(
            q,
            k,
            v,
            attn_metadata,
            out_scale=out_scale,
            out_scale_sf=out_scale_sf,
            kv_scales_sf=kv_scales_sf,
            kv_scales_sf_inv=kv_scales_sf_inv,
            attention_mask=attention_mask,
            mrope_config=mrope_config,
            attention_window_size=attention_window_size,
            attention_mask_data=attention_mask_data,
            enable_attn_nvfp4_output=enable_attn_nvfp4_output,
            output=output[:num_tokens, :] if output is not None else None,
            output_sf=output_sf,
            attention_sinks=attention_sinks)
        if isinstance(attn_output, tuple):
            assert len(
                attn_output
            ) == 2, "attn_output should be a tuple of (output, output_sf)"
            return attn_output[0], attn_output[1]
        return attn_output, None

    def forward_impl(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask,
        attention_window_size: Optional[int],
        attention_mask_data: Optional[torch.Tensor],
        mrope_config: Optional[dict],
        attention_sinks: Optional[torch.Tensor] = None,
    ):
        mrope_rotary_cos_sin = None
        mrope_position_deltas = None
        if mrope_config is not None:
            if "mrope_rotary_cos_sin" in mrope_config:
                mrope_rotary_cos_sin = mrope_config["mrope_rotary_cos_sin"]
            if "mrope_position_deltas" in mrope_config:
                mrope_position_deltas = mrope_config["mrope_position_deltas"]

        # Currently only TRTLLM and FLASHINFER are torch compile compatible backends.
        # Only enable custom inplace op when torch compiling.
        use_custom_inplace_op = (self.register_to_config
                                 and (self.attn_backend == "TRTLLM"
                                      or self.attn_backend == "FLASHINFER")
                                 and is_torch_compiling())

        if use_custom_inplace_op:
            output = self.create_output(q)
            attn_custom_op_inplace(
                q,
                k,
                v,
                attention_mask,
                mrope_rotary_cos_sin,
                mrope_position_deltas,
                attention_window_size,
                attention_mask_data,
                attention_sinks,
                self.layer_idx_str,
                output,
            )
        else:
            output, output_sf = self._attn_impl(q,
                                                k,
                                                v,
                                                attn_metadata,
                                                attention_mask,
                                                mrope_rotary_cos_sin,
                                                mrope_position_deltas,
                                                attention_window_size,
                                                attention_mask_data,
                                                attention_sinks=attention_sinks)
            if output_sf is not None:
                output = Fp4QuantizedTensor(output, output_sf)

        return output

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        attention_window_size: Optional[int] = None,
        attention_mask_data: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for the Attention module.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            attention_mask (AttentionMask): The attention mask type.
            mrope_config (Optional[dict]): The MROPE configuration.
            all_reduce_params (Optional[AllReduceParams]): The all reduce parameters.
            lora_params (Optional[dict]): The LoRA parameters.
            attention_window_size (Optional[int]): The attention window size.
            attention_mask_data (Optional[torch.Tensor]): The attention mask data.
        Returns:
            torch.Tensor: The output tensor.
        """
        qkv = self.qkv_proj(hidden_states)

        if bool(lora_params):
            qkv_lora = self.splitted_qkv_lora(hidden_states, lora_params,
                                              self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

            qkv_lora = self.fused_qkv_lora(hidden_states, lora_params,
                                           self.layer_idx)
            if qkv_lora is not None:
                qkv = qkv + qkv_lora

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            # Single line: view -> chunk -> reshape both q and gate
            q, gate = [
                t.reshape(*orig_shape, -1) for t in torch.chunk(
                    q_gate.view(*orig_shape, self.num_heads, -1), 2, dim=-1)
            ]
        else:
            q, k, v = qkv, None, None

        q, k, v = self.apply_rope(q, k, v, position_ids)
        q, k, v = self.convert_qkv(q, k, v)

        if attention_sinks is not None:
            assert self.attn_backend == "TRTLLM", "Attention sinks are only supported for TRTLLM backend."

        attn_output = self.forward_impl(q,
                                        k,
                                        v,
                                        attn_metadata,
                                        attention_mask,
                                        attention_window_size,
                                        attention_mask_data,
                                        mrope_config=mrope_config,
                                        attention_sinks=attention_sinks)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params,
                                  lora_params=lora_params,
                                  layer_idx=self.layer_idx)
        return attn_output

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        """
        Apply RoPE to the query and key.
        Depending on the implementation, q, k, v could be either fused (q, k, v = concat(q, k, v), None, None) or unfused (none of q, k, v is None).
        Before self.attn.forward, convert_qkv will be called to make sure that the format of (q, k, v) satisfies the requirement of self.attn.
        This method could be overridden in the subclass, in which extra functionalities such as q_norm/k_norm could be added.
        Args:
            q (torch.Tensor): The query tensor.
            k (Optional[torch.Tensor]): The key tensor.
            v (Optional[torch.Tensor]): The value tensor.
            position_ids (torch.Tensor): The position IDs of each token for RoPE.
        Returns:
            tuple: A tuple of (q, k, v).
        """
        # If RoPE is fused into the attention OP, do not apply RoPE here.
        if not self.rope_fusion and position_ids is not None:
            q, k, v = self.split_qkv(q, k, v)
            q, k = self.rotary_emb(position_ids, [q, k])
        return q, k, v

    def apply_qk_norm(self, q, k):
        raise NotImplementedError(
            f"QK norm is not implemented for {self.__class__.__name__}."
            "Please override the `apply_qk_norm` method in the subclass.")


@torch.library.custom_op("trtllm::mla_custom_op_inplace",
                         mutates_args=("output", ))
def mla_custom_op_inplace(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    mla_layer.forward_impl(position_ids, hidden_states, metadata, output=output)


def fp8_block_scaling_bmm_out(
    mat1: torch.Tensor,
    mat2_fp8: torch.Tensor,
    mat2_scale: torch.Tensor,
    out: torch.Tensor,
    mat2_dequant: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sm_version = get_sm_version()
    if sm_version == 90 or sm_version == 89:
        mat1_fp8, mat1_scale = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
            mat1)

        output = out.new_empty(out.shape, dtype=out.dtype, device=out.device)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(mat1_fp8, mat2_fp8,
                                                   mat1_scale, mat2_scale,
                                                   output)
        out.copy_(output)

    elif is_sm_100f(sm_version):
        torch.bmm(mat1.transpose(0, 1), mat2_dequant.transpose(1, 2), out=out)
    else:
        raise NotImplementedError(f"SM{sm_version} is not supported")


class MLA(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        predicted_tokens_per_seq: int,
        max_position_embeddings: int,
        bias: bool,
        aux_stream: Optional[torch.cuda.Stream] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
    ):
        """
        Initialize the MLA module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            qk_nope_head_dim (int): The dimension of the query and key without Rope.
            qk_rope_head_dim (int): The dimension of the Rope of query and key.
            v_head_dim (int): The dimension of the value.
            q_lora_rank (int): The dimension of the compressed query.
            kv_lora_rank (int): The dimension of the compressed key and value.
            predicted_tokens_per_seq (int): The number of predicted tokens per sequence.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            aux_stream (Optional[torch.cuda.Stream]): The auxiliary CUDA stream for running operations in two parallel streams.
            pos_embd_params (PositionalEmbeddingParams): The positional embedding parameters.
            layer_idx (int): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (bool): Whether to use bias in the output projection layer.
            config (ModelConfig): The model configuration.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.predicted_tokens_per_seq = predicted_tokens_per_seq
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        if dense_bias is None:
            self.dense_bias = bias

        if self.q_lora_rank is None:
            self.q_lora_rank = hidden_size
            self.is_lite = True
        else:
            self.is_lite = False

        assert pos_embd_params is not None, "pos_embd_params must be provided in MLA"

        self.register_to_config = False
        if config is not None:
            if "mla_layers" not in config.extra_attrs:
                config.extra_attrs["mla_layers"] = {}
            config.extra_attrs["mla_layers"][self.layer_idx_str] = weakref.ref(
                self)
            self.register_to_config = True

        # only support one kind of sparse attention, dsa now.
        if config.sparse_attention_config is not None:
            self.is_dsa = True
        else:
            self.is_dsa = False

        # tensor parallel
        config = config or ModelConfig()
        tp_size = config.mapping.tp_size
        pp_size = config.mapping.pp_size
        if config.mapping.enable_attention_dp:
            tp_size = 1

        mapping = Mapping(
            world_size=tp_size * pp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            rank=config.mapping.rank,
            gpus_per_node=config.mapping.gpus_per_node,
            enable_attention_dp=config.mapping.enable_attention_dp,
        )

        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size

        rms_norm_eps = config.pretrained_config.rms_norm_eps
        quant_config = config.get_quant_config()
        self.quant_config = quant_config

        if not self.is_lite:
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization)

            self.q_a_layernorm = RMSNorm(hidden_size=self.q_lora_rank,
                                         eps=rms_norm_eps,
                                         dtype=dtype)

            self.q_b_proj = Linear(
                self.q_lora_rank,
                tp_size * self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization)
        else:
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization)

            self.q_proj = Linear(
                self.q_lora_rank,
                tp_size * self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization)
            self.q_b_proj = self.q_proj

        self.kv_a_layernorm = RMSNorm(hidden_size=kv_lora_rank,
                                      dtype=dtype,
                                      eps=rms_norm_eps)

        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            tp_size * self.num_heads *
            (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization)
        # This parameter will view into self.kv_b_proj.weight after loading weights.
        # For dummy weight initialization, this parameter is initialized with empty tensor.
        # Used in forward_generation only
        self.v_b_proj = nn.Parameter(
            torch.empty(
                (self.num_heads, self.v_head_dim, self.kv_lora_rank),
                dtype=dtype,
            ),
            requires_grad=False,
        )

        self.o_proj = Linear(
            self.num_key_value_heads * self.v_head_dim * tp_size,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        mscale_all_dim = pos_embd_params.rope.mscale_all_dim
        scaling_factor = pos_embd_params.rope.scale
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        q_scaling = 1.0 / (mscale * mscale)

        if not self.is_dsa:
            self.mha = create_attention(
                config.attn_backend,
                self.layer_idx,
                self.num_heads,
                head_dim=self.qk_head_dim,
                num_kv_heads=self.num_key_value_heads,
                pos_embd_params=pos_embd_params,
                quant_config=quant_config,
                q_scaling=q_scaling,
                is_mla_enable=True,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                predicted_tokens_per_seq=self.predicted_tokens_per_seq,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                sparse_attention_config=config.sparse_attention_config,
            )

        self.mqa = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            num_kv_heads=1,
            pos_embd_params=pos_embd_params,
            quant_config=quant_config,
            q_scaling=q_scaling,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.kv_lora_rank,
            hidden_size=self.hidden_size,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            sparse_attention_config=config.sparse_attention_config,
            dtype=dtype,
        )

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        self.rope_fusion = self.mqa.support_fused_rope()
        self.rotary_emb = None
        self.apply_rotary_emb = not self.rope_fusion
        if self.apply_rotary_emb:
            self.rotary_emb = RotaryEmbedding(
                pos_embd_params.rope,
                head_dim=self.qk_rope_head_dim,
                is_neox=pos_embd_params.is_neox,
            )

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.mha/mqa has no weights but has states that are related to quant_config,
        # which could be modified after __init__
        if not self.is_dsa:
            self.mha.update_quant_config(self.quant_config)
        self.mqa.update_quant_config(self.quant_config)

        # Although we use FP8 MLA for context/generation phase, the output is still in BF16
        self.out_scale = None

        # k_b_proj_trans's dtype must be consistent with self.kv_b_proj,
        # which can be modified after __init__
        has_fp8_block_scales = (
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales())

        mla_weight_dtype = torch.float8_e4m3fn if has_fp8_block_scales else self.dtype
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (self.num_heads, self.kv_lora_rank, self.qk_nope_head_dim),
                dtype=mla_weight_dtype,
            ),
            requires_grad=False,
        )

        self.k_b_proj_trans_dequant = None
        self.v_b_proj_dequant = None
        if has_fp8_block_scales:
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads,
                        self.kv_lora_rank // 128,
                        self.qk_nope_head_dim // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            # This parameter will view into self.kv_b_proj.weight_scale after loading weights.
            # For dummy weight initialization, this parameter is initialized with empty tensor.
            self.v_b_proj_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads,
                        self.v_head_dim // 128,
                        self.kv_lora_rank // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            if is_sm_100f():
                assert self.dtype == torch.bfloat16
                self.k_b_proj_trans_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads, self.kv_lora_rank,
                         self.qk_nope_head_dim),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
                self.v_b_proj_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads, self.v_head_dim, self.kv_lora_rank),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None

    def apply_rope(
        self,
        q: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        q_pe = q[..., self.qk_nope_head_dim:].reshape(
            -1, self.num_heads * self.qk_rope_head_dim)
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe])
        q[..., self.qk_nope_head_dim:] = q_pe.view(-1, self.num_heads,
                                                   self.qk_rope_head_dim)
        return k_pe

    def create_output(self, hidden_states: torch.Tensor):
        num_tokens = hidden_states.shape[0]
        hidden_size = self.o_proj.in_features
        return hidden_states.new_empty([num_tokens, hidden_size],
                                       dtype=hidden_states.dtype)

    def forward_impl(self, position_ids: Optional[torch.Tensor],
                     hidden_states: torch.Tensor,
                     attn_metadata: AttentionMetadata,
                     output: torch.Tensor) -> None:
        """
        Forward pass for the MLA module.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            all_reduce_params (Optional[AllReduceParams]): The all reduce parameters.

        Returns:
            torch.Tensor: The output tensor.
        """
        # split q, k, v into context and gen batches
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        hidden_states = hidden_states[:num_tokens, ...]
        if position_ids is not None:
            position_ids = position_ids[..., :num_tokens]

        if self.is_lite:
            compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
                [self.kv_lora_rank, self.qk_rope_head_dim], -1)
            compressed_kv = self.kv_a_layernorm(compressed_kv)
            q = hidden_states
        else:
            q, compressed_kv, k_pe = self.kv_a_proj_with_mqa(
                hidden_states).split([
                    self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim
                ], -1)

            q, compressed_kv = maybe_execute_in_parallel(
                lambda: self.q_a_layernorm(q),
                lambda: self.kv_a_layernorm(compressed_kv),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )

        qr = q
        q, latent_cache = maybe_execute_in_parallel(
            lambda: self.q_b_proj(q),
            lambda: torch.concat([compressed_kv, k_pe], dim=-1),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        assert q.shape[
            0] == num_tokens, f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"

        assert output is not None, "output must be provided"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
            hidden_states_ctx = hidden_states[:num_ctx_tokens, ...]
            qr_ctx = qr[:num_ctx_tokens, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, position_ids)

            self.forward_context(
                q_ctx,
                compressed_kv_ctx,
                k_pe_ctx,
                attn_metadata,
                output[:num_ctx_tokens, :],
                latent_cache_ctx,
                hidden_states_ctx,
                qr_ctx,
                position_ids,
            )

        if num_generations > 0:
            q_gen = q[num_ctx_tokens:, ...]
            compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
            k_pe_gen = k_pe[num_ctx_tokens:, ...]
            latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
            hidden_states_gen = hidden_states[num_ctx_tokens:, ...]
            qr_gen = qr[num_ctx_tokens:, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_gen = self.apply_rope(q_gen, k_pe_gen, position_ids)

            self.forward_generation(
                q_gen,
                compressed_kv_gen,
                k_pe_gen,
                attn_metadata,
                output[num_ctx_tokens:num_tokens, :],
                latent_cache_gen,
                hidden_states_gen,
                qr_gen,
                position_ids,
            )

    def forward_context_default(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kv = self.kv_b_proj(compressed_kv)
        k_nope, v = kv.split(
            [
                self.num_heads * self.qk_nope_head_dim,
                self.num_heads * self.v_head_dim
            ],
            -1,
        )

        k = torch.empty_like(q).view(-1, self.num_heads, self.qk_head_dim)
        compiled_copy_(k[..., :self.qk_nope_head_dim],
                       k_nope.view(-1, self.num_heads, self.qk_nope_head_dim))
        if self.apply_rotary_emb:
            k[..., self.qk_nope_head_dim:] = k_pe.view(-1, 1,
                                                       self.qk_rope_head_dim)
        k = k.view(-1, self.num_heads * self.qk_head_dim)

        attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache,
            out_scale=self.out_scale,
            output=output,
        )

        return attn_output

    def forward_context_dsa(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_sparse_mla_kvcache_bf16(q, compressed_kv, k_pe, attn_metadata,
                                       output, latent_cache, hidden_states, qr,
                                       position_ids, is_generation = False)
    
    def forward_generation_dsa(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_sparse_mla_kvcache_bf16(q, compressed_kv, k_pe, attn_metadata,
                                       output, latent_cache, hidden_states, qr,
                                       position_ids, is_generation = True)

    def forward_context_with_cached_kv(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert latent_cache is not None
        trtllm_attention = cast(TrtllmAttention, self.mha)

        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata)

        # copy full_compressed_kv and full_k_pe from paged kv cache
        full_compressed_kv, full_k_pe = trtllm_attention.load_paged_kv_cache_for_mla(
            attn_metadata, q.dtype)
        assert full_compressed_kv.shape[
            0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        assert full_compressed_kv.shape[1] == self.kv_lora_rank
        assert full_k_pe.shape[
            0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        assert full_k_pe.shape[1] == self.qk_rope_head_dim
        assert full_compressed_kv.is_contiguous()
        assert full_k_pe.is_contiguous()

        # compute full_k_nope and full_v from full_compressed_kv
        full_kv = self.kv_b_proj(full_compressed_kv)
        full_k_nope, full_v = full_kv.split(
            [
                self.num_heads * self.qk_nope_head_dim,
                self.num_heads * self.v_head_dim
            ],
            -1,
        )

        full_k_nope = full_k_nope.view(-1, self.num_heads,
                                       self.qk_nope_head_dim)
        full_k_pe = full_k_pe.view(-1, 1, self.qk_rope_head_dim)
        full_k = compiled_cat(
            (full_k_nope, full_k_pe.expand(-1, self.num_heads, -1)), dim=-1)
        full_k = full_k.view(-1, self.num_heads * self.qk_head_dim)

        # release pytorch activation memory
        full_compressed_kv = None
        full_k_pe = None
        full_kv = None
        full_k_nope = None

        # latent_cache must be None to differentiate from normal context phase,
        # so that we can skip applying RoPE and appending KV cache inside attention op
        attn_output = self.mha.forward(
            q,
            full_k,
            full_v,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=None,
            out_scale=self.out_scale,
            output=output,
        )

        return attn_output

    def forward_context_with_chunked_prefill(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        latent_cache: torch.
        Tensor,  # compressed_kv + k_pe [context_tokens, 1, lora_size + rope_size]
        attn_metadata: TrtllmAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        trtllm_attention = cast(TrtllmAttention, self.mha)
        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata)

        # determine the number of loop
        # currently we assume that the chunk size is the same as the max_num_tokens
        chunked_loop_num = attn_metadata.chunked_loop_num

        # [toal_token_q, num_heads, 2] -> [toal_token_q, num_heads] float2
        self.softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads, 2),
            dtype=torch.float,
            device='cuda',
        )
        self.temp_softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads, 2),
            dtype=torch.float,
            device='cuda',
        )

        attn_output = output
        temp_attn_output = q.new_empty(
            (q.size(0), self.num_heads * self.v_head_dim), dtype=q.dtype)

        # use fake cached_cu_seq_len for chunked loop
        origin_kv_lens_cuda_runtime = attn_metadata.kv_lens_cuda_runtime
        origin_kv_lens_runtime = attn_metadata.kv_lens_runtime
        origin_ctx_total_kv_len = attn_metadata.host_total_kv_lens[0]

        for loop_idx in range(chunked_loop_num):
            # {b, chunked_unit_size, h, kv_lora_rank + qk_rope_head_dim} zero padded
            # fetch `loop_idx` chunk from kv cache
            temp_cu_chunked_seq_len = attn_metadata.cu_chunked_seq_len[loop_idx]
            total_ctx_chunked_tokens = attn_metadata.host_cu_chunked_seq_len[
                loop_idx, attn_metadata.num_contexts]
            chunked_global_offset = attn_metadata.chunked_global_offset[
                loop_idx]
            chunked_max_seq_len = attn_metadata.max_chunk_len_per_loop[loop_idx]
            chunked_compressed_kv, chunked_k_pe = trtllm_attention.load_chunked_kv_cache_for_mla(
                metadata=attn_metadata,
                num_ctx_cached_tokens=total_ctx_chunked_tokens,
                cu_chunked_seq_len=temp_cu_chunked_seq_len,
                chunked_global_offset=chunked_global_offset,
                chunked_max_seq_len=chunked_max_seq_len,
                out_dtype=q.dtype)

            # up proj to uncompressed kv
            # [tokens, 2, h, kv_dim], without rope_dim
            chunked_kv = self.kv_b_proj(chunked_compressed_kv)
            chunked_k_nope, chunked_v = chunked_kv.split(
                [
                    self.num_heads * self.qk_nope_head_dim,
                    self.num_heads * self.v_head_dim
                ],
                -1,
            )

            chunked_k_nope = chunked_k_nope.view(-1, self.num_heads,
                                                 self.qk_nope_head_dim)
            chunked_k_pe = chunked_k_pe.view(-1, 1, self.qk_rope_head_dim)
            chunked_k = compiled_cat(
                (chunked_k_nope, chunked_k_pe.expand(-1, self.num_heads, -1)),
                dim=-1)
            chunked_k = chunked_k.view(-1, self.num_heads * self.qk_head_dim)

            # release pytorch activation memory
            chunked_compressed_kv = None
            chunked_k_pe = None
            chunked_kv = None
            chunked_k_nope = None

            # copy chunked_seq_len to replace kv_lens_runtime
            attn_metadata.kv_lens_runtime = attn_metadata.host_chunked_seq_len[
                loop_idx]
            attn_metadata.kv_lens_cuda_runtime = attn_metadata.chunked_seq_len[
                loop_idx]
            attn_metadata.host_total_kv_lens[0] = total_ctx_chunked_tokens

            # do not apply mask for attention within loop
            # latent_cache must be None to differentiate from normal context phase,
            # so that we can skip applying RoPE and appending KV cache inside attention op
            temp_attn_output = self.mha.forward(
                q,
                chunked_k,
                chunked_v,
                attn_metadata,
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                out_scale=self.out_scale,
                attention_mask=PredefinedAttentionMask.FULL,
                softmax_stats_tensor=self.temp_softmax_stats_tensor,
                chunked_prefill_buffer_batch_size=attn_metadata.
                runtime_features.chunked_prefill_buffer_batch_size,
                output=temp_attn_output,
            )
            # merge attn result
            temp_merge_op = attn_metadata.merge_op_tensor[loop_idx]
            trtllm_attention.merge_attention_for_mla(
                attn_output, temp_attn_output, self.softmax_stats_tensor,
                self.temp_softmax_stats_tensor, temp_merge_op, attn_metadata)

        # deal with the uncached kv
        kv = self.kv_b_proj(compressed_kv)
        _, k_pe = latent_cache.view([
            -1, self.kv_lora_rank + self.qk_rope_head_dim
        ]).split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        # final round of attention

        k_nope, v = kv.split(
            [
                self.num_heads * self.qk_nope_head_dim,
                self.num_heads * self.v_head_dim
            ],
            -1,
        )

        k_nope = k_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        k = compiled_cat((k_nope, k_pe.expand(-1, self.num_heads, -1)), dim=-1)
        k = k.view(-1, self.num_heads * self.qk_head_dim)

        # copy q_lens to replace kv_lens_runtime
        attn_metadata.kv_lens_runtime = attn_metadata.prompt_lens_cpu_runtime
        attn_metadata.kv_lens_cuda_runtime = attn_metadata.prompt_lens_cuda_runtime
        attn_metadata.host_total_kv_lens[
            0] = attn_metadata.prompt_lens_cpu_runtime[:attn_metadata.
                                                       num_contexts].sum().item(
                                                       )

        # latent_cache must be None to differentiate from normal context phase,
        # so that we can skip applying RoPE and appending KV cache inside attention op
        temp_attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=None,
            out_scale=self.out_scale,
            softmax_stats_tensor=self.temp_softmax_stats_tensor,
            chunked_prefill_buffer_batch_size=attn_metadata.runtime_features.
            chunked_prefill_buffer_batch_size,
            output=temp_attn_output,
        )
        temp_merge_op = attn_metadata.merge_op_tensor[chunked_loop_num]
        trtllm_attention.merge_attention_for_mla(attn_output, temp_attn_output,
                                                 self.softmax_stats_tensor,
                                                 self.temp_softmax_stats_tensor,
                                                 temp_merge_op, attn_metadata)
        # copy back kv_lens_runtime and kv_lens_cuda_runtime
        attn_metadata.kv_lens_runtime = origin_kv_lens_runtime
        attn_metadata.kv_lens_cuda_runtime = origin_kv_lens_cuda_runtime
        attn_metadata.host_total_kv_lens[0] = origin_ctx_total_kv_len

        return attn_output

    def forward_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_dsa:
            return self.forward_context_dsa(q, compressed_kv, k_pe,
                                            attn_metadata, output, latent_cache,
                                            hidden_states, qr, position_ids)
        elif isinstance(self.mha, TrtllmAttention):
            assert isinstance(attn_metadata, TrtllmAttentionMetadata)
            trtllm_attention = cast(TrtllmAttention, self.mha)
            if trtllm_attention.is_chunked_prefill_for_mla_context(
                    attn_metadata):
                return self.forward_context_with_chunked_prefill(
                    q, compressed_kv, latent_cache, attn_metadata, output)
            elif trtllm_attention.has_cached_kv_for_mla_context(attn_metadata):
                return self.forward_context_with_cached_kv(
                    q, latent_cache, attn_metadata, output)

        return self.forward_context_default(q, compressed_kv, k_pe,
                                            attn_metadata, output, latent_cache)

    def forward_generation(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_dsa:
            return self.forward_generation_dsa(q, compressed_kv, k_pe,
                                            attn_metadata, output, latent_cache,
                                            hidden_states, qr, position_ids)

        num_tokens = q.shape[0]
        q_nope, q_pe = q.view([-1, self.num_heads, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        fused_q = torch.empty(
            [
                num_tokens, self.num_heads,
                (self.kv_lora_rank + self.qk_rope_head_dim)
            ],
            dtype=q.dtype,
            device=q.device,
        )

        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            torch.ops.trtllm.bmm_out(q_nope_t,
                                     self.k_b_proj_trans.transpose(1, 2),
                                     q_nope_out)
        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., :self.kv_lora_rank].transpose(0, 1)

            fp8_block_scaling_bmm_out(
                q_nope,
                self.k_b_proj_trans,
                self.k_b_proj_trans_scale,
                q_nope_out,
                self.k_b_proj_trans_dequant,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        if self.apply_rotary_emb:
            fused_q[..., self.kv_lora_rank:] = q_pe
        fused_q = fused_q.view([
            num_tokens,
            self.num_heads * (self.kv_lora_rank + self.qk_rope_head_dim)
        ])

        attn_out_latent = self.mqa.forward(
            fused_q,
            None,
            None,
            attn_metadata,
            attention_input_type=AttentionInputType.generation_only,
            out_scale=self.out_scale,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by `invokeMLARopeGeneration`
            hidden_states=hidden_states,
            qr=qr,
            position_ids=position_ids,
        )
        fused_q = None

        assert (attn_out_latent.shape[0] == q.shape[0] and
                attn_out_latent.shape[1] == self.num_heads * self.kv_lora_rank)

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads, self.kv_lora_rank])

        attn_output = output.view([num_tokens, self.num_heads, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            torch.ops.trtllm.bmm_out(attn_out_latent.transpose(0, 1),
                                     self.v_b_proj.transpose(1, 2),
                                     attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output


    def forward_sparse_mla_kvcache_bf16(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        is_generation: bool = False,
    ) -> torch.Tensor:
        """
        Forward sparse MLA (DSA) for BF16 KV cache for both context and generation phases using FlashMLA kernels

        To form the input for FlashMLA kernel and adapt our KV cache manager, we need to:
        1. Cannot fuse RoPE into attn anymore; instead, we need to apply RoPE for q/k separately in this kernel
        2. Load full kv cache from paged kv cache via load_paged_kv_cache_for_mla
        3. Append and update kv cache and apply RoPE to q via mla_rope_append_paged_kv_assign_q
        3. Call FlashMLA sparse attention kernel (fmla.sparse_prefill_fwd for ctxa and generation phase)
        """
        assert isinstance(attn_metadata, TrtllmAttentionMetadata), \
            "DSA requires TrtllmAttentionMetadata for now"
        # Step 1: Append current tokens to paged cache and apply RoPE to q
        # This writes latent_cache to paged KV and modifies q in-place
        trtllm_attention = cast(TrtllmAttention, self.mqa)
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata)

        # Step 2: Load full latent cache from paged memory
        if is_generation:
            # Generation: load per-request with variable lengths
            num_generations = attn_metadata.num_generations
            gen_kv_lens = attn_metadata.kv_lens_runtime[attn_metadata.num_contexts:attn_metadata.num_seqs]
            total_gen_kv_tokens = gen_kv_lens.sum().item()
            max_gen_kv_len = gen_kv_lens.max().item()
            
            # Build cumulative indptr
            gen_kv_indptr = torch.zeros(num_generations + 1, dtype=torch.int64, device='cuda')
            torch.cumsum(gen_kv_lens, dim=0, dtype=torch.int64, out=gen_kv_indptr[1:])
            
            # Load from paged cache
            full_compressed_kv, full_k_pe = torch.ops.trtllm.load_paged_kv_cache_for_mla(
                q.dtype,
                num_generations,
                total_gen_kv_tokens,
                max_gen_kv_len,
                gen_kv_indptr,
                attn_metadata.kv_cache_block_offsets[:, attn_metadata.num_contexts:attn_metadata.num_seqs],
                attn_metadata.host_kv_cache_block_offsets[:, attn_metadata.num_contexts:attn_metadata.num_seqs],
                attn_metadata.kv_cache_manager.kv_cache_pool_pointers,
                attn_metadata.kv_cache_manager.kv_cache_pool_mapping,
                trtllm_attention.kv_scale_orig_quant,
                trtllm_attention.kv_scale_quant_orig,
                trtllm_attention.get_local_layer_idx(attn_metadata),
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                attn_metadata.kv_cache_manager.tokens_per_block,
                attn_metadata.kv_cache_manager.max_seq_len,
                0,  # sink_token_length
                attn_metadata.beam_width,
                trtllm_attention.wrapper.quant_mode,
            )
        else:
            # Context: use existing metadata-based loading
            full_compressed_kv, full_k_pe = trtllm_attention.load_paged_kv_cache_for_mla(
                attn_metadata, q.dtype)

        full_latent_cache = torch.cat([full_compressed_kv, full_k_pe], dim=-1)

        num_tokens = q.shape[0]
        # find topk_indicies
        top_k_indices = self.mqa.indexer.sparse_attn_indexer(qr,...)
        topk_indices_reshaped = topk_indices.view(num_tokens, 1, -1)

        q_nope, q_rope = q.view(-1, self.num_heads, self.qk_head_dim).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            
        q_nope_out = torch.empty(
            [
                num_tokens, self.num_heads,
                (self.kv_lora_rank)
            ],
            dtype=q.dtype,
            device=q.device,
        )

        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = q_nope_out.transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            torch.ops.trtllm.bmm_out(q_nope_t,
                                     self.k_b_proj_trans.transpose(1, 2),
                                     q_nope_out)
        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = q_nope_out.transpose(0, 1)

            fp8_block_scaling_bmm_out(
                q_nope,
                self.k_b_proj_trans,
                self.k_b_proj_trans_scale,
                q_nope_out,
                self.k_b_proj_trans_dequant,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        q_nope_out = q_nope_out.transpose(0, 1)
        q_concat = torch.cat([q_nope_out, q_rope], dim=-1)
        # Shape: [num_tokens, num_heads, kv_lora_rank + qk_rope_head_dim]

         # Step 2: Reshape inputs for sparse kernel
        kv_c_and_k_pe_cache = full_latent_cache.view(-1, 1, full_latent_cache.shape[-1])
        
        # Step 3: Handle head padding (kernel requirement)
        sm_version = get_sm_version()
        # TODO: avoid sm version check in the runtime
        padding = 128 if sm_version >= 100 else 64 # kernel limition to 128 for SM100 and 64 for SM90
        if self.num_heads % padding != 0:
            # Assert that padding is mathematically valid
            assert padding % self.num_heads == 0, (
                f"Cannot pad num_heads={self.num_heads} to {padding}. "
                f"Padding must be a multiple of num_heads."
            )
            
            logger.warning_once(
                f"Padding num_heads from {self.num_heads} to {padding} "
                f"due to FlashMLA sparse attention kernel requirement"
            )
            
            # Create padded tensor with zeros for extra heads
            q_padded = q_concat.new_empty((num_tokens, padding, q_concat.shape[2]))
            q_padded[:, :self.num_heads, :] = q_concat
            q_concat = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        attn_out_latent = flash_mla_sparse_prefill(q_concat, kv_c_and_k_pe_cache, topk_indices,
                                          self.softmax_scale)[0] # output: [seq, num_heads, kv_lora_rank]
        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent[:, :self.num_heads, :] # account for padding
        # TODO: seems we need .contiguous() here when padding enabled before pass to bmm? 
        attn_out_latent = attn_out_latent.view(
            [-1, self.num_heads, self.kv_lora_rank])

        assert (attn_out_latent.shape[0] == q.shape[0] and
                attn_out_latent.shape[1] == self.num_heads)

        attn_output = output.view([num_tokens, self.num_heads, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            torch.ops.trtllm.bmm_out(attn_out_latent.transpose(0, 1),
                                     self.v_b_proj.transpose(1, 2),
                                     attn_output.transpose(0, 1))
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
            )
        else:
            raise NotImplementedError(
                f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:

        attn_output = self.create_output(hidden_states)
        if self.register_to_config:
            torch.ops.trtllm.mla_custom_op_inplace(hidden_states, position_ids,
                                                   self.layer_idx_str,
                                                   attn_output)
        else:
            self.forward_impl(position_ids,
                              hidden_states,
                              attn_metadata,
                              output=attn_output)
        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params)
        return attn_output
