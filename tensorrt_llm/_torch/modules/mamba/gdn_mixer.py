# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/qwen3_next.py
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import weakref
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import nn
from transformers import Qwen3NextConfig

from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
    _can_use_flashinfer_gdn_verify,
    _flashinfer_gdn_verify,
    fused_sigmoid_gating_delta_rule_update,
)
from tensorrt_llm._utils import is_flashinfer_gdn_supported_arch
from tensorrt_llm.mapping import Mapping

from ...attention_backend import AttentionMetadata
from ...distributed import AllReduceParams
from ...model_config import ModelConfig
from ...speculative import SpecMetadata
from ...utils import EventType, get_model_extra_attrs, is_torch_compiling
from ..linear import Linear, TensorParallelMode
from ..multi_stream_utils import maybe_execute_in_parallel
from .causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from .causal_conv1d_triton import causal_conv1d_update as causal_conv1d_update_triton
from .fuse_elementwise_ops import (
    extract_transpose_prefill_slice,
    fused_gdn_post_conv,
    pack_gdn_decode_qkv,
)
from .layernorm_gated import RMSNorm as RMSNormGated
from .layernorm_gated import rms_norm_gated_token_major
from .mamba2_metadata import Mamba2Metadata


# FlashInfer GDN prefill is ON by default; set TLLM_USE_FLASHINFER_GDN_PREFILL=0
# to force the vendored Triton chunk_gated_delta_rule everywhere. FlashInfer only
# ships the GDN prefill kernel for Hopper (SM90) and datacenter Blackwell
# (SM100/SM103); on consumer Blackwell (SM120) and other archs it aborts at
# launch, so we fall back to Triton there. Resolution is deferred to first call
# (and cached) so importing this module never initializes CUDA.
@functools.lru_cache(maxsize=1)
def _resolve_chunk_gated_delta_rule():
    if (
        os.getenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1") == "1"
        and is_flashinfer_gdn_supported_arch()
    ):
        from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule as impl
    else:
        from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as impl
    return impl


@torch.compiler.disable
def chunk_gated_delta_rule(*args, **kwargs):
    return _resolve_chunk_gated_delta_rule()(*args, **kwargs)


def _extract_gdn_extra_attrs(layer_idx: str):
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, AttentionMetadata)

    gdn_layers = extra_attrs.get("gdn_layers", None)
    assert gdn_layers is not None, "GDN layer is not registered"
    gdn_layer_ref = gdn_layers.get(layer_idx, None)
    assert gdn_layer_ref is not None, f"Cannot find GDN layer for layer {layer_idx}"
    gdn_layer = gdn_layer_ref()
    assert isinstance(gdn_layer, Qwen3NextGatedDeltaNet)

    return metadata, gdn_layer, extra_attrs.get("spec_metadata", None)


@torch.library.custom_op("trtllm::gdn_custom_op_inplace", mutates_args=("output",))
def gdn_custom_op_inplace(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    attn_metadata, gdn_layer, spec_metadata = _extract_gdn_extra_attrs(layer_idx)
    num_tokens = attn_metadata.num_tokens
    gdn_layer.forward_core(
        mixed_qkv[:num_tokens],
        a[:num_tokens],
        b[:num_tokens],
        attn_metadata,
        attn_metadata.mamba_metadata,
        spec_metadata=spec_metadata,
        output=output[:, :num_tokens, :, :],
    )


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    stride_a_row,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_d = tl.program_id(0), tl.program_id(1)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    # a may be a row-strided view sliced out of the packed ba projection;
    # g is always allocated packed.
    off_a = i_b * stride_a_row + head_off
    off_g = i_b * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off_a, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off_g, blk_g.to(g.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    batch, num_heads = a.shape
    grid = (batch, triton.cdiv(num_heads, 8))
    g = torch.empty(batch, num_heads, dtype=torch.float32, device=a.device)
    fused_gdn_gating_kernel[grid](
        g, A_log, a, dt_bias, a.stride(0), num_heads, beta, threshold, 8, num_warps=1
    )
    return g


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
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
        self.num_k_heads_per_tp = divide(self.num_k_heads, self.attn_tp_size)
        self.num_v_heads_per_tp = divide(self.num_v_heads, self.attn_tp_size)
        self.key_dim_per_tp = self.head_k_dim * self.num_k_heads_per_tp
        self.value_dim_per_tp = self.head_v_dim * self.num_v_heads_per_tp
        self.conv_dim_per_tp = self.key_dim_per_tp * 2 + self.value_dim_per_tp

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.register_to_config = False
        if model_config is not None:
            if "gdn_layers" not in model_config.extra_attrs:
                model_config.extra_attrs["gdn_layers"] = {}
            suffix = 0
            while self.layer_idx_str in model_config.extra_attrs["gdn_layers"]:
                self.layer_idx_str = str(layer_idx) + f"_{suffix}"
                suffix += 1
            model_config.extra_attrs["gdn_layers"][self.layer_idx_str] = weakref.ref(self)
            self.register_to_config = True

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
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False,
        )

        self.in_proj_qkvz = Linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim * 2,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False,
        )
        self.in_proj_ba = Linear(
            self.hidden_size,
            self.num_v_heads * 2,
            bias=False,
            dtype=config.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(
                (self.num_v_heads // self.attn_tp_size),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        A = torch.empty(divide(self.num_v_heads, self.attn_tp_size), dtype=torch.float32).uniform_(
            0, 16
        )
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
            dtype=config.torch_dtype,
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
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=False,
        )

        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.Attention]}
        self.aux_stream = aux_stream

    def _compute_tokenwise_inputs(self, hidden_states: torch.Tensor):
        def _compute_projected_states_qkvz():
            return self.in_proj_qkvz(hidden_states)

        def _compute_projected_states_ba():
            return self.in_proj_ba(hidden_states)

        projected_states_qkvz, projected_states_ba = maybe_execute_in_parallel(
            _compute_projected_states_qkvz,
            _compute_projected_states_ba,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.Attention],
            self.aux_stream,
            disable_on_compile=True,
        )

        # The weight mapper reorders in_proj rows into the dense per-rank
        # layouts [Q|K|V|Z] and [b|a] (see grouped_to_dense_in_proj_qkvz_perm),
        # so every component is a plain column slice of the projection —
        # no split/reshape kernel. Downstream consumers (causal_conv1d,
        # the GDN decode kernels, the gated norm) read these row-strided
        # views in place.
        num_tokens = projected_states_qkvz.shape[0]
        mixed_qkv = projected_states_qkvz[:, : self.conv_dim_per_tp]
        z = projected_states_qkvz[:, self.conv_dim_per_tp :].view(
            num_tokens, self.num_v_heads_per_tp, self.head_v_dim
        )
        b = projected_states_ba[:, : self.num_v_heads_per_tp]
        a = projected_states_ba[:, self.num_v_heads_per_tp :]

        return mixed_qkv, z, a, b

    def _postprocess_gdn_output(
        self,
        attn_out: torch.Tensor,
        z: torch.Tensor,
        all_reduce_params: Optional[AllReduceParams] = None,
    ):
        # z is a [num_tokens, num_v_heads, head_v_dim] view of the in_proj
        # output whose (heads, head_dim) block is contiguous per token; the
        # gated norm reads it through its token stride instead of packing a
        # copy.
        attn_out = rms_norm_gated_token_major(
            attn_out.reshape(-1, self.head_v_dim), z, self.norm.weight, self.norm.eps
        )
        attn_out = attn_out.view(-1, self.value_dim_per_tp)
        return self.out_proj(attn_out, all_reduce_params=all_reduce_params)

    def forward_decode(
        self,
        conv_states,
        ssm_states,
        query_start_loc_long,
        spec_metadata: Optional[SpecMetadata] = None,
        intermediate_conv_states: Optional[torch.Tensor] = None,
        intermediate_ssm_states: Optional[torch.Tensor] = None,
        is_target_verify: bool = False,
        output: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        a = kwargs["a"]
        b = kwargs["b"]
        cache_indices = kwargs["cache_indices"]
        num_decodes = kwargs["num_decodes"]

        if is_target_verify:
            draft_token_num = spec_metadata.runtime_draft_len + 1
            assert num_decodes > 0
            assert mixed_qkv.shape[0] == num_decodes * draft_token_num
            assert a.shape[0] == num_decodes * draft_token_num
            assert b.shape[0] == num_decodes * draft_token_num
            assert intermediate_conv_states is not None
            assert intermediate_ssm_states is not None

            # Speculative verification path:
            # 1. run conv update with per-step intermediate cache writes
            # 2. run recurrent delta rule with intermediate SSM-state cache writes
            # 3. defer final state selection to kv_cache_manager.update_mamba_states()
            intermediate_state_indices = torch.arange(
                num_decodes, dtype=torch.int32, device=cache_indices.device
            )

            mixed_qkv_reshaped = mixed_qkv.reshape(num_decodes, draft_token_num, -1).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update_triton(
                mixed_qkv_reshaped,
                conv_states,
                self.conv1d.weight,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=cache_indices[:num_decodes],
                intermediate_conv_window=intermediate_conv_states,
                intermediate_state_indices=intermediate_state_indices,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).reshape(
                num_decodes * draft_token_num, -1
            )

            key_size = self.key_dim // self.attn_tp_size
            query = mixed_qkv[..., :key_size]
            key = mixed_qkv[..., key_size : key_size * 2]
            value = mixed_qkv[..., key_size * 2 :]

            query = query.reshape(
                num_decodes, draft_token_num, self.num_k_heads // self.attn_tp_size, self.head_k_dim
            )
            key = key.reshape(
                num_decodes, draft_token_num, self.num_k_heads // self.attn_tp_size, self.head_k_dim
            )
            value = value.reshape(
                num_decodes, draft_token_num, self.num_v_heads // self.attn_tp_size, self.head_v_dim
            )

            a = a.reshape(num_decodes, draft_token_num, -1)
            b = b.reshape(num_decodes, draft_token_num, -1)

            # Prefer the FlashInfer MTP kernel (raw a/b gating in-kernel,
            # initial state gathered from the pool via cache indices, per-step
            # intermediate states written to the batch-scoped [:num_decodes]
            # prefix consumed by update_mamba_states()); fall back to the
            # Triton recurrent kernel when unavailable.
            if _can_use_flashinfer_gdn_verify(
                ssm_states, self.head_k_dim, self.head_v_dim, draft_token_num
            ):
                output_d = None
                if output is not None:
                    output_d = output.view(
                        num_decodes,
                        draft_token_num,
                        self.num_v_heads // self.attn_tp_size,
                        self.head_v_dim,
                    )
                return _flashinfer_gdn_verify(
                    A_log=self.A_log,
                    a=a,
                    dt_bias=self.dt_bias,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=query,
                    k=key,
                    v=value,
                    b=b,
                    initial_state_source=ssm_states,
                    initial_state_indices=cache_indices[:num_decodes],
                    intermediate_states_buffer=intermediate_ssm_states[:num_decodes],
                    scale=self.head_k_dim**-0.5,
                    use_qk_l2norm_in_kernel=True,
                    output=output_d,
                ).view(
                    1,
                    num_decodes * draft_token_num,
                    self.num_v_heads // self.attn_tp_size,
                    self.head_v_dim,
                )

            beta = b.sigmoid()
            g = fused_gdn_gating(
                self.A_log,
                a.view(num_decodes * draft_token_num, -1),
                self.dt_bias,
            ).reshape(num_decodes, draft_token_num, -1)

            # Keep intermediate-state indexing consistent with Mamba2Mixer:
            # cache slots [0..num_decodes-1] are consumed by
            # MambaCacheManager.update_mamba_states(), while initial states are
            # gathered from real slot indices.
            recurrent_state_source = ssm_states[cache_indices[:num_decodes]]
            recurrent_state_indices = torch.arange(
                num_decodes, dtype=torch.int32, device=cache_indices.device
            )

            output_d = None
            if output is not None:
                output_d = output.view(
                    num_decodes,
                    draft_token_num,
                    self.num_v_heads // self.attn_tp_size,
                    self.head_v_dim,
                )

            attn_out = fused_recurrent_gated_delta_rule_update(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state_source=recurrent_state_source,
                initial_state_indices=recurrent_state_indices,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                intermediate_states_buffer=intermediate_ssm_states,
                cache_steps=draft_token_num,
                output=output_d,
            )
            return attn_out.view(
                1,
                num_decodes * draft_token_num,
                self.num_v_heads // self.attn_tp_size,
                self.head_v_dim,
            )

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            self.conv1d.weight,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=cache_indices,
        )

        # Keep q/k/v as views over mixed_qkv so the fused decode kernel can
        # consume their native strides without forcing packed copies.
        query = mixed_qkv[..., : self.key_dim_per_tp]
        key = mixed_qkv[..., self.key_dim_per_tp : self.key_dim_per_tp * 2]
        value = mixed_qkv[..., self.key_dim_per_tp * 2 :]
        seq_len = query.shape[0]
        query = query.view(1, seq_len, self.num_k_heads_per_tp, self.head_k_dim)
        key = key.view(1, seq_len, self.num_k_heads_per_tp, self.head_k_dim)
        value = value.view(1, seq_len, self.num_v_heads_per_tp, self.head_v_dim)

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
            output=output,
        )

        return core_attn_out

    def forward_extend(
        self,
        conv_states,
        ssm_states,
        spec_metadata: Optional[SpecMetadata] = None,
        intermediate_conv_states: Optional[torch.Tensor] = None,
        intermediate_ssm_states: Optional[torch.Tensor] = None,
        is_target_verify: bool = False,
        output: Optional[torch.Tensor] = None,
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
        num_decodes = kwargs["num_decodes"]

        conv_states_to_use = conv_states

        seqlen_split_size = [num_prefill_tokens, num_decode_tokens]
        if num_decode_tokens > 0:
            mixed_qkv_p, mixed_qkv_d = torch.split(mixed_qkv, seqlen_split_size, dim=0)
            query_start_loc_p = query_start_loc[: num_prefill + 1]
            has_initial_states_p = has_initial_states[:num_prefill]

            mixed_qkv_p_t = extract_transpose_prefill_slice(
                mixed_qkv_p,
                mixed_qkv_p.shape[0],
                0,
                mixed_qkv_p.shape[1],
            )
            mixed_qkv_p_t = causal_conv1d_fn(
                mixed_qkv_p_t,
                self.conv1d.weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_states_to_use,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_p,
                query_start_loc=query_start_loc_p,
            )

            if is_target_verify:
                a_d = a[num_prefill_tokens:]
                b_d = b[num_prefill_tokens:]
                draft_token_num = spec_metadata.runtime_draft_len + 1
                assert num_decodes > 0
                assert mixed_qkv_d.shape[0] == num_decodes * draft_token_num
                assert a_d.shape[0] == num_decodes * draft_token_num
                assert b_d.shape[0] == num_decodes * draft_token_num
                assert intermediate_conv_states is not None
                assert intermediate_ssm_states is not None

                intermediate_state_indices = torch.arange(
                    num_decodes, dtype=torch.int32, device=state_indices_d.device
                )
                mixed_qkv_d = mixed_qkv_d.reshape(num_decodes, draft_token_num, -1).transpose(1, 2)
                mixed_qkv_d = causal_conv1d_update_triton(
                    mixed_qkv_d,
                    conv_states_to_use,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_state_indices=state_indices_d,
                    intermediate_conv_window=intermediate_conv_states,
                    intermediate_state_indices=intermediate_state_indices,
                )
                mixed_qkv_d = mixed_qkv_d.transpose(1, 2).reshape(num_decode_tokens, -1)
            else:
                mixed_qkv_d = causal_conv1d_update(
                    mixed_qkv_d,
                    conv_states_to_use,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    activation=self.activation,
                    conv_state_indices=state_indices_d,
                )
            if is_target_verify:
                if num_prefill_tokens > 0:
                    query_p, key_p, value_p, g_p, beta_p = fused_gdn_post_conv(
                        mixed_qkv_p_t,
                        None,
                        a[:num_prefill_tokens],
                        b[:num_prefill_tokens],
                        self.A_log,
                        self.dt_bias,
                        self.num_k_heads_per_tp,
                        self.head_k_dim,
                        self.num_v_heads_per_tp,
                        self.head_v_dim,
                        beta_dtype=b.dtype,
                    )
                query_d, key_d, value_d = pack_gdn_decode_qkv(
                    mixed_qkv_d,
                    self.num_k_heads_per_tp,
                    self.head_k_dim,
                    self.num_v_heads_per_tp,
                    self.head_v_dim,
                )
            else:
                query, key, value, g, beta = fused_gdn_post_conv(
                    mixed_qkv_p_t,
                    mixed_qkv_d,
                    a,
                    b,
                    self.A_log,
                    self.dt_bias,
                    self.num_k_heads_per_tp,
                    self.head_k_dim,
                    self.num_v_heads_per_tp,
                    self.head_v_dim,
                )
        else:
            mixed_qkv_t = extract_transpose_prefill_slice(
                mixed_qkv,
                mixed_qkv.shape[0],
                0,
                mixed_qkv.shape[1],
            )
            mixed_qkv_t = causal_conv1d_fn(
                mixed_qkv_t,
                self.conv1d.weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_states_to_use,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )
            query, key, value, g, beta = fused_gdn_post_conv(
                mixed_qkv_t,
                None,
                a,
                b,
                self.A_log,
                self.dt_bias,
                self.num_k_heads_per_tp,
                self.head_k_dim,
                self.num_v_heads_per_tp,
                self.head_v_dim,
            )

        if is_target_verify and num_decode_tokens > 0:
            attn_out_prefill = None
            if num_prefill_tokens > 0:
                recurrent_state_p = ssm_states[state_indices_p]
                output_p = output[:, :num_prefill_tokens, :, :] if output is not None else None
                attn_out_prefill, last_recurrent_state = chunk_gated_delta_rule(
                    q=query_p,
                    k=key_p,
                    v=value_p,
                    g=g_p,
                    beta=beta_p,
                    initial_state=recurrent_state_p,
                    output_final_state=True,
                    cu_seqlens=query_start_loc_long[: num_prefill + 1],
                    head_first=False,
                    use_qk_l2norm_in_kernel=False,
                    output=output_p,
                )
                last_recurrent_state = last_recurrent_state.to(ssm_states.dtype, copy=False)
                ssm_states[state_indices_p] = last_recurrent_state

            draft_token_num = spec_metadata.runtime_draft_len + 1
            query_d = query_d.reshape(
                num_decodes, draft_token_num, self.num_k_heads // self.attn_tp_size, self.head_k_dim
            )
            key_d = key_d.reshape(
                num_decodes, draft_token_num, self.num_k_heads // self.attn_tp_size, self.head_k_dim
            )
            value_d = value_d.reshape(
                num_decodes, draft_token_num, self.num_v_heads // self.attn_tp_size, self.head_v_dim
            )

            a_d = a_d.reshape(num_decodes, draft_token_num, -1)
            b_d = b_d.reshape(num_decodes, draft_token_num, -1)
            out_v_heads = self.num_v_heads // self.attn_tp_size

            output_d = None
            if output is not None:
                output_d = output[:, num_prefill_tokens:, :, :].view(
                    num_decodes,
                    draft_token_num,
                    out_v_heads,
                    self.head_v_dim,
                )

            if _can_use_flashinfer_gdn_verify(
                ssm_states, self.head_k_dim, self.head_v_dim, draft_token_num
            ):
                # FI gathers the initial state from the pool via state_indices_d
                # (no host gather) and writes batch-scoped intermediate states;
                # the [:num_decodes] prefix matches update_mamba_states()'s rows.
                attn_out_decode = _flashinfer_gdn_verify(
                    A_log=self.A_log,
                    a=a_d,
                    dt_bias=self.dt_bias,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=query_d,
                    k=key_d,
                    v=value_d,
                    b=b_d,
                    initial_state_source=ssm_states,
                    initial_state_indices=state_indices_d,
                    intermediate_states_buffer=intermediate_ssm_states[:num_decodes],
                    scale=self.head_k_dim**-0.5,
                    use_qk_l2norm_in_kernel=True,
                    output=output_d,
                ).reshape(1, num_decode_tokens, out_v_heads, self.head_v_dim)
            else:
                beta_d = b_d.sigmoid()
                g_d = fused_gdn_gating(
                    self.A_log,
                    a_d.view(num_decodes * draft_token_num, -1),
                    self.dt_bias,
                ).reshape(num_decodes, draft_token_num, -1)

                recurrent_state_source = ssm_states[state_indices_d]
                recurrent_state_indices = torch.arange(
                    num_decodes, dtype=torch.int32, device=state_indices_d.device
                )

                attn_out_decode = fused_recurrent_gated_delta_rule_update(
                    q=query_d,
                    k=key_d,
                    v=value_d,
                    g=g_d,
                    beta=beta_d,
                    initial_state_source=recurrent_state_source,
                    initial_state_indices=recurrent_state_indices,
                    use_qk_l2norm_in_kernel=True,
                    disable_state_update=True,
                    intermediate_states_buffer=intermediate_ssm_states,
                    cache_steps=draft_token_num,
                    output=output_d,
                ).view(1, num_decode_tokens, out_v_heads, self.head_v_dim)

            if output is not None:
                return output
            if attn_out_prefill is None:
                return attn_out_decode
            return torch.cat((attn_out_prefill, attn_out_decode), dim=1)

        core_attn_out, _ = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            # This path writes recurrent state directly back into the shared
            # pool; callers **must** ensure cache_indices do not alias live slots.
            inplace_indexed_state_update=True,
            output_final_state=False,
            cu_seqlens=query_start_loc_long,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
            output=output,
        )

        return core_attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
        spec_metadata: Optional[SpecMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
    ):
        mixed_qkv, z, a, b = self._compute_tokenwise_inputs(hidden_states)

        if self.register_to_config and is_torch_compiling():
            attn_out = mixed_qkv.new_empty(
                (1, mixed_qkv.shape[0], self.num_v_heads_per_tp, self.head_v_dim)
            )
            gdn_custom_op_inplace(mixed_qkv, a, b, self.layer_idx_str, attn_out)
        else:
            attn_out = self.forward_core(
                mixed_qkv,
                a,
                b,
                attn_metadata,
                mamba_metadata,
                spec_metadata=spec_metadata,
            )

        return self._postprocess_gdn_output(attn_out, z, all_reduce_params)

    def forward_core(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
        spec_metadata: Optional[SpecMetadata] = None,
        output: Optional[torch.Tensor] = None,
    ):
        ### sglang linear attn
        # has_initial_states = None
        # if forward_batch.extend_prefix_lens is not None:
        #     has_initial_states = forward_batch.extend_prefix_lens > 0

        ### mamba2_mixer layer
        # calculate split size
        num_prefills = attn_metadata.num_contexts
        num_decodes = attn_metadata.seq_lens.shape[0] - num_prefills
        num_prefill_tokens = attn_metadata.num_ctx_tokens
        num_decode_tokens = attn_metadata.num_tokens - num_prefill_tokens
        batch_split_size = [num_prefills, num_decodes]
        has_initial_states = mamba_metadata.has_initial_states
        state_indices = mamba_metadata.state_indices[: num_prefills + num_decodes]
        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        conv_states = layer_cache.conv
        ssm_states = layer_cache.temporal

        state_indices_p, state_indices_d = torch.split(state_indices, batch_split_size)
        if num_prefills > 0:
            # PyExecutor guarantees prefill requests are placed before decode requests
            has_initial_states_p = has_initial_states[:num_prefills]
            ssm_states[state_indices_p[~has_initial_states_p]] = torch.zeros(
                (), dtype=ssm_states.dtype, device=ssm_states.device
            )
            conv_states[state_indices_p[~has_initial_states_p]] = torch.zeros(
                (), dtype=conv_states.dtype, device=conv_states.device
            )

        is_target_verify = (
            num_decodes > 0
            and spec_metadata is not None
            and attn_metadata.kv_cache_manager.is_speculative()
            and layer_cache is not None
        )
        intermediate_conv_states = (
            layer_cache.intermediate_conv_window if is_target_verify else None
        )
        intermediate_ssm_states = layer_cache.intermediate_ssm if is_target_verify else None

        kwargs = {
            "mixed_qkv": mixed_qkv,
            "a": a,
            "b": b,
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
            "num_decodes": num_decodes,
        }
        if num_prefills > 0:
            attn_out = self.forward_extend(
                conv_states,
                ssm_states,
                spec_metadata=spec_metadata,
                intermediate_conv_states=intermediate_conv_states,
                intermediate_ssm_states=intermediate_ssm_states,
                is_target_verify=is_target_verify,
                output=output,
                **kwargs,
            )
        else:
            attn_out = self.forward_decode(
                conv_states,
                ssm_states,
                spec_metadata=spec_metadata,
                intermediate_conv_states=intermediate_conv_states,
                intermediate_ssm_states=intermediate_ssm_states,
                is_target_verify=is_target_verify,
                output=output,
                **kwargs,
            )

        return attn_out
