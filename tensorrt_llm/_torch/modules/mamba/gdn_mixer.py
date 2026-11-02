# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/qwen3_next.py
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import weakref
from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl
from torch import nn
from transformers import Qwen3NextConfig

from tensorrt_llm._torch.modules.fla.cached_replay import (
    fused_recurrent_gated_delta_rule_cached_replay_update,
)
from tensorrt_llm._torch.modules.fla.flashinfer_cached_replay import flashinfer_cached_replay_update
from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update
from tensorrt_llm._torch.modules.fla.fused_sigmoid_gating_recurrent import (
    _can_use_flashinfer_gdn_verify,
    _flashinfer_gdn_verify,
    fused_sigmoid_gating_delta_rule_update,
)
from tensorrt_llm._torch.modules.fla.utils import tensor_cache
from tensorrt_llm._utils import is_flashinfer_gdn_prefill_supported_arch, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...attention.backends import AttentionMetadata
from ...autotuner import AutoTuner
from ...distributed import AllReduceParams
from ...model_config import ModelConfig
from ...pyexecutor.breakable_cuda_graph import eager_on_graph, is_in_breakable_cuda_graph
from ...speculative import SpecMetadata
from ...utils import EventType, get_model_extra_attrs, is_gdn_replay_enabled, is_torch_compiling
from ..linear import FP8QDQLinearMethod, Linear, TensorParallelMode
from ..low_m_gemm import low_m_gemm_dispatcher
from ..low_m_mma_gemm import conv_supported, low_m_mma_gemm, warp_mma_supported
from ..multi_stream_utils import maybe_execute_in_parallel_deferred_join
from .causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from .causal_conv1d_triton import causal_conv1d_update as causal_conv1d_update_triton
from .fuse_elementwise_ops import (
    extract_transpose_prefill_slice,
    fused_gdn_post_conv,
    pack_gdn_decode_qkv,
)
from .layernorm_gated import RMSNorm as RMSNormGated
from .layernorm_gated import rms_norm_gated_token_major
from .mamba2_metadata import InforwardSnapshotSplit, Mamba2Metadata
from .mamba2_mixer import _cached_arange
from .recurrent_state_cache import reset_recurrent_state_rows

# Shapes trtllm::gdn_verify covers besides the 128-wide heads (see gdnVerifyKernels.h).
_FUSED_VERIFY_MIN_TOKENS = 2
_FUSED_VERIFY_MAX_TOKENS = 8
_FUSED_VERIFY_HISTORY = 16


class FusedVerifyPlan(NamedTuple):
    """Per-call state of a fused target-verify step (Qwen3NextGatedDeltaNet._fused_verify_plan).

    With fused_verify, a decode-only cached-replay MTP verify step runs the low-M in_proj with the conv1d update in
    its epilogue, or the conv1d update kernel, then one fused replay + gated RMSNorm launch (trtllm::gdn_verify).
    """

    kv_cache_manager: object
    num_decodes: int
    draft_token_num: int
    replay_metadata: object
    layer_cache: object
    state_indices: torch.Tensor
    conv_weight: torch.Tensor


# FlashInfer GDN prefill is ON by default; set TLLM_USE_FLASHINFER_GDN_PREFILL=0
# to force the vendored Triton chunk_gated_delta_rule everywhere. FlashInfer
# ships the GDN prefill kernel for Hopper (SM90), datacenter Blackwell
# (SM100/SM103) and consumer Blackwell (SM120/SM121); on other archs it aborts
# at launch, so we fall back to Triton there. Resolution is deferred to first call
# (and cached) so importing this module never initializes CUDA.
def _use_flashinfer_gdn_prefill() -> bool:
    """Check the prefill backend setting and supported GPU architecture."""
    return (
        os.getenv("TLLM_USE_FLASHINFER_GDN_PREFILL", "1") == "1"
        and is_flashinfer_gdn_prefill_supported_arch()
    )


@functools.lru_cache(maxsize=1)
def _resolve_chunk_gated_delta_rule():
    """Resolve and cache the selected prefill implementation."""
    if _use_flashinfer_gdn_prefill():
        from tensorrt_llm._torch.modules.fla.flashinfer_chunk import chunk_gated_delta_rule as impl
    else:
        from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule as impl
    return impl


@torch.compiler.disable
def chunk_gated_delta_rule(
    *args, state_workspace: Optional[tuple[torch.Tensor, torch.Tensor]] = None, **kwargs
):
    """Dispatch prefill, omitting unused FlashInfer-only workspace arguments."""
    if state_workspace is not None:
        kwargs["state_workspace"] = state_workspace
    return _resolve_chunk_gated_delta_rule()(*args, **kwargs)


@tensor_cache
def _prefill_query_start_loc(query_start_loc: torch.Tensor, num_prefill: int) -> torch.Tensor:
    # FLA caches chunk metadata by tensor identity. Keep this view stable across
    # warmup and CUDA graph capture so preparing the metadata needs no host copy.
    return query_start_loc[: num_prefill + 1]


def _inforward_split_prefill_core(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    split: InforwardSnapshotSplit,
    stage_ssm: torch.Tensor,
    output: Optional[torch.Tensor],
) -> torch.Tensor:
    """Chunked GDN prefill of a context-only batch with in-forward Mamba snapshots.

    Each planned sequence is split into [.., P) and its tail. One chunked prefill over all sequences
    (initial states gathered, final states returned) gives every sequence's final state, written to the
    live slots: the state at P for the planned requests. The tails are then recomputed from the state at
    P, as a separate context iteration of the tail would; their outputs replace the first pass's and their
    final states (the state at the prompt end) go to the staging rows, from which the cache manager
    restores the live slots after committing the snapshot at P.

    Args:
        query, key, value, g, beta: Post-convolution GDN inputs, ``[1, num_tokens, heads, dim]``.
        ssm_states: The layer's SSM state pool, ``[slots, H, V, K]``.
        split: This step's split (``Mamba2Metadata.inforward_split``).
        stage_ssm: The layer's SSM staging rows, ``[rows, H, V, K]``.
        output: Optional preallocated output.
    """
    initial = ssm_states.index_select(0, split.gather)
    core_out, final = chunk_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=split.cu,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
        output=output,
    )
    final = final.to(ssm_states.dtype)
    ssm_states.index_copy_(0, split.live, final.index_select(0, split.main_rows))
    tail = split.tail_tok
    tail_out, tail_final = chunk_gated_delta_rule(
        q=query.index_select(1, tail),
        k=key.index_select(1, tail),
        v=value.index_select(1, tail),
        g=g.index_select(1, tail),
        beta=beta.index_select(1, tail),
        initial_state=final.index_select(0, split.split_rows),
        output_final_state=True,
        cu_seqlens=split.cu_tail,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )
    stage_ssm.index_copy_(0, split.stage_rows, tail_final.to(stage_ssm.dtype))
    core_out[0].index_copy_(0, tail, tail_out[0].to(core_out.dtype))
    return core_out


def _inforward_split_conv_states(
    conv_states: torch.Tensor,
    window: torch.Tensor,
    split: InforwardSnapshotSplit,
    stage_conv: torch.Tensor,
) -> None:
    """Swap the convolution states of the split requests after the prefill convolution.

    The convolution ran over whole chunks, so the live slots hold the window that ends at the prompt end:
    stage it and put the window that ends at P (``window``, the raw ``[k * (d_conv - 1), conv_dim]``
    inputs before P, oldest first) into the live slots.
    """
    stage_conv.index_copy_(0, split.stage_rows, conv_states.index_select(0, split.tail_slots))
    conv_states.index_copy_(
        0,
        split.tail_slots,
        window.view(split.k, conv_states.shape[-1], -1)
        .transpose(1, 2)
        .contiguous()
        .to(conv_states.dtype),
    )


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


def _gdn_prefill_graph_key(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    layer_idx: str,
    output: torch.Tensor,
) -> Optional[tuple]:
    attn_metadata, gdn_layer, _ = _extract_gdn_extra_attrs(layer_idx)
    return gdn_layer.prefill_graph_key(attn_metadata)


def _gdn_prefill_graph_body(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """The core of a context-only batch over all its padded rows, whose launches
    depend only on its prefill graph key."""
    attn_metadata, gdn_layer, spec_metadata = _extract_gdn_extra_attrs(layer_idx)
    mamba_metadata = attn_metadata.mamba_metadata
    # FlashInfer's chunked prefill lets a NaN / Inf row past the last sequence
    # that shares a chunk with real tokens reach their outputs and final state.
    padded = (
        _cached_arange(mixed_qkv.shape[0], mixed_qkv.device)
        >= mamba_metadata.query_start_loc[attn_metadata.num_contexts]
    ).unsqueeze(-1)
    for rows in (mixed_qkv, a, b):
        rows.masked_fill_(padded, 0)
    gdn_layer.forward_core(
        mixed_qkv, a, b, attn_metadata, mamba_metadata, spec_metadata=spec_metadata, output=output
    )


# Under a breakable prefill CUDA graph the GDN core is an eager break; a
# context-only batch's core replays as a CUDA graph per prefill graph key.
maybe_bcg_gdn_custom_op_inplace = eager_on_graph(
    gdn_custom_op_inplace,
    graph_key=_gdn_prefill_graph_key,
    graph_body=_gdn_prefill_graph_body,
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


# The verify path writes each decode request's per-draft-token states into the
# batch-scoped ``intermediate_conv_window`` / ``intermediate_ssm`` buffers at rows
# ``[0, num_decodes)``; ``MambaCacheManager.update_mamba_states()`` reads them
# back from the same rows. The row-index vector is therefore ``arange(num_decodes)``
# for every GDN layer of a step, so build it once per process instead of once
# per layer: prefer the cache-manager-owned arange (allocated together with the
# intermediate buffers), else a cached arange sized to the pool. Slicing a
# persistent buffer is a view, i.e. no allocation and no kernel launch inside the
# CUDA graph. `_cached_arange` is shared with Mamba2Mixer so that both mixers
# hit the same `functools.cache` entry instead of each holding its own copy.


def _verify_intermediate_state_indices(
    kv_cache_manager, num_decodes: int, device: torch.device
) -> torch.Tensor:
    indices = getattr(kv_cache_manager, "intermediate_state_indices", None)
    if indices is None or indices.shape[0] < num_decodes:
        indices = _cached_arange(
            max(kv_cache_manager.get_max_resource_count(), num_decodes), device
        )
    return indices[:num_decodes]


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
        *,
        fused_verify: bool = False,
    ):
        """Gated DeltaNet mixer.

        Args:
            fused_verify: Verify MTP drafts with the fused target-verify kernels where the layer's shapes allow it.
        """
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.pretrained_config = config
        # FlashInfer's chunked prefill reads the sequence boundaries on the
        # device only; the Triton one plans its chunks on the host, so its core
        # cannot replay as a CUDA graph (prefill_graph_key).
        self.prefill_graphable = _use_flashinfer_gdn_prefill()
        replay_enabled = is_gdn_replay_enabled()
        if replay_enabled:
            logger.info_once(
                "GDN MTP replay is requested; set TRTLLM_USE_GDN_REPLAY=0 to disable it",
                key="gdn_mtp_replay_cached",
            )
        else:
            logger.info_once(
                "GDN MTP replay is disabled; set TRTLLM_USE_GDN_REPLAY=1 to enable it",
                key="gdn_mtp_replay_disabled",
            )

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

        self.fused_verify = (
            fused_verify
            and self.head_k_dim == 128
            and self.head_v_dim == 128
            and self.conv_kernel_size == 4
            and self.activation in ("silu", "swish")
            and self.num_v_heads_per_tp % self.num_k_heads_per_tp == 0
            and self.norm.weight.dtype == torch.bfloat16
        )
        # The verify in_proj always runs the warp-MMA GEMM with the conv1d update in its epilogue: its low-M race
        # takes only the tactics that epilogue supports (no split-K), instead of the separate conv1d kernel.
        self.fuse_conv1d_into_in_proj = self.fused_verify and warp_mma_supported()

    def cache_derived_state(self) -> None:
        """Attach downstream static quantization state after loading weights."""
        self.norm.fp8_scale = None
        if isinstance(self.out_proj.quant_method, FP8QDQLinearMethod):
            scale = self.out_proj.quant_method.get_static_input_scale(self.out_proj)
            # detach() strips the Parameter wrapper so nn.Module.__setattr__
            # does not register the derived scale into norm's state_dict; the
            # detached view still shares storage with out_proj.input_scale.
            self.norm.fp8_scale = scale.detach() if scale is not None else None

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def _lowm_in_proj(
        self,
        hidden_states: torch.Tensor,
        h_fp8: Optional[torch.Tensor],
        conv: Optional[dict] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """(in_proj_qkvz, in_proj_ba) outputs from one low-M launch, or None when it does not apply.

        Where the low-M race picked the warp-MMA GEMM for ``in_proj_qkvz``, its FP8 rows and the BF16 ``in_proj_ba``
        rows run as two segments of one ``low_m_mma_gemm`` launch with that tactic. The FP8 segment takes ``h_fp8``
        when given, else quantizes ``hidden_states`` in registers with ``in_proj_qkvz.input_scale``. With ``conv``,
        the epilogue also runs the target-verify conv1d update of the q|k|v channels.
        """
        qkvz, ba = self.in_proj_qkvz, self.in_proj_ba
        dispatcher = low_m_gemm_dispatcher(qkvz)
        if (
            dispatcher is None
            or not isinstance(qkvz.quant_method, FP8QDQLinearMethod)
            or qkvz.bias is not None
            or ba.bias is not None
            or ba.weight.dtype != torch.bfloat16
            # The BF16 segment runs in whole 8-row CTA tiles (2 * HV per rank).
            or ba.weight.shape[0] % 8 != 0
            or hidden_states.dim() != 2
            or hidden_states.dtype != torch.bfloat16
        ):
            return None
        input_scale = qkvz.quant_method.get_static_input_scale(qkvz)
        if input_scale is None:
            return None
        act = "fp8" if h_fp8 is not None else "quant"
        m = hidden_states.shape[0]
        if self.fuse_conv1d_into_in_proj and AutoTuner.get().is_tuning_mode:
            # Tune the race the fused launch below reads (the plain projection tunes the unrestricted one).
            dispatcher.apply_fp8(
                hidden_states if h_fp8 is None else h_fp8,
                qkvz.weight,
                qkvz.weight_scale,
                input_scale,
                hidden_states.dtype,
                single_cta_k=True,
            )
        tactic = dispatcher.warp_mma_tactic(
            m,
            qkvz.weight,
            qkvz.weight_scale,
            input_scale,
            quantize=h_fp8 is None,
            single_cta_k=self.fuse_conv1d_into_in_proj,
        )
        if tactic is None:
            return None
        if conv is not None and not conv_supported(
            m, tactic, conv["weight"].shape[0], conv["tokens"]
        ):
            return None
        return low_m_mma_gemm(
            x16=hidden_states,
            x8=h_fp8,
            w_fp8=qkvz.weight,
            weight_scale=qkvz.weight_scale,
            input_scale=input_scale,
            fp8_act=act,
            w_bf16=ba.weight,
            split_out=qkvz.weight.shape[0],
            out_dtype=hidden_states.dtype,
            tactic=tactic,
            tactic_bf16=(tactic[0], 1, tactic[2] if len(tactic) > 2 else 1),
            conv=conv,
        )

    def _compute_tokenwise_inputs(
        self,
        hidden_states: torch.Tensor,
        h_fp8: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.cuda.Event]]:
        """Project to (mixed_qkv, z, a, b, ba_ready).

        h_fp8: the same activations already quantized to E4M3 with in_proj_qkvz's static input scale; in_proj_ba
        keeps BF16.
        ba_ready: when in_proj_ba ran on the aux stream, the event the caller waits on before the first read of
        a / b (so the conv1d update does not wait for it); None otherwise.
        """
        fused = self._lowm_in_proj(hidden_states, h_fp8) if not is_torch_compiling() else None
        ba_ready = None
        if fused is not None:
            projected_states_qkvz, projected_states_ba = fused
        else:
            qkvz_input = hidden_states
            if h_fp8 is not None and isinstance(self.in_proj_qkvz.quant_method, FP8QDQLinearMethod):
                qkvz_input = h_fp8

            def _compute_projected_states_qkvz():
                return self.in_proj_qkvz(qkvz_input)

            def _compute_projected_states_ba():
                return self.in_proj_ba(hidden_states)

            projected_states_qkvz, projected_states_ba, ba_ready = (
                maybe_execute_in_parallel_deferred_join(
                    _compute_projected_states_qkvz,
                    _compute_projected_states_ba,
                    self.event_dict[EventType.Main],
                    self.event_dict[EventType.Attention],
                    self.aux_stream,
                    disable_on_compile=True,
                )
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

        return mixed_qkv, z, a, b, ba_ready

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
            attn_out.reshape(-1, self.head_v_dim),
            z,
            self.norm.weight,
            self.norm.eps,
            fp8_scale=self.norm.fp8_scale,
        )
        attn_out = attn_out.view(-1, self.value_dim_per_tp)
        return self.out_proj(attn_out, all_reduce_params=all_reduce_params)

    def _fused_verify_plan(
        self,
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata],
        num_rows: int,
    ) -> Optional[FusedVerifyPlan]:
        """The per-call state of a fused target-verify step, or None when the step does not qualify.

        Same conditions as the replay branch of forward_decode (pure decode, speculative cache manager with the
        cached replay, in-layer checkpoint commit) plus the shapes of the fused kernels.
        """
        if (
            not self.fused_verify
            or spec_metadata is None
            or is_torch_compiling()
            or attn_metadata.num_contexts != 0
        ):
            return None
        kv_cache_manager = attn_metadata.kv_cache_manager
        num_decodes = attn_metadata.seq_lens.shape[0]
        if (
            num_decodes == 0
            # The fused kernels read the Triton replay history, not the FlashInfer ring.
            or getattr(kv_cache_manager, "use_gdn_flashinfer_ring_replay", False)
            or not kv_cache_manager.is_speculative()
            or kv_cache_manager.gdn_replay_partitioned(num_decodes)
        ):
            return None
        draft_token_num = spec_metadata.runtime_draft_len + 1
        replay_metadata = kv_cache_manager.get_replay_state_update_metadata()
        if (
            replay_metadata is None
            or draft_token_num != replay_metadata.replay_step_width
            or num_rows != num_decodes * draft_token_num
        ):
            return None
        layer_cache = kv_cache_manager.mamba_layer_cache(self.layer_idx)
        if layer_cache is None or layer_cache.intermediate_conv_window is None:
            return None
        state_indices = attn_metadata.mamba_metadata.state_indices[:num_decodes]
        if not (
            _FUSED_VERIFY_MIN_TOKENS <= draft_token_num <= _FUSED_VERIFY_MAX_TOKENS
            and replay_metadata.replay_history_size == _FUSED_VERIFY_HISTORY
            and layer_cache.temporal.dtype == torch.bfloat16
            and state_indices.dtype == torch.int32
        ):
            return None
        conv_weight = self.conv1d.weight
        if conv_weight.dim() == 3:
            conv_weight = conv_weight.view(conv_weight.shape[0], conv_weight.shape[2])
        return FusedVerifyPlan(
            kv_cache_manager=kv_cache_manager,
            num_decodes=num_decodes,
            draft_token_num=draft_token_num,
            replay_metadata=replay_metadata,
            layer_cache=layer_cache,
            state_indices=state_indices,
            conv_weight=conv_weight,
        )

    def _fused_verify_replay_norm(
        self,
        plan: FusedVerifyPlan,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Replay + gated RMSNorm (+ FP8) of convolved q|k|v rows; returns the out_proj input."""
        layer_cache = plan.layer_cache
        return torch.ops.trtllm.gdn_verify(
            mixed_qkv,
            z,
            a,
            b,
            self.A_log,
            self.dt_bias,
            self.norm.weight,
            self.norm.fp8_scale,
            self.norm.eps,
            layer_cache.temporal,
            plan.state_indices,
            layer_cache.old_x,
            layer_cache.old_B,
            layer_cache.old_dt,
            layer_cache.cache_buf_idx,
            layer_cache.prev_num_accepted_tokens,
        )

    def _fused_verify_inproj(
        self,
        hidden_states: torch.Tensor,
        h_fp8: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata],
    ) -> Optional[torch.Tensor]:
        """The low-M in_proj with the conv1d update in its epilogue (convolved q|k|v, intermediate windows and conv
        state written by the GEMM), then the fused replay + gated RMSNorm: the out_proj input of a decode-only MTP
        target-verify step, or None when the step or the in_proj does not qualify."""
        if low_m_gemm_dispatcher(self.in_proj_qkvz) is None:
            return None
        plan = self._fused_verify_plan(attn_metadata, spec_metadata, hidden_states.shape[0])
        if plan is None:
            return None
        layer_cache = plan.layer_cache
        res = self._lowm_in_proj(
            hidden_states,
            h_fp8,
            conv=dict(
                weight=plan.conv_weight,
                state=layer_cache.conv,
                slots=plan.state_indices,
                windows=layer_cache.intermediate_conv_window,
                tokens=plan.draft_token_num,
            ),
        )
        if res is None:
            return None
        out_qkvz, out_ba = res
        num_tokens = out_qkvz.shape[0]
        mixed_qkv = out_qkvz[:, : self.conv_dim_per_tp]
        z = out_qkvz[:, self.conv_dim_per_tp :].view(
            num_tokens, self.num_v_heads_per_tp, self.head_v_dim
        )
        b = out_ba[:, : self.num_v_heads_per_tp]
        a = out_ba[:, self.num_v_heads_per_tp :]
        return self._fused_verify_replay_norm(plan, mixed_qkv, z, a, b)

    def _fused_verify(
        self,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ba_ready: Optional[torch.cuda.Event],
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata],
    ) -> Optional[torch.Tensor]:
        """The conv1d update kernel, then the fused replay + gated RMSNorm: the out_proj input of a decode-only MTP
        target-verify step whose in_proj did not take the low-M kernel, or None (separate kernels)."""
        plan = self._fused_verify_plan(attn_metadata, spec_metadata, mixed_qkv.shape[0])
        if plan is None:
            return None
        num_decodes = plan.num_decodes
        draft_token_num = plan.draft_token_num
        layer_cache = plan.layer_cache
        # As forward_decode's replay branch: PDL chain conv1d -> fused replay.
        conv_out = causal_conv1d_update_triton(
            mixed_qkv.reshape(num_decodes, draft_token_num, -1).transpose(1, 2),
            layer_cache.conv,
            self.conv1d.weight,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=plan.state_indices,
            intermediate_conv_window=layer_cache.intermediate_conv_window,
            intermediate_state_indices=_verify_intermediate_state_indices(
                plan.kv_cache_manager, num_decodes, plan.state_indices.device
            ),
            launch_dependent_kernels=True,
            launch_with_pdl_wait=True,
            hoist_x=True,
        )
        mixed_qkv = conv_out.transpose(1, 2).reshape(num_decodes * draft_token_num, -1)
        if ba_ready is not None:
            ba_ready.wait()
        return self._fused_verify_replay_norm(plan, mixed_qkv, z, a, b)

    def _replay_verify_recurrent(
        self,
        query,
        key,
        value,
        a,
        b,
        ssm_states,
        state_indices_d,
        num_decodes,
        draft_token_num,
        replay_metadata,
        layer_cache,
        replay_work_items,
        replay_n_writes,
        output_d=None,
        packed_qkv=None,
        use_all_layer_commit=False,
    ):
        """Run MTP target verification via the replay kernel.

        This avoids intermediate-state writes and accepted-state copies; commits
        are deferred via the compact history cache. Cached replay reuses the
        Mamba2 fields as old_x<->U, old_B<->normalized k, and
        old_dt<->cumulative G.
        With FlashInfer ring replay, the history lives in per-layer rings and
        each slot's ring start (cache_base) instead.
        """
        assert replay_metadata is not None, (
            "GDN replay enabled but replay metadata was not allocated."
        )
        assert draft_token_num == replay_metadata.replay_step_width, (
            "GDN replay does not support dynamic draft length yet: "
            f"{draft_token_num} != {replay_metadata.replay_step_width}"
        )
        if draft_token_num > 8 or replay_metadata.replay_history_size > 16:
            raise RuntimeError(
                "GDN cached replay requires draft_token_num <= 8 and replay_history_size <= 16."
            )
        state_indices = state_indices_d[:num_decodes]
        if layer_cache.cache_base is not None:
            # FlashInfer reads the history length and ring start per batch row.
            slots = state_indices.long()
            return flashinfer_cached_replay_update(
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                ssm_states=ssm_states,
                state_indices=state_indices,
                k_ring=layer_cache.k_ring,
                u_ring=layer_cache.u_ring,
                g_ring=layer_cache.g_ring,
                hist_len=layer_cache.prev_num_accepted_tokens[slots],
                cache_base=layer_cache.cache_base[slots],
                history_size=replay_metadata.replay_history_size,
                output=output_d,
            )
        return fused_recurrent_gated_delta_rule_cached_replay_update(
            q=query,
            k=key,
            v=value,
            g=a,
            beta=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            launch_with_pdl=True,
            ssm_states=ssm_states,
            state_indices=state_indices,
            old_u=layer_cache.old_x,
            old_k=layer_cache.old_B,
            old_G=layer_cache.old_dt,
            old_beta=layer_cache.old_dA_cumsum,
            cache_buf_idx=layer_cache.cache_buf_idx,
            prev_num_accepted_tokens=layer_cache.prev_num_accepted_tokens,
            history_size=replay_metadata.replay_history_size,
            replay_work_items=replay_work_items,
            n_writes=replay_n_writes,
            use_qk_l2norm_in_kernel=True,
            packed_qkv=packed_qkv,
            use_all_layer_commit=use_all_layer_commit,
            output=output_d,
        )

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
            assert kwargs.get("use_replay", False) or intermediate_ssm_states is not None

            # Speculative verification path:
            # 1. run conv update with per-step intermediate cache writes
            # 2. run recurrent delta rule with intermediate SSM-state cache writes
            # 3. defer final state selection to kv_cache_manager.update_mamba_states()
            intermediate_state_indices = kwargs["intermediate_state_indices"]

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
                # PDL chain: in_proj GEMM -> conv1d -> replay verify kernel (replay only)
                launch_dependent_kernels=kwargs.get("use_replay", False),
                launch_with_pdl_wait=kwargs.get("use_replay", False),
                hoist_x=True,
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
            if kwargs.get("use_replay", False):
                output_d = None
                if output is not None:
                    output_d = output.view(
                        num_decodes,
                        draft_token_num,
                        self.num_v_heads // self.attn_tp_size,
                        self.head_v_dim,
                    )
                return self._replay_verify_recurrent(
                    query,
                    key,
                    value,
                    a,
                    b,
                    ssm_states,
                    cache_indices,
                    num_decodes,
                    draft_token_num,
                    kwargs.get("replay_metadata"),
                    kwargs.get("layer_cache"),
                    kwargs.get("replay_work_items"),
                    kwargs.get("replay_n_writes"),
                    output_d,
                    packed_qkv=mixed_qkv,
                    use_all_layer_commit=kwargs.get("use_cached_replay_all_layer_commit", False),
                ).view(
                    1,
                    num_decodes * draft_token_num,
                    self.num_v_heads // self.attn_tp_size,
                    self.head_v_dim,
                )

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
            recurrent_state_indices = kwargs["intermediate_state_indices"]

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

        return self._recurrent_update(
            mixed_qkv, a, b, ssm_states, cache_indices, query_start_loc_long, output
        )

    def _recurrent_update(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc_long: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        split_decode = (
            num_decode_tokens > 0
            and not is_target_verify
            and os.getenv("FORCE_DETERMINISTIC", "0") == "1"
        )
        inforward_split = kwargs.get("inforward_split")

        conv_states_to_use = conv_states
        if inforward_split is not None:
            if num_decode_tokens > 0 or is_target_verify:
                raise RuntimeError("In-forward Mamba snapshots need a context-only batch")
            # The convolution state at P: the d_conv - 1 raw inputs before P.
            inforward_window = mixed_qkv.index_select(0, inforward_split.conv_win)

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
                assert kwargs.get("use_replay", False) or intermediate_ssm_states is not None

                intermediate_state_indices = kwargs["intermediate_state_indices"]
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
                    # PDL chain: conv1d -> replay verify kernel (replay only)
                    launch_dependent_kernels=kwargs.get("use_replay", False),
                    hoist_x=True,
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
                layer_cache = kwargs.get("layer_cache")
                if (
                    kwargs.get("use_replay", False)
                    and layer_cache is not None
                    and layer_cache.cache_base is not None
                ):
                    # FlashInfer ring replay needs q/k/v as slices of one buffer
                    # (strided mode), so slice the conv output like forward_decode.
                    key_size = self.key_dim // self.attn_tp_size
                    query_d = mixed_qkv_d[..., :key_size]
                    key_d = mixed_qkv_d[..., key_size : key_size * 2]
                    value_d = mixed_qkv_d[..., key_size * 2 :]
                else:
                    # Copies q/k/v into separate contiguous tensors.
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
                    None if split_decode else mixed_qkv_d,
                    a[:num_prefill_tokens] if split_decode else a,
                    b[:num_prefill_tokens] if split_decode else b,
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
            if inforward_split is not None:
                _inforward_split_conv_states(
                    conv_states, inforward_window, inforward_split, kwargs["inforward_stage"][1]
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
                output_p = output[:, :num_prefill_tokens, :, :] if output is not None else None
                # Same indexed, in-place state I/O as the non-verify prefill path
                # below: read each request's initial state from its pool slot and
                # write the final state back to it inside the kernel, instead of
                # gathering a [num_prefill, H, V, K] copy and scattering it back.
                # Preconditions match that path: fresh slots were zeroed in
                # forward_core, and prefill slots never alias the decode slots
                # verified in the same step.
                attn_out_prefill, _ = chunk_gated_delta_rule(
                    q=query_p,
                    k=key_p,
                    v=value_p,
                    g=g_p,
                    beta=beta_p,
                    initial_state=ssm_states,
                    initial_state_indices=state_indices_p,
                    inplace_indexed_state_update=True,
                    output_final_state=False,
                    cu_seqlens=query_start_loc_long[: num_prefill + 1],
                    head_first=False,
                    use_qk_l2norm_in_kernel=False,
                    output=output_p,
                    state_workspace=kwargs.get("state_workspace"),
                )

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

            if kwargs.get("use_replay", False):
                attn_out_decode = self._replay_verify_recurrent(
                    query_d,
                    key_d,
                    value_d,
                    a_d,
                    b_d,
                    ssm_states,
                    state_indices_d,
                    num_decodes,
                    draft_token_num,
                    kwargs.get("replay_metadata"),
                    kwargs.get("layer_cache"),
                    kwargs.get("replay_work_items"),
                    kwargs.get("replay_n_writes"),
                    output_d,
                    use_all_layer_commit=kwargs.get("use_cached_replay_all_layer_commit", False),
                ).reshape(1, num_decode_tokens, out_v_heads, self.head_v_dim)
            elif _can_use_flashinfer_gdn_verify(
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
                recurrent_state_indices = kwargs["intermediate_state_indices"]

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

        if inforward_split is not None:
            return _inforward_split_prefill_core(
                query,
                key,
                value,
                g,
                beta,
                ssm_states,
                inforward_split,
                kwargs["inforward_stage"][0],
                output,
            )

        core_attn_out, _ = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=state_indices_p if split_decode else cache_indices,
            # This path writes recurrent state directly back into the shared
            # pool; callers **must** ensure cache_indices do not alias live slots.
            inplace_indexed_state_update=True,
            output_final_state=False,
            cu_seqlens=_prefill_query_start_loc(query_start_loc_long, num_prefill)
            if split_decode
            else query_start_loc_long,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
            output=output[:, :num_prefill_tokens]
            if split_decode and output is not None
            else output,
            state_workspace=kwargs.get("state_workspace"),
        )

        if split_decode:
            # Decode retains the same normalization precision and state update
            # kernel when prefill requests share its batch.
            output_d = output[:, num_prefill_tokens:] if output is not None else None
            decode_out = self._recurrent_update(
                mixed_qkv_d,
                a[num_prefill_tokens:],
                b[num_prefill_tokens:],
                ssm_states,
                state_indices_d,
                query_start_loc_long[num_prefill:] - num_prefill_tokens,
                output_d,
            )
            if output is not None:
                return output
            return torch.cat((core_attn_out, decode_out), dim=1)

        return core_attn_out

    def _get_prefill_state_workspace(
        self, attn_metadata: AttentionMetadata, ssm_states: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Reserve maximum state scratch on first use and reuse it across layers."""
        # Blackwell uses indexed pool I/O directly, without state scratch.
        if not _use_flashinfer_gdn_prefill() or is_sm_100f():
            return None
        key = (ssm_states.device, ssm_states.shape[1:])
        # First use during warmup reserves the configured maximum, not the current batch.
        # Sequential layers share this persistent input/output scratch.
        workspaces = self.model_config.extra_attrs.setdefault("gdn_state_workspaces", {})
        workspace = workspaces.get(key)
        capacity = min(
            attn_metadata.max_num_sequences or attn_metadata.max_num_requests,
            attn_metadata.max_num_tokens,
        )
        if workspace is None or workspace[0].shape[0] < capacity:
            state_in = torch.empty(
                (capacity, *ssm_states.shape[1:]), dtype=torch.float32, device=ssm_states.device
            )
            workspace = workspaces[key] = (state_in, torch.empty_like(state_in))
        return workspace

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_metadata: Mamba2Metadata,
        spec_metadata: Optional[SpecMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        h_fp8: Optional[torch.Tensor] = None,
    ):
        fused = self._fused_verify_inproj(hidden_states, h_fp8, attn_metadata, spec_metadata)
        if fused is not None:
            return self.out_proj(fused, all_reduce_params=all_reduce_params)

        mixed_qkv, z, a, b, ba_ready = self._compute_tokenwise_inputs(hidden_states, h_fp8)
        fused = self._fused_verify(mixed_qkv, z, a, b, ba_ready, attn_metadata, spec_metadata)
        if fused is not None:
            return self.out_proj(fused, all_reduce_params=all_reduce_params)
        # Every path below reads a / b.
        if ba_ready is not None:
            ba_ready.wait()
        # Have prefill and decode write into disjoint slices of the same buffer, avoiding
        # concatenation when eager execution mixes both kinds of requests.
        attn_out = mixed_qkv.new_empty(
            (1, mixed_qkv.shape[0], self.num_v_heads_per_tp, self.head_v_dim)
        )

        use_breakable_cuda_graph = not is_torch_compiling() and is_in_breakable_cuda_graph()
        if self.register_to_config and (is_torch_compiling() or use_breakable_cuda_graph):
            maybe_bcg_gdn_custom_op_inplace(mixed_qkv, a, b, self.layer_idx_str, attn_out)
        else:
            attn_out = self.forward_core(
                mixed_qkv,
                a,
                b,
                attn_metadata,
                mamba_metadata,
                spec_metadata=spec_metadata,
                output=attn_out,
            )

        return self._postprocess_gdn_output(attn_out, z, all_reduce_params)

    def prefill_graph_key(self, attn_metadata: AttentionMetadata) -> Optional[tuple]:
        """Key of the CUDA graph of this layer's core under a breakable prefill
        graph, or None to run it eagerly.

        Only context-only batches qualify. Besides the token inputs, the core's
        launches depend on the request count and the in-forward snapshot
        split's shape; the key also names the metadata and state buffers the
        graph reads, so it never replays against moved ones.
        """
        num_contexts = attn_metadata.num_contexts
        if (
            not self.prefill_graphable
            or num_contexts == 0
            or attn_metadata.seq_lens.shape[0] != num_contexts
        ):
            return None
        mamba_metadata = attn_metadata.mamba_metadata
        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        key = (
            num_contexts,
            mamba_metadata.query_start_loc.data_ptr(),
            mamba_metadata.query_start_loc_long.data_ptr(),
            mamba_metadata.state_indices.data_ptr(),
            mamba_metadata.has_initial_states.data_ptr(),
            layer_cache.conv.data_ptr(),
            layer_cache.temporal.data_ptr(),
        )
        split = mamba_metadata.inforward_split
        if split is not None:
            key += (split.k, split.num_tail_tokens, split.cu.data_ptr())
        return key

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
            reset_recurrent_state_rows(
                ssm_states,
                state_indices_p,
                has_initial_states_p,
                conv_states,
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
        intermediate_state_indices = (
            _verify_intermediate_state_indices(
                attn_metadata.kv_cache_manager, num_decodes, state_indices.device
            )
            if is_target_verify
            else None
        )
        replay_metadata = (
            attn_metadata.kv_cache_manager.get_replay_state_update_metadata()
            if is_target_verify
            else None
        )
        use_replay = replay_metadata is not None

        partitioned_replay = use_replay and attn_metadata.kv_cache_manager.gdn_replay_partitioned(
            num_decodes
        )
        # Work items only feed Triton's large-batch replay + all-layer commit;
        # the FlashInfer ring kernel always folds in-kernel.
        pass_replay_work_items = partitioned_replay and not getattr(
            attn_metadata.kv_cache_manager, "use_gdn_flashinfer_ring_replay", False
        )
        use_cached_replay_all_layer_commit = partitioned_replay and getattr(
            attn_metadata.kv_cache_manager,
            "use_gdn_cached_replay_all_layer_commit",
            False,
        )
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
            "intermediate_state_indices": intermediate_state_indices,
            "use_replay": use_replay,
            "replay_metadata": replay_metadata,
            "layer_cache": layer_cache,
            "replay_work_items": mamba_metadata.replay_work_items[:num_decodes]
            if pass_replay_work_items
            else None,
            "replay_n_writes": mamba_metadata.replay_n_writes if pass_replay_work_items else None,
            "use_cached_replay_all_layer_commit": use_cached_replay_all_layer_commit,
        }
        inforward_split = (
            getattr(mamba_metadata, "inforward_split", None) if num_prefills > 0 else None
        )
        if inforward_split is not None:
            kwargs["inforward_split"] = inforward_split
            kwargs["inforward_stage"] = (
                attn_metadata.kv_cache_manager.inforward_snapshot_stage_views(self.layer_idx)
            )
        if num_prefills > 0:
            attn_out = self.forward_extend(
                conv_states,
                ssm_states,
                spec_metadata=spec_metadata,
                intermediate_conv_states=intermediate_conv_states,
                intermediate_ssm_states=intermediate_ssm_states,
                is_target_verify=is_target_verify,
                output=output,
                state_workspace=self._get_prefill_state_workspace(attn_metadata, ssm_states),
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
