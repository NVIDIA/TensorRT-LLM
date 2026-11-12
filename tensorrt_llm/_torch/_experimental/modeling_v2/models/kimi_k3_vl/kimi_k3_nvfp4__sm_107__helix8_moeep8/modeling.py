# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), helix8_moeep8: the target class.

Serves generation over Helix CP8 (tp 1, no attention DP), the routed experts split MoE EP8 on
CUTEDSL_FC12: the c24 and c75 recipes.

The model below is this target's copy of the helix8 targets' SM107 glue
(``KimiK3Sm107ForConditionalGeneration``), built for these constants (``CONSTANTS``).

Engine settings (block reuse, page size) are the recipe's: the model loader applies model defaults of
the checkpoint's built-in class before a Modeling V2 target is routed, so a target cannot set
``llm_args``.

Kimi K3 NVFP4 on Rubin (SM107), the helix8 generation targets: the model.

``KimiK3Sm107ForConditionalGeneration`` serves generation over Helix CP8 (tp 1, no attention DP)
for the two helix8 targets, ``helix8_moeep8`` (routed experts MoE EP8) and ``helix8_moetp2ep4``
(MoE TP2 x EP4). The layout, the MoE split included, comes from the engine's ``Mapping``. The
targets' optimizations (``HELIX8_CONSTANTS``) are built into this code; a target class declares
them as its ``K3_CONSTANTS`` and the class refuses any other set.

The checkpoint is the vision-language model's; this target is text only. It builds no vision
tower (its checkpoint tensors are never read, ``weights.py``) and refuses multimodal input. The
text model ``KimiK3DefaultForCausalLM``:

* 93 decoder layers (``KimiK3DefaultDecoderLayer``): 69 KDA (Kimi Delta Attention, linear
  attention) layers and 24 MLA (absorbed MQA, NoPE) layers, per the 1-indexed
  ``linear_attn_config.kda_layers`` / ``full_attn_layers`` schedule.
* Layer 0 has a dense SiTU MLP; layers 1..92 the latent MoE of ``moe_runtime.py``: 896 routed
  experts at top-16 (sigmoid + ``e_score_correction_bias`` routing), 2 shared experts, and the
  latent down / norm / up projections around the routed experts.
* Attention residuals (``attn_res.py``): before ``input_layernorm``, before
  ``post_attention_layernorm`` and at the model output each token mixes a bank of earlier
  sublayer outputs; a snapshot is appended whenever ``layer_idx % attn_res_block_size == 0``.

Caching: per request, a KDA layer keeps a short-convolution window
``[3 * num_heads * head_dim, W]`` (bf16, sections ``[q | k | v]``) and a delta-rule recurrent
state ``[num_heads, head_dim, head_dim]`` (fp32, ``[H, V, K]``) in ``KimiK3HybridCacheManager``
(KV cache manager V2); an MLA layer uses the paged KV cache with one KV head of
``kv_lora_rank + qk_rope_head_dim`` (576).

Parallelism: attention is head-split over the TP group, its KV sequence split over the CP
group under Helix; there is no attention DP. The routed experts are split MoE TP x EP over the
TP group (EP only unless ``moe_tensor_parallel_size`` / ``moe_expert_parallel_size`` are set).
Routing is computed on every rank, and the routed partial sums are all-reduced in the latent
space, before ``routed_expert_norm`` / ``routed_expert_up_proj``. The shared experts are
tensor-parallel over the TP group and reduce on an auxiliary stream while the routed experts run
on the main stream. ``lm_head`` is vocab-parallel with a gather, so every rank has the full
logits.

Speculative decoding: one-engine DSpark with the family's standalone MLA drafter and worker
(``dspark/``); KDA and MLA verify several tokens per step and keep the accepted state. Chunked
prefill continues a request's KDA state and attends over its cached MLA prefix. KV cache
block reuse is opt-in (``kv_cache_config.enable_block_reuse``); periodic KDA state snapshots
come from ``mamba_state_config.periodic_snapshot_interval``. Pipeline parallelism is not
supported.

Checkpoint loading is in ``weights.py``.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import gc
import math
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F
import triton
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from tensorrt_llm._torch import kimi_k3_cache_policy as _kimi_k3_cache_policy
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch.attention.backends import AttentionMetadata, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
    FP4_MLA_K_RESIDUAL_DIM,
    _fp4_mla_attention_backend,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.layout import (
    _get_fp4_mla_global_scale,
    _get_fp4_mla_kv_cache_tensors,
    _validate_fp4_mla_kv_storage_shape,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.metadata import (
    _materialize_fp4_mla_device_page_table_for_forward,
)
from tensorrt_llm._torch.custom_ops import BufferKind
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE, sm_reserve
from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams, HelixAllToAllNative
from tensorrt_llm._torch.distributed.mnnvl_memory import HelixCpMnnvlMemory, MnnvlMemory
from tensorrt_llm._torch.distributed.ops import use_mnnvl_fabric_handle
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, RuntimeHooks
from tensorrt_llm._torch.utils import (
    ActivationType,
    AuxStreamType,
    Fp4QuantizedTensor,
    get_model_extra_attrs,
    maybe_compiled_cat,
    maybe_compiled_copy_,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..common.kda_metadata import K3MambaMetadata, KDASnapshotPlan
from ..common.multi_stream import do_multi_stream
from ..common.parallel import TensorParallelMode
from ..common.weight_io import check_text_only
from ..dspark.drafter import K3DSparkPath
from ..dspark.shell import KimiK3SpecForCausalLM
from ..kernels.k3r_kda._kda_kernels import (
    PAD_SLOT_ID,
    K3KDAKernelDispatch,
    extract_transpose_prefill_slice,
    kda_mtp_verify_ctas,
    reset_recurrent_state_rows,
)
from ..kernels.k3r_mla_ctx import fp8_static_quant
from ..kernels.k3r_mla_ctx.mla_kv_fp8 import mla_kv_to_fp8
from ..kernels.k3r_spec.spec_decode_head import spec_decode_head
from ..ops.activation.sm107_mlp import GatedMlp, gated_activation, subgroup_mapping
from ..ops.attention import _fp4_mla_decode as fp4_mla_decode
from ..ops.attention import k3_fp4_mla_decode as _decode_op
from ..ops.attention import k3_mla_attention as mla_attn
from ..ops.attention import k3_mla_cache as mla_cache
from ..ops.attention.k3_context_fmha import k3_context_fmha
from ..ops.attention.k3_mla_state import K3MlaAttnState
from ..ops.attention.static_quantize_e4m3 import static_quantize_e4m3
from ..ops.cache.k3_sm107_cache_manager import KimiK3Sm107CacheManager
from ..ops.comm.k3_attn_res_lamport import k3_attn_res_lamport
from ..ops.comm.k3_latent_ar_rmsnorm import k3_latent_ar_rmsnorm
from ..ops.comm.sm107_allreduce import (
    LamportPending,
    TpAllReduce,
    finish_lamport_push,
    lamport_push_target,
    lamport_push_tensor,
)
from ..ops.comm.sm107_allreduce import mnnvl_workspace_mpi_comm as _k3_mnnvl_mpi_comm
from ..ops.common.sm107_launch import USE_PDL, log_constants
from ..ops.common.sm107_streams import fork_join
from ..ops.embedding.sm107_embedding import EmbeddingTable, gather_embedding
from ..ops.gemm.k3_fp8w_lowm_gemm import k3_fp8w_lowm_gemm as k3_fp8w_lowm_gemm_op
from ..ops.gemm.k3_fp8w_lowm_gemm_push import k3_fp8w_lowm_gemm_push as k3_fp8w_lowm_gemm_push_op
from ..ops.gemm.k3_fp8w_lowm_mxfp8_gemm import k3_fp8w_lowm_mxfp8_gemm as k3_fp8w_lowm_mxfp8_gemm_op
from ..ops.gemm.k3_fp8w_lowm_mxfp8_push import k3_fp8w_lowm_mxfp8_push as k3_fp8w_lowm_mxfp8_push_op
from ..ops.gemm.k3_fp8w_mid_gemm import k3_fp8w_mid_gemm as k3_fp8w_mid_gemm_op
from ..ops.gemm.k3_fp8w_mid_gemm_push import k3_fp8w_mid_gemm_push as k3_fp8w_mid_gemm_push_op
from ..ops.gemm.k3_latent_down_fp4 import k3_latent_down_fp4 as k3_latent_down_fp4_op
from ..ops.gemm.k3_mxfp8_gemm import k3_mxfp8_gemm as k3_mxfp8_gemm_op
from ..ops.gemm.k3_mxfp8_gemm_fp8_out import k3_mxfp8_gemm_fp8_out
from ..ops.gemm.sm107_linear import (
    Bf16Linear,
    Fp8BlockScaleLinear,
    fp8_block_scale_gemm,
    prepare_fp8_block_weight,
)
from ..ops.moe.k3_moe_residual_add import k3_moe_residual_add as k3_moe_residual_add_op
from ..ops.moe.k3_route_tail import k3_route_tail as k3_route_tail_op
from ..ops.moe.k3_router_logits import k3_router_logits as k3_router_logits_op
from ..ops.moe.k3r_experts import KERNEL_SETS as _K3_ROUTED_KERNEL_SETS
from ..ops.moe.k3r_experts import K3RoutedExperts, SortedRouting, SortTables, zero_on_side_stream
from ..ops.norm.attn_res_fwd import attn_res_fwd
from ..ops.norm.k3_attn_res_norm import k3_attn_res_norm
from ..ops.norm.rms_norm_gated_token_major import rms_norm_gated_token_major
from ..ops.norm.sm107_norm import RMSNormWeight, rms_norm
from ..ops.ssm.causal_conv1d_fwd import causal_conv1d_fwd
from ..ops.ssm.sm107_short_conv import ShortConvWeight


@dataclasses.dataclass(frozen=True)
class K3Sm107Constants:
    """What one Kimi K3 SM107 target runs. A field is a fixed property of the target."""

    # The speculative decode step's head in one kernel (``RuntimeHooks.decode_prologue``) and the
    # token embedding as a vectorized row copy (formerly TLLM_SPEC_HEAD_FUSION).
    spec_head_fusion: bool
    # MoE (moe_runtime.py, fp8_weight_read.py, modeling.py).
    fuse_moe_residual: bool
    fp8w_lowm_gemm: bool
    ld_fused: bool
    shared_sm: int
    # FC12 die-major expert layout and schedule (the routed experts' kernel set).
    fc12_die_major: bool
    # The routed experts of a MEGAMOE_CUTEDSL layer: the target's MegaMoE module class, which
    # ``K3MoERuntime`` builds with keyword arguments (``num_experts``, ``top_k``, ``hidden_size``,
    # ``intermediate_size``, ``mapping``, ``layer_idx``, the two SiTU soft-caps,
    # ``max_num_tokens``, ``dtype``, ``skip_create_weights_in_init``); ``None`` where the target
    # runs no MegaMoE.
    megamoe_experts: Optional[Callable[..., Any]]
    router_exact_kda: bool
    # The MoE gate GEMM on BF16 operands with FP32 accumulation and output (else the FP32 GEMM of
    # the upcast operands; the two can pick different top-k experts at a tie).
    router_bf16_gate: bool
    # The BF16 shared and latent MoE MLP weights are read as block-scaled FP8 (quantized once after
    # loading).
    fp8_weight_read_moe_mlp: bool
    latent_ar_own_ws: bool
    # Attention residual (attn_res.py, kernels/k3r_attn_res).
    attn_res_pdl: bool
    attn_res_pdl_max_m: int
    attn_res_mxfp8: bool
    attn_res_early_trigger: bool
    # KDA (kda.py).
    kda_prepad_prefill: bool
    kda_fused_conv_prefill: bool
    kda_l2_prefetch: bool
    # MLA and its FP4 generation decode (mla.py, ops/attention/_fp4_mla_decode.py).
    mla_q_norm_pdl: bool
    fp4_mla_page_plan: bool
    fp4_mla_reduce_pdl: bool
    # DSpark (dspark/, kernels/k3r_spec/spec_tail_kernels.py).
    spec_tail_fusion: bool
    dspark_tail_v2: bool
    # The target is a disaggregated context server: its drafter only writes the context KV the
    # generation servers' drafter reads (``context_kv_only`` of ``dspark/drafter.py``).
    dspark_context_kv_only: bool
    # Device memory, in bytes, the engine leaves outside the PyTorch memory cap
    # (``RuntimeHooks.torch_mem_safety_margin``).
    torch_mem_safety_margin: int


CONSTANTS = K3Sm107Constants(
    spec_head_fusion=True,
    fuse_moe_residual=True,
    fp8w_lowm_gemm=True,
    ld_fused=True,
    shared_sm=16,
    fc12_die_major=True,
    megamoe_experts=None,
    router_exact_kda=True,
    router_bf16_gate=True,
    fp8_weight_read_moe_mlp=True,
    latent_ar_own_ws=True,
    attn_res_pdl=True,
    attn_res_pdl_max_m=128,
    attn_res_mxfp8=True,
    attn_res_early_trigger=True,
    kda_prepad_prefill=False,
    kda_fused_conv_prefill=False,
    kda_l2_prefetch=True,
    mla_q_norm_pdl=True,
    fp4_mla_page_plan=True,
    fp4_mla_reduce_pdl=True,
    spec_tail_fusion=True,
    dspark_tail_v2=True,
    dspark_context_kv_only=False,
    torch_mem_safety_margin=4 * 1024**3,
)


def _resolve_kimi_situ_betas(cfg: Any) -> tuple[float, float]:
    """Return the finite SiTu betas required by the routed-expert kernels."""
    config_situ_beta = getattr(cfg, "activation_situ_beta", None)
    situ_beta = 1.0 if config_situ_beta is None else config_situ_beta
    situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
    if situ_linear_beta is None:
        raise ValueError(
            "Kimi K3 routed SiTu experts require activation_situ_linear_beta; "
            "None means an identity linear branch that the fused kernels cannot represent."
        )
    if situ_beta <= 0 or situ_linear_beta <= 0:
        raise ValueError(
            f"Kimi K3 SiTu betas must be positive; got {situ_beta} and {situ_linear_beta}."
        )
    return float(situ_beta), float(situ_linear_beta)


# ---------------------------------------------------------------------------
# Config helpers.
# ---------------------------------------------------------------------------


def _get_text_config(pretrained_config: "PretrainedConfig"):
    """Return the Kimi text config, unwrapping a composite kimi_k3 config."""
    if getattr(pretrained_config, "model_type", None) == "kimi_k3" or (
        not hasattr(pretrained_config, "linear_attn_config")
        and hasattr(pretrained_config, "text_config")
    ):
        return pretrained_config.text_config
    return pretrained_config


def _is_kda_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["kda_layers"]


def _is_mla_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["full_attn_layers"]


# ---------------------------------------------------------------------------
# attn_res: per-token snapshot mixing (HF `_apply_attn_res`).
# ---------------------------------------------------------------------------

# Fused attention-residual kernels (every Kimi K3 recipe runs the "split" topology):
# * up to _FUSED_ATTN_RES_MAX_TOKENS tokens the per-token C++ kernels (one CTA for N <= 4, one 8-CTA
#   cluster per token for N >= 5): trtllm::attn_res_rmsnorm_fwd / attn_res_add_rmsnorm_fwd;
# * above it the persistent kernel (one CTA per SM looping over tokens),
#   trtllm::attn_res_add_rmsnorm_persistent_fwd, where it implements the shape (H == 7168 and
#   2 <= N <= 9);
# * elsewhere trtllm::attn_res_fwd and the RMSNorm, or the FP32 reference past that op's contract.
# The K3 PDL kernels take precedence up to their own token limit.
_FUSED_ATTN_RES_MAX_TOKENS = 32


def _use_persistent_attn_res(M: int, H: int, N: int) -> bool:
    """Whether the persistent kernel serves this mix: H == 7168, 2 <= N <= 9 and more than
    ``_FUSED_ATTN_RES_MAX_TOKENS`` tokens (its grid is sized by the SM count, not the token
    count)."""
    return H == 7168 and 2 <= N <= 9 and M > _FUSED_ATTN_RES_MAX_TOKENS


# Token limit of trtllm::attn_res_fwd (cpp/tensorrt_llm/thop/attnResOp.cpp).
_ATTN_RES_OP_MAX_TOKENS = 16384


def _apply_attn_res_fused(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: RMSNormWeight
) -> Optional[torch.Tensor]:
    """Fused attn_res via the in-tree ``trtllm::attn_res_fwd`` op.

    Returns ``None`` when the call falls outside the fused kernel's
    contract (dtype/shape/arch) so the caller can use the exact fp32 reference
    instead. ``block_residual`` is kept in the kernel-native ``[K, M, H]``
    layout. Candidate order matches the reference: snapshots first, the
    running prefix sum last.
    """
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    if K + 1 > 12 or not (4096 <= H <= 8192 and H % 1024 == 0):
        return None
    attn_res_op = attn_res_fwd
    proj_w = proj.weight.reshape(-1).to(torch.bfloat16).contiguous()
    norm_w = norm.weight.to(torch.bfloat16).contiguous()

    def run(prefix, block):
        t = prefix.shape[0]
        output, _rsigma, _probs, _logits = attn_res_op(
            prefix.reshape(t, 1, H).contiguous(),
            block.reshape(K, t, 1, H).contiguous(),
            proj_w,
            norm_w,
            float(norm.eps),
        )
        return output.reshape(t, H)

    if M <= _ATTN_RES_OP_MAX_TOKENS:
        return run(prefix_sum, block_residual)
    # The op takes at most _ATTN_RES_OP_MAX_TOKENS tokens per call and is
    # per-token, so longer prefill chunks run in slices; the fp32 reference
    # below would otherwise cost seconds per iteration at 24k+ tokens.
    out = torch.empty_like(prefix_sum)
    for lo in range(0, M, _ATTN_RES_OP_MAX_TOKENS):
        hi = min(M, lo + _ATTN_RES_OP_MAX_TOKENS)
        out[lo:hi] = run(prefix_sum[lo:hi], block_residual[:, lo:hi])
    return out


def _note_attn_res_fusion(site: str, fused: bool, M: int, H: int, N: int) -> None:
    """Report whether a fused kernel served the mix, once per shape (a shape gate that rejects
    the call is otherwise silent). Emitted at debug level: it is once per distinct shape, not once
    per process, so it is a diagnostic rather than a summary.
    """
    logger.debug_once(
        f"Kimi K3 attn-res fusion [{site}]: "
        f"{'FUSED' if fused else 'fallback'} (M={M}, H={H}, N={N})",
        key=f"kimi_k3_attn_res_fusion_{site}_{fused}_{M}_{H}_{N}",
    )


# The K3 PDL kernels: up to ``_K3_ATTN_RES_PDL_MAX_M`` tokens, the fused attn_res(+add)
# +RMSNorm runs as the k3_attn_res_norm entry (kernels/k3_attn_res.py):
# a CTA cluster per token that loads the weights and snapshot rows and reduces their statistics
# before griddepcontrol.wait, i.e. under the kernel in front of it. It takes precedence over the
# persistent topology at those token counts. The updated prefix sum is bit-identical; the
# normalized output can differ from the C++ per-token kernels by one BF16 rounding step in a small
# fraction of the elements.
# It also leaves the MoE residual update residual + bf16(routed + shared) of the BF16-add path
# (kimi_k3_moe_residual) to the next layer's pre-attention mix, which computes it bit-identically
# (k3_attn_res_moe_add_rmsnorm) behind the latent-up GEMM.


# Every K3 PDL kernel releases its dependent grid right after ``griddepcontrol.wait`` (k3_attn_res
# ``early_trigger``).
_K3_ATTN_RES_PDL_MAX_M = 128


# Where the mix after attention (A_mid) adds the attention output to the running prefix sum, the
# attention block pushes its o_proj output into the o_proj all-reduce's MNNVL Lamport workspace
# (layers) and leaves the call unfinished; A_mid finishes it (k3_attn_res ``lamport_finish``:
# one-shot transport up to the C++ one-shot size, two-shot above, where each row's owner reduces
# and multicasts it) and advances the workspace: no finish kernel or C++ all-reduce runs in
# between. Where o_proj's GEMM has no push epilogue (the MXFP8 GEMM on a prequantized KDA output)
# a copy kernel pushes its output. The sums have the order and rounding of the finish kernel, so
# A_mid produces the same bits.


class _DeferredMoEResidual(NamedTuple):
    """A MoE layer's ``residual + bf16(routed + shared)`` that the next layer fuses or materializes."""

    residual: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor

    def materialize(self) -> torch.Tensor:
        return k3_moe_residual_add_op(self.routed, self.shared, self.residual)


class _DeferredRpTail(NamedTuple):
    """A MoE layer's ``bf16(residual + all-reduce of the ranks' row-parallel tail partials)``:
    ``result`` is the pushed all-reduce, left for the next layer's pre-attention mix to finish
    (``LamportPending``)."""

    residual: torch.Tensor
    result: Any

    def materialize(self) -> torch.Tensor:
        # A BF16 add rounds the FP32 sum to nearest, as the fused mix does.
        return self.residual + self.result.materialize().view(self.residual.shape)


def _pdl_serves(num_tokens: int) -> bool:
    return num_tokens <= _K3_ATTN_RES_PDL_MAX_M


def _defer_moe_residual(num_tokens: int) -> bool:
    """Whether a MoE layer leaves its BF16 residual update to the next layer."""
    return _pdl_serves(num_tokens)


def _materialize_prefix_sum(hidden_states):
    if isinstance(hidden_states, (_DeferredMoEResidual, _DeferredRpTail)):
        return hidden_states.materialize()
    return hidden_states


def _k3_attn_res_pdl(
    prefix_sum: torch.Tensor,
    addend: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
):
    """``k3_attn_res_norm`` (with ``addend`` as the one addend) where it serves, else ``None``."""
    if not _pdl_serves(prefix_sum.shape[0]):
        return None
    from ..kernels.k3r_attn_res import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    addends = () if addend is None else (addend,)
    if not k3_attn_res.can_use_k3_attn_res(prefix_sum, addends, block_residual, *weights):
        return None
    eps = float(norm.eps)
    out_eps = float(output_norm.eps)
    return k3_attn_res_norm(
        prefix_sum,
        addends,
        block_residual,
        *weights,
        eps,
        out_eps,
        early_trigger=True,
    )


def _k3_attn_res_lamport_applies(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> bool:
    """Whether A_mid on ``prefix_sum`` can read its addend from a Lamport workspace: the
    vectorized PDL attn_res kernel serves the ADD_PREFIX mix."""
    if not _pdl_serves(prefix_sum.shape[0]):
        return False
    from ..kernels.k3r_attn_res import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    return k3_attn_res.can_use_k3_attn_res(prefix_sum, (prefix_sum,), block_residual, *weights)


def _k3_attn_res_pdl_lamport(
    prefix_sum: torch.Tensor,
    addend: LamportPending,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(prefix_sum + all-reduce result, normed mix)``: the mix finishes the pushed all-reduce and
    reads the result from the Lamport workspace (``k3_attn_res_lamport``), or
    ``None`` if that does not apply."""
    if tuple(addend.shape) != tuple(prefix_sum.shape) or not _k3_attn_res_lamport_applies(
        prefix_sum, block_residual, proj, norm, output_norm
    ):
        return None
    M, H = prefix_sum.shape
    _note_attn_res_fusion(
        "add+attn_res+norm/pdl/lamport_finish",
        True,
        M,
        H,
        int(block_residual.shape[0]) + 1,
    )
    ws = addend.workspace
    return k3_attn_res_lamport(
        prefix_sum,
        ws.uc_local,
        ws.flags.data_ptr(),
        block_residual,
        proj.weight.reshape(-1),
        norm.weight,
        output_norm.weight,
        float(norm.eps),
        float(output_norm.eps),
        *addend.finish_args,
        False,
        early_trigger=True,
    )


def _k3_attn_res_pdl_rp_tail(
    deferred: _DeferredRpTail,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
    mxfp8: bool = False,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """``(prefix_sum, normed mix)`` of a deferred row-parallel MoE tail: the mix finishes the
    pushed all-reduce, reads it from the Lamport workspace and adds it to the residual with a
    round-to-nearest BF16 add; ``None`` if the
    vectorized PDL kernel does not serve the shape. With ``mxfp8``, ``(prefix_sum, normed mix, xq,
    xs)``."""
    residual, result = deferred
    if tuple(result.shape) != tuple(residual.shape):
        return None
    from ..kernels.k3r_attn_res import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    if not k3_attn_res.can_use_k3_attn_res(residual, (residual,), block_residual, *weights):
        return None
    M, H = residual.shape
    _note_attn_res_fusion(
        "rp_tail+attn_res+norm/pdl/lamport_finish",
        True,
        M,
        H,
        int(block_residual.shape[0]) + 1,
    )
    ws = result.workspace
    return k3_attn_res_lamport(
        residual,
        ws.uc_local,
        ws.flags.data_ptr(),
        block_residual,
        *weights,
        float(norm.eps),
        float(output_norm.eps),
        *result.finish_args,
        True,
        mxfp8=mxfp8,
        early_trigger=True,
    )


def _k3_attn_res_pdl_moe_add(
    deferred: _DeferredMoEResidual,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
    mxfp8: bool = False,
    entry_trigger: bool = False,
    mx_atom: bool = False,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """``(prefix_sum, normed mix)`` of a deferred MoE residual update, or ``None``; with
    ``mxfp8``, ``(prefix_sum, normed mix, xq, xs)`` (the MXFP8 form of the mix).
    ``entry_trigger`` (with ``mxfp8``): the kernel releases its dependent at entry
    (``k3_attn_res_norm``), for a KDA QKVG projection that stages its weights.
    ``mx_atom`` (with ``mxfp8``): ``xs`` in the mid-M GEMM's layout."""
    residual, routed, shared = deferred
    if not _pdl_serves(residual.shape[0]):
        return None
    from ..kernels.k3r_attn_res import k3_attn_res

    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    if not k3_attn_res.can_use_k3_attn_res(residual, (routed, shared), block_residual, *weights):
        return None
    M, H = residual.shape
    _note_attn_res_fusion("moe_add+attn_res+norm/pdl", True, M, H, int(block_residual.shape[0]) + 1)
    if mxfp8:
        return k3_attn_res_norm(
            residual,
            (routed, shared),
            block_residual,
            *weights,
            float(norm.eps),
            float(output_norm.eps),
            mxfp8=True,
            entry_trigger=entry_trigger,
            mx_atom=mx_atom,
            early_trigger=True,
        )
    return k3_attn_res_norm(
        residual,
        (routed, shared),
        block_residual,
        *weights,
        float(norm.eps),
        float(output_norm.eps),
        early_trigger=True,
    )


def _apply_attn_res_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> Optional[torch.Tensor]:
    """Fuse attention-residual mixing with its immediately following norm."""
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    N = K + 1
    # The fused path is taken for M <= _FUSED_ATTN_RES_MAX_TOKENS, H == 7168 and
    # N <= 12, which is the measured window; larger token counts have not been
    # measured and fall back to the unfused add + attn_res_fwd + RMSNorm path.
    pdl_out = _k3_attn_res_pdl(prefix_sum, None, block_residual, proj, norm, output_norm)
    if pdl_out is not None:
        _note_attn_res_fusion("attn_res+norm/pdl", True, M, H, N)
        return pdl_out
    if _use_persistent_attn_res(M, H, N):
        _, output = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
            prefix_sum.reshape(M, 1, H).contiguous(),
            None,
            block_residual.reshape(K, M, 1, H).contiguous(),
            proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
            norm.weight.to(torch.bfloat16).contiguous(),
            output_norm.weight.to(torch.bfloat16).contiguous(),
            float(norm.eps),
            float(output_norm.eps),
            sm_reserve(),
        )
        _note_attn_res_fusion("attn_res+norm/persistent", True, M, H, N)
        return output.reshape(M, H)

    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
        _note_attn_res_fusion("attn_res+norm", False, M, H, N)
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    output = torch.ops.trtllm.attn_res_rmsnorm_fwd(
        layer_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        float(output_norm.eps),
    )
    _note_attn_res_fusion("attn_res+norm", True, M, H, N)
    return output.reshape(M, H)


def _apply_attn_res_add_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fuse ``prefix_sum + addend``, attention-residual, and trailing norm.

    The production residual add produces a BF16 tensor that remains live across
    the following MLP. The kernel therefore returns that materialized,
    BF16-rounded prefix sum alongside the normalized attention-residual output,
    while avoiding a separate add launch and a re-read of the intermediate by
    attention-residual selection.
    """
    if (
        prefix_sum.dtype is not torch.bfloat16
        or addend.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not addend.is_cuda
        or not block_residual.is_cuda
        or prefix_sum.shape != addend.shape
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    N = K + 1
    # Same measured window as _apply_attn_res_rmsnorm_fused above.
    pdl_out = _k3_attn_res_pdl(prefix_sum, addend, block_residual, proj, norm, output_norm)
    if pdl_out is not None:
        _note_attn_res_fusion("add+attn_res+norm/pdl", True, M, H, N)
        return pdl_out
    if _use_persistent_attn_res(M, H, N):
        updated_prefix_sum, output = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
            prefix_sum.reshape(M, 1, H).contiguous(),
            addend.reshape(M, 1, H).contiguous(),
            block_residual.reshape(K, M, 1, H).contiguous(),
            proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
            norm.weight.to(torch.bfloat16).contiguous(),
            output_norm.weight.to(torch.bfloat16).contiguous(),
            float(norm.eps),
            float(output_norm.eps),
            sm_reserve(),
        )
        _note_attn_res_fusion("add+attn_res+norm/persistent", True, M, H, N)
        return updated_prefix_sum.reshape(M, H), output.reshape(M, H)

    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
        _note_attn_res_fusion("add+attn_res+norm", False, M, H, N)
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    addend_kernel = addend.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    updated_prefix_sum, output = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
        layer_kernel,
        addend_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        float(output_norm.eps),
    )
    _note_attn_res_fusion("add+attn_res+norm", True, M, H, N)
    return updated_prefix_sum.reshape(M, H), output.reshape(M, H)


def _apply_attn_res(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: RMSNormWeight
) -> torch.Tensor:
    """Exact port of HF ``modeling_kimi._apply_attn_res`` (fp32 math).

    prefix_sum:     ``[num_tokens, hidden_size]``
    block_residual: ``[num_snapshots, num_tokens, hidden_size]``

    Inputs fitting the fused kernel's contract dispatch directly to the in-tree
    ``trtllm::attn_res_fwd`` op. Only the fallback boundary restores the HF
    ``[M, K, H]`` layout.
    """
    fused = _apply_attn_res_fused(prefix_sum, block_residual, proj, norm)
    if fused is not None:
        return fused
    # The fp32 reference below is orders of magnitude slower at prefill
    # sizes; say so once per shape rather than fall back silently.
    _note_attn_res_fusion(
        "attn_res",
        False,
        prefix_sum.shape[0],
        prefix_sum.shape[1],
        int(block_residual.shape[0]) + 1,
    )
    block_residual_hf = block_residual.transpose(0, 1)
    v = torch.cat((block_residual_hf, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.eps)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


def _apply_attn_res_and_rmsnorm(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> torch.Tensor:
    """Apply attention-residual selection and the next RMSNorm."""
    fused = _apply_attn_res_rmsnorm_fused(prefix_sum, block_residual, proj, norm, output_norm)
    if fused is not None:
        return fused
    mixed = _apply_attn_res(prefix_sum, block_residual, proj, norm)
    return rms_norm(mixed, output_norm.weight, output_norm.eps)


def _apply_attn_res_add_and_rmsnorm(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNormWeight,
    output_norm: RMSNormWeight,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add an attention output to the running residual, then select and norm.

    ``addend`` may be a ``LamportPending``: the attention block's pushed all-reduce, which the PDL
    attn_res kernel finishes where it serves the mix and the finish kernel turns into a tensor
    otherwise."""
    if isinstance(addend, LamportPending):
        fused = _k3_attn_res_pdl_lamport(
            prefix_sum, addend, block_residual, proj, norm, output_norm
        )
        if fused is not None:
            return fused
        addend = addend.materialize()
    fused = _apply_attn_res_add_rmsnorm_fused(
        prefix_sum, addend, block_residual, proj, norm, output_norm
    )
    if fused is not None:
        return fused
    updated_prefix_sum = prefix_sum + addend
    return updated_prefix_sum, _apply_attn_res_and_rmsnorm(
        updated_prefix_sum, block_residual, proj, norm, output_norm
    )


# Prefill chunk of the kda_prefill entry.
_PREFILL_CHUNK = 64
# Single-token decode: [f_a | b] -> f_b runs on the projection aux stream next to the QKVG GEMM up
# to this many rows.
_KDA_BFA_MULTISTREAM_MAX_ROWS = 128
# Decode verify without the forget-gate round trip. Up to _KDA_FAB_FUSED_MAX_ROWS rows,
# k3_kda_fab runs next to the QKVG GEMM on the aux stream: in f_a mode (BF16 [f_a | b] only,
# the verify kernel computes the forget gate from f_a itself) when that kernel runs in one wave
# or the QKVG GEMM has left the low-M kernel (more rows than _KDA_FUSED_FB_LOWM_ROWS), else as
# the full fab kernel (beta and the forget gate from one fused [f_a | b] -> f_b kernel). The f_b
# MMA costs every wave of the verify kernel, so multi-wave low-M batches keep the full fab kernel.
# Larger row counts run the [f_a | b] -> f_b chain on the aux stream as well. The fab kernels take
# at most 16 heads (the Helix head shards); other shapes keep the [f_a | b] -> f_b chain.
_KDA_FAB_FUSED_MAX_ROWS = 256
_KDA_FUSED_FB_LOWM_ROWS = 224
# Decode verify: once this layer's QKVG projection and
# [f_a | b] kernel are done, the projection aux stream prefetches the next KDA layer's QKVG,
# [f_a | b], f_b and o_proj weights into L2 (k3_l2_prefetch) while this layer's recurrence and
# all-reduce leave HBM idle. Hints only: no result depends on them. Up to
# _KDA_L2_PREFETCH_MAX_ROWS verify rows: the prefetched lines stay in L2 through the MoE block,
# whose weight streams then run about 1 % slower. The prefetch kernel stays on the aux stream,
# which the MoE block of the same layer reuses for its shared experts and joins back; no
# main-stream kernel waits for it.
_KDA_L2_PREFETCH_MAX_ROWS = 16
# KDA mixers of one model by layer index (keyed by the model's text config), for the prefetch of
# the next KDA layer's weights.
_KDA_LAYERS: Dict[int, "weakref.WeakValueDictionary[int, K3KDALinearAttention]"] = {}

_FP8_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
_QKVG_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "g_proj")


def _use_fused_fab(
    x: torch.Tensor, bfa_weight: torch.Tensor, fb_weight: torch.Tensor, num_heads: int
) -> bool:
    """Whether the ``k3_kda_fab_proj`` entry accepts these operands."""
    if not x.is_contiguous():
        return False
    from ..kernels.k3r_kda import k3_kda_fab

    return k3_kda_fab.can_use_k3_kda_fab(
        x.reshape(-1, x.shape[-1]), bfa_weight, fb_weight, num_heads
    )


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _verify_kernel_single_wave(num_heads: int, num_requests: int, device: torch.device) -> bool:
    """Whether the v2 verify kernel's grid (``kda_mtp_verify_ctas``) fits in one wave of one CTA
    per SM."""
    ctas = kda_mtp_verify_ctas(num_heads, num_requests, device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return ctas <= _num_sms(index)


def _use_fab_fa(
    x: torch.Tensor, bfa_weight: torch.Tensor, fb_weight: torch.Tensor, num_heads: int
) -> bool:
    """Whether the verify gate comes from the ``k3_kda_fab_fa`` entry + the in-kernel f_b.

    ``x`` is the ``[requests, steps, hidden]`` verify input."""
    if not x.is_contiguous():
        return False
    num_rows = x.shape[0] * x.shape[1]
    if num_rows <= _KDA_FUSED_FB_LOWM_ROWS and not _verify_kernel_single_wave(
        num_heads, x.shape[0], x.device
    ):
        return False
    if (
        fb_weight.dtype != torch.bfloat16
        or not fb_weight.is_contiguous()
        or tuple(fb_weight.shape) != (num_heads * 128, 128)
        or fb_weight.data_ptr() % 16
    ):
        return False
    from ..kernels.k3r_kda import k3_kda_fab

    return k3_kda_fab.can_use_k3_kda_fab_fa(
        x.reshape(-1, x.shape[-1]), bfa_weight, num_heads, fb_weight
    )


def _o_proj_takes_mxfp8(o_proj: nn.Module, num_rows: int) -> bool:
    """Whether ``o_proj`` takes ``num_rows`` rows as the Rubin MXFP8 operand of the verify
    kernel's epilogue (``forward_prequantized``) rather than BF16 (the K3 FP8 weight-read
    projection decides per row count, see ``takes_prequantized_input``)."""
    takes = getattr(o_proj, "takes_prequantized_input", None)
    if takes is not None:
        return bool(takes(num_rows))
    return bool(getattr(o_proj, "supports_prequantized_input", False))


class _KdaPrequantizedCore(NamedTuple):
    """MXFP8 KDA core emitted by the fused verify epilogue.

    ``activation`` is the ``[tokens, H * V]`` E4M3 post-o_norm core and
    ``scale`` its UE8M0 1x128 scales in the flat uint8 R128c4 layout that
    ``o_proj.forward_prequantized`` consumes.
    """

    activation: torch.Tensor
    scale: torch.Tensor


@dataclass
class _KDAPrefillState:
    """The fp32 state rows the ``kda_prefill`` entry updates in place: the pool itself
    (``kernel_pool`` / ``kernel_indices``), or ``staged_state``, a dense fp32 copy of the
    addressed rows (plus the snapshot rows of a plan) that is written back afterwards."""

    kernel_pool: torch.Tensor
    kernel_indices: torch.Tensor
    staged_state: torch.Tensor | None
    # With a mid-chunk snapshot plan: the snapshot rows of ``kernel_pool`` (the pool's own slots,
    # or the rows after the sequences' in the staged copy).
    snapshot_rows: torch.Tensor | None = None


def _stage_state_rows(ssm_pool: torch.Tensor, slot_indices: torch.Tensor) -> torch.Tensor:
    """Dense fp32 copy of the addressed recurrent-state rows (``_writeback_state_rows`` rounds it
    back into the pool)."""
    return ssm_pool.index_select(0, slot_indices.long()).float()


def _writeback_state_rows(
    ssm_pool: torch.Tensor, slot_indices: torch.Tensor, rows: torch.Tensor
) -> None:
    """Scatter dense state rows back into the pool, rounded to the pool dtype."""
    ssm_pool.index_copy_(0, slot_indices.long(), rows.to(ssm_pool.dtype))


def _stage_state_rows_with_snapshot_rows(
    ssm_pool: torch.Tensor, slot_indices: torch.Tensor, num_snapshots: int
) -> torch.Tensor:
    """``_stage_state_rows`` plus ``num_snapshots`` uninitialized rows after them.

    The prefill kernel writes mid-chunk snapshots into the extra rows.
    """
    num_rows = slot_indices.shape[0]
    staged = torch.empty(
        (num_rows + num_snapshots, *ssm_pool.shape[1:]),
        dtype=torch.float32,
        device=ssm_pool.device,
    )
    staged[:num_rows].copy_(ssm_pool.index_select(0, slot_indices.long()))
    return staged


def _snapshot_conv_windows(
    conv_input: torch.Tensor,
    conv_pool: torch.Tensor,
    snapshot: "KDASnapshotPlan",
) -> torch.Tensor:
    """The raw convolution window ending before each snapshot point, ``[n, 3D, W - 1]`` in
    ``conv_pool``'s dtype.

    ``conv_input`` is the pre-convolution projection, ``[3D, T]``; ``conv_pool`` is
    ``[slots, 3D, W - 1]``. A
    snapshot point is at least one KDA chunk into its sequence's step, so the
    window never reaches the sequence's previous convolution state.
    """
    window = conv_pool.shape[-1]
    offsets = torch.arange(-window, 0, device=conv_input.device)
    token_indices = (snapshot.token_indices[:, None] + offsets).reshape(-1)
    columns = conv_input.index_select(1, token_indices)
    windows = columns.view(-1, snapshot.num_snapshots, window).transpose(0, 1)
    return windows.to(conv_pool.dtype)


def _swap_inforward_conv_windows(
    conv_pool: torch.Tensor,
    stage_conv: torch.Tensor,
    snapshot: "KDASnapshotPlan",
    windows: torch.Tensor,
) -> None:
    """Move each snapshotted sequence's chunk-end convolution window from its live slot
    (``snapshot.slots``) to its staging row and put the window ending at the snapshot point
    (``windows``, :func:`_snapshot_conv_windows`) in the live slot."""
    stage_conv.index_copy_(0, snapshot.stage_rows, conv_pool.index_select(0, snapshot.slots))
    conv_pool.index_copy_(0, snapshot.slots, windows)


def _merge_row_weights(modules: Tuple[nn.Module, ...], pad_rows_to: int = 1) -> torch.Tensor:
    """Concatenate the ``weight`` rows of ``modules`` (zero rows up to a multiple of
    ``pad_rows_to``) and repoint each module's ``weight`` to its row view of the result."""
    weights = [module.weight.data for module in modules]
    padding = (-sum(weight.shape[0] for weight in weights)) % pad_rows_to
    if padding:
        weights.append(weights[0].new_zeros((padding, weights[0].shape[1])))
    fused = torch.cat(weights, dim=0).contiguous()
    offset = 0
    for module in modules:
        rows = module.weight.shape[0]
        module.weight.data = fused[offset : offset + rows]
        offset += rows
    return fused


def _extract_kda_extra_attrs(layer_idx: str):
    """Resolve the live attention metadata and KDA module for ``layer_idx``."""
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"

    metadata_ref = extra_attrs.get("attention_metadata")
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, AttentionMetadata), "Metadata must be AttentionMetadata"

    kda_layers = extra_attrs.get("kda_layers")
    assert kda_layers is not None, "KDA layer registry is not set"
    kda_layer_ref = kda_layers.get(layer_idx)
    assert kda_layer_ref is not None, f"Cannot find KDA layer for layer {layer_idx}"
    kda_layer = kda_layer_ref()
    assert isinstance(kda_layer, K3KDALinearAttention), "KDA layer must be K3KDALinearAttention"
    return metadata, kda_layer


def kda_core_inplace(hidden_states: torch.Tensor, layer_idx: str, output: torch.Tensor) -> None:
    """Run the metadata-dependent KDA core and write it into ``output``."""
    metadata, kda_layer = _extract_kda_extra_attrs(layer_idx)
    kda_layer._forward_impl(hidden_states, metadata, output=output)


maybe_bcg_kda_core_inplace = eager_on_graph(kda_core_inplace)


def _check_kda_shape(lin: dict) -> None:
    """The KDA kernels take Kimi K3's shape only."""
    gate_lower_bound = lin.get("gate_lower_bound", None)
    problems = []
    if lin["head_dim"] != 128:
        problems.append(f"head_dim {lin['head_dim']} (the kernels take 128)")
    if lin["short_conv_kernel_size"] != 4:
        problems.append(f"short_conv_kernel_size {lin['short_conv_kernel_size']} (4)")
    if not lin.get("use_full_rank_gate", True):
        problems.append("a low-rank output gate (the full-rank g_proj)")
    if gate_lower_bound is None or not (math.isfinite(gate_lower_bound) and gate_lower_bound < 0):
        problems.append(f"gate_lower_bound {gate_lower_bound} (a finite negative bound)")
    if problems:
        raise ValueError("Kimi K3 KDA: unsupported linear_attn_config: " + "; ".join(problems))


class K3KDALinearAttention(nn.Module):
    """One Kimi K3 KDA layer (head-TP shard) on K3-owned weights."""

    def __init__(
        self,
        cfg,
        layer_idx: int,
        mapping=None,
        allreduce_strategy=AllReduceStrategy.AUTO,
        aux_stream: Optional[torch.cuda.Stream] = None,
        model_config: Optional[ModelConfig] = None,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()
        lin = cfg.linear_attn_config
        _check_kda_shape(lin)
        self.hidden_size = cfg.hidden_size
        self.layer_idx = layer_idx
        self.head_dim = lin["head_dim"]
        self.head_k_dim = self.head_dim
        self.conv_size = lin["short_conv_kernel_size"]
        self.gate_lower_bound = lin["gate_lower_bound"]
        self.rms_norm_eps = cfg.rms_norm_eps

        if mapping is not None and mapping.tp_size > 1:
            self._kda_tp_size = mapping.tp_size
        else:
            self._kda_tp_size = 1
        self._kda_tp_rank = mapping.tp_rank if self._kda_tp_size > 1 else 0
        # reduce_output=False returns the row-sharded o_proj partials; the
        # caller owns the reduction (helix_a2a reduce-scatters them).
        self._o_allreduce = (
            TpAllReduce(mapping, torch.bfloat16, allreduce_strategy)
            if self._kda_tp_size > 1 and reduce_output
            else None
        )
        # Set by the decoder layer for one forward: a pushed o_proj all-reduce may be returned
        # unfinished, as a LamportPending that A_mid finishes.
        self.ar_finish_deferred = False

        num_heads = lin["num_heads"]
        if num_heads % self._kda_tp_size != 0:
            raise ValueError(
                f"KDA num_heads {num_heads} not divisible by tp_size {self._kda_tp_size}"
            )
        self.num_heads = num_heads // self._kda_tp_size
        projection_size = self.num_heads * self.head_dim
        self.proj_size = projection_size

        projection_quant_configs = (model_config.quant_config_dict or {}) if model_config else {}
        default_quant_config = (
            model_config.quant_config if model_config else None
        ) or QuantConfig()
        declared = {
            name: projection_quant_configs.get(name, default_quant_config).quant_algo
            for name in _FP8_PROJECTIONS
        }
        if any(algo != QuantAlgo.FP8_BLOCK_SCALES for algo in declared.values()):
            raise ValueError(
                "Kimi K3 KDA takes FP8 128 x 128 block-scale q/k/v/g/o projections (the "
                f"checkpoint's FP8_PB_WO); declared {declared}"
            )
        if self.hidden_size % 128 or projection_size % 128:
            raise ValueError("KDA FP8 projection shards must be aligned to 128x128 blocks")

        bf16 = torch.bfloat16
        # This rank's shard (the loader slices the checkpoint by head), so the containers hold
        # rank-local dimensions.
        self.q_proj = Fp8BlockScaleLinear(self.hidden_size, projection_size)
        self.k_proj = Fp8BlockScaleLinear(self.hidden_size, projection_size)
        self.v_proj = Fp8BlockScaleLinear(self.hidden_size, projection_size)
        self.q_conv1d = ShortConvWeight(projection_size, self.conv_size, dtype=bf16)
        self.k_conv1d = ShortConvWeight(projection_size, self.conv_size, dtype=bf16)
        self.v_conv1d = ShortConvWeight(projection_size, self.conv_size, dtype=bf16)
        # Initialized (checkpoint loading overwrites them) with single random-init ops, which
        # MetaInitMode allows: uninitialized NaN bit patterns would poison randomly built layers.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=bf16).uniform_(1, 16))
        )
        self.f_a_proj = Bf16Linear(self.hidden_size, self.head_dim, dtype=bf16)
        self.f_b_proj = Bf16Linear(self.head_dim, projection_size, dtype=bf16)
        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, dtype=bf16).uniform_(math.log(1e-3), math.log(1e-1))
        )
        self.b_proj = Bf16Linear(self.hidden_size, self.num_heads, dtype=bf16)
        self.g_proj = Fp8BlockScaleLinear(self.hidden_size, projection_size)
        self.o_norm = RMSNormWeight(self.head_dim, self.rms_norm_eps, dtype=bf16)
        self.o_proj = Fp8BlockScaleLinear(projection_size, self.hidden_size)
        # Built from q/k/v/g by finalize_decode_weights: the fused [q | k | v | g] projection and
        # its output sections.
        self.qkvg_proj: Optional[nn.Module] = None
        self.qkvg_split_sizes: Optional[List[int]] = None

        self._dispatch = K3KDAKernelDispatch()

        # Built after checkpoint load (finalize_decode_weights): the fused BF16 [f_a | b] weight
        # (rows padded to 8) and the kernel-layout constants.
        self._bfa_proj_weight: Optional[torch.Tensor] = None
        self._w_q_t = self._w_k_t = self._w_v_t = None
        self._A_log_f32 = self._dt_bias_f32 = self._onorm_w_f32 = None
        # Fork/join state for overlapping the small [f_a | b] -> f_b chain
        # with the wide QKVG projection during CUDA-graph execution.
        self._projection_aux_stream = aux_stream
        self._projection_fork_event = torch.cuda.Event()
        self._projection_join_event = torch.cuda.Event()
        # Output buffer of the in-place decode kernel outside breakable CUDA graphs.
        self._o_dense: Optional[torch.Tensor] = None
        # [3D, W] BF16 conv weight of causal_conv1d_fwd and the fp32 [D, W] weights of the
        # verify kernels.
        self._packed_conv_weight: Optional[torch.Tensor] = None
        self._mtp_conv_weights: Optional[Tuple[torch.Tensor, ...]] = None
        # L2 prefetch of the next KDA layer's projection weights (decode verify).
        self._kda_group = _KDA_LAYERS.setdefault(id(cfg), weakref.WeakValueDictionary())
        self._kda_group[layer_idx] = self
        self._l2_prefetch_ranges_cache: Optional[List[Tuple[int, int]]] = None
        self._l2_prefetch_start = torch.cuda.Event()
        # Set by the decoder layer for the next forward only (generation-only batches): the
        # MXFP8 form ``(xq, xs)`` of the generation rows, which the fused QKVG projection of the
        # verify path consumes instead of quantizing the BF16 rows (see qkvg_takes_mxfp8).
        self.next_input_mxfp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self.register_to_config = False
        self.layer_idx_str = str(layer_idx)
        if model_config is not None:
            kda_layers = model_config.extra_attrs.setdefault("kda_layers", {})
            suffix = 0
            while self.layer_idx_str in kda_layers:
                self.layer_idx_str = f"{layer_idx}_{suffix}"
                suffix += 1
            kda_layers[self.layer_idx_str] = weakref.ref(self)
            self.register_to_config = True

    # ------------------------------------------------------------------
    # Post-load weight preparation.
    # ------------------------------------------------------------------

    def finalize_decode_weights(self) -> None:
        """Build the fused projection weights and kernel constants after weight load.

        1. ``qkvg_proj``: one FP8 ``[q | k | v | g]`` projection from the checkpoint codes and
           scales (the swap to the FP8 weight-read GEMM transforms its scales afterwards); the
           four sources become row views of it. ``_bfa_proj_weight``: BF16 ``[f_a | b]`` with
           rows padded to 8; ``f_a_proj`` / ``b_proj`` become row views of it.
        2. Kernel-layout constants: transposed BF16 conv weights ``[W, D]``, fp32 ``A_log`` /
           ``dt_bias`` / ``o_norm.weight`` and the packed / fp32 conv weights of the prefill and
           verify kernels.
        """
        if self.q_proj.weight.device.type != "cuda":
            return
        with torch.no_grad():
            sources = tuple(getattr(self, name) for name in _QKVG_PROJECTIONS)
            if all(isinstance(module, Fp8BlockScaleLinear) for module in sources):
                self.qkvg_proj = self._fuse_checkpoint_projections(sources)
                self.qkvg_split_sizes = [module.out_features for module in sources]
                # Without it the four sources keep their own storage next to the fused copy.
                self._alias_projections_into_fused(sources, self.qkvg_proj)
            self._bfa_proj_weight = _merge_row_weights((self.f_a_proj, self.b_proj), pad_rows_to=8)
            self._build_decode_kernel_constants()

    @staticmethod
    def _fuse_checkpoint_projections(
        modules: Tuple[Fp8BlockScaleLinear, ...],
    ) -> Fp8BlockScaleLinear:
        """One FP8 projection over the rows of ``modules`` (block-aligned): their checkpoint codes
        and FP32 scale grids, copied before any scale preparation."""
        if any(module.out_features % 128 for module in modules[:-1]):
            raise ValueError("KDA FP8 fusion boundaries must be aligned to 128 rows")
        first = modules[0]
        fused = Fp8BlockScaleLinear(
            first.in_features,
            sum(module.out_features for module in modules),
            device=first.weight.device,
        )
        offset = 0
        scale_offset = 0
        for module in modules:
            rows = module.weight.shape[0]
            fused.weight.data[offset : offset + rows].copy_(module.weight.data)
            offset += rows
            scale_rows = module.weight_scale.shape[0]
            fused.weight_scale.data[scale_offset : scale_offset + scale_rows].copy_(
                module.weight_scale.data
            )
            scale_offset += scale_rows
        fused.input_scale.data = first.input_scale.data
        fused.inv_input_scale.data = first.inv_input_scale.data
        return fused

    @staticmethod
    def _alias_projections_into_fused(modules: Tuple[nn.Module, ...], fused: nn.Module) -> None:
        """Repoint the FP8 source projections at row views of the fused codes and scale grid."""
        offset = 0
        scale_offset = 0
        for module in modules:
            rows = module.weight.shape[0]
            module.weight.data = fused.weight.data[offset : offset + rows]
            offset += rows
            scale_rows = module.weight_scale.shape[0]
            module.weight_scale.data = fused.weight_scale.data[
                scale_offset : scale_offset + scale_rows
            ]
            scale_offset += scale_rows

    def _build_decode_kernel_constants(self) -> None:
        """Kernel-layout constants of the decode / verify / prefill kernels."""
        self._w_q_t = (
            self.q_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        self._w_k_t = (
            self.k_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        self._w_v_t = (
            self.v_conv1d.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
        )
        self._A_log_f32 = self.A_log.detach().float().contiguous()
        self._dt_bias_f32 = self.dt_bias.detach().float().contiguous()
        self._onorm_w_f32 = self.o_norm.weight.detach().float().contiguous()
        # Built eagerly too, so the first verify call never allocates (a capture-unsafe lazy
        # allocation).
        self._build_mtp_conv_weights()

    def _build_mtp_conv_weights(self) -> None:
        """Prebuild the packed-prefill and fused-verify convolution weights.

        Building them lazily at first use would allocate at runtime; under
        CUDA graph capture that would bake capture-pool pointers into the
        cached tensors.
        """
        conv_weights = tuple(
            conv.weight.detach().squeeze(1)
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        )
        self._packed_conv_weight = torch.cat(conv_weights, dim=0).to(torch.bfloat16).contiguous()
        self._mtp_conv_weights = tuple(weight.float().contiguous() for weight in conv_weights)

    def _require_finalized(self) -> None:
        if self.qkvg_proj is None or self._bfa_proj_weight is None or self._onorm_w_f32 is None:
            raise RuntimeError(
                "Kimi K3 KDA runs on the fused projections and kernel constants that "
                "finalize_decode_weights() builds after weight load"
            )

    # ------------------------------------------------------------------
    # MXFP8 hand-off of the decoder layer (generation-only batches).
    # ------------------------------------------------------------------

    def qkvg_takes_mxfp8(self, num_rows: int) -> bool:
        """Whether the fused FP8 QKVG projection of ``num_rows`` rows can take their MXFP8 form
        (``next_input_mxfp8``) instead of the BF16 rows: it runs the low-M kernel on them (or the
        staged kernel above the low-M cut-over, :meth:`qkvg_stages_weights`)."""
        takes = getattr(self.qkvg_proj, "takes_mxfp8_input", None)
        if takes is None:
            return False
        return takes(num_rows) or self.qkvg_stages_weights(num_rows) or self.qkvg_mx_atom(num_rows)

    def qkvg_mx_atom(self, num_rows: int) -> bool:
        """Whether the fused QKVG projection takes the MXFP8 rows with the scales in the mid-M
        GEMM's layout (``takes_mid_mxfp8_input``)."""
        takes_mid = getattr(self.qkvg_proj, "takes_mid_mxfp8_input", None)
        return takes_mid is not None and bool(takes_mid(num_rows))

    def qkvg_stages_weights(self, num_rows: int) -> bool:
        """Whether the fused QKVG projection of ``num_rows`` MXFP8 rows stages its weights on chip
        before its ``griddepcontrol.wait``: then the attn_res in front of it may release it at
        kernel entry.

        On the MXFP8 hand-off (``next_input_mxfp8``) of a decode verify the fused QKVG low-M GEMM
        keeps its whole weight slice on chip before its wait (k3_fp8w_lowm_mxfp8.staged_mx_tactic;
        the same bits), also for 33 to 64 rows, and the pre-attention attn_res in front of it
        releases it at kernel entry, so its CTAs stream the QKVG weights while the attn_res and the
        latent-up GEMM before it still run. The L2 prefetch of the next KDA layer's weights
        (:meth:`_issue_l2_prefetch`) is skipped where the next layer stages them."""
        takes = getattr(self.qkvg_proj, "takes_staged_mxfp8_input", None)
        return takes is not None and takes(num_rows)

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """Select the eager path or breakable-CUDA-graph path.

        Returns the o_proj output (all-reduced under head TP), or with the Lamport push
        (``_project_output``) a ``LamportPending`` for the decoder layer's
        A_mid.
        """
        if self.register_to_config and is_in_breakable_cuda_graph():
            core = hidden_states.new_empty(
                (hidden_states.shape[0], self.num_heads, self.head_dim),
                dtype=torch.bfloat16,
            )
            maybe_bcg_kda_core_inplace(hidden_states, self.layer_idx_str, core)
            self.next_input_mxfp8 = None
            return self._project_output(core)

        core = self._forward_impl(hidden_states, attn_metadata)
        # The MXFP8 rows of the decoder layer serve this forward only (consumed by the verify
        # projection, if it ran).
        self.next_input_mxfp8 = None
        # The push serves the one o_proj call of a generation-only batch.
        return self._project_output(core, allow_ar_push=attn_metadata.num_contexts == 0)

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, _KdaPrequantizedCore]:
        """Split the batch into its prefill, decode and verify rows and run their chains.

        ``output`` is the BCG post-o_norm, pre-o_proj core buffer. When it
        is supplied, subpaths fill and return its corresponding slices;
        otherwise this method returns an allocated core tensor. A
        verify-only batch without ``output`` may instead return the fused
        verify epilogue's MXFP8 core (see ``forward_verify_fused``).
        """
        mamba_metadata = attn_metadata.mamba_metadata
        num_prefills = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens
        batch_size = attn_metadata.seq_lens.shape[0]
        state_indices = mamba_metadata.state_indices[:batch_size]
        cu_seqlens = mamba_metadata.query_start_loc_long[: num_prefills + 1]
        num_decodes = batch_size - num_prefills

        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        conv_pool = layer_cache.conv  # [slots, 3D, W - 1] bf16
        ssm_pool = layer_cache.temporal  # [slots, H, V, K] fp32 or bf16
        generation_state_indices = getattr(mamba_metadata, "generation_state_indices", None)
        if generation_state_indices is None:
            generation_state_indices = state_indices[num_prefills:]

        cores: List[torch.Tensor] = []
        if num_prefills > 0:
            snapshot = getattr(mamba_metadata, "kda_snapshot", None)
            inforward_stage = None
            if snapshot is not None and snapshot.stage_rows is not None:
                inforward_stage = attn_metadata.kv_cache_manager.inforward_snapshot_stage_views(
                    self.layer_idx
                )
            prefill_core = self.forward_prefill(
                hidden_states[:num_ctx_tokens],
                cu_seqlens,
                mamba_metadata,
                num_prefills,
                conv_pool,
                ssm_pool,
                state_indices[:num_prefills],
                layer_cache,
                output=output[:num_ctx_tokens] if output is not None else None,
                inforward_stage=inforward_stage,
            )
            if output is None:
                cores.append(prefill_core)
        if num_decodes > 0:
            decode_rows = num_tokens - num_ctx_tokens
            if decode_rows == num_decodes:
                decode_core = self.forward_decode(
                    hidden_states[num_ctx_tokens:num_tokens],
                    conv_pool,
                    ssm_pool,
                    generation_state_indices,
                    layer_cache,
                    output=output[num_ctx_tokens:num_tokens] if output is not None else None,
                )
                if output is None:
                    cores.append(decode_core)
            else:
                # Speculative verification: each generation request carries
                # 1 + draft_len tokens (drafts are padded to the static max,
                # so T is uniform).
                assert decode_rows % num_decodes == 0, (
                    f"ragged generation batch: {decode_rows} tokens for {num_decodes} requests"
                )
                verify_core = self.forward_verify(
                    hidden_states[num_ctx_tokens:num_tokens],
                    decode_rows // num_decodes,
                    layer_cache,
                    ssm_pool,
                    generation_state_indices,
                    output=(output[num_ctx_tokens:num_tokens] if output is not None else None),
                    # The MXFP8 core cannot be concatenated with a prefill
                    # core or stored into the BCG buffer.
                    allow_prequantized_output=output is None and num_prefills == 0,
                    context_only=getattr(attn_metadata.kv_cache_manager, "kda_context_only", False)
                    is True,
                )
                if output is None:
                    cores.append(verify_core)
        if output is not None:
            return output
        return cores[0] if len(cores) == 1 else torch.cat(cores, dim=0)

    def _project_output(
        self,
        core: Union[torch.Tensor, _KdaPrequantizedCore],
        allow_ar_push: bool = False,
    ) -> Union[torch.Tensor, LamportPending]:
        """o_proj on the post-o_norm core, then the head-TP all-reduce of its partials.

        ``allow_ar_push`` (generation-only batches, outside breakable CUDA graphs): o_proj stores
        its partial straight into the all-reduce's Lamport buffers where the low-M GEMM serves it
        (``o_proj.next_ar_push``); the decoder layer's ``ar_finish_deferred`` then makes this
        return a ``LamportPending``.
        """
        num_rows = core.activation.shape[0] if isinstance(core, tuple) else core.shape[0]
        ar_push = None
        if allow_ar_push and self._o_allreduce is not None and hasattr(self.o_proj, "next_ar_push"):
            ar_push = lamport_push_target(
                self._o_allreduce,
                num_rows,
                self.hidden_size,
                deferred=self.ar_finish_deferred,
            )
            self.o_proj.next_ar_push = ar_push

        if isinstance(core, _KdaPrequantizedCore):
            out = self.o_proj.forward_prequantized(core.activation, core.scale)
        else:
            out = self.o_proj(core.reshape(-1, self.proj_size))

        if ar_push is not None:
            self.o_proj.next_ar_push = None
        if ar_push is not None and ar_push.pushed:
            # The decoder layer sets ar_finish_deferred where A_mid finishes the all-reduce: then
            # this returns a LamportPending.
            out_shape = out.shape
            out = finish_lamport_push(
                self._o_allreduce,
                ar_push,
                deferred=self.ar_finish_deferred,
            )
            if not isinstance(out, LamportPending):
                out = out.view(out_shape)
        elif ar_push is not None and ar_push.deferred:
            # o_proj ran without the push epilogue (e.g. the MXFP8 GEMM on the prequantized
            # verify output); a copy kernel pushes it and A_mid finishes.
            lamport_push_tensor(self._o_allreduce, ar_push, out.reshape(ar_push.output.shape))
            out = finish_lamport_push(self._o_allreduce, ar_push, deferred=True)
        elif self._o_allreduce is not None:
            # Head-sharded TP: every rank ran its head shard on the same
            # local batch; sum the row-sharded o_proj partials.
            out = self._o_allreduce(out)
        return out

    def _sync_kda_replay_conv_window(self, layer_cache, slot_indices, conv_pool) -> None:
        """Seed the fused verify kernel's replay conv caches from the live committed conv pool.

        Their committed window (columns ``[0, W-1)``) must hold the live pool's ``W - 1`` raw
        inputs whenever prefill or plain decode advances it. No-op without replay caches (no
        speculative decoding, or a context-only server).
        """
        if not bool(getattr(layer_cache, "has_kda_replay_caches", False)):
            return
        layer_cache.commit_conv_window(slot_indices, conv_pool)

    def _project_gate_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(beta, forget_gate)``: ``F.linear`` on the fused ``[f_a | b]`` weight, then
        ``F.linear`` of its ``f_a`` columns on ``f_b``. ``beta`` and the forget gate are the raw
        logits (the kernels apply the sigmoid / gate)."""
        hd, H = self.head_dim, self.num_heads
        bfa = F.linear(x, self._bfa_proj_weight)
        return bfa[..., hd : hd + H], F.linear(bfa[..., :hd], self.f_b_proj.weight)

    def _project_packed_conv_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(conv_input, onorm_g)`` of the prefill rows ``x`` ``[1, T, hidden]``: the QKVG GEMM,
        its ``[q | k | v]`` columns repacked channel-major ``[3D, T]``
        (``extract_transpose_prefill_slice``, for ``causal_conv1d_fwd``), and its output-gate
        columns."""
        d = self.proj_size
        qkvg = self.qkvg_proj(x)
        onorm_g = qkvg[..., 3 * d : 4 * d]
        qkvg_2d = qkvg.squeeze(0)
        return extract_transpose_prefill_slice(qkvg_2d, qkvg_2d.shape[0], 0, 3 * d), onorm_g

    def forward_prefill(
        self,
        x2d,
        cu_seqlens,
        mamba_metadata,
        num_prefills,
        conv_pool,
        ssm_pool,
        slot_indices,
        layer_cache=None,
        output: Optional[torch.Tensor] = None,
        inforward_stage: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Prefill ``num_prefills`` packed sequences and update their conv / recurrent rows.

        The convolution is ``causal_conv1d_fwd`` + ``fused_kda_post_conv``; Q/K leave it
        L2-normalized for ``kda_prefill``. A single unaligned sequence is chunk-padded by the
        op. A BF16 state pool
        (and every in-forward snapshot plan) is staged through an fp32 copy; a ``kda_snapshot``
        plan also stores mid-chunk states and convolution windows.
        """
        if x2d.shape[0] == 0:
            reset_recurrent_state_rows(
                ssm_pool, slot_indices, mamba_metadata.has_initial_states[:num_prefills], conv_pool
            )
            self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)
            return self._store_core(x2d.new_empty((0, self.num_heads, self.head_dim)), output)
        self._require_finalized()
        x = x2d.unsqueeze(0)  # [1, T, hidden]
        # Metadata owns stable int32 indices; never convert per layer.
        slot_indices = mamba_metadata.state_indices[:num_prefills]

        # Mid-chunk snapshots: the recurrent state and convolution window of a sequence after its
        # first tokens of this step. The prefill op writes the state (snapshot_rows /
        # snapshot_chunks). With ``inforward_stage`` (the in-forward snapshots of the cache
        # manager, ``snapshot.stage_rows`` set) the live slot keeps the state at the snapshot
        # point and the staging row receives the chunk-end state; otherwise the snapshot goes to
        # a pool slot of its own.
        snapshot = getattr(mamba_metadata, "kda_snapshot", None)
        if (snapshot is not None and snapshot.stage_rows is not None) != (
            inforward_stage is not None
        ):
            raise RuntimeError(
                "Kimi K3 KDA in-forward snapshots need both the plan's staging rows and the "
                "cache manager's staging views"
            )

        packed_conv, onorm_g = self._project_packed_conv_input(x)
        snapshot_windows = None
        if snapshot is not None:
            # The window is read before the convolution below, which may overwrite its input in
            # place. A live slot's window still feeds that convolution; it is swapped after it.
            snapshot_windows = _snapshot_conv_windows(packed_conv, conv_pool, snapshot)
            if inforward_stage is None:
                conv_pool.index_copy_(0, snapshot.slots, snapshot_windows)

        # Initial states: present for continuation chunks (chunked prefill)
        # and for prefix-cache hits (block reuse), where the previous
        # conv/recurrent state was onboarded into this request's slot.
        has_init = mamba_metadata.has_initial_states[:num_prefills]
        prepared_state = self._prepare_prefill_state(
            cu_seqlens,
            mamba_metadata,
            num_prefills,
            ssm_pool,
            slot_indices,
            snapshot=snapshot,
        )
        # The convolution input is a contiguous channel-major [3D, T] copy: the op writes its
        # output over it.
        causal_conv1d_fwd(
            packed_conv,
            self._packed_conv_weight,
            None,
            conv_pool,
            mamba_metadata.query_start_loc[: num_prefills + 1],
            mamba_metadata.state_indices[:num_prefills],
            has_init,
            True,
            PAD_SLOT_ID,
        )

        beta, g = self._project_gate_inputs(x)
        g = g.unflatten(-1, (self.num_heads, self.head_dim))

        # Imported here: the entry imports this package's kernels, and the package imports this
        # module.
        from ..ops.ssm.fused_kda_post_conv import fused_kda_post_conv

        beta = beta.float()
        q, k, v = fused_kda_post_conv(
            packed_conv,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        o = self._prefill_recurrence(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            mamba_metadata,
            num_prefills,
            ssm_pool,
            slot_indices,
            prepared_state,
            snapshot=snapshot,
            inforward_stage_ssm=None if inforward_stage is None else inforward_stage[0],
        )
        # Fused-verify replay caches: seed the committed conv window so the
        # first verify round convolves the correct history (pending drafts
        # are zero for a fresh request, so the tail columns are unused).
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)
        if inforward_stage is not None:
            _swap_inforward_conv_windows(conv_pool, inforward_stage[1], snapshot, snapshot_windows)

        core = self._output_gate(o, onorm_g)
        return self._store_core(core, output)

    def _prepare_prefill_state(
        self,
        cu_seqlens: torch.Tensor,
        mamba_metadata,
        num_prefills: int,
        ssm_pool: torch.Tensor,
        slot_indices: torch.Tensor,
        snapshot: Optional["KDASnapshotPlan"] = None,
    ) -> _KDAPrefillState:
        """The state rows the prefill op updates in place, zeroed where a sequence starts fresh.

        The op takes fp32 V-first pool rows. A BF16 pool is staged through a dense fp32 copy of
        the addressed rows (plus the snapshot rows of a ``snapshot`` plan), as is an fp32 pool
        under an in-forward snapshot plan (whose snapshot rows are the live slots, which the
        kernel's chunk-end write would overwrite); :meth:`_prefill_recurrence` writes the copy
        back.
        """
        has_init = mamba_metadata.has_initial_states[:num_prefills]
        staged_state = None
        kernel_pool, kernel_indices = ssm_pool, slot_indices
        snapshot_rows = None if snapshot is None else snapshot.pool_rows
        inforward = snapshot is not None and snapshot.stage_rows is not None
        if ssm_pool.dtype != torch.float32 or inforward:
            if snapshot is None:
                staged_state = _stage_state_rows(ssm_pool, slot_indices)
            else:
                staged_state = _stage_state_rows_with_snapshot_rows(
                    ssm_pool, slot_indices, snapshot.num_snapshots
                )
                snapshot_rows = snapshot.staged_rows
            kernel_pool = staged_state
            kernel_indices = mamba_metadata._arange_buffer[:num_prefills]
        if not self._dispatch.can_use_indexed_prefill(
            state_pool=kernel_pool,
            state_indices=kernel_indices,
            has_initial_states=has_init,
            cu_seqlens=cu_seqlens,
            num_sequences=num_prefills,
        ):
            raise RuntimeError(
                "Kimi K3 KDA prefill: the kda_prefill entry does not take this state pool / "
                "metadata layout (fp32 [slots, H, V, K] rows, 16-byte aligned int32 / int64 "
                "indices, bool has_initial_states)"
            )
        # The convolution clears fresh convolution rows itself.
        reset_recurrent_state_rows(kernel_pool, kernel_indices, has_init)
        return _KDAPrefillState(
            kernel_pool, kernel_indices, staged_state, snapshot_rows=snapshot_rows
        )

    def _prefill_recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        mamba_metadata,
        num_prefills: int,
        ssm_pool: torch.Tensor,
        slot_indices: torch.Tensor,
        state: _KDAPrefillState,
        *,
        snapshot: Optional["KDASnapshotPlan"] = None,
        inforward_stage_ssm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``kda_prefill`` on the token-major L2-normalized Q/K, V, the raw gate and
        beta logits, updating ``state``'s fp32 V-first ``[slots, H, D, D]`` rows in place, then
        the write-back of a staged copy (and the snapshot moves)."""
        o = self._dispatch.prefill_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self._A_log_f32,
            dt_bias=self._dt_bias_f32,
            scale=self.head_k_dim**-0.5,
            lower_bound=self.gate_lower_bound,
            cu_seqlens=cu_seqlens,
            chunk_indices=getattr(mamba_metadata, "kda_chunk_indices", None),
            chunk_size=_PREFILL_CHUNK,
            state_pool=state.kernel_pool,
            state_indices=state.kernel_indices,
            varlen_is_aligned=getattr(mamba_metadata, "kda_varlen_is_aligned", None),
            single_sequence_length=getattr(mamba_metadata, "kda_single_sequence_length", None),
            snapshot_rows=state.snapshot_rows if snapshot is not None else None,
            snapshot_chunks=None if snapshot is None else snapshot.chunks,
        )

        # The convolution persisted the live convolution pool in place.
        staged_state = state.staged_state
        if staged_state is not None:
            if snapshot is None:
                _writeback_state_rows(ssm_pool, slot_indices, staged_state)
            else:
                _writeback_state_rows(ssm_pool, slot_indices, staged_state[:num_prefills])
                if snapshot.stage_rows is not None:
                    # In-forward: the chunk-end states move to the staging rows; the live
                    # slots (``snapshot.slots``) take the states at the snapshot points.
                    inforward_stage_ssm.index_copy_(
                        0, snapshot.stage_rows, ssm_pool.index_select(0, snapshot.slots)
                    )
                _writeback_state_rows(ssm_pool, snapshot.slots, staged_state[num_prefills:])
        elif snapshot is not None and snapshot.stage_rows is not None:
            raise RuntimeError("Kimi K3 KDA in-forward snapshots need a staged state copy")
        return o

    def forward_decode(
        self,
        x2d,
        conv_pool,
        ssm_pool,
        slot_indices,
        layer_cache=None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One token per request: the QKVG GEMM (next to ``[f_a | b]`` -> ``f_b`` on the
        projection aux stream up to 128 rows in CUDA graphs), then the fused decode kernel, which
        convolves, gates, runs the recurrence and the gated RMSNorm, and updates the conv and
        fp32 state pools by slot. A BF16 state pool is staged through dense copies of the
        addressed rows (fp32 state, conv window as is) and written back (state rounded).
        """
        self._require_finalized()
        d = self.proj_size
        hd = self.head_dim
        H = self.num_heads
        B = x2d.shape[0]

        staged_state = staged_conv = None
        slots_long = None
        kernel_state, kernel_state_indices, conv_src = ssm_pool, slot_indices, conv_pool
        if ssm_pool.dtype != torch.float32:
            slots_long = slot_indices.long()
            staged_state = _stage_state_rows(ssm_pool, slots_long)
            staged_conv = conv_pool.index_select(0, slots_long)
            kernel_state, kernel_state_indices, conv_src = staged_state, None, staged_conv

        # The decode kernel writes its core in place: the BCG core buffer, or outside breakable
        # CUDA graphs a persistent buffer whose pointer stays stable across graph captures.
        if output is not None:
            kda_out = output.view(B, 1, H, hd)
        else:
            if self._o_dense is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Kimi K3 KDA decode: its output buffer is allocated by the eager forward "
                        "that precedes CUDA graph capture"
                    )
                self._o_dense = torch.empty(
                    max(conv_pool.shape[0], B),
                    1,
                    H,
                    hd,
                    dtype=torch.bfloat16,
                    device=x2d.device,
                )
            else:
                assert self._o_dense.shape[0] >= B, (
                    f"KDA decode output buffer holds {self._o_dense.shape[0]} rows "
                    f"but the decode batch is {B}; reallocating would corrupt "
                    f"previously captured CUDA graphs"
                )
            kda_out = self._o_dense[:B]

        projection_aux_stream = (
            self._projection_aux_stream if B <= _KDA_BFA_MULTISTREAM_MAX_ROWS else None
        )
        qkvg, (beta, g) = fork_join(
            lambda: self.qkvg_proj(x2d),
            lambda: self._project_gate_inputs(x2d),
            self._projection_fork_event,
            self._projection_join_event,
            projection_aux_stream,
        )
        x_qkvg = qkvg[:, : 3 * d]
        onorm_g = qkvg[:, 3 * d : 4 * d]

        # Section views retain the live pool's slot stride, including V2
        # manager padding. The kernel uses ssm_state_indices for both pools.
        o = self._dispatch.decode_kda(
            x_q=x_qkvg[:, :d].unflatten(-1, (H, hd)).unsqueeze(0),
            x_k=x_qkvg[:, d : 2 * d].unflatten(-1, (H, hd)).unsqueeze(0),
            x_v=x_qkvg[:, 2 * d : 3 * d].unflatten(-1, (H, hd)).unsqueeze(0),
            w_q_t=self._w_q_t,
            w_k_t=self._w_k_t,
            w_v_t=self._w_v_t,
            bias_q=None,
            bias_k=None,
            bias_v=None,
            cs_q=conv_src[:, :d],
            cs_k=conv_src[:, d : 2 * d],
            cs_v=conv_src[:, 2 * d :],
            A_log=self._A_log_f32,
            g=g.unflatten(-1, (H, hd)).unsqueeze(0),
            dt_bias=self._dt_bias_f32,
            beta=beta.unsqueeze(0),
            state=kernel_state,
            onorm_g=onorm_g.unflatten(-1, (H, hd)).unsqueeze(0),
            onorm_weight=self._onorm_w_f32,
            out=kda_out,
            ssm_state_indices=kernel_state_indices,
            scale=hd**-0.5,
            onorm_eps=self.o_norm.eps,
            lower_bound=self.gate_lower_bound,
            use_beta_sigmoid_in_kernel=True,
            verbose=False,
            update_conv_cache=True,
        )
        if staged_state is not None:
            conv_pool.index_copy_(0, slots_long, staged_conv)
            _writeback_state_rows(ssm_pool, slots_long, staged_state)
        # Fused-verify replay caches (spec decoding only): keep the
        # committed conv window in sync with the plain-decode advance.
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)

        # The kernel applied the output gate / RMSNorm and returns the supplied buffer.
        return o.view(B, H, hd)

    def forward_verify(
        self,
        x2d,
        num_steps,
        layer_cache,
        ssm_pool,
        slot_indices,
        output: Optional[torch.Tensor] = None,
        allow_prequantized_output: bool = False,
        context_only: bool = False,
    ) -> Union[torch.Tensor, _KdaPrequantizedCore]:
        """Speculative verification: advance each request ``num_steps`` tokens (1 golden +
        ``num_steps - 1`` padded drafts) with the fused verify kernel on the cache manager's
        replay caches (:meth:`forward_verify_fused`).

        On a context-only server the rows are dummies and take
        ``forward_verify_context_only`` instead (the manager marks such a server with
        ``kda_context_only``, passed in as ``context_only``).
        """
        if bool(getattr(layer_cache, "has_kda_replay_caches", False)):
            return self.forward_verify_fused(
                x2d,
                num_steps,
                layer_cache,
                ssm_pool,
                slot_indices,
                output=output,
                allow_prequantized_output=allow_prequantized_output,
            )
        if context_only or getattr(layer_cache, "context_only", False):
            return self.forward_verify_context_only(x2d, output)
        raise RuntimeError(
            "Kimi K3 KDA verification runs the fused verify kernel on the cache manager's KDA "
            "replay caches, which this layer cache does not have"
        )

    def forward_verify_context_only(
        self, x2d: torch.Tensor, output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Verification rows on a context-only server.

        Such a server (disaggregated context server) runs no real generation
        request; its generation rows are dummies (warmup, attention-DP
        padding) whose outputs are discarded, and its cache manager keeps no
        replay caches for them. The recurrence is skipped and the live pools
        are left untouched; the output projection (``_project_output``) still
        runs on a zero attention core, so the layer's projection and collectives
        match those of a real verification step.
        """
        core = x2d.new_zeros(x2d.shape[0], self.num_heads, self.head_dim)
        return self._store_core(core, output)

    def _project_verify_inputs(self, x: torch.Tensor, num_rows: int, gate_from_fa: bool) -> tuple:
        """The verify projections: the QKVG GEMM on the calling stream, the gate projection on
        the projection aux stream.

        Returns ``(q, k, v, forget_gate, beta, onorm_g, f_a)``, the q / k / v / output-gate
        sections as strided views of the QKVG output (the verify kernel reads them in place) and
        beta as one of the ``[f_a | b]`` output. ``f_a`` is the BF16 ``f_a`` view where
        ``gate_from_fa`` and the f_a-mode fab kernel served these rows (``forget_gate`` is then
        None and the verify kernel computes it), else None.

        The QKVG GEMM takes the decoder layer's MXFP8 rows (``next_input_mxfp8``,
        ``attn_res_mxfp8``) where they were handed over, staging its weights. The gate is
        ``k3_kda_fab_fa`` (f_a mode), ``k3_kda_fab_proj`` (beta and the forget gate) where they
        accept the shapes, else ``F.linear`` ``[f_a | b]`` -> ``f_b``.
        """
        fused_qkvg = self.qkvg_proj
        mxfp8_input, self.next_input_mxfp8 = self.next_input_mxfp8, None

        def _project_qkvg() -> torch.Tensor:
            if mxfp8_input is not None:
                return fused_qkvg.forward_mxfp8(x, *mxfp8_input, stage_weights=True)
            return fused_qkvg(x)

        bfa_weight = self._bfa_proj_weight
        fb_weight = self.f_b_proj.weight
        fab_fa = gate_from_fa and _use_fab_fa(x, bfa_weight, fb_weight, self.num_heads)
        fused_fab = not fab_fa and _use_fused_fab(x, bfa_weight, fb_weight, self.num_heads)

        def _project_gates() -> tuple:
            if fab_fa:
                from ..ops.ssm.k3_kda_fab_fa import k3_kda_fab_fa

                # (beta, f_a): the verify kernel derives the forget gate from f_a.
                # W_fb is warmed in L2 for the verify kernel's per-head load.
                bfa = k3_kda_fab_fa(
                    x.reshape(-1, x.shape[-1]), bfa_weight, self.num_heads, fb_weight
                ).view(*x.shape[:-1], bfa_weight.shape[0])
                return bfa[..., self.head_dim : self.head_dim + self.num_heads], bfa[
                    ..., : self.head_dim
                ]
            if fused_fab:
                from ..ops.ssm.k3_kda_fab_proj import k3_kda_fab_proj

                beta, forget_gate = k3_kda_fab_proj(
                    x.reshape(-1, x.shape[-1]), bfa_weight, fb_weight, self.num_heads
                )
                return (
                    beta.view(*x.shape[:-1], self.num_heads),
                    forget_gate.view(*x.shape[:-1], fb_weight.shape[0]),
                )
            return self._project_gate_inputs(x)

        if fab_fa or fused_fab:
            max_rows = _KDA_FAB_FUSED_MAX_ROWS
        else:
            # Large batches: the unfused chain still overlaps the (all-SM) QKVG GEMM better
            # than it runs after it.
            max_rows = num_rows
        projection_aux_stream = self._projection_aux_stream if 0 < num_rows <= max_rows else None
        qkvg, (beta, gate) = fork_join(
            _project_qkvg,
            _project_gates,
            self._projection_fork_event,
            self._projection_join_event,
            projection_aux_stream,
        )

        d = self.proj_size
        q_proj, k_proj, v_proj = qkvg[..., : 3 * d].split(d, dim=-1)
        onorm_g = qkvg[..., 3 * d : 4 * d]
        if fab_fa:
            return q_proj, k_proj, v_proj, None, beta, onorm_g, gate
        return q_proj, k_proj, v_proj, gate, beta, onorm_g, None

    def _next_kda_layer(self) -> Optional["K3KDALinearAttention"]:
        """The KDA mixer of the next KDA layer of this model, if any."""
        later = [idx for idx in self._kda_group.keys() if idx > self.layer_idx]
        return self._kda_group.get(min(later)) if later else None

    def l2_prefetch_ranges(self) -> List[Tuple[int, int]]:
        """``(address, bytes)`` of the weights the fused verify path streams from HBM: the QKVG
        and o_proj weights and scales, ``[f_a | b]`` and ``f_b``."""
        if self._l2_prefetch_ranges_cache is None:
            from ..kernels.k3r_kda.k3_l2_prefetch import MAX_RANGES, tensor_range

            tensors = []
            for module in (self.qkvg_proj, self.o_proj):
                if module is not None:
                    tensors += [
                        getattr(module, "weight", None),
                        getattr(module, "weight_scale", None),
                    ]
            tensors += [self._bfa_proj_weight, self.f_b_proj.weight]
            ranges = []
            for tensor in tensors:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda and tensor.numel():
                    try:
                        ranges.append(tensor_range(tensor))
                    except ValueError:
                        continue
            self._l2_prefetch_ranges_cache = ranges[:MAX_RANGES]
        return self._l2_prefetch_ranges_cache

    def _issue_l2_prefetch(self, num_rows: int) -> None:
        """Prefetch the next KDA layer's weights into L2 from the projection aux stream, after the
        projections of this layer (see ``_KDA_L2_PREFETCH_MAX_ROWS``)."""
        if (
            num_rows > _KDA_L2_PREFETCH_MAX_ROWS
            or self._projection_aux_stream is None
            or not do_multi_stream()
        ):
            return
        nxt = self._next_kda_layer()
        if nxt is not None and nxt.qkvg_stages_weights(num_rows):
            # The next layer streams its QKVG weights on chip itself.
            return
        ranges = nxt.l2_prefetch_ranges() if nxt is not None else []
        if not ranges:
            return
        from ..ops.ssm.k3_l2_prefetch import k3_l2_prefetch

        # Recorded after the [f_a | b] join: the prefetch starts once QKVG and [f_a | b] are done
        # (both read their weights from HBM), i.e. under the recurrence. The aux stream is joined
        # back by its next user (see _KDA_L2_PREFETCH_MAX_ROWS).
        self._l2_prefetch_start.record()
        with torch.cuda.stream(self._projection_aux_stream):
            self._l2_prefetch_start.wait()
            k3_l2_prefetch(ranges)

    def forward_verify_fused(
        self,
        x2d,
        num_steps,
        layer_cache,
        ssm_pool,
        slot_indices,
        output: Optional[torch.Tensor] = None,
        allow_prequantized_output: bool = False,
    ) -> Union[torch.Tensor, _KdaPrequantizedCore]:
        """Fused multi-token verify (``mtp_ops.py``: the v1 / v2 / two-CTA kernels).

        The runtime packs ``num_steps`` new rows contiguously per request, so the kernel derives
        the row offsets from the request index and reads each request's pending accepted count
        from the slot-indexed replay pool (``packed_token_layout``). One launch replays the
        previous round's accepted drafts from the per-slot replay caches, then processes the new
        tokens, committing the recurrent state and conv windows in place after the golden token
        and caching the new drafts, and applies the gated RMSNorm in its epilogue:

        * with ``fa`` / ``w_fb``: the forget gate ``bf16(f_a @ W_fb^T)`` computed in the
          kernel (BF16 core);
        * ``quantize_output``: where ``allow_prequantized_output`` and ``o_proj`` takes a
          prequantized activation for this row count (SM107), the core quantized to MXFP8 for
          ``o_proj.forward_prequantized``;
        * otherwise the BF16 core, a programmatic dependent of the QKVG GEMM.
        """
        self._require_finalized()
        num_generations = x2d.shape[0] // num_steps
        num_spec = num_steps - 1
        H = self.num_heads
        K = self.head_k_dim
        x = x2d.view(num_generations, num_steps, -1)  # [B, T, hidden]
        T_total = num_generations * num_steps

        # The MXFP8 core cannot be concatenated with a prefill core or stored into the BCG buffer
        # (``allow_prequantized_output``).
        mxfp8_output = (
            allow_prequantized_output
            and output is None
            and _o_proj_takes_mxfp8(self.o_proj, T_total)
            and callable(getattr(self.o_proj, "forward_prequantized", None))
        )
        q_proj, k_proj, v_proj, forget_gate, beta_proj, onorm_g, f_a = self._project_verify_inputs(
            x, T_total, gate_from_fa=not mxfp8_output
        )
        self._issue_l2_prefetch(T_total)
        x_q = q_proj.view(1, T_total, H, K)
        x_k = k_proj.view(1, T_total, H, K)
        x_v = v_proj.view(1, T_total, H, self.head_dim)
        # Raw gate / beta: the kernel applies dt_bias, A_log, the
        # lower-bound sigmoid gate, and the beta sigmoid itself.
        beta = beta_proj.view(1, T_total, H)
        w_q, w_k, w_v = self._mtp_conv_weights
        onorm_g_view = onorm_g.view(1, T_total, H, self.head_dim)

        if f_a is not None:
            out = self._dispatch.mtp_verify_fused_fb(
                x_q=x_q,
                x_k=x_k,
                x_v=x_v,
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                cs_q=layer_cache.kda_conv_q,
                cs_k=layer_cache.kda_conv_k,
                cs_v=layer_cache.kda_conv_v,
                fa=f_a.view(1, T_total, K),
                w_fb=self.f_b_proj.weight.detach(),
                beta=beta,
                A_log=self.A_log.detach(),
                dt_bias=self.dt_bias.detach(),
                recurrent_state=ssm_pool,
                k_cache=layer_cache.kda_k_cache,
                g_cache=layer_cache.kda_g_cache,
                v_cache=layer_cache.kda_v_cache,
                beta_cache=layer_cache.kda_beta_cache,
                ssm_state_indices=slot_indices,
                num_spec=num_spec,
                num_accepted_tokens=layer_cache.prev_num_accepted_tokens,
                lower_bound=self.gate_lower_bound,
                scale=self.head_k_dim**-0.5,
                onorm_g=onorm_g_view,
                onorm_weight=self._onorm_w_f32,
                onorm_eps=self.o_norm.eps,
                use_pdl=True,
            )
            return self._store_core(out, output)

        out = self._dispatch.mtp_verify(
            mxfp8_output=mxfp8_output,
            x_q=x_q,
            x_k=x_k,
            x_v=x_v,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            cs_q=layer_cache.kda_conv_q,
            cs_k=layer_cache.kda_conv_k,
            cs_v=layer_cache.kda_conv_v,
            g=forget_gate.view(1, T_total, H, K),
            beta=beta,
            # .detach(): the CuTe DSL DLPack bridge rejects grad-tracking tensors.
            A_log=self.A_log.detach(),
            dt_bias=self.dt_bias.detach(),
            recurrent_state=ssm_pool,
            k_cache=layer_cache.kda_k_cache,
            g_cache=layer_cache.kda_g_cache,
            v_cache=layer_cache.kda_v_cache,
            beta_cache=layer_cache.kda_beta_cache,
            ssm_state_indices=slot_indices,
            cu_seqlens=None,
            num_spec=num_spec,
            num_accepted_tokens=layer_cache.prev_num_accepted_tokens,
            packed_token_layout=True,
            lower_bound=self.gate_lower_bound,
            scale=self.head_k_dim**-0.5,
            onorm_g=onorm_g_view,
            onorm_weight=self._onorm_w_f32,
            onorm_eps=self.o_norm.eps,
            fuse_output_norm=True,
            # A BF16-output verify launches as a programmatic dependent of the QKVG projection:
            # its state, replay and weight loads run before griddepcontrol.wait.
            use_pdl=not mxfp8_output,
        )
        if mxfp8_output:
            if not isinstance(out, tuple):
                raise RuntimeError("MXFP8-output MTP KDA must return activation and scale")
            return _KdaPrequantizedCore(*out)
        return self._store_core(out, output)

    def _output_gate(self, o: torch.Tensor, onorm_g: torch.Tensor) -> torch.Tensor:
        """The gated output RMSNorm ``rmsnorm(o) * sigmoid(g)`` (``rms_norm_gated_token_major``)
        of the prefill core."""
        return rms_norm_gated_token_major(
            o.reshape(-1, self.head_dim),
            onorm_g.reshape(-1, self.num_heads, self.head_dim),
            self.o_norm.weight,
            self.o_norm.eps,
            gate_activation="sigmoid",
        )

    def _store_core(self, core: torch.Tensor, output: Optional[torch.Tensor]) -> torch.Tensor:
        """Return the normalized core ``[tokens, H, D]``, copied into the BCG buffer ``output``
        where one is given."""
        core = core.reshape(-1, self.num_heads, self.head_dim)
        if output is None:
            return core
        output.copy_(core)
        return output


try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

# Head dimensions (K-nope, K-pe, V) the split-K context FMHA is built for.
_SPLIT_K_HEAD_DIMS = (128, 64, 128)


def split_k_context_unavailable_reason(attn) -> Optional[str]:
    """Why ``attn`` cannot run the Rubin FP8 split-K context FMHA, or None if it can."""
    sm_version = get_sm_version()
    if sm_version != 107:
        return f"requires SM107, got SM{sm_version}"
    if attn.num_heads <= 0 or attn.num_heads != attn.num_kv_heads:
        return (
            "requires matching positive Q/KV head counts per rank, "
            f"got {attn.num_heads}/{attn.num_kv_heads}"
        )
    head_dims = (attn.qk_nope_head_dim, attn.qk_rope_head_dim, attn.v_head_dim)
    if head_dims != _SPLIT_K_HEAD_DIMS:
        return f"requires head dimensions (K-nope, K-pe, V)={_SPLIT_K_HEAD_DIMS}, got {head_dims}"
    from ..kernels.k3r_mla_ctx import k3_context_fmha_op

    if k3_context_fmha_op.IS_K3_CONTEXT_FMHA_AVAILABLE:
        return None
    if k3_context_fmha_op.K3_CONTEXT_FMHA_IMPORT_ERROR is not None:
        return f"CuTe DSL kernel import failed: {k3_context_fmha_op.K3_CONTEXT_FMHA_IMPORT_ERROR!r}"
    return "CuTe DSL kernel lacks split-K epilogue merge support"


def supports_k3_fp8_split_k_context(attn) -> bool:
    """Whether the attention backend ``attn`` can run the Rubin FP8 split-K context FMHA."""
    reason = split_k_context_unavailable_reason(attn)
    if reason is None:
        return True
    logger.info_once(
        f"Kimi K3 MLA: Rubin FP8 split-K context FMHA disabled: {reason}; NVFP4 KV-cache layers "
        "cannot attend a cached prefix (chunked prefill raises).",
        key=("k3_mla_ctx_split_k_unavailable", reason),
    )
    return False


def run_k3_split_k_context_partition(
    attn,
    q: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    softmax_stats: torch.Tensor,
    merge_op: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    *,
    max_q_len: int,
    max_kv_len: int,
    is_causal: bool,
) -> None:
    """Attend FP8 Q to one FP8 KV partition and merge it into ``output`` / ``softmax_stats``.

    ``q`` ``[T_q, H * 192]``, ``k_nope`` / ``v`` ``[T_kv, H * 128]`` with unit inner stride,
    ``k_pe`` ``[T_kv, 64]``, ``output`` ``[T_q, H * 128]`` BF16, ``softmax_stats``
    ``[T_q, H, 2]`` FP32; the softmax scale is the attention backend's.
    """

    sm_scale = 1.0 / (attn.q_scaling * (attn.qk_nope_head_dim + attn.qk_rope_head_dim) ** 0.5)
    num_heads = attn.num_heads
    logger.info_once(
        "Kimi K3 MLA context FMHA: k3_context_fmha (Rubin FP8 split-K, epilogue "
        f"merge) serves {'causal current-chunk' if is_causal else 'cached-prefix'} partitions, "
        f"{num_heads} heads per rank.",
        key=("k3_mla_ctx_split_k_fmha", is_causal, num_heads),
    )
    k3_context_fmha(
        q.view(q.shape[0], num_heads, 192),
        k_nope.view(k_nope.shape[0], num_heads, 128),
        k_pe.view(k_pe.shape[0], 1, 64),
        v.view(v.shape[0], num_heads, 128),
        output.view(q.shape[0], num_heads, 128),
        cu_q_seqlens,
        cu_kv_seqlens,
        softmax_stats,
        merge_op,
        max_q_len,
        max_kv_len,
        sm_scale,
        is_causal,
    )


def load_fp4_mla_chunked_kv_cache_mxfp8(
    metadata,
    layer_idx: int,
    *,
    num_ctx_cached_tokens: int,
    cu_chunked_seq_len: torch.Tensor,
    chunked_global_offset: torch.Tensor,
    chunked_max_seq_len: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Gather one cached-prefix partition of dense FP4 MLA V2 storage as MXFP8 kv_b_proj input.

    The partition and metadata are those of ``load_fp4_mla_chunked_kv_cache``. Returns
    ``((E4M3 [M, 512], UE8M0 [pad_up(M, 128) * 16]), E4M3 [M, 64])``: the BF16-rounded latent
    quantized 1x128 with scales in the K32 R128c4 layout ``cute_dsl_mxfp8_gemm_rubin`` reads, and
    the BF16-rounded RoPE part plus its residual in unit-scale E4M3.
    """
    if kv_lora_rank != 512 or qk_rope_head_dim != FP4_MLA_K_RESIDUAL_DIM:
        raise ValueError(
            "MXFP8 gather requires kv_lora_rank=512 and "
            f"qk_rope_head_dim={FP4_MLA_K_RESIDUAL_DIM}, got {kv_lora_rank} and {qk_rope_head_dim}."
        )
    if num_ctx_cached_tokens < 0:
        raise ValueError("FP4 MLA chunk gather token count must be non-negative.")
    if chunked_max_seq_len < 0:
        raise ValueError("FP4 MLA chunk gather max sequence length must be non-negative.")
    if num_ctx_cached_tokens > 0 and chunked_max_seq_len == 0:
        raise ValueError("A non-empty FP4 MLA chunk gather requires a positive max length.")
    num_contexts = int(metadata.num_contexts)
    if num_ctx_cached_tokens > 0 and num_contexts <= 0:
        raise ValueError("A non-empty FP4 MLA chunk gather requires context requests.")
    tensors = (cu_chunked_seq_len, chunked_global_offset)
    if any(not isinstance(tensor, torch.Tensor) or not tensor.is_cuda for tensor in tensors):
        raise ValueError("FP4 MLA chunk gather metadata must use CUDA tensors.")
    if (
        cu_chunked_seq_len.dtype != torch.int64
        or cu_chunked_seq_len.ndim != 1
        or cu_chunked_seq_len.numel() < num_contexts + 1
        or not cu_chunked_seq_len.is_contiguous()
    ):
        raise ValueError(
            "FP4 MLA chunk gather requires a contiguous int64 cumulative-length tensor."
        )
    if (
        chunked_global_offset.dtype != torch.int64
        or chunked_global_offset.ndim != 1
        or chunked_global_offset.numel() < num_contexts
        or not chunked_global_offset.is_contiguous()
    ):
        raise ValueError("FP4 MLA chunk gather requires contiguous int64 global offsets.")
    if cu_chunked_seq_len.device != chunked_global_offset.device:
        raise ValueError("FP4 MLA chunk gather metadata tensors must share one device.")

    device = cu_chunked_seq_len.device
    compressed_kv = torch.empty(
        (num_ctx_cached_tokens, kv_lora_rank), dtype=torch.float8_e4m3fn, device=device
    )
    compressed_sf = torch.empty(
        (triton.cdiv(num_ctx_cached_tokens, 128) * 128 * (kv_lora_rank // 32),),
        dtype=torch.uint8,
        device=device,
    )
    k_pe = torch.empty(
        (num_ctx_cached_tokens, qk_rope_head_dim), dtype=torch.float8_e4m3fn, device=device
    )
    if num_ctx_cached_tokens == 0:
        return (compressed_kv, compressed_sf), k_pe

    state = metadata.fp4_mla_state
    if not bool(getattr(state, "device_page_table", False)):
        raise RuntimeError("FP4 MLA chunk gather requires fixed-stride device page metadata.")
    _materialize_fp4_mla_device_page_table_for_forward(metadata)
    page_table_stride = int(state.page_table_stride)
    page_ids = state._paged_kv_indices
    if (
        page_table_stride <= 0
        or not isinstance(page_ids, torch.Tensor)
        or page_ids.dtype != torch.int32
        or not page_ids.is_cuda
        or page_ids.device != device
        or page_ids.numel() < num_contexts * page_table_stride
    ):
        raise RuntimeError("FP4 MLA chunk gather received invalid device page metadata.")

    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
    head_dim = kv_lora_rank + qk_rope_head_dim
    storage_head_dim = _validate_fp4_mla_kv_storage_shape(
        kv_cache,
        sf_cache,
        head_dim=head_dim,
        backend=_fp4_mla_attention_backend(),
    )
    if storage_head_dim != head_dim + FP4_MLA_K_RESIDUAL_DIM:
        raise RuntimeError(
            "FP4 MLA chunk gather requires K residual storage for BF16 reconstruction."
        )
    from ..ops.attention.fp4_mla_gather_mxfp8 import fp4_mla_gather_mxfp8

    logger.info_once(
        "Kimi K3 MLA context gather: cached-prefix chunks are read from the FP4 cache as MXFP8 "
        "latent + FP8 K-pe (fp4_mla_gather_mxfp8).",
        key="k3_mla_ctx_gather_mxfp8",
    )
    fp4_mla_gather_mxfp8(
        compressed_kv,
        compressed_sf,
        k_pe,
        kv_cache,
        sf_cache.view(torch.float8_e4m3fn),
        page_ids,
        cu_chunked_seq_len,
        chunked_global_offset,
        _get_fp4_mla_global_scale(metadata, kv_cache.device),
        chunked_max_seq_len,
        metadata.page_size,
        page_table_stride=page_table_stride,
        num_contexts=num_contexts,
    )
    return (compressed_kv, compressed_sf), k_pe


def uses_k3_split_k_context(layer, q: torch.Tensor, output: torch.Tensor) -> bool:
    """Whether the FP4-cache chunked prefill of ``layer`` runs on the split-K context FMHA: BF16 Q
    and output, an NVFP4 latent cache and :func:`supports_k3_fp8_split_k_context`."""
    mha = layer.mha
    return (
        q.dtype == torch.bfloat16
        and output.dtype == torch.bfloat16
        and layer.out_scale is None
        and isinstance(mha, K3MlaAttnState)
        and mha.uses_fp4_mla_attention
        and supports_k3_fp8_split_k_context(mha)
    )


def kv_b_takes_mxfp8(layer) -> bool:
    """Whether kv_b runs as the SM107 MXFP8 GEMM on an MXFP8 cached-prefix gather.

    Needs SM107 with the CuTe DSL Rubin kernels, the 512 / 64 latent / RoPE dimensions of the
    gather, and a bias-free FP8 block-scale kv_b that takes a prequantized CuTe-layout input, with
    its UE8M0 weight scale in the CuTe MXFP8 layout.
    """
    kv_b_proj = layer.kv_b_proj
    weight = getattr(kv_b_proj, "weight", None)
    weight_scale = getattr(kv_b_proj, "weight_scale", None)
    return (
        get_sm_version() == 107
        and IS_CUTLASS_DSL_RUBIN_AVAILABLE
        and layer.kv_lora_rank == 512
        and layer.qk_rope_head_dim == 64
        and getattr(kv_b_proj, "bias", None) is None
        and getattr(kv_b_proj, "supports_prequantized_input", False)
        and weight is not None
        and weight.dtype == torch.float8_e4m3fn
        and weight_scale is not None
        and weight_scale.dtype == torch.uint8
    )


def forward_context_split_k_fp4(
    layer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    latent_cache: torch.Tensor,
    attn_metadata: TrtllmAttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """FP4 MLA chunked prefill of ``layer`` with every partition on the split-K context FMHA."""
    mha = layer.mha
    if latent_cache is None:
        raise RuntimeError("FP4 MLA chunked prefill requires latent_cache.")
    kv_b_mxfp8 = kv_b_takes_mxfp8(layer)
    logger.info_once(
        "Kimi K3 MLA cached-prefix context: NVFP4 KV-cache layers run the Rubin FP8 split-K "
        "context FMHA (k3_context_fmha); cached prefix: "
        + (
            "MXFP8 gather + FP8-output MXFP8 kv_b GEMM"
            if kv_b_mxfp8
            else "BF16 gather + kv_b_proj + FP8 quantization"
        )
        + "; Q FP8 quantization: static_quantize_e4m3 (Triton).",
        key=("k3_mla_ctx_split_k", kv_b_mxfp8),
    )
    lora, rope = layer.kv_lora_rank, layer.qk_rope_head_dim
    heads = layer.num_heads_tp

    # Rotate Q and the current K in place and append the chunk to the FP4 cache; the current
    # chunk's rotated K-pe is then read from latent_cache.
    mla_cache.append_fp4_mla_context_cache(mha, q, latent_cache, attn_metadata)
    current_k_pe = latent_cache.reshape(-1, lora + rope)[:, lora:]
    fp8_unit_scale = mha.kv_cache_scaling_factor
    q_fp8 = static_quantize_e4m3(q.contiguous(), fp8_unit_scale)
    context_cu_q_seqlens = attn_metadata.mla_prepare_ctx_cu_seqlens()
    if context_cu_q_seqlens is None:
        raise RuntimeError("K3 split-K context FMHA requires context requests.")

    # The first partition of every request writes the output and its stats (merge op 2).
    softmax_stats = torch.empty(
        (attn_metadata.num_ctx_tokens, heads, 2), dtype=torch.float, device=q.device
    )
    kv_split = [heads * layer.qk_nope_head_dim, heads * layer.v_head_dim]

    def expand_kv(
        partition_compressed_kv: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        partition_k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if kv_b_mxfp8:
            if isinstance(partition_compressed_kv, tuple):
                activation, activation_scale = partition_compressed_kv
            else:
                activation, activation_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
                    partition_compressed_kv.contiguous()
                )
            # The FP8 output keeps the GEMM's BF16 rounding (unit output scale).
            partition_kv = k3_mxfp8_gemm_fp8_out(
                activation,
                layer.kv_b_proj.weight,
                activation_scale,
                layer.kv_b_proj.weight_scale,
            )
        else:
            partition_kv, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                layer.kv_b_proj(partition_compressed_kv), fp8_unit_scale
            )
        if partition_k_pe.dtype == torch.float8_e4m3fn:
            k_pe_source = "FP8 from the gather"
        elif fp8_static_quant.supports_rows(partition_k_pe):
            k_pe_source = "static_quantize_e4m3_rows (Triton)"
            partition_k_pe = static_quantize_e4m3(partition_k_pe, fp8_unit_scale, rows=True)
        else:
            k_pe_source = "static_quantize_e4m3_per_tensor"
            partition_k_pe, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                partition_k_pe.contiguous(), fp8_unit_scale
            )
        logger.info_once(
            "Kimi K3 MLA context expand: per-chunk kv_b_proj runs as "
            + (
                "k3_mxfp8_gemm_fp8_out (MXFP8 in, FP8 out)"
                if kv_b_mxfp8
                else "kv_b_proj (BF16) + static_quantize_e4m3_per_tensor"
            )
            + f"; K-pe: {k_pe_source}; K-nope / K-pe / V go to the split-K FMHA unconcatenated.",
            key=("k3_mla_ctx_split_k_expand", kv_b_mxfp8, k_pe_source),
        )
        partition_k_nope, partition_v = partition_kv.split(kv_split, -1)
        return partition_k_nope, partition_k_pe, partition_v

    def run_partition(
        partition: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        cu_kv_seqlens: torch.Tensor,
        max_kv_len: int,
        merge_op: torch.Tensor,
        is_causal: bool,
    ) -> None:
        partition_k_nope, partition_k_pe, partition_v = partition
        run_k3_split_k_context_partition(
            mha,
            q_fp8,
            partition_k_nope,
            partition_k_pe,
            partition_v,
            output,
            softmax_stats,
            merge_op,
            context_cu_q_seqlens,
            cu_kv_seqlens,
            max_q_len=attn_metadata.max_ctx_seq_len,
            max_kv_len=max_kv_len,
            is_causal=is_causal,
        )

    for loop_idx in range(attn_metadata.chunked_loop_num):
        cu_chunked_seq_len = attn_metadata.cu_chunked_seq_len[loop_idx]
        total_chunked_tokens = int(
            attn_metadata.host_cu_chunked_seq_len[loop_idx, attn_metadata.num_contexts]
        )
        chunked_max_seq_len = int(attn_metadata.max_chunk_len_per_loop[loop_idx])
        if kv_b_mxfp8:
            chunked_compressed_kv, chunked_k_pe = load_fp4_mla_chunked_kv_cache_mxfp8(
                attn_metadata,
                mha.layer_idx,
                num_ctx_cached_tokens=total_chunked_tokens,
                cu_chunked_seq_len=cu_chunked_seq_len,
                chunked_global_offset=attn_metadata.chunked_global_offset[loop_idx],
                chunked_max_seq_len=chunked_max_seq_len,
                kv_lora_rank=lora,
                qk_rope_head_dim=rope,
            )
        else:
            chunked_compressed_kv, chunked_k_pe = mla_cache.load_chunked_kv_cache_for_mla(
                mha,
                attn_metadata,
                num_ctx_cached_tokens=total_chunked_tokens,
                cu_chunked_seq_len=cu_chunked_seq_len,
                chunked_global_offset=attn_metadata.chunked_global_offset[loop_idx],
                chunked_max_seq_len=chunked_max_seq_len,
                out_dtype=q.dtype,
            )
        partition = expand_kv(chunked_compressed_kv, chunked_k_pe)
        chunked_compressed_kv = None
        chunked_k_pe = None
        run_partition(
            partition,
            cu_chunked_seq_len,
            chunked_max_seq_len,
            attn_metadata.merge_op_tensor[loop_idx],
            is_causal=False,
        )

    run_partition(
        expand_kv(compressed_kv, current_k_pe),
        context_cu_q_seqlens,
        attn_metadata.max_ctx_seq_len,
        attn_metadata.merge_op_tensor[attn_metadata.chunked_loop_num],
        is_causal=True,
    )
    return output


__all__ = [
    "forward_context_split_k_fp4",
    "kv_b_takes_mxfp8",
    "load_fp4_mla_chunked_kv_cache_mxfp8",
    "run_k3_split_k_context_partition",
    "split_k_context_unavailable_reason",
    "supports_k3_fp8_split_k_context",
    "uses_k3_split_k_context",
    "context_cached_kv",
    "context_chunked_prefill",
    "context_default",
    "forward_context",
    "project_context_kv",
    "HelixGenHooks",
    "attn_forward_gen",
    "forward_absorption_generation",
    "forward_gen_q_fused",
    "gen_attention",
    "helix_exchange",
    "helix_gen_zero_kv_mask",
    "helix_tail_applies",
    "latent_row",
    "project_helix_v_partials",
    "q_fused_applies",
    "tail_output_buffer",
    "K3MLAAttention",
]


def project_context_kv(layer, compressed_kv: torch.Tensor) -> torch.Tensor:
    """kv_b over the context heads: ``[T, H * qk_nope | H * v_head_dim]`` (one GEMM, or under Helix
    this rank's head range of the K and V halves, concatenated)."""
    if layer.num_context_heads == layer.num_heads_tp:
        return layer.kv_b_proj(compressed_kv)
    first_head = layer.mapping.cp_rank * layer.num_context_heads
    last_head = first_head + layer.num_context_heads
    key = layer.forward_slice(
        layer.kv_b_proj,
        compressed_kv,
        first_head * layer.qk_nope_head_dim,
        last_head * layer.qk_nope_head_dim,
    )
    value_start = layer.num_heads_tp * layer.qk_nope_head_dim
    value = layer.forward_slice(
        layer.kv_b_proj,
        compressed_kv,
        value_start + first_head * layer.v_head_dim,
        value_start + last_head * layer.v_head_dim,
    )
    # The [all K | all V] row stride the context FP8 preprocessing reads.
    return torch.cat((key, value), dim=-1)


def _expand_kv_bf16(layer, compressed_kv: torch.Tensor, k_pe: torch.Tensor):
    """kv_b, then K ``[T, H * 192]`` (K-nope and the shared K-pe, concatenated) and V."""
    heads = layer.num_context_heads
    kv = project_context_kv(layer, compressed_kv)
    k_nope, v = kv.split([heads * layer.qk_nope_head_dim, heads * layer.v_head_dim], -1)
    k_nope = k_nope.view(-1, heads, layer.qk_nope_head_dim)
    k_pe = k_pe.view(-1, 1, layer.qk_rope_head_dim)
    k = maybe_compiled_cat((k_nope, k_pe.expand(-1, heads, -1)), dim=-1)
    return k.view(-1, heads * layer.qk_head_dim), v


def _expand_kv_fp8(layer, compressed_kv: torch.Tensor, k_pe: torch.Tensor):
    """kv_b, then FP8 K ``[T, H * 192]`` and V ``[T, H * 128]`` in one pass: the bytes the
    attention op's context copy writes for the BF16 K and V."""
    return mla_kv_to_fp8(
        project_context_kv(layer, compressed_kv),
        k_pe.view(-1, layer.qk_rope_head_dim),
        layer.num_context_heads,
        layer.qk_nope_head_dim,
        layer.v_head_dim,
    )


def _fp8_q_args(layer, q_fp8):
    """Attention-op arguments that hand it ``q_fp8`` as the already-quantized FP8 Q."""
    if q_fp8 is None:
        return {}
    return {"quant_q_buffer": q_fp8, "quant_scale_qkv": layer.mha.kv_scale_orig_quant}


def _quantize_q_fp8(layer, q: torch.Tensor) -> torch.Tensor:
    """Q in FP8 as the attention op's context copy quantizes it (unit scale)."""
    q_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        q, layer.mha.kv_scale_orig_quant
    )
    return q_fp8


@functools.cache
def _warmup_expand_kv_fp8(
    num_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, device
):
    """Compile ``mla_kv_to_fp8`` before the first cached-KV context step (k_pe contiguous for
    loaded chunks, a ``latent_cache`` view for the current tokens)."""
    num_tokens = 2
    kv = torch.zeros(
        num_tokens, num_heads * (qk_nope_head_dim + v_head_dim), dtype=torch.bfloat16, device=device
    )
    latent = torch.zeros(
        num_tokens, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16, device=device
    )
    for k_pe in (latent[:, kv_lora_rank:].contiguous(), latent[:, kv_lora_rank:]):
        mla_kv_to_fp8(kv, k_pe, num_heads, qk_nope_head_dim, v_head_dim)


@functools.cache
def _warmup_expand_kv_bf16(
    num_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, dtype, device
):
    """Compile ``maybe_compiled_cat`` for both K layouts (a loaded chunk's contiguous K-pe and the
    current tokens' ``latent_cache`` view) with the token dim dynamic; one token count still
    specializes, so it is warmed separately."""

    def warmup(num_tokens):
        k_nope = torch.empty(
            num_tokens, num_heads * (qk_nope_head_dim + v_head_dim), dtype=dtype, device=device
        )[:, : num_heads * qk_nope_head_dim].view(num_tokens, num_heads, qk_nope_head_dim)
        chunked_k_pe = torch.empty(
            num_tokens, 1, qk_rope_head_dim, dtype=dtype, device=device
        ).expand(-1, num_heads, -1)
        k_pe = torch.empty(
            num_tokens, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
        )[:, :, -qk_rope_head_dim:].expand(-1, num_heads, -1)
        torch._dynamo.maybe_mark_dynamic(k_nope, 0)
        torch._dynamo.maybe_mark_dynamic(chunked_k_pe, 0)
        torch._dynamo.maybe_mark_dynamic(k_pe, 0)
        maybe_compiled_cat((k_nope, chunked_k_pe), dim=-1)
        maybe_compiled_cat((k_nope, k_pe), dim=-1)

    warmup(2)
    warmup(1)


def _kv_b_strided_out(layer) -> bool:
    """Whether the no-prefix expansion writes K-nope / V straight into the attention inputs with
    per-head BMMs: a bias-free BF16 kv_b on SM100f."""
    w = layer.kv_b_proj.weight
    return (
        is_sm_100f()
        and w.dtype == torch.bfloat16
        and tuple(w.shape)
        == (layer.num_heads_tp * (layer.qk_nope_head_dim + layer.v_head_dim), layer.kv_lora_rank)
        and getattr(layer.kv_b_proj, "bias", None) is None
    )


def context_default(
    layer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: torch.Tensor,
) -> torch.Tensor:
    """Context attention without a cached prefix (see the module docstring)."""
    heads = layer.num_context_heads
    nope, rope, vdim = layer.qk_nope_head_dim, layer.qk_rope_head_dim, layer.v_head_dim
    append_owned_kv = md.helix_context_local_slots is not None
    k_pe = None
    if append_owned_kv:
        mla_cache.mla_rope_append_paged_kv_assign_q(layer.mha, q, latent_cache, md)
        k_pe = latent_cache[:, layer.kv_lora_rank :]
    k = torch.empty_like(q).view(-1, heads, layer.qk_head_dim)
    if _kv_b_strided_out(layer):
        num_tokens = compressed_kv.shape[0]
        w = layer.kv_b_proj.weight
        nope_rows = layer.num_heads_tp * nope
        w_k = w[:nope_rows].view(layer.num_heads_tp, nope, layer.kv_lora_rank)
        w_v = w[nope_rows:].view(layer.num_heads_tp, vdim, layer.kv_lora_rank)
        if heads != layer.num_heads_tp:
            first = layer.mapping.cp_rank * heads
            w_k, w_v = w_k[first : first + heads], w_v[first : first + heads]
        ckv = compressed_kv.unsqueeze(0).expand(heads, num_tokens, layer.kv_lora_rank)
        # [H, T, nope] view of K with strides (192, H * 192, 1).
        k_nope_out = k[..., :nope].transpose(0, 1)
        # The context FP8 quantize kernel reads V with the [K-nope | V] row stride of the kv_b
        # output: V is the second half of such a buffer (its first half is never written).
        kv_buf = torch.empty(num_tokens, heads * (nope + vdim), dtype=q.dtype, device=q.device)
        v = kv_buf[:, heads * nope :]
        v_out = v.unflatten(-1, (heads, vdim)).transpose(0, 1)
        bmm_out(ckv, w_k.transpose(1, 2), k_nope_out)
        bmm_out(ckv, w_v.transpose(1, 2), v_out)
    else:
        kv = project_context_kv(layer, compressed_kv)
        k_nope, v = kv.split([heads * nope, heads * vdim], -1)
        maybe_compiled_copy_(k[..., :nope], k_nope.view(-1, heads, nope))
    # With the RoPE fused in the op the K-pe slot stays unwritten; the op reads latent_cache.
    if append_owned_kv:
        k[..., nope:] = k_pe.view(-1, 1, rope)
    k = k.view(-1, heads * layer.qk_head_dim)
    if layer.mha.uses_fp4_mla_attention:
        # The FP8 context FMHA over the step's FP8 scratch cache beside the FP4 scatter.
        return mla_cache.fp4_mla_context_attention(layer.mha, q, k, v, md, output, latent_cache)
    return mla_attn.mla_context_attention(
        layer.mha, q, k, v, md, output, latent_cache=None if append_owned_kv else latent_cache
    )


def context_cached_kv(
    layer,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Context attention over the whole paged cache (cached prefix, no chunked prefill)."""
    mha = layer.mha
    mla_cache.mla_rope_append_paged_kv_assign_q(mha, q, latent_cache, md)
    full_compressed_kv, full_k_pe = mla_cache.load_paged_kv_cache_for_mla(mha, md, q.dtype)
    assert full_compressed_kv.shape == (
        md.num_ctx_cached_tokens + md.num_ctx_tokens,
        layer.kv_lora_rank,
    )
    assert full_k_pe.shape == (md.num_ctx_cached_tokens + md.num_ctx_tokens, layer.qk_rope_head_dim)
    assert full_compressed_kv.is_contiguous() and full_k_pe.is_contiguous()
    q_fp8 = None
    if layer.fp8_cached_kv_context:
        q_fp8 = _quantize_q_fp8(layer, q)
        full_k, full_v = _expand_kv_fp8(layer, full_compressed_kv, full_k_pe)
    else:
        full_k, full_v = _expand_kv_bf16(layer, full_compressed_kv, full_k_pe)
    full_compressed_kv = full_k_pe = None
    return mla_attn.mla_context_attention(
        mha, q, full_k, full_v, md, output, **_fp8_q_args(layer, q_fp8)
    )


def context_chunked_prefill(
    layer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    latent_cache: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Chunked-prefill context attention over an FP8 / BF16 cache (see the module docstring)."""
    mha = layer.mha
    heads = layer.num_context_heads
    mla_cache.mla_rope_append_paged_kv_assign_q(mha, q, latent_cache, md)
    fp8 = layer.fp8_cached_kv_context
    expand = _expand_kv_fp8 if fp8 else _expand_kv_bf16
    # With FP8 attention inputs, Q is quantized once for every partition.
    q_fp8 = _quantize_q_fp8(layer, q) if fp8 else None
    q_args = _fp8_q_args(layer, q_fp8)
    softmax_stats = torch.empty((md.num_ctx_tokens, heads, 2), dtype=torch.float, device="cuda")
    temp_softmax_stats = torch.empty(
        (md.num_ctx_tokens, heads, 2), dtype=torch.float, device="cuda"
    )
    temp_output = q.new_empty((q.size(0), heads * layer.v_head_dim), dtype=q.dtype)
    buffer_batch_size = md.runtime_features.chunked_prefill_buffer_batch_size

    # Each partition attends with its own KV lengths; the originals are restored at the end.
    origin_kv_lens_cuda_runtime = md.kv_lens_cuda_runtime
    origin_kv_lens_runtime = md.kv_lens_runtime
    origin_ctx_total_kv_len = md.host_total_kv_lens[0].item()

    for loop_idx in range(md.chunked_loop_num):
        total_chunked_tokens = md.host_cu_chunked_seq_len[loop_idx, md.num_contexts]
        chunked_compressed_kv, chunked_k_pe = mla_cache.load_chunked_kv_cache_for_mla(
            mha,
            md,
            num_ctx_cached_tokens=total_chunked_tokens,
            cu_chunked_seq_len=md.cu_chunked_seq_len[loop_idx],
            chunked_global_offset=md.chunked_global_offset[loop_idx],
            chunked_max_seq_len=md.max_chunk_len_per_loop[loop_idx],
            out_dtype=q.dtype,
            helix_plan=(
                md.helix_mla_chunk_plans[loop_idx] if md.helix_mla_chunk_plans is not None else None
            ),
        )
        chunked_k, chunked_v = expand(layer, chunked_compressed_kv, chunked_k_pe)
        chunked_compressed_kv = chunked_k_pe = None
        md.kv_lens_runtime = md.host_chunked_seq_len[loop_idx]
        md.kv_lens_cuda_runtime = md.chunked_seq_len[loop_idx]
        md.host_total_kv_lens[0] = total_chunked_tokens
        temp_output = mla_attn.mla_context_attention(
            mha,
            q,
            chunked_k,
            chunked_v,
            md,
            temp_output,
            full_mask=True,
            softmax_stats=temp_softmax_stats,
            chunked_prefill_buffer_batch_size=buffer_batch_size,
            **q_args,
        )
        mla_cache.merge_attention_for_mla(
            mha,
            output,
            temp_output,
            softmax_stats,
            temp_softmax_stats,
            md.merge_op_tensor[loop_idx],
            md,
        )

    # The current chunk, causal; its rotated K-pe is in latent_cache.
    _, k_pe = latent_cache.view([-1, layer.kv_lora_rank + layer.qk_rope_head_dim]).split(
        [layer.kv_lora_rank, layer.qk_rope_head_dim], -1
    )
    k, v = expand(layer, compressed_kv, k_pe)
    md.kv_lens_runtime = md.prompt_lens_cpu_runtime
    md.kv_lens_cuda_runtime = md.prompt_lens_cuda_runtime
    md.host_total_kv_lens[0] = md.prompt_lens_cpu_runtime[: md.num_contexts].sum().item()
    temp_output = mla_attn.mla_context_attention(
        mha,
        q,
        k,
        v,
        md,
        temp_output,
        softmax_stats=temp_softmax_stats,
        chunked_prefill_buffer_batch_size=buffer_batch_size,
        **q_args,
    )
    mla_cache.merge_attention_for_mla(
        mha,
        output,
        temp_output,
        softmax_stats,
        temp_softmax_stats,
        md.merge_op_tensor[md.chunked_loop_num],
        md,
    )
    md.kv_lens_runtime = origin_kv_lens_runtime
    md.kv_lens_cuda_runtime = origin_kv_lens_cuda_runtime
    md.host_total_kv_lens[0] = origin_ctx_total_kv_len
    return output


def forward_context(
    layer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: torch.Tensor,
) -> torch.Tensor:
    """Context attention of the context rows into ``output``: picks the chain by cache state."""
    mha = layer.mha
    if md.helix_context_local_slots is not None:
        if mha.uses_fp4_mla_attention:
            raise NotImplementedError(
                "Owner-only Helix prefill does not support dense FP4 MLA KV cache"
            )
        if md.num_ctx_cached_tokens > 0 and (
            not md.runtime_features.chunked_prefill or get_sm_version() < 100
        ):
            raise NotImplementedError(
                "Helix cached context requires chunked prefill on SM100 or newer"
            )
    if mha.has_cached_kv_for_mla_context_warmup(md):
        dims = (
            layer.num_context_heads,
            layer.qk_nope_head_dim,
            layer.qk_rope_head_dim,
            layer.kv_lora_rank,
            layer.v_head_dim,
        )
        if layer.fp8_cached_kv_context:
            _warmup_expand_kv_fp8(*dims, q.device)
        else:
            _warmup_expand_kv_bf16(*dims, q.dtype, q.device)
    chunked = mha.is_chunked_prefill_for_mla_context(md)
    if chunked and get_sm_version() >= 100:
        if mha.uses_fp4_mla_attention:
            if not uses_k3_split_k_context(layer, q, output):
                raise NotImplementedError(
                    "Kimi K3 MLA: an FP4 MLA cache attends a cached prefix only through the Rubin "
                    "FP8 split-K context FMHA (SM107, BF16 Q and output, see ctx_split_k)"
                )
            return forward_context_split_k_fp4(layer, q, compressed_kv, latent_cache, md, output)
        return context_chunked_prefill(layer, q, compressed_kv, latent_cache, md, output)
    if chunked or mha.has_cached_kv_for_mla_context(md):
        if mha.uses_fp4_mla_attention:
            raise RuntimeError(
                "FP4 MLA attends a cached prefix only through chunked-prefill partitions on "
                "SM100+; enable chunked prefill."
            )
        return context_cached_kv(layer, q, latent_cache, md, output)
    return context_default(layer, q, compressed_kv, md, output, latent_cache)


def _check_cu_result(cu_func_ret):
    """The values a CUDA driver call returned after its status, raising if it failed."""
    if isinstance(cu_func_ret, tuple):
        cu_result, *others = cu_func_ret
        if cu_result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_result)
        if len(others) == 1:
            return others[0]
        elif len(others) > 1:
            return tuple(others)
        else:
            return None
    else:
        if cu_func_ret != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_func_ret)
        return None


# Receive buffer per rank in MiB (two halves).
_TAIL_MB = 16


class K3HelixTail:
    """The fused Helix tail over the CP group of ``mapping`` (one receive workspace per mapping)."""

    _cache: Dict[Mapping, "K3HelixTail"] = {}

    def __init__(self, mapping: Mapping):
        from ..kernels.k3r_helix.k3_helix_tail import SENTINEL

        self.mapping = mapping
        MnnvlMemory.initialize()
        nbytes = _TAIL_MB << 20
        logger.info(f"Rank {mapping.cp_rank} allocating a {nbytes >> 20} MiB Helix tail buffer")
        self.workspace = HelixCpMnnvlMemory(mapping, nbytes)
        self.ws_addr = self.workspace.ptr
        self.rank_stride = self.workspace.rank_stride
        self.half_bytes = nbytes // 2 // 16 * 16
        # Armed through the driver: the kernel only uses raw addresses (see HelixAllToAllFused).
        _check_cu_result(
            cuda.cuMemsetD8Async(
                cuda.CUdeviceptr(self.ws_addr + mapping.cp_rank * self.rank_stride),
                SENTINEL & 0xFF,
                nbytes,
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            )
        )
        self.state = torch.zeros(8, dtype=torch.int64, device="cuda")
        num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self.max_ctas = num_sms
        self.use_pdl = USE_PDL
        # Tests on one GPU: this rank's records go to its own buffer (k3_helix_tail loopback).
        self.loopback = False
        torch.cuda.synchronize()
        HelixCpMnnvlMemory.get_comm(mapping).barrier()

    @staticmethod
    def get(mapping: Mapping) -> "K3HelixTail":
        if mapping not in K3HelixTail._cache:
            K3HelixTail._cache[mapping] = K3HelixTail(mapping)
        return K3HelixTail._cache[mapping]

    @staticmethod
    def applies(
        mapping: Mapping, num_tokens: int, num_heads: int, latent_dim: int, v_head_dim: int
    ) -> bool:
        """Whether the fused tail serves a generation batch of ``num_tokens`` tokens."""
        from ..kernels.k3r_helix import k3_helix_tail as kernel

        if get_sm_version() not in (100, 103, 107) or num_tokens <= 0:
            return False
        if not mapping.has_cp_helix() or mapping.cp_config.get("use_nccl_for_alltoall", True):
            return False
        if not kernel.can_run(mapping.cp_size, num_heads, latent_dim, v_head_dim):
            return False
        nbytes = _TAIL_MB << 20
        record = kernel.record_bytes(num_heads // mapping.cp_size, v_head_dim)
        return num_tokens * mapping.cp_size * record <= nbytes // 2 // 16 * 16

    def __call__(
        self,
        rows: torch.Tensor,
        stats: torch.Tensor,
        w_uv: torch.Tensor,
        zero_kv_mask: Optional[torch.Tensor],
        fold=None,
        out: Optional[torch.Tensor] = None,
        mx_out=None,
        timestamps: Optional[torch.Tensor] = None,
        mx_atom: bool = False,
    ) -> torch.Tensor:
        """Fold, project, exchange and combine one layer's partials.

        Args:
            rows: BF16 ``[T, H * 512]`` decode output (latent rows ``(token, head)``); with
                ``fold`` only the rows the decode stored directly are read.
            stats: FP32 ``[T, H, 2]`` ``(max, sum)`` of ``rows``; read only without ``fold``.
            w_uv: BF16 ``[H, 128, 512]`` V up-projection of every head.
            zero_kv_mask: Optional bool ``[T]``, True where this rank owns no KV of the token.
            fold: ``k3_helix_tail.FoldInputs`` of the deferred split reduction, or None.
            out: Optional BF16 ``[T, H / P * 128]`` output.
            mx_out: Optional ``(gate, xq, xs)``: also store the MXFP8 form of
                ``bf16(out * gate)``, o_proj's gated operand (see ``k3_helix_tail``).
            timestamps: Optional int64 phase stamps (microbenchmarks, see ``k3_helix_tail``).
            mx_atom: (c240) ``xs`` in the mid-M GEMM's scale layout (see ``k3_helix_tail``).

        Returns:
            The combined BF16 ``[T, H / P * 128]`` output of this rank's heads.
        """
        from ..ops.attention.k3_helix_tail import k3_helix_tail

        num_tokens = rows.shape[0]
        num_heads = w_uv.shape[0]
        cp_size = self.mapping.cp_size
        if out is None:
            out = rows.new_empty((num_tokens, num_heads // cp_size * w_uv.shape[1]))
        if zero_kv_mask is not None:
            zero_kv_mask = zero_kv_mask[:num_tokens].contiguous()
        k3_helix_tail(
            rows.reshape(num_tokens * num_heads, -1),
            stats.reshape(num_tokens * num_heads, 2),
            w_uv,
            out,
            fold=fold,
            zero_kv_mask=zero_kv_mask,
            cp_rank=self.mapping.cp_rank,
            cp_size=cp_size,
            ws_addr=self.ws_addr,
            rank_stride=self.rank_stride,
            half_bytes=self.half_bytes,
            state=self.state,
            max_ctas=self.max_ctas,
            use_pdl=self.use_pdl,
            mx_out=mx_out,
            timestamps=timestamps,
            loopback=self.loopback,
            mx_atom=mx_atom,
        )
        return out


def fold_inputs_of(deferred, rows: torch.Tensor, stats: torch.Tensor):
    """``FoldInputs`` of the reduction a ``DeferredPackedReduce`` recorded for this decode (its
    output must be ``rows`` and its statistics ``stats``), or None if it recorded none. A recorded
    call on other buffers is launched instead (the caller then reads ``rows`` and ``stats``)."""
    from ..kernels.k3r_helix.k3_helix_tail import FoldInputs

    args = deferred.args
    if args is None:
        return None
    own_buffers = (
        args["output"].data_ptr() == rows.data_ptr()
        and args["output"].numel() == rows.numel()
        and args["stats"] is not None
        and args["stats"].data_ptr() == stats.data_ptr()
    )
    if not own_buffers:
        deferred.run()
        return None
    return FoldInputs(
        args["partial"],
        args["maxima"],
        args["sums"],
        args["plan"],
        args["q_scale"],
        args["kv_scale"],
        num_requests=args["num_requests"],
        valid_m=args["valid_m"],
        physical_m=args["physical_m"],
        sm_scale=args["sm_scale"],
        output_scale=args["output_scale"],
    )


# Largest generation batch (tokens) the fused query serves (the kernel takes up to 16).
_Q_FUSED_MAX_TOKENS = 16


class HelixGenHooks:
    """What the side-gate forward (``K3MLAAttention._forward_side_gate``) hands the Helix
    generation chain for one forward, and what the chain hands back.

    ``zero_kv_mask_fn`` returns the zero-local-KV mask computed ahead on the side stream (called
    where the exchange needs it); ``pre_exchange_fn`` forks the output gate; ``pre_tail_fn`` joins
    it before the fused tail and returns it (None if not forked yet). Back: ``tail_mx``, the MXFP8
    o_proj operand
    ``(xq, xs)`` the fused tail stored, and ``gen_output``, the combined all-to-all output."""

    __slots__ = (
        "zero_kv_mask_fn",
        "pre_exchange_fn",
        "pre_tail_fn",
        "tail_mx",
        "gen_output",
    )

    def __init__(
        self,
        zero_kv_mask_fn: Callable[[], Optional[torch.Tensor]],
        pre_exchange_fn: Callable[[], None],
        pre_tail_fn: Callable[[], Optional[torch.Tensor]],
    ) -> None:
        self.zero_kv_mask_fn = zero_kv_mask_fn
        self.pre_exchange_fn = pre_exchange_fn
        self.pre_tail_fn = pre_tail_fn
        self.tail_mx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.gen_output: Optional[torch.Tensor] = None


def helix_gen_zero_kv_mask(md: AttentionMetadata, num_tokens: int) -> Optional[torch.Tensor]:
    """Per-token mask of the ``num_tokens`` generation tokens this CP rank owns no KV for.

    With valid speculative verify groups KV ownership is per token (``helix_kv_bounds``);
    otherwise the per-sequence ``kv_lens_cuda == 0`` of the generation requests, expanded over
    their query lengths (``seq_lens_cuda``) with a static output size. None without those
    buffers."""
    helix_kv_bounds = getattr(md, "helix_kv_bounds", None)
    if helix_kv_bounds is not None and getattr(md, "_helix_spec_tokens_valid", False):
        return helix_kv_bounds[:num_tokens] == 0
    kv_lens = getattr(md, "kv_lens_cuda", None)
    seq_lens = getattr(md, "seq_lens_cuda", None)
    if kv_lens is None or seq_lens is None:
        return None
    seqs = slice(md.num_contexts, md.num_contexts + md.num_generations)
    return torch.repeat_interleave(kv_lens[seqs] == 0, seq_lens[seqs], output_size=num_tokens)


def helix_exchange(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    mapping,
    num_heads_tp_cp: int,
    value_dim: int,
    zero_kv_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """All-to-all of this rank's partials over the CP group (fifo v2 on the MNNVL workspace; the
    sender zeroes the ``zero_kv_mask`` rows) and the combine: ``[T, num_heads_tp_cp * value_dim]``
    BF16."""
    num_tokens = partial_o.shape[0]
    cp_size = mapping.cp_size
    partial_o_out, stats_out = HelixAllToAllNative.get(mapping).alltoall_native(
        partial_o.view(num_tokens, cp_size, num_heads_tp_cp * value_dim),
        softmax_stats.view(num_tokens, cp_size, num_heads_tp_cp * 2),
        None if zero_kv_mask is None else zero_kv_mask[:num_tokens],
    )
    return torch.ops.trtllm.helix_post_process_native(
        partial_o_out.view(num_tokens, cp_size, num_heads_tp_cp, value_dim),
        stats_out.view(num_tokens, cp_size, num_heads_tp_cp, 2),
        1.0,
        1,
    )


def project_helix_v_partials(layer, partial_o: torch.Tensor) -> torch.Tensor:
    """Every tp-local head's V up-projection of this CP rank's partials (it commutes with the
    per-head weighted combine): ``[T, H * kv_lora_rank]`` -> ``[T, H * v_head_dim]``."""
    num_tokens = partial_o.shape[0]
    heads = layer.num_heads_tp
    partial_v = partial_o.new_empty((num_tokens, heads, layer.v_head_dim))
    bmm_out(
        partial_o.view(num_tokens, heads, layer.kv_lora_rank).transpose(0, 1),
        layer.helix_v_b_proj_full.transpose(1, 2),
        partial_v.transpose(0, 1),
    )
    return partial_v.view(num_tokens, heads * layer.v_head_dim)


def tail_output_buffer(
    buf: Optional[torch.Tensor], num_tokens: int, cols: int
) -> Optional[torch.Tensor]:
    """``buf`` as the fused tail's BF16 ``[T, cols]`` output, or None where the tail cannot write
    it as is."""
    if (
        buf is None
        or buf.dtype != torch.bfloat16
        or not buf.is_contiguous()
        or buf.numel() != num_tokens * cols
    ):
        return None
    return buf.view(num_tokens, cols)


def helix_tail_applies(layer, num_tokens: int) -> bool:
    """Whether the Helix tail of this batch is one ``k3_helix_tail`` kernel (``helix_tail``)."""
    w_uv = layer.helix_v_b_proj_full
    return (
        isinstance(w_uv, torch.Tensor)
        and w_uv.dtype == torch.bfloat16
        and w_uv.is_contiguous()
        and K3HelixTail.applies(
            layer.mapping, num_tokens, layer.num_heads_tp, layer.kv_lora_rank, layer.v_head_dim
        )
    )


def gen_attention(layer, q: torch.Tensor, md: AttentionMetadata, **kwargs) -> torch.Tensor:
    """The absorbed decode of ``q`` (``[T, H * 576]``) into a new ``[T, H * kv_lora_rank]``: K3's
    packed FP4 decode on an FP4 cache with the CuTe DSL FP4 MLA backend (page plan and PDL
    reduction on); on an FP8 or BF16 cache the CuTe DSL MLA decode where it serves the batch, else
    the attention op's MLA generation (``ops/attention/k3_mla_attention``). ``kwargs``: the buffers of
    the generation cache update (``mla_rope_generation``) and ``softmax_stats_tensor``."""
    if fp4_mla_decode.applies(layer.mqa):
        assert kwargs.get("out_scale") is None
        return _decode_op.k3_fp4_mla_decode(
            layer.mqa,
            q,
            md,
            softmax_stats_tensor=kwargs.get("softmax_stats_tensor"),
            page_plan=True,
            reduce_pdl=True,
        )
    mqa = layer.mqa
    if mqa.uses_fp4_mla_attention:
        raise NotImplementedError(
            "Kimi K3 MLA: an FP4 MLA cache decodes on the CuTe DSL FP4 MLA backend (SM107)"
        )
    assert kwargs.pop("out_scale", None) is None
    softmax_stats = kwargs.pop("softmax_stats_tensor", None)
    output = q.new_empty((q.shape[0], mqa.num_heads * mqa.kv_lora_rank))
    return mla_attn.mla_decode(mqa, q, md, output, softmax_stats=softmax_stats, **kwargs)


def _helix_gen(
    layer,
    q: torch.Tensor,
    md: AttentionMetadata,
    helix_v_up: bool,
    hooks: Optional[HelixGenHooks],
    **kwargs,
) -> torch.Tensor:
    """Decode with softmax statistics, (V up-projection,) exchange and combine."""
    softmax_stats = torch.empty(
        (q.shape[0], layer.num_heads_tp, 2), device=q.device, dtype=torch.float32
    )
    kwargs["softmax_stats_tensor"] = softmax_stats
    partial_o = gen_attention(layer, q, md, **kwargs)
    assert partial_o.shape[-1] == layer.num_heads_tp * layer.kv_lora_rank
    if hooks is not None:
        zero_kv_mask = hooks.zero_kv_mask_fn()
    else:
        zero_kv_mask = helix_gen_zero_kv_mask(md, partial_o.shape[0])
    value_dim = layer.kv_lora_rank
    if helix_v_up:
        partial_o = project_helix_v_partials(layer, partial_o)
        value_dim = layer.v_head_dim
    if hooks is not None:
        hooks.pre_exchange_fn()
    return helix_exchange(
        partial_o, softmax_stats, layer.mapping, layer.num_heads_tp_cp, value_dim, zero_kv_mask
    )


def _helix_gen_fused_tail(
    layer,
    q: torch.Tensor,
    md: AttentionMetadata,
    output: Optional[torch.Tensor],
    hooks: Optional[HelixGenHooks],
    **kwargs,
) -> torch.Tensor:
    """The decode with its split reduction deferred, then one tail kernel (fold, V
    up-projection, exchange, combine) writing ``output`` where it can. With the side gate
    (``hooks``), forked before the decode and joined before the tail, the tail also stores
    o_proj's MXFP8 operand ``bf16(out * gate)`` where o_proj's gated MXFP8 GEMM takes it."""
    from ..kernels.k3r_fp4_mla.fp4_mla_packed_reduce import DeferredPackedReduce

    num_tokens = q.shape[0]
    softmax_stats = torch.empty(
        (num_tokens, layer.num_heads_tp, 2), device=q.device, dtype=torch.float32
    )
    kwargs["softmax_stats_tensor"] = softmax_stats
    if hooks is not None:
        hooks.pre_exchange_fn()
    with DeferredPackedReduce() as deferred:
        partial_o = gen_attention(layer, q, md, **kwargs)
    assert partial_o.shape[-1] == layer.num_heads_tp * layer.kv_lora_rank
    fold = fold_inputs_of(deferred, partial_o, softmax_stats)
    if hooks is not None:
        zero_kv_mask = hooks.zero_kv_mask_fn()
    else:
        zero_kv_mask = helix_gen_zero_kv_mask(md, num_tokens)
    gate = hooks.pre_tail_fn() if hooks is not None else None
    cols = layer.num_heads_tp_cp * layer.v_head_dim
    out = tail_output_buffer(output, num_tokens, cols)
    mx_out = None
    mx_applies = getattr(layer.o_proj, "gated_mxfp8_applies", None)
    if gate is not None and mx_applies is not None:
        if out is None:
            out = partial_o.new_empty((num_tokens, cols))
        if mx_applies(out, gate):
            xq = torch.empty((num_tokens, cols), dtype=torch.float8_e4m3fn, device=q.device)
            xs = torch.empty((num_tokens, cols // 32), dtype=torch.uint8, device=q.device)
            mx_out = (gate, xq, xs)
            hooks.tail_mx = (xq, xs)
    return K3HelixTail.get(layer.mapping)(
        partial_o,
        softmax_stats,
        layer.helix_v_b_proj_full,
        zero_kv_mask,
        fold=fold,
        out=out,
        mx_out=mx_out,
    )


def attn_forward_gen(
    layer,
    q: torch.Tensor,
    md: AttentionMetadata,
    *,
    helix_v_up: bool,
    output: Optional[torch.Tensor] = None,
    hooks: Optional[HelixGenHooks] = None,
    **kwargs,
) -> torch.Tensor:
    """The absorbed decode of ``q`` (``[T, H * 576]``), with the Helix exchange under Helix CP.

    Returns ``[T, H * kv_lora_rank]`` without Helix, ``[T, H / P * v_head_dim]`` with the V
    up-projection before the exchange (``helix_v_up``), ``[T, H / P * kv_lora_rank]`` otherwise.
    ``output``: the layer's buffer for the combined rows, which the fused tail writes."""
    num_tokens = q.shape[0]
    if helix_v_up and helix_tail_applies(layer, num_tokens):
        out = _helix_gen_fused_tail(layer, q, md, output, hooks, **kwargs)
    elif layer.mapping.has_cp_helix():
        out = _helix_gen(layer, q, md, helix_v_up, hooks, **kwargs)
    else:
        return gen_attention(layer, q, md, **kwargs)
    if helix_v_up and hooks is not None:
        hooks.gen_output = out
    return out


def forward_absorption_generation(
    layer,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    md: AttentionMetadata,
    output: torch.Tensor,
    hooks: Optional[HelixGenHooks] = None,
) -> torch.Tensor:
    """Absorbed generation of the ``q`` rows (``[T, H * 192]``, q_b's output) into ``output``
    (``[T, H / P * v_head_dim]``); ``latent_cache`` ``[T, 576]`` is the normalized latent and
    K-pe the cache update appends."""
    num_tokens = q.shape[0]
    heads, lora, rope = layer.num_heads_tp, layer.kv_lora_rank, layer.qk_rope_head_dim
    device = q.device
    q_nope, q_pe = q.view([-1, heads, layer.qk_head_dim]).split(
        [layer.qk_nope_head_dim, rope], dim=-1
    )
    num_seqs = md.num_seqs
    # The RoPE kernel writes seqQOffset[batch + 1] and scans total / per-sequence lengths over the
    # presented rows: one per token under the ragged (token-major) verify presentation.
    cu_rows = (
        max(num_seqs, num_tokens) + 1 if getattr(md, "is_ragged_verify", False) else num_seqs + 1
    )
    cu_q_seqlens = torch.empty(cu_rows, dtype=torch.int32, device=device)
    cu_kv_seqlens = torch.empty(cu_rows, dtype=torch.int32, device=device)
    fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)
    mla_bmm1_scale = mla_bmm2_scale = quant_q_buffer = None
    if layer.quantized_kv:
        mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
        mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
        quant_q_buffer = torch.empty(
            num_tokens, heads, lora + rope, dtype=torch.uint8, device=device
        )
    # [T, H, kv_lora_rank | rope]: the W_UK output and the rotated q_pe the RoPE kernel writes.
    fused_q = torch.empty([num_tokens, heads, lora + rope], dtype=q.dtype, device=device)
    rope_kwargs = {"fuse_fp4_q_quant": True} if layer.fp4_kv else {}

    def rope_and_cache_update():
        mla_cache.mla_rope_generation(
            layer.mqa,
            fused_q,
            q_pe,
            latent_cache,
            md,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            **rope_kwargs,
        )

    # [H, T, qk_nope] x [H, qk_nope, kv_lora_rank] -> [H, T, kv_lora_rank], written into fused_q.
    q_nope_out = fused_q[..., :lora].transpose(0, 1)
    if layer.k_b_proj_trans.dtype == torch.bfloat16:

        def w_uk():
            bmm_out(q_nope.transpose(0, 1), layer.k_b_proj_trans.transpose(1, 2), q_nope_out)

    else:
        w_uk_bf16 = layer.k_b_proj_trans_dequant
        if w_uk_bf16 is None:
            raise NotImplementedError(
                "Kimi K3 MLA: the FP8 absorption BMM runs on the dequantized weight (SM100f)"
            )

        def w_uk():
            torch.bmm(q_nope.transpose(0, 1), w_uk_bf16.transpose(1, 2), out=q_nope_out)

    # The quantized-cache RoPE kernel writes the quantized query too: same stream as the decode.
    rope_stream = None if layer.quantized_kv else layer.aux_stream
    fork_join(w_uk, rope_and_cache_update, layer.ln_events[0], layer.ln_events[1], rope_stream)
    fused_q = fused_q.view([num_tokens, heads * (lora + rope)])

    helix_v_up = layer.helix_v_up_before_exchange
    attn_out_latent = attn_forward_gen(
        layer,
        fused_q,
        md,
        helix_v_up=helix_v_up,
        output=output if helix_v_up else None,
        hooks=hooks,
        out_scale=None,
        latent_cache=latent_cache,
        q_pe=q_pe,
        cu_q_seqlens=cu_q_seqlens,
        cu_kv_seqlens=cu_kv_seqlens,
        fmha_scheduler_counter=fmha_scheduler_counter,
        mla_bmm1_scale=mla_bmm1_scale,
        mla_bmm2_scale=mla_bmm2_scale,
        quant_q_buffer=quant_q_buffer,
    )
    fused_q = None
    if helix_v_up:
        # The combine produced [T, H / P * v_head_dim]; the fused tail wrote it into output (the
        # copy is then a no-op on the same storage).
        output.view(attn_out_latent.shape).copy_(attn_out_latent)
        return output

    heads_out = layer.num_heads_tp_cp
    assert attn_out_latent.shape == (num_tokens, heads_out * lora)
    latent = attn_out_latent.view([-1, heads_out, lora])
    attn_output = output.view([num_tokens, heads_out, layer.v_head_dim])
    # [H, T, kv_lora_rank] x [H, kv_lora_rank, v_head_dim] -> [H, T, v_head_dim].
    if layer.v_b_proj.dtype == torch.bfloat16:
        bmm_out(latent.transpose(0, 1), layer.v_b_proj.transpose(1, 2), attn_output.transpose(0, 1))
    else:
        torch.bmm(
            latent.transpose(0, 1),
            layer.v_b_proj_dequant.transpose(1, 2),
            out=attn_output.transpose(0, 1),
        )
    return output


# ---------------------------------------------------------------------------------------------
# Fused query: generation-only Helix batches of FP4-KV layers.
# ---------------------------------------------------------------------------------------------


def _flashinfer_cute_norm(device: torch.device) -> bool:
    """Whether FlashInfer's ``rmsnorm`` runs its CuTe DSL ``RMSNormKernel`` on ``device`` (the
    arithmetic ``k3_mla_latent_row`` and ``k3_mla_q_fused`` reproduce), not its CUDA JIT kernel."""
    try:
        from flashinfer.norm import _use_cuda_norm
    except ImportError:
        return False
    return not _use_cuda_norm(device)


def q_fused_w_uk(layer) -> Optional[torch.Tensor]:
    """The BF16 absorption weight ``[H, kv_lora_rank, qk_nope_head_dim]`` the W_UK BMM reads (the
    weight, or its dequantized copy)."""
    w_kc = layer.k_b_proj_trans
    if w_kc is not None and w_kc.dtype != torch.bfloat16:
        w_kc = layer.k_b_proj_trans_dequant
    return w_kc


def q_fused_qb_exponents(layer) -> Optional[torch.Tensor]:
    """Block exponents of the q_b weight (``k3_mla_q_fused.block_exponents``), cached per weight
    scale; None if they would have to be built during CUDA graph capture."""
    scale = getattr(layer.q_b_proj, "weight_scale", None)
    if not isinstance(scale, torch.Tensor):
        return None
    cached = layer._q_fused_qb_exp
    if cached is not None and cached[0] is scale:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        return None
    from ..kernels.k3r_mla_q import k3_mla_q_fused

    weight = layer.q_b_proj.weight
    try:
        exponents = k3_mla_q_fused.block_exponents(scale, *weight.shape)
    except ValueError:
        return None
    layer._q_fused_qb_exp = (scale, exponents)
    return exponents


def q_fused_applies(layer, hidden_states: torch.Tensor, md: AttentionMetadata) -> bool:
    """Whether this forward takes :func:`forward_gen_q_fused`: a generation-only Helix batch of an
    FP4-KV layer (V up-projection before the exchange) whose query fits the fused kernel."""
    num_tokens = md.num_tokens
    if (
        hidden_states.dim() != 2
        or hidden_states.shape[0] != num_tokens
        or md.num_contexts != 0
        or md.num_generations <= 0
        or num_tokens % md.num_generations != 0
        or not layer.helix_v_up_before_exchange
        or not layer.mqa.uses_fp4_mla_attention
        or getattr(md, "helix_position_offsets", None) is None
        or getattr(md, "helix_is_inactive_rank", None) is None
        or layer.mqa.rotary_cos_sin is None
        or layer.mqa.rotary_cos_sin.dtype != torch.float32
        or not layer.mqa.rotary_cos_sin.is_contiguous()
    ):
        return False
    from ..kernels.k3r_mla_q import k3_mla_q_fused

    if not (
        layer.num_heads_tp == k3_mla_q_fused.NUM_HEADS
        and layer.q_lora_rank == k3_mla_q_fused.Q_LORA_RANK
        and layer.kv_lora_rank == k3_mla_q_fused.KV_LORA_RANK
        and layer.qk_nope_head_dim == k3_mla_q_fused.NOPE_DIM
        and layer.qk_rope_head_dim == k3_mla_q_fused.ROPE_DIM
        and num_tokens <= min(k3_mla_q_fused.MAX_TOKENS, _Q_FUSED_MAX_TOKENS)
    ):
        return False
    weight = layer.q_b_proj.weight
    w_uk = q_fused_w_uk(layer)
    if w_uk is None or not k3_mla_q_fused.weights_supported(weight, w_uk):
        return False
    return q_fused_qb_exponents(layer) is not None


def latent_row(layer, compressed_kv: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """``concat([kv_a RMSNorm(compressed_kv), k_pe], -1)``: FlashInfer's norm written straight into
    the row and ``k_pe`` copied beside it."""
    norm = layer.kv_a_layernorm
    if not (IS_FLASHINFER_AVAILABLE and compressed_kv.dtype == torch.bfloat16):
        return torch.concat(
            [rms_norm(compressed_kv, norm.weight, norm.variance_epsilon), k_pe], dim=-1
        )
    from flashinfer.norm import rmsnorm

    lora = compressed_kv.shape[-1]
    latent = torch.empty(
        (compressed_kv.shape[0], lora + k_pe.shape[-1]),
        dtype=compressed_kv.dtype,
        device=compressed_kv.device,
    )
    rmsnorm(
        compressed_kv,
        norm.weight,
        norm.variance_epsilon,
        out=latent[:, :lora],
        enable_pdl=USE_PDL,
    )
    latent[:, lora:].copy_(k_pe)
    return latent


def _q_fused_latent_row(layer, compressed_kv: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """:func:`latent_row` in one ``k3_mla_latent_row`` launch where FlashInfer's norm is its CuTe
    DSL kernel (whose arithmetic it repeats, bit for bit)."""
    from ..kernels.k3r_mla_q import k3_mla_q_fused
    from ..ops.attention.k3_mla_latent_row import k3_mla_latent_row

    norm = layer.kv_a_layernorm
    if (
        IS_FLASHINFER_AVAILABLE
        and _flashinfer_cute_norm(compressed_kv.device)
        and k3_mla_q_fused.latent_row_supported(compressed_kv, k_pe, norm.weight)
    ):
        return k3_mla_latent_row(
            compressed_kv,
            k_pe,
            norm.weight,
            norm.variance_epsilon,
            use_pdl=USE_PDL,
        )
    return latent_row(layer, compressed_kv, k_pe)


def forward_gen_q_fused(
    layer,
    hidden_states: torch.Tensor,
    md: AttentionMetadata,
    hooks: Optional[HelixGenHooks] = None,
) -> torch.Tensor:
    """Absorbed Helix generation of an FP4-KV layer with the fused query; returns the combined
    all-to-all output ``[T, H / P * v_head_dim]``.

    After the fused q_a / kv_a projection the KV half (``k3_mla_latent_row``, then the FP4 cache
    update without its query rows) runs on the aux stream (with multi-stream on) and the main
    stream runs ``k3_mla_q_fused`` (on the raw q_a rows with the RMSNorm inside, or after the
    separate q_a RMSNorm), which writes the FP4 query rows into the Q buffers the cache update
    recorded on the metadata. The decode joins both; it reads only those buffers (the BF16
    ``fused_q`` it is handed provides shapes)."""
    from tensorrt_llm._torch.attention.backends.fp4_mla.config import _fp4_mla_packed_rows_per_seq
    from tensorrt_llm._torch.attention.backends.fp4_mla.layout import _get_fp4_mla_q_global_scale

    from ..kernels.k3r_mla_q import k3_mla_q_fused as q_fused_kernel
    from ..ops.attention._k3_fp4_mla_cache import _prepare_fp4_mla_q_buffers
    from ..ops.attention.fp4_mla_kv_append import fp4_mla_kv_append
    from ..ops.attention.k3_mla_q_fused import k3_mla_q_fused

    num_tokens = md.num_tokens
    heads, lora, rope = layer.num_heads_tp, layer.kv_lora_rank, layer.qk_rope_head_dim
    device = hidden_states.device
    gen_len = num_tokens // md.num_generations
    # The Q buffers the KV half records on the metadata, created (zero-filled, once) on this
    # stream, which writes them.
    _prepare_fp4_mla_q_buffers(
        md, num_tokens, heads, device, rows_per_seq=_fp4_mla_packed_rows_per_seq(heads, gen_len)
    )
    q, compressed_kv, k_pe = layer.project_qkv_a(hidden_states).split(
        [layer.q_lora_rank, lora, rope], -1
    )
    kv_fork, kv_ready = layer._kv_events
    kv_stream = layer.aux_stream if do_multi_stream() and layer.aux_stream is not None else None
    if kv_stream is not None:
        kv_fork.record()
    with torch.cuda.stream(kv_stream) if kv_stream is not None else contextlib.nullcontext():
        if kv_stream is not None:
            kv_fork.wait()
        latent_cache = _q_fused_latent_row(layer, compressed_kv, k_pe)
        fp4_mla_kv_append(layer.mqa, latent_cache, md, heads)
        if kv_stream is not None:
            kv_ready.record()
    norm = layer.q_a_layernorm
    fuse_norm = (
        q_fused_kernel.NORM_FUSION_AVAILABLE
        and IS_FLASHINFER_AVAILABLE
        and _flashinfer_cute_norm(device)
        and norm.weight.dtype == torch.bfloat16
        and norm.weight.is_contiguous()
        and q_fused_kernel.input_supported(q)
    )
    if not fuse_norm:
        q = layer.q_a_norm(q)
    assert q_fused_kernel.input_supported(q), "the fused query reads BF16 q_a rows"
    state = md.fp4_mla_state
    rows_per_seq = int(state.prequantized_q_rows_per_seq or 0)
    k3_mla_q_fused(
        q,
        layer.q_b_proj.weight,
        q_fused_qb_exponents(layer),
        q_fused_w_uk(layer),
        _get_fp4_mla_q_global_scale(md, device),
        layer.mqa.rotary_cos_sin,
        md.helix_position_offsets,
        state.prequantized_q,
        state.prequantized_q_sf.view(torch.uint8),
        gen_len=gen_len,
        rows_per_seq=rows_per_seq or gen_len * heads,
        use_pdl=USE_PDL,
        norm=(norm.weight, norm.variance_epsilon) if fuse_norm else None,
    )
    if kv_stream is not None:
        kv_ready.wait()
    fused_q = torch.empty([num_tokens, heads, lora + rope], dtype=q.dtype, device=device)
    q_pe = fused_q[..., lora:]
    num_seqs = md.num_seqs
    attn_out = attn_forward_gen(
        layer,
        fused_q.view([num_tokens, heads * (lora + rope)]),
        md,
        helix_v_up=True,
        hooks=hooks,
        out_scale=None,
        latent_cache=latent_cache,
        q_pe=q_pe,
        cu_q_seqlens=torch.empty(num_seqs + 1, dtype=torch.int32, device=device),
        cu_kv_seqlens=torch.empty(num_seqs + 1, dtype=torch.int32, device=device),
        fmha_scheduler_counter=torch.empty(1, dtype=torch.uint32, device=device),
        mla_bmm1_scale=torch.empty(2, dtype=torch.float32, device=device),
        mla_bmm2_scale=torch.empty(1, dtype=torch.float32, device=device),
        quant_q_buffer=torch.empty(
            num_tokens, heads, lora + rope, dtype=torch.uint8, device=device
        ),
    )
    assert attn_out.shape == (num_tokens, layer.num_heads_tp_cp * layer.v_head_dim)
    return attn_out


# Generation-only Helix batches of at most this many rows (V up-projection before the
# all-to-all) take the side-gate forward: the zero-KV mask and the output gate
# sigmoid(g_proj(x)) run on a side stream, and o_proj gates its input in its GEMM.
_K3_MLA_GATE_SIDE_MAX_M = 128
# One side stream per device, shared by all MLA layers (their side work never overlaps).
_GATE_SIDE_STREAMS: dict[int, torch.cuda.Stream] = {}


def _gate_side_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _GATE_SIDE_STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index, priority=0)
        _GATE_SIDE_STREAMS[index] = stream
    return stream


# ---------------------------------------------------------------------------------------------
# The attention part under breakable CUDA graphs: the layer is found by its index string, so the
# break's frozen arguments are tensors and a string.
# ---------------------------------------------------------------------------------------------


def _registered_layer(layer_idx: str):
    """The step's attention metadata and the MLA layer registered under ``layer_idx``."""
    extra_attrs = get_model_extra_attrs()
    assert extra_attrs is not None, "Model extra attrs is not set"
    metadata_ref = extra_attrs.get("attention_metadata", None)
    assert metadata_ref is not None, "Attention metadata is not set"
    metadata = metadata_ref()
    assert isinstance(metadata, TrtllmAttentionMetadata), "the MLA metadata must be TRTLLM's"
    mla_layers = extra_attrs.get("mla_layers", None)
    assert mla_layers is not None, "Attention layer is not registered"
    layer_ref = mla_layers.get(layer_idx, None)
    assert layer_ref is not None, f"Cannot find attention layer for layer {layer_idx}"
    layer = layer_ref()
    assert isinstance(layer, K3MLAAttention), "the registered MLA layer must be K3's"
    return metadata, layer


def create_mla_outputs(hidden_states: torch.Tensor, layer_idx: str) -> torch.Tensor:
    """The attention output buffer of the registered layer ``layer_idx`` for these rows."""
    _, layer = _registered_layer(layer_idx)
    return layer.create_output(hidden_states)


def mla_custom_op_inplace(
    hidden_states: torch.Tensor, layer_idx: str, output: torch.Tensor
) -> None:
    """The attention part of the registered layer ``layer_idx`` into ``output``, on the step's
    metadata registered with it."""
    metadata, layer = _registered_layer(layer_idx)
    layer.run_attention(hidden_states, metadata, output)


_bcg_mla_custom_op_inplace = eager_on_graph(mla_custom_op_inplace)


# ---------------------------------------------------------------------------------------------
# K3MLAAttention.
# ---------------------------------------------------------------------------------------------


class K3MLAAttention(nn.Module):
    """Kimi K3 MLA (see the module docstring)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rms_norm_eps: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        max_position_embeddings: int = 8192,
        model_config: ModelConfig,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
        mapping_with_cp: Optional[Mapping] = None,
    ) -> None:
        super().__init__()
        config = model_config
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.max_position_embeddings = max_position_embeddings
        self.use_output_gate = use_output_gate
        # The checkpoint's separate q_a / kv_a rows load into one fused projection.
        self.fuse_qkv_a_proj = True
        self.out_scale = None
        if rms_norm_eps is None:
            rms_norm_eps = getattr(config.pretrained_config, "rms_norm_eps", None)
        if rms_norm_eps is None:
            rms_norm_eps = 1e-6
        self.kv_cache_dtype = config.extra_attrs.get("kv_cache_dtype", "auto")

        # Each projection's checkpoint recipe: FP8 128 x 128 block scales or BF16.
        projection_configs = config.quant_config_dict or {}

        def is_fp8(name: str) -> bool:
            algo = getattr(projection_configs.get(name, config.quant_config), "quant_algo", None)
            if algo not in (None, QuantAlgo.FP8_BLOCK_SCALES):
                raise ValueError(f"Kimi K3 MLA: {name} has unsupported quantization {algo}")
            return algo == QuantAlgo.FP8_BLOCK_SCALES

        fp8 = {
            name: is_fp8(name)
            for name in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj")
            + (("g_proj",) if use_output_gate else ())
        }
        self._reject_unsupported(config, mapping_with_cp, fp8)

        # Registered for the custom ops; a later stack (one-model speculative decoding) with the
        # same local layer index registers as "<layer_idx>_<n>".
        self.register_to_config = True
        registry = config.extra_attrs.setdefault("mla_layers", {})
        suffix = 0
        while self.layer_idx_str in registry:
            self.layer_idx_str = f"{layer_idx}_{suffix}"
            suffix += 1
        registry[self.layer_idx_str] = weakref.ref(self)

        # Head sharding: q_b / kv_b over TP, the post-exchange heads of o_proj / g_proj over
        # TP x CP (no attention DP: _reject_unsupported).
        self.mapping = mapping_with_cp if mapping_with_cp is not None else config.mapping
        tp_size, pp_size, cp_size = self.mapping.tp_size, self.mapping.pp_size, self.mapping.cp_size
        world_size = pp_size * tp_size * cp_size
        mapping = Mapping(
            world_size=world_size,
            tp_size=tp_size,
            pp_size=pp_size,
            cp_size=cp_size,
            cp_config=self.mapping.cp_config,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=False,
        )
        self.mapping_o = Mapping(
            world_size=world_size,
            tp_size=tp_size * cp_size,
            pp_size=pp_size,
            cp_size=1,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=False,
        )
        assert num_heads % (tp_size * cp_size) == 0
        self.num_heads_tp = num_heads // tp_size
        self.num_heads_tp_cp = self.num_heads_tp // cp_size
        # Helix context attends this CP rank's head range only.
        self.num_context_heads = (
            self.num_heads_tp_cp if self.mapping.has_cp_helix() else self.num_heads_tp
        )

        self.aux_stream = (aux_stream_dict or {}).get(AuxStreamType.Attention)
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        def projection(fp8: bool, in_features: int, out_features: int, mode, mp: Optional[Mapping]):
            shard = {} if mp is None else {"tp_size": mp.tp_size, "tp_rank": mp.tp_rank}
            if fp8:
                return Fp8BlockScaleLinear(in_features, out_features, tp_mode=mode, **shard)
            return Bf16Linear(in_features, out_features, dtype=dtype, tp_mode=mode, **shard)

        column, row = TensorParallelMode.COLUMN, TensorParallelMode.ROW
        self.kv_a_proj_with_mqa = projection(
            fp8["kv_a_proj_with_mqa"],
            hidden_size,
            q_lora_rank + kv_lora_rank + qk_rope_head_dim,
            None,
            None,
        )
        if fp8["kv_a_proj_with_mqa"]:
            # q_lora + kv_lora + rope rows (2112 for K3) leave a partial last 128-row tile,
            # which the FP8 low-M GEMM serves only on request.
            self.kv_a_proj_with_mqa.fp8w_lowm_partial_n_tile = True
        self.q_a_layernorm = RMSNormWeight(q_lora_rank, rms_norm_eps, dtype=dtype)
        self.q_b_proj = projection(
            fp8["q_b_proj"], q_lora_rank, num_heads * self.qk_head_dim, column, mapping
        )
        self.kv_a_layernorm = RMSNormWeight(kv_lora_rank, rms_norm_eps, dtype=dtype)
        self.kv_b_proj = projection(
            fp8["kv_b_proj"],
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            column,
            mapping,
        )
        self._kv_b_fp8 = isinstance(self.kv_b_proj, Fp8BlockScaleLinear)
        # This CP rank's post-exchange heads of the V absorption (bound by the loader).
        self.v_b_proj = nn.Parameter(
            torch.empty((self.num_heads_tp_cp, v_head_dim, kv_lora_rank), dtype=dtype),
            requires_grad=False,
        )
        # Helix only: the BF16 V weight of every tp-local head, applied to this rank's partials
        # before the all-to-all (``project_helix_v_partials``).
        self.register_buffer("helix_v_b_proj_full", None, persistent=False)
        self.helix_v_up_before_exchange = False
        self.o_proj = projection(
            fp8["o_proj"], num_heads * v_head_dim, hidden_size, row, self.mapping_o
        )
        self.attention_output_hidden_size = self.o_proj.in_features
        if use_output_gate:
            # o_proj's input sharding (under Helix the post-exchange 1/cp head chunk).
            self.g_proj = projection(
                fp8["g_proj"], hidden_size, num_heads * v_head_dim, column, self.mapping_o
            )

        quant_config = config.get_quant_config()
        self.quant_config = quant_config
        # The two attention roles over the engine's KV cache (``ops/attention/k3_mla_state``): the
        # absorbed generation (``mqa``: one 576-dim latent head per query head) and the context
        # (``mha``: 192 / 128-dim heads), each with the identity RoPE table (K3 is NoPE).
        role_dims = dict(
            layer_idx=layer_idx,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            skip_correction_threshold=config.skip_correction_threshold,
        )
        self.mqa = K3MlaAttnState.generation(
            num_heads=self.num_heads_tp,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            **role_dims,
        )
        self.mha = K3MlaAttnState.context(
            num_context_heads=self.num_context_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            **role_dims,
        )

        # Set in create_weights() from the backends' KV cache quantization.
        self.quantized_kv = False
        self.fp4_kv = False
        self.fp8_cached_kv_context = False
        self._weights_transformed = False
        # Side-gate forward: fork / join of the zero-KV mask and the gate on the side stream.
        self._gate_events = tuple(torch.cuda.Event() for _ in range(4))
        # Fused query: fork of the KV half after q_a, join before the decode.
        self._kv_events = (torch.cuda.Event(), torch.cuda.Event())
        # Set by the decoder layer for the next forward only: the MXFP8 form ``(xq, xs)`` of the
        # layer input, which the fused q_a / kv_a projection consumes (see q_a_takes_mxfp8).
        self.next_input_mxfp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # Fused query: (q_b weight scale, its block exponents), built on first use.
        self._q_fused_qb_exp: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if not config.skip_create_weights_in_init:
            self.create_weights()

    @staticmethod
    def _reject_unsupported(
        config: ModelConfig, mapping_with_cp: Optional[Mapping], fp8: dict[str, bool]
    ) -> None:
        """Configurations whose kernel chain K3 does not carry."""
        mapping = mapping_with_cp if mapping_with_cp is not None else config.mapping
        unsupported = []
        if config.attn_backend.upper() != "TRTLLM":
            unsupported.append(f"attention backend {config.attn_backend} (TRTLLM only)")
        if config.sparse_attention_config is not None:
            unsupported.append("sparse attention")
        if fp8["q_a_proj"] != fp8["kv_a_proj_with_mqa"]:
            unsupported.append("q_a / kv_a with different recipes (no fused q_a / kv_a GEMM)")
        if config.extra_attrs.get("kv_cache_dtype", "auto") == "fp8_ds_mla":
            unsupported.append("the fp8_ds_mla KV cache")
        if hasattr(config.pretrained_config, "llama_4_scaling"):
            unsupported.append("Llama-4 attention scaling")
        # The CuTe DSL GEMM / BMM opt-ins of main's MLA, where they would select another kernel:
        # BF16 projections, BF16 absorption BMMs (BF16 kv_b, or the Helix V up-projection), FP8
        # absorption BMMs.
        if config.use_cute_dsl_bf16_gemm and not all(fp8.values()):
            unsupported.append("use_cute_dsl_bf16_gemm with BF16 projections")
        if config.use_cute_dsl_bf16_bmm and (mapping.has_cp_helix() or not fp8["kv_b_proj"]):
            unsupported.append("use_cute_dsl_bf16_bmm")
        if config.use_cute_dsl_blockscaling_bmm and fp8["kv_b_proj"]:
            unsupported.append("use_cute_dsl_blockscaling_bmm")
        if mapping.has_cp_ulysses():
            unsupported.append("Ulysses CP")
        if mapping.cp_size > 1 and not mapping.has_cp_helix():
            unsupported.append(f"CP type {mapping.cp_config.get('cp_type')}")
        if mapping.enable_attention_dp:
            unsupported.append("attention DP (the helix8 targets run none)")
        if mapping.has_cp_helix():
            if mapping.cp_config.get("use_nccl_for_alltoall", True):
                unsupported.append("the NCCL Helix all-to-all (use_nccl_for_alltoall: false)")
            elif mapping.cp_config.get("fifo_version", 2) != 2:
                unsupported.append("the fifo-v1 Helix all-to-all")
        if unsupported:
            raise NotImplementedError("Kimi K3 MLA does not support: " + "; ".join(unsupported))

    def create_weights(self) -> None:
        """Backend KV cache quantization and the absorption weights (their dtype follows kv_b)."""
        self.mha.update_quant_config(self.quant_config)
        self.mqa.update_quant_config(self.quant_config)
        self.quantized_kv = bool(self.mqa.has_fp8_kv_cache or self.mqa.has_fp4_kv_cache)
        self.fp4_kv = bool(
            self.mqa.support_fp4_kv_cache()
            and self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_fp4_kv_cache()
        )
        if self.fp4_kv and not self.mqa.rope_duplicate_data:
            raise ValueError("FP4 MLA requires RopeParams.duplicate_data=True.")
        # Cached-prefix context on the FP8 context FMHA takes FP8 Q / K / V (plain conversions at
        # the op's unit KV scale).
        mha = self.mha
        scale = mha.kv_scale_orig_quant
        self.fp8_cached_kv_context = bool(
            mha.has_fp8_context_mla
            and not self.mapping.has_cp_helix()
            and (self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim) == (128, 64, 128)
            and isinstance(scale, torch.Tensor)
            and bool(torch.all(scale == 1.0).item())
        )
        heads, heads_cp = self.num_heads_tp, self.num_heads_tp_cp
        lora, nope, vdim = self.kv_lora_rank, self.qk_nope_head_dim, self.v_head_dim
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (heads, lora, nope), dtype=torch.float8_e4m3fn if self._kv_b_fp8 else self.dtype
            ),
            requires_grad=False,
        )
        self.k_b_proj_trans_dequant = None
        self.v_b_proj_dequant = None
        if self._kv_b_fp8:
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty((heads, lora // 128, nope // 128), dtype=torch.float32),
                requires_grad=False,
            )
            self.v_b_proj_scale = nn.Parameter(
                torch.empty((heads_cp, vdim // 128, lora // 128), dtype=torch.float32),
                requires_grad=False,
            )
            if is_sm_100f():
                # The absorption BMMs run in BF16 on dequantized copies.
                assert self.dtype == torch.bfloat16
                self.k_b_proj_trans_dequant = nn.Parameter(
                    torch.empty((heads, lora, nope), dtype=self.dtype), requires_grad=False
                )
                self.v_b_proj_dequant = nn.Parameter(
                    torch.empty((heads_cp, vdim, lora), dtype=self.dtype), requires_grad=False
                )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None
        self._weights_transformed = False

    # -----------------------------------------------------------------------------------------
    # Post-load.
    # -----------------------------------------------------------------------------------------

    def _bind_helix_v_up_weight(self, weight: torch.Tensor) -> None:
        """Apply the V up-projection with ``weight`` (BF16 ``[num_heads_tp, v_head_dim,
        kv_lora_rank]``) before the Helix all-to-all."""
        assert self.mapping.has_cp_helix()
        assert weight.dtype == torch.bfloat16
        assert tuple(weight.shape) == (self.num_heads_tp, self.v_head_dim, self.kv_lora_rank)
        self.helix_v_b_proj_full = weight
        self.helix_v_up_before_exchange = True

    def _build_helix_v_up_weight(self) -> None:
        """Under Helix, every tp-local head's BF16 V weight from kv_b's loaded rows (BF16, or the
        raw FP8 codes with their FP32 128 x 128 block scales: this runs before kv_b's own
        transform, which the post-load walk visits after this module)."""
        if not self.mapping.has_cp_helix() or self.dtype != torch.bfloat16:
            return
        weight = self.kv_b_proj.weight
        nope_rows = self.num_heads_tp * self.qk_nope_head_dim
        num_rows = nope_rows + self.num_heads_tp * self.v_head_dim
        v_shape = (self.num_heads_tp, self.v_head_dim, self.kv_lora_rank)
        if weight.is_meta or tuple(weight.shape) != (num_rows, self.kv_lora_rank):
            return
        if weight.dtype == torch.bfloat16:
            v_weight = weight[nope_rows:].view(v_shape)
        elif weight.dtype == torch.float8_e4m3fn:
            scale = self.kv_b_proj.weight_scale
            if scale.dtype != torch.float32 or tuple(scale.shape) != (
                -(-num_rows // 128),
                -(-self.kv_lora_rank // 128),
            ):
                return
            row_blocks = torch.arange(nope_rows, num_rows, device=weight.device) // 128
            v_scale = scale[row_blocks].repeat_interleave(128, dim=1)[:, : self.kv_lora_rank]
            v_weight = (weight[nope_rows:].float() * v_scale).to(self.dtype).view(v_shape)
            if self.v_b_proj_dequant is not None:
                # This rank's dequantized V chunk holds the same values.
                start = self.mapping.cp_rank * self.num_heads_tp_cp
                self.v_b_proj_dequant = nn.Parameter(
                    v_weight[start : start + self.num_heads_tp_cp], requires_grad=False
                )
        else:
            return
        self._bind_helix_v_up_weight(v_weight)

    def transform_weights(self) -> None:
        if self._weights_transformed:
            return
        self._build_helix_v_up_weight()
        self._weights_transformed = True

    def cache_derived_state(self) -> None:
        """The scales of the q_b / kv_b output slices a Helix rank's context heads read."""
        if self.num_context_heads != self.num_heads_tp:
            first_head = self.mapping.cp_rank * self.num_context_heads
            last_head = first_head + self.num_context_heads
            value_start = self.num_heads_tp * self.qk_nope_head_dim
            for proj, start, end in (
                (self.q_b_proj, first_head * self.qk_head_dim, last_head * self.qk_head_dim),
                (
                    self.kv_b_proj,
                    first_head * self.qk_nope_head_dim,
                    last_head * self.qk_nope_head_dim,
                ),
                (
                    self.kv_b_proj,
                    value_start + first_head * self.v_head_dim,
                    value_start + last_head * self.v_head_dim,
                ),
            ):
                if not isinstance(proj, Bf16Linear):
                    proj.prepare_output_slice(start, end)
        self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self.cache_derived_state()

    # -----------------------------------------------------------------------------------------
    # Front.
    # -----------------------------------------------------------------------------------------

    def q_a_mx_atom(self, num_rows: int) -> bool:
        """Whether the fused q_a / kv_a GEMM takes the MXFP8 rows with the scales in the mid-M
        GEMM's layout (``takes_mid_mxfp8_input``)."""
        takes_mid = getattr(self.kv_a_proj_with_mqa, "takes_mid_mxfp8_input", None)
        return takes_mid is not None and bool(takes_mid(num_rows))

    def q_a_takes_mxfp8(self, num_rows: int) -> bool:
        """Whether the fused q_a / kv_a GEMM of ``num_rows`` rows can take their MXFP8 form
        (``next_input_mxfp8``)."""
        takes = getattr(self.kv_a_proj_with_mqa, "takes_mxfp8_input", None)
        return takes is not None and (bool(takes(num_rows)) or self.q_a_mx_atom(num_rows))

    def project_qkv_a(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The fused q_a / kv_a projection ``[T, q_lora | kv_lora | rope]``: on the MXFP8 rows the
        decoder layer handed over for these rows (``forward_mxfp8``), else the GEMM on the BF16
        rows (a BF16 weight: ``cublas_mm``)."""
        mxfp8_input, self.next_input_mxfp8 = self.next_input_mxfp8, None
        proj = self.kv_a_proj_with_mqa
        if isinstance(proj, Bf16Linear):
            return cublas_mm(
                hidden_states,
                proj.weight.t(),
                None,
                out_dtype=None,
                output_buffer_kind=int(BufferKind.DEFAULT),
                group=None,
            )
        if mxfp8_input is not None and mxfp8_input[0].shape[0] == hidden_states.shape[0]:
            return proj.forward_mxfp8(hidden_states, *mxfp8_input)
        return proj(hidden_states)

    def q_a_norm(self, q: torch.Tensor) -> torch.Tensor:
        """The q_a RMSNorm (FlashInfer); with PDL the same kernel releasing its dependent (q_b) at
        launch instead of after its store, so q_b streams its weights while q_a and the norm run."""
        norm = self.q_a_layernorm
        if IS_FLASHINFER_AVAILABLE and USE_PDL:
            from ..kernels.k3r_mla_q.q_norm_pdl import can_use
            from ..ops.attention.rmsnorm_release_at_launch import rmsnorm_release_at_launch

            if can_use(q, norm.weight):
                return rmsnorm_release_at_launch(q, norm.weight, norm.variance_epsilon)
        return rms_norm(q, norm.weight, norm.variance_epsilon)

    @staticmethod
    def forward_slice(proj, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Output features ``[start, end)`` of a column-parallel projection, contiguous."""
        if isinstance(proj, Bf16Linear):
            return torch.nn.functional.linear(x, proj.weight[start:end])
        return proj.forward_slice(x, start, end)

    def _project_q_by_phase(
        self, q: torch.Tensor, num_ctx_tokens: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """q_b of the context rows (a Helix rank's context heads only) and of the generation rows."""
        if self.num_context_heads == self.num_heads_tp or num_ctx_tokens == 0:
            projected = self.q_b_proj(q)
            return projected[:num_ctx_tokens], projected[num_ctx_tokens:]
        first = self.mapping.cp_rank * self.num_context_heads * self.qk_head_dim
        last = first + self.num_context_heads * self.qk_head_dim
        ctx = self.forward_slice(self.q_b_proj, q[:num_ctx_tokens], first, last)
        gen = self.q_b_proj(q[num_ctx_tokens:]) if q.shape[0] > num_ctx_tokens else None
        return ctx, gen

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        md: AttentionMetadata,
        output: torch.Tensor,
        hooks: Optional[HelixGenHooks],
    ) -> None:
        """Front, context and generation attention of the batch's rows into ``output``."""
        num_ctx_tokens = md.num_ctx_tokens
        num_tokens = md.num_tokens
        hidden_states = hidden_states[:num_tokens, ...]
        q, compressed_kv, k_pe = self.project_qkv_a(hidden_states).split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], -1
        )
        kv_norm = self.kv_a_layernorm
        q, compressed_kv = fork_join(
            lambda: self.q_a_norm(q),
            lambda: rms_norm(compressed_kv, kv_norm.weight, kv_norm.variance_epsilon),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        (q_ctx, q_gen), latent_cache = fork_join(
            lambda: self._project_q_by_phase(q, num_ctx_tokens),
            lambda: torch.concat([compressed_kv, k_pe], dim=-1),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        if md.num_contexts > 0:
            forward_context(
                self,
                q_ctx,
                compressed_kv[:num_ctx_tokens, ...],
                md,
                output[:num_ctx_tokens, :],
                latent_cache[:num_ctx_tokens, ...],
            )
        if md.num_generations > 0:
            assert q_gen is not None
            forward_absorption_generation(
                self,
                q_gen,
                latent_cache[num_ctx_tokens:, ...],
                md,
                output[num_ctx_tokens:num_tokens, :],
                hooks,
            )

    def run_attention(
        self,
        hidden_states: torch.Tensor,
        md: AttentionMetadata,
        output: torch.Tensor,
        hooks: Optional[HelixGenHooks] = None,
    ) -> None:
        """The attention part of a forward (everything before the output gate) into ``output``,
        on the mixed-precision KV staging copy where the metadata has one."""
        with md.mixed_precision_kv_staging(self.layer_idx):
            self._forward_attention(hidden_states, md, output, hooks)

    def create_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The attention output ``[rows, H / P * v_head_dim]`` for the layer input's rows."""
        return hidden_states.new_empty(
            [hidden_states.shape[0], self.attention_output_hidden_size], dtype=hidden_states.dtype
        )

    # -----------------------------------------------------------------------------------------
    # Forward.
    # -----------------------------------------------------------------------------------------

    def _gate_and_o_proj(self, hidden_states: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        if self.use_output_gate:
            attn_out = attn_out * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attn_out)

    def _side_gate_applies(self, hidden_states: torch.Tensor, md: AttentionMetadata) -> bool:
        """Whether this forward takes :meth:`_forward_side_gate`."""
        return (
            self.use_output_gate
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and hidden_states.shape[0] == md.num_tokens
            and md.num_contexts == 0
            and self.helix_v_up_before_exchange
            and hasattr(self.o_proj, "gated_applies")
            and hidden_states.shape[0] <= _K3_MLA_GATE_SIDE_MAX_M
        )

    def _forward_side_gate(
        self, hidden_states: torch.Tensor, md: AttentionMetadata
    ) -> torch.Tensor:
        """Generation forward with the zero-KV mask and the output gate on a side stream.

        The mask (one small kernel) is forked once the layer input is written; the gate (one GEMM
        with a sigmoid epilogue) is forked when the all-to-all is enqueued (with the fused Helix
        tail: right before the decode), so it runs on the SMs the all-to-all leaves idle. The main
        stream waits for the mask where the exchange reads it and for the gate before o_proj; with
        the fused Helix tail, before the tail kernel, so o_proj depends on the tail alone and
        gates its input in its GEMM (``forward_gated(_mxfp8)``)."""
        num_tokens = hidden_states.shape[0]
        mask_fork, mask_ready, gate_fork, gate_ready = self._gate_events
        multi_stream = do_multi_stream()
        side = _gate_side_stream(hidden_states.device) if multi_stream else None
        gates = []
        mask_joined = [not multi_stream]
        gate_joined = [not multi_stream]

        def gate_of(x):
            forward_sigmoid = getattr(self.g_proj, "forward_sigmoid", None)
            if forward_sigmoid is not None:
                return forward_sigmoid(x)
            return self.g_proj(x).sigmoid()

        if multi_stream:
            mask_fork.record()
            with torch.cuda.stream(side):
                mask_fork.wait()
                zero_kv_mask = helix_gen_zero_kv_mask(md, num_tokens)
                mask_ready.record()
        else:
            zero_kv_mask = helix_gen_zero_kv_mask(md, num_tokens)

        def zero_kv_mask_fn():
            if not mask_joined[0]:
                mask_ready.wait()
                mask_joined[0] = True
            return zero_kv_mask

        def pre_exchange():
            if multi_stream:
                gate_fork.record()
                with torch.cuda.stream(side):
                    gate_fork.wait()
                    gates.append(gate_of(hidden_states))
                    gate_ready.record()
            else:
                gates.append(gate_of(hidden_states))

        def pre_tail():
            if gates and not gate_joined[0]:
                gate_ready.wait()
                gate_joined[0] = True
            return gates[0] if gates else None

        hooks = HelixGenHooks(zero_kv_mask_fn, pre_exchange, pre_tail)
        if q_fused_applies(self, hidden_states, md):
            attn_out = forward_gen_q_fused(self, hidden_states, md, hooks)
        else:
            output = self.create_output(hidden_states)
            self.run_attention(hidden_states, md, output, hooks)
            attn_out = hooks.gen_output if hooks.gen_output is not None else output
        if not mask_joined[0]:
            mask_ready.wait()
        if gates:
            gate = gates[0]
            if not gate_joined[0]:
                gate_ready.wait()
        else:
            gate = gate_of(hidden_states)
        if hooks.tail_mx is not None:
            return self.o_proj.forward_gated_mxfp8(attn_out, gate, *hooks.tail_mx)
        if self.o_proj.gated_applies(attn_out, gate):
            return self.o_proj.forward_gated(attn_out, gate)
        return self.o_proj(attn_out * gate)

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """o_proj's partial output of the layer for ``hidden_states`` (K3 is NoPE:
        ``position_ids`` is None)."""
        assert position_ids is None, "Kimi K3 MLA is NoPE"
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"Kimi K3 MLA takes BF16 rows, got {type(hidden_states).__name__}")
        try:
            if self._side_gate_applies(hidden_states, attn_metadata):
                return self._forward_side_gate(hidden_states, attn_metadata)
            if q_fused_applies(self, hidden_states, attn_metadata):
                attn_out = forward_gen_q_fused(self, hidden_states, attn_metadata)
                return self._gate_and_o_proj(hidden_states, attn_out)
            if self.register_to_config:
                output = create_mla_outputs(hidden_states, self.layer_idx_str)
                _bcg_mla_custom_op_inplace(hidden_states, self.layer_idx_str, output)
            else:
                output = self.create_output(hidden_states)
                self.run_attention(hidden_states, attn_metadata, output)
            return self._gate_and_o_proj(hidden_states, output)
        finally:
            # Consumed by the projection; never carried into another forward.
            self.next_input_mxfp8 = None


# Identity-RoPE table positions for the MLA backends: max_position_embeddings. K3 is NoPE (the
# table holds cos=1/sin=0), but the chunked-context path indexes the table by absolute position,
# so it must cover max_position_embeddings (~512MB per backend for the 1M-position checkpoint); a
# smaller table is read out of bounds.


def _kimi_k3_fp8_kv_layer_ids(cfg, kv_quant_algo: Optional[QuantAlgo]) -> frozenset:
    """MLA layers whose KV cache falls back to FP8 under NVFP4 KV: the engine's cache policy
    decides (``kimi_k3_cache_policy.get_kimi_k3_fp8_kv_layer_ids``, which the FP4 MLA cache
    manager reads too, so the layer and its cache agree; under NVFP4 the last two MLA layers)."""
    return _kimi_k3_cache_policy.get_kimi_k3_fp8_kv_layer_ids(
        cfg, nvfp4_enabled=kv_quant_algo == QuantAlgo.NVFP4
    )


def resolve_attention_quant_config(
    config: ModelConfig | None, layer_idx: int, projection: str
) -> QuantConfig:
    """Resolve a checkpoint projection, including mixed-precision exclusions."""
    if config is None:
        return QuantConfig()
    global_config = config.quant_config or QuantConfig()
    names = [
        f"{prefix}layers.{layer_idx}.self_attn.{projection}"
        for prefix in ("language_model.model.", "model.", "")
    ]
    if any(global_config.is_module_excluded_from_quantization(name) for name in names):
        return QuantConfig(kv_cache_quant_algo=global_config.kv_cache_quant_algo)
    declarations = config.quant_config_dict or {}
    matches = [declarations[name] for name in names if name in declarations]
    if matches:
        selected = matches[0]
        if any(match.quant_algo != selected.quant_algo for match in matches[1:]):
            raise ValueError(f"Conflicting Kimi K3 quantization aliases for {names[0]}")
    elif global_config.quant_algo == QuantAlgo.MIXED_PRECISION:
        selected = QuantConfig()
    else:
        selected = global_config
    if selected.quant_algo not in (None, QuantAlgo.FP8_BLOCK_SCALES):
        raise ValueError(
            f"Kimi K3 attention projection {names[0]} has unsupported checkpoint "
            f"quantization {selected.quant_algo}"
        )
    if selected.quant_algo == QuantAlgo.FP8_BLOCK_SCALES and selected.group_size not in (None, 128):
        raise ValueError(f"Kimi K3 attention requires 128x128 FP8 blocks for {names[0]}")
    result = copy.copy(selected)
    result.kv_cache_quant_algo = global_config.kv_cache_quant_algo
    return result


# ---------------------------------------------------------------------------
# MLA runtime.
# ---------------------------------------------------------------------------


class K3MLARuntime(nn.Module):
    """Wraps K3 MLA and applies its external TP output reduction."""

    def __init__(
        self,
        cfg: "PretrainedConfig",
        layer_idx: int,
        model_config: ModelConfig,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()

        max_positions = int(cfg.max_position_embeddings)
        self.layer_idx = layer_idx
        # K3MLAAttention owns MLA projection/head sharding. Keep only the
        # final output reduction in this wrapper so the output gate remains
        # between attention and the row-parallel o_proj.
        # Helix: mapping_with_cp (the CP original) activates the base MLA's
        # helix machinery; this wrapper's allreduce over the repurposed
        # mapping sums the base o_proj's tp*cp partials. reduce_output=False
        # leaves that reduction to the caller (helix_a2a reduce-scatter).
        mapping = model_config.mapping.mapping_for_full_rows()
        if mapping.enable_attention_dp:
            raise NotImplementedError("Kimi K3 helix8 MLA: attention DP (the targets run none)")
        reduce_output = reduce_output and mapping.tp_size > 1
        self._o_allreduce = (
            TpAllReduce(mapping, torch.bfloat16, model_config.allreduce_strategy)
            if reduce_output
            else None
        )
        # Set by the decoder layer for one forward: a pushed o_proj all-reduce may be returned
        # unfinished, as a LamportPending that A_mid finishes.
        self.ar_finish_deferred = False
        attention_config = copy.copy(model_config)
        attention_config._frozen = False
        attention_config.quant_config_dict = {
            name: resolve_attention_quant_config(model_config, layer_idx, name)
            for name in (
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "q_b_proj",
                "kv_b_proj",
                "g_proj",
                "o_proj",
            )
        }
        kv_quant_algo = (
            model_config.quant_config.kv_cache_quant_algo
            if model_config.quant_config is not None
            else None
        )
        fp8_kv_layer_ids = _kimi_k3_fp8_kv_layer_ids(cfg, kv_quant_algo)
        if layer_idx in fp8_kv_layer_ids:
            if kv_quant_algo != QuantAlgo.NVFP4:
                raise ValueError(
                    "The Kimi K3 FP8 KV-cache fallback layers require the model-wide KV cache "
                    "dtype to be NVFP4."
                )
            kv_quant_algo = QuantAlgo.FP8
            logger.info_once(
                f"Kimi K3 MLA layers using FP8 KV-cache fallback: {sorted(fp8_kv_layer_ids)}.",
                key="kimi_k3_fp8_kv_fallback_layers",
            )
            for projection_config in attention_config.quant_config_dict.values():
                projection_config.kv_cache_quant_algo = kv_quant_algo
        attention_config.quant_config = QuantConfig(kv_cache_quant_algo=kv_quant_algo)
        attention_config._frozen = model_config._frozen
        self.mixer = K3MLAAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            use_output_gate=cfg.mla_use_output_gate,
            max_position_embeddings=max_positions,
            model_config=attention_config,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
        )

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        o_proj = self.mixer.o_proj
        # o_proj stores its partial straight into the all-reduce's Lamport buffers where the
        # low-M GEMM serves it.
        ar_push = None
        if self._o_allreduce is not None and hasattr(o_proj, "next_ar_push"):
            ar_push = lamport_push_target(
                self._o_allreduce,
                hidden_states.shape[0],
                o_proj.out_features,
                deferred=self.ar_finish_deferred,
            )
            o_proj.next_ar_push = ar_push
        # MLA.forward takes position_ids first; K3 is NoPE, so pass None.
        out = self.mixer(None, hidden_states, attn_metadata)
        if ar_push is not None:
            o_proj.next_ar_push = None
        if ar_push is not None and ar_push.pushed:
            # A_mid may finish the all-reduce (a LamportPending).
            out_shape = out.shape
            out = finish_lamport_push(self._o_allreduce, ar_push, deferred=self.ar_finish_deferred)
            if not isinstance(out, LamportPending):
                out = out.view(out_shape)
        elif ar_push is not None and ar_push.deferred:
            # o_proj ran without the push epilogue; A_mid finishes.
            lamport_push_tensor(self._o_allreduce, ar_push, out.reshape(ar_push.output.shape))
            out = finish_lamport_push(self._o_allreduce, ar_push, deferred=True)
        elif self._o_allreduce is not None:
            # Head-sharded TP: sum the row-sharded o_proj partials across
            # the head-shard group.
            out = self._o_allreduce(out)
        return out


# ---------------------------------------------------------------------------
# FP8 block-scale weight read for attention and replicated MoE-layer MLP projections.
# ---------------------------------------------------------------------------


def _rubin_mxfp8_is_available() -> bool:
    """Whether this rank can run the SM107 native block-scaled MXFP8 GEMM.

    Recomputed rather than cached: the SM version cannot change within a run,
    so weight preparation and forward always agree.
    """
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def _is_fp8_block_projection(module: nn.Module) -> bool:
    """A checkpoint FP8 128 x 128 block-scale projection: the loader reads its codes and scales
    together and post-load swaps it for ``_Fp8BlockScaleWeightReadLinear``."""
    return isinstance(module, Fp8BlockScaleLinear)


def _k3_lowm_gemm_can_run(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    allow_partial_n_tile: bool = False,
) -> bool:
    """Whether the K3 low-M GEMM (``ops.gemm.k3_fp8w_lowm_gemm``) serves this projection."""
    from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

    return k3_fp8w_lowm_gemm.can_run(
        x, weight, weight_scale, allow_partial_n_tile=allow_partial_n_tile
    )


class _Fp8BlockScaleWeightReadLinear(nn.Module):
    """Bias-free FP8 block-scale GEMM for checkpoint attention and converted MLPs.

    Attention loads checkpoint E4M3 codes and 128x128 scales directly. The optional
    MLP path quantizes BF16 weights once at load. On SM100/SM103 the UE8M0
    scales are packed for DeepGEMM, which quantizes BF16 activations inside the
    GEMM. On Rubin (SM107) the scales are packed into the CuTe DSL K32 R128c4
    layout and the activation is quantized by the specialized CuTe quantizer
    before the native MXFP8 GEMM, which runs with K3's fine-grained low-M tuning
    buckets. That dispatch is ``layers.fp8_block_scale_gemm`` (its generic low-M GEMM is not
    prepared here).

    The Rubin agentic kernels sit in front of that dispatch: the K3 low-M GEMM
    (``kernels/k3_fp8w_lowm_gemm.py``) with its gated, sigmoid, MXFP8-input and
    all-reduce-push variants owns the low-M rows (per-shape, per-architecture limit, see
    ``k3_fp8w_lowm_gemm.max_m_for``). It reads a DeepGEMM-packed scale: ``weight_scale`` on
    SM100/SM103, ``lowm_weight_scale`` (repacked from the MXFP8 scale) on SM107.
    """

    def __init__(
        self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor, out_features: int
    ) -> None:
        super().__init__()
        self.in_features = weight_fp8.shape[1]
        self.out_features = out_features
        self._weights_transformed = True
        self._fused_projection = None
        # Buffers (not parameters): these are the module's weights post-load;
        # there is nothing further to load into them and they must not be
        # touched by any later autocast/dtype move.
        self.register_buffer("weight", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        # Scales of the output-row slices :meth:`forward_slice` serves (an
        # ``Fp8BlockScaleWeightSliceCache``), prepared by :meth:`prepare_output_slice`.
        self._output_slice_cache: Optional[nn.Module] = None
        # Set by a caller for the next forward only: a contiguous BF16 ``[M, out_features]``
        # tensor that the low-M GEMM writes its result into. Other paths return their own
        # tensor, so the caller compares the result with the buffer it handed in.
        self.next_output: Optional[torch.Tensor] = None
        # Set by a caller for the next forward only (every forward consumes and clears it): the
        # ``LamportPushTarget`` (``layers.lamport_push_target``) of the all-reduce that
        # follows this projection. Where the low-M GEMM without split-K (or the mid-M GEMM with
        # its push epilogue) serves ``forward`` / ``forward_gated``, it stores its output into
        # that all-reduce's Lamport buffers instead, marks the target pushed and returns the
        # all-reduce's output tensor, which ``layers.finish_lamport_push`` fills.
        self.next_ar_push = None
        # Copied from the source ``Linear`` (the MLA fused q_a/kv_a projection sets it): the
        # low-M GEMM may serve this weight although N is not a multiple of 128.
        self.fp8w_lowm_partial_n_tile = False
        # SM107: the weight scale in the DeepGEMM-packed UE8M0 layout of the K3 low-M GEMM and the
        # fused latent projections (``weight_scale`` holds the CuTe MXFP8 layout there).
        self.lowm_weight_scale: Optional[torch.Tensor] = None

    @property
    def has_fp8_qdq(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def has_w4a8_nvfp4_fp8(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def supports_prequantized_input(self) -> bool:
        """Whether a producer may hand this GEMM an FP8 activation in CuTe layout."""
        return _rubin_mxfp8_is_available()

    @property
    def packed_weight_scale(self) -> Optional[torch.Tensor]:
        """The weight scale in DeepGEMM's packed UE8M0 layout, as the low-M GEMM and the fused
        latent projections read it: ``weight_scale`` on SM100/SM103, ``lowm_weight_scale`` on
        SM107 (``None`` unless :meth:`prepare_lowm_gemm` built it)."""
        if _rubin_mxfp8_is_available():
            return self.lowm_weight_scale
        return self.weight_scale

    def takes_prequantized_input(self, num_rows: int) -> bool:
        """Whether a producer that can emit this GEMM's scale layout should hand ``num_rows``
        rows to :meth:`forward_prequantized` rather than BF16 rows to :meth:`forward`.

        False where :meth:`forward` serves BF16 rows with the low-M GEMM (activation quantized
        inside the kernel) faster than the CuTe MXFP8 GEMM runs on the prequantized operand.
        """
        if not self.supports_prequantized_input:
            return False
        if self.lowm_weight_scale is None:
            return True
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        return num_rows > k3_fp8w_lowm_gemm.bf16_over_prequantized_max_m(
            self.out_features, self.in_features
        )

    @staticmethod
    def quantize_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """BF16 ``[out, in]`` weight -> (FP8 weight, backend-ready scale).

        Both dims must be multiples of 128. Because the 128x128 block scale is
        computed per block, concatenating several such weights along ``out``
        and quantizing the result is per-block identical to quantizing each
        separately (no block crosses a 128-aligned boundary), so a fused
        weight's row slices equal the individually quantized weights.
        """
        # Lazy imports: only pulled in on the FP8 path.
        from tensorrt_llm.deep_gemm.utils.math import per_block_cast_to_fp8

        weight_fp8, weight_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        # Both GEMMs consume UE8M0 scales rather than the FP32 grid.
        return prepare_fp8_block_weight(weight_fp8, weight_scale)

    @staticmethod
    def _prepare_scale_layout(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
        from tensorrt_llm.quantization.utils.fp8_utils import (
            transform_k128_scales_to_cutedsl_mxfp8_layout,
            transform_sf_into_required_layout,
        )

        if _rubin_mxfp8_is_available():
            # cute_dsl_mxfp8_gemm_rubin reads K32 R128c4 UE8M0 scales.
            return transform_k128_scales_to_cutedsl_mxfp8_layout(weight_scale, mn=n, k=k)
        # fp8_swap_ab_gemm with disable_ue8m0_cast=True consumes packed,
        # TMA-aligned UE8M0 scales.
        return transform_sf_into_required_layout(
            weight_scale,
            mn=n,
            k=k,
            recipe=(1, 128, 128),
            is_sfa=False,
        )

    @classmethod
    def from_linear(cls, linear: nn.Module) -> "_Fp8BlockScaleWeightReadLinear":
        """The FP8 weight read of a loaded projection, sharing its codes and scales:

        * ``core.Fp8BlockScaleLinear`` (checkpoint E4M3 codes and FP32 128 x 128 scales, or their
          ``prepare``d form);
        * a bias-free BF16 projection (``nn.Linear`` or ``core.Bf16Linear``): quantized here.
        """
        assert linear.bias is None, "FP8 weight read expects a bias-free Linear"
        if isinstance(linear, Fp8BlockScaleLinear):
            result = cls(linear.weight.detach(), linear.weight_scale.detach(), linear.out_features)
            result.quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
            result.tp_size, result.tp_rank = linear.tp_size, linear.tp_rank
            result._weights_transformed = getattr(linear, "prepared", False)
            result.fp8w_lowm_partial_n_tile = getattr(linear, "fp8w_lowm_partial_n_tile", False)
            return result
        weight_fp8, weight_scale = cls.quantize_weight(linear.weight.data)
        return cls(weight_fp8, weight_scale, linear.out_features)

    def transform_weights(self) -> None:
        if not self._weights_transformed:
            self._output_slice_cache = None
            if self._fused_projection is not None:
                owner_ref, start = self._fused_projection
                owner = owner_ref()
                if owner is None:
                    raise RuntimeError("KDA fused FP8 projection was released before its views")
                if self.weight_scale.dtype != torch.float32 or self.weight_scale.ndim != 2:
                    raise ValueError("KDA fused FP8 views require a raw FP32 scale grid")
                # Codes and raw scale grids alias the fused projection. Only its
                # owner may resmooth them in place; each view packs the updated
                # scales without modifying the shared codes again.
                owner.transform_weights()
                self.weight = owner.weight[start : start + self.out_features]
                self.weight_scale = self._prepare_scale_layout(
                    self.weight_scale, self.out_features, self.in_features
                )
            else:
                self.weight, self.weight_scale = prepare_fp8_block_weight(
                    self.weight, self.weight_scale
                )
            self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self.prepare_lowm_gemm()

    def prepare_lowm_gemm(self) -> bool:
        """Build ``lowm_weight_scale`` on SM107 (the K3 low-M GEMM and the fused latent
        projections read it).

        Derived from the transformed MXFP8 scale, so it follows weight reloads; rebuilt on
        every call (post-load), before CUDA graph capture. Costs N * K / 128 bytes (0.8% of
        the FP8 weight). SM100/SM103 run these GEMMs on ``weight_scale`` itself.
        """
        self.lowm_weight_scale = None
        if not _rubin_mxfp8_is_available():
            return False
        if self._fused_projection is not None:
            # A row view of the KDA fused q/k/v/g weight; the fused module serves forward.
            return False
        self.transform_weights()
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        n, k = self.weight.shape
        if n % k3_fp8w_lowm_gemm.N_ALIGN or k % 128:
            return False
        self.lowm_weight_scale = k3_fp8w_lowm_gemm.pack_block_scale_from_mxfp8_layout(
            self.weight_scale, n, k
        )
        return True

    def _lowm_gemm_applies(self, x_flat: torch.Tensor, weight_scale: torch.Tensor) -> bool:
        """Whether the ``k3_fp8w_lowm_gemm`` entry serves ``x_flat`` with ``weight_scale``."""
        return x_flat.dtype == torch.bfloat16 and _k3_lowm_gemm_can_run(
            x_flat,
            self.weight,
            weight_scale,
            allow_partial_n_tile=self.fp8w_lowm_partial_n_tile,
        )

    def _run_lowm_gemm(
        self,
        x_flat: torch.Tensor,
        weight_scale: torch.Tensor,
        out_buffer: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """The low-M GEMM, into ``out_buffer`` when the caller handed in a usable one."""
        if (
            out_buffer is not None
            and out_buffer.dtype == torch.bfloat16
            and out_buffer.shape == (x_flat.shape[0], self.out_features)
            and out_buffer.is_contiguous()
        ):
            return k3_fp8w_lowm_gemm_op(x_flat, self.weight, weight_scale, out_buffer)
        return k3_fp8w_lowm_gemm_op(x_flat, self.weight, weight_scale)

    def _take_next_output(self, num_rows: int) -> Optional[torch.Tensor]:
        """Clear :attr:`next_output`; return it if it is a contiguous, 16-byte aligned BF16
        ``[num_rows, out_features]`` tensor."""
        out, self.next_output = self.next_output, None
        if (
            out is None
            or out.dtype != torch.bfloat16
            or tuple(out.shape) != (num_rows, self.out_features)
            or not out.is_contiguous()
            or out.data_ptr() % 16
        ):
            return None
        return out

    def _take_next_ar_push(self, x_flat: torch.Tensor, gated: bool = False):
        """Clear :attr:`next_ar_push`; return it if the low-M GEMM can push this forward's
        ``[M, out_features]`` output into it (low-M kernel without split-K; above the low-M
        range, on ungated rows, the mid-M GEMM's push epilogue)."""
        push, self.next_ar_push = self.next_ar_push, None
        if (
            push is None
            or x_flat.dtype != torch.bfloat16
            or tuple(push.output.shape) != (x_flat.shape[0], self.out_features)
        ):
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.ar_push_tactic(
            x_flat.shape[0], self.out_features, self.in_features, gated=gated
        )
        if tactic is None and not gated:
            tactic = k3_fp8w_lowm_gemm.mid_ar_push_tactic(
                x_flat.shape[0], self.out_features, self.in_features
            )
        return push if tactic is not None else None

    def mid_push_pdl_ready(self, num_rows: int) -> bool:
        """``TRTLLM_K3_KDA_MX_PDL``: whether :meth:`forward_prequantized` on the producer's
        ``[num_rows, in_features]`` R128c4 operand will push with the mid-M GEMM (the conditions
        of :meth:`_mid_prequantized_push_tactic` that do not depend on the tensors), which reads
        nothing the producer writes before its ``griddepcontrol.wait``: the producer may then
        release it early."""
        push = self.next_ar_push
        if push is None or self.lowm_weight_scale is None:
            return False
        if num_rows > 128:
            return False
        if tuple(push.output.shape) != (num_rows, self.out_features):
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.mid_ar_push_tactic(num_rows, self.out_features, self.in_features)
        return tactic is not None and tactic.mma_n <= 128

    def _mid_prequantized_push_tactic(self, activation: torch.Tensor, scale: torch.Tensor, push):
        """The mid-M tactic :meth:`forward_prequantized` pushes the KDA verify's MXFP8 operand with,
        or None. Up to 128 rows (one scale-factor block), where the R128c4 scales of the operand are
        the mid-M GEMM's ``[K / 128][1][512 B]`` layout; the mid-M GEMM instead of the CuTe MXFP8
        GEMM, so the projection's FP32 accumulation order can differ there."""
        if push is None or self.lowm_weight_scale is None:
            return None
        if activation.dim() != 2 or not activation.is_contiguous():
            return None
        m, k = activation.shape
        if k != self.in_features or tuple(push.output.shape) != (m, self.out_features):
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        tactic = k3_fp8w_lowm_gemm.mid_ar_push_tactic(m, self.out_features, k)
        if tactic is None or tactic.mma_n > 128:
            return None
        if scale.numel() != (k // 128) * 512 or not scale.is_contiguous():
            return None
        return tactic

    def _run_ar_push(self, x_flat: torch.Tensor, push, gate=None) -> torch.Tensor:
        """The low-M GEMM with the all-reduce push epilogue; returns the all-reduce output."""
        k3_fp8w_lowm_gemm_push_op(
            x_flat, self.weight, self.packed_weight_scale, push.workspace, push.one_shot, gate=gate
        )
        push.pushed = True
        return push.output

    def forward(
        self,
        x: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        out_shape = (*x.shape[:-1], self.out_features)
        x_flat = x.reshape(-1, x.shape[-1])
        out_buffer, self.next_output = self.next_output, None
        ar_push = self._take_next_ar_push(x_flat)

        # The K3 low-M GEMM up to its per-shape M gate, on SM107 with ``lowm_weight_scale``, on
        # SM100 / SM103 with ``weight_scale``; else the MXFP8 (SM107) / swap-AB (SM100 / SM103)
        # GEMM of ``layers.fp8_block_scale_gemm``.
        lowm_scale = self.lowm_weight_scale if _rubin_mxfp8_is_available() else self.weight_scale
        if lowm_scale is not None and self._lowm_gemm_applies(x_flat, lowm_scale):
            if ar_push is not None:
                out = self._run_ar_push(x_flat, ar_push)
            else:
                out = self._run_lowm_gemm(x_flat, lowm_scale, out_buffer)
        else:
            out = fp8_block_scale_gemm(x_flat, self.weight, self.weight_scale)
        return out.reshape(out_shape)

    def prepare_output_slice(self, start: int, end: int) -> None:
        """Prepare the scales of output features ``[start, end)`` for :meth:`forward_slice`.

        The shared MLA calls this at post-load for the context heads of a Helix rank. The slice
        keeps the 128-row scale grid: it covers the whole scale tiles around ``[start, end)``.
        """
        from tensorrt_llm.quantization.utils.fp8_utils import Fp8BlockScaleWeightSliceCache

        if not 0 <= start < end <= self.out_features:
            raise ValueError(f"Invalid output feature range [{start}, {end})")
        self.transform_weights()
        if self._output_slice_cache is None:
            self._output_slice_cache = Fp8BlockScaleWeightSliceCache()
        self._output_slice_cache.prepare(self.weight, self.weight_scale, start, end)

    def forward_slice(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Output features ``[start, end)`` of :meth:`forward`, contiguous.

        One GEMM over the weight rows of the slice's scale tiles (the general path of
        :meth:`forward`: the CuTe MXFP8 GEMM on SM107, ``fp8_swap_ab_gemm`` elsewhere), cropped
        to the slice.
        """
        self.prepare_output_slice(start, end)
        weight, scale, crop = self._output_slice_cache.get(self.weight, start, end)
        x_flat = x.reshape(-1, x.shape[-1])
        if _rubin_mxfp8_is_available():
            act_fp8, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x_flat)
            out = k3_mxfp8_gemm_op(act_fp8, weight, act_scale, scale)
        else:
            out = torch.ops.trtllm.fp8_swap_ab_gemm(
                x_flat, weight, scale, output_dtype=x.dtype, disable_ue8m0_cast=True
            )
        out = out.reshape(*x.shape[:-1], weight.shape[0])
        return out[..., crop : crop + end - start].contiguous()

    def _lowm_forward_scale(self, x_flat: torch.Tensor) -> Optional[torch.Tensor]:
        """The weight scale :meth:`forward` runs the K3 low-M GEMM with on the BF16 rows
        ``x_flat`` (``lowm_weight_scale`` on SM107, ``weight_scale`` on SM100/SM103), or None
        where :meth:`forward` runs another GEMM."""
        if x_flat.dtype != torch.bfloat16:
            return None
        if self.lowm_weight_scale is not None:
            scale = self.lowm_weight_scale
        elif _rubin_mxfp8_is_available():
            return None
        else:
            scale = self.weight_scale
        return scale if self._lowm_gemm_applies(x_flat, scale) else None

    def _runs_low_m_kernel(self, num_rows: int) -> bool:
        """Whether :meth:`forward` on ``num_rows`` BF16 rows would run the low-M kernel (the
        input-independent part of its dispatch: not the Rubin CuTe MXFP8 path, ``num_rows``
        within the low-M kernel's range). On SM107 forward runs it on ``lowm_weight_scale`` up to
        the per-shape gate (``max_m_for``)."""
        if _rubin_mxfp8_is_available() and self.lowm_weight_scale is None:
            return False
        if self.weight.shape[0] % 128 and not self.fp8w_lowm_partial_n_tile:
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        limit = k3_fp8w_lowm_gemm.low_m_max_for(*self.weight.shape)
        if self.lowm_weight_scale is not None:
            limit = min(limit, k3_fp8w_lowm_gemm.max_m_for(*self.weight.shape))
        return 1 <= num_rows <= limit

    def takes_mxfp8_input(self, num_rows: int) -> bool:
        """Whether a producer should hand :meth:`forward_mxfp8` the MXFP8 form of ``num_rows``
        rows: where :meth:`forward` would run the low-M kernel, whose in-kernel quantization the
        MXFP8 GEMM replaces (the attention's first projection also asks
        :meth:`takes_mid_mxfp8_input`)."""
        return self._runs_low_m_kernel(num_rows)

    def takes_mid_mxfp8_input(self, num_rows: int) -> bool:
        """Whether :meth:`forward` on ``num_rows`` BF16 rows runs the mid-M path of
        ``k3_fp8w_lowm_gemm`` (above the low-M cut-over, up to the per-shape gate) with one
        128-row scale block, so that a producer may hand :meth:`forward_mxfp8` the rows' MXFP8 form
        with the scales in that GEMM's layout (``k3_attn_res`` ``mx_atom``): the GEMM then runs
        without its quantization kernel, bitwise the BF16 path."""
        return self.mid_operand_tactic(num_rows) is not None

    def mid_operand_tactic(self, num_rows: int):
        """c240: the mid-M tactic :meth:`forward` runs on ``num_rows`` BF16 rows (SM107, above the
        low-M cut-over, up to the per-shape gate) when it has one 128-row scale block, else None.
        A producer may then store the rows' MXFP8 form in that GEMM's layout
        (``mid_prequant_plan``) for :meth:`forward_mxfp8` / :meth:`forward_mid_mxfp8`."""
        if self.lowm_weight_scale is None:
            return None
        n, k = self.weight.shape
        if n % 128 and not self.fp8w_lowm_partial_n_tile:
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        if (
            not k3_fp8w_lowm_gemm.low_m_max_for(n, k)
            < num_rows
            <= min(128, k3_fp8w_lowm_gemm.max_m_for(n, k))
        ):
            return None
        plan = k3_fp8w_lowm_gemm.mid_prequant_plan(num_rows, n, k)
        return plan[0] if plan is not None and plan[1] == 1 else None

    def forward_mid_mxfp8(
        self, x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor
    ) -> torch.Tensor:
        """``TRTLLM_K3_MID_OPROJ_PC`` (c240): :meth:`forward` on rows whose MXFP8 form a producer
        stored with the scales in the mid-M GEMM's layout (``xs`` 1-D, see
        :meth:`mid_operand_tactic`; ``x`` only gives the output shape): the mid-M GEMM without
        its quantization kernel, with the all-reduce push epilogue where :attr:`next_ar_push`
        takes it. Bitwise :meth:`forward` on the rows the producer quantized."""

        m = xq.shape[0]
        tactic = self.mid_operand_tactic(m)
        if tactic is None or xs.dim() != 1:
            raise RuntimeError("forward_mid_mxfp8 needs the mid-M operand layout")
        x_flat = x.reshape(-1, x.shape[-1])
        push = self._take_next_ar_push(x_flat)
        self.next_output = None
        if push is not None:
            k3_fp8w_mid_gemm_push_op(
                xq.reshape(m, -1),
                xs,
                self.weight,
                self.lowm_weight_scale,
                push.workspace,
                push.one_shot,
                tactic,
            )
            push.pushed = True
            return push.output
        out = torch.empty((m, self.out_features), dtype=torch.bfloat16, device=xq.device)
        k3_fp8w_mid_gemm_op(xq.reshape(m, -1), xs, self.weight, self.lowm_weight_scale, out, tactic)
        return out.reshape(x.shape[:-1] + (self.out_features,))

    def _staged_above_cutover(self, num_rows: int) -> bool:
        """Whether ``forward_mxfp8(..., stage_weights=True)`` serves ``num_rows`` rows above the
        low-M kernel's cut-over with the staged MXFP8 low-M GEMM: :meth:`forward` runs the mid-M
        path of ``k3_fp8w_lowm_gemm`` there, whose bits the staged kernel reproduces
        (``k3_fp8w_lowm_mxfp8.staged_matches_mid_m``)."""
        if self.lowm_weight_scale is None:
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm, k3_fp8w_lowm_mxfp8

        n, k = self.weight.shape
        if num_rows > k3_fp8w_lowm_gemm.max_m_for(n, k):
            return False
        return k3_fp8w_lowm_mxfp8.staged_matches_mid_m(num_rows, n, k)

    def takes_staged_mxfp8_input(self, num_rows: int) -> bool:
        """Whether ``forward_mxfp8(..., stage_weights=True)`` runs the staged MXFP8 low-M GEMM on
        the MXFP8 form of ``num_rows`` rows (bitwise :meth:`forward`): where :meth:`forward` runs
        the low-M kernel, and above its cut-over where it runs the mid-M path with the same
        split-K (up to 64 rows)."""
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8

        n, k = self.weight.shape
        if self._runs_low_m_kernel(num_rows):
            # Also where the shared-memory ring alone holds the CTA's whole weight slice.
            return k3_fp8w_lowm_mxfp8.weights_on_chip(num_rows, n, k)
        return self._staged_above_cutover(num_rows)

    def forward_mxfp8(
        self, x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor, stage_weights: bool = False
    ) -> torch.Tensor:
        """:meth:`forward` on the BF16 rows ``x``, bit for bit, given their MXFP8 form ``(xq,
        xs)`` (E4M3 ``[M, K]`` and one UE8M0 exponent per 1x32 group, e.g. from
        the ``k3_attn_res_norm`` MXFP8 variants): the low-M GEMM on the prequantized operand where
        :meth:`forward` would run the low-M kernel, else :meth:`forward` on ``x``.
        ``stage_weights``: the low-M GEMM keeps its whole weight slice on chip before its
        ``griddepcontrol.wait`` (``k3_fp8w_lowm_mxfp8.staged_mx_tactic``, the same bits), also
        above the low-M cut-over where :meth:`takes_staged_mxfp8_input` says so."""
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8

        x_flat = x.reshape(-1, x.shape[-1])
        if xs.dim() == 1 and self.takes_mid_mxfp8_input(x_flat.shape[0]):
            # The producer stored the scales in the mid-M GEMM's layout.
            from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

            m = x_flat.shape[0]
            tactic, _ = k3_fp8w_lowm_gemm.mid_prequant_plan(m, self.out_features, self.in_features)
            self.next_ar_push = None
            out = self._take_next_output(m)
            if out is None:
                out = torch.empty((m, self.out_features), dtype=torch.bfloat16, device=x.device)
            k3_fp8w_mid_gemm_op(
                xq.reshape(m, -1), xs, self.weight, self.lowm_weight_scale, out, tactic
            )
            return out.reshape(x.shape[:-1] + (self.out_features,))
        allow_partial = self.fp8w_lowm_partial_n_tile
        # The scale forward's low-M branch uses (lowm_weight_scale on SM107).
        w_scale = self.packed_weight_scale
        above_cutover = (
            stage_weights
            and not self._runs_low_m_kernel(x_flat.shape[0])
            and self._staged_above_cutover(x_flat.shape[0])
        )
        if not (
            x.dtype == torch.bfloat16
            and (above_cutover or self._runs_low_m_kernel(x_flat.shape[0]))
            and xq.shape[0] == x_flat.shape[0]
            and _k3_lowm_gemm_can_run(
                x_flat, self.weight, w_scale, allow_partial_n_tile=allow_partial
            )
            and k3_fp8w_lowm_mxfp8.can_run_mxfp8(
                xq,
                xs,
                self.weight,
                w_scale,
                allow_partial_n_tile=allow_partial,
                above_cutover=above_cutover,
            )
        ):
            return self.forward(x)
        out = self._take_next_output(x_flat.shape[0])
        # The MXFP8 GEMM has no all-reduce push epilogue: the caller runs the all-reduce.
        self.next_ar_push = None
        if out is None:
            out = torch.empty(
                (x_flat.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device
            )
        stage_weights = stage_weights and k3_fp8w_lowm_mxfp8.weights_on_chip(
            x_flat.shape[0], self.out_features, self.in_features
        )
        k3_fp8w_lowm_mxfp8_gemm_op(xq, xs, self.weight, w_scale, out, stage_weights=stage_weights)
        return out.reshape(x.shape[:-1] + (self.out_features,))

    def gated_applies(self, x: torch.Tensor, gate: torch.Tensor) -> bool:
        """Whether :meth:`forward_gated` serves ``x`` and ``gate`` with one gated low-M GEMM.

        That is where :meth:`forward` would run the low-M GEMM on ``x * gate`` (same kernel
        family and split), so the gated launch is bitwise equal to the multiply plus
        :meth:`forward`.
        """
        if x.dim() != 2 or not x.is_contiguous():
            return False
        scale = self._lowm_forward_scale(x)
        if scale is None:
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        return k3_fp8w_lowm_gemm.can_run_gated(
            x,
            gate,
            self.weight,
            scale,
            allow_partial_n_tile=self.fp8w_lowm_partial_n_tile,
        )

    def forward_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """``forward(x).sigmoid()``, bit for bit; one low-M GEMM launch with a sigmoid
        epilogue where :meth:`forward` would run the low-M GEMM."""
        x_flat = x.reshape(-1, x.shape[-1])
        scale = self._lowm_forward_scale(x_flat)
        if scale is not None:
            from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

            if x_flat.shape[0] <= k3_fp8w_lowm_gemm.low_m_max_for(
                self.out_features, self.in_features
            ):
                self.next_output = None
                self.next_ar_push = None
                out = k3_fp8w_lowm_gemm_op(x_flat, self.weight, scale, sigmoid=True)
                return out.reshape(x.shape[:-1] + (self.out_features,))
        return self.forward(x).sigmoid()

    def forward_gated(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``forward(x * gate)`` in one gated low-M GEMM launch (where :meth:`gated_applies`).

        ``x`` and ``gate`` are BF16 ``[M, in_features]``; the gate may be a row-strided view.
        """
        ar_push = self._take_next_ar_push(x, gated=True)
        if ar_push is not None:
            self.next_output = None
            return self._run_ar_push(x, ar_push, gate=gate)
        return k3_fp8w_lowm_gemm_op(
            x,
            self.weight,
            self.packed_weight_scale,
            out=self._take_next_output(x.shape[0]),
            gate=gate,
        )

    def gated_mxfp8_applies(self, x: torch.Tensor, gate: torch.Tensor) -> bool:
        """Whether :meth:`forward_gated_mxfp8` runs the MXFP8 low-M GEMM on the producer's MXFP8
        form of ``bf16(x * gate)``: where :meth:`forward_gated` runs
        the gated low-M GEMM, whose tiling the MXFP8 kernel shares (same bits)."""
        if not self.gated_applies(x, gate):
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8

        m, k = x.shape
        try:
            k3_fp8w_lowm_mxfp8.default_mx_tactic(m, self.out_features, k)
        except ValueError:
            return False
        return k % 32 == 0

    def forward_gated_mxfp8(
        self, x: torch.Tensor, gate: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor
    ) -> torch.Tensor:
        """:meth:`forward_gated` ``(x, gate)``, bit for bit, given the MXFP8 form ``(xq, xs)`` of
        ``bf16(x * gate)`` (E4M3 ``[M, K]`` and one UE8M0 exponent per 1x32 group, the gated
        low-M GEMM's own quantization): the MXFP8 low-M GEMM, with the all-reduce push epilogue
        where :attr:`next_ar_push` is set, so the activation path after its
        ``griddepcontrol.wait`` is two TMA loads instead of the BF16 loads, the gate multiply and
        the quantization."""
        from ..kernels.k3r_gemm import k3_fp8w_lowm_mxfp8

        w_scale = self.packed_weight_scale
        if w_scale is None or not k3_fp8w_lowm_mxfp8.can_run_mxfp8(xq, xs, self.weight, w_scale):
            return self.forward_gated(x, gate)
        ar_push = self._take_next_ar_push(x, gated=True)
        if ar_push is not None:
            self.next_output = None
            k3_fp8w_lowm_mxfp8_push_op(
                xq, xs, self.weight, w_scale, ar_push.workspace, ar_push.one_shot
            )
            ar_push.pushed = True
            return ar_push.output
        out = self._take_next_output(x.shape[0])
        if out is None:
            out = torch.empty(
                (x.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device
            )
        k3_fp8w_lowm_mxfp8_gemm_op(xq, xs, self.weight, w_scale, out)
        return out

    def forward_prequantized(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Project an FP8 activation a fused producer already quantized.

        Args:
            activation: ``[..., in_features]`` ``float8_e4m3fn`` activation.
            activation_scale: ``uint8`` UE8M0 scales in the CuTe K32 R128c4
                layout that ``fp8_quantize_1x128_cutedsl_ue8m0`` produces.
        """
        push, self.next_ar_push = self.next_ar_push, None
        if activation.dtype is not torch.float8_e4m3fn:
            raise TypeError("prequantized Kimi K3 activation must be float8_e4m3fn")
        if not self.supports_prequantized_input:
            raise RuntimeError("prequantized Kimi K3 activation requires the Rubin CuTe GEMM")
        if activation_scale.dtype is not torch.uint8:
            raise TypeError("prequantized Kimi K3 activation scale must be uint8 R128c4")
        rows = activation.reshape(-1, activation.shape[-1])
        tactic = self._mid_prequantized_push_tactic(rows, activation_scale, push)
        if tactic is not None:
            # The mid-M GEMM pushes into the o_proj all-reduce.
            k3_fp8w_mid_gemm_push_op(
                rows,
                activation_scale.reshape(-1),
                self.weight,
                self.lowm_weight_scale,
                push.workspace,
                push.one_shot,
                tactic,
            )
            push.pushed = True
            return push.output
        out = k3_mxfp8_gemm_op(
            activation.reshape(-1, activation.shape[-1]),
            self.weight,
            activation_scale,
            self.weight_scale,
        )
        return out.reshape(activation.shape[:-1] + (self.out_features,))


def _swap_linear_to_fp8_weight_read(
    parent: nn.Module,
    attr: str,
    linear_types: Tuple[type, ...] = (nn.Linear,),
) -> int:
    """Replace ``parent.<attr>`` with an FP8 weight-read module if it is a
    plain linear of one of ``linear_types``; return the number of modules
    converted (0 or 1), so callers can accumulate a conversion count.

    Transfer checkpoint FP8 storage or release the original BF16 storage
    after conversion. The loader's transient parameter map must not retain
    an unused BF16 copy until loading completes.
    """
    child = getattr(parent, attr, None)
    if not isinstance(child, linear_types):
        return 0
    converted = _Fp8BlockScaleWeightReadLinear.from_linear(child)
    setattr(parent, attr, converted)
    child.weight.data = child.weight.data.new_empty(0)
    return 1


def _has_weights(module: nn.Module) -> bool:
    """False once ``modeling_utils.remove_weights()`` has stripped a module.

    Post-load finalization walks every decoder layer, so it must skip layers
    whose parameters were dropped — the layer-wise benchmarks keep only the
    profiled slice resident.
    """
    return not getattr(module, "_weights_removed", False)


def _convert_moe_mlps_to_fp8_weight_read(
    model: nn.Module, include_fused_gate_up: bool = True
) -> int:
    """Swap the replicated MoE-layer MLP projections to an FP8 weight read.

    Targets the shared-expert MLP (gate/up/down) and the latent up/down
    projection on every MoE layer — the bias-free BF16 projections that
    attention data-parallelism re-reads in full each decode step. Attention
    follows checkpoint quantization separately. Routed experts and the dense
    layer-0 MLP are not converted. Returns the number of projections converted.
    """
    count = 0

    for layer in model.layers:
        if not _has_weights(layer):
            continue
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        shared = getattr(moe, "shared_experts", None)
        if shared is not None:
            # GatedMlp fuses gate and up into gate_up_proj; keep the split
            # names too so either MLP layout converts. The fused gate_up read
            # only pays off when attention DP re-reads it per rank per step;
            # under TP the bf16 GEMM overlaps on the aux stream and the FP8
            # quantize+GEMM would serialize onto the critical path.
            shared_attrs = (
                ("gate_proj", "up_proj", "gate_up_proj", "down_proj")
                if include_fused_gate_up
                else ("gate_proj", "up_proj", "down_proj")
            )
            for attr in shared_attrs:
                child = getattr(shared, attr, None)
                if isinstance(child, Bf16Linear) and child.tp_size != 1:
                    continue
                count += _swap_linear_to_fp8_weight_read(
                    shared, attr, linear_types=(nn.Linear, Bf16Linear)
                )
        for attr in ("routed_expert_down_proj", "routed_expert_up_proj"):
            count += _swap_linear_to_fp8_weight_read(moe, attr)

    # Return the freed BF16 blocks to the driver so the raw (non-caching-
    # allocator) allocations made during executor creation succeed on the
    # tight DEP16 memory headroom.
    if count:
        gc.collect()
        torch.cuda.empty_cache()
    return count


# Decode-batch routing tail of the CuteDSL MoE (see k3_route_tail): replaces noaux_tc_op ->
# moe_sort with one launch on a cluster of at most eight CTAs (same logits, same routing).
# Largest token count the routing tail serves: up to 256 tokens one cluster, above that (at most
# 2048 tokens, see k3_route_tail.wide_sort_fits) a grid of CTAs whose last CTA does the sort.
_K3_ROUTE_TAIL_MAX_M = 2048

_K3_LD_FUSED_MIN_M = 1
_K3_LD_FUSED_MAX_M = 4096

# For up to _K3_ROUTER_FIRST_MAX_M tokens, the router GEMM is issued (captured) on the main stream
# before latent_down is issued on the latent stream. Both depend only on the MoE input (the latent
# fork event is recorded before the router), so the graph has the same edges as the reverse
# order; only the node creation order differs, which decides which of the two the
# graph launches first once the MoE input's producer releases its dependents (PDL). At decode
# batch sizes latent_down's low-M GEMM is 112 CTAs x ~225 KB of shared memory, and with up to 16
# tokens the router GEMM (dsv3 router MMA kernel) is 112 CTAs whose registers fill most of an SM,
# so no SM holds one of each: launched second, the router gets only the SMs latent_down leaves
# and runs in several waves, and the routing (which gates the routed FC1 together with
# latent_down) ends late. Launched first, the router takes its SMs and latent_down starts on the
# rest. nsys node tracing turns latent_down's cross-stream programmatic edge into a full edge (it
# then waits for the MoE input's producer to complete), so such traces show the router first in
# either order.
_K3_ROUTER_FIRST_MAX_M = 64

# FC12 calls from this many tokens claim the tiles of their own die first (die-major expert
# weights): at fewer tokens the claim overhead outweighs the locality gain.
_K3_FC12_DIE_SCHED_MIN_TOKENS = 24

# SM count of the shared expert's cuBLAS GEMMs. FC12's persistent grid needs whole SMs, so
# shared-expert GEMMs still resident when routing finishes delay its launch; they run on
# _K3_SHARED_SM SMs (torch SM carveout, captured into CUDA graphs) and the routed experts' backend
# keeps that many SMs free of its persistent grids (``sm_reserve``, ``_k3_fc12_options``).
_K3_SHARED_SM = 16


@contextmanager
def _k3_shared_sm_scope():
    """Run the enclosed cuBLAS GEMMs on ``_K3_SHARED_SM`` SMs."""
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    prev = torch._C._get_sm_carveout_experimental()
    torch._C._set_sm_carveout_experimental(max(0, num_sms - _K3_SHARED_SM))
    try:
        yield
    finally:
        torch._C._set_sm_carveout_experimental(prev)


# The exact router: in the MoE blocks of KDA layers, the router logits of M <= 16 tokens come from
# k3_router_exact (bit-identical to dsv3_router_gemm_op; its weight is copied into shared memory
# before griddepcontrol.wait). The faster router also starts the route tail earlier, inside the
# burst of latent_down and the shared gate_up; after an MLA attention block the route tail then
# stretches by ~3 us (b2 gen-only, GB300) and the routing chain gets longer than with
# dsv3_router_gemm_op, while after a KDA attention block the front gets shorter.
# The routed latent all-reduce's own workspace: for up to _K3_LATENT_AR_OWN_WS_MAX_M tokens, the routed
# latent all-reduce + RMSNorm (the same MNNVL Lamport kernels) runs on a workspace of its own instead of the
# one it shares with the shared expert's all-reduce. It then no longer has to follow the shared-expert
# chain: the main stream issues it and latent_up right after the routed FC2 and joins the shared
# stream after latent_up. Applies where latent_up does not add the shared output itself (no fused
# latent_up + residual variant); the output is bit-identical.
# Up to _K3_LATENT_AR_ONESHOT_MAX_BYTES (rows x latent x ranks x 2 B; the workspace is sized for it,
# the other MNNVL all-reduces keep the global one-shot limit: 6291456 runs the 96-row latent
# all-reduce of 8 ranks, 5.5 MB, one-shot) that all-reduce runs k3_latent_ar_norm, bit-identical to
# the C++ one-shot kernel but triggering its programmatic dependents (latent_up) right after its
# griddepcontrol.wait instead of after the cross-rank reduction: its row kernel where that serves
# the size (one 512-thread CTA per row instead of a cluster of four; the rank's own slot is summed
# from registers instead of being polled back from the multicast loopback; slots that arrived are
# not reloaded), else the cluster kernel; elsewhere the C++ kernel (trtllm::mnnvl_fusion_allreduce)
# runs on the same workspace.
# Where latent_up runs the mid-M path of the k3_fp8w_lowm_gemm entry (above the low-M cut-over, e.g.
# the 96-row helix8 b16 mtp5 verify step), the row kernel also stores the MXFP8 form of the
# normalized latent in that GEMM's scale layout and latent_up runs the mid-M GEMM on it
# (k3_fp8w_lowm_gemm.run_mid_prequantized) instead of quantizing the BF16 rows in a kernel of its
# own: bitwise the same projection, one kernel less after the all-reduce.
_K3_LATENT_AR_OWN_WS_MAX_M = 128
_K3_LATENT_AR_ONESHOT_MAX_BYTES = 6291456
# The routed latent all-reduce's own MNNVL workspaces, by all-reduce mapping.
_K3_LATENT_AR_WORKSPACES: Dict[Mapping, Dict[str, Any]] = {}


def _k3_latent_ar_workspace(
    mapping: Mapping, dtype: torch.dtype, hidden_size: int, max_tokens: int
) -> Dict[str, Any]:
    """MNNVL Lamport workspace of the routed latent all-reduce alone (three buffers and the state
    flags, laid out as ``get_or_scale_allreduce_mnnvl_workspace`` lays out the shared one), sized
    for up to ``max_tokens`` rows of ``hidden_size``. Collective over the all-reduce group: every
    rank builds it at the same point of model construction."""
    workspace = _K3_LATENT_AR_WORKSPACES.get(mapping)
    if workspace is not None:
        if workspace["max_tokens"] < max_tokens:
            raise ValueError(
                f"Kimi K3 latent all-reduce workspace serves {workspace['max_tokens']} tokens, "
                f"{max_tokens} requested"
            )
        return workspace
    from tensorrt_llm._torch.distributed.ops import (
        MNNVLAllReduce,
        get_or_scale_allreduce_mnnvl_workspace,
    )
    from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

    # The shared workspace's communicator (split by the all-reduce group) is the same group.
    comm = _k3_mnnvl_mpi_comm(get_or_scale_allreduce_mnnvl_workspace(mapping, dtype))
    need = max(
        MNNVLAllReduce.get_required_workspace_size(m, hidden_size, mapping.tp_size, dtype)
        for m in range(1, max_tokens + 1)
    )
    # One-shot rows under the latent all-reduce's own limit: every rank's rows in one buffer.
    row_bytes = hidden_size * mapping.tp_size * torch.tensor([], dtype=dtype).element_size()
    one_shot_rows = min(max_tokens, _K3_LATENT_AR_ONESHOT_MAX_BYTES // row_bytes)
    need = max(need, one_shot_rows * row_bytes)
    buffer_bytes = -(-need // (1 << 20)) << 20
    use_fabric_handle = use_mnnvl_fabric_handle(mapping)
    handle = McastGPUBuffer(
        3 * buffer_bytes,
        mapping.tp_size,
        mapping.tp_rank,
        mapping.local_rank,
        use_fabric_handle,
        comm.py2f(),
    )
    # Lamport-clean (-0.0) before any rank writes into it.
    buffer = handle.get_uc_buffer(mapping.tp_rank, (3 * buffer_bytes // 4,), torch.float32, 0)
    buffer.fill_(-0.0)
    torch.cuda.synchronize()
    comm.Barrier()
    # [current index, dirty index, bytes per buffer, dirty stages, bytes to clear x4, arrivals]
    flags = torch.tensor(
        [0, 2, buffer_bytes, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device=buffer.device
    )
    workspace = {
        "handle": handle,
        "buffer": buffer.view(dtype).view(3, -1),
        # The multicast view of the same buffers (stores reach every rank's copy).
        "mc": handle.get_mc_buffer((3 * buffer_bytes // 4,), torch.float32, 0),
        "flags": flags,
        "max_tokens": max_tokens,
        # Bytes of one of the three Lamport buffers.
        "buffer_bytes": buffer_bytes,
        "rank": mapping.tp_rank,
        "world_size": mapping.tp_size,
    }
    _K3_LATENT_AR_WORKSPACES[mapping] = workspace
    return workspace


class K3MoEGate(nn.Module):
    """Kimi K3 gate: the router weight, the score-correction bias and the routing kernels.

    Routing is ungrouped sigmoid top-k with the bias added for the selection and the selected
    scores renormalized and scaled (``noaux_tc``); :meth:`route` and :meth:`route_tail` are two
    kernels of the same routing.
    """

    def __init__(
        self,
        config: Any,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size

        if self.moe_router_activation_func != "sigmoid" or not self.moe_renormalize:
            raise ValueError(
                "Kimi K3 routing takes sigmoid scores with top-k weight renormalization"
            )
        if self.num_expert_group != 1 or self.topk_group != 1:
            raise ValueError("Kimi K3 routing is ungrouped (num_expert_group = topk_group = 1)")
        if self.num_experts > 1024 or self.top_k > 32:
            raise ValueError(
                f"the fused routing kernel serves up to 1024 experts and top-32, got "
                f"{self.num_experts} and top-{self.top_k}"
            )

        # The checkpoint stores the gate weight in bf16. Storing it in bf16
        # permits the single bf16xbf16 router GEMM while retaining fp32 output.
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim), dtype=torch.bfloat16, device=device)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32, device=device)
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute fp32 routing logits shaped ``[num_tokens, num_experts]``."""
        hidden_2d = hidden_states.reshape(-1, self.gating_dim)
        if self.weight.dtype == torch.bfloat16 and hidden_2d.dtype == torch.bfloat16:
            return torch.ops.trtllm.dsv3_router_gemm_op(
                hidden_2d.contiguous(),
                self.weight.t(),
                bias=None,
                out_dtype=torch.float32,
            )
        return torch.nn.functional.linear(
            hidden_2d.type(torch.float32),
            self.weight.type(torch.float32),
            None,
        )

    def route(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(ids int32, weights fp32)``, each ``[num_tokens, top_k]``: ``noaux_tc_op``."""
        values, indices = noaux_tc_op(
            logits,
            self.e_score_correction_bias,
            self.num_expert_group,
            self.topk_group,
            self.top_k,
            self.routed_scaling_factor,
        )
        return indices.to(torch.int32), values.to(torch.float32)

    def route_tail(
        self, logits: torch.Tensor, local_expert_offset: int, local_num_experts: int
    ) -> SortedRouting:
        """The CuTe routing tail (``k3_route_tail``, top-k bitwise ``noaux_tc_op``) with the
        ``moe_sort`` tables of experts ``[local_expert_offset, local_expert_offset +
        local_num_experts)``."""
        from ..kernels.k3r_moe import k3_route_tail

        # Above the cluster kernel's tokens, the multi-CTA sort's persistent workspace.
        workspace = (
            k3_route_tail.wide_workspace(logits.device)
            if logits.shape[0] > k3_route_tail.MAX_M
            else None
        )
        outs = k3_route_tail_op(
            logits,
            self.e_score_correction_bias,
            float(self.routed_scaling_factor),
            local_expert_offset,
            local_num_experts,
            k3_route_tail.TILE,
            workspace=workspace,
        )
        return SortedRouting(outs[0], outs[1], SortTables(k3_route_tail.TILE, tuple(outs[2:8])))


# Row-parallel MoE tail for generation-only batches of up to _K3_RP_TAIL_MAX_M tokens (the low-M
# GEMM's range), where the routed latent is all-reduced by the model (MoE TP/EP over the all-reduce
# group, TP-sharded shared expert) and the layer's residual update is fused. After the latent
# all-reduce + RMSNorm, every rank runs one low-M FP8 GEMM over its whole 128-column blocks of the
# normalized latent with the matching latent_up columns (FP8 codes and block scales read in place)
# and pushes its partial [M, hidden] plus its shared-expert partial into the next all-reduce on the
# latent all-reduce's MNNVL workspace: the shared expert keeps its BF16 down GEMM (row-parallel
# shard, no all-reduce) on its stream, and the tail GEMM's push epilogue adds that BF16 partial to
# its BF16 output. The next layer's pre-attention mix finishes that all-reduce (one-shot transport
# up to lamport._K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES, two-shot above) and adds it to the residual
# (``_DeferredRpTail``). This replaces the shared expert's all-reduce and the replicated latent_up.
# Not bitwise: the latent_up sum is split over the ranks.
_K3_RP_TAIL_MAX_M = 32


# ---------------------------------------------------------------------------
# Latent MoE block: router, routed experts (kernel chains of moe.py), shared expert.
# ---------------------------------------------------------------------------

# Routed-expert key spellings that ModelOpt emits for Kimi K3. The NVFP4
# checkpoint (``nvidia/Kimi-K3-NVFP4``) lists every prefix x module-name
# combination in ``quantized_layers``, so a lookup over this product finds it
# without needing the MiniMax-M3-style prefix normalization in ``ModelConfig``.
_K3_ROUTED_EXPERT_KEY_PREFIXES = ("language_model.model.", "model.", "")
_K3_ROUTED_EXPERT_KEY_SUFFIXES = ("block_sparse_moe.experts", "mlp.experts")

# The subset of the above that can be a real module path. ``exclude_modules``
# matches with wildcards and walks ancestor prefixes, so an empty prefix would
# widen what matches instead of just missing, as it does in the dict lookup.
_K3_ROUTED_EXPERT_MODULE_PREFIXES = ("language_model.model.", "model.")

# Routed-expert quantization used when the checkpoint declares nothing per
# layer. The original ``moonshotai/Kimi-K3`` ships a compressed-tensors
# ``mxfp4-pack-quantized`` config with no ModelOpt per-layer entries, and that
# checkpoint is what this default has always served.
_K3_DEFAULT_ROUTED_QUANT_ALGO = QuantAlgo.W4A8_MXFP4_MXFP8


# ---------------------------------------------------------------------------
# Routed-expert checkpoint layout.
#
# K3 streams routed experts one at a time (see ``load_weights``): the tensor names, the per-expert
# loader and the per-layer finalize it needs are decided by the checkpoint's routed-expert
# quantization. The expected-key plan, the loader and the file grouping read them off one spec.
# ---------------------------------------------------------------------------


def _load_nvfp4_expert(
    backend, base, expert_idx, local_slot_id, get_tensor, get_optional_tensor
) -> None:
    for w in ("w1", "w3"):
        if get_optional_tensor(f"{base}.{expert_idx}.{w}.pre_quant_scale") is not None:
            raise NotImplementedError(
                "Kimi K3 routed experts take plain NVFP4 checkpoints; this one carries a "
                "pre-quant activation scale (NVFP4_AWQ / NVFP4_ARC)"
            )
    tensors = {
        f"{w}_{kind}": get_tensor(f"{base}.{expert_idx}.{w}.{kind}")
        for w in ("w1", "w2", "w3")
        for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    }
    backend.load_expert(global_expert_id=expert_idx, local_slot_id=local_slot_id, **tensors)


def _k3_fc12_options(kernels: str) -> Dict[str, Any]:
    """The FC12 options of the routed experts (``K3RoutedExperts`` keyword arguments).

    ``sm_reserve``: SMs the FC12 kernels' persistent grids leave to the kernels the shared stream
    runs beside them (the shared expert's GEMMs confined to ``_K3_SHARED_SM`` SMs).
    ``die_major``: die-alternating expert weights, with die-local tile claiming from
    :data:`_K3_FC12_DIE_SCHED_MIN_TOKENS` tokens per call. The CUTEDSL kernels take neither.
    """
    if kernels != "CUTEDSL_FC12":
        return {}
    return {
        "sm_reserve": _K3_SHARED_SM,
        "die_major": True,
        "die_sched_min_tokens": _K3_FC12_DIE_SCHED_MIN_TOKENS,
    }


class _K3ExpertCkptSpec(NamedTuple):
    """How one routed-expert quantization is spelled and loaded."""

    # Per-``w{1,2,3}`` checkpoint tensor suffixes this layout stores. Every
    # combination is required: a missing one fails the key plan.
    kinds: Tuple[str, ...]
    loader: Callable[..., None]
    # Set of filled slots the loader maintains, checked after the load.
    loaded_slots_attr: str
    # The loader stages per-expert state that the layer's finalize consumes.
    needs_layer_finalize: bool
    # Whole per-expert key suffixes the loader reads WHEN PRESENT (to reject them).
    optional_suffixes: Tuple[str, ...] = ()


_K3_EXPERT_CKPT_SPECS = {
    QuantAlgo.NVFP4: _K3ExpertCkptSpec(
        kinds=("weight", "weight_scale", "weight_scale_2", "input_scale"),
        loader=_load_nvfp4_expert,
        loaded_slots_attr="_streamed_expert_slots",
        needs_layer_finalize=True,
        optional_suffixes=("w1.pre_quant_scale", "w3.pre_quant_scale"),
    ),
}


class _RpTailOperands(NamedTuple):
    """One rank's operands of the row-parallel MoE tail GEMM: K is this
    rank's ``k1`` latent columns from column ``k0`` (``k1 / 128`` whole blocks of latent_up, whose
    FP8 codes the GEMM reads in place); the GEMM's push epilogue adds the shared expert's BF16
    partial."""

    k0: int
    k1: int
    # DeepGEMM-packed UE8M0 scale of the [hidden, k1] tail weight.
    scale: torch.Tensor


def _rp_tail_latent_slice(latent_size: int, world: int, rank: int) -> Tuple[int, int]:
    """``(first column, columns)`` of ``rank``'s 128-column blocks of the latent: whole blocks,
    as even as possible (28 blocks over 8 ranks: 4 on ranks 0-3, 3 on ranks 4-7)."""
    blocks = latent_size // 128
    base, extra = divmod(blocks, world)
    first = rank * base + min(rank, extra)
    return first * 128, (base + (rank < extra)) * 128


def _k3_expert_ckpt_spec(quant_config: QuantConfig) -> _K3ExpertCkptSpec:
    spec = _K3_EXPERT_CKPT_SPECS.get(quant_config.quant_algo)
    if spec is None:
        raise NotImplementedError(
            f"Kimi K3 routed experts are quantized as {quant_config.quant_algo}, for which "
            "no per-expert checkpoint layout is known. Supported: "
            f"{sorted(a.name for a in _K3_EXPERT_CKPT_SPECS)}."
        )
    group_size = getattr(quant_config, "group_size", None)
    if group_size not in (None, 16):
        raise NotImplementedError(
            f"Kimi K3 NVFP4 routed experts take 16-element scale blocks, got group_size={group_size}"
        )
    return spec


class _K3SharedExpert(GatedMlp):
    """The MoE layer's shared expert: :class:`GatedMlp` called through its projection modules,
    so a projection ``_convert_moe_mlps_to_fp8_weight_read`` swapped for an FP8 weight read runs
    its own kernel (a ``Bf16Linear`` without bias or all-reduce is ``F.linear`` either way)."""

    def forward(
        self, x: torch.Tensor, final_all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        h = self.gate_up_proj(x)
        a = gated_activation(h, self.activation, self.beta, self.linear_beta)
        return self.down_proj(a, all_reduce_params=final_all_reduce_params)


class K3MoERuntime(nn.Module):
    """Kimi K3 latent MoE block: the gate routes, the routed experts run the CuTe DSL kernel chain
    of ``ops/moe/k3r_experts``, the shared expert runs beside them."""

    # The routed latent all-reduce's own MNNVL workspace (``None``: no routed all-reduce over
    # more than one rank).
    _latent_ar_ws: Optional[Dict[str, Any]] = None
    # Whether this layer's router logits come from k3_router_exact (the KDA layers; set in
    # ``__init__``).
    _router_exact = False
    # The row-parallel tail: this rank's tail GEMM operands, built after loading (``None``: not
    # applicable to this layer).
    _rp_tail: Optional[_RpTailOperands] = None

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        """Build the routed experts and the shared expert for one MoE layer.

        ``cfg`` is the raw ``PretrainedConfig`` rather than anything derived:
        the SiTU soft-caps and the routed-expert geometry are Kimi K3 fields
        that ``ModelConfig`` does not carry.

        ``aux_stream_dict`` is shared across every layer of the model, so the
        streams reached through it are borrowed and must not be synchronized
        or reassigned here.
        """
        super().__init__()
        if model_config.mapping.enable_attention_dp:
            raise NotImplementedError(
                "the Kimi K3 helix8 MoE runs without attention DP (enable_attention_dp=false)"
            )
        self.layer_idx = layer_idx
        self._router_exact = _is_kda_layer(cfg, layer_idx)
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_token
        self.moe_hidden_size = cfg.routed_expert_hidden_size
        # ValueError (not assert): these guard unsupported checkpoint
        # configurations and must stay active under ``python -O``.
        if self.moe_hidden_size is None:
            raise ValueError("Kimi K3 runtime expects the latent MoE (routed_expert_hidden_size)")
        if not getattr(cfg, "latent_moe_use_norm", False):
            raise ValueError("Kimi K3 runtime expects latent_moe_use_norm=True")

        situ_beta, situ_linear_beta = _resolve_kimi_situ_betas(cfg)
        dtype = torch.bfloat16

        # Routing scores stay fp32: the gate GEMM runs bf16xbf16 with fp32 accumulate/output (the
        # checkpoint stores the gate weight in bf16; saves a per-layer input cast + fp32 splitK
        # pair on the bs1 decode path).
        self.gate = K3MoEGate(cfg)

        routed_moe_model_config = self._routed_moe_model_config(model_config)
        routed_quant_config = self._resolve_routed_quant_config(model_config, layer_idx)
        # Resolved here so ``load_weights`` reads the checkpoint layout off the
        # module instead of re-deriving it at each of its three call sites.
        self.expert_ckpt_spec = _k3_expert_ckpt_spec(routed_quant_config)
        self._moe_backend = routed_moe_model_config.moe_backend
        routed_mapping = routed_moe_model_config.mapping
        if routed_moe_model_config.moe_disable_finalize_fusion:
            raise NotImplementedError(
                "Kimi K3 routed experts run the fused-finalize kernels "
                "(moe_disable_finalize_fusion / FORCE_DETERMINISTIC is not supported)"
            )
        from tensorrt_llm._torch.locality_domain.policy import LocalityDomainExecutionPlanner

        locality_plan = LocalityDomainExecutionPlanner(
            routed_moe_model_config.locality_domain_policy
        ).plan_moe(
            routed_quant_config,
            moe_backend=self._moe_backend,
            use_fused_finalize=True,
            dtype_activation=dtype,
            activation=ActivationType.SiTu.name,
        )
        if locality_plan.enabled:
            raise NotImplementedError(
                "Kimi K3 routed experts do not run the locality-domain MoE kernels; "
                "disable enable_locality_domains"
            )
        self.routed_experts = K3RoutedExperts(
            kernels=self._moe_backend,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.moe_hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            mapping=routed_mapping,
            layer_idx=layer_idx,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            aux_stream=aux_stream_dict.get(AuxStreamType.MoeOutputMemset),
            max_chunk_tokens=routed_moe_model_config.moe_max_num_tokens,
            **_k3_fc12_options(self._moe_backend),
        )
        # The routed latent partial sums, reduced over the TP group (Kimi orders this
        # collective after the shared expert's).
        self.routed_experts.all_reduce = (
            TpAllReduce(routed_mapping, dtype, routed_moe_model_config.allreduce_strategy)
            if routed_mapping.tp_size > 1
            else None
        )
        local_expert_ids = list(self.routed_experts.backend.initial_local_expert_ids)
        if local_expert_ids != list(
            range(local_expert_ids[0], local_expert_ids[0] + len(local_expert_ids))
        ):
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming currently requires a "
                "contiguous static expert partition."
            )
        self.local_expert_ids = tuple(local_expert_ids)
        self.experts_per_rank = len(local_expert_ids)
        self.expert_lo = local_expert_ids[0]
        self.expert_hi = self.expert_lo + self.experts_per_rank

        shared_intermediate = cfg.moe_intermediate_size * cfg.num_shared_experts
        self._shared_intermediate = shared_intermediate
        # Direct MoE-TP leaves both branches as partials for one concatenated
        # all-reduce.
        use_shared_tp = model_config.mapping.tp_size > 1
        self._reduce_routed_output = use_shared_tp
        if self._reduce_routed_output and self.routed_experts.all_reduce is None:
            raise RuntimeError(
                "Kimi K3 direct MoE tensor parallelism requires the routed all-reduce."
            )
        if model_config.use_cute_dsl_bf16_gemm:
            raise NotImplementedError(
                "Kimi K3's shared expert runs the cuBLAS BF16 GEMMs "
                "(use_cute_dsl_bf16_gemm is not supported)"
            )
        self.shared_experts = _K3SharedExpert(
            cfg.hidden_size,
            shared_intermediate,
            activation="situ",
            beta=situ_beta,
            linear_beta=situ_linear_beta,
            dtype=dtype,
            mapping=model_config.mapping,
            reduce_output=use_shared_tp,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        shared_forward = self.shared_experts.forward

        def _shared_forward_confined(*args, **kwargs):
            with _k3_shared_sm_scope():
                return shared_forward(*args, **kwargs)

        self.shared_experts.forward = _shared_forward_confined
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
        # CuteDSL already quantizes the MoE input on the MoeOutputMemset stream;
        # the latent down-projection runs there too, beside the routing chain.
        self.latent_down_stream = (
            aux_stream_dict.get(AuxStreamType.MoeOutputMemset)
            if self._moe_backend in ("CUTEDSL", "CUTEDSL_FC12")
            else None
        )
        has_latent_stream = self.latent_down_stream is not None
        self.moe_latent_fork_event = torch.cuda.Event() if has_latent_stream else None
        self.moe_latent_ready_event = torch.cuda.Event() if has_latent_stream else None
        self.routed_expert_down_proj = nn.Linear(
            cfg.hidden_size, self.moe_hidden_size, bias=False, dtype=dtype
        )
        self.routed_expert_up_proj = nn.Linear(
            self.moe_hidden_size, cfg.hidden_size, bias=False, dtype=dtype
        )
        # FlashInfer RMSNorm on the reduced latent (or folded into the routed all-reduce).
        self.routed_expert_norm = RMSNormWeight(self.moe_hidden_size, cfg.rms_norm_eps, dtype=dtype)
        if self._reduce_routed_output:
            mnnvl = getattr(self.routed_experts.all_reduce, "mnnvl_allreduce", None)
            if mnnvl is not None and mnnvl.mapping.tp_size > 1:
                self._latent_ar_ws = _k3_latent_ar_workspace(
                    mnnvl.mapping, mnnvl.dtype, self.moe_hidden_size, _K3_LATENT_AR_OWN_WS_MAX_M
                )

    def _lu_mid_prequant_plan(self, latent: torch.Tensor):
        """``(tactic, sf_blocks)`` of the mid-M path that
        latent_up's :meth:`_Fp8BlockScaleWeightReadLinear.forward` runs on the BF16 rows ``latent``
        (SM107: the K3 low-M GEMM on ``lowm_weight_scale`` above the low-M cut-over),
        or ``None`` where it runs another GEMM."""
        up = self.routed_expert_up_proj
        if (
            not isinstance(up, _Fp8BlockScaleWeightReadLinear)
            or up.lowm_weight_scale is None
            or up.next_ar_push is not None
        ):
            return None
        if not up._lowm_gemm_applies(latent, up.lowm_weight_scale):
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        return k3_fp8w_lowm_gemm.mid_prequant_plan(latent.shape[0], up.out_features, up.in_features)

    def _latent_up_mid_prequantized(
        self, routed_latent: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor, tactic
    ) -> torch.Tensor:
        """latent_up on the MXFP8 form ``(xq, xs)`` of ``routed_latent`` in the mid-M GEMM's
        layout: what :meth:`_routed_projection` computes, bit for
        bit, without its quantization kernel."""
        up = self.routed_expert_up_proj
        m = xq.shape[0]
        out = up._take_next_output(m)
        if out is None:
            out = torch.empty((m, up.out_features), dtype=torch.bfloat16, device=xq.device)
        k3_fp8w_mid_gemm_op(xq, xs, up.weight, up.lowm_weight_scale, out, tactic)
        return out.reshape(routed_latent.shape[:-1] + (up.out_features,))

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 router logits; in the KDA layers ``k3_router_exact`` serves ``M <= 16`` BF16
        tokens (bit-identical to the ``dsv3_router_gemm_op`` it replaces)."""
        if self._router_exact:
            from ..kernels.k3r_moe import k3_router_exact

            x = hidden_states.reshape(-1, self.gate.gating_dim)
            if k3_router_exact.can_run(x, self.gate.weight):
                return k3_router_logits_op(x, self.gate.weight)
        return self.gate.compute_logits(hidden_states)

    def _shared_needs_all_reduce(self) -> bool:
        """Whether the shared expert's output is a row-parallel partial to all-reduce."""
        return bool(getattr(self.shared_experts.down_proj, "reduce_output", False))

    def post_load_weights(self) -> None:
        self.prepare_rp_tail()

    def cache_derived_state(self) -> None:
        self.prepare_rp_tail()

    def prepare_expert_load(self) -> None:
        """Before this layer's first routed expert streams in."""
        self.routed_experts.backend.prepare_streaming_load()

    def finalize_expert_load(self) -> None:
        """After this layer's last routed expert streamed in."""
        self.routed_experts.backend.finalize()

    def _rp_tail_unsupported(self) -> Optional[str]:
        """Why this layer cannot run the row-parallel MoE tail, or ``None``."""
        if not self._reduce_routed_output:
            return "the routed latent is not all-reduced by the model"
        moe_all_reduce = self.routed_experts.all_reduce
        if getattr(moe_all_reduce, "mnnvl_allreduce", None) is None:
            return "the routed all-reduce is not the MNNVL all-reduce"
        up = self.routed_expert_up_proj
        if not isinstance(up, _Fp8BlockScaleWeightReadLinear) or up._fused_projection is not None:
            return (
                "latent_up is not an FP8 weight-read projection (KIMI_K3_FP8_WEIGHT_READ_MOE_MLP)"
            )
        mlp = self.shared_experts
        if not isinstance(mlp, _K3SharedExpert) or not isinstance(mlp.down_proj, Bf16Linear):
            return "the shared expert does not run the BF16 gate_up -> activation -> down path"
        if not self._shared_needs_all_reduce():
            return "the shared expert is not row-parallel"
        down = mlp.down_proj
        if getattr(down, "bias", None) is not None:
            return "the shared down projection has a bias"
        world = moe_all_reduce.mapping.tp_size
        shard = down.weight.shape[1]
        if shard * world != self._shared_intermediate:
            return f"the shared down shard ({shard}) is not a 1/{world} shard"
        if (
            tuple(up.weight.shape) != (self.hidden_size, self.moe_hidden_size)
            or self.moe_hidden_size % 128
            or self.hidden_size % 128
            or self.moe_hidden_size // 128 < world
        ):
            return "the latent is not split into whole 128-column blocks over the ranks"
        if tuple(down.weight.shape) != (self.hidden_size, shard):
            return "unexpected shared down weight shape"
        return None

    def prepare_rp_tail(self) -> bool:
        """Build this rank's row-parallel tail operands (``_RpTailOperands``): the packed UE8M0
        scale of this rank's latent_up blocks (whose FP8 codes stay where they are). Rebuilt on
        every call (post-load), before CUDA graph capture."""
        self._rp_tail = None
        reason = self._rp_tail_unsupported()
        if reason is not None:
            logger.info_once(
                f"Kimi K3 MoE layers keep the replicated tail (no row-parallel tail): {reason}",
                key=f"k3_rp_tail_off_{reason}",
            )
            return False
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        up = self.routed_expert_up_proj
        up.transform_weights()
        down = self.shared_experts.down_proj
        mapping = self.routed_experts.all_reduce.mapping
        k0, k1 = _rp_tail_latent_slice(self.moe_hidden_size, mapping.tp_size, mapping.tp_rank)
        hidden, shard = down.weight.shape
        with torch.no_grad():
            exponents = k3_fp8w_lowm_gemm.block_exponents_of(
                up.weight_scale, hidden, self.moe_hidden_size
            )[:, k0 // 128 : (k0 + k1) // 128]
            scale = k3_fp8w_lowm_gemm.pack_ue8m0_exponents(exponents.contiguous(), hidden, k1)
        self._rp_tail = _RpTailOperands(k0, k1, scale)
        logger.info_once(
            f"Kimi K3 row-parallel MoE tail, K = latent [{k0}, {k0 + k1}), BF16 shared "
            f"partial added (rank {mapping.tp_rank} of {mapping.tp_size})",
            key=f"k3_rp_tail_on_{k1}_{shard}",
        )
        return True

    def _rp_tail_target(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor], rp_tail_ok: bool
    ):
        """The all-reduce push target of this forward's row-parallel MoE tail, or ``None`` where
        the forward keeps the replicated tail: a generation-only batch of at most
        ``_K3_RP_TAIL_MAX_M`` BF16 tokens with the fused residual update, served by the low-M
        GEMM's push epilogue. The consumer finishes the call: one-shot transport up to
        lamport._K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES, two-shot above."""
        rp = self._rp_tail
        num_tokens = hidden_states.shape[0]
        if (
            rp is None
            or not rp_tail_ok
            or residual is None
            or hidden_states.dtype != torch.bfloat16
            or not 1 <= num_tokens <= _K3_RP_TAIL_MAX_M
        ):
            return None
        from ..kernels.k3r_gemm import k3_fp8w_lowm_gemm

        if k3_fp8w_lowm_gemm.split_source_tactic(num_tokens, self.hidden_size, rp.k1) is None:
            return None
        return lamport_push_target(
            self.routed_experts.all_reduce,
            num_tokens,
            self.hidden_size,
            deferred=True,
        )

    def _rp_shared_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The shared expert's part of the row-parallel tail on this rank: its row-parallel BF16
        down-projection partial without the all-reduce."""
        return self.shared_experts(
            hidden_states, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
        )

    def _rp_tail_forward(
        self,
        routed_output: Callable[[], torch.Tensor],
        identity: torch.Tensor,
        residual: torch.Tensor,
        target,
        defer_residual: bool,
    ):
        """The row-parallel tail: the routed experts (``routed_output``) and the shared expert's
        part (``_rp_shared_output``), then the latent all-reduce + RMSNorm, then this rank's tail
        GEMM pushing its partial into ``target``. Returns a ``_DeferredRpTail``
        (``defer_residual``) or the updated residual."""
        if self.shared_expert_stream is not None and do_multi_stream():
            self.moe_main_event.record()
            routed_out = routed_output()
            with torch.cuda.stream(self.shared_expert_stream):
                self.moe_main_event.wait()
                shared_part = self._rp_shared_output(identity)
                self.moe_shared_event.record()
            self.moe_shared_event.wait()
        else:
            routed_out = routed_output()
            shared_part = self._rp_shared_output(identity)
        moe_all_reduce = self.routed_experts.all_reduce
        routed_latent = moe_all_reduce(
            routed_out,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RMS_NORM,
                norm_weight=self.routed_expert_norm.weight,
                eps=float(self.routed_expert_norm.variance_epsilon),
            ),
        )
        latent = routed_latent.reshape(-1, self.moe_hidden_size)
        num_tokens = latent.shape[0]
        shared_part = shared_part.reshape(num_tokens, -1)
        if not shared_part.is_contiguous():
            shared_part = shared_part.contiguous()
        rp = self._rp_tail
        lo, hi = rp.k0, rp.k0 + rp.k1
        k3_fp8w_lowm_gemm_push_op(
            latent[:, lo:hi],
            self.routed_expert_up_proj.weight[:, lo:hi],
            rp.scale,
            target.workspace,
            target.one_shot,
            addend=shared_part,
            strided=True,
        )
        target.pushed = True
        result = finish_lamport_push(moe_all_reduce, target, deferred=True)
        deferred = _DeferredRpTail(residual.reshape(num_tokens, -1), result)
        if defer_residual:
            return deferred
        return deferred.materialize().view(residual.shape)

    def _new_fc12_prezero(self, hidden_states: torch.Tensor):
        """The routed output and FC12 counters of this forward (``Fc12ZeroedOutput``), allocated
        on the current stream and not zeroed yet, or ``None`` where FC12 zeroes them itself (not
        the FC12 kernels, no tokens)."""
        moe = self.routed_experts
        num_tokens = hidden_states.shape[0]
        if getattr(moe, "kernels", None) != "CUTEDSL_FC12" or num_tokens == 0:
            return None
        return moe.new_zeroed_output(num_tokens, hidden_states.device)

    def _route_tail_target(self, num_tokens: int) -> Optional[Tuple[int, int]]:
        """``(local_expert_offset, local_num_experts)`` if the CuTe routing tail serves this
        forward, else ``None``."""
        backend = self.routed_experts.backend
        slot_start = backend.slot_start
        num_local = backend.expert_size_per_partition
        if backend.num_slots != self.gate.num_experts:
            return None
        from ..kernels.k3r_moe import k3_route_tail

        device = self.gate.e_score_correction_bias.device
        if not k3_route_tail.supports_device(device):
            return None
        if not 1 <= num_tokens <= min(_K3_ROUTE_TAIL_MAX_M, k3_route_tail.WIDE_MAX_M):
            return None
        if num_tokens > k3_route_tail.MAX_M:
            if not k3_route_tail.wide_sort_fits(num_tokens, num_local):
                return None
            # The multi-CTA kernel's workspace must exist before graph capture.
            if not torch.cuda.is_current_stream_capturing():
                k3_route_tail.wide_workspace(device)
            elif not k3_route_tail.wide_workspace_ready(device):
                return None
        return slot_start, num_local

    def _router(self, tail_target: Optional[Tuple[int, int]]):
        """The routing of this forward's router logits, for the experts
        ``[local_expert_offset, local_expert_offset + local_num_experts)`` the routed experts ask
        for: the CuTe routing tail where ``tail_target`` is set (with its ``moe_sort`` tables where
        they are of the experts asked for), else ``noaux_tc_op`` (the experts then run
        ``moe_sort``)."""
        gate = self.gate
        consumes_tables = self.routed_experts.consumes_sort_tables

        def route(logits: torch.Tensor, local_expert_offset: int, local_num_experts: int):
            target = (local_expert_offset, local_num_experts)
            if tail_target is not None:
                routed = gate.route_tail(logits, *tail_target)
                if consumes_tables and target == tail_target:
                    return routed
                return SortedRouting(routed.token_selected_experts, routed.token_final_scales, None)
            return SortedRouting(*gate.route(logits), None)

        return route

    def _fused_projection_applies(self, x: torch.Tensor, projection: nn.Module) -> bool:
        """Whether the fused latent projection replaces ``projection(x)``."""
        if not isinstance(projection, _Fp8BlockScaleWeightReadLinear):
            return False
        packed_scale = projection.packed_weight_scale
        if packed_scale is None or x.dtype != torch.bfloat16:
            return False
        if not _K3_LD_FUSED_MIN_M <= x.shape[0] <= _K3_LD_FUSED_MAX_M:
            return False
        from ..kernels.k3r_gemm import k3_latent_fp8_gemm

        # Where the fused op itself runs the low-M GEMM (SM107), it only adds the epilogue.
        if x.shape[0] <= k3_latent_fp8_gemm.lowm_route_max_m(*projection.weight.shape):
            return True
        if _k3_lowm_gemm_can_run(x, projection.weight, packed_scale):
            return False
        return True

    def _fused_latent_input(self, hidden_states: torch.Tensor) -> Optional[Fp4QuantizedTensor]:
        """NVFP4 routed-expert input straight from the hidden states, or ``None``.

        Runs the latent_down projection with the experts' input quantization fused in
        (``ops.gemm.k3_latent_down_fp4``) when enabled, for the FC12 routed experts (which quantize
        their input with ``fp4_quantize`` against ``fc31_input_scale``) of one launch chain.
        """
        down = self.routed_expert_down_proj
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        if not self._fused_projection_applies(x, down):
            return None
        moe = self.routed_experts
        if getattr(moe, "kernels", None) != "CUTEDSL_FC12" or x.shape[0] > moe.max_chunk_tokens:
            return None
        from ..kernels.k3r_gemm import k3_latent_fp8_gemm

        global_sf = moe.backend.fc31_input_scale
        w_scale = down.packed_weight_scale
        if not k3_latent_fp8_gemm.can_run(x, down.weight, w_scale, global_sf):
            return None
        codes, scales = k3_latent_down_fp4_op(x, down.weight, w_scale, global_sf)
        return Fp4QuantizedTensor(codes, scales, is_sf_swizzled=False)

    def _own_workspace_latent_tail(
        self,
        routed_out: torch.Tensor,
        shared_out: torch.Tensor,
        residual: Optional[torch.Tensor],
        defer_residual: bool,
    ):
        """The routed latent all-reduce + RMSNorm on its own MNNVL
        workspace (``k3_latent_ar_norm`` up to ``_K3_LATENT_AR_ONESHOT_MAX_BYTES``, the C++ fused
        all-reduce otherwise) and latent_up, issued on the main stream before it joins the shared-expert
        stream, whose output only the residual update reads. Returns what ``forward`` returns."""
        workspace = self._latent_ar_ws
        gamma = self.routed_expert_norm.weight
        eps = float(self.routed_expert_norm.variance_epsilon)
        latent = routed_out.reshape(-1, routed_out.shape[-1]).contiguous()
        routed_latent = None
        latent_mxfp8 = None
        mid_plan = None
        from ..kernels.k3r_comm import k3_latent_ar_norm

        world_size = workspace["world_size"]
        one_shot = (
            latent.numel() * world_size * latent.element_size() <= _K3_LATENT_AR_ONESHOT_MAX_BYTES
        )
        row = k3_latent_ar_norm.can_run_row(latent, gamma, world_size, workspace["buffer_bytes"])
        if one_shot and (row or k3_latent_ar_norm.can_run(latent, gamma, world_size)):
            routed_latent = torch.empty_like(latent)
            m, k = latent.shape
            if row:
                # The MXFP8 operand of latent_up's mid-M GEMM, in its scale layout.
                mid_plan = self._lu_mid_prequant_plan(latent)
                if mid_plan is not None:
                    latent_mxfp8 = (
                        torch.empty((m, k), dtype=torch.float8_e4m3fn, device=latent.device),
                        torch.empty(
                            (k // 128) * mid_plan[1] * 512,
                            dtype=torch.uint8,
                            device=latent.device,
                        ),
                    )
            k3_latent_ar_rmsnorm(
                latent,
                gamma,
                eps,
                routed_latent,
                uc_addr=workspace["buffer"].data_ptr(),
                mc_addr=workspace["mc"].data_ptr(),
                flags=workspace["flags"],
                rank=workspace["rank"],
                world_size=world_size,
                mxfp8_out=latent_mxfp8,
                sf_blocks=mid_plan[1] if mid_plan is not None else 0,
                row=row,
            )
        if routed_latent is None:
            (routed_latent,) = torch.ops.trtllm.mnnvl_fusion_allreduce(
                latent,
                gamma,
                None,
                eps,
                workspace["buffer"],
                workspace["flags"],
                True,
                None,
                int(AllReduceFusionOp.RMS_NORM),
            )
        routed_latent = routed_latent.view(routed_out.shape)
        if mid_plan is not None:
            routed_out = self._latent_up_mid_prequantized(routed_latent, *latent_mxfp8, mid_plan[0])
        else:
            routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        self.moe_shared_event.wait()
        if residual is not None:
            if defer_residual:
                return _DeferredMoEResidual(residual, routed_out, shared_out)
            return k3_moe_residual_add_op(routed_out, shared_out, residual)
        return routed_out + shared_out

    @staticmethod
    def _routed_projection(hidden_states: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if not isinstance(projection, nn.Linear):
            return projection(hidden_states)
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(
            hidden_states, projection.weight.t(), None, None
        )

    @staticmethod
    def _select_moe_tp_ep(mapping: Mapping) -> Tuple[int, int]:
        """Resolve the routed-expert ``(moe_tp, moe_ep)`` split.

        Precedence:

        1. Explicit ``moe_tensor_parallel_size`` / ``moe_expert_parallel_size``
           from the user config. Detected via
           ``mapping.moe_tp_ep_user_specified`` so the auto-resolved mapping
           default (``moe_tp=tp_size, moe_ep=1``) is NOT mistaken for a TP
           request.
        2. Default: EP-only (``moe_tp=1, moe_ep=tp_size``), the historical
           K3 layout.
        """
        tp_size = mapping.tp_size
        if getattr(mapping, "moe_tp_ep_user_specified", False):
            return mapping.moe_tp_size, mapping.moe_ep_size
        return 1, tp_size

    @staticmethod
    def _resolve_routed_quant_config(model_config: ModelConfig, layer_idx: int) -> QuantConfig:
        """Routed-expert quantization for ``layer_idx``, taken from the checkpoint.

        ``nvidia/Kimi-K3-NVFP4`` declares the routed experts per layer as
        ``NVFP4`` with ``group_size=16``; the original ``moonshotai/Kimi-K3``
        declares nothing per layer and keeps the historical
        ``W4A8_MXFP4_MXFP8`` default. Reading the checkpoint instead of
        hardcoding is what lets one code path serve both.

        An exclusion outranks the per-layer entry and the default below:
        ``create_weights`` treats an override as authoritative over anything
        ``__post_init__`` wrote, so this return value stands in for both
        quantization passes and exclusion is the one that runs second. It is
        matched as a pattern, so it is asked only about real module names.
        """
        quant_config = model_config.quant_config
        if quant_config is not None and any(
            quant_config.is_module_excluded_from_quantization(
                f"{prefix}layers.{layer_idx}.{suffix}"
            )
            for prefix in _K3_ROUTED_EXPERT_MODULE_PREFIXES
            for suffix in _K3_ROUTED_EXPERT_KEY_SUFFIXES
        ):
            logger.debug(
                "Kimi K3 layer %d routed experts: excluded from quantization, "
                "keeping them unquantized",
                layer_idx,
            )
            return QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)

        per_layer = getattr(model_config, "quant_config_dict", None)
        if per_layer:
            for prefix in _K3_ROUTED_EXPERT_KEY_PREFIXES:
                for suffix in _K3_ROUTED_EXPERT_KEY_SUFFIXES:
                    cfg = per_layer.get(f"{prefix}layers.{layer_idx}.{suffix}")
                    if cfg is not None and cfg.quant_algo is not None:
                        # Logged once per layer: the routed-expert format decides
                        # which MoE backends can serve this checkpoint at all.
                        logger.debug(
                            "Kimi K3 layer %d routed experts: %s (group_size=%s) "
                            "from the checkpoint",
                            layer_idx,
                            cfg.quant_algo,
                            cfg.group_size,
                        )
                        return cfg
        logger.debug(
            "Kimi K3 layer %d routed experts: no per-layer quant config in the "
            "checkpoint, defaulting to %s",
            layer_idx,
            _K3_DEFAULT_ROUTED_QUANT_ALGO,
        )
        return QuantConfig(quant_algo=_K3_DEFAULT_ROUTED_QUANT_ALGO)

    @staticmethod
    def _routed_moe_model_config(model_config: ModelConfig) -> ModelConfig:
        """Build a private routed-expert mapping without mutating the shared
        config. Default split is EP-only; see ``_select_moe_tp_ep``."""
        # The CuTe DSL kernel sets K3 runs (generation: CUTEDSL_FC12).
        if model_config.moe_backend not in _K3_ROUTED_KERNEL_SETS:
            raise ValueError(
                f"Kimi K3 SiTU routed experts run on {_K3_ROUTED_KERNEL_SETS}; "
                f"got {model_config.moe_backend!r}."
            )
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        mapping = model_config.mapping
        # DWDP (setup_dwdp() after loading) collects w3_w1_weight / w2_weight off the CUTEDSL /
        # CUTEDSL_FC12 backends that K3's streaming produces: compatible.
        moe_tp, moe_ep = K3MoERuntime._select_moe_tp_ep(mapping)
        if moe_tp < 1 or moe_ep < 1 or moe_tp * moe_ep != mapping.tp_size:
            raise ValueError(
                f"Kimi K3 routed MoE split moe_tp={moe_tp} x moe_ep={moe_ep} "
                f"must multiply to tp_size={mapping.tp_size}."
            )
        logger.info_once(
            f"Kimi K3 routed MoE parallelism: moe_tp={moe_tp}, "
            f"moe_ep={moe_ep} (tp_size={mapping.tp_size})",
            key="kimi_k3_moe_tp_ep_split",
        )

        mapping_dict = mapping.to_dict()
        mapping_dict["moe_cluster_size"] = 1
        mapping_dict["moe_tp_size"] = moe_tp
        mapping_dict["moe_ep_size"] = moe_ep
        routed_mapping = Mapping.from_dict(mapping_dict)

        routed_model_config = copy.copy(model_config)
        routed_model_config._frozen = False
        routed_model_config.extra_attrs = copy.copy(model_config.extra_attrs)
        routed_model_config.mapping = routed_mapping
        routed_model_config.moe_backend = model_config.moe_backend
        # The routed experts keep the user-configured moe_max_num_tokens as their chunking bound.
        routed_model_config._frozen = True
        return routed_model_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        *,
        residual: Optional[torch.Tensor] = None,
        defer_residual: bool = False,
        rp_tail_ok: bool = False,
    ) -> torch.Tensor:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16.

        With ``residual``, returns ``residual + moe_output``; with ``defer_residual`` as well, a
        ``_DeferredMoEResidual`` instead where that update would be the separate BF16 add, or a
        ``_DeferredRpTail`` where the row-parallel tail runs (the caller allows it with
        ``rp_tail_ok`` for generation-only batches).
        """
        identity = hidden_states
        rp_target = self._rp_tail_target(hidden_states, residual, rp_tail_ok)
        early_routed_in = None
        zeroed = None
        route_target = self._route_tail_target(hidden_states.shape[0])
        router_issued = False
        if self.latent_down_stream is not None and do_multi_stream():
            # Forked ahead of the router GEMM: the latent chain needs only
            # hidden_states. The FC12 output and counters are allocated before the fork and
            # zeroed on the pre-zeroing stream beside the latent chain and the routing, so
            # FC12 directly follows the routing.
            zeroed = self._new_fc12_prezero(hidden_states)
            self.moe_latent_fork_event.record()
            if zeroed is not None:
                zero_on_side_stream(zeroed, self.moe_latent_fork_event)
            if hidden_states.shape[0] <= _K3_ROUTER_FIRST_MAX_M:
                # Router before latent_down in creation order (see _K3_ROUTER_FIRST_MAX_M).
                router_logits = self._router_logits(hidden_states)
                router_issued = True
            with torch.cuda.stream(self.latent_down_stream):
                self.moe_latent_fork_event.wait()
                early_routed_in = self._fused_latent_input(hidden_states)
                if early_routed_in is None:
                    early_routed_in = self._routed_projection(
                        hidden_states, self.routed_expert_down_proj
                    )
                self.moe_latent_ready_event.record()
            # No record_stream (under graph capture it holds every layer's
            # blocks until capture ends): both tensors are read on main only
            # after the join, and later work on this stream forks from main.
        if not router_issued:
            router_logits = self._router_logits(hidden_states)
        moe_all_reduce = self.routed_experts.all_reduce if self._reduce_routed_output else None
        num_tokens = hidden_states.shape[0]

        def _routed_output():
            # Latent down/up projections via the min-latency fused GEMM op:
            # at <=16 tokens (decode graphs) it runs a single pipelined
            # bf16 kernel per projection instead of cuBLAS's split-K GEMV +
            # splitKreduce pair (~17+3.6us -> ~8us for 7168->3584 at M=1);
            # for larger token counts the op falls back to cuBLAS internally.
            # When the FP8 weight-read conversion has
            # replaced the projection module, call it directly: its weight is
            # an e4m3 buffer the bf16 dsv3 op must not read, and its forward
            # is already a single fused GEMM (fp8_swap_ab_gemm).
            logits = router_logits
            moe_kwargs = {}
            if early_routed_in is not None:
                routed_in = early_routed_in
                # The routed experts join the event before their first read of x.
                moe_kwargs = {"input_ready_event": self.moe_latent_ready_event}
            else:
                routed_in = self._fused_latent_input(hidden_states)
                if routed_in is None:
                    routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
            if isinstance(routed_in, Fp4QuantizedTensor):
                moe_kwargs["output_dtype"] = hidden_states.dtype
            # Routed by the CuTe tail (route_target), the sorted routing kernel or
            # noaux_tc_op; then the kernels.
            y = self.routed_experts(
                routed_in,
                logits,
                route=self._router(route_target),
                input_ready_event=moe_kwargs.get("input_ready_event"),
                zeroed_output=zeroed,
                all_rank_num_tokens=all_rank_num_tokens,
            )
            if self._reduce_routed_output:
                return y
            # Communication-backed paths return a complete routed result.
            y = self.routed_expert_norm(y)
            return self._routed_projection(y, self.routed_expert_up_proj)

        def _shared_output():
            """The shared expert's (reduced) output."""
            return self.shared_experts(identity, final_all_reduce_params=None)

        if rp_target is not None:
            return self._rp_tail_forward(
                _routed_output, identity, residual, rp_target, defer_residual
            )
        # The final join includes the shared all-reduce, keeping it ordered before any outer
        # routed reduction or the next layer's collectives.
        if (
            self._latent_ar_ws is not None
            and moe_all_reduce is not None
            and num_tokens <= self._latent_ar_ws["max_tokens"]
            and self.shared_expert_stream is not None
            and do_multi_stream()
        ):
            # The fork / join of fork_join, with the join left open until it is
            # known whether the latent all-reduce runs on its own workspace.
            self.moe_main_event.record()
            routed_out = _routed_output()
            with torch.cuda.stream(self.shared_expert_stream):
                self.moe_main_event.wait()
                shared_out = _shared_output()
                self.moe_shared_event.record()
            if routed_out.dtype == torch.bfloat16:
                return self._own_workspace_latent_tail(
                    routed_out, shared_out, residual, defer_residual
                )
            self.moe_shared_event.wait()
        else:
            # Shared experts depend only on the block input, so overlap their GEMMs
            # with the routed dispatch/expert/combine chain. Multi-stream engages
            # only under CUDA graphs; otherwise both branches run in order on the
            # default stream. The shared expert includes its output all-reduce on
            # the auxiliary stream. The final join includes the shared all-reduce, so it
            # precedes the routed all-reduce: concurrent collectives on different
            # streams can corrupt SYMM_MEM all-reduce state.
            routed_out, shared_out = fork_join(
                _routed_output,
                _shared_output,
                self.moe_main_event,
                self.moe_shared_event,
                self.shared_expert_stream,
            )
        if self._reduce_routed_output:
            # RMSNorm folded into the routed all-reduce (RMS_NORM fusion): the
            # MNNVL kernel normalizes straight out of the lamport buffer, so
            # the reduced row never round-trips through HBM for a norm kernel.
            # Strategies without the fusion fall back inside AllReduce.
            routed_latent = moe_all_reduce(
                routed_out,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RMS_NORM,
                    norm_weight=self.routed_expert_norm.weight,
                    eps=float(self.routed_expert_norm.variance_epsilon),
                ),
            )
            routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        if residual is not None:
            if defer_residual:
                return _DeferredMoEResidual(residual, routed_out, shared_out)
            return k3_moe_residual_add_op(routed_out, shared_out, residual)
        return routed_out + shared_out


if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

# CUDA stream priority of the MoE input stream (latent_down chain, routed-output memset).
# Negative values rank above the main and shared-expert streams, so the latent chain that
# gates the routed GEMM1 gets free SMs first.
_K3_LATENT_STREAM_PRIORITY = 0

# CUDA stream priority of the stream shared by the attention-side aux work and the MoE shared
# expert. Negative values let the shared-expert chain take SMs ahead of the persistent routed
# GEMMs on the main stream, so its all-reduce does not land after them.
_K3_SHARED_STREAM_PRIORITY = 0

# The DFlash/DSpark hidden-state tap captures the pre-norm attn_res mixture (the value the next
# consumer reads), the convention of SGLang and of the RadixArk/Kimi-K3-DSpark drafter
# (``kimi_k3.py _dspark_capture_stream`` -> ``attn_residual.py aggregate_stream``). A drafter
# distilled against the raw running prefix sum (vLLM's ``VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=0``)
# would score lower here with nothing raised (K3 + RadixArk: AR 71.4% -> 66.9% the other way).


# The constants the helix8 targets declare (their ``constants.py``), which this code is built for.
HELIX8_CONSTANTS = K3Sm107Constants(
    spec_head_fusion=True,
    fuse_moe_residual=True,
    fp8w_lowm_gemm=True,
    ld_fused=True,
    shared_sm=16,
    fc12_die_major=True,
    megamoe_experts=None,
    router_exact_kda=True,
    router_bf16_gate=True,
    fp8_weight_read_moe_mlp=True,
    latent_ar_own_ws=True,
    attn_res_pdl=True,
    attn_res_pdl_max_m=128,
    attn_res_mxfp8=True,
    attn_res_early_trigger=True,
    kda_prepad_prefill=False,
    kda_fused_conv_prefill=False,
    kda_l2_prefetch=True,
    mla_q_norm_pdl=True,
    fp4_mla_page_plan=True,
    fp4_mla_reduce_pdl=True,
    spec_tail_fusion=True,
    dspark_tail_v2=True,
    dspark_context_kv_only=False,
    torch_mem_safety_margin=4 * 1024**3,
)


def _refuse_attention_dp(mapping) -> None:
    """The helix8 targets run without attention DP; the code has no attention-DP paths."""
    if mapping.enable_attention_dp:
        raise NotImplementedError(
            "The Kimi K3 helix8 targets run without attention DP (enable_attention_dp=False)."
        )


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


class KimiK3DefaultDecoderLayer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        dtype = torch.bfloat16
        self.is_moe = (
            cfg.num_experts is not None
            and layer_idx >= cfg.first_k_dense_replace
            and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0
        )

        self.is_kda = _is_kda_layer(cfg, layer_idx)
        is_mla = _is_mla_layer(cfg, layer_idx)
        if self.is_kda == is_mla:
            raise ValueError(f"Kimi K3 layer {layer_idx} must be exactly one of KDA/MLA")

        if self.is_kda:
            projection_names = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
            attention_config = copy.copy(model_config)
            attention_config._frozen = False
            attention_config.quant_config_dict = {
                name: resolve_attention_quant_config(model_config, layer_idx, name)
                for name in projection_names
            }
            attention_config._frozen = model_config._frozen
            self.linear_attn = K3KDALinearAttention(
                cfg,
                layer_idx,
                mapping=model_config.mapping.mapping_for_full_rows(),
                allreduce_strategy=model_config.allreduce_strategy,
                aux_stream=aux_stream_dict[AuxStreamType.Attention],
                model_config=attention_config,
            )
        else:
            self.self_attn = K3MLARuntime(
                cfg,
                layer_idx,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
                # CP original stashed by _setup_helix_mappings; None outside helix.
                mapping_with_cp=getattr(model_config, "_helix_mapping_with_cp", None),
            )

        if self.is_moe:
            self.block_sparse_moe = K3MoERuntime(model_config, cfg, layer_idx, aux_stream_dict)
        else:
            # The dense MLP (layer 0): BF16 gate_up GEMM, SiTU-and-mul, down GEMM and, on a TP
            # group, the down projection's all-reduce. It is TP over the largest group of
            # consecutive ranks that divides the intermediate size and stays within one node.
            mlp_mapping = model_config.mapping.mapping_for_full_rows()
            self.mlp_tp_size = math.gcd(cfg.intermediate_size, mlp_mapping.tp_size)
            if self.mlp_tp_size > mlp_mapping.gpus_per_node:
                self.mlp_tp_size = math.gcd(self.mlp_tp_size, mlp_mapping.gpus_per_node)
            self.mlp = GatedMlp(
                cfg.hidden_size,
                cfg.intermediate_size,
                activation="situ",
                beta=getattr(cfg, "activation_situ_beta", None) or 1.0,
                linear_beta=getattr(cfg, "activation_situ_linear_beta", None),
                dtype=dtype,
                mapping=subgroup_mapping(mlp_mapping, self.mlp_tp_size),
                reduce_output=self.mlp_tp_size > 1,
                allreduce_strategy=model_config.allreduce_strategy,
            )

        # FlashInfer RMSNorm on the K3 weights: the input norm after the pre-attention mix and the
        # post-attention norm after A_mid (fused into the attention-residual kernels where they
        # serve, ``trtllm::flashinfer_rmsnorm`` otherwise).
        self.input_layernorm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = RMSNormWeight(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype
        )

        # Attention residual scheme (always on for K3): the score norms and the [1, H] score
        # projections the attention-residual kernels read (``.weight`` / ``.eps``).
        self.attn_res_block_size = cfg.attn_res_block_size
        assert self.attn_res_block_size is not None, (
            "Kimi K3 runtime expects attn_res_block_size to be set"
        )
        self.self_attention_res_norm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.mlp_res_norm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.self_attention_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.mlp_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)

    def _attn_takes_mxfp8(self, num_rows: int, attn_metadata: AttentionMetadata) -> bool:
        """Whether the pre-attention mix should also emit its MXFP8 form for the attention mixer
        (generation-only batches): the KDA fused QKVG projection
        (``qkvg_takes_mxfp8``) or the MLA fused q_a / kv_a projection (``q_a_takes_mxfp8``).

        The mix that folds in the previous MoE residual (k3_attn_res_moe_add_rmsnorm_mx) then also
        stores the MXFP8 form of its output, and that projection runs the low-M GEMM on it
        (k3_fp8w_lowm_mxfp8) instead of quantizing the BF16 rows itself: bitwise the same
        projection."""
        if attn_metadata.num_contexts != 0:
            return False
        if self.is_kda:
            return self.linear_attn.qkvg_takes_mxfp8(num_rows)
        q_a_takes_mxfp8 = getattr(self.self_attn.mixer, "q_a_takes_mxfp8", None)
        return q_a_takes_mxfp8 is not None and q_a_takes_mxfp8(num_rows)

    def _attn_mx_atom(self, num_rows: int) -> bool:
        """Whether that MXFP8 form goes to the mid-M GEMM, i.e. the mix stores its scales in that
        GEMM's layout (k3_attn_res ``mx_atom``)."""
        if self.is_kda:
            mx_atom = getattr(self.linear_attn, "qkvg_mx_atom", None)
        else:
            mx_atom = getattr(self.self_attn.mixer, "q_a_mx_atom", None)
        return mx_atom is not None and bool(mx_atom(num_rows))

    @staticmethod
    def _capture_hidden_states(capture: Tuple[Any, int], value: torch.Tensor) -> None:
        """The DSpark tap of ``capture[1]``."""
        capture[0].maybe_capture_hidden_states(capture[1], value, None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
        capture: Optional[Tuple[Any, int]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Port of HF ``KimiDecoderLayer._forward_attn_residual`` (per token).

        ``block_residual`` is a preallocated snapshot bank in kernel-native
        ``[K_max, M, H]`` layout. Returns the running prefix sum and the
        number of valid bank rows. The prefix sum (input and output) can be a
        ``_DeferredMoEResidual`` or ``_DeferredRpTail``.

        ``capture`` is ``(spec_metadata, layer_id)`` and taps the DSpark aux
        stream for the layer BEFORE this one: the aggregated stream for layer j
        is by definition what its next consumer sees, so the mixture computed
        below already is it. Reading it here beats recomputing it, and is only
        possible because K3 asserts pp_size == 1 -- layer j+1 is always local.
        PP support would need a recompute at the rank boundary.
        """
        all_rank_num_tokens = getattr(attn_metadata, "all_rank_num_tokens", None)
        valid_block_residual = block_residual[:num_snapshots]
        normed = None
        attn_input_mxfp8 = None
        if isinstance(hidden_states, _DeferredRpTail):
            # The previous layer's row-parallel MoE tail: this mix finishes its all-reduce where
            # it can; the DSpark tap path materializes it first.
            if capture is None and num_snapshots > 0:
                rows = hidden_states.residual.shape[0]
                mxfp8 = self._attn_takes_mxfp8(rows, attn_metadata)
                fused = _k3_attn_res_pdl_rp_tail(
                    hidden_states,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                    mxfp8=mxfp8,
                )
                if fused is not None:
                    hidden_states, normed, *attn_input_mxfp8 = fused
            if normed is None:
                hidden_states = hidden_states.materialize()
        elif isinstance(hidden_states, _DeferredMoEResidual):
            # The previous layer's MoE residual update, folded into this mix when possible.
            if capture is None and num_snapshots > 0:
                rows = hidden_states.residual.shape[0]
                mxfp8 = self._attn_takes_mxfp8(rows, attn_metadata)
                fused = _k3_attn_res_pdl_moe_add(
                    hidden_states,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                    mxfp8=mxfp8,
                    # Release the staging KDA QKVG GEMM at kernel entry.
                    entry_trigger=mxfp8
                    and self.is_kda
                    and self.linear_attn.qkvg_stages_weights(rows),
                    mx_atom=mxfp8 and self._attn_mx_atom(rows),
                )
                if fused is not None:
                    hidden_states, normed, *attn_input_mxfp8 = fused
            if normed is None:
                hidden_states = hidden_states.materialize()
        prefix_sum = hidden_states

        if capture is not None:
            # The mixture tap needs the PRE-norm value (SGLang's
            # aggregate_stream is a "softmax mixture, no output norm"), which
            # the fused attn-res + RMSNorm kernel does not expose. Keep the two
            # steps split on captured layers only and fuse everywhere else.
            mixed = (
                _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
                if num_snapshots > 0
                else hidden_states
            )
            self._capture_hidden_states(capture, mixed)
            hidden_states = rms_norm(mixed, self.input_layernorm.weight, self.input_layernorm.eps)
        elif normed is not None:
            hidden_states = normed
        elif num_snapshots > 0:
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            hidden_states = rms_norm(
                hidden_states, self.input_layernorm.weight, self.input_layernorm.eps
            )

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None

        # When A_mid below adds the attention output to the prefix sum, the attention block may
        # hand back its pushed all-reduce unfinished (a LamportPending) for A_mid to finish and
        # read from the workspace. A_mid must do so before the next all-reduce on the workspace
        # re-arms that buffer: the shared expert, whose down-projection all-reduce is the next
        # one, forks after A_mid, and its pushed finish kernel can only become resident once its
        # producer has passed its own griddepcontrol.wait (every grid before, A_mid included, has
        # then completed).
        attn = self.linear_attn if self.is_kda else self.self_attn
        attn.ar_finish_deferred = prefix_sum is not None and _k3_attn_res_lamport_applies(
            prefix_sum,
            valid_block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            self.post_attention_layernorm,
        )
        if self.is_kda:
            if attn_input_mxfp8:
                self.linear_attn.next_input_mxfp8 = tuple(attn_input_mxfp8)
            hidden_states = self.linear_attn(hidden_states, attn_metadata)
        else:
            if attn_input_mxfp8:
                self.self_attn.mixer.next_input_mxfp8 = tuple(attn_input_mxfp8)
            hidden_states = self.self_attn(hidden_states, attn_metadata)
        attn.ar_finish_deferred = False

        if prefix_sum is None:
            prefix_sum = hidden_states
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        else:
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                hidden_states,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        if self.is_moe:
            # The MoE adds its output to the prefix sum itself (or defers that add).
            prefix_sum = self.block_sparse_moe(
                hidden_states,
                all_rank_num_tokens,
                residual=prefix_sum,
                defer_residual=_defer_moe_residual(prefix_sum.shape[0]),
                # The row-parallel tail serves generation-only batches.
                rp_tail_ok=getattr(attn_metadata, "num_contexts", 1) == 0,
            )
            return prefix_sum, num_snapshots
        hidden_states = self.mlp(hidden_states)

        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, num_snapshots

    def skip_forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No-op stand-in for ``forward``, matching ``DecoderLayer.skip_forward``.

        ``modeling_utils.skip_forward()`` only drops a module's weights when it
        finds this attribute, so without it the layer-wise benchmarks would
        allocate all 93 layers instead of the profiled slice.
        """
        return hidden_states, block_residual


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


class KimiK3DefaultModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = _get_text_config(model_config.pretrained_config)
        self._text_cfg = cfg
        dtype = torch.bfloat16

        # Attention and MoE phases are sequential, so their branch-overlap
        # roles share one stream; MoE-internal overlap roles remain separate.
        aux_stream_list = [torch.cuda.Stream(priority=_K3_SHARED_STREAM_PRIORITY)]
        aux_stream_list += [torch.cuda.Stream() for _ in range(2)]
        aux_stream_list.append(torch.cuda.Stream(priority=_K3_LATENT_STREAM_PRIORITY))
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }

        # The whole vocabulary on every rank: a row gather (``gather_embedding`` of 1-D ids on a
        # contiguous table, ``F.embedding`` otherwise).
        self.embed_tokens = EmbeddingTable(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                KimiK3DefaultDecoderLayer(model_config, cfg, layer_idx, self.aux_stream_dict)
                for layer_idx in range(cfg.num_hidden_layers)
            ]
        )
        # The final norm, fused into the output attention-residual mix where that kernel serves.
        self.norm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.output_attn_res_norm = RMSNormWeight(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.output_attn_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.num_attn_res_snapshots = (
            cfg.num_hidden_layers + cfg.attn_res_block_size - 1
        ) // cfg.attn_res_block_size

        # Which convention the drafter tap is on is not recoverable from the
        # served output -- a mismatch only lowers acceptance -- so state it once
        # at construction rather than leaving it to be inferred from an AL.
        logger.info_once(
            "Kimi K3 aux hidden capture: mode=attn_res_stream",
            key="kimi_k3_aux_capture_mode",
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            weight = self.embed_tokens.weight
            if input_ids.dim() == 1 and weight.is_contiguous():
                # The F.embedding rows, copied with 16-byte accesses.
                inputs_embeds = gather_embedding(input_ids, weight)
            else:
                inputs_embeds = F.embedding(input_ids, weight)
        hidden_states = inputs_embeds

        # The FP4 MLA decode plan of the previous step is stale.
        fp4_mla_decode.begin_step(attn_metadata)

        block_residual = hidden_states.new_empty(
            self.num_attn_res_snapshots,
            hidden_states.shape[0],
            hidden_states.shape[1],
        )
        num_snapshots = 0
        capture_set = (
            getattr(spec_metadata, "_capture_layer_set", None)
            if spec_metadata is not None
            else None
        )
        for i, layer in enumerate(self.layers):
            # DFlash/DSpark hidden-state capture. The drafter is distilled on
            # the aggregated stream value -- the pre-norm softmax mixture its
            # next consumer sees -- not on the raw prefix sum a layer returns,
            # which is SGLang's fallback for models without the
            # attention-residual scheme. Capturing the prefix sum costs 4.5pt
            # of draft acceptance on K3 + RadixArk DSpark (AR 66.9% -> 71.4%).
            # The tap fires inside layer i+1, which computes that tensor
            # anyway; see its forward docstring. Ground truth: SGLang
            # kimi_k3.py:2697 _dspark_capture_stream, attn_residual.py:313
            # aggregate_stream_torch.
            capture = None
            if (
                spec_metadata is not None
                and i > 0
                and (capture_set is None or self.layers[i - 1].layer_idx in capture_set)
            ):
                capture = (spec_metadata, self.layers[i - 1].layer_idx)
            hidden_states, num_snapshots = layer(
                hidden_states,
                block_residual,
                num_snapshots,
                attn_metadata,
                capture=capture,
            )
        hidden_states = _materialize_prefix_sum(hidden_states)

        # The last layer has no successor, so this one recompute is
        # unavoidable -- output-side score weights, matching SGLang's
        # layer_idx + 1 >= end_layer branch. Unreachable for K3's capture set
        # against 93 layers; kept so a set that does include the final layer
        # gets the right tensor rather than the raw prefix sum.
        if spec_metadata is not None and len(self.layers) > 0:
            last = self.layers[-1]
            if capture_set is None or last.layer_idx in capture_set:
                tail = (
                    _apply_attn_res(
                        hidden_states,
                        block_residual[:num_snapshots],
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                    )
                    if num_snapshots > 0
                    else hidden_states
                )
                spec_metadata.maybe_capture_hidden_states(last.layer_idx, tail, None)

        output = _apply_attn_res_and_rmsnorm(
            hidden_states,
            block_residual[:num_snapshots],
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            self.norm,
        )
        return output


# The weight mixins: weights.py imports the blocks above from this module.
from .weights import KimiK3DefaultTextWeights, KimiK3DefaultWeights  # noqa: E402


class KimiK3DefaultForCausalLM(
    KimiK3DefaultTextWeights, KimiK3SpecForCausalLM[KimiK3DefaultModelingV2, Any]
):
    """The Kimi K3 text model (KDA + MLA + MoE decoder layers) of ``KimiK3Sm107ForConditionalGeneration``.

    Registered under no architecture: the target class builds it as its ``llm``. Weight loading
    is the ``KimiK3DefaultTextWeights`` mixin's (``weights.py``).

    Speculative decoding: one-engine DSpark with a standalone MLA drafter, built by the family's
    ``KimiK3SpecForCausalLM`` (``dspark/``): the fused spec-tail v2 deployment path, the full
    generation-side drafter (not the context-KV-only one of a context server) and the decode
    step's head in one kernel (``RuntimeHooks.decode_prologue``)."""

    mamba_metadata_cls = K3MambaMetadata
    # The fused spec tail, v2.
    dspark_path = K3DSparkPath(spec_tail_fusion=True, tail_v2=True)
    # A generation server: its drafter is the full one.
    dspark_context_kv_only = False
    # The speculative decode step's head in one kernel.
    spec_decode_prologue = staticmethod(spec_decode_head)

    def __init__(self, model_config: ModelConfig):
        cfg = _get_text_config(model_config.pretrained_config)
        assert model_config.mapping.pp_size == 1, "Kimi K3 does not support pipeline parallelism"
        _refuse_attention_dp(model_config.mapping)
        spec_config = getattr(model_config, "spec_config", None)

        # Helix: swap in the repurposed mapping; restored after super().__init__.
        self._setup_helix_mappings(model_config, cfg, spec_config)
        # Speculative decoding is one-engine DSpark with a standalone drafter checkpoint;
        # KimiK3SpecForCausalLM rejects every other speculative config. The target side is the
        # hidden-state capture in KimiK3DefaultModelingV2.forward.
        super().__init__(
            KimiK3DefaultModelingV2(model_config),
            model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        # The drafter and the worker are built under the repurposed helix mapping (still active
        # here); the drafter gets the CP original as ``_helix_mapping_with_cp``.

        # Restore the CP original: executor-side helix bookkeeping keys off
        # has_cp_helix() at runtime.
        if self.mapping_with_cp is not None:
            model_config._frozen = False
            model_config.mapping = self.mapping_with_cp
            model_config._frozen = True

    def _setup_helix_mappings(
        self,
        model_config: ModelConfig,
        cfg: "PretrainedConfig",
        spec_config: Optional["DecodingBaseConfig"],
    ) -> None:
        """Validate helix preconditions and stage the dual-mapping swap.

        DeepseekV3 pattern: the MLA layers keep the CP original; everything
        else is built against the repurposed mapping (CP ranks become TP
        ranks). Sets ``mapping_with_cp`` (restored after construction) and
        ``_repurposed_tp_mapping`` (load_weights shard selection); both stay
        None outside helix.
        """
        self.mapping_with_cp = None
        self._repurposed_tp_mapping = None
        if not model_config.mapping.has_cp_helix():
            return
        # The K3 helix targets run head-TP attention, the residual stream and the MoE on every
        # row of the CP group; there is no token-sliced (attention-DP) variant.
        if model_config.mapping.enable_attention_dp:
            raise NotImplementedError(
                "Kimi K3 helix runs without attention DP: KDA and MLA are head-TP over the CP "
                "group, so there is no request-level DP to add."
            )
        if spec_config is not None:
            # Helix supports only the standalone DSpark drafter (verified on
            # the V2 superblock ledger); reject everything else loudly rather
            # than let an unsupported spec mode run silently wrong.
            if not spec_config.spec_dec_mode.is_dspark():
                raise ValueError(
                    "Kimi K3 helix supports speculative decoding only with "
                    f"DSpark (standalone drafter); got "
                    f"{spec_config.decoding_type!r}."
                )
            # The SpeculationGate acceptance-rate trip permanently disables
            # speculation mid-flight while enable_spec_decode stays True;
            # in-flight helix requests then fall into the plain generation
            # loop whose position math (total_input_len_cp +
            # py_decoding_iter - 1) is stale once any draft token was
            # accepted -> silently wrong RoPE positions and KV slots. Reject
            # the trip wires until that loop is helix-group aware.
            if (
                spec_config.acceptance_rate_window_size is not None
                or spec_config.acceptance_rate_threshold is not None
            ):
                raise ValueError(
                    "Kimi K3 helix does not support the speculation "
                    "acceptance-rate gate (acceptance_rate_window_size / "
                    "acceptance_rate_threshold): dynamically disabling "
                    "speculation mid-flight leaves helix requests on a "
                    "single-token position formula."
                )
            # max_concurrency is the same trip wire by another name: the
            # drafter re-evaluates should_use_spec_decode on every scheduling
            # iteration and flips enable_spec_decode off as soon as the active
            # batch exceeds the cap. In-flight helix requests then take the
            # plain generation loop, whose position formula counts ITERATIONS
            # (total_input_len_cp + py_decoding_iter - 1) rather than
            # committed tokens, so it is stale by however many draft tokens
            # were accepted -- a wrong RoPE position, and across a ledger page
            # boundary a KV write to the wrong CP rank. Mirror the drafter's
            # own "unset" test (Drafter.should_use_spec_decode returns True
            # when max_concurrency is None) so an unset value is not rejected.
            if spec_config.max_concurrency is not None:
                raise ValueError(
                    "Kimi K3 helix does not support the speculation "
                    "concurrency cutoff (max_concurrency): disabling "
                    "speculation above the cap leaves in-flight helix "
                    "requests on a position formula that assumes one "
                    "committed token per iteration, which accepted draft "
                    "tokens break."
                )
            # draft_len_schedule is the user-facing alternative to
            # max_concurrency (llm_args rejects setting both) and reaches the
            # same end by a route that does not go through
            # should_use_spec_decode at all: py_executor turns speculation off
            # directly once the schedule yields draft_len 0 for the active
            # batch size. Guarding only max_concurrency would leave this door
            # open. Skip the schedule that llm_args synthesized from
            # max_concurrency, so a config that set only that field raises the
            # message above naming the field the user actually wrote.
            if (
                spec_config.draft_len_schedule is not None
                and not spec_config._translated_from_max_concurrency
            ):
                raise ValueError(
                    "Kimi K3 helix does not support the dynamic draft-length "
                    "schedule (draft_len_schedule): a batch size past the "
                    "last entry drops the draft length to 0 and turns "
                    "speculation off mid-run, leaving in-flight helix "
                    "requests on a position formula that assumes one "
                    "committed token per iteration."
                )
        cp = model_config.mapping.cp_size
        repurposed_tp = model_config.mapping.tp_size * cp
        if cfg.num_attention_heads % repurposed_tp != 0:
            raise ValueError(
                f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) "
                f"to divide the MLA head count ({cfg.num_attention_heads})."
            )
        kda_heads = cfg.linear_attn_config["num_heads"]
        if kda_heads % repurposed_tp != 0:
            raise ValueError(
                f"Kimi K3 helix requires tp_size*cp_size ({repurposed_tp}) to "
                f"divide the KDA head count ({kda_heads})."
            )
        # MoE splits apply to the repurposed tp*cp group (helix
        # moe_world_size = tp*cp); default EP-only. The Mapping constructor
        # skips its product check when both sizes are 1, so validate here.
        moe_ep = repurposed_tp
        if model_config.mapping.moe_tp_ep_user_specified:
            moe_tp = model_config.mapping.moe_tp_size
            moe_ep = model_config.mapping.moe_ep_size
            if moe_tp * moe_ep != repurposed_tp:
                raise ValueError(
                    f"Kimi K3 helix: moe_tensor_parallel_size ({moe_tp}) x "
                    f"moe_expert_parallel_size ({moe_ep}) must equal "
                    f"tp_size*cp_size ({repurposed_tp}): MoE runs on the "
                    "repurposed tp*cp group."
                )
        if cfg.num_experts and cfg.num_experts % moe_ep != 0:
            raise ValueError(
                f"Kimi K3 helix requires the MoE EP size ({moe_ep}) to "
                f"divide the routed expert count ({cfg.num_experts}): each "
                "EP rank of the repurposed tp*cp group holds whole experts."
            )
        self.mapping_with_cp = copy.deepcopy(model_config.mapping)
        repurposed = model_config.mapping.repurpose_helix_cp_to_tp()
        # repurpose passes resolved moe sizes, which the Mapping constructor
        # mistakes for user-specified values; restore the flag.
        repurposed.moe_tp_ep_user_specified = self.mapping_with_cp.moe_tp_ep_user_specified
        # load_weights shard selection must use this tp_rank; the restored
        # CP original's tp_rank is 0 on every rank.
        self._repurposed_tp_mapping = repurposed
        model_config._frozen = False
        model_config.mapping = repurposed
        # Side-channel for the MLA layers (avoids threading a kwarg through
        # every intermediate signature).
        model_config._helix_mapping_with_cp = self.mapping_with_cp
        model_config._frozen = True


class KimiK3Sm107ForConditionalGeneration(KimiK3DefaultWeights, PreTrainedModel):
    """Kimi K3 NVFP4 on SM107, sharded as the engine's mapping says.

    The text model ``KimiK3DefaultForCausalLM`` as ``llm`` behind the vision-language model's
    interface: text only, no vision tower (``mm_encoder`` is ``None``, as under multimodal
    disaggregation), and a step carrying multimodal input is refused. The speculative-decoding
    and weight-loading attributes the engine reads off the outer model (``draft_model``, ``lm_head``,
    ``runtime_hooks``, ...) are the text model's. Weight loading is the
    ``KimiK3DefaultWeights`` mixin's (``weights.py``).

    Abstract: a target subclasses it and sets ``K3_CONSTANTS``, which must be
    ``HELIX8_CONSTANTS`` (the code is built for them).
    """

    # The target's constants; set by every target class, checked once in ``__init__``.
    K3_CONSTANTS: ClassVar[K3Sm107Constants]

    _LANG_PREFIX = "language_model."
    mamba_metadata_cls = KimiK3DefaultForCausalLM.mamba_metadata_cls

    @classmethod
    def get_preferred_kv_cache_manager_version(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["V2"]:
        """Prefer KV cache manager V2 (as Kimi K2.5)."""
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["PYTHON"]:
        """Kimi K3 disaggregated serving requires the Python transceiver.

        Only the Python NIXL transceiver (KvCacheTransceiverV2) can move
        the KDA recurrent state; the C++ transceiver has no KDA support.
        Adopted when the user leaves
        ``cache_transceiver_config.transceiver_runtime`` at 'auto' and the
        effective backend is NIXL. An explicit non-Python runtime is
        rejected by ``get_kv_cache_manager_cls`` rather than silently
        routed to a path that cannot transfer the recurrent state.
        """
        return "PYTHON"

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        constants = getattr(type(self), "K3_CONSTANTS", None)
        if not isinstance(constants, K3Sm107Constants):
            raise TypeError(
                f"{type(self).__name__} has no K3_CONSTANTS: build a Kimi K3 SM107 target class "
                "(a subclass that sets K3_CONSTANTS to its K3Sm107Constants), not "
                "KimiK3Sm107ForConditionalGeneration itself"
            )
        if constants != HELIX8_CONSTANTS:
            raise ValueError(
                f"{type(self).__name__}.K3_CONSTANTS differ from the helix8 constants this code is "
                "built for (HELIX8_CONSTANTS in this module)"
            )
        config = model_config.pretrained_config
        # PreTrainedModel.__init__ resolves config._attn_implementation and
        # rejects the default "sdpa" unless the class declares support. This
        # wrapper never runs HF attention itself (TRT-LLM attention is wired
        # inside the sub-modules), so declare support to keep HF init happy.
        self._supports_sdpa = True
        PreTrainedModel.__init__(self, config)

        # Re-entry guard: the tail of this __init__ repoints
        # model_config.pretrained_config at the text_config and remaps the quant
        # exclude list, so running the body twice on the same instance would
        # fail the text_config assert below. If the text backbone already
        # exists this is a re-init on a built instance: no-op.
        if hasattr(self, "llm"):
            return

        self.model_config = model_config
        log_constants(type(self).__module__, type(self).K3_CONSTANTS)
        self._vlm_pretrained_config = config

        # Text only: no vision tower.
        self.mm_encoder = None

        text_model_config = copy.copy(model_config)
        assert hasattr(config, "text_config"), "Kimi K3 config must have text_config"
        text_model_config._frozen = False
        text_model_config.pretrained_config = config.text_config

        # Remap quant exclude_modules: language_model.X -> model.X
        if text_model_config.quant_config.exclude_modules:
            text_model_config.quant_config = copy.copy(text_model_config.quant_config)
            p = self._LANG_PREFIX
            mapped = []
            for m in text_model_config.quant_config.exclude_modules:
                if m.startswith(p):
                    rest = m[len(p) :]
                    if rest.startswith("layers."):
                        rest = "model." + rest
                    mapped.append(rest)
                else:
                    mapped.append(m)
            text_model_config.quant_config.exclude_modules = mapped

        if not text_model_config.skip_create_weights_in_init:
            text_model_config.skip_create_weights_in_init = True
        text_model_config._frozen = True

        self.llm = KimiK3DefaultForCausalLM(text_model_config)

        # Point model_config at the text_config so the executor reads generation
        # params (eos_token_id, ...) from the text backbone, matching K2.5.
        self.config = self.llm.config
        model_config._frozen = False
        model_config.pretrained_config = self.llm.config
        model_config._frozen = True

    # Forward spec-dec / weight-loading attrs to self.llm: this wrapper is not
    # itself a spec-dec model but is the outer model that those paths see.
    @property
    def draft_config(self):
        return self.llm.draft_config

    @property
    def draft_model(self):
        return self.llm.draft_model

    @property
    def model(self):
        return self.llm.model

    @property
    def lm_head(self):
        return self.llm.lm_head

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    @property
    def runtime_hooks(self) -> RuntimeHooks:
        """The text model's step-loop hooks (the engine reads them off the outer model) and the
        4 GiB the engine leaves outside the PyTorch memory cap. A target class adds its own
        declarations to these."""
        return replace(
            self.llm.runtime_hooks or DEFAULT_RUNTIME_HOOKS,
            torch_mem_safety_margin=HELIX8_CONSTANTS.torch_mem_safety_margin,
        )

    def set_guided_decoder(self, *args, **kwargs):
        return self.llm.set_guided_decoder(*args, **kwargs)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        multimodal_params: Optional[List[MultimodalParams]] = None,
        **kwargs,
    ) -> torch.Tensor:
        check_text_only(multimodal_params)
        # The text model embeds ``input_ids`` itself; the wrapper hands it no embeddings.
        inputs_embeds = None
        return self.llm.forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            **kwargs,
        )


__all__ = ["HELIX8_CONSTANTS", "KimiK3DefaultForCausalLM", "KimiK3Sm107ForConditionalGeneration"]


class ModelingV2KimiK3Nvfp4Sm107Helix8Moeep8(KimiK3Sm107ForConditionalGeneration):
    """Kimi K3 NVFP4 on SM107, helix8_moeep8."""

    K3_CONSTANTS: ClassVar[K3Sm107Constants] = CONSTANTS

    @property
    def runtime_hooks(self) -> RuntimeHooks:
        """The family's hooks; the recurrent-state indices and the dummy-request mask are uploaded
        with one kernel when they change, as the reference's helix8 GEN servers run."""
        hooks = super().runtime_hooks
        return replace(
            hooks,
            hybrid_cache=replace(hooks.hybrid_cache, state_index_upload_kernel=True),
            kv_cache_manager_cls=KimiK3Sm107CacheManager,
        )


__all__ = ["ModelingV2KimiK3Nvfp4Sm107Helix8Moeep8"]
