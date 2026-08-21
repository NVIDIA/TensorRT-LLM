# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KimiLinearForCausalLM — Kimi K3 text model, PyTorch backend.

Runtime integration of the Kimi K3 hybrid architecture for the standard
TRT-LLM PyTorch-backend flow (``LLM(model=<ckpt>) -> generate``):

* 93 decoder layers: 69 KDA (Kimi Delta Attention, linear attention) layers
  and 24 MLA (absorbed-MQA, NoPE) layers, per the 1-indexed
  ``linear_attn_config.kda_layers`` / ``full_attn_layers`` schedule.
* Layer 0 uses a dense SiTU MLP; layers 1..92 use the 896-expert latent MoE
  (top-16, sigmoid + e_score_correction_bias routing, MXFP4 routed experts,
  2 shared experts, latent down/norm/up projections).
* The attention-residual ("attn_res") scheme from the HF reference
  ``modeling_kimi.py`` is applied per token: snapshot mixing before
  ``input_layernorm`` / ``post_attention_layernorm`` and at the model output,
  with a new snapshot appended whenever ``layer_idx % attn_res_block_size == 0``.

Caching
-------
KDA states live on the mamba side of a ``MixedMambaHybridCacheManager``
(wired in ``pyexecutor/_util.py``): per layer, a short-conv slot of
``[3 * num_heads * head_dim, W]`` bf16 (the full FLA ``ShortConvolution``
cache window, sections ``[q | k | v]``) and a delta-rule recurrent slot of
``[num_heads, head_dim, head_dim]`` fp32 (``[H, V, K]``, the
``state_v_first`` FLA layout). MLA layers use the paged-KV side with
``num_kv_heads=1`` and ``head_dim = kv_lora_rank + qk_rope_head_dim`` (576),
SELFKONLY, exactly like DeepSeek MLA.

MLA prefill routing
-------------------
The in-tree ``KimiK3MLAAttention`` routes prefill through the normal
unabsorbed MLA context FMHA. It consumes the executor's original mixed-batch
metadata, letting the shared MLA implementation dispatch context and cached
generation work in one forward call.

Parallelism
-----------
The routed-expert bank supports a configurable MoE TP x EP split
(``moe_tp_size * moe_ep_size == mapping.tp_size``); the default is EP-only
(``moe_ep_size == mapping.tp_size``), the historical K3 layout. Under EP each
MoE layer holds a contiguous ``num_experts / moe_ep_size`` slice of the MXFP4
expert bank (whole experts); under MoE TP each rank holds ALL experts, with
w1/w3 column-sharded and w2 row-sharded along the intermediate dim
(``intermediate / moe_tp_size`` per rank; group-32 MXFP4 packed bytes and
scales sliced consistently by the stock TRTLLM-Gen quant-method loaders).
The split is EP-only unless the user sets ``moe_tensor_parallel_size`` /
``moe_expert_parallel_size`` explicitly (or the ``TLLM_K3_MOE_TP_SIZE`` /
``TLLM_K3_MOE_EP_SIZE`` env overrides). Routing is computed replicated; the
routed partial sums — EP partials of whole experts, or TP partials over the
intermediate shards — are all-reduced in the latent space (before
``routed_expert_norm`` / ``routed_expert_up_proj``, which are
nonlinear/linear layers applied to the full sum). When attention DP is off,
the shared experts use standard MLP TP over the model TP group: gate/up are
column-sharded and down is row-sharded. Direct MoE-TP combines the shared
hidden-width partial and routed latent partial into one all-reduce after the
two streams join, then splits them before the routed norm/up projection.
Communication-backed routed paths keep the shared ``GatedMLP`` reduction,
because their routed result is already combined. Under attention DP the
shared experts stay replicated. ``lm_head`` uses the stock ``LMHead``
(vocab-sharded + gather), so logits are identical on all ranks.

Speculative decoding: SA (suffix automaton, one-engine, draft-weight-free);
the KDA/MLA runtimes implement multi-token verification with deferred
state promotion.

Chunked prefill is supported: continuation chunks feed the previous KDA
conv/recurrent state back into the FLA kernels (``use_initial_states``)
and the MLA prefill path natively attends over the cached latent prefix
(``kv_len = cached + q_len``). KV-cache block reuse is supported as an
opt-in via ``kv_cache_config.enable_block_reuse=true``, which routes to
the unified-pool ``CppMambaHybridCacheManager`` (per-block KDA state
snapshots every ``mamba_state_cache_interval`` tokens, FORCE_CHUNK
context chunking).

Not supported: pipeline parallelism, draft-head spec-dec modes
(MTP/Eagle — no draft-head checkpoint exists). SA speculative decoding
is validated only without block reuse (Mixed cache manager).
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import weakref
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple

import torch
from safetensors import safe_open
from torch import nn

from ..._utils import is_sm_100f
from ...logger import logger
from ...mapping import Mapping
from ...models.modeling_utils import QuantAlgo, QuantConfig
from ..attention_backend import AttentionMetadata
from ..distributed import AllReduce, AllReduceParams, AllReduceStrategy
from ..model_config import ModelConfig
from ..modules.fused_moe import ConfigurableMoE, create_moe
from ..modules.fused_moe.routing import DeepSeekV3MoeRoutingMethod
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear as TrtllmLinear
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..modules.situ import SituAndMul
from ..pyexecutor.breakable_cuda_graph import eager_on_graph, is_in_breakable_cuda_graph
from ..utils import ActType_TrtllmGen, get_model_extra_attrs
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model, run_concurrently

# A/B escape hatch: restore nn.Linear for the K3 latent MoE projections
# instead of the min-latency fused GEMM op (read once at import).
_K3_DISABLE_MIN_LATENCY_LATENT_PROJ = (
    os.environ.get("TLLM_K3_DISABLE_MIN_LATENCY_LATENT_PROJ", "0") == "1"
)

_KDA_INDEXED_STATE_POOL_ENABLED = os.environ.get("TLLM_KDA_ENABLE_INDEXED_STATE_POOL", "1") == "1"
# Heuristic ported from SGLang's Blackwell cutoff:
# https://github.com/sgl-project/sglang/blob/e84bbf68efb683c9e2eef4168c5198042544599d/python/sglang/srt/models/kimi_k3.py#L946-L954
# It has not been tuned for TensorRT-LLM; benchmark and retune it for TRT-LLM's
# projection kernels. Verify intentionally counts B * num_steps because those
# flattened token rows form the projection GEMMs' M dimension.
_KDA_BFA_MULTISTREAM_MAX_ROWS = 128

# Routed-expert MoE TP/EP split overrides (read per model init, not import).
# Highest precedence; either one may be set alone, the other is derived from
# tp_size. Without them, an explicit moe_tensor_parallel_size /
# moe_expert_parallel_size pair from the user config is honored, and the
# default stays EP-only (moe_ep == tp_size).
_K3_MOE_TP_ENV = "TLLM_K3_MOE_TP_SIZE"
_K3_MOE_EP_ENV = "TLLM_K3_MOE_EP_SIZE"

if TYPE_CHECKING:
    from transformers import PretrainedConfig

# Identity-RoPE table positions for the MLA backends. K3 is NoPE (the table
# holds cos=1/sin=0), but the chunked-context path indexes the table by
# absolute position, so it must cover max_position_embeddings (~512MB per
# backend for the 1M-position checkpoint); a smaller table is read out of
# bounds. KIMI_K3_MLA_MAX_POSITIONS overrides the size for short-context
# deployments.
_KIMI_K3_MLA_MAX_POSITIONS_ENV = "KIMI_K3_MLA_MAX_POSITIONS"
_KIMI_K3_MLA_DERIVED_PARAM_SUFFIXES = (
    ".self_attn.mixer.k_b_proj_trans",
    ".self_attn.mixer.v_b_proj",
)

# Serve the replicated MoE-layer MLP projections (shared-expert gate/up/down
# and the latent up/down projection) from an FP8 copy of their weights instead
# of BF16. Under attention data-parallelism every rank re-reads these dense
# weights in full on every decode step, so decode is bound by that HBM read;
# an FP8 (e4m3) weight with 128x128 block scales roughly halves those bytes.
# The MLA projections and the routed MXFP4 experts are left untouched (the KDA
# q/k/v/g/o projections have their own switch below). The FP8 weight read is
# lossy relative to BF16, so it is opt-in: set this to "1" to trade accuracy
# for decode bandwidth. Default "0" keeps BF16, which is what the published
# accuracy numbers are measured against.
_KIMI_K3_FP8_WEIGHT_READ_ENV = "KIMI_K3_FP8_WEIGHT_READ"

# Also read the KDA linear-attention q/k/v/g/o projections at FP8 block-scale.
# These are the largest single replicated weight read (~61 GB/rank of the
# ~109 GB BF16 read per decode step). They use the same FP8 path as the MLP
# projections above but are gated separately: the recurrent linear-attention
# core is more accuracy-sensitive than the feed-forward MLPs, so set this to
# "0" to keep the KDA projections in BF16 while still reading the MLPs at FP8.
# The master KIMI_K3_FP8_WEIGHT_READ switch and the SM100 gate still apply.
_KIMI_K3_FP8_WEIGHT_READ_KDA_ENV = "KIMI_K3_FP8_WEIGHT_READ_KDA"

# Also read the MLA (full-attention) q_a/q_b/o and output-gate projections at
# FP8 block-scale. These are the replicated attention weights the MLP pass and
# the KDA pass above leave in BF16, and they are re-read in full by every rank
# each decode step under attention data-parallelism. Two MLA projections are
# deliberately kept in BF16: kv_a_proj_with_mqa outputs kv_lora_rank +
# qk_rope_head_dim (576, not a multiple of 128, so no exact 128x128 block
# scale), and kv_b_proj's weight is consumed directly (not through its forward)
# by the absorbed-decode _kv_b_absorb_split to build the k/v absorb matrices,
# which has no FP8 dequant path. The master KIMI_K3_FP8_WEIGHT_READ switch and
# the SM100 gate still apply.
_KIMI_K3_FP8_WEIGHT_READ_MLA_ENV = "KIMI_K3_FP8_WEIGHT_READ_MLA"

# Expert override (prototype): set to "0" to drop the KimiKDARuntime decode
# fast path — fused qkvg and [f_a | b] projections, persistent conv staging,
# and precomputed kernel-layout constants (``_forward_decode``) — when the
# KDA projections are read at FP8 block-scale. With the fast path kept (the
# default on an enabled master), decode issues the loader's fused FP8
# ``qkvg_proj`` GEMM for q/k/v/g plus one small BF16 GEMV for [f_a | b]
# (``finalize_decode_weights_fp8``), so FP8 weight storage and the decode
# glue savings coexist. Requires the FP8 KDA read to be active; no effect
# otherwise. Default on ("0" disables).
_KIMI_K3_KDA_GLUE_FP8_ENV = "KIMI_K3_KDA_GLUE_FP8"

# FP8 read for the fused shared-expert gate_up_proj. Default follows the
# parallel layout (on under attention DP, off under TP — see the conversion
# helper's comment); set 0/1 to force either.
_KIMI_K3_FP8_WEIGHT_READ_GATE_UP_ENV = "KIMI_K3_FP8_WEIGHT_READ_GATE_UP"


class KimiK3MoEGate(nn.Module):
    """Kimi K3 gate weights and routing method for ``ConfigurableMoE``."""

    def __init__(
        self,
        config: Any,
        *,
        logits_gemm_dtype: torch.dtype | None = None,
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

        assert self.moe_router_activation_func in ("sigmoid", "softmax"), (
            "K3 MoE gate supports sigmoid or softmax scoring only"
        )

        # The checkpoint stores the gate weight in bf16. Storing it in bf16
        # permits the single bf16xbf16 router GEMM while retaining fp32 output.
        weight_dtype = logits_gemm_dtype or torch.float32
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim), dtype=weight_dtype, device=device)
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

    @property
    def routing_method(self) -> DeepSeekV3MoeRoutingMethod:
        """Return the shared DeepSeek-V3 router used by ``ConfigurableMoE``."""
        if self.moe_router_activation_func != "sigmoid":
            raise ValueError("Kimi K3 ConfigurableMoE routing requires sigmoid scores.")
        if not self.moe_renormalize:
            raise ValueError(
                "Kimi K3 ConfigurableMoE routing requires top-k weight renormalization."
            )
        return DeepSeekV3MoeRoutingMethod(
            top_k=self.top_k,
            n_group=self.num_expert_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
            is_fused=True,
        )


class KimiK3RMSNorm(nn.Module):
    """RMSNorm matching the Kimi checkpoint implementation's rounding."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_float = hidden_states.to(torch.float32)
        variance = hidden_states_float.pow(2).mean(-1, keepdim=True)
        hidden_states_float = hidden_states_float * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states_float.to(input_dtype)


def _resolve_fp8_weight_read_gates() -> tuple[bool, bool, bool]:
    """Resolve the FP8 weight-read switches into (master, kda, kda_glue).

    The master switch is opt-in: FP8 weight reads are lossy relative to BF16,
    so a default run keeps BF16 and matches the published accuracy numbers.
    The KDA and KDA-glue switches only narrow an already-enabled master, so
    they stay default-on and are inert while the master is off.
    """
    fp8_weight_read = is_sm_100f() and os.environ.get(_KIMI_K3_FP8_WEIGHT_READ_ENV, "0") not in (
        "",
        "0",
    )
    kda_fp8 = fp8_weight_read and os.environ.get(_KIMI_K3_FP8_WEIGHT_READ_KDA_ENV, "1") != "0"
    kda_glue_fp8 = kda_fp8 and os.environ.get(_KIMI_K3_KDA_GLUE_FP8_ENV, "1") != "0"
    return fp8_weight_read, kda_fp8, kda_glue_fp8


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


KIMI_K3_FUSED_ATTN_RES_ENV = "KIMI_K3_FUSED_ATTN_RES"
"""Set to ``0`` to disable the in-tree fused Torch op
``trtllm::attn_res_fwd`` (Blackwell only). Default: fused with fallback."""

_FUSED_ATTN_RES_ENABLED = os.environ.get(KIMI_K3_FUSED_ATTN_RES_ENV, "1") == "1"


def _apply_attn_res_fused(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: KimiK3RMSNorm
) -> Optional[torch.Tensor]:
    """Fused attn_res via the in-tree ``trtllm::attn_res_fwd`` op.

    Returns ``None`` when the call falls outside the fused kernel's
    contract (dtype/shape/arch) so the caller can use the exact fp32 reference
    instead. ``block_residual`` is kept in the kernel-native ``[K, M, H]``
    layout. Candidate order matches the reference: snapshots first, the
    running prefix sum last.
    """
    if prefix_sum.dtype is not torch.bfloat16:
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    if K + 1 > 12 or M > 16384 or not (4096 <= H <= 8192 and H % 1024 == 0):
        return None
    try:
        attn_res_op = torch.ops.trtllm.attn_res_fwd
    except (AttributeError, RuntimeError):
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    output, _rsigma, _probs, _logits = attn_res_op(
        layer_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
    )
    return output.reshape(M, H)


def _apply_attn_res(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: KimiK3RMSNorm
) -> torch.Tensor:
    """Exact port of HF ``modeling_kimi._apply_attn_res`` (fp32 math).

    prefix_sum:     ``[num_tokens, hidden_size]``
    block_residual: ``[num_snapshots, num_tokens, hidden_size]``

    Unless ``KIMI_K3_FUSED_ATTN_RES=0``, inputs fitting the fused kernel's
    contract dispatch directly to the in-tree ``trtllm::attn_res_fwd`` op.
    Only the fallback boundary restores the HF ``[M, K, H]`` layout.
    """
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_fused(prefix_sum, block_residual, proj, norm)
        if fused is not None:
            return fused
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


# ---------------------------------------------------------------------------
# Dense / shared-expert MLP: fused [gate | up] layout (``GatedMLP``).
#
# The HF checkpoint stores separate ``gate_proj`` / ``up_proj`` tensors;
# ``load_weights`` row-concatenates them into ``gate_up_proj`` (see
# ``_gate_up_ckpt_keys``), replacing two GEMMs + torch.cat with one GEMM.
# ---------------------------------------------------------------------------


_GATE_UP_FUSED_SUFFIX = ".gate_up_proj.weight"


def _gate_up_ckpt_keys(fused_key: str) -> Tuple[str, str]:
    """Checkpoint ``(gate_proj, up_proj)`` keys whose row-concat loads the
    fused ``gate_up_proj`` parameter named by ``fused_key``."""
    return (
        fused_key.replace(_GATE_UP_FUSED_SUFFIX, ".gate_proj.weight"),
        fused_key.replace(_GATE_UP_FUSED_SUFFIX, ".up_proj.weight"),
    )


# ---------------------------------------------------------------------------
# FP8 block-scale weight read for the replicated MoE-layer MLP projections.
# ---------------------------------------------------------------------------


class _Fp8BlockScaleWeightReadLinear(nn.Module):
    """Bias-free linear replacement that reads its weight at FP8.

    The BF16 weight ``[out, in]`` is quantized once (at load) to
    ``float8_e4m3fn`` with 128x128 block scales, then served through the
    DeepGEMM ``fp8_swap_ab_gemm`` kernel — the same FP8 block-scale GEMM the
    quantized DeepSeek block-scale path uses. The activation stays BF16 and is
    quantized inside the kernel, so only the weight's storage/read precision
    changes. Halving the weight bytes cuts the dominant HBM read that bounds
    K3's memory-bound decode step. Both ``out`` and ``in`` are multiples of
    128 for every projection this is applied to, so the block scales cover the
    weight exactly.
    """

    def __init__(
        self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor, out_features: int
    ) -> None:
        super().__init__()
        self.in_features = weight_fp8.shape[1]
        self.out_features = out_features
        # Buffers (not parameters): these are the module's weights post-load;
        # there is nothing further to load into them and they must not be
        # touched by any later autocast/dtype move.
        self.register_buffer("weight", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

    @property
    def has_fp8_qdq(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def has_w4a8_nvfp4_fp8(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @staticmethod
    def quantize_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """BF16 ``[out, in]`` weight -> (FP8 weight, deep_gemm-ready scale).

        Both dims must be multiples of 128. Because the 128x128 block scale is
        computed per block, concatenating several such weights along ``out``
        and quantizing the result is per-block identical to quantizing each
        separately (no block crosses a 128-aligned boundary), so a fused
        weight's row slices equal the individually quantized weights.
        """
        # Lazy imports: only pulled in on the FP8 path.
        from ...deep_gemm.utils.math import per_block_cast_to_fp8
        from ...quantization.utils.fp8_utils import (
            resmooth_to_fp8_e8m0,
            transform_sf_into_required_layout,
        )

        # 128x128 block-scale FP8 weight, then the exact SM100 deep_gemm scale
        # preparation the shipping FP8-block-scale Linear uses: resmooth to
        # UE8M0 and pack the scale into deep_gemm's TMA-aligned MN-major
        # layout. fp8_swap_ab_gemm runs with disable_ue8m0_cast=True, so it
        # consumes this pre-formatted scale directly (a plain FP32 block scale
        # would be misread and produce garbage).
        weight_fp8, weight_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        weight_fp8, weight_scale = resmooth_to_fp8_e8m0(
            weight_fp8.contiguous(), weight_scale.contiguous().float()
        )
        weight_scale = transform_sf_into_required_layout(
            weight_scale,
            mn=weight_fp8.shape[0],
            k=weight_fp8.shape[1],
            recipe=(1, 128, 128),
            is_sfa=False,
        )
        return weight_fp8, weight_scale

    @classmethod
    def from_linear(cls, linear: nn.Linear | TrtllmLinear) -> "_Fp8BlockScaleWeightReadLinear":
        assert linear.bias is None, "FP8 weight read expects a bias-free Linear"
        weight_fp8, weight_scale = cls.quantize_weight(linear.weight.data)
        return cls(weight_fp8, weight_scale, linear.out_features)

    def forward(
        self,
        x: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if lora_params:
            raise NotImplementedError("Kimi K3 FP8 weight read does not support LoRA.")
        out_shape = x.shape[:-1] + (self.out_features,)
        out = torch.ops.trtllm.fp8_swap_ab_gemm(
            x.reshape(-1, x.shape[-1]),
            self.weight,
            self.weight_scale,
            output_dtype=x.dtype,
            disable_ue8m0_cast=True,
        )
        return out.reshape(out_shape)


def _swap_linear_to_fp8_weight_read(
    parent: nn.Module,
    attr: str,
    linear_types: Tuple[type, ...] = (nn.Linear,),
) -> int:
    """Replace ``parent.<attr>`` with an FP8 weight-read module if it is a
    plain linear of one of ``linear_types``; return the number of modules
    converted (0 or 1), so callers can accumulate a conversion count.

    Frees the original BF16 weight storage immediately: the loader holds a
    transient name->Parameter map that keeps it alive until load returns, so
    without this the FP8 copy is purely additive and fragments the pool the
    FP8 GEMM autotuner and KV-cache init need.
    """
    child = getattr(parent, attr, None)
    if not isinstance(child, linear_types):
        return 0
    setattr(parent, attr, _Fp8BlockScaleWeightReadLinear.from_linear(child))
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
    (MLA/KDA), the routed MXFP4 experts and the dense layer-0 MLP are left in
    BF16. Returns the number of projections converted.
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
            # GatedMLP fuses gate and up into gate_up_proj; keep the split
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
                if isinstance(child, TrtllmLinear) and child.tp_size != 1:
                    continue
                count += _swap_linear_to_fp8_weight_read(
                    shared, attr, linear_types=(nn.Linear, TrtllmLinear)
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


def _convert_kda_projections_to_fp8_weight_read(model: nn.Module) -> int:
    """Swap the KDA linear-attention q/k/v/g/o projections to an FP8 weight read.

    Targets the large bias-free BF16 projections of every KDA linear-attention
    layer (``q_proj``/``k_proj``/``v_proj``/``g_proj``/``o_proj``, each
    ``[out, in]`` with both dims a multiple of 128) — the single largest
    replicated weight read, re-read in full by every rank each decode step
    under attention data-parallelism. The smaller state-path projections are
    left in BF16 on purpose: ``b_proj`` outputs ``num_heads`` (not a multiple
    of 128, so no exact 128x128 block scale), and the forget gate
    ``f_a``/``f_b``, the low-rank ``g_a``/``g_b`` gate, the short convolutions
    and ``dt`` are small and feed the accuracy-sensitive recurrent decay.

    ``q_proj``/``k_proj``/``v_proj`` and the full-rank ``g_proj`` all read the
    same normed hidden, so their weights are additionally concatenated into one
    fused ``qkvg_proj`` FP8 GEMM used by prefill, decode, and verification;
    all three consume all four outputs. The fused weight is the only storage —
    the individual ``q_proj``/``k_proj``/``v_proj``/``g_proj`` modules are
    rebuilt to read a **view** of their slice of it (with their own block
    scale), so verify and fallback paths can still call them per projection.
    ``o_proj`` reads the decode-kernel output (not the shared hidden) and is
    converted on its own. Returns the number of projections converted.
    """
    count = 0

    for layer in model.layers:
        if not getattr(layer, "is_kda", False) or not _has_weights(layer):
            continue
        mixer = getattr(getattr(layer, "self_attn", None), "mixer", None)
        if mixer is None:
            continue

        # Projections that read the same normed hidden (g_proj only in the
        # full-rank-gate config; the low-rank g_a/g_b gate stays BF16).
        group = [(a, getattr(mixer, a, None)) for a in ("q_proj", "k_proj", "v_proj", "g_proj")]
        group = [(a, c) for a, c in group if isinstance(c, nn.Linear)]

        if group:
            # One fused FP8 weight [sum(out), in]; row slices equal the
            # individually quantized weights (see quantize_weight).
            fused_bf16 = torch.cat([c.weight.data for _, c in group], dim=0)
            fused_fp8, fused_scale = _Fp8BlockScaleWeightReadLinear.quantize_weight(fused_bf16)
            fused = _Fp8BlockScaleWeightReadLinear(fused_fp8, fused_scale, fused_bf16.shape[0])
            mixer.qkvg_proj = fused
            mixer.qkvg_split_sizes = [c.out_features for _, c in group]
            del fused_bf16

            # Rebuild each projection to read a view of its slice of the fused
            # weight (own block scale); the fused weight is the sole storage.
            offset = 0
            for attr, child in group:
                n = child.out_features
                _, own_scale = _Fp8BlockScaleWeightReadLinear.quantize_weight(child.weight.data)
                setattr(
                    mixer,
                    attr,
                    _Fp8BlockScaleWeightReadLinear(fused.weight[offset : offset + n], own_scale, n),
                )
                # Free the original BF16 storage (the loader's transient
                # name->Parameter map keeps it alive until load returns, so
                # without this the FP8 copy is purely additive on the tight
                # DEP16 pool).
                child.weight.data = child.weight.data.new_empty(0)
                offset += n
                count += 1

        # o_proj reads the decode-kernel output, so it is not part of the fused
        # hidden-reading group; convert it on its own.
        count += _swap_linear_to_fp8_weight_read(mixer, "o_proj")

    if count:
        gc.collect()
        torch.cuda.empty_cache()
    return count


def _convert_mla_projections_to_fp8_weight_read(model: nn.Module) -> int:
    """Swap the MLA q_a/q_b/o and output-gate projections to an FP8 weight read.

    Targets the large bias-free BF16 projections of every MLA (full-attention)
    layer that are read only through their ``forward`` — ``q_a_proj``,
    ``q_b_proj``, ``o_proj`` and, when the output gate is enabled, ``g_proj``,
    each ``[out, in]`` with both dims a multiple of 128 — replicated attention
    weights re-read in full by every rank each decode step under attention
    data-parallelism. Two MLA projections are left in BF16 on purpose:
    ``kv_a_proj_with_mqa`` outputs ``kv_lora_rank + qk_rope_head_dim`` (576, not
    a multiple of 128), and ``kv_b_proj`` supplies ``k_b_proj_trans`` and
    ``v_b_proj`` directly (the absorbed generation path never calls its
    ``forward``), with no FP8 dequant path. Returns the number of projections
    converted.
    """
    count = 0

    for layer in model.layers:
        # MLA layers are the non-KDA layers (each layer is exactly one of the
        # two); their projections live on the KimiK3MLAAttention mixer.
        if getattr(layer, "is_kda", False) or not _has_weights(layer):
            continue
        mixer = getattr(getattr(layer, "self_attn", None), "mixer", None)
        if mixer is None:
            continue
        # g_proj exists only when the MLA output gate is enabled; a missing
        # attr is a safe no-op.
        for attr in ("q_a_proj", "q_b_proj", "o_proj", "g_proj"):
            count += _swap_linear_to_fp8_weight_read(
                mixer, attr, linear_types=(nn.Linear, TrtllmLinear)
            )

    if count:
        gc.collect()
        torch.cuda.empty_cache()
    return count


# ---------------------------------------------------------------------------
# Latent MoE block using the unified ConfigurableMoE stack.
# ---------------------------------------------------------------------------


class KimiK3MoERuntime(nn.Module):
    """Kimi K3 latent MoE block backed by ConfigurableMoE."""

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
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

        situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
        situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
        dtype = torch.bfloat16

        # Routing scores stay fp32; with attention-DP off the gate GEMM runs
        # bf16xbf16 with fp32 accumulate/output (checkpoint stores the gate
        # weight in bf16; saves a per-layer input cast + fp32 splitK pair on
        # the bs1 decode path). Under attention-DP the legacy upcast-to-fp32
        # GEMM is kept: the bf16-input min-latency GEMM's different reduction
        # order flips borderline top-16 picks (GSM8K 96.7 -> 96.1/96.4,
        # 3-run bisect on 62b20dd868), and the bs1-latency win is irrelevant
        # at DEP batch sizes. KIMI_K3_ROUTER_BF16=1/0 forces either path.
        _router_bf16_env = os.environ.get("KIMI_K3_ROUTER_BF16")
        _router_bf16 = (
            _router_bf16_env == "1"
            if _router_bf16_env is not None
            else not model_config.mapping.enable_attention_dp
        )
        self.gate = KimiK3MoEGate(cfg, logits_gemm_dtype=torch.bfloat16 if _router_bf16 else None)

        routed_moe_model_config = self._routed_moe_model_config(model_config)
        routed_quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8)
        routed_moe_kwargs = dict(
            routing_method=self.gate.routing_method,
            num_experts=self.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            dtype=dtype,
            # Kimi owns the latent reduction so direct MoE-TP can combine it
            # with the shared-expert partial in one collective below.
            reduce_results=False,
            model_config=routed_moe_model_config,
            override_quant_config=routed_quant_config,
            layer_idx=layer_idx,
            # Let CommunicationFactory select the best available strategy.
            communication_method=None,
        )
        if routed_moe_model_config.moe_backend == "TRTLLM":
            routed_moe_kwargs.update(
                trtllm_gen_activation_type=ActType_TrtllmGen.SiTu,
                # Cubin alpha is the gate-side SiTU beta; cubin beta is the
                # linear-side SiTU beta.
                trtllm_gen_activation_alpha=float(situ_beta),
                trtllm_gen_activation_beta=float(
                    situ_linear_beta if situ_linear_beta is not None else 1.0
                ),
            )
        elif routed_moe_model_config.moe_backend == "MEGAMOE_DEEPGEMM":
            routed_moe_kwargs.update(
                activation="situ",
                situ_beta=float(situ_beta),
                situ_linear_beta=float(situ_linear_beta if situ_linear_beta is not None else 1.0),
            )
        self.routed_experts = create_moe(**routed_moe_kwargs)
        if not isinstance(self.routed_experts, ConfigurableMoE):
            raise RuntimeError(
                "Kimi K3 requires ConfigurableMoE; ENABLE_CONFIGURABLE_MOE must not be disabled."
            )
        if routed_moe_model_config.moe_backend == "MEGAMOE_DEEPGEMM":
            from ..modules.fused_moe.mega_moe import MegaMoEDeepGemm

            if not isinstance(self.routed_experts.backend, MegaMoEDeepGemm):
                raise RuntimeError(
                    "Kimi K3 explicitly requested MEGAMOE_DEEPGEMM, but the "
                    f"MoE factory selected {type(self.routed_experts.backend).__name__}."
                )
        if self.routed_experts.layer_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "dynamic EPLB or replicated expert slots."
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
        attention_dp = model_config.mapping.enable_attention_dp
        shared_model_config = copy.copy(model_config)
        shared_model_config.quant_config = QuantConfig()
        # Under attention DP each rank owns different tokens, so the shared
        # expert is replicated (TP size 1) and must not reduce across ranks.
        # Direct MoE-TP leaves both branches as partials for one concatenated
        # all-reduce.
        use_shared_tp = not attention_dp and model_config.mapping.tp_size > 1
        routed_all_reduce = self.routed_experts.all_reduce
        if use_shared_tp and routed_all_reduce is None:
            raise RuntimeError(
                "Kimi K3 direct MoE tensor parallelism requires the "
                "ConfigurableMoE all-reduce even when reduce_results=False."
            )
        self._use_combined_all_reduce = use_shared_tp
        self.shared_experts = GatedMLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=shared_intermediate,
            bias=False,
            activation=SituAndMul(
                beta=situ_beta,
                linear_beta=situ_linear_beta,
                use_fused_activation=True,
            ),
            dtype=dtype,
            config=shared_model_config,
            overridden_tp_size=1 if attention_dp else None,
            reduce_output=False,
            layer_idx=layer_idx,
            is_shared_expert=True,
        )
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on) and aux_stream is set; otherwise both run in
        # order on the default stream.
        self.aux_stream = aux_stream
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
        self.routed_expert_down_proj = nn.Linear(
            cfg.hidden_size, self.moe_hidden_size, bias=False, dtype=dtype
        )
        self.routed_expert_up_proj = nn.Linear(
            self.moe_hidden_size, cfg.hidden_size, bias=False, dtype=dtype
        )
        # Stock fused RMSNorm (flashinfer kernel; the no-flashinfer
        # fallback is the same fp32-variance eager math as KimiK3RMSNorm).
        self.routed_expert_norm = RMSNorm(
            hidden_size=self.moe_hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )

    @staticmethod
    def _routed_projection(hidden_states: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if _K3_DISABLE_MIN_LATENCY_LATENT_PROJ or not isinstance(projection, nn.Linear):
            return projection(hidden_states)
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(
            hidden_states, projection.weight.t(), None, None
        )

    @staticmethod
    def _select_moe_tp_ep(mapping: Mapping) -> Tuple[int, int]:
        """Resolve the routed-expert ``(moe_tp, moe_ep)`` split.

        Precedence:

        1. ``TLLM_K3_MOE_TP_SIZE`` / ``TLLM_K3_MOE_EP_SIZE`` env overrides
           (either alone; the other is derived from ``tp_size``).
        2. Explicit ``moe_tensor_parallel_size`` / ``moe_expert_parallel_size``
           from the user config. Detected via
           ``mapping.moe_tp_ep_user_specified`` so the auto-resolved mapping
           default (``moe_tp=tp_size, moe_ep=1``) is NOT mistaken for a TP
           request.
        3. Default: EP-only (``moe_tp=1, moe_ep=tp_size``), the historical
           K3 layout.
        """
        tp_size = mapping.tp_size
        env_tp = os.environ.get(_K3_MOE_TP_ENV)
        env_ep = os.environ.get(_K3_MOE_EP_ENV)
        if env_tp is not None or env_ep is not None:
            moe_tp = int(env_tp) if env_tp is not None else 0
            moe_ep = int(env_ep) if env_ep is not None else 0
            if moe_tp <= 0 and moe_ep > 0:
                moe_tp = tp_size // moe_ep
            elif moe_ep <= 0 and moe_tp > 0:
                moe_ep = tp_size // moe_tp
            return moe_tp, moe_ep
        if getattr(mapping, "moe_tp_ep_user_specified", False):
            return mapping.moe_tp_size, mapping.moe_ep_size
        return 1, tp_size

    @staticmethod
    def _routed_moe_model_config(model_config: ModelConfig) -> ModelConfig:
        """Build a private routed-expert mapping without mutating the shared
        config. Default split is EP-only; see ``_select_moe_tp_ep``."""
        supported_backends = {"TRTLLM", "MEGAMOE_DEEPGEMM"}
        if model_config.moe_backend not in supported_backends:
            raise ValueError(
                "Kimi K3 SiTU routed experts only support the TRTLLM and "
                "MEGAMOE_DEEPGEMM backends; "
                f"got {model_config.moe_backend!r}."
            )
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        mapping = model_config.mapping
        if getattr(mapping, "_dwdp_size", 0) > 1:
            raise NotImplementedError("Kimi K3 packed-checkpoint streaming does not support DWDP.")

        moe_tp, moe_ep = KimiK3MoERuntime._select_moe_tp_ep(mapping)
        if moe_tp < 1 or moe_ep < 1 or moe_tp * moe_ep != mapping.tp_size:
            raise ValueError(
                f"Kimi K3 routed MoE split moe_tp={moe_tp} x moe_ep={moe_ep} "
                f"must multiply to tp_size={mapping.tp_size}."
            )
        if moe_tp > 1 and mapping.enable_attention_dp:
            raise NotImplementedError(
                "Kimi K3 MoE tensor parallelism requires "
                "enable_attention_dp=false (the attention-DP dispatch/combine "
                "path is validated for EP-only splits)."
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
        # MegaMoE uses this value as global DP SymmBuffer capacity, then
        # divides it by EP size for the per-rank allocation. Other backends
        # keep the user-configured value as their MoE chunking bound.
        # Preserve an explicitly larger capacity.
        if routed_model_config.moe_backend == "MEGAMOE_DEEPGEMM":
            default_moe_max_num_tokens = routed_model_config.max_num_tokens * routed_mapping.dp_size
            configured_moe_max_num_tokens = int(routed_model_config.moe_max_num_tokens or 0)
            if configured_moe_max_num_tokens < default_moe_max_num_tokens:
                logger.info_once(
                    "Kimi K3 MegaMoE raises moe_max_num_tokens from "
                    f"{configured_moe_max_num_tokens} to {default_moe_max_num_tokens} "
                    "because the global DP SymmBuffer requires capacity for "
                    "max_num_tokens * dp_size.",
                    key=(
                        "kimi_k3_megamoe_capacity_override_"
                        f"{configured_moe_max_num_tokens}_{default_moe_max_num_tokens}"
                    ),
                )
            routed_model_config.moe_max_num_tokens = max(
                configured_moe_max_num_tokens,
                default_moe_max_num_tokens,
            )
        routed_model_config._frozen = True
        return routed_model_config

    def forward(self, hidden_states: torch.Tensor, all_rank_num_tokens=None) -> torch.Tensor:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16."""
        identity = hidden_states
        router_logits = self.gate.compute_logits(hidden_states)
        moe_all_reduce = self.routed_experts.all_reduce if self._use_combined_all_reduce else None

        def _routed_output():
            # Latent down/up projections via the min-latency fused GEMM op:
            # at <=16 tokens (decode graphs) it runs a single pipelined
            # bf16 kernel per projection instead of cuBLAS's split-K GEMV +
            # splitKreduce pair (~17+3.6us -> ~8us for 7168->3584 at M=1);
            # for larger token counts the op falls back to cuBLAS internally.
            # TLLM_K3_DISABLE_MIN_LATENCY_LATENT_PROJ=1 restores nn.Linear
            # (A/B escape hatch). When the FP8 weight-read conversion has
            # replaced the projection module, call it directly: its weight is
            # an e4m3 buffer the bf16 dsv3 op must not read, and its forward
            # is already a single fused GEMM (fp8_swap_ab_gemm).
            routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
            y = self.routed_experts(
                routed_in,
                router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
            )
            if self._use_combined_all_reduce:
                return y
            # Communication-backed paths return a complete routed result.
            y = self.routed_expert_norm(y)
            return self._routed_projection(y, self.routed_expert_up_proj)

        # Shared experts depend only on the block input, so overlap their GEMMs
        # with the routed dispatch/expert/combine chain. Multi-stream engages
        # only under CUDA graphs; otherwise both branches run in order on the
        # default stream. Direct MoE-TP leaves both branches as partial sums
        # until the streams join, then reduces them with one collective.
        routed_out, shared_out = maybe_execute_in_parallel(
            _routed_output,
            lambda: self.shared_experts(identity),
            self.moe_main_event,
            self.moe_shared_event,
            self.aux_stream,
            disable_on_compile=True,
        )
        if self._use_combined_all_reduce:
            combined = moe_all_reduce(torch.cat((shared_out, routed_out), dim=-1))
            shared_out, routed_latent = torch.split(
                combined,
                (self.hidden_size, self.moe_hidden_size),
                dim=-1,
            )
            # The column split is a strided view; FlashInfer RMSNorm expects
            # a dense last dimension.
            routed_latent = self.routed_expert_norm(routed_latent.contiguous())
            routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        return routed_out + shared_out


# ---------------------------------------------------------------------------
# KDA runtime (pool-backed prefill / decode via the FLA kernels).
# ---------------------------------------------------------------------------


def _kda_split_conv_sections(
    cs: torch.Tensor, d: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a gathered ``[N, 3D, W]`` conv-cache into contiguous q/k/v."""
    return (cs[:, :d].contiguous(), cs[:, d : 2 * d].contiguous(), cs[:, 2 * d :].contiguous())


def _extract_kda_extra_attrs(layer_idx: str):
    """Resolve the live attention metadata and KDA runtime for ``layer_idx``.

    KDA analog of ``mla._extract_mla_extra_attrs``: the breakable-CUDA-graph
    op receives only a string layer id, and recovers the metadata object and
    the ``KimiKDARuntime`` instance from the weak-ref registry published in
    the model's ``extra_attrs`` (so nothing metadata-dependent is baked into
    the captured graph).
    """
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
    assert isinstance(kda_layer, KimiKDARuntime), "KDA layer must be KimiKDARuntime"
    return metadata, kda_layer


def kda_core_inplace(hidden_states: torch.Tensor, layer_idx: str, output: torch.Tensor) -> None:
    """Run the metadata-dependent KDA attention core, writing ``output``.

    Wrapped by ``eager_on_graph`` (see ``maybe_bcg_kda_core_inplace``) so it
    runs eagerly between captured breakable-CUDA-graph segments. ``output`` is
    the pre-``o_proj`` core buffer (``[num_tokens, H, head_dim]``) pre-allocated
    on-graph by the caller; the on-graph epilogue applies ``o_proj`` +
    ``o_allreduce``.
    """
    metadata, kda_layer = _extract_kda_extra_attrs(layer_idx)
    kda_layer._forward_impl(hidden_states, metadata, output=output)


maybe_bcg_kda_core_inplace = eager_on_graph(kda_core_inplace)


class KimiKDARuntime(nn.Module):
    """Wraps the parity-tested ``KimiKDALinearAttention`` parameters with a
    cache-pool-aware forward for the executor flow.

    Parameter names mirror the HF checkpoint 1:1 (the wrapped mixer is
    registered under the layer as ``self_attn``, so e.g.
    ``model.layers.N.self_attn.q_proj.weight`` maps identically).
    """

    def __init__(
        self,
        cfg,
        layer_idx: int,
        mapping=None,
        allreduce_strategy=AllReduceStrategy.AUTO,
        aux_stream: Optional[torch.cuda.Stream] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        # Lazy import: pulls in fla/einops.
        from ..modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

        lin = cfg.linear_attn_config
        self.layer_idx = layer_idx
        self._use_indexed_ssm_pool = _KDA_INDEXED_STATE_POOL_ENABLED
        # Attention-family TP semantics (Qwen3-Next GatedDeltaNet pattern,
        # gdn_mixer.py): replicated under attention-DP — each rank runs
        # its own batch with the full head set — and head-sharded across
        # mapping.tp_size otherwise: every rank holds the same batch, runs
        # its 1/tp head slice, and the row-sharded o_proj partials are
        # all-reduced at the end of forward().
        if mapping is not None and mapping.tp_size > 1 and not mapping.enable_attention_dp:
            self._kda_tp_size = mapping.tp_size
        else:
            self._kda_tp_size = 1
        self._kda_tp_rank = mapping.tp_rank if self._kda_tp_size > 1 else 0
        self._o_allreduce = (
            AllReduce(mapping=mapping, strategy=allreduce_strategy, dtype=torch.bfloat16)
            if self._kda_tp_size > 1
            else None
        )
        num_heads = lin["num_heads"]
        assert num_heads % self._kda_tp_size == 0, (
            f"KDA num_heads {num_heads} not divisible by tp_size {self._kda_tp_size}"
        )
        self.mixer = KimiKDALinearAttention(
            hidden_size=cfg.hidden_size,
            num_heads=num_heads // self._kda_tp_size,
            head_dim=lin["head_dim"],
            conv_kernel_size=lin["short_conv_kernel_size"],
            use_full_rank_gate=lin.get("use_full_rank_gate", True),
            gate_lower_bound=lin.get("gate_lower_bound", None),
            rms_norm_eps=cfg.rms_norm_eps,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            # Use TLLM_KDA_ENABLE_OPT_PREFILL=0 to opt out of the optimized
            # prefill kernel.
            use_optimized_prefill=os.getenv("TLLM_KDA_ENABLE_OPT_PREFILL", "1") == "1",
            use_optimized_decode=True,
        )
        self.proj_size = (num_heads // self._kda_tp_size) * lin["head_dim"]
        # Fused prefill/decode/verify projection weights, built after checkpoint
        # load. BF16 uses separate fused [q | k | v | g] and [f_a | b]
        # GEMMs; FP8 supplies qkvg through the mixer's fused projection and
        # reuses the BF16 [f_a | b] weight.
        self._qkvg_proj_weight: Optional[torch.Tensor] = None
        self._bfa_proj_weight: Optional[torch.Tensor] = None
        self._w_q_t = self._w_k_t = self._w_v_t = None
        self._A_log_f32 = self._dt_bias_f32 = self._onorm_w_f32 = None
        # Fork/join state for overlapping the small [f_a | b] -> f_b chain
        # with the wide qkvg projection during CUDA-graph execution.
        self._projection_aux_stream = aux_stream
        self._projection_fork_event = torch.cuda.Event()
        self._projection_join_event = torch.cuda.Event()
        # Persistent batch-row-dense staging for the fused decode kernel's
        # per-section conv windows. Sized once, on the first decode call,
        # to the conv pool's slot count and never reallocated (see
        # ``_forward_decode``).
        self._cs_dense: Optional[torch.Tensor] = None
        # fp32 [dim, W] conv weights for the fused verify kernel, prebuilt
        # by ``_build_mtp_conv_weights()`` at weight-load finalize time.
        self._mtp_conv_weights: Optional[Tuple[torch.Tensor, ...]] = None

        # Breakable-CUDA-graph registration. When a shared model config is
        # threaded in, publish a weak ref under a unique layer id so the
        # eager-on-graph op (``kda_core_inplace``) can recover this runtime at
        # replay time; ``register_to_config`` gates the BCG path in forward().
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

    def finalize_decode_weights(self) -> None:
        """Build fused projection weights and decode constants after weight load.

        1. Separate fused ``[q | k | v | g]`` and ``[f_a | b]`` projections.
           Keeping the wide qkvg output aligned avoids degrading its GEMM
           kernel selection with the small f_a and b tails. Source parameters
           are repointed to row views of the fused buffers, so prefill and
           verify paths keep using them without duplicate weight storage.
        2. Kernel-layout constants that ``_decode_via_optimized`` used to
           rebuild with ~6 device kernels per layer per decode step:
           transposed conv weights (bf16 ``[W, D]``) and fp32 copies of
           ``A_log`` / ``dt_bias`` / ``o_norm.weight``.
        """
        mixer = self.mixer
        if mixer._dispatch.decode_kernel_path != "optimized" or not mixer.use_full_rank_gate:
            return
        if mixer.q_proj.weight.device.type != "cuda":
            return
        with torch.no_grad():
            qkvg_modules = (
                mixer.q_proj,
                mixer.k_proj,
                mixer.v_proj,
                mixer.g_proj,
            )
            qkvg_weight = self._merge_projection_weights(qkvg_modules)
            # Eight BF16 outputs occupy 16 bytes, so padding keeps each output row
            # aligned for vectorized f_b consumption; it is not a kernel requirement.
            bfa_weight = self._merge_projection_weights(
                (mixer.f_a_proj, mixer.b_proj), pad_rows_to=8
            )
            self._build_decode_kernel_constants()
            self._bfa_proj_weight = bfa_weight
            # Publish last: both weights are required by the BF16 fast path.
            self._qkvg_proj_weight = qkvg_weight

    @staticmethod
    def _merge_projection_weights(
        modules: tuple[nn.Linear, ...], pad_rows_to: int = 1
    ) -> torch.Tensor:
        """Concatenate linear weights and repoint the modules to row views."""
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

    def _build_decode_kernel_constants(self) -> None:
        """Kernel-layout constants shared by both finalize variants."""
        mixer = self.mixer
        self._w_q_t = (
            mixer.q_conv1d.weight.detach()
            .squeeze(1)
            .transpose(0, 1)
            .to(torch.bfloat16)
            .contiguous()
        )
        self._w_k_t = (
            mixer.k_conv1d.weight.detach()
            .squeeze(1)
            .transpose(0, 1)
            .to(torch.bfloat16)
            .contiguous()
        )
        self._w_v_t = (
            mixer.v_conv1d.weight.detach()
            .squeeze(1)
            .transpose(0, 1)
            .to(torch.bfloat16)
            .contiguous()
        )
        self._A_log_f32 = mixer.A_log.detach().float().contiguous()
        self._dt_bias_f32 = mixer.dt_bias.detach().float().contiguous()
        self._onorm_w_f32 = mixer.o_norm.weight.detach().float().contiguous()
        # Build the fused-verify conv constants eagerly too, so the first
        # verify call never allocates (a capture-unsafe lazy allocation).
        self._build_mtp_conv_weights()

    def finalize_decode_weights_fp8(self) -> None:
        """FP8 counterpart of ``finalize_decode_weights()``.

        Runs AFTER ``_convert_kda_projections_to_fp8_weight_read``, so
        q/k/v/g already live in the mixer's fused FP8 ``qkvg_proj`` GEMM.
        Only the two small BF16 projections reading the same hidden —
        ``f_a_proj`` and ``b_proj`` (kept BF16 by the FP8 conversion: outputs
        are not 128-multiples and feed the accuracy-sensitive recurrent
        decay) — are fused here into one ``[f_a | b]`` weight, with the source
        parameters repointed to row views. Prefill, decode, and verification
        then share both fused projections; the kernel-layout constants are
        decode-only.
        """
        mixer = self.mixer
        if mixer._dispatch.decode_kernel_path != "optimized" or not mixer.use_full_rank_gate:
            return
        fused_qkvg = getattr(mixer, "qkvg_proj", None)
        split_sizes = getattr(mixer, "qkvg_split_sizes", None)
        if fused_qkvg is None or split_sizes is None or len(split_sizes) != 4:
            return
        if mixer.f_a_proj.weight.device.type != "cuda":
            return
        with torch.no_grad():
            bfa_weight = self._merge_projection_weights(
                (mixer.f_a_proj, mixer.b_proj), pad_rows_to=8
            )
            self._build_decode_kernel_constants()
            # Publish last: enables fused [f_a | b] in prefill/decode/verify.
            self._bfa_proj_weight = bfa_weight

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """Entry point: select the eager path or the breakable-CUDA-graph path.

        ``hidden_states``: flattened ``[num_tokens, hidden]`` (ctx tokens first,
        then one token per generation request).

        Both paths ultimately run the same core (``_forward_impl``); the split
        is purely eager vs graph. The eager path calls ``_forward_impl``
        directly (no op, no extra-attrs lookup). The BCG path pre-allocates the
        pre-``o_proj`` core on-graph and fills it via the eager-on-graph op —
        which recovers this layer + the live metadata from the extra-attrs
        registry and calls ``_forward_impl(output=core)`` — then keeps o_proj +
        o_allreduce on-graph.
        """
        if self.register_to_config and is_in_breakable_cuda_graph():
            core = hidden_states.new_empty(
                (hidden_states.shape[0], self.mixer.num_heads, self.mixer.head_dim),
                dtype=torch.bfloat16,
            )
            maybe_bcg_kda_core_inplace(hidden_states, self.layer_idx_str, core)
            out = self.mixer.o_proj(core.view(hidden_states.shape[0], self.proj_size))
            if self._o_allreduce is not None:
                out = self._o_allreduce(out)
            return out

        return self._forward_impl(hidden_states, attn_metadata)

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Metadata-dependent KDA core (prefill/decode/verify dispatch).

        This is what the breakable-CUDA-graph op targets, and what the eager
        ``forward`` calls directly. ``output`` is the BCG core buffer
        (``[num_tokens, H, head_dim]``, pre-``o_proj``): when supplied, each
        sub-path writes its post-``o_norm`` result into the corresponding slice
        and this returns ``None`` (the caller applies o_proj + o_allreduce
        on-graph). When ``None`` (eager), it returns the fully projected,
        reduced output tensor.
        """
        mamba_metadata = attn_metadata.mamba_metadata
        num_prefills = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        # Batches may be padded to the CUDA-graph bucket; only the first
        # ``num_tokens`` rows are real.
        num_tokens = attn_metadata.num_tokens
        batch_size = attn_metadata.seq_lens.shape[0]
        # index_copy_/index_select need int64 indices; the int64 mirror is
        # prepared once per step by Mamba2Metadata.prepare() so KDA layers
        # do not each replay an int32->int64 cast inside the decode graph.
        state_indices = getattr(mamba_metadata, "state_indices_long", None)
        if state_indices is None or state_indices.shape[0] != batch_size:
            state_indices = mamba_metadata.state_indices[:batch_size].long()
        cu_seqlens = mamba_metadata.query_start_loc_long[: num_prefills + 1]
        num_decodes = batch_size - num_prefills

        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        conv_pool = layer_cache.conv  # [slots, 3D, W] bf16
        ssm_pool = layer_cache.temporal  # [slots, H, V, K] fp32

        outputs: List[torch.Tensor] = []
        if num_prefills > 0:
            prefill_out = self._forward_prefill(
                hidden_states[:num_ctx_tokens],
                cu_seqlens,
                mamba_metadata,
                num_prefills,
                conv_pool,
                ssm_pool,
                state_indices[:num_prefills],
                layer_cache,
                output=output[:num_ctx_tokens] if output is not None else None,
            )
            if output is None:
                outputs.append(prefill_out)
        if num_decodes > 0:
            decode_rows = num_tokens - num_ctx_tokens
            if decode_rows == num_decodes:
                decode_out = self._forward_decode(
                    hidden_states[num_ctx_tokens:num_tokens],
                    conv_pool,
                    ssm_pool,
                    state_indices[num_prefills:],
                    mamba_metadata,
                    layer_cache,
                    ssm_state_indices=(
                        mamba_metadata.state_indices[num_prefills:batch_size]
                        if self._use_indexed_ssm_pool
                        else None
                    ),
                    output=output[num_ctx_tokens:num_tokens] if output is not None else None,
                )
                if output is None:
                    outputs.append(decode_out)
            else:
                # Speculative verification: each generation request carries
                # 1 + draft_len tokens (drafts are padded to the static max,
                # so T is uniform). Per-step states go to the manager's
                # SpeculativeState scratch buffers — never the live pools —
                # and kv_cache_manager.update_mamba_states() promotes the
                # accepted step after sampling.
                if output is not None:
                    raise NotImplementedError(
                        "Breakable CUDA graph KDA does not support speculative "
                        "verification batches yet"
                    )
                assert decode_rows % num_decodes == 0, (
                    f"ragged generation batch: {decode_rows} tokens for {num_decodes} requests"
                )
                outputs.append(
                    self._forward_verify(
                        hidden_states[num_ctx_tokens:num_tokens],
                        decode_rows // num_decodes,
                        layer_cache,
                        conv_pool,
                        ssm_pool,
                        state_indices[num_prefills:],
                    )
                )

        if output is not None:
            # Core buffer already filled in place; caller applies the epilogue.
            return None
        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        if self._o_allreduce is not None:
            # Head-sharded TP: every rank ran its head shard on the same
            # local batch; sum the row-sharded o_proj partials.
            out = self._o_allreduce(out)
        return out

    def _has_kda_replay_caches(self, layer_cache) -> bool:
        """True when the manager allocated the fused-verify replay caches."""
        return getattr(layer_cache, "kda_qkg_cache", None) is not None

    def _sync_kda_replay_conv_window(
        self, layer_cache, slot_indices, conv_q, conv_k, conv_v
    ) -> None:
        """Seed the replay conv caches' committed window from FLA windows.

        The fused verify kernel keeps its own extended fp32 dim-contiguous
        conv caches; their committed window (columns ``[0, W-1)``) must hold
        the last ``W-1`` raw conv inputs whenever another path (prefill,
        plain decode) advances the base conv pool. The FLA window's oldest
        column drops out of every future convolution, so columns ``[1, W)``
        of the FLA cache map 1:1 onto the committed window.
        """
        if not self._has_kda_replay_caches(layer_cache):
            return
        w = self.mixer.conv_size
        for cache, window in (
            (layer_cache.kda_conv_q, conv_q),
            (layer_cache.kda_conv_k, conv_k),
            (layer_cache.kda_conv_v, conv_v),
        ):
            cache[:, :, : w - 1].index_copy_(0, slot_indices, window[:, :, 1:].to(cache.dtype))

    def _forward_prefill(
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
    ) -> Optional[torch.Tensor]:
        from einops import rearrange

        mixer = self.mixer
        d = self.proj_size
        x = x2d.unsqueeze(0)  # [1, T, hidden]

        onorm_g = None
        if self._qkvg_proj_weight is not None:
            qkvg = torch.nn.functional.linear(x, self._qkvg_proj_weight)
            q_proj_states, k_proj_states, v_proj_states = qkvg[..., : 3 * d].split(d, dim=-1)
            onorm_g = qkvg[..., 3 * d : 4 * d]
        else:
            fused_qkvg = getattr(mixer, "qkvg_proj", None)
            if fused_qkvg is not None:
                qkvg = fused_qkvg(x)
                q_proj_states, k_proj_states, v_proj_states = qkvg[..., : 3 * d].split(d, dim=-1)
                qkvg_split_sizes = getattr(mixer, "qkvg_split_sizes", None)
                if (
                    mixer.use_full_rank_gate
                    and qkvg_split_sizes is not None
                    and len(qkvg_split_sizes) == 4
                ):
                    onorm_g = qkvg[..., 3 * d : 4 * d]
            else:
                q_proj_states = mixer.q_proj(x)
                k_proj_states = mixer.k_proj(x)
                v_proj_states = mixer.v_proj(x)

        # Initial states: present for continuation chunks (chunked prefill)
        # and for prefix-cache hits (block reuse), where the previous
        # conv/recurrent state was onboarded into this request's slot.
        conv_q_in = conv_k_in = conv_v_in = None
        recurrent_in = None
        if mamba_metadata.use_initial_states:
            has_init = mamba_metadata.has_initial_states[:num_prefills]
            cs = conv_pool.index_select(0, slot_indices)
            cs[~has_init] = 0
            conv_q_in, conv_k_in, conv_v_in = _kda_split_conv_sections(cs, d)
            recurrent_in = ssm_pool.index_select(0, slot_indices)
            recurrent_in[~has_init] = 0

        q, conv_q = mixer.q_conv1d(
            q_proj_states, cache=conv_q_in, output_final_state=True, cu_seqlens=cu_seqlens
        )
        k, conv_k = mixer.k_conv1d(
            k_proj_states, cache=conv_k_in, output_final_state=True, cu_seqlens=cu_seqlens
        )
        v, conv_v = mixer.v_conv1d(
            v_proj_states, cache=conv_v_in, output_final_state=True, cu_seqlens=cu_seqlens
        )

        if self._bfa_proj_weight is not None:
            bfa = torch.nn.functional.linear(x, self._bfa_proj_weight)
            f_a = bfa[..., : mixer.head_dim]
            beta = bfa[..., mixer.head_dim : mixer.head_dim + mixer.num_heads].float()
            g = mixer.f_b_proj(f_a)
        else:
            g = mixer.f_b_proj(mixer.f_a_proj(x))
            beta = mixer.b_proj(x).float()
        g = rearrange(g, "... (h d) -> ... h d", d=mixer.head_dim)

        q = rearrange(q, "... (h d) -> ... h d", d=mixer.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=mixer.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=mixer.head_dim)

        # Kernel dispatch (in-tree trtllm::kda_prefill or FLA chunk_kda).
        # Both paths exchange states in the pool's V-first [N, H, V, K]
        # layout, so recurrent_in / final_state map to ssm_pool 1:1.
        lower_bound = mixer.gate_lower_bound
        o, final_state = mixer.prefill_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=mixer.A_log,
            dt_bias=mixer.dt_bias,
            scale=mixer.head_k_dim**-0.5,
            initial_state=recurrent_in,
            safe_gate=lower_bound is not None,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
        )

        # Persist per-request states into the pools.
        conv_pool.index_copy_(
            0, slot_indices, torch.cat([conv_q, conv_k, conv_v], dim=1).to(conv_pool.dtype)
        )
        ssm_pool.index_copy_(0, slot_indices, final_state.to(ssm_pool.dtype))
        # Fused-verify replay caches: seed the committed conv window so the
        # first verify round convolves the correct history (pending drafts
        # are zero for a fresh request, so the tail columns are unused).
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_q, conv_k, conv_v)

        if output is not None:
            # Breakable-CUDA-graph core buffer: write the post-o_norm,
            # pre-o_proj result; o_proj runs on-graph in the caller.
            #
            # Unlike decode — where trtllm::kda_decode fuses the gated
            # RMSNorm and writes the core directly via out= — prefill's
            # trtllm::kda_prefill produces a *pre*-o_norm result and the gated
            # RMSNorm (fla FusedRMSNormGated, no out= buffer) sits between it
            # and the post-o_norm core. So the prefill core write is a copy_
            # at the o_norm boundary. A true kernel-level prefill out= would
            # require a pre-o_norm core (unfusing decode's o_norm) plus an
            # output alias on the CuTe DSL prefill op; deferred. The copy is
            # [num_ctx_tokens, proj_size], negligible next to the chunk kernel.
            og = self._output_gate(x, o, onorm_g)  # [1, T, H, head_dim]
            output.copy_(og.reshape(output.shape))
            return None
        return self._output_gate_and_proj(x, o, onorm_g)

    def _forward_decode(
        self,
        x2d,
        conv_pool,
        ssm_pool,
        slot_indices,
        mamba_metadata=None,
        layer_cache=None,
        ssm_state_indices=None,
        output: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Plain T=1 decode, fast path.

        Calls ``trtllm::kda_decode`` directly with kernel-native layouts
        (nsys 07-24: the reference path spent ~70 us/layer on glue around
        the 5 us kernel — 6 separate in-projection GEMV pairs, per-step
        re-transposition of constant weights, conv-window slice/roll
        copies, per-call torch.arange defaults, and redundant dtype
        casts):

        * one wide fused qkvg GEMV on the main stream, overlapped with the
          fused [f_a | b] GEMV and f_b GEMV on the auxiliary stream for
          CUDA-graph batches up to 128 tokens;
        * conv windows staged with one gather + one repack copy into a
          persistent dense per-section buffer;
        * conv-pool write-back with one cat + one index_copy_;
        * constant tensors (transposed conv weights, fp32 A_log/dt_bias/
          o_norm weight) reused instead of rebuilt per step.

        The conv windows remain gathered batch-row-dense. When stable
        int32 slot indices are supplied, the recurrent-state pool is passed
        directly and the CUDA wrapper selects its indexed-state launch;
        otherwise the state uses the batch-row-dense static layout.
        """
        mixer = self.mixer
        if mixer.decode_kernel_path != "optimized" or mixer.wrong_state_layout:
            ssm_state_indices = None
        if ssm_state_indices is not None:
            logger.info_once(
                "Kimi K3 KDA indexed recurrent-state pool path is active",
                key="kimi_k3_kda_indexed_state_pool",
            )
        else:
            logger.info_once(
                "Kimi K3 KDA static recurrent-state path is active", key="kimi_k3_kda_static_state"
            )
        has_qkvg_projection = (
            self._qkvg_proj_weight is not None or getattr(mixer, "qkvg_proj", None) is not None
        )
        if (
            not has_qkvg_projection
            or self._bfa_proj_weight is None
            or mamba_metadata is None
            or ssm_pool.dtype != torch.float32
        ):
            return self._forward_decode_ref(
                x2d, conv_pool, ssm_pool, slot_indices, layer_cache, ssm_state_indices, output
            )

        d = self.proj_size
        hd = mixer.head_dim
        H = mixer.num_heads
        B = x2d.shape[0]
        W = mixer.conv_size

        # Allocated ONCE at the pool slot count (== per-rank max batch on
        # the Mixed manager; ``slot_indices`` are distinct pool rows and
        # this is the plain one-token-per-request path, so B never exceeds
        # it) and never reallocated: captured CUDA graphs hold this
        # pointer, so a realloc would leave earlier graphs writing into
        # freed memory. Footprint: slots x ~9(H=6)..222(H=96) KB per layer.
        buf = self._cs_dense
        if buf is None:
            if torch.cuda.is_current_stream_capturing():
                # Never allocate inside CUDA graph capture; the reference
                # path is capture-safe (just slower).
                return self._forward_decode_ref(
                    x2d, conv_pool, ssm_pool, slot_indices, layer_cache, ssm_state_indices, output
                )
            buf = torch.empty(
                3, max(conv_pool.shape[0], B), d, W - 1, dtype=torch.bfloat16, device=x2d.device
            )
            self._cs_dense = buf
        else:
            # Fail loudly if the sizing invariant ever breaks: silently
            # reallocating here would hand previously captured CUDA graphs
            # a dangling pointer.
            assert buf.shape[1] >= B, (
                f"KDA decode staging buffer holds {buf.shape[1]} rows but the "
                f"decode batch is {B}; reallocating would corrupt previously "
                f"captured CUDA graphs"
            )

        def _project_qkvg() -> torch.Tensor:
            if self._qkvg_proj_weight is not None:
                return torch.nn.functional.linear(x2d, self._qkvg_proj_weight)
            # FP8 weight read (KIMI_K3_KDA_GLUE_FP8=1) uses the loader's
            # fused FP8 [q | k | v | g] GEMM.
            return mixer.qkvg_proj(x2d)

        def _project_bfa_and_fb() -> tuple[torch.Tensor, torch.Tensor]:
            bfa = torch.nn.functional.linear(x2d, self._bfa_proj_weight)
            f_a = bfa[:, :hd]
            beta = bfa[:, hd : hd + H]
            return beta, mixer.f_b_proj(f_a)

        projection_aux_stream = (
            self._projection_aux_stream if B <= _KDA_BFA_MULTISTREAM_MAX_ROWS else None
        )
        qkvg, (beta, g) = maybe_execute_in_parallel(
            _project_qkvg,
            _project_bfa_and_fb,
            self._projection_fork_event,
            self._projection_join_event,
            projection_aux_stream,
            disable_on_compile=True,
        )
        x_qkv = qkvg[:, : 3 * d]
        onorm_g = qkvg[:, 3 * d : 4 * d]

        # Gather the HF-layout conv windows once, then repack the
        # historical W-1 columns into the kernel's dense per-section
        # [B, d, W-1] layout (single strided copy kernel).
        cs = conv_pool.index_select(0, slot_indices)  # [B, 3d, W]
        cs_dense = buf[:, :B]
        cs_dense.copy_(cs.view(B, 3, d, W)[:, :, :, 1:].permute(1, 0, 2, 3))

        state = (
            ssm_pool if ssm_state_indices is not None else ssm_pool.index_select(0, slot_indices)
        )

        # Breakable-CUDA-graph core buffer: point the kernel's ``out`` at the
        # pre-o_proj core slice so ``trtllm::kda_decode`` writes the post-o_norm
        # result in place (contiguous bf16 ``[B, 1, H, head_dim]``). o_proj runs
        # on-graph in the caller.
        kda_out = output.view(B, 1, H, hd) if output is not None else None

        o = mixer._dispatch.decode_kda(
            x_q=x_qkv[:, :d].unflatten(-1, (H, hd)).unsqueeze(0),
            x_k=x_qkv[:, d : 2 * d].unflatten(-1, (H, hd)).unsqueeze(0),
            x_v=x_qkv[:, 2 * d :].unflatten(-1, (H, hd)).unsqueeze(0),
            w_q_t=self._w_q_t,
            w_k_t=self._w_k_t,
            w_v_t=self._w_v_t,
            bias_q=None,
            bias_k=None,
            bias_v=None,
            cs_q=cs_dense[0],
            cs_k=cs_dense[1],
            cs_v=cs_dense[2],
            A_log=self._A_log_f32,
            g=g.unflatten(-1, (H, hd)).unsqueeze(0),
            dt_bias=self._dt_bias_f32,
            beta=beta.unsqueeze(0),
            state=state,
            onorm_g=onorm_g.unflatten(-1, (H, hd)).unsqueeze(0),
            onorm_weight=self._onorm_w_f32,
            out=kda_out,
            ssm_state_indices=ssm_state_indices,
            cu_seqlens=mamba_metadata._arange_buffer[: B + 1],
            scale=hd**-0.5,
            onorm_eps=mixer.o_norm.eps,
            lower_bound=mixer.gate_lower_bound,
            use_beta_sigmoid_in_kernel=True,
            verbose=False,
            update_conv_cache=False,
        )
        if ssm_state_indices is None:
            ssm_pool.index_copy_(0, slot_indices, state)

        # Roll the HF-layout conv pool by one token: new window =
        # [old columns 1..W-1, x_new]. One cat + one scatter.
        new_win = torch.cat([cs[:, :, 1:], x_qkv.unsqueeze(-1)], dim=-1)
        if new_win.dtype != conv_pool.dtype:
            new_win = new_win.to(conv_pool.dtype)
        conv_pool.index_copy_(0, slot_indices, new_win)
        # Fused-verify replay caches (spec decoding only): keep the
        # committed conv window in sync with the plain-decode advance.
        self._sync_kda_replay_conv_window(
            layer_cache, slot_indices, new_win[:, :d], new_win[:, d : 2 * d], new_win[:, 2 * d :]
        )

        if output is not None:
            # decode_kda already wrote the core in place via out=kda_out.
            return None
        return mixer.o_proj(o.view(B, d))

    def _forward_decode_ref(
        self,
        x2d,
        conv_pool,
        ssm_pool,
        slot_indices,
        layer_cache=None,
        ssm_state_indices=None,
        output: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if output is not None:
            # The reference decode path produces a post-o_proj result and
            # cannot fill the pre-o_proj BCG core buffer. Breakable CUDA
            # graphs therefore require the optimized KDA decode kernel.
            raise NotImplementedError(
                "Breakable CUDA graph KDA requires the optimized decode kernel "
                "(_in_proj_weight is None or ssm_pool is not fp32)"
            )
        from ..modules.kimi_kda.kimi_kda_mixer import KimiKDACachedState

        mixer = self.mixer
        d = self.proj_size
        x = x2d.unsqueeze(1)  # [B, 1, hidden]

        cs = conv_pool.index_select(0, slot_indices)
        conv_q, conv_k, conv_v = _kda_split_conv_sections(cs, d)
        cache = KimiKDACachedState(
            conv_state_q=conv_q,
            conv_state_k=conv_k,
            conv_state_v=conv_v,
            recurrent_state=(
                ssm_pool
                if ssm_state_indices is not None
                else ssm_pool.index_select(0, slot_indices)
            ),
        )
        out, new_cache = mixer.forward_decode(
            x,
            cache,
            ssm_state_indices=ssm_state_indices,
        )

        conv_pool.index_copy_(
            0,
            slot_indices,
            torch.cat(
                [
                    new_cache.conv_state_q,
                    new_cache.conv_state_k,
                    new_cache.conv_state_v,
                ],
                dim=1,
            ).to(conv_pool.dtype),
        )
        if ssm_state_indices is None:
            ssm_pool.index_copy_(0, slot_indices, new_cache.recurrent_state.to(ssm_pool.dtype))
        # Fused-verify replay caches: keep the committed conv window in
        # sync with the plain-decode advance. NOTE: this path is only
        # correct for requests with no pending accepted drafts
        # (prev_num_accepted_tokens == 0); with drafts pending, the live
        # pools lag by the pending prefix and only the fused verify kernel
        # can advance them. The spec workers pad drafts to the static max,
        # so drafted batches always take the verify path.
        self._sync_kda_replay_conv_window(
            layer_cache,
            slot_indices,
            new_cache.conv_state_q,
            new_cache.conv_state_k,
            new_cache.conv_state_v,
        )

        return out.squeeze(1)

    def _forward_verify(
        self, x2d, num_steps, layer_cache, conv_pool, ssm_pool, slot_indices
    ) -> torch.Tensor:
        """Speculative verification: advance each request ``num_steps``
        tokens (1 golden + ``num_steps - 1`` padded drafts).

        Two paths:

        * Fused (``trtllm::kda_mtp_decode``, when the manager allocated the
          KDA replay caches): one kernel launch replays the previous
          round's accepted drafts from the per-slot replay caches, then
          processes the new tokens, committing the recurrent state and conv
          windows **in place** after the golden token and caching the new
          drafts. ``update_mamba_states()`` afterwards only records the
          accepted count for the next round's replay.
        * Legacy (sequential per-step FLA): per-step states go to the
          manager's batch-row-indexed intermediate scratch buffers and
          ``update_mamba_states()`` promotes the accepted step's state
          after sampling.
        """
        if self._has_kda_replay_caches(layer_cache):
            assert self.mixer.verify_kernel_path == "optimized", (
                "KDA replay caches are allocated but the fused verify "
                "kernel is unavailable; the legacy intermediate buffers "
                "were not allocated so there is no fallback"
            )
            return self._forward_verify_fused(x2d, num_steps, layer_cache, ssm_pool, slot_indices)
        return self._forward_verify_sequential(
            x2d, num_steps, layer_cache, conv_pool, ssm_pool, slot_indices
        )

    def _project_verify_inputs(
        self, x: torch.Tensor, num_rows: int
    ) -> Optional[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
        ]
    ]:
        """Project fused QKVG and [f_a | b] inputs for target verification."""
        mixer = self.mixer
        qkvg_weight = self._qkvg_proj_weight
        fused_qkvg = getattr(mixer, "qkvg_proj", None)
        if qkvg_weight is None and fused_qkvg is None:
            return None

        def _project_qkvg() -> torch.Tensor:
            if qkvg_weight is not None:
                return torch.nn.functional.linear(x, qkvg_weight)
            return fused_qkvg(x)

        bfa_weight = self._bfa_proj_weight
        if bfa_weight is not None:

            def _project_bfa_and_fb() -> tuple[torch.Tensor, torch.Tensor]:
                bfa = torch.nn.functional.linear(x, bfa_weight)
                f_a = bfa[..., : mixer.head_dim]
                beta = bfa[..., mixer.head_dim : mixer.head_dim + mixer.num_heads]
                return beta, mixer.f_b_proj(f_a)

            projection_aux_stream = (
                self._projection_aux_stream
                if 0 < num_rows <= _KDA_BFA_MULTISTREAM_MAX_ROWS
                else None
            )
            qkvg, (beta, forget_gate) = maybe_execute_in_parallel(
                _project_qkvg,
                _project_bfa_and_fb,
                self._projection_fork_event,
                self._projection_join_event,
                projection_aux_stream,
                disable_on_compile=True,
            )
        else:
            qkvg = _project_qkvg()
            beta = mixer.b_proj(x)
            forget_gate = mixer.f_b_proj(mixer.f_a_proj(x))

        d = self.proj_size
        q_proj, k_proj, v_proj = (part.contiguous() for part in qkvg[..., : 3 * d].split(d, dim=-1))
        qkvg_split_sizes = getattr(mixer, "qkvg_split_sizes", None)
        has_onorm_gate = qkvg_weight is not None or (
            mixer.use_full_rank_gate and qkvg_split_sizes is not None and len(qkvg_split_sizes) == 4
        )
        onorm_g = qkvg[..., 3 * d : 4 * d].contiguous() if has_onorm_gate else None
        return q_proj, k_proj, v_proj, forget_gate, beta, onorm_g

    def _forward_verify_fused(
        self, x2d, num_steps, layer_cache, ssm_pool, slot_indices
    ) -> torch.Tensor:
        """Fused multi-token verify via ``trtllm::kda_mtp_decode``.

        Token layout: the kernel indexes each request's new tokens at
        ``cu_seqlens[n] + num_accepted[n] + i``. The runtime packs the
        ``num_steps`` new tokens per request contiguously, so we pass
        ``cu_seqlens[n] = n * num_steps - num_accepted[n]`` — the shift
        lands the kernel's reads/writes exactly on the packed rows. A
        negative entry for request 0 is fine: ``bos`` is only ever used
        additively with a token offset ``>= num_accepted``.
        """
        mixer = self.mixer
        num_decodes = x2d.shape[0] // num_steps
        num_spec = num_steps - 1
        H = mixer.num_heads
        K = mixer.head_k_dim
        x = x2d.view(num_decodes, num_steps, -1)  # [B, T, hidden]
        T_total = num_decodes * num_steps

        projections = self._project_verify_inputs(x, T_total)
        if projections is None:
            q_proj = mixer.q_proj(x)
            k_proj = mixer.k_proj(x)
            v_proj = mixer.v_proj(x)
            forget_gate = mixer.f_b_proj(mixer.f_a_proj(x))
            beta_proj = mixer.b_proj(x)
            onorm_g = None
        else:
            q_proj, k_proj, v_proj, forget_gate, beta_proj, onorm_g = projections
        x_q = q_proj.view(1, T_total, H, K)
        x_k = k_proj.view(1, T_total, H, K)
        x_v = v_proj.view(1, T_total, H, mixer.head_dim)
        # Raw gate / beta: the kernel applies dt_bias, A_log, the
        # lower-bound sigmoid gate, and the beta sigmoid itself.
        g = forget_gate.view(1, T_total, H, K)
        beta = beta_proj.contiguous().view(1, T_total, H)

        w_q, w_k, w_v = self._get_mtp_conv_weights()
        lower_bound = (
            mixer.gate_lower_bound_override
            if mixer.gate_lower_bound_override is not None
            else mixer.gate_lower_bound
        )

        pending = layer_cache.prev_num_accepted_tokens[slot_indices].to(
            torch.int32
        )  # accepted drafts of the previous round, per req
        cu_seqlens = torch.arange(
            0, (num_decodes + 1) * num_steps, num_steps, dtype=torch.int32, device=x2d.device
        )
        cu_seqlens[:num_decodes].sub_(pending)

        out = mixer._dispatch.mtp_verify(
            x_q=x_q,
            x_k=x_k,
            x_v=x_v,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            cs_q=layer_cache.kda_conv_q,
            cs_k=layer_cache.kda_conv_k,
            cs_v=layer_cache.kda_conv_v,
            g=g,
            beta=beta,
            # .detach(): the CuTe DSL DLPack bridge rejects grad-tracking
            # tensors.
            A_log=mixer.A_log.detach(),
            dt_bias=mixer.dt_bias.detach(),
            recurrent_state=ssm_pool,
            qkg_cache=layer_cache.kda_qkg_cache,
            v_cache=layer_cache.kda_v_cache,
            beta_cache=layer_cache.kda_beta_cache,
            ssm_state_indices=slot_indices.to(torch.int32),
            cu_seqlens=cu_seqlens,
            num_spec=num_spec,
            num_accepted_tokens=pending,
            lower_bound=lower_bound,
            scale=mixer.head_k_dim**-0.5,
        )
        o = out.view(num_decodes, num_steps, H, mixer.head_dim)
        return self._output_gate_and_proj(x, o, onorm_g)

    def _build_mtp_conv_weights(self) -> None:
        """Prebuild the fp32 ``[dim, W]`` conv weights for the fused verify
        kernel (once, at weight-load finalize time). Building them lazily
        at first use would allocate at runtime; under CUDA graph capture
        that bakes capture-pool pointers into the cached tuple."""
        mixer = self.mixer
        self._mtp_conv_weights = tuple(
            conv.weight.detach().squeeze(1).float().contiguous()
            for conv in (mixer.q_conv1d, mixer.k_conv1d, mixer.v_conv1d)
        )

    def _get_mtp_conv_weights(self) -> Tuple[torch.Tensor, ...]:
        """fp32 ``[dim, W]`` conv weights for the fused verify kernel,
        prebuilt by ``_build_mtp_conv_weights()``."""
        cached = self._mtp_conv_weights
        if cached is None:
            raise RuntimeError(
                "Kimi K3 fused-verify conv weights were not prebuilt; call "
                "_build_mtp_conv_weights() (done by load_weights() and by "
                "finalize_decode_weights() / finalize_decode_weights_fp8()) "
                "after weight load and before the first verify step."
            )
        return cached

    def _forward_verify_sequential(
        self, x2d, num_steps, layer_cache, conv_pool, ssm_pool, slot_indices
    ) -> torch.Tensor:
        """Sequential per-step FLA verification (legacy intermediate-buffer
        path). Live pools are read-only here; ``update_mamba_states()``
        commits the accepted step's state after sampling.
        """
        from einops import rearrange
        from fla.ops.kda import fused_recurrent_kda

        intermediate_conv = layer_cache.intermediate_conv_window
        intermediate_ssm = layer_cache.intermediate_ssm
        assert intermediate_conv is not None and intermediate_ssm is not None, (
            "speculative verification requires the cache manager's "
            "SpeculativeState (legacy intermediate-buffer path)"
        )

        mixer = self.mixer
        d = self.proj_size
        num_decodes = x2d.shape[0] // num_steps
        x = x2d.view(num_decodes, num_steps, -1)  # [B, T, hidden]

        projections = self._project_verify_inputs(x, x2d.shape[0])
        if projections is None:
            q_proj_states = mixer.q_proj(x)
            k_proj_states = mixer.k_proj(x)
            v_proj_states = mixer.v_proj(x)
            g = mixer.f_b_proj(mixer.f_a_proj(x))
            beta = mixer.b_proj(x).float()
            onorm_g = None
        else:
            q_proj_states, k_proj_states, v_proj_states, g, beta, onorm_g = projections
            beta = beta.float()
        g = rearrange(g, "... (h d) -> ... h d", d=mixer.head_dim)

        # Gathered copies — mutated across steps, never written back to the
        # live pools.
        cs = conv_pool.index_select(0, slot_indices)
        conv_q, conv_k, conv_v = _kda_split_conv_sections(cs, d)
        state = ssm_pool.index_select(0, slot_indices)

        step_outputs: List[torch.Tensor] = []
        for t in range(num_steps):
            # ShortConvolution.step updates the (gathered) caches in place.
            q_t, conv_q = mixer.q_conv1d(
                q_proj_states[:, t : t + 1], cache=conv_q, output_final_state=True
            )
            k_t, conv_k = mixer.k_conv1d(
                k_proj_states[:, t : t + 1], cache=conv_k, output_final_state=True
            )
            v_t, conv_v = mixer.v_conv1d(
                v_proj_states[:, t : t + 1], cache=conv_v, output_final_state=True
            )

            q_t = rearrange(q_t, "... (h d) -> ... h d", d=mixer.head_k_dim)
            k_t = rearrange(k_t, "... (h d) -> ... h d", d=mixer.head_k_dim)
            v_t = rearrange(v_t, "... (h d) -> ... h d", d=mixer.head_dim)

            o_t, state = fused_recurrent_kda(
                q=q_t,
                k=k_t,
                v=v_t,
                g=g[:, t : t + 1],
                beta=beta[:, t : t + 1],
                A_log=mixer.A_log,
                dt_bias=mixer.dt_bias,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=mixer.gate_lower_bound,
                state_v_first=True,
            )
            step_outputs.append(o_t)

            # Batch-row indexed ([:num_decodes] prefix), matching
            # update_mamba_states()'s intermediate_state_indices.
            intermediate_conv[:num_decodes, t] = torch.cat([conv_q, conv_k, conv_v], dim=1).to(
                intermediate_conv.dtype
            )
            intermediate_ssm[:num_decodes, t] = state.to(intermediate_ssm.dtype)

        o = torch.cat(step_outputs, dim=1)  # [B, T, H, V]
        return self._output_gate_and_proj(x, o, onorm_g)

    def _output_gate(
        self, x: torch.Tensor, o: torch.Tensor, onorm_g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the gated output RMSNorm, returning the post-o_norm,
        pre-o_proj tensor in ``[..., H, head_dim]`` layout."""
        from einops import rearrange

        mixer = self.mixer
        if onorm_g is not None:
            g_out = onorm_g
        elif mixer.use_full_rank_gate:
            g_out = mixer.g_proj(x)
        else:
            g_out = mixer.g_b_proj(mixer.g_a_proj(x))
        g_out = rearrange(g_out, "... (h d) -> ... h d", d=mixer.head_dim)
        return mixer.o_norm(o, g_out)

    def _output_gate_and_proj(
        self, x: torch.Tensor, o: torch.Tensor, onorm_g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        from einops import rearrange

        o = self._output_gate(x, o, onorm_g)
        o = rearrange(o, "b t h d -> (b t) (h d)")
        return self.mixer.o_proj(o)


class KimiMLARuntime(nn.Module):
    """Wraps K3 MLA and applies its external TP output reduction."""

    def __init__(
        self,
        cfg,
        layer_idx: int,
        model_config: ModelConfig,
    ):
        super().__init__()

        from ..modules.kimi_k3_mla import KimiK3MLAAttention

        max_positions = int(
            os.environ.get(
                _KIMI_K3_MLA_MAX_POSITIONS_ENV,
                cfg.max_position_embeddings,
            )
        )
        self.layer_idx = layer_idx
        # KimiK3MLAAttention owns MLA projection/head sharding. Keep only the
        # final output reduction in this wrapper so the output gate remains
        # between attention and the row-parallel o_proj.
        mapping = model_config.mapping
        reduce_output = not mapping.enable_attention_dp and mapping.tp_size > 1
        self._o_allreduce = (
            AllReduce(
                mapping=mapping,
                strategy=model_config.allreduce_strategy,
                dtype=torch.bfloat16,
            )
            if reduce_output
            else None
        )
        self.mixer = KimiK3MLAAttention(
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
            model_config=model_config,
        )

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        out = self.mixer(hidden_states, attn_metadata)
        if self._o_allreduce is not None:
            # Head-sharded TP: sum the row-sharded o_proj partials across
            # the head-shard group.
            out = self._o_allreduce(out)
        return out


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


class KimiLinearDecoderLayer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        dtype = torch.bfloat16

        self.is_kda = _is_kda_layer(cfg, layer_idx)
        is_mla = _is_mla_layer(cfg, layer_idx)
        if self.is_kda == is_mla:
            raise ValueError(f"Kimi K3 layer {layer_idx} must be exactly one of KDA/MLA")

        if self.is_kda:
            self.self_attn = KimiKDARuntime(
                cfg,
                layer_idx,
                mapping=model_config.mapping,
                allreduce_strategy=model_config.allreduce_strategy,
                aux_stream=aux_stream,
                model_config=model_config,
            )
        else:
            # Forward only the KV-cache quantization to the MLA attention
            # backends (enables FP8 KV cache). The attention projection
            # weights themselves stay BF16 — the model-level weight-quant
            # algo must not leak into the attention backend's weight paths.
            mla_quant_config = None
            kv_quant_algo = (
                model_config.quant_config.kv_cache_quant_algo
                if model_config.quant_config is not None
                else None
            )
            if kv_quant_algo is not None:
                mla_quant_config = QuantConfig(kv_cache_quant_algo=kv_quant_algo)
            mla_model_config = copy.copy(model_config)
            mla_model_config.quant_config = mla_quant_config or QuantConfig()
            self.self_attn = KimiMLARuntime(
                cfg,
                layer_idx,
                model_config=mla_model_config,
            )

        self.is_moe = (
            cfg.num_experts is not None
            and layer_idx >= cfg.first_k_dense_replace
            and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0
        )
        if self.is_moe:
            self.block_sparse_moe = KimiK3MoERuntime(model_config, cfg, layer_idx, aux_stream)
        else:
            situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
            situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
            attention_dp = model_config.mapping.enable_attention_dp
            if attention_dp:
                self.mlp_tp_size = 1
            else:
                self.mlp_tp_size = math.gcd(cfg.intermediate_size, model_config.mapping.tp_size)
                if self.mlp_tp_size > model_config.mapping.gpus_per_node:
                    self.mlp_tp_size = math.gcd(
                        self.mlp_tp_size, model_config.mapping.gpus_per_node
                    )
            mlp_model_config = copy.copy(model_config)
            mlp_model_config.quant_config = QuantConfig()
            # K3's dense layer is BF16, so a unit block size gives the same
            # subgroup selection as DeepSeek-V3. Attention DP replicates the
            # MLP because ranks own different tokens; otherwise the subgroup
            # is block-aligned and stays within one node.
            self.mlp = GatedMLP(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                bias=False,
                activation=SituAndMul(
                    beta=situ_beta,
                    linear_beta=situ_linear_beta,
                    use_fused_activation=True,
                ),
                dtype=dtype,
                config=mlp_model_config,
                overridden_tp_size=self.mlp_tp_size,
                reduce_output=self.mlp_tp_size > 1,
                layer_idx=layer_idx,
            )

        # Stock fused RMSNorm for the plain (whole-tensor) norms; numerics
        # are drop-in for KimiK3RMSNorm (fp32 variance, weight applied
        # after downcast, use_gemma=False).
        self.input_layernorm = RMSNorm(
            hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )

        # Attention residual scheme (always on for K3). The res norms stay
        # KimiK3RMSNorm: they are consumed field-wise (.weight/.eps) by
        # _apply_attn_res and the fused attn_res op, never called as
        # modules.
        self.attn_res_block_size = cfg.attn_res_block_size
        assert self.attn_res_block_size is not None, (
            "Kimi K3 runtime expects attn_res_block_size to be set"
        )
        self.self_attention_res_norm = KimiK3RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.mlp_res_norm = KimiK3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.self_attention_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.mlp_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, int]:
        """Port of HF ``KimiDecoderLayer._forward_attn_residual`` (per token).

        ``block_residual`` is a preallocated snapshot bank in kernel-native
        ``[K_max, M, H]`` layout. Returns the running prefix sum and the
        number of valid bank rows.
        """
        prefix_sum = hidden_states
        valid_block_residual = block_residual[:num_snapshots]

        if num_snapshots > 0:
            hidden_states = _apply_attn_res(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            )

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attn_metadata)

        if prefix_sum is not None:
            prefix_sum = prefix_sum + hidden_states
        else:
            prefix_sum = hidden_states

        hidden_states = _apply_attn_res(
            prefix_sum, valid_block_residual, self.mlp_res_proj, self.mlp_res_norm
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            hidden_states = self.block_sparse_moe(
                hidden_states, getattr(attn_metadata, "all_rank_num_tokens", None)
            )
        else:
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


class KimiLinearModel(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = _get_text_config(model_config.pretrained_config)
        self._text_cfg = cfg
        dtype = torch.bfloat16

        # One side stream shared across all layers. KDA overlaps its small
        # forget-gate projection chain with qkvg during decode and verify;
        # MoE overlaps shared-expert compute and its optional TP reduction
        # with the routed dispatch/expert/combine chain.
        self.aux_stream = torch.cuda.Stream()

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                KimiLinearDecoderLayer(model_config, cfg, layer_idx, self.aux_stream)
                for layer_idx in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)

        # KimiK3RMSNorm (not RMSNorm): consumed field-wise (.weight/.eps)
        # by _apply_attn_res and the fused attn_res op.
        self.output_attn_res_norm = KimiK3RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.output_attn_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.num_attn_res_snapshots = (
            cfg.num_hidden_layers + cfg.attn_res_block_size - 1
        ) // cfg.attn_res_block_size

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
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # Graph-disabled warmups still inherit the runner's padded token
        # bucket. Eager KDA/MLA attention returns only real tokens, so trim the
        # input too and keep the attention-residual stack at the same shape.
        # BCG and generation CUDA graphs retain their static padded tensors:
        # their attention cores consume only the ``num_tokens`` real prefix and
        # the causal-LM wrapper slices logits back before sampling.
        if attn_metadata.padded_num_tokens is not None and not (
            attn_metadata.is_cuda_graph or is_in_breakable_cuda_graph()
        ):
            hidden_states = hidden_states[: attn_metadata.num_tokens]

        block_residual = hidden_states.new_empty(
            self.num_attn_res_snapshots,
            hidden_states.shape[0],
            hidden_states.shape[1],
        )
        num_snapshots = 0
        for layer in self.layers:
            hidden_states, num_snapshots = layer(
                hidden_states, block_residual, num_snapshots, attn_metadata
            )
            if spec_metadata is not None:
                # DFlash hidden-state capture. K3's attn-residual scheme
                # already folds the residual into the running prefix sum
                # returned by each layer, so unlike Qwen3/Llama we pass the
                # full hidden state with residual=None. Whether the drafter
                # is trained against this prefix sum or some other tap point
                # must be confirmed against the K3 drafter training recipe
                # before real weights are used.
                spec_metadata.maybe_capture_hidden_states(layer.layer_idx, hidden_states, None)

        hidden_states = _apply_attn_res(
            hidden_states,
            block_residual[:num_snapshots],
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        )
        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# Causal LM wrapper + weight loading.
# ---------------------------------------------------------------------------


def _materialize(value) -> torch.Tensor:
    """Materialize a (possibly lazy safetensors slice) weight value."""
    if isinstance(value, torch.Tensor):
        return value
    return value[:]


@register_auto_model("KimiLinearForCausalLM")
class KimiLinearForCausalLM(SpecDecOneEngineForCausalLM[KimiLinearModel, Any]):
    """Kimi K3 text core (KDA + MLA + MoE).

    Serves text-only ``kimi_linear`` checkpoints directly, and is reused as the
    text backbone by the multimodal ``KimiK3ForConditionalGeneration`` wrapper
    (``modeling_kimi_k3_vl``). The composite ``KimiK3ForConditionalGeneration``
    architecture is registered by that wrapper, not here."""

    def __init__(self, model_config: ModelConfig):
        cfg = _get_text_config(model_config.pretrained_config)
        assert model_config.mapping.pp_size == 1, "Kimi K3 does not support pipeline parallelism"
        spec_config = getattr(model_config, "spec_config", None)
        # Supported spec-dec modes:
        # - SA (suffix automaton): one-engine in-forward drafting, no draft
        #   weights; the KDA/MLA verify paths below implement multi-token
        #   verification for it.
        # - DFlash: external-drafter parallel drafting; the drafter is a
        #   separate dense checkpoint (K2.7-Code-DFlash schema) consumed by
        #   the generic DFlashForCausalLM wrapper, and the target only has
        #   to expose per-layer hidden states via maybe_capture_hidden_states
        #   (see KimiLinearModel.forward). No trained K3 drafter exists yet;
        #   this path is exercised with synthetic weights
        #   (examples/kimi_k3/make_synthetic_dflash_drafter.py).
        # Modes needing draft heads (MTP/Eagle) are blocked until a
        # draft-head checkpoint exists.
        assert (
            spec_config is None
            or spec_config.spec_dec_mode.is_sa()
            or spec_config.spec_dec_mode.is_dflash()
        ), "Kimi K3 supports speculative decoding only with SA or DFlash"
        super().__init__(
            KimiLinearModel(model_config),
            model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        # - enable_block_reuse defaults off: reuse is supported as an
        #   explicit opt-in (routes to CppMambaHybridCacheManager with
        #   per-block KDA state snapshots); the default stays on the
        #   Mixed manager, which SA speculative decoding requires.
        # - tokens_per_block=64: with 32, the flashinfer trtllm-gen FMHA lib
        #   rejects the MLA (576, 512) generation kernel (marked slower) and
        #   the fallback C++ path requires num_heads % 64 == 0, which K3's
        #   96 query heads violate.
        return {
            "kv_cache_config": {
                "enable_block_reuse": False,
                "tokens_per_block": 64,
            }
        }

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

    # ------------------------------------------------------------------
    # Weight loading (streams the 1.5TB checkpoint; only the rank-local
    # expert slice of each MoE layer is kept: whole experts under MoE EP,
    # the intra-expert intermediate shard of ALL experts under MoE TP —
    # in the TP case every expert tensor is read and sliced, so expect a
    # correspondingly longer load).
    # ------------------------------------------------------------------

    def _trunk_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Named parameters of the trunk only. Spec-dec draft modules
        (e.g. the DFlash drafter attached by SpecDecOneEngineForCausalLM)
        live in a separate checkpoint loaded by
        ModelLoader.load_draft_weights, not in the target checkpoint. MLA
        K/V absorb Parameters are derived by the KV-B loader and are likewise
        excluded from checkpoint jobs."""
        return {
            name: param
            for name, param in self.named_parameters()
            if not name.startswith("draft_model.")
            and not name.endswith(_KIMI_K3_MLA_DERIVED_PARAM_SUFFIXES)
        }

    def checkpoint_name_plan(
        self, prefix: str
    ) -> Tuple[Dict[str, str], Set[str], List[Tuple[int, KimiK3MoERuntime, str]]]:
        """Return ``(name_map, expected_keys, expert_jobs)``.

        ``name_map`` maps every model parameter name to its checkpoint key
        (for fused ``gate_up_proj`` parameters the mapped key is virtual;
        the two real per-half keys come from ``_gate_up_ckpt_keys``);
        ``expected_keys`` additionally covers the rank-local per-expert MXFP4
        tensors; ``expert_jobs`` lists ``(layer_idx, moe_module, key_base)``
        for backend-owned expert slots. Exposed separately so the weight-name
        mapping can be dry-run without touching any tensor data.
        """
        params = self._trunk_parameters()
        expected_keys = set()
        name_map: Dict[str, str] = {}
        for name in params:
            # ConfigurableMoE's backend owns already-packed runtime weights,
            # generated zero biases, and SiTU constants. They do not have
            # one-to-one checkpoint parameter names.
            if ".routed_experts.backend." in name:
                continue
            if name == "lm_head.weight":
                ckpt_key = prefix + "lm_head.weight"
            else:
                # Runtime wrapper modules hold the parity-tested mixers as a
                # "mixer" submodule; the checkpoint names have no such scope.
                ckpt_key = prefix + name.replace(".self_attn.mixer.", ".self_attn.")
            name_map[name] = ckpt_key
            if name.endswith(_GATE_UP_FUSED_SUFFIX):
                # Fused [gate | up] MLP layout (dense mlp / shared_experts):
                # the checkpoint stores two separate tensors.
                expected_keys.update(_gate_up_ckpt_keys(ckpt_key))
            else:
                expected_keys.add(ckpt_key)

        # Backend-owned expert slots (per-expert checkpoint tensors; the
        # rank-local id range — an EP slice of whole experts, or ALL experts
        # when the routed MoE is TP-sharded (moe_ep=1 -> ids 0..num_experts)).
        expert_jobs = []
        for layer_idx, layer in enumerate(self.model.layers):
            if not getattr(layer, "is_moe", False) or not _has_weights(layer):
                continue
            moe = layer.block_sparse_moe
            base = f"{prefix}model.layers.{layer_idx}.block_sparse_moe.experts"
            for expert_idx in moe.local_expert_ids:
                for w in ("w1", "w2", "w3"):
                    expected_keys.add(f"{base}.{expert_idx}.{w}.weight_packed")
                    expected_keys.add(f"{base}.{expert_idx}.{w}.weight_scale")
            expert_jobs.append((layer_idx, moe, base))
        return name_map, expected_keys, expert_jobs

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        prefix = "language_model." if any(k.startswith("language_model.") for k in weights) else ""
        params = self._trunk_parameters()
        name_map, expected_keys, expert_jobs = self.checkpoint_name_plan(prefix)

        self._validate_checkpoint_keys(weights, expected_keys, prefix)
        num_params = self._load_trunk_params(weights, params, name_map)
        self._load_expert_slices(weights, expert_jobs)
        self._finalize_weight_load(num_params, len(expert_jobs))

    def _validate_checkpoint_keys(
        self, weights: Dict[str, torch.Tensor], expected_keys: Set[str], prefix: str
    ) -> None:
        """Key-set validation (both directions): every expected key must be
        present; unmatched checkpoint keys (beyond the expected leftovers)
        only warn."""
        ckpt_keys = set(weights.keys())
        relevant_ckpt_keys = {
            k
            for k in ckpt_keys
            if not (k.startswith("vision_tower.") or k.startswith("mm_projector."))
        }
        missing = sorted(expected_keys - ckpt_keys)
        if missing:
            raise KeyError(
                f"Kimi K3 load_weights: {len(missing)} expected checkpoint "
                f"keys are missing, e.g. {missing[:10]}"
            )
        unexpected = relevant_ckpt_keys - expected_keys
        # Non-local experts and (in layer-truncated debug mode) extra layers
        # are expected leftovers.
        surprising = sorted(
            k
            for k in unexpected
            if ".block_sparse_moe.experts." not in k and not k.startswith(f"{prefix}model.layers.")
        )
        if surprising:
            logger.warning(
                f"Kimi K3 load_weights: {len(surprising)} unmatched "
                f"checkpoint keys, e.g. {surprising[:10]}"
            )

    def _load_trunk_params(
        self,
        weights: Dict[str, torch.Tensor],
        params: Dict[str, torch.nn.Parameter],
        name_map: Dict[str, str],
    ) -> int:
        """Load every non-expert trunk parameter concurrently (with the
        per-parameter TP-shard / pad / fuse conversions) and return the
        number of parameters loaded."""
        # The checkpoint stores every MLA KV-B head as interleaved [K | V]
        # rows. Runtime keeps one DeepSeek-style [all K | all V] parameter
        # instead, so context can project directly into the FMHA layout and
        # absorbed decode can take zero-copy K/V views.
        mla_mixers = [
            layer.self_attn.mixer
            for layer in self.model.layers
            if not getattr(layer, "is_kda", True) and _has_weights(layer)
        ]
        mla_kv_b_mixers = {id(mixer.kv_b_proj.weight): mixer for mixer in mla_mixers}
        mla_head_shard_linears = {}
        for mixer in mla_mixers:
            mla_head_shard_linears[id(mixer.q_b_proj.weight)] = mixer.q_b_proj
            mla_head_shard_linears[id(mixer.o_proj.weight)] = mixer.o_proj
            g_proj = getattr(mixer, "g_proj", None)
            if g_proj is not None:
                mla_head_shard_linears[id(g_proj.weight)] = g_proj

        device = next(self.parameters()).device

        # MLP TP shard index. A dense MLP whose intermediate size does not
        # divide model TP uses a smaller repeated TP subgroup, so its local
        # shard rank is model tp_rank modulo the parameter's shard count.
        model_tp_rank = self.model_config.mapping.tp_rank
        # KDA head-shard (attention-DP off): rank r loads head rows/cols
        # [r*local : (r+1)*local] of every head-major KDA tensor.
        kda_tp_size, kda_tp_rank = 1, 0
        for layer in self.model.layers:
            if getattr(layer, "is_kda", False):
                kda_tp_size = layer.self_attn._kda_tp_size
                kda_tp_rank = layer.self_attn._kda_tp_rank
                break

        def load_param(name: str, param: torch.nn.Parameter):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            if name.endswith(_GATE_UP_FUSED_SUFFIX):
                # Row-concat the checkpoint's separate gate_proj / up_proj
                # tensors into the fused [gate | up] parameter.
                gate_key, up_key = _gate_up_ckpt_keys(name_map[name])
                gate = _materialize(weights[gate_key])
                up = _materialize(weights[up_key])
                inter = param.shape[0] // 2
                if gate.shape[0] != inter and gate.shape[0] % inter == 0:
                    # TP-sharded fused MLP (shared experts on the direct
                    # MoE path, dense MLP with attention-DP off): take this
                    # subgroup rank's matching row block from each half so
                    # the SiTU gate/up pairs stay aligned.
                    shard_count = gate.shape[0] // inter
                    lo = (model_tp_rank % shard_count) * inter
                    gate = gate[lo : lo + inter]
                    up = up[lo : lo + inter]
                if gate.shape != (inter, param.shape[1]) or up.shape != gate.shape:
                    raise ValueError(
                        f"{name}: checkpoint gate/up shapes "
                        f"{tuple(gate.shape)} / {tuple(up.shape)} do not "
                        f"concat to param shape {tuple(param.shape)}"
                    )
                param.data[:inter].copy_(gate.to(param.dtype))
                param.data[inter:].copy_(up.to(param.dtype))
                return
            src = _materialize(weights[name_map[name]])
            if name == "lm_head.weight":
                # LMHead is vocab-sharded (TP column) + gathered; its
                # load_weights shards the full checkpoint tensor.
                self.lm_head.load_weights(weights=[{"weight": src}])
                return
            mla_mixer = mla_kv_b_mixers.get(id(param))
            if mla_mixer is not None:
                h = mla_mixer.num_heads_tp
                n = mla_mixer.qk_nope_head_dim
                v = mla_mixer.v_head_dim
                kv = mla_mixer.kv_lora_rank
                local = mla_mixer.kv_b_proj.load_shard(src, device=param.device).view(h, n + v, kv)
                k_weight, v_weight = local.split([n, v], dim=1)
                param.data.copy_(
                    torch.cat(
                        [
                            k_weight.reshape(h * n, kv),
                            v_weight.reshape(h * v, kv),
                        ],
                        dim=0,
                    ).to(param.dtype)
                )
                mla_mixer.k_b_proj_trans.data.copy_(k_weight.transpose(1, 2))
                mla_mixer.v_b_proj.data.copy_(v_weight)
                return
            if name.endswith(".A_log") and src.numel() != param.numel():
                # The checkpoint pads A_log from [num_heads] to [head_dim]
                # (e.g. [96] -> [128]); the tail must be zeros. Under KDA
                # head-shard TP the param holds this rank's head range
                # instead of the full [num_heads].
                assert src.numel() > param.numel(), (name, src.shape)
                if kda_tp_size > 1:
                    lo = kda_tp_rank * param.numel()
                    src = src[lo : lo + param.numel()]
                else:
                    tail = src[param.numel() :]
                    if tail.abs().max().item() != 0.0:
                        raise ValueError(
                            f"{name}: expected zero padding beyond "
                            f"{param.numel()} entries, got nonzero tail"
                        )
                    src = src[: param.numel()]
            if src.shape != param.shape:
                # Delegate MLA q_b/g/o slicing to the same Linear modules that
                # own their COLUMN/ROW sharding policy. KV-B is handled above.
                mla_sharded_linear = mla_head_shard_linears.get(id(param))
                if mla_sharded_linear is not None:
                    shard = mla_sharded_linear.load_shard(src, device=param.device)
                    if shard.shape != param.shape:
                        raise ValueError(
                            f"{name}: MLA shard shape {tuple(shard.shape)} does "
                            f"not match param shape {tuple(param.shape)}"
                        )
                    param.data.copy_(shard.to(param.dtype))
                    return
                # KDA head-shard (attention-DP off): every mismatching KDA
                # tensor is head-major with the checkpoint exactly
                # kda_tp_size times larger on one axis — q/k/v/g/f_b
                # projections, b_proj, dt_bias, and the depthwise conv
                # weights on dim 0 (rows), o_proj on dim 1 (columns).
                # MLA head-sharded projections were handled by parameter
                # identity above, so shape ratios identify the KDA slices.
                if kda_tp_size > 1 and ".self_attn." in name:
                    if (
                        src.shape[0] == param.shape[0] * kda_tp_size
                        and src.shape[1:] == param.shape[1:]
                    ):
                        s = param.shape[0]
                        lo = kda_tp_rank * s
                        param.data.copy_(src[lo : lo + s].to(param.dtype))
                        return
                    if (
                        src.dim() == 2
                        and src.shape[0] == param.shape[0]
                        and src.shape[1] == param.shape[1] * kda_tp_size
                    ):
                        s = param.shape[1]
                        lo = kda_tp_rank * s
                        param.data.copy_(src[:, lo : lo + s].to(param.dtype))
                        return
                # Shared-expert TP (direct MoE path): the module holds a
                # 1/tp shard of the FFN dim — column shard for gate/up
                # (output rows), row shard for down (input columns).
                if ".shared_experts." in name or ".mlp." in name:
                    # Shared experts (direct MoE path) and the dense L0
                    # MLP (attention-DP off): the fused gate_up_proj is
                    # sliced in its dedicated branch above; here the
                    # unfused halves (if ever configured) and down_proj.
                    if (
                        name.endswith((".gate_proj.weight", ".up_proj.weight"))
                        and src.shape[0] % param.shape[0] == 0
                        and src.shape[1:] == param.shape[1:]
                    ):
                        shard_count = src.shape[0] // param.shape[0]
                        lo = (model_tp_rank % shard_count) * param.shape[0]
                        param.data.copy_(src[lo : lo + param.shape[0]].to(param.dtype))
                        return
                    if (
                        name.endswith(".down_proj.weight")
                        and src.shape[1] % param.shape[1] == 0
                        and src.shape[0] == param.shape[0]
                    ):
                        shard_count = src.shape[1] // param.shape[1]
                        lo = (model_tp_rank % shard_count) * param.shape[1]
                        param.data.copy_(src[:, lo : lo + param.shape[1]].to(param.dtype))
                        return
                raise ValueError(
                    f"{name}: checkpoint shape "
                    f"{tuple(src.shape)} != param shape "
                    f"{tuple(param.shape)}"
                )
            param.data.copy_(src.to(param.dtype))

        param_jobs = [(name, params[name]) for name in name_map]
        run_concurrently(load_param, param_jobs, num_workers=8)

        logger.info(
            f"Kimi K3: loaded {len(mla_mixers)} MLA KV-B projections in grouped runtime layout"
        )
        return len(param_jobs)

    def _load_expert_slices(
        self,
        weights: Dict[str, torch.Tensor],
        expert_jobs: List[Tuple[int, KimiK3MoERuntime, str]],
    ) -> None:
        """Load the rank-local MXFP4 expert slices of every MoE layer into
        the backend expert slots, then verify every slot was filled."""
        device = next(self.parameters()).device

        def load_expert(
            moe: KimiK3MoERuntime, base: str, local_slot_id: int, expert_idx: int, get_tensor
        ):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            backend = moe.routed_experts.backend
            backend.quant_method.load_packed_mxfp4_expert(
                backend,
                global_expert_id=expert_idx,
                local_slot_id=local_slot_id,
                w1_weight=get_tensor(f"{base}.{expert_idx}.w1.weight_packed"),
                w1_weight_scale=get_tensor(f"{base}.{expert_idx}.w1.weight_scale"),
                w2_weight=get_tensor(f"{base}.{expert_idx}.w2.weight_packed"),
                w2_weight_scale=get_tensor(f"{base}.{expert_idx}.w2.weight_scale"),
                w3_weight=get_tensor(f"{base}.{expert_idx}.w3.weight_packed"),
                w3_weight_scale=get_tensor(f"{base}.{expert_idx}.w3.weight_scale"),
            )

        def load_experts_from_weights(layer_idx: int, moe: KimiK3MoERuntime, base: str):
            del layer_idx
            for local_slot_id, expert_idx in enumerate(moe.local_expert_ids):
                load_expert(
                    moe,
                    base,
                    local_slot_id,
                    expert_idx,
                    lambda key: _materialize(weights[key]),
                )

        # ---- backend expert slots: file-grouped streaming ----
        # The shared lazy ``weights`` dict keeps every shard mmapped for the
        # whole load, so pages it touches cannot be dropped until the load
        # ends (fadvise skips mapped pages). The expert slices are ~90 GB of
        # DISTINCT pages per rank — with 4 ranks/node that overruns the job
        # cgroup and OOM-kills the step (observed repeatedly on GB300
        # trays). Instead, group the rank-local expert tensors by shard file
        # and stream each file through a short-lived handle:
        # open -> copy -> close (unmap) -> fadvise(DONTNEED).
        ckpt_dir = getattr(self.model_config.pretrained_config, "_name_or_path", None)
        index_path = os.path.join(ckpt_dir or "", "model.safetensors.index.json")
        if expert_jobs and ckpt_dir and os.path.isfile(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            per_file: Dict[str, list] = {}
            split_file_jobs = []
            for layer_idx, moe, base in expert_jobs:
                del layer_idx
                for local_slot_id, expert_idx in enumerate(moe.local_expert_ids):
                    keys = [
                        f"{base}.{expert_idx}.{w}.{kind}"
                        for w in ("w1", "w2", "w3")
                        for kind in ("weight_packed", "weight_scale")
                    ]
                    files = {weight_map[key] for key in keys}
                    job = (moe, base, local_slot_id, expert_idx)
                    if len(files) == 1:
                        per_file.setdefault(files.pop(), []).append(job)
                    else:
                        split_file_jobs.append((job, files))

            def drop_file_pages(file_name: str):
                path = os.path.join(ckpt_dir, file_name)
                try:
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                except OSError:
                    pass

            def load_expert_file(file_name: str, jobs: list):
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                path = os.path.join(ckpt_dir, file_name)
                with safe_open(path, framework="pt", device="cpu") as fh:
                    for moe, base, local_slot_id, expert_idx in jobs:
                        load_expert(moe, base, local_slot_id, expert_idx, fh.get_tensor)
                # Handle closed -> pages unmapped -> the drop takes effect.
                drop_file_pages(file_name)

            def load_split_file_expert(job, files):
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                with ExitStack() as stack:
                    handles = {
                        file_name: stack.enter_context(
                            safe_open(
                                os.path.join(ckpt_dir, file_name), framework="pt", device="cpu"
                            )
                        )
                        for file_name in files
                    }

                    def get_tensor(key):
                        return handles[weight_map[key]].get_tensor(key)

                    load_expert(*job, get_tensor)
                for file_name in files:
                    drop_file_pages(file_name)

            run_concurrently(load_expert_file, sorted(per_file.items()), num_workers=4)
            run_concurrently(load_split_file_expert, split_file_jobs, num_workers=4)
        else:
            run_concurrently(load_experts_from_weights, expert_jobs, num_workers=4)

        for _, moe, _ in expert_jobs:
            backend = moe.routed_experts.backend
            loaded_slots = getattr(backend, "_packed_mxfp4_loaded_slots", set())
            expected_slots = set(range(backend.expert_size_per_partition))
            if loaded_slots != expected_slots:
                missing_slots = sorted(expected_slots - loaded_slots)
                raise RuntimeError(
                    "Kimi K3 packed expert loading did not fill all backend "
                    f"slots; missing {missing_slots[:10]}."
                )
            backend._weights_transformed = False

    def _finalize_weight_load(self, num_params: int, num_moe_layers: int) -> None:
        """Post-load finalization: build the KDA fused projection constants
        and apply the FP8 weight-read conversions (all behind their env
        switches)."""
        # FP8 weight-read master switch (see the conversion block below).
        # The KDA conversion replaces the decode in-projection GEMV with a
        # fused FP8 qkvg GEMM in the mixer decode path, so when it is enabled
        # the bf16 wrapper fast path (finalize_decode_weights) is NOT built:
        # both fuse the same projections and the wrapper path — checked first
        # at decode — would bypass the FP8 modules entirely, leaving the FP8
        # copies resident but inert. KIMI_K3_FP8_WEIGHT_READ_KDA=0 restores
        # the bf16 wrapper fast path; KIMI_K3_KDA_GLUE_FP8=1 instead rebuilds
        # the wrapper fast path on top of the FP8 modules after the
        # conversion (finalize_decode_weights_fp8), so neither is traded away.
        fp8_weight_read, kda_fp8, kda_glue_fp8 = _resolve_fp8_weight_read_gates()

        # Build the KDA fused projection views and decode kernel constants.
        # This must run after every KDA parameter is loaded and sharded.
        num_kda_fused = 0
        for layer in self.model.layers:
            if getattr(layer, "is_kda", False) and _has_weights(layer):
                if not kda_fp8:
                    layer.self_attn.finalize_decode_weights()
                num_kda_fused += int(
                    layer.self_attn._qkvg_proj_weight is not None
                    and layer.self_attn._bfa_proj_weight is not None
                )
                # The fused-verify conv constants are needed on every
                # configuration that can reach _forward_verify_fused,
                # including ones where neither finalize variant runs (e.g.
                # FP8 KDA weight read with the fused decode glue disabled),
                # and are never computed lazily (a first verify under CUDA
                # graph capture must not allocate). Build them
                # unconditionally; three small fp32 tensors per layer.
                layer.self_attn._build_mtp_conv_weights()
        logger.info(
            f"Kimi K3: loaded {num_params} parameters and the expert "
            f"slices of {num_moe_layers} MoE layers; fused prefill/decode/verify "
            f"projections on {num_kda_fused} KDA layers"
        )

        # FP8 block-scale weight read for the replicated MoE-layer MLPs. The
        # DeepGEMM fp8_swap_ab_gemm kernel is Blackwell-only; keep BF16 on any
        # other SM or when explicitly disabled.
        if fp8_weight_read:
            gate_up_default = "1" if self.model_config.mapping.enable_attention_dp else "0"
            n_fp8 = _convert_moe_mlps_to_fp8_weight_read(
                self.model,
                include_fused_gate_up=os.environ.get(
                    _KIMI_K3_FP8_WEIGHT_READ_GATE_UP_ENV, gate_up_default
                )
                != "0",
            )
            logger.info(
                f"Kimi K3: reading {n_fp8} MoE-layer MLP projections "
                f"(shared-expert + latent) at FP8 block-scale"
            )
            # The KDA q/k/v/g/o projections are the largest single replicated
            # weight read; convert them to the same FP8 block-scale read unless
            # kept in BF16 for accuracy (their own switch — the recurrent core
            # is the most precision-sensitive slice).
            if os.environ.get(_KIMI_K3_FP8_WEIGHT_READ_KDA_ENV, "1") != "0":
                n_kda = _convert_kda_projections_to_fp8_weight_read(self.model)
                logger.info(
                    f"Kimi K3: reading {n_kda} KDA q/k/v/g/o projections "
                    f"at FP8 block-scale (q/k/v/g fused into one prefill/decode/verify GEMM "
                    f"per layer)"
                )
                if kda_glue_fp8:
                    # Rebuild the fused projection path on top of the FP8
                    # modules. This must run after the conversion above so
                    # fused FP8 qkvg_proj exists and only [f_a | b] is fused
                    # in BF16.
                    n_glue = 0
                    for layer in self.model.layers:
                        if getattr(layer, "is_kda", False) and _has_weights(layer):
                            layer.self_attn.finalize_decode_weights_fp8()
                            n_glue += int(layer.self_attn._bfa_proj_weight is not None)
                    logger.info(
                        f"Kimi K3: FP8 fused prefill/decode/verify projections on "
                        f"{n_glue} KDA layers"
                    )
            # The MLA q_a/q_b/o and output-gate projections are the remaining
            # replicated attention weight read the MLP and KDA passes above
            # leave in BF16; convert them to the same FP8 block-scale read
            # (kv_a/kv_b stay BF16 — see the switch's comment) unless kept in
            # BF16 for accuracy.
            if os.environ.get(_KIMI_K3_FP8_WEIGHT_READ_MLA_ENV, "1") != "0":
                n_mla = _convert_mla_projections_to_fp8_weight_read(self.model)
                logger.info(
                    f"Kimi K3: reading {n_mla} MLA q_a/q_b/o/g projections at FP8 block-scale"
                )
