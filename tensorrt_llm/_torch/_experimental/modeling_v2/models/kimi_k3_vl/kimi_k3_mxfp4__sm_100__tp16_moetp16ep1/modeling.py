# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# >>> route B: this target's layout, and decoding without speculation
"""ModelingV2 target: Kimi K3 (MXFP4) / sm_100 / tp16 attention, routed experts moe_tp 16 x moe_ep 1.

Kimi K3's language model: 93 layers, hidden 7168. Every fourth layer (3, 7, ..., 91) is MLA attention with 96
query heads; the others are Kimi Delta Attention (KDA), a gated linear-attention recurrence with per-request state.
Layer 0's MLP is dense. Every other layer's MLP is a latent MoE: 896 routed experts, top-16, expert width 3584.
Attention residuals mix each sublayer's input from a bank of earlier outputs. The routed experts are MXFP4 (the
checkpoint's compressed-tensors default, read as W4A8_MXFP4_MXFP8); everything else is bf16, and so is the KV pool.

`tp16_moetp16ep1` is `tensor_parallel_size: 16` with `moe_tensor_parallel_size: 16` and
`moe_expert_parallel_size: 1`, both set explicitly, no attention data parallelism, all 16 GPUs in one NVLink domain.
Attention is head-split (6 MLA query heads per rank), and every rank holds all 896 experts at a sixteenth of their
intermediate width (192 of 3072 values, zero-padded to 256 when they load). With the expert split left unset the
engine resolves the same sizes, but Kimi K3 then runs the experts expert-parallel over the 16 ranks, a layout no
target serves. This is Kimi K3's layout for decoding without speculation; `tp16_moetp4ep4` serves speculative
decoding.

**This module is `tp16_moetp4ep4`'s, copied** (targets share no files) and changed only inside its marked route B
blocks; test_modeling_v2_kimi_k3_drift.py checks that the two modules match everywhere else.

**Each step is classified once, on the host, from its shape** (`decode_step`, a `DecodeStep` or None), and the
classification decides which kernels each module runs:

* **small**: at most 8 tokens (`DECODE_MAX_TOKENS`, one token tile), context requests included. The token-count
  kernels take it: decode GEMVs, the sandwiches, the embedding and residual epilogues.
* **decode**: a pure decode step of R <= 8 generation requests and no context request; without speculation each
  request has one token. The request-aware kernels take it: MLA attention and its KV store.

Every other step (prefill, mixed steps, decode steps above those bounds) runs the **generic path**: this target's
text model (`KimiLinearModel` below: decoder layers, attention residuals, the MLA / KDA / MoE runtimes), computed
exactly as the built-in Kimi K3 text model computes it, on stock modules and ops that have no catalog entries yet.
`UNCERTIFIED_GENERIC_CALLS` names them.

The text model hands each step's classification to its attention modules, which run a **decode step** on the K3
decode kernels' catalog entries:

* KDA (`K3DecodeKDA`): one token per request, the fused input projection and the plain decode in one
  `ssm/k3_kda_decode_attn` launch.
* MLA (`K3DecodeMLA`): `attention/k3_mla_qkv` (the query path and the step's latent KV rows into the paged cache),
  then `attention/k3_mla_attn_vb_out` (the attention, v_b and the output gate in one launch).

The projections around them run on the decode GEMV sites of `decode_gemv.py` (the [W_a; W_g] projection, `o_proj`
on every classified step), as do the LM head, the embedding and layer 0's dense MLP. A classified step's
attention-residual epilogues (the selection and the RMSNorm after it) take the fused kernels up to one token tile. On
any step of at most 16 tokens, the post-attention all-reduce carries the residual update (`decode_comm.py`): at most 8
tokens on an attention decode branch, o_proj, the all-reduce and the update are one `comm/k3_sandwich_oproj` kernel;
otherwise the attention's unreduced o_proj output goes through `comm/mnnvl_allreduce_attn_res`. The stock MNNVL
all-reduces send one-shot up to 4 MiB. The state those kernels share (the KDA projection's Lamport buffers, the MLA
attention workspace, the decode GEMVs' state, the TP group's MNNVL and sandwich workspaces) lives in typed objects this
target creates in `post_load_weights`, before any graph capture.

The MoE layers run the generic path on every step. `decode_moe.py` is `tp16_moetp4ep4`'s MoE decode path, which this
target does not build (its wide `k3_moe` build does not fit 896 local experts); this layout's own MoE engines
(`moe/k3_moe_m1` and `moe/k3_moe_m2` at one and two tokens, `moe/k3_moe` over all 896 experts up to 8) come with their
wiring as route B blocks.

**What this target asserts rather than adapts**: SM 10.0; the topology above, with the expert split set explicitly;
no speculative decoding; the MXFP4 checkpoint's quantization (W4A16_MXFP4 with no per-layer declarations, so the
routed experts run the W4A8_MXFP4_MXFP8 default and the excluded modules stay bf16); bf16 weights and a bf16 KV pool;
tokens_per_block 64 (the MLA generation kernels K3's 96 heads reach exist only at 64); the V2 hybrid KV / state
manager, which holds the KDA states, with block reuse off and fp32 recurrent states; an all-reduce strategy of AUTO or
MNNVL. The construction-time ones fail in `__init__`, the
per-engine ones on the first forward, each naming the setting. A layer the decode kernels do not take fails the
weight load.

**Text only.** The checkpoint is the vision-language wrapper. This target builds and loads no vision tower (its
weights are a predicted non-load, `weights.py`), and a step carrying multimodal input raises.

**No speculative decoding.** A speculative decoding config fails construction, naming `tp16_moetp4ep4`'s expert
split. The causal LM is still the stock one-engine shell the built-in model builds on, which without that config
builds no drafter and no worker. This target drops `tp16_moetp4ep4`'s KDA verify kernels and the per-token verify
states they keep; the text model's speculative hidden-state taps stay as `tp16_moetp4ep4` has them and never fire.
The DSpark drafter (`K3DSparkDrafter`, which `_build_draft_model` builds) also stays as `tp16_moetp4ep4` has it
and is never built here; its decode GEMV sites are warmed with the others and never called.
"""
# <<< route B

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.k3_drafter_attn_qknorm import (
    k3_drafter_attn_qknorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.k3_mla_attn_vb_out import (
    k3_mla_attn_vb_out,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.k3_mla_attn_workspace import (
    K3MlaAttnWorkspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.k3_mla_qkv import k3_mla_qkv

# >>> route B: no KDA verify kernels, and no trtllm::kda_mtp_decode (the built-in KDA verify's)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.k3_kda_buffers import K3KdaBuffers
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.k3_kda_decode_attn import (
    k3_kda_decode_attn,
)
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import k3_mla_decode_view

# <<< route B
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
from tensorrt_llm._torch.models.modeling_dspark import (
    GQADSparkForCausalLM,
    draft_is_embedded_in_target,
)
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, register_auto_model
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.kimi_k3_mla import KimiK3MLAAttention
from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention
from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import maybe_bcg_kda_core_inplace
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.situ import SituAndMul
from tensorrt_llm._torch.moe.fused_moe import (
    ConfigurableMoE,
    SiTuActivation,
    TRTLLMGenFusedMoE,
    create_moe,
)
from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind
from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._torch.pyexecutor.config_utils import is_mla
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from . import decode_comm as _decode_comm
from . import decode_gemv as _decode_gemv
from . import decode_moe as _decode_moe
from . import weights as _weights

if TYPE_CHECKING:
    from transformers import PretrainedConfig


# The GPU architecture this target IS. Routing will not send another one here, but a direct instantiation could,
# and the certification is per arch.
_SM = (10, 0)

#: Every trtllm op this target reaches for, in its forward and in the weight load. Declared here, asserted in
#: tests/unittest/_torch/modeling_v2: the K3-specific ops of the generic path (attention residuals, KDA, the router
#: and fused-A GEMMs), then the decode kernels'.
REQUIRED_TRTLLM_OPS = (
    "attn_res_fwd",
    "attn_res_rmsnorm_fwd",
    "attn_res_add_rmsnorm_fwd",
    "attn_res_add_rmsnorm_persistent_fwd",
    "kda_prefill",
    "kda_decode",
    # >>> route B: no KDA verify (kda_mtp_decode, k3_kda_attn, k3_kda_verify)
    "dsv3_router_gemm_op",
    "dsv3_fused_a_gemm_op",
    "k3_kda_decode_attn",
    # <<< route B
    "k3_mla_qkv",
    "k3_mla_attn_vb_out",
    # The decode path's GEMVs, the dense MLP's activation, the LM head and the embedding (decode_gemv.py).
    "k3_decode_gemv",
    "k3_ctm_gemv_wide",
    "k3_ctm_gemv_long",
    "k3_situ_mul",
    "k3_head_gemv",
    "k3_embed_norm",
    "allgather",
    # The decode path's collectives (decode_comm.py) and MoE (decode_moe.py).
    "mnnvl_allreduce_attn_res",
    "k3_sandwich_oproj",
    "k3_sandwich_tail",
    "k3_moe_front",
    "k3_moe",
    "k3_latent_reduce",
    "k3_route_quant",
    "mnnvl_allgather_split",
    # The DSpark drafter's decode path (K3DSparkDrafter): its GEMV sites, its block attention, and its all-reduces
    # with the residual add and RMSNorm (the split context projection's with hidden_norm).
    "k3_ctm_gemv",
    "k3_ctm_gemv_swiglu",
    "k3_drafter_attn_qknorm",
    "k3_sandwich_plain",
    "mnnvl_fusion_allreduce",
)

#: The engine surface the first forward checks before this target relies on it: per object, the attributes read.
#: A renamed field upstream fails here, loudly, instead of reading as a default.
REQUIRED_ENGINE_FIELDS = {
    "attn_metadata": (
        "num_contexts",
        "num_generations",
        "num_seqs",
        "num_tokens",
        "seq_lens",
        "tokens_per_block",
        "kv_cache_manager",
        "mamba_metadata",
    ),
    "kv_cache_manager": ("enable_block_reuse", "mamba_layer_cache"),
}

#: Stock code the generic path runs outside the catalog, declared so it is not consumed silently: every
#: tensorrt_llm import of this module that computes (test_modeling_v2_claims.py checks the list both ways). An entry
#: leaves the list when a catalog entry replaces it.
UNCERTIFIED_GENERIC_CALLS = (
    # The checkpoint load and the engine hooks this target inherits, and the causal LM around the text model.
    "tensorrt_llm._torch.models.modeling_kimi_linear.KimiLinearForCausalLM",
    "tensorrt_llm._torch.models.modeling_speculative.SpecDecOneEngineForCausalLM",
    "tensorrt_llm._torch.models.modeling_utils.DecoderModel",
    # The text model's stock modules.
    "tensorrt_llm._torch.modules.kimi_kda.KimiKDALinearAttention",
    "tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer.maybe_bcg_kda_core_inplace",
    # >>> route B: no trtllm::kda_mtp_decode registration (the built-in KDA verify's)
    # <<< route B
    "tensorrt_llm._torch.modules.kimi_k3_mla.KimiK3MLAAttention",
    "tensorrt_llm._torch.moe.fused_moe.create_moe",
    "tensorrt_llm._torch.moe.fused_moe.ConfigurableMoE",
    "tensorrt_llm._torch.moe.fused_moe.TRTLLMGenFusedMoE",
    "tensorrt_llm._torch.moe.fused_moe.routing.DeepSeekV3MoeRoutingMethod",
    "tensorrt_llm._torch.modules.gated_mlp.GatedMLP",
    "tensorrt_llm._torch.modules.situ.SituAndMul",
    "tensorrt_llm._torch.modules.rms_norm.RMSNorm",
    "tensorrt_llm._torch.distributed.AllReduce",
    "tensorrt_llm._torch.modules.multi_stream_utils.maybe_execute_in_parallel",
    # The DSpark drafter: the stock GQA drafter K3DSparkDrafter extends (its block forward where the drafter entries
    # do not take a block, its context projection without the TP group's collective state, its context k / v, its
    # heads and its weight load), and the stock builder's checks for which drafter a checkpoint gets.
    "tensorrt_llm._torch.models.modeling_dspark.GQADSparkForCausalLM",
    "tensorrt_llm._torch.models.modeling_dspark.draft_is_embedded_in_target",
    "tensorrt_llm._torch.models.modeling_dflash.DFlashForCausalLM",
    "tensorrt_llm._torch.pyexecutor.config_utils.is_mla",
)

# The K3 decode kernels' bounds: the token-count kernels take one tile of DECODE_MAX_TOKENS rows; the request-aware
# kernels take MAX_REQUESTS generation requests of at most MAX_TOKENS_PER_REQUEST tokens (1 + 7 drafts with DSpark).
DECODE_MAX_TOKENS = 8
MAX_REQUESTS = 8
MAX_TOKENS_PER_REQUEST = 8

# The MLA generation kernels for K3's 96 query heads exist only at a 64-token page (the built-in model's own
# get_model_defaults sets it for the same reason).
_TOKENS_PER_BLOCK = 64

_LANG_PREFIX = "language_model."


# ----------------------------------------------------------------------------------------------------------------------
# The text model: Kimi K3's decoder (93 layers: KDA / MLA attention, attention residuals, the dense layer-0 MLP and
# the latent MoE), its generic path. Each step's classification goes to the attention modules (`K3DecodeKDA`,
# `K3DecodeMLA` below).
# ----------------------------------------------------------------------------------------------------------------------

# A/B escape hatch: restore nn.Linear for the K3 latent MoE projections
# instead of the min-latency fused GEMM op (read once at import).
_K3_DISABLE_MIN_LATENCY_LATENT_PROJ = (
    os.environ.get("TLLM_K3_DISABLE_MIN_LATENCY_LATENT_PROJ", "0") == "1"
)


# Identity-RoPE table positions for the MLA backends. K3 is NoPE (the table
# holds cos=1/sin=0), but the chunked-context path indexes the table by
# absolute position, so it must cover max_position_embeddings (~512MB per
# backend for the 1M-position checkpoint); a smaller table is read out of
# bounds. KIMI_K3_MLA_MAX_POSITIONS overrides the size for short-context
# deployments.
_KIMI_K3_MLA_MAX_POSITIONS_ENV = "KIMI_K3_MLA_MAX_POSITIONS"


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


KIMI_K3_AUX_ATTN_RES_STREAM_ENV = "KIMI_K3_AUX_ATTN_RES_STREAM"


_AUX_ATTN_RES_STREAM_ENABLED = os.environ.get(KIMI_K3_AUX_ATTN_RES_STREAM_ENV, "1") == "1"


KIMI_K3_FUSED_ATTN_RES_ENV = "KIMI_K3_FUSED_ATTN_RES"


_FUSED_ATTN_RES_ENABLED = os.environ.get(KIMI_K3_FUSED_ATTN_RES_ENV, "1") == "1"


KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV = "KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS"


KIMI_K3_ATTN_RES_TOPOLOGY_ENV = "KIMI_K3_ATTN_RES_TOPOLOGY"


_ATTN_RES_TOPOLOGIES = ("per_token", "persistent", "split")


# Turning the port on should not require knowing which of the three topologies
# is the right one: "1" selects the measured policy (``split``), so enabling the
# feature and choosing the policy are one action through one variable.
_ATTN_RES_TOPOLOGY_ON = "1"


def _read_attn_res_topology() -> str:
    """Default ``per_token``: the persistent kernel is opt-in.

    ``1`` is the only accepted on-value and resolves to ``split``. The named
    topologies stay available for measurement: ``persistent`` uses the
    persistent kernel at every shape it implements, ``per_token`` at none.
    """
    raw = os.environ.get(KIMI_K3_ATTN_RES_TOPOLOGY_ENV, "per_token")
    if raw == _ATTN_RES_TOPOLOGY_ON:
        return "split"
    if raw not in _ATTN_RES_TOPOLOGIES:
        # Loudly, for the same reason as the token ceiling below: a mistyped A/B
        # arm that silently fell back to the default would measure one side
        # twice and report no difference.
        raise ValueError(
            f"{KIMI_K3_ATTN_RES_TOPOLOGY_ENV} must be one of "
            f"{_ATTN_RES_TOPOLOGIES} or {_ATTN_RES_TOPOLOGY_ON!r} "
            f"(which means 'split'), got {raw!r}"
        )
    return raw


def _read_fused_attn_res_max_tokens() -> int:
    """Resolved after the topology, because its default follows it.

    With the persistent port off -- the default -- the ceiling is 1, the
    pre-existing gate: the fused epilogue is taken at the single-token decode
    shape and nowhere else. Enabling the port raises it to 32, the top of the
    measured range, so that "off" keeps meaning "unchanged".
    """
    default = "1" if _ATTN_RES_TOPOLOGY == "per_token" else "32"
    raw = os.environ.get(KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV, default)
    try:
        value = int(raw)
    except ValueError:
        # Failing loudly matters more than usual here: a mistyped A/B arm that
        # silently fell back to the default would measure the candidate twice
        # and report no difference.
        raise ValueError(
            f"{KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV} must be a positive integer, got {raw!r}"
        ) from None
    if value < 1:
        raise ValueError(f"{KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV} must be >= 1, got {value}")
    return value


_ATTN_RES_TOPOLOGY = _read_attn_res_topology()


_FUSED_ATTN_RES_MAX_TOKENS = _read_fused_attn_res_max_tokens()


def _persistent_attn_res_applicable(M: int, H: int, N: int) -> bool:
    """Shape gate for the persistent kernel: H == 7168 and 2 <= N <= 9.

    No token ceiling: the persistent grid is sized by the SM count, not by the
    token count, so prefill is the case it exists for.
    """
    del M  # deliberately unused; see above
    return H == 7168 and 2 <= N <= 9


def _use_persistent_attn_res(M: int, H: int, N: int, max_fused_tokens: int) -> bool:
    """Pick between the two fused kernels for this call site.

    ``persistent`` takes the persistent kernel at every shape it implements;
    ``split`` takes it only above ``max_fused_tokens`` tokens, which stands in
    for the prefill/decode boundary. Shapes the persistent kernel does not
    implement fall through to the caller's existing gate and land on the
    unfused path.
    """
    if not _persistent_attn_res_applicable(M, H, N):
        return False
    if _ATTN_RES_TOPOLOGY == "persistent":
        return True
    return _ATTN_RES_TOPOLOGY == "split" and M > max_fused_tokens


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
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
    ):
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


def _rms_norm_eps(norm: nn.Module) -> float:
    if hasattr(norm, "eps"):
        return float(norm.eps)
    return float(norm.variance_epsilon)


def _note_attn_res_fusion(site: str, fused: bool, M: int, H: int, N: int) -> None:
    """Report whether the fused path was actually reached, once per shape.

    ``_FUSED_ATTN_RES_ENABLED`` only says the feature is switched on, not that
    the shape gate let the call through, and a rejected call looks exactly like
    a disabled one in the logs. Emitted at debug level: it is once per distinct
    shape, not once per process, so it is a diagnostic rather than a summary.
    """
    logger.debug_once(
        f"Kimi K3 attn-res fusion [{site}]: "
        f"{'FUSED' if fused else 'fallback'} (M={M}, H={H}, N={N})",
        key=f"kimi_k3_attn_res_fusion_{site}_{fused}_{M}_{H}_{N}",
    )


def _apply_attn_res_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
    max_fused_tokens: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Fuse attention-residual mixing with its immediately following norm (at most ``max_fused_tokens`` tokens,
    default ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS``)."""
    if max_fused_tokens is None:
        max_fused_tokens = _FUSED_ATTN_RES_MAX_TOKENS
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    N = K + 1
    # The fused path is taken for M <= max_fused_tokens, H == 7168 and N <= 12,
    # which is the measured window; larger token counts have not been measured
    # and fall back to the unfused add + attn_res_fwd + RMSNorm path.
    if _use_persistent_attn_res(M, H, N, max_fused_tokens):
        try:
            persistent_op = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd
        except (AttributeError, RuntimeError):
            return None
        _, output = persistent_op(
            prefix_sum.reshape(M, 1, H).contiguous(),
            None,
            block_residual.reshape(K, M, 1, H).contiguous(),
            proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
            norm.weight.to(torch.bfloat16).contiguous(),
            output_norm.weight.to(torch.bfloat16).contiguous(),
            float(norm.eps),
            _rms_norm_eps(output_norm),
        )
        _note_attn_res_fusion("attn_res+norm/persistent", True, M, H, N)
        return output.reshape(M, H)

    if M > max_fused_tokens or H != 7168 or N > 12:
        _note_attn_res_fusion("attn_res+norm", False, M, H, N)
        return None
    try:
        attn_res_rmsnorm_op = torch.ops.trtllm.attn_res_rmsnorm_fwd
    except (AttributeError, RuntimeError):
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    output = attn_res_rmsnorm_op(
        layer_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )
    _note_attn_res_fusion("attn_res+norm", True, M, H, N)
    return output.reshape(M, H)


def _apply_attn_res_add_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
    max_fused_tokens: Optional[int] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fuse ``prefix_sum + addend``, attention-residual, and trailing norm (``max_fused_tokens`` as in
    ``_apply_attn_res_rmsnorm_fused``).

    The production residual add produces a BF16 tensor that remains live across
    the following MLP. The kernel therefore returns that materialized,
    BF16-rounded prefix sum alongside the normalized attention-residual output,
    while avoiding a separate add launch and a re-read of the intermediate by
    attention-residual selection.
    """
    if max_fused_tokens is None:
        max_fused_tokens = _FUSED_ATTN_RES_MAX_TOKENS
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
    if _use_persistent_attn_res(M, H, N, max_fused_tokens):
        try:
            persistent_op = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd
        except (AttributeError, RuntimeError):
            return None
        updated_prefix_sum, output = persistent_op(
            prefix_sum.reshape(M, 1, H).contiguous(),
            addend.reshape(M, 1, H).contiguous(),
            block_residual.reshape(K, M, 1, H).contiguous(),
            proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
            norm.weight.to(torch.bfloat16).contiguous(),
            output_norm.weight.to(torch.bfloat16).contiguous(),
            float(norm.eps),
            _rms_norm_eps(output_norm),
        )
        _note_attn_res_fusion("add+attn_res+norm/persistent", True, M, H, N)
        return updated_prefix_sum.reshape(M, H), output.reshape(M, H)

    if M > max_fused_tokens or H != 7168 or N > 12:
        _note_attn_res_fusion("add+attn_res+norm", False, M, H, N)
        return None
    try:
        attn_res_add_rmsnorm_op = torch.ops.trtllm.attn_res_add_rmsnorm_fwd
    except (AttributeError, RuntimeError):
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    addend_kernel = addend.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    updated_prefix_sum, output = attn_res_add_rmsnorm_op(
        layer_kernel,
        addend_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )
    _note_attn_res_fusion("add+attn_res+norm", True, M, H, N)
    return updated_prefix_sum.reshape(M, H), output.reshape(M, H)


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


def _apply_attn_res_and_rmsnorm(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
    max_fused_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Apply attention-residual selection and the next RMSNorm. ``max_fused_tokens``: the largest token count the
    fused kernel takes (default ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS``)."""
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_rmsnorm_fused(
            prefix_sum, block_residual, proj, norm, output_norm, max_fused_tokens
        )
        if fused is not None:
            return fused
    return output_norm(_apply_attn_res(prefix_sum, block_residual, proj, norm))


def _apply_attn_res_add_and_rmsnorm(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
    max_fused_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add an attention output to the running residual, then select and norm (``max_fused_tokens`` as in
    ``_apply_attn_res_and_rmsnorm``)."""
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_add_rmsnorm_fused(
            prefix_sum, addend, block_residual, proj, norm, output_norm, max_fused_tokens
        )
        if fused is not None:
            return fused
    updated_prefix_sum = prefix_sum + addend
    return updated_prefix_sum, _apply_attn_res_and_rmsnorm(
        updated_prefix_sum, block_residual, proj, norm, output_norm, max_fused_tokens
    )


# A wide decode step's residual epilogues take the fused add + attn_res + RMSNorm kernels up to this many tokens, where
# they are faster than the add -> attn_res -> RMSNorm chain.
_WIDE_ATTN_RES_MAX_TOKENS = 32


def _attn_res_max_tokens(step: Optional[DecodeStep]) -> Optional[int]:
    """The most tokens of ``step`` the fused attn_res kernels take: one token tile (``DECODE_MAX_TOKENS``) on a step
    ``decode_step`` classifies, ``_WIDE_ATTN_RES_MAX_TOKENS`` on a wide decode step; None on any other step (the
    generic path's ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS``)."""
    if step is None:
        return None
    return _WIDE_ATTN_RES_MAX_TOKENS if step.wide else DECODE_MAX_TOKENS


_K3_ROUTED_EXPERT_KEY_SUFFIXES = ("block_sparse_moe.experts", "mlp.experts")


# The routed experts' module path prefixes. ``exclude_modules`` matches with
# wildcards and walks ancestor prefixes, so an empty prefix would widen what
# matches instead of just missing.
_K3_ROUTED_EXPERT_MODULE_PREFIXES = ("language_model.model.", "model.")


# Routed-expert quantization used when the checkpoint declares nothing per
# layer. The original ``moonshotai/Kimi-K3`` ships a compressed-tensors
# ``mxfp4-pack-quantized`` config with no ModelOpt per-layer entries, and that
# checkpoint is what this default has always served.
_K3_DEFAULT_ROUTED_QUANT_ALGO = QuantAlgo.W4A8_MXFP4_MXFP8


def _load_packed_mxfp4_expert(backend, base, expert_idx, local_slot_id, get_tensor) -> None:
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


class _K3ExpertCkptSpec(NamedTuple):
    """How one routed-expert quantization is spelled and loaded."""

    # Per-``w{1,2,3}`` checkpoint tensor suffixes this layout stores.
    kinds: Tuple[str, ...]
    loader: Callable[..., None]
    # Set of filled slots the loader maintains, checked after the load.
    loaded_slots_attr: str
    # Whether the layer is finalized after its experts load (the MXFP4 loaders write through).
    needs_layer_finalize: bool


_K3_EXPERT_CKPT_SPECS = {
    QuantAlgo.W4A8_MXFP4_MXFP8: _K3ExpertCkptSpec(
        kinds=("weight_packed", "weight_scale"),
        loader=_load_packed_mxfp4_expert,
        loaded_slots_attr="_packed_mxfp4_loaded_slots",
        needs_layer_finalize=False,
    ),
}


def _k3_expert_ckpt_spec(quant_algo: Optional[QuantAlgo]) -> _K3ExpertCkptSpec:
    spec = _K3_EXPERT_CKPT_SPECS.get(quant_algo)
    if spec is None:
        raise NotImplementedError(
            f"Kimi K3 routed experts are quantized as {quant_algo}, for which "
            "no per-expert checkpoint layout is known. Supported: "
            f"{sorted(a.name for a in _K3_EXPERT_CKPT_SPECS)}."
        )
    return spec


class KimiK3MoERuntime(nn.Module):
    """Kimi K3 latent MoE block backed by ConfigurableMoE."""

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

        situ_beta, situ_linear_beta = _resolve_kimi_situ_betas(cfg)
        self._situ_betas = (situ_beta, situ_linear_beta)
        dtype = torch.bfloat16

        # Routing scores stay fp32; the gate GEMM runs bf16xbf16 with fp32
        # accumulate/output (checkpoint stores the gate weight in bf16; saves a
        # per-layer input cast + fp32 splitK pair on the bs1 decode path).
        # KIMI_K3_ROUTER_BF16=0 forces the upcast-to-fp32 GEMM.
        _router_bf16_env = os.environ.get("KIMI_K3_ROUTER_BF16")
        _router_bf16 = _router_bf16_env == "1" if _router_bf16_env is not None else True
        self.gate = KimiK3MoEGate(cfg, logits_gemm_dtype=torch.bfloat16 if _router_bf16 else None)

        routed_moe_model_config = self._routed_moe_model_config(model_config)
        routed_quant_config = self._resolve_routed_quant_config(model_config, layer_idx)
        # Resolved here so ``load_weights`` reads the checkpoint layout off the
        # module instead of re-deriving it at each of its three call sites.
        self.expert_ckpt_spec = _k3_expert_ckpt_spec(routed_quant_config.quant_algo)
        routed_moe_kwargs = dict(
            routing_method=self.gate.routing_method,
            num_experts=self.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            dtype=dtype,
            # Kimi owns the latent reduction so it can order that collective
            # after the shared expert's auxiliary-stream reduction.
            reduce_results=False,
            model_config=routed_moe_model_config,
            override_quant_config=routed_quant_config,
            layer_idx=layer_idx,
            aux_stream_dict=aux_stream_dict,
            # Let CommunicationFactory select the best available strategy.
            communication_method=None,
            activation=SiTuActivation(
                gate_softcap=situ_beta,
                linear_softcap=situ_linear_beta,
            ),
            # A request that silently degraded to CUTLASS would be benchmarked
            # as if it were the backend that was asked for, and the decline is
            # easy to trigger: MegaMoE has its own token / top-k limits and is
            # EP-only, and CuteDSL declines on activation shape, SM version and
            # the CuTe DSL dependency. As measured once: a CUTEDSL request
            # was turned down on every one of the 92 MoE layers, on all 16
            # ranks, and still produced correct text and a zero exit -- the
            # only trace was a warning line per layer. Fail in the resolver
            # instead, which reports the rejection trail.
            #
            # CUTLASS is absent on purpose: it is the fallback target, so
            # "degraded to CUTLASS" is not a thing that can happen to it.
            allow_backend_degradation=routed_moe_model_config.moe_backend
            not in ("MEGAMOE_DEEPGEMM", "MEGAMOE_CUTEDSL", "CUTEDSL"),
        )
        self._check_trtllm_situ_quant(
            routed_moe_model_config.moe_backend, routed_quant_config.quant_algo
        )

        self.routed_experts = create_moe(**routed_moe_kwargs)
        if not isinstance(self.routed_experts, ConfigurableMoE):
            raise RuntimeError(
                "Kimi K3 requires ConfigurableMoE; ENABLE_CONFIGURABLE_MOE must not be disabled."
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
        shared_model_config = copy.copy(model_config)
        shared_model_config.quant_config = QuantConfig()
        # Direct MoE-TP leaves both branches as partials for one concatenated
        # all-reduce.
        use_shared_tp = model_config.mapping.tp_size > 1
        self._reduce_routed_output = (
            use_shared_tp
            and self.routed_experts.backend.scheduler_kind != MoESchedulerKind.FUSED_COMM
        )
        if self._reduce_routed_output and self.routed_experts.all_reduce is None:
            raise RuntimeError(
                "Kimi K3 direct MoE tensor parallelism requires the "
                "ConfigurableMoE all-reduce even when reduce_results=False."
            )
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
            reduce_output=use_shared_tp,
            layer_idx=layer_idx,
            is_shared_expert=True,
        )
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
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
        # The decode path (decode_moe.py) and the decode GEMVs' state, set by the target's post_load_weights.
        self.decode_moe: Optional[_decode_moe.K3DecodeMoeLayer] = None
        self.decode_gemvs: Optional[_decode_gemv.K3DecodeGemvs] = None

    @staticmethod
    def _routed_projection(hidden_states: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if _K3_DISABLE_MIN_LATENCY_LATENT_PROJ or not isinstance(projection, nn.Linear):
            return projection(hidden_states)
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(
            hidden_states, projection.weight.t(), None, None
        )

    @staticmethod
    def _select_moe_tp_ep(mapping: Mapping) -> Tuple[int, int]:
        # >>> route B: this target's split
        """The routed-expert ``(moe_tp, moe_ep)`` split: the user config's explicit
        ``moe_tensor_parallel_size`` / ``moe_expert_parallel_size`` (16 x 1, asserted at
        construction)."""
        # <<< route B
        return mapping.moe_tp_size, mapping.moe_ep_size

    @staticmethod
    def _resolve_routed_quant_config(model_config: ModelConfig, layer_idx: int) -> QuantConfig:
        """Routed-expert quantization for ``layer_idx``: the MXFP4 checkpoint declares nothing per layer (asserted at
        construction), so the experts keep the ``W4A8_MXFP4_MXFP8`` default.

        An exclusion outranks the default: ``create_weights`` treats an override
        as authoritative over anything ``__post_init__`` wrote, so this return
        value stands in for both quantization passes and exclusion is the one
        that runs second. It is matched as a pattern, so it is asked only about
        real module names.
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

        logger.debug(
            "Kimi K3 layer %d routed experts: no per-layer quant config in the "
            "checkpoint, defaulting to %s",
            layer_idx,
            _K3_DEFAULT_ROUTED_QUANT_ALGO,
        )
        return QuantConfig(quant_algo=_K3_DEFAULT_ROUTED_QUANT_ALGO)

    @staticmethod
    def _check_trtllm_situ_quant(moe_backend: str, quant_algo: Optional[QuantAlgo]) -> None:
        """Reject a routed-expert format trtllm-gen has no fused SiTu cubin for.

        trtllm-gen has fused SiTu FC1 cubins for two input formats and no
        standalone SiTu activation kernel, so anything else has to die here
        rather than in a cubin lookup deep inside the runner. Checked against
        the resolved backend, not the K3 architecture branch, because the
        generic FP8_BLOCK_SCALES fallback in ``resolve_moe_backend`` can also
        land on TRTLLM.

        The admitted set is read off the backend rather than restated here,
        because restating it is what broke. This guard was written in #17865
        when MXFP4 was the only fused SiTu drop; #17940 then added the NVFP4
        (group-16 ``Bmm_E2m1_E2m1E2m1_..._siTuGlu_*``) cubins and updated
        ``TRTLLMGenFusedMoE``'s set without touching this copy. For the week
        in between, an NVFP4 K3 checkpoint could not start at all -- and not
        only when TRTLLM was asked for by name, because
        ``ModelConfig.resolve_moe_backend`` sends every K3 architecture to
        TRTLLM, so the default AUTO configuration hit this raise too. The unit
        tests did not catch it: they call ``create_moe`` directly and never
        reach this guard, so the kernel path stayed green while the model path
        was closed.

        A staticmethod, not an inline block, so that the invariant is
        reachable from a test without constructing the whole runtime.
        """
        situ_supported = TRTLLMGenFusedMoE.situ_supported_quant_algos()
        if moe_backend != "TRTLLM" or quant_algo in situ_supported:
            return
        supported = ", ".join(sorted(algo.name for algo in situ_supported))
        raise ValueError(
            f"Kimi K3 routed experts are quantized as {quant_algo}, which the "
            "TRTLLM (trtllm-gen) MoE backend cannot serve: fused SiTu cubins "
            f"exist only for {supported}. Set moe_config.backend to CUTLASS "
            "or MEGAMOE_CUTEDSL."
        )

    @staticmethod
    def _routed_moe_model_config(model_config: ModelConfig) -> ModelConfig:
        """Build a private routed-expert mapping without mutating the shared
        config, with the split of ``_select_moe_tp_ep``."""
        # Every backend here declares ``ActivationType.SiTu`` in its
        # ``activation_support``; the list is not a preference order. CUTEDSL
        # joined once its act-fusion kernel grew the SiTU epilogue.
        supported_backends = {
            "CUTLASS",
            "TRTLLM",
            "CUTEDSL",
            "MEGAMOE_DEEPGEMM",
            "MEGAMOE_CUTEDSL",
        }
        if model_config.moe_backend not in supported_backends:
            raise ValueError(
                "Kimi K3 SiTU routed experts only support the CUTLASS, TRTLLM, "
                "CUTEDSL, MEGAMOE_DEEPGEMM, and MEGAMOE_CUTEDSL backends; "
                f"got {model_config.moe_backend!r}."
            )
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        mapping = model_config.mapping

        moe_tp, moe_ep = KimiK3MoERuntime._select_moe_tp_ep(mapping)
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
        if routed_model_config.moe_backend in {
            "MEGAMOE_DEEPGEMM",
            "MEGAMOE_CUTEDSL",
        }:
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

    def tail_rp_eligible(self, hidden_states: torch.Tensor, step: Optional[DecodeStep]) -> bool:
        """Whether this layer's forward on ``step`` can hand its output on as this rank's unreduced row-parallel
        share (``partial_tail``): the decode path takes the step with that tail."""
        return self.decode_moe is not None and self.decode_moe.takes(hidden_states, step, True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        partial_tail: bool = False,
        step: Optional[DecodeStep] = None,
    ) -> Union[torch.Tensor, _decode_comm.PendingTail]:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16. ``step``: the step's classification
        (``decode_step``); the decode path (``decode_moe.py``) runs the steps it takes. ``partial_tail`` (only where
        ``tail_rp_eligible`` holds): return this rank's unreduced share of the output instead, a ``PendingTail`` at
        most 8 tokens, a tensor on a wide decode step."""
        decode = self.decode_moe
        if decode is not None and decode.takes(hidden_states, step, partial_tail):
            return decode.forward(
                self, hidden_states, self.decode_gemvs, partial_tail, push=step.latent_push
            )
        if partial_tail:
            raise RuntimeError("the row-parallel MoE tail needs the decode path to take the step")
        identity = hidden_states
        router_logits = self.gate.compute_logits(hidden_states)
        moe_all_reduce = self.routed_experts.all_reduce if self._reduce_routed_output else None

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
            if self._reduce_routed_output:
                return y
            # Communication-backed paths return a complete routed result.
            y = self.routed_expert_norm(y)
            return self._routed_projection(y, self.routed_expert_up_proj)

        # Shared experts depend only on the block input, so overlap their GEMMs
        # with the routed dispatch/expert/combine chain. Multi-stream engages
        # only under CUDA graphs; otherwise both branches run in order on the
        # default stream. The shared GatedMLP includes its output all-reduce on
        # the auxiliary stream. The join below must precede the routed
        # all-reduce: concurrent collectives on different streams can corrupt
        # SYMM_MEM all-reduce state.
        routed_out, shared_out = maybe_execute_in_parallel(
            _routed_output,
            lambda: self.shared_experts(identity),
            self.moe_main_event,
            self.moe_shared_event,
            self.shared_expert_stream,
            disable_on_compile=True,
        )
        if self._reduce_routed_output:
            routed_latent = moe_all_reduce(routed_out)
            routed_latent = self.routed_expert_norm(routed_latent)
            routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        return routed_out + shared_out


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


class KimiMLARuntime(nn.Module):
    """Wraps K3 MLA and applies its external TP output reduction."""

    def __init__(
        self,
        cfg: "PretrainedConfig",
        layer_idx: int,
        model_config: ModelConfig,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ) -> None:
        super().__init__()

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
        reduce_output = mapping.tp_size > 1
        self._o_allreduce = (
            AllReduce(
                mapping=mapping,
                strategy=model_config.allreduce_strategy,
                dtype=torch.bfloat16,
            )
            if reduce_output
            else None
        )
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
        attention_config.quant_config = QuantConfig(
            kv_cache_quant_algo=model_config.quant_config.kv_cache_quant_algo
            if model_config.quant_config is not None
            else None
        )
        attention_config._frozen = model_config._frozen
        self.mixer = K3DecodeMLA(
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
        )

    def will_run_decode_branch(
        self, attn_metadata: AttentionMetadata, step: Optional[DecodeStep]
    ) -> bool:
        """Whether the mixer runs ``step`` on the decode kernels (``K3DecodeMLA.will_run_decode_branch``)."""
        return self.mixer.will_run_decode_branch(attn_metadata, step)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        step: Optional[DecodeStep] = None,
        reduce_output: bool = True,
        project_output: bool = True,
    ) -> torch.Tensor:
        """``reduce_output=False`` returns ``o_proj``'s TP partial (no all-reduce); ``project_output=False`` the
        mixer's gated attention output before ``o_proj``, only where ``will_run_decode_branch`` holds."""
        # MLA.forward takes position_ids first; K3 is NoPE, so pass None.
        out = self.mixer(
            None, hidden_states, attn_metadata, step=step, project_output=project_output
        )
        if project_output and reduce_output and self._o_allreduce is not None:
            # Head-sharded TP: sum the row-sharded o_proj partials across
            # the head-shard group.
            out = self._o_allreduce(out)
        return out


class KimiLinearDecoderLayer(nn.Module):
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
            self.linear_attn = K3DecodeKDA(
                cfg,
                layer_idx,
                mapping=model_config.mapping,
                allreduce_strategy=model_config.allreduce_strategy,
                aux_stream=aux_stream_dict[AuxStreamType.Attention],
                model_config=attention_config,
            )
        else:
            self.self_attn = KimiMLARuntime(
                cfg,
                layer_idx,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
            )

        self.is_moe = (
            cfg.num_experts is not None
            and layer_idx >= cfg.first_k_dense_replace
            and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0
        )
        if self.is_moe:
            self.block_sparse_moe = KimiK3MoERuntime(model_config, cfg, layer_idx, aux_stream_dict)
        else:
            situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
            situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
            self.mlp_tp_size = math.gcd(cfg.intermediate_size, model_config.mapping.tp_size)
            # Over MNNVL (one NVLink domain across the nodes, where a cross-node all-reduce costs what a node's
            # does) the MLP stays split over the whole TP group, the per-rank shapes its decode GEMVs take
            # (decode_gemv.SITES); otherwise it stays within one node.
            spans_nodes = self._mnnvl_allreduce() is not None
            if self.mlp_tp_size > model_config.mapping.gpus_per_node and not spans_nodes:
                self.mlp_tp_size = math.gcd(self.mlp_tp_size, model_config.mapping.gpus_per_node)
            mlp_model_config = copy.copy(model_config)
            mlp_model_config.quant_config = QuantConfig()
            # K3's dense layer is BF16, so a unit block size gives the same
            # subgroup selection as DeepSeek-V3.
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
            self._situ = (situ_beta, situ_linear_beta)
            # The decode GEMVs' state (decode_gemv.py), set by the target's cache_derived_state.
            self.decode_gemvs: Optional[_decode_gemv.K3DecodeGemvs] = None

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
        # The decode path's collectives (decode_comm.py), and whether their sandwich takes this layer's o_proj; set
        # by the target's post_load_weights.
        self.decode_comm: Optional[_decode_comm.K3DecodeComm] = None
        self.sandwich_oproj = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
        capture: Optional[Tuple[Any, int]] = None,
        step: Optional[DecodeStep] = None,
        prenormed: bool = False,
        pending_moe_partial: Optional[Union[torch.Tensor, _decode_comm.PendingTail]] = None,
        defer_moe_tail: bool = False,
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, Any]]:
        """Port of HF ``KimiDecoderLayer._forward_attn_residual`` (per token).

        ``block_residual`` is a preallocated snapshot bank in kernel-native
        ``[K_max, M, H]`` layout. Returns the running prefix sum and the
        number of valid bank rows.

        ``capture`` is ``(spec_metadata, layer_id)`` and taps the DSpark aux
        stream for the layer BEFORE this one: the aggregated stream for layer j
        is by definition what its next consumer sees, so the mixture computed
        below already is it. Reading it here beats recomputing it, and is only
        possible because K3 asserts pp_size == 1 -- layer j+1 is always local.
        PP support would need a recompute at the rank boundary.

        ``step`` is the step's classification (``decode_step``), handed to the
        attention module.

        ``prenormed`` (layer 0 on a decode step): ``hidden_states`` already is
        this layer's input norm, and the layer's input, the step's embedding,
        already is in ``block_residual[0]`` (``K3DecodeGemvs.embed_norm``).

        The post-attention step of at most ``AR_ATTN_RES_MAX_TOKENS`` tokens
        runs the attention's all-reduce and the residual update in one
        collective once the target has built its ``decode_comm``: with o_proj
        (``K3DecodeComm.sandwich_oproj``) where the sandwich takes the layer,
        the step and the attention's decode branch, else on the attention's
        unreduced o_proj output (``K3DecodeComm.allreduce_attn_res``).

        ``pending_moe_partial`` (only where ``accepts_moe_partial`` held): the
        previous layer's MoE output, unreduced (``defer_moe_tail``); then
        ``hidden_states`` is the prefix sum without it, and this layer's
        pre-attention step reduces and adds it: a ``PendingTail`` in one
        ``K3DecodeComm.sandwich_tail`` call, a wide decode step's tensor by its
        all-reduce and the fused add + attn_res + RMSNorm. On a tapped layer
        the ``sandwich_tail`` call also stores the tap into the capture slot
        where the speculative metadata exposes it (``capture_view``); other
        tapped layers take the tap after the step.

        ``defer_moe_tail``: return ``(prefix_sum, num_snapshots, partial)``
        instead, ``partial`` this layer's MoE output unreduced, for the next
        consumer to reduce.
        """
        prefix_sum = hidden_states
        valid_block_residual = block_residual[:num_snapshots]
        attn_res_max_tokens = _attn_res_max_tokens(step)
        tail = (
            pending_moe_partial
            if isinstance(pending_moe_partial, _decode_comm.PendingTail)
            else None
        )
        # A snapshot layer whose pre-attention step is the sandwich tail has the kernel store the running prefix sum
        # straight into the bank row it snapshots.
        snapshot_row = None
        if tail is not None and self.layer_idx % self.attn_res_block_size == 0:
            snapshot_row = block_residual[num_snapshots]
        # A tapped layer whose pre-attention step is the sandwich tail has the kernel store the tap straight into the
        # layer's capture slot, where the speculative metadata exposes that slot as a view (``capture_view``).
        tap_view = None
        if tail is not None and capture is not None:
            view_of = getattr(capture[0], "capture_view", None)
            tap_view = view_of(capture[1], prefix_sum.shape[0]) if view_of is not None else None

        if prenormed:
            assert num_snapshots == 0 and self.layer_idx % self.attn_res_block_size == 0
            assert pending_moe_partial is None and capture is None
        elif tail is not None:
            hidden_states, prefix_sum = self.decode_comm.sandwich_tail(
                tail,
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
                updated_out=snapshot_row,
                tap=tap_view,
                tap_updated=not _AUX_ATTN_RES_STREAM_ENABLED,
            )
        elif pending_moe_partial is not None:
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                _decode_comm.wide_all_reduce(self._o_allreduce(), pending_moe_partial),
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
                attn_res_max_tokens,
            )
        elif capture is not None:
            # The mixture tap needs the PRE-norm value, which the fused
            # attn-res + RMSNorm kernel does not expose. Keep the two steps
            # split on captured layers only and fuse everywhere else.
            if num_snapshots > 0:
                hidden_states = _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            # A property of the DRAFTER checkpoint, not a knob: a mismatch only lowers
            # acceptance, silently. hidden_states is the pre-norm attn_res mixture;
            # prefix_only wants the running prefix, already in hand as prefix_sum.
            tapped = hidden_states if _AUX_ATTN_RES_STREAM_ENABLED else prefix_sum
            capture[0].maybe_capture_hidden_states(capture[1], tapped, None)
            hidden_states = self.input_layernorm(hidden_states)
        elif num_snapshots > 0:
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
                attn_res_max_tokens,
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        if capture is not None and pending_moe_partial is not None and tap_view is None:
            # The tapped layer handed its MoE output on and no kernel wrote the tap: the step above reduced the output
            # into prefix_sum. Tap that value's pre-norm attn_res mixture, what the split path captures.
            tapped = (
                _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
                if _AUX_ATTN_RES_STREAM_ENABLED
                else prefix_sum
            )
            capture[0].maybe_capture_hidden_states(capture[1], tapped, None)

        if self.layer_idx % self.attn_res_block_size == 0:
            if not prenormed and snapshot_row is None:
                block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None
        attention = self.linear_attn if self.is_kda else self.self_attn
        comm = self.decode_comm
        if comm is not None and comm.takes_post_attention(hidden_states, step):
            if (
                self.sandwich_oproj
                and step is not None
                and step.num_tokens <= _decode_comm.SANDWICH_MAX_TOKENS
                and attention.will_run_decode_branch(attn_metadata, step)
            ):
                core = attention(hidden_states, attn_metadata, step=step, project_output=False)
                hidden_states, prefix_sum = comm.sandwich_oproj(
                    core,
                    self._o_proj().weight,
                    prefix_sum,
                    valid_block_residual,
                    self.mlp_res_proj,
                    self.mlp_res_norm,
                    self.post_attention_layernorm,
                )
            else:
                partial = attention(hidden_states, attn_metadata, step=step, reduce_output=False)
                hidden_states, prefix_sum = comm.allreduce_attn_res(
                    partial,
                    prefix_sum,
                    valid_block_residual,
                    self.mlp_res_proj,
                    self.mlp_res_norm,
                    self.post_attention_layernorm,
                )
        else:
            if comm is not None and step is not None and step.wide:
                partial = attention(hidden_states, attn_metadata, step=step, reduce_output=False)
                attention_output = _decode_comm.wide_all_reduce(self._o_allreduce(), partial)
            else:
                attention_output = attention(hidden_states, attn_metadata, step=step)
            if prefix_sum is None:
                prefix_sum = attention_output
                hidden_states = _apply_attn_res_and_rmsnorm(
                    prefix_sum,
                    valid_block_residual,
                    self.mlp_res_proj,
                    self.mlp_res_norm,
                    self.post_attention_layernorm,
                    attn_res_max_tokens,
                )
            else:
                prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                    prefix_sum,
                    attention_output,
                    valid_block_residual,
                    self.mlp_res_proj,
                    self.mlp_res_norm,
                    self.post_attention_layernorm,
                    attn_res_max_tokens,
                )
        all_rank_num_tokens = getattr(attn_metadata, "all_rank_num_tokens", None)
        if self.is_moe and defer_moe_tail:
            partial = self.block_sparse_moe(
                hidden_states, all_rank_num_tokens, partial_tail=True, step=step
            )
            return prefix_sum, num_snapshots, partial
        if self.is_moe:
            hidden_states = self.block_sparse_moe(hidden_states, all_rank_num_tokens, step=step)
        else:
            hidden_states = self._dense_mlp(hidden_states, step)

        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, num_snapshots

    def _dense_mlp(self, hidden_states: torch.Tensor, step: Optional[DecodeStep]) -> torch.Tensor:
        """The dense MLP: on a step of at most DECODE_MAX_TOKENS tokens, its GEMVs and activation on the decode
        kernels (``K3DecodeGemvs.dense_mlp``), then the down projection's all-reduce; else the module."""
        gemvs = self.decode_gemvs
        if gemvs is not None and step is not None and step.small:
            out = gemvs.dense_mlp(
                hidden_states,
                self.mlp.gate_up_proj.weight,
                self.mlp.down_proj.weight,
                *self._situ,
            )
            if out is not None:
                return self.mlp.down_proj.all_reduce(out) if self.mlp_tp_size > 1 else out
        return self.mlp(hidden_states)

    def _mnnvl_allreduce(self):
        """The MNNVL all-reduce of this layer's attention output, or None."""
        attention = self.linear_attn if self.is_kda else self.self_attn
        return getattr(getattr(attention, "_o_allreduce", None), "mnnvl_allreduce", None)

    def _o_proj(self) -> nn.Module:
        """This layer's attention output projection (row parallel)."""
        return self.linear_attn.o_proj if self.is_kda else self.self_attn.mixer.o_proj

    def _o_allreduce(self) -> AllReduce:
        """The all-reduce module of this layer's attention output; it also reduces a wide step's MoE partial."""
        return (self.linear_attn if self.is_kda else self.self_attn)._o_allreduce

    def accepts_moe_partial(self, num_snapshots: int) -> bool:
        """Whether this layer's pre-attention step can reduce the previous layer's unreduced MoE output: the target
        built its collective state, and the snapshot bank is not empty."""
        return num_snapshots > 0 and self.decode_comm is not None

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


class KimiLinearModel(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = _get_text_config(model_config.pretrained_config)
        self._text_cfg = cfg
        dtype = torch.bfloat16

        # Attention and MoE phases are sequential, so their branch-overlap
        # roles share one stream; MoE-internal overlap roles remain separate.
        aux_stream_list = [torch.cuda.Stream() for _ in range(4)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                KimiLinearDecoderLayer(model_config, cfg, layer_idx, self.aux_stream_dict)
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
        # The decode path's GEMVs and embedding (decode_gemv.py), built by the target's cache_derived_state once the
        # weights are final. None: every step embeds on the generic path.
        self.decode_gemvs: Optional[_decode_gemv.K3DecodeGemvs] = None

        # Which convention the drafter tap is on is not recoverable from the
        # served output -- a mismatch only lowers acceptance -- so state it once
        # at construction rather than leaving it to be inferred from an AL.
        logger.info_once(
            "Kimi K3 aux hidden capture: mode="
            f"{'attn_res_stream' if _AUX_ATTN_RES_STREAM_ENABLED else 'prefix_only'} "
            f"({KIMI_K3_AUX_ATTN_RES_STREAM_ENV}={int(_AUX_ATTN_RES_STREAM_ENABLED)})",
            key="kimi_k3_aux_capture_mode",
        )

    # >>> route B: no per-token KDA verify states (kda_token_states): no speculative decoding
    @property
    def kda_token_states(self) -> bool:
        """False: this target decodes without speculation, so the hybrid cache manager keeps no KDA state after
        each verify token, and a captured step's latent-push decision (``_latent_push``) reads that."""
        return False

    # <<< route B

    def _latent_push(self, attn_metadata: AttentionMetadata, step: DecodeStep) -> bool:
        """Whether the MoE layers push on ``step`` (``latent_push``): read only while a CUDA graph is being captured,
        from what the graph's key fixes (the step's shape, the attention metadata the decode kernels read) and from
        load-time state, so every rank and every replay of the graph decides alike."""
        if not torch.cuda.is_current_stream_capturing():
            return False
        kda = [layer.linear_attn for layer in self.layers if layer.is_kda]
        mla = [layer.self_attn for layer in self.layers if not layer.is_kda]
        return latent_push(
            step,
            capturing=True,
            breakable=is_in_breakable_cuda_graph(),
            kda_token_states=self.kda_token_states,
            kda_decode_kernels=all(m.takes_k3_kernels and m.k3_buffers is not None for m in kda),
            mla_decode_branch=all(m.will_run_decode_branch(attn_metadata, step) for m in mla[:1]),
        )

    def _defer_moe_tail(
        self,
        i: int,
        hidden_states: torch.Tensor,
        num_snapshots: int,
        spec_metadata,
        capture_set,
        step: Optional[DecodeStep],
    ) -> bool:
        """Whether layer ``i`` hands its MoE output on unreduced (the row-parallel tail, ``decode_moe.py``): its MoE
        takes the step with that tail, and its consumer, the next layer's pre-attention step or the final norm after
        the last layer, accepts it. A layer DSpark taps defers too: the next layer's step reduces the output, then taps
        its pre-norm mixture. An unknown capture set (every layer tapped) and a tapped last layer keep the replicated
        tail."""
        layer = self.layers[i]
        if not (layer.is_moe and layer.block_sparse_moe.tail_rp_eligible(hidden_states, step)):
            return False
        if spec_metadata is not None and (
            capture_set is None or (layer.layer_idx in capture_set and i == len(self.layers) - 1)
        ):
            return False
        return self.layers[min(i + 1, len(self.layers) - 1)].accepts_moe_partial(num_snapshots)

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

        num_tokens = (input_ids if inputs_embeds is None else inputs_embeds).shape[0]
        step = decode_step(attn_metadata, num_tokens)
        if step is not None and self._latent_push(attn_metadata, step):
            step = replace(step, latent_push=True)
        # A decode step embeds and norms for layer 0 in one launch, the embedding written as layer 0's first snapshot.
        prenormed = None
        if (
            inputs_embeds is None
            and step is not None
            and self.decode_gemvs is not None
            and len(self.layers) > 0
            and self.num_attn_res_snapshots > 0
        ):
            table = self.embed_tokens.weight
            block_residual = table.new_empty(
                self.num_attn_res_snapshots, num_tokens, table.shape[1]
            )
            prenormed = self.decode_gemvs.embed_norm(
                input_ids, table, self.layers[0].input_layernorm, block_residual
            )
        if prenormed is not None:
            hidden_states = prenormed
        else:
            hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
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
        # A MoE layer's output handed on unreduced, which the next layer's pre-attention step (or the final norm)
        # reduces: a decode_comm.PendingTail, or a wide decode step's tensor.
        pending_moe_partial = None
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
            defer_moe_tail = self._defer_moe_tail(
                i, hidden_states, num_snapshots, spec_metadata, capture_set, step
            )
            outputs = layer(
                hidden_states,
                block_residual,
                num_snapshots,
                attn_metadata,
                capture=capture,
                step=step,
                prenormed=i == 0 and prenormed is not None,
                pending_moe_partial=pending_moe_partial,
                defer_moe_tail=defer_moe_tail,
            )
            if defer_moe_tail:
                hidden_states, num_snapshots, pending_moe_partial = outputs
            else:
                (hidden_states, num_snapshots), pending_moe_partial = outputs, None

        if isinstance(pending_moe_partial, _decode_comm.PendingTail):
            normed, _ = self.layers[-1].decode_comm.sandwich_tail(
                pending_moe_partial,
                hidden_states,
                block_residual[:num_snapshots],
                self.output_attn_res_proj,
                self.output_attn_res_norm,
                self.norm,
            )
            return normed
        if pending_moe_partial is not None:
            _, normed = _apply_attn_res_add_and_rmsnorm(
                hidden_states,
                _decode_comm.wide_all_reduce(self.layers[-1]._o_allreduce(), pending_moe_partial),
                block_residual[:num_snapshots],
                self.output_attn_res_proj,
                self.output_attn_res_norm,
                self.norm,
                _attn_res_max_tokens(step),
            )
            return normed

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
                    if num_snapshots > 0 and _AUX_ATTN_RES_STREAM_ENABLED
                    else hidden_states
                )
                spec_metadata.maybe_capture_hidden_states(last.layer_idx, tail, None)

        return _apply_attn_res_and_rmsnorm(
            hidden_states,
            block_residual[:num_snapshots],
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            self.norm,
            _attn_res_max_tokens(step),
        )


# ----------------------------------------------------------------------------------------------------------------------
# The attention modules: the built-in KDA and MLA modules, with the steps the decode kernels take on those kernels.
# ----------------------------------------------------------------------------------------------------------------------


class K3DecodeKDA(KimiKDALinearAttention):
    # >>> route B: no verify kernels
    """Kimi K3's KDA attention: the built-in module, with the decode kernels on the steps they take.

    * A decode step of one token per request runs the fused input projection and the plain decode in one
      ``ssm/k3_kda_decode_attn`` launch.
    * On every step ``decode_step`` classifies, ``o_proj`` runs on the ``o_proj`` decode GEMV site where it takes the
      rows, then the module's all-reduce.

    Every other step runs the built-in module. The kernels read one ``[q | k | v | g | f_a | b]`` weight, built at the
    checkpoint load from the module's own, and the device's ``K3KdaBuffers`` and decode GEMVs' state, which the
    target sets in ``post_load_weights``.
    """

    # <<< route B

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The fused [q | k | v | g | f_a | b | pad] projection weight; the six projections' weights and the built-in
        # fused [q | k | v | g] and [f_a | b | pad] ones are views of it.
        self.k3_proj_weight: Optional[torch.Tensor] = None
        # The fused projection's Lamport buffers: one set per device, shared by every KDA layer.
        self.k3_buffers: Optional[K3KdaBuffers] = None
        # The decode GEMVs' state (decode_gemv.py), shared by the target's layers.
        self.decode_gemvs: Optional[_decode_gemv.K3DecodeGemvs] = None

    @property
    def takes_k3_kernels(self) -> bool:
        """Whether the decode kernels can run this layer: its fused projection weight is built."""
        return self.k3_proj_weight is not None

    def finalize_decode_weights(self) -> None:
        """The built-in fused weights, then one ``[q | k | v | g | f_a | b | pad]`` weight of both."""
        super().finalize_decode_weights()
        assert (
            self.use_full_rank_gate
            and self.gate_lower_bound is not None
            and self._qkvg_proj_weight is not None
            and self._bfa_proj_weight is not None
            and self._qkvg_proj_weight.dtype == self._bfa_proj_weight.dtype == torch.bfloat16
        ), (
            f"Kimi K3 KDA layer {self.layer_idx}: the decode kernels read the bf16 fused projections the built-in "
            "module builds on CUDA at head dim 128, with a full-rank output gate and a gate lower bound"
        )
        rows = self._qkvg_proj_weight.shape[0]
        with torch.no_grad():
            fused = self._merge_projection_weights(
                (self.q_proj, self.k_proj, self.v_proj, self.g_proj, self.f_a_proj, self.b_proj),
                pad_rows_to=8,
            )
        self.k3_proj_weight = fused
        self._qkvg_proj_weight, self._bfa_proj_weight = fused[:rows], fused[rows:]

    def will_run_decode_branch(
        self, attn_metadata: AttentionMetadata, step: Optional[DecodeStep]
    ) -> bool:
        """Whether ``forward`` runs ``step`` on the decode branch: a step ``decode_step`` classifies, outside a
        breakable CUDA graph."""
        return step is not None and not is_in_breakable_cuda_graph()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        step: Optional[DecodeStep] = None,
        reduce_output: bool = True,
        project_output: bool = True,
    ) -> torch.Tensor:
        """The built-in forward where ``will_run_decode_branch`` does not hold. On the decode branch: the plain decode
        on ``ssm/k3_kda_decode_attn`` on a decode step of one token per request, else the built-in dispatch; then
        ``o_proj`` on its decode GEMV site and the TP all-reduce.

        On every path, ``reduce_output=False`` returns ``o_proj``'s TP partial (no all-reduce), and
        ``project_output=False`` the post-o_norm core ``[N, H * 128]`` (no ``o_proj``)."""
        if not self.will_run_decode_branch(attn_metadata, step):
            if reduce_output and project_output:
                return super().forward(hidden_states, attn_metadata)
            core = self._builtin_core(hidden_states, attn_metadata).reshape(-1, self.proj_size)
            return self.o_proj(core) if project_output else core
        if (
            step.decode
            and step.tokens_per_request == 1
            and self.takes_k3_kernels
            and self.k3_buffers is not None
        ):
            core = self._k3_decode(hidden_states[: step.num_tokens], attn_metadata)
        else:
            core = self._forward_impl(hidden_states, attn_metadata)
        if not project_output:
            return core.reshape(-1, self.proj_size)
        return self._k3_project_output(core, reduce_output)

    def _builtin_core(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """The built-in forward's core ``[N, H, 128]``: inside a breakable CUDA graph from the eager
        ``maybe_bcg_kda_core_inplace``, else from ``_forward_impl``."""
        if self.register_to_config and is_in_breakable_cuda_graph():
            core = hidden_states.new_empty(
                (hidden_states.shape[0], self.num_heads, self.head_dim), dtype=torch.bfloat16
            )
            maybe_bcg_kda_core_inplace(hidden_states, self.layer_idx_str, core)
            return core
        return self._forward_impl(hidden_states, attn_metadata)

    def _k3_project_output(self, core: torch.Tensor, reduce_output: bool = True) -> torch.Tensor:
        """``o_proj`` on the ``o_proj`` decode GEMV site where it takes the rows (else the module), then, with
        ``reduce_output``, the TP all-reduce."""
        core2d = core.reshape(-1, self.proj_size)
        out = None
        if self.decode_gemvs is not None:
            out = self.decode_gemvs.project("o_proj", core2d, self.o_proj.weight)
        if out is None:
            if reduce_output:
                return self._project_output(core)
            out = self.o_proj(core2d)
        if reduce_output and self._o_allreduce is not None:
            out = self._o_allreduce(out)
        return out

    def _k3_decode(self, x: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        """``ssm/k3_kda_decode_attn``: the core output ``[R, H, 128]`` of one token of each of the step's R requests;
        each slot's conv window and state advance in place."""
        mamba_metadata = attn_metadata.mamba_metadata
        slots = getattr(mamba_metadata, "generation_state_indices", None)
        if slots is None:
            slots = mamba_metadata.state_indices[: x.shape[0]]
        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        w_q, w_k, w_v = self._get_mtp_conv_weights()
        core = k3_kda_decode_attn(
            x.contiguous(),
            self.k3_proj_weight,
            self.f_b_proj.weight,
            w_q,
            w_k,
            w_v,
            self._A_log_f32,
            self._dt_bias_f32,
            self._onorm_w_f32,
            layer_cache.conv,
            layer_cache.temporal,
            slots,
            self.k3_buffers,
            float(self.gate_lower_bound),
            self.head_k_dim**-0.5,
            float(self.o_norm.eps),
        )
        # Speculative decoding's replay caches keep their committed conv window in step with the pool's.
        self._sync_kda_replay_conv_window(layer_cache, slots, layer_cache.conv)
        return core

    # >>> route B: no verify kernels (a verify step needs speculative decoding)
    # <<< route B


class K3DecodeMLA(KimiK3MLAAttention):
    """Kimi K3's MLA attention: the built-in module, with a decode step's attention on the decode kernels.

    A decode step runs ``x [W_a; W_g]^T`` with the gate columns through a sigmoid (the ``mla_ag`` decode GEMV site
    where it takes the rows, else one GEMM), then ``attention/k3_mla_qkv`` (the q_a / kv_a RMSNorms, q_b and the k_b
    absorption into the fused query, the step's latent rows stored into the paged cache) and
    ``attention/k3_mla_attn_vb_out`` (the attention over the paged cache, v_b and the output gate in one launch), then
    ``o_proj`` (the ``o_proj`` site where it takes the rows, else the module). Every other step, and a decode step whose
    cache the kernels do not read (``k3_mla_decode_view`` says why), runs the built-in module.

    ``[W_a; W_g]`` is built at load from the module's weights, which become views of it. The attention workspace is
    the device's ``K3MlaAttnWorkspace``; it and the decode GEMVs' state are set by the target in
    ``post_load_weights``.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # [W_a; W_g]: the fused q_a / kv_a projection's rows, then the output gate's.
        self.k3_ag_weight: Optional[torch.Tensor] = None
        # The decode attention's workspace: one per device, shared by every MLA layer.
        self.k3_workspace: Optional[K3MlaAttnWorkspace] = None
        # The decode GEMVs' state (decode_gemv.py), shared by the target's layers.
        self.decode_gemvs: Optional[_decode_gemv.K3DecodeGemvs] = None

    def post_load_weights(self) -> None:
        """The built-in post-load, then ``[W_a; W_g]`` (once: CUDA graphs captured since read it)."""
        super().post_load_weights()
        if self.k3_ag_weight is not None:
            return
        gaps = self._k3_layout_gaps()
        assert not gaps, (
            f"Kimi K3 MLA layer {self.layer_idx}: the decode kernels do not take {'; '.join(gaps)}"
        )
        qkv_a, gate = self.kv_a_proj_with_mqa, self.g_proj
        rows = qkv_a.weight.shape[0]
        with torch.no_grad():
            fused = torch.cat([qkv_a.weight, gate.weight])
        qkv_a.weight = nn.Parameter(fused[:rows], requires_grad=False)
        gate.weight = nn.Parameter(fused[rows:], requires_grad=False)
        self.k3_ag_weight = fused

    def _k3_layout_gaps(self) -> list:
        """What of this layer the decode kernels do not take (empty when they take all of it)."""
        if not (self.use_output_gate and self.fuse_qkv_a_proj and not self.is_lite):
            return ["a layer without the output gate or the fused q_a / kv_a projection"]
        linears = (self.kv_a_proj_with_mqa, self.g_proj, self.q_b_proj, self.o_proj)
        checks = (
            (not self.mapping.has_cp_helix(), "helix context parallelism"),
            (not self.apply_rotary_emb and not self.llama_4_scaling, "RoPE or llama-4 scaling"),
            (self.sparse_attn_hooks is None, "sparse attention"),
            (
                self.kv_cache_dtype != "fp8_ds_mla"
                and not getattr(self.mqa, "has_fp8_kv_cache", False)
                and not getattr(self.mqa, "has_fp4_kv_cache", False),
                "a quantized KV cache",
            ),
            (
                all(m.weight.dtype == torch.bfloat16 and m.bias is None for m in linears)
                and self.k_b_proj_trans.dtype == self.v_b_proj.dtype == torch.bfloat16,
                "projections other than bf16 and unbiased",
            ),
            (
                not getattr(self.q_a_layernorm, "is_nvfp4", False)
                and not getattr(self.kv_a_layernorm, "use_gemma", False),
                "an NVFP4 q_a norm or a Gemma kv_a norm",
            ),
            (
                self.num_heads_tp % 6 == 0
                and self.kv_lora_rank == 512
                and self.qk_rope_head_dim == 64
                and self.q_lora_rank == 1536,
                f"{self.num_heads_tp} heads, latent {self.kv_lora_rank}, rope {self.qk_rope_head_dim}, "
                f"q_lora {self.q_lora_rank}",
            ),
        )
        return [why for ok, why in checks if not ok]

    def will_run_decode_branch(
        self, attn_metadata: AttentionMetadata, step: Optional[DecodeStep]
    ) -> bool:
        """Whether ``forward`` runs ``step`` on the decode kernels (see ``_k3_step_view``). Only there does it return
        the gated attention output before ``o_proj``; the KV writes forbid running the attention twice, so a caller
        that needs it asks first."""
        return self._k3_step_view(attn_metadata, step) is not None

    def _k3_step_view(
        self, attn_metadata: AttentionMetadata, step: Optional[DecodeStep]
    ) -> Optional[dict]:
        """The paged-cache view the decode kernels read on ``step``: a decode step, this layer's ``[W_a; W_g]`` and
        workspace built, outside a breakable CUDA graph, and a cache ``k3_mla_decode_view`` takes. Else None."""
        if not (
            step is not None
            and step.decode
            and self.k3_ag_weight is not None
            and self.k3_workspace is not None
            and not is_in_breakable_cuda_graph()
        ):
            return None
        return self._k3_decode_view(attn_metadata, step.num_tokens)

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params=None,
        latent_cache_gen: Optional[torch.Tensor] = None,
        step: Optional[DecodeStep] = None,
        project_output: bool = True,
    ) -> torch.Tensor:
        """The built-in forward, except on a decode step whose cache the decode kernels read
        (``will_run_decode_branch``). There only, ``project_output=False`` returns the gated attention output
        ``[M, H * 128]``, the input of ``o_proj``."""
        view = None if latent_cache_gen is not None else self._k3_step_view(attn_metadata, step)
        if view is None:
            if not project_output:
                raise ValueError("project_output=False needs a step will_run_decode_branch takes")
            return super().forward(
                position_ids, hidden_states, attn_metadata, all_reduce_params, latent_cache_gen
            )
        x = hidden_states[: step.num_tokens].contiguous()
        rows = self.kv_a_proj_with_mqa.weight.shape[0]
        gemvs = self.decode_gemvs
        ag = None if gemvs is None else gemvs.project("mla_ag", x, self.k3_ag_weight)
        if ag is None:
            ag = torch.nn.functional.linear(x, self.k3_ag_weight)
            ag[:, rows:].sigmoid_()
        fused_q = k3_mla_qkv(
            ag,
            self.q_a_layernorm.weight,
            float(self.q_a_layernorm.variance_epsilon),
            self.q_b_proj.weight,
            self.k_b_proj_trans,
            self.kv_a_layernorm.weight,
            float(self.kv_a_layernorm.variance_epsilon),
            view["pool"],
            view["row_stride"],
            view["page_table"],
            view["page_offset"],
            view["seq_len"],
        )
        attn_output = self.create_output(x, 0)
        k3_mla_attn_vb_out(
            fused_q,
            view["pool"],
            view["row_stride"],
            view["page_table"],
            view["page_offset"],
            view["seq_len"],
            view["softmax_scale"],
            self.v_b_proj,
            attn_output,
            self.k3_workspace,
            gate=ag,
            gate_col0=rows,
        )
        if not project_output:
            return attn_output
        out = None if gemvs is None else gemvs.project("o_proj", attn_output, self.o_proj.weight)
        if out is None:
            out = self._project_output(
                [attn_output], position_ids, attn_metadata, all_reduce_params
            )
        return out

    def _k3_decode_view(self, attn_metadata: AttentionMetadata, num_tokens: int) -> Optional[dict]:
        """The paged latent cache as the decode kernels read it this step, or None when they do not read it (the
        reason is logged once)."""
        view = k3_mla_decode_view(self.mqa, attn_metadata, num_tokens)
        if isinstance(view, str):
            logger.info_once(
                f"Kimi K3 MLA: the built-in path for a decode step the decode kernels do not read ({view})",
                key=f"k3_mla_decode_view_{view}",
            )
            return None
        return view


# ----------------------------------------------------------------------------------------------------------------------
# The DSpark drafter: the stock GQA drafter, with a decode step's block on the drafter entries.
# ----------------------------------------------------------------------------------------------------------------------

#: The (requests, block tokens) splits attention/k3_drafter_attn_qknorm certifies; R x 7 is DSpark's block at
#: max_draft_len 7.
DRAFTER_ATTN_SPLITS = frozenset(
    {(1, 1), (1, 8), (2, 4), (4, 2), (8, 1), (3, 1), (2, 8), (8, 8)} | {(r, 7) for r in range(1, 9)}
)
# The drafter layout it certifies: 6 query heads and 1 KV head of 64 per rank, context pages of 64 rows, the q / k
# RMSNorm's epsilon and the NeoX RoPE's base.
_DRAFTER_HEADS = (6, 1)
_DRAFTER_HEAD_DIM = 64
_DRAFTER_PAGE = 64
_DRAFTER_EPS = 1e-5
_DRAFTER_ROPE_BASE = 10000.0


def fc_columns(in_features: int, tp_size: int, tp_rank: int) -> Optional[Tuple[int, int]]:
    """Rank ``tp_rank``'s input columns ``[start, end)`` of the drafter's context projection ``fc`` split by input
    feature over ``tp_size`` ranks: equal contiguous blocks in rank order. None where they do not split evenly."""
    if tp_size < 1 or not 0 <= tp_rank < tp_size or in_features <= 0 or in_features % tp_size:
        return None
    width = in_features // tp_size
    return tp_rank * width, (tp_rank + 1) * width


class K3FcSlice(nn.Module):
    """This rank's block of the DSpark drafter's context projection ``fc`` (`fc_columns`): ``weight`` is
    ``fc.weight[:, start:end]``, contiguous. Its output is this rank's partial product; the sum over the TP group's
    ranks is ``fc``'s output."""

    def __init__(self, weight: torch.Tensor, start: int, end: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.start = start
        self.end = end

    @classmethod
    def of(cls, fc_weight: torch.Tensor, tp_size: int, tp_rank: int) -> Optional["K3FcSlice"]:
        """Rank ``tp_rank``'s block of the full ``fc_weight`` ``[out_features, in_features]``: a copy of its columns
        (on one rank, the weight itself); None where the input columns do not split evenly over ``tp_size`` ranks."""
        columns = fc_columns(fc_weight.shape[1], tp_size, tp_rank)
        if columns is None:
            return None
        start, end = columns
        return cls(fc_weight.detach()[:, start:end].contiguous(), start, end)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_states[:, start:end] @ weight.T`` for the full features ``hidden_states`` ``[N, in_features]``: the
        GEMM reads the column block in place, through the rows' stride."""
        return torch.nn.functional.linear(hidden_states[:, self.start : self.end], self.weight)


class K3DSparkDrafter(GQADSparkForCausalLM):
    """Kimi K3's DSpark drafter: the stock GQA drafter, with a decode step's block forward on the drafter entries.

    A block the entries take runs, per layer:

    * the fused q / k / v projection on the ``drafter_qkv`` decode GEMV site;
    * ``attention/k3_drafter_attn_qknorm``: the q / k RMSNorm, the NeoX RoPE, and each request's block attending to
      its paged context and to its own k / v, in one launch. The block's k / v are not stored in the cache: the
      worker writes the accepted rows' context k / v before a block reads them;
    * the output projection on ``drafter_o``, then the module's all-reduce;
    * gate / up on ``drafter_gate_up`` and the down projection with the SiLU-and-mul on ``drafter_down``, then the
      module's all-reduce.

    The norms and the residual adds are the stock modules' unless the drafter runs on the TP group's collective state
    (below); a projection whose site does not take its rows runs its module. Every other block runs the stock
    ``dflash_forward``: a split `DRAFTER_ATTN_SPLITS` does not list, another attention backend, head layout, RoPE or
    normalization, a cache the kernel does not read, or a compile key of the attention that has not run eagerly, under
    CUDA-graph capture.

    Where the target hands the drafter the TP group's collective state (``use_decode_comm``):

    * the context projection's ``fc`` holds only this rank's block of input columns (`K3FcSlice`), and
      ``project_target_hidden`` sums the ranks' partial products with ``hidden_norm`` in one all-reduce;
    * a block the entries take runs each residual add and RMSNorm in the all-reduce before it (o_proj's with the
      post-attention norm, the MLP's with the next layer's input norm, the last one's with the final norm), the
      block's input serving as the first residual, uncopied. Up to 8 rows ``comm/k3_sandwich_plain`` runs the
      projection, its all-reduce and the norm in one launch (o_proj, and with the SiLU-and-mul the down projection
      after ``drafter_gate_up``); otherwise the projection (its site, or its module without the all-reduce) is
      followed by ``comm/mnnvl_fusion_allreduce``. A block whose rows the MNNVL workspace does not hold, or layers
      whose norms or projections the fused all-reduces do not reproduce, keep the stock all-reduces and norms.

    Otherwise ``fc`` stays replicated, with the stock context projection, and the norms and residual adds are the
    stock modules'. The worker that calls the drafter (the stock ``DSparkWorker``), the context k / v and the Markov
    head stay upstream code.
    """

    def __init__(
        self, draft_config: ModelConfig, *, dflash_attention_backend: str = "AUTO"
    ) -> None:
        super().__init__(draft_config, dflash_attention_backend=dflash_attention_backend)
        # The decode GEMVs' state (decode_gemv.py), shared by the target's layers; set by the target once built.
        self.decode_gemvs: Optional[_decode_gemv.K3DecodeGemvs] = None
        # Whether every layer is one the drafter entries take; decided on the first block, the weights loaded.
        self._k3_take: Optional[bool] = None
        # The attention's compile keys that ran eagerly ((more than one request, page stride)); a capture takes
        # only these.
        self._k3_attn_ran: set = set()
        # The TP group's collective state (decode_comm.py), set by the target where it builds one (use_decode_comm).
        self.decode_comm: Optional[_decode_comm.K3DecodeComm] = None
        # The zero residual of the split context projection's all-reduce, up to a decode step's rows.
        self._k3_zero_rows: Optional[torch.Tensor] = None
        # Whether a block's residual adds and RMSNorms run in its all-reduces, and the comm/k3_sandwich_plain forms
        # ("o_proj", "down") compiled for the layers' shapes; set with the collective state.
        self._k3_norms_fuse = False
        self._k3_sandwich_forms: frozenset = frozenset()

    def use_decode_comm(self, comm: _decode_comm.K3DecodeComm) -> None:
        """Run the drafter's collectives on ``comm``, the TP group's decode state (``decode_comm.K3DecodeComm``) the
        target builds where every attention all-reduce runs over MNNVL. The target calls it once the weights are
        loaded, before any CUDA-graph capture. Collective: every rank of the group calls it at the same point.

        * The context projection's ``fc`` keeps only this rank's block of input columns (`K3FcSlice`), where they
          split evenly over the group and the drafter's layers have their TP all-reduce: ``project_target_hidden``
          sums the ranks' partial products and applies ``hidden_norm`` in one ``comm/mnnvl_fusion_allreduce`` call.
        * Where the layers' norms and projections take it (`_k3_norms_take_comm`), a block's residual adds and
          RMSNorms run in its all-reduces (``_k3_block_forward``). Each form of ``comm/k3_sandwich_plain`` whose
          shape every layer shares compiles here, with one call on a zero row of a zero weight; a form that does not
          compile is not used."""
        self.decode_comm = comm
        self._k3_split_fc()
        self._k3_norms_fuse = self._k3_norms_take_comm()
        forms = set()
        if self._k3_norms_fuse:
            layers = self.model.layers
            for form, swiglu, weights in (
                ("o_proj", False, [layer.self_attn.o_proj.weight for layer in layers]),
                ("down", True, [layer.mlp.down_proj.weight for layer in layers]),
            ):
                if len({tuple(w.shape) for w in weights}) == 1 and comm.compile_plain(
                    weights[0], swiglu=swiglu
                ):
                    forms.add(form)
        self._k3_sandwich_forms = frozenset(forms)
        logger.info(
            "Kimi K3 DSpark drafter: residual adds and RMSNorms "
            + (
                f"in the all-reduces (k3_sandwich_plain: {sorted(forms) or 'none'}, else "
                "mnnvl_fusion_allreduce)"
                if self._k3_norms_fuse
                else "on the stock modules (a layer norm or projection the fused all-reduces do not reproduce)"
            )
        )

    def _k3_norms_take_comm(self) -> bool:
        """Whether a block's residual adds and RMSNorms can run in its all-reduces: every layer's input and
        post-attention norms and the final norm are plain bf16 RMSNorms of the hidden width (no Gemma offset, no
        quantized output), and every output and down projection is row parallel with its all-reduce."""
        hidden = self.config.hidden_size

        def plain(norm: nn.Module) -> bool:
            weight = getattr(norm, "weight", None)
            return (
                isinstance(weight, torch.Tensor)
                and weight.dtype == torch.bfloat16
                and tuple(weight.shape) == (hidden,)
                and hasattr(norm, "variance_epsilon")
                and not getattr(norm, "use_gemma", True)
                and not getattr(norm, "is_nvfp4", True)
                and not getattr(norm, "return_hp_output", True)
            )

        def reduces(linear: nn.Module) -> bool:
            return (
                getattr(getattr(linear, "tp_mode", None), "name", None) == "ROW"
                and getattr(linear, "reduce_output", False)
                and getattr(linear, "all_reduce", None) is not None
            )

        return plain(self.model.norm) and all(
            plain(layer.input_layernorm)
            and plain(layer.post_attention_layernorm)
            and reduces(layer.self_attn.o_proj)
            and reduces(layer.mlp.down_proj)
            for layer in self.model.layers
        )

    def _k3_tp_all_reduce(self) -> Optional[nn.Module]:
        """The drafter's own TP all-reduce (its first output projection's row-parallel all-reduce module), or None."""
        o_proj = self.model.layers[0].self_attn.o_proj
        if getattr(getattr(o_proj, "tp_mode", None), "name", None) != "ROW":
            return None
        return getattr(o_proj, "all_reduce", None)

    def _k3_split_fc(self) -> None:
        """Replace the replicated ``fc`` with this rank's block of its input columns (`K3FcSlice`) and size the zero
        residual of the projection's all-reduce, where the drafter runs on the TP group's collective state, ``fc`` is
        a bias-free bf16 projection whose input columns split evenly over the group, and the drafter has its TP
        all-reduce (for rows the MNNVL workspace does not hold). Otherwise ``fc`` stays replicated."""
        fc = getattr(self, "fc", None)
        if self.decode_comm is None or fc is None or isinstance(fc, K3FcSlice):
            return
        mapping = self.model_config.mapping
        weight = getattr(fc, "weight", None)
        sliced = None
        if (
            isinstance(weight, torch.Tensor)
            and weight.dim() == 2
            and weight.dtype == torch.bfloat16
            and getattr(fc, "bias", None) is None
            and self._k3_tp_all_reduce() is not None
        ):
            sliced = K3FcSlice.of(weight, mapping.tp_size, mapping.tp_rank)
        if sliced is None:
            logger.info(
                "Kimi K3 DSpark drafter: fc stays replicated (it is not a bias-free bf16 projection whose input "
                f"columns split evenly over TP{mapping.tp_size}, or the layers have no TP all-reduce)"
            )
            return
        self.fc = sliced
        self._k3_zero_rows = weight.new_zeros(
            MAX_REQUESTS * MAX_TOKENS_PER_REQUEST, weight.shape[0]
        )
        logger.info(
            f"Kimi K3 DSpark drafter: fc split by input feature over TP{mapping.tp_size}: rank {mapping.tp_rank} "
            f"holds columns [{sliced.start}, {sliced.end}), hidden_norm in the all-reduce"
        )

    def load_weights(self, weights, weight_mapper=None, **kwargs):
        """The stock load; on the TP group's collective state, ``fc`` is split again (`_k3_split_fc`)."""
        result = super().load_weights(weights, weight_mapper=weight_mapper, **kwargs)
        self._k3_split_fc()
        return result

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_norm(fc(hidden_states))`` of the captured target features ``[N, in_features]``.

        With ``fc`` split (`K3FcSlice`): this rank's partial product of its columns, then the sum over the TP group
        with ``hidden_norm`` applied in one ``comm/mnnvl_fusion_allreduce`` call (a zero residual) up to a decode
        step's rows the MNNVL workspace holds; more rows (a prefill) go through the drafter's TP all-reduce, then
        ``hidden_norm``. Otherwise the stock projection."""
        fc = self.fc
        if not isinstance(fc, K3FcSlice):
            return super().project_target_hidden(hidden_states)
        partial = fc(hidden_states.to(fc.weight.dtype))
        rows = partial.shape[0]
        zeros = self._k3_zero_rows
        if rows <= zeros.shape[0] and self.decode_comm.takes_allreduce_norm(rows, partial.shape[1]):
            normed, _ = self.decode_comm.allreduce_norm(partial, zeros[:rows], self.hidden_norm)
            return normed
        if rows == 0:
            return self.hidden_norm(partial)
        return self.hidden_norm(self._k3_tp_all_reduce()(partial))

    def dflash_forward(
        self,
        noise_embedding: torch.Tensor,
        query_positions: torch.Tensor,
        num_ctx_per_req: torch.Tensor,
        ctx_k_cache: torch.Tensor,
        ctx_v_cache: torch.Tensor,
        ctx_cache_batch_idx: torch.Tensor,
        ctx_kv_cache: Optional[torch.Tensor] = None,
        ctx_page_table: Optional[torch.Tensor] = None,
        ctx_rows_start: Optional[int] = None,
    ) -> torch.Tensor:
        """The block's hidden states ``[B * block, hidden]``: on the drafter entries where they take it (the class
        docstring), else the stock forward. ``ctx_rows_start``: where ``ctx_cache_batch_idx`` is the contiguous rows
        ``[ctx_rows_start, ctx_rows_start + B)`` of ``ctx_page_table``, their start; the entries then read those rows
        as a view instead of gathering them."""
        keys = self._k3_block_keys(noise_embedding, ctx_kv_cache, ctx_page_table)
        if keys is None:
            return super().dflash_forward(
                noise_embedding,
                query_positions,
                num_ctx_per_req,
                ctx_k_cache,
                ctx_v_cache,
                ctx_cache_batch_idx,
                ctx_kv_cache,
                ctx_page_table,
            )
        out = self._k3_block_forward(
            noise_embedding,
            query_positions,
            num_ctx_per_req,
            ctx_cache_batch_idx,
            ctx_kv_cache,
            ctx_page_table,
            ctx_rows_start,
        )
        if not torch.cuda.is_current_stream_capturing():
            self._k3_attn_ran |= keys
        return out

    def _k3_layers_take(self) -> bool:
        """Whether every layer is one the drafter entries take: plain NeoX RoPE with the q / k RMSNorm at the head
        layout, epsilon and RoPE base the attention certifies, bias-free projections, a plain SiLU-and-mul MLP,
        non-causal unwindowed block attention, and no block convolution or post-attention gate."""
        if self._k3_take is None:
            if self._fused_kv_weight is None:
                self._build_fused_kv_buffers()
            take = (
                self._use_fused_qk_norm_rope
                and not self.has_block_conv
                and type(self)._post_attention_gate is DFlashForCausalLM._post_attention_gate
            )
            for layer_idx, layer in enumerate(self.model.layers):
                attn, mlp = layer.self_attn, layer.mlp
                pos = getattr(attn, "pos_embd_params", None)
                rope = getattr(pos, "rope", None)
                norms = (attn.q_norm, attn.k_norm)
                take = take and (
                    (attn.num_heads, attn.num_key_value_heads) == _DRAFTER_HEADS
                    and attn.head_dim == _DRAFTER_HEAD_DIM
                    and getattr(attn, "is_qk_norm", False)
                    and not getattr(attn, "use_gemma_rms_norm", True)
                    and rope is not None
                    and pos.is_neox
                    and getattr(pos, "mrope_section", None) is None
                    and rope.theta == _DRAFTER_ROPE_BASE
                    and all(
                        norm.variance_epsilon == _DRAFTER_EPS
                        and norm.weight.dtype == torch.bfloat16
                        and tuple(norm.weight.shape) == (_DRAFTER_HEAD_DIM,)
                        for norm in norms
                    )
                    and attn.qkv_proj.bias is None
                    and attn.o_proj.bias is None
                    and mlp.activation is torch.nn.functional.silu
                    and mlp.swiglu_limit is None
                    and mlp.swiglu_alpha in (None, 1.0)
                    and mlp.swiglu_beta in (None, 0.0)
                    and mlp.gate_up_proj.bias is None
                    and mlp.down_proj.bias is None
                    and self._resolve_block_attention(layer_idx) == (False, (-1, -1))
                )
            self._k3_take = bool(take)
            logger.info(
                f"Kimi K3 DSpark drafter: {len(self.model.layers)} layers, decode blocks "
                + (
                    "on k3_drafter_attn_qknorm and the drafter GEMV sites"
                    if self._k3_take
                    else "on the stock block forward (a layer the drafter entries do not take)"
                )
            )
        return self._k3_take

    def _k3_block_keys(
        self,
        noise_embedding: torch.Tensor,
        ctx_kv_cache: Optional[torch.Tensor],
        ctx_page_table: Optional[torch.Tensor],
    ) -> Optional[frozenset]:
        """The attention's compile keys for this block when the drafter entries take it, else None."""
        if (
            self.dflash_attention_backend != "TRTLLM"
            or ctx_kv_cache is None
            or ctx_page_table is None
            or noise_embedding.dtype != torch.bfloat16
            or tuple(noise_embedding.shape[:2]) not in DRAFTER_ATTN_SPLITS
            or is_in_breakable_cuda_graph()
            or not self._k3_layers_take()
        ):
            return None
        num_kv_heads = self.model.layers[0].self_attn.num_key_value_heads
        page = (2, num_kv_heads, _DRAFTER_PAGE, _DRAFTER_HEAD_DIM)
        dense = (
            num_kv_heads * _DRAFTER_PAGE * _DRAFTER_HEAD_DIM,
            _DRAFTER_PAGE * _DRAFTER_HEAD_DIM,
        )
        strides = set()
        for layer_idx in range(len(self.model.layers)):
            cache = ctx_kv_cache[layer_idx]
            if (
                cache.dtype != torch.bfloat16
                or cache.dim() != 5
                or tuple(cache.shape[1:]) != page
                or cache.stride()[1:] != (*dense, _DRAFTER_HEAD_DIM, 1)
                or cache.stride(0) % 8
            ):
                return None
            strides.add(cache.stride(0))
        keys = frozenset((noise_embedding.shape[0] > 1, stride) for stride in strides)
        if torch.cuda.is_current_stream_capturing() and not keys <= self._k3_attn_ran:
            return None
        return keys

    def _k3_block_forward(
        self,
        noise_embedding: torch.Tensor,
        query_positions: torch.Tensor,
        num_ctx_per_req: torch.Tensor,
        ctx_cache_batch_idx: torch.Tensor,
        ctx_kv_cache: torch.Tensor,
        ctx_page_table: torch.Tensor,
        ctx_rows_start: Optional[int] = None,
    ) -> torch.Tensor:
        batch, block = noise_embedding.shape[:2]
        rows = batch * block
        ctx_len = num_ctx_per_req[:batch].to(torch.int32)
        if ctx_rows_start is None:
            page_table = ctx_page_table.index_select(0, ctx_cache_batch_idx.to(torch.long))
        else:
            # The batch's page-table rows are one contiguous run: a view, no gather.
            page_table = ctx_page_table[ctx_rows_start : ctx_rows_start + batch]
        positions = query_positions.reshape(-1).contiguous()
        hidden = noise_embedding.reshape(rows, -1)
        layers = self.model.layers
        comm = self._k3_fused_comm(hidden)
        residual = None
        if comm is not None:
            # The fused all-reduces read the residual and return the updated one: the block's input serves as the
            # first residual, uncopied.
            residual = hidden
            normed = layers[0].input_layernorm(hidden)
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            if comm is None:
                if residual is None:
                    residual = hidden.clone()
                    normed = layer.input_layernorm(hidden)
                else:
                    normed, residual = layer.input_layernorm(hidden, residual)
            qkv = self._k3_project("drafter_qkv", normed, attn.qkv_proj)
            out = torch.empty(rows, attn.q_size, dtype=torch.bfloat16, device=qkv.device)
            k3_drafter_attn_qknorm(
                qkv,
                attn.q_norm.weight,
                attn.k_norm.weight,
                positions,
                attn.q_norm.variance_epsilon,
                attn.pos_embd_params.rope.theta,
                ctx_kv_cache[layer_idx],
                page_table,
                ctx_len,
                attn.num_heads,
                attn.num_key_value_heads,
                out,
            )
            if comm is not None:
                # o_proj's all-reduce applies the post-attention norm; the MLP's the next layer's input norm, after
                # the last layer the final norm.
                next_norm = (
                    layers[layer_idx + 1].input_layernorm
                    if layer_idx + 1 < len(layers)
                    else self.model.norm
                )
                normed, residual = self._k3_project_norm(
                    comm, out, attn.o_proj, residual, layer.post_attention_layernorm
                )
                normed, residual = self._k3_mlp_norm(comm, layer.mlp, normed, residual, next_norm)
                continue
            hidden = self._k3_project("drafter_o", out, attn.o_proj)
            hidden, residual = layer.post_attention_layernorm(hidden, residual)
            hidden = self._k3_mlp(layer.mlp, hidden)
        if comm is not None:
            return normed
        out, _ = self.model.norm(hidden, residual)
        return out

    def _k3_fused_comm(self, hidden: torch.Tensor) -> Optional[_decode_comm.K3DecodeComm]:
        """The TP group's collective state where a block of rows ``hidden`` runs its residual adds and RMSNorms in its
        all-reduces: the drafter runs on it (``use_decode_comm``) with layers that take it, and its MNNVL workspace
        holds an all-reduce of the block's rows; else None (the stock all-reduces and norms)."""
        comm = self.decode_comm
        if comm is None or not self._k3_norms_fuse or not comm.takes_allreduce_norm(*hidden.shape):
            return None
        return comm

    def _k3_project_norm(
        self,
        comm: _decode_comm.K3DecodeComm,
        x: torch.Tensor,
        linear: nn.Module,
        residual: torch.Tensor,
        norm: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(norm(updated), updated)``, ``updated = residual + linear(x)`` for the row-parallel output projection,
        the residual add and ``norm`` in its all-reduce: ``comm/k3_sandwich_plain`` where it takes the rows (the
        projection in the ``drafter_o`` site's arithmetic, in the same launch), else the projection on ``drafter_o``
        where that takes the rows, or the module without its all-reduce, then ``comm/mnnvl_fusion_allreduce``."""
        if "o_proj" in self._k3_sandwich_forms and comm.takes_plain(
            x, linear.weight, residual, norm
        ):
            return comm.sandwich_plain(x, linear.weight, residual, norm)
        gemvs = self.decode_gemvs
        partial = None if gemvs is None else gemvs.project("drafter_o", x, linear.weight)
        if partial is None:
            partial = linear(x, all_reduce_params=_decode_comm.skip_all_reduce())
        return comm.allreduce_norm(partial, residual, norm)

    def _k3_mlp_norm(
        self,
        comm: _decode_comm.K3DecodeComm,
        mlp: nn.Module,
        x: torch.Tensor,
        residual: torch.Tensor,
        norm: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(norm(updated), updated)``, ``updated = residual + mlp(x)``, the residual add and ``norm`` in the down
        projection's all-reduce: gate / up on ``drafter_gate_up``, then ``comm/k3_sandwich_plain``'s SiLU-and-mul
        form where it takes the rows (the ``drafter_down`` site's arithmetic), else the SiLU-and-mul and down
        projection on ``drafter_down``; where those sites do not take the rows, the module without its all-reduce;
        then ``comm/mnnvl_fusion_allreduce``."""
        gemvs = self.decode_gemvs
        gate_up = (
            None if gemvs is None else gemvs.project("drafter_gate_up", x, mlp.gate_up_proj.weight)
        )
        partial = None
        if gate_up is not None:
            if "down" in self._k3_sandwich_forms and comm.takes_plain(
                gate_up, mlp.down_proj.weight, residual, norm, swiglu=True
            ):
                return comm.sandwich_plain(
                    gate_up, mlp.down_proj.weight, residual, norm, swiglu=True
                )
            partial = gemvs.project("drafter_down", gate_up, mlp.down_proj.weight)
        if partial is None:
            partial = mlp(x, final_all_reduce_params=_decode_comm.skip_all_reduce())
        return comm.allreduce_norm(partial, residual, norm)

    def _k3_project(self, site: str, x: torch.Tensor, linear: nn.Module) -> torch.Tensor:
        """``linear(x)``, its GEMM on the ``site`` decode GEMV where that takes the rows, then a row-parallel
        projection's all-reduce."""
        y = None if self.decode_gemvs is None else self.decode_gemvs.project(site, x, linear.weight)
        if y is None:
            return linear(x)
        return _row_parallel_reduce(linear, y)

    def _k3_mlp(self, mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """``mlp(x)``: gate / up on ``drafter_gate_up`` and the SiLU-and-mul with the down projection on
        ``drafter_down`` where both take the rows, then the down projection's all-reduce."""
        gemvs = self.decode_gemvs
        gate_up = (
            None if gemvs is None else gemvs.project("drafter_gate_up", x, mlp.gate_up_proj.weight)
        )
        down = (
            None
            if gate_up is None
            else gemvs.project("drafter_down", gate_up, mlp.down_proj.weight)
        )
        if down is None:
            return mlp(x)
        return _row_parallel_reduce(mlp.down_proj, down)


def _row_parallel_reduce(linear: nn.Module, partial: torch.Tensor) -> torch.Tensor:
    """``partial`` all-reduced as ``linear`` reduces its output: a row-parallel projection's all-reduce, if any."""
    all_reduce = getattr(linear, "all_reduce", None)
    if getattr(getattr(linear, "tp_mode", None), "name", None) == "ROW" and all_reduce is not None:
        return all_reduce(partial)
    return partial


# ----------------------------------------------------------------------------------------------------------------------
# The target: step classification, the construction checks and the registration shell.
# ----------------------------------------------------------------------------------------------------------------------


def _text_model_config(model_config: ModelConfig) -> ModelConfig:
    """The language model's ModelConfig: the checkpoint's text_config, with quant exclusions renamed to match.

    The checkpoint names its language-model modules `language_model.<x>`; the text model's are `<x>`, with
    `layers.*` under `model.`.
    """
    config = model_config.pretrained_config
    text = copy.copy(model_config)
    text._frozen = False
    text.pretrained_config = config.text_config
    excluded = text.quant_config.exclude_modules
    if excluded:
        text.quant_config = copy.copy(text.quant_config)
        renamed = []
        for name in excluded:
            if name.startswith(_LANG_PREFIX):
                name = name[len(_LANG_PREFIX) :]
                if name.startswith("layers."):
                    name = "model." + name
            renamed.append(name)
        text.quant_config.exclude_modules = renamed
    text.skip_create_weights_in_init = True
    text._frozen = True
    return text


@dataclass(frozen=True)
class DecodeStep:
    """A step the Kimi K3 decode kernels take: ``num_tokens`` rows and, on a pure decode step, ``num_requests``
    generation requests of ``tokens_per_request`` tokens each (None on a step with context requests)."""

    num_tokens: int
    num_requests: Optional[int] = None
    tokens_per_request: Optional[int] = None
    #: Whether the MoE layers' latent all-reduce is the routed experts' push form plus ``comm/k3_latent_reduce``
    #: (``decode_moe.py``): set by ``KimiLinearModel.forward`` from ``latent_push``.
    latent_push: bool = False

    @property
    def small(self) -> bool:
        """Whether the step fits one token tile of the token-count kernels."""
        return self.num_tokens <= DECODE_MAX_TOKENS

    @property
    def decode(self) -> bool:
        """Whether the step is a pure decode step the request-aware kernels take."""
        return self.num_requests is not None

    @property
    def wide(self) -> bool:
        """Whether the step is a pure decode step of more than one token tile: its token-count work keeps the decode
        layout's MoE head and tail, on M-general ops."""
        return self.decode and not self.small


def latent_push(
    step: Optional[DecodeStep],
    *,
    capturing: bool,
    breakable: bool,
    kda_token_states: bool,
    kda_decode_kernels: bool,
    mla_decode_branch: bool,
) -> bool:
    """Whether a step's MoE layers push their routed partials (``DecodeStep.latent_push``): a pure decode step of at
    most 8 tokens whose attention layers all run the decode kernels, captured into a CUDA graph and not inside a
    breakable one. The KDA layers run them on one token per request with every KDA layer taking the K3 kernels
    (``kda_decode_kernels``), and on every verify width with the per-token states (``kda_token_states``); the MLA
    layers where their decode branch takes the step (``mla_decode_branch``, the same for every MLA layer of a step).
    Every other step keeps the routed experts' all-reduce: the exchange's call-order invariant needs every kernel
    between a reduce and the next push to wait for its predecessor, which only those kernels were checked for, and a
    graph launch orders every pushing replay behind whatever ran before it."""
    if step is None or not (step.decode and step.small) or not capturing or breakable:
        return False
    if not (kda_token_states or (step.tokens_per_request == 1 and kda_decode_kernels)):
        return False
    return mla_decode_branch


def decode_step(attn_metadata: AttentionMetadata, num_tokens: int) -> Optional[DecodeStep]:
    """The step's shape if any Kimi K3 decode kernel takes it, else None (the generic path runs).

    ``num_tokens`` is the step's token count (the rows of the model input). Read on the host from per-step integers
    and the host copy of the sequence lengths only. A CUDA graph is captured per decode batch shape, and every input
    here is fixed by that shape, so a captured step and its replays are classified alike.
    """
    if num_tokens <= 0:
        return None
    requests = _decode_requests(attn_metadata, num_tokens)
    if requests is not None:
        return DecodeStep(num_tokens, requests, num_tokens // requests)
    if num_tokens <= DECODE_MAX_TOKENS:
        return DecodeStep(num_tokens)
    return None


def _decode_requests(attn_metadata: AttentionMetadata, num_tokens: int) -> Optional[int]:
    """R when the step is R <= 8 generation requests of the same T <= 8 tokens and no context request, else None."""
    if attn_metadata.num_contexts != 0:
        return None
    requests = attn_metadata.num_generations
    if not 0 < requests <= MAX_REQUESTS or num_tokens % requests != 0:
        return None
    tokens = num_tokens // requests
    if not 0 < tokens <= MAX_TOKENS_PER_REQUEST:
        return None
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is not None and seq_lens.device.type == "cpu":
        lens = seq_lens[:requests]
        if lens.numel() != requests or bool((lens != tokens).any()):
            return None
    return requests


def _check_construction(model_config: ModelConfig) -> None:
    """The settings this target is built for that are fixed before the first step."""
    capability = torch.cuda.get_device_capability()
    assert capability == _SM, (
        f"this target is certified on sm_{_SM[0]}{_SM[1]}, running on sm_{capability[0]}{capability[1]}"
    )
    mapping = model_config.mapping
    topology = (
        mapping.world_size,
        mapping.tp_size,
        mapping.pp_size,
        mapping.moe_tp_size,
        mapping.moe_ep_size,
        mapping.enable_attention_dp,
    )
    # >>> route B: this target's topology, its expert split set explicitly, and no speculative decoding
    assert topology == (16, 16, 1, 16, 1, False), (
        "the tp16_moetp16ep1 target needs world_size 16, tensor_parallel_size 16, pipeline_parallel_size 1, "
        "moe_tensor_parallel_size 16, moe_expert_parallel_size 1 and enable_attention_dp false; the engine built "
        f"(world, tp, pp, moe_tp, moe_ep, attention_dp) = {topology}"
    )
    assert mapping.moe_tp_ep_user_specified, (
        "the tp16_moetp16ep1 target needs moe_tensor_parallel_size 16 and moe_expert_parallel_size 1 set "
        "explicitly; with the split unset the engine resolves the same sizes, but Kimi K3 then runs the experts "
        "expert-parallel over the 16 ranks"
    )
    assert getattr(model_config, "spec_config", None) is None, (
        "the tp16_moetp16ep1 target decodes without speculation; for DSpark, DFlash or SA set "
        "moe_tensor_parallel_size 4 and moe_expert_parallel_size 4 (the tp16_moetp4ep4 target)"
    )
    # <<< route B
    assert model_config.torch_dtype == torch.bfloat16, (
        f"this target computes in bf16; the engine resolved dtype {model_config.torch_dtype}"
    )
    kv_algo = model_config.quant_config.kv_cache_quant_algo
    assert kv_algo is None, (
        f"this target's MLA kernels read a bf16 KV pool; kv_cache_config.dtype resolved to {kv_algo}"
    )
    quant_algo = model_config.quant_config.quant_algo
    assert quant_algo == QuantAlgo.W4A16_MXFP4 and not model_config.quant_config_dict, (
        "this target loads the MXFP4 checkpoint, whose compressed-tensors config the model config reads as "
        "W4A16_MXFP4 with no per-layer declarations (its routed experts run the W4A8_MXFP4_MXFP8 default); the "
        f"engine read {quant_algo} with {len(model_config.quant_config_dict or {})} per-layer declarations"
    )
    strategy = model_config.allreduce_strategy
    assert strategy in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL), (
        f"this target runs its all-reduces over MNNVL; allreduce_strategy is {strategy.name}"
    )


# >>> route B: this target's name
@register_auto_model("ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1")
class ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1(KimiLinearForCausalLM):
    # <<< route B
    """The registration shell: this target's text model (`KimiLinearModel` above) behind its checks.

    It inherits the built-in Kimi K3 causal LM for the checkpoint load and the engine hooks (`load_weights` through
    weights.py, the KDA metadata class, the model defaults), which walk the model by its module names; the text model
    keeps the built-in one's.
    """

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config: Any = None) -> Literal["V2"]:
        """The V2 hybrid manager holds the KDA states; the step contract requires it."""
        return "V2"

    def __init__(self, model_config: ModelConfig):
        config = model_config.pretrained_config
        assert getattr(config, "text_config", None) is not None, (
            "this target loads the KimiK3ForConditionalGeneration checkpoint, whose language model is its "
            "text_config"
        )
        _check_construction(model_config)
        text = _text_model_config(model_config)
        # >>> route B: no speculative decoding (_check_construction), so the one-engine shell builds no drafter
        # <<< route B
        # The inherited loader reads these: this target has neither helix context parallelism nor the fp8
        # weight-read conversion of the shared / latent MLPs.
        self._fp8_weight_read_moe_mlp = False
        self.mapping_with_cp = None
        self._repurposed_tp_mapping = None
        cfg = text.pretrained_config
        SpecDecOneEngineForCausalLM.__init__(
            self,
            KimiLinearModel(text),
            text,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        self._step_checked = False
        # >>> route B: no drafter to hand the LM head to
        # The LM head on gemm/k3_head_gemv at decode size, once cache_derived_state has built its state.
        self.logits_processor = _decode_gemv.K3LogitsProcessor(self.logits_processor)
        # <<< route B
        # The executor reads generation settings (eos_token_id, ...) off the model config the engine holds, which
        # must therefore be the text config, as the built-in wrapper leaves it.
        model_config._frozen = False
        model_config.pretrained_config = self.config
        model_config._frozen = True

    def _build_draft_model(
        self, model_config: ModelConfig, draft_config: Optional[ModelConfig]
    ) -> Optional[nn.Module]:
        """DSpark's drafter from a standalone GQA checkpoint is this target's `K3DSparkDrafter`; any other drafter
        is the mode registry's."""
        spec_config = model_config.spec_config
        if (
            spec_config.spec_dec_mode.is_dspark()
            and draft_config is not None
            and not draft_is_embedded_in_target(model_config)
            and not is_mla(draft_config.pretrained_config)
        ):
            return K3DSparkDrafter(
                draft_config, dflash_attention_backend=spec_config.attention_backend
            )
        return super()._build_draft_model(model_config, draft_config)

    def load_weights(self, weights, *args, **kwargs):
        _weights.load(self, weights)

    def cache_derived_state(self) -> None:
        """Build the decode GEMVs' state once the weights are final: the LM head's workspace, and one eager call of
        every decode GEMV kernel at its site's shape (decode_gemv.SITES), so none compiles under capture. Built once:
        a later call keeps it, since CUDA graphs captured in between hold its workspace. The DSpark drafter's sites
        are among them, and the drafter gets the state too."""
        super().cache_derived_state()
        if self.model.decode_gemvs is not None:
            return
        gemvs = _decode_gemv.K3DecodeGemvs.create(self.lm_head)
        self.model.decode_gemvs = gemvs
        self.logits_processor.gemvs = gemvs
        for layer in self.model.layers:
            if not layer.is_moe:
                layer.decode_gemvs = gemvs
        if isinstance(self.draft_model, K3DSparkDrafter):
            self.draft_model.decode_gemvs = gemvs

    def post_load_weights(self) -> None:
        """The state the decode kernels share, built once per device before any CUDA-graph capture and handed to
        every layer of its kind: the KDA projection's Lamport buffers and the MLA decode attention's workspace; the
        decode GEMVs' state (built by ``cache_derived_state``) handed to every attention module; and, where every
        attention all-reduce runs over MNNVL, the TP group's collective state (``K3DecodeComm``, collective: every
        rank builds it here) handed to every layer, with whether its sandwich takes the layer's o_proj, the decode
        path's one-shot ceiling on every stock MNNVL all-reduce (``use_decode_one_shot``), and the MoE decode path
        (``_build_decode_moe``); then the speculative worker's decode kernels (``_gate_spec_worker_kernels``) and the
        DSpark drafter's collectives (``_gate_drafter_comm``)."""
        super().post_load_weights()
        kda = [layer.linear_attn for layer in self.model.layers if layer.is_kda]
        mla = [layer.self_attn.mixer for layer in self.model.layers if not layer.is_kda]
        if kda[0].k3_buffers is not None:
            return  # built by an earlier call; CUDA graphs captured since hold it
        device = self.model.embed_tokens.weight.device
        buffers = K3KdaBuffers.create(device)
        workspace = K3MlaAttnWorkspace.create(device, mla[0].num_heads_tp // 6)
        for module in kda:
            module.k3_buffers = buffers
        for module in mla:
            module.k3_workspace = workspace
        for module in kda + mla:
            module.decode_gemvs = self.model.decode_gemvs
        layers = self.model.layers
        comm = None
        if all(layer._mnnvl_allreduce() is not None for layer in layers):
            for layer in layers:
                layer.sandwich_oproj = _decode_comm.K3DecodeComm.takes_oproj(
                    layer._o_proj(), self.model.num_attn_res_snapshots
                )
            oproj = next((layer._o_proj().weight for layer in layers if layer.sandwich_oproj), None)
            comm = _decode_comm.K3DecodeComm.create(self.model_config.mapping, oproj)
            for layer in layers:
                layer.decode_comm = comm
            _decode_comm.use_decode_one_shot(self)
        # >>> route B: no MoE decode path yet (its build is tp16_moetp4ep4's; k3_moe's wide build does not fit 896
        # local experts). The MoE layers run the generic path until route B's own engines are wired.
        moe_layers = 0
        # <<< route B
        logger.info(
            # >>> route B: no verify kernels
            "Kimi K3 decode kernels: KDA on k3_kda_decode_attn "
            # <<< route B
            f"({sum(m.takes_k3_kernels for m in kda)} / {len(kda)} layers take them), MLA on k3_mla_qkv and "
            f"k3_mla_attn_vb_out ({len(mla)} layers); the post-attention all-reduce of at most "
            f"{_decode_comm.AR_ATTN_RES_MAX_TOKENS} tokens "
            + (
                "with the residual update: k3_sandwich_oproj with o_proj at most "
                f"{_decode_comm.SANDWICH_MAX_TOKENS} decode tokens "
                f"({sum(layer.sandwich_oproj for layer in layers)} / {len(layers)} layers), else "
                "mnnvl_allreduce_attn_res"
                if comm is not None
                else "unfused (an attention all-reduce does not run over MNNVL)"
            )
            + f"; MoE on k3_moe_front, k3_moe and the row-parallel tail ({moe_layers} layers)"
        )
        self._gate_spec_worker_kernels(comm)
        self._gate_drafter_comm(comm)

    def _build_decode_moe(self, comm: _decode_comm.K3DecodeComm) -> int:
        """The MoE decode path (``decode_moe.py``) on every MoE layer it takes: the shared state (collective: the
        front's head workspace), each layer's decode weights with its latent norm folded into the latent up
        projection, the decode GEMVs' state; then one call of every kernel of the path (collective: the front and
        the sandwich tail). Every rank builds it here. Returns the number of MoE layers it takes."""
        mapping = self.model_config.mapping
        moes = [layer.block_sparse_moe for layer in self.model.layers if layer.is_moe]
        gaps = {
            id(moe): _decode_moe.layout_gaps(
                moe, mapping.tp_size, self.model.num_attn_res_snapshots
            )
            for moe in moes
        }
        takes = [moe for moe in moes if not gaps[id(moe)]]
        for reason in sorted({gap for moe in moes for gap in gaps[id(moe)]}):
            logger.info_once(
                f"Kimi K3 MoE decode path off on some layers: {reason}",
                key=f"k3_decode_moe_off_{reason}",
            )
        if not takes:
            return 0
        backend = takes[0].routed_experts.backend
        state = _decode_moe.K3DecodeMoe.create(
            mapping,
            backend.w3_w1_weight.device,
            backend.w3_w1_weight.shape[1] // 2,
            backend.expert_size_per_partition,
            comm.mnnvl,
        )
        for moe in takes:
            _decode_moe.fold_latent_norm(moe)
            moe.decode_moe = _decode_moe.K3DecodeMoeLayer.create(
                moe, state, mapping.tp_rank, mapping.tp_size
            )
            moe.decode_gemvs = self.model.decode_gemvs
        first = takes[0].decode_moe
        first.warm_up(takes[0])
        logger.info(
            "Kimi K3 MoE decode path: the latent all-reduce at <= 8 tokens on "
            + (
                "the routed experts' push form and k3_latent_reduce on pushing steps (a pure decode step captured "
                "into a CUDA graph on the decode kernels), else the routed experts' all-reduce"
                if state.exchange is not None
                else f"the routed experts' all-reduce (no latent exchange at TP {mapping.tp_size})"
            )
        )
        comm.compile_tail(takes[0].moe_hidden_size, first.shared_cols, first.tail_weight)
        return len(takes)

    def _gate_spec_worker_kernels(self, comm: Optional[_decode_comm.K3DecodeComm]) -> bool:
        """Turn the DFlash / DSpark worker's Kimi K3 decode kernels (its ``k3_decode``: ``trtllm::k3_spec_accept``,
        ``k3_ctx_kv`` and ``k3_markov``, on target and draft logits kept vocabulary-sharded) on only when every input
        that path needs exists: the TP group's collective state over MNNVL (``comm``) and the LM head's
        ``gemm/k3_head_gemv`` workspace. ``K3LogitsProcessor.lm_head_shard`` needs that workspace to produce the
        vocabulary shard, so the workspace is a precondition of the path, not a policy choice. The worker still checks
        each step's own conditions. Returns the setting (False without such a worker)."""
        worker = getattr(self, "spec_worker", None)
        if not hasattr(worker, "k3_decode"):
            return False
        gemvs = self.model.decode_gemvs
        worker.k3_decode = (
            comm is not None and gemvs is not None and gemvs.head_workspace is not None
        )
        logger.info(
            f"Kimi K3 decode kernels: {type(worker).__name__} k3_spec_accept, k3_ctx_kv and k3_markov "
            + (
                "on"
                if worker.k3_decode
                else "off (no MNNVL decode state or no k3_head_gemv LM head)"
            )
        )
        return worker.k3_decode

    def _gate_drafter_comm(self, comm: Optional[_decode_comm.K3DecodeComm]) -> bool:
        """Hand the DSpark drafter (`K3DSparkDrafter`) the TP group's collective state ``comm`` where this target built
        it (every attention all-reduce over MNNVL; TP16 is a construction assert): the drafter then runs its context
        projection split over the group, ``hidden_norm`` in the all-reduce, and its blocks' residual adds and RMSNorms
        in their all-reduces (``K3DSparkDrafter.use_decode_comm``, collective: it compiles the drafter's sandwich on
        the group's workspace). Without ``comm`` it keeps the stock replicated ``fc``, all-reduces and norms. The gate
        requires exactly what this path uses: ``comm`` and nothing else, so not the LM head's ``k3_head_gemv``
        workspace, which only the speculative worker's path (`_gate_spec_worker_kernels`) reads. Returns whether the
        drafter took the state (False without such a drafter)."""
        drafter = getattr(self, "draft_model", None)
        if comm is None or not isinstance(drafter, K3DSparkDrafter):
            return False
        drafter.use_decode_comm(comm)
        return True

    def _check_step_contract(self, attn_metadata: AttentionMetadata) -> None:
        """First-forward checks of the engine surface and the per-engine settings."""
        objects = {
            "attn_metadata": attn_metadata,
            "kv_cache_manager": attn_metadata.kv_cache_manager,
        }
        missing = [
            f"{owner}.{name}"
            for owner, names in REQUIRED_ENGINE_FIELDS.items()
            for name in names
            if not hasattr(objects[owner], name)
        ]
        assert not missing, f"engine fields this target reads are missing: {missing}"
        assert attn_metadata.tokens_per_block == _TOKENS_PER_BLOCK, (
            f"this target needs kv_cache_config.tokens_per_block {_TOKENS_PER_BLOCK}; the engine built "
            f"{attn_metadata.tokens_per_block}"
        )
        manager = attn_metadata.kv_cache_manager
        assert isinstance(manager, MambaHybridCacheManagerV2), (
            "this target needs the V2 hybrid KV / state cache manager "
            "(kv_cache_config.use_kv_cache_manager_v2); the engine built "
            f"{type(manager).__name__}"
        )
        assert not manager.enable_block_reuse, (
            "this target runs with kv_cache_config.enable_block_reuse false; the engine enabled it"
        )
        kda_layer = next(layer.layer_idx for layer in self.model.layers if layer.is_kda)
        state_dtype = manager.mamba_layer_cache(kda_layer).temporal.dtype
        assert state_dtype == torch.float32, (
            "this target's KDA kernels keep fp32 recurrent states (kv_cache_config.mamba_ssm_cache_dtype float32 or "
            f"auto); the engine built a {state_dtype} state pool"
        )
        self._step_checked = True

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs.pop("multimodal_params", None):
            raise ValueError(
                "this Kimi K3 target is text only: it loads no vision tower, and a request carried image input"
            )
        if not self._step_checked:
            self._check_step_contract(attn_metadata)
        return super().forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            spec_metadata,
            resource_manager,
            **kwargs,
        )
