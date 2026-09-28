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
``moe_expert_parallel_size`` explicitly. Routing is computed replicated; the
routed partial sums — EP partials of whole experts, or TP partials over the
intermediate shards — are all-reduced in the latent space (before
``routed_expert_norm`` / ``routed_expert_up_proj``, which are nonlinear/linear
layers applied to the full sum). When attention DP is off, the shared experts
use standard MLP TP over the model TP group: gate/up are column-sharded and down
is row-sharded. The shared down projection reduces on the auxiliary stream
while the routed expert chain runs on the main stream. After the streams join,
the routed latent partial is reduced before its norm/up projection.
Fused-communication routed backends already return a complete routed result and
need no outer reduction. Under attention DP the shared experts stay replicated.
``lm_head`` uses the stock ``LMHead`` (vocab-sharded + gather), so logits are
identical on all ranks.

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
import threading
import weakref
from contextlib import ExitStack
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

import torch
from safetensors import safe_open
from torch import nn

from ..._utils import is_sm_100f
from ...logger import logger
from ...mapping import Mapping
from ...models.modeling_utils import QuantAlgo, QuantConfig
from ..attention.backends import AttentionMetadata
from ..distributed import AllReduce, AllReduceParams
from ..model_config import ModelConfig
from ..modules.gated_mlp import GatedMLP
from ..modules.kimi_kda import KimiKDALinearAttention
from ..modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
from ..modules.linear import Linear as TrtllmLinear
from ..modules.linear import TensorParallelMode, load_weight_shard
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..modules.situ import SituAndMul
from ..moe.fused_moe import ConfigurableMoE, SiTuActivation, TRTLLMGenFusedMoE, create_moe
from ..moe.fused_moe.interface import MoESchedulerKind
from ..moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod
from ..utils import AuxStreamType
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model, run_concurrently

# A/B escape hatch: restore nn.Linear for the K3 latent MoE projections
# instead of the min-latency fused GEMM op (read once at import).
_K3_DISABLE_MIN_LATENCY_LATENT_PROJ = (
    os.environ.get("TLLM_K3_DISABLE_MIN_LATENCY_LATENT_PROJ", "0") == "1"
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...llmapi.llm_args import DecodingBaseConfig

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
    ".self_attn.mixer.k_b_proj_trans_scale",
    ".self_attn.mixer.v_b_proj_scale",
    ".self_attn.mixer.k_b_proj_trans_dequant",
    ".self_attn.mixer.v_b_proj_dequant",
)

# BF16 shared/latent MLP weights are optionally quantized once after loading.
# Checkpoint-native attention FP8 is independent of this lossy conversion.
_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV = "KIMI_K3_FP8_WEIGHT_READ_MOE_MLP"


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


KIMI_K3_AUX_ATTN_RES_STREAM_ENV = "KIMI_K3_AUX_ATTN_RES_STREAM"
"""Which residual-stream value the DFlash/DSpark hidden-state tap captures.

``1`` (default) captures the pre-norm attn_res mixture -- the value the next
consumer actually reads. ``0`` captures the raw running prefix sum instead.

Both conventions exist in the wild and a drafter distilled against one scores
lower on the other with nothing raised, so this is a property of the DRAFTER
checkpoint, not a performance knob. SGLang (and therefore RadixArk/Kimi-K3-DSpark)
uses the mixture: ``kimi_k3.py _dspark_capture_stream`` -> ``attn_residual.py
aggregate_stream``. vLLM implements both and defaults to the prefix
(``VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=0``, ``models/kimi_k3/nvidia/model.py
_capture_aux_hidden_stream``), which is what a TorchSpec-distilled drafter may
have been trained against. Measured cost of getting it wrong on K3 + RadixArk:
AR 71.4% -> 66.9%.

Per-checkpoint measurements on K3, GSM8K AL, n=200, TEP8:

===================  ==================  ============  ======
drafter              stream (default 1)  prefix (0)    delta
===================  ==================  ============  ======
RadixArk (GQA)       71.8%               --            --
Inferact (MLA)       65.7%               66.6%         +0.9pt
===================  ==================  ============  ======

So the default is right for RadixArk. For Inferact the prefix convention
matches its vLLM/TorchSpec lineage and measures better, but +0.9pt at n=200 is
inside this harness's noise band (it treats RadixArk's own 71-73% spread as
noise), so this is a direction, not a settled requirement -- unlike the 4.5pt
RadixArk case above, which was unambiguous. Both Inferact acc_len values (5.60
and 5.66) sit on its model card's 5.64, so neither convention is grossly wrong
for it.

Deriving this from checkpoint metadata is not possible today: neither published
drafter's config records which capture convention it was distilled against."""

_AUX_ATTN_RES_STREAM_ENABLED = os.environ.get(KIMI_K3_AUX_ATTN_RES_STREAM_ENV, "1") == "1"

KIMI_K3_FUSED_ATTN_RES_ENV = "KIMI_K3_FUSED_ATTN_RES"
"""Set to ``0`` to disable the in-tree fused Torch op
``trtllm::attn_res_fwd`` (Blackwell only). Default: fused with fallback."""

_FUSED_ATTN_RES_ENABLED = os.environ.get(KIMI_K3_FUSED_ATTN_RES_ENV, "1") == "1"


KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV = "KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS"
"""Largest per-rank token count for which the fused epilogue is taken.

Accepts any integer >= 1. The default follows ``KIMI_K3_ATTN_RES_TOPOLOGY``:
1 when the persistent port is off (the pre-existing decode-only gate) and 32
when it is on. Setting this explicitly overrides both.

Read at import, so CUDA-graph capture and replay cannot disagree about it.
"""

KIMI_K3_ATTN_RES_TOPOLOGY_ENV = "KIMI_K3_ATTN_RES_TOPOLOGY"
"""Which fused attention-residual kernel topology to use.

Accepted values:

  ``per_token``   one CTA (N<=4) or one 8-CTA cluster (N>=5) per token; the
                  persistent kernel is never used. This is the default, i.e.
                  the feature is off and behaviour is unchanged.
  ``persistent``  one CTA per SM looping over tokens, used at every shape the
                  persistent kernel implements (H == 7168 and 2 <= N <= 9).
  ``split``       persistent above ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS``
                  tokens, per-token at or below it.
  ``1``           the only accepted on-value; resolves to ``split``.

Off by default because the persistent kernel measures as no-regression on
NVFP4 but as a 2.98x context-phase regression on W4A8_MXFP4_MXFP8, through a
downstream interaction with the TRTLLM-gen MoE rather than through this kernel.

Read at import, so CUDA-graph capture and replay cannot disagree about it.
"""

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


def _use_persistent_attn_res(M: int, H: int, N: int) -> bool:
    """Pick between the two fused kernels for this call site.

    ``persistent`` takes the persistent kernel at every shape it implements;
    ``split`` takes it only above ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS`` tokens,
    which stands in for the prefill/decode boundary. Shapes the persistent
    kernel does not implement fall through to the caller's existing gate and
    land on the unfused path.
    """
    if not _persistent_attn_res_applicable(M, H, N):
        return False
    if _ATTN_RES_TOPOLOGY == "persistent":
        return True
    return _ATTN_RES_TOPOLOGY == "split" and M > _FUSED_ATTN_RES_MAX_TOKENS


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
    if _use_persistent_attn_res(M, H, N):
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

    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
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
    if _use_persistent_attn_res(M, H, N):
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

    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
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
) -> torch.Tensor:
    """Apply attention-residual selection and the next RMSNorm."""
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_rmsnorm_fused(prefix_sum, block_residual, proj, norm, output_norm)
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add an attention output to the running residual, then select and norm."""
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_add_rmsnorm_fused(
            prefix_sum, addend, block_residual, proj, norm, output_norm
        )
        if fused is not None:
            return fused
    updated_prefix_sum = prefix_sum + addend
    return updated_prefix_sum, _apply_attn_res_and_rmsnorm(
        updated_prefix_sum, block_residual, proj, norm, output_norm
    )


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


def _shard_head_major_param(
    name: str,
    src: torch.Tensor,
    param: torch.nn.Parameter,
    *,
    kda_tp_size: int,
    kda_tp_rank: int,
    model_tp_rank: int,
) -> torch.Tensor:
    """TP-shard a KDA ``linear_attn`` or shared-expert / dense-MLP
    ``down_proj`` checkpoint tensor down to this rank's local slice.

    Returns ``src`` unchanged when it already matches ``param`` (replicated or
    single-rank tensors) and for every other name — the caller then routes
    MLA head-shards and the shape-match copy. The actual slicing is delegated
    to :func:`load_weight_shard` so the framework owns the ceil-divide
    semantics.

    - ``.linear_attn.`` names are KDA head-major tensors (``linear_attn``
      exists only on KDA layers): ``o_proj`` shards its input columns (ROW),
      every other projection its output rows (COLUMN), by ``kda_tp_size``.
    - a shared-expert / dense-MLP ``down_proj`` is ROW-sharded on its input
      columns; the fused ``gate_up_proj`` is row-concatenated and returns
      before this helper, so ``down_proj`` is the only half that reaches here.
      The TP factor comes from the checkpoint-vs-param shapes; a subgroup
      smaller than model TP repeats, so the shard index is ``model_tp_rank``
      modulo the parameter's shard count.
    """
    if src.shape == param.shape:
        return src
    if ".linear_attn." in name:
        mode = (
            TensorParallelMode.ROW if name.endswith(".o_proj.weight") else TensorParallelMode.COLUMN
        )
        return load_weight_shard(src, kda_tp_size, kda_tp_rank, mode, device=param.device)
    if name.endswith(".down_proj.weight") and (".shared_experts." in name or ".mlp." in name):
        assert src.shape[1] % param.shape[1] == 0, (
            f"{name}: checkpoint input dim {src.shape[1]} is not "
            f"divisible by param input dim {param.shape[1]}"
        )
        tp = src.shape[1] // param.shape[1]
        return load_weight_shard(
            src, tp, model_tp_rank % tp, TensorParallelMode.ROW, device=param.device
        )
    return src


def _helix_cp_v_b_shard(
    v_weight: torch.Tensor,
    *,
    num_heads_tp_cp: int,
    cp_rank: int,
) -> torch.Tensor:
    """Select this CP rank's ``v_b_proj`` head chunk from a tp-local KV-B split.

    Under Helix context-parallel (cp_size > 1) ``v_b_proj`` holds only this
    rank's 1/cp post-all-to-all head chunk, while ``kv_b_proj`` and
    ``k_b_proj_trans`` keep every tp-local head. When cp_size == 1
    (``num_heads_tp_cp`` equals the full tp-local head count) the tensor is
    returned unchanged. ``v_weight`` is ``[num_heads_tp, v_head_dim,
    kv_lora_rank]``; only its leading head axis is sliced.
    """
    if num_heads_tp_cp != v_weight.shape[0]:
        lo = cp_rank * num_heads_tp_cp
        v_weight = v_weight[lo : lo + num_heads_tp_cp]
    return v_weight


# ---------------------------------------------------------------------------
# FP8 block-scale weight read for attention and replicated MoE-layer MLP projections.
# ---------------------------------------------------------------------------


class _Fp8BlockScaleWeightReadLinear(nn.Module):
    """Bias-free FP8 block-scale GEMM for checkpoint attention and converted MLPs.

    Attention loads checkpoint E4M3 codes and 128x128 scales directly. The optional
    MLP path quantizes BF16 weights once at load. Both prepare UE8M0 scales
    for DeepGEMM and quantize BF16 activations inside the GEMM.
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

        weight_fp8, weight_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        return _Fp8BlockScaleWeightReadLinear._prepare_weights(weight_fp8, weight_scale)

    @staticmethod
    def _prepare_weights(
        weight_fp8: torch.Tensor, weight_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from ...quantization.utils.fp8_utils import (
            resmooth_to_fp8_e8m0,
            transform_sf_into_required_layout,
        )

        # fp8_swap_ab_gemm with disable_ue8m0_cast=True consumes packed,
        # TMA-aligned UE8M0 scales rather than the checkpoint's FP32 grid.
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
        if isinstance(linear, TrtllmLinear) and linear.has_fp8_block_scales:
            result = cls(linear.weight.detach(), linear.weight_scale.detach(), linear.out_features)
            result.quant_config = linear.quant_config
            result.tp_size, result.tp_rank = linear.tp_size, linear.tp_rank
            result._weights_transformed = False
            return result
        weight_fp8, weight_scale = cls.quantize_weight(linear.weight.data)
        return cls(weight_fp8, weight_scale, linear.out_features)

    def transform_weights(self) -> None:
        if not self._weights_transformed:
            self.weight, self.weight_scale = self._prepare_weights(self.weight, self.weight_scale)
            if self._fused_projection is not None:
                owner_ref, start = self._fused_projection
                owner = owner_ref()
                if owner is None:
                    raise RuntimeError("KDA fused FP8 projection was released before its views")
                owner.transform_weights()
                self.weight = owner.weight[start : start + self.out_features]
            self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()

    def forward(
        self,
        x: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        out_shape = (*x.shape[:-1], self.out_features)
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

    Transfer checkpoint FP8 storage or release the original BF16 storage
    after conversion. The loader's transient parameter map must not retain
    an unused BF16 copy until loading completes.
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


# ---------------------------------------------------------------------------
# Latent MoE block using the unified ConfigurableMoE stack.
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
# Routed-expert checkpoint layouts.
#
# K3 streams routed experts one at a time (see ``load_weights``), so the tensor
# names, the per-expert loader and the finalization it needs are all decided by
# the checkpoint's routed-expert quantization rather than by the MoE backend.
# Three call sites need that decision — the expected-key plan, the loader, and
# the file grouping — so it lives in one spec instead of three conditionals.
# ---------------------------------------------------------------------------


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


def _load_nvfp4_expert(backend, base, expert_idx, local_slot_id, get_tensor) -> None:
    backend.quant_method.load_streaming_nvfp4_expert(
        backend,
        global_expert_id=expert_idx,
        local_slot_id=local_slot_id,
        **{
            f"{w}_{kind}": get_tensor(f"{base}.{expert_idx}.{w}.{kind}")
            for w in ("w1", "w2", "w3")
            for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
        },
    )


class _K3ExpertCkptSpec(NamedTuple):
    """How one routed-expert quantization is spelled and loaded."""

    # Per-``w{1,2,3}`` checkpoint tensor suffixes this layout stores.
    kinds: Tuple[str, ...]
    loader: Callable[..., None]
    # Set of filled slots the loader maintains, checked after the load.
    loaded_slots_attr: str
    # NVFP4 defers cat/pad/interleave and the alpha computation to
    # ``process_weights_after_loading``; the MXFP4 loaders write through.
    needs_layer_finalize: bool


_K3_EXPERT_CKPT_SPECS = {
    QuantAlgo.W4A8_MXFP4_MXFP8: _K3ExpertCkptSpec(
        kinds=("weight_packed", "weight_scale"),
        loader=_load_packed_mxfp4_expert,
        loaded_slots_attr="_packed_mxfp4_loaded_slots",
        needs_layer_finalize=False,
    ),
    QuantAlgo.NVFP4: _K3ExpertCkptSpec(
        kinds=("weight", "weight_scale", "weight_scale_2", "input_scale"),
        loader=_load_nvfp4_expert,
        loaded_slots_attr="_streamed_expert_slots",
        needs_layer_finalize=True,
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
            # the CuTe DSL dependency. Measured 2026-09-08: a CUTEDSL request
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
        attention_dp = model_config.mapping.enable_attention_dp
        shared_model_config = copy.copy(model_config)
        shared_model_config.quant_config = QuantConfig()
        # Under attention DP each rank owns different tokens, so the shared
        # expert is replicated (TP size 1) and must not reduce across ranks.
        # Direct MoE-TP leaves both branches as partials for one concatenated
        # all-reduce.
        use_shared_tp = not attention_dp and model_config.mapping.tp_size > 1
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
            overridden_tp_size=1 if attention_dp else None,
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
        config. Default split is EP-only; see ``_select_moe_tp_ep``."""
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

    def forward(self, hidden_states: torch.Tensor, all_rank_num_tokens=None) -> torch.Tensor:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16."""
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


# ---------------------------------------------------------------------------
# MLA runtime.
# ---------------------------------------------------------------------------


class KimiMLARuntime(nn.Module):
    """Wraps K3 MLA and applies its external TP output reduction."""

    def __init__(
        self,
        cfg: "PretrainedConfig",
        layer_idx: int,
        model_config: ModelConfig,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        mapping_with_cp: Optional[Mapping] = None,
    ) -> None:
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
        # Helix: mapping_with_cp (the CP original) activates the base MLA's
        # helix machinery; this wrapper's allreduce over the repurposed
        # mapping sums the base o_proj's tp*cp partials.
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
            model_config=attention_config,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
        )

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        # MLA.forward takes position_ids first; K3 is NoPE, so pass None.
        out = self.mixer(None, hidden_states, attn_metadata)
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
            self.linear_attn = KimiKDALinearAttention(
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
                # CP original stashed by _setup_helix_mappings; None outside helix.
                mapping_with_cp=getattr(model_config, "_helix_mapping_with_cp", None),
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
        capture: Optional[Tuple[Any, int]] = None,
    ) -> Tuple[torch.Tensor, int]:
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
        """
        prefix_sum = hidden_states
        valid_block_residual = block_residual[:num_snapshots]

        if capture is not None:
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
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None
        if self.is_kda:
            hidden_states = self.linear_attn(hidden_states, attn_metadata)
        else:
            hidden_states = self.self_attn(hidden_states, attn_metadata)

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

        # Which convention the drafter tap is on is not recoverable from the
        # served output -- a mismatch only lowers acceptance -- so state it once
        # at construction rather than leaving it to be inferred from an AL.
        logger.info_once(
            "Kimi K3 aux hidden capture: mode="
            f"{'attn_res_stream' if _AUX_ATTN_RES_STREAM_ENABLED else 'prefix_only'} "
            f"({KIMI_K3_AUX_ATTN_RES_STREAM_ENV}={int(_AUX_ATTN_RES_STREAM_ENABLED)})",
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
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

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
                hidden_states, block_residual, num_snapshots, attn_metadata, capture=capture
            )

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
        )


# ---------------------------------------------------------------------------
# Causal LM wrapper + weight loading.
# ---------------------------------------------------------------------------


_FP8_BLOCK_SCALE_SUFFIX = "_scale"
_FP8_BLOCK = 128


def _fp8_block_scale_key(weight_key: str) -> str:
    return weight_key + _FP8_BLOCK_SCALE_SUFFIX


def _checkpoint_fp8_pair(ckpt_key: str, src: torch.Tensor, weights):
    """``(fp8 weight, fp32 128x128 block scale)`` if this tensor is FP8_PB_WO.

    ``None`` for an ordinary (BF16) checkpoint tensor. Raises when an FP8
    tensor has no companion scale rather than letting the caller fall through
    to a cast that would silently drop it.
    """
    if src.dtype != torch.float8_e4m3fn:
        return None
    scale_key = _fp8_block_scale_key(ckpt_key)
    if scale_key not in weights and scale_key + "_inv" in weights:
        scale_key += "_inv"
    if scale_key not in weights:
        raise KeyError(
            f"Kimi K3: {ckpt_key} is FP8 E4M3 but has no {scale_key} or {scale_key}_inv; refusing "
            "to load it as if it were unquantized."
        )
    scale = _materialize(weights[scale_key]).float()
    # The checkpoint stores the block scale 4-D as
    # [ceil(N/128), 1, ceil(K/128), 1]. Normalize here, once, so every
    # consumer sees the plain 2-D [n_blocks_m, n_blocks_k] that both the
    # dequantization below and deep_gemm's transform_sf_into_required_layout
    # expect -- the latter asserts on rank and gave an unhelpful
    # ``assert sf.dim() == ...`` when handed the raw 4-D tensor.
    if scale.dim() == 4:
        scale = scale.reshape(scale.shape[0], scale.shape[2])
    return src, scale


def _dequantize_fp8_block_scaled(ckpt_key: str, src: torch.Tensor, weights) -> torch.Tensor:
    """Apply checkpoint scales for projections explicitly configured as BF16.

    Checkpoint-native FP8 attention bypasses this loader. This handles an
    excluded projection or another unquantized trunk parameter whose source
    tensor is quantized; casting the codes alone would discard its scales.
    """
    pair = _checkpoint_fp8_pair(ckpt_key, src, weights)
    if pair is None:
        return src
    src, scale = pair
    out = src.to(torch.float32)
    if scale.numel() == 1:
        return (out * scale.reshape(())).to(torch.bfloat16)
    expanded = scale.repeat_interleave(_FP8_BLOCK, dim=0).repeat_interleave(_FP8_BLOCK, dim=1)
    if expanded.shape[0] < out.shape[0] or expanded.shape[1] < out.shape[1]:
        raise ValueError(
            f"Kimi K3: {_fp8_block_scale_key(ckpt_key)} covers "
            f"{tuple(expanded.shape)} but {ckpt_key} is {tuple(out.shape)}."
        )
    return (out * expanded[: out.shape[0], : out.shape[1]]).to(torch.bfloat16)


def _materialize(value) -> torch.Tensor:
    """Materialize a (possibly lazy safetensors slice) weight value."""
    if isinstance(value, torch.Tensor):
        return value
    # ``[:]`` is how a lazy slice is realized, but it is invalid on a 0-dim
    # entry. The NVFP4 checkpoint stores weight_scale_2 / input_scale as
    # scalars, so realize those entries with ``[()]`` instead.
    get_shape = getattr(value, "get_shape", None)
    if get_shape is not None and len(get_shape()) == 0:
        return value[()]
    return value[:]


@register_auto_model("KimiLinearForCausalLM")
class KimiLinearForCausalLM(SpecDecOneEngineForCausalLM[KimiLinearModel, Any]):
    """Kimi K3 text core (KDA + MLA + MoE).

    Serves text-only ``kimi_linear`` checkpoints directly, and is reused as the
    text backbone by the multimodal ``KimiK3ForConditionalGeneration`` wrapper
    (``modeling_kimi_k3_vl``). The composite ``KimiK3ForConditionalGeneration``
    architecture is registered by that wrapper, not here."""

    mamba_metadata_cls = KimiK3MambaMetadata

    def __init__(self, model_config: ModelConfig):
        mlp_fp8 = os.environ.get(_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV, "0")
        if mlp_fp8 not in ("0", "1"):
            raise ValueError(
                f"{_KIMI_K3_FP8_WEIGHT_READ_MOE_MLP_ENV} must be 0 or 1; got {mlp_fp8!r}"
            )
        self._fp8_weight_read_moe_mlp = mlp_fp8 == "1"
        cfg = _get_text_config(model_config.pretrained_config)
        assert model_config.mapping.pp_size == 1, "Kimi K3 does not support pipeline parallelism"
        spec_config = getattr(model_config, "spec_config", None)

        # Helix: swap in the repurposed mapping; restored after super().__init__.
        self._setup_helix_mappings(model_config, cfg, spec_config)
        # Supported spec-dec modes:
        # - SA (suffix automaton): one-engine in-forward drafting, no draft
        #   weights; the KDA/MLA verify paths below implement multi-token
        #   verification for it.
        # - DFlash: external-drafter parallel drafting; the drafter is a
        #   separate dense checkpoint (K2.7-Code-DFlash schema) consumed by
        #   the generic DFlashForCausalLM wrapper, and the target only has
        #   to expose per-layer hidden states via maybe_capture_hidden_states
        #   (see KimiLinearModel.forward).
        # - DSpark: the same external-drafter flow with the Markov and
        #   confidence heads enabled (RadixArk/Kimi-K3-DSpark and friends).
        #   The target side is identical -- the capture in
        #   KimiLinearModel.forward is unconditional -- so this gate is the
        #   only place the mode has to be admitted.
        # Modes needing draft heads (MTP/Eagle) are blocked until a
        # draft-head checkpoint exists.
        assert (
            spec_config is None
            or spec_config.spec_dec_mode.is_sa()
            or spec_config.spec_dec_mode.is_dflash()
            or spec_config.spec_dec_mode.is_dspark()
        ), "Kimi K3 supports speculative decoding only with SA, DFlash or DSpark"
        super().__init__(
            KimiLinearModel(model_config),
            model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

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
        if model_config.mapping.enable_attention_dp:
            raise ValueError(
                "Kimi K3 helix phase 1 requires enable_attention_dp="
                "False: the helix ADP token-scatter conflicts with the "
                "per-request locality of KDA recurrent state."
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

    def _attention_fp8_linears(self) -> list[tuple[str, TrtllmLinear, nn.Module]]:
        """Collect checkpoint-backed FP8 attention projections for weight loading.

        The loader uses this list to enumerate required weight/scale keys, exclude
        these parameters from ordinary trunk loading, and load FP8 codes and scales
        together. Derived QKVG fusion is excluded because it is built after loading.

        Returns:
            Tuples of runtime module path, linear, and owning attention module.
        """
        projections = []
        for index, layer in enumerate(self.model.layers):
            if not _has_weights(layer):
                continue
            if layer.is_kda:
                attention = layer.linear_attn
                scope = f"model.layers.{index}.linear_attn"
            else:
                attention = layer.self_attn.mixer
                scope = f"model.layers.{index}.self_attn.mixer"
            for attr, module in attention.named_children():
                if attr == "qkvg_proj":
                    # Decode fusion is derived from separately loaded checkpoint projections.
                    continue
                if isinstance(module, TrtllmLinear) and module.has_fp8_block_scales:
                    projections.append((f"{scope}.{attr}", module, attention))
        return projections

    def _load_attention_fp8(self, weights: Dict[str, torch.Tensor], prefix: str) -> int:
        """Load checkpoint FP8 codes/scales without a BF16 projection intermediate."""
        projections = self._attention_fp8_linears()
        for name, linear, attention in projections:
            # Both attention implementations use self_attn keys in the checkpoint.
            key = (
                prefix
                + name.replace(".linear_attn.", ".self_attn.").replace(
                    ".self_attn.mixer.", ".self_attn."
                )
                + ".weight"
            )
            attr = name.rsplit(".", 1)[1]
            if attr == "kv_a_proj_with_mqa" and attention.fuse_qkv_a_proj:
                # The runtime fuses Q-A/KV-A, but the checkpoint stores them separately.
                # Construction checks that the fusion boundary aligns with FP8 blocks.
                q_key = key.replace("kv_a_proj_with_mqa.weight", "q_a_proj.weight")
                pairs = []
                for part_key, rows in (
                    (q_key, attention.q_lora_rank),
                    (key, attention.kv_lora_rank + attention.qk_rope_head_dim),
                ):
                    part = _materialize(weights[part_key])
                    pair = _checkpoint_fp8_pair(part_key, part, weights)
                    if pair is None:
                        raise ValueError(
                            f"{part_key}: quant config declares FP8 but tensor is not E4M3"
                        )
                    codes, scales = pair
                    expected = (rows, attention.hidden_size)
                    if tuple(codes.shape) != expected:
                        raise ValueError(
                            f"{part_key}: checkpoint shape {tuple(codes.shape)} != {expected}"
                        )
                    if tuple(scales.shape) != tuple(math.ceil(d / _FP8_BLOCK) for d in expected):
                        raise ValueError(f"{part_key}: scale shape does not cover {expected}")
                    pairs.append(pair)
                linear.load_weights(
                    [
                        {
                            "weight": torch.cat([pair[0] for pair in pairs], dim=0),
                            "weight_scale": torch.cat([pair[1] for pair in pairs], dim=0),
                        }
                    ]
                )
                continue
            source = weights[key]
            shape = (
                tuple(source.shape)
                if isinstance(source, torch.Tensor)
                else tuple(source.get_shape())
            )
            local_shape = (linear.out_features, linear.in_features)
            scale_slice = None
            if isinstance(attention, KimiKDALinearAttention):
                # KDA linears already have rank-local dimensions. Slice the lazy source
                # before materializing it, and apply the same shard to its block scales.
                split_dim = 1 if attr == "o_proj" else 0
                tp_size = attention._kda_tp_size
                tp_rank = attention._kda_tp_rank
                expected_shape = list(local_shape)
                expected_shape[split_dim] *= tp_size
                if shape != tuple(expected_shape):
                    raise ValueError(f"{key}: checkpoint shape {shape} != {tuple(expected_shape)}")
                if tp_size > 1:
                    width = local_shape[split_dim]
                    start, end = tp_rank * width, (tp_rank + 1) * width
                    if start % _FP8_BLOCK or end % _FP8_BLOCK:
                        raise ValueError(f"{key}: TP slice [{start}:{end}] is not block-aligned")
                    indices = [slice(None), slice(None)]
                    indices[split_dim] = slice(start, end)
                    source = source[tuple(indices)]
                    scale_indices = [slice(None), slice(None)]
                    scale_indices[split_dim] = slice(
                        start // _FP8_BLOCK, math.ceil(end / _FP8_BLOCK)
                    )
                    scale_slice = tuple(scale_indices)
            pair = _checkpoint_fp8_pair(key, _materialize(source), weights)
            if pair is None:
                raise ValueError(
                    f"{key}: quant config declares FP8 but checkpoint tensor is not E4M3"
                )
            weight, scale = pair
            full_scale_shape = tuple(math.ceil(dim / _FP8_BLOCK) for dim in shape)
            if tuple(scale.shape) != full_scale_shape:
                raise ValueError(f"{key}: scale shape {tuple(scale.shape)} != {full_scale_shape}")
            if scale_slice is not None:
                scale = scale[scale_slice]
            if isinstance(attention, KimiKDALinearAttention):
                if tuple(scale.shape) != tuple(math.ceil(dim / _FP8_BLOCK) for dim in local_shape):
                    raise ValueError(f"{key}: scale shard does not cover local shape {local_shape}")
                linear.load_weights([{"weight": weight, "weight_scale": scale}])
            elif attr == "kv_b_proj":
                # KV-B also supplies decode absorption weights; reorder codes and scales
                # together and retain raw FP8 copies for the absorption kernels.
                self._load_mla_fp8_kv_b(attention, weight, scale)
            else:
                # Other MLA projections delegate TP sharding to the common Linear loader.
                expected_shape = list(local_shape)
                if linear.tp_mode is not None:
                    split_dim = 0 if linear.tp_mode == TensorParallelMode.COLUMN else 1
                    expected_shape[split_dim] *= linear.tp_size
                if shape != tuple(expected_shape):
                    raise ValueError(f"{key}: checkpoint shape {shape} != {tuple(expected_shape)}")
                linear.load_weights([{"weight": weight, "weight_scale": scale}])
        return len(projections)

    @staticmethod
    def _load_mla_fp8_kv_b(attention: nn.Module, weight: torch.Tensor, scale: torch.Tensor) -> None:
        """Reorder checkpoint KV-B heads and their scale grid together."""
        linear = attention.kv_b_proj
        h, n, v, k = (
            attention.num_heads_tp,
            attention.qk_nope_head_dim,
            attention.v_head_dim,
            attention.kv_lora_rank,
        )
        if any(dim % _FP8_BLOCK for dim in (n, v, k)):
            raise ValueError(
                "Kimi MLA FP8 absorption requires block-aligned head and latent dimensions"
            )
        expected_shape = (h * linear.tp_size * (n + v), k)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(f"Kimi MLA KV-B shape {tuple(weight.shape)} != {expected_shape}")
        device = linear.weight.device
        local = linear.load_shard(weight, device=device).reshape(h, n + v, k)
        local_scale = linear.load_shard(scale, scale_span=_FP8_BLOCK, device=device).reshape(
            h, (n + v) // _FP8_BLOCK, k // _FP8_BLOCK
        )
        key_weight, value_weight = local.split((n, v), dim=1)
        key_scale, value_scale = local_scale.split((n // _FP8_BLOCK, v // _FP8_BLOCK), dim=1)
        linear.weight.data.copy_(
            torch.cat((key_weight.reshape(h * n, k), value_weight.reshape(h * v, k)))
        )
        linear.weight_scale.data.copy_(
            torch.cat((key_scale.flatten(0, 1), value_scale.flatten(0, 1)))
        )
        attention.k_b_proj_trans.data.copy_(key_weight.transpose(1, 2))
        attention.k_b_proj_trans_scale.data.copy_(key_scale.transpose(1, 2))
        value_weight = _helix_cp_v_b_shard(
            value_weight,
            num_heads_tp_cp=attention.num_heads_tp_cp,
            cp_rank=attention.mapping.cp_rank,
        )
        value_scale = _helix_cp_v_b_shard(
            value_scale,
            num_heads_tp_cp=attention.num_heads_tp_cp,
            cp_rank=attention.mapping.cp_rank,
        )
        # Absorption must keep raw checkpoint codes independent of Linear's
        # backend-specific weight transformations.
        attention.v_b_proj = nn.Parameter(value_weight.clone().contiguous(), requires_grad=False)
        attention.v_b_proj_scale.data.copy_(value_scale)
        for target, codes, scales in (
            (
                attention.k_b_proj_trans_dequant,
                key_weight.transpose(1, 2),
                key_scale.transpose(1, 2),
            ),
            (attention.v_b_proj_dequant, value_weight, value_scale),
        ):
            if target is not None:
                expanded = scales.repeat_interleave(_FP8_BLOCK, -2).repeat_interleave(
                    _FP8_BLOCK, -1
                )
                target.data.copy_((codes.float() * expanded).to(target.dtype))
        linear.process_weights_after_loading()

    def _trunk_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Named parameters of the trunk only. Spec-dec draft modules
        (e.g. the DFlash drafter attached by SpecDecOneEngineForCausalLM)
        live in a separate checkpoint loaded by
        ModelLoader.load_draft_weights, not in the target checkpoint. MLA
        K/V absorb Parameters are derived by the KV-B loader and are likewise
        excluded from checkpoint jobs."""
        fp8_prefixes = tuple(name + "." for name, _, _ in self._attention_fp8_linears())
        return {
            name: param
            for name, param in self.named_parameters()
            if not name.startswith("draft_model.")
            and not name.startswith(fp8_prefixes)
            and ".linear_attn.qkvg_proj." not in name
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
                if ".linear_attn." in name:
                    ckpt_key = prefix + name.replace(".linear_attn.", ".self_attn.")
                else:
                    # MLA retains its runtime/mixer hierarchy; the checkpoint
                    # has no intermediate ``mixer`` scope.
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
            kinds = moe.expert_ckpt_spec.kinds
            for expert_idx in moe.local_expert_ids:
                for w in ("w1", "w2", "w3"):
                    for kind in kinds:
                        expected_keys.add(f"{base}.{expert_idx}.{w}.{kind}")
            expert_jobs.append((layer_idx, moe, base))
        for name, _, _ in self._attention_fp8_linears():
            key = (
                prefix
                + name.replace(".linear_attn.", ".self_attn.").replace(
                    ".self_attn.mixer.", ".self_attn."
                )
                + ".weight"
            )
            expected_keys.update((key, _fp8_block_scale_key(key)))
        for index, layer in enumerate(self.model.layers):
            if getattr(layer, "is_kda", True) or not _has_weights(layer):
                continue
            attention = layer.self_attn.mixer
            if getattr(attention, "fuse_qkv_a_proj", False):
                key = f"{prefix}model.layers.{index}.self_attn.q_a_proj.weight"
                expected_keys.add(key)
                if attention.kv_a_proj_with_mqa.has_fp8_block_scales:
                    expected_keys.add(_fp8_block_scale_key(key))
        return name_map, expected_keys, expert_jobs

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        prefix = "language_model." if any(k.startswith("language_model.") for k in weights) else ""
        params = self._trunk_parameters()
        name_map, expected_keys, expert_jobs = self.checkpoint_name_plan(prefix)

        self._validate_checkpoint_keys(weights, expected_keys, prefix)
        num_params = self._load_trunk_params(weights, params, name_map)
        num_params += self._load_attention_fp8(weights, prefix)
        self._load_expert_slices(weights, expert_jobs)
        self._finalize_weight_load(num_params, len(expert_jobs))
        device = next(self.parameters()).device
        if device.type == "cuda":
            # Lazy source mappings are load-scoped; finish nonblocking H2D
            # work before the caller can release the weights container.
            torch.cuda.synchronize(device)

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
        missing = sorted(
            key
            for key in expected_keys - ckpt_keys
            if not (key.endswith(".weight_scale") and key + "_inv" in ckpt_keys)
        )
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
        mla_fused_a_mixers = {
            id(mixer.kv_a_proj_with_mqa.weight): mixer
            for mixer in mla_mixers
            if getattr(mixer, "fuse_qkv_a_proj", False)
        }
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
        # Under helix the modules were sharded against the repurposed
        # mapping; the restored CP original's tp_rank is 0 on every rank.
        model_tp_rank = (
            self._repurposed_tp_mapping.tp_rank
            if self._repurposed_tp_mapping is not None
            else self.model_config.mapping.tp_rank
        )
        # KDA head-shard (attention-DP off): rank r loads head rows/cols
        # [r*local : (r+1)*local] of every head-major KDA tensor.
        kda_tp_size, kda_tp_rank = 1, 0
        for layer in self.model.layers:
            if getattr(layer, "is_kda", False):
                kda_tp_size = layer.linear_attn._kda_tp_size
                kda_tp_rank = layer.linear_attn._kda_tp_rank
                break

        COL = TensorParallelMode.COLUMN  # gate/up column shard (output rows)

        def load_param(name: str, param: torch.nn.Parameter) -> None:
            if device.type == "cuda":
                torch.cuda.set_device(device)
            if name.endswith(_GATE_UP_FUSED_SUFFIX):
                # Row-concat the checkpoint's separate gate_proj / up_proj
                # tensors into the fused [gate | up] parameter.
                gate_key, up_key = _gate_up_ckpt_keys(name_map[name])
                inter = param.shape[0] // 2
                # Materialize + FP8-block-dequant each half (this branch reads
                # its own sources, so it needs the same FP8 handling as the
                # single-tensor path below), then column(TP)-shard and
                # row-concat. TP factor from the checkpoint-vs-param shapes; a
                # subgroup smaller than model TP repeats, so the shard index is
                # model tp_rank modulo the half's shard count.
                gate_full = _dequantize_fp8_block_scaled(
                    gate_key, _materialize(weights[gate_key]), weights
                )
                tp = gate_full.shape[0] // inter
                # load_weight_shard returns the whole tensor when tp <= 1, so
                # the rank needs no tp > 1 guard.
                rk = model_tp_rank % tp
                gate = load_weight_shard(gate_full, tp, rk, COL, device=param.device)
                up = load_weight_shard(
                    _dequantize_fp8_block_scaled(up_key, _materialize(weights[up_key]), weights),
                    tp,
                    rk,
                    COL,
                    device=param.device,
                )
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
            src = _dequantize_fp8_block_scaled(name_map[name], src, weights)
            fused_a = mla_fused_a_mixers.get(id(param))
            if fused_a is not None:
                q_key = name_map[name].replace("kv_a_proj_with_mqa.weight", "q_a_proj.weight")
                q_weight = _dequantize_fp8_block_scaled(
                    q_key, _materialize(weights[q_key]), weights
                )
                if q_weight.shape != (fused_a.q_lora_rank, fused_a.hidden_size) or src.shape != (
                    fused_a.kv_lora_rank + fused_a.qk_rope_head_dim,
                    fused_a.hidden_size,
                ):
                    raise ValueError(
                        f"{name}: checkpoint Q-A/KV-A shapes do not match MLA dimensions"
                    )
                src = torch.cat((q_weight, src), dim=0)
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
                # Share the loaded BF16 V rows with context GEMM instead of keeping a
                # separate copy. Wrap the view to retain v_b_proj's parameter registration.
                v_weight = _helix_cp_v_b_shard(
                    param[h * n :].view(h, v, kv),
                    num_heads_tp_cp=mla_mixer.num_heads_tp_cp,
                    cp_rank=mla_mixer.mapping.cp_rank,
                )
                mla_mixer.v_b_proj = nn.Parameter(v_weight, requires_grad=False)
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
            # KDA ``linear_attn`` (head-major) and shared-expert / dense-MLP
            # ``down_proj`` shards are shape-derived; replicated tensors and
            # every other name pass through unchanged. MLA head-shards and the
            # shape-match copy are handled by the block below.
            src = _shard_head_major_param(
                name,
                src,
                param,
                kda_tp_size=kda_tp_size,
                kda_tp_rank=kda_tp_rank,
                model_tp_rank=model_tp_rank,
            )

            if src.shape != param.shape:
                # MLA q_b/g/o head-shard: delegate to the same Linear modules
                # that own their COLUMN/ROW sharding policy (#17684 removed the
                # 96->128 head padding). KV-B is handled above; KDA and
                # shared-expert/MLP shards were resolved in the pre-block above.
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
                raise ValueError(
                    f"{name}: shard/pad result {tuple(src.shape)} != param "
                    f"shape {tuple(param.shape)}"
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

        # Layouts whose per-expert loader only stages its input must have the
        # staging containers created before any thread runs, and must be
        # finalized once the layer's last slot lands: nothing else on this path
        # calls process_weights_after_loading. Which thread completes a layer is
        # a race, so the completion test and the "already finalized" bookkeeping
        # are one critical section.
        finalize_lock = threading.Lock()
        finalized_backends = set()
        prepared_backends = set()

        def ensure_prepared(moe: KimiK3MoERuntime):
            """Prepare a layer's streaming state on its FIRST expert, not up front.

            Preparing every layer before the load starts is what OOM-ed the
            MegaMoE CuteDSL backend: it keeps its raw NVFP4 source params as
            0-element placeholders and rematerializes them at full shape here,
            so preparing all 92 layers held 92 layers of raw weights at once
            instead of the handful actually being filled. Cutlass did not care
            because its parameters are allocated either way.

            Bounded lazily instead. The paired shrink already happens per layer
            in process_weights_after_loading, so the live set is whatever is
            genuinely in flight -- measured at 1 layer per shard file for this
            checkpoint (its rank-local experts are 1:1 with files), so ~4 with
            4 loader threads. Correctness does not depend on that layout
            though; a checkpoint that split a layer across files would only
            raise the peak, not break this.

            The membership add happens AFTER preparing, so the lock-free fast
            path can only ever be stale in the safe direction.
            """
            spec = moe.expert_ckpt_spec
            if not spec.needs_layer_finalize:
                return
            backend = moe.routed_experts.backend
            if id(backend) in prepared_backends:
                return
            with finalize_lock:
                if id(backend) in prepared_backends:
                    return
                backend.quant_method.prepare_streaming_expert_load(backend)
                prepared_backends.add(id(backend))

        def maybe_finalize_layer(moe: KimiK3MoERuntime):
            spec = moe.expert_ckpt_spec
            if not spec.needs_layer_finalize:
                return
            backend = moe.routed_experts.backend
            with finalize_lock:
                loaded = len(getattr(backend, spec.loaded_slots_attr, ()))
                if loaded != backend.expert_size_per_partition:
                    return
                if id(backend) in finalized_backends:
                    return
                finalized_backends.add(id(backend))
            # Computes the alphas, interleaves the w2 scales, and (MegaMoE)
            # packs into the mega buffers and shrinks the raw source params
            # back to placeholders. Paired with ensure_prepared above, that
            # pairing is what bounds the per-layer footprint. The per-expert
            # drain inside load_streaming_nvfp4_expert separately bounds the
            # Cutlass w3_w1 staging, which is per expert rather than per layer.
            backend.process_weights_after_loading()

        def load_expert(
            moe: KimiK3MoERuntime, base: str, local_slot_id: int, expert_idx: int, get_tensor
        ):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            backend = moe.routed_experts.backend
            # Every route into an expert goes through here (file-grouped,
            # split-file and the shared-dict fallback alike), so this is the
            # one place preparation has to be hooked.
            ensure_prepared(moe)
            moe.expert_ckpt_spec.loader(backend, base, expert_idx, local_slot_id, get_tensor)
            maybe_finalize_layer(moe)

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
        # The lazy loader records the directory it opened; prefer it over
        # ``_name_or_path``, which transformers no longer populates (it is
        # empty on transformers 5.x, which silently sent the whole load down
        # the fallback below and OOM-killed the step).
        ckpt_dir = getattr(weights, "checkpoint_dir", None) or getattr(
            self.model_config.pretrained_config, "_name_or_path", None
        )
        index_path = os.path.join(ckpt_dir or "", "model.safetensors.index.json")
        checkpoint_prefix = getattr(weights, "checkpoint_prefix", "")

        def checkpoint_key(key: str) -> str:
            return f"{checkpoint_prefix}{key}"

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
                        for kind in moe.expert_ckpt_spec.kinds
                    ]
                    files = {weight_map[checkpoint_key(key)] for key in keys}
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
                        load_expert(
                            moe,
                            base,
                            local_slot_id,
                            expert_idx,
                            lambda key: fh.get_tensor(checkpoint_key(key)),
                        )
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
                        source_key = checkpoint_key(key)
                        return handles[weight_map[source_key]].get_tensor(source_key)

                    load_expert(*job, get_tensor)
                for file_name in files:
                    drop_file_pages(file_name)

            run_concurrently(load_expert_file, sorted(per_file.items()), num_workers=4)
            run_concurrently(load_split_file_expert, split_file_jobs, num_workers=4)
        else:
            # Falling back is a silent loss of the whole point of the block
            # above: the shared lazy dict keeps every shard mapped, which is
            # the OOM this streaming path exists to avoid. Say so.
            if expert_jobs:
                logger.warning(
                    f"Kimi K3: no safetensors index at '{index_path}', so routed "
                    "experts are loaded from the shared lazy weight dict instead "
                    "of being streamed per shard file. Every shard stays mapped "
                    "for the whole load, which OOM-kills the step at DEP8 scale."
                )
            run_concurrently(load_experts_from_weights, expert_jobs, num_workers=4)

        for _, moe, _ in expert_jobs:
            spec = moe.expert_ckpt_spec
            backend = moe.routed_experts.backend
            loaded_slots = getattr(backend, spec.loaded_slots_attr, set())
            expected_slots = set(range(backend.expert_size_per_partition))
            if loaded_slots != expected_slots:
                missing_slots = sorted(expected_slots - loaded_slots)
                raise RuntimeError(
                    "Kimi K3 streaming expert loading did not fill all backend "
                    f"slots; missing {missing_slots[:10]}."
                )
            if (
                spec.needs_layer_finalize
                and expected_slots
                and id(backend) not in finalized_backends
            ):
                # Unreachable via load_expert (the last slot finalizes), so
                # reaching it means the two bookkeeping paths disagree. Guarded
                # on expected_slots because a layer that owns no local slots is
                # never prepared and so is legitimately never finalized.
                raise RuntimeError(
                    "Kimi K3 streaming expert loading filled every slot but "
                    "never finalized the layer."
                )
            backend._weights_transformed = False

    def _finalize_weight_load(self, num_params: int, num_moe_layers: int) -> None:
        """Finalize checkpoint-loaded attention and optional BF16 MLP conversion."""
        for layer in self.model.layers:
            if not _has_weights(layer):
                continue
            if getattr(layer, "is_kda", False):
                attention = layer.linear_attn
                attention.finalize_decode_weights()
                attention._build_mtp_conv_weights()
            else:
                attention = layer.self_attn.mixer
            for name, linear in attention.named_children():
                if isinstance(linear, TrtllmLinear) and linear.has_fp8_block_scales:
                    _swap_linear_to_fp8_weight_read(attention, name, (TrtllmLinear,))
            fused = getattr(attention, "qkvg_proj", None)
            if isinstance(fused, _Fp8BlockScaleWeightReadLinear):
                offset = 0
                for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
                    projection = getattr(attention, name)
                    projection._fused_projection = (weakref.ref(fused), offset)
                    offset += projection.out_features
        logger.info(
            f"Kimi K3: loaded {num_params} parameters and expert slices of {num_moe_layers} layers"
        )
        if self._fp8_weight_read_moe_mlp and is_sm_100f():
            converted = _convert_moe_mlps_to_fp8_weight_read(
                self.model, include_fused_gate_up=self.model_config.mapping.enable_attention_dp
            )
            logger.info(f"Kimi K3: quantized {converted} BF16 shared/latent MLP projections to FP8")
