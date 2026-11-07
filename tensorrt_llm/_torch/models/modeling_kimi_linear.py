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
KDA states use ``KimiK3HybridCacheManager`` with the ``KvCacheConfig``
default ``use_kv_cache_manager_v2=True`` (selected in ``pyexecutor/_util.py``).
Setting this field to ``"auto"`` falls back to V1 for
``KimiLinearForCausalLM``, which declares no V2 model preference.
Per layer, the state consists of a short-conv slot of
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
opt-in via ``kv_cache_config.enable_block_reuse=true`` on the default V2
manager. Configuring ``mamba_state_config.periodic_snapshot_interval``
adds periodic KDA state snapshots; reuse uses FORCE_CHUNK context chunking.

Not supported: pipeline parallelism, draft-head spec-dec modes
(MTP/Eagle — no draft-head checkpoint exists). SA speculative decoding
is validated only without block reuse.
"""

from __future__ import annotations

import copy
import enum
import functools
import gc
import json
import math
import os
import threading
import weakref
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from safetensors import safe_open
from torch import nn

from ..._utils import get_sm_version, is_sm_100f
from ...logger import logger
from ...mapping import Mapping
from ...models.modeling_utils import QuantAlgo, QuantConfig
from ..attention.attention import _helix_cp_pad
from ..attention.backends import AttentionMetadata
from ..cute_dsl_utils import sm_reserve
from ..distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
    MnnvlPendingResult,
    MnnvlWorkspaceResult,
    allgather,
)
from ..distributed.ops import MnnvlOwnWorkspace, MnnvlPushTarget
from ..distributed.symm_mem_allgather import SymmetricMemoryAllGather
from ..kimi_k3_cache_policy import KIMI_K3_FP8_KV_LAYERS_ENV, get_kimi_k3_fp8_kv_layer_ids
from ..model_config import ModelConfig
from ..modules.dwdp.dense.manager import get_global_dense_dwdp_manager
from ..modules.gated_mlp import GatedMLP
from ..modules.helix_tail import HelixTail
from ..modules.kimi_kda import KimiKDALinearAttention
from ..modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
from ..modules.linear import (
    Fp8BlockScaleLowMGemmRange,
    TensorParallelMode,
    fp8_block_scale_ue8m0_gemm,
    load_weight_shard,
)
from ..modules.linear import Linear as TrtllmLinear
from ..modules.multi_stream_utils import do_multi_stream, maybe_execute_in_parallel_deferred_join
from ..modules.projection_capabilities import (
    GatedEpilogueProjection,
    OutputAllReduceProjection,
    PrequantizedInputProjection,
    reduces_output,
)
from ..modules.rms_norm import RMSNorm
from ..modules.situ import SituAndMul
from ..moe.fused_moe import (
    ConfigurableMoE,
    MoEImplBase,
    SiTuActivation,
    TRTLLMGenFusedMoE,
    create_moe,
)
from ..moe.fused_moe.impl_contract import MoESortTables, PublishedSortTables, SortedRouting
from ..moe.fused_moe.interface import MoESchedulerKind
from ..moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod
from ..nccl_window_tensor_scope import nccl_window_tensor_scoped_module
from ..pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, RuntimeHooks
from ..speculative.spec_head_kernels import spec_decode_head
from ..utils import AuxStreamType, Fp4QuantizedTensor
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
    from ...quantization.utils.fp8_utils import Fp8BlockScaleWeightSliceCache

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

# The two token-count bounds of the MoE front's issue order (``KimiK3MoERuntime._front_plan``).
# Expected routed rows per rank (tokens x top-k x local experts / experts) from which the routed
# experts wait for the shared expert's GEMMs.
_SHARED_GEMMS_LEAD_MIN_ROUTED_ROWS = 80
# Largest token count for which the router GEMM is issued before latent_down.
_ROUTER_FIRST_MAX_TOKENS = 64
# Largest token count for which the sharded decode front forks the shared expert after its head
# instead of before it: while the head streams the router and latent_down weights it needs the
# SMs a fork at the start would hand to the shared GEMMs. The bound is the batch range where that
# ordering is the faster one.
_MOE_FRONT_FORK_AT_HEAD_MAX_TOKENS = 16
# The router-logits kernel pays where the K3 latent stream lets its result be consumed
# early: measured on GB300 (SM103) and VR200 (SM107). SM100 keeps ``dsv3_router_gemm_op``:
# without the overlap the shorter launch only shortens the route tail it replaced.
_ROUTER_LOGITS_KERNEL_SMS = ((10, 3), (10, 7))
# SMs the shared expert's GEMMs keep to while they run beside routed experts whose backend leaves
# that many SMs free of its persistent grid (``MoEImplBase.reserve_sms_for_concurrent_kernels``).
_SHARED_EXPERT_SMS_BESIDE_ROUTED_GRID = 16
# The routed latent all-reduce (+ RMSNorm) of the MoE tail runs on an MNNVL workspace of its own
# (``_latent_all_reduce_workspace``), one-shot while all ranks together send at most this many
# latent rows.
_LATENT_ALL_REDUCE_ONE_SHOT_MAX_ROWS = 1316


def _latent_all_reduce_workspace(latent_size: int) -> MnnvlOwnWorkspace:
    """The own MNNVL workspace of the routed latent all-reduce of ``latent_size``-wide BF16 rows.

    It does not order against the shared expert's all-reduce on the group's workspace. One call
    per MoE layer, a decoder layer apart, so the kernel takes the workspace state early.
    """
    return MnnvlOwnWorkspace(
        "kimi_k3_routed_latent",
        _LATENT_ALL_REDUCE_ONE_SHOT_MAX_ROWS * latent_size * torch.bfloat16.itemsize,
        early_state=True,
    )


@contextmanager
def _cublas_sm_limit(num_sms: int) -> Iterator[None]:
    """Run the cuBLAS GEMMs issued inside on ``num_sms`` SMs (no limit for 0).

    Uses torch's SM carveout, which cuBLAS reads when a GEMM is issued, so CUDA graph capture
    records the confined launch.
    """
    if num_sms <= 0:
        yield
        return
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    previous = torch._C._get_sm_carveout_experimental()
    torch._C._set_sm_carveout_experimental(total - num_sms)
    try:
        yield
    finally:
        torch._C._set_sm_carveout_experimental(previous)


class _MoEFrontPlan(NamedTuple):
    """Issue order of one decode MoE front on the latent stream (``KimiK3MoERuntime._front_plan``).

    ``router_first``: the router GEMM is issued before latent_down. ``shared_first``: the routed
    experts wait for the shared expert's GEMMs. ``nvfp4_scale``: global scale of the NVFP4 input the
    latent chain hands the routed experts (``None``: BF16). ``latent_down_nvfp4``: latent_down's
    epilogue produces it (else the latent chain quantizes latent_down's output).
    """

    router_first: bool
    shared_first: bool
    nvfp4_scale: Optional[torch.Tensor]
    latent_down_nvfp4: bool


class KimiK3MoEGate(nn.Module):
    """Kimi K3 gate weights and routing method for ``ConfigurableMoE``."""

    def __init__(
        self,
        config: Any,
        *,
        logits_gemm_dtype: torch.dtype | None = None,
        logits_kernel: bool = False,
        device: torch.device | None = None,
    ) -> None:
        """``logits_kernel``: compute the logits of up to ``kimi_k3_router_logits.MAX_M`` BF16
        tokens with that CuTe kernel (bit-identical to ``dsv3_router_gemm_op``; it reads the
        weight before its programmatic-dependent-launch wait)."""
        from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        super().__init__()
        self.config = config
        self.logits_kernel = logits_kernel and IS_CUTLASS_DSL_AVAILABLE
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
            if self.logits_kernel and not torch.compiler.is_compiling():
                from ..moe.kernels.blackwell import kimi_k3_router_logits

                x = hidden_2d.contiguous()
                if (
                    kimi_k3_router_logits.can_run(x, self.weight)
                    # Measured SMs only (_ROUTER_LOGITS_KERNEL_SMS): an unmeasured SM keeps
                    # the kernel it replaces.
                    and torch.cuda.get_device_capability(x.device) in _ROUTER_LOGITS_KERNEL_SMS
                ):
                    return torch.ops.trtllm.kimi_k3_router_logits(x, self.weight)
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
"""Set to ``0`` to run every attention-residual mix as the PyTorch reference.
Default: the fused kernels (``trtllm::attn_res_fwd``, ``_attn_res_kernel``)
where they serve the shape. ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS`` bounds the
per-token fused mixes (decode) without touching ``trtllm::attn_res_fwd``."""

_FUSED_ATTN_RES_ENABLED = os.environ.get(KIMI_K3_FUSED_ATTN_RES_ENV, "1") == "1"


KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV = "KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS"
"""Largest token count of an attention-residual mix (+ RMSNorm) that runs on a per-token fused
kernel: the CuTe DSL decode kernel where it serves the shape (``kimi_k3_attn_res``: up to 192
tokens; SM100, SM103, and SM107 with a Rubin-capable CuTe DSL), else the C++ per-token ops
(``trtllm::attn_res_rmsnorm_fwd`` / ``attn_res_add_rmsnorm_fwd``: H == 7168, N <= 12). A positive
integer, default 192. ``1`` keeps the per-token kernels to single-token decode. Larger mixes take
the persistent kernel with ``KIMI_K3_ATTN_RES_TOPOLOGY=1``, else run unfused. See
``_attn_res_kernel``.

Read at import, so CUDA-graph capture and replay cannot disagree about it.
"""

KIMI_K3_ATTN_RES_TOPOLOGY_ENV = "KIMI_K3_ATTN_RES_TOPOLOGY"
"""``per_token`` (default): attention-residual mixes (+ RMSNorm) that no per-token kernel takes
(above ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS`` tokens, or a shape those kernels do not serve) run
unfused. ``1`` (alias ``split``): they run on the persistent kernel (one CTA per SM looping over
tokens) when H == 7168 and 2 <= N <= 9. See ``_attn_res_kernel``.

Off by default because the persistent kernel measures as no-regression on
NVFP4 but as a 2.98x context-phase regression on W4A8_MXFP4_MXFP8, through a
downstream interaction with the TRTLLM-gen MoE rather than through this kernel.

Read at import, so CUDA-graph capture and replay cannot disagree about it.
"""


def _read_persistent_attn_res() -> bool:
    raw = os.environ.get(KIMI_K3_ATTN_RES_TOPOLOGY_ENV, "per_token")
    if raw == "persistent":
        logger.warning(
            f"{KIMI_K3_ATTN_RES_TOPOLOGY_ENV}=persistent is treated as '1': the per-token kernels "
            f"serve the mixes up to {KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV} tokens and the "
            "persistent kernel the larger ones"
        )
    if raw not in ("per_token", "1", "split", "persistent"):
        # Loudly: a mistyped A/B arm that silently fell back to the default would
        # measure one side twice and report no difference.
        raise ValueError(
            f"{KIMI_K3_ATTN_RES_TOPOLOGY_ENV} must be 'per_token', '1', 'split' or 'persistent', "
            f"got {raw!r}"
        )
    return raw != "per_token"


def _read_fused_attn_res_max_tokens() -> int:
    raw = os.environ.get(KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV, "192")
    try:
        value = int(raw)
    except ValueError:
        # Loudly, for the same reason as the topology above.
        raise ValueError(
            f"{KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV} must be a positive integer, got {raw!r}"
        ) from None
    if value < 1:
        raise ValueError(f"{KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS_ENV} must be >= 1, got {value}")
    return value


_PERSISTENT_ATTN_RES_ENABLED = _read_persistent_attn_res()
_FUSED_ATTN_RES_MAX_TOKENS = _read_fused_attn_res_max_tokens()


class _AttnResKernel(enum.Enum):
    """Kernel of an attention-residual mix + RMSNorm, chosen by ``_attn_res_kernel``."""

    #: ``kimi_k3_attn_res`` (CuTe DSL, PDL prologue, a CTA cluster per token).
    DECODE = enum.auto()
    #: ``trtllm::attn_res_rmsnorm_fwd`` / ``attn_res_add_rmsnorm_fwd`` (C++, per token).
    PER_TOKEN = enum.auto()
    #: ``trtllm::attn_res_add_rmsnorm_persistent_fwd`` (one CTA per SM looping over tokens).
    PERSISTENT = enum.auto()
    #: ``_apply_attn_res`` and the module RMSNorm.
    UNFUSED = enum.auto()


@functools.cache
def _decode_attn_res_on_device(device_index: int) -> bool:
    from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        return False
    from ..kernels.blackwell import kimi_k3_attn_res

    return kimi_k3_attn_res.is_supported_device(device_index)


def _attn_res_decode_max_tokens(hidden: int, num_candidates: int, device_index: int) -> int:
    """Largest token count of a BF16 ``[M, hidden]`` mix over ``num_candidates`` on the device
    that the decode kernel serves (``kimi_k3_attn_res.supports`` up to
    ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS``); 0 where it serves none. ``_select_attn_res_kernel``
    picks the decode kernel exactly from 1 up to there."""
    if not _FUSED_ATTN_RES_ENABLED or not _decode_attn_res_on_device(device_index):
        return 0
    from ..kernels.blackwell import kimi_k3_attn_res

    max_tokens = min(kimi_k3_attn_res.MAX_TOKENS, _FUSED_ATTN_RES_MAX_TOKENS)
    return max_tokens if kimi_k3_attn_res.supports(max_tokens, hidden, num_candidates) else 0


@functools.cache
def _per_token_attn_res_on_device(device_index: int) -> bool:
    """Whether the C++ per-token attention-residual ops run here (built, SM10x)."""
    try:
        torch.ops.trtllm.attn_res_rmsnorm_fwd
        torch.ops.trtllm.attn_res_add_rmsnorm_fwd
    except (AttributeError, RuntimeError):
        return False
    return torch.cuda.get_device_capability(device_index)[0] == 10


def _select_attn_res_kernel(
    num_tokens: int, hidden: int, num_candidates: int, device_index: int
) -> _AttnResKernel:
    """``_attn_res_kernel`` for a BF16 CUDA mix of ``num_tokens`` rows of ``hidden``."""
    if not _FUSED_ATTN_RES_ENABLED:
        return _AttnResKernel.UNFUSED
    if num_tokens <= _FUSED_ATTN_RES_MAX_TOKENS:
        if _decode_attn_res_on_device(device_index):
            from ..kernels.blackwell import kimi_k3_attn_res

            if kimi_k3_attn_res.supports(num_tokens, hidden, num_candidates):
                return _AttnResKernel.DECODE
        if hidden == 7168 and num_candidates <= 12 and _per_token_attn_res_on_device(device_index):
            return _AttnResKernel.PER_TOKEN
    if _PERSISTENT_ATTN_RES_ENABLED and hidden == 7168 and 2 <= num_candidates <= 9:
        return _AttnResKernel.PERSISTENT
    return _AttnResKernel.UNFUSED


def _attn_res_kernel(prefix_sum: torch.Tensor, num_candidates: int) -> _AttnResKernel:
    """The kernel of the attention-residual mix + RMSNorm of ``prefix_sum [M, H]`` over
    ``num_candidates`` (the snapshots and the prefix sum).

    Up to ``KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS`` tokens (default 192) a per-token kernel: the CuTe
    DSL decode kernel where it serves the shape (``kimi_k3_attn_res.supports``: up to 192 tokens,
    on a device ``kimi_k3_attn_res.is_supported_device`` accepts), else the C++ per-token ops
    (H == 7168, N <= 12). Otherwise the persistent kernel where ``KIMI_K3_ATTN_RES_TOPOLOGY=1``
    and it serves the shape, else unfused. This is the only place that decides it: the fused
    mixes and the MoE layers' residual deferral (``_defers_moe_residual``) both ask here. The
    rows whose attention o_proj all-reduce the mix after attention finishes are sized at load
    time from ``_attn_res_decode_max_tokens``
    (``KimiLinearDecoderLayer.attention_output_reduction``); a mix that does not pick the decode
    kernel for such a result finishes it into a tensor (``MnnvlPendingResult.materialize``).
    """
    if prefix_sum.dtype is not torch.bfloat16 or not prefix_sum.is_cuda:
        return _AttnResKernel.UNFUSED
    num_tokens, hidden = prefix_sum.shape
    kernel = _select_attn_res_kernel(
        num_tokens, hidden, num_candidates, prefix_sum.device.index or 0
    )
    if kernel is _AttnResKernel.UNFUSED and _FUSED_ATTN_RES_ENABLED:
        _note_attn_res_fusion("attn_res+norm/select", False, num_tokens, hidden, num_candidates)
    return kernel


# Largest token count per trtllm::attn_res_fwd call (cpp/tensorrt_llm/thop/attnResOp.cpp).
_ATTN_RES_OP_MAX_TOKENS = 16384


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
    if K + 1 > 12 or not (4096 <= H <= 8192 and H % 1024 == 0):
        return None
    try:
        attn_res_op = torch.ops.trtllm.attn_res_fwd
    except (AttributeError, RuntimeError):
        return None
    proj_weight = proj.weight.reshape(-1).to(torch.bfloat16).contiguous()
    norm_weight = norm.weight.to(torch.bfloat16).contiguous()

    def run(prefix: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
        T = prefix.shape[0]
        mixed, _rsigma, _probs, _logits = attn_res_op(
            prefix.reshape(T, 1, H).contiguous(),
            block.reshape(K, T, 1, H).contiguous(),
            proj_weight,
            norm_weight,
            float(norm.eps),
        )
        return mixed.reshape(T, H)

    if M <= _ATTN_RES_OP_MAX_TOKENS:
        return run(prefix_sum, block_residual)
    # The op is per-token, so prefill chunks above its token limit run in slices.
    output = torch.empty_like(prefix_sum)
    for start in range(0, M, _ATTN_RES_OP_MAX_TOKENS):
        end = min(M, start + _ATTN_RES_OP_MAX_TOKENS)
        output[start:end] = run(prefix_sum[start:end], block_residual[:, start:end])
    return output


def _rms_norm_eps(norm: nn.Module) -> float:
    if hasattr(norm, "eps"):
        return float(norm.eps)
    return float(norm.variance_epsilon)


class _DeferredMoEResidual(NamedTuple):
    """A MoE layer's residual update ``residual + bf16(routed + shared)``, left to the next
    attention-residual mix, which folds it into its kernel (or materializes it). After the
    column-parallel tail ``routed`` is the pending all-gather call that stands for
    ``bf16(routed + shared)`` and ``shared`` is None (``KimiK3MoERuntime._forward_parts``)."""

    residual: torch.Tensor
    routed: Union[torch.Tensor, MnnvlPendingResult]
    shared: Optional[torch.Tensor]

    def materialize(self) -> torch.Tensor:
        if isinstance(self.routed, MnnvlPendingResult):
            return self.residual + self.routed.materialize()
        return self.residual + (self.routed + self.shared)


def _defers_moe_residual(prefix_sum: torch.Tensor, num_candidates: int) -> bool:
    """Whether a MoE layer leaves its residual update to the next layer's pre-attention mix (or
    the model's output mix) over ``num_candidates``: where the decode kernel serves that mix and
    folds the update in."""
    return _attn_res_kernel(prefix_sum, num_candidates) is _AttnResKernel.DECODE


def _fold_moe_residual(
    deferred: _DeferredMoEResidual,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``(prefix_sum, normed mix)``: the deferred MoE residual update folded into the mix over
    ``block_residual`` where the decode kernel serves it, else ``(materialized update, None)``
    and the caller runs the mix."""
    if block_residual.shape[0] > 0:
        folded = _attn_res_decode_kernel(
            deferred.residual,
            (deferred.routed, deferred.shared),
            block_residual,
            proj,
            norm,
            output_norm,
        )
        if folded is not None:
            return folded[0], folded[1]
    return deferred.materialize(), None


class _AttnResVariant(NamedTuple):
    """A compiled variant of the decode attention-residual kernel (``kimi_k3_attn_res``)."""

    num_candidates: int
    add_mode: int
    cluster: int
    mxfp8: bool
    snapshots_after_wait: bool


def _decode_attn_res_variants(
    layers: Sequence[nn.Module],
    attn_res_block_size: int,
    hidden: int,
    max_tokens: int,
    device_index: int,
    helix_a2a_first_layer: int,
    verify_batches: bool,
) -> set[_AttnResVariant]:
    """The decode-kernel variants ``KimiLinearModel.forward`` can launch on mixes of up to
    ``max_tokens`` rows, from the layers' kinds and the snapshot schedule (``layer_idx``,
    ``is_kda``, ``is_moe``, ``token_sliced``) and ``_attn_res_kernel``.

    ``verify_batches``: speculative-verify batches run, so a KDA layer's mix that folds in the
    previous MoE residual may also emit the MXFP8 form of its output (for the row counts its
    mixer's fused projection takes, ``KimiKDALinearAttention.takes_prequantized_input``). Mixes
    on captured layers run unfused and are not separated out: this is a superset.
    """
    from ..kernels.blackwell import kimi_k3_attn_res as kernel

    token_counts = range(1, min(max_tokens, kernel.MAX_TOKENS) + 1)
    variants = set()

    def add(num_candidates, add_mode, mxfp8=False, after_wait=False, takes=None):
        for m in token_counts:
            if takes is not None and not takes(m):
                continue
            if (
                _select_attn_res_kernel(m, hidden, num_candidates, device_index)
                is _AttnResKernel.DECODE
            ):
                cluster = kernel.cluster_size(m, num_candidates)
                variants.add(_AttnResVariant(num_candidates, add_mode, cluster, mxfp8, after_wait))

    num_snapshots = 0
    previous_moe = False
    for i, layer in enumerate(layers):
        if num_snapshots > 0:
            # Pre-attention mix: a MoE predecessor's residual update is folded in where this
            # kernel runs (``_defers_moe_residual`` asks the same selector), a plain row else.
            if previous_moe:
                add(num_snapshots + 1, kernel.ADD_MOE)
                if verify_batches and layer.is_kda and not layer.token_sliced:
                    takes = layer.linear_attn.qkvg_takes_prequantized
                    add(num_snapshots + 1, kernel.ADD_MOE, mxfp8=True, takes=takes)
            else:
                add(num_snapshots + 1, kernel.ADD_NONE, after_wait=i == helix_a2a_first_layer)
        snapshot = layer.layer_idx % attn_res_block_size == 0
        num_snapshots += snapshot
        if num_snapshots > 0:
            # Post-attention mix: the new snapshot is the prefix sum, else the attention output
            # is added to it.
            add(num_snapshots + 1, kernel.ADD_NONE if snapshot else kernel.ADD_PREFIX)
        previous_moe = layer.is_moe
    if num_snapshots > 0:
        # Output mix of the model.
        add(num_snapshots + 1, kernel.ADD_NONE)
    return variants


def _attn_res_decode_kernel(
    layer: torch.Tensor,
    addends: Tuple[Union[torch.Tensor, MnnvlWorkspaceResult], ...],
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
    mxfp8: bool = False,
    entry_trigger: bool = False,
    snapshots_after_wait: bool = False,
) -> Optional[Tuple[Optional[torch.Tensor], ...]]:
    """``(updated prefix sum or None, normed mix)`` from the decode attention-residual kernel
    (``kimi_k3_attn_res``: PDL prologue, cluster per token) where ``_attn_res_kernel`` picks it,
    else None.

    ``addends``: ``()``; ``(addend,)`` for ``layer + addend``, where ``addend`` may be an
    all-reduce result left in its workspace (read in place) or a pushed all-reduce left pending
    (finished by the kernel, before its grid wait where one-shot); ``(routed, shared)`` for the MoE
    residual update ``layer + bf16(routed + shared)``. With ``mxfp8`` (MoE residual update only)
    the result also holds the MXFP8 form ``(activation, scale)`` of the normed mix, and
    ``entry_trigger`` releases its consumer at kernel entry (``kimi_k3_attn_res``).
    ``snapshots_after_wait`` (no addends only): ``block_residual`` was written by the kernel
    launched immediately before, so the kernel reads it after its PDL wait.
    """
    num_candidates = int(block_residual.shape[0]) + 1
    if _attn_res_kernel(layer, num_candidates) is not _AttnResKernel.DECODE:
        return None
    M, H = layer.shape
    if snapshots_after_wait and addends:
        raise ValueError("snapshots_after_wait is only implemented for the mix without addends")
    if len(addends) == 2 and isinstance(addends[0], MnnvlPendingResult):
        return _attn_res_decode_finish_all_gather(
            layer, addends[0], block_residual, proj, norm, output_norm, mxfp8, entry_trigger
        )
    in_place = len(addends) == 1 and isinstance(addends[0], MnnvlWorkspaceResult)
    if in_place and tuple(addends[0].shape) != (M, H):
        return None
    # The kernel reads dense rows. ``layer`` and the addends are read after its PDL wait, so a copy
    # here is safe (and a no-op for the decoder's tensors). The weights and the snapshot bank are
    # read before the wait, so they are passed as they are: a conversion copy would be the grid
    # the kernel reads ahead of. The ops reject operands that are not contiguous BF16.
    layer = layer.contiguous()
    if not in_place:
        addends = tuple(addend.contiguous() for addend in addends)
    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    _note_attn_res_fusion(f"attn_res+norm/decode/add{len(addends)}", True, M, H, num_candidates)
    eps, out_eps = float(norm.eps), _rms_norm_eps(output_norm)
    if in_place and isinstance(addends[0], MnnvlPendingResult):
        pending = addends[0]
        world, rank, mc_base, two_shot, *_ = pending.finish_args
        return torch.ops.trtllm.kimi_k3_attn_res_add_rmsnorm_finish_addend(
            layer,
            pending.view.buffer,
            mc_base,
            pending.view.flags,
            world,
            rank,
            two_shot,
            # One-shot calls are finished before the grid wait (no cross-rank CTA dependency).
            not two_shot,
            block_residual,
            *weights,
            eps,
            out_eps,
            entry_trigger,
        )
    if in_place:
        view = addends[0].view
        return torch.ops.trtllm.kimi_k3_attn_res_add_rmsnorm_in_place_addend(
            layer, view.buffer, view.flags, block_residual, *weights, eps, out_eps
        )
    if not addends:
        normed = torch.ops.trtllm.kimi_k3_attn_res_rmsnorm(
            layer, block_residual, *weights, eps, out_eps, snapshots_after_wait
        )
        return None, normed
    if len(addends) == 1:
        return torch.ops.trtllm.kimi_k3_attn_res_add_rmsnorm(
            layer, addends[0], block_residual, *weights, eps, out_eps
        )
    if mxfp8:
        return torch.ops.trtllm.kimi_k3_attn_res_moe_add_rmsnorm_mxfp8(
            layer, *addends, block_residual, *weights, eps, out_eps, entry_trigger
        )
    return torch.ops.trtllm.kimi_k3_attn_res_moe_add_rmsnorm(
        layer, *addends, block_residual, *weights, eps, out_eps
    )


def _attn_res_persistent_kernel(
    prefix_sum: torch.Tensor,
    addend: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(updated prefix sum, normed mix)`` from the persistent attention-residual kernel, on the
    SMs left beside a kernel resident for the whole step (``cute_dsl_utils.sm_reserve``)."""
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    updated_prefix_sum, output = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
        prefix_sum.reshape(M, 1, H).contiguous(),
        None if addend is None else addend.reshape(M, 1, H).contiguous(),
        block_residual.reshape(K, M, 1, H).contiguous(),
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        _rms_norm_eps(output_norm),
        sm_reserve(),
    )
    site = "attn_res+norm/persistent" if addend is None else "add+attn_res+norm/persistent"
    _note_attn_res_fusion(site, True, M, H, K + 1)
    return updated_prefix_sum.reshape(M, H), output.reshape(M, H)


def _attn_res_per_token_kernel(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor | None,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """``(updated prefix sum or None, normed mix)`` from the C++ per-token attention-residual ops
    (``trtllm::attn_res_rmsnorm_fwd``, with an ``addend`` ``attn_res_add_rmsnorm_fwd``)."""
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    args = (
        block_residual.reshape(K, M, 1, H).contiguous(),
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )
    layer = prefix_sum.reshape(M, 1, H).contiguous()
    if addend is None:
        _note_attn_res_fusion("attn_res+norm/per_token", True, M, H, K + 1)
        return None, torch.ops.trtllm.attn_res_rmsnorm_fwd(layer, *args).reshape(M, H)
    updated_prefix_sum, normed = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
        layer, addend.reshape(M, 1, H).contiguous(), *args
    )
    _note_attn_res_fusion("add+attn_res+norm/per_token", True, M, H, K + 1)
    return updated_prefix_sum.reshape(M, H), normed.reshape(M, H)


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
    snapshots_after_wait: bool = False,
) -> Optional[torch.Tensor]:
    """Fuse attention-residual mixing with its immediately following norm, on the kernel
    ``_attn_res_kernel`` picks; None where it picks none. ``snapshots_after_wait``: see
    ``_attn_res_decode_kernel``."""
    kernel = _attn_res_kernel(prefix_sum, int(block_residual.shape[0]) + 1)
    if kernel is _AttnResKernel.DECODE:
        return _attn_res_decode_kernel(
            prefix_sum,
            (),
            block_residual,
            proj,
            norm,
            output_norm,
            snapshots_after_wait=snapshots_after_wait,
        )[1]
    if kernel is _AttnResKernel.PER_TOKEN:
        return _attn_res_per_token_kernel(
            prefix_sum, None, block_residual, proj, norm, output_norm
        )[1]
    if kernel is _AttnResKernel.PERSISTENT:
        return _attn_res_persistent_kernel(
            prefix_sum, None, block_residual, proj, norm, output_norm
        )[1]
    return None


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
    attention-residual selection. None where ``_attn_res_kernel`` picks no kernel.
    """
    if addend.dtype is not torch.bfloat16 or prefix_sum.shape != addend.shape:
        return None
    kernel = _attn_res_kernel(prefix_sum, int(block_residual.shape[0]) + 1)
    if kernel is _AttnResKernel.DECODE:
        return _attn_res_decode_kernel(
            prefix_sum, (addend,), block_residual, proj, norm, output_norm
        )
    if kernel is _AttnResKernel.PER_TOKEN:
        return _attn_res_per_token_kernel(
            prefix_sum, addend, block_residual, proj, norm, output_norm
        )
    if kernel is _AttnResKernel.PERSISTENT:
        return _attn_res_persistent_kernel(
            prefix_sum, addend, block_residual, proj, norm, output_norm
        )
    return None


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
    snapshots_after_wait: bool = False,
) -> torch.Tensor:
    """Apply attention-residual selection and the next RMSNorm.

    ``snapshots_after_wait``: ``block_residual`` was written by the kernel launched immediately
    before this call (``_attn_res_decode_kernel``).
    """
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_rmsnorm_fused(
            prefix_sum, block_residual, proj, norm, output_norm, snapshots_after_wait
        )
        if fused is not None:
            return fused
    return output_norm(_apply_attn_res(prefix_sum, block_residual, proj, norm))


def _attn_res_decode_finish_all_gather(
    layer: torch.Tensor,
    pending: MnnvlPendingResult,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
    mxfp8: bool,
    entry_trigger: bool = False,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """The MoE residual update of the column-parallel tail folded into the decode mix: the mix
    finishes the pending all-gather call (before its grid wait) whose all-gather rows are the
    routed output and whose slot sum is the shared expert's all-reduce. ``entry_trigger`` (with
    ``mxfp8``): as in :func:`_attn_res_decode_kernel`."""
    if not pending.all_gather or tuple(pending.shape) != tuple(layer.shape):
        return None
    world, rank, mc_base, _, chunk, sanitize, _ = pending.finish_args
    weights = (proj.weight.reshape(-1), norm.weight, output_norm.weight)
    M, H = layer.shape
    _note_attn_res_fusion("attn_res+norm/decode/cp_tail", True, M, H, block_residual.shape[0] + 1)
    args = (
        layer.contiguous(),
        pending.view.buffer,
        mc_base,
        pending.view.flags,
        world,
        rank,
        chunk,
        sanitize,
        block_residual.contiguous(),
        *weights,
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )
    if mxfp8:
        return torch.ops.trtllm.kimi_k3_attn_res_moe_finish_rmsnorm_mxfp8(*args, entry_trigger)
    return torch.ops.trtllm.kimi_k3_attn_res_moe_finish_rmsnorm(*args)


def _apply_attn_res_add_and_rmsnorm(
    prefix_sum: torch.Tensor,
    addend: Union[torch.Tensor, MnnvlWorkspaceResult],
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiK3RMSNorm,
    output_norm: nn.Module,
    entry_trigger: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add an attention output to the running residual, then select and norm.

    ``addend`` may be the attention output's all-reduce left in its workspace, finished or
    pending: the decode attention-residual kernel reads (and finishes) it there where it serves
    the mix, else it is materialized. ``entry_trigger``: the kernel that finishes a pending
    all-reduce releases its dependent at entry (the caller checked that the next launch is the
    sharded MoE front's head, ``KimiK3MoERuntime.moe_front_releases_early``).
    """
    if isinstance(addend, MnnvlWorkspaceResult):
        if _FUSED_ATTN_RES_ENABLED:
            decode = _attn_res_decode_kernel(
                prefix_sum,
                (addend,),
                block_residual,
                proj,
                norm,
                output_norm,
                entry_trigger=entry_trigger,
            )
            if decode is not None:
                return decode
        addend = addend.materialize()
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


def _rubin_mxfp8_is_available() -> bool:
    """Whether this rank can run the SM107 native block-scaled MXFP8 GEMM.

    Recomputed rather than cached: the SM version cannot change within a run,
    so weight preparation and forward always agree.
    """
    from ..cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def _low_m_gemm_nvfp4_max_m(out_features: int, in_features: int) -> int:
    """Largest token count for which the low-M GEMM emits the NVFP4 quantization of its output
    on the current device (0: never)."""
    from ..kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.nvfp4_out_max_m_for(out_features, in_features)


def _low_m_gemm_gated_applies(x: torch.Tensor, gate: torch.Tensor, out_features: int) -> bool:
    from ..kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.gate_fits(x, gate, out_features)


def _low_m_gemm_push_applies(num_rows: int, out_features: int, in_features: int) -> bool:
    from ..kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.all_reduce_push_applies(num_rows, out_features, in_features)


def _low_m_gemm_push_max_m(out_features: int, in_features: int) -> int:
    from ..kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.all_reduce_push_max_m_for(out_features, in_features)


def _low_m_gemm_scale_from_mxfp8(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The DeepGEMM-packed scale the low-M GEMM reads, from the SM107 CuTe MXFP8 scale."""
    from ..kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.pack_block_scale_from_mxfp8_layout(weight_scale, n, k)


class _Fp8BlockScaleWeightReadLinear(
    GatedEpilogueProjection, PrequantizedInputProjection, OutputAllReduceProjection
):
    """Bias-free FP8 block-scale GEMM for checkpoint attention and converted MLPs.

    Attention loads checkpoint E4M3 codes and 128x128 scales directly. The optional
    MLP path quantizes BF16 weights once at load. On SM100/SM103 the UE8M0
    scales are packed for DeepGEMM, which quantizes BF16 activations inside the
    GEMM. On Rubin (SM107) the scales are packed into the CuTe DSL K32 R128c4
    layout and the activation is quantized by the specialized CuTe quantizer
    before the native MXFP8 GEMM, which runs with K3's fine-grained low-M tuning
    buckets. On both, the low-M GEMM is a candidate of forward like in the
    generic FP8 block-scale linear (``Fp8BlockScaleLowMGemmRange``); it reads a
    DeepGEMM-packed scale, which is ``weight_scale`` on SM100/SM103 and a copy
    repacked from the MXFP8 scale on SM107.

    A row-parallel projection built with ``output_all_reduce`` returns the
    all-reduce of its partial sums from every forward. Where the low-M GEMM
    serves a forward and the all-reduce runs the MNNVL kernel, the GEMM pushes
    its output straight into the all-reduce's Lamport buffers and a finish
    kernel completes it, bitwise like the GEMM followed by the all-reduce. Up
    to ``in_place_max_rows`` rows the consumer reads the result in the
    workspace and finishes the all-reduce itself: such a forward returns the
    call left pending (``MnnvlPendingResult``), also from
    :meth:`forward_prequantized`, whose GEMM has no push epilogue (a copy
    kernel pushes its output).
    """

    def __init__(
        self,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        out_features: int,
        output_all_reduce: Optional[AllReduce] = None,
        in_place_max_rows: int = 0,
    ) -> None:
        super().__init__()
        self.in_features = weight_fp8.shape[1]
        self.out_features = out_features
        self._weights_transformed = True
        self._fused_projection = None
        self._output_slice_cache: Optional[Fp8BlockScaleWeightSliceCache] = None
        # Buffers (not parameters): these are the module's weights post-load;
        # there is nothing further to load into them and they must not be
        # touched by any later autocast/dtype move.
        self.register_buffer("weight", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        # Rows the low-M GEMM serves and the DeepGEMM-packed scale it reads; set at post-load.
        self._low_m = Fp8BlockScaleLowMGemmRange()
        self._low_m_weight_scale: Optional[torch.Tensor] = None
        # Largest row count :meth:`forward_nvfp4` serves (0: never); set at post-load.
        self._nvfp4_out_max_m = 0
        # Largest row count :meth:`forward` pushes into ``output_all_reduce`` from the low-M
        # GEMM (``all_reduce_push_max_m_for``, at least the GEMM's range; 0: never).
        self._push_max_m = 0
        self._output_all_reduce = output_all_reduce
        self._in_place_max_rows = in_place_max_rows
        # ``(first row, rows)`` of the weight :meth:`forward_rows_prequantized` projects onto, and
        # their FP8 rows and DeepGEMM-packed scale (built at post-load).
        self._output_rows = None
        self._output_rows_operands = None

    @property
    def has_fp8_qdq(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def has_w4a8_nvfp4_fp8(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def has_r128c4_weight_scale(self) -> bool:
        """Match the ``Linear`` interface: ``_prepare_weights`` packs R128c4 scales where the
        Rubin MXFP8 GEMM runs."""
        return _rubin_mxfp8_is_available()

    @property
    def output_all_reduce(self) -> Optional[AllReduce]:
        return self._output_all_reduce

    @property
    def supports_prequantized_input(self) -> bool:
        """Whether :meth:`forward_prequantized` serves every row count (SM107: the CuTe MXFP8
        GEMM)."""
        return _rubin_mxfp8_is_available()

    def takes_prequantized_input(self, num_rows: int) -> bool:
        """Whether a producer should hand :meth:`forward_prequantized` the MXFP8 form of
        ``num_rows`` rows instead of the BF16 rows: where :meth:`forward` runs the low-M GEMM, so
        an operand quantized with its arithmetic
        (``fp8_block_scale_low_m_gemm.mxfp8_quantize_reference``) projects bitwise like
        :meth:`forward` and the GEMM skips the quantization."""
        return 1 <= num_rows <= self._low_m.max_m

    def stages_weights(self, num_rows: int) -> bool:
        """Whether :meth:`forward_prequantized` with ``stage_weights`` holds each CTA's whole
        weight slice on chip before its ``griddepcontrol.wait`` for ``num_rows`` rows
        (``fp8_block_scale_low_m_gemm.weights_on_chip``), so that its producer may release it at
        the producer's entry."""
        from ..kernels.blackwell import fp8_block_scale_low_m_gemm

        return self.takes_prequantized_input(num_rows) and (
            fp8_block_scale_low_m_gemm.weights_on_chip(
                num_rows, self.out_features, self.in_features
            )
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
        from ...deep_gemm.utils.math import per_block_cast_to_fp8

        weight_fp8, weight_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        return _Fp8BlockScaleWeightReadLinear._prepare_weights(weight_fp8, weight_scale)

    @staticmethod
    def _prepare_weights(
        weight_fp8: torch.Tensor, weight_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from ...quantization.utils.fp8_utils import (
            resmooth_to_fp8_e8m0,
            transform_k128_scales_to_cutedsl_mxfp8_layout,
            transform_sf_into_required_layout,
        )

        # Both GEMMs consume UE8M0 scales rather than the checkpoint's FP32 grid.
        weight_fp8, weight_scale = resmooth_to_fp8_e8m0(
            weight_fp8.contiguous(), weight_scale.contiguous().float()
        )
        if _rubin_mxfp8_is_available():
            # cute_dsl_mxfp8_gemm_rubin reads K32 R128c4 UE8M0 scales.
            weight_scale = transform_k128_scales_to_cutedsl_mxfp8_layout(
                weight_scale, mn=weight_fp8.shape[0], k=weight_fp8.shape[1]
            )
            return weight_fp8, weight_scale
        # fp8_swap_ab_gemm with disable_ue8m0_cast=True consumes packed,
        # TMA-aligned UE8M0 scales.
        weight_scale = transform_sf_into_required_layout(
            weight_scale,
            mn=weight_fp8.shape[0],
            k=weight_fp8.shape[1],
            recipe=(1, 128, 128),
            is_sfa=False,
        )
        return weight_fp8, weight_scale

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear | TrtllmLinear,
        output_all_reduce: Optional[AllReduce] = None,
        in_place_max_rows: int = 0,
    ) -> "_Fp8BlockScaleWeightReadLinear":
        """The FP8 weight-read form of ``linear``; see the class for ``output_all_reduce`` and
        ``in_place_max_rows``."""
        assert linear.bias is None, "FP8 weight read expects a bias-free Linear"
        reduction = (output_all_reduce, in_place_max_rows)
        if isinstance(linear, TrtllmLinear) and linear.has_fp8_block_scales:
            weight, weight_scale = linear.weight.detach(), linear.weight_scale.detach()
            result = cls(weight, weight_scale, linear.out_features, *reduction)
            result.quant_config = linear.quant_config
            result.tp_size, result.tp_rank = linear.tp_size, linear.tp_rank
            result._weights_transformed = False
            return result
        weight_fp8, weight_scale = cls.quantize_weight(linear.weight.data)
        return cls(weight_fp8, weight_scale, linear.out_features, *reduction)

    def transform_weights(self) -> None:
        if not self._weights_transformed:
            self._output_slice_cache = None
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
        self._prepare_low_m_gemm()

    @property
    def is_fused_row_view(self) -> bool:
        """Whether this projection is a row view of a fused KDA projection (no low-M GEMM of
        its own)."""
        return self._fused_projection is not None

    def prepare_low_m_weight_scale(self) -> Optional[torch.Tensor]:
        """The DeepGEMM-packed weight scale the low-M GEMM reads, or None where that GEMM does
        not serve this weight. Runs this module's post-load first, for an owner whose own
        post-load the walk reaches before this module's."""
        self.post_load_weights()
        return self._low_m_weight_scale

    def _prepare_low_m_gemm(self) -> None:
        """Select the low-M GEMM for this weight where it has a DeepGEMM-packed scale.

        On SM107 that scale is a copy (N * K / 128 bytes) derived from the MXFP8
        ``weight_scale``; row views of a fused KDA projection have none (the fused module
        serves decode).
        """
        n, k = self.out_features, self.in_features
        self._low_m = Fp8BlockScaleLowMGemmRange.for_weight(n, k)
        if not _rubin_mxfp8_is_available():
            self._low_m_weight_scale = self.weight_scale
        elif self._low_m.max_m and self._fused_projection is None:
            self._low_m_weight_scale = _low_m_gemm_scale_from_mxfp8(self.weight_scale, n, k)
        else:
            self._low_m, self._low_m_weight_scale = Fp8BlockScaleLowMGemmRange(), None
        self._nvfp4_out_max_m = _low_m_gemm_nvfp4_max_m(n, k) if self._low_m.max_m else 0
        self._push_max_m = (
            _low_m_gemm_push_max_m(n, k)
            if self._low_m.max_m and self._output_all_reduce is not None
            else 0
        )
        self._output_rows_operands = None
        if self._output_rows is not None and self._low_m.max_m:
            c0, rows = self._output_rows
            scale = self._low_m_weight_scale
            if scale is not None and scale.dim() == 2 and scale.stride(0) == 1:
                # MN-major packed scale: one int32 row of N per four K blocks.
                rows_scale = scale.t()[:, c0 : c0 + rows].contiguous()
                self._output_rows_operands = (self.weight[c0 : c0 + rows], rows_scale)

    def set_output_rows(self, rows: Optional[Tuple[int, int]]) -> None:
        """Rows ``(first, count)`` of the weight (whole 128-row blocks) that
        :meth:`forward_rows_prequantized` projects onto; set before post-load."""
        if rows is not None and (rows[0] % 128 or rows[1] % 128 or sum(rows) > self.out_features):
            raise ValueError(f"output rows {rows} are not whole 128-row blocks of the weight")
        self._output_rows = rows

    def serves_output_rows(self, num_rows: int) -> bool:
        """Whether :meth:`forward_rows_prequantized` serves ``num_rows`` input rows: where
        :meth:`forward_prequantized` runs the low-M kernel, which the row slice runs too."""
        if self._output_rows_operands is None or not self.takes_prequantized_input(num_rows):
            return False
        from ..kernels.blackwell import fp8_block_scale_low_m_gemm

        n, k = self.out_features, self.in_features
        return num_rows <= min(
            fp8_block_scale_low_m_gemm.kernel_max_m_for(n, k),
            fp8_block_scale_low_m_gemm.low_m_max_for(self._output_rows[1], k),
        )

    def forward_rows_prequantized(
        self, activation: torch.Tensor, activation_scale: torch.Tensor
    ) -> torch.Tensor:
        """Output columns ``[first, first + count)`` of :meth:`forward_prequantized` (see
        :meth:`set_output_rows`), bitwise: the same kernel at the whole projection's tactic on
        the weight's row slice. For row counts where :meth:`serves_output_rows`."""
        from ..kernels.blackwell import fp8_block_scale_low_m_gemm

        weight, scale = self._output_rows_operands
        m = activation.shape[0]
        out = torch.empty((m, weight.shape[0]), dtype=torch.bfloat16, device=activation.device)
        return fp8_block_scale_low_m_gemm.run_low_m_gemm_prequantized(
            activation,
            activation_scale,
            weight,
            scale,
            out,
            tactic=fp8_block_scale_low_m_gemm.default_prequantized_tactic(
                m, self.out_features, self.in_features
            ),
        )

    def forward_slice(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        self.prepare_output_slice(start, end)
        assert self._output_slice_cache is not None
        weight, scale, crop = self._output_slice_cache.get(self.weight, start, end)
        out = self._apply_with_weights(x, weight, scale)
        return out[..., crop : crop + end - start].contiguous()

    def prepare_output_slice(self, start: int, end: int) -> None:
        """Prepare fixed output-slice scales after checkpoint transformation."""
        from ...quantization.utils.fp8_utils import Fp8BlockScaleWeightSliceCache

        self.transform_weights()
        if self._output_slice_cache is None:
            self._output_slice_cache = Fp8BlockScaleWeightSliceCache()
        self._output_slice_cache.prepare(self.weight, self.weight_scale, start, end)

    def _push_target(self, x_flat: torch.Tensor, all_reduce_params: Optional[AllReduceParams]):
        """The all-reduce of this forward's output as a target the low-M GEMM pushes into, or
        None where the GEMM stores its output and the all-reduce runs after it."""
        if self._output_all_reduce is None:
            return None
        num_rows = x_flat.shape[0]
        # Above the GEMM's range where its push beats the default GEMM + all-reduce.
        serves = self._low_m.serves(x_flat) or (
            1 <= num_rows <= self._push_max_m
            and x_flat.dtype == torch.bfloat16
            and x_flat.is_contiguous()
        )
        if not serves or not _low_m_gemm_push_applies(
            num_rows, self.out_features, self.in_features
        ):
            return None
        return self._output_all_reduce.push_target(
            num_rows,
            self.out_features,
            all_reduce_params=all_reduce_params,
            consumer_finishes=num_rows <= self._in_place_max_rows,
        )

    def _prequantized_push_target(self, num_rows: int):
        """:meth:`_push_target` of :meth:`forward_prequantized`, where it runs the low-M GEMM."""
        if self._output_all_reduce is None or not self.takes_prequantized_input(num_rows):
            return None
        if not _low_m_gemm_push_applies(num_rows, self.out_features, self.in_features):
            return None
        return self._output_all_reduce.push_target(
            num_rows,
            self.out_features,
            all_reduce_params=None,
            consumer_finishes=num_rows <= self._in_place_max_rows,
        )

    def _push(self, x_flat: torch.Tensor, target, gate: Optional[torch.Tensor] = None):
        """The low-M GEMM pushing into ``target``, then the all-reduce's finish."""
        from ..kernels.blackwell import fp8_block_scale_low_m_gemm

        fp8_block_scale_low_m_gemm.run_low_m_gemm_all_reduce_push(
            x_flat, self.weight, self._low_m_weight_scale, target.view, target.one_shot, gate=gate
        )
        return self._output_all_reduce.finish_push(target)

    def _reduce(
        self, out: torch.Tensor, all_reduce_params: Optional[AllReduceParams]
    ) -> torch.Tensor:
        if self._output_all_reduce is None:
            return out
        return self._output_all_reduce(out, all_reduce_params=all_reduce_params)

    def forward(
        self,
        x: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        layer_idx: Optional[int] = None,
    ) -> Union[torch.Tensor, MnnvlWorkspaceResult]:
        x_flat = x.reshape(-1, x.shape[-1])
        push_target = self._push_target(x_flat, all_reduce_params)
        if push_target is not None:
            out = self._push(x_flat, push_target)
            if isinstance(out, MnnvlWorkspaceResult):
                return out
            return out.reshape(*x.shape[:-1], self.out_features)
        out = self._apply_with_weights(x, self.weight, self.weight_scale)
        return self._reduce(out.reshape(x_flat.shape[0], -1), all_reduce_params).reshape(out.shape)

    def _apply_with_weights(
        self, x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        out_shape = (*x.shape[:-1], weight.shape[0])
        x_flat = x.reshape(-1, x.shape[-1])
        if not _rubin_mxfp8_is_available():
            low_m = self._low_m if weight is self.weight else Fp8BlockScaleLowMGemmRange()
            out = fp8_block_scale_ue8m0_gemm(
                x_flat, weight, weight_scale, low_m, output_dtype=x.dtype
            )
        elif weight is self.weight and self._low_m.serves(x_flat):
            out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(
                x_flat, self.weight, self._low_m_weight_scale
            )
        else:
            act_fp8, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x_flat)
            out = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                act_fp8,
                weight,
                act_scale,
                weight_scale,
                output_dtype=torch.bfloat16,
                fine_grained_m=True,
            )
        return out.reshape(out_shape)

    def emits_nvfp4(self, num_rows: int) -> bool:
        """Whether :meth:`forward_nvfp4` serves ``num_rows`` rows; never for a projection that
        reduces its output."""
        return self._output_all_reduce is None and 1 <= num_rows <= self._nvfp4_out_max_m

    def forward_nvfp4(
        self, x: torch.Tensor, global_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``fp4_quantize`` (``global_scale``, 16-element groups, linear scales) of the
        projection, from one low-M GEMM whose epilogue quantizes (bitwise the GEMM at the same
        tactic followed by ``fp4_quantize``). Returns E2M1 codes ``uint8 [M, out / 2]`` and E4M3
        scales ``uint8 [M * out / 16]``. For ``num_rows`` where :meth:`emits_nvfp4`."""
        if self._output_all_reduce is not None:
            raise RuntimeError("forward_nvfp4 of a projection that reduces its output")
        return torch.ops.trtllm.fp8_block_scale_low_m_gemm_nvfp4(
            x.reshape(-1, x.shape[-1]), self.weight, self._low_m_weight_scale, global_scale
        )

    def forward_gated(
        self, x: torch.Tensor, gate: torch.Tensor
    ) -> Union[torch.Tensor, MnnvlWorkspaceResult]:
        """``forward(x * gate)``, bit for bit, for an output-gated projection.

        Where :meth:`forward` would run the low-M kernel, one gated launch multiplies inside
        the GEMM (``gate`` may then be a row-strided BF16 view, see ``check_gate``).
        """
        x_flat = x.reshape(-1, x.shape[-1])
        gate_flat = gate.reshape(x_flat.shape)
        if self._low_m.serves(x_flat, kernel=True) and _low_m_gemm_gated_applies(
            x_flat, gate_flat, self.out_features
        ):
            out_shape = (*x.shape[:-1], self.out_features)
            push_target = self._push_target(x_flat, None)
            if push_target is not None:
                out = self._push(x_flat, push_target, gate=gate_flat)
                return out if isinstance(out, MnnvlWorkspaceResult) else out.reshape(out_shape)
            out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(
                x_flat, self.weight, self._low_m_weight_scale, gate=gate_flat
            )
            return self._reduce(out, None).reshape(out_shape)
        return self.forward(x * gate)

    def forward_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """``forward(x).sigmoid()``, bit for bit; one launch where :meth:`forward` would run
        the low-M kernel. Not for a projection that reduces its output."""
        if self._output_all_reduce is not None:
            raise RuntimeError("forward_sigmoid of a projection that reduces its output")
        x_flat = x.reshape(-1, x.shape[-1])
        if self._low_m.serves(x_flat, kernel=True):
            out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(
                x_flat, self.weight, self._low_m_weight_scale, sigmoid=True
            )
            return out.reshape(*x.shape[:-1], self.out_features)
        return self.forward(x).sigmoid()

    def forward_prequantized(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        stage_weights: bool = False,
    ) -> Union[torch.Tensor, MnnvlWorkspaceResult]:
        """Project an FP8 activation a fused producer already quantized.

        Runs the low-M GEMM where :meth:`takes_prequantized_input`, the CuTe MXFP8 GEMM at any
        other row count where :attr:`supports_prequantized_input`. Where it runs the low-M GEMM
        a projection built with ``output_all_reduce`` pushes from the GEMM like :meth:`forward`;
        otherwise up to ``in_place_max_rows`` rows a copy kernel pushes the output. Up to
        ``in_place_max_rows`` rows the all-reduce is left pending for its consumer.
        ``stage_weights``: the low-M GEMM holds its weights on chip before its wait where
        :meth:`stages_weights` (same bits).

        Args:
            activation: ``[..., in_features]`` ``float8_e4m3fn`` activation.
            activation_scale: ``uint8`` UE8M0 1x32 scales in the CuTe K32 R128c4
                layout (``fp8_quantize_1x128_cutedsl_ue8m0`` replicates 1x128 scales
                into it).
        """
        if activation.dtype is not torch.float8_e4m3fn:
            raise TypeError("prequantized Kimi K3 activation must be float8_e4m3fn")
        if activation_scale.dtype is not torch.uint8:
            raise TypeError("prequantized Kimi K3 activation scale must be uint8 R128c4")
        activation_flat = activation.reshape(-1, activation.shape[-1])
        num_rows = activation_flat.shape[0]
        push_target = self._prequantized_push_target(num_rows)
        if push_target is not None:
            from ..kernels.blackwell import fp8_block_scale_low_m_gemm

            fp8_block_scale_low_m_gemm.run_low_m_gemm_prequantized_all_reduce_push(
                activation_flat,
                activation_scale,
                self.weight,
                self._low_m_weight_scale,
                push_target.view,
                push_target.one_shot,
            )
            out = self._output_all_reduce.finish_push(push_target)
            if isinstance(out, MnnvlWorkspaceResult):
                return out
            return out.reshape(activation.shape[:-1] + (self.out_features,))
        if self.takes_prequantized_input(num_rows):
            out = torch.ops.trtllm.fp8_block_scale_low_m_gemm_prequantized(
                activation_flat,
                activation_scale,
                self.weight,
                self._low_m_weight_scale,
                stage_weights,
            )
        elif not self.supports_prequantized_input:
            raise RuntimeError(f"no GEMM for a prequantized Kimi K3 activation of {num_rows} rows")
        else:
            out = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                activation_flat,
                self.weight,
                activation_scale,
                self.weight_scale,
                output_dtype=torch.bfloat16,
                fine_grained_m=True,
            )
        target = self._pending_push_target(num_rows)
        if target is not None:
            self._output_all_reduce.push_input(target, out)
            return self._output_all_reduce.finish_push(target)
        return self._reduce(out, None).reshape(activation.shape[:-1] + (self.out_features,))

    def _pending_push_target(self, num_rows: int):
        """The all-reduce of a ``num_rows``-row output a copy kernel pushes, left pending for its
        consumer, or None (no gain over the C++ all-reduce when a finish kernel would follow)."""
        if self._output_all_reduce is None or not 1 <= num_rows <= self._in_place_max_rows:
            return None
        return self._output_all_reduce.push_target(
            num_rows, self.out_features, consumer_finishes=True
        )


def _bf16_weight(linear: nn.Module) -> Optional[torch.Tensor]:
    """The BF16 ``[out, in]`` weight of a bias-free projection: a BF16 linear's own, or an FP8
    weight-read projection's codes times their 128x128 block scales (exact in BF16); else None."""
    if isinstance(linear, _Fp8BlockScaleWeightReadLinear):
        from ..kernels.blackwell import kimi_k3_moe_tail

        packed = linear.prepare_low_m_weight_scale()
        n, k = linear.out_features, linear.in_features
        if packed is None or n % 128 or k % 128:
            return None
        exps = kimi_k3_moe_tail.block_exponents(packed, n, k).float() - 127.0
        scale = torch.exp2(exps).repeat_interleave(128, 0).repeat_interleave(128, 1)
        return (linear.weight.float() * scale).to(torch.bfloat16).contiguous()
    if isinstance(linear, TrtllmLinear) and linear.weight.dtype == torch.bfloat16:
        return linear.weight.data.contiguous()
    return None


def _swap_linear_to_fp8_weight_read(
    parent: nn.Module,
    attr: str,
    linear_types: Tuple[type, ...] = (nn.Linear,),
    output_all_reduce: Optional[AllReduce] = None,
    in_place_max_rows: int = 0,
) -> int:
    """Replace ``parent.<attr>`` with an FP8 weight-read module if it is a
    plain linear of one of ``linear_types``; return the number of modules
    converted (0 or 1), so callers can accumulate a conversion count.
    ``output_all_reduce`` and ``in_place_max_rows``: see
    ``_Fp8BlockScaleWeightReadLinear``.

    Transfer checkpoint FP8 storage or release the original BF16 storage
    after conversion. The loader's transient parameter map must not retain
    an unused BF16 copy until loading completes.
    """
    child = getattr(parent, attr, None)
    if not isinstance(child, linear_types):
        return 0
    setattr(
        parent,
        attr,
        _Fp8BlockScaleWeightReadLinear.from_linear(child, output_all_reduce, in_place_max_rows),
    )
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
        up = moe.routed_expert_up_proj
        if isinstance(moe, KimiK3MoERuntime) and isinstance(up, _Fp8BlockScaleWeightReadLinear):
            up.set_output_rows(moe.latent_up_rows)

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


def _load_packed_mxfp4_expert(
    backend, base, expert_idx, local_slot_id, get_tensor, get_optional_tensor
) -> None:
    del get_optional_tensor  # this layout stores no optional per-expert tensor
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


def _load_nvfp4_expert(
    backend, base, expert_idx, local_slot_id, get_tensor, get_optional_tensor
) -> None:
    # Only an NVFP4_AWQ / NVFP4_ARC calibration writes a pre-quant scale; a
    # plain NVFP4 checkpoint has none and the pair stays None, which leaves
    # ``fc31_act_scale`` unset exactly as before. Dropping it for a checkpoint
    # that HAS one would be silently wrong -- the activation would be quantized
    # unsmoothed against an input scale calibrated on the smoothed one.
    pre_quant_scales = {
        f"{w}_pre_quant_scale": get_optional_tensor(f"{base}.{expert_idx}.{w}.pre_quant_scale")
        for w in ("w1", "w3")
    }
    backend.quant_method.load_streaming_nvfp4_expert(
        backend,
        global_expert_id=expert_idx,
        local_slot_id=local_slot_id,
        **{
            f"{w}_{kind}": get_tensor(f"{base}.{expert_idx}.{w}.{kind}")
            for w in ("w1", "w2", "w3")
            for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
        },
        **pre_quant_scales,
    )


class _K3ExpertCkptSpec(NamedTuple):
    """How one routed-expert quantization is spelled and loaded."""

    # Per-``w{1,2,3}`` checkpoint tensor suffixes this layout stores. Every
    # combination is required: a missing one fails the key plan.
    kinds: Tuple[str, ...]
    loader: Callable[..., None]
    # Set of filled slots the loader maintains, checked after the load.
    loaded_slots_attr: str
    # NVFP4 defers cat/pad/interleave and the alpha computation to
    # ``process_weights_after_loading``; the MXFP4 loaders write through.
    needs_layer_finalize: bool
    # Whole per-expert key suffixes the loader passes through WHEN PRESENT, for
    # a calibration artifact only some checkpoints of this layout carry. Spelled
    # per weight rather than per kind, because they are not stored for every
    # ``w``, and kept out of ``kinds`` because demanding them would reject the
    # checkpoints that have none.
    optional_suffixes: Tuple[str, ...] = ()


_K3_NVFP4_EXPERT_CKPT_SPEC = _K3ExpertCkptSpec(
    kinds=("weight", "weight_scale", "weight_scale_2", "input_scale"),
    loader=_load_nvfp4_expert,
    loaded_slots_attr="_streamed_expert_slots",
    needs_layer_finalize=True,
    optional_suffixes=("w1.pre_quant_scale", "w3.pre_quant_scale"),
)

_K3_EXPERT_CKPT_SPECS = {
    QuantAlgo.W4A8_MXFP4_MXFP8: _K3ExpertCkptSpec(
        kinds=("weight_packed", "weight_scale"),
        loader=_load_packed_mxfp4_expert,
        loaded_slots_attr="_packed_mxfp4_loaded_slots",
        needs_layer_finalize=False,
    ),
    QuantAlgo.NVFP4: _K3_NVFP4_EXPERT_CKPT_SPEC,
    # The two NVFP4 calibration recipes store plain NVFP4 expert weights plus
    # the per-channel pre-quant scale above, so they share one layout; the MoE
    # quantization method folds them onto the NVFP4 pipeline as well.
    QuantAlgo.NVFP4_AWQ: _K3_NVFP4_EXPERT_CKPT_SPEC,
    QuantAlgo.NVFP4_ARC: _K3_NVFP4_EXPERT_CKPT_SPEC,
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


def _is_helix_a2a(model_config: ModelConfig) -> bool:
    """Helix + attention DP: CP ranks repurposed to TP, MoE on token slices."""
    mapping = model_config.mapping
    return mapping.enable_attention_dp and mapping.helix_repurposed_cp_size > 1


class KimiK3MoERuntime(nn.Module):
    """Kimi K3 latent MoE block backed by ConfigurableMoE."""

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        *,
        follows_kda_attention: bool = False,
    ):
        """Build the routed experts and the shared expert for one MoE layer.

        ``cfg`` is the raw ``PretrainedConfig`` rather than anything derived:
        the SiTU soft-caps and the routed-expert geometry are Kimi K3 fields
        that ``ModelConfig`` does not carry.

        ``aux_stream_dict`` is shared across every layer of the model, so the
        streams reached through it are borrowed and must not be synchronized
        or reassigned here.

        ``follows_kda_attention``: the layer's attention is KDA (selects the router-logits
        kernel, see below).
        """
        super().__init__()
        self.layer_idx = layer_idx
        # SMs the shared expert's GEMMs keep to beside the routed experts (0: all).
        self._shared_sms = 0
        # ``(first row, rows)`` of latent_up's output this rank computes in the column-parallel
        # tail (None: no column-parallel tail).
        self.latent_up_rows: Optional[Tuple[int, int]] = None
        # The routed latent all-reduce on its own MNNVL workspace (``_latent_all_reduce_workspace``)
        # for the sizes it runs one-shot (``_latent_all_reduce``); None: the routed experts'.
        self._own_latent_all_reduce: Optional[AllReduce] = None
        # The sharded decode front (``kimi_k3_moe_front``): the routed all-reduce group's
        # workspace and this layer's operands on this rank, built after loading where it applies.
        self._moe_front_ws = None
        self._moe_front_plan = None
        # Lossy decode kernels at <= 8 rows (``_prepare_min_latency_tail``): latent_up's UE8M0
        # block exponents for the one-kernel MoE tail (``kimi_k3_moe_tail``), the shared expert's
        # weights in the persistent kernel's order, its workspace and TP size
        # (``kimi_k3_shared_expert``); None where they do not apply.
        self._min_latency = model_config.enable_min_latency
        self._moe_tail_wexp: Optional[torch.Tensor] = None
        self._shared_frags: Optional[torch.Tensor] = None
        self._shared_ws = None
        self._shared_world = 0
        # Row-independent conditions of the column-parallel tail and of latent_up's MXFP8 input;
        # set in cache_derived_state.
        self._column_parallel_tail_supported = False
        self._latent_up_mxfp8_supported = False
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
        # at DEP batch sizes. helix_a2a ranks share requests and keep bf16.
        # KIMI_K3_ROUTER_BF16=1/0 forces either path.
        _router_bf16_env = os.environ.get("KIMI_K3_ROUTER_BF16")
        _router_bf16 = (
            _router_bf16_env == "1"
            if _router_bf16_env is not None
            else model_config.mapping.tp_group_shares_requests
        )
        # The router-logits kernel serves the MoE blocks after KDA attention: there the earlier
        # routing shortens the MoE front, while after MLA attention it lands in the burst of
        # latent_down and the shared expert's first GEMM and the front gets longer.
        self.gate = KimiK3MoEGate(
            cfg,
            logits_gemm_dtype=torch.bfloat16 if _router_bf16 else None,
            logits_kernel=follows_kda_attention,
        )

        routed_moe_model_config = self._routed_moe_model_config(model_config)
        # The router-logits kernel pays where the K3 latent stream lets its result be
        # consumed early (CuteDSL / CuteDSL-FC12); elsewhere (the default TRTLLM backend,
        # no-graph forwards) the shorter launch only shortens the route tail, so those
        # keep ``trtllm::dsv3_router_gemm_op``.
        if self.gate.logits_kernel:
            self.gate.logits_kernel = routed_moe_model_config.moe_backend in (
                "CUTEDSL",
                "CUTEDSL_FC12",
            )
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
            # Lets a FUSED_COMM backend return its globally combined routed
            # output; ``_reduce_routed_output`` then skips that reduction.
            accepts_global_routed_output=True,
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
            not in ("MEGAMOE_DEEPGEMM", "MEGAMOE_CUTEDSL", "CUTEDSL", "CUTEDSL_FC12"),
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
        # Fused-comm MegaMoE runs its own cross-rank exchange inside the
        # kernel. Order the shared-expert all-reduce after the routed MoE
        # instead of running the two collectives concurrently (startup
        # deadlocks). CuteDSL is the validated scope, not a correctness
        # dependency: DeepGEMM rejects TP without attention DP.
        self._defer_shared_allreduce = (
            use_shared_tp
            and routed_moe_model_config.moe_backend == "MEGAMOE_CUTEDSL"
            and self.routed_experts.backend.scheduler_kind == MoESchedulerKind.FUSED_COMM
        )
        self._reduce_routed_output = (
            use_shared_tp and not self.routed_experts.returns_global_routed_output
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
        self._shared_all_reduce = use_shared_tp
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
        self.moe_routed_event = torch.cuda.Event() if self._defer_shared_allreduce else None
        # CuteDSL already quantizes the MoE input on the MoeOutputMemset stream;
        # the latent down-projection runs there too, beside the routing chain.
        self.latent_down_stream = (
            aux_stream_dict.get(AuxStreamType.MoeOutputMemset)
            if routed_moe_model_config.moe_backend in ("CUTEDSL", "CUTEDSL_FC12")
            else None
        )
        has_latent_stream = self.latent_down_stream is not None
        self.moe_latent_fork_event = torch.cuda.Event() if has_latent_stream else None
        self.moe_latent_ready_event = torch.cuda.Event() if has_latent_stream else None
        # Shared GEMMs and the latent chain done (``_MoEFrontPlan.shared_first``).
        self.moe_shared_gemm_event = torch.cuda.Event() if has_latent_stream else None
        # The shared GEMMs can lead the routed experts (``_front_plan``) only as their own
        # step on the shared stream, with the all-reduce following them there: not deferred
        # behind a fused-comm MoE, and not fused into the down projection's GEMM.
        shared_down = self.shared_experts.down_proj
        self._shared_gemms_can_lead = (
            has_latent_stream
            and self.shared_expert_stream is not None
            and not self._defer_shared_allreduce
            and not (isinstance(shared_down, TrtllmLinear) and shared_down.use_fused_gemm_allreduce)
        )
        # The routed grid can run SM-short only where the backend sizes a persistent
        # grid it can shrink (``MoEImplBase.sm_reserve_ceiling``; the fused FC12 leaf
        # offers 16). This is the construction-time capability; the per-launch reserve
        # and its carveout pairing are decided per forward (_shared_sms_now), because
        # multi-stream is a per-forward state and the FP8 weight-read swap can move the
        # shared projections onto a CuTe DSL GEMM after construction.
        self._shared_sms = (
            min(
                _SHARED_EXPERT_SMS_BESIDE_ROUTED_GRID,
                self.routed_experts.backend.sm_reserve_ceiling(),
            )
            if not self._defer_shared_allreduce
            else 0
        )
        # Kimi K3's routed experts may be placed die-major on a two-domain device (the backend
        # decides at load).
        if isinstance(self.routed_experts.backend, MoEImplBase):
            self.routed_experts.backend.allow_die_major_weights()
        self.routed_expert_down_proj = nn.Linear(
            cfg.hidden_size, self.moe_hidden_size, bias=False, dtype=dtype
        )
        self.routed_expert_up_proj = nn.Linear(
            self.moe_hidden_size, cfg.hidden_size, bias=False, dtype=dtype
        )
        # The column-parallel tail (``_forward_parts``) splits latent_up's output into whole
        # 128-row blocks per rank of the routed all-reduce's group, as evenly as possible.
        if self._reduce_routed_output and not self._defer_shared_allreduce:
            mapping = self.routed_experts.all_reduce.mapping
            blocks, world = cfg.hidden_size // 128, mapping.tp_size
            if cfg.hidden_size % 128 == 0 and blocks >= world:
                first = mapping.tp_rank * (blocks // world) + min(mapping.tp_rank, blocks % world)
                count = blocks // world + (mapping.tp_rank < blocks % world)
                self.latent_up_rows = (first * 128, count * 128)
        # Built on every rank of the routed all-reduce group at the same point (the workspace is
        # collective); MNNVL only, where the routed all-reduce runs MNNVL.
        routed_ar = self.routed_experts.all_reduce
        if (
            self._reduce_routed_output
            and not self._defer_shared_allreduce
            and routed_ar.mnnvl_runs_fusion(AllReduceFusionOp.RMS_NORM, self.moe_hidden_size)
        ):
            own = AllReduce(
                routed_ar.mapping,
                strategy=AllReduceStrategy.MNNVL,
                dtype=dtype,
                mnnvl_own_workspace=_latent_all_reduce_workspace(self.moe_hidden_size),
            )
            if own.mnnvl_runs_fusion(AllReduceFusionOp.RMS_NORM, self.moe_hidden_size):
                self._own_latent_all_reduce = own
        # Stock fused RMSNorm (flashinfer kernel; the no-flashinfer
        # fallback is the same fp32-variance eager math as KimiK3RMSNorm).
        self.routed_expert_norm = RMSNorm(
            hidden_size=self.moe_hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )

    def _latent_all_reduce(self, num_tokens: int) -> AllReduce:
        """The all-reduce of a ``num_tokens``-row routed latent: on its own workspace where that
        runs one-shot (the size rule of ``_latent_all_reduce_workspace``), else the routed
        experts' all-reduce (on the group's shared workspace)."""
        own = self._own_latent_all_reduce
        if own is not None and own.mnnvl_runs_one_shot(num_tokens, self.moe_hidden_size):
            return own
        return self.routed_experts.all_reduce

    def _supports_column_parallel_tail(self) -> bool:
        """The row-independent part of :meth:`runs_column_parallel_tail`."""
        if (
            self.latent_up_rows is None
            or not isinstance(self.routed_expert_up_proj, _Fp8BlockScaleWeightReadLinear)
            or self.moe_hidden_size % 256
        ):
            return False
        routed_ar = self.routed_experts.all_reduce
        down = self.shared_experts.down_proj
        shared_ar = down.all_reduce if isinstance(down, TrtllmLinear) else None
        return (
            shared_ar is not None
            and shared_ar.mnnvl_allreduce is not None
            and shared_ar.symm_mem_allreduce is None
            and shared_ar.mapping.tp_group == routed_ar.mapping.tp_group
            and routed_ar.mnnvl_runs_fusion(
                AllReduceFusionOp.RMS_NORM_QUANT_MXFP8, self.moe_hidden_size
            )
        )

    def runs_column_parallel_tail(self, num_tokens: int) -> bool:
        """Whether a forward of ``num_tokens`` rows can run the column-parallel tail: latent_up
        projects onto this rank's rows (``latent_up_rows``) with the whole projection's low-M
        kernel, the routed latent all-reduce runs MNNVL one-shot (the C++ size rule, which also
        bounds the tail's slot traffic) and the shared expert reduces with the MNNVL plain sum
        on the same group (``_column_parallel_tail_supported``, set in cache_derived_state)."""
        return (
            self._column_parallel_tail_supported
            and not torch.compiler.is_compiling()
            and self.routed_expert_up_proj.serves_output_rows(num_tokens)
            and self.routed_experts.all_reduce.mnnvl_runs_one_shot(num_tokens, self.moe_hidden_size)
        )

    def _column_parallel_tail(
        self,
        routed_latent: torch.Tensor,
        shared_partial: Optional[torch.Tensor],
        tail_targets=None,
        shared_pushed: bool = False,
    ) -> Tuple[Union[torch.Tensor, MnnvlPendingResult], Optional[torch.Tensor]]:
        """The MoE tail split by latent_up's output rows (``runs_column_parallel_tail``).

        The latent all-reduce + RMSNorm hands the MXFP8 latent over (``RMS_NORM_QUANT_MXFP8``),
        this rank projects it onto its 128-row blocks of latent_up (``forward_rows_prequantized``,
        the whole projection's bits), and an all-gather call on the shared expert's all-reduce
        workspace carries the shared partial in its slots and these columns in its all-gather
        rows. Returns ``(call, None)``: the pending call stands for the replicated tail's
        ``bf16(latent_up + shared all-reduce)``, which the next decode mix finishes, while each
        rank reads 1 / W of latent_up. Every rank runs the same calls in this order on the main
        stream: the pushes read the rotation state the latent all-reduce left. ``shared_pushed``
        (with ``tail_targets``): the persistent shared expert stored its partial in the all-gather
        call's slots (``_shared_push_target``); ``shared_partial`` is then None.
        """
        up = self.routed_expert_up_proj
        shared_ar = self.shared_experts.down_proj.all_reduce
        if tail_targets is not None:
            # enable_min_latency: the routed experts pushed their latent rows; one kernel finishes
            # the latent all-reduce and stores this rank's columns into the all-gather call.
            tail = self._moe_tail(
                routed_latent, shared_partial, *tail_targets, shared_pushed=shared_pushed
            )
            return tail, None
        latent, latent_scale = self._latent_all_reduce(routed_latent.shape[0])(
            routed_latent,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RMS_NORM_QUANT_MXFP8,
                norm_weight=self.routed_expert_norm.weight,
                eps=float(self.routed_expert_norm.variance_epsilon),
            ),
        )
        num_tokens = latent.shape[0]
        shared_partial = shared_partial.reshape(num_tokens, self.hidden_size).contiguous()
        target = shared_ar.push_target(num_tokens, self.hidden_size, all_gather=True)
        if target is None:
            return up.forward_prequantized(latent, latent_scale), shared_ar(shared_partial)
        columns = up.forward_rows_prequantized(latent, latent_scale)
        shared_ar.push_input(target, shared_partial)
        shared_ar.push_input(target, columns, columns=self.latent_up_rows[0])
        return shared_ar.finish_push(target), None

    def _moe_tail(
        self,
        routed_latent: torch.Tensor,
        shared_partial: Optional[torch.Tensor],
        ag,
        latent,
        shared_pushed: bool = False,
    ):
        """The one-kernel MoE tail (``kimi_k3_moe_tail``) on the latent all-reduce the routed
        experts pushed into, then the shared partial's push unless ``shared_pushed`` (stored by
        the persistent shared expert on its stream); returns the pending all-gather call."""
        from ..kernels.blackwell import kimi_k3_moe_tail

        num_tokens = routed_latent.shape[0]
        shared_ar = self.shared_experts.down_proj.all_reduce
        lat_view, ag_view = latent.view, ag.view
        kimi_k3_moe_tail.run_moe_tail(
            m=num_tokens,
            world=lat_view.world,
            lat_uc=lat_view.uc_local,
            lat_flags=lat_view.flags,
            weight=self.routed_expert_up_proj.weight,
            wexp=self._moe_tail_wexp,
            gamma=self.routed_expert_norm.weight,
            eps=float(self.routed_expert_norm.variance_epsilon),
            ag_base=ag_view.mc_base,
            ag_flags=ag_view.flags,
            c0=self.latent_up_rows[0],
            rows=self.latent_up_rows[1],
            ag_follows=lat_view.flags.data_ptr() == ag_view.flags.data_ptr(),
        )
        if not shared_pushed:
            shared_ar.push_input(
                ag, shared_partial.reshape(num_tokens, self.hidden_size).contiguous()
            )
        return shared_ar.finish_push(ag)

    def _latent_up_takes_mxfp8(self, routed_latent: torch.Tensor) -> bool:
        """Whether the routed latent all-reduce hands latent_up the MXFP8 form of the normalized
        latent: where latent_up's FP8 low-M GEMM serves the rows and the MNNVL kernels run the
        all-reduce (``RMS_NORM_QUANT_MXFP8``, the GEMM's own quantization: the same bits;
        ``_latent_up_mxfp8_supported``, set in cache_derived_state)."""
        return (
            self._latent_up_mxfp8_supported
            and routed_latent.dim() == 2
            and routed_latent.dtype == torch.bfloat16
            and self.routed_expert_up_proj.takes_prequantized_input(routed_latent.shape[0])
        )

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def cache_derived_state(self) -> None:
        self._column_parallel_tail_supported = self._supports_column_parallel_tail()
        self._latent_up_mxfp8_supported = (
            self._reduce_routed_output
            and isinstance(self.routed_expert_up_proj, _Fp8BlockScaleWeightReadLinear)
            and self.routed_expert_norm.weight.dtype == torch.bfloat16
            and self.routed_experts.all_reduce.mnnvl_runs_fusion(
                AllReduceFusionOp.RMS_NORM_QUANT_MXFP8, self.moe_hidden_size
            )
        )
        self._prepare_moe_front()
        self._prepare_min_latency_tail()

    def _prepare_min_latency_tail(self) -> None:
        """``enable_min_latency`` (lossy): on the column-parallel tail of at most 8 rows the MoE tail
        is one kernel fed by the routed experts' pushed latent rows (``kimi_k3_moe_tail``) and the
        shared expert one persistent kernel beside the routed experts
        (``kimi_k3_shared_expert``). Builds their operands on this rank."""
        self._moe_tail_wexp = self._shared_frags = self._shared_ws = None
        if not self._min_latency or not self._column_parallel_tail_supported:
            return
        from ..kernels.blackwell import kimi_k3_moe_tail, kimi_k3_shared_expert

        up = self.routed_expert_up_proj
        world = self.routed_experts.all_reduce.mapping.tp_size
        if (
            isinstance(self.routed_experts.backend, MoEImplBase)
            and tuple(up.weight.shape) == (kimi_k3_moe_tail.HIDDEN, kimi_k3_moe_tail.LATENT)
            and kimi_k3_moe_tail.can_run(1, world, self.latent_up_rows[1])
        ):
            up_scale = up.prepare_low_m_weight_scale()
            if up_scale is not None:
                wexp = kimi_k3_moe_tail.block_exponents(
                    up_scale, kimi_k3_moe_tail.HIDDEN, kimi_k3_moe_tail.LATENT
                )
                if 1 <= int(wexp.min().item()) and int(wexp.max().item()) <= 254:
                    self._moe_tail_wexp = wexp
        mlp = self.shared_experts
        if not isinstance(mlp.activation, SituAndMul):
            return
        gate_up = _bf16_weight(mlp.gate_up_proj)
        down = _bf16_weight(mlp.down_proj)
        sms = self._shared_sms or _SHARED_EXPERT_SMS_BESIDE_ROUTED_GRID
        if gate_up is None or down is None or gate_up.shape[1] != kimi_k3_shared_expert.HIDDEN:
            return
        world, rem = divmod(2 * kimi_k3_shared_expert.INTER, gate_up.shape[0])
        if rem or not kimi_k3_shared_expert.can_run(1, world, sms):
            return
        with torch.no_grad():
            self._shared_frags = kimi_k3_shared_expert.persist_fragments(gate_up, down, sms)
        self._shared_ws = kimi_k3_shared_expert.PersistWorkspace(world, gate_up.device)
        self._shared_world = world

    def _min_latency_tail_applies(self, num_tokens: int) -> bool:
        """Whether this column-parallel tail forward runs the one-kernel MoE tail."""
        from ..kernels.blackwell import kimi_k3_moe_tail

        return self._moe_tail_wexp is not None and kimi_k3_moe_tail.can_run(
            num_tokens, self.routed_experts.all_reduce.mapping.tp_size, self.latent_up_rows[1]
        )

    def _min_latency_tail_targets(self, num_tokens: int):
        """``(all-gather call, latent all-reduce)`` push targets of a one-kernel MoE tail, where
        the routed experts can push their output into the latent all-reduce, or None."""
        from ..kernels.blackwell import kimi_k3_moe_tail

        if not self._min_latency_tail_applies(num_tokens):
            return None
        shared_ar = self.shared_experts.down_proj.all_reduce
        # The all-gather call first: it may grow a workspace both calls share.
        ag = shared_ar.push_target(num_tokens, self.hidden_size, all_gather=True)
        latent = self._latent_all_reduce(num_tokens).push_target(
            num_tokens, self.moe_hidden_size, consumer_finishes=True
        )
        if (
            ag is None
            or latent is None
            or not latent.one_shot
            or latent.view.world != ag.view.world
            or not kimi_k3_moe_tail.can_run(num_tokens, latent.view.world, self.latent_up_rows[1])
            or not self.routed_experts.pushes_output(num_tokens)
        ):
            return None
        return ag, latent

    def _min_latency_shared_applies(self, num_tokens: int) -> bool:
        """Whether the persistent shared expert (``kimi_k3_shared_expert``) serves the rows."""
        from ..kernels.blackwell import kimi_k3_shared_expert

        return self._shared_frags is not None and 1 <= num_tokens <= kimi_k3_shared_expert.MAX_M

    def _shared_push_target(self, num_tokens: int, ag, latent):
        """``ag`` where the persistent shared expert stores its partial into this all-gather
        call's slots itself (output push) on the shared expert's stream, beside the routed
        experts and the MoE tail; else None.

        The kernel reads the call's rotation state, so the call must already be the current one
        on its workspace when it starts: not where it follows the latent all-reduce on one
        workspace (``ag_follows``), which makes it current only once the tail finishes that
        all-reduce. The next decode mix polls the slots, so the tail does not wait for them."""
        if not self._min_latency_shared_applies(num_tokens):
            return None
        if ag.view.flags.data_ptr() == latent.view.flags.data_ptr():
            return None
        if ag.view.world != self._shared_world:
            return None
        return ag

    def _min_latency_shared(
        self, identity: torch.Tensor, push: Optional[MnnvlPushTarget] = None
    ) -> Optional[torch.Tensor]:
        """The shared expert's partial from the persistent kernel where it applies, else None.
        With ``push`` (``_shared_push_target``) the kernel stores the partial into the call's
        slots instead and returns None."""
        from ..kernels.blackwell import kimi_k3_shared_expert

        num_tokens = identity.shape[0]
        sms = self._shared_sms or _SHARED_EXPERT_SMS_BESIDE_ROUTED_GRID
        if not self._min_latency_shared_applies(num_tokens):
            return None
        x = identity.reshape(num_tokens, self.hidden_size).contiguous()
        act = self.shared_experts.activation
        out, push_args = None, {}
        if push is not None:
            view = push.view
            push_args = dict(push_base=view.mc_base, push_flags=view.flags, rank=view.rank)
        else:
            out = torch.empty((num_tokens, self.hidden_size), dtype=x.dtype, device=x.device)
        return kimi_k3_shared_expert.run_shared_expert(
            x=x,
            frags=self._shared_frags,
            world=self._shared_world,
            sms=sms,
            ws=self._shared_ws,
            out=out,
            beta=float(act.beta),
            linear_beta=act.linear_beta,
            **push_args,
        )

    def _moe_front_unsupported(self) -> Optional[str]:
        """Why this layer cannot run the sharded decode front (``kimi_k3_moe_front``), or
        ``None``."""
        from ..moe.kernels.blackwell import kimi_k3_moe_front

        moe = self.routed_experts
        if not self._reduce_routed_output:
            return "the routed output is not all-reduced over the TP group"
        mnnvl = moe.all_reduce.mnnvl_allreduce
        if mnnvl is None or not kimi_k3_moe_front.supports_world(mnnvl.mapping.tp_size):
            return "the routed all-reduce is not MNNVL over a supported group"
        if not moe.routes_with_sort_tables(kimi_k3_moe_front.M_MAX):
            return "the routed experts are not routed by the moe_sort routing kernel"
        if moe.nvfp4_input_scale(1) is None:
            return "the routed experts do not take an NVFP4 input"
        if (
            self.hidden_size != kimi_k3_moe_front.HIDDEN
            or self.moe_hidden_size != kimi_k3_moe_front.LATENT
            or self.num_experts != kimi_k3_moe_front.NUM_EXPERTS
            or self.gate.weight.dtype != torch.bfloat16
        ):
            return "not the Kimi-K3 geometry with a BF16 router"
        down = self.routed_expert_down_proj
        if not isinstance(down, _Fp8BlockScaleWeightReadLinear) or down.is_fused_row_view:
            return "latent_down is not an FP8 weight-read projection"
        return None

    def _prepare_moe_front(self) -> None:
        """Build this layer's sharded decode front on this rank (collective over the routed
        all-reduce group on the first layer: the workspace), or keep the replicated front."""
        self._moe_front_plan = None
        reason = self._moe_front_unsupported()
        down = self.routed_expert_down_proj
        down_scale = None
        if reason is None:
            down_scale = down.prepare_low_m_weight_scale()
            if down_scale is None:
                reason = "the low-M GEMM does not serve latent_down"
        if reason is not None:
            logger.info_once(
                f"Kimi K3: MoE layers keep the replicated decode front: {reason}",
                key=f"k3_moe_front_off_{reason}",
            )
            return
        from ..moe.kernels.blackwell import kimi_k3_moe_front

        backend = self.routed_experts.backend
        self._moe_front_ws = kimi_k3_moe_front.moe_front_workspace(
            self.routed_experts.all_reduce.mnnvl_allreduce.mapping
        )
        ws = self._moe_front_ws
        with torch.no_grad():
            self._moe_front_plan = kimi_k3_moe_front.make_plan(
                down.weight,
                down_scale,
                self.gate.weight,
                self.gate.e_score_correction_bias,
                float(self.gate.routed_scaling_factor),
                ws.world,
                ws.rank,
                backend.slot_start,
                backend.expert_size_per_partition,
            )
        logger.info_once(
            f"Kimi K3: sharded MoE decode front over {ws.world} ranks", key="k3_moe_front_on"
        )

    def _moe_front_applies(self, hidden_states: torch.Tensor) -> bool:
        """Whether this forward runs the sharded decode front: a layer it was prepared for,
        ``1 .. M_MAX`` rows the routed experts take as an NVFP4 input, and kernels already
        compiled when a CUDA graph is being captured."""
        plan = self._moe_front_plan
        if plan is None or torch.compiler.is_compiling():
            return False
        from ..moe.kernels.blackwell import kimi_k3_moe_front

        num_tokens = hidden_states.shape[0]
        if not 1 <= num_tokens <= kimi_k3_moe_front.M_MAX:
            return False
        scale = self.routed_experts.nvfp4_input_scale(num_tokens)
        if scale is None or not kimi_k3_moe_front.can_run(hidden_states, plan, scale):
            return False
        return not torch.cuda.is_current_stream_capturing() or kimi_k3_moe_front.is_compiled(
            num_tokens, plan, self._moe_front_publish_word(num_tokens) is not None
        )

    def _moe_front_publish_word(self, num_tokens: int) -> Optional[torch.Tensor]:
        """The routed experts' word for the front to publish the top-k into, where they build
        their tables from it (``MoEImplBase.published_routing_word``)."""
        backend = self.routed_experts.backend
        if not isinstance(backend, MoEImplBase):
            return None
        return backend.published_routing_word(num_tokens)

    def moe_front_releases_early(self, like: torch.Tensor) -> bool:
        """Whether the mix that produces this layer's MoE input may release its dependent at
        kernel entry: the next forward on rows shaped like ``like`` (contiguous BF16 ``[M,
        hidden]``) runs the sharded decode front, whose head, the next launch on the stream, reads
        only weights before its grid wait."""
        return like.dim() == 2 and like.is_contiguous() and self._moe_front_applies(like)

    def _moe_front(
        self, hidden_states: torch.Tensor, after_head: Optional[Callable[[], None]] = None
    ) -> Tuple[Fp4QuantizedTensor, SortedRouting]:
        """The sharded decode front on the current stream: the routed experts' NVFP4 input and
        the routing with this rank's ``moe_sort`` tables; ``after_head`` runs between its two
        launches."""
        from ..moe.kernels.blackwell import kimi_k3_moe_front, kimi_k3_routing

        num_tokens = hidden_states.shape[0]
        publish_word = self._moe_front_publish_word(num_tokens)
        ids, weights, sort_outputs, codes, scales = kimi_k3_moe_front.run_moe_front(
            hidden_states,
            self._moe_front_plan,
            self._moe_front_ws,
            self.routed_experts.nvfp4_input_scale(num_tokens),
            publish_word,
            after_head,
        )
        if publish_word is not None:
            tables = PublishedSortTables()
        else:
            tables = MoESortTables(kimi_k3_routing.TILE_SIZE, tuple(sort_outputs))
        routing = SortedRouting(ids, weights, tables)
        return Fp4QuantizedTensor(codes, scales, is_sf_swizzled=False), routing

    @staticmethod
    def _routed_projection(hidden_states: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        if _K3_DISABLE_MIN_LATENCY_LATENT_PROJ or not isinstance(projection, nn.Linear):
            return projection(hidden_states)
        return torch.ops.trtllm.dsv3_fused_a_gemm_op(
            hidden_states, projection.weight.t(), None, None
        )

    def _can_fuse_router_down(self, hidden_states: torch.Tensor) -> bool:
        """Whether one fused GEMM can produce the router logits and the latent
        down-projection (trtllm::dsv3_router_latent_gemm_op).

        Scoped to the validated configuration (TP MEGAMOE_CUTEDSL with fused
        communication on SM107); the fusion itself does not depend on it.
        """
        if (
            torch.compiler.is_compiling()
            or _K3_DISABLE_MIN_LATENCY_LATENT_PROJ
            or not hidden_states.is_cuda
            or self.routed_experts.use_dp
            or self.routed_experts.parallel_size <= 1
            or self.routed_experts.moe_backend != "MEGAMOE_CUTEDSL"
            or self.routed_experts.backend.scheduler_kind != MoESchedulerKind.FUSED_COMM
            or get_sm_version() != 107
        ):
            return False
        down = self.routed_expert_down_proj
        if not isinstance(down, nn.Linear) or down.bias is not None:
            return False
        if (
            hidden_states.ndim != 2
            or not 1 <= hidden_states.shape[0] <= 16
            or hidden_states.shape[1] != 7168
            or self.gate.weight.shape != (896, 7168)
            or down.weight.shape != (3584, 7168)
        ):
            return False
        # Inspect the loaded weights: FP8 conversion can replace the down module.
        return all(
            tensor.dtype == torch.bfloat16
            and tensor.is_cuda
            and tensor.device == hidden_states.device
            and tensor.stride() == (7168, 1)
            and tensor.data_ptr() % 16 == 0
            for tensor in (hidden_states, self.gate.weight, down.weight)
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
        # joined once its act-fusion kernel grew the SiTU epilogue, and
        # CUTEDSL_FC12 (Rubin/SM107) once its fused FC1+FC2 kernel did.
        supported_backends = {
            "CUTLASS",
            "TRTLLM",
            "CUTEDSL",
            "CUTEDSL_FC12",
            "MEGAMOE_DEEPGEMM",
            "MEGAMOE_CUTEDSL",
        }
        if model_config.moe_backend not in supported_backends:
            raise ValueError(
                "Kimi K3 SiTU routed experts only support the CUTLASS, TRTLLM, "
                "CUTEDSL, CUTEDSL_FC12, MEGAMOE_DEEPGEMM, and MEGAMOE_CUTEDSL backends; "
                f"got {model_config.moe_backend!r}."
            )
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        mapping = model_config.mapping
        if getattr(mapping, "_dwdp_size", 0) > 1:
            # DWDP does not hook the load path at all: setup_dwdp() scans the
            # model *after* every weight is loaded, reads w3_w1_weight /
            # w2_weight and their block scales straight off the backend
            # (dwdp/setup.py::collect_moe_params), copies them into fabric
            # memory, frees the originals, and rebinds param.data to a
            # composite VA view.  K3's packed-checkpoint streaming produces
            # exactly those tensors, so the two are compatible -- but only on
            # the NVFP4 CuteDSL leaves that declare supports_dwdp (CUTEDSL,
            # CUTEDSL_FC12).
            #
            # MEGAMOE_CUTEDSL is excluded deliberately rather than by
            # oversight: its process_weights_after_loading packs into the mega
            # buffers and shrinks the raw source params back to placeholders
            # (see maybe_finalize_layer), which has never been checked against
            # DWDP's copy-then-free of those same originals.
            if model_config.moe_backend not in ("CUTEDSL", "CUTEDSL_FC12"):
                raise NotImplementedError(
                    "Kimi K3 with DWDP requires moe_config.backend CUTEDSL or "
                    f"CUTEDSL_FC12; got {model_config.moe_backend!r}."
                )

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
            # helix_a2a ranks route ceil(max_num_tokens / tp) rows each.
            rank_max_num_tokens = routed_model_config.max_num_tokens
            if _is_helix_a2a(model_config):
                rank_max_num_tokens = -(-rank_max_num_tokens // routed_mapping.tp_size)
            default_moe_max_num_tokens = rank_max_num_tokens * routed_mapping.dp_size
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

    def _shared_sms_now(self) -> int:
        """SMs the shared expert's GEMMs keep to beside the routed grid this forward,
        the per-launch reserve's pairing (the torch SM carveout).

        Zero unless this forward runs the shared expert on a side stream --
        ``do_multi_stream``, or an AutoTuner pass, which must profile the shrunk
        FC12 grid the captured graphs go on to launch -- or when the shared
        projections no longer go through torch's cuBLAS (``F.linear``): the FP8
        weight-read swap can have replaced them with a CuTe DSL GEMM after
        construction, and those kernels ignore the carveout."""
        from ..autotuner import AutoTuner

        if not (do_multi_stream() or AutoTuner.get().is_tuning_mode):
            return 0
        down = self.shared_experts.down_proj
        if (
            isinstance(down, _Fp8BlockScaleWeightReadLinear)
            or getattr(down, "use_cute_dsl_bf16_gemm", False)
            or getattr(down, "use_custom_cublas_mm", False)
        ):
            return 0
        return self._shared_sms

    def _front_plan(self, num_tokens: int) -> _MoEFrontPlan:
        """Issue order of a decode MoE front with the latent stream (multi-stream, i.e. CUDA
        graphs), decided in this one place from what the routed experts and latent_down declare.

        - The router GEMM goes before latent_down up to ``_ROUTER_FIRST_MAX_TOKENS`` tokens.
          The graph edges are the same either way; the creation order decides which of the two
          independent GEMMs launches first when the MoE input's producer releases them. At
          decode sizes each fills most of the SMs, so the second one runs on what the first
          leaves, and the routing gates the routed experts. (Which router kernel computes the
          logits is the gate's choice, see ``KimiK3MoEGate``.)
        - When the routed experts take an NVFP4 input as is (``nvfp4_input_scale``), latent_down
          emits it from its GEMM epilogue where the projection serves this many rows
          (``_Fp8BlockScaleWeightReadLinear.emits_nvfp4`` -- the weight-read swap itself is the
          existing ``KIMI_K3_FP8_WEIGHT_READ_MOE_MLP`` opt-in, so a default deployment never
          takes this branch), and only when the sorting kernel routes this batch
          (``routes_with_sort_tables``): the epilogue's route tactics were tuned beside it.
        - When the routing kernel also writes the experts' ``moe_sort`` tables
          (``routes_with_sort_tables``), the routing chain is short enough that the persistent
          routed grid, which holds every SM, launches while the shared expert's GEMMs or
          all-reduce are pending; those then wait for the whole grid. From
          ``_SHARED_GEMMS_LEAD_MIN_ROUTED_ROWS`` expected routed rows per rank the routed
          experts therefore wait for the shared GEMMs, on an NVFP4 input the latent chain
          produced, and the shared all-reduce follows its GEMMs on the shared stream. Below
          that the wait costs more than it saves. A routed grid that leaves SMs free
          (``_shared_sms``) instead runs beside the shared expert, never after it.
        Same kernels on the same inputs in every order.
        """
        moe = self.routed_experts
        nvfp4_scale = moe.nvfp4_input_scale(num_tokens)
        down = self.routed_expert_down_proj
        return _MoEFrontPlan(
            router_first=num_tokens <= _ROUTER_FIRST_MAX_TOKENS,
            shared_first=(
                self._shared_gemms_can_lead
                and not self._shared_sms
                and nvfp4_scale is not None
                and num_tokens * self.top_k * self.experts_per_rank // self.num_experts
                >= _SHARED_GEMMS_LEAD_MIN_ROUTED_ROWS
                and moe.routes_with_sort_tables(num_tokens)
            ),
            nvfp4_scale=nvfp4_scale,
            latent_down_nvfp4=(
                nvfp4_scale is not None
                and isinstance(down, _Fp8BlockScaleWeightReadLinear)
                and down.emits_nvfp4(num_tokens)
                # The fused epilogue's route tactics were tuned with the routing kernel
                # beside them; without it (e.g. a locality-domain policy) it is a loss.
                and moe.routes_with_sort_tables(num_tokens)
            ),
        )

    def _latent_chain(
        self, hidden_states: torch.Tensor, plan: _MoEFrontPlan
    ) -> torch.Tensor | Fp4QuantizedTensor:
        """latent_down and, as ``plan`` says, the routed experts' NVFP4 quantization."""
        if plan.latent_down_nvfp4:
            codes, scales = self.routed_expert_down_proj.forward_nvfp4(
                hidden_states, plan.nvfp4_scale
            )
            return Fp4QuantizedTensor(codes, scales, is_sf_swizzled=False)
        latent = self._routed_projection(hidden_states, self.routed_expert_down_proj)
        if plan.shared_first:
            # Quantized here, the routed input waits for neither the routing nor the shared
            # expert.
            return self.routed_experts.quantize_nvfp4_input(latent)
        return latent

    def _issue_shared_gemms(self, identity: torch.Tensor) -> torch.Tensor:
        """The shared expert's GEMMs, without their all-reduce, on the shared stream after the
        router GEMM; ``moe_shared_gemm_event`` marks them and the latent chain done. Returns
        the (partial) shared output."""
        self.moe_main_event.record()
        with torch.cuda.stream(self.shared_expert_stream):
            self.moe_main_event.wait()
            partial = self.shared_experts(
                identity, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
            )
            self.moe_latent_ready_event.wait()
            self.moe_shared_gemm_event.record()
        return partial

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]] = None,
        return_parts: bool = False,
        column_parallel_tail: bool = False,
    ) -> Union[
        torch.Tensor, Tuple[Union[torch.Tensor, MnnvlPendingResult], Optional[torch.Tensor]]
    ]:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16.

        Returns the BF16 sum of the routed and the shared experts' outputs, or with
        ``return_parts`` the two outputs ``(routed, shared)`` (the decoder layer leaves their sum
        to the residual update it may defer). ``column_parallel_tail`` (with ``return_parts``):
        see :meth:`_forward_parts`.
        """
        if column_parallel_tail and not return_parts:
            raise ValueError("column_parallel_tail needs return_parts")
        routed_out, shared_out = self._forward_parts(
            hidden_states, all_rank_num_tokens, column_parallel_tail
        )
        if return_parts:
            return routed_out, shared_out
        return routed_out + shared_out

    def _forward_parts(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]] = None,
        column_parallel_tail: bool = False,
    ) -> Tuple[Union[torch.Tensor, MnnvlPendingResult], Optional[torch.Tensor]]:
        """The routed and the shared experts' outputs.

        ``column_parallel_tail`` (the caller's generation-only forwards whose MoE residual update
        the next decode mix folds in): where :meth:`runs_column_parallel_tail` holds, the
        routed output is instead the pending all-gather call that stands for that sum and the
        shared output is None (see :meth:`_column_parallel_tail`).
        """
        cp_tail = column_parallel_tail and self.runs_column_parallel_tail(hidden_states.shape[0])
        # enable_min_latency: the one-kernel tail (the routed experts then push their output).
        tail_targets = None
        if cp_tail and self._min_latency:
            tail_targets = self._min_latency_tail_targets(hidden_states.shape[0])
        identity = hidden_states
        projected_input = None
        early_routed_in = None
        shared_partial = None
        sharded_front = self._moe_front_applies(hidden_states)
        # The shared expert forks after the front's head (see _MOE_FRONT_FORK_AT_HEAD_MAX_TOKENS).
        fork_at_head = (
            sharded_front
            and hidden_states.shape[0] <= _MOE_FRONT_FORK_AT_HEAD_MAX_TOKENS
            and self.shared_expert_stream is not None
            and do_multi_stream()
            and not self._defer_shared_allreduce
        )
        forked_shared = []
        if sharded_front:
            # The routed branch starts with the sharded front, which routes and produces the
            # NVFP4 routed input; the logits only carry the shape.
            router_logits = hidden_states.new_empty(
                (hidden_states.shape[0], self.num_experts), dtype=torch.float32
            )
        elif self._can_fuse_router_down(hidden_states):
            router_logits, projected_input = torch.ops.trtllm.dsv3_router_latent_gemm_op(
                hidden_states, self.gate.weight, self.routed_expert_down_proj.weight
            )
        else:
            plan = None
            router_logits = None
            if (
                self.latent_down_stream is not None
                and do_multi_stream()
                and not torch.compiler.is_compiling()
            ):
                plan = self._front_plan(hidden_states.shape[0])
                # Forked ahead of the router GEMM: the latent chain needs only
                # hidden_states.
                self.moe_latent_fork_event.record()
                if plan.router_first:
                    router_logits = self.gate.compute_logits(hidden_states)
                with torch.cuda.stream(self.latent_down_stream):
                    self.moe_latent_fork_event.wait()
                    early_routed_in = self._latent_chain(hidden_states, plan)
                    self.moe_latent_ready_event.record()
                # No record_stream (under graph capture it holds every layer's
                # blocks until capture ends): both tensors are read on main only
                # after the join, and later work on this stream forks from main.
            if router_logits is None:
                router_logits = self.gate.compute_logits(hidden_states)
            if plan is not None and plan.shared_first:
                shared_partial = self._issue_shared_gemms(identity)
        moe_all_reduce = (
            self._latent_all_reduce(hidden_states.shape[0]) if self._reduce_routed_output else None
        )
        overlap_shared_allreduce = (
            self._defer_shared_allreduce
            and do_multi_stream()
            and self.shared_expert_stream is not None
            and not torch.compiler.is_compiling()
        )
        # enable_min_latency with the shared expert on its own stream: its partial goes into the
        # MoE tail's all-gather call there, and the main stream joins that stream after the tail.
        shared_push = None
        if (
            tail_targets is not None
            and shared_partial is None
            and self.shared_expert_stream is not None
            and do_multi_stream()
            and not torch.compiler.is_compiling()
        ):
            shared_push = self._shared_push_target(hidden_states.shape[0], *tail_targets)

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
            moe_kwargs = {}
            if sharded_front:
                routed_in, moe_kwargs["sorted_routing"] = self._moe_front(
                    hidden_states, _fork_shared if fork_at_head else None
                )
            elif projected_input is not None:
                routed_in = projected_input
            elif early_routed_in is not None:
                routed_in = early_routed_in
                # ConfigurableMoE joins the event before its first read of x.
                moe_kwargs = {
                    "input_ready_event": (
                        self.moe_latent_ready_event
                        if shared_partial is None
                        else self.moe_shared_gemm_event
                    )
                }
            else:
                routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
            if isinstance(routed_in, Fp4QuantizedTensor):
                moe_kwargs["output_dtype"] = hidden_states.dtype
            if tail_targets is not None:
                moe_kwargs["output_push"] = tail_targets[1].view
            # Only a launch with the shared expert beside it gives up SMs (an in-order
            # launch keeps the full grid); the AutoTuner pass requests the reserve the
            # captured graphs launch with, so tuning and capture share a cache key.
            # Re-asked per forward: eager forwards launch here, a captured graph bakes
            # the launch's reserve in.
            if not torch.compiler.is_compiling():
                self.routed_experts.backend.reserve_sms_for_concurrent_kernels(
                    self._shared_sms_now()
                )
            y = self.routed_experts(
                routed_in,
                router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
                **moe_kwargs,
            )
            if overlap_shared_allreduce:
                self.moe_routed_event.record()
            if self._reduce_routed_output:
                return y
            # Communication-backed paths return a complete routed result.
            y = self.routed_expert_norm(y)
            return self._routed_projection(y, self.routed_expert_up_proj)

        def _shared_output():
            if cp_tail and self._min_latency:
                # The persistent shared expert (lossy), beside the routed grid.
                if shared_push is not None:
                    # Its partial goes straight into the all-gather call's slots: no output.
                    self._min_latency_shared(identity, push=shared_push)
                    return None
                partial = self._min_latency_shared(identity)
                if partial is not None:
                    return partial
            # Beside the routed grid (multi-stream / the AutoTuner pass) the shared GEMMs
            # keep to the SMs it leaves -- one per-forward source of truth with the
            # reserve the routed launch asked for.
            with _cublas_sm_limit(self._shared_sms_now()):
                shared_out = self.shared_experts(
                    identity,
                    final_all_reduce_params=(
                        AllReduceParams(enable_allreduce=False)
                        if self._defer_shared_allreduce or cp_tail
                        else None
                    ),
                )
            if overlap_shared_allreduce:
                # Wait for all routed chunks, but overlap the norm/up tail.
                self.moe_routed_event.wait()
                shared_out = self.shared_experts.down_proj.all_reduce(shared_out)
            return shared_out

        # Shared experts depend only on the block input, so overlap their GEMMs
        # with the routed dispatch/expert/combine chain. Multi-stream engages
        # only under CUDA graphs; otherwise both branches run in order on the
        # default stream. The shared GatedMLP includes its output all-reduce on
        # the auxiliary stream (deferred past the routed MoE for fused-comm
        # MegaMoE). The final join includes the shared all-reduce, so it
        # precedes the routed all-reduce: concurrent collectives on different
        # streams can corrupt SYMM_MEM all-reduce state.
        def _fork_shared():
            self.moe_main_event.record()
            with torch.cuda.stream(self.shared_expert_stream):
                self.moe_main_event.wait()
                forked_shared.append(_shared_output())
                self.moe_shared_event.record()

        # The join of the shared expert's stream where it follows the latent tail.
        shared_join = None
        if fork_at_head:
            routed_out = _routed_output()
            shared_out = forked_shared[0]
            if shared_push is not None:
                # The main stream reads nothing the shared stream wrote: joined after the tail.
                shared_join = self.moe_shared_event
            else:
                self.moe_shared_event.wait()
                if not torch.cuda.is_current_stream_capturing():
                    shared_out.record_stream(torch.cuda.current_stream())
        elif shared_partial is not None:
            routed_out = _routed_output()
            with torch.cuda.stream(self.shared_expert_stream):
                shared_out = shared_partial
                if self._shared_all_reduce and not cp_tail:
                    shared_out = self.shared_experts.down_proj.all_reduce(shared_partial)
                self.moe_shared_event.record()
            self.moe_shared_event.wait()
        else:
            routed_out, shared_out, shared_join = maybe_execute_in_parallel_deferred_join(
                _routed_output,
                _shared_output,
                self.moe_main_event,
                self.moe_shared_event,
                self.shared_expert_stream,
                disable_on_compile=True,
            )
            if (
                shared_join is not None
                and shared_push is None
                and (
                    cp_tail
                    or moe_all_reduce is None
                    or moe_all_reduce is not self._own_latent_all_reduce
                )
            ):
                shared_join.wait()
                shared_join = None
            # Else the latent all-reduce, on its own workspace, and latent_up (or the MoE tail the
            # shared expert pushed beside) are issued before the main stream joins the shared
            # expert's stream.
        if self._defer_shared_allreduce and not overlap_shared_allreduce:
            shared_out = self.shared_experts.down_proj.all_reduce(shared_out)
        if cp_tail:
            tail = self._column_parallel_tail(
                routed_out, shared_out, tail_targets, shared_pushed=shared_push is not None
            )
            if shared_join is not None:
                # Before the next decode mix finishes the all-gather call the shared stream
                # pushed into.
                shared_join.wait()
            return tail
        if self._reduce_routed_output and self._latent_up_takes_mxfp8(routed_out):
            # The MNNVL all-reduce also quantizes the normalized latent the way latent_up's
            # low-M GEMM would (RMS_NORM_QUANT_MXFP8), which then skips its quantization.
            latent, latent_scale = moe_all_reduce(
                routed_out,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RMS_NORM_QUANT_MXFP8,
                    norm_weight=self.routed_expert_norm.weight,
                    eps=float(self.routed_expert_norm.variance_epsilon),
                ),
            )
            routed_out = self.routed_expert_up_proj.forward_prequantized(latent, latent_scale)
        elif self._reduce_routed_output:
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
        if shared_join is not None:
            shared_join.wait()
        return routed_out, shared_out


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


# ---------------------------------------------------------------------------
# helix_a2a: token-sliced residual stream around head-TP attention.
# ---------------------------------------------------------------------------

# Helix + enable_attention_dp: every MoE layer keeps its residual stream
# (prefix sum, attn-res bank, norms, MoE) on this rank's ceil(T / cp) rows, the
# split the executor's helix all_rank_num_tokens assumes. KDA and MLA stay
# head-TP over the CP group on all T rows: an all-gather feeds them and their
# o_proj partials are reduce-scattered. The MoE runs the attention-DP
# all-to-all with a replicated shared expert. The all-gather is symm_mem
# MULTIMEM (NCCL fallback) and the reduce-scatter is AllReduce + own rows: at
# 16 ranks, T=512 they take 19.4 / 28.1 us against NCCL's 40.0 / 38.4 us.


def _multimem_allgather(mapping: Mapping, max_bytes: int) -> SymmetricMemoryAllGather:
    """SymmetricMemoryAllGather admitted for this GPU and TP size.

    The upstream allow-list stops at SM 10.0 and 8 ranks; MULTIMEM all-gather
    also runs at 16 and 32 ranks on Rubin. The class's rendezvous probe still
    disables it when MULTIMEM is not usable.
    """
    major, minor = torch.cuda.get_device_capability()
    capability = f"{major}.{minor}"
    cls = type(
        "_HelixA2AMultimemAllGather",
        (SymmetricMemoryAllGather,),
        {
            "_WORLD_SIZES_MULTIMEM": {capability: [mapping.tp_size]},
            "_MAX_SIZES": {capability: {mapping.tp_size: max_bytes}},
        },
    )
    # Reuse WORLD when it is the TP group (helix GEN: world == tp * cp).
    group = (
        torch.distributed.group.WORLD
        if torch.distributed.is_initialized()
        and torch.distributed.get_world_size() == mapping.tp_size
        else None
    )
    return cls(mapping, dtype=torch.bfloat16, group=group)


class KimiK3HelixA2A(nn.Module):
    """Moves rows between the full T-row batch and this rank's chunk.

    Row ownership and zero padding follow the upstream helix + attention-DP
    path (``_helix_cp_pad``): rank r owns rows ``[r * chunk, (r + 1) * chunk)``.
    """

    def __init__(
        self,
        mapping: Mapping,
        hidden_size: int,
        max_num_tokens: int,
        allreduce_strategy: AllReduceStrategy,
    ):
        super().__init__()
        if mapping.tp_size <= 1 or mapping.enable_attention_dp:
            raise ValueError(
                "Kimi K3 helix_a2a needs the head-TP (non-attention-DP) mapping; got "
                f"tp_size={mapping.tp_size}, enable_attention_dp={mapping.enable_attention_dp}."
            )
        self.mapping = mapping
        self.world_size = mapping.tp_size
        self.rank = mapping.tp_rank
        padded_rows = -(-max_num_tokens // self.world_size) * self.world_size
        # 1 MiB granularity; can_use_symm_mem rejects out_bytes >= max.
        symm = _multimem_allgather(mapping, ((padded_rows * hidden_size * 2 >> 20) + 1) << 20)
        if symm.disabled:
            logger.warning(
                "Kimi K3 helix_a2a: symm_mem MULTIMEM all-gather unavailable at "
                f"tp_size={self.world_size}; using NCCL all-gather."
            )
        self._symm_allgather = None if symm.disabled else symm
        self._allreduce = AllReduce(
            mapping=mapping, strategy=allreduce_strategy, dtype=torch.bfloat16
        )
        logger.info_once(
            f"Kimi K3 helix_a2a: token-sliced MoE layers over tp_size={self.world_size}, "
            f"all-gather={'nccl' if symm.disabled else 'symm'}",
            key="kimi_k3_helix_a2a",
        )

    def all_rank_num_tokens(self, num_tokens: int) -> List[int]:
        return [-(-num_tokens // self.world_size)] * self.world_size

    def scatter_rows(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """This rank's chunk of a rank-replicated tensor (rows on dim -2)."""
        x, chunk = _helix_cp_pad(x, num_tokens, self.world_size)
        return x.narrow(-2, self.rank * chunk, chunk).contiguous()

    def all_gather(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """``[chunk, H]`` per rank -> ``[num_tokens, H]`` on every rank."""
        gathered = None
        if self._symm_allgather is not None:
            gathered = self._symm_allgather(x, dim=0)
        if gathered is None:
            gathered = allgather(x, self.mapping, dim=0)
        return gathered[:num_tokens]

    def reduce_scatter(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Row-parallel partial sums ``[num_tokens, H]`` -> this rank's ``[chunk, H]``."""
        return self.scatter_rows(self._allreduce(x), num_tokens)


class KimiMLARuntime(nn.Module):
    """Wraps K3 MLA and applies its external TP output reduction."""

    def __init__(
        self,
        cfg: "PretrainedConfig",
        layer_idx: int,
        model_config: ModelConfig,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
        helix_tail: Optional[HelixTail] = None,
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
        # mapping sums the base o_proj's tp*cp partials. reduce_output=False
        # leaves that reduction to the caller (helix_a2a reduce-scatter).
        mapping = model_config.mapping.mapping_for_full_rows()
        reduce_output = reduce_output and not mapping.enable_attention_dp and mapping.tp_size > 1
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
        kv_quant_algo = (
            model_config.quant_config.kv_cache_quant_algo
            if model_config.quant_config is not None
            else None
        )
        # The KV cache manager stores these layers in their own FP8 pool, so
        # the attention backend runs them as FP8 KV-cache MLA.
        fp8_kv_layer_ids = get_kimi_k3_fp8_kv_layer_ids(
            cfg, nvfp4_enabled=kv_quant_algo == QuantAlgo.NVFP4, mapping=mapping
        )
        if layer_idx in fp8_kv_layer_ids:
            if kv_quant_algo != QuantAlgo.NVFP4:
                raise ValueError(
                    f"{KIMI_K3_FP8_KV_LAYERS_ENV} requires the model-wide "
                    "KV cache dtype to be NVFP4."
                )
            kv_quant_algo = QuantAlgo.FP8
            logger.info_once(
                f"Kimi K3 MLA layers using FP8 KV-cache fallback: {sorted(fp8_kv_layer_ids)}.",
                key="kimi_k3_fp8_kv_fallback_layers",
            )
        attention_config.quant_config = QuantConfig(kv_cache_quant_algo=kv_quant_algo)
        for projection_config in attention_config.quant_config_dict.values():
            projection_config.kv_cache_quant_algo = kv_quant_algo
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
            helix_tail=helix_tail,
        )

    def takes_prequantized_input(self, num_rows: int, attn_metadata: AttentionMetadata) -> bool:
        """Whether :meth:`forward` consumes the MXFP8 form of its ``num_rows`` input rows
        (``prequantized_input``, see ``KimiK3MLAAttention.takes_prequantized_input``)."""
        return self.mixer.takes_prequantized_input(num_rows, attn_metadata)

    @property
    def output_all_reduce(self) -> Optional[AllReduce]:
        """The all-reduce of the o_proj partial sums (None without head-sharded TP); an
        ``o_proj`` built with it (``reduces_output``) runs it itself."""
        return self._o_allreduce

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        prequantized_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, MnnvlWorkspaceResult]:
        """The attention output, reduced over the head shards; an ``MnnvlWorkspaceResult`` where
        ``o_proj`` leaves its all-reduce in the workspace."""
        # MLA.forward takes position_ids first; K3 is NoPE, so pass None.
        out = self.mixer(None, hidden_states, attn_metadata, prequantized_input=prequantized_input)
        if self._o_allreduce is not None and not reduces_output(self.mixer.o_proj):
            # Head-sharded TP: sum the row-sharded o_proj partials across
            # the head-shard group.
            out = self._o_allreduce(out)
        return out


# ---------------------------------------------------------------------------
# Dense DWDP: the tables its setup reads off every decoder layer.
# ---------------------------------------------------------------------------

# Plain attributes (a tensor, or a tuple of tensors) under the dense roots that dense DWDP may
# move as weights: the fused KDA projections, decode constants and the low-M GEMM's scale copies.
# Parameters and buffers need no entry.
DWDP_DENSE_ATTR_ALLOW: frozenset[str] = frozenset(
    {
        "_low_m_weight_scale",
        "_bfa_proj_weight",
        "_qkvg_proj_weight",
        "_gate_b_t",
        "_packed_conv_weight",
        "_mtp_conv_weights",
        "_w_q_t",
        "_w_k_t",
        "_w_v_t",
        "_A_log_f32",
        "_dt_bias_f32",
        "_onorm_w_f32",
    }
)

# Names that never make dense DWDP move a storage: per-call scratch tensors and the routing bias.
# A reference under one of these names to a storage that moves through another reference is still
# repointed with it.
DWDP_DENSE_EXCLUDE: frozenset[str] = frozenset(
    {
        "_o_dense",
        "_fused_q_placeholder",
        "softmax_stats_tensor",
        "temp_softmax_stats_tensor",
        "e_score_correction_bias",
    }
)

_DENSE_DWDP_SEGMENTS = ("attn", "moe")
# Children of the MoE block that hold its dense weights: the router, the shared expert and the
# latent projections and norm. The routed experts belong to expert DWDP and are never a dense root.
_DENSE_DWDP_MOE_ROOTS = (
    "gate",
    "shared_experts",
    "routed_expert_down_proj",
    "routed_expert_up_proj",
    "routed_expert_norm",
)


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


@nccl_window_tensor_scoped_module
class KimiLinearDecoderLayer(nn.Module):
    # Read by dense DWDP setup next to ``dense_dwdp_roots``.
    dense_dwdp_attr_allow: frozenset[str] = DWDP_DENSE_ATTR_ALLOW
    dense_dwdp_exclude: frozenset[str] = DWDP_DENSE_EXCLUDE

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        helix_tail: Optional[HelixTail] = None,
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
        # helix_a2a: an MoE layer keeps its residual stream and MoE on this
        # rank's token rows, attention on all rows (see KimiK3HelixA2A).
        self.token_sliced = self.is_moe and _is_helix_a2a(model_config)

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
                mapping=model_config.mapping.mapping_for_full_rows(),
                allreduce_strategy=model_config.allreduce_strategy,
                aux_stream=aux_stream_dict[AuxStreamType.Attention],
                model_config=attention_config,
                reduce_output=not self.token_sliced,
            )
        else:
            self.self_attn = KimiMLARuntime(
                cfg,
                layer_idx,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
                mapping_with_cp=model_config.mapping_with_cp,
                reduce_output=not self.token_sliced,
                helix_tail=helix_tail,
            )

        if self.is_moe:
            self.block_sparse_moe = KimiK3MoERuntime(
                model_config,
                cfg,
                layer_idx,
                aux_stream_dict,
                follows_kda_attention=self.is_kda,
            )
        else:
            situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
            situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
            # Dense layers run on every row (helix_a2a slices only MoE layers).
            mlp_mapping = model_config.mapping.mapping_for_full_rows()
            if mlp_mapping.enable_attention_dp:
                self.mlp_tp_size = 1
            else:
                self.mlp_tp_size = math.gcd(cfg.intermediate_size, mlp_mapping.tp_size)
                if self.mlp_tp_size > mlp_mapping.gpus_per_node:
                    self.mlp_tp_size = math.gcd(self.mlp_tp_size, mlp_mapping.gpus_per_node)
            mlp_model_config = copy.copy(model_config)
            mlp_model_config._frozen = False
            mlp_model_config.mapping = mlp_mapping
            mlp_model_config.quant_config = QuantConfig()
            mlp_model_config._frozen = model_config._frozen
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

        # The dense DWDP manager exists before the model is built (DwdpManager registers it with
        # expert DWDP, py_executor_creator when dense DWDP runs standalone); forward() reaches the
        # dense DWDP runtime through it.
        self._dense_dwdp = get_global_dense_dwdp_manager()

    def dense_dwdp_roots(self, segment: str) -> list[tuple[str, nn.Module]]:
        """The modules holding this layer's dense DWDP weights of ``segment``.

        Implements ``modules.dwdp.dense.types.DenseRootsProvider``. Dense DWDP moves the
        replicated weights reachable from these roots (see ``DWDP_DENSE_ATTR_ALLOW`` and
        ``DWDP_DENSE_EXCLUDE``) and reads them back only inside the windows ``forward`` opens.

        Args:
            segment: ``"attn"`` for the attention weights, ``"moe"`` for the MoE-dense weights:
                the router, the shared expert and the latent projections. The routed experts
                are never a root.

        Returns:
            ``(path relative to this layer, module)`` pairs in walk order; [] when the layer's
            weights stay replicated: a layer without the MoE block (layer 0 of Kimi-K3) or one
            whose weights were removed (layer-wise benchmarks).

        Raises:
            ValueError: ``segment`` is neither ``"attn"`` nor ``"moe"``.
        """
        if segment not in _DENSE_DWDP_SEGMENTS:
            raise ValueError(
                f"Kimi K3 dense DWDP segment must be one of {_DENSE_DWDP_SEGMENTS}; got {segment!r}"
            )
        if not self.is_moe or not _has_weights(self):
            return []
        if segment == "attn":
            if self.is_kda:
                return [("linear_attn", self.linear_attn)]
            return [("self_attn", self.self_attn)]
        moe = self.block_sparse_moe
        return [(f"block_sparse_moe.{name}", getattr(moe, name)) for name in _DENSE_DWDP_MOE_ROOTS]

    def forward(
        self,
        hidden_states: torch.Tensor | _DeferredMoEResidual,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
        capture: Optional[Tuple[Any, int]] = None,
        helix_a2a: Optional[KimiK3HelixA2A] = None,
        defer_moe_residual: bool = False,
        fresh_block_residual: bool = False,
    ) -> Tuple[torch.Tensor | _DeferredMoEResidual, int]:
        """Port of HF ``KimiDecoderLayer._forward_attn_residual`` (per token).

        ``block_residual`` is a preallocated snapshot bank in kernel-native
        ``[K_max, M, H]`` layout. Returns the running prefix sum and the
        number of valid bank rows.

        With ``defer_moe_residual`` (the caller passes the result on to the next
        decoder layer, or the last layer's to the model's output mix), a MoE layer may
        return its residual update unapplied, as a ``_DeferredMoEResidual``: where the
        next decode mix folds it in (``_defers_moe_residual``). ``hidden_states`` is
        accordingly a tensor or the ``_DeferredMoEResidual`` of the previous layer;
        without the flag the result is always a tensor.

        ``capture`` is ``(spec_metadata, layer_id)`` and taps the DSpark aux
        stream for the layer BEFORE this one: the aggregated stream for layer j
        is by definition what its next consumer sees, so the mixture computed
        below already is it. Reading it here beats recomputing it, and is only
        possible because K3 asserts pp_size == 1 -- layer j+1 is always local.
        PP support would need a recompute at the rank boundary.

        On a ``token_sliced`` layer M is this rank's chunk of the batch and
        ``helix_a2a`` moves rows between that chunk and the full batch.

        ``fresh_block_residual``: the last kernel launched before this layer wrote
        ``block_residual`` (the helix_a2a row scatter ahead of the first token-sliced
        layer), so the pre-attention mix reads the snapshots only after its PDL wait.
        """
        sliced = helix_a2a if self.token_sliced else None
        num_tokens = attn_metadata.num_tokens if sliced is not None else None
        if self.token_sliced and helix_a2a is None:
            raise RuntimeError(
                f"Kimi K3 layer {self.layer_idx} is token-sliced but got no helix_a2a"
            )

        def _full_rows(t: torch.Tensor) -> torch.Tensor:
            return t if sliced is None else sliced.all_gather(t, num_tokens)

        # Dense DWDP read windows: every read of this layer's attention weights happens between
        # H1 (wait_layer) and H2 (attention_done), every read of its MoE-dense weights (router,
        # shared expert, latent projections) between H1 and H3 (layer_done). Their staging slots
        # are refilled with a later layer's weights once a window closes, so a read outside its
        # window sees another layer's bytes.
        manager = getattr(self, "_dense_dwdp", None)
        dense = manager.dense if manager is not None else None
        if dense is not None and not dense.covers(self.layer_idx):
            dense = None
        if dense is not None:
            dense.wait_layer(self.layer_idx)  # H1
        valid_block_residual = block_residual[:num_snapshots]
        normed = None
        attn_input_mxfp8 = None
        if isinstance(hidden_states, _DeferredMoEResidual):
            # The previous MoE layer's residual update, folded into this layer's mix, which
            # also hands the attention the MXFP8 form of its input where it consumes one.
            if capture is None and num_snapshots > 0:
                num_rows = hidden_states.residual.shape[0]
                attention = self.linear_attn if self.is_kda else self.self_attn
                # Under row slicing the mixer reads the gathered rows, not this rank's.
                mxfp8 = sliced is None and attention.takes_prequantized_input(
                    num_rows, attn_metadata
                )
                folded = _attn_res_decode_kernel(
                    hidden_states.residual,
                    (hidden_states.routed, hidden_states.shared),
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                    mxfp8=mxfp8,
                    # A KDA QKVG GEMM holding its weights on chip starts beside the mix.
                    entry_trigger=mxfp8 and self.is_kda and attention.qkvg_stages_weights(num_rows),
                )
                if folded is not None:
                    hidden_states, normed, *attn_input_mxfp8 = folded
            if normed is None:
                hidden_states = hidden_states.materialize()
        prefix_sum = hidden_states

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
            capture[0].maybe_capture_hidden_states(capture[1], _full_rows(tapped), None)
            hidden_states = self.input_layernorm(hidden_states)
        elif normed is not None:
            hidden_states = normed
        elif num_snapshots > 0:
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
                snapshots_after_wait=fresh_block_residual,
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None

        # Attention is head-TP over every row: KDA rows are gathered in and its
        # row-sharded o_proj partials reduce-scattered back out here; helix MLA
        # does both itself (_helix_cp_allgather_input / _helix_cp_output_projection).
        prequantized_input = tuple(attn_input_mxfp8) if attn_input_mxfp8 else None
        if self.is_kda:
            hidden_states = self.linear_attn(
                _full_rows(hidden_states),
                attn_metadata,
                prequantized_input=prequantized_input,
            )
            if sliced is not None:
                hidden_states = sliced.reduce_scatter(hidden_states, num_tokens)
        else:
            hidden_states = self.self_attn(
                hidden_states, attn_metadata, prequantized_input=prequantized_input
            )
        if dense is not None:
            dense.attention_done(self.layer_idx)  # H2: no attention weight is read after this.

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
                entry_trigger=self.is_moe
                and self.block_sparse_moe.moe_front_releases_early(prefix_sum),
            )
        if self.is_moe:
            defers = defer_moe_residual and _defers_moe_residual(prefix_sum, num_snapshots + 1)
            routed, shared = self.block_sparse_moe(
                hidden_states,
                sliced.all_rank_num_tokens(num_tokens)
                if sliced is not None
                else getattr(attn_metadata, "all_rank_num_tokens", None),
                return_parts=True,
                # The next layer's decode mix finishes the column-parallel tail's call.
                column_parallel_tail=defers and attn_metadata.num_contexts == 0,
            )
            if dense is not None:
                # H3: no MoE-dense weight is read after this; a deferred residual update carries
                # only the routed and shared outputs.
                dense.layer_done(self.layer_idx)
            residual_update = _DeferredMoEResidual(prefix_sum, routed, shared)
            if defers:
                return residual_update, num_snapshots
            return residual_update.materialize(), num_snapshots

        hidden_states = self.mlp(hidden_states)
        if dense is not None:
            dense.layer_done(self.layer_idx)  # H3: no MoE-dense weight is read after this.
        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, num_snapshots

    def attention_output_reduction(self) -> Tuple[Optional[AllReduce], int]:
        """What an FP8 weight-read attention ``o_proj`` is built with: the attention's all-reduce
        of the o_proj partial sums (it then pushes its partial sums straight into it, MNNVL), and
        the largest row count whose all-reduce the mix after attention finishes itself.

        That mix finishes it where the decode attention-residual kernel serves it: never on
        snapshot layers, whose attention output becomes the running prefix sum; on the others,
        which follow ``layer_idx // attn_res_block_size + 1`` snapshots, over that many
        candidates plus the prefix sum.
        """
        attention = self.linear_attn if self.is_kda else self.self_attn
        if self.layer_idx % self.attn_res_block_size == 0:
            return attention.output_all_reduce, 0
        num_candidates = self.layer_idx // self.attn_res_block_size + 2
        return attention.output_all_reduce, _attn_res_decode_max_tokens(
            self.hidden_size, num_candidates, torch.cuda.current_device()
        )

    def skip_forward(
        self,
        hidden_states: torch.Tensor | _DeferredMoEResidual,
        block_residual: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor | _DeferredMoEResidual, torch.Tensor]:
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
        # One fused Helix generation tail (and receive buffer) for every MLA layer.
        mapping_with_cp = model_config.mapping_with_cp
        self.helix_tail = (
            HelixTail(mapping_with_cp)
            if mapping_with_cp is not None
            and HelixTail.max_tokens(
                mapping_with_cp,
                cfg.num_attention_heads // mapping_with_cp.tp_size,
                cfg.kv_lora_rank,
                cfg.v_head_dim,
            )
            > 0
            else None
        )
        self.layers = nn.ModuleList(
            [
                KimiLinearDecoderLayer(
                    model_config, cfg, layer_idx, self.aux_stream_dict, helix_tail=self.helix_tail
                )
                for layer_idx in range(cfg.num_hidden_layers)
            ]
        )
        # Rows are rank-replicated up to the first token-sliced layer (the dense
        # layers stay on all rows), and sliced from there to the end.
        sliced = [layer.token_sliced for layer in self.layers]
        self.helix_a2a_first_layer = sliced.index(True) if any(sliced) else len(sliced)
        if not all(sliced[self.helix_a2a_first_layer :]):
            raise ValueError(
                "Kimi K3 helix_a2a needs every layer after the first MoE layer to be MoE"
            )
        # After the layers: MegaMoE bootstraps torch.distributed under MPI,
        # which the symm_mem all-gather needs.
        self.helix_a2a = (
            KimiK3HelixA2A(
                model_config.mapping.mapping_for_full_rows(),
                cfg.hidden_size,
                model_config.max_num_tokens,
                model_config.allreduce_strategy,
            )
            if any(sliced)
            else None
        )
        # With an FP4 KV cache the FP4 MLA backend writes a generation-only step's metadata on
        # this stream ahead of the first MLA layer; every MLA layer has the same per-rank query
        # heads.
        self._mla_num_heads = next(
            (layer.self_attn.mixer.num_heads_tp for layer in self.layers if not layer.is_kda), 0
        )
        self._fp4_mla_metadata_stream = (
            torch.cuda.Stream()
            if self._mla_num_heads > 0
            and model_config.quant_config is not None
            and model_config.quant_config.quant_mode.has_fp4_kv_cache()
            else None
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

        # The dense DWDP manager exists before the model is built (DwdpManager registers it with
        # expert DWDP, py_executor_creator when dense DWDP runs standalone); forward() primes dense
        # DWDP through it.
        self._dense_dwdp = get_global_dense_dwdp_manager()

    def _prewarm_k3_moe_front_kernels(self) -> int:
        """Compile the MoE front's kernels (sorting routing, router logits) ahead of serving,
        so no first 65-256-token forward compiles them; returns the kernels compiled (0 when
        this deployment keeps the front on the ops the kernels replace)."""
        from ..moe.kernels.blackwell import kimi_k3_router_logits, kimi_k3_routing

        runtime = next(
            (
                getattr(layer, "block_sparse_moe", None)
                for layer in self.layers
                if getattr(layer, "block_sparse_moe", None) is not None
            ),
            None,
        )
        if runtime is None:
            return 0
        compiled = 0
        if runtime.gate.logits_kernel:
            compiled += kimi_k3_router_logits.precompile()
        if runtime.routed_experts.sort_routing_applies:
            compiled += kimi_k3_routing.precompile(self.num_experts)
        return compiled

    def prewarm_decode_kernels(self, max_num_tokens: int, verify_batches: bool) -> int:
        """JIT-compile the decode attention-residual kernel variants (``kimi_k3_attn_res``), the
        MoE front's sorting-routing and router-logits kernels, and the prequantized FP8 low-M
        GEMMs that forwards of up to ``max_num_tokens`` tokens reach, so no served step compiles
        them; returns the number of kernels compiled.

        The decode-kernel prewarm hook ``DecoderWarmup`` calls on every module that defines it.
        Depends only on the model layout, the device and the attention-residual settings, so
        every rank compiles the same set. ``verify_batches``: this worker runs speculative verify
        batches, which hand the KDA projections MXFP8 operands.
        """
        device_index = torch.cuda.current_device()
        compiled = self._prewarm_k3_moe_front_kernels()
        if not _FUSED_ATTN_RES_ENABLED or not _decode_attn_res_on_device(device_index):
            return compiled
        from ..kernels.blackwell import fp8_block_scale_low_m_gemm, kimi_k3_attn_res

        cfg = self._text_cfg
        variants = _decode_attn_res_variants(
            self.layers,
            cfg.attn_res_block_size,
            cfg.hidden_size,
            max_num_tokens,
            device_index,
            self.helix_a2a_first_layer,
            verify_batches,
        )
        for variant in sorted(variants):
            kimi_k3_attn_res.precompile(
                variant.num_candidates,
                variant.add_mode,
                variant.cluster,
                cfg.hidden_size,
                variant.mxfp8,
                variant.snapshots_after_wait,
            )
        compiled += len(variants)
        if not verify_batches:
            return compiled
        # Prequantized GEMMs: the KDA q/k/v/g projections fed by the mix's MXFP8 output, and the
        # o_proj fed by the fused verify epilogue where it takes an MXFP8 operand (SM107).
        mxfp8_rows = 0
        if any(v.mxfp8 for v in variants):
            mxfp8_rows = min(
                max_num_tokens, kimi_k3_attn_res.MAX_TOKENS, _FUSED_ATTN_RES_MAX_TOKENS
            )
        rows_by_shape: dict[tuple[int, int], int] = {}
        staged_by_shape: dict[tuple[int, int], Callable[[int], bool]] = {}

        def need(
            linear: nn.Module | None, rows: int, stages: Callable[[int], bool] | None = None
        ) -> None:
            if isinstance(linear, _Fp8BlockScaleWeightReadLinear):
                key = (linear.out_features, linear.in_features)
                # The prequantized GEMM takes the whole low-M GEMM range (takes_prequantized_input).
                rows = min(rows, linear._low_m.max_m)
                rows_by_shape[key] = max(rows, rows_by_shape.get(key, 0))
                if stages is not None:
                    staged_by_shape[key] = stages

        for layer in self.layers:
            if not layer.is_kda:
                continue
            mixer = layer.linear_attn
            if not layer.token_sliced and mixer.qkvg_takes_prequantized(1):
                # The mix releases a q/k/v/g projection that stages its weights at its entry.
                need(getattr(mixer, "qkvg_proj", None), mxfp8_rows, mixer.qkvg_stages_weights)
            o_proj = getattr(mixer, "o_proj", None)
            if getattr(o_proj, "supports_prequantized_input", False):
                need(o_proj, max_num_tokens)
        for (n, k), rows in sorted(rows_by_shape.items()):
            if rows > 0:
                compiled += fp8_block_scale_low_m_gemm.precompile_prequantized(
                    n, k, rows, staged_by_shape.get((n, k))
                )
        return compiled

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

        # Dense DWDP H0: make sure the first two layers' dense weights (and, with expert DWDP,
        # their experts) are enqueued before any layer runs (the previous forward's end or the
        # non-overlap executor loop usually already did). After a forward that stopped midway, a
        # WAR barrier on this stream comes first.
        manager = getattr(self, "_dense_dwdp", None)
        if manager is not None and manager.dense is not None:
            manager.ensure_primed()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        num_tokens = attn_metadata.num_tokens if self.helix_a2a is not None else None
        if self.helix_a2a is not None and hidden_states.shape[0] != num_tokens:
            # Rows are split by the metadata token count on every rank.
            raise ValueError(
                f"Kimi K3 helix_a2a does not support padded batches (got "
                f"{hidden_states.shape[0]} rows, metadata says {num_tokens} tokens)."
            )

        # FP4 MLA layers run on the TRTLLM backend, whose metadata holds the FP4 MLA state.
        fp4_mla_state = (
            attn_metadata.fp4_mla_state if self._fp4_mla_metadata_stream is not None else None
        )
        if fp4_mla_state is not None and do_multi_stream():
            fp4_mla_state.launch_generation_metadata_ahead(
                attn_metadata, num_heads=self._mla_num_heads, stream=self._fp4_mla_metadata_stream
            )

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
        last_captured = (
            spec_metadata is not None
            and len(self.layers) > 0
            and (capture_set is None or self.layers[-1].layer_idx in capture_set)
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
            if i == self.helix_a2a_first_layer:
                # Rows are rank-replicated up to here: keep this rank's chunk. hidden_states is a
                # tensor, never a _DeferredMoEResidual: token_sliced implies is_moe and every layer
                # from the first sliced one on is sliced, so the layers before it are dense and
                # only MoE layers defer their residual update.
                hidden_states = self.helix_a2a.scatter_rows(hidden_states, num_tokens)
                block_residual = self.helix_a2a.scatter_rows(block_residual, num_tokens)
            hidden_states, num_snapshots = layer(
                hidden_states,
                block_residual,
                num_snapshots,
                attn_metadata,
                capture=capture,
                helix_a2a=self.helix_a2a,
                # The last layer's goes to the output mix, which folds it in (not where that
                # layer's pre-norm mixture is captured below).
                defer_moe_residual=i + 1 < len(self.layers) or not last_captured,
                # The bank scatter above is the last kernel before this layer's first mix.
                fresh_block_residual=i == self.helix_a2a_first_layer,
            )
        if fp4_mla_state is not None:
            # Joined by the first MLA layer; a no-op unless no layer read it.
            fp4_mla_state.join_generation_metadata()
        sliced = self.helix_a2a is not None
        output = None
        if isinstance(hidden_states, _DeferredMoEResidual):
            # The last MoE layer's residual update, folded into the output mix.
            hidden_states, output = _fold_moe_residual(
                hidden_states,
                block_residual[:num_snapshots],
                self.output_attn_res_proj,
                self.output_attn_res_norm,
                self.norm,
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
                if sliced:
                    tail = self.helix_a2a.all_gather(tail, num_tokens)
                spec_metadata.maybe_capture_hidden_states(last.layer_idx, tail, None)

        if output is None:
            output = _apply_attn_res_and_rmsnorm(
                hidden_states,
                block_residual[:num_snapshots],
                self.output_attn_res_proj,
                self.output_attn_res_norm,
                self.norm,
            )
        hidden_states = output
        if sliced:
            hidden_states = self.helix_a2a.all_gather(hidden_states, num_tokens)
        return hidden_states


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

    @property
    def runtime_hooks(self) -> RuntimeHooks:
        """The base hooks plus the speculative decode step head in one kernel.

        With ``enable_min_latency`` the decode step's small input uploads, the one-model
        sampler's Philox seeds and offsets included, also go out as one upload and one scatter at
        the graph replay (``pack_step_inputs``, ``pack_spec_rng_uploads``).
        """
        min_latency = self.model_config.enable_min_latency
        return replace(
            super().runtime_hooks or DEFAULT_RUNTIME_HOOKS,
            decode_prologue=spec_decode_head,
            pack_step_inputs=min_latency,
            pack_spec_rng_uploads=min_latency,
        )

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
        # Helix + attention DP = helix_a2a: KDA keeps head-TP state across the
        # CP group (every CP rank holds the same requests) and only the MoE
        # layers run attention-DP on ceil(T / cp) token rows each.
        if model_config.mapping.enable_attention_dp and model_config.mapping.tp_size != 1:
            raise NotImplementedError(
                "Kimi K3 helix with enable_attention_dp requires tp_size=1 "
                f"(got tp_size={model_config.mapping.tp_size}): KDA and MLA are "
                "head-TP over the CP group, so there is no request-level DP to add."
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
        model_config.mapping_with_cp = self.mapping_with_cp
        model_config._frozen = True

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        # - Keep block reuse opt-in because its combination with SA
        #   speculative decoding is not validated. Both settings use V2
        #   with the default use_kv_cache_manager_v2=True. The "auto"
        #   setting follows the model preference and can resolve to V1.
        # - tokens_per_block=64: keep the validated K3 page size. Earlier
        #   releases needed it to avoid a legacy MLA fallback that could not
        #   serve attention-DP's 96 query heads. The current FlashInfer
        #   TRTLLM-Gen backend declines H=96 at either page size, allowing
        #   the updated legacy TRTLLM-Gen autotuner to select a Q16 kernel.
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
            # Required suffixes only: ``optional_suffixes`` are per-checkpoint
            # (an AWQ pre-quant scale), and demanding them here would reject
            # every checkpoint of the same layout that does not carry them.
            # Unmatched expert keys only warn, and this validator already
            # exempts expert keys from that, so the present ones stay quiet.
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
        if getattr(self.model, "_dense_dwdp_active", False):
            # The dense weights are views of staging slots that several layers share: a load
            # would write into them, and the next gather overwrites that from the row store.
            raise RuntimeError(
                "Kimi K3 load_weights is not supported while dense DWDP is active: the dense "
                "weights live in staging slots shared across layers"
            )
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
            moe: KimiK3MoERuntime,
            base: str,
            local_slot_id: int,
            expert_idx: int,
            get_tensor,
            get_optional_tensor,
        ):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            backend = moe.routed_experts.backend
            # Every route into an expert goes through here (file-grouped,
            # split-file and the shared-dict fallback alike), so this is the
            # one place preparation has to be hooked.
            ensure_prepared(moe)
            moe.expert_ckpt_spec.loader(
                backend, base, expert_idx, local_slot_id, get_tensor, get_optional_tensor
            )
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
                    lambda key: _materialize(weights[key]) if key in weights else None,
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
                spec = moe.expert_ckpt_spec
                for local_slot_id, expert_idx in enumerate(moe.local_expert_ids):
                    keys = [
                        f"{base}.{expert_idx}.{w}.{kind}"
                        for w in ("w1", "w2", "w3")
                        for kind in spec.kinds
                    ]
                    # An optional tensor this checkpoint does store has to be
                    # counted here: it decides which files the job opens, and
                    # ``get_optional_tensor`` below can only read a file that
                    # ``files`` named.
                    keys += [
                        key
                        for key in (
                            f"{base}.{expert_idx}.{suffix}" for suffix in spec.optional_suffixes
                        )
                        if checkpoint_key(key) in weight_map
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
                    # A job lands here only when every key it needs -- optional
                    # ones included, see the grouping above -- is in this file,
                    # so presence in this file is the whole question.
                    file_keys = set(fh.keys())
                    for moe, base, local_slot_id, expert_idx in jobs:
                        load_expert(
                            moe,
                            base,
                            local_slot_id,
                            expert_idx,
                            lambda key: fh.get_tensor(checkpoint_key(key)),
                            lambda key: (
                                fh.get_tensor(checkpoint_key(key))
                                if checkpoint_key(key) in file_keys
                                else None
                            ),
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

                    def get_optional_tensor(key):
                        source_key = checkpoint_key(key)
                        # Absent from the index means the checkpoint does not
                        # store it; present means the grouping above put its
                        # file in ``handles``.
                        if source_key not in weight_map:
                            return None
                        return handles[weight_map[source_key]].get_tensor(source_key)

                    load_expert(*job, get_tensor, get_optional_tensor)
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
            o_proj_reduction = layer.attention_output_reduction()
            for name, linear in attention.named_children():
                if isinstance(linear, TrtllmLinear) and linear.has_fp8_block_scales:
                    reduction = o_proj_reduction if name == "o_proj" else (None, 0)
                    _swap_linear_to_fp8_weight_read(attention, name, (TrtllmLinear,), *reduction)
            fused = getattr(attention, "qkvg_proj", None)
            if isinstance(fused, _Fp8BlockScaleWeightReadLinear):
                attention.accept_prequantized_qkvg_input()
                offset = 0
                for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
                    projection = getattr(attention, name)
                    projection._fused_projection = (weakref.ref(fused), offset)
                    offset += projection.out_features
                    # Repoint to the fused rows now rather than in post_load:
                    # deferring kept every layer's standalone codes alongside
                    # its fused copy, ~22.6 GiB at peak across K3's 69 KDA
                    # layers, which overflows a DEP8 context rank.
                    projection.transform_weights()
        logger.info(
            f"Kimi K3: loaded {num_params} parameters and expert slices of {num_moe_layers} layers"
        )
        if self._fp8_weight_read_moe_mlp and is_sm_100f():
            converted = _convert_moe_mlps_to_fp8_weight_read(
                self.model, include_fused_gate_up=self.model_config.mapping.enable_attention_dp
            )
            logger.info(f"Kimi K3: quantized {converted} BF16 shared/latent MLP projections to FP8")
