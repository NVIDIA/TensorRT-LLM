# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MegaMoE CuteDSL NVFP4 backend.

ConfigurableMoE-compatible MoE backend wrapping the ported
architecture-specific MegaMoE kernels (fused dispatch + FC1 + activation + FC2 +
combine) from
``tensorrt_llm/_torch/cute_dsl_kernels/cutedsl_megamoe``. The kernel is
invoked through the standard CuteDSL TunableRunner / torch op pattern;
the runner + op live in
``tensorrt_llm/_torch/moe/custom_ops/cute_dsl_megamoe_custom_op.py``. This
file only owns:

  * capability gating (``can_implement``)
  * lifecycle hooks (``__init__`` / ``create_weights`` /
    ``load_weights`` / ``post_load_weights`` /
    ``validate_configurable_moe``)
  * EP process group resolution
  * BF16 -> NVFP4 activation quantization (``quantize_input``)
  * ``run_moe`` boundary: stage activation + topk into the kernel ABI,
    build the ``MegaMoECuteDslWeightView`` from the quant method, call
    ``torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell``, return the
    ``(T, hidden)`` output (the kernel collapses the top-k axis).

``run_moe`` is a single unified path for both topologies. Only the
SOURCE of the kernel's input/output buffers branches on ``ep_size``:

  * ``ep_size == 1``: local CUDA tensors (cudaMalloc). No
    ``torch.distributed`` dependency, no rendezvous, no cuMem VMM
    overhead. ``peer_offsets = [0]`` collapses the kernel's
    ``peer_rank_ptr_mapper.map(local_addr, 0, off) == local_addr +
    off`` to a self-mapped pointer (NVSHMEM degenerate convention).
  * ``ep_size > 1``: regions carved out of the build-time-rendezvous'd
    :class:`~tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op.MegaMoeSymmMemProvider`
    symmetric buffer; ``peer_offsets[r] = peer_base[r] - local_base``
    enables in-kernel cross-GPU NVLink load/store via
    ``peer_rank_ptr_mapper.map``.

``_acquire_buffers`` is the only branch point; staging, kernel launch,
and output finalization are identical across topologies.

Remaining hard gate:

  * Multi-rank execution requires the cuMem symmetric-memory provider
    to have completed its rendezvous at ``create_weights`` time
    (``self._symm_provider`` non-None); ``run_moe`` raises
    ``MegaMoeCuteDslUnavailable`` otherwise with an actionable message
    pointing at Ray / DeviceMesh / mpirun.

The kernel ABI threads per-expert ``fc31_alpha`` / ``fc2_alpha`` /
``fc1_norm_const`` through the fused FC1/FC2 path. ``fc1_norm_const``
preserves each expert's raw ``w2.input_scale`` as a reciprocal global
scale for the FC1-output NVFP4 quant, so real NVFP4 checkpoints with
non-1 scales compute correctly.
"""

from __future__ import annotations

import os
import socket
import time
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

# ``megamoe_activation_sf_bytes_per_row`` lives at module top of the
# custom-op file (NOT inside its ``IS_MEGAMOE_OP_AVAILABLE`` gate), so
# it is always importable. The provider / shared-workspace helpers used
# in ``_alloc_symm_provider`` and ``_ensure_local_staging`` ARE inside
# that gate and therefore stay lazy at the call site.
from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
    megamoe_activation_sf_bytes_per_row,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ....autotuner import AutoTuner
from ....cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType, Fp4QuantizedTensor
from ..activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivation,
    MoEActivationSupport,
)
from ..impl_base import MoEImplBase, apply_moe_impl_construction_state
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    MoEStaticCapability,
)
from ..impl_environment import MoEDep, MoEEnvFlag, env_flag_is_on
from ..impl_identity import MoEImplDescriptor, MoEImplId, register_moe_impl
from ..interface import MoESchedulerKind, MoEWeightLoadingMode, _reject
from ..quantization import NVFP4MegaMoECuteDslMethod
from ..routing import BaseMoeRoutingMethod

__all__ = [
    "MegaMoECuteDsl",
    "MegaMoeCuteDslUnavailable",
    "MegaMoECuteDslWeightView",
    "TrtllmCutedslMegaMoeNvfp4Impl",
    "is_megamoe_cute_dsl_runtime_available",
]


# Optional NVTX ranges partition host submission overhead.
_MM_NVTX_ON = None


def _nvtx_on() -> bool:
    global _MM_NVTX_ON
    if _MM_NVTX_ON is None:
        import os as _o

        _MM_NVTX_ON = _o.environ.get("TRTLLM_MOE_REBALANCE_NVTX") == "1"
    return _MM_NVTX_ON


def _nvtx(name: str) -> None:
    if _nvtx_on():
        torch.cuda.nvtx.range_push(name)


def _nvtx_end() -> None:
    if _nvtx_on():
        torch.cuda.nvtx.range_pop()


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


class MegaMoeCuteDslUnavailable(RuntimeError):
    """Raised when the active environment cannot import the symbols required by
    the active MegaMoE kernel (cu13 Cutlass DSL + cute_nvgpu MMA
    atoms / cutlass._mlir APIs used by sym_buffer)."""


_RUNTIME_PROBE_CACHE: Dict[int, Union[bool, str]] = {}

_MEGAMOE_PRIME_LADDER = os.environ.get("MEGAMOE_PRIME_LADDER", "1") == "1"
_MEGAMOE_PRIMED_LADDERS: set = set()

# Distinguish an absent arm attribute from an explicit False when restoring it.
_PRIME_ATTR_UNSET = object()


def is_megamoe_cute_dsl_runtime_available() -> Tuple[bool, Optional[str]]:
    """Return whether the CUDA 13 Cutlass DSL runtime exposes all symbols the
    ported MegaMoE CuteDSL kernel needs.

    Stricter than ``IS_CUTLASS_DSL_AVAILABLE``, which only confirms that
    ``cutlass`` / ``cutlass.cute`` import cleanly. The MegaMoE kernel
    ABI also requires ``cutlass.torch.from_dlpack``, ``cutlass._mlir``,
    ``cute_nvgpu`` MMA atoms, and async-copy helpers. SM107 additionally
    requires ``cutlass.utils.rubin_helpers``. Older builds return
    ``(False, reason)``.

    Returns ``(True, None)`` on success or ``(False, reason)`` with an
    actionable message. The result is cached for the process lifetime.
    """
    sm_version = get_sm_version() if torch.cuda.is_available() else -1
    cached = _RUNTIME_PROBE_CACHE.get(sm_version)
    if cached is True:
        return True, None
    if isinstance(cached, str):
        return False, cached

    if not IS_CUTLASS_DSL_AVAILABLE:
        reason = (
            "Cutlass DSL is not importable on this environment; install "
            "nvidia-cutlass-dsl[cu13] to enable MegaMoECuteDsl."
        )
        _RUNTIME_PROBE_CACHE[sm_version] = reason
        return False, reason

    try:
        import cutlass  # noqa: F401
        import cutlass.cute as cute  # noqa: F401
        import cutlass.pipeline  # noqa: F401
        import cutlass.torch  # noqa: F401
        from cutlass._mlir import ir  # noqa: F401
        from cutlass.base_dsl.native_struct import native_struct  # noqa: F401
        from cutlass.cutlass_dsl import (  # noqa: F401
            Int32,
            Int64,
            Uint8,
            dsl_user_op,
            extract_mlir_values,
            new_from_mlir_values,
        )
    except ImportError as e:
        reason = (
            f"MegaMoECuteDsl requires CUDA 13 Cutlass DSL symbols; got "
            f"ImportError={e!r}. Install nvidia-cutlass-dsl[cu13]>=4.5.0 "
            f"(see PR #14354)."
        )
        _RUNTIME_PROBE_CACHE[sm_version] = reason
        return False, reason

    try:
        from cutlass.cute.nvgpu import cpasync, tcgen05  # noqa: F401
    except ImportError as e:
        reason = (
            f"MegaMoECuteDsl requires cutlass.cute.nvgpu.tcgen05 + cpasync; "
            f"missing {e!r}. Install a Blackwell-capable cutlass-dsl wheel."
        )
        _RUNTIME_PROBE_CACHE[sm_version] = reason
        return False, reason

    if sm_version == 107:
        try:
            try:
                import cutlass.memory as cutlass_memory
            except ImportError:
                import cutlass.utils as cutlass_memory
            import cutlass.utils.rubin_helpers as rubin_helpers

            if not hasattr(rubin_helpers, "make_blockscaled_trivial_tiled_mma"):
                raise ImportError(
                    "cutlass.utils.rubin_helpers lacks make_blockscaled_trivial_tiled_mma"
                )
            if not hasattr(cutlass_memory, "get_smem_capacity_in_bytes"):
                raise ImportError("Cutlass DSL lacks get_smem_capacity_in_bytes")
            if cutlass_memory.get_smem_capacity_in_bytes("sm_107") <= 0:
                raise ValueError("invalid SM107 shared-memory capacity")
            if cute.arch.get_max_tmem_alloc_cols("sm_107") <= 0:
                raise ValueError("invalid SM107 tensor-memory capacity")
        except (
            AttributeError,
            ImportError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
            reason = (
                f"MegaMoECuteDsl on SM107 requires Rubin support in the "
                f"Cutlass DSL wheel; missing {e!r}. Use a Rubin-capable "
                f"Cutlass DSL build."
            )
            _RUNTIME_PROBE_CACHE[sm_version] = reason
            return False, reason

    try:
        from ....cute_dsl_kernels.cutedsl_megamoe import (  # noqa: F401
            BlackwellInferenceMegaMoE,
            RubinInferenceGenphaseMegaMoE,
            RubinInferenceLocalMegaMoE,
            RubinInferenceMegaMoE,
        )
    except ImportError as e:
        reason = (
            f"Ported MegaMoE NVFP4 kernel package failed to import: "
            f"{e!r}. Verify tensorrt_llm/_torch/cute_dsl_kernels/"
            f"cutedsl_megamoe is in the install tree."
        )
        _RUNTIME_PROBE_CACHE[sm_version] = reason
        return False, reason

    _RUNTIME_PROBE_CACHE[sm_version] = True
    return True, None


# ---------------------------------------------------------------------------
# Tensor dtype helpers
# ---------------------------------------------------------------------------
#
# ``cutlass_torch.from_dlpack`` derives the cute tensor ``element_type``
# from the torch dtype. The backend stores activations / weights as raw
# uint8 for portability, but the kernel needs NVFP4 / FP8 dtypes:
#
#   * NVFP4 packed tensors -> ``torch.float4_e2m1fn_x2`` (raw uint8 trips
#     "unsupported a_dtype/b_dtype: Int8 / Float4E2M1FN").
#   * FP8 block-scale tensors -> ``torch.float8_e4m3fn`` (raw uint8 trips
#     "expects the 'sf_dtype' Op parameter to be one of Float8E8M0FNU"
#     because cute falls back to MXFP4 when sf_dtype is not FP8).
#
# Applying the views before the custom op call also keeps the
# autotuner's ``_create_tensor_like`` aligned with the runner's
# ``_to_cute``.


# The launch path memoizes these dtype views without copying their storage.
def _as_nvfp4(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float4_e2m1fn_x2 else t.view(torch.float4_e2m1fn_x2)


def _as_fp8_sf(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float8_e4m3fn else t.view(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Weight view passed to ``run_moe``
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MegaMoECuteDslWeightView:
    """Bundles the MegaMoE-format weight tensors built by
    ``NVFP4MegaMoECuteDslMethod.process_weights_after_loading``.

    The kernel reads these as local-only (NOT through symmetric heap);
    placement is unconstrained CUDA memory. Shapes match the
    MegaMoE kernel ABI.

    ``fc31_alpha`` / ``fc2_alpha`` / ``fc1_norm_const`` are per-expert
    NVFP4 scale tensors consumed by the kernel ABI.
    ``fc31_alpha`` and ``fc2_alpha`` are passed through as the FC1 / FC2
    per-expert global scales, and ``fc1_norm_const`` is built from each
    expert's raw ``w2.input_scale`` as the FC1-output / FC2-input NVFP4 quant
    norm_const in ``NVFP4MegaMoECuteDslMethod.process_weights_after_loading``.
    """

    # NVFP4 packed bytes, stored natural ``(slots, N, K_bytes)`` with K
    # (hidden//2 for fc1, intermediate//2 for fc2) innermost / stride-1.
    # ``uint8`` and ``float4_e2m1fn_x2`` are both 1 byte/element, so the
    # ``torch.float4_e2m1fn_x2`` re-view is a same-shape dtype reinterpret
    # (each byte holds 2 packed fp4 along K). The kernel-input prep in the
    # runner transposes the last two dims to a K-major ``(slots, K_bytes, N)``
    # VIEW (K stays stride-1) before the kernel call.
    # Storage shapes registered in ``NVFP4MegaMoECuteDslMethod.create_weights``.
    fc1_weight: torch.Tensor  # uint8 storage (slots, expand_intermediate, hidden//2)
    # FP8 atom-swizzled per-slot blocked scale, flattened to 1-D per slot.
    fc1_weight_sf: torch.Tensor  # uint8 storage (slots, fc1_sf_flat_size)
    fc2_weight: torch.Tensor  # uint8 storage (slots, hidden, intermediate//2)
    fc2_weight_sf: torch.Tensor  # uint8 storage (slots, fc2_sf_flat_size)
    # NVFP4 per-expert scale tensors consumed by the kernel ABI.
    fc31_alpha: torch.Tensor  # (slots,) fp32; FC1 per-expert global scale
    fc2_alpha: torch.Tensor  # (slots,) fp32; FC2 per-expert global scale
    # (slots,) fp32; FC1-output (= FC2-input) NVFP4 quant norm_const, one
    # reciprocal raw w2.input_scale per local expert slot.
    fc1_norm_const: torch.Tensor


@dataclass(frozen=True)
class _MegaMoeBuffers:
    """Unified kernel-ABI view over MegaMoE CuteDSL's user-domain buffers.

    Single-rank and multi-rank execution differ ONLY in where these
    tensors physically live:

      * ``ep_size == 1``: local CUDA memory; ``peer_offsets == [0]``.
      * ``ep_size > 1``: peer-mapped symmetric heap regions from
        ``MegaMoeSymmMemProvider``; ``peer_offsets[r] = peer_base[r] -
        local_base``.

    ``topk_idx_local`` stays in plain CUDA memory in BOTH paths because
    the kernel reads it through ``input_topk_idx_buffer[token, slot]``
    only on the local rank -- peers never call
    ``peer_rank_ptr_mapper.map`` on it.

    All tensors are sized to ``max_num_tokens`` along the leading
    dimension so the kernel's compile-time constexpr matches the
    buffer-time ``max_tokens_per_rank``.
    """

    activation: torch.Tensor  # (max_T, hidden // 2) uint8 (NVFP4 packed)
    activation_sf: torch.Tensor  # (max_T, sf_bytes_per_row) uint8 (FP8 SF)
    topk_weights: torch.Tensor  # (max_T, top_k) float32
    combine_output: torch.Tensor  # (max_T, top_k, hidden) bf16
    shared_workspace: torch.Tensor  # (shared_ws_bytes,) uint8
    peer_offsets: List[int]  # length == world_size; [0] for single-rank
    topk_idx_local: torch.Tensor  # (max_T, top_k), int32 on Rubin, otherwise int64


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@register_moe_impl
class TrtllmCutedslMegaMoeNvfp4Impl(MoEImplBase):
    """``trtllm.cutedsl.mega_moe.nvfp4``.

    MoE backend wrapping the exported MegaMoE CuteDSL NVFP4 fused kernels;
    ``MegaMoECuteDsl`` below is an alias onto this class.

    The per-architecture kernel variants dispatch inside this one identity,
    because architecture is not an id segment: all four segments hold whichever
    kernel runs. The class name carries its ``kernel_name`` segment because
    ``CuteDslFusedMoE``'s ``trtllm.cutedsl.grouped_gemm.nvfp4`` will differ from
    this id in that field alone.

    Capability gate (``can_implement``): SM100/103/107 + NVFP4 +
    bfloat16 activation + CUDA 13 Cutlass DSL runtime present.

    Topology source-of-truth: :meth:`_acquire_buffers`.

      * ``ep_size == 1``: local CUDA tensors and ``peer_offsets = [0]``,
        which collapses the kernel's ``peer_rank_ptr_mapper.map(local,
        0, off)`` to a self-mapped pointer (NVSHMEM degenerate).
      * ``ep_size > 1``: regions carved out of
        :class:`~tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op.MegaMoeSymmMemProvider`'s
        rendezvous'd symmetric buffer. ``create_weights`` performs the
        (collective) ``torch_symm_mem.rendezvous`` at build time so
        forward time stays free of cross-rank IPC. ``run_moe`` raises
        :class:`MegaMoeCuteDslUnavailable` if the provider was not
        allocated (e.g. ``torch.distributed`` not initialised).
    """

    descriptor = MoEImplDescriptor(
        identity=MoEImplId("trtllm", "cutedsl", "mega_moe", "nvfp4"),
        # Kernel owns dispatch + GEMM1 + SwiGLU + GEMM2 + combine via the
        # CuteDSL three-stage dispatch primitives + NVLink barrier; the
        # scheduler must skip host-side comm and lockstep every chunk.
        scheduler_kind=MoESchedulerKind.FUSED_COMM,
        # Static and dynamic EPLB both work: see ``_supports_load_balancer``
        # below for why the MegaMoE-format derived parameters migrate
        # atomically.
        capabilities=MoEStaticCapability(supports_eplb=True, supports_prefill_rebalance=True),
        doc="MegaMoE CuteDSL fused NVFP4 kernels: NVFP4 weights and activations, SM100/103/107.",
    )

    # Read off the descriptor, not restated, so the published contract and the
    # one the scheduler reads cannot drift apart.
    scheduler_kind = descriptor.scheduler_kind

    capabilities = descriptor.capabilities

    input_requirement = descriptor.input_requirement

    _SUPPORTED_ACTIVATION_DTYPES = frozenset({torch.bfloat16})

    # The elementwise function and its constants are codegen-time literals, so
    # only a uniform per-layer value is representable.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.SwigluBias, ActivationType.SiTu}),
        alpha_beta=ActivationParamShape.UNIFORM_SCALAR,
        limit=ActivationParamShape.UNIFORM_SCALAR,
    )

    # Legal combine wire formats; must stay in sync with
    # ``CombineFormat.parse`` in the kernel package's token_comm.py.
    _SUPPORTED_COMBINE_FORMATS = frozenset({"bf16", "32e4m3xe8m0", "16e2m1xbf16"})

    # Disabled defaults also cover construction paths before weight allocation.
    _rebalance_slots_active: int = 0
    _quantize_reserved_sms: int = 0
    # Resident count H is fixed when create_weights first widens the backend to H + S.
    _rebalance_home_experts: Optional[int] = None
    # Resolved ``MoeRebalanceConfig`` or ``None``; same class-level fallback.
    _rebalance_config = None
    # Allocate the arena once; the copy endpoint retains its plane identities.
    _rebalance_arena = None
    # Bind the producer after loading has finalized the seven live plane aliases.
    _rebalance_scheduler_group = None

    # ------------------------------------------------------------------
    # Capability gating
    # ------------------------------------------------------------------

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """Check static eligibility; runtime providers and tensor values are validated later."""
        is_minimax_swiglu_bias = (
            p.swiglu_gptoss_style
            and p.activation_type == ActivationType.SwigluBias
            and p.bias is False
            and p.activation_constants == frozenset({"alpha", "beta", "clamp"})
        )
        if not is_sm_100f(d.env.sm) and d.env.sm != 107:
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                f"MegaMoECuteDsl requires SM100, SM103 or SM107; got SM{d.env.sm}.",
            )
        if p.dtype_act not in cls._SUPPORTED_ACTIVATION_DTYPES:
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                f"MegaMoECuteDsl supports activations in "
                f"{cls._SUPPORTED_ACTIVATION_DTYPES}, got {p.dtype_act}.",
            )
        if p.swiglu_gptoss_style and not is_minimax_swiglu_bias:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                "MegaMoECuteDsl supports MiniMax-style SwigluBias without "
                "expert bias, but not the GPT-OSS expert-bias package.",
            )
        if p.quant_algo != QuantAlgo.NVFP4:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"MegaMoECuteDsl supports NVFP4 only, got quant_algo={p.quant_algo}.",
            )
        # ``hidden_size % 32`` covers the kernel's NVFP4 SF leg
        # alignment; the SF row width is padded to
        # ``round_up(ceil(hidden/16), 4)`` at every allocation site (see
        # ``megamoe_activation_sf_bytes_per_row``).
        if p.hidden_size is not None and (p.hidden_size <= 0 or p.hidden_size % 32 != 0):
            return _reject(
                MoERejectReason.SHAPE_UNALIGNED,
                f"MegaMoECuteDsl requires positive hidden_size divisible "
                f"by 32 (NVFP4 SF leg alignment); got {p.hidden_size}.",
            )
        # The kernel's expand_intermediate = 2 * intermediate must be
        # divisible by 2 * Fc1GateUpInterleave (32) -> intermediate % 16.
        if p.intermediate_size is not None and (
            p.intermediate_size <= 0 or p.intermediate_size % 16 != 0
        ):
            return _reject(
                MoERejectReason.SHAPE_UNALIGNED,
                f"MegaMoECuteDsl requires positive intermediate_size "
                f"divisible by 16 (Fc1GateUpInterleave); got "
                f"{p.intermediate_size}.",
            )
        if not d.env.has_dep(MoEDep.MEGAMOE_CUTEDSL_RUNTIME):
            return _reject(
                MoERejectReason.DEP_MISSING,
                "MegaMoECuteDsl requires nvidia-cutlass-dsl[cu13] >= 4.5.0",
            )
        # The fused path also requires the ``trtllm::cute_dsl_megamoe_nvfp4_*``
        # custom ops to be registered (strict import of every kernel symbol in
        # cute_dsl_megamoe_custom_op).
        if p.activation_type not in cls.activation_support.kinds:
            supported = ", ".join(sorted(a.name for a in cls.activation_support.kinds))
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"MegaMoECuteDsl does not support activation {p.activation}; "
                f"supported: {supported}",
            )
        if d.tp_size != 1:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"MegaMoECuteDsl is EP-only (moe_tp_size=1); got tp_size={d.tp_size}.",
            )
        if d.cluster_size != 1:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"MegaMoECuteDsl assumes cluster_size=1; got cluster_size={d.cluster_size}.",
            )
        if d.num_slots % max(d.ep_size, 1) != 0:
            return _reject(
                MoERejectReason.SLOTS_NOT_DIVISIBLE_BY_EP,
                f"MegaMoECuteDsl requires num_slots ({d.num_slots}) "
                f"divisible by ep_size ({d.ep_size}).",
            )
        # The fused kernel returns a globally combined routed output. MiniMax-M3
        # accounts for that when composing routed and shared experts; the other
        # current model wrappers would apply a second AllReduce under TEP.
        if d.parallel_size > 1 and not d.use_dp and not is_minimax_swiglu_bias:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                "MegaMoECuteDsl supports TEP only for bias-free MiniMax-style "
                "SwigluBias; other model compositions would reduce its already-global "
                "routed output twice.",
            )
        # ADP wider than EP would need an outer allgather + reducescatter
        # wrapper that this backend does not have.
        if d.use_dp and d.parallel_size > 1 and d.ep_size != d.parallel_size:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"MegaMoECuteDsl with enable_attention_dp=True requires "
                f"ep_size == parallel_size (got ep_size={d.ep_size}, "
                f"parallel_size={d.parallel_size}).",
            )
        if not d.env.has_dep(MoEDep.MEGAMOE_CUTEDSL_OP):
            return _reject(
                MoERejectReason.DEP_MISSING,
                "MegaMoECuteDsl requires the trtllm::cute_dsl_megamoe_nvfp4_* custom ops",
            )
        return MoEEligibility.ok()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    @staticmethod
    def _reject_rebalance_dwdp_overlap(mapping, helper_slots: int, layer_idx=None) -> None:
        """Reject simultaneous helper-slot rebalance and DWDP before allocation.

        DWDP makes every rank own all experts and rewrites the slot axis after
        weight creation, invalidating the H + S arena geometry.
        """
        if int(helper_slots) <= 0:
            return
        if not bool(getattr(mapping, "dwdp_enabled", False)):
            return
        raise ValueError(
            f"MoE rebalance S={helper_slots} at layer {layer_idx} cannot run with "
            f"DWDP size {getattr(mapping, 'dwdp_size', '?')}: DWDP rewrites the "
            "H + S weight axis. Disable rebalance or use dwdp_size=1."
        )

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        init_load_balancer: bool = False,
        activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
        **kwargs,
    ) -> None:
        # ``aux_stream_dict`` is accepted for ``create_moe_backend`` signature
        # uniformity but ignored: FUSED_COMM kernels must not use the chunk
        # overlap stream because launch order must be lockstep across EP.
        del aux_stream_dict
        super().__init__(eplb=None)
        apply_moe_impl_construction_state(
            self,
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=None,
            weight_loading_mode=weight_loading_mode,
            layer_idx=layer_idx,
            activation=activation,
            init_load_balancer=init_load_balancer,
        )

        # Topology / activation eligibility is owned by ``can_implement``.
        # Keep construction-only invariants that are not part of (p, d).
        if apply_router_weight_on_input:
            raise ValueError(
                "MegaMoECuteDsl does not support apply_router_weight_on_input; "
                "the fused kernel applies routing weights on the MoE output."
            )
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # Preserve V4's pre-quantization routing weights; other models apply them after FC2.
        self.apply_topk_in_fc1 = "DeepseekV4ForCausalLM" in (
            getattr(model_config.pretrained_config, "architectures", None) or ()
        )
        self._topk_idx_dtype = torch.int32 if get_sm_version() == 107 else torch.int64

        # The wire format must match on all EP ranks: it sizes the symmetric workspace.
        combine_format = os.environ.get("MEGAMOE_COMBINE_FORMAT", "bf16")
        if combine_format not in self._SUPPORTED_COMBINE_FORMATS:
            raise ValueError(
                f"MEGAMOE_COMBINE_FORMAT must be one of "
                f"{sorted(self._SUPPORTED_COMBINE_FORMATS)}; got {combine_format!r}."
            )
        self.combine_format = combine_format

        # AutoTuner tactic-sweep opt-in via MEGAMOE_TACTIC_AUTOTUNE=1
        # (bench-only knob; default OFF so serving warmup never pays the
        # MERGE lockstep sweep). Must be set identically on ALL ranks (it
        # gates collective tuning behavior).
        self.tactic_autotune = os.environ.get("MEGAMOE_TACTIC_AUTOTUNE", "0") == "1"

        # Resolve helper capacity once; create_weights and routing share this value.
        self._rebalance_config = getattr(model_config, "moe_rebalance", None)
        # The config and disable override must agree on every EP rank.
        _rebalance_env_disabled = env_flag_is_on(MoEEnvFlag.MOE_REBALANCE_DISABLE.value)
        _rebalance_requested = (
            self._rebalance_config is not None
            and bool(getattr(self._rebalance_config, "enabled", False))
            # ``helper_slots_per_rank == 0`` is defined as equivalent to
            # disabled (see MoeRebalanceConfig): no helper capacity means no
            # receive pool to allocate and M == H anyway.
            and int(getattr(self._rebalance_config, "helper_slots_per_rank", 0)) > 0
            and not _rebalance_env_disabled
            and bool(self.capabilities.supports_prefill_rebalance)
        )
        # S controls the weight axis, symmetric workspace, and compiled helper gate.
        self._rebalance_autotune_helpers_initialized = False
        self._rebalance_slots_active = (
            int(getattr(self._rebalance_config, "helper_slots_per_rank", 0))
            if _rebalance_requested
            else 0
        )
        self._quantize_reserved_sms = 0
        if self._rebalance_slots_active > 0:
            from .rebalance_slot_scheduler_v2 import TMA_COPY_SM_COUNT

            self._quantize_reserved_sms = TMA_COPY_SM_COUNT
            if not hasattr(torch.ops.trtllm.fp4_quantize, "sm_budget"):
                raise RuntimeError(
                    "ON MegaMoE quantization requires the native "
                    "trtllm::fp4_quantize.sm_budget overload; rebuild the "
                    "TensorRT-LLM native operators."
                )
        self._rebalance_warmup = False
        self._rebalance_plan_ran = False
        # Reject DWDP before it can rewrite the same slot axis.
        self._reject_rebalance_dwdp_overlap(
            getattr(self, "mapping", None),
            self._rebalance_slots_active,
            layer_idx,
        )
        if self._rebalance_slots_active > 0:
            logger.debug(
                f"[MegaMoECuteDsl] prefill rebalance active "
                f"(S={self._rebalance_slots_active}); "
                f"create_weights will widen expert_size_per_partition to "
                f"M = H + S."
            )

        # Buffer sizing. MoE layers execute serially per forward; one pool
        # sized to the worst-case per-rank tokens covers every layer. The
        # kernel compile takes this as the static ``max_tokens_per_rank``.
        self.max_num_tokens = int(
            getattr(model_config, "moe_max_num_tokens", 0)
            or getattr(model_config, "max_num_tokens", 0)
            or 4096
        )

        # Adaptive ``max_tokens_per_rank`` bucketing: one symmetric provider +
        # kernel per ladder bucket; per launch pick the smallest bucket that
        # fits the lockstep chunk max (see ``_select_launch_max_tokens``). The
        # ladder cap is min(moe_max_num_tokens, engine max_num_tokens) -- both
        # rank-identical, keeping the ladder rank-identical (the
        # NVLink-barrier invariant).
        per_rank_cap = self.max_num_tokens
        _engine_max = int(getattr(model_config, "max_num_tokens", 0) or 0)
        if _engine_max > 0:
            per_rank_cap = min(per_rank_cap, _engine_max)
        self._maxt_buckets = self._resolve_maxt_buckets(per_rank_cap)
        # Per-bucket symmetric providers (bucket -> MegaMoeSymmMemProvider);
        # filled in ``create_weights`` on the multi-rank EP path.
        self._symm_providers = {}
        self._genphase_direct_input_buckets: set[int] = set()
        # Lockstep per-chunk cross-rank max for the NEXT run_moe (set via
        # ``set_adaptive_launch_tokens``); ``None`` -> full bucket.
        self._active_launch_max_tokens: Optional[int] = None

        # Resolve EP ProcessGroup at construction. Resolving at forward
        # time would be collective on a non-synchronous call stack and
        # deadlock under PP / layer-skip. Construction is globally
        # synchronous across ranks during model build.
        try:
            self._ep_pg = self._resolve_ep_pg()
        except RuntimeError as e:
            # Single-rank tests do not always initialize torch.distributed.
            # The kernel's single-rank degenerate path does not need a PG.
            logger.debug(
                f"[MegaMoECuteDsl] EP PG not resolvable ({e!r}); falling back "
                f"to single-rank degenerate mode at run_moe time."
            )
            self._ep_pg = None
        # Pure-DEP invariant checked at (rank-synchronous) construction: the
        # op's barrier-reset fence runs on the WORLD group, so WORLD != EP
        # would build fine and then fail or hang mid-serving. ValueError on
        # purpose -- the ``except RuntimeError`` above is the single-rank
        # fallback and must not swallow this.
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() != self.ep_size:
            raise ValueError(
                f"MegaMoECuteDsl requires pure DEP (torch.distributed WORLD "
                f"== EP): the forward-time barrier-reset fence runs on the "
                f"WORLD process group. Got WORLD={dist.get_world_size()}, "
                f"EP={self.ep_size}."
            )

        # WEAK ref to the AutoTuner profiling scratch provider: the custom
        # op's process-global cache is the SOLE strong owner, so
        # ``release_megamoe_profiling_scratch()`` reclaims it and this ref
        # goes dead (no per-module teardown). Materialized lazily by the
        # deferred factory in ``_launch_megamoe_kernel``.
        self._profiling_scratch_provider_ref = None
        self._profiling_scratch_kwargs = None
        self._weights_loaded = False
        self._weights_created = False
        self.quant_method = None
        # Per-instance staging cache (key -> {tensor name: tensor}) and
        # last-staged-T tracker; together they implement the always-pad-
        # to-max_T launch contract by refreshing only the rows that
        # changed between calls. ``_last_staged_T`` is keyed by bucket max_T:
        # each bucket's staging buffer has its own row-reset history (a shared
        # scalar would reset the wrong buffer when buckets alternate).
        self._local_staging_cache: Dict[Tuple, Dict[str, torch.Tensor]] = {}
        self._last_staged_T: Dict[int, int] = {}
        # Cache immutable provider regions and dtype views, retaining their owners.
        # Read the cache switch once at construction; hits validate object identity
        # and dtype views also detect storage rebinding.
        self._regions_memo: Dict[int, Tuple] = {}
        self._host_memo_flag = os.environ.get("MEGAMOE_HOST_MEMO", "1") != "0"
        self._dtype_view_memo: Dict[Tuple, Tuple] = {}
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    # ------------------------------------------------------------------
    # Compute axis
    # ------------------------------------------------------------------
    @property
    def compute_slot_count(self) -> int:
        """Return the kernel's single slot axis M = H + S.

        create_weights widens expert_size_per_partition itself; adding S again
        would double-count helpers. Read the live value because ConfigurableMoE
        can assign it after backend construction.
        """
        return int(self.expert_size_per_partition)

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_maxt_buckets(max_num_tokens: int) -> List[int]:
        """Resolve the adaptive ``max_tokens_per_rank`` bucket ladder.

        Every EP rank derives the same architecture-specific ladder (divergence
        breaks the cross-rank NVLink barrier). Rubin adds generation-phase
        rungs through 1024; Blackwell keeps the existing 256/1024/4096 ladder.
        Kept small -- each rung multiplies the compile/capture/memory surface.
        """
        full = int(max_num_tokens)
        if full <= 0:
            full = 4096
        rungs = (
            (32, 64, 128, 192, 256, 384, 512, 1024, 4096)
            if get_sm_version() == 107
            else (256, 1024, 4096)
        )
        cand = {b for b in rungs if b < full}
        cand.add(full)
        return sorted(cand)

    def _select_launch_max_tokens(self, chunk_max_tokens: Optional[int]) -> int:
        """Pick the smallest ladder bucket that fits ``chunk_max_tokens``.

        ``chunk_max_tokens`` is the lockstep ``max(all_rank_num_tokens)`` for
        the chunk (rank-identical) or ``None`` (-> full bucket); input and
        ladder rank-identical => bucket rank-identical, so all ranks launch
        the same compiled kernel against same-sized symmetric buffers.

        Tuning mode with the ``tactic_autotune`` opt-in ALWAYS takes the top
        bucket (profiling scratch and tuned-cache keys are sized/keyed for
        it); both gates are rank-identical so the bucket stays lockstep.
        """
        buckets = self._maxt_buckets
        if self.tactic_autotune and AutoTuner.get().is_tuning_mode:
            return buckets[-1]
        if chunk_max_tokens is None or chunk_max_tokens <= 0:
            return buckets[-1]
        for b in buckets:
            if chunk_max_tokens <= b:
                return b
        # RAISE (rank-identical input -> lockstep failure). Clamping instead
        # would raise only on ranks whose local count exceeds the clamp,
        # desyncing the NVLink barrier and hanging the survivors.
        raise RuntimeError(
            f"MegaMoE-CuteDSL: chunk max_tokens_per_rank={chunk_max_tokens} exceeds "
            f"the top adaptive bucket {buckets[-1]} (per-rank cap = "
            f"min(moe_max_num_tokens, max_num_tokens)). Raise moe_max_num_tokens or "
            f"reduce the per-rank chunk."
        )

    def set_adaptive_launch_tokens(self, chunk_max_tokens: Optional[int]) -> None:
        """Scheduler hook: record the lockstep per-chunk cross-rank max token
        count for the next ``run_moe``.

        MUST be ``max(all_rank_num_tokens)`` for the chunk (rank-identical) so
        all ranks select the same bucket; ``None`` -> full bucket. Consumed
        and reset by the next ``_run_moe``.
        """
        self._active_launch_max_tokens = chunk_max_tokens

    def _supports_load_balancer(self) -> bool:
        # Both static and dynamic EPLB are supported: the four MegaMoE-
        # format derived parameters (``mega_fc{1,2}_weight{,_sf}``) are
        # registered with the load balancer in
        # ``NVFP4MegaMoECuteDslMethod._register_mega_shared_staging``
        # and migrate atomically with the underlying NVFP4 raw weights
        # + scales already handled by the base/grandparent.
        return True

    def validate_configurable_moe(self, moe) -> None:
        """Wrapper-only checks that ``can_implement`` cannot see.

        Topology / slot / ADP eligibility is owned by ``can_implement``.
        This validates ConfigurableMoE attributes populated after the backend
        is constructed: host-side ``comm`` must stay None for FUSED_COMM, and
        top-k / max-token bounds used to size the fused kernel.
        """
        if moe.comm is not None:
            raise ValueError(
                f"MegaMoECuteDsl requires moe.comm is None (FUSED_COMM "
                f"backends must not layer host-side communication on top "
                f"of the fused kernel); got moe.comm={type(moe.comm).__name__}."
            )
        top_k = moe.routing_method.experts_per_token
        # EXPERIMENTAL, pending kernel-side confirmation: raised 13 -> 16 to
        # admit Kimi K3, whose routing is top-16. The original bound was
        # described as matching *external coverage*, i.e. what had been
        # validated, not a kernel capability -- and nothing in the CuTe DSL
        # kernel hardcodes it: num_topk is a sizing parameter there
        # (max_tokens_per_rank * num_topk, (max_tokens, num_topk, hidden)).
        # That makes the widening plausible, NOT proven: shared-memory and
        # register budgets grow with num_topk, and topk_reduce.py may carry
        # fixed unrolls. Accuracy at one shape is evidence, not coverage.
        _MEGAMOE_CUTEDSL_MAX_TOP_K = 16
        if top_k > _MEGAMOE_CUTEDSL_MAX_TOP_K:
            raise ValueError(
                f"MegaMoECuteDsl supports experts_per_token <= "
                f"{_MEGAMOE_CUTEDSL_MAX_TOP_K}; got {top_k}."
            )
        if moe.moe_max_num_tokens <= 0:
            raise ValueError(
                f"MegaMoECuteDsl requires moe_max_num_tokens > 0; got {moe.moe_max_num_tokens}."
            )
        # Dynamic EPLB is intentionally allowed: the quant method
        # registers mega-format derived parameters alongside the raw
        # NVFP4 family so per-slot migration stays byte-consistent.

    # ------------------------------------------------------------------
    # EP process-group resolution (no collective at forward time)
    # ------------------------------------------------------------------
    def _maybe_init_torch_dist_under_mpi(self):
        """Bootstrap a torch.distributed NCCL WORLD group under the MPI orchestrator.

        The MPI orchestrator (trtllm-bench / mpirun) never calls
        ``init_process_group``, but the symmetric-memory rendezvous needs a
        ProcessGroup; for pure DEP a node-local NCCL WORLD suffices. No-op
        under Ray or single-rank.
        """
        if not dist.is_available() or dist.is_initialized():
            return
        from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size

        try:
            world = mpi_world_size()
            rank = mpi_rank()
        except Exception as e:  # not under MPI either -> leave uninitialized
            logger.debug(
                f"[MegaMoECuteDsl] MPI rank query failed ({e!r}); "
                "skipping torch.distributed bootstrap."
            )
            return
        if world <= 1:
            return
        # Check pure DEP BEFORE any collective: under PP/CP hybrids some ranks
        # never construct a MegaMoE module, so the bcast/init below would hang
        # instead of failing loud. Rank-identical inputs -> lockstep raise.
        if world != self.ep_size:
            raise ValueError(
                f"MegaMoECuteDsl requires pure DEP (MPI WORLD == EP) to "
                f"bootstrap torch.distributed: got MPI world={world}, "
                f"EP={self.ep_size}."
            )

        # Rank 0 draws a free port (a fixed one collides across same-node
        # jobs). The bcast is UNCONDITIONAL and rank 0 alone decides
        # (honoring ITS pre-set MASTER_* env): gating the collective on
        # per-rank env presence would desync the communicator when the vars
        # are exported on only a subset of ranks. No bind-race retry: a sound
        # one needs cross-rank failure agreement, and the close()->bind()
        # window is negligible next to the fixed-port collision this replaces.
        def _pick_rendezvous():
            addr = os.environ.get("MASTER_ADDR")
            port = os.environ.get("MASTER_PORT")
            if addr and port:
                return (addr, port)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            try:
                addr = socket.gethostbyname(socket.gethostname())
            except OSError:
                # Unresolvable hostname (minimal containers / broken DNS):
                # this bootstrap is documented single-node, loopback works.
                addr = "127.0.0.1"
            return (addr, str(port))

        host, port = mpi_comm().bcast(_pick_rendezvous() if rank == 0 else None, root=0)
        os.environ["MASTER_ADDR"] = str(host)
        os.environ["MASTER_PORT"] = str(port)
        # The executor already bound this rank's device before the model was
        # built (base_worker: global_mpi_rank() % device_count). Re-deriving it
        # here could pick a different GPU (e.g. a disagg ctx/gen pair sharing a
        # node) and would move only the layers built after this point, so bind
        # NCCL to the current device instead of selecting one.
        device_id = None
        if torch.cuda.is_available():
            device_id = torch.device("cuda", torch.cuda.current_device())
        logger.info(
            f"[MegaMoECuteDsl] torch.distributed not initialized under MPI; "
            f"bootstrapping NCCL WORLD group (rank={rank}/{world}, "
            f"device={device_id}, "
            f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}) for the "
            f"EP rendezvous."
        )
        dist.init_process_group(
            backend="cuda:nccl,cpu:gloo",
            rank=rank,
            world_size=world,
            device_id=device_id,
        )

    def _resolve_ep_pg(self):
        """Return the torch.distributed ProcessGroup for the EP sub-world.

        Mirrors :meth:`MegaMoEDeepGemm._resolve_ep_pg` so the two MegaMoE
        backends share the same fallback chain.
        """
        self._maybe_init_torch_dist_under_mpi()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "MegaMoECuteDsl requires torch.distributed to be initialized "
                "before module construction (mpirun or Ray) for multi-rank "
                "execution."
            )
        try:
            pg = self.mapping.moe_ep_group_pg
            log_fn = logger.info if self.layer_idx == 0 else logger.debug
            log_fn(
                f"[MegaMoECuteDsl] layer={self.layer_idx} using "
                f"mapping.moe_ep_group_pg (DeviceMesh path)."
            )
            return pg
        except (NotImplementedError, AttributeError):
            pass
        world_size = dist.get_world_size()
        if self.ep_size == world_size:
            log_fn = logger.info if self.layer_idx == 0 else logger.debug
            log_fn(
                f"[MegaMoECuteDsl] layer={self.layer_idx} using dist.group.WORLD "
                f"(EP == world_size == {world_size})."
            )
            return dist.group.WORLD
        raise RuntimeError(
            f"MegaMoECuteDsl: cannot resolve EP ProcessGroup. The current "
            f"mapping does not expose ``moe_ep_group_pg`` and EP "
            f"({self.ep_size}) is a strict subset of world ({world_size})."
        )

    # ------------------------------------------------------------------
    # Weight lifecycle
    # ------------------------------------------------------------------
    def _get_quant_method(self):
        if self.quant_config is None or not self.quant_config.layer_quant_mode.has_nvfp4():
            raise NotImplementedError("MegaMoECuteDsl supports NVFP4 quantization only.")
        return NVFP4MegaMoECuteDslMethod()

    def create_weights(self):
        """Build-time weight + symmetric-buffer allocation.

        Order:
          0. Widen the slot axis to ``M = H + S`` when helper slots are
             configured (MoE rebalance). MUST precede steps 1 and 2-3:
             step 1 sizes the symmetric workspace from
             ``compute_slot_count`` and step 3 sizes every weight
             Parameter from ``expert_size_per_partition``.
          1. Allocate symmetric-memory provider for multi-rank EP
             (collective rendezvous; MUST run at build time -- not from
             ``run_moe`` -- because forward time may be inside CUDA
             graph capture or non-lockstep PP/layer-skip).
          2. Resolve quantization method.
          3. Delegate parameter registration to the quant method.
          4. Flip ``_weights_created``.

        The symm provider is shared across MoE layers with the same
        (group, layout) via the module-scope cache in
        ``cute_dsl_megamoe_custom_op.py``; only the first layer that
        reaches this point pays the rendezvous cost, and every EP rank
        hits this code in lockstep because ``ConfigurableMoE`` calls
        ``create_weights`` on every rank after backend construction.
        """
        if self._weights_created:
            return
        # ConfigurableMoE has synchronized resident geometry H before this call.
        # Widen only the backend to M = H + S; the wrapper and checkpoint expert IDs
        # retain their logical H-sized axis. Repeated allocation must retain the same
        # geometry rather than adding S again.
        helper_slots = int(self._rebalance_slots_active)
        if helper_slots > 0:
            if self._rebalance_home_experts is None:
                self._rebalance_home_experts = int(self.expert_size_per_partition)
                self.expert_size_per_partition = self._rebalance_home_experts + helper_slots
                log_fn = logger.info if self.layer_idx in (0, None) else logger.debug
                log_fn(
                    f"[MegaMoECuteDsl] layer={self.layer_idx} MoE rebalance "
                    f"widened the slot axis: H={self._rebalance_home_experts} "
                    f"resident + S={helper_slots} helper "
                    f"= M={self.expert_size_per_partition} "
                    f"(ep_size={self.ep_size}, "
                    f"global slot space P = E + S*EP = "
                    f"{self.expert_size_per_partition * self.ep_size}). "
                    f"All seven live weight planes are allocated M rows tall."
                )
            elif int(self.expert_size_per_partition) != self._rebalance_home_experts + helper_slots:
                # Someone reset the axis between calls (a re-run of the
                # ``_BACKEND_SYNC_ATTRS`` loop, or S changed). Re-widening
                # would be guesswork and leaving it would allocate H-sized
                # weights for an M-sized kernel, so refuse.
                raise RuntimeError(
                    f"MegaMoECuteDsl: expert_size_per_partition was reset "
                    f"between create_weights() calls -- expected "
                    f"M={self._rebalance_home_experts + helper_slots} "
                    f"(H={self._rebalance_home_experts} + S={helper_slots}), "
                    f"found {int(self.expert_size_per_partition)}. The MoE "
                    f"rebalance slot axis must not be re-synced after "
                    f"create_weights has widened it."
                )
        # Step 1: build-time symmetric memory allocation (multi-rank only).
        # Single-rank degenerate uses local CUDA tensors and skips here.
        if self.ep_size > 1:
            self._alloc_symm_provider()
        # Allocate the H + S arena before the quant method creates its Parameters.
        # Weight loading binds aliases into this allocation before copy can publish.
        if self._rebalance_slots_active > 0 and getattr(self, "_rebalance_arena", None) is None:
            # The hierarchical multicast arena: one multicast group per HALO
            # level, rather than one per (source, destination) pair.
            from .rebalance_live_arena_v2 import (
                allocate_rebalance_arena_v2 as allocate_rebalance_arena,
            )

            self._rebalance_arena = allocate_rebalance_arena(
                home_experts=int(self._rebalance_home_experts),
                helper_slots=int(self._rebalance_slots_active),
                hidden_size=int(self.hidden_size),
                intermediate_size=int(self.intermediate_size_per_partition),
                mapping=self.mapping,
                device=int(self.mapping.local_rank),
                layer_idx=self.layer_idx,
            )
        # Step 2-3: quant method registers all NVFP4 + MegaMoE-format params.
        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)
        # Step 4.
        self._weights_created = True

    @property
    def _symm_provider(self):
        """Full-bucket symmetric provider, or ``None`` (pre-``create_weights``
        or single-rank) -- callers use this as the provider-available guard.
        """
        return self._symm_providers.get(self._maxt_buckets[-1])

    def _use_in_kernel_fc2_reduce(self, max_tokens_per_rank: int) -> bool:
        if get_sm_version() != 107 or self.combine_format != "bf16":
            return False
        from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
            _sm107_use_in_kernel_fc2_reduce,
        )

        return _sm107_use_in_kernel_fc2_reduce(max_tokens_per_rank, self.apply_topk_in_fc1)

    def _alloc_symm_provider(self):
        """Build-time symmetric provider allocation. See ``create_weights``.

        One provider per adaptive bucket. Each ``get_megamoe_symm_provider``
        call is a COLLECTIVE rendezvous, so the loop must iterate identical
        buckets in identical order on every EP rank (guaranteed: the ladder is
        a pure function of rank-identical inputs); the module-scope cache
        makes only the first layer pay. Raises MegaMoeCuteDslUnavailable when
        no ProcessGroup is available (hard error for multi-rank).
        """
        from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
            get_megamoe_symm_provider,
            query_megamoe_kernel_properties,
        )

        if self._ep_pg is None:
            raise MegaMoeCuteDslUnavailable(
                "MegaMoECuteDsl multi-rank requires a torch.distributed EP "
                "ProcessGroup. Use Ray / DeviceMesh (mapping.moe_ep_group_pg) "
                "or initialize torch.distributed before model build."
            )
        top_k = self.routing_method.experts_per_token
        full_T = self._maxt_buckets[-1]
        self._profiling_scratch_provider_ref = None
        for max_T in self._maxt_buckets:
            (
                shared_workspace_bytes,
                supports_direct_inputs,
            ) = query_megamoe_kernel_properties(
                world_size=self.ep_size,
                num_topk=top_k,
                num_experts_per_rank=self.compute_slot_count,
                hidden_size=self.hidden_size,
                intermediate_size_per_partition=int(self.intermediate_size_per_partition),
                expand_intermediate_size_per_partition=int(
                    self.expand_intermediate_size_per_partition
                ),
                max_tokens_per_rank=max_T,
                apply_topk_in_fc1=bool(self.apply_topk_in_fc1),
                gate_up_clamp=self.act_clamp,
                swiglu_alpha=(
                    self.act_alpha if self.activation_type is ActivationType.SwigluBias else None
                ),
                swiglu_beta=(
                    self.act_beta if self.activation_type is ActivationType.SwigluBias else None
                ),
                situ_beta=(self.act_alpha if self.activation_type is ActivationType.SiTu else None),
                situ_linear_beta=(
                    self.act_beta if self.activation_type is ActivationType.SiTu else None
                ),
                in_kernel_fc2_reduce=self._use_in_kernel_fc2_reduce(max_T),
                combine_format=self.combine_format,
            )
            if supports_direct_inputs:
                self._genphase_direct_input_buckets.add(max_T)
            provider = get_megamoe_symm_provider(
                process_group=self._ep_pg,
                world_size=self.ep_size,
                rank=self.ep_rank,
                hidden_size=self.hidden_size,
                max_tokens_per_rank=max_T,
                num_topk=top_k,
                output_dtype=self.dtype or torch.bfloat16,
                combine_format=self.combine_format,
                shared_workspace_bytes=shared_workspace_bytes,
            )
            self._symm_providers[max_T] = provider
            # AutoTuner profiling scratch: record only the request kwargs
            # (full bucket only, matching tuning mode's forced top bucket);
            # the multi-GiB allocation itself is deferred to the op's
            # profiling pre-hook via the factory in ``_launch_megamoe_kernel``,
            # so a process that never tunes never pays it.
            if max_T == full_T and self.tactic_autotune:
                self._profiling_scratch_kwargs = dict(
                    process_group=self._ep_pg,
                    world_size=self.ep_size,
                    rank=self.ep_rank,
                    hidden_size=self.hidden_size,
                    max_tokens_per_rank=max_T,
                    num_topk=top_k,
                    output_dtype=self.dtype or torch.bfloat16,
                    combine_format=self.combine_format,
                    shared_workspace_bytes=shared_workspace_bytes,
                )
        if len(self._maxt_buckets) > 1 and self.ep_rank == 0:
            logger.info(
                "[MegaMoECuteDsl] adaptive max_tokens_per_rank buckets=%s "
                "(one symmetric provider + kernel per bucket; per-launch bucket "
                "= smallest >= max(all_rank_num_tokens))",
                self._maxt_buckets,
            )

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False) -> None:
        if self.quant_method is None:
            self.create_weights()
        # Match CutlassFusedMoE.load_weights: callers pass ``[weights_dict]``.
        # ``FusedMoEMethodBase.load_expert_weights_to_dst`` treats the inner
        # value as a Dict (``weights[f"{expert_id}.w1.weight"]``), so unwrap
        # the single-element list before forwarding. Forward
        # ``weight_loading_mode`` explicitly because the base signature is
        # ``(module, weights, weight_loading_mode, allow_partial_loading=False)``;
        # passing ``allow_partial_loading`` (a bool) as the 3rd positional arg
        # would be interpreted as the mode and trip ``NotImplementedError``.
        # Same ``-O``-safe pattern as the constructor: validate caller-
        # supplied input with an explicit raise so the check is not
        # silently stripped in optimised builds.
        if len(weights) != 1:
            raise ValueError(
                "MegaMoECuteDsl.load_weights expects a single-element list, "
                f"got {len(weights)} entries."
            )
        weights = weights[0]

        self.quant_method.load_weights(
            self, weights, self.weight_loading_mode, allow_partial_loading=allow_partial_loading
        )

    def post_load_weights(self) -> None:
        if self.quant_method is None:
            self.create_weights()
        self.transform_weights()
        self.cache_derived_state()
        self._build_rebalance_scheduler_group()

    def is_rebalance_active(self) -> bool:
        """Whether this forward produces a helper-weight generation."""
        if self._rebalance_slots_active <= 0 or not bool(
            getattr(self, "_rebalance_arm_open", True)
        ):
            return False
        # The executor marks the whole autotune phase as warmup. Explicit
        # ON tactic profiling must still run its real HALO/TMA producer.
        # Internal bucket priming remains disarmed by _rebalance_arm_open.
        if AutoTuner.get().is_tuning_mode:
            return bool(self.tactic_autotune)
        return not self._rebalance_warmup

    def _build_rebalance_scheduler_group(self) -> None:
        """Build the collective producer once, after live weights are bound.

        All EP ranks must enter in the same order. Building lazily during forward
        could leave peers waiting if another rank fails before joining.
        """
        if int(self._rebalance_slots_active) <= 0:
            return
        if self._rebalance_scheduler_group is not None:
            return
        from .rebalance_slot_scheduler_v2 import build_rebalance_slot_scheduler_group_v2 as _build

        if self.tactic_autotune:
            # The synthetic profiling workload visits every physical slot.
            # A real HALO plan may leave some helpers inactive, so initialize
            # their seven planes once with valid resident bytes before any
            # producer or consumer can use this arena. Real TMA copies still
            # replace active helpers and publish the only READY generation.
            arena = self._rebalance_arena
            home = int(arena.home_experts)
            with torch.no_grad():
                # Storage aliases are contiguous uint8/float32, avoiding
                # arithmetic/copy dispatch on PyTorch's packed-FP4 dtype.
                for plane in arena.tekit_alias_views:
                    for helper in range(int(arena.helper_slots)):
                        plane[home + helper].copy_(plane[helper % home], non_blocking=True)
            self._rebalance_autotune_helpers_initialized = True

        self._rebalance_scheduler_group = _build(self)

    # Staged-hook guards: ConfigurableMoE delegates these straight to the
    # backend, and the base MoE implementations dereference
    # ``self.quant_method`` unguarded.
    def transform_weights(self) -> None:
        if self.quant_method is None:
            self.create_weights()
        super().transform_weights()

    def cache_derived_state(self) -> None:
        # A weights-removed reader runs materialize -> this hook and never
        # ``post_load_weights``, so the guard cannot move there: this is the
        # only place that pass can bind the quant method.
        if self.quant_method is None:
            self.create_weights()
        super().cache_derived_state()
        # ConfigurableMoE delegates this hook on both ordinary and staged loading paths.
        # Build once, after plane aliases are final.
        self._build_rebalance_scheduler_group()

    def pre_reload_weights(self) -> None:
        raise NotImplementedError(
            "MegaMoE-CuteDSL does not support hot weight reloading; its "
            "source weights are released after initial packing."
        )

    def _build_weight_view(self) -> MegaMoECuteDslWeightView:
        """Bundle the MegaMoE-format weight tensors registered by the
        quant method. ``run_moe`` calls this once per chunk so the
        kernel sees the latest dynamic-EPLB migration outcome (once
        that path lands; currently the slots are static).
        """
        _view = MegaMoECuteDslWeightView(
            fc1_weight=self.mega_fc1_weight,
            fc1_weight_sf=self.mega_fc1_weight_sf,
            fc2_weight=self.mega_fc2_weight,
            fc2_weight_sf=self.mega_fc2_weight_sf,
            fc31_alpha=self.fc31_alpha,
            fc2_alpha=self.fc2_alpha,
            fc1_norm_const=self.fc1_norm_const,
        )
        return _view

    # ------------------------------------------------------------------
    # MoE-contract methods
    # ------------------------------------------------------------------
    def quantize_input(
        self,
        x: Union[torch.Tensor, "Fp4QuantizedTensor"],
        *,
        post_quant_comm: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """BF16 -> NVFP4 packed activation + plain K-major FP8 SF.

        Reuses ``torch.ops.trtllm.fp4_quantize`` with
        ``is_sf_swizzled=False`` so the SF tensor lands in the plain
        K-major layout expected by the MegaMoE kernel.
        ``self.fc31_input_scale`` is the per-tensor FP32 input scale
        registered by the quantization method's ``create_weights``; the
        value defaults to 1.0 until the checkpoint loader sets it.

        Empty input (``x.shape[0] == 0``) short-circuits to empty NVFP4
        + empty SF without launching a quantization kernel, so the
        ``FusedCommMoEScheduler`` can call ``quantize_input`` uniformly
        for zero-token chunks.
        """
        del post_quant_comm  # MegaMoE owns dispatch / combine in-kernel.
        del kwargs
        if isinstance(x, Fp4QuantizedTensor):
            raise NotImplementedError(
                "MegaMoECuteDsl.quantize_input expects BF16 activation; "
                "pre-quantized Fp4QuantizedTensor is not yet supported."
            )

        hidden = x.shape[1]
        sf_cols = megamoe_activation_sf_bytes_per_row(hidden)
        x_bf16 = x.to(torch.bfloat16).contiguous()
        if x_bf16.shape[0] == 0:
            empty_x = torch.empty((0, hidden // 2), dtype=torch.uint8, device=x_bf16.device)
            empty_sf = torch.empty((0, sf_cols), dtype=torch.uint8, device=x_bf16.device)
            return empty_x, empty_sf
        # Budget the same eight SMs as the concurrent HALO-Q/TMA stream.
        # The default overload remains unchanged for load balance OFF.
        if self._quantize_reserved_sms:
            x_fp4, x_sf = torch.ops.trtllm.fp4_quantize.sm_budget(
                x_bf16,
                self.fc31_input_scale,
                16,
                False,
                False,
                reserved_sms=self._quantize_reserved_sms,
            )
        else:
            x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(
                x_bf16,
                self.fc31_input_scale,
                16,  # scaling_vector_size == Nvfp4BlockSize
                False,  # sf_use_ue8m0
                False,  # is_sf_swizzled - MegaMoE expects plain K-major
            )
        # ``fp4_quantize(is_sf_swizzled=False)`` returns LINEAR layout
        # ``(rows, ceil(hidden/16))`` with no column pad. The kernel TMA
        # load needs ``pad_up(ceil_div(hidden, 16), 4)`` bytes per row
        # (== ``megamoe_activation_sf_bytes_per_row``), so
        # 32-aligned-but-not-64-aligned hidden sizes (1568, 1632, 2080)
        # come back 2 bytes short; pad the tail before returning.
        raw_cols = ceil_div(hidden, 16)
        x_sf_raw = x_sf.view(x_bf16.shape[0], raw_cols)
        if sf_cols == raw_cols:
            return x_fp4, x_sf_raw
        padded_sf = torch.zeros((x_bf16.shape[0], sf_cols), dtype=torch.uint8, device=x_bf16.device)
        padded_sf[:, :raw_cols] = x_sf_raw
        return x_fp4, padded_sf

    def _wait_rebalance_routes(self) -> None:
        """Order route consumers after HALO-Q while leaving setup overlapped."""
        if self._rebalance_plan_ran:
            self._rebalance_scheduler_group.wait_for_routes()

    def run_moe(self, *a, **k):
        _nvtx("mm/run_moe")
        try:
            return self._nvtxwrap_run_moe(*a, **k)
        finally:
            _nvtx_end()

    def _nvtxwrap_run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run the fused MegaMoE CuteDSL kernel on pre-quantized inputs.

        Keeps the scheduler's ``int32`` token-selected experts on SM107 and
        delegates the staging + kernel launch to :meth:`_run_moe`, which
        returns the reduced ``(T, hidden)`` output. Other architectures keep
        the existing ``int64`` kernel ABI.
        """
        del workspace  # The symmetric buffer is this backend's own workspace.
        x = ctx.x
        token_selected_experts = ctx.token_selected_experts
        token_final_scales = ctx.token_final_scales
        x_sf = ctx.x_sf
        output_dtype = ctx.output_dtype
        if output_dtype is None:
            output_dtype = self.dtype or torch.bfloat16
        if x_sf is None:
            raise ValueError("MegaMoECuteDsl requires x_sf from quantize_input")

        # Surface a missing multi-rank symm provider BEFORE the weights
        # guard so callers can distinguish "no provider" from "weights
        # not loaded" in tests and runtime fallbacks.
        if self.ep_size > 1 and getattr(self, "_symm_provider", None) is None:
            raise MegaMoeCuteDslUnavailable(
                "MegaMoECuteDsl multi-rank run_moe requires the cuMem "
                "symmetric-memory provider, but no provider was allocated "
                "for this backend instance. The provider rendezvous runs at "
                "create_weights() time and needs a live torch.distributed "
                "EP ProcessGroup; spawn the workload via Ray / DeviceMesh "
                "or mpirun so the rendezvous can complete."
            )

        if not self._weights_created or self.quant_method is None:
            raise RuntimeError(
                "MegaMoECuteDsl.run_moe called before create_weights / "
                "load_weights / post_load_weights finished. The MegaMoE-"
                "format weight tensors are missing."
            )

        weight_view = self._build_weight_view()
        num_tokens = int(x.shape[0])
        hidden = self.hidden_size
        top_k = int(token_selected_experts.shape[-1])
        device = x.device

        topk_weights_f32 = token_final_scales.to(torch.float32).contiguous()
        # Rubin borrows the contiguous int32 scheduler output without a GPU
        # operation. A conversion on another architecture is itself a reader.
        if (
            token_selected_experts.dtype != self._topk_idx_dtype
            or not token_selected_experts.is_contiguous()
        ):
            self._wait_rebalance_routes()
        topk_idx = token_selected_experts.to(self._topk_idx_dtype).contiguous()

        return self._run_moe(
            x=x,
            x_sf=x_sf,
            topk_idx=topk_idx,
            topk_weights=topk_weights_f32,
            weight_view=weight_view,
            num_tokens=num_tokens,
            top_k=top_k,
            hidden=hidden,
            device=device,
            output_dtype=output_dtype,
        )

    def _ensure_local_staging(self, *, top_k: int, hidden: int, device, output_dtype, max_T: int):
        """Allocate (and cache) the per-instance local staging tensors.

        Always allocates ``topk_idx`` (the kernel reads it as a local-only
        buffer in BOTH topologies; peers never call
        ``peer_rank_ptr_mapper.map`` on it). The user-domain entries --
        ``activation`` / ``activation_sf`` / ``topk_weights`` /
        ``combine_output`` / ``shared_workspace`` -- are allocated only
        for ``ep_size == 1``; multi-rank pulls them from the symmetric
        provider's regions instead.

        All staging tensors are sized to ``max_T`` (the per-launch adaptive
        bucket) along dim 0 so the kernel's constexpr ``num_tokens`` matches the
        buffer-time ``max_tokens_per_rank``. Diverging the two would make
        ``_dispatch_prep`` round 3 (``MAX_SLOT_C = num_tokens * num_topk``
        in dispatch_kernel.py) write per-(expert, rank) advertise cards
        at the wrong stride relative to the symm allocation
        (``max_tokens_per_rank * num_topk`` in megamoe_kernel.py),
        silently corrupting multi-rank metadata. The cache is keyed by
        ``max_T`` so each bucket owns a distinct staging set.
        """
        max_T = int(max_T)
        cache_key = (max_T, top_k, hidden, str(device), output_dtype)
        cached = self._local_staging_cache
        if cache_key in cached:
            return cached[cache_key]

        # ``topk_idx`` defaults to -1 so dispatch_prep skips padded tail
        # rows (``if expert_id >= Int32(0):`` in dispatch_kernel.py).
        staging = {
            "topk_idx": torch.full(
                (max_T, top_k),
                -1,
                dtype=self._topk_idx_dtype,
                device=device,
            ),
        }
        if self.ep_size == 1:
            sf_bytes_per_row = megamoe_activation_sf_bytes_per_row(hidden)
            # ``topk_weights`` defaults to 0 so stale combine rows
            # contribute nothing. Multi-rank uses the symm provider's
            # topk_weights region instead.
            staging["topk_weights"] = torch.zeros(
                (max_T, top_k), dtype=torch.float32, device=device
            )
            staging["activation"] = torch.empty(
                (max_T, hidden // 2), dtype=torch.uint8, device=device
            )
            staging["activation_sf"] = torch.empty(
                (max_T, sf_bytes_per_row), dtype=torch.uint8, device=device
            )
            # Always 1: the kernel collapses the top-k axis in-op (TopkReduce
            # form-A / REDG form-B); the op aliases this as the 2D output.
            combine_k = 1
            staging["combine_output"] = torch.empty(
                (max_T, combine_k, hidden),
                dtype=torch.bfloat16,
                device=device,
            )
            # Shared-workspace probe lives behind ``IS_MEGAMOE_OP_AVAILABLE``
            # in cute_dsl_megamoe_custom_op so the import stays lazy.
            from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
                query_megamoe_kernel_properties,
            )

            shared_bytes, supports_direct_inputs = query_megamoe_kernel_properties(
                world_size=1,
                num_topk=top_k,
                num_experts_per_rank=self.compute_slot_count,
                hidden_size=hidden,
                intermediate_size_per_partition=int(self.intermediate_size_per_partition),
                expand_intermediate_size_per_partition=int(
                    self.expand_intermediate_size_per_partition
                ),
                max_tokens_per_rank=max_T,
                apply_topk_in_fc1=bool(self.apply_topk_in_fc1),
                gate_up_clamp=self.act_clamp,
                swiglu_alpha=(
                    self.act_alpha if self.activation_type is ActivationType.SwigluBias else None
                ),
                swiglu_beta=(
                    self.act_beta if self.activation_type is ActivationType.SwigluBias else None
                ),
                situ_beta=(self.act_alpha if self.activation_type is ActivationType.SiTu else None),
                situ_linear_beta=(
                    self.act_beta if self.activation_type is ActivationType.SiTu else None
                ),
                in_kernel_fc2_reduce=self._use_in_kernel_fc2_reduce(max_T),
                combine_format=self.combine_format,
            )
            if supports_direct_inputs:
                self._genphase_direct_input_buckets.add(max_T)
            # zeros (not empty): the leading counter prefix is an atomic-add
            # target the kernel only tail-resets for the NEXT launch, so the
            # FIRST launch needs it pre-zeroed (one-time; cached per bucket).
            staging["shared_workspace"] = torch.zeros(
                shared_bytes, dtype=torch.uint8, device=device
            )
        cached[cache_key] = staging
        return staging

    # MEGAMOE_HOST_MEMO=0 disables view caching for comparison or diagnosis.
    def _host_memo_on(self) -> bool:
        return self._host_memo_flag

    def _memo_regions(self, provider):
        """Cache views of an immutable provider, retaining its identity and lifetime."""
        if not self._host_memo_on():
            return provider.get_regions()
        key = id(provider)
        hit = self._regions_memo.get(key)
        # Retain the provider and check identity to prevent object-ID reuse.
        if hit is not None and hit[0] is provider:
            return hit[1]
        regions = provider.get_regions()
        self._regions_memo[key] = (provider, regions)
        return regions

    def _memo_dtype_view(self, t: torch.Tensor, caster) -> torch.Tensor:
        """Cache dtype views while retaining the source tensor.

        In-place helper-weight copies remain visible through these aliases.
        Storage rebinding must invalidate the cached view even if identity matches.
        """
        if not self._host_memo_on():
            return caster(t)
        key = (id(t), caster)
        hit = self._dtype_view_memo.get(key)
        # Check identity and data_ptr: assigning Parameter.data can replace storage
        # without changing the object. These dtype views preserve shape and stride.
        if hit is not None and hit[0] is t and hit[1].data_ptr() == t.data_ptr():
            return hit[1]
        v = caster(t)
        self._dtype_view_memo[key] = (t, v)
        return v

    def _acquire_buffers(
        self, *, top_k: int, hidden: int, device, output_dtype, launch_max_T: int
    ) -> _MegaMoeBuffers:
        """Resolve the kernel's input/output buffers for the ``launch_max_T``
        adaptive bucket.

        This is the ONLY structural branch between single-rank and multi-
        rank execution; the source of activation / activation_sf /
        topk_weights / combine_output / shared_workspace differs per the
        :class:`_MegaMoeBuffers` contract. ``topk_idx_local`` always lives
        in plain CUDA memory. ``launch_max_T`` selects both the local staging
        set and (multi-rank) the symmetric provider, which MUST share the same
        ``max_T`` as the kernel's compile-time ``max_tokens_per_rank``.
        """
        staging = self._ensure_local_staging(
            top_k=top_k, hidden=hidden, device=device, output_dtype=output_dtype, max_T=launch_max_T
        )
        if self.ep_size == 1:
            return _MegaMoeBuffers(
                activation=staging["activation"],
                activation_sf=staging["activation_sf"],
                topk_weights=staging["topk_weights"],
                combine_output=staging["combine_output"],
                shared_workspace=staging["shared_workspace"],
                peer_offsets=[0],
                topk_idx_local=staging["topk_idx"],
            )
        # Multi-rank: the provider must have been rendezvous'd at
        # build time (``create_weights``) -- doing it at forward time
        # would violate the build-time collective rule and deadlock
        # under PP / layer-skip.
        if self._symm_provider is None:
            raise MegaMoeCuteDslUnavailable(
                f"MegaMoECuteDsl multi-rank (ep_size={self.ep_size}) "
                f"requires a symmetric-memory provider built at "
                f"create_weights time. self._symm_provider is None -- "
                f"check that the EP ProcessGroup was resolvable when the "
                f"backend was constructed (mapping.moe_ep_group_pg or a "
                f"named dist.new_group), or that "
                f"model_config.skip_create_weights_in_init was not set "
                f"without a follow-up create_weights() call."
            )
        # ``launch_max_T`` comes from ``_select_launch_max_tokens``, so it is
        # always a ladder member.
        provider = self._symm_providers[launch_max_T]
        if provider.num_topk != top_k:
            raise MegaMoeCuteDslUnavailable(
                f"MegaMoECuteDsl symm provider was built for top_k="
                f"{provider.num_topk} but run_moe called with "
                f"top_k={top_k}; recreate the backend."
            )
        regions = self._memo_regions(provider)
        return _MegaMoeBuffers(
            activation=regions.activation,
            activation_sf=regions.activation_sf,
            topk_weights=regions.topk_weights,
            combine_output=regions.combine_output,
            shared_workspace=regions.shared_workspace,
            peer_offsets=regions.peer_offsets,
            topk_idx_local=staging["topk_idx"],
        )

    def _stage_inputs(self, *a, **k):
        _nvtx("mm/stage_inputs")
        try:
            return self._nvtxwrap__stage_inputs(*a, **k)
        finally:
            _nvtx_end()

    def _nvtxwrap__stage_inputs(
        self,
        *,
        bufs: _MegaMoeBuffers,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens: int,
        top_k: int,
        stage_activation: bool,
        stage_routes: bool = True,
    ) -> None:
        """Copy live rows of the user-domain inputs into the kernel's
        pre-allocated buffers and refresh the padded tail.

        Same code for single-rank (writes land in local CUDA) and
        multi-rank (writes land in symmetric heap regions visible to
        peers). The buffer source is selected upstream by
        :meth:`_acquire_buffers`.

        Tail policy:

          * ``topk_idx_local``: always ``-1`` outside live rows.
            Allocated as ``-1`` once in ``_ensure_local_staging``;
            only the rows we previously wrote (``[num_tokens,
            last_T)``) need resetting. New tail rows
            ``[last_T, max_T)`` already hold ``-1`` from prior calls.
          * ``topk_weights``: always ``0.0`` outside live rows. The
            combine kernel writes one cell per
            ``(token, k in [0, top_k))`` regardless of the
            ``topk_idx == -1`` mask, so a stale non-zero weight in
            the tail could corrupt the combine reduction (especially
            on peer ranks via NVLink). One cheap zero kernel covers
            it.
        """
        max_T = bufs.topk_idx_local.shape[0]
        last_T = self._last_staged_T.get(max_T)
        if stage_routes and last_T is not None and last_T > num_tokens:
            bufs.topk_idx_local[num_tokens:last_T].fill_(-1)
        if num_tokens > 0:
            if stage_activation:
                bufs.activation[:num_tokens].copy_(x.view(torch.uint8), non_blocking=True)
                bufs.activation_sf[:num_tokens].copy_(x_sf.view(torch.uint8), non_blocking=True)
            bufs.topk_weights[:num_tokens, :top_k].copy_(topk_weights, non_blocking=True)
        if num_tokens < max_T:
            bufs.topk_weights[num_tokens:max_T, :top_k].zero_()
        if stage_routes:
            if num_tokens > 0:
                self._wait_rebalance_routes()
                bufs.topk_idx_local[:num_tokens].copy_(topk_idx, non_blocking=True)
            self._last_staged_T[max_T] = num_tokens

    def _launch_megamoe_kernel(
        self,
        *,
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        weight_view: MegaMoECuteDslWeightView,
        combine_output: torch.Tensor,
        shared_workspace: torch.Tensor,
        world_size: int,
        local_rank: int,
        top_k: int,
        hidden: int,
        peer_offsets: List[int],
        num_tokens: int,
        output_dtype: torch.dtype,
        launch_max_T: int,
    ) -> torch.Tensor:
        """Launch the fused MegaMoE CuteDSL kernel and return its reduced output.

        Single-rank and multi-rank reach this point with identical kernel
        inputs; only the source of the staged buffers differs (decided
        upstream by :meth:`_acquire_buffers`). NVFP4 / FP8-SF dtype views
        happen through the module-level :func:`_as_nvfp4` /
        :func:`_as_fp8_sf` helpers (the kernel rejects raw uint8 byte
        tensors).
        """
        # Hand the symmetric profiling scratch (or a deferred factory) to the
        # op for tuning-mode profiling launches; cleared in the ``finally`` so
        # a later same-process call for a different shape never sees a stale
        # scratch (ownership: see ``_profiling_scratch_provider_ref``).
        from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
            set_active_megamoe_profiling_scratch,
            set_active_megamoe_profiling_scratch_factory,
        )

        scratch_provider = (
            self._profiling_scratch_provider_ref()
            if self._profiling_scratch_provider_ref is not None
            else None
        )
        scratch_factory = None
        if (
            scratch_provider is None
            and self._profiling_scratch_kwargs is not None
            and world_size > 1
        ):
            if AutoTuner.get().is_tuning_mode:
                # Tuning but no live provider (lazy-only, or an earlier
                # engine's release freed it): hand a DEFERRED factory -- a
                # cache-hitting tuning forward (e.g. the spec-decode draft)
                # must never allocate the multi-GiB scratch. It runs in the
                # op's profiling pre-hook (where the collective rendezvous is
                # lockstep-safe) and refreshes this module's weakref.
                from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
                    get_megamoe_profiling_scratch,
                )

                def _deferred_scratch_factory(_self=self, _get=get_megamoe_profiling_scratch):
                    provider = _get(**_self._profiling_scratch_kwargs)
                    _self._profiling_scratch_provider_ref = weakref.ref(provider)
                    return provider.get_regions()

                scratch_factory = _deferred_scratch_factory
        in_kernel_fc2_reduce = self._use_in_kernel_fc2_reduce(launch_max_T)
        if in_kernel_fc2_reduce and num_tokens > 0:
            combine_output[:num_tokens].zero_()

        # Do not memoize scratch-provider regions: a strong reference would keep
        # its multi-GiB allocation alive after profiling scratch is released.
        set_active_megamoe_profiling_scratch(
            scratch_provider.get_regions()
            if (scratch_provider is not None and world_size > 1)
            else None
        )
        set_active_megamoe_profiling_scratch_factory(scratch_factory)
        # READY generation comes from the producer, never a per-launch local counter.
        # All EP ranks must agree; helper weights and READY flags refer to one arena.
        # The optional injected transport must publish that same generation.
        helper_ready_flags = None
        helper_ready_generation = 0
        # A rank-identical bypass retains M-shaped storage but compiles out the
        # helper READY gate by passing S=0; no token may target a helper in this mode.
        _bypass_S_this_launch = not self.is_rebalance_active()
        _S_active = 0 if _bypass_S_this_launch else int(self._rebalance_slots_active)
        # Record the actual helper-slot ABI for verification.
        self._last_S_to_kernel = _S_active
        if _S_active > 0:
            arena = getattr(self, "_rebalance_arena", None)
            if arena is None:
                raise RuntimeError(
                    "MegaMoECuteDsl: rebalance helper slots are live "
                    f"(S={self._rebalance_slots_active}) but no arena is bound, "
                    "so the READY terminal flags do not exist. The kernel would "
                    "read helper weights with no arrival proof."
                )
            generation = getattr(self, "_rebalance_generation", 0)
            if type(generation) is not int or not (1 <= generation < (1 << 63)):
                raise RuntimeError(
                    "MegaMoECuteDsl: rebalance helper slots are live "
                    f"(S={self._rebalance_slots_active}) but no READY "
                    "generation is published on the backend as "
                    f"``_rebalance_generation`` (got {generation!r}). The "
                    "[H, M) rows of the seven weight planes then carry no "
                    "arrival proof and the kernel would read them "
                    "unconditionally. Publish a rank-identical int in "
                    "[1, 2**63) alongside the weight transport that fills "
                    "them."
                )
            group = getattr(self, "_rebalance_scheduler_group", None)
            transport = getattr(self, "_rebalance_weight_transport", None)
            if group is not None:
                # The routing seam already submitted the matching scheduler and weight copy.
                _nvtx("mm/arena_ready_gen")
                try:
                    _rg = arena.ready_generation()
                finally:
                    _nvtx_end()
                if _rg != generation:
                    _nvtx("mm/arena_publish_gen")
                    try:
                        arena.publish_ready_generation(generation)
                    finally:
                        _nvtx_end()
            elif transport is None:
                raise RuntimeError(
                    "MegaMoECuteDsl: rebalance helper slots are live "
                    f"(S={self._rebalance_slots_active}) but no helper-weight "
                    "transport is installed on the backend as "
                    "``_rebalance_weight_transport``. Nothing would push the "
                    "hot experts into the [H, M) rows or publish the "
                    "uint64[EP] terminals for generation "
                    f"{generation}, so the in-device gate would spin to its "
                    "bounded budget and trap. In the end state this seam is "
                    "the external scheduler's SAMI copy submission."
                )
            # Submit an injected transport once per generation; repeated chunks share its lease.
            elif arena.ready_generation() != generation:
                transport(arena=arena, generation=generation, backend=self)
                arena.publish_ready_generation(generation)
            helper_ready_flags = arena.terminal_flags_tensor()
            _nvtx("mm/arena_ready_gen2")
            try:
                helper_ready_generation = int(arena.ready_generation())
            finally:
                _nvtx_end()
        # Quantization, independent staging, output clears, and host setup
        # are already enqueued. The router inside this op first reads routes.
        self._wait_rebalance_routes()
        try:
            torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell(
                # Cached dtype views alias the live staging and in-place weight storage.
                activation=self._memo_dtype_view(activation, _as_nvfp4),
                activation_sf=self._memo_dtype_view(activation_sf, _as_fp8_sf),
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                fc1_weight=self._memo_dtype_view(weight_view.fc1_weight, _as_nvfp4),
                fc1_weight_sf=self._memo_dtype_view(weight_view.fc1_weight_sf, _as_fp8_sf),
                fc2_weight=self._memo_dtype_view(weight_view.fc2_weight, _as_nvfp4),
                fc2_weight_sf=self._memo_dtype_view(weight_view.fc2_weight_sf, _as_fp8_sf),
                fc1_alpha=weight_view.fc31_alpha,
                fc2_alpha=weight_view.fc2_alpha,
                fc1_norm_const=weight_view.fc1_norm_const,
                combine_output=combine_output,
                shared_workspace=shared_workspace,
                world_size=world_size,
                local_rank=local_rank,
                num_topk=top_k,
                num_experts_per_rank=self.compute_slot_count,
                hidden_size=hidden,
                intermediate_size_per_partition=int(self.intermediate_size_per_partition),
                expand_intermediate_size_per_partition=int(
                    self.expand_intermediate_size_per_partition
                ),
                # Bucket max_T == the staging/symm buffer leading dim above.
                max_tokens_per_rank=int(launch_max_T),
                peer_offsets=peer_offsets,
                apply_topk_in_fc1=bool(self.apply_topk_in_fc1),
                gate_up_clamp=self.act_clamp,
                swiglu_alpha=(
                    self.act_alpha if self.activation_type is ActivationType.SwigluBias else None
                ),
                swiglu_beta=(
                    self.act_beta if self.activation_type is ActivationType.SwigluBias else None
                ),
                act_alpha=(self.act_alpha if self.activation_type is ActivationType.SiTu else None),
                act_beta=(self.act_beta if self.activation_type is ActivationType.SiTu else None),
                in_kernel_fc2_reduce=in_kernel_fc2_reduce,
                combine_format=self.combine_format,
                tactic_autotune=bool(self.tactic_autotune),
                num_tokens=num_tokens,
                # Helper-free launches omit the READY gate.
                helper_expert_count=_S_active,
                hot_expert_weight_ready_flags=helper_ready_flags,
                hot_expert_weight_ready_generation=int(helper_ready_generation),
            )
        finally:
            # Always clear: a raising op call must not leave the factory
            # global alive (its closure strongly references this module).
            set_active_megamoe_profiling_scratch(None)
            set_active_megamoe_profiling_scratch_factory(None)
        rebalance_group = getattr(self, "_rebalance_scheduler_group", None)
        # Bypass plan and finish together to preserve generation pairing.
        plan_ran = bool(getattr(self, "_rebalance_plan_ran", True))
        # The routing seam marks bypass explicitly for the next chunk.
        self._rebalance_plan_ran = False
        if rebalance_group is not None and plan_ran and _S_active > 0:
            # Finish records the last local consumer on MAIN. COPY must wait for it before
            # the next HALO-Q exchange permits a new generation to overwrite this live bank.
            rebalance_group.finish()
        if num_tokens == 0:
            return torch.empty((0, hidden), dtype=output_dtype, device=combine_output.device)
        # The kernel already collapsed the top-k axis into the (T, 1, hidden)
        # output (TopkReduce form-A / REDG form-B); just drop the singleton.
        return combine_output[:num_tokens].squeeze(1).to(output_dtype)

    def _prime_ladder(
        self,
        *,
        live_T: int,
        weight_view: MegaMoECuteDslWeightView,
        top_k: int,
        hidden: int,
        device,
        output_dtype: torch.dtype,
    ) -> None:
        """Warm-up: launch every other adaptive bucket once with zero live tokens.

        Each bucket owns a compiled kernel, a local workspace and a one-time
        barrier fence; serving never autotunes this op, so without this the
        first chunk landing in a new bucket pays ``cute.compile`` inside a
        serving iteration on every EP rank at once. Zero live tokens is the
        existing scheduler-invariant path (``topk_idx == -1`` everywhere); the
        ladder, the tuning-mode gate and the done-set are rank-identical.
        """
        key = (
            tuple(self._maxt_buckets),
            self.ep_size,
            hidden,
            int(self.intermediate_size_per_partition),
            int(self.expert_size_per_partition),
            top_k,
            self.combine_format,
            self.act_clamp,
            self.act_alpha,
            self.act_beta,
            self.apply_topk_in_fc1,
            str(device),
            output_dtype,
        )
        if key in _MEGAMOE_PRIMED_LADDERS:
            return
        t0 = time.perf_counter()
        primed = []
        # Bucket priming must disarm rebalance itself, not just skip finish:
        # the arm determines whether the kernel includes the READY ABI. Restore
        # an originally absent attribute by deleting it rather than assigning True.
        _prev_arm = getattr(self, "_rebalance_arm_open", _PRIME_ATTR_UNSET)
        _prev_plan_ran = getattr(self, "_rebalance_plan_ran", _PRIME_ATTR_UNSET)
        self._rebalance_arm_open = False
        try:
            self._prime_ladder_buckets(
                live_T=live_T,
                top_k=top_k,
                hidden=hidden,
                device=device,
                output_dtype=output_dtype,
                weight_view=weight_view,
                primed=primed,
            )
        finally:
            if _prev_arm is _PRIME_ATTR_UNSET:
                try:
                    del self._rebalance_arm_open
                except AttributeError:
                    pass
            else:
                self._rebalance_arm_open = _prev_arm
            if _prev_plan_ran is _PRIME_ATTR_UNSET:
                try:
                    del self._rebalance_plan_ran
                except AttributeError:
                    pass
            else:
                self._rebalance_plan_ran = _prev_plan_ran
        _MEGAMOE_PRIMED_LADDERS.add(key)
        if self.ep_rank == 0:
            logger.info(
                f"[MegaMoECuteDsl] primed adaptive buckets {primed} (live bucket "
                f"{live_T}) in {time.perf_counter() - t0:.1f}s"
            )

    def _prime_ladder_buckets(
        self,
        *,
        live_T,
        top_k,
        hidden,
        device,
        output_dtype,
        weight_view,
        primed,
    ):
        """Prime each max-token bucket; the caller manages the rebalance arm state."""
        for max_T in self._maxt_buckets:
            if max_T == live_T:
                continue
            bufs = self._acquire_buffers(
                top_k=top_k,
                hidden=hidden,
                device=device,
                output_dtype=output_dtype,
                launch_max_T=max_T,
            )
            self._stage_inputs(
                bufs=bufs,
                x=None,
                x_sf=None,
                topk_idx=None,
                topk_weights=None,
                num_tokens=0,
                top_k=top_k,
                stage_activation=False,
            )
            # Bucket priming skips the routing seam, so it must also skip finish.
            # It primes the S=0 bypass; ON tactic profiling separately runs its real
            # HALO-Q/TMA producer and READY protocol.
            self._rebalance_plan_ran = False
            self._launch_megamoe_kernel(
                activation=bufs.activation,
                activation_sf=bufs.activation_sf,
                topk_idx=bufs.topk_idx_local,
                topk_weights=bufs.topk_weights[:, :top_k],
                weight_view=weight_view,
                combine_output=bufs.combine_output,
                shared_workspace=bufs.shared_workspace,
                world_size=self.ep_size,
                local_rank=self.ep_rank,
                top_k=top_k,
                hidden=hidden,
                peer_offsets=bufs.peer_offsets,
                num_tokens=0,
                output_dtype=output_dtype,
                launch_max_T=max_T,
            )
            primed.append(max_T)

    def _run_moe(self, *a, **k):
        _nvtx("mm/run_moe_inner")
        try:
            return self._nvtxwrap__run_moe(*a, **k)
        finally:
            _nvtx_end()

    def _nvtxwrap__run_moe(
        self,
        *,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        weight_view: MegaMoECuteDslWeightView,
        num_tokens: int,
        top_k: int,
        hidden: int,
        device,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Unified MegaMoE CuteDSL forward: acquire -> stage -> launch.

        The kernel is launched with ``T = launch_max_T`` -- the smallest
        adaptive bucket >= the lockstep cross-rank chunk token count (set by
        ``set_adaptive_launch_tokens``), capped at ``max_num_tokens``. Live
        tokens fill the first ``num_tokens`` rows and the tail is masked via
        ``topk_idx == -1`` (skipped by dispatch_kernel) and zero
        ``topk_weights`` (combine stale-data guard). An exact-size genphase
        bucket can consume the caller's inputs directly because it has no
        padded tail and only reads its routing inputs locally.

        ``FusedCommMoEScheduler`` invariant 7 forces every EP rank to
        cross the NVLink barrier even with zero local tokens; only
        single-rank short-circuits ``num_tokens == 0`` because no peer
        is waiting.
        """
        # Pick and consume the adaptive bucket; un-hinted callers default to
        # the full bucket.
        launch_max_T = self._select_launch_max_tokens(self._active_launch_max_tokens)
        self._active_launch_max_tokens = None
        if num_tokens > launch_max_T:
            # By construction num_tokens <= max(all_rank_num_tokens) <=
            # launch_max_T; fail loudly rather than desync the barrier.
            raise RuntimeError(
                f"MegaMoECuteDsl run_moe got {num_tokens} tokens but the "
                f"selected adaptive bucket is sized for {launch_max_T} "
                f"(buckets={self._maxt_buckets}, max_num_tokens="
                f"{self.max_num_tokens}). This indicates the scheduler's "
                f"lockstep chunk-max hint diverged from the staged tokens."
            )
        if num_tokens == 0 and self.ep_size == 1:
            return torch.empty((0, hidden), dtype=output_dtype, device=device)

        if (
            self.ep_size > 1
            and not self.tactic_autotune
            and _MEGAMOE_PRIME_LADDER
            and AutoTuner.get().is_tuning_mode
        ):
            self._prime_ladder(
                live_T=launch_max_T,
                weight_view=weight_view,
                top_k=top_k,
                hidden=hidden,
                device=device,
                output_dtype=output_dtype,
            )

        bufs = self._acquire_buffers(
            top_k=top_k,
            hidden=hidden,
            device=device,
            output_dtype=output_dtype,
            launch_max_T=launch_max_T,
        )
        use_direct_inputs = (
            self.ep_size > 1
            and not self.tactic_autotune
            and num_tokens == launch_max_T
            and launch_max_T in self._genphase_direct_input_buckets
            and x.shape == bufs.activation.shape
            and x_sf.shape == bufs.activation_sf.shape
            and x.is_contiguous()
            and x_sf.is_contiguous()
        )
        use_direct_routing_inputs = (
            use_direct_inputs
            and topk_idx.shape == (launch_max_T, top_k)
            and topk_weights.shape == (launch_max_T, top_k)
            and topk_idx.dtype == self._topk_idx_dtype
            and topk_weights.dtype == torch.float32
            and topk_idx.device == device
            and topk_weights.device == device
            and topk_idx.is_contiguous()
            and topk_weights.is_contiguous()
            and topk_idx.data_ptr() % 16 == 0
            and topk_weights.data_ptr() % 16 == 0
        )
        borrowed_routes = None
        if self._rebalance_plan_ran:
            scheduler_routes = self._rebalance_scheduler_group.scheduler.outputs.physical_slot_ids
            if (
                topk_idx.dtype == self._topk_idx_dtype
                and topk_idx.data_ptr() == scheduler_routes.data_ptr()
                and launch_max_T <= scheduler_routes.shape[0]
            ):
                # HALO-Q writes the padded tail too. The existing consumer
                # event prevents the next generation from overwriting it.
                borrowed_routes = scheduler_routes[:launch_max_T]
        if not use_direct_routing_inputs:
            self._stage_inputs(
                bufs=bufs,
                x=x,
                x_sf=x_sf,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_tokens=num_tokens,
                top_k=top_k,
                stage_activation=not use_direct_inputs,
                stage_routes=borrowed_routes is None,
            )
        return self._launch_megamoe_kernel(
            activation=x if use_direct_inputs else bufs.activation,
            activation_sf=x_sf if use_direct_inputs else bufs.activation_sf,
            topk_idx=(
                borrowed_routes
                if borrowed_routes is not None
                else topk_idx
                if use_direct_routing_inputs
                else bufs.topk_idx_local
            ),
            topk_weights=(
                topk_weights if use_direct_routing_inputs else bufs.topk_weights[:, :top_k]
            ),
            weight_view=weight_view,
            combine_output=bufs.combine_output,
            shared_workspace=bufs.shared_workspace,
            world_size=self.ep_size,
            local_rank=self.ep_rank,
            top_k=top_k,
            hidden=hidden,
            peer_offsets=bufs.peer_offsets,
            num_tokens=num_tokens,
            output_dtype=output_dtype,
            launch_max_T=launch_max_T,
        )


# An alias, not a base class, so there is no second class to keep in step.
MegaMoECuteDsl = TrtllmCutedslMegaMoeNvfp4Impl
