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
``tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe``. The kernel is
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

import functools
import math
import os
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.distributed as dist

# ``megamoe_activation_sf_bytes_per_row`` lives at module top of the
# custom-op file (NOT inside its ``IS_MEGAMOE_OP_AVAILABLE`` gate), so
# it is always importable. The provider / shared-workspace helpers used
# in ``_alloc_symm_provider`` and ``_ensure_local_staging`` ARE inside
# that gate and therefore stay lazy at the call site.
from tensorrt_llm._torch.moe import live_rows
from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
    megamoe_activation_sf_bytes_per_row,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div, pad_up
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ....autotuner import AutoTuner
from ....cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ....model_config import _DEEPSEEK_V4_ARCHITECTURES as _V4_ARCHITECTURES
from ....model_config import _DEEPSEEK_V41_ARCHITECTURES as _V41_ARCHITECTURES
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
    MoEFusedRouteQuant,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    MoEStaticCapability,
)
from ..impl_environment import MoEDep
from ..impl_identity import MoEImplDescriptor, MoEImplId, register_moe_impl
from ..interface import MoESchedulerKind, MoEWeightLoadingMode, _reject
from ..moe_warmup_placement import RouteCounters
from ..quantization import NVFP4MegaMoECuteDslMethod
from ..routing import BaseMoeRoutingMethod
from .shared_expert import (
    build_shared_expert_weights,
    mxfp8_quantize_megamoe,
    shared_activation_buffers,
)

if IS_CUTLASS_DSL_AVAILABLE:
    from ....moe.kernels.blackwell.dsv4_route_quant import dsv4_route, dsv4_route_nvfp4_quant

__all__ = [
    "MegaMoECuteDsl",
    "MegaMoeCuteDslUnavailable",
    "MegaMoECuteDslWeightView",
    "TrtllmCutedslMegaMoeNvfp4Impl",
    "is_megamoe_cute_dsl_runtime_available",
]


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


_DEEPSEEK_V4_ARCHITECTURES = _V4_ARCHITECTURES | _V41_ARCHITECTURES


def _warn_shared_expert_fusion_ignored(reason: str) -> None:
    logger.warning_once(
        f"moe_config.megamoe_fuse_shared_expert is ignored: {reason}",
        key="megamoe_fuse_shared_expert_ignored",
    )


class MegaMoeCuteDslUnavailable(RuntimeError):
    """Raised when the active environment cannot import the symbols required by
    the active MegaMoE kernel (cu13 Cutlass DSL + cute_nvgpu MMA
    atoms / cutlass._mlir APIs used by sym_buffer)."""


_RUNTIME_PROBE_CACHE: Dict[int, Union[bool, str]] = {}

_MEGAMOE_PRIME_LADDER = os.environ.get("MEGAMOE_PRIME_LADDER", "1") == "1"
_MEGAMOE_PRIMED_LADDERS: set = set()


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
        from ....moe.kernels.megamoe_kernels import (  # noqa: F401
            BlackwellInferenceMegaMoE,
            RubinInferenceGenphaseMegaMoE,
            RubinInferenceLocalMegaMoE,
            RubinInferenceMegaMoE,
        )
    except ImportError as e:
        reason = (
            f"Ported MegaMoE NVFP4 kernel package failed to import: "
            f"{e!r}. Verify tensorrt_llm/_torch/moe/kernels/"
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


def _as_nvfp4(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float4_e2m1fn_x2 else t.view(torch.float4_e2m1fn_x2)


def _as_fp8_sf(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.float8_e4m3fn else t.view(torch.float8_e4m3fn)


def _slot_table(global_assignments: List[int]) -> torch.Tensor:
    """The int32 expert -> slot table of a placement with one copy per expert (slot -> expert list)."""
    table = [0] * len(global_assignments)
    for slot, expert in enumerate(global_assignments):
        table[expert] = slot
    return torch.tensor(table, dtype=torch.int32)


def replica_slot_table(
    global_assignments: List[int], num_experts: int, ep_rank: int
) -> torch.Tensor:
    """The routing table of a placement with copies (slot -> expert list holding every expert):
    ``num_experts`` words ``(copies << 16) | start``, then each expert's slots rotated by
    ``ep_rank``. A pick in row r takes copy ``h(r) % copies``, h(r) = (r * 0x9E3779B1 mod 2^32) >> 16,
    so ranks spread a hot expert's rows over its copies. Copies hold the same bytes."""
    slots_of: List[List[int]] = [[] for _ in range(num_experts)]
    for slot, expert in enumerate(global_assignments):
        slots_of[int(expert)].append(slot)
    meta, flat = [], []
    for slots in slots_of:
        if not slots or len(global_assignments) >= 1 << 16:
            raise ValueError("MegaMoE replica placement must hold every expert in < 65536 slots")
        k = ep_rank % len(slots)
        meta.append((len(slots) << 16) | len(flat))
        flat.extend(slots[k:] + slots[:k])
    return torch.tensor(meta + flat, dtype=torch.int32)


def _replica_slots(table: torch.Tensor, expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """``replica_slot_table``'s slots of ``expert_ids`` [rows, top_k], as the routing launch picks."""
    rows = torch.arange(expert_ids.shape[0], device=expert_ids.device, dtype=torch.int64)[:, None]
    meta = table[expert_ids.long()].long()
    copy = (((rows * 0x9E3779B1) & 0xFFFFFFFF) >> 16) % (meta >> 16)
    return table[num_experts + (meta & 0xFFFF) + copy].to(torch.int32)


class _QuantizedRows(NamedTuple):
    """The activation rows a quantization wrote into a bucket's staging for the next run_moe: the
    views it returned, the staging they view, and their row count. A run_moe given these very
    views needs no storage check."""

    x: torch.Tensor
    x_sf: torch.Tensor
    activation: torch.Tensor
    rows: int


def _is_staging_view(view: torch.Tensor, staging: torch.Tensor, rows: int) -> bool:
    """The bucket owns the complete row view, including an empty view with a zero data pointer."""
    return (
        view.shape == (rows, *staging.shape[1:])
        and view.stride() == staging.stride()
        and view.dtype == staging.dtype
        and view.device == staging.device
        and view.storage_offset() == staging.storage_offset()
        and view.untyped_storage().data_ptr() == staging.untyped_storage().data_ptr()
    )


@functools.cache
def _fp4_quantize_has_sm_budget() -> bool:
    if hasattr(torch.ops.trtllm.fp4_quantize, "sm_budget"):
        return True
    logger.warning(
        "fp4_quantize.sm_budget is unavailable; per-layer dynamic EPLB "
        "quantization will use the selected backend without SM control"
    )
    return False


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


class MegaMoeRouteStaging(NamedTuple):
    """The route staging of one bucket a caller may fill in place of ``run_moe``
    (:meth:`MegaMoECuteDsl.route_staging`)."""

    topk_idx: torch.Tensor
    """``[bucket, top_k]`` slot ids in the kernel's index dtype; -1 past the live rows."""
    topk_weights: torch.Tensor
    """``[bucket, top_k]`` float32 gate weights; 0 past the live rows."""
    bucket: int
    live_row_word: Optional[torch.Tensor]
    """The forward's live-row word (``moe/live_rows.py``) when the bucket's kernel drops the rows
    of padding requests, else None: the rows the staging keeps, and the statistics count."""


@dataclass(frozen=True)
class _StagingKey:
    """Key of one local staging set (``_ensure_local_staging``)."""

    max_T: int
    top_k: int
    hidden: int
    device: str
    output_dtype: torch.dtype


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
    peer_offsets: Tuple[int, ...]  # length == world_size; (0,) for single-rank
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
    ``TrtllmCutedslGroupedGemmNvfp4Impl``'s ``trtllm.cutedsl.grouped_gemm.nvfp4``
    differs from this id in that field alone.

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
        capabilities=MoEStaticCapability(
            supports_eplb=True,
            supports_per_layer_dynamic_eplb=True,
            supports_adaptive_eplb=True,
        ),
        doc="MegaMoE CuteDSL fused NVFP4 kernels: NVFP4 weights and activations, SM100/103/107.",
    )

    # Read off the descriptor, not restated, so the published contract and the
    # one the scheduler reads cannot drift apart.
    scheduler_kind = descriptor.scheduler_kind

    capabilities = descriptor.capabilities

    input_requirement = descriptor.input_requirement

    # run_moe issues MoERunContext.before_launch itself: after staging and
    # before the kernel, bucket priming included.
    takes_before_launch = True

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
        # TEP (attention TP + MoE EP) needs no gate here: the fused kernel
        # returns a globally combined routed output, and FusedCommMoEScheduler
        # hands it to callers that reduce externally as a rank partial. It is
        # correct but not cheap: every TP rank dispatches the same replicated
        # tokens, so expert compute runs parallel_size times (the reason
        # MegaMoEDeepGemm rejects TEP). validate_configurable_moe warns once.
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

        # DeepSeek V4 and V4.1 apply routing weights before FC2 input quantization.
        architectures = getattr(model_config.pretrained_config, "architectures", None) or ()
        self.apply_topk_in_fc1 = any(
            architecture in _DEEPSEEK_V4_ARCHITECTURES for architecture in architectures
        )
        self._topk_idx_dtype = torch.int32 if get_sm_version() == 107 else torch.int64
        # The fused launch writes unpadded scale rows, so only shapes without SF padding fuse.
        self._fuses_route_quant = (
            self.routing_method.sqrt_softplus_bias_selection
            and megamoe_activation_sf_bytes_per_row(self.hidden_size) == self.hidden_size // 16
        )

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

        # The FP8 shared expert runs inside the fused kernel once fuse_shared_expert succeeds.
        self.num_fused_shared_expert = type(self).fused_shared_expert_count(model_config)
        if self.num_fused_shared_expert and not (
            combine_format == "bf16"
            and self.activation_type is ActivationType.Swiglu
            and self.act_alpha is None
            and self.act_beta is None
        ):
            _warn_shared_expert_fusion_ignored("needs the bf16 combine and a plain SwiGLU")
            self.num_fused_shared_expert = 0
        self._shared_expert: Optional[dict] = None
        # (launch bucket, live tokens) of the last MXFP8 quantization of the shared expert's input.
        self._shared_quantized: Optional[Tuple[int, int]] = None

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
        # Launch buffers per (bucket, top_k, hidden, device, dtype, inference mode).
        self._launch_buffers: Dict[Tuple, _MegaMoeBuffers] = {}
        self._genphase_direct_input_buckets: set[int] = set()
        # Buckets whose kernel masks routes with an invalid expert in its top-k reduce: their
        # staging drops the rows of the forward's padding requests (moe/live_rows.py).
        self._padding_drop_buckets: set[int] = set()
        # Lockstep per-chunk cross-rank max for the NEXT run_moe (set via
        # ``set_adaptive_launch_tokens``); ``None`` -> full bucket.
        self._active_launch_max_tokens: Optional[int] = None
        self._active_launch_output_dtype: Optional[torch.dtype] = None
        # (routes view or None, bucket route staging) handed out to the next run_moe.
        self._handed_out_route_staging: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        # (bucket ids buffer, ids, weights, x, x_sf) written by the last fused route+quant launch.
        self._prestaged: Optional[Tuple[torch.Tensor, ...]] = None
        # An adaptive balancer's route for the next launch, which its dispatch router maps and counts.
        self._adaptive_route_launch: Optional[tuple] = None
        # Expert -> slot table of a static placement (built in create_weights), with copies a
        # ``replica_slot_table``, and its slot -> expert assignments.
        self._slot_of_expert: Optional[torch.Tensor] = None
        self._expert_of_slot: Optional[torch.Tensor] = None
        self._replicated = False
        # The rows that ``quantize_input`` (or a fused staging) quantized into for the NEXT run_moe.
        self._quantized_into: Optional[_QuantizedRows] = None

        # Resolve EP ProcessGroup at construction. Resolving at forward
        # time would be collective on a non-synchronous call stack and
        # deadlock under PP / layer-skip. Construction is globally
        # synchronous across ranks during model build.
        #
        # The bootstrap stays OUTSIDE the single-rank fallback below. Every
        # torch.distributed error is a RuntimeError, and swallowing a failed
        # ``init_process_group`` makes the next MegaMoE layer retry it on the
        # failing rank alone. ``init_process_group`` names the default group
        # from a counter it bumps before the rendezvous, so the retrying rank
        # exchanges gloo addresses under different store keys than its peers
        # and every rank hangs until the process-group timeout.
        self._maybe_init_torch_dist_under_mpi()
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
        # Cute views of this module's weights, reused across launches.
        self._cute_weight_cache: Dict[tuple, Any] = {}
        self._weights_loaded = False
        self._weights_created = False
        self.quant_method = None
        # Per-instance staging cache (key -> {tensor name: tensor}), one
        # staging set per bucket max_T; ``_stage_inputs`` keeps the
        # always-pad-to-max_T launch contract on every launch.
        self._local_staging_cache: Dict[_StagingKey, Dict[str, torch.Tensor]] = {}
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    # ------------------------------------------------------------------
    # Compute axis
    # ------------------------------------------------------------------
    @property
    def compute_slot_count(self) -> int:
        """Return the kernel-only slot axis ``M = H + S``.

        ``expert_size_per_partition`` remains the resident/checkpoint axis H.
        The per-layer dynamic EPLB layer handle supplies the S helper slots;
        without one, M is H. Only the seven live kernel planes, symmetric
        workspace, and launch geometry use this wider axis.
        """
        resident_experts = int(self.expert_size_per_partition)
        layer_load_balancer = self.layer_load_balancer
        if layer_load_balancer is None or not layer_load_balancer.produces_replica_plan:
            return resident_experts
        return resident_experts + int(layer_load_balancer.helper_slots_per_rank)

    # ------------------------------------------------------------------
    # Backend facts the per-layer dynamic EPLB layer handle reads
    # ------------------------------------------------------------------
    @property
    def ep_process_group(self):
        """The EP ProcessGroup resolved at construction, or ``None`` on one rank."""
        return self._ep_pg

    @property
    def max_tokens_per_rank_capacity(self) -> int:
        """The largest per-rank token count of any launch bucket."""
        return int(self._maxt_buckets[-1])

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

    def set_adaptive_launch_tokens(
        self, chunk_max_tokens: Optional[int], *, output_dtype: Optional[torch.dtype] = None
    ) -> None:
        """Scheduler hook: record the lockstep per-chunk cross-rank max token
        count for the next ``run_moe``.

        MUST be ``max(all_rank_num_tokens)`` for the chunk (rank-identical) so
        all ranks select the same bucket; ``None`` -> full bucket. Consumed
        and reset by the next ``_run_moe``. Set it before ``quantize_input``:
        with the hint set, ``quantize_input`` may quantize into that bucket's
        staging.
        Supply ``output_dtype`` for direct preparation; otherwise preparation uses ordinary
        tensors because the next launch's staging is not yet identified.
        """
        self._active_launch_max_tokens = chunk_max_tokens
        self._active_launch_output_dtype = output_dtype

    def _take_launch_preparation(self) -> Tuple[int, Optional[_QuantizedRows]]:
        """The bucket of this launch and the rows ``quantize_input`` staged for it.

        Every ``_run_moe`` path takes these first, which consumes the bucket hint, the output dtype
        that lets ``quantize_input`` stage into that bucket, and the staged-rows record, so none
        of them outlives its launch. Un-hinted callers get the full bucket.
        """
        launch_max_T = self._select_launch_max_tokens(self._active_launch_max_tokens)
        self._active_launch_max_tokens = None
        self._active_launch_output_dtype = None
        quantized_into, self._quantized_into = self._quantized_into, None
        return launch_max_T, quantized_into

    def _may_read_inputs_in_place(self, num_tokens: int, bucket: int) -> bool:
        """Whether a launch of ``num_tokens`` rows into ``bucket`` may read its inputs where they
        are (an exact-size generation-phase bucket across ranks) instead of staging them.

        This is the bucket half of the rule: ``_run_moe`` also requires the inputs' shapes and
        contiguity to match the bucket's buffers, so False means "staged", not True "not staged".
        """
        return (
            self.ep_size > 1
            and not self.tactic_autotune
            and num_tokens == bucket
            and bucket in self._genphase_direct_input_buckets
        )

    def route_staging(
        self, num_tokens: int, top_k: int, device: torch.device
    ) -> Optional[MegaMoeRouteStaging]:
        """The index and weight staging of the bucket the next ``run_moe`` selects, for a
        load balancer that writes slot ids and gate weights straight into it.

        The bucket is the one ``set_adaptive_launch_tokens`` hinted (so the hint precedes the
        route). The caller writes rows ``[0, num_tokens)`` as ``trtllm::megamoe_stage_routes``
        stages them under the returned ``live_row_word``: the slots and float32 gate weights of
        the live rows, -1 and 0 in the rows of padding requests and in rows ``[num_tokens,
        bucket)``. It then passes ``topk_idx[:num_tokens]`` as the run's
        ``token_selected_experts``: ``run_moe`` recognises its own staging by address and does
        not stage the routes again. Returns None when that ``run_moe`` does not stage routes (no
        hint yet, tactic autotuning, a bucket whose inputs are read in place, buffers not built);
        the caller then routes into a tensor of its own.
        """
        if top_k != self.routing_method.experts_per_token:
            return None
        bufs = self._staging_of_next_run(num_tokens, device)
        if bufs is None:
            return None
        bucket = bufs.topk_idx_local.shape[0]
        # run_moe takes routes found in this staging only after this hand-out (whose caller
        # writes the gate weights beside the slots): _routes_staged.
        self._handed_out_route_staging = (None, bufs.topk_idx_local)
        word = live_rows.current_word(device) if bucket in self._padding_drop_buckets else None
        return MegaMoeRouteStaging(bufs.topk_idx_local, bufs.topk_weights, bucket, word)

    def _routes_staged(
        self,
        bufs: "_MegaMoeBuffers",
        topk_idx: torch.Tensor,
        num_tokens: int,
        top_k: int,
        replica_plan: Optional[object],
    ) -> bool:
        """Whether the routes of this launch are the live rows of the bucket's index staging, written
        by a caller of :meth:`route_staging` together with the gate weights beside them.

        The launch then reads both from the staging, so routes found there without that hand-out
        (gate weights possibly stale) and routes a replica plan would rewrite are refused."""
        handed_out, self._handed_out_route_staging = self._handed_out_route_staging, None
        # The view the one-launch staging returned, handed on as is, needs no storage check.
        staged = (
            handed_out is not None
            and topk_idx is handed_out[0]
            and handed_out[1] is bufs.topk_idx_local
        ) or (
            _is_staging_view(topk_idx, bufs.topk_idx_local, num_tokens)
            and topk_idx.shape == (num_tokens, top_k)
        )
        if not staged:
            if handed_out is not None:
                raise ValueError(
                    "routes from route_staging must retain the bucket's row-view identity"
                )
            return False
        if handed_out is None or (
            handed_out[1] is not bufs.topk_idx_local
            and handed_out[1].data_ptr() != bufs.topk_idx_local.data_ptr()
        ):
            raise ValueError(
                "routes in the MegaMoE route staging must come from route_staging, which writes "
                "the gate weights beside them"
            )
        if replica_plan is not None:
            raise ValueError(
                "routes in the MegaMoE route staging cannot also follow a replica plan"
            )
        return True

    def _staging_of_next_run(self, num_tokens: int, device) -> Optional[_MegaMoeBuffers]:
        """The buffers the next ``run_moe`` stages ``num_tokens`` rows on ``device`` into.

        Without a bucket hint (EP1, attention TP) the next run launches the full bucket on every
        rank, so that bucket's staging is the one. None when ``quantize_input`` should not write
        into the staging: tactic autotuning, a bucket whose inputs may be read in place, or a
        staging set not built yet.
        """
        if self.tactic_autotune or not self._weights_created:
            return None
        bucket = self._select_launch_max_tokens(self._active_launch_max_tokens)
        if num_tokens > bucket or self._may_read_inputs_in_place(num_tokens, bucket):
            return None
        output_dtype = self._active_launch_output_dtype
        if output_dtype is None:
            return None  # The next launch's output buffer, and hence staging, is not known.
        key = _StagingKey(
            bucket,
            self.routing_method.experts_per_token,
            self.hidden_size,
            str(device),
            output_dtype,
        )
        if key not in self._local_staging_cache:
            return None
        return self._acquire_buffers(
            top_k=key.top_k,
            hidden=key.hidden,
            device=device,
            output_dtype=key.output_dtype,
            launch_max_T=bucket,
        )

    # ------------------------------------------------------------------
    # Adaptive EPLB backend methods (``supports_adaptive_eplb``)
    # ------------------------------------------------------------------
    # The MNNVL memory the slot parameters point into once adaptive EPLB moved them.
    adaptive_eplb_block: Optional[object] = None

    @property
    def slot_id_dtype(self) -> torch.dtype:
        """The integer dtype the kernel reads slot ids in."""
        return self._topk_idx_dtype

    def slot_weight_planes(self) -> List[Tuple[str, int]]:
        """Names and byte alignments of the live weight planes, whose leading axis is the slot.

        A load balancer that moves expert weights between ranks relocates these parameters whole;
        each must start at a multiple of its alignment. Raises ``ValueError`` when a parameter is
        not the plane the quant method declares.
        """
        spec = self.quant_method.live_weight_plane_spec(self)
        slots = int(self.compute_slot_count)
        planes = []
        for plane in spec.planes:
            shape, stride, dtype_name, _, storage_alignment = plane.storage_metadata(slots)
            param = self.get_parameter(plane.name)
            declared = (getattr(torch, dtype_name), tuple(shape), tuple(stride))
            if (param.dtype, tuple(param.shape), tuple(param.stride())) != declared:
                raise ValueError(
                    f"MoE layer {self.layer_idx}, plane {plane.name!r}: the parameter is not the "
                    "plane the quant method declares"
                )
            kernel_alignment = plane.kernel_metadata(slots)[-1]
            alignment = math.lcm(int(plane.alignment), storage_alignment, kernel_alignment)
            planes.append((plane.name, alignment))
        return planes

    def check_weights_movable(self) -> None:
        """Raise once the kernel holds views of the weight planes (after its first launch)."""
        if self._cute_weight_cache:
            raise RuntimeError(
                f"MoE layer {self.layer_idx} has launched its kernel: its cute weight views "
                "hold the current plane addresses, so its planes can no longer move"
            )

    def decode_time_model(
        self, expert_bytes: int, rates: Tuple[float, float], *, tile_rows: int
    ) -> Dict[str, float]:
        """Parameters of the decode kernels' time model at ``rates`` (``policy.TimeModel``).

        A slot that receives rows streams its expert's ``expert_bytes`` once, at ``rates[0]``
        bytes/s; a row costs its FC1 and FC2 FLOPs at ``rates[1]`` FLOP/s; every tile of
        ``tile_rows`` rows (the M of the kernels' MMA tile, :meth:`decode_tile_rows`) of a slot
        beyond its first costs a tile of rows, and every such tile of the GPU's busiest slot half a
        tile more, the serial tail of that slot. Only this structure is the backend's: the rates,
        the scale and the read cost per slot come from a calibration on the GPU.
        """
        bandwidth, peak = rates
        inner = int(self.expand_intermediate_size_per_partition) + int(
            self.intermediate_size_per_partition
        )
        row_us = 2.0 * self.hidden_size * inner / peak * 1e6
        return dict(
            per_active_slot_us=expert_bytes / bandwidth * 1e6,
            per_extra_tile_us=tile_rows * row_us,
            per_row_us=row_us,
            hot_slot_extra_tile_us=tile_rows / 2 * row_us,
            scale=1.0,
            tile_rows=int(tile_rows),
        )

    def _decode_tactic(self, tokens: int) -> Tuple:
        """The default tactic of a decode step of ``tokens`` tokens per GPU, found without a launch."""
        from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
            _default_megamoe_tactic_for_problem,
        )

        max_t = self._select_launch_max_tokens(tokens)
        return _default_megamoe_tactic_for_problem(
            sm_version=get_sm_version(),
            max_tokens_per_rank=max_t,
            num_tokens=tokens,
            apply_topk_in_fc1=bool(self.apply_topk_in_fc1),
            in_kernel_fc2_reduce=self._use_in_kernel_fc2_reduce(max_t),
            combine_format=self.combine_format,
        )

    def decode_tile_rows(self, tokens: List[int]) -> int:
        """Rows of one MMA tile (its M) of the decode kernels: the tactic of the largest decode bucket
        of ``tokens`` tokens per GPU."""
        return int(self._decode_tactic(max(int(t) for t in tokens))[0][0])

    def decode_calibration_identity(self, tokens: List[int]) -> Dict[str, object]:
        """What a calibration of :meth:`decode_time_model` holds for: the kernels' geometry and
        combine format, and the tactic a decode step of each of ``tokens`` tokens per GPU resolves
        to, found without a launch. A bucket of more tokens than the largest launch takes runs in
        no single launch: it has none."""
        tactics = {}
        for t in tokens:
            try:
                self._select_launch_max_tokens(int(t))
            except RuntimeError:
                continue
            tactic = self._decode_tactic(int(t))
            tactics[str(int(t))] = "autotune" if self.tactic_autotune else repr(tactic)
        return dict(
            backend=type(self).__name__,
            combine_format=self.combine_format,
            hidden_size=int(self.hidden_size),
            intermediate_size=int(self.intermediate_size_per_partition),
            top_k=int(self.routing_method.experts_per_token),
            tactics=tactics,
        )

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
        if not moe.use_dp and moe.parallel_size > 1:
            logger.warning_once(
                f"MegaMoECuteDsl under attention TP (TEP, tp_size="
                f"{moe.parallel_size}) dispatches the same tokens from every TP "
                f"rank, so expert compute is repeated {moe.parallel_size} times. "
                "Prefer enable_attention_dp=True for throughput.",
                key="megamoe_cutedsl_tep_redundant_compute",
            )
        # Dynamic EPLB is intentionally allowed: the quant method
        # registers mega-format derived parameters alongside the raw
        # NVFP4 family so per-slot migration stays byte-consistent.

    @classmethod
    def fused_shared_expert_count(cls, model_config: ModelConfig) -> int:
        """Run the FP8 shared expert inside the fused kernel (moe_config.megamoe_fuse_shared_expert).

        Only on SM100/SM103 under attention DP with one shared expert, which is then rank-local.
        """
        if not model_config.megamoe_fuse_shared_expert:
            return 0
        # Only DeepseekV4MoE hands its shared expert over and keeps it when the fusion is refused.
        architectures = getattr(model_config.pretrained_config, "architectures", None) or ()
        mapping = model_config.mapping
        n_shared_experts = getattr(model_config.pretrained_config, "n_shared_experts", 0) or 0
        if not any(architecture in _DEEPSEEK_V4_ARCHITECTURES for architecture in architectures):
            reason = "not a DeepSeek-V4 model"
        elif get_sm_version() not in (100, 103):
            reason = "needs SM100/SM103"
        elif not mapping.enable_attention_dp or mapping.moe_tp_size != 1:
            reason = "needs attention DP with moe_tensor_parallel_size 1"
        elif n_shared_experts != 1:
            reason = "needs exactly one shared expert"
        else:
            return 1
        _warn_shared_expert_fusion_ignored(reason)
        return 0

    def can_fuse_shared_expert(self, shared_experts) -> bool:
        """Whether ``fuse_shared_expert`` would take over this MLP (builds and drops its weights)."""
        return (
            self.num_fused_shared_expert > 0
            and self._shared_expert is None
            and build_shared_expert_weights(shared_experts, self.hidden_size) is not None
        )

    def decline_shared_expert(self) -> None:
        """Keep the shared expert separate: ``fuse_shared_expert`` refuses it from now on."""
        self.num_fused_shared_expert = 0

    def fuse_shared_expert(self, shared_experts) -> bool:
        """Take over a gated FP8 block-scale MLP; returns whether the fused kernel now runs it."""
        if self.num_fused_shared_expert == 0 or self._shared_expert is not None:
            return False
        weights = build_shared_expert_weights(shared_experts, self.hidden_size)
        if weights is None:
            logger.warning_once(
                "[MegaMoECuteDsl] the shared expert is not a fusable FP8 block-scale MLP; "
                "it keeps running separately.",
                key="megamoe_shared_expert_not_fused",
            )
            return False
        for name in ("fc1_weight", "fc1_weight_sf", "fc2_weight", "fc2_weight_sf"):
            self.register_buffer(f"shared_{name}", weights.pop(name), persistent=False)
        self._shared_expert = weights
        device = self.shared_fc1_weight.device
        for max_T in self._maxt_buckets:
            shared_activation_buffers(max_T, self.hidden_size, device)
        logger.info_once(
            "[MegaMoECuteDsl] the FP8 shared expert runs inside the fused kernel.",
            key="megamoe_shared_expert_fused",
        )
        return True

    def _shared_activation_target(
        self, num_tokens: int
    ) -> Optional[Tuple[int, torch.Tensor, torch.Tensor]]:
        """The launch bucket and its MXFP8 buffers for the shared expert's input, if fused and they fit."""
        if self._shared_expert is None:
            return None
        launch_max_T = self._select_launch_max_tokens(self._active_launch_max_tokens)
        if num_tokens > launch_max_T:
            # run_moe rejects this launch; the bucket's buffers must not be overrun first.
            return None
        x8, x8_sf = shared_activation_buffers(
            launch_max_T, self.hidden_size, self.shared_fc1_weight.device
        )
        return launch_max_T, x8, x8_sf

    # ------------------------------------------------------------------
    # EP process-group resolution (no collective at forward time)
    # ------------------------------------------------------------------
    def _maybe_init_torch_dist_under_mpi(self):
        """Bootstrap a torch.distributed NCCL WORLD group under the MPI orchestrator.

        The symmetric-memory rendezvous needs a ProcessGroup; for pure DEP a
        node-local NCCL WORLD suffices. No-op under Ray or single-rank. Failures
        propagate: a rank that retried ``init_process_group`` alone would never
        rendezvous with its peers.
        """
        from tensorrt_llm._torch.distributed.initialization import ensure_cutedsl_torch_distributed

        ensure_cutedsl_torch_distributed(self.ep_size)

    def _resolve_ep_pg(self):
        """Return the torch.distributed ProcessGroup for the EP sub-world.

        Mirrors :meth:`MegaMoEDeepGemm._resolve_ep_pg` so the two MegaMoE
        backends share the same fallback chain. The caller runs
        :meth:`_maybe_init_torch_dist_under_mpi` first.
        """
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
        # Step 1: build-time symmetric memory allocation (multi-rank only).
        # Single-rank degenerate uses local CUDA tensors and skips here.
        if self.ep_size > 1:
            self._alloc_symm_provider()
        # Step 2-3: quant method registers all NVFP4 + MegaMoE-format params.
        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)
        # A static placement (one copy per expert or a replicated cover) is applied by the routing
        # launch, which writes slots.
        self._slot_of_expert = self._make_slot_table()
        if self._slot_of_expert is not None:
            with torch.inference_mode(False):
                self._expert_of_slot = torch.tensor(
                    list(self.initial_global_assignments),
                    dtype=torch.int32,
                    device=self._slot_of_expert.device,
                )
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
        self._launch_buffers.clear()
        for max_T in self._maxt_buckets:
            properties = query_megamoe_kernel_properties(
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
            shared_workspace_bytes = properties.shared_workspace_bytes
            if properties.supports_direct_inputs:
                self._genphase_direct_input_buckets.add(max_T)
            if properties.masks_invalid_routes:
                self._padding_drop_buckets.add(max_T)
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
                adaptive_route=getattr(self.layer_load_balancer, "router_route", None) is not None,
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

    def pre_reload_weights(self) -> None:
        raise NotImplementedError(
            "MegaMoE-CuteDSL does not support hot weight reloading; its "
            "source weights are released after initial packing."
        )

    @property
    def maps_expert_slots(self) -> bool:
        """Whether this layer maps routed experts to slots itself (a static placement), so the
        scheduler hands it expert ids instead of running the balancer's routing."""
        return self._slot_of_expert is not None

    def _routing_table(self, global_assignments: List[int]) -> torch.Tensor:
        if self._replicated:
            return replica_slot_table(global_assignments, int(self.num_experts), int(self.ep_rank))
        return _slot_table(global_assignments)

    def _make_slot_table(self) -> Optional[torch.Tensor]:
        balancer = self.layer_load_balancer
        if balancer is None or not balancer.is_static_routing() or self.ep_size == 1:
            return None
        assignments = list(self.initial_global_assignments)
        if set(assignments) != set(range(self.num_experts)):
            return None
        self._replicated = self.num_slots != self.num_experts
        # A normal (not inference) tensor at a fixed address, rewritten in place when experts move.
        with torch.inference_mode(False):
            return self._routing_table(assignments).to(
                torch.device("cuda", torch.cuda.current_device())
            )

    def on_expert_placement_changed(self, global_assignments: List[int]) -> None:
        """The warmup placement moved experts between slots: rewrite the routing table in place, so
        CUDA graphs captured before the move route to the new slots. The caller moved the per-slot
        weights and the balancer's table, with the device idle."""
        with torch.inference_mode():
            self._slot_of_expert.copy_(self._routing_table(global_assignments))
            self._expert_of_slot.copy_(torch.tensor(global_assignments, dtype=torch.int32))

    def slots_of_experts(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """The slots the static placement routes ``expert_ids`` [rows, top_k] to, as the launch does."""
        if self._replicated:
            return _replica_slots(self._slot_of_expert, expert_ids, int(self.num_experts))
        return torch.index_select(self._slot_of_expert, 0, expert_ids.reshape(-1)).view(
            expert_ids.shape
        )

    def experts_of_slots(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """The experts the static placement holds in ``slot_ids``."""
        return torch.index_select(self._expert_of_slot, 0, slot_ids.reshape(-1)).view(
            slot_ids.shape
        )

    def _build_weight_view(self) -> MegaMoECuteDslWeightView:
        """Bundle the MegaMoE-format weight tensors registered by the
        quant method. ``run_moe`` calls this once per chunk so the
        kernel sees the latest dynamic-EPLB migration outcome (once
        that path lands; currently the slots are static).
        """
        return MegaMoECuteDslWeightView(
            fc1_weight=self.mega_fc1_weight,
            fc1_weight_sf=self.mega_fc1_weight_sf,
            fc2_weight=self.mega_fc2_weight,
            fc2_weight_sf=self.mega_fc2_weight_sf,
            fc31_alpha=self.fc31_alpha,
            fc2_alpha=self.fc2_alpha,
            fc1_norm_const=self.fc1_norm_const,
        )

    # ------------------------------------------------------------------
    # MoE-contract methods
    # ------------------------------------------------------------------
    # The routing launch adds its picks to the warmup placement's counters.
    counts_routes = True

    def _route_quant_fuses(self) -> bool:
        # The fused launch quantizes x as is; a pre-quant scale (AWQ/ARC) must apply first.
        return self._fuses_route_quant and getattr(self, "fc31_act_scale", None) is None

    def takes_router_logit_partials(self) -> bool:
        return self._route_quant_fuses()

    def route_from_logit_partials(
        self,
        router_logits: Union[torch.Tensor, "RouterLogitPartials"],  # noqa: F821
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sum partial logits in the route kernel; leave adaptive input staging to run_moe."""
        ids, weights, _ = dsv4_route(
            router_logits,
            self.routing_method.e_score_correction_bias,
            self.routing_method.top_k,
            self.routing_method.routed_scaling_factor,
            torch.float32,
            self.hidden_size,
        )
        return ids, weights

    def _router_launch_max_tokens(self, num_tokens: int) -> Optional[int]:
        """The launch bucket a fused route+quant of ``num_tokens`` rows writes, or None."""
        if self._active_launch_max_tokens is None or self.tactic_autotune or num_tokens == 0:
            return None
        launch_max_T = self._select_launch_max_tokens(self._active_launch_max_tokens)
        if num_tokens > launch_max_T or launch_max_T in self._genphase_direct_input_buckets:
            return None
        return launch_max_T

    def routes_adaptive_in_router(self, num_tokens: int) -> bool:
        """Whether a launch of ``num_tokens`` rows can take an adaptive balancer's routes as expert
        ids, mapped to slots and counted by its dispatch router (either router: the one-round push
        and the metadata push carry the same adaptive route): the model's fused route+quant writes
        the launch bucket. Otherwise the balancer stages the slots itself."""
        if not IS_CUTLASS_DSL_AVAILABLE or self._slot_of_expert is not None:
            return False
        if not self._route_quant_fuses():
            return False
        return self._router_launch_max_tokens(num_tokens) is not None

    def try_fused_route_quant(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        route_counters: Optional[RouteCounters] = None,
        adaptive_route: Optional[tuple] = None,
    ) -> Optional[MoEFusedRouteQuant]:
        """Sqrt-softplus routing and NVFP4 input quantization in one launch, written straight into
        the launch bucket's kernel inputs once the scheduler has chosen the bucket; a static
        placement's slots go into the bucket and ``route_counters`` get the picks. With
        ``adaptive_route`` (an adaptive balancer's ``router_route``) the bucket keeps the expert ids,
        which the launch's dispatch router maps and counts (``routes_mapped``)."""
        if not self._route_quant_fuses():
            return None
        bucket = self._route_quant_bucket(x.shape[0], self.routing_method.top_k, x.device, x.dtype)
        if adaptive_route is not None and (
            bucket is None or not self.routes_adaptive_in_router(x.shape[0])
        ):
            return None
        table = self._slot_of_expert
        shared_target = self._shared_activation_target(x.shape[0])
        # A bucket whose kernel drops masked routes stages the padding requests' rows as -1 / 0,
        # as megamoe_stage_routes does on the other paths.
        word = None
        if bucket is not None and bucket[0].shape[0] in self._padding_drop_buckets:
            word = live_rows.current_word(x.device)
        ids, weights, x_fp4, x_sf, _ = dsv4_route_nvfp4_quant(
            router_logits,
            self.routing_method.e_score_correction_bias,
            x,
            self.fc31_input_scale,
            self.routing_method.top_k,
            self.routing_method.routed_scaling_factor,
            torch.float32,
            staged=bucket,
            slot_of_expert=None if bucket is None else table,
            replicas=bucket is not None and self._replicated,
            counters=route_counters,
            mxfp8=None if shared_target is None else shared_target[1:],
            live_row_word=word,
        )
        if shared_target is not None:
            self._shared_quantized = (shared_target[0], x.shape[0])
        if bucket is not None:
            self._prestaged = (bucket[0], ids, weights, x_fp4, x_sf)
            # The activation rows are in the bucket as quantize_input leaves them, so a launch
            # given other routes (perfect router, calibrator replay) re-stages only the routes.
            self._quantized_into = _QuantizedRows(x_fp4, x_sf, bucket[2], x.shape[0])
        if adaptive_route is not None:
            table_row, rows, forwards, enabled, first = adaptive_route
            self._adaptive_route_launch = (
                table_row,
                rows,
                forwards,
                enabled,
                int(x.shape[0]),
                first,
            )
        return MoEFusedRouteQuant(
            ids,
            weights,
            x_fp4,
            x_sf,
            routes_counted=route_counters is not None,
            routes_mapped=adaptive_route is not None,
        )

    def _route_quant_bucket(self, num_tokens: int, top_k: int, device, output_dtype):
        """The launch bucket's (ids, weights, activation, scales) when the fused launch can write them
        directly, else None: the hinted bucket, or without a hint (EP1, attention TP) the full bucket,
        which the next run launches on every rank."""
        if self.tactic_autotune or num_tokens == 0:
            return None
        output_dtype = self._active_launch_output_dtype
        if output_dtype is None:
            return None
        launch_max_T = self._select_launch_max_tokens(self._active_launch_max_tokens)
        if num_tokens > launch_max_T or launch_max_T in self._genphase_direct_input_buckets:
            return None
        bufs = self._acquire_buffers(
            top_k=top_k,
            hidden=self.hidden_size,
            device=device,
            output_dtype=output_dtype,
            launch_max_T=launch_max_T,
        )
        return bufs.topk_idx_local, bufs.topk_weights, bufs.activation, bufs.activation_sf

    def quantize_input(
        self,
        x: Union[torch.Tensor, "Fp4QuantizedTensor"],
        *,
        post_quant_comm: bool = False,
        reserved_sms: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """BF16 -> NVFP4 packed activation + plain K-major FP8 SF.

        Reuses ``torch.ops.trtllm.fp4_quantize`` with
        ``is_sf_swizzled=False`` so the SF tensor lands in the plain
        K-major layout expected by the MegaMoE kernel.
        ``self.fc31_input_scale`` is the per-tensor FP32 input scale
        registered by the quantization method's ``create_weights``; the
        value defaults to 1.0 until the checkpoint loader sets it.

        A checkpoint that carries a per-channel pre-quant scale (NVFP4_AWQ /
        NVFP4_ARC) gets it applied before quantization: the quantization
        method loads it into ``self.fc31_act_scale`` (``[1, hidden]``), and
        ``fc31_input_scale`` was calibrated on the scaled activation. The
        kernel itself only ever sees the quantized input.

        ``reserved_sms`` is the number of SMs left to the concurrent
        HALO-Q/TMA kernel of a replica plan; zero uses the default
        quantization overload.

        With an SF row that needs no padding, the rows are quantized straight
        into the staging of the bucket the next ``run_moe`` selects: the one
        ``set_adaptive_launch_tokens`` hinted, or without a hint (EP1, attention
        TP) the full bucket, once that staging is built (with the same SM
        reservation), and the returned tensors are views of it. On EP>1 that staging is the
        symmetric region every same-shape MegaMoE layer shares, so the views
        hold these rows only until the next MegaMoE ``quantize_input`` or
        ``run_moe``. That ``run_moe`` raises unless it launches exactly these
        rows from that staging.

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
        if x.shape[0] == 0:
            empty_x = torch.empty((0, hidden // 2), dtype=torch.uint8, device=x.device)
            empty_sf = torch.empty((0, sf_cols), dtype=torch.uint8, device=x.device)
            return empty_x, empty_sf
        x_bf16 = x.to(torch.bfloat16).contiguous()
        x_routed = self._activation_to_quantize(x)
        raw_cols = ceil_div(hidden, 16)
        # A forward that run_moe stages: quantize straight into the bucket's activation
        # rows, which run_moe recognizes by address and does not copy again.
        staged = (
            self._staging_of_next_run(x_bf16.shape[0], x_bf16.device)
            if sf_cols == raw_cols
            else None
        )
        if staged is not None:
            rows = x_bf16.shape[0]
            x_fp4 = staged.activation[:rows]
            x_sf = staged.activation_sf[:rows]
            # Positional, linear scale factors, and the replica plan's SMs left to HALO-Q/TMA.
            torch.ops.trtllm.fp4_quantize.into(
                x_routed, self.fc31_input_scale, x_fp4, x_sf, False, reserved_sms
            )
            shared_target = self._shared_activation_target(rows)
            if shared_target is not None:
                mxfp8_quantize_megamoe(x_bf16, shared_target[1], shared_target[2])
                self._shared_quantized = (shared_target[0], rows)
            self._quantized_into = _QuantizedRows(x_fp4, x_sf, staged.activation, rows)
            return x_fp4, x_sf
        # Leave a replica plan's SMs to its concurrent HALO-Q/TMA kernel. The
        # default overload remains unchanged without a plan.
        if reserved_sms > 0 and _fp4_quantize_has_sm_budget():
            # Positional: a keyword argument costs the op dispatch about a microsecond.
            x_fp4, x_sf = torch.ops.trtllm.fp4_quantize.sm_budget(
                x_routed, self.fc31_input_scale, 16, False, False, reserved_sms
            )
        else:
            x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(
                x_routed,
                self.fc31_input_scale,
                16,  # scaling_vector_size == Nvfp4BlockSize
                False,  # sf_use_ue8m0
                False,  # is_sf_swizzled - MegaMoE expects plain K-major
            )
        shared_target = self._shared_activation_target(x_bf16.shape[0])
        if shared_target is not None:
            # The shared expert takes the unscaled input.
            mxfp8_quantize_megamoe(x_bf16, shared_target[1], shared_target[2])
            self._shared_quantized = (shared_target[0], x_bf16.shape[0])
        # ``fp4_quantize(is_sf_swizzled=False)`` returns LINEAR layout
        # ``(rows, ceil(hidden/16))`` with no column pad. The kernel TMA
        # load needs ``pad_up(ceil_div(hidden, 16), 4)`` bytes per row
        # (== ``megamoe_activation_sf_bytes_per_row``), so
        # 32-aligned-but-not-64-aligned hidden sizes (1568, 1632, 2080)
        # come back 2 bytes short; pad the tail before returning.
        x_sf_raw = x_sf.view(x_bf16.shape[0], raw_cols)
        if sf_cols == raw_cols:
            return x_fp4, x_sf_raw
        padded_sf = torch.zeros((x_bf16.shape[0], sf_cols), dtype=torch.uint8, device=x_bf16.device)
        padded_sf[:, :raw_cols] = x_sf_raw
        return x_fp4, padded_sf

    def _activation_to_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` as the NVFP4 quantization reads it: contiguous bfloat16, times a checkpoint's
        per-channel pre-quant scale ``fc31_act_scale`` (``[1, hidden]``) if it has one, on which
        ``fc31_input_scale`` was calibrated."""
        x_bf16 = x.to(torch.bfloat16).contiguous()
        fc31_act_scale = getattr(self, "fc31_act_scale", None)
        if fc31_act_scale is not None:
            # Product stays bf16, which is what the quantization is given on
            # every route into it: ``_SUPPORTED_ACTIVATION_DTYPES`` admits bf16
            # activations only, so ``fc31_act_scale``, created at
            # ``module.dtype``, is bf16 as well.
            x_bf16 = x_bf16 * fc31_act_scale
        return x_bf16

    def supports_fused_prepare(self) -> bool:
        """Whether ``run_moe`` takes the BF16 activation (``x_sf`` None) and quantizes it in the
        launch that stages the routes (:meth:`_stage_bf16_inputs`). A layer with a replica plan
        keeps ``quantize_input``, which leaves SMs to the plan's concurrent HALO-Q/TMA kernel."""
        balancer = self.layer_load_balancer
        return hasattr(torch.ops.trtllm, "megamoe_stage_inputs") and (
            balancer is None
            or not (balancer.produces_replica_plan or balancer.is_dynamic_routing())
        )

    def _stage_bf16_inputs(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        route: Optional[Callable[..., torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``x`` and stage the routes in one launch, into the staging of the bucket the
        next launch selects: ``megamoe_stage_inputs``, or with ``route`` (a load balancer's,
        ``MoERunContext.route``) the balancer's route launch, which maps the expert ids
        ``topk_idx`` to slots there.

        Returns the activation, its scale factors and the routes, which ``_run_moe`` finds staged.
        Without a staging (inputs read in place, tactic autotuning, SF rows that need padding)
        the route and the quantization launch apart.
        """
        num_tokens, top_k = topk_idx.shape
        hidden = x.shape[1]
        staged = None
        if megamoe_activation_sf_bytes_per_row(hidden) == ceil_div(hidden, 16):
            staged = self._staging_of_next_run(num_tokens, x.device)
        if staged is None:
            if route is not None:
                topk_idx = route()
            return (*self.quantize_input(x), topk_idx)
        bufs = staged
        x_fp4, x_sf = bufs.activation[:num_tokens], bufs.activation_sf[:num_tokens]
        quantize = (self._activation_to_quantize(x), self.fc31_input_scale, x_fp4, x_sf)
        self._quantized_into = _QuantizedRows(x_fp4, x_sf, bufs.activation, num_tokens)
        shared_target = self._shared_activation_target(num_tokens)
        if shared_target is not None and num_tokens > 0:
            mxfp8_quantize_megamoe(
                x.to(torch.bfloat16).contiguous(), shared_target[1], shared_target[2]
            )
            self._shared_quantized = (shared_target[0], num_tokens)
        if route is not None:
            return x_fp4, x_sf, route(quantize=quantize)
        if self._slot_of_expert is not None:
            topk_idx = self.slots_of_experts(topk_idx)
        torch.ops.trtllm.megamoe_stage_inputs(
            *quantize,
            topk_idx,
            topk_weights,
            bufs.topk_idx_local,
            bufs.topk_weights,
            live_rows.current_word(x.device)
            if bufs.topk_idx_local.shape[0] in self._padding_drop_buckets
            else None,
        )
        staged_routes = bufs.topk_idx_local[:num_tokens]
        # The routes are staged with their gate weights, as a route_staging caller stages them.
        self._handed_out_route_staging = (staged_routes, bufs.topk_idx_local)
        # Static IDs have already been mapped above; do not map the staged slots again.
        if self._slot_of_expert is not None:
            self._prestaged = (bufs.topk_idx_local, staged_routes, topk_weights, x_fp4, x_sf)
        return x_fp4, x_sf, staged_routes

    def _replica_plan_launch_args(self, replica_plan):
        """Validate and unpack the caller-owned per-forward replica plan."""
        if replica_plan is None:
            return None
        resident_experts = int(replica_plan.resident_experts_per_rank)
        compute_slots = int(replica_plan.compute_slots_per_rank)
        helper_slots = compute_slots - resident_experts
        expected_resident = int(self.expert_size_per_partition)
        expected_compute = self.compute_slot_count
        if resident_experts != expected_resident or compute_slots != expected_compute:
            raise RuntimeError(
                "MegaMoECuteDsl replica-plan geometry does not match its allocated "
                f"weights: plan H/M={resident_experts}/{compute_slots}, "
                f"backend H/M={expected_resident}/{expected_compute}."
            )
        generation = int(replica_plan.ready_generation)
        reserved_sms = int(replica_plan.reserved_sms)
        if helper_slots <= 0 or not 1 <= generation < (1 << 63) or reserved_sms <= 0:
            raise RuntimeError(
                "MegaMoECuteDsl received an invalid replica plan: "
                f"S={helper_slots}, generation={generation}, "
                f"reserved_sms={reserved_sms}."
            )
        return (
            compute_slots,
            helper_slots,
            replica_plan.ready_flags,
            generation,
            reserved_sms,
        )

    def _wait_rebalance_routes(self, replica_plan=None) -> None:
        """Order route consumers after HALO-Q.

        The activation, the gate weights and both tails stage before this wait, overlapping the
        HALO-Q copy; only the live rows' slot ids stage after it."""
        if replica_plan is not None:
            replica_plan.wait_for_routes()

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run the fused MegaMoE CuteDSL kernel on pre-quantized inputs.

        Delegates the staging + kernel launch to :meth:`_run_moe`; staging widens
        the ``int32`` expert ids to the kernel's index type.
        """
        del workspace  # The symmetric buffer is this backend's own workspace.
        x = ctx.x
        token_selected_experts = ctx.token_selected_experts
        token_final_scales = ctx.token_final_scales
        x_sf = ctx.x_sf
        replica_plan = getattr(ctx, "replica_plan", None)
        route = getattr(ctx, "route", None)
        output_dtype = ctx.output_dtype
        if output_dtype is None:
            output_dtype = self.dtype or torch.bfloat16
        self._active_launch_output_dtype = output_dtype
        if x_sf is None and not self.supports_fused_prepare():
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

        topk_weights_f32 = token_final_scales
        if topk_weights_f32.dtype != torch.float32 or not topk_weights_f32.is_contiguous():
            topk_weights_f32 = topk_weights_f32.to(torch.float32).contiguous()
        if x_sf is None:
            x, x_sf, token_selected_experts = self._stage_bf16_inputs(
                x, token_selected_experts, topk_weights_f32, route
            )
        elif route is not None:
            token_selected_experts = route()

        # Routes stay as produced: the route staging widens int32 to the
        # kernel's index dtype, and it is the first route reader.
        return self._run_moe(
            x=x,
            x_sf=x_sf,
            topk_idx=token_selected_experts,
            topk_weights=topk_weights_f32,
            weight_view=weight_view,
            num_tokens=num_tokens,
            top_k=top_k,
            hidden=hidden,
            device=device,
            output_dtype=output_dtype,
            replica_plan=replica_plan,
            before_launch=getattr(ctx, "before_launch", None),
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
        cache_key = _StagingKey(max_T, top_k, hidden, str(device), output_dtype)
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

            properties = query_megamoe_kernel_properties(
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
            if properties.supports_direct_inputs:
                self._genphase_direct_input_buckets.add(max_T)
            if properties.masks_invalid_routes:
                self._padding_drop_buckets.add(max_T)
            # zeros (not empty): the leading counter prefix is an atomic-add
            # target the kernel only tail-resets for the NEXT launch, so the
            # FIRST launch needs it pre-zeroed (one-time; cached per bucket).
            staging["shared_workspace"] = torch.zeros(
                properties.shared_workspace_bytes, dtype=torch.uint8, device=device
            )
        cached[cache_key] = staging
        return staging

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
        Each bundle is built once; views made under inference mode cannot be
        updated in place outside it, so the mode is part of the key.
        """
        key = (
            launch_max_T,
            top_k,
            hidden,
            device,
            output_dtype,
            torch.is_inference_mode_enabled(),
        )
        bufs = self._launch_buffers.get(key)
        if bufs is None:
            bufs = self._build_buffers(
                top_k=top_k,
                hidden=hidden,
                device=device,
                output_dtype=output_dtype,
                launch_max_T=launch_max_T,
            )
            self._launch_buffers[key] = bufs
        return bufs

    def _build_buffers(
        self, *, top_k: int, hidden: int, device, output_dtype, launch_max_T: int
    ) -> _MegaMoeBuffers:
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
                peer_offsets=(0,),
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
        regions = provider.get_regions()
        return _MegaMoeBuffers(
            activation=regions.activation,
            activation_sf=regions.activation_sf,
            topk_weights=regions.topk_weights,
            combine_output=regions.combine_output,
            shared_workspace=regions.shared_workspace,
            peer_offsets=regions.peer_offsets,
            topk_idx_local=staging["topk_idx"],
        )

    def _stage_inputs(
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
        replica_plan=None,
        live_row_word: Optional[torch.Tensor] = None,
    ) -> None:
        """Copy live rows of the user-domain inputs into the kernel's
        pre-allocated buffers and reset the padded tail.

        Same code for single-rank (writes land in local CUDA) and
        multi-rank (writes land in symmetric heap regions visible to
        peers). The buffer source is selected upstream by
        :meth:`_acquire_buffers`.

        One op stages the routes of the whole bucket: rows ``[0,
        num_tokens)`` of ``topk_idx_local`` and ``topk_weights`` get the
        slot ids (widened to the kernel's index dtype) and gate weights,
        rows ``[num_tokens, max_T)`` get ``-1`` and ``0.0``. The kernel
        routes every row whose slot id is valid, and the combine writes
        one cell per ``(token, k in [0, top_k))`` whatever the id, so
        both tails are rewritten on every launch. The op depends on
        ``num_tokens`` alone, so a CUDA graph replay resets rows that an
        eager launch into the same bucket wrote after the capture.

        With ``live_row_word`` (the forward's word, ``moe/live_rows.py``),
        the op reads it on the device and stages only the leading live
        rows' routes; the rows of padding requests get ``-1`` and ``0.0``
        like the tail, so the kernel neither routes nor computes them.

        With a ``replica_plan`` the staging splits around its route wait:
        the gate weights and both tails before it, the live rows' slot ids
        (at most ``num_tokens`` rows) after it.
        """
        if topk_idx.shape[0] != num_tokens:
            raise ValueError(
                f"MegaMoECuteDsl staging got {topk_idx.shape[0]} route rows for {num_tokens} tokens"
            )
        if stage_activation and num_tokens > 0:
            bufs.activation[:num_tokens].copy_(x.view(torch.uint8), non_blocking=True)
            bufs.activation_sf[:num_tokens].copy_(x_sf.view(torch.uint8), non_blocking=True)
        if replica_plan is None:
            self._wait_rebalance_routes(replica_plan)  # nothing to wait for
            torch.ops.trtllm.megamoe_stage_routes(
                topk_idx, topk_weights, bufs.topk_idx_local, bufs.topk_weights, live_row_word
            )
            return
        # A replica plan's HALO-Q kernel rewrites the routes: the gate weights and both tails
        # stage while it runs, and only the live rows' slot ids wait for its routes.
        torch.ops.trtllm.megamoe_stage_routes(
            topk_idx,
            topk_weights,
            bufs.topk_idx_local,
            bufs.topk_weights,
            live_row_word,
            live_ids=False,
        )
        self._wait_rebalance_routes(replica_plan)
        torch.ops.trtllm.megamoe_stage_routes(
            topk_idx,
            topk_weights,
            bufs.topk_idx_local,
            bufs.topk_weights,
            live_row_word,
            rest=False,
        )

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
        replica_plan=None,
        staged_inputs: bool = False,
        before_launch=None,
    ) -> torch.Tensor:
        """Launch the fused MegaMoE CuteDSL kernel and return its reduced output.

        Single-rank and multi-rank reach this point with identical kernel
        inputs; only the source of the staged buffers differs (decided
        upstream by :meth:`_acquire_buffers`). NVFP4 / FP8-SF dtype views
        happen through the module-level :func:`_as_nvfp4` /
        :func:`_as_fp8_sf` helpers (the kernel rejects raw uint8 byte
        tensors). ``staged_inputs`` states that activation, activation_sf,
        topk_idx and topk_weights are this bucket's launch buffers, whose cute
        views the op may then reuse. ``before_launch`` is
        ``MoERunContext.before_launch``, issued right before the op.
        """
        # Hand the symmetric profiling scratch (or a deferred factory) to the
        # op for tuning-mode profiling launches; cleared in the ``finally`` so
        # a later same-process call for a different shape never sees a stale
        # scratch (ownership: see ``_profiling_scratch_provider_ref``).
        from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
            run_cute_dsl_megamoe_nvfp4_blackwell,
            set_active_megamoe_cute_buffer_cache,
            set_active_megamoe_cute_weight_cache,
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
        shared_kwargs = {}
        if self._shared_expert is not None:
            shared_activation, shared_activation_sf = shared_activation_buffers(
                launch_max_T, hidden, combine_output.device
            )
            shared_kwargs = dict(
                shared_fc1_weight=self.shared_fc1_weight,
                shared_fc1_weight_sf=self.shared_fc1_weight_sf,
                shared_fc2_weight=self.shared_fc2_weight,
                shared_fc2_weight_sf=self.shared_fc2_weight_sf,
                shared_activation=shared_activation,
                shared_activation_sf=shared_activation_sf,
                shared_num_tokens=num_tokens,
                shared_intermediate_size=self._shared_expert["intermediate_size"],
                shared_gate_up_clamp=self._shared_expert["gate_up_clamp"],
            )
            if num_tokens > 0:
                # The bucket's combine buffer is overwritten by the next layer, and no add
                # copies the result out of it any more. The kernel ABI requires the output's
                # row count to keep the bucket's divisibility; rows past num_tokens are dropped.
                combine_output = torch.empty(
                    (pad_up(num_tokens, math.gcd(launch_max_T, 128)), 1, hidden),
                    dtype=combine_output.dtype,
                    device=combine_output.device,
                )

        set_active_megamoe_profiling_scratch(
            scratch_provider.get_regions()
            if (scratch_provider is not None and world_size > 1)
            else None
        )
        set_active_megamoe_profiling_scratch_factory(scratch_factory)
        # READY generation comes only from the caller-owned replica plan.
        # Helper-free launches (OFF and internal bucket priming) keep the
        # M-shaped allocation but compile out the READY gate with S=0.
        helper_ready_flags = None
        helper_ready_generation = 0
        compute_slots = self.compute_slot_count
        plan_args = self._replica_plan_launch_args(replica_plan)
        if plan_args is None:
            _S_active = 0
        else:
            (
                compute_slots,
                _S_active,
                helper_ready_flags,
                helper_ready_generation,
                _reserved_sms,
            ) = plan_args
        # Quantization, independent staging, output clears, and host setup
        # are already enqueued. The router inside this op first reads routes.
        self._wait_rebalance_routes(replica_plan)
        if before_launch is not None:
            before_launch()
        # Eager launches call the op's implementation directly: the custom-op
        # dispatch costs tens of microseconds of host time per layer.
        megamoe_op = (
            torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell
            if torch.compiler.is_compiling()
            else run_cute_dsl_megamoe_nvfp4_blackwell
        )
        # An adaptive balancer's routes staged as expert ids: the router of this launch maps and counts them.
        adaptive_route, self._adaptive_route_launch = self._adaptive_route_launch, None
        adaptive_kwargs = {}
        if adaptive_route is not None:
            table_row, rows, forwards, enabled, route_tokens, first = adaptive_route
            word = (
                live_rows.current_word(combine_output.device)
                if launch_max_T in self._padding_drop_buckets
                else live_rows.all_rows_word(combine_output.device)
            )
            adaptive_kwargs = dict(
                adaptive_route_experts=int(self.num_experts),
                adaptive_route_tables=table_row,
                adaptive_route_rows=rows,
                adaptive_route_forwards=forwards,
                adaptive_route_enabled=enabled,
                adaptive_route_live_word=word,
                adaptive_route_tokens=int(route_tokens),
                adaptive_route_first=bool(first),
            )
        set_active_megamoe_cute_weight_cache(self._cute_weight_cache)
        # The provider owns its symmetric regions, so it owns their cute views.
        provider = self._symm_providers.get(launch_max_T) if staged_inputs else None
        set_active_megamoe_cute_buffer_cache(provider.cute_views if provider is not None else None)
        try:
            megamoe_op(
                activation=_as_nvfp4(activation),
                activation_sf=_as_fp8_sf(activation_sf),
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                fc1_weight=_as_nvfp4(weight_view.fc1_weight),
                fc1_weight_sf=_as_fp8_sf(weight_view.fc1_weight_sf),
                fc2_weight=_as_nvfp4(weight_view.fc2_weight),
                fc2_weight_sf=_as_fp8_sf(weight_view.fc2_weight_sf),
                fc1_alpha=weight_view.fc31_alpha,
                fc2_alpha=weight_view.fc2_alpha,
                fc1_norm_const=weight_view.fc1_norm_const,
                combine_output=combine_output,
                shared_workspace=shared_workspace,
                world_size=world_size,
                local_rank=local_rank,
                num_topk=top_k,
                num_experts_per_rank=compute_slots,
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
                **shared_kwargs,
                # Helper-free launches omit the READY gate.
                helper_expert_count=_S_active,
                hot_expert_weight_ready_flags=helper_ready_flags,
                hot_expert_weight_ready_generation=int(helper_ready_generation),
                **adaptive_kwargs,
            )
        finally:
            # Always clear: a raising op call must not leave the factory
            # global alive (its closure strongly references this module).
            set_active_megamoe_profiling_scratch(None)
            set_active_megamoe_profiling_scratch_factory(None)
            set_active_megamoe_cute_weight_cache(None)
            set_active_megamoe_cute_buffer_cache(None)
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
            self.compute_slot_count,
            top_k,
            self.combine_format,
            self.act_clamp,
            self.act_alpha,
            self.act_beta,
            self.apply_topk_in_fc1,
            str(device),
            output_dtype,
            None
            if self._shared_expert is None
            else (self._shared_expert["intermediate_size"], self._shared_expert["gate_up_clamp"]),
        )
        if key in _MEGAMOE_PRIMED_LADDERS:
            return
        primed = []
        self._prime_ladder_buckets(
            live_T=live_T,
            top_k=top_k,
            hidden=hidden,
            device=device,
            output_dtype=output_dtype,
            weight_view=weight_view,
            primed=primed,
        )
        _MEGAMOE_PRIMED_LADDERS.add(key)
        if self.ep_rank == 0:
            logger.debug(
                "[MegaMoECuteDsl] adaptive bucket priming complete: count=%s",
                len(primed),
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
        """Prime each max-token bucket."""
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
            # A launch without rows: empty routes, so the op writes -1 and 0 into every row of the
            # bucket.
            self._stage_inputs(
                bufs=bufs,
                x=None,
                x_sf=None,
                topk_idx=bufs.topk_idx_local.new_empty((0, top_k)),
                topk_weights=bufs.topk_weights.new_empty((0, top_k)),
                num_tokens=0,
                top_k=top_k,
                stage_activation=False,
            )
            # Bucket priming has no replica plan and therefore compiles the S=0 path.
            self._launch_megamoe_kernel(
                activation=bufs.activation,
                activation_sf=bufs.activation_sf,
                topk_idx=bufs.topk_idx_local,
                topk_weights=bufs.topk_weights,
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

    def _run_moe(
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
        replica_plan=None,
        before_launch=None,
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
        launch_max_T, quantized_into = self._take_launch_preparation()
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
        if self._shared_expert is not None:
            quantized, self._shared_quantized = self._shared_quantized, None
            if num_tokens > 0 and quantized != (launch_max_T, num_tokens):
                raise RuntimeError(
                    f"MegaMoECuteDsl: the fused shared expert's MXFP8 input was written for "
                    f"(bucket, tokens) = {quantized}, but this launch is ({launch_max_T}, "
                    f"{num_tokens}); route + quantize or quantize_input must precede run_moe."
                )
        if num_tokens == 0 and self.ep_size == 1:
            self._prestaged = None
            self._handed_out_route_staging = None
            return torch.empty((0, hidden), dtype=output_dtype, device=device)

        if (
            self.ep_size > 1
            and not self.tactic_autotune
            and _MEGAMOE_PRIME_LADDER
            and AutoTuner.get().is_tuning_mode
        ):
            # The ladder launches the fused kernel too (MoERunContext.before_launch).
            if before_launch is not None:
                before_launch()
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
        prestaged, self._prestaged = self._prestaged, None
        # The fused route+quant launch already wrote exactly these inputs into this bucket, as expert
        # slots when a static placement maps them.
        inputs_staged = (
            prestaged is not None
            and prestaged[0] is bufs.topk_idx_local
            and all(a is b for a, b in zip((topk_idx, topk_weights, x, x_sf), prestaged[1:]))
        )
        # Validate the handed-out slot view before any conversion can discard its identity.
        routes_staged = inputs_staged
        if not inputs_staged or self._handed_out_route_staging is not None:
            routes_staged = self._routes_staged(bufs, topk_idx, num_tokens, top_k, replica_plan)
        if not routes_staged:
            if replica_plan is None and (
                topk_idx.dtype != torch.int32 or not topk_idx.is_contiguous()
            ):
                topk_idx = topk_idx.to(torch.int32).contiguous()
            if self._slot_of_expert is not None:
                topk_idx = self.slots_of_experts(topk_idx)
        use_direct_inputs = (
            not inputs_staged
            and self._may_read_inputs_in_place(num_tokens, launch_max_T)
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
        # quantize_input already quantized into this bucket's staging: x and x_sf are its
        # leading rows, so there is nothing to copy. The very views it returned need no storage
        # check.
        activation_staged = (
            quantized_into is not None
            and x is quantized_into.x
            and x_sf is quantized_into.x_sf
            and quantized_into.activation is bufs.activation
        )
        if not activation_staged:
            activation_staged = _is_staging_view(
                x, bufs.activation, num_tokens
            ) and _is_staging_view(x_sf, bufs.activation_sf, num_tokens)
            recorded = (
                None
                if quantized_into is None
                else (quantized_into.activation.data_ptr(), quantized_into.rows)
            )
            # Both sides decide the bucket: a mismatch would copy the rows again, or launch stale
            # ones.
            if not inputs_staged and recorded != (
                (bufs.activation.data_ptr(), num_tokens) if activation_staged else None
            ):
                raise RuntimeError(
                    f"MegaMoECuteDsl: quantize_input quantized (staging address, rows) "
                    f"{recorded}, but run_moe launches {num_tokens} rows at {x.data_ptr():#x} "
                    f"into bucket {launch_max_T}, whose staging is at "
                    f"{bufs.activation.data_ptr():#x}. Set the bucket hint before quantize_input "
                    f"and run_moe on its output."
                )
        if routes_staged:
            if not use_direct_inputs and not activation_staged and num_tokens > 0:
                bufs.activation[:num_tokens].copy_(x.view(torch.uint8), non_blocking=True)
                bufs.activation_sf[:num_tokens].copy_(x_sf.view(torch.uint8), non_blocking=True)
        elif not use_direct_routing_inputs:
            self._stage_inputs(
                bufs=bufs,
                x=x,
                x_sf=x_sf,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_tokens=num_tokens,
                top_k=top_k,
                stage_activation=not use_direct_inputs and not activation_staged,
                replica_plan=replica_plan,
                live_row_word=(
                    live_rows.current_word(device)
                    if launch_max_T in self._padding_drop_buckets
                    else None
                ),
            )
        return self._launch_megamoe_kernel(
            activation=x if use_direct_inputs else bufs.activation,
            activation_sf=x_sf if use_direct_inputs else bufs.activation_sf,
            topk_idx=topk_idx if use_direct_routing_inputs else bufs.topk_idx_local,
            topk_weights=topk_weights if use_direct_routing_inputs else bufs.topk_weights,
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
            replica_plan=replica_plan,
            # Direct routing inputs imply direct activation inputs.
            staged_inputs=not use_direct_inputs,
            before_launch=before_launch,
        )


# An alias, not a base class, so there is no second class to keep in step.
MegaMoECuteDsl = TrtllmCutedslMegaMoeNvfp4Impl
