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
``Sm100MegaMoEKernel`` (fused dispatch + FC1 + activation + FC2 +
combine) from
``tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4``. The kernel is
invoked through the standard CuteDSL TunableRunner / torch op pattern;
the runner + op live in
``tensorrt_llm/_torch/custom_ops/cute_dsl_megamoe_custom_op.py``. This
file only owns:

  * capability gating (``can_implement``)
  * lifecycle hooks (``__init__`` / ``create_weights`` /
    ``load_weights`` / ``post_load_weights`` /
    ``validate_configurable_moe``)
  * EP process group resolution
  * BF16 -> NVFP4 activation quantization (``quantize_input``)
  * ``run_moe`` boundary: stage activation + topk into the kernel ABI,
    build the ``MegaMoECuteDslWeightView`` from the quant method, call
    ``torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell``, sum the
    per-topk axis (form A), return ``(T, hidden)`` output.

``run_moe`` is a single unified path for both topologies. Only the
SOURCE of the kernel's input/output buffers branches on ``ep_size``:

  * ``ep_size == 1``: local CUDA tensors (cudaMalloc). No
    ``torch.distributed`` dependency, no rendezvous, no cuMem VMM
    overhead. ``peer_offsets = [0]`` collapses the kernel's
    ``peer_rank_ptr_mapper.map(local_addr, 0, off) == local_addr +
    off`` to a self-mapped pointer (NVSHMEM degenerate convention).
  * ``ep_size > 1``: regions carved out of the build-time-rendezvous'd
    :class:`~tensorrt_llm._torch.custom_ops.cute_dsl_megamoe_custom_op.MegaMoeSymmMemProvider`
    symmetric buffer; ``peer_offsets[r] = peer_base[r] - local_base``
    enables in-kernel cross-GPU NVLink load/store via
    ``peer_rank_ptr_mapper.map``.

``_acquire_buffers`` is the only branch point; staging, kernel launch,
and the host-side top-k reduction are identical across topologies.

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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

# ``megamoe_activation_sf_bytes_per_row`` lives at module top of the
# custom-op file (NOT inside its ``IS_MEGAMOE_OP_AVAILABLE`` gate), so
# it is always importable. The provider / shared-workspace helpers used
# in ``_alloc_symm_provider`` and ``_ensure_local_staging`` ARE inside
# that gate and therefore stay lazy at the call site.
from ....custom_ops.cute_dsl_megamoe_custom_op import megamoe_activation_sf_bytes_per_row
from ....cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType, Fp4QuantizedTensor
from ..interface import MoE, MoESchedulerKind, MoEWeightLoadingMode
from ..quantization import NVFP4MegaMoECuteDslMethod
from ..routing import BaseMoeRoutingMethod

__all__ = [
    "MegaMoECuteDsl",
    "MegaMoeCuteDslUnavailable",
    "MegaMoECuteDslWeightView",
    "is_megamoe_cute_dsl_runtime_available",
]


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


class MegaMoeCuteDslUnavailable(RuntimeError):
    """Raised when the active environment cannot import the symbols required by
    the ported ``Sm100MegaMoEKernel`` (cu13 Cutlass DSL + cute_nvgpu MMA
    atoms / cutlass._mlir APIs used by sym_buffer)."""


_RUNTIME_PROBE_CACHE: Optional[Union[bool, str]] = None


def is_megamoe_cute_dsl_runtime_available() -> Tuple[bool, Optional[str]]:
    """Return whether the CUDA 13 Cutlass DSL runtime exposes all symbols the
    ported MegaMoE CuteDSL kernel needs.

    Stricter than ``IS_CUTLASS_DSL_AVAILABLE``, which only confirms that
    ``cutlass`` / ``cutlass.cute`` import cleanly. The MegaMoE kernel
    ABI also requires ``cutlass.torch.from_dlpack``, ``cutlass._mlir``
    APIs used by ``sym_buffer.py``, the ``cute_nvgpu`` MMA atoms used
    by ``kernel_fc12.py``, and the async-copy helpers used by
    ``dispatch_kernel.py``. PR
    https://github.com/NVIDIA/TensorRT-LLM/pull/14354 pins
    ``nvidia-cutlass-dsl[cu13]==4.5.0`` which is the first release that
    ships all of them; older wheels return ``(False, reason)``.

    Returns ``(True, None)`` on success or ``(False, reason)`` with an
    actionable message. The result is cached for the process lifetime.
    """
    global _RUNTIME_PROBE_CACHE
    if _RUNTIME_PROBE_CACHE is True:
        return True, None
    if isinstance(_RUNTIME_PROBE_CACHE, str):
        return False, _RUNTIME_PROBE_CACHE

    if not IS_CUTLASS_DSL_AVAILABLE:
        reason = (
            "Cutlass DSL is not importable on this environment; install "
            "nvidia-cutlass-dsl[cu13] to enable MegaMoECuteDsl."
        )
        _RUNTIME_PROBE_CACHE = reason
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
        _RUNTIME_PROBE_CACHE = reason
        return False, reason

    try:
        from cutlass.cute.nvgpu import cpasync, tcgen05  # noqa: F401
    except ImportError as e:
        reason = (
            f"MegaMoECuteDsl requires cutlass.cute.nvgpu.tcgen05 + cpasync; "
            f"missing {e!r}. Install a Blackwell-capable cutlass-dsl wheel."
        )
        _RUNTIME_PROBE_CACHE = reason
        return False, reason

    try:
        # mega_moe_cute_dsl.py lives at
        # tensorrt_llm/_torch/modules/fused_moe/mega_moe/mega_moe_cute_dsl.py;
        # four dots take us back to tensorrt_llm._torch where
        # cute_dsl_kernels.mega_moe_nvfp4 is registered.
        from ....cute_dsl_kernels.mega_moe_nvfp4 import (  # noqa: F401
            Nvfp4BlockSize,
            SfPaddingBlock,
            to_blocked,
        )
    except ImportError as e:
        reason = (
            f"Ported MegaMoE NVFP4 kernel package failed to import: "
            f"{e!r}. Verify tensorrt_llm/_torch/cute_dsl_kernels/"
            f"mega_moe_nvfp4 is in the install tree."
        )
        _RUNTIME_PROBE_CACHE = reason
        return False, reason

    _RUNTIME_PROBE_CACHE = True
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


# ---------------------------------------------------------------------------
# Weight view passed to ``run_moe``
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MegaMoECuteDslWeightView:
    """Bundles the MegaMoE-format weight tensors built by
    ``NVFP4MegaMoECuteDslMethod.process_weights_after_loading``.

    The kernel reads these as local-only (NOT through symmetric heap);
    placement is unconstrained CUDA memory. Shapes match the
    ``Sm100MegaMoEKernel.__call__`` ABI.

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
    topk_idx_local: torch.Tensor  # (max_T, top_k) int64, always-local


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class MegaMoECuteDsl(MoE):
    """MoE backend wrapping the ported MegaMoE CuteDSL NVFP4 fused kernel.

    Capability gate (``can_implement``): SM100 family + NVFP4 +
    bfloat16 activation + CUDA 13 Cutlass DSL runtime present.

    Topology source-of-truth: :meth:`_acquire_buffers`.

      * ``ep_size == 1``: local CUDA tensors and ``peer_offsets = [0]``,
        which collapses the kernel's ``peer_rank_ptr_mapper.map(local,
        0, off)`` to a self-mapped pointer (NVSHMEM degenerate).
      * ``ep_size > 1``: regions carved out of
        :class:`~tensorrt_llm._torch.custom_ops.cute_dsl_megamoe_custom_op.MegaMoeSymmMemProvider`'s
        rendezvous'd symmetric buffer. ``create_weights`` performs the
        (collective) ``torch_symm_mem.rendezvous`` at build time so
        forward time stays free of cross-rank IPC. ``run_moe`` raises
        :class:`MegaMoeCuteDslUnavailable` if the provider was not
        allocated (e.g. ``torch.distributed`` not initialised).
    """

    _SUPPORTED_ACTIVATION_DTYPES = frozenset({torch.bfloat16})

    # Kernel owns dispatch + GEMM1 + SwiGLU + GEMM2 + combine via the
    # CuteDSL three-stage dispatch primitives + NVLink barrier; the
    # scheduler must skip host-side comm and lockstep every chunk.
    scheduler_kind = MoESchedulerKind.FUSED_COMM

    # ------------------------------------------------------------------
    # Capability gating
    # ------------------------------------------------------------------
    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Static capability query: SM/dtype/quant/shape only.

        Does NOT probe checkpoint tensor values. The kernel ABI consumes
        per-expert scales directly, so there is no checkpoint-value
        rejection for non-1 alpha products. The SwiGLU
        clamp (``swiglu_limit``) is validated for uniformity in
        ``__init__`` (``_resolve_gate_up_clamp``), not here, because
        ``can_implement`` is a static query that does not see per-tensor
        checkpoint values.

        Multi-rank execution gate (NVSHMEM provider) is NOT in this
        query either, by analogy to ``MegaMoEDeepGemm.can_implement``;
        ``run_moe`` is where the provider absence becomes a hard error
        for ``ep_size > 1`` topologies.
        """
        sm = get_sm_version()
        if not is_sm_100f(sm):
            return False, (f"MegaMoECuteDsl requires SM100 family (SM100 or SM103); got SM{sm}.")
        if dtype_activation not in cls._SUPPORTED_ACTIVATION_DTYPES:
            return False, (
                f"MegaMoECuteDsl supports activations in "
                f"{cls._SUPPORTED_ACTIVATION_DTYPES}, got {dtype_activation}."
            )
        if swiglu_gptoss_style:
            return False, "MegaMoECuteDsl does not support swiglu_gptoss_style."
        if quant_algo != QuantAlgo.NVFP4:
            return False, (f"MegaMoECuteDsl supports NVFP4 only, got quant_algo={quant_algo}.")
        # ``hidden_size % 32`` covers the kernel's NVFP4 SF leg
        # alignment; the SF row width is padded to
        # ``round_up(ceil(hidden/16), 4)`` at every allocation site (see
        # ``megamoe_activation_sf_bytes_per_row``).
        if hidden_size is not None and (hidden_size <= 0 or hidden_size % 32 != 0):
            return False, (
                f"MegaMoECuteDsl requires positive hidden_size divisible "
                f"by 32 (NVFP4 SF leg alignment); got {hidden_size}."
            )
        # The kernel's expand_intermediate = 2 * intermediate must be
        # divisible by 2 * Fc1GateUpInterleave (32) -> intermediate % 16.
        if intermediate_size is not None and (
            intermediate_size <= 0 or intermediate_size % 16 != 0
        ):
            return False, (
                f"MegaMoECuteDsl requires positive intermediate_size "
                f"divisible by 16 (Fc1GateUpInterleave); got "
                f"{intermediate_size}."
            )
        ok, reason = is_megamoe_cute_dsl_runtime_available()
        if not ok:
            return False, reason
        # The fused path also requires the ``trtllm::cute_dsl_megamoe_nvfp4_*``
        # custom op to be registered (strict import of every kernel symbol in
        # cute_dsl_megamoe_custom_op). Read the flag dynamically from the
        # custom-op module so it reflects the live registration state.
        from ....custom_ops import cute_dsl_megamoe_custom_op as _megamoe_op

        if not _megamoe_op.IS_MEGAMOE_OP_AVAILABLE:
            return False, _megamoe_op.MEGAMOE_OP_UNAVAILABLE_REASON
        return True, None

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
        init_load_balancer: bool = True,
        without_comm: bool = False,
        activation_type: ActivationType = ActivationType.Swiglu,
        swiglu_limit: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        # ``aux_stream_dict`` is accepted for ``create_moe_backend`` signature
        # uniformity but ignored: FUSED_COMM kernels must not use the chunk
        # overlap stream because launch order must be lockstep across EP.
        del aux_stream_dict
        super().__init__(
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
            activation_type=activation_type,
            swiglu_limit=swiglu_limit,
            init_load_balancer=init_load_balancer,
        )

        # Constructor-time invariant checks raise ValueError so that
        # Python ``-O`` (which strips ``assert``) does not silently let an
        # invalid topology through.
        if self.tp_size != 1:
            raise ValueError(
                f"MegaMoECuteDsl is EP-only (moe_tp_size=1); got tp_size={self.tp_size}."
            )
        if self.cluster_size != 1:
            raise ValueError(
                f"MegaMoECuteDsl assumes cluster_size=1; got cluster_size={self.cluster_size}."
            )
        if self.num_slots % max(self.ep_size, 1) != 0:
            raise ValueError(
                f"MegaMoECuteDsl requires num_slots ({self.num_slots}) "
                f"divisible by ep_size ({self.ep_size})."
            )

        if self.use_dp and self.parallel_size > 1 and self.ep_size != self.parallel_size:
            raise ValueError(
                f"MegaMoECuteDsl with enable_attention_dp=True requires "
                f"ep_size == parallel_size (got ep_size={self.ep_size}, "
                f"parallel_size={self.parallel_size}). ADP > EP would "
                f"require an outer allgather + reducescatter wrapper."
            )

        if apply_router_weight_on_input:
            raise ValueError(
                "MegaMoECuteDsl does not support apply_router_weight_on_input; "
                "the fused kernel applies routing weights on the MoE output."
            )
        if activation_type != ActivationType.Swiglu:
            raise ValueError(
                f"MegaMoECuteDsl only supports ActivationType.Swiglu (got {activation_type})."
            )
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # topk-score application point. v2 default is the deepgemm graph
        # (apply_topk_in_fc1=True): the fused kernel folds the topk score into
        # the SwiGLU output before the fc1-out NVFP4 quant and the host reduces
        # combine_output.sum(dim=1). Kept as an internal backend constant until
        # the transformers route is GPU-validated and promoted to MoeConfig.
        self.apply_topk_in_fc1 = True

        # Cross-rank combine path. ``False`` (default): the FC2 epilogue writes
        # ``combine_output`` directly (scattered symmetric writes back to the
        # source rank). ``True``: the kernel stages FC2 output in a local
        # ``fc2_output_workspace`` and a fused in-kernel NVLink ``token_back_by_push``
        # bulk-returns it to the source rank's ``combine_output`` -- faster for
        # multi-rank EP at the cost of the extra (local) fc2_output_workspace +
        # fc2_done_counter budget (auto-sized by ``get_workspace_sizes``). The
        # ``combine_output`` shape / host ``.sum(dim=1)`` reduce are unchanged
        # (those depend on ``in_kernel_fc2_reduce``, not this knob). Internal
        # backend constant for now; flip to opt into the fused-combine path.
        self.token_back_by_dispatch = False

        # FC2 output store path (codegen-time). ``True`` (default): non-bulk
        # TMA store (upstream default). ``False``: bulk store path. Kept as an
        # internal backend attribute so different shapes/cases can pick the
        # cheaper store; it changes the generated kernel, so it is part of the
        # runner ``unique_id`` / compile-cache key (never a per-call runtime kwarg).
        self.non_ubulk_fc2_store = True

        # SwiGLU clamp: map the model-provided per-layer ``swiglu_limit`` tensor
        # to the kernel's codegen-time scalar ``gate_up_clamp``. The MegaMoE
        # kernel clamps the post-fc1_alpha real gate/up, so the model value is
        # used directly (NO trtllm-gen-style div_(fc31_alpha) normalization).
        # Reject non-uniform / per-expert clamp: the kernel bakes one constant.
        self.gate_up_clamp = self._resolve_gate_up_clamp(swiglu_limit)

        # Buffer sizing. MoE layers execute serially per forward; one pool
        # sized to the worst-case per-rank tokens covers every layer. The
        # kernel compile takes this as the static ``max_tokens_per_rank``.
        self.max_num_tokens = int(
            getattr(model_config, "moe_max_num_tokens", 0)
            or getattr(model_config, "max_num_tokens", 0)
            or 4096
        )

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

        # Weight tensors are owned by the quant method. ``_symm_provider``
        # is the symmetric-memory provider for multi-rank EP execution;
        # allocated build-time in ``create_weights`` (collective
        # rendezvous), shared across MoE layers via the module-scope
        # cache in ``cute_dsl_megamoe_custom_op.py``. ``None`` for the
        # single-rank degenerate path.
        self._symm_provider = None
        self._weights_loaded = False
        self._weights_created = False
        self._post_load_done = False
        self.quant_method = None
        # Per-instance staging cache (key -> {tensor name: tensor}) and
        # last-staged-T tracker; together they implement the always-pad-
        # to-max_T launch contract by refreshing only the rows that
        # changed between calls.
        self._local_staging_cache: Dict[Tuple, Dict[str, torch.Tensor]] = {}
        self._last_staged_T: Optional[int] = None
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_gate_up_clamp(
        swiglu_limit: Optional[torch.Tensor],
    ) -> Optional[float]:
        """Reduce a per-layer ``swiglu_limit`` tensor to a single codegen-time
        ``gate_up_clamp`` float, or ``None`` when no clamp is configured.

        The MegaMoE kernel bakes ``gate_up_clamp`` into the compiled kernel as
        one scalar, so only a uniform (per-layer) clamp is representable.
        Non-uniform / per-expert clamp is rejected with a clear ``ValueError``
        rather than silently using one element. GPT-OSS-style clamp is rejected
        earlier in ``can_implement`` via ``swiglu_gptoss_style``.
        """
        if swiglu_limit is None:
            return None
        if not isinstance(swiglu_limit, torch.Tensor):
            # Accept a plain python scalar for robustness.
            return float(swiglu_limit)
        flat = swiglu_limit.detach().reshape(-1)
        if flat.numel() == 0:
            return None
        first = flat[0]
        if flat.numel() > 1 and not torch.allclose(flat, first.expand_as(flat), rtol=1e-5, atol=0):
            raise ValueError(
                "MegaMoECuteDsl only supports a uniform (per-layer) "
                "swiglu_limit because the kernel bakes gate_up_clamp as a "
                "codegen-time scalar; got a non-uniform / per-expert "
                f"swiglu_limit with values {flat.cpu().tolist()}."
            )
        return float(first.item())

    def _supports_load_balancer(self) -> bool:
        # Both static and dynamic EPLB are supported: the four MegaMoE-
        # format derived parameters (``mega_fc{1,2}_weight{,_sf}``) are
        # registered with the load balancer in
        # ``NVFP4MegaMoECuteDslMethod._register_mega_shared_staging``
        # and migrate atomically with the underlying NVFP4 raw weights
        # + scales already handled by the base/grandparent.
        return True

    def validate_configurable_moe(self, moe) -> None:
        """Mirrors :meth:`MegaMoEDeepGemm.validate_configurable_moe`.

        Enforces the MegaMoECuteDsl wrapper-level invariants (EP-only,
        ``moe.comm is None``, ``num_slots % moe_ep_size == 0``,
        ``experts_per_token <= 13``, ``moe_max_num_tokens > 0``) listed
        inline below.

        ``ConfigurableMoE.__init__`` calls this at the very end (after
        ``self.comm`` / ``self.moe_max_num_tokens`` and every EPLB /
        num_slots / ep_size attribute are populated -- see
        ``configurable_moe.py`` ``validate_backend`` docstring), so
        every attribute touched below may be read directly.
        """
        if moe.comm is not None:
            raise ValueError(
                f"MegaMoECuteDsl requires moe.comm is None (FUSED_COMM "
                f"backends must not layer host-side communication on top "
                f"of the fused kernel); got moe.comm={type(moe.comm).__name__}."
            )
        if moe.mapping.moe_tp_size != 1:
            raise ValueError(
                f"MegaMoECuteDsl is EP-only (moe_tp_size=1); got {moe.mapping.moe_tp_size}."
            )
        # NOTE: ``mapping.tp_size`` is the *wrapper-level* TP size used by
        # attention, not by the MoE layer. In DEP / TEP modes the wrapper
        # sets ``tp_size = world_size`` while ``moe_tp_size = 1``; the
        # MegaMoECuteDsl kernel only cares about the MoE axes
        # (``moe_ep_size`` / ``moe_tp_size``) — see
        # ``_create_mapping_for_parallel_mode`` in test_moe_module.py.
        if moe.num_slots % moe.mapping.moe_ep_size != 0:
            raise ValueError(
                f"MegaMoECuteDsl requires num_slots ({moe.num_slots}) "
                f"divisible by moe_ep_size ({moe.mapping.moe_ep_size})."
            )
        if moe.use_dp and moe.parallel_size > 1 and moe.mapping.moe_ep_size != moe.parallel_size:
            raise ValueError(
                f"MegaMoECuteDsl with enable_attention_dp requires "
                f"moe_ep_size == parallel_size (got "
                f"moe_ep_size={moe.mapping.moe_ep_size}, "
                f"parallel_size={moe.parallel_size})."
            )
        top_k = moe.routing_method.experts_per_token
        if top_k > 13:
            raise ValueError(
                f"MegaMoECuteDsl supports experts_per_token <= 13 "
                f"(matches external coverage); got {top_k}."
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
    def _resolve_ep_pg(self):
        """Return the torch.distributed ProcessGroup for the EP sub-world.

        Mirrors :meth:`MegaMoEDeepGemm._resolve_ep_pg` so the two MegaMoE
        backends share the same fallback chain.
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
        self._symm_provider = None
        if self.ep_size > 1:
            self._symm_provider = self._alloc_symm_provider()
        # Step 2-3: quant method registers all NVFP4 + MegaMoE-format params.
        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)
        # Step 4.
        self._weights_created = True

    def _alloc_symm_provider(self):
        """Build-time symmetric provider allocation. See ``create_weights``.

        Returns a :class:`MegaMoeSymmMemProvider` from the module-scope
        cache. Raises :class:`MegaMoeCuteDslUnavailable` with an
        actionable message when no ProcessGroup is available -- that
        would block the rendezvous and is a hard error for multi-rank.
        """
        from ....custom_ops.cute_dsl_megamoe_custom_op import (
            get_megamoe_symm_provider,
            query_megamoe_shared_workspace_bytes,
        )

        if self._ep_pg is None:
            raise MegaMoeCuteDslUnavailable(
                "MegaMoECuteDsl multi-rank requires a torch.distributed EP "
                "ProcessGroup. Use Ray / DeviceMesh (mapping.moe_ep_group_pg) "
                "or initialize torch.distributed before model build."
            )
        top_k = self.routing_method.experts_per_token
        shared_workspace_bytes = query_megamoe_shared_workspace_bytes(
            world_size=self.ep_size,
            local_rank=self.ep_rank,
            num_topk=top_k,
            num_experts_per_rank=int(self.expert_size_per_partition),
            hidden_size=self.hidden_size,
            intermediate_size_per_partition=int(self.intermediate_size_per_partition),
            expand_intermediate_size_per_partition=int(self.expand_intermediate_size_per_partition),
            max_tokens_per_rank=int(self.max_num_tokens),
        )
        return get_megamoe_symm_provider(
            process_group=self._ep_pg,
            world_size=self.ep_size,
            rank=self.ep_rank,
            hidden_size=self.hidden_size,
            max_tokens_per_rank=int(self.max_num_tokens),
            num_topk=top_k,
            output_dtype=self.dtype or torch.bfloat16,
            shared_workspace_bytes=shared_workspace_bytes,
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
        # Eager loading path: ``FusedMoEMethodBase.load_weights`` already
        # ran ``quant_method.process_weights_after_loading(self)`` at its
        # tail. Mark the sentinel so a subsequent
        # ``backend.process_weights_after_loading()`` becomes a no-op
        # instead of re-stacking ``mega_fc*_weight*`` from
        # already-finalised parent buffers.
        if not allow_partial_loading:
            self._post_load_done = True

    def post_load_weights(self) -> None:
        if self.quant_method is None:
            self.create_weights()
        self.transform_weights()
        self.cache_derived_state()

    def process_weights_after_loading(self) -> None:
        """Run quant-method weight transforms; idempotent across calls.

        The real MegaMoE-format build (``[w3|w1]`` cat, 16-atom gate/up
        interleave, ``to_blocked`` swizzle, and ``fc1_norm_const`` setup) lives in
        :meth:`NVFP4MegaMoECuteDslMethod.process_weights_after_loading`.
        This hook must dispatch to that method directly so two paths
        both reach it:

          * Eager loading (``allow_partial_loading=False``) -- fired by
            ``FusedMoEMethodBase.load_weights`` itself.
          * Partial loading (RLHF reload, etc.) -- ``load_weights``
            skips its tail call, so the caller invokes this hook on
            ``ConfigurableMoE`` -> backend to finalise.

        ``_post_load_done`` keeps the call idempotent: a second
        invocation after eager finalisation must not re-run the
        transforms (``_build_mega_format_weights`` would re-stack
        ``mega_fc*_weight*`` from already-finalised parent buffers).
        """
        if getattr(self, "_post_load_done", False):
            return
        if self.quant_method is None:
            self.create_weights()
        self.quant_method.process_weights_after_loading(self)
        self._post_load_done = True

    def pre_reload_weights(self) -> None:
        """Reset cached state before a hot weight reload.

        ``_post_load_done`` is cleared so the next ``process_weights_after_loading``
        re-runs the MegaMoE-format weight transforms over the new
        checkpoint bytes. The symmetric-memory provider is forward-time
        scratch that does not need to be re-rendezvoused on weight
        reload; we keep it as-is to avoid an unnecessary collective.
        """
        self._post_load_done = False
        if self.quant_method is not None and hasattr(self.quant_method, "pre_reload_weights"):
            self.quant_method.pre_reload_weights(self)

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
        x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(
            x_bf16,
            self.fc31_input_scale,
            16,  # scaling_vector_size == Nvfp4BlockSize
            False,  # sf_use_ue8m0
            False,  # is_sf_swizzled - MegaMoE expects plain K-major
        )
        # ``fp4_quantize(is_sf_swizzled=False)`` returns LINEAR layout
        # ``(rows, ceil(hidden/16))`` with no column pad. The kernel TMA
        # load needs ``round_up(ceil(hidden/16), 4)`` bytes per row, so
        # 32-aligned-but-not-64-aligned hidden sizes (1568, 1632, 2080)
        # come back 2 bytes short; pad the tail before returning.
        raw_cols = (hidden + 15) // 16
        x_sf_raw = x_sf.view(x_bf16.shape[0], raw_cols)
        if sf_cols == raw_cols:
            return x_fp4, x_sf_raw
        padded_sf = torch.zeros((x_bf16.shape[0], sf_cols), dtype=torch.uint8, device=x_bf16.device)
        padded_sf[:, :raw_cols] = x_sf_raw
        return x_fp4, padded_sf

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: Optional[torch.Tensor] = None,
        *,
        output_dtype: Optional[torch.dtype] = None,
        **unused_kwargs,
    ) -> torch.Tensor:
        """Run the fused MegaMoE CuteDSL kernel on pre-quantized inputs.

        Casts ``token_selected_experts`` to ``int64`` (the scheduler keeps
        ``int32`` for the EPLB stats kernel; the MegaMoE kernel reads
        ``topk_idx`` as Int64) and delegates the staging + kernel launch
        to :meth:`_run_moe`. The host then sums the form-A
        ``(T, top_k, hidden)`` combine output along the top-k axis.
        """
        del unused_kwargs
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

        topk_idx_i64 = token_selected_experts.to(torch.int64).contiguous()
        topk_weights_f32 = token_final_scales.to(torch.float32).contiguous()

        return self._run_moe(
            x=x,
            x_sf=x_sf,
            topk_idx=topk_idx_i64,
            topk_weights=topk_weights_f32,
            weight_view=weight_view,
            num_tokens=num_tokens,
            top_k=top_k,
            hidden=hidden,
            device=device,
            output_dtype=output_dtype,
        )

    def _ensure_local_staging(self, *, top_k: int, hidden: int, device, output_dtype):
        """Allocate (and cache) the per-instance local staging tensors.

        Always allocates ``topk_idx`` (the kernel reads it as a local-only
        buffer in BOTH topologies; peers never call
        ``peer_rank_ptr_mapper.map`` on it). The user-domain entries --
        ``activation`` / ``activation_sf`` / ``topk_weights`` /
        ``combine_output`` / ``shared_workspace`` -- are allocated only
        for ``ep_size == 1``; multi-rank pulls them from the symmetric
        provider's regions instead.

        All staging tensors are sized to ``max_num_tokens`` along dim 0
        so the kernel's constexpr ``num_tokens`` matches the buffer-time
        ``max_tokens_per_rank``. Diverging the two would make
        ``_dispatch_prep`` round 3 (``MAX_SLOT_C = num_tokens * num_topk``
        in dispatch_kernel.py) write per-(expert, rank) advertise cards
        at the wrong stride relative to the symm allocation
        (``max_tokens_per_rank * num_topk`` in megamoe_kernel.py),
        silently corrupting multi-rank metadata.
        """
        max_T = int(self.max_num_tokens)
        cache_key = (max_T, top_k, hidden, str(device), output_dtype)
        cached = self._local_staging_cache
        if cache_key in cached:
            return cached[cache_key]

        # ``topk_idx`` defaults to -1 so dispatch_prep skips padded tail
        # rows (``if expert_id >= Int32(0):`` in dispatch_kernel.py).
        staging = {
            "topk_idx": torch.full((max_T, top_k), -1, dtype=torch.int64, device=device),
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
            staging["combine_output"] = torch.empty(
                (max_T, top_k, hidden),
                dtype=torch.bfloat16,
                device=device,
            )
            # Shared-workspace probe lives behind ``IS_MEGAMOE_OP_AVAILABLE``
            # in cute_dsl_megamoe_custom_op so the import stays lazy.
            from ....custom_ops.cute_dsl_megamoe_custom_op import (
                query_megamoe_shared_workspace_bytes,
            )

            shared_bytes = query_megamoe_shared_workspace_bytes(
                world_size=1,
                local_rank=0,
                num_topk=top_k,
                num_experts_per_rank=int(self.expert_size_per_partition),
                hidden_size=hidden,
                intermediate_size_per_partition=int(self.intermediate_size_per_partition),
                expand_intermediate_size_per_partition=int(
                    self.expand_intermediate_size_per_partition
                ),
                max_tokens_per_rank=max_T,
            )
            staging["shared_workspace"] = torch.empty(
                shared_bytes, dtype=torch.uint8, device=device
            )
        cached[cache_key] = staging
        return staging

    def _acquire_buffers(self, *, top_k: int, hidden: int, device, output_dtype) -> _MegaMoeBuffers:
        """Resolve the kernel's input/output buffers.

        This is the ONLY structural branch between single-rank and multi-
        rank execution; the source of activation / activation_sf /
        topk_weights / combine_output / shared_workspace differs per the
        :class:`_MegaMoeBuffers` contract. ``topk_idx_local`` always lives
        in plain CUDA memory.
        """
        staging = self._ensure_local_staging(
            top_k=top_k, hidden=hidden, device=device, output_dtype=output_dtype
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
        if self._symm_provider.num_topk != top_k:
            raise MegaMoeCuteDslUnavailable(
                f"MegaMoECuteDsl symm provider was built for top_k="
                f"{self._symm_provider.num_topk} but run_moe called with "
                f"top_k={top_k}; recreate the backend."
            )
        regions = self._symm_provider.get_regions()
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
        last_T = getattr(self, "_last_staged_T", None)
        if last_T is not None and last_T > num_tokens:
            bufs.topk_idx_local[num_tokens:last_T].fill_(-1)
        if num_tokens > 0:
            bufs.topk_idx_local[:num_tokens].copy_(topk_idx, non_blocking=True)
            bufs.activation[:num_tokens].copy_(x.view(torch.uint8), non_blocking=True)
            bufs.activation_sf[:num_tokens].copy_(x_sf.view(torch.uint8), non_blocking=True)
            bufs.topk_weights[:num_tokens, :top_k].copy_(topk_weights, non_blocking=True)
        if num_tokens < max_T:
            bufs.topk_weights[num_tokens:max_T, :top_k].zero_()
        self._last_staged_T = num_tokens

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
    ) -> torch.Tensor:
        """Launch the fused MegaMoE CuteDSL kernel and reduce form-A output.

        Single-rank and multi-rank reach this point with identical kernel
        inputs; only the source of the staged buffers differs (decided
        upstream by :meth:`_acquire_buffers`). The host-side top-k reduction
        is the same across topologies. NVFP4 / FP8-SF dtype views happen
        through the module-level :func:`_as_nvfp4` / :func:`_as_fp8_sf`
        helpers (the kernel rejects raw uint8 byte tensors).
        """
        torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell(
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
            num_experts_per_rank=int(self.expert_size_per_partition),
            hidden_size=hidden,
            intermediate_size_per_partition=int(self.intermediate_size_per_partition),
            expand_intermediate_size_per_partition=int(self.expand_intermediate_size_per_partition),
            max_tokens_per_rank=int(self.max_num_tokens),
            peer_offsets=peer_offsets,
            apply_topk_in_fc1=bool(self.apply_topk_in_fc1),
            gate_up_clamp=self.gate_up_clamp,
            token_back_by_dispatch=bool(self.token_back_by_dispatch),
            non_ubulk_fc2_store=bool(self.non_ubulk_fc2_store),
        )
        if num_tokens == 0:
            return torch.empty((0, hidden), dtype=output_dtype, device=combine_output.device)
        # Deepgemm graph (apply_topk_in_fc1=True): the kernel already folded
        # the topk score into the per-route BF16 terms, so the host reduce is
        # a plain sum over the top-k axis. Accumulate in fp32 explicitly to
        # match the design reference ``bf16(sum_fp32(term))`` and to be robust
        # against any future change to the bf16 reduction accumulator type.
        out = combine_output[:num_tokens].to(torch.float32).sum(dim=1).to(output_dtype)
        return out

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
    ) -> torch.Tensor:
        """Unified MegaMoE CuteDSL forward: acquire -> stage -> launch.

        The kernel is always launched with ``T = max_num_tokens`` (its
        compile-time constexpr); live tokens fill the first
        ``num_tokens`` rows and the tail is masked via ``topk_idx == -1``
        (skipped by dispatch_kernel) and zero ``topk_weights`` (combine
        stale-data guard).

        ``FusedCommMoEScheduler`` invariant 7 forces every EP rank to
        cross the NVLink barrier even with zero local tokens; only
        single-rank short-circuits ``num_tokens == 0`` because no peer
        is waiting.
        """
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(
                f"MegaMoECuteDsl run_moe got {num_tokens} tokens but the "
                f"staging buffer is sized for {self.max_num_tokens}. Raise "
                f"model_config.moe_max_num_tokens so peers do not read "
                f"invalid rows."
            )
        if num_tokens == 0 and self.ep_size == 1:
            return torch.empty((0, hidden), dtype=output_dtype, device=device)

        bufs = self._acquire_buffers(
            top_k=top_k, hidden=hidden, device=device, output_dtype=output_dtype
        )
        self._stage_inputs(
            bufs=bufs,
            x=x,
            x_sf=x_sf,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens=num_tokens,
            top_k=top_k,
        )
        return self._launch_megamoe_kernel(
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
            num_tokens=num_tokens,
            output_dtype=output_dtype,
        )
