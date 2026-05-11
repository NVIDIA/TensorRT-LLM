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
"""MegaMoE — DeepGEMM ``fp8_fp4_mega_moe`` as a first-class MoE backend.

This backend owns capability checks, routing/activation quantization, and the
fused kernel entry point. ``W4A8MXFP4MXFP8MegaMoEDeepGemmMethod`` owns the
DG-native weight tensors, checkpoint loading, scale conversion, SymmBuffer
allocation, and DeepGEMM weight transform.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType
from ..interface import MoE, MoESchedulerKind, MoEWeightLoadingMode
from ..quantization import (
    W4A8MXFP4MXFP8MegaMoEDeepGemmMethod,
    _import_deep_gemm,
    _MegaMoEUnavailable,
)
from ..routing import BaseMoeRoutingMethod

__all__ = ["MegaMoEDeepGemm"]

# Process-global DG SymmBuffer cache. The cached object is mutable
# forward-time activation workspace (input ``x`` / routing slots /
# L1+L2 GEMM intermediates), not immutable weight state. Reuse relies
# on the current TRT-LLM execution contract that MegaMoE layers run
# serially within a forward pass; concurrent MegaMoE forwards sharing
# a key would race on the same scratch buffers.
_MEGA_MOE_SYMM_BUFFER_CACHE: Dict[tuple, object] = {}

# ---- Fused MXFP8 per-token quant backends --------------------------------
# We want: BF16 (m, H) → FP8 E4M3 (m, H) + packed-UE8M0 SF (m, H/32/4) int32.
# Three candidates, in preference order:
#
#   1. ``torch.ops.trtllm.mxfp8_quantize(x, False, alignment=32)`` — TRT-LLM
#      C++ CUDA kernel. Roundtrip-verified byte-identical to DG's Python
#      helper (fp8 bytes + SF after u8→int32 reshape). Fastest by 5-25×
#      vs torch.compile, one kernel launch (~11 us regardless of seq).
#      Requires ``libth_common.so`` to be loaded; ``ConfigurableMoE`` pulls
#      this in on construction so it's always registered by the time
#      ``backend.quantize_input`` runs.
#
#   2. ``torch.compile(dg.per_token_cast_to_fp8, dynamic=True)`` — fallback
#      when the TRT-LLM op isn't registered (e.g. slim builds, standalone
#      DG tests). Inductor fuses the ~8 elementwise kernels into 1-2
#      Triton kernels but still pays one launch per seq boundary.
#
# ``_FUSED_PER_TOKEN_CAST`` caches the fallback so we don't re-compile on
# every module creation.
_FUSED_PER_TOKEN_CAST = None


def _trtllm_mxfp8_quantize_available() -> bool:
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "mxfp8_quantize")


def _quantize_bf16_to_fp8_ue8m0(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (x_fp8, x_sf) in DG mega_moe's expected layout (packed int32)."""
    m, n = x.shape
    # Packed-UE8M0 stores 4 u8 scales per int32 over a 32-element block,
    # so n must be a multiple of 128 for the int32 view below to land on
    # an integer last-dim. Misaligned shapes would otherwise fail with a
    # cryptic reshape/view error; surface a clear contract here instead.
    if n % 128 != 0:
        raise ValueError(
            f"_quantize_bf16_to_fp8_ue8m0 requires hidden_size % 128 == 0 "
            f"(packed-UE8M0 int32 SF stride); got hidden_size={n}"
        )
    if _trtllm_mxfp8_quantize_available():
        # ``is_sf_swizzled_layout=False`` → flat row-major uint8 SF, one
        # byte per 32-element group. ``alignment=32`` → MXFP8 block size.
        x_fp8, x_sf_u8 = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=32)
        # DG wants (m, n/32/4) int32 with 4 u8 UE8M0 packed per int32.
        # TRT-LLM emits (m*n/32,) uint8 in the same byte order, so a
        # reshape + view is a zero-copy reinterpret.
        return x_fp8, x_sf_u8.view(m, n // 32).view(torch.int32)

    global _FUSED_PER_TOKEN_CAST
    if _FUSED_PER_TOKEN_CAST is None:
        dg = _import_deep_gemm()
        base = dg.per_token_cast_to_fp8

        def _call(t: torch.Tensor):
            return base(t, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

        _FUSED_PER_TOKEN_CAST = torch.compile(_call, dynamic=True, fullgraph=False)
    return _FUSED_PER_TOKEN_CAST(x)


class MegaMoEDeepGemm(MoE):
    """MoE backend wrapping DeepGEMM's fused ``fp8_fp4_mega_moe`` kernel."""

    _SUPPORTED_ACTIVATION_DTYPES = frozenset({torch.bfloat16})

    # Kernel owns dispatch + GEMM1 + SwiGLU + GEMM2 + combine via NVLink
    # SymmBuffer; ConfigurableMoE must NOT layer host-side comm on top.
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
        # Note: we intentionally do NOT probe ``torch.distributed`` state here.
        # ``can_implement`` is a static capability query (SM / dtype / quant /
        # shape). Whether a live EP ProcessGroup exists is a runtime concern,
        # not a capability one, and ``__init__``'s ``_resolve_ep_pg`` will
        # surface a clear error if dist is not initialized by the launcher.
        sm = get_sm_version()
        if sm != 100:
            return False, (
                f"MegaMoEDeepGemm requires SM100 (only arch with "
                f"sm100_fp8_fp4_mega_moe.cuh in DeepGEMM); got SM{sm}"
            )
        if dtype_activation not in cls._SUPPORTED_ACTIVATION_DTYPES:
            return False, (
                f"MegaMoEDeepGemm supports activations in "
                f"{cls._SUPPORTED_ACTIVATION_DTYPES}, got {dtype_activation}"
            )
        if swiglu_gptoss_style:
            return False, "MegaMoEDeepGemm does not support swiglu_gptoss_style"
        if quant_algo != QuantAlgo.W4A8_MXFP4_MXFP8:
            return False, (f"MegaMoEDeepGemm supports W4A8_MXFP4_MXFP8 only, got {quant_algo}")
        # Packed-UE8M0 per-token SF layout has two constraints. First,
        # the quantizer reinterprets 4 u8 scales as one int32, so K must
        # be divisible by 128. Second, DeepGEMM MegaMoE feeds SF buffers
        # through TMA; one u8 scale is stored per 32 K elements and the
        # per-token SF row must be 16B aligned. The TMA constraint is
        # stricter: (K / 32) % 16 == 0, so K must be divisible by 512.
        # Enforce the backend constraint here so the factory can fall
        # back cleanly before DG SymmBuffer allocation.
        if hidden_size is not None and hidden_size % 512 != 0:
            return False, (
                f"MegaMoEDeepGemm requires hidden_size % 512 == 0 "
                f"(DeepGEMM TMA-aligned packed-UE8M0 SF row); "
                f"got hidden_size={hidden_size}"
            )
        if intermediate_size is not None and intermediate_size % 512 != 0:
            return False, (
                f"MegaMoEDeepGemm requires intermediate_size % 512 == 0 "
                f"(DeepGEMM TMA-aligned packed-UE8M0 SF row); "
                f"got intermediate_size={intermediate_size}"
            )
        try:
            _import_deep_gemm()
        except _MegaMoEUnavailable as e:
            return False, str(e)
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
        activation_type: ActivationType = ActivationType.Swiglu,
        init_load_balancer: bool = True,
        without_comm: bool = False,
        # DG tunables.
        activation: str = "swiglu",
        activation_clamp: Optional[float] = None,
        fast_math: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            layer_idx=layer_idx,
            activation_type=activation_type,
            init_load_balancer=init_load_balancer,
        )

        # Assert supported topologies early so unsupported configurations
        # fall back via ``can_implement`` rather than crashing later in DG.
        assert self.tp_size == 1, (
            f"MegaMoEDeepGemm is EP-only (moe_tp_size=1); got tp_size={self.tp_size}"
        )
        assert self.cluster_size == 1, (
            f"MegaMoEDeepGemm assumes cluster_size=1; got cluster_size={self.cluster_size}"
        )
        # The DG SymmBuffer is sized to ``num_slots`` and sharded evenly over
        # EP ranks. Without EPLB, ``num_slots == num_experts`` so the two
        # constraints collapse; with EPLB ``num_slots`` may exceed
        # ``num_experts`` and ``num_experts % ep_size == 0`` is too strict.
        if self.num_slots % max(self.ep_size, 1) != 0:
            raise ValueError(
                f"MegaMoEDeepGemm requires num_slots ({self.num_slots}) "
                f"divisible by ep_size ({self.ep_size})."
            )

        # ADP semantics: DG's fp8_fp4_mega_moe subsumes cross-rank token
        # dispatch into its internal symm_mem exchange. When EP spans
        # *all* ranks that may carry tokens (i.e. ``ep_size ==
        # parallel_size``), no outer allgather / reducescatter is needed:
        # every token's origin rank is inside the EP group and DG returns
        # results to that origin. If EP is a strict subset of
        # parallel_size (e.g. attention-DP > moe_ep_size), some tokens
        # live on ranks that the DG kernel cannot reach — that topology
        # is not yet supported.
        if self.use_dp and self.parallel_size > 1:
            assert self.ep_size == self.parallel_size, (
                f"MegaMoEDeepGemm with enable_attention_dp=True requires "
                f"ep_size == parallel_size (got ep_size={self.ep_size}, "
                f"parallel_size={self.parallel_size}). Configurations "
                f"with ADP > EP are not yet supported; add the standard "
                f"allgather(pre) + reducescatter(post) wrapper before "
                f"calling fp8_fp4_mega_moe to support them."
            )

        # apply_router_weight_on_input pre-multiplies routing weights
        # onto x before the MoE compute (used by some top-1 models). DG's
        # fused kernel applies the weights on the MoE output instead;
        # mixing the two produces wrong math. Reject loudly — a silent
        # fallback would break llama-min-latency-style paths that set
        # this flag to True and assume top-1 semantics.
        assert not apply_router_weight_on_input, (
            "MegaMoEDeepGemm does not support apply_router_weight_on_input. "
            "DG's fp8_fp4_mega_moe applies routing weights on the MoE "
            "output, not by pre-scaling the input — the two paths are "
            "not equivalent. Use a different MoE backend for models that "
            "require pre-scaling, or extend the kernel call."
        )
        # DG's fp8_fp4_mega_moe currently only ships a fused SwiGLU
        # activation path. Reject other ActivationType values explicitly so
        # ``create_moe_backend`` callers do not silently get SwiGLU when
        # they asked for GELU / etc.
        if activation_type != ActivationType.Swiglu:
            raise ValueError(
                f"MegaMoEDeepGemm only supports ActivationType.Swiglu (got {activation_type})."
            )
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation
        self.activation_clamp = activation_clamp
        self.fast_math = fast_math

        # Buffer sizing. MoE layers execute serially per forward; a single
        # process-level pool sized to worst-case per-rank tokens serves all.
        self.max_num_tokens = int(
            getattr(model_config, "moe_max_num_tokens", 0)
            or getattr(model_config, "max_num_tokens", 0)
            or 4096
        )

        # Resolve the EP ProcessGroup at module construction — creating a
        # group at forward time would be collective on a non-synchronous
        # call stack and deadlock under PP / layer-skip. Construction is
        # globally synchronous across ranks during model build.
        self._ep_pg = self._resolve_ep_pg()

        # Cache the bundled DeepGEMM module once at construction. ``_import_deep_gemm``
        # does a fresh ``hasattr`` / ``inspect.signature`` check on every call;
        # paying that on every forward (``run_moe`` path) shows up in host-side
        # CPU overhead even though the underlying ``import`` is cached by Python.
        self._dg = _import_deep_gemm()

        # NVLink SymmBuffer activation workspace. Allocation is a
        # model-build-period collective (``symm_mem.rendezvous`` over the
        # EP group); allocating from ``run_moe`` would deadlock under
        # PP / layer-skip paths where some ranks may not enter this
        # layer in lockstep, and would also fail under CUDA graph
        # capture because rendezvous is a host-side IPC operation.
        # ``_resolve_ep_pg`` above relies on the same lockstep window.
        #
        # The actual allocation is deferred to ``create_weights`` so that
        # ConfigurableMoE has a chance to overwrite EPLB-derived
        # attributes (``num_slots``, ``expert_size_per_partition``, ...)
        # via ``_BACKEND_SYNC_ATTRS`` before we size the buffer. When the
        # backend is constructed with ``init_load_balancer=False`` (the
        # ConfigurableMoE path), ``MoE.__init__`` only seeds
        # ``num_slots = num_experts`` as a placeholder; under EPLB the
        # real slot count is larger, and sizing the SymmBuffer here would
        # produce ``sym_buffer.num_experts != num_experts_per_rank *
        # num_ranks`` at forward time. ``create_weights`` is also
        # collective across ranks (called by ConfigurableMoE on all ranks
        # right after the sync, or by the backend itself when used
        # standalone with ``init_load_balancer=True``), so deferring keeps
        # the rendezvous lockstep guarantee intact.
        # See ``_alloc_symm_buffer`` for the cache contract.
        self._symm_buffer = None

        # Weight tensors and DG transforms are owned by the quant method.
        self._t_l1 = None
        self._t_l2 = None
        self._weights_loaded = False
        self._weights_created = False
        self.quant_method = None
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _supports_load_balancer(self) -> bool:
        # The DeepGEMM mega kernel routes by `topk_idx` interpreted as slot id
        # (range [0, num_slots)) once the SymmBuffer is sized to num_slots.
        # Dynamic EPLB migrates the transformed DG tensors registered by the
        # quantization method, not the raw checkpoint-layout weights.
        return True

    def validate_configurable_moe(self, moe) -> None:
        """Assert ``num_slots % ep_size == 0`` for the DG global slot count.

        ``moe`` is the owning ``ConfigurableMoE``; its ``num_slots`` /
        ``ep_size`` / load-balancer flags are populated by ``MoE.__init__``
        before ``validate_backend`` runs, so they're stable here.
        """
        # SymmBuffer.num_experts (= num_slots in the DG kernel) must divide
        # evenly across EP ranks because each rank's weight shard is
        # ``num_slots // ep_size`` slots.
        if moe.num_slots % moe.ep_size != 0:
            raise ValueError(
                f"MegaMoEDeepGemm requires num_slots ({moe.num_slots}) "
                f"divisible by ep_size ({moe.ep_size}). Adjust the EPLB "
                f"replication factor or ep_size."
            )

    # ------------------------------------------------------------------
    # EP process-group resolution (no collective at forward time)
    # ------------------------------------------------------------------
    def _resolve_ep_pg(self):
        """Return the torch.distributed ProcessGroup for the EP sub-world.

        Prefers ``mapping.moe_ep_group_pg`` (DeviceMeshTopology, Ray path)
        because it was built once at Mapping init. Falls back to
        ``dist.group.WORLD`` only when ``ep_size == world_size`` (single
        EP subset covers all ranks).

        Does NOT call ``dist.new_group`` — that's collective and unsafe to
        invoke from any path that may skip ranks (e.g. PP-isolated layer
        forwards). When the mapping cannot provide a PG and EP is a
        strict subset of world, we raise with a clear message pointing
        at ``mpi_disabled=1`` / Ray as the supported path.
        """
        if not dist.is_initialized():
            raise RuntimeError(
                "MegaMoEDeepGemm requires torch.distributed to be "
                "initialized before module construction (mpirun or Ray)."
            )
        # Preferred: reuse the existing PG from the mapping (Ray / DeviceMesh).
        # Log at info() only on layer 0 so deep models do not spam N copies of
        # the same message; deeper layers log at debug() for triage.
        try:
            pg = self.mapping.moe_ep_group_pg
            log_fn = logger.info if self.layer_idx == 0 else logger.debug
            log_fn(
                f"[MegaMoE] layer={self.layer_idx} using mapping.moe_ep_group_pg (DeviceMesh path)"
            )
            return pg
        except (NotImplementedError, AttributeError):
            pass
        # Fallback: degenerate to WORLD when EP spans all ranks.
        world_size = dist.get_world_size()
        if self.ep_size == world_size:
            log_fn = logger.info if self.layer_idx == 0 else logger.debug
            log_fn(
                f"[MegaMoE] layer={self.layer_idx} using dist.group.WORLD "
                f"(EP == world_size == {world_size})"
            )
            return dist.group.WORLD
        raise RuntimeError(
            f"MegaMoEDeepGemm: cannot resolve EP ProcessGroup. The current "
            f"mapping does not expose ``moe_ep_group_pg`` and EP "
            f"({self.ep_size}) is a strict subset of world "
            f"({world_size}). Use DeviceMeshTopology (TLLM_DISABLE_MPI=1) "
            f"so the EP PG is constructed once at Mapping init, or set "
            f"ep_size == world_size."
        )

    # ------------------------------------------------------------------
    # SymmBuffer activation workspace (collective resource)
    # ------------------------------------------------------------------
    def _alloc_symm_buffer(self) -> None:
        """Allocate (or fetch from cache) the DG NVLink SymmBuffer.

        The SymmBuffer is forward-time activation workspace
        (input ``x`` / ``x_sf``, ``topk_idx``/``topk_weights``, L1/L2
        GEMM intermediates) backed by NVLink symmetric memory. Allocation
        runs ``symm_mem.rendezvous`` over the EP group plus a barrier and
        ``cuda.synchronize`` (see DeepGEMM ``mega/__init__.py``); this is
        a build-time collective and must not run on ``run_moe``: a
        non-lockstep rank would deadlock the rendezvous, and CUDA graph
        capture would fail on the host-side IPC handle exchange.

        Buffers are shared across layers via ``_MEGA_MOE_SYMM_BUFFER_CACHE``
        keyed on the (EP-PG, slot/expert/topk/shape/activation) tuple.
        Sharing is safe only while MegaMoE layer forwards are issued
        serially within a forward pass; concurrent MegaMoE forwards
        sharing a key would race on the same scratch buffers.

        Both ``num_slots`` and ``num_experts`` participate in the cache
        key because two layers with the same ``num_experts`` but
        different EPLB replication factors must not collide on the same
        cached buffer.

        Invariant: the SymmBuffer's ``num_experts`` parameter is the
        GLOBAL slot count (``kNumExperts`` in the DG kernel). With EPLB
        this equals ``num_slots`` (``>= num_experts``); without EPLB
        ``ConfigurableMoE`` syncs ``num_slots == num_experts`` so the
        contract holds in both cases. See ``CHUNKING_DESIGN.md §5.3.2``
        for the local-vs-global axis split.
        """
        if self._symm_buffer is not None:
            return
        key = (
            id(self._ep_pg),
            self.num_experts,
            self.num_slots,
            self.max_num_tokens,
            self.routing_method.experts_per_token,
            self.hidden_size,
            self.intermediate_size,
            self.activation,
        )
        cached = _MEGA_MOE_SYMM_BUFFER_CACHE.get(key)
        if cached is None:
            cached = self._dg.get_symm_buffer_for_mega_moe(
                self._ep_pg,
                self.num_slots,
                self.max_num_tokens,
                self.routing_method.experts_per_token,
                self.hidden_size,
                self.intermediate_size,
                True,
                self.activation,
            )
            _MEGA_MOE_SYMM_BUFFER_CACHE[key] = cached
            # Log only on the first layer; deeper layers reuse the cache
            # and would otherwise spam N copies of an identical line.
            log_fn = logger.info if self.layer_idx == 0 else logger.debug
            log_fn(
                f"[MegaMoE] layer={self.layer_idx} allocated DG "
                f"SymmBuffer: {cached.buffer.nbytes / 2**30:.2f} GiB"
            )
        self._symm_buffer = cached

    # ------------------------------------------------------------------
    # Weight lifecycle
    # ------------------------------------------------------------------
    def _get_quant_method(self):
        if (
            self.quant_config is None
            or not self.quant_config.layer_quant_mode.has_w4a8_mxfp4_mxfp8()
        ):
            raise NotImplementedError("MegaMoEDeepGemm supports W4A8_MXFP4_MXFP8 quantization only")
        return W4A8MXFP4MXFP8MegaMoEDeepGemmMethod()

    def create_weights(self):
        if self._weights_created:
            return
        # Allocate the DG NVLink SymmBuffer here (lazily) rather than from
        # ``__init__`` because ConfigurableMoE only syncs the EPLB-derived
        # attributes (``num_slots``, ``expert_size_per_partition``, ...)
        # onto the backend AFTER backend ``__init__`` returns, just before
        # calling ``backend.create_weights()``. Sizing the SymmBuffer in
        # ``__init__`` would therefore use the placeholder
        # ``num_slots = num_experts`` and break EPLB at forward time
        # (DeepGEMM asserts ``num_experts == num_experts_per_rank *
        # num_ranks`` in ``mega.hpp``). Both call sites (the
        # ConfigurableMoE-driven path and the standalone
        # ``init_load_balancer=True`` path that runs ``create_weights``
        # from ``__init__``) reach this point on every EP rank in
        # lockstep, preserving the rendezvous safety invariant.
        self._alloc_symm_buffer()
        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)
        self._weights_created = True

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False) -> None:
        if self.quant_method is None:
            self.create_weights()
        self.quant_method.load_weights(self, weights, allow_partial_loading)

    def post_load_weights(self) -> None:
        if self.quant_method is None:
            self.create_weights()
        self.quant_method.post_load_weights(self)

    # ------------------------------------------------------------------
    # MoE-contract methods
    # ------------------------------------------------------------------
    def quantize_input(self, x, *, post_quant_comm: bool = False, **kwargs):
        """BF16 → FP8-E4M3 + packed-UE8M0 per-token SF (gran_k=32).

        Delegates to ``_quantize_bf16_to_fp8_ue8m0`` which picks the
        fastest available backend (TRT-LLM C++ op ~11 us at any seq,
        or ``torch.compile`` fallback ~60-260 us). Byte-identical
        output across all paths so DG's ``fp8_fp4_mega_moe`` consumes
        it unchanged.
        """
        del post_quant_comm  # MegaMoE runs pre-quant comm via DG SymmBuffer
        x_bf16 = x.to(torch.bfloat16).contiguous()
        return _quantize_bf16_to_fp8_ue8m0(x_bf16)

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
        """Run the fused kernel with pre-quantized activations.

        ConfigurableMoE computes routing and calls ``quantize_input`` before
        invoking this method, so the backend receives the same FP8+SF+topk
        contract at this unified backend entry point.
        """
        assert not unused_kwargs, (
            f"MegaMoEDeepGemm.run_moe got unexpected kwargs: {sorted(unused_kwargs)}"
        )
        if output_dtype is None:
            output_dtype = self.dtype or torch.bfloat16
        if x_sf is None:
            raise ValueError("MegaMoEDeepGemm requires x_sf from quantize_input")
        dg = self._dg
        buf = self._symm_buffer
        assert buf is not None, (
            "MegaMoE SymmBuffer not allocated — _alloc_symm_buffer should "
            "run unconditionally in __init__; check for a subclass that "
            "skipped the parent constructor."
        )
        num_tokens = x.shape[0]
        assert num_tokens <= self.max_num_tokens, (
            f"MegaMoE got {num_tokens} tokens but buffer is sized for "
            f"{self.max_num_tokens}. Raise model_config.moe_max_num_tokens."
        )

        if num_tokens > 0:
            buf.x[:num_tokens].copy_(x)
            buf.x_sf[:num_tokens].copy_(x_sf)
            buf.topk_idx[:num_tokens].copy_(token_selected_experts.to(torch.int64))
            buf.topk_weights[:num_tokens].copy_(token_final_scales.to(torch.float32))

        y = torch.empty((num_tokens, self.hidden_size), dtype=torch.bfloat16, device=buf.x.device)
        dg.fp8_fp4_mega_moe(
            y,
            self._t_l1,
            self._t_l2,
            buf,
            activation=self.activation,
            activation_clamp=self.activation_clamp,
            fast_math=self.fast_math,
        )
        return y.to(output_dtype)
