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
DG-native weight tensors, checkpoint loading, scale conversion, and DeepGEMM
weight transform. The backend owns SymmBuffer allocation and reuse.
"""

from __future__ import annotations

import os
import socket
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType
from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
)
from ..impl_environment import MoEDep
from ..interface import MoE, MoESchedulerKind, MoEWeightLoadingMode, _reject
from ..quantization import W4A8MXFP4MXFP8MegaMoEDeepGemmMethod, _import_deep_gemm
from ..routing import BaseMoeRoutingMethod

__all__ = ["MegaMoEDeepGemm"]

# Process-global DG SymmBuffer cache. The cached object is mutable
# forward-time activation workspace (input ``x`` / routing slots /
# L1+L2 GEMM intermediates), not immutable weight state. Reuse relies
# on the current TRT-LLM execution contract that MegaMoE layers run
# serially within a forward pass; concurrent MegaMoE forwards sharing
# a key would race on the same scratch buffers.
#
# Keyed on buffer geometry only, never on the EP group's identity: a
# worker process outlives the LLM built in it, so keying on the group
# would mint a fresh key per LLM and stack a second allocation on top
# of the first (and ``id()`` is recycled, so it can also alias a dead
# group). Since the key no longer distinguishes groups, a hit is
# re-validated against the caller's live group in ``_alloc_symm_buffer``.
# ``release_symm_buffer_cache`` bounds the lifetime.
_MEGA_MOE_SYMM_BUFFER_CACHE: Dict[tuple, object] = {}


def _free_symm_buffer(buffered: object) -> int:
    """Release one DG SymmBuffer's symmetric memory. Returns its size in bytes.

    ``SymmBuffer.destroy`` nulls only ``handle``/``buffer``/``group``/``x``/
    ``x_sf``, leaving 10 of the 12 tensor views sliced out of the allocation
    in place -- and any surviving view pins the whole thing. So sweep every
    remaining tensor attribute too, otherwise this frees nothing whenever the
    object is held by a reference cycle rather than dropped outright.
    """
    nbytes = buffered.buffer.nbytes
    buffered.destroy()
    for name, value in list(vars(buffered).items()):
        if isinstance(value, torch.Tensor):
            setattr(buffered, name, None)
    return nbytes


def release_symm_buffer_cache() -> None:
    """Free every cached DG SymmBuffer. Call once per executor teardown.

    The EP group these buffers were rendezvoused over is destroyed on
    executor shutdown, so nothing may reuse them afterwards. They must be
    dropped explicitly: the backing symmetric memory sits outside PyTorch's
    caching allocator, so neither a GC nor ``torch.cuda.empty_cache()``
    reclaims it, and a reused worker process would otherwise carry a full
    activation workspace into the next LLM's memory budget.
    """
    if not _MEGA_MOE_SYMM_BUFFER_CACHE:
        return
    # Detach before freeing so a raising free cannot leave a half-destroyed
    # buffer in the cache for a later lookup to trip over.
    buffers = list(_MEGA_MOE_SYMM_BUFFER_CACHE.values())
    _MEGA_MOE_SYMM_BUFFER_CACHE.clear()
    total_bytes = sum(_free_symm_buffer(buffered) for buffered in buffers)
    logger.info(
        f"[MegaMoE] released {len(buffers)} DG SymmBuffer(s): {total_bytes / 2**30:.2f} GiB"
    )


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


def _assert_num_slots_divisible_by_ep(num_slots: int, ep_size: int) -> None:
    """The DG SymmBuffer is sized to the global slot count and sharded evenly.

    Each rank's weight shard is ``num_slots // ep_size`` slots, so a
    non-divisible layout does not fail -- it silently produces wrong per-rank
    slot ranges. That wrong-answer shape is why this is re-checked outside
    ``can_implement`` (see the callers) instead of trusting the resolution path
    alone: direct construction and post-``__init__`` EPLB syncs never go through
    it.
    """
    if num_slots % max(ep_size, 1) != 0:
        raise ValueError(
            f"MegaMoEDeepGemm requires num_slots ({num_slots}) divisible by "
            f"ep_size ({ep_size}). Adjust the EPLB replication factor or ep_size."
        )


class MegaMoEDeepGemm(MoE):
    """MoE backend wrapping DeepGEMM's fused ``fp8_fp4_mega_moe`` kernel."""

    _SUPPORTED_ACTIVATION_DTYPES = frozenset({torch.bfloat16})

    # Kernel owns dispatch + GEMM1 + gated activation + GEMM2 + combine via NVLink
    # SymmBuffer; ConfigurableMoE must NOT layer host-side comm on top.
    scheduler_kind = MoESchedulerKind.FUSED_COMM

    # MegaMoE partitions the global slot table, not just the raw expert count.
    # Let backend-specific num_slots checks handle EPLB/non-divisible layouts.
    _supports_non_divisible_ep: bool = True

    # ------------------------------------------------------------------
    # Capability gating
    # ------------------------------------------------------------------

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        # Process-group availability is validated during construction.
        if not is_sm_100f(d.env.sm):
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                f"MegaMoEDeepGemm requires SM100 family (SM100 or SM103) "
                f"for DeepGEMM's fp8_fp4_mega_moe kernel; got SM{d.env.sm}",
            )
        if p.dtype_act not in cls._SUPPORTED_ACTIVATION_DTYPES:
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                f"MegaMoEDeepGemm supports activations in "
                f"{cls._SUPPORTED_ACTIVATION_DTYPES}, got {p.dtype_act}",
            )
        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                "MegaMoEDeepGemm does not support swiglu_gptoss_style",
            )
        if p.quant_algo != QuantAlgo.W4A8_MXFP4_MXFP8:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"MegaMoEDeepGemm supports W4A8_MXFP4_MXFP8 only, got {p.quant_algo}",
            )
        # TMA requires packed-UE8M0 scale-factor rows to be 16-byte aligned (K % 512 == 0).
        if p.hidden_size is not None and p.hidden_size % 512 != 0:
            return _reject(
                MoERejectReason.SHAPE_UNALIGNED,
                f"MegaMoEDeepGemm requires hidden_size % 512 == 0 "
                f"(DeepGEMM TMA-aligned packed-UE8M0 SF row); "
                f"got hidden_size={p.hidden_size}",
            )
        if p.intermediate_size is not None and p.intermediate_size % 512 != 0:
            return _reject(
                MoERejectReason.SHAPE_UNALIGNED,
                f"MegaMoEDeepGemm requires intermediate_size % 512 == 0 "
                f"(DeepGEMM TMA-aligned packed-UE8M0 SF row); "
                f"got intermediate_size={p.intermediate_size}",
            )
        if p.activation_type != ActivationType.Swiglu:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"MegaMoEDeepGemm only supports ActivationType.Swiglu (got {p.activation})",
            )
        if d.tp_size != 1:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"MegaMoEDeepGemm is EP-only (moe_tp_size=1); got tp_size={d.tp_size}",
            )
        if d.cluster_size != 1:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"MegaMoEDeepGemm assumes cluster_size=1; got cluster_size={d.cluster_size}",
            )
        if d.num_slots % max(d.ep_size, 1) != 0:
            return _reject(
                MoERejectReason.SLOTS_NOT_DIVISIBLE_BY_EP,
                f"MegaMoEDeepGemm requires num_slots ({d.num_slots}) "
                f"divisible by ep_size ({d.ep_size})",
            )
        # DG's fp8_fp4_mega_moe assumes the MoE input is partitioned across
        # ranks. DEP > EP leaves some tokens unreachable; TEP replicates the
        # input so dispatch sees parallel_size duplicate copies, which is
        # arithmetically correct but ~parallel_size times slower.
        if d.parallel_size > 1:
            if not d.use_dp:
                return _reject(
                    MoERejectReason.TOPOLOGY_UNSUPPORTED,
                    f"MegaMoEDeepGemm does not support TEP "
                    f"(enable_attention_dp=False, parallel_size={d.parallel_size})",
                )
            if d.ep_size != d.parallel_size:
                return _reject(
                    MoERejectReason.TOPOLOGY_UNSUPPORTED,
                    f"MegaMoEDeepGemm with enable_attention_dp=True requires "
                    f"ep_size == parallel_size (got ep_size={d.ep_size}, "
                    f"parallel_size={d.parallel_size})",
                )
        if not d.env.has_dep(MoEDep.DEEPGEMM_MEGAMOE):
            return _reject(
                MoERejectReason.DEP_MISSING,
                "MegaMoEDeepGemm requires a DeepGEMM build exposing fp8_fp4_mega_moe",
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
        activation_type: ActivationType = ActivationType.Swiglu,
        init_load_balancer: bool = True,
        # DG tunables. ``activation=None`` infers Kimi K3 SiTU from the
        # pretrained config and otherwise defaults to SwiGLU.
        activation: Optional[str] = None,
        swiglu_limit_scalar: Optional[float] = None,
        fast_math: bool = True,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
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

        # Topology / activation eligibility is owned by ``can_implement``.
        # Keep construction-only invariants that are not part of (p, d):
        # apply_router_weight_on_input is a call-site flag, not a deployment
        # field, so it stays here until it is modeled on MoEProblem/Deployment.
        if apply_router_weight_on_input:
            raise ValueError(
                "MegaMoEDeepGemm does not support apply_router_weight_on_input. "
                "DG's fp8_fp4_mega_moe applies routing weights on the MoE "
                "output, not by pre-scaling the input — the two paths are "
                "not equivalent. Use a different MoE backend for models that "
                "require pre-scaling, or extend the kernel call."
            )
        # Also gated in ``can_implement``, but that only covers the resolution
        # path; this catches direct construction.
        _assert_num_slots_divisible_by_ep(self.num_slots, self.ep_size)
        activation, situ_beta, situ_linear_beta = self._resolve_activation_config(
            model_config,
            activation=activation,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )
        if activation == "situ" and swiglu_limit_scalar is not None:
            raise ValueError("MegaMoEDeepGemm SiTU does not support activation_clamp.")
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation
        self.swiglu_limit_scalar = swiglu_limit_scalar
        self.fast_math = fast_math
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta

        # Buffer sizing. MoE layers execute serially per forward; a single
        # process-level pool sized to worst-case per-rank tokens serves all.
        self.max_num_tokens = int(
            getattr(model_config, "moe_max_num_tokens", 0)
            or getattr(model_config, "max_num_tokens", 0)
            or 4096
        )
        # Under attention DP, ``ModelConfig`` pre-multiplies
        # ``moe_max_num_tokens`` by ``dp_size``. The DG SymmBuffer further
        # scales the pool by ``num_ranks`` (= ep_size, which equals dp_size
        # in the supported full-ADP topology asserted above). Without this
        # divide the buffer is sized as ``dp_size * ep_size *
        # max_num_tokens``, doubling the EP factor and exploding HBM (see
        # ``layout::Workspace`` / ``get_num_max_pool_tokens`` in DG).
        if self.use_dp and self.ep_size > 1:
            self.max_num_tokens = max(1, (self.max_num_tokens + self.ep_size - 1) // self.ep_size)

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
        # The actual allocation is deferred to ``cache_derived_state`` so
        # ConfigurableMoE can first sync EPLB-derived attributes (``num_slots``,
        # ``expert_size_per_partition``, ...) via ``_BACKEND_SYNC_ATTRS`` and
        # MetaInitMode can exit before the collective. ``MoE.__init__`` only
        # seeds ``num_slots = num_experts`` as a placeholder when the backend
        # is constructed with ``init_load_balancer=False``; sizing the buffer
        # here would therefore break EPLB. The loader walks the cache stage in
        # deterministic module order on all EP ranks, preserving rendezvous
        # lockstep for ordinary and GMS RO loads.
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

    @staticmethod
    def _resolve_activation_config(
        model_config: ModelConfig,
        *,
        activation: Optional[str],
        situ_beta: Optional[float],
        situ_linear_beta: Optional[float],
    ) -> Tuple[str, Optional[float], Optional[float]]:
        pretrained_config = model_config.pretrained_config
        text_config = getattr(pretrained_config, "text_config", None)
        config_situ_beta = getattr(pretrained_config, "activation_situ_beta", None)
        config_situ_linear_beta = getattr(pretrained_config, "activation_situ_linear_beta", None)
        if config_situ_beta is None:
            config_situ_beta = getattr(text_config, "activation_situ_beta", None)
        if config_situ_linear_beta is None:
            config_situ_linear_beta = getattr(text_config, "activation_situ_linear_beta", None)
        if activation is None:
            activation = "situ" if config_situ_beta is not None else "swiglu"
        activation = activation.lower()
        if activation not in ("swiglu", "situ"):
            raise ValueError(
                f"MegaMoEDeepGemm activation must be 'swiglu' or 'situ'; got {activation!r}."
            )
        if activation == "swiglu":
            if situ_beta is not None or situ_linear_beta is not None:
                raise ValueError("SiTU beta parameters require activation='situ'.")
            return activation, None, None

        situ_beta = config_situ_beta if situ_beta is None else situ_beta
        situ_linear_beta = config_situ_linear_beta if situ_linear_beta is None else situ_linear_beta
        if situ_beta is None or situ_linear_beta is None:
            raise ValueError(
                "MegaMoEDeepGemm SiTU requires activation_situ_beta and "
                "activation_situ_linear_beta in the pretrained config, or "
                "explicit situ_beta and situ_linear_beta arguments."
            )
        if situ_beta <= 0 or situ_linear_beta <= 0:
            raise ValueError("MegaMoEDeepGemm SiTU beta parameters must be positive.")
        return activation, float(situ_beta), float(situ_linear_beta)

    def _supports_load_balancer(self) -> bool:
        # The DeepGEMM mega kernel routes by `topk_idx` interpreted as slot id
        # (range [0, num_slots)) once the SymmBuffer is sized to num_slots.
        # Dynamic EPLB migrates the transformed DG tensors registered by the
        # quantization method, not the raw checkpoint-layout weights.
        return True

    def validate_configurable_moe(self, moe) -> None:
        """Re-assert the DG global slot count after ``ConfigurableMoE`` wiring.

        ``can_implement`` gates the same invariant, but only on the resolution
        path and only on the slot count the balancer config advertises at select
        time. ``moe`` is the owning ``ConfigurableMoE``, whose ``num_slots`` /
        ``ep_size`` have since been synced through ``_BACKEND_SYNC_ATTRS``, so
        this is the one place that sees the layout the kernel will actually run.
        """
        _assert_num_slots_divisible_by_ep(moe.num_slots, moe.ep_size)

    @staticmethod
    def _maybe_init_dist_from_mpi() -> None:
        """Initialize torch.distributed from mpi4py when running under TRT-LLM's MPI executor.

        Safe to call collectively: every MPI rank reaches MegaMoE's
        ``__init__`` synchronously during model build. Rank 0 picks a
        free port and bcasts (host, port) so single- and multi-node both
        work; the retry on RuntimeError absorbs the close()→bind() race
        on busy hosts. ``device_id`` uses intra-node local rank because
        ``global_rank % device_count`` is wrong on multi-node launches.
        """
        try:
            from mpi4py import MPI
        except ImportError:
            return
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        if world_size <= 1:
            return

        try:
            local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
            local_rank = local_comm.Get_rank()
        except Exception:
            local_rank = int(
                os.environ.get(
                    "OMPI_COMM_WORLD_LOCAL_RANK",
                    os.environ.get(
                        "MV2_COMM_WORLD_LOCAL_RANK", rank % max(1, torch.cuda.device_count())
                    ),
                )
            )

        def _pick_rendezvous():
            if rank == 0:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("", 0))
                    port = sock.getsockname()[1]
                host = socket.gethostbyname(socket.gethostname())
                return (host, port)
            return None

        # Respect pre-set launcher env vars (Slurm, Ray, torchrun).
        if not all(os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")):
            host, port = comm.bcast(_pick_rendezvous(), root=0)
            os.environ.setdefault("MASTER_ADDR", host)
            os.environ.setdefault("MASTER_PORT", str(port))
            os.environ.setdefault("RANK", str(rank))
            os.environ.setdefault("WORLD_SIZE", str(world_size))
        else:
            host = os.environ["MASTER_ADDR"]
            port = int(os.environ["MASTER_PORT"])

        device_id = None
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_id = torch.device("cuda", local_rank % torch.cuda.device_count())
        try:
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                device_id=device_id,
            )
        except RuntimeError:
            # close()→bind() race: redraw a port and retry once.
            new_host, new_port = comm.bcast(_pick_rendezvous(), root=0)
            os.environ["MASTER_ADDR"] = new_host
            os.environ["MASTER_PORT"] = str(new_port)
            host, port = new_host, new_port
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                device_id=device_id,
            )
        logger.info(
            f"[MegaMoE] Initialized torch.distributed from mpi4py "
            f"(rank={rank}, local_rank={local_rank}, "
            f"world_size={world_size}, master={host}:{port}, backend=nccl)"
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
            # TRT-LLM's default executor uses mpi4py and never calls
            # torch.distributed.init; bootstrap from MPI when present.
            self._maybe_init_dist_from_mpi()
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
        keyed on the (EP-size, slot/expert/topk/shape/activation) tuple; see
        that cache's definition for why the key carries no group identity.
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
            self.ep_size,
            self.num_experts,
            self.num_slots,
            self.max_num_tokens,
            self.routing_method.experts_per_token,
            self.hidden_size,
            self.intermediate_size,
            self.activation,
        )
        cached = _MEGA_MOE_SYMM_BUFFER_CACHE.get(key)
        if cached is not None and cached.group is not self._ep_pg:
            # Geometry-equal but rendezvoused over a different (already
            # destroyed) EP group: an LLM torn down without
            # release_symm_buffer_cache() leaves such an entry behind, and its
            # buffer's peer mappings are dead. Free it here so the driver can
            # reuse those pages for the allocation below -- symmetric memory is
            # outside the caching allocator, so this is the only way to get it
            # back within the process. Evict before freeing so a raising free
            # cannot leave a half-destroyed buffer behind for the next lookup.
            del _MEGA_MOE_SYMM_BUFFER_CACHE[key]
            freed = _free_symm_buffer(cached)
            cached = None
            logger.info(
                f"[MegaMoE] layer={self.layer_idx} released stale DG "
                f"SymmBuffer: {freed / 2**30:.2f} GiB"
            )
        if cached is None:
            cached = self._dg.get_symm_buffer_for_mega_moe(
                self._ep_pg,
                self.num_slots,
                self.max_num_tokens,
                self.routing_method.experts_per_token,
                self.hidden_size,
                self.intermediate_size,
                num_shared_experts=0,
                mma_type="fp8xfp4",
                activation=self.activation,
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
        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)
        self._weights_created = True

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False) -> None:
        if self.quant_method is None:
            self.create_weights()
        self.quant_method.load_weights(self, weights, allow_partial_loading)

    def cache_derived_state(self) -> None:
        # The rendezvous cannot run from create_weights under MetaInitMode.
        # This stage runs in deterministic module order after materialization
        # for both ordinary loads and GMS RO readers.
        self._alloc_symm_buffer()
        super().cache_derived_state()

    def post_load_weights(self) -> None:
        if self.quant_method is None:
            self.create_weights()
        self.transform_weights()
        self.cache_derived_state()

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

        Zero-token short-circuit: returns the DG empty layout (FP8 +
        packed-UE8M0 int32 SF) directly. ``FusedCommMoEScheduler``
        unconditionally calls ``quantize_input`` for every chunk
        including zero-token chunks so peer ranks can cross the in-kernel
        NVLink barrier; ``torch.ops.trtllm.mxfp8_quantize`` rejects empty
        input on some builds, so the empty-layout synthesis stays here
        rather than in the scheduler.
        """
        del post_quant_comm  # MegaMoE runs pre-quant comm via DG SymmBuffer
        x_bf16 = x.to(torch.bfloat16).contiguous()
        if x_bf16.shape[0] == 0:
            device = x_bf16.device
            hidden = x_bf16.shape[1]
            x_fp8 = torch.empty((0, hidden), dtype=torch.float8_e4m3fn, device=device)
            # Packed-UE8M0 int32 SF: one int32 per 128 input elements per row,
            # same stride contract as the non-empty runs for run_moe.
            x_sf = torch.empty((0, max(hidden // 128, 0)), dtype=torch.int32, device=device)
            return x_fp8, x_sf
        return _quantize_bf16_to_fp8_ue8m0(x_bf16)

    def supports_fused_prepare(self) -> bool:
        """Whether ``run_moe`` can prepare DG SymmBuffer directly from BF16 input."""
        return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "megamoe_prepare")

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run the fused kernel with either BF16 or pre-quantized activations.

        The fused-prepare path receives BF16 activations and writes the
        FP8+SF+topk SymmBuffer fields in one custom op. The fallback path
        keeps the original ``quantize_input`` + copy contract.
        """
        del workspace  # The SymmBuffer is this backend's own workspace.
        x = ctx.x
        token_selected_experts = ctx.token_selected_experts
        token_final_scales = ctx.token_final_scales
        x_sf = ctx.x_sf
        output_dtype = ctx.output_dtype
        if output_dtype is None:
            output_dtype = self.dtype or torch.bfloat16
        dg = self._dg
        buf = self._symm_buffer
        assert buf is not None, (
            "MegaMoE SymmBuffer not allocated — _alloc_symm_buffer runs in "
            "cache_derived_state; ensure the model loader's cache/post-load "
            "pass ran before forward (and was not skipped by _weights_removed)."
        )
        num_tokens = x.shape[0]
        assert num_tokens <= self.max_num_tokens, (
            f"MegaMoE got {num_tokens} tokens but buffer is sized for "
            f"{self.max_num_tokens}. Raise model_config.moe_max_num_tokens."
        )

        if num_tokens > 0:
            if x_sf is None:
                if not self.supports_fused_prepare():
                    raise ValueError("MegaMoEDeepGemm requires x_sf from quantize_input")
                torch.ops.trtllm.megamoe_prepare(
                    x.to(torch.bfloat16).contiguous(),
                    token_selected_experts.contiguous(),
                    token_final_scales.contiguous(),
                    buf.x,
                    buf.x_sf,
                    buf.topk_idx,
                    buf.topk_weights,
                )
            else:
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
            activation_clamp=self.swiglu_limit_scalar,
            fast_math=self.fast_math,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
        )
        return y.to(output_dtype)
