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
"""MegaMoEDeepGemmFusedMoE — DeepGEMM fp8_fp4_mega_moe as a first-class MoE backend.

Quantization scheme
-------------------
W4A8_MXFP4_MXFP8: packed MXFP4 weights (4 bits, UE8M0 block scale per 32
input elements) × FP8 E4M3 activations quantized per-token with UE8M0
block scales. Matches ``TRTLLMGenFusedMoE``'s W4A8_MXFP4_MXFP8 math; the
difference is the kernel implementation — DG fuses EP dispatch + GEMM1 +
SwiGLU + GEMM2 + EP combine into a single launch.

Weight storage (DG-native, EP-only Phase 1)
------------------------------------------
* ``w3_w1_weight``       : uint8 [E_local, 2*I, H // 2]   (MXFP4 nibbles)
* ``w3_w1_weight_scale`` : uint8 [E_local, 2*I, H // 32]  (UE8M0)
* ``w2_weight``          : uint8 [E_local, H,   I // 2]
* ``w2_weight_scale``    : uint8 [E_local, H,   I // 32]

The SAME raw bytes that ``torch.ops.trtllm.fp4_quantize`` emits, so the
``VANILLA`` / ``FUSED_GATE_UP_PROJ`` loader key schemas used by TRT-LLM
models work directly. On ``post_load_weights`` the uint8 scales are
lifted to DG's fp32-UE8M0 format and both weight and scale tensors are
passed through ``deep_gemm.transform_weights_for_mega_moe``.

Forward (hot path)
------------------
Per-forward on ``num_tokens_real`` unpadded tokens per rank:
  1. Routing (external via ``routing_method.apply``).
  2. Pre-quant hidden states BF16 → FP8 E4M3 with packed UE8M0 block
     scales (``deep_gemm.utils.per_token_cast_to_fp8``).
  3. Copy FP8 hidden / SF / topk_idx / topk_weights into the shared DG
     ``SymmBuffer``.
  4. Fire ``deep_gemm.fp8_fp4_mega_moe(y, t_l1, t_l2, buf, ...)``.
  5. Return ``y`` (bf16 [num_tokens_real, H], combined across EP).

The ``num_tokens_real`` count comes from ``all_rank_num_tokens`` when
provided (attention-DP padding support), matching the shape contract
of ``MoE.forward_fake`` and ``ConfigurableMoE._forward_chunk_impl``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType, Fp4QuantizedTensor
from ..interface import MoE, MoEWeightLoadingMode
from ..routing import BaseMoeRoutingMethod

__all__ = ["MegaMoEDeepGemmFusedMoE"]


# Process-level cache: one DG SymmBuffer shared across all MegaMoE layers.
# Same rationale as the legacy MnnvlMoe workspace singleton — MoE layers
# run serially per forward, so overlapping workspaces are unnecessary and
# per-layer allocation would need ~``num_layers × buffer_size`` symm
# memory (OOM on DSV3-scale). Keyed on (pg_id, config) so different EP
# subsets or shapes get separate buffers.
_SYMM_BUFFER_CACHE: dict = {}


def _import_deep_gemm():
    """Return the bundled ``tensorrt_llm.deep_gemm`` module.

    Strict: no fall-back to a standalone ``deep_gemm`` wheel. TRT-LLM
    guarantees the bundled binding is the compatible-by-construction
    source; silently switching to whatever external ``deep_gemm`` is
    installed on the box would make capability depend on pip state and
    break the guarantee.

    Raises ``_MegaMoEUnavailable`` when the bundled module is missing or
    too old (mega_moe symbols absent, or ``per_token_cast_to_fp8`` lacks
    ``use_packed_ue8m0``). ``can_implement`` catches this and returns
    ``(False, reason)`` cleanly.
    """
    import inspect

    try:
        from tensorrt_llm import deep_gemm as _dg
    except ImportError as e:
        raise _MegaMoEUnavailable(f"tensorrt_llm.deep_gemm not importable: {e}") from e

    missing = [
        n
        for n in (
            "fp8_fp4_mega_moe",
            "get_symm_buffer_for_mega_moe",
            "transform_weights_for_mega_moe",
        )
        if not hasattr(_dg, n)
    ]
    if missing:
        raise _MegaMoEUnavailable(
            f"tensorrt_llm.deep_gemm missing mega_moe symbols {missing}; "
            f"upgrade the TRT-LLM bundled DeepGEMM to a release that "
            f"includes fp8_fp4_mega_moe."
        )

    p_fp8 = getattr(_dg, "per_token_cast_to_fp8", None)
    if p_fp8 is None or "use_packed_ue8m0" not in inspect.signature(p_fp8).parameters:
        raise _MegaMoEUnavailable(
            "tensorrt_llm.deep_gemm.per_token_cast_to_fp8 does not accept "
            "use_packed_ue8m0=; upgrade the bundled DeepGEMM."
        )
    return _dg


def _import_dg_fp8_cast():
    """Return ``per_token_cast_to_fp8`` from the same bundled module the kernel lives in.

    Only this cast is used on the hot path; we do NOT require
    ``per_token_cast_to_fp4`` since the backend consumes MXFP4 weights
    that are already pre-quantized by the caller (see ``load_weights``)
    and the transform runs on those raw bytes.
    """
    dg = _import_deep_gemm()
    return dg.per_token_cast_to_fp8


class _MegaMoEUnavailable(RuntimeError):
    """Signals that the bundled DeepGEMM doesn't expose the full mega_moe API.

    ``can_implement`` converts this into a clean ``(False, reason)``
    instead of a hard import error.
    """


def _ue8m0_uint8_to_fp32(sf_uint8: torch.Tensor) -> torch.Tensor:
    """Convert UE8M0 stored as uint8 → fp32 with matching numeric value.

    UE8M0 is an 8-bit unsigned exponent (no sign, no mantissa). The
    numerically-equivalent fp32 has the same 8 exponent bits with sign=0
    and mantissa=0 — achieved by shifting 23 bits left.
    """
    assert sf_uint8.dtype == torch.uint8
    return (sf_uint8.to(torch.int32) << 23).contiguous().view(torch.float32)


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


class MegaMoEDeepGemmFusedMoE(MoE):
    """MoE backend wrapping DeepGEMM's fused ``fp8_fp4_mega_moe`` kernel."""

    _SUPPORTED_ACTIVATION_DTYPES = frozenset({torch.bfloat16})

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
        sm = get_sm_version()
        if sm != 100:
            return False, (
                f"MegaMoEDeepGemmFusedMoE requires SM100 (only arch with "
                f"sm100_fp8_fp4_mega_moe.cuh in DeepGEMM); got SM{sm}"
            )
        if dtype_activation not in cls._SUPPORTED_ACTIVATION_DTYPES:
            return False, (
                f"MegaMoEDeepGemmFusedMoE supports activations in "
                f"{cls._SUPPORTED_ACTIVATION_DTYPES}, got {dtype_activation}"
            )
        if swiglu_gptoss_style:
            return False, "MegaMoEDeepGemmFusedMoE does not support swiglu_gptoss_style"
        if quant_algo != QuantAlgo.W4A8_MXFP4_MXFP8:
            return False, (
                f"MegaMoEDeepGemmFusedMoE supports W4A8_MXFP4_MXFP8 only, got {quant_algo}"
            )
        # Packed-UE8M0 per-token SF layout: 4 u8 scales reinterpreted as
        # int32 per 128-element row stride, so hidden/intermediate must be
        # divisible by 128. Divisible-by-32 shapes like ``hidden=2880``
        # would quantize cleanly but fail the int32 reshape at first
        # forward — reject at ``can_implement`` time so the factory can
        # fall back cleanly.
        if hidden_size is not None and hidden_size % 128 != 0:
            return False, (
                f"MegaMoEDeepGemmFusedMoE requires hidden_size % 128 == 0 "
                f"(packed-UE8M0 int32 SF stride); got hidden_size={hidden_size}"
            )
        if intermediate_size is not None and intermediate_size % 128 != 0:
            return False, (
                f"MegaMoEDeepGemmFusedMoE requires intermediate_size % 128 == 0 "
                f"(packed-UE8M0 int32 SF stride); got intermediate_size="
                f"{intermediate_size}"
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

        # Phase 1 — assert supported topologies early.
        assert self.tp_size == 1, (
            f"MegaMoEDeepGemmFusedMoE Phase 1 is EP-only (moe_tp_size=1); got tp_size={self.tp_size}"
        )
        assert self.cluster_size == 1, (
            f"MegaMoEDeepGemmFusedMoE Phase 1 assumes cluster_size=1; got cluster_size={self.cluster_size}"
        )
        assert num_experts % max(self.ep_size, 1) == 0

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
                f"MegaMoEDeepGemmFusedMoE with enable_attention_dp=True requires "
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
            "MegaMoEDeepGemmFusedMoE does not support apply_router_weight_on_input. "
            "DG's fp8_fp4_mega_moe applies routing weights on the MoE "
            "output, not by pre-scaling the input — the two paths are "
            "not equivalent. Use a different MoE backend for models that "
            "require pre-scaling, or extend the kernel call."
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

        # Deferred: weight transform + SymmBuffer allocation happen on
        # ``post_load_weights`` which is also a global sync point.
        self._symm_buffer = None
        self._t_l1 = None
        self._t_l2 = None
        self._weights_loaded = False

        self._create_mega_weights()

    def _supports_load_balancer(self) -> bool:
        # Phase 1: EPLB off. Follow-up via token_selected_slots.
        return False

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
                "MegaMoEDeepGemmFusedMoE requires torch.distributed to be "
                "initialized before module construction (mpirun or Ray)."
            )
        # Preferred: reuse the existing PG from the mapping (Ray / DeviceMesh).
        try:
            pg = self.mapping.moe_ep_group_pg
            logger.info(
                f"[MegaMoE] layer={self.layer_idx} using mapping.moe_ep_group_pg (DeviceMesh path)"
            )
            return pg
        except (NotImplementedError, AttributeError):
            pass
        # Fallback: degenerate to WORLD when EP spans all ranks.
        world_size = dist.get_world_size()
        if self.ep_size == world_size:
            logger.info(
                f"[MegaMoE] layer={self.layer_idx} using dist.group.WORLD "
                f"(EP == world_size == {world_size})"
            )
            return dist.group.WORLD
        raise RuntimeError(
            f"MegaMoEDeepGemmFusedMoE: cannot resolve EP ProcessGroup. The current "
            f"mapping does not expose ``moe_ep_group_pg`` and EP "
            f"({self.ep_size}) is a strict subset of world "
            f"({world_size}). Use DeviceMeshTopology (TLLM_DISABLE_MPI=1) "
            f"so the EP PG is constructed once at Mapping init, or set "
            f"ep_size == world_size."
        )

    # ------------------------------------------------------------------
    # Weight lifecycle
    # ------------------------------------------------------------------
    def _create_mega_weights(self) -> None:
        E = self.expert_size_per_partition
        H = self.hidden_size
        inter = self.intermediate_size
        # Divisible-by-128 (not 32) — packed-UE8M0 SF in
        # ``_quantize_bf16_to_fp8_ue8m0`` reinterprets ``H/32`` bytes as
        # ``H/128`` int32 values per row, so H/32 must be a multiple of 4.
        # ``can_implement`` rejects before we get here under the factory
        # path, but keep the asserts as defensive dev checks for direct
        # MegaMoEDeepGemmFusedMoE construction.
        assert H % 128 == 0, f"hidden {H} must be divisible by 128"
        assert inter % 128 == 0, f"intermediate {inter} must be divisible by 128"

        self.register_parameter(
            "w3_w1_weight",
            nn.Parameter(torch.empty(E, inter * 2, H // 2, dtype=torch.uint8), requires_grad=False),
        )
        self.register_parameter(
            "w3_w1_weight_scale",
            nn.Parameter(
                torch.empty(E, inter * 2, H // 32, dtype=torch.uint8), requires_grad=False
            ),
        )
        self.register_parameter(
            "w2_weight",
            nn.Parameter(torch.empty(E, H, inter // 2, dtype=torch.uint8), requires_grad=False),
        )
        self.register_parameter(
            "w2_weight_scale",
            nn.Parameter(torch.empty(E, H, inter // 32, dtype=torch.uint8), requires_grad=False),
        )

    def create_weights(self):
        # No-op: allocated in __init__. Provided for MoE-contract symmetry.
        return

    # ----- Per-loading-mode weight unpacking helpers -------------------
    def _iter_vanilla_expert_weights(self, w: Dict, expert_id: int):
        """Return (w1, w3, w2, w1_sf, w3_sf, w2_sf) as CPU uint8 tensors.

        Used for VANILLA / W4A8_CUSTOM key schema
        ``{eid}.w*.weight[_scale]``.
        """
        return (
            w[f"{expert_id}.w1.weight"],
            w[f"{expert_id}.w3.weight"],
            w[f"{expert_id}.w2.weight"],
            w[f"{expert_id}.w1.weight_scale"],
            w[f"{expert_id}.w3.weight_scale"],
            w[f"{expert_id}.w2.weight_scale"],
        )

    def _iter_fused_gate_up_expert_weights(self, w: Dict, expert_id: int):
        """FUSED_GATE_UP_PROJ schema (gpt-oss / llama).

        Mirrors ``MoEWeightLoader.load_expert_weights`` in quantization.py:
        ``gate_up_proj[expert_id]`` is transposed then chunked along dim 0
        into (w1, w3). ``down_proj[expert_id]`` is transposed to get w2.
        """
        w1w3 = w["gate_up_proj"][expert_id].transpose(0, 1).contiguous()
        w1, w3 = w1w3.chunk(2, dim=0)
        w2 = w["down_proj"][expert_id].transpose(0, 1).contiguous()

        w1w3_sf = w["gate_up_proj_weight_scale"][expert_id].transpose(0, 1).contiguous()
        w1_sf, w3_sf = w1w3_sf.chunk(2, dim=0)
        w2_sf = w["down_proj_weight_scale"][expert_id].transpose(0, 1).contiguous()
        return w1, w3, w2, w1_sf, w3_sf, w2_sf

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False) -> None:
        """Load MXFP4 weights + UE8M0 block scales for this rank's experts.

        Supports VANILLA / W4A8_CUSTOM (per-expert ``{eid}.w*.*`` keys)
        and FUSED_GATE_UP_PROJ (stacked ``gate_up_proj`` / ``down_proj``)
        loading modes, matching ``MoEWeightLoader`` conventions.
        """
        assert len(weights) == 1, f"MegaMoE expects one weight dict, got {len(weights)}"
        w = weights[0]

        mode = self.weight_loading_mode
        if mode in (MoEWeightLoadingMode.VANILLA, MoEWeightLoadingMode.W4A8_CUSTOM):
            get_expert = self._iter_vanilla_expert_weights
        elif mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
            get_expert = self._iter_fused_gate_up_expert_weights
        else:
            raise NotImplementedError(
                f"MegaMoE load_weights unsupported weight_loading_mode={mode}"
            )

        def _to_u8(t: torch.Tensor) -> torch.Tensor:
            return t.cuda().view(torch.uint8)

        local_ids = list(self.initial_local_expert_ids)
        for slot_id, expert_id in enumerate(local_ids):
            w1, w3, w2, w1_sf, w3_sf, w2_sf = get_expert(w, expert_id)

            # Stack [w1 | w3] along intermediate dim. DG's
            # ``_interleave_l1_weights`` (deep_gemm/mega/__init__.py:78)
            # interprets the first half of the L1 weight as gate and the
            # second half as up. TRT-LLM's MoE convention (consistent
            # across ``modeling_gpt_oss.py`` and the ``FUSED_GATE_UP_PROJ``
            # loader at ``quantization.py:362-365``) maps
            # ``w1 = gate_proj``, ``w3 = up_proj`` (HF ``gate_up_proj``
            # is ``[gate | up]`` along out_dim, ``chunk(2)[0]`` -> w1,
            # ``chunk(2)[1]`` -> w3). Therefore the right cat order is
            # ``[w1 | w3]`` so DG sees ``[gate | up]`` and computes
            # ``silu(gate) * up`` correctly. The earlier ``[w3, w1]``
            # order silently swapped which side the silu was applied to,
            # producing ``silu(up) * gate`` and ~94% mismatch vs reference.
            self.w3_w1_weight.data[slot_id].copy_(
                torch.cat([_to_u8(w1), _to_u8(w3)], dim=0), non_blocking=True
            )
            self.w3_w1_weight_scale.data[slot_id].copy_(
                torch.cat([_to_u8(w1_sf), _to_u8(w3_sf)], dim=0), non_blocking=True
            )
            self.w2_weight.data[slot_id].copy_(_to_u8(w2), non_blocking=True)
            self.w2_weight_scale.data[slot_id].copy_(_to_u8(w2_sf), non_blocking=True)

        self._weights_loaded = True

    def post_load_weights(self) -> None:
        """Finalize loaded weights for the MegaMoE hot path.

        Allocates the DG SymmBuffer (collective rendezvous) and runs
        ``transform_weights_for_mega_moe`` on this rank's weights.

        Both operations happen here because (a) rendezvous is collective
        and must run at a globally-synchronous time (post-load is), and
        (b) the transform needs the loaded weight bytes. Both are
        idempotent via internal guards.

        Phase 1 reload limitations (tracked as follow-up, consciously
        skipped):
        * ``allow_partial_loading`` is ignored in ``load_weights`` —
          a partial load will overwrite an incomplete weight subset.
        * This method returns early once ``_t_l1`` is set, so repeated
          ``post_load_weights`` calls after a weight reload will keep
          the **stale** transformed tensors. Invalidate ``_t_l1`` /
          ``_t_l2`` manually if you reload weights at runtime.
        * ``_SYMM_BUFFER_CACHE`` is a process-global cache keyed on
          (pg_id, shape, experts-per-token) and is never evicted;
          long-running processes that construct many MegaMoE layers
          of the same shape share one buffer. OOM at buffer allocation
          is the only signal that invalidation is needed.
        EPLB migration (Phase 2) will require addressing all three.
        """
        assert self._weights_loaded, "post_load_weights before load_weights"
        dg = _import_deep_gemm()

        if self._symm_buffer is None:
            key = (
                id(self._ep_pg),
                self.num_experts,
                self.max_num_tokens,
                self.routing_method.experts_per_token,
                self.hidden_size,
                self.intermediate_size,
                self.activation,
            )
            cached = _SYMM_BUFFER_CACHE.get(key)
            if cached is None:
                cached = dg.get_symm_buffer_for_mega_moe(
                    self._ep_pg,
                    self.num_experts,
                    self.max_num_tokens,
                    self.routing_method.experts_per_token,
                    self.hidden_size,
                    self.intermediate_size,
                    True,  # use_fp8_dispatch
                    self.activation,
                )
                _SYMM_BUFFER_CACHE[key] = cached
                logger.info(
                    f"[MegaMoE] layer={self.layer_idx} allocated DG "
                    f"SymmBuffer: {cached.buffer.nbytes / 2**30:.2f} GiB"
                )
            self._symm_buffer = cached

        if self._t_l1 is not None:
            return

        E = self.expert_size_per_partition
        H = self.hidden_size
        inter = self.intermediate_size

        l1_sf_fp32 = _ue8m0_uint8_to_fp32(self.w3_w1_weight_scale)
        # Bundled ``tensorrt_llm.deep_gemm.transform_sf_into_required_layout``
        # expects a 3-tuple ``(gm_sfa, gm_sfb, gk)`` recipe; for a 3-tuple
        # recipe the C++ side also requires ``is_sfa`` to be set so it
        # picks between the SFA / SFB granularity. These weight scales are
        # the SFB (B-operand) side of the grouped GEMM so pass
        # ``is_sfa=False``. ``num_groups=E`` marks this as a per-expert
        # grouped SF tensor.
        l1_sf = dg.transform_sf_into_required_layout(
            l1_sf_fp32, mn=inter * 2, k=H, recipe=(1, 1, 32), num_groups=E, is_sfa=False
        )

        l2_sf_fp32 = _ue8m0_uint8_to_fp32(self.w2_weight_scale)
        l2_sf = dg.transform_sf_into_required_layout(
            l2_sf_fp32, mn=H, k=inter, recipe=(1, 1, 32), num_groups=E, is_sfa=False
        )

        l1_w = self.w3_w1_weight.view(torch.int8)
        l2_w = self.w2_weight.view(torch.int8)

        self._t_l1, self._t_l2 = dg.transform_weights_for_mega_moe((l1_w, l1_sf), (l2_w, l2_sf))
        logger.info(
            f"[MegaMoE] layer={self.layer_idx} weight transform done "
            f"t_l1=(w {tuple(self._t_l1[0].shape)}/{self._t_l1[0].dtype}, "
            f"sf {tuple(self._t_l1[1].shape)}/{self._t_l1[1].dtype})"
        )

    # ------------------------------------------------------------------
    # Abstract MoE-contract methods (not used in this backend)
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

    def run_moe(self, *args, **kwargs):
        raise NotImplementedError(
            "MegaMoE's fused kernel replaces run_moe — call ``forward_impl`` "
            "or ``run_with_prequant`` with pre-computed FP8+SF+topk."
        )

    # ------------------------------------------------------------------
    # Fast path: accept already-quantized inputs from the outer pipeline.
    # ------------------------------------------------------------------
    def run_with_prequant(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Kernel-only path: 4 x ``buf.copy_()`` + empty-alloc + kernel launch.

        Matches DG's own ``run_fused`` shape contract so the GPU work
        here is exactly what DG benchmarks report.

        Caller is responsible for:
          * slicing ``x_real`` / ``router_logits_real`` to ``num_tokens``
          * running the routing method to produce ``topk_idx`` (int64) and
            ``topk_weights`` (float32)
          * BF16 → FP8 per-token quant (``quantize_input`` above)
          * sizing ``num_tokens`` appropriately vs ``max_num_tokens``
        """
        dg = _import_deep_gemm()
        buf = self._symm_buffer
        assert buf is not None, "MegaMoE SymmBuffer not allocated — post_load_weights missing?"
        assert num_tokens <= self.max_num_tokens, (
            f"MegaMoE got {num_tokens} tokens but buffer is sized for "
            f"{self.max_num_tokens}. Raise model_config.moe_max_num_tokens."
        )

        if num_tokens > 0:
            buf.x[:num_tokens].copy_(x_fp8)
            buf.x_sf[:num_tokens].copy_(x_sf)
            buf.topk_idx[:num_tokens].copy_(topk_idx)
            buf.topk_weights[:num_tokens].copy_(topk_weights)

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

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------
    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        input_ids: Optional[torch.IntTensor] = None,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(x, Fp4QuantizedTensor):
            raise NotImplementedError(
                "MegaMoE Phase 1 expects BF16 activation; kernel does its own FP8 quant internally."
            )
        dg = _import_deep_gemm()

        assert do_finalize, "MegaMoE always finalizes inside the fused kernel"
        if output_dtype is None:
            output_dtype = x.dtype

        # ----- Resolve real (unpadded) token count -----------------------
        # MoE.forward_fake contract: return shape [num_tokens_real, H]
        # where num_tokens_real = all_rank_num_tokens[moe_ep_rank] when
        # provided (attention-DP padding case). x.shape[0] may be larger
        # than num_tokens_real under ``use_dp_padding=True``.
        # Phase 1 asserts ``ep_size == parallel_size`` so the per-EP-rank
        # entry is also the per-DP-rank entry.
        if all_rank_num_tokens is not None:
            num_tokens = int(all_rank_num_tokens[self.mapping.moe_ep_rank])
        else:
            num_tokens = x.shape[0]
        assert num_tokens <= x.shape[0], f"num_tokens ({num_tokens}) > x.shape[0] ({x.shape[0]})"
        assert num_tokens <= self.max_num_tokens, (
            f"MegaMoE got {num_tokens} tokens but buffer is sized for "
            f"{self.max_num_tokens}. Raise model_config.moe_max_num_tokens."
        )

        # Note: DO NOT short-circuit when num_tokens == 0. DG's
        # fp8_fp4_mega_moe is a collective on the EP symm-mem group —
        # skipping the kernel on a zero-token rank would hang peers
        # whose tokens route to experts on this rank (they block waiting
        # for this rank's kernel entry). DG's own test exercises this
        # with uneven per-rank token counts; we mirror that contract.

        buf = self._symm_buffer
        assert buf is not None, "MegaMoE SymmBuffer not allocated — post_load_weights missing?"

        if num_tokens > 0:
            # Slice to real tokens (skip DP-padded rows, if any).
            x_real = x[:num_tokens]
            router_logits_real = router_logits[:num_tokens]

            # ----- Routing ----------------------------------------------
            # Upstream ``BaseMoeRoutingMethod.apply`` takes only
            # ``router_logits``. ``input_ids`` is accepted by this method
            # for forward-compat but ignored at this layer.
            topk_idx, topk_weights = self.routing_method.apply(router_logits_real)
            topk_idx = topk_idx.to(torch.int64)
            topk_weights = topk_weights.to(torch.float32)

            # ----- Pre-quant activations via the fused Inductor path ----
            x_fp8, x_sf = self.quantize_input(x_real)

            # ----- Write into symm buffer -------------------------------
            buf.x[:num_tokens].copy_(x_fp8)
            buf.x_sf[:num_tokens].copy_(x_sf)
            buf.topk_idx[:num_tokens].copy_(topk_idx)
            buf.topk_weights[:num_tokens].copy_(topk_weights)

        # ----- Kernel launch (always, even when num_tokens == 0) --------
        y = torch.empty((num_tokens, self.hidden_size), dtype=torch.bfloat16, device=x.device)
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
