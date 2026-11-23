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
"""MegaMoE local CuteDSL NVFP4 backend of the MiniMax-M3.1 DEP targets.

``trtllm.cutedsl.mega_moe_local.nvfp4`` is the stock MegaMoE CuteDSL backend
(``TrtllmCutedslMegaMoeNvfp4Impl``) with the M3.1 host-side levers on top:

* ROUTEQ: the fused-comm scheduler's ``try_fused_route_quant`` routes, quantizes and stages into the launch bucket's
  kernel buffers in one launch (``noaux_tc_fp4_quantize_stage_out``), byte for byte what routing, ``quantize_input``
  and ``_stage_inputs`` write, padded tail included. From ``ROUTEQ_PORT_MIN_ROWS`` rows the seat is the routing launch
  plus FlashInfer's NVFP4 quantizer (ROUTEQ_PORT); above ``ROUTEQ_SPLIT_TAIL_MIN_CELLS`` padded cells two fills write
  the tail (ROUTEQ_SPLIT_TAIL). ROUTE_BESIDE_SHARED stages the seat on the aux stream beside a serial shared expert.
* the pad mask (``set_pad_mask``): the rows a model marks as padding route to no expert, with ids -1 and weights 0.
* the shared-expert placement (``impl_delta.SharedExpertPlanMixin``).

Importing this module registers it (``register``): its identity, and a rank immediately ahead of the stock leaf in the
``MEGAMOE_CUTEDSL`` family, so every layer it declines resolves to stock; ``TRTLLM_M31_DISABLE_MEGAMOE_LOCAL=1``
declines every layer. A runtime whose MoE resolution takes no out-of-tree leaf builds the stock leaf.

The TAIL lever (the routed output left unreduced as a :class:`MegaMoEDeferredTopkReduce` for the model's tail) is
declined on this runtime: the generation-phase kernel stores its pre-reduced plane bit-inverted for its own top-k
reduce, which the tail reduce does not read.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from typing import Dict, List, Optional, Tuple

import torch

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    MoEStaticCapability,
)
from tensorrt_llm._torch.moe.fused_moe.impl_identity import MoEImplDescriptor, MoEImplId
from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind, _reject
from tensorrt_llm._torch.moe.fused_moe.mega_moe import TrtllmCutedslMegaMoeNvfp4Impl
from tensorrt_llm._torch.moe.fused_moe.routing import (
    DeepSeekV3MoeRoutingMethod,
    Deepseekv3RoutingImpl,
    MiniMaxM3MoeRoutingMethod,
)
from tensorrt_llm.logger import logger

from ....native import family_ops
from .impl_delta import SharedExpertPlanMixin
from .local_plan import DISABLE_ENV, lever_enabled, megamoe_local_requested, vd_static_rejection

__all__ = [
    "MegaMoECuteDslLocal",
    "MegaMoEDeferredTopkReduce",
    "TrtllmCutedslMegaMoeLocalNvfp4Impl",
    "is_registered",
    "register",
]

#: Padded (row, top-k) cells above which the route-quant launch leaves the padded tail to two fills: the launch
#: resets the tail from a single CTA, which at prefill rows outlasts the routing and quantizing CTAs.
ROUTEQ_SPLIT_TAIL_MIN_CELLS = 4096
#: Rows from which the route-quant seat is the routing launch plus FlashInfer's NVFP4 quantizer writing the stage; the
#: single fused launch is faster below it.
ROUTEQ_PORT_MIN_ROWS = 2048
#: Why no launch leaves its top-k reduce to the caller on this runtime.
TAIL_RUNTIME_DECLINE = (
    "the generation-phase kernel stores its pre-reduced plane bit-inverted for its own top-k reduce, "
    "which the tail reduce does not read"
)


@dataclasses.dataclass(frozen=True)
class MegaMoEDeferredTopkReduce:
    """The routed output of a launch that left its top-k reduce to the caller.

    ``plane`` is the ``[T, K, H]`` BF16 view of the launch's pre-reduced plane in
    its shared workspace, which this rank's next MegaMoE launch overwrites: reduce
    it on the launch's stream before then. ``scores`` is the ``[T, K]`` FP32 top-k
    weights the reduce applies.
    """

    plane: torch.Tensor
    scores: torch.Tensor

    def add_into(self, shared: torch.Tensor) -> torch.Tensor:
        """``shared.add_(routed)`` of the stock path, bitwise; returns ``shared``."""
        from ....kernels.megamoe.megamoe_topk_reduce_tail import megamoe_topk_reduce_shared_add

        megamoe_topk_reduce_shared_add(
            self.plane, self.scores, shared, enable_pdl=get_env_enable_pdl()
        )
        return shared

    def add_rmsnorm_into(
        self,
        shared: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        q: Optional[torch.Tensor] = None,
        sf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``shared.add_(routed)`` then FlashInfer's ``fused_add_rmsnorm(shared, residual)``,
        bitwise while ``megamoe_topk_reduce_add_rmsnorm_matches_flashinfer`` holds; given ``q``
        and ``sf``, also ``trtllm::mxfp8_quantize(shared, True)`` of the normalized rows into them."""
        from ....kernels.megamoe.megamoe_topk_reduce_tail import megamoe_topk_reduce_add_rmsnorm

        megamoe_topk_reduce_add_rmsnorm(
            self.plane,
            self.scores,
            shared,
            residual,
            weight,
            eps,
            enable_pdl=get_env_enable_pdl(),
            q=q,
            sf=sf,
        )
        return shared, residual


# Routing methods whose ``apply`` on FP32 contiguous logits and bias is exactly
# ``Deepseekv3RoutingImpl.apply``, i.e. one ``noaux_tc_op`` call.
_ROUTE_QUANT_ROUTING_METHODS = (DeepSeekV3MoeRoutingMethod, MiniMaxM3MoeRoutingMethod)
# The fused route-quantize kernel's limits.
_ROUTE_QUANT_MAX_EXPERTS = 256
_ROUTE_QUANT_MAX_TOPK = 32
_ROUTE_QUANT_HIDDEN_MULTIPLE = 64


def _route_quant_key(x: torch.Tensor, router_logits: torch.Tensor) -> Tuple:
    """What identifies a route-quant seat's inputs: the memory, shape, strides and dtype of both tensors."""
    return tuple((t.data_ptr(), tuple(t.shape), t.stride(), t.dtype) for t in (x, router_logits))


def _flashinfer_nvfp4_quantizer(device: torch.device):
    """FlashInfer's NVFP4 quantizer module for the device, which writes into caller-owned outputs, or None.

    None also while a FlashInfer NVFP4 environment knob is set: those change the quantized bits.
    """
    try:
        from flashinfer.quantization import fp4_quantization
    except ImportError:
        return None
    if any(os.environ.get(name) for name in getattr(fp4_quantization, "NVFP4_QUANT_ENV_VARS", ())):
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return _load_flashinfer_nvfp4_quantizer(f"{major}{minor}")


@functools.cache
def _flashinfer_fp4_quantize_tail_args() -> Tuple:
    """The quantizer's trailing arguments after ``enable_pdl``: FlashInfer releases with the NVFP4 4over6 recipe take
    its code, here the environment's (plain NVFP4: the quantizer is not used while a FlashInfer NVFP4 knob is set)."""
    from flashinfer.quantization import fp4_quantization

    code = getattr(fp4_quantization, "NVFP4_4OVER6_CODE_FROM_ENV", None)
    return () if code is None else (code,)


@functools.cache
def _load_flashinfer_nvfp4_quantizer(arch: str):
    from flashinfer.quantization import fp4_quantization

    gen = getattr(fp4_quantization, f"gen_fp4_quantization_sm{arch}_module", None)
    if gen is None:
        return None
    try:
        return gen().build_and_load()
    except Exception as e:
        logger.warning(
            f"MegaMoECuteDslLocal: FlashInfer's SM{arch} NVFP4 quantizer did not load ({e})."
        )
        return None


class TrtllmCutedslMegaMoeLocalNvfp4Impl(SharedExpertPlanMixin, TrtllmCutedslMegaMoeNvfp4Impl):
    """``trtllm.cutedsl.mega_moe_local.nvfp4``; ``MegaMoECuteDslLocal`` is an alias.

    Tried ahead of the stock leaf unless ``TRTLLM_M31_DISABLE_MEGAMOE_LOCAL=1``,
    for layers the stock leaf admits on SM107 without a load balancer.
    """

    descriptor = MoEImplDescriptor(
        identity=MoEImplId("trtllm", "cutedsl", "mega_moe_local", "nvfp4"),
        scheduler_kind=MoESchedulerKind.FUSED_COMM,
        # Each half of a rank's experts belongs to one locality domain, which a load balancer would not keep.
        capabilities=MoEStaticCapability(supports_eplb=False),
        input_requirement=TrtllmCutedslMegaMoeNvfp4Impl.descriptor.input_requirement,
        doc=(
            "MegaMoE CuteDSL fused NVFP4 kernels with the MiniMax-M3.1 fused route-quantize seat, pad mask and "
            "shared-expert placement: NVFP4 weights and activations, SM107."
        ),
    )

    scheduler_kind = descriptor.scheduler_kind

    capabilities = descriptor.capabilities

    input_requirement = descriptor.input_requirement

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """Check static eligibility; never raises.

        A decline by the backend's own deployment checks on SM107 is logged once
        per reason, because a decline to the stock sibling resolves as
        ``pinned`` and is otherwise reported only at debug level.
        """
        environ = dict(d.env.env_flags)
        if not megamoe_local_requested(environ):
            return _reject(MoERejectReason.PATH_NOT_ENABLED, f"{DISABLE_ENV}=1 is set.")
        stock = super().can_implement(p, d)
        if not stock.eligible:
            return _reject(stock.reject_reason, stock.detail)
        if d.env.sm != 107:
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                f"MegaMoECuteDslLocal requires SM107; got SM{d.env.sm}.",
            )
        num_experts = d.num_slots if p.num_experts is None else p.num_experts
        detail = vd_static_rejection(
            ep_size=d.ep_size,
            num_experts=num_experts,
            num_slots=d.num_slots,
            moe_tp_size=d.tp_size,
            eplb_enabled=d.eplb_enabled,
            world_size=d.parallel_size,
        )
        if detail is not None:
            reason = (
                MoERejectReason.EPLB_UNSUPPORTED
                if d.eplb_enabled or d.num_slots != num_experts
                else MoERejectReason.TOPOLOGY_UNSUPPORTED
            )
            detail = f"MegaMoECuteDslLocal: {detail}."
            logger.warning_once(
                f"MegaMoECuteDslLocal declined ({reason.value}): {detail} "
                f"The layer runs the stock MegaMoECuteDsl.",
                key=f"megamoe_local_vd_{reason.value}",
            )
            return _reject(reason, detail)
        return MoEEligibility.ok()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._init_shared_expert_plan()
        self._route_quant_enabled = lever_enabled("ROUTEQ", os.environ)
        self._route_beside_shared_enabled = lever_enabled("ROUTE_BESIDE_SHARED", os.environ)
        self._route_quant_split_tail_enabled = lever_enabled("ROUTEQ_SPLIT_TAIL", os.environ)
        self._route_quant_port_enabled = lever_enabled("ROUTEQ_PORT", os.environ)
        self._tail_enabled = lever_enabled("TAIL", os.environ)
        # What ``prestage_route_quant`` launched for the next ``try_fused_route_quant``: its inputs' key and result,
        # and optionally a callable run when a call for other inputs uses the seat up (the seat's producer may have
        # left something out that the backend's own route-quant then needs).
        self._prestaged_route_quant: Optional[Tuple] = None
        self._pad_mask: Optional[torch.Tensor] = None
        self._route_quant_empty_stage: Dict[Tuple, Tuple[torch.Tensor, ...]] = {}
        # Per bucket, the rows the latest seat staged. Bookkeeping for callers that stage the seat themselves (the
        # dep2ctx boundary); the stock staging resets the whole padded tail on every call and does not read it.
        self._last_staged_T: Dict[int, int] = {}

    def _supports_load_balancer(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Pad mask
    # ------------------------------------------------------------------
    def set_pad_mask(self, pad_mask: Optional[torch.Tensor]) -> None:
        """Model hook: the next launch sends the rows where ``pad_mask`` (uint8, one entry per row) is nonzero to no
        expert, with ids -1 and weights 0 as in the padded tail.

        That launch consumes the mask and must have exactly ``pad_mask.numel()`` rows, so a chunked step takes none.
        """
        if pad_mask is not None:
            reason = self.pad_mask_decline_reason()
            if reason is not None:
                raise ValueError(f"MegaMoECuteDslLocal cannot mask pad rows: {reason}")
            if pad_mask.dtype != torch.uint8 or pad_mask.dim() != 1:
                raise ValueError(
                    f"pad_mask must be a 1-D uint8 tensor, got {pad_mask.dtype} of shape {tuple(pad_mask.shape)}"
                )
        self._pad_mask = pad_mask

    def pad_mask_decline_reason(self) -> Optional[str]:
        """Why weights 0 cannot silence a pad row's stale combine slots here, or None."""
        if self.apply_topk_in_fc1:
            return "the top-k weights apply in FC1, so the reduce is unweighted"
        return None

    def _take_pad_mask(self, num_tokens: int) -> Optional[torch.Tensor]:
        pad_mask, self._pad_mask = self._pad_mask, None
        if pad_mask is not None and pad_mask.numel() != num_tokens:
            raise ValueError(f"the pad mask has {pad_mask.numel()} rows, the launch {num_tokens}")
        return pad_mask

    # ------------------------------------------------------------------
    # TAIL: the top-k reduce left to the model's tail
    # ------------------------------------------------------------------
    def tail_static_decline_reason(self) -> Optional[str]:
        """Why no launch of this layer can leave its top-k reduce to the caller, or None."""
        if not self._tail_enabled:
            return "the TAIL lever is off"
        return TAIL_RUNTIME_DECLINE

    def tail_decline_reason(
        self, all_rank_num_tokens: Optional[List[int]], num_tokens: int
    ) -> Optional[str]:
        """Why the next forward's launch cannot leave its top-k reduce to the caller, or None. Never raises."""
        return self.tail_static_decline_reason()

    def request_deferred_topk_reduce(
        self, all_rank_num_tokens: Optional[List[int]], num_tokens: int
    ) -> Optional[str]:
        """Arm the next launch to return a :class:`MegaMoEDeferredTopkReduce`; returns why not, or None. Every launch
        returns the reduced tensor on this runtime (``tail_static_decline_reason``)."""
        return self.tail_decline_reason(all_rank_num_tokens, num_tokens)

    def cancel_deferred_topk_reduce(self) -> None:
        """Disarm the next launch: it returns the reduced tensor."""

    def _run_moe(self, **kwargs):
        try:
            return super()._run_moe(**kwargs)
        finally:
            # A mask the launch did not stage must not reach the next step.
            self._pad_mask = None

    # ------------------------------------------------------------------
    # ROUTEQ: the fused route-quantize seat
    # ------------------------------------------------------------------
    def _route_quant_decline_reason(
        self, x: torch.Tensor, router_logits: torch.Tensor
    ) -> Optional[str]:
        """Why ``try_fused_route_quant`` must decline, or None if the fused launch is exact.

        Never raises. The routing checks come first and need no device.
        """
        if not self._route_quant_enabled:
            return "the ROUTEQ lever is off"
        routing_method = self.routing_method
        if type(routing_method) not in _ROUTE_QUANT_ROUTING_METHODS:
            return f"routing method {type(routing_method).__name__}"
        impl = getattr(routing_method, "routing_impl", None)
        if type(impl) is not Deepseekv3RoutingImpl or not impl.is_fused:
            return "routing is not the fused Deepseekv3RoutingImpl"
        if impl.n_group != 1 or impl.topk_group != 1:
            return f"n_group {impl.n_group}, topk_group {impl.topk_group}"
        if getattr(routing_method, "output_dtype", torch.float32) != torch.float32:
            return "routing output dtype is not float32"
        if not isinstance(router_logits, torch.Tensor) or router_logits.dtype != torch.float32:
            return "router logits are not float32"
        bias = routing_method.e_score_correction_bias
        if not isinstance(bias, torch.Tensor) or bias.dtype != torch.float32:
            return "e_score_correction_bias is not float32"
        if router_logits.ndim != 2 or router_logits.shape[1] > _ROUTE_QUANT_MAX_EXPERTS:
            return f"router logits shape {tuple(router_logits.shape)}"
        if impl.top_k > _ROUTE_QUANT_MAX_TOPK:
            return f"top_k {impl.top_k}"
        if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
            return "the activation is not bf16"
        num_tokens, hidden = x.shape if x.ndim == 2 else (0, 0)
        if (
            num_tokens == 0
            or hidden != self.hidden_size
            or hidden % _ROUTE_QUANT_HIDDEN_MULTIPLE != 0
            or router_logits.shape[0] != num_tokens
            or bias.ndim != 1
            or bias.numel() != router_logits.shape[1]
        ):
            return "shapes"
        scale = self.fc31_input_scale
        if not all(
            isinstance(t, torch.Tensor) and t.is_cuda and t.device == x.device
            for t in (x, router_logits, bias, scale)
        ):
            return "a tensor is not on the activation's CUDA device"
        # The first AutoTuner.get() builds the tuner, which needs a CUDA device.
        if self.tactic_autotune or AutoTuner.get().is_tuning_mode:
            return "autotuning"
        if scale.dtype != torch.float32 or scale.numel() < 1:
            return "the input scale is not a float32 tensor"
        if not (x.is_contiguous() and router_logits.is_contiguous() and bias.is_contiguous()):
            return "non-contiguous inputs"
        if self._topk_idx_dtype != torch.int32:
            return "the kernel's top-k ids are not int32"
        if not self._weights_created or self.quant_method is None:
            return "weights are not created"
        chunk_max = self._active_launch_max_tokens
        if chunk_max is not None and chunk_max > self._maxt_buckets[-1]:
            return "the chunk exceeds the top bucket"
        launch_max_t = self._select_launch_max_tokens(chunk_max)
        if num_tokens > launch_max_t:
            return "more rows than the bucket"
        if self.ep_size > 1 and (
            launch_max_t not in self._symm_providers
            or self._symm_providers[launch_max_t].num_topk != impl.top_k
        ):
            return "no symmetric provider for the bucket and top_k"
        return None

    def try_fused_route_quant(
        self, x: torch.Tensor, router_logits: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Route, quantize and stage in one launch, straight into the bucket's kernel buffers.

        The launch writes what ``noaux_tc_op``, ``quantize_input`` and ``_stage_inputs``
        would, byte for byte, including the padded tail (ids -1, weights 0) and the
        rows of a pending ``set_pad_mask``. A seat ``prestage_route_quant`` launched for
        these same tensors is taken instead of a second launch. Above
        ``ROUTEQ_SPLIT_TAIL_MIN_CELLS`` padded cells the launch stages only the live rows
        and two fills write the same tail bytes
        (``TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_ROUTEQ_SPLIT_TAIL=1`` keeps the tail in the launch).
        From ``ROUTEQ_PORT_MIN_ROWS`` rows the routing launch alone writes the ids, weights and
        tail, and FlashInfer's NVFP4 quantizer the activations
        (``TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_ROUTEQ_PORT=1`` keeps the fused launch).
        """
        prestaged = self._take_prestaged_route_quant(x, router_logits)
        if prestaged is not None:
            return prestaged
        if self._route_quant_decline_reason(x, router_logits) is not None:
            return None
        num_tokens, hidden = x.shape
        impl = self.routing_method.routing_impl
        top_k = impl.top_k
        launch_max_t = self._select_launch_max_tokens(self._active_launch_max_tokens)
        bufs = self._acquire_buffers(
            top_k=top_k,
            hidden=hidden,
            device=x.device,
            output_dtype=x.dtype,
            launch_max_T=launch_max_t,
        )
        if bufs.activation_sf.shape[1] != hidden // 16:
            return None
        topk_weights = bufs.topk_weights[:, :top_k]
        pad_mask = self._take_pad_mask(num_tokens)
        masked = {} if pad_mask is None else {"pad_mask": pad_mask}
        quantizer = self._route_quant_port_quantizer(x, bufs, num_tokens)
        ops = family_ops()
        if quantizer is not None:
            ops.noaux_tc_route_stage_out(
                router_logits,
                self.routing_method.e_score_correction_bias,
                top_k,
                impl.routed_scaling_factor,
                bufs.topk_idx_local,
                topk_weights,
                **masked,
            )
            quantizer.fp4_quantize(
                x,
                self.fc31_input_scale,
                bufs.activation[:num_tokens],
                bufs.activation_sf[:num_tokens],
                16,
                False,
                False,
                False,
                False,
                get_env_enable_pdl(),
                *_flashinfer_fp4_quantize_tail_args(),
            )
            logger.info_once(
                f"MegaMoECuteDslLocal: from {ROUTEQ_PORT_MIN_ROWS} rows the route-quant seat is the routing launch "
                "and FlashInfer's NVFP4 quantizer (TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_ROUTEQ_PORT=1 turns it off).",
                key="megamoe_local_routeq_port",
            )
            return self._staged_route_quant(bufs, topk_weights, launch_max_t, num_tokens)
        split_tail = self._route_quant_split_tail_enabled and (
            (bufs.topk_idx_local.shape[0] - num_tokens) * top_k > ROUTEQ_SPLIT_TAIL_MIN_CELLS
        )
        stage_rows = num_tokens if split_tail else bufs.topk_idx_local.shape[0]
        ops.noaux_tc_fp4_quantize_stage_out(
            router_logits,
            self.routing_method.e_score_correction_bias,
            x,
            self.fc31_input_scale,
            top_k,
            impl.routed_scaling_factor,
            num_tokens,
            bufs.topk_idx_local[:stage_rows],
            topk_weights[:stage_rows],
            bufs.activation,
            bufs.activation_sf,
            *self._route_quant_empty(x.device, top_k, hidden),
            **masked,
        )
        if split_tail:
            bufs.topk_idx_local[num_tokens:, :top_k].fill_(-1)
            topk_weights[num_tokens:].zero_()
            logger.info_once(
                f"MegaMoECuteDslLocal: the route-quant launch staged {num_tokens} of "
                f"{bufs.topk_idx_local.shape[0]} rows and two fills reset the padded tail "
                "(TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_ROUTEQ_SPLIT_TAIL=1 turns it off).",
                key="megamoe_local_routeq_split_tail",
            )
        return self._staged_route_quant(bufs, topk_weights, launch_max_t, num_tokens)

    def _staged_route_quant(
        self, bufs, topk_weights: torch.Tensor, launch_max_t: int, num_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._last_staged_T[launch_max_t] = num_tokens
        return (
            bufs.topk_idx_local[:num_tokens],
            topk_weights[:num_tokens],
            bufs.activation[:num_tokens],
            bufs.activation_sf[:num_tokens],
        )

    def _route_quant_port_quantizer(self, x: torch.Tensor, bufs, num_tokens: int):
        """FlashInfer's NVFP4 quantizer when the seat runs as the routing launch plus that quantizer, else None.

        The quantizer writes linear scale factors in 16-row blocks and treats its outputs as contiguous rows,
        so the stage must hold the whole last block in contiguous buffers.
        """
        if not self._route_quant_port_enabled or num_tokens < ROUTEQ_PORT_MIN_ROWS:
            return None
        if (
            self.fc31_input_scale.numel() != 1
            or -(-num_tokens // 16) * 16 > bufs.activation_sf.shape[0]
            or not (bufs.activation.is_contiguous() and bufs.activation_sf.is_contiguous())
        ):
            return None
        return _flashinfer_nvfp4_quantizer(x.device)

    def route_beside_shared(self, max_tokens_per_rank: Optional[int]) -> bool:
        """Whether a step whose shared expert runs serially routes and stages beside it on the aux stream instead.

        Neither the router GEMM nor the route-quant seat reads the shared expert's output, so both can run beside it,
        and the routed launch starts after the two. Off with ``TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_ROUTE_BESIDE_SHARED=1``.
        """
        return self._route_beside_shared_enabled and self.shared_expert_serial(max_tokens_per_rank)

    def prestage_route_quant(
        self, x: torch.Tensor, router_logits: torch.Tensor, chunk_max_tokens: Optional[int]
    ) -> None:
        """Launch the route-quant seat for ``x`` now, at the bucket the scheduler picks for ``chunk_max_tokens``
        (``set_adaptive_launch_tokens``); the next ``try_fused_route_quant`` with the same tensors takes its result."""
        hint = self._active_launch_max_tokens
        self._active_launch_max_tokens = chunk_max_tokens
        try:
            result = self.try_fused_route_quant(x, router_logits)
        finally:
            self._active_launch_max_tokens = hint
        if result is not None:
            self._prestaged_route_quant = (_route_quant_key(x, router_logits), result)

    def _take_prestaged_route_quant(
        self, x: torch.Tensor, router_logits: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """The prestaged seat's result when it was launched for these tensors, else None; either way it is used up."""
        prestaged, self._prestaged_route_quant = self._prestaged_route_quant, None
        if prestaged is None:
            return None
        rows = int(x.shape[0])
        if prestaged[0] != _route_quant_key(x, router_logits):
            on_miss = prestaged[2] if len(prestaged) > 2 else None
            if on_miss is not None:
                on_miss()
            logger.warning_once(
                f"MegaMoECuteDslLocal: at {rows} rows the routed call missed the route-quant seat staged beside the "
                "shared expert (it was for other inputs) and launches its own.",
                key=f"megamoe_local_route_beside_shared_missed_{rows}",
            )
            return None
        logger.info_once(
            f"MegaMoECuteDslLocal: at {rows} rows the routed call took the route-quant seat staged beside the serial "
            "shared expert (TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_ROUTE_BESIDE_SHARED=1 turns it off).",
            key=f"megamoe_local_route_beside_shared_taken_{rows}",
        )
        return prestaged[1]

    def _route_quant_empty(self, device, top_k: int, hidden: int) -> Tuple[torch.Tensor, ...]:
        key = (str(device), top_k, hidden)
        if key not in self._route_quant_empty_stage:
            self._route_quant_empty_stage[key] = (
                torch.empty((0, top_k), dtype=torch.int32, device=device),
                torch.empty((0, top_k), dtype=torch.float32, device=device),
                torch.empty((0, hidden // 2), dtype=torch.uint8, device=device),
                torch.empty((0, hidden // 16), dtype=torch.uint8, device=device),
            )
        return self._route_quant_empty_stage[key]

    def _stage_inputs(
        self,
        *,
        bufs,
        x,
        x_sf,
        topk_idx,
        topk_weights,
        num_tokens,
        top_k,
        stage_activation,
        replica_plan=None,
    ) -> None:
        """Skips what ``try_fused_route_quant`` already wrote into these buffers."""
        staged = (
            num_tokens > 0
            and topk_idx.data_ptr() == bufs.topk_idx_local.data_ptr()
            and x.data_ptr() == bufs.activation.data_ptr()
            and x_sf.data_ptr() == bufs.activation_sf.data_ptr()
        )
        if not staged:
            super()._stage_inputs(
                bufs=bufs,
                x=x,
                x_sf=x_sf,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_tokens=num_tokens,
                top_k=top_k,
                stage_activation=stage_activation,
                replica_plan=replica_plan,
            )
            pad_mask = self._take_pad_mask(num_tokens)
            if pad_mask is not None:
                pad = pad_mask.bool().unsqueeze(1)
                bufs.topk_idx_local[:num_tokens, :top_k].masked_fill_(pad, -1)
                bufs.topk_weights[:num_tokens, :top_k].masked_fill_(pad, 0.0)
            return
        if topk_weights.data_ptr() != bufs.topk_weights.data_ptr():
            bufs.topk_weights[:num_tokens, :top_k].copy_(topk_weights, non_blocking=True)
        self._last_staged_T[bufs.topk_idx_local.shape[0]] = num_tokens


MegaMoECuteDslLocal = TrtllmCutedslMegaMoeLocalNvfp4Impl


def register() -> bool:
    """Make the backend selectable: its identity registered, ranked immediately ahead of the stock leaf in the
    ``MEGAMOE_CUTEDSL`` family, and its disable switch part of the MoE environment the resolution reads. Idempotent;
    returns whether the backend is selectable (False on a runtime without the resolution's registration hooks)."""
    from tensorrt_llm._torch.moe.fused_moe import impl_environment, moe_resolution

    rank = getattr(moe_resolution, "register_impl_ahead_of", None)
    env_flag = getattr(impl_environment, "register_optional_env_flag", None)
    if rank is None or env_flag is None:
        return False
    env_flag(DISABLE_ENV)
    rank(
        TrtllmCutedslMegaMoeLocalNvfp4Impl,
        ahead_of=TrtllmCutedslMegaMoeNvfp4Impl,
        family="MEGAMOE_CUTEDSL",
    )
    return True


def is_registered() -> bool:
    """Whether the MoE resolution can select the backend on this runtime."""
    from tensorrt_llm._torch.moe.fused_moe.moe_resolution import backend_family_of

    return backend_family_of(TrtllmCutedslMegaMoeLocalNvfp4Impl) == "MEGAMOE_CUTEDSL"


register()
