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
"""TrtllmCutedslFusedFc12Nvfp4Impl: ``trtllm.cutedsl.fused_fc12.nvfp4``.

FC1+FC2-fused CuteDSL NVFP4 MoE backend for Rubin (SM107). One persistent kernel
(``trtllm::cute_dsl_nvfp4_fc12_fused_rubin``) does gather + FC1 GEMM + gated
epilogue (SwiGLU, or SiTU for Kimi K3) + requant + FC2 GEMM + finalize
(scatter-add into ``moe_output``), so the FC1->FC2 intermediate never
round-trips through global memory.

Uses the grouped-GEMM NVFP4 weight layout (``NVFP4CuteDslFusedMoEMethod``),
``NvFp4WeightView``, ``quantize_input_nvfp4`` and ``CuteDslFusedMoENvfp4Runner``
(narrowed to the fused kernel's tiles) without inheriting from that leaf. No
locality-domain (uGPU) support.
"""

import os
from dataclasses import dataclass

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ....autotuner import AutoTuner
from ....custom_ops.cute_dsl_custom_ops import fc12_counter_layout, fc12_die_sched_requested
from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType, EventType
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
    MoEInputRequirement,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    MoEStaticCapability,
    require_comm_plan,
)
from ..impl_environment import MoEDep
from ..impl_identity import MoEImplDescriptor, MoEImplId, register_moe_impl
from ..interface import MoESchedulerKind, _reject
from ..quantization import MoEWeightLoadingMode, NVFP4CuteDslFusedMoEMethod
from ..routing import BaseMoeRoutingMethod
from .identity import KERNEL_FUSED_FC12, PROVIDER_TRTLLM, TECHNIQUE_CUTEDSL
from .kernel_inputs import (
    NvFp4WeightView,
    finalize_routing_scales,
    presorted_routing,
    quantize_input_nvfp4,
    sort_routing,
)
from .runners import CuteDslFusedMoENvfp4Runner

# TRTLLM_K3_FC12_DIE_LAYOUT=1 (default 0): on a two-die GPU (Rubin R200), post_load_weights
# re-homes this backend's expert weights (w3_w1_weight, w2_weight) into die-alternating memory
# (``die_locality.die_alternating_copy``: 2 MiB chunks alternating between the dies, same bytes
# and layout) and releases the originals; results are unchanged. TRTLLM_K3_FC12_DIE_SCHED=1 then
# lets FC12 CTAs read mostly their own die's weights. Not applied with DWDP or EPLB, which rebind
# or migrate the weights at run time.
_K3_FC12_DIE_LAYOUT_ENV = "TRTLLM_K3_FC12_DIE_LAYOUT"
# TRTLLM_K3_ULL_EXPERTS (default 0): up to 32 tokens, the Kimi-K3 small-batch FC1 + SiTU + FC2 +
# combine kernel of the Kimi K3 (k3_rubin) package (``_K3_ULL_EXPERTS_MODULE``) serves the call
# on the same weights and routing tables instead of the fused FC12 kernel.
_K3_ULL_EXPERTS_ENV = "TRTLLM_K3_ULL_EXPERTS"
# TRTLLM_K3_ULL_EXPERTS_DIE_MAJOR=1 (default 0; with TRTLLM_K3_ULL_EXPERTS=1): post_load_weights
# re-homes the expert weights and block scales die-major (k3_ull_experts.die_major_weights: every
# expert's first half of W1 row blocks and of W2 output rows on die 0, the second halves on die 1;
# the parameters keep their shapes over the new bytes) and the ULL experts kernel claims its work
# per die (EP and MoE-TP layouts). Takes precedence over TRTLLM_K3_FC12_DIE_LAYOUT. The FC12 kernel
# cannot read that layout: every call of such a layer must be served by the ULL kernel
# (k3_ull_experts.ull_experts_max_tokens), else the call raises.
_K3_ULL_DIE_MAJOR_ENV = "TRTLLM_K3_ULL_EXPERTS_DIE_MAJOR"
_K3_ULL_EXPERTS_MODULE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.moe.k3_ull_experts"
)


def _fc12_counter_words(num_local_experts: int, num_tokens: int, top_k: int) -> int:
    """FC12 counter words for ``moe_sort``'s tile count at the 128-row tile (more tiles than at
    256 rows) of ``num_tokens`` tokens with ``top_k`` experts each over ``num_local_experts``."""
    expanded = num_tokens * top_k
    filled = min(num_local_experts, expanded)
    max_tiles = filled + (expanded - filled) // 128
    return fc12_counter_layout(max_tiles, die_sched=fc12_die_sched_requested())[2]


def _k3_ull_experts():
    """The Kimi-K3 (k3_rubin) package's ULL experts module (imported on first use)."""
    import importlib

    return importlib.import_module(_K3_ULL_EXPERTS_MODULE)


@dataclass
class Fc12ZeroedOutput:
    """Output and FC12 counters of one ``CuteDslFc12FusedMoE`` call, zeroed ahead of it.

    ``buffer`` (int32) holds the BF16 ``output`` ``[num_tokens, hidden_size]``, then, from a
    128-byte boundary, ``counters`` (enough words for ``fc12_counter_layout`` of every tile count
    the routing can produce). The caller zeroes ``buffer`` on a stream ordered after its
    allocation, records ``ready`` there and sets ``CuteDslFc12FusedMoE.zeroed_output``.
    """

    buffer: torch.Tensor
    output: torch.Tensor
    counters: torch.Tensor
    ready: torch.cuda.Event


class CuteDslFc12FusedMoENvfp4Runner(CuteDslFusedMoENvfp4Runner):
    """Outer autotune runner for the fused FC12 backend.

    Identical to the parent except the routing-tile candidate set: the fused
    kernel supports the 128-wide tile (1-CTA) and the 256-wide tile (2-CTA,
    cluster (2,1)), so restrict ``_tile_sizes`` to ``[128, 256]`` (the parent
    also offers 512, which the fused v1 kernel does not support). The inner
    fused runner derives mma_tiler_m == tile_size and cluster M == tile_size //
    128 from the selected tile, mirroring the CuteDSL grouped-GEMM runners.
    """

    @staticmethod
    def _tile_sizes() -> list[int]:
        return [128, 256]


@register_moe_impl
class TrtllmCutedslFusedFc12Nvfp4Impl(MoEImplBase):
    """``trtllm.cutedsl.fused_fc12.nvfp4``: FC1+FC2-fused CuteDSL NVFP4 MoE (Rubin/SM107).

    Takes the constructor arguments of ``CuteDslFusedMoEBase``.

    When the top-k ids carry ``moe_sort`` outputs for this rank (see
    :func:`presorted_routing`; the Kimi-K3 routing tail attaches them), the
    backend runs at their tile size instead of the autotuned one and skips
    ``moe_sort``.

    A caller that zeroed the output and the kernel's counters ahead of the call
    (:meth:`new_zeroed_output`, ``zeroed_output``) saves the counter fill and
    the output memset the fused op otherwise issues right before the kernel.
    """

    #: Set right before a call by a caller that zeroed its output and counters
    #: (:class:`Fc12ZeroedOutput`); taken by that call's ``run_moe_nvfp4``.
    zeroed_output: Fc12ZeroedOutput | None = None
    # ``(output, counters)`` of the taken ``zeroed_output`` until the fused op
    # on that output consumes them.
    _zeroed_counters: tuple[torch.Tensor, torch.Tensor] | None = None

    # ConfigurableMoE reads this off the backend type: the fused kernel takes
    # per-tile expert ranges, so EP need not divide the expert count.
    _supports_non_divisible_ep: bool = True

    descriptor = MoEImplDescriptor(
        identity=MoEImplId(PROVIDER_TRTLLM, TECHNIQUE_CUTEDSL, KERNEL_FUSED_FC12, "nvfp4"),
        scheduler_kind=MoESchedulerKind.EXTERNAL_COMM,
        capabilities=MoEStaticCapability(
            supports_dwdp=True,
            supports_eplb=True,
            supports_apply_router_weight_on_input=True,
        ),
        input_requirement=MoEInputRequirement(routing_scales_dtype=torch.float32),
        doc="Rubin (SM107) NVFP4 fused FC1 + SwiGLU/SiTU + FC2 + finalize persistent "
        "CuTe DSL grouped GEMM.",
    )

    # FC1 weights are the gate/up pair and the epilogue is a trace-time
    # specialization of SwiGLU (+ clamp) or SiTU, whose two soft-caps arrive as
    # the uniform-scalar alpha/beta pair (``SiTuActivation.constants()``). Relu2
    # is not gated and is not supported, so the resolver turns such layers down
    # here.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.SiTu}),
        alpha_beta=ActivationParamShape.UNIFORM_SCALAR,
        limit=ActivationParamShape.UNIFORM_SCALAR,
        limit_when_absent=float("inf"),
    )

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------
    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """FC12 fused CuteDSL: NVFP4 on Rubin (SM107) only.

        Pure gate -- reads only ``p`` and ``d`` (no ``get_sm_version()``, no
        import probes, no ``os.environ``); the frozen environment lives in
        ``d.env``, including whether the installed CuTe DSL carries the Rubin
        helpers (``MoEDep.CUTEDSL_RUBIN``).
        """
        sm_version = d.env.sm

        # Output is hardcoded to bfloat16, so activation must match.
        if p.dtype_act != torch.bfloat16:
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                f"CuteDslFc12FusedMoE only supports bfloat16 activation, got {p.dtype_act}",
            )

        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                "CuteDslFc12FusedMoE does not support swiglu_gptoss_style",
            )

        if p.quant_algo != QuantAlgo.NVFP4:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"CuteDslFc12FusedMoE only supports NVFP4, got quant_algo={p.quant_algo}",
            )

        if sm_version != 107:
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                f"CuteDslFc12FusedMoE targets Rubin (SM107), got SM{sm_version}",
            )

        # The fused FC12 kernel scatter-adds FC2 straight into moe_output;
        # there is no unfused FC2 variant, so a deployment that disabled
        # finalize fusion (moe_disable_finalize_fusion, or any LoRA) cannot
        # be served. Same guard the CuteDSL grouped-GEMM leaves apply on SM107.
        if not d.fused_finalize_enabled:
            return _reject(
                MoERejectReason.FINALIZE_FUSION_REQUIRED,
                "CuteDslFc12FusedMoE only has a fused-finalize FC2",
            )

        # Read off the collected environment so an offline tuner on a GPU-less
        # host reaches the verdict a serving process would.
        if not d.env.has_dep(MoEDep.CUTEDSL_RUBIN):
            return _reject(
                MoERejectReason.DEP_MISSING,
                "CuteDslFc12FusedMoE (SM107 NVFP4) requires CuTe DSL Rubin support",
            )

        return MoEEligibility.ok()

    # ------------------------------------------------------------------
    # Construction and weights.
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype | None = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream] | None = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
        apply_router_weight_on_input: bool = False,
        layer_idx: int | None = None,
        activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
        init_load_balancer: bool = False,
    ) -> None:
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
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            layer_idx=layer_idx,
            activation=activation,
            init_load_balancer=init_load_balancer,
        )
        self.apply_router_weight_on_input = apply_router_weight_on_input
        # can_implement already turned down deployments without fused finalize;
        # kept as the runner's enable_finalize_fusion flag for cache-key parity
        # with TrtllmCutedslGroupedGemmNvfp4Impl.
        self.use_fused_finalize = (
            not model_config.moe_disable_finalize_fusion and model_config.lora_config is None
        )
        # The scheduler records/waits ``event_dict[EventType.Main]`` around the
        # aux stream in multi-stream mode (moe_scheduler.py), so the event has to
        # exist even though the fused op issues its own moe_output memset.
        if self.aux_stream_dict is None:
            self.aux_stream_dict = {}
        self.event_dict = {EventType.Main: torch.cuda.Event()}
        self.scaling_vector_size = 16
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _get_quant_method(self) -> object:
        if (
            self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_any_quant(exclude_kv_cache=True)
            and self.quant_config.layer_quant_mode.has_nvfp4()
        ):
            # The fused kernel consumes w3_w1 / w2 and their block scales unchanged.
            return NVFP4CuteDslFusedMoEMethod()
        raise ValueError(f"CuteDslFc12FusedMoE only supports NVFP4, got {self.quant_config}")

    def _check_configs(self) -> None:
        assert self._weights_created
        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"

    def _supports_load_balancer(self) -> bool:
        return True

    def supports_moe_output_in_alltoall_workspace(self) -> bool:
        # The fused op finalizes straight into moe_output, which may be the
        # all-to-all combine workspace.
        return True

    def load_weights(self, weights: list[dict], allow_partial_loading: bool = False) -> None:
        super().load_weights(weights, allow_partial_loading=allow_partial_loading)
        # DWDP registers the freshly loaded expert weights with its handle
        # collector; same hook TrtllmCutedslGroupedGemmNvfp4Impl exposes
        # (supports_dwdp=True).
        dwdp_handle_collector = getattr(self, "dwdp_handle_collector", None)
        if dwdp_handle_collector is not None:
            dwdp_handle_collector.register_weights(self)

    # ------------------------------------------------------------------
    # Output and counters zeroed ahead of a call (caller-driven)
    # ------------------------------------------------------------------
    def new_zeroed_output(
        self, num_tokens: int, top_k: int, device: torch.device, ready: torch.cuda.Event
    ) -> Fc12ZeroedOutput:
        """Allocate, on the current stream, the output and FC12 counters of a call on
        ``num_tokens`` tokens with ``top_k`` experts each (not zeroed yet).

        The fused op checks the counter size again.
        """
        counter_words = _fc12_counter_words(self.expert_size_per_partition, num_tokens, top_k)
        out_elems = num_tokens * self.hidden_size
        # BF16 pairs per int32 word; the counters start on a 128-byte line.
        out_words = (out_elems + 1) // 2
        out_words = (out_words + 31) // 32 * 32
        buffer = torch.empty(out_words + counter_words, dtype=torch.int32, device=device)
        output = (
            buffer[:out_words].view(torch.bfloat16)[:out_elems].view(num_tokens, self.hidden_size)
        )
        return Fc12ZeroedOutput(buffer, output, buffer[out_words:], ready)

    def drop_zeroed_output(self) -> None:
        """Join a ``zeroed_output`` no call took, so the current stream may reuse its buffer."""
        zeroed, self.zeroed_output = self.zeroed_output, None
        if zeroed is not None:
            zeroed.ready.wait()

    def _take_zeroed_output(
        self, moe_output: torch.Tensor | None, num_tokens: int
    ) -> torch.Tensor | None:
        """Take ``zeroed_output`` for this call: join its stream, then return its output in place
        of ``moe_output`` if the caller passed none or that same output and the shape fits (its
        counters then wait for the fused op in ``_zeroed_counters``), else ``moe_output``."""
        zeroed, self.zeroed_output = self.zeroed_output, None
        # Zeroed on another stream: join it before anything can touch the buffer.
        zeroed.ready.wait()
        self._zeroed_counters = None
        same = moe_output is not None and (
            moe_output.data_ptr() == zeroed.output.data_ptr()
            and tuple(moe_output.shape) == tuple(zeroed.output.shape)
        )
        if (moe_output is not None and not same) or tuple(zeroed.output.shape) != (
            num_tokens,
            self.hidden_size,
        ):
            return moe_output
        self._zeroed_counters = (zeroed.output, zeroed.counters)
        return zeroed.output

    # ------------------------------------------------------------------
    # Die placement of the expert weights (two-die GPUs)
    # ------------------------------------------------------------------
    def post_load_weights(self) -> None:
        super().post_load_weights()
        if (
            os.environ.get(_K3_ULL_DIE_MAJOR_ENV, "0") == "1"
            and self.place_expert_weights_die_major()
        ):
            return
        if os.environ.get(_K3_FC12_DIE_LAYOUT_ENV, "0") == "1":
            self.place_expert_weights_per_die()

    def place_expert_weights_die_major(self) -> bool:
        """TRTLLM_K3_ULL_EXPERTS_DIE_MAJOR: re-home the expert weights and their block scales
        die-major for the ULL experts kernel. Returns whether they are placed (placing twice is a
        no-op); the SM -> die table is measured here too, ahead of CUDA graph capture."""
        from ....die_locality import sm_die_table

        if getattr(self, "_ull_die_major", False):
            return True
        ull = _k3_ull_experts()
        names = ("w3_w1_weight", "w3_w1_weight_scale", "w2_weight", "w2_weight_scale")
        if ull.ull_experts_mode() <= 0 or not all(hasattr(self, n) for n in names):
            return False
        # die-major weights are read only by the ULL experts kernel, which needs a FC1 row order
        # it reads
        if ull.fc1_gate_up16(self) is None:
            return False
        if getattr(self, "dwdp_handle_collector", None) is not None or self._using_load_balancer():
            logger.warning_once(
                f"{_K3_ULL_DIE_MAJOR_ENV}: not with DWDP or EPLB, which rebind or migrate the "
                "expert weights",
                key="ull_die_major_skipped",
            )
            return False
        params = [getattr(self, n) for n in names]
        # [experts, rows, cols] tensors whose rows split into whole 128-row blocks
        if any(p.data.dim() != 3 or p.data.size(1) % 256 for p in params):
            return False
        copies, placed = ull.die_major_weights(*(p.data for p in params))
        if not placed:
            logger.warning_once(
                f"{_K3_ULL_DIE_MAJOR_ENV}: device has no die-split memory; expert weights keep "
                "the default placement",
                key="ull_die_major_unavailable",
            )
            return False
        for p, c in zip(params, copies):
            p.data = c.view(p.data.shape)
        del copies
        self._ull_die_major = True
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        sm_die_table(self.w3_w1_weight.device.index)
        return True

    def place_expert_weights_per_die(self) -> bool:
        """Re-home ``w3_w1_weight`` and ``w2_weight`` into die-alternating memory.

        Each weight is copied (same shape, dtype and bytes) into one VA range whose 2 MiB chunks
        alternate between the two dies, its parameter is pointed at the copy and the original
        is released. With ``TRTLLM_K3_FC12_DIE_SCHED`` the SM -> die table is measured here too,
        ahead of CUDA graph capture. Returns whether the weights were placed: not on devices
        with one die, with DWDP or EPLB, or when the driver does not home a chunk as requested.
        """
        from ....die_locality import die_alternating_copy, sm_die_table

        if getattr(self, "dwdp_handle_collector", None) is not None or self._using_load_balancer():
            logger.warning_once(
                f"{_K3_FC12_DIE_LAYOUT_ENV}: not placing FC12 expert weights per die with "
                "DWDP or EPLB, which rebind or migrate them",
                key="fc12_die_layout_skipped",
            )
            return False
        copies = []
        for name in ("w3_w1_weight", "w2_weight"):
            copy = die_alternating_copy(getattr(self, name).data)
            if copy is None:
                logger.warning_once(
                    f"{_K3_FC12_DIE_LAYOUT_ENV}: device has no die-alternating memory; "
                    "FC12 expert weights keep the default placement",
                    key="fc12_die_layout_unavailable",
                )
                return False
            copies.append(copy)
        for name, copy in zip(("w3_w1_weight", "w2_weight"), copies):
            getattr(self, name).data = copy
        del copies
        # The originals' blocks back to the driver: the next layer's copies are driver
        # allocations outside the caching allocator.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if fc12_die_sched_requested():
            sm_die_table(self.w3_w1_weight.device.index)
        return True

    def _build_local_weight_view(self) -> NvFp4WeightView:
        """Build the weight view from this backend's per-layer weights."""
        return NvFp4WeightView(
            w3_w1_weight=self.w3_w1_weight,
            fc1_weight_scale=self.quant_scales.fc1_weight_block,
            fc1_global_scale=self.quant_scales.fc1_global,
            w2_weight=self.w2_weight,
            fc2_weight_scale=self.quant_scales.fc2_weight_block,
            fc2_global_scale=self.quant_scales.fc2_global,
            expert_size_per_partition=self.expert_size_per_partition,
            slot_start=self.slot_start,
        )

    # ------------------------------------------------------------------
    # Input quantization and dispatch (NVFP4 only)
    # ------------------------------------------------------------------
    quantize_input = quantize_input_nvfp4

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        del workspace  # the fused kernel allocates its own intermediates
        plan = require_comm_plan(self, ctx)
        return self.run_moe_nvfp4(
            x=ctx.x,
            token_selected_experts=ctx.token_selected_experts,
            token_final_scales=finalize_routing_scales(self, ctx),
            x_sf=ctx.x_sf,
            moe_output=plan.moe_output,
            enable_alltoall=plan.enable_alltoall,
            weight_view=self._build_local_weight_view(),
        )

    # Fused-kernel dispatch, with its own autotuner key and outer runner so
    # FC12 tactics never mix with the grouped-GEMM leaf's.
    def run_moe_nvfp4(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor | None,
        x_sf: torch.Tensor | None = None,
        moe_output: torch.Tensor | None = None,
        enable_alltoall: bool = False,
        weight_view: NvFp4WeightView | None = None,
    ) -> torch.Tensor:
        assert self.has_nvfp4
        assert weight_view is not None
        output_dtype = torch.bfloat16

        if getattr(self, "zeroed_output", None) is not None:
            moe_output = self._take_zeroed_output(moe_output, token_final_scales.size(0))

        if moe_output is None:
            moe_output = torch.empty(
                (token_final_scales.size(0), self.hidden_size), dtype=output_dtype, device=x.device
            )
        else:
            assert moe_output.size() == (token_final_scales.size(0), self.hidden_size)
            assert moe_output.dtype == output_dtype

        # Empty micro-batches: skip autotuning (synthetic grouped-GEMM inputs
        # require at least one output row).
        if token_selected_experts.size(0) == 0:
            return moe_output

        effective_top_k = token_selected_experts.size(-1)
        tuner = AutoTuner.get()
        runner = CuteDslFc12FusedMoENvfp4Runner(
            forward_impl=self.run_moe_nvfp4_impl,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=weight_view.expert_size_per_partition,
            local_expert_offset=weight_view.slot_start,
            enable_finalize_fusion=self.use_fused_finalize,
            enable_alltoall=enable_alltoall,
            # The inner kernel cache keys on the epilogue; the outer AutoTuner
            # unique_id must too, or SwiGLU and SiTU layers share a tactic.
            workload_identity=self._epilogue_identity(),
        )
        inputs = [x, token_selected_experts, token_final_scales, x_sf, moe_output, weight_view]
        # Routing tables a router attached for this rank fix the tile size:
        # running at their tile skips moe_sort (the Kimi-K3 routing tail emits
        # 128-row tiles, the best FC12 tile at decode batch sizes).
        presorted = presorted_routing(
            token_selected_experts, weight_view.slot_start, weight_view.expert_size_per_partition
        )
        if presorted is not None and presorted[0] in CuteDslFc12FusedMoENvfp4Runner._tile_sizes():
            return runner(inputs, tactic=presorted[0])
        _, best_tactic = tuner.choose_one(
            "CuteDslFc12FusedMoE::run_moe_nvfp4",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    def _situ_betas(self) -> tuple[float, float]:
        """``(situ_beta, situ_linear_beta)`` for the fused op; -1 disables them.

        ``SiTuActivation.constants()`` lands gate_softcap in ``act_alpha`` and
        linear_softcap in ``act_beta``, the same mapping
        ``TrtllmCutedslGroupedGemmNvfp4Impl`` uses.
        """
        if self.activation_type != ActivationType.SiTu:
            return -1.0, -1.0
        return float(self.act_alpha), float(self.act_beta)

    def _epilogue_identity(self) -> tuple[int, float, float]:
        return (int(self.activation_type), *self._situ_betas())

    def run_moe_nvfp4_impl(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor | None,
        x_sf: torch.Tensor,
        moe_output: torch.Tensor,
        weight_view: NvFp4WeightView,
        enable_alltoall: bool = False,
        tile_size: int = 128,
        recv_expert_count: torch.Tensor | None = None,
        deep_ep_expert_capacity: int | None = None,
        use_count_native_expert_metadata: bool = False,
    ) -> torch.Tensor:
        """Single fused FC1+FC2 op."""
        # The shared CuteDslFusedMoENvfp4Runner.forward always forwards the
        # DeepEP direct-metadata knobs. This backend never enables that path
        # (can_use_deep_ep_direct_metadata() stays False), so they must arrive
        # unset; anything else means the runner and the backend disagree.
        assert recv_expert_count is None and deep_ep_expert_capacity is None, (
            "CuteDslFc12FusedMoE does not consume DeepEP expert-major metadata"
        )
        assert not use_count_native_expert_metadata, (
            "CuteDslFc12FusedMoE does not support count-native DeepEP metadata"
        )
        effective_top_k = token_selected_experts.size(1)
        esp = weight_view.expert_size_per_partition
        situ_beta, situ_linear_beta = self._situ_betas()
        slot_start = weight_view.slot_start
        # Counters zeroed with ``moe_output`` (``zeroed_output``), for the one
        # call on that output; profiling calls on other tensors drop them.
        fused_kwargs = {}
        if getattr(self, "_zeroed_counters", None) is not None:
            zeroed_out, zeroed_counters = self._zeroed_counters
            self._zeroed_counters = None
            if zeroed_out is moe_output:
                fused_kwargs["zeroed_counters"] = zeroed_counters

        routing = sort_routing(
            self, token_selected_experts, token_final_scales, weight_view, tile_size
        )
        (
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
        ) = routing

        if os.environ.get(_K3_ULL_EXPERTS_ENV, "0") != "0":
            ull = _k3_ull_experts()
            if ull.ull_experts_applies(self, x.size(0), weight_view, tile_size, enable_alltoall):
                # Writes every element of moe_output.
                return ull.run_ull_experts(
                    self,
                    x,
                    x_sf,
                    moe_output,
                    weight_view,
                    token_final_scales,
                    routing,
                    effective_top_k,
                )
        if getattr(self, "ull_moeblk", None) is not None:
            self.ull_moeblk = None
            raise RuntimeError(
                "TRTLLM_K3_ULL_MOEBLK: the front published the top-k without moe_sort tables, but "
                "the ULL experts kernel does not serve this call"
            )
        if getattr(self, "_ull_die_major", False):
            raise RuntimeError(
                f"{_K3_ULL_DIE_MAJOR_ENV}: this layer's expert weights are die-major, which only "
                f"the ULL experts kernel reads; {x.size(0)} tokens are outside its range (lower "
                "max_num_tokens)"
            )

        # One fused op: gather + FC1 GEMM + gated act (SwiGLU or SiTU) + requant
        # + FC2 GEMM + finalize (scatter-add into moe_output = a2a combine
        # workspace).
        # fc1_alpha/fc2_alpha map 1:1 to the two-op path's per-expert global
        # scales (the kernel takes split alphas). The three atomic counters are
        # allocated + zeroed inside the op runner, and the moe_output zeroing
        # memset is issued there right before the fused kernel (so it is the
        # kernel's immediate stream predecessor and the PDL prologue can overlap
        # it; expanded_idx_to_permuted_idx / ep_size / enable_alltoall are
        # passed through for it), unless ``zeroed_counters`` come with an output
        # that is already zero: then the fused kernel directly follows the
        # routing on this stream.
        torch.ops.trtllm.cute_dsl_nvfp4_fc12_fused_rubin(
            input=x.view(torch.float4_e2m1fn_x2),
            fc1_weight=weight_view.w3_w1_weight.view(torch.float4_e2m1fn_x2),
            input_scale=x_sf.view(torch.uint8),
            fc1_weight_scale=weight_view.fc1_weight_scale.view(torch.uint8),
            fc1_alpha=weight_view.fc1_global_scale,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_sf=self.fc2_input_scale,
            fc2_weight=weight_view.w2_weight.view(torch.float4_e2m1fn_x2),
            fc2_weight_scale=weight_view.fc2_weight_scale.view(torch.uint8),
            fc2_alpha=weight_view.fc2_global_scale,
            output=moe_output,
            token_final_scales=token_final_scales,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=esp,
            local_expert_offset=slot_start,
            tile_size=tile_size,
            swiglu_limit=self.act_clamp,
            ep_size=self.mapping.moe_ep_size,
            enable_alltoall=enable_alltoall,
            scaling_vector_size=16,
            activation_type=int(self.activation_type),
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            **fused_kwargs,
        )
        return moe_output
