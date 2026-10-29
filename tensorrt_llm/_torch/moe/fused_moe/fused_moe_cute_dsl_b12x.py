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

from dataclasses import replace

import torch
from packaging.version import Version

from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...utils import ActivationType, Fp4QuantizedTensor
from .activation import MoEActivationSupport
from .cutlass import CutlassFusedMoEBase, TrtllmCutlassNvfp4Impl
from .cutlass.eligibility import BF16_ONLY, check_dtype, check_oss_cutlass_build
from .cutlass.lora import moe_lora_active
from .impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEInputRequirement,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    MoEStaticCapability,
    canonical_quant,
    normalize_quant,
    nvfp4_fc1_row_alignment_rejection,
    require_comm_plan,
    require_layer_quant_format,
)
from .impl_environment import MoEDep
from .interface import _reject

# Shared MoE output buffer pool, keyed by (max_num_tokens, hidden_size, dtype,
# device). ``B12xMoEWrapper.__init__`` allocates a private
# ``(max_num_tokens, hidden_size)`` output tensor per instance; with one
# wrapper per MoE layer that is ``num_layers * max_num_tokens * hidden_size``
# bytes of GPU memory holding identical-shape buffers that are written
# sequentially. We fold them into a single shared buffer because MoE layers
# run sequentially on the same CUDA stream, and the wrapper consumes its
# previous output before the next layer is dispatched. Within one layer, the
# scheduler runs the chunks of a multi-chunk forward on one stream and copies
# every chunk but the last out of it -- the trailing ``torch.cat`` copies that
# one -- because the leaves declare ``run_moe_reuses_buffers``.
_SHARED_MOE_OUTPUT_BUF: dict = {}

# ActivationType -> b12x activation string. b12x currently exposes "relu2"
# (Nemotron-style x * relu(x)) and "silu" (SwiGLU-style x * silu(gate)).
_ACTIVATION_MAP = {
    ActivationType.Relu2: "relu2",
    ActivationType.Swiglu: "silu",
}

# FlashInfer's SM12x W4A16 fused MoE (0.6.17 and 0.6.18) selects a TC-decode
# "ultra-wide" FC2 tile of tile_n=512 / tile_k=32 for small m whenever that
# collapses FC2 to FC1's wave count (m=3 and m=4 for Qwen3.6-35B-A3B on the
# 188-SM RTX PRO 6000). tile_k=32 sits below the tile_k>=64 floor of its
# generic ``_candidate_tile_fits`` check, which the auto-selection skips but
# the ``force_tile_config`` re-pin across the custom-op boundary does not, so
# the kernel it just selected is rejected with "force_tile_config fc2 tile
# (tile_k=32, tile_n=512) does not fit ..." and CUDA-graph warmup aborts
# executor init (nvbug 6721561). Accepting the ultra tile in the re-validation
# under the same rule that produced it is the upstream fix; this shim mirrors
# it so the pinned wheel behaves like the fixed one. Drop it once the
# flashinfer pin carries ``_ultra_wide_fc2_tile_fits``.
_FLASHINFER_W4A16_ULTRA_WIDE_FC2_TILE = (512, 32, 256)  # (tile_n, tile_k, cta_threads)
_FLASHINFER_W4A16_ULTRA_WIDE_FC2_SINCE = Version("0.6.17")


def _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation() -> bool:
    """Make flashinfer's W4A16 ``force_tile_config`` re-validation accept the
    ultra-wide FC2 tile its own auto-selection produces.

    Idempotent. Returns True when the shim is active, False when flashinfer is
    absent, predates the ultra-wide override, or already ships the fix.
    """
    try:
        import flashinfer
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_w4a16_kernel as kernel
    except ImportError:
        return False
    if Version(flashinfer.__version__) < _FLASHINFER_W4A16_ULTRA_WIDE_FC2_SINCE:
        return False
    if hasattr(kernel, "_ultra_wide_fc2_tile_fits"):
        return False
    upstream_fits = getattr(kernel, "_candidate_tile_fits", None)
    smem_footprint = getattr(kernel, "_shared_memory_footprint", None)
    scale_group_size = getattr(kernel, "_scale_group_size", None)
    if upstream_fits is None or smem_footprint is None or scale_group_size is None:
        return False
    if getattr(upstream_fits, "_trtllm_ultra_wide_fc2_shim", False):
        return True

    def candidate_tile_fits(
        *,
        problem_n: int,
        problem_k: int,
        cta_m_blocks: int,
        tile_n: int,
        tile_k: int,
        cta_threads: int,
        max_shared_mem: int,
        scale_format: str = "e4m3_k16",
        weight_layout: str = "packed",
        allow_logical_tail: bool = False,
    ) -> bool:
        if (int(tile_n), int(tile_k), int(cta_threads)) != _FLASHINFER_W4A16_ULTRA_WIDE_FC2_TILE:
            return upstream_fits(
                problem_n=problem_n,
                problem_k=problem_k,
                cta_m_blocks=cta_m_blocks,
                tile_n=tile_n,
                tile_k=tile_k,
                cta_threads=cta_threads,
                max_shared_mem=max_shared_mem,
                scale_format=scale_format,
                weight_layout=weight_layout,
                allow_logical_tail=allow_logical_tail,
            )
        # Same rule as the auto-selection override: exact N/K tiling, one scale
        # group per k-tile, and shared-memory fit.
        if int(problem_n) % int(tile_n) != 0 or int(problem_k) % int(tile_k) != 0:
            return False
        if int(tile_k) % int(scale_group_size(scale_format)) != 0:
            return False
        smem_bytes = smem_footprint(
            cta_m_blocks=cta_m_blocks,
            tile_n=tile_n,
            tile_k=tile_k,
            scale_format=scale_format,
            weight_layout=weight_layout,
        )
        return int(smem_bytes) <= int(max_shared_mem)

    candidate_tile_fits._trtllm_ultra_wide_fc2_shim = True
    candidate_tile_fits._trtllm_upstream_fits = upstream_fits
    kernel._candidate_tile_fits = candidate_tile_fits
    logger.info_once(
        f"flashinfer {flashinfer.__version__}: accepting the SM12x W4A16 ultra-wide FC2 "
        "tile (tile_n=512, tile_k=32) in force_tile_config re-validation (nvbug 6721561).",
        key="flashinfer_w4a16_ultra_wide_fc2_tile_shim",
    )
    return True


class CuteDslB12xFusedMoEBase:
    """What the two B12x SM120/121 leaves share: the gates, the format check,
    the quant method, the ``run_moe`` admission and the decode call, keyed by
    each leaf's ``b12x_quant_algo``.

    A mixin, because the NVFP4 leaf inherits the CUTLASS NVFP4 leaf (for
    prefill) while the W4A16 leaf inherits only ``CutlassFusedMoEBase``.
    ``CuteDslB12xFusedMoE`` names this class, so ``isinstance`` /
    ``issubclass`` against it matches both leaves.
    """

    #: Neither leaf publishes an identity; without this the NVFP4 leaf would
    #: publish its CUTLASS parent's.
    descriptor = None
    #: The one format the leaf serves; ``None`` on this family class, which no
    #: layer runs on. Compared as ``MoEProblem.identity_quant`` spells it, so the
    #: NVFP4 leaf also admits the NVFP4_AWQ / NVFP4_ARC calibration aliases.
    b12x_quant_algo: QuantAlgo | None = None
    #: Whether the b12x kernel quantizes activations. It does so against the
    #: global scales ``NVFP4CuteDslB12xFusedMoEMethod.transform_weights`` fixes
    #: at load time, so the gate declines ``force_dynamic_quantization``.
    b12x_quantizes_activations: bool = False

    # The decode path reads no expert bias, fuses no LoRA and has no EPLB
    # support; the gates still decline EPLB explicitly because the inherited
    # ``_supports_load_balancer()`` answers True. Router weights folded into the
    # input are out too: the decode path hands ``token_final_scales`` straight to
    # the b12x wrapper, which has no defined behaviour for the ``None`` the fold
    # leaves there.
    capabilities = MoEStaticCapability(
        supports_moe_lora=False,
        supports_dwdp=True,
        supports_expert_bias=False,
        supports_apply_router_weight_on_input=False,
    )

    # Pinned rather than inherited, so a change in the CUTLASS parent cannot
    # retarget these backends. ``B12xMoEWrapper`` keeps one workspace across
    # calls (FlashInfer caches it by shape even without CUDA graphs) and, with
    # CUDA graphs, returns a view of its output buffer, which
    # ``_SHARED_MOE_OUTPUT_BUF`` also shares across layers.
    input_requirement = MoEInputRequirement(
        routing_scales_dtype=torch.float32, run_moe_reuses_buffers=True
    )

    # ``B12xMoEWrapper`` reads and writes bfloat16 only. Pinned, so the NVFP4
    # leaf does not answer with its CUTLASS parent's set.
    supported_dtypes = BF16_ONLY

    # The kinds ``_ACTIVATION_MAP`` gates on; the decode kernel takes no
    # activation constants.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.Relu2})
    )

    # Read by ``ConfigurableMoE._reject_non_divisible_ep_backend()``; moot while
    # the gates reject ``ep_size != 1``.
    _supports_non_divisible_ep: bool = True

    # SM120 = desktop Blackwell (RTX 5090 / GB202); SM121 = GB10 / DGX Spark.
    _SUPPORTED_SM_VERSIONS = frozenset({120, 121})

    @classmethod
    def _identity_quant(cls) -> str:
        return normalize_quant(canonical_quant(cls.b12x_quant_algo))

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        if cls.b12x_quant_algo is None:
            return _reject(
                MoERejectReason.NOT_AN_IMPL,
                f"{cls.__name__} names the B12x family; query "
                f"CuteDslB12xNvfp4FusedMoE or CuteDslB12xW4a16Nvfp4FusedMoE",
            )
        if p.identity_quant != cls._identity_quant():
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"{cls.__name__} implements quant={cls._identity_quant()}, got {p.identity_quant}",
            )
        sm_version = d.env.sm
        if sm_version not in cls._SUPPORTED_SM_VERSIONS:
            sm_list = "/".join(f"SM{v}" for v in sorted(cls._SUPPORTED_SM_VERSIONS))
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                f"{cls.__name__} requires {sm_list}, got SM{sm_version}",
            )
        rejection = check_dtype(cls, p)
        if rejection is not None:
            return rejection
        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} does not support swiglu_gptoss_style",
            )
        if p.activation_type not in _ACTIVATION_MAP:
            supported = ", ".join(a.name for a in _ACTIVATION_MAP)
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"{cls.__name__} does not support activation {p.activation}; "
                f"supported: {supported}",
            )
        if d.moe_lora_enabled:
            return _reject(
                MoERejectReason.LORA_UNSUPPORTED,
                f"{cls.__name__} does not fuse routed-expert LoRA",
            )
        # The decode kernel ships in the FlashInfer wheel.
        if not d.env.has_dep(MoEDep.FLASHINFER):
            return _reject(
                MoERejectReason.DEP_MISSING,
                f"{cls.__name__} requires the flashinfer package",
            )
        if d.eplb_enabled:
            return _reject(
                MoERejectReason.EPLB_UNSUPPORTED,
                f"{cls.__name__} does not support the MoE load balancer",
            )
        # b12x quantizes the FC2 input inside the fused kernel against a scale
        # passed in, with no per-call amax to derive it from. Per-call scales
        # would therefore reach only the CUTLASS prefill chunks, and decode
        # would keep the load-time ones.
        if cls.b12x_quantizes_activations and d.force_dynamic_quantization:
            return _reject(
                MoERejectReason.DYNAMIC_QUANT_UNSUPPORTED,
                f"{cls.__name__} quantizes decode activations with load-time "
                "scales and cannot honour force_dynamic_quantization",
            )
        # No expert-parallel dispatch/combine kernel: EP must stay at 1.
        if d.ep_size != 1:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"{cls.__name__} requires ep_size == 1 (got {d.ep_size})",
            )
        # Attention-DP is a separate axis from EP: with moe_tp == tp the layer
        # can have ep_size == 1 and still sit behind a DP allgather /
        # reducescatter that the b12x wrapper has never been exercised under.
        # ``use_dp and parallel_size > 1`` is exactly ``mapping.dp_size > 1``
        # (``Mapping.dp_size`` is ``tp_size`` when attention-DP is on).
        if d.use_dp and d.parallel_size > 1:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"{cls.__name__} does not support attention-DP (parallel_size={d.parallel_size})",
            )
        rejection = nvfp4_fc1_row_alignment_rejection(p, d)
        if rejection is not None:
            return rejection
        # Last, so a build limit is never blamed for a problem the gates above
        # turn down anyway. Only the NVFP4 leaf issues the CUTLASS op, for its
        # prefill chunks.
        rejection = check_oss_cutlass_build(cls, p, d)
        if rejection is not None:
            return replace(
                rejection,
                detail="NVFP4 prefill routes through CUTLASS: " + rejection.detail,
            )
        return MoEEligibility.ok()

    def __init__(self, *args: object, **kwargs: object) -> None:
        # ``ModelConfig`` is consumed by the inherited ``__init__`` for cache
        # / mapping setup but isn't kept on ``self``. The b12x wrapper needs the
        # ``use_cuda_graph`` flag at construction time, so capture it here
        # before delegating.
        model_config = kwargs.get("model_config", None)
        self._b12x_use_cuda_graph = bool(getattr(model_config, "use_cuda_graph", False))

        super().__init__(*args, **kwargs)

        self._b12x_weights: dict | None = None
        self.b12x_wrapper = None

    def _check_quant_config_is_my_format(self) -> None:
        require_layer_quant_format(self, self._identity_quant())

    def _get_quant_method(self) -> object:
        """The b12x-aware NVFP4 quant method; its ``transform_weights`` attaches
        the ``b12x_wrapper`` / ``_b12x_weights`` the decode path reads."""
        from .quantization import NVFP4CuteDslB12xFusedMoEMethod

        return NVFP4CuteDslB12xFusedMoEMethod()

    def _reject_fp4_input(self, x: torch.Tensor | Fp4QuantizedTensor) -> None:
        """b12x quantizes activations itself and cannot take pre-quantized FP4."""
        if isinstance(x, Fp4QuantizedTensor):
            raise ValueError(
                f"{type(self).__name__} does not accept Fp4QuantizedTensor input "
                "on the b12x path; b12x performs its own input quantization."
            )

    def _admit_run_context(self, ctx: MoERunContext) -> None:
        """Refuse what neither route of the leaf honours, then join the
        input-ready event before any kernel reads ``ctx.x``.

        On both routes the leaf finalizes, fuses no routed-expert LoRA, routes
        nothing itself and holds every expert (its gate rejects
        ``ep_size > 1``), so a context asking otherwise fails here instead of
        having the request dropped. ``lora_params`` describes every adapter of
        the forward, so only MoE LoRA on this layer is such a request -- with
        no layer index to read it by, any ``lora_params`` is.
        ``all_rank_num_tokens`` has no reader on either route: ``run_moe`` uses
        it only for the all-to-all tuner shapes, and all-to-all is refused.
        """
        name = type(self).__name__
        if not ctx.do_finalize:
            raise NotImplementedError(
                f"{name} always finalizes, so it cannot return the unfinalized "
                "(gemm2_output, expert_weights, expanded_idx_to_permuted_idx) "
                "triple that do_finalize=False asks for."
            )
        if ctx.lora_params:
            # Reading the layer's adapters out of ``lora_params`` needs a
            # layer index. With none the leaf cannot tell MoE LoRA from the
            # adapters it may ignore, so it refuses every ``lora_params``
            # rather than return output a caller reads as adapted.
            if self.layer_idx is None:
                raise NotImplementedError(
                    f"{name} does not fuse routed-expert LoRA and has no "
                    "layer_idx to tell whether lora_params carries any; pass "
                    "layer_idx to create_moe_backend, or hand run_moe no "
                    "lora_params."
                )
            if moe_lora_active(ctx.lora_params, self.layer_idx):
                raise NotImplementedError(
                    f"{name} does not fuse routed-expert LoRA; got MoE LoRA "
                    f"tensors for layer {self.layer_idx}."
                )
        if ctx.token_selected_experts is None or ctx.token_final_scales is None:
            raise ValueError(
                f"{name} needs precomputed token_selected_experts and "
                "token_final_scales; it cannot route from router_logits, and it "
                "does not accept the None scales apply_router_weight_on_input "
                "leaves behind once the scheduler has folded them into x."
            )
        if require_comm_plan(self, ctx).enable_alltoall:
            raise NotImplementedError(
                f"{name} holds every expert (ep_size == 1) and cannot take "
                "tokens dispatched by all-to-all."
            )
        if ctx.x_ready_event is not None:
            ctx.x_ready_event.wait()

    @nvtx_range("[b12x] decode")
    def _run_b12x_decode(self, ctx: MoERunContext) -> torch.Tensor:
        """Run the FlashInfer b12x kernel. The path both leaves reach."""
        plan = require_comm_plan(self, ctx)
        moe_output = plan.moe_output
        if self.b12x_wrapper is None or self._b12x_weights is None:
            raise RuntimeError(
                f"{type(self).__name__}.run_moe called before "
                "process_weights_after_loading completed."
            )
        if ctx.x_sf is not None:
            raise ValueError(
                f"{type(self).__name__} expects unquantized input (x_sf=None) "
                "on the b12x path; got a precomputed scale factor."
            )
        # The b12x kernel writes bfloat16, which is why ``supported_dtypes``
        # admits bf16 activations alone: ``ConfigurableMoE`` derives
        # ``output_dtype`` from the activation, so a layer built through the
        # factory cannot get here. This is the backstop for a caller driving
        # ``run_moe`` directly.
        if ctx.output_dtype not in (None, torch.bfloat16):
            raise ValueError(
                f"{type(self).__name__} writes bfloat16 on the b12x path; "
                f"cannot honour output_dtype={ctx.output_dtype}."
            )

        # Annotate the kwargs spread + wrapper entry separately so we can
        # attribute the per-layer Python dispatch cost vs. the kernel cost.
        with nvtx_range("[b12x] wrapper.run"):
            out = self.b12x_wrapper.run(
                x=ctx.x,
                token_selected_experts=ctx.token_selected_experts,
                token_final_scales=ctx.token_final_scales,
                **self._b12x_weights,
            )

        # B12xMoEWrapper allocates its own output buffer for CUDA-graph
        # compatibility; copy into ``moe_output`` when the caller provided one.
        if moe_output is not None:
            with nvtx_range("[b12x] out_copy"):
                moe_output.copy_(out)
            return moe_output
        return out


#: Family name; both leaves subclass it. Not constructible: no MoE ancestry.
CuteDslB12xFusedMoE = CuteDslB12xFusedMoEBase


class CuteDslB12xNvfp4FusedMoE(CuteDslB12xFusedMoEBase, TrtllmCutlassNvfp4Impl):
    """NVFP4 on SM120 / SM121: CUTLASS for prefill, b12x for decode.

    Inherits the CUTLASS NVFP4 leaf because the prefill chunk it routes away is
    NVFP4 grouped GEMM, so ``super().run_moe`` / ``super().quantize_input``
    reach exactly the right implementation.

    Both weight layouts are staged once at load time on the same module -- the
    inherited ``post_load_weights`` builds the standard CUTLASS NVFP4 layout
    first, then ``NVFP4CuteDslB12xFusedMoEMethod`` adds the b12x views and
    wrapper on top -- so the per-call switch below picks a compute path and
    never swaps a layout.
    """

    b12x_quant_algo = QuantAlgo.NVFP4
    b12x_quantizes_activations = True

    # Prefill chunks (``x.shape[0] >= threshold``) route via CUTLASS NVFP4
    # GroupGEMM; decode (``x.shape[0] < threshold``) uses b12x. 64 cleanly
    # separates conc=1 prefill (m=2048 with ``max_num_tokens=2048``) from
    # decode (m=1) and stays robust against future chunked-prefill splits
    # that might shrink prefill chunk size.
    _PREFILL_VIA_CUTLASS_THRESHOLD = 64

    def _route_to_cutlass(self, x) -> bool:
        """Return ``True`` iff this call should take the inherited CUTLASS path.

        ``Fp4QuantizedTensor`` inputs always stay on the b12x path, which
        rejects them.
        """
        return isinstance(x, torch.Tensor) and x.shape[0] >= self._PREFILL_VIA_CUTLASS_THRESHOLD

    @nvtx_range("[b12x] quantize_input")
    def quantize_input(
        self,
        x: torch.Tensor | Fp4QuantizedTensor,
        post_quant_comm: bool = True,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prefill chunks take the inherited NVFP4 quantization so the
        downstream ``run_moe`` can call CUTLASS NVFP4 GroupGEMM. Decode chunks
        stay high precision, because b12x quantizes activations internally;
        only an AWQ pre-quant scale is applied to them here.
        """
        if self._route_to_cutlass(x):
            return super().quantize_input(x, post_quant_comm=post_quant_comm, **kwargs)
        self._reject_fp4_input(x)
        # An NVFP4_AWQ / NVFP4_ARC checkpoint carries a per-channel pre-quant
        # scale that the b12x kernel has no operand for; apply it here, as the
        # CUTLASS prefill path does, before b12x quantizes with fc31_input_scale.
        if self.fc31_act_scale is not None:
            x = x * self.fc31_act_scale
        return x, None

    @nvtx_range("[b12x] run_moe")
    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: dict | None = None,
    ) -> torch.Tensor:
        self._admit_run_context(ctx)
        if self._route_to_cutlass(ctx.x):
            # The inherited ``run_moe`` forwards ``output_dtype`` straight into
            # the C++ ``trtllm::fused_moe`` op, which requires a concrete
            # high-precision ``ScalarType`` (uint8 / FP4-packed activations are
            # rejected at the kernel epilogue with "Invalid output type Byte").
            # ``ConfigurableMoE.forward`` always fills ``output_dtype``, so this
            # only narrows the type for anything driving ``run_moe`` without it.
            _HIGH_PRECISION = {torch.float16, torch.bfloat16, torch.float32}
            cutlass_output_dtype = ctx.output_dtype
            if cutlass_output_dtype is None:
                cutlass_output_dtype = (
                    ctx.x.dtype
                    if isinstance(ctx.x, torch.Tensor) and ctx.x.dtype in _HIGH_PRECISION
                    else torch.bfloat16
                )
            return super().run_moe(
                replace(ctx, output_dtype=cutlass_output_dtype),
                workspace=workspace,
            )
        del workspace  # The b12x wrapper allocates its own intermediates.
        return self._run_b12x_decode(ctx)


class CuteDslB12xW4a16Nvfp4FusedMoE(CuteDslB12xFusedMoEBase, CutlassFusedMoEBase):
    """W4A16_NVFP4 on SM120 / SM121: b12x for every call.

    Inherits no CUTLASS leaf: this format never reaches a CUTLASS kernel (no
    prefill fallback). ``MoEWeightOwnerMixin.post_load_weights`` runs
    ``NVFP4CuteDslB12xFusedMoEMethod.transform_weights`` to build the b12x
    wrapper.

    The FlashInfer W4A16 tile-selector shim is applied unconditionally here:
    every instance of this class runs the W4A16 decode kernel.
    """

    b12x_quant_algo = QuantAlgo.W4A16_NVFP4
    issues_fused_moe_op = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation()

    @nvtx_range("[b12x] quantize_input")
    def quantize_input(
        self,
        x: torch.Tensor | Fp4QuantizedTensor,
        post_quant_comm: bool = True,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Hand the activation over in high precision, with an AWQ pre-quant
        scale applied to it.

        The wrapper runs in ``quant_mode="w4a16"``: it dequantizes the NVFP4
        weights and leaves the bf16 / fp16 activation alone, so there is no
        activation quantization here at all. That is the kernel's own range;
        the leaf admits bf16 activations only (``supported_dtypes``), because
        the wrapper hands the kernel a bfloat16 output buffer. The pre-quant
        scale is a separate concern. An AWQ-style checkpoint stores ``W / s``, and
        ``load_quant_scales`` fills ``fc31_act_scale`` from its
        ``*.pre_quant_scale`` keys regardless of ``quant_algo``, so the weights
        the kernel dequantizes expect the scaled activation ``x * s``. Same as
        the NVFP4 sibling's decode path above and the CUTLASS W4A16 leaf.
        """
        del post_quant_comm, kwargs
        self._reject_fp4_input(x)
        if self.fc31_act_scale is not None:
            x = x * self.fc31_act_scale
        return x, None

    @nvtx_range("[b12x] run_moe")
    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: dict | None = None,
    ) -> torch.Tensor:
        del workspace  # The b12x wrapper allocates its own intermediates.
        self._admit_run_context(ctx)
        return self._run_b12x_decode(ctx)
