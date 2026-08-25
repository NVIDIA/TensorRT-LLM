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
from typing import Optional, Tuple, Union

import torch

from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...utils import ActivationType, Fp4QuantizedTensor
from .fused_moe_cutlass import CutlassFusedMoE
from .impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    MoEStaticCapability,
    require_comm_plan,
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
# previous output before the next layer is dispatched.
_SHARED_MOE_OUTPUT_BUF: dict = {}

# ActivationType -> b12x activation string. b12x currently exposes "relu2"
# (Nemotron-style x * relu(x)) and "silu" (SwiGLU-style x * silu(gate)).
_ACTIVATION_MAP = {
    ActivationType.Relu2: "relu2",
    ActivationType.Swiglu: "silu",
}


class CuteDslB12xFusedMoE(CutlassFusedMoE):
    """B12x NVFP4 fused-MoE backend for SM120 / SM121.

    Large prefill chunks use CUTLASS; decode uses FlashInfer's b12x kernel.

    Inherits ``CutlassFusedMoE`` rather than only the shared blocks -- the same
    shortcut its siblings (CuteDsl, DeepGemm, Marlin) take, but here it is a
    real dependency: ``_route_to_cutlass`` sends every
    NVFP4 prefill chunk through ``CutlassFusedMoE.quantize_input`` /
    ``CutlassFusedMoE.run_moe``, which read the whole Cutlass execution state
    (chunking stream and events, ``use_fused_finalize``, the tuner flags, the
    LoRA slot helpers, ``_tuner_shapes``, ``_run_moe_w4a16_nvfp4``).

    Two ``CuteDslFusedMoE.__init__`` side effects are deliberately dropped, not
    restated: the ``AuxStreamType.MoeOutputMemset`` / ``EventType`` entries, and
    the ``swiglu_limit_scalar or inf`` fallback. Both are read only by
    ``CuteDslFusedMoE.run_moe_nvfp4*``, which the ``run_moe`` override below
    never reaches. Restate them before routing any CuteDSL path through this
    class -- ``event_dict`` can now be None and ``swiglu_limit_scalar`` unset.
    """

    # Restated rather than inherited: the LoRA gate this replaces compared the
    # exact class and answered False here, while the DWDP gate used isinstance
    # and answered True through CuteDslFusedMoE.
    capabilities = MoEStaticCapability(supports_moe_lora=False, supports_dwdp=True)

    # This and ``supports_moe_output_in_alltoall_workspace`` came through
    # ``CuteDslFusedMoE`` before the reparent and Cutlass answers differently on
    # both, so both are restated to keep the declared values unchanged. Read by
    # ``ConfigurableMoE._reject_non_divisible_ep_backend()``; moot for this class
    # in practice because ``can_implement`` rejects ``ep_size != 1`` outright.
    _supports_non_divisible_ep: bool = True

    def supports_moe_output_in_alltoall_workspace(self) -> bool:
        return self.has_nvfp4

    # SM versions on which the FlashInfer b12x NVFP4 MoE kernel is available.
    # SM120 = desktop Blackwell (RTX 5090 / GB202); SM121 = GB10 / DGX Spark.
    _SUPPORTED_SM_VERSIONS = frozenset({120, 121})

    # Prefill chunks (``x.shape[0] >= threshold``) route via CUTLASS NVFP4
    # GroupGEMM; decode (``x.shape[0] < threshold``) uses b12x. 64 cleanly
    # separates conc=1 prefill (m=2048 with ``max_num_tokens=2048``) from
    # decode (m=1) and stays robust against future chunked-prefill splits
    # that might shrink prefill chunk size.
    _PREFILL_VIA_CUTLASS_THRESHOLD = 64

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        sm_version = d.env.sm
        if sm_version not in cls._SUPPORTED_SM_VERSIONS:
            sm_list = "/".join(f"SM{v}" for v in sorted(cls._SUPPORTED_SM_VERSIONS))
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                f"CuteDslB12xFusedMoE requires {sm_list}, got SM{sm_version}",
            )
        if p.quant_algo not in {QuantAlgo.NVFP4, QuantAlgo.W4A16_NVFP4}:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"CuteDslB12xFusedMoE only supports NVFP4 or W4A16_NVFP4 quantization "
                f"(got quant_algo={p.quant_algo})",
            )
        if p.dtype_act not in {torch.float16, torch.bfloat16}:
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                f"CuteDslB12xFusedMoE NVFP4 requires float16 or bfloat16 "
                f"activation dtype (got {p.dtype_act})",
            )
        if p.swiglu_gptoss_style:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                "CuteDslB12xFusedMoE does not support swiglu_gptoss_style",
            )
        if p.activation_type not in _ACTIVATION_MAP:
            supported = ", ".join(a.name for a in _ACTIVATION_MAP)
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"CuteDslB12xFusedMoE does not support activation "
                f"{p.activation}; supported: {supported}",
            )
        # The decode kernel ships in the FlashInfer wheel.
        if not d.env.has_dep(MoEDep.FLASHINFER):
            return _reject(
                MoERejectReason.DEP_MISSING,
                "CuteDslB12xFusedMoE requires the flashinfer package",
            )
        # No expert-parallel dispatch/combine kernel: EP must stay at 1.
        if d.ep_size != 1:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"CuteDslB12xFusedMoE requires ep_size == 1 (got {d.ep_size})",
            )
        # Attention-DP is a separate axis from EP: with moe_tp == tp the layer
        # can have ep_size == 1 and still sit behind a DP allgather /
        # reducescatter that the b12x wrapper has never been exercised under.
        # ``use_dp and parallel_size > 1`` is exactly ``mapping.dp_size > 1``
        # (``Mapping.dp_size`` is ``tp_size`` when attention-DP is on).
        if d.use_dp and d.parallel_size > 1:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"CuteDslB12xFusedMoE does not support attention-DP "
                f"(parallel_size={d.parallel_size})",
            )
        return MoEEligibility.ok()

    def __init__(self, *args, **kwargs):
        # ``ModelConfig`` is consumed by the inherited ``__init__`` for cache
        # / mapping setup but isn't kept on ``self``. b12x's wrapper needs the
        # ``use_cuda_graph`` flag at construction time, so capture it here
        # before delegating.
        model_config = kwargs.get("model_config", None)
        self._b12x_use_cuda_graph = bool(getattr(model_config, "use_cuda_graph", False))

        super().__init__(*args, **kwargs)

        # No alltoall guard here: alltoall is picked by the wrapper's
        # communication strategy, and ``can_implement`` already rejects the
        # topologies that could pick it (ep_size != 1, attention-DP with
        # parallel_size > 1).
        self._b12x_weights: Optional[dict] = None
        self.b12x_wrapper = None

    def _get_quant_method(self):
        # Route NVFP4 to the b12x-aware quant method so weight prep
        # (SF un-normalization, ``convert_sf_to_mma_layout``,
        # ``B12xMoEWrapper`` instantiation) lives next to the rest of the
        # NVFP4 quant-method family, while every other quant algo (and the
        # unquantized fallback) continues to resolve via the parent.
        if (
            self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_any_quant(exclude_kv_cache=True)
            and self.quant_config.layer_quant_mode.has_nvfp4()
        ):
            from .quantization import NVFP4CuteDslB12xFusedMoEMethod

            return NVFP4CuteDslB12xFusedMoEMethod()
        return super()._get_quant_method()

    def _route_to_cutlass(self, x) -> bool:
        """Return ``True`` iff this call should fall back to the inherited
        CUTLASS path (NVFP4 prefill chunk). ``Fp4QuantizedTensor`` inputs
        always stay on the b12x path (which rejects them) so the existing
        error message is preserved."""
        quant_config = getattr(self, "quant_config", None)
        if quant_config is not None and quant_config.quant_algo == QuantAlgo.W4A16_NVFP4:
            return False
        return isinstance(x, torch.Tensor) and x.shape[0] >= self._PREFILL_VIA_CUTLASS_THRESHOLD

    # ``post_load_weights`` is inherited from ``CutlassFusedMoE`` and
    # dispatches to ``self.quant_method.transform_weights(self)`` — for this
    # backend ``self.quant_method`` is ``NVFP4CuteDslB12xFusedMoEMethod``
    # (see ``_get_quant_method`` override), which performs the SF un-normalization,
    # ``convert_sf_to_mma_layout`` reshape, ``B12xMoEWrapper`` instantiation,
    # and the cross-layer shared output buffer dance. The wrapper and the
    # bundled weight dict are attached to this module as ``self.b12x_wrapper``
    # / ``self._b12x_weights``, which the decode path below consumes.

    @nvtx_range("[b12x] quantize_input")
    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Hybrid dispatch entrypoint for activation handling.

        NVFP4 prefill chunks take the inherited
        :meth:`CutlassFusedMoE.quantize_input` path so the downstream
        ``run_moe`` can call CUTLASS NVFP4 GroupGEMM. Decode chunks and
        W4A16_NVFP4 chunks pass through unchanged because b12x quantizes
        activations internally (consumes a bf16 / fp16 ``x`` and produces its
        own scale factors).
        """
        if self._route_to_cutlass(x):
            return CutlassFusedMoE.quantize_input(
                self, x, post_quant_comm=post_quant_comm, **kwargs
            )
        if isinstance(x, Fp4QuantizedTensor):
            raise ValueError(
                "CuteDslB12xFusedMoE does not accept Fp4QuantizedTensor input "
                "on the b12x decode path; b12x performs its own input quantization."
            )
        return x, None

    @nvtx_range("[b12x] run_moe")
    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor:
        plan = require_comm_plan(self, ctx)
        x = ctx.x
        if self._route_to_cutlass(x):
            # ``CutlassFusedMoE.run_moe`` forwards ``output_dtype`` straight
            # into the C++ ``trtllm::fused_moe`` op, which requires a concrete
            # high-precision ``ScalarType`` (uint8 / FP4-packed activations are
            # rejected at the kernel epilogue with "Invalid output type Byte").
            # ``ConfigurableMoE.forward`` always fills ``output_dtype``, so this
            # only narrows the type for anything driving ``run_moe`` without it.
            _HIGH_PRECISION = {torch.float16, torch.bfloat16, torch.float32}
            cutlass_output_dtype = ctx.output_dtype
            if cutlass_output_dtype is None:
                cutlass_output_dtype = (
                    x.dtype
                    if isinstance(x, torch.Tensor) and x.dtype in _HIGH_PRECISION
                    else torch.bfloat16
                )
            return CutlassFusedMoE.run_moe(
                self,
                replace(ctx, output_dtype=cutlass_output_dtype),
                workspace=workspace,
            )
        token_selected_experts = ctx.token_selected_experts
        token_final_scales = ctx.token_final_scales
        x_sf = ctx.x_sf
        moe_output = plan.moe_output
        if self.b12x_wrapper is None or self._b12x_weights is None:
            raise RuntimeError(
                "CuteDslB12xFusedMoE.run_moe called before process_weights_after_loading completed."
            )
        if x_sf is not None:
            raise ValueError(
                "CuteDslB12xFusedMoE expects unquantized input (x_sf=None) "
                "on the b12x decode path; got a precomputed scale factor."
            )

        # Annotate the kwargs spread + wrapper entry separately so we can
        # attribute the per-layer Python dispatch cost vs. the kernel cost.
        with nvtx_range("[b12x] wrapper.run"):
            out = self.b12x_wrapper.run(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                **self._b12x_weights,
            )

        # B12xMoEWrapper allocates its own output buffer for CUDA-graph
        # compatibility. If the caller provided ``moe_output`` (e.g. an
        # alltoall workspace tensor), copy into it; ``can_implement`` rejects
        # the topologies that could pick alltoall, so this is a defensive
        # path for future workspace-driven uses.
        if moe_output is not None:
            with nvtx_range("[b12x] out_copy"):
                moe_output.copy_(out)
            return moe_output
        return out
