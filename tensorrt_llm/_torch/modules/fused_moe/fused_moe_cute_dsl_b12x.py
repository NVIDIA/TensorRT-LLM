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

from typing import Optional, Tuple, Union

import torch

from tensorrt_llm._utils import get_sm_version, nvtx_range
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...utils import ActivationType, Fp4QuantizedTensor
from .fused_moe_cute_dsl import CuteDslFusedMoE
from .fused_moe_cutlass import CutlassFusedMoE
from .interface import _warn_and_return

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


class CuteDslB12xFusedMoE(CuteDslFusedMoE):
    """Hybrid CUTLASS-prefill / b12x-decode NVFP4 fused-MoE backend for SM120 / SM121.

    Member of the cuteDSL backend family: the decode kernel
    (``flashinfer.B12xMoEWrapper.run``) is JIT-compiled CuTe DSL, so the
    backend slots in next to :class:`CuteDslFusedMoE` (which targets SM100 /
    SM103). The hybrid prefill path still routes through the C++ CUTLASS
    NVFP4 GroupGEMM via explicit :class:`CutlassFusedMoE` method calls; the
    parent class on the MRO does not change which kernels execute, only
    where the b12x backend sits in the family.

    Composition (see ``MOE_DEVELOPER_GUIDE.md`` for the full explainer):

    - **Prefill (``m >= _PREFILL_VIA_CUTLASS_THRESHOLD``)** explicitly
      invokes :class:`CutlassFusedMoE` NVFP4 GroupGEMM. The b12x kernel's
      12-CTA-per-token MMA pattern is suboptimal at large ``m``.
    - **Decode (``m <  _PREFILL_VIA_CUTLASS_THRESHOLD``)** dispatches to
      FlashInfer's ``B12xMoEWrapper.run`` — a kernel purpose-built for
      ``m=1`` / small routed-row counts.

    NVFP4 weights are loaded via :class:`NVFP4CuteDslB12xFusedMoEMethod`
    (an :class:`NVFP4CutlassFusedMoEMethod` subclass returned by
    ``_get_quant_method``). The inherited CUTLASS NVFP4 layout is finalised
    by the base class, and the b12x-shaped tensors (un-normalised FP8 SF,
    ``convert_sf_to_mma_layout`` reshape, ``B12xMoEWrapper`` instance) are
    materialised on top by the quant method's ``transform_weights``. Both
    layouts coexist in memory and the dispatcher picks per call based on
    ``x.shape[0]``.

    CUDA graph capture only covers decode, so captured graphs always replay
    the b12x path; eager prefill always runs CUTLASS — there is no graph
    capture conflict.

    The backend hard-rejects EP (b12x has no dispatch / combine kernel),
    MoE alltoall, ``Fp4QuantizedTensor`` input, ``swiglu_gptoss_style``
    biased SwiGLU, and activations outside ``{Relu2, Swiglu}``. It is
    selected on the ``CUTEDSL`` MoE path when SM120 / SM121 + NVFP4 +
    flashinfer-importable gates pass (see ``create_moe.get_moe_cls``).
    """

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
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        sm_version = get_sm_version()
        if sm_version not in cls._SUPPORTED_SM_VERSIONS:
            sm_list = "/".join(f"SM{v}" for v in sorted(cls._SUPPORTED_SM_VERSIONS))
            return _warn_and_return(f"CuteDslB12xFusedMoE requires {sm_list}, got SM{sm_version}")
        if quant_algo != QuantAlgo.NVFP4:
            return _warn_and_return(
                f"CuteDslB12xFusedMoE only supports NVFP4 quantization "
                f"(got quant_algo={quant_algo})"
            )
        if dtype_activation not in {torch.float16, torch.bfloat16}:
            return _warn_and_return(
                f"CuteDslB12xFusedMoE NVFP4 requires float16 or bfloat16 "
                f"activation dtype (got {dtype_activation})"
            )
        if swiglu_gptoss_style:
            return _warn_and_return("CuteDslB12xFusedMoE does not support swiglu_gptoss_style")
        return True, None

    def __init__(self, *args, **kwargs):
        # ``ModelConfig`` is consumed by the inherited ``__init__`` for cache
        # / mapping setup but isn't kept on ``self``. b12x's wrapper needs the
        # ``use_cuda_graph`` flag at construction time, so capture it here
        # before delegating.
        model_config = kwargs.get("model_config", None)
        self._b12x_use_cuda_graph = bool(getattr(model_config, "use_cuda_graph", False))

        super().__init__(*args, **kwargs)

        # b12x has no expert-parallel dispatch/combine kernel, so EP must be
        # disabled. dp_size > 1 implies the alltoall path which b12x can't run.
        if self.ep_size != 1:
            raise ValueError(
                f"CuteDslB12xFusedMoE requires ep_size == 1 "
                f"(got ep_size={self.ep_size}); use --moe_backend CUTLASS for EP."
            )
        if self.enable_alltoall:
            raise ValueError("CuteDslB12xFusedMoE does not support MoE alltoall communication.")
        if self.activation_type not in _ACTIVATION_MAP:
            supported = ", ".join(a.name for a in _ACTIVATION_MAP)
            raise ValueError(
                f"CuteDslB12xFusedMoE does not support activation "
                f"{ActivationType(self.activation_type).name}; "
                f"supported: {supported}."
            )

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
        CUTLASS path (prefill chunk). ``Fp4QuantizedTensor`` inputs always
        stay on the b12x path (which rejects them) so the existing error
        message is preserved."""
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

        Prefill chunks (``x.shape[0] >= _PREFILL_VIA_CUTLASS_THRESHOLD``) take
        the inherited :meth:`CutlassFusedMoE.quantize_input` path so the
        downstream ``run_moe`` can call CUTLASS NVFP4 GroupGEMM. Decode
        chunks pass through unchanged because b12x quantizes activations
        internally (consumes a bf16 / fp16 ``x`` and produces its own scale
        factors).
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
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: Optional[torch.Tensor] = None,
        is_sf_swizzled: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        tuner_num_tokens: Optional[int] = None,
        tuner_top_k: Optional[int] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: Optional[bool] = None,
    ) -> torch.Tensor:
        if self._route_to_cutlass(x):
            # ``CutlassFusedMoE.run_moe`` forwards ``output_dtype`` straight
            # into the C++ ``trtllm::fused_moe`` op, which requires a concrete
            # high-precision ``ScalarType`` (uint8 / FP4-packed activations are
            # rejected at the kernel epilogue with "Invalid output type Byte").
            # Schedulers that drive ``run_moe`` directly (the KV-cache capacity
            # probe, for one) leave ``output_dtype`` unset, so fall back to
            # ``x.dtype`` if it is a real compute dtype, else bf16. Mirrors the
            # ``forward_chunk`` convention while staying safe for the FP4
            # quant-input path (``x`` is uint8 after ``quantize_input``).
            _HIGH_PRECISION = {torch.float16, torch.bfloat16, torch.float32}
            cutlass_output_dtype = output_dtype
            if cutlass_output_dtype is None:
                cutlass_output_dtype = (
                    x.dtype
                    if isinstance(x, torch.Tensor) and x.dtype in _HIGH_PRECISION
                    else torch.bfloat16
                )
            return CutlassFusedMoE.run_moe(
                self,
                x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                x_sf=x_sf,
                is_sf_swizzled=is_sf_swizzled,
                output_dtype=cutlass_output_dtype,
                tuner_num_tokens=tuner_num_tokens,
                tuner_top_k=tuner_top_k,
                moe_output=moe_output,
                enable_alltoall=enable_alltoall,
            )
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
        # alltoall workspace tensor), copy into it; CuteDslB12xFusedMoE
        # currently rejects alltoall in __init__, so this is a defensive
        # path for future workspace-driven uses.
        if moe_output is not None:
            with nvtx_range("[b12x] out_copy"):
                moe_output.copy_(out)
            return moe_output
        return out
