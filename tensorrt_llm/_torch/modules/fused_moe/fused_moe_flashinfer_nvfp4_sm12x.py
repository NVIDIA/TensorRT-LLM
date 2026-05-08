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
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...utils import ActivationType, Fp4QuantizedTensor
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


class FlashInferNvfp4Sm12xFusedMoE(CutlassFusedMoE):
    """Hybrid CUTLASS-prefill / b12x-decode NVFP4 fused-MoE backend for SM120 / SM121.

    Composition (see ``MOE_DEVELOPER_GUIDE.md`` for the full explainer):

    - **Prefill (``m >= _PREFILL_VIA_CUTLASS_THRESHOLD``)** routes through the
      inherited :class:`CutlassFusedMoE` NVFP4 GroupGEMM. The b12x kernel's
      12-CTA-per-token MMA pattern is suboptimal at large ``m``.
    - **Decode (``m <  _PREFILL_VIA_CUTLASS_THRESHOLD``)** dispatches to
      FlashInfer's ``B12xMoEWrapper.run`` — a kernel purpose-built for
      ``m=1`` / small routed-row counts.

    NVFP4 weights are loaded once via the inherited NVFP4 quant method;
    ``post_load_weights`` then prepares the b12x-shaped weight tensors
    alongside the existing CUTLASS layout. Both layouts coexist in memory
    and the dispatcher picks per call based on ``x.shape[0]``.

    CUDA graph capture only covers decode, so captured graphs always replay
    the b12x path; eager prefill always runs CUTLASS — there is no graph
    capture conflict.

    The backend hard-rejects EP (b12x has no dispatch / combine kernel),
    MoE alltoall, ``Fp4QuantizedTensor`` input, ``swiglu_gptoss_style``
    biased SwiGLU, and activations outside ``{Relu2, Swiglu}``. It is
    selected via ``moe_config.backend: FLASHINFER_NVFP4SM12X``.
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
            return _warn_and_return(
                f"FlashInferNvfp4Sm12xFusedMoE requires {sm_list}, got SM{sm_version}"
            )
        if quant_algo != QuantAlgo.NVFP4:
            return _warn_and_return(
                f"FlashInferNvfp4Sm12xFusedMoE only supports NVFP4 quantization "
                f"(got quant_algo={quant_algo})"
            )
        if dtype_activation not in {torch.float16, torch.bfloat16}:
            return _warn_and_return(
                f"FlashInferNvfp4Sm12xFusedMoE NVFP4 requires float16 or bfloat16 "
                f"activation dtype (got {dtype_activation})"
            )
        if swiglu_gptoss_style:
            return _warn_and_return(
                "FlashInferNvfp4Sm12xFusedMoE does not support swiglu_gptoss_style"
            )
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
                f"FlashInferNvfp4Sm12xFusedMoE requires ep_size == 1 "
                f"(got ep_size={self.ep_size}); use --moe_backend CUTLASS for EP."
            )
        if self.enable_alltoall:
            raise ValueError(
                "FlashInferNvfp4Sm12xFusedMoE does not support MoE alltoall communication."
            )
        if self.activation_type not in _ACTIVATION_MAP:
            supported = ", ".join(a.name for a in _ACTIVATION_MAP)
            raise ValueError(
                f"FlashInferNvfp4Sm12xFusedMoE does not support activation "
                f"{ActivationType(self.activation_type).name}; "
                f"supported: {supported}."
            )

        self._b12x_weights: Optional[dict] = None
        self.b12x_wrapper = None

    def _route_to_cutlass(self, x) -> bool:
        """Return ``True`` iff this call should fall back to the inherited
        CUTLASS path (prefill chunk). ``Fp4QuantizedTensor`` inputs always
        stay on the b12x path (which rejects them) so the existing error
        message is preserved."""
        return isinstance(x, torch.Tensor) and x.shape[0] >= self._PREFILL_VIA_CUTLASS_THRESHOLD

    def post_load_weights(self):
        """Build the b12x weight dict and instantiate ``B12xMoEWrapper``.

        Called by ``model_loader`` after ``load_weights`` finishes. The NVFP4
        quant method's ``process_weights_after_loading`` has already run as
        part of ``load_weights``, so the inherited ``w3_w1_weight`` /
        ``w2_weight`` / ``*_weight_scale`` / ``*_alpha`` / ``*_input_scale``
        tensors are populated; we just convert them to the layout b12x
        expects.
        """
        super().post_load_weights()

        try:
            from flashinfer import B12xMoEWrapper
            from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        except ImportError as e:
            raise RuntimeError(
                "FlashInferNvfp4Sm12xFusedMoE requires the `flashinfer` package "
                "(B12xMoEWrapper, cute_dsl.utils.convert_sf_to_mma_layout). "
                f"Original import error: {e}"
            ) from e

        num_local_experts = self.w3_w1_weight.shape[0]
        # Tensor shapes use the *padded* per-rank dims because TP partitions
        # may pad ``intermediate_size`` up to a kernel-friendly boundary.
        # Recover them from the actual stored tensors rather than the logical
        # model config so reshapes stay valid under TP > 1.
        _, w3w1_out_dim, _ = self.w3_w1_weight.shape  # (E, 2*I_pad, H//16)
        _, w2_out_dim, w2_in_packed = self.w2_weight.shape  # (E, H, I_pad//16)
        w3w1_in_dim = self.hidden_size
        w2_in_dim = w2_in_packed * 16

        # b12x reuses the per-expert ``w1_alpha`` tensor as both (a) the
        # online activation-quant ``global_scale`` and (b) the FC1 epilogue
        # output-dequant multiplier. That dual use is only self-consistent
        # when the FP4 weight block scales are stored in their *unnormalized*
        # form (raw ``max_block / FP4_MAX``), not divided out by the
        # per-tensor ``weight_scale_2``. HF / ModelOpt NVFP4 checkpoints
        # store the normalized variant so the FP8 block scales fit in range,
        # and TRT-LLM's NVFP4 loader preserves that form. To match b12x's
        # convention we recover ``weight_scale_2 = fc_alpha * fc_input_scale``
        # and multiply each expert's FP8 block scales by it before handing
        # them to ``convert_sf_to_mma_layout``. With the un-normalized scales
        # in place we pass ``w1_alpha = w2_alpha = 1 / fc_input_scale``
        # (== ``s_in``) so the kernel's dual-use cancels algebraically and
        # the stored input-side block scales remain FP8-representable.
        w1_w_scale_2 = (self.fc31_alpha * self.fc31_input_scale).to(torch.float32)
        w2_w_scale_2 = (self.fc2_alpha * self.fc2_input_scale).to(torch.float32)

        w1_sf_fp8_norm = self.w3_w1_weight_scale.view(torch.float8_e4m3fn).float()
        w2_sf_fp8_norm = self.w2_weight_scale.view(torch.float8_e4m3fn).float()

        # Broadcast per-expert scalar over the trailing dims (E, *).
        bcast1 = w1_w_scale_2.view(-1, *([1] * (w1_sf_fp8_norm.dim() - 1)))
        bcast2 = w2_w_scale_2.view(-1, *([1] * (w2_sf_fp8_norm.dim() - 1)))
        w1_sf_fp8 = (w1_sf_fp8_norm * bcast1).to(torch.float8_e4m3fn)
        w2_sf_fp8 = (w2_sf_fp8_norm * bcast2).to(torch.float8_e4m3fn)

        w1_sf_b12x = convert_sf_to_mma_layout(
            w1_sf_fp8, m=w3w1_out_dim, k=w3w1_in_dim, num_groups=num_local_experts
        )
        w2_sf_b12x = convert_sf_to_mma_layout(
            w2_sf_fp8, m=w2_out_dim, k=w2_in_dim, num_groups=num_local_experts
        )

        w1_alpha_b12x = (
            (1.0 / self.fc31_input_scale).expand(self.num_experts).to(torch.float32).contiguous()
        )
        w2_alpha_b12x = (
            (1.0 / self.fc2_input_scale).expand(self.num_experts).to(torch.float32).contiguous()
        )
        fc2_input_scale_b12x = (1.0 / self.fc2_input_scale).to(torch.float32)

        # TRT-LLM packs 16 FP4 values per int64. flashinfer's internal
        # ``view(torch.float4_e2m1fn_x2)`` requires byte-contiguous storage
        # (stride[-1] == 1 in bytes); a uint8 view of the int64 tensor
        # provides that without copying.
        self._b12x_weights = dict(
            w1_weight=self.w3_w1_weight.view(torch.uint8),
            w1_weight_sf=w1_sf_b12x,
            w1_alpha=w1_alpha_b12x,
            w2_weight=self.w2_weight.view(torch.uint8),
            w2_weight_sf=w2_sf_b12x,
            w2_alpha=w2_alpha_b12x,
            fc2_input_scale=fc2_input_scale_b12x,
        )

        self.b12x_wrapper = B12xMoEWrapper(
            num_experts=self.num_experts,
            top_k=self.routing_method.experts_per_token,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size_per_partition,
            use_cuda_graph=self._b12x_use_cuda_graph,
            max_num_tokens=self.moe_max_num_tokens,
            activation=_ACTIVATION_MAP[self.activation_type],
        )

        # Replace the wrapper's per-instance output buffer with a shared one.
        # Layers run sequentially on a single stream, so a single buffer of the
        # right shape is correct and saves
        # ``(num_moe_layers - 1) * max_num_tokens * hidden_size * 2`` bytes —
        # ~2.5 GB on Nemotron-Super-120B with ``max_num_tokens=2048``,
        # ``hidden=8192``, bf16, 80 MoE layers.
        if self.b12x_wrapper._moe_output is not None:
            buf = self.b12x_wrapper._moe_output
            key = (buf.shape[0], buf.shape[1], buf.dtype, str(buf.device))
            shared = _SHARED_MOE_OUTPUT_BUF.get(key)
            if shared is None:
                _SHARED_MOE_OUTPUT_BUF[key] = buf
            else:
                # Free the freshly allocated buffer; reuse the existing one.
                self.b12x_wrapper._moe_output = shared

        logger.info_once(
            f"FlashInferNvfp4Sm12xFusedMoE active: hidden={self.hidden_size}, "
            f"intermediate={self.intermediate_size_per_partition}, "
            f"experts={self.num_experts}, top_k="
            f"{self.routing_method.experts_per_token}, "
            f"activation={_ACTIVATION_MAP[self.activation_type]}.",
            key="flashinfer_nvfp4_sm12x_moe_active",
        )

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
                "FlashInferNvfp4Sm12xFusedMoE does not accept Fp4QuantizedTensor input "
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
            return CutlassFusedMoE.run_moe(
                self,
                x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                x_sf=x_sf,
                is_sf_swizzled=is_sf_swizzled,
                output_dtype=output_dtype,
                tuner_num_tokens=tuner_num_tokens,
                tuner_top_k=tuner_top_k,
                moe_output=moe_output,
                enable_alltoall=enable_alltoall,
            )
        if self.b12x_wrapper is None or self._b12x_weights is None:
            raise RuntimeError(
                "FlashInferNvfp4Sm12xFusedMoE.run_moe called before "
                "process_weights_after_loading completed."
            )
        if x_sf is not None:
            raise ValueError(
                "FlashInferNvfp4Sm12xFusedMoE expects unquantized input (x_sf=None) "
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
        # alltoall workspace tensor), copy into it; FlashInferNvfp4Sm12xFusedMoE
        # currently rejects alltoall in __init__, so this is a defensive
        # path for future workspace-driven uses.
        if moe_output is not None:
            with nvtx_range("[b12x] out_copy"):
                moe_output.copy_(out)
            return moe_output
        return out
