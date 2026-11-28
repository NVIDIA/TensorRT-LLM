# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 block-scale weight-read linear for the Kimi K3 projections.

`Fp8BlockScaleWeightReadLinear` replaces a bias-free linear after its weights load. It keeps the
weight in FP8 with one scale per 128x128 block, so a decode step reads half the bytes of a BF16
weight, and it picks the FP8 GEMM for the GPU and the number of rows. The Kimi K3 loader
(``models/modeling_kimi_k3_loader.py``) puts it in place of the checkpoint's FP8 attention
projections and, optionally, of the replicated BF16 MoE-layer MLP projections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm.functional import AllReduceParams

from ..._utils import get_sm_version
from .linear import Fp8BlockScaleLowMGemmRange, fp8_block_scale_ue8m0_gemm
from .linear import Linear as TrtllmLinear

if TYPE_CHECKING:
    from ...quantization.utils.fp8_utils import Fp8BlockScaleWeightSliceCache

# The prequantized low-M kernel's folded scale load does not advance to a second R128c4
# row block, so ``takes_prequantized_input`` serves at most one such block of rows.
_PREQUANTIZED_INPUT_MAX_ROWS = 128


def _rubin_mxfp8_is_available() -> bool:
    """Whether this rank can run the SM107 native block-scaled MXFP8 GEMM.

    Recomputed rather than cached: the SM version cannot change within a run,
    so weight preparation and forward always agree.
    """
    from ..cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def _low_m_gemm_nvfp4_max_m(out_features: int, in_features: int) -> int:
    """Largest token count for which the low-M GEMM emits the NVFP4 quantization of its output
    on the current device (0: never)."""
    from ..kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.nvfp4_out_max_m_for(out_features, in_features)


def _low_m_gemm_gated_applies(x: torch.Tensor, gate: torch.Tensor, out_features: int) -> bool:
    from ..kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.gate_fits(x, gate, out_features)


def _low_m_gemm_scale_from_mxfp8(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The DeepGEMM-packed scale the low-M GEMM reads, from the SM107 CuTe MXFP8 scale."""
    from ..kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.pack_block_scale_from_mxfp8_layout(weight_scale, n, k)


class Fp8BlockScaleWeightReadLinear(nn.Module):
    """Bias-free FP8 block-scale GEMM for checkpoint attention and converted MLPs.

    Attention loads checkpoint E4M3 codes and 128x128 scales directly. The optional
    MLP path quantizes BF16 weights once at load. On SM100/SM103 the UE8M0
    scales are packed for DeepGEMM, which quantizes BF16 activations inside the
    GEMM. On Rubin (SM107) the scales are packed into the CuTe DSL K32 R128c4
    layout and the activation is quantized by the specialized CuTe quantizer
    before the native MXFP8 GEMM, which runs with K3's fine-grained low-M tuning
    buckets. On both, the low-M GEMM is a candidate of forward like in the
    generic FP8 block-scale linear (``Fp8BlockScaleLowMGemmRange``); it reads a
    DeepGEMM-packed scale, which is ``weight_scale`` on SM100/SM103 and a copy
    repacked from the MXFP8 scale on SM107.
    """

    def __init__(
        self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor, out_features: int
    ) -> None:
        super().__init__()
        self.in_features = weight_fp8.shape[1]
        self.out_features = out_features
        self._weights_transformed = True
        self._fused_projection = None
        self._output_slice_cache: Optional[Fp8BlockScaleWeightSliceCache] = None
        # Buffers (not parameters): these are the module's weights post-load;
        # there is nothing further to load into them and they must not be
        # touched by any later autocast/dtype move.
        self.register_buffer("weight", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        # Rows the low-M GEMM serves and the DeepGEMM-packed scale it reads; set at post-load.
        self._low_m = Fp8BlockScaleLowMGemmRange()
        self._low_m_weight_scale: Optional[torch.Tensor] = None
        # Largest row count :meth:`forward_nvfp4` serves (0: never); set at post-load.
        self._nvfp4_out_max_m = 0

    @property
    def has_fp8_qdq(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def has_w4a8_nvfp4_fp8(self) -> bool:
        """Match the ``Linear`` interface consumed by ``GatedMLP``."""
        return False

    @property
    def supports_prequantized_input(self) -> bool:
        """Whether :meth:`forward_prequantized` serves every row count (SM107: the CuTe MXFP8
        GEMM)."""
        return _rubin_mxfp8_is_available()

    def takes_prequantized_input(self, num_rows: int) -> bool:
        """Whether a producer should hand :meth:`forward_prequantized` the MXFP8 form of
        ``num_rows`` rows instead of the BF16 rows: where :meth:`forward` runs the low-M kernel,
        so an operand quantized with its arithmetic
        (``fp8_block_scale_low_m_gemm.mxfp8_quantize_reference``) projects bitwise like
        :meth:`forward` and the kernel skips the quantization.

        At most one 128-row R128c4 scale block: the kernel's folded scale load does not
        advance to the next row block, so a wider forward takes the BF16 rows."""
        return 1 <= num_rows <= min(self._low_m.kernel_max_m, _PREQUANTIZED_INPUT_MAX_ROWS)

    @staticmethod
    def quantize_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """BF16 ``[out, in]`` weight -> (FP8 weight, backend-ready scale).

        Both dims must be multiples of 128. Because the 128x128 block scale is
        computed per block, concatenating several such weights along ``out``
        and quantizing the result is per-block identical to quantizing each
        separately (no block crosses a 128-aligned boundary), so a fused
        weight's row slices equal the individually quantized weights.
        """
        # Lazy imports: only pulled in on the FP8 path.
        from ...deep_gemm.utils.math import per_block_cast_to_fp8

        weight_fp8, weight_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        return Fp8BlockScaleWeightReadLinear._prepare_weights(weight_fp8, weight_scale)

    @staticmethod
    def _prepare_weights(
        weight_fp8: torch.Tensor, weight_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from ...quantization.utils.fp8_utils import (
            resmooth_to_fp8_e8m0,
            transform_k128_scales_to_cutedsl_mxfp8_layout,
            transform_sf_into_required_layout,
        )

        # Both GEMMs consume UE8M0 scales rather than the checkpoint's FP32 grid.
        weight_fp8, weight_scale = resmooth_to_fp8_e8m0(
            weight_fp8.contiguous(), weight_scale.contiguous().float()
        )
        if _rubin_mxfp8_is_available():
            # cute_dsl_mxfp8_gemm_rubin reads K32 R128c4 UE8M0 scales.
            weight_scale = transform_k128_scales_to_cutedsl_mxfp8_layout(
                weight_scale, mn=weight_fp8.shape[0], k=weight_fp8.shape[1]
            )
            return weight_fp8, weight_scale
        # fp8_swap_ab_gemm with disable_ue8m0_cast=True consumes packed,
        # TMA-aligned UE8M0 scales.
        weight_scale = transform_sf_into_required_layout(
            weight_scale,
            mn=weight_fp8.shape[0],
            k=weight_fp8.shape[1],
            recipe=(1, 128, 128),
            is_sfa=False,
        )
        return weight_fp8, weight_scale

    @classmethod
    def from_linear(cls, linear: nn.Linear | TrtllmLinear) -> "Fp8BlockScaleWeightReadLinear":
        assert linear.bias is None, "FP8 weight read expects a bias-free Linear"
        if isinstance(linear, TrtllmLinear) and linear.has_fp8_block_scales:
            result = cls(linear.weight.detach(), linear.weight_scale.detach(), linear.out_features)
            result.quant_config = linear.quant_config
            result.tp_size, result.tp_rank = linear.tp_size, linear.tp_rank
            result._weights_transformed = False
            return result
        weight_fp8, weight_scale = cls.quantize_weight(linear.weight.data)
        return cls(weight_fp8, weight_scale, linear.out_features)

    def transform_weights(self) -> None:
        if not self._weights_transformed:
            self._output_slice_cache = None
            self.weight, self.weight_scale = self._prepare_weights(self.weight, self.weight_scale)
            if self._fused_projection is not None:
                owner_ref, start = self._fused_projection
                owner = owner_ref()
                if owner is None:
                    raise RuntimeError("KDA fused FP8 projection was released before its views")
                owner.transform_weights()
                self.weight = owner.weight[start : start + self.out_features]
            self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self._prepare_low_m_gemm()

    def _prepare_low_m_gemm(self) -> None:
        """Select the low-M GEMM for this weight where it has a DeepGEMM-packed scale.

        On SM107 that scale is a copy (N * K / 128 bytes) derived from the MXFP8
        ``weight_scale``; row views of a fused KDA projection have none (the fused module
        serves decode).
        """
        n, k = self.out_features, self.in_features
        self._low_m = Fp8BlockScaleLowMGemmRange.for_weight(n, k)
        if not _rubin_mxfp8_is_available():
            self._low_m_weight_scale = self.weight_scale
        elif self._low_m.max_m and self._fused_projection is None:
            self._low_m_weight_scale = _low_m_gemm_scale_from_mxfp8(self.weight_scale, n, k)
        else:
            self._low_m, self._low_m_weight_scale = Fp8BlockScaleLowMGemmRange(), None
        self._nvfp4_out_max_m = _low_m_gemm_nvfp4_max_m(n, k) if self._low_m.max_m else 0

    def forward_slice(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        self.prepare_output_slice(start, end)
        assert self._output_slice_cache is not None
        weight, scale, crop = self._output_slice_cache.get(self.weight, start, end)
        out = self._apply_with_weights(x, weight, scale)
        return out[..., crop : crop + end - start].contiguous()

    def prepare_output_slice(self, start: int, end: int) -> None:
        """Prepare fixed output-slice scales after checkpoint transformation."""
        from ...quantization.utils.fp8_utils import Fp8BlockScaleWeightSliceCache

        self.transform_weights()
        if self._output_slice_cache is None:
            self._output_slice_cache = Fp8BlockScaleWeightSliceCache()
        self._output_slice_cache.prepare(self.weight, self.weight_scale, start, end)

    def forward(
        self,
        x: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        return self._apply_with_weights(x, self.weight, self.weight_scale)

    def _apply_with_weights(
        self, x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        out_shape = (*x.shape[:-1], weight.shape[0])
        x_flat = x.reshape(-1, x.shape[-1])
        if not _rubin_mxfp8_is_available():
            low_m = self._low_m if weight is self.weight else Fp8BlockScaleLowMGemmRange()
            out = fp8_block_scale_ue8m0_gemm(
                x_flat, weight, weight_scale, low_m, output_dtype=x.dtype
            )
        elif weight is self.weight and self._low_m.serves(x_flat):
            out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(
                x_flat, self.weight, self._low_m_weight_scale
            )
        else:
            act_fp8, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x_flat)
            out = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                act_fp8,
                weight,
                act_scale,
                weight_scale,
                output_dtype=torch.bfloat16,
                fine_grained_m=True,
            )
        return out.reshape(out_shape)

    def emits_nvfp4(self, num_rows: int) -> bool:
        """Whether :meth:`forward_nvfp4` serves ``num_rows`` rows."""
        return 1 <= num_rows <= self._nvfp4_out_max_m

    def forward_nvfp4(
        self, x: torch.Tensor, global_scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``fp4_quantize`` (``global_scale``, 16-element groups, linear scales) of the
        projection, from one low-M GEMM whose epilogue quantizes (bitwise the GEMM at the same
        tactic followed by ``fp4_quantize``). Returns E2M1 codes ``uint8 [M, out / 2]`` and E4M3
        scales ``uint8 [M * out / 16]``. For ``num_rows`` where :meth:`emits_nvfp4`."""
        return torch.ops.trtllm.fp8_block_scale_low_m_gemm_nvfp4(
            x.reshape(-1, x.shape[-1]), self.weight, self._low_m_weight_scale, global_scale
        )

    def forward_gated(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``forward(x * gate)``, bit for bit, for an output-gated projection.

        Where :meth:`forward` would run the low-M kernel, one gated launch multiplies inside
        the GEMM (``gate`` may then be a row-strided BF16 view, see ``check_gate``).
        """
        x_flat = x.reshape(-1, x.shape[-1])
        gate_flat = gate.reshape(x_flat.shape)
        if self._low_m.serves(x_flat, kernel=True) and _low_m_gemm_gated_applies(
            x_flat, gate_flat, self.out_features
        ):
            out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(
                x_flat, self.weight, self._low_m_weight_scale, gate=gate_flat
            )
            return out.reshape(*x.shape[:-1], self.out_features)
        return self.forward(x * gate)

    def forward_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """``forward(x).sigmoid()``, bit for bit; one launch where :meth:`forward` would run
        the low-M kernel."""
        x_flat = x.reshape(-1, x.shape[-1])
        if self._low_m.serves(x_flat, kernel=True):
            out = torch.ops.trtllm.fp8_block_scale_low_m_gemm(
                x_flat, self.weight, self._low_m_weight_scale, sigmoid=True
            )
            return out.reshape(*x.shape[:-1], self.out_features)
        return self.forward(x).sigmoid()

    def forward_prequantized(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Project an FP8 activation a fused producer already quantized.

        Runs the low-M kernel where :meth:`takes_prequantized_input`, the CuTe MXFP8 GEMM at any
        other row count where :attr:`supports_prequantized_input`.

        Args:
            activation: ``[..., in_features]`` ``float8_e4m3fn`` activation.
            activation_scale: ``uint8`` UE8M0 1x32 scales in the CuTe K32 R128c4
                layout (``fp8_quantize_1x128_cutedsl_ue8m0`` replicates 1x128 scales
                into it).
        """
        if activation.dtype is not torch.float8_e4m3fn:
            raise TypeError("prequantized Kimi K3 activation must be float8_e4m3fn")
        if activation_scale.dtype is not torch.uint8:
            raise TypeError("prequantized Kimi K3 activation scale must be uint8 R128c4")
        activation_flat = activation.reshape(-1, activation.shape[-1])
        num_rows = activation_flat.shape[0]
        if self.takes_prequantized_input(num_rows):
            out = torch.ops.trtllm.fp8_block_scale_low_m_gemm_prequantized(
                activation_flat, activation_scale, self.weight, self._low_m_weight_scale
            )
        elif not self.supports_prequantized_input:
            raise RuntimeError(f"no GEMM for a prequantized Kimi K3 activation of {num_rows} rows")
        else:
            out = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
                activation_flat,
                self.weight,
                activation_scale,
                self.weight_scale,
                output_dtype=torch.bfloat16,
                fine_grained_m=True,
            )
        return out.reshape(activation.shape[:-1] + (self.out_features,))
