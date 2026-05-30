# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from torch import nn

from ..._utils import get_sm_version, nvtx_range
from ..utils import Fp4QuantizedTensor, maybe_compile


class LayerNorm(nn.Module):
    """Layer normalization module with configurable weight and bias parameters.

    This implementation provides standard layer normalization with optional
    learnable parameters and residual connection support.

    For DiT models (e.g., Wan 2.2), this module also supports a fused
    "LayerNorm + (optional AdaLN modulation) + NVFP4 quantization" fast path
    when ``quantize_type="nvfp4"`` is set at construction time and an
    ``nvfp4_scale`` is attached (typically by post-load weight propagation
    from the downstream NVFP4 Linear's ``input_scale``).

    Args:
        hidden_size: The size of the hidden dimension to normalize.
        eps: Small constant for numerical stability.
        dtype: Optional data type for parameters.
        device: Optional device for parameters.
        has_weights: Whether to include learnable weight parameters.
        has_bias: Whether to include learnable bias parameters.
        quantize_type: Optional. Pass ``"nvfp4"`` to enable the fused
            LayerNorm + NVFP4 quantization fast path. The fused path
            additionally requires:
              - A calibrated ``nvfp4_scale`` attached to the module
                (a scalar FP32 tensor, typically the downstream
                Linear's ``input_scale``).
              - Blackwell (SM100) or newer; on unsupported architectures
                this falls back to the unfused path silently.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        eps: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        has_weights: bool = True,
        has_bias: bool = True,
        quantize_type: Optional[str] = None,
    ):
        super().__init__()
        if has_weights:
            self.weight = nn.Parameter(
                torch.ones(hidden_size, dtype=dtype, device=device))
        else:
            self.register_buffer('weight',
                                 torch.ones(hidden_size,
                                            dtype=dtype,
                                            device=device),
                                 persistent=False)
        if has_bias:
            self.bias = nn.Parameter(
                torch.zeros(hidden_size, dtype=dtype, device=device))
        else:
            self.register_buffer('bias',
                                 torch.zeros(hidden_size,
                                             dtype=dtype,
                                             device=device),
                                 persistent=False)
        self.variance_epsilon = eps
        self.has_weights = has_weights
        self.has_bias = has_bias

        if quantize_type is not None and quantize_type != "nvfp4":
            raise NotImplementedError(
                f"Quantize type {quantize_type} not implemented in LayerNorm")
        self.is_nvfp4 = quantize_type == "nvfp4"

        # fused_layernorm_quantize requires SM100 (Blackwell) because the
        # kernel uses cvt.rn.satfinite.e2m1x2.f32. On older GPUs, silently
        # fall back to the unfused path -- the downstream NVFP4 Linear will
        # still produce correct (just slower) outputs.
        if self.is_nvfp4:
            sm_version = get_sm_version()
            if not (100 <= sm_version < 120):
                self.is_nvfp4 = False

    @staticmethod
    def _validate_adaln_pair(
        scale_msa: Optional[torch.Tensor],
        shift_msa: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Enforce that AdaLN modulation is provided as a pair.

        The fused kernel binding (``cpp/tensorrt_llm/thop/fusedLayerNormQuant.cpp``)
        TORCH_CHECKs that ``scale_msa.has_value() == shift_msa.has_value()``.
        The historical unfused path silently treated one-sided inputs as
        "modulation absent", which would diverge from the fused path on
        non-SM100 GPUs or when no ``nvfp4_scale`` is attached. Centralize
        the contract here so both paths see identical semantics.
        """
        if (scale_msa is None) != (shift_msa is None):
            raise ValueError(
                "scale_msa and shift_msa must be provided together "
                "(both None or both tensors); got "
                f"scale_msa={'tensor' if scale_msa is not None else 'None'}, "
                f"shift_msa={'tensor' if shift_msa is not None else 'None'}.")
        return scale_msa, shift_msa

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = ...,
        *,
        scale_msa: Optional[torch.Tensor] = None,
        shift_msa: Optional[torch.Tensor] = None,
        seq_len_per_batch: int = 1,
    ) -> Union[torch.Tensor, Fp4QuantizedTensor, Tuple[torch.Tensor,
                                                       torch.Tensor]]:
        """Apply layer normalization to input tensor.

        Args:
            hidden_states: Input tensor to normalize. Last dim must match hidden_size.
            residual: Optional residual tensor to add to hidden_states before
                normalization. If provided, the returned tuple's second element
                is the pre-norm sum (input + residual) cast back to input dtype.
            scale_msa: Optional AdaLN modulation scale, shape [B, N] or
                broadcastable to hidden_states. When provided alongside
                shift_msa, the normalized output is multiplied by (1 + scale_msa)
                and shifted by shift_msa, matching the AdaLN-DiT convention.
            shift_msa: Optional AdaLN modulation shift, same shape as scale_msa.
            seq_len_per_batch: When the fused NVFP4 path is active and
                scale_msa has shape [B, N] (smaller than hidden_states), this
                tells the kernel how many rows of input share one modulation
                vector. Default 1 (per-row scale_msa).

        Returns:
            One of:
            - ``torch.Tensor`` (unfused path, no residual): the normalized
              (and optionally modulated) output.
            - ``Fp4QuantizedTensor`` (fused NVFP4 path, no residual): a wrapper
              around the packed FP4 tensor and its scale-factor tensor.
            - ``Tuple[torch.Tensor, torch.Tensor]`` (with residual): normalized
              output and the input+residual sum.
        """
        # Centralized pair validation so the unfused fallback sees the same
        # AdaLN contract the fused kernel enforces (see _validate_adaln_pair).
        scale_msa, shift_msa = self._validate_adaln_pair(scale_msa, shift_msa)

        # Fused NVFP4 fast path. Triggered only when:
        #   1. nvfp4 quant is enabled and supported on this GPU,
        #   2. an `nvfp4_scale` has been attached to this module,
        #   3. no residual (the fused kernel doesn't support residual yet).
        nvfp4_scale = getattr(self, "nvfp4_scale", None)
        if (self.is_nvfp4 and nvfp4_scale is not None and residual is ...):
            # Distinct NVTX label so the fused path shows up clearly in nsys
            # (vs the auto-generated `LayerNorm.forward` range from
            # `--enable_layerwise_nvtx_marker`). Counting these in
            # `nsys stats --report nvtxsum` gives a per-block confirmation
            # that the fast path actually fired.
            with nvtx_range("LN+NVFP4 fused", color="green"):
                return self._forward_nvfp4_fused(hidden_states, scale_msa,
                                                 shift_msa, seq_len_per_batch,
                                                 nvfp4_scale)

        with nvtx_range("LN unfused", color="grey"):
            return self._forward_unfused(hidden_states, residual, scale_msa,
                                         shift_msa)

    @maybe_compile(dynamic=True)
    def _forward_unfused(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        scale_msa: Optional[torch.Tensor],
        shift_msa: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Reference path. Mathematically equivalent to the fused path
        (up to FP rounding). Used when NVFP4 is not enabled or supported.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if isinstance(residual, torch.Tensor):
            hidden_states = hidden_states + residual.to(torch.float32)
            residual = hidden_states.to(input_dtype)

        # LayerNorm in FP32. Output stays in FP32 to keep the modulation
        # math at full precision, matching the original Wan formulation
        # `self.norm1(x.float()) * (1 + scale_msa) + shift_msa`.
        hidden_states = nn.functional.layer_norm(
            hidden_states,
            (hidden_states.shape[-1], ),
            weight=self.weight,
            bias=self.bias,
            eps=self.variance_epsilon,
        )

        # AdaLN modulation in FP32: y = ln_out * (1 + scale_msa) + shift_msa.
        # Broadcasting handles the typical Wan shapes: hidden_states is
        # [B, S, N], scale_msa/shift_msa are [B, 1, N], result is [B, S, N].
        if scale_msa is not None and shift_msa is not None:
            hidden_states = (hidden_states * (1 + scale_msa.to(torch.float32)) +
                             shift_msa.to(torch.float32))

        # Cast back to input dtype after all FP32 math is done.
        hidden_states = hidden_states.to(input_dtype)

        if residual is ...:
            return hidden_states
        else:
            return hidden_states, residual

    def _forward_nvfp4_fused(
        self,
        hidden_states: torch.Tensor,
        scale_msa: Optional[torch.Tensor],
        shift_msa: Optional[torch.Tensor],
        seq_len_per_batch: int,
        nvfp4_scale: torch.Tensor,
    ) -> Fp4QuantizedTensor:
        """Fused LayerNorm + (optional AdaLN modulation) + NVFP4 quantize.

        Dispatches to two configurations of the same CUDA kernel:
          - AdaLN: scale_msa/shift_msa provided, LN affine is None.
          - Plain LN: ln_weight/ln_bias from self, scale_msa/shift_msa are None.
        Both end with the same NVFP4 quantization step.
        """
        orig_shape = tuple(hidden_states.shape)
        n = int(orig_shape[-1])
        hs_2d = hidden_states.reshape(-1, n).contiguous()
        sf_scale = nvfp4_scale.contiguous()

        if scale_msa is not None and shift_msa is not None:
            # AdaLN case (norm1, norm3 in Wan 2.2): no LN affine, just modulation.
            # scale_msa / shift_msa typically arrive as [B, 1, N]; reshape to [B, N].
            # Cast to input dtype (callers often keep modulation in FP32 for the
            # unfused path's precision; the fused kernel reads bf16/fp16).
            s_2d = scale_msa.reshape(-1, n).to(hs_2d.dtype).contiguous()
            sh_2d = shift_msa.reshape(-1, n).to(hs_2d.dtype).contiguous()
            ln_w = None
            ln_b = None
            # Validate the row-to-batch ratio so the kernel's
            # batch_idx = row / seq_len_per_batch indexing is correct.
            if hs_2d.shape[0] != s_2d.shape[0] * seq_len_per_batch:
                raise ValueError(
                    f"hidden_states M={hs_2d.shape[0]} must equal "
                    f"scale_msa B={s_2d.shape[0]} * seq_len_per_batch={seq_len_per_batch}."
                )
        else:
            # Plain LN case (norm2 in Wan 2.2): use learned weight and bias.
            s_2d = None
            sh_2d = None
            ln_w = self.weight.to(hs_2d.dtype).contiguous()
            ln_b = self.bias.to(hs_2d.dtype).contiguous()

        fp4_tensor, sf_tensor = torch.ops.trtllm.fused_layernorm_quantize(
            hs_2d,
            ln_w,
            ln_b,
            s_2d,
            sh_2d,
            sf_scale,
            seq_len_per_batch,
            self.variance_epsilon,
            16,  # sf_vec_size
        )

        # Reshape the [M, N/2] FP4 packed tensor back to match the input rank.
        if len(orig_shape) != 2:
            fp4_tensor = fp4_tensor.reshape(*orig_shape[:-1], n // 2)

        return Fp4QuantizedTensor(
            fp4_tensor=fp4_tensor,
            scaling_factor=sf_tensor,
            is_sf_swizzled=True,
        )

    def skip_forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = ...,
        *,
        scale_msa: Optional[torch.Tensor] = None,
        shift_msa: Optional[torch.Tensor] = None,
        seq_len_per_batch: int = 1,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Skip normalization and return inputs unchanged.

        Note: when this skip path is used together with the AdaLN call
        signature, scale_msa/shift_msa are silently ignored, matching the
        intent of "no-op LayerNorm."
        """
        if residual is ...:
            return hidden_states
        else:
            return hidden_states, residual
