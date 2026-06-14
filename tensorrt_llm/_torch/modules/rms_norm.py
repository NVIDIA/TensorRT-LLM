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

import enum
from types import EllipsisType  # https://stackoverflow.com/a/66636313
from typing import Optional, Tuple, TypeAlias, Union, cast

import torch
from torch import nn

from ..._utils import get_sm_version
from ..cuda_tile_utils import IS_CUDA_TILE_AVAILABLE
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from ..utils import Fp4QuantizedTensor

# Hidden-dim bounds of the warp-specialized fused_add_rms_norm_quant kernel
# (cpp/tensorrt_llm/thop/fusedAddRMSNormQuant.cpp).
_WS_MIN_N = 2048
_WS_MAX_N = 16384
# Row-count crossover for the layer-boundary add+RMSNorm+quant edge: below this
# the one-CTA-per-row reduce_fusion kernel (fused_add_rmsnorm_fp4_quantize) is
# faster; at/above it the warp-specialized kernel's DMA-ahead pipeline wins.
# Measured at N=7168 on GB200 (ws first overtakes at M=4096, ~6% faster; M<=3072
# favors reduce_fusion). See analysis/bench_rmsnorm_threshold_n7168.py.
_WS_M_THRESHOLD = 4096


class RMSNorm(nn.Module):

    _ARGUMENT_NOT_SPECIFIED_SENTINEL = ...
    _ArgumentNotSpecifiedSentinelType: TypeAlias = EllipsisType

    def __init__(
        self,
        *,
        hidden_size: int,
        eps: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        has_weights: bool = True,
        use_gemma: bool = False,
        quantize_type: Optional[str] = None,
        use_cuda_tile: bool = False,
        return_hp_output: bool = False,
    ):
        super().__init__()

        if use_gemma and not has_weights:
            raise ValueError("has_weights must be True if use_gemma is True")
        if quantize_type is not None:
            if quantize_type != "nvfp4":
                raise NotImplementedError(
                    f"Quantize type {quantize_type} not implemented in RMSNorm")
        self.is_nvfp4 = quantize_type == "nvfp4"
        if use_cuda_tile and not IS_CUDA_TILE_AVAILABLE:
            raise ValueError(
                "cuda.tile is not available, please install cuda-tile pypi package"
            )

        if has_weights:
            if not use_gemma:
                self.weight = nn.Parameter(
                    torch.ones(hidden_size, dtype=dtype, device=device))
            else:
                self.weight = nn.Parameter(
                    torch.zeros(hidden_size, dtype=dtype, device=device))
        else:
            self.register_buffer('weight',
                                 torch.ones(hidden_size,
                                            dtype=dtype,
                                            device=device),
                                 persistent=False)
        self.variance_epsilon = eps
        self.use_gemma = use_gemma
        self.use_cuda_tile = use_cuda_tile

        # fused_add_rms_norm_quant only supports SM 9.x / 10.x because:
        #  - Device code is guarded by is_major_v<9> || is_major_v<10>
        #    (ws_layernorm.cuh:828, low_latency_layernorm.cuh:157).
        # On unsupported SMs, fall back to flashinfer/generic RMSNorm and let
        # the downstream linear layer handle FP4 quantization.
        if self.is_nvfp4:
            sm_version = get_sm_version()
            if not (90 <= sm_version < 120):
                self.is_nvfp4 = False
                return_hp_output = False
        self.return_hp_output = return_hp_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Union[
            Optional[torch.Tensor],
            _ArgumentNotSpecifiedSentinelType] = _ARGUMENT_NOT_SPECIFIED_SENTINEL,
        *,
        return_norm_out: bool = False,
    ) -> Union[torch.Tensor, Fp4QuantizedTensor, Tuple[Union[
            torch.Tensor, Fp4QuantizedTensor], Optional[torch.Tensor]], Tuple[
                Fp4QuantizedTensor, torch.Tensor, torch.Tensor]]:
        has_residual = residual is not self._ARGUMENT_NOT_SPECIFIED_SENTINEL
        if not has_residual:
            residual = None

        # Fused (add +) RMSNorm + NVFP4 input-quantize: when this norm feeds an
        # NVFP4 Linear whose static input_scale is attached as self.nvfp4_scale,
        # fold this norm and that Linear's input-quant into one kernel. With a
        # residual this is the layer-boundary add+RMSNorm+quant; without, the
        # residual-less variant (e.g. q_a_layernorm -> q_b_proj). The actual
        # kernel is chosen by a size gate inside _fused_nvfp4_quant. The BF16
        # post-RMSNorm value is also produced when return_norm_out (stashed on
        # the Fp4QuantizedTensor for DSA consumers / returned for residual-less
        # callers) or when self.return_hp_output (appended as an extra output
        # for MoE-gate consumers). When is_nvfp4 but no scale is attached yet
        # (e.g. a layer's first input_layernorm whose consumer has no static
        # scale), fall through to the plain norm below.
        nvfp4_scale = getattr(self, "nvfp4_scale",
                              None) if self.is_nvfp4 else None
        if nvfp4_scale is not None and not self.use_gemma:
            return self._fused_nvfp4_quant(hidden_states, residual, nvfp4_scale,
                                           return_norm_out)

        if self.return_hp_output:
            raise ValueError(
                "Auxiliary high precision output is only supported for NVFP4 fused path"
            )

        if self.use_cuda_tile:
            if isinstance(residual, torch.Tensor):
                # Use fused residual kernel
                hidden_states = hidden_states.contiguous()
                residual = residual.contiguous()
                torch.ops.trtllm.cuda_tile_rms_norm_fuse_residual_(
                    x=hidden_states,
                    residual=residual,
                    weight=self.weight,
                    eps=self.variance_epsilon,
                    static_persistent=True,
                    gather=True,
                    use_gemma=self.use_gemma,
                )
            else:
                hidden_states = torch.ops.trtllm.cuda_tile_rms_norm(
                    x=hidden_states,
                    weight=self.weight,
                    eps=self.variance_epsilon,
                    static_persistent=True,
                    gather=True,
                    use_gemma=self.use_gemma,
                )
        elif IS_FLASHINFER_AVAILABLE:
            from ..custom_ops import (flashinfer_fused_add_rmsnorm,
                                      flashinfer_gemma_fused_add_rmsnorm,
                                      flashinfer_gemma_rmsnorm,
                                      flashinfer_rmsnorm)
            if residual is not None:
                if not self.use_gemma:
                    flashinfer_fused_add_rmsnorm(hidden_states, residual,
                                                 self.weight,
                                                 self.variance_epsilon)
                else:
                    flashinfer_gemma_fused_add_rmsnorm(hidden_states, residual,
                                                       self.weight,
                                                       self.variance_epsilon)
            else:
                if not self.use_gemma:
                    hidden_states = flashinfer_rmsnorm(hidden_states,
                                                       self.weight,
                                                       self.variance_epsilon)
                else:
                    hidden_states = flashinfer_gemma_rmsnorm(
                        hidden_states, self.weight, self.variance_epsilon)
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            if residual is not None:
                hidden_states = hidden_states + residual.to(torch.float32)
                residual = hidden_states.to(input_dtype)

            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance +
                                                        self.variance_epsilon)
            if not self.use_gemma:
                hidden_states = self.weight * hidden_states.to(input_dtype)
            else:
                hidden_states = (self.weight +
                                 1) * hidden_states.to(input_dtype)

        if has_residual:
            return hidden_states, cast(Optional[torch.Tensor], residual)
        else:
            return hidden_states

    def _ws_kernel_eligible(self, hidden_states: torch.Tensor,
                            residual: Optional[torch.Tensor]) -> bool:
        """Whether the warp-specialized fused_add_rms_norm_quant kernel should
        serve this call instead of the one-CTA-per-row reduce_fusion kernel.

        ws only wins, and is only legal, for the contiguous rank-2 residual-add
        edge with a large enough row count and an in-range hidden dim. The
        residual-less and row-strided (column-slice) edges have no ws equivalent
        and always use reduce_fusion. See _WS_M_THRESHOLD / _WS_MIN_N."""
        if residual is None:
            return False
        n = hidden_states.shape[-1]
        if not (_WS_MIN_N <= n <= _WS_MAX_N and n % 16 == 0):
            return False
        m = 1
        for d in hidden_states.shape[:-1]:
            m *= d
        if m < _WS_M_THRESHOLD:
            return False
        # ws requires contiguous rank-2 input + residual (it cannot read a
        # row-strided column slice and issues padded vectorized stores).
        return hidden_states.is_contiguous() and residual.is_contiguous()

    def _fused_nvfp4_quant(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        sf_scale: torch.Tensor,
        return_norm_out: bool,
    ):
        """Fold this (add +) RMSNorm + the consuming NVFP4 Linear's input-quant
        into one kernel. ``sf_scale`` is that Linear's static input_scale.

        Picks the kernel by problem size (see _ws_kernel_eligible): the
        warp-specialized ``fused_add_rms_norm_quant`` for the large contiguous
        residual edge, else the one-CTA-per-row ``fused_add_rmsnorm_fp4_quantize``
        / ``fused_rmsnorm_fp4_quantize`` (which also serve the residual-less and
        row-strided column-slice edges ws cannot).

        With a residual: returns ``(Fp4QuantizedTensor, residual_out)``. Without:
        returns an ``Fp4QuantizedTensor`` (the residual-less q_a edge). When
        ``return_norm_out`` the BF16 post-RMSNorm view is stashed on the
        Fp4QuantizedTensor's ``bf16_hidden_states`` (and, residual-less, returned
        alongside). When ``self.return_hp_output`` the BF16 normed value is
        appended as a trailing output for MoE-gate consumers."""
        # Either flag means "also produce the BF16 normed value".
        want_norm = return_norm_out or self.return_hp_output

        if self._ws_kernel_eligible(hidden_states, residual):
            return self._fused_nvfp4_quant_ws(hidden_states, residual, sf_scale,
                                              return_norm_out)

        if residual is not None:
            results = torch.ops.trtllm.fused_add_rmsnorm_fp4_quantize(
                hidden_states,
                residual,
                self.weight,
                sf_scale,
                float(self.variance_epsilon),
                want_norm,
            )
            if want_norm:
                bf16_hs, act_fp4, act_sf, residual_out = results
            else:
                act_fp4, act_sf, residual_out = results
                bf16_hs = None
            fp4 = Fp4QuantizedTensor(act_fp4,
                                     act_sf,
                                     bf16_hidden_states=bf16_hs)
            outputs = [fp4, residual_out]
            if self.return_hp_output:
                outputs.append(bf16_hs)
            return tuple(outputs)

        orig_shape = tuple(hidden_states.shape)
        n = orig_shape[-1]
        hs_2d = hidden_states.reshape(-1, n)
        results = torch.ops.trtllm.fused_rmsnorm_fp4_quantize(
            hs_2d,
            self.weight,
            sf_scale,
            float(self.variance_epsilon),
            want_norm,
        )
        if want_norm:
            norm_out, act_fp4, act_sf = results
            norm_out = norm_out.reshape(orig_shape)
        else:
            act_fp4, act_sf = results
            norm_out = None
        if len(orig_shape) != 2:
            act_fp4 = act_fp4.reshape(*orig_shape[:-1], n // 2)
        fp4 = Fp4QuantizedTensor(act_fp4, act_sf, bf16_hidden_states=norm_out)
        return (fp4, norm_out) if return_norm_out else fp4

    def _fused_nvfp4_quant_ws(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        sf_scale: torch.Tensor,
        return_norm_out: bool,
    ):
        """Warp-specialized fused add+RMSNorm+NVFP4-quant
        (torch.ops.trtllm.fused_add_rms_norm_quant) for the large contiguous
        residual edge. Returns the same shape as the reduce_fusion residual
        branch of _fused_nvfp4_quant."""
        want_norm = return_norm_out or self.return_hp_output
        orig_shape = tuple(hidden_states.shape)
        n = int(orig_shape[-1])
        hs_2d = hidden_states.reshape(-1, n).contiguous()
        res_2d = residual.reshape(-1, n).contiguous()
        gamma = self.weight.contiguous()

        results = torch.ops.trtllm.fused_add_rms_norm_quant(
            hs_2d,
            res_2d,
            gamma,
            sf_scale.contiguous(),
            True,
            eps=self.variance_epsilon,
            output_hp_norm=want_norm,
        )
        normed_fp4_i32, residual_out_2d, sf_fused = results[:3]
        normed_fp4_u8 = normed_fp4_i32.view(torch.uint8)
        if len(orig_shape) != 2:
            normed_fp4_u8 = normed_fp4_u8.reshape(*orig_shape[:-1], n // 2)
            residual_out = residual_out_2d.reshape(orig_shape)
        else:
            residual_out = residual_out_2d

        bf16_hs = results[3].reshape(orig_shape) if want_norm else None
        fp4 = Fp4QuantizedTensor(normed_fp4_u8,
                                 sf_fused,
                                 bf16_hidden_states=bf16_hs)
        outputs = [fp4, residual_out]
        if self.return_hp_output:
            outputs.append(bf16_hs)
        return tuple(outputs)

    def skip_forward(
        self,
        hidden_states: torch.Tensor,
        residual: Union[
            Optional[torch.Tensor],
            _ArgumentNotSpecifiedSentinelType] = _ARGUMENT_NOT_SPECIFIED_SENTINEL,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if residual is self._ARGUMENT_NOT_SPECIFIED_SENTINEL:
            return hidden_states
        else:
            return hidden_states, cast(Optional[torch.Tensor], residual)


class GroupRMSNormKernelSelection(enum.Enum):
    heuristic = 0
    base = 1
    large_batch = 2


def group_rms_norm(
        inputs: list[torch.Tensor],
        weights: Optional[list[torch.Tensor]] = [],
        eps: Optional[float] = 1e-5,
        weight_bias: Optional[float] = 0.0,
        kernel: GroupRMSNormKernelSelection = GroupRMSNormKernelSelection.
    heuristic,
        outputs: Optional[list[torch.Tensor]] = None) -> list[torch.Tensor]:
    '''Group RMS Normalization optimized for up to 2 inputs.

    This function applies RMS normalization to multiple inputs simultaneously,
    achieving better performance than normalizing each tensor separately with multi-stream.

    Args:
        inputs: List of input tensors to normalize
        weights: Optional list of weight tensors corresponding to each input
        eps: Small constant added to variance for numerical stability
        weight_bias: Optional bias added to weights during normalization
        kernel: Kernel selection strategy:
            - heuristic: Automatically selects optimal kernel based on inputs and hardware
            - base: Uses base kernel (optimal for most cases)
            - large_batch: Uses large batch kernel (may be better for large batches)
        outputs: Optional pre-allocated output tensors (created if None)

    Returns:
        List of normalized tensors with the same shapes as inputs

    Technical Details:
        Available kernel implementations:
        - Base kernel: Allocates warps proportional to the sum of last dimensions,
          providing better SM occupancy for most workloads.
        - Large batch kernel: Allocates warps proportional to the maximum last dimension,
          which can be more efficient for large batch sizes with 2 inputs.

        The heuristic mode uses a logistic regression model trained on benchmark data
        to dynamically select the optimal kernel based on batch size, input dimensions,
        and GPU architecture. This selection is optimized for compute capabilities 9.x and 10.x.
    '''
    out = outputs
    if out is None:
        out = [torch.empty_like(input) for input in inputs]
    match kernel:
        case GroupRMSNormKernelSelection.heuristic:
            torch.ops.trtllm.group_rms_norm_heuristic(inputs, out, weights, eps,
                                                      weight_bias)
        case GroupRMSNormKernelSelection.base:
            torch.ops.trtllm.group_rms_norm_base(inputs, out, weights, eps,
                                                 weight_bias)
        case GroupRMSNormKernelSelection.large_batch:
            torch.ops.trtllm.group_rms_norm_large_batch(inputs, out, weights,
                                                        eps, weight_bias)
    return out
