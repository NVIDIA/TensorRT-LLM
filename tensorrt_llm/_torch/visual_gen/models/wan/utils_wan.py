# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""Wan fused normalization modules.

Keeps kernel selection, eligibility, tensor contracts, and dispatch behind
model-level normalization operations used by WanBlock.
"""

from typing import NamedTuple, Optional, Sequence, Union

import torch

from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.utils import Fp4QuantizedTensor

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.pertoken_adaln import (
        fused_pertoken_adaln as _fused_pertoken_adaln,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.pertoken_adaln import (
        fused_pertoken_adaln_residual as _fused_pertoken_adaln_residual,
    )

    _PERTOKEN_ADALN_IMPORT_OK = True
except (ImportError, OSError):
    _fused_pertoken_adaln = None
    _fused_pertoken_adaln_residual = None
    _PERTOKEN_ADALN_IMPORT_OK = False

_PERTOKEN_ADALN_ALIGNMENT = 32


def _pertoken_adaln_arch_ok(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability(device) == (10, 0)
    except RuntimeError:
        return False


class _PerTokenAdaLNContext(NamedTuple):
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor
    ffn_shift: torch.Tensor
    ffn_scale: torch.Tensor
    ffn_gate: torch.Tensor
    table_rows: torch.Tensor


class WanPerTokenAdaLN:
    """Adapt Wan's BF16 per-token normalization sites to the fused SM100 kernels.

    Downstream Linear modules may independently quantize these BF16 activations.
    """

    def __init__(
        self,
        hidden_size: int,
        dtype: torch.dtype,
        competing_fusion: bool,
    ) -> None:
        self._eligible = (
            _PERTOKEN_ADALN_IMPORT_OK
            and dtype == torch.bfloat16
            and hidden_size % 256 == 0
            and hidden_size <= 8192
            and not competing_fusion
        )
        self._enabled = False

    def set_runtime_enabled(self, enabled: bool) -> None:
        self._enabled = self._eligible and enabled

    def prepare(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        scale_shift_table: torch.Tensor,
    ) -> Optional[_PerTokenAdaLNContext]:
        if not (
            self._enabled
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and temb.device == x.device
            and temb.dtype == x.dtype
            and temb.ndim == 4
            and temb.shape == (*x.shape[:2], 6, x.shape[-1])
            and x.stride(-1) == 1
            and temb.stride(-1) == 1
            and scale_shift_table.device == x.device
        ):
            return None

        table_rows = scale_shift_table[0].float()
        return _PerTokenAdaLNContext(
            shift=temb[:, :, 0],
            scale=temb[:, :, 1],
            gate=temb[:, :, 2],
            ffn_shift=temb[:, :, 3],
            ffn_scale=temb[:, :, 4],
            ffn_gate=temb[:, :, 5],
            table_rows=table_rows,
        )

    def normalize_input(
        self,
        x: torch.Tensor,
        context: _PerTokenAdaLNContext,
        eps: float,
    ) -> torch.Tensor:
        assert _fused_pertoken_adaln is not None
        return _fused_pertoken_adaln(
            x,
            context.shift,
            context.scale,
            context.table_rows[0],
            context.table_rows[1],
            eps,
        )

    def add_self_attention_and_normalize(
        self,
        residual: torch.Tensor,
        attention_output: torch.Tensor,
        context: _PerTokenAdaLNContext,
        norm: torch.nn.Module,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if not (isinstance(norm, LayerNorm) and norm.weight is not None and norm.bias is not None):
            return None
        assert _fused_pertoken_adaln_residual is not None
        return _fused_pertoken_adaln_residual(
            residual,
            attention_output,
            context.gate,
            context.table_rows[2],
            norm.weight.float(),
            norm.bias.float(),
            None,
            None,
            None,
            None,
            norm.variance_epsilon,
        )

    def add_cross_attention_and_normalize(
        self,
        residual: torch.Tensor,
        attention_output: torch.Tensor,
        context: _PerTokenAdaLNContext,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert _fused_pertoken_adaln_residual is not None
        return _fused_pertoken_adaln_residual(
            residual,
            attention_output,
            None,
            None,
            None,
            None,
            context.ffn_shift,
            context.ffn_scale,
            context.table_rows[3],
            context.table_rows[4],
            eps,
        )

    @staticmethod
    def self_attention_gate(context: _PerTokenAdaLNContext) -> torch.Tensor:
        return context.table_rows[2] + context.gate.float()

    @staticmethod
    def ffn_gate(context: _PerTokenAdaLNContext) -> torch.Tensor:
        return context.table_rows[5] + context.ffn_gate.float()


class WanPerTokenAdaLNRuntime:
    """Resolve shared runtime eligibility before individually compiled Wan blocks."""

    def __init__(self, adapters: Sequence[WanPerTokenAdaLN]) -> None:
        self._adapters = tuple(adapters)
        self._runtime_key: Optional[tuple[torch.device, torch.dtype, bool]] = None
        self._enabled = False

    def prepare(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_size = x.shape[-1]
        temb_layout_ok = (
            temb.ndim == 4
            and temb.device == x.device
            and temb.dtype == x.dtype
            and temb.stride(-1) == 1
            and temb.stride(0) % hidden_size == 0
            and temb.stride(1) % hidden_size == 0
            and temb.data_ptr() % _PERTOKEN_ADALN_ALIGNMENT == 0
        )
        runtime_key = (x.device, x.dtype, temb_layout_ok)
        if self._runtime_key != runtime_key:
            runtime_ok = (
                x.dtype == torch.bfloat16 and _pertoken_adaln_arch_ok(x.device) and temb_layout_ok
            )
            for adapter in self._adapters:
                adapter.set_runtime_enabled(runtime_ok)
            self._enabled = runtime_ok and any(adapter._enabled for adapter in self._adapters)
            self._runtime_key = runtime_key

        if self._enabled and (
            x.stride(-1) != 1
            or x.stride(0) % hidden_size != 0
            or x.stride(1) % hidden_size != 0
            or x.data_ptr() % _PERTOKEN_ADALN_ALIGNMENT != 0
        ):
            return x.clone(memory_format=torch.contiguous_format)
        return x


def get_nvfp4_input_scale(linear) -> Optional[torch.Tensor]:
    """Return the calibrated NVFP4 input_scale for a Linear, or None if not applicable.

    Returns None when the linear is not NVFP4-quantized, uses a non-16 group size,
    has an AWQ pre_quant_scale that must be folded into bf16 input first, or uses
    dynamic quantization (which recomputes input_scale per-forward and cannot
    consume a statically pre-quantized FP4 tensor).
    """
    if linear is None:
        return None
    scale = getattr(linear, "input_scale", None)
    if scale is None:
        return None
    if getattr(linear, "scaling_vector_size", None) != 16:
        return None
    if getattr(linear, "pre_quant_scale", None) is not None:
        return None
    if getattr(linear, "force_dynamic_quantization", False):
        return None
    return scale


def apply_fused_layernorm_adaln_quant(
    x: torch.Tensor,
    scale_msa: torch.Tensor,
    shift_msa: torch.Tensor,
    seq_len_per_batch: int,
    fp4_input_scale: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> Union[torch.Tensor, "Fp4QuantizedTensor"]:
    """Fused LayerNorm + AdaLN (y = (1 + scale_msa) * x_hat + shift_msa) + optional NVFP4 quant.

    Used for norm1 and norm3 in WanBlock (no learned affine params; modulation from timestep emb).
    Returns Fp4QuantizedTensor when fp4_input_scale is provided, else a bf16 tensor.
    """
    # .contiguous() handles non-contiguous views (chunk/squeeze/reshape patterns) and
    # transposed layouts injected by torch.compile's inductor memory planner.
    x = x.contiguous()
    scale_msa = scale_msa.to(dtype=x.dtype).contiguous()
    shift_msa = shift_msa.to(dtype=x.dtype).contiguous()
    if fp4_input_scale is not None:
        y_fp4, sf_out = torch.ops.trtllm.fused_adaptive_layernorm_quant(
            x, None, None, scale_msa, shift_msa, fp4_input_scale, seq_len_per_batch, eps
        )
        return Fp4QuantizedTensor(y_fp4, sf_out)
    out = torch.ops.trtllm.fused_adaptive_layernorm(
        x, None, None, scale_msa, shift_msa, seq_len_per_batch, eps
    )
    return out


def apply_fused_layernorm_affine_quant(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    fp4_input_scale: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> Union[torch.Tensor, "Fp4QuantizedTensor"]:
    """Fused LayerNorm + affine (learned weight/bias) + optional NVFP4 quant.

    Used for norm2 in WanBlock (learned LN params; no AdaLN modulation).
    Returns Fp4QuantizedTensor when fp4_input_scale is provided, else a bf16 tensor.
    """
    x = x.contiguous()
    # seq_len_per_batch is unused on the affine path (kernel only reads it under HAS_MODULATION).
    # Pass 0 so any future kernel change that accidentally reads it here fails loudly.
    seq_len_per_batch = 0
    if fp4_input_scale is not None:
        y_fp4, sf_out = torch.ops.trtllm.fused_adaptive_layernorm_quant(
            x,
            ln_weight.to(x.dtype),
            ln_bias.to(x.dtype),
            None,
            None,
            fp4_input_scale,
            seq_len_per_batch,
            eps,
        )
        return Fp4QuantizedTensor(y_fp4, sf_out)
    out = torch.ops.trtllm.fused_adaptive_layernorm(
        x, ln_weight.to(x.dtype), ln_bias.to(x.dtype), None, None, seq_len_per_batch, eps
    )
    return out
