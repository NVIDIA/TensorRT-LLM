# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-Connection residual streams for Qwen4-Exp models."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.mapping import Mapping

from ..linear import Linear
from ..low_m_gemm import low_m_gemm_fused_epilogue_enabled
from ..mamba.layernorm_gated import RMSNorm as TritonRMSNorm
from .hyper_connection_kernels import (
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_moe_finalize_combine_norm,
    hc_silu,
)

__all__ = ["GroupedRMSNorm", "HCResidual", "Qwen4ExpHyperConnection"]

_PACKED_PROJECTION_ALIGNMENT = 16
_TMA_STRIDE_ALIGNMENT_BYTES = 16
# M=8 is supported by both kernels but does not beat the ordinary per-layer
# graph path. Keep capability in the primitives and policy here evidence-bound.
_FUSED_MIX_ROWS = frozenset((1, 2, 4))


class GroupedRMSNorm(TritonRMSNorm):
    """Gemma-style grouped RMS norm using TRT-LLM's grouped Triton kernel.

    A native implementation remains available for CPU/model-construction tests.

    Normalizes over ``group_size`` slices of the last dim independently (RMS,
    fp32 accumulation) and scales by ``(1.0 + weight)`` (Gemma convention: the
    stored weight is a delta around 1). When ``group_size is None`` the whole
    last dim is one group (a plain Gemma RMSNorm).

    For Qwen4-Exp ``hc_per_branch_norm=True``: ``normalized_shape=10240``,
    ``group_size=2560`` — each of the 4 streams is RMS-normalized over its 2560
    dims and scaled by its own slice of the ``[10240]`` weight.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        group_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if group_size is not None and normalized_shape % group_size != 0:
            raise ValueError(
                f"normalized_shape ({normalized_shape}) must be divisible by "
                f"group_size ({group_size})"
            )
        super().__init__(
            normalized_shape,
            eps=eps,
            group_size=group_size,
            dtype=dtype,
            device=device,
            weight_is_delta=True,
        )
        # Gemma init: the weight is a delta around 1, so a fresh (unloaded)
        # module is the identity scale. Constructing a new tensor is required
        # here because TRT-LLM meta initialization rejects in-place ``zero_``.
        self.weight = nn.Parameter(torch.zeros(normalized_shape, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The base class is a Triton kernel and therefore CUDA-only. Serving
        # always takes the branch above; the Torch path below keeps the module
        # runnable on CPU, which is what lets the unit tests check this class
        # itself rather than a stand-in defined in a fixture.
        if x.is_cuda:
            return super().forward(x)
        input_dtype = x.dtype
        x_float = x.float()
        if self.group_size == self.hidden_size:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(variance + self.variance_epsilon)
        else:
            x_grouped = x_float.reshape(
                *x_float.shape[:-1],
                x_float.shape[-1] // self.group_size,
                self.group_size,
            )
            variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
            x_norm = (x_grouped * torch.rsqrt(variance + self.variance_epsilon)).flatten(-2)
        return (x_norm * (1.0 + self.weight.float())).to(input_dtype)


# The original bundle and its combine logits cross the wrapped attention/MLP.
HCResidual = Tuple[torch.Tensor, Optional[torch.Tensor]]


class Qwen4ExpHyperConnection(nn.Module):
    """Qwen4-Exp gated Hyper-Connection block.

    Args:
        hc_count: number of parallel residual streams (config ``hc_count``, 4).
        hidden_size: per-stream hidden size (config ``hidden_size``, 2560).
        hc_lowrank: low-rank width of the mix down/up projection
            (config ``hc_lowrank``, 320).
        rms_norm_eps: epsilon for the grouped RMS norm (config ``rms_norm_eps``).
        hc_per_branch_norm: when True (the Qwen4-Exp checkpoint layout) the norm
            weight is ``[hc_count*hidden_size]`` grouped by ``hidden_size``;
            when False it is a shared ``[hidden_size]`` weight.
        dtype: parameter dtype used to construct the module.
        use_mix / use_combine: which sub-projections to build. The per-layer
            attn/mlp mixers use both; the model's final ``hyper_connection_mixer``
            uses ``use_combine=False`` (mix-only "last norm").
    """

    def __init__(
        self,
        hc_count: int,
        hidden_size: int,
        hc_lowrank: int,
        rms_norm_eps: float = 1e-6,
        hc_per_branch_norm: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        use_mix: bool = True,
        use_combine: bool = True,
        device: Optional[torch.device] = None,
        mapping: Optional[Mapping] = None,
        use_cute_dsl_bf16_gemm: bool = False,
    ) -> None:
        super().__init__()
        if not (use_mix or use_combine):
            raise ValueError("use_mix or use_combine must be set")
        if hc_count <= 0 or hidden_size <= 0 or hc_lowrank <= 0:
            raise ValueError("hc_count, hidden_size, and hc_lowrank must be positive")
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.hc_lowrank = hc_lowrank
        self.hc_per_branch_norm = hc_per_branch_norm
        self.params_dtype = dtype
        self.use_mix = use_mix
        self.use_combine = use_combine
        self._fused_mix_requested = use_mix and low_m_gemm_fused_epilogue_enabled()

        norm_dim = hidden_size * hc_count if hc_per_branch_norm else hidden_size
        norm_group_size = hidden_size if hc_per_branch_norm else None
        self.hc_norm = GroupedRMSNorm(
            norm_dim, eps=rms_norm_eps, group_size=norm_group_size, dtype=dtype, device=device
        )

        hc_dim = hc_count * hidden_size
        if use_mix:
            self.input_mix_injection_offset = hc_lowrank
            logical_rows = hc_lowrank + (hc_count if use_combine else 0)
            packed_rows = (
                (logical_rows + _PACKED_PROJECTION_ALIGNMENT - 1)
                // _PACKED_PROJECTION_ALIGNMENT
                * _PACKED_PROJECTION_ALIGNMENT
            )
            self.input_mix_padding = packed_rows - logical_rows
            if use_combine:
                self.input_mix_weight_down_block_inject = Linear(
                    hc_dim,
                    packed_rows,
                    bias=False,
                    dtype=dtype,
                    mapping=mapping,
                    reduce_output=False,
                    use_cute_dsl_bf16_gemm=use_cute_dsl_bf16_gemm,
                )
            else:
                self.input_mix_weight_down = Linear(
                    hc_dim,
                    hc_lowrank,
                    bias=False,
                    dtype=dtype,
                    mapping=mapping,
                    reduce_output=False,
                    use_cute_dsl_bf16_gemm=use_cute_dsl_bf16_gemm,
                )
            self.input_mix_weight_up = Linear(
                hc_lowrank,
                hc_dim,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                reduce_output=False,
                use_cute_dsl_bf16_gemm=use_cute_dsl_bf16_gemm,
            )
            self.register_buffer(
                "_input_mix_weight_up_interleaved",
                None,
                persistent=False,
            )
            if device is not None:
                self.input_mix_weight_up.to(device=device)
                if use_combine:
                    self.input_mix_weight_down_block_inject.to(device=device)
                else:
                    self.input_mix_weight_down.to(device=device)

    @classmethod
    def from_config(
        cls,
        config: object,
        use_mix: bool = True,
        use_combine: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        mapping: Optional[Mapping] = None,
        use_cute_dsl_bf16_gemm: bool = False,
    ) -> "Qwen4ExpHyperConnection":
        """Build from a ``Qwen4ExpTextConfig``-like object."""
        return cls(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            hc_per_branch_norm=getattr(config, "hc_per_branch_norm", True),
            dtype=dtype,
            use_mix=use_mix,
            use_combine=use_combine,
            device=device,
            mapping=mapping,
            use_cute_dsl_bf16_gemm=use_cute_dsl_bf16_gemm,
        )

    def _normed_bundle(self, hyper_input: torch.Tensor) -> torch.Tensor:
        """Grouped-RMSNorm the ``[..., hc_count*hidden_size]`` bundle (per-stream)."""
        if self.hc_per_branch_norm:
            return self.hc_norm(hyper_input)
        # Shared-weight variant: norm each stream over hidden_size with the same
        # [hidden_size] weight, then re-flatten.
        return self.hc_norm(hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))).flatten(
            -2
        )

    def _packed_down_and_injection(self, normed: torch.Tensor) -> torch.Tensor:
        """Project the mix gate and combine logits in one aligned GEMM."""
        return self.input_mix_weight_down_block_inject(normed)

    @staticmethod
    def _fused_cuda_eligible(*tensors: Optional[torch.Tensor]) -> bool:
        """Whether a fused pointwise kernel can take these row-major operands.

        ``None`` stands for an optional operand the caller does not have, which
        constrains nothing.
        """
        operands = [tensor for tensor in tensors if tensor is not None]
        if not operands:
            return False
        first = operands[0]
        return all(
            tensor.ndim == 2
            and tensor.is_cuda
            and tensor.device == first.device
            and tensor.dtype == first.dtype
            and tensor.dtype in (torch.bfloat16, torch.float16)
            and tensor.stride(-1) == 1
            for tensor in operands
        )

    @staticmethod
    def _fused_norm_weight_eligible(
        weight: torch.Tensor,
        reference: torch.Tensor,
    ) -> bool:
        """Check the vector norm weight omitted from the row-tensor check."""
        return (
            weight.ndim == 1
            and weight.is_cuda
            and weight.device == reference.device
            and weight.dtype == reference.dtype
            and weight.dtype in (torch.bfloat16, torch.float16)
            and weight.is_contiguous()
        )

    def _mix_normed(
        self,
        hyper_input: torch.Tensor,
        normed: torch.Tensor,
    ) -> Tuple[torch.Tensor, HCResidual]:
        """Project a normalized HC bundle and retain its combine logits."""
        hc, hs = self.hc_count, self.hidden_size
        # Graph batches use exact power-of-two rows. Keep all other calls off
        # the eligibility helper so the ordinary eager path does not pay for it.
        if self._fused_mix_requested and normed.ndim == 2 and normed.shape[0] in _FUSED_MIX_ROWS:
            fused = self._fused_mix_normed(normed)
            if fused is not None:
                mixed, injection_logits = fused
                return mixed, (hyper_input, injection_logits)
        if self.use_combine:
            packed = self._packed_down_and_injection(normed)
            down = packed[..., : self.hc_lowrank]
            injection_logits = packed[
                ...,
                self.input_mix_injection_offset : self.input_mix_injection_offset + hc,
            ]
        else:
            down = self.input_mix_weight_down(normed)
            injection_logits = None
        if self._fused_cuda_eligible(down):
            gate = hc_silu(down, hc)
        else:
            gate = F.silu(down / hc)
        gate = self.input_mix_weight_up(gate)
        if self._fused_cuda_eligible(normed, gate):
            mixed = hc_gate_mix(normed, gate, hc)
        else:
            gate = torch.sigmoid(gate).unflatten(-1, (hc, hs))
            mixed = (gate * normed.unflatten(-1, (hc, hs))).mean(dim=-2)

        return mixed.to(self.params_dtype), (hyper_input, injection_logits)

    def _interleaved_up_weight(self) -> torch.Tensor | None:
        """Lazily cache the grouped-reduction output-row layout."""
        weight = self._input_mix_weight_up_interleaved
        if weight is None:
            # Allocating or reordering a parameter during capture would give
            # replay the wrong lifetime; the caller keeps the unfused path.
            if torch.cuda.is_current_stream_capturing():
                return None
            from ...cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
                interleave_grouped_output_rows,
            )

            weight = interleave_grouped_output_rows(
                self.input_mix_weight_up.weight.detach(),
                self.hc_count,
            )
            self._input_mix_weight_up_interleaved = weight
        return weight

    def _fused_mix_normed(
        self,
        normed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Fuse the two adjacent GEMM epilogues for a small graph batch."""
        if not self._fused_mix_requested or torch.is_grad_enabled():
            return None

        hc_count = self.hc_count
        hidden_size = self.hidden_size
        lowrank = self.hc_lowrank
        rows = int(normed.shape[0]) if normed.ndim == 2 else 0
        packed_weight = (
            self.input_mix_weight_down_block_inject.weight
            if self.use_combine
            else self.input_mix_weight_down.weight
        )
        up_weight = self.input_mix_weight_up.weight
        hc_size = hc_count * hidden_size
        logical_rows = lowrank + (hc_count if self.use_combine else 0)
        if not (
            hc_count >= 2
            and self.params_dtype == torch.bfloat16
            and normed.ndim == 2
            and rows in _FUSED_MIX_ROWS
            and normed.shape == (rows, hc_size)
            and packed_weight.ndim == 2
            and packed_weight.shape[0] >= logical_rows
            and packed_weight.shape[0] % _PACKED_PROJECTION_ALIGNMENT == 0
            and packed_weight.shape[1] == hc_size
            and up_weight.shape == (hc_size, lowrank)
            and normed.is_cuda
            and packed_weight.is_cuda
            and up_weight.is_cuda
            and normed.device == packed_weight.device == up_weight.device
            and normed.dtype == torch.bfloat16
            and packed_weight.dtype == torch.bfloat16
            and up_weight.dtype == torch.bfloat16
            and normed.is_contiguous()
            and packed_weight.is_contiguous()
            and up_weight.is_contiguous()
            and up_weight.stride(0) * up_weight.element_size() % _TMA_STRIDE_ALIGNMENT_BYTES == 0
            and not any(tensor.data_ptr() % 32 for tensor in (normed, packed_weight, up_weight))
        ):
            return None
        major, minor = torch.cuda.get_device_capability(normed.device)
        if not is_sm_100f(major * 10 + minor):
            return None

        from ...cute_dsl_kernels.blackwell.low_m_bf16_direct import (
            default_tactic as default_direct_tactic,
        )
        from ...cute_dsl_kernels.blackwell.low_m_bf16_direct import get_direct_dense_epilogue_runner
        from ...cute_dsl_kernels.blackwell.low_m_bf16_direct import (
            validate_tactic as validate_direct_tactic,
        )
        from ...cute_dsl_kernels.blackwell.low_m_bf16_splitk import (
            default_tactic,
            get_splitk_dense_epilogue_runner,
            validate_sigmoid_grouped_reduce_tactic,
        )
        from ...flashinfer_utils import get_env_enable_pdl

        packed_size = int(packed_weight.shape[0])
        try:
            direct_tactic = default_direct_tactic(rows, packed_size, hc_size)
            validate_direct_tactic(direct_tactic, rows, packed_size, hc_size)
            grouped_tactic = default_tactic(rows, hc_size, lowrank)
            validate_sigmoid_grouped_reduce_tactic(
                grouped_tactic,
                rows,
                hc_size,
                lowrank,
                hc_count,
            )
        except ValueError:
            return None

        run_direct_epilogue = get_direct_dense_epilogue_runner("silu_prefix")
        run_splitk_epilogue = get_splitk_dense_epilogue_runner("sigmoid_grouped_reduce")
        interleaved_weight = self._interleaved_up_weight()
        if interleaved_weight is None:
            return None

        pdl = get_env_enable_pdl()
        packed = torch.empty(
            (rows, packed_size),
            dtype=normed.dtype,
            device=normed.device,
        )
        run_direct_epilogue(
            normed,
            packed_weight.t(),
            packed,
            pdl,
            direct_tactic,
            scale=1.0 / hc_count,
            prefix=lowrank,
        )

        mixed = torch.empty(
            (rows, hidden_size),
            dtype=normed.dtype,
            device=normed.device,
        )
        run_splitk_epilogue(
            packed[:, :lowrank],
            interleaved_weight.t(),
            normed,
            mixed,
            pdl,
            grouped_tactic,
            reduction_scale=1.0 / hc_count,
            group_count=hc_count,
        )
        injection_logits = (
            packed[
                ...,
                self.input_mix_injection_offset : self.input_mix_injection_offset + hc_count,
            ]
            if self.use_combine
            else None
        )
        return mixed, injection_logits

    def mix(self, hyper_input: torch.Tensor) -> Tuple[torch.Tensor, HCResidual]:
        """10240 -> 2560. Returns ``(mixed_input, residual_state)`` where
        the original bundle and precomputed injection logits are threaded to
        ``combine`` without retaining the normalized temporary."""
        if not self.use_mix:
            raise RuntimeError("mix() called on a combine-only Hyper-Connection")
        hc, hs = self.hc_count, self.hidden_size
        if hyper_input.shape[-1] != hc * hs:
            raise ValueError(
                f"hyper_input last dim {hyper_input.shape[-1]} != hc_count*hidden ({hc * hs})"
            )
        # Idle ranks and graph padding can present an empty logical batch.
        if hyper_input.numel() == 0:
            mixed = hyper_input.new_empty((*hyper_input.shape[:-1], hs), dtype=self.params_dtype)
            injection_logits = (
                hyper_input.new_empty((*hyper_input.shape[:-1], hc)) if self.use_combine else None
            )
            return mixed, (hyper_input, injection_logits)

        normed = self._normed_bundle(hyper_input)
        return self._mix_normed(hyper_input, normed)

    def combine(self, block_output: torch.Tensor, residual: HCResidual) -> torch.Tensor:
        """2560 -> 10240. Injects ``block_output`` into the 4 residual streams
        with a learned per-stream sigmoid gate."""
        if not self.use_combine:
            raise RuntimeError("combine() called on a mix-only Hyper-Connection")
        hc, hs = self.hc_count, self.hidden_size
        hyper_input, injection_logits = residual
        if (
            hyper_input.shape[-1] != hc * hs
            or block_output.shape[-1] != hs
            or hyper_input.shape[:-1] != block_output.shape[:-1]
        ):
            raise ValueError("Hyper-Connection combine received incompatible hidden dimensions")
        if injection_logits is not None and injection_logits.shape != (
            *block_output.shape[:-1],
            hc,
        ):
            raise ValueError("Hyper-Connection injection logits do not match the block output")
        if block_output.numel() == 0:
            return hyper_input.to(self.params_dtype)

        if injection_logits is None:
            raise RuntimeError("Hyper-Connection combine is missing injection logits")
        if self._fused_cuda_eligible(hyper_input, block_output, injection_logits):
            return hc_combine(hyper_input, block_output, injection_logits, hc)
        streams = hyper_input.unflatten(-1, (hc, hs))
        inject_gate = 2.0 * torch.sigmoid(injection_logits / hc)
        injection = block_output.unsqueeze(-2) * inject_gate.unsqueeze(-1)
        return (streams + injection).flatten(-2).to(self.params_dtype)

    def combine_and_mix(
        self,
        block_output: torch.Tensor,
        previous_residual: HCResidual,
    ) -> Tuple[torch.Tensor, torch.Tensor, HCResidual]:
        """Combine a preceding block and prepare this HC block's input.

        The CUDA path fuses the preceding residual injection with this module's
        grouped Gemma RMSNorm. This boundary occurs between attention and MLP,
        where no PLE update or collective intervenes.
        """
        hyper_input, injection_logits = previous_residual
        if injection_logits is None:
            raise RuntimeError("Hyper-Connection combine-and-mix is missing injection logits")
        if (
            hyper_input.shape[-1] != self.hc_count * self.hidden_size
            or block_output.shape[-1] != self.hidden_size
            or hyper_input.shape[:-1] != block_output.shape[:-1]
            or injection_logits.shape != (*block_output.shape[:-1], self.hc_count)
        ):
            raise ValueError("Hyper-Connection combine-and-mix operands have incompatible shapes")
        if block_output.numel() == 0:
            mixed, residual = self.mix(hyper_input)
            return hyper_input, mixed, residual
        if self._fused_cuda_eligible(
            hyper_input,
            block_output,
            injection_logits,
        ) and self._fused_norm_weight_eligible(self.hc_norm.weight, hyper_input):
            hidden_states, normed = hc_combine_norm(
                hyper_input,
                block_output,
                injection_logits,
                self.hc_norm.weight,
                self.hc_norm.variance_epsilon,
                self.hc_count,
            )
        else:
            hc, hs = self.hc_count, self.hidden_size
            streams = hyper_input.unflatten(-1, (hc, hs))
            inject_gate = 2.0 * torch.sigmoid(injection_logits / hc)
            hidden_states = (
                (streams + block_output.unsqueeze(-2) * inject_gate.unsqueeze(-1))
                .flatten(-2)
                .to(self.params_dtype)
            )
            normed = self._normed_bundle(hidden_states)
        mixed, residual = self._mix_normed(hidden_states, normed)
        return hidden_states, mixed, residual

    def can_fuse_deferred_moe(self, hidden_states: torch.Tensor) -> bool:
        """Whether this HC instance may request deferred TP1 MoE operands."""
        if not (
            self._fused_mix_requested
            and not torch.is_grad_enabled()
            and hidden_states.ndim == 2
            and hidden_states.shape[0] in (1, 2, 4, 8)
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_cuda
        ):
            return False
        major, minor = torch.cuda.get_device_capability(hidden_states.device)
        return is_sm_100f(major * 10 + minor)

    def combine_deferred_moe_and_mix(
        self,
        routed_output: torch.Tensor,
        shared_output: torch.Tensor,
        shared_gate_logits: torch.Tensor,
        previous_residual: HCResidual,
    ) -> Tuple[torch.Tensor, torch.Tensor, HCResidual]:
        """Finalize a TP1 MoE while combining and normalizing its HC output."""
        hyper_input, injection_logits = previous_residual
        if injection_logits is None:
            raise RuntimeError("deferred MoE combine is missing injection logits")
        row_tensors = (
            hyper_input,
            routed_output,
            shared_output,
            shared_gate_logits,
            injection_logits,
        )
        eligible = (
            self.can_fuse_deferred_moe(hyper_input)
            and self.hc_per_branch_norm
            and self._fused_cuda_eligible(*row_tensors)
            and all(
                tensor.shape[0] <= 1 or tensor.stride(0) >= tensor.shape[1]
                for tensor in row_tensors
            )
            and self._fused_norm_weight_eligible(self.hc_norm.weight, hyper_input)
        )
        if eligible:
            hidden_states, normed = hc_moe_finalize_combine_norm(
                hyper_input,
                routed_output,
                shared_output,
                shared_gate_logits,
                injection_logits,
                self.hc_norm.weight,
                self.hc_norm.variance_epsilon,
                self.hc_count,
            )
            mixed, residual = self._mix_normed(hidden_states, normed)
            return hidden_states, mixed, residual

        from ...moe.fused_shared_expert import fused_sigmoid_gate_mul_add

        block_output = fused_sigmoid_gate_mul_add(
            routed_output,
            shared_gate_logits,
            shared_output,
        )
        return self.combine_and_mix(block_output, previous_residual)

    def extra_repr(self) -> str:
        return (
            f"hc_count={self.hc_count}, hidden_size={self.hidden_size}, "
            f"hc_lowrank={self.hc_lowrank}, "
            f"hc_per_branch_norm={self.hc_per_branch_norm}, "
            f"use_mix={self.use_mix}, use_combine={self.use_combine}"
        )
