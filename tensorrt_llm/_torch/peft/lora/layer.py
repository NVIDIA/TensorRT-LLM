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

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from os import getenv
from typing import Dict, List, Optional

import torch

from ...autotuner import (AutoTuner, DynamicTensorSpec, OptimizationProfile,
                          TunableRunner, TuningConfig)
from ...modules.multi_stream_utils import (do_multi_stream,
                                           maybe_execute_in_parallel)
from ...utils import (get_last_power_of_2_num_tokens_buckets,
                      last_positive_power_of_2)
from .cuda_graph_lora_params import CudaGraphLoraParams

_FP8_LORA_TMA_ALIGNMENT = 16


def _validate_fp8_lora_cuda_graph_alignment(slot_ranks_host: torch.Tensor,
                                            hidden_size: int,
                                            output_hidden_sizes: List[int],
                                            max_rank: int) -> int:
    if max_rank < _FP8_LORA_TMA_ALIGNMENT or max_rank % _FP8_LORA_TMA_ALIGNMENT != 0:
        raise ValueError(
            f"FP8 LoRA CUDA graph mode requires max LoRA rank to be a "
            f"multiple of {_FP8_LORA_TMA_ALIGNMENT} and at least "
            f"{_FP8_LORA_TMA_ALIGNMENT}. Got max rank {max_rank}.")

    active_ranks = slot_ranks_host[slot_ranks_host > 0]
    if active_ranks.numel() > 0:
        min_active_rank = int(active_ranks.min().item())
        has_misaligned_rank = bool(
            torch.any(active_ranks % _FP8_LORA_TMA_ALIGNMENT != 0).item())
        if min_active_rank < _FP8_LORA_TMA_ALIGNMENT or has_misaligned_rank:
            raise ValueError(
                f"FP8 LoRA CUDA graph mode requires active LoRA ranks "
                f"to be multiples of {_FP8_LORA_TMA_ALIGNMENT} and at "
                f"least {_FP8_LORA_TMA_ALIGNMENT}. Got active ranks "
                f"{active_ranks.tolist()}.")
    else:
        min_active_rank = max_rank

    fp8_dims = [hidden_size, *output_hidden_sizes]
    misaligned_dims = [
        dim for dim in fp8_dims if dim % _FP8_LORA_TMA_ALIGNMENT != 0
    ]
    if misaligned_dims:
        raise ValueError(
            f"FP8 LoRA CUDA graph mode requires hidden and output sizes "
            f"to be multiples of {_FP8_LORA_TMA_ALIGNMENT}. Got "
            f"{fp8_dims}.")

    return min(hidden_size, min_active_rank)


# TODO: Potentially move this fallback to LoraConfig.
TRTLLM_SPLITK_VAL = int(getenv("TRTLLM_SPLITK_VAL", "8"))
_LORA_SPLIT_K_CANDIDATES = (1, 2, 4, 8, 16)


@dataclass
class GroupedGemmParamsOutput:
    in_sizes: Optional[torch.Tensor] = None
    out_sizes: Optional[torch.Tensor] = None
    a_offset: Optional[torch.Tensor] = None
    d_offset: Optional[torch.Tensor] = None
    d_prime_offset: Optional[torch.Tensor] = None
    lda: Optional[torch.Tensor] = None
    ldb: Optional[torch.Tensor] = None
    ldd: Optional[torch.Tensor] = None
    ldb_prime: Optional[torch.Tensor] = None
    ldd_prime: Optional[torch.Tensor] = None
    splitk_offsets: Optional[torch.Tensor] = None
    reordered_input: Optional[torch.Tensor] = None


@dataclass
class GroupedGemmParamsInput:
    x: torch.Tensor
    output_buffer: torch.Tensor
    intermediate_buffer: torch.Tensor
    max_lora_size: int
    max_rank: int
    slot_counts: torch.Tensor
    slot_ranks: torch.Tensor
    slot_offsets_full: torch.Tensor
    b_ptrs: torch.Tensor
    b_prime_ptrs: torch.Tensor
    sorted_ids: torch.Tensor
    output_hidden_sizes: torch.Tensor
    output_sizes_offset: torch.Tensor

    @property
    def slot_offsets(self):
        return self.slot_offsets_full[:-1]


class LoraModuleType(IntEnum):
    """Enum class representing different types of modules that can have LoRA adapters.

    This enum maps to the different attention and MLP components in a transformer model
    that can be adapted using LoRA weights.
    """
    ATTENTION_QKV = 0  # Combined QKV projection
    ATTENTION_Q = 1  # Query projection
    ATTENTION_K = 2  # Key projection
    ATTENTION_V = 3  # Value projection
    ATTENTION_DENSE = 4  # Output projection after attention

    MLP_H_TO_4H = 5  # First MLP projection (hidden to 4x hidden)
    MLP_4H_TO_H = 6  # Second MLP projection (4x hidden back to hidden)
    MLP_GATE = 7  # Gate projection in MLP

    CROSS_ATTENTION_QKV = 8  # Cross-attention QKV projection
    CROSS_ATTENTION_Q = 9  # Cross-attention Query projection
    CROSS_ATTENTION_K = 10  # Cross-attention Key projection
    CROSS_ATTENTION_V = 11  # Cross-attention Value projection
    CROSS_ATTENTION_DENSE = 12  # Cross-attention output projection

    MOE_H_TO_4H = 13  # MoE first projection
    MOE_4H_TO_H = 14  # MoE second projection
    MOE_GATE = 15  # MoE gate projection
    MOE_ROUTER = 16  # MoE router

    MLP_ROUTER = 17  # MLP router
    MLP_GATE_UP = 18  # Combined gate and up projections

    SHARED_EXPERT_H_TO_4H = 19  # Shared expert first projection
    SHARED_EXPERT_4H_TO_H = 20  # Shared expert second projection
    SHARED_EXPERT_GATE = 21  # Shared expert gate projection

    MAMBA_IN_PROJ = 22  # Mamba input projection
    MAMBA_OUT_PROJ = 23  # Mamba output projection

    MOE_LATENT_FC1 = 24  # MoE latent fc1 projection (fc1_latent_proj)
    MOE_LATENT_FC2 = 25  # MoE latent fc2 projection (fc2_latent_proj)

    def __str__(self):
        """Return the name of the enum value."""
        return self.name

    @classmethod
    def from_string(cls, name: str) -> "LoraModuleType":
        """Convert a string to the corresponding LoraModuleType.

        Args:
            name: The string name of the module type

        Returns:
            The corresponding LoraModuleType enum value

        Raises:
            ValueError: If the name doesn't match any LoraModuleType
        """
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unknown LoRA module type: {name}")

    @property
    def is_attention(self) -> bool:
        """Check if this is an attention module type."""
        return self in {
            self.ATTENTION_QKV, self.ATTENTION_Q, self.ATTENTION_K,
            self.ATTENTION_V, self.ATTENTION_DENSE, self.CROSS_ATTENTION_QKV,
            self.CROSS_ATTENTION_Q, self.CROSS_ATTENTION_K,
            self.CROSS_ATTENTION_V, self.CROSS_ATTENTION_DENSE
        }

    @property
    def is_mlp(self) -> bool:
        """Check if this is an MLP module type."""
        return self in {
            self.MLP_H_TO_4H, self.MLP_4H_TO_H, self.MLP_GATE, self.MLP_GATE_UP,
            self.MLP_ROUTER
        }

    @property
    def is_moe(self) -> bool:
        """Check if this is a Mixture of Experts (MoE) module type."""
        return self in {
            self.MOE_H_TO_4H, self.MOE_4H_TO_H, self.MOE_GATE, self.MOE_ROUTER,
            self.MOE_LATENT_FC1, self.MOE_LATENT_FC2
        }

    @property
    def is_mamba(self) -> bool:
        """Check if this is a Mamba module type."""
        return self in {self.MAMBA_IN_PROJ, self.MAMBA_OUT_PROJ}


# Canonical routed-expert MoE LoRA module set and module->kernel-slot mapping,
# shared by the MoE factory/validator (create_moe.py, validation.py), the
# adapter-layout helpers (moe_layout.py), and the CUTLASS MoE backend
# (fused_moe_cutlass.py). Keep this the single source of truth.
#
# Slot convention (see loraFC1 / doActivationKernel in moe_kernels.cu): the
# kernel applies the "fc1" LoRA to the gate (SiLU) half of the packed FC1
# output and the "gated" LoRA to the up (linear) half. With the canonical
# convention (moe_h_to_4h = w1 gate/SiLU, moe_gate = w3 up/linear,
# moe_4h_to_h = w2 down), this maps moe_h_to_4h->fc1, moe_gate->gated,
# moe_4h_to_h->fc2.
MOE_LORA_MODULE_NAMES = ("moe_h_to_4h", "moe_4h_to_h", "moe_gate")
MOE_LORA_MODULE_TO_KERNEL_SLOT = {
    LoraModuleType.MOE_H_TO_4H: "fc1",
    LoraModuleType.MOE_GATE: "gated",
    LoraModuleType.MOE_4H_TO_H: "fc2",
}


def add_lora_result(output: torch.Tensor,
                    lora_result: Optional[torch.Tensor]) -> torch.Tensor:
    if lora_result is not None:
        output.add_(lora_result.to(output.dtype))
    return output


class LoraLayer(torch.nn.Module):

    def __init__(self, lora_module_types: List[LoraModuleType],
                 output_hidden_sizes: List[int]):
        super().__init__()

        self.lora_module_types = lora_module_types
        self.output_hidden_sizes = output_hidden_sizes
        assert len(lora_module_types) == len(output_hidden_sizes)

        self._par_events: List[torch.cuda.Event] | None = None
        self._split_k_runner: Optional["_LoraGroupedGemmRunner"] = None

    @staticmethod
    def forward_with_base(
        base_forward: Callable[[], torch.Tensor],
        lora_layers: tuple["LoraLayer", ...],
        x: torch.Tensor,
        lora_params: dict,
        layer_idx: int | None,
    ) -> torch.Tensor:
        """
        Run the base and LoRA branches and merge their outputs.

        Args:
            base_forward: Forward call for base model projection
            lora_layers: Tuple of LoRA layers to be called
            x: Input tensor
            lora_params: CUDA Graph compatible LoRA parameters
            layer_idx: Current layer index

        Returns:
            LoRA + base model output tensor

        Note that lora_layers needs to be a tuple in order to
        handle fused/unfused modules (e.g., QKV), where both
        variants are invoked but only one runs through.
        """
        cuda_graph_params = lora_params.get('cuda_graph_params')
        has_lora_layer = bool(cuda_graph_params) and any(
            CudaGraphLoraParams.LoraLayerKey(
                layer_idx=layer_idx,
                module_ids=tuple(layer.lora_module_types),
            ) in cuda_graph_params.layer_info for layer in lora_layers)

        lora_aux_stream = lora_params.get("lora_aux_stream")
        execute_in_parallel = (has_lora_layer and lora_aux_stream is not None
                               and do_multi_stream()
                               and not torch.compiler.is_compiling())

        # Pack all LoRA forwards (e.g., fused/unfused) in a single tuple
        def lora_forward() -> tuple[torch.Tensor | None, ...]:
            return tuple(
                lora_layer(x, lora_params, layer_idx)
                for lora_layer in lora_layers)

        if execute_in_parallel:
            assert lora_aux_stream is not None
            # Lazy allocation of parallel events
            if lora_layers[0]._par_events is None:
                lora_layers[0]._par_events = [
                    torch.cuda.Event(), torch.cuda.Event()
                ]

            base_output, lora_outputs = maybe_execute_in_parallel(
                base_forward,
                lora_forward,
                lora_layers[0]._par_events[0],
                lora_layers[0]._par_events[1],
                lora_aux_stream,
                disable_on_compile=True,
            )
        else:
            base_output, lora_outputs = base_forward(), lora_forward()

        for lora_output in lora_outputs:
            if not isinstance(lora_output, torch.Tensor):
                continue
            if execute_in_parallel:
                lora_output.record_stream(torch.cuda.current_stream())
            base_output = add_lora_result(base_output, lora_output)

        return base_output

    def forward(
        self,
        x,
        lora_params: Dict,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:

        if not bool(lora_params):
            return None

        input_dtype = x.dtype
        data_type = lora_params.get("data_type")
        if data_type is not None and input_dtype != data_type:
            if data_type == torch.float8_e4m3fn and input_dtype in (
                    torch.float16, torch.bfloat16, torch.float32):
                fp8_max = torch.finfo(data_type).max
                x = x.clamp(min=-fp8_max, max=fp8_max).to(data_type)
            else:
                raise TypeError(
                    f"LoRA input dtype {input_dtype} must match PEFT cache dtype "
                    f"{data_type}.")

        use_cuda_graph_mode = lora_params.get("use_cuda_graph_mode", False)
        if use_cuda_graph_mode:
            result = self._forward_cuda_graph_mode(x, lora_params, layer_idx)
        else:
            result = self._forward_eager_mode(x, lora_params, layer_idx)

        if isinstance(result, torch.Tensor) and result.dtype != input_dtype:
            result = result.to(input_dtype)
        return result

    def prepare_grouped_gemm_buffers(self, input: GroupedGemmParamsInput):
        device = input.x.device
        bs, input_hidden_size = input.x.shape
        shape_2d = (len(self.lora_module_types), input.max_lora_size
                    )  # [num_layer_modules, max_lora_size]
        shape_3d = shape_2d + (3, )
        sum_out_sizes = sum(self.output_hidden_sizes)

        input.output_buffer.fill_(0)
        input.intermediate_buffer.fill_(0)

        # reorder input
        reordered_input = torch.index_select(input.x, 0, input.sorted_ids[:bs])

        # a [bs, hidden]
        lda = torch.full(shape_2d,
                         input_hidden_size,
                         dtype=CudaGraphLoraParams.LD_DTYPE,
                         device=device)

        # b [input_hidden_size, lora_rank]
        ldb = lda

        # a_prime / d [num_layer_modules, bs, max_rank]
        ldd = torch.full(shape_2d,
                         input.max_rank,
                         dtype=CudaGraphLoraParams.LD_DTYPE,
                         device=device)

        # b_prime [lora_rank, module_output_size]
        ldb_prime = input.slot_ranks.unsqueeze(0).to(
            dtype=CudaGraphLoraParams.LD_DTYPE).repeat(shape_2d[0], 1)

        # d_prime [bs, sum_of_each_module_output_sizes]
        ldd_prime = torch.full(shape_2d,
                               sum_out_sizes,
                               dtype=CudaGraphLoraParams.LD_DTYPE,
                               device=device)

        # reordered a [bs, hidden], each module has the same offset
        a_offset = input.slot_offsets * input_hidden_size
        a_offset = a_offset.unsqueeze(0).repeat(shape_2d[0], 1)

        # d [num_layer_modules, bs, max_rank]
        d_offset = (input.slot_offsets.unsqueeze(0) + torch.arange(
            shape_2d[0], device=device, dtype=CudaGraphLoraParams.PTR_DTYPE).
                    unsqueeze(1) * bs) * input.max_rank

        # d' [bs, sum_of_each_module_output_sizes]
        bs_offset = input.slot_offsets.unsqueeze(0)  # [1, max_lora_size]
        bs_offset = bs_offset * sum_out_sizes
        out_offset = input.output_sizes_offset.unsqueeze(
            1)  # [num_layer_modules, 1]
        d_prime_offset = bs_offset + out_offset

        # sizes
        in_sizes = torch.empty(shape_3d,
                               dtype=CudaGraphLoraParams.SIZES_DTYPE,
                               device=device)
        out_sizes = torch.empty_like(in_sizes)

        slot_counts = input.slot_counts.unsqueeze(0)  # [1, max_lora_size]
        ranks = input.slot_ranks.unsqueeze(0)  # [1, max_lora_size]
        output_hidden_sizes = input.output_hidden_sizes.unsqueeze(
            1)  # [num_layer_modules, 1]

        in_sizes[:, :, 0] = slot_counts
        in_sizes[:, :, 1] = ranks
        in_sizes[:, :, 2] = input_hidden_size

        out_sizes[:, :, 0] = slot_counts
        out_sizes[:, :, 1] = output_hidden_sizes
        out_sizes[:, :, 2] = ranks

        # disable unused modules / lora with ptr being zeros
        in_sizes *= (input.b_ptrs != 0).unsqueeze(-1)
        out_sizes *= (input.b_prime_ptrs != 0).unsqueeze(-1)

        # splitk_offsets: [num_layer_modules, max_lora_size]
        # splitk offtsets (m * n) for the first grouped gemm with (m, n, k) = (slot_counts, slot_ranks, input_hidden_size)
        splitk_offsets = torch.zeros(shape_2d,
                                     dtype=CudaGraphLoraParams.LD_DTYPE,
                                     device=device)

        splitk_offsets.view(-1)[1:] = in_sizes.view(-1, 3)[:-1, 0]  #  = M
        splitk_offsets.view(-1)[1:] *= in_sizes.view(-1, 3)[:-1, 1]  # *= N
        splitk_offsets.view(-1).cumsum_(dim=0)

        # add base addresses to offset tensors on GPU
        dtype_element_size = input.x.element_size()
        a_offset *= dtype_element_size
        a_offset += reordered_input.data_ptr()

        d_offset *= dtype_element_size
        d_offset += input.intermediate_buffer.data_ptr()

        d_prime_offset *= dtype_element_size
        d_prime_offset += input.output_buffer.data_ptr()

        return GroupedGemmParamsOutput(in_sizes=in_sizes,
                                       out_sizes=out_sizes,
                                       a_offset=a_offset,
                                       d_offset=d_offset,
                                       d_prime_offset=d_prime_offset,
                                       lda=lda,
                                       ldb=ldb,
                                       ldd=ldd,
                                       ldb_prime=ldb_prime,
                                       ldd_prime=ldd_prime,
                                       splitk_offsets=splitk_offsets,
                                       reordered_input=reordered_input)

    def _prepare_grouped_gemm_buffers_fused(self,
                                            input: GroupedGemmParamsInput):
        device = input.x.device
        bs, input_hidden_size = input.x.shape
        shape_2d = (len(self.lora_module_types), input.max_lora_size
                    )  # [num_layer_modules, max_lora_size]
        shape_3d = shape_2d + (3, )
        sum_out_sizes = sum(self.output_hidden_sizes)

        in_sizes = torch.empty(shape_3d,
                               dtype=CudaGraphLoraParams.SIZES_DTYPE,
                               device=device)
        out_sizes = torch.empty_like(in_sizes)
        a_offset = torch.empty(shape_2d,
                               dtype=CudaGraphLoraParams.PTR_DTYPE,
                               device=device)
        d_offset = torch.empty_like(a_offset)
        d_prime_offset = torch.empty_like(a_offset)
        lda = torch.empty(shape_2d,
                          dtype=CudaGraphLoraParams.LD_DTYPE,
                          device=device)
        ldb = lda
        ldd = torch.empty_like(lda)
        ldb_prime = torch.empty_like(lda)
        ldd_prime = torch.empty_like(lda)
        splitk_offsets = torch.empty(shape_2d,
                                     dtype=CudaGraphLoraParams.LD_DTYPE,
                                     device=device)
        reordered_input = torch.empty_like(input.x)
        torch.ops.trtllm.lora_group_gemm_param_fill_row_reorder_fusion(
            # output parameters
            in_sizes,
            out_sizes,
            a_offset,
            d_offset,
            d_prime_offset,
            lda,
            ldd,
            ldb_prime,
            ldd_prime,
            splitk_offsets,
            reordered_input,

            # input parameters
            input.max_lora_size,
            input.max_rank,
            sum_out_sizes,
            input_hidden_size,
            bs,  # batch_size
            input.slot_counts,
            input.slot_ranks,
            input.slot_offsets,
            input.output_hidden_sizes,
            input.output_sizes_offset,
            input.b_ptrs,
            input.b_prime_ptrs,
            input.x,
            input.sorted_ids[:bs],
            input.intermediate_buffer,
            input.output_buffer,
            input.x.dtype)

        return GroupedGemmParamsOutput(in_sizes=in_sizes,
                                       out_sizes=out_sizes,
                                       a_offset=a_offset,
                                       d_offset=d_offset,
                                       d_prime_offset=d_prime_offset,
                                       lda=lda,
                                       ldb=ldb,
                                       ldd=ldd,
                                       ldb_prime=ldb_prime,
                                       ldd_prime=ldd_prime,
                                       splitk_offsets=splitk_offsets,
                                       reordered_input=reordered_input)

    def _prepare_max_sizes_cpu(self, bs: int, input_hidden_size: int,
                               max_lora_size: int, max_rank: int):
        shape_2d = (len(self.lora_module_types), max_lora_size)
        shape_3d = shape_2d + (3, )
        # dummy max sizes, on CPU
        host_max_in_sizes = torch.empty(
            shape_3d, dtype=CudaGraphLoraParams.SIZES_DTYPE
        )  # m: batch_size, n: max_lora_rank, k: input_hidden_size
        host_max_out_sizes = torch.empty_like(
            host_max_in_sizes
        )  # m: batch_size, n: max_output_hidden_size, k: max_lora_rank
        host_max_in_sizes[:, :, 0] = bs
        host_max_in_sizes[:, :, 1] = max_rank
        host_max_in_sizes[:, :, 2] = input_hidden_size

        host_max_out_sizes[:, :, 0] = bs
        host_max_out_sizes[:, :, 1] = torch.tensor(
            self.output_hidden_sizes,
            dtype=CudaGraphLoraParams.SIZES_DTYPE).unsqueeze(1)
        host_max_out_sizes[:, :, 2] = max_rank

        return host_max_in_sizes, host_max_out_sizes

    def _forward_cuda_graph_mode_impl(
        self,
        inputs: List[torch.Tensor],
        max_lora_size: int,
        max_rank: int,
        problem_count: int,
        min_kn: int,
        split_k: int,
    ) -> torch.Tensor:
        """Run the complete CUDA-graph LoRA path with a fixed split-K."""
        x = inputs[0]
        batch_size, hidden_size = x.shape
        output_buffer = torch.empty(
            (batch_size, sum(self.output_hidden_sizes)),
            dtype=x.dtype,
            device=x.device,
        )
        params_input = GroupedGemmParamsInput(
            x=x,
            output_buffer=output_buffer,
            intermediate_buffer=torch.empty(
                (len(self.lora_module_types), batch_size, max_rank),
                dtype=x.dtype,
                device=x.device,
            ),
            max_lora_size=max_lora_size,
            max_rank=max_rank,
            slot_counts=inputs[1],
            slot_ranks=inputs[2],
            slot_offsets_full=inputs[3],
            b_ptrs=inputs[4],
            b_prime_ptrs=inputs[5],
            sorted_ids=inputs[6],
            output_hidden_sizes=inputs[7],
            output_sizes_offset=inputs[8],
        )
        host_max_in_sizes, host_max_out_sizes = self._prepare_max_sizes_cpu(
            batch_size,
            hidden_size,
            max_lora_size,
            max_rank,
        )
        grouped_gemm_params = self._prepare_grouped_gemm_buffers_fused(
            params_input)

        torch.ops.trtllm.lora_grouped_gemm_cuda_graph(
            grouped_gemm_params.in_sizes,
            grouped_gemm_params.out_sizes,
            grouped_gemm_params.a_offset,
            params_input.b_ptrs,
            grouped_gemm_params.d_offset,
            params_input.b_prime_ptrs,
            grouped_gemm_params.d_prime_offset,
            problem_count,
            grouped_gemm_params.lda,
            grouped_gemm_params.ldb,
            grouped_gemm_params.ldd,
            grouped_gemm_params.ldb_prime,
            grouped_gemm_params.ldd_prime,
            host_max_in_sizes,
            host_max_out_sizes,
            grouped_gemm_params.splitk_offsets,
            params_input.x.dtype,
            min_kn,
            split_k,
        )

        # PyTorch does not implement index_copy_ for FP8 tensors.
        if output_buffer.dtype == torch.float8_e4m3fn:
            output_buffer = output_buffer.to(torch.bfloat16)

        restored_output = torch.empty_like(output_buffer)
        restored_output.index_copy_(
            0,
            params_input.sorted_ids[:batch_size],
            output_buffer,
        )
        return restored_output

    def _forward_cuda_graph_mode(
        self,
        x: torch.Tensor,
        lora_params: Dict,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Forward pass using CUDA Graph compatible LoRA parameters.

        Args:
            x: Input tensor
            lora_params: CUDA Graph compatible LoRA parameters
            layer_idx: Current layer index

        Returns:
            LoRA output tensor or None
        """

        cuda_graph_params: CudaGraphLoraParams = lora_params.get(
            'cuda_graph_params')
        # Get layer-specific parameters
        layer_key = CudaGraphLoraParams.LoraLayerKey(
            layer_idx=layer_idx, module_ids=tuple(self.lora_module_types))

        if not cuda_graph_params or not cuda_graph_params.layer_info or layer_key not in cuda_graph_params.layer_info:
            return None

        layer_params = cuda_graph_params.get_layer_params(layer_key)

        # Skip layers that don't have LoRA modules
        if layer_params is None:
            return None  # Pass-through for layers without LoRA modules

        _, hidden_size = x.shape
        max_rank = cuda_graph_params.max_rank
        if x.dtype == torch.float8_e4m3fn:
            min_kn = _validate_fp8_lora_cuda_graph_alignment(
                cuda_graph_params.slot_ranks_host, hidden_size,
                self.output_hidden_sizes, max_rank)
        else:
            min_kn = min(
                hidden_size, 8, max_rank
            )  # TODO: hardcode to 8 for now, for alignments in kernels, might have alignment error if rank is less than 8!

        problem_count = cuda_graph_params.get_problem_count(layer_key)
        if self._split_k_runner is None:
            self._split_k_runner = _LoraGroupedGemmRunner(
                layer=self,
                layer_idx=layer_idx,
                input_hidden_size=hidden_size,
                max_rank=max_rank,
                max_lora_size=cuda_graph_params.max_lora_size,
                problem_count=problem_count,
                dtype=x.dtype,
                min_kn=min_kn,
            )
        runner = self._split_k_runner
        runner.min_kn = min_kn
        runner_inputs = [
            x,
            cuda_graph_params.slot_counts,
            cuda_graph_params.slot_ranks,
            cuda_graph_params.slot_offsets_full,
            layer_params.d_b_ptrs,
            layer_params.d_b_prime_ptrs,
            cuda_graph_params.sorted_ids,
            layer_params.d_output_sizes,
            layer_params.d_output_sizes_offset,
        ]
        _, split_k = AutoTuner.get().choose_one(
            "trtllm::lora_grouped_gemm_cuda_graph",
            [runner],
            runner.tuning_config,
            runner_inputs,
        )
        return runner(runner_inputs, tactic=split_k)

    def _forward_eager_mode(
        self,
        x: torch.Tensor,
        lora_params: Dict,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Eager-mode forward pass using the original LoRA implementation.

        Args:
            x: Input tensor
            lora_params: LoRA parameters for eager mode
            layer_idx: Current layer index

        Returns:
            LoRA output tensor or None
        """
        lora_ranks = []
        lora_weight_pointers = []
        active_lora_module_ids = []

        # Check if this layer has any LoRA weights
        layer_params = lora_params.get(layer_idx, {})

        for module_idx in self.lora_module_types:
            module_idx = int(module_idx)
            if module_idx in layer_params:
                active_lora_module_ids.append(module_idx)
                lora_ranks.append(layer_params[module_idx]['adapter_size'])
                lora_weight_pointers.append(
                    layer_params[module_idx]['weight_pointers'])

        num_seqs = lora_params['num_seqs']

        if len(active_lora_module_ids) == 0:
            return None
        else:
            lora_outputs = torch.ops.trtllm.lora_grouped_gemm(
                x,
                lora_params['host_request_types'][:num_seqs],
                lora_ranks,
                lora_weight_pointers,
                lora_params['prompt_lens_cpu'][:num_seqs],
                self.output_hidden_sizes,
                False,  # transA
                True,  # transB
                max([r.max() for r in lora_ranks]),
                0,
                True,  # TODO smor- should be lora_params["remove_input_padding"], support in loraOp as well
            )
            if isinstance(lora_outputs, torch.Tensor):
                return lora_outputs
            else:
                # For multiple LoRA modules, some might not be executed in grouped gemm.
                # For those modules not executed, we create zero tensors with matching dimensions.
                # Finally we concatenate all tensors (both LoRA outputs and zero tensors) in order.
                lora_output = []
                for module_idx in self.lora_module_types:
                    if int(module_idx) in active_lora_module_ids:
                        lora_output.append(lora_outputs.pop(0))
                    else:
                        lora_output.append(
                            torch.zeros(list(x.shape[:-1]) + [
                                self.output_hidden_sizes[
                                    self.lora_module_types.index(module_idx)]
                            ],
                                        dtype=x.dtype,
                                        device=x.device))
                lora_output = torch.cat(lora_output, dim=-1)
                return lora_output


class _LoraGroupedGemmRunner(TunableRunner):
    """Tune split-K for one logical LoRA layer and token-count bucket."""

    def __init__(
        self,
        layer: LoraLayer,
        layer_idx: int,
        input_hidden_size: int,
        max_rank: int,
        max_lora_size: int,
        problem_count: int,
        dtype: torch.dtype,
        min_kn: int,
    ):
        self.layer = layer
        self.layer_idx = layer_idx
        self.input_hidden_size = input_hidden_size
        self.max_rank = max_rank
        self.max_lora_size = max_lora_size
        self.problem_count = problem_count
        self.dtype = dtype
        self.min_kn = min_kn
        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(DynamicTensorSpec(
                0,
                0,
                get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2,
            ), ),
            inputs_pre_hook=self._prepare_synthetic_inputs,
        )

    def unique_id(self):
        return (
            self.layer_idx,
            tuple(
                int(module_type)
                for module_type in self.layer.lora_module_types),
            tuple(self.layer.output_hidden_sizes),
            self.input_hidden_size,
            self.max_rank,
            self.max_lora_size,
            self.problem_count,
            self.dtype,
            self.min_kn,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[int]:
        del inputs, profile, kwargs
        k_tiles = max(1, self.input_hidden_size // 64)
        return [
            split_k for split_k in _LORA_SPLIT_K_CANDIDATES
            if split_k <= k_tiles
        ]

    def _prepare_synthetic_inputs(
            self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Build one active-adapter problem for the requested token bucket."""
        token_carrier = inputs[0]
        num_tokens = token_carrier.shape[0]
        device = token_carrier.device
        module_count = len(self.layer.lora_module_types)
        shape_2d = (module_count, self.max_lora_size)

        b_ptrs = torch.zeros(shape_2d,
                             dtype=CudaGraphLoraParams.PTR_DTYPE,
                             device=device)
        b_prime_ptrs = torch.zeros_like(b_ptrs)
        keepalive = []
        for module_idx, output_size in enumerate(
                self.layer.output_hidden_sizes):
            lora_a = torch.ones((self.max_rank, self.input_hidden_size),
                                dtype=self.dtype,
                                device=device)
            lora_b = torch.ones((output_size, self.max_rank),
                                dtype=self.dtype,
                                device=device)
            b_ptrs[module_idx, 0] = lora_a.data_ptr()
            b_prime_ptrs[module_idx, 0] = lora_b.data_ptr()
            keepalive.extend((lora_a, lora_b))

        slot_counts = torch.zeros(self.max_lora_size,
                                  dtype=CudaGraphLoraParams.SIZES_DTYPE,
                                  device=device)
        slot_counts[0] = num_tokens
        slot_ranks = torch.zeros_like(slot_counts)
        slot_ranks[0] = self.max_rank
        slot_offsets_full = torch.zeros(self.max_lora_size + 1,
                                        dtype=CudaGraphLoraParams.PTR_DTYPE,
                                        device=device)
        slot_offsets_full[1:] = num_tokens

        output_hidden_sizes = torch.tensor(
            self.layer.output_hidden_sizes,
            dtype=CudaGraphLoraParams.SIZES_DTYPE,
            device=device)
        output_sizes_offset = CudaGraphLoraParams.get_offset_from_counts(
            output_hidden_sizes).to(dtype=CudaGraphLoraParams.PTR_DTYPE)

        return [
            token_carrier,
            slot_counts,
            slot_ranks,
            slot_offsets_full,
            b_ptrs,
            b_prime_ptrs,
            torch.arange(num_tokens, dtype=torch.int64, device=device),
            output_hidden_sizes,
            output_sizes_offset,
        ] + keepalive

    def forward(
        self,
        /,
        inputs: List[torch.Tensor],
        *,
        tactic: int = -1,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        split_k = TRTLLM_SPLITK_VAL if tactic == -1 else tactic
        return self.layer._forward_cuda_graph_mode_impl(
            inputs,
            self.max_lora_size,
            self.max_rank,
            self.problem_count,
            self.min_kn,
            split_k,
        )


class MoeLoraLayer(LoraLayer):
    """Marker LoraLayer for routed-expert MoE modules.

    Routed-expert LoRA is *fused* into the MoE kernel
    (torch.ops.trtllm.fused_moe with LoRA kwargs); the actual GEMMs are not
    performed by LoraLayer.forward. This subclass exists so that
    CudaGraphLoraManager._initialize_from_model and the LoRA target-module
    validator can discover MoE LoRA layers via isinstance(module, LoraLayer)
    and inspect their lora_module_types / output_hidden_sizes, without
    altering the standalone LoraLayer call semantics elsewhere in the model.

    Calling forward() directly is a programming error: the MoE module owns
    LoRA application.
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MoeLoraLayer is a discovery marker; routed-expert LoRA is applied "
            "inside torch.ops.trtllm.fused_moe via the MoE module. Do not call "
            "MoeLoraLayer.forward directly.")
