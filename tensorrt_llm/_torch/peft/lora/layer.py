from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional

import torch

from .cuda_graph_lora_params import CudaGraphLoraParams


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
            self.MOE_H_TO_4H, self.MOE_4H_TO_H, self.MOE_GATE, self.MOE_ROUTER
        }


class LoraLayer(torch.nn.Module):

    def __init__(self, lora_module_types: List[LoraModuleType],
                 output_hidden_sizes: List[int]):
        super().__init__()

        self.lora_module_types = lora_module_types
        self.output_hidden_sizes = output_hidden_sizes
        assert len(lora_module_types) == len(output_hidden_sizes)

    def forward(
        self,
        x,
        lora_params: Dict,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:

        if bool(lora_params):
            # Check if we're using CUDA Graph mode
            use_cuda_graph_mode = lora_params.get('use_cuda_graph_mode', False)

            if use_cuda_graph_mode:
                return self._forward_cuda_graph_mode(x, lora_params, layer_idx)
            else:
                return self._forward_eager_mode(x, lora_params, layer_idx)
        else:
            return None

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

    def _prepare_max_sizes_cpu(self,
                               cuda_graph_lora_params: CudaGraphLoraParams,
                               layer_key: CudaGraphLoraParams.LoraLayerKey,
                               bs: int, input_hidden_size: int):
        layer_params = cuda_graph_lora_params.get_layer_params(layer_key)
        shape_2d = (len(self.lora_module_types),
                    cuda_graph_lora_params.max_lora_size
                    )  # [num_layer_modules, max_lora_size]
        shape_3d = shape_2d + (3, )
        # dummy max sizes, on CPU
        host_max_in_sizes = torch.empty(
            shape_3d, dtype=CudaGraphLoraParams.SIZES_DTYPE
        )  # m: batch_size, n: max_lora_rank, k: input_hidden_size
        host_max_out_sizes = torch.empty_like(
            host_max_in_sizes
        )  # m: batch_size, n: max_output_hidden_size, k: max_lora_rank
        host_max_in_sizes[:, :, 0] = bs
        host_max_in_sizes[:, :, 1] = cuda_graph_lora_params.max_rank
        host_max_in_sizes[:, :, 2] = input_hidden_size

        host_max_out_sizes[:, :, 0] = bs
        host_max_out_sizes[:, :, 1] = layer_params.h_output_sizes.unsqueeze(1)
        host_max_out_sizes[:, :, 2] = cuda_graph_lora_params.max_rank

        return host_max_in_sizes, host_max_out_sizes

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
            return 0  # Pass-through for layers without LoRA modules

        batch_size, hidden_size = x.shape[0], x.shape[-1]
        num_layer_modules = len(self.lora_module_types)
        max_rank = cuda_graph_params.max_rank
        total_output_size = sum(self.output_hidden_sizes)
        min_kn = min(
            hidden_size, 8, max_rank
        )  # TODO: hardcode to 8 for now, for alignments in kernels, might have alignment error if rank is less than 8!

        output_buffer = torch.empty(batch_size,
                                    total_output_size,
                                    dtype=x.dtype,
                                    device=x.device)

        host_max_in_sizes, host_max_out_sizes = self._prepare_max_sizes_cpu(
            cuda_graph_params, layer_key, batch_size, hidden_size)

        # Intermediate buffer: [num_layer_modules, batch_size, max_rank]
        intermediate_buffer = torch.empty(
            [num_layer_modules, batch_size, max_rank],
            dtype=x.dtype,
            device=x.device)

        params_fill_input = GroupedGemmParamsInput(
            x=x,
            output_buffer=output_buffer,
            intermediate_buffer=intermediate_buffer,
            max_lora_size=cuda_graph_params.max_lora_size,
            max_rank=cuda_graph_params.max_rank,
            slot_counts=cuda_graph_params.slot_counts,
            slot_ranks=cuda_graph_params.slot_ranks,
            slot_offsets_full=cuda_graph_params.slot_offsets_full,
            b_ptrs=layer_params.d_b_ptrs,
            b_prime_ptrs=layer_params.d_b_prime_ptrs,
            sorted_ids=cuda_graph_params.sorted_ids,
            output_hidden_sizes=layer_params.d_output_sizes,
            output_sizes_offset=layer_params.d_output_sizes_offset)
        grouped_gemm_params = self._prepare_grouped_gemm_buffers_fused(
            params_fill_input)

        torch.ops.trtllm.lora_grouped_gemm_cuda_graph(
            grouped_gemm_params.in_sizes, grouped_gemm_params.out_sizes,
            grouped_gemm_params.a_offset, layer_params.d_b_ptrs,
            grouped_gemm_params.d_offset, layer_params.d_b_prime_ptrs,
            grouped_gemm_params.d_prime_offset,
            cuda_graph_params.get_problem_count(layer_key),
            grouped_gemm_params.lda, grouped_gemm_params.ldb,
            grouped_gemm_params.ldd, grouped_gemm_params.ldb_prime,
            grouped_gemm_params.ldd_prime, host_max_in_sizes,
            host_max_out_sizes, grouped_gemm_params.splitk_offsets,
            grouped_gemm_params.reordered_input.dtype, min_kn)

        # TODO: move to kernel
        restored_output = torch.zeros_like(output_buffer)
        restored_output.index_copy_(0,
                                    cuda_graph_params.sorted_ids[:batch_size],
                                    output_buffer)
        return restored_output

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

        for module_idx in self.lora_module_types:
            module_idx = int(module_idx)
            if module_idx in lora_params[layer_idx]:
                active_lora_module_ids.append(module_idx)
                lora_ranks.append(
                    lora_params[layer_idx][module_idx]['adapter_size'])
                lora_weight_pointers.append(
                    lora_params[layer_idx][module_idx]['weight_pointers'])

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
