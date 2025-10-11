from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Dict, List, Optional

import torch

from .cuda_graph_lora_params import CudaGraphLoraParams


class DelayedAssert:

    def __init__(self, store_stack: bool = False):
        self.assertions = []
        self.store_stack = store_stack

    def add(self, result: bool, msg: str):
        import traceback
        self.assertions.append(
            (bool(result), str(msg), traceback.format_stack()))

    def get_msg(self):
        ret = ['Some assertions failed:']
        for result, msg, stack in self.assertions:
            ret.append('\n'.join([
                f'Assert result: {result}', msg,
                ''.join(stack) if self.store_stack else ''
            ]))
        ret = '\n-----------------------------------------\n'.join(ret)
        ret = 'Some assertions failed:\n' + ret
        return ret

    def clear(self):
        self.assertions.clear()

    def assert_all(self):
        assert all(ret[0] for ret in self.assertions), self.get_msg()
        self.clear()


# TODO: remove
TEST_GEMM = False
PRINT_AND_ASSERT = False

GATHER = True
PARAM_PREP = True
GROUPED_GEMM = True
SCATTER = True
FILL_OUTPUT_0 = False
RETURN_0_DIRECTLY = False
RETURN_NONE_DIRECTLY = False
COMPARE_WITH_PY = False


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

    @property
    def slot_offsets(self):
        return self.slot_offsets_full[:-1]


def compare_grouped_gemm_params(
    params: GroupedGemmParamsOutput,
    ref: GroupedGemmParamsOutput,
    params_input: GroupedGemmParamsInput,
    params_to_store_msg: List[str] | None = ['splitk_offsets'],
    params_exclude_msg: List[str] | None = None,
):
    assert not (params_to_store_msg and params_exclude_msg)

    bs, input_hidden_size = params.reordered_input.shape
    asserter = DelayedAssert()
    params_dict = asdict(params)
    ref_dict = asdict(ref)

    if not params_to_store_msg:
        params_to_store_msg = set(params_dict.keys())
    if params_exclude_msg:
        for name in params_exclude_msg:
            params_to_store_msg.discard(name)

    def get_msg(name: str, v: torch.Tensor, ref_v: torch.Tensor):
        is_get_msg = any(p in name or name in p for p in params_to_store_msg)
        header = f"\n\n{name=}\n"
        return f"{header} {v=}\n {ref_v=}\n diff:\n{v - ref_v}" if is_get_msg else header

    for name in params_dict.keys():
        v = params_dict[name]
        ref_v = ref_dict[name]
        if name not in ("reordered_input", "a_offset"):
            asserter.add(
                v.allclose(ref_v),
                get_msg(name, v, ref_v),
            )

    # Test a_offset separately
    offset = params.a_offset - params.reordered_input.data_ptr()
    ref_offset = ref.a_offset - ref.reordered_input.data_ptr()
    asserter.add(
        (offset == ref_offset).all(),
        # 'a_offset_fused',
        get_msg("a_offset", offset, ref_offset))

    # Test reordered_input separately
    valid_row = params_input.slot_offsets_full[-1].cpu().item()
    valid_rows = params.reordered_input[:valid_row]
    ref_valid_rows = ref.reordered_input[:valid_row]
    asserter.add(
        valid_rows.allclose(ref_valid_rows),
        get_msg(f"valid part({valid_row=}, {bs=}) of reordered_input",
                valid_rows, ref_valid_rows))

    # check intermediate buffer and output buffer are all zeros
    asserter.add(
        torch.all(params_input.intermediate_buffer == 0),
        get_msg("intermediate buffer", params_input.intermediate_buffer, 0))
    asserter.add(torch.all(params_input.output_buffer == 0),
                 get_msg("output buffer", params_input.output_buffer, 0))

    if valid_row < bs:
        invalid_rows = params.reordered_input[valid_row:]
        ref_invalid_rows = ref.reordered_input[valid_row:]
        asserter.add(
            torch.all(invalid_rows == 0),
            get_msg("invalid part of reordered_input", invalid_rows,
                    ref_invalid_rows))
    else:
        asserter.add(
            True,
            f"valid_row is full {valid_row=} v. bs: {params_dict['reordered_input'].shape[0]=}"
        )
    asserter.assert_all()
    '''
    print(asserter.get_msg())
    asserter.clear()
    '''


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
    PTR_DTYPE = torch.int64
    LD_DTYPE = torch.int64
    SIZES_DTYPE = torch.int32

    def __init__(self, lora_module_types: List[LoraModuleType],
                 output_hidden_sizes: List[int]):
        super().__init__()

        self.lora_module_types = lora_module_types
        self.output_hidden_sizes = torch.tensor(output_hidden_sizes,
                                                dtype=self.SIZES_DTYPE)
        self.output_hidden_sizes_list = output_hidden_sizes
        assert len(lora_module_types) == len(output_hidden_sizes)
        self.output_sizes_offset = CudaGraphLoraParams.get_offset_from_counts(
            self.output_hidden_sizes).to(
                dtype=self.PTR_DTYPE)  # [num_layer_modules]
        if PARAM_PREP:
            self.output_sizes_offset_device = self.output_sizes_offset.to(
                device='cuda')
            self.output_hidden_size_device = self.output_hidden_sizes.to(
                device='cuda')

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
                return self._forward_legacy_mode(x, lora_params, layer_idx)
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
        if GATHER:
            # reordered_input = x[cuda_graph_params.sorted_ids[:batch_size]].contiguous()
            reordered_input = torch.index_select(input.x, 0,
                                                 input.sorted_ids[:bs])
            # reordered_input = torch.index_select(x, 0, sorted_indices)
            # reordered_input = torch.gather(
            #     x, 0, cuda_graph_params.sorted_ids[:batch_size].unsqueeze(1).expand_as(x).contiguous())
        else:
            reordered_input = input.x

        # a [bs, hidden]
        lda = torch.full(shape_2d,
                         input_hidden_size,
                         dtype=self.LD_DTYPE,
                         device=device)

        # b [input_hidden_size, lora_rank]
        ldb = lda

        # a_prime / d [num_layer_modules, bs, max_rank]
        ldd = torch.full(shape_2d,
                         input.max_rank,
                         dtype=self.LD_DTYPE,
                         device=device)

        # b_prime [lora_rank, module_output_size]
        ldb_prime = input.slot_ranks.unsqueeze(0).to(
            dtype=self.LD_DTYPE).repeat(shape_2d[0], 1)

        # d_prime [bs, sum_of_each_module_output_sizes]
        ldd_prime = torch.full(shape_2d,
                               sum_out_sizes,
                               dtype=self.LD_DTYPE,
                               device=device)

        # reordered a [bs, hidden], each module has the same offset
        a_offset = input.slot_offsets * input_hidden_size
        a_offset = a_offset.unsqueeze(0).repeat(shape_2d[0], 1)

        # d [num_layer_modules, bs, max_rank]
        d_offset = (input.slot_offsets.unsqueeze(0) + torch.arange(
            shape_2d[0], device=device, dtype=self.PTR_DTYPE).unsqueeze(1) *
                    bs) * input.max_rank

        # d' [bs, sum_of_each_module_output_sizes]
        bs_offset = input.slot_offsets.unsqueeze(0)  # [1, max_lora_size]
        bs_offset = bs_offset * sum_out_sizes
        out_offset = self.output_sizes_offset_device.unsqueeze(
            1)  # [num_layer_modules, 1]
        d_prime_offset = bs_offset + out_offset
        '''
        # change to another mem layout
        d_prime_offset = bs_offset * self.output_hidden_size_device.unsqueeze(1)  # [1, max_lora_size] * [num_layer_modules, 1]
        '''

        # sizes
        in_sizes = torch.empty(shape_3d, dtype=self.SIZES_DTYPE, device=device)
        out_sizes = torch.empty_like(in_sizes)

        slot_counts = input.slot_counts.unsqueeze(0)  # [1, max_lora_size]
        ranks = input.slot_ranks.unsqueeze(0)  # [1, max_lora_size]
        output_hidden_sizes = self.output_hidden_size_device.unsqueeze(
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
                                     dtype=self.LD_DTYPE,
                                     device=device)  # (layer_problem_count,)

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

        in_sizes = torch.empty(shape_3d, dtype=self.SIZES_DTYPE, device=device)
        out_sizes = torch.empty_like(in_sizes)
        a_offset = torch.empty(shape_2d, dtype=self.PTR_DTYPE, device=device)
        d_offset = torch.empty_like(a_offset)
        d_prime_offset = torch.empty_like(a_offset)
        lda = torch.empty(shape_2d, dtype=self.LD_DTYPE, device=device)
        ldb = lda
        ldd = torch.empty_like(lda)
        ldb_prime = torch.empty_like(lda)
        ldd_prime = torch.empty_like(lda)
        splitk_offsets = torch.empty(shape_2d,
                                     dtype=self.LD_DTYPE,
                                     device=device)  # (layer_problem_count,)
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
            self.output_hidden_size_device,
            self.output_sizes_offset_device,
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
                               bs: int, input_hidden_size: int):
        shape_2d = (len(self.lora_module_types),
                    cuda_graph_lora_params.max_lora_size
                    )  # [num_layer_modules, max_lora_size]
        shape_3d = shape_2d + (3, )
        # dummy max sizes, on CPU
        host_max_in_sizes = torch.empty(
            shape_3d, dtype=self.SIZES_DTYPE
        )  # m: batch_size, n: max_lora_rank, k: input_hidden_size
        host_max_out_sizes = torch.empty_like(
            host_max_in_sizes
        )  # m: batch_size, n: max_output_hidden_size, k: max_lora_rank
        host_max_in_sizes[:, :, 0] = bs
        host_max_in_sizes[:, :, 1] = cuda_graph_lora_params.max_rank
        host_max_in_sizes[:, :, 2] = input_hidden_size

        host_max_out_sizes[:, :, 0] = bs
        host_max_out_sizes[:, :, 1] = self.output_hidden_sizes.unsqueeze(1)
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
        if RETURN_NONE_DIRECTLY:
            return None

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
            cuda_graph_params, batch_size, hidden_size)

        if RETURN_0_DIRECTLY:
            return output_buffer

        # Intermediate buffer: [num_layer_modules, batch_size, max_rank]
        intermediate_buffer = torch.empty(
            [num_layer_modules, batch_size, max_rank],
            dtype=x.dtype,
            device=x.device)

        if PARAM_PREP:
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
                sorted_ids=cuda_graph_params.sorted_ids)
            grouped_gemm_params = self._prepare_grouped_gemm_buffers_fused(
                params_fill_input)

            dtype_element_size = x.element_size()

            # sorted_slot_ids, sorted_indices = torch.sort(cuda_graph_params.slot_ids[:batch_size], stable=True)
            '''
            temp_b_prime = torch.zeros(max(self.output_hidden_sizes_list), cuda_graph_params.max_rank * 5, dtype=x.dtype, device=x.device)
            # layer_params.d_b_prime_ptrs[1] = temp_b_prime.data_ptr()
            layer_params.d_b_prime_ptrs.fill_(temp_b_prime.data_ptr())

            # TODO tmp to pass utest
            layer_params.d_b_ptrs[layer_params.d_b_ptrs == 0] = intermediate_buffer.data_ptr()
            layer_params.d_b_prime_ptrs[layer_params.d_b_prime_ptrs == 0] = intermediate_buffer.data_ptr()
            '''

        if PRINT_AND_ASSERT:
            print(
                f'--------------------------------layer key: {layer_key}--------------------------------'
            )
            print(f'cuda graph params values:')
            print(
                f'sorted_ids (size: {cuda_graph_params.sorted_ids.shape}): {cuda_graph_params.sorted_ids[:batch_size].cpu()}'
            )
            print(
                f'slot ranks (size: {cuda_graph_params.slot_ranks.shape}): {cuda_graph_params.slot_ranks.cpu()}'
            )
            print(
                f'slot counts (size: {cuda_graph_params.slot_counts.shape}): {cuda_graph_params.slot_counts.cpu()}'
            )
            print(
                f'slot offsets (size: {cuda_graph_params.slot_offsets.shape}): {cuda_graph_params.slot_offsets.cpu()}'
            )
            print(
                f'max rank: {cuda_graph_params.max_rank}; max lora size: {cuda_graph_params.max_lora_size}; problem count: {cuda_graph_params.get_problem_count(layer_key)}'
            )

            print(f'buffers values:')
            print(
                f'input_size: {x.shape}, reordered_input_size: {reordered_input.shape}, output_buffer_size: {output_buffer.shape}, intermediate_buffer_size: {intermediate_buffer.shape}, output_buffer_shape: {output_buffer.shape}'
            )
            print(
                f'output_hidden_sizes: {self.output_hidden_sizes}; ouput_size_offset_device: {self.output_sizes_offset_device.cpu()}'
            )

            print(f'calculated buffers')

            print(
                f'in_sizes (size: {grouped_gemm_params.in_sizes.shape}): {grouped_gemm_params.in_sizes.cpu()}'
            )
            print(
                f'out_sizes (size: {grouped_gemm_params.out_sizes.shape}): {grouped_gemm_params.out_sizes.cpu()}'
            )

            print(
                f'a_offset (size: {grouped_gemm_params.a_offset.shape}): {grouped_gemm_params.a_offset.cpu()}'
            )
            print(
                f'lda (size: {grouped_gemm_params.lda.shape}): {grouped_gemm_params.lda.cpu()}'
            )

            print(
                f'layer_params.d_b_ptrs (size: {layer_params.d_b_ptrs.shape}): {layer_params.d_b_ptrs.cpu()}'
            )
            print(f'ldb (size: {ldb.shape}): {ldb.cpu()}')

            print(
                f'd_offset (size: {grouped_gemm_params.d_offset.shape}): {grouped_gemm_params.d_offset.cpu()}'
            )
            print(f'ldd (size: {ldd.shape}): {ldd.cpu()}')

            print(
                f'd_prime_offset (size: {grouped_gemm_params.d_prime_offset.shape}): {grouped_gemm_params.d_prime_offset.cpu()}'
            )
            print(
                f'ldd_prime (size: {grouped_gemm_params.ldd_prime.shape}): {grouped_gemm_params.ldd_prime.cpu()}'
            )

            print(
                f'layer_params.d_b_prime_ptrs (size: {layer_params.d_b_prime_ptrs.shape}): {layer_params.d_b_prime_ptrs.cpu()}'
            )
            print(
                f'ldb_prime (size: {grouped_gemm_params.ldb_prime.shape}): {grouped_gemm_params.ldb_prime.cpu()}'
            )

            print(f'dtype_element_size: {dtype_element_size}')
            print(
                f'splitk_offsets (size: {grouped_gemm_params.splitk_offsets.shape}): {grouped_gemm_params.splitk_offsets.cpu()}'
            )

            print('b ptrs:')
            print(
                f'layer_params.d_b_ptrs (size {layer_params.d_b_ptrs.shape}):\n{layer_params.d_b_ptrs.cpu()}\nzeros except 2nd module {(layer_params.d_b_ptrs == 0).sum() - (layer_params.d_b_ptrs[1] == 0).sum()}'
            )
            print(
                f'layer_params.d_b_prime_ptrs (size {layer_params.d_b_prime_ptrs.shape}):\n{layer_params.d_b_prime_ptrs.cpu()}\nzeros except 2nd module {(layer_params.d_b_prime_ptrs == 0).sum() - (layer_params.d_b_prime_ptrs[1] == 0).sum()}'
            )

            print(
                f'd_b_ptrs.data_ptr() % (8 * {dtype_element_size}): {layer_params.d_b_ptrs.data_ptr() % (8 * dtype_element_size)}'
            )
            print(
                f'd_b_prime_ptrs.data_ptr() % (8 * {dtype_element_size}): {layer_params.d_b_prime_ptrs.data_ptr() % (8 * dtype_element_size)}'
            )
            print(
                f'reordered_input.data_ptr() % (8 * {dtype_element_size}): {grouped_gemm_params.reordered_input.data_ptr() % (8 * dtype_element_size)}'
            )
            print(
                f'intermediate_buffer.data_ptr() % (8 * {dtype_element_size}): {intermediate_buffer.data_ptr() % (8 * dtype_element_size)}'
            )
            print(
                f'output_buffer.data_ptr() % (8 * {dtype_element_size}): {output_buffer.data_ptr() % (8 * dtype_element_size)}'
            )
        '''
        a_offset.fill_(0)
        d_prime_offset.fill_(0)
        d_offset.fill_(0)
        '''

        if PRINT_AND_ASSERT:
            assert output_buffer.is_contiguous()
            out_splitted = [
                output_buffer[:, s:s + le] for s, le in zip(
                    self.output_sizes_offset, self.output_hidden_sizes)
            ]
            # assert not any(out.is_contiguous() for out in out_splitted)
            pyt_strides = torch.tensor([out.stride(0) for out in out_splitted],
                                       dtype=self.LD_DTYPE,
                                       device=x.device)  # nModules,
            assert torch.all(
                grouped_gemm_params.ldd_prime == pyt_strides.unsqueeze(1))
            pyt_addr = torch.tensor([out.data_ptr() for out in out_splitted],
                                    dtype=self.PTR_DTYPE,
                                    device=x.device)
            assert torch.all(pyt_addr == grouped_gemm_params.d_prime_offset[:,
                                                                            0])
            print(f'pyt_strides: {pyt_strides.cpu()}')
            print(f'ldd_prime: {grouped_gemm_params.ldd_prime.cpu()}')
            print(f'pyt_addr: {pyt_addr.cpu()}')
            print(f'd_prime_ptr: {grouped_gemm_params.d_prime_offset.cpu()}')

            def assert_aligned(ptr, alignment):
                assert torch.all((ptr % alignment) == 0)

            assert_aligned(grouped_gemm_params.a_offset, 8 * dtype_element_size)
            assert_aligned(grouped_gemm_params.d_offset, 8 * dtype_element_size)
            assert_aligned(grouped_gemm_params.d_prime_offset,
                           8 * dtype_element_size)
            assert_aligned(layer_params.d_b_ptrs, 8 * dtype_element_size)
            assert_aligned(layer_params.d_b_prime_ptrs, 8 * dtype_element_size)

            assert_aligned(grouped_gemm_params.in_sizes[:, :, 1:], 8)
            assert_aligned(grouped_gemm_params.out_sizes[:, :, 1:], 8)
            assert_aligned(grouped_gemm_params.lda, 8)
            assert_aligned(grouped_gemm_params.ldb, 8)
            assert_aligned(grouped_gemm_params.ldd, 8)
            assert_aligned(grouped_gemm_params.ldb_prime, 8)
            assert_aligned(grouped_gemm_params.ldd_prime, 8)
            '''
            # fake buffers
            bs = [torch.zeros([r, hidden_size], device=x.device, dtype=x.dtype) for _ in self.output_hidden_sizes for r in cuda_graph_params.slot_ranks]
            b2s = [torch.zeros([o, r], device=x.device, dtype=x.dtype) for o in self.output_hidden_sizes for r in cuda_graph_params.slot_ranks]

            b_ptrs = torch.tensor([b.data_ptr() for b in bs], device=x.device, dtype=self.PTR_DTYPE)
            b2_ptrs = torch.tensor([b2.data_ptr() for b2 in b2s], device=x.device, dtype=self.PTR_DTYPE)
            '''
            '''
            # check sizes are in range
            # in_sizes: [num_layer_modules, max_lora_size, (slot_counts, ranks, input_hidden_size)]
            # out_sizes: [num_layer_modules, max_lora_size, (slot_counts, output_hidden_sizes, ranks)]
            a_mem = in_sizes[:, :, 0] * lda
            d_mem = in_sizes[:, :, 0] * ldd

            a_max_mem = a_mem + a_offset
            a_max_mem = a_max_mem.max().cpu()

            d_max_mem = d_mem + d_offset
            d_max_mem = d_max_mem.max().cpu()

            d_prime_max_mem = d_prime_offset + (out_sizes - 1) * ldd_prime + out_sizes[:, :, 1]
            d_prime_max_mem = d_prime_max_mem.max().cpu()

            ret1 = torch.all(a_max_mem > x.nelement()).item()
            wrong_a = torch.gather(x, 0, cuda_graph_params.sorted_ids[:batch_size].unsqueeze(1).expand([-1, x.shape[1]]))
            retx = torch.all(a_max_mem > wrong_a.nelement()).item()
            ret2 = torch.all(d_max_mem > intermediate_buffer.nelement()).item()
            ret3 = torch.all(d_prime_max_mem > output_buffer.nelement()).item()

            msg = []
            if ret1:
                msg.append(f"a out of range: max accessed mem: {a_max_mem.item()}, total mem: {x.nelement()}")
            if retx:
                msg.append(f"wrong a out of range: max accessed mem: {a_max_mem.item()}, total mem: {wrong_a.nelement()}; wrong_a.shape: {wrong_a.shape}; sorted_ids: {cuda_graph_params.sorted_ids.cpu()}; batch_size: {batch_size}")
            if ret2:
                msg.append(f"d out of range: max accessed mem: {d_max_mem.item()}, total mem: {intermediate_buffer.nelement()}")
            if ret3:
                msg.append(f"d_prime out of range: max accessed mem: {d_prime_max_mem.item()}, total mem: {output_buffer.nelement()}")
            if torch.any(layer_params.d_b_ptrs == 0):
                msg.append(f"d_b_ptrs has zeros, {layer_params.d_b_ptrs.cpu()}")
            if torch.any(layer_params.d_b_prime_ptrs == 0):
                msg.append(f"d_b_prime_ptrs has zeros, {layer_params.d_b_prime_ptrs.cpu()}")

            if msg:
                msg.append(f"in_sizes: {in_sizes.cpu()}\nout_sizes: {out_sizes.cpu()}\nlda: {lda.cpu()}\nldb: {ldb.cpu()}\nldd: {ldd.cpu()}\nldb_prime: {ldb_prime.cpu()}\nldd_prime: {ldd_prime.cpu()}")
                msg.append(f"slot_offsets: {cuda_graph_params.slot_offsets.cpu()}\nslot_counts: {cuda_graph_params.slot_counts.cpu()}\nslot_ranks: {cuda_graph_params.slot_ranks.cpu()}\noutput_hidden_sizes: {self.output_hidden_size_device.cpu()}")
                msg.append(f"d_offset: {d_offset.cpu()}\nd_prime_offset: {d_prime_offset.cpu()}")
                msg.append(f"output_sizes_offset: {self.output_sizes_offset_device.cpu()}")
            if msg:
                msg = ['=' * 100, f'LeyerKey: {layer_key}'] + msg
                msg.append("=" * 100)
                print("\n".join(msg))
            '''
            '''
            def tall(x: torch.Tensor):
                return x.cpu().item()

            for module_idx in range(in_sizes.shape[0]):
                for lora_id in range(in_sizes.shape[1]):
                    print(f'start test for module_idx: {module_idx}, lora_id: {lora_id}')
                    nTok = in_sizes[module_idx, lora_id, 0]
                    rank = in_sizes[module_idx, lora_id, 1]
                    input_hidden_size = in_sizes[module_idx, lora_id, 2]
                    assert x.shape[1] == input_hidden_size

                    assert out_sizes[module_idx, lora_id, 0] == nTok
                    outsz = out_sizes[module_idx, lora_id, 1]
                    assert out_sizes[module_idx, lora_id, 2] == rank

                    a_start = a_offset[module_idx, lora_id]
                    a_end = a_start + lda[module_idx, lora_id] * nTok * reordered_input.element_size()
                    assert a_start >= reordered_input.data_ptr()
                    assert a_end <= reordered_input.data_ptr() + reordered_input.nelement() * reordered_input.element_size()

                    d_start = d_offset[module_idx, lora_id]
                    d_end = d_start + ldd[module_idx, lora_id] * nTok * intermediate_buffer.element_size()
                    assert d_start >= intermediate_buffer.data_ptr()
                    assert d_end <= intermediate_buffer.data_ptr() + intermediate_buffer.nelement() * intermediate_buffer.element_size()

                    d_prime_start = d_prime_offset[module_idx, lora_id]
                    d_prime_end = d_prime_start + (ldd_prime[module_idx, lora_id] * (nTok - 1) + outsz) * output_buffer.element_size()
                    assert d_prime_start >= output_buffer.data_ptr()
                    assert d_prime_end <= output_buffer.data_ptr() + output_buffer.nelement() * output_buffer.element_size()

            '''

            for ttt in (
                    reordered_input,  # Input tensor
                    in_sizes,  # GEMM sizes for lora_in
                    out_sizes,  # GEMM sizes for lora_out
                    a_offset,  # Input offsets
                    layer_params.d_b_ptrs,  # Lora_in weight pointers
                    # fake_b_ptrs,
                    # b_ptrs,
                    d_offset,  # Intermediate output offsets
                    layer_params.d_b_prime_ptrs,  # Lora_out weight pointers
                    # fake_b_prime_ptrs,
                    # b2_ptrs,
                    d_prime_offset,  # Final output offsets
                    # cuda_graph_params.slot_ids,  # Slot IDs (for reference)
                    reordered_input,
                    cuda_graph_params.
                    sorted_ids[:
                               batch_size],  # Sorted indices for gather/scatter
                    cuda_graph_params.get_problem_count(
                        layer_key),  # Number of GEMM problems
                    intermediate_buffer,  # Intermediate buffer (LoRA only)
                    output_buffer,  # Output buffer (all tokens)
                    lda,  # Leading dimensions for A matrices
                    ldb,  # Leading dimensions for B matrices
                    ldd,  # Leading dimensions for C matrices (reusing d_ld_d as placeholder)
                    ldb_prime,
                    ldd_prime,
                    host_max_in_sizes,
                    host_max_out_sizes,
                    splitk_offsets,
            ):
                if isinstance(ttt, torch.Tensor):
                    assert ttt.is_contiguous(
                    ), f'{ttt=}'.split('=')[0] + ' is not contiguous'
        '''
        # fake_b_ptrs = torch.full_like(layer_params.d_b_ptrs, intermediate_buffer.data_ptr())
        # fake_b_prime_ptrs = torch.full_like(layer_params.d_b_prime_ptrs, intermediate_buffer.data_ptr())
        # fake_b_ptrs = torch.zeros_like(layer_params.d_b_ptrs)
        # fake_b_prime_ptrs = torch.zeros_like(layer_params.d_b_prime_ptrs)
        '''

        if TEST_GEMM:
            # start constructing fake gemm buffers
            m00, n00, k00 = 7, 16, 8
            # m01, n01, k01 = 24, 7, 16
            m02, n02, k02 = 5, 40, 32
            # m21, n21, k21 = 22, 31, 12
            m10, n10, k10 = m00, 48, n00
            m12, n12, k12 = m02, 56, n02

            ld_offset = 8

            a0 = torch.eye(m00, k00 + ld_offset, dtype=x.dtype,
                           device=x.device)  # m. n. k = 24, 32, 16
            b0 = torch.eye(n00, k00 + ld_offset, dtype=x.dtype,
                           device=x.device) * 2

            a02 = torch.eye(m02,
                            k02 + ld_offset,
                            dtype=x.dtype,
                            device=x.device)  # m. n. k = 22, 31, 12
            b02 = torch.eye(
                n02, k02 + ld_offset, dtype=x.dtype, device=x.device) * 5

            d00 = torch.zeros(m00,
                              n00 + ld_offset,
                              dtype=x.dtype,
                              device=x.device)
            d01 = torch.zeros(24, 7, dtype=x.dtype, device=x.device)
            d02 = torch.zeros(m02,
                              n02 + ld_offset,
                              dtype=x.dtype,
                              device=x.device)

            b1 = torch.eye(n10, k10 + ld_offset, dtype=x.dtype,
                           device=x.device) * 3  # m. n. k = 24, 48, 32
            b12 = torch.eye(
                n12, k12 + ld_offset, dtype=x.dtype,
                device=x.device) * 7  # m. n. k = 22, 27, 31

            d10 = torch.zeros(m10,
                              n10 + ld_offset,
                              dtype=x.dtype,
                              device=x.device)
            d11 = torch.zeros(6, 43, dtype=x.dtype, device=x.device)
            d12 = torch.zeros(m12,
                              n12 + ld_offset,
                              dtype=x.dtype,
                              device=x.device)

            # problem_sizes1 = torch.tensor([[24, 32, 16], [24, 32, 16]], dtype=self.SIZES_DTYPE, device=x.device)
            problem_sizes1 = torch.tensor(
                [[m00, n00, k00], [0, 0, 0], [0, 0, 0], [m02, n02, k02]],
                dtype=self.SIZES_DTYPE,
                device=x.device)
            lda = torch.tensor([k00, 17, 16, k02],
                               dtype=self.LD_DTYPE,
                               device=x.device) + ld_offset
            ldb = torch.tensor([k00, 16, 16, k02],
                               dtype=self.LD_DTYPE,
                               device=x.device) + ld_offset
            ldd = torch.tensor([n00, 32, 32, n02],
                               dtype=self.LD_DTYPE,
                               device=x.device) + ld_offset

            problem_sizes2 = torch.tensor(
                [[m10, n10, k10], [0, 0, 0], [0, 0, 0], [m12, n12, k12]],
                dtype=self.SIZES_DTYPE,
                device=x.device)
            ldb1 = torch.tensor([k10, 32, 32, k12],
                                dtype=self.LD_DTYPE,
                                device=x.device) + ld_offset
            ldd1 = torch.tensor([n10, 48, 48, n12],
                                dtype=self.LD_DTYPE,
                                device=x.device) + ld_offset

            a0_ptr = torch.tensor(
                [a0.data_ptr(),
                 a0.data_ptr(),
                 a0.data_ptr(),
                 a02.data_ptr()],
                dtype=self.PTR_DTYPE,
                device=x.device)
            b0_ptr = torch.tensor(
                [b0.data_ptr(), 0, 0, b02.data_ptr()],
                dtype=self.PTR_DTYPE,
                device=x.device)
            d0_ptr = torch.tensor([
                d00.data_ptr(),
                d01.data_ptr(),
                d01.data_ptr(),
                d02.data_ptr()
            ],
                                  dtype=self.PTR_DTYPE,
                                  device=x.device)
            b1_ptr = torch.tensor(
                [b1.data_ptr(), 0, 0, b12.data_ptr()],
                dtype=self.PTR_DTYPE,
                device=x.device)
            d1_ptr = torch.tensor([
                d10.data_ptr(),
                d11.data_ptr(),
                d11.data_ptr(),
                d12.data_ptr()
            ],
                                  dtype=self.PTR_DTYPE,
                                  device=x.device)

            torch.ops.trtllm.lora_grouped_gemm_cuda_graph(
                problem_sizes1, problem_sizes2, a0_ptr, b0_ptr, d0_ptr, b1_ptr,
                d1_ptr, problem_sizes1.shape[0], lda, ldb, ldd, ldb1, ldd1,
                host_max_in_sizes, host_max_out_sizes, splitk_offsets, a0.dtype,
                torch.minimum(problem_sizes1[:, 1],
                              problem_sizes1[:, 2]).min().item())

            print(f'd00 (2): {d00[:10, :10]}')
            print(f'd01 (0): {d01[:10, :10]}')
            print(f'd02 (5): {d02[:10, :10]}')
            print(f'd10 (6): {d10[:10, :10]}')
            print(f'd11 (0): {d11[:10, :10]}')
            print(f'd12 (35): {d12[:10, :10]}')

            return 0
        else:
            if GROUPED_GEMM and PARAM_PREP:
                if COMPARE_WITH_PY:
                    grouped_gemm_params_py = self.prepare_grouped_gemm_buffers(
                        params_fill_input)
                    compare_grouped_gemm_params(grouped_gemm_params,
                                                grouped_gemm_params_py,
                                                params_fill_input)
                    print(
                        f"✅ {layer_key=} Fused kernel correctness test passed!")

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
            '''
            restored_output = output_buffer
            output_buffer = torch.cat(out_buffers, dim=-1)
            '''
            '''
            restored_output = torch.zeros_like(output_buffer)
            restored_output.index_copy_(
                0,
                cuda_graph_params.sorted_ids[:batch_size],
                output_buffer)
            '''
            # restored_output = output_buffer[cuda_graph_params.sorted_ids[:batch_size]].contiguous()
            # TODO: move to kernel
            restored_output = torch.zeros_like(output_buffer)
            '''
            restored_output.scatter_(
                dim=0,
                index=cuda_graph_params.sorted_ids[:batch_size].unsqueeze(
                    1).expand_as(output_buffer).contiguous(),
                src=output_buffer)
            '''
            if SCATTER:
                restored_output.index_copy_(
                    0,
                    cuda_graph_params.sorted_ids[:batch_size],
                    # sorted_indices,
                    output_buffer)

            if PRINT_AND_ASSERT:
                print(
                    f'output_buffer abs sum (size: {output_buffer.shape}): {output_buffer.abs().sum(dim=1).cpu()}'
                )
                print(
                    f'restored_output abs sum (size: {restored_output.shape}): {restored_output.abs().sum(dim=1).cpu()}'
                )

            if FILL_OUTPUT_0:
                restored_output.fill_(0)
            return restored_output

    def _forward_legacy_mode(
        self,
        x: torch.Tensor,
        lora_params: Dict,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Legacy forward pass using the original LoRA implementation.

        Args:
            x: Input tensor
            lora_params: Legacy LoRA parameters
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
                self.output_hidden_sizes_list,
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
                                self.output_hidden_sizes_list[
                                    self.lora_module_types.index(module_idx)]
                            ],
                                        dtype=x.dtype,
                                        device=x.device))
                lora_output = torch.cat(lora_output, dim=-1)
                return lora_output
