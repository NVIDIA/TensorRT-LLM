# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.parameter import Parameter

from tensorrt_llm.functional import AllReduceParams, AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

from ..distributed import allgather
from .fla.op import gdc_wait_with_memory_clobber
from .linear import Linear, TensorParallelMode


def _matmul_with_output_dtype(input: torch.Tensor, weight: torch.Tensor,
                              output_dtype: torch.dtype | None) -> torch.Tensor:
    """Compute ``input @ weight.T`` with an optional output dtype override.

    None keeps the model dtype. An FP32 override stores the result directly
    in FP32 without first rounding it to the BF16/FP16 model dtype.
    """
    out_features = weight.shape[0]
    output_shape = (*input.shape[:-1], out_features)
    input_2d = input.reshape(-1, input.shape[-1])
    mm_kwargs = {}
    if output_dtype is not None and output_dtype != weight.dtype:
        mm_kwargs["out_dtype"] = output_dtype
    output = torch.mm(input_2d, weight.t(), **mm_kwargs)
    return output.view(output_shape)


@triton.jit
def _embedding_gather_kernel(ids_ptr, w_ptr, out_ptr, num_embeddings,
                             H: tl.constexpr, BLOCK: tl.constexpr,
                             PDL: tl.constexpr):
    """out[t] = weight[ids[t]] for ids in [0, num_embeddings), zeros otherwise (flashinfer's rejection sampler
    pads with -1): the non-parallel masked lookup's values in one launch per call."""
    t = tl.program_id(0)
    if PDL:
        gdc_wait_with_memory_clobber()
    tok = tl.load(ids_ptr + t).to(tl.int64)
    ok = (tok >= 0) & (tok < num_embeddings)
    row = tl.where(ok, tok, 0)
    for h0 in tl.static_range(0, H, BLOCK):
        offs = h0 + tl.arange(0, BLOCK)
        m = offs < H
        v = tl.load(w_ptr + row * H + offs, mask=m & ok, other=0.0)
        tl.store(out_ptr + t.to(tl.int64) * H + offs, v, mask=m)
    if PDL:
        # PDL rule: release the dependents only after a CTA barrier that follows the last global access (the ids
        # may live in a block the caching allocator recycles for a dependent's pre-wait write)
        tl.debug_barrier()
        tl.extra.cuda.gdc_launch_dependents()


def embedding_gather(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """One-launch lookup of a full (non-parallel) table, bitwise the masked F.embedding path."""
    ids = input.reshape(-1)
    if not ids.is_contiguous():
        ids = ids.contiguous()
    num_embeddings, hidden = weight.shape
    out = torch.empty((ids.numel(), hidden),
                      dtype=weight.dtype,
                      device=weight.device)
    if ids.numel() > 0:
        from ..flashinfer_utils import get_env_enable_pdl
        pdl = get_env_enable_pdl()
        block = min(triton.next_power_of_2(hidden), 8192)
        _embedding_gather_kernel[(ids.numel(), )](ids,
                                                  weight,
                                                  out,
                                                  num_embeddings,
                                                  H=hidden,
                                                  BLOCK=block,
                                                  PDL=pdl,
                                                  num_warps=4,
                                                  launch_pdl=pdl)
    return out.view(*input.shape, hidden)


class LMHead(Linear):
    """LM head layer.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.
        dtype (Optional[torch.dtype]): type of the parameters.
        mapping (Optional[Mapping]): parallelism configuration.
            If not provided, the embedding is not parallelized.
        output_dtype (Optional[torch.dtype]): dtype of the logits written by
            the GEMM. None keeps the parameter dtype. torch.float32 keeps the
            weights and inputs in the parameter dtype but writes the float32
            accumulator directly, so logits are not rounded to bf16/fp16.
            Only supported for unquantized, non-ROW-parallel heads.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,
        reduce_output: bool = True,
        use_custom_cublas_mm: bool = False,
        quant_config=None,
        output_dtype: Optional[torch.dtype] = None,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    ):
        local_in_features = embedding_dim
        local_out_features = num_embeddings
        mapping = mapping or Mapping()
        self.enable_lm_head_tp_in_adp = mapping.enable_attention_dp and \
            getattr(mapping, 'enable_lm_head_tp_in_adp', False)

        tp_size = mapping.tp_size

        # Attention DP doesn't work with embedding parallelization.
        if mapping.enable_attention_dp:
            tensor_parallel_mode = None

        if tensor_parallel_mode == TensorParallelMode.ROW:
            local_in_features = math.ceil(embedding_dim / tp_size)
            self.padding_size = tp_size * local_in_features - embedding_dim
        elif tensor_parallel_mode == TensorParallelMode.COLUMN:
            local_out_features = math.ceil(num_embeddings / tp_size)
            self.padding_size = tp_size * local_out_features - num_embeddings
        else:
            self.padding_size = 0

        # Linear re-shards only the dim selected by tensor_parallel_mode, so
        # only that dim may be passed as local*tp (to carry the ceil-div
        # padding); the other dim must be the true full size — Linear keeps it
        # as-is, and quant_method.create_weights sizes the (packed) quantized
        # weight and scales from it.
        in_features_full = (local_in_features * tp_size if tensor_parallel_mode
                            == TensorParallelMode.ROW else embedding_dim)
        out_features_full = (local_out_features *
                             tp_size if tensor_parallel_mode
                             == TensorParallelMode.COLUMN else num_embeddings)
        super().__init__(
            in_features_full,
            out_features_full,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
            reduce_output=reduce_output,
            use_custom_cublas_mm=use_custom_cublas_mm,
            quant_config=quant_config,
            allreduce_strategy=allreduce_strategy,
        )

        if tensor_parallel_mode == TensorParallelMode.ROW:
            if self.tp_rank == self.tp_size - 1:
                local_in_features -= self.padding_size
        self.in_features = local_in_features
        self.out_features = local_out_features
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if self.has_any_quant:
            # Keep the quantized weights created by Linear (e.g. NVFP4 packed
            # weight + scales). The plain-Parameter override below only exists
            # to (a) drop bias and (b) shrink the ROW-mode padded shard, both
            # of which assume a dense high-precision weight.
            if self.padding_size > 0 or tensor_parallel_mode == TensorParallelMode.ROW:
                raise NotImplementedError(
                    "Quantized LMHead does not support vocab/hidden padding or "
                    "ROW tensor-parallel mode")
            if self.enable_lm_head_tp_in_adp:
                logger.error(
                    "Quantized LMHead constructed with lm_head TP in ADP: "
                    "the spec-decoding head slices the raw weight, which is "
                    "incompatible with quantized (packed) weights. The "
                    "lm_head quant entry should have been dropped upstream "
                    f"(quant_algo={quant_config.quant_algo}).")
                raise NotImplementedError(
                    "Quantized LMHead does not support lm_head TP in ADP "
                    "(spec-decoding head slices the raw weight)")
            # lm_head has historically always been bf16; make the switch to a
            # quantized head visible so unexpected accuracy/perf behavior is
            # attributable without a profiler.
            logger.info(f"LMHead is quantized: quant_algo="
                        f"{quant_config.quant_algo}, weight shape "
                        f"{tuple(self.weight.shape)} dtype {self.weight.dtype}")
        else:
            weight_shape = (self.out_features, self.in_features)
            self.weight = Parameter(torch.empty(weight_shape, dtype=dtype))
            self.register_parameter("bias", None)

        if output_dtype is not None:
            if self.has_any_quant:
                raise NotImplementedError(
                    "LMHead output_dtype is only supported for unquantized "
                    f"LM heads (quant_algo={quant_config.quant_algo})")
            if self.tp_mode == TensorParallelMode.ROW:
                raise NotImplementedError(
                    "LMHead output_dtype does not support ROW tensor-parallel "
                    "mode")
        self.output_dtype = output_dtype

    def _writes_output_dtype(self) -> bool:
        """Whether the GEMM must store a dtype different from its weights."""
        return (self.output_dtype is not None
                and self.output_dtype != self.weight.dtype)

    @property
    def vocab_size_padded(self) -> int:
        if self.tp_mode == TensorParallelMode.COLUMN and self.gather_output:
            return self.out_features * self.tp_size
        else:
            return self.out_features

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        mapping_lm_head_tp: Optional[Mapping] = None,
        is_spec_decoding_head: bool = False,
        gather_output: bool = True,
    ) -> torch.Tensor:
        """``gather_output=False`` (COLUMN tensor parallel): this rank's
        vocabulary shard, without the all-gather."""
        if not gather_output and self.tp_mode == TensorParallelMode.COLUMN:
            return self.apply_linear(input, self.bias)
        if is_spec_decoding_head and self.enable_lm_head_tp_in_adp:
            # For LM head TP in ADP, we need to slice the weight for the LM head
            tp_rank = mapping_lm_head_tp.tp_rank
            tp_size = mapping_lm_head_tp.tp_size
            slice_width = ceil_div(self.out_features, tp_size)
            slice_start = tp_rank * slice_width
            slice_end = min((tp_rank + 1) * slice_width, self.out_features)
            weight = self.weight[slice_start:slice_end, :]
            output = _matmul_with_output_dtype(input, weight, self.output_dtype)
        else:
            output = super().forward(input, all_reduce_params=all_reduce_params)
        if (self.tp_mode == TensorParallelMode.COLUMN and self.gather_output
                and self.padding_size > 0):
            output = output[..., :-self.padding_size]

        return output

    def apply_linear(self,
                     input: torch.Tensor,
                     bias: Optional[torch.Tensor],
                     lora_params: Optional[dict] = None,
                     layer_idx: Optional[int] = None) -> torch.Tensor:
        """Project hidden states in the configured GEMM output dtype."""
        # Linear.forward runs the GEMM through apply_linear for the COLUMN and
        # non-parallel modes, so the TP all-gather and the vocab-padding slice
        # in forward() apply unchanged to the output_dtype logits.
        if self._writes_output_dtype():
            assert bias is None, "LMHead has no bias"
            return _matmul_with_output_dtype(input, self.weight,
                                             self.output_dtype)
        return super().apply_linear(input, bias, lora_params, layer_idx)

    def skip_forward(
            self,
            input: torch.Tensor,
            *,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        output_shape = input.shape[:-1] + (self.num_embeddings, )
        output = input.new_empty(output_shape)
        return output

    def load_weights(self,
                     weights: List[Dict],
                     allow_partial_loading: bool = False):
        original_weight = None
        if self.tp_mode == TensorParallelMode.COLUMN:
            if self.tp_rank == self.tp_size - 1 and self.padding_size > 0:
                original_weight = self.weight.data.zero_()
                self.weight.data = self.weight[:-self.padding_size, :]

        super().load_weights(weights,
                             allow_partial_loading=allow_partial_loading)

        if original_weight is not None:
            self.weight.data = original_weight


def get_masked_input_and_mask(
    input_: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vocab_mask = (input_ >= vocab_start_index) & (input_ < vocab_end_index)
    valid_offset = vocab_start_index * vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask.unsqueeze(-1)


def pre_comm_embedding_ops(
    input_: torch.Tensor,
    weight: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    tp_mode: TensorParallelMode,
    vocab_start_index: int,
    vocab_end_index: int,
    gather_output: bool,
    padding_size: int,
):
    if tp_mode == TensorParallelMode.COLUMN:
        input_, input_mask = get_masked_input_and_mask(
            input_,
            vocab_start_index,
            vocab_end_index,
        )
    else:
        # The rejection kernel (trtllm::chain_speculative_sampling) pads non-accepted
        # tokens with -1. When the full vocab is local (non-TP or ROW TP), mask
        # out-of-range ids (e.g. -1) over [0, weight.shape[0]) to avoid an OOB
        # embedding lookup.
        input_, input_mask = get_masked_input_and_mask(
            input_,
            0,
            weight.shape[0],
        )

    # Get the embeddings.
    output = F.embedding(input_, weight)

    output.masked_fill_(input_mask, 0)

    if tp_mode == TensorParallelMode.ROW and gather_output:
        if tp_rank == tp_size - 1 and padding_size > 0:
            output = F.pad(output, (0, padding_size))

    return output


def embedding_skip_forward_impl(input: torch.Tensor, embedding_dim: int,
                                dtype: torch.dtype) -> torch.Tensor:
    output_shape = input.shape[:] + (embedding_dim, )
    output = input.new_empty(output_shape, dtype=dtype)
    return output


@torch.library.custom_op("trtllm::embedding_skip_forward", mutates_args=())
def embedding_skip_forward(input: torch.Tensor, embedding_dim: int,
                           dtype: torch.dtype) -> torch.Tensor:
    return embedding_skip_forward_impl(input, embedding_dim, dtype)


@embedding_skip_forward.register_fake
def _(input, embedding_dim, dtype):
    return embedding_skip_forward_impl(input, embedding_dim, dtype)


class Embedding(LMHead):
    """Embedding layer.

    Adapted from torch.nn.Embedding.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.
        dtype (Optional[torch.dtype]): type of the parameters.
        mapping (Optional[Mapping]): parallelism configuration.
            If not provided, the embedding is not parallelized.
        use_gather_kernel (bool): a non-parallel table looks tokens up with one
            gather launch (``embedding_gather``) instead of the masked lookup.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,
        reduce_output: bool = True,
        enable_torch_compile_for_embedding: Optional[bool] = False,
        use_custom_cublas_mm: bool = False,
        use_gather_kernel: bool = False,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
            reduce_output=reduce_output,
            use_custom_cublas_mm=use_custom_cublas_mm,
            allreduce_strategy=allreduce_strategy,
        )

        self.enable_torch_compile_for_embedding = enable_torch_compile_for_embedding
        self.use_gather_kernel = use_gather_kernel
        # Built once: every torch.compile() call deep-copies the Inductor
        # config, even with disable=True; compilation itself is lazy.
        self._pre_comm_ops_func = (torch.compile(pre_comm_embedding_ops,
                                                 options={"max-autotune": True})
                                   if enable_torch_compile_for_embedding else
                                   pre_comm_embedding_ops)

        if self.tp_size > 1:
            slice_width = math.ceil(num_embeddings / self.tp_size)
            self.vocab_start_index = self.tp_rank * slice_width
            self.vocab_end_index = min((self.tp_rank + 1) * slice_width,
                                       num_embeddings)
        else:
            self.vocab_start_index = 0
            self.vocab_end_index = num_embeddings

    def forward(self, input):
        """Embedding lookup; with ``use_gather_kernel`` a non-parallel table does the masked lookup as one
        gather launch."""
        if (self.tp_size == 1 and self.use_gather_kernel and input.is_cuda
                and self.weight.dim() == 2 and self.weight.is_contiguous()):
            return embedding_gather(input, self.weight)
        if self.tp_size > 1 and not torch.compiler.is_compiling():
            embedding_ops_func = self._pre_comm_ops_func
        elif self.tp_size > 1:
            # Run the ops before all_reduce/all_gather.
            # We use torch.compile() to fuse the tiny pointwise ops before all_reduce/all_gather for Embedding module.
            embedding_ops_func = torch.compile(
                pre_comm_embedding_ops,
                options={"max-autotune": True},
                disable=not self.enable_torch_compile_for_embedding)
        else:
            # Skip torch.compile when TP size is 1 to avoid unnecessary host overhead
            embedding_ops_func = pre_comm_embedding_ops
        output = embedding_ops_func(input, self.weight, self.tp_size,
                                    self.tp_rank, self.tp_mode,
                                    self.vocab_start_index,
                                    self.vocab_end_index, self.gather_output,
                                    self.padding_size)

        # Run the all_reduce/all_gather.
        if self.tp_size > 1:
            if self.tp_mode == TensorParallelMode.COLUMN:
                # Reduce across all the model parallel GPUs.
                output = self.all_reduce(output)
            elif self.tp_mode == TensorParallelMode.ROW:
                if self.gather_output:
                    # Run allgather.
                    output = allgather(output, self.mapping)
                    # Remove the padding.
                    if self.padding_size > 0:
                        output = output[..., :-self.padding_size]

        return output

    def skip_forward(self, input):
        return embedding_skip_forward(input, self.embedding_dim, self.dtype)
