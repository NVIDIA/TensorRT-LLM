# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from tensorrt_llm.functional import AllReduceParams
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

from ..distributed import allgather
from .linear import Linear, TensorParallelMode


class LMHead(Linear):
    """LM head layer.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.
        dtype (Optional[torch.dtype]): type of the parameters.
        mapping (Optional[Mapping]): parallelism configuration.
            If not provided, the embedding is not parallelized.
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
        )

        if tensor_parallel_mode == TensorParallelMode.ROW:
            if self.tp_rank == self.tp_size - 1:
                local_in_features -= self.padding_size
        self.in_features = local_in_features
        self.out_features = local_out_features
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._enable_derived_nvfp4_m1 = os.getenv(
            "TLLM_EXPERIMENTAL_DERIVED_NVFP4_LM_HEAD", "0") == "1"
        self.register_buffer("_derived_nvfp4_weight", None, persistent=False)
        self.register_buffer("_derived_nvfp4_weight_scale",
                             None,
                             persistent=False)
        self.register_buffer("_derived_nvfp4_weight_scale_2",
                             None,
                             persistent=False)

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

    @property
    def vocab_size_padded(self) -> int:
        if self.tp_mode == TensorParallelMode.COLUMN and self.gather_output:
            return self.out_features * self.tp_size
        else:
            return self.out_features

    def _supports_derived_nvfp4_m1(self) -> bool:
        if not self._enable_derived_nvfp4_m1 or self.has_any_quant:
            return False
        if (self.tp_size != 1
                or self.tp_mode not in (None, TensorParallelMode.COLUMN)
                or self.padding_size != 0):
            return False
        if (self.weight.device.type != "cuda"
                or self.weight.dtype != torch.bfloat16
                or not self.weight.is_contiguous() or self.in_features % 16 != 0
                or self.out_features % 128 != 0):
            return False
        return torch.cuda.get_device_capability(self.weight.device) in ((10, 0),
                                                                        (10, 3))

    def _can_use_derived_nvfp4_m1(self, input: torch.Tensor) -> bool:
        if not self._supports_derived_nvfp4_m1():
            return False
        return (input.device == self.weight.device
                and input.dtype == torch.bfloat16 and input.is_contiguous()
                and input.numel() == self.in_features)

    @torch.no_grad()
    def _build_derived_nvfp4_weight(self) -> bool:
        if not self._supports_derived_nvfp4_m1():
            return False
        if self._derived_nvfp4_weight is not None:
            return True
        if torch.cuda.is_current_stream_capturing():
            return False

        nvfp4_max = 448.0 * 6.0
        weight_amax = torch.amax(torch.abs(self.weight)).float()
        weight_scale = nvfp4_max / weight_amax
        weight, block_scale = torch.ops.trtllm.fp4_quantize(
            self.weight.detach(), weight_scale, 16, False)
        self._derived_nvfp4_weight = weight
        self._derived_nvfp4_weight_scale = block_scale
        self._derived_nvfp4_weight_scale_2 = weight_amax / nvfp4_max
        logger.info(
            "Enabled derived NVFP4 M=1 LMHead path for dense BF16 weight "
            f"shape {tuple(self.weight.shape)}")
        return True

    def _derived_nvfp4_m1_forward(
            self, input: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._can_use_derived_nvfp4_m1(input):
            return None
        if not self._build_derived_nvfp4_weight():
            return None

        nvfp4_max = 448.0 * 6.0
        input_2d = input.reshape(1, self.in_features)
        input_amax = torch.amax(torch.abs(input_2d)).float()
        input_scale = nvfp4_max / input_amax
        alpha = (input_amax / nvfp4_max) * self._derived_nvfp4_weight_scale_2
        input_fp4, input_block_scale = torch.ops.trtllm.fp4_quantize(
            input_2d, input_scale, 16, False)
        output = torch.ops.trtllm.nvfp4_gemm(
            input_fp4,
            self._derived_nvfp4_weight,
            input_block_scale,
            self._derived_nvfp4_weight_scale,
            alpha,
            torch.bfloat16,
            allowed_backends="cutedsl",
        )
        return output.reshape(input.shape[:-1] + (self.out_features, ))

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        mapping_lm_head_tp: Optional[Mapping] = None,
        is_spec_decoding_head: bool = False,
    ) -> torch.Tensor:
        output = None
        if not is_spec_decoding_head:
            output = self._derived_nvfp4_m1_forward(input)
        if output is None:
            if is_spec_decoding_head and self.enable_lm_head_tp_in_adp:
                # For LM head TP in ADP, we need to slice the weight for the LM head
                tp_rank = mapping_lm_head_tp.tp_rank
                tp_size = mapping_lm_head_tp.tp_size
                slice_width = ceil_div(self.out_features, tp_size)
                slice_start = tp_rank * slice_width
                slice_end = min((tp_rank + 1) * slice_width, self.out_features)
                output = F.linear(input, self.weight[slice_start:slice_end, :],
                                  None)
            else:
                output = super().forward(input,
                                         all_reduce_params=all_reduce_params)
        if (self.tp_mode == TensorParallelMode.COLUMN and self.gather_output
                and self.padding_size > 0):
            output = output[..., :-self.padding_size]

        return output

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

        if self._enable_derived_nvfp4_m1:
            self._build_derived_nvfp4_weight()


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
        # flashinfer's rejection kernel (chain_speculative_sampling) pads non-accepted
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
        )

        self.enable_torch_compile_for_embedding = enable_torch_compile_for_embedding

        if self.tp_size > 1:
            slice_width = math.ceil(num_embeddings / self.tp_size)
            self.vocab_start_index = self.tp_rank * slice_width
            self.vocab_end_index = min((self.tp_rank + 1) * slice_width,
                                       num_embeddings)
        else:
            self.vocab_start_index = 0
            self.vocab_end_index = num_embeddings

    def forward(self, input):
        if self.tp_size > 1:
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
