import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from tensorrt_llm.functional import AllReduceParams
from tensorrt_llm.mapping import Mapping

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
    ):
        local_in_features = embedding_dim
        local_out_features = num_embeddings
        mapping = mapping or Mapping()
        tp_size = mapping.tp_size

        if tensor_parallel_mode == TensorParallelMode.ROW:
            local_in_features = math.ceil(embedding_dim / tp_size)
            self.padding_size = tp_size * local_in_features - embedding_dim
        elif tensor_parallel_mode == TensorParallelMode.COLUMN:
            local_out_features = math.ceil(num_embeddings / tp_size)
            self.padding_size = tp_size * local_out_features - num_embeddings

        super().__init__(
            local_in_features * tp_size,
            local_out_features * tp_size,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
        )

        if tensor_parallel_mode == TensorParallelMode.ROW:
            if self.tp_rank == self.tp_size - 1:
                local_in_features -= self.padding_size
        self.in_features = local_in_features
        self.out_features = local_out_features
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        weight_shape = (self.out_features, self.in_features)
        self.weight = Parameter(torch.empty(weight_shape, dtype=dtype))
        self.register_parameter("bias", None)

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
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        output = super().forward(input, all_reduce_params=all_reduce_params)
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

    def load_weights(self, weights: List[Dict]):
        original_weight = None
        if self.tp_mode == TensorParallelMode.COLUMN:
            if self.tp_rank == self.tp_size - 1 and self.padding_size > 0:
                original_weight = self.weight.data.zero_()
                self.weight.data = self.weight[:-self.padding_size, :]

        super().load_weights(weights)

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
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
        )
        if self.tp_size > 1:
            slice_width = math.ceil(num_embeddings / self.tp_size)
            self.vocab_start_index = self.tp_rank * slice_width
            self.vocab_end_index = min((self.tp_rank + 1) * slice_width,
                                       num_embeddings)

    def forward(self, input):
        if self.tp_size > 1:
            if self.tp_mode == TensorParallelMode.COLUMN:
                # Build the mask.
                input, input_mask = get_masked_input_and_mask(
                    input,
                    self.vocab_start_index,
                    self.vocab_end_index,
                )
        # Get the embeddings.
        output = F.embedding(input, self.weight)
        # Mask the output embedding.
        if self.tp_size > 1:
            if self.tp_mode == TensorParallelMode.COLUMN:
                output.masked_fill_(input_mask, 0)
                # Reduce across all the model parallel GPUs.
                output = self.all_reduce(output)
            elif self.tp_mode == TensorParallelMode.ROW:
                if self.gather_output:
                    if self.tp_rank == self.tp_size - 1 and self.padding_size > 0:
                        output = F.pad(output, (0, self.padding_size))
                    output = allgather(output, self.mapping)
                    if self.padding_size > 0:
                        output = output[..., :-self.padding_size]

        return output

    def skip_forward(self, input):
        output_shape = input.shape[:] + (self.embedding_dim, )
        output = torch.empty(
            output_shape,
            dtype=self.dtype,
            device=input.device,
        )
        return output
