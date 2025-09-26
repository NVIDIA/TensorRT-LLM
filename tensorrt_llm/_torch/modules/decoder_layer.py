from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ..attention_backend import AttentionMetadata


class DecoderLayer(nn.Module, ABC):

    @abstractmethod
    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = ...,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        ...

    def skip_forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = ...,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is ...:
            return hidden_states
        else:
            return hidden_states, residual
