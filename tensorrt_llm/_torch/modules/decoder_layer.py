from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ..attention_backend import AttentionMetadata


class DecoderLayer(nn.Module, ABC):

    @abstractmethod
    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        ...
