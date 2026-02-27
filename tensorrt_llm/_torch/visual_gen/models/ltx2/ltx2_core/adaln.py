# Ported from https://github.com/Lightricks/LTX-2
# packages/ltx-core/src/ltx_core/model/transformer/adaln.py

from typing import Optional, Tuple

import torch

from .timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings


class AdaLayerNormSingle(torch.nn.Module):
    """Adaptive layer norm (adaLN-single) from PixArt-Alpha.

    Produces scale/shift/gate modulation parameters from timestep embeddings.
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3
        )
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(
            embedding_dim, embedding_coefficient * embedding_dim, bias=True
        )

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep
