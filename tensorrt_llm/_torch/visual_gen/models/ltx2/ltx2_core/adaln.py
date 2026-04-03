# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from typing import Optional, Tuple

import torch

from .timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings


class AdaLayerNormSingle(torch.nn.Module):
    """Adaptive layer norm (adaLN-single) from PixArt-Alpha.

    Produces scale/shift/gate modulation parameters from timestep embeddings.
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6, make_linear=None):
        super().__init__()
        if make_linear is None:
            make_linear = torch.nn.Linear
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            make_linear=make_linear,
        )
        self.silu = torch.nn.SiLU()
        self.linear = make_linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep
