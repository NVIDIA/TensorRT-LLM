# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash drafter's input block embedding from one combined lookup (Triton copy)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.spec_tail_kernels import dflash_noise_embedding


class K3DflashNoiseEmbedding(OpWrapper):
    """``[G, block_size, H]`` (new, the dtype of ``combined_embed``): row ``combined_embed[g]``
    (the bonus token's embedding) at block slot 0 and ``combined_embed[G]`` (the mask token's)
    at every other slot, from ``combined_embed = embed([bonus_0 .. bonus_{G-1}, mask])``."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS = (
        Cell(
            why="dep16 (c383): 16 requests, blocks of 6, hidden 7168",
            spec=dict(num_gens=16, block_size=6),
        ),
        Cell(
            why="helix8 c75/c174: 8 requests, blocks of 8",
            spec=dict(num_gens=8, block_size=8),
        ),
        Cell(
            why="helix16_moetp4ep4 (ULL c1): one request, a block of 8",
            spec=dict(num_gens=1, block_size=8),
        ),
    )

    note = """
    A pure copy: grid (G, block_size, cdiv(H, 1024)). A non-contiguous ``combined_embed`` is made
    contiguous first.
    """

    def __call__(self, combined_embed: torch.Tensor, block_size: int) -> torch.Tensor:
        return dflash_noise_embedding(combined_embed, block_size)

    def reference(self, combined_embed: torch.Tensor, block_size: int) -> torch.Tensor:
        num_gens = combined_embed.shape[0] - 1
        out = combined_embed[num_gens:].expand(num_gens, -1).unsqueeze(1)
        out = out.repeat(1, block_size, 1)
        out[:, 0] = combined_embed[:num_gens]
        return out


k3_dflash_noise_embedding = K3DflashNoiseEmbedding()
