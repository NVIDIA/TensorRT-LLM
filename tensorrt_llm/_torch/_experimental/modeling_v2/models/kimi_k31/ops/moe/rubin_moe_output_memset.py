# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero the rows of a CuteDSL MoE output that the FC2 finalize scatter-adds into, in one engine
call (``trtllm::moe_output_memset_inplace``), ahead of ``rubin_cutedsl_fc2_finalize``."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The routed experts' latent width (the output's row) and experts per token.
_HIDDEN, _TOP_K = 3584, 16


def _cell(rows: int, local: int, tile: int, ep_size: int, enable_alltoall: bool) -> dict:
    return dict(
        tokens=rows,
        hidden=_HIDDEN,
        top_k=_TOP_K,
        local_num_experts=local,
        tile_tokens_dim=tile,
        ep_size=ep_size,
        enable_alltoall=enable_alltoall,
    )


class RubinMoeOutputMemset(OpWrapper):
    """``torch.ops.trtllm.moe_output_memset_inplace``: zero ``input`` (``[T, hidden]`` BF16) where
    the finalize will add into it. Without the all-to-all, or with ``ep_size <= top_k`` (every
    K3.1 layout: EP 8 or 16, top 16), every row is zeroed with one memset. With the all-to-all
    and ``ep_size > top_k``, only the rows of tokens with a local pair are zeroed (a kernel over
    the sort tables). In place; returns nothing.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TEP8 context: no all-to-all, a 2048-row chunk",
            spec=_cell(2048, 112, 128, 8, False),
        ),
        Cell(
            why="DEP16 context: all-to-all at EP16 = top 16, the full memset",
            spec=_cell(1024, 56, 128, 16, True),
        ),
        Cell(
            why="all-to-all with EP above top-k: the sparse kernel (EP32, uncertified layout)",
            spec=_cell(512, 28, 128, 32, True),
        ),
    )

    note = """
    The rows the sparse kernel skips keep their old contents; the finalize never writes them, and
    the all-to-all combine never reads them. Run it on the memset stream beside FC1 and join before
    FC2, as the CuteDSL backend does (``record_stream`` on eager steps too).
    """

    def __call__(
        self,
        input: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        tile_tokens_dim: int,
        top_k: int,
        ep_size: int,
        enable_alltoall: bool,
    ) -> None:
        torch.ops.trtllm.moe_output_memset_inplace(
            input=input,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            tile_tokens_dim=tile_tokens_dim,
            top_k=top_k,
            ep_size=ep_size,
            enable_alltoall=enable_alltoall,
        )

    def reference(
        self,
        input: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        tile_tokens_dim: int,
        top_k: int,
        ep_size: int,
        enable_alltoall: bool,
    ) -> torch.Tensor:
        """A copy of ``input`` with the zeroed rows zero. The sparse case zeroes each token that
        has a pair in a written permuted row (the kernel's row walk); ``top_k`` and the tile size
        are read from the tables."""
        out = input.clone()
        if not enable_alltoall or ep_size <= top_k:
            return out.zero_()
        rows = int(num_non_exiting_tiles.reshape(-1)[0]) * tile_tokens_dim
        p = torch.arange(rows, device=input.device)
        live = p < tile_idx_to_mn_limit.long()[p // tile_tokens_dim]
        tokens = permuted_idx_to_expanded_idx.long()[p[live]] // top_k
        out[tokens] = 0
        return out

    def is_valid(
        self,
        input: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        tile_tokens_dim: int,
        top_k: int,
        ep_size: int,
        enable_alltoall: bool,
    ) -> None:
        # The kernel and the memset write through data_ptr() with the row length as the stride.
        assert input.is_contiguous(), "input must be dense: rows are addressed by data_ptr()"
        for name, t in (
            ("tile_idx_to_mn_limit", tile_idx_to_mn_limit),
            ("permuted_idx_to_expanded_idx", permuted_idx_to_expanded_idx),
            ("expanded_idx_to_permuted_idx", expanded_idx_to_permuted_idx),
        ):
            assert t.is_contiguous(), f"{name} must be dense: the kernel reads its data_ptr()"


rubin_moe_output_memset = RubinMoeOutputMemset()
