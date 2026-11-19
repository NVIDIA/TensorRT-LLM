# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero the rows of a CuteDSL MoE output that the FC2 finalize scatter-adds into, in one engine
call (``trtllm::moe_output_memset_inplace``), ahead of ``rubin_cutedsl_fc2_finalize``."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

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
    K3.1 layout: EP 8 or 16, top 16), every row is zeroed with one memset; the engine's sparse
    zeroing (all-to-all with ``ep_size > top_k``) is refused. In place; returns nothing.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TEP8 context: no all-to-all, a 2048-row chunk",
            spec=_cell(2048, 112, 128, 8, False),
        ),
        Cell(
            why="DEP16 context: all-to-all at EP16 = top 16, the full memset",
            spec=_cell(1024, 56, 128, 16, True),
        ),
    )

    note = """
    Run it on the memset stream beside FC1 and join before FC2, as the CuteDSL backend does
    (``record_stream`` on eager steps too).
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
        """``input`` with every row zero (a copy)."""
        return torch.zeros_like(input)

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
        # The sparse zeroing (all-to-all with EP above top-k) keeps rows the full memset zeroes.
        assert not enable_alltoall or ep_size <= top_k, (
            f"all-to-all at EP {ep_size} > top {top_k}: the engine zeroes rows sparsely, which "
            "no K3.1 layout runs"
        )
        # The kernel and the memset write through data_ptr() with the row length as the stride.
        assert input.is_contiguous(), "input must be dense: rows are addressed by data_ptr()"
        for name, t in (
            ("tile_idx_to_mn_limit", tile_idx_to_mn_limit),
            ("permuted_idx_to_expanded_idx", permuted_idx_to_expanded_idx),
            ("expanded_idx_to_permuted_idx", expanded_idx_to_permuted_idx),
        ):
            assert t.is_contiguous(), f"{name} must be dense: the kernel reads its data_ptr()"


rubin_moe_output_memset = RubinMoeOutputMemset()
