# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuteDSL FC1 of the K3.1 routed experts: gather, grouped NVFP4 GEMM, SiTU and NVFP4
requantization of the intermediate in one engine call
(``trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin`` on SM107,
``trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell`` on SM103)."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from tensorrt_llm._utils import get_sm_version

from ._rubin_moe_common import SF_VEC, dequantize_nvfp4, quantize_nvfp4
from .rubin_fc12 import Nvfp4Weight, situ


def _cell(rows: int, local: int, inter: int, tile: int) -> dict:
    return dict(tokens=rows, local=local, inter=inter, tile_size=tile, top_k=16, hidden=3584)


def fc1_rows_reference(
    x: torch.Tensor,
    experts: Sequence[Mapping[str, Nvfp4Weight]],
    tile_idx_to_group_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    *,
    top_k: int,
    tile_size: int,
    situ_beta: float,
    situ_linear_beta: float,
) -> torch.Tensor:
    """FC1 of every written permuted row, FP32 ``[M, I]``: ``situ(gate, up)`` of the token's
    NVFP4 input against its tile's expert, quantize-dequantized to NVFP4 under the FC2 input's
    global scale; rows the sort leaves unwritten are 0.

    ``experts[j]`` is local expert ``j`` as stored (``w1`` gate, ``w3`` up, ``w2`` down); the
    global scales are ``1 / max(input_scale)`` over the experts, as the kernels take them.
    """
    g1 = 1.0 / float(
        torch.stack([e[w][3].float().reshape(()) for e in experts for w in ("w1", "w3")]).max()
    )
    g2 = 1.0 / float(torch.stack([e["w2"][3].float().reshape(()) for e in experts]).max())
    xq = quantize_nvfp4(x.float(), g1)
    m = permuted_idx_to_expanded_idx.numel()
    inter = experts[0]["w1"][0].shape[0]
    out = torch.zeros(m, inter, dtype=torch.float32, device=x.device)
    rows = int(num_non_exiting_tiles.reshape(-1)[0]) * tile_size
    p = torch.arange(rows, device=x.device)
    live = p < tile_idx_to_mn_limit.long()[p // tile_size]
    p = p[live]
    tokens = permuted_idx_to_expanded_idx.long()[p] // top_k
    owner = tile_idx_to_group_idx.long()[p // tile_size]

    def weight(w: Nvfp4Weight) -> torch.Tensor:
        return dequantize_nvfp4(w[0], w[1]) * w[2].float()

    with true_fp32_matmul():
        for j, expert in enumerate(experts):
            mine = owner == j
            if not bool(mine.any()):
                continue
            xe = xq[tokens[mine]]
            act = situ(
                xe @ weight(expert["w1"]).t(),
                xe @ weight(expert["w3"]).t(),
                situ_beta,
                situ_linear_beta,
            )
            out[p[mine]] = quantize_nvfp4(act, g2)
    return out


class RubinCutedslFc1(OpWrapper):
    """FC1 of the routed experts on the CuteDSL path: for each written permuted row ``p`` (tile
    ``p // tile_size`` below ``num_non_exiting_tiles``, ``p`` below its tile's ``mn_limit``), the
    token ``permuted_idx_to_expanded_idx[p] // top_k`` gathered from ``input`` times its tile's
    expert, scaled by ``alpha[e]``, through the gated epilogue, requantized to NVFP4 under
    ``global_sf``. Returns ``(c, c_sf)``: ``c`` ``[M, I/2]`` float4_e2m1fn_x2 and ``c_sf``
    ``[M * I/16]`` uint8 (128x4 swizzled), ``M = permuted_idx_to_expanded_idx.numel()``: what
    ``rubin_cutedsl_fc2_finalize`` takes.

    Operands as the engine NVFP4 CuteDSL MoE method lays them out: ``input`` ``[T, H/2]`` and its
    linear block scales ``input_scale`` ``[T, H/16]``; ``weight`` ``[E, 2I, H/2]`` (gate / up rows
    interleaved in the order the GPU's kernel reads: groups of 16 gate then 16 up rows on SM107,
    of 64 up then 64 gate rows on SM103) with 128x4-swizzled ``weight_scale`` in the same row
    order; ``alpha`` ``[E]``; ``global_sf`` ``[1]``, the FC2 input's global scale. The epilogue is
    ``activation_type`` with the SiTU soft caps (-1 disables), the engine op's arguments and
    defaults.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TEP8 context: 2048 rows, EP8's 112 experts at I 3072, tile 128",
            spec=_cell(2048, 112, 3072, 128),
        ),
        Cell(why="TEP8 context at tile 256", spec=_cell(2048, 112, 3072, 256)),
        Cell(why="TEP8 context at tile 512", spec=_cell(2048, 112, 3072, 512)),
        Cell(
            why="DEP16 context after the all-to-all: 1024 received rows, 56 experts, tile 128",
            spec=_cell(1024, 56, 3072, 128),
        ),
    )

    note = """
    The tactic is the engine runner's, tuned under the engine op's own key; the tile must be the
    sort's (SM103's kernel takes tiles 128 and 256). Rows the sort does not fill are not part of
    the result. The caller owns the input quantization (linear scales), the sort, FC2 and the
    routing-weighted combine. The cells were driven on SM107 and SM103 (tiles 128 and 256 there); SM103
    runs the engine's Blackwell kernel on the same operands except the FC1 row order.
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        activation_type: int,
        situ_beta: float = -1.0,
        situ_linear_beta: float = -1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if get_sm_version() != 107:
            return torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
                input=input.view(torch.float4_e2m1fn_x2),
                weight=weight.view(torch.float4_e2m1fn_x2),
                input_scale=input_scale.view(torch.uint8),
                weight_scale=weight_scale.view(torch.uint8),
                alpha=alpha,
                tile_idx_to_group_idx=tile_idx_to_group_idx,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                global_sf=global_sf,
                num_experts=num_experts,
                top_k=top_k,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset,
                tile_size=tile_size,
                activation_type=activation_type,
                # The Blackwell op takes the soft caps only with the SiTU epilogue.
                situ_beta=None if situ_beta < 0 else situ_beta,
                situ_linear_beta=None if situ_linear_beta < 0 else situ_linear_beta,
            )
        return torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin(
            input=input.view(torch.float4_e2m1fn_x2),
            weight=weight.view(torch.float4_e2m1fn_x2),
            input_scale=input_scale.view(torch.uint8),
            weight_scale=weight_scale.view(torch.uint8),
            alpha=alpha,
            tile_idx_to_group_idx=tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_sf=global_sf,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            tile_size=tile_size,
            output_tensor=None,
            output_sf_tensor=None,
            activation_type=activation_type,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    def reference(
        self,
        x: torch.Tensor,
        experts: Sequence[Mapping[str, Nvfp4Weight]],
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        *,
        top_k: int,
        tile_size: int,
        situ_beta: float,
        situ_linear_beta: float,
    ) -> torch.Tensor:
        """``fc1_rows_reference``: the dequantized ``c`` per permuted row, FP32 ``[M, I]``.

        The documented exception to mirroring (as ``rubin_fc12``'s): it takes the BF16 input and
        the experts as stored rather than the backend's operand layout. The cells hold
        ``(c, c_sf)`` to it through FC2: the kernels' chain against this reference finalized by
        ``rubin_cutedsl_fc2_finalize``'s.
        """
        return fc1_rows_reference(
            x,
            experts,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            top_k=top_k,
            tile_size=tile_size,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    def is_valid(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        activation_type: int,
        situ_beta: float = -1.0,
        situ_linear_beta: float = -1.0,
    ) -> None:
        for name, t in (
            ("input", input),
            ("weight", weight),
            ("input_scale", input_scale),
            ("weight_scale", weight_scale),
            ("permuted_idx_to_expanded_idx", permuted_idx_to_expanded_idx),
        ):
            assert t.is_contiguous(), f"{name} must be dense: the kernel reads its data_ptr()"
        # Linear input scales: one e4m3 byte per 16 input values, row-major [T, H/16].
        rows, half_hidden = input.view(torch.uint8).shape
        assert input_scale.numel() == rows * half_hidden * 2 // SF_VEC, (
            "input_scale must hold one linear e4m3 scale per 16 values of each row"
        )
        assert alpha.numel() == weight.shape[0], "one alpha per local expert"
        # Tables cut for another tile map rows to the wrong experts with no error.
        assert permuted_idx_to_expanded_idx.numel() == tile_idx_to_group_idx.numel() * tile_size, (
            f"the tables are not cut for tile {tile_size}"
        )


rubin_cutedsl_fc1 = RubinCutedslFc1()
