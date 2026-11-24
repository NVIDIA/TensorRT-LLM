# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuteDSL FC2 of the K3.1 routed experts on SM107: grouped NVFP4 GEMM and the routing-weighted
scatter-add into the output, in one engine call
(``trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin``).

Also here, for the targets' construction (not an entry): ``check_cutedsl_lever_env``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ._rubin_moe_common import (
    FC1_TILE256_TACTIC,
    FC2_TILE256_TACTIC,
    check_cutedsl_tactic,
    dequantize_nvfp4,
)
from .rubin_fc12 import Nvfp4Weight

#: Engine variables of the CuteDSL MoE path the flat targets keep at the value they run:
#: the FC2 N-tile pin must be unset (the engine op reads it under this entry), and the input
#: quantization runs in order (the flat targets never fork it).
CUTEDSL_LEVER_ENV = {
    "TRTLLM_CUTEDSL_FC2_N_TILE_SIZE": (None,),
    "TLLM_MOE_ASYNC_INPUT_QUANT": (None, "0"),
}


#: A cell's FC1 and FC2 tactics at tile 256: the ones a context server pins.
_TILE256_TACTICS = dict(fc1_tactic=FC1_TILE256_TACTIC, fc2_tactic=FC2_TILE256_TACTIC)


def _cell(rows: int, local: int, inter: int, tile: int) -> dict:
    return dict(tokens=rows, local=local, inter=inter, tile_size=tile, top_k=16, hidden=3584)


def fc2_finalize_reference(
    act: torch.Tensor,
    experts: Sequence[Mapping[str, Nvfp4Weight]],
    tile_idx_to_group_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    token_final_scales: torch.Tensor,
    output: torch.Tensor,
    *,
    top_k: int,
    tile_size: int,
) -> torch.Tensor:
    """``output`` plus, for every written permuted row ``p``, the token's routing weight times
    ``act[p] @ w2[e]^T`` (``e`` the tile's local expert, ``w2`` as stored): FP32 ``[T, H]``."""
    out = output.float().clone()
    rows = int(num_non_exiting_tiles.reshape(-1)[0]) * tile_size
    p = torch.arange(rows, device=act.device)
    live = p < tile_idx_to_mn_limit.long()[p // tile_size]
    p = p[live]
    expanded = permuted_idx_to_expanded_idx.long()[p]
    tokens, slots = expanded // top_k, expanded % top_k
    owner = tile_idx_to_group_idx.long()[p // tile_size]
    with true_fp32_matmul():
        for j, expert in enumerate(experts):
            mine = owner == j
            if not bool(mine.any()):
                continue
            w2 = expert["w2"]
            y = act[p[mine]] @ (dequantize_nvfp4(w2[0], w2[1]) * w2[2].float()).t()
            weight = token_final_scales[tokens[mine], slots[mine]].float().unsqueeze(1)
            out.index_add_(0, tokens[mine], y * weight)
    return out


class RubinCutedslFc2Finalize(OpWrapper):
    """FC2 of the routed experts on the CuteDSL path, finalized into ``output``: for each written
    permuted row, ``alpha[e] * deq(input[p]) @ deq(weight[e])^T`` times the pair's routing weight,
    added into ``output[token]`` (BF16 ``[T, H]``; every row it adds into starts at zero:
    ``rubin_moe_output_memset`` zeroes the whole output, ``rubin_moe_zero_local_rows`` the
    received rows behind the DEP16 context's all-to-all). In place; returns nothing.

    ``input`` / ``input_scale`` are ``rubin_cutedsl_fc1``'s ``(c, c_sf)``; ``weight``
    ``[E, H, I/2]`` and its 128x4-swizzled ``weight_scale`` as the engine NVFP4 CuteDSL MoE method
    lays them out; ``alpha`` ``[E]``; the sort tables at the same ``tile_size``;
    ``token_final_scales`` ``[T, top_k]`` FP32.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TEP8 context: 2048 rows, 112 experts, tile 128", spec=_cell(2048, 112, 3072, 128)
        ),
        Cell(why="TEP8 context at tile 256", spec=_cell(2048, 112, 3072, 256)),
        Cell(why="TEP8 context at tile 512", spec=_cell(2048, 112, 3072, 512)),
        Cell(
            why="DEP16 context: 1024 received rows into the all-to-all's combine buffer",
            spec=_cell(1024, 56, 3072, 128),
        ),
        Cell(
            why="DEP12 context, rank 0: 768 received rows, 75 experts into the combine buffer",
            spec=_cell(768, 75, 3072, 128),
        ),
        Cell(
            why="DEP12 context, rank 8: 768 received rows, 74 experts from 600",
            spec=dict(_cell(768, 74, 3072, 128), local_expert_offset=600),
        ),
        Cell(
            why="DEP12 context, rank 11: the last window, 74 experts from 822 to expert 895",
            spec=dict(_cell(768, 74, 3072, 128), local_expert_offset=822),
        ),
        Cell(
            why="DEP8 context, rank 7: 512 received rows, the last 112 experts from 784",
            spec=dict(_cell(512, 112, 3072, 128), local_expert_offset=784),
        ),
        Cell(
            why="DEP16 context at tile 256, the 2-CTA N256 tactic the context pins",
            spec=dict(_cell(1024, 56, 3072, 256), **_TILE256_TACTICS),
        ),
        Cell(
            why="DEP12 context, rank 8, at tile 256 with the pinned 2-CTA N256 tactic",
            spec=dict(
                _cell(768, 74, 3072, 256),
                local_expert_offset=600,
                **_TILE256_TACTICS,
            ),
        ),
        Cell(
            why="DEP8 context, rank 7, at tile 256 with the pinned 2-CTA N256 tactic",
            spec=dict(
                _cell(512, 112, 3072, 256),
                local_expert_offset=784,
                **_TILE256_TACTICS,
            ),
        ),
    )

    note = """
    The scatter-add of a token's pairs is atomic, so the order of its adds and the output's last
    bit vary between launches; the test's band covers it. Each pair's term (its FP32 products over
    K in one order, ``mma_tiler`` K 256, no split-K, scaled and rounded to BF16) is the same bits at
    any tile and tactic. The tactic is the engine runner's: ``precomputed_tactic`` pins one (the
    engine op's argument), ``None`` takes the AutoTuner's pick under the engine op's key; a pinned
    tactic's MMA tile M must be the sort's tile. ``TRTLLM_CUTEDSL_FC2_N_TILE_SIZE`` (read by that
    op) must stay unset, which ``check_cutedsl_lever_env`` asserts at construction.
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        token_final_scales: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        precomputed_tactic: str | None = None,
    ) -> None:
        torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin(
            input=input.view(torch.float4_e2m1fn_x2),
            weight=weight.view(torch.float4_e2m1fn_x2),
            input_scale=input_scale.view(torch.uint8),
            weight_scale=weight_scale.view(torch.uint8),
            alpha=alpha,
            output=output,
            tile_idx_to_group_idx=tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            token_final_scales=token_final_scales,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            tile_size=tile_size,
            output_dtype=torch.bfloat16,
            precomputed_tactic=precomputed_tactic,
        )

    def reference(
        self,
        act: torch.Tensor,
        experts: Sequence[Mapping[str, Nvfp4Weight]],
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        token_final_scales: torch.Tensor,
        output: torch.Tensor,
        *,
        top_k: int,
        tile_size: int,
    ) -> torch.Tensor:
        """``fc2_finalize_reference`` (FP32 ``[T, H]``). The documented exception to mirroring (as
        ``rubin_fc12``'s): ``act`` is the FC1 activation per permuted row, dequantized
        (``rubin_cutedsl_fc1.reference``), and the experts are as stored."""
        return fc2_finalize_reference(
            act,
            experts,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            token_final_scales,
            output,
            top_k=top_k,
            tile_size=tile_size,
        )

    def is_valid(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        token_final_scales: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        precomputed_tactic: str | None = None,
    ) -> None:
        for name, t in (
            ("input", input),
            ("weight", weight),
            ("output", output),
            ("token_final_scales", token_final_scales),
            ("permuted_idx_to_expanded_idx", permuted_idx_to_expanded_idx),
        ):
            assert t.is_contiguous(), f"{name} must be dense: the kernel reads its data_ptr()"
        assert output.dtype == torch.bfloat16, "the finalize adds BF16 rows"
        assert token_final_scales.dtype == torch.float32, "the routing weights are read as FP32"
        assert token_final_scales.shape == (output.shape[0], top_k), (
            f"routing of {tuple(token_final_scales.shape)} for {output.shape[0]} rows, top {top_k}"
        )
        assert alpha.numel() == weight.shape[0], "one alpha per local expert"
        assert permuted_idx_to_expanded_idx.numel() == tile_idx_to_group_idx.numel() * tile_size, (
            f"the tables are not cut for tile {tile_size}"
        )
        check_cutedsl_tactic(precomputed_tactic, tile_size, weight.shape[1])


rubin_cutedsl_fc2_finalize = RubinCutedslFc2Finalize()


def check_cutedsl_lever_env(environ: Mapping[str, str]) -> None:
    """Raise unless the CuteDSL MoE variables in ``environ`` hold the value the flat targets run:
    ``TRTLLM_CUTEDSL_FC2_N_TILE_SIZE`` unset (the engine FC2 op under this entry reads it at each
    call) and ``TLLM_MOE_ASYNC_INPUT_QUANT`` unset or 0 (the flat targets quantize in order)."""
    for name, allowed in CUTEDSL_LEVER_ENV.items():
        raw = environ.get(name)
        if raw not in allowed:
            raise ValueError(
                f"{name}={raw} is not what the flat target runs (allowed: "
                f"{', '.join('unset' if a is None else a for a in allowed)})"
            )
