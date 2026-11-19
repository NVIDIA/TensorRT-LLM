# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed experts on the TRTLLM-Gen MXFP4 x MXFP8 runner with a pinned tactic.

The catalog runner entry (``trtllm::mxe4m3_mxe2m1_block_scale_moe_runner``) picks its ``[tile_n, config]`` tactic
with the autotuner, per process, on dummy routing, per power-of-two bucket of the rank's token count. Under attention
DP that bucket is the per-rank count while the runner sees every EP rank's rows (16 x rows/rank at DEP16). This entry
runs the same runner class with a tactic from a per-layout table (:func:`pinned_tactic`), and with no tactic it falls
back to the catalog entry. Every pinned tactic's output is bitwise-equal to the autotuned tactic's (the tactic changes
the token tiling and the CTA schedule, not the per-expert K order of either GEMM; checked per cell in the entry's
test).

:func:`k31_runner_moe` is the routed experts' entry: the table's tactic where it has one, else the catalog entry.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.moe.mxe4m3_mxe2m1_block_scale_moe_runner import (
    mxe4m3_mxe2m1_block_scale_moe_runner,
)

#: K3.1 routed experts: 896 experts, top-16, latent hidden 3584, inter 3072 (per-rank 3072 / moe_tp).
NUM_EXPERTS = 896
TOP_K = 16
HIDDEN = 3584
#: TRTLLM-Gen routing method id of DeepSeek-V3 routing (K3.1: sigmoid scores, one group, top-k renormalized).
ROUTING_DEEPSEEK_V3 = 2
#: ``ActType_TrtllmGen.SiTu``: the fused SiTU-GLU FC1 epilogue of the K3.1 cubins.
ACT_SITU = 3

#: ``(lo, hi, tile_n, config)`` per ``(local experts, local intermediate)`` layout: for a call with ``lo <= runner
#: rows <= hi`` the runner runs ``[tile_n, config]``. Rows are the rows the runner sees (attention DP: every EP rank's
#: rows, ``ep x max rows per rank``). Layouts: DEP16 (56 x 3072), TEP16 (224 x 768), TEP32 (112 x 768), TEP8 prefill
#: (112 x 3072). Every listed tactic's output is bitwise-equal to the autotuned one's at those shapes (some other TEP
#: tactics are not -- split-K configs -- and are never listed).
_PINS: dict[Tuple[int, int], Tuple[Tuple[int, int, int, int], ...]] = {
    (56, 3072): ((128, 255, 32, 18), (256, 511, 64, 11), (512, 32767, 128, 18)),
    (224, 768): ((1, 11, 8, 128), (12, 255, 16, 50)),
    (112, 768): ((1, 11, 8, 128), (12, 31, 8, 140), (32, 127, 16, 48)),
    (112, 3072): ((512, 32768, 128, 18),),
}


def pinned_tactic(
    local_num_experts: int, intermediate_size: int, num_rows: int
) -> Optional[Tuple[int, int]]:
    """The table's ``(tile_n, config)`` for this layout and runner row count, or ``None``."""
    for lo, hi, tile_n, config in _PINS.get((int(local_num_experts), int(intermediate_size)), ()):
        if lo <= num_rows <= hi:
            return tile_n, config
    return None


@functools.lru_cache(maxsize=1)
def _runner():
    """The TRTLLM-Gen MXFP4 x MXFP8 runner class with the fused SiTU-GLU FC1 epilogue."""
    return torch.classes.trtllm.MxE4m3MxE2m1BlockScaleMoERunner(ACT_SITU, True)


@functools.lru_cache(maxsize=1024)
def _listed_tactics(
    hidden: int,
    intermediate_size: int,
    local_num_experts: int,
    num_rows: int,
    valid_intermediate_size: int,
) -> frozenset:
    """The ``(tile_n, config)`` pairs the runner lists for this shape and row count (bounded cache: a
    prefill server sees many distinct row counts)."""
    return frozenset(
        tuple(int(v) for v in t)
        for t in _runner().get_valid_configs(
            TOP_K,
            hidden,
            intermediate_size,
            local_num_experts,
            num_rows,
            HIDDEN,
            valid_intermediate_size,
        )
    )


def _engine_op(
    hidden_states,
    hidden_states_scale,
    topk_ids,
    topk_weights,
    gemm1_weights,
    gemm1_weights_scale,
    gemm1_alpha,
    gemm1_beta,
    gemm2_weights,
    gemm2_weights_scale,
    intermediate_size,
    valid_intermediate_size,
    local_expert_offset,
    local_num_experts,
    output,
    tune_max_num_tokens,
    use_dp,
):
    """The catalog runner entry (autotuned tactic) on pre-routed MXFP8 rows."""
    out = mxe4m3_mxe2m1_block_scale_moe_runner(
        None,
        None,
        hidden_states,
        hidden_states_scale.flatten(),
        gemm1_weights,
        gemm1_weights_scale,
        None,
        gemm1_alpha,
        gemm1_beta,
        None,
        gemm2_weights,
        gemm2_weights_scale,
        None,
        NUM_EXPERTS,
        TOP_K,
        1,
        1,
        intermediate_size,
        HIDDEN,
        valid_intermediate_size,
        local_expert_offset,
        local_num_experts,
        1.0,
        ROUTING_DEEPSEEK_V3,
        ACT_SITU,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        tune_max_num_tokens=tune_max_num_tokens,
        use_dp=use_dp,
    )
    return output if output is not None else out


class K31W4A8MoeRunner(OpWrapper):
    """One rank's routed experts: routing from the given top-16 ids + FC1 (SiTU-GLU, MXFP8
    requant) + FC2 + finalize, on the TRTLLM-Gen MXFP4 x MXFP8 runner, with ``tactic`` pinned.

    Returns ``[rows, valid_hidden_size]`` BF16: ``sum_k topk_weights[r, k] * expert_k(x_r)`` over
    the ``k`` whose expert this rank owns (``local_expert_offset`` .. ``+ local_num_experts``),
    FP32 accumulate, BF16 expert outputs, routing weight applied at finalize. ``tactic=None`` runs
    the engine op (autotuned). A ``tactic`` the runner does not list for this row count falls back
    to the engine op too, so a table entry can never launch an unsupported tile.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 receiving rank, 96 rows/rank x 16 ranks: the pinned token tile 128 "
            "(autotuner picks 32 at the per-rank bucket)",
            spec=dict(rows=1536, local=56, inter=3072, tactic=(128, 18)),
        ),
        Cell(
            why="DEP16, 224 rows/rank x 16: the autotuner picks token tile 64, the pin keeps 128",
            spec=dict(rows=3584, local=56, inter=3072, tactic=(128, 18)),
        ),
        Cell(
            why="TEP16 (MoE TP4 x EP4) 72 rows: tile 16 inside the 64-row bucket",
            spec=dict(rows=72, local=224, inter=768, tactic=(16, 50)),
        ),
        Cell(
            why="TEP32 (MoE TP4 x EP8) one speculative verify step: 8 rows, tile 8",
            spec=dict(rows=8, local=112, inter=768, tactic=(8, 128)),
        ),
        Cell(
            why="TEP8 prefill chunk 8192 rows over 112 x 3072: tile 128 (the autotuner picks 256 at 16384)",
            spec=dict(rows=8192, local=112, inter=3072, tactic=(128, 18)),
        ),
        Cell(
            why="no pin: the catalog entry's autotuned tactic (the fallback path)",
            spec=dict(rows=24, local=224, inter=768, tactic=None),
        ),
    )

    note = """
    Bitwise contract: every tactic in the pin table produces the engine op's output bit for bit
    on the same inputs (the test checks each cell against `reference`, the engine op). Weights are
    the runner-layout tensors (`w3_w1_weight` [local, 2 * inter, hidden / 2] e2m1 shuffled
    rows, `w3_w1_weight_scale` / `w2_weight_scale` swizzled UE8M0, `w2_weight`), the input is
    MXFP8 (`mxfp8_quantize(x, False, alignment=512)`: E4M3 [rows, 3584] + linear UE8M0 scales), ids
    int32 and weights BF16 [rows, 16] with global expert ids (ids outside this rank's window are
    skipped). The runner keeps its own routing (permutation from the ids) and finalize kernels.
    CUDA-graph safe (all workspaces come from the caching allocator per call, like the engine op).
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_alpha: torch.Tensor,
        gemm1_beta: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        intermediate_size: int,
        valid_intermediate_size: Optional[int],
        local_expert_offset: int,
        local_num_experts: int,
        tactic: Optional[Tuple[int, int]],
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> torch.Tensor:
        rows, hidden = hidden_states.shape
        valid_inter = valid_intermediate_size or intermediate_size
        if tactic is not None and (int(tactic[0]), int(tactic[1])) in _listed_tactics(
            hidden, intermediate_size, local_num_experts, rows, valid_inter
        ):
            out = _runner().run_moe(
                None,
                None,
                hidden_states,
                hidden_states_scale.flatten(),
                gemm1_weights,
                gemm1_weights_scale,
                None,
                gemm1_alpha,
                gemm1_beta,
                None,
                gemm2_weights,
                gemm2_weights_scale,
                None,
                None,
                None,
                None,
                NUM_EXPERTS,
                TOP_K,
                1,
                1,
                intermediate_size,
                HIDDEN,
                valid_intermediate_size,
                local_expert_offset,
                local_num_experts,
                1.0,
                ROUTING_DEEPSEEK_V3,
                [int(tactic[0]), int(tactic[1])],
                topk_weights,
                topk_ids,
                output,
            )
            return output if output is not None else out
        return _engine_op(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            gemm1_weights,
            gemm1_weights_scale,
            gemm1_alpha,
            gemm1_beta,
            gemm2_weights,
            gemm2_weights_scale,
            intermediate_size,
            valid_intermediate_size,
            local_expert_offset,
            local_num_experts,
            output,
            tune_max_num_tokens,
            use_dp,
        )

    def reference(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_alpha: torch.Tensor,
        gemm1_beta: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        intermediate_size: int,
        valid_intermediate_size: Optional[int],
        local_expert_offset: int,
        local_num_experts: int,
        tactic: Optional[Tuple[int, int]],
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> torch.Tensor:
        """The engine op with the autotuner's tactic (``tactic`` is ignored).
        The entry's plain-torch semantics are the engine op's own entry
        (``catalog/moe/mxe4m3_mxe2m1_block_scale_moe_runner``); this entry only changes which
        of the runner's tactics runs, so its reference is that op, bit for bit."""
        return _engine_op(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            gemm1_weights,
            gemm1_weights_scale,
            gemm1_alpha,
            gemm1_beta,
            gemm2_weights,
            gemm2_weights_scale,
            intermediate_size,
            valid_intermediate_size,
            local_expert_offset,
            local_num_experts,
            output,
            tune_max_num_tokens,
            use_dp,
        )

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_alpha: torch.Tensor,
        gemm1_beta: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        intermediate_size: int,
        valid_intermediate_size: Optional[int],
        local_expert_offset: int,
        local_num_experts: int,
        tactic: Optional[Tuple[int, int]],
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> None:
        """The runner reads raw pointers: a strided view is read as if dense (see the catalog runner
        entry), and BF16 weights with int32 ids are the only routing dtypes it reads right."""
        for name, t in (
            ("hidden_states", hidden_states),
            ("hidden_states_scale", hidden_states_scale),
            ("topk_ids", topk_ids),
            ("topk_weights", topk_weights),
            ("gemm1_weights", gemm1_weights),
            ("gemm1_weights_scale", gemm1_weights_scale),
            ("gemm2_weights", gemm2_weights),
            ("gemm2_weights_scale", gemm2_weights_scale),
        ):
            assert t.is_contiguous(), f"{name} must be contiguous"
        assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.bfloat16


k31_w4a8_moe_runner = K31W4A8MoeRunner()


def k31_runner_moe(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w3_w1_weight: torch.Tensor,
    w3_w1_weight_scale: torch.Tensor,
    gemm1_alpha: torch.Tensor,
    gemm1_beta: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    intermediate_size: int,
    valid_intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    tune_max_num_tokens: int,
    use_dp: bool,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One rank's pre-routed MXFP4 x MXFP8 routed experts on the TRTLLM-Gen runner: with :func:`pinned_tactic`'s
    tactic for this layout and row count where the table has one (:data:`k31_w4a8_moe_runner`), else the engine op
    with the autotuner's tactic. The arguments are those of :class:`K31W4A8MoeRunner` (BF16 weights, int32 ids);
    ``output``: a ``[rows, 3584]`` BF16 buffer the runner writes and returns."""
    tactic = None
    if (
        x.dtype == torch.float8_e4m3fn
        and x.shape[-1] == HIDDEN
        and w3_w1_weight_scale.shape[-1] == x.shape[-1] // 32
    ):
        tactic = pinned_tactic(
            local_num_experts, valid_intermediate_size or intermediate_size, x.shape[0]
        )
    return k31_w4a8_moe_runner(
        x,
        x_sf,
        topk_ids,
        topk_weights,
        w3_w1_weight,
        w3_w1_weight_scale,
        gemm1_alpha,
        gemm1_beta,
        w2_weight,
        w2_weight_scale,
        intermediate_size,
        valid_intermediate_size,
        local_expert_offset,
        local_num_experts,
        tactic,
        output,
        tune_max_num_tokens,
        use_dp,
    )
