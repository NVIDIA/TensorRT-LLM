# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L2 prefetch of a decode MoE layer's routed FC1 weights ahead of the grouped GEMM: the
first N-tiles of the experts the layer's previous decode step routed the most tokens to."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.moe import moe_decode_l2_prefetch as _kernel

LOCAL_EXPERTS = 64
#: The prefetch the target runs: N-tiles 0..3 of the top 12 experts (40.5 MiB per rank) on
#: 64 one-warp CTAs. 54 MiB measured a net loss.
TOP_E, NTILES, CTAS = 12, 4, 64
#: Bytes one (expert, N-tile) slice moves: the NVFP4 weight rows and their scale atoms.
SLICE_BYTES = _kernel.W_SLICE + _kernel.S_SLICE
reference_selection = _kernel.reference_selection


@dataclass(frozen=True)
class CompiledL2Prefetch:
    """The compiled prefetch: `top_e` experts, `ntiles` N-tiles each, on `ctas` CTAs."""

    fn: Callable
    top_e: int
    ntiles: int
    ctas: int


def compile(  # noqa: A001 -- the kernel module's own name
    device_index: int, top_e: int = TOP_E, ntiles: int = NTILES, ctas: int = CTAS
) -> CompiledL2Prefetch:
    """Compile for `device_index`. Call at load; the kernel writes no memory, so nothing to
    warm beyond the compile."""
    fn = _kernel.compile_prefetch(
        top_e=top_e, ntiles=ntiles, ctas=ctas, carveout=False, device=int(device_index)
    )
    return CompiledL2Prefetch(fn, top_e, ntiles, ctas)


class CuteDslMoeL2Prefetch(OpWrapper):
    """Prefetch into L2 the FC1 weight and scale slices of N-tiles `0..ntiles-1` of the
    `top_e` local experts with the largest counts in `expert_hist` (int32 `[64]`; ties to the
    lower index, zero counts never). `fc1_w` / `fc1_s` are one layer's stacked FC1 operands in
    the grouped GEMM's layout (`[64, 4096, 3072]` uint8, `[64, 32, 96, 512]` e4m3). Writes no
    memory and changes no result; returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="a served histogram: 40 active experts", spec=dict(active=40)),
        Cell(why="fewer active experts than top_e", spec=dict(active=5)),
        Cell(why="a never-written histogram: nothing issued", spec=dict(active=0)),
    )

    note = """
    The target forks a side stream at a decode MoE layer's o_proj end (9..64 rows, the push
    top-k's widths) and runs this op there on the counts the layer's previous decode step's
    top-k wrote (`cute_dsl_moe_topk_routing` with `expert_hist`); the routed call joins the
    fork before FC2. On B300 GG1 runs 4-5 us shorter and the decode step 0.13 ms shorter.
    """

    def __call__(
        self,
        expert_hist: torch.Tensor,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        compiled: CompiledL2Prefetch,
    ) -> None:
        w, ws, s, ss = _kernel._layer_bases(fc1_w, fc1_s)
        compiled.fn(expert_hist, w, ws, s, ss, 0)

    def reference(
        self,
        expert_hist: torch.Tensor,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        compiled: CompiledL2Prefetch,
    ) -> None:
        """No output: a prefetch writes nothing (`reference_selection` states which slices)."""
        return None

    def is_valid(
        self,
        expert_hist: torch.Tensor,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        compiled: CompiledL2Prefetch,
    ) -> None:
        assert (
            expert_hist.dtype == torch.int32
            and tuple(expert_hist.shape) == (LOCAL_EXPERTS,)
            and expert_hist.is_contiguous()
            and expert_hist.data_ptr() % 16 == 0
        ), f"expert_hist must be a contiguous 16-byte aligned int32 [{LOCAL_EXPERTS}]"
        _kernel._layer_bases(fc1_w, fc1_s)


cute_dsl_moe_l2_prefetch = CuteDslMoeL2Prefetch()
