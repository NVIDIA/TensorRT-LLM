# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 fused FC1 + SiTU + FC2 NVFP4 MoE on SM107 (FC12).

``rubin_fc12`` is one launch of the family launcher (``kernels/moe/rubin_fc12_launch.py``): the
engine op's arguments in its order (SiTU only: no SwiGLU clamp or activation choice), plus the MMA
N and the die-local claim's variant (``die_claim``: the whole scheduler warp claims, in the
family's ``kernels/moe/rubin_fc12_claim.py``) with its weight-stream L2 prefetch
(``l2_prefetch``). Nothing is registered as a torch op. The reference takes the NVFP4 numerics
the MoE entries share from
``_rubin_moe_common.py``.

Also here, for the targets' construction (not an entry): the runtime-helper builder
``rubin_fc12_place_per_die`` re-homes the experts into die-alternating memory.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from tensorrt_llm.logger import logger

from ...kernels.moe.rubin_fc12_launch import (
    DIE_CLAIMS,
    MMA_N,
    TUNER_KEY,
    check_levers,
    fc12_tactics,
    launch_rubin_fc12,
)
from ._rubin_moe_common import SF_VEC, dequantize_nvfp4, quantize_nvfp4

__all__ = [
    "DIE_CLAIMS",
    "MMA_N",
    "TUNER_KEY",
    "fc12_tactics",
    "nvfp4_situ_moe_reference",
    "rubin_fc12",
    "rubin_fc12_place_per_die",
    "situ",
]

#: One stored expert projection: ``(codes uint8 [N, K/2], block scales e4m3 [N, K/16],
#: weight_scale_2 fp32 [], input_scale fp32 [])`` as the ModelOpt checkpoint keeps them.
Nvfp4Weight = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def situ(gate: torch.Tensor, up: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    """Kimi SiTU: ``beta tanh(g / beta) sigmoid(g) * linear_beta tanh(u / linear_beta)``."""
    return (
        beta
        * torch.tanh(gate / beta)
        * torch.sigmoid(gate)
        * linear_beta
        * torch.tanh(up / linear_beta)
    )


def nvfp4_situ_moe_reference(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: Sequence[Mapping[str, Nvfp4Weight]],
    *,
    local_expert_offset: int,
    situ_beta: float,
    situ_linear_beta: float,
) -> torch.Tensor:
    """One rank's routed experts in plain torch, FP32 throughout: ``[T, hidden]``.

    ``experts[j]`` is local expert ``local_expert_offset + j`` as stored: ``{"w1", "w3", "w2"}``
    -> ``Nvfp4Weight`` (w1 the gate, w3 the up projection; this rank's MoE-TP slice). The input
    and the FC1 activation are quantized to NVFP4 under the layer's global scales, ``1 /
    max(input_scale)`` over the rank's experts, as the kernels do. Ids outside the window add
    nothing, so the ranks of an expert-parallel split sum to the layer.
    """
    g1 = 1.0 / float(
        torch.stack([e[w][3].float().reshape(()) for e in experts for w in ("w1", "w3")]).max()
    )
    g2 = 1.0 / float(torch.stack([e["w2"][3].float().reshape(()) for e in experts]).max())
    xq = quantize_nvfp4(x.float(), g1)
    out = torch.zeros(
        x.shape[0], experts[0]["w2"][0].shape[0], dtype=torch.float32, device=x.device
    )

    def weight(w: Nvfp4Weight) -> torch.Tensor:
        return dequantize_nvfp4(w[0], w[1]) * w[2].float()

    with true_fp32_matmul():
        for j, expert in enumerate(experts):
            tok, slot = (topk_ids == local_expert_offset + j).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            xe = xq[tok]
            act = situ(
                xe @ weight(expert["w1"]).t(),
                xe @ weight(expert["w3"]).t(),
                situ_beta,
                situ_linear_beta,
            )
            y = quantize_nvfp4(act, g2) @ weight(expert["w2"]).t()
            out.index_add_(0, tok, y * topk_weights[tok, slot].float().unsqueeze(1))
    return out


#: The FC12 levers of the two layouts (the targets' ``_FC12_*`` launch constants): tep8 lets the
#: tuner pick MMA N; helix8 pins MMA N 256 and claims on the whole scheduler warp.
_TEP8 = dict(mma_n=None)
_HELIX8 = dict(_TEP8, mma_n=256, die_claim="uniform")


class RubinFc12(OpWrapper):
    """One fused FC1 + gated epilogue + FC2 + finalize launch over NVFP4 experts on presorted
    routing tables; ``output`` receives the routing-weighted sum (scatter-add).

    The engine's ``cute_dsl_nvfp4_fc12_fused_rubin`` arguments in the same order, less the SwiGLU
    clamp and the activation choice (the epilogue is SiTU at positive betas; ``tile_size`` is
    128), then the scheduling levers: ``mma_n`` (the one MMA N the tuner may pick; None: 128 and
    256) and ``die_claim``, how a die-local claim runs
    (None: the engine kernel, lane 0 of the scheduler warp alone; ``"uniform"``: the whole warp,
    which keeps the TMA-B and MMA warps on the uniform datapath; the same tiles either way); and
    ``l2_prefetch`` (0..8, default 0, only with ``"uniform"``): K-blocks of the B / SFB weight
    stream that kernel's TMA-B warp prefetches into L2 ahead of its loads (the same loads, so the
    same result). The launch leaves 16 SMs free of FC12 CTAs and claims tiles die-locally from 24
    tokens on when both weights sit in die-alternating memory (``rubin_fc12_place_per_die``).
    Returns nothing.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TEP8 generation b1 k7: 8 rows, EP8's 112 local experts, tile 128 presorted, "
            "below the die floor",
            spec=dict(rows=8, local=112, inter=3072, presorted=True, **_TEP8),
        ),
        Cell(
            why="helix8 generation b16 k5: 96 rows, TP2xEP4's 224 half-width experts, N 256, "
            "die-local claiming on",
            spec=dict(rows=96, local=224, inter=1536, presorted=True, **_HELIX8),
        ),
        Cell(
            why="helix8 generation b8 k7: 64 rows, N 256",
            spec=dict(rows=64, local=224, inter=1536, presorted=True, **_HELIX8),
        ),
        Cell(
            why="helix8 b16 k5 with the engine kernel's die-local claim (lane 0 alone)",
            spec=dict(
                rows=96, local=224, inter=1536, presorted=True, **dict(_HELIX8, die_claim=None)
            ),
        ),
        Cell(
            why="TEP8 generation b64 k7: 512 rows, moe_sort's tables at the 128-row tile",
            spec=dict(rows=512, local=112, inter=3072, presorted=False, **_TEP8),
        ),
    )

    note = """
    Layout and scales are the engine NVFP4 CuteDSL MoE method's (w3_w1 and w2 as it lays them out,
    block scales 128x4 swizzled, per-expert alphas, the FC2 input's global scale as
    ``global_sf``). The tables are ``moe_sort``'s at ``tile_size``. Unless ``zeroed_counters``
    comes zeroed with a zeroed ``output``, the launch zeroes both itself right before the kernel.
    The kernel keeps no state; the compiled kernel is cached per tactic, tile, epilogue, cluster
    count, die table, claim variant and L2 prefetch depth. At equal levers and
    ``die_claim`` None the launch is the engine runner's, argument for argument. ``"uniform"``
    takes the family subclass of the engine kernel for every die-local launch inside its Int32
    bound (any K3.1 shape) and the engine kernel otherwise; it issues the engine claim's atomics in
    the same order, so rows the finalize sums from at most two experts are bitwise the engine
    kernel's. ``l2_prefetch`` changes no load, barrier or arithmetic of that kernel, so the same
    rows are bitwise its own at P 0. The family tests check both row equalities, the Int32 bound
    at its edge and that a launch takes the uniform claim only where it applies.
    """

    def __call__(
        self,
        input: torch.Tensor,
        fc1_weight: torch.Tensor,
        input_scale: torch.Tensor,
        fc1_weight_scale: torch.Tensor,
        fc1_alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_scale: torch.Tensor,
        fc2_alpha: torch.Tensor,
        output: torch.Tensor,
        token_final_scales: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        ep_size: int,
        enable_alltoall: bool,
        scaling_vector_size: int,
        situ_beta: float,
        situ_linear_beta: float,
        zeroed_counters: Optional[torch.Tensor] = None,
        *,
        mma_n: Optional[int],
        die_claim: Optional[str] = None,
        l2_prefetch: int = 0,
    ) -> None:
        launch_rubin_fc12(
            (
                input,
                fc1_weight,
                input_scale,
                fc1_weight_scale,
                fc1_alpha,
                tile_idx_to_group_idx,
                tile_idx_to_mn_limit,
                permuted_idx_to_expanded_idx,
                num_non_exiting_tiles,
                global_sf,
                fc2_weight,
                fc2_weight_scale,
                fc2_alpha,
                output,
                token_final_scales,
                expanded_idx_to_permuted_idx,
            ),
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            tile_size=tile_size,
            scaling_vector_size=scaling_vector_size,
            ep_size=ep_size,
            enable_alltoall=enable_alltoall,
            situ_beta=float(situ_beta),
            situ_linear_beta=float(situ_linear_beta),
            zeroed_counters=zeroed_counters,
            mma_n=mma_n,
            die_claim=die_claim,
            l2_prefetch=l2_prefetch,
        )

    def reference(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        experts: Sequence[Mapping[str, Nvfp4Weight]],
        *,
        local_expert_offset: int,
        situ_beta: float,
        situ_linear_beta: float,
    ) -> torch.Tensor:
        """``nvfp4_situ_moe_reference`` (FP32 ``[T, hidden]``).

        The documented exception to mirroring, as the catalog's NVFP4 MoE entry takes it:
        ``__call__`` takes the prepared operands (FC1 rows and block scales as the backend lays
        them out, sorted routing), and a reference given those would re-derive the layout rather
        than the computation. This takes the BF16 input, the routing and the experts as stored.
        The levers (``die_claim`` and ``l2_prefetch`` included) change only which CTA computes a
        tile and when its weights reach L2, never the result.
        """
        return nvfp4_situ_moe_reference(
            x,
            topk_ids,
            topk_weights,
            experts,
            local_expert_offset=local_expert_offset,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    def is_valid(
        self,
        input: torch.Tensor,
        fc1_weight: torch.Tensor,
        input_scale: torch.Tensor,
        fc1_weight_scale: torch.Tensor,
        fc1_alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_scale: torch.Tensor,
        fc2_alpha: torch.Tensor,
        output: torch.Tensor,
        token_final_scales: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        ep_size: int,
        enable_alltoall: bool,
        scaling_vector_size: int,
        situ_beta: float,
        situ_linear_beta: float,
        zeroed_counters: Optional[torch.Tensor] = None,
        *,
        mma_n: Optional[int],
        die_claim: Optional[str] = None,
        l2_prefetch: int = 0,
    ) -> None:
        """Inputs the launch takes and answers wrongly. The runner rejects a bad N, tile or
        NVFP4 operand dtype itself, and the kernel non-positive SiTU betas; the runner hands the
        kernel bare pointers and sizes the launch from ``permuted_idx_to_expanded_idx`` and
        ``tile_size`` alone. ``zeroed_counters`` and ``output`` must both be zero when the counters
        are given (the launch then skips its own zeroing)."""
        check_levers(mma_n, die_claim, l2_prefetch)
        # Only the tactic filter checks the block size; the fallback tactic skips it.
        assert scaling_vector_size == SF_VEC, f"NVFP4 blocks are 16, got {scaling_vector_size}"
        tensors = dict(
            input=input,
            fc1_weight=fc1_weight,
            input_scale=input_scale,
            fc1_weight_scale=fc1_weight_scale,
            fc1_alpha=fc1_alpha,
            tile_idx_to_group_idx=tile_idx_to_group_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_sf=global_sf,
            fc2_weight=fc2_weight,
            fc2_weight_scale=fc2_weight_scale,
            fc2_alpha=fc2_alpha,
            output=output,
            token_final_scales=token_final_scales,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        )
        for name, t in tensors.items():
            assert t.is_contiguous(), f"{name} must be dense: the kernel reads its data_ptr()"
        tables = (
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            expanded_idx_to_permuted_idx,
        )
        assert all(t.dtype == torch.int32 for t in tables), "the sort tables are read as int32"
        assert token_final_scales.dtype == torch.float32, "the routing weights are read as FP32"
        tiles = tile_idx_to_group_idx.numel()
        assert tile_idx_to_mn_limit.numel() == tiles, "the two per-tile tables differ in length"
        # Tables cut for another tile map rows to the wrong experts.
        assert permuted_idx_to_expanded_idx.numel() == tiles * tile_size, (
            f"{permuted_idx_to_expanded_idx.numel()} permuted rows over {tiles} tiles: the tables "
            f"are not cut for tile {tile_size}"
        )
        rows = (output.size(0), top_k)
        assert expanded_idx_to_permuted_idx.shape == token_final_scales.shape == rows, (
            f"routing of {tuple(token_final_scales.shape)} for {rows[0]} output rows, top {top_k}"
        )
        if zeroed_counters is not None:
            assert not bool(output.any()), "zeroed_counters given with an output that is not zero"
            assert not bool(zeroed_counters.any()), "zeroed_counters are not zero"


rubin_fc12 = RubinFc12()


def rubin_fc12_place_per_die(
    w: Mapping[str, torch.nn.Parameter],
    layers: Sequence[int],
    *,
    fc1: str = "l{}_moe_fc1_w",
    fc2: str = "l{}_moe_fc2_w",
) -> int:
    """Re-home each MoE layer's FC12 expert weights into die-alternating memory (runtime-helper builder).

    For every layer in ``layers``, ``w[fc1.format(i)]`` and ``w[fc2.format(i)]`` are copied
    (same shape, dtype and bytes) into memory whose 2 MiB chunks alternate between the dies; the
    parameter is pointed at the copy and the original goes back to the driver, layer by layer,
    as the engine FC12 backend's ``place_expert_weights_per_die`` does. The SM -> die table of
    the die schedule is measured here, ahead of CUDA graph capture. Call it in
    ``post_load_weights`` before any view of these parameters is taken. Returns the number of
    layers placed: 0 on a device with one die, or when the driver does not home a chunk as asked
    (the weights then keep the default placement).
    """
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.rubin_die_locality import (
        die_alternating_copy,
        sm_die_table,
    )

    placed = 0
    for i in layers:
        names = (fc1.format(i), fc2.format(i))
        copies = [die_alternating_copy(w[name].data) for name in names]
        if any(c is None for c in copies):
            logger.warning(
                "rubin_fc12_place_per_die: no die-alternating memory on this device; the FC12 "
                "experts keep the default placement"
            )
            return placed
        for name, copy in zip(names, copies):
            w[name].data = copy
        del copies
        # The originals back to the driver: the next layer's copies are driver allocations
        # outside the caching allocator.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        placed += 1
    if placed:
        first = w[fc1.format(layers[0])]
        logger.info(
            f"rubin_fc12_place_per_die: {placed} MoE layers' FC12 experts die-alternating "
            f"({fc1.format(layers[0])} at {first.data_ptr():#x})"
        )
        sm_die_table(first.device.index)
    return placed
