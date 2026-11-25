# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Symmetric-memory workspace of the MegaMoE kernel: the cross-rank buffers it reads and writes
over NVLink, and the launch-bucket ladder that sizes them.

`megamoe_token_buckets` is the entry (the ladder every rank must derive identically). The rest are
build-time helpers of the owning target: the EP group, the bucket pick, the size probe of the kernel's
shared workspace, the **collective** rendezvous (`megamoe_symm_regions`) and the local top-k buffer. The
engine's provider cache behind `megamoe_symm_regions` is process-wide state: the target calls it for every
bucket at build time, in the same order on every rank, keeps the returned regions and passes them to
each layer call (all layers share them, layers run one after another). Never call it inside a forward or
a graph capture. The 4-rank matrix beside the tests certifies the rendezvous.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_LADDER = (256, 1024, 4096)


class MegaMoeTokenBuckets(OpWrapper):
    """The `max_tokens_per_rank` ladder: one compiled kernel and one symmetric buffer per rung.

    Rungs of `(256, 1024, 4096)` below `max_num_tokens`, then `max_num_tokens` itself (4096 when it
    is 0). Every EP rank must derive the same ladder from the same inputs: a rank on another rung
    launches a differently sized kernel and the NVLink barrier never completes.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep32 (max_num_tokens 150): a single rung",
            spec=dict(max_num_tokens=150, expected=(150,)),
        ),
        Cell(why="the first rung exactly", spec=dict(max_num_tokens=256, expected=(256,))),
        Cell(why="one past the first rung", spec=dict(max_num_tokens=257, expected=(256, 257))),
        Cell(
            why="dep8 (max_num_tokens 1280): two rungs below the cap",
            spec=dict(max_num_tokens=1280, expected=(256, 1024, 1280)),
        ),
        Cell(
            why="the top rung exactly", spec=dict(max_num_tokens=4096, expected=(256, 1024, 4096))
        ),
        Cell(
            why="past the ladder", spec=dict(max_num_tokens=8192, expected=(256, 1024, 4096, 8192))
        ),
        Cell(
            why="unset (0) means the full ladder",
            spec=dict(max_num_tokens=0, expected=(256, 1024, 4096)),
        ),
    )
    note = "The engine's own `MegaMoECuteDsl._resolve_maxt_buckets` is the same ladder; the test compares them."

    def __call__(self, max_num_tokens: int, sm_version: int = 103) -> List[int]:
        full = int(max_num_tokens) if int(max_num_tokens) > 0 else 4096
        return sorted({b for b in _LADDER if b < full} | {full})

    def reference(self, max_num_tokens: int, sm_version: int = 103) -> List[int]:
        """The rungs below the cap, then the cap, written out rung by rung."""
        full = max_num_tokens if max_num_tokens > 0 else 4096
        rungs = []
        for rung in (256, 1024, 4096):
            if rung < full:
                rungs.append(rung)
        return rungs + [full]

    def is_valid(self, max_num_tokens: int, sm_version: int = 103) -> None:
        assert max_num_tokens >= 0, "a negative cap is read as unset"
        assert sm_version == 103, "the ladder is certified for sm_103 (the argument is not read)"


megamoe_token_buckets = MegaMoeTokenBuckets()


def megamoe_pick_bucket(buckets: List[int], max_rank_tokens: int) -> int:
    """The smallest rung >= `max_rank_tokens`, the maximum over every rank's token count.

    The count must be identical on every rank (it is, under attention data parallelism, the
    `max(all_rank_num_tokens)` every rank already holds). A count past the top rung raises on every
    rank alike rather than clamping on some.
    """
    for b in buckets:
        if max_rank_tokens <= b:
            return b
    raise RuntimeError(
        f"max rank token count {max_rank_tokens} exceeds the top MegaMoE bucket {buckets[-1]}"
    )


def megamoe_ep_process_group(mapping):
    """The torch.distributed group the kernel exchanges over: the expert-parallel group.

    Under the MPI orchestrator nothing has initialized torch.distributed, so this first brings up a
    NCCL WORLD group over the MPI session (a collective: every rank calls it, at build time). The
    kernel's barrier-reset fence runs on WORLD, so the deployment must be pure DEP:
    `mapping.moe_ep_size == world_size` (anything else raises here, on every rank, instead of
    hanging in the first launch).
    """
    import torch.distributed as dist

    from tensorrt_llm._torch.distributed.initialization import ensure_cutedsl_torch_distributed

    ensure_cutedsl_torch_distributed(mapping.moe_ep_size)
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("MegaMoE needs torch.distributed (mpirun or Ray) for its rendezvous")
    if dist.get_world_size() != mapping.moe_ep_size:
        raise ValueError(
            f"MegaMoE requires pure DEP: world {dist.get_world_size()} != EP {mapping.moe_ep_size}"
        )
    try:
        return mapping.moe_ep_group_pg
    except (NotImplementedError, AttributeError):
        return dist.group.WORLD


def megamoe_shared_workspace_bytes(
    *,
    world_size: int,
    num_topk: int,
    num_experts_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
    max_tokens_per_rank: int,
    gate_up_clamp: Optional[float],
    combine_format: str = "bf16",
    apply_topk_in_fc1: bool = True,
) -> Tuple[int, bool]:
    """(bytes of the kernel's symmetric `shared_workspace`, whether this bucket takes its inputs in
    place) for the default tactic of this problem. Must be called with the very arguments the
    launch will use: a workspace sized for another kernel is written out of bounds."""
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
        query_megamoe_kernel_properties,
    )

    return query_megamoe_kernel_properties(
        world_size=world_size,
        num_topk=num_topk,
        num_experts_per_rank=num_experts_per_rank,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        expand_intermediate_size_per_partition=2 * intermediate_size,
        max_tokens_per_rank=max_tokens_per_rank,
        apply_topk_in_fc1=apply_topk_in_fc1,
        gate_up_clamp=gate_up_clamp,
        in_kernel_fc2_reduce=False,
        combine_format=combine_format,
    )


def megamoe_symm_regions(
    *,
    process_group,
    world_size: int,
    rank: int,
    hidden_size: int,
    max_tokens_per_rank: int,
    num_topk: int,
    combine_format: str,
    shared_workspace_bytes: int,
):
    """The bucket's symmetric buffers: `activation` `[max_T, hidden/2]` u8, `activation_sf`
    `[max_T, sf_bytes]` u8, `topk_weights` `[max_T, top_k]` f32, `combine_output` `[max_T, 1,
    hidden]` bf16, `shared_workspace` `[bytes]` u8, and `peer_offsets` (a `world_size` list of
    int, each peer's base minus this rank's).

    **Collective**: the first call per (group, layout) allocates one symmetric buffer and
    rendezvouses over `process_group`, so every rank must make the same calls in the same order,
    at build time and never inside a forward or a CUDA-graph capture. Later calls with the same
    arguments return the cached buffers, which is how every MoE layer shares one workspace (layers
    run one after another).
    """
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import (
        get_megamoe_symm_provider,
    )

    provider = get_megamoe_symm_provider(
        process_group=process_group,
        world_size=world_size,
        rank=rank,
        hidden_size=hidden_size,
        max_tokens_per_rank=max_tokens_per_rank,
        num_topk=num_topk,
        output_dtype=torch.bfloat16,
        combine_format=combine_format,
        shared_workspace_bytes=shared_workspace_bytes,
    )
    return provider.get_regions()


def megamoe_topk_idx_buffer(max_tokens_per_rank: int, num_topk: int, device="cuda") -> torch.Tensor:
    """The rank-local expert-slot buffer `[max_T, top_k]` int64, all -1 (an idle row).

    Plain device memory, not symmetric: peers never read it. Rows past the live token count must
    stay -1 (the dispatch skips them), which `megamoe_stage_inputs` maintains.
    """
    return torch.full((max_tokens_per_rank, num_topk), -1, dtype=torch.int64, device=device)
