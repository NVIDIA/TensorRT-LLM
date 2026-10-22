# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Eligibility policy of the cross_rank_topk context-partitioned indexer region.

Stdlib only, like ``policy``. cross_rank_topk runs inside the indexer_topk envelope (same GPU,
indexer shape, page format, verify width, selection engine and
``max_seq_len`` bound; see ``policy.decide``) and additionally needs:

- exactly one generation request (the in-kernel schedule and the exchange
  slots are per request);
- a tensor-parallel group of a GPU-verified size, without attention DP;
- the compiled ``glm_moe_dsa_cross_rank_topk`` op and a local logits width it accepts.

The decision of a forward (``decide_cp_uniform``) reads only inputs that are
identical on all ranks of the group for the same forward, so the ranks make
the same decision. The facts of a rank's build and device (SM version, CuTe
DSL runtime, compiled op, the kernel's local width limit; ``rank_local_reason``)
are not decided per rank: the workspace creation that follows agrees on them
collectively (``cross_rank_workspace``), so no rank declines on its own while its
peers enter the exchange.
"""

from typing import Tuple

from . import policy

# Tensor-parallel group sizes the GPU tests verify (one node with 4 GPUs, and
# a GEN group of 8 GPUs on two hosts).
SUPPORTED_WORLD_SIZES: Tuple[int, ...] = (4, 8)


def decide_cp(
    *,
    world_size: int,
    op_available: bool,
    local_capacity: int,
    max_local_capacity: int,
    allow_draft_step: bool,
    **indexer_topk_inputs,
) -> policy.Decision:
    """Decide whether cross_rank_topk may run the GEN indexer logits + top-k region.

    Args:
        world_size: Tensor-parallel group size of the GEN ranks.
        op_available: Whether ``torch.ops.trtllm.glm_moe_dsa_cross_rank_topk`` exists.
        local_capacity: Local logits columns the group needs at the indexer
            ``max_seq_len`` (largest per-rank page share).
        max_local_capacity: Largest local width the compiled kernel accepts.
        allow_draft_step: Whether an MTP draft step may engage cross_rank_topk.
        **indexer_topk_inputs: Every keyword input of ``policy.decide``.

    Returns:
        The decision with a fallback reason when ineligible.
    """
    in_draft = bool(indexer_topk_inputs.get("in_mtp_draft_loop", False))
    if in_draft and not allow_draft_step:
        return policy.Decision(False, "MTP draft step (target layers only)")
    base = policy.decide(**dict(indexer_topk_inputs, in_mtp_draft_loop=False))
    if not base.eligible:
        return base
    if indexer_topk_inputs["num_generations"] != 1:
        return policy.Decision(
            False, f"{indexer_topk_inputs['num_generations']} generation requests != 1"
        )
    if world_size not in SUPPORTED_WORLD_SIZES:
        return policy.Decision(
            False, f"tensor-parallel size {world_size} not in {SUPPORTED_WORLD_SIZES}"
        )
    if not op_available:
        return policy.Decision(False, "glm_moe_dsa_cross_rank_topk op is not built")
    if local_capacity <= 0 or local_capacity > max_local_capacity:
        return policy.Decision(
            False, f"local logits width {local_capacity} > kernel limit {max_local_capacity}"
        )
    return policy.Decision(True)


def rank_local_reason(
    *,
    op_available: bool,
    local_capacity: int,
    max_local_capacity: int,
    sm_version: int,
    cute_dsl_available: bool,
) -> str:
    """Why this rank's build or device cannot run cross_rank_topk ("" when it can).

    These are the inputs of ``decide_cp`` that are facts of one rank rather
    than of the forward. The caller hands the reason to the collective
    workspace creation, which keeps every rank native when any rank has one.

    Args:
        op_available: Whether ``torch.ops.trtllm.glm_moe_dsa_cross_rank_topk`` exists.
        local_capacity: Local logits columns the group needs (rank-uniform).
        max_local_capacity: Largest local width this rank's compiled kernel
            accepts (derived from the device's shared memory).
        sm_version: This rank's compute capability, e.g. 103.
        cute_dsl_available: Whether the CUTLASS DSL runtime is importable.
    """
    if sm_version not in policy.SUPPORTED_SM_VERSIONS:
        return f"sm{sm_version} unsupported"
    if not cute_dsl_available:
        return "CUTLASS DSL unavailable"
    if not op_available:
        return "glm_moe_dsa_cross_rank_topk op is not built"
    if local_capacity > max_local_capacity:
        return f"local logits width {local_capacity} > kernel limit {max_local_capacity}"
    return ""


def decide_cp_uniform(
    *, world_size: int, local_capacity: int, allow_draft_step: bool, **indexer_topk_inputs
) -> policy.Decision:
    """``decide_cp`` on the rank-uniform inputs of a forward.

    The rank-local facts (``rank_local_reason``) count as capable here; they
    are agreed by the workspace creation. ``indexer_topk_inputs`` are the keyword inputs
    of ``policy.decide`` other than ``sm_version`` and ``cute_dsl_available``.
    """
    return decide_cp(
        world_size=world_size,
        op_available=True,
        local_capacity=local_capacity,
        max_local_capacity=max(local_capacity, 0),
        allow_draft_step=allow_draft_step,
        **dict(
            indexer_topk_inputs,
            sm_version=policy.SUPPORTED_SM_VERSIONS[0],
            cute_dsl_available=True,
        ),
    )
