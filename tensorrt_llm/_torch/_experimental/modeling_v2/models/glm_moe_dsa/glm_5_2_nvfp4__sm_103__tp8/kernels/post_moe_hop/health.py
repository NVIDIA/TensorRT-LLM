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
"""Host-side health report of the post_moe_hop post-MoE hops (bounded error reporting).

The post_moe_hop kernels never hang and never report errors through the host on their
own: a timed-out poll or an out-of-range permutation entry sets a sticky bit
in the device error word of the rank's post_moe_hop workspace, and every later post_moe_hop call
of that rank publishes NaN until a collective workspace reset, which makes
the outputs of the whole TP group NaN instead of silently wrong.
``check_health`` reads those words and is
meant to be called outside timed regions, e.g. once after warmup and CUDA
graph capture and once at the end of a serving window. It costs one device
synchronization and one 32-byte copy per workspace (one workspace serves all
layers of a TP group).
"""

from dataclasses import dataclass

from .workspace import PostMoeHopWorkspace, _all_ranks_agree

ERROR_BIT_NAMES = {1: "lamport_timeout", 2: "row_sum_timeout", 4: "permutation_out_of_range"}


class PostMoeHopError(RuntimeError):
    """A post_moe_hop reported a device error; the run that produced it is invalid."""


@dataclass(frozen=True)
class PostMoeHopHealth:
    """Counters of every post_moe_hop workspace on this rank at one check point.

    Attributes:
        where: caller-supplied label of the check point.
        workspaces: post_moe_hop workspaces in this process (0 when no hop engaged).
        calls: completed hop launches since the last workspace reset, summed
            over workspaces (the dispatch engagement counter; graph replays
            included).
        error_bits: OR of the sticky device error words on this rank.
        timeout_polls: largest poll count at which a poll gave up.
        layout_repairs: host-side count of calls that needed an aligned copy.
        group_error: some rank of the TP group reported an error (collective
            checks only; equals ``error_bits != 0`` otherwise).
    """

    where: str
    workspaces: int
    calls: int
    error_bits: int
    timeout_polls: int
    layout_repairs: int
    group_error: bool

    @property
    def errors(self) -> list:
        return [name for bit, name in ERROR_BIT_NAMES.items() if self.error_bits & bit]


def check_health(
    where: str, *, collective: bool = False, raise_on_error: bool = True
) -> PostMoeHopHealth:
    """Report the post_moe_hop counters of this rank and fail on any device error.

    Args:
        where: label for the log line, e.g. ``"after_warmup"``.
        collective: also agree on the verdict across each workspace's TP group
            (every rank of the group must call this at the same point); then
            every rank fails together when any rank has an error.
        raise_on_error: raise ``PostMoeHopError`` on an error (after logging it).

    Returns:
        The counters; logged once per call at INFO (ERROR on failure).

    Raises:
        PostMoeHopError: a device error word is nonzero (on this rank, or on any
            rank of the group when ``collective``) and ``raise_on_error``.
    """
    from tensorrt_llm.logger import logger

    workspaces = PostMoeHopWorkspace.instances()
    calls = bits = polls = repairs = 0
    group_error = False
    for workspace in workspaces:
        counters = workspace.counters()
        calls += counters["calls"]
        bits |= counters["error_bits"]
        polls = max(polls, counters["timeout_polls"])
        repairs += counters["layout_repairs"]
        if collective:
            group_error |= not _all_ranks_agree(workspace.comm, counters["error_bits"] == 0)
    group_error |= bits != 0
    rank = workspaces[0].mapping.tp_rank if workspaces else -1
    health = PostMoeHopHealth(where, len(workspaces), calls, bits, polls, repairs, group_error)
    summary = (
        f"[glm_moe_dsa] post_moe_hop health {where}: tp_rank={rank} workspaces={len(workspaces)} "
        f"calls={calls} error_bits={bits:#x} {health.errors} timeout_polls={polls} "
        f"layout_repairs={repairs} group_error={group_error}"
    )
    if not group_error:
        logger.info(summary)
        return health
    logger.error(summary.replace("post_moe_hop health", "POST_MOE_HOP ERROR"))
    if raise_on_error:
        raise PostMoeHopError(
            f"{summary}; the run is invalid (a post_moe_hop workspace needs a collective "
            "reset before further use)"
        )
    return health
