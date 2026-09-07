# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worktree isolation for perf-optimize's parallel item batch."""

from __future__ import annotations

import re
from pathlib import Path

from agent_flow.git_worktree import GitWorktreeIsolation
from agent_flow.orchestration import Node

_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

__all__ = ["CandidateWorktreeIsolation", "item_node_id"]


def item_node_id(index: int, item_id: str) -> str:
    """Return the scheduler node id for the ``index``-th item of a round.

    The id doubles as the worktree directory leaf and the branch suffix, so an
    author-supplied roadmap id is slugged to something filesystem- and ref-safe.
    """
    safe = _UNSAFE_ID_CHARS.sub("-", item_id).strip("-.")[:48] or "item"
    return f"item_{index + 1}_{safe}"


class CandidateWorktreeIsolation(GitWorktreeIsolation):
    """Git-worktree isolation whose branch cleanup is deferred to the campaign.

    The scheduler reclaims a node's branch as soon as nothing in the graph
    depends on it. perf-optimize's Integrator is deliberately *not* a graph node
    — it runs after ``scheduler.run()`` returns and cherry-picks each candidate's
    commit — so under the stock provider every branch would already be gone.
    ``reclaim`` is therefore a no-op and ``_finish_item_batch`` frees the
    branches once integration has consumed them; everything else is upstream's.
    """

    async def reclaim(self, node: Node) -> None:
        return None

    def worktree_path(self, node_id: str) -> Path:
        """The path this provider would hand ``node_id``, without acquiring it.

        Lets the campaign precompute what the scheduler will pass ``run_node``,
        so the checkpointed ledger and the provider cannot disagree.
        """
        return self._worktree_path(Node(id=node_id, type="roadmap_item"))

    def branch_name(self, node_id: str) -> str:
        """The branch this provider would use for ``node_id`` (see above)."""
        return self._branch(Node(id=node_id, type="roadmap_item"))
