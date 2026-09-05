# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worktree isolation for perf-optimize's parallel item batch.

The campaign drives :class:`~agent_flow.orchestration.NodeScheduler` over one
flat batch of roadmap items per round, and each item runs in its own git
worktree supplied by :class:`~agent_flow.git_worktree.GitWorktreeIsolation`.
One thing has to change for this workflow, and this module is that one thing.

The scheduler's contract is that a node's *durable output* (its branch) lives
exactly as long as some node still depends on it — so a node with no dependents
is reclaimed (branch deleted) the moment it goes terminal, and ``run`` sweeps
the rest on return. That is right for a graph whose fan-in node is *in* the
graph. Here the Integrator deliberately is **not** a graph node: it runs after
``scheduler.run()`` returns and cherry-picks each candidate's commit. Under the
stock provider every item branch would already be deleted by then.

:class:`CandidateWorktreeIsolation` therefore defers reclamation: ``acquire`` /
``release`` / ``prepare`` / ``commit`` behave exactly as upstream (so a finished
item still gives its worktree directory back promptly), while ``reclaim`` is a
no-op. The campaign frees the branches itself in ``_finish_item_batch``, once
integration has consumed them.
"""

from __future__ import annotations

import re
from pathlib import Path

from agent_flow.git_worktree import GitWorktreeIsolation
from agent_flow.orchestration import Node

# Everything outside this class is collapsed to ``-`` when slugging a roadmap id.
_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

__all__ = ["CandidateWorktreeIsolation", "item_node_id"]


def item_node_id(index: int, item_id: str) -> str:
    """Return the scheduler node id for the ``index``-th item of a round.

    The id doubles as the worktree directory leaf (the provider joins it onto
    ``worktrees_root``) and as the branch suffix, so it must be filesystem- and
    ref-safe. Roadmap ids are author-supplied strings, so everything outside
    ``[A-Za-z0-9._-]`` collapses to ``-`` and the result is length-capped.

    Args:
        index: 0-based position of the item within the round's batch.
        item_id: The roadmap item's id.

    Returns:
        A slug of the form ``item_<n>_<safe-id>``.
    """
    safe = _UNSAFE_ID_CHARS.sub("-", item_id).strip("-.")[:48] or "item"
    return f"item_{index + 1}_{safe}"


class CandidateWorktreeIsolation(GitWorktreeIsolation):
    """Git-worktree isolation that leaves branch cleanup to the campaign.

    Identical to :class:`~agent_flow.git_worktree.GitWorktreeIsolation` except
    that :meth:`reclaim` does nothing: perf-optimize's Integrator runs outside
    the graph and still needs every candidate branch after ``scheduler.run()``
    has returned. See the module docstring for why.
    """

    async def reclaim(self, node: Node) -> None:
        """Keep ``node``'s branch — the out-of-graph Integrator still needs it."""
        return None

    def worktree_path(self, node_id: str) -> Path:
        """Absolute worktree path this provider would use for ``node_id``.

        Lets the campaign precompute the same path the provider will hand a
        node, so the checkpointed batch ledger and the scheduler never disagree
        about where an item ran.
        """
        return self._worktree_path(Node(id=node_id, type="roadmap_item"))

    def branch_name(self, node_id: str) -> str:
        """Branch name this provider would use for ``node_id`` (see above)."""
        return self._branch(Node(id=node_id, type="roadmap_item"))
