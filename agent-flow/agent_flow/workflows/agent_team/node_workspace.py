"""Per-node sub-workspace for concurrent agent-team nodes.

When the workflow runs the execution graph concurrently, each node needs its
own private ``status.md`` / ``progress.yaml`` so that parallel nodes never
share writes to the shared-workspace files. This module carves out a
sub-workspace per node under ``workspace/nodes/<slug>/`` and hands back the
node-scoped :class:`~agent_flow.workflows.agent_team.progress.ProgressContext`
and :class:`~agent_flow.workflows.agent_team.status.StatusContext` that point
at those private files.

The per-node files are seeded exactly the way the shared workspace seeds them:
``status.md`` starts as an empty string and ``progress.yaml`` is initialized
via :func:`init_progress_file` (a schema-valid empty progress log), so the
node runner can reuse the existing progress/status tool factories unchanged.

Creation is idempotent: :func:`create_node_workspace` may be called twice for
the same node id (e.g. on resume) and will only (re)seed a file when it is
absent or empty — an already-written, non-empty file is never clobbered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .progress import ProgressContext, init_progress_file
from .status import StatusContext

# Sub-workspaces live under ``<workspace>/nodes/<slug>/``.
NODES_DIRNAME = "nodes"
STATUS_FILENAME = "status.md"
PROGRESS_FILENAME = "progress.yaml"


def node_dir_slug(node_id: str) -> str:
    r"""Map a node id to a filesystem-safe leaf directory name.

    Node ids may contain path separators (e.g. ``s1/g2``) that are illegal in
    a single path component; ``/`` and ``\\`` are replaced with ``_`` so the id
    collapses to one safe leaf directory. Ids like ``s1.g2`` pass through
    unchanged (dots are legal in a path component).

    This intentionally duplicates the git-worktree slug rule instead of
    importing ``agent_flow.git_worktree.node_id_to_slug`` — the workflow layer
    must not depend on git.

    Args:
        node_id: The node's globally unique id.

    Returns:
        A string safe to use as a single path component.
    """
    return node_id.replace("/", "_").replace("\\", "_")


@dataclass(frozen=True)
class NodeWorkspace:
    """A node's private sub-workspace: its directory and scoped context files.

    ``progress_context()`` / ``status_context()`` return fresh contexts pointed
    at this node's private ``progress.yaml`` / ``status.md`` so the existing
    progress/status tool factories operate on per-node state.
    """

    node_id: str
    dir: Path
    status_path: Path
    progress_path: Path

    def progress_context(self) -> ProgressContext:
        """Return a :class:`ProgressContext` bound to this node's progress.yaml."""
        return ProgressContext(path=self.progress_path)

    def status_context(self) -> StatusContext:
        """Return a :class:`StatusContext` bound to this node's status.md."""
        return StatusContext(path=self.status_path)


def _is_absent_or_empty(path: Path) -> bool:
    """True when ``path`` does not exist or holds only whitespace."""
    if not path.is_file():
        return True
    return not path.read_text(encoding="utf-8").strip()


def create_node_workspace(workspace: Path, node_id: str) -> NodeWorkspace:
    """Create (idempotently) the private sub-workspace for ``node_id``.

    Ensures ``workspace/nodes/<slug>/`` exists and seeds it with an empty
    ``status.md`` and a schema-valid empty ``progress.yaml`` (via
    :func:`init_progress_file`). Seeding only happens when the target file is
    absent or empty, so calling this twice for the same id on resume preserves
    any already-written non-empty ``status.md`` / ``progress.yaml``.

    Args:
        workspace: The shared workspace root.
        node_id: The node's globally unique id.

    Returns:
        The :class:`NodeWorkspace` describing the node's private paths.
    """
    node_dir = workspace / NODES_DIRNAME / node_dir_slug(node_id)
    node_dir.mkdir(parents=True, exist_ok=True)

    status_path = node_dir / STATUS_FILENAME
    progress_path = node_dir / PROGRESS_FILENAME

    if _is_absent_or_empty(status_path):
        status_path.write_text("", encoding="utf-8")
    if _is_absent_or_empty(progress_path):
        init_progress_file(progress_path)

    return NodeWorkspace(
        node_id=node_id,
        dir=node_dir,
        status_path=status_path,
        progress_path=progress_path,
    )
