# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give the orchestrator a writable ``$HOME`` so it can spawn agents at all.

THE PROBLEM

A sandboxed session's ``$HOME`` is read-only. The orchestrator spawns a fresh
``claude`` CLI per role, that CLI authenticates on its own, and its first act
is to take a lock beside the credential store::

    API Error: EROFS: read-only file system, mkdir
    '/home/<user>/.config/anthropic/credentials.lock'

The turn fails with ``stop_reason='stop_sequence'`` and zero tokens, so the
message reads as a model failure. It is not: no agent ever started.

This cost the previous run its entire agent transport. It worked around it by
emitting prompts for a human-driven session to dispatch, which verified the
prompts and the gates but left the orchestrator's own dispatch path
unexercised. This module is the fix, so that the next run does not have to.

WHAT IT DOES

Stage a private home under the workspace, seed it with a *copy* of the real
credential store, and return the environment that points a child process at
it. Applied to ``os.environ`` at startup, every backend inherits it --- Claude
Code and Codex alike --- with no per-backend plumbing.

WHAT IT DOES NOT DO

It does not solve credential *rotation*. The copy holds a refresh token the
parent session also holds; a child that refreshes may invalidate the other's.
Measured on the previous run: a copied credential set works for roughly forty
minutes. :func:`refresh_credentials` re-seeds from the real store, which is
cheap enough to call before every launch, and the honest operating model is:
the run checkpoints per stage, so an auth failure is a resume rather than a
loss. Do not read this module as making the problem go away.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

#: Where the real credential store lives, relative to the real ``$HOME``.
CREDENTIAL_SUBPATH = Path(".config/anthropic")

#: Subdirectories of the real home worth copying, beyond the credentials.
#: ``.codex`` is here because the PlanReviewer runs on Codex and keeps its own
#: state there.
OPTIONAL_SUBPATHS = (Path(".codex"),)

#: The child's Claude Code config directory. Deliberately NOT named
#: ``.claude``: a sandbox that denies writes under ``**/.claude`` --- which is
#: the common shape, since that is where settings and skills live --- would
#: make the child unable to write its own config, and the failure is a warning
#: rather than an error, so it is easy to miss.
CLAUDE_CONFIG_DIRNAME = "claude-config"


@dataclass(frozen=True)
class AgentHome:
    """A staged home directory and the environment that selects it."""

    root: Path
    env: dict[str, str]

    def apply(self) -> None:
        """Export this home into the current process's environment.

        Children inherit it, which is the whole point: no backend needs to
        know it exists.
        """
        os.environ.update(self.env)


def home_is_writable(home: Path | None = None) -> bool:
    """Whether the real ``$HOME`` accepts the write the CLI needs.

    Tested by creating and removing a directory beside the credential store,
    because that is literally what fails --- ``os.access`` reports the
    permission bits and a sandbox is not a permission bit.
    """
    root = Path(home or os.path.expanduser("~"))
    probe = root / CREDENTIAL_SUBPATH / ".agent_home_write_probe"
    try:
        probe.mkdir(parents=True, exist_ok=True)
        probe.rmdir()
        return True
    except OSError:
        return False


def refresh_credentials(root: Path, home: Path | None = None) -> bool:
    """Re-seed ``root`` from the real credential store. Returns success.

    Cheap --- two small JSON files --- so call it before every launch rather
    than once at setup. A stale copy fails as an authentication error several
    minutes into a turn, which is a much more expensive way to find out.
    """
    source = Path(home or os.path.expanduser("~")) / CREDENTIAL_SUBPATH
    if not source.is_dir():
        return False
    destination = root / CREDENTIAL_SUBPATH
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        return True
    except OSError:
        return False


def stage_agent_home(
    root: Path,
    *,
    home: Path | None = None,
    git_identity: tuple[str, str] | None = None,
) -> AgentHome:
    """Create ``root`` as a usable home and return the selecting environment.

    ``git_identity`` is ``(name, email)``. It is written to the staged
    ``.gitconfig`` because the committer's ``git commit`` needs an identity
    and, once ``$HOME`` is redirected, the real one is no longer on the
    search path --- a run that gets this far and then fails every commit with
    *"Please tell me who you are"* has traded one infrastructure failure for
    another.
    """
    # ABSOLUTE, always. The env this returns is read by a CHILD process whose
    # cwd is not ours -- the workspace, usually -- so a relative HOME resolves
    # somewhere else and the child silently builds a second home underneath
    # the first. Observed exactly once, as
    # `<workspace>/<workspace>/agent-home/`, and it is invisible until
    # somebody lists the directory.
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    refresh_credentials(root, home=home)

    real = Path(home or os.path.expanduser("~"))
    for sub in OPTIONAL_SUBPATHS:
        source, destination = real / sub, root / sub
        if source.is_dir() and not destination.exists():
            try:
                shutil.copytree(source, destination)
            except OSError:
                # Best effort. A missing Codex state directory costs a cold
                # start, not a run.
                pass

    config_dir = root / CLAUDE_CONFIG_DIRNAME
    config_dir.mkdir(parents=True, exist_ok=True)
    marker = config_dir / ".claude.json"
    if not marker.exists():
        marker.write_text("{}\n", encoding="utf-8")

    if git_identity:
        name, email = git_identity
        (root / ".gitconfig").write_text(
            f"[user]\n\tname = {name}\n\temail = {email}\n", encoding="utf-8"
        )

    return AgentHome(
        root=root,
        env={
            "HOME": str(root),
            # The credential store is resolved through XDG, not through HOME,
            # on the platforms this runs on. Setting only HOME leaves the
            # child reading the real (read-only) store and the failure is
            # identical to having done nothing.
            "XDG_CONFIG_HOME": str(root / ".config"),
            "CLAUDE_CONFIG_DIR": str(config_dir),
        },
    )


def ensure_agent_home(
    workspace: Path,
    *,
    home: Path | None = None,
    git_identity: tuple[str, str] | None = None,
) -> AgentHome | None:
    """Stage a home under ``workspace`` iff the real one cannot be written.

    Returns ``None`` when the real home is fine, so that an ordinary
    workstation run is untouched and nothing has to be configured for it.
    """
    if home_is_writable(home):
        return None
    return stage_agent_home(Path(workspace) / "agent-home", home=home, git_identity=git_identity)


__all__ = [
    "AgentHome",
    "CLAUDE_CONFIG_DIRNAME",
    "CREDENTIAL_SUBPATH",
    "ensure_agent_home",
    "home_is_writable",
    "refresh_credentials",
    "stage_agent_home",
]
