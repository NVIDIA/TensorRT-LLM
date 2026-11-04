# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node-local trees: the run's checkouts on the node's own disk instead of the shared filesystem.

A run whose task sets ``environment.local_trees`` keeps the trees it makes by
the thousand under one node-local root (:func:`root_for`, exported to every
process of the run as :data:`ROOT_ENV`) instead of in its workspace on the
shared filesystem, whose inode quota and per-file metadata cost they spend:

- ``<root>/snapshots/<commit>`` and ``<root>/commits/<commit>`` are COMMIT
  TREES: one commit checked out read-only, its native build linked from the
  source checkout. The path alone names what is there, so a tree a job names
  that its node does not hold yet is made there before the job starts
  (:func:`materialize`, run by the spool scheduler): a standalone repository
  that borrows the source's objects (``objects/info/alternates``), checked out
  sparse with the cone recorded when the tree was first asked for, its Git
  LFS files read from the source's LFS store
  (:func:`publish`, beside the source's git directory, so any node reads it).
- ``<root>/worktrees/<task>`` and ``<root>/tune/<slug>`` are MUTABLE TREES:
  git worktrees of the source checkout, registered (and locked, so a ``git
  worktree prune`` on another node keeps them) in its shared git directory,
  their files on this node. A job submitted against one runs against a commit
  tree of what it holds then (:func:`freeze`): its HEAD, or a commit of its
  uncommitted edits on top kept under ``refs/mv2flow/wip/<name>``.

Every commit's objects stay in the source's shared object store, so a lost
node loses no commit; only a mutable tree's edits since its last job are lost.

Standard-library only, and runnable by path: the scheduler runs ``localtree.py
ensure <tree> --repo <checkout> -- <command>`` as a job's first step.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

#: The run's node-local root; unset, the run keeps its trees in its workspace.
ROOT_ENV = "MV2FLOW_LOCAL_TREES"
#: The checkout a commit tree is made from when the job names none.
REPO_ENV = "MV2FLOW_REPO"
#: The directory under ``environment.local_trees`` every run of one user shares.
DIRNAME_PREFIX = "mv2flow-trees"
COMMIT_KINDS = ("snapshots", "commits")
MUTABLE_KINDS = ("worktrees", "tune")
#: Written last into a commit tree: its commit and cone, as JSON.
READY_FILENAME = ".mv2flow-tree"
#: Beside the source's git directory: what each commit tree checks out.
MANIFEST_DIRNAME = "mv2flow-trees"
WIP_REF_PREFIX = "refs/mv2flow/wip/"
#: A commit tree on a job's node that no job asked for in this long is removed.
SWEEP_AGE_S = 24 * 3600.0
CHECKOUT_FLAGS = ("-c", "checkout.workers=16")
#: Never part of a frozen tree: the agent sandbox's own files.
FREEZE_EXCLUDES = (".claude", ".cc-writes", ".mcp.json")
NATIVE_ROOT = "tensorrt_llm"
_MAKE_LOCKS = ".make-locks"
_WIP_IDENTITY = {
    "GIT_AUTHOR_NAME": "mv2flow",
    "GIT_AUTHOR_EMAIL": "mv2flow@localhost",
    "GIT_COMMITTER_NAME": "mv2flow",
    "GIT_COMMITTER_EMAIL": "mv2flow@localhost",
}
_TREE = re.compile(
    r"^(?P<root>.*/" + DIRNAME_PREFIX + r"[^/]*/[^/]+)/"
    r"(?P<kind>snapshots|commits|worktrees|tune)/(?P<name>[^/].*)$"
)
_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class LocalTreeError(RuntimeError):
    """A node-local tree could not be made or frozen."""


# ------------------------------------------------------------------- naming


def root_for(base: str | Path, workspace: str | Path, user: str | None = None) -> Path:
    """The node-local root of the run whose workspace is ``workspace``, under ``base``.

    One directory per user under ``base``, one per workspace under that: the
    workspace's name, for whoever reads the disk, and a digest of its real
    path, so two workspaces of one name keep apart.
    """
    real = os.path.realpath(str(workspace))
    digest = hashlib.sha1(real.encode("utf-8")).hexdigest()[:10]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(real) or "run")
    user = user or os.environ.get("USER") or str(os.getuid())
    return Path(base) / f"{DIRNAME_PREFIX}-{user}" / f"{name}-{digest}"


def local_root(environ: Mapping[str, str] | None = None) -> Path | None:
    """The run's node-local root, from :data:`ROOT_ENV`; ``None`` when the run has none."""
    value = (os.environ if environ is None else environ).get(ROOT_ENV, "").strip()
    return Path(value) if value.startswith("/") else None


def parse(path: str | Path) -> tuple[Path, str, str] | None:
    """``(root, kind, name)`` of a node-local tree's path; ``None`` for any other path."""
    match = _TREE.match(os.path.normpath(str(path)))
    if match is None:
        return None
    return Path(match["root"]), match["kind"], match["name"]


def commit_of(path: str | Path) -> str | None:
    """The commit a commit tree's path names; ``None`` for any other path."""
    parsed = parse(path)
    if parsed is None or parsed[1] not in COMMIT_KINDS or not _SHA.match(parsed[2]):
        return None
    return parsed[2]


def is_commit_tree(path: str | Path) -> bool:
    return commit_of(path) is not None


def is_mutable_tree(path: str | Path) -> bool:
    parsed = parse(path)
    return parsed is not None and parsed[1] in MUTABLE_KINDS


def commit_tree(root: str | Path, kind: str, commit: str) -> Path:
    if kind not in COMMIT_KINDS or not _SHA.match(commit):
        raise ValueError(f"no commit tree {kind}/{commit}")
    return Path(root) / kind / commit


def wip_ref(name: str) -> str:
    return WIP_REF_PREFIX + re.sub(r"[^A-Za-z0-9._-]", "-", name)


def tree_name(path: str | Path) -> str:
    """A mutable tree's name for its WIP ref: ``<kind>-<name>``."""
    parsed = parse(path)
    if parsed is None:
        raise ValueError(f"{path} is not a node-local tree")
    return f"{parsed[1]}-{parsed[2]}".replace("/", "-")


# ------------------------------------------------------------------- git


def _git(
    cwd: str | Path, *args: str, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def _checked(cwd: str | Path, *args: str, env: Mapping[str, str] | None = None) -> str:
    done = _git(cwd, *args, env=env)
    if done.returncode != 0:
        lines = (done.stderr or done.stdout).strip().splitlines()
        raise LocalTreeError(f"git {' '.join(args[:3])} in {cwd}: {lines[-1] if lines else ''}")
    return done.stdout.strip()


def common_dir(repo: str | Path) -> Path:
    """``repo``'s shared git directory, absolute."""
    path = Path(_checked(repo, "rev-parse", "--git-common-dir"))
    return path if path.is_absolute() else Path(repo) / path


def native_artifacts(repo: str | Path) -> list[str]:
    """Git-ignored entries directly under ``tensorrt_llm/`` in ``repo``: the native build."""
    out = _git(
        repo, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory",
        "--", f"{NATIVE_ROOT}/",
    ).stdout  # fmt: skip
    found = []
    for line in out.splitlines():
        parts = line.strip().rstrip("/").split("/")
        if len(parts) == 2 and parts[0] == NATIVE_ROOT and parts[1] != "__pycache__":
            if not parts[1].endswith(".egg-info"):
                found.append("/".join(parts))
    return found


def link_native(repo: str | Path, tree: str | Path) -> None:
    """Link ``repo``'s native build into ``tree``, as a task worktree has it."""
    for rel in native_artifacts(repo):
        target = Path(tree) / rel
        if target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(Path(repo) / rel)


def cone_of(tree: str | Path) -> tuple[str, ...] | None:
    """The directories a sparse checkout ``tree`` holds; ``None`` when it is whole."""
    where = _git(tree, "rev-parse", "--git-path", "info/sparse-checkout").stdout.strip()
    if not where:
        return None
    path = Path(where) if os.path.isabs(where) else Path(tree) / where
    if not path.is_file():
        return None
    listed = _git(
        tree, "-c", "core.sparseCheckout=true", "-c", "core.sparseCheckoutCone=true",
        "sparse-checkout", "list",
    )  # fmt: skip
    if listed.returncode != 0:
        return None
    dirs = tuple(sorted(line.strip() for line in listed.stdout.splitlines() if line.strip()))
    return dirs or None


def _union(a: Sequence[str] | None, b: Sequence[str] | None) -> tuple[str, ...] | None:
    """Both cones in one; ``None`` (the whole tree) holds every cone."""
    if a is None or b is None:
        return None
    dirs = set(a) | set(b)
    return tuple(sorted(d for d in dirs if not any(o != d and d.startswith(o + "/") for o in dirs)))


def _covers(have: Sequence[str] | None, want: Sequence[str] | None) -> bool:
    if have is None:
        return True
    if want is None:
        return False
    return all(any(w == h or w.startswith(h + "/") for h in have) for w in want)


# ------------------------------------------------------------------- manifests


def manifest_path(repo: str | Path, tree: str | Path) -> Path:
    digest = hashlib.sha1(os.path.normpath(str(tree)).encode("utf-8")).hexdigest()[:20]
    return common_dir(repo) / MANIFEST_DIRNAME / f"{digest}.json"


def read_manifest(repo: str | Path, tree: str | Path) -> dict | None:
    try:
        data = json.loads(manifest_path(repo, tree).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def publish(
    repo: str | Path, tree: str | Path, cone: Sequence[str] | None
) -> tuple[str, ...] | None:
    """Record what commit tree ``tree`` checks out, widened by ``cone``; the cone recorded."""
    commit = commit_of(tree)
    if commit is None:
        raise ValueError(f"{tree} is not a commit tree")
    path = manifest_path(repo, tree)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _flock(path.with_suffix(".lock")):
        known = read_manifest(repo, tree)
        cone = tuple(cone) if cone is not None else None
        if known is not None:
            have = known.get("cone")
            cone = _union(tuple(have) if isinstance(have, list) else None, cone)
            if (tuple(have) if isinstance(have, list) else None) == cone:
                return cone
        record = {"tree": os.path.normpath(str(tree)), "commit": commit, "cone": cone}
        tmp = path.with_name(f".{path.name}.{os.getpid()}")
        tmp.write_text(json.dumps(record) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    return cone


@contextlib.contextmanager
def _flock(path: Path, *, wait: bool = True) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# ------------------------------------------------------------------- commit trees


def ready(tree: str | Path) -> dict | None:
    """What a finished commit tree holds (``commit``, ``cone``); ``None`` when it is not one."""
    commit = commit_of(tree)
    try:
        data = json.loads((Path(tree) / READY_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("commit") != commit:
        return None
    return data


def materialize(
    tree: str | Path,
    repo: str | Path,
    cone: Sequence[str] | None | type = ...,
    *,
    log=lambda message: None,
) -> Path:
    """Make commit tree ``tree`` on this node from ``repo``'s objects; it, ready.

    ``cone`` is what it checks out (``None``: the whole tree), recorded for
    every node that makes it later; left out, the recorded cone is used, the
    whole tree when none was. A tree already here that holds less than asked
    is widened. One maker at a time per tree, across processes.
    """
    tree, repo = Path(tree), Path(repo)
    commit = commit_of(tree)
    if commit is None:
        raise LocalTreeError(f"{tree} is not a commit tree (<root>/snapshots|commits/<commit>)")
    if cone is ...:
        known = read_manifest(repo, tree)
        have = known.get("cone") if known else None
        want = tuple(have) if isinstance(have, list) else None
    else:
        want = publish(repo, tree, cone)  # type: ignore[arg-type]
    with _flock(tree.parent / _MAKE_LOCKS / f"{commit}.lock"):
        done = ready(tree)
        if done is not None:
            have = done.get("cone")
            have = tuple(have) if isinstance(have, list) else None
            if not _covers(have, want):
                _widen(tree, have, want, commit, log)
            with contextlib.suppress(OSError):
                os.utime(tree / READY_FILENAME)
            return tree
        _make(tree, repo, commit, want, log)
    _sweep(tree.parent, keep=commit)
    return tree


def _make(tree: Path, repo: Path, commit: str, cone: tuple[str, ...] | None, log) -> None:
    started = time.monotonic()
    objects = common_dir(repo) / "objects"
    if _git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        raise LocalTreeError(f"{repo} holds no commit {commit[:10]}")
    _remove(tree)
    tmp = tree.with_name(f".{tree.name}.{socket.gethostname()}-{os.getpid()}")
    _remove(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        _checked(tmp.parent, "init", "-q", str(tmp))
        (tmp / ".git" / "objects" / "info" / "alternates").write_text(f"{objects}\n")
        # Git LFS content comes from the source's store, as a worktree of it reads it.
        _checked(tmp, "config", "lfs.storage", str(objects.parent / "lfs"))
        _checked(tmp, "update-ref", "--no-deref", "HEAD", commit)
        if cone is not None:
            _checked(
                tmp, *CHECKOUT_FLAGS, "sparse-checkout", "set", "--cone", "--skip-checks", *cone
            )
        _checked(tmp, *CHECKOUT_FLAGS, "reset", "--hard", "-q", commit)
        link_native(repo, tmp)
        # The ready file and the native links are not the commit's: git status stays clean.
        exclude = [f"/{READY_FILENAME}", *(f"/{rel}" for rel in native_artifacts(repo))]
        (tmp / ".git" / "info").mkdir(exist_ok=True)
        (tmp / ".git" / "info" / "exclude").write_text("\n".join(exclude) + "\n")
        _read_only(tmp)
        (tmp / READY_FILENAME).write_text(
            json.dumps({"commit": commit, "cone": cone, "source": str(repo)}) + "\n"
        )
        os.rename(tmp, tree)
    except BaseException:
        _remove(tmp)
        raise
    how = f"sparse: {', '.join(cone)}" if cone is not None else "whole"
    log(f"commit tree {tree} made from {repo} in {time.monotonic() - started:.1f}s, {how}")


def _widen(tree: Path, have, want, commit: str, log) -> None:
    if want is None:
        _checked(tree, *CHECKOUT_FLAGS, "sparse-checkout", "disable")
    else:
        _checked(tree, *CHECKOUT_FLAGS, "sparse-checkout", "add", "--skip-checks", *want)
    _read_only(tree)
    record = ready(tree) or {}
    record.update(commit=commit, cone=_union(have, want) if want else None)
    (tree / READY_FILENAME).write_text(json.dumps(record) + "\n")
    log(f"commit tree {tree} widened to {', '.join(want) if want else 'the whole tree'}")


def _read_only(tree: Path) -> None:
    """Take write permission off every file in ``tree`` but git's own; symlinks stay as they are."""
    for directory, dirs, names in os.walk(tree):
        if directory == str(tree) and ".git" in dirs:
            dirs.remove(".git")
        for name in names:
            if directory == str(tree) and name == READY_FILENAME:
                continue
            path = os.path.join(directory, name)
            with contextlib.suppress(OSError):
                mode = os.lstat(path).st_mode
                if stat.S_ISREG(mode) and mode & 0o222:
                    os.chmod(path, mode & ~0o222)


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path, ignore_errors=True)


def remove(tree: str | Path) -> bool:
    """Remove commit tree ``tree`` unless another process is making it; whether it went."""
    tree = Path(tree)
    commit = commit_of(tree)
    if commit is None:
        raise ValueError(f"{tree} is not a commit tree")
    with _flock(tree.parent / _MAKE_LOCKS / f"{commit}.lock", wait=False) as locked:
        if not locked:
            return False
        _remove(tree)
    with contextlib.suppress(OSError):
        (tree.parent / _MAKE_LOCKS / f"{commit}.lock").unlink()
    return True


def _sweep(kind_dir: Path, keep: str, max_age: float = SWEEP_AGE_S) -> list[str]:
    """Remove the commit trees under ``kind_dir`` no job asked for in ``max_age``; their names.

    Each use of a tree touches its ready file, so one a job uses stays.
    """
    removed = []
    now = time.time()
    with contextlib.suppress(OSError):
        for tree in kind_dir.iterdir():
            if tree.name == keep or not _SHA.match(tree.name):
                continue
            try:
                age = now - (tree / READY_FILENAME).stat().st_mtime
            except OSError:
                continue
            if age > max_age and remove(tree):
                removed.append(tree.name)
    return removed


# ------------------------------------------------------------------- mutable trees


def freeze(tree: str | Path, root: str | Path | None = None) -> Path:
    """A commit tree of what mutable tree ``tree`` holds now, its cone recorded; its path.

    A clean tree is its HEAD. One with edits is a commit of them on top of
    HEAD, under ``refs/mv2flow/wip/<kind>-<name>``: tracked and untracked files
    alike, less what git ignores and the sandbox's own files. The tree's
    index and files are left as they were. The commit tree is not made here:
    the node that runs the job makes it (:func:`materialize`).
    """
    tree = Path(tree)
    parsed = parse(tree)
    if parsed is None or parsed[1] not in MUTABLE_KINDS:
        raise ValueError(f"{tree} is not a mutable node-local tree")
    root = Path(root) if root is not None else parsed[0]
    head = _checked(tree, "rev-parse", "--verify", "HEAD^{commit}")
    cone = cone_of(tree)
    excludes = [f":(exclude){p}" for p in FREEZE_EXCLUDES]
    edits = _checked(tree, "status", "--porcelain", "--untracked-files=all", "--", ".", *excludes)
    commit = head
    if edits:
        commit = _wip_commit(tree, head)
        if _checked(tree, "rev-parse", f"{commit}^{{tree}}") == _checked(
            tree, "rev-parse", "HEAD^{tree}"
        ):
            commit = head
        else:
            _checked(tree, "update-ref", wip_ref(tree_name(tree)), commit)
    target = commit_tree(root, "commits", commit)
    publish(tree, target, cone)
    return target


def _wip_commit(tree: Path, head: str) -> str:
    index = Path(_checked(tree, "rev-parse", "--path-format=absolute", "--git-path", "index"))
    scratch = index.with_name(f"mv2flow-freeze-{os.getpid()}.index")
    try:
        if index.exists():
            shutil.copyfile(index, scratch)
        env = {"GIT_INDEX_FILE": str(scratch), **_WIP_IDENTITY}
        excludes = [f":(exclude){p}" for p in FREEZE_EXCLUDES]
        _checked(tree, "add", "-A", "--", ".", *excludes, env=env)
        written = _checked(tree, "write-tree", env=env)
        message = f"mv2flow: uncommitted edits of {tree_name(tree)} on {head[:10]}"
        return _checked(tree, "commit-tree", written, "-p", head, "-m", message, env=env)
    finally:
        with contextlib.suppress(OSError):
            scratch.unlink()


def restore_wip(tree: str | Path, *, log=lambda message: None) -> bool:
    """Put back the edits a lost copy of mutable tree ``tree`` last froze; whether it did.

    Only onto the commit they were frozen on, and only into a tree that holds
    no edits of its own; they come back uncommitted.
    """
    tree = Path(tree)
    ref = wip_ref(tree_name(tree))
    wip = _git(tree, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").stdout.strip()
    if not wip:
        return False
    head = _checked(tree, "rev-parse", "HEAD")
    parent = _git(tree, "rev-parse", "--verify", "--quiet", f"{wip}^").stdout.strip()
    why = ""
    if parent != head:
        why = f"they are on {parent[:10]}, not on this tree's {head[:10]}"
    elif _checked(tree, "status", "--porcelain", "--untracked-files=no"):
        why = "this tree holds edits of its own"
    if why:
        log(f"{tree.name}: edits frozen in {ref} ({wip[:10]}) not put back: {why}")
        return False
    flags = ()
    if cone_of(tree) is not None:
        flags = ("-c", "core.sparseCheckout=true", "-c", "core.sparseCheckoutCone=true")
    _checked(tree, *flags, "read-tree", "-m", "-u", "HEAD", wip)
    _checked(tree, *flags, "reset", "-q")
    log(f"{tree.name}: restored the uncommitted edits frozen in {ref} ({wip[:10]})")
    return True


def lock_worktree(repo: str | Path, tree: str | Path) -> None:
    """Keep a node-local worktree's registration from a ``git worktree prune`` on another node."""
    reason = f"mv2flow node-local tree on {socket.gethostname()}"
    _git(repo, "worktree", "lock", "--reason", reason, str(tree))


def forget_worktree(repo: str | Path, tree: str | Path) -> None:
    """Drop the registration of a node-local worktree whose files are not on this node."""
    if Path(tree).exists():
        return
    _git(repo, "worktree", "unlock", str(tree))
    _git(repo, "worktree", "prune")


# ------------------------------------------------------------------- jobs


def job_tree(tree: str | Path | None) -> str | None:
    """The tree a job submitted against ``tree`` names: a mutable tree frozen, any other as is."""
    if not tree:
        return None
    if is_mutable_tree(tree):
        return str(freeze(tree))
    return str(tree)


def source_of(tree: str | Path, environ: Mapping[str, str] | None = None) -> str:
    """The checkout a commit tree for a job against node-local ``tree`` is made from; ``""``: unknown.

    The run's :data:`REPO_ENV`; else a mutable tree's main checkout, or the
    one a commit tree on this node was made from.
    """
    value = (os.environ if environ is None else environ).get(REPO_ENV, "")
    if value:
        return os.path.abspath(value)
    if is_mutable_tree(tree) and os.path.isdir(tree):
        common = common_dir(tree).resolve()
        return str(common.parent if common.name == ".git" else common)
    made = ready(tree) if is_commit_tree(tree) else None
    return str(made.get("source") or "") if made else ""


def ensure_command(
    tree: str, repo: str | None, command: Sequence[str], python: str | None = None
) -> list[str]:
    """``command`` preceded by making commit tree ``tree`` here and entering it."""
    argv = [python or sys.executable, str(Path(__file__).resolve()), "ensure"]
    if repo:
        argv += ["--repo", repo]
    return [*argv, tree, "--", *command]


def _ensure(args: argparse.Namespace) -> int:
    repo = args.repo or os.environ.get(REPO_ENV)
    if not repo:
        print(
            f"localtree: {args.tree} is not on this node, and no checkout to make it from "
            f"(--repo, or {REPO_ENV})",
            file=sys.stderr,
        )
        return 2
    try:
        materialize(args.tree, repo, log=lambda message: print(f"[localtree] {message}"))
    except (LocalTreeError, OSError) as exc:
        print(f"localtree: {exc}", file=sys.stderr)
        return 2
    command = [c for c in args.command if c != "--"] if args.command else []
    if not command:
        return 0
    sys.stdout.flush()
    os.chdir(args.tree)
    os.execvp(command[0], command)
    return 127  # not reached


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localtree.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    ensure = sub.add_parser(
        "ensure", help="make a commit tree on this node, then run a command in it"
    )
    ensure.add_argument("tree")
    ensure.add_argument(
        "--repo", help=f"the checkout whose objects it is made of; default ${REPO_ENV}"
    )
    ensure.add_argument("command", nargs=argparse.REMAINDER)
    root = sub.add_parser("root", help="print the node-local root of a workspace")
    root.add_argument("base")
    root.add_argument("workspace")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.action == "ensure":
        return _ensure(args)
    print(root_for(args.base, args.workspace))
    return 0


if __name__ == "__main__":
    sys.exit(main())
