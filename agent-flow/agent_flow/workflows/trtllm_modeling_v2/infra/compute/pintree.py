#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The node half of the tree seam: a pin, its build record, and the pinned tree on a node.

A **pin** is a source commit plus a **build record**: the content digest of each native library
of the tree, by its path in the tree. A commit does not carry the libraries git ignores, so a
commit alone does not say what a job ran. Where a job runs a pin, the libraries there are
checked against the record before the job starts, and every process reports the identity of
the libraries it found (``resolvehook``); a job whose libraries are not the record's has **no
result**.

**The identity of a build** (:func:`build_id`) is one name for a build record: by content,
never by a file's size or times. A file's digest is computed once per node and remembered
under the file's identity on disk (:class:`DigestCache`), so the processes of a job, and the
jobs after it, read each library's bytes once and not once each.

::

    <pins>/<id>.json                 a pin: commit, build record, where its tree goes on a node
    <tree>/.mv2flow-pin              in a pinned tree: the pin file it was made from
    <spool>/logs/<job>.tree.jsonl    per host of a job: the tree it ran and what was verified there
    <git-common-dir>/mv2flow-trees.json   where a checkout's run keeps its pins and its trees

**Where trees are.** There is one kind: on the node's own disk unless the spec states
another place: a pin's tree on the node its job runs on, under the same path on every node
(:func:`local_root`). A tree is a whole checkout whose git directory is its own and sits in the
tree (:func:`make_tree`): it holds ``HEAD``, the index and nothing else, and points at the
repository for objects, refs and configuration (git's ``commondir``). The repository is told
nothing of it, so the storage every node shares gains no file for a tree, only the objects and
refs of commits, and no ``git worktree prune`` anywhere can take a tree away. A pin's tree is
made once per node and used again while it is there; it is removed when no job that waits or
runs names its pin and it has not been used for a stated time (:func:`sweep`).

A job names its pin in its header (``#spool: pin=<pin file>``). The watcher starts it through
:func:`run`: the pin's tree is made on this node (:func:`at`), the libraries are checked, and
the job's command runs inside the tree.

A tree nobody pinned yet is pinned where the job is submitted (:func:`pin_tree`): what the
tree holds then, uncommitted edits included, becomes a commit no branch moves, so a job never
runs a tree that changes under it.

Standard-library only and runnable by path. It is also loaded by path into whatever
interpreter a job's process is (``resolvehook``) and run by a holder's lead under the batch
host's own, so it is written for Python 3.6 and newer: no annotations, nothing later::

    pintree.py run --pin FILE --job NAME [--record FILE] -- <command...>
"""

import argparse
import contextlib
import errno
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shutil
import socket
import stat as stat_mod
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

#: Shared-library name patterns: what a build record digests.
LIBRARY_PATTERNS = ("*.so", "*.so.*")
#: The packages whose libraries a record covers: the modules a job is watched for.
MODULES_ENV = "MV2FLOW_RESOLVE_MODULES"
DEFAULT_MODULES = ("tensorrt_llm",)

#: In a pinned tree, written last: the pin file it was made from. A tree without it was cut short.
PIN_MARKER = ".mv2flow-pin"
#: Under a place for trees: the pin files and the pinned trees.
PINS_DIRNAME = "pins"
#: Under what every node sees of a run's trees: the digests of the libraries, kept by file.
BUILDS_DIRNAME = "builds"
#: Beside the pins directory: per landed task, the commit its branch landed as.
INTEGRATED_FILENAME = "integrated.json"
#: In a checkout's git directory: where its run keeps trees.
CONFIG_FILENAME = "mv2flow-trees.json"
#: A pin of a tree that was not committed is kept alive under this ref prefix.
PIN_REF_PREFIX = "refs/mv2flow/pins"

#: What the job's processes are told: the pin, its tree, and the libraries the record names.
PIN_ENV = "MV2FLOW_PIN"
TREE_ENV = "MV2FLOW_TREE"
LIBS_ENV = "MV2FLOW_TREE_LIBS"

#: The prelude's exit code when the pin's tree could not be made, or its libraries are not the
#: record's: the job did not run, and has no result.
RC_NO_TREE = 221

#: A digest is remembered only for a file last changed at least this long ago: a file changed
#: twice within the clock's grain would otherwise keep the first digest.
RACY_S = 2.0
#: Where a node remembers digests, unless this names another file.
CACHE_ENV = "MV2FLOW_DIGEST_CACHE"
#: Files one checkout writes at once (``checkout.workers``, git 2.32 or later).
CHECKOUT_WORKERS = 16
#: Where trees go when the spec states no place: this directory on each node's own disk,
#: unless the environment names another.
LOCAL_BASE = "/tmp"
LOCAL_BASE_ENV = "MV2FLOW_TREES_DIR"
#: A pin's tree that nobody holds is removed once unused for this long.
DEFAULT_PIN_AGE_S = 1800.0

_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
#: What a pinned tree's directory is called: its pin's id (commit and build), or a bare commit.
_KEY = re.compile(r"^(?:[0-9a-f]{40}(?:[0-9a-f]{24})?|[0-9a-f]{12}-[0-9a-f]{16})$")
#: What an operator's sandbox leaves in every directory its shell enters: files, and directories.
_SANDBOX_NAMES = (".mcp.json",)
_SANDBOX_DIRS = (".claude", ".cc-writes")


class PinError(RuntimeError):
    """A pin could not be made, read, or turned into a tree here."""


# ------------------------------------------------------------------ files


def _read_json(path):
    try:
        with open(str(path), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(".%s.%d.%s.tmp" % (path.name, os.getpid(), uuid.uuid4().hex[:8]))
    with open(str(staging), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(str(staging), str(path))


def git_env(env=None):
    """The environment of a git command that may write a tree: ``env``, else this process's.

    A tree is made of what the repository itself holds. A git-lfs file is its pointer there,
    on every host: where git is set up to fill such files in, making a tree would fetch them
    (a node has no network, and the fetch would write the repository every node shares) or
    fail on the first one. ``GIT_LFS_SKIP_SMUDGE=0`` in the environment asks for the files.
    """
    env = dict(os.environ if env is None else env)
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    return env


def _git(cwd, *args, **kwargs):
    env = git_env(kwargs.pop("env", None))
    # No optional locks: a question asked of a worker's tree must not take its index lock.
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(cwd)] + list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
    )


def _git_out(cwd, *args, **kwargs):
    done = _git(cwd, *args, **kwargs)
    if done.returncode != 0:
        raise PinError("git %s: %s" % (" ".join(args), done.stderr.strip() or "failed"))
    return done.stdout.strip()


def _last_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "no message"


def modules():
    """The packages whose libraries a build record covers."""
    said = [m.strip() for m in os.environ.get(MODULES_ENV, "").split(",") if m.strip()]
    return tuple(said) or DEFAULT_MODULES


# ----------------------------------------------------------- build records


def libraries(tree, packages=None):
    """Each shared library under the tree's packages, as a path relative to the tree, sorted.

    A linked directory is followed: a worktree links the checkout's native build.
    """
    tree = str(tree)
    found = set()
    for package in packages or modules():
        top = os.path.join(tree, *package.split("."))
        seen = set()
        for directory, names, files in os.walk(top, followlinks=True):
            real = os.path.realpath(directory)
            if real in seen:  # a link that leads back: walked once
                names[:] = []
                continue
            seen.add(real)
            names[:] = [n for n in names if n != "__pycache__"]
            for name in files:
                if any(fnmatch.fnmatch(name, pattern) for pattern in LIBRARY_PATTERNS):
                    path = os.path.join(directory, name)
                    if os.path.isfile(path):
                        found.add(os.path.relpath(path, tree).replace(os.sep, "/"))
    return sorted(found)


def unread(tree, packages=None):
    """The links under the tree's packages that lead nowhere on this host, as paths in the tree.

    A checkout whose native build is a link into an image shows no library on a host outside
    that image: a pin taken there cannot record them, and says which links it could not follow.
    """
    tree = str(tree)
    found = set()
    for package in packages or modules():
        top = os.path.join(tree, *package.split("."))
        for directory, names, files in os.walk(top):
            names[:] = [n for n in names if n != "__pycache__"]
            for name in names + files:
                path = os.path.join(directory, name)
                if os.path.islink(path) and not os.path.exists(path):
                    found.add(os.path.relpath(path, tree).replace(os.sep, "/"))
    return sorted(found)


def digest_file(path):
    """The sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


class DigestCache:
    """Digests remembered by a file's identity on disk: inode, size, and both of its times.

    Rewriting a file changes its change time whatever is done to its modification time, and a
    file renamed into place has another inode, so a remembered digest is never one of other
    bytes. Without a file (``path`` None) nothing is remembered between processes.
    :func:`default_cache` is the file a node's processes share.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._known = (_read_json(self.path) or {}) if self.path else {}
        self._dirty = False

    @staticmethod
    def _key(path, st):
        return "%s|%d|%d|%d|%d" % (
            os.path.realpath(path),
            st.st_ino,
            st.st_size,
            st.st_mtime_ns,
            st.st_ctime_ns,
        )

    def digest(self, path):
        st = os.stat(path)
        key = self._key(path, st)
        known = self._known.get(key)
        if known:
            return known
        value = digest_file(path)
        if time.time() - max(st.st_mtime, st.st_ctime) >= RACY_S:
            self._known[key] = value
            self._dirty = True
        return value

    def put(self, path, value):
        """Remember ``value`` for a file whose bytes the caller just wrote itself."""
        try:
            self._known[self._key(path, os.stat(path))] = value
            self._dirty = True
        except OSError:
            pass

    def save(self):
        """Write what was learnt, over what other processes wrote meanwhile."""
        if self.path and self._dirty:
            try:
                merged = _read_json(self.path) or {}
                merged.update(self._known)
                _write_json(self.path, merged)
            except OSError:
                pass
            self._dirty = False


def default_cache():
    """The digest cache of this node and user: ``$MV2FLOW_DIGEST_CACHE``, else a file in /tmp."""
    path = os.environ.get(CACHE_ENV)
    if not path:
        path = os.path.join(tempfile.gettempdir(), "mv2flow-digests-%d.json" % os.getuid())
    return DigestCache(path)


def build_record(tree, packages=None, cache=None):
    """``{library path in the tree: sha256 of its bytes}`` for each native library of ``tree``.

    By content and by the library's place in the tree, nothing else: the same bytes copied to
    another disk, at another time, are the same build.
    """
    cache = cache or DigestCache()
    record = {}
    for name in libraries(tree, packages):
        try:
            record[name] = cache.digest(os.path.join(str(tree), name))
        except OSError:
            continue
    cache.save()
    return record


def record_digest(record):
    """One short name for a build record."""
    text = "\n".join("%s %s" % (name, record[name]) for name in sorted(record))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def record_diff(expected, found):
    """How ``found`` differs from the ``expected`` record, one phrase per library; [] if equal."""
    out = []
    for name in sorted(set(expected) | set(found)):
        want, have = expected.get(name), found.get(name)
        if want == have:
            continue
        if have is None:
            out.append("%s is missing (the record has %s)" % (name, want[:12]))
        elif want is None:
            out.append("%s is not in the record (found %s)" % (name, have[:12]))
        else:
            out.append("%s is %s, the record has %s" % (name, have[:12], want[:12]))
    return out


def build_id(tree, names=None, packages=None, cache=None):
    """The identity of the native build in ``tree``: the name of its build record, by content.

    Of the libraries ``names`` (paths in the tree, as a pin's record names them; one that is
    not there counts as missing), else of every library found under the tree's packages.
    ``None`` when nothing names a library and none is found: there is no build to identify.
    This is what each process of a job reports (``resolvehook``): equal to
    ``record_digest(pin["build"])`` exactly when the libraries it found are the pin's.
    """
    cache = cache or default_cache()
    if names is None:
        record = build_record(tree, packages, cache)
        return record_digest(record) if record else None
    record = {}
    for name in names:
        try:
            record[name] = cache.digest(os.path.join(str(tree), name))
        except OSError:
            record[name] = "missing"
    cache.save()
    return record_digest(record)


# -------------------------------------------------------------- pin files


def pin_id(commit, record):
    return "%s-%s" % (commit[:12], record_digest(record))


def write_pin(directory, pin):
    """Write ``pin`` under ``directory`` and return its file. A pin that was given out never changes.

    The file is ``<id>.json``, the id being the pin's commit and build. When a file of that
    id already says something else (the same source and build, placed elsewhere: the run's
    place for trees was moved), the old file stays as it is, for the jobs that name it, and
    this pin gets a file of its own.
    """
    pin = dict(pin)
    pin["id"] = pin_id(pin["commit"], pin.get("build") or {})
    path = Path(directory) / ("%s.json" % pin["id"])
    known = _read_json(path)
    if known is not None and known != pin:
        text = json.dumps(pin, sort_keys=True).encode()
        path = path.with_name("%s-%s.json" % (pin["id"], hashlib.sha256(text).hexdigest()[:8]))
        known = _read_json(path)
    if known != pin:
        _write_json(path, pin)
    return path


def read_pin(path):
    pin = _read_json(path)
    if not pin or not _COMMIT.match(str(pin.get("commit") or "")):
        raise PinError("%s is not a pin file" % path)
    if not isinstance(pin.get("build"), dict):
        raise PinError("%s carries no build record" % path)
    return pin


def pin_file_of(tree):
    """The pin file a pinned tree was made from; ``None`` for any other directory."""
    marker = _read_json(os.path.join(str(tree), PIN_MARKER))
    path = (marker or {}).get("pin")
    return str(path) if path and os.path.isfile(str(path)) else None


def config_path(checkout):
    """Where a checkout's run says how it keeps trees: a file in its git directory."""
    common = _git(checkout, "rev-parse", "--git-common-dir")
    if common.returncode != 0 or not common.stdout.strip():
        return None
    path = Path(common.stdout.strip())
    if not path.is_absolute():
        path = Path(str(checkout)) / path
    return path / CONFIG_FILENAME


def read_config(tree):
    """The trees configuration of the checkout ``tree`` belongs to; ``None`` when it has none."""
    path = config_path(tree)
    return _read_json(path) if path else None


def write_config(checkout, config):
    path = config_path(checkout)
    if path is None:
        raise PinError("%s is not a git checkout" % checkout)
    if _read_json(path) != config:
        _write_json(path, config)
    return path


# ------------------------------------------------------------ a tree, on a node


def is_sandbox_placeholder(path):
    parts = path.strip("/").split("/")
    return parts[-1] in _SANDBOX_NAMES or any(part in _SANDBOX_DIRS for part in parts[:-1])


def native_artifacts(repo, packages=None):
    """Entries git ignores directly under ``repo``'s packages: the native build a worktree links."""
    found = []
    for package in packages or modules():
        top = "/".join(package.split("."))
        done = _git(
            repo,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--directory",
            "--",
            top + "/",
        )
        for line in done.stdout.splitlines():
            parts = line.strip().rstrip("/").split("/")
            depth = len(top.split("/"))
            if len(parts) != depth + 1 or "/".join(parts[:depth]) != top:
                continue
            if parts[-1] == "__pycache__" or parts[-1].endswith(".egg-info"):
                continue
            found.append("/".join(parts))
    return found


def link_native(repo, tree):
    """Link the checkout's native build into ``tree``: every worktree runs the one shared build."""
    for rel in native_artifacts(repo):
        target = Path(str(tree)) / rel
        if target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(Path(str(repo)) / rel)


@contextlib.contextmanager
def repo_lock(repo):
    """One at a time on a repository: a task's branch is made, a task lands."""
    common = Path(_git_out(repo, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = Path(str(repo)) / common
    with open(str(common / "mv2flow-worktree.lock"), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def freeze(tree, name=""):
    """The commit of what ``tree`` holds now: its HEAD, or a commit of its edits on top of it.

    Nothing in the tree changes: not its branch, its index or its files. Edits (tracked,
    untracked and not ignored) are committed through a temporary index and kept alive under
    :data:`PIN_REF_PREFIX`. The temporary index starts as a copy of the tree's own, so git
    reads again only the files that changed since, not the whole tree. Left out are the
    sandbox's placeholders and a link the tree's HEAD lacks that leads out of the tree (the
    link to the native build); a link the task added to a file of the tree is source, and is
    pinned. The commit carries its parent's date and a fixed
    message, so the same files on the same HEAD are the same commit however often they are
    pinned. The ref is ``branch/<the tree's branch>`` under :data:`PIN_REF_PREFIX`, so a tree
    lost with its disk is made again from its branch and that last pin; a tree on no branch is
    kept under ``name/<name>``. The two never meet, whatever a job or a branch is called.
    """
    head = _git_out(tree, "rev-parse", "HEAD")
    branch = _git(tree, "symbolic-ref", "-q", "--short", "HEAD").stdout.strip()
    status = _git_out(tree, "status", "--porcelain", "--untracked-files=all")
    if not status.strip():
        if branch:
            _git_out(tree, "update-ref", "%s/branch/%s" % (PIN_REF_PREFIX, branch), head)
        return head
    with tempfile.TemporaryDirectory(prefix="mv2flow-pin-") as scratch:
        env = dict(os.environ)
        env["GIT_INDEX_FILE"] = os.path.join(scratch, "index")
        env.setdefault("GIT_AUTHOR_NAME", "mv2flow pin")
        env.setdefault("GIT_AUTHOR_EMAIL", "mv2flow-pin@localhost")
        env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
        env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = _git_out(
            tree, "show", "-s", "--format=%cI", "HEAD"
        )
        own = _git(tree, "rev-parse", "--git-path", "index").stdout.strip()
        own = own if os.path.isabs(own) else os.path.join(str(tree), own)
        if own and os.path.isfile(own):
            # With its time: git reads again, by content, every file as young as its index.
            # A copy stamped now would make an edit of the checkout's own second, of the
            # same size, look unchanged.
            shutil.copy2(own, env["GIT_INDEX_FILE"])
        else:
            _git_out(tree, "read-tree", "HEAD", env=env)
        _git_out(tree, "add", "-A", env=env)
        raw = _git_out(tree, "diff", "--cached", "--raw", "--no-renames", "-z", "HEAD", env=env)
        fields = raw.split("\0")
        drop = []
        for meta, path in zip(fields[0::2], fields[1::2]):
            modes = meta.split()
            added_link = len(modes) >= 2 and modes[1] == "120000" and modes[0] != ":120000"
            if (added_link and _leads_out(tree, path)) or is_sandbox_placeholder(path):
                drop.append(path)
        if drop:
            _git_out(tree, "rm", "-q", "--cached", "--ignore-unmatch", "--", *drop, env=env)
        written = _git_out(tree, "write-tree", env=env)
        if written == _git_out(tree, "rev-parse", "HEAD^{tree}"):
            commit = head
        else:
            commit = _git_out(
                tree, "commit-tree", written, "-p", head, "-m", "mv2flow pin", env=env
            )
    if branch:
        _git_out(tree, "update-ref", "%s/branch/%s" % (PIN_REF_PREFIX, branch), commit)
    elif commit != head:
        label = re.sub(r"[^A-Za-z0-9._-]", "-", name or commit[:12])
        _git_out(tree, "update-ref", "%s/name/%s" % (PIN_REF_PREFIX, label), commit)
    return commit


def _leads_out(tree, path):
    """Whether the link at ``path`` in ``tree`` points outside the tree (or cannot be read)."""
    link = os.path.join(str(tree), path)
    try:
        target = os.readlink(link)
    except OSError:
        return True
    if os.path.isabs(target):
        return True
    inside = os.path.normpath(os.path.join(os.path.dirname(path), target))
    return inside == ".." or inside.startswith("../")


def last_pin(repo, branch):
    """The last pin of the tree on ``branch``, when it is that branch's tip plus edits; else ``None``.

    ``(pin commit, the files it changes)``: what a tree made again on the branch gets back.
    A pin taken before the branch's last commit says nothing the branch does not.
    """
    ref = "%s/branch/%s^{commit}" % (PIN_REF_PREFIX, branch)
    pin = _git(repo, "rev-parse", "-q", "--verify", ref)
    tip = _git(repo, "rev-parse", "-q", "--verify", "refs/heads/%s^{commit}" % branch)
    pin, tip = pin.stdout.strip(), tip.stdout.strip()
    if not pin or not tip or pin == tip:
        return None
    if _git(repo, "rev-parse", "-q", "--verify", pin + "^").stdout.strip() != tip:
        return None
    changed = _git(repo, "diff", "--name-only", "--no-renames", tip, pin).stdout.split("\n")
    return pin, [name for name in changed if name]


def local_root(workspace):
    """Where a run keeps its trees on a node's own disk: the same path on every node.

    Named for the workspace and its path, so two runs never meet, under :data:`LOCAL_BASE`
    (``$MV2FLOW_TREES_DIR`` names another base).
    """
    workspace = os.path.abspath(str(workspace))
    name = re.sub(r"[^A-Za-z0-9._-]", "-", os.path.basename(workspace.rstrip("/"))) or "run"
    digest = hashlib.sha256(workspace.encode()).hexdigest()[:8]
    user = os.environ.get("USER") or str(os.getuid())
    base = os.environ.get(LOCAL_BASE_ENV) or LOCAL_BASE
    return os.path.join(base, "mv2flow-trees-%s" % user, "%s-%s" % (name, digest))


def common_dir(repo):
    """The repository's git directory, the one every tree of it shares, as an absolute path."""
    path = _git_out(repo, "rev-parse", "--git-common-dir")
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(str(repo), path))


def make_tree(repo, tree, head):
    """Check the whole of ``repo`` out at ``tree``, with a git directory of the tree's own.

    ``head`` is a commit (the tree is detached there) or ``ref: refs/heads/<branch>``. The
    tree's ``.git`` is a directory on the tree's disk holding ``HEAD`` and the index, and a
    ``commondir`` line naming the repository: objects, refs, configuration and hooks are the
    repository's, and a commit made in the tree lands there. Nothing is written to the
    repository to make the tree, and nothing there lists it.
    """
    tree = str(tree)
    gitdir = os.path.join(tree, ".git")
    os.makedirs(gitdir)
    with open(os.path.join(gitdir, "commondir"), "w", encoding="utf-8") as handle:
        handle.write(common_dir(repo) + "\n")
    with open(os.path.join(gitdir, "HEAD"), "w", encoding="utf-8") as handle:
        handle.write(head + "\n")
    done = _git(tree, "-c", "checkout.workers=%d" % CHECKOUT_WORKERS, "reset", "--hard", "-q")
    if done.returncode != 0:
        raise PinError("checking out %s at %s: %s" % (head[:60], tree, _last_line(done.stderr)))


def pins_root(root):
    return Path(str(root)) / PINS_DIRNAME


def pinned_tree(root, key):
    """Where a pin's read-only tree is kept under the place ``root``.

    ``key`` is the pin's id, so the same source with another build is another directory and
    a directory stands for one pin for as long as it exists; a bare commit names the tree of a
    caller that holds no pin file.
    """
    return pins_root(root) / key


def _lock_path(root, key):
    return pins_root(root) / ".locks" / ("%s.lock" % key)


def pinned_commit(tree):
    """The commit whose pinned tree ``tree`` is, by the marker written when it was made; else ``""``."""
    commit = str((_read_json(os.path.join(str(tree), PIN_MARKER)) or {}).get("commit") or "")
    return commit if _COMMIT.match(commit) else ""


def _ready(tree, commit):
    return pinned_commit(tree) == commit


def _intact(tree, commit):
    """Whether a pinned tree still holds its commit: no tracked file changed, replaced or gone.

    Asked each time a tree is used again, so it must cost next to nothing: git compares each
    file's stats with its index entry and reads none. Two things make that hold. The index
    was settled when the tree was made (:func:`settle_index`). And a file's change time is
    left out of the comparison: making the tree read-only moved it for every file, after
    the index was written, and git would otherwise read all of them again, through their
    filters, every time (half a minute for the engine's tree). What is still seen: a file
    edited in place (its size or modification time), one replaced through its directory
    (its inode), one removed. What is not: bytes rewritten in place with the same size and
    the modification time put back, which nothing does by accident.
    """
    head = _git(tree, "rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != commit:
        return False
    status = _git(
        tree,
        "-c",
        "core.trustctime=false",
        "--no-optional-locks",
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    return status.returncode == 0 and not status.stdout.strip()


def settle_index(tree, wait):
    """Make git trust a fresh checkout's index by stats alone, without reading a file again.

    Git takes a file for possibly changed, and reads it (through its filters) on every
    question, while the file's time is not older than the index's own, by the second: after a
    checkout that is most of the tree, until the index is next written. Nobody has edited the
    tree yet, so the index's time is simply set after every file's. With ``wait`` (a tree
    somebody will edit) the second the checkout ended in is waited out first, so that no later
    edit can share a file's second and go unseen; a read-only tree needs no wait.
    """
    index = os.path.join(str(tree), ".git", "index")
    if wait:
        time.sleep(1.02 - time.time() % 1.0)
    stamp = time.time() + (0.0 if wait else 2.0)
    with contextlib.suppress(OSError):
        os.utime(index, (stamp, stamp))


def _read_only(tree):
    """Take write permission off every file in ``tree``; directories keep it, for bytecode caches.

    A symlink is left alone (it is the checkout's native build), and so is the tree's own git
    directory.
    """
    for directory, names, files in os.walk(str(tree)):
        if directory == str(tree):
            names[:] = [name for name in names if name != ".git"]
        for name in files:
            path = os.path.join(directory, name)
            with contextlib.suppress(OSError):
                mode = os.lstat(path).st_mode
                if stat_mod.S_ISREG(mode) and mode & 0o222:
                    os.chmod(path, mode & ~0o222)


def _discard(tree):
    """Remove ``tree``, whatever state a cut-short making left it in."""
    if os.path.lexists(str(tree)):
        for directory, names, _ in os.walk(str(tree)):
            for name in names:
                with contextlib.suppress(OSError):
                    os.chmod(os.path.join(directory, name), 0o755)
        shutil.rmtree(str(tree), ignore_errors=True)


def _make(repo, tree, commit, pin_file):
    """Check ``commit`` out whole at ``tree``: detached, the native build linked, read-only."""
    _discard(tree)
    Path(str(tree)).parent.mkdir(parents=True, exist_ok=True)
    try:
        make_tree(repo, tree, commit)
    except (PinError, OSError):
        _discard(tree)
        raise
    link_native(repo, tree)
    _read_only(tree)
    settle_index(tree, wait=False)
    _write_json(os.path.join(str(tree), PIN_MARKER), {"commit": commit, "pin": str(pin_file)})


def hold(root, key, repo=None, pin_file="", commit=None):
    """An open file whose shared lock holds the tree ``key`` under ``root``, made first if missing.

    ``key`` is the tree's directory (:func:`pinned_tree`): a pin's id, with ``commit`` its
    commit, or a bare commit. Whoever reads a pinned tree on this node holds it this way;
    :func:`sweep` removes only a tree nobody holds. The lock goes with the process that holds
    the file, so nothing is left to expire, and the lock file itself is never removed: a
    reader and a sweeper always meet on the same one. With ``repo`` a missing tree is made
    (one maker at a time) and one that is there is used again, once it is seen to hold its
    commit still (:func:`_intact`); without, a missing tree is an error. Each hold marks the
    tree used.
    """
    commit = commit or key
    tree, path = pinned_tree(root, key), _lock_path(root, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise unusable(root, exc)
    made = 0
    for _ in range(5):
        handle = open(str(path), "a")
        with contextlib.suppress(OSError):
            os.utime(str(path), None)
        fcntl.flock(handle, fcntl.LOCK_SH)
        if _ready(tree, commit) and _intact(tree, commit):
            return handle
        fcntl.flock(handle, fcntl.LOCK_UN)
        if repo is None:
            handle.close()
            raise PinError("no tree of %s at %s" % (commit[:10], tree))
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            # Missing, cut short, or no longer its commit: made again, never trusted as it is.
            if not (_ready(tree, commit) and _intact(tree, commit)):
                _make(repo, tree, commit, pin_file)
                made += 1
        except OSError as exc:
            raise unusable(root, exc)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
    raise PinError(
        "the tree of %s at %s was made %d time(s) and each time was gone again or did not hold "
        "its commit: `git -C %s status` says what differs" % (commit[:10], tree, made, tree)
    )


def unusable(root, exc):
    """One line for a place for trees that cannot be used, as the error a caller reports."""
    why = getattr(exc, "strerror", None) or str(exc)
    return PinError(
        "the place for trees %s cannot be used (%s): state another one in "
        "environment.trees.root" % (root, why)
    )


def sweep(root, keep=(), age_s=0.0, now=None):
    """Remove the pinned trees under ``root`` on this node that are done with; their commits.

    A tree stays while any of these holds: it is of a commit or a pin in ``keep``; a process
    on this node holds it; it was last used less than ``age_s`` ago. A job that waits is
    not asked: the node it starts on makes the tree again, in seconds, if it is gone.
    """
    directory = pins_root(root)
    removed = []
    if not directory.is_dir():
        return removed
    now = time.time() if now is None else now
    kept = set(keep)
    prefixes = tuple(commit[:12] + "-" for commit in keep)
    for tree in sorted(directory.iterdir()):
        name = tree.name
        if not _KEY.match(name) or not tree.is_dir():
            continue
        if name in kept or (prefixes and name.startswith(prefixes)):
            continue
        path = _lock_path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            used = os.stat(str(path)).st_mtime
        except OSError:
            used = os.stat(str(tree)).st_mtime
        if now - used < age_s:
            continue
        with open(str(path), "a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EACCES):
                    continue
                raise
            try:
                marker = _read_json(os.path.join(str(tree), PIN_MARKER)) or {}
                _discard(tree)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        removed.append(str(marker.get("commit") or name))
    return removed


def step_argv(pin_file, job, record_file, python):
    """The prefix under which one step of a job of several nodes runs its command in the pin.

    Each node of such a job makes (or finds) the pin's tree itself and checks its libraries
    there: ``<prefix> <command...>`` runs the command inside that tree.
    """
    argv = [
        str(python),
        os.path.abspath(__file__),
        "run",
        "--pin",
        str(pin_file),
        "--job",
        str(job),
    ]
    return argv + (["--record", str(record_file)] if record_file else []) + ["--"]


def record(config, commit, build, unread=()):
    """A pin as its file holds it: the commit, the build record, and where a node makes its tree.

    ``config`` is the run's (:func:`read_config`): ``repo`` the repository every node sees,
    ``root`` where trees go on each node, ``keep_s`` how long an unused pin's tree is kept.
    ``unread`` are the links the build record could not follow where the pin is taken
    (:func:`unread`): the libraries below them are recorded by each node a job runs on.
    """
    root = config["root"]
    pin = {
        "commit": commit,
        "build": build,
        "repo": config["repo"],
        "root": root,
        "tree": str(pinned_tree(root, pin_id(commit, build))),
        "keep_s": float(config.get("keep_s", DEFAULT_PIN_AGE_S)),
    }
    if unread:
        pin["unread"] = list(unread)
    return pin


def git_pin(tree, config, name=""):
    """Pin ``tree``: the commit of what it holds, and its build record.

    ``pins`` of the run's config is where it keeps what every node must see (default ``root``).
    """
    commit = freeze(tree, name)
    shared = config.get("pins") or config["root"]
    cache = DigestCache(Path(shared) / BUILDS_DIRNAME / "digests.json")
    return record(config, commit, build_record(tree, cache=cache), unread(tree))


def at(pin, job=""):
    """``pin``'s tree on this node, a detached whole tree held for this process; its path.

    Made here the first time a job of the pin runs on this node, from the repository every
    node sees, and used again by every later one. Before it, the trees on this node that
    nobody holds and that went unused for the pin's ``keep_s`` are removed.
    The lock that holds the tree is left open and inheritable: the job's processes keep it
    until the last of them exits.
    """
    commit, root = pin["commit"], pin["root"]
    key = pin.get("id") or pin_id(commit, pin.get("build") or {})
    with contextlib.suppress(OSError):
        sweep(root, keep={key}, age_s=float(pin.get("keep_s", DEFAULT_PIN_AGE_S)))
    handle = hold(root, key, repo=pin["repo"], pin_file=pin.get("file", ""), commit=commit)
    os.set_inheritable(handle.fileno(), True)
    _HELD.append(handle)
    return str(pinned_tree(root, key))


#: Locks this process holds on pinned trees, kept open until it exits or becomes the job.
_HELD = []


# ------------------------------------------------------------ pin, at, run


def pin_tree(tree, name=""):
    """Pin the tree a job is about to run in; the pin file, or ``None`` for a tree nothing pins.

    A tree that is itself a pin's is its pin. Any other tree of a checkout whose run keeps
    trees is pinned for that run. A directory outside any such checkout (not a git tree, or
    one no run configured) has nothing that could pin it: the job runs it as it is, and its
    receipt names no pin.
    """
    known = pin_file_of(tree)
    if known:
        return known
    config = read_config(tree)
    if not config or not config.get("root"):
        return None
    pin = git_pin(tree, config, name)
    return str(write_pin(pins_root(config.get("pins") or config["root"]), pin))


def verify(pin, tree):
    """Check ``tree``'s libraries against ``pin``'s build record; ``(ok, detail, build id)``.

    The digests are the node's (:func:`default_cache`): a library this node already read, and
    that has not changed since, is not read again, here or by the job's processes.

    A library below a link the pin could not follow where it was taken (its ``unread``) is in
    no record to compare with. It is recorded here instead: ``pin["build"]`` is completed with
    what this node found there, and the identity returned is that completed record's, which
    every process of the job on this node reports and every node of the job must agree on
    (``provenance.check_pinned``).
    """
    expected, found = pin["build"], build_record(tree, cache=default_cache())
    links = [link.rstrip("/") + "/" for link in pin.get("unread") or ()]
    here = {
        name: digest
        for name, digest in found.items()
        if name not in expected and (name + "/").startswith(tuple(links))
    }
    differ = record_diff(expected, {n: d for n, d in found.items() if n not in here})
    if differ:
        shown = "; ".join(differ[:4]) + (" and %d more" % (len(differ) - 4) if differ[4:] else "")
        detail = "the libraries of %s are not those its pin records: %s" % (tree, shown)
        return False, detail, ""
    if here:
        pin["build"] = {**expected, **here}
        pin["recorded_here"] = sorted(here)
    return True, "", record_digest(pin["build"])


def _note(record_file, entry):
    """Append one line about this host's tree to the job's tree record."""
    if not record_file:
        return
    entry = dict(entry, host=socket.gethostname(), time=time.time())
    try:
        Path(record_file).parent.mkdir(parents=True, exist_ok=True)
        with open(str(record_file), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError:
        pass


def read_tree_records(path):
    """Every well-formed line of a job's tree record; a missing file has none."""
    out = []
    try:
        with open(str(path), encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            out.append(entry)
    return out


def _say(message):
    sys.stderr.write("[mv2flow-tree] %s\n" % message)
    sys.stderr.flush()


def run(pin_file, job, record_file, argv, steps=False):
    """Make the pin's tree here, check its libraries, and become ``argv`` inside it.

    Never returns on success. On failure it returns :data:`RC_NO_TREE` after writing why to
    the job's tree record: the job has no result. With ``steps`` (a job of several nodes) no
    tree is made here: each step makes its own, through this same command.
    """
    started = time.time()
    try:
        pin = read_pin(pin_file)
        pin["file"] = str(pin_file)
        os.environ[PIN_ENV] = str(pin_file)
        if steps:
            _say("job %s spans nodes: each step makes the tree of %s itself" % (job, pin["id"]))
            os.execvp(argv[0], argv)
        tree = at(pin, job)
    except (PinError, OSError, ValueError, subprocess.SubprocessError) as exc:
        _note(record_file, {"ok": False, "detail": str(exc), "pin": str(pin_file)})
        _say("no tree for job %s: %s" % (job, exc))
        return RC_NO_TREE
    try:
        ok, detail, build = verify(pin, tree)
    except (PinError, OSError) as exc:
        ok, detail, build = False, str(exc), ""
    entry = {
        "ok": ok,
        "detail": detail,
        "pin": str(pin_file),
        "commit": pin["commit"],
        "tree": tree,
        "build": build,
        "seconds": round(time.time() - started, 3),
    }
    late = pin.get("recorded_here") or []
    if late:
        entry["recorded_here"] = len(late)
    _note(record_file, entry)
    if not ok:
        _say("no result for job %s: %s" % (job, detail))
        return RC_NO_TREE
    _say(
        "job %s runs pin %s at %s (%d librar(ies) verified in %.1fs%s)"
        % (
            job,
            pin["id"],
            tree,
            len(pin["build"]) - len(late),
            time.time() - started,
            "; %d more recorded here, which the pin could not read where it was taken" % len(late)
            if late
            else "",
        )
    )
    become(pin, tree, argv)


def become(pin, tree, argv):
    """Become ``argv`` inside ``tree``, with the pin's tree first on the path of what it imports."""
    env = os.environ
    env[TREE_ENV] = tree
    env[LIBS_ENV] = ":".join(sorted(pin["build"]))
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != tree]
    hooks = [p for p in paths if os.path.basename(p) == "resolvehook"]
    rest = [p for p in paths if p not in hooks]
    env["PYTHONPATH"] = os.pathsep.join(hooks + [tree] + rest)
    os.chdir(tree)
    os.execvp(argv[0], argv)


# -------------------------------------------------------------------- CLI


def build_parser():
    parser = argparse.ArgumentParser(prog="pintree.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action")
    sub.required = True
    go = sub.add_parser("run", help="make a pin's tree here, check it, run a command in it")
    go.add_argument("--pin", required=True)
    go.add_argument("--job", default="")
    go.add_argument("--record", default="")
    go.add_argument("--steps", action="store_true", help="a job of several nodes run by a lead")
    go.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    command = [a for a in args.command]
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("pintree: nothing to run after --", file=sys.stderr)
        return 2
    return run(args.pin, args.job, args.record, command, steps=args.steps)


__all__ = [
    "BUILDS_DIRNAME",
    "INTEGRATED_FILENAME",
    "CONFIG_FILENAME",
    "LIBRARY_PATTERNS",
    "CACHE_ENV",
    "DEFAULT_PIN_AGE_S",
    "LIBS_ENV",
    "LOCAL_BASE",
    "LOCAL_BASE_ENV",
    "PINS_DIRNAME",
    "PIN_ENV",
    "PIN_MARKER",
    "RC_NO_TREE",
    "TREE_ENV",
    "DigestCache",
    "PinError",
    "at",
    "build_id",
    "build_record",
    "default_cache",
    "digest_file",
    "freeze",
    "become",
    "git_pin",
    "hold",
    "last_pin",
    "libraries",
    "link_native",
    "local_root",
    "make_tree",
    "native_artifacts",
    "pin_file_of",
    "pin_id",
    "pin_tree",
    "pinned_commit",
    "pinned_tree",
    "read_config",
    "read_pin",
    "read_tree_records",
    "record",
    "record_diff",
    "record_digest",
    "repo_lock",
    "run",
    "settle_index",
    "step_argv",
    "sweep",
    "unread",
    "unusable",
    "verify",
    "write_config",
    "write_pin",
]


if __name__ == "__main__":
    sys.exit(main())
