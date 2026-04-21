#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
FetchContent cache manager for TensorRT-LLM 3rdparty dependencies.

Creates and maintains bare git reference repos that accelerate
``git clone`` via the ``--reference`` mechanism.

No explicit ``init`` step — the cache is populated on the fly, every
time ``FetchContent_MakeAvailable`` finishes populating a dep.

Two ``update`` modes:

* ``--src <path>`` single-module mode — invoked by the
  ``FetchContent_MakeAvailable`` override in ``3rdparty/CMakeLists.txt``
  the moment a dep's src is populated (and walks its ``.git/modules/``
  for submodules too).  This is the primary path.
* ``--build-dir <path>`` full-scan mode — walks ``_deps/`` and
  ``.git/modules/`` to rebuild the cache in bulk.  Used for manual
  repair or when the cmake override is not available (older trees).

Usage::

    python fetch_cache.py update --cache-dir <dir> --src <path>
    python fetch_cache.py update --cache-dir <dir> --build-dir <dir>
"""

import argparse
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

# Applied to every bare cache repo via :func:`_apply_safety_config`.
#
# * ``gc.auto`` / ``gc.pruneExpire`` — prevent accidental GC of objects
#   that other clones reach through ``--reference`` alternates.
# * ``transfer.fsckObjects`` / ``fetch.fsckObjects`` — validate the
#   structure of every incoming object at fetch time.  The src repo is
#   untrusted (attacker-controlled content is part of the threat model,
#   see the ``_ensure_cache`` docstring); fsck blocks malformed objects
#   from entering the cache and later tripping up ``--reference`` users.
SAFETY_CONFIG = {
    "gc.auto": "0",
    "gc.pruneExpire": "never",
    "transfer.fsckObjects": "true",
    "fetch.fsckObjects": "true",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _repo_name(url: str) -> str:
    """``https://x/y/Foo.git`` -> ``Foo``."""
    return os.path.basename(url.rstrip("/")).removesuffix(".git")


def _run_git(args, **kwargs):
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    return subprocess.run(["git"] + args, **kwargs)


def _apply_safety_config(bare_dir: str) -> None:
    for key, val in SAFETY_CONFIG.items():
        _run_git(["config", key, val], cwd=bare_dir)


_SECTION_RE = re.compile(r'^\[\s*([^"\]\s]+)(?:\s+"([^"]*)")?\s*\]\s*$')
_KV_RE = re.compile(r'^(\S+)\s*=\s*(.*)$')


# Refnames that are safe to round-trip through a standin's ``packed-refs``
# file.  Only the three categories we actually fetch (heads, origin remotes,
# tags) are admitted, and each segment is restricted to characters that can
# appear verbatim on a ``<sha> <refname>`` line without escaping.  Anything
# that could carry shell- or git-metacharacter injection (spaces, newlines,
# ``;``, ``\``, ``@{`` revision syntax, Unicode controls) is dropped.  See
# :func:`_is_safe_refname` for the git-semantic post-checks (``..`` /
# ``.lock`` / empty components).
_SAFE_REFNAME_RE = re.compile(
    r"^refs/(?:heads|remotes/origin|tags)"
    r"/[A-Za-z0-9][A-Za-z0-9._/-]*$"
)


def _is_safe_refname(refname: str) -> bool:
    """Accept only refnames that the standin can safely advertise.

    Conservative: legitimate but rare ref shapes (``@``, non-ASCII) are
    rejected.  Under-advertising only costs cache hit rate, which the
    threat model in :func:`_ensure_cache` explicitly accepts.
    """
    if not _SAFE_REFNAME_RE.match(refname):
        return False
    # git refname rules that fall outside the character-class test.
    if ".." in refname or "//" in refname:
        return False
    if refname.endswith(("/", ".lock", ".")):
        return False
    return True


def _locate_src_git_dir(src_dir: str) -> str | None:
    """Filesystem-only probe for src's git dir.

    Returns ``<src_dir>/.git`` for worktree layouts, ``<src_dir>`` for
    the raw git-dir layout handed in by :func:`_walk_submodule_dirs`,
    or ``None`` if neither shape is detectable.  Never invokes git —
    same rationale as :func:`_read_src_ref_shas`.
    """
    for git_dir in (os.path.join(src_dir, ".git"), src_dir):
        if os.path.isfile(os.path.join(git_dir, "HEAD")):
            return git_dir
    return None


def _read_src_origin_url(src_dir: str) -> str | None:
    """Parse remote.origin.url from src's git config as plain text.

    Same threat-model rationale as :func:`_read_src_ref_shas`: running
    ``git config`` with cwd=src_dir would load src's ``.git/config``
    through git's normal path (``include.path`` chains and friends),
    which is an attacker-controlled code-execution surface.  We parse
    the file directly instead.

    The parser handles the subset of git config syntax that ``git
    clone`` emits (simple ini-ish ``[section]`` / ``[section
    "subsection"]`` / ``key = value``).  Quoted values, backslash
    continuations and ``include.*`` directives are intentionally
    ignored: a misparsed or absent URL only affects the cache key for
    this update, which the threat model permits.
    """
    # Worktree: <src_dir>/.git/config.  Direct git-dir (submodule
    # layout handed in by _walk_submodule_dirs): <src_dir>/config.
    for git_dir in (os.path.join(src_dir, ".git"), src_dir):
        config_path = os.path.join(git_dir, "config")
        if os.path.isfile(config_path):
            break
    else:
        return None

    try:
        with open(config_path, "r") as fp:
            in_origin = False
            for raw in fp:
                line = raw.strip()
                if not line or line[0] in (";", "#"):
                    continue
                m = _SECTION_RE.match(line)
                if m:
                    section, subsection = m.group(1), m.group(2)
                    in_origin = (section.lower() == "remote"
                                 and subsection == "origin")
                    continue
                if in_origin:
                    m = _KV_RE.match(line)
                    if m and m.group(1).lower() == "url":
                        value = m.group(2).strip()
                        # Strip optional surrounding double quotes — git
                        # clone doesn't emit them but users who hand-edit
                        # config sometimes do.  Everything else (escapes,
                        # continuations) we pass through verbatim.
                        if len(value) >= 2 and value[0] == value[-1] == '"':
                            value = value[1:-1]
                        return value or None
    except OSError:
        pass
    return None


def _ensure_cache(src_dir: str, cache_dir: str) -> str | None:
    """Create or update a bare cache repo from a local *src_dir*.

    Threat model: *src_dir* is untrusted.  It may contain
    attacker-controlled objects, refs, or ``.git`` metadata, and may be
    concurrently mutated by another process while this function runs.
    The invariant we preserve is **monotonic append-only**: the cache
    can acquire new objects and new SHA-named ref anchors, but
    previously-recorded legit content cannot be overwritten or deleted
    by any code path reachable from an untrusted src.

    How we get there:

    * ``git init --bare`` + :func:`_safe_fetch_into_cache` — transfers
      objects reachable from src's refs.  The fetch does **not** target
      src directly; it goes through an ephemeral standin bare so
      ``git-upload-pack`` never runs inside src's attacker-controlled
      ``.git/config``.  Read :func:`_safe_fetch_into_cache` for the
      full rationale, including the ``uploadpack.packObjectsHook`` /
      ``core.fsmonitor`` / ``core.hooksPath`` / ``include.path``
      code-exec vectors being closed.  ``transfer.fsckObjects`` (set
      via :data:`SAFETY_CONFIG`) validates object structure on entry
      for this fetch, so malformed objects never land in the pack.
    * **No ``--update-shallow``.**  When src is shallow (common: cmake
      FetchContent clones most deps with ``GIT_SHALLOW TRUE``), passing
      this flag writes ``.git/shallow`` in the bare.  Removing that file
      is an out-of-band mutation that races with concurrent fetches
      holding git's ``shallow.lock`` — a direct conflict with the
      no-locks idempotent-concurrency model of this cache.  Without the
      flag, git silently rejects shallow-boundary refs on the refspec
      side (``warning: rejected refs/heads/... because shallow roots
      are not allowed to be updated``) but still transfers the objects.
      ``_replicate_src_refs`` below re-establishes the refs we need.
    * :func:`_replicate_src_refs` adds a ``refs/fetch-cache/have/<sha>``
      anchor for every src ref tip using a CAS ``update-ref`` — the
      mechanism that gives us the append-only invariant.

    Concurrency / partial-failure: every step is idempotent.
    ``git init --bare`` re-runs harmlessly on an existing bare,
    ``git config`` is a plain key/value write, ``git fetch`` resumes
    whatever objects are missing via git's internal locking + tmp+rename
    object writes, and ``update-ref`` is serialized per-ref by
    ``packed-refs.lock``.  We deliberately do *not* delete a partial
    bare on failure: leaving it in place lets the next update heal it
    and avoids a "rmtree races fetch" hazard between parallel builds.

    Returns the cache path on success, None on failure.
    """
    url = _read_src_origin_url(src_dir)
    if not url:
        return None
    name = _repo_name(url)
    bare = os.path.join(cache_dir, f"{name}.git")

    src_git = _locate_src_git_dir(src_dir)
    if src_git is None:
        return None

    if not os.path.isfile(os.path.join(bare, "HEAD")):
        r = _run_git(["init", "--bare", bare])
        if r.returncode != 0:
            logger.info("%s.git: init --bare failed, skipping", name)
            return None
    # Re-apply on every update: safety keys may have been added since
    # this bare was first created by an older version of this script.
    _apply_safety_config(bare)

    if _safe_fetch_into_cache(bare, src_git) != 0:
        # Leave the bare in place; next update resumes.
        logger.info("%s.git: fetch failed, will retry on next update", name)
        return None

    _replicate_src_refs(src_git, bare)
    logger.info("updated %s.git <- %s", name, src_dir)
    return bare


# ---------------------------------------------------------------------------
# SHA-named ref anchors — the append-only guarantee
# ---------------------------------------------------------------------------

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _read_src_ref_shas(src_dir: str) -> set[str]:
    """Enumerate ref-tip SHAs from *src_dir* by parsing ref files directly.

    **Rationale**: src is untrusted per the threat model in
    :func:`_ensure_cache` and may be concurrently mutated.  Any git
    command with cwd inside src loads src's ``.git/config``, which
    over git's history has been a code-execution vector
    (``include.path`` chains, ``core.fsmonitor``, several named
    CVEs).  Reading ref files as plain text sidesteps the entire
    class of attack.

    Safety rests on two filters, both applied *after* the read:

      * SHA-1 regex on file content — anything that isn't a 40-hex
        string is dropped, so even accidentally-readable content
        can't become an anchor.
      * ``update-ref`` in :func:`_replicate_src_refs` refuses
        non-existent objects, so even a valid-looking SHA that
        isn't actually in the bare is silently skipped.

    Probing for the git dir is intentionally sloppy: under-detecting
    means we anchor fewer SHAs, which the threat model explicitly
    permits (a bad src may cause us to cache less, but cannot
    corrupt the cache).
    """
    # Worktree layout stores refs under .git/; submodule dirs handed
    # in by _walk_submodule_dirs are themselves the git dir.  Anything
    # else just yields no SHAs.
    for git_dir in (os.path.join(src_dir, ".git"), src_dir):
        if os.path.isfile(os.path.join(git_dir, "HEAD")):
            break
    else:
        return set()

    shas: set[str] = set()

    # packed-refs: one SHA per "<sha> <refname>" line, plus "^<sha>"
    # peeled lines for annotated tags (which we also want to anchor —
    # they're the commit SHAs upstream will recognize in "have").
    try:
        with open(os.path.join(git_dir, "packed-refs"), "r") as fp:
            for line in fp:
                line = line.strip()
                if not line or line[0] == "#":
                    continue
                candidate = line[1:] if line[0] == "^" else line.split(None, 1)[0]
                if _SHA_RE.match(candidate):
                    shas.add(candidate)
    except OSError:
        pass

    # Loose refs under refs/**.  Regular-file gate: opening a FIFO or
    # socket would block indefinitely waiting for a writer, letting an
    # attacker hang the update by replacing a ref file with `mkfifo`.
    # Device files etc. are equally nonsensical as ref content.  Skip
    # anything that isn't a plain regular file.
    refs_dir = os.path.join(git_dir, "refs")
    for dirpath, _, filenames in os.walk(refs_dir):
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            try:
                if not stat.S_ISREG(os.stat(path).st_mode):
                    continue
                with open(path, "r") as fp:
                    content = fp.read(41).strip()
            except OSError:
                continue
            if _SHA_RE.match(content):
                shas.add(content)

    return shas


def _read_src_shallow_shas(src_git: str) -> list[str]:
    """Return SHA-validated lines from ``<src_git>/shallow``.

    cmake FetchContent clones most deps with ``GIT_SHALLOW TRUE``, so
    src is typically a shallow repo: its ``shallow`` file lists the
    commits whose parents are intentionally absent from the object
    store.  When we later fetch from the standin bare (whose objects
    come from src via alternates), ``upload-pack`` walks commit
    graphs and will die with "Failed to traverse parents of commit
    <SHA>" unless standin also knows those commits are shallow
    boundaries.  Replicating the file into the standin fixes that.

    Same plain-text, fsmode-gated read as
    :func:`_read_src_ref_shas`; SHA filter drops anything that isn't
    a 40-hex line.  An attacker-supplied bogus SHA is harmless — git
    just treats nonexistent shallow entries as no-ops.
    """
    path = os.path.join(src_git, "shallow")
    shas: list[str] = []
    try:
        if not stat.S_ISREG(os.stat(path).st_mode):
            return shas
        with open(path, "r") as fp:
            for raw in fp:
                sha = raw.strip()
                if _SHA_RE.match(sha):
                    shas.append(sha)
    except OSError:
        pass
    return shas


def _read_src_refs(src_git: str) -> list[tuple[str, str]]:
    """Parse ``(refname, sha)`` pairs from *src_git* as plain text.

    Stricter companion to :func:`_read_src_ref_shas`: preserves the
    refname (needed so the standin bare in :func:`_safe_fetch_into_cache`
    can re-advertise src's refs to ``upload-pack`` without opening src's
    ``.git/config``) and drops any entry that fails
    :func:`_is_safe_refname`.  Refnames that pass can be written
    verbatim onto a ``<sha> <refname>`` line in ``packed-refs`` —
    the character-class whitelist already forbids whitespace and
    shell metacharacters, which is the property that lets us skip
    quoting.

    Reads are pure filesystem I/O for the same reason as
    :func:`_read_src_ref_shas` — running git with cwd inside src
    would load src's ``.git/config``, an attacker-controlled
    code-execution surface.
    """
    pairs: list[tuple[str, str]] = []

    # packed-refs: "<sha> <refname>" per line.  "^<sha>" peeled lines
    # annotate the preceding annotated-tag entry and carry no refname
    # of their own, so they're deliberately skipped here.  (Peeled
    # commit SHAs still get anchored by _read_src_ref_shas /
    # _replicate_src_refs — the standin advertisement path is a
    # different concern.)
    try:
        with open(os.path.join(src_git, "packed-refs"), "r") as fp:
            for raw in fp:
                line = raw.strip()
                if not line or line[0] in ("#", "^"):
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                sha, refname = parts
                if _SHA_RE.match(sha) and _is_safe_refname(refname):
                    pairs.append((refname, sha))
    except OSError:
        pass

    # Loose refs under refs/**.  The path relative to git_dir *is*
    # the refname (git's on-disk layout).  Regular-file gate defends
    # against FIFO/socket content for the same reason as in
    # _read_src_ref_shas.
    refs_dir = os.path.join(src_git, "refs")
    for dirpath, _, filenames in os.walk(refs_dir):
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            refname = os.path.relpath(path, src_git).replace(os.sep, "/")
            if not _is_safe_refname(refname):
                continue
            try:
                if not stat.S_ISREG(os.stat(path).st_mode):
                    continue
                with open(path, "r") as fp:
                    content = fp.read(41).strip()
            except OSError:
                continue
            if _SHA_RE.match(content):
                pairs.append((refname, content))

    return pairs


def _safe_fetch_into_cache(bare: str, src_git: str) -> int:
    """Fetch objects from untrusted *src_git* into *bare* without
    letting ``git-upload-pack`` run with ``GIT_DIR=<src_git>``.

    **Threat being closed.**  ``git fetch <src_path>`` uses git's
    local transport: the fetching process forks ``upload-pack`` with
    ``GIT_DIR=<src_path>/.git``, and upload-pack loads
    ``<src_path>/.git/config`` before sending a single object.  Per
    the threat model in :func:`_ensure_cache`, that config file is
    **attacker-controlled** — src was populated by a malicious
    clone, and the dep name (and thus cache key) is itself an
    attacker input.  Known code-execution keys that fire from
    upload-pack at this moment:

      * ``uploadpack.packObjectsHook`` — documented to run instead
        of ``git pack-objects``; straight ``exec()``
      * ``core.fsmonitor`` — runs on index / ref-scan operations
      * ``core.hooksPath`` — redirects hook lookup, making
        ``post-upload-pack`` an attacker-controlled binary
      * ``include.path`` / ``includeIf.*.path`` — chains into
        further attacker-controlled config files, layering any of
        the above on top

    ``update`` is a hot path (once per dep per build), so a single
    malicious dep compromises every builder who cache-hits it.

    **Mitigation: standin bare + alternates.**  We build an
    ephemeral "standin" bare repo inside *cache_dir* (the only
    trusted writable location in the threat model), point it at
    src's objects via ``objects/info/alternates``, re-publish
    whitelisted ``(refname, sha)`` pairs into the standin's
    ``packed-refs``, and fetch from the standin.  ``upload-pack``
    then runs with ``GIT_DIR=<standin>`` and reads the standin's
    config — which we just created via ``git init --bare`` with
    hardened env, so it contains only git's defaults.

    Why this works:

      * ``objects/info/alternates`` is a pure object-store pointer.
        git's object layer follows it to resolve SHAs, but the
        config layer does **not** open any file rooted at the
        alternate path.  So even though the alternate points deep
        into attacker-controlled territory, no attacker config is
        loaded.
      * Refs in the standin come from :func:`_read_src_refs`, which
        already filters both SHA (40-hex) and refname
        (:func:`_is_safe_refname`).  Attacker cannot smuggle
        ``refs/heads/foo\\n<evil>`` — the character class rejects
        newlines, spaces, and shell metacharacters before they hit
        ``packed-refs``.
      * ``fetch.fsckObjects`` (set on *bare* via
        :data:`SAFETY_CONFIG`) still runs: the standin→bare fetch
        is a regular pack negotiation and receives real pack data,
        which cache-side git fscks on entry.

    **Defense in depth.**  Independent of the standin:

      * ``GIT_CONFIG_SYSTEM=/dev/null`` + ``GIT_CONFIG_GLOBAL=/dev/null``
        mask out ``/etc/gitconfig`` and ``~/.gitconfig`` for every
        git in this call tree (including upload-pack via env
        inheritance).  Covers shared CI machines where
        ``/etc/gitconfig`` might carry another user's hook /
        fsmonitor path.
      * ``-c uploadpack.packObjectsHook= -c core.fsmonitor=
        -c core.hooksPath=/dev/null`` force-empty the known
        code-exec keys; these propagate to upload-pack via
        ``GIT_CONFIG_PARAMETERS`` and outrank any value that
        survived the config masking.

    **Cleanup.**  The standin is created with
    ``tempfile.mkdtemp(dir=os.path.dirname(bare), prefix=".fc-standin-")``
    (same filesystem as the cache, unique name, leading dot keeps it
    out of the way of any ``ls`` over cache keys which never start
    with a dot) and unconditionally ``rmtree``'d in the ``finally``
    block.  A hard-kill between mkdtemp and cleanup leaks a small
    temp dir; it's safe to remove out-of-band, and none of its
    contents is load-bearing across processes.
    """
    cache_parent = os.path.dirname(bare)
    standin = tempfile.mkdtemp(prefix=".fc-standin-", dir=cache_parent)
    try:
        # Hardened env: mask system / global config so neither
        # ``git init --bare`` (for standin) nor the subsequent fetch
        # and its forked ``upload-pack`` can read ``/etc/gitconfig``
        # or ``~/.gitconfig``.  Also drop vars that would redirect
        # git to an unexpected git dir / worktree.
        hardened_env = {
            **os.environ,
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_CONFIG_GLOBAL": "/dev/null",
        }
        for var in ("XDG_CONFIG_HOME", "GIT_CONFIG", "GIT_DIR",
                    "GIT_WORK_TREE", "GIT_COMMON_DIR"):
            hardened_env.pop(var, None)

        r = _run_git(["init", "--bare", "-q", standin], env=hardened_env)
        if r.returncode != 0:
            logger.info("standin init failed for %s", src_git)
            return r.returncode

        # alternates: absolute path (relative alternates resolve
        # against the *containing* objects/ dir, which for standin is
        # standin's own, not src's).  One line, one path — no
        # recursion needed here; if src itself chains alternates
        # further, git's object layer follows them and fsck still
        # validates every object that lands in *bare*.
        alt_info = os.path.join(standin, "objects", "info")
        os.makedirs(alt_info, exist_ok=True)
        with open(os.path.join(alt_info, "alternates"), "w") as fp:
            fp.write(os.path.abspath(os.path.join(src_git, "objects"))
                     + "\n")

        # Re-publish src's refs as the standin's refs.  Writing
        # packed-refs directly (instead of ``update-ref``) avoids a
        # process fork per ref and sidesteps any lingering config
        # reads; the content is a simple text format that upload-pack
        # reads the same way as the loose-ref alternative.
        refs = _read_src_refs(src_git)
        with open(os.path.join(standin, "packed-refs"), "w") as fp:
            fp.write("# pack-refs with: peeled fully-peeled sorted \n")
            for refname, sha in sorted(refs):
                fp.write(f"{sha} {refname}\n")

        # Inherit src's shallow boundary.  Without this, ``upload-pack``
        # in standin would walk into the missing parent of each
        # shallow-tipped commit and die with "Failed to traverse
        # parents".  Absent file = src isn't shallow, nothing to do.
        shallow_shas = _read_src_shallow_shas(src_git)
        if shallow_shas:
            with open(os.path.join(standin, "shallow"), "w") as fp:
                for sha in shallow_shas:
                    fp.write(sha + "\n")

        r = _run_git(
            [
                # Force-empty known code-exec keys on both sides of
                # the fetch (GIT_CONFIG_PARAMETERS propagates to
                # upload-pack via the subprocess env).
                "-c", "uploadpack.packObjectsHook=",
                "-c", "core.fsmonitor=",
                "-c", "core.hooksPath=/dev/null",
                # Allow local-path fetch even when the host-wide
                # default is ``protocol.file.allow=user`` under a
                # non-interactive build (legitimate here: standin
                # path was just created by us).
                "-c", "protocol.file.allow=always",
                "fetch", "--no-tags", "--no-auto-gc",
                "--no-write-fetch-head",
                standin,
                "+refs/heads/*:refs/fetch-cache/heads/*",
                "+refs/remotes/origin/*:refs/fetch-cache/remotes/origin/*",
                "+refs/tags/*:refs/fetch-cache/tags/*",
            ],
            cwd=bare,
            env=hardened_env,
        )
        return r.returncode
    finally:
        shutil.rmtree(standin, ignore_errors=True)


def _replicate_src_refs(src_dir: str, bare: str) -> None:
    """Anchor each of *src_dir*'s ref tips in *bare* under
    ``refs/fetch-cache/have/<sha>``.

    Why SHA-named refs:

    * **Refname = SHA** ⇒ distinct SHAs never collide on a refname,
      same SHA is idempotent.  There is no reachable code path that
      rewrites a previously-recorded ``have/<sha>`` to point elsewhere,
      which is what makes the cache append-only even when src is
      hostile.
    * ``update-ref <ref> <sha> ""`` (empty old-value) = atomic CAS
      "create only if ref does not exist".  Concurrent writers trying
      to write the same (sha, ref) pair can't trample each other; the
      loser's CAS fails and the net state is "anchor present".  git
      also natively rejects ``update-ref`` against a nonexistent
      object, which covers dangling refs / src-side TOCTOU without a
      separate pre-check.

    SHAs come from :func:`_read_src_ref_shas` — we deliberately do
    **not** run ``git for-each-ref`` in src_dir; see that function's
    docstring for the threat-model rationale.

    All ref writes are serialized by git's packed-refs.lock, so this
    function is safe to call concurrently on the same bare from
    multiple processes.
    """
    for sha in _read_src_ref_shas(src_dir):
        _run_git(
            ["update-ref", f"refs/fetch-cache/have/{sha}", sha, ""],
            cwd=bare,
        )


# ---------------------------------------------------------------------------
# update — single-module mode (called by wrapper after each clone)
# ---------------------------------------------------------------------------

def cmd_update_src(cache_dir: str, src_dir: str) -> int:
    """Create or update a bare cache from a single *src_dir* and any of
    its submodule git dirs.

    Called by the ``FetchContent_MakeAvailable`` override the moment a
    populate finishes, so the cache fills up module-by-module as the
    build progresses.  If the build aborts midway, every module that
    has already been populated is still reusable on the next build.
    """
    if not os.path.isdir(src_dir):
        logger.info("src %s does not exist, nothing to update", src_dir)
        return 0
    os.makedirs(cache_dir, exist_ok=True)
    _ensure_cache(src_dir, cache_dir)
    _update_submodules(src_dir, cache_dir)
    return 0


# ---------------------------------------------------------------------------
# update — full-scan mode (manual rebuild / legacy fallback)
# ---------------------------------------------------------------------------

def cmd_update(cache_dir: str, build_dir: str) -> int:
    """Scan ``_deps/`` for git repos, create or update bare caches."""
    deps_dir = os.path.join(build_dir, "_deps")
    if not os.path.isdir(deps_dir):
        logger.info("No _deps directory in %s, nothing to update", build_dir)
        return 0

    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Updating FetchContent cache: %s", cache_dir)

    for entry in sorted(os.listdir(deps_dir)):
        if not entry.endswith("-src"):
            continue
        src_dir = os.path.join(deps_dir, entry)
        git_marker = os.path.join(src_dir, ".git")
        if not (os.path.isdir(git_marker) or os.path.isfile(git_marker)):
            continue

        # Top-level dep
        _ensure_cache(src_dir, cache_dir)

        # Submodule repos (scan .git/modules/)
        _update_submodules(src_dir, cache_dir)

    logger.info("FetchContent cache update finished.")
    return 0


def _update_submodules(src_dir: str, cache_dir: str) -> None:
    """Cache bare repos found under ``.git/modules/`` (recursively).

    Probes for the git dir via plain filesystem checks rather than
    ``git rev-parse --git-dir`` — running git with cwd=src_dir loads
    src's ``.git/config``, which is an attacker-controlled code-
    execution surface (same rationale as :func:`_read_src_ref_shas`).
    """
    for git_dir in (os.path.join(src_dir, ".git"), src_dir):
        if os.path.isfile(os.path.join(git_dir, "HEAD")):
            break
    else:
        return
    modules_dir = os.path.join(git_dir, "modules")
    if os.path.isdir(modules_dir):
        _walk_submodule_dirs(modules_dir, cache_dir, rel_prefix="")


def _walk_submodule_dirs(modules_dir: str, cache_dir: str,
                         rel_prefix: str) -> None:
    """Cache every git repo reachable under ``modules_dir``.

    Children are either real submodule git dirs (HEAD + objects/) —
    cache them and recurse into their ``modules/`` — or path-segment
    directories when a submodule path contains a slash (e.g.
    ``third-party/fmt`` -> ``third-party/``) — recurse to find the git
    dirs inside.
    """
    try:
        it = os.scandir(modules_dir)
    except OSError:
        return
    with it:
        for entry in it:
            if not entry.is_dir(follow_symlinks=False):
                continue
            sub = entry.path
            rel = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name
            is_git_dir = (os.path.isfile(os.path.join(sub, "HEAD"))
                          and os.path.isdir(os.path.join(sub, "objects")))
            if is_git_dir:
                _ensure_cache(sub, cache_dir)
                nested = os.path.join(sub, "modules")
                if os.path.isdir(nested):
                    _walk_submodule_dirs(nested, cache_dir, rel)
            else:
                # Intermediate path segment (submodule path with a slash).
                _walk_submodule_dirs(sub, cache_dir, rel)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="FetchContent cache manager")
    p.add_argument(
        "--log-prefix",
        default="",
        help="String prepended to every log line (used by the CMake "
             "FetchContent override to align nested output with the "
             "surrounding dep-name column)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("update",
                        help="Create or update cache from build artifacts")
    sp.add_argument("--cache-dir", required=True, help="Cache directory")
    mode = sp.add_mutually_exclusive_group(required=True)
    mode.add_argument("--src",
                      help="Single-module mode: path to a freshly-cloned "
                           "src repo (used by fetch_cache_wrapper.py)")
    mode.add_argument("--build-dir",
                      help="Full-scan mode: walk _deps/ under this cmake "
                           "build dir (manual rebuild / legacy fallback)")

    args = p.parse_args()

    # Escape stray '%' in the caller-supplied prefix so logging's %-formatter
    # doesn't interpret it as a format directive.  Log to stdout so CMake's
    # execute_process can pipe output through without merging with stderr.
    prefix_escaped = args.log_prefix.replace("%", "%%")
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format=f"{prefix_escaped}[fetch-cache] %(message)s",
    )

    if args.cmd == "update":
        if args.src:
            return cmd_update_src(args.cache_dir, args.src)
        return cmd_update(args.cache_dir, args.build_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
