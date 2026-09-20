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
FetchContent cache writer for TensorRT-LLM 3rdparty dependencies.

Creates and maintains bare git reference repos so subsequent clones
accelerate via ``git clone --reference <bare>``.  See
``3rdparty/fetch-cache.md`` for the end-to-end flow, cache layout, and
threat model.

Two ``update`` modes:

* ``--src <path>`` — invoked by the ``FetchContent_MakeAvailable``
  override in ``3rdparty/CMakeLists.txt`` right after a dep's populate
  completes (walks the dep's ``.git/modules/`` too).  Primary path.
* ``--build-dir <path>`` — walks ``_deps/`` in bulk.  Manual repair
  or reseed from an existing build tree.

Both modes also mirror src's git-LFS objects into the bare's LFS pool;
see ``3rdparty/fetch-cache.md`` "LFS pool" for the design treatment.
"""

import argparse
import hashlib
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

# Required by invariants I2 and I3 in fetch-cache.md:
# gc.* preserves objects that other clones reach through --reference;
# fsckObjects blocks malformed inbound objects from entering the cache.
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


def _iter_dir(path: str):
    """Yield ``os.DirEntry`` for each child of *path*; empty on OSError."""
    try:
        it = os.scandir(path)
    except OSError:
        return
    with it:
        yield from it


def _apply_safety_config(bare_dir: str) -> None:
    for key, val in SAFETY_CONFIG.items():
        _run_git(["config", key, val], cwd=bare_dir)


# ---------------------------------------------------------------------------
# LFS pool helpers — see fetch-cache.md "LFS pool" for the design treatment
# (layout, consumer pattern, why the pool exists separately from alternates).
# ---------------------------------------------------------------------------

_OID_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX2_RE = re.compile(r"^[0-9a-f]{2}$")
# 16x headroom over real-world LFS payloads while still rejecting an
# attacker-crafted src advertising a multi-TB "object".
_LFS_MAX_SIZE = 64 * (1 << 30)
_LFS_BUFSIZE = 1 << 20


def _sha256_regular_file(path: str) -> str | None:
    """Hex-encoded SHA-256 of *path*'s contents, or None on read error."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fp:
            while True:
                chunk = fp.read(_LFS_BUFSIZE)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _walk_src_lfs_objects(src_git: str):
    """Yield ``(oid, abs_path)`` for each plausible LFS object under
    ``<src_git>/lfs/objects/``.

    Pre-filtered to the LFS-standard ``<2>/<2>/<64-hex>`` layout; symlinks
    are rejected at every level (I3).  SHA-256 verification still runs
    per-file in :func:`_ingest_lfs_object`.
    """
    lfs_root = os.path.join(src_git, "lfs", "objects")
    for s1 in _iter_dir(lfs_root):
        if not _HEX2_RE.match(s1.name) or not s1.is_dir(follow_symlinks=False):
            continue
        for s2 in _iter_dir(s1.path):
            if not _HEX2_RE.match(s2.name) or not s2.is_dir(follow_symlinks=False):
                continue
            for f in _iter_dir(s2.path):
                oid = f.name
                if not _OID_RE.match(oid):
                    continue
                if not oid.startswith(s1.name + s2.name):
                    continue
                if not f.is_file(follow_symlinks=False):
                    continue
                yield oid, f.path


def _ingest_lfs_object(src_path: str, dst_path: str, oid: str) -> bool:
    """Place src_path's contents at *dst_path* iff its SHA-256 matches *oid*.

    Sequence: (1) copy src into a sibling tempfile; (2) re-hash the
    staged file and compare to *oid*; (3) atomic rename into place (I2).
    """
    dst_dir = os.path.dirname(dst_path)
    os.makedirs(dst_dir, exist_ok=True)

    try:
        st = os.stat(src_path, follow_symlinks=False)
    except OSError:
        return False
    if not stat.S_ISREG(st.st_mode):
        return False
    if st.st_size > _LFS_MAX_SIZE:
        logger.warning(
            "lfs %s: %d bytes exceeds %d cap, skipping",
            oid, st.st_size, _LFS_MAX_SIZE,
        )
        return False

    fd, tmp_path = tempfile.mkstemp(prefix=f".lfs-tmp.{oid[:16]}.", dir=dst_dir)
    try:
        try:
            with os.fdopen(fd, "wb") as dst, open(src_path, "rb") as src:
                shutil.copyfileobj(src, dst, _LFS_BUFSIZE)
        except OSError as e:
            logger.warning("lfs copy %s: %s", oid, e)
            return False

        actual = _sha256_regular_file(tmp_path)
        if actual != oid:
            logger.warning(
                "lfs %s: content hashes to %s, dropping",
                oid, actual,
            )
            return False

        try:
            os.rename(tmp_path, dst_path)
        except OSError as e:
            logger.warning("lfs rename %s: %s", oid, e)
            return False
        return True
    finally:
        # Success path already renamed tmp_path away; cleanup is a no-op
        # there and a recovery on every error path.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _update_lfs_pool(src_git: str, bare: str) -> None:
    """Mirror src's LFS objects into ``<bare>/lfs/objects/`` (I2)."""
    pool = os.path.join(bare, "lfs", "objects")
    ingested = skipped = failed = 0
    for oid, src_path in _walk_src_lfs_objects(src_git):
        dst_path = os.path.join(pool, oid[:2], oid[2:4], oid)
        if os.path.isfile(dst_path):
            skipped += 1
            continue
        if _ingest_lfs_object(src_path, dst_path, oid):
            ingested += 1
        else:
            failed += 1
    if ingested or failed:
        logger.info(
            "lfs pool: %d ingested, %d already cached, %d rejected -> %s",
            ingested, skipped, failed, pool,
        )


_SECTION_RE = re.compile(r'^\[\s*([^"\]\s]+)(?:\s+"([^"]*)")?\s*\]\s*$')
_KV_RE = re.compile(r'^(\S+)\s*=\s*(.*)$')


# Refnames admitted into the standin's packed-refs.  Three fetched
# categories only; quoting/escaping rationale lives in fetch-cache.md
# (threat model I3).  git-semantic checks (.., .lock, empty components)
# happen in _is_safe_refname.
_SAFE_REFNAME_RE = re.compile(
    r"^refs/(?:heads|remotes/origin|tags)"
    r"/[A-Za-z0-9][A-Za-z0-9._/-]*$"
)


def _is_safe_refname(refname: str) -> bool:
    """Accept only refnames the standin can advertise verbatim.

    Under-advertising only costs cache hit rate, which the threat model
    (fetch-cache.md) explicitly permits; rare shapes (``@``, non-ASCII) are
    dropped.
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

    ``<src_dir>/.git`` for worktree layouts, ``<src_dir>`` itself for
    the raw git-dir layout handed in by :func:`_walk_submodule_dirs`.
    Never invokes git (I3).
    """
    for git_dir in (os.path.join(src_dir, ".git"), src_dir):
        if os.path.isfile(os.path.join(git_dir, "HEAD")):
            return git_dir
    return None


def _read_src_origin_url(src_dir: str) -> str | None:
    """Parse ``remote.origin.url`` from src's git config as plain text (I3).

    Handles the subset of git config syntax that ``git clone`` emits
    (plain ``[section]`` / ``[section "subsection"]`` / ``key = value``).
    Quoted values, backslash continuations, and ``include.*`` directives
    are intentionally ignored: a misparsed or absent URL only affects
    this update's cache key, which I2 + I3 permit.
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


def _src_is_shallow(src_git: str) -> bool:
    """Decide whether *src_git* is a shallow repo (picks the cache pool).

    Non-empty ``<src_git>/shallow`` ⇒ shallow pool; missing/empty/
    non-regular ⇒ full pool (I5).  ``stat`` + ``S_ISREG`` avoids
    running git inside src (I3).
    """
    path = os.path.join(src_git, "shallow")
    try:
        st = os.stat(path)
    except OSError:
        return False
    if not stat.S_ISREG(st.st_mode):
        return False
    return st.st_size > 0


def _is_connected(bare: str, sha: str) -> bool:
    """Return True iff *sha*'s commit graph is fully present in *bare* (I5).

    ``rev-list --quiet`` exits 0 only when every commit reached is
    readable; pack negotiation ships all trees/blobs reachable from
    transferred commits, so commit-graph connectivity implies content
    connectivity.
    """
    r = subprocess.run(
        ["git", "-C", bare, "rev-list", "--quiet", sha, "--"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    return r.returncode == 0


def _existing_have_shas(bare: str) -> set[str]:
    """Enumerate SHAs already anchored under ``refs/fetch-cache/have/``.

    Caller uses the result as the skip set for the connectivity
    memoization described in I5 of fetch-cache.md.  One git invocation;
    set membership on the caller side is effectively free next to a
    single rev-list walk.
    """
    r = _run_git(
        ["for-each-ref", "--format=%(refname)", "refs/fetch-cache/have/"],
        cwd=bare,
    )
    if r.returncode != 0:
        return set()
    prefix = "refs/fetch-cache/have/"
    shas: set[str] = set()
    for line in r.stdout.splitlines():
        name = line.strip()
        if name.startswith(prefix):
            candidate = name[len(prefix):]
            if _SHA_RE.match(candidate):
                shas.add(candidate)
    return shas


def _prune_disconnected_fetch_refs(bare: str, verified: set[str]) -> None:
    """Delete any ``refs/fetch-cache/{heads,remotes,tags}/...`` whose
    tip fails :func:`_is_connected`.

    Keeps the advertisement namespace honest so ``clone --reference``
    negotiation never offers an ancestor-less tip as a "have" (I5).
    *verified* is the memoization skip set.
    """
    r = _run_git(
        ["for-each-ref",
         "--format=%(refname) %(objectname)",
         "refs/fetch-cache/heads",
         "refs/fetch-cache/remotes",
         "refs/fetch-cache/tags"],
        cwd=bare,
    )
    if r.returncode != 0:
        return
    for line in r.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        refname, sha = parts
        if sha in verified:
            continue
        if not _is_connected(bare, sha):
            logger.info("%s: prune disconnected %s (%s)",
                        os.path.basename(bare), refname, sha[:8])
            _run_git(["update-ref", "-d", refname, sha], cwd=bare)


def _src_has_no_own_objects(src_git: str) -> bool:
    """Return True iff src has no objects of its own — ``objects/``
    holds nothing but at most an ``info/alternates`` pointer.

    Pure FS probe (I3).  Conservative by design: any file under
    ``objects/`` other than ``info/alternates`` (a pack, a loose
    object, a ``commit-graph``, anything) makes this return False and
    the caller falls back to the regular standin fetch.  We don't try
    to decide whether such a file is content the bare already has —
    the slow path is itself a safe no-op when content is genuinely
    cached, so over-bailing costs us at most one redundant fetch.
    """
    objects_dir = os.path.join(src_git, "objects")
    alt_path = os.path.join(objects_dir, "info", "alternates")
    for dirpath, _, filenames in os.walk(objects_dir):
        for fn in filenames:
            if os.path.join(dirpath, fn) != alt_path:
                return False
    return True


def _ensure_cache(src_dir: str, cache_dir: str) -> str | None:
    """Create or update a bare cache repo from untrusted *src_dir*.

    Steps: ``git init --bare`` (idempotent), apply safety config,
    fetch objects via the standin-bare route (see
    :func:`_safe_fetch_into_cache`), then replicate src's ref tips as
    ``refs/fetch-cache/have/<sha>`` anchors — shallow pool unchecked,
    full pool gated on :func:`_is_connected`.

    Concurrency and partial failure: every step is idempotent (I4),
    and a half-built bare is **never** removed — leaving it in place
    lets the next update resume and avoids a "rmtree races fetch"
    hazard.  Returns the cache path on success, None on failure.

    End-to-end rationale (split cache, threat model, invariants) is
    in ``3rdparty/fetch-cache.md``.
    """
    url = _read_src_origin_url(src_dir)
    if not url:
        return None
    name = _repo_name(url)

    src_git = _locate_src_git_dir(src_dir)
    if src_git is None:
        return None

    # "shallow"/"full" are hardcoded constants, never derived from
    # src — the trust boundary (fetch-cache.md) relies on this.
    shallow = _src_is_shallow(src_git)
    subdir = "shallow" if shallow else "full"
    bare_parent = os.path.join(cache_dir, subdir)
    bare = os.path.join(bare_parent, f"{name}.git")

    # Up-to-date short-circuit.  When src has no objects of its own,
    # the regular update would fetch zero git bytes — there is simply
    # nothing for src to contribute to the git side.  Skip the entire
    # standin / fetch / replicate / safety-config pass.  SAFETY_CONFIG
    # only affects fetch-time behavior (fsck on inbound objects, gc
    # disabled), and we are doing neither; any newly-added safety key
    # gets picked up the next time real content arrives (slow path).
    # Missing refs/fetch-cache/have/ anchors likewise stay missing.
    # ``clone --reference`` walks the alternate's refs via
    # ``for_each_alternate_ref`` and folds each ref's SHA into the
    # negotiation as a "have" advertisement, so a missing anchor
    # only loses the corresponding have-line and costs a few extra
    # bytes on the next clone.  The objects themselves are in the
    # alternate's ``objects/`` regardless, so correctness is
    # unchanged; the next real update fills the anchors in.  LFS
    # objects are orthogonal: a consumer that ran ``git lfs pull``
    # after a ``--reference`` clone has no own git objects but may
    # carry new LFS payloads, so the LFS pool walk runs on this
    # path too.  Cold caches (no HEAD yet) bypass this branch and
    # fall through below.
    if (os.path.isfile(os.path.join(bare, "HEAD"))
            and _src_has_no_own_objects(src_git)):
        _update_lfs_pool(src_git, bare)
        logger.info("up-to-date %s/%s.git <- %s", subdir, name, src_dir)
        return bare

    os.makedirs(bare_parent, exist_ok=True)

    if not os.path.isfile(os.path.join(bare, "HEAD")):
        r = _run_git(["init", "--bare", bare])
        if r.returncode != 0:
            logger.info("%s/%s.git: init --bare failed, skipping",
                        subdir, name)
            return None
    # Re-apply on every update so the bare always carries the current
    # SAFETY_CONFIG, regardless of which writer first created it.
    _apply_safety_config(bare)

    if _safe_fetch_into_cache(bare, src_git) != 0:
        # Leave the bare in place; next update resumes.
        logger.info("%s/%s.git: fetch failed, will retry on next update",
                    subdir, name)
        return None

    if shallow:
        _replicate_src_refs(src_git, bare)
    else:
        # Memoize connectivity across prune and replicate passes
        # (see fetch-cache.md I5, memoization subsection).
        verified = _existing_have_shas(bare)
        _prune_disconnected_fetch_refs(bare, verified)
        _replicate_src_refs_checked(src_git, bare, verified)
    _update_lfs_pool(src_git, bare)
    logger.info("updated %s/%s.git <- %s", subdir, name, src_dir)
    return bare


# ---------------------------------------------------------------------------
# SHA-named ref anchors — the append-only guarantee
# ---------------------------------------------------------------------------

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _read_src_ref_shas(src_dir: str) -> set[str]:
    """Enumerate ref-tip SHAs by parsing src's ref files as plain text (I3).

    Two filters applied after the read guarantee only real objects
    become anchors:

      * SHA-1 regex on file content (drops accidentally-readable
        non-SHA content).
      * ``update-ref`` in :func:`_replicate_src_refs` refuses
        nonexistent objects (drops valid-looking SHAs that aren't
        actually in the bare).

    Under-detecting a git dir only lowers cache hit rate, which the
    threat model permits.
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

    # Loose refs under refs/**.  S_ISREG gate: opening a FIFO/socket
    # would hang the update forever waiting for a writer — trivial
    # DoS via `mkfifo refs/heads/main`.
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

    Content is replicated into the standin bare so its upload-pack
    knows which commits are shallow boundaries and does not die on
    "Failed to traverse parents of commit <SHA>".  Bogus attacker
    SHAs are harmless — git treats nonexistent shallow entries as
    no-ops.
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
    """Parse ``(refname, sha)`` pairs from *src_git* as plain text (I3).

    Stricter than :func:`_read_src_ref_shas`: preserves the refname
    (needed so the standin's ``packed-refs`` can re-advertise src's
    refs to ``upload-pack``) and drops anything failing
    :func:`_is_safe_refname`, so surviving entries go onto a
    ``<sha> <refname>`` line verbatim without quoting.
    """
    pairs: list[tuple[str, str]] = []

    # "^<sha>" peeled lines annotate the preceding annotated-tag entry
    # and carry no refname, so they're skipped here.  (Peeled commit
    # SHAs still get anchored via _read_src_ref_shas.)
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

    # The path relative to git_dir *is* the refname (git's on-disk
    # layout).  S_ISREG gate matches _read_src_ref_shas (FIFO/socket DoS).
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
    letting ``upload-pack`` run with ``GIT_DIR=<src_git>``.

    The standin-bare + alternates construction and the full list of
    code-exec keys it closes are documented in ``3rdparty/fetch-cache.md``
    (threat model, invariant I3).  Inline comments below flag the
    **local** reason each line exists.
    """
    # System temp by default (tmpfs perf on shared-FS caches); fall back
    # to the cache parent when the system temp is not writable.  Design
    # treatment in fetch-cache.md "Performance model" and threat model I3.
    try:
        standin = tempfile.mkdtemp(prefix=".fc-standin-")
    except OSError:
        standin = tempfile.mkdtemp(prefix=".fc-standin-", dir=os.path.dirname(bare))
    try:
        # Masks /etc/gitconfig and ~/.gitconfig for both the standin
        # init and the forked upload-pack (env inheritance), and drops
        # vars that would redirect git to an unexpected git dir.
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

        # Absolute path: relative alternates resolve against the
        # standin's own objects/ dir, not src's.  If src itself chains
        # alternates further, git's object layer follows; fsck on
        # bare's side still validates every inbound object.
        alt_info = os.path.join(standin, "objects", "info")
        os.makedirs(alt_info, exist_ok=True)
        with open(os.path.join(alt_info, "alternates"), "w") as fp:
            fp.write(os.path.abspath(os.path.join(src_git, "objects"))
                     + "\n")

        # Writing packed-refs directly (not via update-ref) avoids
        # a fork per ref; _is_safe_refname already guarantees the
        # "<sha> <refname>" line needs no escaping.
        refs = _read_src_refs(src_git)
        with open(os.path.join(standin, "packed-refs"), "w") as fp:
            fp.write("# pack-refs with: peeled fully-peeled sorted \n")
            for refname, sha in sorted(refs):
                fp.write(f"{sha} {refname}\n")

        # Inherit src's shallow boundary so standin's upload-pack
        # stops at the shallow tips instead of dying on missing
        # parents.  Absent file = src isn't shallow, nothing to do.
        shallow_shas = _read_src_shallow_shas(src_git)
        if shallow_shas:
            with open(os.path.join(standin, "shallow"), "w") as fp:
                for sha in shallow_shas:
                    fp.write(sha + "\n")

        r = _run_git(
            [
                # Force-empty code-exec keys on both sides (propagates
                # to upload-pack via GIT_CONFIG_PARAMETERS).
                "-c", "uploadpack.packObjectsHook=",
                "-c", "core.fsmonitor=",
                "-c", "core.hooksPath=/dev/null",
                # Host-wide default is protocol.file.allow=user under
                # non-interactive builds; override since standin is
                # a local path we just created.
                "-c", "protocol.file.allow=always",
                # --no-tags disables git's implicit tag auto-follow
                # (which would land tags under refs/tags/* in the bare
                # and pollute the non-fetch-cache namespace); the
                # explicit +refs/tags/*:refs/fetch-cache/tags/* refspec
                # below still mirrors every tag into our own namespace.
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
    """Anchor each src ref tip under ``refs/fetch-cache/have/<sha>``
    (shallow pool variant).

    SHA-named refname + CAS ``update-ref <ref> <sha> ""`` gives the
    monotonic append-only guarantee (fetch-cache.md I2).  Shallow boundaries
    are fine here — shallow consumers stop at their own
    ``.git/shallow`` written by upstream negotiation, see
    :func:`_replicate_src_refs_checked` for the full-pool variant.
    """
    for sha in _read_src_ref_shas(src_dir):
        _run_git(
            ["update-ref", f"refs/fetch-cache/have/{sha}", sha, ""],
            cwd=bare,
        )


def _replicate_src_refs_checked(src_dir: str, bare: str,
                                verified: set[str]) -> None:
    """Full-pool variant of :func:`_replicate_src_refs`: skip SHAs
    whose ancestry is not fully present in *bare* (fetch-cache.md I5).

    *verified* is the memoization skip set — those SHAs bypass both
    ``rev-list`` and the redundant CAS ``update-ref``, which is the
    main steady-state speedup when most refs are unchanged.
    """
    for sha in _read_src_ref_shas(src_dir):
        if sha in verified:
            continue
        if not _is_connected(bare, sha):
            logger.info("%s: skip disconnected have/%s",
                        os.path.basename(bare), sha[:8])
            continue
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
# update — full-scan mode (manual rebuild)
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
    """Cache bare repos under ``<src>/.git/modules/`` (recursively).

    Git-dir probing is filesystem-only (I3).
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
    for entry in _iter_dir(modules_dir):
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
                           "build dir (manual rebuild)")

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
