# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The key of a row of evidence: what a check measured, by content.

A row of the ledger (``contract_status.jsonl``) is a check's result on one key. The key is
a digest of everything the check consumes, taken by content: two states of the target that
give a check the same key hold the same evidence for it, whatever their commits are called
and wherever their files sit; any difference in what the check consumes is another key.
Only exact equality counts. ``""`` means "not established": something the check consumes
has no content identity, and then nothing is reused for it.

    key = sha256(canonical JSON of {schema, check, source, build, inputs, prerequisites})

``check``
    The check's id, its class, and the content of the flow's own checkers (``checks/``).
``source``
    The commit's files, each by git's blob id and mode, as the check's class sees them
    (:func:`consumed`). Three fixed projections, no analysis of imports:

    ===========  =====================================================================
    ``session``  a deployment's session (``deployment:<name>``): the whole tree less
                 documentation, CI configuration, test registration, and the
                 repository's tests and examples
    ``native``   a native test run (``kernel:``, ``cpu:``, ``tests:``,
                 the final tests): the session's, and the repository's tests and their
                 helpers
    ``custom``   anything else: the whole tree
    ===========  =====================================================================

    A file the spec names as an input of a check (an options file, a prompts file, an
    image, an anchor table: :func:`gate_inputs`) is in every projection, wherever it
    lives. So a commit that only adds
    documentation or registers a test leaves a session's key as it is, and changes the
    key of the tests it touches.
``build``
    The native libraries the tree loads, each by the sha256 of its bytes and its place in
    the tree (``pintree.build_record``): the same bytes on another disk at another time
    are the same build, and other bytes under the same name, size and time are not.
``inputs``
    What the check reads from outside the tree, as the contract digests it
    (``contract.check_inputs``): the deployment as the spec resolves it, the checkpoint
    with its tokenizer and processor, the workload, datasets and protocol, the reference
    and the image. For a row's check: the row, and its queue item, as the survey wrote
    them. For the final tests: the runtime they run in (``contract.runtime_inputs``).
    Scheduler values (allocation ids, accounts, deadlines, walls) are in none. Nor is
    what a directory holds that an environment variable names (the scratch, a cache,
    the run's own cache of compiled kernels): the run writes there as it goes, and a
    key that moved with it would measure an unchanged candidate again.

    What of them is read by content, and what is not, on purpose: a file up to 64 MiB
    by its bytes; a larger one (an image, a weight file) by its size and, when it is a
    link, by what it links to; in a directory, the files up to 32 MiB each (256 MiB in
    all) by their bytes and the rest by place and size. Hashing the weights would cost
    more than the load the key saves, so a weight file rewritten in place to the same
    size is not seen; a model hub's snapshot links each file to an object named for
    its content, and that name is in the key. A file's digest is remembered by its
    path, size and modification time. What is named and is not there has no identity:
    a checkpoint given by an absolute path that does not exist gives a session no key
    (``""``), and nothing is reused for it.
``prerequisites``
    The key of each result the check is judged against: for a deployment judged relative
    to another, that one's session on the same commit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Collection, Iterator, Mapping, Sequence

from ..checks import boot as boot_checks
from ..infra.compute import pintree
from ..infra.git import CommitPolicy, git_output
from ..operators import coverage as vocab_mod
from ..operators import rows as rows_mod
from ..spec import deployments as deployments_mod
from .validation import contract as contract_mod

#: The key's schema: a change to what a key covers is another schema, and another key.
SCHEMA = "mv2-evidence-v1"
#: The three classes of check, by what of the tree each consumes.
SESSION, NATIVE, CUSTOM = "session", "native", "custom"
#: The check ids of each class, by the part before the colon.
SESSION_CHECKS: tuple[str, ...] = ("deployment",)
NATIVE_CHECKS: tuple[str, ...] = ("kernel", "cpu", "tests", "final-tests")
#: The checks of one row of the survey: their inputs are the row.
ROW_CHECKS: tuple[str, ...] = ("kernel", "cpu")

#: Where no check of a fixed class reads: documentation and CI configuration.
PROSE_PREFIXES: tuple[str, ...] = ("docs/", ".github/")
PROSE_SUFFIXES: tuple[str, ...] = (".md",)
#: Test registration no test runner consumes: CI's lists of what it runs.
REGISTRATION_PREFIXES: tuple[str, ...] = ("tests/integration/test_lists/",)
#: What a deployment's session does not load: the repository's tests and examples.
TESTS_PREFIXES: tuple[str, ...] = ("tests/", "examples/")
#: The spec's blocks whose values may name a file a check reads.
GATE_INPUT_BLOCKS: tuple[str, ...] = ("accuracy", "deployments", "environment")

#: What gives a check's inputs digest by the check's id; ``None`` when it is unknown.
InputsOf = Callable[[str], "str | None"]


def check_class(check_id: str) -> str:
    """The class of ``check_id``: which projection of the tree it consumes."""
    name = check_id.partition(":")[0]
    if name in SESSION_CHECKS:
        return SESSION
    if name in NATIVE_CHECKS:
        return NATIVE
    return CUSTOM


def consumed(
    kind: str, path: str, *, keep: Collection[str] = (), denied: Sequence[str] = ()
) -> bool:
    """Whether a check of class ``kind`` consumes ``path``, relative to the checkout.

    ``denied`` are the run's own paths (its workspace), in no key. ``keep`` are the files
    the spec names as inputs (:func:`gate_inputs`), in every key: an explicit input wins
    over every exclusion. Markdown is documentation wherever it lies, the runtime
    package included: a guide or a note a task edits there after a session passed would
    otherwise cost that session's load again. Markdown a check does read is named by the
    spec, and is then an input like any other.
    """
    if path.startswith(tuple(denied)):
        return False
    if path in keep or kind == CUSTOM:
        return True
    if (
        path.startswith(PROSE_PREFIXES)
        or path.lower().endswith(PROSE_SUFFIXES)
        or path.startswith(REGISTRATION_PREFIXES)
    ):
        return False
    return kind == NATIVE or not path.startswith(TESTS_PREFIXES)


def gate_inputs(task_data: Mapping[str, Any], repo: str | Path) -> frozenset[str]:
    """The files in the checkout a check reads as inputs, relative to the checkout.

    The image a multimodal boot sends when the spec names none, and every file inside
    the checkout that a check-facing block of the spec names
    (:data:`GATE_INPUT_BLOCKS`): an options file, a prompts file, an image, a script in
    a command, and the accuracy anchor tables.
    """
    root = Path(repo)
    keep = {boot_checks.DEFAULT_BOOT_IMAGE}
    for block in GATE_INPUT_BLOCKS:
        for word in _path_words(task_data.get(block)):
            path = _in_checkout(word, root)
            if path is not None:
                keep.add(path)
    return frozenset(keep)


def _in_checkout(name: str, root: Path) -> str | None:
    """``name``, a path absolute or relative to ``root``, as a file in the checkout; else ``None``."""
    candidate = Path(name) if os.path.isabs(name) else root / name
    try:
        if candidate.is_file():
            return candidate.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        pass
    return None


def _path_words(value: Any) -> Iterator[str]:
    """Every word of every string in ``value`` that could name a checkout file."""
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _path_words(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _path_words(item)
    elif isinstance(value, str):
        for word in value.split():
            word = word.strip("'\"").rpartition("=")[2]
            if word:
                yield word


def tree_entries(listing: str) -> dict[str, str]:
    """``{path: "<mode> <object id>"}`` from ``git ls-tree -r -z``.

    The mode is part of a file's identity: an executable bit, or a link (whose object is
    its target).
    """
    entries: dict[str, str] = {}
    for entry in listing.split("\0"):
        meta, tab, path = entry.partition("\t")
        fields = meta.split()
        if tab and len(fields) == 3:
            entries[path] = f"{fields[0]} {fields[2]}"
    return entries


def committed(repo: str | Path, commit: str, git=git_output) -> dict[str, str] | None:
    """Every path of ``commit``'s tree with its mode and object id; ``None`` when git cannot list it."""
    listing = git(repo, "ls-tree", "-r", "-z", "--full-tree", commit)
    return None if listing is None else tree_entries(listing)


def build_of(repo: str | Path) -> dict[str, str]:
    """The native build the checkout holds, by content (``pintree.build_record``).

    The digests are remembered by each library's identity on disk (inode, size and both
    of its times: ``pintree.DigestCache``), so a library is read once per build and a
    rewrite under the same name, size and modification time is read again.
    """
    return pintree.build_record(Path(repo), cache=pintree.default_cache())


_CHECKER: list[str] = []


def checker_digest() -> str:
    """The content of the flow's own checkers: what measures and judges (``checks/``, and this file).

    Read once per process: a process runs the code it started with. What says which tree
    a check's commands imported (and whether a result is void for it) is not in it: that
    is decided when a result is measured and is written in its row, and a change to it
    is no reason to load every deployment of a resumed run again.
    """
    if not _CHECKER:
        root = Path(__file__).resolve().parent.parent
        digest = hashlib.sha256()
        files = sorted((root / "checks").rglob("*.py")) + [Path(__file__).resolve()]
        for path in files:
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
        _CHECKER.append(digest.hexdigest()[:16])
    return _CHECKER[0]


def _digest(value: Any) -> str:
    """One digest of ``value`` as canonical JSON: sorted keys, no NaN, no ambiguity of spacing."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_digest(entries: Mapping[str, str]) -> str:
    """One digest of a projection of a tree: each path with its mode and object id."""
    digest = hashlib.sha256()
    for path in sorted(entries):
        digest.update(os.fsencode(f"{path}\0{entries[path]}\n"))
    return digest.hexdigest()


def row_inputs(workspace: str | Path, check_id: str) -> str:
    """What a row's check is asked to show, as one digest: the row, and its queue item.

    As the survey wrote them and the run keeps them (``work.yaml``, ``vocabulary.yaml``),
    less what only sizes the job (its wall and nodes). A row the run does not hold is
    digested as unknown: its tests are in the source projection all the same, and no
    task of the run reads a result for a row it does not hold.
    """
    _, _, ref = check_id.partition(":")
    found: dict[str, Any] = {"row": None, "item": None}
    try:
        row = next((r for r in rows_mod.load_work(workspace) if r.id == ref), None)
    except Exception:  # a work file that cannot be read says nothing of the row
        row = None
    if row is not None:
        stated = dataclasses.asdict(row)
        for name in ("wall", "nodes"):
            stated.pop(name, None)
        found["row"] = stated
    path = Path(workspace) / vocab_mod.VOCABULARY_FILENAME
    try:
        items = vocab_mod.load_vocabulary(path) if path.is_file() else []
    except Exception:  # an unreadable vocabulary names no item
        items = []
    item = next((i for i in items if str(i.n) == ref), None)
    if item is not None:
        found["item"] = dataclasses.asdict(item) if dataclasses.is_dataclass(item) else str(item)
    return _digest(found)[:16]


def key(
    task_data: Mapping[str, Any],
    workspace: str | Path,
    commit: str,
    check_id: str,
    *,
    inputs_of: InputsOf,
    entries: Mapping[str, str] | None = None,
    build: Mapping[str, str] | None = None,
    extra: Any = None,
    git=git_output,
) -> str:
    """The key of ``check_id``'s evidence on ``commit``; ``""`` when it is not established.

    ``inputs_of`` gives the inputs digest of a contract check by its id, as the run's
    contract has it. ``entries`` is the commit's listing (:func:`committed`) and
    ``build`` the native build's record when the caller has them. ``extra`` is whatever
    else the check consumes that only its caller knows (the final tests' selection).
    Not established: git cannot list the commit, or an input of the check has no
    identity (its inputs are unknown, or the checkpoint a session loads is named by an
    absolute path that is not there).
    """
    repo = str(task_data.get("trtllm_repo_path") or "")
    if not repo or not commit:
        return ""
    kind = check_class(check_id)
    name, _, ref = check_id.partition(":")
    checkpoint = task_data.get("checkpoint_path")
    if kind == SESSION and isinstance(checkpoint, str) and os.path.isabs(checkpoint):
        if not os.path.exists(checkpoint):
            return ""  # a name with nothing behind it identifies no content
    try:
        if entries is None:
            entries = committed(repo, commit, git)
            if entries is None:
                return ""
        policy = CommitPolicy.from_task(task_data, workspace=workspace)
        keep = gate_inputs(task_data, repo)
        source = {
            path: entry
            for path, entry in entries.items()
            if consumed(kind, path, keep=keep, denied=policy.denied_paths)
        }
        if build is None:
            build = build_of(repo)
    except (OSError, ValueError):
        # Unreadable, or a name git printed that is not text: unknown, not unchanged.
        return ""
    if name in ROW_CHECKS:
        inputs: str | None = row_inputs(workspace, check_id)
    elif name == "final-tests":
        inputs = contract_mod.runtime_inputs(task_data)
    else:
        inputs = inputs_of(check_id)
    if inputs is None:
        return ""
    prerequisites: list[str] = []
    if kind == SESSION:
        base = _relative_to(task_data, ref)
        if base:
            # Judged against that deployment's own score: on other evidence of it, this
            # is another judgment.
            first = key(
                task_data,
                workspace,
                commit,
                f"{name}:{base}",
                inputs_of=inputs_of,
                entries=entries,
                build=build,
                git=git,
            )
            if not first:
                return ""
            prerequisites.append(first)
    payload = {
        "schema": SCHEMA,
        "check": {"id": check_id, "class": kind, "implementation": checker_digest()},
        "source": source_digest(source),
        "build": pintree.record_digest(dict(build)),
        "inputs": inputs,
        "prerequisites": prerequisites,
        "extra": extra,
    }
    return f"key:{_digest(payload)[:32]}"


def _relative_to(task_data: Mapping[str, Any], name: str) -> str:
    """The deployment ``name`` is judged against, when it has benchmarks; ``""`` for none."""
    try:
        found = deployments_mod.resolve(task_data)
    except Exception:  # a spec that does not resolve names no deployment
        return ""
    deployment = next((d for d in found if d.name == name), None)
    if deployment is None or not deployment.benchmarks or not deployment.relative_to:
        return ""
    return deployment.relative_to if any(d.name == deployment.relative_to for d in found) else ""


__all__ = [
    "CUSTOM",
    "GATE_INPUT_BLOCKS",
    "NATIVE",
    "SCHEMA",
    "SESSION",
    "build_of",
    "check_class",
    "checker_digest",
    "committed",
    "consumed",
    "gate_inputs",
    "key",
    "row_inputs",
    "source_digest",
    "tree_entries",
]
