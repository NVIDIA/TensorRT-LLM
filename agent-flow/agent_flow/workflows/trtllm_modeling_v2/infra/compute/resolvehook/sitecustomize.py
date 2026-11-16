# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record, in every Python process of a spool job, which file each watched module came from.

The spool scheduler puts this directory first on every job's ``PYTHONPATH``, so this
``sitecustomize`` runs at interpreter start in the job's shell, every MPI rank and every worker
the engine spawns. For each module in ``MV2FLOW_RESOLVE_MODULES`` (default ``tensorrt_llm``) it
appends one JSON line to ``$MV2FLOW_RESOLVE_LOG`` (the origin, the identity of the native build
beside it, the process's role instance ``$MV2FLOW_ROLE`` and its world rank) and writes one
``[mv2flow-resolve] ...`` line to stderr; the scheduler judges the records (``provenance.py``).
The next ``sitecustomize`` on ``sys.path`` then runs, so a site or distribution hook this one
shadows still takes effect.
"""

import importlib.abc
import importlib.util
import json
import os
import socket
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


def _rank():
    for name in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "RANK", "SLURM_PROCID"):
        if os.environ.get(name):
            return os.environ[name]
    return "-"


def _root(module, origin):
    """The directory ``module`` was imported from: the one that holds its top-level package."""
    package = os.path.dirname(origin) if origin.endswith("__init__.py") else origin
    for _ in module.split("."):
        package = os.path.dirname(package)
    return package


def _pintree():
    """``pintree.py``, which sits beside this directory, loaded by its path.

    This file runs where the flow's package may not be importable.
    """
    name = "mv2flow_pintree"
    if name not in sys.modules:
        path = os.path.join(os.path.dirname(_HERE), "pintree.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


def _build(module, origin):
    """The identity of the native build beside ``origin``: by the content of its libraries.

    ``pintree.build_id``: the name of the build record of the libraries the package holds, or,
    in a pinned job, of exactly the ones the pin's record names (``$MV2FLOW_TREE_LIBS``, paths
    in the tree). Other bytes are another identity whatever a file's size and times say; the
    same bytes anywhere are the same one. Each library's digest is read from the node's cache
    when the file has not changed since some process digested it, so a job's ranks do not each
    read every library. ``None`` when the package holds no library, or the identity cannot be
    taken: a record without one counts for no rank.
    """
    if not origin:
        return None
    try:
        named = os.environ.get("MV2FLOW_TREE_LIBS")
        names = [n for n in named.split(":") if n] if named is not None else None
        top = module.split(".")[0]
        return _pintree().build_id(_root(module, origin), names=names, packages=[top])
    except Exception:
        return None


def _record(module, spec):
    origin = None
    if spec is not None:
        origin = spec.origin
        if origin in (None, "namespace") and spec.submodule_search_locations:
            origin = list(spec.submodule_search_locations)[0]
    entry = {
        "module": module,
        "origin": origin,
        "build": _build(module, origin),
        "role": os.environ.get("MV2FLOW_ROLE") or None,
        "pid": os.getpid(),
        "rank": _rank(),
        "host": socket.gethostname(),
        "tree": os.environ.get("MV2FLOW_TREE"),
        "job": os.environ.get("MV2FLOW_SPOOL_JOB"),
        "time": time.time(),
    }
    log = os.environ.get("MV2FLOW_RESOLVE_LOG")
    if log:
        try:
            with open(log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError:
            pass
    try:
        sys.stderr.write(
            f"[mv2flow-resolve] pid={entry['pid']} rank={entry['rank']} "
            f"module={module} origin={origin}\n"
        )
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


class _ResolveRecorder(importlib.abc.MetaPathFinder):
    """Ask the other finders where a watched module is, record the answer, return it."""

    def __init__(self, modules):
        self.remaining = set(modules)

    def find_spec(self, name, path=None, target=None):
        if name not in self.remaining:
            return None
        self.remaining.discard(name)
        spec = None
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(name, path, target)
            if spec is not None:
                break
        if not self.remaining and self in sys.meta_path:
            sys.meta_path.remove(self)
        _record(name, spec)
        return spec


def _install():
    modules = [
        m.strip()
        for m in os.environ.get("MV2FLOW_RESOLVE_MODULES", "tensorrt_llm").split(",")
        if m.strip()
    ]
    if modules:
        sys.meta_path.insert(0, _ResolveRecorder(modules))


def _chain():
    """Run the next ``sitecustomize`` on ``sys.path``, which this one shadows.

    Only entries after this one's directory are searched: an earlier hook of this kind hands on
    forward to this one, and handing back to it would loop.
    """
    here = os.path.realpath(_HERE)
    entries = [os.path.realpath(os.path.abspath(e or ".")) for e in sys.path]
    start = entries.index(here) + 1 if here in entries else 0
    for directory in entries[start:]:
        if directory == here:
            continue
        candidate = os.path.join(directory, "sitecustomize.py")
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as handle:
                code = compile(handle.read(), candidate, "exec")
            exec(code, {"__name__": "sitecustomize", "__file__": candidate})
            return


_install()
_chain()
