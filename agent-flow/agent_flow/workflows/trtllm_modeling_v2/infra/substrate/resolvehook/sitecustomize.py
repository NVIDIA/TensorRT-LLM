# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record, in every Python process of a spool job, which file each watched module came from.

The spool scheduler puts this directory first on every job's ``PYTHONPATH``,
so this ``sitecustomize`` runs at interpreter start in the job's shell, in
every MPI rank and in every worker the engine spawns. For each module named in
``MV2FLOW_RESOLVE_MODULES`` (default ``tensorrt_llm``) it records the file
the import resolved to:

- one JSON line appended to ``$MV2FLOW_RESOLVE_LOG``;
- one ``[mv2flow-resolve] ...`` line on stderr.

The scheduler compares the records after the job: ranks that imported the
package from different checkouts, or from a checkout other than the job's
``tree``, make the result unusable, and the receipt says so.

Afterwards the next ``sitecustomize`` on ``sys.path`` runs, so a site or
distribution hook this one shadows still takes effect.
"""

import importlib.abc
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


def _record(module, spec):
    origin = None
    if spec is not None:
        origin = spec.origin
        if origin in (None, "namespace") and spec.submodule_search_locations:
            origin = list(spec.submodule_search_locations)[0]
    entry = {
        "module": module,
        "origin": origin,
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

    Only the entries after this one's directory are searched: another hook of
    this kind earlier on the path hands on forward to this one, and handing
    back to it would loop.
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
