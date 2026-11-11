# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Have every rank's executor worker print what its engine runs, as it sets the engine up.

The checks put this directory first on ``PYTHONPATH`` and set
``MV2FLOW_RANK_FACTS=1`` (``check_runtime.RANK_FACTS_ENV``), so this
``sitecustomize`` runs at interpreter start in every process of the check's
job. In each it waits for ``tensorrt_llm.executor.base_worker`` to be
imported and wraps ``BaseWorker.setup_engine``: once the worker's engine is
set up, the worker prints one ``MV2FLOW_RANK_ENGINE`` record with its rank,
its engine's resolved configuration (the worker's ``llm_args``, every field),
the class of the model it built and where it imported ``tensorrt_llm`` from.
``checks/attest.py`` compares the record with the options the check declared.

Without the variable this module only hands on to the next ``sitecustomize``
on ``sys.path``, the one it shadows; that hand-on moves forward along
``sys.path`` only, so two hooks of this kind never hand on to each other in a
loop.

Standard library only.
"""

import enum
import json
import os
import sys

_HERE = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
_ENABLED = os.environ.get("MV2FLOW_RANK_FACTS") == "1"
_MODULE = "tensorrt_llm.executor.base_worker"
_MARKER = "MV2FLOW_RANK_ENGINE"


def _plain(value, depth=0):
    """``value`` as JSON takes it: a config object as a mapping of its fields, an enum by its value."""
    if isinstance(value, enum.Enum):
        value = value.value
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth > 8:
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v, depth + 1) for v in value]
    fields = getattr(type(value), "model_fields", None)
    if fields:
        found = {}
        for name, field in fields.items():
            found[name] = _plain(getattr(value, name, None), depth + 1)
            # An options file may spell a field by an alias: the same value answers to it.
            alias = getattr(field, "validation_alias", None)
            spelled = [getattr(field, "alias", None), *(getattr(alias, "choices", None) or [alias])]
            for other in spelled:
                if isinstance(other, str):
                    found.setdefault(other, found[name])
        return found
    if hasattr(value, "__dict__") and not callable(value):
        public = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return {k: _plain(v, depth + 1) for k, v in public.items()}
    return str(value)


def _rank(worker):
    """The MPI rank of ``worker``: its own ``rank``, else the launcher's rank variables."""
    rank = getattr(worker, "rank", None)
    if isinstance(rank, int) and not isinstance(rank, bool):
        return rank
    for name in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID"):
        if os.environ.get(name, "").isdigit():
            return int(os.environ[name])
    return None


def _report(worker):
    """Print this rank's record; a worker that cannot be read prints why, which reads unknown."""
    record = {"rank": _rank(worker)}
    try:
        import tensorrt_llm

        model = getattr(
            getattr(getattr(worker, "engine", None), "model_engine", None), "model", None
        )
        record["options"] = _plain(worker.llm_args)
        record["target"] = (
            f"{type(model).__module__}.{type(model).__qualname__}" if model is not None else None
        )
        record["origin"] = str(getattr(tensorrt_llm, "__file__", "") or "")
    # An observation of an engine that is up must never stop it.
    except Exception as exc:  # noqa: BLE001
        record = {"rank": record["rank"], "error": f"{type(exc).__name__}: {exc}"[:500]}
    sys.stdout.write(f"{_MARKER} {json.dumps(record, sort_keys=True, default=str)}\n")
    sys.stdout.flush()


def _wrap(module):
    cls = getattr(module, "BaseWorker", None)
    original = getattr(cls, "setup_engine", None)
    if original is None or getattr(original, "_mv2flow_observer", False):
        return

    def setup_engine(worker, *args, **kwargs):
        result = original(worker, *args, **kwargs)
        try:
            _report(worker)
        # An observation of an engine that is up must never stop it.
        except Exception as exc:  # noqa: BLE001
            try:
                sys.stderr.write(f"[mv2flow-rank-facts] not read: {type(exc).__name__}: {exc}\n")
            except (OSError, ValueError):
                pass
        return result

    setup_engine.__wrapped__ = original
    setup_engine._mv2flow_observer = True
    cls.setup_engine = setup_engine


class _Loader:
    """The module's own loader, with :func:`_wrap` run once the module is executed."""

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _wrap(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _Watcher:
    """Find :data:`_MODULE` with the other finders and hand back its spec, loader wrapped."""

    def find_spec(self, name, path=None, target=None):
        if name != _MODULE:
            return None
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        for finder in list(sys.meta_path):
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(name, path, target)
            if spec is None:
                continue
            if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                spec.loader = _Loader(spec.loader)
            return spec
        return None


def _chain():
    """Run the next ``sitecustomize`` on ``sys.path`` after this one's directory."""
    entries = [os.path.realpath(os.path.abspath(e or ".")) for e in sys.path]
    start = entries.index(_HERE) + 1 if _HERE in entries else 0
    for directory in entries[start:]:
        if directory == _HERE:
            continue
        candidate = os.path.join(directory, "sitecustomize.py")
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as handle:
                code = compile(handle.read(), candidate, "exec")
            exec(code, {"__name__": "sitecustomize", "__file__": candidate})
            return


if _ENABLED:
    sys.meta_path.insert(0, _Watcher())
_chain()
