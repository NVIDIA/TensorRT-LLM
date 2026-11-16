# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one rank's worker says of its engine, for a session's attestation.

The rank hook (``rankhook/sitecustomize.py``) calls :func:`emit_rank` once the worker's
engine is set up. The engine has no public report of a worker's resolved configuration,
so the record is read from the worker itself.
"""

from __future__ import annotations

import enum
import os
import socket
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import workload_io as wio  # noqa: E402 -- imported by path, see the package docstring


def worker_rank(worker: Any) -> int | None:
    """The MPI rank of ``worker``: its own ``rank``, else the launcher's rank variables."""
    rank = getattr(worker, "rank", None)
    if isinstance(rank, int) and not isinstance(rank, bool):
        return rank
    for name in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID"):
        text = os.environ.get(name, "")
        if text.isdigit():
            return int(text)
    return None


#: What one record holds at most of a collection or of an object that is no configuration:
#: a tokenizer's vocabulary or a loader's state is not an option, and a record is one line.
_MAX_ITEMS, _MAX_TEXT = 256, 400


def plain_config(value: Any, depth: int = 0) -> Any:
    """``value`` as JSON takes it: a config object as a mapping of its fields, an enum by its value.

    Only the configuration is walked: the worker's arguments, and under them mappings,
    lists and config objects (their ``model_fields``). Anything else is named by its text.
    """
    if isinstance(value, enum.Enum):
        value = value.value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_TEXT]
    if isinstance(value, (dict, list, tuple, set, frozenset)) and len(value) > _MAX_ITEMS:
        return f"<{type(value).__name__} of {len(value)}>"
    if depth > 8:
        return str(value)[:_MAX_TEXT]
    if isinstance(value, dict):
        return {str(k): plain_config(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain_config(v, depth + 1) for v in value]
    fields = getattr(type(value), "model_fields", None)
    if fields:
        found = {}
        for name, field in fields.items():
            found[name] = plain_config(getattr(value, name, None), depth + 1)
            # An options file may spell a field by an alias: the same value answers to it.
            alias = getattr(field, "validation_alias", None)
            spelled = [getattr(field, "alias", None), *(getattr(alias, "choices", None) or [alias])]
            for other in spelled:
                if isinstance(other, str):
                    found.setdefault(other, found[name])
        return found
    if depth == 0 and hasattr(value, "__dict__"):
        # Arguments that are no config object: their public attributes are the options.
        public = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return {k: plain_config(v, depth + 1) for k, v in public.items()}
    return str(value)[:_MAX_TEXT]


def observe(worker: Any, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """What a deployment's session compares with what it declared, for one rank.

    ``options``: the worker's resolved arguments, every field. ``target``: the class of
    the model it built. ``origin``: where it imported ``tensorrt_llm`` from. ``host`` and
    ``devices``: where it runs and the devices it was shown. ``role`` and ``instance``:
    which server of the session it belongs to, as the session labelled it.
    """
    import tensorrt_llm

    environ = os.environ if environ is None else environ
    engine = getattr(worker, "engine", None)
    model = getattr(getattr(engine, "model_engine", None), "model", None)
    return {
        "options": plain_config(worker.llm_args),
        "target": f"{type(model).__module__}.{type(model).__qualname__}"
        if model is not None
        else None,
        "origin": str(getattr(tensorrt_llm, "__file__", "") or ""),
        # The scheduler's name for this node where there is one: what a session's
        # placement is stated in.
        "host": environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "devices": [d for d in str(environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if d],
        "role": environ.get("MV2FLOW_SERVER_ROLE") or environ.get("MV2FLOW_ROLE") or None,
        "instance": environ.get("MV2FLOW_INSTANCE") or None,
    }


def emit_rank(worker: Any) -> None:
    """Print the ``MV2FLOW_RANK_ENGINE`` record of one rank's worker, as its engine is up.

    The record holds what a session attests (:func:`observe`). A worker that cannot be
    read prints ``{"rank", "error"}``, which the check reports as a rank it could not attest.
    """
    try:
        record = {**observe(worker), "rank": worker_rank(worker)}
    except Exception as exc:  # noqa: BLE001 -- an observation never fails the engine it reads
        record = {"rank": worker_rank(worker), "error": f"{type(exc).__name__}: {exc}"[:500]}
    keep_rank_record(record)
    wio.emit(wio.RANK_ENGINE_MARKER, record)


def keep_rank_record(record: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> None:
    """Write this rank's record to its own file, where the session named a directory for them.

    The ranks of one world print to one stream, and two records printed at once can
    arrive as one broken line. A file of this rank's own has no other writer. The line
    printed after it stays, for a log's reader and for a session that named no directory.
    """
    environ = os.environ if environ is None else environ
    directory = environ.get(wio.RANK_RECORDS_ENV)
    if not directory:
        return
    instance = environ.get("MV2FLOW_ROLE") or "server"
    try:
        wio.keep_record(wio.rank_record_file(directory, instance, record.get("rank")), record)
    except OSError:
        pass  # an observation: the printed record is still there
