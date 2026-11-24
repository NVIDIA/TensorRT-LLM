# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Settings that shared runtime code reads for a modeling_v2 target (the switchboard). The settings come from the
target's profile; none is read from the environment.

The environment is read in two places only. :func:`disagg_role` takes the server role, which selects the profile,
from ``TRTLLM_DISAGG_ROLE``: ``trtllm-serve --server_role`` publishes the role there (and in
``llm_args.env_overrides``, read first, for MPI ranks spawned before the variable was set), and no other input of
the model carries it. :func:`warn_retired_env` only looks for the switches the profiles replaced, to warn that
setting them has no effect.

A modeling_v2 target that runs one fixed set of optimizations per deployment profile keeps the settings read inside
its own package as its features; the ones read by code outside it (the DSA backend and the KV-V2 cache manager) live
here. The target's ``route()`` picks the profile and calls :func:`configure` with its :class:`DecodeSwitches` before
the target's modules are built.

Before :func:`configure`, and in every process that never calls it (every other model), :func:`active` is
``DecodeSwitches()``: each default is the value the shared code takes when the setting is off, so those deployments
run the code paths they ran before these settings existed.

Read sites call :func:`active` when they run, never at import: the shared modules are imported before ``route()``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
from collections.abc import Mapping
from typing import Any, Dict, Iterable, Iterator, Optional


@dataclasses.dataclass(frozen=True)
class DecodeSwitches:
    """One profile's settings for shared code; each default is the plain path's value."""

    # attention/backends/sparse/dsa/metadata.py: fused Helix DSA decode metadata.
    dsa_helix_fused_meta: bool = False
    # attention/backends/sparse/dsa/helix_key_shard.py: key-sharded decode Top-K over the Helix ranks.
    dsa_helix_key_shard: bool = False
    # pyexecutor/kv_cache/kv_cache_manager_v2.py: zero the GPU pools at construction.
    kv_v2_zero_gpu_pools: bool = False

    def changed(self) -> Dict[str, Any]:
        """The fields that differ from the plain path, by name."""
        base = DecodeSwitches()
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) != getattr(base, f.name)
        }


#: The environment variables these fields replaced -> the field. They are no longer read; see :func:`configure`.
RETIRED_ENV: Dict[str, str] = {
    "TRTLLM_DSA_HELIX_FUSED_META": "dsa_helix_fused_meta",
    "TRTLLM_DSA_HELIX_KEY_SHARD": "dsa_helix_key_shard",
    "TRTLLM_KV_V2_ZERO_GPU_POOLS": "kv_v2_zero_gpu_pools",
}

#: The serving role that ``trtllm-serve --server_role`` publishes to the process and to every MPI rank.
DISAGG_ROLE_ENV = "TRTLLM_DISAGG_ROLE"

_ACTIVE = DecodeSwitches()
_SOURCE: Optional[str] = None
# Where active() was first called before configure() (a read site that may have kept the plain path's value).
_EARLY_READER: Optional[str] = None
_WARNED_ENV: set = set()


def _logger():
    from tensorrt_llm.logger import logger

    return logger


def active() -> DecodeSwitches:
    """The configured settings (``DecodeSwitches()`` before :func:`configure`). Call it where the setting is used."""
    if _SOURCE is None and _EARLY_READER is None:
        _note_early_reader()
    return _ACTIVE


def _note_early_reader() -> None:
    global _EARLY_READER
    try:
        frame = sys._getframe(2)
        _EARLY_READER = f"{frame.f_code.co_filename}:{frame.f_lineno}"
    except ValueError:
        _EARLY_READER = "<unknown>"


def configured() -> Optional[str]:
    """The source that configured the settings (the profile), or None."""
    return _SOURCE


def configure(switches: DecodeSwitches, source: str, *, warn_retired: bool = True) -> None:
    """Make ``switches`` the active settings; once per process.

    Configuring the same settings again is a no-op; different settings raise, since shared code may already have
    run with the first ones. ``warn_retired=False`` skips :func:`warn_retired_env` (a caller that reads no
    environment variable).
    """
    global _ACTIVE, _SOURCE
    if not isinstance(switches, DecodeSwitches):
        raise TypeError(f"decode switches: expected DecodeSwitches, got {type(switches).__name__}")
    if _SOURCE is not None:
        if switches != _ACTIVE:
            raise RuntimeError(
                f"decode switches: configured from {_SOURCE!r} in this process, cannot configure from "
                f"{source!r}: {_diff(_ACTIVE, switches)}"
            )
        return
    _ACTIVE, _SOURCE = switches, source
    log = _logger()
    changed = switches.changed()
    line = f"decode switches: {source} ({len(changed)} set)"
    if changed:
        line += ": " + ", ".join(f"{name}={value}" for name, value in sorted(changed.items()))
    log.info(line)
    if changed and _EARLY_READER is not None:
        log.warning(
            f"decode switches: read before they were configured (first at {_EARLY_READER}); a read site "
            "that kept that value runs the plain path"
        )
    if warn_retired:
        warn_retired_env(RETIRED_ENV)


def _diff(old: DecodeSwitches, new: DecodeSwitches) -> str:
    return ", ".join(
        f"{f.name}={getattr(old, f.name)}->{getattr(new, f.name)}"
        for f in dataclasses.fields(old)
        if getattr(old, f.name) != getattr(new, f.name)
    )


@contextlib.contextmanager
def override(**fields: Any) -> Iterator[DecodeSwitches]:
    """Tests: replace some fields of the active settings inside the ``with`` block, then restore them."""
    global _ACTIVE
    saved = _ACTIVE
    _ACTIVE = dataclasses.replace(saved, **fields)
    try:
        yield _ACTIVE
    finally:
        _ACTIVE = saved


def disagg_role(llm_args: Any = None) -> str:
    """``"context"`` on a disaggregated context server, else ``"generation"`` (also a server started without a
    role, which cannot be told apart from an aggregated one).

    ``trtllm-serve --server_role`` puts the role in the process environment and in ``llm_args.env_overrides``,
    which reach MPI ranks spawned before it was set; the overrides are read first.
    """
    overrides = (
        llm_args.get("env_overrides")
        if isinstance(llm_args, Mapping)
        else getattr(llm_args, "env_overrides", None)
    )
    role = (overrides or {}).get(DISAGG_ROLE_ENV)
    if role is None:
        role = os.environ.get(DISAGG_ROLE_ENV)
    return "context" if role == "context" else "generation"


def warn_retired_env(names: Iterable[str]) -> None:
    """Warn once about each of ``names`` that the environment still sets: nothing reads them any more."""
    stale = sorted(name for name in names if name in os.environ and name not in _WARNED_ENV)
    if not stale:
        return
    _WARNED_ENV.update(stale)
    _logger().warning(
        "decode switches: these environment variables are no longer read (the target's profile decides): "
        + ", ".join(stale)
    )


def _reset() -> None:
    """Tests: forget the configuration, as in a fresh process."""
    global _ACTIVE, _SOURCE, _EARLY_READER
    _ACTIVE, _SOURCE, _EARLY_READER = DecodeSwitches(), None, None
    _WARNED_ENV.clear()
