# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Environment switches of dense DWDP; context workers only, no ``llm_args`` field.

* ``TRTLLM_DWDP_DENSE`` -- turn dense DWDP on; default off.
* ``TRTLLM_DWDP_DENSE_SEGMENTS`` -- ``attn,moe`` (default) or ``attn`` for attention only.
* ``TRTLLM_DWDP_DENSE_REQUIRED`` -- default on: a failed setup raises on every rank; off: every rank
  drops dense DWDP and keeps serving with replicated dense weights.
* ``TRTLLM_DWDP_DENSE_MIN_BYTES`` -- storages smaller than this stay replicated; default 1 MiB.
* ``TRTLLM_DWDP_DENSE_STORE_CHUNK_MB`` -- 0 (default): one store handle; > 0: whole layers per
  handle of about this many MiB.
* ``TRTLLM_DWDP_DENSE_STRICT`` -- turn demand fetches, leaks, unknown large attributes, shared
  storages and slot watchdog mismatches into errors; default off.
* ``TRTLLM_DWDP_DENSE_DEBUG`` -- 1: fill the slots with NaN at commit, and name the holders of
  the first 8 moved storages that outlive the commit (each a scan of every live Python object);
  2: also run the per-layer slot checksum watchdog (each layer's H1 checks its slots against the
  layer's checksum; slow: every slot is read once more and the host waits for the GPU once per
  forward); default 0.

Booleans accept ``1/true/yes/on`` and ``0/false/no/off`` (any case); unset or empty means the
default.  A flag that defaults to off is on only for an explicit true value; any other value logs
a warning and keeps the default.  Integers that do not parse, or are negative, log a warning and
keep the default.  An invalid segment list raises.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from tensorrt_llm.logger import logger

from .types import (
    DEBUG_FILL_BYTE,
    DEFAULT_MIN_BYTES,
    LOG_PREFIX,
    SEGMENT_ORDER,
    DigestSwitches,
    SegmentKind,
)

DENSE_ENV = "TRTLLM_DWDP_DENSE"
DENSE_SEGMENTS_ENV = "TRTLLM_DWDP_DENSE_SEGMENTS"
DENSE_REQUIRED_ENV = "TRTLLM_DWDP_DENSE_REQUIRED"
DENSE_MIN_BYTES_ENV = "TRTLLM_DWDP_DENSE_MIN_BYTES"
DENSE_STORE_CHUNK_MB_ENV = "TRTLLM_DWDP_DENSE_STORE_CHUNK_MB"
DENSE_STRICT_ENV = "TRTLLM_DWDP_DENSE_STRICT"
DENSE_DEBUG_ENV = "TRTLLM_DWDP_DENSE_DEBUG"

_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")


def _bool_env(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    logger.warning(
        f"{LOG_PREFIX} {name}={raw!r} is not one of 1/true/yes/on or 0/false/no/off; "
        f"using {int(default)}"
    )
    return default


def _int_env(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        value = -1
    if value < 0:
        logger.warning(
            f"{LOG_PREFIX} {name}={raw!r} is not a non-negative integer; using {default}"
        )
        return default
    return value


def parse_segments(value: str) -> tuple[SegmentKind, ...]:
    """Parse a ``TRTLLM_DWDP_DENSE_SEGMENTS`` value.

    Args:
        value: Comma-separated segment names, any case, surrounding blanks ignored, repeats
            allowed; e.g. ``"attn,moe"`` or ``"attn"``.

    Returns:
        The segments in canonical order (``attn`` before ``moe``).

    Raises:
        ValueError: An unknown name, an empty list, or a list without ``attn``: the attention
            segment drives the per-layer prefetch, so it is always enabled.
    """
    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    known = {kind.value for kind in SEGMENT_ORDER}
    unknown = sorted(set(names) - known)
    if unknown or SegmentKind.ATTN.value not in names:
        raise ValueError(
            f"{DENSE_SEGMENTS_ENV}={value!r} must list attn and optionally moe "
            f"(comma-separated); unknown: {unknown}"
        )
    return tuple(kind for kind in SEGMENT_ORDER if kind.value in names)


def dense_dwdp_requested(environ: Mapping[str, str] | None = None) -> bool:
    """Whether ``TRTLLM_DWDP_DENSE`` asks for dense DWDP; reads no other variable.

    Args:
        environ: Environment to read; ``os.environ`` if None.
    """
    return _bool_env(os.environ if environ is None else environ, DENSE_ENV, False)


@dataclass(frozen=True)
class DenseConfig:
    """Dense DWDP switches, normally read once with :meth:`from_env`.

    Attributes:
        enabled: ``TRTLLM_DWDP_DENSE``.
        segments: ``TRTLLM_DWDP_DENSE_SEGMENTS`` in canonical order; always starts with ``ATTN``.
            Strings are accepted and converted.
        required: ``TRTLLM_DWDP_DENSE_REQUIRED``.
        min_bytes: ``TRTLLM_DWDP_DENSE_MIN_BYTES``.
        store_chunk_mb: ``TRTLLM_DWDP_DENSE_STORE_CHUNK_MB``.
        strict: ``TRTLLM_DWDP_DENSE_STRICT``.
        debug: ``TRTLLM_DWDP_DENSE_DEBUG``.
    """

    enabled: bool = False
    segments: tuple[SegmentKind, ...] = SEGMENT_ORDER
    required: bool = True
    min_bytes: int = DEFAULT_MIN_BYTES
    store_chunk_mb: int = 0
    strict: bool = False
    debug: int = 0

    def __post_init__(self) -> None:
        segments = _canonical_segments(self.segments)
        object.__setattr__(self, "segments", segments)
        for name in ("min_bytes", "store_chunk_mb", "debug"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"DenseConfig.{name} must be non-negative, got {getattr(self, name)}"
                )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> DenseConfig:
        """Read every ``TRTLLM_DWDP_DENSE*`` variable.

        Args:
            environ: Environment to read; ``os.environ`` if None.

        Raises:
            ValueError: ``TRTLLM_DWDP_DENSE_SEGMENTS`` is invalid (see :func:`parse_segments`).
        """
        env = os.environ if environ is None else environ
        raw_segments = env.get(DENSE_SEGMENTS_ENV)
        if raw_segments is None or not raw_segments.strip():
            segments = SEGMENT_ORDER
        else:
            segments = parse_segments(raw_segments)
        return cls(
            enabled=_bool_env(env, DENSE_ENV, False),
            segments=segments,
            required=_bool_env(env, DENSE_REQUIRED_ENV, True),
            min_bytes=_int_env(env, DENSE_MIN_BYTES_ENV, DEFAULT_MIN_BYTES),
            store_chunk_mb=_int_env(env, DENSE_STORE_CHUNK_MB_ENV, 0),
            strict=_bool_env(env, DENSE_STRICT_ENV, False),
            debug=_int_env(env, DENSE_DEBUG_ENV, 0),
        )

    @property
    def has_moe(self) -> bool:
        """Whether the MoE-dense segment is enabled."""
        return SegmentKind.MOE in self.segments

    @property
    def store_chunk_bytes(self) -> int:
        """Store chunk limit in bytes; 0 means a single chunk."""
        return self.store_chunk_mb << 20

    @property
    def slot_fill_byte(self) -> int:
        """Byte the slots are filled with at commit: 0, or ``DEBUG_FILL_BYTE`` (NaN) if debug."""
        return DEBUG_FILL_BYTE if self.debug >= 1 else 0

    @property
    def slot_watchdog(self) -> bool:
        """Whether the per-layer slot checksum watchdog runs (``TRTLLM_DWDP_DENSE_DEBUG >= 2``)."""
        return self.debug >= 2

    def digest_switches(self, num_ranks: int) -> DigestSwitches:
        """The switches the layout depends on, for a group of ``num_ranks`` ranks."""
        return DigestSwitches(
            segments=self.segments,
            min_bytes=self.min_bytes,
            store_chunk_mb=self.store_chunk_mb,
            num_ranks=num_ranks,
        )

    def describe(self) -> str:
        """One-line summary for logs."""
        return (
            f"segments={','.join(kind.value for kind in self.segments)} "
            f"required={int(self.required)} min_bytes={self.min_bytes} "
            f"store_chunk_mb={self.store_chunk_mb} strict={int(self.strict)} debug={self.debug}"
        )


def _canonical_segments(segments: Iterable[SegmentKind | str]) -> tuple[SegmentKind, ...]:
    try:
        kinds = {SegmentKind(kind) for kind in segments}
    except ValueError as exc:
        raise ValueError(f"DenseConfig.segments has an unknown segment: {exc}") from None
    if SegmentKind.ATTN not in kinds:
        raise ValueError("DenseConfig.segments must include the attention segment")
    return tuple(kind for kind in SEGMENT_ORDER if kind in kinds)
