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

"""Pattern-spec parsers for ``--comm_pattern`` / ``--expert_pattern``."""

from __future__ import annotations

from typing import Any, Dict, Tuple

_COMM_PATTERN_NAMES: Tuple[str, ...] = (
    "random",
    "balanced_alltoall",
    "receiver_hotspot",
    "pair_hotspot",
    "local_only",
    "ring",
)

_EXPERT_PATTERN_NAMES: Tuple[str, ...] = ("random", "balanced", "hotspot")


def _parse_pattern_spec(spec: str) -> Tuple[str, Dict[str, str]]:
    """Parse ``name,k1=v1,k2=v2`` into ``(name, {k1: v1, k2: v2})``.

    File-based routing control is handled by ``--routing_pattern_file``.
    """
    raw = str(spec).strip()
    if not raw:
        raise ValueError("empty pattern spec")
    if raw.startswith("file:"):
        return "file", {"path": raw[len("file:") :]}
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"invalid pattern spec: {spec!r}")
    name = parts[0]
    kwargs: Dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"invalid pattern fragment {part!r} in {spec!r}; expected k=v")
        k, v = part.split("=", 1)
        kwargs[k.strip()] = v.strip()
    return name, kwargs


def _pop_hotness_kwarg(raw: Dict[str, str], kwargs: Dict[str, Any], *, label: str) -> None:
    """Parse the optional ``hotness=<ratio>`` shared by comm/expert patterns."""
    if "hotness" not in raw:
        return
    value = float(raw["hotness"])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} hotness must be in [0, 1]; got {value}")
    kwargs["hotness"] = value


def _parse_typed_pattern(
    spec: str, *, label: str, valid_names: Tuple[str, ...]
) -> Tuple[str, Dict[str, str]]:
    """Common prefix of ``_parse_comm_pattern`` / ``_parse_expert_pattern``.

    Parses the ``name[:k=v,...]`` form, rejects the legacy ``file:<path>``
    prefix (now handled via ``--routing_pattern_file``), and validates that
    ``name`` is one of the supported pattern names for ``label``.
    """
    name, raw = _parse_pattern_spec(spec)
    if name == "file":
        raise ValueError(f"{label} no longer accepts file:<path>; use --routing_pattern_file")
    if name not in valid_names:
        raise ValueError(f"unknown {label} {name!r}; supported: {valid_names}")
    return name, raw


def _parse_comm_pattern(spec: str) -> Tuple[str, Dict[str, Any]]:
    name, raw = _parse_typed_pattern(spec, label="comm_pattern", valid_names=_COMM_PATTERN_NAMES)
    kwargs: Dict[str, Any] = {}
    _pop_hotness_kwarg(raw, kwargs, label="comm_pattern")
    for int_key in ("rank", "src", "dst"):
        if int_key in raw:
            kwargs[int_key] = int(raw[int_key])
    if name == "receiver_hotspot":
        if "hotness" not in kwargs:
            raise ValueError("receiver_hotspot requires hotness=<ratio>")
        kwargs.setdefault("rank", 0)
    if name == "pair_hotspot":
        if "hotness" not in kwargs or "src" not in kwargs or "dst" not in kwargs:
            raise ValueError("pair_hotspot requires hotness=<ratio>, src=<src>, dst=<dst>")
    return name, kwargs


def _parse_expert_pattern(spec: str) -> Tuple[str, Dict[str, Any]]:
    name, raw = _parse_typed_pattern(
        spec, label="expert_pattern", valid_names=_EXPERT_PATTERN_NAMES
    )
    kwargs: Dict[str, Any] = {}
    _pop_hotness_kwarg(raw, kwargs, label="expert_pattern")
    if "active_experts" in raw:
        kwargs["active_experts"] = int(raw["active_experts"])
        if kwargs["active_experts"] <= 0:
            raise ValueError("expert_pattern active_experts must be > 0")
    if name == "hotspot" and "hotness" not in kwargs and "active_experts" not in kwargs:
        raise ValueError(
            "expert_pattern hotspot requires hotness=<ratio> or active_experts=<count>"
        )
    return name, kwargs
