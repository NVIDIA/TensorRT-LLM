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
"""Byte-exact per-tensor weight manifests for weight-sharing qualification.

A weight manifest records one SHA-256 digest per registered parameter and buffer
of a root `nn.Module`, together with the metadata that pins the tensor layout
(`dtype`, `shape`, `stride`, `storage_offset`, `nbytes`). Two manifests compare
equal only when every tensor is byte-for-byte identical, which is deliberately
stronger than the exact-value equality of `torch.testing.assert_close(rtol=0,
atol=0, equal_nan=True)`: the manifest distinguishes `+0.0` from `-0.0` and one
NaN payload from another.

The ModelExpress (MX) qualification harness uses two manifest *families*:

* `final` -- written once per rank at the end of `ModelLoader.load` for every
  role (HF baseline, MX donor, MX receiver), after all post-load finalization.
* `transfer` -- written inside the MX checkpoint loader at the donor publish
  point and at the receiver's full P2P success point (MX roles only).

Everything here is inert unless `MX_WEIGHT_MANIFEST_DIR` is set: production
loads never synchronize, hash, or write anything. When the directory is set the
role name must also be provided through `MX_WEIGHT_MANIFEST_ROLE`; both reach
the rank processes through `LLM(env_overrides=...)`. When the dump is active,
every failure raises -- nothing is swallowed and no partial file is left behind.

Canonical bytes of a tensor are defined as
`t.detach().reshape(-1).contiguous().cpu().view(torch.uint8)`, so the digest
covers the logical (row-major) element bytes; physical layout is compared via
the recorded metadata instead. `ModelLoader.reload` is intentionally not
covered: incremental weight updates own their own lifecycle.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import socket
import time
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from tensorrt_llm.logger import logger

WEIGHT_MANIFEST_FORMAT_VERSION = 1
WEIGHT_MANIFEST_DIR_ENV = "MX_WEIGHT_MANIFEST_DIR"
WEIGHT_MANIFEST_ROLE_ENV = "MX_WEIGHT_MANIFEST_ROLE"
WEIGHT_MANIFEST_FAMILIES = ("final", "transfer")
WEIGHT_MANIFEST_FILE_PATTERN = re.compile(
    r"^manifest\.(?P<family>final|transfer)\.(?P<role>[A-Za-z0-9_-]+)\.rank(?P<rank>\d+)\.json$"
)

PARAM_KIND = "param"
BUFFER_KIND = "buffer"
WEIGHT_MANIFEST_KINDS = (PARAM_KIND, BUFFER_KIND)
SKIP_REASON_META_DEVICE = "meta_device"
SKIP_REASON_NON_STRIDED = "non_strided_layout"

_ROLE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_METADATA_FIELDS = ("kind", "dtype", "shape", "stride", "storage_offset", "nbytes")


@dataclass(frozen=True)
class WeightManifestEntry:
    """One hashed tensor: identity, layout metadata, and its byte digest."""

    fqn: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    nbytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fqn": self.fqn,
            "kind": self.kind,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "nbytes": self.nbytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WeightManifestEntry":
        return cls(
            fqn=str(payload["fqn"]),
            kind=str(payload["kind"]),
            dtype=str(payload["dtype"]),
            shape=tuple(int(dim) for dim in payload["shape"]),
            stride=tuple(int(dim) for dim in payload["stride"]),
            storage_offset=int(payload["storage_offset"]),
            nbytes=int(payload["nbytes"]),
            sha256=str(payload["sha256"]),
        )


@dataclass(frozen=True)
class SkippedTensor:
    """A registered tensor that was recorded but not hashed, with the reason."""

    fqn: str
    kind: str
    reason: str
    dtype: str
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fqn": self.fqn,
            "kind": self.kind,
            "reason": self.reason,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkippedTensor":
        return cls(
            fqn=str(payload["fqn"]),
            kind=str(payload["kind"]),
            reason=str(payload["reason"]),
            dtype=str(payload["dtype"]),
            shape=tuple(int(dim) for dim in payload["shape"]),
        )


@dataclass(frozen=True)
class WeightManifest:
    """The full per-rank manifest.

    `entries` and `skipped` are sorted by fully-qualified name. `alias_groups`
    lists the storage-equivalence classes with at least two members (each group
    sorted by name, groups ordered by first appearance in sorted-name order).
    `manifest_sha256` covers the canonical JSON of `entries` only; `context` is
    informational and never participates in comparison.
    """

    manifest_format_version: int
    entries: tuple[WeightManifestEntry, ...]
    skipped: tuple[SkippedTensor, ...]
    alias_groups: tuple[tuple[str, ...], ...]
    manifest_sha256: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_format_version": self.manifest_format_version,
            "entries": [entry.to_dict() for entry in self.entries],
            "skipped": [skipped.to_dict() for skipped in self.skipped],
            "alias_groups": [list(group) for group in self.alias_groups],
            "manifest_sha256": self.manifest_sha256,
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WeightManifest":
        version = payload["manifest_format_version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(f"Weight manifest format version must be an int, got {version!r}")
        return cls(
            manifest_format_version=version,
            entries=tuple(WeightManifestEntry.from_dict(item) for item in payload["entries"]),
            skipped=tuple(SkippedTensor.from_dict(item) for item in payload.get("skipped", [])),
            alias_groups=tuple(
                tuple(str(name) for name in group) for group in payload.get("alias_groups", [])
            ),
            manifest_sha256=str(payload["manifest_sha256"]),
            context=dict(payload.get("context", {})),
        )

    def entries_by_fqn(
        self, kinds: Collection[str] = WEIGHT_MANIFEST_KINDS
    ) -> dict[str, WeightManifestEntry]:
        return {entry.fqn: entry for entry in self.entries if entry.kind in kinds}


@dataclass(frozen=True)
class WeightManifestWriteResult:
    """Timing and size facts about one written manifest file."""

    path: Path
    build_seconds: float
    write_seconds: float
    entry_count: int
    skipped_count: int
    bytes_hashed: int


@dataclass(frozen=True)
class WeightManifestDiff:
    """Structured difference between two manifests.

    `exempted_digest_diffs` records digest differences on tensors that matched an
    exemption pattern; they are informational and do not make the diff non-empty.
    """

    missing_in_actual: tuple[str, ...] = ()
    unexpected_in_actual: tuple[str, ...] = ()
    metadata_diffs: tuple[tuple[str, str, Any, Any], ...] = ()
    digest_diffs: tuple[str, ...] = ()
    exempted_digest_diffs: tuple[str, ...] = ()
    skipped_only_in_expected: tuple[tuple[str, str], ...] = ()
    skipped_only_in_actual: tuple[tuple[str, str], ...] = ()
    alias_groups_only_in_expected: tuple[tuple[str, ...], ...] = ()
    alias_groups_only_in_actual: tuple[tuple[str, ...], ...] = ()
    expected_context: dict[str, Any] = field(default_factory=dict)
    actual_context: dict[str, Any] = field(default_factory=dict)
    expected_entries: dict[str, WeightManifestEntry] = field(default_factory=dict, repr=False)
    actual_entries: dict[str, WeightManifestEntry] = field(default_factory=dict, repr=False)

    @property
    def is_empty(self) -> bool:
        return not (
            self.missing_in_actual
            or self.unexpected_in_actual
            or self.metadata_diffs
            or self.digest_diffs
            or self.skipped_only_in_expected
            or self.skipped_only_in_actual
            or self.alias_groups_only_in_expected
            or self.alias_groups_only_in_actual
        )

    def describe(self, expected_label: str, actual_label: str, *, limit: int = 20) -> str:
        """Render a human-readable report of the differences.

        The wording intentionally avoids the phrases the MX E2E harness treats
        as receiver failure markers so the report can be logged anywhere.
        """

        def context_summary(context: Mapping[str, Any]) -> str:
            keys = ("role", "family", "rank", "boundary")
            return ", ".join(f"{key}={context[key]!r}" for key in keys if key in context) or "n/a"

        def clipped(items: Iterable[Any]) -> tuple[list[Any], int]:
            items = list(items)
            return items[:limit], max(0, len(items) - limit)

        def entry_line(fqn: str) -> str:
            expected = self.expected_entries.get(fqn)
            actual = self.actual_entries.get(fqn)
            reference = expected or actual
            dtype = reference.dtype if reference else "?"
            shape = list(reference.shape) if reference else "?"
            expected_digest = expected.sha256[:12] if expected else "-"
            actual_digest = actual.sha256[:12] if actual else "-"
            return f"  {fqn} {dtype} {shape} expected={expected_digest} actual={actual_digest}"

        if self.is_empty:
            return f"Weight manifests are identical: {expected_label} vs {actual_label}"

        lines = [
            f"Weight manifests differ: {expected_label} vs {actual_label}",
            f"  expected context: {context_summary(self.expected_context)}",
            f"  actual context: {context_summary(self.actual_context)}",
            "  counts: "
            f"only-in-expected={len(self.missing_in_actual)}, "
            f"only-in-actual={len(self.unexpected_in_actual)}, "
            f"metadata={len(self.metadata_diffs)}, "
            f"digest={len(self.digest_diffs)}, "
            f"exempted-digest={len(self.exempted_digest_diffs)}, "
            f"skipped-only-in-expected={len(self.skipped_only_in_expected)}, "
            f"skipped-only-in-actual={len(self.skipped_only_in_actual)}, "
            f"alias-groups-only-in-expected={len(self.alias_groups_only_in_expected)}, "
            f"alias-groups-only-in-actual={len(self.alias_groups_only_in_actual)}",
        ]

        def section(title: str, items: Iterable[Any], render) -> None:
            shown, hidden = clipped(items)
            if not shown:
                return
            lines.append(f" {title}:")
            lines.extend(render(item) for item in shown)
            if hidden:
                lines.append(f"  ... and {hidden} more")

        section("tensors only in expected", self.missing_in_actual, entry_line)
        section("tensors only in actual", self.unexpected_in_actual, entry_line)
        section(
            "metadata differs",
            self.metadata_diffs,
            lambda item: f"  {item[0]} {item[1]}: expected={item[2]!r} actual={item[3]!r}",
        )
        section("digest differs", self.digest_diffs, entry_line)
        section("digest differs (exempted)", self.exempted_digest_diffs, entry_line)
        section(
            "skipped only in expected",
            self.skipped_only_in_expected,
            lambda item: f"  {item[0]} ({item[1]})",
        )
        section(
            "skipped only in actual",
            self.skipped_only_in_actual,
            lambda item: f"  {item[0]} ({item[1]})",
        )
        section(
            "alias groups only in expected",
            self.alias_groups_only_in_expected,
            lambda group: f"  {list(group)}",
        )
        section(
            "alias groups only in actual",
            self.alias_groups_only_in_actual,
            lambda group: f"  {list(group)}",
        )
        return "\n".join(lines)


def canonical_tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Return the canonical one-dimensional `uint8` view of a tensor's bytes.

    Flattening first makes the byte stream well-defined for 0-dim tensors and
    for any physical layout; `view(torch.uint8)` reinterprets without numeric
    conversion so every dtype (including `bfloat16` and FP8 variants) is
    covered.
    """
    flat = tensor.detach().reshape(-1).contiguous().cpu()
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    return flat.view(torch.uint8)


def _canonical_json_digest(obj: Any) -> str:
    """SHA-256 of the canonical JSON encoding (same recipe as `SourceIdentity`)."""
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> tuple[str, int]:
    """Return `(sha256_hex, nbytes)` over the canonical bytes of `tensor`."""
    raw = canonical_tensor_bytes(tensor)
    # `numpy()` exposes the buffer protocol without copying; `tobytes()` would
    # double the host memory for multi-GiB embedding tables.
    digest = hashlib.sha256(raw.numpy()).hexdigest()
    return digest, int(raw.numel())


def _iter_named_tensors(model: nn.Module) -> list[tuple[str, str, torch.Tensor]]:
    """Collect every registered parameter and buffer, aliases included."""
    named: list[tuple[str, str, torch.Tensor]] = []
    for fqn, tensor in model.named_parameters(remove_duplicate=False):
        if tensor is not None:
            named.append((fqn, PARAM_KIND, tensor))
    for fqn, tensor in model.named_buffers(remove_duplicate=False):
        if tensor is not None:
            named.append((fqn, BUFFER_KIND, tensor))
    named.sort(key=lambda item: (item[0], item[1]))
    return named


def build_weight_manifest(
    model: nn.Module, *, context: Optional[Mapping[str, Any]] = None
) -> WeightManifest:
    """Hash every registered tensor of `model` into a `WeightManifest`.

    Every CUDA device that holds a tensor is synchronized before any byte is
    read, and the device-to-host copies are blocking, so asynchronous loading,
    transform, or P2P writes are complete and visible. Peak extra host memory
    is one tensor's canonical bytes.
    """
    started = time.perf_counter()
    entries: list[WeightManifestEntry] = []
    skipped: list[SkippedTensor] = []
    alias_members: dict[tuple[str, int], list[str]] = {}
    bytes_hashed = 0

    with torch.no_grad():
        named = _iter_named_tensors(model)
        devices = sorted({tensor.device for _, _, tensor in named if tensor.is_cuda}, key=str)
        for device in devices:
            torch.cuda.synchronize(device)

        for fqn, kind, tensor in named:
            dtype = str(tensor.dtype)
            shape = tuple(int(dim) for dim in tensor.shape)
            if tensor.device.type == "meta":
                skipped.append(SkippedTensor(fqn, kind, SKIP_REASON_META_DEVICE, dtype, shape))
                continue
            if tensor.layout is not torch.strided:
                skipped.append(SkippedTensor(fqn, kind, SKIP_REASON_NON_STRIDED, dtype, shape))
                continue

            digest, nbytes = _tensor_digest(tensor)
            bytes_hashed += nbytes
            entries.append(
                WeightManifestEntry(
                    fqn=fqn,
                    kind=kind,
                    dtype=dtype,
                    shape=shape,
                    stride=tuple(int(dim) for dim in tensor.stride()),
                    storage_offset=int(tensor.storage_offset()),
                    nbytes=nbytes,
                    sha256=digest,
                )
            )
            storage = tensor.untyped_storage()
            if storage.nbytes() > 0:
                key = (str(tensor.device), int(storage.data_ptr()))
                alias_members.setdefault(key, []).append(fqn)

    # `dict` preserves first-seen order, which is sorted-name order here.
    alias_groups = tuple(
        tuple(sorted(members)) for members in alias_members.values() if len(members) >= 2
    )
    manifest_sha256 = _canonical_json_digest([entry.to_dict() for entry in entries])

    merged_context: dict[str, Any] = dict(context or {})
    merged_context.update(
        {
            "torch_version": torch.__version__,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_seconds": time.perf_counter() - started,
            "bytes_hashed": bytes_hashed,
            "entry_count": len(entries),
            "skipped_count": len(skipped),
        }
    )
    return WeightManifest(
        manifest_format_version=WEIGHT_MANIFEST_FORMAT_VERSION,
        entries=tuple(entries),
        skipped=tuple(skipped),
        alias_groups=alias_groups,
        manifest_sha256=manifest_sha256,
        context=merged_context,
    )


def serialize_weight_manifest(manifest: WeightManifest) -> str:
    """Return the canonical JSON text of a manifest (sorted keys, trailing newline)."""
    return json.dumps(manifest.to_dict(), sort_keys=True, indent=1, default=str) + "\n"


def write_weight_manifest(manifest: WeightManifest, path: Path) -> None:
    """Atomically write `manifest` to `path`; refuse to overwrite an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Weight manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(serialize_weight_manifest(manifest), encoding="utf-8")
    os.replace(temporary, path)


def load_weight_manifest(path: Path) -> WeightManifest:
    """Load a manifest previously written by `write_weight_manifest`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return WeightManifest.from_dict(payload)


def manifest_file_name(family: str, role: str, rank: int) -> str:
    """Return `manifest.{family}.{role}.rank{rank}.json` after validating the parts."""
    _validate_family(family)
    _validate_role(role)
    _validate_rank(rank)
    return f"manifest.{family}.{role}.rank{rank}.json"


def _validate_family(family: str) -> None:
    if family not in WEIGHT_MANIFEST_FAMILIES:
        raise ValueError(
            f"Unknown weight manifest family {family!r}; expected one of {WEIGHT_MANIFEST_FAMILIES}"
        )


def _validate_role(role: str) -> None:
    if not isinstance(role, str) or not _ROLE_PATTERN.fullmatch(role):
        raise ValueError(
            f"Invalid weight manifest role {role!r}; expected a non-empty [A-Za-z0-9_-]+ name "
            f"(set through the {WEIGHT_MANIFEST_ROLE_ENV} environment variable)"
        )


def _validate_rank(rank: Any) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"Invalid weight manifest rank {rank!r}; expected a non-negative int")


def _alias_partition(manifest: WeightManifest, fqns: Collection[str]) -> set[frozenset[str]]:
    partition: set[frozenset[str]] = set()
    for group in manifest.alias_groups:
        members = frozenset(name for name in group if name in fqns)
        if len(members) >= 2:
            partition.add(members)
    return partition


def compare_weight_manifests(
    expected: WeightManifest,
    actual: WeightManifest,
    *,
    kinds: Collection[str] = WEIGHT_MANIFEST_KINDS,
    exempt_patterns: Collection[str] = (),
) -> WeightManifestDiff:
    """Compare two manifests byte-for-byte on the selected tensor kinds.

    Manifests with different `manifest_format_version` never compare (raises
    `ValueError`). `exempt_patterns` are `fnmatch` patterns on fully-qualified
    names whose *digest* differences are reported separately and do not make the
    diff non-empty; metadata differences are never exempted.
    """
    if expected.manifest_format_version != actual.manifest_format_version:
        raise ValueError(
            "Weight manifests with different format versions never compare: "
            f"{expected.manifest_format_version} vs {actual.manifest_format_version}"
        )
    kinds = tuple(kinds)
    for kind in kinds:
        if kind not in WEIGHT_MANIFEST_KINDS:
            raise ValueError(
                f"Unknown tensor kind {kind!r}; expected one of {WEIGHT_MANIFEST_KINDS}"
            )
    patterns = tuple(exempt_patterns)

    expected_entries = expected.entries_by_fqn(kinds)
    actual_entries = actual.entries_by_fqn(kinds)
    expected_skipped = {(item.fqn, item.reason) for item in expected.skipped if item.kind in kinds}
    actual_skipped = {(item.fqn, item.reason) for item in actual.skipped if item.kind in kinds}
    expected_alias = _alias_partition(expected, expected_entries.keys())
    actual_alias = _alias_partition(actual, actual_entries.keys())

    base = dict(
        expected_context=dict(expected.context),
        actual_context=dict(actual.context),
        expected_entries=expected_entries,
        actual_entries=actual_entries,
    )

    full_scope = set(kinds) == set(WEIGHT_MANIFEST_KINDS)
    if (
        full_scope
        and expected.manifest_sha256 == actual.manifest_sha256
        and expected_skipped == actual_skipped
        and expected_alias == actual_alias
    ):
        return WeightManifestDiff(**base)

    expected_names = set(expected_entries)
    actual_names = set(actual_entries)
    metadata_diffs: list[tuple[str, str, Any, Any]] = []
    digest_diffs: list[str] = []
    exempted_digest_diffs: list[str] = []
    for fqn in sorted(expected_names & actual_names):
        left = expected_entries[fqn]
        right = actual_entries[fqn]
        for field_name in _METADATA_FIELDS:
            left_value = getattr(left, field_name)
            right_value = getattr(right, field_name)
            if left_value != right_value:
                metadata_diffs.append((fqn, field_name, left_value, right_value))
        if left.sha256 != right.sha256:
            if any(fnmatch.fnmatchcase(fqn, pattern) for pattern in patterns):
                exempted_digest_diffs.append(fqn)
            else:
                digest_diffs.append(fqn)

    return WeightManifestDiff(
        missing_in_actual=tuple(sorted(expected_names - actual_names)),
        unexpected_in_actual=tuple(sorted(actual_names - expected_names)),
        metadata_diffs=tuple(metadata_diffs),
        digest_diffs=tuple(digest_diffs),
        exempted_digest_diffs=tuple(exempted_digest_diffs),
        skipped_only_in_expected=tuple(sorted(expected_skipped - actual_skipped)),
        skipped_only_in_actual=tuple(sorted(actual_skipped - expected_skipped)),
        alias_groups_only_in_expected=tuple(
            sorted(tuple(sorted(group)) for group in expected_alias - actual_alias)
        ),
        alias_groups_only_in_actual=tuple(
            sorted(tuple(sorted(group)) for group in actual_alias - expected_alias)
        ),
        **base,
    )


def maybe_write_weight_manifest(
    model: nn.Module,
    *,
    family: str,
    rank: int,
    context: Optional[Mapping[str, Any]] = None,
) -> Optional[WeightManifestWriteResult]:
    """Write a manifest for this rank when `MX_WEIGHT_MANIFEST_DIR` is set.

    Returns `None` without touching the model when the directory variable is
    unset or empty. When it is set, the role comes from
    `MX_WEIGHT_MANIFEST_ROLE`, the target file is
    `manifest.{family}.{role}.rank{rank}.json`, and any problem (invalid
    family, role, or rank; an already existing target; hashing or I/O errors)
    raises so a misconfigured qualification run fails loudly.
    """
    directory = os.environ.get(WEIGHT_MANIFEST_DIR_ENV)
    if not directory:
        return None

    role = os.environ.get(WEIGHT_MANIFEST_ROLE_ENV, "")
    file_name = manifest_file_name(family, role, rank)
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / file_name
    if target.exists():
        raise FileExistsError(
            f"Weight manifest already exists: {target}. A rank process is expected to load "
            "a model once per role; remove stale files or use a fresh directory."
        )

    merged_context: dict[str, Any] = dict(context or {})
    merged_context.update({"role": role, "family": family, "rank": rank})
    manifest = build_weight_manifest(model, context=merged_context)

    write_started = time.perf_counter()
    write_weight_manifest(manifest, target)
    write_seconds = time.perf_counter() - write_started

    result = WeightManifestWriteResult(
        path=target,
        build_seconds=float(manifest.context["build_seconds"]),
        write_seconds=write_seconds,
        entry_count=len(manifest.entries),
        skipped_count=len(manifest.skipped),
        bytes_hashed=int(manifest.context["bytes_hashed"]),
    )
    logger.info(
        f"Wrote {family} weight manifest {target} ({result.entry_count} tensors, "
        f"{result.skipped_count} skipped, {len(manifest.alias_groups)} alias groups, "
        f"{result.bytes_hashed / (1 << 20):.1f} MiB hashed) in "
        f"{result.build_seconds + result.write_seconds:.3f} s"
    )
    return result


__all__ = [
    "BUFFER_KIND",
    "PARAM_KIND",
    "SKIP_REASON_META_DEVICE",
    "SKIP_REASON_NON_STRIDED",
    "WEIGHT_MANIFEST_DIR_ENV",
    "WEIGHT_MANIFEST_FAMILIES",
    "WEIGHT_MANIFEST_FILE_PATTERN",
    "WEIGHT_MANIFEST_FORMAT_VERSION",
    "WEIGHT_MANIFEST_KINDS",
    "WEIGHT_MANIFEST_ROLE_ENV",
    "SkippedTensor",
    "WeightManifest",
    "WeightManifestDiff",
    "WeightManifestEntry",
    "WeightManifestWriteResult",
    "build_weight_manifest",
    "canonical_tensor_bytes",
    "compare_weight_manifests",
    "load_weight_manifest",
    "manifest_file_name",
    "maybe_write_weight_manifest",
    "serialize_weight_manifest",
    "write_weight_manifest",
]
