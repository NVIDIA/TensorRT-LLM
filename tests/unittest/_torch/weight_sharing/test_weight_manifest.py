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
"""Unit tests for the byte-exact weight manifest (`weight_manifest.py`).

The corruption-injection tests pin the contract the ModelExpress qualification
harness relies on: a single flipped bit fails the comparison and names exactly
the affected tensor, and the manifest is strictly stronger than
`torch.testing.assert_close(rtol=0, atol=0, equal_nan=True)` because it
distinguishes signed zeros and NaN payloads.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.weight_sharing import (
    WEIGHT_MANIFEST_DIR_ENV,
    WEIGHT_MANIFEST_FILE_PATTERN,
    WEIGHT_MANIFEST_FORMAT_VERSION,
    WEIGHT_MANIFEST_ROLE_ENV,
    SkippedTensor,
    build_weight_manifest,
    canonical_tensor_bytes,
    compare_weight_manifests,
    load_weight_manifest,
    manifest_file_name,
    maybe_write_weight_manifest,
    serialize_weight_manifest,
    write_weight_manifest,
)
from tensorrt_llm._torch.weight_sharing import weight_manifest as weight_manifest_mod
from tensorrt_llm._torch.weight_sharing.source_identity import _canonical_hash

# CPU tests carry `pytest.mark.cpu_only` individually so the CUDA test below stays
# eligible for GPU stages (which run with `-m "not cpu_only"`).

# Phrases the MX E2E harness treats as receiver failure markers; manifest log
# lines and reports must never contain them.
_FAILURE_MARKERS = (
    "falling back to disk",
    "partial fallback",
    "size mismatch",
    "still missing",
    "mx p2p transfer failed",
)


class _TwoTensorModule(nn.Module):
    """A parameter plus a buffer with deterministic values."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        values = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
        self.weight = nn.Parameter(values.to(dtype))
        self.register_buffer("scale", torch.tensor([0.5, 0.25], dtype=torch.float32))


class _NamedParamsModule(nn.Module):
    """Parameters registered in the given order, each filled with its index."""

    def __init__(self, names: list[str], offset: float = 0.0) -> None:
        super().__init__()
        for index, name in enumerate(names):
            setattr(self, name, nn.Parameter(torch.full((2,), float(index) + offset)))


class _TiedModule(nn.Module):
    """Two linears with identical values; optionally tied plus an aliasing buffer."""

    def __init__(self, *, tied: bool) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4, bias=False)
        self.b = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.a.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))
            self.b.weight.copy_(self.a.weight)
        if tied:
            self.b.weight = self.a.weight
            self.register_buffer("view_of_a", self.a.weight.detach()[0])
        else:
            self.register_buffer("view_of_a", self.a.weight.detach()[0].clone())


def _flip_bit(tensor: torch.Tensor, byte_index: int, bit: int) -> None:
    """Flip one bit of a contiguous tensor in place through its uint8 view."""
    raw = tensor.detach().reshape(-1).view(torch.uint8)
    raw[byte_index] = int(raw[byte_index]) ^ (1 << bit)


def _fqns(manifest, kind: str | None = None) -> list[str]:
    return [entry.fqn for entry in manifest.entries if kind is None or entry.kind == kind]


def _digest_of(manifest, fqn: str) -> str:
    return next(entry.sha256 for entry in manifest.entries if entry.fqn == fqn)


def _assert_close_passes(left: torch.Tensor, right: torch.Tensor) -> None:
    torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)


# --------------------------------------------------------------------------- #
# Canonical bytes and digests
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_canonical_tensor_bytes_matches_design_formula():
    tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).t()
    assert not tensor.is_contiguous()

    raw = canonical_tensor_bytes(tensor)

    assert raw.dtype == torch.uint8
    assert raw.dim() == 1
    assert raw.numel() == tensor.numel() * tensor.element_size()
    expected = tensor.detach().reshape(-1).contiguous().cpu().view(torch.uint8)
    assert torch.equal(raw, expected)

    scalar = torch.tensor(1.5, dtype=torch.float32)
    assert canonical_tensor_bytes(scalar).numel() == 4
    assert canonical_tensor_bytes(torch.empty(0, dtype=torch.bfloat16)).numel() == 0


@pytest.mark.cpu_only
def test_canonical_json_digest_matches_source_identity_hash():
    sample = {"b": [1, 2, {"z": torch.bfloat16}], "a": ("x", None, 3.5)}
    assert weight_manifest_mod._canonical_json_digest(sample) == _canonical_hash(sample)


@pytest.mark.cpu_only
def test_entries_sorted_by_fqn_regardless_of_registration_order():
    manifest = build_weight_manifest(_NamedParamsModule(["zeta", "alpha", "mid"]))
    assert _fqns(manifest) == ["alpha", "mid", "zeta"]
    assert all(entry.kind == "param" for entry in manifest.entries)


@pytest.mark.cpu_only
def test_manifest_is_deterministic_and_context_does_not_affect_digest():
    module = _TwoTensorModule()
    first = build_weight_manifest(module, context={"boundary": "first"})
    second = build_weight_manifest(module, context={"boundary": "second", "extra": 1})

    assert first.manifest_format_version == WEIGHT_MANIFEST_FORMAT_VERSION
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.entries == second.entries
    assert first.skipped == second.skipped
    assert first.alias_groups == second.alias_groups
    assert first.manifest_sha256 == weight_manifest_mod._canonical_json_digest(
        [entry.to_dict() for entry in first.entries]
    )
    assert first.context["boundary"] == "first"
    assert second.context["extra"] == 1
    for key in ("torch_version", "hostname", "pid", "created_at", "build_seconds", "bytes_hashed"):
        assert key in first.context
    assert first.context["entry_count"] == 2
    assert first.context["skipped_count"] == 0
    # 8 bf16 elements + 2 fp32 elements.
    assert first.context["bytes_hashed"] == 8 * 2 + 2 * 4


@pytest.mark.cpu_only
def test_entry_metadata_records_layout_and_kinds():
    manifest = build_weight_manifest(_TwoTensorModule())
    by_fqn = manifest.entries_by_fqn()
    weight = by_fqn["weight"]
    scale = by_fqn["scale"]

    assert weight.kind == "param"
    assert weight.dtype == "torch.bfloat16"
    assert weight.shape == (2, 4)
    assert weight.stride == (4, 1)
    assert weight.storage_offset == 0
    assert weight.nbytes == 16
    assert scale.kind == "buffer"
    assert scale.dtype == "torch.float32"
    assert scale.nbytes == 8
    assert manifest.entries_by_fqn(("buffer",)).keys() == {"scale"}


# --------------------------------------------------------------------------- #
# Corruption injection: the contract the E2E harness relies on
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_single_bit_flip_fails_manifest_and_names_only_that_fqn():
    module = _TwoTensorModule()
    reference = module.weight.detach().clone()
    before = build_weight_manifest(module)

    # Element 3 (0.375) lives in bytes 6-7; bit 0 of byte 6 is its lowest mantissa bit.
    _flip_bit(module.weight, byte_index=6, bit=0)
    after = build_weight_manifest(module)

    with pytest.raises(AssertionError):
        _assert_close_passes(module.weight.detach(), reference)

    diff = compare_weight_manifests(before, after)
    assert not diff.is_empty
    assert diff.digest_diffs == ("weight",)
    assert diff.metadata_diffs == ()
    assert diff.missing_in_actual == () and diff.unexpected_in_actual == ()
    assert diff.exempted_digest_diffs == ()
    assert _digest_of(before, "scale") == _digest_of(after, "scale")


@pytest.mark.cpu_only
def test_signed_zero_flip_fails_manifest_but_passes_assert_close():
    module = _TwoTensorModule()
    assert module.weight[0, 0].item() == 0.0
    reference = module.weight.detach().clone()
    before = build_weight_manifest(module)

    # Byte 1 is the high byte of element 0; bit 7 is the sign bit: +0.0 -> -0.0.
    _flip_bit(module.weight, byte_index=1, bit=7)
    assert module.weight[0, 0].item() == 0.0
    assert torch.signbit(module.weight[0, 0]).item()
    after = build_weight_manifest(module)

    _assert_close_passes(module.weight.detach(), reference)
    diff = compare_weight_manifests(before, after)
    assert diff.digest_diffs == ("weight",)
    assert not diff.is_empty


@pytest.mark.cpu_only
def test_nan_payload_change_fails_manifest_but_passes_assert_close():
    module = _TwoTensorModule()
    with torch.no_grad():
        module.weight[0, 0] = float("nan")
    reference = module.weight.detach().clone()
    before = build_weight_manifest(module)

    # Flip the lowest mantissa bit of element 0: still NaN, different payload.
    _flip_bit(module.weight, byte_index=0, bit=0)
    assert torch.isnan(module.weight[0, 0]).item()
    after = build_weight_manifest(module)

    _assert_close_passes(module.weight.detach(), reference)
    diff = compare_weight_manifests(before, after)
    assert diff.digest_diffs == ("weight",)
    assert not diff.is_empty


# --------------------------------------------------------------------------- #
# Layout, aliases, skipped tensors
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_non_contiguous_tensor_hashes_logical_bytes_and_records_stride():
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    class _Holder(nn.Module):
        def __init__(self, tensor: torch.Tensor) -> None:
            super().__init__()
            self.weight = nn.Parameter(tensor)

    strided = _Holder(base.t())
    assert not strided.weight.is_contiguous()
    dense = _Holder(base.t().contiguous())

    strided_manifest = build_weight_manifest(strided)
    dense_manifest = build_weight_manifest(dense)

    assert _digest_of(strided_manifest, "weight") == _digest_of(dense_manifest, "weight")
    assert strided_manifest.entries_by_fqn()["weight"].stride == (1, 4)
    assert dense_manifest.entries_by_fqn()["weight"].stride == (3, 1)

    diff = compare_weight_manifests(strided_manifest, dense_manifest)
    assert diff.digest_diffs == ()
    assert diff.metadata_diffs == (("weight", "stride", (1, 4), (3, 1)),)
    assert not diff.is_empty


@pytest.mark.cpu_only
def test_alias_groups_partition_tied_and_view_tensors():
    tied = build_weight_manifest(_TiedModule(tied=True))
    untied = build_weight_manifest(_TiedModule(tied=False))

    assert _fqns(tied) == ["a.weight", "b.weight", "view_of_a"]
    assert tied.alias_groups == (("a.weight", "b.weight", "view_of_a"),)
    assert untied.alias_groups == ()
    # Same bytes everywhere: only the alias structure differs.
    assert tied.manifest_sha256 == untied.manifest_sha256

    diff = compare_weight_manifests(tied, untied)
    assert diff.digest_diffs == () and diff.metadata_diffs == ()
    assert diff.alias_groups_only_in_expected == (("a.weight", "b.weight", "view_of_a"),)
    assert diff.alias_groups_only_in_actual == ()
    assert not diff.is_empty


@pytest.mark.cpu_only
def test_empty_storage_tensors_do_not_form_alias_groups():
    module = _NamedParamsModule([])
    module.first = nn.Parameter(torch.empty(0))
    module.second = nn.Parameter(torch.empty(0))

    manifest = build_weight_manifest(module)

    assert manifest.alias_groups == ()
    assert [entry.nbytes for entry in manifest.entries] == [0, 0]
    assert all(entry.sha256 == hashlib.sha256(b"").hexdigest() for entry in manifest.entries)


@pytest.mark.cpu_only
def test_meta_tensors_are_skipped_with_reason():
    class _PartlyMeta(nn.Module):
        def __init__(self, materialized: bool) -> None:
            super().__init__()
            device = "cpu" if materialized else "meta"
            self.weight = nn.Parameter(torch.zeros(2, 2, device=device))
            self.register_buffer("scale", torch.ones(2))

    meta_manifest = build_weight_manifest(_PartlyMeta(materialized=False))
    real_manifest = build_weight_manifest(_PartlyMeta(materialized=True))

    assert _fqns(meta_manifest) == ["scale"]
    assert meta_manifest.skipped == (
        SkippedTensor("weight", "param", "meta_device", "torch.float32", (2, 2)),
    )
    assert meta_manifest.context["skipped_count"] == 1

    diff = compare_weight_manifests(meta_manifest, real_manifest)
    assert diff.unexpected_in_actual == ("weight",)
    assert diff.skipped_only_in_expected == (("weight", "meta_device"),)
    assert diff.skipped_only_in_actual == ()
    assert not diff.is_empty


# --------------------------------------------------------------------------- #
# Comparison semantics
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_format_version_differences_never_compare():
    manifest = build_weight_manifest(_TwoTensorModule())
    other = dataclasses.replace(
        manifest, manifest_format_version=WEIGHT_MANIFEST_FORMAT_VERSION + 1
    )
    with pytest.raises(ValueError, match="format versions"):
        compare_weight_manifests(manifest, other)


@pytest.mark.cpu_only
def test_identical_manifests_compare_empty_and_describe_says_so():
    module = _TwoTensorModule()
    diff = compare_weight_manifests(build_weight_manifest(module), build_weight_manifest(module))
    assert diff.is_empty
    assert "identical" in diff.describe("baseline rank0", "receiver rank0")


@pytest.mark.cpu_only
def test_compare_exemptions_apply_only_to_digests():
    expected = build_weight_manifest(_TwoTensorModule())
    shifted_module = _TwoTensorModule()
    with torch.no_grad():
        shifted_module.weight.add_(1)
    shifted = build_weight_manifest(shifted_module)

    strict = compare_weight_manifests(expected, shifted)
    assert strict.digest_diffs == ("weight",)
    assert not strict.is_empty

    exempted = compare_weight_manifests(expected, shifted, exempt_patterns=("wei*",))
    assert exempted.is_empty
    assert exempted.digest_diffs == ()
    assert exempted.exempted_digest_diffs == ("weight",)

    class _Reshaped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, 2, dtype=torch.bfloat16))
            self.register_buffer("scale", torch.tensor([0.5, 0.25], dtype=torch.float32))

    reshaped = compare_weight_manifests(
        expected, build_weight_manifest(_Reshaped()), exempt_patterns=("weight",)
    )
    assert ("weight", "shape", (2, 4), (4, 2)) in reshaped.metadata_diffs
    assert reshaped.exempted_digest_diffs == ("weight",)
    assert not reshaped.is_empty


@pytest.mark.cpu_only
def test_compare_kinds_filter_ignores_other_kind():
    expected = build_weight_manifest(_TwoTensorModule())
    module = _TwoTensorModule()
    with torch.no_grad():
        module.scale.mul_(2)
    actual = build_weight_manifest(module)

    assert compare_weight_manifests(expected, actual).digest_diffs == ("scale",)
    assert compare_weight_manifests(expected, actual, kinds=("param",)).is_empty
    with pytest.raises(ValueError, match="Unknown tensor kind"):
        compare_weight_manifests(expected, actual, kinds=("weights",))


@pytest.mark.cpu_only
def test_describe_lists_first_differences_and_avoids_failure_markers():
    names = [f"p{index}" for index in range(5)]
    expected = build_weight_manifest(
        _NamedParamsModule(names), context={"role": "baseline", "family": "final", "rank": 0}
    )
    actual = build_weight_manifest(
        _NamedParamsModule(names, offset=1.0),
        context={"role": "receiver", "family": "final", "rank": 0, "boundary": "end"},
    )
    diff = compare_weight_manifests(expected, actual)
    assert diff.digest_diffs == tuple(names)

    report = diff.describe("baseline rank0", "receiver rank0", limit=2)

    assert "Weight manifests differ: baseline rank0 vs receiver rank0" in report
    assert "role='baseline'" in report and "role='receiver'" in report
    assert "boundary='end'" in report
    assert "digest=5" in report
    # Entry lines render " expected=<digest>"; the counts line uses "-expected=".
    assert report.count(" expected=") == 2
    assert "... and 3 more" in report
    assert "torch.float32 [2]" in report
    lowered = report.lower()
    assert not any(marker in lowered for marker in _FAILURE_MARKERS)


# --------------------------------------------------------------------------- #
# Serialization and file naming
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_manifest_file_name_validates_parts():
    assert manifest_file_name("final", "baseline", 0) == "manifest.final.baseline.rank0.json"
    assert manifest_file_name("transfer", "receiver", 7) == "manifest.transfer.receiver.rank7.json"
    with pytest.raises(ValueError, match="family"):
        manifest_file_name("interim", "baseline", 0)
    with pytest.raises(ValueError, match="role"):
        manifest_file_name("final", "bad role", 0)
    with pytest.raises(ValueError, match="role"):
        manifest_file_name("final", "", 0)
    with pytest.raises(ValueError, match="rank"):
        manifest_file_name("final", "baseline", -1)
    with pytest.raises(ValueError, match="rank"):
        manifest_file_name("final", "baseline", True)


@pytest.mark.cpu_only
def test_write_load_roundtrip_is_atomic_and_refuses_overwrite(tmp_path: Path):
    manifest = build_weight_manifest(_TiedModule(tied=True), context={"boundary": "unit"})
    path = tmp_path / manifest_file_name("final", "baseline", 0)

    write_weight_manifest(manifest, path)

    assert load_weight_manifest(path) == manifest
    assert not list(tmp_path.glob("*.tmp"))
    match = WEIGHT_MANIFEST_FILE_PATTERN.match(path.name)
    assert match is not None
    assert match.group("family") == "final"
    assert match.group("role") == "baseline"
    assert match.group("rank") == "0"

    text = path.read_text(encoding="utf-8")
    assert text == serialize_weight_manifest(manifest)
    assert text.endswith("\n")
    payload = json.loads(text)
    assert list(payload) == sorted(payload)
    assert payload["alias_groups"] == [["a.weight", "b.weight", "view_of_a"]]

    with pytest.raises(FileExistsError):
        write_weight_manifest(manifest, path)


# --------------------------------------------------------------------------- #
# Env-gated writer
# --------------------------------------------------------------------------- #


@pytest.mark.cpu_only
def test_maybe_write_is_noop_without_dir(monkeypatch, tmp_path: Path):
    monkeypatch.delenv(WEIGHT_MANIFEST_DIR_ENV, raising=False)
    monkeypatch.setenv(WEIGHT_MANIFEST_ROLE_ENV, "baseline")

    assert maybe_write_weight_manifest(_TwoTensorModule(), family="final", rank=0) is None
    monkeypatch.setenv(WEIGHT_MANIFEST_DIR_ENV, "")
    assert maybe_write_weight_manifest(_TwoTensorModule(), family="final", rank=0) is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.cpu_only
def test_maybe_write_requires_valid_role(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(WEIGHT_MANIFEST_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(WEIGHT_MANIFEST_ROLE_ENV, raising=False)
    with pytest.raises(ValueError, match="role"):
        maybe_write_weight_manifest(_TwoTensorModule(), family="final", rank=0)

    monkeypatch.setenv(WEIGHT_MANIFEST_ROLE_ENV, "bad role")
    with pytest.raises(ValueError, match="role"):
        maybe_write_weight_manifest(_TwoTensorModule(), family="final", rank=0)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.cpu_only
def test_maybe_write_rejects_bad_family_or_rank(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(WEIGHT_MANIFEST_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(WEIGHT_MANIFEST_ROLE_ENV, "donor")
    with pytest.raises(ValueError, match="family"):
        maybe_write_weight_manifest(_TwoTensorModule(), family="interim", rank=0)
    with pytest.raises(ValueError, match="rank"):
        maybe_write_weight_manifest(_TwoTensorModule(), family="final", rank=-1)
    with pytest.raises(ValueError, match="rank"):
        maybe_write_weight_manifest(_TwoTensorModule(), family="final", rank=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.cpu_only
def test_maybe_write_writes_expected_path_and_context(monkeypatch, tmp_path: Path):
    target_dir = tmp_path / "manifests"
    monkeypatch.setenv(WEIGHT_MANIFEST_DIR_ENV, str(target_dir))
    monkeypatch.setenv(WEIGHT_MANIFEST_ROLE_ENV, "receiver")
    messages: list[str] = []

    class _Logger:
        def info(self, message: str) -> None:
            messages.append(message)

    monkeypatch.setattr(weight_manifest_mod, "logger", _Logger())

    result = maybe_write_weight_manifest(
        _TwoTensorModule(),
        family="transfer",
        rank=1,
        context={"boundary": "receiver_p2p_success", "checkpoint_format": "MX"},
    )

    assert result is not None
    assert result.path == target_dir / "manifest.transfer.receiver.rank1.json"
    assert result.entry_count == 2
    assert result.skipped_count == 0
    assert result.bytes_hashed == 8 * 2 + 2 * 4
    assert result.build_seconds >= 0 and result.write_seconds >= 0

    manifest = load_weight_manifest(result.path)
    assert manifest.context["role"] == "receiver"
    assert manifest.context["family"] == "transfer"
    assert manifest.context["rank"] == 1
    assert manifest.context["boundary"] == "receiver_p2p_success"
    assert manifest.context["checkpoint_format"] == "MX"
    assert manifest.context["entry_count"] == 2

    assert len(messages) == 1
    assert "Wrote transfer weight manifest" in messages[0]
    assert not any(marker in messages[0].lower() for marker in _FAILURE_MARKERS)

    with pytest.raises(FileExistsError):
        maybe_write_weight_manifest(_TwoTensorModule(), family="transfer", rank=1)


@pytest.mark.cpu_only
def test_load_rejects_stale_manifest_digest(tmp_path: Path):
    manifest = build_weight_manifest(_TwoTensorModule())
    path = tmp_path / manifest_file_name("final", "baseline", 0)
    write_weight_manifest(manifest, path)

    # Tamper with an entry but keep the stored whole-manifest digest.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_sha256"):
        load_weight_manifest(path)


@pytest.mark.cpu_only
def test_compare_does_not_trust_stale_manifest_digest():
    manifest = build_weight_manifest(_TwoTensorModule())
    tampered_entries = tuple(
        dataclasses.replace(entry, sha256="0" * 64) if entry.fqn == "weight" else entry
        for entry in manifest.entries
    )
    # Same stored `manifest_sha256`, different entries: the fast path must not apply.
    stale = dataclasses.replace(manifest, entries=tampered_entries)

    diff = compare_weight_manifests(manifest, stale)

    assert diff.digest_diffs == ("weight",)
    assert not diff.is_empty


# --------------------------------------------------------------------------- #
# CUDA synchronization (skipped without a GPU)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_tensors_are_synchronized_before_hashing():
    module = _TwoTensorModule(dtype=torch.float32).cuda()
    expected_weight = (torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8) * 3
    expected_digest = hashlib.sha256(canonical_tensor_bytes(expected_weight).numpy()).hexdigest()

    with torch.no_grad():
        module.weight.mul_(3)  # enqueued asynchronously; the manifest must observe it
    manifest = build_weight_manifest(module)

    assert _digest_of(manifest, "weight") == expected_digest
    assert manifest.entries_by_fqn()["weight"].dtype == "torch.float32"
    assert manifest.context["bytes_hashed"] == 8 * 4 + 2 * 4
