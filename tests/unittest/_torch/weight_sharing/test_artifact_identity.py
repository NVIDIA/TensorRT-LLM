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
"""Tests for immutable checkpoint artifact identities."""

from pathlib import Path

import pytest

from tensorrt_llm._torch.weight_sharing import ArtifactIdentity


def _write_checkpoint(path: Path, weights: bytes = b"weights") -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"architectures":["LlamaForCausalLM"]}')
    (path / "model.safetensors").write_bytes(weights)


def test_local_checkpoint_identity_is_path_independent(tmp_path: Path) -> None:
    left = tmp_path / "left" / "checkpoint"
    right = tmp_path / "right" / "checkpoint"
    _write_checkpoint(left)
    _write_checkpoint(right)

    assert ArtifactIdentity.from_checkpoint(left) == ArtifactIdentity.from_checkpoint(right)


def test_local_checkpoint_identity_binds_file_contents(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_checkpoint(left, weights=b"fine-tune-a")
    _write_checkpoint(right, weights=b"fine-tune-b")

    assert ArtifactIdentity.from_checkpoint(left) != ArtifactIdentity.from_checkpoint(right)


def test_local_checkpoint_identity_ignores_cache_and_scm_metadata(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_checkpoint(left)
    _write_checkpoint(right)
    (left / ".cache").mkdir()
    (left / ".cache" / "download.lock").write_text("transient")
    (right / ".git").mkdir()
    (right / ".git" / "HEAD").write_text("ref: refs/heads/main")

    assert ArtifactIdentity.from_checkpoint(left) == ArtifactIdentity.from_checkpoint(right)


def test_local_checkpoint_identity_rejects_nested_directory_symlink(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    external_weights = tmp_path / "external-weights"
    external_weights.mkdir()
    (external_weights / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "weights").symlink_to(external_weights, target_is_directory=True)

    with pytest.raises(ValueError, match="nested symlinked directories"):
        ArtifactIdentity.from_checkpoint(checkpoint)


def test_hf_snapshot_identity_binds_revision_across_cache_roots(tmp_path: Path) -> None:
    revision = "a" * 40
    left = tmp_path / "cache-a" / "models--org--model" / "snapshots" / revision
    right = tmp_path / "cache-b" / "models--org--model" / "snapshots" / revision
    left.mkdir(parents=True)
    right.mkdir(parents=True)

    left_identity = ArtifactIdentity.from_checkpoint(left)
    right_identity = ArtifactIdentity.from_checkpoint(right)
    assert left_identity == right_identity
    assert left_identity.scheme == "hf_snapshot_revision"


def test_hf_snapshot_identity_binds_revision_and_subpath(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--org--model" / "snapshots"
    revision_a = snapshot / ("a" * 40)
    revision_b = snapshot / ("b" * 40)
    (revision_a / "variant-a").mkdir(parents=True)
    (revision_a / "variant-b").mkdir()
    revision_b.mkdir(parents=True)

    root_identity = ArtifactIdentity.from_checkpoint(revision_a)
    assert root_identity != ArtifactIdentity.from_checkpoint(revision_b)
    assert root_identity != ArtifactIdentity.from_checkpoint(revision_a / "variant-a")
    assert ArtifactIdentity.from_checkpoint(
        revision_a / "variant-a"
    ) != ArtifactIdentity.from_checkpoint(revision_a / "variant-b")


def test_serialization_roundtrip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    identity = ArtifactIdentity.from_checkpoint(checkpoint)

    assert ArtifactIdentity.from_dict(identity.to_dict()) == identity


def test_rejects_unknown_format_version(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    payload = ArtifactIdentity.from_checkpoint(checkpoint).to_dict()
    payload["format_version"] += 1

    with pytest.raises(ValueError, match="Unsupported ArtifactIdentity format version"):
        ArtifactIdentity.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scheme", "unknown", "Unsupported ArtifactIdentity scheme"),
        ("scheme", [], "scheme must be a string"),
        ("digest", "not-a-digest", "64-character hex value"),
        ("digest", 1, "digest must be a string"),
    ],
)
def test_rejects_invalid_serialized_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    payload = ArtifactIdentity.from_checkpoint(checkpoint).to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        ArtifactIdentity.from_dict(payload)


@pytest.mark.parametrize("version", [True, "1"])
def test_rejects_non_integer_format_version(tmp_path: Path, version: object) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    payload = ArtifactIdentity.from_checkpoint(checkpoint).to_dict()
    payload["format_version"] = version

    with pytest.raises(ValueError, match="format version must be an integer"):
        ArtifactIdentity.from_dict(payload)


def test_rejects_missing_or_empty_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ArtifactIdentity.from_checkpoint(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="contains no files"):
        ArtifactIdentity.from_checkpoint(empty)
