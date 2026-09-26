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
"""Tests for how Cosmos3 fetches and prepares its guardrail checkpoint.

No GPU, no network: ``snapshot_download`` is replaced with a recorder, and the
symlink handling runs against a synthetic HF-cache-shaped tree under tmp_path.
"""

import os
import pathlib

import huggingface_hub
import pytest
from huggingface_hub.errors import GatedRepoError

from tensorrt_llm._torch.visual_gen.models.cosmos3 import guardrails
from tensorrt_llm._torch.visual_gen.models.cosmos3.guardrails import (
    GUARDRAIL_ALLOW_PATTERNS,
    GUARDRAIL_HF_REPO,
    GUARDRAIL_REVISION,
    _materialize_nltk_data,
    download_guardrail_checkpoint,
)

pytestmark = [pytest.mark.cpu_only, pytest.mark.cosmos3]


class _RecordingSnapshotDownload:
    """Stand-in for huggingface_hub.snapshot_download that scripts its outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _pinned(call):
    return (call["repo_id"], call["revision"], call["allow_patterns"])


def _install(monkeypatch, fake):
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return fake


class TestDownloadGuardrailCheckpoint:
    def test_cache_hit_makes_one_offline_call(self, monkeypatch, tmp_path):
        fake = _install(monkeypatch, _RecordingSnapshotDownload([str(tmp_path)]))

        assert download_guardrail_checkpoint() == str(tmp_path)

        assert len(fake.calls) == 1
        assert fake.calls[0]["local_files_only"] is True
        assert _pinned(fake.calls[0]) == (
            GUARDRAIL_HF_REPO,
            GUARDRAIL_REVISION,
            GUARDRAIL_ALLOW_PATTERNS,
        )

    def test_cache_miss_downloads_with_the_same_mask_and_pin(self, monkeypatch, tmp_path):
        fake = _install(
            monkeypatch, _RecordingSnapshotDownload([FileNotFoundError(), str(tmp_path)])
        )

        assert download_guardrail_checkpoint() == str(tmp_path)

        assert len(fake.calls) == 2
        assert fake.calls[0]["local_files_only"] is True
        assert "local_files_only" not in fake.calls[1]
        # Dropping the mask on either path restores the 17 GB full-repo fetch.
        for call in fake.calls:
            assert _pinned(call) == (
                GUARDRAIL_HF_REPO,
                GUARDRAIL_REVISION,
                GUARDRAIL_ALLOW_PATTERNS,
            )

    def test_gated_repo_is_reported_as_value_error(self, monkeypatch):
        _install(
            monkeypatch, _RecordingSnapshotDownload([FileNotFoundError(), GatedRepoError("gated")])
        )

        with pytest.raises(ValueError, match="accepted the terms of use"):
            download_guardrail_checkpoint()

    def test_nltk_data_is_materialized_before_returning(self, monkeypatch, tmp_path):
        snap = _hf_cache_like_snapshot(tmp_path)
        _install(monkeypatch, _RecordingSnapshotDownload([str(snap)]))

        download_guardrail_checkpoint()

        assert not (
            snap / "blocklist/nltk_data/tokenizers/punkt_tab/english/collocations.tab"
        ).is_symlink()


def _hf_cache_like_snapshot(tmp_path: pathlib.Path) -> pathlib.Path:
    """blobs/ holds the bytes; snapshots/<rev>/ reaches them through symlinks."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for name in ("colloc", "wordnet", "gore", "resnet"):
        (blobs / name).write_bytes(f"payload:{name}".encode())

    snap = tmp_path / "snapshots" / GUARDRAIL_REVISION
    english = snap / "blocklist" / "nltk_data" / "tokenizers" / "punkt_tab" / "english"
    english.mkdir(parents=True)
    (snap / "blocklist" / "nltk_data" / "corpora").mkdir()
    (snap / "blocklist" / "custom").mkdir()
    (snap / "face_blur_filter").mkdir()

    # hf_hub writes relative symlinks; an absolute one must be handled the same.
    (english / "collocations.tab").symlink_to(os.path.relpath(blobs / "colloc", english))
    (snap / "blocklist" / "nltk_data" / "corpora" / "wordnet.zip").symlink_to(blobs / "wordnet")
    # Under blocklist/ but outside nltk_data: must stay a symlink.
    (snap / "blocklist" / "custom" / "gore").symlink_to(blobs / "gore")
    # Different subtree entirely: must stay a symlink.
    (snap / "face_blur_filter" / "Resnet50_Final.pth").symlink_to(blobs / "resnet")
    return snap


class TestMaterializeNltkData:
    def test_replaces_only_nltk_data_symlinks_with_regular_files(self, tmp_path):
        snap = _hf_cache_like_snapshot(tmp_path)
        nltk = snap / "blocklist" / "nltk_data"

        _materialize_nltk_data(str(snap))

        for rel, blob in (
            ("tokenizers/punkt_tab/english/collocations.tab", "colloc"),
            ("corpora/wordnet.zip", "wordnet"),
        ):
            path = nltk / rel
            assert not path.is_symlink()
            assert path.is_file()
            assert path.read_bytes() == f"payload:{blob}".encode()
            # nltk >= 3.10.3 also refuses hardlinks; a copy must be a fresh inode.
            assert path.stat().st_nlink == 1
        assert (snap / "blocklist" / "custom" / "gore").is_symlink()
        assert (snap / "face_blur_filter" / "Resnet50_Final.pth").is_symlink()
        assert not list(nltk.rglob("*.materialize"))

    def test_is_idempotent(self, tmp_path):
        snap = _hf_cache_like_snapshot(tmp_path)
        target = snap / "blocklist" / "nltk_data" / "corpora" / "wordnet.zip"

        _materialize_nltk_data(str(snap))
        first = target.stat()
        _materialize_nltk_data(str(snap))

        assert target.stat().st_ino == first.st_ino
        assert target.read_bytes() == b"payload:wordnet"

    def test_blobs_are_left_intact(self, tmp_path):
        snap = _hf_cache_like_snapshot(tmp_path)

        _materialize_nltk_data(str(snap))

        assert (tmp_path / "blobs" / "colloc").read_bytes() == b"payload:colloc"

    def test_missing_nltk_data_is_a_no_op(self, tmp_path):
        (tmp_path / "face_blur_filter").mkdir()

        _materialize_nltk_data(str(tmp_path))  # must not raise

    def test_symlinked_directory_is_skipped(self, tmp_path):
        snap = _hf_cache_like_snapshot(tmp_path)
        real_dir = tmp_path / "somewhere"
        real_dir.mkdir()
        link = snap / "blocklist" / "nltk_data" / "linked_dir"
        link.symlink_to(real_dir)

        _materialize_nltk_data(str(snap))

        assert link.is_symlink()

    def test_uses_module_logger_for_the_count(self, tmp_path, monkeypatch):
        messages = []
        monkeypatch.setattr(guardrails.logger, "debug", lambda msg: messages.append(msg))
        snap = _hf_cache_like_snapshot(tmp_path)

        _materialize_nltk_data(str(snap))

        assert any("Materialized 2 nltk_data symlinks" in m for m in messages)
