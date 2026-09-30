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
"""Tests for how Cosmos3 fetches its guardrail checkpoint.

No GPU, no network: ``snapshot_download`` is replaced with a recorder.
"""

import huggingface_hub
import pytest
from huggingface_hub.errors import GatedRepoError

from tensorrt_llm._torch.visual_gen.models.cosmos3.guardrails import (
    GUARDRAIL_ALLOW_PATTERNS,
    GUARDRAIL_HF_REPO,
    GUARDRAIL_REVISION,
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


class _GatedRepo(GatedRepoError):
    """A GatedRepoError that skips HfHubHTTPError.__init__.

    Depending on the huggingface_hub version that constructor demands a live
    ``requests.Response``; the code under test only needs the exception type.
    """

    def __init__(self):
        Exception.__init__(self, "gated")


def _pinned(call):
    return (call["repo_id"], call["revision"], call["allow_patterns"])


def _install(monkeypatch, fake):
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return fake


def test_allow_patterns_name_exactly_the_two_subtrees_the_guardrail_loads():
    # The download tests compare calls against this constant, so they cannot
    # notice it regressing to None or to the whole repo. Pin the literal here.
    assert GUARDRAIL_ALLOW_PATTERNS == ["blocklist/*", "face_blur_filter/*"]


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
        _install(monkeypatch, _RecordingSnapshotDownload([FileNotFoundError(), _GatedRepo()]))

        with pytest.raises(ValueError, match="accepted the terms of use") as excinfo:
            download_guardrail_checkpoint()

        assert isinstance(excinfo.value.__cause__, GatedRepoError)
