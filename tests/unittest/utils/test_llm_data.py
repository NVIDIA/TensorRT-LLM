# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Unit tests for the fail-loud checkpoint resolver in ``test_common.llm_data``."""

import pytest

from utils.llm_data import get_checkpoint


@pytest.mark.cpu_only
def test_get_checkpoint_returns_staged_path(tmp_path, monkeypatch) -> None:
    """A staged checkpoint under LLM_MODELS_ROOT resolves to its absolute path."""
    monkeypatch.setenv("LLM_MODELS_ROOT", str(tmp_path))
    (tmp_path / "my-model").mkdir()

    assert get_checkpoint("my-model") == str(tmp_path / "my-model")


@pytest.mark.cpu_only
def test_get_checkpoint_raises_on_missing(tmp_path, monkeypatch) -> None:
    """A missing checkpoint fails loudly instead of silently skipping the test."""
    monkeypatch.setenv("LLM_MODELS_ROOT", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="absent-model"):
        get_checkpoint("absent-model")
