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
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _pattern_matcher import PatternMatcher  # noqa: E402


@pytest.fixture
def matcher(tmp_path: Path) -> PatternMatcher:
    deps = [
        {
            "name": "googletest",
            "git_repository": "https://github.com/google/googletest",
            "git_tag": "v1.15.2",
        },
        {
            "name": "eigen",
            "git_repository": "https://gitlab.com/libeigen/eigen.git",
            "git_tag": "3.4.0",
        },
        {
            "name": "deepep",
            "display_name": "DeepEP",
            "git_repository": "${github_base_url}/deepseek-ai/DeepEP",
            "git_tag": "abc123",
        },
    ]
    fetch_content = tmp_path / "fetch_content.json"
    fetch_content.write_text(json.dumps({"dependencies": deps}))
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    return PatternMatcher(metadata_dir, fetch_content)


def test_source_url_keeps_trailing_git_letters(matcher):
    # "googletest" ends in "t", which a character-set rstrip(".git") would eat.
    assert (
        matcher.get_source_url("googletest", "v1.15.2")
        == "https://github.com/google/googletest/tree/v1.15.2"
    )


def test_source_url_removes_dot_git_suffix(matcher):
    assert matcher.get_source_url("eigen") == "https://gitlab.com/libeigen/eigen/tree/3.4.0"


def test_source_url_expands_github_base_url(matcher):
    assert matcher.get_source_url("DeepEP") == "https://github.com/deepseek-ai/DeepEP/tree/abc123"


def test_source_url_version_mismatch(matcher):
    assert matcher.get_source_url("googletest", "v1.0.0") == ""
