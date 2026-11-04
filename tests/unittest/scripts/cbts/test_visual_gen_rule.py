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
"""VisualGenRule claims the VisualGen serving package and narrows to VisualGen entries."""

from __future__ import annotations

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import Block, Stage, YAMLIndex
from cbts.command.main import Selector
from cbts.rules.base import PRInputs
from cbts.rules.visual_gen import _VG_OUTWARD_PREFIXES, VisualGenRule, _is_vg_claim

pytestmark = pytest.mark.cpu_only

SERVE_PACKAGE = "tensorrt_llm/serve/api/visual_gen/"
CPU_STAGE = "CPU-Generic-x86-1"
VISUAL_GEN_ENTRIES = (
    "unittest/_torch/visual_gen/test_trtllm_serve_endpoints.py",
    "unittest/llmapi/apps/visual_gen/test_serve_visual_gen_isolation.py",
    "unittest/api_stability/visual_gen/test_visual_gen_serve_api.py",
)
OTHER_ENTRIES = (
    "unittest/api_stability",
    "unittest/llmapi/apps/test_serve_lazy_import.py",
)


def _rule() -> tuple[VisualGenRule, dict[str, Stage]]:
    index = YAMLIndex()
    index.blocks = [Block("l0_cpu", 0, {}, [*VISUAL_GEN_ENTRIES, *OTHER_ENTRIES])]
    stages = {CPU_STAGE: Stage(CPU_STAGE, "l0_cpu", "x86_64", 1, 1)}
    return VisualGenRule(index, stages), stages


@pytest.mark.parametrize(
    "path",
    [
        SERVE_PACKAGE + "serving_videos.py",
        SERVE_PACKAGE + "protocol.py",
        SERVE_PACKAGE + "api_router.py",
        SERVE_PACKAGE + "__init__.py",
    ],
)
def test_serve_visual_gen_package_is_claimed(path: str) -> None:
    assert _is_vg_claim(path)


@pytest.mark.parametrize(
    "path",
    [
        SERVE_PACKAGE + "README.md",
        "tensorrt_llm/serve/api/visual_gen_extra/x.py",
        "tensorrt_llm/serve/api/openai/chat/serving.py",
        "tensorrt_llm/serve/openai_server.py",
        "tensorrt_llm/serve/openai_protocol.py",
        "tensorrt_llm/serve/visual_gen_utils.py",
    ],
)
def test_paths_next_to_the_package_are_not_claimed(path: str) -> None:
    assert not _is_vg_claim(path)


def test_serve_visual_gen_package_is_not_outward() -> None:
    assert not any(SERVE_PACKAGE.startswith(prefix) for prefix in _VG_OUTWARD_PREFIXES)


def test_serve_visual_gen_change_narrows_to_visual_gen_entries() -> None:
    rule, _ = _rule()
    changed = [SERVE_PACKAGE + "serving_videos.py", SERVE_PACKAGE + "protocol.py"]

    result = rule.apply(PRInputs(changed_files=changed, diffs={}))

    assert result is not None
    assert result.scope == "visualgenonly"
    assert result.handled_files == set(changed)
    assert result.affected_stages == {CPU_STAGE}
    assert result.sanity_relevant is False
    assert set(result.block_filters[("l0_cpu", 0)]) == set(VISUAL_GEN_ENTRIES)


def test_mixed_change_falls_back() -> None:
    rule, stages = _rule()
    pr = PRInputs(
        changed_files=[SERVE_PACKAGE + "serving_videos.py", "tensorrt_llm/serve/openai_server.py"],
        diffs={},
    )

    result = Selector(stages).run(pr, [rule])

    assert result.scope is None
    fallbacks = [
        reason
        for reason in result.reasons
        if isinstance(reason, dict) and reason.get("reason") == "unhandled_files"
    ]
    assert len(fallbacks) == 1
    assert fallbacks[0]["files"] == ["tensorrt_llm/serve/openai_server.py"]
