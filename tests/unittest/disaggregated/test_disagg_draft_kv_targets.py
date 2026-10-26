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
"""The MiniMax-M3.1 sm_107 targets opt in to the disagg drafter context window.

Each target's ``RuntimeHooks`` is read from its modeling.py source (the targets need SM107 kernels
to import), so the opt-in is checked on any machine.
"""

import ast
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

pytestmark = pytest.mark.cpu_only

_FAMILY = Path(_family.__file__).resolve().parent
_SM107_TARGETS = ("tp2", "tp4", "tp8")


def _runtime_hooks_keywords(modeling: Path) -> dict:
    tree = ast.parse(modeling.read_text(), filename=str(modeling))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RuntimeHooks"
    ]
    assert len(calls) == 1, f"{modeling}: expected one RuntimeHooks(...) call, found {len(calls)}"
    return {kw.arg: kw.value for kw in calls[0].keywords}


@pytest.mark.parametrize("tp", _SM107_TARGETS)
def test_sm107_target_opts_in_to_the_draft_window_transfer(tp):
    modeling = _FAMILY / f"minimax_m31_nvfp4__sm_107__{tp}" / "modeling.py"
    keywords = _runtime_hooks_keywords(modeling)
    assert "disagg_draft_kv" in keywords, f"{modeling.parent.name} does not set disagg_draft_kv"
    value = keywords["disagg_draft_kv"]
    assert isinstance(value, ast.Constant) and value.value is True
