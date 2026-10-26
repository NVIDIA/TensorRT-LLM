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
"""The shared-path opt-ins the MiniMax-M3.1 sm_107 targets arm: every target's ``_runtime_hooks`` passes each of them
as a literal True, and each is off in the default RuntimeHooks (so a model that declares none, GB300's included, keeps
the stock path). Read from modeling.py's source, so the heavy modeling modules are not imported. CPU only."""

import ast
import dataclasses
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

pytestmark = pytest.mark.cpu_only

SHARED_OPT_INS = (
    "draft_pool_fixes",
    "msa_kv_cache_fixes",
    "draft_kv_reserve_guard",
    "kv_gpu_prefetch",
    "context_token_array",
    "msa_prefill_fork",
    "proxy_ms_reuse",
    "context_onehot_index_fill",
    "dflash_host_ctx_alloc",
)
TARGETS = tuple(f"minimax_m31_nvfp4__sm_107__{tp}" for tp in ("tp2", "tp4", "tp8"))


def _hook_keywords(target: str) -> dict:
    path = Path(_family.__file__).resolve().parent / target / "modeling.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_runtime_hooks"]
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "RuntimeHooks"
    ]
    assert len(calls) == 1, f"{target}: expected one RuntimeHooks(...) in _runtime_hooks"
    return {kw.arg: kw.value for kw in calls[0].keywords}


@pytest.mark.parametrize("target", TARGETS)
def test_target_arms_every_shared_opt_in(target):
    keywords = _hook_keywords(target)
    for field in SHARED_OPT_INS:
        value = keywords.get(field)
        assert isinstance(value, ast.Constant) and value.value is True, (
            f"{target} does not arm {field}"
        )


def test_shared_opt_ins_are_off_by_default():
    defaults = {f.name: f.default for f in dataclasses.fields(RuntimeHooks)}
    assert all(defaults[field] is False for field in SHARED_OPT_INS)
