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
"""tensorrt_llm._torch.q35_env: the Qwen3.5 fast-path gate and the TRTLLM_Q35_PROFILE role profiles.

The GPU is mocked, so this runs on any machine.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch import q35_env

_GEN_NAMES = {
    "TRTLLM_Q35_CACHED_STREAM_EVENTS",
    "TRTLLM_Q35_DRAFT_BOOKKEEPING",
    "TRTLLM_Q35_DRAFT_FP8_HEAD",
    "TRTLLM_Q35_DRAFT_TAIL",
    "TRTLLM_Q35_FUSED_ACCEPT",
    "TRTLLM_Q35_FUSED_STEP0",
    "TRTLLM_Q35_LL_ARGMAX_EXCHANGE",
    "TRTLLM_Q35_PDL_EARLY_TRIGGER",
    "TRTLLM_Q35_POSITIONS_HOIST",
    "TRTLLM_Q35_REPLICATED_EMBED",
    "TRTLLM_Q35_SAMPLER_EVENT_RING",
    "TRTLLM_Q35_SAMPLER_HOST_RING",
    "TRTLLM_Q35_SAMPLER_ZEROCOPY",
    "TRTLLM_Q35_SAMPLING_SCAN_CACHE",
    "TRTLLM_Q35_SHARDED_TARGET_SAMPLING",
    "TRTLLM_Q35_SKIP_MROPE_COS_SIN",
    "TRTLLM_Q35_SPEC_PREPARE_CACHE",
}
_CTX_NAMES = {
    "TRTLLM_Q35_BRANCH_SNAPSHOT",
    "TRTLLM_Q35_CTX_ATTN_PLAN",
    "TRTLLM_Q35_CTX_LEAN_TOKENS",
    "TRTLLM_Q35_CUSTOM_OP_EAGER",
    "TRTLLM_Q35_GDN_GRAPH_DIRECT",
    "TRTLLM_Q35_GDN_GRAPH_LEAN",
    "TRTLLM_Q35_GDN_PREFILL_GRAPH",
    "TRTLLM_Q35_SKIP_MROPE_COS_SIN",
}
_ALL_SWITCHES = set(q35_env._GEN) | set(q35_env._CTX)


def _config(model_type="qwen3_5_moe", attention_dp=False):
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(model_type=model_type),
        mapping=SimpleNamespace(enable_attention_dp=attention_dp),
    )


def _q35_names():
    return {name for name in os.environ if name.startswith("TRTLLM_Q35_")}


def _switches_set():
    return _q35_names() - {q35_env.PROFILE_ENV}


@pytest.fixture
def env(monkeypatch):
    """No TRTLLM_Q35_* variable set, fresh activation state and one recording reader instead of the real ones.

    The reader records (fast paths on, TRTLLM_Q35_FUSED_ACCEPT) every time it runs. Tests set TRTLLM_Q35_*
    variables through os.environ directly; the fixture removes all of them afterwards and restores the ones that
    were set before.
    """
    saved = {name: os.environ.pop(name) for name in _q35_names()}
    seen = []
    monkeypatch.setattr(
        q35_env,
        "_readers",
        [
            lambda: seen.append(
                (q35_env.fast_paths_enabled(), os.environ.get("TRTLLM_Q35_FUSED_ACCEPT"))
            )
        ],
    )
    monkeypatch.setattr(q35_env, "_profile_applied", False)
    monkeypatch.setattr(q35_env, "_fast_paths", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 3))
    yield monkeypatch, seen
    for name in _q35_names():
        del os.environ[name]
    os.environ.update(saved)


def test_tables():
    assert set(q35_env._GEN) == _GEN_NAMES
    assert set(q35_env._CTX) == _CTX_NAMES
    assert all(value == "1" for value in {**q35_env._GEN, **q35_env._CTX}.values())
    assert q35_env.PROFILE_ENV not in _ALL_SWITCHES


@pytest.mark.parametrize("value", [None, "", "off"])
def test_profile_unset_enables_fast_paths_only(env, value):
    _, seen = env
    if value is not None:
        os.environ[q35_env.PROFILE_ENV] = value
    q35_env.activate(_config())
    assert not _switches_set()
    assert q35_env.fast_paths_enabled()
    assert seen == [(True, None)]


@pytest.mark.parametrize("attention_dp", [False, True])
def test_gen_profile(env, attention_dp):
    _, seen = env
    os.environ[q35_env.PROFILE_ENV] = "gen"
    q35_env.activate(_config(attention_dp=attention_dp))
    assert {name: os.environ[name] for name in _switches_set()} == q35_env._GEN
    assert seen == [(True, "1")]  # the readers ran once, after the environment was set


def test_ctx_profile_without_attention_dp(env):
    _, seen = env
    os.environ[q35_env.PROFILE_ENV] = "ctx"
    q35_env.activate(_config(attention_dp=False))
    assert {name: os.environ[name] for name in _switches_set()} == dict.fromkeys(_CTX_NAMES, "1")
    assert seen == [(True, None)]


def test_ctx_profile_with_attention_dp(env):
    _, seen = env
    os.environ[q35_env.PROFILE_ENV] = "ctx"
    q35_env.activate(_config(attention_dp=True))
    assert not _switches_set()
    assert seen == [(True, None)]


@pytest.mark.parametrize("profile", ["gen", "ctx"])
def test_environment_wins(env, profile):
    os.environ[q35_env.PROFILE_ENV] = profile
    os.environ["TRTLLM_Q35_SKIP_MROPE_COS_SIN"] = "0"  # in both profiles
    q35_env.activate(_config())
    assert os.environ["TRTLLM_Q35_SKIP_MROPE_COS_SIN"] == "0"
    expected = q35_env.profile_defaults(profile, False)
    assert "TRTLLM_Q35_SKIP_MROPE_COS_SIN" in expected
    for name, value in expected.items():
        if name != "TRTLLM_Q35_SKIP_MROPE_COS_SIN":
            assert os.environ[name] == value, name


@pytest.mark.parametrize("value", ["on", "1", "GEN", "gen,ctx", "decode"])
def test_invalid_profile_raises(env, value):
    _, seen = env
    os.environ[q35_env.PROFILE_ENV] = value
    with pytest.raises(ValueError, match=q35_env.PROFILE_ENV):
        q35_env.activate(_config())
    assert not _switches_set()
    assert seen == []


@pytest.mark.parametrize("profile", [None, "gen", "ctx"])
@pytest.mark.parametrize(
    "model_type,capability",
    [
        ("qwen3_next", (10, 3)),
        ("llama", (10, 3)),
        ("qwen3_5_moe", (10, 0)),
        ("qwen3_5_moe", (9, 0)),
    ],
)
def test_other_models_and_gpus(env, profile, model_type, capability):
    monkeypatch, seen = env
    if profile is not None:
        os.environ[q35_env.PROFILE_ENV] = profile
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)
    q35_env.activate(_config(model_type=model_type))
    assert not _switches_set()
    assert not q35_env.fast_paths_enabled()
    assert seen == []


def test_no_gpu(env):
    monkeypatch, seen = env
    os.environ[q35_env.PROFILE_ENV] = "gen"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    q35_env.activate(_config())
    assert not _switches_set()
    assert not q35_env.fast_paths_enabled()
    assert seen == []


def test_profile_once_per_process(env):
    _, seen = env
    os.environ[q35_env.PROFILE_ENV] = "gen"
    q35_env.activate(_config())
    assert seen == [(True, "1")]
    os.environ[q35_env.PROFILE_ENV] = "ctx"
    del os.environ["TRTLLM_Q35_FUSED_ACCEPT"]
    q35_env.activate(_config())
    assert seen == [(True, "1")]
    assert "TRTLLM_Q35_FUSED_ACCEPT" not in os.environ
    assert "TRTLLM_Q35_BRANCH_SNAPSHOT" not in os.environ


def test_fast_paths_follow_the_loaded_model(env):
    monkeypatch, seen = env
    q35_env.activate(_config())
    assert q35_env.fast_paths_enabled()
    q35_env.activate(_config(model_type="qwen3_next"))
    assert not q35_env.fast_paths_enabled()
    assert seen == [(True, None), (False, None)]


def test_set_fast_paths_reruns_readers(env):
    _, seen = env
    q35_env.set_fast_paths(True)
    q35_env.set_fast_paths(True)
    q35_env.set_fast_paths(False)
    assert seen == [(True, None), (False, None)]


def _is_environ(node):
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ"
    )


def _import_time_reads(tree):
    """(line, name) of every environment read outside a function body whose variable name is a string literal or a
    module-level string constant."""
    constants = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant):
            if isinstance(stmt.value.value, str):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = stmt.value.value

    def name_of(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        return None

    reads = []
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        name = None
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            environ_method = (
                isinstance(func, ast.Attribute)
                and func.attr in ("get", "pop", "setdefault")
                and _is_environ(func.value)
            )
            getenv = (isinstance(func, ast.Attribute) and func.attr == "getenv") or (
                isinstance(func, ast.Name) and func.id == "getenv"
            )
            if environ_method or getenv:
                name = name_of(node.args[0])
        elif isinstance(node, ast.Subscript) and _is_environ(node.value):
            name = name_of(node.slice)
        elif isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in node.ops
        ):
            if any(_is_environ(comparator) for comparator in node.comparators):
                name = name_of(node.left)
        if name is not None:
            reads.append((node.lineno, name))
        stack.extend(ast.iter_child_nodes(node))
    return reads


def test_profile_switches_are_not_read_at_import():
    """A profile switch read at import time would not see the profile: it has to be read in a function (at call
    time, or in a @q35_env.register reader that activate() re-runs)."""
    package = Path(tensorrt_llm.__file__).parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for line, name in _import_time_reads(tree):
            if name in _ALL_SWITCHES:
                offenders.append(f"{path.relative_to(package.parent)}:{line}: {name}")
    assert not offenders, "TRTLLM_Q35_* profile switches read at import time:\n" + "\n".join(
        sorted(offenders)
    )
