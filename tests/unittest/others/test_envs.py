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

import ast
from pathlib import Path

import pytest

from tensorrt_llm import envs


def _resolve_env_name_arg(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "envs"
        and hasattr(envs, node.attr)
    ):
        attr_value = getattr(envs, node.attr)
        if isinstance(attr_value, str):
            return attr_value
    return None


def test_get_env_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv(envs.TLLM_BENCHMARK_REQ_QUEUES_SIZE, raising=False)
    assert envs.get_env(envs.TLLM_BENCHMARK_REQ_QUEUES_SIZE) == 0


def test_get_env_parses_boolean_values(monkeypatch):
    monkeypatch.setenv(envs.TRTLLM_SERVER_DISABLE_GC, "1")
    assert envs.get_env(envs.TRTLLM_SERVER_DISABLE_GC) is True

    monkeypatch.setenv(envs.TRTLLM_SERVER_DISABLE_GC, "false")
    assert envs.get_env(envs.TRTLLM_SERVER_DISABLE_GC) is False


def test_get_env_parses_int_and_float(monkeypatch):
    monkeypatch.setenv(envs.TLLM_BENCHMARK_REQ_QUEUES_SIZE, "17")
    assert envs.get_env(envs.TLLM_BENCHMARK_REQ_QUEUES_SIZE) == 17

    monkeypatch.setenv(envs.TRTLLM_RAY_PER_WORKER_GPUS, "2.5")
    assert envs.get_env(envs.TRTLLM_RAY_PER_WORKER_GPUS) == 2.5


def test_get_env_returns_structured_range_as_string(monkeypatch):
    monkeypatch.setenv(envs.EXPERT_STATISTIC_ITER_RANGE, "100-200")
    assert envs.get_env(envs.EXPERT_STATISTIC_ITER_RANGE) == "100-200"


def test_get_env_raises_for_invalid_value(monkeypatch):
    monkeypatch.setenv(envs.TRTLLM_SERVER_DISABLE_GC, "maybe")
    with pytest.raises(ValueError, match=envs.TRTLLM_SERVER_DISABLE_GC):
        envs.get_env(envs.TRTLLM_SERVER_DISABLE_GC)


def test_get_env_parses_override_default_by_spec_type(monkeypatch):
    monkeypatch.delenv(envs.TRTLLM_SERVER_DISABLE_GC, raising=False)
    assert envs.get_env(envs.TRTLLM_SERVER_DISABLE_GC, "1") is True
    assert envs.get_env(envs.TRTLLM_SERVER_DISABLE_GC, "0") is False


def test_list_envs_and_get_spec():
    specs = envs.list_envs()
    assert specs
    assert any(spec.name == envs.TLLM_ALLOW_LONG_MAX_MODEL_LEN for spec in specs)
    assert envs.get_spec(envs.TLLM_ALLOW_LONG_MAX_MODEL_LEN).type == "bool"
    assert envs.get_spec(envs.EXPERT_STATISTIC_ITER_RANGE).type == "str"


def test_env_specs_cover_all_env_names():
    for key, spec in envs.ENV_SPECS.items():
        assert spec.name == key


def test_env_constants_cover_all_env_names():
    env_symbol_by_name = {}
    for attr_name, attr_value in vars(envs).items():
        if isinstance(attr_value, str) and attr_value in envs.ENV_SPECS:
            env_symbol_by_name.setdefault(attr_value, set()).add(attr_name)

    for env_name in envs.ENV_SPECS:
        assert env_name in env_symbol_by_name
        if env_name == "tllm_mpi_size":
            assert "TLLM_MPI_SIZE" in env_symbol_by_name[env_name]
        else:
            assert env_name in env_symbol_by_name[env_name]


def test_env_specs_do_not_use_generic_placeholder_docs():
    for name, spec in envs.ENV_SPECS.items():
        assert spec.doc != f"TensorRT-LLM environment variable {name}."


def test_selected_env_specs_types_and_defaults():
    expected = {
        "AD_DUMP_GRAPHS_DIR": ("str", None),
        "BUILDER_FORCE_NUM_PROFILES": ("int", None),
        "LM_HEAD_TP_SIZE": ("int", None),
        "OVERRIDE_QUANT_ALGO": ("str", None),
        "TLLM_AUTOTUNER_CACHE_PATH": ("str", None),
        "TLLM_NUMA_AWARE_WORKER_AFFINITY": ("str", None),
        "TLLM_PROFILE_START_STOP": ("str", None),
        "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR": ("str", None),
        "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY": ("str", None),
        "TLLM_TORCH_PROFILE_TRACE": ("str", None),
        "TRTLLM_ALLREDUCE_FUSION_WORKSPACE_SIZE": ("int", None),
        "TRTLLM_DEEP_EP_TOKEN_LIMIT": ("int", None),
        "TRTLLM_DISABLE_UNIFIED_CONVERTER": ("str", None),
        "TRTLLM_FORCE_ALLTOALL_METHOD": ("str", None),
        "TRTLLM_FORCE_COMM_METHOD": ("str", None),
        "TRTLLM_MOE_A2A_WORKSPACE_MB": ("int", None),
        "TRTLLM_PRINT_SKIP_SOFTMAX_STAT": ("bool", False),
        "TRTLLM_PRINT_STACKS_PERIOD": ("int", -1),
        "TRTLLM_RAY_BUNDLE_INDICES": ("str", None),
        "TRTLLM_WINDOW_SIZE_SHARES": ("str", None),
        "TRTLLM_WORKER_PRINT_STACKS_PERIOD": ("int", -1),
        "ENABLE_CONFIGURABLE_MOE": ("bool", True),
        "TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION": ("bool", True),
    }
    for name, (typ, default) in expected.items():
        spec = envs.get_spec(name)
        assert spec.type == typ
        assert spec.default == default


def test_internal_code_uses_get_env_only():
    project_root = Path(__file__).resolve().parents[3]
    violations = []

    for py_file in (project_root / "tensorrt_llm").rglob("*.py"):
        if py_file.name == "envs.py":
            continue
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "getenv":
                continue
            if not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id != "envs":
                continue
            violations.append(f"{py_file}:{node.lineno}")

    assert not violations, "Use envs.get_env for all TensorRT-LLM env access:\n" + "\n".join(
        violations
    )


def test_internal_code_uses_env_constants_for_known_env_names():
    project_root = Path(__file__).resolve().parents[3]
    violations = []

    for py_file in (project_root / "tensorrt_llm").rglob("*.py"):
        if py_file.name == "envs.py":
            continue
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "get_env":
                continue
            if not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id != "envs":
                continue
            if not node.args:
                continue
            arg0 = node.args[0]
            if not isinstance(arg0, ast.Constant):
                continue
            if not isinstance(arg0.value, str):
                continue
            if arg0.value not in envs.ENV_SPECS:
                continue
            violations.append(f"{py_file}:{node.lineno}:{arg0.value}")

    assert not violations, (
        "Use envs.<ENV_NAME> constants instead of string literals in envs.get_env for known env vars:\n"
        + "\n".join(violations)
    )


def test_internal_code_has_no_redundant_typed_wrappers():
    expected_wrapper_by_type = {
        "int": "int",
        "float": "float",
        "bool": "bool",
    }
    project_root = Path(__file__).resolve().parents[3]
    violations = []

    for py_file in (project_root / "tensorrt_llm").rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            wrapper_type = expected_wrapper_by_type.get(node.func.id)
            if wrapper_type is None:
                continue
            if len(node.args) != 1:
                continue
            inner = node.args[0]
            if not isinstance(inner, ast.Call):
                continue
            if not isinstance(inner.func, ast.Attribute):
                continue
            if inner.func.attr != "get_env":
                continue
            if not isinstance(inner.func.value, ast.Name):
                continue
            if inner.func.value.id != "envs":
                continue
            if not inner.args:
                continue
            env_name = _resolve_env_name_arg(inner.args[0])
            if env_name not in envs.ENV_SPECS:
                continue
            if envs.ENV_SPECS[env_name].type == wrapper_type:
                violations.append(f"{py_file}:{node.lineno}:{env_name}")

    assert not violations, (
        "Avoid redundant int/float/bool wrappers around envs.get_env for typed envs:\n"
        + "\n".join(violations)
    )


def test_internal_code_has_no_redundant_defaults_equal_env_specs():
    true_values = {"1", "true", "yes", "y", "on"}
    false_values = {"0", "false", "no", "n", "off"}

    def default_equals_spec(env_name, value):
        spec = envs.ENV_SPECS[env_name]
        if value is None:
            return spec.default is None
        if spec.type == "str":
            return isinstance(value, str) and value == spec.default
        if spec.type == "int":
            try:
                return int(value) == spec.default
            except (TypeError, ValueError):
                return False
        if spec.type == "float":
            try:
                return float(value) == float(spec.default)
            except (TypeError, ValueError):
                return False
        if spec.type == "bool":
            if isinstance(value, bool):
                return value == spec.default
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in true_values:
                    return spec.default is True
                if normalized in false_values:
                    return spec.default is False
            return False
        return False

    project_root = Path(__file__).resolve().parents[3]
    violations = []

    for py_file in (project_root / "tensorrt_llm").rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "get_env":
                continue
            if not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id != "envs":
                continue
            if len(node.args) < 2:
                continue
            env_name = _resolve_env_name_arg(node.args[0])
            if env_name is None:
                continue
            if not isinstance(node.args[1], ast.Constant):
                continue
            if env_name not in envs.ENV_SPECS:
                continue
            if default_equals_spec(env_name, node.args[1].value):
                violations.append(f"{py_file}:{node.lineno}:{env_name}")

    assert not violations, (
        "Remove redundant get_env defaults that duplicate EnvSpec defaults:\n"
        + "\n".join(violations)
    )
