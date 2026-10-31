# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy rank overrides cannot escape their engine's lifetime."""

import ast
import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ACCURACY_ROOT = Path(__file__).resolve().parents[3] / "integration/defs/accuracy/modeling_v2"
_SOURCE = _ACCURACY_ROOT / "_env.py"
_SPEC = importlib.util.spec_from_file_location("_modeling_v2_accuracy_env", _SOURCE)
_ENV = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ENV)
scoped_env_overrides = _ENV.scoped_env_overrides

_KEY = "TRTLLM_MODELING_V2_TEST_ENV_SCOPE"
_OTHER_KEY = "TRTLLM_MODELING_V2_TEST_ENV_UNRELATED"


@pytest.mark.parametrize("previous", [None, "", "existing"])
@pytest.mark.parametrize("failure", [None, "constructor", "evaluation", "shutdown"])
def test_engine_scope_restores_environment(monkeypatch, previous, failure) -> None:
    if previous is None:
        monkeypatch.delenv(_KEY, raising=False)
    else:
        monkeypatch.setenv(_KEY, previous)
    events = []
    overrides = {_KEY: "engine"}

    class Engine:
        def __init__(self, env_overrides: dict[str, str]) -> None:
            assert env_overrides == overrides
            assert os.environ[_KEY] == "engine"
            events.append("construct")
            if failure == "constructor":
                raise RuntimeError("constructor")

        def __enter__(self) -> "Engine":
            return self

        def __exit__(self, *exc) -> None:
            assert os.environ[_KEY] == "engine"
            events.append("shutdown")
            if failure == "shutdown":
                raise RuntimeError("shutdown")

        def evaluate(self) -> None:
            assert os.environ[_KEY] == "engine"
            events.append("evaluate")
            if failure == "evaluation":
                raise RuntimeError("evaluation")

    def run() -> None:
        with scoped_env_overrides(overrides) as rank_env, Engine(rank_env) as engine:
            assert rank_env is not overrides
            engine.evaluate()

    if failure is None:
        run()
    else:
        with pytest.raises(RuntimeError, match=failure):
            run()

    assert os.environ.get(_KEY) == previous
    assert events == (
        ["construct"] if failure == "constructor" else ["construct", "evaluate", "shutdown"]
    )
    assert overrides == {_KEY: "engine"}


def test_nested_scopes_restore_their_own_values(monkeypatch) -> None:
    monkeypatch.setenv(_KEY, "original")
    with scoped_env_overrides({_KEY: "outer"}):
        assert os.environ[_KEY] == "outer"
        with scoped_env_overrides({_KEY: "inner"}):
            assert os.environ[_KEY] == "inner"
        assert os.environ[_KEY] == "outer"
    assert os.environ[_KEY] == "original"


def test_scope_preserves_unrelated_changes(monkeypatch) -> None:
    monkeypatch.delenv(_KEY, raising=False)
    monkeypatch.setenv(_OTHER_KEY, "before")
    with scoped_env_overrides({_KEY: "engine"}):
        os.environ[_OTHER_KEY] = "during"
    assert _KEY not in os.environ
    assert os.environ[_OTHER_KEY] == "during"


def test_scope_restores_keys_mutated_by_engine(monkeypatch) -> None:
    monkeypatch.setenv(_KEY, "original")
    with scoped_env_overrides({_KEY: "engine"}):
        del os.environ[_KEY]
    assert os.environ[_KEY] == "original"


def test_scope_restores_partial_setup_failure(monkeypatch) -> None:
    monkeypatch.setenv(_KEY, "original")
    with pytest.raises(ValueError, match="null byte"):
        with scoped_env_overrides({_KEY: "engine", "INVALID\x00KEY": "value"}):
            pytest.fail("invalid overrides must fail during scope setup")
    assert os.environ[_KEY] == "original"


@pytest.mark.parametrize(
    "source", sorted(_ACCURACY_ROOT.glob("test_*.py")), ids=lambda path: path.name
)
def test_accuracy_engines_are_inside_environment_scopes(source: Path) -> None:
    tree = ast.parse(source.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LLM"
    ]
    assert calls, f"{source.name} contains no accuracy engine"
    scoped = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        has_scope = False
        for item in node.items:
            context = item.context_expr
            if not isinstance(context, ast.Call) or not isinstance(context.func, ast.Name):
                continue
            if context.func.id == "scoped_env_overrides":
                has_scope = True
            elif context.func.id == "LLM" and has_scope:
                scoped.add(context)
    assert set(calls) <= scoped, (
        f"{source.name} constructs LLM outside an environment scope at lines "
        f"{sorted(call.lineno for call in set(calls) - scoped)}"
    )
