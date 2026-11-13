# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GEN_NATIVE_GLUE CPU controls: the TP2 kill switches and the generic drafter SiLU routing, without CUDA imports."""

import ast
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[7]
TARGET = ROOT / (
    "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2"
)
MODEL = TARGET / "modeling.py"


def _load_glue(monkeypatch, env: dict):
    for name in list(sys.modules):
        if name.startswith("t2u2_glue_pkg"):
            del sys.modules[name]
    for key in [k for k in __import__("os").environ if k.startswith("TRTLLM_M31_")]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    package = types.ModuleType("t2u2_glue_pkg")
    package.__path__ = [str(TARGET)]
    with patch.dict(sys.modules, {"t2u2_glue_pkg": package}):
        mods = {}
        for name in ("m31_switches", "gen_glue"):
            spec = importlib.util.spec_from_file_location(
                f"t2u2_glue_pkg.{name}", TARGET / f"{name}.py"
            )
            mods[name] = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mods[name]
            spec.loader.exec_module(mods[name])
        return mods["gen_glue"]


# Each part with the item ids its call site passes (modeling.py / dspark.py).
LEGACY = {
    "FSEL": ("TP2_FSEL",),
    "NATIVE_GLUE": ("DRAFT_SWIGLU",),
    "SHARED_SWIGLU_MXFP8_NATIVE": ("SHARED_SWIGLU_MXFP8_NATIVE",),
}
# Each part's own kill switch id.
PART_SWITCH = {
    "FSEL": "TP2_FSEL",
    "NATIVE_GLUE": "GEN_GLUE_NATIVE",
    "SHARED_SWIGLU_MXFP8_NATIVE": "SHARED_SWIGLU_MXFP8_NATIVE",
}


def test_default_on(monkeypatch) -> None:
    glue = _load_glue(monkeypatch, {})
    for part, legacy in LEGACY.items():
        assert glue.glue_disabled(part, *legacy) is False


def test_item_switch_kills_every_part(monkeypatch) -> None:
    glue = _load_glue(monkeypatch, {"TRTLLM_M31_DISABLE_GEN_NATIVE_GLUE": "1"})
    for part, legacy in LEGACY.items():
        assert glue.glue_disabled(part, *legacy) is True


@pytest.mark.parametrize("part", sorted(LEGACY))
def test_part_switch_kills_only_that_part(monkeypatch, part) -> None:
    glue = _load_glue(monkeypatch, {f"TRTLLM_M31_DISABLE_{PART_SWITCH[part]}": "on"})
    for other, legacy in LEGACY.items():
        assert glue.glue_disabled(other, *legacy) is (other == part)


@pytest.mark.parametrize("part", sorted(LEGACY))
def test_legacy_switch_still_kills(monkeypatch, part) -> None:
    glue = _load_glue(monkeypatch, {f"TRTLLM_M31_DISABLE_{LEGACY[part][0]}": "1"})
    for other, legacy in LEGACY.items():
        assert glue.glue_disabled(other, *legacy) is (other == part)


def test_bad_value_fails_loudly(monkeypatch) -> None:
    with pytest.raises(ValueError, match="TRTLLM_M31_DISABLE_TP2_FSEL"):
        _load_glue(monkeypatch, {"TRTLLM_M31_DISABLE_TP2_FSEL": "maybe"})


def test_engagement_logged_once_per_process(monkeypatch) -> None:
    glue = _load_glue(monkeypatch, {})
    log = Mock(spec_set=["info"])
    glue.log_engagement(log, 0, FSEL="rows<=8", SHARED_SWIGLU_MXFP8_NATIVE="rows 17..32")
    glue.log_engagement(log, 0, FSEL="rows<=8", SHARED_SWIGLU_MXFP8_NATIVE="rows 17..32")
    log.info.assert_called_once()
    line = log.info.call_args.args[0]
    assert "GEN_NATIVE_GLUE engaged on rank 0" in line
    assert "SHARED_SWIGLU_MXFP8_NATIVE=rows 17..32" in line


def _assignments():
    tree = ast.parse(MODEL.read_text())
    return tree, {
        n.targets[0].id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }


def test_generic_drafter_silu_switch() -> None:
    _, assignments = _assignments()
    expr = compile(ast.Expression(assignments["DRAFT_SILU_GENERIC_CPP"]), str(MODEL), "eval")
    for base in (False, True):
        for killed in (False, True):
            got = eval(
                expr,
                {"DRAFT_SILU_CPP": base, "glue_disabled": lambda part, *legacy: killed},
            )
            assert got is (base and not killed)


def test_generic_drafter_layer_routes_the_cpp_silu() -> None:
    tree, _ = _assignments()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mlp"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    ]
    assert calls, "the generic drafter layer's self.mlp call is missing"
    for call in calls:
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kw.get("silu_cpp") == "DRAFT_SILU_GENERIC_CPP", ast.unparse(call)


def test_drafter_mlp_takes_cpp_silu_and_falls_back() -> None:
    tree, _ = _assignments()
    mlp = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "mlp"
        and [a.arg for a in node.args.args] == ["self", "h", "reduce", "silu_cpp"]
    )
    for silu_cpp, cpp_ok, want in (
        (True, True, "cpp"),
        (True, False, "triton"),
        (False, True, "triton"),
    ):
        cpp, triton = object(), object()
        scope = dict(
            DRAFT_SILU_CPP=True,
            DRAFT_SILU_PATH="m3_draft_silu",
            m3_draft_silu=Mock(return_value=cpp if cpp_ok else None),
            m3_silu_and_mul=Mock(return_value=triton),
            taken=Mock(),
            uncovered=Mock(),
            m3_allreduce=Mock(),
            torch=types.SimpleNamespace(Tensor=object),
        )
        exec(compile(ast.Module(body=[mlp], type_ignores=[]), str(MODEL), "exec"), scope)
        seen = []
        gu = types.SimpleNamespace(shape=(7, 18432))
        owner = types.SimpleNamespace(
            _linear=lambda x, key: gu if key == "gu" else seen.append(x) or x,
            allreduce=None,
        )
        scope["mlp"](owner, object(), False, silu_cpp)
        assert seen == [cpp if want == "cpp" else triton]
        assert scope["m3_silu_and_mul"].called is (want == "triton")
