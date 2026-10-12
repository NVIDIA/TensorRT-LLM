# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA CuTe DSL backend selection and optional scheduling hints."""

import importlib

import cutlass.cute as cute
import pytest

import tensorrt_llm._torch.attention.backends.fp4_mla.config as fp4_mla_config
import tensorrt_llm._torch.attention.backends.fp4_mla.cute_dsl_utils as cute_dsl_utils

pytestmark = pytest.mark.cpu_only

_HINTS = ("cfence", "warp_switch")


@pytest.fixture
def reload_cute_dsl_utils():
    yield lambda: importlib.reload(cute_dsl_utils)
    importlib.reload(cute_dsl_utils)


@pytest.fixture
def fresh_availability():
    fp4_mla_config._cutedsl_backend_available.cache_clear()
    yield
    fp4_mla_config._cutedsl_backend_available.cache_clear()


@pytest.mark.parametrize("hint", _HINTS)
def test_missing_schedule_hint_is_a_no_op(monkeypatch, reload_cute_dsl_utils, hint):
    monkeypatch.delattr(cute.nvgpu, hint, raising=False)

    utils = reload_cute_dsl_utils()

    assert getattr(utils, hint) is utils._skip_schedule_hint
    assert getattr(utils, hint)() is None


@pytest.mark.parametrize("hint", _HINTS)
def test_available_schedule_hint_is_the_dsl_function(monkeypatch, reload_cute_dsl_utils, hint):
    calls = []

    def dsl_hint():
        calls.append(hint)

    monkeypatch.setattr(cute.nvgpu, hint, dsl_hint, raising=False)

    utils = reload_cute_dsl_utils()
    getattr(utils, hint)()

    assert getattr(utils, hint) is dsl_hint
    assert calls == [hint]


def test_cutedsl_availability_needs_only_the_kernel_imports(monkeypatch, fresh_availability):
    queried = []

    def find_spec(name):
        queried.append(name)
        return object()

    monkeypatch.setattr(fp4_mla_config.importlib.util, "find_spec", find_spec)

    assert fp4_mla_config._cutedsl_backend_available()
    assert "ctm" not in queried
    assert set(queried) == set(fp4_mla_config._CUTEDSL_REQUIRED_MODULES)


@pytest.mark.parametrize("missing", fp4_mla_config._CUTEDSL_REQUIRED_MODULES)
def test_cutedsl_availability_requires_each_kernel_import(monkeypatch, fresh_availability, missing):
    monkeypatch.setattr(
        fp4_mla_config.importlib.util,
        "find_spec",
        lambda name: None if name == missing else object(),
    )

    assert not fp4_mla_config._cutedsl_backend_available()


def test_cutedsl_availability_without_cutlass(monkeypatch, fresh_availability):
    def find_spec(name):
        raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")

    monkeypatch.setattr(fp4_mla_config.importlib.util, "find_spec", find_spec)

    assert not fp4_mla_config._cutedsl_backend_available()


@pytest.mark.parametrize(
    ("sm", "available", "expected"),
    [
        (107, True, "cutedsl"),
        (107, False, "triton"),
        (100, True, "triton"),
    ],
)
def test_default_backend_selection(monkeypatch, sm, available, expected):
    monkeypatch.delenv(fp4_mla_config.FP4_MLA_ATTENTION_BACKEND_ENV, raising=False)
    monkeypatch.setattr(fp4_mla_config, "get_sm_version", lambda: sm)
    monkeypatch.setattr(fp4_mla_config, "_cutedsl_backend_available", lambda: available)

    assert fp4_mla_config._fp4_mla_attention_backend() == expected


def test_backend_env_override_wins(monkeypatch):
    monkeypatch.setenv(fp4_mla_config.FP4_MLA_ATTENTION_BACKEND_ENV, "CuteDSL")
    monkeypatch.setattr(fp4_mla_config, "get_sm_version", lambda: 100)
    monkeypatch.setattr(fp4_mla_config, "_cutedsl_backend_available", lambda: False)

    assert fp4_mla_config._fp4_mla_attention_backend() == "cutedsl"
