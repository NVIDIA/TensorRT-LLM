# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA CuTe DSL backend selection and kernel DSL API usage."""

import ast
import inspect

import pytest

import tensorrt_llm._torch.attention.backends.fp4_mla.config as fp4_mla_config
import tensorrt_llm._torch.attention.backends.fp4_mla.fp4_mla_cutedsl_mufu16 as mufu16
import tensorrt_llm._torch.attention.backends.fp4_mla.fp4_mla_cutedsl_mufu16_fused_v_transpose as mufu16_fused_v_transpose  # noqa: E501

pytestmark = pytest.mark.cpu_only

# Scheduling-hint APIs absent from the public nvidia-cutlass-dsl release.
_NON_PUBLIC_SCHEDULE_HINTS = ("cfence", "warp_switch")


@pytest.fixture
def fresh_availability():
    fp4_mla_config._cutedsl_backend_available.cache_clear()
    yield
    fp4_mla_config._cutedsl_backend_available.cache_clear()


@pytest.mark.parametrize("kernel_module", [mufu16, mufu16_fused_v_transpose])
def test_kernels_do_not_use_non_public_schedule_hints(kernel_module):
    tree = ast.parse(inspect.getsource(kernel_module))
    hint_uses = [
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if (isinstance(node, ast.Attribute) and node.attr in _NON_PUBLIC_SCHEDULE_HINTS)
        or (isinstance(node, ast.Name) and node.id in _NON_PUBLIC_SCHEDULE_HINTS)
    ]

    assert hint_uses == []


def test_cutedsl_availability_needs_only_the_kernel_imports(monkeypatch, fresh_availability):
    queried = []

    def find_spec(name):
        queried.append(name)
        return object()

    monkeypatch.setattr(fp4_mla_config.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(fp4_mla_config, "_cutedsl_dsl_supports_fp4_mla_kernels", lambda: True)

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
    monkeypatch.setattr(fp4_mla_config, "_cutedsl_dsl_supports_fp4_mla_kernels", lambda: True)

    assert not fp4_mla_config._cutedsl_backend_available()


def test_cutedsl_availability_requires_the_kernel_dsl_api(monkeypatch, fresh_availability):
    monkeypatch.setattr(fp4_mla_config.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(fp4_mla_config, "_cutedsl_dsl_supports_fp4_mla_kernels", lambda: False)

    assert not fp4_mla_config._cutedsl_backend_available()


@pytest.mark.parametrize("supported", [False, True])
def test_kernel_dsl_api_probe_reads_the_descriptor_signature(monkeypatch, supported):
    import cutlass.experimental.primitives as primitives

    class SmemDesc:
        if supported:

            @staticmethod
            def build(start_address, k_segment_offset=0):
                pass

        else:

            @staticmethod
            def build(start_address):
                pass

    monkeypatch.setattr(primitives, "Tcgen05SmemDesc", SmemDesc)

    assert fp4_mla_config._cutedsl_dsl_supports_fp4_mla_kernels() is supported


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
