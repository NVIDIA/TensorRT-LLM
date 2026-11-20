# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA CuTe DSL backend selection."""

import pytest

import tensorrt_llm._torch.attention.backends.fp4_mla.config as fp4_mla_config

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def fresh_availability():
    fp4_mla_config._cutedsl_backend_available.cache_clear()
    yield
    fp4_mla_config._cutedsl_backend_available.cache_clear()


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
