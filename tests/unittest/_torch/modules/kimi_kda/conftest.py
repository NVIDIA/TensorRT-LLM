# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the KDA unit tests."""

import pytest


@pytest.fixture
def cute_compiles(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Type names of the objects passed to every cute.compile call made while the fixture is active."""
    import cutlass.cute as cute

    calls: list[str] = []
    real_compile = cute.compile

    def counting_compile(*args, **kwargs):
        calls.append(type(args[0]).__name__)
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", counting_compile)
    return calls


@pytest.fixture(autouse=True)
def _flashinfer_prefill_switch_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every KDA test with ``TLLM_KDA_FLASHINFER_PREFILL`` unset unless the test sets it.

    The in-tree prefill tests must not follow a value exported in the shell.
    """
    monkeypatch.delenv("TLLM_KDA_FLASHINFER_PREFILL", raising=False)
