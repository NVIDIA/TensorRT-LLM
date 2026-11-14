# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the KDA unit tests."""

import pytest


@pytest.fixture
def cute_compiles(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Type names of the kernel objects that `cute.compile` gets while the fixture is active.

    The returned list grows by one entry per compilation.
    """
    import cutlass.cute as cute

    calls: list[str] = []
    real_compile = cute.compile

    def counting_compile(*args, **kwargs):
        calls.append(type(args[0]).__name__)
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", counting_compile)
    return calls
