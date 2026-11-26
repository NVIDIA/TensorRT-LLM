# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 test fixtures shared across modules."""

import pytest


@pytest.fixture
def swa_fp8_storage_exponent(monkeypatch: pytest.MonkeyPatch):
    """Return a setter of this process's swa_fp8 storage-scale exponent ``e`` for one test.

    swa_fp8 rows then hold E4M3(x / 2**e). The resolved exponent is cached per process, so
    the cache is cleared when it is set and again before the environment is restored.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
        SWA_FP8_AMAX_ENV,
        swa_fp8_scale_exponent,
    )

    def set_exponent(exponent: int) -> None:
        monkeypatch.setenv(SWA_FP8_AMAX_ENV, repr(448.0 * 2.0**exponent))
        swa_fp8_scale_exponent.cache_clear()
        assert swa_fp8_scale_exponent() == exponent

    yield set_exponent
    swa_fp8_scale_exponent.cache_clear()


@pytest.fixture
def q_fp8_exponent(monkeypatch: pytest.MonkeyPatch):
    """Return a setter of this process's PrimTS Q scale exponent ``e`` for one test.

    PrimTS then reads E4M3(q / 2**e). As for ``swa_fp8_storage_exponent``, the cached
    exponent is cleared when it is set and again before the environment is restored.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import Q_FP8_AMAX_ENV
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
        q_fp8_scale_exponent as resolve,
    )

    def set_exponent(exponent: int) -> None:
        monkeypatch.setenv(Q_FP8_AMAX_ENV, repr(448.0 * 2.0**exponent))
        resolve.cache_clear()
        assert resolve() == exponent

    yield set_exponent
    resolve.cache_clear()
