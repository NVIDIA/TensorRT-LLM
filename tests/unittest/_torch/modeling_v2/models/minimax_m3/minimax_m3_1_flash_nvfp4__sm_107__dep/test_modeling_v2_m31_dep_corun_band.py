# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_CORUN_BAND on stub layers and unbuilt MegaMoE backends.

The install gives each 8-way MegaMoE backend the target's co-run band where the environment it was built with sets none:
the default, the off switch, an explicit band, other expert-parallel sizes, and a backend no install reached (the
built-in model and the other targets). CPU only.
"""

import os
import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    corun_band,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_plan as plan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local.local_impl import (
    TrtllmCutedslMegaMoeLocalNvfp4Impl,
)

pytestmark = pytest.mark.cpu_only

BAND = plan.SHARED_CORUN_BAND_ENV
BAND_OFF = plan.SHARED_CORUN_BAND_DISABLE_ENV
LADDER = (32, 64, 128)
ROWS = (9, 64, 96, 128)


@pytest.fixture(autouse=True)
def _no_scope_env(monkeypatch):
    for name in (
        BAND,
        BAND_OFF,
        plan.SHARED_SERIAL_DISABLE_ENV,
        plan.SHARED_SERIAL_MAX_ROWS_ENV,
        plan.SHARED_SERIAL_MIN_EP_SIZE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def _backend(ep_size=8):
    """An unbuilt local MegaMoE backend with the state these decisions read, the knobs as they are now."""
    backend = TrtllmCutedslMegaMoeLocalNvfp4Impl.__new__(TrtllmCutedslMegaMoeLocalNvfp4Impl)
    backend._maxt_buckets = list(LADDER)
    backend.tactic_autotune = False
    backend.ep_size = ep_size
    backend._lever_environ = plan.lever_environ(os.environ)
    return backend


def _layer(backend):
    return types.SimpleNamespace(
        block_sparse_moe=types.SimpleNamespace(experts=types.SimpleNamespace(backend=backend))
    )


def _serial(backend):
    return [backend.shared_expert_serial(rows) for rows in ROWS]


def test_the_default_band():
    assert corun_band.DEFAULT_BAND == (32, 64) and corun_band.EP_SIZE == 8


def test_install_gives_the_8_way_backends_the_default_band():
    dep8, dep16, dep4 = _backend(8), _backend(16), _backend(4)
    layers = [_layer(dep8), _layer(dep16), _layer(dep4), _layer(object()), types.SimpleNamespace()]
    assert corun_band.install(layers) == 1
    assert dep8.shared_corun_band() == (32, 64)
    assert _serial(dep8) == [True, False, True, True]
    for backend in (dep16, dep4):
        assert backend.shared_corun_band() is None and all(_serial(backend))
    assert corun_band.install(layers) == 1


def test_the_band_reaches_the_shared_expert_placement():
    backend = _backend(8)
    corun_band.install([_layer(backend)])
    assert not backend.shared_expert_serial(64)
    assert backend.shared_expert_serial(128)


def test_an_explicit_band_stays_as_set(monkeypatch):
    monkeypatch.setenv(BAND, "64:128")
    backend = _backend(8)
    assert corun_band.install([_layer(backend)]) == 0
    assert backend.shared_corun_band() == (64, 128)
    assert _serial(backend) == [True, True, False, False]
    monkeypatch.setenv(BAND, "32:64")
    wide = _backend(16)
    assert corun_band.install([_layer(wide)]) == 0
    assert wide.shared_corun_band() == (32, 64)


def test_the_off_switch_keeps_every_bucket_serial(monkeypatch):
    monkeypatch.setenv(BAND_OFF, "1")
    backend = _backend(8)
    assert corun_band.install([_layer(backend)]) == 0
    assert backend.shared_corun_band() is None and all(_serial(backend))
    monkeypatch.setenv(BAND, "32:64")
    both = _backend(8)
    assert corun_band.install([_layer(both)]) == 0
    assert both.shared_corun_band() is None and all(_serial(both))


def test_a_backend_no_install_reached_keeps_every_bucket_serial():
    """The built-in model and the other targets build the same backend and never run the install."""
    backend = _backend(8)
    assert backend.shared_corun_band() is None and all(_serial(backend))
