# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_IDXK_PRODUCER_NVFP4's install and kill switch, CPU only.

On a built-in model with the horizontal producer's NVFP4 index-K mode, the install switches the producer to writing
NVFP4 index-K pages itself, owned by this target; the off switch leaves the sink + pack, and a built-in model without
the mode installs nothing and says so. The built-in model is a stand-in with the mode's two functions or without them.
"""

import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        idxk_producer,
        m31_switches,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

IMPORTED_DEP_IDXK_PRODUCER_NVFP4 = idxk_producer.DEP_IDXK_PRODUCER_NVFP4


def _builtin_with_the_mode():
    state = {"on": False, "owner": None}

    def set_mode(on, owner):
        state.update(on=bool(on), owner=owner)

    return types.SimpleNamespace(
        set_index_k_producer_nvfp4=set_mode, index_k_producer_nvfp4=lambda: state["on"], state=state
    )


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch):
    monkeypatch.setattr(idxk_producer, "DEP_IDXK_PRODUCER_NVFP4", True)


def test_install_switches_the_producer_to_nvfp4_pages(monkeypatch):
    builtin = _builtin_with_the_mode()
    monkeypatch.setattr(idxk_producer, "_mm3", builtin)
    assert idxk_producer.install()
    assert builtin.index_k_producer_nvfp4() and builtin.state["owner"] == "M3.1 DEP"
    assert idxk_producer.install()
    assert idxk_producer.status(True) == "on"


def test_install_sets_nothing_when_off(monkeypatch):
    builtin = _builtin_with_the_mode()
    monkeypatch.setattr(idxk_producer, "_mm3", builtin)
    monkeypatch.setattr(idxk_producer, "DEP_IDXK_PRODUCER_NVFP4", False)
    assert not idxk_producer.install()
    assert not builtin.index_k_producer_nvfp4()
    assert idxk_producer.status(False) == "off (TRTLLM_M31_DISABLE_DEP_IDXK_PRODUCER_NVFP4=1)"


def test_a_builtin_model_without_the_mode_installs_nothing(monkeypatch):
    monkeypatch.setattr(idxk_producer, "_mm3", types.SimpleNamespace())
    assert not idxk_producer.install()
    assert idxk_producer.status(False).startswith(
        "not installed (the built-in model has no NVFP4 index-K producer mode"
    )


def test_the_switch_is_the_dep_idxk_producer_nvfp4_item():
    assert IMPORTED_DEP_IDXK_PRODUCER_NVFP4 == (
        not m31_switches.m31_disabled("DEP_IDXK_PRODUCER_NVFP4")
    )
