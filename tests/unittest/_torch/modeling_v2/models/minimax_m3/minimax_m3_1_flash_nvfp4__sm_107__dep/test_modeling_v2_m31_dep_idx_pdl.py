# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_IDX_PDL's install and kill switch, CPU only.

On a built-in model with the horizontal producer's early PDL trigger, the install turns that trigger on and the decode
reader's pre-wait prefetch with it; the off switch leaves both off, and a built-in model without the trigger installs
nothing and says so. The built-in model is a stand-in with the trigger's two functions or without them.
"""

import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels import (
    nvfp4_index_k,
)

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        idx_pdl,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


def _builtin_with_trigger():
    state = {"on": False}
    return types.SimpleNamespace(
        set_idx_pdl_early_trigger=lambda on: state.__setitem__("on", bool(on)),
        idx_pdl_early_trigger=lambda: state["on"],
    )


@pytest.fixture(autouse=True)
def _builtin_settings(monkeypatch):
    """The reader's default prefetch and the switch on, restored after each test."""
    monkeypatch.setattr(nvfp4_index_k, "_DECODE_PREWAIT_PREFETCH", 0)
    monkeypatch.setattr(idx_pdl, "DEP_IDX_PDL", True)


def test_install_turns_on_the_trigger_and_the_prefetch(monkeypatch):
    builtin = _builtin_with_trigger()
    monkeypatch.setattr(idx_pdl, "_mm3", builtin)
    assert idx_pdl.install()
    assert builtin.idx_pdl_early_trigger()
    assert nvfp4_index_k.decode_prewait_prefetch() == idx_pdl.PREWAIT_PREFETCH_PAGES > 0
    assert idx_pdl.status(True) == f"on (reader prefetch {idx_pdl.PREWAIT_PREFETCH_PAGES} pages)"


def test_install_sets_nothing_when_off(monkeypatch):
    builtin = _builtin_with_trigger()
    monkeypatch.setattr(idx_pdl, "_mm3", builtin)
    monkeypatch.setattr(idx_pdl, "DEP_IDX_PDL", False)
    assert not idx_pdl.install()
    assert not builtin.idx_pdl_early_trigger()
    assert nvfp4_index_k.decode_prewait_prefetch() == 0
    assert idx_pdl.status(False) == "off (TRTLLM_M31_DISABLE_DEP_IDX_PDL=1)"


def test_a_builtin_model_without_the_trigger_installs_nothing(monkeypatch):
    monkeypatch.setattr(idx_pdl, "_mm3", types.SimpleNamespace())
    assert not idx_pdl.install()
    assert nvfp4_index_k.decode_prewait_prefetch() == 0
    assert idx_pdl.status(False).startswith(
        "not installed (the built-in model has no producer early trigger"
    )


@pytest.mark.parametrize("pages", (-1, 33))
def test_the_prefetch_page_count_is_bounded(pages):
    with pytest.raises(ValueError):
        nvfp4_index_k.set_decode_prewait_prefetch(pages)
