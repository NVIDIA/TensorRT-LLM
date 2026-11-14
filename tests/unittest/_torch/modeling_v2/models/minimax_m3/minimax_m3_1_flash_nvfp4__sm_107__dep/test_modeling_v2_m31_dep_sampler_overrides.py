# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DEP target's sampler-override install (``modeling._install_sampler_overrides``) against stand-in registries.

Each of the eight items installs its function through its registry's setters only when its switch is on and its op is
built; an item whose registry this runtime does not have installs nothing and logs that instead of its engagement.
CPU only: the registry setters, the ops and the engagement lines are stand-ins.
"""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        gumbel_draft,
        logits_widen,
        markov_glue,
        modeling,
        pad_fill,
        rej_mc_fi,
        sfp_mc_fi,
        softmax_mc,
        topp_det,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

#: item -> (module, its switch, the module functions that say whether its op is built, what each setter installs).
ITEMS = {
    "SOFTMAX_MC": (softmax_mc, "SOFTMAX_MC", ("mc_op",), ("softmax_mc",)),
    "TOPP_DET": (topp_det, "TOPP_DET", ("topp_det_op",), ("top_p_renorm_det",)),
    "DEP_PAD_FILL": (pad_fill, "DEP_PAD_FILL", (), ("pad_rows_fill",)),
    "DEP_REJ_MC_FI": (
        rej_mc_fi,
        "DEP_REJ_MC_FI",
        ("mc_fi_op", "mc_fi_slots_op"),
        ("chain_spec_mc_fi", "chain_spec_mc_fi_slots"),
    ),
    "DEP_SFP_MC_FI": (
        sfp_mc_fi,
        "DEP_SFP_MC_FI",
        ("sfp_mc_fi_op",),
        ("sampling_from_probs_mc_fi",),
    ),
    "DEP_MARKOV_GLUE": (markov_glue, "DEP_MARKOV_GLUE", (), (None,)),
    "DEP_GUMBEL_DRAFT": (gumbel_draft, "DEP_GUMBEL_DRAFT", ("block_sampler",), ("draw_block",)),
    "DEP_LOGITS_WIDEN": (logits_widen, "DEP_LOGITS_WIDEN", (), ("widen",)),
}


@pytest.fixture
def registry(monkeypatch):
    """Every item's registry as recording setters, every switch off and every op built; returns (installed, engaged):
    the function each setter received, by ``<item>[<setter index>]``, and the items that logged their engagement."""
    installed, engaged = {}, []

    def setter(key):
        return lambda fn: installed.__setitem__(key, fn)

    table = {
        item: tuple(setter(f"{item}[{i}]") for i in range(len(fns)))
        for item, (_, _, _, fns) in ITEMS.items()
    }
    monkeypatch.setattr(modeling, "_OVERRIDE_SETTERS", table)
    for item, (module, switch, built, _) in ITEMS.items():
        monkeypatch.setattr(module, switch, False)
        for name in built:
            monkeypatch.setattr(module, name, lambda: object())
        monkeypatch.setattr(module, "log_engagement", lambda item=item: engaged.append(item))
    return installed, engaged


def _expected(item):
    module, _, _, fns = ITEMS[item]
    return {f"{item}[{i}]": getattr(module, fn) for i, fn in enumerate(fns) if fn is not None}


@pytest.mark.parametrize("item", list(ITEMS))
def test_an_item_on_and_built_installs_through_its_setters(registry, monkeypatch, item):
    installed, engaged = registry
    module, switch, _, _ = ITEMS[item]
    monkeypatch.setattr(module, switch, True)
    modeling._install_sampler_overrides()
    if item == "DEP_MARKOV_GLUE":
        assert list(installed) == ["DEP_MARKOV_GLUE[0]"]
        assert isinstance(installed["DEP_MARKOV_GLUE[0]"], markov_glue.MarkovGlue)
    else:
        assert installed == _expected(item)
    assert sorted(engaged) == sorted(ITEMS)


@pytest.mark.parametrize("item", [item for item, spec in ITEMS.items() if spec[2]])
def test_an_item_whose_op_is_not_built_installs_nothing(registry, monkeypatch, item):
    installed, _ = registry
    module, switch, built, _ = ITEMS[item]
    monkeypatch.setattr(module, switch, True)
    monkeypatch.setattr(module, built[-1], lambda: None)
    modeling._install_sampler_overrides()
    assert installed == {}


def test_every_item_off_installs_nothing(registry):
    installed, engaged = registry
    modeling._install_sampler_overrides()
    assert installed == {} and sorted(engaged) == sorted(ITEMS)


def test_an_item_without_its_registry_installs_nothing_and_says_so(registry, monkeypatch):
    installed, engaged = registry
    lines = []
    monkeypatch.setattr(modeling.logger, "info", lines.append)
    missing = {"DEP_PAD_FILL", "DEP_REJ_MC_FI"}
    table = dict(modeling._OVERRIDE_SETTERS)
    table["DEP_PAD_FILL"] = (None,)
    table["DEP_REJ_MC_FI"] = (table["DEP_REJ_MC_FI"][0], None)
    monkeypatch.setattr(modeling, "_OVERRIDE_SETTERS", table)
    for module, switch, _, _ in ITEMS.values():
        monkeypatch.setattr(module, switch, True)
    modeling._install_sampler_overrides()
    assert not any(key.startswith(tuple(missing)) for key in installed)
    assert sorted(engaged) == sorted(set(ITEMS) - missing)
    for item in missing:
        assert any(
            line.startswith(f"[M3.1 DEP] {item}: not taken: its override registry")
            for line in lines
        )
