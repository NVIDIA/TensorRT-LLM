# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The routing of the ULL disaggregated recipes (``examples/kimi_k31/ull_disagg``), on CPU.

Each recipe (C1, C4 and C8; ``*_e2e`` the performance case, ``*_gsm`` the accuracy case) runs a
TEP8 context server and a Helix CP16 generation server. Their worker configs, read from the yamls,
land on the targets the recipes name: the CTX on ``kimi_k31_pro_nvfp4__sm_107__tep8``, which takes
only a text-only deployment with FP8 drafter history (``disable_mm_encoder`` and
``speculative_config.context_kv_dtype: fp8``), and the GEN on the ULL route
(``kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1``), which its layout reaches before that rule.
The same CTX without ``disable_mm_encoder`` starts on the configurable
``kimi_k31_pro_nvfp4__sm_10x__default`` target with no error, which is why the yamls set it.

The targets are stand-ins and the checkpoint is ``test_routing_sm107.py``'s 93-layer export;
each config is routed on both loads, as the engine routes it. The yamls' drafter path is a
placeholder: the drafter is ``test_routing_sm107.py``'s DFlash2 checkpoint at the yaml's draft
length, with the yaml's other drafter fields.
"""

import sys
import types
from pathlib import Path

import pytest
import test_routing_sm107 as _sm107
import yaml

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.common import (
    rubin_ull_features as ull,
)
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

pytestmark = pytest.mark.cpu_only

_EXAMPLES = Path(__file__).resolve().parents[6] / "examples" / "kimi_k31" / "ull_disagg"
_RECIPES = ("C1_e2e", "C1_gsm", "C4_e2e", "C4_gsm", "C8_e2e")
_ULL = (
    "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1",
    "ModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1",
)


@pytest.fixture(autouse=True)
def ull_stand_in(monkeypatch):
    """Route as on Rubin with stand-ins for the sm_107 targets and the ULL target (importing one
    needs its GPU kernels); the profile the ULL route selects is recorded instead of set."""
    _sm107._on_gpu(monkeypatch, (10, 7))
    _sm107._stand_in_targets(monkeypatch)
    module = types.ModuleType(f"{_sm107._FAMILY}.{_ULL[0]}.modeling")
    setattr(module, _ULL[1], type(_ULL[1], (), {}))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    selected = []
    monkeypatch.setattr(ull, "select", selected.append)
    return selected


class _UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that refuses a mapping with a repeated key (``safe_load`` keeps the last
    one, so a repeated key hides an edit of the other)."""


def _unique_mapping(loader, node, deep=False):
    keys = [loader.construct_object(key, deep=deep) for key, _ in node.value]
    repeated = sorted({str(key) for key in keys if keys.count(key) > 1})
    if repeated:
        raise yaml.constructor.ConstructorError(
            None, None, f"repeated keys {repeated}", node.start_mark
        )
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _worker(recipe: str, role: str) -> dict:
    """``worker_config.<role>`` of the recipe's yaml."""
    config = yaml.safe_load((_EXAMPLES / f"{recipe}.yaml").read_text())
    return config["worker_config"][role]


def _route(monkeypatch, tmp_path, worker: dict, role: str) -> str:
    """The target class name a ``role`` server of the yaml worker config ``worker`` resolves to, on
    both loads of the model config."""
    routing_role = {"ctx": "context", "gen": "generation"}[role]
    _sm107._as_role(monkeypatch, routing_role)
    options = dict(worker)
    draft = dict(options.pop("speculative_config"))
    assert draft.pop("decoding_type") == "DFlash"
    draft.pop("speculative_model")
    spec_config = _sm107._dflash2(tmp_path, draft.pop("max_draft_len"), **draft)
    checkpoint = _sm107._checkpoint_dir(tmp_path)
    args = TorchLlmArgs(model=checkpoint, **options)
    names = {
        modeling_v2_resolve(config, args).__name__
        for config in _sm107._loaded_configs(checkpoint, args, spec_config)
    }
    assert len(names) == 1, names
    return names.pop()


@pytest.mark.parametrize("recipe", _RECIPES)
def test_the_recipes_ask_for_the_specialized_ctx_target(recipe):
    ctx = _worker(recipe, "ctx")
    assert ctx["disable_mm_encoder"] is True
    assert ctx["speculative_config"]["context_kv_dtype"] == "fp8"


@pytest.mark.parametrize("recipe", sorted(path.stem for path in _EXAMPLES.glob("*.yaml")))
def test_the_recipes_set_each_key_once(recipe):
    yaml.load((_EXAMPLES / f"{recipe}.yaml").read_text(), Loader=_UniqueKeyLoader)


@pytest.mark.parametrize("recipe", _RECIPES)
def test_the_ctx_routes_to_tep8(monkeypatch, tmp_path, recipe):
    name = _route(monkeypatch, tmp_path, _worker(recipe, "ctx"), "ctx")
    assert name == _sm107._SPECIALIZED["tep8"][1]


@pytest.mark.parametrize("recipe", _RECIPES)
def test_the_ctx_without_disable_mm_encoder_routes_to_the_default_target(
    monkeypatch, tmp_path, recipe
):
    ctx = _worker(recipe, "ctx")
    del ctx["disable_mm_encoder"]
    assert _route(monkeypatch, tmp_path, ctx, "ctx") == _sm107._DEFAULT[1]


@pytest.mark.parametrize("recipe", _RECIPES)
def test_the_gen_routes_to_the_ull_target(monkeypatch, tmp_path, ull_stand_in, recipe):
    name = _route(monkeypatch, tmp_path, _worker(recipe, "gen"), "gen")
    assert name == _ULL[1]
    assert [(key.topology, key.role) for key in ull_stand_in] == [("ull16", "generation")] * 2
    # The ULL layout is routed before the specialized targets' text-only rule.
    assert "disable_mm_encoder" not in _worker(recipe, "gen")
