# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The routing of the layer-wise benchmark's Kimi K3.1 engine config, on CPU.

``examples/layer_wise_benchmarks/config_gen_k31_engine.yaml`` builds the first layers of the
tep8 serving recipe through the engine backend, an aggregated server (the generation role). Its
``llm_args``, over the arguments ``run.py`` adds (the cache dtypes, the graph batch sizes), land
on ``kimi_k31_pro_nvfp4__sm_107__tep8``, which takes only a text-only deployment with FP8 drafter
history (``disable_mm_encoder`` and ``speculative_config.context_kv_dtype: fp8``). Without either
key routing tries the configurable ``kimi_k31_pro_nvfp4__sm_10x__default`` target, which does not
run the config's MoE backend (CUTEDSL_FC12), and the run stops with NoTarget.

The targets are stand-ins and the checkpoint is ``test_routing_sm107.py``'s 93-layer export;
the config is routed on both loads, as the engine routes it. The yaml's drafter path is a
placeholder: the drafter is ``test_routing_sm107.py``'s DFlash2 checkpoint at the yaml's draft
length.
"""

import copy
from pathlib import Path

import pytest
import test_routing_sm107 as _sm107
import yaml

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.routing import DISAGG_ROLE_ENV
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

pytestmark = pytest.mark.cpu_only

_CONFIG = (
    Path(__file__).resolve().parents[6]
    / "examples"
    / "layer_wise_benchmarks"
    / "config_gen_k31_engine.yaml"
)


@pytest.fixture(autouse=True)
def rubin(monkeypatch):
    """Route as on Rubin with stand-ins for the sm_107 targets, as an aggregated server (no
    server role in the environment)."""
    _sm107._on_gpu(monkeypatch, (10, 7))
    _sm107._stand_in_targets(monkeypatch)
    monkeypatch.delenv(DISAGG_ROLE_ENV, raising=False)


def _merge(base: dict, override: dict) -> dict:
    """``base`` with ``override`` applied, nested dicts merged key by key (``run.py``'s
    ``merge_llm_args``)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _llm_args(config: dict) -> dict:
    """The engine's arguments: the yaml's ``llm_args`` over those ``run.py`` builds from the
    yaml's options (the model and the drafter aside)."""
    engine = {
        "max_seq_len": config["max_seq_len"],
        "kv_cache_config": {
            "dtype": config["kv_cache_dtype"],
            "mamba_ssm_cache_dtype": config["mamba_ssm_cache_dtype"],
            "enable_block_reuse": False,
        },
        "cuda_graph_config": {"batch_sizes": sorted(set(config["batch_size"]))},
    }
    return _merge(engine, config["llm_args"])


def _route(tmp_path, options: dict) -> str:
    """The target class name the engine arguments ``options`` resolve to, on both loads of the
    model config."""
    options = copy.deepcopy(options)
    draft = options.pop("speculative_config")
    assert draft.pop("decoding_type") == "DFlash"
    draft.pop("speculative_model")
    # The helper's drafter history defaults to FP8; route the config's own value (or none).
    draft.setdefault("context_kv_dtype", None)
    spec_config = _sm107._dflash2(tmp_path, draft.pop("max_draft_len"), **draft)
    checkpoint = _sm107._checkpoint_dir(tmp_path)
    args = TorchLlmArgs(model=checkpoint, **options)
    names = {
        modeling_v2_resolve(config, args).__name__
        for config in _sm107._loaded_configs(checkpoint, args, spec_config)
    }
    assert len(names) == 1, names
    return names.pop()


def test_the_config_asks_for_the_specialized_target():
    llm_args = yaml.safe_load(_CONFIG.read_text())["llm_args"]
    assert llm_args["disable_mm_encoder"] is True
    assert llm_args["speculative_config"]["context_kv_dtype"] == "fp8"


def test_the_config_routes_to_tep8(tmp_path):
    options = _llm_args(yaml.safe_load(_CONFIG.read_text()))
    assert _route(tmp_path, options) == _sm107._SPECIALIZED["tep8"][1]


@pytest.mark.parametrize(
    "drop",
    [("disable_mm_encoder",), ("speculative_config", "context_kv_dtype")],
    ids=["no_disable_mm_encoder", "no_context_kv_dtype"],
)
def test_without_a_routing_key_no_target_takes_the_config(tmp_path, drop):
    options = _llm_args(yaml.safe_load(_CONFIG.read_text()))
    *parents, key = drop
    node = options
    for parent in parents:
        node = node[parent]
    del node[key]
    # modeling_v2: require turns the NoTarget into a ValueError that names both refusals.
    with pytest.raises(ValueError, match=r"specialized targets .*configurable default"):
        _route(tmp_path, options)
