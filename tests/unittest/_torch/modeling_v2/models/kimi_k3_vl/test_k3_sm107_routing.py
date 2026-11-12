# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``route`` picks the SM107 Kimi K3 target of each recipe's role, and declines every other layout.

The layouts come from the perf-sanity deployments of the SM107 targets
(``tests/scripts/perf-sanity/disaggregated/vr200_kimi-k3-nvfp4-dspark_gsm8k_*.yaml``): each
worker's parallel fields build a real ``Mapping`` as ``ParallelConfig.to_mapping`` does, and the
fields ``route`` reads off ``llm_args`` (MoE backend, DWDP, KV cache dtype) come from the same
worker config. No model is loaded.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import routing
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_REPO = Path(__file__).resolve().parents[6]
_RECIPES = _REPO / "tests/scripts/perf-sanity/disaggregated"
_SM107 = (10, 7)
_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"

# Recipe file stem -> (context target, generation target).
_RECIPE_TARGETS = {
    "vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx16_dwdp16_gen1_dep16_bs16_dspark5": (
        "kimi_k3_nvfp4__sm_107__tp1",
        "kimi_k3_nvfp4__sm_107__dep16",
    ),
    "vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx16_dwdp16_gen1_helix8_bs20_dspark5": (
        "kimi_k3_nvfp4__sm_107__tp1",
        "kimi_k3_nvfp4__sm_107__helix8_moetp2ep4",
    ),
    "vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx16_dwdp16_gen1_helix8_bs8_dspark7": (
        "kimi_k3_nvfp4__sm_107__tp1",
        "kimi_k3_nvfp4__sm_107__helix8_moeep8",
    ),
}

_TARGET_CLASSES = {
    "kimi_k3_nvfp4__sm_107__tp1": "ModelingV2KimiK3Nvfp4Sm107Tp1",
    "kimi_k3_nvfp4__sm_107__helix8_moeep8": "ModelingV2KimiK3Nvfp4Sm107Helix8Moeep8",
    "kimi_k3_nvfp4__sm_107__helix8_moetp2ep4": "ModelingV2KimiK3Nvfp4Sm107Helix8Moetp2ep4",
    "kimi_k3_nvfp4__sm_107__dep16": "ModelingV2KimiK3Nvfp4Sm107Dep16",
}


def _text_config():
    return SimpleNamespace(
        **dict(zip(routing._SM107_SHAPE_FIELDS, routing._SM107_SHAPE)),
    )


def _model_config(mapping: Mapping, *, spec: str = "DSPARK", expert_algo: str = "NVFP4"):
    is_dspark = spec == "DSPARK"
    spec_config = (
        None
        if spec is None
        else SimpleNamespace(
            spec_dec_mode=SimpleNamespace(name=spec, is_dspark=lambda: is_dspark),
        )
    )
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(
            text_config=_text_config(),
            quantization_config={
                "quantized_layers": {
                    "language_model.model.layers.1.mlp.experts": {"quant_algo": expert_algo}
                }
            },
        ),
        mapping=mapping,
        spec_config=spec_config,
        quant_config=SimpleNamespace(kv_cache_quant_algo=None),
    )


def _worker(stem: str, role: str) -> dict:
    return yaml.safe_load((_RECIPES / f"{stem}.yaml").read_text())["worker_config"][role]


def _mapping(worker: dict) -> Mapping:
    """``ParallelConfig.to_mapping`` of a worker config (rank 0)."""
    tp = worker.get("tensor_parallel_size", 1)
    pp = worker.get("pipeline_parallel_size", 1)
    cp = worker.get("context_parallel_size", 1)
    cp_config = dict(worker.get("cp_config") or {}) if cp > 1 else {}
    return Mapping(
        world_size=tp * pp * cp,
        rank=0,
        tp_size=tp,
        pp_size=pp,
        cp_size=cp,
        cp_config=cp_config,
        enable_attention_dp=worker.get("enable_attention_dp", False),
        moe_tp_size=worker.get("moe_tensor_parallel_size", -1),
        moe_ep_size=worker.get("moe_expert_parallel_size", -1),
    )


def _llm_args(worker: dict, **overrides):
    fields = {
        "moe_config": SimpleNamespace(backend=worker["moe_config"]["backend"]),
        "dwdp_config": worker.get("dwdp_config"),
        "kv_cache_config": SimpleNamespace(dtype=worker["kv_cache_config"]["dtype"]),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _route_name(model_config, llm_args) -> str:
    return routing._check_sm107(model_config, llm_args)


@pytest.mark.parametrize("stem", sorted(_RECIPE_TARGETS))
@pytest.mark.parametrize("role", ["ctx", "gen"])
def test_each_recipe_role_routes_to_its_target(stem, role):
    worker = _worker(stem, role)
    expected = _RECIPE_TARGETS[stem][0 if role == "ctx" else 1]
    assert _route_name(_model_config(_mapping(worker)), _llm_args(worker)) == expected


def test_context_servers_of_a_dwdp12_recipe_route_to_tp1():
    """A recipe with 12 context servers (DWDP12): the layout of each is still one rank."""
    worker = _worker(
        "vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx16_dwdp16_gen1_helix8_bs20_dspark5", "ctx"
    )
    worker = {**worker, "dwdp_config": {**worker["dwdp_config"], "dwdp_size": 12}}
    assert (
        _route_name(_model_config(_mapping(worker)), _llm_args(worker))
        == "kimi_k3_nvfp4__sm_107__tp1"
    )


def _helix8(moe_tp: int, moe_ep: int, **cp_config) -> Mapping:
    return Mapping(
        world_size=8,
        rank=0,
        tp_size=1,
        cp_size=8,
        cp_config={"cp_type": "HELIX", "use_nccl_for_alltoall": False, **cp_config},
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
    )


_FC12 = SimpleNamespace(
    moe_config=SimpleNamespace(backend="CUTEDSL_FC12"),
    dwdp_config=None,
    kv_cache_config=SimpleNamespace(dtype="nvfp4"),
)


@pytest.mark.parametrize(
    "case, model_config, llm_args, says",
    [
        (
            "helix8 moe 4x2",
            lambda: _model_config(_helix8(4, 2)),
            _FC12,
            "moe_tp=4 moe_ep=2",
        ),
        (
            "helix8 with NCCL all-to-all",
            lambda: _model_config(_helix8(1, 8, use_nccl_for_alltoall=True)),
            _FC12,
            "cp_nccl_alltoall=True",
        ),
        (
            "helix8 on CUTEDSL",
            lambda: _model_config(_helix8(2, 4)),
            SimpleNamespace(**{**vars(_FC12), "moe_config": SimpleNamespace(backend="CUTEDSL")}),
            "moe_backend=CUTEDSL",
        ),
        (
            "helix16 moe 4x4 (ULL)",
            lambda: _model_config(
                Mapping(
                    world_size=16,
                    rank=0,
                    tp_size=1,
                    cp_size=16,
                    cp_config={"cp_type": "HELIX", "use_nccl_for_alltoall": False},
                    moe_tp_size=4,
                    moe_ep_size=4,
                )
            ),
            _FC12,
            "cp=16",
        ),
        (
            "attention DP over 8 ranks",
            lambda: _model_config(
                Mapping(world_size=8, rank=0, tp_size=8, enable_attention_dp=True, moe_ep_size=8)
            ),
            SimpleNamespace(
                **{**vars(_FC12), "moe_config": SimpleNamespace(backend="MEGAMOE_CUTEDSL")}
            ),
            "world_size=8",
        ),
        (
            "one rank without DWDP",
            lambda: _model_config(Mapping(world_size=1, rank=0, tp_size=1)),
            SimpleNamespace(**{**vars(_FC12), "moe_config": SimpleNamespace(backend="CUTEDSL")}),
            "dwdp=False",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_other_layouts_get_no_target(case, model_config, llm_args, says):
    with pytest.raises(NoTarget) as error:
        _route_name(model_config(), llm_args)
    message = str(error.value)
    assert says in message
    for target in _TARGET_CLASSES:
        assert target in message


def test_fp8_kv_no_spec_and_non_nvfp4_experts_get_no_target():
    worker = _worker("vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx16_dwdp16_gen1_dep16_bs16_dspark5", "gen")
    mapping = _mapping(worker)
    with pytest.raises(NoTarget, match="NVFP4 KV cache"):
        _route_name(
            _model_config(mapping),
            _llm_args(worker, kv_cache_config=SimpleNamespace(dtype="fp8")),
        )
    with pytest.raises(NoTarget, match="DSpark"):
        _route_name(_model_config(mapping, spec=None), _llm_args(worker))
    with pytest.raises(NoTarget, match="DSpark"):
        _route_name(_model_config(mapping, spec="MTP"), _llm_args(worker))
    with pytest.raises(NoTarget, match="NVFP4 routed experts"):
        _route_name(_model_config(mapping, expert_algo="FP8"), _llm_args(worker))


def test_route_returns_the_target_class_only_on_sm107():
    worker = _worker("vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx16_dwdp16_gen1_helix8_bs8_dspark7", "gen")
    model_config, llm_args = _model_config(_mapping(worker)), _llm_args(worker)
    cls = routing.route(model_config, llm_args, _SM107)
    assert cls.__name__ == _TARGET_CLASSES["kimi_k3_nvfp4__sm_107__helix8_moeep8"]
    assert cls.__module__ == f"{_FAMILY}.kimi_k3_nvfp4__sm_107__helix8_moeep8.modeling"
    with pytest.raises(NoTarget):
        routing.route(model_config, llm_args, (10, 3))


def test_pdl_off_gets_no_target(monkeypatch):
    """The targets' kernels launch with PDL as a constant, so PDL turned off declines them, read
    from ``env_overrides`` first (what every rank applies), then the environment."""
    worker = _worker("vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx16_dwdp16_gen1_dep16_bs16_dspark5", "gen")
    model_config = _model_config(_mapping(worker))
    monkeypatch.delenv("TRTLLM_ENABLE_PDL", raising=False)
    assert _route_name(model_config, _llm_args(worker)) == "kimi_k3_nvfp4__sm_107__dep16"
    with pytest.raises(NoTarget, match="TRTLLM_ENABLE_PDL='0'"):
        _route_name(model_config, _llm_args(worker, env_overrides={"TRTLLM_ENABLE_PDL": "0"}))
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    with pytest.raises(NoTarget, match="PDL"):
        _route_name(model_config, _llm_args(worker))
    assert (
        _route_name(model_config, _llm_args(worker, env_overrides={"TRTLLM_ENABLE_PDL": "1"}))
        == "kimi_k3_nvfp4__sm_107__dep16"
    )
