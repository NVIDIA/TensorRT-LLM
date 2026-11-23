# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HY4-Flash routing on CPU, with the target class stubbed in ``sys.modules``."""

import sys
from types import ModuleType

import pytest
import torch
from _hyv4_flash_config import config_dict  # noqa: E402

from tensorrt_llm._torch.configs import HYV4FlashConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import NoTarget, modeling_v2_resolve
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash import routing
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_SM103 = (10, 3)
_DUMMY_MODEL = "/tmp/dummy_model"
_TARGET_PACKAGE = "tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default"
_TARGET_MODULE = f"{_TARGET_PACKAGE}.modeling"

_MEMBERS = {
    "tep2": dict(world_size=2, tp_size=2, moe_tp_size=1, moe_ep_size=2),
    "tp4": dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
    "tep8": dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8),
}


@pytest.fixture(autouse=True)
def _on_sm103_with_a_stubbed_target(monkeypatch):
    package = ModuleType(_TARGET_PACKAGE)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, _TARGET_PACKAGE, package)
    module = ModuleType(_TARGET_MODULE)
    module.ModelingV2Hy4FlashFp8Sm103Default = type(
        "ModelingV2Hy4FlashFp8Sm103Default", (), {"__module__": _TARGET_MODULE}
    )
    monkeypatch.setitem(sys.modules, _TARGET_MODULE, module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM103)


def _model_config(spec_config=None, hidden_size=None, **mapping_kwargs) -> ModelConfig:
    checkpoint = config_dict()
    if hidden_size is not None:
        checkpoint["text_config"]["hidden_size"] = hidden_size
    return ModelConfig(
        pretrained_config=HYV4FlashConfig.from_dict(checkpoint),
        mapping=Mapping(**mapping_kwargs),
        spec_config=spec_config,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
    )


def _dflash(**kwargs) -> DFlashDecodingConfig:
    return DFlashDecodingConfig(decoding_type="DFlash", max_draft_len=5, **kwargs)


@pytest.mark.parametrize("dfly2", [False, True], ids=["no-spec", "dfly2"])
@pytest.mark.parametrize("member", list(_MEMBERS))
def test_admitted_layouts_route_to_the_target(member, dfly2) -> None:
    config = _model_config(_dflash(mask_token_id=120009) if dfly2 else None, **_MEMBERS[member])
    target = modeling_v2_resolve(config, TorchLlmArgs(model=_DUMMY_MODEL, modeling_v2="require"))
    assert target is sys.modules[_TARGET_MODULE].ModelingV2Hy4FlashFp8Sm103Default


@pytest.mark.parametrize(
    "config_kwargs, capability, missed",
    [
        (dict(hidden_size=4096, **_MEMBERS["tp4"]), _SM103, "config shape"),
        (dict(world_size=2, tp_size=2, moe_tp_size=2, moe_ep_size=1), _SM103, r"world_size=2\b"),
        (dict(_MEMBERS["tp4"]), (10, 7), r"compute capability \(10, 7\)"),
        (
            dict(spec_config=_dflash(speculative_model="/tmp/draft_model"), **_MEMBERS["tep4"]),
            _SM103,
            "separate draft checkpoint",
        ),
    ],
    ids=["other-shape", "tp2-moe-tp2", "sm107", "separate-draft"],
)
def test_rejected_configs_raise_no_target(config_kwargs, capability, missed) -> None:
    config = _model_config(**config_kwargs)
    with pytest.raises(NoTarget, match=missed):
        routing.route(config, TorchLlmArgs(model=_DUMMY_MODEL), capability)
