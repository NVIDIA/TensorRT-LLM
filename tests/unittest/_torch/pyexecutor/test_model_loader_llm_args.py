# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-specific LLM args are rejected for models that do not declare them."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.modeling_v2 import NoTarget
from tensorrt_llm._torch.models.modeling_qwen3_5 import Qwen3_5MoeForCausalLM, Qwen3_5MoeVLModel
from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor import model_loader
from tensorrt_llm._torch.pyexecutor.model_loader import _reject_unsupported_model_llm_args
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

__extra_import_path__ = ["../modeling_v2/models/kimi_k31"]

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "model_cls, supported",
    [
        (DecoderModelForCausalLM, False),
        (Qwen3NextForCausalLM, True),
        (Qwen3_5MoeForCausalLM, True),
        (Qwen3_5MoeVLModel, True),
    ],
)
@pytest.mark.parametrize("name", ["replicate_embedding", "mnnvl_allreduce_fp8_payload"])
def test_model_specific_llm_args(model_cls, supported, name):
    llm_args = SimpleNamespace(replicate_embedding=False, mnnvl_allreduce_fp8_payload=False)
    _reject_unsupported_model_llm_args(llm_args, model_cls)
    setattr(llm_args, name, True)
    with nullcontext() if supported else pytest.raises(ValueError, match=name):
        _reject_unsupported_model_llm_args(llm_args, model_cls)


class _ModelingV2Target(DecoderModelForCausalLM):
    """A modeling_v2 target that implements no model-specific LLM arg."""

    @classmethod
    def get_model_defaults(cls, llm_args):
        return {}


@pytest.mark.parametrize("name", [None, "replicate_embedding", "mnnvl_allreduce_fp8_payload"])
def test_modeling_v2_only_architecture_checks_its_target(monkeypatch, tmp_path, name):
    """An architecture without an engine model class takes its modeling_v2 target, which the loader
    checks as it checks an engine class: a set arg the target does not implement is rejected."""
    monkeypatch.setattr(
        model_loader.AutoModelForCausalLM, "_resolve_class", staticmethod(lambda config: None)
    )
    monkeypatch.setattr(model_loader, "get_registered_model_class", lambda arch: None)
    monkeypatch.setattr(
        model_loader, "modeling_v2_resolve", lambda config, llm_args: _ModelingV2Target
    )
    checkpoint_loader = MagicMock()
    config = checkpoint_loader.load_config.return_value
    config.pretrained_config.architectures = ["ModelingV2OnlyForCausalLM"]
    config.pretrained_config.hybrid_override_pattern = None
    args = TorchLlmArgs(model=str(tmp_path), **({name: True} if name else {}))
    if name is None:
        model_loader.ModelLoader.load_config_with_defaults(str(tmp_path), args, checkpoint_loader)
        return
    with pytest.raises(ValueError, match=f"{name} is not supported by _ModelingV2Target"):
        model_loader.ModelLoader.load_config_with_defaults(str(tmp_path), args, checkpoint_loader)


def test_modeling_v2_only_architecture_reports_route_refusal_in_auto(monkeypatch, tmp_path):
    from _k31_builders import HfConfigOnly, checkpoint_dir, on_gpu

    on_gpu(monkeypatch, (10, 0))
    checkpoint = checkpoint_dir(tmp_path)
    args = TorchLlmArgs(model=checkpoint)
    with pytest.raises(NoTarget) as refusal:
        model_loader.ModelLoader.load_config_with_defaults(checkpoint, args, HfConfigOnly())
    assert str(refusal.value) == (
        "'KimiK31ForConditionalGeneration' has no built-in fallback, and no modeling_v2 target "
        "matches this deployment: sm_100 is not validated yet"
    )
