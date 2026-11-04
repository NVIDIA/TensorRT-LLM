# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral-Large-4 native paths leave the other Mistral-native formats as they were.

`mistral` and `mistral_large_3` checkpoints share the params.json adaptation and the HF config
lookup with `mistral_large_4`. Only Mistral-Large-4 loads its config from params.json for the
LLM API. The other formats keep the vLLM mapping of a native `quantization` block.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import (
    _remap_mistral_quantization_args,
)

pytestmark = pytest.mark.cpu_only


def _fp8(**quantization):
    return {"quantization": {"qformat_weight": "fp8_e4m3", **quantization}}


_BLOCK_FP8 = dict(
    qscheme_act="BLOCK",
    qscheme_weight="BLOCK",
    qdynamic_act=True,
    block_size_out=128,
    block_size_in=128,
)


@pytest.mark.parametrize("qscheme_act", ["TENSOR", None])
def test_other_formats_keep_the_vllm_quantization_mapping(qscheme_act):
    assert _remap_mistral_quantization_args(_fp8(qscheme_act=qscheme_act)) == {
        "quantization_config": {"quant_method": "fp8", "activation_scheme": "static"}
    }
    with pytest.raises(AssertionError):
        _remap_mistral_quantization_args(_fp8(**_BLOCK_FP8))


@pytest.mark.parametrize(
    "checkpoint_format", ["mistral_large_4", "mistral", "mistral_large_3", None]
)
def test_hf_config_lookup_names_only_the_mistral_large_4_format(monkeypatch, checkpoint_format):
    from tensorrt_llm.llmapi import llm as llm_module

    calls = []
    monkeypatch.setattr(
        llm_module.ModelLoader,
        "load_hf_model_config",
        staticmethod(lambda model, **kwargs: calls.append(kwargs)),
    )
    args = SimpleNamespace(model="m", trust_remote_code=False, checkpoint_format=checkpoint_format)
    llm_module.BaseLLM._try_load_hf_model_config(SimpleNamespace(args=args))
    expected = {"trust_remote_code": False}
    if checkpoint_format == "mistral_large_4":
        expected["checkpoint_format"] = "mistral_large_4"
    assert calls == [expected]
