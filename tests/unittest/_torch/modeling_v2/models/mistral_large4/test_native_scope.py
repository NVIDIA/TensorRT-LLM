# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral-Large-4 native paths leave the other Mistral-native formats as they were.

`mistral` and `mistral_large_3` checkpoints share MistralNativeInputProcessor, the image
processor, the params.json adaptation and the HF config lookup with `mistral_large_4`. Only
Mistral-Large-4 places images at [IMG] markers, keeps prompt token ids and loads its config
from params.json for the LLM API. Its params.json states a ModelOpt `quantization_config`
and a native `quantization` block is refused; the other formats keep the vLLM mapping of it.
"""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.models import modeling_mistral
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import (
    _remap_mistral_quantization_args,
    adapt_config_dict,
)
from tensorrt_llm._torch.models.modeling_mistral import (
    MistralCommonImageProcessor,
    MistralNativeInputProcessor,
)

pytestmark = pytest.mark.cpu_only


class _Encoded(dict):
    __getattr__ = dict.__getitem__


class _RecordingTokenizer:
    """Records the conversation the image processor renders."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append((conversation, kwargs))
        ids = [1, 3, 4]
        return _Encoded(input_ids=torch.tensor([ids]) if kwargs["return_tensors"] else ids)


def _image_processor(native_chat):
    tokenizer = _RecordingTokenizer()
    processor = MistralCommonImageProcessor(
        tokenizer=SimpleNamespace(transformers_tokenizer=tokenizer),
        dtype=torch.bfloat16,
        native_chat=native_chat,
    )
    return processor, tokenizer


def test_other_formats_append_images_and_keep_literal_markers():
    processor, tokenizer = _image_processor(native_chat=False)
    processor("a [IMG] b", images=[Image.new("RGB", (4, 4))])
    ((conversation, kwargs),) = tokenizer.calls
    content = conversation[0]["content"]
    assert content[0] == {"type": "text", "text": "a [IMG] b"}
    assert [part["type"] for part in content] == ["text", "image"]
    assert kwargs["return_tensors"] == "pt"


def test_mistral_large_4_places_images_at_markers():
    processor, tokenizer = _image_processor(native_chat=True)
    processor("a [IMG] b", images=[Image.new("RGB", (4, 4))])
    ((conversation, kwargs),) = tokenizer.calls
    assert [part["type"] for part in conversation[0]["content"]] == ["text", "image", "text"]
    assert kwargs["return_tensors"] is None
    with pytest.raises(ValueError, match="placeholder count"):
        processor("a [IMG] [IMG] b", images=[Image.new("RGB", (4, 4))])


@pytest.mark.parametrize("model_type", ["mistral_large_4", "mistral_common"])
def test_only_mistral_large_4_keeps_prompt_token_ids(monkeypatch, model_type):
    text_tokenizer = SimpleNamespace(encode=lambda text: [42])
    monkeypatch.setattr(
        modeling_mistral.MistralTokenizer,
        "from_pretrained",
        lambda path: SimpleNamespace(transformers_tokenizer=text_tokenizer),
    )
    config = SimpleNamespace(model_type=model_type, torch_dtype=torch.bfloat16)
    processor = MistralNativeInputProcessor(model_path="checkpoint", config=config, tokenizer=None)
    native = model_type == "mistral_large_4"
    assert processor.supports_token_id_mm_expansion is native
    assert processor.processor.native_chat is native
    assert MistralNativeInputProcessor.supports_token_id_mm_expansion is False
    if not native:
        # The other formats tokenize the text prompt, as before.
        prompt = {"prompt": "hi", "prompt_token_ids": [1, 2]}
        assert processor.call_with_text_prompt(prompt, None) == ([42], None)


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


def test_mistral_large_4_refuses_a_native_quantization_block():
    params = {"dsa": {"topk": 2048}, "kv_lora_rank": 512, "lerope": True, **_fp8(**_BLOCK_FP8)}
    with pytest.raises(ValueError, match="ModelOpt MIXED_PRECISION"):
        adapt_config_dict(params)


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


def _nvfp4_quantization_config():
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.routing import (
        _MIXED_PRECISION_LAYERS,
    )

    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "quantized_layers": _MIXED_PRECISION_LAYERS,
    }


def _processor(monkeypatch, vision_config):
    monkeypatch.setattr(
        modeling_mistral.MistralTokenizer,
        "from_pretrained",
        lambda path: SimpleNamespace(transformers_tokenizer=None),
    )
    config = SimpleNamespace(
        model_type="mistral_large_4",
        torch_dtype=torch.bfloat16,
        vision_config=vision_config,
        quantization_config=_nvfp4_quantization_config(),
    )
    return MistralNativeInputProcessor(model_path="checkpoint", config=config, tokenizer=None)


@pytest.mark.parametrize("prompt", ["token_ids", "text"])
def test_an_export_without_a_vision_encoder_takes_no_images(monkeypatch, prompt):
    # Its target is the language model only: an image request fails here, on its own, and
    # never reaches the engine.
    processor = _processor(monkeypatch, vision_config=None)
    inputs = {"multi_modal_data": {"image": [Image.new("RGB", (4, 4))]}}
    if prompt == "token_ids":
        inputs["prompt_token_ids"] = [1, 10, 2]
        call = processor.call_with_token_ids
    else:
        inputs["prompt"] = "[IMG]"
        call = processor.call_with_text_prompt
    with pytest.raises(ValueError, match="serves text only"):
        call(inputs, None)


def test_the_nvfp4_export_takes_images(monkeypatch):
    processor = _processor(monkeypatch, vision_config=SimpleNamespace(image_token_id=10))
    processor._check_images_served([Image.new("RGB", (4, 4))])
