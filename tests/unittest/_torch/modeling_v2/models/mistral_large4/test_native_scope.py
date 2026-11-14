# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mistral-Large-4 native paths leave the other Mistral-native checkpoints as they were.

Mistral-Large-4 is read through the same `mistral` format as the other Mistral-native
checkpoints, and shares MistralNativeInputProcessor, the image processor, the params.json
adaptation and the HF config lookup with them. Only Mistral-Large-4 places images at [IMG]
markers, keeps prompt token ids and loads its config from params.json for the LLM API. The
other checkpoints keep the vLLM mapping of a native `quantization` block.
"""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.models import modeling_mistral
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import (
    _remap_mistral_quantization_args,
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
    ((conversation, _),) = tokenizer.calls
    content = conversation[0]["content"]
    assert content[0] == {"type": "text", "text": "a [IMG] b"}
    assert [part["type"] for part in content] == ["text", "image"]


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


@pytest.mark.parametrize("checkpoint_format", ["mistral", "mistral_large_3", "HF", None])
@pytest.mark.parametrize("is_mistral_large_4", [True, False])
def test_hf_config_lookup_names_the_native_format_only_for_mistral_large_4(
    monkeypatch, checkpoint_format, is_mistral_large_4
):
    from tensorrt_llm._torch.models.checkpoints.mistral import config_loader
    from tensorrt_llm.llmapi import llm as llm_module

    calls = []
    monkeypatch.setattr(
        llm_module.ModelLoader,
        "load_hf_model_config",
        staticmethod(lambda model, **kwargs: calls.append(kwargs)),
    )
    monkeypatch.setattr(
        config_loader, "is_mistral_large4_checkpoint", lambda path: is_mistral_large_4
    )
    args = SimpleNamespace(model="m", trust_remote_code=False, checkpoint_format=checkpoint_format)
    llm_module.BaseLLM._try_load_hf_model_config(SimpleNamespace(args=args))
    expected = {"trust_remote_code": False}
    if is_mistral_large_4 and checkpoint_format in ("mistral", "mistral_large_3"):
        expected["checkpoint_format"] = checkpoint_format
    assert calls == [expected]
