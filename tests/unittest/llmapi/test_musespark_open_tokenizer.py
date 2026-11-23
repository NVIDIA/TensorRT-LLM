# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Muse Spark Open runs on the default tokenizer.

Nothing model-specific is needed: transformers loads the chat template from
``chat_template.jinja``, ``tokenizer_config.json`` names the pad token, and
the second EOS id in ``generation_config.json`` becomes a stop id the way it
does for every model. These tests pin those facts on the released checkpoints'
own files, loaded without remote code. Each runs once per released checkpoint
under ``LLM_MODELS_ROOT`` and skips the ones that are absent.
"""

import json
import os

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm.llmapi.llm_utils import ModelLoader
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.tokenizer import load_hf_tokenizer

pytestmark = pytest.mark.cpu_only

CHECKPOINTS = ("Muse-Spark-Open-816B-A42B-NVFP4", "Muse-Spark-Open-816B-A42B")

BEGIN_OF_TEXT = 200000
END_OF_TEXT = 200001
END_OF_MESSAGE = 200007
END_OF_TURN = 200008
PAD = 200018

# Raw completion: no chat template and no BOS. The ids come from an independent
# implementation of the model, so a mismatch is a real tokenizer difference.
RAW_PROMPTS = [
    ("The capital of France is", [954, 7963, 323, 11698, 373]),
    (
        'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
        [1251, 105436, 2736, 1960, 277, 10938, 9586, 290, 307, 9588, 75714, 1416, 34807],
    ),
    ("12345 and 987 plus 1000000", [7235, 2127, 341, 220, 41834, 3449, 220, 1135, 1052, 28]),
]


@pytest.fixture(scope="module", params=CHECKPOINTS)
def checkpoint_dir(request):
    path = os.path.join(llm_models_root(), request.param)
    # An absent or empty directory is a checkpoint not synced to this machine
    # (a submodule's directory exists before its contents arrive). A synced
    # checkpoint must be complete.
    if not os.path.isdir(path) or not os.listdir(path):
        pytest.skip(f"{request.param} is not under LLM_MODELS_ROOT")
    assert os.path.isfile(os.path.join(path, "tokenizer.json")), f"{path} has no tokenizer.json"
    return path


@pytest.fixture(scope="module")
def tokenizer(checkpoint_dir):
    tokenizer = load_hf_tokenizer(checkpoint_dir, trust_remote_code=False)
    assert tokenizer is not None
    return tokenizer


def test_the_chat_template_comes_from_the_checkpoint_file(checkpoint_dir, tokenizer):
    with open(os.path.join(checkpoint_dir, "tokenizer_config.json"), encoding="utf-8") as f:
        assert json.load(f).get("chat_template") is None, (
            "tokenizer_config.json now carries the template; revisit this test's premise"
        )
    with open(os.path.join(checkpoint_dir, "chat_template.jinja"), encoding="utf-8") as f:
        assert tokenizer.tokenizer.chat_template == f.read()


def test_the_generation_prompt_ends_on_the_assistant_header(tokenizer):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    # The model's first emitted text is the ` to=RECIPIENT<|message|>` header
    # that follows `<|start|>assistant`.
    assert rendered.endswith("<|start|>assistant")
    assert "<|start|>user<|message|>hi<|eot|>" in rendered


def test_bos_comes_from_the_chat_template_not_the_tokenizer(tokenizer):
    """The chat template starts every prompt with BOS; the tokenizer adds none.

    Completions are encoded with special tokens, so a tokenizer that started
    adding BOS would change every raw completion prompt.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    assert rendered.startswith("<|begin_of_text|>")
    assert BEGIN_OF_TEXT not in tokenizer.encode("hi")


@pytest.mark.parametrize("text,expected_ids", RAW_PROMPTS)
def test_raw_completion_ids(tokenizer, text, expected_ids):
    assert tokenizer.encode(text, add_special_tokens=False) == expected_ids


def test_the_pad_token_comes_from_the_tokenizer_config(tokenizer):
    assert tokenizer.pad_token_id == PAD
    assert tokenizer.convert_ids_to_tokens(PAD) == "<|finetune_right_pad|>"


def test_both_eos_ids_stop_generation_and_end_of_message_does_not(checkpoint_dir, tokenizer):
    """``<|eom|>`` must never stop generation.

    It ends every reasoning block and non-final tool call, and the model keeps
    going after it.
    """
    assert tokenizer.convert_ids_to_tokens(END_OF_TEXT) == "<|end_of_text|>"
    assert tokenizer.convert_ids_to_tokens(END_OF_TURN) == "<|eot|>"
    assert tokenizer.convert_ids_to_tokens(END_OF_MESSAGE) == "<|eom|>"
    generation_config = ModelLoader.load_hf_generation_config(checkpoint_dir)
    params = SamplingParams()._setup(tokenizer, None, generation_config)
    assert params.end_id == END_OF_TEXT
    assert params.stop_token_ids == [END_OF_TURN]


def test_end_of_message_decodes_as_text_only_with_special_tokens_kept(tokenizer):
    assert tokenizer.decode([END_OF_MESSAGE], skip_special_tokens=False) == "<|eom|>"
    assert tokenizer.decode([END_OF_MESSAGE], skip_special_tokens=True) == ""


def test_the_config_audio_marker_ids_match_the_tokenizer(checkpoint_dir, tokenizer):
    """The config hardcodes the audio marker ids; the checkpoint's tokenizer must agree."""
    args = load_pretrained_config(checkpoint_dir).audio_perception_modality_args
    assert tokenizer.convert_ids_to_tokens(args["next_audio_token"]) == "<|next_audio|>"
    assert tokenizer.convert_ids_to_tokens(args["empty_audio_token"]) == "<|empty_audio|>"
