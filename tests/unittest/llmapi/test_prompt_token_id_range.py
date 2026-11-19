# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prompt token ids outside the vocabulary are rejected before submission."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import tensorrt_llm.llmapi.llm as llm_module
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm import (
    BaseLLM,
    PreprocessedInputs,
    _check_prompt_token_id_range,
    _multimodal_params_replace_prompt_ids,
)
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

_VOCAB = 1000


def _as(kind, ids):
    if kind == "list":
        return list(ids)
    if kind == "numpy":
        return np.asarray(ids, dtype=np.int32)
    return torch.tensor(ids, dtype=torch.int32)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_in_range_ids_are_accepted(kind):
    _check_prompt_token_id_range(_as(kind, [0, 5, _VOCAB - 1]), _VOCAB, allow_out_of_vocab=False)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_id_at_vocab_size_is_rejected(kind):
    with pytest.raises(RequestError, match=f"prompt token id {_VOCAB} does not fit a vocabulary"):
        _check_prompt_token_id_range(_as(kind, [1, _VOCAB, 2]), _VOCAB, allow_out_of_vocab=False)


@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_negative_id_is_rejected(kind):
    with pytest.raises(
        RequestError, match="^Token ID out of range: prompt token ids must be non-negative, got -1"
    ):
        _check_prompt_token_id_range(_as(kind, [1, -1, 2]), _VOCAB, allow_out_of_vocab=False)


def test_negative_id_is_rejected_even_when_out_of_vocab_is_allowed():
    with pytest.raises(RequestError, match="must be non-negative"):
        _check_prompt_token_id_range([-5, 3], _VOCAB, allow_out_of_vocab=True)


def test_out_of_vocab_placeholders_are_allowed_when_requested():
    _check_prompt_token_id_range([1, _VOCAB, _VOCAB + 7], _VOCAB, allow_out_of_vocab=True)


def test_unknown_vocab_size_only_checks_the_sign():
    _check_prompt_token_id_range([1, 10**9], None, allow_out_of_vocab=False)


def test_empty_prompt_is_left_to_other_checks():
    _check_prompt_token_id_range([], _VOCAB, allow_out_of_vocab=False)


@pytest.mark.parametrize(
    "hf_model_config,expected",
    [
        (None, None),
        (SimpleNamespace(), None),
        (SimpleNamespace(vocab_size=_VOCAB), _VOCAB),
        (SimpleNamespace(vocab_size=0), None),
        # Multimodal configs keep the text embedding size in text_config.
        (SimpleNamespace(vocab_size=5, text_config=SimpleNamespace(vocab_size=_VOCAB)), _VOCAB),
        (SimpleNamespace(vocab_size=_VOCAB, text_config=SimpleNamespace()), _VOCAB),
    ],
)
def test_prompt_vocab_size_reads_the_text_config_first(hf_model_config, expected):
    llm = SimpleNamespace(_hf_model_config=hf_model_config)
    assert BaseLLM._prompt_vocab_size(llm) == expected


@pytest.mark.parametrize(
    "multimodal_params,expected",
    [
        (None, False),
        (MultimodalParams(), False),
        (MultimodalParams(multimodal_data={"mrope_config": {"mrope_position_ids": None}}), False),
        (MultimodalParams(multimodal_data={"multimodal_embedding": torch.zeros(1)}), True),
        (MultimodalParams(multimodal_data={"image": {"pixel_values": torch.zeros(1)}}), True),
    ],
)
def test_only_replacing_multimodal_content_exempts_prompt_ids(multimodal_params, expected):
    assert _multimodal_params_replace_prompt_ids(multimodal_params) is expected


def _fake_llm(monkeypatch, *, encoder_decoder=False, preprocess=None):
    """A real BaseLLM, built without __init__, that runs generate_async up to the executor.

    Only the collaborators that need a loaded model are replaced on the
    instance; every other method is the real one.
    """
    executor = MagicMock()
    executor.is_shutdown.return_value = False
    llm = BaseLLM.__new__(BaseLLM)
    llm._encode_only = False
    llm._executor = executor
    llm._hf_model_config = SimpleNamespace(vocab_size=_VOCAB, is_encoder_decoder=encoder_decoder)
    llm.args = SimpleNamespace(return_perf_metrics=False)
    llm._tokenizer = None
    llm._post_processor_hook = None
    llm._prepare_sampling_params = lambda sampling_params: sampling_params or SamplingParams()
    llm._check_arguments = lambda *args, **kwargs: None
    llm._get_decoder_prompt_token_ids = lambda sampling_params: [0]
    if preprocess is not None:
        llm._preprocess = preprocess
    monkeypatch.setattr(
        llm_module.RequestOutput, "_from_generation_result", lambda *args, **kwargs: "output"
    )
    return llm, executor


_GEN_ONLY = SimpleNamespace(request_type="generation_only")


@pytest.mark.parametrize(
    "inputs,kwargs,encoder_decoder",
    [
        (PreprocessedInputs(prompt_token_ids=[1, _VOCAB]), {}, False),
        (PreprocessedInputs(prompt_token_ids=[1, -1]), {}, False),
        (PreprocessedInputs(prompt_token_ids=[-1]), {"disaggregated_params": _GEN_ONLY}, False),
        (
            PreprocessedInputs(
                prompt_token_ids=[1, _VOCAB],
                multimodal_params=MultimodalParams(
                    multimodal_data={"mrope_config": {"mrope_position_ids": None}}
                ),
            ),
            {},
            False,
        ),
        (PreprocessedInputs(prompt_token_ids=[0], encoder_input_token_ids=[1, _VOCAB]), {}, True),
    ],
    ids=["above_vocab", "negative", "negative_gen_only", "mrope_only", "encoder_ids"],
)
def test_generate_async_rejects_before_submission(monkeypatch, inputs, kwargs, encoder_decoder):
    llm, executor = _fake_llm(monkeypatch, encoder_decoder=encoder_decoder)
    with pytest.raises(RequestError):
        BaseLLM.generate_async(llm, inputs, **kwargs)
    executor.generate_async.assert_not_called()


@pytest.mark.parametrize("inputs", ["a text prompt", [1, _VOCAB]], ids=["text", "token_ids"])
def test_generate_async_rejects_raw_prompts_before_submission(monkeypatch, inputs):
    preprocess = MagicMock(return_value=([1, _VOCAB], None, None, None))
    llm, executor = _fake_llm(monkeypatch, preprocess=preprocess)
    with pytest.raises(RequestError, match=f"prompt token id {_VOCAB} does not fit a vocabulary"):
        BaseLLM.generate_async(llm, inputs)
    preprocess.assert_called_once()
    assert preprocess.call_args.args[0] == inputs
    executor.generate_async.assert_not_called()


@pytest.mark.parametrize(
    "inputs,kwargs",
    [
        (PreprocessedInputs(prompt_token_ids=[0, _VOCAB - 1]), {}),
        (PreprocessedInputs(prompt_token_ids=[1, _VOCAB + 3]), {"disaggregated_params": _GEN_ONLY}),
        (
            PreprocessedInputs(
                prompt_token_ids=[1, _VOCAB + 3],
                multimodal_params=MultimodalParams(
                    multimodal_data={"multimodal_embedding": torch.zeros(1)}
                ),
            ),
            {},
        ),
        (
            PreprocessedInputs(prompt_token_ids=[1, _VOCAB + 3]),
            {"prompt_adapter_request": object()},
        ),
    ],
    ids=["in_range", "gen_only", "multimodal_embedding", "prompt_adapter"],
)
def test_generate_async_submits_accepted_prompts(monkeypatch, inputs, kwargs):
    llm, executor = _fake_llm(monkeypatch)
    assert BaseLLM.generate_async(llm, inputs, **kwargs) == "output"
    executor.generate_async.assert_called_once()


def _fake_encode_llm():
    """A real BaseLLM in encode-only mode with a stub encoder executor."""
    encoder = MagicMock()
    encoder.model_engine = SimpleNamespace(max_seq_len=64, max_num_tokens=256, batch_size=4)
    encoder.batch_forward.return_value = {"logits": torch.zeros(2, 3)}
    llm = BaseLLM.__new__(BaseLLM)
    llm._encode_only = True
    llm._encoder_executor = encoder
    llm._hf_model_config = SimpleNamespace(vocab_size=_VOCAB)
    return llm, encoder


@pytest.mark.parametrize(
    "inputs",
    [[1, _VOCAB], [[0, 1], [2, _VOCAB + 5]], [[0, 1], [-1, 2]]],
    ids=["single", "batched_above_vocab", "batched_negative"],
)
def test_encode_rejects_before_forward(inputs):
    llm, encoder = _fake_encode_llm()
    with pytest.raises(RequestError):
        llm.encode(inputs, return_raw_logits=True)
    encoder.batch_forward.assert_not_called()


@pytest.mark.parametrize(
    "inputs,model_kwargs",
    [([[0, 1], [2, _VOCAB - 1]], {}), ([[0, 1], [2, _VOCAB + 5]], {"inputs_embeds": object()})],
    ids=["in_range", "inputs_embeds"],
)
def test_encode_forwards_accepted_inputs(inputs, model_kwargs):
    llm, encoder = _fake_encode_llm()
    llm.encode(inputs, return_raw_logits=True, **model_kwargs)
    encoder.batch_forward.assert_called_once()
