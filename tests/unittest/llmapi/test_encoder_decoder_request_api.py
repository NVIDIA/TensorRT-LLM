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

from inspect import signature
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.llmapi.llm import BaseLLM, PreprocessedInputs
from tensorrt_llm.sampling_params import SamplingParams


def _sampling_params():
    return SamplingParams(max_tokens=5, end_id=1, pad_id=0)


class _FakeExecutor(GenerationExecutor):
    def __init__(self):
        super().__init__()
        self.submitted = []

    def submit(self, request):
        self.submitted.append(request)
        result = MagicMock()
        result.request_id = 0
        return result

    def abort_request(self, request_id):
        pass

    def shutdown(self):
        pass


def _make_llm_for_preprocess(decoder_start_token_id=0):
    llm = BaseLLM.__new__(BaseLLM)
    llm.args = SimpleNamespace(
        backend="pytorch",
        enable_chunked_prefill=True,
        return_perf_metrics=False,
        stream_interval=1,
        parallel_config=SimpleNamespace(cp_size=1),
    )
    llm._generation_config = None
    llm._hf_model_config = SimpleNamespace(decoder_start_token_id=decoder_start_token_id)
    llm.input_processor = SimpleNamespace()
    llm._tokenizer = None
    return llm


def _make_llm_with_mock_executor(decoder_start_token_id=0):
    llm = _make_llm_for_preprocess(decoder_start_token_id)
    result = MagicMock()
    result._streaming = False
    result.metrics_dict = {}
    llm._executor = MagicMock()
    llm._executor.is_shutdown.return_value = False
    llm._executor.generate_async.return_value = result
    return llm


def test_encoder_decoder_kwargs_do_not_shift_priority_position():
    generate_params = list(signature(BaseLLM.generate).parameters)
    generate_async_params = list(signature(BaseLLM.generate_async).parameters)

    assert generate_params.index("priority") < generate_params.index("encoder_inputs")
    assert generate_async_params.index("priority") < generate_async_params.index("encoder_inputs")


def test_generation_request_stores_encoder_input_token_ids():
    req = GenerationRequest(
        prompt_token_ids=[0],
        sampling_params=_sampling_params(),
        encoder_input_token_ids=[11, 12, 13],
    )

    assert req.prompt_token_ids == [0]
    assert req.encoder_input_token_ids == [11, 12, 13]


def test_generation_executor_forwards_encoder_input_token_ids():
    executor = _FakeExecutor()

    executor.generate_async(
        prompt_token_ids=[0],
        sampling_params=_sampling_params(),
        encoder_input_token_ids=[21, 22],
    )

    assert executor.submitted[0].encoder_input_token_ids == [21, 22]


def test_base_worker_forwards_encoder_input_token_ids_to_executor_request():
    import tensorrt_llm.executor.base_worker as bw_mod
    from tensorrt_llm.executor.base_worker import BaseWorker

    captured = {}

    class CapturingRequest:
        def __init__(self, *args, **kwargs):
            captured["encoder_input_token_ids"] = kwargs.get("encoder_input_token_ids")
            self.py_num_logprobs = None
            self.py_lora_path = None
            self.py_logprobs_mode = None

    req = GenerationRequest(
        prompt_token_ids=[0],
        sampling_params=_sampling_params(),
        encoder_input_token_ids=[31, 32],
    )
    req.set_id(42)

    worker = MagicMock()
    worker.llm_args = MagicMock()
    worker.llm_args.return_perf_metrics = False
    worker._executor_config = None
    worker._is_pytorch_backend = False
    worker.max_seq_len = None
    worker.engine = MagicMock()
    worker.engine.enqueue_request = MagicMock(return_value=42)

    with patch.object(bw_mod.tllm, "Request", CapturingRequest):
        BaseWorker._enqueue_request(worker, req, result_wait_queue=None)

    assert captured["encoder_input_token_ids"] == [31, 32]


def test_preprocess_synthesizes_decoder_start_token_for_encoder_request():
    llm = _make_llm_for_preprocess(decoder_start_token_id=0)

    inputs = BaseLLM.preprocess(
        llm,
        {"encoder_input_token_ids": [41, 42]},
        sampling_params=_sampling_params(),
    )

    assert inputs.prompt_token_ids == [0]
    assert inputs.encoder_input_token_ids == [41, 42]


def test_preprocess_accepts_decoder_input_token_ids_for_encoder_request():
    llm = _make_llm_for_preprocess(decoder_start_token_id=None)

    inputs = BaseLLM.preprocess(
        llm,
        {
            "encoder_input_token_ids": [51, 52],
            "decoder_input_token_ids": [2, 3],
        },
        sampling_params=_sampling_params(),
    )

    assert inputs.prompt_token_ids == [2, 3]
    assert inputs.encoder_input_token_ids == [51, 52]


def test_preprocess_accepts_explicit_encoder_token_kwarg():
    llm = _make_llm_for_preprocess()

    inputs = BaseLLM.preprocess(
        llm,
        [2, 3],
        sampling_params=_sampling_params(),
        encoder_input_token_ids=[55, 56],
    )

    assert inputs.prompt_token_ids == [2, 3]
    assert inputs.encoder_input_token_ids == [55, 56]


def test_preprocess_requires_decoder_start_token_when_decoder_input_missing():
    llm = _make_llm_for_preprocess(decoder_start_token_id=None)

    with pytest.raises(ValueError, match="decoder_start_token_id"):
        BaseLLM.preprocess(
            llm,
            {"encoder_input_token_ids": [61, 62]},
            sampling_params=_sampling_params(),
        )


def test_generate_async_forwards_preprocessed_encoder_input_token_ids():
    llm = _make_llm_with_mock_executor()

    BaseLLM.generate_async(
        llm,
        PreprocessedInputs(
            prompt_token_ids=[0],
            encoder_input_token_ids=[71, 72],
        ),
        sampling_params=_sampling_params(),
    )

    assert llm._executor.generate_async.call_args.kwargs["encoder_input_token_ids"] == [71, 72]


def test_generate_async_accepts_encoder_token_kwarg_with_preprocessed_inputs():
    llm = _make_llm_with_mock_executor()

    BaseLLM.generate_async(
        llm,
        PreprocessedInputs(prompt_token_ids=[0]),
        sampling_params=_sampling_params(),
        encoder_input_token_ids=[81, 82],
    )

    assert llm._executor.generate_async.call_args.kwargs["encoder_input_token_ids"] == [81, 82]


def test_generate_async_rejects_conflicting_preprocessed_encoder_tokens():
    llm = _make_llm_with_mock_executor()

    with pytest.raises(ValueError, match="Conflicting encoder_input_token_ids"):
        BaseLLM.generate_async(
            llm,
            PreprocessedInputs(
                prompt_token_ids=[0],
                encoder_input_token_ids=[91, 92],
            ),
            sampling_params=_sampling_params(),
            encoder_input_token_ids=[93, 94],
        )


def test_generate_async_rejects_raw_kwargs_with_preprocessed_inputs():
    llm = _make_llm_with_mock_executor()

    with pytest.raises(ValueError, match="encoder_inputs cannot"):
        BaseLLM.generate_async(
            llm,
            PreprocessedInputs(prompt_token_ids=[0]),
            sampling_params=_sampling_params(),
            encoder_inputs="source",
        )

    with pytest.raises(ValueError, match="decoder_input_token_ids cannot"):
        BaseLLM.generate_async(
            llm,
            PreprocessedInputs(prompt_token_ids=[0]),
            sampling_params=_sampling_params(),
            decoder_input_token_ids=[1],
        )


def test_generate_async_accepts_old_positional_priority_argument():
    llm = _make_llm_with_mock_executor()

    BaseLLM.generate_async(
        llm,
        [0],
        _sampling_params(),
        None,
        None,
        False,
        None,
        None,
        None,
        None,
        None,
        None,
        0.7,
    )

    assert llm._executor.generate_async.call_args.kwargs["priority"] == 0.7
