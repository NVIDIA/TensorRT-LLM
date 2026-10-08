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

import pytest
import torch

from tensorrt_llm.rocm.llm import LLM, _StopStrings
from tensorrt_llm.rocm.sampling import SamplingParams
from tensorrt_llm.rocm.validation import tiny_model_and_tokenizer, validate

pytestmark = pytest.mark.cpu_only


def test_offline_logits_cache_and_generation_parity() -> None:
    report = validate("cpu", "torch", ("float32",))
    assert report["passed"], [check for check in report["checks"] if not check["passed"]]
    assert not report["native_kernels_executed"]
    assert len(report["checks"]) > 40


def test_batched_generation_preserves_prompt_order_and_token_inputs(engine) -> None:
    params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
    texts = ["tok4 tok5", "tok6", "tok7 tok8 tok9"]
    result = engine.generate(texts, params)
    assert [output.prompt for output in result] == texts
    assert [output.prompt_token_ids for output in result] == [[4, 5], [6], [7, 8, 9]]
    assert len({output.request_id for output in result}) == 3
    assert all(len(output.outputs[0].token_ids) == 4 for output in result)
    token_result = engine.generate([[4, 5], [6]], params)
    assert [output.outputs[0].token_ids for output in token_result] == [
        output.outputs[0].token_ids for output in result[:2]
    ]
    assert engine.generate([], params) == []


def test_seeded_sampling_restores_rng_state(engine) -> None:
    params = SamplingParams(
        temperature=0.8, top_k=10, top_p=0.9, seed=47, max_tokens=5, ignore_eos=True
    )
    state = torch.random.get_rng_state().clone()
    first = engine.generate("tok4 tok5", params)
    second = engine.generate("tok4 tok5", params)
    assert first[0].outputs[0].token_ids == second[0].outputs[0].token_ids
    assert torch.equal(state, torch.random.get_rng_state())


def test_multiple_samples_and_beams(engine) -> None:
    for params in (
        SamplingParams(n=2, temperature=0.8, seed=5, max_tokens=3),
        SamplingParams(n=2, beam_width=2, temperature=0, max_tokens=3),
    ):
        outputs = engine.generate("tok4", params)[0].outputs
        assert len(outputs) == 2
        assert [output.index for output in outputs] == [0, 1]


def test_stop_strings_do_not_match_the_prompt(engine) -> None:
    criterion = _StopStrings(engine.tokenizer, ["tok4"], prompt_width=1, minimum=0)
    assert not criterion(torch.tensor([[4, 5]]), None).item()
    assert criterion(torch.tensor([[5, 4]]), None).item()


def test_context_validation_streaming_and_shutdown(engine) -> None:
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        engine.generate("tok4", SamplingParams(max_tokens=128))
    with pytest.raises(NotImplementedError, match="Streaming"):
        engine.generate("tok4", streaming=True)
    with pytest.raises(ValueError, match="outside"):
        engine.generate([1000], SamplingParams(max_tokens=1))
    engine.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        engine.generate("tok4")


@pytest.mark.parametrize(
    "options",
    [
        {"tensor_parallel_size": 2},
        {"pipeline_parallel_size": 2},
        {"cuda_graph_config": {}},
        {"quant_config": {}},
    ],
)
def test_no_silent_unsupported_option_fallback(options) -> None:
    with pytest.raises(NotImplementedError):
        LLM("unused", device="cpu", **options)


def test_quantized_checkpoint_rejected() -> None:
    model, tokenizer = tiny_model_and_tokenizer()
    model.config.quantization_config = {"quant_method": "nvfp4"}
    with pytest.raises(NotImplementedError, match="Quantized"):
        LLM(model, tokenizer, device="cpu")


def test_native_kernel_requires_gpu() -> None:
    with pytest.raises(ValueError, match="real RDNA4"):
        LLM("unused", device="cpu", kernels="hip")
