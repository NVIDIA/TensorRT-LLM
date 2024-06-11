import os as _os
import sys as _sys
import unittest
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from tensorrt_llm._utils import mpi_world_size
from tensorrt_llm.bindings import TrtGptModelOptionalParams
from tensorrt_llm.executor import (GenerationExecutor, GenerationRequest,
                                   SamplingParams)
from tensorrt_llm.hlapi.llm import LLM, ModelConfig

_sys.path.append(_os.path.join(_os.path.dirname(__file__), '..'))
from utils.cpp_paths import *  # noqa
from utils.llm_data import llm_models_root
from utils.util import similar

WORLD_SIZE = mpi_world_size()


@pytest.fixture(scope="module")
def llama_7b_path(engine_path: Path) -> Path:
    path = engine_path / "llama7b"

    if not path.exists():
        config = ModelConfig(str(llm_models_root() /
                                 "llama-models/llama-7b-hf"))
        llm = LLM(config)
        llm.save(str(path))

    return path


@pytest.fixture(scope="module")
def llama_7b_bs2_path(engine_path: Path) -> Path:
    path = engine_path / "llama7b_bs2"

    if not path.exists():
        config = ModelConfig(str(llm_models_root() /
                                 "llama-models/llama-7b-hf"))
        config.build_config.max_beam_width = 2
        # TODO[chunweiy]: switch to executor backend
        llm = LLM(config)
        llm.save(str(path))

    return path


@pytest.fixture(scope="module")
def llama_7b_tp2_path(engine_path: Path) -> Path:
    path = engine_path / "llama7b-tp2"

    if not path.exists():
        config = ModelConfig(str(llm_models_root() /
                                 "llama-models/llama-7b-hf"))
        config.parallel_config.tp_size = 2
        llm = LLM(config)
        llm.save(str(path))

    return path


@pytest.mark.skipif(WORLD_SIZE != 1, reason="Must run on single MPI rank")
def test_generation_bs2(llama_7b_bs2_path: Path):
    tokenizer = llama_7b_bs2_path
    prompt = "A B C D"
    max_new_tokens = 8
    executor_config = TrtGptModelOptionalParams()
    executor_config.max_beam_width = 2

    with GenerationExecutor.create(llama_7b_bs2_path,
                                   tokenizer,
                                   executor_config=executor_config) as executor:
        result = executor.generate(prompt,
                                   sampling_params=SamplingParams(
                                       max_new_tokens=max_new_tokens,
                                       beam_width=2))
        assert similar(result.text[0], 'E F G H I J K L')
        assert similar(result.text[1], 'E F G H I K L M')


@pytest.mark.skipif(WORLD_SIZE != 1, reason="Must run on single MPI rank")
def test_sync_generation(llama_7b_path: Path):
    tokenizer = llama_7b_path
    prompt = "A B C D"
    expected_output = "E F G H"
    expected_long_output = "E F G H I J K L"
    split_output = ["E", " F", " G", " H", " I", " J", " K", " L"]
    sampling_params0 = SamplingParams(max_new_tokens=4)
    sampling_params1 = SamplingParams(max_new_tokens=8)
    with GenerationExecutor.create(llama_7b_path, tokenizer) as executor:
        # Simple generations (synchronous)
        result = executor.generate(prompt, sampling_params=sampling_params0)
        assert result.text == expected_output

        results = executor.generate(
            [prompt, prompt],
            sampling_params=[sampling_params0, sampling_params1])
        for result, expected in zip(results,
                                    (expected_output, expected_long_output)):
            assert result.text == expected

        # Simple generations (asynchronous)
        #
        # Iterate the partial results when streaming
        future = executor.generate_async(prompt,
                                         streaming=True,
                                         sampling_params=sampling_params0)
        for idx, partial_result in enumerate(future):
            assert partial_result.text_diff == split_output[idx]

        # Iterate the partial results when streaming
        # Streaming results in nested loop
        futures = executor.generate_async(
            [prompt, prompt],
            streaming=True,
            sampling_params=[sampling_params0, sampling_params1])
        for future in futures:
            for idx, partial_result in enumerate(future):
                assert partial_result.text_diff == split_output[idx]

        # Low-level api with .submit
        # Submit a batch of requests
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        futures = []
        for _ in range(5):
            futures.append(
                executor.submit(
                    GenerationRequest(
                        prompt,
                        tokenizer=AutoTokenizer.from_pretrained(llama_7b_path),
                        sampling_params=sampling_params0)))

        for future in executor.wait_first_completed(futures):
            assert future.done
            assert future.result().text == "".join(split_output[:4])


@pytest.mark.skipif(torch.cuda.device_count() < 2 or WORLD_SIZE != 2,
                    reason="Must run on 2 MPI ranks with at least 2 GPUs")
def test_sync_generation_tp_main_node_only(llama_7b_tp2_path: Path):
    prompt = "deep learning"
    sampling_params = SamplingParams(max_new_tokens=4)

    with GenerationExecutor.create(llama_7b_tp2_path,
                                   llama_7b_tp2_path) as executor:

        executor.block_subordinates()
        # from now on, only rank0 lives in the with statement
        # other nodes wait at the "end" of the with statement

        result = executor.generate(prompt, sampling_params=sampling_params)
        assert result.text == "<s> deep learning, neural network,"


@pytest.mark.skipif(torch.cuda.device_count() < 2 or WORLD_SIZE != 1,
                    reason="Must run on 1 MPI rank with at least 2 GPUs")
def test_sync_generation_tp_inner(llama_7b_tp2_path: Path):
    prompt = "deep learning"
    tp_size = 2
    sampling_params = SamplingParams(max_new_tokens=4)

    executor = GenerationExecutor.create(llama_7b_tp2_path,
                                         llama_7b_tp2_path,
                                         model_world_size=tp_size)
    result = executor.generate(prompt, sampling_params=sampling_params)
    assert result.text == ", neural network,"
    executor.shutdown()


if __name__ == "__main__":
    unittest.main()
