import asyncio
import json
import os as _os
import sys
from pathlib import Path

import pytest
import torch

from tensorrt_llm._utils import mpi_world_size
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor import (GenerationExecutor, GenerationRequest,
                                   SamplingParams)
from tensorrt_llm.hlapi import LLM, BuildConfig
from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer

sys.path.append(_os.path.join(_os.path.dirname(__file__), '..'))
import tempfile

from utils.llm_data import llm_models_root
from utils.util import similar

WORLD_SIZE = mpi_world_size()


@pytest.fixture(scope="module")
def engine_path():
    return Path(tempfile.tempdir) / "llm_engine"


@pytest.fixture(scope="module")
def llama_7b_path(engine_path: Path) -> Path:
    path = engine_path / "llama7b"

    if not path.exists():
        model_dir = str(llm_models_root() / "llama-models/llama-7b-hf")
        llm = LLM(model_dir)
        llm.save(str(path))

    return path


@pytest.fixture(scope="module")
def llama_7b_bs2_path(engine_path: Path) -> Path:
    path = engine_path / "llama7b_bs2"

    if not path.exists():
        model_dir = str(llm_models_root() / "llama-models/llama-7b-hf")
        build_config = BuildConfig()
        build_config.max_beam_width = 2
        # TODO[chunweiy]: switch to executor backend
        llm = LLM(model_dir, build_config=build_config)
        llm.save(str(path))

    return path


@pytest.fixture(scope="module")
def llama_7b_tp2_path(engine_path: Path) -> Path:
    path = engine_path / "llama7b-tp2"

    if not path.exists():
        model_dir = str(llm_models_root() / "llama-models/llama-7b-hf")
        llm = LLM(model_dir, tensor_parallel_size=2)
        llm.save(str(path))

    return path


@pytest.mark.skipif(WORLD_SIZE != 1, reason="Must run on single MPI rank")
def test_generation_bs2(llama_7b_bs2_path: Path):
    tokenizer = TransformersTokenizer.from_pretrained(llama_7b_bs2_path)
    prompt = "A B C D"
    prompt_token_ids = tokenizer.encode(prompt)
    max_tokens = 8

    with GenerationExecutor.create(
            llama_7b_bs2_path,
            executor_config=tllm.ExecutorConfig(max_beam_width=2)) as executor:
        result = executor.generate(prompt_token_ids,
                                   sampling_params=SamplingParams(
                                       max_tokens=max_tokens, beam_width=2))
        assert similar(tokenizer.decode(result.outputs[0].token_ids),
                       'E F G H I J K L')
        assert similar(tokenizer.decode(result.outputs[1].token_ids),
                       'E F G H I K L M')


@pytest.mark.skipif(WORLD_SIZE != 1, reason="Must run on single MPI rank")
def test_sync_generation(llama_7b_path: Path):
    tokenizer = TransformersTokenizer.from_pretrained(llama_7b_path)
    prompt = "A B C D"
    prompt_token_ids = tokenizer.encode(prompt)

    expected_output = "E F G H"
    expected_long_output = "E F G H I J K L"
    sampling_params0 = SamplingParams(max_tokens=4)
    sampling_params1 = SamplingParams(max_tokens=8)
    with GenerationExecutor.create(llama_7b_path) as executor:
        # Simple generations (synchronous)
        result = executor.generate(prompt_token_ids,
                                   sampling_params=sampling_params0)
        assert tokenizer.decode(result.outputs[0].token_ids) == expected_output

        results = executor.generate(
            [prompt_token_ids, prompt_token_ids],
            sampling_params=[sampling_params0, sampling_params1])
        for result, expected in zip(results,
                                    (expected_output, expected_long_output)):
            print(f"result: {result}")
            assert tokenizer.decode(result.outputs[0].token_ids) == expected

        # Iterate the partial results when streaming
        future = executor.generate_async(prompt_token_ids,
                                         sampling_params=sampling_params0,
                                         streaming=True)
        for partial_result in future:
            partial_text = tokenizer.decode(partial_result.outputs[0].token_ids)
            print(f"partial_text: {partial_text}")
            assert expected_output.startswith(partial_text)

        # Iterate the partial results when streaming
        # Streaming results in nested loop
        for sampling_params in [sampling_params0, sampling_params1]:
            future = executor.generate_async(prompt_token_ids,
                                             sampling_params=sampling_params,
                                             streaming=True)
            for partial_result in future:
                partial_text = tokenizer.decode(
                    partial_result.outputs[0].token_ids)
                print(f"partial_text: {partial_text}")
                assert expected_long_output.startswith(partial_text)

        # Low-level api with .submit
        # Submit a batch of requests
        futures = []
        for _ in range(5):
            futures.append(
                executor.submit(
                    GenerationRequest(prompt_token_ids,
                                      sampling_params=sampling_params0)))

        for future in executor.wait_first_completed(futures):
            assert future.done
            assert tokenizer.decode(
                future.result().outputs[0].token_ids) == expected_output


@pytest.mark.skipif(torch.cuda.device_count() < 2 or WORLD_SIZE != 2,
                    reason="Must run on 2 MPI ranks with at least 2 GPUs")
def test_sync_generation_tp_main_node_only(llama_7b_tp2_path: Path):
    tokenizer = TransformersTokenizer.from_pretrained(llama_7b_tp2_path)
    prompt = "deep learning"
    prompt_token_ids = tokenizer.encode(prompt)
    sampling_params = SamplingParams(max_tokens=4)

    with GenerationExecutor.create(llama_7b_tp2_path) as executor:

        executor.block_subordinates()
        # from now on, only rank0 lives in the with statement
        # other nodes wait at the "end" of the with statement

        result = executor.generate(prompt_token_ids,
                                   sampling_params=sampling_params)
        assert tokenizer.decode(
            result.outputs[0].token_ids) == "<s> deep learning, neural network,"


@pytest.mark.skipif(torch.cuda.device_count() < 2 or WORLD_SIZE != 1,
                    reason="Must run on 1 MPI rank with at least 2 GPUs")
def test_sync_generation_tp_inner(llama_7b_tp2_path: Path):
    tokenizer = TransformersTokenizer.from_pretrained(llama_7b_tp2_path)
    prompt = "deep learning"
    prompt_token_ids = tokenizer.encode(prompt)
    tp_size = 2
    sampling_params = SamplingParams(max_tokens=4)

    executor = GenerationExecutor.create(llama_7b_tp2_path,
                                         model_world_size=tp_size)

    async def async_stats_task():
        # asyncio event loop must be created before first generation in order to
        # use async APIs.
        result = executor.generate(prompt_token_ids,
                                   sampling_params=sampling_params)
        assert tokenizer.decode(
            result.outputs[0].token_ids) == ", neural network,"

        stats = await executor.aget_stats()
        stats = json.loads(stats)
        assert stats["iter"] == 0
        assert stats["cpuMemUsage"] > 0
        assert stats["gpuMemUsage"] > 0
        assert stats["inflightBatchingStats"]["numCtxTokens"] == 3
        assert stats["inflightBatchingStats"]["numGenRequests"] == 0
        assert stats["kvCacheStats"]["usedNumBlocks"] == 1

    asyncio.run(async_stats_task())

    stats = executor.get_stats()
    assert json.loads(stats)["iter"] == 1
    executor.shutdown()
