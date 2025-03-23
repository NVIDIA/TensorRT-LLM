from pathlib import Path

# isort: off
from utils.llm_data import llm_models_root
# isort: on

import pytest

from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.scaffolding.controller import (MajorityVoteController,
                                                 NativeGenerationController)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.worker import ProposerWorker, SamplingParams


@pytest.fixture(scope="module")
def deepseek_distill_7b_path() -> Path:
    model_dir = llm_models_root() / "DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B"
    return model_dir


@pytest.fixture(scope="module")
def sampling_params():
    return SamplingParams(max_tokens=2048)


@pytest.fixture(scope="module")
def pytorch_config():
    return PyTorchConfig(
        mixed_decoder=True,
        enable_overlap_scheduler=True,
    )


@pytest.fixture(scope="module")
def default_prompt():
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\r\n\r\n"
    return prompt


def create_proposer_worker(deepseek_distill_7b_path, sampling_params,
                           pytorch_config):
    return ProposerWorker(
        deepseek_distill_7b_path,
        pytorch_backend_config=pytorch_config,
        sampling_params=sampling_params,
    )


def create_scaffolding_llm_with_native_generation_controller(
        deepseek_distill_7b_path, sampling_params, pytorch_config):
    proposer_worker = create_proposer_worker(deepseek_distill_7b_path,
                                             sampling_params, pytorch_config)
    prototype_generation_controller = NativeGenerationController()
    return ScaffoldingLlm(
        prototype_generation_controller,
        {NativeGenerationController.WorkerTag.GENERATION: proposer_worker},
    )


def create_scaffolding_llm_with_majority_vote_controller(
        deepseek_distill_7b_path, sampling_params, pytorch_config, samples_num):
    proposer_worker = create_proposer_worker(deepseek_distill_7b_path,
                                             sampling_params, pytorch_config)

    workers = {}
    prototype_generation_controller = NativeGenerationController()
    workers[NativeGenerationController.WorkerTag.GENERATION] = proposer_worker

    prototype_majority_vote_controller = MajorityVoteController(
        prototype_generation_controller,
        default_sample_num=samples_num,
    )

    llm = ScaffoldingLlm(
        prototype_majority_vote_controller,
        workers=workers,
    )

    return llm


def test_unbatched_scaffolding_sync(default_prompt, deepseek_distill_7b_path,
                                    sampling_params, pytorch_config):
    scaffolding_llm = create_scaffolding_llm_with_native_generation_controller(
        deepseek_distill_7b_path, sampling_params, pytorch_config)
    result = scaffolding_llm.generate(default_prompt)
    assert isinstance(result.output.output_str, str) and len(
        result.output.output_str) > 0, "Output should be a non-empty string"
    scaffolding_llm.shutdown(shutdown_wokers=True)


def test_batched_scaffolding_sync(default_prompt, deepseek_distill_7b_path,
                                  sampling_params, pytorch_config):
    scaffolding_llm = create_scaffolding_llm_with_native_generation_controller(
        deepseek_distill_7b_path, sampling_params, pytorch_config)
    batch_size = 3
    prompts = [default_prompt] * batch_size
    results = scaffolding_llm.generate(prompts)
    assert len(results) == batch_size
    for result in results:
        assert isinstance(result.output.output_str, str) and len(
            result.output.output_str) > 0, "Output should be a non-empty string"
    scaffolding_llm.shutdown(shutdown_wokers=True)


def test_async_scaffolding_generation(default_prompt, deepseek_distill_7b_path,
                                      sampling_params, pytorch_config):

    async def run_async_test():
        scaffolding_llm = create_scaffolding_llm_with_native_generation_controller(
            deepseek_distill_7b_path, sampling_params, pytorch_config)
        future = scaffolding_llm.generate_async(default_prompt)
        result = await future.aresult()
        assert isinstance(result.output.output_str, str) and len(
            result.output.output_str) > 0, "Output should be a non-empty string"
        scaffolding_llm.shutdown(shutdown_wokers=True)

    import asyncio
    asyncio.run(run_async_test())


def test_majority_vote(default_prompt, deepseek_distill_7b_path,
                       sampling_params, pytorch_config):
    scaffolding_llm = create_scaffolding_llm_with_majority_vote_controller(
        deepseek_distill_7b_path,
        sampling_params,
        pytorch_config,
        samples_num=3)
    result = scaffolding_llm.generate(default_prompt)
    assert isinstance(result.output.output_str, str) and len(
        result.output.output_str) > 0, "Output should be a non-empty string"
    scaffolding_llm.shutdown(shutdown_wokers=True)
