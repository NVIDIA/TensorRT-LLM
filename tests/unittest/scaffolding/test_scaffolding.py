# autoflake: skip_file

from scaffolding.test_worker import (create_trtllm_worker,
                                     deepseek_distill_7b_path, default_prompt)

from tensorrt_llm.scaffolding import (MajorityVoteController,
                                      NativeGenerationController,
                                      ScaffoldingLlm)


def create_scaffolding_llm_with_native_generation_controller(
        deepseek_distill_7b_path):
    trtllm_worker = create_trtllm_worker(deepseek_distill_7b_path)
    prototype_generation_controller = NativeGenerationController(
        sampling_params={
            "max_tokens": 8,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        })
    return ScaffoldingLlm(
        prototype_generation_controller,
        {NativeGenerationController.WorkerTag.GENERATION: trtllm_worker},
    )


def create_scaffolding_llm_with_majority_vote_controller(
        deepseek_distill_7b_path, samples_num):
    trtllm_worker = create_trtllm_worker(deepseek_distill_7b_path)

    workers = {}
    prototype_generation_controller = NativeGenerationController()
    workers[NativeGenerationController.WorkerTag.GENERATION] = trtllm_worker

    prototype_majority_vote_controller = MajorityVoteController(
        prototype_generation_controller,
        default_sample_num=samples_num,
    )

    llm = ScaffoldingLlm(
        prototype_majority_vote_controller,
        workers=workers,
    )

    return llm


def test_unbatched_scaffolding_sync(default_prompt, deepseek_distill_7b_path):
    scaffolding_llm = create_scaffolding_llm_with_native_generation_controller(
        deepseek_distill_7b_path)
    result = scaffolding_llm.generate(default_prompt)
    assert isinstance(result.output.output_str, str) and len(
        result.output.output_str) > 0, "Output should be a non-empty string"
    scaffolding_llm.shutdown(shutdown_workers=True)


def test_batched_scaffolding_sync(default_prompt, deepseek_distill_7b_path):
    scaffolding_llm = create_scaffolding_llm_with_native_generation_controller(
        deepseek_distill_7b_path)
    batch_size = 3
    prompts = [default_prompt] * batch_size
    results = scaffolding_llm.generate(prompts)
    assert len(results) == batch_size
    for result in results:
        assert isinstance(result.output.output_str, str) and len(
            result.output.output_str) > 0, "Output should be a non-empty string"
    scaffolding_llm.shutdown(shutdown_workers=True)


def test_async_scaffolding_generation(default_prompt, deepseek_distill_7b_path):

    async def run_async_test():
        scaffolding_llm = create_scaffolding_llm_with_native_generation_controller(
            deepseek_distill_7b_path)
        future = scaffolding_llm.generate_async(default_prompt)
        result = await future.aresult()
        assert isinstance(result.output.output_str, str) and len(
            result.output.output_str) > 0, "Output should be a non-empty string"
        scaffolding_llm.shutdown(shutdown_workers=True)

    import asyncio
    asyncio.run(run_async_test())


def test_majority_vote(default_prompt, deepseek_distill_7b_path):
    scaffolding_llm = create_scaffolding_llm_with_majority_vote_controller(
        deepseek_distill_7b_path, samples_num=3)
    result = scaffolding_llm.generate(default_prompt)
    assert isinstance(result.output.output_str, str) and len(
        result.output.output_str) > 0, "Output should be a non-empty string"
    scaffolding_llm.shutdown(shutdown_workers=True)
