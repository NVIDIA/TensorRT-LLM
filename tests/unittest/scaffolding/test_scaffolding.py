# autoflake: skip_file

from scaffolding.test_worker import (create_trtllm_worker, default_prompt,
                                     trtllm_model_path)

from tensorrt_llm.scaffolding import NativeGenerationController, ScaffoldingLlm


def create_scaffolding_llm_with_native_generation_controller(trtllm_model_path):
    trtllm_worker = create_trtllm_worker(trtllm_model_path)
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


def test_unbatched_scaffolding_sync(default_prompt, trtllm_model_path):
    scaffolding_llm = create_scaffolding_llm_with_native_generation_controller(
        trtllm_model_path)
    try:
        result = scaffolding_llm.generate(default_prompt)
        assert isinstance(result.outputs[0].text, str) and len(
            result.outputs[0].text) > 0, "Output should be a non-empty string"
    finally:
        scaffolding_llm.shutdown(shutdown_workers=True)


def test_batched_scaffolding_sync(default_prompt, trtllm_model_path):
    scaffolding_llm = create_scaffolding_llm_with_native_generation_controller(
        trtllm_model_path)
    try:
        batch_size = 3
        prompts = [default_prompt] * batch_size
        results = scaffolding_llm.generate(prompts)
        assert len(results) == batch_size
        for result in results:
            assert isinstance(result.outputs[0].text, str) and len(
                result.outputs[0].text
            ) > 0, "Output should be a non-empty string"
    finally:
        scaffolding_llm.shutdown(shutdown_workers=True)


def test_async_scaffolding_generation(default_prompt, trtllm_model_path):

    async def run_async_test():
        scaffolding_llm = create_scaffolding_llm_with_native_generation_controller(
            trtllm_model_path)
        try:
            future = scaffolding_llm.generate_async(default_prompt)
            result = await future.aresult()
            assert isinstance(result.outputs[0].text, str) and len(
                result.outputs[0].text
            ) > 0, "Output should be a non-empty string"
        finally:
            scaffolding_llm.shutdown(shutdown_workers=True)

    import asyncio
    asyncio.run(run_async_test())
