import asyncio
import multiprocessing
import os
import sys
import time
import unittest

import httpx
import pytest
import torch
import uvicorn
from fastapi import FastAPI

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.external_api import APIDrafter
from tensorrt_llm.llmapi import (CudaGraphConfig, ExternalAPIConfig,
                                 KvCacheConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import similar

PORT = 8001
DEFAULT_DRAFT_TOKENS = [0, 1, 2]


def create_server():
    app = FastAPI()

    @app.post("/generate")
    async def repeat_draft_tokens(request: dict):
        draft_tokens = request["prefix"]
        if "extra_token" in request:
            draft_tokens.append(request["extra_token"])
        return {
            "draft_tokens": DEFAULT_DRAFT_TOKENS,
            "draft_tokens_2": draft_tokens,
        }

    @app.post("/generate_nested")
    async def generate_nested_response(request: dict):
        return {
            "data": {
                "predictions": {
                    "tokens": DEFAULT_DRAFT_TOKENS
                }
            },
            "nested_list": [
                {
                    "tokens": DEFAULT_DRAFT_TOKENS
                },
            ]
        }

    @app.post("/generate_wrong")
    async def generate_wrong_tokens(request: dict):
        draft_tokens = "Hello world!"
        return {
            "draft_tokens": draft_tokens,
        }

    @app.post("/generate_none")
    async def generate_none_tokens(request: dict):
        return {
            "draft_tokens": [],
        }

    uvicorn.run(app, host="0.0.0.0", port=PORT)


@pytest.fixture(scope="module")
def setup_server():
    process = multiprocessing.Process(target=create_server, daemon=True)
    process.start()
    # wait for server to start
    count = 0
    while count < 10:
        try:
            # check if server is running successfully
            response = httpx.post(f"http://localhost:{PORT}/generate",
                                  json={"prefix": [1, 2, 3]})
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(0.5)
        count += 0.5
    # wait for tests to run
    yield
    process.terminate()


@pytest.mark.parametrize(
    "disable_overlap_scheduler,use_cuda_graph,attn_backend",
    [[True, False, "TRTLLM"], [True, True, "TRTLLM"],
     [True, False, "FLASHINFER"]])
def test_llama_user_provided(setup_server, disable_overlap_scheduler: bool,
                             use_cuda_graph: bool, attn_backend: str):

    max_batch_size = 2
    max_draft_len = 4

    # endpoint is required to be a non-null value
    with pytest.raises(Exception):
        spec_config = ExternalAPIConfig(max_draft_len=max_draft_len)

    # test that endpoint can be hit successfully
    extra_token = 3
    custom_prefix = [4, 5, 6]
    get_draft = lambda drafter: asyncio.run(
        drafter.get_draft_tokens(prefix=custom_prefix,
                                 request_id=0,
                                 end_id=0,
                                 max_sequence_length=max_draft_len))

    # no template, no response field
    spec_config = ExternalAPIConfig(
        max_draft_len=max_draft_len,
        endpoint=f"http://localhost:{PORT}/generate")
    spec_drafter = APIDrafter(spec_config)
    draft_tokens = get_draft(spec_drafter)
    assert draft_tokens == DEFAULT_DRAFT_TOKENS

    # with template, no response field
    spec_config = ExternalAPIConfig(
        max_draft_len=max_draft_len,
        endpoint=f"http://localhost:{PORT}/generate",
        template={
            "extra_token": extra_token,
        })
    spec_drafter = APIDrafter(spec_config)
    draft_tokens = get_draft(spec_drafter)
    assert draft_tokens == DEFAULT_DRAFT_TOKENS

    # no template, with response field
    spec_config = ExternalAPIConfig(
        max_draft_len=max_draft_len,
        endpoint=f"http://localhost:{PORT}/generate",
        response_field="draft_tokens_2")
    spec_drafter = APIDrafter(spec_config)
    draft_tokens = get_draft(spec_drafter)
    assert draft_tokens == custom_prefix

    # with template, with response field
    spec_config = ExternalAPIConfig(
        max_draft_len=max_draft_len,
        endpoint=f"http://localhost:{PORT}/generate",
        template={
            "extra_token": extra_token,
        },
        response_field="draft_tokens_2")
    spec_drafter = APIDrafter(spec_config)
    draft_tokens = get_draft(spec_drafter)
    assert draft_tokens == custom_prefix + [extra_token]

    # test correct drafting length (max_draft_len = 4)
    draft_tokens = asyncio.run(
        spec_drafter.get_draft_tokens(prefix=[0, 1, 2, 3, 4, 5, 6],
                                      request_id=0,
                                      end_id=0,
                                      max_sequence_length=max_draft_len))
    assert draft_tokens == [0, 1, 2, 3]

    # test nested response field
    spec_config = ExternalAPIConfig(
        max_draft_len=max_draft_len,
        endpoint=f"http://localhost:{PORT}/generate_nested",
        response_field="data.predictions.tokens")
    spec_drafter = APIDrafter(spec_config)
    draft_tokens = get_draft(spec_drafter)
    assert draft_tokens == DEFAULT_DRAFT_TOKENS

    spec_config = ExternalAPIConfig(
        max_draft_len=max_draft_len,
        endpoint=f"http://localhost:{PORT}/generate_nested",
        response_field="nested_list.0.tokens")
    spec_drafter = APIDrafter(spec_config)
    draft_tokens = get_draft(spec_drafter)
    assert draft_tokens == DEFAULT_DRAFT_TOKENS

    spec_config = ExternalAPIConfig(
        max_draft_len=max_draft_len,
        endpoint=f"http://localhost:{PORT}/generate_nested",
        response_field="data.predictions.wrong")
    spec_drafter = APIDrafter(spec_config)
    draft_tokens = get_draft(spec_drafter)
    assert draft_tokens == []

    spec_config = ExternalAPIConfig(
        max_draft_len=max_draft_len,
        endpoint=f"http://localhost:{PORT}/generate_nested",
        response_field="nested_list.3.tokens")
    spec_drafter = APIDrafter(spec_config)
    draft_tokens = get_draft(spec_drafter)
    assert draft_tokens == []

    # test wrong response field type
    spec_config = ExternalAPIConfig(
        max_draft_len=max_draft_len,
        endpoint=f"http://localhost:{PORT}/generate_wrong",
    )
    spec_drafter = APIDrafter(spec_config)
    draft_tokens = get_draft(spec_drafter)
    assert draft_tokens == []

    # test non-existent endpoint
    spec_config = ExternalAPIConfig(
        max_draft_len=max_draft_len,
        endpoint=f"http://localhost:{PORT}/nonexistent",
    )
    spec_drafter = APIDrafter(spec_config)
    draft_tokens = get_draft(spec_drafter)
    assert draft_tokens == []

    # spec dec correctness test
    # no draft tokens generated, so should be identical to target
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 20:
        pytest.skip("Not enough memory to load target model")

    kv_cache_config = KvCacheConfig(enable_block_reuse=False)
    cuda_graph_config = CudaGraphConfig(
        batch_sizes=[1]) if use_cuda_graph else None

    llm_common_config = dict( \
        model=llm_models_root() / "llama-3.1-model" /"Meta-Llama-3.1-8B",
        backend='pytorch',
        attn_backend=attn_backend,
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_num_tokens=2048,
    )
    prompts = [
        "The capital of France is",
        "The president of the United States is",
    ]

    spec_config = ExternalAPIConfig(
        max_draft_len=max_draft_len,
        endpoint=f"http://localhost:{PORT}/generate_none",
    )
    sampling_params = SamplingParams(max_tokens=32)

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        # The spec decode algorithm currently guarantees identical results
        assert similar(text_spec, text_ref)


if __name__ == "__main__":
    unittest.main()
