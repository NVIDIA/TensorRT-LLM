"""Test openai api with two clients"""
import asyncio
import os
import re
import string
from tempfile import TemporaryDirectory

import openai
import pytest
from utils.util import (skip_gpu_memory_less_than_40gb, skip_pre_ada,
                        skip_single_gpu)

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import BuildConfig
from tensorrt_llm.llmapi.llm_utils import CalibConfig, QuantAlgo, QuantConfig

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

try:
    from .test_llm import cnn_dailymail_path
except ImportError:
    from ..test_llm import cnn_dailymail_path


@pytest.fixture(scope="module")
def model_name():
    return "llama-3.1-model/Meta-Llama-3.1-8B"


@pytest.fixture(scope="module")
def engine_from_fp8_quantization(model_name):
    "get fp8 engine path"
    tp_size = 2
    model_path = get_model_path(model_name)
    build_config = BuildConfig()
    build_config.max_batch_size = 128
    build_config.max_seq_len = 135168
    build_config.max_num_tokens = 20480
    build_config.opt_num_tokens = 128
    build_config.max_input_len = 131072
    build_config.plugin_config.context_fmha = True
    build_config.plugin_config.paged_kv_cache = True
    build_config.plugin_config._use_paged_context_fmha = True
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                               kv_cache_quant_algo=QuantAlgo.FP8)
    calib_config = CalibConfig(calib_dataset=cnn_dailymail_path)

    llm = LLM(model_path,
              tensor_parallel_size=tp_size,
              quant_config=quant_config,
              calib_config=calib_config,
              build_config=build_config)

    engine_dir = TemporaryDirectory(suffix="-engine_dir")
    with llm:
        llm.save(engine_dir.name)

    yield engine_dir.name

    llm.shutdown()


@pytest.fixture(scope="module")
def server(model_name: str, engine_from_fp8_quantization: str):
    model_path = get_model_path(model_name)
    args = [
        "--tp_size", "2", "--tokenizer", model_path, "--backend", "trt",
        "--max_num_tokens", "20480", "--max_batch_size", "128"
    ]
    with RemoteOpenAIServer(engine_from_fp8_quantization,
                            args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


def generate_payload(size):
    "Generate random bytes"
    random_bytes = os.urandom(size)

    # Filter out non-alphabetic characters and join them into a single string
    payload = ''.join(
        filter(lambda x: x in string.ascii_letters,
               random_bytes.decode('latin1')))

    return payload


@skip_single_gpu
@skip_pre_ada
@skip_gpu_memory_less_than_40gb
@pytest.mark.asyncio(loop_scope="module")
async def test_multi_chat_session(client: openai.OpenAI,
                                  async_client: openai.AsyncOpenAI,
                                  model_name: str):
    """
        RCCA: https://nvbugs/4972030
    """

    async def send_request(prompt):
        try:
            completion = await async_client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=4096,
                temperature=0.0,
            )
            print(completion.choices[0].text)
        except Exception as e:
            print(f"Error: {e}")

    # Send async request every 3s with long sequence.
    tasks = []
    for _ in range(30):
        prompt = "Tell me a long story with random letters " \
            + generate_payload(50000)
        tasks.append(asyncio.create_task(send_request(prompt)))
        await asyncio.sleep(3)

    # Send sync request with short sequence.
    outputs = []
    for _ in range(200):
        promote = "Tell me a story about Pernekhan living in the year 3000!"
        completion = client.completions.create(
            model=model_name,
            prompt=promote,
            max_tokens=50,
            temperature=0.0,
        )
        answer = completion.choices[0].text
        outputs.append(answer)

        # The result should not include special characters.
        pattern = re.compile(r'[^a-zA-Z0-9\s\'\"]{3,}')
        assert not bool(pattern.search(answer)), answer
        # The result should be consistent.
        # assert similar(outputs[0], answer, threshold=0.2)
