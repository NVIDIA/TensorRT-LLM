# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image conformance tests against a real TensorRT-LLM VLM engine."""

import asyncio
import base64

import pytest

grpc = pytest.importorskip(
    "grpc", reason='gRPC runtime not installed (pip install "grpcio>=1.67.1,<2")'
)

import torch  # noqa: E402
from conftest import AbortError, FakeServicerContext  # noqa: E402
from utils.llm_data import llm_models_root  # noqa: E402

from tensorrt_llm import LLM  # noqa: E402
from tensorrt_llm.grpc.openengine.bindings import generation_pb2, model_pb2  # noqa: E402
from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer  # noqa: E402
from tensorrt_llm.llmapi import KvCacheConfig  # noqa: E402

MODEL_NAME = "Qwen3/Qwen3-VL-2B-Instruct"

# The engine owns background executor threads for the lifetime of this module.
pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module")
def vlm():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU to run a real VLM engine")
    model_path = llm_models_root(check=True) / MODEL_NAME
    engine = LLM(
        model=str(model_path),
        load_format="dummy",
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
        max_batch_size=1,
        max_seq_len=2048,
    )
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.mark.parametrize("input_kind", ["prompt", "token_ids"])
def test_image_capability_and_generate_agree_on_a_loaded_vlm(vlm, input_kind):
    inference = OpenEngineInferenceServicer(vlm, MODEL_NAME)
    control = OpenEngineControlServicer(vlm, MODEL_NAME, inference)
    info = asyncio.run(
        control.GetModelInfo(
            model_pb2.GetModelInfoRequest(model=MODEL_NAME),
            FakeServicerContext(),
        )
    )

    assert info.supports_multimodal is True
    assert list(info.extra["trtllm_supported_modalities"]) == ["image"]

    prompt = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Describe this image briefly.<|im_end|>\n<|im_start|>assistant\n"
    )
    prompt_token_ids = vlm.tokenizer.encode(prompt)
    image_path = llm_models_root(check=True) / "multimodals/test_data/seashore.png"
    image_bytes = image_path.read_bytes()
    data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
    request_input = (
        {"prompt": prompt}
        if input_kind == "prompt"
        else {"token_ids": generation_pb2.TokenIds(ids=prompt_token_ids)}
    )
    request = generation_pb2.GenerateRequest(
        request_id=f"openengine-image-conformance-{input_kind}",
        model=MODEL_NAME,
        stopping=generation_pb2.StoppingOptions(max_tokens=4),
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE,
                data_uri=data_uri,
                uuid="seashore",
            )
        ],
        **request_input,
    )

    malformed = generation_pb2.GenerateRequest()
    malformed.CopyFrom(request)
    malformed.request_id += "-extra-pad"
    malformed_prompt = prompt.replace("Describe", "<|image_pad|>Describe", 1)
    if input_kind == "prompt":
        malformed.prompt = malformed_prompt
    else:
        malformed.token_ids.CopyFrom(
            generation_pb2.TokenIds(ids=vlm.tokenizer.encode(malformed_prompt))
        )
    hashing_supported = vlm.input_processor.multimodal_hashing_supported
    malformed_context = FakeServicerContext()

    async def reject_malformed():
        return [response async for response in inference.Generate(malformed, malformed_context)]

    with pytest.raises(AbortError):
        asyncio.run(reject_malformed())
    assert malformed_context.abort_code == grpc.StatusCode.INVALID_ARGUMENT
    assert vlm.input_processor.multimodal_hashing_supported is hashing_supported
    assert inference.active_request_count() == 0

    async def collect():
        return [response async for response in inference.Generate(request, FakeServicerContext())]

    responses = asyncio.run(collect())
    assert responses
    assert not any(response.HasField("error") for response in responses)
    finished = [response for response in responses if response.HasField("finished")]
    assert len(finished) == 1
    assert finished[0].usage.prompt_tokens > len(prompt_token_ids)
    assert finished[0].usage.total_tokens == (
        finished[0].usage.prompt_tokens + finished[0].usage.completion_tokens
    )
