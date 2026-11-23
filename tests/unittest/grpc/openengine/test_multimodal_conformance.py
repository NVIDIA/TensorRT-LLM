# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal conformance tests against a real TensorRT-LLM VLM engine."""

import asyncio
import base64
import importlib
from pathlib import Path

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
from tensorrt_llm.llmapi import KvCacheConfig, SchedulerConfig  # noqa: E402

MODEL_NAME = "Qwen3/Qwen3-VL-2B-Instruct"
WHISPER_MODEL_NAME = "openai/whisper-tiny"
_EXPECTED_TRANSCRIPT_FRAGMENT = "thoughts affected hester"

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


def _servicers(vlm):
    inference = OpenEngineInferenceServicer(vlm, MODEL_NAME)
    control = OpenEngineControlServicer(vlm, MODEL_NAME, inference)
    return inference, control


def _model_info(control):
    return asyncio.run(
        control.GetModelInfo(
            model_pb2.GetModelInfoRequest(model=MODEL_NAME),
            FakeServicerContext(),
        )
    )


@pytest.mark.parametrize("input_kind", ["prompt", "token_ids"])
def test_image_capability_agrees_with_generate(vlm, input_kind):
    inference, control = _servicers(vlm)
    info = _model_info(control)

    assert info.supports_multimodal is True
    assert "image" in info.extra["trtllm_supported_modalities"]

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


def test_video_capability_agrees_with_generate(vlm):
    try:
        importlib.import_module("cv2")
    except (ImportError, OSError) as error:
        pytest.fail(f"video conformance requires a working OpenCV install: {error}", pytrace=False)
    inference, control = _servicers(vlm)
    info = _model_info(control)
    assert info.supports_multimodal is True
    assert "video" in info.extra["trtllm_supported_modalities"]

    video_path = (
        Path(__file__).parents[2] / "_torch/visual_gen/test_data/cosmos3_v2v_ref_9f_bframes.mp4"
    )

    video_prompt = (
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        "Describe this video briefly.<|im_end|>\n<|im_start|>assistant\n"
    )
    video_request = generation_pb2.GenerateRequest(
        request_id="openengine-video-conformance",
        model=MODEL_NAME,
        token_ids=generation_pb2.TokenIds(ids=vlm.tokenizer.encode(video_prompt)),
        stopping=generation_pb2.StoppingOptions(max_tokens=4),
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_VIDEO,
                data_uri="data:video/mp4;base64,"
                + base64.b64encode(video_path.read_bytes()).decode(),
                uuid="colors",
            )
        ],
    )

    async def collect_video():
        return [
            response async for response in inference.Generate(video_request, FakeServicerContext())
        ]

    video_responses = asyncio.run(collect_video())
    assert video_responses
    assert not any(response.HasField("error") for response in video_responses)
    video_finished = [response for response in video_responses if response.HasField("finished")]
    assert len(video_finished) == 1
    assert video_finished[0].usage.prompt_tokens > len(video_request.token_ids.ids)


def _whisper_model_path() -> Path:
    root = llm_models_root(check=True)
    candidates = [
        root / "whisper-models/whisper-tiny",
        root / "whisper/whisper-tiny",
        root / "whisper-tiny",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"whisper-tiny not found under {root}")


def _whisper_audio_path(model_path: Path) -> Path:
    root = llm_models_root(check=True)
    filename = "1221-135766-0002.wav"
    candidates = [
        model_path / filename,
        root / f"whisper-models/large-v3/{filename}",
        root / f"whisper-models/large-v2/{filename}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Whisper test audio not found; tried {candidates}")


@pytest.fixture
def whisper():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU to run a real Whisper engine")
    model_path = _whisper_model_path()
    audio_path = _whisper_audio_path(model_path)
    engine = LLM(
        model=str(model_path),
        disable_overlap_scheduler=True,
        enable_chunked_prefill=False,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            free_gpu_memory_fraction=0.2,
            cross_kv_cache_fraction=0.5,
            use_kv_cache_manager_v2=True,
        ),
        max_batch_size=1,
        max_input_len=1500,
        max_num_tokens=1500,
        scheduler_config=SchedulerConfig(use_python_scheduler=True),
    )
    try:
        yield engine, audio_path
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "task_prompt", ["", "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"]
)
def test_audio_capability_agrees_with_generate(whisper, task_prompt):
    whisper, audio_path = whisper
    inference = OpenEngineInferenceServicer(whisper, WHISPER_MODEL_NAME)
    control = OpenEngineControlServicer(whisper, WHISPER_MODEL_NAME, inference)
    info = asyncio.run(
        control.GetModelInfo(
            model_pb2.GetModelInfoRequest(model=WHISPER_MODEL_NAME),
            FakeServicerContext(),
        )
    )

    assert info.supports_multimodal is True
    assert info.supports_text_input is True
    assert info.supports_token_ids_input is False
    assert list(info.extra["trtllm_supported_modalities"]) == ["audio"]

    audio_uri = "data:audio/wav;base64," + base64.b64encode(audio_path.read_bytes()).decode()
    request = generation_pb2.GenerateRequest(
        request_id="openengine-audio-conformance",
        model=WHISPER_MODEL_NAME,
        prompt=task_prompt,
        sampling=generation_pb2.SamplingParams(temperature=0.0),
        stopping=generation_pb2.StoppingOptions(max_tokens=96),
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_AUDIO,
                data_uri=audio_uri,
            )
        ],
    )

    token_request = generation_pb2.GenerateRequest()
    token_request.CopyFrom(request)
    token_request.request_id += "-token-input"
    token_request.token_ids.CopyFrom(generation_pb2.TokenIds(ids=[1]))
    token_context = FakeServicerContext()

    async def reject_token_input():
        return [response async for response in inference.Generate(token_request, token_context)]

    with pytest.raises(AbortError):
        asyncio.run(reject_token_input())
    assert token_context.abort_code == grpc.StatusCode.UNIMPLEMENTED
    assert inference.active_request_count() == 0

    async def collect_audio():
        return [response async for response in inference.Generate(request, FakeServicerContext())]

    responses = asyncio.run(collect_audio())
    assert responses
    assert not any(response.HasField("error") for response in responses)
    token_ids = [
        token.token_id
        for response in responses
        if response.HasField("token")
        for token in response.token.tokens
    ]
    transcript = whisper.tokenizer.decode(token_ids, skip_special_tokens=True).lower()
    assert _EXPECTED_TRANSCRIPT_FRAGMENT in transcript
    finished = [response for response in responses if response.HasField("finished")]
    assert len(finished) == 1
    assert finished[0].usage.completion_tokens == len(token_ids)
