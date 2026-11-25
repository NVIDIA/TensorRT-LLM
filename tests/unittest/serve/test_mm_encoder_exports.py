# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The encoder HTTP response owns its exports until it delivers the body."""

import asyncio
import base64
import gc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from fastapi.testclient import TestClient
from test_common.serve.app_factory import AggConfig

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.disagg_auth import build_internal_disagg_auth_headers
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, to_disaggregated_params

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def encoder(serve_env, monkeypatch):
    server = serve_env.build_openai_server(AggConfig(role=ServerRole.MM_ENCODER))
    server.model = "test-model"
    server.model_config = None
    server.multimodal_server_config = None
    server.tokenizer = server.processor = server.chat_template = None
    server.allow_request_chat_template = False
    server.await_disconnected = AsyncMock()
    server._extract_metrics = AsyncMock()

    def parse(*args, **kwargs):
        return [], asyncio.sleep(0, result=({"image": ["fixture"]}, None)), {}, []

    monkeypatch.setattr("tensorrt_llm.serve.openai_server.parse_chat_messages_coroutines", parse)
    return server


def _completion(encoder, count):
    handles = [
        SharedTensorContainer.from_tensor(torch.full((2, 4), i, dtype=torch.float)).dump_to_dict()
        for i in range(count)
    ]
    paths = [
        Path("/dev/shm") / base64.b64decode(h["storage_handle"]).decode().lstrip("/")
        for h in handles
    ]
    record = {
        "multimodal_hashes": [[i + 1] * 8 for i in range(count)],
        "multimodal_positions": list(range(1, 2 * count, 2)),
        "multimodal_lengths": [2] * count,
        "multimodal_uuids": [f"item-{i}" for i in range(count)],
        "multimodal_item_run_cu_offsets": list(range(count + 1)),
        "multimodal_run_positions": list(range(1, 2 * count, 2)),
        "multimodal_run_lengths": [2] * count,
    }
    ids = [7] + [42] * (2 * count) + [8]
    promise = SimpleNamespace(
        request_id=1,
        prompt_token_ids=ids,
        aresult=AsyncMock(),
        abort=MagicMock(),
        disaggregated_params=DisaggregatedParams(
            multimodal_embedding_handles=handles,
            multimodal_input=record,
            mrope_position_ids_handle=SharedTensorContainer.from_tensor_bytes(
                torch.zeros(3, 1, len(ids), dtype=torch.int64)
            ).dump_to_dict(),
            mrope_position_deltas_handle=SharedTensorContainer.from_tensor_bytes(
                torch.zeros(1, 1, dtype=torch.int32)
            ).dump_to_dict(),
        ),
    )
    encoder.generator.generate_async = MagicMock(return_value=promise)
    gc.collect()
    assert all(path.exists() for path in paths)
    return promise, paths


def _request(encoder):
    return ChatCompletionRequest(
        model=encoder.model,
        messages=[{"role": "user", "content": "hi"}],
        prompt_token_ids=[7, 8],
        truncate_prompt_tokens=32,
        add_special_tokens=False,
        mm_processor_kwargs={"do_normalize": False},
    )


@pytest.mark.parametrize(("count", "render_prompt"), [(1, False), (2, True)])
def test_encoder_delivers_prompt_record_and_exports(encoder, count, render_prompt, monkeypatch):
    promise, paths = _completion(encoder, count)
    request = _request(encoder)
    if render_prompt:
        request.prompt_token_ids = None

        async def render(resources, mm_coroutines, **kwargs):
            assert kwargs["truncate_prompt_tokens"] == request.truncate_prompt_tokens
            return SimpleNamespace(text="rendered"), await mm_coroutines

        monkeypatch.setattr(
            "tensorrt_llm.serve.openai_server.render_conversation_with_media", render
        )
    response = TestClient(encoder.app).post(
        "/v1/chat/completions", json=request.model_dump(exclude_none=True)
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prompt_token_ids"] == promise.prompt_token_ids
    choice = body["choices"][0]
    params = choice["disaggregated_params"]
    original = promise.disaggregated_params
    assert params["multimodal_input"] == original.multimodal_input
    assert params["mrope_position_ids_handle"] == original.mrope_position_ids_handle
    assert params["mrope_position_deltas_handle"] == original.mrope_position_deltas_handle
    assert params["multimodal_embedding_handles"] == original.multimodal_embedding_handles
    if count == 1:
        assert choice["mm_embedding_handle"] == params["multimodal_embedding_handles"][0]
    submitted = encoder.generator.generate_async.call_args.kwargs
    assert submitted["inputs"]["mm_processor_kwargs"] == request.mm_processor_kwargs
    assert submitted["inputs"].get("chat_template_applied", False) == render_prompt
    sampling_params = submitted["sampling_params"]
    assert sampling_params.truncate_prompt_tokens == 32 and not sampling_params.add_special_tokens
    for i, handle in enumerate(params["multimodal_embedding_handles"]):
        assert torch.equal(
            SharedTensorContainer.from_dict(handle).get_local_view(), torch.full((2, 4), float(i))
        )
    gc.collect()
    assert not any(path.exists() for path in paths)


def test_encoder_pruned_video_retains_handle_only_reply(encoder):
    promise, paths = _completion(encoder, 1)
    handle = promise.disaggregated_params.multimodal_embedding_handles[0]
    handle["metadata"] = {"retained_token_counts": [1, 1]}
    legacy_position = SharedTensorContainer.from_tensor(torch.ones(1)).dump_to_dict()
    position_path = Path("/dev/shm") / base64.b64decode(
        legacy_position["storage_handle"]
    ).decode().lstrip("/")
    promise.disaggregated_params.mrope_position_ids_handle = legacy_position
    response = TestClient(encoder.app).post(
        "/v1/chat/completions", json=_request(encoder).model_dump(exclude_none=True)
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("prompt_token_ids") is None
    choice = body["choices"][0]
    assert choice.get("disaggregated_params") is None
    assert choice["mm_embedding_handle"] == handle
    SharedTensorContainer.release_dict(handle)
    gc.collect()
    assert not paths[0].exists() and not position_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("drop", ["shape", "validation", "metrics", "cancel", "disconnect", "send"])
async def test_encoder_releases_every_undelivered_export(encoder, drop):
    promise, paths = _completion(encoder, 2)
    request = _request(encoder)
    raw_request = encoder._synthetic_request(request.model_dump(exclude_none=True))
    if drop == "shape":
        del promise.disaggregated_params.multimodal_embedding_handles[0]["tensor_size"]
    elif drop == "validation":
        promise.disaggregated_params.multimodal_input["multimodal_lengths"] = []
    elif drop in ("metrics", "cancel"):
        encoder._extract_metrics.side_effect = (
            asyncio.CancelledError() if drop == "cancel" else ValueError("metrics failed")
        )
    if drop == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await encoder.openai_mm_encoder(request, raw_request)
    else:
        response = await encoder.openai_mm_encoder(request, raw_request)
        if drop in ("disconnect", "send"):
            assert response.status_code == 200
            if drop == "disconnect":
                raw_request.is_disconnected = AsyncMock(return_value=True)
                send = AsyncMock()
                await response(raw_request.scope, raw_request.receive, send)
                send.assert_not_awaited()
            else:
                send = AsyncMock(side_effect=[None, OSError("disconnected")])
                with pytest.raises(OSError, match="disconnected"):
                    await response(raw_request.scope, raw_request.receive, send)
        else:
            assert response.status_code in (400, 500)
    gc.collect()
    assert not any(path.exists() for path in paths)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["openai_mm_encoder", "chat_harmony"])
async def test_wrong_worker_retires_only_authenticated_exports(encoder, route):
    promise, paths = _completion(encoder, 1)
    encoder._internal_disagg_auth_key = "test-secret"
    request = _request(encoder)
    request.disaggregated_params = to_disaggregated_params(
        promise.disaggregated_params, encoder=True
    )
    request.disaggregated_params.request_type = "context_and_generation"
    raw_request = encoder._synthetic_request(request.model_dump(exclude_none=True))
    handler = getattr(encoder, route)
    assert (await handler(request, raw_request)).status_code == 401
    assert paths[0].exists()
    headers = build_internal_disagg_auth_headers("test-secret", request)
    raw_request = encoder._synthetic_request(request.model_dump(exclude_none=True))
    raw_request.scope["headers"].extend((k.encode(), v.encode()) for k, v in headers.items())
    assert (await handler(request, raw_request)).status_code == 400
    gc.collect()
    assert not paths[0].exists()


@pytest.mark.asyncio
async def test_cancel_while_executor_publishes_result(encoder):
    promise, paths = _completion(encoder, 2)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def complete():
        started.set()
        await cancelled.wait()

    promise.aresult = complete
    promise.abort.side_effect = cancelled.set
    request = _request(encoder)
    raw_request = encoder._synthetic_request(request.model_dump(exclude_none=True))
    handler = asyncio.create_task(encoder.openai_mm_encoder(request, raw_request))
    await asyncio.wait_for(started.wait(), timeout=3)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(handler, timeout=3)
    promise.abort.assert_called_once()
    gc.collect()
    assert not any(path.exists() for path in paths)
