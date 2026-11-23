# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The multimodal encoder route and the exports in the encoder's result.

The result carries the embeddings and the MRoPE positions as exported handles, which only
a consumer that opens them, or ``release_dict``, consumes. The route answers with the
embedding handles alone. The exports here are CPU tensors: each lives in a ``/dev/shm``
file until its handles are consumed.
"""

import asyncio
import base64
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import torch
from test_common.serve.app_factory import AggConfig

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

pytestmark = pytest.mark.cpu_only


def _export(tensor: torch.Tensor) -> tuple[dict, Path]:
    handle = SharedTensorContainer.from_tensor(tensor).dump_to_dict()
    path = Path("/dev/shm") / base64.b64decode(handle["storage_handle"]).decode().lstrip("/")
    return handle, path


@pytest.mark.asyncio
async def test_the_route_delivers_the_embedding_exports_and_releases_the_mrope_exports(serve_env):
    server = serve_env.build_openai_server(AggConfig(role=ServerRole.MM_ENCODER))
    server.model = "test-model"
    server.allow_request_chat_template = False
    server.model_config = None
    server.multimodal_server_config = None
    server.enable_store = False
    server.conversation_store = None
    server.use_harmony = False
    server.tokenizer = None
    server.processor = None
    server.chat_template = None
    server.tool_parser = None
    server.metrics_collector = None
    server.await_disconnected = AsyncMock()
    server._extract_metrics = AsyncMock()

    embedding, embedding_file = _export(torch.ones(2, 1024))
    position_ids, position_ids_file = _export(torch.zeros(3, 1024))
    position_deltas, position_deltas_file = _export(torch.zeros(1024))
    promise = SimpleNamespace(
        request_id="1",
        aresult=AsyncMock(),
        disaggregated_params=DisaggregatedParams(
            multimodal_embedding_handles=[embedding],
            mrope_position_ids_handle=position_ids,
            mrope_position_deltas_handle=position_deltas,
        ),
    )
    server.generator = SimpleNamespace(generate_async=lambda **kwargs: promise)
    request = ChatCompletionRequest(
        model=server.model,
        messages=[{"role": "user", "content": "hi"}],
        prompt_token_ids=[7, 8],
    )
    gc.collect()
    assert all(path.exists() for path in (embedding_file, position_ids_file, position_deltas_file))

    with patch(
        "tensorrt_llm.serve.openai_server.parse_chat_messages_coroutines",
        side_effect=lambda *args, **kwargs: ([], asyncio.sleep(0, result=(None, None)), {}, []),
    ):
        response = await server.openai_mm_encoder(
            request, server._synthetic_request(request.model_dump(exclude_none=True))
        )
    gc.collect()

    assert response.status_code == 200
    assert not position_ids_file.exists()
    assert not position_deltas_file.exists()
    delivered = json.loads(response.body)["choices"][0]["mm_embedding_handle"]
    assert torch.equal(
        SharedTensorContainer.from_dict(delivered).get_local_view(), torch.ones(2, 1024)
    )
