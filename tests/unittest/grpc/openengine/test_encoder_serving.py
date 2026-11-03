# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from conftest import FakeServicerContext
from test_encoder import model, native_result, request

from tensorrt_llm.grpc.openengine import servicer

pytestmark = pytest.mark.cpu_only


@pytest.mark.asyncio
async def test_cancelled_encode_reclaims_native_exports(tmp_path, monkeypatch):
    llm, _ = model()
    result, _ = native_result()
    llm.args = SimpleNamespace(mm_encoder_only=True, guided_decoding_backend=None)
    llm.tokenizer = None

    class Handle:
        disaggregated_params = result.disaggregated_params
        finished = True
        error = None
        aborted = False

        async def __aiter__(self):
            yield self

        async def aresult(self):
            return self

        def abort(self):
            self.aborted = True

    handle = Handle()
    llm.generate_async = lambda **kwargs: handle
    monkeypatch.setenv("TRTLLM_OPENENGINE_ENCODER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        servicer,
        "_prepare_input_from_request",
        lambda *args: SimpleNamespace(media=(), inputs="prompt"),
    )
    context = FakeServicerContext()
    context.is_cancelled = True
    req = request()
    req.encoder.encode_only = True
    inference = servicer.OpenEngineInferenceServicer(llm, "qwen")
    assert [value async for value in inference.Generate(req, context)] == []
    assert handle.aborted
    assert handle.disaggregated_params.multimodal_embedding_handles is None
    assert handle.disaggregated_params.mrope_position_ids_handle is None
    assert not list(tmp_path.glob("*.safetensors"))
    assert not inference._encoder_cleanup_tasks
