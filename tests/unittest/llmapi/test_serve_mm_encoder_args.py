# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from tensorrt_llm.commands import serve as serve_cmd


def test_launch_mm_encoder_server_drops_cli_memory_shorthand(monkeypatch):
    captured = {}

    class FakeMultimodalEncoder:
        def __init__(self, **kwargs):
            captured["encoder_kwargs"] = kwargs

    class FakeOpenAIServer:
        def __init__(self, **kwargs):
            captured["server_kwargs"] = kwargs

        def __call__(self, host, port):
            captured["server_call"] = (host, port)
            return "server-call"

    monkeypatch.setattr(serve_cmd, "MultimodalEncoder", FakeMultimodalEncoder)
    monkeypatch.setattr(serve_cmd, "OpenAIServer", FakeOpenAIServer)
    monkeypatch.setattr(
        serve_cmd.asyncio,
        "run",
        lambda server_call: captured.setdefault("asyncio_run_arg", server_call),
    )

    serve_cmd.launch_mm_encoder_server(
        host="127.0.0.1",
        port=12345,
        encoder_args={
            "model": "qwen3-vl-test-model",
            "backend": "pytorch",
            "max_batch_size": 1,
            "max_num_tokens": 8192,
            "build_config": object(),
            "free_gpu_memory_fraction": 0.18,
        },
    )

    assert captured["encoder_kwargs"] == {
        "model": "qwen3-vl-test-model",
        "backend": "pytorch",
        "max_batch_size": 1,
        "max_num_tokens": 8192,
    }
    assert captured["server_kwargs"]["generator"].__class__ is FakeMultimodalEncoder
    assert captured["server_kwargs"]["model"] == "qwen3-vl-test-model"
    assert captured["server_call"] == ("127.0.0.1", 12345)
    assert captured["asyncio_run_arg"] == "server-call"
