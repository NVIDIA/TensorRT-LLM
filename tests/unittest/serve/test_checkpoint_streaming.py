# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint ingress retains real streamed handlers through disconnect cleanup."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import create_autospec, patch

import pytest
from test_common.serve.app_factory import AggConfig, isolated_serve_env
from test_common.serve.fakes import ScriptedOutput, ScriptedStep

from tensorrt_llm.executor.checkpoint import CheckpointBackend, CheckpointProfile
from tensorrt_llm.serve.checkpoint import CheckpointIngress

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("disconnect", [False, True])
def test_stream_retirement_precedes_checkpoint_drain(tmp_path, disconnect):
    backend = create_autospec(CheckpointBackend, instance=True, spec_set=True)
    backend.describe.return_value = CheckpointProfile(
        profile_digest="a" * 64, cohort_digest="b" * 64, ranks=[0]
    )
    ingress = CheckpointIngress()
    boundary = SimpleNamespace(
        ingress=ingress,
        describe=backend.describe,
        step=backend.step,
        generate=backend.generate,
        validate_probe=backend.validate_probe,
        fail_closed=backend.fail_closed,
    )
    with (
        isolated_serve_env(tmp_path) as env,
        patch("tensorrt_llm.serve.checkpoint.ExecutorCheckpointBackend", return_value=boundary),
    ):
        server = env.build_openai_server(AggConfig(checkpoint=True))
        server.generator.script(
            ScriptedOutput(
                steps=(ScriptedStep(text="hello", token_ids=(6,)),), hold_after_steps=True
            )
        )

        async def scenario():
            started = asyncio.Event()
            disconnected = asyncio.Event()
            delivered_body = False
            body = json.dumps(
                {
                    "model": "fake-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
            ).encode()

            async def receive():
                nonlocal delivered_body
                if not delivered_body:
                    delivered_body = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await disconnected.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.body" and message.get("body"):
                    started.set()

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/chat/completions",
                "raw_path": b"/v1/chat/completions",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1234),
                "server": ("testserver", 80),
                "state": {},
            }
            task = asyncio.create_task(server.app(scope, receive, send))
            try:
                await asyncio.wait_for(started.wait(), 3)
                assert server.generator.generate_calls
                assert ingress.active == 1
                server.checkpoint_lifecycle._cancelled.set()
                with pytest.raises(TimeoutError, match="not drained"):
                    await asyncio.to_thread(ingress.wait_drained, 0.01)
                if disconnect:
                    disconnected.set()
                else:
                    server.generator.generate_calls[0].result.release()
                await asyncio.wait_for(task, 3)
                assert ingress.active == 0
                ingress.wait_drained(0)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        asyncio.run(scenario())
