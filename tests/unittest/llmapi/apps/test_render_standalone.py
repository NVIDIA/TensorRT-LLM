# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The standalone renderer: loading without weights, and starting with no GPU."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request

import pytest

from tensorrt_llm.inputs.registry import DefaultInputProcessor
from tensorrt_llm.serve.render import RenderResources, render_chat
from tensorrt_llm.serve.render.__main__ import build_parser

from .render_helpers import chat_request, make_tokenizer, resources

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

TINY_LLAMA_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "vocab_size": 340,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "bos_token_id": 1,
    "eos_token_id": 2,
}

CHAT_BODY = {
    "model": "tiny",
    "messages": [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "this is a test"},
        {"role": "user", "content": "get the weather"},
    ],
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A checkpoint directory with a tokenizer and a config and nothing else."""
    directory = tmp_path_factory.mktemp("tiny_checkpoint")
    make_tokenizer().save_pretrained(directory)
    (directory / "config.json").write_text(json.dumps(TINY_LLAMA_CONFIG))
    return directory


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestLoad:
    def test_the_fixture_has_no_weights(self, checkpoint) -> None:
        names = {path.name for path in checkpoint.iterdir()}
        assert not any(name.endswith((".safetensors", ".bin", ".pt")) for name in names)
        assert "config.json" in names

    def test_load_reads_only_tokenizer_and_config(self, checkpoint) -> None:
        loaded = RenderResources.load(str(checkpoint))

        assert loaded.model_type == "llama"
        assert isinstance(loaded.input_processor, DefaultInputProcessor)
        assert loaded.hf_config is not None
        assert loaded.use_harmony is False

    def test_a_loaded_renderer_matches_one_built_from_the_tokenizer(self, checkpoint) -> None:
        loaded = RenderResources.load(str(checkpoint))
        request = chat_request(add_special_tokens=True)

        from_dir = render_chat(request, loaded)
        from_memory = render_chat(request, resources(make_tokenizer()))

        assert from_dir.token_ids == from_memory.token_ids

    def test_the_server_template_option_is_loaded(self, checkpoint) -> None:
        loaded = RenderResources.load(
            str(checkpoint), chat_template="X{% for m in messages %}{{ m.content }}{% endfor %}"
        )
        assert render_chat(chat_request(), loaded, tokenize=False).text.startswith("X")

    def test_a_missing_checkpoint_is_a_clear_error(self, tmp_path) -> None:
        with pytest.raises(Exception):  # noqa: B017 - the exact type depends on the loader
            RenderResources.load(str(tmp_path / "does-not-exist"))


class TestCommandLine:
    def test_the_options_that_change_the_prompt_use_the_serve_names(self) -> None:
        args = build_parser().parse_args(
            [
                "--model",
                "m",
                "--chat_template",
                "t",
                "--custom_tokenizer",
                "deepseek_v32",
                "--tool_parser",
                "qwen3",
                "--reasoning_parser",
                "qwen3",
                "--trust_remote_code",
                "--enable_tokenization_cache",
                "--allow_request_chat_template",
            ]
        )

        assert args.chat_template == "t"
        assert args.custom_tokenizer == "deepseek_v32"
        assert args.tool_parser == "qwen3"
        assert args.reasoning_parser == "qwen3"
        assert args.trust_remote_code and args.enable_tokenization_cache
        assert args.allow_request_chat_template
        assert args.host == "localhost"

    def test_model_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_a_bad_model_exits_non_zero_with_a_message(self, tmp_path) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tensorrt_llm.serve.render", "--model", str(tmp_path / "nope")],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode != 0
        assert "trtllm-render" in result.stderr


class TestCpuOnlyStartup:
    """The renderer builds with no GPU, no weights and no server or executor object."""

    def test_building_the_app_initializes_no_cuda_and_imports_no_server(self, checkpoint) -> None:
        code = textwrap.dedent(
            """
            import json, sys
            import torch
            from tensorrt_llm.serve.render import RenderResources
            from tensorrt_llm.serve.render._http import build_render_app

            resources = RenderResources.load(sys.argv[1])
            build_render_app(resources)
            servers = ("tensorrt_llm.serve.openai_server", "tensorrt_llm.serve.openai_disagg_server")
            print(json.dumps({
                "cuda_initialized": torch.cuda.is_initialized(),
                "servers_imported": [m for m in servers if m in sys.modules],
            }))
            """
        )
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        result = subprocess.run(
            [sys.executable, "-c", code, str(checkpoint)],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )

        assert result.returncode == 0, result.stderr[-2000:]
        report = json.loads(result.stdout.strip().splitlines()[-1])
        assert report == {"cuda_initialized": False, "servers_imported": []}

    def test_the_process_serves_render_requests_with_no_gpu_visible(self, checkpoint) -> None:
        port = _free_port()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorrt_llm.serve.render",
                "--model",
                str(checkpoint),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log_level",
                "warning",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.time() + 600
            while True:
                assert process.poll() is None, process.stdout.read()[-2000:]
                try:
                    with urllib.request.urlopen(f"{base}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, ConnectionError, OSError):
                    pass
                assert time.time() < deadline, "the renderer did not become healthy in time"
                time.sleep(1)

            request = urllib.request.Request(
                f"{base}/v1/chat/completions/render",
                data=json.dumps(CHAT_BODY).encode(),
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                prepared = json.loads(response.read())
            with urllib.request.urlopen(f"{base}/server_info", timeout=10) as response:
                info = json.loads(response.read())
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()

        expected = render_chat(chat_request(), resources(make_tokenizer())).token_ids
        assert prepared["token_ids"] == expected
        assert info["render_fingerprint"]["digest"] == prepared["fingerprint"]["digest"]
