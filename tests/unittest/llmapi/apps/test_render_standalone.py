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

import ctypes
import fnmatch
import glob
import importlib.machinery
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
import warnings

import pytest

from tensorrt_llm.inputs.registry import DefaultInputProcessor
from tensorrt_llm.serve.render import RenderResources, fingerprints_match, render_chat
from tensorrt_llm.serve.render.__main__ import build_parser

from .render_helpers import chat_request, checkpoint_dir, make_tokenizer, resources

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

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
    """A real checkpoint directory with the tokenizer and config files and no weights."""
    return checkpoint_dir(tmp_path_factory.mktemp("qwen3_checkpoint"))


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

        assert loaded.model_type == "qwen3"
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

    def test_a_router_and_a_worker_that_load_the_same_checkpoint_have_one_fingerprint(
        self, checkpoint
    ) -> None:
        # Both sides go through their real loading code, nothing is a stand-in: the router's
        # own tokenizer loader, and the loader a worker (and this renderer) uses, including
        # AutoProcessor, which returns the tokenizer for a text-only checkpoint. These differed
        # in the template sources and in the tokenizer's init options before they were
        # resolved/loaded the same way, so no router ever matched a real worker.
        from tensorrt_llm.serve.router import KvCacheAwareRouter

        router = KvCacheAwareRouter(
            server_role=None,
            servers=[],
            use_tokens=False,
            max_batch_size=32,
            tokens_per_block=32,
            use_harmony=False,
            tokenizer_dir=str(checkpoint),
            model_path=str(checkpoint),
        )
        worker = RenderResources.load(str(checkpoint))
        # The shape that made a plain comparison of template sources disagree.
        assert worker.processor is not None
        assert worker.processor.chat_template == worker.tokenizer.tokenizer.chat_template

        router_fingerprint = router._render_resources("tiny").fingerprint()

        assert router_fingerprint["compared"]["model_type"] == "qwen3"
        assert fingerprints_match(router_fingerprint, worker.fingerprint())

    def test_a_missing_checkpoint_is_a_clear_error(self, tmp_path) -> None:
        with pytest.raises(Exception):  # noqa: B017 - the exact type depends on the loader
            RenderResources.load(str(tmp_path / "does-not-exist"))

    def test_an_unreadable_model_config_is_refused_not_rendered_generically(self, tmp_path) -> None:
        # A tokenizer and no config: rendering would silently lose the model type (the
        # serving extension, Harmony) and produce a prompt a worker would not.
        tokenizer_only = checkpoint_dir(tmp_path, config=False)

        with pytest.raises(Exception, match="onfig"):
            RenderResources.load(str(tokenizer_only))


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

    def test_the_launcher_loads_a_cuda_stub_only_when_the_driver_is_missing(
        self, monkeypatch
    ) -> None:
        path = shutil.which("trtllm-render")
        assert path is not None, "trtllm-render is not installed"
        # The installed launcher starts the renderer end to end.
        result = subprocess.run(
            [sys.executable, path, "--help"], capture_output=True, text=True, timeout=600
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "--model" in result.stdout

        loader = importlib.machinery.SourceFileLoader("trtllm_render_launcher", path)
        launcher = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(loader.name, loader)
        )
        loader.exec_module(launcher)

        stub = "{}/lib64/stubs/libcuda.so"
        installed = ["/usr/local/cuda-12.4", "/usr/local/cuda-13.0"] + [
            stub.format(root)
            for root in (
                "/usr/local/cuda",
                "/usr/local/cuda-12.4",
                "/usr/local/cuda-13.0",
                "/opt/cuda",
            )
        ]
        driver_present = True
        unloadable = set()
        loaded = []

        def cdll(name, mode=0):
            loaded.append((name, mode))
            if (name == "libcuda.so.1" and not driver_present) or name in unloadable:
                raise OSError(name)

        def fake_glob(pattern):
            depth = len(pattern.split("/"))
            return [
                p for p in installed if len(p.split("/")) == depth and fnmatch.fnmatch(p, pattern)
            ]

        def load():
            loaded.clear()
            return launcher._load_cuda_stub_if_needed()

        monkeypatch.setattr(ctypes, "CDLL", cdll)
        monkeypatch.setattr(glob, "glob", fake_glob)
        monkeypatch.setattr(os.path, "realpath", lambda p: p)
        monkeypatch.delenv("CUDA_HOME", raising=False)

        with warnings.catch_warnings():
            assert load() is None
            assert loaded == [("libcuda.so.1", 0)]

            driver_present = False
            assert load() == stub.format("/usr/local/cuda")
            assert loaded[-1] == (stub.format("/usr/local/cuda"), ctypes.RTLD_GLOBAL)

            monkeypatch.setenv("CUDA_HOME", "/opt/cuda")
            load()
            assert loaded[-1][0] == stub.format("/opt/cuda")

            monkeypatch.delenv("CUDA_HOME")
            unloadable.add(stub.format("/usr/local/cuda"))
            load()
            assert loaded[-1][0] == stub.format("/usr/local/cuda-13.0")

            unloadable.update(p for p in installed if p.endswith("libcuda.so"))
            with pytest.raises(SystemExit, match=r"tried /usr/local/cuda/lib64/stubs/libcuda.so"):
                load()

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
        assert info["render_fingerprint"]["digest"]
        # The identity lives on /server_info only, not on each render response.
        assert "fingerprint" not in prepared
