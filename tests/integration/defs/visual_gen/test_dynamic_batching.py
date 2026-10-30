# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request batching through the public API, worker IPC and native image pipelines."""

import os
import re
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier

import pytest
import requests
import torch
import yaml
from PIL import Image
from test_common.llm_data import get_checkpoint

from defs.common import get_free_port_in_ci
from defs.examples.visual_gen.visual_gen_test_utils import (
    _assert_lpips_below_threshold,
    _golden_media_path,
    _run_lpips_eval,
)
from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "checkpoint,cuda_graph,num_images,golden,steps,compile_enabled,quant_algo,quant_dynamic",
    [
        ("FLUX.1-dev", False, 1, "flux1_lpips_golden.png", 4, False, None, False),
        ("FLUX.1-dev", True, 1, "flux1_lpips_golden.png", 4, False, None, False),
        ("FLUX.1-dev", False, 2, None, 4, False, None, False),
        ("FLUX.2-dev", False, 1, "flux2_lpips_golden.png", 4, False, None, False),
        ("qwen-image", False, 2, None, 4, False, None, False),
        ("qwen-image", True, 2, None, 4, False, None, False),
        ("qwen-image", False, 2, None, 50, False, None, False),
        ("qwen-image", False, 2, None, 50, True, None, False),
        ("qwen-image", True, 1, None, 4, True, None, False),
        ("FLUX.1-dev", False, 1, None, 4, False, "NVFP4", True),
        ("FLUX.2-dev", False, 1, None, 4, False, "NVFP4", True),
        ("FLUX.1-dev", False, 1, None, 4, False, "FP8", True),
        ("FLUX.2-dev", False, 1, None, 4, False, "FP8", True),
        ("FLUX.1-dev-modelopt-NVFP4", False, 1, None, 4, False, "NVFP4", False),
    ],
    ids=[
        "flux1",
        "flux1-cuda-graph",
        "flux1-multiple-outputs",
        "flux2",
        "qwen-image",
        "qwen-image-cuda-graph",
        "qwen-image-50-steps",
        "qwen-image-compile-50-steps",
        "qwen-image-compile-cuda-graph",
        "flux1-dynamic-nvfp4",
        "flux2-dynamic-nvfp4",
        "flux1-dynamic-fp8",
        "flux2-dynamic-fp8",
        "flux1-static-nvfp4",
    ],
)
def test_dynamic_batching_matches_singletons(
    checkpoint,
    cuda_graph,
    num_images,
    golden,
    steps,
    compile_enabled,
    quant_algo,
    quant_dynamic,
    tmp_path,
    capfd,
    monkeypatch,
):
    if quant_algo == "NVFP4" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 requires Blackwell or newer")
    expect_batching = not (quant_algo == "NVFP4" or (quant_dynamic and quant_algo == "FP8"))
    monkeypatch.setenv("TLLM_LOG_LEVEL", "INFO")
    prompts = [
        "a tiny astronaut hatching from an egg on the moon",
        "A red fox resting under an oak tree in autumn, sunlight filtering through golden leaves.",
    ]
    params = [
        VisualGenParams(
            height=256,
            width=256,
            num_inference_steps=steps,
            guidance_scale=3.5,
            seed=seed,
            num_images_per_prompt=num_images,
        )
        for seed in (42, 17)
    ]
    args = VisualGenArgs(
        batching_config={"max_batch_size": 2 * num_images, "max_queue_delay_ms": 100},
        compilation_config={"skip_warmup": True},
        torch_compile_config={"enable": compile_enabled},
        cuda_graph_config={"enable": cuda_graph},
        # Static checkpoints supply their calibrated scales and excluded layers.
        quant_config={"quant_algo": quant_algo, "dynamic": True} if quant_dynamic else None,
    )
    # Serial calls exercise the singleton path on the same weights/device/RNG mode.
    with VisualGen(model=get_checkpoint(checkpoint), args=args) as generator:
        baselines = [generator.generate(prompt, param) for prompt, param in zip(prompts, params)]
        handles = [
            generator.generate_async(prompt, param) for prompt, param in zip(prompts, params)
        ]
        results = [handle.result(timeout=600) for handle in handles]
        assert [out.request_id for out in results] == [handle.request_id for handle in handles]
        if expect_batching:
            assert results[0].metrics.generation == results[1].metrics.generation
        for i, (baseline, result) in enumerate(zip(baselines, results)):
            assert result.image.shape == (num_images, 256, 256, 3)
            assert result.image.dtype == torch.uint8
            assert result.image.count_nonzero() > 0
            baseline.save([tmp_path / f"singleton_{i}_{j}.png" for j in range(num_images)])
            result.save([tmp_path / f"batched_{i}_{j}.png" for j in range(num_images)])

        # A new singleton shape after a batch must remain usable, including graph replay.
        trailing = generator.generate(prompts[0], params[0])
        assert trailing.image.shape == baselines[0].image.shape

    captured = capfd.readouterr()
    log = captured.out + captured.err
    if not expect_batching:
        assert "Dynamic batching is inactive" in log
        assert "Dynamic batch: requests=" not in log
    else:
        assert "Dynamic batch: requests=" in log
    if quant_algo is not None:
        assert f"Quantization: {quant_algo}" in log
        assert f"Dynamic weight quant: {quant_dynamic}" in log
    for i, prompt in enumerate(prompts):
        for j in range(num_images):
            score = _run_lpips_eval(
                tmp_path,
                f"batch_{i}_{j}",
                "image",
                prompt,
                tmp_path / f"singleton_{i}_{j}.png",
                tmp_path / f"batched_{i}_{j}.png",
            )
            _assert_lpips_below_threshold(score, 0.05)
    if golden is not None:
        reference = _golden_media_path(tmp_path, golden, "Dynamic batching golden image")
        score = _run_lpips_eval(
            tmp_path,
            "batch_golden",
            "image",
            prompts[0],
            reference,
            tmp_path / "batched_0_0.png",
        )
        _assert_lpips_below_threshold(score, 0.05)


@contextmanager
def _batching_http_server(tmp_path):
    """Launch the serving example with inexpensive compilation settings for CI."""
    root = Path(__file__).resolve().parents[4]
    example = root / "examples/visual_gen/serve/configs/dynamic_batching.yml"
    config = yaml.safe_load(example.read_text())
    config.update(
        compilation_config={"skip_warmup": True},
        torch_compile_config={"enable": False},
        cuda_graph_config={"enable": False},
    )
    # Allow HTTP/thread scheduling skew on busy CI hosts while preserving the
    # example's batch-size limit and loading its real serving configuration.
    config["batching_config"]["max_queue_delay_ms"] = 1000
    config_path = tmp_path / "server.yml"
    config_path.write_text(yaml.safe_dump(config))
    port = get_free_port_in_ci()
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "server.log"
    env = dict(
        os.environ,
        TLLM_LOG_LEVEL="INFO",
        PYTHONUNBUFFERED="1",
        TRTLLM_MEDIA_STORAGE_PATH=str(tmp_path / "media"),
    )
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                "trtllm-serve",
                get_checkpoint("FLUX.1-dev"),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--visual_gen_args",
                str(config_path),
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text()
                try:
                    if requests.get(f"{url}/health", timeout=5).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(0.2)
            else:
                pytest.fail(f"VisualGen server did not become ready:\n{log_path.read_text()}")
            yield url, log_path
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dynamic_batching_http(tmp_path):
    """The example YAML merges HTTP requests and serves each request's own image."""
    prompts = [
        "a tiny astronaut hatching from an egg on the moon",
        "A red fox resting under an oak tree in autumn, sunlight filtering through golden leaves.",
    ]
    payloads = [
        dict(
            prompt=prompt,
            seed=seed,
            size="256x256",
            num_inference_steps=4,
            guidance_scale=3.5,
            n=1,
            response_format="url",
            format="png",
        )
        for prompt, seed in zip(prompts, (42, 17))
    ]
    with _batching_http_server(tmp_path) as (url, log_path):

        def generate(payload, name, barrier=None):
            if barrier is not None:
                barrier.wait(timeout=30)
            response = requests.post(f"{url}/v1/images/generations", json=payload, timeout=300)
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["size"] == "256x256"
            assert data["output_format"] == "png"
            assert len(data["data"]) == 1
            item = data["data"][0]
            assert item["revised_prompt"] == payload["prompt"]
            assert item["url"].startswith(f"{url}/v1/images/image_")
            content = requests.get(item["url"], timeout=30)
            assert content.status_code == 200, content.text
            path = tmp_path / f"{name}.png"
            path.write_bytes(content.content)
            with Image.open(path) as image:
                assert image.size == (256, 256)
                assert image.convert("RGB").getbbox() is not None
            return path, item["url"]

        baselines = [generate(payload, f"singleton_{i}") for i, payload in enumerate(payloads)]
        log_offset = len(log_path.read_text())
        with ThreadPoolExecutor(max_workers=len(payloads)) as clients:
            barrier = Barrier(len(payloads))
            futures = [
                clients.submit(generate, payload, f"batched_{i}", barrier)
                for i, payload in enumerate(payloads)
            ]
            batched = [future.result(timeout=300) for future in futures]

        batch_log = log_path.read_text()[log_offset:]
        assert re.search(r"Dynamic batch: requests=\(\d+, \d+\), images=2", batch_log), batch_log
        assert len({image_url for _, image_url in baselines + batched}) == 4

    for i, ((baseline, _), (image, _)) in enumerate(zip(baselines, batched)):
        score = _run_lpips_eval(tmp_path, f"http_{i}", "image", prompts[i], baseline, image)
        _assert_lpips_below_threshold(score, 0.05)
