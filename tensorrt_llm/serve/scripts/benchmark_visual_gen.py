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
"""Benchmark online serving throughput for VisualGen (image/video generation).

On the server side, run:
    trtllm-serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --visual_gen_args <config.yaml>

On the client side, run:
    python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --backend openai-videos \
        --prompt "A cat playing in the park" \
        --num-prompts 5 \
        --size 480x832 \
        --num-frames 81 \
        --fps 16 \
        --num-inference-steps 50 \
        --max-concurrency 1 \
        --save-result
"""

import argparse
import asyncio
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from argparse import ArgumentParser as FlexibleArgumentParser
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Optional

import aiohttp
import numpy as np
import yaml
from tqdm.asyncio import tqdm

from tensorrt_llm.bench.benchmark.visual_gen_utils import (
    VisualGenRequestOutput,
    VisualGenSampleRequest,
    build_visual_gen_result_dict,
    calculate_metrics,
    load_visual_gen_prompts,
    print_visual_gen_results,
)
from tensorrt_llm.serve.visual_gen_metrics import (
    SERVER_TIMING_HEADER,
    VISUAL_GEN_DENOISE_TIMING,
    VISUAL_GEN_GENERATION_TIMING,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class VisualGenRequestInput:
    """HTTP request payload for online (server) benchmarking."""

    prompt: str
    api_url: str
    model: str
    size: Optional[str] = None
    seconds: Optional[float] = None
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    extra_body: Optional[dict] = None
    input_reference: Optional[str] = None
    validate_audio: bool = False


def _build_payload_common(request_input: VisualGenRequestInput) -> dict:
    """Build common payload fields shared by image and video generation."""
    payload: dict[str, Any] = {
        "model": request_input.model,
        "prompt": request_input.prompt,
    }
    if request_input.size is not None:
        payload["size"] = request_input.size
    if request_input.num_inference_steps is not None:
        payload["num_inference_steps"] = request_input.num_inference_steps
    if request_input.guidance_scale is not None:
        payload["guidance_scale"] = request_input.guidance_scale
    if request_input.negative_prompt is not None:
        payload["negative_prompt"] = request_input.negative_prompt
    if request_input.seed is not None:
        payload["seed"] = request_input.seed
    if request_input.extra_body:
        payload.update(request_input.extra_body)
    return payload


def _build_image_payload(request_input: VisualGenRequestInput) -> dict[str, Any]:
    """Build the JSON payload for ``/v1/images/generations``."""
    payload = _build_payload_common(request_input)
    payload["response_format"] = "b64_json"
    payload["n"] = 1
    return payload


def _build_video_payload(request_input: VisualGenRequestInput) -> dict[str, Any]:
    """Build the JSON or multipart fields for ``/v1/videos/generations``."""
    payload = _build_payload_common(request_input)
    if request_input.seconds is not None:
        payload["seconds"] = request_input.seconds
    if request_input.fps is not None:
        payload["fps"] = request_input.fps
    if request_input.num_frames is not None:
        payload["num_frames"] = request_input.num_frames
    return payload


def _get_headers(*, json_content: bool) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'unused')}",
    }
    if json_content:
        headers["Content-Type"] = "application/json"
    return headers


def _build_multipart_form(
    payload: dict[str, Any], input_reference: str, media_file: BinaryIO
) -> aiohttp.FormData:
    """Build a multipart request, leaving Content-Type generation to aiohttp."""
    form = aiohttp.FormData()
    for key, value in payload.items():
        if value is None:
            continue
        if key == "extra_params":
            if not isinstance(value, dict):
                raise ValueError("extra_params must be a JSON object")
            field_value = json.dumps(value)
        elif isinstance(value, (dict, list, bool)):
            field_value = json.dumps(value)
        else:
            field_value = str(value)
        form.add_field(key, field_value)
    form.add_field(
        "input_reference",
        media_file,
        filename=Path(input_reference).name,
        content_type="application/octet-stream",
    )
    return form


def _validate_audio_response(response_body: bytes) -> None:
    """Require at least one audio stream in a generated MP4 response."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate audio benchmark responses")

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(response_body)
            temp_path = temp_file.name
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                temp_path,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ValueError(
                "ffprobe could not inspect the generated video response: "
                f"{completed.stderr.strip()}"
            )
        if "audio" not in completed.stdout.split():
            raise ValueError("Generated video response does not contain an audio stream")
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def _parse_server_timing_header(headers: Any) -> dict[str, float]:
    """Parse required VisualGen Server-Timing metrics into seconds.

    Online VisualGen perf sanity gates on engine-side generation time, so a
    successful response without valid ``Server-Timing`` metadata is treated as
    a failed benchmark request instead of silently contributing a zero sample.
    """
    value = headers.get(SERVER_TIMING_HEADER)
    if value is None:
        raise ValueError(f"Missing VisualGen timing response header: {SERVER_TIMING_HEADER}")

    timings = {}
    for entry in value.split(","):
        parts = [part.strip() for part in entry.split(";")]
        name = parts[0]
        for parameter in parts[1:]:
            key, _, parameter_value = parameter.partition("=")
            if key.strip() == "dur":
                timings[name] = float(parameter_value) / 1000.0
                break
    return timings


def _get_server_timing_metric(
    timings: dict[str, float], name: str, *, require_positive: bool
) -> float:
    """Return a required Server-Timing metric, in seconds."""
    if name not in timings:
        raise ValueError(f"Missing VisualGen Server-Timing metric: {name}")
    timing = timings[name]
    if not math.isfinite(timing) or timing < 0 or (require_positive and timing <= 0):
        raise ValueError(f"Invalid VisualGen Server-Timing metric {name}: {timing}")
    return timing


async def _do_post(
    request_input: VisualGenRequestInput,
    payload: dict[str, Any],
    pbar: Optional[tqdm],
    session: Optional[aiohttp.ClientSession],
) -> VisualGenRequestOutput:
    """Execute HTTP POST, measure E2E latency, return output."""
    request_session = session or aiohttp.ClientSession(
        trust_env=True,
        timeout=AIOHTTP_TIMEOUT,
        connector=aiohttp.TCPConnector(limit=0, limit_per_host=0),
    )

    output = VisualGenRequestOutput()
    st = time.perf_counter()
    try:

        async def _consume_response(response: aiohttp.ClientResponse) -> None:
            if response.status == 200:
                response_body = await response.read()
                output.latency = time.perf_counter() - st
                if request_input.validate_audio:
                    await asyncio.to_thread(_validate_audio_response, response_body)
                server_timings = _parse_server_timing_header(response.headers)
                output.generation = _get_server_timing_metric(
                    server_timings,
                    VISUAL_GEN_GENERATION_TIMING,
                    require_positive=True,
                )
                output.denoise = _get_server_timing_metric(
                    server_timings,
                    VISUAL_GEN_DENOISE_TIMING,
                    require_positive=False,
                )
                output.success = True
            else:
                body = await response.text()
                output.error = f"HTTP {response.status}: {body}"
                output.success = False

        if request_input.input_reference is None:
            async with request_session.post(
                url=request_input.api_url,
                json=payload,
                headers=_get_headers(json_content=True),
            ) as response:
                await _consume_response(response)
        else:
            with open(request_input.input_reference, "rb") as media_file:
                form = _build_multipart_form(payload, request_input.input_reference, media_file)
                async with request_session.post(
                    url=request_input.api_url,
                    data=form,
                    headers=_get_headers(json_content=False),
                ) as response:
                    await _consume_response(response)
    except Exception as e:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
        output.exception_type = e.__class__.__name__
    finally:
        if session is None:
            await request_session.close()

    if pbar:
        pbar.update(1)
    return output


async def async_request_image_generation(
    request_input: VisualGenRequestInput,
    pbar: Optional[tqdm] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> VisualGenRequestOutput:
    """POST /v1/images/generations and measure E2E latency."""
    payload = _build_image_payload(request_input)
    return await _do_post(request_input, payload, pbar, session)


async def async_request_video_generation(
    request_input: VisualGenRequestInput,
    pbar: Optional[tqdm] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> VisualGenRequestOutput:
    """POST /v1/videos/sync (sync endpoint) and measure E2E latency."""
    payload = _build_video_payload(request_input)
    return await _do_post(request_input, payload, pbar, session)


VISUAL_GEN_REQUEST_FUNCS = {
    "openai-images": async_request_image_generation,
    "openai-videos": async_request_video_generation,
}


async def get_request(
    input_requests: list[VisualGenSampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[VisualGenSampleRequest, None]:
    """Asynchronously generates requests at a specified rate with optional burstiness."""
    assert burstiness > 0, f"A positive burstiness factor is expected, but given {burstiness}."
    theta = 1.0 / (request_rate * burstiness)
    for request in input_requests:
        yield request
        if request_rate == float("inf"):
            continue
        interval = np.random.gamma(shape=burstiness, scale=theta)
        await asyncio.sleep(interval)


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    input_requests: list[VisualGenSampleRequest],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    selected_percentiles: list[float],
    max_concurrency: Optional[int],
    gen_params: dict[str, Any],
    extra_body: Optional[dict],
    input_reference: Optional[str] = None,
    require_audio: bool = False,
    no_test_input: bool = False,
    request_timeout: float = 6 * 60 * 60,
    num_gpus: int = 1,
) -> dict[str, Any]:
    if backend not in VISUAL_GEN_REQUEST_FUNCS:
        raise ValueError(
            f"Unknown backend: {backend}. Available: {list(VISUAL_GEN_REQUEST_FUNCS.keys())}"
        )

    request_func = VISUAL_GEN_REQUEST_FUNCS[backend]

    def _make_request_input(prompt: str) -> VisualGenRequestInput:
        return VisualGenRequestInput(
            prompt=prompt,
            api_url=api_url,
            model=model_id,
            size=gen_params.get("size"),
            seconds=gen_params.get("seconds"),
            fps=gen_params.get("fps"),
            num_frames=gen_params.get("num_frames"),
            num_inference_steps=gen_params.get("num_inference_steps"),
            guidance_scale=gen_params.get("guidance_scale"),
            negative_prompt=gen_params.get("negative_prompt"),
            seed=gen_params.get("seed"),
            extra_body=extra_body,
            input_reference=input_reference,
            validate_audio=require_audio,
        )

    if not no_test_input:
        print("Starting initial single prompt test run...")
        test_input = _make_request_input(input_requests[0].prompt)
        test_output = await request_func(request_input=test_input)
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark "
                "arguments are correctly specified. "
                f"Error: {test_output.error}"
            )
        else:
            print("Initial test run completed. Starting main benchmark run...")
    else:
        print("Skipping initial test run. Starting main benchmark run...")

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests), desc="Benchmarking")

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(req_input, pbar_ref, sess):
        if semaphore is None:
            return await request_func(request_input=req_input, pbar=pbar_ref, session=sess)
        async with semaphore:
            return await request_func(request_input=req_input, pbar=pbar_ref, session=sess)

    timeout = aiohttp.ClientTimeout(total=request_timeout)
    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    async with aiohttp.ClientSession(
        trust_env=True,
        timeout=timeout,
        connector=aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=True),
    ) as session:
        async for request in get_request(input_requests, request_rate, burstiness):
            request_input = _make_request_input(request.prompt)
            tasks.append(asyncio.create_task(limited_request_func(request_input, pbar, session)))

        outputs: list[VisualGenRequestOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        selected_percentiles=selected_percentiles,
        num_gpus=num_gpus,
    )

    print_visual_gen_results(backend, model_id, benchmark_duration, metrics)

    result = build_visual_gen_result_dict(
        backend=backend,
        model_id=model_id,
        benchmark_duration=benchmark_duration,
        metrics=metrics,
        outputs=outputs,
        gen_params=gen_params,
    )
    denoises = [output.denoise for output in outputs if output.success]
    result.update(_summarize_metric("denoise", denoises, selected_percentiles))
    result["denoises"] = denoises

    num_inference_steps = gen_params.get("num_inference_steps")
    if num_inference_steps is not None:
        seconds_per_step = [denoise / num_inference_steps for denoise in denoises]
        result.update(
            _summarize_metric(
                "seconds_per_denoising_step",
                seconds_per_step,
                selected_percentiles,
            )
        )
        result["seconds_per_denoising_step"] = seconds_per_step
    result["audio_validated"] = require_audio

    _print_trtllm_measurements(result)

    return result


def _summarize_metric(
    name: str, values: list[float], selected_percentiles: list[float]
) -> dict[str, Any]:
    """Return the standard aggregate fields for one request timing."""
    percentiles = {
        f"p{int(percentile) if int(percentile) == percentile else percentile}": (
            float(np.percentile(values, percentile)) if values else 0.0
        )
        for percentile in selected_percentiles
    }
    return {
        f"mean_{name}": float(np.mean(values)) if values else 0.0,
        f"median_{name}": float(np.median(values)) if values else 0.0,
        f"std_{name}": float(np.std(values)) if values else 0.0,
        f"min_{name}": float(np.min(values)) if values else 0.0,
        f"max_{name}": float(np.max(values)) if values else 0.0,
        f"percentiles_{name}": percentiles,
    }


def _print_trtllm_measurements(result: dict[str, Any]) -> None:
    """Print the benchmark measurements used by the VisualGen perf schema."""
    print("{s:{c}^{n}}".format(s=" TRT-LLM Measurements ", n=60, c="="))
    print(f"{'Avg. Diffusion Time (s):':<40} {result['mean_denoise']:<10.4f}")
    print(f"{'Avg. Generation Time (s):':<40} {result['mean_generation']:<10.4f}")
    if "mean_seconds_per_denoising_step" in result:
        print(
            f"{'Avg. Seconds per Denoising Step (s/it):':<40} "
            f"{result['mean_seconds_per_denoising_step']:<10.4f}"
        )
    print(f"{'Request Latency (s):':<40} {result['mean_latency']:<10.4f}")
    print("=" * 60)


def load_prompts(args: argparse.Namespace) -> list[VisualGenSampleRequest]:
    """Load prompts from --prompt or --prompt-file (delegates to shared util)."""
    return load_visual_gen_prompts(args.prompt, args.prompt_file, args.num_prompts)


def _parse_extra_body(raw_value: Optional[str]) -> Optional[dict[str, Any]]:
    """Parse and validate the optional top-level request-body object."""
    if raw_value is None or raw_value == "":
        return None
    try:
        extra_body = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in --extra-body: {exc}") from exc
    if not isinstance(extra_body, dict):
        raise ValueError("--extra-body must be a JSON object")
    extra_params = extra_body.get("extra_params")
    if extra_params is not None and not isinstance(extra_params, dict):
        raise ValueError("--extra-body.extra_params must be a JSON object")
    return extra_body


def _validate_input_reference(path_value: Optional[str]) -> Optional[str]:
    """Validate the conditioning-media path before any benchmark request."""
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise ValueError(f"Input reference file does not exist: {path_value}")
    return str(path.resolve())


def _validate_request_configuration(
    *,
    backend: str,
    model_id: str,
    input_reference: Optional[str],
    extra_body: Optional[dict[str, Any]],
    require_audio: bool,
) -> None:
    """Reject incompatible client request combinations before benchmarking."""
    if input_reference is not None and backend != "openai-videos":
        raise ValueError("--input-reference is supported only by the openai-videos backend")
    if input_reference is not None and extra_body and "input_reference" in extra_body:
        raise ValueError(
            "Specify conditioning media with --input-reference, not both "
            "--input-reference and --extra-body.input_reference"
        )
    if not require_audio:
        return
    if backend != "openai-videos":
        raise ValueError("--require-audio is supported only by the openai-videos backend")
    if "cosmos3-edge" in model_id.lower():
        raise ValueError("Cosmos3-Edge has no audio tower and cannot run an audio benchmark")
    extra_params = (extra_body or {}).get("extra_params")
    if not isinstance(extra_params, dict) or extra_params.get("enable_audio") is not True:
        raise ValueError(
            "--require-audio requires --extra-body with extra_params.enable_audio=true"
        )
    if (extra_body or {}).get("format") != "mp4":
        raise ValueError("--require-audio requires MP4 output (set --extra-body.format='mp4')")
    if shutil.which("ffprobe") is None:
        raise ValueError("--require-audio needs ffprobe to verify the generated audio stream")


def _resolve_num_gpus(args: argparse.Namespace) -> int:
    """Determine the number of GPUs from explicit arg or server config YAML.

    Priority: --num-gpus (explicit) > --visual-gen-args YAML > default 1.
    """
    if args.num_gpus is not None:
        return args.num_gpus

    if args.visual_gen_args is not None:
        with open(args.visual_gen_args, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        from tensorrt_llm.commands.utils import get_visual_gen_num_gpus

        return get_visual_gen_num_gpus(config)

    return 1


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model

    endpoint_map = {
        "openai-images": "/v1/images/generations",
        "openai-videos": "/v1/videos/sync",
    }
    endpoint = args.endpoint or endpoint_map.get(backend)
    if endpoint is None:
        raise ValueError(
            f"Cannot resolve endpoint for backend '{backend}'. "
            "Please specify --endpoint explicitly."
        )

    if args.base_url is not None:
        api_url = f"{args.base_url}{endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{endpoint}"

    input_requests = load_prompts(args)

    gen_params: dict[str, Any] = {}
    if args.size is not None:
        gen_params["size"] = args.size
    if args.seconds is not None:
        gen_params["seconds"] = args.seconds
    if args.fps is not None:
        gen_params["fps"] = args.fps
    if args.num_frames is not None:
        gen_params["num_frames"] = args.num_frames
    if args.num_inference_steps is not None:
        gen_params["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        gen_params["guidance_scale"] = args.guidance_scale
    if args.negative_prompt is not None:
        gen_params["negative_prompt"] = args.negative_prompt
    if args.seed is not None:
        gen_params["seed"] = args.seed

    extra_body = _parse_extra_body(args.extra_body)
    input_reference = _validate_input_reference(args.input_reference)
    _validate_request_configuration(
        backend=backend,
        model_id=model_id,
        input_reference=input_reference,
        extra_body=extra_body,
        require_audio=args.require_audio,
    )

    num_gpus = _resolve_num_gpus(args)

    gc.disable()

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            input_requests=input_requests,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            max_concurrency=args.max_concurrency,
            gen_params=gen_params,
            extra_body=extra_body,
            input_reference=input_reference,
            require_audio=args.require_audio,
            no_test_input=args.no_test_input,
            request_timeout=args.request_timeout,
            num_gpus=num_gpus,
        )
    )

    if args.save_result:
        result_json: dict[str, Any] = {}

        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["num_prompts"] = args.num_prompts
        if input_reference is not None:
            result_json["input_reference"] = Path(input_reference).name

        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    key, value = item.split("=", 1)
                    result_json[key.strip()] = value.strip()
                else:
                    raise ValueError("Invalid metadata format. Please use KEY=VALUE format.")

        result_json = {**result_json, **benchmark_result}

        if not args.save_detailed:
            for field_name in [
                "latencies",
                "generations",
                "denoises",
                "seconds_per_denoising_step",
                "errors",
            ]:
                result_json.pop(field_name, None)

        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency
        result_json["num_gpus"] = num_gpus
        result_json["audio_validated"] = benchmark_result["audio_validated"]

        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}" if args.max_concurrency is not None else ""
        )
        file_name = (
            f"{backend}-{args.request_rate}qps"
            f"{max_concurrency_str}-{base_model_id}"
            f"-{current_dt}.json"
        )
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)

        with open(file_name, "w", encoding="utf-8") as outfile:
            json.dump(result_json, outfile, indent=2)

        print(f"Results saved to: {file_name}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark VisualGen (image/video generation) serving."
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="openai-videos",
        choices=list(VISUAL_GEN_REQUEST_FUNCS.keys()),
        help="Backend API type.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. Wan-AI/Wan2.1-T2V-14B).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--base-url", type=str, default=None, help="Full base URL (overrides --host/--port)."
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="API endpoint path (auto-resolved from backend if not specified).",
    )

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single text prompt (repeated --num-prompts times).",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to prompt file. Supports plain text (one prompt "
        "per line) or JSONL with 'text'/'prompt' field.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=5, help="Number of prompts to benchmark."
    )

    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--size",
        type=str,
        default=None,
        help=(
            "Output resolution in WxH format (e.g. 480x832) or 'auto'. "
            "Omitted uses the checkpoint default."
        ),
    )
    gen_group.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Video duration in seconds. Omitted uses the checkpoint default.",
    )
    gen_group.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second. Omitted uses the checkpoint default.",
    )
    gen_group.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Total frames to generate. Overrides --seconds at the server.",
    )
    gen_group.add_argument(
        "--num-inference-steps", type=int, default=None, help="Number of diffusion denoising steps."
    )
    gen_group.add_argument(
        "--guidance-scale", type=float, default=None, help="Classifier-free guidance scale."
    )
    gen_group.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    gen_group.add_argument(
        "--negative-prompt", type=str, default=None, help="Negative prompt (concepts to avoid)."
    )
    gen_group.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help=(
            "JSON object of extra top-level request fields. Model-specific fields "
            "belong in the nested extra_params object."
        ),
    )
    gen_group.add_argument(
        "--input-reference",
        type=str,
        default=None,
        help="Image or video conditioning file for multipart video requests.",
    )
    gen_group.add_argument(
        "--require-audio",
        action="store_true",
        help="Validate that every MP4 video response contains an audio stream with ffprobe.",
    )

    traffic_group = parser.add_argument_group("Traffic Control")
    traffic_group.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate (req/s). Default inf sends all at once.",
    )
    traffic_group.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor for request generation. 1.0 = Poisson process.",
    )
    traffic_group.add_argument(
        "--max-concurrency", type=int, default=None, help="Maximum concurrent requests."
    )
    traffic_group.add_argument(
        "--request-timeout",
        type=float,
        default=6 * 60 * 60,
        help="Request timeout in seconds (default: 6 hours).",
    )

    parser.add_argument(
        "--visual-gen-args",
        dest="visual_gen_args",
        type=str,
        default=None,
        help="Path to the server config YAML (same file passed to trtllm-serve "
        "via --visual_gen_args). Parallelism settings are read to "
        "automatically determine the number of GPUs.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs used by the server. Overrides the value inferred "
        "from --visual-gen-args. Defaults to 1 if neither is given.",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--save-result", action="store_true", help="Save results to JSON file."
    )
    output_group.add_argument(
        "--save-detailed", action="store_true", help="Include per-request details in saved results."
    )
    output_group.add_argument(
        "--result-dir", type=str, default=None, help="Directory for result files."
    )
    output_group.add_argument(
        "--result-filename", type=str, default=None, help="Custom result filename."
    )
    output_group.add_argument(
        "--metric-percentiles",
        type=str,
        default="50,90,99",
        help="Comma-separated percentile values (default: '50,90,99').",
    )
    output_group.add_argument(
        "--metadata",
        type=str,
        nargs="*",
        default=None,
        help="Key=value pairs to add to result metadata.",
    )

    parser.add_argument("--disable-tqdm", action="store_true", help="Disable progress bar.")
    parser.add_argument(
        "--no-test-input", action="store_true", help="Skip the initial single-prompt test run."
    )

    args = parser.parse_args()

    if args.prompt is None and args.prompt_file is None:
        parser.error("Either --prompt or --prompt-file must be specified.")

    main(args)
