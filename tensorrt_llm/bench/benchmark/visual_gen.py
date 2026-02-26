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

r"""Offline benchmark for VisualGen (image/video generation) models.

Usage:
    trtllm-bench --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --model_path /path/to/checkpoint \
        visual-gen --extra_visual_gen_options config.yaml <benchmark_args>
"""

import json
import os
import time
from datetime import datetime
from typing import Optional

import click

from tensorrt_llm.bench.benchmark.visual_gen_utils import (
    VisualGenBenchmarkMetrics,
    VisualGenRequestOutput,
    build_visual_gen_result_dict,
    calculate_metrics,
    load_visual_gen_prompts,
    print_visual_gen_results,
)
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.logger import logger


def _parse_size(size_str: str) -> tuple[Optional[int], Optional[int]]:
    """Parse WxH size string into (width, height). Returns (None, None) for 'auto'."""
    if size_str.lower() == "auto":
        return None, None
    parts = size_str.lower().split("x")
    if len(parts) != 2:
        raise click.BadParameter(
            f"Size must be 'auto' or WxH format (e.g. 480x832), got '{size_str}'"
        )
    return int(parts[0]), int(parts[1])


@click.command(name="visual-gen", context_settings={"show_default": True})
@click.option(
    "--extra_visual_gen_options",
    type=str,
    default=None,
    help="Path to a YAML file with extra VisualGen model options "
    "(same format as trtllm-serve --extra_visual_gen_options).",
)
@click.option(
    "--prompt",
    type=str,
    default=None,
    help="Single text prompt (repeated --num_prompts times).",
)
@click.option(
    "--prompt_file",
    type=str,
    default=None,
    help="Path to prompt file. Supports plain text (one prompt per line) "
    "or JSONL with 'text'/'prompt' field.",
)
@click.option(
    "--num_prompts",
    type=int,
    default=5,
    help="Number of prompts to benchmark.",
)
@click.option(
    "--size",
    type=str,
    default="auto",
    help="Output resolution in WxH format (e.g. 480x832) or 'auto'.",
)
@click.option(
    "--seconds",
    type=float,
    default=4.0,
    help="Video duration in seconds.",
)
@click.option(
    "--fps",
    type=int,
    default=16,
    help="Frames per second.",
)
@click.option(
    "--num_frames",
    type=int,
    default=None,
    help="Total frames to generate. Overrides --seconds (computed as num_frames / fps).",
)
@click.option(
    "--num_inference_steps",
    type=int,
    default=None,
    help="Number of diffusion denoising steps.",
)
@click.option(
    "--guidance_scale",
    type=float,
    default=None,
    help="Classifier-free guidance scale.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)
@click.option(
    "--negative_prompt",
    type=str,
    default=None,
    help="Negative prompt (concepts to avoid).",
)
@click.option(
    "--max_concurrency",
    type=int,
    default=1,
    help="Maximum concurrent generation requests.",
)
@click.option(
    "--warmup",
    type=int,
    default=1,
    help="Number of warmup requests before benchmarking.",
)
@click.option(
    "--save_result",
    is_flag=True,
    default=False,
    help="Save results to a JSON file.",
)
@click.option(
    "--save_detailed",
    is_flag=True,
    default=False,
    help="Include per-request details (latencies, errors) in saved results.",
)
@click.option(
    "--result_dir",
    type=str,
    default=None,
    help="Directory for result files.",
)
@click.option(
    "--result_filename",
    type=str,
    default=None,
    help="Custom result filename.",
)
@click.option(
    "--metric_percentiles",
    type=str,
    default="50,90,99",
    help="Comma-separated percentile values.",
)
@click.pass_obj
def visual_gen_command(
    bench_env: BenchmarkEnvironment,
    extra_visual_gen_options: Optional[str],
    prompt: Optional[str],
    prompt_file: Optional[str],
    num_prompts: int,
    size: str,
    seconds: float,
    fps: int,
    num_frames: Optional[int],
    num_inference_steps: Optional[int],
    guidance_scale: Optional[float],
    seed: int,
    negative_prompt: Optional[str],
    max_concurrency: int,
    warmup: int,
    save_result: bool,
    save_detailed: bool,
    result_dir: Optional[str],
    result_filename: Optional[str],
    metric_percentiles: str,
) -> None:
    """Benchmark VisualGen (image/video generation) models offline."""
    import yaml

    from tensorrt_llm.commands.utils import get_visual_gen_model_type, get_visual_gen_num_gpus
    from tensorrt_llm.llmapi.visual_gen import VisualGen, VisualGenParams

    if prompt is None and prompt_file is None:
        raise click.UsageError("Either --prompt or --prompt_file must be specified.")
    if prompt is not None and prompt_file is not None:
        raise click.UsageError("--prompt and --prompt_file are mutually exclusive.")

    model = bench_env.model
    model_path = str(bench_env.checkpoint_path or model)

    # Build diffusion config (same pattern as trtllm-serve _serve_visual_gen)
    visual_gen_config: dict = {
        "model": model_path,
        "model_type": get_visual_gen_model_type(model_path),
    }
    if extra_visual_gen_options is not None:
        with open(extra_visual_gen_options, "r") as f:
            visual_gen_extra_args = yaml.safe_load(f) or {}
        visual_gen_config.update(visual_gen_extra_args)

    n_workers = get_visual_gen_num_gpus(visual_gen_config)
    parallel_config = visual_gen_config.get("parallel", {})
    if parallel_config:
        logger.info(f"World size: {n_workers}")
        logger.info(f"CFG size: {parallel_config.get('dit_cfg_size', 1)}")
        logger.info(f"Ulysses size: {parallel_config.get('dit_ulysses_size', 1)}")

    # Parse generation parameters
    width, height = _parse_size(size)
    if num_frames is not None:
        seconds = num_frames / fps
        logger.info(f"Computed seconds={seconds:.3f} from num_frames={num_frames} / fps={fps}")

    gen_params_kwargs: dict = {"seed": seed, "frame_rate": float(fps)}
    if height is not None:
        gen_params_kwargs["height"] = height
    if width is not None:
        gen_params_kwargs["width"] = width
    if num_frames is not None:
        gen_params_kwargs["num_frames"] = num_frames
    if num_inference_steps is not None:
        gen_params_kwargs["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        gen_params_kwargs["guidance_scale"] = guidance_scale

    gen_params = VisualGenParams(**gen_params_kwargs)

    gen_params_for_report = {
        "size": size,
        "seconds": seconds,
        "fps": fps,
    }
    if num_inference_steps is not None:
        gen_params_for_report["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        gen_params_for_report["guidance_scale"] = guidance_scale
    if negative_prompt is not None:
        gen_params_for_report["negative_prompt"] = negative_prompt
    gen_params_for_report["seed"] = seed

    # Load prompts
    input_requests = load_visual_gen_prompts(prompt, prompt_file, num_prompts)
    selected_percentiles = [float(p) for p in metric_percentiles.split(",")]

    # Initialize VisualGen
    logger.info(f"Initializing VisualGen ({model_path})")
    visual_gen = VisualGen(
        model_path=model_path,
        n_workers=n_workers,
        diffusion_config=visual_gen_config,
    )

    try:
        # Warmup
        if warmup > 0:
            logger.info(f"Running {warmup} warmup request(s)...")
            for i in range(warmup):
                warmup_prompt = input_requests[i % len(input_requests)].prompt
                visual_gen.generate(inputs=warmup_prompt, params=gen_params)
            logger.info("Warmup complete.")

        # Main benchmark
        logger.info(
            f"Starting benchmark: {len(input_requests)} requests, max_concurrency={max_concurrency}"
        )

        benchmark_start = time.perf_counter()
        outputs = _run_benchmark(
            visual_gen=visual_gen,
            input_requests=input_requests,
            gen_params=gen_params,
            negative_prompt=negative_prompt,
            max_concurrency=max_concurrency,
        )
        benchmark_duration = time.perf_counter() - benchmark_start

    finally:
        visual_gen.shutdown()

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        selected_percentiles=selected_percentiles,
        num_gpus=n_workers,
    )

    print_visual_gen_results(
        backend="offline",
        model_id=model_path,
        benchmark_duration=benchmark_duration,
        metrics=metrics,
    )

    if save_result:
        _save_results(
            backend="offline",
            model_id=model_path,
            benchmark_duration=benchmark_duration,
            metrics=metrics,
            outputs=outputs,
            gen_params=gen_params_for_report,
            num_prompts=num_prompts,
            max_concurrency=max_concurrency,
            num_gpus=n_workers,
            save_detailed=save_detailed,
            result_dir=result_dir,
            result_filename=result_filename,
        )


def _run_benchmark(
    visual_gen,
    input_requests,
    gen_params,
    negative_prompt: Optional[str],
    max_concurrency: int,
) -> list[VisualGenRequestOutput]:
    """Run the benchmark loop, dispatching requests with concurrency control."""
    import asyncio

    outputs: list[VisualGenRequestOutput] = []

    if max_concurrency <= 1:
        outputs = _run_sequential(visual_gen, input_requests, gen_params, negative_prompt)
    else:
        outputs = asyncio.run(
            _run_concurrent(
                visual_gen,
                input_requests,
                gen_params,
                negative_prompt,
                max_concurrency,
            )
        )

    return outputs


def _run_sequential(
    visual_gen, input_requests, gen_params, negative_prompt
) -> list[VisualGenRequestOutput]:
    """Run requests one at a time, measuring per-request latency."""
    outputs = []

    for req in input_requests:
        output = VisualGenRequestOutput()
        inputs = (
            {"prompt": req.prompt, "negative_prompt": negative_prompt}
            if negative_prompt
            else req.prompt
        )
        st = time.perf_counter()
        try:
            visual_gen.generate(inputs=inputs, params=gen_params)
            output.e2e_latency = time.perf_counter() - st
            output.success = True
        except Exception as e:
            output.e2e_latency = time.perf_counter() - st
            output.success = False
            output.error = str(e)
            output.exception_type = e.__class__.__name__
            logger.error(f"Request failed: {e}")

        outputs.append(output)

    return outputs


async def _run_concurrent(
    visual_gen, input_requests, gen_params, negative_prompt, max_concurrency
) -> list[VisualGenRequestOutput]:
    """Run requests concurrently using generate_async with a semaphore."""
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrency)
    outputs: list[VisualGenRequestOutput] = [VisualGenRequestOutput() for _ in input_requests]

    async def _generate_one(idx, req):
        inputs = (
            {"prompt": req.prompt, "negative_prompt": negative_prompt}
            if negative_prompt
            else req.prompt
        )
        async with semaphore:
            output = outputs[idx]
            st = time.perf_counter()
            try:
                future = visual_gen.generate_async(inputs=inputs, params=gen_params)
                await future.result()
                output.e2e_latency = time.perf_counter() - st
                output.success = True
            except Exception as e:
                output.e2e_latency = time.perf_counter() - st
                output.success = False
                output.error = str(e)
                output.exception_type = e.__class__.__name__
                logger.error(f"Request {idx} failed: {e}")

    tasks = [_generate_one(i, req) for i, req in enumerate(input_requests)]
    await asyncio.gather(*tasks)

    return outputs


def _save_results(
    backend: str,
    model_id: str,
    benchmark_duration: float,
    metrics: VisualGenBenchmarkMetrics,
    outputs: list[VisualGenRequestOutput],
    gen_params: dict,
    num_prompts: int,
    max_concurrency: int,
    num_gpus: int,
    save_detailed: bool,
    result_dir: Optional[str],
    result_filename: Optional[str],
) -> None:
    """Save benchmark results to a JSON file."""
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")

    result_json = build_visual_gen_result_dict(
        backend=backend,
        model_id=model_id,
        benchmark_duration=benchmark_duration,
        metrics=metrics,
        outputs=outputs,
        gen_params=gen_params,
    )

    result_json["date"] = current_dt
    result_json["num_prompts"] = num_prompts
    result_json["max_concurrency"] = max_concurrency
    result_json["num_gpus"] = num_gpus

    if not save_detailed:
        for field_name in ["e2e_latencies", "errors"]:
            result_json.pop(field_name, None)

    base_model_id = model_id.split("/")[-1]
    concurrency_str = f"-concurrency{max_concurrency}" if max_concurrency is not None else ""
    file_name = f"offline{concurrency_str}-{base_model_id}-{current_dt}.json"
    if result_filename:
        file_name = result_filename
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        file_name = os.path.join(result_dir, file_name)

    with open(file_name, "w", encoding="utf-8") as outfile:
        json.dump(result_json, outfile, indent=2)

    print(f"Results saved to: {file_name}")
