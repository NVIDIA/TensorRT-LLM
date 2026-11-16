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
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import click

from tensorrt_llm.logger import logger

from .config import load_prompt_records as _load_prompt_records
from .evaluators import build_image_evaluator as _build_image_evaluator
from .generators import build_visual_generator as _build_visual_generator
from .pipeline import ImageGenerationEvaluationPipeline


def _relative_result_path(path: str | None, output_dir: Path) -> str | None:
    if path is None:
        return None
    image_path = Path(path)
    try:
        return str(image_path.relative_to(output_dir))
    except ValueError:
        return str(image_path)


def _result_to_dict(
    *,
    index: int,
    prompt_id: str,
    result: Any,
    image_path: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "id": prompt_id,
        "index": index,
        "prompt": result.prompt,
        "score": result.score,
        "level1_scores": result.level1_scores,
        "level2_scores": result.level2_scores,
        "level3_scores": result.level3_scores,
        "parse_failures": result.parse_failures,
        "image_path": _relative_result_path(image_path, output_dir),
        "error": result.error,
    }


def _write_outputs(
    *,
    response: Any,
    prompt_ids: Sequence[str],
    output_dir: Path,
    image_paths: dict[int, str],
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        _result_to_dict(
            index=index,
            prompt_id=prompt_ids[index],
            result=result,
            image_path=image_paths.get(index),
            output_dir=output_dir,
        )
        for index, result in enumerate(response.results)
    ]
    summary = {
        "metadata": metadata or {},
        "created": response.created,
        "aggregate_score": response.aggregate_score,
        "aggregation": response.aggregation,
        "timing": response.timing,
        "results": results,
    }

    summary_path = output_dir / "results.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result) + "\n")

    return summary_path, jsonl_path


def _close_runtime(runtime: Any) -> None:
    close = getattr(runtime, "close", None)
    if callable(close):
        close()
        return
    shutdown = getattr(runtime, "shutdown", None)
    if callable(shutdown):
        shutdown()


class ImageGenerationEval:
    @click.command("image_generation_eval")
    @click.option(
        "--visual_gen_args",
        "--config",
        "visual_gen_args",
        type=str,
        default=None,
        help="Optional YAML file with VisualGen generator args. --config is an alias.",
    )
    @click.option(
        "--evaluator",
        required=True,
        type=str,
        help="Image evaluator model path or model identifier.",
    )
    @click.option(
        "--evaluator-options",
        type=click.Path(dir_okay=False, exists=True),
        default=None,
        help="Optional YAML file with evaluator runtime options.",
    )
    @click.option(
        "--prompts",
        required=True,
        type=click.Path(dir_okay=False, exists=True),
        help="Prompt file. Supports JSONL, JSON list, or one prompt per line.",
    )
    @click.option(
        "--output-dir",
        required=True,
        type=click.Path(file_okay=False),
        help="Directory for result JSON and generated image artifacts.",
    )
    @click.option(
        "--criteria",
        multiple=True,
        help="Evaluator criterion/dimension to score. Repeat to select multiple.",
    )
    @click.option(
        "--num-samples",
        type=int,
        default=None,
        help="Optional cap for smoke runs.",
    )
    @click.pass_context
    def command(
        ctx,
        visual_gen_args: str | None,
        evaluator: str,
        evaluator_options: str | None,
        prompts: str,
        output_dir: str,
        criteria: tuple[str, ...],
        num_samples: int | None,
    ) -> None:
        root_config = ctx.obj if isinstance(ctx.obj, dict) else {}
        model = root_config.get("model")
        if not model:
            raise click.UsageError("image_generation_eval requires root --model.")
        root_visual_gen_args = root_config.get("extra_llm_api_options")
        if (
            visual_gen_args is not None
            and root_visual_gen_args is not None
            and visual_gen_args != root_visual_gen_args
        ):
            raise click.UsageError(
                "Specify generator config once. Prefer image_generation_eval "
                "--visual_gen_args; root --config is only a compatibility alias "
                "for this task."
            )
        generator_config = visual_gen_args or root_visual_gen_args

        prompt_ids, prompt_texts = _load_prompt_records(prompts, num_samples)
        output_path = Path(output_dir)
        generator = None
        evaluator_runtime = None
        try:
            generator, generation_params = _build_visual_generator(model, generator_config)
            evaluator_runtime = _build_image_evaluator(evaluator, evaluator_options)
            pipeline = ImageGenerationEvaluationPipeline(generator, evaluator_runtime)
            response = pipeline.run(
                prompt_texts,
                generation_params=generation_params,
                dimensions=list(criteria) if criteria else None,
                image_output_dir=output_path / "generated_images",
            )
            image_paths = {
                index: result.image_path
                for index, result in enumerate(response.results)
                if result.image_path is not None
            }
            summary_path, jsonl_path = _write_outputs(
                response=response,
                prompt_ids=prompt_ids,
                output_dir=output_path,
                image_paths=image_paths,
                metadata={
                    "generator_model": model,
                    "generator_config": generator_config,
                    "evaluator_model": evaluator,
                    "evaluator_options": evaluator_options,
                    "criteria": list(criteria) if criteria else None,
                    "num_prompts": len(prompt_texts),
                },
            )
        finally:
            if evaluator_runtime is not None:
                _close_runtime(evaluator_runtime)
            if generator is not None:
                _close_runtime(generator)

        logger.info(f"Image generation evaluation results: {summary_path}")
        logger.info(f"Per-sample image generation evaluation results: {jsonl_path}")
        click.echo(f"Aggregate score: {response.aggregate_score}")
        click.echo(f"Results: {summary_path}")
