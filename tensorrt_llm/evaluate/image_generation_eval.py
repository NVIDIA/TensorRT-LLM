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
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Sequence

import click
import yaml

from tensorrt_llm.logger import logger

_GENERATION_PARAMS_CONFIG_KEYS = (
    "generation_params",
    "visual_gen_params",
    "extra_visual_gen_options",
)


def _load_yaml_mapping(path: str | None, *, param_hint: str) -> dict[str, Any]:
    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise click.BadParameter(
            f"{param_hint} must contain a YAML mapping at the document root",
            param_hint=param_hint,
        )
    return dict(config)


def _load_prompt_records(
    prompts_path: str,
    num_samples: int | None = None,
) -> tuple[list[str], list[str]]:
    path = Path(prompts_path)
    if not path.exists():
        raise click.BadParameter(f"Prompt file does not exist: {path}", param_hint="--prompts")
    if num_samples is not None and num_samples < 1:
        raise click.BadParameter("--num-samples must be >= 1", param_hint="--num-samples")

    if path.suffix == ".json":
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list):
            raise click.BadParameter(
                "JSON prompt files must contain a list", param_hint="--prompts"
            )
        items = parsed
    else:
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                items.append(json.loads(line))
            else:
                items.append(line)

    prompt_ids: list[str] = []
    prompts: list[str] = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            prompt = item.get("prompt")
            prompt_id = item.get("id", str(index))
        else:
            prompt = item
            prompt_id = str(index)
        if not isinstance(prompt, str) or not prompt:
            raise click.BadParameter(
                "Every prompt entry must contain a non-empty string prompt",
                param_hint="--prompts",
            )
        prompt_ids.append(str(prompt_id))
        prompts.append(prompt)
        if num_samples is not None and len(prompts) >= num_samples:
            break

    if not prompts:
        raise click.BadParameter("Prompt file did not contain any prompts", param_hint="--prompts")
    return prompt_ids, prompts


def _split_generator_config(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    visual_gen_args_config = dict(config)
    generation_params_config: dict[str, Any] = {}

    for key in _GENERATION_PARAMS_CONFIG_KEYS:
        value = visual_gen_args_config.pop(key, None)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise click.BadParameter(f"{key} must be a mapping", param_hint="--config")
        generation_params_config.update(value)

    backend = visual_gen_args_config.pop("backend", None)
    if backend not in (None, "pytorch"):
        raise click.BadParameter(
            "image_generation_eval only supports backend: pytorch for the generator",
            param_hint="--config",
        )

    return visual_gen_args_config, generation_params_config


def _build_visual_generator(model: str, config_path: str | None):
    from tensorrt_llm.visual_gen import VisualGen, VisualGenArgs

    config = _load_yaml_mapping(config_path, param_hint="--config")
    visual_gen_args_config, generation_params_config = _split_generator_config(config)
    visual_gen_args = (
        VisualGenArgs.from_dict(visual_gen_args_config) if visual_gen_args_config else None
    )
    visual_gen = VisualGen(model=model, args=visual_gen_args)
    params = visual_gen.default_params

    for key, value in generation_params_config.items():
        if key not in type(params).model_fields:
            raise click.BadParameter(
                f"Unknown VisualGenParams field in --config: {key}",
                param_hint="--config",
            )
        setattr(params, key, value)

    return visual_gen, params


def _build_image_evaluator(evaluator_model: str, evaluator_options: str | None):
    from tensorrt_llm.visual_gen.generation_evaluation import (
        QwenImageBenchEvaluator,
        QwenImageBenchEvaluatorArgs,
    )

    config = _load_yaml_mapping(evaluator_options, param_hint="--evaluator-options")
    evaluator_type = config.pop("type", "qwen_image_bench")
    if evaluator_type != "qwen_image_bench":
        raise click.BadParameter(
            f"Unsupported image evaluator type: {evaluator_type}",
            param_hint="--evaluator-options",
        )

    embedded_model = config.pop("model", None)
    if embedded_model is not None and str(embedded_model) != evaluator_model:
        raise click.BadParameter(
            "Specify the evaluator model with --evaluator. Do not also set a "
            "different model in --evaluator-options.",
            param_hint="--evaluator-options",
        )

    arg_names = {field.name for field in fields(QwenImageBenchEvaluatorArgs)}
    unknown = sorted(set(config) - arg_names)
    if unknown:
        raise click.BadParameter(
            f"Unknown Qwen Image Bench evaluator options: {unknown}",
            param_hint="--evaluator-options",
        )

    evaluator_args = replace(
        QwenImageBenchEvaluatorArgs(),
        **{key: config[key] for key in config},
    )
    return QwenImageBenchEvaluator(evaluator_model, evaluator_args)


def _save_result_images(response: Any, output_dir: Path) -> dict[int, str]:
    from tensorrt_llm.visual_gen.output import VisualGenOutput

    image_paths: dict[int, str] = {}
    image_dir = output_dir / "generated_images"
    for index, result in enumerate(response.results):
        if result.image is None:
            continue
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{index:04d}.png"
        VisualGenOutput(image=result.image).save(image_path)
        image_paths[index] = str(image_path)
    return image_paths


def _result_to_dict(
    *,
    index: int,
    prompt_id: str,
    result: Any,
    image_path: str | None,
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
        "image_path": image_path,
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


class ImageGenerationEval:
    @click.command("image_generation_eval")
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
        evaluator: str,
        evaluator_options: str | None,
        prompts: str,
        output_dir: str,
        criteria: tuple[str, ...],
        num_samples: int | None,
    ) -> None:
        from tensorrt_llm.visual_gen.generation_evaluation import ImageGenerationEvaluationPipeline

        root_config = ctx.obj if isinstance(ctx.obj, dict) else {}
        model = root_config.get("model")
        if not model:
            raise click.UsageError("image_generation_eval requires root --model.")

        prompt_ids, prompt_texts = _load_prompt_records(prompts, num_samples)
        output_path = Path(output_dir)
        generator = None
        evaluator_runtime = None
        try:
            generator, generation_params = _build_visual_generator(
                model, root_config.get("extra_llm_api_options")
            )
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
                    "generator_config": root_config.get("extra_llm_api_options"),
                    "evaluator_model": evaluator,
                    "evaluator_options": evaluator_options,
                    "criteria": list(criteria) if criteria else None,
                    "num_prompts": len(prompt_texts),
                },
            )
        finally:
            if evaluator_runtime is not None:
                evaluator_runtime.close()
            if generator is not None:
                generator.shutdown()

        logger.info(f"Image generation evaluation results: {summary_path}")
        logger.info(f"Per-sample image generation evaluation results: {jsonl_path}")
        click.echo(f"Aggregate score: {response.aggregate_score}")
        click.echo(f"Results: {summary_path}")
