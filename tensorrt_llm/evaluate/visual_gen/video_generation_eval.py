# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import click

from .config import load_generator_config, load_prompt_records, load_yaml_mapping
from .evaluators import build_video_evaluator, ordered_video_paths, resolve_video_directory
from .generators import build_visual_generator
from .image_generation_eval import _close_runtime


class VideoGenerationEval:
    @click.command("video_generation_eval")
    @click.option(
        "--visual_gen_args",
        "--config",
        type=click.Path(exists=True, dir_okay=False),
        help="Generator YAML containing VisualGen args and generation_params.",
    )
    @click.option(
        "--evaluator",
        type=click.Choice(["lpips"]),
        default="lpips",
        show_default=True,
        help="Video scoring metric.",
    )
    @click.option(
        "--evaluator-options",
        required=True,
        type=click.Path(exists=True, dir_okay=False),
        help="Evaluator YAML: reference_dir and optional generated_dir; files use prompt order (0000.mp4, ...).",
    )
    @click.option(
        "--prompts",
        required=True,
        type=click.Path(exists=True, dir_okay=False),
        help="Prompts as text lines, JSONL records, or a .json list.",
    )
    @click.option(
        "--output-dir",
        required=True,
        type=click.Path(file_okay=False),
        help="Destination for reports and newly generated videos.",
    )
    @click.option(
        "--num-samples", type=click.IntRange(min=1), help="Evaluate only the first N prompts."
    )
    @click.pass_context
    def command(
        ctx: click.Context,
        visual_gen_args: str | None,
        evaluator: str,
        evaluator_options: str,
        prompts: str,
        output_dir: str,
        num_samples: int | None,
    ) -> None:
        root = ctx.obj or {}
        root_config = root.get("extra_llm_api_options")
        if visual_gen_args and root_config and visual_gen_args != root_config:
            raise click.UsageError("Specify generator config only once.")
        generator_config = load_generator_config(
            visual_gen_args or root_config,
            explicit_cli_telemetry=root.get("explicit_cli_telemetry", False),
        )
        ids, texts = load_prompt_records(prompts, num_samples)
        output = Path(output_dir).resolve()
        config = load_yaml_mapping(evaluator_options, param_hint="--evaluator-options")
        base_dir = Path(evaluator_options).resolve().parent
        generated_dir = config.pop("generated_dir", None)
        existing = generated_dir is not None
        if existing:
            media = resolve_video_directory(generated_dir, base_dir, "generated_dir")
            paths = ordered_video_paths(media, len(ids))
        else:
            if not root.get("model"):
                raise click.UsageError("--model is required when generated_dir is omitted.")
            media = output / "generated_videos"
            paths = [media / f"{index:04d}.mp4" for index in range(len(ids))]
        metric = build_video_evaluator(
            evaluator, config, base_dir, ids, output_paths=None if existing else paths
        )
        generator = None
        results = []
        generation_params = None
        try:
            if not existing:
                generator, params = build_visual_generator(root["model"], generator_config)
                if params.seed is None:
                    params.seed = 42
                generation_params = params.model_dump(mode="json", exclude_none=True)
                media.mkdir(parents=True, exist_ok=True)
            output.mkdir(parents=True, exist_ok=True)
            (output / "results.json").unlink(missing_ok=True)
            with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
                for index, (prompt_id, prompt, path) in enumerate(
                    zip(ids, texts, paths, strict=True)
                ):
                    try:
                        if not existing:
                            generated = generator.generate(inputs=[prompt], params=params)
                            if len(generated) != 1:
                                raise ValueError("Expected one generated video per prompt")
                            item = generated[0]
                            if item.error or item.video is None:
                                raise ValueError(item.error or "Generator returned no video")
                            item.save(path)
                        results.append(
                            {
                                "id": prompt_id,
                                "prompt": prompt,
                                "video_path": str(path)
                                if existing
                                else str(path.relative_to(output)),
                                "scores": metric.evaluate(prompt_id, prompt, path),
                            }
                        )
                    except (ValueError, RuntimeError, OSError) as exc:
                        raise click.ClickException(
                            f"Prompt index {index} (id={prompt_id!r}): {exc}"
                        ) from exc
                    stream.write(json.dumps(results[-1]) + "\n")
                    stream.flush()
            aggregate = {
                name: sum(row["scores"][name] for row in results) / len(results)
                for name in results[0]["scores"]
            }
            payload = {
                "evaluator": evaluator,
                "evaluator_options": evaluator_options,
                "generator_model": None if existing else root["model"],
                "generation_params": generation_params,
                "aggregation": "mean",
                "aggregate_scores": aggregate,
                "results": results,
            }
            (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
        finally:
            try:
                if generator is not None:
                    _close_runtime(generator)
            finally:
                metric.close()
        click.echo(f"Aggregate scores: {aggregate}")
        click.echo(f"Results: {output / 'results.json'}")
