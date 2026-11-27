# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any

import click

from .config import load_generator_config, load_prompt_records, load_yaml_mapping
from .evaluators import (
    build_video_evaluator,
    ordered_video_paths,
    resolve_video_directory,
    validate_video_evaluator_options,
)
from .generators import build_visual_generator
from .image_generation_eval import _close_runtime


def _prepare_params(params: Any, evaluator: str) -> None:
    if evaluator == "rmse":
        params.extra_params = {**(params.extra_params or {}), "output_type": "latent"}
    if params.seed is None:
        params.seed = 42


def _generate_artifact(
    generator: Any, params: Any, prompt: str, path: Path, evaluator: str
) -> None:
    generated = generator.generate(inputs=[prompt], params=params)
    if len(generated) != 1:
        raise ValueError("Expected one generated video per prompt")
    item = generated[0]
    if item.error or item.video is None:
        raise ValueError(item.error or "Generator returned no video")
    if evaluator == "rmse":
        from .rmse import save_latents

        save_latents(item, path)
    else:
        item.save(path)


def _generate_references(
    model: str, config: dict, evaluator: str, ids: list[str], prompts: list[str], paths: list[Path]
) -> dict:
    generator, params = build_visual_generator(model, config)
    try:
        _prepare_params(params, evaluator)
        for index, (prompt_id, prompt, path) in enumerate(zip(ids, prompts, paths, strict=True)):
            try:
                _generate_artifact(generator, params, prompt, path, evaluator)
            except (ValueError, RuntimeError, OSError) as exc:
                raise click.ClickException(
                    f"Reference prompt index {index} (id={prompt_id!r}): {exc}"
                ) from exc
        return params.model_dump(mode="json", exclude_none=True)
    finally:
        _close_runtime(generator)


class VideoGenerationEval:
    @click.command("video_generation_eval")
    @click.option(
        "--visual_gen_args",
        "--config",
        type=click.Path(exists=True, dir_okay=False),
        help="Generator YAML containing VisualGen args and generation_params.",
    )
    @click.option(
        "--reference-config",
        type=click.Path(exists=True, dir_okay=False),
        help="Generate references with this VisualGen YAML; optional model defaults to --model.",
    )
    @click.option(
        "--media-dir",
        type=click.Path(file_okay=False),
        help=(
            "Retain both generations in a new subdirectory here; otherwise use temporary storage. "
            "Requires --reference-config."
        ),
    )
    @click.option(
        "--evaluator",
        type=click.Choice(["lpips", "rmse"]),
        default="lpips",
        show_default=True,
        help="Video scoring metric.",
    )
    @click.option(
        "--evaluator-options",
        type=click.Path(exists=True, dir_okay=False),
        help=(
            "Evaluator YAML: reference_dir and optional generated_dir; files use prompt order "
            "(0000.mp4 for LPIPS, 0000.safetensors for latent RMSE)."
        ),
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
        reference_config: str | None,
        media_dir: str | None,
        evaluator: str,
        evaluator_options: str | None,
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
        if len(set(ids)) != len(ids):
            raise click.BadParameter("Prompt IDs must be unique", param_hint="--prompts")
        output = Path(output_dir).resolve()
        config = load_yaml_mapping(evaluator_options, param_hint="--evaluator-options")
        base_dir = Path(evaluator_options).resolve().parent if evaluator_options else Path.cwd()
        if media_dir == "":
            raise click.BadParameter("Must be a non-empty directory path", param_hint="--media-dir")
        if media_dir and not reference_config:
            raise click.UsageError("--media-dir requires --reference-config.")
        if reference_config and ({"reference_dir", "generated_dir"} & config.keys()):
            raise click.UsageError(
                "--reference-config cannot be combined with reference_dir or generated_dir."
            )
        generated_dir = config.pop("generated_dir", None)
        validate_video_evaluator_options(evaluator, config)
        existing = generated_dir is not None
        suffix = ".safetensors" if evaluator == "rmse" else ".mp4"
        reference_model = None
        reference_params = None
        reference_paths = None
        artifacts = None
        if reference_config:
            reference_options = load_generator_config(
                reference_config,
                explicit_cli_telemetry=root.get("explicit_cli_telemetry", False),
            )
            reference_model = reference_options.pop("model", root.get("model"))
            if not isinstance(reference_model, str) or not reference_model:
                raise click.UsageError("Set --model or model in --reference-config.")
            if not root.get("model"):
                raise click.UsageError("--model is required for target generation.")
            if media_dir:
                parent = Path(media_dir).expanduser().resolve()
                parent.mkdir(parents=True, exist_ok=True)
                artifacts = Path(mkdtemp(prefix="trtllm-eval-", dir=parent))
            else:
                artifacts = Path(ctx.with_resource(TemporaryDirectory(prefix="trtllm-eval-")))
            references = artifacts / "reference"
            references.mkdir()
            reference_paths = [references / f"{index:04d}{suffix}" for index in range(len(ids))]
            reference_params = _generate_references(
                reference_model, reference_options, evaluator, ids, texts, reference_paths
            )
            config["reference_dir"] = str(references)
            click.echo(
                f"Intermediate artifacts: {artifacts} ({'retained' if media_dir else 'temporary'})"
            )
        if existing:
            media = resolve_video_directory(generated_dir, base_dir, "generated_dir")
            paths = ordered_video_paths(media, len(ids), suffix)
        else:
            if not root.get("model"):
                raise click.UsageError("--model is required when generated_dir is omitted.")
            media = (
                (artifacts / "target")
                if artifacts
                else output / ("generated_latents" if evaluator == "rmse" else "generated_videos")
            )
            paths = [media / f"{index:04d}{suffix}" for index in range(len(ids))]
        metric = build_video_evaluator(
            evaluator, config, base_dir, ids, output_paths=None if existing else paths
        )
        generator = None
        results = []
        generation_params = None
        try:
            if not existing:
                generator, params = build_visual_generator(root["model"], generator_config)
                _prepare_params(params, evaluator)
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
                            _generate_artifact(generator, params, prompt, path, evaluator)
                        scores = metric.evaluate(prompt_id, prompt, path)
                        if results and scores.keys() != results[0]["scores"].keys():
                            raise ValueError("Scored modalities must match across all prompts")
                        results.append(
                            {
                                "id": prompt_id,
                                "prompt": prompt,
                                ("latent_path" if evaluator == "rmse" else "video_path"): str(path)
                                if existing or artifacts
                                else str(path.relative_to(output)),
                                "scores": scores,
                            }
                        )
                    except (ValueError, RuntimeError, OSError) as exc:
                        raise click.ClickException(
                            f"Prompt index {index} (id={prompt_id!r}): {exc}"
                        ) from exc
                    if reference_paths:
                        results[-1]["reference_path"] = str(reference_paths[index])
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
            if reference_config:
                payload.update(
                    reference_model=reference_model,
                    reference_config=reference_config,
                    reference_generation_params=reference_params,
                    artifacts_retained=media_dir is not None,
                    artifacts_dir=str(artifacts),
                )
            (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
        finally:
            try:
                if generator is not None:
                    _close_runtime(generator)
            finally:
                metric.close()
        click.echo(f"Aggregate scores: {aggregate}")
        click.echo(f"Results: {output / 'results.json'}")
