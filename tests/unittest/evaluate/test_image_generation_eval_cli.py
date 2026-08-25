# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass

import click
import pytest
from click.testing import CliRunner

import tensorrt_llm.commands.eval as eval_cmd
from tensorrt_llm.evaluate.visual_gen.config import split_generator_config
from tensorrt_llm.evaluate.visual_gen.evaluators import build_image_evaluator
from tensorrt_llm.evaluate.visual_gen.types import QwenImageBenchResult


@dataclass
class _FakeVisualGenOutput:
    image: object | None = None
    error: str | None = None

    def save(self, path):
        from pathlib import Path

        image_path = Path(path)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_text(str(self.image), encoding="utf-8")
        return image_path


class _FakeGenerator:
    def __init__(self):
        self.calls = []
        self.shutdown_called = False

    def generate(self, inputs, params=None):
        self.calls.append((list(inputs), params))
        return [_FakeVisualGenOutput(image=f"image:{prompt}") for prompt in inputs]

    def shutdown(self):
        self.shutdown_called = True


class _FakeEvaluator:
    def __init__(self):
        self.calls = []
        self.closed = False

    def evaluate_batch(self, prompts, images, dimensions):
        self.calls.append((list(prompts), list(images), list(dimensions)))
        return [
            QwenImageBenchResult(
                prompt=prompt,
                dimensions=list(dimensions),
                level1_scores={dim: 100.0 for dim in dimensions},
                total_score=100.0,
            )
            for prompt in prompts
        ]

    def close(self):
        self.closed = True


def test_image_generation_eval_help_does_not_construct_text_llm(monkeypatch):
    def fail_llm_init(*args, **kwargs):
        raise AssertionError("text LLM should not be constructed")

    monkeypatch.setattr(eval_cmd, "PyTorchLLM", fail_llm_init)

    result = CliRunner().invoke(
        eval_cmd.main,
        ["--model", "generator", "image_generation_eval", "--help"],
    )

    assert result.exit_code == 0
    assert "--evaluator" in result.output
    assert "--evaluator-options" in result.output


def test_visual_gen_args_is_documented_primary_config_option():
    result = CliRunner().invoke(eval_cmd.main, ["--help"])

    assert result.exit_code == 0
    assert result.output.index("--visual_gen_args") < result.output.index("--config")


def test_config_alias_reaches_image_generation_eval_help():
    result = CliRunner().invoke(
        eval_cmd.main,
        [
            "--model",
            "generator",
            "--config",
            "generator.yaml",
            "image_generation_eval",
            "--help",
        ],
    )

    assert result.exit_code == 0
    assert "--evaluator" in result.output


def test_image_generation_eval_cli_runs_pipeline_with_fakes(tmp_path, monkeypatch):
    from tensorrt_llm.evaluate.visual_gen import image_generation_eval

    generator = _FakeGenerator()
    evaluator = _FakeEvaluator()
    generator_config = tmp_path / "generator.yaml"
    generator_config.write_text("generation_params:\n  seed: 123\n", encoding="utf-8")
    evaluator_config = tmp_path / "evaluator.yaml"
    evaluator_config.write_text("max_batch_size: 1\n", encoding="utf-8")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "prompt": "first"}),
                json.dumps({"id": "b", "prompt": "second"}),
            ]
        ),
        encoding="utf-8",
    )

    def fake_build_generator(model, config_path):
        assert model == "generator"
        assert config_path == str(generator_config)
        return generator, object()

    def fake_build_evaluator(evaluator_model, evaluator_options):
        assert evaluator_model == "judge"
        assert evaluator_options == str(evaluator_config)
        return evaluator

    monkeypatch.setattr(image_generation_eval, "_build_visual_generator", fake_build_generator)
    monkeypatch.setattr(image_generation_eval, "_build_image_evaluator", fake_build_evaluator)
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(
        eval_cmd.main,
        [
            "--model",
            "generator",
            "--visual_gen_args",
            str(generator_config),
            "image_generation_eval",
            "--evaluator",
            "judge",
            "--evaluator-options",
            str(evaluator_config),
            "--prompts",
            str(prompts),
            "--output-dir",
            str(output_dir),
            "--criteria",
            "Quality",
        ],
    )

    assert result.exit_code == 0, result.output
    assert generator.shutdown_called
    assert evaluator.closed
    expected_images = [
        str(output_dir / "generated_images" / "0000.png"),
        str(output_dir / "generated_images" / "0001.png"),
    ]
    assert generator.calls == [(["first", "second"], generator.calls[0][1])]
    assert evaluator.calls == [(["first", "second"], expected_images, ["Quality"])]

    summary = json.loads((output_dir / "results.json").read_text())
    assert summary["metadata"]["generator_model"] == "generator"
    assert summary["metadata"]["evaluator_model"] == "judge"
    assert summary["aggregate_score"] == 100.0
    assert summary["results"][0]["id"] == "a"
    assert summary["results"][0]["image_path"] == "generated_images/0000.png"
    assert (output_dir / "generated_images" / "0000.png").read_text() == "image:first"
    assert (output_dir / "results.jsonl").exists()


def test_generator_config_rejects_backend_selector():
    with pytest.raises(click.BadParameter, match="Do not set a generator backend"):
        split_generator_config({"backend": "pytorch"})


@pytest.mark.parametrize("selector_key", ["model", "type", "backend"])
def test_evaluator_options_reject_selector_fields(tmp_path, selector_key):
    evaluator_config = tmp_path / "evaluator.yaml"
    evaluator_config.write_text(f"{selector_key}: bad\n", encoding="utf-8")

    with pytest.raises(click.BadParameter, match="Do not set model, type, or backend"):
        build_image_evaluator("qwen-image-bench", str(evaluator_config))
