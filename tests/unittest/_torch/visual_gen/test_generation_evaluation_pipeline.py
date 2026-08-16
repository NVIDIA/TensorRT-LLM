# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tensorrt_llm.visual_gen.generation_evaluation import (
    ImageGenerationEvaluationPipeline,
    QwenImageBenchResult,
    aggregate_total_score,
    compute_dimension_score,
    extract_json_from_response,
    fix_score_json,
    parse_dimension_output,
    validate_generation_evaluation_request,
)


def test_extract_json_from_response_ignores_thinking_and_code_fence():
    response = """<think>reasoning</think>
```json
{"Realism": {"Physical Logic": {"score": 2}}}
```
"""

    assert extract_json_from_response(response) == {
        "Realism": {"Physical Logic": {"score": 2}}
    }


def test_dimension_score_maps_qwen_bench_scores():
    fixed = fix_score_json(
        {
            "Physical Logic": {"score": 2},
            "Material Texture": {"score": 1},
            "Noise": {"score": "N/A"},
        },
        "Quality",
    )

    assert fixed == {
        "Realism": {
            "Physical Logic": {"score": 2},
            "Material Texture": {"score": 1},
        },
        "Detail": {"Noise": {"score": "N/A"}},
    }
    score = compute_dimension_score(fixed)
    assert score["level2_scores"] == {"Realism": 80.0, "Detail": None}
    assert score["level3_scores"] == {
        "Realism": {"Physical Logic": 100.0, "Material Texture": 60.0},
        "Detail": {"Noise": None},
    }
    assert score["level1_score"] == 80.0


def test_parse_dimension_output_returns_none_on_invalid_json():
    assert parse_dimension_output("not json", "Quality") == (None, None)


def test_aggregate_total_score_ignores_none_dimensions():
    assert (
        aggregate_total_score(
            {
                "Quality": {"level1_score": 100.0},
                "Aesthetics": {"level1_score": 60.0},
                "Alignment": {"level1_score": None},
            }
        )
        == 80.0
    )


@dataclass
class _FakeVisualGenOutput:
    image: object | None = None
    error: str | None = None


class _FakeGenerator:
    def generate(self, inputs, params=None):
        return [
            _FakeVisualGenOutput(image=f"image:{prompt}")
            if prompt != "generation failure"
            else _FakeVisualGenOutput(error="generation failed")
            for prompt in inputs
        ]


class _FakeEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate_batch(self, prompts, images, dimensions):
        self.calls.append((list(prompts), list(images), list(dimensions)))
        return [
            QwenImageBenchResult(
                prompt=prompt,
                dimensions=list(dimensions),
                level1_scores={dim: 100.0 for dim in dimensions},
                total_score=100.0 if prompt == "good" else None,
                parse_failures=[] if prompt == "good" else list(dimensions),
            )
            for prompt in prompts
        ]


def test_pipeline_handles_partial_generation_and_evaluation_failures():
    evaluator = _FakeEvaluator()
    pipeline = ImageGenerationEvaluationPipeline(_FakeGenerator(), evaluator)

    response = pipeline.run(
        ["good", "generation failure", "parse failure"],
        dimensions=["Quality"],
    )

    assert evaluator.calls == [
        (["good", "parse failure"], ["image:good", "image:parse failure"], ["Quality"])
    ]
    assert response.aggregate_score == 100.0
    assert response.aggregation == {
        "method": "mean",
        "num_prompts": 3,
        "num_successful": 1,
        "num_failed": 2,
    }
    assert [result.error for result in response.results] == [
        None,
        "generation failed",
        None,
    ]
    assert response.results[2].parse_failures == ["Quality"]


def test_pipeline_omits_images_unless_requested():
    response = ImageGenerationEvaluationPipeline(
        _FakeGenerator(), _FakeEvaluator()
    ).run(["good"], dimensions=["Quality"])

    assert response.results[0].image is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prompts": []}, "prompts"),
        ({"prompts": [""]}, "empty"),
        ({"prompts": ["ok"], "generation_n": 2}, "generation.n"),
        ({"prompts": ["ok"], "dimensions": ["Unknown"]}, "Unsupported"),
        ({"prompts": ["ok"], "aggregation_method": "median"}, "mean"),
    ],
)
def test_validate_generation_evaluation_request(kwargs, match):
    with pytest.raises(ValueError, match=match):
        validate_generation_evaluation_request(**kwargs)
