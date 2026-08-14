# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
from pathlib import Path


def _load_eval_module():
    repo_root = Path(__file__).resolve().parents[4]
    script_path = (
        repo_root / "examples" / "models" / "core" / "multimodal" / "qwen_image_bench_eval.py"
    )
    spec = importlib.util.spec_from_file_location("qwen_image_bench_eval", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_user_prompt_contains_dimension_and_image_marker():
    module = _load_eval_module()

    user_prompt = module.build_user_prompt("A cute cat playing piano", "Alignment")

    assert "A cute cat playing piano" in user_prompt
    assert "# Generated Image\n<image>" in user_prompt
    assert "# Evaluation Dimension\nAlignment" in user_prompt
    assert "Contact Interaction" in user_prompt


def test_parse_dimension_output_fixes_flat_scores_and_aggregates():
    module = _load_eval_module()
    output = """model reasoning</think>
{
  "Physical Logic": {"score": 0},
  "Material Texture": {"score": 1},
  "Resolution": {"score": "N/A"}
}
"""

    fixed_scores, dimension_score = module.parse_dimension_output(output, "Quality")

    assert fixed_scores == {
        "Realism": {
            "Physical Logic": {"score": 0},
            "Material Texture": {"score": 1},
        },
        "Resolution": {
            "Resolution": {"score": "N/A"},
        },
    }
    assert dimension_score["level2_scores"] == {
        "Realism": 30.0,
        "Resolution": None,
    }
    assert dimension_score["level1_score"] == 30.0


def test_creative_generation_feature_mapping_alias():
    module = _load_eval_module()
    output = '{"Feature Mapping": {"score": 2}, "Logical Resolution": {"score": 1}}'

    fixed_scores, dimension_score = module.parse_dimension_output(output, "Creative Generation")

    assert fixed_scores == {
        "Feature Matching": {"Feature Matching": {"score": 2}},
        "Logical Resolution": {"Logical Resolution": {"score": 1}},
    }
    assert dimension_score["level1_score"] == 80.0


def test_parse_dimension_output_handles_non_object_json():
    module = _load_eval_module()

    fixed_scores, dimension_score = module.parse_dimension_output("[]", "Quality")

    assert fixed_scores is None
    assert dimension_score is None


def test_total_score_averages_available_level1_scores():
    module = _load_eval_module()

    total_score = module.aggregate_total_score(
        {
            "Quality": {"level1_score": 30.0},
            "Aesthetics": {"level1_score": None},
            "Alignment": {"level1_score": 90.0},
        }
    )

    assert total_score == 60.0
