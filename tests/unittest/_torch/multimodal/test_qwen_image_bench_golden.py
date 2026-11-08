# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import importlib.util
import json
import os
from pathlib import Path

import pytest

PROMPT = "A cute cat playing piano"
DIMENSIONS = ["Alignment"]
GOLDEN_PATH = (
    Path(__file__).resolve().parent / "golden" / "qwen_image_bench_cat_piano_alignment.json"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_eval_module():
    script_path = (
        _repo_root() / "examples" / "models" / "core" / "multimodal" / "qwen_image_bench_eval.py"
    )
    spec = importlib.util.spec_from_file_location("qwen_image_bench_eval", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _qwen_image_bench_model_dir() -> Path:
    model_dir = os.environ.get("QWEN_IMAGE_BENCH_MODEL_DIR")
    if not model_dir:
        pytest.skip("Set QWEN_IMAGE_BENCH_MODEL_DIR to run this golden E2E test.")

    model_path = Path(model_dir)
    if not (model_path / "config.json").exists():
        pytest.skip(f"Qwen-Image-Bench config.json not found under {model_path}.")
    return model_path


def _eval_args(model_dir: Path, image_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model_path=str(model_dir),
        prompt=PROMPT,
        image_path=str(image_path),
        output_path=None,
        dimensions=DIMENSIONS,
        backend="pytorch",
        image_data_format="pt",
        max_tokens=4096,
        max_num_tokens=8192,
        max_seq_len=8192,
        kv_cache_max_tokens=8192,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.05,
        use_fast_processor=False,
        enable_block_reuse=False,
        include_raw_outputs=True,
    )


def _stable_result(result: dict) -> dict:
    image_name = result.get("image_name")
    if image_name is None:
        image_name = Path(result["image_path"]).name

    return {
        "prompt": result["prompt"],
        "image_name": image_name,
        "dimensions": result["dimensions"],
        "level1_scores": result["level1_scores"],
        "level2_scores": result["level2_scores"],
        "level3_scores": result["level3_scores"],
        "total_score": result["total_score"],
        "parse_failures": result["parse_failures"],
        "parsed_scores": result["parsed_scores"],
    }


@pytest.mark.threadleak(enabled=False)
def test_qwen_image_bench_cat_piano_golden_output():
    model_dir = _qwen_image_bench_model_dir()
    image_path = _repo_root() / "examples" / "visual_gen" / "cat_piano.png"
    if not image_path.exists():
        pytest.skip(f"Golden image not found: {image_path}")

    module = _load_eval_module()
    actual = _stable_result(module.evaluate(_eval_args(model_dir, image_path)))
    expected = _stable_result(json.loads(GOLDEN_PATH.read_text()))

    assert actual == expected
