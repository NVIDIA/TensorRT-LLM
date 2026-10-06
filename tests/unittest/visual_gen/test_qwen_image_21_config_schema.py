# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-level checks for Qwen-Image-2.1 VisualGen example configs."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "examples" / "visual_gen" / "configs"
CONFIGS = [
    CONFIG_DIR / "qwen-image-2.1-bf16-1gpu.yaml",
    CONFIG_DIR / "qwen-image-2.1-bf16-ci.yaml",
]
SUPPORTED_TOP_LEVEL_KEYS = {
    "attention_config",
    "parallel_config",
    "cuda_graph_config",
    "torch_compile_config",
}


def _read(path: Path) -> str:
    assert path.is_file(), f"missing config: {path.relative_to(REPO_ROOT)}"
    return path.read_text(encoding="utf-8")


def _top_level_yaml_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith(" "):
            continue
        if ":" in line:
            keys.add(line.split(":", 1)[0])
    return keys


def test_qwen_image_21_configs_use_stable_model_name_comments() -> None:
    for path in CONFIGS:
        text = _read(path)
        assert "qwen-image-2.1" in text
        assert "Qwen/Qwen-Image-2.1" in text


def test_qwen_image_21_configs_do_not_encode_one_step_smoke_as_e2e_default() -> None:
    for path in CONFIGS:
        text = _read(path)
        assert "default_num_inference_steps: 30" in text
        assert "num_inference_steps: 1" not in text


def test_qwen_image_21_configs_keep_checkpoint_policy_out_of_visual_gen_args_yaml() -> None:
    for path in CONFIGS:
        text = _read(path)
        keys = _top_level_yaml_keys(text)
        assert keys <= SUPPORTED_TOP_LEVEL_KEYS
        assert "checkpoint_policy:" not in text
        assert "allow_download:" not in text


def test_qwen_image_21_ci_config_is_not_full_e2e_evidence() -> None:
    text = _read(CONFIG_DIR / "qwen-image-2.1-bf16-ci.yaml")
    assert "full_e2e=false" in text
    assert "media_sanity_required=false" in text
    assert "lpips_required=false" in text
