# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for the shared Qwen-Image 2.1 BF16 example and CI config."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "examples" / "visual_gen" / "configs"
CONFIG = CONFIG_DIR / "qwen-image-2.1-bf16-1gpu.yaml"


def test_qwen_image_21_shared_baseline_config() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config == {
        "attention_config": {"backend": "VANILLA"},
        "parallel_config": {"cfg_size": 1, "ulysses_size": 1},
        "cuda_graph_config": {"enable": False},
        "torch_compile_config": {"enable": False},
    }


def test_qwen_image_21_perf_sanity_uses_shared_baseline() -> None:
    perf_path = (
        REPO_ROOT
        / "tests"
        / "scripts"
        / "perf-sanity"
        / "visual_gen"
        / "qwen_image_21_blackwell.yaml"
    )
    config = yaml.safe_load(perf_path.read_text(encoding="utf-8"))
    for server in config["server_configs"]:
        assert server["visual_gen_args_path"] == str(CONFIG.relative_to(REPO_ROOT))
