#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Generate a performance regression test list from the config database.

This script:
1. Reads recipes from the examples/configs/database directory
2. Generates test config files per GPU type (e.g., config_database_b200_nvl.yaml)
3. Generates llm_config_database.yml test list with condition blocks grouped by GPU name and count
"""

import copy
from collections import defaultdict
from pathlib import Path

import yaml

from examples.configs.database.database import (
    DATABASE_LIST_PATH,
    Recipe,
    RecipeList,
    select_key_recipes,
)

REPO_ROOT = Path(__file__).parent.parent
PERF_SANITY_DIR = REPO_ROOT / "tests" / "scripts" / "perf-sanity"
TEST_LIST_PATH = (
    REPO_ROOT / "tests" / "integration" / "test_lists" / "qa" / "llm_config_database.yml"
)
ITERATIONS = 10
# Mapping from HuggingFace model IDs to MODEL_PATH_DICT keys used by the test framework
# in tests/integration/defs/perf/test_perf_sanity.py
MODEL_NAME_MAPPING = {
    "deepseek-ai/DeepSeek-R1-0528": "deepseek_r1_0528_fp8",
    "nvidia/DeepSeek-R1-0528-FP4-v2": "deepseek_r1_0528_fp4_v2",
    "openai/gpt-oss-120b": "gpt_oss_120b_fp4",
}


# GPU type to condition wildcards mapping for test list
# Note: cpu is used to distinguish between e.g. H200_SXM and GH200
GPU_WILDCARDS = {
    "B200_NVL": {"gpu": ["*b200*"], "cpu": "x86_64", "linux_distribution_name": "ubuntu*"},
    "H200_SXM": {"gpu": ["*h200*"], "cpu": "x86_64", "linux_distribution_name": "ubuntu*"},
    "H100_SXM": {"gpu": ["*h100*"], "cpu": "x86_64", "linux_distribution_name": "ubuntu*"},
    "GH200": {"gpu": ["*gh200*"], "cpu": "aarch64", "linux_distribution_name": "ubuntu*"},
    "GB200": {"gpu": ["*gb200*"], "cpu": "aarch64", "linux_distribution_name": "ubuntu*"},
}


def generate_server_name(recipe: Recipe) -> str:
    """Generate a unique server name from recipe."""
    model_slug = recipe.model.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"{model_slug}_{recipe.isl}_{recipe.osl}_conc{recipe.concurrency}_gpu{recipe.num_gpus}"


def generate_client_name(recipe: Recipe) -> str:
    """Generate client config name."""
    return f"con{recipe.concurrency}_isl{recipe.isl}_osl{recipe.osl}"


def recipe_to_server_config(recipe: Recipe, llm_api_config: dict) -> dict:
    """Convert a recipe + LLM API config to aggr_server format."""
    model_name = MODEL_NAME_MAPPING.get(recipe.model)
    if not model_name:
        raise ValueError(f"Model not found in MODEL_NAME_MAPPING: {recipe.model}")

    server_config = {
        "name": generate_server_name(recipe),
        "model_name": model_name,
        "gpus": recipe.num_gpus,
        # Enable scenario-only matching for baseline comparison
        "match_mode": "scenario",
    }

    # Copy LLM API config fields
    for key, value in llm_api_config.items():
        server_config[key] = value

    # Disable KV cache reuse to ensure consistency
    if "kv_cache_config" not in server_config:
        server_config["kv_cache_config"] = {}
    server_config["kv_cache_config"]["enable_block_reuse"] = False

    # Add client configs
    server_config["client_configs"] = [
        {
            "name": generate_client_name(recipe),
            "concurrency": recipe.concurrency,
            "iterations": ITERATIONS,
            "isl": recipe.isl,
            "osl": recipe.osl,
            "random_range_ratio": 0.0,  # Fixed ISL/OSL for reproducibility
            "backend": "openai",
            "streaming": True,
        }
    ]

    return server_config


def group_recipes_by_scenario(recipes: RecipeList) -> dict:
    """Group recipes by scenario key (model, gpu, isl, osl, num_gpus)."""
    groups = defaultdict(list)
    for recipe in recipes:
        key = (recipe.model, recipe.gpu, recipe.isl, recipe.osl, recipe.num_gpus)
        groups[key].append(recipe)
    return groups


def filter_to_key_recipes(recipes: RecipeList) -> list[Recipe]:
    """Filter recipes to only key configs (min latency, balanced, max throughput)."""
    scenario_groups = group_recipes_by_scenario(recipes)
    key_recipes = []
    for scenario_recipes in scenario_groups.values():
        for recipe, _ in select_key_recipes(scenario_recipes):
            key_recipes.append(recipe)
    return key_recipes


def group_recipes_by_gpu(recipes: list[Recipe]) -> dict[str, list[Recipe]]:
    """Group recipes by GPU type."""
    groups = defaultdict(list)
    for recipe in recipes:
        groups[recipe.gpu].append(recipe)
    return groups


def group_recipes_by_num_gpus(recipes: list[Recipe]) -> dict[int, list[Recipe]]:
    """Group recipes by num_gpus within a GPU type."""
    groups = defaultdict(list)
    for recipe in recipes:
        groups[recipe.num_gpus].append(recipe)
    return groups


def generate_aggr_config(recipes: list[Recipe]) -> dict[str, list[dict]]:
    """Generate aggr_server config from recipes."""
    server_configs = []

    for recipe in recipes:
        llm_api_config = recipe.load_config()
        server_config = recipe_to_server_config(recipe, llm_api_config)
        server_configs.append(server_config)

    return {"server_configs": server_configs}


def generate_condition_entry(
    gpu_name: str, num_gpus: int, config_name: str, server_names: list
) -> dict:
    # using copy.deepcopy to avoid creating YAML anchors
    wildcards = copy.deepcopy(GPU_WILDCARDS[gpu_name])
    condition = {
        "wildcards": wildcards,
        "ranges": {"system_gpu_count": {"gte": num_gpus}},
    }

    tests = [
        f"perf/test_perf_sanity.py::test_e2e[aggr_upload-{config_name}-{name}]"
        for name in server_names
    ]
    return {"condition": condition, "tests": tests}


def generate_tests(test_list_path: Path = TEST_LIST_PATH, test_config_dir: Path = PERF_SANITY_DIR):
    test_list_path.parent.mkdir(parents=True, exist_ok=True)

    all_recipes = RecipeList.from_yaml(DATABASE_LIST_PATH)
    recipes = filter_to_key_recipes(all_recipes)
    print(f"Selected {len(recipes)} key recipes from {len(all_recipes)} total")

    gpu_groups = group_recipes_by_gpu(recipes)
    condition_entries = []
    config_files = {}

    for gpu_name in sorted(gpu_groups.keys()):
        gpu_recipes = gpu_groups[gpu_name]
        config_name = f"config_database_{gpu_name.lower()}"
        config_path = test_config_dir / f"{config_name}.yaml"

        aggr_config = generate_aggr_config(gpu_recipes)
        config_content = yaml.dump(
            aggr_config, default_flow_style=False, sort_keys=False, width=120
        )

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        print(f"Generated {config_path}")

        config_files[config_path] = config_content

        # Generate condition entries grouped by num_gpus
        num_gpus_groups = group_recipes_by_num_gpus(gpu_recipes)
        for num_gpus in sorted(num_gpus_groups.keys()):
            server_names = [generate_server_name(r) for r in num_gpus_groups[num_gpus]]
            entry = generate_condition_entry(gpu_name, num_gpus, config_name, server_names)
            condition_entries.append(entry)

    test_list = {
        "version": "0.0.1",
        "llm_config_database": condition_entries,
    }

    header = """# ===============================================================================
# Config Database Performance Tests (AUTO-GENERATED)
# ===============================================================================
# Generated by: scripts/generate_config_database_tests.py
#
# These tests use scenario-only matching (match_mode: scenario) for baselines.
# Baselines are matched by (model, gpu, isl, osl, concurrency, num_gpus) instead
# of full config fields, allowing configs to evolve while maintaining comparison.
#
# To regenerate:
#   python scripts/generate_config_database_tests.py
# ===============================================================================

"""
    with open(test_list_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(test_list, f, default_flow_style=False, sort_keys=False, width=120)
    print(f"Generated {test_list_path}")


if __name__ == "__main__":
    generate_tests()
