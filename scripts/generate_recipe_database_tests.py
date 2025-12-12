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
"""Generate aggr_server configs and test list from the config recipe database.

This script:
1. Reads recipes from lookup.yaml
2. Generates aggr_server-format config file per GPU type (e.g., recipe_database_b200.yaml)
3. Generates llm_recipe_database.yml test list with condition blocks grouped by num_gpus

Usage:
    python scripts/generate_recipe_database_tests.py [OPTIONS]

Options:
    --iterations NUM    Number of benchmark iterations (default: 10)
    --output-dir DIR    Output directory for configs (default: tests/scripts/perf-sanity)
    --test-list PATH    Output path for test list YAML
    --lookup-yaml PATH  Path to lookup.yaml (default: auto-detect)
"""

import argparse
import copy
from collections import defaultdict
from pathlib import Path

import yaml

# GPU type to condition mapping for test list
GPU_CONDITIONS = {
    "B200_NVL": {
        "ranges": {
            "compute_capability": {"gte": 10.0, "lte": 10.3},
            "gpu_memory": {"gte": 180},
        },
    },
    "H200_SXM": {
        "ranges": {
            "compute_capability": {"gte": 9.0, "lte": 9.0},
            "gpu_memory": {"gte": 140},
        },
    },
    "H100_SXM": {
        "ranges": {
            "compute_capability": {"gte": 9.0, "lte": 9.0},
            "gpu_memory": {"gte": 80},
        },
    },
}

# GPU type to config file prefix mapping
GPU_CONFIG_PREFIX = {
    "B200_NVL": "recipe_database_b200",
    "H200_SXM": "recipe_database_h200",
    "H100_SXM": "recipe_database_h100",
}


def load_recipes(yaml_path: Path) -> list:
    """Load recipes from lookup.yaml."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_recipe_config(repo_root: Path, config_path: str) -> dict:
    """Load the LLM API config for a recipe."""
    full_path = repo_root / config_path
    if not full_path.exists():
        print(f"Warning: Config file not found: {full_path}")
        return {}
    with open(full_path) as f:
        return yaml.safe_load(f) or {}


def generate_server_name(recipe: dict) -> str:
    """Generate a unique server name from recipe."""
    model_slug = recipe["model"].replace("/", "_").replace("-", "_").replace(".", "_")
    return f"{model_slug}_{recipe['isl']}_{recipe['osl']}_conc{recipe['concurrency']}_gpu{recipe['num_gpus']}"


def generate_client_name(recipe: dict) -> str:
    """Generate client config name."""
    return f"con{recipe['concurrency']}_isl{recipe['isl']}_osl{recipe['osl']}"


def recipe_to_server_config(recipe: dict, llm_api_config: dict, iterations: int) -> dict:
    """Convert a recipe + LLM API config to aggr_server format."""
    server_config = {
        "name": generate_server_name(recipe),
        "model_name": recipe["model"],
        "gpus": recipe["num_gpus"],
        # Enable scenario-only matching for baseline comparison
        "match_mode": "scenario",
    }

    # Copy LLM API config fields (excluding backend which is handled separately)
    for key, value in llm_api_config.items():
        if key not in ["backend", "print_iter_log"]:
            server_config[key] = value

    # Add client configs
    server_config["client_configs"] = [
        {
            "name": generate_client_name(recipe),
            "concurrency": recipe["concurrency"],
            "iterations": iterations,
            "isl": recipe["isl"],
            "osl": recipe["osl"],
            "random_range_ratio": 0.0,  # Fixed ISL/OSL for reproducibility
            "backend": "openai",
            "streaming": True,
        }
    ]

    return server_config


def group_recipes_by_gpu(recipes: list) -> dict:
    """Group recipes by GPU type."""
    groups = defaultdict(list)
    for recipe in recipes:
        groups[recipe["gpu"]].append(recipe)
    return groups


def group_recipes_by_num_gpus(recipes: list) -> dict:
    """Group recipes by num_gpus within a GPU type."""
    groups = defaultdict(list)
    for recipe in recipes:
        groups[recipe["num_gpus"]].append(recipe)
    return groups


def generate_aggr_config(recipes: list, repo_root: Path, iterations: int) -> dict:
    """Generate aggr_server config from recipes."""
    server_configs = []

    for recipe in recipes:
        llm_api_config = load_recipe_config(repo_root, recipe["config_path"])
        if not llm_api_config:
            continue
        server_config = recipe_to_server_config(recipe, llm_api_config, iterations)
        server_configs.append(server_config)

    return {"server_configs": server_configs}


def generate_config_name(gpu_type: str) -> str:
    return GPU_CONFIG_PREFIX.get(gpu_type, f"recipe_database_{gpu_type.lower()}")


def generate_test_name(config_name: str, server_name: str) -> str:
    return f"perf/test_perf.py::test_perf[perf_sanity_upload-{config_name}-{server_name}]"


def generate_condition_entry(
    gpu_type: str, num_gpus: int, config_name: str, server_names: list
) -> dict:
    base_condition = GPU_CONDITIONS.get(gpu_type, {})
    condition = copy.deepcopy(base_condition)
    condition["ranges"]["system_gpu_count"] = {"gte": num_gpus}

    tests = [generate_test_name(config_name, name) for name in server_names]
    return {"condition": condition, "tests": tests}


def write_test_list_with_header(test_list_path: Path, test_list: dict):
    """Write test list YAML with a header comment."""
    header = """# ===============================================================================
# Recipe Database Performance Tests (Auto-generated)
# ===============================================================================
# Generated by: scripts/generate_recipe_database_tests.py
#
# These tests use scenario-only matching (match_mode: scenario) for baselines.
# Baselines are matched by (model, gpu, isl, osl, concurrency, num_gpus) instead
# of full config fields, allowing configs to evolve while maintaining comparison.
#
# To regenerate:
#   python scripts/generate_recipe_database_tests.py
# ===============================================================================

"""
    with open(test_list_path, "w") as f:
        f.write(header)
        yaml.dump(test_list, f, default_flow_style=False, sort_keys=False, width=120)


def main():
    parser = argparse.ArgumentParser(
        description="Generate performance regression test list from recipe database"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for config files (default: tests/scripts/perf-sanity)",
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        default=None,
        help="Output path for test list YAML (default: tests/integration/test_lists/qa/llm_recipe_database.yml)",
    )
    parser.add_argument(
        "--lookup-yaml",
        type=Path,
        default=None,
        help="Path to lookup.yaml (default: auto-detect)",
    )
    args = parser.parse_args()

    # Find repo root and paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    if args.lookup_yaml:
        lookup_path = args.lookup_yaml
    else:
        lookup_path = repo_root / "examples" / "configs" / "database" / "lookup.yaml"

    if not lookup_path.exists():
        raise FileNotFoundError(f"lookup.yaml not found at {lookup_path}")

    output_dir = args.output_dir or (repo_root / "tests" / "scripts" / "perf-sanity")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_list_path = args.test_list or (
        repo_root / "tests" / "integration" / "test_lists" / "qa" / "llm_recipe_database.yml"
    )

    recipes = load_recipes(lookup_path)
    print(f"Loaded {len(recipes)} recipes from {lookup_path}")

    gpu_groups = group_recipes_by_gpu(recipes)
    condition_entries = []

    for gpu_type in sorted(gpu_groups.keys()):
        gpu_recipes = gpu_groups[gpu_type]
        config_name = generate_config_name(gpu_type)
        config_path = output_dir / f"{config_name}.yaml"

        aggr_config = generate_aggr_config(gpu_recipes, repo_root, args.iterations)

        with open(config_path, "w") as f:
            yaml.dump(aggr_config, f, default_flow_style=False, sort_keys=False, width=120)
        print(f"Generated {config_path} with {len(aggr_config['server_configs'])} server configs")

        # Generate condition entries grouped by num_gpus
        num_gpus_groups = group_recipes_by_num_gpus(gpu_recipes)
        for num_gpus in sorted(num_gpus_groups.keys()):
            server_names = [generate_server_name(r) for r in num_gpus_groups[num_gpus]]
            entry = generate_condition_entry(gpu_type, num_gpus, config_name, server_names)
            condition_entries.append(entry)

    test_list = {
        "version": "0.0.1",
        "llm_recipe_database": condition_entries,
    }

    test_list_path.parent.mkdir(parents=True, exist_ok=True)
    write_test_list_with_header(test_list_path, test_list)
    print(f"Generated {test_list_path}")


if __name__ == "__main__":
    main()
