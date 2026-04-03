# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.configs.database.database import (  # noqa: E402
    CURATED_LIST_PATH,
    DATABASE_LIST_PATH,
    CuratedRecipeList,
    RecipeList,
    assign_profile,
)

MODEL_INFO = {
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": {
        "display_name": "Nemotron v3 Super (NVFP4)",
        "url": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    },
    "deepseek-ai/DeepSeek-R1-0528": {
        "display_name": "DeepSeek-R1",
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528",
    },
    "nvidia/DeepSeek-R1-0528-FP4-v2": {
        "display_name": "DeepSeek-R1 (NVFP4)",
        "url": "https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-v2",
    },
    "nvidia/DeepSeek-R1-FP4": {
        "display_name": "DeepSeek-R1 (NVFP4)",
        "url": "https://huggingface.co/nvidia/DeepSeek-R1-FP4",
    },
    "nvidia/DeepSeek-R1-FP4-v2": {
        "display_name": "DeepSeek-R1 (NVFP4)",
        "url": "https://huggingface.co/nvidia/DeepSeek-R1-FP4-v2",
    },
    "openai/gpt-oss-120b": {
        "display_name": "gpt-oss-120b",
        "url": "https://huggingface.co/openai/gpt-oss-120b",
    },
    "Qwen/Qwen3-Next-80B-A3B-Thinking": {
        "display_name": "Qwen3-Next-80B",
        "url": "https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking",
    },
    "Qwen/Qwen3-30B-A3B": {
        "display_name": "Qwen3-30B-A3B",
        "url": "https://huggingface.co/Qwen/Qwen3-30B-A3B",
    },
    "nvidia/Llama-3.3-70B-Instruct-FP8": {
        "display_name": "Llama-3.3-70B (FP8)",
        "url": "https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8",
    },
    "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8": {
        "display_name": "Llama 4 Scout (FP8)",
        "url": "https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
    },
    "nvidia/Kimi-K2-Thinking-NVFP4": {
        "display_name": "Kimi-K2-Thinking (NVFP4)",
        "url": "https://huggingface.co/nvidia/Kimi-K2-Thinking-NVFP4",
    },
}


@dataclass(frozen=True)
class RecipeRow:
    model: str
    model_display_name: str
    model_url: str
    gpu: str
    num_gpus: int
    isl: int
    osl: int
    concurrency: int
    config_path: str
    gpu_display: str
    performance_profile: str
    command: str
    config_filename: str
    config_github_url: str
    config_raw_url: str


@dataclass(frozen=True)
class CuratedRow:
    model: str
    model_display_name: str
    model_url: str
    scenario: str
    gpu_compatibility: str
    config_path: str
    config_filename: str
    config_github_url: str
    config_raw_url: str
    command: str


def _model_display_and_url(model: str) -> tuple[str, str]:
    if model in MODEL_INFO:
        info = MODEL_INFO[model]
        return info["display_name"], info["url"]
    return model, ""


def build_curated_rows(yaml_path: Path) -> list[CuratedRow]:
    """Parse curated recipe YAML and return CuratedRow entries for JSON serialization."""
    curated_list = CuratedRecipeList.from_yaml(Path(yaml_path))
    rows: list[CuratedRow] = []
    for entry in curated_list:
        if entry.disagg:
            continue
        model_display_name, model_url = _model_display_and_url(entry.model)
        config_path = entry.config_path
        config_filename = os.path.basename(config_path)
        config_github_url = f"https://github.com/NVIDIA/TensorRT-LLM/blob/main/{config_path}"
        config_raw_url = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/{config_path}"
        command = f"trtllm-serve {entry.model} --config ${{TRTLLM_DIR}}/{config_path}"
        rows.append(
            CuratedRow(
                model=entry.model,
                model_display_name=model_display_name,
                model_url=model_url,
                scenario=entry.scenario,
                gpu_compatibility=entry.gpu_compatibility,
                config_path=config_path,
                config_filename=config_filename,
                config_github_url=config_github_url,
                config_raw_url=config_raw_url,
                command=command,
            )
        )
    return rows


def build_rows(yaml_path) -> list[RecipeRow]:
    recipe_list = RecipeList.from_yaml(Path(yaml_path))

    model_groups = defaultdict(lambda: defaultdict(list))
    for recipe in recipe_list:
        key = (recipe.gpu, recipe.num_gpus, recipe.isl, recipe.osl)
        model_groups[recipe.model][key].append(recipe)

    rows: list[RecipeRow] = []

    sorted_models = sorted(model_groups.keys())
    for model in sorted_models:
        subgroups = model_groups[model]
        sorted_keys = sorted(
            subgroups.keys(),
            key=lambda k: (str(k[0]), int(k[1] or 0), int(k[2] or 0), int(k[3] or 0)),
        )

        model_display_name, model_url = _model_display_and_url(model)

        for key in sorted_keys:
            entries = subgroups[key]
            entries.sort(key=lambda x: x.concurrency)

            for idx, entry in enumerate(entries):
                gpu = entry.gpu
                num_gpus = entry.num_gpus
                gpu_display = f"{num_gpus}x{gpu}" if num_gpus and num_gpus > 1 else gpu
                isl = entry.isl
                osl = entry.osl
                conc = entry.concurrency
                config_path = entry.config_path

                profile = assign_profile(len(entries), idx, conc)

                command = f"trtllm-serve {model} --config ${{TRTLLM_DIR}}/{config_path}"

                config_filename = os.path.basename(config_path)
                config_github_url = (
                    f"https://github.com/NVIDIA/TensorRT-LLM/blob/main/{config_path}"
                )
                config_raw_url = (
                    f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/{config_path}"
                )

                rows.append(
                    RecipeRow(
                        model=model,
                        model_display_name=model_display_name,
                        model_url=model_url,
                        gpu=gpu,
                        num_gpus=num_gpus,
                        isl=isl,
                        osl=osl,
                        concurrency=conc,
                        config_path=config_path,
                        gpu_display=gpu_display,
                        performance_profile=profile,
                        command=command,
                        config_filename=config_filename,
                        config_github_url=config_github_url,
                        config_raw_url=config_raw_url,
                    )
                )

    return rows


def generate_json(yaml_path: Path, output_file: Path, curated_yaml_path: Path | None = None):
    rows = build_rows(yaml_path)

    source_path = Path(yaml_path)
    try:
        source = str(source_path.relative_to(REPO_ROOT))
    except ValueError:
        source = str(source_path)

    models = {}
    for row in rows:
        if row.model not in models:
            models[row.model] = {
                "display_name": row.model_display_name,
                "url": row.model_url,
            }

    curated_entries = []
    if curated_yaml_path and curated_yaml_path.exists():
        curated_rows = build_curated_rows(curated_yaml_path)
        curated_entries = [asdict(r) for r in curated_rows]
        for crow in curated_rows:
            if crow.model not in models:
                models[crow.model] = {
                    "display_name": crow.model_display_name,
                    "url": crow.model_url,
                }

    payload = {
        "source": source,
        "models": models,
        "entries": [asdict(r) for r in rows],
        "curated_entries": curated_entries,
    }

    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    yaml_path = Path(DATABASE_LIST_PATH)
    if not yaml_path.exists():
        print(f"Error: YAML file not found at {yaml_path}", file=sys.stderr)
        sys.exit(1)
    json_output_path = REPO_ROOT / "docs/source/_static/config_db.json"
    curated_path = CURATED_LIST_PATH if CURATED_LIST_PATH.exists() else None
    generate_json(yaml_path, output_file=json_output_path, curated_yaml_path=curated_path)
