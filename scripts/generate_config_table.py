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
    DATABASE_LIST_PATH,
    RecipeList,
    assign_profile,
)

MODEL_INFO = {
    "deepseek-ai/DeepSeek-R1-0528": {
        "display_name": "DeepSeek-R1",
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528",
    },
    "nvidia/DeepSeek-R1-0528-FP4-v2": {
        "display_name": "DeepSeek-R1 (NVFP4)",
        "url": "https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4-v2",
    },
    "openai/gpt-oss-120b": {
        "display_name": "gpt-oss-120b",
        "url": "https://huggingface.co/openai/gpt-oss-120b",
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


def _model_display_and_url(model: str) -> tuple[str, str]:
    if model in MODEL_INFO:
        info = MODEL_INFO[model]
        return info["display_name"], info["url"]
    return model, ""


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


def generate_rst(yaml_path, output_file=None):
    rows = build_rows(yaml_path)
    model_groups = defaultdict(list)
    for row in rows:
        model_groups[row.model].append(row)

    lines = []

    lines.append(".. start-config-table-note")
    lines.append(".. include:: ../_includes/note_sections.rst")
    lines.append("   :start-after: .. start-note-traffic-patterns")
    lines.append("   :end-before: .. end-note-traffic-patterns")
    lines.append(".. end-config-table-note")
    lines.append("")

    sorted_models = sorted(model_groups.keys())

    for model in sorted_models:
        lines.append(f".. start-{model}")
        lines.append("")

        model_display_name, model_url = _model_display_and_url(model)
        if model_url:
            title_text = f"`{model_display_name} <{model_url}>`_"
        else:
            title_text = model

        lines.append(f".. _{model}:")
        lines.append("")
        lines.append(title_text)
        lines.append("~" * len(title_text))
        lines.append("")

        lines.append(".. list-table::")
        lines.append("   :width: 100%")
        lines.append("   :header-rows: 1")
        lines.append("   :widths: 12 15 15 13 20 25")
        lines.append("")
        lines.append("   * - GPU")
        lines.append("     - Performance Profile")
        lines.append("     - ISL / OSL")
        lines.append("     - Concurrency")
        lines.append("     - Config")
        lines.append("     - Command")

        entries = sorted(
            model_groups[model],
            key=lambda r: (
                str(r.gpu),
                int(r.num_gpus or 0),
                int(r.isl or 0),
                int(r.osl or 0),
                int(r.concurrency or 0),
            ),
        )

        for row in entries:
            config_link = f"`{row.config_filename} <{row.config_github_url}>`_"
            lines.append(f"   * - {row.gpu_display}")
            lines.append(f"     - {row.performance_profile}")
            lines.append(f"     - {row.isl} / {row.osl}")
            lines.append(f"     - {row.concurrency}")
            lines.append(f"     - {config_link}")
            lines.append(f"     - ``{row.command}``")

        lines.append("")
        lines.append(f".. end-{model}")
        lines.append("")

    output_text = "\n".join(lines)
    if output_file:
        with open(output_file, "w") as f:
            f.write(output_text)
    else:
        print(output_text)


def generate_json(yaml_path, output_file):
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

    payload = {
        "source": source,
        "models": models,
        "entries": [asdict(r) for r in rows],
    }

    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    yaml_path = Path(DATABASE_LIST_PATH)
    if not yaml_path.exists():
        print(f"Error: YAML file not found at {yaml_path}", file=sys.stderr)
        sys.exit(1)
    output_path = REPO_ROOT / "docs/source/deployment-guide/config_table.rst"
    json_output_path = REPO_ROOT / "docs/source/_static/config_db.json"
    generate_rst(yaml_path, output_file=output_path)
    generate_json(yaml_path, output_file=json_output_path)
