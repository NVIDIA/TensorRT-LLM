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


import os
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent

# Add repo root to path for examples.configs.database import
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


def generate_rst(yaml_path, output_file=None):
    """Generate RST table from YAML config database.

    Args:
        yaml_path: Path to lookup.yaml (str or Path)
        output_file: Optional output file path. If None, prints to stdout.
    """
    recipe_list = RecipeList.from_yaml(Path(yaml_path))

    # Group by model -> (gpu, num_gpus, isl, osl) -> list of recipes
    model_groups = defaultdict(lambda: defaultdict(list))
    for recipe in recipe_list:
        key = (recipe.gpu, recipe.num_gpus, recipe.isl, recipe.osl)
        model_groups[recipe.model][key].append(recipe)

    lines = []

    # Include note_sections.rst at the top (relative include for Sphinx)
    lines.append(".. include:: ../_includes/note_sections.rst")
    lines.append("   :start-after: .. start-note-traffic-patterns")
    lines.append("   :end-before: .. end-note-traffic-patterns")
    lines.append("")

    sorted_models = sorted(model_groups.keys())

    for model in sorted_models:
        lines.append(f".. start-{model}")
        lines.append("")

        if model in MODEL_INFO:
            info = MODEL_INFO[model]
            title_text = f"`{info['display_name']} <{info['url']}>`_"
        else:
            title_text = model

        lines.append(f".. _{model}:")
        lines.append("")
        lines.append(title_text)
        lines.append("^" * len(title_text))
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

        subgroups = model_groups[model]
        sorted_keys = sorted(
            subgroups.keys(),
            key=lambda k: (str(k[0]), int(k[1] or 0), int(k[2] or 0), int(k[3] or 0)),
        )

        for key in sorted_keys:
            entries = subgroups[key]
            entries.sort(key=lambda x: x.concurrency)
            n = len(entries)

            for idx, entry in enumerate(entries):
                gpu = entry.gpu
                num_gpus = entry.num_gpus
                gpu_display = f"{num_gpus}x{gpu}" if num_gpus and num_gpus > 1 else gpu
                isl = entry.isl
                osl = entry.osl
                conc = entry.concurrency
                config_path = entry.config_path

                profile = assign_profile(n, idx, conc)

                full_config_path = config_path
                command = f"trtllm-serve {model} --config ${{TRTLLM_DIR}}/{full_config_path}"

                config_filename = os.path.basename(full_config_path)

                github_url = f"https://github.com/NVIDIA/TensorRT-LLM/blob/main/{full_config_path}"
                config_link = f"`{config_filename} <{github_url}>`_"

                lines.append(f"   * - {gpu_display}")
                lines.append(f"     - {profile}")
                lines.append(f"     - {isl} / {osl}")
                lines.append(f"     - {conc}")
                lines.append(f"     - {config_link}")
                lines.append(f"     - ``{command}``")

        lines.append("")
        lines.append(f".. end-{model}")
        lines.append("")

    output_text = "\n".join(lines)
    if output_file:
        with open(output_file, "w") as f:
            f.write(output_text)
        print(f"Generated table written to: {output_file}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    yaml_path = DATABASE_LIST_PATH
    if not yaml_path.exists():
        print(f"Error: YAML file not found at {yaml_path}", file=sys.stderr)
        sys.exit(1)
    output_path = REPO_ROOT / "docs/source/deployment-guide/config_table.rst"
    generate_rst(yaml_path, output_file=output_path)
