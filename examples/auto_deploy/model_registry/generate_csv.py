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

"""Generate a CSV from models.yaml with HF model id, link, and build_and_run_ad.py command."""

import csv
from pathlib import Path

import yaml

# Paths relative to TensorRT-LLM repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_YAML = Path(__file__).resolve().parent / "models.yaml"
BUILD_SCRIPT = "examples/auto_deploy/build_and_run_ad.py"
OUTPUT_CSV = Path(__file__).resolve().parent / "models.csv"

HF_BASE_URL = "https://huggingface.co"


def build_command(model_name: str) -> str:
    return f"python {BUILD_SCRIPT} --model {model_name} --use-registry"


def main():
    with open(MODELS_YAML) as f:
        data = yaml.safe_load(f)

    rows = []
    for entry in data.get("models", []):
        name = entry["name"]
        hf_link = f"{HF_BASE_URL}/{name}"
        command = build_command(name)
        rows.append(
            {"hf_model_id": name, "hf_model_link": hf_link, "build_and_run_ad_command": command}
        )

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["hf_model_id", "hf_model_link", "build_and_run_ad_command"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} models to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
