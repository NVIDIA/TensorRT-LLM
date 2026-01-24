# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Portions of this file are derived from VBench,
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Shanghai AI Laboratory.
#
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
#
# This file includes code adapted from the VBench repository (https://github.com/Vchitect/VBench).
# The original code is licensed under the Apache License, Version 2.0.
import argparse
import json

DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

SEMANTIC_WEIGHT = 1
QUALITY_WEIGHT = 4

QUALITY_LIST = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
]

SEMANTIC_LIST = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]


def get_normalized_score(ori_data):
    normalized_score = {}
    for dim in DIM_WEIGHT:
        dim_shift = dim.replace(" ", "_")
        min_val = NORMALIZE_DIC[dim]["Min"]
        max_val = NORMALIZE_DIC[dim]["Max"]
        normalized_score[dim] = (ori_data[dim_shift][0] - min_val) / (max_val - min_val)
        normalized_score[dim] = normalized_score[dim] * DIM_WEIGHT[dim]
    return normalized_score


def get_quality_score(normalized_score):
    quality_score = []
    for key in QUALITY_LIST:
        quality_score.append(normalized_score[key])
    quality_score = sum(quality_score) / sum([DIM_WEIGHT[i] for i in QUALITY_LIST])
    return quality_score


def get_semantic_score(normalized_score):
    semantic_score = []
    for key in SEMANTIC_LIST:
        semantic_score.append(normalized_score[key])
    semantic_score = sum(semantic_score) / sum([DIM_WEIGHT[i] for i in SEMANTIC_LIST])
    return semantic_score


def get_final_score(quality_score, semantic_score):
    return (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT) / (
        QUALITY_WEIGHT + SEMANTIC_WEIGHT
    )


def main():
    parser = argparse.ArgumentParser(description="Load submission file")
    parser.add_argument(
        "--eval_results_path",
        type=str,
        required=True,
        help="Path to the Evaluation Results File",
        default="evaluation_results.json",
    )
    args = parser.parse_args()
    with open(args.eval_results_path, "r") as f:
        eval_results = json.load(f)
    normalized_score = get_normalized_score(eval_results)
    quality_score = get_quality_score(normalized_score)
    semantic_score = get_semantic_score(normalized_score)
    final_score = get_final_score(quality_score, semantic_score)
    print(f"Quality Score: {quality_score}")
    print(f"Semantic Score: {semantic_score}")
    print(f"Final Score: {final_score}")


if __name__ == "__main__":
    main()
