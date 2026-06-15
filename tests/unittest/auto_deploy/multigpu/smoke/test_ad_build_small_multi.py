# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Testing build_and_run_ad end2end."""

import pytest
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize(
    "model_hub_id, llm_extra_args",
    [
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            {
                "transforms": {
                    "insert_cached_attention": {"backend": "flashinfer"},
                    # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/9878
                    # "compile_model": {"backend": "torch-opt"},
                    "compile_model": {"backend": "torch-cudagraph"},
                },
            },
        ),
    ],
)
def test_build_ad(world_size: int, model_hub_id: str, llm_extra_args: dict):
    experiment_config = get_small_model_config(model_hub_id, **llm_extra_args)

    experiment_config["args"]["world_size"] = world_size
    experiment_config["args"]["runtime"] = "trtllm"  # Default runtime set to trtllm

    experiment_config = ExperimentConfig(**experiment_config)
    print(f"Experiment Config: {experiment_config}")
    main(experiment_config)
