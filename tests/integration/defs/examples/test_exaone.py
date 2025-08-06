# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Module test_exaone test exaone examples."""

import pytest
from defs.common import (convert_weights, generate_summary_cmd, venv_check_call,
                         venv_mpi_check_call)
from defs.conftest import get_sm_version, skip_post_blackwell
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@skip_post_blackwell
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("data_type", ['bfloat16', 'float16'])
@pytest.mark.parametrize("llm_exaone_model_root",
                         ['exaone_3.0_7.8b_instruct', 'exaone_deep_2.4b'],
                         indirect=True)
@pytest.mark.parametrize("use_weight_only",
                         [pytest.param(True, marks=skip_post_blackwell), False],
                         ids=["enable_weight_only", "disable_weight_only"])
def test_llm_exaone_1gpu(data_type, exaone_example_root, llm_exaone_model_root,
                         llama_example_root, llm_datasets_root, llm_rouge_root,
                         llm_venv, cmodel_dir, engine_dir, num_beams,
                         use_weight_only, timeout_manager):

    print("Build engines...")
    model_name = "exaone"

    # Convert weights with timeout management
    with timeout_manager.timed_operation("convert"):
        model_dir = convert_weights(
            llm_venv=llm_venv,
            # NOTE
            # EXAONE is based on llama so reuse llama's checkpoint converter
            example_root=llama_example_root,
            cmodel_dir=cmodel_dir,
            model=model_name,
            model_path=llm_exaone_model_root,
            data_type=data_type,
            use_weight_only=use_weight_only,
            timeout=timeout_manager.remaining_timeout)

    # Build engines with timeout management
    with timeout_manager.timed_operation("build"):
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={model_dir}",
            f"--output_dir={engine_dir}",
            f"--max_beam_width={num_beams}",
        ]
        check_call(" ".join(build_cmd),
                   shell=True,
                   env=llm_venv._new_env,
                   timeout=timeout_manager.remaining_timeout)

    rouge1_threshold = {
        1: 22,
        2: 22,
        4: 23,
    }[num_beams]

    # Run summary with timeout management
    print("Run summarize...")
    summary_cmd = generate_summary_cmd(
        exaone_example_root,
        hf_model_dir=llm_exaone_model_root,
        engine_dir=engine_dir,
        data_type=data_type,
        tensorrt_llm_rouge1_threshold=rouge1_threshold,
        use_py_session=False,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root,
        num_beams=num_beams,
    )

    with timeout_manager.timed_operation("summary"):
        venv_check_call(llm_venv,
                        summary_cmd,
                        timeout=timeout_manager.remaining_timeout)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("num_beams", [1],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("data_type", ['float16'])
@pytest.mark.parametrize("llm_exaone_model_root",
                         ['exaone_3.0_7.8b_instruct', 'exaone_deep_2.4b'],
                         indirect=True)
def test_llm_exaone_2gpu(data_type, exaone_example_root, llm_exaone_model_root,
                         llama_example_root, llm_datasets_root, llm_rouge_root,
                         llm_venv, cmodel_dir, engine_dir, num_beams,
                         timeout_manager):

    tp_size = 2
    print("Build engines...")
    model_name = "exaone"

    # Convert weights with timeout management
    with timeout_manager.timed_operation("convert"):
        model_dir = convert_weights(
            llm_venv=llm_venv,
            # NOTE
            # EXAONE is based on llama so reuse llama's checkpoint converter
            example_root=llama_example_root,
            cmodel_dir=cmodel_dir,
            model=model_name,
            model_path=llm_exaone_model_root,
            data_type=data_type,
            tp_size=tp_size,
            pp_size=1,
            timeout=timeout_manager.remaining_timeout)

    # Build engines with timeout management
    with timeout_manager.timed_operation("build"):
        build_cmd = [
            "trtllm-build", f"--checkpoint_dir={model_dir}",
            f"--output_dir={engine_dir}", f"--max_beam_width={num_beams}"
        ]
        check_call(" ".join(build_cmd),
                   shell=True,
                   env=llm_venv._new_env,
                   timeout=timeout_manager.remaining_timeout)

    # Run summary with timeout management
    print("Run summarize...")
    summary_cmd = generate_summary_cmd(
        exaone_example_root,
        hf_model_dir=llm_exaone_model_root,
        engine_dir=engine_dir,
        data_type=data_type,
        tensorrt_llm_rouge1_threshold=22,
        use_py_session=False,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root,
        num_beams=num_beams,
    )

    with timeout_manager.timed_operation("summary"):
        venv_mpi_check_call(
            llm_venv, ["mpirun", "-n", f"{tp_size}", "--allow-run-as-root"],
            summary_cmd,
            timeout=timeout_manager.remaining_timeout)
