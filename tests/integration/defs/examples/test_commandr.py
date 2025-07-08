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
"""Module test_commandr test commandr examples."""
import os
import time

import pytest
from defs.common import (convert_weights, generate_summary_cmd, venv_check_call,
                         venv_mpi_check_call)
from defs.conftest import (get_gpu_device_list, get_sm_version,
                           skip_post_blackwell)
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@pytest.mark.skip_less_device_memory(80000)
@skip_post_blackwell
@pytest.mark.parametrize("use_weight_only", [True, False],
                         ids=["enable_weight_only", "disable_weight_only"])
def test_llm_commandr_v01_single_gpu_summary(commandr_example_root,
                                             llm_commandr_v01_model_root,
                                             llm_datasets_root, llm_rouge_root,
                                             llm_venv, cmodel_dir, engine_dir,
                                             use_weight_only):
    "Build & run commandr_v01 on single gpu."
    if "GH200" in get_gpu_device_list()[0] and not use_weight_only:
        pytest.skip("OOM on GH200. https://nvbugs/5250460")

    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_commandr_v01_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=commandr_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_commandr_v01_model_root,
                               data_type=dtype,
                               use_weight_only=use_weight_only)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = [
        f"{commandr_example_root}/../../../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        f"{llm_commandr_v01_model_root}",
        "--data_type",
        "fp16",
        "--check_accuracy",
        f"--engine_dir={engine_dir}",
        "--tensorrt_llm_rouge1_threshold=12",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
    ]

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_host_memory(1000000)
@pytest.mark.parametrize("use_weight_only",
                         [pytest.param(True, marks=skip_post_blackwell), False],
                         ids=["enable_weight_only", "disable_weight_only"])
def test_llm_commandr_plus_4gpus_summary(commandr_example_root,
                                         llm_commandr_plus_model_root,
                                         llm_datasets_root, llm_rouge_root,
                                         llm_venv, cmodel_dir, engine_dir,
                                         use_weight_only, timeout_from_marker):
    "Build & run Command-R+ with smoothquant on 4 gpus."
    dtype = 'float16'
    tp_size = 4
    model_name = os.path.basename(llm_commandr_plus_model_root)
    print("Converting checkpoint...")
    remaining_timeout = timeout_from_marker
    convert_start = time.time()
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=commandr_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_commandr_plus_model_root,
                               data_type=dtype,
                               tp_size=tp_size,
                               gpus=tp_size,
                               use_weight_only=use_weight_only,
                               timeout=remaining_timeout)
    convert_time = time.time() - convert_start
    remaining_timeout -= convert_time
    if remaining_timeout <= 0:
        raise TimeoutError("Timeout exceeded after convert phase!")

    print("Building engines...")
    build_start = time.time()
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={4}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
    ]

    run_cmd = [
        f"{commandr_example_root}/../../../run.py",
        f"--max_output_len={50}",
        f"--tokenizer_dir={llm_commandr_plus_model_root}",
        f"--engine_dir={engine_dir}",
    ]

    check_call(" ".join(build_cmd),
               shell=True,
               env=llm_venv._new_env,
               timeout=remaining_timeout)
    build_time = time.time() - build_start
    remaining_timeout -= build_time
    if remaining_timeout <= 0:
        raise TimeoutError("Timeout exceeded after build phase!")

    print("Running engines...")
    run_start = time.time()
    venv_mpi_check_call(
        llm_venv,
        ["mpirun", "-n", str(tp_size), "--allow-run-as-root"],
        run_cmd,
        timeout=remaining_timeout)
    run_time = time.time() - run_start
    remaining_timeout -= run_time
    if remaining_timeout <= 0:
        raise TimeoutError("Timeout exceeded after run phase!")

    print("Running summary...")
    time.time()
    summary_cmd = generate_summary_cmd(
        commandr_example_root,
        hf_model_dir=llm_commandr_plus_model_root,
        data_type="fp16",
        engine_dir=engine_dir,
        dataset_dir=llm_datasets_root,
        rouge_dir=llm_rouge_root)

    venv_mpi_check_call(
        llm_venv,
        ["mpirun", "-n", str(tp_size), "--allow-run-as-root"],
        summary_cmd,
        timeout=remaining_timeout)
