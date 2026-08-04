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

import os

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


@pytest.mark.parametrize("gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("dtype", ['bfloat16', 'float16'])
@pytest.mark.parametrize("mamba_model_root", [
    pytest.param('mamba-130m', marks=skip_post_blackwell), 'mamba-2.8b',
    'mamba-1.4b', 'mamba-790m', 'mamba-370m', 'mamba2-130m', 'mamba2-2.7b',
    'mamba2-1.3b', 'mamba2-780m', 'mamba2-370m',
    pytest.param('mamba-codestral-7B-v0.1', marks=skip_post_blackwell)
],
                         indirect=True)
def test_llm_mamba_1gpu(mamba_example_root, mamba_model_root,
                        llm_gptneox_model_root, llm_mathstral_model_root,
                        llm_datasets_root, llm_rouge_root, llm_venv,
                        gemm_plugin, dtype, cmodel_dir, engine_dir):
    "Build & Run mamba model with one gpu"
    print("Build engines...")

    model_name = os.path.basename(mamba_model_root)
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=mamba_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=mamba_model_root,
                                data_type=dtype)
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--paged_kv_cache=disable",
        "--max_batch_size=8",
    ]
    if gemm_plugin:
        build_cmd.append("--gemm_plugin=auto")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print(f'Run {model_name}...')
    tokenizer_dir = llm_mathstral_model_root if model_name == "mamba-codestral-7B-v0.1" else llm_gptneox_model_root
    summary_cmd = generate_summary_cmd(mamba_example_root,
                                       hf_model_dir=mamba_model_root,
                                       tokenizer_dir=tokenizer_dir,
                                       data_type=dtype,
                                       engine_dir=engine_dir,
                                       batch_size=8,
                                       tensorrt_llm_rouge1_threshold="13.5",
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("mamba_model_root", ['mamba-codestral-7B-v0.1'],
                         indirect=True)
def test_llm_mamba2_2gpu(mamba_example_root, mamba_model_root,
                         llm_gptneox_model_root, llm_mathstral_model_root,
                         llm_datasets_root, llm_rouge_root, llm_venv,
                         cmodel_dir, engine_dir):
    "Build & Run mamba2 model with two gpus"
    print("Build engines...")

    model_name = mamba_model_root.split('/')[-1]
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=mamba_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=mamba_model_root,
                                data_type='float16',
                                tp_size=2)
    build_cmd = [
        "trtllm-build",
        "--gemm_plugin=auto",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--paged_kv_cache=disable",
        "--max_batch_size=8",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print(f'Run {model_name}...')
    tokenizer_dir = llm_mathstral_model_root
    summary_cmd = generate_summary_cmd(mamba_example_root,
                                       hf_model_dir=mamba_model_root,
                                       tokenizer_dir=tokenizer_dir,
                                       data_type='float16',
                                       engine_dir=engine_dir,
                                       batch_size=8,
                                       tensorrt_llm_rouge1_threshold="19.0",
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)
