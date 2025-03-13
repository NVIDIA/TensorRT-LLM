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

import pytest
from defs.common import (convert_weights, generate_summary_cmd, venv_check_call,
                         venv_mpi_check_call)
from defs.trt_test_alternative import check_call


@pytest.mark.skip_less_device_memory(40000)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("deepseek_v2_model_root", ['DeepSeek-V2-Lite'],
                         indirect=True)
def test_llm_deepseek_v2_lite_summary(deepseek_v2_example_root,
                                      deepseek_v2_model_root, llm_datasets_root,
                                      llm_rouge_root, llm_venv, cmodel_dir,
                                      engine_dir, num_beams):
    model_name = 'deepseek_v2_lite'

    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=deepseek_v2_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=deepseek_v2_model_root,
                                data_type="bfloat16")
    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=bfloat16",
        f"--max_beam_width={num_beams}",
        "--use_paged_context_fmha=enable",
        "--max_seq_len=4096",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = generate_summary_cmd(deepseek_v2_example_root,
                                       hf_model_dir=deepseek_v2_model_root,
                                       data_type="bf16",
                                       engine_dir=engine_dir,
                                       num_beams=num_beams,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_host_memory(1536_000)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("deepseek_v2_model_root", ['DeepSeek-V2'],
                         indirect=True)
def test_llm_deepseek_v2_8gpu_summary(deepseek_v2_example_root,
                                      deepseek_v2_model_root, llm_datasets_root,
                                      llm_rouge_root, llm_venv, cmodel_dir,
                                      engine_dir, num_beams):
    model_name = 'deepseek_v2'

    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=deepseek_v2_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=deepseek_v2_model_root,
                                data_type="bfloat16",
                                gpus=8,
                                workers=8,
                                tp_size=8,
                                pp_size=1)
    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=bfloat16",
        f"--max_beam_width={num_beams}",
        "--use_paged_context_fmha=enable",
        "--max_seq_len=4096",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = generate_summary_cmd(deepseek_v2_example_root,
                                       hf_model_dir=deepseek_v2_model_root,
                                       data_type="bf16",
                                       engine_dir=engine_dir,
                                       num_beams=num_beams,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "8", "--allow-run-as-root"],
                        summary_cmd)
