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
from defs.common import convert_weights, parse_mpi_cmd, venv_mpi_check_call
from defs.conftest import get_device_memory, get_sm_version
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


# @pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "use_gpt_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("context_fmha_type", [
    "enable_context_fmha", "enable_context_fmha_fp32_acc",
    "disable_context_fmha"
])
@pytest.mark.parametrize("dtype", ['float16', 'bfloat16'])
def test_llm_internlm2_7b_1node_1gpu(internlm2_example_root,
                                     llm_internlm2_7b_model_root,
                                     llm_datasets_root, llm_rouge_root,
                                     llm_venv, cmodel_dir, engine_dir,
                                     use_gpt_attention_plugin, use_gemm_plugin,
                                     context_fmha_type, dtype, num_beams):
    "Build & Run internlm2-7b with 1 gpu"
    if dtype == "bfloat16" and not use_gemm_plugin:
        pytest.skip("Please use gemm plugin when dtype is bfloat16.")
    if num_beams == 4 and get_device_memory() < 50000:
        pytest.skip("device memory is insufficient.")

    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=f"{internlm2_example_root}",
                                cmodel_dir=cmodel_dir,
                                model="internlm2-7b",
                                model_path=llm_internlm2_7b_model_root,
                                data_type=dtype,
                                gpus=1,
                                tp_size=1)

    build_cmd = [
        "python3 -m tensorrt_llm.commands.build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--max_beam_width={num_beams}",
        f"--max_batch_size=1",
    ]

    if use_gpt_attention_plugin:
        build_cmd.append("--remove_input_padding=enable")
        build_cmd.append(f"--gpt_attention_plugin={dtype}")
    else:
        build_cmd.append("--gpt_attention_plugin=disable")
        build_cmd.append("--remove_input_padding=disable")
        build_cmd.append("--paged_kv_cache=disable")

    if use_gemm_plugin:
        build_cmd.append(f"--gemm_plugin={dtype}")
    else:
        build_cmd.append("--gemm_plugin=disable")

    if context_fmha_type == "enable_context_fmha":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == "disable_context_fmha":
        build_cmd.append("--context_fmha=disable")

    print("Building engines...")
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run internlm2-7b...')
    data_type = "fp16" if dtype == "float16" else "bf16"
    summary_cmd = [
        f"{internlm2_example_root}/../../../summarize.py", "--test_trt_llm",
        "--hf_model_dir", llm_internlm2_7b_model_root, "--engine_dir",
        engine_dir, "--data_type", data_type, "--check_accuracy",
        f"--num_beams={num_beams}", f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}"
    ]
    if context_fmha_type == "enable_context_fmha_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_mpi_check_call(
        llm_venv, parse_mpi_cmd(["mpirun", "-n", "1", "--allow-run-as-root"]),
        summary_cmd)
