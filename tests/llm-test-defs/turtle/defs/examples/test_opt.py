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
from defs.common import convert_weights, venv_check_call, venv_mpi_check_call
from defs.conftest import get_device_memory
from defs.trt_test_alternative import check_call

OPT_LIST = {
    "opt-125m": {
        "build": [],
        "infer": ["--tensorrt_llm_rouge1_threshold=14"]
    },
    "opt-350m": {
        "build": [],
        "infer": ["--tensorrt_llm_rouge1_threshold=19"]
    },
    "opt-2.7b": {
        "build": [],
        "infer": ["--tensorrt_llm_rouge1_threshold=20"]
    }
}


@pytest.mark.parametrize("llm_opt_model_root",
                         ['opt-125m', 'opt-350m', 'opt-2.7b'],
                         indirect=True)
@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("context_fmha_type",
                         ['enabled', 'enabled_with_fp32_acc', 'disabled'])
def test_llm_opt_single_gpu_summary(opt_example_root, llm_venv,
                                    llm_opt_model_root, llm_datasets_root,
                                    llm_rouge_root, cmodel_dir, engine_dir,
                                    use_attention_plugin, use_gemm_plugin,
                                    context_fmha_type):
    "Build & run opt summary on single gpu"
    model_name = os.path.basename(llm_opt_model_root)
    dtype = "float16"
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=opt_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llm_opt_model_root,
                                data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--output_dir={engine_dir}",
    ]

    if use_attention_plugin:
        build_cmd.append(f"--gpt_attention_plugin={dtype}")
        if context_fmha_type == 'enabled':
            build_cmd.append("--context_fmha=enable")
        elif context_fmha_type == 'disabled':
            build_cmd.append("--context_fmha=disable")
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--context_fmha=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
        ])

    if use_gemm_plugin:
        build_cmd.append(f"--gemm_plugin={dtype}")
    else:
        build_cmd.append("--gemm_plugin=disable")

    build_cmd.extend(OPT_LIST[model_name]['build'])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = [
        f"{opt_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_opt_model_root}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    summary_cmd.extend(OPT_LIST[model_name]['infer'])
    if context_fmha_type == "enabled_with_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("llm_opt_model_root", ['opt-66b'], indirect=True)
@pytest.mark.parametrize("use_plugin", [True, False],
                         ids=["enable_plugin", "disable_plugin"])
@pytest.mark.parametrize("context_fmha_type",
                         ['enabled', 'enabled_with_fp32_acc', 'disabled'])
def test_llm_opt_4gpus_summary(opt_example_root, llm_venv, cmodel_dir,
                               engine_dir, llm_opt_model_root,
                               llm_datasets_root, llm_rouge_root, use_plugin,
                               context_fmha_type):
    "Build & run opt 66b summary on single node 4 gpus"
    if not use_plugin and get_device_memory() < 50000:
        pytest.skip("device memory is insufficient.")

    model_name = os.path.basename(llm_opt_model_root)
    dtype = "float16"
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=opt_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llm_opt_model_root,
                                data_type=dtype,
                                gpus=4)

    print("Building engines...")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--output_dir={engine_dir}",
        f"--workers={4}",
    ]
    if use_plugin:
        build_cmd.append(f"--gpt_attention_plugin={dtype}")
        build_cmd.append(f"--gemm_plugin={dtype}")
        if context_fmha_type == 'enabled':
            build_cmd.append("--context_fmha=enable")
        elif context_fmha_type == 'disabled':
            build_cmd.append("--context_fmha=disable")
    else:
        build_cmd.extend([
            "--gpt_attention_plugin=disable",
            "--context_fmha=disable",
            "--paged_kv_cache=disable",
            "--remove_input_padding=disable",
            "--gemm_plugin=disable",
        ])

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = [
        f"{opt_example_root}/../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        f"{llm_opt_model_root}",
        "--data_type",
        "fp16",
        "--check_accuracy",
        f"--engine_dir={engine_dir}",
        "--tensorrt_llm_rouge1_threshold=18",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
        '--kv_cache_free_gpu_memory_fraction=0.8',
    ]
    if context_fmha_type == "enabled_with_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "4", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("embedding_sharding_dim", [0, 1])
@pytest.mark.parametrize("llm_opt_model_root",
                         ['opt-125m', 'opt-350m', 'opt-2.7b'],
                         indirect=True)
def test_llm_opt_parallel_embedding_2gpu(opt_example_root, llm_venv,
                                         llm_opt_model_root, llm_datasets_root,
                                         llm_rouge_root, cmodel_dir, engine_dir,
                                         embedding_sharding_dim):
    "OPT with parallel embedding"
    print("Converting OPT model into FastTransformer format...")
    model_name = os.path.basename(llm_opt_model_root)
    dtype = "float16"
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=opt_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llm_opt_model_root,
                                data_type=dtype,
                                gpus=2,
                                use_parallel_embedding=True,
                                embedding_sharding_dim=embedding_sharding_dim)

    print("Building engines...")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--output_dir={engine_dir}",
        f"--workers={2}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    summary_cmd = [
        f"{opt_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_opt_model_root}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        "--tensorrt_llm_rouge1_threshold=14",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_mpi_check_call(llm_venv, ["mpirun", "--allow-run-as-root", "-np", "2"],
                        summary_cmd)
