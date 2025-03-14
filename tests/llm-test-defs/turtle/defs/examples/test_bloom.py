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
"""Module test_bloom test bloom examples."""
import pytest
from defs.common import (convert_weights, generate_summary_cmd, venv_check_call,
                         venv_mpi_check_call)
from defs.conftest import skip_post_blackwell
from defs.trt_test_alternative import check_call


@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("use_gpt_plugin", [True, False],
                         ids=["enable_gpt_plugin", "disable_gpt_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("use_weight_only", [True, False],
                         ids=["enable_weight_only", "disable_weight_only"])
def test_llm_bloom_560m_1node_1gpus(bloom_example_root,
                                    llm_bloom_560m_model_root,
                                    llm_datasets_root, llm_rouge_root, llm_venv,
                                    cmodel_dir, engine_dir, use_gpt_plugin,
                                    use_gemm_plugin, use_weight_only,
                                    num_beams):
    "Build & Run bloom 560m with one gpu"
    print("Building engines...")
    dtype = "float16"
    model_name = "bloom-560M-weight_only" if use_weight_only else "bloom-560M"
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=bloom_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=llm_bloom_560m_model_root,
                                data_type=dtype,
                                use_weight_only=use_weight_only)
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}", f"--max_batch_size=1",
        f"--max_input_len=1024", f"--max_num_tokens=1024",
        f"--output_dir={engine_dir}", f"--max_beam_width={num_beams}"
    ]
    if use_gpt_plugin:
        build_cmd.append(f"--gpt_attention_plugin={dtype}")
        build_cmd.append("--remove_input_padding=enable")
    else:
        build_cmd.append(f"--gpt_attention_plugin=disable")
        build_cmd.append(f"--paged_kv_cache=disable")
        build_cmd.append(f"--remove_input_padding=disable")

    if use_gemm_plugin:
        build_cmd.append(f"--gemm_plugin={dtype}")
    else:
        build_cmd.append(f"--gemm_plugin=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run bloom 560m...')
    summary_cmd = generate_summary_cmd(bloom_example_root,
                                       hf_model_dir=llm_bloom_560m_model_root,
                                       data_type="fp16",
                                       engine_dir=engine_dir,
                                       num_beams=num_beams,
                                       tensorrt_llm_rouge1_threshold="13.8",
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("per_token_channel", [True, False],
                         ids=["enable_ptpc", "disable_ptpc"])
def test_llm_bloom_560m_smooth_single_gpu_summary(bloom_example_root, llm_venv,
                                                  llm_bloom_560m_model_root,
                                                  llm_datasets_root,
                                                  llm_rouge_root, cmodel_dir,
                                                  per_token_channel, engine_dir,
                                                  num_beams):
    "bloom-560m-smooth test on single gpu"
    dtype = "float16"
    per_channel = per_token = False

    if per_token_channel:
        per_channel = per_token = True

    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=bloom_example_root,
        cmodel_dir=cmodel_dir,
        model="bloom-smooth",
        model_path=llm_bloom_560m_model_root,
        data_type=dtype,
        smoothquant=0.5,
        per_channel=per_channel,
        per_token=per_token,
        calib_dataset=f"{llm_datasets_root}/cimec/lambada")

    print("Building engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", f"--gpt_attention_plugin={dtype}",
        f"--max_beam_width={num_beams}"
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    summary_cmd = generate_summary_cmd(bloom_example_root,
                                       hf_model_dir=llm_bloom_560m_model_root,
                                       data_type="fp16",
                                       engine_dir=engine_dir,
                                       num_beams=num_beams,
                                       tensorrt_llm_rouge1_threshold="13",
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
def test_llm_bloom_560m_1node_2gpus(bloom_example_root,
                                    llm_bloom_560m_model_root,
                                    llm_datasets_root, llm_rouge_root, llm_venv,
                                    cmodel_dir, engine_dir, num_beams):
    "Build & Run bloom 560m with two gpus"
    print("Building engines...")
    dtype = 'float16'

    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=bloom_example_root,
                                cmodel_dir=cmodel_dir,
                                model="bloom-560M",
                                model_path=llm_bloom_560m_model_root,
                                data_type=dtype,
                                gpus=2)
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--max_beam_width={num_beams}",
        f"--output_dir={engine_dir}",
        f"--workers={2}",
        f"--max_batch_size={8}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        "--paged_kv_cache=enable",
        "--remove_input_padding=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run bloom 560m...')
    summary_cmd = generate_summary_cmd(bloom_example_root,
                                       hf_model_dir=llm_bloom_560m_model_root,
                                       data_type="fp16",
                                       num_beams=num_beams,
                                       engine_dir=engine_dir,
                                       tensorrt_llm_rouge1_threshold="14",
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("embedding_sharding_dim", [0, 1])
def test_llm_bloom_176b_1node_8gpus(bloom_example_root,
                                    llm_bloom_176b_model_root,
                                    llm_datasets_root, llm_rouge_root, llm_venv,
                                    engine_dir, embedding_sharding_dim,
                                    num_beams, cmodel_dir):
    """
        Build & Run bloom 176b with 8 gpus.
        This case don't support disabled plugins
    """
    print("Building engines...")

    dtype = 'float16'
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=bloom_example_root,
                                cmodel_dir=cmodel_dir,
                                model="bloom-176B",
                                model_path=llm_bloom_176b_model_root,
                                data_type=dtype,
                                gpus=8,
                                tp_size=8,
                                use_parallel_embedding=True,
                                embedding_sharding_dim=embedding_sharding_dim,
                                workers=8)

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--max_beam_width={num_beams}",
        f"--output_dir={engine_dir}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        f"--workers={8}",
        f"--max_batch_size={8}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run bloom 176b...')
    summary_cmd = generate_summary_cmd(bloom_example_root,
                                       hf_model_dir=llm_bloom_176b_model_root,
                                       data_type="fp16",
                                       num_beams=num_beams,
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "8", "--allow-run-as-root"],
                        summary_cmd)


@skip_post_blackwell
def test_llm_bloom_560m_int8_kv_single_gpu_summary(bloom_example_root,
                                                   llm_bloom_560m_model_root,
                                                   llm_datasets_root,
                                                   llm_rouge_root, llm_venv,
                                                   cmodel_dir, engine_dir):
    "bloom-560m with int8 kv test on single gpu"
    model_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=bloom_example_root,
        cmodel_dir=cmodel_dir,
        model="bloom-560m-kv",
        model_path=llm_bloom_560m_model_root,
        int8_kv_cache=True,
        use_weight_only=True,
        calib_dataset=f"{llm_datasets_root}/cimec/lambada")

    print("Building engines...")
    dtype = 'float16'
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    summary_cmd = generate_summary_cmd(bloom_example_root,
                                       hf_model_dir=llm_bloom_560m_model_root,
                                       data_type="fp16",
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root,
                                       tensorrt_llm_rouge1_threshold=14)

    venv_check_call(llm_venv, summary_cmd)
