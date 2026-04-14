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
from defs.common import convert_weights, venv_check_call
from defs.conftest import get_sm_version, skip_post_blackwell
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@skip_post_blackwell
@pytest.mark.parametrize("batch_size", [1, 8], ids=['bs1', 'bs8'])
@pytest.mark.parametrize("data_type", ['bfloat16'])
@pytest.mark.parametrize("num_medusa_heads", [4], ids=['4-heads'])
@pytest.mark.parametrize("medusa_model_roots", ["medusa-vicuna-7b-v1.3"],
                         indirect=True)
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_medusa_1gpu(batch_size, data_type, medusa_model_roots,
                         medusa_example_root, llm_datasets_root, llm_rouge_root,
                         num_medusa_heads, llm_venv, cmodel_dir, engine_dir,
                         use_py_session):
    print("Build engines...")
    model_name = "medusa"

    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=medusa_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=medusa_model_roots,
                                data_type=data_type)

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        f"--max_beam_width=1",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--max_input_len=1024",
        "--max_seq_len=1536",
        f"--max_batch_size={batch_size}",
        "--paged_kv_cache=enable",
        '--speculative_decoding_mode=medusa',
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")

    summary_cmd = [
        f"{medusa_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{medusa_model_roots[0]}", "--tokenizer_dir",
        f"{medusa_model_roots[0]}", f"--engine_dir={engine_dir}",
        "--check_accuracy", "--tensorrt_llm_rouge1_threshold=24",
        "--medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]",
        f"--temperature=1.0", f"--max_ite=40", f"--batch_size={batch_size}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    if use_py_session:
        summary_cmd.append("--use_py_session")

    venv_check_call(llm_venv, summary_cmd)


@skip_post_blackwell
@pytest.mark.parametrize("batch_size", [1, 8], ids=['bs1', 'bs8'])
@pytest.mark.parametrize("data_type", ['bfloat16', 'float16'])
@pytest.mark.parametrize("num_medusa_heads", [4], ids=['4-heads'])
@pytest.mark.parametrize("medusa_model_roots", ["medusa-vicuna-7b-v1.3"],
                         indirect=True)
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
@pytest.mark.parametrize("base_model_datatype", ['fp8'])
def test_llm_medusa_with_qaunt_base_model_1gpu(
        batch_size, data_type, medusa_model_roots, medusa_example_root,
        base_model_datatype, llm_datasets_root, llm_rouge_root,
        num_medusa_heads, llm_venv, cmodel_dir, engine_dir, use_py_session):

    model_name = f"vicuna_meudsa_quant_base_mode_{base_model_datatype}"
    quant_model_ckpt_output_path = os.path.join(cmodel_dir, model_name)

    print("Quant base model to FP8 and combine medusa head")
    quant_cmd = [
        f"{medusa_example_root}/../quantization/quantize.py",
        f"--model_dir={medusa_model_roots[0]}", f"--dtype={data_type}",
        f"--qformat={base_model_datatype}",
        f"--kv_cache_dtype={base_model_datatype}",
        f"--output_dir={quant_model_ckpt_output_path}", "--calib_size=512",
        f"--medusa_model_dir={medusa_model_roots[1]}",
        f"--num_medusa_heads={num_medusa_heads}"
    ]

    # https://nvbugs/4658787
    # WAR before medusa tests can work offline
    env = {"HF_DATASETS_OFFLINE": "0"}
    venv_check_call(llm_venv, quant_cmd, env=env)

    print("Build engines...")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={quant_model_ckpt_output_path}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        f"--max_beam_width=1",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--max_input_len=1024",
        "--max_seq_len=1536",
        f"--max_batch_size={batch_size}",
        "--paged_kv_cache=enable",
        '--speculative_decoding_mode=medusa',
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")

    summary_cmd = [
        f"{medusa_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{medusa_model_roots[0]}", "--tokenizer_dir",
        f"{medusa_model_roots[0]}", f"--engine_dir={engine_dir}",
        "--check_accuracy", "--tensorrt_llm_rouge1_threshold=24",
        "--medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]",
        f"--temperature=1.0", f"--max_ite=40", f"--batch_size={batch_size}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    if use_py_session:
        summary_cmd.append("--use_py_session")

    venv_check_call(llm_venv, summary_cmd)
