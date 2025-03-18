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
from defs.common import convert_weights, venv_check_call
from defs.conftest import skip_post_blackwell, skip_pre_ada
from defs.trt_test_alternative import check_call


@pytest.mark.parametrize("use_dynamic_tree", [False, True],
                         ids=['eagle1', 'eagle2'])
@pytest.mark.parametrize("batch_size", [1, 8], ids=['bs1', 'bs8'])
@pytest.mark.parametrize("data_type", ['float16'])
@pytest.mark.parametrize("eagle_model_roots", ["EAGLE-Vicuna-7B-v1.3"],
                         indirect=True)
def test_llm_eagle_1gpu(batch_size, data_type, use_dynamic_tree,
                        eagle_model_roots, eagle_example_root,
                        llm_datasets_root, llm_rouge_root, llm_venv, cmodel_dir,
                        engine_dir):
    print("Build engines...")
    model_name = "eagle"

    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=eagle_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=eagle_model_roots,
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
        "--use_paged_context_fmha=enable",
        "--max_input_len=1024",
        "--max_seq_len=1536",
        f"--max_batch_size={batch_size}",
        "--paged_kv_cache=enable",
        '--speculative_decoding_mode=eagle',
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run run...")
    run_cmd = [
        f"{eagle_example_root}/../run.py",
        "--max_output_len=100",
        f"--tokenizer_dir={eagle_model_roots[0]}",
        "--log_level=verbose",
        f"--engine_dir={engine_dir}",
    ]
    if use_dynamic_tree:
        run_cmd.extend(
            [f"--eagle_dynamic_tree_max_top_k={3}", "--eagle_use_dynamic_tree"])

    venv_check_call(llm_venv, run_cmd)

    print("Run summarize...")
    summary_cmd = [
        f"{eagle_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{eagle_model_roots[0]}", "--tokenizer_dir",
        f"{eagle_model_roots[0]}", f"--engine_dir={engine_dir}",
        "--check_accuracy", "--tensorrt_llm_rouge1_threshold=24",
        "--eagle_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]",
        f"--max_ite=40", f"--batch_size={batch_size}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    if use_dynamic_tree:
        summary_cmd.extend(
            [f"--eagle_dynamic_tree_max_top_k={3}", "--eagle_use_dynamic_tree"])

    venv_check_call(llm_venv, summary_cmd)


# TODO: remove skip_post_blackwell after Speculative decoding is supported.
@skip_post_blackwell
@skip_pre_ada
@pytest.mark.parametrize("batch_size", [8], ids=['bs8'])
@pytest.mark.parametrize("data_type", ['float16'])
@pytest.mark.parametrize("eagle_model_roots", ["llama3.1-eagle-8b-hf_v0.5"],
                         indirect=True)
def test_llm_eagle_1gpu(batch_size, data_type, eagle_model_roots,
                        eagle_example_root, llm_datasets_root, llm_rouge_root,
                        llm_venv, cmodel_dir, engine_dir):
    print("Build engines...")
    model_name = "eagle"

    # Although the datatype is float16, the actual weights are FP8.
    # The datatype in the convert stage is used for the input and output of the plugin.

    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=eagle_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=eagle_model_roots,
                                data_type=data_type)

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        f"--max_beam_width=1",
        "--use_paged_context_fmha=enable",
        f"--max_batch_size={batch_size}",
        "--speculative_decoding_mode=eagle",
        "--multiple_profiles=enable"  # also test multiple_profiles
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run run...")

    run_cmd = [
        f"{eagle_example_root}/../run.py", f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={eagle_model_roots}",
        "--eagle_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]",
        "--max_output_len=100"
    ]

    venv_check_call(llm_venv, run_cmd)
