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
from defs.common import (convert_weights, get_dummy_spec_decoding_heads,
                         venv_check_call)
from defs.conftest import get_sm_version, skip_fp8_pre_ada, skip_post_blackwell
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


@pytest.mark.parametrize("batch_size", [1, 8], ids=['bs1', 'bs8'])
@pytest.mark.parametrize("medusa_model_roots", ["llama3.1-medusa-8b-hf_v0.1"],
                         indirect=True)
@pytest.mark.parametrize("use_py_session", [False, True],
                         ids=["use_cpp_session", "use_py_session"])
def test_llm_medusa_fp8_modelOpt_ckpt_1gpu(batch_size, medusa_model_roots,
                                           medusa_example_root,
                                           llm_datasets_root, llm_rouge_root,
                                           llm_venv, cmodel_dir, engine_dir,
                                           use_py_session):
    skip_fp8_pre_ada(use_fp8=True)

    model_ckpt_dir = convert_weights(llm_venv=llm_venv,
                                     example_root=medusa_example_root,
                                     cmodel_dir=cmodel_dir,
                                     model="llama",
                                     model_path=medusa_model_roots[0])

    print("Build engines...")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gemm_plugin=float16",
        '--speculative_decoding_mode=medusa',
        f"--max_batch_size={batch_size}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")

    summary_cmd = [
        f"{medusa_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{medusa_model_roots[0]}", "--tokenizer_dir",
        f"{medusa_model_roots[0]}", f"--engine_dir={engine_dir}",
        "--check_accuracy", "--tensorrt_llm_rouge1_threshold=24",
        "--medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [1, 6], [0, 7, 0]]",
        f"--temperature=1.0", f"--max_ite=40", f"--batch_size={batch_size}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    if use_py_session:
        summary_cmd.append("--use_py_session")

    venv_check_call(llm_venv, summary_cmd)


def test_with_dummy_medusa(hf_model_root, medusa_example_root, llm_venv,
                           cmodel_dir, engine_dir, batch_size, data_type,
                           num_medusa_heads, use_py_session, model_type):

    # We unset WORLD_SIZE while running tests in specific cluster nodes to
    # deal with a bug in transformers library. Trainer initialization in
    # get_dummy_spec_decoding_heads() function fails if WORLD_SIZE is unset.
    # Preemptively skip tests if WORLD_SIZE is unset.
    if os.environ.get("WORLD_SIZE") is None:
        pytest.skip(
            "[test_with_dummy_medusa] Skipping test due to missing WORLD_SIZE env variable."
        )

    print("Creating dummy Medusa heads...")
    get_dummy_spec_decoding_heads(hf_model_dir=hf_model_root,
                                  save_dir=llm_venv.get_working_directory(),
                                  mode='medusa',
                                  num_heads=num_medusa_heads)

    print("Converting to TRTLLM checkpoints...")
    model_name = model_type + "_medusa"
    converted_model_path = os.path.join(cmodel_dir, model_name)
    converted_ckpt_dir = f'{converted_model_path}/{data_type}/1-gpu'
    convert_cmd = [
        f"{medusa_example_root}/convert_checkpoint.py", "--model_dir",
        os.path.join(llm_venv.get_working_directory(), 'fp8'), "--output_dir",
        converted_ckpt_dir, f"--dtype={data_type}", "--tp_size=1",
        "--pp_size=1", f"--model_type={model_type}"
    ]
    venv_check_call(llm_venv, convert_cmd)

    print("Building engine...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={converted_ckpt_dir}",
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

    print("Run run.py...")
    run_cmd = [
        f"{medusa_example_root}/../run.py",
        f"--tokenizer_dir={hf_model_root}",
        f"--engine_dir={engine_dir}",
        "--max_output_len=100",
        "--medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]",
        f"--temperature=1.0",
    ]
    if use_py_session:
        run_cmd.append("--use_py_session")

    venv_check_call(llm_venv, run_cmd)


@pytest.mark.skip(reason="https://nvbugs/5219534")
@pytest.mark.parametrize("llama_model_root",
                         ['llama-v2-7b-hf', 'llama-3.1-8b', 'llama-3.2-1b'],
                         indirect=True)
def test_llama_medusa_1gpu(llama_model_root,
                           medusa_example_root,
                           llm_datasets_root,
                           llm_rouge_root,
                           llm_venv,
                           cmodel_dir,
                           engine_dir,
                           batch_size=1,
                           data_type='bfloat16',
                           num_medusa_heads=4,
                           use_py_session=True):

    test_with_dummy_medusa(hf_model_root=llama_model_root,
                           medusa_example_root=medusa_example_root,
                           llm_venv=llm_venv,
                           cmodel_dir=cmodel_dir,
                           engine_dir=engine_dir,
                           batch_size=batch_size,
                           data_type=data_type,
                           num_medusa_heads=num_medusa_heads,
                           use_py_session=use_py_session,
                           model_type='llama')


@pytest.mark.skip(reason="https://nvbugs/5219534")
@pytest.mark.parametrize("code_llama_model_root", ['CodeLlama-7b-Instruct'],
                         indirect=True)
def test_codellama_medusa_1gpu(code_llama_model_root,
                               medusa_example_root,
                               llm_datasets_root,
                               llm_rouge_root,
                               llm_venv,
                               cmodel_dir,
                               engine_dir,
                               batch_size=1,
                               data_type='bfloat16',
                               num_medusa_heads=4,
                               use_py_session=True):

    test_with_dummy_medusa(hf_model_root=code_llama_model_root,
                           medusa_example_root=medusa_example_root,
                           llm_venv=llm_venv,
                           cmodel_dir=cmodel_dir,
                           engine_dir=engine_dir,
                           batch_size=batch_size,
                           data_type=data_type,
                           num_medusa_heads=num_medusa_heads,
                           use_py_session=use_py_session,
                           model_type='llama')


@pytest.mark.parametrize("llm_mistral_model_root", ['mistral-7b-v0.1'],
                         indirect=True)
def test_mistral_medusa_1gpu(llm_mistral_model_root,
                             medusa_example_root,
                             llm_datasets_root,
                             llm_rouge_root,
                             llm_venv,
                             cmodel_dir,
                             engine_dir,
                             batch_size=1,
                             data_type='bfloat16',
                             num_medusa_heads=4,
                             use_py_session=True):

    test_with_dummy_medusa(hf_model_root=llm_mistral_model_root,
                           medusa_example_root=medusa_example_root,
                           llm_venv=llm_venv,
                           cmodel_dir=cmodel_dir,
                           engine_dir=engine_dir,
                           batch_size=batch_size,
                           data_type=data_type,
                           num_medusa_heads=num_medusa_heads,
                           use_py_session=use_py_session,
                           model_type='mistral')


@pytest.mark.parametrize("llm_qwen_model_root", [
    "qwen_7b_chat", "qwen1.5_7b_chat", "qwen2_7b_instruct",
    "qwen2_0.5b_instruct", "qwen2.5_1.5b_instruct"
],
                         indirect=True)
def test_qwen_medusa_1gpu(llm_qwen_model_root,
                          medusa_example_root,
                          llm_datasets_root,
                          llm_rouge_root,
                          llm_venv,
                          cmodel_dir,
                          engine_dir,
                          batch_size=1,
                          data_type='bfloat16',
                          num_medusa_heads=4,
                          use_py_session=True):

    test_with_dummy_medusa(hf_model_root=llm_qwen_model_root,
                           medusa_example_root=medusa_example_root,
                           llm_venv=llm_venv,
                           cmodel_dir=cmodel_dir,
                           engine_dir=engine_dir,
                           batch_size=batch_size,
                           data_type=data_type,
                           num_medusa_heads=num_medusa_heads,
                           use_py_session=use_py_session,
                           model_type='qwen')


@pytest.mark.parametrize("llm_phi_model_root", [
    "phi-2", "Phi-3-mini-128k-instruct", "Phi-3-small-128k-instruct",
    "Phi-3.5-mini-instruct", "Phi-4-mini-instruct"
],
                         indirect=True)
def test_phi_medusa_1gpu(llm_phi_model_root,
                         medusa_example_root,
                         llm_datasets_root,
                         llm_rouge_root,
                         llm_venv,
                         cmodel_dir,
                         engine_dir,
                         batch_size=1,
                         data_type='bfloat16',
                         num_medusa_heads=4,
                         use_py_session=True):

    test_with_dummy_medusa(hf_model_root=llm_phi_model_root,
                           medusa_example_root=medusa_example_root,
                           llm_venv=llm_venv,
                           cmodel_dir=cmodel_dir,
                           engine_dir=engine_dir,
                           batch_size=batch_size,
                           data_type=data_type,
                           num_medusa_heads=num_medusa_heads,
                           use_py_session=use_py_session,
                           model_type='phi')
