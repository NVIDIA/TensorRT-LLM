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
"""Module test_chatglm test chatglm examples."""
import os
import shutil

import pytest
from defs.common import (convert_weights, generate_summary_cmd, similar,
                         venv_check_call, venv_check_output,
                         venv_mpi_check_call)
from defs.conftest import skip_fp8_pre_ada, skip_post_blackwell
from defs.trt_test_alternative import check_call, exists


@pytest.mark.skip_less_device_memory(24000)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("use_weight_only", [True, False],
                         ids=["enable_weight_only", "disable_weight_only"])
@pytest.mark.parametrize(
    "remove_input_padding", [True, False],
    ids=["enable_remove_input_padding", "disable_remove_input_padding"])
@pytest.mark.parametrize(
    "paged_kv_cache", [True, False],
    ids=["enable_paged_kv_cache", "disable_paged_kv_cache"])
def test_llm_chatglm_6b_single_gpu_summary(
        chatglm_6b_example_root, llm_chatglm_6b_model_root, llm_datasets_root,
        llm_rouge_root, llm_venv, cmodel_dir, engine_dir, use_weight_only,
        remove_input_padding, paged_kv_cache, num_beams):
    "Build & run chatglm-6b on single gpu."
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_chatglm_6b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=chatglm_6b_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_chatglm_6b_model_root,
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
        f"--max_beam_width={num_beams}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
    ]

    if remove_input_padding:
        build_cmd.append("--remove_input_padding=enable")
    else:
        build_cmd.append("--remove_input_padding=disable")
    if paged_kv_cache:
        build_cmd.append("--paged_kv_cache=enable")
    else:
        build_cmd.append("--paged_kv_cache=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")

    # fix HF error for chatglm-6b, hope to remove this in the future
    model_temp_dir = chatglm_6b_example_root + "/chatglm-6b/model_temp_dir"
    if not exists(model_temp_dir):
        shutil.copytree(llm_chatglm_6b_model_root, model_temp_dir)
        shutil.copy(
            chatglm_6b_example_root + "/chatglm-6b/tokenization_chatglm.py",
            model_temp_dir)

    summary_cmd = [
        f"{chatglm_6b_example_root}/../summarize.py",
        "--test_trt_llm",
        "--hf_model_dir",
        f"{model_temp_dir}",
        "--data_type",
        "fp16",
        "--check_accuracy",
        f"--engine_dir={engine_dir}",
        "--tensorrt_llm_rouge1_threshold=11.7",
        "--max_ite=40",
        f"--num_beams={num_beams}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
    ]

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device_memory(24000)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "context_fmha_type",
    ['enable_fmha', 'disable_fmha', 'enable_fmha_fp32_acc'])
@pytest.mark.parametrize("use_weight_only", [True, False],
                         ids=["enable_weight_only", "disable_weight_only"])
@pytest.mark.parametrize(
    "remove_input_padding", [True, False],
    ids=["enable_remove_input_padding", "disable_remove_input_padding"])
@pytest.mark.parametrize(
    "paged_kv_cache", [True, False],
    ids=["enable_paged_kv_cache", "disable_paged_kv_cache"])
@pytest.mark.parametrize("llm_chatglm2_6b_model_root",
                         ["chatglm2-6b", "chatglm2-6b-32k"],
                         indirect=True)
def test_llm_chatglm2_6b_single_gpu_summary(
        chatglm2_6b_example_root, llm_chatglm2_6b_model_root, llm_datasets_root,
        llm_rouge_root, llm_venv, cmodel_dir, engine_dir, context_fmha_type,
        use_weight_only, remove_input_padding, paged_kv_cache, num_beams):
    "Build & run chatglm2-6b on single gpu."
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_chatglm2_6b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=chatglm2_6b_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_chatglm2_6b_model_root,
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
        f"--max_beam_width={num_beams}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
    ]
    if context_fmha_type == "enable_fmha":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == "disable_fmha":
        build_cmd.append("--context_fmha=disable")
    if remove_input_padding:
        build_cmd.append("--remove_input_padding=enable")
    else:
        build_cmd.append("--remove_input_padding=disable")
    if paged_kv_cache:
        build_cmd.append("--paged_kv_cache=enable")
    else:
        build_cmd.append("--paged_kv_cache=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    # fix HF error for chatglm2-6b, hope to remove this in the future
    # https://nvbugspro.nvidia.com/bug/5063476
    model_temp_dir = chatglm2_6b_example_root + "/chatglm2-6b/model_temp_dir"
    if not exists(model_temp_dir):
        shutil.copytree(llm_chatglm2_6b_model_root, model_temp_dir)
        shutil.copy(
            chatglm2_6b_example_root + "/chatglm2-6b/tokenization_chatglm.py",
            model_temp_dir)

    print("Run summarize...")
    threshold_score = 16
    if "32k" in model_name:
        threshold_score = 20
    summary_cmd = [
        f"{chatglm2_6b_example_root}/../summarize.py", "--test_trt_llm",
        f"--engine_dir={engine_dir}", "--hf_model_dir", f"{model_temp_dir}",
        "--data_type=fp16", "--check_accuracy",
        f"--tensorrt_llm_rouge1_threshold={threshold_score}",
        f"--num_beams={num_beams}", f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}"
    ]
    if context_fmha_type == "enable_fmha_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device_memory(24000)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "context_fmha_type",
    ['enable_fmha', 'disable_fmha', 'enable_fmha_fp32_acc'])
@pytest.mark.parametrize("use_weight_only", [True, False],
                         ids=["enable_weight_only", "disable_weight_only"])
@pytest.mark.parametrize(
    "remove_input_padding", [True, False],
    ids=["enable_remove_input_padding", "disable_remove_input_padding"])
@pytest.mark.parametrize(
    "paged_kv_cache", [True, False],
    ids=["enable_paged_kv_cache", "disable_paged_kv_cache"])
@pytest.mark.parametrize("llm_chatglm3_6b_model_root",
                         ["chatglm3-6b", "chatglm3-6b-32k", "chatglm3-6b-base"],
                         indirect=True)
def test_llm_chatglm3_6b_single_gpu_summary(
        chatglm3_6b_example_root, llm_chatglm3_6b_model_root, llm_datasets_root,
        llm_rouge_root, llm_venv, cmodel_dir, engine_dir, context_fmha_type,
        use_weight_only, remove_input_padding, paged_kv_cache, num_beams):
    "Build & run chatglm3-6b on single gpu."
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_chatglm3_6b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=chatglm3_6b_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_chatglm3_6b_model_root,
                               data_type=dtype,
                               use_weight_only=use_weight_only)

    print("Building engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}", f"--max_batch_size={8}",
        f"--max_input_len={924}", f"--max_seq_len={1024}",
        f"--max_beam_width={num_beams}", f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}"
    ]
    if context_fmha_type == "enable_fmha":
        build_cmd.append("--context_fmha=enable")
    elif context_fmha_type == "disable_fmha":
        build_cmd.append("--context_fmha=disable")
    if remove_input_padding:
        build_cmd.append("--remove_input_padding=enable")
    else:
        build_cmd.append("--remove_input_padding=disable")
    if paged_kv_cache:
        build_cmd.append("--paged_kv_cache=enable")
    else:
        build_cmd.append("--paged_kv_cache=disable")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    # fix HF error for chatglm3-6b-32k, hope to remove this in the future
    # https://nvbugspro.nvidia.com/bug/5116684
    if model_name == "chatglm3-6b-32k":
        model_temp_dir = chatglm3_6b_example_root + f"/{model_name}/model_temp_dir"
        if not exists(model_temp_dir):
            shutil.copytree(llm_chatglm3_6b_model_root, model_temp_dir)
            shutil.copy(
                f"{chatglm3_6b_example_root}/{model_name}/tokenization_chatglm.py",
                model_temp_dir)
    else:
        model_temp_dir = llm_chatglm3_6b_model_root

    print("Run summarize...")
    summary_cmd = [
        f"{chatglm3_6b_example_root}/../summarize.py", "--test_trt_llm",
        f"--engine_dir={engine_dir}", "--hf_model_dir", f"{model_temp_dir}",
        "--data_type=fp16", "--check_accuracy",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    if context_fmha_type == "enable_fmha_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_check_call(llm_venv, summary_cmd)


# TODO: add more test case for input_padding, paged_kv_cache, num_beams
@pytest.mark.skip_less_device_memory(24000)
@pytest.mark.parametrize("use_weight_only", [True, False],
                         ids=["enable_weight_only", "disable_weight_only"])
@pytest.mark.parametrize("llm_glm_4_9b_model_root",
                         ["glm-4-9b", "glm-4-9b-chat"],
                         indirect=True)
def test_llm_glm_4_9b_single_gpu_summary(glm_4_9b_example_root,
                                         llm_glm_4_9b_model_root,
                                         llm_datasets_root, llm_rouge_root,
                                         llm_venv, cmodel_dir, engine_dir,
                                         use_weight_only):
    "Build & run glm-4-9b on single gpu."
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_glm_4_9b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=glm_4_9b_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_glm_4_9b_model_root,
                               data_type=dtype,
                               use_weight_only=use_weight_only)

    print("Building engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}", f"--max_batch_size={8}",
        f"--max_input_len={924}", f"--max_seq_len={1024}",
        f"--gpt_attention_plugin={dtype}"
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")

    # fix HF error in glm-4-9b, hope to remove this in the future
    # https://nvbugspro.nvidia.com/bug/5025895
    model_temp_dir = glm_4_9b_example_root + "/glm-4-9b/model_temp_dir"
    if not exists(model_temp_dir):
        shutil.copytree(llm_glm_4_9b_model_root, model_temp_dir)
        shutil.copy(glm_4_9b_example_root + "/glm-4-9b/tokenization_chatglm.py",
                    model_temp_dir)

    summary_cmd = [
        f"{glm_4_9b_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{model_temp_dir}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("llm_chatglm3_6b_model_root", ["chatglm3-6b-32k"],
                         indirect=True)
@pytest.mark.parametrize("test_case", ["long_input_1", "long_input_2"],
                         indirect=True)
def test_llm_chatglm3_6b_long_sq(chatglm3_6b_example_root,
                                 llm_chatglm3_6b_model_root, llm_datasets_root,
                                 llm_rouge_root, llm_venv, cmodel_dir,
                                 engine_dir, num_beams, test_case):
    """
        Build & run chatglm3-6b on single gpu with long input sequence.
        RCCA https://nvbugs/4380455
        max_input_len=32640,
        max_output_len=32k - max_input_len,
    """
    max_input_len = test_case['max_input_len']
    max_output_len = test_case['max_output_len']
    dtype = 'float16'
    model_name = os.path.basename(llm_chatglm3_6b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=chatglm3_6b_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_chatglm3_6b_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={max_input_len}",
        f"--max_seq_len={max_output_len+max_input_len}",
        f"--max_beam_width={num_beams}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        "--context_fmha=enable",
        "--remove_input_padding=enable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = generate_summary_cmd(chatglm3_6b_example_root,
                                       hf_model_dir=llm_chatglm3_6b_model_root,
                                       max_input_length=max_input_len,
                                       output_len=max_output_len,
                                       data_type="fp16",
                                       engine_dir=engine_dir,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)

    run_cmd = [
        f"{chatglm3_6b_example_root}/../run.py",
        f"--input_file={test_case['input_file']}",
        f"--max_input_len={max_input_len}",
        f"--max_output_len={max_output_len}",
        f"--tokenizer_dir={llm_chatglm3_6b_model_root}",
        f"--engine_dir={engine_dir}"
    ]

    expect_result = test_case['expect_output']
    result = venv_check_output(llm_venv, run_cmd)

    index = result.find("Output [Text 0 Beam 0]:")
    output = result[index + len("Output [Text 0 Beam 0]:"):-1]
    output = ' '.join(output.split())
    assert any([similar(output, expect)
                for expect in expect_result]), f"output is: {output}"


@pytest.mark.skip_less_device_memory(24000)
@pytest.mark.parametrize("use_weight_only", [True, False],
                         ids=["enable_weight_only", "disable_weight_only"])
def test_llm_glm_10b_single_gpu_run(glm_10b_example_root,
                                    llm_glm_10b_model_root, llm_venv,
                                    cmodel_dir, engine_dir, use_weight_only):
    "Build & run glm-10b on single gpu."
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_glm_10b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=glm_10b_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_glm_10b_model_root,
                               data_type=dtype,
                               use_weight_only=use_weight_only)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={256}",
        f"--max_seq_len={512}",
        f"--max_beam_width={1}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        "--paged_kv_cache=disable",
        "--remove_input_padding=disable",
        "--context_fmha=disable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    input_text = [
        "GLM is a General Language Model pretrained with an[MASK]objective and can be finetuned on various natural language understanding and generation tasks.",
        "The NVIDIA® GeForce RTX™ 4090 is the ultimate[MASK]. It brings an enormous leap in performance, efficiency, and AI-powered graphics. Experience ultra-high performance gaming, incredibly detailed virtual worlds, unprecedented productivity, and new ways to create. It’s powered by the NVIDIA Ada Lovelace architecture and comes with 24 GB of G6X memory to deliver the ultimate experience for gamers and creators."
    ]
    run_cmd = [
        f"{glm_10b_example_root}/../run.py", "--max_output_len=128",
        "--use_py_session", f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={llm_glm_10b_model_root}", "--no_add_special_tokens",
        f"--input_text", input_text[0], input_text[1]
    ]

    output = venv_check_output(llm_venv, run_cmd)
    assert (output.find('" end-to-end multi-"') != -1)
    assert (output.find(
        '" graphics card for the most demanding games, virtual reality, and professional applications"'
    ) != -1)

    print("Building engines with remove input padding...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={256}",
        f"--max_seq_len={512}",
        f"--max_beam_width={1}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        "--paged_kv_cache=disable",
        "--context_fmha=disable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")
    output = venv_check_output(llm_venv, run_cmd)
    assert (output.find('" end-to-end multi-"') != -1)
    assert (output.find(
        '" graphics card for the most demanding games, virtual reality, and professional applications"'
    ) != -1)

    print("Running inference with inflight batching...")
    run_cmd = [
        f"{glm_10b_example_root}/../run.py", "--max_output_len=128",
        f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={llm_glm_10b_model_root}", "--no_add_special_tokens",
        f"--input_text", input_text[0], input_text[1]
    ]
    output = venv_check_output(llm_venv, run_cmd)
    assert (output.find('" end-to-end multi-"') != -1)
    assert (output.find(
        '" graphics card for the most demanding games, virtual reality, and professional applications"'
    ) != -1)


@pytest.mark.skip_less_device_memory(24000)
def test_llm_chatglm_6b_single_gpu_run(chatglm_6b_example_root,
                                       llm_chatglm_6b_model_root, llm_venv,
                                       cmodel_dir, engine_dir):
    "Build & run chatglm_6b on single gpu."
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(llm_chatglm_6b_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=chatglm_6b_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_chatglm_6b_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={256}",
        f"--max_seq_len={512}",
        f"--max_beam_width={1}",
        f"--gpt_attention_plugin={dtype}",
        "--context_fmha=disable",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    # fix remained error in chatglm_6b, hope to remove this in the future
    model_temp_dir = chatglm_6b_example_root + "/chatglm-6b/model_temp_dir"
    if not exists(model_temp_dir):
        shutil.copytree(llm_chatglm_6b_model_root, model_temp_dir)
        shutil.copy(
            chatglm_6b_example_root + "/chatglm-6b/tokenization_chatglm.py",
            model_temp_dir)

    print("Running inference...")
    input_text = [
        "What is nvidia?",
        "What is the difference between chatglm and glm?",
    ]
    run_cmd = [
        f"{chatglm_6b_example_root}/../run.py", "--max_output_len=128",
        f"--engine_dir={engine_dir}", f"--tokenizer_dir={model_temp_dir}",
        f"--input_text", input_text[0], input_text[1]
    ]

    output = venv_check_output(llm_venv, run_cmd)
    assert (output.find(
        'The company was founded in 1994 and is headquartered in California, USA'
    ) != -1)
    assert (output.find(
        '`glm` is a library for building and using GLM models, while `chatglm` is a tool for building and testing GLM models using the `glm` library'
    ) != -1)


@pytest.mark.skip_less_device_memory(24000)
@pytest.mark.parametrize("llm_chatglm2_6b_model_root",
                         ["chatglm2-6b", "chatglm2-6b-32k"],
                         indirect=True)
def test_llm_chatglm2_6b_smoothquant_summary(chatglm2_6b_example_root,
                                             llm_chatglm2_6b_model_root,
                                             llm_datasets_root, llm_rouge_root,
                                             llm_venv, engine_dir, cmodel_dir):
    "Build & run chatglm2-6b with smoothquant on single gpu."
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(
        llm_chatglm2_6b_model_root) + "-smoothquant-0.5"
    ckpt_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=chatglm2_6b_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llm_chatglm2_6b_model_root,
        data_type=dtype,
        smoothquant=0.5,
        per_channel=True,
        per_token=True,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail")

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={1}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = [
        f"{chatglm2_6b_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_chatglm2_6b_model_root}", "--data_type",
        "fp16", "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device_memory(24000)
@pytest.mark.parametrize("llm_chatglm3_6b_model_root",
                         ["chatglm3-6b", "chatglm3-6b-base", "chatglm3-6b-32k"],
                         indirect=True)
def test_llm_chatglm3_6b_smoothquant_summary(chatglm3_6b_example_root,
                                             llm_chatglm3_6b_model_root,
                                             llm_datasets_root, llm_rouge_root,
                                             llm_venv, engine_dir, cmodel_dir):
    "Build & run chatglm3-6b with smoothquant on single gpu."
    print("Converting checkpoint...")
    dtype = 'float16'
    model_name = os.path.basename(
        llm_chatglm3_6b_model_root) + "-smoothquant-0.5"
    ckpt_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=chatglm3_6b_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llm_chatglm3_6b_model_root,
        data_type=dtype,
        smoothquant=0.5,
        per_channel=True,
        per_token=True,
        calib_dataset=f"{llm_datasets_root}/cnn_dailymail")

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={1}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = [
        f"{chatglm3_6b_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_chatglm3_6b_model_root}", "--data_type",
        "fp16", "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("debug_mode", [True, False],
                         ids=["enable_debug", "disable_debug"])
@pytest.mark.parametrize("llm_chatglm3_6b_model_root", ["chatglm3-6b"],
                         indirect=True)
def test_llm_chatglm3_6b_2gpus_summary(chatglm3_6b_example_root,
                                       llm_chatglm3_6b_model_root,
                                       llm_datasets_root, llm_rouge_root,
                                       llm_venv, cmodel_dir, engine_dir,
                                       debug_mode):
    "Build & run chatglm3-6b with smoothquant on 2 gpus."
    dtype = 'float16'
    print("Converting checkpoint...")
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=chatglm3_6b_example_root,
                               cmodel_dir=cmodel_dir,
                               model="chatglm3-6b",
                               model_path=llm_chatglm3_6b_model_root,
                               data_type=dtype,
                               tp_size=2,
                               gpus=2)

    print("Building engines...")
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

    input_text = "What's new between ChatGLM3-6B and ChatGLM2-6B?"

    run_cmd = [
        f"{chatglm3_6b_example_root}/../run.py",
        f"--input_text={input_text}",
        f"--max_output_len={50}",
        f"--tokenizer_dir={llm_chatglm3_6b_model_root}",
        f"--engine_dir={engine_dir}",
    ]

    if debug_mode:
        build_cmd.append("--enable_debug_output")
        run_cmd.append("--debug_mode")

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        run_cmd)

    summary_cmd = generate_summary_cmd(chatglm3_6b_example_root,
                                       hf_model_dir=llm_chatglm3_6b_model_root,
                                       data_type="fp16",
                                       engine_dir=engine_dir,
                                       debug_mode=debug_mode,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.parametrize(
    "qformat",
    ["fp8", pytest.param("int4_awq", marks=skip_post_blackwell())])
@pytest.mark.parametrize("llm_chatglm3_6b_model_root", ["chatglm3-6b"],
                         indirect=True)
def test_llm_chatglm3_6b_quantization_summary(chatglm3_6b_example_root,
                                              llm_chatglm3_6b_model_root,
                                              llm_datasets_root, llm_rouge_root,
                                              llm_venv, engine_dir, qcache_dir,
                                              qformat):
    "Build & run chatglm3-6b with fp8 on 1 gpus."
    skip_fp8_pre_ada(use_fp8=qformat == "fp8")
    print("Quantize checkpoint...")
    dtype = 'float16'
    ckpt_dir = f"{qcache_dir}/quantization"
    quantize_cmd = [
        f"{chatglm3_6b_example_root}/../quantization/quantize.py",
        f"--model_dir={llm_chatglm3_6b_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={dtype}",
        f"--output_dir={ckpt_dir}",
        f"--qformat={qformat}",
    ]

    if qformat == "fp8":
        quantize_cmd.append("--kv_cache_dtype=fp8")

    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_beam_width={4}",
        f"--gemm_plugin={dtype}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    summary_cmd = generate_summary_cmd(chatglm3_6b_example_root,
                                       hf_model_dir=llm_chatglm3_6b_model_root,
                                       data_type="fp16",
                                       engine_dir=engine_dir,
                                       num_beams=4,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("llm_chatglm3_6b_model_root", ["chatglm3-6b-128k"],
                         indirect=True)
@pytest.mark.parametrize("dataset_name", ["SlimPajama-6B"])
def test_llm_chatglm3_6b_long_context_ppl(chatglm3_6b_example_root,
                                          llm_chatglm3_6b_model_root, llm_venv,
                                          engine_dir, cmodel_dir,
                                          llm_datasets_root, dataset_name):

    env_cmd = "pip install transformers==4.44.2"
    check_call(env_cmd, shell=True, env=llm_venv._new_env)
    "Build & run chatglm3-6b on long context ppl."
    dtype = 'float16'
    max_input_len = 16384
    print("Converting checkpoint...")
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=chatglm3_6b_example_root,
                               cmodel_dir=cmodel_dir,
                               model="chatglm3-6b",
                               model_path=llm_chatglm3_6b_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
        f"--max_input_len={max_input_len}",
        f"--max_seq_len={1 + max_input_len}",
        f"--gemm_plugin={dtype}",
        f"--gather_context_logits",
        f"--max_num_tokens=4096",
        f"--use_paged_context_fmha=enable",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run context ppl evaluation...")
    summary_cmd = generate_summary_cmd(
        chatglm3_6b_example_root,
        tokenizer_dir=llm_chatglm3_6b_model_root,
        data_type="fp16",
        engine_dir=engine_dir,
        dataset_dir=f"{llm_datasets_root}/{dataset_name}",
        eval_task="eval_context_ppl",
        max_input_len=max_input_len,
        batch_size=1,
        max_ite=200,
        tensorrt_llm_ppl_threshold=8,
        max_tokens_in_paged_kv_cache=int(max_input_len * 1.2),
        enable_chunked_context=True,
        min_input_length=10000)

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "1", "--allow-run-as-root"],
                        summary_cmd)
