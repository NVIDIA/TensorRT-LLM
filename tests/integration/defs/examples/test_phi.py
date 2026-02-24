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
import csv
import os

import defs.ci_profiler
import pytest
from defs.common import (convert_weights, quantize_data,
                         test_llm_torch_multi_lora_support,
                         test_multi_lora_support, venv_check_call,
                         venv_mpi_check_call)
from defs.conftest import (get_device_memory, get_sm_version, skip_fp8_pre_ada,
                           skip_post_blackwell, skip_pre_ada)
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@pytest.fixture(scope="module")
def phi_example_root(llm_root, llm_venv):
    "Get phi example root"
    example_root = os.path.join(llm_root, "examples", "models", "core", "phi")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@skip_post_blackwell
@pytest.mark.skip_less_device_memory(40000)
@pytest.mark.parametrize("num_beams", [1, 2, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize(
    "context_fmha_type",
    ["enable_fmha", "enable_fmha_with_fp32_acc", "disable_fmha"])
@pytest.mark.parametrize(
    "use_attention_plugin", [True, False],
    ids=["enable_attention_plugin", "disable_attention_plugin"])
@pytest.mark.parametrize("use_gemm_plugin", [True, False],
                         ids=["enable_gemm_plugin", "disable_gemm_plugin"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("llm_phi_model_root", [
    "phi-2", "Phi-3-mini-4k-instruct", "Phi-3-mini-128k-instruct",
    "Phi-3-small-8k-instruct", "Phi-3-small-128k-instruct",
    "Phi-3.5-mini-instruct"
],
                         indirect=True)
def test_llm_phi_single_gpu_summary(phi_example_root, llm_phi_model_root,
                                    llm_datasets_root, llm_rouge_root, llm_venv,
                                    cmodel_dir, engine_dir,
                                    use_attention_plugin, use_gemm_plugin,
                                    dtype, context_fmha_type, num_beams):
    "Build & run phi on single gpu."
    if (not use_attention_plugin or not use_gemm_plugin) \
        and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    if context_fmha_type != "disable_fmha":
        # --enable_context_fmha / --enable_context_fmha_fp32_acc
        # have to be used together with --use_gpt_attention_plugin
        use_attention_plugin = True

    print("Converting checkpoint...")
    model_name = os.path.basename(llm_phi_model_root)

    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=phi_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_phi_model_root,
                               data_type=dtype)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={16}",
        f"--max_input_len={1024}",
        f"--max_seq_len={2048}",
        f"--max_beam_width={num_beams}",
    ]

    if use_attention_plugin:
        build_cmd.append(f"--gpt_attention_plugin={dtype}")
        if context_fmha_type == "enable_fmha":
            build_cmd.append("--context_fmha=enable")
        elif context_fmha_type == "disable_fmha":
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

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run phi...')
    run_cmd = [
        f"{phi_example_root}/../../../run.py",
        "--max_output_len=50",
        f"--engine_dir={engine_dir}",
        f"--tokenizer_dir={llm_phi_model_root}",
    ]
    venv_check_call(llm_venv, run_cmd)

    rouge1_threshold = 20
    if model_name == 'Phi-3-small-8k-instruct': rouge1_threshold = 18.0
    if model_name == 'Phi-3-small-128k-instruct': rouge1_threshold = 19.0

    summary_cmd = [
        f"{phi_example_root}/../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_phi_model_root}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--tensorrt_llm_rouge1_threshold={rouge1_threshold}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    if context_fmha_type == "enable_fmha_with_fp32_acc":
        summary_cmd.append("--enable_context_fmha_fp32_acc")

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(40000)
@pytest.mark.parametrize("num_beams", [1, 4],
                         ids=lambda num_beams: f'nb:{num_beams}')
@pytest.mark.parametrize("llm_phi_model_root", [
    "phi-2", "Phi-3-mini-4k-instruct", "Phi-3-mini-128k-instruct",
    "Phi-3-small-8k-instruct", "Phi-3-small-128k-instruct",
    'Phi-3.5-MoE-instruct'
],
                         indirect=True)
def test_llm_phi_1node_2gpus_summary(phi_example_root, llm_phi_model_root,
                                     llm_datasets_root, llm_rouge_root,
                                     llm_venv, cmodel_dir, engine_dir,
                                     num_beams):
    "Build & run phi on 2 gpus."

    print("Converting checkpoint...")
    model_name = os.path.basename(llm_phi_model_root)
    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=phi_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=llm_phi_model_root,
                               data_type="float16",
                               tp_size=2,
                               pp_size=1)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={16}",
        f"--max_input_len={1024}",
        f"--max_seq_len={2048}",
        f"--max_beam_width={num_beams}",
        "--gemm_plugin=float16",
        "--gpt_attention_plugin=float16",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print('Run phi...')

    rouge1_threshold = 21.2
    if model_name == 'Phi-3.5-MoE-instruct': rouge1_threshold = 24.0
    summary_cmd = [
        f"{phi_example_root}/../../../summarize.py", "--test_trt_llm",
        "--hf_model_dir", f"{llm_phi_model_root}", "--data_type", "fp16",
        "--check_accuracy", f"--engine_dir={engine_dir}",
        f"--tensorrt_llm_rouge1_threshold={rouge1_threshold}",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "2", "--allow-run-as-root"],
                        summary_cmd)


@pytest.mark.parametrize("data_type", ["float16", "fp8"],
                         ids=["base_fp16", "base_fp8"])
@pytest.mark.parametrize("lora_data_type", ["float16"], ids=["lora_fp16"])
@pytest.mark.parametrize("llm_phi_model_root", ["Phi-3-mini-4k-instruct"],
                         indirect=True)
@pytest.mark.parametrize("llm_lora_model_root",
                         ["Phi-3-mini-4k-instruct-ru-lora"],
                         indirect=True)
def test_llm_phi_lora_1gpu(data_type, lora_data_type, phi_example_root,
                           llm_phi_model_root, llm_datasets_root, llm_venv,
                           cmodel_dir, engine_dir, llm_lora_model_root,
                           qcache_dir_without_install_package):
    "run phi lora test on 1gpu"
    print("Converting checkpoint...")
    model_name = 'phi-3-lora'
    if data_type == 'fp8':
        skip_fp8_pre_ada(use_fp8=True)
        if get_sm_version() >= 100:
            pytest.skip("FP8 is not supported on post-Blackwell architectures")
        model_dir = quantize_data(
            llm_venv,
            phi_example_root,
            model_dir=llm_phi_model_root,
            calib_dataset=f"{llm_datasets_root}/cnn_dailymail",
            dtype="float16",
            qformat="fp8",
            kv_cache_dtype="fp8",
            quantize_dir=qcache_dir_without_install_package,
            calib_size=512)
    else:
        model_dir = convert_weights(llm_venv=llm_venv,
                                    example_root=phi_example_root,
                                    cmodel_dir=cmodel_dir,
                                    model=model_name,
                                    model_path=llm_phi_model_root)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}",
        "--lora_plugin=auto",
        "--gemm_plugin=auto",
        "--max_batch_size=8",
        f"--lora_dir={llm_lora_model_root}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    ref_1 = [
        1, 1815, 366, 3867, 5837, 304, 17545, 18240, 310, 9892, 16397, 322,
        8338, 265, 29888, 21211, 29973, 306, 29915, 29885, 3063, 363, 907, 1230,
        322, 9045, 29891, 9522, 5547, 393, 11039, 403, 1716, 285, 21211, 29889,
        29871
    ]

    ref_2 = [
        1815, 366, 3867, 5837, 304, 17545, 18240, 310, 9892, 16397, 322, 8338,
        265, 29888, 21211, 29973, 13, 13, 7900, 22137, 29901, 315, 13946, 368,
        29991, 2266, 526, 777, 907, 1230, 5837, 304, 13389, 9892, 16397, 322
    ]

    input_text = "Can you provide ways to eat combinations of bananas and dragonfruits?"

    print(f"Run inference with lora id 0...")
    venv_check_call(llm_venv, [
        f"{phi_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        "--lora_task_uids=0",
        f"--tokenizer_dir={llm_lora_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/use_lora.csv",
        "--use_py_session",
    ])

    with open(f"{llm_venv.get_working_directory()}/use_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]
    assert ref_1 == predict or data_type != "float16"

    print(f"Run inference with lora id -1...")
    venv_check_call(llm_venv, [
        f"{phi_example_root}/../../../run.py",
        "--max_output_len=20",
        f"--input_text={input_text}",
        "--lora_task_uids=-1",
        f"--tokenizer_dir={llm_phi_model_root}",
        f"--engine_dir={engine_dir}",
        f"--output_csv={llm_venv.get_working_directory()}/no_lora.csv",
        "--use_py_session",
    ])

    with open(f"{llm_venv.get_working_directory()}/no_lora.csv") as f:
        predict = csv.reader(f)
        predict = next(predict)
    predict = [int(p) for p in predict]

    assert ref_2 == predict or data_type != "float16"


@skip_pre_ada
@pytest.mark.parametrize("data_type", ['float16', 'bfloat16'])
@pytest.mark.parametrize("qformat", ['fp8'])
@pytest.mark.parametrize("llm_phi_model_root", [
    pytest.param("phi-2", marks=skip_post_blackwell),
    pytest.param("Phi-3-mini-128k-instruct", marks=skip_post_blackwell),
    pytest.param("Phi-3-small-128k-instruct", marks=skip_post_blackwell),
    pytest.param("Phi-3.5-mini-instruct", marks=skip_post_blackwell),
    "Phi-3.5-MoE-instruct", "Phi-4-mini-instruct"
],
                         indirect=True)
def test_llm_phi_quantization_1gpu(data_type, llm_phi_model_root, llm_venv,
                                   cmodel_dir, engine_dir, phi_example_root,
                                   llm_datasets_root, llm_rouge_root, qformat):
    "Run phi quantization tests"
    # Workaround for Modelopt can't convert Phi-3 on multi GPUs.
    gpu_constraint = {"CUDA_VISIBLE_DEVICES": "0"}

    print("Convert checkpoint by modelopt...")
    convert_cmd = [
        f"{phi_example_root}/../../../quantization/quantize.py",
        f"--model_dir={llm_phi_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={data_type}",
        f"--qformat={qformat}",
        f"--kv_cache_dtype={qformat}",
        f"--output_dir={cmodel_dir}",
    ]
    venv_check_call(llm_venv, convert_cmd, env=gpu_constraint)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={cmodel_dir}",
        f"--output_dir={engine_dir}",
        "--max_input_len=3000",
        "--max_seq_len=3100",
        f"--max_batch_size={16}",
    ]

    build_env = {
        **llm_venv._new_env,
        **gpu_constraint
    } if llm_venv._new_env else gpu_constraint
    check_call(" ".join(build_cmd), shell=True, env=build_env)

    print("Run summarize...")
    threshold_score = 24.0
    model_name = os.path.basename(llm_phi_model_root)
    if model_name == "phi-2":
        threshold_score = 22.0

    summary_cmd = [
        f"{phi_example_root}/../../../summarize.py",
        "--test_trt_llm",
        f"--hf_model_dir={llm_phi_model_root}",
        f"--tokenizer_dir={llm_phi_model_root}",
        f"--engine_dir={engine_dir}",
        "--check_accuracy",
        f"--tensorrt_llm_rouge1_threshold={threshold_score}",
        "--max_ite=40",
        f"--batch_size={16}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
    ]

    venv_check_call(llm_venv, summary_cmd, env=gpu_constraint)


@skip_pre_ada
@skip_post_blackwell
@pytest.mark.parametrize("llm_phi_model_root", [
    "phi-2", "Phi-3-mini-128k-instruct", "Phi-3-small-128k-instruct",
    "Phi-3.5-mini-instruct", "Phi-3.5-MoE-instruct", "Phi-4-mini-instruct"
],
                         indirect=True)
def test_phi_fp8_with_bf16_lora(llm_phi_model_root,
                                llm_venv,
                                cmodel_dir,
                                engine_dir,
                                phi_example_root,
                                llm_datasets_root,
                                llm_rouge_root,
                                data_type='bfloat16',
                                qformat='fp8'):
    "Run Phi models with multiple pseudo LoRAs."

    model_name = os.path.basename(llm_phi_model_root)
    if model_name == "Phi-3.5-MoE-instruct" and \
        get_device_memory() < 95000:
        pytest.skip(f"This test is only supported when memory >= 95000")

    # Quantize the base model to fp8.
    print("Convert checkpoint by modelopt...")
    convert_cmd = [
        f"{phi_example_root}/../../../quantization/quantize.py",
        f"--model_dir={llm_phi_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={data_type}",
        f"--qformat={qformat}",
        f"--kv_cache_dtype={qformat}",
        f"--output_dir={cmodel_dir}",
    ]
    # Workaround for Modelopt can't convert Phi-3-small-128k-instruct on multi GPUs.
    env = None
    if model_name == "Phi-3-small-128k-instruct":
        env = {"CUDA_VISIBLE_DEVICES": "0"}
    venv_check_call(llm_venv, convert_cmd, env=env)

    print("Creating pseudo LoRAs...")
    hf_target_modules = {
        "phi-2": ["q_proj", "k_proj", "v_proj"],
        "Phi-3-mini-128k-instruct": ["qkv_proj"],
        "Phi-3-small-128k-instruct": ["query_key_value"],
        "Phi-3.5-mini-instruct": ["qkv_proj"],
        "Phi-3.5-MoE-instruct":
        ["q_proj", "k_proj", "v_proj", "w1", "w2", "w3"],
        "Phi-4-mini-instruct": ["qkv_proj"],
    }
    trtllm_target_modules = {
        "phi-2": ["attn_q", "attn_k", "attn_v"],
        "Phi-3-mini-128k-instruct": ["attn_qkv"],
        "Phi-3-small-128k-instruct": ["attn_qkv"],
        "Phi-3.5-mini-instruct": ["attn_qkv"],
        "Phi-3.5-MoE-instruct": [
            "attn_q", "attn_k", "attn_v", "moe_h_to_4h", "moe_4h_to_h",
            "moe_gate"
        ],
        "Phi-4-mini-instruct": ["attn_qkv"],
    }
    model_name = os.path.basename(llm_phi_model_root)
    test_multi_lora_support(
        hf_model_dir=llm_phi_model_root,
        tllm_ckpt_dir=cmodel_dir,
        engine_dir=engine_dir,
        llm_venv=llm_venv,
        example_root=phi_example_root,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=hf_target_modules[model_name],
        target_trtllm_modules=trtllm_target_modules[model_name],
        zero_lora_weights=True,
    )


@pytest.mark.skip(
    reason="TODO: Resolve an import issue with transformers's LossKwargs")
@skip_pre_ada
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("llm_phi_model_root", ['Phi-4-mini-instruct'],
                         indirect=True)
def test_phi_4_mini_instruct_with_bf16_lora_torch(
        phi_example_root, llm_datasets_root, qcache_dir_without_install_package,
        llm_venv, engine_dir, llm_phi_model_root):
    """Run Phi-4-mini-instruct with multiple dummy LoRAs using LLM-API Torch backend."""

    expected_outputs = {
        'Phi-4-mini-instruct': ["...", "...", "...", "...", "..."],
    }

    print("Testing with LLM-API Torch backend...")

    defs.ci_profiler.start("test_llm_torch_multi_lora_support")
    model_name = os.path.basename(llm_phi_model_root).lower()
    test_llm_torch_multi_lora_support(
        hf_model_dir=llm_phi_model_root,
        llm_venv=llm_venv,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["qkv_proj"],
        target_trtllm_modules=["attn_qkv"],
        zero_lora_weights=True,
        tensor_parallel_size=1,
        expected_outputs=expected_outputs[model_name])
    defs.ci_profiler.stop("test_llm_torch_multi_lora_support")
