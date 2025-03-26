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
import uuid
from pathlib import Path

import pytest
from defs.common import (generate_mmlu_cmd, generate_summary_cmd,
                         test_multi_lora_support, venv_check_call)
from defs.conftest import (evaltool_mmlu_post_process,
                           evaltool_wikilingua_post_process, get_device_memory,
                           skip_fp8_pre_ada, skip_post_blackwell,
                           skip_pre_hopper)
from defs.trt_test_alternative import check_call
from evaltool.constants import (EVALTOOL_INFERENCE_SERVER_STARTUP_SCRIPT,
                                EVALTOOL_INFERENCE_SERVER_STOP_SCRIPT,
                                EVALTOOL_MMLU_CONFIG, EVALTOOL_MMLU_RESULT_FILE,
                                EVALTOOL_WIKILINGUA_CONFIG,
                                EVALTOOL_WIKILINGUA_RESULT_FILE)


def get_vocab_file(model_path):
    "get vocab file"
    if "keras" in model_path or "ax" in model_path:
        if "2b" in model_path:
            vocab_file = f"{model_path}/../gemma-2b-it-flax/tokenizer.model"
        elif "7b" in model_path:
            vocab_file = f"{model_path}/../gemma-7b-it-flax/tokenizer.model"
    else:
        vocab_file = f"{model_path}/tokenizer.model"

    return vocab_file


def get_ckpt_dir(model_path):
    "get ckpt dir"
    if "ax" in model_path:
        if "2b" in model_path:
            ckpt_dir = f"{model_path}/2b-it"
        elif "7b" in model_path:
            ckpt_dir = f"{model_path}/7b-it"
    else:
        ckpt_dir = model_path

    return ckpt_dir


def get_ckpt_type(model_path):
    "get ckpt type"
    if "torch" in model_path:
        ckpt_type = "torch"
    elif "keras" in model_path:
        ckpt_type = "keras"
    elif "ax" in model_path:
        ckpt_type = "jax"
    else:
        ckpt_type = "hf"

    return ckpt_type


@skip_post_blackwell
@skip_pre_hopper
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("data_type", ['bfloat16', 'float16'])
@pytest.mark.parametrize("qformat", ['fp8', 'int4_awq', 'int8_sq'])
@pytest.mark.parametrize(
    "gemma_model_root",
    ["gemma-2b", "gemma-7b", "gemma-2-9b-it", "gemma-2-27b-it"],
    indirect=True)
def test_llm_hf_gemma_quantization_1gpu(batch_size, data_type, gemma_model_root,
                                        llm_venv, cmodel_dir, engine_dir,
                                        gemma_example_root, llm_datasets_root,
                                        llm_rouge_root, qformat):
    "run gemma quantization tests"
    print("Convert checkpoint by modelopt...")
    kv_cache_dtype = 'fp8' if qformat == 'fp8' else 'int8'
    convert_cmd = [
        f"{gemma_example_root}/../quantization/quantize.py",
        f"--model_dir={gemma_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={data_type}",
        f"--qformat={qformat}",
        f"--kv_cache_dtype={kv_cache_dtype}",
        f"--output_dir={cmodel_dir}",
        "--device_map=sequential",
    ]
    venv_check_call(llm_venv, convert_cmd)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={cmodel_dir}",
        f"--output_dir={engine_dir}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--max_beam_width=1",
        "--max_input_len=3000",
        "--max_seq_len=3100",
        f"--max_batch_size={batch_size}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    # Currently, gemma-7b has poor performance on FP8.
    # We should use mmlu to verify in future.
    threshold_score = 19.5
    if "gemma-7b" in gemma_model_root:
        threshold_score = 18

    summary_cmd = [
        f"{gemma_example_root}/../summarize.py",
        "--test_trt_llm",
        f"--hf_model_dir={gemma_model_root}",
        f"--tokenizer_dir={gemma_model_root}",
        f"--engine_dir={engine_dir}",
        "--check_accuracy",
        f"--tensorrt_llm_rouge1_threshold={threshold_score}",
        "--max_ite=40",
        f"--batch_size={batch_size}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}",
    ]
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("data_type", ['float16', 'bfloat16'])
@pytest.mark.parametrize("test_case", [
    'other',
    pytest.param('fp8_kv_cache', marks=skip_post_blackwell),
    pytest.param('smooth_quant', marks=skip_post_blackwell),
    pytest.param('wo_int8', marks=skip_post_blackwell),
    pytest.param('wo_int4', marks=skip_post_blackwell),
    pytest.param('int8_kv_cache', marks=skip_post_blackwell)
])
@pytest.mark.parametrize("gemma_model_root", [
    "gemma-2b", "gemma-7b", "gemma-2b-torch", "gemma-7b-torch",
    "gemma-2b-keras", "gemma-7b-keras", "gemma-2b-it-flax", "gemma-7b-it-flax",
    "gemma-2-9b-it", "gemma-2-27b-it"
],
                         indirect=True)
def test_llm_gemma_1gpu_summary(batch_size, data_type, gemma_model_root,
                                llm_venv, cmodel_dir, engine_dir,
                                gemma_example_root, llm_datasets_root,
                                llm_rouge_root, test_case):
    "run gemm test on 1 gpu"
    skip_fp8_pre_ada(use_fp8=test_case == "fp8_kv_cache")
    if "smooth_quant" in test_case and "bfloat16" in data_type:
        pytest.skip("TensorRT-LLM does not support SmoothQuant with bfloat16.")

    if any(params in gemma_model_root for params in
           ["gemma-7b", "9b", "27b"]) and get_device_memory() < 50000:
        pytest.skip(f"Insufficient device memory for {gemma_model_root}.")

    ckpt_type = get_ckpt_type(gemma_model_root)
    ckpt_dir = get_ckpt_dir(gemma_model_root)
    vocab_file = get_vocab_file(gemma_model_root)

    print("Convert checkpoint ...")
    convert_cmd = [
        f"{gemma_example_root}/convert_checkpoint.py",
        f"--ckpt-type={ckpt_type}",
        f"--model-dir={ckpt_dir}",
        f"--dtype={data_type}",
        f"--output-model-dir={cmodel_dir}",
    ]

    if "fp8_kv" in test_case:
        convert_cmd.extend(["--enable_fp8", "--fp8_kv_cache"])
    elif "smooth" in test_case:
        convert_cmd.append("--use_smooth_quant_plugin=0.5")
        convert_cmd.append(f"--tokenizer_dir={vocab_file}")
        convert_cmd.append(
            f"--calib_dataset={llm_datasets_root}/ccdv/cnn_dailymail")
    elif "int8_kv" in test_case:
        convert_cmd.append("--calibrate_kv_cache")
        convert_cmd.append(f"--tokenizer_dir={vocab_file}")
        convert_cmd.append(
            f"--calib_dataset={llm_datasets_root}/ccdv/cnn_dailymail")
    elif 'wo_int4' in test_case:
        if ckpt_type != "jax":
            pytest.skip("Only verify int4_wo on jax checkpoint.")
        convert_cmd.append("--use-weight-only-with-precision=int4")
    elif 'wo_int8' in test_case:
        convert_cmd.append("--use-weight-only-with-precision=int8")

    venv_check_call(llm_venv, convert_cmd)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={cmodel_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={batch_size}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--max_beam_width=1",
        "--max_input_len=3000",
        "--max_seq_len=3100",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summarize...")
    summary_cmd = generate_summary_cmd(gemma_example_root,
                                       engine_dir=engine_dir,
                                       max_ite=40,
                                       batch_size=batch_size,
                                       tensorrt_llm_rouge1_threshold=15,
                                       dataset_dir=llm_datasets_root,
                                       rouge_dir=llm_rouge_root)

    if ckpt_type == "hf":
        summary_cmd.extend([
            f"--hf_model_dir={gemma_model_root}",
            f"--tokenizer_dir={gemma_model_root}"
        ])
    else:
        summary_cmd.append(f"--vocab_file={vocab_file}")

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("data_type", ['float16', 'bfloat16'])
@pytest.mark.parametrize("test_case", [
    'other', 'fp8_kv_cache', 'smooth_quant', 'wo_int8', 'wo_int4',
    'int8_kv_cache'
])
@pytest.mark.parametrize("gemma_model_root", [
    "gemma-2b", "gemma-7b", "gemma-2b-torch", "gemma-7b-torch",
    "gemma-2b-keras", "gemma-7b-keras", "gemma-2b-it-flax", "gemma-7b-it-flax"
],
                         indirect=True)
def test_llm_gemma_1gpu_mmlu(batch_size, data_type, gemma_model_root, llm_venv,
                             cmodel_dir, engine_dir, gemma_example_root,
                             llm_rouge_root, llm_datasets_root, test_case):
    "run gemm test on 1 gpu"
    if "smooth_quant" in test_case and "bfloat16" in data_type:
        pytest.skip("TensorRT-LLM does not support SmoothQuant with bfloat16.")
    ckpt_type = get_ckpt_type(gemma_model_root)
    ckpt_dir = get_ckpt_dir(gemma_model_root)
    vocab_file = get_vocab_file(gemma_model_root)

    print("Download checkpoint")
    data_path = Path(engine_dir) / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    print("Convert checkpoint ...")
    convert_cmd = [
        f"{gemma_example_root}/convert_checkpoint.py",
        f"--ckpt-type={ckpt_type}",
        f"--model-dir={ckpt_dir}",
        f"--dtype={data_type}",
        f"--output-model-dir={cmodel_dir}",
    ]

    if "fp8_kv" in test_case:
        convert_cmd.extend(["--enable_fp8", "--fp8_kv_cache"])
    elif "smooth" in test_case:
        convert_cmd.append("--use_smooth_quant_plugin=0.5")
        convert_cmd.append(f"--tokenizer_dir={vocab_file}")
        convert_cmd.append(
            f"--calib_dataset={llm_datasets_root}/ccdv/cnn_dailymail")
    elif "int8_kv" in test_case:
        convert_cmd.append("--calibrate_kv_cache")
        convert_cmd.append(f"--tokenizer_dir={vocab_file}")
        convert_cmd.append(
            f"--calib_dataset={llm_datasets_root}/ccdv/cnn_dailymail")
    elif 'wo_int4' in test_case:
        if ckpt_type != "jax":
            pytest.skip("Only verify int4_wo on jax checkpoint.")
        convert_cmd.append("--use-weight-only-with-precision=int4")
    elif 'wo_int8' in test_case:
        convert_cmd.append("--use-weight-only-with-precision=int8")

    venv_check_call(llm_venv, convert_cmd)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={cmodel_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={batch_size}",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--max_beam_width=1",
        "--max_input_len=3000",
        "--max_seq_len=3100",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run mmlu...")
    summary_cmd = generate_mmlu_cmd(gemma_example_root,
                                    engine_dir=engine_dir,
                                    accuracy_threshold=37,
                                    data_dir=f"{llm_datasets_root}/mmlu")

    if ckpt_type == "hf":
        summary_cmd.extend([
            f"--hf_model_dir={gemma_model_root}",
            f"--tokenizer_dir={gemma_model_root}"
        ])
    else:
        summary_cmd.append(f"--vocab_file={vocab_file}")

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.parametrize("gemma_model_root", ["gemma-2b", "gemma-7b"],
                         indirect=True)
def test_llm_gemma_1gpu_evaltool(gemma_model_root, llm_venv, cmodel_dir,
                                 engine_dir, gemma_example_root, evaltool_root):
    ckpt_type = get_ckpt_type(gemma_model_root)
    ckpt_dir = get_ckpt_dir(gemma_model_root)
    assert ckpt_type == 'hf'

    print("Convert checkpoint ...")
    data_type = "float16"
    convert_cmd = [
        f"{gemma_example_root}/convert_checkpoint.py",
        f"--ckpt-type={ckpt_type}",
        f"--model-dir={ckpt_dir}",
        f"--dtype={data_type}",
        f"--output-model-dir={cmodel_dir}",
    ]
    venv_check_call(llm_venv, convert_cmd)

    print("Build engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={cmodel_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size=8",
        f"--gpt_attention_plugin={data_type}",
        f"--gemm_plugin={data_type}",
        "--gather_context_logits",
        "--max_input_len=8000",
        "--max_seq_len=7048",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Lm evaluation harness")
    # start inference server
    start_inference_server = [
        EVALTOOL_INFERENCE_SERVER_STARTUP_SCRIPT, "-e", engine_dir, "-t",
        gemma_model_root, "-d", evaltool_root, "-m", "1024"
    ]
    check_call(" ".join(start_inference_server), shell=True)

    task_list = ['mmlu', 'wikilingua']
    try:
        for task in task_list:
            project_id = str(uuid.uuid4())
            if task == "wikilingua":
                config_file = EVALTOOL_WIKILINGUA_CONFIG
                result_file = EVALTOOL_WIKILINGUA_RESULT_FILE

            if task == "mmlu":
                config_file = EVALTOOL_MMLU_CONFIG
                result_file = EVALTOOL_MMLU_RESULT_FILE

            # Update config dynamically
            import yaml

            model_name = os.path.basename(gemma_model_root)
            with open(config_file, 'r') as f:
                lm_eval_config = yaml.safe_load(f)
                lm_eval_config['model']['llm_name'] = model_name
                lm_eval_config['model']['tokenizer_path'] = gemma_model_root

            config_file = os.path.join(llm_venv.get_working_directory(),
                                       "lm_eval_config.yaml")
            with open(config_file, 'w') as f:
                yaml.dump(lm_eval_config, f)

            # launch evaluation
            run_cmd = [
                f"cd {evaltool_root}",
                "&&",
                "source .venv/bin/activate",
                "&&",
                "python3",
                f"evaltool/interfaces/cli/main.py",
                "project",
                "launch",
                f"--eval_project_config_file '{config_file}'",
                "--infra_name local",
                f"--output_dir '{llm_venv.get_working_directory()}'",
                f"--project_id {project_id}",
            ]
            check_call(" ".join(run_cmd), shell=True, executable="/bin/bash")

            # process result
            result_path = f"{llm_venv.get_working_directory()}/{project_id}/{result_file}"
            check_call(f"cat {result_path}", shell=True)

            if task == 'mmlu':
                # Gemma-7b produce 0 accuracy even for HF model.
                if '7b' in gemma_model_root:
                    evaltool_mmlu_post_process(result_path, 0.0, 100)
                elif '2b' in gemma_model_root:
                    # Gemma-2b HF result 0.3837 and TRTLLM 0.3826.
                    # evaltool_mmlu_post_process(result_path, 0.4230, 0.006)
                    evaltool_mmlu_post_process(result_path, 0.3826, 0.006)
            if task == 'wikilingua':
                if '7b' in gemma_model_root:
                    evaltool_wikilingua_post_process(result_path, 0.0, 100)
                elif '2b' in gemma_model_root:
                    evaltool_wikilingua_post_process(result_path, 0.1620, 0.003)
    finally:
        # stop the server
        check_call(f"{EVALTOOL_INFERENCE_SERVER_STOP_SCRIPT}", shell=True)


@skip_pre_hopper
@pytest.mark.parametrize(
    "gemma_model_root",
    ["gemma-2b", "gemma-7b", "gemma-2-9b-it", "gemma-2-27b-it"],
    indirect=True)
def test_hf_gemma_fp8_base_bf16_multi_lora(gemma_model_root,
                                           llm_venv,
                                           cmodel_dir,
                                           engine_dir,
                                           gemma_example_root,
                                           llm_datasets_root,
                                           data_type='bfloat16',
                                           qformat='fp8',
                                           batch_size=8):
    "Run Gemma models with multiple dummy LoRAs."

    print("Convert checkpoint by modelopt...")
    kv_cache_dtype = 'fp8' if qformat == 'fp8' else 'int8'
    convert_cmd = [
        f"{gemma_example_root}/../quantization/quantize.py",
        f"--model_dir={gemma_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        f"--dtype={data_type}",
        f"--qformat={qformat}",
        f"--kv_cache_dtype={kv_cache_dtype}",
        f"--output_dir={cmodel_dir}",
    ]
    venv_check_call(llm_venv, convert_cmd)

    test_multi_lora_support(
        hf_model_dir=gemma_model_root,
        tllm_ckpt_dir=cmodel_dir,
        engine_dir=engine_dir,
        llm_venv=llm_venv,
        example_root=gemma_example_root,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        target_trtllm_modules=["attn_q", "attn_k", "attn_v"],
        zero_lora_weights=True,
    )
