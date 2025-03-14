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
from defs.common import (convert_weights, test_multi_lora_support,
                         venv_mpi_check_call)
from defs.trt_test_alternative import check_call


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "llm_granite_model_root",
    ["granite-3.0-1b-a400m-instruct", "granite-3.0-2b-instruct"],
    indirect=True)
def test_llm_granite(llama_example_root, llm_granite_model_root,
                     llm_datasets_root, llm_rouge_root, llm_venv, cmodel_dir,
                     engine_dir, dtype):
    print("Converting checkpoint...")
    model_name = os.path.basename(llm_granite_model_root)
    os.environ['TRTLLM_DISABLE_UNIFIED_CONVERTER'] = '1'

    ckpt_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llm_granite_model_root,
        data_type=dtype,
    )

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--max_batch_size=8",
        "--max_input_len=924",
        "--max_seq_len=1024",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        f"--moe_plugin={dtype}",
        f"--workers=1",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    summary_cmd = [
        f"{llama_example_root}/../summarize.py",
        f"--engine_dir={engine_dir}",
        f"--hf_model_dir={llm_granite_model_root}",
        f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}"
        "--test_trt_llm",
        "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=25",
        "--batch_size=8",
        "--max_ite=40",
    ]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "1", "--allow-run-as-root"],
                        summary_cmd)

    del os.environ['TRTLLM_DISABLE_UNIFIED_CONVERTER']


@pytest.mark.parametrize(
    "llm_granite_model_root",
    ["granite-3.0-1b-a400m-instruct", "granite-3.0-2b-instruct"],
    indirect=True)
def test_granite_bf16_lora(llama_example_root,
                           llm_datasets_root,
                           qcache_dir,
                           llm_rouge_root,
                           llm_venv,
                           engine_dir,
                           cmodel_dir,
                           llm_granite_model_root,
                           num_beams=1):
    "Run Granite 3.0 models with multiple dummy LoRAs."

    # TODO: Enable fp8 quantization when ModelOpt changes for Granite are available.
    print("Converting checkpoint...")
    model_name = os.path.basename(llm_granite_model_root)
    os.environ['TRTLLM_DISABLE_UNIFIED_CONVERTER'] = '1'
    dtype = 'bfloat16'

    ckpt_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llm_granite_model_root,
        data_type=dtype,
    )
    target_hf_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
    ]
    target_trtllm_modules = [
        "attn_q",
        "attn_k",
        "attn_v",
    ]
    if model_name == "granite-3.0-1b-a400m-instruct":
        target_hf_modules += ["moe_h_to_4h", "moe_4h_to_h", "moe_gate"]
        target_trtllm_modules += ["moe_h_to_4h", "moe_4h_to_h", "moe_gate"]

    test_multi_lora_support(
        hf_model_dir=llm_granite_model_root,
        tllm_ckpt_dir=ckpt_dir,
        engine_dir=engine_dir,
        llm_venv=llm_venv,
        example_root=llama_example_root,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=target_hf_modules,
        target_trtllm_modules=target_trtllm_modules,
        zero_lora_weights=True,
    )
