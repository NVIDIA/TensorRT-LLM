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
import time

import pytest
from defs.common import (convert_weights, test_multi_lora_support,
                         venv_mpi_check_call)
from defs.conftest import get_sm_version
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def disable_unified_converter():
    os.environ['TRTLLM_DISABLE_UNIFIED_CONVERTER'] = '1'
    yield
    del os.environ['TRTLLM_DISABLE_UNIFIED_CONVERTER']


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
        f"{llama_example_root}/../../../summarize.py",
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
    start_time = time.time()
    print("Converting checkpoint...")
    convert_start = time.time()
    model_name = os.path.basename(llm_granite_model_root)
    dtype = 'bfloat16'

    ckpt_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llm_granite_model_root,
        data_type=dtype,
    )
    convert_end = time.time()
    print(
        f"Convert checkpoint completed in {(convert_end - convert_start):.2f} seconds."
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

    print("Calling test_multi_lora_support...")
    test_multi_lora_start = time.time()
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
    test_multi_lora_end = time.time()
    print(
        f"test_multi_lora_support completed in {(test_multi_lora_end - test_multi_lora_start):.2f} seconds"
    )

    total_time = time.time() - start_time
    print(f"Total function execution time: {total_time:.2f} seconds")
