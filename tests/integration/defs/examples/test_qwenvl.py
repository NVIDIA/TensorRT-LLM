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
"""Module test_qwen test qwenvl examples."""

import json
import os
import re

import pytest
from defs.common import venv_check_call, venv_check_output
from defs.conftest import get_sm_version
from defs.trt_test_alternative import check_call

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


@pytest.fixture(scope="module")
def qwenvl_example_root(llm_root, llm_venv):
    "Get qwenvl example root"
    example_root = os.path.join(llm_root, "examples", "models", "core",
                                "qwenvl")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.mark.parametrize("llm_qwen_model_root", ["qwen-vl-chat"], indirect=True)
def test_llm_qwenvl_single_gpu_summary(qwenvl_example_root, llm_qwen_model_root,
                                       llm_venv, engine_dir):
    "Build & run qwenvl on 1 gpu."
    workspace = llm_venv.get_working_directory()

    print("Generate vit onnx file and engine...")
    plan_file = f"{workspace}/plan/visual_encoder/visual_encoder_fp16.plan"
    onnx_file = f"{workspace}/visual_encoder/visual_encoder.onnx"
    image = f"{qwenvl_example_root}/pics/demo.jpeg"
    vit_cmd = [
        f"{qwenvl_example_root}/vit_onnx_trt.py",
        f"--pretrained_model_path={llm_qwen_model_root}",
        f"--planFile={plan_file}",
        f"--onnxFile={onnx_file}",
        f"--image_url={image}",
    ]

    venv_check_call(llm_venv, vit_cmd)

    print("Quantize weight...")
    convert_cmd = [
        f"{qwenvl_example_root}/../qwen/convert_checkpoint.py",
        f"--model_dir={llm_qwen_model_root}",
        f"--output_dir={workspace}/Qwen-VL-Chat",
        f"--dtype=float16",
    ]

    venv_check_call(llm_venv, convert_cmd)

    print("Build TRT-LLM engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={workspace}/Qwen-VL-Chat",
        f"--gemm_plugin=float16",
        f"--gpt_attention_plugin=float16",
        f"--max_input_len=2048",
        f"--max_seq_len=3072",
        f"--max_prompt_embedding_table_size=2048",
        f"--remove_input_padding=enable",
        f"--max_beam_width=4",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={8}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run summary...")
    image_path = json.dumps({"image": image})

    run_cmd = [
        f"{qwenvl_example_root}/run.py",
        f"--tokenizer_dir={llm_qwen_model_root}",
        f"--qwen_engine_dir={engine_dir}",
        f"--vit_engine_path={workspace}/plan/visual_encoder/visual_encoder_fp16.plan",
        f"--images_path={image_path}",
        "--num_beams=4",
    ]

    output = venv_check_output(llm_venv, run_cmd)
    output = [line for line in output.split("\n") if "Output(beam:" in line]
    print(output)

    print("Verify the output...")
    results = []
    for item in output:
        match = re.search(r"Output\(beam: \d+\): \"(.*)", item)
        if match:
            results.append(match.group(1))

    for item in results:
        # check the output if it contains key words
        if ("dog" in item or "labrador" in item) and (
                "sea" in item or "beach" in item) and ("woman" in item
                                                       or "girl" in item):
            pass
        else:
            assert False, f"output is: {item}"
