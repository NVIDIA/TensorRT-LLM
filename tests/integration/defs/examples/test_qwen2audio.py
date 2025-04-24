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
"""Module test_qwen test qwen2audio examples."""

import os
import re

import pytest
from defs.common import venv_check_call, venv_check_output
from defs.trt_test_alternative import check_call


@pytest.fixture(scope="module")
def qwen2audio_example_root(llm_root, llm_venv):
    "Get qwen2audio example root"
    example_root = os.path.join(llm_root, "examples", "models", "core",
                                "qwen2audio")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.mark.parametrize("llm_qwen_model_root", ["qwen2_audio_7b_instruct"],
                         indirect=True)
def test_llm_qwen2audio_single_gpu(qwen2audio_example_root, llm_qwen_model_root,
                                   llm_venv, engine_dir):
    "Build & run qwen2audio on 1 gpu."
    workspace = llm_venv.get_working_directory()

    print("Generate audio engine...")
    audio_engine_dir = f"{engine_dir}/audio"
    audio_cmd = [
        f"{qwen2audio_example_root}/../multimodal/build_multimodal_engine.py",
        f"--model_type=qwen2_audio",
        f"--model_path={llm_qwen_model_root}",
        f"--max_batch_size=32",
        f"--output_dir={audio_engine_dir}",
    ]

    venv_check_call(llm_venv, audio_cmd)

    print("Convert checkpoint...")
    convert_cmd = [
        f"{qwen2audio_example_root}/../qwen/convert_checkpoint.py",
        f"--model_dir={llm_qwen_model_root}",
        f"--output_dir={workspace}/Qwen2-Audio",
        f"--dtype=float16",
    ]

    venv_check_call(llm_venv, convert_cmd)

    print("Build TRT-LLM engine...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={workspace}/Qwen2-Audio",
        f"--gemm_plugin=float16",
        f"--gpt_attention_plugin=float16",
        f"--max_prompt_embedding_table_size=4096",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engine...")
    audio_url = f"{qwen2audio_example_root}/audio/glass-breaking-151256.mp3"

    run_cmd = [
        f"{qwen2audio_example_root}/run.py",
        f"--tokenizer_dir={llm_qwen_model_root}",
        f"--engine_dir={engine_dir}",
        f"--audio_engine_path={audio_engine_dir}/model.engine",
        f"--audio_url={audio_url}",
    ]

    output = venv_check_output(llm_venv, run_cmd)
    output = [line for line in output.split("\n") if "Output:" in line]
    print(output)

    print("Verify the output...")
    results = []
    for item in output:
        match = re.search(r"Output: \"(.*)", item)
        if match:
            results.append(match.group(1))

    for item in results:
        # check the output if it contains key words
        item = item.lower()
        if ("glass" in item) and ("shatter" in item or "break" in item):
            pass
        else:
            assert False, f"output is: {item}"
