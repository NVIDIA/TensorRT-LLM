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

import json
import os
import tempfile
from pathlib import Path

import pytest
from defs.common import convert_weights, venv_check_call
from defs.conftest import llm_models_root, unittest_path
from defs.trt_test_alternative import check_call

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi.llm_utils import BuildConfig


def test_llmapi_quant_llama_70b(llm_root, engine_dir, llm_venv):
    # Test quantizing llama-70b model with only 2 H100 GPUs
    # The background: there is a bug preventing quantization of llama-70b model with <tp-size> GPUs
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(',')
    if len(visible_devices) < 2:
        visible_devices = ['0', '1']
    visible_devices = visible_devices[:2]

    env = {
        'CUDA_VISIBLE_DEVICES': ','.join(visible_devices),
    }
    print(f'env: {env}')

    script_path = Path(
        llm_root
    ) / "tests/integration/defs/examples/run_llm_fp8_quant_llama_70b.py"
    llm_venv.run_cmd([str(script_path)], env=env)


run_llm_path = os.path.join(os.path.dirname(__file__), "_run_llmapi_llm.py")


@pytest.mark.parametrize("model_name,model_path", [
    ("llama", "llama-models-v2/llama-v2-7b-hf"),
])
def test_llmapi_load_engine_from_build_command_with_lora(
        llm_root, llm_venv, engine_dir, model_name, model_path):
    llama_example_root = os.path.join(llm_root, "examples", "models", "core",
                                      model_name)
    dtype = 'bfloat16'
    cmodel_dir = os.path.join(engine_dir, f"{model_name}-engine")

    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=f'{llm_models_root()}/{model_path}',
                               data_type=dtype)

    engine_dir = os.path.join(engine_dir, f"{model_name}-engine")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={1}",
        f"--max_input_len={924}",
        f"--max_seq_len={1024}",
        f"--max_beam_width={1}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
        f"--lora_plugin={dtype}",
        f"--lora_target_modules=attn_q",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    venv_check_call(llm_venv, [
        run_llm_path,
        "--model_dir",
        engine_dir,
    ])


@pytest.mark.skip(reason="https://nvbugs/5574355")
@pytest.mark.parametrize("model_name,model_path", [
    ("llama", "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"),
])
def test_llmapi_build_command_parameters_align(llm_root, llm_venv, engine_dir,
                                               model_name, model_path):
    llama_example_root = os.path.join(llm_root, "examples", model_name)
    dtype = 'float16'
    cmodel_dir = os.path.join(engine_dir, f"{model_name}-engine")

    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=f'{llm_models_root()}/{model_path}',
                               data_type=dtype)

    engine_dir = os.path.join(engine_dir, f"{model_name}-engine")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={4}",
        f"--max_input_len={111}",
        f"--max_seq_len={312}",
        f"--max_beam_width={4}",
        f"--gemm_plugin={dtype}",
        f"--gpt_attention_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    build_config = BuildConfig()
    # change some building parameters
    build_config.max_batch_size = 4
    build_config.max_beam_width = 4
    build_config.max_input_len = 111
    build_config.strongly_typed = True
    build_config.max_seq_len = 312
    build_config.plugin_config._gemm_plugin = dtype
    build_config.plugin_config._gpt_attention_plugin = dtype

    llm = LLM(model=f'{llm_models_root()}/{model_path}',
              build_config=build_config)
    tmpdir = tempfile.TemporaryDirectory()
    llm.save(tmpdir.name)
    build_cmd_cfg = None
    build_llmapi_cfg = None

    with open(os.path.join(engine_dir, "config.json"), "r") as f:
        engine_config = json.load(f)

        build_cmd_cfg = BuildConfig.from_dict(
            engine_config["build_config"]).to_dict()

    with open(os.path.join(tmpdir.name, "config.json"), "r") as f:
        llm_api_engine_cfg = json.load(f)

        build_llmapi_cfg = BuildConfig.from_dict(
            llm_api_engine_cfg["build_config"]).to_dict()

    assert build_cmd_cfg == build_llmapi_cfg


def test_llmapi_load_ckpt_from_convert_command(llm_root, llm_venv, engine_dir):
    llama_example_root = os.path.join(llm_root, "examples", "models", "core",
                                      "llama")
    dtype = 'float16'
    cmodel_dir = os.path.join(engine_dir, "llama-7b-cmodel")

    ckpt_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=llama_example_root,
        cmodel_dir=cmodel_dir,
        model='llama-7b',
        model_path=f'{llm_models_root()}/llama-models/llama-7b-hf',
        data_type=dtype)

    venv_check_call(llm_venv, [
        run_llm_path,
        "--model_dir",
        ckpt_dir,
    ])


def test_llmapi_exit(llm_venv):
    llm_exit_script = unittest_path() / "llmapi/run_llm_exit.py"
    llama_model_dir = Path(
        llm_models_root()) / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"

    run_command = [
        str(llm_exit_script), "--model_dir",
        str(llama_model_dir), "--tp_size", "1"
    ]
    venv_check_call(llm_venv, run_command)


@pytest.mark.skip_less_device(2)
def test_llmapi_exit_multi_gpu(llm_venv):
    llm_exit_script = unittest_path() / "llmapi/run_llm_exit.py"
    llama_model_dir = Path(
        llm_models_root()) / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"

    run_command = [
        str(llm_exit_script), "--model_dir",
        str(llama_model_dir), "--tp_size", "2"
    ]
    venv_check_call(llm_venv, run_command)


@pytest.mark.parametrize("model_name,model_path", [
    ("llama", "llama-models/llama-7b-hf"),
    ("llama", "codellama/CodeLlama-7b-Instruct-hf"),
])
def test_llmapi_load_engine_from_build_command(llm_root, llm_venv, engine_dir,
                                               model_name, model_path):
    llama_example_root = os.path.join(llm_root, "examples", "models", "core",
                                      model_name)
    dtype = 'float16'
    cmodel_dir = os.path.join(engine_dir, f"{model_name}-engine")

    ckpt_dir = convert_weights(llm_venv=llm_venv,
                               example_root=llama_example_root,
                               cmodel_dir=cmodel_dir,
                               model=model_name,
                               model_path=f'{llm_models_root()}/{model_path}',
                               data_type=dtype)

    engine_dir = os.path.join(engine_dir, f"{model_name}-engine")

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

    venv_check_call(llm_venv, [
        run_llm_path,
        "--model_dir",
        engine_dir,
    ])
