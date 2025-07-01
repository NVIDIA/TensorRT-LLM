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
from pathlib import Path

import pytest
from defs.common import venv_check_call
from defs.conftest import llm_models_root, unittest_path


def test_llmapi_chat_example(llm_root, llm_venv):
    # Test for the examples/apps/chat.py
    example_root = Path(os.path.join(llm_root, "examples", "apps"))
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    llm_venv.run_cmd(["-m", "pytest", str(test_root / "_test_llm_chat.py")])


def test_llmapi_server_example(llm_root, llm_venv):
    # Test for the examples/apps/fastapi_server.py
    test_root = unittest_path() / "llmapi" / "apps"
    llm_venv.run_cmd(["-m", "pytest", str(test_root / "_test_llm_server.py")])


### LLMAPI examples
def _run_llmapi_example(llm_root, engine_dir, llm_venv, script_name: str,
                        *args):
    example_root = Path(llm_root) / "examples" / "llm-api"
    engine_dir = Path(engine_dir) / "llmapi"
    if not engine_dir.exists():
        engine_dir.mkdir(parents=True)
    examples_script = example_root / script_name

    run_command = [str(examples_script)] + list(args)

    # Create llm models softlink to avoid duplicated downloading for llm api example
    src_dst_dict = {
        f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0":
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        f"{llm_models_root()}/vicuna-7b-v1.3":
        f"{llm_venv.get_working_directory()}/lmsys/vicuna-7b-v1.3",
        f"{llm_models_root()}/medusa-vicuna-7b-v1.3":
        f"{llm_venv.get_working_directory()}/FasterDecoding/medusa-vicuna-7b-v1.3",
        f"{llm_models_root()}/llama3.1-medusa-8b-hf_v0.1":
        f"{llm_venv.get_working_directory()}/nvidia/Llama-3.1-8B-Medusa-FP8",
    }

    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    cnn_dailymail_src = f"{llm_models_root()}/datasets/cnn_dailymail"
    cnn_dailymail_dst = f"{llm_venv.get_working_directory()}/cnn_dailymail"
    if not os.path.islink(cnn_dailymail_dst):
        os.symlink(cnn_dailymail_src,
                   cnn_dailymail_dst,
                   target_is_directory=True)

    venv_check_call(llm_venv, run_command)


@pytest.mark.skip(reason="https://nvbugs/5365825")
def test_llmapi_quickstart(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "quickstart_example.py")


@pytest.mark.skip(reason="https://nvbugs/5365825")
def test_llmapi_example_inference(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "llm_inference.py")


@pytest.mark.skip(reason="https://nvbugs/5365825")
def test_llmapi_example_inference_async(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_inference_async.py")


@pytest.mark.skip(reason="https://nvbugs/5365825")
def test_llmapi_example_inference_async_streaming(llm_root, engine_dir,
                                                  llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_inference_async_streaming.py")


@pytest.mark.skip(reason="https://nvbugs/5365825")
def test_llmapi_example_multilora(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "llm_multilora.py")


@pytest.mark.skip(reason="https://nvbugs/5365825")
def test_llmapi_example_guided_decoding(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_guided_decoding.py")


@pytest.mark.skip_less_device(2)
def test_llmapi_example_distributed_tp2(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_inference_distributed.py")


@pytest.mark.skip(reason="https://nvbugs/5365825")
def test_llmapi_example_logits_processor(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_logits_processor.py")


@pytest.mark.skip(reason="https://nvbugs/5365825")
def test_llmapi_quickstart_atexit(llm_root, engine_dir, llm_venv):
    script_path = Path(
        llm_root
    ) / "tests/integration/defs/examples/run_llm_quickstart_atexit.py"
    llm_venv.run_cmd([str(script_path)])
