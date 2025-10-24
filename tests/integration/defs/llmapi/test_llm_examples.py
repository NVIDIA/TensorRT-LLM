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
import subprocess
import sys
import threading
from pathlib import Path
from subprocess import PIPE, Popen

import pytest
from defs.common import venv_check_call
from defs.conftest import llm_models_root, unittest_path

from tensorrt_llm.executor.utils import LlmLauncherEnvs


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
def _setup_llmapi_example_softlinks(llm_venv):
    """Create softlinks for LLM models to avoid duplicated downloading for llm api examples"""
    # Create llm models softlink to avoid duplicated downloading for llm api example
    src_dst_dict = {
        # TinyLlama-1.1B-Chat-v1.0
        f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0":
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # vicuna-7b-v1.3
        f"{llm_models_root()}/vicuna-7b-v1.3":
        f"{llm_venv.get_working_directory()}/lmsys/vicuna-7b-v1.3",
        # medusa-vicuna-7b-v1.3
        f"{llm_models_root()}/medusa-vicuna-7b-v1.3":
        f"{llm_venv.get_working_directory()}/FasterDecoding/medusa-vicuna-7b-v1.3",
        # llama3.1-medusa-8b-hf_v0.1
        f"{llm_models_root()}/llama3.1-medusa-8b-hf_v0.1":
        f"{llm_venv.get_working_directory()}/nvidia/Llama-3.1-8B-Medusa-FP8",
        # Llama-3.1-8B-Instruct
        f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct":
        f"{llm_venv.get_working_directory()}/meta-llama/Llama-3.1-8B-Instruct",
        # DeepSeek-V3-Lite/bf16
        f"{llm_models_root()}/DeepSeek-V3-Lite/bf16":
        f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/bf16",
        # EAGLE3-LLaMA3.1-Instruct-8B
        f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B":
        f"{llm_venv.get_working_directory()}/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
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


def _run_llmapi_example(llm_root, engine_dir, llm_venv, script_name: str,
                        *args):
    example_root = Path(llm_root) / "examples" / "llm-api"
    engine_dir = Path(engine_dir) / "llmapi"
    if not engine_dir.exists():
        engine_dir.mkdir(parents=True)
    examples_script = example_root / script_name

    run_command = [str(examples_script)] + list(args)

    _setup_llmapi_example_softlinks(llm_venv)

    venv_check_call(llm_venv, run_command)


def _mpirun_llmapi_example(llm_root,
                           llm_venv,
                           script_name: str,
                           tp_size: int,
                           spawn_extra_main_process: bool = True,
                           *args):
    """Run an llmapi example script with mpirun.

    Args:
        llm_root: Root directory of the LLM project
        llm_venv: Virtual environment object
        script_name: Name of the example script to run
        tp_size: Tensor parallelism size (number of MPI processes)
        spawn_extra_main_process: Whether to spawn extra main process (default: True)
        *args: Additional arguments to pass to the example script
    """
    example_root = Path(llm_root) / "examples" / "llm-api"
    examples_script = example_root / script_name

    # Set environment variable for spawn_extra_main_process
    env_vars = os.environ.copy()
    LlmLauncherEnvs.set_spawn_extra_main_process(spawn_extra_main_process)
    env_vars[LlmLauncherEnvs.TLLM_SPAWN_EXTRA_MAIN_PROCESS] = os.environ[
        LlmLauncherEnvs.TLLM_SPAWN_EXTRA_MAIN_PROCESS]

    run_command = [
        "mpirun", "-n",
        str(tp_size), "--oversubscribe", "--allow-run-as-root"
    ]
    # Pass environment variables through mpirun
    for key, value in [(LlmLauncherEnvs.TLLM_SPAWN_EXTRA_MAIN_PROCESS,
                        env_vars[LlmLauncherEnvs.TLLM_SPAWN_EXTRA_MAIN_PROCESS])
                       ]:
        run_command.extend(["-x", f"{key}={value}"])
    run_command.extend(["python", str(examples_script)] + list(args))

    _setup_llmapi_example_softlinks(llm_venv)

    print(' '.join(run_command))

    with Popen(run_command,
               env=env_vars,
               stdout=PIPE,
               stderr=PIPE,
               bufsize=1,
               start_new_session=True,
               universal_newlines=True,
               cwd=llm_venv.get_working_directory()) as process:

        # Function to read from a stream and write to output
        def read_stream(stream, output_stream):
            for line in stream:
                output_stream.write(line)
                output_stream.flush()

        # Create threads to read stdout and stderr concurrently
        stdout_thread = threading.Thread(target=read_stream,
                                         args=(process.stdout, sys.stdout))
        stderr_thread = threading.Thread(target=read_stream,
                                         args=(process.stderr, sys.stderr))

        # Start both threads
        stdout_thread.start()
        stderr_thread.start()

        # Wait for the process to complete
        return_code = process.wait()

        # Wait for both threads to finish reading
        stdout_thread.join()
        stderr_thread.join()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, run_command)


def test_llmapi_quickstart(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "quickstart_example.py")


def test_llmapi_example_inference(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "llm_inference.py")


def test_llmapi_example_inference_async(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_inference_async.py")


def test_llmapi_example_inference_async_streaming(llm_root, engine_dir,
                                                  llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_inference_async_streaming.py")


def test_llmapi_example_multilora(llm_root, engine_dir, llm_venv):
    cmd_line_args = [
        "--chatbot_lora_dir",
        f"{llm_models_root()}/llama-models-v2/sft-tiny-chatbot",
        "--mental_health_lora_dir",
        f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0-mental-health-conversational",
        "--tarot_lora_dir",
        f"{llm_models_root()}/llama-models-v2/tinyllama-tarot-v1"
    ]
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "llm_multilora.py",
                        *cmd_line_args)


def test_llmapi_example_guided_decoding(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_guided_decoding.py")


@pytest.mark.skip_less_device(2)
def test_llmapi_example_distributed_tp2(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_inference_distributed.py")


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize(
    "spawn_extra_main_process", [True, False],
    ids=["spawn_extra_main_process", "no_spawn_extra_main_process"])
def test_llmapi_example_launch_distributed_tp2(llm_root, llm_venv,
                                               spawn_extra_main_process: bool):
    _mpirun_llmapi_example(llm_root,
                           llm_venv,
                           "llm_inference_distributed.py",
                           tp_size=2,
                           spawn_extra_main_process=spawn_extra_main_process)


def test_llmapi_example_logits_processor(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_logits_processor.py")


def test_llmapi_quickstart_atexit(llm_root, engine_dir, llm_venv):
    script_path = Path(
        llm_root
    ) / "tests/integration/defs/examples/run_llm_quickstart_atexit.py"
    llm_venv.run_cmd([str(script_path)])


@pytest.mark.skip_less_device_memory(80000)
def test_llmapi_speculative_decoding_mtp(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_speculative_decoding.py", "MTP", "--model",
                        f"{llm_models_root()}/DeepSeek-V3-Lite/bf16")


@pytest.mark.skip_less_device_memory(80000)
def test_llmapi_speculative_decoding_eagle3(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_speculative_decoding.py", "EAGLE3")


@pytest.mark.skip_less_device_memory(80000)
def test_llmapi_speculative_decoding_ngram(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "llm_speculative_decoding.py", "NGRAM")


@pytest.mark.skip(reason="https://nvbugs/5365825"
                  )  # maybe unrelated, but this test will always timeout
def test_llmapi_sampling(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "llm_sampling.py")


@pytest.mark.skip(reason="https://nvbugs/5365825"
                  )  # maybe unrelated, but this test will always timeout
def test_llmapi_runtime(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv, "llm_runtime.py")


@pytest.mark.parametrize("model", ["Qwen2-0.5B"])
def test_llmapi_kv_cache_connector(llm_root, llm_venv, model):
    script_path = Path(
        llm_root) / "examples" / "llm-api" / "llm_kv_cache_connector.py"
    model_path = f"{llm_models_root()}/{model}"

    venv_check_call(llm_venv, [str(script_path), model_path])


def test_llmapi_tensorrt_engine(llm_root, engine_dir, llm_venv):
    _run_llmapi_example(llm_root, engine_dir, llm_venv,
                        "_tensorrt_engine/quickstart_example.py")
