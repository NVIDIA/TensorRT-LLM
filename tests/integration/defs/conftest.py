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
# -*- coding: utf-8 -*-

import datetime
import gc
import os
import platform
import re
import shutil
import subprocess as sp
import tempfile
import time
import urllib.request
import warnings
from functools import wraps
from pathlib import Path
from typing import Iterable, Sequence

import defs.ci_profiler
import psutil
import pytest
import torch
import tqdm
import yaml
from _pytest.mark import ParameterSet

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.bindings import ipc_nvls_supported
from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size

from .perf.gpu_clock_lock import GPUClockLock
from .perf.session_data_writer import SessionDataWriter
from .test_list_parser import (TestCorrectionMode, apply_waives,
                               get_test_name_corrections_v2, handle_corrections,
                               modify_by_test_list, preprocess_test_list_lines)
from .trt_test_alternative import (call, check_output, exists, is_windows,
                                   is_wsl, makedirs, print_info, print_warning,
                                   wsl_to_win_path)
from .utils.periodic_junit import PeriodicJUnitXML

try:
    from llm import trt_environment
except ImportError:
    trt_environment = None

# TODO: turn off this when the nightly storage issue is resolved.
DEBUG_CI_STORAGE = os.environ.get("DEBUG_CI_STORAGE", False)
GITLAB_API_USER = os.environ.get("GITLAB_API_USER")
GITLAB_API_TOKEN = os.environ.get("GITLAB_API_TOKEN")


def print_storage_usage(path, tag, capfd):
    if DEBUG_CI_STORAGE:
        stat = shutil.disk_usage(path)
        with capfd.disabled():
            print_info(
                f"\nUsage of {path} {stat} @{tag}, used in GB: {stat.used/(2**30)}"
            )


def wget(url, out):
    filename = os.path.basename(url)
    os.makedirs(out, exist_ok=True)
    urllib.request.urlretrieve(url, os.path.join(out, filename))


def llm_models_root() -> str:
    """Return LLM_MODELS_ROOT path if it is set in env, assert when it's set but not a valid path."""

    root = Path("/home/scratch.trt_llm_data/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ.get("LLM_MODELS_ROOT"))

    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")

    assert root.exists(), (
        "You shall set LLM_MODELS_ROOT env or be able to access scratch.trt_llm_data to run this test"
    )

    return str(root)


def tests_path() -> Path:
    return (Path(os.path.dirname(__file__)) / "../..").resolve()


def unittest_path() -> Path:
    return tests_path() / "unittest"


def integration_path() -> Path:
    return tests_path() / "integration"


def cached_in_llm_models_root(path_relative_to_llm_models_root,
                              fail_if_path_is_invalid=False):
    """
    Use this decorator to declare a cached path in the LLM_MODELS_ROOT directory.

    That decorator is intended to be used with pytest.fixture functions which prepare and return a data path for some tests.

    The cache is only queried when llm_models_root() does not return None, and the cache is skipped otherwise.
    When the cache is queried, and the specified path does not exist, the function:

        - Triggers an AssertionFailure when fail_if_path_is_invalid is True,
        - Ignore the invalid path and fallbacks to calling the fixture otherwise.

    The purpose of the `fail_if_path_is_invalid` is the following:
        -  If you submit a test and the data is not in the cached NFS LLM_MODELS_ROOT dir yet, you can use `fail_if_path_is_invalid=False` (the default).
        In that case, the fixture will use the fallback path and ignore the cache miss in the CI. After submitting the data to the cached NFS LLM_MODELS_ROOT dir,
        your test will automatically pickup the cached data.

        - If your data is known to always be in the LLM_MODELS_ROOT, and you want to make sure that the test fails loudly when it misses in cache,
          you should specify fail_if_path_is_invalid=True to force the failure. It is useful for when a cache miss will cause a big performance drop for the CI jobs.

       Example:
       If you have a fixture which downloads the SantaCoder repo and returns its path for one SantaCoder test, you can do the following:

       @pytest.fixture(scope="session")
       def llm_gpt2_santacoder_model_root(llm_venv):
           workspace = llm_venv.get_working_directory()
           gpt2_santacoder_model_root = os.path.join(workspace, "santacoder")
           call(
               f"git clone https://huggingface.co/bigcode/santacoder {gpt2_santacoder_model_root}",
               shell=True)
           return gpt2_santacoder_model_root

        At some point, if you decide to cache the SantaCoder in the LLM_MODELS_ROOT, you can decorate the fixture to enforce the test to
        use the ${LLM_MODELS_ROOT}/santacoder cached directory. You can upload SantaCoder to that location before or after submitting
        this code since there is a fallback path to clone the repo if it is not found in cache.

        @pytest.fixture(scope="session")
        @cached_in_llm_models_root("santacoder")
        def llm_gpt2_santacoder_model_root(llm_venv):
            ... keep the original code
    """

    def wrapper(f):

        @wraps(f)
        def decorated(*args, **kwargs):
            cached_dir = f"{llm_models_root()}/{path_relative_to_llm_models_root}"
            if os.path.exists(cached_dir):
                return cached_dir
            elif fail_if_path_is_invalid:
                assert (
                    False
                ), f"{cached_dir} does not exist, and fail_if_path_is_invalid is True, please check the cache directory"
            return f(*args, **kwargs)

        return decorated

    return wrapper


# Fixture about whether the current pipeline is running in TRT environment.
@pytest.fixture(scope="session")
def is_trt_environment():
    return trt_environment is not None


# Helper function to get llm_root. Do not define it as a fixture so that this
# function can be used during test collection phase.
def get_llm_root(trt_config=None, gitlab_token=None):
    if trt_environment:
        return trt_environment.setup_tensorrt_llm_repo(trt_config, gitlab_token)
    llm_repo_root = os.environ.get("LLM_ROOT", None)
    if llm_repo_root is None:
        llm_repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        print_warning(
            f"The LLM_ROOT env var is not defined! Using {llm_repo_root} as LLM_ROOT."
        )
    return llm_repo_root


@pytest.fixture(scope="session")
def llm_root():
    return get_llm_root()


@pytest.fixture(scope="session")
def llm_backend_root():
    llm_root_directory = get_llm_root()
    llm_backend_repo_root = os.path.join(llm_root_directory, "triton_backend")
    return llm_backend_repo_root


@pytest.fixture(scope="session")
def llm_datasets_root() -> str:
    return os.path.join(llm_models_root(), "datasets")


@pytest.fixture(scope="session")
def llm_rouge_root() -> str:
    return os.path.join(llm_models_root(), "rouge")


@pytest.fixture(scope="module")
def bert_example_root(llm_root):
    "Get bert example root"
    example_root = os.path.join(llm_root, "examples", "models", "core", "bert")

    return example_root


@pytest.fixture(scope="module")
def enc_dec_example_root(llm_root):
    "Get encoder-decoder example root"
    example_root = os.path.join(llm_root, "examples", "models", "core",
                                "enc_dec")

    return example_root


@pytest.fixture(scope="module")
def whisper_example_root(llm_root, llm_venv):
    "Get whisper example root"
    example_root = os.path.join(llm_root, "examples", "models", "core",
                                "whisper")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])
    return example_root


@pytest.fixture(scope="module")
def opt_example_root(llm_root, llm_venv):
    "Get opt example root"

    example_root = os.path.join(llm_root, "examples", "models", "contrib",
                                "opt")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def llama_example_root(llm_root, llm_venv):
    "Get llama example root"

    example_root = os.path.join(llm_root, "examples", "models", "core", "llama")
    try:
        llm_venv.run_cmd([
            "-m",
            "pip",
            "install",
            "-r",
            os.path.join(example_root, "requirements.txt"),
        ])
    except:
        print("pip install error!")

    return example_root


@pytest.fixture(scope="module")
def llmapi_example_root(llm_root, llm_venv):
    "Get llm api example root"

    example_root = os.path.join(llm_root, "examples", "llm-api")

    return example_root


@pytest.fixture(scope="module")
def disaggregated_example_root(llm_root, llm_venv):
    "Get disaggregated example root"

    example_root = os.path.join(llm_root, "examples", "disaggregated")

    return example_root


@pytest.fixture(scope="module")
def gemma_example_root(llm_root, llm_venv):
    "Get gemma example root"

    example_root = os.path.join(llm_root, "examples", "models", "core", "gemma")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="function")
def gemma_model_root(request):
    "Get gemma model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    if hasattr(request, "param"):
        gemma_model_root = os.path.join(models_root, f"gemma/{request.param}")

    assert exists(gemma_model_root), f"{gemma_model_root} does not exist!"

    return gemma_model_root


@pytest.fixture(scope="function")
def minitron_model_root(request):
    "Get minitron model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    if hasattr(request, "param"):
        assert request.param == "4b"
        minitron_model_root = os.path.join(models_root,
                                           "nemotron/Minitron-4B-Base")

    assert exists(minitron_model_root), f"{minitron_model_root} does not exist!"

    return minitron_model_root


@pytest.fixture(scope="function")
def mistral_nemo_model_root(request):
    "Get Mistral Nemo model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    if hasattr(request, "param"):
        assert request.param == "Mistral-Nemo-12b-Base"
        mistral_nemo_model_root = os.path.join(models_root,
                                               "Mistral-Nemo-Base-2407")
    assert exists(
        mistral_nemo_model_root), f"{mistral_nemo_model_root} does not exist!"
    return mistral_nemo_model_root


@pytest.fixture(scope="function")
def mistral_nemo_minitron_model_root(request):
    "Get Mistral Nemo Minitron model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    if hasattr(request, "param"):
        assert request.param == "Mistral-NeMo-Minitron-8B-Instruct"
        mistral_nemo_minitron_model_root = os.path.join(
            models_root, "Mistral-NeMo-Minitron-8B-Instruct")
    assert exists(mistral_nemo_minitron_model_root
                  ), f"{mistral_nemo_minitron_model_root} does not exist!"
    return mistral_nemo_minitron_model_root


@pytest.fixture(scope="module")
def gpt_example_root(llm_root, llm_venv):
    "Get gpt example root"
    example_root = os.path.join(llm_root, "examples", "models", "core", "gpt")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def gptj_example_root(llm_root, llm_venv):
    "Get gptj example root"
    example_root = os.path.join(llm_root, "examples", "models", "contrib",
                                "gptj")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def glm_4_9b_example_root(llm_root, llm_venv):
    "Get glm-4-9b example root"
    example_root = os.path.join(llm_root, "examples", "models", "core",
                                "glm-4-9b")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def exaone_example_root(llm_root, llm_venv):
    "Get EXAONE example root"
    example_root = os.path.join(llm_root, "examples", "models", "core",
                                "exaone")

    return example_root


@pytest.fixture(scope="function")
def llm_exaone_model_root(request) -> str:
    "Get EXAONE model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    exaone_model_root = os.path.join(models_root, "exaone")
    if hasattr(request, "param"):
        if request.param == "exaone_3.0_7.8b_instruct":
            exaone_model_root = os.path.join(models_root, "exaone")
        elif request.param == "exaone_deep_2.4b":
            exaone_model_root = os.path.join(models_root, "EXAONE-Deep-2.4B")

    return exaone_model_root


@pytest.fixture(scope="module")
def falcon_example_root(llm_root, llm_venv):
    "Get falcon example root"
    example_root = os.path.join(llm_root, "examples", "models", "contrib",
                                "falcon")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="session")
def plugin_gen_path(llm_root):
    "Path to the plugin_gen.py script"
    return os.path.join(llm_root, "tensorrt_llm", "tools", "plugin_gen",
                        "plugin_gen.py")


@pytest.fixture(scope="module")
def internlm2_example_root(llm_root, llm_venv):
    "Get internlm2 example root"
    example_root = os.path.join(llm_root, "examples", "models", "core",
                                "internlm2")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def qwen_example_root(llm_root, llm_venv):
    "Get qwen example root"
    example_root = os.path.join(llm_root, "examples", "models", "core", "qwen")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def draft_target_model_example_root(llm_root, llm_venv):
    "Get Draft-Target-Model example root"
    example_root = os.path.join(llm_root, "examples", "draft_target_model")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def ngram_example_root(llm_root, llm_venv):
    "Get NGram example root"
    example_root = os.path.join(llm_root, "examples", "ngram")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def medusa_example_root(llm_root, llm_venv):
    "Get medusa example root"
    example_root = os.path.join(llm_root, "examples", "medusa")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def redrafter_example_root(llm_root, llm_venv):
    "Get ReDrafter example root"
    example_root = os.path.join(llm_root, "examples", "redrafter")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def eagle_example_root(llm_root, llm_venv):
    "Get EAGLE example root"
    example_root = os.path.join(llm_root, "examples", "eagle")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def mamba_example_root(llm_root, llm_venv):
    "Get mamba example root"
    example_root = os.path.join(llm_root, "examples", "models", "core", "mamba")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    yield example_root

    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(llm_root, "requirements.txt")
    ])


@pytest.fixture(scope="module")
def recurrentgemma_example_root(llm_root, llm_venv):
    "Get recurrentgemma example root"
    example_root = os.path.join(llm_root, "examples", "models", "core",
                                "recurrentgemma")

    # install requirements
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    yield example_root

    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(llm_root, "requirements.txt")
    ])


@pytest.fixture(scope="module")
def nemotron_nas_example_root(llm_root, llm_venv):
    example_root = os.path.join(llm_root, "examples", "models", "core",
                                "nemotron_nas")

    yield example_root


@pytest.fixture(scope="module")
def nemotron_example_root(llm_root, llm_venv):
    "Get nemotron example root"
    example_root = os.path.join(llm_root, "examples", "models", "core",
                                "nemotron")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])
    return example_root


@pytest.fixture(scope="module")
def commandr_example_root(llm_root, llm_venv):
    "Get commandr example root"
    example_root = os.path.join(llm_root, "examples", "models", "core",
                                "commandr")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def deepseek_v2_example_root(llm_root, llm_venv):
    "Get deepseek v2 example root"
    example_root = os.path.join(llm_root, "examples", "models", "contrib",
                                "deepseek_v2")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="function")
def deepseek_v3_model_root(request):
    models_root = llm_models_root()
    if request.param == "DeepSeek-V3":
        deepseek_v3_model_root = os.path.join(models_root, "DeepSeek-V3")
    elif request.param == "DeepSeek-V3-Lite-bf16":
        deepseek_v3_model_root = os.path.join(models_root, "DeepSeek-V3-Lite",
                                              "bf16")
    elif request.param == "DeepSeek-V3-Lite-fp8":
        deepseek_v3_model_root = os.path.join(models_root, "DeepSeek-V3-Lite",
                                              "fp8")
    elif request.param == "DeepSeek-V3-Lite-nvfp4_moe_only":
        deepseek_v3_model_root = os.path.join(models_root, "DeepSeek-V3-Lite",
                                              "nvfp4_moe_only")
    assert exists(
        deepseek_v3_model_root), f"{deepseek_v3_model_root} does not exist!"
    return deepseek_v3_model_root


@pytest.fixture(scope="session")
def trt_performance_cache_name():
    return "performance.cache"


@pytest.fixture(scope="session")
def trt_performance_cache_fpath(llm_venv, trt_performance_cache_name):
    workspace = llm_venv.get_working_directory()
    fpath = os.path.join(workspace, trt_performance_cache_name)
    if is_wsl():
        return wsl_to_win_path(fpath)
    return fpath


# Get the executing perf case name
@pytest.fixture(autouse=True)
def perf_case_name(request):
    return request.node.nodeid


@pytest.fixture(scope="session")
def output_dir(request):
    output = request.config.getoption("--output-dir")
    if output:
        os.makedirs(str(output), exist_ok=True)
    return output


@pytest.fixture(scope="session")
def trt_gpu_clock_lock(request):
    """
    Fixture for the GPUClockLock, used to interface with pynvml to get system properties and to lock/monitor GPU clocks.
    """
    gpu_list = get_gpu_device_list()
    gpu_ids = [gpu.split()[1][:-1] for gpu in gpu_list]  # Extract GPU IDs
    gpu_ids_str = ",".join(gpu_ids)
    gpu_clock_lock = GPUClockLock(
        gpu_id=gpu_ids_str,
        interval_ms=1000.0,
    )

    yield gpu_clock_lock

    gpu_clock_lock.teardown()


@pytest.fixture(scope="session")
def llm_session_data_writer(request, trt_gpu_clock_lock, output_dir):
    """
    Fixture for the SessionDataWriter, used to write session data to output directory.
    """
    session_data_writer = SessionDataWriter(
        log_output_directory=output_dir,
        output_formats=request.config.getoption("--perf-log-formats"),
        gpu_clock_lock=trt_gpu_clock_lock,
    )

    yield session_data_writer

    session_data_writer.teardown()


@pytest.fixture(scope="session")
def custom_user_workspace(request):
    return request.config.getoption("--workspace")


@pytest.fixture(scope="session")
def llm_venv(llm_root, custom_user_workspace):
    workspace_dir = custom_user_workspace
    subdir = datetime.datetime.now().strftime("ws-%Y-%m-%d-%H-%M-%S")
    if workspace_dir is None:
        workspace_dir = "llm-test-workspace"
    workspace_dir = os.path.join(workspace_dir, subdir)
    from defs.local_venv import PythonVenvRunnerImpl

    venv = PythonVenvRunnerImpl("", "", "python3",
                                os.path.join(os.getcwd(), workspace_dir))
    yield venv
    # Remove the workspace directory
    if os.path.exists(workspace_dir):
        print(f"Cleaning up workspace: {workspace_dir}")
        try:
            shutil.rmtree(workspace_dir)
        except Exception as e:
            print(f"Failed to clean up workspace: {e}")


@pytest.fixture(scope="session")
@cached_in_llm_models_root("gpt-next/megatron_converted_843m_tp1_pp1.nemo",
                           True)
def gpt_next_root():
    "get gpt-next/megatron_converted_843m_tp1_pp1.nemo"
    raise RuntimeError("megatron_converted_843m_tp1_pp1.nemo must be cached")


@pytest.fixture(scope="function")
def bert_model_root(hf_bert_model_root):
    "Get bert model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    bert_model_name = hf_bert_model_root
    bert_model_root = os.path.join(models_root, bert_model_name)

    assert os.path.exists(
        bert_model_root
    ), f"{bert_model_root} does not exist under NFS LLM_MODELS_ROOT dir"

    return (bert_model_name, bert_model_root)


@pytest.fixture(scope="function")
def enc_dec_model_root(request):
    "Get enc-dec model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    tllm_model_name = request.param
    if not "wmt" in tllm_model_name:
        # HuggingFace root
        enc_dec_model_root = os.path.join(models_root, tllm_model_name)
    else:
        # FairSeq root
        enc_dec_model_root = os.path.join(models_root, "fairseq-models",
                                          tllm_model_name)

    assert os.path.exists(
        enc_dec_model_root
    ), f"{enc_dec_model_root} does not exist under NFS LLM_MODELS_ROOT dir"

    return (tllm_model_name, enc_dec_model_root)


@pytest.fixture(scope="function")
def whisper_model_root(request):
    "Get whisper model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    assert request.param in [
        "large-v2",
        "large-v3",
    ], "whisper only supports large-v2 or large-v3 for now"
    tllm_model_name = request.param
    whisper_model_root = os.path.join(models_root, "whisper-models",
                                      tllm_model_name)
    assert os.path.exists(
        whisper_model_root
    ), f"{whisper_model_root} does not exist under NFS LLM_MODELS_ROOT dir"

    return (tllm_model_name, whisper_model_root)


@pytest.fixture(scope="function")
def whisper_example_audio_file(whisper_model_root):
    return os.path.join(whisper_model_root[1], "1221-135766-0002.wav")


@pytest.fixture(scope="function")
def multimodal_model_root(request, llm_venv):
    "Get multimodal model root"
    models_root = os.path.join(llm_models_root(), "multimodals")
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    tllm_model_name = request.param
    if "VILA" in tllm_model_name:
        models_root = os.path.join(llm_models_root(), "vila")
    if "cogvlm-chat" in tllm_model_name:
        models_root = os.path.join(llm_models_root(), "cogvlm-chat")
    if "video-neva" in tllm_model_name:
        models_root = os.path.join(llm_models_root(), "video-neva")
        tllm_model_name = tllm_model_name + ".nemo"
    if "neva-22b" in tllm_model_name:
        models_root = os.path.join(llm_models_root(), "neva")
        tllm_model_name = tllm_model_name + ".nemo"
    elif "Llama-3.2" in tllm_model_name:
        models_root = os.path.join(llm_models_root(), "llama-3.2-models")
    elif "Mistral-Small" in tllm_model_name:
        models_root = llm_models_root()

    multimodal_model_root = os.path.join(models_root, tllm_model_name)

    if "llava-onevision" in tllm_model_name and "video" in tllm_model_name:
        multimodal_model_root = multimodal_model_root[:-6]
    elif "llava-v1.6" in tllm_model_name and "vision-trtllm" in tllm_model_name:
        multimodal_model_root = multimodal_model_root[:-14]

    assert os.path.exists(
        multimodal_model_root
    ), f"{multimodal_model_root} does not exist under NFS LLM_MODELS_ROOT dir"

    yield (tllm_model_name, multimodal_model_root)

    if "llava-onevision" in tllm_model_name:
        llm_venv.run_cmd(["-m", "pip", "uninstall", "llava", "-y"])


def remove_file(fn):
    if os.path.isfile(fn) or os.path.islink(fn):
        os.remove(fn)


@pytest.fixture(scope="module")
@cached_in_llm_models_root("replit-code-v1_5-3b", True)
def llm_replit_code_v1_5_3b_model_root():
    "Get replit-code-v1_5-3b model root"
    raise RuntimeError("replit-code-v1_5-3b must be cached")


@pytest.fixture(scope="module")
@cached_in_llm_models_root("gpt2", True)
def llm_gpt2_model_root():
    "Get gpt2 model root"
    raise RuntimeError("gpt2 must be cached")


@pytest.fixture(scope="module")
@cached_in_llm_models_root("gpt2-medium", True)
def llm_gpt2_medium_model_root():
    "Get gpt2 medium model root"
    raise RuntimeError("gpt2-medium must be cached")


@pytest.fixture(scope="module")
@cached_in_llm_models_root("GPT-2B-001_bf16_tp1.nemo", True)
def llm_gpt2_next_model_root():
    "get gpt-2b-001_bf16_tp1.nemo"
    raise RuntimeError("GPT-2B-001_bf16_tp1.nemo must be cached")


@pytest.fixture(scope="module")
@cached_in_llm_models_root("santacoder", True)
def llm_gpt2_santacoder_model_root():
    "get santacoder data"
    raise RuntimeError("santacoder must be cached")


@pytest.fixture(scope="module")
def llm_gpt2_starcoder_model_root(llm_venv, request):
    "get starcoder-model"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    starcoder_model_root = os.path.join(models_root, "starcoder-model")
    if hasattr(request, "param"):
        if request.param == "starcoder":
            starcoder_model_root = os.path.join(models_root, "starcoder-model")
        elif request.param == "starcoderplus":
            starcoder_model_root = os.path.join(models_root, "starcoderplus")
        elif request.param == "starcoder2":
            starcoder_model_root = os.path.join(models_root, "starcoder2-model")

    return starcoder_model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("starcoder2-3b", True)
def llm_gpt2_starcoder2_model_root():
    "get starcoder2-3b"
    raise RuntimeError("starcoder2-3b must be cached")


@pytest.fixture(scope="function")
def starcoder_model_root(request):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    if request.param == "starcoder":
        starcoder_model_root = os.path.join(models_root, "starcoder-model")
    elif request.param == "starcoder2-15b":
        starcoder_model_root = os.path.join(models_root, "starcoder2-model")
    elif request.param == "starcoder2-3b":
        starcoder_model_root = os.path.join(models_root, "starcoder2-3b")
    elif request.param == "starcoderplus":
        starcoder_model_root = os.path.join(models_root, "starcoderplus")

    assert os.path.exists(
        starcoder_model_root
    ), f"{starcoder_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return starcoder_model_root


@pytest.fixture(scope="function")
def llm_gpt2b_lora_model_root(request):
    "get gpt2b lora model"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    model_root_list = []
    lora_root = os.path.join(models_root, "lora", "gpt-next-2b")
    if hasattr(request, "param"):
        if isinstance(request.param, tuple):
            model_list = list(request.param)
        else:
            model_list = [request.param]

        for item in model_list:
            if item == "gpt2b_lora-900.nemo":
                model_root_list.append(
                    os.path.join(lora_root, "gpt2b_lora-900.nemo"))
            elif item == "gpt2b_lora-stories.nemo":
                model_root_list.append(
                    os.path.join(lora_root, "gpt2b_lora-stories.nemo"))

    return ",".join(model_root_list)


@pytest.fixture(scope="module")
def llama_tokenizer_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    # Use llama-7b-hf to load tokenizer
    llama_tokenzier_model_root = os.path.join(models_root, "llama-models",
                                              "llama-7b-hf")
    return llama_tokenzier_model_root


@pytest.fixture(scope="module")
def llama_v2_tokenizer_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    llama_v2_tokenizer_model_root = os.path.join(models_root, "llama-models-v2")

    assert os.path.exists(
        llama_v2_tokenizer_model_root
    ), f"{llama_v2_tokenizer_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return llama_v2_tokenizer_model_root


@pytest.fixture(scope="function")
def llama_model_root(request):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    if request.param == "llama-7b":
        llama_model_root = os.path.join(models_root, "llama-models",
                                        "llama-7b-hf")
    elif request.param == "llama-30b":
        llama_model_root = os.path.join(models_root, "llama-models",
                                        "llama-30b-hf")
    elif request.param == "TinyLlama-1.1B-Chat-v1.0":
        llama_model_root = os.path.join(models_root, "llama-models-v2",
                                        "TinyLlama-1.1B-Chat-v1.0")
    elif request.param == "llama-v2-7b":
        llama_model_root = os.path.join(models_root, "llama-models-v2", "7B")
    elif request.param == "llama-v2-70b":
        llama_model_root = os.path.join(models_root, "llama-models-v2", "70B")
    elif request.param == "llama-v2-70b-hf":
        llama_model_root = os.path.join(models_root, "llama-models-v2",
                                        "llama-v2-70b-hf")
    elif request.param == "Llama-2-7B-AWQ":
        llama_model_root = os.path.join(models_root, "llama-models-v2",
                                        "Llama-2-7B-AWQ")
    elif request.param == "Llama-2-7B-GPTQ":
        llama_model_root = os.path.join(models_root, "llama-models-v2",
                                        "Llama-2-7B-GPTQ")
    elif request.param == "llama-v2-13b-hf":
        llama_model_root = os.path.join(models_root, "llama-models-v2",
                                        "llama-v2-13b-hf")
    elif request.param == "llama-v2-7b-hf":
        llama_model_root = os.path.join(models_root, "llama-models-v2",
                                        "llama-v2-7b-hf")
    elif request.param == "llama-v2-70b-hf":
        llama_model_root = os.path.join(models_root, "llama-models-v2",
                                        "llama-v2-70b-hf")
    elif request.param == "llama-v3-8b-hf":
        llama_model_root = os.path.join(models_root, "llama-models-v3", "8B")
    elif request.param == "llama-v3-8b-instruct-hf":
        llama_model_root = os.path.join(models_root, "llama-models-v3",
                                        "llama-v3-8b-instruct-hf")
    elif request.param == "Llama-3-8B-Instruct-Gradient-1048k":
        llama_model_root = os.path.join(models_root, "llama-models-v3",
                                        "Llama-3-8B-Instruct-Gradient-1048k")
    elif request.param == "Llama-3-70B-Instruct-Gradient-1048k":
        llama_model_root = os.path.join(models_root, "llama-models-v3",
                                        "Llama-3-70B-Instruct-Gradient-1048k")
    elif request.param == "llama-3.1-405b":
        llama_model_root = os.path.join(models_root, "llama-3.1-model",
                                        "Meta-Llama-3.1-405B")
    elif request.param == "llama-3.1-405b-fp8":
        llama_model_root = os.path.join(models_root, "llama-3.1-model",
                                        "Meta-Llama-3.1-405B-FP8")
    elif request.param == "llama-3.1-70b":
        llama_model_root = os.path.join(models_root, "llama-3.1-model",
                                        "Meta-Llama-3.1-70B")
    elif request.param == "llama-3.1-8b":
        llama_model_root = os.path.join(models_root, "llama-3.1-model",
                                        "Meta-Llama-3.1-8B")
    elif request.param == "llama-3.1-8b-instruct-hf-fp8":
        llama_model_root = os.path.join(models_root, "llama-3.1-model",
                                        "Llama-3.1-8B-Instruct-FP8")
    elif request.param == "llama-3.1-8b-instruct":
        llama_model_root = os.path.join(models_root, "llama-3.1-model",
                                        "Llama-3.1-8B-Instruct")
    elif request.param == "llama-3.1-8b-hf-nvfp4":
        llama_model_root = os.path.join(models_root, "nvfp4-quantized",
                                        "Meta-Llama-3.1-8B")
    elif request.param == "llama-3.1-70b-instruct":
        llama_model_root = os.path.join(models_root, "llama-3.1-model",
                                        "Meta-Llama-3.1-70B-Instruct")
    elif request.param == "llama-3.2-1b":
        llama_model_root = os.path.join(models_root, "llama-3.2-models",
                                        "Llama-3.2-1B")
    elif request.param == "llama-3.2-1b-instruct":
        llama_model_root = os.path.join(models_root, "llama-3.2-models",
                                        "Llama-3.2-1B-Instruct")
    elif request.param == "llama-3.2-3b":
        llama_model_root = os.path.join(models_root, "llama-3.2-models",
                                        "Llama-3.2-3B")
    elif request.param == "llama-3.2-3b-instruct":
        llama_model_root = os.path.join(models_root, "llama-3.2-models",
                                        "Llama-3.2-3B-Instruct")
    elif request.param == "llama-3.3-70b-instruct":
        llama_model_root = os.path.join(models_root, "llama-3.3-models",
                                        "Llama-3.3-70B-Instruct")
    assert os.path.exists(
        llama_model_root
    ), f"{llama_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return llama_model_root


@pytest.fixture(scope="function")
def code_llama_model_root(request):
    "get CodeLlama model data"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    if request.param == "CodeLlama-7b-Instruct":
        codellama_model_root = os.path.join(models_root, "codellama",
                                            "CodeLlama-7b-Instruct-hf")
    elif request.param == "CodeLlama-13b-Instruct":
        codellama_model_root = os.path.join(models_root, "codellama",
                                            "CodeLlama-13b-Instruct-hf")
    elif request.param == "CodeLlama-34b-Instruct":
        codellama_model_root = os.path.join(models_root, "codellama",
                                            "CodeLlama-34b-Instruct-hf")
    elif request.param == "CodeLlama-70b-hf":
        codellama_model_root = os.path.join(models_root, "codellama",
                                            "CodeLlama-70b-hf")
    return codellama_model_root


@pytest.fixture(scope="function")
def draft_target_model_roots(request):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    draft_model_root = None
    target_model_root = None
    if request.param == "gpt2":
        draft_model_root = os.path.join(models_root, "gpt2-medium")
        target_model_root = os.path.join(models_root, "gpt2-medium")
    elif request.param == "llama_v2":
        draft_model_root = os.path.join(models_root,
                                        "llama-models-v2/llama-v2-7b-hf")
        target_model_root = os.path.join(models_root,
                                         "llama-models-v2/llama-v2-13b-hf")

    assert os.path.exists(
        draft_model_root
    ), f"Draft-Target-Model draft model path {draft_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    assert os.path.exists(
        target_model_root
    ), f"Draft-Target-Model target model path {target_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return draft_model_root, target_model_root


@pytest.fixture(scope="function")
def ngram_root(request):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    if request.param == "gpt2":
        models_root = os.path.join(models_root, "gpt2-medium")
    elif request.param == "llama_v2":
        models_root = os.path.join(models_root,
                                   "llama-models-v2/llama-v2-13b-hf")
    assert os.path.exists(
        models_root
    ), f"NGram model path {models_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return models_root


@pytest.fixture(scope="function")
def medusa_model_roots(request):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    base_model_root_for_medusa = None
    medusa_heads_model_root = None
    if request.param == "medusa-vicuna-7b-v1.3":
        base_model_root_for_medusa = os.path.join(models_root, "vicuna-7b-v1.3")
        medusa_heads_model_root = os.path.join(models_root,
                                               "medusa-vicuna-7b-v1.3")
    elif request.param == "llama3.1-medusa-8b-hf_v0.1":
        base_model_root_for_medusa = os.path.join(models_root,
                                                  "llama3.1-medusa-8b-hf_v0.1")
        medusa_heads_model_root = base_model_root_for_medusa
    assert os.path.exists(
        base_model_root_for_medusa
    ), f"Medusa base model path {base_model_root_for_medusa} does not exist under NFS LLM_MODELS_ROOT dir"
    assert os.path.exists(
        medusa_heads_model_root
    ), f"Medusa heads model path {medusa_heads_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return base_model_root_for_medusa, medusa_heads_model_root


@pytest.fixture(scope="function")
def lookahead_model_roots(request):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    base_model_root_for_lookahead = None
    if request.param == "vicuna-7b-v1.3":
        base_model_root_for_lookahead = os.path.join(models_root,
                                                     "vicuna-7b-v1.3")
    assert os.path.exists(
        base_model_root_for_lookahead
    ), f"Lookahead base model path {base_model_root_for_lookahead} does not exist under NFS LLM_MODELS_ROOT dir"
    return base_model_root_for_lookahead


@pytest.fixture(scope="function")
def redrafter_model_roots(request):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    base_model_root_for_redrafter = None
    redrafter_drafting_model_root = None
    if request.param == "redrafter-vicuna-7b-v1.3":
        base_model_root_for_redrafter = os.path.join(models_root,
                                                     "vicuna-7b-v1.3")
        redrafter_drafting_model_root = os.path.join(
            models_root, "redrafter-vicuna-7b-v1.3")
    assert os.path.exists(
        base_model_root_for_redrafter
    ), f"ReDrafter base model path {base_model_root_for_redrafter} does not exist under NFS LLM_MODELS_ROOT dir"
    assert os.path.exists(
        redrafter_drafting_model_root
    ), f"ReDrafter heads model path {redrafter_drafting_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return base_model_root_for_redrafter, redrafter_drafting_model_root


@pytest.fixture(scope="function")
def eagle_model_roots(request):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    base_model_root_for_eagle = None
    eagle_heads_model_root = None
    if request.param == "EAGLE-Vicuna-7B-v1.3":
        # Test the checkpoint released from HF, which requires two separate weights,
        # one for the base model and one for the EagleNets.
        base_model_root_for_eagle = os.path.join(models_root, "vicuna-7b-v1.3")
        eagle_heads_model_root = os.path.join(models_root,
                                              "EAGLE-Vicuna-7B-v1.3")
        assert os.path.exists(
            base_model_root_for_eagle
        ), f"EAGLE base model path {base_model_root_for_eagle} does not exist under NFS LLM_MODELS_ROOT dir"
        assert os.path.exists(
            eagle_heads_model_root
        ), f"EAGLE heads model path {eagle_heads_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
        return base_model_root_for_eagle, eagle_heads_model_root

    elif request.param == "llama3.1-eagle-8b-hf_v0.5":
        # Test the checkpoint released from ModelOpt, which only requires one weight,
        # which includes both the base model and EagleNets, and is an FP8 datatype.
        modelopt_checkpoint_root_for_eagle = os.path.join(
            models_root, "modelopt-hf-model-hub", "llama3.1-eagle-8b-hf_v0.5")
        assert os.path.exists(
            modelopt_checkpoint_root_for_eagle
        ), f"EAGLE ModelOpt checkpoint path {modelopt_checkpoint_root_for_eagle} does not exist under NFS LLM_MODELS_ROOT dir"
        return modelopt_checkpoint_root_for_eagle
    else:
        assert "Error Eagle weight's name"


@pytest.fixture(scope="function")
def mamba_model_root(request):
    "get mamba model data"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    mamba_model_root = os.path.join(models_root, "mamba", "mamba-130m-hf")
    if hasattr(request, "param"):
        if request.param == "mamba-2.8b":
            mamba_model_root = os.path.join(models_root, "mamba",
                                            "mamba-2.8b-hf")
        elif request.param == "mamba-130m":
            mamba_model_root = os.path.join(models_root, "mamba",
                                            "mamba-130m-hf")
        elif request.param == "mamba-1.4b":
            mamba_model_root = os.path.join(models_root, "mamba",
                                            "mamba-1.4b-hf")
        elif request.param == "mamba-790m":
            mamba_model_root = os.path.join(models_root, "mamba",
                                            "mamba-790m-hf")
        elif request.param == "mamba-370m":
            mamba_model_root = os.path.join(models_root, "mamba",
                                            "mamba-370m-hf")
        elif request.param == "mamba2-2.7b":
            mamba_model_root = os.path.join(models_root, "mamba2",
                                            "mamba2-2.7b")
        elif request.param == "mamba2-1.3b":
            mamba_model_root = os.path.join(models_root, "mamba2",
                                            "mamba2-1.3b")
        elif request.param == "mamba2-780m":
            mamba_model_root = os.path.join(models_root, "mamba2",
                                            "mamba2-780m")
        elif request.param == "mamba2-370m":
            mamba_model_root = os.path.join(models_root, "mamba2",
                                            "mamba2-370m")
        elif request.param == "mamba2-130m":
            mamba_model_root = os.path.join(models_root, "mamba2",
                                            "mamba2-130m")
        elif request.param == "mamba-codestral-7B-v0.1":
            mamba_model_root = os.path.join(models_root, "mamba2",
                                            "mamba-codestral-7B-v0.1")

    assert exists(mamba_model_root), f"{mamba_model_root} does not exist!"

    return mamba_model_root


@pytest.fixture(scope="function")
def recurrentgemma_model_root(request):
    "get recurrentgemma model data"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    assert hasattr(request, "param"), "Param is missing!"

    if request.param == "recurrentgemma-2b":
        recurrentgemma_model_root = os.path.join(models_root, "recurrentgemma",
                                                 "recurrentgemma-2b")
    elif request.param == "recurrentgemma-2b-it":
        recurrentgemma_model_root = os.path.join(models_root, "recurrentgemma",
                                                 "recurrentgemma-2b-it")
    elif request.param == "recurrentgemma-2b-flax":
        recurrentgemma_model_root = os.path.join(models_root, "recurrentgemma",
                                                 "recurrentgemma-2b-flax", "2b")
    elif request.param == "recurrentgemma-2b-it-flax":
        recurrentgemma_model_root = os.path.join(models_root, "recurrentgemma",
                                                 "recurrentgemma-2b-it-flax",
                                                 "2b-it")

    assert exists(recurrentgemma_model_root
                  ), f"{recurrentgemma_model_root} does not exist!"

    return recurrentgemma_model_root


@pytest.fixture(scope="function")
def nemotron_nas_model_root(request):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    assert hasattr(request, "param"), "Param is missing!"

    nemotron_nas_model_root = os.path.join(models_root, "nemotron-nas",
                                           request.param)

    assert exists(
        nemotron_nas_model_root), f"{nemotron_nas_model_root} doesn't exist!"

    return nemotron_nas_model_root


@pytest.fixture(scope="function")
def llm_lora_model_root(request):
    "get lora model path"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    assert hasattr(request, "param"), "Param is missing!"
    model_list = []
    model_root_list = []
    if isinstance(request.param, tuple):
        model_list = list(request.param)
    else:
        model_list = [request.param]

    for item in model_list:
        if item == "chinese-llama-2-lora-13b":
            model_root_list.append(
                os.path.join(models_root, "llama-models-v2",
                             "chinese-llama-2-lora-13b"))
        elif item == "Japanese-Alpaca-LoRA-7b-v0":
            model_root_list.append(
                os.path.join(models_root, "llama-models",
                             "Japanese-Alpaca-LoRA-7b-v0"))
        elif item == "luotuo-lora-7b-0.1":
            model_root_list.append(
                os.path.join(models_root, "llama-models", "luotuo-lora-7b-0.1"))
        elif item == "Ko-QWEN-7B-Chat-LoRA":
            model_root_list.append(
                os.path.join(models_root, "Ko-QWEN-7B-Chat-LoRA"))
        elif item == "Qwen1.5-7B-Chat-750Mb-lora":
            model_root_list.append(
                os.path.join(models_root, "Qwen1.5-7B-Chat-750Mb-lora"))
        elif item == "Upcycled-Qwen1.5-MoE2.7B-LoRA":
            model_root_list.append(
                os.path.join(models_root, "Upcycled-Qwen1.5-MoE2.7B-LoRA"))
        elif item == "Phi-3-mini-4k-instruct-ru-lora":
            model_root_list.append(
                os.path.join(models_root, "lora", "phi",
                             "Phi-3-mini-4k-instruct-ru-lora"))
        elif item == "peft-lora-starcoder2-15b-unity-copilot":
            model_root_list.append(
                os.path.join(
                    models_root,
                    "lora",
                    "starcoder",
                    "peft-lora-starcoder2-15b-unity-copilot",
                ))
        elif item == "chinese-mixtral-lora":
            model_root_list.append(
                os.path.join(models_root, "chinese-mixtral-lora"))
        elif item == "komt-mistral-7b-v1-lora":
            model_root_list.append(
                os.path.join(models_root, "komt-mistral-7b-v1-lora"))
        elif item == "Llama-3_3-Nemotron-Super-49B-v1-lora-adapter_NIM_r32":
            model_root_list.append(
                os.path.join(
                    models_root, "nemotron-nas",
                    "Llama-3_3-Nemotron-Super-49B-v1-lora-adapter_NIM_r32"))

    return ",".join(model_root_list)


@pytest.fixture(scope="function")
def llm_dora_model_root(request):
    "get dora model path"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    assert hasattr(request, "param"), "Param is missing!"
    model_list = []
    model_root_list = []
    if isinstance(request.param, tuple):
        model_list = list(request.param)
    else:
        model_list = [request.param]

    for item in model_list:
        if item == "commonsense-llama-v3-8b-dora-r32":
            model_root_list.append(
                os.path.join(
                    models_root,
                    "llama-models-v3",
                    "DoRA-weights",
                    "llama_dora_commonsense_checkpoints",
                    "LLama3-8B",
                    "dora_r32",
                ))

    return ",".join(model_root_list)


@pytest.fixture(scope="function")
def llm_mistral_model_root(request):
    "get mistral model path"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    model_root = os.path.join(models_root, "mistral-7b-v0.1")
    if request.param == "mistral-7b-v0.1":
        model_root = os.path.join(models_root, "mistral-7b-v0.1")
    if request.param == "mistral-nemo-instruct-2407":
        model_root = os.path.join(models_root, "Mistral-Nemo-Instruct-2407")
    if request.param == "komt-mistral-7b-v1":
        model_root = os.path.join(models_root, "komt-mistral-7b-v1")
    if request.param == "mistral-7b-v0.3":
        model_root = os.path.join(models_root, "Mistral-7B-Instruct-v0.3")

    return model_root


@pytest.fixture(scope="function")
def llm_mixtral_model_root(request):
    "get mixtral model path"
    models_root = llm_models_root()
    model_root = os.path.join(models_root, "Mixtral-8x7B-v0.1")
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    if request.param == "Mixtral-8x7B-v0.1":
        model_root = os.path.join(models_root, "Mixtral-8x7B-v0.1")
    if request.param == "Mixtral-8x22B-v0.1":
        model_root = os.path.join(models_root, "Mixtral-8x22B-v0.1")
    if request.param == "Mixtral-8x7B-Instruct-v0.1":
        model_root = os.path.join(models_root, "Mixtral-8x7B-Instruct-v0.1")

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("mathstral-7B-v0.1", True)
def llm_mathstral_model_root(llm_venv):
    "return mathstral-7B-v0.1 model root"

    workspace = llm_venv.get_working_directory()
    long_mathstral_model_root = os.path.join(workspace, "mathstral-7B-v0.1")

    return long_mathstral_model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("LongAlpaca-7B", True)
def llm_long_alpaca_model_root(llm_venv):
    "return long alpaca model root"

    workspace = llm_venv.get_working_directory()
    long_alpaca_model_root = os.path.join(workspace, "LongAlpaca-7B")

    return long_alpaca_model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("gpt-neox-20b", True)
def llm_gptneox_model_root(llm_venv):
    "return gptneox model root"

    workspace = llm_venv.get_working_directory()
    gptneox_model_root = os.path.join(workspace, "gpt-neox-20b")

    return gptneox_model_root


@pytest.fixture(scope="function")
def llm_phi_model_root(request):
    "return phi model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    if "Phi-3.5" in request.param:
        phi_model_root = os.path.join(models_root, "Phi-3.5/" + request.param)
    elif "Phi-3" in request.param:
        phi_model_root = os.path.join(models_root, "Phi-3/" + request.param)
    else:
        phi_model_root = os.path.join(models_root, request.param)

    assert os.path.exists(
        phi_model_root
    ), f"{phi_model_root} does not exist under NFS LLM_MODELS_ROOT dir"

    return phi_model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("falcon-180b", True)
def llm_falcon_180b_model_root():
    "prepare falcon 180b model & return falcon model root"
    raise RuntimeError("falcon 180b must be cached")


@pytest.fixture(scope="module")
@cached_in_llm_models_root("falcon-11B", True)
def llm_falcon_11b_model_root(llm_venv):
    "prepare falcon-11B model & return falcon model root"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "falcon-11B")

    call(f"git clone https://huggingface.co/tiiuae/falcon-11B {model_root}",
         shell=True)

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("email_composition", True)
def llm_gpt2_next_8b_model_root():
    raise RuntimeError("gpt-next 8b must be cached")


@pytest.fixture(scope="function")
def llm_glm_4_9b_model_root(request):
    "prepare glm-4-9b model & return model path"
    model_name = request.param
    models_root = llm_models_root()
    if model_name == "glm-4-9b":
        model_root = os.path.join(models_root, "glm-4-9b")
    elif model_name == "glm-4-9b-chat":
        model_root = os.path.join(models_root, "glm-4-9b-chat")
    elif model_name == "glm-4-9b-chat-1m":
        model_root = os.path.join(models_root, "glm-4-9b-chat-1m")
    elif model_name == "glm-4v-9b":
        model_root = os.path.join(models_root, "glm-4v-9b")

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("internlm-chat-7b", True)
def llm_internlm_7b_model_root(llm_venv):
    "prepare internlm 7b model"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "internlm-chat-7b")

    call(
        f"git clone https://huggingface.co/internlm/internlm-chat-7b {model_root}",
        shell=True,
    )

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("internlm2-7b", True)
def llm_internlm2_7b_model_root(llm_venv):
    "prepare internlm2 7b model"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "internlm2-7b")

    call(
        f"git clone https://huggingface.co/internlm/internlm2-7b {model_root}",
        shell=True,
    )

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("internlm-chat-20b", True)
def llm_internlm_20b_model_root(llm_venv):
    "prepare internlm 20b model"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "internlm-chat-20b")

    call(
        f"git clone https://huggingface.co/internlm/internlm-chat-20b {model_root}",
        shell=True,
    )

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("Qwen-7B-Chat", True)
def llm_qwen_7b_model_root(llm_venv):
    "prepare qwen-7b model & return model path"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "Qwen-7B-Chat")

    return model_root


@pytest.fixture(scope="function")
def llm_qwen_model_root(request, llm_venv):
    "prepare qwen model & return model path"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    qwen_model_root = os.path.join(models_root, "Qwen-7B-Chat")

    if hasattr(request, "param"):
        if request.param == "qwen_7b_chat":
            qwen_model_root = os.path.join(models_root, "Qwen-7B-Chat")
        elif request.param == "qwen_14b_chat":
            qwen_model_root = os.path.join(models_root, "Qwen-14B-Chat")
        elif request.param == "qwen_72b_chat":
            qwen_model_root = os.path.join(models_root, "Qwen-72B-Chat")
        elif request.param == "qwen_7b_chat_int4":
            qwen_model_root = os.path.join(models_root, "Qwen-7B-Chat-Int4")
        elif request.param == "qwen-vl-chat":
            qwen_model_root = os.path.join(models_root, "Qwen-VL-Chat")
        elif request.param == "qwen1.5_7b_chat_awq":
            qwen_model_root = os.path.join(models_root, "Qwen1.5-7B-Chat-AWQ")
        elif request.param == "qwen1.5_0.5b_chat":
            qwen_model_root = os.path.join(models_root, "Qwen1.5-0.5B-Chat")
        elif request.param == "qwen1.5_7b_chat":
            qwen_model_root = os.path.join(models_root, "Qwen1.5-7B-Chat")
        elif request.param == "qwen1.5_14b_chat":
            qwen_model_root = os.path.join(models_root, "Qwen1.5-14B-Chat")
        elif request.param == "qwen1.5_moe_a2.7b_chat":
            qwen_model_root = os.path.join(models_root,
                                           "Qwen1.5-MoE-A2.7B-Chat")
        elif request.param == "qwen1.5_72b_chat":
            qwen_model_root = os.path.join(models_root, "Qwen1.5-72B-Chat")
        elif request.param == "qwen1.5_moe_a2.7b_chat":
            qwen_model_root = os.path.join(models_root,
                                           "Qwen1.5-MoE-A2.7B-Chat")
        elif request.param == "qwen1.5_14b_chat_int4":
            qwen_model_root = os.path.join(models_root,
                                           "Qwen1.5-14B-Chat-GPTQ-Int4")
        elif request.param == "qwen2_0.5b_instruct":
            qwen_model_root = os.path.join(models_root, "Qwen2-0.5B-Instruct")
        elif request.param == "qwen2_7b_instruct":
            qwen_model_root = os.path.join(models_root, "Qwen2-7B-Instruct")
        elif request.param == "qwen2_7b_awq":
            qwen_model_root = os.path.join(models_root, "Qwen2-7B-Instruct-AWQ")
        elif request.param == "qwen2_57b_a14b":
            qwen_model_root = os.path.join(models_root, "Qwen2-57B-A14B")
        elif request.param == "qwen2_72b_instruct":
            qwen_model_root = os.path.join(models_root, "Qwen2-72B-Instruct")
        elif request.param == "qwen2_vl_7b_instruct":
            qwen_model_root = os.path.join(models_root, "Qwen2-VL-7B-Instruct")
        elif request.param == "qwen2_audio_7b_instruct":
            qwen_model_root = os.path.join(models_root,
                                           "Qwen2-Audio-7B-Instruct")
        elif request.param == "qwen2.5_0.5b_instruct":
            qwen_model_root = os.path.join(models_root, "Qwen2.5-0.5B-Instruct")
        elif request.param == "qwen2.5_1.5b_instruct":
            qwen_model_root = os.path.join(models_root, "Qwen2.5-1.5B-Instruct")
        elif request.param == "qwen2.5_7b_instruct":
            qwen_model_root = os.path.join(models_root, "Qwen2.5-7B-Instruct")
        elif request.param == "qwen2.5_14b_instruct_int4":
            qwen_model_root = os.path.join(models_root,
                                           "Qwen2.5-14B-Instruct-GPTQ-Int4")
        elif request.param == "qwen2.5_72b_instruct":
            qwen_model_root = os.path.join(models_root, "Qwen2.5-72B-Instruct")

    assert exists(qwen_model_root), f"{qwen_model_root} does not exist!"

    return qwen_model_root


@pytest.fixture(scope="function")
def llm_granite_model_root(request):
    models_root = llm_models_root()
    model_name = request.param
    granite_model_root = os.path.join(models_root, model_name)
    assert exists(granite_model_root), f"{granite_model_root} does not exist!"
    return granite_model_root


@pytest.fixture(scope="session")
@cached_in_llm_models_root("nemotron/Nemotron-3-8B-Base-4k.nemo", True)
def llm_nemotron_3_8b_model_root():
    "get nemotron/Nemotron-3-8B-Base-4k.nemo"
    raise RuntimeError("nemotron/Nemotron-3-8B-Base-4k.nemo must be cached")


@pytest.fixture(scope="session")
@cached_in_llm_models_root("nemotron/Nemotron-4-15B-Base.nemo", True)
def llm_nemotron_4_15b_model_root():
    "get nemotron/Nemotron-4-15B-Base.nemo"
    raise RuntimeError("nemotron/Nemotron-4-15B-Base.nemo must be cached")


@pytest.fixture(scope="session")
def mmlu_dataset_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    mmlu_dataset_root = os.path.join(models_root, "datasets", "mmlu")

    assert os.path.exists(
        mmlu_dataset_root
    ), f"{mmlu_dataset_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return mmlu_dataset_root


@pytest.fixture(scope="function")
def deepseek_model_root(request):
    "get deepseek model"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    if request.param == "deepseek-coder-6.7b-instruct":
        model_root = os.path.join(models_root, "deepseek-coder-6.7b-instruct")

    return model_root


@pytest.fixture(scope="module")
def llm_commandr_v01_model_root(llm_venv):
    "prepare command-r model & return model path"
    models_root = llm_models_root()
    model_root = os.path.join(models_root, "c4ai-command-r-v01")

    return model_root


@pytest.fixture(scope="module")
def llm_commandr_plus_model_root(llm_venv):
    "prepare command-r-plus model & return model path"
    models_root = llm_models_root()
    model_root = os.path.join(models_root, "c4ai-command-r-plus")

    return model_root


@pytest.fixture(scope="module")
def llm_aya_23_8b_model_root(llm_venv):
    "prepare Aya-23-8B model & return model path"
    models_root = llm_models_root()
    model_root = os.path.join(models_root, "aya-23-8B")

    return model_root


@pytest.fixture(scope="module")
def llm_aya_23_35b_model_root(llm_venv):
    "prepare Aya-23-35B model & return model path"
    models_root = llm_models_root()
    model_root = os.path.join(models_root, "aya-23-35B")

    return model_root


@pytest.fixture(scope="function")
def engine_dir(llm_venv, capfd):
    "Get engine dir"
    engine_path = os.path.join(llm_venv.get_working_directory(), "engines")
    print_storage_usage(llm_venv.get_working_directory(),
                        "before removing existing engines", capfd)
    # clean the engine dir for each case.
    cur_time = time.time()
    expire = time.time() + 60
    while exists(engine_path) and cur_time < expire:
        shutil.rmtree(engine_path, ignore_errors=True)
        time.sleep(2)
        cur_time = time.time()

    print_storage_usage(llm_venv.get_working_directory(),
                        "after removing existing engines", capfd)
    return engine_path


@pytest.fixture(scope="function")
def cmodel_dir(llm_venv):
    "converted model dir"
    model_dir = os.path.join(llm_venv.get_working_directory(), "cmodels")

    yield model_dir

    if exists(model_dir):
        shutil.rmtree(model_dir)


@pytest.fixture(scope="function")
def cmodel_base_dir(llm_venv):
    "converted base model dir for redrafter"
    model_dir = os.path.join(llm_venv.get_working_directory(), "cmodels_base")

    yield model_dir

    if exists(model_dir):
        shutil.rmtree(model_dir)


@pytest.fixture(scope="module")
def qcache_dir(llm_venv, llm_root):
    "get quantization cache dir"
    defs.ci_profiler.start("qcache_dir")

    cache_dir = os.path.join(llm_venv.get_working_directory(), "qcache")

    quantization_root = os.path.join(llm_root, "examples", "quantization")

    # Fix the issue that the requirements.txt is not available on aarch64.
    if "aarch64" not in platform.machine() and get_sm_version() >= 89:
        llm_venv.run_cmd([
            "-m",
            "pip",
            "install",
            "-r",
            os.path.join(quantization_root, "requirements.txt"),
        ])

    if not exists(cache_dir):
        makedirs(cache_dir)

    yield cache_dir

    if exists(cache_dir):
        shutil.rmtree(cache_dir)

    defs.ci_profiler.stop("qcache_dir")
    print(
        f"qcache_dir: {defs.ci_profiler.elapsed_time_in_sec('qcache_dir')} sec")


@pytest.fixture(scope="module")
def qcache_dir_without_install_package(llm_venv, llm_root):
    "get quantization cache dir"
    defs.ci_profiler.start("qcache_dir_without_install_package")

    cache_dir = os.path.join(llm_venv.get_working_directory(), "qcache")

    if not exists(cache_dir):
        makedirs(cache_dir)

    yield cache_dir

    if exists(cache_dir):
        shutil.rmtree(cache_dir)

    defs.ci_profiler.stop("qcache_dir_without_install_package")
    print(
        f"qcache_dir_without_install_package: {defs.ci_profiler.elapsed_time_in_sec('qcache_dir_without_install_package')} sec"
    )


@pytest.fixture(scope="module")
def star_attention_input_root(llm_root):
    "Get star attention input file dir"
    star_attention_input_root = unittest_path() / "_torch" / "multi_gpu"

    return star_attention_input_root


def parametrize_with_ids(
    argnames: str | Sequence[str],
    argvalues: Iterable[ParameterSet | Sequence[object] | object],
    **kwargs,
):
    """An alternative to pytest.mark.parametrize with automatically generated test ids."""
    if isinstance(argnames, str):
        argname_list = [n.strip() for n in argnames.split(",")]
    else:
        argname_list = argnames

    case_ids = []
    for case_argvalues in argvalues:
        if isinstance(case_argvalues, ParameterSet):
            case_argvalues = case_argvalues.values
        elif case_argvalues is None or isinstance(case_argvalues,
                                                  (str, float, int, bool)):
            case_argvalues = (case_argvalues, )
        assert len(case_argvalues) == len(argname_list)

        case_id = [
            f"{name}={value}"
            for name, value in zip(argname_list, case_argvalues)
        ]
        case_ids.append("-".join(case_id))

    return pytest.mark.parametrize(argnames, argvalues, ids=case_ids, **kwargs)


@pytest.fixture(autouse=True)
def skip_by_device_count(request):
    "fixture for skip less device count"
    if request.node.get_closest_marker("skip_less_device"):
        device_count = get_device_count()
        expected_count = request.node.get_closest_marker(
            "skip_less_device").args[0]
        if expected_count > int(device_count):
            pytest.skip(
                f"Device count {device_count} is less than {expected_count}")


@pytest.fixture(autouse=True)
def skip_by_mpi_world_size(request):
    "fixture for skip less mpi world size"
    if request.node.get_closest_marker("skip_less_mpi_world_size"):
        mpi_world_size = get_mpi_world_size()
        device_count = get_device_count()
        if mpi_world_size == 1:
            # For mpi_world_size == 1 case, we only need to check device count since we can spawn mpi workers in the test itself
            total_count = device_count
        else:
            # Otherwise, we follow the mpi world size setting
            total_count = mpi_world_size
        expected_count = request.node.get_closest_marker(
            "skip_less_mpi_world_size").args[0]
        if expected_count > int(total_count):
            pytest.skip(
                f"Total world size {total_count} is less than {expected_count}")


@pytest.fixture(autouse=True)
def skip_by_device_memory(request):
    "fixture for skip less device memory"
    if request.node.get_closest_marker("skip_less_device_memory"):
        device_memory = get_device_memory()
        expected_memory = request.node.get_closest_marker(
            "skip_less_device_memory").args[0]
        if expected_memory > int(device_memory):
            pytest.skip(
                f"Device memory {device_memory} is less than {expected_memory}")


def get_sm_version():
    "get compute capability"
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor


def get_gpu_device_list():
    "get device list"
    with tempfile.TemporaryDirectory() as temp_dirname:
        suffix = ".exe" if is_windows() else ""
        # TODO: Use NRSU because we can't assume nvidia-smi across all platforms.
        cmd = " ".join(["nvidia-smi" + suffix, "-L"])
        output = check_output(cmd, shell=True, cwd=temp_dirname)
    return [l.strip() for l in output.strip().split("\n")]


def check_device_contain(keyword_list):
    "check device not contain keyword"
    device = get_gpu_device_list()[0]
    return any(keyword in device for keyword in keyword_list)


skip_pre_ada = pytest.mark.skipif(
    get_sm_version() < 89,
    reason="This test is not supported in pre-Ada architecture")

skip_pre_hopper = pytest.mark.skipif(
    get_sm_version() < 90,
    reason="This test is not supported in pre-Hopper architecture",
)

skip_pre_blackwell = pytest.mark.skipif(
    get_sm_version() < 100,
    reason="This test is not supported in pre-Blackwell architecture",
)

skip_post_blackwell = pytest.mark.skipif(
    get_sm_version() >= 100,
    reason="This test is not supported in post-Blackwell architecture",
)

skip_post_blackwell_ultra = pytest.mark.skipif(
    get_sm_version() >= 103,
    reason="This test is not supported in post-Blackwell-Ultra architecture",
)

skip_device_contain_gb200 = pytest.mark.skipif(
    check_device_contain(["GB200"]),
    reason="This test is not supported on GB200 or GB100",
)

skip_no_nvls = pytest.mark.skipif(not ipc_nvls_supported(),
                                  reason="NVLS is not supported")
skip_no_hopper = pytest.mark.skipif(
    get_sm_version() != 90,
    reason="This test is only  supported in Hopper architecture")

skip_no_sm120 = pytest.mark.skipif(get_sm_version() != 120,
                                   reason="This test is for SM120")

skip_arm = pytest.mark.skipif(
    "aarch64" in platform.machine(),
    reason="This test is not supported on ARM architecture",
)


def skip_fp8_pre_ada(use_fp8):
    "skip fp8 tests if sm version less than 8.9"
    if use_fp8 and get_sm_version() < 89:
        pytest.skip("FP8 is not supported on pre-Ada architectures")


def skip_fp4_pre_blackwell(use_fp4):
    "skip fp4 tests if sm version less than 10.0 or greater or equal to 12.0"
    if use_fp4 and (get_sm_version() < 100 or get_sm_version() >= 120):
        pytest.skip("FP4 is not supported on pre-Blackwell architectures")


@pytest.fixture(autouse=True)
def skip_device_not_contain(request):
    "skip test if device not contain keyword"
    if request.node.get_closest_marker("skip_device_not_contain"):
        keyword_list = request.node.get_closest_marker(
            "skip_device_not_contain").args[0]
        if not check_device_contain(keyword_list):
            pytest.skip(
                f"Device {get_gpu_device_list()[0]} does not contain keyword in {keyword_list}."
            )


def get_device_count():
    "return device count"
    return len(get_gpu_device_list())


def get_device_memory():
    "get gpu memory"
    memory = 0
    with tempfile.TemporaryDirectory() as temp_dirname:
        suffix = ".exe" if is_windows() else ""
        # TODO: Use NRSU because we can't assume nvidia-smi across all platforms.
        cmd = " ".join([
            "nvidia-smi" + suffix, "--query-gpu=memory.total",
            "--format=csv,noheader"
        ])
        # Try to get memory from nvidia-smi first, if failed, fallback to system memory from /proc/meminfo
        # This fallback is needed for systems with unified memory (e.g. DGX Spark)
        try:
            output = check_output(cmd, shell=True, cwd=temp_dirname)
            memory_str = output.strip().split()[0]
            # Check if nvidia-smi returned a valid numeric value
            if "N/A" in memory_str:
                raise ValueError("nvidia-smi returned invalid memory info")
            memory = int(memory_str)
        except (sp.CalledProcessError, ValueError, IndexError):
            # Fallback to system memory from /proc/meminfo (in kB, convert to MiB)
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            memory = int(
                                line.split()[1]) // 1024  # Convert kB to MiB
                            break
            except:
                memory = 8192  # Default 8GB if all else fails

    return memory


def pytest_addoption(parser):
    parser.addoption(
        "--test-list",
        "-F",
        action="store",
        default=None,
        help="Path to the file containing the list of tests to run",
    )
    parser.addoption(
        "--workspace",
        "--ws",
        action="store",
        default=None,
        help="Workspace path to store temp data generated during the tests",
    )
    parser.addoption(
        "--waives-file",
        "-S",
        action="store",
        default=None,
        help=
        "Specify a file containing a list of waives, one per line. After filtering collected tests, Pytest will "
        "apply the waive state specified by this file to the set of tests to be run.",
    )
    parser.addoption(
        "--output-dir",
        "-O",
        action="store",
        default=None,
        help=
        "Directory to store test output. Should point to a new or existing empty directory.",
    )
    parser.addoption(
        "--test-prefix",
        "-P",
        action="store",
        default=None,
        help=
        "It is useful when using such prefix to mapping waive lists for specific GPU, such as 'GH200'",
    )
    parser.addoption(
        "--regexp",
        "-R",
        action="store",
        default=None,
        help="A regexp to specify which tests to run",
    )
    parser.addoption(
        "--apply-test-list-correction",
        "-C",
        action="store_true",
        help=
        "Attempt to automatically correct invalid test names in filter files and print the correct name in terminal. "
        "If the correct name cannot be determined, the invalid test name will be printed to the terminal as well.",
    )
    parser.addoption("--perf",
                     action="store_true",
                     help="'--perf' will run perf tests")
    parser.addoption(
        "--run-ray",
        action="store_true",
        default=False,
        help=
        "Enable Ray orchestrator path for integration tests (disables MPI).",
    )
    parser.addoption(
        "--perf-log-formats",
        help=
        "Supply either 'yaml' or 'csv' as values. Supply multiple same flags for multiple formats.",
        action="append",
        default=[],
    )
    parser.addoption(
        "--test-model-suites",
        action="store",
        default=None,
        help=
        "Specify test model suites separated by semicolons or spaces. Each suite can contain special characters. "
        "Example: --test-model-suites=suite1;suite2;suite3 or --test-model-suites=suite1 suite2 suite3",
    )
    parser.addoption(
        "--periodic-junit",
        action="store_true",
        default=False,
        help=
        "Enable periodic JUnit XML reporter. This reporter leverages pytest's built-in junitxml "
        "for reliable test result handling. Saves progress periodically to prevent data loss on "
        "interruption. Requires --output-dir to be set.",
    )
    parser.addoption(
        "--periodic-interval",
        action="store",
        type=int,
        default=18000,
        help=
        "Time interval in seconds between periodic saves (default: 18000s = 5 hours). "
        "Only used with --periodic-junit.",
    )
    parser.addoption(
        "--periodic-batch-size",
        action="store",
        type=int,
        default=10,
        help=
        "Number of completed tests before triggering a periodic save (default: 10). "
        "Only used with --periodic-junit.",
    )


@pytest.hookimpl(trylast=True)
def pytest_generate_tests(metafunc: pytest.Metafunc):
    if metafunc.definition.function.__name__ != "test_unittests_v2":
        return
    testlist_path = metafunc.config.getoption("--test-list")
    if not testlist_path:
        return

    with open(testlist_path, "r") as f:
        lines = f.readlines()
        lines = preprocess_test_list_lines(testlist_path, lines)

    uts = []
    ids = []
    for line in lines:
        if line.startswith("unittest/"):
            if " TIMEOUT " in line:
                # Process for marker TIMEOUT
                case_part, timeout_part = line.split(" TIMEOUT ", 1)
                case = case_part.strip()
                timeout_str = timeout_part.strip()
                timeout_num_match = re.search(r"\(?(\d+)\)?", timeout_str)
                if timeout_num_match:
                    timeout_min = int(timeout_num_match.group(1))
                    timeout_sec = timeout_min * 60
                else:
                    raise ValueError(
                        f"Invalid TIMEOUT format: {timeout_str} in line: {line}"
                    )
                mark = pytest.mark.timeout(int(timeout_sec))
                uts.append(pytest.param(case, marks=mark))
                # Change back id to include timeout information
                ids.append(f"{case} TIMEOUT {timeout_str}")
            else:
                uts.append(line.strip())
    metafunc.parametrize("case", uts, ids=lambda x: x)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_collection_modifyitems(session, config, items):
    testlist_path = config.getoption("--test-list")
    waives_file = config.getoption("--waives-file")
    test_prefix = config.getoption("--test-prefix")
    perf_test = config.getoption("--perf")
    test_model_suites = config.getoption("--test-model-suites")

    if perf_test:
        global ALL_PYTEST_ITEMS
        ALL_PYTEST_ITEMS = None

        import copy

        # Do not import at global level since that would create cyclic imports.
        from .perf.test_perf import generate_perf_tests

        # Perf tests are generated based on the test list to speed up the test collection time.
        items = generate_perf_tests(session, config, items)

        ALL_PYTEST_ITEMS = copy.copy(items)

    if test_prefix:
        # Override the internal nodeid of each item to contain the correct test prefix.
        # This is needed for reporting to correctly process the test name in order to bucket
        # it into the appropriate test suite.
        for item in items:
            item._nodeid = "{}/{}".format(test_prefix, item._nodeid)

    regexp = config.getoption("--regexp")

    if testlist_path:
        modify_by_test_list(testlist_path, items, config)

    if regexp is not None:
        deselect_by_regex(regexp, items, test_prefix, config)

    if test_model_suites:
        deselect_by_test_model_suites(test_model_suites, items, test_prefix,
                                      config)

    if waives_file:
        apply_waives(waives_file, items, config)

    # We have to remove prefix temporarily before splitting the test list
    # After that change back the test id.
    for item in items:
        if test_prefix and item._nodeid.startswith(f"{test_prefix}/"):
            item._nodeid = item._nodeid[len(f"{test_prefix}/"):]
    yield
    for item in items:
        if test_prefix:
            item._nodeid = f"{test_prefix}/{item._nodeid}"


def pytest_configure(config):
    # avoid thread leak of tqdm's TMonitor
    tqdm.tqdm.monitor_interval = 0
    if config.getoption("--run-ray"):
        os.environ["TLLM_DISABLE_MPI"] = "1"

    # Initialize PeriodicJUnitXML reporter if enabled
    periodic = config.getoption("--periodic-junit", default=False)
    output_dir = config.getoption("--output-dir", default=None)

    if periodic and output_dir:
        periodic_interval = config.getoption("--periodic-interval")
        periodic_batch_size = config.getoption("--periodic-batch-size")

        # Create output directory early (like --junitxml does) to avoid conflicts with other plugins
        # that may need to write to the same directory (e.g., pytest-split)
        os.makedirs(output_dir, exist_ok=True)

        # Create the reporter with logger
        xmlpath = os.path.join(output_dir, "results.xml")
        reporter = PeriodicJUnitXML(
            xmlpath=xmlpath,
            interval=periodic_interval,
            batch_size=periodic_batch_size,
            logger={
                'info': print_info,
                'warning': print_warning
            },
        )

        # Configure and register the reporter
        reporter.pytest_configure(config)
        config.pluginmanager.register(reporter, 'periodic_junit')

        print_info("PeriodicJUnitXML reporter registered")
        print_info(
            f"  Interval: {periodic_interval}s ({periodic_interval/60:.1f} min)"
        )
        print_info(f"  Batch size: {periodic_batch_size} tests")
    elif periodic and not output_dir:
        print_warning(
            "Warning: --periodic-junit requires --output-dir to be set. "
            "Periodic reporting disabled.")


def deselect_by_test_model_suites(test_model_suites, items, test_prefix,
                                  config):
    """Filter tests based on the test model suites specified.
    If a test matches any of the test model suite names, it is considered selected.

    Args:
        test_model_suites: String containing test model suite names separated by semicolons
        items: List of pytest items to filter
        test_prefix: Test prefix if any
        config: Pytest config object
    """
    if not test_model_suites:
        return

    # Split by semicolon or space and strip whitespace
    suite_names = [
        suite.strip() for suite in test_model_suites.replace(';', ' ').split()
        if suite.strip()
    ]

    if not suite_names:
        return

    selected = []
    deselected = []

    for item in items:
        # Get the test name without prefix for comparison
        test_name = item.nodeid
        if test_prefix and test_name.startswith(f"{test_prefix}/"):
            test_name = test_name[len(f"{test_prefix}/"):]

        # Check if any suite name matches the test name
        found = False
        for suite_name in suite_names:
            if suite_name in test_name or test_name.endswith(suite_name):
                found = True
                break

        if found:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def deselect_by_regex(regexp, items, test_prefix, config):
    """Filter out tests based on the patterns specified in the given list of regular expressions.
    If a test matches *any* of the expressions in the list it is considered selected."""
    compiled_regexes = []
    regex_list = []
    r = re.compile(regexp)
    compiled_regexes.append(r)
    regex_list.append(regexp)

    selected = []
    deselected = []

    corrections = get_test_name_corrections_v2(set(regex_list),
                                               set(it.nodeid for it in items),
                                               TestCorrectionMode.REGEX)
    handle_corrections(corrections, test_prefix)

    for item in items:
        found = False
        for regex in compiled_regexes:
            if regex.search(item.nodeid):
                found = True
                break
        if found:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if call.when == "call":
        report.file = str(item.fspath)
        report.line = str(item.location[1])
        report.url = ""


@pytest.fixture(scope="session")
def all_pytest_items():
    """
    Provides all pytest items available in the current test definitions, before any
    filtering has been applied.
    """
    return ALL_PYTEST_ITEMS


@pytest.fixture(scope="session")
def test_root():
    return os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope="function")
def test_case(request, llm_root):
    "get test case"
    test_cases_file = "tests/integration/defs/test_cases.yml"
    input_file_dir = "tests/integration/test_input_files"
    test_cases_file_path = os.path.join(llm_root, test_cases_file)
    case_name = request.param

    with open(test_cases_file_path, "r", encoding="UTF-8") as file:
        test_cases = yaml.safe_load(file)

    case = test_cases["test_cases"][case_name]
    input_file = case["input_file"]

    case["input_file"] = os.path.join(llm_root, input_file_dir, input_file)

    return case


def check_nvlink():
    "check nvlink status"
    with tempfile.TemporaryDirectory() as temp_dirname:
        try:
            suffix = ".exe" if is_windows() else ""
            # TODO: Use NRSU because we can't assume nvidia-smi across all platforms.
            cmd = " ".join(["nvidia-smi" + suffix, "nvlink", "-s", "-i", "0"])
            output = check_output(cmd, shell=True, cwd=temp_dirname)
        except sp.CalledProcessError:
            return False

    if len(output.strip()) == 0:
        return False

    return "inActive" not in output.strip()


skip_nvlink_inactive = pytest.mark.skipif(check_nvlink() is False,
                                          reason="nvlink is inactive.")

skip_ray = pytest.mark.skipif(
    os.environ.get("TLLM_DISABLE_MPI") == "1",
    reason="This test is skipped for Ray orchestrator.")


@pytest.fixture(scope="function")
def eval_venv(llm_venv):
    "set UCC_TEAM_IDS_POOL_SIZE=1024"

    llm_venv._new_env["UCC_TEAM_IDS_POOL_SIZE"] = "1024"

    yield llm_venv

    llm_venv._new_env.pop("UCC_TEAM_IDS_POOL_SIZE")


def get_host_total_memory():
    "get host memory Mib"
    memory = psutil.virtual_memory().total

    return int(memory / 1024 / 1024)


@pytest.fixture(autouse=True)
def skip_by_host_memory(request):
    "fixture for skip less host memory"
    if request.node.get_closest_marker("skip_less_host_memory"):
        host_memory = get_host_total_memory()
        expected_memory = request.node.get_closest_marker(
            "skip_less_host_memory").args[0]
        if expected_memory > int(host_memory):
            pytest.skip(
                f"Host memory {host_memory} is less than {expected_memory}")


IS_UNDER_CI_ENV = "JENKINS_HOME" in os.environ

gpu_warning_threshold = 1024 * 1024 * 1024


def collect_status(item: pytest.Item):
    if not IS_UNDER_CI_ENV:
        return

    import psutil
    import pynvml

    pynvml.nvmlInit()

    handles = {
        idx: pynvml.nvmlDeviceGetHandleByIndex(idx)
        for idx in range(pynvml.nvmlDeviceGetCount())
    }

    deadline = time.perf_counter() + 60  # 1 min
    observed_used = 0
    global gpu_warning_threshold

    while time.perf_counter() < deadline:
        observed_used = max(
            pynvml.nvmlDeviceGetMemoryInfo(device).used
            for device in handles.values())
        if observed_used <= gpu_warning_threshold:
            break
        time.sleep(1)
    else:
        gpu_warning_threshold = max(observed_used, gpu_warning_threshold)
        warnings.warn(
            f"Test {item.name} does not free up GPU memory correctly!")

    gpu_memory = {}
    for idx, device in handles.items():
        total_used = pynvml.nvmlDeviceGetMemoryInfo(device).used // 1024 // 1024
        total = pynvml.nvmlDeviceGetMemoryInfo(device).total // 1024 // 1024
        detail = pynvml.nvmlDeviceGetComputeRunningProcesses(device)
        process = {}

        for entry in detail:
            try:
                p = psutil.Process(entry.pid)
                host_memory_in_mbs = p.memory_full_info().uss // 1024 // 1024
                process[entry.pid] = (
                    entry.usedGpuMemory // 1024 // 1024,
                    host_memory_in_mbs,
                    p.cmdline(),
                )
            except Exception:
                pass

        gpu_memory[idx] = {
            "total_used": total_used,
            "total": total,
            "process": process
        }
    print("\nCurrent memory status:")
    print(gpu_memory)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    ret = yield
    collect_status(item)
    return ret


@pytest.fixture(scope="function")
def deterministic_test_root(llm_root, llm_venv):
    "Get deterministic test root"
    deterministic_root = os.path.join(llm_root,
                                      "tests/integration/defs/deterministic")

    return deterministic_root


@pytest.fixture(scope="function")
def disaggregated_test_root(llm_root, llm_venv):
    "Get disaggregated test root"
    disaggregated_root = os.path.join(llm_root,
                                      "tests/integration/defs/disaggregated")

    return disaggregated_root


@pytest.fixture(scope="function")
def serve_test_root(llm_root):
    "Get servetest root"
    serve_root = os.path.join(llm_root, "tests/integration/defs/examples/serve")

    return serve_root


@pytest.fixture(scope="function")
def tritonserver_test_root(llm_root):
    "Get tritonserver test root"
    tritonserver_root = os.path.join(llm_root,
                                     "tests/integration/defs/triton_server")

    return tritonserver_root


@pytest.fixture
def timeout_from_marker(request):
    """Get timeout value from pytest timeout marker."""
    timeout_marker = request.node.get_closest_marker("timeout")
    if timeout_marker:
        return timeout_marker.args[0] if timeout_marker.args else None
    return None


@pytest.fixture
def timeout_from_command_line(request):
    """Get timeout value from command line --timeout parameter."""
    # Get timeout from command line argument
    timeout_arg = request.config.getoption("--timeout", default=None)
    if timeout_arg is not None:
        return float(timeout_arg)
    return None


@pytest.fixture
def timeout_manager(timeout_from_command_line, timeout_from_marker):
    """Create a TimeoutManager instance with priority: marker > cmdline > config."""
    from defs.utils.timeout_manager import TimeoutManager

    # Priority: marker > command line
    timeout_value = None

    if timeout_from_marker is not None:
        timeout_value = timeout_from_marker
    elif timeout_from_command_line is not None:
        timeout_value = timeout_from_command_line

    return TimeoutManager(timeout_value)


@pytest.fixture(autouse=True)
def torch_empty_cache() -> None:
    """
    Manually empty the torch CUDA cache before each test, to reduce risk of OOM errors.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


@pytest.fixture(autouse=True)
def ray_cleanup(llm_venv) -> None:
    yield

    if mpi_disabled():
        llm_venv.run_cmd([
            "-m",
            "ray.scripts.scripts",
            "stop",
        ])
