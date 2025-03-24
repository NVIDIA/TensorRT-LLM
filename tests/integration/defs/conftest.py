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
import json
import os
import re
import shutil
import subprocess as sp
import tempfile
import time
import urllib.request
from functools import wraps
from pathlib import Path

import defs.ci_profiler
import psutil
import pytest
import yaml

from .perf.gpu_clock_lock import GPUClockLock
from .perf.session_data_writer import SessionDataWriter
from .test_list_parser import (TestCorrectionMode, apply_waives,
                               get_test_name_corrections_v2, handle_corrections,
                               modify_by_test_list, preprocess_test_list_lines)
from .trt_test_alternative import (call, check_output, exists, is_windows,
                                   is_wsl, makedirs, print_info, print_warning,
                                   wsl_to_win_path)

try:
    from llm import trt_environment
except ImportError:
    trt_environment = None

# TODO: turn off this when the nightly storage issue is resolved.
DEBUG_CI_STORAGE = os.environ.get("DEBUG_CI_STORAGE", False)
GITLAB_API_USER = os.environ.get("GITLAB_API_USER")
GITLAB_API_TOKEN = os.environ.get("GITLAB_API_TOKEN")
EVALTOOL_REPO_URL = os.environ.get("EVALTOOL_REPO_URL")
LLM_GATE_WAY_CLIENT_ID = os.environ.get("LLM_GATE_WAY_CLIENT_ID")
LLM_GATE_WAY_TOKEN = os.environ.get("LLM_GATE_WAY_TOKEN")


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
    '''return LLM_MODELS_ROOT path if it is set in env, assert when it's set but not a valid path
    '''
    DEFAULT_LLM_MODEL_ROOT = os.path.join("/scratch.trt_llm_data", "llm-models")
    LLM_MODELS_ROOT = os.environ.get("LLM_MODELS_ROOT", DEFAULT_LLM_MODEL_ROOT)

    return LLM_MODELS_ROOT


def tests_path() -> Path:
    return (Path(os.path.dirname(__file__)) / "../..").resolve()


def unittest_path() -> Path:
    return tests_path() / "unittest"


def integration_path() -> Path:
    return tests_path() / "integration"


def cached_in_llm_models_root(path_relative_to_llm_models_root,
                              fail_if_path_is_invalid=False):
    '''
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
    '''

    def wrapper(f):

        @wraps(f)
        def decorated(*args, **kwargs):
            if llm_models_root() is not None:
                cached_dir = f"{llm_models_root()}/{path_relative_to_llm_models_root}"
                if os.path.exists(cached_dir):
                    return cached_dir
                elif fail_if_path_is_invalid:
                    assert False, f"{cached_dir} does not exist, and fail_if_path_is_invalid is True, please check the cache directory"
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
def llm_datasets_root() -> str:
    return os.path.join(llm_models_root(), "datasets")


@pytest.fixture(scope="session")
def llm_rouge_root() -> str:
    return os.path.join(llm_models_root(), "rouge")


@pytest.fixture(scope="module")
def bert_example_root(llm_root):
    "Get bert example root"
    example_root = os.path.join(llm_root, "examples", "bert")

    return example_root


@pytest.fixture(scope="module")
def enc_dec_example_root(llm_root):
    "Get encoder-decoder example root"
    example_root = os.path.join(llm_root, "examples", "enc_dec")

    return example_root


@pytest.fixture(scope="module")
def whisper_example_root(llm_root, llm_venv):
    "Get whisper example root"
    example_root = os.path.join(llm_root, "examples", "whisper")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])
    return example_root


@pytest.fixture(scope="module")
def opt_example_root(llm_root, llm_venv):
    "Get opt example root"

    example_root = os.path.join(llm_root, "examples", "opt")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def llama_example_root(llm_root, llm_venv):
    "Get llama example root"

    example_root = os.path.join(llm_root, "examples", "llama")
    try:
        llm_venv.run_cmd([
            "-m", "pip", "install", "-r",
            os.path.join(example_root, "requirements.txt")
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

    example_root = os.path.join(llm_root, "examples", "gemma")
    # https://nvbugs/4559583 Jax dependency broke the entire pipeline in TRT container
    # due to the dependency incompatibility with torch, which forced reinstall everything
    # and caused pipeline to fail. We manually install gemma dependency as a WAR.
    llm_venv.run_cmd(["-m", "pip", "install", "safetensors~=0.4.1", "nltk"])
    # Install Jax because it breaks dependency
    import platform
    google_extension = [
        "-f",
        "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    ]

    # WAR the new posting of "nvidia-cudnn-cu12~=9.0".
    # "jax[cuda12_pip]~=0.4.19" specifies "nvidia-cudnn-cu12>=8.9" but actually requires "nvidia-cudnn-cu12~=8.9".
    if "x86_64" in platform.machine():
        llm_venv.run_cmd(["-m", "pip", "install", "nvidia-cudnn-cu12~=8.9"])

    if "Windows" in platform.system():
        llm_venv.run_cmd([
            "-m", "pip", "install", "jax~=0.4.19", "jaxlib~=0.4.19", "--no-deps"
        ] + google_extension)
    else:
        llm_venv.run_cmd([
            "-m", "pip", "install", "jax[cuda12_pip]~=0.4.19",
            "jaxlib[cuda12_pip]~=0.4.19", "--no-deps"
        ] + google_extension)
    llm_venv.run_cmd(["-m", "pip", "install", "flax~=0.8.0"])
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
    example_root = os.path.join(llm_root, "examples", "gpt")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def mpt_example_root(llm_root, llm_venv):
    "Get mpt example root"

    example_root = os.path.join(llm_root, "examples", "mpt")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def gptj_example_root(llm_root, llm_venv):
    "Get gptj example root"
    example_root = os.path.join(llm_root, "examples", "gptj")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def gptneox_example_root(llm_root, llm_venv):
    "Get gptneox example root"
    example_root = os.path.join(llm_root, "examples", "gptneox")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def bloom_example_root(llm_root, llm_venv):
    "Get bloom example root"
    example_root = os.path.join(llm_root, "examples", "bloom")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def chatglm_6b_example_root(llm_root, llm_venv):
    "Get chatglm-6b example root"
    example_root = os.path.join(llm_root, "examples", "chatglm")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def chatglm2_6b_example_root(llm_root, llm_venv):
    "Get chatglm2-6b example root"
    example_root = os.path.join(llm_root, "examples", "chatglm")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def chatglm3_6b_example_root(llm_root, llm_venv):
    "Get chatglm3-6b example root"
    example_root = os.path.join(llm_root, "examples", "chatglm")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def glm_10b_example_root(llm_root, llm_venv):
    "Get glm-10b example root"
    example_root = os.path.join(llm_root, "examples", "chatglm")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def glm_4_9b_example_root(llm_root, llm_venv):
    "Get glm-4-9b example root"
    example_root = os.path.join(llm_root, "examples", "chatglm")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def exaone_example_root(llm_root, llm_venv):
    "Get EXAONE example root"
    example_root = os.path.join(llm_root, "examples", "exaone")

    return example_root


@pytest.fixture(scope="function")
def llm_exaone_model_root(request) -> str:
    "Get EXAONE model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    assert request.param == "exaone", "Is the name of model root is exaone?"

    exaone_model_root = os.path.join(models_root, request.param)
    assert exists(exaone_model_root), f"{exaone_model_root} does not exist!"

    return exaone_model_root


@pytest.fixture(scope="module")
def falcon_example_root(llm_root, llm_venv):
    "Get falcon example root"
    example_root = os.path.join(llm_root, "examples", "falcon")
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
def baichuan_example_root(llm_root, llm_venv):
    "Get baichuan example root"
    example_root = os.path.join(llm_root, "examples", "baichuan")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def internlm_example_root(llm_root, llm_venv):
    "Get internlm example root"
    example_root = os.path.join(llm_root, "examples", "internlm")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def internlm2_example_root(llm_root, llm_venv):
    "Get internlm2 example root"
    example_root = os.path.join(llm_root, "examples", "internlm2")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def qwen_example_root(llm_root, llm_venv):
    "Get qwen example root"
    example_root = os.path.join(llm_root, "examples", "qwen")
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
def prompt_lookup_example_root(llm_root, llm_venv):
    "Get Prompt-Lookup example root"
    example_root = os.path.join(llm_root, "examples", "prompt_lookup")
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
def skywork_example_root(llm_root, llm_venv):
    "Get skywork example root"
    example_root = os.path.join(llm_root, "examples", "skywork")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def mamba_example_root(llm_root, llm_venv):
    "Get mamba example root"
    example_root = os.path.join(llm_root, "examples", "mamba")
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
def dbrx_example_root(llm_root, llm_venv):
    "Get dbrx example root"
    example_root = os.path.join(llm_root, "examples", "dbrx")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def jais_example_root(llm_root, llm_venv):
    "Get jais example root"
    example_root = os.path.join(llm_root, "examples", "jais")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def recurrentgemma_example_root(llm_root, llm_venv):
    "Get recurrentgemma example root"
    example_root = os.path.join(llm_root, "examples", "recurrentgemma")

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
    example_root = os.path.join(llm_root, "examples", "nemotron_nas")

    yield example_root


@pytest.fixture(scope="module")
def nemotron_example_root(llm_root, llm_venv):
    "Get nemotron example root"
    example_root = os.path.join(llm_root, "examples", "nemotron")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])
    return example_root


@pytest.fixture(scope="module")
def commandr_example_root(llm_root, llm_venv):
    "Get commandr example root"
    example_root = os.path.join(llm_root, "examples", "commandr")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="module")
def sdxl_example_root(llm_root, llm_venv):
    "Get stable diffusion example root"
    example_root = os.path.join(llm_root, "examples", "sdxl")

    llm_venv.run_cmd(
        ["-m", "pip", "install", "numpy", "pillow", "torchmetrics"])

    return example_root


@pytest.fixture(scope="module")
def deepseek_v2_example_root(llm_root, llm_venv):
    "Get deepseek v2 example root"
    example_root = os.path.join(llm_root, "examples", "deepseek_v2")
    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(example_root, "requirements.txt")
    ])

    return example_root


@pytest.fixture(scope="function")
def deepseek_v3_model_root(request):
    models_root = llm_models_root()
    if (request.param == "DeepSeek-V3"):
        deepseek_v3_model_root = os.path.join(models_root, "DeepSeek-V3")
    elif (request.param == "DeepSeek-V3-Lite-bf16"):
        deepseek_v3_model_root = os.path.join(models_root, "DeepSeek-V3-Lite",
                                              "bf16")
    elif (request.param == "DeepSeek-V3-Lite-fp8"):
        deepseek_v3_model_root = os.path.join(models_root, "DeepSeek-V3-Lite",
                                              "fp8")
    elif (request.param == "DeepSeek-V3-Lite-nvfp4_moe_only"):
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
    return PythonVenvRunnerImpl("", "", "python3",
                                os.path.join(os.getcwd(), workspace_dir))


@pytest.fixture(scope="session")
@cached_in_llm_models_root("gpt-next/megatron_converted_843m_tp1_pp1.nemo",
                           True)
def gpt_next_root():
    "get gpt-next/megatron_converted_843m_tp1_pp1.nemo"
    raise RuntimeError("megatron_converted_843m_tp1_pp1.nemo must be cached")


@pytest.fixture(scope="function")
def llm_opt_model_root(request):
    "Get opt model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    opt_model_root = os.path.join(models_root, "opt-125m")

    if hasattr(request, "param"):
        if request.param == "opt-350m":
            opt_model_root = os.path.join(models_root, "opt-350m")
        elif request.param == "opt-2.7b":
            opt_model_root = os.path.join(models_root, "opt-2.7b")
        elif request.param == "opt-66b":
            opt_model_root = os.path.join(models_root, "opt-66b")
        elif request.param == "opt-125m":
            opt_model_root = os.path.join(models_root, "opt-125m")

    assert exists(opt_model_root), f"{opt_model_root} does not exist!"

    return opt_model_root


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
        "large-v2", "large-v3"
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
    models_root = os.path.join(llm_models_root(), 'multimodals')
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    tllm_model_name = request.param
    if 'VILA' in tllm_model_name:
        models_root = os.path.join(llm_models_root(), 'vila')
    if 'cogvlm-chat' in tllm_model_name:
        models_root = os.path.join(llm_models_root(), 'cogvlm-chat')
    if 'video-neva' in tllm_model_name:
        models_root = os.path.join(llm_models_root(), 'video-neva')
        tllm_model_name = tllm_model_name + ".nemo"
    if 'neva-22b' in tllm_model_name:
        models_root = os.path.join(llm_models_root(), 'neva')
        tllm_model_name = tllm_model_name + ".nemo"
    elif 'Llama-3.2' in tllm_model_name:
        models_root = os.path.join(llm_models_root(), 'llama-3.2-models')

    multimodal_model_root = os.path.join(models_root, tllm_model_name)

    if 'llava-onevision' in tllm_model_name and 'video' in tllm_model_name:
        multimodal_model_root = multimodal_model_root[:-6]
    elif 'llava-v1.6' in tllm_model_name and 'vision-trtllm' in tllm_model_name:
        multimodal_model_root = multimodal_model_root[:-14]

    assert os.path.exists(
        multimodal_model_root
    ), f"{multimodal_model_root} does not exist under NFS LLM_MODELS_ROOT dir"

    yield (tllm_model_name, multimodal_model_root)

    if 'llava-onevision' in tllm_model_name:
        llm_venv.run_cmd(['-m', 'pip', 'uninstall', 'llava', '-y'])


@pytest.fixture(scope="function")
def update_transformers(llm_venv, llm_root):

    yield

    llm_venv.run_cmd([
        "-m", "pip", "install", "-r",
        os.path.join(llm_root, "requirements.txt")
    ])


def remove_file(fn):
    if os.path.isfile(fn) or os.path.islink(fn):
        os.remove(fn)


@pytest.fixture(scope="module")
@cached_in_llm_models_root("mpt-7b", True)
def llm_mpt_7b_model_root():
    "Get mpt model root"
    raise RuntimeError("mpt-7b must be cached")


@pytest.fixture(scope="module")
@cached_in_llm_models_root("mpt-125m", True)
def llm_mpt_125m_model_root():
    "get mpt 125m model path"
    raise RuntimeError("mpt-125m must be cached")


@pytest.fixture(scope="function")
@cached_in_llm_models_root("mpt-30b", True)
def llm_mpt_30b_model_root():
    "get mpt 30b model path"
    raise RuntimeError("mpt-30b must be cached")


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
    elif request.param == "llama-3.1-8b-hf-nvfp4":
        llama_model_root = os.path.join(models_root, "nvfp4-quantized",
                                        "Meta-Llama-3.1-8B")
    elif request.param == "llama-3.1-70b-instruct":
        llama_model_root = os.path.join(models_root, "llama-3.1-model",
                                        "Meta-Llama-3.1-70B-Instruct")
    elif request.param == "llama-3.2-1b":
        llama_model_root = os.path.join(models_root, "llama-3.2-models",
                                        "Llama-3.2-1B")
    elif request.param == "llama-3.2-3b":
        llama_model_root = os.path.join(models_root, "llama-3.2-models",
                                        "Llama-3.2-3B")
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
def prompt_lookup_root(request):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    if request.param == "gpt2":
        models_root = os.path.join(models_root, "gpt2-medium")
    elif request.param == "llama_v2":
        models_root = os.path.join(models_root,
                                   "llama-models-v2/llama-v2-13b-hf")
    assert os.path.exists(
        models_root
    ), f"Prompt-Lookup model path {models_root} does not exist under NFS LLM_MODELS_ROOT dir"
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
def skywork_model_root(request):
    models_root = llm_models_root()

    skywork_model_root = None
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    if request.param == "Skywork-13B-base":
        skywork_model_root = os.path.join(models_root, "Skywork-13B-base")
    elif request.param == "Skywork-13B-Math":
        skywork_model_root = os.path.join(models_root, "Skywork-13B-Math")
    else:
        raise NotImplementedError("The model is not yet supported in Skywork")

    assert os.path.exists(
        skywork_model_root,
    ), f"{skywork_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return skywork_model_root


@pytest.fixture(scope="function")
def mamba_model_root(request):
    "get mamba model data"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    mamba_model_root = os.path.join(models_root, 'mamba', "mamba-130m-hf")
    if hasattr(request, "param"):
        if request.param == "mamba-2.8b":
            mamba_model_root = os.path.join(models_root, 'mamba',
                                            "mamba-2.8b-hf")
        elif request.param == "mamba-130m":
            mamba_model_root = os.path.join(models_root, 'mamba',
                                            "mamba-130m-hf")
        elif request.param == "mamba-1.4b":
            mamba_model_root = os.path.join(models_root, 'mamba',
                                            "mamba-1.4b-hf")
        elif request.param == "mamba-790m":
            mamba_model_root = os.path.join(models_root, 'mamba',
                                            "mamba-790m-hf")
        elif request.param == "mamba-370m":
            mamba_model_root = os.path.join(models_root, 'mamba',
                                            "mamba-370m-hf")
        elif request.param == "mamba2-2.7b":
            mamba_model_root = os.path.join(models_root, 'mamba2',
                                            "mamba2-2.7b")
        elif request.param == "mamba2-1.3b":
            mamba_model_root = os.path.join(models_root, 'mamba2',
                                            "mamba2-1.3b")
        elif request.param == "mamba2-780m":
            mamba_model_root = os.path.join(models_root, 'mamba2',
                                            "mamba2-780m")
        elif request.param == "mamba2-370m":
            mamba_model_root = os.path.join(models_root, 'mamba2',
                                            "mamba2-370m")
        elif request.param == "mamba2-130m":
            mamba_model_root = os.path.join(models_root, 'mamba2',
                                            "mamba2-130m")
        elif request.param == "mamba-codestral-7B-v0.1":
            mamba_model_root = os.path.join(models_root, 'mamba2',
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


@pytest.fixture(scope='module')
def smaug_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    smaug_model_root = os.path.join(models_root, "Smaug-72B-v0.1")
    assert os.path.exists(
        smaug_model_root
    ), f"{smaug_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return smaug_model_root


@pytest.fixture(scope="function")
def jais_model_root(request):
    models_root = llm_models_root()

    jais_model_root = None
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    supported_models = [
        "jais-13b",
        "jais-13b-chat",
        "jais-30b-v1",
        "jais-30b-chat-v1",
        "jais-30b-v3",
        "jais-30b-chat-v3",
    ]
    for model_name in supported_models:
        if request.param == model_name:
            jais_model_root = os.path.join(models_root, model_name)
            break
    else:
        raise NotImplementedError("The model is not yet supported in jais")

    assert os.path.exists(
        jais_model_root,
    ), f"{jais_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return jais_model_root


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
                os.path.join(models_root, "lora", "starcoder",
                             "peft-lora-starcoder2-15b-unity-copilot"))
        elif item == "chinese-mixtral-lora":
            model_root_list.append(
                os.path.join(models_root, "chinese-mixtral-lora"))
        elif item == "komt-mistral-7b-v1-lora":
            model_root_list.append(
                os.path.join(models_root, "komt-mistral-7b-v1-lora"))

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
                os.path.join(models_root, "llama-models-v3", "DoRA-weights",
                             "llama_dora_commonsense_checkpoints", "LLama3-8B",
                             "dora_r32"))

    return ",".join(model_root_list)


@pytest.fixture(scope="function")
def llm_mistral_model_root(request):
    "get mistral model path"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    model_root = os.path.join(models_root, "mistral-7b-v0.1")
    if request.param == "mistral-7b-v0.1":
        model_root = os.path.join(models_root, "mistral-7b-v0.1")
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
@cached_in_llm_models_root("gpt-j-6b", True)
def llm_gptj_model_root(llm_venv):
    "prepare gptj model & return gptj model root"
    workspace = llm_venv.get_working_directory()
    gptj_model_root = os.path.join(workspace, "gptj")
    call(
        f"git clone https://huggingface.co/EleutherAI/gpt-j-6b {gptj_model_root}",
        shell=True)
    remove_file(f"{gptj_model_root}/pytorch_model.bin")
    wget(
        "https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/pytorch_model.bin",
        out=gptj_model_root)
    wget(
        "https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/tokenizer.json",
        out=gptj_model_root)
    wget(
        "https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/tokenizer_config.json",
        out=gptj_model_root)
    wget("https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/vocab.json",
         out=gptj_model_root)
    wget("https://huggingface.co/EleutherAI/gpt-j-6b/resolve/main/merges.txt",
         out=gptj_model_root)

    return gptj_model_root


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

    if 'Phi-3.5' in request.param:
        phi_model_root = os.path.join(models_root, 'Phi-3.5/' + request.param)
    elif 'Phi-3' in request.param:
        phi_model_root = os.path.join(models_root, 'Phi-3/' + request.param)
    else:
        phi_model_root = os.path.join(models_root, request.param)

    assert os.path.exists(
        phi_model_root
    ), f"{phi_model_root} does not exist under NFS LLM_MODELS_ROOT dir"

    return phi_model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("bloom-560m", True)
def llm_bloom_560m_model_root(llm_venv):
    "prepare bloom 560m model & return bloom model root"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "bloom-560m")

    call(f"git clone https://huggingface.co/bigscience/bloom-560m {model_root}",
         shell=True)

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("bloom-3b", True)
def llm_bloom_3b_model_root(llm_venv):
    "prepare bloom 3b model & return bloom model root"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "bloom-3b")

    call(f"git clone https://huggingface.co/bigscience/bloom-3b {model_root}",
         shell=True)

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("bloom-7b1", True)
def llm_bloom_7b1_model_root(llm_venv):
    "prepare bloom 7b1 model & return bloom model root"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "bloom-7b1")

    call(f"git clone https://huggingface.co/bigscience/bloom-7b1 {model_root}",
         shell=True)

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("bloom", True)
def llm_bloom_176b_model_root(llm_venv):
    "prepare bloom 176b model & return bloom model root"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "bloom-176b")
    # There is no https://huggingface.co/bigscience/bloom-176b, just */bloom
    call(f"git clone https://huggingface.co/bigscience/bloom {model_root}",
         shell=True)

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("falcon-rw-1b", True)
def llm_falcon_rw_1b_model_root(llm_venv):
    "prepare falcon-rw-1b model & return falcon model root"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "falcon-rw-1b")

    call(f"git clone https://huggingface.co/tiiuae/falcon-rw-1b {model_root}",
         shell=True)

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("falcon-7b-instruct", True)
def llm_falcon_7b_model_root(llm_venv):
    "prepare falcon-7b model & return falcon model root"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "falcon-7b-instruct")

    call(
        f"git clone https://huggingface.co/tiiuae/falcon-7b-instruct {model_root}",
        shell=True)

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("falcon-40b", True)
def llm_falcon_40b_model_root(llm_venv):
    "prepare falcon 40b model & return falcon model root"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "falcon-40b-instruct")

    call(
        f"git clone https://huggingface.co/tiiuae/falcon-40b-instruct "
        f"{model_root}",
        shell=True)

    return model_root


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


@pytest.fixture(scope="module")
@cached_in_llm_models_root("chatglm-6b", True)
def llm_chatglm_6b_model_root(llm_venv):
    "prepare chatglm-6b model & return model path"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "chatglm-6b")

    return model_root


@pytest.fixture(scope="function")
def llm_chatglm2_6b_model_root(request):
    "prepare chatglm2_6b models"
    model_name = request.param
    models_root = llm_models_root()
    if model_name == "chatglm2-6b":
        model_root = os.path.join(models_root, "chatglm2-6b")
    elif model_name == "chatglm2-6b-32k":
        model_root = os.path.join(models_root, "chatglm2-6b-32k")

    return model_root


@pytest.fixture(scope="function")
def llm_chatglm3_6b_model_root(request):
    "prepare chatglm3-6b model & return model path"
    model_name = request.param
    models_root = llm_models_root()
    if model_name == "chatglm3-6b":
        model_root = os.path.join(models_root, "chatglm3-6b")
    elif model_name == "chatglm3-6b-32k":
        model_root = os.path.join(models_root, "chatglm3-6b-32k")
    elif model_name == "chatglm3-6b-base":
        model_root = os.path.join(models_root, "chatglm3-6b-base")
    elif model_name == "chatglm3-6b-128k":
        model_root = os.path.join(models_root, "chatglm3-6b-128k")

    return model_root


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
@cached_in_llm_models_root("glm-10b", True)
def llm_glm_10b_model_root(llm_venv):
    "prepare glm-10b model & return model path"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "glm-10b")

    return model_root


@pytest.fixture(scope="function")
def llm_baichuan_model_version_and_root(request):
    "prepare baichuan model & return model version and model root"
    model_version = request.param
    repo_name_dict = {
        "v1_7b": "Baichuan-7B",
        "v1_13b": "Baichuan-13B-Chat",
        "v2_7b": "Baichuan2-7B-Chat",
        "v2_13b": "Baichuan2-13B-Chat",
    }
    assert model_version in repo_name_dict
    repo_name = repo_name_dict[model_version]
    models_root = llm_models_root()
    baichuan_model_root = os.path.join(models_root, repo_name)

    return model_version, baichuan_model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("internlm-chat-7b", True)
def llm_internlm_7b_model_root(llm_venv):
    "prepare internlm 7b model"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "internlm-chat-7b")

    call(
        f"git clone https://huggingface.co/internlm/internlm-chat-7b {model_root}",
        shell=True)

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("internlm2-7b", True)
def llm_internlm2_7b_model_root(llm_venv):
    "prepare internlm2 7b model"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "internlm2-7b")

    call(f"git clone https://huggingface.co/internlm/internlm2-7b {model_root}",
         shell=True)

    return model_root


@pytest.fixture(scope="module")
@cached_in_llm_models_root("internlm-chat-20b", True)
def llm_internlm_20b_model_root(llm_venv):
    "prepare internlm 20b model"
    workspace = llm_venv.get_working_directory()
    model_root = os.path.join(workspace, "internlm-chat-20b")

    call(
        f"git clone https://huggingface.co/internlm/internlm-chat-20b {model_root}",
        shell=True)

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
def llm_dbrx_model_root(request):
    models_root = llm_models_root()
    model_name = request.param
    dbrx_model_root = os.path.join(models_root, model_name)
    assert exists(dbrx_model_root), f"{dbrx_model_root} does not exist!"
    return dbrx_model_root


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


@pytest.fixture(scope="function")
def deepseek_v2_model_root(request):
    models_root = llm_models_root()
    model_name = request.param
    deepseek_v2_model_root = os.path.join(models_root, model_name)
    assert exists(
        deepseek_v2_model_root), f"{deepseek_v2_model_root} does not exist!"
    return deepseek_v2_model_root


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


def evaltool_mmlu_post_process(results_path, baseline, threshold):
    # Note: In the older version of the lm-harness result file,
    # there are 57 values.
    # The latest version of lm-harness includes
    # 4 additional categories and 1 whole dataset in the result file.
    # We need to exclude these new categories and
    # the whole dataset when calculating the average.

    with open(results_path) as f:
        result = json.load(f)
        acc_acc = 0.0
        tasks_to_ignore = [
            "mmlu_str", "mmlu_str_stem", "mmlu_str_other",
            "mmlu_str_social_sciences", "mmlu_str_humanities"
        ]
        total_task = len(result['results']) - len(tasks_to_ignore)
        assert total_task == 57
        for sub_task in result['results']:
            if sub_task in tasks_to_ignore:
                continue
            acc_acc += float(result['results'][sub_task]['exact_match,none'])
        avg_acc = acc_acc / total_task
        print("MMLU avg accuracy:", avg_acc)
        assert abs(avg_acc - baseline) <= threshold


def evaltool_wikilingua_post_process(results_path, baseline, threshold):
    with open(results_path) as f:
        result = json.load(f)
        rouge_l = result['results']['wikilingua_english']['rougeL,none']
        print("Wikilingua_english rouge_L:", rouge_l)
        assert abs(rouge_l - baseline) <= threshold


def evaltool_humaneval_post_process(results_path, baseline, threshold):
    with open(results_path) as f:
        result = json.load(f)
        print(result)
        acc = result[0]['humaneval']['pass@1']
        assert abs(acc - baseline) <= threshold


def evaltool_mtbench_post_process(results_path, baseline, threshold):
    with open(results_path) as f:
        get_result = False
        for total_score in f:
            if total_score.startswith('total'):
                get_result = True
                total_score = float(total_score.split(',')[1].strip())
                assert abs(total_score - baseline) <= threshold
        assert get_result


@pytest.fixture(scope="module")
def evaltool_root(llm_venv):
    if GITLAB_API_USER is None or GITLAB_API_TOKEN is None or EVALTOOL_REPO_URL is None:
        pytest.skip(
            "Need to set GITLAB_API_USER, GITLAB_API_TOKEN, and EVALTOOL_REPO_URL env vars to run evaltool tests."
        )
    workspace = llm_venv.get_working_directory()
    clone_dir = os.path.join(workspace, "eval-tool")
    repo_url = f"https://{GITLAB_API_USER}:{GITLAB_API_TOKEN}@{EVALTOOL_REPO_URL}"
    branch_name = "dev/0.9"

    from evaltool.constants import EVALTOOL_SETUP_SCRIPT
    evaltool_setup_cmd = [
        EVALTOOL_SETUP_SCRIPT, "-b", branch_name, "-d", clone_dir, "-r",
        repo_url
    ]
    call(" ".join(evaltool_setup_cmd), shell=True)
    return clone_dir


@pytest.fixture(scope='module')
def grok_model_root():
    "get grok model"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    model_root = os.path.join(models_root, "grok-1")

    return model_root


@pytest.fixture(scope='module')
def grok_code_root(llm_venv):
    "get grok model"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    workspace = llm_venv.get_working_directory()
    code_root_src = os.path.join(models_root, "grok-github")
    code_root = os.path.join(workspace, "grok-github")
    shutil.copytree(code_root_src, code_root, dirs_exist_ok=True)

    ckpt_file = os.path.join(code_root, "checkpoint.py")
    assert exists(ckpt_file)

    with open(ckpt_file, 'r', encoding='UTF-8') as file:
        filedata = file.read()

    filedata = filedata.replace('/dev/shm', workspace)

    with open(ckpt_file, 'w', encoding='UTF-8') as file:
        file.write(filedata)

    return code_root


@pytest.fixture(scope="function")
def llm_dit_model_root(request):
    "return dit model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    if 'fp8' in request.param and '512' in request.param:
        dit_model_root = os.path.join(models_root,
                                      "DiT-XL-2-512x512.FP8.Linear.pt")
    else:
        if '512' in request.param:
            dit_model_root = os.path.join(models_root, "DiT-XL-2-512x512.pt")
        else:
            dit_model_root = os.path.join(models_root, "DiT-XL-2-256x256.pt")

    assert os.path.exists(
        dit_model_root
    ), f"{dit_model_root} does not exist under NFS LLM_MODELS_ROOT dir"

    return dit_model_root


@pytest.fixture(scope="function")
def mmdit_model_root(request):
    "return mmdit model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    mmdit_model_root = os.path.join(models_root, request.param)

    assert os.path.exists(
        mmdit_model_root
    ), f"{mmdit_model_root} does not exist under NFS LLM_MODELS_ROOT dir"

    return mmdit_model_root


@pytest.fixture(scope="function")
def stdit_model_root(request):
    "return stdit model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    stdit_model_root = os.path.join(models_root, request.param)

    assert os.path.exists(
        stdit_model_root
    ), f"{stdit_model_root} does not exist under NFS LLM_MODELS_ROOT dir"

    return stdit_model_root


@pytest.fixture(scope="function")
def sdxl_model_root(request):
    "return Stable Diffusion XL model root"
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    return os.path.join(models_root, "stable-diffusion-xl-base-1.0")


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


@pytest.fixture(scope="module")
def qcache_dir(llm_venv, llm_root):
    "get quantization cache dir"
    defs.ci_profiler.start("qcache_dir")

    cache_dir = os.path.join(llm_venv.get_working_directory(), "qcache")

    quantization_root = os.path.join(llm_root, "examples", "quantization")

    import platform

    # Fix the issue that the requirements.txt is not available on aarch64.
    if "aarch64" not in platform.machine() and get_sm_version() >= 89:
        llm_venv.run_cmd([
            "-m", "pip", "install", "-r",
            os.path.join(quantization_root, "requirements.txt")
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


@pytest.fixture(autouse=True)
def skip_by_device_count(request):
    "fixture for skip less device count"
    if request.node.get_closest_marker('skip_less_device'):
        device_count = get_device_count()
        expected_count = request.node.get_closest_marker(
            'skip_less_device').args[0]
        if expected_count > int(device_count):
            pytest.skip(
                f'Device count {device_count} is less than {expected_count}')


@pytest.fixture(autouse=True)
def skip_by_device_memory(request):
    "fixture for skip less device memory"
    if request.node.get_closest_marker('skip_less_device_memory'):
        device_memory = get_device_memory()
        expected_memory = request.node.get_closest_marker(
            'skip_less_device_memory').args[0]
        if expected_memory > int(device_memory):
            pytest.skip(
                f'Device memory {device_memory} is less than {expected_memory}')


def get_sm_version():
    "get compute capability"
    with tempfile.TemporaryDirectory() as temp_dirname:
        suffix = ".exe" if is_windows() else ""
        # TODO: Use NRSU because we can't assume nvidia-smi across all platforms.
        cmd = " ".join([
            "nvidia-smi" + suffix, "--query-gpu=compute_cap",
            "--format=csv,noheader"
        ])
        output = check_output(cmd, shell=True, cwd=temp_dirname)

    compute_cap = output.strip().split("\n")[0]
    sm_major, sm_minor = list(map(int, compute_cap.split(".")))

    return sm_major * 10 + sm_minor


skip_pre_ada = pytest.mark.skipif(
    get_sm_version() < 89,
    reason="This test is not supported in pre-Ada architecture")

skip_pre_hopper = pytest.mark.skipif(
    get_sm_version() < 90,
    reason="This test is not supported in pre-Hopper architecture")

skip_pre_blackwell = pytest.mark.skipif(
    get_sm_version() < 100,
    reason="This test is not supported in pre-Blackwell architecture")

skip_post_blackwell = pytest.mark.skipif(
    get_sm_version() >= 100,
    reason="This test is not supported in post-Blackwell architecture")


def skip_fp8_pre_ada(use_fp8):
    "skip fp8 tests if sm version less than 8.9"
    if use_fp8 and get_sm_version() < 89:
        pytest.skip("FP8 is not supported on pre-Ada architectures")


def skip_fp4_pre_blackwell(use_fp4):
    "skip fp4 tests if sm version less than 10.0"
    if use_fp4 and get_sm_version() < 100:
        pytest.skip("FP4 is not supported on pre-Blackwell architectures")


def skip_if_no_nvls(llm_venv):
    output_str = llm_venv.run_output(
        "from tensorrt_llm.bindings import ipc_nvls_supported; print('NVLS supported' if ipc_nvls_supported() else 'False')"
    )
    if 'NVLS supported' not in output_str:
        pytest.skip("NVLS is not supported")


@pytest.fixture(autouse=True)
def skip_device_not_contain(request):
    "skip test if device not contain keyword"
    if request.node.get_closest_marker('skip_device_not_contain'):
        keyword_list = request.node.get_closest_marker(
            'skip_device_not_contain').args[0]
        device = get_gpu_device_list()[0]
        if not any(keyword in device for keyword in keyword_list):
            pytest.skip(
                f"Device {device} does not contain keyword in {keyword_list}.")


def get_gpu_device_list():
    "get device list"
    with tempfile.TemporaryDirectory() as temp_dirname:
        suffix = ".exe" if is_windows() else ""
        # TODO: Use NRSU because we can't assume nvidia-smi across all platforms.
        cmd = " ".join(["nvidia-smi" + suffix, "-L"])
        output = check_output(cmd, shell=True, cwd=temp_dirname)
    return [l.strip() for l in output.strip().split("\n")]


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
        output = check_output(cmd, shell=True, cwd=temp_dirname)
        memory = int(output.strip().split()[0])

    return memory


#
# When test parameters have an empty id, older versions of pytest ignored that parameter when generating the
# test node's ID completely. This however was actually a bug, and not expected behavior that got fixed in newer
# versions of pytest:https://github.com/pytest-dev/pytest/pull/6607. TRT test defs however rely on this behavior
# for quite a few test names. This is a hacky WAR that restores the old behavior back so that the
# test names do not change. Note: This might break in a future pytest version.
#
# TODO: Remove this hack once the test names are fixed.
#

from _pytest.python import CallSpec2

CallSpec2.id = property(
    lambda self: "-".join(map(str, filter(None, self._idlist))))


def pytest_addoption(parser):
    parser.addoption(
        "--test-list",
        "-F",
        action="store",
        default=None,
        help="Path to the file containing the list of tests to run")
    parser.addoption(
        "--workspace",
        "--ws",
        action="store",
        default=None,
        help="Workspace path to store temp data generated during the tests")
    parser.addoption(
        "--waives-file",
        "-S",
        action="store",
        default=None,
        help=
        "Specify a file containing a list of waives, one per line. After filtering collected tests, Pytest will "
        "apply the waive state specified by this file to the set of tests to be run."
    )
    parser.addoption(
        "--output-dir",
        "-O",
        action="store",
        default=None,
        help=
        "Directory to store test output. Should point to a new or existing empty directory."
    )
    parser.addoption(
        "--test-prefix",
        "-P",
        action="store",
        default=None,
        help=
        "It is useful when using such prefix to mapping waive lists for specific GPU, such as 'GH200'"
    )
    parser.addoption("--regexp",
                     "-R",
                     action='store',
                     default=None,
                     help="A regexp to specify which tests to run")
    parser.addoption(
        "--apply-test-list-correction",
        "-C",
        action='store_true',
        help=
        "Attempt to automatically correct invalid test names in filter files and print the correct name in terminal. "
        "If the correct name cannot be determined, the invalid test name will be printed to the terminal as well."
    )
    parser.addoption("--perf",
                     action="store_true",
                     help="'--perf' will run perf tests")
    parser.addoption(
        "--perf-log-formats",
        help=
        "Supply either 'yaml' or 'csv' as values. Supply multiple same flags for multiple formats.",
        action="append",
        default=[])


@pytest.hookimpl(trylast=True)
def pytest_generate_tests(metafunc: pytest.Metafunc):
    if metafunc.definition.function.__name__ != 'test_unittests_v2':
        return
    testlist_path = metafunc.config.getoption("--test-list")
    if not testlist_path:
        return

    with open(testlist_path, "r") as f:
        lines = f.readlines()
        lines = preprocess_test_list_lines(testlist_path, lines)

    uts = []
    for line in lines:
        if line.startswith("unittest/"):
            uts.append(line.strip())
    metafunc.parametrize("case", uts, ids=lambda x: x)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_collection_modifyitems(session, config, items):
    testlist_path = config.getoption("--test-list")
    waives_file = config.getoption("--waives-file")
    test_prefix = config.getoption("--test-prefix")
    perf_test = config.getoption("--perf")

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
def turtle_root():
    return os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope="function")
def test_case(request, llm_root):
    "get test case"
    test_cases_file = "tests/integration/defs/test_cases.yml"
    input_file_dir = "tests/integration/test_input_files"
    test_cases_file_path = os.path.join(llm_root, test_cases_file)
    case_name = request.param

    with open(test_cases_file_path, 'r', encoding='UTF-8') as file:
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
    if request.node.get_closest_marker('skip_less_host_memory'):
        host_memory = get_host_total_memory()
        expected_memory = request.node.get_closest_marker(
            'skip_less_host_memory').args[0]
        if expected_memory > int(host_memory):
            pytest.skip(
                f'Host memory {host_memory} is less than {expected_memory}')


IS_UNDER_CI_ENV = 'JENKINS_HOME' in os.environ


def collect_status():
    if not IS_UNDER_CI_ENV:
        return

    import psutil
    import pynvml
    pynvml.nvmlInit()

    handles = {
        idx: pynvml.nvmlDeviceGetHandleByIndex(idx)
        for idx in range(pynvml.nvmlDeviceGetCount())
    }

    gpu_memory = {}
    for idx, device in handles.items():
        total_used = pynvml.nvmlDeviceGetMemoryInfo(device).used // 1024 // 1024
        total = pynvml.nvmlDeviceGetMemoryInfo(device).total // 1024 // 1024
        detail = pynvml.nvmlDeviceGetComputeRunningProcesses(device)
        process = {}

        for entry in detail:
            host_memory_in_mbs = -1
            try:
                host_memory_in_mbs = psutil.Process(
                    entry.pid).memory_full_info().uss // 1024 // 1024
                process[entry.pid] = (entry.usedGpuMemory // 1024 // 1024,
                                      host_memory_in_mbs)
            except:
                pass

        gpu_memory[idx] = {
            "total_used": total_used,
            'total': total,
            "process": process
        }
    print('\nCurrent memory status:')
    print(gpu_memory)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    ret = yield
    collect_status()
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
