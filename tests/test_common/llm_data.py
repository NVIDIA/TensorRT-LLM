# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared utilities for local LLM model paths and HuggingFace download mocking."""

import os
from functools import wraps
from pathlib import Path
from typing import Optional
from unittest.mock import patch

# Mapping from HuggingFace Hub ID to local subdirectory under LLM_MODELS_ROOT.
# NOTE: hf_id_to_llm_models_subdir below will fall back to checking if the model name exists
# in LLM_MODELS_ROOT if not present here, so it's not required to list models that already
# exist as a top-level directory in LLM_MODELS_ROOT.
HF_ID_TO_LLM_MODELS_SUBDIR = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama-3.1-model/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-model/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B": "llama-3.1-model/Meta-Llama-3.1-8B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama4-models/Llama-4-Scout-17B-16E-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "Mistral-Small-3.1-24B-Instruct-2503",
    "Qwen/Qwen3-30B-A3B": "Qwen3/Qwen3-30B-A3B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct": "Phi-3/Phi-3-mini-4k-instruct",
    "deepseek-ai/DeepSeek-V3": "DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1": "DeepSeek-R1/DeepSeek-R1",
    "ibm-ai-platform/Bamba-9B-v2": "Bamba-9B-v2",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2": "NVIDIA-Nemotron-Nano-12B-v2",
    "nvidia/NVIDIA-Nemotron-Nano-31B-A3-v3": "NVIDIA-Nemotron-Nano-31B-A3-v3",
    "nvidia/Nemotron-Nano-3-30B-A3.5B-dev-1024": "Nemotron-Nano-3-30B-A3.5B-dev-1024",
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B": "EAGLE3-LLaMA3.1-Instruct-8B",
}


def llm_models_root(check: bool = False) -> Optional[Path]:
    root = Path("/home/scratch.trt_llm_data/llm-models/")

    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ.get("LLM_MODELS_ROOT"))

    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")

    if check:
        assert root.exists(), (
            "You must set LLM_MODELS_ROOT env or be able to access /home/scratch.trt_llm_data to run this test"
        )

    return root if root.exists() else None


def llm_datasets_root() -> str:
    return os.path.join(llm_models_root(check=True), "datasets")


def hf_id_to_local_model_dir(hf_hub_id: str) -> str | None:
    """Return the local model directory under LLM_MODELS_ROOT for a given HuggingFace Hub ID.

    Raises ValueError if the model is not found in LLM_MODELS_ROOT. This is meant to ensure that tests do not download
    models from HuggingFace.
    """
    root = llm_models_root()
    if root is None:
        return None

    if hf_hub_id in HF_ID_TO_LLM_MODELS_SUBDIR:
        return str(root / HF_ID_TO_LLM_MODELS_SUBDIR[hf_hub_id])

    # Fall back to checking if the model name exists as a top-level directory in LLM_MODELS_ROOT
    model_name = hf_hub_id.split("/")[-1]
    if os.path.isdir(root / model_name):
        return str(root / model_name)

    raise ValueError(f"HuggingFace model '{hf_hub_id}' not found in LLM_MODELS_ROOT")


def mock_snapshot_download(repo_id: str, **kwargs) -> str:
    """Mock huggingface_hub.snapshot_download that returns an existing local model directory.

    NOTE: This function does not currently handle the revision / allow_patterns / ignore_patterns parameters.
    """
    local_path = hf_id_to_local_model_dir(repo_id)
    if local_path is None:
        raise ValueError(f"Model '{repo_id}' not found in LLM_MODELS_ROOT")
    return local_path


def with_mocked_hf_download_for_single_gpu(func):
    """Decorator to mock huggingface_hub.snapshot_download for tests.

    When applied, any calls to snapshot_download will be redirected to use
    local model paths from LLM_MODELS_ROOT instead of downloading from HuggingFace.

    NOTE: We must patch snapshot_download at the location where it's actually imported
    with 'from huggingface_hub import snapshot_download', since that creates a
    local binding that won't be affected by patching huggingface_hub.snapshot_download.

    Additionally sets HF_HUB_OFFLINE=1 to ensure no network requests are made to
    HuggingFace.

    WARNING: This decorator only works for single-GPU tests. For multi-GPU tests, the
    mock won't be applied in MPI worker processes.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with (
            patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}),
            patch(
                "tensorrt_llm.llmapi.utils.snapshot_download", side_effect=mock_snapshot_download
            ),
        ):
            return func(*args, **kwargs)

    return wrapper
