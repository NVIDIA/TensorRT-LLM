# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import psutil
import pynvml

from tensorrt_llm.logger import logger

DEFAULT_HF_MODEL_DIRS = {
    "BaichuanForCausalLM": "baichuan-inc/Baichuan-13B-Chat",
    "BloomForCausalLM": "bigscience/bloom-560m",
    "GLMModel": "THUDM/glm-10b",
    "ChatGLMModel": "THUDM/chatglm3-6b",
    "ChatGLMForCausalLM": "THUDM/chatglm3-6b",
    "FalconForCausalLM": "tiiuae/falcon-rw-1b",
    "GPTForCausalLM": "gpt2-medium",
    "GPTJForCausalLM": "EleutherAI/gpt-j-6b",
    "GPTNeoXForCausalLM": "EleutherAI/gpt-neox-20b",
    "InternLMForCausalLM": "internlm/internlm-chat-7b",
    "InternLM2ForCausalLM": "internlm/internlm2-chat-7b",
    "LlamaForCausalLM": "meta-llama/Llama-2-7b-hf",
    "MPTForCausalLM": "mosaicml/mpt-7b",
    "PhiForCausalLM": "microsoft/phi-2",
    "OPTForCausalLM": "facebook/opt-350m",
    "QWenLMHeadModel": "Qwen/Qwen-7B",
    "QWenForCausalLM": "Qwen/Qwen-7B",
    "Qwen2ForCausalLM": "Qwen/Qwen1.5-7B",
    "Qwen2MoeForCausalLM": "Qwen/Qwen1.5-MoE-A2.7B",
    "RecurrentGemmaForCausalLM": "google/recurrentgemma-2b",
}


# cgroup v2, then v1. A v1 file holds a huge sentinel value when there is no limit.
_CGROUP_MEMORY_LIMIT_FILES = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)


def _cgroup_memory_limit():
    """Return the cgroup memory limit in bytes, or None if there is none."""
    for path in _CGROUP_MEMORY_LIMIT_FILES:
        try:
            with open(path) as f:
                value = f.read().strip()
        except OSError:
            continue
        if value.isdigit():
            return int(value)
        return None
    return None


def get_device_memory():
    """Return the total memory of GPU 0 in GiB.

    On unified-memory systems such as DGX Spark (GB10), NVML does not report
    device memory and ``nvmlDeviceGetMemoryInfo`` raises
    ``NVMLError_NotSupported``. The GPU then shares system memory, so fall
    back to the total system memory, capped by the cgroup memory limit when
    running inside a container that has one.
    """
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        try:
            total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        except pynvml.NVMLError_NotSupported:
            total_memory = psutil.virtual_memory().total
            cgroup_limit = _cgroup_memory_limit()
            if cgroup_limit is not None:
                total_memory = min(total_memory, cgroup_limit)
            logger.warning(
                "NVML does not support device memory queries for GPU 0 "
                "(e.g. unified memory on DGX Spark). Using system memory "
                f"({total_memory / (1024**3):.2f} GiB, capped by the cgroup "
                "memory limit if any) to estimate "
                "max_batch_size and max_num_tokens."
            )
    finally:
        pynvml.nvmlShutdown()

    return total_memory / (1024**3)
