# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for Router Replay (R3).

Runs enable_return_routed_experts on a real (small) MoE model and asserts the
per-token routing is returned with the right shape, including concurrent requests
(mirrors the vLLM / SGLang feature tests).
"""

import os

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import skip_gpu_memory_less_than_40gb

from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams

_MODEL = "Qwen1.5-MoE-A2.7B-Chat"  # small CUTLASS separated-routing MoE


def _model_dir():
    root = llm_models_root()
    path = os.path.join(root, _MODEL) if root else None
    if not path or not os.path.isdir(path):
        pytest.skip(f"{_MODEL} not available under llm_models_root()")
    return path


def _check_routes(routes, prompt_len, gen_len):
    assert routes is not None, "routed_experts missing on output"
    assert isinstance(routes, torch.Tensor)
    assert routes.ndim == 3  # [seq_len - 1, num_moe_layers, top_k]
    assert routes.shape[0] == prompt_len + gen_len - 1
    assert routes.shape[1] > 0 and routes.shape[2] > 0
    # expert ids are non-negative (or the -1 fail-closed sentinel), int-typed.
    assert routes.dtype in (torch.int16, torch.int32, torch.int64)
    assert int(routes.min()) >= -1


@skip_gpu_memory_less_than_40gb
def test_return_routed_experts_shape_and_concurrency():
    llm = LLM(model=_model_dir(), enable_return_routed_experts=True)
    try:
        sp = SamplingParams(max_tokens=16, temperature=0.0, return_routed_experts=True)
        prompts = ["The capital of France is", "1 + 1 =", "Hello, my name is"]
        outputs = llm.generate(prompts, sp)
        assert len(outputs) == len(prompts)
        for req in outputs:
            out = req.outputs[0]
            _check_routes(
                out.routed_experts, prompt_len=len(req.prompt_token_ids), gen_len=len(out.token_ids)
            )
    finally:
        llm.shutdown()


# NOTE: per-request gating (returning routes only for requests whose
# SamplingParams.return_routed_experts is True) is a follow-up: it needs the flag
# plumbed onto the C++ executor OutputConfig / request. The engine-level
# enable_return_routed_experts currently returns routes for all requests, which
# is what the MoE-RL rollout use case wants.
