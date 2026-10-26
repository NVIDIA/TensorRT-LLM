# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Whole-model gates for the step5 modeling_v2 target.

Every test passes ``modeling_v2="require"``: under ``"auto"`` a configuration
that missed the target's routing would fall back to the built-in model, pass,
and report the built-in's numbers as the target's.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

from ...conftest import llm_models_root
from ..accuracy_core import GSM8K, LlmapiAccuracyTestHarness

# The target is certified on sm_103 only.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="the step5 modeling_v2 target is certified on sm_103 only"
)

# Check the target's step contract on the first forward, against metadata the
# engine prepared; the flag has to reach the worker ranks.
_VALIDATE = {MODELING_V2_VALIDATE_ENV: "1"}


def _require_target(monkeypatch: pytest.MonkeyPatch, **env: str) -> dict:
    """``LLM(...)`` kwargs that build the target, with ``env`` on the ranks.

    ``LLM`` applies ``env_overrides`` to this process and never restores it, so
    ``monkeypatch`` sets it too, keeping it from leaking into later tests.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return dict(modeling_v2="require", env_overrides=dict(env))


def _layout(layout_id, world_size, moe_tp_size, moe_ep_size, attention_dp):
    """One parallel layout as a pytest param, with the device count it needs."""
    return pytest.param(
        dict(
            tensor_parallel_size=world_size,
            moe_tensor_parallel_size=moe_tp_size,
            moe_expert_parallel_size=moe_ep_size,
            enable_attention_dp=attention_dp,
        ),
        id=layout_id,
        marks=pytest.mark.skip_less_device(world_size),
    )


# The layouts the target's routing admits (`_LAYOUTS` in models/step5/routing.py).
_TP4 = _layout("tp4", 4, 4, 1, False)
_TEP4 = _layout("tep4", 4, 1, 4, False)
_DEP4 = _layout("dep4", 4, 1, 4, True)
_DEP2 = _layout("dep2", 2, 1, 2, True)

# One GB300 tray: each attention path (heads split across ranks, or whole per
# rank under attention DP) and each expert split (MoE TP, or EP).
_TRAY_LAYOUTS = [_TP4, _TEP4, _DEP4]


class TestModelingV2Step5PreviewNvfp4Sm103Unified(LlmapiAccuracyTestHarness):
    """step5-preview-nvfp4 / sm_103 / unified: gsm8k at each layout and KV-cache dtype.

    The engine settings are the serving envelope the target is certified for
    (ISL 8192 + OSL 1024, 128 requests), with chunked prefill, KV-cache block
    reuse and padded CUDA graphs up to batch 128 on in every case. The few-shot
    gsm8k prompts share their prefix, so a block-reuse defect in the
    sliding-window or sparse-attention state shows as an accuracy loss.

    The score is the evaluator's default: the mean of strict-match and
    flexible-extract, the protocol the references were measured under.
    """

    MODEL_NAME = "stepfun-ai/Step-5-Preview"
    MODEL_PATH = f"{llm_models_root()}/Step-5-Preview-NVFP4"

    ENGINE = dict(
        max_seq_len=9216,
        max_batch_size=128,
        max_num_tokens=8192,
        enable_chunked_prefill=True,
        cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=128),
    )

    @staticmethod
    def _kv_cache(dtype: str = "auto") -> KvCacheConfig:
        return KvCacheConfig(free_gpu_memory_fraction=0.7, enable_block_reuse=True, dtype=dtype)

    def _gsm8k(self, monkeypatch, parallel, kv_dtype="auto", **engine):
        with LLM(
            self.MODEL_PATH,
            kv_cache_config=self._kv_cache(kv_dtype),
            **{**self.ENGINE, **engine},
            **parallel,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_not_sm103
    @pytest.mark.parametrize("parallel", [*_TRAY_LAYOUTS, _DEP2])
    def test_gsm8k(self, monkeypatch, parallel):
        """bf16 KV cache, at every layout routing admits.

        dep2 is the layout of each instance of the disaggregated deployment
        (context and generation, attention DP2 and EP2); here it serves
        aggregated.
        """
        self._gsm8k(monkeypatch, parallel)

    @skip_not_sm103
    @pytest.mark.parametrize("parallel", _TRAY_LAYOUTS)
    def test_gsm8k_fp8_kv(self, monkeypatch, parallel):
        """FP8 KV cache: the attention K/V of every layer stored as e4m3."""
        self._gsm8k(monkeypatch, parallel, kv_dtype="fp8")

    @skip_not_sm103
    @pytest.mark.parametrize("parallel", _TRAY_LAYOUTS)
    def test_gsm8k_nvfp4_kv(self, monkeypatch, parallel):
        """NVFP4 KV cache: the attention K/V of every layer stored as NVFP4."""
        self._gsm8k(monkeypatch, parallel, kv_dtype="nvfp4")

    @skip_not_sm103
    @pytest.mark.parametrize("parallel", _TRAY_LAYOUTS)
    def test_gsm8k_chunked_prefill(self, monkeypatch, parallel):
        """Prefill in 512-token chunks.

        Every few-shot prompt is longer than one chunk, so the sliding-window
        K/V, the sparse indexer's keys and region summaries, and the sparse
        selection all cross chunk boundaries: each chunk after the first
        attends to and selects from state an earlier step wrote.
        """
        self._gsm8k(monkeypatch, parallel, max_num_tokens=512)
