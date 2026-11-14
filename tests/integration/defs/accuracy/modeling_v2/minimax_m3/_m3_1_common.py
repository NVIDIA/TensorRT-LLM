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
"""Shared harness for the MiniMax-M3.1-Flash modeling_v2 accuracy targets.

Every test passes ``modeling_v2="require"`` to ``LLM``: under ``"auto"`` a configuration that missed a target's
criteria would fall back to the built-in implementation and report its numbers as the target's. ``require`` also builds
the DEP target, which the default mode leaves to the built-in model.

The deployment is the one ``models/minimax_m3/routing.py`` routes to the M3.1 targets: the MiniMax-M3.1-Flash
checkpoint (MXFP8 dense projections, packed NVFP4 routed experts) on sm_107 (Rubin), the MSA sparse attention with the
fused QKV/index projection and an FP8 indexer, an NVFP4 KV cache, and either no speculation or the checkpoint's
standalone DSpark drafter. Each target subclass sets its parallel layout.

GSM8K is scored against ``references/gsm8k.yaml`` under ``MODEL_NAME``; ``GSM8K.evaluate`` raises "Not registered
specs" until a measured reference for these specs is registered there (see ``accuracy/README.md``).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Dict

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MiniMaxM3SparseAttentionConfig,
)
from tensorrt_llm.quantization import QuantAlgo

from ....conftest import llm_models_root
from ...accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from .._env import scoped_env_overrides

# The targets assert their own SM at construction: certification is per GPU architecture.
skip_not_sm107 = pytest.mark.skipif(
    get_sm_version() != 107,
    reason="the MiniMax-M3.1-Flash modeling_v2 targets are certified on sm_107 only",
)


class _MiniMaxM31Target(LlmapiAccuracyTestHarness):
    MODEL_NAME = "MiniMaxAI/MiniMax-M3.1-Flash"
    MODEL_PATH = f"{llm_models_root()}/MiniMax-M3.1-Flash"
    # The standalone DSpark drafter ships inside the checkpoint directory.
    DSPARK_PATH = f"{MODEL_PATH}/dspark"
    # The parallel layout the target routes on (and any backend it requires), as LLM arguments.
    DEPLOYMENT: Dict[str, Any] = dict(tensor_parallel_size=2, moe_expert_parallel_size=1)

    @contextmanager
    def _llm(self, dspark: bool) -> Iterator[LLM]:
        spec = (
            DSparkDecodingConfig(max_draft_len=7, speculative_model=self.DSPARK_PATH)
            if dspark
            else None
        )
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                **self.DEPLOYMENT,
                kv_cache_config=KvCacheConfig(
                    free_gpu_memory_fraction=0.6, enable_block_reuse=False, dtype="nvfp4"
                ),
                sparse_attention_config=MiniMaxM3SparseAttentionConfig(
                    implementation="msa",
                    indexer_kv_dtype="fp8",
                    fuse_qkv_index_projection=True,
                    sparse_disable_index_value=True,
                ),
                max_seq_len=4096,
                max_batch_size=64,
                cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=64),
                speculative_config=spec,
                trust_remote_code=True,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            yield llm

    def _gsm8k(self, dspark: bool) -> None:
        with self._llm(dspark) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.MXFP8
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
