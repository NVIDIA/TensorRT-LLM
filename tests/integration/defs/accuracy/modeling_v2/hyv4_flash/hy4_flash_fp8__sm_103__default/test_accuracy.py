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
"""GSM8K gates for the hyv4_flash target on sm_103.

Variants without a reference row (FP8 KV, chunked prefill, block reuse) are measured beside core
in the same test and held to the sampling error of the difference.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.quantization import QuantAlgo

from .....conftest import llm_models_root
from ....accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from ..._env import scoped_env_overrides
from ._common import (
    DEP4,
    GSM8K_EVALUATOR_KWARGS,
    KV_CACHE_FRACTION,
    TEP4,
    TEP8,
    TP4,
    VALIDATE,
    assert_target_engine,
    dfly2,
    engine,
    gsm8k_samples,
    relative_band,
    sampling,
    skip_not_sm103,
)


@pytest.fixture(autouse=True)
def _score_flexible_extract(mocker):
    """The reference rows are flexible-extract scores; the checkpoint seldom writes ``#### N``."""
    mocker.patch.dict(GSM8K.EVALUATE_KWARGS, {"scores_filter": "exact_match,flexible-extract"})


_LAYOUTS = [
    pytest.param(TP4, id="tp4", marks=pytest.mark.skip_less_device(4)),
    pytest.param(TEP4, id="tep4", marks=pytest.mark.skip_less_device(4)),
    pytest.param(DEP4, id="dep4", marks=pytest.mark.skip_less_device(4)),
    pytest.param(TEP8, id="tep8", marks=pytest.mark.skip_less_mpi_world_size(8)),
]

# The Engram tables live in host memory beside the weights.
_HOST_MEMORY_MB = 204800


class TestModelingV2Hy4FlashFp8Sm103Default(LlmapiAccuracyTestHarness):
    MODEL_NAME = "tencent/Hy4-flash-FP8"
    MODEL_PATH = f"{llm_models_root()}/Hy4-flash-FP8"

    def _llm(self, scope, **options) -> LLM:
        return LLM(self.MODEL_PATH, **options, modeling_v2="require", env_overrides=scope)

    def _evaluate(self, llm) -> float:
        return GSM8K(self.MODEL_NAME).evaluate(
            llm, extra_evaluator_kwargs=GSM8K_EVALUATOR_KWARGS, sampling_params=sampling()
        )

    @skip_not_sm103
    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device_memory(145000)
    @pytest.mark.skip_less_host_memory(_HOST_MEMORY_MB)
    @pytest.mark.parametrize("parallel", _LAYOUTS)
    def test_gsm8k(self, parallel):
        with (
            scoped_env_overrides(VALIDATE) as rank_env,
            self._llm(rank_env, **engine(), **parallel) as llm,
        ):
            assert_target_engine(llm)
            self._evaluate(llm)

    @skip_not_sm103
    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(145000)
    @pytest.mark.skip_less_host_memory(_HOST_MEMORY_MB)
    def test_dflash_gsm8k(self):
        with (
            scoped_env_overrides(VALIDATE) as rank_env,
            self._llm(rank_env, speculative_config=dfly2(), **engine(), **TEP4) as llm,
        ):
            assert_target_engine(llm)
            self._evaluate(llm)

    def _paired_gsm8k(self, variant: dict, expect=None) -> None:
        """Core (asserted against its row) and ``variant`` at tep4, within ``relative_band``."""
        with (
            scoped_env_overrides(VALIDATE) as rank_env,
            self._llm(rank_env, **engine(), **TEP4) as llm,
        ):
            assert_target_engine(llm)
            core = self._evaluate(llm)
        samples = gsm8k_samples()
        with (
            scoped_env_overrides(VALIDATE) as rank_env,
            self._llm(rank_env, **engine(**variant), **TEP4) as llm,
        ):
            assert_target_engine(llm)
            if expect is not None:
                expect(llm)
            evaluator = GSM8K(self.MODEL_NAME).create_evaluator(samples, GSM8K_EVALUATOR_KWARGS)
            measured = evaluator.evaluate(llm, sampling(), False, **GSM8K.EVALUATE_KWARGS)
        band = relative_band(core, samples)
        assert abs(measured - core) <= band, (
            f"core={core:.4f} variant={measured:.4f} band={band:.4f}"
        )

    @skip_not_sm103
    @pytest.mark.timeout(10800)
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(145000)
    @pytest.mark.skip_less_host_memory(_HOST_MEMORY_MB)
    def test_kv_fp8_gsm8k(self):
        def expect(llm):
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8

        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=KV_CACHE_FRACTION, enable_block_reuse=False, dtype="fp8"
        )
        self._paired_gsm8k(dict(kv_cache_config=kv_cache_config), expect)

    @skip_not_sm103
    @pytest.mark.timeout(10800)
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(145000)
    @pytest.mark.skip_less_host_memory(_HOST_MEMORY_MB)
    def test_chunked_gsm8k(self):
        self._paired_gsm8k(dict(enable_chunked_prefill=True, max_num_tokens=512))

    @skip_not_sm103
    @pytest.mark.timeout(10800)
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(145000)
    @pytest.mark.skip_less_host_memory(_HOST_MEMORY_MB)
    def test_reuse_gsm8k(self):
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=KV_CACHE_FRACTION, enable_block_reuse=True
        )
        self._paired_gsm8k(dict(kv_cache_config=kv_cache_config))
