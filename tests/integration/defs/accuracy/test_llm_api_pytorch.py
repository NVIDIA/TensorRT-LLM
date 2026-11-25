# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional
from unittest import mock

import pytest
import torch
import torch._inductor.config as inductor_config
from datasets import load_dataset
from defs.conftest import get_sm_version, is_sm_100f
from mpi4py.futures import MPIPoolExecutor
from pytest_mock import MockerFixture
from test_common.moe_tactics import (fp8_block_scales_moe_config,
                                     fp8_block_scales_moe_tactic,
                                     skip_unless_moe_tactic_runs_here,
                                     sm100f_moe_tactic_or_cutlass)

from tensorrt_llm import LLM
from tensorrt_llm._torch.model_config import MoeLoadBalancerConfig

# isort: off
from tensorrt_llm.llmapi import (
    AttentionDpConfig, AutoTunerNvMMHConfig, CudaGraphConfig,
    DFlashDecodingConfig, DSparkDecodingConfig, Eagle3DecodingConfig,
    KvCacheConfig, MambaStateConfig, MoeConfig, MTPDecodingConfig,
    PrefillCudaGraphBackend, QSASparseAttentionConfig, SamplingParams,
    SchedulerConfig, SkipSoftmaxAttentionConfig, SAEnhancerConfig,
    TorchCompileConfig)
# isort: on
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import (llm_models_root, parametrize_with_ids, skip_no_hopper,
                        skip_no_rubin, skip_no_sm100f, skip_no_sm120,
                        skip_post_hopper, skip_pre_ada, skip_pre_blackwell,
                        skip_pre_hopper, skip_ray, skip_x86)
from ..deepseek_v41_gsm8k import GSM8K_TEST_TIMEOUT, evaluate_gsm8k

# isort: off
from .accuracy_core import (
    GSM8K, MMLU, CnnDailymail, GPQADiamond, JsonModeEval,
    LlmapiAccuracyTestHarness, LongBenchV2, assert_acceptance_length,
    assert_acceptance_length_for_llm, assert_guided_decoding_regex,
    assert_kv_cache_reuse_for_llm, compute_acceptance_length)

# isort: on


# Keep helper definitions below imports so new imports do not need E402
# suppressions in this legacy test file.
def patch_mpi_pool_session_for_env(mocker, env_vars: dict):
    """Patch MpiPoolSession._start_mpi_pool to propagate environment variables to MPI child processes.

    Uses MPIPoolExecutor's built-in `env` parameter instead of `initializer` to avoid
    segfault issues during process cleanup (UCX memory cache conflicts with PyTorch
    tensor cleanup during Py_FinalizeEx).

    Args:
        mocker: pytest-mock mocker fixture
        env_vars: Dictionary of environment variable name -> value to propagate
    """
    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    def patched_start_mpi_pool(self):
        assert not self.mpi_pool, 'MPI session already started'
        self.mpi_pool = MPIPoolExecutor(max_workers=self.n_workers,
                                        path=sys.path,
                                        env=env_vars)

    mocker.patch.object(MpiPoolSession, '_start_mpi_pool',
                        patched_start_mpi_pool)


def _count_prims_ts_phase_calls(mocker):
    """Count PrimTS phase launches while keeping the real kernels installed."""
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts import PrimsTSFmha

    calls = {
        "context": 0,
        "generation": 0,
        "mla_generation": 0,
    }

    def patch_method(method_name, counter_name):
        original = getattr(PrimsTSFmha, method_name)

        def counted(self, params):
            calls[counter_name] += 1
            return original(self, params)

        mocker.patch.object(PrimsTSFmha, method_name, counted)

    patch_method("run_context", "context")
    patch_method("run_generation", "generation")
    patch_method("run_mla_generation", "mla_generation")
    return calls


def _enable_dense_prims_ts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a multi-GPU MLA accuracy case on dense PrimTS attention.

    Dense PrimTS MLA decode is opt-in on SM100 and SM103. The MPI workers read
    TLLM_FMHA_LIBS from this process when the LLM starts them. Other GPUs never
    select PrimTS, so the variable stays unset there: session reuse respawns
    its cached MPI pool whenever the environment changes.
    """
    if is_sm_100f():
        monkeypatch.setenv("TLLM_FMHA_LIBS", "+prims_ts")


# The only TestDeepSeekV3Lite four-GPU case on dense PrimTS MLA decode, with
# MTP, attention DP, CUDA graphs and the overlap scheduler. It is registered in
# l0_dgx_b300, l0_gb200_multi_gpus and l0_gb300_multi_gpus.
_DENSE_PRIMS_TS_BF16_4GPUS_ID = (
    "ep4-mtp_nextn=2-attention_dp=True-cuda_graph=True-overlap_scheduler=True-torch_compile=False"
)


# MPI session reuse cannot safely reset Userbuffers between Engines. Tests
# using this helper must keep ``torch_compile=True`` in their node id; tests
# with unconditional compile configs must keep ``piecewise_cuda_graph`` in
# their node id. Keep this contract in sync with
# test_common.session_reuse_hooks._PRIVATE_NODEID_PATTERNS.
def _get_default_torch_compile_config(torch_compile):
    return TorchCompileConfig(enable_fullgraph=True,
                              enable_piecewise_cuda_graph=True,
                              capture_num_tokens=[2048, 8192],
                              max_num_streams=3) if torch_compile else None


def _deepseek_v3_mtp_config(mtp_nextn: int,
                            moe_tactic: str) -> Optional[MTPDecodingConfig]:
    """MTP config for a DeepSeek-V3 case that pins the target by moe_tactic.

    The target's tactic does not reach the MTP layers, whose experts are
    BF16, so the draft pins the same tactic. cutedsl.grouped_gemm is the
    exception: CuTe DSL's BF16 grouped GEMM runs on SM107 only, so those
    cases leave the MTP layers on AUTO, which is CUTLASS for DeepSeek-V3.

    Args:
        mtp_nextn: Draft length; 0 disables MTP.
        moe_tactic: The target's moe_config.tactic.

    Returns:
        The MTP config, or None when mtp_nextn is 0.
    """
    if mtp_nextn == 0:
        return None
    return MTPDecodingConfig(
        max_draft_len=mtp_nextn,
        moe_tactic=None if moe_tactic == "cutedsl.grouped_gemm" else moe_tactic)


def _latest_kv_cache_stats(llm):
    """Cumulative KV cache stats, warmup-adjusted, from the last iteration.

    Requires the LLM to be built with enable_iter_perf_stats=True; without it
    the reuse counters are never populated.
    """
    entries = [
        s["kvCacheStats"] for s in llm.get_stats(timeout=5)
        if s.get("kvCacheStats")
    ]
    assert entries, "No kvCacheStats reported; is enable_iter_perf_stats set?"
    return entries[-1]


def _run_multinode_accuracy(model_path,
                            model_name,
                            *,
                            benchmarks,
                            tp_size=2,
                            pp_size=1,
                            ep_size=1,
                            draft_model_path=None,
                            max_draft_len=2,
                            kv_cache_config=None,
                            max_num_tokens=4096,
                            max_batch_size=1,
                            **llm_kwargs):
    benchmark_task_map = {
        "mmlu": MMLU,
        "gsm8k": GSM8K,
    }
    if kv_cache_config is None:
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5,
                                        enable_block_reuse=draft_model_path
                                        is None)
    spec_config = None
    if draft_model_path is not None:
        spec_config = Eagle3DecodingConfig(max_draft_len=max_draft_len,
                                           speculative_model=draft_model_path)

    with LLM(model_path,
             tensor_parallel_size=tp_size,
             pipeline_parallel_size=pp_size,
             moe_expert_parallel_size=ep_size,
             max_num_tokens=max_num_tokens,
             max_batch_size=max_batch_size,
             kv_cache_config=kv_cache_config,
             speculative_config=spec_config,
             **llm_kwargs) as llm:
        for benchmark in benchmarks:
            task_cls = benchmark_task_map.get(benchmark)
            if task_cls is None:
                raise ValueError(f"Unsupported benchmark: {benchmark}")
            task = task_cls(model_name)
            task.evaluate(llm)


class TestMinistral8BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
    MODEL_PATH = f"{llm_models_root()}/Ministral-8B-Instruct-2410"

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    def test_fp8(self):
        # Test with FP8 quantization if pre-quantized model is available
        model_path = f"{llm_models_root()}/Ministral-8B-Instruct-2410-FP8"
        try:
            with LLM(model_path) as llm:
                assert llm.args.quant_config.quant_algo == QuantAlgo.FP8
                task = GSM8K(self.MODEL_NAME)
                task.evaluate(llm)
                task = MMLU(self.MODEL_NAME)
                task.evaluate(llm)
        except (FileNotFoundError, OSError):
            pytest.skip("FP8 pre-quantized Ministral-8B model not available")


@skip_pre_hopper
class TestDeepSeekV3Lite(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V3-Lite"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"

    @skip_pre_blackwell
    @pytest.mark.skip_less_device_memory(60000)
    def test_prims_ts_bfloat16(self, mocker):
        if get_sm_version() not in (100, 103):
            pytest.skip("PrimTS requires SM100 or SM103")

        calls = _count_prims_ts_phase_calls(mocker)
        env = {
            "TLLM_FMHA_LIBS": "+prims_ts",
            "TLLM_WORKER_USE_SINGLE_PROCESS": "1",
        }
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=0.75,
            tokens_per_block=32,
        )
        # Keep the TP=1 worker in this process so the call counter observes the
        # real PrimTS launch. Compile Inductor kernels synchronously because its
        # process-global async reader thread otherwise outlives this test and is
        # reported as a leak; persistent compiler caches remain enabled.
        with (inductor_config.patch(compile_threads=1),
              mock.patch.dict(os.environ, env)):
            with LLM(self.MODEL_PATH,
                     attn_backend="TRTLLM",
                     kv_cache_config=kv_cache_config,
                     disable_overlap_scheduler=True,
                     enable_chunked_prefill=False,
                     enable_attention_dp=False,
                     cuda_graph_config=None,
                     speculative_config=None,
                     max_batch_size=1350) as llm:
                calls.update({name: 0 for name in calls})
                task = GSM8K(self.MODEL_NAME)
                task.evaluate(llm)

        assert calls["mla_generation"] > 0

    @pytest.mark.skip_less_device_memory(60000)
    @parametrize_with_ids("v2_kv_cache", [True, False])
    # Chunked Prefill for MLA can only be enabled on SM100
    @parametrize_with_ids("enable_chunked_prefill", [False, True])
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, False, False),
                           (False, True, False), (False, False, True),
                           (False, True, True), (True, True, True)])
    # Only Hopper and Blackwell MLA kernel supports MTP
    @parametrize_with_ids("mtp_nextn", [0, 2])
    def test_bfloat16(self, mtp_nextn, attention_dp, cuda_graph,
                      overlap_scheduler, torch_compile, enable_chunked_prefill,
                      v2_kv_cache):
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=0.75,
            use_kv_cache_manager_v2=v2_kv_cache,
        )
        torch_compile_config = _get_default_torch_compile_config(torch_compile)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
        )
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(max_draft_len=mtp_nextn)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 enable_chunked_prefill=enable_chunked_prefill,
                 max_num_tokens=256 if enable_chunked_prefill else 8192,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config,
                 max_batch_size=1350,
                 max_stats_len=-1,
                 enable_iter_perf_stats=True) as llm:  # GSM8K has 1319 requests
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            if mtp_nextn > 0:
                acceptance_length = compute_acceptance_length(llm)
                print(f"[AL] test_bfloat16 mtp_nextn={mtp_nextn} "
                      f"acceptance_length = {acceptance_length:.3f}")
                assert_acceptance_length(
                    "TestDeepSeekV3Lite::test_bfloat16",
                    acceptance_length,
                )

    @pytest.mark.skip_less_device_memory(60000)
    @pytest.mark.parametrize("enable_chunked_prefill", [False, True],
                             ids=["unchunked", "chunked"])
    def test_nvfp4_mla(self, monkeypatch: pytest.MonkeyPatch,
                       enable_chunked_prefill: bool) -> None:
        from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
            _fp4_mla_attention_backend, fp4_mla_generation_backend_supported)
        if not fp4_mla_generation_backend_supported(
                get_sm_version(), _fp4_mla_attention_backend()):
            pytest.skip("requires a supported FP4 MLA generation backend")
        from tensorrt_llm._torch.attention.backends.fmha.fp4_mla import \
            Fp4MlaFmha
        from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import \
            Fp4MlaKVCacheManagerV2
        from tensorrt_llm._torch.attention.backends.trtllm import \
            TrtllmAttention
        from tensorrt_llm._torch.attention.mla import MLA

        # Inspect the actual worker's quantization, cache pools and FMHA choice.
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        kv_cache_config = KvCacheConfig(dtype="nvfp4",
                                        free_gpu_memory_fraction=0.75,
                                        use_kv_cache_manager_v2=True,
                                        enable_block_reuse=False)
        with LLM(
                f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only",
                attn_backend="TRTLLM",
                kv_cache_config=kv_cache_config,
                enable_chunked_prefill=enable_chunked_prefill,
                cuda_graph_config=CudaGraphConfig(),
                autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                max_num_tokens=8192,
                max_batch_size=1350,
        ) as llm:
            executor = llm._executor.engine
            quant_config = executor.model_engine.model.model_config.quant_config
            assert quant_config.quant_algo == QuantAlgo.NVFP4
            assert quant_config.quant_mode.has_fp4_kv_cache()
            manager = executor.kv_cache_manager
            assert isinstance(manager, Fp4MlaKVCacheManagerV2)
            assert manager.fp4_mla_layout.get_mla_v_scale_pool().numel() > 0
            assert manager.fp4_mla_layout.get_fp4_mla_hp_pool().numel() > 0
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            mla_layers = [
                module.mqa for module in executor.model_engine.model.modules()
                if isinstance(module, MLA)
            ]
            assert mla_layers
            for layer in mla_layers:
                assert isinstance(layer, TrtllmAttention)
                assert layer.has_fp4_kv_cache
                assert any(
                    isinstance(fmha, Fp4MlaFmha)
                    for fmha in layer._fmha_manager._cache.values())

    @skip_no_rubin
    @pytest.mark.skip_less_device_memory(60000)
    @parametrize_with_ids("enable_block_reuse,enable_chunked_prefill,mtp_nextn",
                          [(True, False, 0), (False, True, 0), (True, True, 0),
                           (True, True, 3)])
    def test_nvfp4_mla_cached_context(self, monkeypatch,
                                      enable_block_reuse: bool,
                                      enable_chunked_prefill: bool,
                                      mtp_nextn: int):
        """FP4 MLA context attention over a prefix already in the FP4 cache.

        GSM8K prompts share their few-shot examples, so with block reuse most
        requests reuse that prefix; a 512-token budget splits every prompt
        into several chunked-prefill chunks.

        Decode runs with the 4x path dispatcher on, so the decoder CUDA graphs
        are keyed by KV regime. The 16 heads of DeepSeek-V3-Lite fill no
        256-row 4x item, so every graph shape keeps one regime and the
        dispatcher keeps the 2x kernel.
        """
        # env_overrides also sets it in this process; monkeypatch removes it
        # from this process once the test ends.
        env = {"TRTLLM_FP4_MLA_CUTEDSL_4X": "1"}
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        kv_cache_config = KvCacheConfig(dtype="nvfp4",
                                        free_gpu_memory_fraction=0.75,
                                        use_kv_cache_manager_v2=True,
                                        enable_block_reuse=enable_block_reuse)
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(max_draft_len=mtp_nextn)
        with LLM(
                f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only_mtp",
                attn_backend="TRTLLM",
                env_overrides=env,
                kv_cache_config=kv_cache_config,
                enable_chunked_prefill=enable_chunked_prefill,
                max_num_tokens=512 if enable_chunked_prefill else 8192,
                speculative_config=mtp_config,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device_memory(60000)
    @parametrize_with_ids("enable_chunked_prefill", [False, True])
    def test_bfloat16_flashinfer(self, enable_chunked_prefill):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 attn_backend="FLASHINFER",
                 enable_chunked_prefill=enable_chunked_prefill,
                 max_num_tokens=512 if enable_chunked_prefill else 8192) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device_memory(60000)
    @parametrize_with_ids("enable_chunked_prefill", [False, True])
    @parametrize_with_ids("attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, True, True)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    def test_bfloat16_python_scheduler(self, mtp_nextn, attention_dp,
                                       cuda_graph, overlap_scheduler,
                                       enable_chunked_prefill):
        scheduler_config = SchedulerConfig(use_python_scheduler=True)
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6, )
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            scheduler_config=scheduler_config,
        )
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(max_draft_len=mtp_nextn)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 enable_chunked_prefill=enable_chunked_prefill,
                 max_num_tokens=256 if enable_chunked_prefill else 8192,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device_memory(60000)
    def test_bfloat16_mtp_sa(self):
        """Accuracy test for MTP + Suffix Automaton (MTP+SA) speculative decoding."""
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(),
        )
        mtp_config = MTPDecodingConfig(max_draft_len=2, use_sa_spec=True)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 max_num_tokens=8192,
                 **pytorch_config,
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_acc_spec="use_sa_spec")

    @pytest.mark.skip_less_device_memory(60000)
    def test_bfloat16_mtp_sa_global_pool(self):
        """Accuracy test for MTP + Suffix Automaton with global pool enabled."""
        max_batch_size = 32
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        pytorch_config = dict(
            max_batch_size=max_batch_size,
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=max_batch_size,
                                              enable_padding=True),
        )
        mtp_config = MTPDecodingConfig(
            max_draft_len=2,
            sa_config=SAEnhancerConfig(enable_global_pool=True))
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 max_num_tokens=8192,
                 **pytorch_config,
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_acc_spec="use_sa_spec")

    @skip_pre_blackwell
    @pytest.mark.skip_less_device_memory(60000)
    def test_mtp_dynamic_draft_len(self):
        draft_len_schedule = {50: 4, 200: 3, 350: 2}
        max_draft_len = 4
        cuda_graph_config = CudaGraphConfig(max_batch_size=500)
        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=cuda_graph_config,
        )
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        mtp_config = MTPDecodingConfig(
            num_nextn_predict_layers=max_draft_len,
            max_draft_len=max_draft_len,
            draft_len_schedule=draft_len_schedule,
        )
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 enable_chunked_prefill=False,
                 max_num_tokens=8192,
                 **pytorch_config,
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device_memory(60000)
    def test_mtp_sa_dynamic_draft_len(self):
        """Accuracy test for MTP + SA with dynamic draft length."""
        draft_len_schedule = {50: 4, 200: 3, 350: 2}
        max_draft_len = 4
        cuda_graph_config = CudaGraphConfig(max_batch_size=500)
        pytorch_config = dict(
            max_batch_size=500,
            disable_overlap_scheduler=False,
            cuda_graph_config=cuda_graph_config,
        )
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        mtp_config = MTPDecodingConfig(
            num_nextn_predict_layers=max_draft_len,
            max_draft_len=max_draft_len,
            sa_config=SAEnhancerConfig(enable_global_pool=True),
            draft_len_schedule=draft_len_schedule,
        )
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 enable_chunked_prefill=False,
                 max_num_tokens=8192,
                 **pytorch_config,
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_acc_spec="use_sa_spec")

    @skip_pre_blackwell
    @pytest.mark.skip_less_device_memory(60000)
    def test_mtp_eagle_dynamic_draft_len(self):
        draft_len_schedule = {50: 4, 200: 3, 350: 2}
        max_draft_len = 4
        cuda_graph_config = CudaGraphConfig(max_batch_size=500)
        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=cuda_graph_config,
        )
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        # Force MTP-Eagle one-model path.
        mtp_config = MTPDecodingConfig(
            num_nextn_predict_layers=max_draft_len,
            max_draft_len=max_draft_len,
            use_mtp_vanilla=False,
            draft_len_schedule=draft_len_schedule,
        )
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_cache_config,
                 enable_chunked_prefill=False,
                 max_num_tokens=8192,
                 **pytorch_config,
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @parametrize_with_ids("mtp_nextn", [0, 2])
    def test_bfloat16_4gpus_kv_cache_aware_routing(self, mtp_nextn):
        """Accuracy test for attention DP with KV cache-aware routing."""
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75,
                                        enable_block_reuse=True)
        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(),
        )
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(max_draft_len=mtp_nextn)
        attention_dp_config = AttentionDpConfig(
            enable_kv_cache_aware_routing=True, )
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=True,
                 attention_dp_config=attention_dp_config,
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, False, False),
                           (False, True, False), (False, False, True),
                           (False, True, True), (True, True, True)])
    # Only Hopper and Blackwell MLA kernel supports MTP
    @parametrize_with_ids("mtp_nextn",
                          [0, pytest.param(2, marks=skip_pre_hopper)])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4),
                                                         (2, 2, 1), (1, 4, 1)],
                             ids=["tp4", "ep4", "tp2pp2", "pp4"])
    def test_bfloat16_4gpus(self, request: pytest.FixtureRequest,
                            monkeypatch: pytest.MonkeyPatch, tp_size, pp_size,
                            ep_size, mtp_nextn, attention_dp, cuda_graph,
                            overlap_scheduler, torch_compile):
        if request.node.callspec.id == _DENSE_PRIMS_TS_BF16_4GPUS_ID:
            _enable_dense_prims_ts(monkeypatch)
        if pp_size > 1 and mtp_nextn > 0:
            num_hidden_layers = 30
            pp_partition = [num_hidden_layers // pp_size + 1] * pp_size
            pp_partition[-1] = num_hidden_layers - sum(pp_partition[:-1])
        else:
            pp_partition = None
        free_gpu_memory_fraction = 0.6 if torch_compile else 0.75
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=free_gpu_memory_fraction)
        torch_compile_config = _get_default_torch_compile_config(torch_compile)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
        )
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(max_draft_len=mtp_nextn)
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 pp_partition=pp_partition,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 autotuner_nvmmh_config=(AutoTunerNvMMHConfig()
                                         if pp_size > 1 else None),
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @parametrize_with_ids("mtp_nextn",
                          [0, pytest.param(2, marks=skip_pre_hopper)])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4)],
                             ids=["tp4", "ep4"])
    def test_bfloat16_4gpus_python_scheduler(self, tp_size, pp_size, ep_size,
                                             mtp_nextn):
        scheduler_config = SchedulerConfig(use_python_scheduler=True)
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        pytorch_config = dict(cuda_graph_config=CudaGraphConfig(), )
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(max_draft_len=mtp_nextn)
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 scheduler_config=scheduler_config,
                 **pytorch_config,
                 enable_attention_dp=True,
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, False, False),
                           (False, True, False, False),
                           (False, False, True, False),
                           (False, False, False, True),
                           (True, False, True, True), (True, True, True, True)])
    @parametrize_with_ids("mtp", ["disable", "eagle", "vanilla"])
    def test_fp8_block_scales(self, mtp, fp8kv, attention_dp, cuda_graph,
                              overlap_scheduler, torch_compile):
        if torch_compile and mtp != "disable":
            pytest.skip("https://nvbugs/5252313")
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        torch_compile_config = _get_default_torch_compile_config(torch_compile)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=fp8_block_scales_moe_config(),
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        mtp_nextn = 2
        if mtp == "eagle":
            mtp_config = MTPDecodingConfig(
                max_draft_len=mtp_nextn,
                moe_tactic=fp8_block_scales_moe_tactic())
        elif mtp == "vanilla":
            mtp_config = MTPDecodingConfig(
                max_draft_len=mtp_nextn,
                use_mtp_vanilla=True,
                moe_tactic=fp8_block_scales_moe_tactic())

        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:

            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("cuda_graph", [False, True])
    def test_cute_dsl_bf16_bmm(self, cuda_graph):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)
        pytorch_config = dict(
            disable_overlap_scheduler=True,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            use_cute_dsl_bf16_bmm=True,
        )

        with LLM(
                self.MODEL_PATH,
                kv_cache_config=kv_cache_config,
                **pytorch_config,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("cuda_graph", [False, True])
    def test_cute_dsl_bf16_gemm(self, cuda_graph):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)
        pytorch_config = dict(
            disable_overlap_scheduler=True,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            use_cute_dsl_bf16_gemm=True,
        )

        with LLM(
                self.MODEL_PATH,
                kv_cache_config=kv_cache_config,
                **pytorch_config,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @parametrize_with_ids("mtp_nextn", [0, 2])
    def test_fp8_block_scales_cuda_graph_padding(self, mtp_nextn):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(
                max_draft_len=mtp_nextn,
                moe_tactic=fp8_block_scales_moe_tactic())
        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(
                max_batch_size=512,
                enable_padding=True,
            ),
            moe_config=fp8_block_scales_moe_config(),
        )
        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_hopper
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @parametrize_with_ids("attention_dp", [False, True])
    def test_fp8_block_scales_cuda_graph_padding_4gpus(self, mtp_nextn,
                                                       attention_dp):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(
                max_draft_len=mtp_nextn,
                moe_tactic=fp8_block_scales_moe_tactic())
        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(enable_padding=True),
            moe_config=fp8_block_scales_moe_config(),
        )

        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                 tensor_parallel_size=4,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_hopper
    @skip_ray
    @parametrize_with_ids("sampler_async_worker", [True, False])
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, False, False),
                           (False, True, False, False),
                           (False, False, True, False),
                           (False, False, False, True),
                           (False, True, True, True), (True, False, True, True),
                           (True, True, True, True)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4),
                                                         (2, 2, 1), (1, 4, 1)],
                             ids=["tp4", "ep4", "tp2pp2", "pp4"])
    def test_fp8_block_scales_4gpus(self, tp_size, pp_size, ep_size, mtp_nextn,
                                    fp8kv, attention_dp, cuda_graph,
                                    overlap_scheduler, torch_compile,
                                    sampler_async_worker):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)
        torch_compile_config = _get_default_torch_compile_config(torch_compile)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=fp8_block_scales_moe_config(),
            sampler_force_async_worker=sampler_async_worker,
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(
                max_draft_len=mtp_nextn,
                moe_tactic=fp8_block_scales_moe_tactic())

        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:

            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_blackwell
    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4)],
                             ids=["tp4", "ep4"])
    def test_cute_dsl_bf16_bmm_4gpus(self, tp_size, pp_size, ep_size,
                                     cuda_graph):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)
        pytorch_config = dict(
            disable_overlap_scheduler=True,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            use_cute_dsl_bf16_bmm=True,
        )

        with LLM(
                self.MODEL_PATH,
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                kv_cache_config=kv_cache_config,
                **pytorch_config,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_blackwell
    @parametrize_with_ids("cuda_graph", [False, True])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4)],
                             ids=["tp4", "ep4"])
    def test_cute_dsl_bf16_gemm_4gpus(self, tp_size, pp_size, ep_size,
                                      cuda_graph):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)
        pytorch_config = dict(
            disable_overlap_scheduler=True,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            use_cute_dsl_bf16_gemm=True,
        )

        with LLM(
                self.MODEL_PATH,
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                kv_cache_config=kv_cache_config,
                **pytorch_config,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(60000)
    def test_fp8_block_scales_agg_helix_tp2cp2(self) -> None:
        """GSM8K on aggregated Helix (TP2 x CP2) with FP8 KV and chunked prefill.

        Aggregated because no cache transceiver is configured: one executor
        runs both the context and generation phases on the CP ranks.
        The chunk-aware MLA profile needs a request of 9 x max_num_tokens
        tokens below max_seq_len, which the model caps at 4096.
        max_num_tokens=256 gives a 2304-token profile, so the KV estimation
        measures cached-KV chunk staging instead of subtracting a context-MLA
        workspace reserve.
        """
        with LLM(
                f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                tensor_parallel_size=2,
                context_parallel_size=2,
                moe_expert_parallel_size=2,
                cp_config={
                    "cp_type": "HELIX",
                    "tokens_per_block": 64
                },
                max_batch_size=32,
                max_num_tokens=256,
                max_seq_len=4096,
                enable_chunked_prefill=True,
                kv_cache_config=KvCacheConfig(
                    dtype="fp8",
                    enable_block_reuse=False,
                    tokens_per_block=64,
                    use_kv_cache_manager_v2=True,
                ),
        ) as llm:
            transceiver = llm.args.cache_transceiver_config
            assert transceiver is None or transceiver.backend is None
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_hopper
    def test_fp8_block_scales_4gpus_static_eplb(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)

        num_experts = 72
        num_slots = 80
        first_k_dense_replace = 1
        num_hidden_layers = 30
        initial_global_assignments = {}
        for i in range(first_k_dense_replace, num_hidden_layers):
            initial_global_assignments[i] = [(i + j) % num_experts
                                             for j in range(num_slots)]
        eplb_config = MoeLoadBalancerConfig(
            num_slots=num_slots,
            initial_global_assignments=initial_global_assignments,
            layer_updates_per_iter=0)
        pytorch_backend_options = dict(
            cuda_graph_config=CudaGraphConfig(),
            moe_config=fp8_block_scales_moe_config(load_balancer=eplb_config))
        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 kv_cache_config=kv_cache_config,
                 **pytorch_backend_options,
                 enable_attention_dp=True) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_device_not_contain(["GB200"])
    @parametrize_with_ids("moe_tactic", ["cutlass", "trtllm_gen"])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    def test_bfloat16_4gpus_online_eplb(self, moe_tactic, mtp_nextn):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)
        num_slots = 80
        eplb_config = MoeLoadBalancerConfig(num_slots=num_slots,
                                            layer_updates_per_iter=2)
        pytorch_config = dict(cuda_graph_config=CudaGraphConfig(),
                              moe_config=MoeConfig(tactic=moe_tactic,
                                                   load_balancer=eplb_config))
        mtp_config = _deepseek_v3_mtp_config(mtp_nextn, moe_tactic)
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 kv_cache_config=kv_cache_config,
                 enable_attention_dp=True,
                 **pytorch_config,
                 speculative_config=mtp_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_device_not_contain(["GB200"])
    @parametrize_with_ids("moe_tactic", ["trtllm_gen"])
    @parametrize_with_ids("fp8kv", [True, False])
    def test_nvfp4_4gpus_online_eplb(self, moe_tactic, fp8kv):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)
        num_slots = 80
        eplb_config = MoeLoadBalancerConfig(num_slots=num_slots,
                                            layer_updates_per_iter=2)
        pytorch_config = dict(cuda_graph_config=CudaGraphConfig(),
                              moe_config=MoeConfig(tactic=moe_tactic,
                                                   load_balancer=eplb_config))
        if fp8kv:
            kv_cache_config.dtype = "fp8"

        with LLM(
                f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only",
                tensor_parallel_size=4,
                moe_expert_parallel_size=4,
                kv_cache_config=kv_cache_config,
                **pytorch_config,
                enable_attention_dp=True,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, False, False),
                           (False, True, False, False),
                           (False, False, True, False),
                           (False, False, False, True),
                           (True, False, True, True), (True, True, True, True)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @parametrize_with_ids("moe_tactic",
                          ["cutlass", "trtllm_gen", "cutedsl.grouped_gemm"])
    def test_nvfp4(self, fp8kv, attention_dp, cuda_graph, overlap_scheduler,
                   torch_compile, mtp_nextn, moe_tactic):
        skip_unless_moe_tactic_runs_here(moe_tactic)

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        torch_compile_config = _get_default_torch_compile_config(torch_compile)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=MoeConfig(tactic=moe_tactic))
        mtp_config = _deepseek_v3_mtp_config(mtp_nextn, moe_tactic)

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only_mtp",
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids(
        "fp8kv,attention_dp,cuda_graph,overlap_scheduler",
        [(False, False, False, False)],
    )
    def test_cute_dsl_nvfp4(
        self,
        fp8kv,
        attention_dp,
        cuda_graph,
        overlap_scheduler,
        torch_compile,
    ):
        """Test NVFP4 with CuTe DSL blockscaling mm (GEMM+SwiGLU fusion for shared experts)."""
        sm_version = get_sm_version()
        if not is_sm_100f(sm_version):
            pytest.skip("CuTe DSL blockscaling mm only supports SM 100 family")

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)
        torch_compile_config = _get_default_torch_compile_config(torch_compile)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=MoeConfig(tactic="cutedsl.grouped_gemm"),
            use_cute_dsl_blockscaling_mm=True,
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        with LLM(
                f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only",
                kv_cache_config=kv_cache_config,
                **pytorch_config,
                enable_attention_dp=attention_dp,
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_blackwell
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids(
        "fp8kv,attention_dp,cuda_graph,overlap_scheduler",
        [(False, False, False, False)],
    )
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size",
        [(4, 1, 1), (4, 1, 4), (2, 2, 1), (1, 4, 1)],
        ids=["tp4", "ep4", "tp2pp2", "pp4"],
    )
    def test_cute_dsl_nvfp4_4gpus(
        self,
        tp_size,
        pp_size,
        ep_size,
        fp8kv,
        attention_dp,
        cuda_graph,
        overlap_scheduler,
        torch_compile,
    ):
        """Test NVFP4 4 GPUs with CuTe DSL blockscaling mm (GEMM+SwiGLU fusion for shared experts)."""
        sm_version = get_sm_version()
        if not is_sm_100f(sm_version):
            pytest.skip("CuTe DSL blockscaling mm only supports SM 100 family")

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)
        torch_compile_config = _get_default_torch_compile_config(torch_compile)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=MoeConfig(tactic="cutedsl.grouped_gemm"),
            use_cute_dsl_blockscaling_mm=True,
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        with LLM(
                f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only",
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                kv_cache_config=kv_cache_config,
                **pytorch_config,
                enable_attention_dp=attention_dp,
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @parametrize_with_ids("v2_kv_cache", [False, True])
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids("fp8kv,cuda_graph,overlap_scheduler",
                          [(False, False, False), (True, True, True)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @parametrize_with_ids(
        "batch_wait_timeout_iters,batch_wait_max_tokens_ratio", [(0, 0),
                                                                 (10, 1.0),
                                                                 (10, 0),
                                                                 (0, 0.75)])
    def test_nvfp4_batch_waiting(self, torch_compile, fp8kv, cuda_graph,
                                 overlap_scheduler, mtp_nextn,
                                 batch_wait_timeout_iters,
                                 batch_wait_max_tokens_ratio, v2_kv_cache):
        moe_tactic = "cutlass"
        # V2 KV cache manager auto-selects the V2 scheduler, which is the
        # path that grows ctx KV during scheduling and needs the
        # delay-batching ctx revert to release pages during the wait window.
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75,
                                        use_kv_cache_manager_v2=v2_kv_cache)
        torch_compile_config = _get_default_torch_compile_config(torch_compile)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            batch_wait_timeout_iters=batch_wait_timeout_iters,
            batch_wait_max_tokens_ratio=batch_wait_max_tokens_ratio,
            moe_config=MoeConfig(tactic=moe_tactic))
        mtp_config = _deepseek_v3_mtp_config(mtp_nextn, moe_tactic)
        if fp8kv:
            kv_cache_config.dtype = "fp8"
        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only_mtp",
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=False,
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_no_rubin
    @parametrize_with_ids("uses_spcompress", [False, True])
    def test_spcompress_attention(self, uses_spcompress: bool):
        sparse_attention_config = None
        if uses_spcompress:
            sparse_attention_config = SkipSoftmaxAttentionConfig(
                threshold_scale_factor={
                    "prefill": 0,
                    "decode": 0
                },
                uses_spcompress=uses_spcompress)

        kv_cache_config = KvCacheConfig(dtype="fp8",
                                        free_gpu_memory_fraction=0.75)

        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only_mtp",
                 attn_backend="TRTLLM",
                 kv_cache_config=kv_cache_config,
                 sparse_attention_config=sparse_attention_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_device(4)
    @skip_pre_blackwell
    @parametrize_with_ids("torch_compile", [False, True])
    @parametrize_with_ids(
        "fp8kv,attention_dp,cuda_graph,overlap_scheduler,low_precision_combine",
        [
            (False, False, False, False, False),
            (True, False, False, False, False),
            (False, True, False, False, False),
            (False, False, True, False, False),
            (False, False, False, True, False),
            (True, False, True, True, False),
            (True, True, True, True, False),
            (False, True, False, False, True),
            (True, True, True, True, True),
        ])
    @pytest.mark.parametrize("tp_size,pp_size,ep_size", [(4, 1, 1), (4, 1, 4),
                                                         (2, 2, 1), (1, 4, 1)],
                             ids=["tp4", "ep4", "tp2pp2", "pp4"])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @parametrize_with_ids("moe_tactic",
                          ["cutlass", "trtllm_gen", "cutedsl.grouped_gemm"])
    def test_nvfp4_4gpus(self, fp8kv, attention_dp, cuda_graph,
                         overlap_scheduler, low_precision_combine, tp_size,
                         pp_size, ep_size, torch_compile, mtp_nextn,
                         moe_tactic):
        # create_strategy passes use_low_precision_moe_combine to
        # NVLinkOneSided, NVLinkTwoSided, NcclEP, and DeepEPLowLatency. It
        # returns None or AllGatherReduceScatter, without the flag, when
        # attention DP is off or dp_size == 1, moe_tp_size != 1, or the tactic
        # disallows all-to-all. dp_size is tp_size under attention_dp, and
        # moe_tp_size is tp_size // ep_size.
        tactic_allows_alltoall = moe_tactic != "cutlass"
        if moe_tactic == "cutlass":
            from tensorrt_llm._torch.moe.fused_moe.impl_environment import \
                collect_moe_environment

            # Cutlass NVFP4 issues the fused_moe op, which rejects all-to-all
            # unless the build enables the OSS Cutlass MoE GEMM.
            tactic_allows_alltoall = collect_moe_environment().oss_cutlass_moe
        if low_precision_combine and not (attention_dp and tp_size > 1
                                          and tp_size == ep_size
                                          and tactic_allows_alltoall):
            pytest.skip(
                "use_low_precision_moe_combine is unused unless attention_dp, "
                "dp_size > 1, moe_tp_size == 1, and the tactic allows all-to-all"
            )
        skip_unless_moe_tactic_runs_here(moe_tactic)

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        # Picewise Cuda Graph cannot be enabled for nvfp4 attention dp.
        torch_compile_config = _get_default_torch_compile_config(torch_compile)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=MoeConfig(
                tactic=moe_tactic,
                use_low_precision_moe_combine=low_precision_combine,
            ),
        )

        mtp_config = _deepseek_v3_mtp_config(mtp_nextn, moe_tactic)

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        with LLM(
                f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only_mtp",
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                kv_cache_config=kv_cache_config,
                **pytorch_config,
                enable_attention_dp=attention_dp,
                autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                speculative_config=mtp_config,
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skipif(get_sm_version() != 107,
                        reason="fine-grained sync requires SM107")
    @parametrize_with_ids("torch_compile", [False])
    @parametrize_with_ids("fp8kv,attention_dp,cuda_graph,overlap_scheduler",
                          [(False, False, False, False),
                           (True, False, True, True), (True, True, True, True)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @parametrize_with_ids("moe_tactic", ["trtllm_gen"])
    @parametrize_with_ids("enable_autotuner", [False, True])
    def test_nvfp4_fine_grained_sync(self, fp8kv: bool, attention_dp: bool,
                                     cuda_graph: bool, overlap_scheduler: bool,
                                     torch_compile: bool, mtp_nextn: int,
                                     moe_tactic: str,
                                     enable_autotuner: bool) -> None:
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        torch_compile_config = _get_default_torch_compile_config(torch_compile)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            torch_compile_config=torch_compile_config,
            moe_config=MoeConfig(tactic=moe_tactic))
        mtp_config = _deepseek_v3_mtp_config(mtp_nextn, moe_tactic)

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        with LLM(f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only_mtp",
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 enable_autotuner=enable_autotuner,
                 use_fine_grained_sync=True,
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids(
        "fp8kv,attention_dp,cuda_graph,overlap_scheduler",
        [(False, False, False, False),
         pytest.param(True, False, False, False, marks=skip_no_hopper),
         (False, True, False, False), (False, False, True, False),
         (False, False, False, True), (False, True, True, True),
         pytest.param(True, True, True, True, marks=skip_no_hopper)])
    @parametrize_with_ids("mtp_nextn", [0, 2])
    @parametrize_with_ids("quant_dtype", [
        pytest.param("none", marks=skip_pre_hopper),
        pytest.param("fp8", marks=skip_no_hopper),
        pytest.param("nvfp4", marks=skip_pre_blackwell)
    ])
    def test_no_kv_cache_reuse(self, quant_dtype, mtp_nextn, fp8kv,
                               attention_dp, cuda_graph, overlap_scheduler):
        if quant_dtype == "nvfp4" and mtp_nextn > 0:
            pytest.skip("MTP is not supported for NVFP4")

        model_path = self.MODEL_PATH
        if quant_dtype == "fp8":
            model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/fp8"
        elif quant_dtype == "nvfp4":
            model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only"

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75,
                                        enable_block_reuse=False)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
        )
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(max_draft_len=mtp_nextn)

        if quant_dtype == "none":
            assert not fp8kv
        else:
            if fp8kv:
                kv_cache_config.dtype = "fp8"

        with LLM(model_path,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:
            if quant_dtype == "fp8":
                assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
            elif quant_dtype == "nvfp4":
                assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids("fp8kv,overlap_scheduler", [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ])
    @parametrize_with_ids("kv_cache_reuse", [True, False])
    @parametrize_with_ids("quant_dtype", [
        pytest.param("none", marks=skip_pre_hopper),
        pytest.param("fp8", marks=skip_pre_hopper),
        pytest.param("nvfp4", marks=skip_pre_blackwell)
    ])
    # currently, chunked prefill is not supported for fp8 and nvfp4
    def test_chunked_prefill(self, quant_dtype, kv_cache_reuse, fp8kv,
                             overlap_scheduler):
        model_path = self.MODEL_PATH
        if quant_dtype == "fp8":
            model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/fp8"
        elif quant_dtype == "nvfp4":
            model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/nvfp4_moe_only"

        if quant_dtype == "none" and fp8kv:
            pytest.skip("only fp8 and nvfp4 support fp8 kv cache")

        assert_reuse = (quant_dtype == "none" and kv_cache_reuse
                        and overlap_scheduler)
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=0.6,
            enable_block_reuse=kv_cache_reuse,
            # ADP pads an idle rank with a short dummy request; don't commit
            # that partial block while asserting exact reused-block counts.
            **({
                "enable_partial_reuse": False
            } if assert_reuse else {}),
        )
        pytorch_config = dict(disable_overlap_scheduler=not overlap_scheduler, )
        if quant_dtype == "fp8":
            pytorch_config["moe_config"] = fp8_block_scales_moe_config()
        mtp_config = None

        if quant_dtype == "none":
            assert not fp8kv
        else:
            if fp8kv:
                kv_cache_config.dtype = "fp8"

        with LLM(model_path,
                 kv_cache_config=kv_cache_config,
                 enable_chunked_prefill=True,
                 max_num_tokens=512,
                 **pytorch_config,
                 enable_attention_dp=True,
                 autotuner_nvmmh_config=(AutoTunerNvMMHConfig()
                                         if quant_dtype == "nvfp4" else None),
                 speculative_config=mtp_config) as llm:

            if quant_dtype == "fp8":
                assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
            elif quant_dtype == "nvfp4":
                assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            if assert_reuse:
                assert_kv_cache_reuse_for_llm(llm, [1] + [42] * 255)

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    def _evaluate_guided_decoding(self, backend: str, mtp_nextn: int, mocker,
                                  **llm_kwargs):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        cuda_graph_config = CudaGraphConfig(enable_padding=True)
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(max_draft_len=mtp_nextn)
        llm = LLM(self.MODEL_PATH,
                  guided_decoding_backend=backend,
                  kv_cache_config=kv_cache_config,
                  cuda_graph_config=cuda_graph_config,
                  enable_chunked_prefill=True,
                  max_num_tokens=256,
                  speculative_config=mtp_config,
                  **llm_kwargs)
        with llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    @parametrize_with_ids("mtp_nextn",
                          [0, pytest.param(2, marks=skip_pre_hopper)])
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding(self, backend: str, mtp_nextn: int, mocker):
        self._evaluate_guided_decoding(backend, mtp_nextn, mocker)

    @skip_pre_hopper
    def test_guided_decoding_breakable_prefill_cuda_graph(self, mocker):
        self._evaluate_guided_decoding(
            "xgrammar",
            2,
            mocker,
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE)

    @pytest.mark.skip_less_device(4)
    @parametrize_with_ids("mtp_nextn",
                          [0, pytest.param(2, marks=skip_pre_hopper)])
    @pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
    def test_guided_decoding_4gpus(self, backend: str, mtp_nextn: int, mocker):
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75)
        cuda_graph_config = CudaGraphConfig(enable_padding=True)
        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(max_draft_len=mtp_nextn)
        llm = LLM(self.MODEL_PATH,
                  tensor_parallel_size=4,
                  moe_expert_parallel_size=4,
                  enable_attention_dp=True,
                  guided_decoding_backend=backend,
                  kv_cache_config=kv_cache_config,
                  cuda_graph_config=cuda_graph_config,
                  enable_chunked_prefill=True,
                  max_num_tokens=256,
                  speculative_config=mtp_config)
        with llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    def test_dummy_load_format(self):
        llm = LLM(
            f"{llm_models_root()}/DeepSeek-V3-Lite/fp8",
            load_format="dummy",
            moe_config=fp8_block_scales_moe_config(),
        )
        with llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, is_integration_test=True)


@pytest.mark.timeout(14400)
@pytest.mark.skip_less_device_memory(80000)
class TestDeepSeekR1(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-R1"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1"

    @skip_pre_blackwell
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,mtp_nextn,fp8kv,attention_dp,enable_lm_head_tp_in_adp,cuda_graph,overlap_scheduler,max_batch_size,moe_tactic",
        [
            #  Use a larger batch_size to speed up the tests
            pytest.param(8,
                         1,
                         4,
                         3,
                         False,
                         False,
                         False,
                         True,
                         True,
                         32,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         4,
                         3,
                         False,
                         False,
                         False,
                         True,
                         True,
                         32,
                         "trtllm_gen",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         4,
                         3,
                         False,
                         True,
                         True,
                         True,
                         True,
                         32,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         4,
                         3,
                         False,
                         True,
                         True,
                         True,
                         True,
                         32,
                         "trtllm_gen",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(4,
                         1,
                         4,
                         3,
                         False,
                         True,
                         True,
                         True,
                         True,
                         16,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(4)),
            pytest.param(8,
                         1,
                         8,
                         0,
                         True,
                         True,
                         False,
                         True,
                         True,
                         32,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         1,
                         0,
                         True,
                         True,
                         False,
                         True,
                         True,
                         32,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(4,
                         1,
                         1,
                         0,
                         True,
                         True,
                         False,
                         True,
                         True,
                         16,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(4)),
            pytest.param(8,
                         1,
                         8,
                         1,
                         True,
                         True,
                         False,
                         True,
                         True,
                         32,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         8,
                         1,
                         True,
                         True,
                         False,
                         True,
                         True,
                         8,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(1,
                         4,
                         1,
                         1,
                         True,
                         True,
                         False,
                         True,
                         True,
                         32,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(4)),
        ],
        ids=[
            "latency", "latency_trtllmgen", "latency_adp_lmtp",
            "latency_trtllmgen_adp_lmtp", "latency_adp_lmtp_tp4", "throughput",
            "throughput_tp8", "throughput_tp4", "throughput_mtp",
            "throughput_bs8_mtp", "throughput_pp4_mtp"
        ])
    def test_nvfp4_multi_gpus(self, tp_size, pp_size, ep_size, mtp_nextn, fp8kv,
                              attention_dp, enable_lm_head_tp_in_adp,
                              cuda_graph, overlap_scheduler, max_batch_size,
                              moe_tactic):
        skip_unless_moe_tactic_runs_here(moe_tactic)

        # Wrong MTP draft tokens cannot be caught by the accuracy scores:
        # rejected drafts are regenerated by the target model, so accuracy
        # stays intact and the only observable symptom is a collapsed
        # acceptance length (https://nvbugs/6460072). Assert on it for the
        # ADP + LM-head-TP cases, whose draft sampling takes a dedicated path.
        check_acceptance = mtp_nextn > 0 and enable_lm_head_tp_in_adp

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.70)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(tactic=moe_tactic))

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = _deepseek_v3_mtp_config(mtp_nextn, moe_tactic)
        with LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1-FP4",
                 max_batch_size=max_batch_size,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 enable_lm_head_tp_in_adp=enable_lm_head_tp_in_adp,
                 enable_iter_perf_stats=check_acceptance,
                 max_stats_len=-1,
                 autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                 speculative_config=mtp_config) as llm:

            assert llm.args.moe_config.tactic == moe_tactic
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            # This covers the case with relatively large seqlen in the generation phase.
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            # Commented out because GPQA takes too long to run
            # task = GPQADiamond(self.MODEL_NAME)
            # task.evaluate(llm,
            #               extra_evaluator_kwargs=dict(apply_chat_template=True))

            if check_acceptance:
                stats = llm.get_stats(timeout=2)
                spec_iters = [
                    stat['specDecodingStats'] for stat in stats
                    if stat.get('specDecodingStats')
                    and stat['specDecodingStats']['numDraftTokens'] > 0
                ]
                assert spec_iters, \
                    "No iterations with speculative decoding stats"
                accepted = sum(s["numAcceptedTokens"] for s in spec_iters)
                reqs = sum(s["numRequestsWithDraftTokens"] for s in spec_iters)
                acceptance_length = (accepted + reqs) / reqs

                assert acceptance_length >= 1.5, (
                    f"MTP acceptance length {acceptance_length:.2f} < 1.5: "
                    "draft tokens are likely wrong in the ADP + LM-head-TP "
                    "sampling path.")

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    @skip_pre_blackwell
    @pytest.mark.skip_less_device_memory(95000)
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,mtp_nextn,fp8kv,attention_dp,cuda_graph,overlap_scheduler,max_batch_size,moe_tactic",
        [
            #  Use a larger batch_size to speed up the tests
            pytest.param(8,
                         1,
                         4,
                         3,
                         False,
                         False,
                         True,
                         True,
                         32,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         4,
                         3,
                         False,
                         False,
                         True,
                         True,
                         32,
                         "trtllm_gen",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         8,
                         0,
                         True,
                         True,
                         True,
                         True,
                         32,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         1,
                         0,
                         True,
                         True,
                         True,
                         True,
                         32,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(4,
                         1,
                         1,
                         0,
                         True,
                         True,
                         True,
                         True,
                         16,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(4)),
            pytest.param(8,
                         1,
                         8,
                         1,
                         True,
                         True,
                         True,
                         True,
                         32,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(8,
                         1,
                         8,
                         1,
                         True,
                         True,
                         True,
                         True,
                         8,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
        ],
        ids=[
            "latency", "latency_trtllmgen", "throughput", "throughput_tp8",
            "throughput_tp4", "throughput_mtp", "throughput_bs8_mtp"
        ])
    def test_nvfp4_multi_gpus_sm120(self, tp_size, pp_size, ep_size, mtp_nextn,
                                    fp8kv, attention_dp, cuda_graph,
                                    overlap_scheduler, max_batch_size,
                                    moe_tactic, mocker):
        skip_unless_moe_tactic_runs_here(moe_tactic)

        # SM120 (96 GB GPUs) is tight on memory for this DeepSeek-R1 NVFP4 config;
        # expandable_segments avoids OOM from cache fragmentation when the autotuner
        # asks for a large contiguous workspace after kv_cache allocation.
        patch_mpi_pool_session_for_env(
            mocker, {"PYTORCH_ALLOC_CONF": "expandable_segments:True"})

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(tactic=moe_tactic))

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = _deepseek_v3_mtp_config(mtp_nextn, moe_tactic)
        with LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1-0528-FP4-v2",
                 max_batch_size=max_batch_size,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:

            assert llm.args.moe_config.tactic == moe_tactic
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            # This covers the case with relatively large seqlen in the generation phase.
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            # Commented out because GPQA takes too long to run
            # task = GPQADiamond(self.MODEL_NAME)
            # task.evaluate(llm,
            #               extra_evaluator_kwargs=dict(apply_chat_template=True))

    @skip_pre_blackwell
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,mtp_nextn,fp8kv,attention_dp,cuda_graph,overlap_scheduler,max_batch_size,moe_tactic",
        [
            #  Use a larger batch_size to speed up the tests
            pytest.param(8,
                         1,
                         4,
                         3,
                         False,
                         False,
                         True,
                         True,
                         32,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(8)),
            pytest.param(4,
                         1,
                         1,
                         0,
                         True,
                         True,
                         True,
                         True,
                         16,
                         "cutlass",
                         marks=pytest.mark.skip_less_mpi_world_size(4)),
        ],
        ids=["latency", "throughput_tp4"])
    def test_nvfp4_multi_gpus_chunked_prefill(self, tp_size, pp_size, ep_size,
                                              mtp_nextn, fp8kv, attention_dp,
                                              cuda_graph, overlap_scheduler,
                                              max_batch_size, moe_tactic):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.70)
        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(tactic=moe_tactic))

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = _deepseek_v3_mtp_config(mtp_nextn, moe_tactic)
        with LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1-FP4",
                 max_batch_size=max_batch_size,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config,
                 autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                 enable_chunked_prefill=True,
                 max_num_tokens=512) as llm:

            assert llm.args.moe_config.tactic == moe_tactic
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    def test_nvfp4_multi_gpus_corner_case(self):
        """Test the corner case of the NVFP4 model.

        When using the same value for max_seq_len and max_num_tokens, there will
        be no enough kv block for the dummy requests in CUDA graph warmup when
        creating the py_executor before estimating kv cache. Then CUDA graph
        capture will be triggered when estimating kv cache. This may cause some
        errors. More info in https://nvbugs/5485325.
        """
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.80,
                                        dtype="fp8",
                                        enable_block_reuse=False)
        moe_tactic = sm100f_moe_tactic_or_cutlass("trtllm_gen")
        pytorch_config = dict(disable_overlap_scheduler=False,
                              cuda_graph_config=CudaGraphConfig(
                                  enable_padding=True, max_batch_size=1024),
                              moe_config=MoeConfig(tactic=moe_tactic))

        mtp_config = _deepseek_v3_mtp_config(1, moe_tactic)
        with LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1-FP4",
                 tensor_parallel_size=8,
                 pipeline_parallel_size=1,
                 moe_expert_parallel_size=8,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=False,
                 speculative_config=mtp_config,
                 autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                 max_seq_len=5120,
                 max_num_tokens=5120) as llm:

            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_mpi_world_size(8)
    @skip_pre_hopper
    @pytest.mark.skip_less_device_memory(140000)
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,mtp_nextn,fp8kv,attention_dp,cuda_graph,overlap_scheduler,max_batch_size,moe_tactic",
        [(8, 1, 4, 3, False, False, True, True, 1, None),
         (8, 1, 8, 0, True, True, True, True, 24, None),
         (8, 1, 8, 1, True, True, True, True, 24, None),
         (8, 1, 8, 1, True, True, True, True, 24, "trtllm_gen")],
        ids=[
            "latency", "throughput", "throughput_mtp", "throughput_mtp_trtllm"
        ])
    def test_fp8_blockscale(self, tp_size, pp_size, ep_size, mtp_nextn, fp8kv,
                            attention_dp, cuda_graph, overlap_scheduler,
                            max_batch_size, moe_tactic):
        # None selects the FP8 block-scale default for this GPU.
        if moe_tactic is None:
            moe_tactic = fp8_block_scales_moe_tactic()
        else:
            skip_unless_moe_tactic_runs_here(moe_tactic)
        if is_sm_100f():
            moe_config = MoeConfig(tactic=moe_tactic, max_num_tokens=16384)
            kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)
        else:
            moe_config = MoeConfig(tactic=moe_tactic)
            kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=moe_config,
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(max_draft_len=mtp_nextn,
                                           moe_tactic=moe_tactic)
        with LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1",
                 max_batch_size=max_batch_size,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_mpi_world_size(8)
    @skip_pre_hopper
    @pytest.mark.skip_less_device_memory(140000)
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,mtp_nextn,fp8kv,attention_dp,cuda_graph,overlap_scheduler,max_batch_size",
        [(8, 1, 4, 3, False, False, True, True, 1),
         (8, 1, 8, 0, True, True, True, True, 24)],
        ids=["latency", "throughput"])
    def test_fp8_blockscale_chunked_prefill(self, tp_size, pp_size, ep_size,
                                            mtp_nextn, fp8kv, attention_dp,
                                            cuda_graph, overlap_scheduler,
                                            max_batch_size):
        if is_sm_100f():
            moe_config = fp8_block_scales_moe_config(max_num_tokens=16384)
            kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)
        else:
            moe_config = MoeConfig()
            kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=moe_config,
        )

        if fp8kv:
            kv_cache_config.dtype = "fp8"

        mtp_config = None
        if mtp_nextn > 0:
            mtp_config = MTPDecodingConfig(
                max_draft_len=mtp_nextn,
                moe_tactic=fp8_block_scales_moe_tactic())
        with LLM(f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1",
                 max_batch_size=max_batch_size,
                 tensor_parallel_size=tp_size,
                 pipeline_parallel_size=pp_size,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 **pytorch_config,
                 enable_attention_dp=attention_dp,
                 speculative_config=mtp_config,
                 enable_chunked_prefill=True,
                 max_num_tokens=512) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


def _make_deepseekv4_eplb_config(model_path, layer_updates_per_iter, ep_size=8):
    """Build a MoeLoadBalancerConfig for DeepSeek V4 from the HF config.

    All V4 layers run MoE (no first_k_dense_replace prefix), so static
    assignments cover every layer in 0..num_hidden_layers-1. Extra slots
    per rank match the TestNemotronV3Super pattern (16 redundant per rank).
    """
    with open(f"{model_path}/config.json") as f:
        cfg = json.load(f)
    num_experts = cfg["n_routed_experts"]
    num_slots = num_experts + 16 * ep_size
    if layer_updates_per_iter > 0:
        return MoeLoadBalancerConfig(
            num_slots=num_slots, layer_updates_per_iter=layer_updates_per_iter)
    num_hidden_layers = cfg["num_hidden_layers"]
    initial_global_assignments = {
        i: [(i + j) % num_experts for j in range(num_slots)]
        for i in range(num_hidden_layers)
    }
    return MoeLoadBalancerConfig(
        num_slots=num_slots,
        initial_global_assignments=initial_global_assignments,
        layer_updates_per_iter=0)


DEEPSEEKV4_TEST_MAX_BATCH_SIZE = 128
# DeepSeek-V4-Pro (~806 GB) fits on 4 GPUs only with 288 GB-class memory.
DEEPSEEKV4_PRO_4GPU_MIN_MEMORY_MB = 250000


@skip_pre_blackwell
@pytest.mark.skip_device_not_contain(["GB300"])
@pytest.mark.timeout(GSM8K_TEST_TIMEOUT)
class TestDeepSeekV41(LlmapiAccuracyTestHarness):
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4.1-Flash"

    # Checkpoint suffix: DeepSeek's MXFP4 experts, NVFP4 experts, and NVFP4 experts with MXFP4 Engram tables.
    @pytest.mark.parametrize("tp_size,attention_dp,checkpoint", [
        pytest.param(4,
                     False,
                     "",
                     id="mxfp4-tp4",
                     marks=pytest.mark.skip_less_mpi_world_size(4)),
        pytest.param(4,
                     False,
                     "-NVFP4",
                     id="nvfp4-tp4",
                     marks=pytest.mark.skip_less_mpi_world_size(4)),
        pytest.param(4,
                     True,
                     "-NVFP4",
                     id="nvfp4-dp4",
                     marks=pytest.mark.skip_less_mpi_world_size(4)),
        pytest.param(8,
                     False,
                     "-NVFP4",
                     id="nvfp4-tp8",
                     marks=pytest.mark.skip_less_mpi_world_size(8)),
        pytest.param(8,
                     True,
                     "-NVFP4",
                     id="nvfp4-dp8",
                     marks=pytest.mark.skip_less_mpi_world_size(8)),
        pytest.param(4,
                     True,
                     "-1017-NVFP4",
                     id="nvfp4-1017-dp4",
                     marks=pytest.mark.skip_less_mpi_world_size(4)),
    ])
    def test_gsm8k_dspark(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                          tp_size: int, attention_dp: bool,
                          checkpoint: str) -> None:
        for name in tuple(os.environ):
            if name.startswith("TRTLLM_V41_"):
                monkeypatch.delenv(name)

        with LLM(
                self.MODEL_PATH + checkpoint,
                tensor_parallel_size=tp_size,
                moe_expert_parallel_size=tp_size,
                gpus_per_node=4,
                enable_attention_dp=attention_dp,
                custom_tokenizer="deepseek_v41",
                trust_remote_code=True,
                max_seq_len=8192,
                max_num_tokens=8192,
                max_batch_size=64,
                enable_chunked_prefill=True,
                disable_overlap_scheduler=False,
                enable_iter_perf_stats=True,
                cuda_graph_config=CudaGraphConfig(
                    batch_sizes=[1, 2, 4, 8, 16, 32, 64], enable_padding=True),
                moe_config=MoeConfig(tactic="trtllm_gen"),
                allreduce_strategy="NCCL",
                kv_cache_config=KvCacheConfig(dtype="fp8",
                                              enable_block_reuse=True,
                                              enable_swa_scratch_reuse=False,
                                              tokens_per_block=128,
                                              max_tokens=262144,
                                              free_gpu_memory_fraction=0.5),
                sparse_attention_config={"algorithm": "csa2"},
                speculative_config=DSparkDecodingConfig(max_draft_len=5),
        ) as llm:
            request_outputs = []
            evaluate_gsm8k(llm, tmp_path, request_outputs=request_outputs)
            assert request_outputs
            for output in request_outputs:
                assert output.finished
                assert output.decoding_iter > 0
            # Request metrics include every attention-DP rank.
            acceptance_length = sum(
                output.avg_decoded_tokens_per_iter * output.decoding_iter
                for output in request_outputs) / sum(
                    output.decoding_iter for output in request_outputs)
            print(f"[AL] DS-V4.1{checkpoint} tp={tp_size} "
                  f"attention_dp={attention_dp}: {acceptance_length:.3f}")
            assert acceptance_length >= 3.0, (
                f"DSpark acceptance length {acceptance_length:.3f} < 3.0")

    @pytest.mark.skip_less_device(4)
    @pytest.mark.threadleak(enabled=False)
    def test_breakable_prefill_cuda_graph(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Breakable prefill graphs give eager prefill's greedy tokens on DEP4.

        Bounded replay plans every attention-DP prefill batch, so the 128- and 1,500-token
        prompts both run eagerly with the decoder pass from its own graphs.
        """
        for name in tuple(os.environ):
            if name.startswith("TRTLLM_V41_"):
                monkeypatch.delenv(name)

        # Text keeps the greedy choices well apart; graph and eager rounding differ slightly.
        text = (
            "TensorRT-LLM accelerates reliable large language model inference "
            "with efficient attention, parallelism, and CUDA graphs. ") * 100
        sampling_params = SamplingParams(max_tokens=8,
                                         min_tokens=8,
                                         temperature=0,
                                         ignore_eos=True,
                                         detokenize=False,
                                         add_special_tokens=False)

        def run(backend):
            with LLM(
                    self.MODEL_PATH + "-NVFP4",
                    tensor_parallel_size=4,
                    moe_expert_parallel_size=4,
                    gpus_per_node=4,
                    enable_attention_dp=True,
                    custom_tokenizer="deepseek_v41",
                    trust_remote_code=True,
                    # Breakable prefill graphs serve the checkpoint text-only.
                    disable_mm_encoder=True,
                    max_seq_len=4096,
                    max_num_tokens=2048,
                    max_batch_size=8,
                    cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8],
                                                      enable_padding=True),
                    moe_config=MoeConfig(tactic="trtllm_gen"),
                    kv_cache_config=KvCacheConfig(dtype="fp8",
                                                  enable_block_reuse=False,
                                                  tokens_per_block=128,
                                                  max_tokens=65536,
                                                  free_gpu_memory_fraction=0.5),
                    sparse_attention_config={"algorithm": "csa2"},
                    prefill_cuda_graph_backend=backend,
                    prefill_capture_num_tokens=[128, 256, 512, 1024]) as llm:
                text_ids = llm.tokenizer.encode(text, add_special_tokens=False)
                token_ids = []
                for prompt in (text_ids[:128], text_ids[:1500]):
                    output = llm.generate_async(
                        prompt, sampling_params=sampling_params).result()
                    token_ids.append(output.outputs[0].token_ids)
                return token_ids

        assert run(PrefillCudaGraphBackend.BREAKABLE) == run(
            PrefillCudaGraphBackend.DISABLED)


_DEEPSEEK_V4_GSM8K_SYSTEM_PROMPT = (
    "Solve the problem carefully. End your response with a final line exactly "
    "in the form #### <answer>, using the simplest numeric form without units "
    "or trailing zeros.")

DEEPSEEKV4_PARALLELISM = pytest.mark.parametrize("tp_size,attention_dp", [
    pytest.param(
        4, True, id="dp4", marks=pytest.mark.skip_less_mpi_world_size(4)),
    pytest.param(
        8, True, id="dp8", marks=pytest.mark.skip_less_mpi_world_size(8)),
])


def _skip_unless_deepseekv4_4gpu_memory(tp_size, min_device_memory_for_4gpus):
    """Skip the 4-GPU case on GPUs too small for the checkpoint.

    ``min_device_memory_for_4gpus`` (MB): e.g. DeepSeek-V4-Pro needs 288
    GB-class GPUs. The rank count is DEEPSEEKV4_PARALLELISM's marker.
    """
    if tp_size == 4:
        # Query the GPUs directly: get_device_memory() falls back to host memory
        # when nvidia-smi fails, which says nothing about GPU capacity.
        local_gpus = min(tp_size, torch.cuda.device_count())
        if local_gpus == 0:
            pytest.skip("Cannot determine GPU memory for the 4-GPU DeepSeek-V4 "
                        "configuration")
        gpu_memory_mb = min(
            torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)
            for i in range(local_gpus))
        if gpu_memory_mb < min_device_memory_for_4gpus:
            pytest.skip(
                f"4 GPUs need at least {min_device_memory_for_4gpus} MB "
                "each for this DeepSeek-V4 checkpoint")


def _deepseekv4_pro_agg_llm_kwargs(tp_size=8, attention_dp=True, **overrides):
    kwargs = dict(
        tensor_parallel_size=tp_size,
        moe_expert_parallel_size=tp_size,
        moe_config=MoeConfig(tactic="trtllm_gen"),
        enable_attention_dp=attention_dp,
        max_seq_len=4096,
        max_batch_size=16,
        max_num_tokens=8192,
        custom_tokenizer="deepseek_v4",
        # Cap the KV cache so the padded CUDA-graph capture (batch sizes up to
        # 1024) and other post-KV resources have headroom; without this the
        # default fraction (~0.9) leaves too little and engine init OOMs on
        # large-memory GPUs. These tests need very little KV.
        kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                      tokens_per_block=128,
                                      dtype="fp8",
                                      host_cache_size=0,
                                      free_gpu_memory_fraction=0.6),
        cuda_graph_config=CudaGraphConfig(batch_sizes=[
            1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 768, 1024
        ],
                                          enable_padding=True),
        speculative_config=MTPDecodingConfig(max_draft_len=1),
    )
    kwargs.update(overrides)
    return kwargs


def _run_deepseekv4_eplb(model_name: str,
                         model_path: str,
                         moe_tactic: str,
                         eplb_config: MoeLoadBalancerConfig,
                         mtp_nextn: int = 0,
                         tensor_parallel_size: int = 4) -> None:
    # Default config targets 4x B300 (~288 GB/GPU): plenty of headroom at
    # TP=4 for V4-Flash NVFP4 (~36 GB/rank weights) or V4-Flash-Base FP8
    # (~71 GB/rank weights), even with EPLB redundancy + DeepGemm MoE
    # workspace + cuda graph capture.
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5)
    pytorch_config = dict(cuda_graph_config=CudaGraphConfig(),
                          moe_config=MoeConfig(tactic=moe_tactic,
                                               load_balancer=eplb_config))
    mtp_config = (MTPDecodingConfig(
        num_nextn_predict_layers=mtp_nextn) if mtp_nextn > 0 else None)
    with LLM(model_path,
             tensor_parallel_size=tensor_parallel_size,
             moe_expert_parallel_size=tensor_parallel_size,
             kv_cache_config=kv_cache_config,
             enable_attention_dp=True,
             max_batch_size=DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
             max_seq_len=4096,
             **pytorch_config,
             speculative_config=mtp_config) as llm:
        task = GSM8K(model_name)
        task.evaluate(llm)


@pytest.mark.timeout(14400)
@pytest.mark.skip_less_device_memory(140000)
@skip_pre_blackwell
class TestDeepSeekV4Flash(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4-Flash"

    @pytest.mark.skip_less_mpi_world_size(4)
    def test_auto_dtype(self):
        # Aggregate (non-disagg, non-EPLB) coverage. NVFP4 weights are ~71
        # GB/rank at TP=2, ~36 GB/rank at TP=4 — TP=4 fits comfortably on
        # 4x B200 178GB. The TRTLLM-Gen MoE is pinned because it serves the
        # V4-Flash MXFP4 routed experts.
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5)
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 moe_config=MoeConfig(tactic="trtllm_gen"),
                 enable_attention_dp=True,
                 max_batch_size=DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
                 max_seq_len=4096,
                 max_num_tokens=4096,
                 guided_decoding_backend="xgrammar",
                 autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                 kv_cache_config=kv_cache_config) as llm:
            assert_guided_decoding_regex(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_mpi_world_size(4)
    def test_nvfp4_kv(self):
        kv_cache_config = KvCacheConfig(dtype="nvfp4",
                                        free_gpu_memory_fraction=0.5)
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 moe_config=MoeConfig(tactic="trtllm_gen"),
                 enable_attention_dp=True,
                 enable_chunked_prefill=True,
                 max_batch_size=DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
                 max_seq_len=4096,
                 max_num_tokens=512,
                 kv_cache_config=kv_cache_config) as llm:
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_no_rubin
    @pytest.mark.skip_less_mpi_world_size(4)
    def test_nvfp4_plus(self, monkeypatch):
        # Serves the MXFP4 routed experts as NVFP4_PLUS on the CuteDSL MoE:
        # the E8M0 weight scales convert exactly to UE5M3, and activations are
        # quantized to E2M1 with a UE5M3 scale per 32 elements. env_overrides
        # sets the override in this process and on every worker rank;
        # monkeypatch removes it from this process once the test ends.
        env = {"OVERRIDE_QUANT_ALGO": "NVFP4_PLUS"}
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5)
        with LLM(self.MODEL_PATH,
                 env_overrides=env,
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 moe_config=MoeConfig(tactic="cutedsl.grouped_gemm"),
                 enable_attention_dp=True,
                 max_batch_size=DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
                 max_seq_len=4096,
                 max_num_tokens=4096,
                 kv_cache_config=kv_cache_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_acc_spec="moe_quant_algo=NVFP4_PLUS")

    @pytest.mark.skip_less_mpi_world_size(4)
    def test_tep_mtp_separate_draft_kv_cache(self):
        # TEP (attention_dp=False) + one-model MTP exercises the separate draft
        # KV cache manager path. CUDA graphs cover draft capture and replay.
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5,
                                        dtype="fp8")
        with LLM(
                self.MODEL_PATH,
                tensor_parallel_size=4,
                moe_expert_parallel_size=4,
                moe_config=MoeConfig(tactic="trtllm_gen"),
                enable_attention_dp=False,
                speculative_config=MTPDecodingConfig(max_draft_len=1),
                max_batch_size=DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
                max_seq_len=4096,
                max_num_tokens=4096,
                cuda_graph_config=CudaGraphConfig(enable_padding=True),
                kv_cache_config=kv_cache_config,
                max_stats_len=-1,
                enable_iter_perf_stats=True,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            assert_acceptance_length_for_llm(
                "TestDeepSeekV4Flash::test_tep_mtp_separate_draft_kv_cache",
                llm,
            )

    @pytest.mark.skip_less_mpi_world_size(4)
    @parametrize_with_ids("moe_tactic", ["trtllm_gen", "deepgemm.mega_moe"])
    def test_nvfp4_4gpus_static_eplb(self, moe_tactic):
        eplb_config = _make_deepseekv4_eplb_config(self.MODEL_PATH,
                                                   layer_updates_per_iter=0,
                                                   ep_size=4)
        _run_deepseekv4_eplb(self.MODEL_NAME, self.MODEL_PATH, moe_tactic,
                             eplb_config)

    @pytest.mark.skip_less_mpi_world_size(4)
    @parametrize_with_ids("moe_tactic", ["trtllm_gen"])
    @parametrize_with_ids("mtp_nextn", [0, 1])
    def test_nvfp4_4gpus_online_eplb(self, moe_tactic, mtp_nextn):
        eplb_config = _make_deepseekv4_eplb_config(self.MODEL_PATH,
                                                   layer_updates_per_iter=2,
                                                   ep_size=4)
        _run_deepseekv4_eplb(self.MODEL_NAME,
                             self.MODEL_PATH,
                             moe_tactic,
                             eplb_config,
                             mtp_nextn=mtp_nextn)

    @DEEPSEEKV4_PARALLELISM
    @pytest.mark.threadleak(enabled=False)
    def test_mixed_breakable_cuda_graph(self, tp_size, attention_dp):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        base_prompt_ids = tokenizer.encode(
            "TensorRT-LLM accelerates reliable large language model inference "
            "with efficient attention, parallelism, and CUDA graphs. ",
            add_special_tokens=False,
        )
        assert base_prompt_ids

        def make_prompt(prompt_length):
            return (base_prompt_ids *
                    ((prompt_length + len(base_prompt_ids) - 1) //
                     len(base_prompt_ids)))[:prompt_length]

        generation_prompt = make_prompt(64)
        context_prompt = make_prompt(129)
        sampling_params = SamplingParams(
            max_tokens=8,
            min_tokens=8,
            seed=42,
            temperature=0,
            ignore_eos=True,
            detokenize=False,
            add_special_tokens=False,
        )
        common_llm_kwargs = dict(
            tensor_parallel_size=tp_size,
            moe_expert_parallel_size=tp_size,
            moe_config=MoeConfig(tactic="trtllm_gen"),
            enable_attention_dp=attention_dp,
            max_batch_size=8,
            max_num_tokens=1024,
            max_seq_len=2048,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=False,
                dtype="fp8",
                free_gpu_memory_fraction=0.6,
            ),
            cuda_graph_config=CudaGraphConfig(
                batch_sizes=[1, 2, 4, 6, 8],
                enable_padding=True,
            ),
        )

        def run(backend):
            with LLM(
                    self.MODEL_PATH,
                    **common_llm_kwargs,
                    prefill_cuda_graph_backend=backend,
                    prefill_capture_num_tokens=[128, 256, 512, 1024],
            ) as llm:
                generation_request = llm.generate_async(
                    generation_prompt,
                    sampling_params=sampling_params,
                    streaming=True,
                )
                next(generation_request)
                assert not generation_request.finished

                # Admit a context request while the first request is decoding.
                context_request = llm.generate_async(
                    context_prompt,
                    sampling_params=sampling_params,
                    streaming=False,
                )
                generation_output = generation_request.result()
                context_output = context_request.result()
                return [
                    generation_output.outputs[0].token_ids,
                    context_output.outputs[0].token_ids,
                ]

        eager_token_ids = run(PrefillCudaGraphBackend.DISABLED)
        breakable_token_ids = run(PrefillCudaGraphBackend.BREAKABLE)
        assert breakable_token_ids == eager_token_ids


_DEEPSEEK_V4_GSM8K_SYSTEM_PROMPT = (
    "Solve the problem carefully. End your response with a final line exactly "
    "in the form #### <answer>, using the simplest numeric form without units "
    "or trailing zeros.")


@pytest.mark.timeout(14400)
@pytest.mark.skip_less_device_memory(140000)
@skip_pre_blackwell
class TestDeepSeekV4Pro(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V4-Pro"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4-Pro"
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt=_DEEPSEEK_V4_GSM8K_SYSTEM_PROMPT,
    )

    @DEEPSEEKV4_PARALLELISM
    def test_gsm8k_full_accuracy(self, tp_size, attention_dp):
        _skip_unless_deepseekv4_4gpu_memory(
            tp_size,
            min_device_memory_for_4gpus=DEEPSEEKV4_PRO_4GPU_MIN_MEMORY_MB)
        with LLM(self.MODEL_PATH,
                 autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                 **_deepseekv4_pro_agg_llm_kwargs(
                     tp_size=tp_size, attention_dp=attention_dp)) as llm:
            task = GSM8K(self.MODEL_NAME)
            acc_params = task.get_hypothesis_testing_params(
                dtype=llm.args.dtype,
                quant_algo=llm.args.quant_config.quant_algo,
                kv_cache_quant_algo=llm.args.quant_config.kv_cache_quant_algo,
                spec_dec_algo=llm.args.speculative_config.decoding_type)
            assert acc_params.num_samples == GSM8K.NUM_SAMPLES
            with mock.patch.dict(os.environ, {"INTEGRATION_TEST": "0"}):
                task.evaluate(
                    llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)


@pytest.mark.timeout(14400)
@pytest.mark.skip_less_device_memory(140000)
@skip_pre_blackwell
class TestDeepSeekV4ProDSpark(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V4-Pro"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4-Pro-DSpark"
    # NVFP4 export (nvidia/deepseek-v4-pro-nvfp4-dspark) with the embedded
    # DSpark drafter.
    NVFP4_MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4-Pro-nvfp4-DSpark"
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt=_DEEPSEEK_V4_GSM8K_SYSTEM_PROMPT,
    )

    @DEEPSEEKV4_PARALLELISM
    def test_gsm8k_megamoe_deepgemm(self, tp_size, attention_dp):
        _skip_unless_deepseekv4_4gpu_memory(
            tp_size,
            min_device_memory_for_4gpus=DEEPSEEKV4_PRO_4GPU_MIN_MEMORY_MB)
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        free_gpu_memory_fraction=0.5)
        # A target tactic does not reach the drafter; pin it to the same
        # implementation.
        moe_tactic = "deepgemm.mega_moe"
        spec_config = DSparkDecodingConfig(max_draft_len=5,
                                           speculative_model=self.MODEL_PATH,
                                           moe_tactic=moe_tactic)
        with LLM(self.MODEL_PATH,
                 attn_backend="TRTLLM",
                 tensor_parallel_size=tp_size,
                 moe_expert_parallel_size=tp_size,
                 enable_attention_dp=attention_dp,
                 moe_config=MoeConfig(tactic=moe_tactic),
                 max_batch_size=DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
                 max_seq_len=4096,
                 max_num_tokens=4096,
                 kv_cache_config=kv_cache_config,
                 enable_chunked_prefill=False,
                 disable_overlap_scheduler=True,
                 custom_tokenizer="deepseek_v4",
                 max_stats_len=-1,
                 enable_iter_perf_stats=True,
                 autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                 speculative_config=spec_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            acc_params = task.get_hypothesis_testing_params(
                dtype=llm.args.dtype,
                quant_algo=llm.args.quant_config.quant_algo,
                kv_cache_quant_algo=llm.args.quant_config.kv_cache_quant_algo,
                spec_dec_algo=llm.args.speculative_config.decoding_type)
            assert acc_params.num_samples == GSM8K.NUM_SAMPLES
            with mock.patch.dict(os.environ, {"INTEGRATION_TEST": "0"}):
                task.evaluate(
                    llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            acceptance_length = compute_acceptance_length(llm)
            print(f"[AL] test_gsm8k_megamoe_deepgemm[dp{tp_size}] "
                  f"acceptance_length = {acceptance_length:.3f}")
            assert_acceptance_length(
                f"TestDeepSeekV4ProDSpark::test_gsm8k_megamoe_deepgemm[dp{tp_size}]",
                acceptance_length)

    @DEEPSEEKV4_PARALLELISM
    def test_gsm8k_megamoe_deepgemm_draft_len_schedule(self, tp_size,
                                                       attention_dp):
        """GSM8K under a schedule that trims, then turns off, verified drafts.

        The evaluation submits every sample up front, so each attention-DP
        rank first runs full batches without speculation and then passes
        through the shorter and the full draft length as it drains, out of
        step with its peers.
        """
        _skip_unless_deepseekv4_4gpu_memory(
            tp_size,
            min_device_memory_for_4gpus=DEEPSEEKV4_PRO_4GPU_MIN_MEMORY_MB)
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        free_gpu_memory_fraction=0.5)
        schedule = {32: 5, 96: 3}
        spec_config = DSparkDecodingConfig(max_draft_len=5,
                                           draft_len_schedule=schedule,
                                           speculative_model=self.MODEL_PATH)
        with LLM(self.MODEL_PATH,
                 attn_backend="TRTLLM",
                 tensor_parallel_size=tp_size,
                 moe_expert_parallel_size=tp_size,
                 enable_attention_dp=attention_dp,
                 moe_config=MoeConfig(backend="MEGAMOE_DEEPGEMM"),
                 max_batch_size=DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
                 max_seq_len=4096,
                 max_num_tokens=4096,
                 kv_cache_config=kv_cache_config,
                 enable_chunked_prefill=False,
                 disable_overlap_scheduler=True,
                 custom_tokenizer="deepseek_v4",
                 max_stats_len=-1,
                 enable_iter_perf_stats=True,
                 speculative_config=spec_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            acc_params = task.get_hypothesis_testing_params(
                dtype=llm.args.dtype,
                quant_algo=llm.args.quant_config.quant_algo,
                kv_cache_quant_algo=llm.args.quant_config.kv_cache_quant_algo,
                spec_dec_algo=llm.args.speculative_config.decoding_type)
            assert acc_params.num_samples == GSM8K.NUM_SAMPLES
            with mock.patch.dict(os.environ, {"INTEGRATION_TEST": "0"}):
                task.evaluate(
                    llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            # The schedule decides how many steps speculate, so the acceptance
            # length has no fixed floor; this still requires that some did.
            acceptance_length = compute_acceptance_length(llm)
            print(f"[AL] draft_len_schedule[dp{tp_size}] "
                  f"acceptance_length = {acceptance_length:.3f}")

    @pytest.mark.skip_less_device_memory(DEEPSEEKV4_PRO_4GPU_MIN_MEMORY_MB)
    @pytest.mark.skip_less_mpi_world_size(4)
    def test_gsm8k_dep4_megamoe_cutedsl(self):
        # MegaMoE CuTe DSL is the backend that serves the NVFP4 export. DEP4
        # doubles the per-rank model footprint relative to DEP8, so bound both
        # the serving and the CUDA-graph batch to what fits without a warmup
        # OOM.
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        free_gpu_memory_fraction=0.5)
        # A target tactic does not reach the drafter; pin it to the same
        # implementation.
        moe_tactic = "cutedsl.mega_moe"
        spec_config = DSparkDecodingConfig(
            max_draft_len=5,
            speculative_model=self.NVFP4_MODEL_PATH,
            moe_tactic=moe_tactic)
        # Build the evaluator (dataset and harness) before the LLM claims the
        # GPUs.
        task = GSM8K(self.MODEL_NAME)
        evaluator = task.create_evaluator(GSM8K.NUM_SAMPLES,
                                          self.EXTRA_EVALUATOR_KWARGS)
        with LLM(self.NVFP4_MODEL_PATH,
                 attn_backend="TRTLLM",
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 enable_attention_dp=True,
                 moe_config=MoeConfig(tactic=moe_tactic),
                 max_batch_size=64,
                 cuda_graph_config=CudaGraphConfig(max_batch_size=64),
                 max_seq_len=4096,
                 max_num_tokens=4096,
                 kv_cache_config=kv_cache_config,
                 enable_chunked_prefill=False,
                 disable_overlap_scheduler=True,
                 custom_tokenizer="deepseek_v4",
                 max_stats_len=-1,
                 enable_iter_perf_stats=True,
                 autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                 speculative_config=spec_config) as llm:
            acc_params = task.get_hypothesis_testing_params(
                dtype=llm.args.dtype,
                quant_algo=llm.args.quant_config.quant_algo,
                kv_cache_quant_algo=llm.args.quant_config.kv_cache_quant_algo,
                spec_dec_algo=llm.args.speculative_config.decoding_type)
            assert acc_params.num_samples == GSM8K.NUM_SAMPLES
            with mock.patch.dict(os.environ, {"INTEGRATION_TEST": "0"}):
                task.evaluate(llm, prepared_evaluator=evaluator)
            # Gate on acceptance as well as on the score: a drafter regression
            # on this backend costs speed only, which the accuracy gate
            # cannot see.
            acceptance_length = compute_acceptance_length(llm)
            print(f"[AL] test_gsm8k_dep4_megamoe_cutedsl "
                  f"acceptance_length = {acceptance_length:.3f}")
            assert_acceptance_length(
                "TestDeepSeekV4ProDSpark::test_gsm8k_dep4_megamoe_cutedsl",
                acceptance_length)


@pytest.mark.timeout(14400)
@pytest.mark.skip_less_device_memory(140000)
@skip_pre_blackwell
class TestDeepSeekV4FlashDSpark(LlmapiAccuracyTestHarness):
    # V4-Flash with its embedded DSpark drafter and NVFP4 routed experts.
    MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4-Flash-nvfp4-DSpark"

    @pytest.mark.skip_less_mpi_world_size(4)
    def test_gsm8k_dep4_confidence_scheduling(self, tmp_path: Path) -> None:
        """Confidence-scheduled DSpark verification keeps GSM8K accuracy on DEP4."""
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        # Compact verify windows cost less than native K+1 rows, as in a measured table.
        cost_table = tmp_path / "dspark_costs.json"
        cost_table.write_text(
            json.dumps({
                "schema_version": 3,
                "costs": {
                    "5": {
                        str(g): {
                            "token_counts": [0, 3 * g, 4 * g, 5 * g],
                            "step_time_ms": [10.0, 8.5, 9.0, 9.5],
                        }
                        for g in batch_sizes
                    }
                },
            }))
        spec_config = DSparkDecodingConfig(
            max_draft_len=5,
            speculative_model=self.MODEL_PATH,
            enable_confidence_scheduling=True,
            confidence_cost_table_path=str(cost_table))
        with LLM(self.MODEL_PATH,
                 attn_backend="TRTLLM",
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 enable_attention_dp=True,
                 moe_config=MoeConfig(backend="MEGAMOE_CUTEDSL"),
                 max_batch_size=64,
                 cuda_graph_config=CudaGraphConfig(batch_sizes=batch_sizes,
                                                   enable_padding=True),
                 max_seq_len=4096,
                 max_num_tokens=4096,
                 kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                               free_gpu_memory_fraction=0.5),
                 enable_chunked_prefill=False,
                 disable_overlap_scheduler=True,
                 speculative_config=spec_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            with mock.patch.dict(os.environ, {"INTEGRATION_TEST": "0"}):
                task.evaluate(llm)


@pytest.mark.timeout(14400)
@pytest.mark.skip_less_device_memory(140000)
@skip_pre_blackwell
class TestDeepSeekV41FlashDSpark(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V4.1-Flash"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4.1-Flash"

    @pytest.mark.skip_less_mpi_world_size(4)
    # nvfp4_1017: the 1017 NVFP4 export, with NVFP4 routed experts and MXFP4 Engram tables.
    @parametrize_with_ids("nvfp4_1017", [False, True])
    @parametrize_with_ids("attention_dp", [False, True])
    @parametrize_with_ids("bounded_replay", [False, True])
    def test_gsm8k_tp4(self, bounded_replay: bool, attention_dp: bool,
                       nvfp4_1017: bool, monkeypatch: pytest.MonkeyPatch,
                       tmp_path: Path) -> None:
        """Full GSM8K with DSpark, block reuse and concurrent requests."""
        env = {
            "TRTLLM_V41_ENCODER_REPLAY": str(int(bounded_replay)),
            "TRTLLM_V41_DECODER_BOUNDED_REPLAY": str(int(bounded_replay)),
            "TLLM_KV_CACHE_MANAGER_V2_BACKEND": "cpp",
            "TLLM_EVAL_MAX_IN_FLIGHT": "128",
            "TLLM_EVAL_PARTIAL_SCORES_EVERY": "64",
        }
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        # Keep the full accuracy gate instead of the one-sample smoke mode.
        monkeypatch.setenv("INTEGRATION_TEST", "0")

        model_path = self.MODEL_PATH + ("-1017-NVFP4" if nvfp4_1017 else "")
        extra_acc_spec = "engram_quant=mxfp4" if nvfp4_1017 else None
        with LLM(model_path,
                 env_overrides=env,
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 gpus_per_node=4,
                 enable_attention_dp=attention_dp,
                 custom_tokenizer="deepseek_v41",
                 trust_remote_code=True,
                 max_seq_len=8192,
                 max_num_tokens=8192,
                 max_batch_size=64,
                 enable_chunked_prefill=True,
                 disable_overlap_scheduler=False,
                 enable_iter_perf_stats=True,
                 cuda_graph_config=CudaGraphConfig(
                     batch_sizes=[1, 2, 4, 8, 16, 32, 64], enable_padding=True),
                 kv_cache_config=KvCacheConfig(dtype="fp8",
                                               enable_block_reuse=True,
                                               enable_swa_scratch_reuse=False,
                                               tokens_per_block=128,
                                               max_tokens=262144,
                                               free_gpu_memory_fraction=0.5),
                 sparse_attention_config={"algorithm": "csa2"},
                 speculative_config=DSparkDecodingConfig(
                     max_draft_len=5, speculative_model=model_path)) as llm:
            task = GSM8K(self.MODEL_NAME)
            acc_params = task.get_hypothesis_testing_params(
                dtype=llm.args.dtype,
                quant_algo=llm.args.quant_config.quant_algo,
                kv_cache_quant_algo=llm.args.quant_config.kv_cache_quant_algo,
                spec_dec_algo=llm.args.speculative_config.decoding_type,
                extra_acc_spec=extra_acc_spec)
            assert acc_params.num_samples == 1319
            task.EVALUATE_KWARGS = dict(
                task.EVALUATE_KWARGS,
                sampling_override=True,
                scores_filter="exact_match,flexible-extract")
            evaluator_kwargs = dict(
                num_fewshot=5,
                random_seed=0,
                apply_chat_template=True,
                chat_template_kwargs={"thinking": False},
                system_prompt=_DEEPSEEK_V4_GSM8K_SYSTEM_PROMPT,
                log_samples=True,
                output_path=str(tmp_path / "results"),
                output_dir=str(tmp_path / "outputs"))
            score = task.evaluate(llm,
                                  extra_acc_spec=extra_acc_spec,
                                  extra_evaluator_kwargs=evaluator_kwargs,
                                  sampling_params=SamplingParams(
                                      max_tokens=256,
                                      truncate_prompt_tokens=4096,
                                      temperature=0,
                                      seed=0))

        with (tmp_path / "results" / "samples_gsm8k.json").open() as stream:
            results = json.load(stream)
        assert results["n-samples"]["gsm8k"]["effective"] == 1319
        for filter_name in ("strict-match", "flexible-extract"):
            rows = [
                row for row in results["samples"]["gsm8k"]
                if row["filter"] == filter_name
            ]
            assert len(rows) == 1319
            assert len({row["doc_id"] for row in rows}) == 1319
        assert score > 91.0, (
            f"GSM8K flexible exact match {score:.4f}% must exceed 91%; "
            f"bounded_replay={bounded_replay}, results={tmp_path}")


@pytest.mark.timeout(14400)
@pytest.mark.skip_less_device_memory(140000)
@skip_pre_blackwell
class TestDeepSeekV4FlashBase(LlmapiAccuracyTestHarness):
    MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-Base"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4-Flash-Base"

    @pytest.mark.skip_less_mpi_world_size(4)
    @parametrize_with_ids("moe_tactic", ["deepgemm.grouped_gemm", "trtllm_gen"])
    def test_auto_dtype(self, moe_tactic):
        # Aggregate (non-disagg, non-EPLB) smoke test. FP8 weights ~71 GB/rank
        # at TP=4 — fits on 4x B300 (~288 GB/GPU). 1-sample smoke. CUTLASS is
        # Hopper-only on Blackwell and skipped here.
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5)
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 moe_config=MoeConfig(tactic=moe_tactic),
                 enable_attention_dp=True,
                 max_batch_size=DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
                 max_seq_len=4096,
                 kv_cache_config=kv_cache_config) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm, is_integration_test=True)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, is_integration_test=True)

    @pytest.mark.skip_less_mpi_world_size(4)
    def test_fp8_chunked_prefill(self):
        kv_cache_config = KvCacheConfig(dtype="fp8",
                                        free_gpu_memory_fraction=0.5,
                                        enable_block_reuse=False)
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 moe_config=MoeConfig(tactic="deepgemm.grouped_gemm"),
                 cuda_graph_config=CudaGraphConfig(
                     max_batch_size=DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
                     enable_padding=True),
                 enable_attention_dp=True,
                 enable_chunked_prefill=True,
                 max_batch_size=DEEPSEEKV4_TEST_MAX_BATCH_SIZE,
                 max_num_tokens=128,
                 max_seq_len=4096,
                 kv_cache_config=kv_cache_config) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, is_integration_test=True)

    # CUTLASS is omitted: V4 Flash-Base FP8 block-scale weights take a
    # Hopper-only wgmma kernel path (CutlassFp8BlockScaleGemmRunner::moeGemm)
    # that has no SM100/SM103 implementation. The deepgemm.grouped_gemm tactic
    # runs the same weights through DeepGEMM's grouped GEMM on B200/B300.
    @pytest.mark.skip_less_mpi_world_size(4)
    @parametrize_with_ids("moe_tactic", ["deepgemm.grouped_gemm"])
    def test_fp8_4gpus_static_eplb(self, moe_tactic):
        eplb_config = _make_deepseekv4_eplb_config(self.MODEL_PATH,
                                                   layer_updates_per_iter=0,
                                                   ep_size=4)
        _run_deepseekv4_eplb(self.MODEL_NAME, self.MODEL_PATH, moe_tactic,
                             eplb_config)

    @pytest.mark.skip_less_mpi_world_size(4)
    @parametrize_with_ids("moe_tactic", ["deepgemm.grouped_gemm"])
    def test_fp8_4gpus_online_eplb(self, moe_tactic):
        eplb_config = _make_deepseekv4_eplb_config(self.MODEL_PATH,
                                                   layer_updates_per_iter=2,
                                                   ep_size=4)
        _run_deepseekv4_eplb(self.MODEL_NAME, self.MODEL_PATH, moe_tactic,
                             eplb_config)


@pytest.mark.timeout(10800)
@pytest.mark.skip_less_device_memory(100000)
class TestKimiK25(LlmapiAccuracyTestHarness):

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skip_less_device_memory(120000)
    @pytest.mark.parametrize(
        "ep_size,attention_dp",
        [(1, False), (1, True), (8, False), (8, True)],
        ids=["tp8", "tp8_attn_dp", "ep8", "dep8"],
    )
    def test_nvfp4(self, ep_size, attention_dp):
        model_name = "moonshotai/Kimi-K2.5"
        model_path = f"{llm_models_root()}/Kimi-K2.5-NVFP4"
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)

        with LLM(model_path,
                 tensor_parallel_size=8,
                 max_batch_size=16,
                 pipeline_parallel_size=1,
                 moe_expert_parallel_size=ep_size,
                 kv_cache_config=kv_cache_config,
                 enable_attention_dp=attention_dp,
                 trust_remote_code=True,
                 autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                 speculative_config=None) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = MMLU(model_name)
            task.evaluate(llm)
            task = GSM8K(model_name)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skip_less_device_memory(183000)
    @pytest.mark.timeout(14400)
    @pytest.mark.filterwarnings(
        "ignore:.*Calling super.*encode.*add_special_tokens.*:UserWarning")
    @pytest.mark.filterwarnings(
        "ignore:.*configuration is not supported by the fused routing kernel.*:UserWarning"
    )
    def test_nvfp4_longseq_trtllm_moe_stress(self, mocker):
        """Long-sequence MoE stress test with PDL enabled.

        RCCA: https://nvbugspro.nvidia.com/bug/5661741

        Note: the original repro used Kimi-K2-Thinking NVFP4. The model was
        switched to Kimi-K2.5-NVFP4 because Kimi-K2-Thinking NVFP4 is low
        priority, so this is not the original repro configuration.
        """
        patch_mpi_pool_session_for_env(mocker, {"TRTLLM_ENABLE_PDL": "1"})
        model_path = f"{llm_models_root()}/Kimi-K2.5-NVFP4"
        target_len = 250000
        kv_cache_config = KvCacheConfig(
            dtype="fp8",
            free_gpu_memory_fraction=0.75,
            enable_block_reuse=True,
            enable_partial_reuse=False,
            event_buffer_max_size=1024,
        )

        with LLM(
                model_path,
                tensor_parallel_size=8,
                moe_expert_parallel_size=4,
                moe_config=MoeConfig(tactic="trtllm_gen"),
                enable_chunked_prefill=True,
                trust_remote_code=True,
                kv_cache_config=kv_cache_config,
                max_num_tokens=8192,
                max_seq_len=262144,
                max_batch_size=32,
                enable_attention_dp=True,
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            # Build long token sequences from dataset
            tokenizer = llm.tokenizer
            dataset_path = f"{llm_models_root()}/datasets/Crystalcareai/Code-feedback-sharegpt-renamed"
            dataset = load_dataset(dataset_path, split="train[:2000]")
            long_token_list = []
            for row in dataset:
                msg = row["messages"][0]["value"]
                tokens = tokenizer.encode(msg, add_special_tokens=False)
                if not tokens:
                    continue
                repeat = target_len // len(tokens) + 1
                long_tokens = (tokens * repeat)[:target_len]
                long_token_list.append(long_tokens)
            assert len(long_token_list) > 0, "No valid samples found"

            samples_per_batch = 8
            sampling_params_greedy = SamplingParams(max_tokens=8)
            sampling_params_sampling = SamplingParams(max_tokens=8,
                                                      temperature=0.8,
                                                      top_p=0.95)

            num_samples = len(long_token_list)
            max_batch_count = 15

            for batch_idx in range(max_batch_count):
                start_idx = (batch_idx * samples_per_batch) % num_samples
                indices = [(start_idx + i) % num_samples
                           for i in range(samples_per_batch)]
                batch_inputs = [long_token_list[i] for i in indices]

                for output in llm.generate(
                        batch_inputs, sampling_params=sampling_params_greedy):
                    token_ids = output.outputs[0].token_ids
                    assert len(token_ids) > 0
                    assert not all(tid == 0 for tid in token_ids)

                for output in llm.generate(
                        batch_inputs, sampling_params=sampling_params_sampling):
                    token_ids = output.outputs[0].token_ids
                    assert len(token_ids) > 0
                    assert not all(tid == 0 for tid in token_ids)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skip_less_device_memory(183000)
    @pytest.mark.timeout(14400)
    @pytest.mark.filterwarnings(
        "ignore:.*Calling super.*encode.*add_special_tokens.*:UserWarning")
    @pytest.mark.filterwarnings(
        "ignore:.*configuration is not supported by the fused routing kernel.*:UserWarning"
    )
    def test_nvfp4_longseq_trtllm_moe_async_cancel(self, mocker):
        """Long-sequence MoE async streaming test with cancellation.

        RCCA: https://nvbugspro.nvidia.com/bug/5661741

        Note: the original repro used Kimi-K2-Thinking NVFP4. The model was
        switched to Kimi-K2.5-NVFP4 because Kimi-K2-Thinking NVFP4 is low
        priority, so this is not the original repro configuration.
        """
        patch_mpi_pool_session_for_env(mocker, {"TRTLLM_ENABLE_PDL": "1"})
        model_path = f"{llm_models_root()}/Kimi-K2.5-NVFP4"
        target_len = 250000
        kv_cache_config = KvCacheConfig(
            dtype="fp8",
            free_gpu_memory_fraction=0.75,
            enable_block_reuse=True,
            enable_partial_reuse=False,
            event_buffer_max_size=1024,
        )

        with LLM(
                model_path,
                tensor_parallel_size=8,
                moe_expert_parallel_size=4,
                moe_config=MoeConfig(tactic="trtllm_gen"),
                enable_chunked_prefill=True,
                trust_remote_code=True,
                kv_cache_config=kv_cache_config,
                max_num_tokens=4096,
                max_seq_len=262144,
                max_batch_size=8,
                enable_attention_dp=True,
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            # Build long token sequences from dataset
            tokenizer = llm.tokenizer
            dataset_path = f"{llm_models_root()}/datasets/Crystalcareai/Code-feedback-sharegpt-renamed"
            dataset = load_dataset(dataset_path, split="train[:2000]")
            long_token_list = []
            for row in dataset:
                msg = row["messages"][0]["value"]
                tokens = tokenizer.encode(msg, add_special_tokens=False)
                if not tokens:
                    continue
                repeat = target_len // len(tokens) + 1
                long_tokens = (tokens * repeat)[:target_len]
                long_token_list.append(long_tokens)
            assert len(long_token_list) > 0, "No valid samples found"

            async_batch_size = 6
            num_async_batches = 3
            cancel_ratio = 0.5
            num_samples = len(long_token_list)

            async def handle_one_request(async_gen, should_cancel):
                chunks_received = 0
                max_chunks_before_cancel = 5
                try:
                    async for chunk in async_gen:
                        chunks_received += 1
                        if chunk.outputs:
                            token_ids = chunk.outputs[0].token_ids
                            assert len(token_ids) > 0
                            assert not all(tid == 0 for tid in token_ids)
                        if should_cancel and chunks_received >= max_chunks_before_cancel:
                            break
                except Exception:
                    if not should_cancel:
                        raise

            async def run_streaming_with_cancellation():
                for async_batch_idx in range(num_async_batches):
                    start_idx = (async_batch_idx *
                                 async_batch_size) % num_samples
                    indices = [(start_idx + i) % num_samples
                               for i in range(async_batch_size)]
                    batch_inputs = [long_token_list[i] for i in indices]

                    sampling_params = SamplingParams(max_tokens=50,
                                                     temperature=0.8,
                                                     top_p=0.95)
                    async_results = [
                        llm.generate_async(inp,
                                           sampling_params=sampling_params,
                                           streaming=True)
                        for inp in batch_inputs
                    ]

                    tasks = [
                        asyncio.create_task(
                            handle_one_request(
                                gen, idx < async_batch_size * cancel_ratio))
                        for idx, gen in enumerate(async_results)
                    ]

                    await asyncio.wait_for(asyncio.gather(*tasks), timeout=300)

            asyncio.run(run_streaming_with_cancellation())

            # Verify LLM still works after cancellations (bug 5661741 symptom check)
            verify_batch_size = 4
            verify_inputs = [
                long_token_list[i % num_samples]
                for i in range(verify_batch_size)
            ]
            verify_params = SamplingParams(max_tokens=16)

            for output in llm.generate(verify_inputs,
                                       sampling_params=verify_params):
                token_ids = output.outputs[0].token_ids
                assert len(token_ids) > 0
                assert not all(tid == 0 for tid in token_ids)


class TestLagunaXS(LlmapiAccuracyTestHarness):
    MODEL_NAME = "poolside/laguna-XS.2"
    MODEL_PATH = f"{llm_models_root()}/Laguna-XS.2"
    FP8_MODEL_PATH = f"{llm_models_root()}/Laguna-XS.2-FP8"
    NVFP4_MODEL_PATH = f"{llm_models_root()}/Laguna-XS.2-NVFP4"
    MAX_SEQ_LEN = 4096
    MAX_NUM_TOKENS = 4096
    MAX_BATCH_SIZE = 128

    def _run_accuracy(self,
                      model_path,
                      expected_quant_algo=None,
                      expected_kv_cache_quant_algo=None):
        if not os.path.exists(model_path):
            pytest.skip(f"Model directory {model_path} does not exist")

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)

        with LLM(model_path,
                 max_seq_len=self.MAX_SEQ_LEN,
                 max_num_tokens=self.MAX_NUM_TOKENS,
                 max_batch_size=self.MAX_BATCH_SIZE,
                 kv_cache_config=kv_cache_config) as llm:
            if expected_quant_algo is not None:
                assert llm.args.quant_config.quant_algo == expected_quant_algo
            if expected_kv_cache_quant_algo is not None:
                assert (llm.args.quant_config.kv_cache_quant_algo ==
                        expected_kv_cache_quant_algo)

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_hopper
    @pytest.mark.skip_less_device_memory(80000)
    def test_bf16(self):
        self._run_accuracy(self.MODEL_PATH)

    @skip_pre_hopper
    @pytest.mark.skip_less_device_memory(80000)
    def test_fp8(self):
        self._run_accuracy(self.FP8_MODEL_PATH,
                           expected_quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                           expected_kv_cache_quant_algo=QuantAlgo.FP8)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device_memory(80000)
    def test_nvfp4(self):
        self._run_accuracy(self.NVFP4_MODEL_PATH,
                           expected_quant_algo=QuantAlgo.NVFP4,
                           expected_kv_cache_quant_algo=QuantAlgo.FP8)


class TestLagunaXS_2_1(LlmapiAccuracyTestHarness):
    MODEL_NAME = "poolside/Laguna-XS-2.1"
    POOLSIDE_PATH = f"{llm_models_root()}/poolside"

    MODEL_PATH = f"{POOLSIDE_PATH}/Laguna-XS-2.1"
    DFLASH_MODEL_PATH = f"{POOLSIDE_PATH}/Laguna-XS-2.1-DFlash"
    FP8_MODEL_PATH = f"{POOLSIDE_PATH}/Laguna-XS-2.1-FP8"
    FP8_DFLASH_MODEL_PATH = f"{POOLSIDE_PATH}/Laguna-XS-2.1-DFlash-FP8"
    NVFP4_MODEL_PATH = f"{POOLSIDE_PATH}/Laguna-XS-2.1-NVFP4"
    NVFP4_DFLASH_MODEL_PATH = f"{POOLSIDE_PATH}/Laguna-XS-2.1-DFlash-NVFP4"

    GSM8K_EVALUATOR_KWARGS = dict(apply_chat_template=True)

    def _run_dflash_accuracy(self,
                             model_path,
                             dflash_model_path,
                             expected_quant_algo=None,
                             expected_kv_cache_quant_algo=None):

        spec_config = DFlashDecodingConfig(
            max_draft_len=4,
            speculative_model=dflash_model_path,
        )
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9,
                                        enable_block_reuse=False,
                                        use_kv_cache_manager_v2=True)

        with LLM(model_path,
                 max_seq_len=4096,
                 max_num_tokens=4096,
                 max_batch_size=128,
                 kv_cache_config=kv_cache_config,
                 speculative_config=spec_config) as llm:
            if expected_quant_algo is not None:
                assert llm.args.quant_config.quant_algo == expected_quant_algo
            if expected_kv_cache_quant_algo is not None:
                assert (llm.args.quant_config.kv_cache_quant_algo ==
                        expected_kv_cache_quant_algo)

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.GSM8K_EVALUATOR_KWARGS)

    @pytest.mark.skip_less_device_memory(90000)
    def test_bf16_dflash(self):
        self._run_dflash_accuracy(self.MODEL_PATH, self.DFLASH_MODEL_PATH)

    @skip_pre_hopper
    @pytest.mark.skip_less_device_memory(80000)
    def test_fp8_dflash(self):
        self._run_dflash_accuracy(
            self.FP8_MODEL_PATH,
            self.FP8_DFLASH_MODEL_PATH,
            expected_quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
            expected_kv_cache_quant_algo=QuantAlgo.FP8)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device_memory(80000)
    def test_nvfp4_dflash(self):
        self._run_dflash_accuracy(self.NVFP4_MODEL_PATH,
                                  self.NVFP4_DFLASH_MODEL_PATH,
                                  expected_quant_algo=QuantAlgo.NVFP4,
                                  expected_kv_cache_quant_algo=QuantAlgo.FP8)


@pytest.mark.skip_less_device_memory(80000)
@skip_pre_blackwell
class TestQwen3_6_35B_A3B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
    MODEL_PATH = f"{llm_models_root()}/Qwen3.6-35B-A3B-NVFP4"
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        chat_template_kwargs=dict(enable_thinking=False),
    )
    GSM8K_MAX_OUTPUT_LEN = 512

    def _run_gsm8k(self, model_path: str, moe_tactic: str,
                   chunked_prefill: bool, mocker: MockerFixture) -> None:
        """Run single-GPU GSM8K on an NVFP4 checkpoint."""
        if not os.path.exists(model_path):
            pytest.skip(f"Model directory {model_path} does not exist")

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8,
                                        enable_block_reuse=False)
        cuda_graph_config = CudaGraphConfig(enable_padding=True,
                                            max_batch_size=128)
        moe_config = MoeConfig(tactic=moe_tactic)

        with LLM(model_path,
                 trust_remote_code=True,
                 tensor_parallel_size=1,
                 moe_expert_parallel_size=1,
                 max_seq_len=4096,
                 max_batch_size=128,
                 max_num_tokens=512 if chunked_prefill else 8192,
                 kv_cache_config=kv_cache_config,
                 cuda_graph_config=cuda_graph_config,
                 enable_chunked_prefill=chunked_prefill,
                 autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                 moe_config=moe_config) as llm:
            mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN",
                                self.GSM8K_MAX_OUTPUT_LEN)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @pytest.mark.parametrize("moe_tactic",
                             ["trtllm_gen", "cutedsl.grouped_gemm"])
    def test_nvfp4(self, moe_tactic, mocker):
        # TRTLLM-Gen / CuteDSL NVFP4 MoE backends only support SM100/103 family
        if get_sm_version() not in (100, 103):
            pytest.skip("Qwen3.6-35B-A3B NVFP4 MoE test runs on SM100/103 only")

        self._run_gsm8k(self.MODEL_PATH, moe_tactic, False, mocker)

    @skip_no_sm120
    def test_nvfp4_w4a16(self, mocker):
        self._run_gsm8k(self.MODEL_PATH, "b12x", True, mocker)

    @pytest.mark.parametrize("dflash_attention_backend", ["VANILLA", "TRTLLM"])
    def test_nvfp4_dflash(self, mocker, dflash_attention_backend):
        if get_sm_version() not in (100, 103):
            pytest.skip(
                "Qwen3.6-35B-A3B NVFP4 DFlash test runs on SM100/SM103 only")

        dflash_model_path = f"{llm_models_root()}/Qwen3.6-35B-A3B-DFlash"
        if not os.path.exists(
                self.MODEL_PATH) or not os.path.exists(dflash_model_path):
            pytest.skip("Qwen3.6-35B-A3B NVFP4 target or DFlash draft model "
                        "directory does not exist")

        mocker.patch.dict(os.environ, {"TRTLLM_USE_GDN_REPLAY": "1"})
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=0.8,
            enable_block_reuse=False,
            dtype="fp8",
            mamba_ssm_cache_dtype="bfloat16",
        )
        cuda_graph_config = CudaGraphConfig(enable_padding=True,
                                            max_batch_size=8)
        spec_config = DFlashDecodingConfig(
            max_draft_len=7,
            speculative_model=dflash_model_path,
            attention_backend=dflash_attention_backend)

        with LLM(self.MODEL_PATH,
                 trust_remote_code=True,
                 tensor_parallel_size=1,
                 moe_expert_parallel_size=1,
                 max_seq_len=8192,
                 max_batch_size=8,
                 enable_chunked_prefill=True,
                 kv_cache_config=kv_cache_config,
                 cuda_graph_config=cuda_graph_config,
                 moe_config=MoeConfig(tactic="trtllm_gen"),
                 autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                 speculative_config=spec_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
            assert llm.args.speculative_config.decoding_type == 'DFlash'
            mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN",
                                self.GSM8K_MAX_OUTPUT_LEN)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @pytest.mark.timeout(3600)
    @skip_no_sm100f
    @pytest.mark.parametrize("mode", ["baseline", "mtp3", "dflash7"])
    def test_gpqa_diamond(
        self,
        mode: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Evaluate sampled reasoning with CUDA graphs and speculative decoding."""
        model_root = Path(llm_models_root())
        model_path = Path(self.MODEL_PATH)
        assert model_path.is_dir(), f"Missing target checkpoint: {model_path}"

        spec_config = None
        if mode == "mtp3":
            spec_config = MTPDecodingConfig(num_nextn_predict_layers=3,
                                            use_rejection_sampling=False)
        elif mode == "dflash7":
            draft_path = model_root / "Qwen3.6-35B-A3B-DFlash"
            assert draft_path.is_dir(
            ), f"Missing DFlash checkpoint: {draft_path}"
            spec_config = DFlashDecodingConfig(
                speculative_model=str(draft_path), max_draft_len=7)

        monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
        max_seq_len = GPQADiamond.MAX_INPUT_LEN + GPQADiamond.MAX_OUTPUT_LEN
        with LLM(
                str(model_path),
                trust_remote_code=True,
                tensor_parallel_size=1,
                moe_expert_parallel_size=1,
                max_batch_size=16,
                max_seq_len=max_seq_len,
                max_num_tokens=8192,
                enable_chunked_prefill=True,
                cuda_graph_config=CudaGraphConfig(enable_padding=True,
                                                  max_batch_size=16),
                kv_cache_config=KvCacheConfig(
                    enable_block_reuse=False,
                    free_gpu_memory_fraction=0.9,
                    dtype="fp8",
                    mamba_ssm_cache_dtype="bfloat16",
                    use_kv_cache_manager_v2=False,
                ),
                moe_config=MoeConfig(backend="TRTLLM"),
                speculative_config=spec_config,
        ) as llm:
            task = GPQADiamond(self.MODEL_NAME)
            # Caller temperature and top_p take precedence over the task's greedy defaults.
            task.EVALUATE_KWARGS = dict(getattr(task, "EVALUATE_KWARGS", {}),
                                        sampling_override=True)
            task.evaluate(
                llm,
                sampling_params=SamplingParams(
                    temperature=1.0,
                    top_p=0.95,
                    seed=0,
                    max_tokens=GPQADiamond.MAX_OUTPUT_LEN,
                    truncate_prompt_tokens=GPQADiamond.MAX_INPUT_LEN,
                ),
                extra_evaluator_kwargs=dict(
                    apply_chat_template=True,
                    chat_template_kwargs=dict(enable_thinking=True),
                ),
            )


@pytest.mark.timeout(28800)
class TestQwen3_8_2_4T_A95B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen3.8-2.4T-A95B"
    GSM8K_MAX_OUTPUT_LEN = 512
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        system_prompt=("Use at most three short reasoning sentences, then "
                       "end with `#### NUMBER`. Do not restate the problem."),
        chat_template_kwargs=dict(enable_thinking=True,
                                  reasoning_effort="xhigh"),
    )

    def _run_fp8_block_scales(self, tensor_parallel_size: int,
                              pipeline_parallel_size: int, ep_size: int,
                              attention_dp: bool, moe_tactic: str,
                              max_draft_len: Optional[int],
                              mocker: MockerFixture) -> None:
        model_path = f"{llm_models_root()}/Qwen3.8-2.4T-A95B-FP8"
        if not os.path.exists(model_path):
            pytest.skip(f"Model directory {model_path} does not exist")

        max_batch_size = 4 if attention_dp else 32
        kv_cache_fraction = 0.5 if max_draft_len is not None else 0.75
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=kv_cache_fraction,
            enable_block_reuse=False,
            mamba_ssm_cache_dtype="bfloat16")
        cuda_graph_config = CudaGraphConfig(max_batch_size=max_batch_size,
                                            enable_padding=True)
        if max_draft_len is None:
            mtp_config = None
        else:
            # The target's tactic does not reach the MTP layer, so it pins
            # it too.
            mtp_config = MTPDecodingConfig(max_draft_len=max_draft_len,
                                           moe_tactic=moe_tactic)

        with LLM(model_path,
                 trust_remote_code=True,
                 tensor_parallel_size=tensor_parallel_size,
                 pipeline_parallel_size=pipeline_parallel_size,
                 moe_expert_parallel_size=ep_size,
                 max_seq_len=8192,
                 max_num_tokens=8192,
                 max_batch_size=max_batch_size,
                 enable_attention_dp=attention_dp,
                 kv_cache_config=kv_cache_config,
                 cuda_graph_config=cuda_graph_config,
                 moe_config=MoeConfig(tactic=moe_tactic),
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
            mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN",
                                self.GSM8K_MAX_OUTPUT_LEN)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_x86
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skipif(
        get_sm_version() not in (100, 103, 107),
        reason="This test is supported only on SM100, SM103 and SM107")
    def test_fp8_tp16_mtp3_trtllm(self, mocker):
        # The default AUTO uses MNNVL only when all TP ranks share one NVL72 fabric.
        self._run_fp8_block_scales(tensor_parallel_size=16,
                                   pipeline_parallel_size=1,
                                   ep_size=1,
                                   attention_dp=False,
                                   moe_tactic="trtllm_gen",
                                   max_draft_len=3,
                                   mocker=mocker)

    @skip_x86
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skipif(get_sm_version() not in (103, 107),
                        reason="This test is supported only on SM103 and SM107")
    def test_fp8_adp16_deepgemm(self, mocker):
        self._run_fp8_block_scales(tensor_parallel_size=16,
                                   pipeline_parallel_size=1,
                                   ep_size=16,
                                   attention_dp=True,
                                   moe_tactic="deepgemm.grouped_gemm",
                                   max_draft_len=None,
                                   mocker=mocker)

    def _run_nvfp4(self, tensor_parallel_size: int, ep_size: int,
                   attention_dp: bool, moe_tactic: str,
                   max_draft_len: Optional[int], mocker: MockerFixture) -> None:
        model_path = f"{llm_models_root()}/Inferact-Qwen3.8-2.4T-A95B-NVFP4"
        if not os.path.exists(model_path):
            pytest.fail(f"Model directory {model_path} does not exist")

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8,
                                        enable_block_reuse=False,
                                        mamba_ssm_cache_dtype="bfloat16")
        max_batch_size = 4 if attention_dp else 32
        cuda_graph_config = CudaGraphConfig(max_batch_size=max_batch_size,
                                            enable_padding=True)
        if max_draft_len is None:
            mtp_config = None
        else:
            # The target's tactic does not reach the MTP layer, so it pins
            # it too.
            mtp_config = MTPDecodingConfig(max_draft_len=max_draft_len,
                                           moe_tactic=moe_tactic)

        with LLM(model_path,
                 trust_remote_code=True,
                 tensor_parallel_size=tensor_parallel_size,
                 moe_expert_parallel_size=ep_size,
                 max_seq_len=8192,
                 max_num_tokens=8192,
                 max_batch_size=max_batch_size,
                 enable_attention_dp=attention_dp,
                 kv_cache_config=kv_cache_config,
                 cuda_graph_config=cuda_graph_config,
                 moe_config=MoeConfig(tactic=moe_tactic),
                 autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                 speculative_config=mtp_config) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN",
                                self.GSM8K_MAX_OUTPUT_LEN)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_x86
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skipif(get_sm_version() not in (103, 107),
                        reason="This test is supported only on SM103 and SM107")
    def test_nvfp4_tp8_mtp3_trtllm(self, mocker):
        self._run_nvfp4(tensor_parallel_size=8,
                        ep_size=1,
                        attention_dp=False,
                        moe_tactic="trtllm_gen",
                        max_draft_len=3,
                        mocker=mocker)

    @skip_x86
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skipif(get_sm_version() not in (103, 107),
                        reason="This test is supported only on SM103 and SM107")
    def test_nvfp4_adp16_cutedsl(self, mocker):
        self._run_nvfp4(tensor_parallel_size=16,
                        ep_size=16,
                        attention_dp=True,
                        moe_tactic="cutedsl.grouped_gemm",
                        max_draft_len=None,
                        mocker=mocker)


@pytest.mark.timeout(28800)
class TestQwen3_8_Flash_Next(LlmapiAccuracyTestHarness):
    """Qwen3.8-Flash-Next: Gated-DeltaNet + QSA hybrid with MoE, PLE, and MTP3."""

    MODEL_NAME = "Qwen/Qwen3.8-Flash-Next"
    MAX_NUM_TOKENS = 8192
    MAX_BATCH_SIZE = 16
    GSM8K_MAX_OUTPUT_LEN = 512
    GSM8K_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        system_prompt=("Use at most three short reasoning sentences, then "
                       "end with `#### NUMBER`. Do not restate the problem."),
        chat_template_kwargs=dict(enable_thinking=True,
                                  reasoning_effort="xhigh"),
    )

    MMLU_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt=("Answer with a single letter: A, B, C, or D. "
                       "Output only that letter and nothing else."),
        chat_template_kwargs=dict(enable_thinking=False),
    )

    def _build_llm(
        self,
        model_path: str,
        tensor_parallel_size: int,
        moe_backend: str,
        max_draft_len: Optional[int],
        check_acceptance_length: bool,
        moe_expert_parallel_size: int = 1,
        enable_attention_dp: bool = False,
        cover_guided_decoding: bool = False,
        enable_block_reuse: bool = False,
        sparse_attention_config: Optional[QSASparseAttentionConfig] = None,
        kv_cache_dtype: str = "auto",
        enable_nvmmh: bool = False,
    ) -> LLM:
        """Construct the engine shared by both evaluation tasks.

        Leaving `sparse_attention_config` unset keeps the model default, which
        is QSA with every indexer knob at its default.
        """
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5,
                                        enable_block_reuse=enable_block_reuse,
                                        dtype=kv_cache_dtype,
                                        mamba_ssm_cache_dtype="bfloat16")
        if enable_block_reuse:
            # Attention pages alone cannot restore GDN or PLE state, so the
            # runtime turns reuse back off without a snapshot policy.
            kv_cache_config.mamba_state_config.periodic_snapshot_interval = 256
        cuda_graph_config = CudaGraphConfig(max_batch_size=self.MAX_BATCH_SIZE,
                                            enable_padding=True)
        mtp_config = (None if max_draft_len is None else MTPDecodingConfig(
            max_draft_len=max_draft_len))
        stats_args = (dict(max_stats_len=-1, enable_iter_perf_stats=True)
                      if check_acceptance_length else {})
        guided_decoding_args = ({
            "guided_decoding_backend": "xgrammar"
        } if cover_guided_decoding else {})
        sparse_args = ({} if sparse_attention_config is None else {
            "sparse_attention_config": sparse_attention_config
        })
        return LLM(
            model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            moe_expert_parallel_size=moe_expert_parallel_size,
            enable_attention_dp=enable_attention_dp,
            max_num_tokens=self.MAX_NUM_TOKENS,
            enable_chunked_prefill=True,
            max_batch_size=self.MAX_BATCH_SIZE,
            kv_cache_config=kv_cache_config,
            cuda_graph_config=cuda_graph_config,
            # Stays on backend literals: on SM103 the default
            # modeling_v2 routes the NVFP4 checkpoint to the
            # Qwen3.8-Flash-Next target, which honors no tactic, so a
            # tactic would build the built-in model instead.
            moe_config=MoeConfig(backend=moe_backend),
            autotuner_nvmmh_config=(AutoTunerNvMMHConfig()
                                    if enable_nvmmh else None),
            speculative_config=mtp_config,
            **sparse_args,
            **stats_args,
            **guided_decoding_args)

    def _run_evals(
            self,
            model_path: str,
            tensor_parallel_size: int,
            moe_backend: str,
            max_draft_len: Optional[int],
            expected_quant_algo: Optional[QuantAlgo],
            monkeypatch: pytest.MonkeyPatch,
            mocker,
            moe_expert_parallel_size: int = 1,
            enable_attention_dp: bool = False,
            cover_guided_decoding: bool = False,
            enable_block_reuse: bool = False,
            sparse_attention_config: Optional[QSASparseAttentionConfig] = None,
            acceptance_length_key: Optional[str] = None,
            kv_cache_dtype: str = "auto",
            ple_host_offload: bool = True,
            enable_nvmmh: bool = False) -> None:
        if not os.path.exists(model_path):
            pytest.skip(f"Model directory {model_path} does not exist")

        monkeypatch.setenv("TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD",
                           "1" if ple_host_offload else "0")

        # A case opts into the acceptance-length guard by naming its own entry
        # in references/acceptance_length.yaml.
        check_acceptance_length = acceptance_length_key is not None
        with self._build_llm(model_path,
                             tensor_parallel_size,
                             moe_backend,
                             max_draft_len,
                             check_acceptance_length,
                             moe_expert_parallel_size=moe_expert_parallel_size,
                             enable_attention_dp=enable_attention_dp,
                             cover_guided_decoding=cover_guided_decoding,
                             enable_block_reuse=enable_block_reuse,
                             sparse_attention_config=sparse_attention_config,
                             kv_cache_dtype=kv_cache_dtype,
                             enable_nvmmh=enable_nvmmh) as llm:
            assert llm.args.quant_config.quant_algo == expected_quant_algo
            if cover_guided_decoding:
                assert_guided_decoding_regex(llm)
            mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN",
                                self.GSM8K_MAX_OUTPUT_LEN)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.GSM8K_EVALUATOR_KWARGS)
            if check_acceptance_length:
                assert_acceptance_length_for_llm(
                    f"TestQwen3_8_Flash_Next::{acceptance_length_key}",
                    llm,
                )
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.MMLU_EVALUATOR_KWARGS)

    @skip_pre_hopper
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(100000)
    @pytest.mark.skip_less_host_memory(131072)
    def test_bf16_tep4_cutlass(self, monkeypatch: pytest.MonkeyPatch,
                               mocker) -> None:
        """BF16 TEP4, no MTP, PLE offloaded to pinned host memory."""
        self._run_evals(f"{llm_models_root()}/Qwen3.8-Flash-Next",
                        tensor_parallel_size=4,
                        moe_backend="CUTLASS",
                        max_draft_len=None,
                        expected_quant_algo=None,
                        monkeypatch=monkeypatch,
                        mocker=mocker,
                        moe_expert_parallel_size=4,
                        cover_guided_decoding=True)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(82000)
    @pytest.mark.skip_less_host_memory(98304)
    def test_fp8_adp4_mtp3_trtllm_ple_offload(self,
                                              monkeypatch: pytest.MonkeyPatch,
                                              mocker) -> None:
        """Block-FP8 on four GPUs with attention DP, MTP3 and PLE offload."""
        self._run_evals(
            f"{llm_models_root()}/Qwen3.8-Flash-Next-FP8",
            tensor_parallel_size=4,
            moe_backend="TRTLLM",
            max_draft_len=3,
            expected_quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
            monkeypatch=monkeypatch,
            mocker=mocker,
            moe_expert_parallel_size=4,
            enable_attention_dp=True,
            acceptance_length_key="test_fp8_adp4_mtp3_trtllm_ple_offload")

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(82000)
    @pytest.mark.skip_less_host_memory(98304)
    def test_fp8_adp4_mtp3_prefix_cache(self, monkeypatch: pytest.MonkeyPatch,
                                        mocker) -> None:
        """test_fp8_adp4_mtp3_trtllm_ple_offload with block reuse enabled.

        A reused prefix restores the Gated DeltaNet and PLE state from a
        recurrent-state snapshot instead of recomputing it, so accuracy holds
        only if the snapshot carries every role in the recurrent page.
        """
        self._run_evals(f"{llm_models_root()}/Qwen3.8-Flash-Next-FP8",
                        tensor_parallel_size=4,
                        moe_backend="TRTLLM",
                        max_draft_len=3,
                        expected_quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                        monkeypatch=monkeypatch,
                        mocker=mocker,
                        moe_expert_parallel_size=4,
                        enable_attention_dp=True,
                        enable_block_reuse=True)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(70000)
    def test_nvfp4_adp4_mtp3_trtllm(self, monkeypatch: pytest.MonkeyPatch,
                                    mocker) -> None:
        """NVFP4 on four GPUs with attention DP, MTP3 and the PLE table on GPU.

        Blackwell-only, so the GVR Top-K prerequisites hold. This is the case
        that keeps both indexer options under accuracy guard: the heuristic
        Top-K, and the draft loop reusing one selection.

        The checkpoint stores the PLE n-gram table as scaled FP8. Under
        attention DP each rank owns a row slice of it and masks the rows it
        does not own before the reduce-scatter, while the rows are still FP8.
        """
        self._run_evals(f"{llm_models_root()}/Qwen3.8-Flash-Next-NVFP4",
                        tensor_parallel_size=4,
                        moe_backend="TRTLLM",
                        max_draft_len=3,
                        expected_quant_algo=QuantAlgo.MIXED_PRECISION,
                        monkeypatch=monkeypatch,
                        mocker=mocker,
                        moe_expert_parallel_size=4,
                        enable_attention_dp=True,
                        sparse_attention_config=QSASparseAttentionConfig(
                            enable_heuristic_topk=True,
                            index_share_for_mtp_iteration=True),
                        acceptance_length_key="test_nvfp4_adp4_mtp3_trtllm",
                        ple_host_offload=False,
                        enable_nvmmh=True)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(70000)
    @pytest.mark.skip_less_host_memory(131072)
    def test_nvfp4_adp4_cutlass_ple_offload(self,
                                            monkeypatch: pytest.MonkeyPatch,
                                            mocker) -> None:
        """NVFP4 with attention DP4, expert EP4, and PLE host offload."""
        self._run_evals(f"{llm_models_root()}/Qwen3.8-Flash-Next-NVFP4",
                        tensor_parallel_size=4,
                        moe_backend="CUTLASS",
                        max_draft_len=None,
                        expected_quant_algo=QuantAlgo.MIXED_PRECISION,
                        monkeypatch=monkeypatch,
                        mocker=mocker,
                        moe_expert_parallel_size=4,
                        enable_attention_dp=True,
                        enable_nvmmh=True)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device_memory(100000)
    @pytest.mark.skip_less_host_memory(131072)
    def test_nvfp4_1gpu_cutedsl_ple_offload(self,
                                            monkeypatch: pytest.MonkeyPatch,
                                            mocker) -> None:
        """NVFP4 on one GPU without MTP and the PLE table offloaded to host.

        CuteDSL serves NVFP4 but not the FP8 block scales of the MTP layer's
        experts, so this checkpoint only fits the backend without MTP.
        """
        self._run_evals(f"{llm_models_root()}/Qwen3.8-Flash-Next-NVFP4",
                        tensor_parallel_size=1,
                        moe_backend="CUTEDSL",
                        max_draft_len=None,
                        expected_quant_algo=QuantAlgo.MIXED_PRECISION,
                        monkeypatch=monkeypatch,
                        mocker=mocker,
                        enable_nvmmh=True)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(2)
    @pytest.mark.skip_less_device_memory(100000)
    @pytest.mark.skip_less_host_memory(98304)
    def test_nvfp4_tp2_cutedsl(self, monkeypatch: pytest.MonkeyPatch,
                               mocker) -> None:
        """CuteDSL cannot serve the MTP layer's FP8 experts, so run without MTP."""
        self._run_evals(f"{llm_models_root()}/Qwen3.8-Flash-Next-NVFP4",
                        tensor_parallel_size=2,
                        moe_backend="CUTEDSL",
                        max_draft_len=None,
                        expected_quant_algo=QuantAlgo.MIXED_PRECISION,
                        monkeypatch=monkeypatch,
                        mocker=mocker,
                        moe_expert_parallel_size=2,
                        ple_host_offload=False)

    @skip_pre_hopper
    @pytest.mark.skip_less_mpi_world_size(2)
    @pytest.mark.skip_less_device_memory(100000)
    @pytest.mark.skip_less_host_memory(98304)
    def test_fp8_tp2_cutlass(self, monkeypatch: pytest.MonkeyPatch,
                             mocker) -> None:
        """MTP at TP2 fails in the FP8 block-scale GEMM (K=320), so run without MTP."""
        self._run_evals(f"{llm_models_root()}/Qwen3.8-Flash-Next-FP8",
                        tensor_parallel_size=2,
                        moe_backend="CUTLASS",
                        max_draft_len=None,
                        expected_quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                        monkeypatch=monkeypatch,
                        mocker=mocker,
                        moe_expert_parallel_size=2,
                        ple_host_offload=False)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(70000)
    @pytest.mark.skip_less_host_memory(131072)
    def test_nvfp4_adp4_megamoe_cutedsl_ple_offload(
            self, monkeypatch: pytest.MonkeyPatch, mocker) -> None:
        """NVFP4 on four GPUs with attention DP on the fused-comm MegaMoE kernel.

        MegaMoE-CuteDSL serves NVFP4 only, so the MTP layer's FP8 block-scale
        experts keep this checkpoint on the no-MTP configuration.
        """
        self._run_evals(f"{llm_models_root()}/Qwen3.8-Flash-Next-NVFP4",
                        tensor_parallel_size=4,
                        moe_backend="MEGAMOE_CUTEDSL",
                        max_draft_len=None,
                        expected_quant_algo=QuantAlgo.MIXED_PRECISION,
                        monkeypatch=monkeypatch,
                        mocker=mocker,
                        moe_expert_parallel_size=4,
                        enable_attention_dp=True,
                        enable_nvmmh=True)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(70000)
    @pytest.mark.skip_less_host_memory(131072)
    def test_nvfp4_tep4_fp8kv_trtllm_ple_offload(
            self, monkeypatch: pytest.MonkeyPatch, mocker) -> None:
        """NVFP4 with TEP4 and an FP8 KV cache.

        Each rank holds one KV head, so an FP8 K/V page is the size of a QSA
        index-K page.
        """
        self._run_evals(f"{llm_models_root()}/Qwen3.8-Flash-Next-NVFP4",
                        tensor_parallel_size=4,
                        moe_backend="TRTLLM",
                        max_draft_len=None,
                        expected_quant_algo=QuantAlgo.MIXED_PRECISION,
                        monkeypatch=monkeypatch,
                        mocker=mocker,
                        moe_expert_parallel_size=4,
                        kv_cache_dtype="fp8",
                        enable_nvmmh=True)

    @skip_pre_blackwell
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(70000)
    @pytest.mark.skip_less_host_memory(131072)
    def test_nvfp4_adp4_nvfp4kv_trtllm_ple_offload(
            self, monkeypatch: pytest.MonkeyPatch, mocker) -> None:
        """NVFP4 on four GPUs with attention DP and an NVFP4 KV cache.

        Each rank holds both KV heads, so an NVFP4 K/V page is the size of a
        QSA index-K page. QSA attention runs on the regular backend for an
        NVFP4 KV cache.
        """
        self._run_evals(f"{llm_models_root()}/Qwen3.8-Flash-Next-NVFP4",
                        tensor_parallel_size=4,
                        moe_backend="TRTLLM",
                        max_draft_len=None,
                        expected_quant_algo=QuantAlgo.MIXED_PRECISION,
                        monkeypatch=monkeypatch,
                        mocker=mocker,
                        moe_expert_parallel_size=4,
                        enable_attention_dp=True,
                        kv_cache_dtype="nvfp4",
                        enable_nvmmh=True)


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(183000)
@pytest.mark.timeout(28800)
class TestDeepSeekR1LongBenchV2(LlmapiAccuracyTestHarness):
    MODEL_NAME = "DeepSeek-R1-0528"

    @pytest.mark.skip_less_mpi_world_size(8)
    def test_fp8_8gpus(self):
        model_dir = f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1-0528"
        if not os.path.exists(model_dir):
            pytest.skip(f"Model directory {model_dir} does not exist")

        # Configure model settings
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4,
                                        enable_block_reuse=True,
                                        enable_partial_reuse=False,
                                        dtype="fp8")

        cuda_graph_config = CudaGraphConfig(enable_padding=True,
                                            max_batch_size=32)

        mtp_config = MTPDecodingConfig(max_draft_len=3,
                                       moe_tactic=fp8_block_scales_moe_tactic())

        moe_config = fp8_block_scales_moe_config(max_num_tokens=32000)

        pytorch_config = dict(cuda_graph_config=cuda_graph_config,
                              kv_cache_config=kv_cache_config,
                              speculative_config=mtp_config,
                              moe_config=moe_config,
                              enable_chunked_prefill=True,
                              enable_autotuner=True)

        # Create LLM instance and evaluate
        with LLM(model_dir,
                 tensor_parallel_size=8,
                 moe_expert_parallel_size=8,
                 max_num_tokens=32000,
                 max_batch_size=32,
                 **pytorch_config) as llm:

            task = LongBenchV2(self.MODEL_NAME)

            sampling_params = SamplingParams(
                max_tokens=32000,
                truncate_prompt_tokens=128000,
            )

            task.evaluate(llm, sampling_params=sampling_params)

    @pytest.mark.skip_less_mpi_world_size(4)
    @pytest.mark.private_mpi_session
    def test_nvfp4_4gpus(self):
        model_dir = f"{llm_models_root()}/DeepSeek-R1/DeepSeek-R1-0528-FP4"
        if not os.path.exists(model_dir):
            pytest.skip(f"Model directory {model_dir} does not exist")

        # Configure model settings (no MOE config for FP4 version)
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8,
                                        enable_block_reuse=True,
                                        enable_partial_reuse=False,
                                        dtype="fp8")

        cuda_graph_config = CudaGraphConfig(enable_padding=True,
                                            max_batch_size=32)

        mtp_config = MTPDecodingConfig(max_draft_len=3)

        pytorch_config = dict(cuda_graph_config=cuda_graph_config,
                              kv_cache_config=kv_cache_config,
                              speculative_config=mtp_config,
                              enable_chunked_prefill=True,
                              enable_autotuner=True)

        # Create LLM instance and evaluate
        with LLM(model_dir,
                 tensor_parallel_size=4,
                 moe_expert_parallel_size=4,
                 max_num_tokens=32000,
                 max_batch_size=32,
                 **pytorch_config) as llm:

            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4

            task = LongBenchV2(self.MODEL_NAME)

            sampling_params = SamplingParams(
                max_tokens=32000,
                truncate_prompt_tokens=128000,
            )

            task.evaluate(llm, sampling_params=sampling_params)


class TestMistralLarge3_675B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistral/Mistral-Large-3-675B"

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @pytest.mark.skip_less_device_memory(183000)
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler,moe_tactic,eagle3",
        [
            (4, 1, 4, False, True, True, "trtllm_gen", False),
            (4, 1, 4, False, True, True, "trtllm_gen", True),
        ],
        ids=[
            "latency_moe_trtllm",
            "latency_moe_trtllm_eagle",
        ],
    )
    def test_nvfp4_4gpus(self, tp_size, pp_size, ep_size, attention_dp,
                         cuda_graph, overlap_scheduler, moe_tactic, eagle3):

        skip_unless_moe_tactic_runs_here(moe_tactic)

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(tactic=moe_tactic))

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4,
                                        enable_block_reuse=not eagle3)
        spec_config = None
        if eagle3:
            spec_config = Eagle3DecodingConfig(
                max_draft_len=2,
                speculative_model=
                f"{llm_models_root()}/Mistral-Large-3-675B/Mistral-Large-3-675B-Instruct-2512-Eagle/",
                eagle3_model_arch="mistral_large3")
        with LLM(
                f"{llm_models_root()}/Mistral-Large-3-675B/Mistral-Large-3-675B-Instruct-2512-NVFP4/",
                checkpoint_format="mistral_large_3",
                disable_mm_encoder=True,
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                **pytorch_config,
                enable_attention_dp=attention_dp,
                kv_cache_config=kv_cache_config,
                autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                speculative_config=spec_config) as llm:

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skip_less_device_memory(183000)
    @pytest.mark.parametrize(
        "tp_size,pp_size,ep_size,attention_dp,cuda_graph,overlap_scheduler,moe_tactic,eagle3",
        [
            (8, 1, 8, False, True, True, "deepgemm.grouped_gemm", False),
        ],
        ids=[
            "latency_moe_deepgemm",
        ],
    )
    def test_fp8(self, tp_size, pp_size, ep_size, attention_dp, cuda_graph,
                 overlap_scheduler, moe_tactic, eagle3):

        skip_unless_moe_tactic_runs_here(moe_tactic)

        pytorch_config = dict(
            disable_overlap_scheduler=not overlap_scheduler,
            cuda_graph_config=CudaGraphConfig() if cuda_graph else None,
            moe_config=MoeConfig(tactic=moe_tactic))

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4,
                                        enable_block_reuse=not eagle3)
        spec_config = None
        if eagle3:
            spec_config = Eagle3DecodingConfig(
                max_draft_len=2,
                speculative_model=
                f"{llm_models_root()}/Mistral-Large-3-675B/Mistral-Large-3-675B-Instruct-2512-Eagle/",
                eagle3_model_arch="mistral_large3")
        with LLM(
                f"{llm_models_root()}/Mistral-Large-3-675B/Mistral-Large-3-675B-Instruct-2512/",
                checkpoint_format="mistral_large_3",
                disable_mm_encoder=True,
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                moe_expert_parallel_size=ep_size,
                **pytorch_config,
                enable_attention_dp=attention_dp,
                kv_cache_config=kv_cache_config,
                speculative_config=spec_config) as llm:

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


# Reading this 352 GB checkpoint cold over a slow model-store link takes most of
# an hour before evaluation starts. A test-list TIMEOUT still takes precedence.
@pytest.mark.timeout(7200)
class TestNemotronV3Ultra(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-Ultra-V3"
    MODEL_PATH = f"{llm_models_root()}/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
    # No thinking mode for now.
    EXTRA_EVALUATOR_KWARGS = dict(chat_template_kwargs=dict(
        enable_thinking=False))

    def _run_nvfp4_4gpus_eplb(self, moe_tactic: str,
                              eplb_config: MoeLoadBalancerConfig,
                              model_path: str) -> None:
        # The cutlass case covers the other GPUs.
        skip_unless_moe_tactic_runs_here(moe_tactic)
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,
            mamba_ssm_cache_dtype="float16",
            free_gpu_memory_fraction=0.5,
        )
        max_batch_size = 32
        cuda_graph_config = CudaGraphConfig(max_batch_size=max_batch_size,
                                            enable_padding=True)
        moe_config = MoeConfig(tactic=moe_tactic, load_balancer=eplb_config)
        pytorch_config = dict(cuda_graph_config=cuda_graph_config,
                              moe_config=moe_config)
        with LLM(
                model_path,
                kv_cache_config=kv_cache_config,
                max_batch_size=max_batch_size,
                tensor_parallel_size=4,
                moe_expert_parallel_size=4,
                enable_attention_dp=True,
                autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                **pytorch_config,
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_pre_hopper
    @skip_post_hopper
    @pytest.mark.skip_less_device_memory(80000)
    @pytest.mark.skip_less_mpi_world_size(8)
    def test_nvfp4_marlin_8gpus(self):
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        mamba_ssm_cache_dtype="float16",
                                        free_gpu_memory_fraction=0.5)

        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=8,
                 moe_expert_parallel_size=8,
                 moe_config=MoeConfig(tactic="marlin"),
                 kv_cache_config=kv_cache_config,
                 max_batch_size=8,
                 nvfp4_gemm_config={"allowed_backends": ["marlin"]}) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @pytest.mark.skip_less_device_memory(80000)
    @parametrize_with_ids("moe_tactic",
                          ["trtllm_gen", "cutlass", "cutedsl.grouped_gemm"])
    def test_nvfp4_4gpus_static_eplb(self, moe_tactic):
        from transformers import AutoConfig
        model_path = f"{llm_models_root()}/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
        model_cfg = AutoConfig.from_pretrained(model_path,
                                               trust_remote_code=True)
        num_experts = model_cfg.n_routed_experts
        # num_slots should be larger than or equal to num_experts and should be divisible by parallel_size.
        # Assign extra 16 expert slots per rank.
        extra_num_slots = 16 * 4
        num_slots = num_experts + extra_num_slots
        hybrid_pattern = model_cfg.hybrid_override_pattern
        moe_layer_indices = [
            i for i, ch in enumerate(hybrid_pattern) if ch == 'E'
        ]
        initial_global_assignments = {}
        for i in moe_layer_indices:
            initial_global_assignments[i] = [(i + j) % num_experts
                                             for j in range(num_slots)]
        eplb_config = MoeLoadBalancerConfig(
            num_slots=num_slots,
            initial_global_assignments=initial_global_assignments,
            layer_updates_per_iter=0)
        self._run_nvfp4_4gpus_eplb(moe_tactic, eplb_config, model_path)

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_device_not_contain(["GB200"])
    @parametrize_with_ids("moe_tactic",
                          ["trtllm_gen", "cutlass", "cutedsl.grouped_gemm"])
    def test_nvfp4_4gpus_online_eplb(self, moe_tactic):
        model_path = f"{llm_models_root()}/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
        num_experts = 512  # 512 experts per token for Nemotron V3 Ultra.
        # num_slots should be larger than or equal to num_experts and should be divisible by parallel_size.
        # Assign extra 16 expert slots per rank.
        extra_num_slots = 16 * 4
        num_slots = num_experts + extra_num_slots
        eplb_config = MoeLoadBalancerConfig(num_slots=num_slots,
                                            layer_updates_per_iter=2)
        self._run_nvfp4_4gpus_eplb(moe_tactic, eplb_config, model_path)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.parametrize("moe_tactic",
                             ["trtllm_gen", "cutlass", "cutedsl.grouped_gemm"],
                             ids=["trtllm", "cutlass", "cutedsl"])
    @pytest.mark.parametrize(
        "attention_dp",
        [
            False,
            True,
        ],
        ids=[
            "attention_dp_off",
            "attention_dp_on",
        ],
    )
    def test_nvfp4_8gpus(self, attention_dp, moe_tactic):
        # No test list runs the cutlass case, so this test does not run off
        # the SM100 family.
        skip_unless_moe_tactic_runs_here(moe_tactic)
        max_batch_size = 4 if attention_dp else 32
        with LLM(
                f"{llm_models_root()}/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
                kv_cache_config=KvCacheConfig(
                    enable_block_reuse=False,
                    mamba_ssm_cache_dtype="float16",
                    free_gpu_memory_fraction=0.5,
                ),
                max_batch_size=max_batch_size,
                tensor_parallel_size=8,
                moe_expert_parallel_size=8,
                pipeline_parallel_size=1,
                enable_attention_dp=attention_dp,
                cuda_graph_config=CudaGraphConfig(max_batch_size=max_batch_size,
                                                  enable_padding=True),
                disable_overlap_scheduler=False,
                moe_config=MoeConfig(tactic=moe_tactic),
                autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @pytest.mark.parametrize(
        "tp_size, ep_size, periodic_snapshot_interval, attention_dp, use_mtp",
        [
            (4, 1, 256, False, True),
            (4, 4, 256, False, False),
            (4, 4, 256, True, False),
            (4, 4, 512, True, True),
        ],
        ids=["TP4_MTP", "TEP4", "ADP4", "ADP4_MTP"],
    )
    def test_nvfp4_4gpus_block_reuse(self, tp_size, ep_size,
                                     periodic_snapshot_interval, attention_dp,
                                     use_mtp):
        moe_tactic = sm100f_moe_tactic_or_cutlass("trtllm_gen")
        # The target's tactic does not reach the MTP layers, so they pin it too.
        mtp_config = MTPDecodingConfig(num_nextn_predict_layers=3,
                                       moe_tactic=moe_tactic)
        max_batch_size = 4 if attention_dp else 32
        with LLM(
                f"{llm_models_root()}/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
                kv_cache_config=KvCacheConfig(
                    enable_block_reuse=True,
                    avg_seq_len=2048,
                    mamba_ssm_cache_dtype="float16",
                    mamba_state_config=MambaStateConfig(
                        periodic_snapshot_interval=periodic_snapshot_interval),
                    free_gpu_memory_fraction=0.5,
                ),
                max_batch_size=max_batch_size,
                tensor_parallel_size=tp_size,
                moe_expert_parallel_size=ep_size,
                pipeline_parallel_size=1,
                enable_attention_dp=attention_dp,
                autotuner_nvmmh_config=AutoTunerNvMMHConfig(),
                cuda_graph_config=CudaGraphConfig(max_batch_size=max_batch_size,
                                                  enable_padding=True),
                disable_overlap_scheduler=False,
                moe_config=MoeConfig(tactic=moe_tactic),
                speculative_config=mtp_config if use_mtp else None,
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(4)
    @pytest.mark.parametrize(
        "tp_size, ep_size, pp_size, attention_dp",
        [
            (2, 1, 2, False),
            (2, 2, 2, False),
            (2, 2, 2, True),
        ],
        ids=["TP2_PP2", "TEP2_PP2", "ADP2_PP2"],
    )
    def test_nvfp4_parallelism(self, tp_size, ep_size, pp_size, attention_dp):
        max_batch_size = 8 if attention_dp else 32
        with LLM(
                f"{llm_models_root()}/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
                kv_cache_config=KvCacheConfig(
                    enable_block_reuse=False,
                    mamba_ssm_cache_dtype="float16",
                    free_gpu_memory_fraction=0.6,
                ),
                max_batch_size=max_batch_size,
                tensor_parallel_size=tp_size,
                moe_expert_parallel_size=ep_size,
                pipeline_parallel_size=pp_size,
                enable_attention_dp=attention_dp,
                cuda_graph_config=CudaGraphConfig(max_batch_size=max_batch_size,
                                                  enable_padding=True),
                disable_overlap_scheduler=False,
                moe_config=MoeConfig(tactic=sm100f_moe_tactic_or_cutlass(
                    "cutedsl.grouped_gemm")),
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(140000)
class TestStep3p7SpeculativeDecoding(LlmapiAccuracyTestHarness):
    MODEL_NAME = "stepfun-ai/Step-3.7-Flash"
    MODEL_PATH = f"{llm_models_root()}/Step-3.7-Flash-FP8"

    def _create_llm(self, **kwargs):
        return LLM(
            self.MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=4,
            moe_expert_parallel_size=4,
            moe_config=MoeConfig(tactic="trtllm_gen"),
            disable_overlap_scheduler=True,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=False,
                free_gpu_memory_fraction=0.5,
            ),
            max_batch_size=32,
            max_num_tokens=4096,
            max_seq_len=8192,
            **kwargs,
        )

    @pytest.mark.skip_less_mpi_world_size(4)
    def test_guided_decoding(self):
        with self._create_llm(attn_backend="FLASHINFER",
                              guided_decoding_backend="xgrammar") as llm:
            assert_guided_decoding_regex(llm)

    @pytest.mark.skip_less_mpi_world_size(4)
    def test_step3p7_mtp(self):
        with self._create_llm(
                max_stats_len=-1,
                enable_iter_perf_stats=True,
                speculative_config=MTPDecodingConfig(max_draft_len=3),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            assert_acceptance_length_for_llm(
                "TestStep3p7SpeculativeDecoding::test_step3p7_mtp",
                llm,
            )


class TestNemotron35Lightning(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-3.5-Lightning"
    MODEL_PATH = f"{llm_models_root()}/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    # Test with no thinking to save time.
    EXTRA_EVALUATOR_KWARGS = dict(chat_template_kwargs=dict(
        enable_thinking=False))

    def _run_mtp3_chunked_prefill(
            self,
            moe_tactic: str,
            nvfp4_gemm_config: Optional[dict[str, list[str]]] = None) -> None:
        """Evaluate the MTP=3 + chunked-prefill combination on one MoE tactic.

        The checkpoint is MIXED_PRECISION: routed experts, shared experts and
        lm_head are W4A16_NVFP4, the Mamba projections are FP8 and the MTP
        layers are left unquantized. Chunked prefill, CUDA graphs and the
        overlap scheduler are enabled together so the combination with MTP
        drafting is covered end to end. Only the MoE tactic and the NVFP4
        dense GEMM backend differ between callers; everything else is held
        fixed so the two runs are comparable against the same accuracy
        references.
        """
        max_batch_size = 32
        mtp_config = MTPDecodingConfig(max_draft_len=3)
        extra_args = {}
        if nvfp4_gemm_config is not None:
            extra_args["nvfp4_gemm_config"] = nvfp4_gemm_config
        with LLM(
                self.MODEL_PATH,
                kv_cache_config=KvCacheConfig(
                    enable_block_reuse=False,
                    mamba_ssm_cache_dtype="float32",
                    free_gpu_memory_fraction=0.7,
                ),
                max_batch_size=max_batch_size,
                # Chunk size below the evaluation prompt lengths so the
                # multi-chunk Mamba/attention prefill path is actually taken.
                enable_chunked_prefill=True,
                max_num_tokens=1024,
                cuda_graph_config=CudaGraphConfig(max_batch_size=max_batch_size,
                                                  enable_padding=True),
                disable_overlap_scheduler=False,
                moe_config=MoeConfig(tactic=moe_tactic),
                speculative_config=mtp_config,
                **extra_args,
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_no_hopper
    def test_nvfp4_marlin_mtp3_chunked_prefill(self):
        self._run_mtp3_chunked_prefill(
            moe_tactic="marlin",
            nvfp4_gemm_config={"allowed_backends": ["marlin"]})

    @skip_no_sm120
    def test_nvfp4_cutedsl_mtp3_chunked_prefill(self):
        # The W4A16_NVFP4 experts have a CuteDSL implementation on SM120/SM121
        # only (the FlashInfer b12x kernels).
        self._run_mtp3_chunked_prefill(moe_tactic="b12x")
