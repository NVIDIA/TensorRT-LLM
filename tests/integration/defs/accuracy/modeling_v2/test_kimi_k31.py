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
"""Accuracy gates of the Kimi K3.1 modeling_v2 targets.

Kimi K3.1 Pro (DSA sparse attention on the MLA layers, MXFP4 routed experts) at two layouts:
DEP16 (attention DP, 96 query heads per rank padded to 128 in the sparse MLA) and TEP8 (attention
TP, 12 heads per rank), with FP8 index keys and the exact radix top-k. The gates run an FP8 latent
KV cache (``kv_cache_config.dtype='fp8'``), except the general target's ``bf16_kv`` case, which
runs ``'auto'``: the checkpoint's BF16 latents, which routing gives to the general target at every
layout. ``modeling_v2="require"`` rules out a fallback. On sm_103 the family routes the FP8 layouts
to its perf targets (``kimi_k31_pro_mxfp4__sm_103__tep`` / ``__dep``), so the general target's FP8
gates ask for the general target with ``TLLM_K31_DEFAULT_TARGET`` and the perf targets' gates run
on sm_103 only.

The sm_107 specialized targets run their one recipe: the ModelOpt NVFP4 export as stored, an FP8
latent KV cache, FP8 index keys with the GVR top-k, an FP32 KDA state and DFlash2 speculative
decoding. DEP16 and TEP8 run here as aggregated servers, which take the generation server's role
(its MoE backend and draft length 7); helix8 serves generation only, so its gate runs disaggregated
behind a DEP16 context server at draft length 5; cep (8 or 16 ranks) serves context only, so its
gates run disaggregated in front of a TEP8 generation server at draft length 7.

The common NVFP4 default serves SM100, SM103, and SM107 with configurable parallelism,
independent main/indexer/KDA cache formats, BF16 or FP8 draft history, and image input.
Its class below defines two cache recipes and checks the supported MoE backends.
Eligible fixed-layout SM107 configurations remain assigned to the specialized targets.

The Rubin gates are in no CI test list. A Rubin host has 4 GPUs and CI has no multi-host
Rubin stage, while DEP16 needs an MPI world of 16 ranks and TEP8 one of 8: they run by hand on a
multi-host allocation. The helix8 gate needs 24 GPUs on one host (see its class).
"""

import json
import time
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.routing import DEFAULT_TARGET_ENV
from tensorrt_llm.inputs import default_multimodal_input_loader
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DeepSeekSparseAttentionConfig,
    DFlashDecodingConfig,
    KvCacheConfig,
    MoeConfig,
    SamplingParams,
)
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.serve.openai_server import OpenAIServer

from ...conftest import get_sm_version, llm_models_root, skip_no_rubin, skip_pre_blackwell
from ..accuracy_core import GSM8K, GPQADiamond, LlmapiAccuracyTestHarness
from ..test_disaggregated_serving import (
    DEFAULT_ACC_EVALUATION_TIMEOUT,
    launch_disaggregated_llm,
    run_accuracy_test,
)

_MODEL_NAME = "moonshotai/Kimi-K3.1-Pro"
_MODEL_PATH = f"{llm_models_root()}/Kimi-K3.1-Pro"

_skip_no_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="the Kimi K3.1 perf targets run on sm_103 (GB300, B300)"
)


def _kv_cache_config(dtype: str = "fp8") -> KvCacheConfig:
    return KvCacheConfig(
        use_kv_cache_manager_v2=True,
        dtype=dtype,
        tokens_per_block=64,
        enable_block_reuse=False,
        free_gpu_memory_fraction=0.5,
    )


def _target_env(monkeypatch: pytest.MonkeyPatch, general: bool) -> dict:
    """``env_overrides`` that pick the general target over a perf target, or nothing.

    ``LLM`` applies ``env_overrides`` to this process and never restores it, so ``monkeypatch``
    sets (or clears) the variable here too, keeping it from leaking into later tests.
    """
    if not general:
        monkeypatch.delenv(DEFAULT_TARGET_ENV, raising=False)
        return {}
    monkeypatch.setenv(DEFAULT_TARGET_ENV, "1")
    return dict(env_overrides={DEFAULT_TARGET_ENV: "1"})


def _dep16_kwargs(max_seq_len: int, env: dict) -> dict:
    return dict(
        tensor_parallel_size=16,
        moe_expert_parallel_size=16,
        enable_attention_dp=True,
        modeling_v2="require",
        max_batch_size=8,
        max_num_tokens=8192,
        max_seq_len=max_seq_len,
        trust_remote_code=True,
        cuda_graph_config=CudaGraphConfig(max_batch_size=8),
        kv_cache_config=_kv_cache_config(),
        moe_config=MoeConfig(max_num_tokens=32768),
        sparse_attention_config=DeepSeekSparseAttentionConfig(indexer_k_dtype="fp8"),
        **env,
    )


def _tep8_kwargs(env: dict, kv_dtype: str = "fp8") -> dict:
    return dict(
        tensor_parallel_size=8,
        moe_expert_parallel_size=8,
        enable_attention_dp=False,
        modeling_v2="require",
        max_batch_size=8,
        max_num_tokens=8192,
        max_seq_len=8192,
        trust_remote_code=True,
        cuda_graph_config=CudaGraphConfig(max_batch_size=8),
        kv_cache_config=_kv_cache_config(kv_dtype),
        moe_config=MoeConfig(max_num_tokens=33024),
        sparse_attention_config=DeepSeekSparseAttentionConfig(indexer_k_dtype="fp8"),
        **env,
    )


def _gsm8k(kwargs: dict) -> None:
    with LLM(_MODEL_PATH, **kwargs) as llm:
        GSM8K(_MODEL_NAME).evaluate(llm)


def _gpqa_diamond_dep16(env: dict) -> None:
    """GPQA Diamond on DEP16, with the model's chat template."""
    max_seq_len = GPQADiamond.MAX_INPUT_LEN + GPQADiamond.MAX_OUTPUT_LEN
    sampling_params = SamplingParams(
        max_tokens=GPQADiamond.MAX_OUTPUT_LEN,
        truncate_prompt_tokens=GPQADiamond.MAX_INPUT_LEN,
    )
    with LLM(_MODEL_PATH, **_dep16_kwargs(max_seq_len, env)) as llm:
        GPQADiamond(_MODEL_NAME).evaluate(
            llm,
            sampling_params=sampling_params,
            extra_evaluator_kwargs=dict(apply_chat_template=True),
        )


@skip_pre_blackwell
class TestModelingV2KimiK31ProMxfp4Sm10xDefault(LlmapiAccuracyTestHarness):
    """``kimi_k31_pro_mxfp4__sm_10x__default`` at DEP16 and TEP8."""

    MODEL_NAME = _MODEL_NAME
    MODEL_PATH = _MODEL_PATH

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_dep16(self, monkeypatch):
        """GSM8K on the DEP16 configuration."""
        _gsm8k(_dep16_kwargs(8192, _target_env(monkeypatch, general=True)))

    @pytest.mark.timeout(10800)
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gpqa_diamond_dep16(self, monkeypatch):
        """GPQA Diamond on the DEP16 configuration, with the model's chat template."""
        _gpqa_diamond_dep16(_target_env(monkeypatch, general=True))

    @pytest.mark.timeout(3600)
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_tep8(self, monkeypatch):
        """GSM8K on the TEP8 configuration."""
        _gsm8k(_tep8_kwargs(_target_env(monkeypatch, general=True)))

    @pytest.mark.timeout(3600)
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_tep8_bf16_kv(self, monkeypatch):
        """GSM8K on the TEP8 configuration with ``kv_cache_config.dtype='auto'``.

        ``auto`` keeps the checkpoint's BF16 latent KV cache, which routing gives to this target at every layout,
        so no target switch is set.
        """
        _gsm8k(_tep8_kwargs(_target_env(monkeypatch, general=False), kv_dtype="auto"))


@_skip_no_sm103
class TestModelingV2KimiK31ProMxfp4Sm103Tep(LlmapiAccuracyTestHarness):
    """``kimi_k31_pro_mxfp4__sm_103__tep`` at TEP8."""

    MODEL_NAME = _MODEL_NAME
    MODEL_PATH = _MODEL_PATH

    @pytest.mark.timeout(3600)
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_tep8(self, monkeypatch):
        """GSM8K on the TEP8 configuration."""
        _gsm8k(_tep8_kwargs(_target_env(monkeypatch, general=False)))


@_skip_no_sm103
class TestModelingV2KimiK31ProMxfp4Sm103Dep(LlmapiAccuracyTestHarness):
    """``kimi_k31_pro_mxfp4__sm_103__dep`` at DEP16."""

    MODEL_NAME = _MODEL_NAME
    MODEL_PATH = _MODEL_PATH

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_dep16(self, monkeypatch):
        """GSM8K on the DEP16 configuration."""
        _gsm8k(_dep16_kwargs(8192, _target_env(monkeypatch, general=False)))

    @pytest.mark.timeout(10800)
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gpqa_diamond_dep16(self, monkeypatch):
        """GPQA Diamond on the DEP16 configuration, with the model's chat template."""
        _gpqa_diamond_dep16(_target_env(monkeypatch, general=False))


# ---------------------------------------------------------------------------------------------
# Specialized targets (sm_107): kimi_k31_pro_nvfp4__sm_107__{dep16,helix8,tep8}
# ---------------------------------------------------------------------------------------------

_K31_PRO = "moonshotai/Kimi-K3.1-Pro"
# The ModelOpt NVFP4 export (nvidia/kimi_k31_pro_nvfp4) and the DFlash2 MLA drafter
# (radixark/kimi-k3.1-dflash2-preview).
_NVFP4_PATH = f"{llm_models_root()}/Kimi-K3.1-Pro-NVFP4"
_DFLASH2_PATH = f"{llm_models_root()}/Kimi-K3.1-DFlash2-Preview"


def _nvfp4_sparse_attention() -> dict:
    """FP8 index keys and the self-sampling GVR top-k of the recipe."""
    return dict(
        algorithm="dsa",
        indexer_k_dtype="fp8",
        enable_heuristic_topk=True,
        use_self_sampling_topk=True,
        gvr_deterministic_topk=True,
    )


def _nvfp4_kv_cache() -> dict:
    """The FP8 latent KV cache, its 64-token pages and the FP32 KDA state."""
    return dict(
        use_kv_cache_manager_v2=True,
        dtype="fp8",
        tokens_per_block=64,
        enable_block_reuse=False,
        mamba_ssm_cache_dtype="float32",
        free_gpu_memory_fraction=0.5,
    )


def _dflash2(max_draft_len: int) -> DFlashDecodingConfig:
    return DFlashDecodingConfig(
        max_draft_len=max_draft_len, speculative_model=_DFLASH2_PATH, context_kv_dtype="fp8"
    )


def _nvfp4_aggregated(
    tensor_parallel_size: int, enable_attention_dp: bool, moe_backend: str, max_batch_size: int
) -> dict:
    """An aggregated server of the recipe: the generation role, draft length 7.

    The overlap scheduler stays off: DFlash2 under it is not validated on these targets.
    """
    return dict(
        tensor_parallel_size=tensor_parallel_size,
        moe_expert_parallel_size=tensor_parallel_size,
        enable_attention_dp=enable_attention_dp,
        modeling_v2="require",
        disable_mm_encoder=True,
        max_batch_size=max_batch_size,
        max_num_tokens=8192,
        max_seq_len=8192,
        trust_remote_code=True,
        disable_overlap_scheduler=True,
        cuda_graph_config=CudaGraphConfig(max_batch_size=max_batch_size, enable_padding=True),
        kv_cache_config=KvCacheConfig(**_nvfp4_kv_cache()),
        moe_config=MoeConfig(backend=moe_backend, use_low_precision_moe_combine=True),
        sparse_attention_config=DeepSeekSparseAttentionConfig(**_nvfp4_sparse_attention()),
        speculative_config=_dflash2(7),
    )


@skip_no_rubin
class TestModelingV2KimiK31ProNvfp4Sm107Dep16(LlmapiAccuracyTestHarness):
    """``kimi_k31_pro_nvfp4__sm_107__dep16``: attention DP and MoE EP over 16 ranks."""

    MODEL_NAME = _K31_PRO
    MODEL_PATH = _NVFP4_PATH

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_dep16(self):
        """GSM8K on DEP16 with DFlash2 (draft length 7), MegaMoE at batch 12."""
        kwargs = _nvfp4_aggregated(16, True, "MEGAMOE_CUTEDSL", max_batch_size=12)
        with LLM(self.MODEL_PATH, **kwargs) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


@skip_no_rubin
class TestModelingV2KimiK31ProNvfp4Sm107Tep8(LlmapiAccuracyTestHarness):
    """``kimi_k31_pro_nvfp4__sm_107__tep8``: TP and MoE EP over 8 ranks."""

    MODEL_NAME = _K31_PRO
    MODEL_PATH = _NVFP4_PATH

    @pytest.mark.timeout(3600)
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_tep8(self):
        """GSM8K on TEP8 with DFlash2 (draft length 7), FC12 at batch 8."""
        kwargs = _nvfp4_aggregated(8, False, "CUTEDSL_FC12", max_batch_size=8)
        with LLM(self.MODEL_PATH, **kwargs) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


@skip_no_rubin
class TestModelingV2KimiK31ProNvfp4Sm107Helix8(LlmapiAccuracyTestHarness):
    """``kimi_k31_pro_nvfp4__sm_107__helix8``: Helix context parallelism over 8 ranks.

    A generation layout: the gate runs it disaggregated behind a DEP16 context server, 24 GPUs on
    one host. No Rubin host has 24 GPUs, so this test skips there; a disaggregated helix8
    deployment over several hosts is the gate in practice.
    """

    MODEL_NAME = _K31_PRO
    MODEL_PATH = _NVFP4_PATH

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device(24)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_helix8(self):
        """GSM8K through a helix8 generation server at batch 16, draft length 5, MoE TP 2 x EP 4.

        The generation server runs eagerly, without the overlap scheduler.
        """
        transceiver = {"backend": "NIXL", "transceiver_runtime": "PYTHON"}
        spec = {
            "decoding_type": "DFlash",
            "max_draft_len": 5,
            "context_kv_dtype": "fp8",
            "speculative_model": _DFLASH2_PATH,
        }
        ctx_server_config = {
            "tensor_parallel_size": 16,
            "moe_expert_parallel_size": 16,
            "enable_attention_dp": True,
            "modeling_v2": "require",
            "disable_mm_encoder": True,
            "max_batch_size": 16,
            "max_num_tokens": 8192,
            "max_seq_len": 8192,
            "trust_remote_code": True,
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "kv_cache_config": _nvfp4_kv_cache(),
            "moe_config": {"backend": "CUTEDSL", "use_low_precision_moe_combine": True},
            "sparse_attention_config": _nvfp4_sparse_attention(),
            "speculative_config": spec,
            "cache_transceiver_config": transceiver,
        }
        gen_server_config = {
            "tensor_parallel_size": 1,
            "context_parallel_size": 8,
            "moe_tensor_parallel_size": 2,
            "moe_expert_parallel_size": 4,
            "cp_config": {
                "cp_type": "HELIX",
                "tokens_per_block": 64,
                "use_nccl_for_alltoall": False,
                "fifo_version": 2,
            },
            "enable_attention_dp": False,
            "modeling_v2": "require",
            "disable_mm_encoder": True,
            "max_batch_size": 16,
            "max_num_tokens": 96,
            "max_seq_len": 8192,
            "trust_remote_code": True,
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "kv_cache_config": _nvfp4_kv_cache(),
            "moe_config": {"backend": "CUTEDSL_FC12", "use_low_precision_moe_combine": True},
            "sparse_attention_config": _nvfp4_sparse_attention(),
            "speculative_config": spec,
            "cache_transceiver_config": transceiver,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {"num_instances": 1, "urls": ["localhost:8001"]},
            "generation_servers": {"num_instances": 1, "urls": ["localhost:8002"]},
        }
        with launch_disaggregated_llm(
            disaggregated_server_config,
            ctx_server_config,
            gen_server_config,
            self.MODEL_PATH,
            request_timeout_s=DEFAULT_ACC_EVALUATION_TIMEOUT,
        ) as llm:
            # The DuckLLM arguments do not carry what the servers run; set the specs the reference
            # lookup matches on (the export, an FP8 KV cache, DFlash2).
            llm.args.quant_config.quant_algo = QuantAlgo.MIXED_PRECISION
            llm.args.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            llm.args.speculative_config = _dflash2(5)
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])


@skip_no_rubin
class TestModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1(LlmapiAccuracyTestHarness):
    """``kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1``: Helix CP over 16 ranks, MoE TP 16 x EP 1.

    The low-latency generation target. A generation layout: the gate runs it disaggregated behind
    a TEP8 context server, 24 GPUs on one host. No Rubin host has 24 GPUs, so this test skips
    there; a disaggregated deployment over several hosts is the gate in practice.
    """

    MODEL_NAME = _K31_PRO
    MODEL_PATH = _NVFP4_PATH

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device(24)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_helix16(self):
        """GSM8K through the helix16 generation server at batch 4, draft length 7.

        The context server is the TEP8 DFlash2 context target (context KV only).
        """
        transceiver = {"backend": "NIXL", "transceiver_runtime": "PYTHON"}
        spec = {
            "decoding_type": "DFlash",
            "max_draft_len": 7,
            "context_kv_dtype": "fp8",
            "speculative_model": _DFLASH2_PATH,
        }
        ctx_server_config = {
            "tensor_parallel_size": 8,
            "moe_expert_parallel_size": 8,
            "enable_attention_dp": False,
            "modeling_v2": "require",
            "disable_mm_encoder": True,
            "max_batch_size": 4,
            "max_num_tokens": 8192,
            "max_seq_len": 8192,
            "trust_remote_code": True,
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
            "kv_cache_config": _nvfp4_kv_cache(),
            "moe_config": {"backend": "CUTEDSL", "use_low_precision_moe_combine": False},
            "sparse_attention_config": _nvfp4_sparse_attention(),
            "speculative_config": {**spec, "context_kv_only": True},
            "cache_transceiver_config": transceiver,
        }
        gen_server_config = {
            "tensor_parallel_size": 1,
            "context_parallel_size": 16,
            "moe_tensor_parallel_size": 16,
            "moe_expert_parallel_size": 1,
            "cp_config": {
                "cp_type": "HELIX",
                "tokens_per_block": 64,
                "use_nccl_for_alltoall": False,
                "fifo_version": 2,
            },
            "enable_attention_dp": False,
            "modeling_v2": "require",
            "max_batch_size": 4,
            "max_num_tokens": 32,
            "max_seq_len": 8192,
            "trust_remote_code": True,
            "cuda_graph_config": {"batch_sizes": [1, 2, 4], "enable_padding": True},
            "kv_cache_config": _nvfp4_kv_cache(),
            "moe_config": {"backend": "CUTEDSL", "use_low_precision_moe_combine": False},
            "sparse_attention_config": _nvfp4_sparse_attention(),
            "speculative_config": spec,
            "cache_transceiver_config": transceiver,
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {"num_instances": 1, "urls": ["localhost:8001"]},
            "generation_servers": {"num_instances": 1, "urls": ["localhost:8002"]},
        }
        with launch_disaggregated_llm(
            disaggregated_server_config,
            ctx_server_config,
            gen_server_config,
            self.MODEL_PATH,
            request_timeout_s=DEFAULT_ACC_EVALUATION_TIMEOUT,
        ) as llm:
            # The DuckLLM arguments do not carry what the servers run; set the specs the reference
            # lookup matches on (the export, an FP8 KV cache, DFlash2).
            llm.args.quant_config.quant_algo = QuantAlgo.MIXED_PRECISION
            llm.args.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            llm.args.speculative_config = _dflash2(7)
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])


@skip_no_rubin
class TestModelingV2KimiK31ProNvfp4Sm107Helix16Wide(LlmapiAccuracyTestHarness):
    """``kimi_k31_pro_nvfp4__sm_107__helix16_wide``: the ULL layout's wide-batch fork.

    Routed only under ``TLLM_K31_HELIX16_WIDE=1``; gated as the ULL target (24 GPUs on one host,
    so it skips on every Rubin host), at batch 12.
    """

    MODEL_NAME = _K31_PRO
    MODEL_PATH = _NVFP4_PATH

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device(24)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_helix16_wide(self):
        """GSM8K through the wide helix16 generation server at batch 12, draft length 7."""
        transceiver = {"backend": "NIXL", "transceiver_runtime": "PYTHON"}
        spec = {
            "decoding_type": "DFlash",
            "max_draft_len": 7,
            "context_kv_dtype": "fp8",
            "speculative_model": _DFLASH2_PATH,
        }
        ctx_server_config = {
            "tensor_parallel_size": 8,
            "moe_expert_parallel_size": 8,
            "enable_attention_dp": False,
            "modeling_v2": "require",
            "max_batch_size": 12,
            "max_num_tokens": 8192,
            "max_seq_len": 8192,
            "trust_remote_code": True,
            "disable_overlap_scheduler": True,
            "disable_mm_encoder": True,
            "cuda_graph_config": None,
            "kv_cache_config": _nvfp4_kv_cache(),
            "moe_config": {"backend": "CUTEDSL", "use_low_precision_moe_combine": False},
            "sparse_attention_config": _nvfp4_sparse_attention(),
            "speculative_config": {**spec, "context_kv_only": True},
            "cache_transceiver_config": transceiver,
        }
        gen_server_config = {
            "tensor_parallel_size": 1,
            "context_parallel_size": 16,
            "moe_tensor_parallel_size": 16,
            "moe_expert_parallel_size": 1,
            "cp_config": {
                "cp_type": "HELIX",
                "tokens_per_block": 64,
                "use_nccl_for_alltoall": False,
                "fifo_version": 2,
            },
            "enable_attention_dp": False,
            "modeling_v2": "require",
            "max_batch_size": 12,
            "max_num_tokens": 96,
            "max_seq_len": 8192,
            "trust_remote_code": True,
            "cuda_graph_config": {"batch_sizes": list(range(1, 13)), "enable_padding": True},
            "kv_cache_config": _nvfp4_kv_cache(),
            "moe_config": {"backend": "CUTEDSL", "use_low_precision_moe_combine": False},
            "sparse_attention_config": _nvfp4_sparse_attention(),
            "speculative_config": spec,
            "cache_transceiver_config": transceiver,
            "env_overrides": {"TLLM_K31_HELIX16_WIDE": "1"},
        }
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {"num_instances": 1, "urls": ["localhost:8001"]},
            "generation_servers": {"num_instances": 1, "urls": ["localhost:8002"]},
        }
        with launch_disaggregated_llm(
            disaggregated_server_config,
            ctx_server_config,
            gen_server_config,
            self.MODEL_PATH,
            request_timeout_s=DEFAULT_ACC_EVALUATION_TIMEOUT,
        ) as llm:
            llm.args.quant_config.quant_algo = QuantAlgo.MIXED_PRECISION
            llm.args.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            llm.args.speculative_config = _dflash2(7)
            run_accuracy_test(llm, self.MODEL_NAME, ["GSM8K"])


def _gsm8k_cep(
    model_name: str, model_path: str, cp: int, max_num_tokens: int, moe_backend: str = "CUTEDSL"
) -> None:
    """GSM8K through a cep context server over ``cp`` ranks and a TEP8 generation server.

    Draft length 7; both servers run eagerly, without the overlap scheduler. The context server is
    TP 1 with Ulysses context parallelism over ``cp`` ranks, the latent stored once per group
    (``dkv_config`` page_split), and MoE EP ``cp`` through ``moe_backend`` (the all-to-all around
    the CuteDSL experts, or MegaMoE).
    """
    transceiver = {"backend": "NIXL", "transceiver_runtime": "PYTHON"}
    spec = {"decoding_type": "DFlash", "max_draft_len": 7, "speculative_model": _DFLASH2_PATH}
    ctx_server_config = {
        "tensor_parallel_size": 1,
        "context_parallel_size": cp,
        "cp_config": {"cp_type": "ULYSSES"},
        "dkv_config": {"attention_mode": "sp", "kv_layout": "page_split"},
        "moe_expert_parallel_size": cp,
        "moe_tensor_parallel_size": 1,
        "enable_attention_dp": False,
        "modeling_v2": "require",
        "max_batch_size": 16,
        "max_num_tokens": max_num_tokens,
        "max_seq_len": 8192,
        "trust_remote_code": True,
        "disable_overlap_scheduler": True,
        "cuda_graph_config": None,
        "kv_cache_config": _nvfp4_kv_cache(),
        "moe_config": {"backend": moe_backend, "use_low_precision_moe_combine": True},
        "sparse_attention_config": _nvfp4_sparse_attention(),
        "speculative_config": spec,
        "cache_transceiver_config": transceiver,
    }
    gen_server_config = {
        "tensor_parallel_size": 8,
        "moe_expert_parallel_size": 8,
        "moe_tensor_parallel_size": 1,
        "enable_attention_dp": False,
        "modeling_v2": "require",
        "max_batch_size": 8,
        "max_num_tokens": 64,
        "max_seq_len": 8192,
        "trust_remote_code": True,
        "disable_overlap_scheduler": True,
        "cuda_graph_config": None,
        "kv_cache_config": _nvfp4_kv_cache(),
        "moe_config": {"backend": "CUTEDSL_FC12", "use_low_precision_moe_combine": True},
        "sparse_attention_config": _nvfp4_sparse_attention(),
        "speculative_config": spec,
        "cache_transceiver_config": transceiver,
    }
    disaggregated_server_config = {
        "hostname": "localhost",
        "port": 8000,
        "backend": "pytorch",
        "context_servers": {"num_instances": 1, "urls": ["localhost:8001"]},
        "generation_servers": {"num_instances": 1, "urls": ["localhost:8002"]},
    }
    with launch_disaggregated_llm(
        disaggregated_server_config,
        ctx_server_config,
        gen_server_config,
        model_path,
        request_timeout_s=DEFAULT_ACC_EVALUATION_TIMEOUT,
    ) as llm:
        # The DuckLLM arguments do not carry what the servers run; set the specs the reference
        # lookup matches on (the export, an FP8 KV cache, DFlash2).
        llm.args.quant_config.quant_algo = QuantAlgo.MIXED_PRECISION
        llm.args.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
        llm.args.speculative_config = _dflash2(7)
        run_accuracy_test(llm, model_name, ["GSM8K"])


@skip_no_rubin
class TestModelingV2KimiK31ProNvfp4Sm107Cep(LlmapiAccuracyTestHarness):
    """``kimi_k31_pro_nvfp4__sm_107__cep``: context parallelism over 8 or 16 ranks.

    A context layout: each gate runs it disaggregated in front of a TEP8 generation server on one
    host, 16 GPUs (cep8) or 24 (cep16). A Rubin host has 4 GPUs, so these tests skip there; a
    disaggregated cep + tep8 deployment over four or six hosts is the gate in practice.
    """

    MODEL_NAME = _K31_PRO
    MODEL_PATH = _NVFP4_PATH

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device(16)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_cep8(self):
        """cep8: 112 experts and the whole attention weights a rank, so a smaller step budget."""
        _gsm8k_cep(self.MODEL_NAME, self.MODEL_PATH, 8, 16384)

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device(24)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_cep16(self):
        _gsm8k_cep(self.MODEL_NAME, self.MODEL_PATH, 16, 16 * 8192)

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device(24)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_cep16_megamoe(self):
        """cep16 with the context MoE through MegaMoE (``MEGAMOE_CUTEDSL``)."""
        _gsm8k_cep(self.MODEL_NAME, self.MODEL_PATH, 16, 16 * 8192, "MEGAMOE_CUTEDSL")


# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(
    get_sm_version() not in (100, 103, 107), reason="NVFP4 target requires SM100, SM103, or SM107"
)
class TestModelingV2KimiK31ProNvfp4Sm10xDefault(LlmapiAccuracyTestHarness):
    """The NVIDIA checkpoint, two independent-cache recipes, aggregated DFlash2."""

    MODEL_NAME = "nvidia/kimi_k31_pro_nvfp4"
    MODEL_PATH = f"{llm_models_root()}/Kimi-K3.1-Pro-NVFP4"
    DRAFT_PATH = f"{llm_models_root()}/Kimi-K3.1-Dflash2-Preview"

    def _kwargs(self, recipe: str, attention_dp: bool, moe_backend: str) -> dict:
        fp4 = recipe == "fp4_bf16"
        return dict(
            tensor_parallel_size=16 if attention_dp else 8,
            moe_expert_parallel_size=16 if attention_dp else 8,
            enable_attention_dp=attention_dp,
            modeling_v2="require",
            env_overrides={"TLLM_KIMI_K3_FP8_KV_LAYERS": ""},
            max_batch_size=8,
            max_num_tokens=4096,
            max_seq_len=8192,
            trust_remote_code=True,
            enable_chunked_prefill=True,
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=8),
            kv_cache_config=KvCacheConfig(
                use_kv_cache_manager_v2=True,
                dtype="nvfp4" if fp4 else "fp8",
                mamba_ssm_cache_dtype="bfloat16" if fp4 else "float32",
                tokens_per_block=64,
                enable_block_reuse=False,
                free_gpu_memory_fraction=0.5,
            ),
            sparse_attention_config=DeepSeekSparseAttentionConfig(
                indexer_k_dtype="fp4" if fp4 else "fp8"
            ),
            moe_config=MoeConfig(backend=moe_backend, max_num_tokens=33024),
            speculative_config=DFlashDecodingConfig(
                speculative_model=self.DRAFT_PATH,
                max_draft_len=4,
                spec_dec_variant=2,
                context_kv_dtype="auto",
                use_rejection_sampling=True,
                attention_backend="VANILLA",
            ),
        )

    def _generation_smokes(self, llm, tmp_path) -> None:
        greedy = SamplingParams(max_tokens=64, temperature=0)
        prompts = [
            llm.tokenizer.apply_chat_template(
                [dict(role="user", content=instruction)],
                tokenize=True,
                add_generation_prompt=True,
                thinking=False,
            )
            for instruction in (
                "What is 7 plus 5? Reply with just the number.",
                "What is 3 times 4? Reply with just the number.",
            )
        ] + [[55] * 4097]

        def token_ids(outputs):
            return [list(output.outputs[0].token_ids) for output in outputs]

        baseline = token_ids(llm.generate(prompts, greedy))
        assert all(0 < len(tokens) <= 64 for tokens in baseline)
        sampled = llm.generate(
            prompts[:2],
            SamplingParams(max_tokens=1024, temperature=0.7, end_id=-1),
        )
        assert all(len(output.outputs[0].token_ids) == 1024 for output in sampled)

        image_path = tmp_path / "red.png"
        Image.new("RGB", (112, 112), color=(220, 30, 30)).save(image_path)
        image_inputs = default_multimodal_input_loader(
            tokenizer=llm.tokenizer,
            model_dir=self.MODEL_PATH,
            model_type="kimi_k31",
            modality="image",
            prompts=["Describe the image briefly."],
            media=[str(image_path)],
            image_data_format="pil",
        )
        image_outputs = llm.generate(image_inputs, greedy)
        assert 0 < len(image_outputs[0].outputs[0].token_ids) <= 64
        assert token_ids(llm.generate(prompts, greedy))[:2] == baseline[:2]

    def _serving_smoke(self, llm) -> None:
        previous_media_executor = BaseMediaIO._executor
        server = OpenAIServer(
            generator=llm,
            model=self.MODEL_NAME,
            tool_parser=None,
            server_role=None,
            metadata_server_cfg=None,
        )
        BaseMediaIO.set_executor(previous_media_executor)
        # The serving lifespan owns shutdown, so exercise it after the direct API calls.
        with ExitStack() as cleanup:
            cleanup.callback(server._input_proc_executor.shutdown, wait=True, cancel_futures=True)
            cleanup.callback(server._media_load_executor.shutdown, wait=True, cancel_futures=True)
            client = cleanup.enter_context(TestClient(server.app))
            response = client.post(
                "/v1/chat/completions",
                json=dict(
                    model=self.MODEL_NAME,
                    messages=[dict(role="user", content="What is 1 + 1?")],
                    max_tokens=32,
                    temperature=0,
                ),
            )
            assert response.status_code == 200, response.text
            assert response.json()["usage"]["completion_tokens"] > 0
            response = client.post(
                "/v1/completions",
                json=dict(
                    model=self.MODEL_NAME,
                    prompt="The capital of France is",
                    max_tokens=16,
                    temperature=0,
                    stream=True,
                ),
            )
            assert response.status_code == 200, response.text
            assert "data: [DONE]" in response.text
            print(
                json.dumps(dict(serving_smoke="passed", chat_status=200, stream_status=200)),
                flush=True,
            )

    def _evaluate(
        self, monkeypatch, tmp_path, recipe: str, attention_dp: bool, moe_backend: str
    ) -> None:
        if recipe == "fp4_bf16":
            monkeypatch.setenv("TLLM_KIMI_K3_FP8_KV_LAYERS", "")
        else:
            monkeypatch.delenv("TLLM_KIMI_K3_FP8_KV_LAYERS", raising=False)
        start = time.monotonic()
        with LLM(self.MODEL_PATH, **self._kwargs(recipe, attention_dp, moe_backend)) as llm:
            loaded = time.monotonic()
            self._generation_smokes(llm, tmp_path)
            smokes_completed = time.monotonic()
            try:
                score = GSM8K(self.MODEL_NAME).evaluate(
                    llm,
                    extra_acc_spec=recipe,
                    sampling_params=SamplingParams(max_tokens=2048, truncate_prompt_tokens=4096),
                    extra_evaluator_kwargs=dict(
                        apply_chat_template=True,
                        chat_template_kwargs=dict(thinking=True),
                        preserve_caller_max_tokens=True,
                        log_samples=True,
                        output_path=str(tmp_path / "gsm8k"),
                        output_dir=str(tmp_path / "generations"),
                    ),
                )
            except AssertionError:
                self._serving_smoke(llm)
                raise
            print(
                json.dumps(
                    dict(
                        recipe=recipe,
                        moe_backend=moe_backend,
                        attention_dp=attention_dp,
                        gsm8k=score,
                        startup_seconds=loaded - start,
                        generation_smoke_seconds=smokes_completed - loaded,
                        evaluation_seconds=time.monotonic() - smokes_completed,
                    )
                ),
                flush=True,
            )
            self._serving_smoke(llm)

    @pytest.mark.timeout(7200)
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skip_less_device_memory(250000)
    @pytest.mark.parametrize(
        "recipe,moe_backend",
        [
            pytest.param(recipe, backend, id=f"{recipe}-{backend}")
            for backend in ("TRTLLM", "CUTEDSL", "MEGAMOE_CUTEDSL")
            for recipe in ("fp8_fp32", "fp4_bf16")
        ],
    )
    def test_gsm8k_tep8(self, monkeypatch, tmp_path, recipe, moe_backend):
        self._evaluate(monkeypatch, tmp_path, recipe, attention_dp=False, moe_backend=moe_backend)

    @pytest.mark.timeout(7200)
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skip_less_device_memory(250000)
    @pytest.mark.parametrize(
        "recipe,moe_backend",
        [
            pytest.param(recipe, backend, id=f"{recipe}-{backend}")
            for backend in ("TRTLLM", "CUTEDSL", "MEGAMOE_CUTEDSL")
            for recipe in ("fp8_fp32", "fp4_bf16")
        ],
    )
    def test_gsm8k_dep16(self, monkeypatch, tmp_path, recipe, moe_backend):
        self._evaluate(monkeypatch, tmp_path, recipe, attention_dp=True, moe_backend=moe_backend)
