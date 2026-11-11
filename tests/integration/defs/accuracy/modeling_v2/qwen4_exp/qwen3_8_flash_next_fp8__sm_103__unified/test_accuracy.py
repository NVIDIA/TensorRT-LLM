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
"""Whole-model gates for the qwen3_8_flash_next_fp8__sm_103__unified target on sm_103.

GSM8K at tp1, tep4, dep4 and dep2 disaggregated; long-context pass keys at tep4.
"""

import pytest
from transformers import AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    KvCacheConfig,
    MambaStateConfig,
    MoeConfig,
    MTPDecodingConfig,
    QSASparseAttentionConfig,
    SamplingParams,
)
from tensorrt_llm.quantization import QuantAlgo

from .....conftest import llm_models_root
from ....accuracy_core import (
    GSM8K,
    LlmapiAccuracyTestHarness,
    assert_acceptance_length,
    compute_acceptance_length,
)
from ....test_disaggregated_serving import launch_disaggregated_llm, run_accuracy_test
from ..._env import scoped_env_overrides
from ..qwen3_8_flash_next_nvfp4__sm_103__default.test_accuracy import (
    _PASSKEY_LENGTHS,
    _PASSKEY_MAX_TOKENS,
    _acceptance_length_from_requests,
    _passkey_prompts,
)

skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="modeling_v2 targets in this batch are certified on sm_103 only"
)

# Check the step contract on the first forward; the flag has to reach the worker ranks.
_VALIDATE = {MODELING_V2_VALIDATE_ENV: "1"}

# Workers spawn their second rank over MPI in a root container (docker/Dockerfile.multi).
_DISAGG_WORKER_ENV = {
    **_VALIDATE,
    "PRTE_ALLOW_RUN_AS_ROOT": "1",
    "PRTE_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
}


def _disagg_server_config() -> dict:
    """One context and one generation server on this host."""
    return {
        "hostname": "localhost",
        "port": 8000,
        "backend": "pytorch",
        "context_servers": {"num_instances": 1, "urls": ["localhost:8001"]},
        "generation_servers": {"num_instances": 1, "urls": ["localhost:8002"]},
    }


def _passkey_acceptance(llm, prompts, label: str, draft_len: int) -> float:
    """Pass-key retrieval on ``llm`` and the requests' mean acceptance length.

    Every key must be stated back, and two identical requests far apart in one batch (different
    slots) must give the same tokens, which fails if a request's selection leaks into another
    slot. Needs no iteration statistics: the requests carry their own speculative metrics.
    """
    texts = [prompt for prompt, _ in prompts] + [prompts[len(prompts) // 2][0]]
    outputs = llm.generate(
        texts,
        SamplingParams(max_tokens=_PASSKEY_MAX_TOKENS, temperature=0, return_perf_metrics=True),
        use_tqdm=False,
    )
    missed = [
        key
        for (_, key), output in zip(prompts, outputs)
        if key not in output.outputs[0].text[:_PASSKEY_MAX_TOKENS]
    ]
    assert not missed, (
        f"{label}: pass keys not stated back: {missed} of {[key for _, key in prompts]}"
    )
    twins = outputs[len(prompts) // 2], outputs[-1]
    assert twins[0].outputs[0].token_ids == twins[1].outputs[0].token_ids, (
        f"{label}: the same prompt in two slots gave different tokens"
    )
    acceptance_length = _acceptance_length_from_requests(outputs, draft_len)
    print(f"[AL] {label} acceptance_length = {acceptance_length:.3f}")
    return acceptance_length


_TEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False)
_DEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True)

# Only tep4 runs the breakable prefill graph: routing admits it under attention TP alone.
_MTP3_ENGINES = [
    pytest.param(_TEP4, "fp8", True, {}, id="tep4_fp8_kv_breakable"),
    pytest.param(_DEP4, "auto", False, {}, id="dep4"),
    pytest.param(
        _DEP4, "auto", False, {"TRTLLM_FORCE_COMM_METHOD": "NVLINK_ONE_SIDED"}, id="dep4_alltoall"
    ),
]


class TestModelingV2Qwen38FlashNextFp8Sm103Unified(LlmapiAccuracyTestHarness):
    """Every test requires modeling_v2, so a missed route fails rather than scoring the built-in."""

    # References are keyed by this name plus the resolved quantization (FP8_BLOCK_SCALES).
    MODEL_NAME = "Qwen/Qwen3.8-Flash-Next"
    MODEL_PATH = f"{llm_models_root()}/Qwen3.8-Flash-Next-FP8"

    # The anchors' protocol: the chat template thinks, so the system prompt caps the reasoning
    # and asks for the bare answer, which then fits GSM8K_MAX_OUTPUT_LEN.
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        system_prompt=(
            "Use at most three short reasoning sentences, then end with `#### NUMBER`. "
            "Do not restate the problem."
        ),
        chat_template_kwargs=dict(enable_thinking=True, reasoning_effort="xhigh"),
    )

    # The anchor's `max_tokens`; the class default (256) truncates this checkpoint's answers.
    GSM8K_MAX_OUTPUT_LEN = 512

    # The checkpoint's bf16 K/V and a bf16 recurrent state, the one the target serves.
    KV_CACHE = KvCacheConfig(
        free_gpu_memory_fraction=0.6,
        enable_block_reuse=False,
        mamba_ssm_cache_dtype="bfloat16",
    )

    # The trtllm-gen FP8 block-scale runner is the one MoE backend the target implements.
    MOE = MoeConfig(backend="TRTLLM")

    ENGINE = dict(
        max_seq_len=9216,
        max_batch_size=128,
        max_num_tokens=8192,
        enable_chunked_prefill=False,
        kv_cache_config=KV_CACHE,
        cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=128),
        moe_config=MOE,
    )

    # One-model MTP-Eagle with the checkpoint's single MTP module replayed three times.
    MTP3 = MTPDecodingConfig(max_draft_len=3)

    # The acceptance-length floors in references/acceptance_length.yaml: gsm8k over bf16 KV
    # (`mtp3`) and over an FP8 pool (`mtp3_fp8_kv`), and the long-context pass keys.
    ACCEPTANCE_KEY_MTP3 = "ModelingV2Qwen38FlashNextFp8Sm103Unified::mtp3"
    ACCEPTANCE_KEY_LONG_CONTEXT = "ModelingV2Qwen38FlashNextFp8Sm103Unified::mtp3_long_context"

    # The breakable prefill CUDA graph, up to max_num_tokens.
    BREAKABLE_PREFILL = dict(
        prefill_cuda_graph_backend="breakable",
        prefill_capture_num_tokens=[512, 1024, 2048, 4096, 8192],
    )

    def _gsm8k(self, mocker, llm, label: str, key: str | None = None) -> None:
        """gsm8k against the engine's FP8_BLOCK_SCALES row of references/gsm8k.yaml.

        Under speculation also the acceptance length, held to the floor under `key` in
        references/acceptance_length.yaml: rejection sampling keeps the score of a miscomputed
        draft layer, so only the acceptance length sees it. It comes from the engine's
        iteration statistics, so the engine needs ``enable_iter_perf_stats`` and
        ``max_stats_len=-1`` (the default queue keeps only the last 1000 iterations).
        """
        assert llm.args.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES, (
            f"{label}: the engine resolved quant_algo {llm.args.quant_config.quant_algo}; the "
            "references are this checkpoint's FP8_BLOCK_SCALES rows"
        )
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)
        GSM8K(self.MODEL_NAME).evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
        if llm.args.speculative_config is None:
            return
        acceptance_length = compute_acceptance_length(llm)
        print(f"[AL] {label} acceptance_length = {acceptance_length:.3f}")
        assert_acceptance_length(key, acceptance_length)

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(145000)
    @pytest.mark.parametrize("parallel, kv_dtype, breakable, env", _MTP3_ENGINES)
    def test_mtp3_gsm8k(self, mocker, request, parallel, kv_dtype, breakable, env):
        """Three drafts with block reuse and chunked prefill, against the FP8_BLOCK_SCALES MTP row.

        Block reuse needs the Gated-DeltaNet and PLE snapshots (every 256 tokens); without them the
        runtime turns reuse off. An FP8 pool reads the FP8_BLOCK_SCALES row with an FP8 KV cache
        and its own acceptance-length floor (`mtp3_fp8_kv`).
        """
        if env.get("TRTLLM_FORCE_COMM_METHOD") == "NVLINK_ONE_SIDED":
            from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import (
                NVLinkOneSided,
            )

            # The target declines the all-to-all where it cannot run it.
            if not NVLinkOneSided.is_platform_supported():
                pytest.skip("no MNNVL on this node: the all-to-all MoE round trip cannot run")
        kv_cache_config = KvCacheConfig(
            dtype=kv_dtype,
            free_gpu_memory_fraction=0.6,
            enable_block_reuse=True,
            mamba_ssm_cache_dtype="bfloat16",
            mamba_state_config=MambaStateConfig(periodic_snapshot_interval=256),
        )
        engine = dict(self.ENGINE, enable_chunked_prefill=True, kv_cache_config=kv_cache_config)
        key = (
            self.ACCEPTANCE_KEY_MTP3
            if kv_dtype == "auto"
            else f"{self.ACCEPTANCE_KEY_MTP3}_{kv_dtype}_kv"
        )
        if breakable:
            engine.update(self.BREAKABLE_PREFILL)
        with (
            scoped_env_overrides({**_VALIDATE, **env}) as rank_env,
            LLM(
                self.MODEL_PATH,
                speculative_config=self.MTP3,
                enable_iter_perf_stats=True,
                max_stats_len=-1,
                **engine,
                **parallel,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            if kv_dtype != "auto":
                assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo(kv_dtype.upper()), (
                    f"kv_cache_config.dtype={kv_dtype} resolved kv_cache_quant_algo "
                    f"{llm.args.quant_config.kv_cache_quant_algo}"
                )
            self._gsm8k(mocker, llm, request.node.callspec.id, key)

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(145000)
    def test_mtp3_long_context(self):
        """tep4 with three drafts and IndexShare: pass keys over 3k-6k tokens.

        The prompts are past the QSA indexer's 2048-token budget, so the target's decode steps and
        its draft layer's take the sparse attention path, which gsm8k never leaves, and IndexShare
        makes the draft loop reuse the selection of its first step. A miscomputed draft path keeps
        the text correct and only lowers the acceptance length, which is held to its floor in
        references/acceptance_length.yaml.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        prompts = _passkey_prompts(tokenizer, _PASSKEY_LENGTHS)
        with (
            scoped_env_overrides(_VALIDATE) as rank_env,
            LLM(
                self.MODEL_PATH,
                speculative_config=self.MTP3,
                sparse_attention_config=QSASparseAttentionConfig(
                    index_share_for_mtp_iteration=True
                ),
                **self.ENGINE,
                **_TEP4,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            acceptance_length = _passkey_acceptance(
                llm, prompts, "tep4 mtp3 long context", self.MTP3.max_draft_len
            )
        assert_acceptance_length(self.ACCEPTANCE_KEY_LONG_CONTEXT, acceptance_length)

    @skip_not_sm103
    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(145000)
    @pytest.mark.skip_less_host_memory(204800)
    def test_disaggregated_dep2(self, mocker):
        """dep2 on both engines, one context and one generation server of two GPUs each.

        Both engines run one layout, so every pool matches on both sides of the NIXL transceiver.
        """
        common_config = {
            "modeling_v2": "require",
            "tensor_parallel_size": 2,
            "moe_expert_parallel_size": 2,
            "enable_attention_dp": True,
            "max_seq_len": 9216,
            "max_batch_size": 128,
            "max_num_tokens": 8192,
            "enable_chunked_prefill": False,
            "moe_config": {"backend": "TRTLLM"},
            "kv_cache_config": {
                "enable_block_reuse": False,
                "mamba_ssm_cache_dtype": "bfloat16",
                "free_gpu_memory_fraction": 0.6,
            },
            # The V2 hybrid manager's pools travel over the Python transceiver.
            "cache_transceiver_config": {"backend": "NIXL", "transceiver_runtime": "PYTHON"},
        }
        ctx_server_config = {
            **common_config,
            "disable_overlap_scheduler": True,
            "cuda_graph_config": None,
        }
        gen_server_config = {
            **common_config,
            "disable_overlap_scheduler": False,
            "cuda_graph_config": {"enable_padding": True, "max_batch_size": 128},
        }

        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)
        with launch_disaggregated_llm(
            _disagg_server_config(),
            ctx_server_config,
            gen_server_config,
            self.MODEL_PATH,
            max_workers=128,
            extra_env=_DISAGG_WORKER_ENV,
            # What every rank of both engines prints once it has bound the manager's pools.
            assert_worker_log_contains="modeling_v2 qwen4_exp fp8 unified target: rank",
        ) as llm:
            # The reference is keyed by this quantization; the launcher reads none off the name.
            llm.args.quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
            run_accuracy_test(
                llm,
                self.MODEL_NAME,
                ["GSM8K"],
                extra_evaluator_kwargs={GSM8K: self.EXTRA_EVALUATOR_KWARGS},
                timeout=3000,
            )

    @skip_not_sm103
    def test_tp1_gsm8k(self, mocker):
        """One rank without speculation, against the FP8_BLOCK_SCALES row."""
        with (
            scoped_env_overrides(_VALIDATE) as rank_env,
            LLM(
                self.MODEL_PATH,
                **self.ENGINE,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            self._gsm8k(mocker, llm, "tp1")
