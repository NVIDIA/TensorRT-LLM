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

import re

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.configs import KimiK3Config, KimiLinearConfig
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    KvCacheConfig,
    MambaStateConfig,
    MoeConfig,
    SADecodingConfig,
    SamplingParams,
    SchedulingParams,
)
from tensorrt_llm.sampling_params import GuidedDecodingParams

from ..conftest import llm_models_root, skip_pre_blackwell
from .accuracy_core import (
    GSM8K,
    ForceTokenLogitsProcessor,
    LlmapiAccuracyTestHarness,
    assert_acceptance_length,
    compute_acceptance_length,
)


@pytest.mark.timeout(10800)
class TestKimiK3(LlmapiAccuracyTestHarness):
    MODEL_NAME = "moonshotai/Kimi-K3"
    MODEL_PATH = f"{llm_models_root()}/Kimi-K3"

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(16)
    # The 16-GPU K3 recipes are qualified on GB300 (one NVL72 domain) only:
    # on 2-node 180-190 GiB parts (B200/GB200, InfiniBand between nodes) the
    # EP16 MoE-comm bring-up hangs and the KV-budget assumptions do not hold,
    # so gate on GB300-class device memory. B300 clears this memory gate but
    # pairs 8-GPU nodes over InfiniBand (same non-NVL72 topology) -- do not
    # schedule these tests on B300; that exclusion is enforced by QA's
    # platform selection, not by this marker.
    @pytest.mark.skip_less_device_memory(200000)
    @pytest.mark.parametrize("mode", ["baseline", "reuse", "sa"])
    def test_w4a16_mxfp4(self, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify Kimi K3 accuracy and its key model-feature matrix entries.

        The three modes retain the existing full-GSM8K coverage. The baseline
        mode also exercises attention DP, the overlap scheduler, CUDA graphs,
        chunked prefill, Torch sampling, a logits processor, and guided
        decoding. The reuse mode requires an observed hybrid-cache hit. The SA
        mode remains an acceptance-length guard; SA is supported but is not a
        column in the model-feature matrix.
        """
        if mode == "baseline":
            monkeypatch.setenv("TLLM_METRICS_ALL_RANKS", "1")
            monkeypatch.setenv("TRTLLM_XGUIDANCE_LENIENT", "1")
        self._assert_checkpoint_routing()

        kv_cache_kwargs = dict(
            enable_block_reuse=False,
            free_gpu_memory_fraction=0.25,
            # tokens_per_block=64 keeps the MLA generation path on the
            # flashinfer trtllm-gen kernel (K3 has 96 query heads).
            tokens_per_block=64,
        )
        llm_kwargs = dict(
            tensor_parallel_size=16,
            moe_expert_parallel_size=16,
            enable_attention_dp=True,
            max_batch_size=32,
            max_num_tokens=8192,
            max_seq_len=8192,
            trust_remote_code=True,
            enable_chunked_prefill=True,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=32),
            moe_config=MoeConfig(
                max_num_tokens=33024,
                use_low_precision_moe_combine=True,
            ),
            enable_iter_perf_stats=True,
            max_stats_len=256,
            return_perf_metrics=True,
        )
        if mode == "baseline":
            llm_kwargs["guided_decoding_backend"] = "xgrammar"
        elif mode == "reuse":
            kv_cache_kwargs["enable_block_reuse"] = True
            # Hybrid models expose reusable prefixes only at KDA state
            # snapshot boundaries; without a snapshot cadence, block reuse
            # silently never engages.
            kv_cache_kwargs["mamba_state_config"] = MambaStateConfig(periodic_snapshot_interval=256)
        else:
            llm_kwargs.update(
                max_batch_size=8,
                disable_overlap_scheduler=True,
                enable_chunked_prefill=False,
                cuda_graph_config=CudaGraphConfig(max_batch_size=8),
                speculative_config=SADecodingConfig(max_draft_len=2),
                max_stats_len=-1,
            )
            # Log corpus-aggregate acceptance length and rate at evaluation
            # end. QA records these values from the test log.
            monkeypatch.setenv("TLLM_EVAL_SPEC_STATS", "1")

        with LLM(
            self.MODEL_PATH,
            kv_cache_config=KvCacheConfig(**kv_cache_kwargs),
            **llm_kwargs,
        ) as llm:
            self._assert_resolved_args(llm, mode)
            if mode == "baseline":
                self._assert_attention_dp(llm)
                self._assert_logits_processor(llm)
                self._assert_guided_decoding(llm)
            elif mode == "reuse":
                self._assert_kv_cache_reuse(llm)

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            if mode == "sa":
                acceptance_length = compute_acceptance_length(llm)
                print(
                    "[AL] TestKimiK3::test_w4a16_mxfp4[sa] "
                    f"acceptance_length = {acceptance_length:.3f}"
                )
                assert_acceptance_length(
                    "TestKimiK3::test_w4a16_mxfp4",
                    acceptance_length,
                )

    def _assert_checkpoint_routing(self) -> None:
        config = load_pretrained_config(self.MODEL_PATH, trust_remote_code=True)
        assert isinstance(config, KimiK3Config)
        assert config.architectures == ["KimiK3ForConditionalGeneration"]
        assert isinstance(config.text_config, KimiLinearConfig)
        assert config.text_config.model_type == "kimi_linear"
        assert config.text_config.num_nextn_predict_layers == 0
        assert getattr(config.text_config, "sliding_window", None) is None

    @staticmethod
    def _assert_resolved_args(llm: LLM, mode: str) -> None:
        assert llm.args.enable_attention_dp is True
        assert llm.args.kv_cache_config.enable_block_reuse is (mode == "reuse")
        # K3's routed-expert quantization is nested in the composite checkpoint
        # and is not represented by the modelopt-style args quant_algo field.
        assert llm.args.quant_config.quant_algo is None
        if mode == "sa":
            assert llm.args.disable_overlap_scheduler is True
            assert llm.args.enable_chunked_prefill is False
        else:
            assert llm.args.disable_overlap_scheduler is False
            assert llm.args.enable_chunked_prefill is True
            assert llm.args.cuda_graph_config is not None
            assert llm.args.cuda_graph_config.enable_padding is True

    @staticmethod
    def _assert_attention_dp(llm: LLM) -> None:
        num_ranks = 16
        output_length = 8
        prompts = [[1, 42, 43] for _ in range(num_ranks)]
        scheduling_params = [
            SchedulingParams(attention_dp_rank=rank, attention_dp_relax=False)
            for rank in range(num_ranks)
        ]
        outputs = llm.generate(
            prompts,
            sampling_params=SamplingParams(
                max_tokens=output_length,
                temperature=0,
                end_id=-1,
            ),
            scheduling_params=scheduling_params,
            use_tqdm=False,
        )
        stats_entries = llm.get_stats(timeout=10)

        assert isinstance(outputs, list)
        assert len(outputs) == num_ranks
        reference_token_ids = outputs[0].outputs[0].token_ids
        for rank, output in enumerate(outputs):
            token_ids = output.outputs[0].token_ids
            assert len(token_ids) == output_length
            assert token_ids == reference_token_ids, (
                f"ADP rank {rank} produced tokens that differ from rank 0"
            )

        assert stats_entries
        for rank in range(num_ranks):
            rank_stats = [
                entry
                for entry in stats_entries
                if entry.get("attentionDpRank", entry.get("rank")) == rank
            ]
            assert rank_stats, f"No iteration statistics reported for ADP rank {rank}"
            inflight_batching_stats = [
                entry.get("inflightBatchingStats", {}) for entry in rank_stats
            ]
            assert sum(stats.get("numContextRequests", 0) for stats in inflight_batching_stats) == 1
            assert (
                max(
                    (stats.get("numGenRequests", 0) for stats in inflight_batching_stats),
                    default=0,
                )
                == 1
            )

    @staticmethod
    def _assert_kv_cache_reuse(llm: LLM) -> None:
        prompt_token_ids = [1] + [42] * 510 + [43]
        output_length = 8
        sampling_params = SamplingParams(
            max_tokens=output_length,
            temperature=0,
            end_id=-1,
            return_perf_metrics=True,
        )
        # Sequential requests on an otherwise idle default router are assigned
        # to the same ADP rank, so explicit rank pinning is unnecessary here.

        cold_output = llm.generate(
            [prompt_token_ids],
            sampling_params=sampling_params,
            use_tqdm=False,
        )[0].outputs[0]
        warm_output = llm.generate(
            [prompt_token_ids],
            sampling_params=sampling_params,
            use_tqdm=False,
        )[0].outputs[0]

        cold_metrics = cold_output.request_perf_metrics
        warm_metrics = warm_output.request_perf_metrics
        assert cold_metrics is not None
        assert warm_metrics is not None
        assert cold_metrics.kv_cache_metrics.num_reused_blocks == 0
        assert warm_metrics.kv_cache_metrics.num_reused_blocks > 0
        assert len(cold_output.token_ids) == output_length
        assert warm_output.token_ids == cold_output.token_ids

    @staticmethod
    def _assert_logits_processor(llm: LLM) -> None:
        forced_token_id = 22
        output_length = 4
        outputs = llm.generate(
            [[1, 42, 43]],
            sampling_params=SamplingParams(
                max_tokens=output_length,
                temperature=0,
                end_id=-1,
                logits_processor=ForceTokenLogitsProcessor(forced_token_id),
            ),
            use_tqdm=False,
        )

        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert outputs[0].outputs[0].token_ids == [forced_token_id] * output_length

    @staticmethod
    def _assert_guided_decoding(llm: LLM) -> None:
        prompt_token_ids = llm.tokenizer.encode("Return exactly two decimal digits:")
        outputs = llm.generate(
            [prompt_token_ids],
            sampling_params=SamplingParams(
                max_tokens=8,
                guided_decoding=GuidedDecodingParams(regex=r"[0-9]{2}"),
            ),
            use_tqdm=False,
        )

        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert re.fullmatch(r"[0-9]{2}", outputs[0].outputs[0].text)
