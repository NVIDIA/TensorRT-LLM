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

import json

import jsonschema
import pytest

from tensorrt_llm import LLM
from tensorrt_llm.evaluate import JsonModeEval as JsonModeEvaluator
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
    RequestOutput,
    SamplingParams,
    SchedulingParams,
)
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import llm_models_root, parametrize_with_ids, skip_pre_blackwell, skip_ray
from .accuracy_core import GSM8K, ForceTokenLogitsProcessor, JsonModeEval, LlmapiAccuracyTestHarness


class _JsonModeGrammarEval(JsonModeEvaluator):
    def compute_score(
        self,
        outputs: list[RequestOutput],
        references: list[str],
        schemas: list[str],
    ) -> float:
        del references
        assert outputs
        assert len(outputs) == len(schemas)
        for output, schema in zip(outputs, schemas, strict=True):
            parsed_output = json.loads(output.outputs[0].text)
            jsonschema.validate(parsed_output, json.loads(schema))
        return 100.0


@skip_pre_blackwell
class TestGLM5FP8(LlmapiAccuracyTestHarness):
    MODEL_NAME = "zai-org/GLM-5-FP8"
    MODEL_PATH = f"{llm_models_root()}/GLM-5-FP8"

    @parametrize_with_ids("tp_size,ep_size", [(8, 8)])
    @pytest.mark.skip_less_mpi_world_size(8)
    def test_8gpus(self, tp_size, ep_size):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)

        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=128, enable_padding=True),
            moe_config=MoeConfig(backend="DEEPGEMM"),
            speculative_config=MTPDecodingConfig(),
            enable_chunked_prefill=True,
        )

        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            moe_expert_parallel_size=ep_size,
            kv_cache_config=kv_cache_config,
            max_seq_len=8192,
            **pytorch_config,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


class TestGLM52NVFP4(LlmapiAccuracyTestHarness):
    MODEL_NAME = "zai-org/GLM-5.2"
    MODEL_PATH = f"{llm_models_root()}/GLM-5.2-NVFP4"

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    @parametrize_with_ids("tp_size,ep_size", [(8, 8)])
    def test_tep(self, tp_size, ep_size):
        # GLM-5.2 reuses the DeepSeek-V3.2 path (MLA + DSA) with cross-layer
        # indexer sharing. NVFP4 weights run on the CuteDSL MoE backend with
        # MTP speculative decoding. The checkpoint keeps the leading dense
        # layers and per-MoE-layer shared_experts / self_attn in higher
        # precision.
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)

        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=128, enable_padding=True),
            moe_config=MoeConfig(backend="CUTEDSL"),
            speculative_config=MTPDecodingConfig(max_draft_len=1),
            enable_chunked_prefill=True,
        )

        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            moe_expert_parallel_size=ep_size,
            kv_cache_config=kv_cache_config,
            max_seq_len=8192,
            **pytorch_config,
        ) as llm:
            assert llm.args.kv_cache_config.enable_block_reuse is True
            assert llm.args.disable_overlap_scheduler is False
            assert llm.args.cuda_graph_config is not None
            assert llm.args.cuda_graph_config.enable_padding is True
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @skip_ray
    @pytest.mark.skip_less_mpi_world_size(8)
    @parametrize_with_ids("tp_size,ep_size", [(8, 8)])
    def test_dep(self, tp_size, ep_size):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)

        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=128, enable_padding=True),
            moe_config=MoeConfig(backend="CUTEDSL"),
            speculative_config=MTPDecodingConfig(max_draft_len=1),
            enable_chunked_prefill=True,
        )

        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            moe_expert_parallel_size=ep_size,
            enable_attention_dp=True,
            kv_cache_config=kv_cache_config,
            max_seq_len=8192,
            **pytorch_config,
        ) as llm:
            assert llm.args.enable_attention_dp is True
            assert llm.args.kv_cache_config.enable_block_reuse is True
            assert llm.args.disable_overlap_scheduler is False
            assert llm.args.cuda_graph_config is not None
            assert llm.args.cuda_graph_config.enable_padding is True
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    @parametrize_with_ids("tp_size,ep_size", [(8, 8)])
    def test_tep_nvfp4kv(self, tp_size, ep_size):
        """Exercise the GLM-5.2 NVFP4 KV cache decode path."""
        model_name = "zai-org/GLM-5.2"
        model_path = f"{llm_models_root()}/GLM-5.2-NVFP4"
        kv_cache_config = KvCacheConfig(
            dtype="nvfp4",
            free_gpu_memory_fraction=0.7,
            use_kv_cache_manager_v2=True,
        )

        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=128, enable_padding=True),
            moe_config=MoeConfig(backend="CUTEDSL"),
            enable_chunked_prefill=False,
        )

        with LLM(
            model_path,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            moe_expert_parallel_size=ep_size,
            kv_cache_config=kv_cache_config,
            max_seq_len=8192,
            **pytorch_config,
        ) as llm:
            assert llm.args.kv_cache_config.use_kv_cache_manager_v2 is True
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4
            task = GSM8K(model_name)
            task.evaluate(llm)

    @pytest.mark.timeout(900)
    @skip_pre_blackwell
    @skip_ray
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.threadleak(enabled=False)
    def test_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        num_ranks = 8
        max_num_tokens = 256
        monkeypatch.setenv("TLLM_METRICS_ALL_RANKS", "1")
        monkeypatch.setenv("TRTLLM_XGUIDANCE_LENIENT", "1")

        # Guided decoding backends are selected when the executor is created.
        # Use XGrammar so all runtime checks can share one model load.
        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=num_ranks,
            pipeline_parallel_size=1,
            moe_expert_parallel_size=num_ranks,
            enable_attention_dp=True,
            disable_overlap_scheduler=True,
            cuda_graph_config=None,
            enable_chunked_prefill=True,
            enable_autotuner=False,
            max_batch_size=num_ranks,
            max_num_tokens=max_num_tokens,
            max_seq_len=2048,
            max_stats_len=128,
            enable_iter_perf_stats=True,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                # ADP pads ranks with dummy requests; keep the shared short
                # prompt from committing a partial block to those requests.
                enable_partial_reuse=False,
                free_gpu_memory_fraction=0.6,
                dtype="fp8",
            ),
            sparse_attention_config=DeepSeekSparseAttentionConfig(
                skip_indexer_for_short_seqs=False,
            ),
            moe_config=MoeConfig(
                backend="CUTEDSL",
                disable_finalize_fusion=True,
            ),
            guided_decoding_backend="xgrammar",
            return_perf_metrics=True,
        ) as llm:
            assert llm.args.enable_attention_dp is True
            assert llm.args.enable_chunked_prefill is True
            assert llm.args.max_num_tokens == max_num_tokens
            assert llm.args.kv_cache_config.enable_block_reuse is True
            assert llm.args.sparse_attention_config.skip_indexer_for_short_seqs is False
            self._assert_attention_dp(llm, num_ranks)
            self._assert_kv_cache_reuse(llm)
            self._assert_chunked_prefill(llm)
            self._assert_logits_processor(llm)
            self._assert_guided_decoding(llm)

    @staticmethod
    def _assert_attention_dp(llm: LLM, num_ranks: int) -> None:
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
            num_context_requests = sum(
                stats.get("numContextRequests", 0) for stats in inflight_batching_stats
            )
            assert num_context_requests == 1, (
                f"Expected exactly one real context request on ADP rank {rank}, "
                f"got {num_context_requests}"
            )
            num_context_tokens = sum(
                stats.get("numCtxTokens", 0) for stats in inflight_batching_stats
            )
            assert num_context_tokens == len(prompts[rank]), (
                f"Expected {len(prompts[rank])} real context tokens on ADP rank "
                f"{rank}, got {num_context_tokens}"
            )
            max_generation_requests = max(
                (stats.get("numGenRequests", 0) for stats in inflight_batching_stats),
                default=0,
            )
            assert max_generation_requests == 1, (
                f"Expected exactly one real generation request per iteration on ADP "
                f"rank {rank}, got {max_generation_requests}"
            )
            assert any(stats.get("numGenKvTokens", 0) > 0 for stats in inflight_batching_stats), (
                f"No real generation KV-cache work executed on ADP rank {rank}"
            )

    @staticmethod
    def _assert_kv_cache_reuse(llm: LLM) -> None:
        prompt_token_ids = [1] + [42] * 255
        output_length = 8
        sampling_params = SamplingParams(
            max_tokens=output_length,
            temperature=0,
            end_id=-1,
            return_perf_metrics=True,
        )
        scheduling_params = [SchedulingParams(attention_dp_rank=0, attention_dp_relax=False)]

        cold_output = llm.generate(
            [prompt_token_ids],
            sampling_params=sampling_params,
            scheduling_params=scheduling_params,
            use_tqdm=False,
        )[0].outputs[0]
        warm_output = llm.generate(
            [prompt_token_ids],
            sampling_params=sampling_params,
            scheduling_params=scheduling_params,
            use_tqdm=False,
        )[0].outputs[0]

        cold_metrics = cold_output.request_perf_metrics
        warm_metrics = warm_output.request_perf_metrics
        assert cold_metrics is not None
        assert warm_metrics is not None
        assert cold_metrics.kv_cache_metrics.num_reused_blocks == 0
        assert warm_metrics.kv_cache_metrics.num_reused_blocks > 0
        assert len(cold_output.token_ids) == output_length
        assert len(warm_output.token_ids) == output_length
        assert warm_output.token_ids == cold_output.token_ids

    @staticmethod
    def _assert_chunked_prefill(llm: LLM) -> None:
        prompt_length = 768
        output_length = 16
        # Keep this prefix distinct from the cache-reuse request so all three
        # expected context chunks execute even though block reuse is enabled.
        prompt_token_ids = [1] + [44] * (prompt_length - 2) + [45]
        outputs = llm.generate(
            [prompt_token_ids],
            sampling_params=SamplingParams(
                max_tokens=output_length,
                temperature=0,
                end_id=-1,
            ),
            use_tqdm=False,
        )

        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) == output_length
        time_breakdown = outputs[0].time_breakdown_metrics
        assert time_breakdown is not None
        context_chunks = time_breakdown.get("ctx_chunk_metrics")
        assert isinstance(context_chunks, list)
        assert len(context_chunks) >= 3

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
        evaluator = _JsonModeGrammarEval(
            dataset_path=JsonModeEval.DATASET_DIR,
            num_samples=4,
            random_seed=0,
            apply_chat_template=True,
        )
        score = evaluator.evaluate(
            llm,
            SamplingParams(
                max_tokens=JsonModeEval.MAX_OUTPUT_LEN,
                truncate_prompt_tokens=JsonModeEval.MAX_INPUT_LEN,
            ),
        )

        assert score == 100.0

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    @parametrize_with_ids("tp_size,ep_size", [(8, 8)])
    def test_mtp_index_share(self, tp_size, ep_size):
        # Like test_tep but max_draft_len=3, exercising DSA indexer Top-K reuse
        # across MTP draft steps (index_share_for_mtp_iteration=true from the checkpoint).
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)

        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=128, enable_padding=True),
            moe_config=MoeConfig(backend="CUTEDSL"),
            speculative_config=MTPDecodingConfig(max_draft_len=3),
            enable_chunked_prefill=True,
        )

        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            moe_expert_parallel_size=ep_size,
            kv_cache_config=kv_cache_config,
            max_seq_len=8192,
            **pytorch_config,
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.NVFP4
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            self._assert_mtp_acceptance_rate(llm)

    @staticmethod
    def _assert_mtp_acceptance_rate(llm: LLM) -> None:
        raw_prompts = [
            "The capital of France is",
            "The president of the United States is",
            "The future of AI is",
        ]
        prompts = [
            llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in raw_prompts
        ]
        token_ids = [llm.tokenizer.encode(prompt) for prompt in prompts]
        sampling_params = SamplingParams(
            max_tokens=128,
            temperature=0,
            end_id=-1,
            return_perf_metrics=True,
        )
        outputs = llm.generate(
            token_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        total_drafted = 0
        total_accepted = 0
        assert isinstance(outputs, list)
        assert len(outputs) == len(token_ids)
        for i, output in enumerate(outputs):
            request_perf_metrics = output.outputs[0].request_perf_metrics
            assert request_perf_metrics is not None
            speculative_metrics = request_perf_metrics.speculative_decoding
            assert speculative_metrics is not None
            num_drafted = int(speculative_metrics.total_draft_tokens)
            num_accepted = int(speculative_metrics.total_accepted_draft_tokens)
            assert num_drafted > 0
            accept_rate = num_accepted / num_drafted
            total_drafted += num_drafted
            total_accepted += num_accepted
            print(
                f"GLM-5.2 MTP index-share prompt {i} acceptance rate: "
                f"{accept_rate:.2%} ({num_accepted}/{num_drafted} tokens)"
            )

        aggregate_accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
        print(
            "GLM-5.2 MTP index-share aggregate acceptance rate: "
            f"{aggregate_accept_rate:.2%} ({total_accepted}/"
            f"{total_drafted} tokens across {len(token_ids)} prompts)"
        )
        assert aggregate_accept_rate > 0.2, (
            f"Aggregate acceptance rate {aggregate_accept_rate:.2%} below threshold 20%"
        )
