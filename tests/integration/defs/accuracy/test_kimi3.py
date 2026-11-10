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

import random

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.configs import KimiK3Config, KimiLinearConfig
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MambaStateConfig,
    MoeConfig,
    SADecodingConfig,
    SamplingParams,
    SchedulingParams,
)
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import llm_models_root, skip_pre_blackwell
from .accuracy_core import (
    GSM8K,
    ForceTokenLogitsProcessor,
    GPQADiamond,
    LlmapiAccuracyTestHarness,
    acceptance_length_from_iteration_stats,
    assert_acceptance_length,
    assert_acceptance_length_for_llm,
    assert_guided_decoding_regex,
    assert_kv_cache_reuse_for_llm,
)

# KDA recurrent state is snapshotted every SNAPSHOT_INTERVAL tokens, and under
# aggregated Helix a reused prefix ends on a ledger block of
# tokens_per_block x cp_size tokens.
SNAPSHOT_INTERVAL = 256
HELIX_TOKENS_PER_BLOCK = 64
HELIX_CP_SIZE = 2
LEDGER_TOKENS = HELIX_TOKENS_PER_BLOCK * HELIX_CP_SIZE


def _assert_helix_reuse_blocks(
    llm: LLM,
    prompt_len: int,
    snapshot_interval: int = SNAPSHOT_INTERVAL,
    ledger_tokens: int = LEDGER_TOKENS,
) -> None:
    """Cold-then-warm run of a fresh prompt; check how deep the warm run reuses.

    A KDA state is reusable only at a snapshot point, and a reused prefix must
    end on a ledger block. The deepest usable point is the last multiple of
    snapshot_interval strictly before the prompt end, so a prompt not longer
    than one interval reuses nothing. snapshot_interval must be a multiple of
    ledger_tokens. Token output is not compared: greedy decoding is not
    run-to-run deterministic on this config even without reuse.
    """
    # Distinct tokens per length so no earlier probe shares a prefix with this one.
    prompt = [prompt_len] + [1000 + prompt_len] * (prompt_len - 2) + [2000 + prompt_len]
    sampling_params = SamplingParams(
        max_tokens=8, temperature=0, end_id=-1, return_perf_metrics=True
    )
    reused = []
    for _ in range(2):  # cold, then warm
        out = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)[0]
        reused.append(out.outputs[0].request_perf_metrics.kv_cache_metrics.num_reused_blocks)
    expected = (prompt_len - 1) // snapshot_interval * (snapshot_interval // ledger_tokens)
    assert reused == [0, expected], (
        f"prompt_len={prompt_len}: reused blocks (cold, warm) {reused}, expected [0, {expected}]"
    )


def _assert_drafter_reads_reused_prefix(
    llm: LLM, snapshot_interval: int, ledger_tokens: int
) -> None:
    """Cold-then-warm run of a recall prompt; the warm run must keep its acceptance length.

    The prompt lists facts and asks for them back, so the answer copies text
    that lies inside the reused prefix. A drafter that cannot read the reused
    prefix's K/V drafts that copy blind and its acceptance length drops, while
    the target's output is unaffected because verification is lossless.
    Repeated filler text would not show this: the recomputed tail alone lets
    a drafter predict a repetition.
    """
    rng = random.Random(17645)
    facts = " ".join(
        f"Fact {i}: the access code for locker {i} is {rng.randrange(100000, 1000000)}."
        for i in range(1, 91)
    )
    prompt = llm.tokenizer.encode(
        f"{facts}\nRepeat every fact above exactly, starting from Fact 1.\nFact 1:"
    )
    sampling_params = SamplingParams(
        max_tokens=128, temperature=0, ignore_eos=True, return_perf_metrics=True
    )
    results = [
        llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)[0]
        for _ in range(2)  # cold, then warm
    ]
    reused = [r.outputs[0].request_perf_metrics.kv_cache_metrics.num_reused_blocks for r in results]
    cold_al, warm_al = (float(r.avg_decoded_tokens_per_iter) for r in results)
    expected = (len(prompt) - 1) // snapshot_interval * (snapshot_interval // ledger_tokens)
    print(
        f"[reuse] recall prompt_len = {len(prompt)}, reused blocks (cold, warm) = {reused}, "
        f"AL cold = {cold_al:.3f}, warm = {warm_al:.3f}"
    )
    assert reused == [0, expected], f"reused blocks (cold, warm) {reused}, expected [0, {expected}]"
    assert cold_al > 1.0, f"cold AL {cold_al:.3f}: the drafter accepted nothing"
    assert warm_al >= cold_al - 0.2, (
        f"warm AL {warm_al:.3f} < cold AL {cold_al:.3f} - 0.2: the drafter does not "
        "read the reused prefix's K/V"
    )


@pytest.mark.timeout(10800)
class TestKimiK3(LlmapiAccuracyTestHarness):
    MODEL_NAME = "moonshotai/Kimi-K3"
    MODEL_PATH = f"{llm_models_root()}/Kimi-K3"
    DSPARK_MODEL_PATH = f"{llm_models_root()}/RadixArk-Kimi-K3-DSpark"

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(16)
    # The 16-GPU K3 recipes are qualified on GB300 (one NVL72 domain) only:
    # on 2-node 180-190 GiB parts (B200/GB200, InfiniBand between nodes) the
    # EP16 MoE-comm bring-up hangs and the KV-budget assumptions do not hold,
    # so gate on GB300-class device memory. B300 clears this memory gate but
    # pairs 8-GPU nodes over InfiniBand (same non-NVL72 topology) -- do not
    # schedule these tests on B300; that exclusion is enforced by QA's
    # platform selection and by the CI stage's gb300-only gpu wildcard,
    # not by this marker.
    @pytest.mark.skip_less_device_memory(200000)
    @pytest.mark.parametrize("mode", ["baseline", "reuse", "sa", "dspark"])
    def test_w4a16_mxfp4(self, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify Kimi K3 accuracy and its key model-feature matrix entries.

        The three modes retain the existing full-GSM8K coverage. The baseline
        mode also exercises attention DP, the overlap scheduler, CUDA graphs,
        chunked prefill, Torch sampling, a logits processor, and guided
        decoding. The reuse mode requires an observed hybrid-cache hit. The SA
        and DSpark modes also guard acceptance length for the two speculative
        decoding techniques advertised in the model-feature matrix.

        The baseline and sa modes run post-merge in the GB300 16-GPU 4-node CI
        stage (test-db list l0_gb300_multi_nodes_node4_gpu16.yml); all four
        modes also run in QA's weekly multinode pipeline
        (qa/llm_function_multinode.txt).
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
                max_stats_len=-1,
            )
            if mode == "sa":
                llm_kwargs.update(
                    cuda_graph_config=CudaGraphConfig(max_batch_size=8),
                    speculative_config=SADecodingConfig(max_draft_len=2),
                )
            else:
                # Kimi K3 DSpark verification has only been qualified in eager
                # mode. The external drafter and its captured target states
                # also need more headroom than suffix automaton.
                kv_cache_kwargs["free_gpu_memory_fraction"] = 0.20
                llm_kwargs.update(
                    cuda_graph_config=None,
                    speculative_config=DSparkDecodingConfig(
                        max_draft_len=7,
                        speculative_model=self.DSPARK_MODEL_PATH,
                    ),
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
                assert_guided_decoding_regex(llm)
            elif mode == "reuse":
                # Sequential requests on the idle default router are assigned
                # to the same ADP rank, so explicit rank pinning is unnecessary.
                assert_kv_cache_reuse_for_llm(llm, [1] + [42] * 510 + [43])

            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            if mode in ("sa", "dspark"):
                assert_acceptance_length_for_llm(
                    f"TestKimiK3::test_w4a16_mxfp4[{mode}]",
                    llm,
                )

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skip_less_device_memory(200000)
    def test_gpqa_diamond_w4a16_mxfp4(self) -> None:
        """Run GPQA Diamond on the K3 checkpoint with DEP16."""
        max_seq_len = GPQADiamond.MAX_INPUT_LEN + GPQADiamond.MAX_OUTPUT_LEN
        sampling_params = SamplingParams(
            max_tokens=GPQADiamond.MAX_OUTPUT_LEN,
            truncate_prompt_tokens=GPQADiamond.MAX_INPUT_LEN,
        )

        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=16,
            moe_expert_parallel_size=16,
            enable_attention_dp=True,
            max_batch_size=32,
            max_num_tokens=8192,
            max_seq_len=max_seq_len,
            trust_remote_code=True,
            enable_chunked_prefill=True,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=32),
            moe_config=MoeConfig(
                max_num_tokens=33024,
                use_low_precision_moe_combine=True,
            ),
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=0.25,
                tokens_per_block=64,
            ),
        ) as llm:
            # K3 stores its compressed-tensors quantization configuration in
            # text_config, so the LLM-args reference matcher sees no quant algo.
            assert llm.args.quant_config.quant_algo is None
            task = GPQADiamond(self.MODEL_NAME)
            task.evaluate(
                llm,
                sampling_params=sampling_params,
                # The provider reference and the 90.404% qualification run use
                # the model's chat template.
                extra_evaluator_kwargs=dict(apply_chat_template=True),
            )

    # The NVFP4 requant: the Blackwell/Rubin deployment precision, served
    # here by the aggregate DEP8 recipe rather than the DEP16 one above.
    NVFP4_MODEL_PATH = f"{llm_models_root()}/Kimi-K3-NVFP4"

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    # Attention DP replicates every non-expert weight on each rank, so DEP8
    # holds far more weight per GPU than TEP8 does; only GB300-class memory
    # leaves room for the KV cache.
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_nvfp4_dep8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full GSM8K on the NVFP4 checkpoint with the aggregate DEP8 recipe.

        TP=8 / EP=8 with attention DP, TRTLLM-Gen MoE (fused NVFP4 SiTu
        cubins), CUDA graphs and chunked prefill.
        """
        self._run_gsm8k_nvfp4_dep8(monkeypatch, kv_cache_dtype="auto")

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_nvfp4_kv_dep8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The DEP8 NVFP4 recipe on an NVFP4 KV cache, which selects FP4 MLA.

        The checkpoint declares no KV-cache quantization, so the recipe above
        never reaches the FP4 MLA path: the hybrid KV cache manager with the
        FP4 MLA layout, the FP8 fallback layers, and FP4 MLA chunked prefill.
        """
        self._run_gsm8k_nvfp4_dep8(monkeypatch, kv_cache_dtype="nvfp4")

    def _run_gsm8k_nvfp4_dep8(
        self, monkeypatch: pytest.MonkeyPatch, *, kv_cache_dtype: str
    ) -> None:
        # Attention DP also replicates the shared-expert and latent MoE MLPs.
        # Kept in BF16 (the default) they cost about 15.7 GiB more per rank
        # and leave DEP8 no KV-cache room, so read them as FP8 block-scale
        # weights; the GSM8K reference was measured with this cast. The
        # variable is read in the worker ranks while the model is built, so it
        # travels as an LLM env override. LLM also sets it in this process and
        # does not restore it, hence the monkeypatch teardown.
        fp8_moe_mlp_env = {"KIMI_K3_FP8_WEIGHT_READ_MOE_MLP": "1"}
        for key, value in fp8_moe_mlp_env.items():
            monkeypatch.setenv(key, value)
        with LLM(
            self.NVFP4_MODEL_PATH,
            env_overrides=fp8_moe_mlp_env,
            tensor_parallel_size=8,
            moe_expert_parallel_size=8,
            enable_attention_dp=True,
            max_batch_size=8,
            max_num_tokens=8192,
            max_seq_len=8192,
            trust_remote_code=True,
            disable_overlap_scheduler=False,
            enable_chunked_prefill=True,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=8),
            moe_config=MoeConfig(
                backend="TRTLLM",
                max_num_tokens=131072,
                use_low_precision_moe_combine=True,
            ),
            kv_cache_config=KvCacheConfig(
                dtype=kv_cache_dtype,
                enable_block_reuse=False,
                # Attention DP replicates KV per rank. A smaller fraction lets
                # clamp_max_seq_len_for_mem drop max_seq_len below a GSM8K
                # prompt, which then overruns max_blocks_per_seq mid-run.
                free_gpu_memory_fraction=0.4,
                # FP4 MLA raises this to its 128-token page size.
                tokens_per_block=64,
            ),
        ) as llm:
            # The requant ships hf_quant_config.json, so the matcher sees
            # MIXED_PRECISION -- only the experts are 4-bit.
            assert llm.args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
            if kv_cache_dtype == "nvfp4":
                assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    # TP4 leaves about 203 GiB of weights per GPU; only GB300-class memory
    # leaves room for the KV cache.
    @pytest.mark.skip_less_device_memory(250000)
    @pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
    def test_gsm8k_nvfp4_tp4_helix2(self, kv_cache_dtype: str) -> None:
        """Full GSM8K on the NVFP4 checkpoint with aggregated Helix (TP4 x CP2).

        Helix splits each sequence's KV cache across the two CP ranks while
        both ranks run the full prefill, so this covers the Helix KV capacity
        profiling, the cross-rank KV layout and the Helix decode path in
        aggregated serving.
        """
        with LLM(
            self.NVFP4_MODEL_PATH,
            tensor_parallel_size=4,
            context_parallel_size=2,
            moe_expert_parallel_size=8,
            cp_config={"cp_type": "HELIX", "tokens_per_block": 64},
            max_batch_size=8,
            max_num_tokens=8192,
            max_seq_len=8192,
            trust_remote_code=True,
            enable_chunked_prefill=True,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=8),
            moe_config=MoeConfig(backend="CUTLASS"),
            kv_cache_config=KvCacheConfig(
                dtype=kv_cache_dtype,
                enable_block_reuse=False,
                tokens_per_block=64,
                use_kv_cache_manager_v2=True,
            ),
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_blackwell
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.skip_less_device_memory(250000)
    @pytest.mark.parametrize(
        "kv_cache_dtype,mode",
        [("auto", "reuse"), ("fp8", "reuse"), ("fp8", "reuse_preempt")],
        ids=["auto-reuse", "fp8-reuse", "fp8-reuse_preempt"],
    )
    def test_gsm8k_nvfp4_tp4_helix2_block_reuse(self, kv_cache_dtype: str, mode: str) -> None:
        """Full GSM8K with block reuse on aggregated Helix (TP4 x CP2).

        Same engine as test_gsm8k_nvfp4_tp4_helix2 with block reuse on, held to
        the same GSM8K reference. KDA state is reusable only at snapshot points,
        and under Helix a reused prefix ends on a ledger block (tokens_per_block
        x cp_size = 128 tokens), so a misaligned snapshot restores the wrong
        recurrent state. The probes on an idle engine check the reuse depth at
        each prompt length. GSM8K checks accuracy with reuse enabled: its
        few-shot exemplars differ per question, so it reuses almost nothing,
        and its reuse counter is printed, not asserted. reuse_preempt caps the
        KV pool so requests are suspended and resumed; counters confirm that
        preemption happened.
        """
        kv_cache_kwargs = {
            # Without a snapshot cadence a hybrid model never reuses.
            "mamba_state_config": MambaStateConfig(periodic_snapshot_interval=SNAPSHOT_INTERVAL),
        }
        llm_kwargs = {}
        if mode == "reuse_preempt":
            # max_tokens counts global ledger tokens under Helix. The pool holds
            # the longest request but not eight concurrent ones.
            kv_cache_kwargs["max_tokens"] = 4096
            llm_kwargs["max_seq_len"] = 3072
        else:
            llm_kwargs["max_seq_len"] = 8192
        with LLM(
            self.NVFP4_MODEL_PATH,
            tensor_parallel_size=4,
            context_parallel_size=2,
            moe_expert_parallel_size=8,
            cp_config={"cp_type": "HELIX", "tokens_per_block": HELIX_TOKENS_PER_BLOCK},
            max_batch_size=8,
            max_num_tokens=8192,
            trust_remote_code=True,
            enable_chunked_prefill=True,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=8),
            moe_config=MoeConfig(backend="CUTLASS"),
            kv_cache_config=KvCacheConfig(
                dtype=kv_cache_dtype,
                enable_block_reuse=True,
                tokens_per_block=64,
                use_kv_cache_manager_v2=True,
                **kv_cache_kwargs,
            ),
            enable_iter_perf_stats=True,
            max_stats_len=-1,
            **llm_kwargs,
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
            assert llm.args.kv_cache_config.enable_block_reuse is True
            # Reuse depth on an idle engine follows the snapshot alignment:
            # nothing up to one interval, then one interval at a time.
            for prompt_len in (129, 257, 513, 1000):
                _assert_helix_reuse_blocks(llm, prompt_len)
            llm.get_stats(timeout=2)  # keep the counters below GSM8K-only
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            # Reuse counters are informational (see the docstring); the
            # preemption counters are asserted for reuse_preempt.
            stats = llm.get_stats(timeout=2)
            windows = [w for s in stats for w in (s.get("kvCacheIterationStats") or {}).values()]
            reused_blocks = sum(w.get("iterReusedBlocks", 0) for w in windows)
            suspended = sum(s.get("iterSuspendedRequests", 0) for s in stats)
            resumed = sum(s.get("iterResumedRequests", 0) for s in stats)
            totals = {
                key: sum(w.get(key, 0) for w in windows)
                for key in (
                    "iterAllocTotalBlocks",
                    "iterMissedBlocks",
                    "iterSnapshotLookups",
                    "iterSnapshotHits",
                    "iterReusedTokens",
                    "iterUnreusedTokens",
                )
            }
            print(
                f"[reuse] reused_blocks = {reused_blocks}, "
                f"suspended = {suspended}, resumed = {resumed}, {totals}"
            )
            if mode == "reuse_preempt":
                assert suspended > 0 and resumed > 0

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
        if mode in ("sa", "dspark"):
            assert llm.args.disable_overlap_scheduler is True
            assert llm.args.enable_chunked_prefill is False
            if mode == "dspark":
                assert llm.args.cuda_graph_config is None
                assert llm.args.speculative_config.decoding_type == "DSpark"
                assert (
                    str(llm.args.speculative_config.speculative_model)
                    == TestKimiK3.DSPARK_MODEL_PATH
                )
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


@pytest.mark.timeout(3600)
# TP8 leaves ~182 GiB of weights per GPU before any KV cache, past what a
# 184 GiB B200/GB200 part holds. Scheduled on 2x4 GB300
# (l0_gb300_multi_nodes_node2_gpu8).
@pytest.mark.skip_less_device_memory(250000)
@skip_pre_blackwell
class TestKimiK3DSpark(LlmapiAccuracyTestHarness):
    """Kimi K3 driven by a standalone DSpark drafter.

    Split from TestKimiK3 because both the drafter and the target precision
    are checkpoint axes here: the RadixArk artifact below is a qwen3-shaped
    DSparkDraftModel while Inferact ships a K3DSparkModel, and the target is
    the NVFP4 requant rather than TestKimiK3's MXFP4 weights.
    """

    MODEL_NAME = "moonshotai/Kimi-K3"
    # The NVFP4 requant, not the MXFP4 weights TestKimiK3 runs: it is the
    # Blackwell deployment precision and nothing else covers it end to end.
    MODEL_PATH = f"{llm_models_root()}/Kimi-K3-NVFP4"
    DSPARK_MODEL_PATH = f"{llm_models_root()}/RadixArk-Kimi-K3-DSpark"
    INFERACT_DSPARK_MODEL_PATH = f"{llm_models_root()}/Inferact-Kimi-K3-DSpark"

    @pytest.mark.skip_less_mpi_world_size(8)
    def test_gsm8k_tep8(self):
        """GSM8K plus an acceptance-length gate on the TEP8 DSpark recipe.

        TEP, not TestKimiK3's DEP16 recipe: with attention DP off the drafter gets
        its own KV cache pool, which is the path this covers. Accuracy alone
        cannot detect a broken drafter -- speculative decoding is lossless, so
        a drafter producing garbage scores the same and only runs slower --
        hence the acceptance-length assertion.
        """
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False, free_gpu_memory_fraction=0.20, tokens_per_block=64
        )
        spec_config = DSparkDecodingConfig(
            max_draft_len=7, speculative_model=self.DSPARK_MODEL_PATH, attention_backend="TRTLLM"
        )
        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=8,
            moe_expert_parallel_size=8,
            enable_attention_dp=False,
            max_batch_size=8,
            max_num_tokens=4096,
            max_seq_len=8192,
            trust_remote_code=True,
            # The drafter runs between target steps, and its hidden-state
            # capture needs a whole-sequence prefill.
            disable_overlap_scheduler=True,
            enable_chunked_prefill=False,
            cuda_graph_config=CudaGraphConfig(max_batch_size=8),
            # Not the TRTLLM backend TestKimiK3 uses: trtllm-gen ships
            # SiTu cubins only for W4A8_MXFP4_MXFP8, so K3's NVFP4 routed
            # experts leave MEGAMOE_CUTEDSL and CUTLASS as the choices.
            moe_config=MoeConfig(
                backend="MEGAMOE_CUTEDSL", max_num_tokens=33024, use_low_precision_moe_combine=True
            ),
            kv_cache_config=kv_cache_config,
            max_stats_len=-1,
            enable_iter_perf_stats=True,
            speculative_config=spec_config,
        ) as llm:
            # Unlike TestKimiK3's MXFP4 weights, the requant ships
            # hf_quant_config.json, so the matcher sees MIXED_PRECISION --
            # only the experts are 4-bit.
            assert llm.args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            assert_acceptance_length_for_llm(
                "TestKimiK3DSpark::test_gsm8k_tep8",
                llm,
            )

    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.parametrize("tp_size,cp_size", [(4, 2), (1, 8)], ids=["tp4cp2", "tp1cp8"])
    def test_gsm8k_nvfp4_helix_fp8kv(self, tp_size: int, cp_size: int):
        """GSM8K plus an acceptance-length gate on aggregated Helix with DSpark.

        Block reuse is off: this isolates DSpark x aggregated Helix (context
        phase, mixed batches, verify groups) from the reuse path. Uses the
        MLA-shaped Inferact drafter, whose block decode AUTO resolves to CUTEDSL.
        Generation attention runs num_heads / tp_size heads per rank (24 at
        TP4, 96 at TP1); CuTe DSL MLA serves the multi-token Helix verify
        group at either. FP8 KV applies to the target only, the drafter keeps
        its pool in bf16.
        """
        kv_cache_config = KvCacheConfig(
            dtype="fp8",
            enable_block_reuse=False,
            free_gpu_memory_fraction=0.20,
            tokens_per_block=64,
            use_kv_cache_manager_v2=True,
        )
        spec_config = DSparkDecodingConfig(
            max_draft_len=7, speculative_model=self.INFERACT_DSPARK_MODEL_PATH
        )
        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=tp_size,
            context_parallel_size=cp_size,
            moe_expert_parallel_size=8,
            cp_config={"cp_type": "HELIX", "tokens_per_block": 64},
            max_batch_size=8,
            max_num_tokens=8192,
            max_seq_len=8192,
            trust_remote_code=True,
            disable_overlap_scheduler=True,
            enable_chunked_prefill=True,
            cuda_graph_config=None,
            moe_config=MoeConfig(backend="CUTLASS"),
            kv_cache_config=kv_cache_config,
            max_stats_len=-1,
            enable_iter_perf_stats=True,
            speculative_config=spec_config,
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            assert_acceptance_length_for_llm(
                f"TestKimiK3DSpark::test_gsm8k_nvfp4_helix_fp8kv[tp{tp_size}cp{cp_size}]", llm
            )

    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.parametrize("tp_size,cp_size", [(4, 2), (1, 8)], ids=["tp4cp2", "tp1cp8"])
    def test_gsm8k_nvfp4_helix_fp8kv_block_reuse(self, tp_size: int, cp_size: int):
        """test_gsm8k_nvfp4_helix_fp8kv with block reuse on.

        The target pool reuses whole ledger blocks (64 x cp_size tokens) ending
        at a KDA snapshot point, while the dense drafter's pool matches in
        64-token pages. The scheduler pairs the two pools on one reuse depth,
        so a target reuse hit means the drafter's K/V for that prefix was
        reused too; a cold-then-warm recall prompt checks the drafter can read
        it. GSM8K then checks accuracy with reuse on. Its few-shot exemplars
        differ per question, so it reuses almost nothing and its counters are
        not asserted.
        """
        ledger_tokens = HELIX_TOKENS_PER_BLOCK * cp_size
        # Snapshot points are floored to ledger blocks, and points below the
        # first block are dropped, so the interval must span a ledger block.
        snapshot_interval = max(SNAPSHOT_INTERVAL, ledger_tokens)
        kv_cache_config = KvCacheConfig(
            dtype="fp8",
            enable_block_reuse=True,
            free_gpu_memory_fraction=0.20,
            tokens_per_block=HELIX_TOKENS_PER_BLOCK,
            use_kv_cache_manager_v2=True,
            mamba_state_config=MambaStateConfig(periodic_snapshot_interval=snapshot_interval),
        )
        spec_config = DSparkDecodingConfig(
            max_draft_len=7, speculative_model=self.INFERACT_DSPARK_MODEL_PATH
        )
        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=tp_size,
            context_parallel_size=cp_size,
            moe_expert_parallel_size=8,
            cp_config={"cp_type": "HELIX", "tokens_per_block": HELIX_TOKENS_PER_BLOCK},
            max_batch_size=8,
            max_num_tokens=8192,
            max_seq_len=8192,
            trust_remote_code=True,
            disable_overlap_scheduler=True,
            enable_chunked_prefill=True,
            cuda_graph_config=None,
            moe_config=MoeConfig(backend="CUTLASS"),
            kv_cache_config=kv_cache_config,
            max_stats_len=-1,
            enable_iter_perf_stats=True,
            speculative_config=spec_config,
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            assert llm.args.kv_cache_config.enable_block_reuse is True
            # Nothing at one interval, then one interval of ledger blocks at a time.
            for prompt_len in (
                snapshot_interval,
                snapshot_interval + 1,
                2 * snapshot_interval + 1,
                4 * snapshot_interval - 24,
            ):
                _assert_helix_reuse_blocks(llm, prompt_len, snapshot_interval, ledger_tokens)
            _assert_drafter_reads_reused_prefix(llm, snapshot_interval, ledger_tokens)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.skip_less_mpi_world_size(8)
    def test_gsm8k_nvfp4_helix_fp8kv_kv_pressure(self):
        """test_gsm8k_nvfp4_helix_fp8kv with a KV pool too small for the batch.

        A burst of long prompts forces suspension to the host tier; GSM8K
        accuracy and acceptance length must then hold, so a drafter resuming
        with bad K/V shows up as an AL drop. TP4 x CP2 only: at CP8 no
        max_tokens both fits one request and overflows eight.
        """
        kv_cache_config = KvCacheConfig(
            dtype="fp8",
            enable_block_reuse=False,
            free_gpu_memory_fraction=0.20,
            # Fits one 3072-token request; eight 1456-token ones fill it to
            # within the per-step draft reservations, so the burst suspends.
            max_tokens=12288,
            avg_seq_len=1024,
            tokens_per_block=HELIX_TOKENS_PER_BLOCK,
            use_kv_cache_manager_v2=True,
        )
        spec_config = DSparkDecodingConfig(
            max_draft_len=7, speculative_model=self.INFERACT_DSPARK_MODEL_PATH
        )
        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=4,
            context_parallel_size=2,
            moe_expert_parallel_size=8,
            cp_config={"cp_type": "HELIX", "tokens_per_block": HELIX_TOKENS_PER_BLOCK},
            max_batch_size=8,
            max_num_tokens=8192,
            max_seq_len=3072,
            trust_remote_code=True,
            disable_overlap_scheduler=True,
            enable_chunked_prefill=True,
            cuda_graph_config=None,
            moe_config=MoeConfig(backend="CUTLASS"),
            kv_cache_config=kv_cache_config,
            max_stats_len=-1,
            enable_iter_perf_stats=True,
            speculative_config=spec_config,
        ) as llm:
            assert llm.args.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
            assert llm.args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
            rng = random.Random(17645)
            burst = [[rng.randrange(1000, 100000) for _ in range(1200)] for _ in range(8)]
            burst_params = SamplingParams(max_tokens=256, temperature=0, ignore_eos=True)
            outputs = llm.generate(burst, sampling_params=burst_params, use_tqdm=False)
            assert all(len(o.outputs[0].token_ids) == 256 for o in outputs)
            burst_stats = llm.get_stats(timeout=2)
            burst_suspended = sum(s.get("iterSuspendedRequests", 0) for s in burst_stats)
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
            # A separate drain, so the AL gate scores GSM8K only.
            stats = llm.get_stats(timeout=2)
            suspended = sum(s.get("iterSuspendedRequests", 0) for s in stats)
            resumed = sum(s.get("iterResumedRequests", 0) for s in stats)
            acceptance_length = acceptance_length_from_iteration_stats(stats)
            print(
                f"[kv_pressure] burst suspended = {burst_suspended}, GSM8K "
                f"suspended = {suspended}, resumed = {resumed}, "
                f"acceptance_length = {acceptance_length:.6f}"
            )
            assert burst_suspended > 0
            assert_acceptance_length(
                "TestKimiK3DSpark::test_gsm8k_nvfp4_helix_fp8kv[tp4cp2]",
                acceptance_length,
            )
