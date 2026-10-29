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
"""Whole-model gates for the mistral_large4 modeling_v2 targets.

Every test here passes ``modeling_v2="require"`` to ``LLM``. Under ``"auto"`` a
configuration that missed a target's criteria -- or fell outside its bounds --
would quietly fall back to the built-in implementation, pass, and report the
built-in's numbers as the target's.

These targets are multi-rank, and the mode has to hold on every rank rather
than only on the one running the test body. The switch is an LLM API argument,
so it reaches the ranks the way the rest of the arguments do.
"""

import contextlib
import json
import os
import threading
import time
from functools import partial
from pathlib import Path

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, DSparkDecodingConfig, KvCacheConfig

from ...conftest import llm_models_root
from ..accuracy_core import (
    GSM8K,
    MMMU,
    GPQADiamond,
    LlmapiAccuracyTestHarness,
    acceptance_length_from_iteration_stats,
    assert_acceptance_length,
)
from . import llguidance_panel, multimodal, ruler, spec_evidence

# The targets assert their own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103,
    reason="mistral_large4 modeling_v2 targets are certified on sm_103 only",
)


class _Nvfp4AccuracyDiagnostics(LlmapiAccuracyTestHarness):
    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k(self):
        """The identity accuracy gate, at the harness default protocol.

        That is the protocol the registered reference was set under, so the anchor and the
        measurement read the same metric. The engine also checks its step contract on its
        first forward.
        """
        task = GSM8K(self.MODEL_NAME)
        with LLM(
            self.MODEL_PATH,
            checkpoint_format="mistral_large_4",
            kv_cache_config=self.KV,
            modeling_v2="require",
            env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
            **self.PARALLEL,
            **self.BOUNDS,
        ) as llm:
            task.evaluate(llm)

    # GSM8K with 24 exemplars: every prompt is 3152-5115 tokens, past index_topk, so the
    # indexer's selection prunes context on every sample (5-shot prompts stay under 2048,
    # where the engine skips the indexer). A key of its own in gsm8k.yaml, scored by the
    # reference implementation on its fp8_ds_mla cache.
    MODEL_NAME_24SHOT = "mistralai/Mistral-Large-4-1T-A52B-gsm8k-24shot"

    class GSM8K24Shot(GSM8K):
        MAX_INPUT_LEN = 8192

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k_24shot(self):
        """24-shot GSM8K, where the indexer's selection prunes every prompt.

        `trtllm-eval gsm8k --num_fewshot 24 --max_input_length 8192`, otherwise the harness
        protocol of the 5-shot gate.
        """
        task = self.GSM8K24Shot(self.MODEL_NAME_24SHOT)
        with LLM(
            self.MODEL_PATH,
            checkpoint_format="mistral_large_4",
            kv_cache_config=self.KV,
            modeling_v2="require",
            **self.PARALLEL,
            **self.BOUNDS,
        ) as llm:
            task.evaluate(llm, extra_evaluator_kwargs=dict(num_fewshot=24))

    # A passkey hidden in filler text, asked for at the end. GSM8K's prompts stay under the
    # indexer's index_topk (2048), where the engine skips the indexer; these prompts are about
    # 30000 tokens -- one context step, within max_num_tokens 32768 -- so the indexer scores and
    # selects for every query row past 2048, in the context step and in each decode step.
    PASSKEY_TOKENS = 30000
    PASSKEY_FILLER = (
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    )

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_passkey_retrieval(self):
        """The long-context witness: a passkey recalled from a ~30000-token prompt.

        At three depths of the prompt, greedy decoding must reproduce the passkey. A wrong
        selection, rope or SSMax position past 2048 reads the wrong positions and loses it.
        """
        params = SamplingParams(max_tokens=12, temperature=0.0)
        with LLM(
            self.MODEL_PATH,
            checkpoint_format="mistral_large_4",
            kv_cache_config=self.KV,
            modeling_v2="require",
            **self.PARALLEL,
            **self.BOUNDS,
        ) as llm:
            unit = len(llm.tokenizer.encode(self.PASSKEY_FILLER * 10)) / 10
            reps = int(self.PASSKEY_TOKENS / unit)
            for depth, key in ((0.2, 71432), (0.5, 58219), (0.8, 30476)):
                before = int(reps * depth)
                prompt = (
                    "There is an important piece of information hidden inside a lot of "
                    "irrelevant text. Find it and memorize it. I will quiz you about it.\n\n"
                    + self.PASSKEY_FILLER * before
                    + f"\nThe pass key is {key}. Remember it. {key} is the pass key.\n"
                    + self.PASSKEY_FILLER * (reps - before)
                    + "\nWhat is the pass key? The pass key is"
                )
                tokens = len(llm.tokenizer.encode(prompt))
                assert 2048 < tokens <= self.BOUNDS["max_seq_len"] - params.max_tokens, tokens
                answer = llm.generate([prompt], params)[0].outputs[0].text
                assert str(key) in answer, (depth, tokens, answer)


class TestModelingV2MistralLarge4Nvfp4Sm103Tep4(_Nvfp4AccuracyDiagnostics):
    """mistral-large-4-nvfp4 / sm_103 / tep4, the language model."""

    MODEL_NAME = "mistralai/Mistral-Large-4-1T-A52B"
    # The Mistral-native ModelOpt MIXED_PRECISION export as published (params.json,
    # tekken.json, consolidated safetensors: NVFP4 routed experts, FP8 block-scale linears).
    MODEL_PATH = f"{llm_models_root()}/Mistral-Large-4-1T-A52B-NVFP4"

    # The parallel topology the path's tep4 segment declares: attention heads and experts both
    # sharded four ways, attention DP off. Routing derives the target *from* these; attention DP
    # would be a different weight layout and so a different target.
    TEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False)

    # The engine bounds the target serves: 16 sequences, unchunked steps of up to 32768 tokens
    # (a whole 32K prompt in one step), CUDA graphs padded over batches 1-16.
    BOUNDS = dict(
        max_batch_size=16,
        max_num_tokens=32768,
        max_seq_len=32768,
        enable_chunked_prefill=False,
        cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
    )

    # The latent KV cache at its default: the checkpoint's fp8_kv_cache makes it the
    # fp8_ds_mla cache the reference implementation serves (656 B per token and owner layer,
    # MistralLarge4CacheManager), read as kv_cache_quant_algo FP8; the checkpoint's quant_algo
    # is MIXED_PRECISION, so the harness reads the MIXED_PRECISION / FP8 row.
    KV = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85)

    PARALLEL = TEP4


class TestModelingV2MistralLarge4Nvfp4Sm103Dep4(_Nvfp4AccuracyDiagnostics):
    """mistral-large-4-nvfp4 / sm_103 / dep4, the language model."""

    MODEL_NAME = "mistralai/Mistral-Large-4-1T-A52B"
    # The same ModelOpt MIXED_PRECISION export as the tep4 class.
    MODEL_PATH = f"{llm_models_root()}/Mistral-Large-4-1T-A52B-NVFP4"

    # The parallel topology the path's dep4 segment declares: attention data parallel over the
    # four ranks (each holds the whole attention and serves its own requests) and the routed
    # experts four ways. Routing derives the target *from* these; attention DP off is the tep4
    # target.
    DEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True)

    # The engine bounds the target serves, per rank under attention DP: 16 sequences, unchunked
    # steps of up to 32768 tokens (a whole 32K prompt in one step), CUDA graphs padded over
    # batches 1-16.
    BOUNDS = dict(
        max_batch_size=16,
        max_num_tokens=32768,
        max_seq_len=32768,
        enable_chunked_prefill=False,
        cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
    )

    # The latent KV cache at its default: the checkpoint's fp8_kv_cache makes it the
    # fp8_ds_mla cache (MistralLarge4CacheManager), read as kv_cache_quant_algo FP8; the
    # checkpoint's quant_algo is MIXED_PRECISION, so the harness reads the MIXED_PRECISION / FP8
    # row.
    KV = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85)

    PARALLEL = DEP4


class TestModelingV2MistralLarge4Nvfp4Sm103Tp4(_Nvfp4AccuracyDiagnostics):
    """mistral-large-4-nvfp4 / sm_103 / tp4, the language model."""

    MODEL_NAME = "mistralai/Mistral-Large-4-1T-A52B"
    # The same ModelOpt MIXED_PRECISION export as the tep4 class.
    MODEL_PATH = f"{llm_models_root()}/Mistral-Large-4-1T-A52B-NVFP4"

    # The parallel topology the path's tp4 segment declares: attention heads sharded four ways
    # and the routed experts tensor parallel (every rank holds all 128 experts, each cut to a
    # quarter of its width), attention DP off. Routing derives the target *from* these; expert
    # parallelism is the tep4 target. A bare tensor_parallel_size 4 resolves to the same split.
    TP4 = dict(
        tensor_parallel_size=4,
        moe_tensor_parallel_size=4,
        moe_expert_parallel_size=1,
        enable_attention_dp=False,
    )

    # The engine bounds the target serves: 16 sequences, unchunked steps of up to 32768 tokens
    # (a whole 32K prompt in one step), CUDA graphs padded over batches 1-16.
    BOUNDS = dict(
        max_batch_size=16,
        max_num_tokens=32768,
        max_seq_len=32768,
        enable_chunked_prefill=False,
        cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
    )

    # The latent KV cache at its default: the checkpoint's fp8_kv_cache makes it the
    # fp8_ds_mla cache (MistralLarge4CacheManager), read as kv_cache_quant_algo FP8; the
    # checkpoint's quant_algo is MIXED_PRECISION, so the harness reads the MIXED_PRECISION / FP8
    # row.
    KV = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85)

    PARALLEL = TP4


# The eight-rank deployments: eight ranks over two nodes of four (one NVL72 segment). Each node
# sees four devices, so the gate is the MPI world the launcher started (one node of eight,
# unlaunched, counts its devices); the inherited tests' four-device gate holds on every node.
_EIGHT_RANKS = pytest.mark.skip_less_mpi_world_size(8)


class TestModelingV2MistralLarge4Nvfp4Sm103Tep8(_Nvfp4AccuracyDiagnostics):
    """mistral-large-4-nvfp4 / sm_103 / tep8, the language model."""

    pytestmark = _EIGHT_RANKS

    MODEL_NAME = "mistralai/Mistral-Large-4-1T-A52B"
    # The same ModelOpt MIXED_PRECISION export as the tep4 class.
    MODEL_PATH = f"{llm_models_root()}/Mistral-Large-4-1T-A52B-NVFP4"

    # The parallel topology the path's tep8 segment declares: eight ranks over two nodes of four,
    # attention heads and experts both sharded eight ways, attention DP off. Routing derives the
    # target *from* these: the tep4 target, which serves eight ranks as well as four.
    # tensor_parallel_size 8 alone would shard every expert eight ways instead (the tp4 target),
    # and attention DP would be a different weight layout (the dep4 target).
    TEP8 = dict(tensor_parallel_size=8, moe_expert_parallel_size=8, enable_attention_dp=False)

    # The engine bounds the target serves: 16 sequences, unchunked steps of up to 32768 tokens
    # (a whole 32K prompt in one step), CUDA graphs padded over batches 1-16.
    BOUNDS = dict(
        max_batch_size=16,
        max_num_tokens=32768,
        max_seq_len=32768,
        enable_chunked_prefill=False,
        cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
    )

    # The latent KV cache at its default: the checkpoint's fp8_kv_cache makes it the
    # fp8_ds_mla cache the reference implementation serves (656 B per token and owner layer,
    # MistralLarge4CacheManager), read as kv_cache_quant_algo FP8; the checkpoint's quant_algo
    # is MIXED_PRECISION, so the harness reads the MIXED_PRECISION / FP8 row.
    KV = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85)

    PARALLEL = TEP8


class TestModelingV2MistralLarge4Nvfp4Sm103Tp8(_Nvfp4AccuracyDiagnostics):
    """mistral-large-4-nvfp4 / sm_103 / tp8, the language model."""

    pytestmark = _EIGHT_RANKS

    MODEL_NAME = "mistralai/Mistral-Large-4-1T-A52B"
    # The same ModelOpt MIXED_PRECISION export as the tep4 class.
    MODEL_PATH = f"{llm_models_root()}/Mistral-Large-4-1T-A52B-NVFP4"

    # The parallel topology the path's tp8 segment declares: eight ranks over two nodes of four,
    # attention heads sharded eight ways, every routed expert on every rank cut to an eighth of
    # its width, attention DP off. Routing derives the target *from* these: the tp4 target, which
    # serves eight ranks as well as four. Experts sharded by expert (moe_expert_parallel_size 8)
    # is the tep4 target, and attention DP would be a different weight layout. A bare
    # tensor_parallel_size 8 resolves to the same split.
    TP8 = dict(
        tensor_parallel_size=8,
        moe_tensor_parallel_size=8,
        moe_expert_parallel_size=1,
        enable_attention_dp=False,
    )

    # The engine bounds the target serves: 16 sequences, unchunked steps of up to 32768 tokens
    # (a whole 32K prompt in one step), CUDA graphs padded over batches 1-16.
    BOUNDS = dict(
        max_batch_size=16,
        max_num_tokens=32768,
        max_seq_len=32768,
        enable_chunked_prefill=False,
        cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
    )

    # The latent KV cache at its default: the checkpoint's fp8_kv_cache makes it the
    # fp8_ds_mla cache the reference implementation serves (656 B per token and owner layer,
    # MistralLarge4CacheManager), read as kv_cache_quant_algo FP8; the checkpoint's quant_algo
    # is MIXED_PRECISION, so the harness reads the MIXED_PRECISION / FP8 row.
    KV = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85)

    PARALLEL = TP8


class TestModelingV2MistralLarge4Nvfp4Sm103Dep8(_Nvfp4AccuracyDiagnostics):
    """mistral-large-4-nvfp4 / sm_103 / dep8, the language model."""

    pytestmark = _EIGHT_RANKS

    MODEL_NAME = "mistralai/Mistral-Large-4-1T-A52B"
    # The same ModelOpt MIXED_PRECISION export as the tep4 class.
    MODEL_PATH = f"{llm_models_root()}/Mistral-Large-4-1T-A52B-NVFP4"

    # The parallel topology the path's dep8 segment declares: attention data parallel over
    # eight ranks on two nodes of four (each holds the whole attention and serves its own
    # requests) and the routed experts eight ways. Routing derives the target *from* these: the
    # dep4 target, which serves eight ranks as well as four. Attention DP off, or
    # tensor_parallel_size 8 alone, is another weight layout.
    DEP8 = dict(tensor_parallel_size=8, moe_expert_parallel_size=8, enable_attention_dp=True)

    # The engine bounds the target serves, per rank under attention DP: 16 sequences, unchunked
    # steps of up to 32768 tokens (a whole 32K prompt in one step), CUDA graphs padded over
    # batches 1-16.
    BOUNDS = dict(
        max_batch_size=16,
        max_num_tokens=32768,
        max_seq_len=32768,
        enable_chunked_prefill=False,
        cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
    )

    # The latent KV cache at its default: the checkpoint's fp8_kv_cache makes it the
    # fp8_ds_mla cache (MistralLarge4CacheManager, a pool per rank under attention DP), read as
    # kv_cache_quant_algo FP8; the checkpoint's quant_algo is MIXED_PRECISION, so the harness reads
    # the MIXED_PRECISION / FP8 row.
    KV = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85)

    PARALLEL = DEP8


# The 1M specializations of the classes above: the same target and topology at the checkpoint's
# own context (max_position_embeddings 1048576; per rank under attention DP), prefilled in
# chunked steps of up to max_num_tokens 32768. A specialization fixes its serving
# configuration, so each is a class of its own: it inherits its 32K class's tests, model,
# topology (and eight-rank gate), KV cache and the rest of its bounds, and the 32K class keeps
# the certified recipe (max_seq_len = max_num_tokens = 32768, chunked prefill off).


class TestModelingV2MistralLarge4Nvfp4Sm103Tep4Ctx1m(TestModelingV2MistralLarge4Nvfp4Sm103Tep4):
    """mistral-large-4-nvfp4 / sm_103 / tep4 at the 1M envelope, the language model."""

    BOUNDS = dict(
        TestModelingV2MistralLarge4Nvfp4Sm103Tep4.BOUNDS,
        max_seq_len=1048576,
        enable_chunked_prefill=True,
    )


class TestModelingV2MistralLarge4Nvfp4Sm103Dep4Ctx1m(TestModelingV2MistralLarge4Nvfp4Sm103Dep4):
    """mistral-large-4-nvfp4 / sm_103 / dep4 at the 1M envelope, the language model."""

    BOUNDS = dict(
        TestModelingV2MistralLarge4Nvfp4Sm103Dep4.BOUNDS,
        max_seq_len=1048576,
        enable_chunked_prefill=True,
    )


class TestModelingV2MistralLarge4Nvfp4Sm103Tp4Ctx1m(TestModelingV2MistralLarge4Nvfp4Sm103Tp4):
    """mistral-large-4-nvfp4 / sm_103 / tp4 at the 1M envelope, the language model."""

    BOUNDS = dict(
        TestModelingV2MistralLarge4Nvfp4Sm103Tp4.BOUNDS,
        max_seq_len=1048576,
        enable_chunked_prefill=True,
    )


class TestModelingV2MistralLarge4Nvfp4Sm103Tep8Ctx1m(TestModelingV2MistralLarge4Nvfp4Sm103Tep8):
    """mistral-large-4-nvfp4 / sm_103 / tep8 at the 1M envelope, the language model."""

    BOUNDS = dict(
        TestModelingV2MistralLarge4Nvfp4Sm103Tep8.BOUNDS,
        max_seq_len=1048576,
        enable_chunked_prefill=True,
    )


class TestModelingV2MistralLarge4Nvfp4Sm103Tp8Ctx1m(TestModelingV2MistralLarge4Nvfp4Sm103Tp8):
    """mistral-large-4-nvfp4 / sm_103 / tp8 at the 1M envelope, the language model."""

    BOUNDS = dict(
        TestModelingV2MistralLarge4Nvfp4Sm103Tp8.BOUNDS,
        max_seq_len=1048576,
        enable_chunked_prefill=True,
    )


class TestModelingV2MistralLarge4Nvfp4Sm103Dep8Ctx1m(TestModelingV2MistralLarge4Nvfp4Sm103Dep8):
    """mistral-large-4-nvfp4 / sm_103 / dep8 at the 1M envelope, the language model."""

    BOUNDS = dict(
        TestModelingV2MistralLarge4Nvfp4Sm103Dep8.BOUNDS,
        max_seq_len=1048576,
        enable_chunked_prefill=True,
    )


@contextlib.contextmanager
def _iteration_coverage(llm: LLM, path: Path, coverage: dict):
    """One stats consumer for raw evidence, rank coverage and this workload's AL."""
    stop = threading.Event()
    errors, spec_stats = [], []
    llm.get_stats(timeout=0.1)

    def drain(stream):
        for item in llm.get_stats(timeout=0.1):
            row = json.loads(item) if isinstance(item, str) else item
            stream.write(json.dumps(row) + "\n")
            if row.get("specDecodingStats"):
                spec_stats.append({"specDecodingStats": row["specDecodingStats"]})
            rank = row.get("attentionDpRank", 0 if not llm.args.enable_attention_dp else None)
            stats = row.get("inflightBatchingStats") or {}
            shape = coverage.setdefault(rank, {"context_tokens": 0, "decode_batch": 0})
            shape["context_tokens"] = max(shape["context_tokens"], stats.get("numCtxTokens", 0))
            shape["decode_batch"] = max(shape["decode_batch"], stats.get("numGenRequests", 0))
        stream.flush()

    def collect():
        try:
            with path.open("w") as stream:
                while not stop.is_set():
                    drain(stream)
                    stop.wait(0.2)
                drain(stream)
        except Exception as error:
            errors.append(str(error))

    worker = threading.Thread(target=collect, name="accuracy-iterations", daemon=True)
    worker.start()
    try:
        yield errors, spec_stats
    finally:
        stop.set()
        worker.join(timeout=15)
        assert not worker.is_alive(), "Iteration reader did not stop"


class TestModelingV2MistralLarge4Nvfp4Accuracy(LlmapiAccuracyTestHarness):
    MODEL_PATH = f"{llm_models_root()}/Mistral-Large-4-1T-A52B-NVFP4"
    DRAFT_LENGTH = 3
    # MMMU's output budget: at reasoning effort low the answer follows a reasoning block, which
    # the task yaml's 512 tokens cut off in about a third of the samples.
    MMMU_OUTPUT_LEN = 8192
    # The allowed GSM8K gap between the two arms.
    GSM8K_MAX_DELTA = 2 * 0.75

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize(
        "parallel",
        [
            "tep4",
            "dep4",
            "tp4",
            pytest.param("tep8", marks=_EIGHT_RANKS),
            pytest.param("dep8", marks=_EIGHT_RANKS),
            pytest.param("tp8", marks=_EIGHT_RANKS),
            # The 1M specializations of the cases above (the *Ctx1m classes).
            "tep4_1m",
            "dep4_1m",
            "tp4_1m",
            pytest.param("tep8_1m", marks=_EIGHT_RANKS),
            pytest.param("dep8_1m", marks=_EIGHT_RANKS),
            pytest.param("tp8_1m", marks=_EIGHT_RANKS),
        ],
    )
    @pytest.mark.parametrize("profile", ["quick", "full", "long_lite", "long_full"])
    def test_accuracy(self, parallel, profile, tmp_path, monkeypatch, record_property):
        """Every profile runs native DSpark; full profiles add fresh spec OFF GSM8K.

        ``long_lite`` and ``long_full`` run on the 1M specializations only: RULER rows at 64K,
        128K and the 1M bin (cache witnesses over prefixes of up to 1M tokens, the mixed phase
        and exact niah / ruler_vt retrieval), a completed prompt of at least
        ruler.LONG_WITNESS_TOKENS and every step's context tokens within max_num_tokens, then the
        multimodal panel and GSM8K as in quick / full. long_full also gates its 64K / 128K RULER
        means on references/ruler.yaml.
        """
        target = {
            "tep4": TestModelingV2MistralLarge4Nvfp4Sm103Tep4,
            "dep4": TestModelingV2MistralLarge4Nvfp4Sm103Dep4,
            "tp4": TestModelingV2MistralLarge4Nvfp4Sm103Tp4,
            "tep8": TestModelingV2MistralLarge4Nvfp4Sm103Tep8,
            "dep8": TestModelingV2MistralLarge4Nvfp4Sm103Dep8,
            "tp8": TestModelingV2MistralLarge4Nvfp4Sm103Tp8,
            "tep4_1m": TestModelingV2MistralLarge4Nvfp4Sm103Tep4Ctx1m,
            "dep4_1m": TestModelingV2MistralLarge4Nvfp4Sm103Dep4Ctx1m,
            "tp4_1m": TestModelingV2MistralLarge4Nvfp4Sm103Tp4Ctx1m,
            "tep8_1m": TestModelingV2MistralLarge4Nvfp4Sm103Tep8Ctx1m,
            "dep8_1m": TestModelingV2MistralLarge4Nvfp4Sm103Dep8Ctx1m,
            "tp8_1m": TestModelingV2MistralLarge4Nvfp4Sm103Tp8Ctx1m,
        }[parallel]
        envelope = max(ruler.cell_counts(profile))
        if target.BOUNDS["max_seq_len"] < envelope:
            pytest.skip(
                f"{profile} needs max_seq_len {envelope}; {target.__name__} serves "
                f"{target.BOUNDS['max_seq_len']}"
            )
        model_path = self.MODEL_PATH
        is_long = profile.startswith("long")
        world = target.PARALLEL["tensor_parallel_size"]
        attention_dp = target.PARALLEL["enable_attention_dp"]
        for name in (
            "INTEGRATION_TEST",
            "TRTLLM_ACCURACY_NO_REFERENCE",
            "TRTLLM_POPULATE_ACCEPTANCE_LENGTH",
        ):
            assert os.environ.get(name, "0") == "0", f"Disable {name} for accuracy validation"
        data = Path(os.environ.get("TRTLLM_RULER_DATA", f"{llm_models_root()}/datasets/ruler"))
        # Every profile serves images and runs the multimodal panel; the 32K ones also score MMMU.
        scores_mmmu = profile in ("quick", "full")
        # Every profile loads the checkpoint as published, in its Mistral-native format;
        # MISTRAL_LARGE4_CHECKPOINT_DIR names a copy of it elsewhere.
        model_path = os.environ.get("MISTRAL_LARGE4_CHECKPOINT_DIR", model_path)
        # The panel's structured-output case needs xgrammar.
        checkpoint = {"checkpoint_format": "mistral_large_4", "guided_decoding_backend": "xgrammar"}
        rows = ruler.load_inputs(data, profile, Path(model_path) / "tekken.json")
        if profile == "long_lite" and attention_dp:
            # One attention-DP rank prefills a 1M request alone, about 15 minutes per row: the
            # quick 1M case stops at 128K there, and long_full keeps the 1M rows.
            rows = [row for row in rows if row["max_length"] <= 131072]
        root = (
            Path(os.environ.get("TRTLLM_ACCURACY_OUTPUT_DIR", tmp_path)) / f"{parallel}-{profile}"
        )
        root.mkdir(parents=True, exist_ok=False)
        record_property("artifacts", str(root))
        # 16 sequences per rank under attention DP (each rank batches its own), 16 in all
        # otherwise.
        concurrency = 16 * world if attention_dp else 16
        monkeypatch.setenv("TLLM_EVAL_MAX_IN_FLIGHT", str(concurrency))
        failures = []
        spec_config = DSparkDecodingConfig(
            max_draft_len=self.DRAFT_LENGTH,
            speculative_model=model_path,
        )
        summary = {
            "parallel": parallel,
            "profile": profile,
            "reuse": True,
            "speculative_config": spec_config.model_dump(mode="json"),
        }
        llm_kwargs = dict(
            kv_cache_config=target.KV.model_copy(update={"enable_block_reuse": True}),
            modeling_v2="require",
            enable_iter_perf_stats=True,
            env_overrides={MODELING_V2_VALIDATE_ENV: "1", "TLLM_LOG_LEVEL": "info"},
            **checkpoint,
            **target.PARALLEL,
            **target.BOUNDS,
        )
        coverage = {}
        started = time.monotonic()
        try:
            factory = (
                partial(spec_evidence.traced_llm, trace=root / "spec-trace") if is_long else LLM
            )
            with factory(
                model_path,
                speculative_config=spec_config,
                **llm_kwargs,
            ) as llm:
                ready = time.monotonic()
                summary["speculative_config"] = llm.args.speculative_config.model_dump(mode="json")
                summary["startup_s"] = ready - started
                assert llm.args.kv_cache_config.enable_block_reuse
                with _iteration_coverage(llm, root / "ruler-iterations.jsonl", coverage) as (
                    errors,
                    _,
                ):
                    try:
                        summary["ruler"] = ruler.evaluate(
                            llm,
                            rows,
                            profile,
                            root,
                            concurrency,
                            target.MODEL_NAME,
                            collect_spec=is_long,
                        )
                    except AssertionError as error:
                        failures.append(str(error))
                    try:
                        summary["multimodal"] = multimodal.evaluate(llm, profile, root)
                        failures.extend(
                            f"multimodal: {failure}"
                            for failure in summary["multimodal"]["failures"]
                        )
                    except AssertionError as error:
                        failures.append(f"multimodal: {error}")
                failures.extend(f"RULER iteration reader: {error}" for error in errors)
                tasks = [GSM8K(target.MODEL_NAME)]
                if scores_mmmu:
                    tasks.append(MMMU(target.MODEL_NAME))
                if profile == "full":
                    tasks.append(GPQADiamond(target.MODEL_NAME))
                for task in tasks:
                    with _iteration_coverage(
                        llm, root / f"{task.DATASET}-iterations.jsonl", coverage
                    ) as (errors, spec_stats):
                        kwargs = dict(
                            log_samples=True,
                            output_path=str(root / f"{task.DATASET}.json"),
                            output_dir=str(root / task.DATASET),
                        )
                        evaluate_kwargs = {}
                        if isinstance(task, GSM8K):
                            kwargs["num_fewshot"] = 5
                        elif isinstance(task, MMMU):
                            kwargs.update(
                                dataset_path=os.environ.get("TRTLLM_MMMU_DATA", MMMU.DATASET_DIR),
                                chat_template_kwargs={"reasoning_effort": "low"},
                                preserve_caller_max_tokens=True,
                            )
                            # The task yaml sets greedy decoding and the stop string; its 512
                            # tokens yield to MMMU_OUTPUT_LEN. The reference baseline truncates
                            # no prompt.
                            budget = target.BOUNDS["max_seq_len"] - self.MMMU_OUTPUT_LEN
                            evaluate_kwargs = dict(
                                extra_acc_spec=f"native-mm-{profile}",
                                sampling_params=SamplingParams(
                                    max_tokens=self.MMMU_OUTPUT_LEN,
                                    truncate_prompt_tokens=budget,
                                ),
                            )
                        elif isinstance(task, GPQADiamond):
                            kwargs.update(
                                apply_chat_template=True,
                                shuffle_dataset=False,
                                chat_template_kwargs={"reasoning_effort": "low"},
                            )
                        try:
                            summary[task.DATASET] = task.evaluate(
                                llm, extra_evaluator_kwargs=kwargs, **evaluate_kwargs
                            )
                        except AssertionError as error:
                            failures.append(f"{task.DATASET}: {error}")
                    failures.extend(f"{task.DATASET} iteration reader: {error}" for error in errors)
                    if isinstance(task, GSM8K):
                        try:
                            al = acceptance_length_from_iteration_stats(spec_stats)
                            summary["gsm8k_acceptance_length"] = al
                            assert_acceptance_length(
                                f"ModelingV2MistralLarge4Nvfp4::{parallel.removesuffix('_1m')}::dspark3",
                                al,
                            )
                        except (AssertionError, KeyError, ValueError) as error:
                            failures.append(f"GSM8K acceptance length: {error}")
                summary["coverage"] = coverage
                if "multimodal" in summary:
                    with (root / "ruler-iterations.jsonl").open() as stream:
                        failures.extend(
                            f"multimodal: {failure}"
                            for failure in multimodal.rank_evidence(
                                map(json.loads, stream),
                                summary["multimodal"]["windows"],
                                attention_dp,
                            )
                        )
                ranks = range(world if attention_dp else 1)
                if any(
                    coverage.get(rank, {}).get("context_tokens", 0) < 32112
                    or coverage.get(rank, {}).get("decode_batch", 0) < 16
                    for rank in ranks
                ):
                    failures.append(f"INCONCLUSIVE: envelope {coverage}")
                if profile.startswith("long"):
                    longest = ruler.longest_completed(root, rows)
                    summary["longest_prompt_tokens"] = longest
                    need = min(ruler.LONG_WITNESS_TOKENS, max(len(r["prompt_ids"]) for r in rows))
                    if longest < need:
                        failures.append(
                            f"INCONCLUSIVE: no completed {need}-token prompt ({longest})"
                        )
                    step = target.BOUNDS["max_num_tokens"]
                    if any(shape["context_tokens"] > step for shape in coverage.values()):
                        failures.append(
                            f"FAIL: a step's context tokens exceeded {step}: {coverage}"
                        )
                summary["test_wall_s"] = time.monotonic() - ready
            summary["teardown_s"] = time.monotonic() - ready - summary["test_wall_s"]
            if is_long:
                longest_bin = max(row["max_length"] for row in rows)
                need = min(
                    ruler.LONG_WITNESS_TOKENS,
                    min(len(row["prompt_ids"]) for row in rows if row["max_length"] == longest_bin),
                )
                summary["spec_evidence"] = spec_evidence.evaluate(root, need, world, attention_dp)
                failures.extend(summary["spec_evidence"]["failures"])
            if profile in ("full", "long_full"):
                # Release the target and draft completely before starting the other arm.
                # Same specialization, reuse, bounds and evaluator; only speculation differs.
                origin = root / "spec-off"
                origin.mkdir()
                task = GSM8K(target.MODEL_NAME)
                with LLM(model_path, **llm_kwargs) as llm:
                    with _iteration_coverage(llm, origin / "iterations.jsonl", {}) as (
                        errors,
                        _,
                    ):
                        try:
                            score = task.evaluate(
                                llm,
                                extra_evaluator_kwargs=dict(
                                    num_fewshot=5,
                                    log_samples=True,
                                    output_path=str(origin / f"{task.DATASET}.json"),
                                    output_dir=str(origin / task.DATASET),
                                ),
                            )
                            summary["gsm8k_spec_off"] = score
                        except AssertionError as error:
                            failures.append(f"GSM8K spec OFF: {error}")
                    failures.extend(f"Spec OFF iteration reader: {error}" for error in errors)
                if task.DATASET in summary and "gsm8k_spec_off" in summary:
                    delta = summary[task.DATASET] - summary["gsm8k_spec_off"]
                    summary["gsm8k_delta_pp"] = delta
                    summary["gsm8k_max_delta_pp"] = self.GSM8K_MAX_DELTA
                    if abs(delta) > self.GSM8K_MAX_DELTA:
                        failures.append(
                            f"GSM8K spec ON/OFF delta {delta:+.4f} pp exceeds "
                            f"{self.GSM8K_MAX_DELTA:.2f} pp"
                        )
        except Exception as error:
            failures.append(f"{type(error).__name__}: {error}")
            raise
        finally:
            summary["failures"] = list(failures)
            (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        assert not failures, failures

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_llguidance(self, tmp_path, record_property):
        """Guided decoding with llguidance on tp4 with native DSpark, over frontend-rendered chat ids.

        Cases (llguidance_panel): JSON, JSON after a reasoning section, and mistral-common's tool
        grammars under tool_choice auto, required and named.
        """
        target = TestModelingV2MistralLarge4Nvfp4Sm103Tp4
        model_path = os.environ.get("MISTRAL_LARGE4_CHECKPOINT_DIR", self.MODEL_PATH)
        root = Path(os.environ.get("TRTLLM_ACCURACY_OUTPUT_DIR", tmp_path)) / "tp4-llguidance"
        root.mkdir(parents=True, exist_ok=False)
        record_property("artifacts", str(root))
        with LLM(
            model_path,
            speculative_config=DSparkDecodingConfig(
                max_draft_len=self.DRAFT_LENGTH, speculative_model=model_path
            ),
            checkpoint_format="mistral_large_4",
            guided_decoding_backend="llguidance",
            kv_cache_config=target.KV.model_copy(update={"enable_block_reuse": True}),
            modeling_v2="require",
            env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
            **target.PARALLEL,
            **target.BOUNDS,
        ) as llm:
            summary = llguidance_panel.evaluate(llm, root)
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        failures = [failure for failure in summary["failures"] if failure.startswith("FAIL")]
        assert not failures, failures
