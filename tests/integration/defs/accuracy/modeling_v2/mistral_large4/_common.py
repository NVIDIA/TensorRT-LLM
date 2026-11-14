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
"""Shared harness for the mistral_large4 modeling_v2 accuracy targets.

Each target's gate class is in ``<target>/test_accuracy.py``. It names the target's
architecture and layout and lists its tests, which run the ``_test_*`` implementations here;
the layouts and the deployments are here too.

Every test passes ``modeling_v2="require"`` to ``LLM``. Under ``"auto"`` a
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

from ....conftest import llm_models_root
from ...accuracy_core import (
    GSM8K,
    MMMU,
    GPQADiamond,
    LlmapiAccuracyTestHarness,
    acceptance_length_from_iteration_stats,
    assert_acceptance_length,
)
from .. import llguidance_panel, multimodal, ruler, spec_evidence
from .._env import scoped_env_overrides

# Certification is per GPU architecture, and a receipt from another one says nothing here:
# each gate class runs only on its target's architecture.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103,
    reason="the mistral_large4 sm_103 modeling_v2 targets are certified on sm_103 only",
)
skip_not_sm107 = pytest.mark.skipif(
    get_sm_version() != 107,
    reason="the mistral_large4 sm_107 modeling_v2 targets are certified on sm_107 only",
)


# The eight-rank deployments: eight ranks over two nodes of four (one NVL72 segment). Each node
# sees four devices, so the gate is the MPI world the launcher started (one node of eight,
# unlaunched, counts its devices); the tests' four-device gate holds on every node.
_EIGHT_RANKS = pytest.mark.skip_less_mpi_world_size(8)

# The engine bounds every target serves (per rank under attention DP): 16 sequences, unchunked
# steps of up to 32768 tokens (a whole 32K prompt in one step), CUDA graphs padded over batches
# 1-16. The certified recipe: max_seq_len = max_num_tokens = 32768, chunked prefill off.
_BOUNDS = dict(
    max_batch_size=16,
    max_num_tokens=32768,
    max_seq_len=32768,
    enable_chunked_prefill=False,
    cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
)

# The 1M envelope: the same target and topology at the checkpoint's own context
# (max_position_embeddings 1048576; per rank under attention DP), prefilled in chunked steps of
# up to max_num_tokens 32768.
_BOUNDS_1M = dict(_BOUNDS, max_seq_len=1048576, enable_chunked_prefill=True)

# The latent KV cache at its default: the checkpoint's fp8_kv_cache makes it the fp8_ds_mla
# cache the reference implementation serves (656 B per token and owner layer,
# MistralLarge4CacheManager, a pool per rank under attention DP), read as kv_cache_quant_algo
# FP8; the checkpoint's quant_algo is MIXED_PRECISION, so the harness reads the
# MIXED_PRECISION / FP8 row.
_KV = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85)


# The profiles ``test_accuracy`` runs, and the ones ``test_accuracy_without_dspark`` runs.
PROFILES = ["quick", "full", "long_lite", "long_full"]
PROFILES_WITHOUT_DSPARK = ["full", "long_full"]


def pytest_generate_tests(metafunc):
    """Run each gate test at every deployment its target serves.

    A deployment is the target's layout on four ranks of one node or on eight over two nodes of
    four, at the 32K recipe or at the 1M envelope: ``tep4``, ``tep8``, ``tep4_1m``, ``tep8_1m``.
    """
    if "deployment" not in metafunc.fixturenames:
        return
    metafunc.parametrize(
        "deployment",
        [
            pytest.param(name, marks=_EIGHT_RANKS) if name.split("_")[0].endswith("8") else name
            for name in metafunc.cls.deployments()
        ],
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


@contextlib.contextmanager
def _phase_timing(timings: dict, phase: str):
    started = time.monotonic()
    print(f"Accuracy phase {phase}: started", flush=True)
    try:
        yield
    finally:
        timings[phase] = time.monotonic() - started
        print(f"Accuracy phase {phase}: {timings[phase]:.3f} s", flush=True)


class _Nvfp4Target(LlmapiAccuracyTestHarness):
    """The gate of one target: every test runs at each deployment the target serves.

    A subclass names its layout and its parallel arguments at four and at eight ranks;
    `pytest_generate_tests` runs each test at both, at the 32K recipe and at the 1M envelope.
    """

    MODEL_NAME = "mistralai/Mistral-Large-4-1T-A52B"
    # The Mistral-native ModelOpt MIXED_PRECISION export as published (params.json,
    # tekken.json, consolidated safetensors: NVFP4 routed experts, FP8 block-scale linears).
    MODEL_PATH = f"{llm_models_root()}/Mistral-Large-4-1T-A52B-NVFP4"
    KV = _KV
    # The layout's short name, and its parallel arguments by rank count.
    LAYOUT = ""
    LAYOUTS = {}

    DRAFT_LENGTH = 3
    # MMMU's output budget: at reasoning effort low the answer follows a reasoning block, which
    # the task yaml's 512 tokens cut off in about a third of the samples.
    MMMU_OUTPUT_LEN = 8192
    # The allowed GSM8K gap between the two arms.
    GSM8K_MAX_DELTA = 2 * 0.75

    @classmethod
    def deployments(cls):
        """Every deployment the target serves, named ``<layout><ranks>[_1m]``."""
        return [f"{cls.LAYOUT}{ranks}{ctx}" for ctx in ("", "_1m") for ranks in sorted(cls.LAYOUTS)]

    @classmethod
    def serving(cls, deployment):
        """A deployment's parallel arguments and engine bounds."""
        name, _, ctx = deployment.partition("_")
        layout = cls.LAYOUTS[int(name.removeprefix(cls.LAYOUT))]
        return layout, (_BOUNDS_1M if ctx == "1m" else _BOUNDS)

    def _test_gsm8k(self, deployment):
        """The identity accuracy gate, at the harness default protocol.

        That is the protocol the registered reference was set under, so the anchor and the
        measurement read the same metric. The engine also checks its step contract on its
        first forward.
        """
        layout, bounds = self.serving(deployment)
        task = GSM8K(self.MODEL_NAME)
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                self.MODEL_PATH,
                checkpoint_format="mistral",
                kv_cache_config=self.KV,
                modeling_v2="require",
                env_overrides=rank_env,
                **layout,
                **bounds,
            ) as llm,
        ):
            task.evaluate(llm)

    # GSM8K with 24 exemplars: every prompt is 3152-5115 tokens, past index_topk, so the
    # indexer's selection prunes context on every sample (5-shot prompts stay under 2048,
    # where the engine skips the indexer). A key of its own in gsm8k.yaml, scored by the
    # reference implementation on its fp8_ds_mla cache.
    MODEL_NAME_24SHOT = "mistralai/Mistral-Large-4-1T-A52B-gsm8k-24shot"

    class GSM8K24Shot(GSM8K):
        MAX_INPUT_LEN = 8192

    def _test_gsm8k_24shot(self, deployment):
        """24-shot GSM8K, where the indexer's selection prunes every prompt.

        `trtllm-eval gsm8k --num_fewshot 24 --max_input_length 8192`, otherwise the harness
        protocol of the 5-shot gate.
        """
        layout, bounds = self.serving(deployment)
        task = self.GSM8K24Shot(self.MODEL_NAME_24SHOT)
        with (
            scoped_env_overrides({}) as rank_env,
            LLM(
                self.MODEL_PATH,
                checkpoint_format="mistral",
                kv_cache_config=self.KV,
                modeling_v2="require",
                env_overrides=rank_env,
                **layout,
                **bounds,
            ) as llm,
        ):
            task.evaluate(llm, extra_evaluator_kwargs=dict(num_fewshot=24))

    # A passkey hidden in filler text, asked for at the end. GSM8K's prompts stay under the
    # indexer's index_topk (2048), where the engine skips the indexer; these prompts are about
    # 30000 tokens -- one context step, within max_num_tokens 32768 -- so the indexer scores and
    # selects for every query row past 2048, in the context step and in each decode step.
    PASSKEY_TOKENS = 30000
    PASSKEY_FILLER = (
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    )

    def _test_passkey_retrieval(self, deployment):
        """The long-context witness: a passkey recalled from a ~30000-token prompt.

        At three depths of the prompt, greedy decoding must reproduce the passkey. A wrong
        selection, rope or SSMax position past 2048 reads the wrong positions and loses it.
        """
        layout, bounds = self.serving(deployment)
        params = SamplingParams(max_tokens=12, temperature=0.0)
        with (
            scoped_env_overrides({}) as rank_env,
            LLM(
                self.MODEL_PATH,
                checkpoint_format="mistral",
                kv_cache_config=self.KV,
                modeling_v2="require",
                env_overrides=rank_env,
                **layout,
                **bounds,
            ) as llm,
        ):
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
                assert 2048 < tokens <= bounds["max_seq_len"] - params.max_tokens, tokens
                answer = llm.generate([prompt], params)[0].outputs[0].text
                assert str(key) in answer, (depth, tokens, answer)

    def _test_accuracy(self, deployment, profile, tmp_path, monkeypatch, record_property):
        """Every profile runs native DSpark; full profiles add fresh spec OFF GSM8K.

        ``long_lite`` and ``long_full`` run on the 1M specializations only: RULER rows at 64K,
        128K and the 1M bin (cache witnesses over prefixes of up to 1M tokens, the mixed phase
        and exact niah / ruler_vt retrieval), a completed prompt of at least
        ruler.LONG_WITNESS_TOKENS and every step's context tokens within max_num_tokens, then the
        multimodal panel and GSM8K as in quick / full. long_full also gates its 64K / 128K RULER
        means on references/ruler.yaml.
        """
        self._run_accuracy(
            deployment, profile, tmp_path, monkeypatch, record_property, use_dspark=True
        )

    def _test_accuracy_without_dspark(
        self, deployment, profile, tmp_path, monkeypatch, record_property
    ):
        """The complete accuracy and cache protocol with native DSpark disabled.

        ``full`` covers the 32K datasets; ``long_full`` covers the long-context datasets and
        1M witnesses. Only draft acceptance and speculative execution evidence are omitted.
        """
        self._run_accuracy(
            deployment, profile, tmp_path, monkeypatch, record_property, use_dspark=False
        )

    def _run_accuracy(
        self, deployment, profile, tmp_path, monkeypatch, record_property, *, use_dspark: bool
    ):
        run_started = time.monotonic()
        layout, bounds = self.serving(deployment)
        envelope = max(ruler.cell_counts(profile))
        if bounds["max_seq_len"] < envelope:
            pytest.skip(
                f"{profile} needs max_seq_len {envelope}; {deployment} serves "
                f"{bounds['max_seq_len']}"
            )
        model_path = self.MODEL_PATH
        is_long = profile.startswith("long")
        world = layout["tensor_parallel_size"]
        attention_dp = layout["enable_attention_dp"]
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
        checkpoint = {"checkpoint_format": "mistral", "guided_decoding_backend": "xgrammar"}
        rows = ruler.load_inputs(data, profile, Path(model_path) / "tekken.json")
        if profile == "long_lite" and attention_dp:
            # One attention-DP rank prefills a 1M request alone, about 15 minutes per row: the
            # quick 1M case stops at 128K there, and long_full keeps the 1M rows.
            rows = [row for row in rows if row["max_length"] <= 131072]
        gpu_arch = f"sm_{get_sm_version()}"
        root = (
            Path(os.environ.get("TRTLLM_ACCURACY_OUTPUT_DIR", tmp_path))
            / gpu_arch
            / f"{deployment}-{profile}{'' if use_dspark else '-spec-off'}"
        )
        root.mkdir(parents=True, exist_ok=False)
        record_property("artifacts", str(root))
        record_property("gpu_arch", gpu_arch)
        record_property("dspark", use_dspark)
        # 16 sequences per rank under attention DP (each rank batches its own), 16 in all
        # otherwise.
        concurrency = 16 * world if attention_dp else 16
        monkeypatch.setenv("TLLM_EVAL_MAX_IN_FLIGHT", str(concurrency))
        failures = []
        spec_config = (
            DSparkDecodingConfig(
                max_draft_len=self.DRAFT_LENGTH,
                speculative_model=model_path,
            )
            if use_dspark
            else None
        )
        timings = {}
        summary = {
            "gpu_arch": gpu_arch,
            "deployment": deployment,
            "profile": profile,
            "reuse": True,
            "dspark": use_dspark,
            "speculative_config": spec_config.model_dump(mode="json") if spec_config else None,
            "phase_wall_s": timings,
        }
        llm_kwargs = dict(
            kv_cache_config=self.KV.model_copy(update={"enable_block_reuse": True}),
            modeling_v2="require",
            enable_iter_perf_stats=True,
            env_overrides={MODELING_V2_VALIDATE_ENV: "1", "TLLM_LOG_LEVEL": "info"},
            **checkpoint,
            **layout,
            **bounds,
        )
        coverage = {}
        started = time.monotonic()
        try:
            factory = (
                partial(spec_evidence.traced_llm, trace=root / "spec-trace")
                if is_long and use_dspark
                else LLM
            )
            with (
                scoped_env_overrides(llm_kwargs["env_overrides"]) as rank_env,
                factory(
                    model_path,
                    speculative_config=spec_config,
                    **dict(llm_kwargs, env_overrides=rank_env),
                ) as llm,
            ):
                ready = time.monotonic()
                assert (llm.args.speculative_config is not None) == use_dspark
                summary["speculative_config"] = (
                    llm.args.speculative_config.model_dump(mode="json") if use_dspark else None
                )
                summary["startup_s"] = ready - started
                assert llm.args.kv_cache_config.enable_block_reuse
                with _iteration_coverage(llm, root / "ruler-iterations.jsonl", coverage) as (
                    errors,
                    _,
                ):
                    try:
                        with _phase_timing(timings, "ruler"):
                            summary["ruler"] = ruler.evaluate(
                                llm,
                                rows,
                                profile,
                                root,
                                concurrency,
                                self.MODEL_NAME,
                                collect_spec=is_long and use_dspark,
                            )
                    except AssertionError as error:
                        failures.append(str(error))
                    try:
                        with _phase_timing(timings, "multimodal"):
                            summary["multimodal"] = multimodal.evaluate(llm, profile, root)
                        failures.extend(
                            f"multimodal: {failure}"
                            for failure in summary["multimodal"]["failures"]
                        )
                    except AssertionError as error:
                        failures.append(f"multimodal: {error}")
                failures.extend(f"RULER iteration reader: {error}" for error in errors)
                tasks = [GSM8K(self.MODEL_NAME)]
                if scores_mmmu:
                    tasks.append(MMMU(self.MODEL_NAME))
                if profile == "full":
                    tasks.append(GPQADiamond(self.MODEL_NAME))
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
                            budget = bounds["max_seq_len"] - self.MMMU_OUTPUT_LEN
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
                            with _phase_timing(timings, task.DATASET):
                                summary[task.DATASET] = task.evaluate(
                                    llm, extra_evaluator_kwargs=kwargs, **evaluate_kwargs
                                )
                        except AssertionError as error:
                            failures.append(f"{task.DATASET}: {error}")
                    failures.extend(f"{task.DATASET} iteration reader: {error}" for error in errors)
                    if isinstance(task, GSM8K) and use_dspark:
                        try:
                            al = acceptance_length_from_iteration_stats(spec_stats)
                            summary["gsm8k_acceptance_length"] = al
                            assert_acceptance_length(
                                f"ModelingV2MistralLarge4Nvfp4::{deployment.removesuffix('_1m')}::dspark3",
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
                    step = bounds["max_num_tokens"]
                    if any(shape["context_tokens"] > step for shape in coverage.values()):
                        failures.append(
                            f"FAIL: a step's context tokens exceeded {step}: {coverage}"
                        )
                summary["test_wall_s"] = time.monotonic() - ready
            summary["teardown_s"] = time.monotonic() - ready - summary["test_wall_s"]
            if is_long and use_dspark:
                longest_bin = max(row["max_length"] for row in rows)
                need = min(
                    ruler.LONG_WITNESS_TOKENS,
                    min(len(row["prompt_ids"]) for row in rows if row["max_length"] == longest_bin),
                )
                summary["spec_evidence"] = spec_evidence.evaluate(root, need, world, attention_dp)
                failures.extend(summary["spec_evidence"]["failures"])
            if use_dspark and profile in ("full", "long_full"):
                # Release the target and draft completely before starting the other arm.
                # Same specialization, reuse, bounds and evaluator; only speculation differs.
                origin = root / "spec-off"
                origin.mkdir()
                task = GSM8K(self.MODEL_NAME)
                spec_off_started = time.monotonic()
                with (
                    scoped_env_overrides(llm_kwargs["env_overrides"]) as rank_env,
                    LLM(model_path, **dict(llm_kwargs, env_overrides=rank_env)) as llm,
                ):
                    spec_off_ready = time.monotonic()
                    summary["spec_off_startup_s"] = spec_off_ready - spec_off_started
                    with _iteration_coverage(llm, origin / "iterations.jsonl", {}) as (
                        errors,
                        _,
                    ):
                        try:
                            with _phase_timing(timings, "gsm8k_spec_off"):
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
                    spec_off_completed = time.monotonic()
                summary["spec_off_teardown_s"] = time.monotonic() - spec_off_completed
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
            summary["total_wall_s"] = time.monotonic() - run_started
            summary["failures"] = list(failures)
            (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        assert not failures, failures


class _TepTarget(_Nvfp4Target):
    """The tep layout, the language model."""

    # Tensor-parallel attention with expert-parallel experts: the attention heads and the routed
    # experts both sharded over every rank, attention DP off -- four ways on one node, or eight
    # over two nodes of four. Routing derives the target *from* these: attention DP would be a
    # different weight layout (the dep target), and tensor_parallel_size alone shards every
    # expert by width instead (the tp target).
    LAYOUT = "tep"
    LAYOUTS = {
        4: dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False),
        8: dict(tensor_parallel_size=8, moe_expert_parallel_size=8, enable_attention_dp=False),
    }


class _DepTarget(_Nvfp4Target):
    """The dep layout, the language model."""

    # Attention data parallel over every rank (each holds the whole attention and serves its
    # own requests) with the routed experts split by expert over every rank, four on one node
    # or eight over two nodes of four. Routing derives the target *from* these: attention DP
    # off is the tep target, and tensor_parallel_size alone another weight layout.
    LAYOUT = "dep"
    LAYOUTS = {
        4: dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True),
        8: dict(tensor_parallel_size=8, moe_expert_parallel_size=8, enable_attention_dp=True),
    }


class _TpTarget(_Nvfp4Target):
    """The tp layout, the language model."""

    # Tensor parallel throughout: the attention heads sharded over every rank and every routed
    # expert on every rank, cut to its share of the width, attention DP off -- four ways on one
    # node, or eight over two nodes of four. Routing derives the target *from* these: expert
    # parallelism is the tep target. A bare tensor_parallel_size resolves to the same split.
    LAYOUT = "tp"
    LAYOUTS = {
        4: dict(
            tensor_parallel_size=4,
            moe_tensor_parallel_size=4,
            moe_expert_parallel_size=1,
            enable_attention_dp=False,
        ),
        8: dict(
            tensor_parallel_size=8,
            moe_tensor_parallel_size=8,
            moe_expert_parallel_size=1,
            enable_attention_dp=False,
        ),
    }

    def _test_llguidance(self, tmp_path, record_property):
        """Guided decoding with llguidance on tp4 with native DSpark, over frontend-rendered chat ids.

        Cases (llguidance_panel): JSON, JSON after a reasoning section, and mistral-common's tool
        grammars under tool_choice auto, required and named.
        """
        layout, bounds = self.serving("tp4")
        model_path = os.environ.get("MISTRAL_LARGE4_CHECKPOINT_DIR", self.MODEL_PATH)
        gpu_arch = f"sm_{get_sm_version()}"
        root = (
            Path(os.environ.get("TRTLLM_ACCURACY_OUTPUT_DIR", tmp_path))
            / gpu_arch
            / "tp4-llguidance"
        )
        root.mkdir(parents=True, exist_ok=False)
        record_property("artifacts", str(root))
        record_property("gpu_arch", gpu_arch)
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(
                model_path,
                speculative_config=DSparkDecodingConfig(
                    max_draft_len=self.DRAFT_LENGTH, speculative_model=model_path
                ),
                checkpoint_format="mistral",
                guided_decoding_backend="llguidance",
                kv_cache_config=self.KV.model_copy(update={"enable_block_reuse": True}),
                modeling_v2="require",
                env_overrides=rank_env,
                **layout,
                **bounds,
            ) as llm,
        ):
            summary = llguidance_panel.evaluate(llm, root)
        summary["gpu_arch"] = gpu_arch
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        failures = [failure for failure in summary["failures"] if failure.startswith("FAIL")]
        assert not failures, failures
