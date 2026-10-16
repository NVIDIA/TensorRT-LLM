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
"""Whole-model gates for the qwen4_exp modeling_v2 target.

Every test passes ``modeling_v2="require"``: under ``"auto"`` a configuration
that missed the target's routing would fall back to the built-in model, pass,
and report the built-in's numbers as the target's.
"""

import random
import re

import pytest
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as target_modeling,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.inputs import VideoData
from tensorrt_llm.inputs.utils import (
    ConversationMessage,
    MultimodalDataTracker,
    apply_chat_template,
    apply_mm_placeholders,
)
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
    QSASparseAttentionConfig,
    SamplingParams,
)

from ...conftest import llm_models_root
from ..accuracy_core import (
    GSM8K,
    MMMU,
    LlmapiAccuracyTestHarness,
    assert_acceptance_length,
    compute_acceptance_length,
)

# The target is certified on sm_103 only.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="modeling_v2 targets in this batch are certified on sm_103 only"
)

# Check the target's step contract on the first forward, against metadata the
# engine prepared; the flag has to reach the worker rank.
_VALIDATE = {MODELING_V2_VALIDATE_ENV: "1"}


def _require_target(monkeypatch: pytest.MonkeyPatch, **env: str) -> dict:
    """``LLM(...)`` kwargs that build the target, with ``env`` on the ranks.

    ``LLM`` applies ``env_overrides`` to this process and never restores it, so
    ``monkeypatch`` sets it too, keeping it from leaking into later tests.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return dict(modeling_v2="require", env_overrides=dict(env))


def _target_core(llm: LLM):
    """The target's core module.

    Reachable only with the executor in this process (``TLLM_WORKER_USE_SINGLE_PROCESS=1``),
    where the iteration stats that ``compute_acceptance_length`` reads are not collected.
    """
    return llm._executor.engine.model_engine.model.model


def _acceptance_length_from_requests(outputs, draft_len: int) -> float:
    """Mean acceptance length from the requests' own speculative-decoding metrics.

    Needs ``SamplingParams(return_perf_metrics=True)``. A draft loop yields one target token
    plus the drafts it accepted.
    """
    accepted = drafted = 0
    for output in outputs:
        spec = output.outputs[0].request_perf_metrics.speculative_decoding
        accepted += spec.total_accepted_draft_tokens
        drafted += spec.total_draft_tokens
    assert drafted > 0, "no request drafted a token"
    return 1.0 + accepted * draft_len / drafted


# ---------------------------------------------------------------- long prompts ---
# Prompts past the QSA indexer's 2048-token budget, so every decode step of the
# target and of its MTP draft layer takes the sparse attention path.

_PASSKEY_LENGTHS = (3000, 4000, 5000, 6000)
_PASSKEY_LENGTHS_LONG = (3000, 6000, 9000, 12000)
# 33 prompts, each past the indexer's budget, 32 of which one batch carries.
_FORCED_LENGTHS = tuple(range(2200, 6400, 400))
_FORCED_TOKENS = 64
_PASSKEY_DEPTHS = (0.15, 0.5, 0.85)
_PASSKEY_MAX_TOKENS = 160

_COMPONENTS = (
    "scheduler",
    "allocator",
    "tokenizer",
    "router",
    "profiler",
    "exporter",
    "planner",
    "balancer",
    "resolver",
    "dispatcher",
)
_ACTIONS = (
    "reads",
    "rewrites",
    "caches",
    "validates",
    "forwards",
    "compacts",
    "audits",
    "replays",
    "indexes",
    "batches",
)
_OBJECTS = (
    "pending requests",
    "block tables",
    "shard manifests",
    "routing hints",
    "retry budgets",
    "lease records",
    "checkpoint deltas",
    "quota counters",
)
_CLAUSES = (
    "before the next flush",
    "whenever the queue drains",
    "after every shard reports",
    "once the lease expires",
    "while the window is open",
    "until the barrier clears",
)
_STATES = ("consistent", "bounded", "ordered", "reproducible", "idle")
_TOPICS = ("rules", "limits", "guarantees", "hooks", "settings")


def _section(rng: random.Random, number: int) -> str:
    heading = f"## Section {number}: {rng.choice(_COMPONENTS).capitalize()} {rng.choice(_TOPICS)}"
    sentences = [
        f"The {rng.choice(_COMPONENTS)} {rng.choice(_ACTIONS)} the {rng.choice(_OBJECTS)} "
        f"{rng.choice(_CLAUSES)}, so the {rng.choice(_OBJECTS)} stay {rng.choice(_STATES)}."
        for _ in range(6)
    ]
    return heading + "\n\n" + " ".join(sentences)


def _passkey_prompts(tokenizer, lengths=_PASSKEY_LENGTHS) -> list[tuple[str, str]]:
    """``(chat prompt, passkey)`` pairs over ``lengths`` x ``_PASSKEY_DEPTHS``.

    Each prompt is a generated document of about that many tokens with one pass key sentence
    at that relative depth, the same bytes in every run.
    """
    rng = random.Random(20261014)
    question = (
        "Based on the document above, first state the secret pass key on a line of its own "
        "in the form 'Pass key: <number>'. Then list, one per line, the section headings "
        "that appear in the document, in order."
    )
    prompts = []
    for length in lengths:
        for depth in _PASSKEY_DEPTHS:
            key = str(rng.randint(100000, 999999))
            sections, tokens = [], 0
            while tokens < length - 150:
                sections.append(_section(rng, len(sections) + 1))
                tokens += len(tokenizer.encode(sections[-1]))
            needle = (
                f"IMPORTANT NOTE FOR THE READER: the secret pass key of this document is {key}. "
                f"Remember the pass key {key} exactly."
            )
            position = max(1, min(len(sections) - 1, round(depth * len(sections))))
            body = sections[:position] + [needle] + sections[position:]
            text = "Document:\n\n" + "\n\n".join(body) + "\n\n" + question
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append((prompt, key))
    return prompts


# ------------------------------------------------------------------ word problems ---

_WORD_PROBLEM_SUFFIX = "Solve it step by step and finish with 'The answer is <number>.'"
_WORD_PROBLEMS = (
    (
        "A bakery bakes 7 trays of 12 muffins and sells all but 9. How many muffins did it sell?",
        "75",
    ),
    (
        "Tom has 3 boxes of 24 pencils and gives away 19 pencils. How many pencils does he "
        "have left?",
        "53",
    ),
    (
        "A train travels 60 miles per hour for 2.5 hours, then 40 miles per hour for 3 hours. "
        "How many miles does it travel in total?",
        "270",
    ),
    (
        "Sara buys 5 notebooks at $3 each and 4 pens at $2 each and pays with a $50 bill. "
        "How many dollars of change does she get?",
        "27",
    ),
    ("A rectangle is 14 cm long and 9 cm wide. What is its perimeter in cm?", "46"),
    (
        "125 students attend; 40% play soccer and one fifth of the others play tennis. "
        "How many students play tennis?",
        "15",
    ),
)


# ---------------------------------------------------------------- media inputs ---

_MODEL_TYPE = "qwen4_exp"
_RED, _BLUE, _GREEN = (220, 30, 30), (30, 60, 220), (30, 170, 60)
# name -> (square colour, direction, colour word, direction phrases an answer may use)
_VIDEOS = {
    "red_rightward": (_RED, "right", "red", ("left to right", "to the right", "rightward")),
    "blue_leftward": (_BLUE, "left", "blue", ("right to left", "to the left", "leftward")),
    "green_downward": (_GREEN, "down", "green", ("top to bottom", "downward", "down")),
}
_VIDEO_QUESTION = (
    "What colour is the moving square, and in which direction does it move across the frame? "
    "Answer in one short sentence."
)
_MIXED_QUESTION = (
    "The image shows a square and the video shows a moving square. Answer in exactly this "
    "format: 'Image: <colour of its square>. Video: <colour of its square>.'"
)


def _synthetic_video(name: str, num_frames: int = 16, size: int = 256) -> VideoData:
    """A square of the named colour crossing a plain background in the named direction.

    The frames are built in memory, so no video decoder is involved.
    """
    color, direction, _, _ = _VIDEOS[name]
    side, margin = 48, 8
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        travel = margin + t * (size - side - 2 * margin)
        centre = (size - side) // 2
        x, y = {
            "right": (travel, centre),
            "left": (size - side - travel, centre),
            "down": (centre, travel),
        }[direction]
        frame = Image.new("RGB", (size, size), (235, 235, 235))
        ImageDraw.Draw(frame).rectangle((x, y, x + side, y + side), fill=color)
        frames.append(frame)
    return VideoData(
        frames=frames,
        metadata=dict(
            total_num_frames=num_frames,
            fps=2.0,
            duration=num_frames / 2.0,
            frames_indices=list(range(num_frames)),
        ),
    )


def _synthetic_image(color: tuple, size: int = 256) -> Image.Image:
    """A square of ``color`` on a plain background."""
    image = Image.new("RGB", (size, size), (235, 235, 235))
    quarter = size // 4
    ImageDraw.Draw(image).rectangle((quarter, quarter, size - quarter, size - quarter), fill=color)
    return image


def _media_request(tokenizer, processor, text: str, items: list[tuple[str, object]]) -> dict:
    """One request carrying ``items``, ``(modality, data)`` pairs in prompt order.

    It holds the chat prompt with its placeholders, the media and the ``mm_item_order``
    manifest that splices several modalities back in prompt order.
    """
    tracker = MultimodalDataTracker(_MODEL_TYPE)
    for modality, data in items:
        tracker.add_data(modality, data)
    counts = tracker.placeholder_counts()
    message = ConversationMessage(role="user", content=text, media=[])
    apply_mm_placeholders(_MODEL_TYPE, message, counts, tracker)
    prompt = apply_chat_template(
        model_type=_MODEL_TYPE,
        tokenizer=tokenizer,
        processor=processor,
        conversation=[message],
        add_generation_prompt=True,
        mm_placeholder_counts=[counts],
        chat_template_kwargs=dict(enable_thinking=False),
    )
    media, _ = tracker.retrieve_all_sync()
    return dict(prompt=prompt, multi_modal_data=media, mm_item_order=tracker.item_order())


class TestModelingV2Qwen38FlashNextNvfp4Sm103Tp1(LlmapiAccuracyTestHarness):
    """qwen3.8-flash-next-nvfp4 / sm_103 / tp1: text, MTP and multimodal."""

    # References are keyed by model name plus the quantization the engine
    # resolves for this checkpoint (MIXED_PRECISION).
    MODEL_NAME = "Qwen/Qwen3.8-Flash-Next"
    MODEL_PATH = f"{llm_models_root()}/Qwen3.8-Flash-Next-NVFP4"

    # The gsm8k anchors were measured under this protocol. `enable_thinking:
    # false` is required: with thinking on, the answer does not finish inside
    # GSM8K_MAX_OUTPUT_LEN.
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        chat_template_kwargs=dict(enable_thinking=False),
    )

    # The anchor's `max_tokens`; the class default (256) truncates this checkpoint's answers.
    GSM8K_MAX_OUTPUT_LEN = 512

    # `mamba_ssm_cache_dtype` is float32 because the checkpoint declares
    # `mamba_ssm_dtype: float32` (the engine default is bfloat16); the bf16-state cases
    # set it to bfloat16. `dtype` is left at the checkpoint's bf16 KV, which is what the
    # anchors were measured at; only the fp8 KV cases set it.
    KV_CACHE = KvCacheConfig(
        free_gpu_memory_fraction=0.6,
        enable_block_reuse=True,
        mamba_ssm_cache_dtype="float32",
    )
    KV_CACHE_BF16_STATE = KvCacheConfig(
        free_gpu_memory_fraction=0.6,
        enable_block_reuse=True,
        mamba_ssm_cache_dtype="bfloat16",
    )
    KV_CACHE_FP8 = KvCacheConfig(
        dtype="fp8",
        free_gpu_memory_fraction=0.6,
        enable_block_reuse=True,
        mamba_ssm_cache_dtype="float32",
    )
    # Block reuse is off for the vision path, which requires it.
    KV_CACHE_MULTIMODAL = KvCacheConfig(
        free_gpu_memory_fraction=0.6,
        enable_block_reuse=False,
        mamba_ssm_cache_dtype="float32",
    )

    # Required, not a tuning choice: the MTP draft layer's FP8 block-scale
    # experts are not supported by the default CUTLASS MoE on sm_103. Set for
    # every case so they share one engine configuration.
    MOE = MoeConfig(backend="TRTLLM")

    ENGINE = dict(
        max_seq_len=8192,
        max_batch_size=32,
        enable_chunked_prefill=True,
        kv_cache_config=KV_CACHE,
        cuda_graph_config=CudaGraphConfig(),
        moe_config=MOE,
    )

    # One-model MTP-Eagle with the checkpoint's single MTP module replayed
    # `max_draft_len` times.
    MTP1 = MTPDecodingConfig(max_draft_len=1)
    MTP3 = MTPDecodingConfig(max_draft_len=3)
    MTP5 = MTPDecodingConfig(max_draft_len=5)
    MTP7 = MTPDecodingConfig(max_draft_len=7)

    # Acceptance-length anchors in references/acceptance_length.yaml, keyed by
    # target and variant.
    ACCEPTANCE_KEY = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp1"
    ACCEPTANCE_KEY_FP8_KV = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp1_fp8_kv"
    ACCEPTANCE_KEY_MTP3 = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp3"
    ACCEPTANCE_KEY_MTP3_CUTEDSL = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp3_cutedsl"
    ACCEPTANCE_KEY_MTP5 = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp5"
    ACCEPTANCE_KEY_MTP7 = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp7"
    ACCEPTANCE_KEY_MTP7_LONG_CONTEXT = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp7_long_context"
    ACCEPTANCE_KEY_MTP3_BF16_STATE = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp3_bf16_state"
    ACCEPTANCE_KEY_MTP7_BF16_STATE_LONG_CONTEXT = (
        "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp7_bf16_state_long_context"
    )
    ACCEPTANCE_KEY_LONG_CONTEXT = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp3_long_context"
    ACCEPTANCE_KEY_FP8_KV_LONG_CONTEXT = (
        "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp3_fp8_kv_long_context"
    )
    ACCEPTANCE_KEY_GDN_LEGACY = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp1_gdn_legacy"

    def _gsm8k_with_acceptance(self, mocker, monkeypatch, engine, speculative_config, key, label):
        """gsm8k against the reference row the engine resolves, then the acceptance length.

        The acceptance length of the same run is held to the floor under ``key``.
        """
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)

        with LLM(
            self.MODEL_PATH,
            speculative_config=speculative_config,
            enable_iter_perf_stats=True,
            **engine,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            acceptance_length = compute_acceptance_length(llm)
            print(f"[AL] {label} acceptance_length = {acceptance_length:.3f}")
            assert_acceptance_length(key, acceptance_length)

    @skip_not_sm103
    def test_gsm8k(self, mocker, monkeypatch):
        """The text model without speculation, against the gsm8k anchor."""
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)

        with LLM(
            self.MODEL_PATH,
            **self.ENGINE,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_not_sm103
    def test_gsm8k_cutedsl_moe(self, mocker, monkeypatch):
        """The text model with the trunk's routed experts on CuteDSL, against the same gsm8k anchor.

        Same as ``test_gsm8k`` with ``moe_config.backend="CUTEDSL"``.
        """
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)
        engine = dict(self.ENGINE, moe_config=MoeConfig(backend="CUTEDSL"))

        with LLM(
            self.MODEL_PATH,
            **engine,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_not_sm103
    def test_gsm8k_bf16_ssm_state(self, mocker, monkeypatch):
        """The text model with the recurrent state stored in bf16, against the fp32 gsm8k anchor.

        ``mamba_ssm_cache_dtype="bfloat16"`` (what ``auto`` resolves to for this checkpoint) rounds
        the state to bf16 on every decode token and every prefill call, a lossy substitution
        for the checkpoint's fp32 state. It reads the same reference row as ``test_gsm8k``, so
        the rounding has to stay inside that row's tolerance. The decode step runs its
        recurrence in the batch-invariant form that a bf16 pool needs.
        """
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)
        engine = dict(self.ENGINE, kv_cache_config=self.KV_CACHE_BF16_STATE)

        with LLM(
            self.MODEL_PATH,
            **engine,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_not_sm103
    def test_mtp1_gsm8k_and_acceptance(self, mocker, monkeypatch):
        """MTP with one draft: gsm8k against the MTP reference, plus acceptance length.

        A miscomputed draft layer only makes generation slower (every draft is
        rejected, the text stays correct), so acceptance length is the only
        detector. The floor in references/acceptance_length.yaml is a collapse
        tripwire.
        """
        self._gsm8k_with_acceptance(
            mocker, monkeypatch, self.ENGINE, self.MTP1, self.ACCEPTANCE_KEY, "mtp1"
        )

    @skip_not_sm103
    def test_mtp1_fp8_kv_gsm8k_and_acceptance(self, mocker, monkeypatch):
        """MTP with one draft over an fp8 KV cache: the FP8 KV row of references/gsm8k.yaml.

        ``kv_cache_config.dtype="fp8"`` makes the engine resolve ``kv_cache_quant_algo`` FP8,
        which the reference lookup keys on: this case reads the FP8 row where
        ``test_mtp1_gsm8k_and_acceptance`` (no KV quantization) reads the bf16 MTP row. The
        draft layer runs over the same fp8 pool; its acceptance length has a floor of its own,
        since fp8 storage can cost drafts the bf16 cache would have kept.
        """
        engine = dict(self.ENGINE, kv_cache_config=self.KV_CACHE_FP8)
        self._gsm8k_with_acceptance(
            mocker, monkeypatch, engine, self.MTP1, self.ACCEPTANCE_KEY_FP8_KV, "mtp1 fp8 kv"
        )

    @skip_not_sm103
    def test_mtp3_gsm8k_and_acceptance(self, mocker, monkeypatch):
        """MTP with three drafts: the MTP row of references/gsm8k.yaml, and a length floor.

        Three drafts replay the one MTP module three times, with the verify step running four
        rows per request, a different shape from ``test_mtp1_gsm8k_and_acceptance`` on every
        attention and recurrent-state call of the verify step. Drafts after the first can be
        rejected while the first stays accepted, which holds the length near the one-draft
        value, so the floor has its own entry.
        """
        self._gsm8k_with_acceptance(
            mocker, monkeypatch, self.ENGINE, self.MTP3, self.ACCEPTANCE_KEY_MTP3, "mtp3"
        )

    @skip_not_sm103
    def test_mtp3_cutedsl_gsm8k_and_acceptance(self, mocker, monkeypatch):
        """MTP with three drafts over a trunk whose routed experts run on CuteDSL.

        The trunk's CuteDSL experts and the draft layer's FP8 block-scale experts run in one
        forward only here: ``test_gsm8k_cutedsl_moe`` has no draft layer and the other MTP
        cases run the trunk on trtllm-gen. Acceptance length detects a trunk hidden state the
        draft layer cannot use, which accuracy does not.
        """
        engine = dict(self.ENGINE, moe_config=MoeConfig(backend="CUTEDSL"))
        self._gsm8k_with_acceptance(
            mocker,
            monkeypatch,
            engine,
            self.MTP3,
            self.ACCEPTANCE_KEY_MTP3_CUTEDSL,
            "mtp3 cutedsl",
        )

    @skip_not_sm103
    def test_mtp5_gsm8k_and_acceptance(self, mocker, monkeypatch):
        """MTP with five drafts: the MTP row of references/gsm8k.yaml, and a length floor.

        Six verify rows per request, a width that no other case runs: the cached replay at
        ``T = 6`` and ``thop_attention`` at ``predicted_tokens_per_seq`` 6, with the all-layer
        commit from 16 requests on.
        """
        self._gsm8k_with_acceptance(
            mocker, monkeypatch, self.ENGINE, self.MTP5, self.ACCEPTANCE_KEY_MTP5, "mtp5"
        )

    @skip_not_sm103
    def test_mtp7_gsm8k_and_acceptance(self, mocker, monkeypatch):
        """MTP with seven drafts, the longest the target runs: the MTP row and a length floor.

        Eight verify rows per request is the cached replay kernel's ceiling (``T = 8``, history
        16): a slot that accepts every draft fills its history every other step, so the
        checkpoint commit, in-kernel below 16 requests and all-layer from there on, runs twice
        as often as at three drafts. The engine captures a CUDA graph for every batch size up
        to 32, so every captured step carries up to 256 verify rows.
        """
        self._gsm8k_with_acceptance(
            mocker, monkeypatch, self.ENGINE, self.MTP7, self.ACCEPTANCE_KEY_MTP7, "mtp7"
        )

    @skip_not_sm103
    def test_mtp3_bf16_ssm_state_gsm8k_and_acceptance(self, mocker, monkeypatch):
        """MTP with three drafts over a bf16 recurrent state: the MTP row, and a length floor.

        The cached replay rebuilds the state from a checkpoint that is now bf16 and rounds it
        once at every commit; drafts after the first are the ones a degraded state would lose,
        so the length floor has its own entry.
        """
        engine = dict(self.ENGINE, kv_cache_config=self.KV_CACHE_BF16_STATE)
        self._gsm8k_with_acceptance(
            mocker,
            monkeypatch,
            engine,
            self.MTP3,
            self.ACCEPTANCE_KEY_MTP3_BF16_STATE,
            "mtp3 bf16 state",
        )

    def _long_context_passkey(
        self, monkeypatch, speculative_config, engine, index_share, lengths, key_name, label
    ):
        """Pass key retrieval over prompts past the indexer's budget, at ``speculative_config``.

        Runs with the executor in this process to read the draft layer's op calls; the
        acceptance length comes from the requests' own metrics. Two identical requests in one
        batch must produce the same tokens, which fails if a request's shared selection leaks
        into another slot.
        """
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        reuse_calls = []
        reuse = target_modeling.qsa_mtp_reuse_topk

        def counting_reuse(*args, **kwargs):
            reuse_calls.append(1)
            return reuse(*args, **kwargs)

        monkeypatch.setattr(target_modeling, "qsa_mtp_reuse_topk", counting_reuse)
        engine = dict(
            engine,
            sparse_attention_config=QSASparseAttentionConfig(
                index_share_for_mtp_iteration=index_share
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        prompts = _passkey_prompts(tokenizer, lengths)
        # One prompt twice, far apart in the batch, so the two copies hold different slots.
        texts = [prompt for prompt, _ in prompts] + [prompts[len(prompts) // 2][0]]

        with LLM(
            self.MODEL_PATH,
            speculative_config=speculative_config,
            **engine,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            assert _target_core(llm).mtp_index_share == index_share
            outputs = llm.generate(
                texts,
                SamplingParams(
                    max_tokens=_PASSKEY_MAX_TOKENS, temperature=0, return_perf_metrics=True
                ),
                use_tqdm=False,
            )

        missed = [
            key
            for (_, key), output in zip(prompts, outputs)
            if key not in output.outputs[0].text[:_PASSKEY_MAX_TOKENS]
        ]
        assert not missed, f"pass keys not stated back: {missed} of {[key for _, key in prompts]}"
        twins = outputs[len(prompts) // 2], outputs[-1]
        assert twins[0].outputs[0].token_ids == twins[1].outputs[0].token_ids
        if index_share:
            assert reuse_calls, "IndexShare is on and the reuse op never ran"
        else:
            assert not reuse_calls, "IndexShare is off and the reuse op ran"
        acceptance_length = _acceptance_length_from_requests(
            outputs, speculative_config.max_draft_len
        )
        print(f"[AL] {label} acceptance_length = {acceptance_length:.3f}")
        assert_acceptance_length(getattr(self, key_name), acceptance_length)

    @pytest.mark.parametrize(
        "fp8_kv, index_share, key_name",
        [
            pytest.param(False, True, "ACCEPTANCE_KEY_LONG_CONTEXT", id="index_share"),
            pytest.param(True, False, "ACCEPTANCE_KEY_FP8_KV_LONG_CONTEXT", id="fp8_kv"),
        ],
    )
    @skip_not_sm103
    def test_mtp3_long_context_passkey(self, monkeypatch, fp8_kv, index_share, key_name):
        """Pass key retrieval over 3k-6k-token prompts, where the sparse attention path runs.

        The gsm8k cases never leave the dense path (their prompts are under the indexer's
        2048-token budget), so this is the only case that runs the target's QSA selection, the
        draft layer's selection and, with ``fp8_kv``, the fp8 sparse kernels end to end. A pass
        key stated back correctly needs the right blocks selected at every depth.

        ``index_share`` makes the draft loop reuse the selection of its first step; the
        acceptance length then must not fall below the floor taken from the same prompts
        with the draft layer selecting on every step, and the reuse op must have run.
        """
        engine = dict(self.ENGINE, kv_cache_config=self.KV_CACHE_FP8 if fp8_kv else self.KV_CACHE)
        self._long_context_passkey(
            monkeypatch,
            self.MTP3,
            engine,
            index_share,
            _PASSKEY_LENGTHS,
            key_name,
            f"mtp3 long context fp8_kv={fp8_kv}",
        )

    @skip_not_sm103
    def test_mtp7_long_context_passkey(self, monkeypatch):
        """Pass key retrieval over 3k-12k-token prompts with seven drafts and IndexShare on.

        Seven drafts run the sparse path with eight rows per request in the verify step and
        draft decode steps that reuse the first step's selection up to six positions past it,
        where the groups completed since the capture displace two of its ids. 12k tokens past
        the 8k default needs ``max_seq_len`` 16384.
        """
        engine = dict(self.ENGINE, max_seq_len=16384)
        self._long_context_passkey(
            monkeypatch,
            self.MTP7,
            engine,
            True,
            _PASSKEY_LENGTHS_LONG,
            "ACCEPTANCE_KEY_MTP7_LONG_CONTEXT",
            "mtp7 long context",
        )

    @skip_not_sm103
    def test_mtp7_bf16_ssm_state_long_context_passkey(self, monkeypatch):
        """``test_mtp7_long_context_passkey`` over a bf16 recurrent state.

        The state is rounded to bf16 at every cached-replay checkpoint commit, each time a
        slot's history of 16 accepted tokens fills, and once per prefill chunk, so the rounding
        adds up with the length of the prompt and of the answer; pass keys stated back over
        3k-12k tokens bound that drift.
        """
        engine = dict(self.ENGINE, max_seq_len=16384, kv_cache_config=self.KV_CACHE_BF16_STATE)
        self._long_context_passkey(
            monkeypatch,
            self.MTP7,
            engine,
            True,
            _PASSKEY_LENGTHS_LONG,
            "ACCEPTANCE_KEY_MTP7_BF16_STATE_LONG_CONTEXT",
            "mtp7 bf16 state long context",
        )

    @pytest.mark.parametrize(
        "forced, accepted_rows",
        [
            pytest.param("1e-9", 1, id="all_rejected"),
            pytest.param("7", 8, id="all_accepted"),
        ],
    )
    @skip_not_sm103
    def test_mtp7_forced_acceptance_extremes(self, monkeypatch, forced, accepted_rows):
        """Seven drafts with every draft rejected, then with every draft accepted, at batch 32.

        ``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` overrides the sampler's acceptance:
        a vanishing fraction accepts the target's own token and nothing else, ``7`` accepts
        the seven drafts as well. 32 requests over 2k-6k-token prompts run the sparse path,
        the all-layer replay commit and the CUDA graph at 32 x 8 = 256 verify rows with the
        most rows rolled back at every step (seven) and with none rolled back, a history that
        then fills to its end every second step. The acceptance length, read from the
        requests' own metrics, must be exactly the forced row count: the acceptance
        bookkeeping of the verify step moved by the number of rows forced, not by what the
        drafts say.

        The text under forced acceptance is not compared with anything: the drafts a forced
        step accepts are not the target's tokens, and a run's tokens depend on the verify
        step's row count through the GEMM and attention kernel selection, so a run with
        another draft length or none differs from this one even with every draft rejected.
        The rolled-back state itself is pinned bitwise by ``test_modeling_v2_spec_rollback``.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        prompts = [prompt for prompt, _ in _passkey_prompts(tokenizer, _FORCED_LENGTHS)][:32]
        env = {"TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": forced}

        with LLM(
            self.MODEL_PATH,
            speculative_config=self.MTP7,
            **self.ENGINE,
            **_require_target(monkeypatch, **env, **_VALIDATE),
        ) as llm:
            outputs = llm.generate(
                prompts,
                SamplingParams(
                    max_tokens=_FORCED_TOKENS,
                    temperature=0,
                    ignore_eos=True,
                    return_perf_metrics=True,
                ),
                use_tqdm=False,
            )

        assert [len(o.outputs[0].token_ids) for o in outputs] == [_FORCED_TOKENS] * len(prompts)
        acceptance_length = _acceptance_length_from_requests(outputs, self.MTP7.max_draft_len)
        print(f"[AL] mtp7 forced acceptance of {accepted_rows} rows: {acceptance_length:.3f}")
        assert acceptance_length == pytest.approx(accepted_rows, abs=0.05)

    @skip_not_sm103
    def test_mtp1_gdn_legacy_verify(self, monkeypatch):
        """MTP with one draft and ``TRTLLM_USE_GDN_REPLAY=0``: the per-step recurrent buffers.

        The engine runs the cached-replay verify step unless the variable is 0; this is the
        only case that binds and runs the other one. A few word problems keep it short: their
        answers say the recurrent state survives rejected drafts, and the acceptance length
        says the drafts still land.
        """
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{question} {_WORD_PROBLEM_SUFFIX}"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for question, _ in _WORD_PROBLEMS
        ]

        with LLM(
            self.MODEL_PATH,
            speculative_config=self.MTP1,
            **self.ENGINE,
            **_require_target(monkeypatch, TRTLLM_USE_GDN_REPLAY="0", **_VALIDATE),
        ) as llm:
            outputs = llm.generate(
                prompts,
                SamplingParams(max_tokens=384, temperature=0, return_perf_metrics=True),
                use_tqdm=False,
            )
            bound = _target_core(llm)._bound

        assert bound.replay is None and bound.int_ssm, (
            "TRTLLM_USE_GDN_REPLAY=0 but the target bound the cached-replay verify state"
        )
        wrong = [
            (question, answer, output.outputs[0].text[-120:])
            for (question, answer), output in zip(_WORD_PROBLEMS, outputs)
            if f"answer is {answer}" not in output.outputs[0].text.replace("**", "")
        ]
        assert not wrong, f"wrong or missing answers: {wrong}"
        acceptance_length = _acceptance_length_from_requests(outputs, self.MTP1.max_draft_len)
        print(f"[AL] mtp1 gdn legacy acceptance_length = {acceptance_length:.3f}")
        assert_acceptance_length(self.ACCEPTANCE_KEY_GDN_LEGACY, acceptance_length)

    @skip_not_sm103
    def test_mtp1_mmmu(self, monkeypatch):
        """The vision tower and mrope path under MTP, against the MTP row of references/mmmu.yaml.

        Block reuse is off here (on in the text cases above) because the vision path
        requires it.
        """
        engine = dict(self.ENGINE, kv_cache_config=self.KV_CACHE_MULTIMODAL)

        # MMMU adds <|endoftext|> to the stop token, as the built-in model's
        # MMMU test does.
        sampling_params = SamplingParams(
            max_tokens=MMMU.MAX_OUTPUT_LEN,
            truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
            stop="<|endoftext|>",
        )

        with LLM(
            self.MODEL_PATH,
            speculative_config=self.MTP1,
            **engine,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            task = MMMU(self.MODEL_NAME)
            task.evaluate(
                llm,
                sampling_params=sampling_params,
                extra_evaluator_kwargs=dict(chat_template_kwargs=dict(enable_thinking=False)),
            )

    @skip_not_sm103
    def test_mtp1_video(self, monkeypatch):
        """Synthetic videos through the engine: the only case that sends a video.

        Three videos of a moving square (frames built in memory, since the container has no video
        decoder) must each be answered with the right colour and direction. A request mixing an
        image and a video, in either order, must reach the forward: its ``mm_item_order``
        manifest has to survive the input processor, the request and the admission check, which
        refuses a mixed request without a usable one. A frame past the vision tower's patch
        bound (a 3072x3072 image is 192x192 = 36864 patches) must fail as that request's own
        error, and the request after it must still complete; ``max_seq_len`` is raised so that
        image reaches the admission check rather than the sequence length check. The order of
        the splice itself is pinned by ``test_modeling_v2_qwen4_exp_multimodal_embeds``.
        """
        engine = dict(self.ENGINE, kv_cache_config=self.KV_CACHE_MULTIMODAL, max_seq_len=16384)
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        processor = AutoProcessor.from_pretrained(
            self.MODEL_PATH, use_fast=True, trust_remote_code=True
        )
        greedy = SamplingParams(max_tokens=96, temperature=0)

        def ask(text, items):
            request = _media_request(tokenizer, processor, text, items)
            return llm.generate_async(request, sampling_params=greedy).result().outputs[0].text

        with LLM(
            self.MODEL_PATH,
            speculative_config=self.MTP1,
            **engine,
            **_require_target(monkeypatch, **_VALIDATE),
        ) as llm:
            for name, (_, _, colour, directions) in _VIDEOS.items():
                answer = ask(_VIDEO_QUESTION, [("video", _synthetic_video(name))]).lower()
                assert colour in answer and any(d in answer for d in directions), (name, answer)

            # Each label carries its own item's colour.
            image = _synthetic_image((250, 200, 0))
            for items, video_colour in (
                ([("video", _synthetic_video("blue_leftward")), ("image", image)], "blue"),
                ([("image", image), ("video", _synthetic_video("red_rightward"))], "red"),
            ):
                answer = ask(_MIXED_QUESTION, items).lower()
                assert re.search(r"image:\W*yellow", answer), answer
                assert re.search(rf"video:\W*{video_colour}", answer), answer

            with pytest.raises(RequestError, match="frames of 192x192 = 36864 patches"):
                ask(
                    "What shape and colour is in the image?",
                    [("image", _synthetic_image(_RED, 3072))],
                )
            answer = ask(_VIDEO_QUESTION, [("video", _synthetic_video("red_rightward"))]).lower()
            assert "red" in answer and "right" in answer, answer
