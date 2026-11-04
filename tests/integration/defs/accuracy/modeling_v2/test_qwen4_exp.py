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
import time

import pytest
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as target_modeling,
)
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner
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
from ._env import scoped_env_overrides

# The target is certified on sm_103 only.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="modeling_v2 targets in this batch are certified on sm_103 only"
)

# Check the target's step contract on the first forward, against metadata the
# engine prepared; the flag has to reach the worker rank.
_VALIDATE = {MODELING_V2_VALIDATE_ENV: "1"}


def _target_core(llm: LLM):
    """The target's core module.

    Reachable only with the executor in this process (``TLLM_WORKER_USE_SINGLE_PROCESS=1``),
    where the iteration stats that ``compute_acceptance_length`` reads are not collected.
    """
    return llm._executor.engine.model_engine.model.model


def _count_prefill_graph_replays(monkeypatch: pytest.MonkeyPatch) -> dict[int, int]:
    """Replays of the breakable prefill graph by bucket, filled as the engine runs.

    Needs the executor in this process. Without it a case that sets
    ``prefill_cuda_graph_backend="breakable"`` could pass on the eager path.
    """
    replays: dict[int, int] = {}
    replay = BreakableCUDAGraphRunner.replay

    def counting_replay(self, num_tokens):
        replays[num_tokens] = replays.get(num_tokens, 0) + 1
        return replay(self, num_tokens)

    monkeypatch.setattr(BreakableCUDAGraphRunner, "replay", counting_replay)
    return replays


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
    # `mamba_ssm_dtype: float32` (the engine default is bfloat16); the NVFP4- and fp8-KV cases
    # set it to bfloat16, what `auto` resolves to for this checkpoint. `dtype` is left at the
    # checkpoint's bf16 KV, which is what the anchors were measured at; only the fp8 and NVFP4
    # KV cases set it.
    KV_CACHE = KvCacheConfig(
        free_gpu_memory_fraction=0.6,
        enable_block_reuse=True,
        mamba_ssm_cache_dtype="float32",
    )
    KV_CACHE_FP8_BF16_STATE = KvCacheConfig(
        dtype="fp8",
        free_gpu_memory_fraction=0.6,
        enable_block_reuse=True,
        mamba_ssm_cache_dtype="bfloat16",
    )
    # Block reuse is on, which (with chunked prefill and the MTP drafts) is what lets the
    # attention op read an NVFP4 pool: its kernels exist for the paged-context path only.
    KV_CACHE_NVFP4_BF16_STATE = KvCacheConfig(
        dtype="nvfp4",
        free_gpu_memory_fraction=0.6,
        enable_block_reuse=True,
        mamba_ssm_cache_dtype="bfloat16",
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

    # The breakable prefill CUDA graph is opt-in; these buckets cover the prompts of
    # `test_breakable_prefill_matches_eager`.
    BREAKABLE_PREFILL = dict(
        prefill_cuda_graph_backend="breakable",
        prefill_capture_num_tokens=[128, 256, 512, 1024, 2048, 3072],
    )
    ENGINE_BREAKABLE = dict(ENGINE, **BREAKABLE_PREFILL)

    # One-model MTP-Eagle with the checkpoint's single MTP module replayed
    # `max_draft_len` times.
    MTP1 = MTPDecodingConfig(max_draft_len=1)
    MTP3 = MTPDecodingConfig(max_draft_len=3)
    MTP7 = MTPDecodingConfig(max_draft_len=7)

    # Acceptance-length anchors in references/acceptance_length.yaml, keyed by
    # target and variant.
    ACCEPTANCE_KEY = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp1"
    ACCEPTANCE_KEY_LONG_CONTEXT = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp3_long_context"
    ACCEPTANCE_KEY_OPT_IN = "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp3_cutedsl_fp8_kv_bf16_state"
    ACCEPTANCE_KEY_OPT_IN_LONG_CONTEXT = (
        "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp3_cutedsl_fp8_kv_bf16_state_long_context"
    )
    ACCEPTANCE_KEY_MTP7_LONG_CONTEXT = (
        "ModelingV2Qwen38FlashNextNvfp4Sm103Tp1::mtp7_nvfp4_kv_bf16_state_long_context"
    )

    def _gsm8k_and_acceptance(self, mocker, llm, key, label):
        """gsm8k on ``llm``, then the same run's acceptance length against ``key``.

        gsm8k is read against the reference row the engine resolves; the acceptance length comes
        from the engine's iteration statistics, so the engine needs ``enable_iter_perf_stats``.
        """
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)
        GSM8K(self.MODEL_NAME).evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
        acceptance_length = compute_acceptance_length(llm)
        print(f"[AL] {label} acceptance_length = {acceptance_length:.3f}")
        assert_acceptance_length(key, acceptance_length)

    def _passkeys(self, llm, lengths, draft_len, key, label):
        """Pass key retrieval on ``llm`` over prompts past the indexer's budget.

        Every key must be stated back, and two identical requests far apart in one batch
        (different slots) must give the same tokens, which fails if a request's selection leaks
        into another slot. The acceptance length, from the requests' own metrics, is held to the
        floor under ``key``.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        prompts = _passkey_prompts(tokenizer, lengths)
        texts = [prompt for prompt, _ in prompts] + [prompts[len(prompts) // 2][0]]
        outputs = llm.generate(
            texts,
            SamplingParams(max_tokens=_PASSKEY_MAX_TOKENS, temperature=0, return_perf_metrics=True),
            use_tqdm=False,
        )
        missed = [
            k
            for (_, k), output in zip(prompts, outputs)
            if k not in output.outputs[0].text[:_PASSKEY_MAX_TOKENS]
        ]
        assert not missed, f"pass keys not stated back: {missed} of {[k for _, k in prompts]}"
        twins = outputs[len(prompts) // 2], outputs[-1]
        assert twins[0].outputs[0].token_ids == twins[1].outputs[0].token_ids
        acceptance_length = _acceptance_length_from_requests(outputs, draft_len)
        print(f"[AL] {label} acceptance_length = {acceptance_length:.3f}")
        assert_acceptance_length(key, acceptance_length)

    def _mmmu(self, llm):
        """MMMU on ``llm`` against the reference row the engine resolves."""
        # MMMU adds <|endoftext|> to the stop token, as the built-in model's
        # MMMU test does.
        sampling_params = SamplingParams(
            max_tokens=MMMU.MAX_OUTPUT_LEN,
            truncate_prompt_tokens=MMMU.MAX_INPUT_LEN,
            stop="<|endoftext|>",
        )
        MMMU(self.MODEL_NAME).evaluate(
            llm,
            sampling_params=sampling_params,
            extra_evaluator_kwargs=dict(chat_template_kwargs=dict(enable_thinking=False)),
        )

    def _video_checks(self, llm):
        """Synthetic videos through ``llm``.

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
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        processor = AutoProcessor.from_pretrained(
            self.MODEL_PATH, use_fast=True, trust_remote_code=True
        )
        greedy = SamplingParams(max_tokens=96, temperature=0)

        def ask(text, items):
            request = _media_request(tokenizer, processor, text, items)
            return llm.generate_async(request, sampling_params=greedy).result().outputs[0].text

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

    @skip_not_sm103
    def test_gsm8k(self, mocker):
        """The text model without speculation, against the gsm8k anchor."""
        mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", self.GSM8K_MAX_OUTPUT_LEN)

        with (
            scoped_env_overrides(_VALIDATE) as rank_env,
            LLM(
                self.MODEL_PATH,
                **self.ENGINE,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm, extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_not_sm103
    def test_mtp1_gsm8k_mmmu_and_video(self, mocker):
        """gsm8k with its acceptance-length floor, MMMU and synthetic videos on one MTP engine.

        - gsm8k against the MTP row of references/gsm8k.yaml. A miscomputed draft layer only
          makes generation slower (every draft rejected, the text still correct), so the
          acceptance length of the same run is the only detector; its floor is a collapse
          tripwire.
        - MMMU against the MTP row of references/mmmu.yaml: the vision tower and the mrope path.
        - Videos (`_video_checks`): the only case that sends a video.

        Block reuse is off, as the vision path requires; it does not change the text numerics.
        ``max_seq_len`` is raised for the video checks' oversized image.
        """
        engine = dict(self.ENGINE, kv_cache_config=self.KV_CACHE_MULTIMODAL, max_seq_len=16384)
        with (
            scoped_env_overrides(_VALIDATE) as rank_env,
            LLM(
                self.MODEL_PATH,
                speculative_config=self.MTP1,
                enable_iter_perf_stats=True,
                **engine,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            self._gsm8k_and_acceptance(mocker, llm, self.ACCEPTANCE_KEY, "mtp1")
            self._mmmu(llm)
            self._video_checks(llm)

    @skip_not_sm103
    def test_mtp3_cutedsl_fp8_kv_bf16_state(self, mocker):
        """gsm8k and 3k-6k-token pass keys with three drafts over the opt-in choices.

        One engine runs gsm8k with an acceptance-length floor, then the pass keys. The trunk's
        routed experts run on CuteDSL next to the draft layer's FP8 block-scale experts; the
        KV cache is FP8 (unit scales) and the recurrent state bf16 (what ``auto``
        resolves to for this checkpoint), rounded at every decode token, prefill call and
        cached-replay commit. gsm8k reads the FP8-KV MTP row of references/gsm8k.yaml; its
        prompts stay on the dense attention path. The pass keys run the FP8 sparse store and
        GQA kernels of the target and of the draft layer (IndexShare off, so the draft layer
        selects on every step). Three drafts give the verify step four rows per request.
        Acceptance length detects a trunk hidden state or a drifted state the draft layer
        cannot use, which accuracy does not.
        """
        engine = dict(
            self.ENGINE,
            moe_config=MoeConfig(backend="CUTEDSL"),
            kv_cache_config=self.KV_CACHE_FP8_BF16_STATE,
            sparse_attention_config=QSASparseAttentionConfig(index_share_for_mtp_iteration=False),
        )
        with (
            scoped_env_overrides(_VALIDATE) as rank_env,
            LLM(
                self.MODEL_PATH,
                speculative_config=self.MTP3,
                enable_iter_perf_stats=True,
                **engine,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            self._gsm8k_and_acceptance(mocker, llm, self.ACCEPTANCE_KEY_OPT_IN, "mtp3 opt-in")
            self._passkeys(
                llm,
                _PASSKEY_LENGTHS,
                self.MTP3.max_draft_len,
                self.ACCEPTANCE_KEY_OPT_IN_LONG_CONTEXT,
                "mtp3 opt-in long context",
            )

    def _long_context_passkey(
        self, monkeypatch, speculative_config, engine, index_share, lengths, key, label
    ):
        """`_passkeys` with the executor in this process, checking the op calls too.

        The draft layer's IndexShare reuse op must run exactly when IndexShare is on, and no
        prefill graph may replay.
        """
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        reuse_calls = []
        reuse = target_modeling.qsa_mtp_reuse_topk

        def counting_reuse(*args, **kwargs):
            reuse_calls.append(1)
            return reuse(*args, **kwargs)

        monkeypatch.setattr(target_modeling, "qsa_mtp_reuse_topk", counting_reuse)
        replays = _count_prefill_graph_replays(monkeypatch)
        engine = dict(
            engine,
            sparse_attention_config=QSASparseAttentionConfig(
                index_share_for_mtp_iteration=index_share
            ),
        )
        with (
            scoped_env_overrides(_VALIDATE) as rank_env,
            LLM(
                self.MODEL_PATH,
                speculative_config=speculative_config,
                **engine,
                modeling_v2="require",
                env_overrides=rank_env,
            ) as llm,
        ):
            assert _target_core(llm).mtp_index_share == index_share
            self._passkeys(llm, lengths, speculative_config.max_draft_len, key, label)
        assert not replays, replays
        if index_share:
            assert reuse_calls, "IndexShare is on and the reuse op never ran"
        else:
            assert not reuse_calls, "IndexShare is off and the reuse op ran"

    @skip_not_sm103
    def test_mtp3_long_context_passkey(self, monkeypatch):
        """Pass key retrieval over 3k-6k-token prompts with three drafts and IndexShare.

        The KV cache is bf16; the prompts are long enough to run the sparse attention path. The
        gsm8k prompts never leave the dense path (they are under the indexer's 2048-token
        budget), so the pass-key cases are the ones that run the target's QSA selection and the
        draft layer's end to end; a pass key stated back correctly needs the right blocks
        selected at every depth. IndexShare makes the draft loop reuse the selection of its
        first step: the acceptance length must not fall below the floor taken from the same
        prompts with the draft layer selecting on every step, and the reuse op must have run.
        """
        self._long_context_passkey(
            monkeypatch,
            self.MTP3,
            self.ENGINE,
            True,
            _PASSKEY_LENGTHS,
            self.ACCEPTANCE_KEY_LONG_CONTEXT,
            "mtp3 long context",
        )

    @skip_not_sm103
    def test_breakable_prefill_matches_eager(self, monkeypatch):
        """Greedy continuations of the same prompts, with the breakable prefill graph and without.

        MTP with three drafts, a 3072-token chunk budget, and prompts of a natural document
        cut at the edges of the buckets 128 and 1024 (one token under, at, one over), at the
        top bucket 3072 and one past it, and a 5000-token prompt prefilled in two chunks. A
        batch of word problems and a 5000-token prompt then prefills while the earlier ones
        decode, so replayed steps carry generation rows beside the context rows.

        A prompt that fills its bucket runs the same kernels on the same shapes as the eager
        step, so its tokens must be identical. A padded prompt changes the row count every
        GEMM sees, which moves the logits by what a different batch size does (about 0.2 nats
        at most here); it must still agree on its first four tokens. The same engine given
        the same prompts twice must repeat itself exactly. Replays are counted, since a
        replay that never happens passes everything above.
        """
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        replays = _count_prefill_graph_replays(monkeypatch)
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        document = tokenizer.encode(_passkey_prompts(tokenizer, (6000,))[0][0])
        lengths = (127, 128, 129, 1023, 1024, 1025, 3072, 3073, 5000)
        filled = (128, 1024, 3072)
        # Word problems, answered in full, and one prompt of five thousand tokens prefilled in
        # chunks while they decode. A word problem's first token can be a near tie between two
        # openings, so its answer is checked, not its tokens.
        problems = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{question} {_WORD_PROBLEM_SUFFIX}"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for question, _ in _WORD_PROBLEMS
        ]
        greedy = SamplingParams(max_tokens=24, temperature=0)
        answering = SamplingParams(max_tokens=384, temperature=0)

        def run(engine):
            with (
                scoped_env_overrides(_VALIDATE) as rank_env,
                LLM(
                    self.MODEL_PATH,
                    speculative_config=self.MTP3,
                    **engine,
                    modeling_v2="require",
                    env_overrides=rank_env,
                ) as llm,
            ):
                single = [
                    llm.generate({"prompt_token_ids": document[:n]}, greedy, use_tqdm=False)
                    .outputs[0]
                    .token_ids
                    for n in lengths
                ]
                again = [
                    llm.generate({"prompt_token_ids": document[:n]}, greedy, use_tqdm=False)
                    .outputs[0]
                    .token_ids
                    for n in lengths
                ]
                # The word problems are decoding when the long prompt arrives: their short
                # prefills are done well inside the pause, and a step-by-step answer is
                # still being written after it.
                futures = [llm.generate_async(prompt, answering) for prompt in problems]
                time.sleep(0.2)
                assert not all(future.finished for future in futures), (
                    "every word problem finished before the long prompt arrived"
                )
                long_future = llm.generate_async({"prompt_token_ids": document[:5000]}, greedy)
                answers = [future.result().outputs[0].text for future in futures]
                long_tokens = long_future.result().outputs[0].token_ids
            return single, again, answers, long_tokens

        single, again, answers, long_tokens = run(dict(self.ENGINE_BREAKABLE, max_num_tokens=3072))
        assert {128, 256, 1024, 2048, 3072} <= set(replays), replays
        assert single == again, "the same engine repeated itself with a different answer"

        replays.clear()
        eager_single, _, eager_answers, eager_long_tokens = run(
            dict(self.ENGINE, max_num_tokens=3072)
        )
        assert not replays, replays

        for n, got, want in zip(lengths, single, eager_single):
            head = len(want) if n in filled else 4
            assert got[:head] == want[:head], (n, got, want)
        for texts in (answers, eager_answers):
            wrong = [
                (question, answer, text[-120:])
                for (question, answer), text in zip(_WORD_PROBLEMS, texts)
                if f"answer is {answer}" not in text.replace("**", "")
            ]
            assert not wrong, f"wrong or missing answers: {wrong}"
        assert long_tokens[:4] == eager_long_tokens[:4], (long_tokens, eager_long_tokens)

    @skip_not_sm103
    def test_mtp7_long_context_passkey(self, monkeypatch):
        """Pass key retrieval over 3k-12k-token prompts with seven drafts and IndexShare.

        The KV cache is NVFP4 and the recurrent state bf16. Seven drafts run the sparse path
        with eight rows per request in the verify step, the cached replay kernel's ceiling
        (``T = 8``), and draft decode steps that reuse the first step's selection up to six
        positions past it, where the groups completed since the capture displace two of its
        ids. The NVFP4 pool runs the target's and the draft layer's NVFP4 sparse store and GQA
        kernels, and the context rows' dequantized scratch. The bf16 state is rounded at every
        cached-replay checkpoint commit and once per prefill chunk, so the rounding adds up
        with the length of the prompt and of the answer; pass keys stated back over 3k-12k
        tokens bound that drift. 12k tokens past the 8k default needs ``max_seq_len`` 16384.
        """
        engine = dict(
            self.ENGINE, max_seq_len=16384, kv_cache_config=self.KV_CACHE_NVFP4_BF16_STATE
        )
        self._long_context_passkey(
            monkeypatch,
            self.MTP7,
            engine,
            True,
            _PASSKEY_LENGTHS_LONG,
            self.ACCEPTANCE_KEY_MTP7_LONG_CONTEXT,
            "mtp7 nvfp4 kv bf16 state long context",
        )
