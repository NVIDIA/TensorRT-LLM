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

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[3] / "integration" / "defs" / "accuracy" / "mixed_modality.py"
_SPEC = importlib.util.spec_from_file_location("mixed_modality", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
mixed_modality = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mixed_modality)

MODALITY_AUDIO = mixed_modality.MODALITY_AUDIO
MODALITY_IMAGE = mixed_modality.MODALITY_IMAGE
MODALITY_VIDEO = mixed_modality.MODALITY_VIDEO
MixedModalityEvaluator = mixed_modality.MixedModalityEvaluator
MixedModalityInputSample = mixed_modality.MixedModalityInputSample
MixedModalitySample = mixed_modality.MixedModalitySample
MixedModalityTargetResult = mixed_modality.MixedModalityTargetResult
_audio_wer = mixed_modality._audio_wer
_assert_pure_baseline_not_degraded = mixed_modality._assert_pure_baseline_not_degraded
_extract_choice_answer = mixed_modality._extract_choice_answer
_format_mixed_modality_prompt = mixed_modality._format_mixed_modality_prompt
_make_pure_target_samples = mixed_modality._make_pure_target_samples
_score_target_predictions = mixed_modality._score_target_predictions
_summarize_paired_baseline = mixed_modality._summarize_paired_baseline
_summarize_target_results = mixed_modality._summarize_target_results
select_mixed_modality_samples = mixed_modality.select_mixed_modality_samples


def _input_sample(
    modality: str, keyword: str, question: str, idx: int = 0
) -> MixedModalityInputSample:
    suffix = {
        MODALITY_IMAGE: "jpg",
        MODALITY_AUDIO: "wav",
        MODALITY_VIDEO: "mp4",
    }[modality]
    return MixedModalityInputSample(
        sample_id=f"{modality}-{idx}",
        media=f"/tmp/{modality}-{idx}.{suffix}",
        question=question,
        keyword=keyword,
        source_paths=(f"/tmp/{modality}-{idx}.{suffix}",),
    )


def _sample(
    sample_id: str = "mixed-0",
    target_modality: str = MODALITY_IMAGE,
    active_modalities: tuple[str, ...] = (MODALITY_IMAGE, MODALITY_AUDIO, MODALITY_VIDEO),
    image_keyword: str = "B",
    audio_keyword: str = "the speaker says democracy clearly",
    video_keyword: str = "C",
    audio_question: str = "Transcribe the spoken audio exactly.",
) -> MixedModalitySample:
    item_specs = {
        MODALITY_IMAGE: (image_keyword, "Which subject is shown?"),
        MODALITY_AUDIO: (audio_keyword, audio_question),
        MODALITY_VIDEO: (video_keyword, "What happens in the basketball clip?"),
    }
    items = {
        modality: _input_sample(modality, item_specs[modality][0], item_specs[modality][1])
        for modality in active_modalities
    }
    return MixedModalitySample(
        sample_id=sample_id,
        items=items,
        target_modality=target_modality,
    )


def _input_samples_by_modality(
    active_modalities: tuple[str, ...], sample_count: int
) -> dict[str, list[MixedModalityInputSample]]:
    return {
        modality: [
            _input_sample(
                modality,
                keyword=f"{modality}-answer-{idx}",
                question=f"{modality} question {idx}",
                idx=idx,
            )
            for idx in range(sample_count)
        ]
        for modality in active_modalities
    }


def test_choice_extraction_accepts_common_answer_formats():
    assert _extract_choice_answer("Answer: (B)", "E") == "B"
    assert _extract_choice_answer("C. The man is upside down.", "D") == "C"
    assert _extract_choice_answer("The best option is d", "D") == "D"


def test_audio_wer_scores_near_transcripts():
    assert _audio_wer("the speaker says democracy", "the speaker says democracy clearly") == 20.0
    assert _audio_wer("unrelated words", "the speaker says democracy clearly") > 50.0


def test_target_prompt_only_asks_selected_query_modality():
    sample = _sample(
        target_modality=MODALITY_AUDIO,
        audio_question="Transcribe the committee hearing exactly.",
    )

    prompt = _format_mixed_modality_prompt(sample)

    assert "one image, one audio clip, and one video" in prompt
    assert "random distractors" in prompt
    assert "Answer only the audio question" in prompt
    assert "Transcribe the committee hearing exactly." in prompt
    assert sample.items[MODALITY_IMAGE].question not in prompt
    assert sample.items[MODALITY_VIDEO].question not in prompt


def test_image_video_prompt_omits_audio_for_qwen_vl_style_requests():
    sample = _sample(
        target_modality=MODALITY_VIDEO,
        active_modalities=(MODALITY_IMAGE, MODALITY_VIDEO),
    )

    prompt = _format_mixed_modality_prompt(sample)

    assert "one image and one video" in prompt
    assert "audio" not in prompt
    assert "Answer only the video question" in prompt
    assert sample.items[MODALITY_IMAGE].question not in prompt
    assert sample.items[MODALITY_VIDEO].question in prompt


def test_target_only_scoring_uses_selected_modality_only():
    results = _score_target_predictions(
        [
            "image: B\naudio: unrelated words\nvideo: Answer: D",
            "unrelated words",
            "video: Answer: C",
        ],
        [
            _sample("mixed-image", target_modality=MODALITY_IMAGE),
            _sample("mixed-audio", target_modality=MODALITY_AUDIO),
            _sample("mixed-video", target_modality=MODALITY_VIDEO),
        ],
    )

    summary = _summarize_target_results(results)

    assert [result.is_correct for result in results] == [True, False, True]
    assert results[0].distractor_modalities == (MODALITY_AUDIO, MODALITY_VIDEO)
    assert summary.target_accuracy == 100.0 * 2.0 / 3.0
    assert summary.correct_targets == 2
    assert summary.total_requests == 3
    assert summary.target_correct_counts == {
        MODALITY_IMAGE: 1,
        MODALITY_AUDIO: 0,
        MODALITY_VIDEO: 1,
    }
    assert summary.target_total_counts == {
        MODALITY_IMAGE: 1,
        MODALITY_AUDIO: 1,
        MODALITY_VIDEO: 1,
    }


def test_pure_target_samples_keep_only_selected_modality():
    sample = _sample("mixed-audio", target_modality=MODALITY_AUDIO)

    pure_samples = _make_pure_target_samples([sample])

    assert len(pure_samples) == 1
    pure_sample = pure_samples[0]
    assert pure_sample.sample_id == "mixed-audio|pure-audio"
    assert pure_sample.target_modality == MODALITY_AUDIO
    assert tuple(pure_sample.items) == (MODALITY_AUDIO,)
    assert pure_sample.items[MODALITY_AUDIO] == sample.items[MODALITY_AUDIO]

    prompt = _format_mixed_modality_prompt(pure_sample)

    assert "one audio clip" in prompt
    assert "No distractor media are included." in prompt
    assert "random distractors" not in prompt


def test_paired_pure_baseline_summarizes_and_validates_each_target():
    mixed_results = [
        MixedModalityTargetResult(
            sample_id="mixed-image",
            target_modality=MODALITY_IMAGE,
            prediction="B",
            is_correct=True,
            expected_keyword="B",
            distractor_modalities=(MODALITY_AUDIO, MODALITY_VIDEO),
        ),
        MixedModalityTargetResult(
            sample_id="mixed-audio",
            target_modality=MODALITY_AUDIO,
            prediction="wrong transcript",
            is_correct=False,
            expected_keyword="correct transcript",
            distractor_modalities=(MODALITY_IMAGE, MODALITY_VIDEO),
        ),
        MixedModalityTargetResult(
            sample_id="mixed-video",
            target_modality=MODALITY_VIDEO,
            prediction="C",
            is_correct=True,
            expected_keyword="C",
            distractor_modalities=(MODALITY_IMAGE, MODALITY_AUDIO),
        ),
    ]
    pure_results = [
        result._replace(
            sample_id=f"{result.sample_id}|pure-{result.target_modality}",
            distractor_modalities=(),
        )
        for result in mixed_results
    ]
    pure_results[1] = pure_results[1]._replace(
        prediction="correct transcript",
        is_correct=True,
    )
    target_modalities = (MODALITY_IMAGE, MODALITY_AUDIO, MODALITY_VIDEO)
    mixed_summary = _summarize_target_results(mixed_results, target_modalities)
    pure_summary = _summarize_target_results(pure_results, target_modalities)

    paired_summary = _summarize_paired_baseline(
        mixed_results=mixed_results,
        pure_results=pure_results,
        mixed_summary=mixed_summary,
        pure_summary=pure_summary,
        target_modalities=target_modalities,
    )

    assert paired_summary.mixed_minus_pure_accuracy == pytest.approx(-100.0 / 3.0)
    assert paired_summary.mixed_minus_pure_accuracy_by_target == {
        MODALITY_IMAGE: 0.0,
        MODALITY_AUDIO: -100.0,
        MODALITY_VIDEO: 0.0,
    }
    assert paired_summary.paired_counts_by_target[MODALITY_AUDIO] == {
        "sample_count": 1,
        "mixed_correct": 0,
        "pure_correct": 1,
        "both_correct": 0,
        "mixed_only": 0,
        "pure_only": 1,
        "both_wrong": 0,
    }
    _assert_pure_baseline_not_degraded(
        mixed_results=mixed_results,
        pure_results=pure_results,
        mixed_summary=mixed_summary,
        pure_summary=pure_summary,
        target_modalities=target_modalities,
        max_accuracy_drop=40.0,
        max_per_target_accuracy_drop=100.0,
    )
    with pytest.raises(AssertionError, match="for audio"):
        _assert_pure_baseline_not_degraded(
            mixed_results=mixed_results,
            pure_results=pure_results,
            mixed_summary=mixed_summary,
            pure_summary=pure_summary,
            target_modalities=target_modalities,
            max_accuracy_drop=40.0,
            max_per_target_accuracy_drop=10.0,
        )


def test_deterministic_sample_selection_for_omni_modalities():
    input_samples = _input_samples_by_modality((MODALITY_IMAGE, MODALITY_AUDIO, MODALITY_VIDEO), 5)

    first = select_mixed_modality_samples(
        input_samples,
        num_samples=3,
        random_seed=7,
        active_modalities=(MODALITY_IMAGE, MODALITY_AUDIO, MODALITY_VIDEO),
    )
    second = select_mixed_modality_samples(
        input_samples,
        num_samples=3,
        random_seed=7,
        active_modalities=(MODALITY_IMAGE, MODALITY_AUDIO, MODALITY_VIDEO),
    )

    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert [sample.target_modality for sample in first] == [
        sample.target_modality for sample in second
    ]
    assert Counter(sample.target_modality for sample in first) == Counter(
        {MODALITY_IMAGE: 1, MODALITY_AUDIO: 1, MODALITY_VIDEO: 1}
    )
    assert all(
        tuple(sample.items) == (MODALITY_IMAGE, MODALITY_AUDIO, MODALITY_VIDEO) for sample in first
    )


def test_deterministic_sample_selection_for_image_video_modalities():
    input_samples = _input_samples_by_modality((MODALITY_IMAGE, MODALITY_VIDEO), 6)

    samples = select_mixed_modality_samples(
        input_samples,
        num_samples=4,
        random_seed=11,
        active_modalities=(MODALITY_IMAGE, MODALITY_VIDEO),
    )

    assert Counter(sample.target_modality for sample in samples) == Counter(
        {MODALITY_IMAGE: 2, MODALITY_VIDEO: 2}
    )
    assert all(tuple(sample.items) == (MODALITY_IMAGE, MODALITY_VIDEO) for sample in samples)
    assert all(MODALITY_AUDIO not in sample.items for sample in samples)


def _dataset_paths(active_modalities: tuple[str, ...]) -> dict[str, str]:
    return {modality: f"/tmp/{modality}" for modality in active_modalities}


@pytest.mark.parametrize(
    "num_samples, match",
    [
        pytest.param(0, "must be positive", id="zero"),
        pytest.param(-3, "must be positive", id="negative"),
        pytest.param(1, "at least the number of target", id="fewer_than_targets"),
    ],
)
def test_evaluator_rejects_invalid_num_samples(num_samples, match):
    """num_samples must be validated fail-fast in the evaluator constructor."""
    active = (MODALITY_IMAGE, MODALITY_AUDIO, MODALITY_VIDEO)
    with pytest.raises(ValueError, match=match):
        MixedModalityEvaluator(
            modality_dataset_paths=_dataset_paths(active),
            active_modalities=active,
            target_modalities=active,
            num_samples=num_samples,
        )


@pytest.mark.parametrize(
    "num_samples, match",
    [
        pytest.param(0, "must be positive", id="zero"),
        pytest.param(-1, "must be positive", id="negative"),
        pytest.param(1, "at least the number of target", id="fewer_than_targets"),
    ],
)
def test_select_samples_rejects_invalid_num_samples(num_samples, match):
    """select_mixed_modality_samples shares the same fail-fast num_samples gate."""
    input_samples = _input_samples_by_modality((MODALITY_IMAGE, MODALITY_VIDEO), 4)
    with pytest.raises(ValueError, match=match):
        select_mixed_modality_samples(
            input_samples,
            num_samples=num_samples,
            random_seed=0,
            active_modalities=(MODALITY_IMAGE, MODALITY_VIDEO),
            target_modalities=(MODALITY_IMAGE, MODALITY_VIDEO),
        )


def test_evaluator_empty_target_modalities_raises_not_defaults():
    """An explicit empty target tuple must reach validation, not silently default.

    Truthiness defaulting (`target_modalities or self.active_modalities`) would
    swallow `()` and substitute the active set; the explicit-None check keeps an
    empty tuple flowing into `_normalize_modalities`, which rejects it.
    """
    active = (MODALITY_IMAGE, MODALITY_VIDEO)
    with pytest.raises(ValueError, match="at least 1 modality"):
        MixedModalityEvaluator(
            modality_dataset_paths=_dataset_paths(active),
            active_modalities=active,
            target_modalities=(),
        )


def test_select_samples_empty_target_modalities_raises_not_defaults():
    """select_mixed_modality_samples also rejects an explicit empty target tuple."""
    input_samples = _input_samples_by_modality((MODALITY_IMAGE, MODALITY_VIDEO), 4)
    with pytest.raises(ValueError, match="at least 1 modality"):
        select_mixed_modality_samples(
            input_samples,
            num_samples=2,
            random_seed=0,
            active_modalities=(MODALITY_IMAGE, MODALITY_VIDEO),
            target_modalities=(),
        )
