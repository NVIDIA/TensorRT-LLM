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
from pathlib import Path

_MODULE_PATH = Path(__file__).parents[3] / "integration" / "defs" / "accuracy" / "mixed_modality.py"
_SPEC = importlib.util.spec_from_file_location("mixed_modality", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
mixed_modality = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mixed_modality)

MixedModalityInputSample = mixed_modality.MixedModalityInputSample
MixedModalityOmniEvaluator = mixed_modality.MixedModalityOmniEvaluator
MixedModalitySample = mixed_modality.MixedModalitySample
_audio_wer = mixed_modality._audio_wer
_extract_choice_answer = mixed_modality._extract_choice_answer
_extract_response_sections = mixed_modality._extract_response_sections
_score_predictions = mixed_modality._score_predictions
select_mixed_modality_samples = mixed_modality.select_mixed_modality_samples


def _sample(
    sample_id: str = "mixed-0",
    image_keyword: str = "B",
    audio_keyword: str = "the speaker says democracy clearly",
    video_keyword: str = "C",
) -> MixedModalitySample:
    return MixedModalitySample(
        sample_id=sample_id,
        image=object(),
        audio=object(),
        video_path="/tmp/video.mp4",
        image_question="Which subject is shown?",
        video_question="What happens in the basketball clip?",
        expected_keywords={
            "image": image_keyword,
            "audio": audio_keyword,
            "video": video_keyword,
        },
        source_paths=("/tmp/image.jpg", "/tmp/audio.wav", "/tmp/video.mp4"),
    )


def test_parser_passes_three_labeled_sections():
    sections = _extract_response_sections(
        """
        image: B
        audio: the speaker says democracy clearly.
        video: C
        """
    )

    assert sections == {
        "image": "B",
        "audio": "the speaker says democracy clearly.",
        "video": "C",
    }


def test_missing_section_fails_sample():
    results = _score_predictions(
        ["image: B\nvideo: C"],
        [_sample()],
    )

    assert len(results) == 1
    assert not results[0].is_correct
    assert results[0].missing_sections == ("audio",)


def test_choice_extraction_accepts_common_answer_formats():
    assert _extract_choice_answer("Answer: (B)", "E") == "B"
    assert _extract_choice_answer("C. The man is upside down.", "D") == "C"
    assert _extract_choice_answer("The best option is d", "D") == "D"


def test_audio_wer_scores_near_transcripts():
    assert _audio_wer("the speaker says democracy", "the speaker says democracy clearly") == 20.0
    assert _audio_wer("unrelated words", "the speaker says democracy clearly") > 50.0


def test_score_requires_all_three_modalities_to_match():
    results = _score_predictions(
        [
            ("image: B\naudio: the speaker says democracy clearly\nvideo: Answer: C"),
            ("image: B\naudio: unrelated words\nvideo: Answer: D"),
        ],
        [
            _sample("mixed-0"),
            _sample("mixed-1"),
        ],
    )

    assert [result.is_correct for result in results] == [True, False]


def test_deterministic_sample_selection():
    image_samples = [
        MixedModalityInputSample(
            sample_id=f"image-{idx}",
            media=f"image-{idx}.jpg",
            question=f"image question {idx}",
            keyword=f"subject{idx}",
            source_paths=(f"image-{idx}.jpg",),
        )
        for idx in range(5)
    ]
    audio_samples = [
        MixedModalityInputSample(
            sample_id=f"audio-{idx}",
            media=f"audio-{idx}.wav",
            question=f"audio question {idx}",
            keyword=f"spoken{idx}",
            source_paths=(f"audio-{idx}.wav",),
        )
        for idx in range(5)
    ]
    video_samples = [
        MixedModalityInputSample(
            sample_id=f"video-{idx}",
            media=f"video-{idx}.mp4",
            question=f"video question {idx}",
            keyword=f"motion{idx}",
            source_paths=(f"video-{idx}.mp4",),
        )
        for idx in range(5)
    ]

    first = select_mixed_modality_samples(
        image_samples,
        audio_samples,
        video_samples,
        num_samples=3,
        random_seed=7,
    )
    second = select_mixed_modality_samples(
        image_samples,
        audio_samples,
        video_samples,
        num_samples=3,
        random_seed=7,
    )

    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert [sample.sample_id for sample in first] == [
        "image-4+audio-2+video-3",
        "image-0+audio-3+video-2",
        "image-3+audio-1+video-0",
    ]


def test_generate_outputs_waits_after_each_request(monkeypatch):
    class Future:
        def __init__(self, llm, output):
            self.llm = llm
            self.output = output

        def result(self):
            assert self.llm.inflight == 1
            self.llm.inflight = 0
            return self.output

    class FakeLlm:
        def __init__(self):
            self.inflight = 0
            self.requests = []

        def generate_async(self, request_input, sampling_params, streaming):
            assert self.inflight == 0
            self.inflight = 1
            self.requests.append((request_input, sampling_params, streaming))
            return Future(self, f"output-{request_input}")

    evaluator = MixedModalityOmniEvaluator(
        mmmu_dataset_path="/tmp/mmmu",
        voxpopuli_dataset_path="/tmp/voxpopuli",
        videomme_dataset_path="/tmp/videomme",
        num_samples=2,
    )
    monkeypatch.setattr(
        evaluator,
        "_make_input",
        lambda llm, sample, input_context, video_cache: sample.sample_id,
    )
    sampling_params = {"max_tokens": 1}
    llm = FakeLlm()

    outputs = evaluator._generate_outputs_serially(
        llm,
        [_sample("mixed-0"), _sample("mixed-1")],
        input_context=object(),
        sampling_params=sampling_params,
        streaming=False,
    )

    assert outputs == ["output-mixed-0", "output-mixed-1"]
    assert llm.requests == [
        ("mixed-0", sampling_params, False),
        ("mixed-1", sampling_params, False),
    ]
    assert llm.inflight == 0
