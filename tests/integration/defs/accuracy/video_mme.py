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
import copy
import json
import re
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

from tqdm import tqdm

import tensorrt_llm.profiler as profiler
from tensorrt_llm.evaluate.interface import (
    Evaluator,
    dump_inference_results,
    get_chat_template_kwargs,
    get_model_context,
)
from tensorrt_llm.inputs import (
    ConversationMessage,
    MultimodalData,
    MultimodalDataTracker,
    add_multimodal_placeholders,
    load_video,
)
from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.inputs.utils import _resolve_content_format, resolve_hf_chat_template
from tensorrt_llm.inputs.utils import apply_chat_template as trtllm_apply_chat_template
from tensorrt_llm.llmapi import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams

PROMPT_PREAMBLE = (
    "Answer the question based on the video. Select the best option. "
    "Respond with only the option letter (A, B, C, or D)."
)
ANNOTATIONS_FILE = "annotations.jsonl"


class VideoQASample(NamedTuple):
    """A single multiple-choice video QA sample."""

    sample_id: str
    video_path: str
    question: str
    options: tuple[str, ...]
    answer: str


class VideoQASampleResult(NamedTuple):
    """Per-sample scoring result for MCQ video QA."""

    sample_id: str
    prediction: str
    predicted_letter: Optional[str]
    reference: str

    @property
    def is_correct(self) -> bool:
        return self.predicted_letter == self.reference


class _VideoInputContext(NamedTuple):
    """Per-LLM context needed to render chat-templated video prompts."""

    model_type: str
    processor: Any
    content_format: ContentFormat
    chat_template_kwargs: dict[str, Any]


class VideoMME(Evaluator):
    """Evaluator for a local Video-MME-style multiple-choice video QA shard."""

    def __init__(
        self,
        dataset_path: str,
        num_samples: Optional[int] = None,
        random_seed: int = 0,
        num_frames: int = 8,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            random_seed=random_seed,
            apply_chat_template=True,
            chat_template_kwargs=chat_template_kwargs,
            output_dir=output_dir,
        )
        self.dataset_path = Path(dataset_path)
        self.annotations_path = self.dataset_path / ANNOTATIONS_FILE
        self.num_samples = num_samples
        self.num_frames = num_frames

    def generate_samples(self) -> Iterable[tuple]:
        # Required by the abstract base, but `evaluate()` is fully overridden so
        # the base-class flow that would call `generate_samples` is never reached.
        raise NotImplementedError

    def compute_score(
        self,
        outputs: list[RequestOutput],
        references: list[str],
        samples: list[VideoQASample],
    ) -> float:
        predictions = [output.outputs[0].text for output in outputs]
        sample_results = _score_video_qa_predictions(predictions, references, samples)

        correct = sum(result.is_correct for result in sample_results)
        accuracy = 100.0 * correct / len(sample_results) if sample_results else 0.0
        logger.info(f"VideoMME accuracy: {accuracy:.2f} ({correct}/{len(sample_results)})")
        if correct != len(sample_results):
            logger.info(f"VideoMME mismatches:\n{_format_mismatch_report(sample_results)}")
        return accuracy

    def evaluate(
        self,
        llm: Any,
        sampling_params: Optional[SamplingParams] = None,
        streaming: bool = False,
    ) -> float:
        profiler.start("trtllm exec")
        input_context = self._make_input_context(llm)
        samples = list(tqdm(self._iter_samples(), desc="Loading samples", total=self.num_samples))
        video_cache: dict[str, Any] = {}
        inputs = [
            self._make_input(llm, sample, input_context, video_cache)
            for sample in tqdm(samples, desc="Loading inputs")
        ]

        futures = []
        for request_input in tqdm(inputs, desc="Submitting requests"):
            params = (
                copy.deepcopy(sampling_params) if sampling_params is not None else SamplingParams()
            )
            futures.append(
                llm.generate_async(
                    request_input,
                    sampling_params=params,
                    streaming=streaming,
                )
            )
        outputs = [future.result() for future in tqdm(futures, desc="Fetching responses")]

        if self.output_dir:
            dump_inference_results(self.output_dir, outputs, getattr(llm, "tokenizer", None))

        profiler.stop("trtllm exec")
        elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
        logger.info(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")
        profiler.reset("trtllm exec")

        references = [sample.answer for sample in samples]
        return self.compute_score(outputs, references, samples)

    def _iter_samples(self) -> Iterable[VideoQASample]:
        with open(self.annotations_path, "r", encoding="utf-8") as annotations_file:
            for idx, line in enumerate(annotations_file):
                if self.num_samples is not None and idx >= self.num_samples:
                    break
                row = json.loads(line)
                video_path = Path(str(row["video_path"]))
                if not video_path.is_absolute():
                    video_path = self.dataset_path / video_path
                yield VideoQASample(
                    sample_id=str(row.get("sample_id", idx)),
                    video_path=str(video_path),
                    question=str(row["question"]),
                    options=tuple(str(option) for option in row["options"]),
                    answer=_normalize_answer(str(row["answer"])),
                )

    def _make_input_context(self, llm: Any) -> _VideoInputContext:
        _, model_type = get_model_context(llm)
        processor = getattr(getattr(llm, "input_processor", None), "processor", None)
        hf_chat_template = resolve_hf_chat_template(llm.tokenizer, processor, None, None)
        return _VideoInputContext(
            model_type=model_type,
            processor=processor,
            content_format=_resolve_content_format(model_type, hf_chat_template),
            chat_template_kwargs=get_chat_template_kwargs(
                processor or llm.tokenizer, self.chat_template_kwargs
            ),
        )

    def _make_input(
        self,
        llm: Any,
        sample: VideoQASample,
        input_context: _VideoInputContext,
        video_cache: dict[str, Any],
    ) -> dict[str, Any]:
        mm_data_tracker = MultimodalDataTracker(input_context.model_type)
        video = video_cache.get(sample.video_path)
        if video is None:
            video = load_video(sample.video_path, num_frames=self.num_frames)
            video_cache[sample.video_path] = video

        prompt_text = _format_video_qa_prompt(sample)
        conv = ConversationMessage(
            role="user",
            content=prompt_text,
            media=[
                MultimodalData(
                    modality="video",
                    data=video,
                    is_embedding=False,
                )
            ],
        )
        for mdata in conv["media"]:
            mm_data_tracker.add_data(
                mdata["modality"],
                mdata["data"],
                is_embedding=mdata["is_embedding"],
            )

        mm_placeholder_counts = mm_data_tracker.placeholder_counts()
        if mm_placeholder_counts and input_context.content_format != ContentFormat.OPENAI:
            conv["content"] = add_multimodal_placeholders(
                input_context.model_type, conv["content"], mm_placeholder_counts
            )

        prompt = trtllm_apply_chat_template(
            model_type=input_context.model_type,
            tokenizer=llm.tokenizer,
            processor=input_context.processor,
            conversation=[conv],
            add_generation_prompt=True,
            mm_placeholder_counts=[mm_placeholder_counts],
            chat_template_kwargs=input_context.chat_template_kwargs,
        )

        request_input = {"prompt": prompt}
        multi_modal_data, _ = mm_data_tracker.retrieve_all_sync()
        if multi_modal_data:
            request_input["multi_modal_data"] = multi_modal_data
        return request_input


def _format_video_qa_prompt(sample: VideoQASample) -> str:
    option_lines = "\n".join(
        _normalize_option(option, idx) for idx, option in enumerate(sample.options)
    )
    return f"{PROMPT_PREAMBLE}\n\nQuestion: {sample.question}\nOptions:\n{option_lines}\nAnswer:"


def _normalize_option(option: str, idx: int) -> str:
    option = option.strip()
    if re.match(r"^[A-D][.)]\s+", option, flags=re.IGNORECASE):
        return option
    label = chr(ord("A") + idx)
    return f"{label}. {option}"


def _normalize_answer(answer: str) -> str:
    answer = answer.strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"Expected answer to be one of A, B, C, D; got {answer!r}.")
    return answer


def extract_choice_answer(text: str) -> Optional[str]:
    """Extract an A-D answer from model output."""
    match = re.search(r"\b([A-D])\b", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _score_video_qa_predictions(
    predictions: list[str],
    references: list[str],
    samples: list[VideoQASample],
) -> list[VideoQASampleResult]:
    sample_results = []
    for sample, prediction, reference in zip(samples, predictions, references, strict=True):
        sample_results.append(
            VideoQASampleResult(
                sample_id=sample.sample_id,
                prediction=prediction,
                predicted_letter=extract_choice_answer(prediction),
                reference=reference,
            )
        )
    return sample_results


def _format_mismatch_report(
    sample_results: list[VideoQASampleResult],
    limit: int = 10,
) -> str:
    mismatches = [result for result in sample_results if not result.is_correct][:limit]
    lines = []
    for result in mismatches:
        prediction = re.sub(r"\s+", " ", result.prediction).strip()
        if len(prediction) > 240:
            prediction = f"{prediction[:237]}..."
        lines.append(
            f"sample={result.sample_id} predicted={result.predicted_letter} "
            f"reference={result.reference} text={prediction!r}"
        )
    return "\n".join(lines)
