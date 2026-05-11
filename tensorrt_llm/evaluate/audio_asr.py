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
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

import soundfile
from tqdm import tqdm

import tensorrt_llm.profiler as profiler
from tensorrt_llm.inputs import (
    ConversationMessage,
    MultimodalData,
    MultimodalDataTracker,
    add_multimodal_placeholders,
)
from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.inputs.utils import _resolve_content_format, load_audio, resolve_hf_chat_template
from tensorrt_llm.inputs.utils import apply_chat_template as trtllm_apply_chat_template
from tensorrt_llm.llmapi import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import SamplingParams

from .interface import Evaluator, get_chat_template_kwargs


class MultimodalASRSample(NamedTuple):
    """A single ASR sample paired with its audio media and reference transcript."""

    sample_id: str
    media: Any
    prompt: str
    transcript: str


class ASRWERSampleResult(NamedTuple):
    """Per-sample WER scoring result, retained so callers can audit worst cases."""

    sample_id: str
    prediction: str
    reference: str
    normalized_prediction: str
    normalized_reference: str
    edits: int
    prediction_words: int
    reference_words: int
    wer: float


class _ASRInputContext(NamedTuple):
    """Per-LLM context needed to render chat-templated audio prompts.

    Built once per evaluation run and reused across samples to avoid re-resolving the model type,
    processor, and chat-template settings on every request.
    """

    model_type: str
    processor: Any
    content_format: ContentFormat
    chat_template_kwargs: dict[str, Any]


class AudioASREvaluator(Evaluator):
    """Evaluator for audio automatic-speech-recognition (ASR) tasks.

    Loads an HF audio dataset, builds chat-templated multimodal prompts (text + audio), submits them
    to the LLM, and scores predictions against references using corpus-level Word Error Rate (WER,
    lower is better).

    NOTE: currently only tested for the VoxPopuli EN test split.
    """

    DEFAULT_PROMPT = (
        "Transcribe the spoken content to written english text, with punctuation "
        "and capitalizations."
    )

    def __init__(
        self,
        dataset_path: str,
        num_samples: Optional[int] = None,
        random_seed: int = 0,
        split: str = "test",
        prompt: str = DEFAULT_PROMPT,
        text_column: str = "normalized_text",
        sample_id_column: str = "audio_id",
        system_prompt: Optional[str] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        # `apply_chat_template` is always True for this evaluator: multimodal
        # placeholders must be spliced into a chat-formatted prompt for the model
        # to consume audio inputs correctly.
        super().__init__(
            random_seed=random_seed,
            apply_chat_template=True,
            system_prompt=system_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.split = split
        self.prompt = prompt
        self.text_column = text_column
        self.sample_id_column = sample_id_column

    def generate_samples(self) -> Iterable[tuple]:
        for sample in self._iter_samples():
            yield sample.prompt, None, sample.transcript, sample

    def _iter_samples(self, dataset: Optional[Any] = None) -> Iterable[MultimodalASRSample]:
        if dataset is None:
            dataset = _load_local_hf_dataset(self.dataset_path, self.split)
        for idx, row in enumerate(dataset):
            if self.num_samples is not None and idx >= self.num_samples:
                break
            transcript = self._get_transcript(row)
            yield MultimodalASRSample(
                sample_id=str(row.get(self.sample_id_column, row.get("id", idx))),
                media=_get_audio_data(row["audio"], self.dataset_path),
                prompt=str(row.get("question", self.prompt)),
                transcript=transcript,
            )

    def compute_score(
        self, outputs: list[RequestOutput], references: list[str], *auxiliaries
    ) -> float:
        """Compute corpus WER (%) over `outputs` vs `references`.

        `auxiliaries[0]` is expected to be the per-sample list of `MultimodalASRSample` so each
        prediction can be tagged with its sample id.
        """
        predictions = [output.outputs[0].text for output in outputs]
        samples = auxiliaries[0]
        wer, sample_results = _compute_wer(
            predictions,
            references,
            sample_ids=[sample.sample_id for sample in samples],
        )
        for result in sample_results:
            logger.debug(
                f"ASR sample {result.sample_id} wer={result.wer:.2f} "
                f"prediction={result.prediction!r} reference={result.reference!r}"
            )

        if wer > 100.0:
            logger.info(
                "ASR WER exceeded 100. This is possible when insertions exceed "
                f"reference words. Worst samples:\n{_format_worst_sample_report(sample_results)}"
            )
        return wer

    def evaluate(
        self,
        llm: Any,
        sampling_params: Optional[SamplingParams] = None,
        streaming: bool = False,
    ) -> float:
        profiler.start("trtllm exec")
        input_context = self._make_input_context(llm)
        dataset = _load_local_hf_dataset(self.dataset_path, self.split)
        num_samples = self._get_num_samples(dataset)
        samples = list(tqdm(self._iter_samples(dataset), desc="Loading samples", total=num_samples))
        inputs = [
            self._make_input(llm, sample, input_context)
            for sample in tqdm(samples, desc="Loading inputs")
        ]
        futures = []
        references = []
        scoring_samples = []
        for sample, request_input in tqdm(
            zip(samples, inputs, strict=True), desc="Submitting requests", total=len(samples)
        ):
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
            references.append(sample.transcript)
            scoring_samples.append(_sample_for_scoring(sample))
        outputs = [future.result() for future in tqdm(futures, desc="Fetching responses")]

        profiler.stop("trtllm exec")
        elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
        logger.info(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")
        profiler.reset("trtllm exec")

        wer = self.compute_score(outputs, references, scoring_samples)
        logger.info(f"audio ASR WER: {wer:.2f}")
        return wer

    def _get_num_samples(self, dataset: Any) -> Optional[int]:
        try:
            num_samples = len(dataset)
        except TypeError:
            return self.num_samples

        if self.num_samples is not None:
            return min(self.num_samples, num_samples)
        return num_samples

    def _get_transcript(self, row: dict[str, Any]) -> str:
        candidate_columns = [
            self.text_column,
            "normalized_text",
            "raw_text",
            "text",
            "transcript",
        ]
        for column in candidate_columns:
            value = row.get(column)
            if value is not None and str(value).strip():
                return str(value)
        raise KeyError(f"Could not find a transcript column in row keys: {sorted(row)}")

    def _make_inputs(self, llm: Any, samples: list[MultimodalASRSample]) -> list[dict[str, Any]]:
        input_context = self._make_input_context(llm)
        return [self._make_input(llm, sample, input_context) for sample in samples]

    def _make_input_context(self, llm: Any) -> _ASRInputContext:
        """Resolve the model type, processor, and chat-template settings once per run."""
        _, model_type = _get_model_context(llm)
        processor = getattr(getattr(llm, "input_processor", None), "processor", None)
        hf_chat_template = resolve_hf_chat_template(llm.tokenizer, processor, None, None)
        return _ASRInputContext(
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
        sample: MultimodalASRSample,
        input_context: _ASRInputContext,
    ) -> dict[str, Any]:
        """Build a `generate_async`-ready dict with chat-templated prompt + audio data."""
        mm_data_tracker = MultimodalDataTracker(input_context.model_type)
        audio = load_audio(sample.media) if isinstance(sample.media, str) else sample.media
        conv = ConversationMessage(
            role="user",
            content=sample.prompt,
            media=[
                MultimodalData(
                    modality="audio",
                    data=audio,
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

        conversation: list[ConversationMessage] = []
        mm_placeholder_counts_by_message: list[dict[str, int]] = []
        if self.system_prompt is not None:
            conversation.append(ConversationMessage(role="system", content=self.system_prompt))
            mm_placeholder_counts_by_message.append({})
        conversation.append(conv)
        mm_placeholder_counts_by_message.append(mm_placeholder_counts)

        prompt = trtllm_apply_chat_template(
            model_type=input_context.model_type,
            tokenizer=llm.tokenizer,
            processor=input_context.processor,
            conversation=conversation,
            add_generation_prompt=True,
            mm_placeholder_counts=mm_placeholder_counts_by_message,
            chat_template_kwargs=input_context.chat_template_kwargs,
        )

        input = {"prompt": prompt}
        multi_modal_data, _ = mm_data_tracker.retrieve_all_sync()
        if multi_modal_data:
            input["multi_modal_data"] = multi_modal_data
        return input


def _resolve_media_path(dataset_path: str, media_path: str) -> str:
    path = Path(media_path)
    if path.is_absolute():
        return str(path)

    dataset_relative_path = Path(dataset_path) / path
    if dataset_relative_path.exists():
        return str(dataset_relative_path)

    return media_path


def _sample_for_scoring(sample: MultimodalASRSample) -> MultimodalASRSample:
    return MultimodalASRSample(
        sample_id=sample.sample_id,
        media=None,
        prompt=sample.prompt,
        transcript=sample.transcript,
    )


def _get_audio_data(audio_value: Any, dataset_path: str) -> Any:
    """Materialize an HF-dataset 'audio' cell into a path or (array, sampling_rate) tuple.

    Accepts the HF Audio feature variants (str path, dict with path/bytes/array, or pre-decoded
    tuple) and resolves dataset-relative paths against `dataset_path`.

    Raises ValueError for unsupported inputs or when a referenced path is missing.
    """
    if isinstance(audio_value, str):
        return _resolve_media_path(dataset_path, audio_value)
    if isinstance(audio_value, dict):
        path = audio_value.get("path")
        if path:
            resolved_path = _resolve_media_path(dataset_path, path)
            if Path(resolved_path).exists():
                return resolved_path
        if audio_value.get("bytes") is not None:
            return soundfile.read(BytesIO(audio_value["bytes"]))
        if "array" in audio_value and "sampling_rate" in audio_value:
            return audio_value["array"], audio_value["sampling_rate"]
        if path:
            raise ValueError(
                f"Audio path {path!r} does not exist relative to dataset "
                f"{dataset_path!r} and no inline bytes/array were provided."
            )
    if isinstance(audio_value, tuple):
        return audio_value
    raise ValueError(
        f"Expected audio to contain local samples or a local path, got: {audio_value!r}"
    )


def _load_local_hf_dataset(dataset_path: str, split: str) -> Any:
    from datasets import Audio, load_dataset

    path = Path(dataset_path)
    if path.is_file() and path.suffix == ".parquet":
        dataset = load_dataset("parquet", data_files={split: str(path)}, split=split)
    else:
        parquet_files = _find_parquet_files(path, split) if path.is_dir() else []
        if parquet_files:
            dataset = load_dataset(
                "parquet",
                data_files={split: [str(file) for file in parquet_files]},
                split=split,
            )
        else:
            dataset = load_dataset(dataset_path, split=split)
    if "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def _find_parquet_files(dataset_path: Path, split: str) -> list[Path]:
    patterns = [
        f"**/{split}-*.parquet",
        f"{split}/**/*.parquet",
    ]
    parquet_files: list[Path] = []
    for pattern in patterns:
        parquet_files.extend(dataset_path.glob(pattern))
    return sorted(set(parquet_files))


def _normalize_asr_text(text: str) -> str:
    """Lightweight ASR text normalizer.

    Casefolds, strips chat/special-token markers (`<|...|>`, `<...>`, `[...]`, parenthesized asides
    ), removes a small set of common English fillers, and drops combining / symbol / punctuation
    Unicode categories so that WER reflects word-level differences. This is intentionally simpler
    than the whisper-normalizer; it is meant as a deliberately conservative in-house normalizer that
    does not depend on third-party packages.
    """
    text = text.casefold()
    text = re.sub(r"<\|.*?\|>", " ", text)
    text = re.sub(r"[<\[][^>\]]*[>\]]", " ", text)
    text = re.sub(r"\(([^)]+?)\)", " ", text)
    text = re.sub(r"\b(hmm|mm|mhm|mmm|uh|um)\b", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char
        if unicodedata.category(char)[0] not in "MSP" and unicodedata.category(char) != "Mn"
        else " "
        for char in text
    )
    return re.sub(r"\s+", " ", text).strip()


def _levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous_row = list(range(len(hypothesis) + 1))
    for row_idx, reference_token in enumerate(reference, start=1):
        current_row = [row_idx]
        for col_idx, hypothesis_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if reference_token == hypothesis_token else 1
            current_row.append(
                min(
                    previous_row[col_idx] + 1,
                    current_row[col_idx - 1] + 1,
                    previous_row[col_idx - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]


def _compute_wer(
    predictions: list[str],
    references: list[str],
    sample_ids: Optional[list[str]] = None,
) -> tuple[float, list[ASRWERSampleResult]]:
    if sample_ids is None:
        sample_ids = [str(idx) for idx in range(len(predictions))]

    sample_results = []
    total_edits = 0
    total_words = 0
    for sample_id, prediction, reference in zip(sample_ids, predictions, references):
        normalized_prediction = _normalize_asr_text(prediction)
        normalized_reference = _normalize_asr_text(reference)
        prediction_words = normalized_prediction.split()
        reference_words = normalized_reference.split()
        edits = _levenshtein_distance(reference_words, prediction_words)
        reference_word_count = len(reference_words)
        total_edits += edits
        total_words += reference_word_count
        wer = 100.0 * edits / reference_word_count if reference_word_count else 0.0
        sample_results.append(
            ASRWERSampleResult(
                sample_id=sample_id,
                prediction=prediction,
                reference=reference,
                normalized_prediction=normalized_prediction,
                normalized_reference=normalized_reference,
                edits=edits,
                prediction_words=len(prediction_words),
                reference_words=reference_word_count,
                wer=wer,
            )
        )
    return 100.0 * total_edits / total_words if total_words else 0.0, sample_results


def _truncate_text(text: str, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _format_worst_sample_report(sample_results: list[ASRWERSampleResult], limit: int = 5) -> str:
    worst_samples = sorted(
        sample_results,
        key=lambda result: (result.wer, result.edits),
        reverse=True,
    )[:limit]
    lines = []
    for result in worst_samples:
        lines.append(
            f"sample={result.sample_id} wer={result.wer:.2f} "
            f"edits={result.edits} ref_words={result.reference_words} "
            f"pred_words={result.prediction_words}\n"
            f"  pred={_truncate_text(result.normalized_prediction)!r}\n"
            f"  ref ={_truncate_text(result.normalized_reference)!r}"
        )
    return "\n".join(lines)


def _get_model_context(llm: Any) -> tuple[str, str]:
    model_dir = getattr(llm, "_hf_model_dir", None) or getattr(llm, "model", None)
    if model_dir is None:
        raise ValueError("The LLM object does not expose a model directory.")

    config_path = Path(model_dir) / "config.json"
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    model_type = config.get("model_type")
    if model_type is None:
        raise KeyError(f"'model_type' is missing from {config_path}.")
    return str(model_dir), model_type
