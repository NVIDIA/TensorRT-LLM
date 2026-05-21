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
import ast
import copy
import hashlib
import json
import logging
import random
import re
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, NamedTuple, Optional

from PIL import Image

if TYPE_CHECKING:
    from tensorrt_llm.llmapi import RequestOutput
    from tensorrt_llm.sampling_params import SamplingParams


logger = logging.getLogger(__name__)


try:
    from tensorrt_llm.evaluate.interface import Evaluator
except ImportError:

    class Evaluator:  # type: ignore[no-redef]
        def __init__(
            self,
            random_seed: int = 0,
            apply_chat_template: bool = False,
            chat_template_kwargs: Optional[dict[str, Any]] = None,
            output_dir: Optional[str] = None,
        ) -> None:
            self.apply_chat_template = apply_chat_template
            self.chat_template_kwargs = chat_template_kwargs
            self.output_dir = output_dir


ANNOTATIONS_FILE = "annotations.jsonl"
AUDIO_WER_THRESHOLD = 50.0
DEFAULT_NUM_FRAMES = 8
SECTION_NAMES = ("image", "audio", "video")
PROMPT_TEMPLATE = """You are given one image, one audio clip, and one video in this single request.
Answer in exactly three labeled lines:
image: <option letter for the image question>
audio: <verbatim transcript of the audio>
video: <option letter for the video question>

Image question: {image_question}
Video question: {video_question}
"""


class MixedModalityInputSample(NamedTuple):
    sample_id: str
    media: Any
    question: str
    keyword: str
    source_paths: tuple[str, ...]


class MixedModalitySample(NamedTuple):
    sample_id: str
    image: Any
    audio: Any
    video_path: str
    image_question: str
    video_question: str
    expected_keywords: dict[str, str]
    source_paths: tuple[str, ...]


class MixedModalitySampleResult(NamedTuple):
    sample_id: str
    prediction: str
    section_matches: dict[str, bool]
    missing_sections: tuple[str, ...]
    expected_keywords: dict[str, str]

    @property
    def is_correct(self) -> bool:
        return not self.missing_sections and all(self.section_matches.values())


class _MixedModalityInputContext(NamedTuple):
    model_type: str
    processor: Any
    content_format: Any
    chat_template_kwargs: dict[str, Any]


class MixedModalityOmniEvaluator(Evaluator):
    """Evaluator for one-request image+audio+video Nemotron Nano Omni coverage.

    Validation procedure: first run with `TRTLLM_ACCURACY_NO_REFERENCE=1` to
    confirm the evaluator and model path execute without thresholds, then collect
    BF16 and NVFP4 on B200 when available, or B100 as a Blackwell fallback when
    B200 is unavailable; collect FP8 on H100. Fill those measured scores in
    `references/mixed_modality_omni.yaml` and re-run the three parametrized
    `TestNanoV3Omni::test_auto_dtype` variants so `assert_passing` verifies the
    final thresholds.
    """

    def __init__(
        self,
        mmmu_dataset_path: str,
        voxpopuli_dataset_path: str,
        videomme_dataset_path: str,
        num_samples: Optional[int] = None,
        random_seed: int = 0,
        num_frames: int = DEFAULT_NUM_FRAMES,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            random_seed=random_seed,
            apply_chat_template=True,
            chat_template_kwargs=chat_template_kwargs,
            output_dir=output_dir,
        )
        self.mmmu_dataset_path = Path(mmmu_dataset_path)
        self.voxpopuli_dataset_path = voxpopuli_dataset_path
        self.videomme_dataset_path = Path(videomme_dataset_path)
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.num_frames = num_frames

    def generate_samples(self) -> Iterable[tuple]:
        # Required by the abstract base, but `evaluate()` owns the multimodal
        # request construction so the base-class sample loop is unused.
        raise NotImplementedError

    def compute_score(
        self,
        outputs: list["RequestOutput"],
        references: list[dict[str, str]],
        samples: list[MixedModalitySample],
    ) -> float:
        predictions = [output.outputs[0].text for output in outputs]
        sample_results = _score_predictions(predictions, samples)
        correct_sections = sum(sum(result.section_matches.values()) for result in sample_results)
        total_sections = len(sample_results) * len(SECTION_NAMES)
        accuracy = 100.0 * correct_sections / total_sections if total_sections else 0.0
        logger.info(
            f"MixedModality section accuracy: {accuracy:.2f} ({correct_sections}/{total_sections})"
        )
        if correct_sections != total_sections:
            logger.info(f"MixedModality mismatches:\n{_format_mismatch_report(sample_results)}")
        return accuracy

    def evaluate(
        self,
        llm: Any,
        sampling_params: Optional["SamplingParams"] = None,
        streaming: bool = False,
    ) -> float:
        import tensorrt_llm.profiler as profiler
        from tensorrt_llm.evaluate.interface import dump_inference_results

        profiler.start("trtllm exec")
        input_context = self._make_input_context(llm)
        samples = list(self._iter_samples())
        _log_source_digest(samples)
        outputs = self._generate_outputs_serially(
            llm,
            samples,
            input_context,
            sampling_params,
            streaming,
        )

        if self.output_dir:
            dump_inference_results(self.output_dir, outputs, getattr(llm, "tokenizer", None))

        profiler.stop("trtllm exec")
        elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
        logger.info(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")
        profiler.reset("trtllm exec")

        references = [sample.expected_keywords for sample in samples]
        return self.compute_score(outputs, references, samples)

    def _generate_outputs_serially(
        self,
        llm: Any,
        samples: list[MixedModalitySample],
        input_context: _MixedModalityInputContext,
        sampling_params: Optional["SamplingParams"],
        streaming: bool,
    ) -> list["RequestOutput"]:
        video_cache: dict[str, Any] = {}
        outputs = []
        for sample in samples:
            request_input = self._make_input(llm, sample, input_context, video_cache)
            if sampling_params is None:
                from tensorrt_llm.sampling_params import SamplingParams

                params = SamplingParams()
            else:
                params = copy.deepcopy(sampling_params)
            outputs.append(
                llm.generate_async(
                    request_input,
                    sampling_params=params,
                    streaming=streaming,
                ).result()
            )
        return outputs

    def _iter_samples(self) -> Iterable[MixedModalitySample]:
        source_sample_count = _target_source_sample_count(self.num_samples)
        image_samples = _load_mmmu_input_samples(self.mmmu_dataset_path, source_sample_count)
        audio_samples = _load_voxpopuli_input_samples(
            self.voxpopuli_dataset_path, source_sample_count
        )
        video_samples = _load_videomme_input_samples(
            self.videomme_dataset_path, source_sample_count
        )
        yield from select_mixed_modality_samples(
            image_samples,
            audio_samples,
            video_samples,
            self.num_samples,
            self.random_seed,
        )

    def _make_input_context(self, llm: Any) -> _MixedModalityInputContext:
        from tensorrt_llm.evaluate.interface import get_chat_template_kwargs, get_model_context
        from tensorrt_llm.inputs.utils import _resolve_content_format, resolve_hf_chat_template

        _, model_type = get_model_context(llm)
        processor = getattr(getattr(llm, "input_processor", None), "processor", None)
        hf_chat_template = resolve_hf_chat_template(llm.tokenizer, processor, None, None)
        return _MixedModalityInputContext(
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
        sample: MixedModalitySample,
        input_context: _MixedModalityInputContext,
        video_cache: dict[str, Any],
    ) -> dict[str, Any]:
        from tensorrt_llm.inputs import (
            ConversationMessage,
            MultimodalData,
            MultimodalDataTracker,
            add_multimodal_placeholders,
            load_video,
        )
        from tensorrt_llm.inputs.content_format import ContentFormat
        from tensorrt_llm.inputs.utils import apply_chat_template as trtllm_apply_chat_template
        from tensorrt_llm.inputs.utils import load_audio

        mm_data_tracker = MultimodalDataTracker(input_context.model_type)
        image = _materialize_image(sample.image)
        audio = load_audio(sample.audio) if isinstance(sample.audio, str) else sample.audio
        video = video_cache.get(sample.video_path)
        if video is None:
            video = load_video(sample.video_path, num_frames=self.num_frames)
            video_cache[sample.video_path] = video

        prompt_text = _format_mixed_modality_prompt(sample)
        conv = ConversationMessage(
            role="user",
            content=prompt_text,
            media=[
                MultimodalData(modality="image", data=image, is_embedding=False),
                MultimodalData(modality="video", data=video, is_embedding=False),
                MultimodalData(modality="audio", data=audio, is_embedding=False),
            ],
            content_parts=[
                {"type": "image", "media_index": 0},
                {"type": "video", "media_index": 1},
                {"type": "audio", "media_index": 2},
                prompt_text,
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


def select_mixed_modality_samples(
    image_samples: list[MixedModalityInputSample],
    audio_samples: list[MixedModalityInputSample],
    video_samples: list[MixedModalityInputSample],
    num_samples: Optional[int],
    random_seed: int,
) -> list[MixedModalitySample]:
    rng = random.Random(random_seed)
    image_samples = list(image_samples)
    audio_samples = list(audio_samples)
    video_samples = list(video_samples)
    rng.shuffle(image_samples)
    rng.shuffle(audio_samples)
    rng.shuffle(video_samples)

    available_samples = min(len(image_samples), len(audio_samples), len(video_samples))
    sample_count = available_samples if num_samples is None else num_samples
    if sample_count > available_samples:
        raise ValueError(
            "Not enough mixed-modality source samples: "
            f"requested={sample_count}, image={len(image_samples)}, "
            f"audio={len(audio_samples)}, video={len(video_samples)}."
        )

    selected = []
    for image_sample, audio_sample, video_sample in zip(
        image_samples[:sample_count],
        audio_samples[:sample_count],
        video_samples[:sample_count],
        strict=True,
    ):
        selected.append(
            MixedModalitySample(
                sample_id=(
                    f"{image_sample.sample_id}+{audio_sample.sample_id}+{video_sample.sample_id}"
                ),
                image=image_sample.media,
                audio=audio_sample.media,
                video_path=str(video_sample.media),
                image_question=image_sample.question,
                video_question=video_sample.question,
                expected_keywords={
                    "image": image_sample.keyword,
                    "audio": audio_sample.keyword,
                    "video": video_sample.keyword,
                },
                source_paths=(
                    *image_sample.source_paths,
                    *audio_sample.source_paths,
                    *video_sample.source_paths,
                ),
            )
        )
    return selected


def _target_source_sample_count(num_samples: Optional[int]) -> Optional[int]:
    if num_samples is None:
        return None
    return max(num_samples * 2, num_samples + 8)


def _load_mmmu_input_samples(
    dataset_path: Path, max_samples: Optional[int] = None
) -> list[MixedModalityInputSample]:
    samples = []
    for idx, (row, _default_subject) in enumerate(_iter_mmmu_rows(dataset_path)):
        image = _get_mmmu_image(row)
        if image is None:
            continue
        answer = _normalize_choice_answer(str(row.get("answer", "")), max_choice="E")
        if answer is None:
            continue
        samples.append(
            MixedModalityInputSample(
                sample_id=str(row.get("id", row.get("sample_id", f"mmmu-{idx}"))),
                media=image,
                question=_format_mmmu_question(row),
                keyword=answer,
                source_paths=_get_media_source_paths(image),
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def _iter_mmmu_rows(dataset_path: Path) -> Iterable[tuple[dict[str, Any], str]]:
    from datasets import get_dataset_config_names, load_dataset

    config_names: list[str] = []
    try:
        config_names = list(get_dataset_config_names(str(dataset_path)))
    except Exception as exc:
        logger.info(f"Could not enumerate MMMU configs from {dataset_path}: {exc}")

    if not config_names:
        for split in ("validation", "val", "test"):
            try:
                dataset = load_dataset(str(dataset_path), split=split)
            except Exception:
                continue
            for row in dataset:
                yield dict(row), str(row.get("subject", "unknown"))
            return

    for config_name in config_names:
        for split in ("validation", "val", "test"):
            try:
                dataset = load_dataset(str(dataset_path), config_name, split=split)
            except Exception:
                continue
            for row in dataset:
                yield dict(row), config_name
            break


def _load_voxpopuli_input_samples(
    dataset_path: str, max_samples: Optional[int] = None
) -> list[MixedModalityInputSample]:
    from tensorrt_llm.evaluate.audio_asr import _get_audio_data, _load_local_hf_dataset

    dataset = _load_local_hf_dataset(dataset_path, "test")
    samples = []
    for idx, row in enumerate(dataset):
        transcript = str(
            row.get("normalized_text")
            or row.get("raw_text")
            or row.get("text")
            or row.get("transcript")
            or ""
        )
        if not transcript.strip():
            continue
        media = _get_audio_data(row["audio"], dataset_path)
        samples.append(
            MixedModalityInputSample(
                sample_id=str(row.get("audio_id", row.get("id", f"voxpopuli-{idx}"))),
                media=media,
                question="Transcribe the spoken audio exactly.",
                keyword=transcript,
                source_paths=_get_media_source_paths(media),
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def _load_videomme_input_samples(
    dataset_path: Path, max_samples: Optional[int] = None
) -> list[MixedModalityInputSample]:
    samples = []
    annotations_path = dataset_path / ANNOTATIONS_FILE
    with open(annotations_path, "r", encoding="utf-8") as annotations_file:
        for idx, line in enumerate(annotations_file):
            row = json.loads(line)
            answer = _normalize_choice_answer(str(row.get("answer", "")), max_choice="D")
            if answer is None:
                continue
            video_path = Path(str(row["video_path"]))
            if not video_path.is_absolute():
                video_path = dataset_path / video_path
            samples.append(
                MixedModalityInputSample(
                    sample_id=str(row.get("sample_id", idx)),
                    media=str(video_path),
                    question=_format_video_question(row),
                    keyword=answer,
                    source_paths=(str(video_path),),
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def _get_mmmu_image(row: dict[str, Any]) -> Any:
    for key in ("image", "image_1", "image_0"):
        image = row.get(key)
        if image is not None:
            return image
    for key, image in row.items():
        if key.startswith("image") and image is not None:
            return image
    return None


def _materialize_image(image: Any) -> Any:
    from tensorrt_llm.inputs.utils import load_image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return load_image(image, format="pil")
    if isinstance(image, dict):
        path = image.get("path")
        if path:
            return load_image(path, format="pil")
        if image.get("bytes") is not None:
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
    return image


def _get_media_source_paths(media: Any) -> tuple[str, ...]:
    if isinstance(media, str):
        return (media,)
    if isinstance(media, dict) and media.get("path"):
        return (str(media["path"]),)
    path = getattr(media, "filename", None)
    if path:
        return (str(path),)
    return ()


def _format_mixed_modality_prompt(sample: MixedModalitySample) -> str:
    return PROMPT_TEMPLATE.format(
        image_question=sample.image_question,
        video_question=sample.video_question,
    )


def _format_mmmu_question(row: dict[str, Any]) -> str:
    question = str(row.get("question", row.get("input", "Answer the image question.")))
    options = _parse_options(row.get("options"))
    if not options:
        return f"{question}\nRespond with only the option letter."
    return f"{question}\nOptions:\n{_format_options(options)}\nRespond with only the option letter."


def _format_video_question(row: dict[str, Any]) -> str:
    question = str(row["question"])
    options = _parse_options(row.get("options"))
    if not options:
        return f"{question}\nRespond with only the option letter."
    return f"{question}\nOptions:\n{_format_options(options)}\nRespond with only the option letter."


def _parse_options(options: Any) -> list[str]:
    if options is None:
        return []
    if isinstance(options, str):
        try:
            parsed_options = ast.literal_eval(options)
        except (SyntaxError, ValueError):
            return [options]
        options = parsed_options
    if isinstance(options, (list, tuple)):
        return [str(option) for option in options]
    return [str(options)]


def _format_options(options: list[str]) -> str:
    lines = []
    for idx, option in enumerate(options):
        letter = chr(ord("A") + idx)
        if re.match(r"^[A-Z]\.", option):
            lines.append(option)
        else:
            lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def _extract_response_sections(text: str) -> dict[str, str]:
    section_pattern = re.compile(
        r"(?ims)^\s*(image|audio|video)\s*:\s*(.*?)(?=^\s*(?:image|audio|video)\s*:|\Z)"
    )
    return {
        match.group(1).casefold(): re.sub(r"\s+", " ", match.group(2)).strip()
        for match in section_pattern.finditer(text)
    }


def _extract_choice_answer(text: str, max_choice: str) -> Optional[str]:
    max_choice = max_choice.upper()
    explicit = re.search(
        rf"\b(?:answer\s*(?:is|:)?|option)\s*\(?\s*([A-{max_choice}a-{max_choice.lower()}])\s*\)?",
        text,
    )
    if explicit:
        return explicit.group(1).upper()

    leading = re.match(
        rf"^\s*\(?([A-{max_choice}a-{max_choice.lower()}])\)?\s*(?:[.:\n\r]|\s+[A-Z]|\s*$)",
        text,
    )
    if leading:
        return leading.group(1).upper()

    choices = re.findall(
        rf"(?<![A-Za-z])([A-{max_choice}a-{max_choice.lower()}])(?![A-Za-z])",
        text,
    )
    return choices[-1].upper() if choices else None


def _normalize_choice_answer(answer: str, max_choice: str) -> Optional[str]:
    answer = answer.strip().upper()
    if re.fullmatch(rf"[A-{max_choice.upper()}]", answer):
        return answer
    return _extract_choice_answer(answer, max_choice)


def _audio_wer(prediction: str, reference: str) -> float:
    prediction_words = _normalize_scoring_text(prediction).split()
    reference_words = _normalize_scoring_text(reference).split()
    if not reference_words:
        return 0.0 if not prediction_words else 100.0
    edits = _levenshtein_distance(reference_words, prediction_words)
    return 100.0 * edits / len(reference_words)


def _normalize_scoring_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"<\|.*?\|>", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
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


def _score_predictions(
    predictions: list[str],
    samples: list[MixedModalitySample],
) -> list[MixedModalitySampleResult]:
    results = []
    for prediction, sample in zip(predictions, samples, strict=True):
        sections = _extract_response_sections(prediction)
        missing_sections = tuple(section for section in SECTION_NAMES if section not in sections)
        section_matches = {
            "image": (
                "image" in sections
                and _extract_choice_answer(sections["image"], "E")
                == sample.expected_keywords["image"]
            ),
            "audio": (
                "audio" in sections
                and _audio_wer(sections["audio"], sample.expected_keywords["audio"])
                <= AUDIO_WER_THRESHOLD
            ),
            "video": (
                "video" in sections
                and _extract_choice_answer(sections["video"], "D")
                == sample.expected_keywords["video"]
            ),
        }
        results.append(
            MixedModalitySampleResult(
                sample_id=sample.sample_id,
                prediction=prediction,
                section_matches=section_matches,
                missing_sections=missing_sections,
                expected_keywords=sample.expected_keywords,
            )
        )
    return results


def _log_source_digest(samples: list[MixedModalitySample]) -> None:
    entries = []
    for sample in samples:
        for source_path in sample.source_paths:
            path = Path(source_path)
            if path.exists():
                entries.append(f"{path.name}:{path.stat().st_size}")
            else:
                entries.append(f"{path.name}:missing")
    digest_input = "\n".join(sorted(entries)).encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()[:16]
    logger.info(f"MixedModality source digest: sha256={digest}; files={len(entries)}")


def _format_mismatch_report(
    sample_results: list[MixedModalitySampleResult],
    limit: int = 10,
) -> str:
    mismatches = [result for result in sample_results if not result.is_correct][:limit]
    lines = []
    for result in mismatches:
        prediction = re.sub(r"\s+", " ", result.prediction).strip()
        if len(prediction) > 240:
            prediction = f"{prediction[:237]}..."
        lines.append(
            f"sample={result.sample_id} missing={result.missing_sections} "
            f"matches={result.section_matches} expected={result.expected_keywords} "
            f"text={prediction!r}"
        )
    return "\n".join(lines)
