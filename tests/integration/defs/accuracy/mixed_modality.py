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
import os
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

# Expected failures when probing a `datasets` source for configs/splits that may
# not exist (absent config, missing split, malformed schema). Narrow the
# dataset-discovery handlers to these so genuinely unexpected failures (network
# errors, OOM, etc.) surface instead of being silently swallowed.
_DATASET_PROBE_ERRORS = (FileNotFoundError, ValueError, KeyError)
MODALITY_IMAGE = "image"
MODALITY_AUDIO = "audio"
MODALITY_VIDEO = "video"
SUPPORTED_MODALITIES = (MODALITY_IMAGE, MODALITY_AUDIO, MODALITY_VIDEO)
MODALITY_DISPLAY_NAMES = {
    MODALITY_IMAGE: "one image",
    MODALITY_AUDIO: "one audio clip",
    MODALITY_VIDEO: "one video",
}
PROMPT_TEMPLATE = """You are given {modality_list} in this single request.
{distractor_line}
Answer only the {target_modality} question below.

{target_question}
"""


class MixedModalityInputSample(NamedTuple):
    """One source example from a single-modality dataset."""

    sample_id: str
    media: Any
    question: str
    keyword: str
    source_paths: tuple[str, ...]


class MixedModalitySample(NamedTuple):
    """One mixed request assembled from modality-specific samples."""

    sample_id: str
    items: dict[str, MixedModalityInputSample]
    target_modality: str


class MixedModalityTargetResult(NamedTuple):
    """Scoring result for the selected target modality in one request."""

    sample_id: str
    target_modality: str
    prediction: str
    is_correct: bool
    expected_keyword: str
    distractor_modalities: tuple[str, ...]


class MixedModalityTargetScoreSummary(NamedTuple):
    """Aggregate target-only accuracy for mixed or pure requests."""

    target_accuracy: float
    correct_targets: int
    total_requests: int
    target_correct_counts: dict[str, int]
    target_total_counts: dict[str, int]


class MixedModalityPairedBaselineSummary(NamedTuple):
    """Mixed-versus-pure paired baseline deltas and counts."""

    mixed_minus_pure_accuracy: float
    mixed_minus_pure_accuracy_by_target: dict[str, float]
    paired_counts_by_target: dict[str, dict[str, int]]


class _MixedModalityInputContext(NamedTuple):
    """Model input-format context reused while constructing requests."""

    model_type: str
    processor: Any
    content_format: Any
    chat_template_kwargs: dict[str, Any]


class MixedModalityEvaluator(Evaluator):
    """Evaluator for one-request mixed media coverage.

    Validation uses target-only randomized query scoring: every request carries
    all configured active modalities, but asks for one seeded random target
    modality while the other media remain random distractors. Each mixed
    request is also paired with a pure request containing only the selected
    target modality, and the mixed-vs-pure delta is always gated.
    """

    def __init__(
        self,
        modality_dataset_paths: dict[str, str],
        active_modalities: tuple[str, ...] = SUPPORTED_MODALITIES,
        target_modalities: Optional[tuple[str, ...]] = None,
        num_samples: Optional[int] = None,
        random_seed: int = 0,
        num_frames: int = DEFAULT_NUM_FRAMES,
        extract_video_audio: bool = False,
        pure_baseline_max_accuracy_drop: float = 5.0,
        pure_baseline_max_per_target_accuracy_drop: float = 10.0,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            random_seed=random_seed,
            apply_chat_template=True,
            chat_template_kwargs=chat_template_kwargs,
            output_dir=output_dir,
        )
        self.active_modalities = _normalize_modalities(active_modalities, "active", min_count=2)
        self.target_modalities = _normalize_modalities(
            self.active_modalities if target_modalities is None else target_modalities,
            "target",
            min_count=1,
        )
        _validate_target_modalities(self.active_modalities, self.target_modalities)
        _validate_num_samples(num_samples, self.target_modalities)
        self.modality_dataset_paths = dict(modality_dataset_paths)
        _validate_dataset_paths(self.active_modalities, self.modality_dataset_paths)
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.num_frames = num_frames
        # When True, the video loader extracts each video's embedded audio track
        # (load_video(..., extract_audio=True)) so the video item rides into the
        # model as a video carrying audio. Nano then hoists that embedded audio
        # to a first-class top-level `(audio, k)` item (no ghosts, no shared
        # slots, no post-encode re-placement), exercising the hoisted-audio path
        # when video is mixed with other modalities. Requires PyAV at runtime
        # (TRTLLM_ENABLE_PYAV=1); see _materialize_media.
        self.extract_video_audio = extract_video_audio
        if self.extract_video_audio:
            # Media (incl. audio extraction) is materialized in this process
            # before each request is submitted, so enabling the PyAV gate here is
            # sufficient. extract_audio_from_video reads this env var at call time.
            os.environ.setdefault("TRTLLM_ENABLE_PYAV", "1")
        self.pure_baseline_max_accuracy_drop = pure_baseline_max_accuracy_drop
        self.pure_baseline_max_per_target_accuracy_drop = pure_baseline_max_per_target_accuracy_drop

    def generate_samples(self) -> Iterable[tuple]:
        """Keep the base evaluator sample loop disabled for this custom flow."""
        # Required by the abstract base, but `evaluate()` owns the multimodal
        # request construction so the base-class sample loop is unused.
        raise NotImplementedError

    def compute_score(
        self,
        outputs: list["RequestOutput"],
        references: list[dict[str, str]],
        samples: list[MixedModalitySample],
    ) -> float:
        """Return target-only accuracy for generated request outputs."""
        _, target_summary = self._score_outputs(outputs, samples, label="MixedModality")
        return target_summary.target_accuracy

    def _score_outputs(
        self,
        outputs: list["RequestOutput"],
        samples: list[MixedModalitySample],
        label: str,
    ) -> tuple[list[MixedModalityTargetResult], MixedModalityTargetScoreSummary]:
        """Score request outputs against the selected target modality only."""
        predictions = [output.outputs[0].text for output in outputs]
        target_results = _score_target_predictions(predictions, samples)
        target_summary = _summarize_target_results(target_results, self.target_modalities)
        logger.info(
            f"{label} target accuracy: "
            f"{target_summary.target_accuracy:.2f} "
            f"({target_summary.correct_targets}/{target_summary.total_requests})"
        )
        logger.info(
            f"{label} per-target correct counts: "
            + ", ".join(
                f"{modality}="
                f"{target_summary.target_correct_counts[modality]}/"
                f"{target_summary.target_total_counts[modality]}"
                for modality in self.target_modalities
            )
        )
        if target_summary.correct_targets != target_summary.total_requests:
            logger.info(
                f"{label} target mismatches:\n{_format_target_mismatch_report(target_results)}"
            )
        return target_results, target_summary

    def evaluate(
        self,
        llm: Any,
        sampling_params: Optional["SamplingParams"] = None,
        streaming: bool = False,
    ) -> float:
        """Run mixed requests, paired pure baselines, and degradation gates."""
        import tensorrt_llm.profiler as profiler
        from tensorrt_llm.evaluate.interface import dump_inference_results

        profiler.start("trtllm exec")
        input_context = self._make_input_context(llm)
        samples = list(self._iter_samples())
        _log_source_digest(samples)
        _log_target_modality_counts(samples, self.target_modalities)
        outputs = self._generate_outputs_serially(
            llm,
            samples,
            input_context,
            sampling_params,
            streaming,
        )
        pure_samples = _make_pure_target_samples(samples)
        pure_outputs = self._generate_outputs_serially(
            llm,
            pure_samples,
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

        mixed_results, mixed_summary = self._score_outputs(outputs, samples, label="MixedModality")
        pure_results, pure_summary = self._score_outputs(
            pure_outputs, pure_samples, label="MixedModality pure baseline"
        )
        _assert_pure_baseline_not_degraded(
            mixed_results=mixed_results,
            pure_results=pure_results,
            mixed_summary=mixed_summary,
            pure_summary=pure_summary,
            target_modalities=self.target_modalities,
            max_accuracy_drop=self.pure_baseline_max_accuracy_drop,
            max_per_target_accuracy_drop=(self.pure_baseline_max_per_target_accuracy_drop),
        )

        return mixed_summary.target_accuracy

    def _generate_outputs_serially(
        self,
        llm: Any,
        samples: list[MixedModalitySample],
        input_context: _MixedModalityInputContext,
        sampling_params: Optional["SamplingParams"],
        streaming: bool,
    ) -> list["RequestOutput"]:
        """Submit requests one at a time to bound media cache pressure."""
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
        """Load per-modality sources and yield deterministic mixed samples."""
        # Oversample source rows so target-modality filtering still yields enough.
        source_sample_count = (
            None if self.num_samples is None else max(self.num_samples * 2, self.num_samples + 8)
        )
        input_samples_by_modality = {
            modality: _load_input_samples(
                modality, self.modality_dataset_paths[modality], source_sample_count
            )
            for modality in self.active_modalities
        }
        yield from select_mixed_modality_samples(
            input_samples_by_modality,
            self.num_samples,
            self.random_seed,
            active_modalities=self.active_modalities,
            target_modalities=self.target_modalities,
        )

    def _make_input_context(self, llm: Any) -> _MixedModalityInputContext:
        """Resolve chat-template and content-format state from the LLM."""
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
        """Build one LLM request carrying all sample media and one target question."""
        from tensorrt_llm.inputs import (
            ConversationMessage,
            MultimodalData,
            MultimodalDataTracker,
            add_multimodal_placeholders,
        )
        from tensorrt_llm.inputs.content_format import ContentFormat
        from tensorrt_llm.inputs.utils import apply_chat_template as trtllm_apply_chat_template

        mm_data_tracker = MultimodalDataTracker(input_context.model_type)
        prompt_text = _format_mixed_modality_prompt(sample)
        media = []
        content_parts: list[Any] = []
        for media_index, (modality, item) in enumerate(sample.items.items()):
            media_data = _materialize_media(
                modality,
                item.media,
                video_cache,
                self.num_frames,
                extract_video_audio=self.extract_video_audio,
            )
            media.append(MultimodalData(modality=modality, data=media_data, is_embedding=False))
            content_parts.append({"type": modality, "media_index": media_index})
        content_parts.append(prompt_text)

        conv = ConversationMessage(
            role="user",
            content=prompt_text,
            media=media,
            content_parts=content_parts,
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
    input_samples_by_modality: dict[str, list[MixedModalityInputSample]],
    num_samples: Optional[int],
    random_seed: int,
    active_modalities: Optional[tuple[str, ...]] = None,
    target_modalities: Optional[tuple[str, ...]] = None,
) -> list[MixedModalitySample]:
    """Select deterministic mixed samples with randomized target assignment."""
    rng = random.Random(random_seed)
    active_modalities = _normalize_modalities(
        tuple(input_samples_by_modality) if active_modalities is None else active_modalities,
        "active",
        min_count=2,
    )
    target_modalities = _normalize_modalities(
        active_modalities if target_modalities is None else target_modalities,
        "target",
        min_count=1,
    )
    _validate_target_modalities(active_modalities, target_modalities)
    _validate_num_samples(num_samples, target_modalities)

    samples_by_modality = {
        modality: list(input_samples_by_modality[modality]) for modality in active_modalities
    }
    for samples in samples_by_modality.values():
        rng.shuffle(samples)

    available_samples = min(len(samples) for samples in samples_by_modality.values())
    sample_count = available_samples if num_samples is None else num_samples
    if sample_count > available_samples:
        counts = ", ".join(
            f"{modality}={len(samples_by_modality[modality])}" for modality in active_modalities
        )
        raise ValueError(
            f"Not enough mixed-modality source samples: requested={sample_count}, {counts}."
        )

    selected_targets = _select_target_modalities(sample_count, rng, target_modalities)
    selected = []
    for sample_idx, target_modality in enumerate(selected_targets):
        items = {
            modality: samples_by_modality[modality][sample_idx] for modality in active_modalities
        }
        selected.append(
            MixedModalitySample(
                sample_id="+".join(item.sample_id for item in items.values()),
                items=items,
                target_modality=target_modality,
            )
        )
    return selected


def _normalize_modalities(
    modalities: tuple[str, ...], label: str, min_count: int
) -> tuple[str, ...]:
    """Validate modality names, duplicates, and minimum set size."""
    if len(modalities) < min_count:
        raise ValueError(
            f"Mixed modality {label} set must contain at least {min_count} modality entries."
        )
    seen = set()
    normalized = []
    for modality in modalities:
        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported mixed modality {label} entry: {modality!r}. "
                f"Supported modalities: {SUPPORTED_MODALITIES}."
            )
        if modality in seen:
            raise ValueError(f"Duplicate mixed modality {label} entry: {modality!r}.")
        seen.add(modality)
        normalized.append(modality)
    return tuple(normalized)


def _validate_target_modalities(
    active_modalities: tuple[str, ...],
    target_modalities: tuple[str, ...],
) -> None:
    """Ensure queried modalities are present in each mixed request."""
    missing_targets = set(target_modalities) - set(active_modalities)
    if missing_targets:
        raise ValueError(
            "Mixed modality target modalities must be a subset of active modalities: "
            f"active={active_modalities}, target={target_modalities}."
        )


def _validate_num_samples(
    num_samples: Optional[int],
    target_modalities: tuple[str, ...],
) -> None:
    """Fail fast on an unusable sample count before any GPU work runs.

    `num_samples` must be positive and at least the number of target modalities
    so each target is exercised at least once. Without this check an invalid
    count is only caught by the per-target baseline gate after both serial
    evaluation passes have already consumed GPU time.
    """
    if num_samples is None:
        return
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if num_samples < len(target_modalities):
        raise ValueError(
            f"num_samples ({num_samples}) must be at least the number of target "
            f"modalities ({len(target_modalities)}) so each target is exercised."
        )


def _validate_dataset_paths(
    active_modalities: tuple[str, ...],
    modality_dataset_paths: dict[str, str],
) -> None:
    """Ensure every active modality has a configured dataset path."""
    missing_paths = [
        modality for modality in active_modalities if modality not in modality_dataset_paths
    ]
    if missing_paths:
        raise ValueError("Missing mixed modality dataset path(s): " + ", ".join(missing_paths))


def _select_target_modalities(
    sample_count: int,
    rng: random.Random,
    target_modalities: tuple[str, ...],
) -> list[str]:
    """Generate a balanced, shuffled target-modality schedule."""
    selected_targets = [
        target_modalities[idx % len(target_modalities)] for idx in range(sample_count)
    ]
    rng.shuffle(selected_targets)
    return selected_targets


def _load_input_samples(
    modality: str,
    dataset_path: str,
    max_samples: Optional[int],
) -> list[MixedModalityInputSample]:
    """Dispatch source dataset loading by modality."""
    if modality == MODALITY_IMAGE:
        return _load_mmmu_input_samples(Path(dataset_path), max_samples)
    if modality == MODALITY_AUDIO:
        return _load_voxpopuli_input_samples(dataset_path, max_samples)
    if modality == MODALITY_VIDEO:
        return _load_videomme_input_samples(Path(dataset_path), max_samples)
    raise ValueError(f"Unsupported mixed modality input loader: {modality!r}.")


def _load_mmmu_input_samples(
    dataset_path: Path, max_samples: Optional[int] = None
) -> list[MixedModalityInputSample]:
    """Load image question-answer samples from an MMMU-style dataset."""
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
    """Yield MMMU rows across available configs and evaluation splits."""
    from datasets import get_dataset_config_names, load_dataset

    config_names: list[str] = []
    try:
        config_names = list(get_dataset_config_names(str(dataset_path)))
    except _DATASET_PROBE_ERRORS as exc:
        logger.info(f"Could not enumerate MMMU configs from {dataset_path}: {exc}")

    if not config_names:
        for split in ("validation", "val", "test"):
            try:
                dataset = load_dataset(str(dataset_path), split=split)
            except _DATASET_PROBE_ERRORS:
                continue
            for row in dataset:
                yield dict(row), str(row.get("subject", "unknown"))
            return

    for config_name in config_names:
        for split in ("validation", "val", "test"):
            try:
                dataset = load_dataset(str(dataset_path), config_name, split=split)
            except _DATASET_PROBE_ERRORS:
                continue
            for row in dataset:
                yield dict(row), config_name
            break


def _load_voxpopuli_input_samples(
    dataset_path: str, max_samples: Optional[int] = None
) -> list[MixedModalityInputSample]:
    """Load audio transcription samples from a VoxPopuli-style dataset."""
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
    """Load video multiple-choice samples from a VideoMME-style dataset."""
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
    """Return the first available image field from an MMMU row."""
    for key in ("image", "image_1", "image_0"):
        image = row.get(key)
        if image is not None:
            return image
    for key, image in row.items():
        if key.startswith("image") and image is not None:
            return image
    return None


def _materialize_media(
    modality: str,
    media: Any,
    video_cache: dict[str, Any],
    num_frames: int,
    extract_video_audio: bool = False,
) -> Any:
    """Load or normalize media payloads just before request construction.

    When `extract_video_audio` is set, videos are loaded with
    `extract_audio=True` so each `VideoData` carries its embedded audio track.
    Fed into the model as a `MultimodalData(modality="video", ...)`, this drives
    the Nano video-audio interleave path. Audio extraction needs PyAV, which is
    lazily imported and gated behind `TRTLLM_ENABLE_PYAV=1`; callers that enable
    this flag must set that env var (the loader raises otherwise).
    """
    if modality == MODALITY_IMAGE:
        return _materialize_image(media)
    if modality == MODALITY_AUDIO:
        from tensorrt_llm.inputs.utils import load_audio

        return load_audio(media) if isinstance(media, str) else media
    if modality == MODALITY_VIDEO:
        from tensorrt_llm.inputs import load_video

        video_path = str(media)
        cache_key = (video_path, extract_video_audio)
        video = video_cache.get(cache_key)
        if video is None:
            video = load_video(
                video_path,
                num_frames=num_frames,
                extract_audio=extract_video_audio,
            )
            video_cache[cache_key] = video
        return video
    raise ValueError(f"Unsupported mixed modality media: {modality!r}.")


def _materialize_image(image: Any) -> Any:
    """Normalize image payload variants to a PIL-compatible image."""
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
    """Return paths that identify a sample media source."""
    if isinstance(media, str):
        return (media,)
    if isinstance(media, dict) and media.get("path"):
        return (str(media["path"]),)
    path = getattr(media, "filename", None)
    if path:
        return (str(path),)
    return ()


def _format_mixed_modality_prompt(sample: MixedModalitySample) -> str:
    """Format the target-only prompt with distractor context."""
    target_modality = _require_target_modality(sample)
    if len(sample.items) > 1:
        distractor_line = "The non-target media are random distractors."
    else:
        distractor_line = "No distractor media are included."
    return PROMPT_TEMPLATE.format(
        modality_list=_format_modality_list(tuple(sample.items)),
        distractor_line=distractor_line,
        target_modality=target_modality,
        target_question=_target_question(sample, target_modality),
    )


def _format_modality_list(modalities: tuple[str, ...]) -> str:
    """Return human-readable modality list text for the prompt."""
    display_names = [MODALITY_DISPLAY_NAMES[modality] for modality in modalities]
    if len(display_names) == 1:
        return display_names[0]
    if len(display_names) == 2:
        return f"{display_names[0]} and {display_names[1]}"
    return f"{', '.join(display_names[:-1])}, and {display_names[-1]}"


def _format_mmmu_question(row: dict[str, Any]) -> str:
    """Format an MMMU question and choices for target scoring."""
    question = str(row.get("question", row.get("input", "Answer the image question.")))
    options = _parse_options(row.get("options"))
    if not options:
        return f"{question}\nRespond with only the option letter."
    return f"{question}\nOptions:\n{_format_options(options)}\nRespond with only the option letter."


def _format_video_question(row: dict[str, Any]) -> str:
    """Format a video question and choices for target scoring."""
    question = str(row["question"])
    options = _parse_options(row.get("options"))
    if not options:
        return f"{question}\nRespond with only the option letter."
    return f"{question}\nOptions:\n{_format_options(options)}\nRespond with only the option letter."


def _parse_options(options: Any) -> list[str]:
    """Normalize dataset option fields into a list of strings."""
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
    """Render choices with option letters when the dataset omits them."""
    lines = []
    for idx, option in enumerate(options):
        letter = chr(ord("A") + idx)
        if re.match(r"^[A-Z]\.", option):
            lines.append(option)
        else:
            lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def _format_target_mismatch_report(
    target_results: list[MixedModalityTargetResult],
    limit: int = 10,
) -> str:
    """Summarize target-scoring failures for logs."""
    mismatches = [result for result in target_results if not result.is_correct][:limit]
    lines = []
    for result in mismatches:
        prediction = re.sub(r"\s+", " ", result.prediction).strip()
        if len(prediction) > 240:
            prediction = f"{prediction[:237]}..."
        lines.append(
            f"sample={result.sample_id} target={result.target_modality} "
            f"distractors={result.distractor_modalities} "
            f"expected={result.expected_keyword!r} text={prediction!r}"
        )
    return "\n".join(lines)


def _extract_response_sections(text: str) -> dict[str, str]:
    """Extract optional modality-prefixed answer sections from model text."""
    modality_pattern = "|".join(SUPPORTED_MODALITIES)
    section_pattern = re.compile(
        rf"(?ims)^\s*({modality_pattern})\s*:\s*(.*?)(?=^\s*(?:{modality_pattern})\s*:|\Z)"
    )
    return {
        match.group(1).casefold(): re.sub(r"\s+", " ", match.group(2)).strip()
        for match in section_pattern.finditer(text)
    }


def _extract_choice_answer(text: str, max_choice: str) -> Optional[str]:
    """Extract a multiple-choice option from model text."""
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
    """Normalize answer keys and generated choices to one letter."""
    answer = answer.strip().upper()
    if re.fullmatch(rf"[A-{max_choice.upper()}]", answer):
        return answer
    return _extract_choice_answer(answer, max_choice)


def _require_target_modality(sample: MixedModalitySample) -> str:
    """Return a valid target modality or raise a sample-shape error."""
    target_modality = sample.target_modality
    if target_modality not in sample.items:
        raise ValueError(
            f"Mixed modality target sample {sample.sample_id!r} has invalid "
            f"target modality: {target_modality!r}; active={tuple(sample.items)}."
        )
    return target_modality


def _target_question(sample: MixedModalitySample, target_modality: str) -> str:
    """Return the question for the selected target modality."""
    return sample.items[target_modality].question


def _audio_wer(prediction: str, reference: str) -> float:
    """Compute word error rate percentage for audio transcript scoring.

    Delegates to the shared ASR normalizer and edit-distance helpers in
    `tensorrt_llm.evaluate.audio_asr` so mixed-audio scoring is consistent with
    the pure-audio evaluator (same tag/bracket/filler stripping and WER contract).
    """
    from tensorrt_llm.evaluate.audio_asr import _levenshtein_distance, _normalize_asr_text

    prediction_words = _normalize_asr_text(prediction).split()
    reference_words = _normalize_asr_text(reference).split()
    if not reference_words:
        return 0.0 if not prediction_words else 100.0
    edits = _levenshtein_distance(reference_words, prediction_words)
    return 100.0 * edits / len(reference_words)


def _score_target_predictions(
    predictions: list[str],
    samples: list[MixedModalitySample],
) -> list[MixedModalityTargetResult]:
    """Score each prediction against only its selected target modality."""
    results = []
    for prediction, sample in zip(predictions, samples, strict=True):
        target_modality = _require_target_modality(sample)
        target_item = sample.items[target_modality]
        results.append(
            MixedModalityTargetResult(
                sample_id=sample.sample_id,
                target_modality=target_modality,
                prediction=prediction,
                is_correct=_score_target_answer(
                    target_modality,
                    prediction,
                    target_item.keyword,
                ),
                expected_keyword=target_item.keyword,
                distractor_modalities=tuple(
                    modality for modality in sample.items if modality != target_modality
                ),
            )
        )
    return results


def _score_target_answer(
    target_modality: str,
    prediction: str,
    expected_keyword: str,
) -> bool:
    """Apply modality-specific target-answer scoring."""
    sections = _extract_response_sections(prediction)
    target_text = sections.get(target_modality, prediction)
    if target_modality == MODALITY_IMAGE:
        return _extract_choice_answer(target_text, "E") == expected_keyword
    if target_modality == MODALITY_AUDIO:
        return _audio_wer(target_text, expected_keyword) <= AUDIO_WER_THRESHOLD
    if target_modality == MODALITY_VIDEO:
        return _extract_choice_answer(target_text, "D") == expected_keyword
    raise ValueError(f"Unsupported target modality: {target_modality!r}.")


def _summarize_target_results(
    target_results: list[MixedModalityTargetResult],
    target_modalities: Optional[tuple[str, ...]] = None,
) -> MixedModalityTargetScoreSummary:
    """Aggregate target-only correctness overall and by modality."""
    if target_modalities is None:
        target_modalities = tuple(
            dict.fromkeys(result.target_modality for result in target_results)
        )
    total_requests = len(target_results)
    target_total_counts = {
        modality: sum(result.target_modality == modality for result in target_results)
        for modality in target_modalities
    }
    target_correct_counts = {
        modality: sum(
            result.target_modality == modality and result.is_correct for result in target_results
        )
        for modality in target_modalities
    }
    correct_targets = sum(result.is_correct for result in target_results)
    target_accuracy = 100.0 * correct_targets / total_requests if total_requests else 0.0
    return MixedModalityTargetScoreSummary(
        target_accuracy=target_accuracy,
        correct_targets=correct_targets,
        total_requests=total_requests,
        target_correct_counts=target_correct_counts,
        target_total_counts=target_total_counts,
    )


def _make_pure_target_samples(
    samples: list[MixedModalitySample],
) -> list[MixedModalitySample]:
    """Build paired single-modality requests for mixed-vs-pure gating."""
    pure_samples = []
    for sample in samples:
        target_modality = _require_target_modality(sample)
        pure_samples.append(
            MixedModalitySample(
                sample_id=f"{sample.sample_id}|pure-{target_modality}",
                items={target_modality: sample.items[target_modality]},
                target_modality=target_modality,
            )
        )
    return pure_samples


def _summarize_paired_baseline(
    mixed_results: list[MixedModalityTargetResult],
    pure_results: list[MixedModalityTargetResult],
    mixed_summary: MixedModalityTargetScoreSummary,
    pure_summary: MixedModalityTargetScoreSummary,
    target_modalities: tuple[str, ...],
) -> MixedModalityPairedBaselineSummary:
    """Compute mixed-minus-pure deltas from aligned sample pairs."""
    mixed_minus_pure_by_target = {}
    paired_counts_by_target = {}
    for modality in target_modalities:
        pairs = [
            (mixed, pure)
            for mixed, pure in zip(mixed_results, pure_results, strict=True)
            if mixed.target_modality == modality
        ]
        if any(mixed.target_modality != pure.target_modality for mixed, pure in pairs):
            raise AssertionError(f"Pure baseline target mismatch for modality {modality}.")
        sample_count = len(pairs)
        mixed_correct = sum(mixed.is_correct for mixed, _ in pairs)
        pure_correct = sum(pure.is_correct for _, pure in pairs)
        mixed_accuracy = 100.0 * mixed_correct / sample_count if sample_count else 0.0
        pure_accuracy = 100.0 * pure_correct / sample_count if sample_count else 0.0
        mixed_minus_pure_by_target[modality] = mixed_accuracy - pure_accuracy
        paired_counts_by_target[modality] = {
            "sample_count": sample_count,
            "mixed_correct": mixed_correct,
            "pure_correct": pure_correct,
            "both_correct": sum(mixed.is_correct and pure.is_correct for mixed, pure in pairs),
            "mixed_only": sum(mixed.is_correct and not pure.is_correct for mixed, pure in pairs),
            "pure_only": sum(not mixed.is_correct and pure.is_correct for mixed, pure in pairs),
            "both_wrong": sum(
                not mixed.is_correct and not pure.is_correct for mixed, pure in pairs
            ),
        }
    return MixedModalityPairedBaselineSummary(
        mixed_minus_pure_accuracy=(mixed_summary.target_accuracy - pure_summary.target_accuracy),
        mixed_minus_pure_accuracy_by_target=mixed_minus_pure_by_target,
        paired_counts_by_target=paired_counts_by_target,
    )


def _assert_pure_baseline_not_degraded(
    mixed_results: list[MixedModalityTargetResult],
    pure_results: list[MixedModalityTargetResult],
    mixed_summary: MixedModalityTargetScoreSummary,
    pure_summary: MixedModalityTargetScoreSummary,
    target_modalities: tuple[str, ...],
    max_accuracy_drop: float,
    max_per_target_accuracy_drop: float,
) -> None:
    """Fail when mixed accuracy drops below paired pure-baseline gates."""
    paired_summary = _summarize_paired_baseline(
        mixed_results=mixed_results,
        pure_results=pure_results,
        mixed_summary=mixed_summary,
        pure_summary=pure_summary,
        target_modalities=target_modalities,
    )
    logger.info(
        "MixedModality pure baseline comparison: "
        f"mixed={mixed_summary.target_accuracy:.2f}, "
        f"pure={pure_summary.target_accuracy:.2f}, "
        f"mixed_minus_pure={paired_summary.mixed_minus_pure_accuracy:.2f}, "
        f"max_allowed_drop={max_accuracy_drop:.2f}"
    )
    logger.info(
        "MixedModality pure baseline per-target comparison: "
        + ", ".join(
            f"{modality}=mixed_minus_pure "
            f"{paired_summary.mixed_minus_pure_accuracy_by_target[modality]:.2f} "
            f"counts={paired_summary.paired_counts_by_target[modality]}"
            for modality in target_modalities
        )
    )

    aggregate_drop = -paired_summary.mixed_minus_pure_accuracy
    if aggregate_drop > max_accuracy_drop:
        raise AssertionError(
            "Mixed modality target accuracy regressed against pure baseline: "
            f"mixed={mixed_summary.target_accuracy:.2f}, "
            f"pure={pure_summary.target_accuracy:.2f}, "
            f"drop={aggregate_drop:.2f}, max_allowed_drop={max_accuracy_drop:.2f}."
        )

    for modality in target_modalities:
        sample_count = paired_summary.paired_counts_by_target[modality]["sample_count"]
        if sample_count == 0:
            raise AssertionError(
                f"Mixed modality pure baseline did not exercise target modality {modality!r}."
            )
        modality_drop = -paired_summary.mixed_minus_pure_accuracy_by_target[modality]
        if modality_drop > max_per_target_accuracy_drop:
            raise AssertionError(
                "Mixed modality target accuracy regressed against pure baseline "
                f"for {modality}: drop={modality_drop:.2f}, "
                f"max_allowed_drop={max_per_target_accuracy_drop:.2f}, "
                f"counts={paired_summary.paired_counts_by_target[modality]}."
            )


def _log_source_digest(samples: list[MixedModalitySample]) -> None:
    """Log a stable digest of materialized source media paths."""
    entries = []
    for sample in samples:
        for item in sample.items.values():
            for source_path in item.source_paths:
                path = Path(source_path)
                if path.exists():
                    entries.append(f"{path.name}:{path.stat().st_size}")
                else:
                    entries.append(f"{path.name}:missing")
    digest_input = "\n".join(sorted(entries)).encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()[:16]
    logger.info(f"MixedModality source digest: sha256={digest}; files={len(entries)}")


def _log_target_modality_counts(
    samples: list[MixedModalitySample],
    target_modalities: tuple[str, ...],
) -> None:
    """Log target-modality coverage for the selected samples."""
    target_counts = {
        modality: sum(sample.target_modality == modality for sample in samples)
        for modality in target_modalities
    }
    logger.info(
        "MixedModality target modality counts: "
        + ", ".join(
            f"{modality}={target_counts[modality]}/{len(samples)}" for modality in target_modalities
        )
    )
