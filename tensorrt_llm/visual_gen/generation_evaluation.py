# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Offline image-generation evaluation pipeline utilities.

This module is the library form of the Qwen Image plus Qwen Image Bench flow.
It intentionally does not register a serving route; the public API layer can be
added once this model contract is stable.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

SCORE_MAP = {0: 0.0, 1: 60.0, 2: 100.0}

QUALITY_CHECKLIST = """## Realism
- Physical Logic: Does the image adhere to real-world physical laws (e.g., gravity, reflection, shadow direction, object stability)?
- Material Texture: Do the surface materials of objects (such as skin, fabric, metal, wood) exhibit realistic texture and material properties?
## Detail
- Noise: Is the image rich in detail without excessive noise or unnatural smoothing?
- Edge Clarity: Are the outlines and edges of objects sharp, well-defined, and free from blurring or aliasing?
- Naturalness: Does the image appear natural and free from the artificial "plastic" or "greasy" look commonly associated with AI-generated images?
## Resolution
- Resolution: Is the overall image resolution high-definition, free from visible pixelation or compression artifacts?"""

AESTHETICS_CHECKLIST = """## Composition
- Composition: Is the composition of the image balanced, visually guided, and aesthetically pleasing?
## Color Harmony
- Color Harmony: Is the overall color palette harmonious, cohesive, and appropriate for the mood of the image?
## Lighting
- Lighting & Atmosphere: Does the lighting and shadow atmosphere of the image (such as contrast between light and dark, and the overall lighting atmosphere) match the scene setting of the prompt?
## Anatomical Portraiture
- Anatomical Fidelity: Are the facial feature proportions, skeletal structure, and limb articulation anatomically correct and consistent with human biology? Does the facial skin exhibit realistic micro-level textures such as pores and fine lines?
## Emotional Expression
- Emotional Expression: Does the image's overall aesthetic tone effectively convey the intended emotion and mood described in the prompt?
## Style Control
- Style Control: Does the image accurately capture and represent the specific artistic style requested in the prompt (e.g., Van Gogh's brushwork, Cyberpunk aesthetic)?"""

ALIGNMENT_CHECKLIST = """## Attributes
- Quantity: Does the number of objects in the image match the quantity specified in the prompt?
- Facial Expression: Does the facial expression of the person or animal accurately reflect the emotional state specified in the prompt?
- Material Properties: Do the materials of objects in the image match the material descriptions in the prompt?
- Color: Do the colors of objects in the image match the color specifications in the prompt?
- Shape: Do the shapes of objects in the image match the shape descriptions in the prompt?
- Size: Do the sizes of objects in the image match the size specifications in the prompt?
## Actions
- Contact Interaction: If the prompt involves physical contact between subjects, is the contact interaction depicted naturally and realistically?
- Non-contact Interaction: If the prompt involves non-contact relationships between subjects, is the spatial and social relationship depicted naturally and logically?
- Full-body Action: Does the overall posture and body action of the subject (person or animal) accurately perform the activity described in the prompt?
## Layout
- 2D Space: Are the relative positions of objects on the 2D plane (e.g., left/right, top/bottom, foreground/background) consistent with the prompt's spatial instructions?
- 3D Space: Does the layout, occlusion, and relative position of objects in 3D space conform to the prompt requirements or spatial logic?
## Relations
- Composition Relationship: Does the image successfully integrate multiple elements into a visually coherent and logically consistent whole?
- Difference/Similarity: Are the specified differences or similarities in shape, color, or material between objects accurately represented?
- Containment: Are the containment or enclosure relationships between objects correctly depicted?
## Scene
- Real-world Scene: Does the scene type and environmental setting (e.g., office, forest, street) match the location described in the prompt?
- Virtual Scene: Are the elements within a fictional or fantasy scene internally consistent and logically coherent?"""

REAL_WORLD_FIDELITY_CHECKLIST = """## Fairness
- Social Bias: Does the image avoid reinforcing social biases by automatically associating specific genders with particular professions or settings?
- Cultural Fairness: Is the image free from stereotypical portrayals based on region, race, or cultural background?
## Safety & Compliance
- Safety & Compliance: Is the image safe and compliant, effectively avoiding prohibited content such as pornography, violence, or hate symbols?
## World Knowledge
- Animals: Are real-world animals depicted with anatomically accurate features and realistic biological details?
- Objects: Are the typical appearance, structure, brand logo, or iconic characteristics of real-world items accurately reproduced?
- Information Visualization: Does the image accurately and clearly translate abstract or scientific concepts from the prompt into an effective and understandable visual form?
- Temporal Characteristics: Does the image accurately reflect the iconic elements of a specific historical period (e.g., technology, clothing, architecture, lifestyle of that era)?
- Cultural Elements: Are the cultural elements (such as symbols, traditional clothing, rituals, and customs) accurately depicted and consistent with real-world cultural practices?"""

CREATIVE_GENERATION_CHECKLIST = """## Imagination
- Imagination: Does the image demonstrate creative originality and imaginative thinking when combining novel or surreal elements?
## Feature Matching
- Feature Matching: Are the multi-element fusion regions in the image visually seamless, without abrupt breaks, harsh edges, or logical contradictions?
## Logical Resolution
- Logical Resolution: Does the image accurately depict causal relationships between events (e.g., breaking glass -> shards flying, rain -> wet surfaces)?
## Text Rendering
- Text Accuracy: If the image contains text, is the text clear, legible, and free from garbled characters, misspellings, or typographical errors?
- Text Layout: Is the text layout (e.g., centering, alignment, line spacing, margins) in the image visually appealing and professionally structured?
- Font: Does the font style used in the image match the font type specified in the prompt (e.g., SimSun, Heiti, handwritten, serif)?
- Cross-lingual Generation: Does the image correctly follow the translation instructions in the prompt, producing accurate text in the target language?
## Design Applications
- Graphic Design: Does the graphic design (e.g., advertisement, poster) exhibit a clear information hierarchy, effective visual guidance, and professional layout?
- Product Design: Does the product design in the image demonstrate reasonable industrial design logic (e.g., ergonomic grip, logical interface placement, structural integrity)?
- Spatial Design: Does the interior or architectural space conform to the principles of perspective, proportion, and building design standards?
- Fashion Styling: Does the clothing cut and silhouette match the style described in the prompt (e.g., Hanfu, cyberpunk, haute couture)? Does the makeup style (e.g., smoky eyes, nude makeup, theatrical look) suit the occasion and character setting?
- Game Design: Do the game props and UI elements have practical in-game usability (e.g., icon recognizability, interactive affordances, clear feedback cues)?
- Art Design: Does the image successfully demonstrate the specific artistic design style required by the prompt (e.g., unique brushstrokes, distinctive color scheme, coherent artistic language)?
## Visual Storytelling
- Cinematic Style: Does the image reproduce the signature visual language of the specific director referenced in the prompt (e.g., Wes Anderson's symmetrical composition, Wong Kar-wai's warm color palette)?
- Camera / Lens Style: Does the image reflect the characteristic imaging effects of the specific photographic equipment or lens referenced in the prompt (e.g., film grain, bokeh, digital sharpening)?
- Storyboard Creation: Does the image's scene composition follow the panel layout requirements outlined in the prompt (e.g., three-panel, four-panel, split-screen)?
- Shot Sizes: Does the image meet the framing and shot size requirements specified in the prompt (e.g., close-up, medium shot, wide shot)?
- Composition: Does the image follow the specific composition rules required by the prompt (e.g., rule of thirds, golden ratio, leading lines)?
- Angles: Does the camera angle comply with the prompt's specification (e.g., bird's-eye view, low angle, Dutch angle)?
- Comic Creation: Does the image conform to the comic style required by the prompt (e.g., American comics, Japanese manga, European BD)?"""

DIM_TO_CHECKLIST = {
    "Quality": QUALITY_CHECKLIST,
    "Aesthetics": AESTHETICS_CHECKLIST,
    "Alignment": ALIGNMENT_CHECKLIST,
    "Real-world Fidelity": REAL_WORLD_FIDELITY_CHECKLIST,
    "Creative Generation": CREATIVE_GENERATION_CHECKLIST,
}

DEFAULT_DIMENSIONS = tuple(DIM_TO_CHECKLIST)

SYSTEM_PROMPT = (
    "You are an expert evaluator for text-to-image (T2I) generation quality. "
    "Given an image and the text prompt used to generate it, you evaluate the image "
    "on specific quality criteria using a structured checklist."
)

USER_PROMPT_TEMPLATE = """\
# Text Prompt Used to Generate the Image
{prompt}

# Generated Image
<image>

# Evaluation Dimension
{level1_dim}

# Scoring Rules
- **0 (Fail)**: Clear defect present. Would noticeably reduce image quality.
- **1 (Pass)**: No defect. Meets baseline expectations.
- **2 (Excel)**: Exceptionally executed. Only when concrete excellence is observable.
- **N/A**: This criterion does not apply to this image/prompt.

# Evaluation Checklist
{format_checklist}

# Output Format
Respond with a valid JSON object only (no markdown code blocks):
{{
  "{{level2_dim}}": {{
    "{{level3_dim}}": {{"score": 0|1|2}},
    "{{level3_dim}}": {{"score": "N/A"}}
  }}
}}"""

CHECKLIST_L3_TO_L2 = {
    "Quality": {
        "Physical Logic": "Realism",
        "Material Texture": "Realism",
        "Noise": "Detail",
        "Edge Clarity": "Detail",
        "Naturalness": "Detail",
        "Resolution": "Resolution",
    },
    "Aesthetics": {
        "Composition": "Composition",
        "Color Harmony": "Color Harmony",
        "Lighting & Atmosphere": "Lighting",
        "Anatomical Fidelity": "Anatomical Portraiture",
        "Emotional Expression": "Emotional Expression",
        "Style Control": "Style Control",
    },
    "Alignment": {
        "Quantity": "Attributes",
        "Facial Expression": "Attributes",
        "Material Properties": "Attributes",
        "Color": "Attributes",
        "Shape": "Attributes",
        "Size": "Attributes",
        "Contact Interaction": "Actions",
        "Non-contact Interaction": "Actions",
        "Full-body Action": "Actions",
        "2D Space": "Layout",
        "3D Space": "Layout",
        "Composition Relationship": "Relations",
        "Difference/Similarity": "Relations",
        "Containment": "Relations",
        "Real-world Scene": "Scene",
        "Virtual Scene": "Scene",
    },
    "Real-world Fidelity": {
        "Social Bias": "Fairness",
        "Cultural Fairness": "Fairness",
        "Safety & Compliance": "Safety & Compliance",
        "Animals": "World Knowledge",
        "Objects": "World Knowledge",
        "Information Visualization": "World Knowledge",
        "Temporal Characteristics": "World Knowledge",
        "Cultural Elements": "World Knowledge",
    },
    "Creative Generation": {
        "Imagination": "Imagination",
        "Feature Matching": "Feature Matching",
        "Logical Resolution": "Logical Resolution",
        "Text Accuracy": "Text Rendering",
        "Text Layout": "Text Rendering",
        "Font": "Text Rendering",
        "Cross-lingual Generation": "Text Rendering",
        "Graphic Design": "Design Applications",
        "Product Design": "Design Applications",
        "Spatial Design": "Design Applications",
        "Fashion Styling": "Design Applications",
        "Game Design": "Design Applications",
        "Art Design": "Design Applications",
        "Cinematic Style": "Visual Storytelling",
        "Camera / Lens Style": "Visual Storytelling",
        "Storyboard Creation": "Visual Storytelling",
        "Shot Sizes": "Visual Storytelling",
        "Composition": "Visual Storytelling",
        "Angles": "Visual Storytelling",
        "Comic Creation": "Visual Storytelling",
    },
}

L3_RENAME = {
    "Creative Generation": {
        "Feature Mapping": "Feature Matching",
    },
}


@dataclass(frozen=True)
class QwenImageBenchEvaluatorArgs:
    backend: str = "pytorch"
    image_data_format: str = "pt"
    max_tokens: int = 4096
    max_num_tokens: int = 8192
    max_seq_len: int = 8192
    kv_cache_max_tokens: int = 8192
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    use_fast_processor: bool = False
    enable_block_reuse: bool = False
    include_raw_outputs: bool = False
    max_batch_size: int = 1


@dataclass
class QwenImageBenchResult:
    prompt: str
    dimensions: list[str]
    level1_scores: dict[str, float | None] = field(default_factory=dict)
    level2_scores: dict[str, dict[str, float | None] | None] = field(default_factory=dict)
    level3_scores: dict[str, dict[str, dict[str, float | None]] | None] = field(
        default_factory=dict
    )
    total_score: float | None = None
    parse_failures: list[str] = field(default_factory=list)
    parsed_scores: dict[str, Any] = field(default_factory=dict)
    raw_outputs: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ImageGenerationEvaluationResult:
    prompt: str
    score: float | None
    level1_scores: dict[str, float | None] = field(default_factory=dict)
    level2_scores: dict[str, dict[str, float | None] | None] = field(default_factory=dict)
    level3_scores: dict[str, dict[str, dict[str, float | None]] | None] = field(
        default_factory=dict
    )
    parse_failures: list[str] = field(default_factory=list)
    image: Any | None = None
    image_path: str | None = None
    error: str | None = None


@dataclass
class GenerationEvaluationResponse:
    created: int
    aggregate_score: float | None
    aggregation: dict[str, Any]
    results: list[ImageGenerationEvaluationResult]
    timing: dict[str, float]


def build_user_prompt(prompt: str, level1_dim: str) -> str:
    _validate_dimensions([level1_dim])
    return USER_PROMPT_TEMPLATE.format(
        prompt=prompt,
        level1_dim=level1_dim,
        format_checklist=DIM_TO_CHECKLIST[level1_dim],
    )


def extract_json_from_response(response_text: str) -> dict[str, Any] | None:
    text = response_text
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :]
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def map_score(raw_score: Any) -> float | None:
    if isinstance(raw_score, str) and raw_score.upper() == "N/A":
        return None
    try:
        return SCORE_MAP[int(raw_score)]
    except (KeyError, TypeError, ValueError):
        return None


def mean_non_none(values: Sequence[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def fix_score_json(score_json: dict[str, Any] | None, level1_dim: str) -> dict[str, Any] | None:
    if not score_json:
        return score_json

    _validate_dimensions([level1_dim])
    mapping = CHECKLIST_L3_TO_L2.get(level1_dim, {})
    rename = L3_RENAME.get(level1_dim, {})
    first_val = next(iter(score_json.values()), None)

    if isinstance(first_val, dict) and "score" in first_val:
        result: dict[str, Any] = {}
        for level3_name, score_obj in score_json.items():
            level3_name = rename.get(level3_name, level3_name)
            level2_name = mapping.get(level3_name, level3_name)
            result.setdefault(level2_name, {})[level3_name] = score_obj
        return result

    result = {}
    for level2_key, level3_dict in score_json.items():
        if not isinstance(level3_dict, dict):
            continue
        for level3_name, score_obj in level3_dict.items():
            level3_name = rename.get(level3_name, level3_name)
            correct_level2 = mapping.get(level3_name, level2_key)
            result.setdefault(correct_level2, {})[level3_name] = score_obj
    return result


def compute_dimension_score(score_json: dict[str, Any]) -> dict[str, Any]:
    level2_scores = {}
    level3_scores = {}

    for level2_name, level3_dict in score_json.items():
        level3_scores[level2_name] = {}
        level3_mapped = []
        for level3_name, score_obj in level3_dict.items():
            raw = score_obj.get("score") if isinstance(score_obj, dict) else score_obj
            mapped = map_score(raw)
            level3_scores[level2_name][level3_name] = mapped
            if mapped is not None:
                level3_mapped.append(mapped)
        level2_scores[level2_name] = mean_non_none(level3_mapped)

    return {
        "level1_score": mean_non_none(list(level2_scores.values())),
        "level2_scores": level2_scores,
        "level3_scores": level3_scores,
    }


def aggregate_total_score(dim_results: dict[str, dict[str, Any]]) -> float | None:
    return mean_non_none(
        [
            result["level1_score"]
            for result in dim_results.values()
            if result is not None and result.get("level1_score") is not None
        ]
    )


def parse_dimension_output(
    output_text: str, level1_dim: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    score_json = extract_json_from_response(output_text)
    if not score_json:
        return None, None
    fixed_score_json = fix_score_json(score_json, level1_dim)
    if fixed_score_json is None:
        return None, None
    return fixed_score_json, compute_dimension_score(fixed_score_json)


def validate_generation_evaluation_request(
    *,
    prompts: Sequence[str],
    generation_n: int = 1,
    dimensions: Sequence[str] | None = None,
    aggregation_method: str = "mean",
) -> tuple[list[str], list[str]]:
    prompt_list = _validate_prompts(prompts)
    dimension_list = _validate_dimensions(dimensions)
    if generation_n != 1:
        raise ValueError("generation.n must be 1 for the Phase 1 pipeline")
    if aggregation_method != "mean":
        raise ValueError("Only mean aggregation is supported for the Phase 1 pipeline")
    return prompt_list, dimension_list


def save_generated_image_for_evaluation(output: Any, image_path: Path) -> str:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    save = getattr(output, "save", None)
    if callable(save):
        return str(save(image_path))

    from tensorrt_llm.visual_gen.output import VisualGenOutput

    return str(VisualGenOutput(image=getattr(output, "image", None)).save(image_path))


def make_qwen_image_bench_input(
    *,
    llm: Any,
    processor: Any,
    model_type: str,
    prompt: str,
    image: Any,
    level1_dim: str,
    image_data_format: str,
) -> dict[str, Any]:
    from tensorrt_llm.inputs.content_format import ContentFormat
    from tensorrt_llm.inputs.utils import (
        MultimodalDataTracker,
        _resolve_content_format,
        apply_chat_template,
        interleave_mm_placeholders,
        load_image,
        resolve_hf_chat_template,
    )

    user_text = build_user_prompt(prompt, level1_dim)
    if isinstance(image, str | Path):
        image_data = load_image(str(image), format=image_data_format, device="cpu")
    else:
        image_data = image

    tracker = MultimodalDataTracker(model_type)
    tracker.add_data("image", image_data, is_embedding=False)
    mm_counts = tracker.placeholder_counts()
    placeholder_modalities = tracker.placeholder_modalities()

    system_conv = {"role": "system", "content": SYSTEM_PROMPT, "media": []}
    before, sep, after = user_text.partition("<image>")
    if sep:
        content_parts: list[str | dict[str, Any]] = [
            before,
            {"type": "image", "media_index": 0},
            after,
        ]
    else:
        content_parts = [user_text, {"type": "image", "media_index": 0}]
    user_conv = {
        "role": "user",
        "content": user_text,
        "media": [
            {
                "modality": "image",
                "data": image_data,
                "is_embedding": False,
            }
        ],
        "content_parts": content_parts,
    }

    hf_template = resolve_hf_chat_template(llm.tokenizer, processor, None, None)
    content_format = _resolve_content_format(model_type, hf_template)
    if content_format != ContentFormat.OPENAI:
        user_conv["content"] = interleave_mm_placeholders(
            model_type,
            content_parts,
            mm_counts,
            placeholder_modalities,
        )

    rendered_prompt = apply_chat_template(
        model_type=model_type,
        tokenizer=llm.tokenizer,
        processor=processor,
        conversation=[system_conv, user_conv],
        add_generation_prompt=True,
        mm_placeholder_counts=[{}, mm_counts],
        chat_template_kwargs={"enable_thinking": True},
    )
    mm_data, _ = tracker.retrieve_all_sync()
    return {"prompt": rendered_prompt, "multi_modal_data": mm_data}


class QwenImageBenchEvaluator:
    """Reusable Qwen Image Bench evaluator.

    The model is initialized lazily on first evaluation so parser and aggregation
    utilities can be imported without model dependencies or checkpoint assets.
    """

    def __init__(
        self,
        model_path: str | Path,
        args: QwenImageBenchEvaluatorArgs | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.args = args or QwenImageBenchEvaluatorArgs()
        self._llm: Any | None = None
        self._llm_context: Any | None = None
        self._processor: Any | None = None
        self._model_type: str | None = None
        self._sampling_params: Any | None = None

    def __enter__(self) -> "QwenImageBenchEvaluator":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def start(self) -> "QwenImageBenchEvaluator":
        if self._llm is not None:
            return self

        from transformers import AutoProcessor

        from tensorrt_llm.llmapi import KvCacheConfig
        from tensorrt_llm.llmapi.llm import LLM, SamplingParams

        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Expected local Qwen-Image-Bench checkpoint with config.json: {self.model_path}"
            )

        model_config = json.loads(config_path.read_text())
        self._model_type = model_config["model_type"]
        self._processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            use_fast=self.args.use_fast_processor,
            trust_remote_code=True,
        )
        self._sampling_params = SamplingParams(
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
        )
        self._llm_context = LLM(
            model=str(self.model_path),
            backend=self.args.backend,
            trust_remote_code=True,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=self.args.enable_block_reuse,
                max_tokens=self.args.kv_cache_max_tokens,
            ),
            max_batch_size=self.args.max_batch_size,
            max_num_tokens=self.args.max_num_tokens,
            max_seq_len=self.args.max_seq_len,
        )
        self._llm = self._llm_context.__enter__()
        return self

    def close(self) -> None:
        if self._llm_context is not None:
            self._llm_context.__exit__(None, None, None)
        self._llm = None
        self._llm_context = None

    def evaluate_batch(
        self,
        prompts: Sequence[str],
        images: Sequence[Any],
        dimensions: Sequence[str] | None = None,
    ) -> list[QwenImageBenchResult]:
        prompt_list = _validate_prompts(prompts)
        image_list = list(images)
        if len(prompt_list) != len(image_list):
            raise ValueError("prompts and images must have the same length")
        dimension_list = _validate_dimensions(dimensions)
        self.start()
        return [
            self._evaluate_one(prompt, image, dimension_list)
            for prompt, image in zip(prompt_list, image_list)
        ]

    def _evaluate_one(self, prompt: str, image: Any, dimensions: list[str]) -> QwenImageBenchResult:
        assert self._llm is not None
        assert self._processor is not None
        assert self._model_type is not None
        assert self._sampling_params is not None

        raw_outputs: dict[str, str] = {}
        parsed_scores: dict[str, Any] = {}
        dimension_results: dict[str, dict[str, Any]] = {}
        parse_failures: list[str] = []

        try:
            for level1_dim in dimensions:
                trtllm_input = make_qwen_image_bench_input(
                    llm=self._llm,
                    processor=self._processor,
                    model_type=self._model_type,
                    prompt=prompt,
                    image=image,
                    level1_dim=level1_dim,
                    image_data_format=self.args.image_data_format,
                )
                output_text = (
                    self._llm.generate([trtllm_input], sampling_params=self._sampling_params)[0]
                    .outputs[0]
                    .text
                )
                raw_outputs[level1_dim] = output_text
                fixed_score_json, dimension_score = parse_dimension_output(output_text, level1_dim)
                parsed_scores[level1_dim] = fixed_score_json
                if dimension_score is None:
                    parse_failures.append(level1_dim)
                else:
                    dimension_results[level1_dim] = dimension_score
        except Exception as e:  # noqa: BLE001 - convert item failure into result data.
            return QwenImageBenchResult(
                prompt=prompt,
                dimensions=dimensions,
                error=str(e),
                raw_outputs=raw_outputs if self.args.include_raw_outputs else {},
            )

        return QwenImageBenchResult(
            prompt=prompt,
            dimensions=dimensions,
            level1_scores={
                dim: (dimension_results.get(dim) or {}).get("level1_score") for dim in dimensions
            },
            level2_scores={
                dim: (dimension_results.get(dim) or {}).get("level2_scores") for dim in dimensions
            },
            level3_scores={
                dim: (dimension_results.get(dim) or {}).get("level3_scores") for dim in dimensions
            },
            total_score=aggregate_total_score(dimension_results),
            parse_failures=parse_failures,
            parsed_scores=parsed_scores,
            raw_outputs=raw_outputs if self.args.include_raw_outputs else {},
        )


class ImageGenerationEvaluationPipeline:
    """Offline Qwen Image generation plus Qwen Image Bench evaluation pipeline."""

    def __init__(self, generator: Any, evaluator: Any) -> None:
        self.generator = generator
        self.evaluator = evaluator

    def run(
        self,
        prompts: Sequence[str],
        *,
        generation_params: Any | None = None,
        generation_n: int = 1,
        dimensions: Sequence[str] | None = None,
        aggregation_method: str = "mean",
        return_images: bool = False,
        image_output_dir: str | Path | None = None,
    ) -> GenerationEvaluationResponse:
        prompt_list, dimension_list = validate_generation_evaluation_request(
            prompts=prompts,
            generation_n=generation_n,
            dimensions=dimensions,
            aggregation_method=aggregation_method,
        )

        start_time = time.monotonic()
        generation_start = time.monotonic()
        generated = self.generator.generate(inputs=prompt_list, params=generation_params)
        generation_seconds = time.monotonic() - generation_start

        outputs = generated if isinstance(generated, list) else [generated]
        if len(outputs) != len(prompt_list):
            raise ValueError("Generator returned a different number of outputs than input prompts")

        results: list[ImageGenerationEvaluationResult | None] = [None] * len(prompt_list)
        eval_prompts: list[str] = []
        eval_images: list[Any] = []
        eval_indices: list[int] = []
        eval_image_paths: dict[int, str] = {}
        image_output_path = Path(image_output_dir) if image_output_dir is not None else None

        for idx, (prompt, output) in enumerate(zip(prompt_list, outputs)):
            error = getattr(output, "error", None)
            image = getattr(output, "image", None)
            if error is not None:
                results[idx] = ImageGenerationEvaluationResult(
                    prompt=prompt,
                    score=None,
                    image=image if return_images else None,
                    error=error,
                )
                continue
            if image is None:
                results[idx] = ImageGenerationEvaluationResult(
                    prompt=prompt,
                    score=None,
                    error="Generator output did not contain an image",
                )
                continue
            eval_image = image
            if image_output_path is not None:
                image_path = save_generated_image_for_evaluation(
                    output, image_output_path / f"{idx:04d}.png"
                )
                eval_image_paths[idx] = image_path
                eval_image = image_path
            eval_prompts.append(prompt)
            eval_images.append(eval_image)
            eval_indices.append(idx)

        evaluation_start = time.monotonic()
        evaluator_results = (
            self.evaluator.evaluate_batch(eval_prompts, eval_images, dimension_list)
            if eval_prompts
            else []
        )
        evaluation_seconds = time.monotonic() - evaluation_start

        if len(evaluator_results) != len(eval_indices):
            raise ValueError(
                "Evaluator returned a different number of results than generated images"
            )

        for idx, eval_result, image in zip(eval_indices, evaluator_results, eval_images):
            results[idx] = ImageGenerationEvaluationResult(
                prompt=eval_result.prompt,
                score=eval_result.total_score,
                level1_scores=eval_result.level1_scores,
                level2_scores=eval_result.level2_scores,
                level3_scores=eval_result.level3_scores,
                parse_failures=eval_result.parse_failures,
                image=image if return_images else None,
                image_path=eval_image_paths.get(idx),
                error=eval_result.error,
            )

        aggregation_start = time.monotonic()
        finalized_results = [result for result in results if result is not None]
        successful_scores = [
            result.score
            for result in finalized_results
            if result.error is None and result.score is not None
        ]
        aggregate_score = mean_non_none(successful_scores)
        aggregation_seconds = time.monotonic() - aggregation_start

        return GenerationEvaluationResponse(
            created=int(time.time()),
            aggregate_score=aggregate_score,
            aggregation={
                "method": aggregation_method,
                "num_prompts": len(prompt_list),
                "num_successful": len(successful_scores),
                "num_failed": len(prompt_list) - len(successful_scores),
            },
            results=finalized_results,
            timing={
                "generation_seconds": generation_seconds,
                "evaluation_seconds": evaluation_seconds,
                "aggregation_seconds": aggregation_seconds,
                "total_seconds": time.monotonic() - start_time,
            },
        )


def _validate_prompts(prompts: Sequence[str]) -> list[str]:
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise ValueError("prompts must be a non-empty sequence of strings")
    prompt_list = list(prompts)
    if not prompt_list or any(not isinstance(prompt, str) for prompt in prompt_list):
        raise ValueError("prompts must be a non-empty sequence of strings")
    if any(not prompt for prompt in prompt_list):
        raise ValueError("prompts cannot contain empty strings")
    return prompt_list


def _validate_dimensions(dimensions: Sequence[str] | None = None) -> list[str]:
    dimension_list = list(dimensions) if dimensions is not None else list(DEFAULT_DIMENSIONS)
    if not dimension_list:
        raise ValueError("dimensions must be non-empty")
    unknown = [dim for dim in dimension_list if dim not in DIM_TO_CHECKLIST]
    if unknown:
        raise ValueError(f"Unsupported Qwen Image Bench dimensions: {unknown}")
    return dimension_list
