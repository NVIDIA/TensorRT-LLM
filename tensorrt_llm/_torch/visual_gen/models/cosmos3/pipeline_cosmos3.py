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

import functools
import json
import math
import os
import time
from typing import Any, Iterable, List, Optional, Union

import PIL.Image
import torch
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.

    def tqdm(iterable, **kwargs):
        return iterable


from tensorrt_llm._torch.visual_gen.models.wan.vae_loader import load_wan_vae
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, RefSlotSpec, RoleSpec
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import (
    classify_worker_error,
    postprocess_video_tensor,
    synchronize_media_prepare_status,
)
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.inputs.media_io import ImageMediaIO
from tensorrt_llm.logger import logger
from tensorrt_llm.media.decoding import decode_video_reference_window, video_stream_info

from .action import (
    ACTION_MODE_INVERSE_DYNAMICS,
    DEFAULT_ACTION_VIEW_POINT,
    action_reference_frame_step,
    action_reference_size,
    action_start_frame_offset,
    build_action_json_prompt,
    build_vision_condition_mask,
    normalize_action_mode,
    pil_to_rgb,
    prepare_action_latents,
    resize_and_pad_action_image,
    resolve_action_size,
    resolve_domain_id,
)
from .defaults import (
    COSMOS3_720P_PARAMS,
    COSMOS3_ENVELOPES,
    COSMOS3_EXTRA_SPECS,
    COSMOS3_GENERATION_DEFAULTS,
    COSMOS3_V2V_DEFAULT_FLOW_SHIFT,
    _normalize_condition_video_keep,
    _normalize_condition_video_latent_indexes,
    resolve_domain_action_config,
    _validate_video_reference,
)
from .guardrails import check_video_safety, download_guardrail_checkpoint
from .negative_prompt import COSMOS3_VIDEO_NEGATIVE_PROMPT
from .sampling import DISTILLED_GUIDANCE_SCALE, Cosmos3SamplingPolicy, load_scheduler
from .sound_tokenizer import LatentAutoEncoderV2
from .transfer import (
    TRANSFER_DEFAULTS,
    Cosmos3TransferConfig,
    decode_media_to_uint8_cthw,
    find_closest_target_size,
    load_or_compute_control_frames,
    pad_temporal_frames,
    resolve_transfer_config,
    uint8_cthw_to_normalized_5d,
)
from .transformer_cosmos3 import NEMOTRON_DENSE_RECIPE, Cosmos3VFMTransformer, resolve_arch_recipe

# Image modes declare no negative prompt in the reference
# while every video mode points at ``neg_prompts.json``.
COSMOS3_DEFAULT_NEGATIVE_PROMPT = ""


@functools.lru_cache(maxsize=1)
def default_video_negative_prompt() -> str:
    """The reference's default negative prompt for video modes.

    Serialized the way the reference loads it -- ``json.dumps(json.loads(...))`` --
    so the text reaching the tokenizer is byte-identical.
    """
    return json.dumps(COSMOS3_VIDEO_NEGATIVE_PROMPT)


def default_negative_prompt(output_type: str) -> str:
    """Default negative prompt for a request, keyed on output kind not request mode.

    The reference wires its negative prompt into every video mode and none of the
    image ones, so anything producing an image defaults to empty.
    """
    return (
        COSMOS3_DEFAULT_NEGATIVE_PROMPT
        if output_type == "image"
        else default_video_negative_prompt()
    )


# NOTE: Intentional typo in "give" instead of "given" to match training setup.
COSMOS3_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate videos from a give prompt."
)
COSMOS3_T2I_SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate images from a give prompt."
)
COSMOS3_V2V_FLOW_SHIFT = 10.0
# Fraction of a transfer hint that may be mirror-padding before it is worth a
# warning. A few frames of tail ping-pong is normal when clips differ slightly;
# beyond this the control is mostly invented and the client likely sent clips of
# different videos.
CONTROL_LENGTH_MISMATCH_RATIO = 0.10
COSMOS3_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
COSMOS3_DEFAULT_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
COSMOS3_IMAGE_RESOLUTION_TEMPLATE = "This image is of {height}x{width} resolution."

TRTLLM_DISABLE_COSMOS3_GUARDRAILS = os.environ.get("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", "0") == "1"

# Public offload component names for the two transformer towers. The "reasoner"
# (understanding) pathway is the causal language model that processes text; the
# "generator" (generation) pathway is the stack of cross-attention layers that
# produces video tokens. Only the heavy decoder-layer ModuleLists are offloaded;
# the small shared embeddings/projections/norms stay resident on GPU.
COSMOS3_REASONER_OFFLOAD_COMPONENT = "reasoner"
COSMOS3_GENERATOR_OFFLOAD_COMPONENT = "generator"
# Opt-in guardrail offload components (rank 0 only): the Qwen3Guard text checker and
# the RetinaFace video face-blur model. Kept out of the CPU defaults below.
COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT = "text_guardrail"
COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT = "video_guardrail"
_COSMOS3_DEFAULT_OFFLOAD_STAGES = (
    (COSMOS3_REASONER_OFFLOAD_COMPONENT,),
    (COSMOS3_GENERATOR_OFFLOAD_COMPONENT,),
)

# ``W,H`` bucket names the reference builds requests from. A request there starts
# as a (resolution, bucket) pair and the bucket string is carried into the prompt
# verbatim; we only ever see the resolved height/width, so map back to the nearest
# bucket. Emitting the exact reduced ratio instead would put a string the model
# never saw in training into the caption (832x480 reduces to "15,26", not "16,9").
COSMOS3_ASPECT_RATIO_BUCKETS = ("1,1", "4,3", "3,4", "16,9", "9,16")


def _aspect_ratio_bucket(height: int, width: int) -> str:
    """Nearest reference aspect-ratio bucket for a resolved frame size."""
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Cosmos3 aspect ratio needs positive dimensions, got height={height}, width={width}."
        )
    ratio = width / height
    return min(
        COSMOS3_ASPECT_RATIO_BUCKETS,
        key=lambda bucket: abs(
            math.log(ratio / (int(bucket.split(",")[0]) / int(bucket.split(",")[1])))
        ),
    )


def _validate_sampling_recipe(family: str, use_native_flow_schedule: bool, sampling) -> None:
    """Family, model_index schedule flag, and scheduler recipe must form a
    known-supported combination — the pieces come from three different
    checkpoint files and a mismatch samples the wrong trajectory silently.
    """
    if family == NEMOTRON_DENSE_RECIPE.name:
        if sampling.is_distilled:
            raise ValueError(
                "Distilled (fixed-sigma FlowMatchEuler) sampling is not supported for "
                "the Edge (nemotron_dense) family; no such checkpoint exists."
            )
        if not use_native_flow_schedule:
            raise ValueError(
                "Edge (nemotron_dense) checkpoints must declare "
                "use_native_flow_schedule: true in model_index.json. Without it the "
                "checkpoint's karras scheduler config would sample the wrong "
                "trajectory; a missing flag means a broken or stale conversion."
            )
    elif use_native_flow_schedule:
        raise ValueError(
            "use_native_flow_schedule is only supported for the Edge "
            f"(nemotron_dense) family, but this checkpoint's family is {family!r}."
        )


def _assert_anchor_matches(image_latent: torch.Tensor, latents: torch.Tensor) -> None:
    """The I2V conditioning frame must be writable into the denoised latents as-is.

    Both derive from the pipeline dtype/device, so a mismatch means an upstream
    change broke that. Slice assignment would hide it behind a per-step cast
    rather than fail, which is why this is checked and never coerced.
    """
    if image_latent.dtype != latents.dtype or image_latent.device != latents.device:
        raise RuntimeError(
            "Cosmos3 I2V conditioning latent must match the denoised latents: got "
            f"conditioning {image_latent.dtype} on {image_latent.device}, expected "
            f"{latents.dtype} on {latents.device}."
        )


def _validate_temporal_compression(transformer, vae_scale_factor_temporal: int) -> None:
    """A config-declared temporal compression factor must match the VAE."""
    if (
        getattr(transformer, "temporal_compression_factor_declared", False)
        and transformer.temporal_compression_factor != vae_scale_factor_temporal
    ):
        raise ValueError(
            f"Transformer config declares temporal_compression_factor="
            f"{transformer.temporal_compression_factor}, but the VAE reports "
            f"scale_factor_temporal={vae_scale_factor_temporal}."
        )


def _condition_pixel_frame_count(
    condition_video_latent_indexes: Iterable[int],
    temporal_compression: int,
) -> int:
    return max(condition_video_latent_indexes) * int(temporal_compression) + 1


def _load_reference_image(data: bytes):
    """Load an I2V reference, reporting unreadable content as a client error.

    The worker's load is the acceptance check — the serve boundary only
    routes on the container signature — so PIL's ``OSError``
    (``UnidentifiedImageError`` for a bad header, plain ``OSError`` partway
    through a truncated file) has to become a ``ValueError`` here, or a bad
    upload would be reported as a server fault.
    """
    try:
        return ImageMediaIO(format="pil").load_bytes(data)
    except OSError as exc:
        raise ValueError(
            f"Image reference could not be decoded; it may be truncated, "
            f"corrupt, or in an unsupported format: {exc}"
        ) from exc


@register_pipeline(
    "Cosmos3OmniMoTPipeline",
    hf_ids=[
        "nvidia/Cosmos3-Nano",
        "nvidia/Cosmos3-Super",
        "nvidia/Cosmos3-Super-Image2Video",
        "nvidia/Cosmos3-Super-Image2Video-4Step",
        "nvidia/Cosmos3-Super-Text2Image",
        "nvidia/Cosmos3-Super-Text2Image-4Step",
        "nvidia/Cosmos3-Edge",
    ],
    doc="Cosmos3 Omnimodal world models.",
)
class Cosmos3OmniMoTPipeline(BasePipeline):
    def __init__(self, pipeline_config):
        primary_pretrained_config = pipeline_config.primary_pretrained_config
        self.audio_gen = False
        # Checkpoint fact vs runtime capability: the checkpoint may ship
        # action weights, but action generation is not implemented here.
        self.has_action_weights = False
        self.action_gen = False
        # Pre-load placeholder; load_standard_components derives the real
        # policy from the checkpoint's scheduler via from_scheduler().
        self.sampling = Cosmos3SamplingPolicy()
        # Independent of the extra-param spec, whose default stays None so an
        # omitted value survives the executor's default merge and reaches
        # forward() as "unset".
        self.default_use_system_prompt = False
        self.use_native_flow_schedule = False
        self.family = resolve_arch_recipe(primary_pretrained_config).name
        # Schedulers are identified by their resolved (flow_shift, karras) pair
        self._scheduler_cache: dict = {}
        if getattr(
            primary_pretrained_config,
            "audio_gen",
            getattr(primary_pretrained_config, "sound_gen", False),
        ):
            logger.info("Initializing Cosmos3OmniMoTPipeline with audio generation.")
            self.audio_gen = True

        if getattr(primary_pretrained_config, "action_gen", False):
            logger.info("Initializing Cosmos3OmniMoTPipeline with action generation.")
            self.has_action_weights = True
            self.action_gen = True

        super().__init__(pipeline_config)

    def _mode_params(self, mode: str) -> dict:
        """Generation default table for this checkpoint family and request mode."""
        return COSMOS3_GENERATION_DEFAULTS[(self.family, mode)]

    def _resolve_generation_params(self, mode: str, **values) -> dict:
        """Fill None values: sampling-policy overrides win, then the mode
        table, then the video table (for fields the image table omits)."""
        mode_params = self._mode_params(mode)
        video_params = self._mode_params("video")
        sampling_overrides = self.sampling.generation_default_overrides()
        resolved = {}
        for key, value in values.items():
            if value is None:
                if key in sampling_overrides:
                    value = sampling_overrides[key]
                else:
                    value = mode_params.get(key, video_params.get(key))
            resolved[key] = value
        return resolved

    def _log_envelope_advisory(
        self,
        *,
        is_t2i: bool,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        max_sequence_length: int,
    ) -> None:
        """One advisory line for requests outside the model-card envelope.

        The envelope is documented support, not enforced validation: the
        reference runtime accepts a wider range, so out-of-envelope requests
        run — they just carry no quality claim. Families without a declared
        envelope get no advisory.

        One line per request, not one per rank: every rank runs this code on a
        TP/Ulysses worker.
        """
        if self.rank != 0:
            return
        env = COSMOS3_ENVELOPES.get(self.family)
        if env is None:
            return
        outside = []
        if (height, width) not in env["resolutions"]:
            outside.append(f"{width}x{height} resolution")
        if not is_t2i:
            lo, hi = env["num_frames"]
            if not lo <= num_frames <= hi:
                outside.append(f"num_frames={num_frames} (validated: {lo}-{hi})")
            lo, hi = env["frame_rate"]
            if not lo <= frame_rate <= hi:
                outside.append(f"frame_rate={frame_rate} (validated: {lo}-{hi})")
        if max_sequence_length > env["max_sequence_length"]:
            outside.append(
                f"max_sequence_length={max_sequence_length} (validated: "
                f"<= {env['max_sequence_length']})"
            )
        if outside:
            logger.warning(
                "Request is outside the model-card validated envelope "
                f"({'; '.join(outside)}); generation proceeds but quality may degrade."
            )

    def _init_transformer(self) -> None:
        logger.info("Initializing Cosmos3VFMTransformer")
        model_config = self.pipeline_config.model_configs["transformer"]
        self.transformer = Cosmos3VFMTransformer(model_config)

    # =========================================================================
    # Offloading
    # =========================================================================

    def default_offload_stages(self) -> tuple[tuple[str, ...], ...]:
        """Offload the reasoner and generator towers as separate stages.

        Only invoked when ``cpu_offload_config.enable`` is true (the base class
        short-circuits before calling this). Stages are held in CPU storage
        while inactive and moved onto the pipeline GPU when active.
        """
        return _COSMOS3_DEFAULT_OFFLOAD_STAGES

    def offload_pipeline_components(self) -> dict[str, nn.Module]:
        """Expose the two transformer towers, VAE, and guardrails as offload components.

        Cosmos3 packs both pathways into a single ``transformer`` module, so the
        default ``BasePipeline.offload_pipeline_components`` (which looks for a
        ``transformer.blocks`` ModuleList) does not apply. We expose the heavy
        decoder-layer ModuleLists of each tower individually so they can be
        brought on/off the GPU independently. The opt-in guardrail components wrap
        the underlying safety nn.Modules (loaded on rank 0 only).
        """
        components: dict[str, nn.Module] = {}

        transformer = getattr(self, "transformer", None)
        if transformer is not None:
            language_model = getattr(transformer, "language_model", None)
            reasoner_layers = (
                getattr(language_model, "layers", None) if language_model is not None else None
            )
            if reasoner_layers is not None:
                components[COSMOS3_REASONER_OFFLOAD_COMPONENT] = reasoner_layers

            generator_layers = getattr(transformer, "gen_layers", None)
            if generator_layers is not None:
                components[COSMOS3_GENERATOR_OFFLOAD_COMPONENT] = generator_layers

        vae = getattr(self, PipelineComponent.VAE.value, None)
        if vae is not None:
            components[PipelineComponent.VAE.value] = vae

        # Guardrails (rank 0 only). CosmosSafetyChecker is an nn.Module but its
        # GuardrailRunner children are plain objects, so expose the real safety
        # nn.Modules (Qwen3Guard, RetinaFaceFilter) wrapped in a ModuleList.
        safety_checker = getattr(self, "safety_checker", None)
        if safety_checker is not None:
            for component_name, runner in (
                (COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT, safety_checker.text_guardrail),
                (COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT, safety_checker.video_guardrail),
            ):
                modules = [m for m in runner.models if isinstance(m, nn.Module)]
                if modules:
                    components[component_name] = nn.ModuleList(modules)

        return components

    def extra_offload_component_names(self) -> set[str]:
        # Guardrails load on rank 0 only; treat their names as valid on all ranks
        # so explicit multi-GPU stages don't fail during validation. Other ranks
        # drop them later via the offloader's stage filtering.
        return {COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT, COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT}

    def load_weights(self, weights: dict) -> None:
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            self.transformer.eval()

    def load_standard_components(
        self, checkpoint_dir: str, device: torch.device, skip_components: Optional[list] = []
    ) -> None:
        skip_components = skip_components or []

        # Prompting defaults are checkpoint-declared: distilled conversions
        # carry ``default_use_system_prompt`` in model_index.json (diffusers'
        # distilled blocks default it to True); older checkpoints omit it and
        # keep the historical False.
        model_index_path = os.path.join(checkpoint_dir, "model_index.json")
        if os.path.exists(model_index_path):
            with open(model_index_path) as f:
                model_index = json.load(f)
            self.default_use_system_prompt = bool(
                model_index.get("default_use_system_prompt", self.default_use_system_prompt)
            )
            self.use_native_flow_schedule = bool(
                model_index.get("use_native_flow_schedule", self.use_native_flow_schedule)
            )

        if self.audio_gen and PipelineComponent.SOUND_TOKENIZER not in skip_components:
            logger.info("Loading audio tokenizer...")
            self.audio_tokenizer = (
                LatentAutoEncoderV2.from_pretrained(
                    checkpoint_dir,
                    subfolder=PipelineComponent.SOUND_TOKENIZER,
                )
                .to(device)
                .to(self.dtype)
                .eval()
            )

        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir,
                subfolder="text_tokenizer",
            )

        # Cosmos3 canonical defaults — overwritten if VAE is loaded
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 16

        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading VAE...")
            vae_device = (
                torch.device("cpu")
                if PipelineComponent.VAE.value in self.offloader.requested_components()
                else device
            )
            self.vae = load_wan_vae(
                checkpoint_dir,
                vae_device,
                dtype=torch.bfloat16,
            )

            self.vae_scale_factor_temporal = getattr(
                self.vae.config, "scale_factor_temporal", self.vae_scale_factor_temporal
            )
            self.vae_scale_factor_spatial = getattr(
                self.vae.config, "scale_factor_spatial", self.vae_scale_factor_spatial
            )
            _validate_temporal_compression(self.transformer, self.vae_scale_factor_temporal)
            self.transformer.temporal_compression_factor = self.vae_scale_factor_temporal

        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading scheduler...")
            # The scheduler class comes from the checkpoint: UniPC for base
            # checkpoints, FlowMatchEuler (fixed stochastic schedule) for
            # distilled ones. The policy holds the derived immutable facts.
            self.scheduler = load_scheduler(checkpoint_dir)
            self.sampling = Cosmos3SamplingPolicy.from_scheduler(
                self.scheduler, native_flow_schedule=self.use_native_flow_schedule
            )
            _validate_sampling_recipe(self.family, self.use_native_flow_schedule, self.sampling)
            # Each stream's variants derive from that stream's own instance, so
            # keep both as untouched bases for _scheduler_for().
            self._base_scheduler = self.scheduler
            if self.audio_gen:
                # Separate instance so video and audio scheduler states don't
                # collide (schedulers mutate internal state on every .step()).
                self.audio_scheduler = type(self.scheduler).from_config(self.scheduler.config)
                self._base_audio_scheduler = self.audio_scheduler

            if self.action_gen:
                # Action uses its own scheduler for the same reason as audio.
                self.action_scheduler = type(self.scheduler).from_config(self.scheduler.config)
                self._base_action_scheduler = self.action_scheduler

        # Re-check the env var in case it was changed after initialization like in unit tests.
        guardrails_disabled = os.environ.get("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", "0") == "1"
        global TRTLLM_DISABLE_COSMOS3_GUARDRAILS
        TRTLLM_DISABLE_COSMOS3_GUARDRAILS = guardrails_disabled
        if not TRTLLM_DISABLE_COSMOS3_GUARDRAILS:
            # lazy import
            try:
                from cosmos_guardrail import CosmosSafetyChecker
            except (ImportError, ModuleNotFoundError):
                raise ValueError(
                    "Cosmos Guardrail is not installed. This is in violation of the "
                    "[NVIDIA Open Model License Agreement]"
                    "(https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
                    "Please run the following installation commands or "
                    "explicitly disable guardrails by setting TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1 "
                    "(user is responsible for deploying the model without guardrails). "
                    "- `pip install cosmos_guardrail==0.3.0 && pip uninstall opencv-python`"
                )
            # Guardrails are only evaluated on rank 0; load them only there to avoid
            # dead model weights occupying GPU memory on every other rank.
            if self.rank == 0:
                # the download guardrail checkpoint will bypass CosmosSafetyChecker's checkpoint download.
                # Both will use HF_HOME as the cache directory.
                download_guardrail_checkpoint()
                self.safety_checker = CosmosSafetyChecker()
                self.safety_checker.to(device)

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @property
    def default_warmup_resolutions(self):
        video = self._mode_params("video")
        return [(video["height"], video["width"])]

    @property
    def default_warmup_num_frames(self):
        return [self._mode_params("video")["num_frames"]]

    @property
    def default_warmup_steps(self):
        # Distilled checkpoints only run their fixed schedule length.
        return self.sampling.num_steps(super().default_warmup_steps)

    @property
    def default_generation_params(self):
        """Fields merged by the executor into every request.

        These are the video-mode values — what an unmodified request runs.
        A request that selects another mode re-resolves them in ``infer()``,
        which tells a merged default from a caller-supplied value via
        ``model_fields_set``. Key membership also declares these fields
        supported during request validation. ``flow_shift`` is
        pipeline-internal, not a request field.
        """
        defaults = {k: v for k, v in self._mode_params("video").items() if k != "flow_shift"}
        return {**defaults, **self.sampling.generation_default_overrides()}

    def classify_request_failure(self, exc: BaseException) -> Optional[str]:
        """Cosmos3 rejects unusable request content with ``ValueError`` and
        reports capacity exhaustion as ``MemoryError``, so those map onto the
        response channel's client / capacity classes."""
        return classify_worker_error(exc)

    @property
    def extra_param_specs(self):
        # ``use_system_prompt`` keeps its None default here on purpose: the
        # executor materializes these into every request, so publishing the
        # checkpoint's boolean would destroy "unset" before forward() can
        # resolve it by mode. The checkpoint value is exposed separately as
        # ``default_use_system_prompt``.
        return dict(COSMOS3_EXTRA_SPECS)

    @property
    def ref_slot_specs(self) -> dict[str, RefSlotSpec]:
        return {
            # image (I2V) and video (V2V) are both optional; Cosmos3 also runs
            # T2V with neither.
            "image_reference": RefSlotSpec(
                modality="image",
                roles=[RoleSpec(role="first_frame", min=0, max=1)],
            ),
            "video_reference": RefSlotSpec(
                modality="video",
                roles=[RoleSpec(role="reference", min=0, max=1)],
            ),
        }

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        # Checkpoint-aware guidance: distilled defaults carry a concrete 1.0;
        # base defaults leave it None ("by mode") — warmup runs the video mode.
        defaults = self.default_generation_params
        guidance_scale = defaults["guidance_scale"]
        if guidance_scale is None:
            guidance_scale = self._mode_params("video")["guidance_scale"]
        with torch.no_grad():
            self.forward(
                prompt="warmup",
                negative_prompt="",
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=42,
                max_sequence_length=defaults["max_sequence_length"],
                use_guardrails=False,
                image=None,
                enable_audio=False,
            )

    def _scheduler_for(
        self,
        target_shift: Optional[float],
        use_karras_sigmas: Optional[bool] = None,
        *,
        stream: str = "video",
    ) -> Any:
        """The scheduler for one resolved sampling configuration, built once.

        A scheduler's identity is its resolved ``(flow_shift, karras)`` pair,
        not the request mode: modes that resolve to the same pair share an
        instance, and a pair is never rebuilt once seen. Streams get separate
        instances at the same configuration because schedulers mutate internal
        state on every ``.step()`` — video and audio denoise in lockstep, so
        they must share the knobs but not the object.
        """
        if stream == "audio":
            base = getattr(self, "_base_audio_scheduler", None) or self.audio_scheduler
        elif stream == "action":
            base = getattr(self, "_base_action_scheduler", None) or self.action_scheduler
        else:
            base = getattr(self, "_base_scheduler", None) or self.scheduler
        # Only configurations this checkpoint can resolve to on its own are
        # memoized. ``flow_shift`` is a caller-supplied float with no bounded
        # domain, so caching every value seen would let a client grow the cache
        # for the worker's lifetime; a one-off value builds a scheduler that is
        # discarded with the request instead.
        if target_shift is not None and target_shift not in self._cacheable_flow_shifts():
            return self.sampling.set_flow_shift(
                base, target_shift, use_karras_sigmas=use_karras_sigmas
            )

        cache = getattr(self, "_scheduler_cache", None)
        if cache is None:
            cache = self._scheduler_cache = {}
        key = (target_shift, use_karras_sigmas, stream)
        if key not in cache:
            cache[key] = self.sampling.set_flow_shift(
                base, target_shift, use_karras_sigmas=use_karras_sigmas
            )
        return cache[key]

    def _cacheable_flow_shifts(self) -> frozenset:
        """Flow shifts reachable without a caller override, for this checkpoint.

        Derived rather than fixed so a new family or mode table is picked up
        automatically: the per-mode generation tables, the checkpoint's own
        shift, V2V's stronger shift, and the per-hint transfer presets. Entries
        are still created lazily, so a mode that is never served never builds
        one.

        Transfer's shifts are included even though every hint currently declares
        the same value as V2V: tuning one off that value would otherwise drop it
        out of the cache silently and rebuild a scheduler on every request for
        that hint.
        """
        cached = getattr(self, "_cacheable_flow_shifts_cache", None)
        if cached is not None:
            return cached
        shifts = {COSMOS3_V2V_FLOW_SHIFT, COSMOS3_V2V_DEFAULT_FLOW_SHIFT}
        checkpoint_shift = getattr(self.sampling, "checkpoint_flow_shift", None)
        if checkpoint_shift is not None:
            shifts.add(float(checkpoint_shift))
        for mode in ("video", "image"):
            table = COSMOS3_GENERATION_DEFAULTS.get((self.family, mode)) or {}
            mode_shift = table.get("flow_shift")
            if mode_shift is not None:
                shifts.add(float(mode_shift))
        for hint_defaults in TRANSFER_DEFAULTS.values():
            hint_shift = hint_defaults.get("flow_shift")
            if hint_shift is not None:
                shifts.add(float(hint_shift))
        cached = self._cacheable_flow_shifts_cache = frozenset(shifts)
        return cached

    def _release_scheduler_solver_state(self) -> None:
        """Drop the multistep solver's retained model outputs after a request.

        UniPC keeps ``solver_order`` previous outputs, which are latent-sized --
        tens of MB of device memory that a cached scheduler would otherwise pin
        until its next use. ``set_timesteps`` resets them at the start of every
        request anyway, so this only shortens how long they are held; it frees
        references rather than allocating, and generation is strictly serial, so
        nothing else can be mid-loop on these instances.

        The live schedulers are covered too: a caller-supplied ``flow_shift``
        outside the cacheable set builds a one-off instance that never enters the
        cache, and it stays reachable here until the next request replaces it.
        """
        cached_schedulers = self._scheduler_cache.values()
        live_schedulers = (getattr(self, "scheduler", None), getattr(self, "audio_scheduler", None))
        for scheduler in (*cached_schedulers, *live_schedulers):
            if scheduler is None:
                continue
            order = getattr(getattr(scheduler, "config", None), "solver_order", None)
            if order is None:
                continue
            if getattr(scheduler, "model_outputs", None) is not None:
                scheduler.model_outputs = [None] * order
            if getattr(scheduler, "timestep_list", None) is not None:
                scheduler.timestep_list = [None] * order

    def infer(self, req):
        extra_params = req.params.extra_params or {}
        output_type = extra_params.get("output_type", "video")
        is_t2i = str(output_type).lower() == "image"
        transfer_config = resolve_transfer_config(extra_params, req.params, req.prompt)

        # Caller-assigned values win. Anything still carrying a pipeline
        # default — unset, or merged by the executor from the video table —
        # resolves against this request's own mode exactly once.
        specified = req.params.model_fields_set

        def as_given(field_name):
            value = getattr(req.params, field_name)
            return value if field_name in specified else None

        refs_v = req.params.video_reference
        video = refs_v[0].content if refs_v else None
        if video is not None:
            _validate_video_reference(video)
        is_action = extra_params.get("action_mode") is not None
        if is_action:
            # Action resolves its whole recipe in forward() -- the canvas from
            # the resolution bucket, the frame rate from the embodiment, the
            # sampling recipe from the action table -- so a caller value passes
            # through and an untouched field arrives as None. The embodiment's
            # rate is a trained property, so an action request does not follow
            # the source clip the way a video request below does.
            height = as_given("height")
            width = as_given("width")
            num_inference_steps = as_given("num_inference_steps")
            guidance_scale = as_given("guidance_scale")
            frame_rate = as_given("frame_rate")
        else:
            # Source-derived sizes go in as non-None, so they win over the mode
            # table: _resolve_generation_params only fills what is still None.
            height, width = as_given("height"), as_given("width")
            frame_rate = req.params.frame_rate
            wants_source_size = "height" not in specified and "width" not in specified
            # Timing resolves as a unit. A caller who pinned `num_frames` -- directly,
            # or via `seconds`, which the serving layer already converted at the
            # default rate -- has fixed the output duration; adopting a different
            # frame rate underneath that would silently stretch or compress it.
            wants_source_fps = "frame_rate" not in specified and "num_frames" not in specified
            if wants_source_size or wants_source_fps:
                source = video
                if source is None and transfer_config is not None:
                    # Transfer can run on precomputed controls alone; then the
                    # control clip is what defines the structure to match.
                    source = next(
                        (hint.control for hint in transfer_config.ordered_hints if hint.control),
                        None,
                    )
                # One header read serves both: no GPU, no frame decoded.
                info = video_stream_info(source) if source is not None else None
                if info is not None:
                    if wants_source_size:
                        # Follow the reference's aspect, so a portrait or square
                        # source is not center-cropped into the default landscape
                        # bucket. Skipped when either dimension was named: that
                        # states an intent, and overriding half of it would be
                        # worse than leaving it alone.
                        width, height = find_closest_target_size(info.height, info.width, 720)
                    if wants_source_fps and info.frame_rate is not None:
                        # Emitting an 8 fps source at the 24 fps default would play
                        # it back at 3x speed and misreport its duration to the
                        # text conditioning.
                        frame_rate = info.frame_rate
                    if self.rank == 0:
                        logger.info(
                            f"Cosmos3 following the source: "
                            f"{width}x{height} (WxH) @ {frame_rate} fps"
                        )
            resolved = self._resolve_generation_params(
                "image" if is_t2i else "video",
                height=height,
                width=width,
                num_inference_steps=as_given("num_inference_steps"),
                guidance_scale=as_given("guidance_scale"),
            )
            height = resolved["height"]
            width = resolved["width"]
            num_inference_steps = resolved["num_inference_steps"]
            guidance_scale = resolved["guidance_scale"]
        refs_i = req.params.image_reference

        return self.forward(
            prompt=req.prompt,
            negative_prompt=req.params.negative_prompt,
            image=refs_i[0].content if refs_i else None,
            height=height,
            width=width,
            num_frames=req.params.num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=req.params.seed,
            max_sequence_length=req.params.max_sequence_length,
            frame_rate=frame_rate,
            use_duration_template=extra_params.get(
                "use_duration_template",
                COSMOS3_EXTRA_SPECS["use_duration_template"].default,
            ),
            use_resolution_template=extra_params.get(
                "use_resolution_template",
                COSMOS3_EXTRA_SPECS["use_resolution_template"].default,
            ),
            # None = unset; forward() resolves it by mode / checkpoint.
            use_system_prompt=extra_params.get("use_system_prompt"),
            use_guardrails=extra_params.get("use_guardrails", True),
            enable_audio=extra_params.get("enable_audio", False),
            output_type=output_type,
            video=video,
            condition_video_latent_indexes=extra_params.get("condition_video_latent_indexes"),
            condition_video_keep=extra_params.get("condition_video_keep"),
            flow_shift=extra_params.get("flow_shift"),
            action_mode=extra_params.get("action_mode"),
            domain_name=extra_params.get("domain_name"),
            domain_id=extra_params.get("domain_id"),
            raw_action_dim=extra_params.get("raw_action_dim"),
            action_chunk_size=extra_params.get("action_chunk_size"),
            action=extra_params.get("action"),
            action_resolution=extra_params.get("action_resolution"),
            action_fps=extra_params.get("action_fps"),
            view_point=extra_params.get("view_point", DEFAULT_ACTION_VIEW_POINT),
            transfer_config=transfer_config,
        )

    def _apply_metadata_templates(
        self,
        prompt: str,
        *,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        duration_template: Optional[str] = COSMOS3_DURATION_TEMPLATE,
        resolution_template: Optional[str] = COSMOS3_DEFAULT_RESOLUTION_TEMPLATE,
        force_duration_template: bool = False,
    ) -> str:
        """Append duration and resolution metadata as sentences.

        ``duration_template`` / ``resolution_template`` of ``None`` disables that
        template.  A JSON positive prompt instead gets the metadata injected as
        object fields by ``_format_prompt_with_metadata``; negative prompts always
        come here, matching the reference, so a JSON negative prompt keeps its
        serialized form and gains the sentences after it.
        """
        parts: List[str] = []
        head = prompt.rstrip(".").strip()
        if head:
            parts.append(head)
        if duration_template is not None and (num_frames > 1 or force_duration_template):
            # Fractional on purpose: the reference's text path keeps the exact value
            # and lets the template's own precision render it, unlike its JSON path,
            # which truncates (cosmos-framework _format_prompt_with_template).
            duration = num_frames / frame_rate
            parts.append(duration_template.format(duration=duration, fps=frame_rate).rstrip("."))
        if resolution_template is not None:
            parts.append(resolution_template.format(height=height, width=width).rstrip("."))
        if not parts:
            return ""
        return ". ".join(parts) + "."

    def _format_prompt_with_metadata(
        self,
        prompt: str,
        *,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        duration_template: Optional[str],
        resolution_template: Optional[str],
        force_duration_template: bool = False,
    ) -> str:
        """Apply cosmos-framework-style metadata to plain text or JSON prompts."""
        stripped = prompt.strip()
        if stripped.startswith("{"):
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                data = None
            else:
                if isinstance(data, dict):
                    if duration_template is not None and (
                        num_frames > 1 or force_duration_template
                    ):
                        # Truncated, not rounded, and integer-valued even though the
                        # text template above stays fractional: both mirror the
                        # reference (cosmos-framework _format_json_prompt_with_template).
                        data["duration"] = f"{int(num_frames / frame_rate)}s"
                        data["fps"] = float(frame_rate)
                    else:
                        # A still carries no duration: drop whatever the caller's
                        # JSON declared rather than leaving it stale.
                        data.pop("duration", None)
                        data.pop("fps", None)
                    if resolution_template is not None:
                        data["resolution"] = {"H": int(height), "W": int(width)}
                        data["aspect_ratio"] = _aspect_ratio_bucket(height, width)
                    return json.dumps(data)

        return self._apply_metadata_templates(
            prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            duration_template=duration_template,
            resolution_template=resolution_template,
            force_duration_template=force_duration_template,
        )

    def _resize_and_center_crop_image(
        self, image: PIL.Image.Image, height: int, width: int
    ) -> PIL.Image.Image:
        """Match Cosmos3 reference preprocessing for conditioning images."""
        orig_w, orig_h = image.size
        scaling_ratio = max(width / orig_w, height / orig_h)
        resize_w = int(math.ceil(scaling_ratio * orig_w))
        resize_h = int(math.ceil(scaling_ratio * orig_h))

        image = image.resize((resize_w, resize_h), PIL.Image.Resampling.LANCZOS)

        left = max((resize_w - width) // 2, 0)
        top = max((resize_h - height) // 2, 0)
        return image.crop((left, top, left + width, top + height))

    @nvtx_range("_tokenize_prompt", color="blue")
    def _tokenize_prompt(
        self,
        text: str,
        max_sequence_length: int,
        use_system_prompt: bool = False,
        system_prompt: Optional[str] = None,
    ):
        """Tokenize a prompt using the Qwen2 chat template.

        Returns (input_ids, attention_mask) as [1, S] tensors on device.
        """
        conversations = (
            [{"role": "system", "content": system_prompt or COSMOS3_DEFAULT_SYSTEM_PROMPT}]
            if use_system_prompt
            else []
        )
        conversations.append(
            {"role": "user", "content": text},
        )
        token_ids = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        reserved_tokens = 2
        if max_sequence_length < reserved_tokens:
            raise ValueError(
                f"max_sequence_length must be at least {reserved_tokens}, got {max_sequence_length}"
            )
        token_ids = token_ids[: max_sequence_length - reserved_tokens]
        token_ids.append(self.tokenizer.eos_token_id)  # 151645
        token_ids.append(self.tokenizer.convert_tokens_to_ids("<|vision_start|>"))  # 151652
        seq_len = len(token_ids)

        # Pad to max_sequence_length. TRT's shared denoiser concatenates CFG
        # prompt tensors, so cond/uncond sequence lengths must match.
        pad_len = max_sequence_length - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len
        token_ids = token_ids + [self.tokenizer.pad_token_id or 0] * pad_len

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    # =========================================================================
    # Latent preparation
    # =========================================================================

    @nvtx_range("_prepare_latents", color="blue")
    def _prepare_latents(self, height, width, num_frames, generator):
        num_channels_latents = self.transformer.latent_channel_size
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            1,
            num_channels_latents,
            num_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)

    # =========================================================================
    # I2V latent preparation
    # =========================================================================

    def _encode_conditioning_video(
        self,
        image_tensor: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """VAE-encode a conditioning image as a full-length video.

        The WAN VAE has temporal compression (factor 4), so encoding a single
        frame produces degenerate temporal features.  Following imaginaire4's
        ``build_conditioned_video_batch``, we fill the entire pixel-space video
        with the conditioning image (repeating it across all frames) so the
        temporal encoder sees plausible content everywhere.  The caller then
        keeps only the conditioned latent frame(s) and replaces the rest with
        noise.

        Args:
            image_tensor: [1, 3, H, W] in [-1, 1]
            num_frames: total pixel frames for the video

        Returns:
            [1, C, T_latent, H_latent, W_latent] normalized latent of the
            full conditioning video.
        """
        # Build pixel-space video: repeat the conditioning image across all frames
        # image_tensor: [1, 3, H, W] -> [1, 3, 1, H, W] -> [1, 3, num_frames, H, W]
        video = image_tensor.unsqueeze(2).expand(-1, -1, num_frames, -1, -1).contiguous()
        return self._encode_video_tensor(video)

    def _prepare_latents_i2v(
        self,
        image_tensor: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare initial latents with frame 0 conditioned on the input image.

        The conditioning image is repeated across all pixel frames before VAE
        encoding so the temporal encoder sees plausible content everywhere
        (avoids degenerate single-frame encoding with the WAN VAE's temporal
        compression).  Only frame 0 of the resulting latent is kept clean;
        the rest is replaced with noise.

        Returns:
            latents: [1, C, T_lat, H_lat, W_lat] with frame 0 = image, rest = noise
            velocity_mask: [1, 1, T_lat, 1, 1] with frame 0 = 0, rest = 1
            image_latent: [1, C, 1, H_lat, W_lat] clean frame 0 for re-injection
        """
        C = self.transformer.latent_channel_size
        T_lat = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # Pure noise
        noise = randn_tensor(
            (
                1,
                C,
                T_lat,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            ),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # Encode full conditioning video (image repeated across all frames)
        cond_latent = self._encode_conditioning_video(
            image_tensor,
            num_frames,
        )  # [1, C, T_lat, H_lat, W_lat]

        # Keep only frame 0 for conditioning; replace rest with noise
        image_latent = cond_latent[:, :, 0:1, :, :]  # [1, C, 1, H_lat, W_lat]

        condition_mask = torch.zeros(1, 1, T_lat, 1, 1, device=self.device, dtype=self.dtype)
        condition_mask[:, :, 0, :, :] = 1.0

        latents = condition_mask * cond_latent + (1.0 - condition_mask) * noise

        velocity_mask = 1.0 - condition_mask
        return latents, velocity_mask, image_latent

    def _conditioning_anchor_post_step(self, image_latent: Optional[torch.Tensor]):
        """Per-step re-anchor of the conditioned frame for distilled sampling.

        The distilled FlowMatchEuler step is stochastic: it re-noises every
        position, including the frame the velocity mask holds still, so the
        conditioning frame the model reads as clean context degrades from step
        2 on. Writing the clean latent back after every scheduler step keeps
        it clean (diffusers' distilled loop re-anchors the same way).
        Deterministic UniPC steps never move a zero-velocity frame, so base
        checkpoints need no per-step anchor and keep their exact behavior.

        Returns a ``post_step_fn`` for ``BasePipeline.denoise``, or ``None``
        when no anchoring is needed.
        """
        if not self.sampling.is_distilled or image_latent is None:
            return None

        def post_step_fn(latents: torch.Tensor, extra_stream_latents):
            # In-place: writes one latent frame, no full-tensor copies.
            _assert_anchor_matches(image_latent, latents)
            latents[:, :, 0:1] = image_latent
            return latents, extra_stream_latents

        return post_step_fn

    # =========================================================================
    # VAE decode
    # =========================================================================

    def _decode_latents_raw(self, latents):
        latents = latents.to(self.vae.dtype)

        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            if not hasattr(self, "_latents_mean"):
                self._latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, -1, 1, 1, 1)
                    .to(self.device, self.vae.dtype)
                )
                self._latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, -1, 1, 1, 1)
                    .to(self.device, self.vae.dtype)
                )
            latents = (latents * self._latents_std) + self._latents_mean
        else:
            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
            latents = latents / scaling_factor

        with self.offloader.context_if_requested(PipelineComponent.VAE.value):
            return self.vae.decode(latents, return_dict=False)[0]

    @nvtx_range("_decode_latents", color="blue")
    def _decode_latents(self, latents):
        return postprocess_video_tensor(self._decode_latents_raw(latents))

    # =========================================================================
    # Audio generation
    # =========================================================================

    def decode_audio(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode audio latent tokens back to waveform.

        Args:
            latent: Audio latent tensor of shape (B, C, T).

        Returns:
            Waveform tensor of shape (B, audio_channels, N_samples).
        """
        return self.audio_tokenizer.decode(latent).float()  # [B, audio_channels, N_samples]

    def _condition_frames_to_video_tensor(self, frames: torch.Tensor) -> torch.Tensor:
        """Normalize uint8 ``[T, H, W, C]`` device frames to ``[1, 3, T, H, W]``.

        Same value mapping as ``VideoProcessor.preprocess`` (``[0, 255]`` →
        ``[-1, 1]``), applied to the target-resolution frames the worker
        decode (``decode_video_reference_window``) retains.
        """
        if frames.shape[0] < 1:
            raise ValueError("Cosmos3 condition video must contain at least one frame.")
        x = frames.to(torch.float32).div_(255.0).mul_(2.0).sub_(1.0)
        return x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()

    def _encode_video_tensor(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """VAE-encode a preprocessed pixel video [1, 3, T, H, W]."""
        if video_tensor.ndim == 4:
            video_tensor = video_tensor.unsqueeze(0)
        if video_tensor.ndim != 5 or video_tensor.shape[0] != 1 or video_tensor.shape[1] != 3:
            raise ValueError(
                f"Cosmos3 video tensor must have shape [1, 3, T, H, W], got {tuple(video_tensor.shape)}."
            )

        video = video_tensor.to(device=self.device, dtype=self.vae.dtype)
        with self.offloader.context_if_requested(PipelineComponent.VAE.value):
            latent = self.vae.encode(video).latent_dist.mode()

        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, -1, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            latent = (latent - latents_mean) / latents_std
        else:
            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
            latent = latent * scaling_factor

        return latent.to(self.dtype)

    # =========================================================================
    # Video to video
    # =========================================================================

    def _prepare_latents_v2v(
        self,
        video_tensor: torch.Tensor,
        num_frames: int,
        generator: torch.Generator,
        condition_video_latent_indexes: Iterable[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare V2V latents with explicit clean conditioned latent frames."""
        if video_tensor.ndim == 4:
            video_tensor = video_tensor.unsqueeze(0)
        if video_tensor.ndim != 5 or video_tensor.shape[0] != 1 or video_tensor.shape[1] != 3:
            raise ValueError(
                "Cosmos3 video tensor must have shape [1, 3, T, H, W], "
                f"got {tuple(video_tensor.shape)}."
            )
        if video_tensor.shape[2] < 1:
            raise ValueError("Cosmos3 V2V video tensor must contain at least one frame.")

        C = self.transformer.latent_channel_size
        T_lat = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        H_lat = video_tensor.shape[-2] // self.vae_scale_factor_spatial
        W_lat = video_tensor.shape[-1] // self.vae_scale_factor_spatial
        indexes = _normalize_condition_video_latent_indexes(condition_video_latent_indexes)
        out_of_range = [index for index in indexes if index >= T_lat]
        if out_of_range:
            # Mode-aware bound (num_frames may be a mode-deferred default, so
            # this cannot run at coordinator preflight); client error class.
            raise ValueError(
                "Cosmos3 condition_video_latent_indexes contains indexes outside the latent video: "
                f"indexes={indexes}, latent_frames={T_lat}."
            )

        noise = randn_tensor(
            (1, C, T_lat, H_lat, W_lat),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        condition_pixel_frames = _condition_pixel_frame_count(
            indexes, self.vae_scale_factor_temporal
        )
        condition_video = video_tensor[:, :, :condition_pixel_frames]
        if condition_video.shape[2] < condition_pixel_frames:
            pad = condition_video[:, :, -1:].repeat(
                1, 1, condition_pixel_frames - condition_video.shape[2], 1, 1
            )
            condition_video = torch.cat([condition_video, pad], dim=2)

        cond_latent = self._encode_video_tensor(condition_video)
        expected_prefix = (1, C, max(indexes) + 1, H_lat, W_lat)
        if (
            cond_latent.shape[0] != expected_prefix[0]
            or cond_latent.shape[1] != expected_prefix[1]
            or cond_latent.shape[2] < expected_prefix[2]
            or cond_latent.shape[3:] != expected_prefix[3:]
        ):
            raise ValueError(
                "Cosmos3 V2V condition latent shape mismatch: "
                f"encoded={tuple(cond_latent.shape)}, expected at least {expected_prefix}."
            )

        condition_mask = torch.zeros(1, 1, T_lat, 1, 1, device=self.device, dtype=self.dtype)
        condition_latents = torch.zeros_like(noise)
        for index in indexes:
            condition_mask[:, :, index, :, :] = 1.0
            condition_latents[:, :, index : index + 1] = cond_latent[:, :, index : index + 1]
        latents = condition_mask * condition_latents + (1.0 - condition_mask) * noise
        velocity_mask = 1.0 - condition_mask
        return latents, velocity_mask, condition_latents

    # =========================================================================
    # Action generation
    # =========================================================================

    def _preprocess_action_image(
        self, image: PIL.Image.Image, target_h: int, target_w: int
    ) -> torch.Tensor:
        image = resize_and_pad_action_image(image, target_h, target_w)
        return self.video_processor.preprocess(image, height=target_h, width=target_w)

    def _preprocess_action_first_frame(
        self, image: Any, video: Any, target_h: int, target_w: int
    ) -> torch.Tensor:
        """Conditioning frame for policy / forward_dynamics as ``[1, 3, H, W]``.

        Either source is accepted: an image goes through PIL, video bytes take
        frame 0 off NVDEC. Both land on the padded canvas, so the two entry
        points produce the same conditioning for the same picture.
        """
        if image is not None:
            return self._preprocess_action_image(pil_to_rgb(image), target_h, target_w)
        if not isinstance(video, bytes):
            raise ValueError(
                "Cosmos3 action conditioning requires an image or encoded MP4/AVI "
                f"bytes, got {type(video).__name__}."
            )
        frames_u8 = decode_video_reference_window(
            video,
            first_frame=0,
            last_frame=0,
            target_h=target_h,
            target_w=target_w,
            device=self.device,
            resize="fit",
        )
        return self._condition_frames_to_video_tensor(frames_u8).squeeze(2)

    def _prepare_latents_action_video(
        self,
        video_tensor: torch.Tensor,
        mode: str,
        num_frames: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        C = self.transformer.latent_channel_size
        T_lat = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        H_lat = video_tensor.shape[-2] // self.vae_scale_factor_spatial
        W_lat = video_tensor.shape[-1] // self.vae_scale_factor_spatial

        noise = randn_tensor(
            (1, C, T_lat, H_lat, W_lat),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        cond_latent = self._encode_video_tensor(video_tensor)
        if cond_latent.shape[2:] != noise.shape[2:]:
            raise ValueError(
                "Cosmos3 action video latent shape mismatch: "
                f"encoded={tuple(cond_latent.shape)}, expected={tuple(noise.shape)}."
            )
        condition_mask = build_vision_condition_mask(
            mode,
            num_frames,
            self.vae_scale_factor_temporal,
            device=self.device,
            dtype=self.dtype,
        )
        latents = condition_mask * cond_latent + (1.0 - condition_mask) * noise
        velocity_mask = 1.0 - condition_mask
        return latents, velocity_mask, cond_latent

    def _prepare_action_latents(
        self,
        *,
        mode: str,
        action_chunk_size: int,
        raw_action_dim: Optional[int],
        generator: torch.Generator,
        action_input: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return prepare_action_latents(
            mode=mode,
            action_chunk_size=action_chunk_size,
            raw_action_dim=raw_action_dim,
            action_dim=int(self.transformer.action_dim),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
            action_input=action_input,
        )

    # =========================================================================
    # Transfer
    # =========================================================================

    def _prepare_transfer_latents(
        self,
        target_video: torch.Tensor,
        current_conditional_frames: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition_latents = self._encode_video_tensor(target_video)
        noise = randn_tensor(
            condition_latents.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        condition_mask = torch.zeros(
            1,
            1,
            condition_latents.shape[2],
            1,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        if current_conditional_frames > 0:
            latent_frames = (current_conditional_frames - 1) // self.vae_scale_factor_temporal + 1
            condition_mask[:, :, :latent_frames] = 1.0
        latents = condition_mask * condition_latents + (1.0 - condition_mask) * noise
        velocity_mask = 1.0 - condition_mask
        return latents, velocity_mask, condition_mask * condition_latents

    # =========================================================================
    # Forward (main generation entry point)
    # =========================================================================

    @nvtx_range("Cosmos3OmniMoTPipeline.forward")
    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[str] = None,
        image: Optional[Union[PIL.Image.Image, torch.Tensor, bytes]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = 42,
        max_sequence_length: Optional[int] = None,
        frame_rate: Optional[float] = None,
        use_duration_template: bool = COSMOS3_EXTRA_SPECS["use_duration_template"].default,
        use_resolution_template: bool = COSMOS3_EXTRA_SPECS["use_resolution_template"].default,
        use_system_prompt: Optional[bool] = None,
        use_guardrails: bool = COSMOS3_EXTRA_SPECS["use_guardrails"].default,
        enable_audio: bool = COSMOS3_EXTRA_SPECS["enable_audio"].default,
        output_type: str = COSMOS3_EXTRA_SPECS["output_type"].default,
        video: bytes | None = None,  # encoded MP4/AVI reference (V2V and action)
        condition_video_latent_indexes: Iterable[int] | None = None,
        condition_video_keep: str | None = None,
        flow_shift: Optional[float] = None,
        action_mode: Optional[str] = None,
        domain_name: Optional[str] = None,
        domain_id: Optional[int] = None,
        raw_action_dim: Optional[int] = None,
        action_chunk_size: Optional[int] = None,
        action: Any = None,
        action_resolution: Optional[int] = None,
        action_fps: Optional[float] = None,
        view_point: Optional[str] = DEFAULT_ACTION_VIEW_POINT,
        transfer_config: Optional[Cosmos3TransferConfig] = None,
    ):
        """Run one generation. ``infer()`` is the resolved entry point.

        Production requests arrive through ``infer()`` with fully resolved
        values; unset (None) numeric parameters resolve here from the same
        per-variant mode tables, so direct internal callers get
        checkpoint-appropriate values (including the fixed distilled
        steps/guidance).

        ``use_system_prompt=None`` means "unset": V2V always uses the system
        prompt, and every other mode takes the checkpoint-declared default, so
        warmup and other direct callers build the same prompt as served
        requests.
        """
        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        use_guardrails = use_guardrails and not TRTLLM_DISABLE_COSMOS3_GUARDRAILS

        normalized_action_mode = normalize_action_mode(action_mode)
        do_action = normalized_action_mode is not None
        if do_action and not self.action_gen:
            raise ValueError(
                "Cosmos3 action generation was requested, but this checkpoint "
                "does not enable action_gen."
            )
        if do_action and enable_audio:
            raise ValueError("Cosmos3 does not support joint action and audio generation.")

        # Text-to-image mode: same checkpoint/forward path as T2V, but a single
        # latent frame, image-flavored prompt templates, flow_shift=3.0, a CFG
        # guidance interval, and an image (rather than video) output.
        output_type = str(output_type).lower()
        if output_type not in ("video", "image"):
            raise ValueError(f"output_type must be 'video' or 'image', got {output_type!r}.")
        is_t2i = output_type == "image"

        mode_params = self._mode_params(output_type)
        if do_action:
            # The embodiment resolves the canvas, clip length and frame rate
            # below; only the sampling recipe and the text budget come from
            # tables (distilled overrides still win inside the resolver).
            resolved = self._resolve_generation_params(
                "action",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
            )
            num_inference_steps = resolved["num_inference_steps"]
            guidance_scale = resolved["guidance_scale"]
            max_sequence_length = resolved["max_sequence_length"]
        else:
            resolved = self._resolve_generation_params(
                output_type,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                frame_rate=frame_rate,
            )
            height = resolved["height"]
            width = resolved["width"]
            num_frames = resolved["num_frames"]
            num_inference_steps = resolved["num_inference_steps"]
            guidance_scale = resolved["guidance_scale"]
            max_sequence_length = resolved["max_sequence_length"]
            frame_rate = resolved["frame_rate"]

        self.sampling.validate_request(num_inference_steps, guidance_scale)

        # Skipped for action: its dims resolve from the embodiment below, and
        # the envelope is a video/image model-card claim.
        if not do_action:
            self._log_envelope_advisory(
                is_t2i=is_t2i,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                max_sequence_length=max_sequence_length,
            )

        if image is not None and video is not None:
            raise ValueError(
                "Cosmos3 generation supports text-only, text + image, "
                "or text + video input, but not both image and video."
            )
        if is_t2i and video is not None:
            raise ValueError(
                "Cosmos3 video-to-video generation is supported only for video outputs."
            )
        # Action reads its reference through the same `video` bytes, but it is
        # not V2V: the reference is an observation, not a clip to continue. Left
        # in, an action request's prompt would depend on whether the caller
        # passed the same frame as an image or as a one-frame clip.
        is_v2v = video is not None and not is_t2i and not do_action
        if use_system_prompt is None:
            # V2V always wants it; otherwise the checkpoint declares the default.
            # Transfer opts out for reference parity (vllm-omni
            # `_forward_transfer` defaults it False).
            if transfer_config is not None:
                use_system_prompt = False
            else:
                use_system_prompt = is_v2v or self.default_use_system_prompt
        else:
            use_system_prompt = bool(use_system_prompt)
        if transfer_config is not None:
            if is_t2i:
                raise ValueError("Cosmos3 transfer inference is supported only for video outputs.")
            if enable_audio:
                raise ValueError(
                    "Cosmos3 transfer inference cannot be combined with sound generation."
                )
            if image is not None:
                # _forward_transfer takes no image: structure comes from the
                # control hints and the first chunk conditions on `video`. Say
                # so rather than dropping the reference silently.
                raise ValueError(
                    "Cosmos3 transfer inference cannot be combined with an image reference; "
                    "pass the conditioning clip as the 'video' extra param instead."
                )

        guidance_interval = None
        resolved_action_fps: Optional[float] = None
        if is_t2i:
            if image is not None:
                raise ValueError(
                    "Cosmos3 text-to-image (output_type='image') does not accept an image input."
                )
            if do_action:
                raise ValueError("Cosmos3 action generation does not support output_type='image'.")
            num_frames = 1
            # T2I force-disables audio instead of rejecting it, so an image
            # request never trips the audio-weight presence check below.
            enable_audio = False
            guidance_interval = mode_params["guidance_interval"]

        if do_action:
            # num_frames is derived from the action chunk, never taken from the
            # request: both references fix it at chunk_size + 1 (diffusers
            # rejects a caller-supplied num_frames for action runs outright).
            action_cfg = resolve_domain_action_config(
                domain_name=domain_name,
                domain_id=domain_id,
                raw_action_dim=raw_action_dim,
                action_chunk_size=action_chunk_size,
                action_resolution=action_resolution,
                frame_rate=frame_rate,
                action_fps=action_fps,
            )
            if self.rank == 0:
                for warning in action_cfg["warnings"]:
                    logger.warning(warning)
                if action_cfg["preset_key"] is not None:
                    logger.info(
                        f"Cosmos3 action domain preset {action_cfg['preset_key']!r}: "
                        f"raw_action_dim={action_cfg['raw_action_dim']}, "
                        f"action_chunk_size={action_cfg['action_chunk_size']}, "
                        f"action_resolution={action_cfg['action_resolution']}, "
                        f"frame_rate={action_cfg['frame_rate']:.1f}, "
                        f"action_fps={action_cfg['action_fps']:.1f}, "
                        f"num_frames={action_cfg['num_frames']}"
                    )

            raw_action_dim = action_cfg["raw_action_dim"]
            action_chunk_size = action_cfg["action_chunk_size"]
            action_resolution = action_cfg["action_resolution"]
            num_frames = action_cfg["num_frames"]
            frame_rate = action_cfg["frame_rate"]
            resolved_action_fps = action_cfg["action_fps"]
            enable_audio = False

        # Flow shift is a mode table fact unless the request overrides it, and
        # V2V additionally wants the uniform sigma grid. Both streams take the
        # same knobs so video and audio never step on different schedules.
        mode_shift = mode_params.get("flow_shift")
        if mode_shift is None:
            mode_shift = self.sampling.checkpoint_flow_shift
        if is_v2v:
            # V2V wants a stronger shift and the uniform sigma schedule.
            target_shift = COSMOS3_V2V_FLOW_SHIFT if flow_shift is None else flow_shift
            target_karras = False
        else:
            target_shift = mode_shift if flow_shift is None else flow_shift
            target_karras = None
        if transfer_config is None:
            # Transfer applies the hint's own shift inside _forward_transfer, so
            # rebuilding the schedulers here would only be undone a few lines later.
            # Action never arrives with a transfer_config (the two are mutually
            # exclusive), so its stream is always rebuilt here.
            self.scheduler = self._scheduler_for(target_shift, target_karras)
            if getattr(self, "audio_scheduler", None) is not None:
                self.audio_scheduler = self._scheduler_for(
                    target_shift, target_karras, stream="audio"
                )
            if getattr(self, "action_scheduler", None) is not None:
                self.action_scheduler = self._scheduler_for(
                    target_shift, target_karras, stream="action"
                )

        if self.rank == 0:
            logger.info(
                f"Cosmos3 generation dims: {width}x{height} (WxH), num_frames={num_frames}, "
                f"num_inference_steps={num_inference_steps}, guidance_scale={guidance_scale:.2f}, "
                f"frame_rate={frame_rate:.1f}"
            )

        # Weight-presence guard, not workflow policy: the request explicitly
        # asks for audio, but the checkpoint ships no audio tower. Silently
        # returning a silent video would hide the capability limit.
        if enable_audio and not self.audio_gen:
            raise ValueError(
                "enable_audio=True, but this checkpoint has no audio tower "
                "(transformer config declares sound_gen=false). Drop enable_audio "
                "or use an audio-capable Cosmos3 checkpoint."
            )

        if resolved_action_fps is None:
            resolved_action_fps = frame_rate

        action_source_h = action_source_w = None
        if do_action:
            if isinstance(image, torch.Tensor) or isinstance(video, torch.Tensor):
                raise ValueError(
                    "Cosmos3 action generation does not support tensor image/video inputs; "
                    "pass a PIL image or image path, or encoded MP4/AVI video bytes."
                )
            # Header probe for video bytes, direct measure for an image: the
            # canvas is the bucket closest to the source's shape, so the size
            # has to be known before anything is decoded at it.
            #
            # This is the first per-rank read of the reference, so it converges
            # like the decode below: a missing file or an unreachable URL on one
            # rank must not leave the others walking into the collectives.
            probe_error: Optional[Exception] = None
            try:
                if image is not None and not isinstance(image, PIL.Image.Image):
                    # Resolve once: the bundled prompts point at https frames,
                    # and the probe and the conditioning frame both read it.
                    image = pil_to_rgb(image)
                action_source_h, action_source_w = action_reference_size(
                    action_mode=normalized_action_mode,
                    image=image,
                    video=video,
                )
            except Exception as exc:
                probe_error = exc
            synchronize_media_prepare_status(probe_error)
            height, width = resolve_action_size(
                height, width, action_source_h, action_source_w, action_resolution
            )

        if self.rank == 0:
            logger.info(
                f"Cosmos3 generation dims: {width}x{height} (WxH), num_frames={num_frames}, "
                f"num_inference_steps={num_inference_steps}, guidance_scale={guidance_scale:.2f}, "
                f"frame_rate={frame_rate:.1f}"
            )
            if do_action:
                logger.info(
                    f"Cosmos3 action dims: action_chunk_size={action_chunk_size}, "
                    f"action_resolution={action_resolution}, "
                    f"action_fps={resolved_action_fps:.1f}, "
                    f"source={action_source_w}x{action_source_h} "
                    f"(aspect {action_source_w / action_source_h:.3f})"
                )

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        if batch_size > 1:
            # TODO: support batch generation
            raise ValueError("Batch generation is not supported for Cosmos3")

        # Validate image input — only single image is supported for batch generation
        if image is not None and not isinstance(image, (PIL.Image.Image, torch.Tensor, bytes)):
            raise ValueError(
                f"`image` must be a PIL.Image, torch.Tensor, or encoded bytes, "
                f"got {type(image)}. Batch of different images is not supported; "
                f"use a single image with multiple prompts instead."
            )

        # Text guardrail — check both positive and user-supplied negative prompts.
        # None negative_prompt means the empty default will be used (safe); skip it.
        text_blocked = torch.zeros((), device=self.device, dtype=torch.int32)
        if self.rank == 0 and use_guardrails and self.safety_checker is not None:
            prompts_to_check = list(prompt)
            if negative_prompt is not None:
                prompts_to_check.append(negative_prompt)
            with self.offloader.context_if_requested(COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT):
                for p in prompts_to_check:
                    is_safe = self.safety_checker.check_text_safety(p)
                    if not is_safe:
                        logger.warning("Text guardrail blocked prompt")
                        text_blocked.fill_(1)
                        break

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(text_blocked, src=0)

        if text_blocked.item():
            return PipelineOutput()

        if transfer_config is not None:
            return self._forward_transfer(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                max_frames=transfer_config.max_frames,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                use_system_prompt=use_system_prompt,
                use_duration_template=False,
                use_resolution_template=False,
                seed=seed,
                frame_rate=frame_rate,
                num_frames=num_frames,
                use_guardrails=use_guardrails,
                timer=timer,
                transfer_config=transfer_config,
                video=video,
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)

        if negative_prompt is None:
            negative_prompt = default_negative_prompt(output_type)

        if do_action:
            # Action checkpoints were trained on a structured JSON caption that
            # already carries duration/fps/resolution/aspect_ratio, so the flat
            # templates are skipped here and the negative prompt stays verbatim.
            prompt = [
                build_action_json_prompt(
                    p,
                    view_point=view_point,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    height=height,
                    width=width,
                )
                for p in prompt
            ]
        else:
            # Positive prompt: forward duration/resolution templates.  T2I has no
            # duration concept (single image) and uses the image-flavored
            # resolution template.
            use_duration_template = use_duration_template and not is_t2i
            dur_tmpl = COSMOS3_DURATION_TEMPLATE if use_duration_template else None
            if use_resolution_template:
                res_tmpl = (
                    COSMOS3_IMAGE_RESOLUTION_TEMPLATE
                    if is_t2i
                    else COSMOS3_DEFAULT_RESOLUTION_TEMPLATE
                )
            else:
                res_tmpl = None

            # Negative prompt: mirror positive metadata (cosmos-framework CLI default
            # when ``negative_prompt_keep_metadata`` promotes mode to ``same``).
            # Always the plain-text templates, never the JSON field injection the
            # positive branch uses: the reference appends these sentences to the
            # negative prompt whether or not it is a JSON object, so a JSON negative
            # prompt ends up as the serialized object followed by the sentences.
            negative_prompt = self._apply_metadata_templates(
                negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                duration_template=dur_tmpl,
                resolution_template=res_tmpl,
                force_duration_template=False,
            )

            prompt = [
                self._format_prompt_with_metadata(
                    p,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    duration_template=dur_tmpl,
                    resolution_template=res_tmpl,
                )
                for p in prompt
            ]
        logger.info(f"Prompt with metadata: '{prompt}'")

        prompt = prompt[0]

        # 1. Tokenize prompts (no separate text encoder — transformer embeds internally)
        logger.info("Tokenizing prompts...")
        system_prompt = COSMOS3_T2I_SYSTEM_PROMPT if is_t2i else COSMOS3_DEFAULT_SYSTEM_PROMPT
        cond_ids, cond_mask = self._tokenize_prompt(
            prompt, max_sequence_length, use_system_prompt, system_prompt=system_prompt
        )
        uncond_ids, uncond_mask = self._tokenize_prompt(
            negative_prompt, max_sequence_length, use_system_prompt, system_prompt=system_prompt
        )

        # 2. Prepare latents
        condition_latents = None
        image_latent = None
        velocity_mask = None
        action_latents = None
        action_velocity_mask = None
        action_condition_latents = None
        action_domain_id = None
        action_frame_offset = 1
        resolved_raw_action_dim = raw_action_dim

        if do_action:
            if action_chunk_size not in {num_frames, num_frames - 1}:
                raise ValueError(
                    "Cosmos3 num_frames must equal action_chunk_size or action_chunk_size + 1."
                )
            action_domain_id = resolve_domain_id(
                domain_id=domain_id,
                domain_name=domain_name,
                require_explicit=True,
            )
            num_domains = getattr(self.transformer, "num_embodiment_domains", None)
            if num_domains is not None and not 0 <= action_domain_id < num_domains:
                raise ValueError(
                    f"Cosmos3 action domain_id must be in [0, {num_domains}), "
                    f"got {action_domain_id}."
                )
            action_frame_offset = action_start_frame_offset(
                normalized_action_mode, action_chunk_size, num_frames
            )

            if normalized_action_mode == ACTION_MODE_INVERSE_DYNAMICS:
                if not isinstance(video, bytes):
                    raise ValueError(
                        "Cosmos3 inverse_dynamics requires encoded MP4/AVI bytes "
                        f"(the 'video' extra-param contract), got {type(video).__name__}."
                    )
                prepare_error: Optional[Exception] = None
                try:
                    source_info = video_stream_info(video)
                    source_frame_rate = source_info.frame_rate if source_info else None
                    frame_step = action_reference_frame_step(source_frame_rate, frame_rate)
                    if self.rank == 0:
                        if frame_step > 1:
                            logger.info(
                                f"Cosmos3 action reference: {source_frame_rate} fps source "
                                f"thinned to {frame_rate} fps, keeping every {frame_step} "
                                f"frames of {(num_frames - 1) * frame_step + 1}"
                            )
                        elif source_frame_rate is not None and source_frame_rate < frame_rate:
                            logger.warning(
                                f"Cosmos3 action reference is {source_frame_rate} fps but "
                                f"{normalized_action_mode} expects {frame_rate} fps: frames are "
                                "further apart than the model was trained on and cannot be "
                                "thinned to match. Re-encode the reference at the higher rate, "
                                "or pass frame_rate explicitly to accept this spacing."
                            )
                    # "fit" rather than the V2V default: an action reference is
                    # padded to the canvas, never cropped to it, because the
                    # gripper and target sit at the frame edge.
                    frames_u8 = decode_video_reference_window(
                        video,
                        first_frame=0,
                        last_frame=(num_frames - 1) * frame_step,
                        target_h=height,
                        target_w=width,
                        device=self.device,
                        resize="fit",
                        frame_step=frame_step,
                    )
                    if frames_u8.shape[0] < num_frames:
                        raise ValueError(
                            f"Cosmos3 inverse_dynamics requires {num_frames} frames at "
                            f"{frame_rate} fps; a {source_frame_rate} fps reference supplies "
                            f"{frames_u8.shape[0]} once thinned by {frame_step} "
                            f"({(num_frames - 1) * frame_step + 1} source frames needed)."
                        )
                    video_tensor = self._condition_frames_to_video_tensor(frames_u8)
                    del frames_u8
                    latents, velocity_mask, condition_latents = self._prepare_latents_action_video(
                        video_tensor,
                        normalized_action_mode,
                        num_frames,
                        generator,
                    )
                    del video_tensor
                except Exception as exc:
                    prepare_error = exc
                # Every rank decodes independently; converge before the
                # transformer's collectives so a failure cannot hang the job.
                synchronize_media_prepare_status(prepare_error)
            else:
                prepare_error = None
                try:
                    image_tensor = self._preprocess_action_first_frame(image, video, height, width)
                    if image_tensor.ndim == 4:
                        video_tensor = (
                            image_tensor.unsqueeze(2)
                            .expand(-1, -1, num_frames, -1, -1)
                            .contiguous()
                        )
                    else:
                        video_tensor = image_tensor
                    latents, velocity_mask, condition_latents = self._prepare_latents_action_video(
                        video_tensor,
                        normalized_action_mode,
                        num_frames,
                        generator,
                    )
                except Exception as exc:
                    prepare_error = exc
                synchronize_media_prepare_status(prepare_error)
                image_latent = None

            (
                action_latents,
                action_velocity_mask,
                action_condition_latents,
                resolved_raw_action_dim,
            ) = self._prepare_action_latents(
                mode=normalized_action_mode,
                action_chunk_size=action_chunk_size,
                raw_action_dim=raw_action_dim,
                generator=generator,
                action_input=action,
            )
        elif image is not None:
            prepare_error: Optional[Exception] = None
            try:
                if isinstance(image, bytes):
                    image = _load_reference_image(image)

                if isinstance(image, PIL.Image.Image):
                    image = image.convert("RGB")
                    image = self._resize_and_center_crop_image(image, height=height, width=width)
                    image = self.video_processor.preprocess(
                        image,
                        height=height,
                        width=width,
                    )

                latents, velocity_mask, image_latent = self._prepare_latents_i2v(
                    image, height=height, width=width, num_frames=num_frames, generator=generator
                )
            except Exception as exc:
                prepare_error = exc
            # Same convergence as the V2V branch: every rank loads the image
            # independently, so a rank that failed while others entered the
            # transformer's collectives would hang the job.
            synchronize_media_prepare_status(prepare_error)
        elif video is not None:
            prepare_error: Optional[Exception] = None
            try:
                condition_video_latent_indexes = _normalize_condition_video_latent_indexes(
                    condition_video_latent_indexes
                )
                # Bound-check the indexes against the OUTPUT latent length
                # before any window math: an out-of-range index would
                # otherwise size the decode ring (keep="last" decodes to EOF
                # through it) from a request that is deterministically
                # invalid.
                num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
                out_of_range = [i for i in condition_video_latent_indexes if i >= num_latent_frames]
                if out_of_range:
                    raise ValueError(
                        f"Cosmos3 condition_video_latent_indexes {out_of_range} are out "
                        f"of range for a {num_frames}-frame output "
                        f"({num_latent_frames} latent frames)."
                    )
                if isinstance(video, bytes):
                    window = _condition_pixel_frame_count(
                        condition_video_latent_indexes, self.vae_scale_factor_temporal
                    )
                    # The conditioning window is a Cosmos3 constraint: it is
                    # derived from indexes into the *output* latent timeline,
                    # already bound-checked above. The decoder just returns
                    # the frames asked for. "last" is a negative range, which
                    # costs a decode to EOS -- the caller's choice to make.
                    if _normalize_condition_video_keep(condition_video_keep) == "first":
                        first_frame, last_frame = 0, window - 1
                    else:
                        first_frame, last_frame = -window, -1
                    frames_u8 = decode_video_reference_window(
                        video,
                        first_frame=first_frame,
                        last_frame=last_frame,
                        target_h=height,
                        target_w=width,
                        device=self.device,
                    )
                else:
                    raise ValueError(
                        "Cosmos3 V2V reference must be encoded MP4/AVI bytes "
                        f"(the 'video' extra-param contract), got "
                        f"{type(video).__name__}."
                    )
                condition_pixels = self._condition_frames_to_video_tensor(frames_u8)
                del frames_u8

                if self.rank == 0:
                    logger.info(
                        f"Cosmos3 V2V conditioning: frames={condition_pixels.shape[2]}, "
                        f"latent_indexes={condition_video_latent_indexes}"
                    )
                latents, velocity_mask, condition_latents = self._prepare_latents_v2v(
                    condition_pixels,
                    num_frames=num_frames,
                    generator=generator,
                    condition_video_latent_indexes=condition_video_latent_indexes,
                )
                # The VAE-encoded condition latents are all the denoise loop
                # needs; drop the decoded pixels before the long generation.
                del condition_pixels
            except Exception as exc:
                prepare_error = exc
            # Per-rank decode/prepare can fail non-uniformly (NVDEC init,
            # corrupt stream, allocation); converge all ranks on one outcome
            # before any model collective so healthy ranks cannot hang.
            synchronize_media_prepare_status(prepare_error)
        else:
            latents = self._prepare_latents(height, width, num_frames, generator)

        # Compute video shape in latent space
        T_latent = latents.shape[2]
        H_latent = latents.shape[3]
        W_latent = latents.shape[4]
        video_shape = (T_latent, H_latent, W_latent)

        # 3. Set up scheduler
        self.sampling.set_timesteps(self.scheduler, num_inference_steps, device=self.device)
        if do_action:
            self.sampling.set_timesteps(
                self.action_scheduler, num_inference_steps, device=self.device
            )

        # 3b. Audio noise init — latent length matches diffusers Cosmos3OmniPipeline.prepare_latents.
        do_audio = enable_audio and self.audio_gen and hasattr(self, "audio_tokenizer")
        audio_latents = None
        if do_audio:
            audio_cfg = self.audio_tokenizer.model_config
            n_audio_samples = int(num_frames / frame_rate * audio_cfg["sampling_rate"])
            hop_size = math.prod(audio_cfg["dec_strides"])
            T_audio = (n_audio_samples + hop_size - 1) // hop_size
            audio_latents = randn_tensor(
                (1, self.transformer.audio_dim, T_audio),
                generator=generator,
                device=self.device,
                dtype=latents.dtype,
            )
            # Audio uses the same scheduler type/config as video.
            self.sampling.set_timesteps(
                self.audio_scheduler, num_inference_steps, device=self.device
            )

        # 4. Build forward_fn for the denoise loop
        action_domain_ids_tensor = None
        if do_action and action_domain_id is not None:
            action_domain_ids_tensor = torch.tensor(
                [action_domain_id], dtype=torch.long, device=self.device
            )

        def forward_fn(
            latent_input,
            extra_stream_latents,
            step_index,
            timestep,
            encoder_hidden_states,
            extra_tensors,
        ):
            """Cosmos3 forward function for BasePipeline.denoise().

            Since Cosmos3 embeds text internally, we pass token IDs via extra_tensors
            rather than through encoder_hidden_states.
            """
            current_audio = extra_stream_latents.get("audio") if extra_stream_latents else None
            current_action = extra_stream_latents.get("action") if extra_stream_latents else None

            action_domain_ids = action_domain_ids_tensor
            if (
                action_domain_ids is not None
                and current_action is not None
                and action_domain_ids.shape[0] == 1
                and current_action.shape[0] > 1
            ):
                action_domain_ids = action_domain_ids.expand(current_action.shape[0])

            result = self.transformer(
                hidden_states=latent_input,
                timestep=timestep / self.scheduler.config.num_train_timesteps,
                raw_timestep=timestep,
                text_ids=extra_tensors["text_ids"],
                text_mask=extra_tensors["text_mask"],
                video_shape=video_shape,
                fps=frame_rate,
                noisy_frame_mask=velocity_mask,
                audio_latents=current_audio,
                offload_context=self.offloader.context_if_requested,
                action_latents=current_action,
                action_domain_ids=action_domain_ids,
                action_noisy_mask=action_velocity_mask,
                action_start_frame_offset=action_frame_offset,
                action_fps=resolved_action_fps,
            )

            video_noise_pred = result.video
            audio_noise_pred = result.audio
            action_noise_pred = result.action

            if velocity_mask is not None:
                video_noise_pred = video_noise_pred * velocity_mask

            if action_noise_pred is not None:
                if action_velocity_mask is not None:
                    action_noise_pred = action_noise_pred * action_velocity_mask
                if (
                    resolved_raw_action_dim is not None
                    and 0 < resolved_raw_action_dim < action_noise_pred.shape[-1]
                ):
                    action_noise_pred = action_noise_pred.clone()
                    action_noise_pred[..., resolved_raw_action_dim:] = 0
                return video_noise_pred, {"action": action_noise_pred}

            if audio_noise_pred is not None:
                return video_noise_pred, {"audio": audio_noise_pred}
            return video_noise_pred

        def post_step_fn(step_latents, step_extra_stream_latents):
            # V2V (and action's clean vision frames): re-impose the condition
            # latents after every scheduler step. I2V deliberately keeps its
            # pre-existing behavior -- velocity mask during the loop, one
            # write-back after it -- so this does not alter I2V denoising.
            if velocity_mask is not None and condition_latents is not None:
                step_latents = (
                    velocity_mask * step_latents + (1.0 - velocity_mask) * condition_latents
                )
            if (
                action_velocity_mask is not None
                and action_condition_latents is not None
                and step_extra_stream_latents is not None
                and "action" in step_extra_stream_latents
            ):
                action_key = step_extra_stream_latents["action"]
                step_extra_stream_latents["action"] = (
                    action_velocity_mask * action_key
                    + (1.0 - action_velocity_mask) * action_condition_latents
                )
            return step_latents, step_extra_stream_latents

        # 5. Build CFG tensors — text_ids and text_mask need to be split for CFG
        #    BasePipeline.denoise batches [uncond, cond] when guidance_scale > 1
        #    We pass text IDs/masks through extra_cfg_tensors so they get split correctly
        extra_cfg_tensors = {
            "text_ids": (cond_ids, uncond_ids),
            "text_mask": (cond_mask, uncond_mask),
        }

        self.transformer.reset_cache()

        # 6. Denoise
        timer.mark_denoise_start()
        extra_streams = None
        if do_action:
            extra_streams = {"action": (action_latents, self.action_scheduler)}
        elif do_audio:
            extra_streams = {"audio": (audio_latents, self.audio_scheduler)}
        # FUTURE(action+audio): merge both keys; extend forward_fn return dict and post_step_fn.
        should_pin_condition_latents = condition_latents is not None and velocity_mask is not None
        denoise_result = self.denoise(
            latents=latents,
            scheduler=self.scheduler,
            prompt_embeds=cond_ids,  # placeholder — actual conditioning via extra_cfg_tensors
            neg_prompt_embeds=uncond_ids,
            guidance_scale=guidance_scale,
            forward_fn=forward_fn,
            extra_cfg_tensors=extra_cfg_tensors,
            extra_streams=extra_streams,
            guidance_interval=guidance_interval,
            # V2V and action pin the conditioning latents; distilled I2V
            # re-anchors the conditioning frame. A request carries an image or
            # a video, never both, so at most one of these applies.
            post_step_fn=(
                post_step_fn
                if (do_action or should_pin_condition_latents)
                else self._conditioning_anchor_post_step(image_latent)
            ),
            scheduler_step_kwargs=self.sampling.scheduler_step_kwargs(generator),
        )

        if extra_streams is not None:
            latents, extra_latents = denoise_result
            audio_latents = extra_latents.get("audio")
            if do_action:
                action_latents = extra_latents.get("action")
        else:
            latents = denoise_result
            audio_latents = None

        self._release_scheduler_solver_state()

        timer.mark_post_start()

        # 7. Decode video
        logger.info("Decoding video...")
        decode_start = time.time()

        if image_latent is not None:
            # In-place: the loop output is consumed only by the decode below.
            _assert_anchor_matches(image_latent, latents)
            latents[:, :, 0:1] = image_latent

        video = self.decode_latents(latents, self._decode_latents)

        # 7b. Decode audio
        waveform = None
        if do_audio and audio_latents is not None:
            logger.info("Decoding audio...")
            waveform = self.decode_audio(audio_latents)  # [B, audio_channels, N_samples]

        # Video guardrail
        if self.rank == 0:
            logger.info(f"Video decoded in {time.time() - decode_start:.2f}s")
            logger.info(f"Total pipeline time: {time.time() - pipeline_start:.2f}s")

            if use_guardrails and self.safety_checker is not None:
                with self.offloader.context_if_requested(COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT):
                    video = check_video_safety(video, self.safety_checker)

        timer.mark_end()

        if is_t2i:
            # Collapse the single decoded frame [B, T=1, H, W, C] -> [B, H, W, C].
            image = video[:, 0] if video is not None else None
            return timer.fill(PipelineOutput(image=image))

        return timer.fill(
            PipelineOutput(
                video=video,
                frame_rate=frame_rate,
                audio=waveform,
                audio_sample_rate=self.audio_tokenizer.model_config["sampling_rate"]
                if waveform is not None
                else None,
                # Sliced to the embodiment's real width, so the trailing dim is
                # raw_action_dim; the mode and embodiment are the caller's own
                # request and are not echoed back.
                action=action_latents[:, :, :resolved_raw_action_dim].float().cpu()
                if do_action and action_latents is not None
                else None,
            )
        )

    @staticmethod
    def _get_transfer_num_chunks(
        total_frames: int,
        frames_per_chunk: int,
        conditional_frames: int,
    ) -> tuple[int, int]:
        if frames_per_chunk <= 0:
            raise ValueError("Cosmos3 transfer frames_per_chunk must be positive.")
        if total_frames <= frames_per_chunk:
            return 1, frames_per_chunk
        stride = frames_per_chunk - conditional_frames
        if stride <= 0:
            raise ValueError(
                "Cosmos3 transfer num_conditional_frames must be smaller than num_video_frames_per_chunk."
            )
        remaining = total_frames - frames_per_chunk
        extra_chunks = remaining // stride + (1 if remaining % stride else 0)
        return 1 + extra_chunks, stride

    def _warn_on_control_length_mismatch(
        self, per_hint_frames: dict[str, torch.Tensor], total_frames: int
    ) -> None:
        """Flag hints short enough that mirror-padding invents most of their control.

        Measured as padding actually applied rather than as disagreement between
        hints, because that is what reaches the model: a request that pins
        ``num_frames`` down to the shortest clip pads nothing and stays silent,
        while a few frames of tail ping-pong is well under the threshold. A
        large gap is almost always a client that sent clips of different videos.
        """
        if self.rank != 0 or total_frames <= 0:
            return
        short = {
            key: frames.shape[1]
            for key, frames in per_hint_frames.items()
            if (total_frames - frames.shape[1]) / total_frames > CONTROL_LENGTH_MISMATCH_RATIO
        }
        if not short:
            return
        counts = ", ".join(f"{key}: {count}" for key, count in sorted(short.items()))
        logger.warning(
            f"Cosmos3 transfer control length mismatch: {counts} against {total_frames} frames "
            f"generated. The short hints are mirror-padded up to that length, so the output is "
            f"conditioned on control content that does not exist in them. Supply clips of "
            f"matching length, or set num_frames to the shortest one."
        )

    @staticmethod
    def _positive_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return value

    @staticmethod
    def _transfer_active_at(
        timestep: torch.Tensor,
        interval: tuple[float, float] | None,
    ) -> bool:
        """Is guidance active at ``timestep``?

        ``interval`` is in the scheduler's own timestep units (raw, typically
        0-1000 counting down), not a fraction of the schedule -- ``[0.0, 0.8]``
        selects the last step rather than the first 80%.
        """
        if interval is None:
            return True
        t_scalar = float(timestep.item()) if torch.is_tensor(timestep) else float(timestep)
        lo, hi = interval
        return float(lo) <= t_scalar <= float(hi)

    @staticmethod
    def _combine_transfer_predictions(
        *,
        cond_full: torch.Tensor,
        cond_no_control: torch.Tensor | None,
        uncond_full: torch.Tensor | None,
        guidance_scale: float,
        control_guidance: float,
    ) -> torch.Tensor:
        needs_control_cfg = cond_no_control is not None and control_guidance != 1.0
        needs_text_cfg = uncond_full is not None and guidance_scale > 1.0

        if needs_control_cfg and needs_text_cfg:
            control_cond = cond_no_control + control_guidance * (cond_full - cond_no_control)
            return uncond_full + guidance_scale * (control_cond - uncond_full)
        if needs_control_cfg:
            return cond_no_control + control_guidance * (cond_full - cond_no_control)
        if needs_text_cfg:
            return uncond_full + guidance_scale * (cond_full - uncond_full)
        return cond_full

    def diffuse_transfer(
        self,
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        uncond_ids: torch.Tensor,
        uncond_mask: torch.Tensor,
        guidance_scale: float,
        control_guidance: float,
        control_guidance_interval: tuple[float, float] | None,
        control_latents: list[torch.Tensor],
        shared_kwargs: dict[str, Any],
        velocity_mask: torch.Tensor,
        condition_latents: torch.Tensor,
        generator: torch.Generator,
        guidance_interval: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Run Cosmos3 transfer denoising with sequential control/text CFG branches."""

        branch_caches: dict[str, tuple[Any, Any]] = {}

        def run_branch(
            cache_key: str,
            *,
            text_ids: torch.Tensor,
            text_mask: torch.Tensor,
            branch_control_latents: list[torch.Tensor] | None,
            timestep: torch.Tensor,
        ) -> torch.Tensor:
            self.transformer.cached_kv, self.transformer.cached_freqs_gen = branch_caches.get(
                cache_key,
                (None, None),
            )
            result = self.transformer(
                hidden_states=latents,
                timestep=timestep / self.scheduler.config.num_train_timesteps,
                raw_timestep=timestep,
                text_ids=text_ids,
                text_mask=text_mask,
                control_latents=branch_control_latents,
                offload_context=self.offloader.context_if_requested,
                **shared_kwargs,
            )
            branch_caches[cache_key] = (
                self.transformer.cached_kv,
                self.transformer.cached_freqs_gen,
            )
            if result.video is None:
                raise ValueError("Cosmos3 transfer diffusion expects video predictions.")
            return result.video

        self.transformer.reset_cache()
        try:
            transfer_steps = tqdm(
                timesteps,
                total=len(timesteps),
                desc="Transfer denoising",
                disable=self.rank != 0,
                dynamic_ncols=True,
            )
            for t in transfer_steps:
                timestep = t.expand(latents.shape[0])
                step_guidance = (
                    float(guidance_scale) if self._transfer_active_at(t, guidance_interval) else 1.0
                )
                step_control = (
                    float(control_guidance)
                    if self._transfer_active_at(t, control_guidance_interval)
                    else 1.0
                )

                cond_full = run_branch(
                    "transfer_cond_full",
                    text_ids=cond_ids,
                    text_mask=cond_mask,
                    branch_control_latents=control_latents,
                    timestep=timestep,
                )
                cond_no_control = None
                if step_control != 1.0:
                    cond_no_control = run_branch(
                        "transfer_cond_no_control",
                        text_ids=cond_ids,
                        text_mask=cond_mask,
                        branch_control_latents=None,
                        timestep=timestep,
                    )

                uncond_full = None
                if step_guidance > 1.0:
                    uncond_full = run_branch(
                        "transfer_uncond_full",
                        text_ids=uncond_ids,
                        text_mask=uncond_mask,
                        branch_control_latents=control_latents,
                        timestep=timestep,
                    )

                noise_pred = self._combine_transfer_predictions(
                    cond_full=cond_full,
                    cond_no_control=cond_no_control,
                    uncond_full=uncond_full,
                    guidance_scale=step_guidance,
                    control_guidance=step_control,
                )
                noise_pred = noise_pred * velocity_mask
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    return_dict=False,
                    **self.sampling.scheduler_step_kwargs(generator),
                )[0]
                latents = velocity_mask * latents + (1.0 - velocity_mask) * condition_latents
        finally:
            self.transformer.reset_cache()

        return latents

    def _forward_transfer(
        self,
        *,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[str],
        height: int,
        width: int,
        max_frames: int,
        num_inference_steps: int,
        max_sequence_length: int,
        use_system_prompt: bool,
        use_duration_template: bool,
        use_resolution_template: bool,
        seed: int,
        frame_rate: float,
        num_frames: int,
        use_guardrails: bool,
        timer: CudaPhaseTimer,
        transfer_config: Cosmos3TransferConfig,
        video: Optional[bytes],
    ) -> PipelineOutput:
        if self.rank == 0:
            logger.info(f"Cosmos3 transfer target={width}x{height} (WxH)")

        # Decode the input video and every precomputed control on this rank's
        # NVDEC. A decode can fail non-uniformly across ranks (decoder init,
        # corrupt stream, allocation), so converge all ranks on one outcome
        # before any model collective rather than hanging the healthy ones.
        # The decoder sizes its retention ring from the requested window, so
        # asking for `max_frames` (5000 by default) would reserve ~14 GB at 720p
        # before a single frame lands. Bound it by what is actually generated:
        # the output is `num_frames` long, and `max_frames` stays the ceiling.
        decode_frames = min(int(max_frames), int(transfer_config.num_frames or num_frames))
        if self.rank == 0:
            logger.info(f"Cosmos3 transfer decoding up to {decode_frames} frames per reference")

        input_frames = None
        per_hint_frames: dict[str, torch.Tensor] = {}
        prepare_error: Optional[Exception] = None
        try:
            if video is not None:
                input_frames = decode_media_to_uint8_cthw(
                    video,
                    height=height,
                    width=width,
                    max_frames=decode_frames,
                    device=self.device,
                )

            for hint in transfer_config.ordered_hints:
                frames = load_or_compute_control_frames(
                    hint,
                    height=height,
                    width=width,
                    max_frames=decode_frames,
                    input_frames=input_frames,
                    device=self.device,
                )
                if frames.shape[1] < 1:
                    raise ValueError(f"Cosmos3 transfer hint '{hint.key}' produced no frames.")
                per_hint_frames[hint.key] = frames
            if not per_hint_frames:
                raise ValueError("Cosmos3 transfer requires at least one control hint.")
        except Exception as exc:
            prepare_error = exc
        synchronize_media_prepare_status(prepare_error)

        # The longest hint sets the length. Taking the first hint's instead would
        # make the result depend on TRANSFER_HINT_KEYS order rather than on the
        # request: with the same two clips, the earlier modality would win, so a
        # shorter first hint silently truncated the longer one (the chunk loop
        # never reads past total_frames) while a longer one padded it. No
        # further clamp is needed -- every hint was decoded under decode_frames,
        # which already bounds them by num_frames.
        total_frames = max(1, max(frames.shape[1] for frames in per_hint_frames.values()))
        self._warn_on_control_length_mismatch(per_hint_frames, total_frames)
        per_hint_frames = {
            key: pad_temporal_frames(frames, total_frames)
            for key, frames in per_hint_frames.items()
        }
        if input_frames is not None:
            input_frames = pad_temporal_frames(input_frames, total_frames)

        temporal_compression = self.vae_scale_factor_temporal
        chunk_frames = 1 if total_frames == 1 else transfer_config.num_video_frames_per_chunk
        chunk_frames = (
            math.ceil((chunk_frames - 1) / temporal_compression) * temporal_compression + 1
        )
        num_chunks, stride = self._get_transfer_num_chunks(
            total_frames,
            chunk_frames,
            transfer_config.num_conditional_frames,
        )
        padded_frames = max(total_frames, chunk_frames)
        per_hint_frames = {
            key: pad_temporal_frames(frames, padded_frames)
            for key, frames in per_hint_frames.items()
        }
        if input_frames is not None:
            input_frames = pad_temporal_frames(input_frames, padded_frames)

        # The encoded reference carries no frame rate across the extra-param
        # boundary, so the hint's configured fps (wsm's 10) wins over the
        # request's, which falls back to the mode default.
        frame_rate = (
            self._positive_float(transfer_config.fps) or self._positive_float(frame_rate) or 24.0
        )
        num_inference_steps = num_inference_steps or COSMOS3_720P_PARAMS["num_inference_steps"]
        guidance_scale = (
            float(transfer_config.guidance_scale)
            if transfer_config.guidance_scale is not None
            else COSMOS3_720P_PARAMS["guidance_scale"]
        )
        if self.sampling.is_distilled:
            # A distilled checkpoint runs one schedule and bakes guidance into
            # the weights, so the per-hint guidance presets cannot apply. Say so
            # rather than silently sampling off-distribution.
            num_inference_steps = self.sampling.num_steps(num_inference_steps)
            if guidance_scale != DISTILLED_GUIDANCE_SCALE:
                logger.warning(
                    f"Cosmos3 transfer on a distilled checkpoint: overriding "
                    f"guidance_scale {guidance_scale} with the mandated "
                    f"{DISTILLED_GUIDANCE_SCALE}; per-hint guidance presets do not apply."
                )
            guidance_scale = DISTILLED_GUIDANCE_SCALE
        flow_shift_target = float(
            transfer_config.flow_shift
            if transfer_config.flow_shift is not None
            else COSMOS3_V2V_DEFAULT_FLOW_SHIFT
        )
        max_sequence_length = max_sequence_length or COSMOS3_720P_PARAMS["max_sequence_length"]
        self._guidance_scale = guidance_scale
        self._num_timesteps = num_inference_steps
        # forward() skips its own scheduler rebuild for transfer, so this is the
        # only setup on this path: assigning is what keeps a previous request's
        # scheduler from carrying over on a reused worker.
        self.scheduler = self._scheduler_for(flow_shift_target, use_karras_sigmas=False)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        if negative_prompt is None:
            negative_prompt = COSMOS3_DEFAULT_NEGATIVE_PROMPT

        # Transfer prompts are already upsampled by the benchmark/config path.
        # Keep them verbatim; duration/resolution templates would change parity.
        prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        prompt = prompt[0]
        prompt = transfer_config.emphasized_prompt(prompt)
        if self.rank == 0:
            logger.info(f"Transfer prompt: '{prompt}'")

        # 1. Tokenize prompts (no separate text encoder — transformer embeds internally)
        logger.info("Tokenizing prompts...")
        system_prompt = COSMOS3_DEFAULT_SYSTEM_PROMPT
        cond_ids, cond_mask = self._tokenize_prompt(
            prompt, max_sequence_length, use_system_prompt, system_prompt=system_prompt
        )
        uncond_ids, uncond_mask = self._tokenize_prompt(
            negative_prompt, max_sequence_length, use_system_prompt, system_prompt=system_prompt
        )

        # Finished frames land in host memory as they are produced rather than
        # stacking on the GPU for the whole run: at 720p a decoded chunk is
        # ~0.5 GB, so a long generation would pile several GB of *completed*
        # output on top of the denoise working set -- and `torch.cat` would then
        # briefly double it. Only the next chunk's conditioning has to stay
        # resident.
        #
        # Pinned, because the copy is then asynchronous and overlaps the next
        # chunk's denoise; into pageable memory the same copy blocks and runs
        # ~40x slower. Allocated per request rather than held, since torch's
        # caching host allocator keeps the block and only the first request of a
        # given size reaches cudaHostAlloc -- so deployments that never serve
        # transfer page-lock nothing.
        #
        # Assembled on rank 0 alone: every rank decodes each chunk because the
        # next one conditions on it, but the executor sends only rank 0's
        # response, so assembling on the others allocates a pinned buffer per
        # rank to build a video nobody reads.
        assembles_output = self.rank == 0
        show_controls = transfer_config.show_control_condition
        show_input_panel = transfer_config.show_input and input_frames is not None
        panels = (len(per_hint_frames) if show_controls else 0) + (1 if show_input_panel else 0)
        host_output = (
            torch.empty(
                (1, total_frames, height, width * (panels + 1), 3),
                dtype=torch.uint8,
                # Page-locking needs a CUDA context; only the CPU-only unit
                # tests run without one, and there is nothing to overlap there.
                pin_memory=torch.cuda.is_available(),
            )
            if assembles_output
            else None
        )
        frames_written = 0
        previous_output: torch.Tensor | None = None

        # Every chunk's decode is part of generation here (it feeds the next
        # chunk), so the whole loop counts as denoise; post covers assembly.
        timer.mark_denoise_start()
        for chunk_id in range(num_chunks):
            start_frame = chunk_id * stride
            end_frame = min(start_frame + chunk_frames, total_frames)
            control_norms = {
                key: uint8_cthw_to_normalized_5d(
                    pad_temporal_frames(frames[:, start_frame:end_frame], chunk_frames),
                    dtype=self.dtype,
                )
                for key, frames in per_hint_frames.items()
            }
            target_norm = torch.zeros_like(next(iter(control_norms.values())))
            current_conditional_frames = 0

            if chunk_id == 0 and transfer_config.num_first_chunk_conditional_frames > 0:
                if input_frames is None:
                    raise ValueError(
                        "Cosmos3 transfer num_first_chunk_conditional_frames > 0 requires a video input."
                    )
                current_conditional_frames = min(
                    transfer_config.num_first_chunk_conditional_frames,
                    input_frames.shape[1],
                    chunk_frames,
                )
                if current_conditional_frames > 0:
                    input_cond = uint8_cthw_to_normalized_5d(
                        input_frames[:, :current_conditional_frames],
                        dtype=self.dtype,
                    )
                    target_norm[:, :, :current_conditional_frames] = input_cond
                    if current_conditional_frames < chunk_frames:
                        fill = target_norm[
                            :, :, current_conditional_frames - 1 : current_conditional_frames
                        ]
                        target_norm[:, :, current_conditional_frames:] = fill.expand(
                            -1,
                            -1,
                            chunk_frames - current_conditional_frames,
                            -1,
                            -1,
                        )
            elif chunk_id > 0 and previous_output is not None:
                current_conditional_frames = min(
                    transfer_config.num_conditional_frames,
                    previous_output.shape[2],
                    chunk_frames,
                )
                if current_conditional_frames > 0:
                    target_norm[:, :, :current_conditional_frames] = previous_output[
                        :, :, -current_conditional_frames:
                    ].to(target_norm)
                    if current_conditional_frames < chunk_frames:
                        fill = target_norm[
                            :, :, current_conditional_frames - 1 : current_conditional_frames
                        ]
                        target_norm[:, :, current_conditional_frames:] = fill.expand(
                            -1,
                            -1,
                            chunk_frames - current_conditional_frames,
                            -1,
                            -1,
                        )

            control_latents = [
                self._encode_video_tensor(control) for control in control_norms.values()
            ]
            latents, velocity_mask, condition_latents = self._prepare_transfer_latents(
                target_norm,
                current_conditional_frames,
                generator,
            )
            video_shape = (latents.shape[2], latents.shape[3], latents.shape[4])
            shared_kwargs = dict(
                video_shape=video_shape,
                fps=frame_rate,
                noisy_frame_mask=velocity_mask,
                transfer_share_vision_temporal_positions=transfer_config.share_vision_temporal_positions,
            )

            self.sampling.set_timesteps(self.scheduler, num_inference_steps, device=self.device)
            latents = self.diffuse_transfer(
                latents=latents,
                timesteps=self.scheduler.timesteps,
                cond_ids=cond_ids,
                cond_mask=cond_mask,
                uncond_ids=uncond_ids,
                uncond_mask=uncond_mask,
                guidance_scale=guidance_scale,
                control_guidance=transfer_config.control_guidance,
                control_guidance_interval=transfer_config.control_guidance_interval,
                control_latents=control_latents,
                shared_kwargs=shared_kwargs,
                velocity_mask=velocity_mask,
                condition_latents=condition_latents,
                generator=generator,
            )
            # Deliberately the raw decode rather than `decode_latents()`: the
            # decoded chunk is this loop's *input* as well as its output — the
            # next chunk conditions on its tail — so every rank needs it. Owning
            # one rank's decode and broadcasting would ship ~0.5 GB per chunk
            # and serialize the loop behind a collective, which costs more than
            # recomputing the decode locally.
            output_video = self._decode_latents_raw(latents).clamp(-1, 1)

            # Chunk 0 keeps every frame; later chunks drop the overlap they were
            # conditioned on. The tail trim that used to follow the final concat
            # happens here instead, as a bound on what each chunk contributes.
            skip = 0 if chunk_id == 0 else current_conditional_frames
            take = min(output_video.shape[2] - skip, total_frames - frames_written)
            if assembles_output and take > 0:
                chunk = output_video[:, :, skip : skip + take]
                panel_tensors = []
                if show_controls:
                    panel_tensors.extend(
                        control_norms[key][:, :, skip : skip + take] for key in per_hint_frames
                    )
                if show_input_panel:
                    panel_tensors.append(
                        uint8_cthw_to_normalized_5d(
                            input_frames[:, frames_written : frames_written + take],
                            dtype=torch.float32,
                        )
                    )
                if panel_tensors:
                    chunk = torch.cat([p.to(chunk) for p in panel_tensors] + [chunk], dim=-1)
                # Post-processing is elementwise, so doing it per chunk is
                # bitwise identical to doing it on the assembled clip -- and it
                # halves the transfer, since uint8 crosses instead of bf16.
                host_output[:, frames_written : frames_written + take].copy_(
                    postprocess_video_tensor(chunk), non_blocking=True
                )
                frames_written += take

            # Only the tail is read as the next chunk's conditioning (one frame
            # by default), so keep that slice rather than pinning the whole
            # ~0.5 GB chunk across the next denoise.
            keep = min(transfer_config.num_conditional_frames, output_video.shape[2])
            previous_output = output_video[:, :, -keep:].clone() if keep > 0 else None

        timer.mark_post_start()

        # The chunk copies are asynchronous, so the host buffer is not readable
        # until the stream drains. The upload below would be correctly ordered
        # without this, but the guardrail reads the frames on the CPU.
        #
        # Only false under the CPU-only unit tests, where the copies above were
        # host-to-host and there is no stream to wait on (``CudaPhaseTimer``
        # disables itself on the same condition).
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

        # Same screening the other modes get: rank 0, post-processed frames,
        # and a None result means the guardrail rejected the clip. With the
        # debug panels enabled the caller's own control frames ride along in
        # the same tensor and are screened too, which errs safe. Screening the
        # host copy rather than a device one removes a full round trip:
        # ``check_video_safety`` moves to CPU internally and returns on the
        # input tensor's device.
        video = host_output
        if self.rank == 0 and use_guardrails and self.safety_checker is not None:
            with self.offloader.context_if_requested(COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT):
                video = check_video_safety(video, self.safety_checker)
        # Back to the device the other pipelines hand back, now that denoising
        # has released its working set.
        if video is not None:
            video = video.to(self.device)

        timer.mark_end()
        return timer.fill(
            PipelineOutput(
                video=video,
                frame_rate=frame_rate,
            )
        )
