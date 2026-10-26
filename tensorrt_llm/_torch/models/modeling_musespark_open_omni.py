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
"""MuseSparkOpen perception: the NaViT vision tower, the wav_ae audio tower,
the projection, and the merge.

The decoder lives in :mod:`modeling_musespark_open` and is used here unchanged.
The audio tower and its token accounting live in :mod:`musespark_open_audio`;
this module wires them into the input processor, the encoder wrapper, and the
placeholder registry.
Keeping the two files apart is deliberate: the decoder's token-level parity is
already pinned by a set of banked gates, and every line changed there reopens
them.

Two things about this model differ from the shape a reader coming from the
other VLMs in this directory would expect.

**The projected vision rows are handed over un-normed.** The reference
implementation runs vision rows through a weightless ``perception_emb_norm``
inside its encoder and merges them after the text embeddings have had their own
norm. Our decoder (:meth:`MuseSparkOpenModel.forward`) instead applies
``embed_norm`` to whatever ``inputs_embeds`` it is given, and ``embed_norm`` is
the *same function*: a weightless ``RMSNorm(hidden_size, eps=rms_norm_eps)``.
Since RMSNorm is per-row, merging first and norming once is row-for-row
identical to norming first and merging -- and it costs no change to the
decoder. Normalizing in the encoder as well would apply it twice, which is not
a no-op: measured in bf16 it is bitwise identical at row scale 1.0 and 50.0 but
moves the result by 1.6e-02 relative at row scale 0.02, four times bf16 unit
roundoff. So the tower returns raw projected rows and the decoder norms text
and vision alike, exactly once, and then scales both by ``m_emb``.

**The inner LLM is imported, not resolved through ``AutoModelForCausalLM``.**
The usual wrapper pattern resolves the decoder from a ``text_config``
sub-config. This checkpoint has no ``text_config``: ``MuseSparkOpenConfig`` *is*
the text config, with ``vision_config`` hanging off it. Going through the auto
registry would mean rewriting ``architectures`` on a config copy purely to look
up a class this module already imports, so it imports the class.

Positions are plain 1D sequential and shared with the text: no mRoPE and no
per-image position reset. Image rows are windowed by the decoder's variable
sliding-window schedule exactly like ordinary text.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel

from ...inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ContentFormat,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    register_input_processor,
)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..model_config import ModelConfig
from ..modules.embedding import Embedding
from .checkpoints.hf.musespark_open_weight_mapper import load_musespark_open_omni_weights
from .modeling_multimodal_mixin import MultimodalModelMixin
from .modeling_musespark_open import MuseSparkOpenForCausalLM
from .modeling_utils import register_auto_model, register_vision_encoder
from .musespark_open_audio import (
    AUDIO_TARGET_SAMPLE_RATE,
    AUDIO_TOKEN_IDS,
    MuseSparkOpenAudioEncoder,
    audio_placeholder_ids,
    num_audio_tokens,
    resampled_num_samples,
    to_mono_24k,
    tokenizer_vocab,
    verify_audio_token_ids,
)
from .musespark_open_vision import MuseSparkOpenNaViTVisionTower
from .musespark_open_vision_processing import (
    VISION_TOKEN_IDS,
    MuseSparkOpenPreprocessConfig,
    image_placeholder_ids,
    least_distortion_fit,
    num_placeholder_rows,
    preprocess_image,
    preprocess_video,
    video_grid_thw,
    video_placeholder_ids,
    with_processor_config,
)

__all__ = [
    "MEDIA_MARKERS",
    "MEDIA_MARKER_ALIASES",
    "MuseSparkOpenOmniForConditionalGeneration",
    "MuseSparkOpenOmniInputProcessor",
    "MuseSparkOpenVisionEncoder",
]

# What the chat template leaves in the prompt where a medium was attached.
# These are PLAIN TEXT, not special tokens: the template writes them from a
# content part (``render_content`` in chat_template.jinja), and it rewrites
# any literal occurrence in caller-supplied text to "< image>" so a prompt
# cannot forge one. The tokenizer has no added token for either string, so each
# is a short run of ordinary ids whose interior depends on what sits next to it
# -- which is why they are located in the text and not in the id stream.
# "<audio>" follows the same contract: the template maps audio / input_audio /
# audio_url content parts to the literal string "<audio>" and escapes any
# caller-typed occurrence to "< audio>".
MEDIA_MARKERS = {"image": "<image>", "video": "<video>", "audio": "<audio>"}

# The same slot, spelled the other way. The open-export checkpoint ships a
# chat template that writes the control-token spellings -- "<|patch|>",
# "<|video|>", "<|audio|>" -- where the earlier template wrote the plain
# strings above, while still escaping the plain forms out of caller text. A
# scan that knows only one spelling finds zero markers against the other
# template and every media request fails with "prompt has 0 <modality>
# markers", so both spellings are accepted and whichever the active template
# emits is the one that matches.
#
# MEDIA_MARKERS stays the canonical modality -> placeholder map (it is what
# the input-processor registration publishes and what the modality key set is
# taken from); this is the scan alphabet only.
MEDIA_MARKER_ALIASES: Dict[str, Tuple[str, ...]] = {
    "image": (MEDIA_MARKERS["image"], "<|patch|>"),
    "video": (MEDIA_MARKERS["video"], "<|video|>"),
    "audio": (MEDIA_MARKERS["audio"], "<|audio|>"),
}

#: Per-item audio token budget for the runtime profiler, in decoder rows
#: (1 row = 80 ms, so 1024 rows = 81.92 s per clip). Neither reference has an
#: audio-aware cap: vLLM falls through to a 1024 default and SGLang has none,
#: so 1024 matches the served behavior. Deliberately NOT clamped by
#: ``max_num_encoder_tokens`` -- that is the vision encoder's attention
#: budget, and the audio tower has no attention.
_MAX_AUDIO_TOKENS_PER_ITEM = 1024

#: Dummy audio for profiling: one second, matching the references' dummy
#: (24000 samples = 13 rows). Profiling therefore does not exercise long
#: clips, same as in both references.
_DUMMY_AUDIO_SAMPLES = AUDIO_TARGET_SAMPLE_RATE


class MuseSparkOpenOmniInputProcessor(
    BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder
):
    """Turn (text, images/videos) into token ids plus per-modality tensors.

    There is no HF ``AutoProcessor`` for this model in the pinned
    ``transformers``, so the preprocessing and the placeholder expansion are
    local (``musespark_open_vision_processing``), transcribed from the source
    semantics rather than imported from a shim.
    """

    supports_tokenization_cache = True

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer,
        trust_remote_code: bool = True,
        enable_tokenization_cache: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self._config = config
        self._tokenizer = tokenizer
        self._model_path = model_path
        # A media-free prompt is ``tokenizer.encode(text, add_special_tokens=False)``
        # (``_encode``), which is exactly what the prefix-tokenization cache
        # computes, so its spliced ids are the ids this processor would produce.
        # A processor built without a tokenizer (the engine's geometry copy)
        # never tokenizes a served prompt and gets no cache.
        self._init_tokenization_cache(enable_tokenization_cache, tokenizer)
        # `config.json` is not the whole preprocessing config on newer exports:
        # they drop `vision_config.navit_config` and declare the same knobs in
        # `processor_config.json` under HuggingFace spellings. Without this the
        # per-frame video token budget silently falls back to the module
        # default on a checkpoint that declares its own. Returns the same
        # object for exports that ship no such file.
        self._vision_config = with_processor_config(config.vision_config, model_path)
        # Audio is gated on the config exactly as both references gate it:
        # `audio_perception_modality_args.input_mode == "wav"`. The other keys
        # in that dict (`encoder_family`, `next_audio_token`,
        # `empty_audio_token`) have zero consumers in either reference and are
        # deliberately not read here either.
        audio_args = getattr(config, "audio_perception_modality_args", None) or {}
        self._audio_enabled = audio_args.get("input_mode") == "wav"
        self._audio_ids_verified = False

    # -- the properties the framework instantiates this through ----------
    #
    # All five are abstract on the two base classes. Leaving any of them out
    # makes the class abstract, so the framework cannot construct it at all --
    # which the unit tests missed entirely because they only ever called the
    # module-level preprocessing helpers and never built a processor.
    @property
    def processor(self):
        """No HF AutoProcessor exists for this model in the pinned transformers.

        The preprocessing this model needs lives in
        ``musespark_open_vision_processing`` instead, transcribed from the source
        semantics. Returning the tokenizer would be a lie that only shows up
        later as a confusing attribute error, so this says so plainly.
        """
        raise AttributeError(
            "MuseSparkOpen has no HF AutoProcessor: image and video preprocessing "
            "live in musespark_open_vision_processing, driven by call_with_text_prompt"
        )

    @property
    def tokenizer(self):
        """The tokenizer, built from the checkpoint when none was handed over.

        ``model_engine.py`` constructs a second copy of this processor with
        ``tokenizer=None`` and the framework expects the processor to supply its
        own: ``create_input_processor`` wraps construction in
        ``hf_remote_code_lock()`` precisely because "input processors build an
        AutoTokenizer/AutoProcessor", and the sibling processors do it. Without
        this, a media-free request through an engine-built processor died on
        ``NoneType has no attribute 'encode'``.

        Built lazily rather than in ``__init__`` for two reasons: the
        geometry-only copy the engine profiles with never tokenizes, so every
        rank would pay for a load it does not use; and a processor built for
        geometry against a path with no tokenizer would fail construction it
        never needed.

        Built with the caller's own ``trust_remote_code`` and ``use_fast``,
        which the base class already carries -- not with
        ``from_pretrained``'s defaults. ``use_fast`` in particular has to match
        whatever the rest of the stack built, because the slow and fast
        tokenizers for this checkpoint do not agree on digit tokenization, and
        a processor that silently disagreed with the engine's tokenizer would
        show up as wrong answers rather than as an error.
        """
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=self._trust_remote_code,
                use_fast=self.use_fast,
            )
        return self._tokenizer

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def dtype(self) -> torch.dtype:
        return self._config.torch_dtype

    # -- what the runtime counts ----------------------------------------
    #
    # These four answer "which positions carry vision rows, and how many does
    # this item get". The framework calls them on the hashing path, once per
    # item per request, and an exception in any of them is caught and logged as
    # "Multimodal hashing failed:" -- not raised. So getting one wrong does not
    # break a request, it silently turns off embedding caching and re-runs the
    # tower on every chunked-prefill chunk.
    def get_mm_token_ids(self) -> torch.Tensor:
        """The ids whose positions the merge overwrites with encoder rows.

        Only the three embed-carrying ids, and no specials. The base class would
        otherwise look for ``self.processor.mm_token_ids``, and this model has
        no HF processor to ask.

        No framing token is included because none sits *inside* an image unit:
        an image expands to one unbroken run of ``<|patch|>`` between its start
        and end markers, so the run is already one span. A video genuinely is
        several spans -- its per-clip timestamp text is ordinary text between
        the ``<|video|>`` runs and cannot be enumerated as ids -- and the
        runtime's per-item run offsets handle that; adding ``<|vid_start|>``
        and friends here would not merge the runs, it would only inflate the
        count this class is required to keep in step with
        ``get_num_tokens_per_*``. The audio delimiters
        (``<|soft_audio_start|>`` / ``<|soft_audio_end|>``) are likewise
        excluded: only the interior ``<|audio|>`` run carries encoder rows.
        """
        return torch.tensor(
            [VISION_TOKEN_IDS["patch"], VISION_TOKEN_IDS["video"], AUDIO_TOKEN_IDS["audio"]],
            dtype=torch.int32,
        )

    def get_mm_special_token_ids(self) -> None:
        """None: every id in :meth:`get_mm_token_ids` carries an embedding."""
        return None

    def get_num_tokens_per_image(self, *, image: Any, **kwargs) -> int:
        """How many decoder rows this image will occupy.

        Signature is the framework's, which hands over the image itself -- a
        PIL image, a CHW tensor, or an HWC uint8 array, depending on which
        media loader produced it. Only its size is read; the canvas fit decides
        the row count, so this answers without touching pixels.

        The temporal extent is ONE. ``patch_temporal`` is how many frames are
        packed into a clip, and a still image is duplicated to fill one clip;
        it is not a second clip. ``preprocess_image`` accordingly returns
        ``grid_thw = (1, h, w)``, and passing ``patch_temporal`` here instead
        reported exactly twice the real row count -- to the KV-cache profiler
        and to anything else that sizes work from this number.
        """
        cfg = self._vision_config
        image_width, image_height = _image_size(image)
        width, height = least_distortion_fit(
            image_width,
            image_height,
            patch_size=cfg.patch_size,
            downsample_factor=cfg.downsample_factor,
            min_tokens=cfg.min_num_image_tokens,
            max_tokens=cfg.max_num_image_tokens,
        )
        grid = (1, height // cfg.patch_size, width // cfg.patch_size)
        return num_placeholder_rows(grid, cfg)

    def get_num_tokens_per_video(
        self,
        *,
        video: Any,
        video_metadata: Optional[Dict[str, Any]] = None,
        video_audio: Any = None,
        **kwargs,
    ) -> int:
        """How many decoder rows this video will occupy.

        Sampling is what makes this more than a per-frame multiply: 2 fps out
        of the source rate decides how many frames survive, so the source frame
        rate is required. It arrives in ``video_metadata['fps']`` when a media
        loader decoded the video, which is the normal serving path; a caller
        handing over bare frames must pass it as ``mm_processor_kwargs``.
        """
        del video_audio
        frames = list(video)
        metadata = video_metadata or {}
        fps = metadata.get("fps", kwargs.get("fps"))
        if fps is None:
            raise ValueError(
                "MuseSparkOpen cannot count a video's tokens without the source "
                "frame rate: sampling is 2 fps out of it, so the frame count "
                "and therefore the token count depend on it."
            )
        width, height = _image_size(frames[0])
        grid = video_grid_thw(
            len(frames),
            float(fps),
            width,
            height,
            self._vision_config,
            total_num_frames=metadata.get("total_num_frames"),
        )
        return num_placeholder_rows(grid, self._vision_config)

    def get_num_tokens_per_audio(
        self,
        *,
        audio: Any,
        **kwargs,
    ) -> int:
        """How many decoder rows this audio clip will occupy.

        ``ceil(samples_at_24kHz / 1920)``, computed post-resample and
        post-mono-downmix, exclusive of the two delimiter tokens (which keep
        their own text embeddings and are not in :meth:`get_mm_token_ids`).
        The resample itself is not run here: ``resampled_num_samples`` is
        exactly the length ``librosa.resample(fix=True)`` produces, so the
        count matches the real path without touching the samples. A bare array
        with no rate is trusted to already be 24 kHz, the same trust both
        references place in their parsers.
        """
        del kwargs
        if not self._audio_enabled:
            raise ValueError(
                "audio input arrived but this checkpoint's config does not "
                "enable the waveform encoder (audio_perception_modality_args."
                "input_mode != 'wav')"
            )
        sample_rate = AUDIO_TARGET_SAMPLE_RATE
        if isinstance(audio, tuple) and len(audio) == 2:
            audio, sample_rate = audio
        elif isinstance(audio, dict):
            sample_rate = audio.get("sampling_rate", audio.get("sample_rate", sample_rate))
            audio = audio.get("array", audio.get("audio", audio.get("waveform")))
        shape = getattr(audio, "shape", None) or (len(audio),)
        # 2-D input: the time axis is the longer one -- the same
        # shorter-axis-is-channels rule the downmix in to_mono_24k applies.
        length = int(max(shape)) if len(shape) == 2 else int(shape[0])
        return num_audio_tokens(resampled_num_samples(length, int(sample_rate)))

    def get_mm_max_tokens_per_item(self, max_num_encoder_tokens: Optional[int] = None):
        """Largest legal item per modality, bounded by the runtime budget.

        The two numbers are in different units and mixing them silently
        overstates what fits. The caps in the vision config count DECODER rows
        -- what the item costs the prompt, after the 2x2 pixel shuffle -- while
        ``max_num_encoder_tokens`` is the encoder's own attention budget, in
        patches. One row is ``downsample_factor**2`` patches, so the budget is
        converted before the two are compared; taking ``min`` of them directly
        let an item four times too large past the bound.
        """
        cfg = self._vision_config
        image_cap = cfg.max_num_image_tokens
        video_cap = cfg.max_num_video_tokens
        if max_num_encoder_tokens is not None:
            rows_affordable = max(1, max_num_encoder_tokens // cfg.downsample_factor**2)
            image_cap = min(image_cap, rows_affordable)
            video_cap = min(video_cap, rows_affordable)
        caps = {"image": image_cap, "video": video_cap}
        if self._audio_enabled:
            # See _MAX_AUDIO_TOKENS_PER_ITEM for why this ignores
            # max_num_encoder_tokens.
            caps["audio"] = _MAX_AUDIO_TOKENS_PER_ITEM
        return caps

    def get_preferred_media_io_kwargs(self) -> Dict[str, Dict[str, Any]]:
        """Ask the loaders for PIL and for the model's own video sampling.

        Format: the framework's default is ``pt`` -- ``ToTensor()`` output, a
        float32 CHW tensor in 0..1 -- and this model's preprocessing needs
        PIL: ``to_rgb`` composites RGBA by image *mode* and ``resize_lanczos``
        calls ``Image.resize``. Declaring the format is the framework's own
        mechanism for that and it avoids a lossy-looking round-trip through
        0..1 floats on every request. ``_to_pil`` still normalizes whatever
        arrives, because a per-request ``media_io_kwargs`` override outranks
        this.

        Video sampling: ``VideoMediaIO`` defaults to ``num_frames=10,
        fps=30`` -- 10 frames per video regardless of length -- while this
        model's declared policy is ``sampling_fps`` capped at
        ``num_frames_per_video`` (2 fps / 96 frames on the shipped
        checkpoint, ``processor_config.json`` values included via
        ``with_processor_config``). Without declaring it here, a bare
        ``trtllm-serve`` boot silently hands the model 10 frames per video.
        The loader samples ``min(num_frames, floor(duration * fps))``
        uniformly, which is exactly the model's own policy. A server-level
        ``--media_io_kwargs`` and per-request kwargs still override these
        defaults (``VideoMediaIO.merge_kwargs`` also drops the unmentioned
        ``fps``/``num_frames`` partner when an override sets only one).
        """
        cfg = MuseSparkOpenPreprocessConfig.from_config(self._vision_config)
        video: Dict[str, Any] = {
            "format": "pil",
            "num_frames": cfg.num_frames_per_video,
        }
        if cfg.sampling_fps:
            video["fps"] = cfg.sampling_fps
        return {"image": {"format": "pil"}, "video": video}

    def get_dummy_mm_data(self, *, max_num_encoder_tokens: int, mm_counts, dtype):
        """Worst-case encoder tensors, built directly rather than via PIL.

        Used to size the encoder's memory contribution at startup, so the grid
        and the pixel rows have to describe the SAME item: the tower segments
        the packed batch by ``t * h * w`` read off the grid, and a grid that
        claims more patches than were allocated walks off the end of
        ``pixel_values``.

        ``t`` is 1 here, not ``patch_temporal``. ``patch_temporal`` is how many
        frames are packed into one clip and it is already inside ``patch_dim``
        (``num_channels * patch_temporal * patch_size**2``); using it as a clip
        count described exactly twice the patches this allocates.
        """
        cfg = self._vision_config
        patch_dim = cfg.num_channels * cfg.patch_temporal * cfg.patch_size**2
        # `max_num_encoder_tokens` is the AGGREGATE encoder-attention budget for
        # the whole batch, not an allowance per item, and `mm_counts` says how
        # many items share it. Sizing each item at the per-item maximum
        # overshot it by exactly `count`: a budget of 256 with 4 items
        # allocated 4 x 256 patches. The profiler sizes encoder memory from
        # what this returns, so overshooting reserves memory the runtime will
        # never be asked for.
        total_items = sum(count for count in mm_counts.values() if count) or 1
        share = max(1, max_num_encoder_tokens // total_items)
        caps = self.get_mm_max_tokens_per_item(share)
        data: Dict[str, Any] = {}
        for modality, count in mm_counts.items():
            if not count:
                continue
            if modality == "audio":
                if not self._audio_enabled:
                    continue
                # One second per dummy clip, float32 raw waveform -- the
                # encoder input is the waveform itself, not pixel patches, so
                # the vision grid arithmetic below does not apply.
                data["audio"] = {
                    "waveforms": torch.zeros(count * _DUMMY_AUDIO_SAMPLES, dtype=torch.float32),
                    "sample_counts": torch.tensor(
                        [_DUMMY_AUDIO_SAMPLES] * count, dtype=torch.int64
                    ),
                }
                continue
            rows = caps[modality]
            # rows are post-pixel-shuffle, so each costs downsample_factor**2
            # patches on the encoder side. Both sides of the grid stay even:
            # the pixel shuffle merges 2x2 patches and an odd side would leave
            # a partial row.
            patches = rows * cfg.downsample_factor**2
            side = max(1, int(patches**0.5))
            side -= side % cfg.downsample_factor
            side = max(side, cfg.downsample_factor)
            other = max(patches // side, cfg.downsample_factor)
            other -= other % cfg.downsample_factor
            grid = torch.tensor([[1, side, other]] * count, dtype=torch.int32)
            data[modality] = {
                "pixel_values": torch.zeros((count * side * other, patch_dim), dtype=dtype),
                f"{modality}_grid_thw": grid,
            }
        if len(data) > 1:
            # A mixed-modality request needs a prompt-order manifest (see
            # _items_in_prompt_order); give the dummy one in a fixed order so
            # profiling with several modalities does not trip the refusal that
            # protects real requests.
            data["mm_item_order"] = [
                {"modality": modality, "index": index}
                for modality in data
                if modality != "mm_item_order"
                for index in range(mm_counts.get(modality, 0))
            ]
        return data

    # -- the real request path ------------------------------------------
    def call_with_text_prompt(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt = inputs.get("prompt")
        mm_data = inputs.get("multi_modal_data") or {}

        if not mm_data:
            ids = (
                self._encode_with_tokenization_cache(text_prompt, sampling_params)
                if text_prompt
                else None
            )
            return (self._encode(text_prompt) if ids is None else ids), None

        # Every modality key must be consumed below. Anything else -- including
        # "audio" against a checkpoint whose config does not enable the
        # waveform encoder -- is refused rather than silently dropped: a
        # dropped modality answers from text alone with no error anywhere.
        unknown = set(mm_data) - set(MEDIA_MARKERS)
        if unknown:
            raise ValueError(
                f"MuseSparkOpen cannot process modalities {sorted(unknown)}; "
                f"supported: {sorted(MEDIA_MARKERS)}"
            )
        if mm_data.get("audio") and not self._audio_enabled:
            raise ValueError(
                "audio input arrived but this checkpoint's config does not "
                "enable the waveform encoder (audio_perception_modality_args."
                "input_mode != 'wav')"
            )

        cfg = self._vision_config
        multimodal_data: Dict[str, Dict[str, Any]] = {}
        expansions: List[Tuple[str, List[int]]] = []

        images = mm_data.get("image") or []
        if images:
            pixel_values, grids = [], []
            for image in images:
                values, grid = preprocess_image(_to_pil(image), cfg)
                pixel_values.append(values)
                grids.append(grid)
                expansions.append(("image", image_placeholder_ids(grid, cfg)))
            multimodal_data["image"] = {
                "pixel_values": torch.cat(pixel_values, dim=0),
                "image_grid_thw": torch.tensor(grids, dtype=torch.int32),
            }

        videos = mm_data.get("video") or []
        if videos:
            processor_kwargs = inputs.get("mm_processor_kwargs") or {}
            pixel_values, grids = [], []
            for video in videos:
                frames, fps, sampling = _frames_and_fps(video, processor_kwargs)
                values, grid, timestamps = preprocess_video(
                    [_to_pil(frame) for frame in frames], fps, cfg, **sampling
                )
                pixel_values.append(values)
                grids.append(grid)
                expansions.append(
                    # self.tokenizer, not self._tokenizer: a video's per-clip
                    # timestamp text is tokenized as ordinary text, so this path
                    # needs a real tokenizer exactly as _encode does. Reading the
                    # raw attribute skipped the lazy build and handed None to a
                    # helper that rejects it -- an engine-built processor (which
                    # is always constructed with tokenizer=None) would take a
                    # video request and fail on it while images went through.
                    ("video", video_placeholder_ids(grid, timestamps, cfg, self.tokenizer))
                )
            multimodal_data["video"] = {
                "pixel_values": torch.cat(pixel_values, dim=0),
                "video_grid_thw": torch.tensor(grids, dtype=torch.int32),
            }

        audios = mm_data.get("audio") or []
        if audios:
            # Splicing puts encoder rows on the <|audio|> positions by id, so
            # the ids must be what THIS tokenizer means by those names. Checked
            # once per processor, by vocab-name lookup exactly as both
            # references resolve them.
            if not self._audio_ids_verified:
                verify_audio_token_ids(tokenizer_vocab(self.tokenizer))
                self._audio_ids_verified = True
            default_rate = (inputs.get("mm_processor_kwargs") or {}).get("sample_rate")
            waveforms: List[torch.Tensor] = []
            for audio in audios:
                # 24 kHz mono float32; the token count is defined on THIS
                # waveform (post-resample, post-downmix), and the encoder
                # emits exactly ceil(N/1920) rows for it by construction.
                waveform = to_mono_24k(audio, default_rate)
                waveforms.append(torch.from_numpy(waveform))
                expansions.append(("audio", audio_placeholder_ids(num_audio_tokens(len(waveform)))))
            # Variable-length clips travel as one packed 1-D tensor plus the
            # per-clip lengths: the engine's staging moves tensors only, and a
            # ragged list would not survive to_handle/to_device. sample_counts
            # stays on the host (it is not in multimodal_data_device_paths) --
            # the encoder reads it to split the pack, which is a host read.
            multimodal_data["audio"] = {
                "waveforms": torch.cat(waveforms),
                "sample_counts": torch.tensor([w.numel() for w in waveforms], dtype=torch.int64),
            }

        # The placeholder expansion carries its own start/end markers, so it
        # replaces the marker the chat template left behind rather than being
        # inserted next to it.
        fused, item_order = self._splice_placeholders(text_prompt, expansions)
        # Prompt order travels WITH the pixels, in the same dict, because that is
        # the channel this model already knows arrives intact. The encoder has to
        # lay its rows out in prompt order -- ``fuse_input_embeds`` consumes them
        # positionally -- and the only place that order is known for certain is
        # here, where the placeholder runs were spliced into the token stream.
        # ``MultimodalParams.mm_item_order`` carries the same information from the
        # chat parser and the encoder still accepts it, but on the served path it
        # did not reach the tower: measured 2026-09-09, an image+video request
        # arrived with the field empty and the encoder refused it. A manifest
        # that is sometimes absent is not a contract. Non-tensor values in
        # ``multimodal_data`` are an established shape (``special_token_offsets``
        # is a plain list) and ``_apply_tensor_operation`` returns str/int leaves
        # untouched, so this survives to_handle / to_tensor / to_device.
        multimodal_data["mm_item_order"] = item_order
        return fused, {"multimodal_data": multimodal_data}

    def _splice_placeholders(
        self, text: str, expansions: List[Tuple[str, List[int]]]
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Tokenize ``text``, replacing each media marker with its expanded run.

        Markers are consumed in prompt order, which is the order the chat
        handler decoded the media in, so item *i* of a modality lands on marker
        *i* of that modality.

        The marker is found in the TEXT and the segments between markers are
        tokenized separately. Two wrong ways to do this, both tried:

        * Matching the special ``<|image|>`` 200090 in the id stream finds
          nothing. Neither shipped template emits it: the earlier one writes the
          plain string ``<image>`` (``render_content`` in
          chat_template.jinja), for which the tokenizer has no added token,
          and the open-export one writes ``<|patch|>``. So 200090 never appears
          in a templated prompt either way. Which spelling a prompt carries is a
          property of the checkpoint's template, which is why the scan accepts
          both (``MEDIA_MARKER_ALIASES``) instead of assuming one.
        * Matching ``tokenizer.encode("<image>")`` in the id stream finds
          nothing either as soon as two media are adjacent, which is the norm
          rather than an edge case -- every MindCube item carries 2 to 4 images
          and the template writes them with no separator. BPE merges the seam:
          ``<image>`` alone is ``[40, 3405, 42]`` but ``<image><image>`` is
          ``[40, 3405, 2696, 3405, 42]``, where 2696 is ``"><"``. A fixed
          subsequence cannot match a run whose interior tokens depend on how
          many markers are adjacent.

        Splitting the text is safe here for a specific reason, not by luck: each
        expansion begins and ends with a special token (``<|image_start|>`` /
        ``<|image_end|>``, ``<|vid_start|>`` / ``<|vid_end|>``), and a special
        token never merges with its neighbours. So each text segment sits
        against a token boundary the tokenizer would have produced anyway, which
        is exactly the shape of the frozen reference prompts.

        Returns the fused ids AND the prompt-order manifest, because this loop
        is the one place that knows it: it walks the TEXT and takes whichever
        marker comes next, so the order it consumes items in *is* the order the
        placeholder runs sit in the token stream. Deriving it anywhere else --
        from the per-modality dicts, say -- would be a reconstruction, and the
        encoder's rows have to match this, not a reconstruction of it.
        """
        pending: Dict[str, List[List[int]]] = {modality: [] for modality in MEDIA_MARKERS}
        for modality, ids in expansions:
            pending[modality].append(ids)

        out: List[int] = []
        item_order: List[Dict[str, Any]] = []
        used = {modality: 0 for modality in MEDIA_MARKERS}
        position = 0
        while position < len(text):
            modality, start, marker = self._next_marker(text, position)
            if modality is None:
                break
            out.extend(self._encode(text[position:start]))
            index = used[modality]
            if index >= len(pending[modality]):
                raise ValueError(
                    f"prompt has more {modality} markers than {modality} items: "
                    f"marker {index + 1} with only {len(pending[modality])} provided"
                )
            out.extend(pending[modality][index])
            item_order.append({"modality": modality, "index": index})
            used[modality] = index + 1
            position = start + len(marker)
        out.extend(self._encode(text[position:]))

        for modality, count in used.items():
            if count != len(pending[modality]):
                raise ValueError(
                    f"prompt has {count} {modality} markers but "
                    f"{len(pending[modality])} {modality} items were provided"
                )
        return out, item_order

    def _next_marker(self, text: str, position: int) -> Tuple[Optional[str], int, str]:
        """The first marker at or after ``position``: modality, start, spelling.

        Every modality is searched together and the earliest wins, so an
        interleaved prompt keeps its media in the order they were written.
        Every accepted spelling of a modality is searched, because which one
        appears depends on the checkpoint's chat template; the matched spelling
        comes back so the caller advances past the marker it actually found.
        On a tie the longer spelling wins, so a marker is never consumed as a
        prefix of itself.
        """
        best_modality, best_start, best_marker = None, len(text), ""
        for modality, markers in MEDIA_MARKER_ALIASES.items():
            for marker in markers:
                found = text.find(marker, position)
                if found == -1:
                    continue
                if found < best_start or (found == best_start and len(marker) > len(best_marker)):
                    best_modality, best_start, best_marker = modality, found, marker
        return best_modality, best_start, best_marker

    def _encode(self, text: str) -> List[int]:
        """Tokenize a text segment, through the lazily-built tokenizer."""
        return self.tokenizer.encode(text, add_special_tokens=False) if text else []


def _image_size(image: Any) -> Tuple[int, int]:
    """``(width, height)`` of whichever image representation arrived.

    The media loaders disagree on purpose -- the hashing path feeds CHW tensors
    to avoid a ToPIL round-trip, the "np" format yields HWC uint8, and direct
    callers still pass PIL -- so the size, which is all this module wants, is
    read from whichever came.
    """
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:  # PIL: (width, height)
        return int(size[0]), int(size[1])
    shape = getattr(image, "shape", None)
    if shape is None:
        raise TypeError(f"cannot read an image size from {type(image).__name__}")
    if len(shape) == 3 and int(shape[0]) in (1, 3, 4):  # CHW tensor
        return int(shape[2]), int(shape[1])
    return int(shape[1]), int(shape[0])  # HWC array


def _to_pil(image: Any) -> Any:
    """Whatever image representation arrived, as a PIL image.

    The media loaders hand over three different things and which one is not the
    caller's choice: the hashing path feeds CHW tensors to avoid a ToPIL
    round-trip, the "np" media format yields HWC uint8, and a direct caller
    still passes PIL. The preprocessing downstream is written against PIL --
    ``to_rgb`` composites RGBA by mode and ``resize_lanczos`` calls
    ``Image.resize`` -- so a tensor reaching it raised ``'Tensor' object has no
    attribute 'convert'``. The hashing path CATCHES that and logs "Multimodal
    hashing failed:", so the symptom was not a failed request but a served
    engine that answered while silently re-running the tower on every
    chunked-prefill chunk. Normalizing here keeps the knowledge at the one
    boundary where the representations differ, instead of spreading isinstance
    checks through the preprocessing.
    """
    from PIL import Image  # noqa: PLC0415

    if isinstance(image, Image.Image):
        return image
    array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = array.transpose(1, 2, 0)  # CHW -> HWC
    if array.dtype != np.uint8:
        # Float loaders use the 0..1 convention; anything wider is already
        # 0..255 and only needs the dtype narrowed.
        array = array * 255.0 if float(array.max()) <= 1.0 else array
        array = array.round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array, mode="RGBA" if array.shape[-1] == 4 else "RGB")


def _frames_and_fps(
    video: Any, processor_kwargs: Dict[str, Any]
) -> Tuple[Any, float, Dict[str, Any]]:
    """Split a video item into frames, its source frame rate, and how it was sampled.

    The frame rate is not optional and has no safe default: sampling is 2 fps
    out of the source rate, so guessing it changes how many frames are kept and
    therefore how many placeholder tokens the prompt gets -- which would show up
    as a length mismatch against the encoder much later, or not at all.

    The sampling metadata matters for the same reason and was previously
    dropped. When a decoder has already subsampled, ``frames`` is not the whole
    video: ``frames_indices`` are the ORIGINAL indices of what survived, so a
    timestamp is ``original_index / fps`` rather than ``position / fps``, and
    ``total_num_frames`` is the source count the 2-fps target is computed
    against. Without them a decoder-subsampled video gets both the wrong clip
    timestamps and the wrong number of clips.

    The serving path delivers a ``VideoData``, whose ``metadata`` carries all
    three; the dict and pair forms are for callers holding frames they decoded
    themselves.
    """
    frames, fps = video, None
    sampling: Dict[str, Any] = {}
    metadata = getattr(video, "metadata", None)
    if metadata is not None and hasattr(video, "frames"):  # VideoData
        frames, fps = video.frames, metadata.get("fps")
        sampling = _sampling_from(metadata)
    elif isinstance(video, dict):
        frames = video.get("frames", video.get("video"))
        fps = video.get("fps")
        sampling = _sampling_from(video)
    elif isinstance(video, tuple) and len(video) == 2:
        frames, fps = video
    if fps is None:
        fps = processor_kwargs.get("fps")
    if fps is None:
        raise ValueError(
            "MuseSparkOpen needs the source frame rate to sample a video: pass it "
            "as mm_processor_kwargs={'fps': ...}, or hand the video over as "
            "{'frames': ..., 'fps': ...}."
        )
    return frames, float(fps), sampling


def _sampling_from(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """The two sampling fields ``preprocess_video`` accepts, when present.

    Both are omitted rather than defaulted when the decoder did not report
    them, because ``preprocess_video``'s own defaults ("frames is the whole
    video") are the right answer in that case and a guess here would not be.
    """
    sampling: Dict[str, Any] = {}
    indices = metadata.get("frames_indices", metadata.get("frame_indices"))
    if indices is not None:
        sampling["frame_indices"] = list(indices)
    total = metadata.get("total_num_frames")
    if total is not None:
        sampling["total_num_frames"] = int(total)
    return sampling


#: The two PIXEL modalities the vision tower encodes, and where each keeps its
#: geometry. Iterating this in a fixed order is what made mixed requests wrong
#: before ``_items_in_prompt_order`` existed -- see its docstring. Audio is
#: handled alongside, not here: its items are waveforms, not pixel/grid pairs.
_VISION_MODALITIES: Tuple[Tuple[str, str], ...] = (
    ("image", "image_grid_thw"),
    ("video", "video_grid_thw"),
)


def validate_vision_segment_lengths(mm_data: Dict[str, Any], max_seq_len: int) -> None:
    """Reject, at admission, any clip whose full-attention segment cannot run.

    The tower presents each temporal clip to full attention as one segment of
    ``grid_h * grid_w`` patches, and ``prepare_attn_metadata`` raises when a
    segment exceeds the configured maximum. That check fires mid-forward,
    where the executor's error handling fails every request in the scheduled
    batch, so the same arithmetic runs here too, per request, before
    scheduling. The grids are host tensors the input processor computed, so
    this is a cheap host read. The message matches the tower's so clients see
    the same error text wherever the condition is caught.
    """
    if not isinstance(mm_data, dict) or max_seq_len <= 0:
        return
    for modality, grid_key in _VISION_MODALITIES:
        item = mm_data.get(modality)
        if not isinstance(item, dict) or item.get("pixel_values") is None:
            continue
        grid = item.get(grid_key)
        if grid is None:
            continue
        for row in grid:
            segment = int(row[1]) * int(row[2])
            if segment > max_seq_len:
                raise ValueError(
                    f"vision attention segment length {segment} exceeds the configured maximum "
                    f"{max_seq_len}; raise the encoder token budget or lower max_pixels"
                )


def _per_item_chunks(
    item: Dict[str, Any], grid_key: str
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Split one modality's packed tensors back into per-item pairs.

    ``call_with_text_prompt`` concatenates every item of a modality into one
    ``pixel_values`` and stacks their grids, so item boundaries survive only in
    the grid: row ``i`` describes ``t * h * w`` patches. Reading them off the
    grid is a HOST read, not a device sync -- ``multimodal_data_device_paths``
    deliberately leaves ``*_grid_thw`` behind, and the tower's
    ``_host_grid_rows`` raises if it ever arrives on the device.
    """
    values, grid = item["pixel_values"], item[grid_key]
    chunks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    start = 0
    for row in grid:
        patches = int(row[0]) * int(row[1]) * int(row[2])
        chunks.append((values[start : start + patches], row.reshape(1, -1)))
        start += patches
    if start != values.shape[0]:
        raise ValueError(
            f"{grid_key} describes {start} patches but pixel_values holds "
            f"{values.shape[0]}; the input processor and the encoder disagree on geometry"
        )
    return chunks


def _per_audio_chunks(item: Dict[str, Any]) -> List[torch.Tensor]:
    """Split one request's packed audio back into per-clip waveforms.

    ``call_with_text_prompt`` concatenates every clip into one 1-D
    ``waveforms`` tensor; the boundaries survive in ``sample_counts``, which
    stays on the HOST (it is absent from ``multimodal_data_device_paths``), so
    reading it here is not a device sync.
    """
    waveforms, counts = item["waveforms"], item["sample_counts"]
    lengths = [int(count) for count in counts]
    if sum(lengths) != waveforms.shape[0]:
        raise ValueError(
            f"audio sample_counts describes {sum(lengths)} samples but waveforms "
            f"holds {waveforms.shape[0]}; the input processor and the encoder "
            "disagree on the packing"
        )
    return list(torch.split(waveforms, lengths))


def _items_in_prompt_order(params: Any) -> List[Tuple[str, Any]]:
    """One request's media items as ``(modality, payload)``, in prompt order.

    The payload is ``(pixel_values, grid_thw)`` for image/video and a 1-D
    waveform tensor for audio.

    The rows this encoder returns are consumed positionally: ``fuse_input_embeds``
    writes them into the ``mm_token_ids`` positions of the prompt, left to right.
    So row order has to be PROMPT order, and iterating modalities in a fixed
    order is not that. A ``<video><image>`` prompt made the point concretely --
    every image was encoded before every video, so the video's placeholder run
    received the image's rows and vice versa, and the model answered about the
    wrong medium with no error anywhere.

    The manifest is one ``{"modality", "index"}`` entry per item in prompt
    order. Two sources, in this priority:

    1. ``multimodal_data["mm_item_order"]``, written by our own input processor
       from the loop that actually spliced the placeholder runs into the token
       stream. This is the authoritative one and it is preferred, because it
       rides the same dict as the pixels and therefore arrives whenever they do.
    2. ``MultimodalParams.mm_item_order``, the framework's field, carrying the
       order the chat handler decoded the media in.

    Source 2 alone is not enough, and that is measured rather than assumed: on
    2026-09-09 a served ``image``+``video`` request reached this function with
    the framework field empty, even though the chat front end had built the
    manifest correctly -- so it is lost somewhere between the two, and a
    manifest that is only sometimes present cannot be the contract. Keeping the
    framework field as source 2 costs nothing and covers entrypoints that do not
    go through our processor.

    Reordering the encoder INPUT rather than its output is exact here because
    the tower is per-item segmented -- ``cu_seqlens``, the window permutation and
    the pixel-shuffle permutation are all built per clip from the grid -- so
    permuting items permutes the output rows and changes nothing else.

    Requests that carry a single modality need no manifest: within a modality,
    item order already is prompt order. A request with both and no manifest from
    either source cannot be ordered at all and is refused rather than silently
    guessed. That refusal is now unreachable from anything our processor built,
    which matters: this runs inside ``forward``, so raising here takes the whole
    executor down rather than failing one request.
    """
    data = params.multimodal_data or {}
    per_modality: Dict[str, List[Any]] = {
        modality: _per_item_chunks(data[modality], grid_key)
        for modality, grid_key in _VISION_MODALITIES
        if data.get(modality)
    }
    if data.get("audio"):
        per_modality["audio"] = _per_audio_chunks(data["audio"])
    if not per_modality:
        return []

    manifest = data.get("mm_item_order") or getattr(params, "mm_item_order", None)
    if not manifest:
        if len(per_modality) > 1:
            raise ValueError(
                f"request carries {sorted(per_modality)} but no mm_item_order; "
                "a mixed-modality request must state its prompt order"
            )
        modality, chunks = next(iter(per_modality.items()))
        return [(modality, chunk) for chunk in chunks]

    ordered: List[Tuple[str, Any]] = []
    for entry in manifest:
        chunks = per_modality.get(entry["modality"])
        if chunks is None:
            # A modality that arrived with no data attached; the count check
            # below catches a manifest/data mismatch.
            continue
        ordered.append((entry["modality"], chunks[entry["index"]]))
    expected = sum(len(chunks) for chunks in per_modality.values())
    if len(ordered) != expected:
        raise ValueError(
            f"mm_item_order names {len(ordered)} of the {expected} media items "
            "attached to the request; every item must appear exactly once"
        )
    return ordered


class MuseSparkOpenVisionEncoder(nn.Module):
    """Runs the towers once over every media item in the batch.

    Despite the (registration-fixed) name this is the model's whole perception
    side: the packed NaViT vision tower plus, when the config enables it, the
    ``wav_ae`` audio tower. Returns a single tensor whose first dimension is
    the total placeholder count across the batch. That is a hard contract, not
    a convenience: if the length disagrees with what the input processor
    emitted, ``get_multimodal_embeddings`` silently declines to cache the
    result and the tower is re-run from scratch on every chunked-prefill chunk.

    Rows come out in request order and, within a request, in prompt order --
    see :func:`_items_in_prompt_order` for why the second half of that is not
    free. Both towers emit decoder-width (``hidden_size``) rows, and both
    return them UN-normed: the decoder applies its weightless ``embed_norm``
    to the merged embedding tensor exactly once, text/vision/audio alike,
    which is per-row identical to the references' pre-norm-the-audio-rows
    ordering (audio merges before the token-embedding norm in training).
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        self.tower = MuseSparkOpenNaViTVisionTower(model_config)
        config = model_config.pretrained_config
        self._vision_config = config.vision_config
        # Same config gate as the input processor and both references. The
        # audio tower is plain-torch BF16 and NEVER quantized: the checkpoint
        # stores it BF16 with no scales, and hf_quant_config.json's
        # exclude_modules does not (and cannot) exempt it -- see
        # musespark_open_audio.py for the full statement.
        audio_args = getattr(config, "audio_perception_modality_args", None) or {}
        self.audio_tower: Optional[MuseSparkOpenAudioEncoder] = None
        if audio_args.get("input_mode") == "wav":
            self.audio_tower = MuseSparkOpenAudioEncoder(config.hidden_size)

    def forward(self, multimodal_params: List[Any]) -> torch.Tensor:
        ordered: List[Tuple[str, Any]] = []
        for params in multimodal_params:
            ordered.extend(_items_in_prompt_order(params))
        if not ordered:
            return torch.empty(0)

        pixel_values, grids, waveforms = [], [], []
        for modality, payload in ordered:
            if modality == "audio":
                waveforms.append(payload)
            else:
                values, grid = payload
                pixel_values.append(values)
                grids.append(grid)

        vision_rows = None
        if pixel_values:
            # Casting to the tower's dtype is this wrapper's job, and nothing
            # upstream does it. The preprocessing normalizes ``(x/255 - 0.5)/0.5``
            # in fp32 on purpose -- that is the reference's arithmetic -- and the
            # engine stages ``image.pixel_values`` to the device without changing
            # its dtype, because ``multimodal_data_device_paths`` only moves
            # tensors. So a bf16 tower was being handed an fp32 tensor and
            # ``conv1_linear`` raised "expected mat1 and mat2 to have the same
            # dtype, but got: float != c10::BFloat16" on every image request that
            # reached the engine. The module-level parity harnesses never saw it
            # because each applied this cast itself before calling the tower.
            packed = torch.cat(pixel_values, dim=0).to(dtype=self.tower.dtype)
            # One tower pass for the whole vision batch: the tower is packed
            # variable-resolution, so per-item passes would only add launches.
            vision_rows = self.tower(packed, torch.cat(grids, dim=0))
        if not waveforms:
            return vision_rows

        if self.audio_tower is None:
            # The input processor refuses audio when the config disables it,
            # so reaching here means data arrived through a path that skipped
            # it. Refuse rather than drop: a silently dropped clip answers
            # from text alone with no error anywhere.
            raise ValueError(
                "audio items reached the encoder but the config does not enable "
                "the waveform encoder (audio_perception_modality_args.input_mode "
                "!= 'wav')"
            )
        # Per-clip passes, as in both references: clips have unequal lengths
        # and each is padded independently to its own multiple of 1920. The
        # tower casts the float32 waveform to its own dtype (BF16).
        audio_rows = self.audio_tower(waveforms)
        if vision_rows is None:
            return torch.cat(audio_rows, dim=0)

        # Interleave back into prompt order. The vision tower emitted its rows
        # in the order its items appear in ``ordered``; per-item row counts
        # come from the grids, a host read (see _per_item_chunks).
        rows: List[torch.Tensor] = []
        vision_offset, audio_index = 0, 0
        for modality, payload in ordered:
            if modality == "audio":
                rows.append(audio_rows[audio_index])
                audio_index += 1
            else:
                _, grid = payload
                count = num_placeholder_rows(
                    [int(x) for x in grid.reshape(-1)], self._vision_config
                )
                rows.append(vision_rows[vision_offset : vision_offset + count])
                vision_offset += count
        if vision_offset != vision_rows.shape[0]:
            raise ValueError(
                f"vision tower emitted {vision_rows.shape[0]} rows but the grids "
                f"describe {vision_offset}; the interleave would misalign audio"
            )
        return torch.cat(rows, dim=0)


@register_vision_encoder(MuseSparkOpenVisionEncoder)
@register_auto_model("MuseSparkOpenOmniForCausalLM")
@register_input_processor(
    MuseSparkOpenOmniInputProcessor,
    model_type="musespark_open_omni",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map=MEDIA_MARKERS,
        # Placement is unused under OPENAI: the template writes each marker
        # where the content part sits, so media stay interleaved with the text.
        placeholder_placement=MultimodalPlaceholderPlacement.AFTER_TEXT,
        placeholders_separator="",
        # The chat template consumes content dicts natively and emits the
        # markers itself. STRING would be actively wrong here: it merges the
        # placeholder into the message text, and the template's esc() rewrites a
        # literal "<image>" in text to "< image>" -- the media would be dropped
        # and the request would answer about nothing.
        content_format=ContentFormat.OPENAI,
    ),
)
class MuseSparkOpenOmniForConditionalGeneration(MultimodalModelMixin, PreTrainedModel):
    """The vision tower, the decoder, and the merge between them.

    This class owns the checkpoint's only architecture string,
    ``MuseSparkOpenOmniForCausalLM``. It used to name the pure-text
    :class:`MuseSparkOpenForCausalLM`, which was right while there was no tower;
    now that there is one, a text-only registration would make the vision
    weights unreachable for the checkpoints that carry them. Registration is
    first-wins, so this is a takeover and not an addition: the decoder keeps no
    architecture string of its own and is reached by import, from here and from
    its tests. A request with no media is a straight pass-through to it
    (``fuse_input_embeds`` with an empty embed list), which is what makes the
    takeover safe and what ``test_musespark_open_omni_textpath.py`` pins.

    **It inherits ``MultimodalModelMixin`` and uses the mixin's ``forward``.**
    The mixin is what ``_util.py:_create_dummy_encoder_inputs`` tests for with
    ``isinstance``, and that gate is the only door to startup encoder profiling:
    a non-mixin model returns immediately there, so the profiler never runs the
    tower, never sees its activations, and hands the whole unmeasured amount to
    the KV cache.

    Using the mixin's merge is safe here, and that is a statement about the call
    graph rather than a hope. Both paths call the same
    ``fuse_input_embeds`` (``modeling_multimodal_utils.py``), which with
    non-None ``mm_token_ids`` takes the in-vocab fast path: embed the whole
    ``input_ids`` through the embedding layer, then overwrite the vision rows.
    So as long as :attr:`text_embedding_layer` is the RAW ``Embedding`` and
    :attr:`multimodal_token_ids` names the same two ids, the ``inputs_embeds``
    handed to the decoder is the same tensor a hand-written forward produced.

    The delicate part of this model is where the norm sits, and the mixin does
    not move it: the tower returns un-normed rows on purpose, and
    ``MuseSparkOpenModel.forward`` applies the weightless ``embed_norm`` and
    ``m_emb`` to the whole ``inputs_embeds`` it receives, text and vision alike,
    exactly once. Handing ``text_embedding_layer`` a norm-wrapped module instead
    of the raw ``Embedding`` would norm text twice; that invariant is pinned by
    ``test_musespark_open_omni_textpath.py`` and the merge numbers by
    ``test_musespark_open_omni_merge.py``.

    Two behaviours DO change with the mixin's forward, both in the safe
    direction. ``select_multimodal_params`` drops context params with no active
    multimodal tokens in the chunk before the encoder runs, where the old
    forward passed the whole context slice; and ``_validate_embeddings`` raises
    on a row-count mismatch where ``get_multimodal_embeddings`` only warned and
    silently declined to cache -- a mismatch there means the tower re-runs on
    every chunked-prefill chunk, so failing loudly is what we want.

    The mixin's other two features stay off because both are opt-in ClassVars
    left at their ``False`` defaults: ``supports_encoder_cache`` (the persistent
    cross-request cache -- ``encoder_cache_active`` returns False, so
    ``_util.py`` reserves no cache bytes) and
    ``supports_mm_encoder_item_scheduling`` (item scheduling and its
    ``MultimodalScheduler`` wrap). Turning either on brings contracts of its own
    (``build_multimodal_encoder_input``, an item-metadata provider) and neither
    is needed for parity. The per-request, per-chunk caching in
    ``get_multimodal_embeddings`` is untouched and still depends only on the
    encoder returning one tensor whose first dim equals the placeholder count.
    """

    config_class = type(None)  # resolved from model_config, not from a HF class

    # Breakable prefill CUDA graphs are safe here: the wrapped decoder is a
    # plain ``DecoderModelForCausalLM`` (``self.llm``), and the mixin's no-MM
    # ``prepare_multimodal_inputs`` path passes the executor's persistent
    # ``input_ids`` buffer through untouched, which is the buffer the captured
    # graph reads on replay. Multimodal prefill batches never replay -- the
    # executor gate (``model_engine.forward``) sends any batch carrying
    # ``multimodal_params`` or ``inputs_embeds`` down the eager path.
    supports_breakable_prefill_cuda_graph = True

    # -- class-level contracts the takeover must not drop ----------------
    #
    # The framework asks these three of the registered CLASS, before any
    # instance exists: ``get_model_defaults`` at
    # ``pyexecutor/model_loader.py:712``, ``get_preferred_transceiver_runtime``
    # at ``llmapi/llm_utils.py:733``, ``get_preferred_kv_cache_manager_version``
    # at ``llmapi/llm_utils.py:636``. So a property or an instance attribute
    # cannot carry them, and the ``language_model`` property below -- which is
    # how every other decoder contract survives the takeover -- does not reach
    # them either.
    #
    # All three lookups are ``getattr``/``hasattr`` guarded, because "this model
    # has no preference" is a legal answer. That is what made the omission
    # silent and expensive: registration moved the architecture string from
    # ``MuseSparkOpenForCausalLM`` to this wrapper, the resolver then found no
    # method here, left ``preferred_version`` at ``None``, and built a V1 KV
    # cache manager for a model whose 46 windowed + 16 full-context layers are
    # the reason it asks for V2. No error, no warning; a parity test caught
    # it only because it probes the manager the engine actually built and
    # compares it against what the model class asks for.
    #
    # Answering by DELEGATION rather than by repeating ``"V2"`` is the point.
    # The defect is a class-level contract lost in a takeover, so the fix that
    # matches it is the one that cannot be lost again: a copied constant would
    # need hand-syncing the next time the decoder gains or changes a preference,
    # and nothing would fail until an engine silently picked the wrong manager.
    # ``test_musespark_open_omni_textpath.py`` pins the set of hooks as well as
    # their values, so a fourth hook added upstream fails the test rather than
    # being quietly dropped here.
    @classmethod
    def get_preferred_kv_cache_manager_version(
        cls, pretrained_config: Any = None
    ) -> Optional[Literal["V1", "V2"]]:
        """Whatever the wrapped decoder prefers; today ``"V2"``."""
        return MuseSparkOpenForCausalLM.get_preferred_kv_cache_manager_version(pretrained_config)

    @classmethod
    def requires_kv_cache_manager_v2(cls, pretrained_config: Any = None) -> bool:
        """Whatever the wrapped decoder requires; today ``True``."""
        return MuseSparkOpenForCausalLM.requires_kv_cache_manager_v2(pretrained_config)

    @classmethod
    def get_preferred_transceiver_runtime(
        cls, pretrained_config: Any = None
    ) -> Optional[Literal["CPP", "PYTHON"]]:
        """Whatever the wrapped decoder prefers; today no preference.

        Delegated rather than pinned to ``"PYTHON"``. The sibling VLM wrappers
        pin it so that disaggregated serving over NIXL keeps a V2 manager, but
        the decoder this wraps states no preference, and inventing one here
        would give the wrapper a disaggregation behaviour its own decoder does
        not have -- a change no gate in this tier measures.
        """
        return MuseSparkOpenForCausalLM.get_preferred_transceiver_runtime(pretrained_config)

    @classmethod
    def get_model_defaults(cls, llm_args: Any) -> Dict[str, Any]:
        """Whatever the wrapped decoder defaults to; today ``{}``."""
        return MuseSparkOpenForCausalLM.get_model_defaults(llm_args)

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        config = model_config.pretrained_config
        super().__init__(config)
        if hasattr(self, "llm"):
            return  # re-entry guard: post_config() can bring us back here

        self.mm_encoder = None
        if not model_config.disable_mm_encoder:
            self.mm_encoder = MuseSparkOpenVisionEncoder(model_config)
        self.llm = MuseSparkOpenForCausalLM(model_config)

        self.model_config = model_config
        self.post_config()

    def post_config(self):
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    @property
    def language_model(self) -> MuseSparkOpenForCausalLM:
        """The text decoder this wraps, under the name the framework uses.

        ``MultimodalModelMixin`` calls its inner decoder ``language_model`` and
        everything that reaches past a VLM wrapper -- the guided decoder, the
        bring-up tests' layer probes -- looks for that name. Spelling it here
        means the takeover did not quietly rename the decoder for every caller
        that used to get it straight from the architecture string.
        """
        return self.llm

    # ``vocab_size_padded`` and ``infer_max_seq_len`` are NOT repeated here:
    # the mixin already answers both by delegating to ``language_model``, which
    # is the decoder. A copy would be one more thing to keep in sync.

    # The speculative-decoding members live on the decoder (it subclasses
    # ``SpecDecOneEngineForCausalLM``), but ``model_loader`` materializes
    # draft weights against the TOP-level model -- the class the checkpoint's
    # architecture string names, i.e. this wrapper. Without these delegations
    # an external-drafter config (DFlash) dies in
    # ``_materialize_draft_checkpoint_weights`` with
    # ``no attribute 'draft_config'``.

    @property
    def draft_config(self):
        return self.llm.draft_config

    @property
    def draft_model(self):
        return self.llm.draft_model

    @property
    def spec_worker(self):
        return self.llm.spec_worker

    @property
    def use_separate_draft_kv_cache(self) -> bool:
        return self.llm.use_separate_draft_kv_cache

    def load_draft_weights(self, *args, **kwargs):
        return self.llm.load_draft_weights(*args, **kwargs)

    @property
    def multimodal_token_ids(self) -> torch.IntTensor:
        """The in-vocab ids that carry encoder rows.

        Handed to ``fuse_input_embeds`` so it can locate the media positions
        without an out-of-vocabulary sentinel and without a device sync.
        ``<|audio|>`` carries audio rows; its delimiters keep their own text
        embeddings and are deliberately absent.
        """
        return torch.tensor(
            [VISION_TOKEN_IDS["patch"], VISION_TOKEN_IDS["video"], AUDIO_TOKEN_IDS["audio"]],
            dtype=torch.int32,
        )

    @property
    def mm_token_ids(self) -> torch.IntTensor:
        """Deprecated spelling of :attr:`multimodal_token_ids`.

        ``model_engine.py`` prefers the new name and falls back to this one, and
        the bring-up harnesses still read it. Kept as an alias so neither breaks
        while the older spelling is retired.
        """
        return self.multimodal_token_ids

    @property
    def text_embedding_layer(self) -> Embedding:
        """The RAW token embedding, deliberately not a norm-wrapped module.

        ``fuse_input_embeds`` embeds the whole prompt through this layer and
        then overwrites the vision rows, so whatever runs here runs on the text
        rows. The decoder applies ``embed_norm`` (and ``m_emb``) to the merged
        tensor afterwards, once, over text and vision alike -- returning
        anything normed here would norm the text twice, which is not a no-op in
        bf16 at small row scale. That is why this returns
        ``llm.model.embed_tokens`` and never ``embed_norm``.
        """
        return self.llm.model.embed_tokens

    @property
    def embedding_dim(self) -> int:
        return self.text_embedding_layer.embedding_dim

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.text_embedding_layer.weight.dtype

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        """Only the pixels and the packed waveforms go to the device.

        ``*_grid_thw`` and ``audio.sample_counts`` are deliberately absent.
        They are geometry the towers need on the *host* -- to segment the
        packed batches into per-item runs -- so staging them on the device
        would only force the towers to read them back, which is a sync inside
        ``forward`` and the one thing Contract 1 bans. A handful of ints per
        item, not tensors worth moving.
        """
        return [
            "image.pixel_values",
            "video.pixel_values",
            "audio.waveforms",
            "multimodal_embedding",
        ]

    def encode_multimodal_inputs(self, multimodal_params: List[Any]) -> torch.Tensor:
        """The mixin's uniform encoder entry; one tower pass over the batch.

        The only inherited method the framework calls directly. Its one caller
        is the startup KV-cache profiler (``_util.py:1012``), which runs the
        encoder at the scheduling limits so the tower's activations land in the
        measured peak instead of being handed to the KV cache. The request path
        does not come through here -- ``forward`` reaches the same encoder via
        ``get_multimodal_embeddings``, which is what keeps the per-chunk cache.

        Delegation, not a second implementation: the wrapper's dtype cast and
        the one-tensor return contract are in ``MuseSparkOpenVisionEncoder`` and
        must not be duplicated where they could drift apart.
        """
        return self.mm_encoder.forward(multimodal_params)

    # No ``forward`` here on purpose: ``MultimodalModelMixin.forward`` runs the
    # encoder, selects the rows active in this chunk and calls the same
    # ``fuse_input_embeds`` a hand-written one would. See the class docstring
    # for why that is merge-neutral and which two invariants make it so.

    def get_language_model_extra_forward_kwargs(
        self,
        *,
        raw_input_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        mm_inputs: Any,
        spec_metadata: Any = None,
        lora_params: Optional[Dict[str, Any]] = None,
        resource_manager: Any = None,
        **forward_kwargs: Any,
    ) -> Dict[str, Any]:
        """Forward ``spec_metadata``, ``lora_params`` and ``resource_manager`` to the decoder.

        The mixin's default drops all three, and so did the wrapper's own forward, so
        neither speculative decoding nor LoRA could reach the decoder through
        this class -- silently, because ``DecoderModelForCausalLM.forward``
        defaults them to ``None`` rather than complaining. The resource manager
        carries the draft KV cache a one-model drafter reads (DFlash on TRTLLM
        binds its context cache to it); without it the drafter falls back to a
        private arena.
        """
        del raw_input_ids, position_ids, mm_inputs, forward_kwargs
        return {
            "spec_metadata": spec_metadata,
            "lora_params": lora_params,
            "resource_manager": resource_manager,
        }

    def load_weights(self, weights, weight_mapper=None) -> Dict[str, int]:
        """Load the tower and the decoder from one checkpoint.

        Both towers come out of the same shard set, so the accounting is done
        once over the whole key list rather than twice over halves of it.
        """
        if self.mm_encoder is None:
            return self.llm.load_weights(weights, weight_mapper)
        summary = load_musespark_open_omni_weights(self, weights)
        logger.info(f"MuseSparkOpenOmni weight accounting: {summary}")
        return summary

    # The towers are BF16 and were never quantized: all perception tensors
    # (vision AND the 72 audio tensors) are BF16 and the checkpoint carries no
    # scale tensor under any perception prefix. Note that
    # ``hf_quant_config.json``'s exclude_modules does NOT name either tower --
    # relying on that list would build NVFP4 towers -- so the exemption is
    # explicit in code instead: the vision tower modules are built with
    # quant_config=None, and the audio tower is plain ``nn.Linear`` /
    # ``nn.Conv1d`` that the quantization plumbing structurally cannot reach
    # (see musespark_open_audio.py).
    @property
    def vision_tower(self) -> MuseSparkOpenNaViTVisionTower:
        return self.mm_encoder.tower

    @property
    def audio_encoder(self) -> Optional[MuseSparkOpenAudioEncoder]:
        """The wav_ae tower, or None when the config or CLI disables it.

        The weight mapper keys the audio-tensor disposition off this: present
        means the 72 checkpoint tensors are consumed, absent means they keep
        the loud-guard skip.
        """
        if self.mm_encoder is None:
            return None
        return self.mm_encoder.audio_tower

    def validate_multimodal_request_data(self, mm_data: Dict[str, Any]) -> None:
        """Admission check: every vision clip must fit the tower's segment cap.

        This model opts out of item scheduling, so the executor's only other
        multimodal admission check (``initialize_multimodal_encoder_request``)
        never runs for it. Without this hook an over-budget image is caught
        only by ``prepare_attn_metadata`` inside the tower forward, which
        fails every request in the scheduled batch instead of just this one.
        """
        if self.mm_encoder is None:
            return
        tower = getattr(self.mm_encoder, "tower", None)
        if tower is None:
            return
        validate_vision_segment_lengths(mm_data, tower.attn_max_seq_len)
