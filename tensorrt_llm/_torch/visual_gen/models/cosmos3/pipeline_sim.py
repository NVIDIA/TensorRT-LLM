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
"""Cosmos3 Sim: the chunked, causal rollout pipeline.

``Cosmos3SimPipeline`` is registered under the ``_class_name`` of each checkpoint it
serves (``Cosmos3NanoSimBimanualPipeline``). The exact name is what keeps
``AutoPipeline`` from routing such a checkpoint to the bidirectional
``Cosmos3OmniMoTPipeline`` through its ``"Cosmos3" in class_name`` fallback, where
it would load and produce wrong video.
"""

from __future__ import annotations

import contextlib
import itertools
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import PIL.Image
import torch
import torch.distributed as dist
from diffusers.utils.torch_utils import randn_tensor

from tensorrt_llm._utils import nvtx_range, prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div

from ...cache import BlockCausalKVCacheManager, RolloutSpec
from ...cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from ...pipeline_registry import register_pipeline
from ...utils import make_noise_generator, postprocess_video_tensor
from ..wan.vae_loader import load_wino_vae
from .action import (
    ACTION_MODE_FORWARD_DYNAMICS,
    crop_action_latent,
    normalize_action_mode,
    resolve_domain_id,
    vision_condition_indexes,
)
from .defaults import VIDEO_RES_SIZE_INFO, resolve_raw_action_dim
from .parameter_resolver import SIM_CONFIG_KEY, Cosmos3CheckpointSettings, Cosmos3Embodiment
from .pipeline_cosmos3 import Cosmos3OmniMoTPipeline
from .sampling import FixedStochasticSchedule

# Tokens per K/V cache page. The cache takes any power of two; the attention kernels
# decide: trtllm-gen ships its paged context kernels for 32-token pages, cuDNN's paged
# graph takes any size. The pool is built only after the backend confirms the size.
KV_TOKENS_PER_PAGE = 32

# Chunks of frames a conv's frame buffer holds in the VAE that decodes on the rollout's
# GPU, where that memory comes out of the K/V cache.
ROLLOUT_GPU_FRAME_BUFFER_CHUNKS = 1

PIPELINE_CONFIG_DEFAULTS = {
    # Fraction of the GPU memory left after the warmup peak that the K/V cache takes.
    "kv_cache_free_gpu_memory_fraction": 0.9,
    # Latent frames a generated frame attends to, counting the attention sinks and the
    # frame itself. None: what the checkpoint declares.
    "kv_cache_inference_size": None,
    # Leading latent frames kept visible once the window slides past them. Needs a
    # window. None: what the checkpoint declares.
    "attention_sink_size": None,
    # GPU that decodes finished chunks, e.g. "cuda:1". None: the generator's own.
    "vae_decode_device": None,
}
"""The ``pipeline_config`` keys of the Sim pipeline. All are fixed for the engine's
lifetime: the cache is sized at warmup and CUDA graphs are captured over its layout."""


def _validate_domain_ids(value: Any) -> None:
    ids = value if isinstance(value, (list, tuple)) else [value]
    if not ids or any(isinstance(d, bool) or not isinstance(d, int) for d in ids):
        raise ValueError("domain_id must be an integer or a non-empty list of integers")


def _validate_domain_names(value: Any) -> None:
    names = value if isinstance(value, (list, tuple)) else [value]
    if not names or any(not isinstance(n, str) for n in names):
        raise ValueError("domain_name must be a string or a non-empty list of strings")


def split_row_domain_ids(domain_id: Any) -> tuple[Any, Optional[tuple[int, ...]]]:
    """A request's ``domain_id`` is one id or one id per action row. Returns the id
    the generic resolution should see (the first row's) and the per-row tuple, or
    ``(domain_id, None)`` for a single id."""
    if not isinstance(domain_id, (list, tuple)):
        return domain_id, None
    if not domain_id:
        raise ValueError("Cosmos3 Sim domain_id list must not be empty.")
    row_ids = tuple(int(d) for d in domain_id)
    return row_ids[0], row_ids


def _gpu_index(device: torch.device) -> int:
    """The GPU a CUDA device names; ``cuda`` without an index is the current one."""
    return torch.cuda.current_device() if device.index is None else device.index


def domain_runs(row_ids: Sequence[int], lead_rows: int = 0) -> list[tuple[int, int]]:
    """``(row_count, domain_id)`` of each stretch of consecutive action rows that share
    an embodiment, in row order. ``lead_rows`` more rows ahead of the first stretch
    take its embodiment."""
    runs = [(sum(1 for _ in rows), int(domain)) for domain, rows in itertools.groupby(row_ids)]
    if not runs:
        raise ValueError("Cosmos3 Sim got no embodiment for its action rows.")
    runs[0] = (runs[0][0] + lead_rows, runs[0][1])
    return runs


@dataclass(frozen=True)
class Cosmos3SimSettings:
    """The rollout geometry the pipeline runs: the checkpoint's, with the serving
    config's window in place of the checkpoint's where it sets one. Frames are latent
    frames throughout."""

    chunk_frames: int
    """Latent frames generated together after the first chunk."""
    chunk_partition: tuple[int, ...]
    """Leading chunk sizes from the start; the last repeats."""
    history_frames: Optional[int]
    """Recent frames before a chunk that it attends to, sinks excluded: the window.
    ``None``: full history; the pipeline sizes the cache from the memory left after
    warmup."""
    sink_frames: int
    """Leading frames pinned for the whole rollout."""
    text_cache_max_len: int
    """Most prompt tokens the fixed region may hold; longer prompts are rejected."""
    action_tokens_per_frame: int
    """Action rows packed before each frame's vision tokens."""
    embodiments: dict[str, Cosmos3Embodiment]
    """Embodiments declared under ``conditioning.embodiments``, by lower-case name."""
    default_embodiment: str
    """``conditioning.default_embodiment``: the embodiment of a request that names
    none. One of ``embodiments``, with a domain id."""
    resolution: Optional[str]
    """The frame-size tier the checkpoint was trained at; ``None`` where it declares
    none."""

    def largest_frame(self) -> tuple[int, int]:
        """``(height, width)`` of the biggest frame of the checkpoint's resolution tier:
        the warmup shape when the serving config lists no warmup resolutions. Raises
        when the checkpoint declares no tier, or one without a frame-size table."""
        sizes = VIDEO_RES_SIZE_INFO.get(self.resolution)
        if sizes is None:
            raise ValueError(
                f"Cosmos3 Sim: the checkpoint's resolution tier is {self.resolution!r}, "
                f"which is not one of {sorted(VIDEO_RES_SIZE_INFO)}; list the first-frame "
                "sizes to warm up under compilation_config.resolutions."
            )
        shapes = [(h, w) for (w, h) in sizes.values()]
        return max(shapes, key=lambda hw: hw[0] * hw[1])

    @classmethod
    def from_config(
        cls,
        config: Cosmos3CheckpointSettings,
        *,
        inference_size: Optional[int] = None,
        sink_frames: Optional[int] = None,
    ) -> "Cosmos3SimSettings":
        """``inference_size`` and ``sink_frames`` are the serving config's window and
        sinks; ``None`` keeps the checkpoint's. Raises for a checkpoint without the
        rollout block and for sinks that do not fit the window."""
        if config.rollout is None:
            raise ValueError(
                f"Not a Cosmos3 Sim checkpoint: transformer/config.json has no "
                f"'{SIM_CONFIG_KEY}' block."
            )
        rollout = config.rollout.with_window(inference_size, sink_frames)
        return cls(
            chunk_frames=rollout.chunk_frames,
            chunk_partition=rollout.chunk_partition,
            history_frames=rollout.history_frames,
            sink_frames=rollout.sink_frames,
            text_cache_max_len=rollout.text_cache_max_len,
            action_tokens_per_frame=rollout.action_tokens_per_frame,
            embodiments=dict(rollout.embodiments),
            default_embodiment=rollout.default_embodiment,
            resolution=rollout.resolution,
        )

    def resolve_embodiment(
        self,
        domain_name: Optional[str],
        domain_id: Optional[int],
        raw_action_dim: Optional[int],
    ) -> tuple[Optional[str], Optional[int], Optional[int]]:
        """Complete a request's ``(domain_name, domain_id, raw_action_dim)`` from the
        checkpoint's embodiment table: a missing domain falls back to the declared
        default, a missing id or width is filled in, and a given one that disagrees
        with the checkpoint is an error. Embodiments the checkpoint does not list
        pass through untouched."""
        if (domain_name is None or not str(domain_name).strip()) and domain_id is None:
            domain_name = self.default_embodiment
        if domain_name is not None and str(domain_name).strip():
            name = str(domain_name).strip().lower()
        elif domain_id is not None:
            name = next(
                (n for n, e in self.embodiments.items() if e.domain_id == int(domain_id)), None
            )
        else:
            name = None
        spec = self.embodiments.get(name) if name is not None else None
        if spec is None:
            return domain_name, domain_id, raw_action_dim
        if spec.domain_id is not None:
            if domain_id is not None and int(domain_id) != spec.domain_id:
                raise ValueError(
                    f"Cosmos3 Sim domain_id={domain_id} contradicts the checkpoint, which "
                    f"maps {name!r} to domain_id={spec.domain_id}."
                )
            domain_id = spec.domain_id
        if spec.action_dim is not None:
            if raw_action_dim is not None and int(raw_action_dim) != spec.action_dim:
                raise ValueError(
                    f"Cosmos3 Sim raw_action_dim={raw_action_dim} contradicts the checkpoint, "
                    f"which expects {spec.action_dim} values per action row for {name!r}."
                )
            raw_action_dim = spec.action_dim
        return name, domain_id, raw_action_dim

    def row_domain_ids(self, names: Sequence[str]) -> tuple[int, ...]:
        """The domain id of each action row's embodiment, from the checkpoint's table."""
        if not names:
            raise ValueError("Cosmos3 Sim domain_name list must not be empty.")
        ids = []
        for name in names:
            spec = self.embodiments.get(str(name).strip().lower())
            if spec is None or spec.domain_id is None:
                raise ValueError(
                    f"The Cosmos3 Sim checkpoint declares no domain id for embodiment "
                    f"{name!r}; it lists {sorted(self.embodiments)}."
                )
            ids.append(spec.domain_id)
        return tuple(ids)

    def chunk_ranges(self, num_latent_frames: int) -> list[tuple[int, int]]:
        """Latent frame ranges ``[f0, f1)`` of the rollout's chunks, in order: the
        partition's chunks first, then ``chunk_frames`` at a time, the last possibly
        shorter."""
        ranges: list[tuple[int, int]] = []
        f0 = 0
        sizes: Sequence[int] = self.chunk_partition
        i = 0
        while f0 < num_latent_frames:
            size = sizes[i] if i < len(sizes) else self.chunk_frames
            f1 = min(f0 + size, num_latent_frames)
            ranges.append((f0, f1))
            f0, i = f1, i + 1
        return ranges


class _ChunkDecoder:
    """Pixel frames of a rollout, decoded chunk by chunk as the chunks finish. Each
    chunk's uint8 frames are copied straight into one pinned host buffer, so GPU
    memory does not grow with the video's length; ``finish`` returns that buffer,
    ``[1, T, H, W, 3]``, the video the pipeline would otherwise decode after the
    rollout. The VAE takes the rollout's latents as they are: the checkpoint's latent
    statistics are folded into it.

    On the generator's GPU the decode runs on the main stream, between one chunk's
    last denoising step and the next chunk's first: a GPU at its power cap gains
    nothing from overlapping the two. With ``stream`` on another GPU holding its own
    copy of the VAE, each chunk's latents (a few hundred KB) hop over as they finish
    and the decode leaves the generator's GPU alone."""

    def __init__(self, vae, stream: Optional[torch.cuda.Stream], num_pixel_frames: int) -> None:
        self.vae = vae
        self.stream = stream
        self.num_pixel_frames = num_pixel_frames
        self.host: Optional[torch.Tensor] = None
        self.written = 0
        self.held: list[torch.Tensor] = []
        with self._on_decode_stream():
            vae.decode_stream_start()
        self.open = True

    @contextlib.contextmanager
    def _on_decode_stream(self):
        if self.stream is None:
            yield
            return
        with torch.cuda.device(self.stream.device), torch.cuda.stream(self.stream):
            yield

    def submit(self, chunk: torch.Tensor) -> None:
        if self.stream is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream())
        with self._on_decode_stream():
            if self.stream is not None:
                ready.wait()
                if chunk.device != self.stream.device:
                    chunk = chunk.to(self.stream.device, non_blocking=True)
                else:
                    # The decoding stream reads the caller's memory: keep it out of
                    # the main stream's pool until close().
                    self.held.append(chunk)
            frames = postprocess_video_tensor(self.vae.decode_stream_step(chunk))
            if self.host is None:
                _, _, height, width, channels = frames.shape
                self.host = torch.empty(
                    (1, self.num_pixel_frames, height, width, channels),
                    dtype=frames.dtype,
                    pin_memory=prefer_pinned(),
                )
            t = frames.shape[1]
            # The copy is queued on the decoding stream; the frames' memory goes back
            # to that stream's pool and is reused only by work ordered after the copy.
            self.host[:, self.written : self.written + t].copy_(frames, non_blocking=True)
            self.written += t

    def close(self) -> None:
        """End the VAE's clip and wait for the decoding stream. No effect once closed."""
        if not self.open:
            return
        self.open = False
        with self._on_decode_stream():
            self.vae.decode_stream_end()
            torch.cuda.current_stream().synchronize()
        self.held.clear()

    def finish(self) -> torch.Tensor:
        self.close()  # the CPU reads the host buffer next
        if self.written != self.num_pixel_frames:
            raise RuntimeError(
                f"Cosmos3 Sim chunk decode wrote {self.written} of "
                f"{self.num_pixel_frames} pixel frames."
            )
        video, self.host = self.host, None
        return video


@dataclass
class _SimRollout:
    """One open rollout: the request, the opened K/V cache and what every chunk needs."""

    request: Any
    prepared: Any
    seed: int
    kv_cache: BlockCausalKVCacheManager
    text_len: int
    history_frames: int
    latent_shape: tuple[int, int, int]
    given_frames: tuple[int, ...]
    """Latent frames the request supplies: they keep their latents and are not denoised."""
    steps: FixedStochasticSchedule
    offload: Any


@register_pipeline(
    "Cosmos3NanoSimBimanualPipeline",
    hf_ids=["nvidia/Cosmos3-Nano-Sim-Bimanual"],
    defaults=PIPELINE_CONFIG_DEFAULTS,
    doc="Cosmos3 Nano Sim: chunked causal world model with a rolling K/V cache.",
)
class Cosmos3SimPipeline(Cosmos3OmniMoTPipeline):
    """``Cosmos3OmniMoTPipeline`` with the chunked causal rollout in place of the
    whole-clip denoising loop. The generic request resolution (with the embodiment
    taken from the checkpoint), the action rows and guardrails are inherited. Its own:
    the VAE (WinoVae with the latent statistics folded in), the first frame encoded
    alone, prompts tokenized without padding, warmup with served requests and the K/V
    cache sized from their measured peak."""

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self.sim: Optional[Cosmos3SimSettings] = None  # set by load_standard_components
        # The page pool's size; what it holds in frames depends on each request's frame
        # geometry. Until it is settled (by warmup from the memory left after the
        # measured peak, or by the first request when warmup is skipped) it is the size
        # of the pool warmup measures over.
        self._pool_tokens: Optional[int] = None
        self._pool_settled = False
        # Tokens per frame of the rollouts warmup measured, and how often the pool it
        # measures over was built.
        self._measured_tokens_per_frame: set[int] = set()
        self._measuring_pool_builds = 0
        # (canvas height, canvas width, frame height, frame width) in pixels of the
        # rollouts whose graphs and kernels exist.
        self._warmed_frame_sizes: set[tuple[int, int, int, int]] = set()
        self._kv_cache: Optional[BlockCausalKVCacheManager] = None
        # Finished chunks are decoded as they complete; on a second GPU, when the
        # serving config names one, with its own VAE instance and stream.
        self._decode_stream: Optional[torch.cuda.Stream] = None
        self._decoded_video: Optional[torch.Tensor] = None
        # The checkpoint's denoising schedule on the device and the request schedule
        # it was built for.
        self._steps: Optional[FixedStochasticSchedule] = None
        self._steps_schedule: Any = None
        self._settings = {**PIPELINE_CONFIG_DEFAULTS, **pipeline_config.extra_attrs}
        fraction = self._settings["kv_cache_free_gpu_memory_fraction"]
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0.0 < fraction <= 1.0
        ):
            raise ValueError(
                "pipeline_config.kv_cache_free_gpu_memory_fraction must be in (0, 1], got "
                f"{fraction!r}."
            )
        self._kv_fraction = float(fraction)
        decode_device = self._settings["vae_decode_device"]
        self._decode_device = None if decode_device is None else torch.device(decode_device)
        if self._decode_device is not None and self._decode_device.type != "cuda":
            raise ValueError(
                f"pipeline_config.vae_decode_device must be a CUDA device, got {decode_device!r}."
            )
        self._decode_vae = None

    @property
    def extra_param_specs(self):
        specs = super().extra_param_specs
        specs["domain_id"] = specs["domain_id"].model_copy(
            update={
                "type": "int | list[int]",
                "description": (
                    "Embodiment domain id for action generation: one id, or one id per "
                    "action row for a trajectory that switches embodiment."
                ),
                "validator": _validate_domain_ids,
            }
        )
        specs["domain_name"] = specs["domain_name"].model_copy(
            update={
                "type": "str | list[str]",
                "description": (
                    "Embodiment domain name for action generation: one of the checkpoint's "
                    "embodiments, or one name per action row for a trajectory that switches "
                    "embodiment."
                ),
                "validator": _validate_domain_names,
            }
        )
        return specs

    def _load_vae(self, checkpoint_dir: str, device: torch.device) -> torch.nn.Module:
        # Every process must encode a first frame to the same latent: the rollout turns a
        # difference in its low bits into another video. So cuDNN keeps its heuristics
        # instead of timing the encoder's convs.
        options = {"conv_autotuning": False}
        # Only the VAE that decodes chunks on the rollout's GPU takes the shallow depth:
        # a decode GPU's buffers are not the cache's memory, and a whole-clip decode
        # would pay the extra history copies twice per frame.
        streams_here = (
            self._decode_device is None
            and not self.offloader.requested_components()
            and self.pipeline_config.parallel.parallel_vae_size <= 1
        )
        if streams_here:
            options["frame_buffer_chunks"] = ROLLOUT_GPU_FRAME_BUFFER_CHUNKS
        vae = load_wino_vae(checkpoint_dir, device, dtype=torch.bfloat16, options=options)
        # The folded weights stay on the device they were built on. A VAE that is an
        # offload stage is loaded on the CPU and moved for every decode, so it keeps the
        # plain conv and _scale_latents_for_decode undoes the statistics for it.
        if torch.device(device).type == "cuda":
            vae.fold_latent_affine(
                torch.tensor(vae.config.latents_mean), torch.tensor(vae.config.latents_std)
            )
        return vae

    def load_standard_components(self, checkpoint_dir, device, skip_components=None):
        super().load_standard_components(checkpoint_dir, device, skip_components)
        self.sim = Cosmos3SimSettings.from_config(
            self.resolver.config(),
            inference_size=self._settings["kv_cache_inference_size"],
            sink_frames=self._settings["attention_sink_size"],
        )
        if self.sim.history_frames is None:
            logger.info("Cosmos3 Sim: full history, the K/V cache is sized from free memory.")
        else:
            logger.info(
                f"Cosmos3 Sim: each chunk attends to {self.sim.sink_frames} sink frames and "
                f"the {self.sim.history_frames} latent frames before it."
            )
        self._check_decode_device()
        if self._decode_device is not None and self.rank == 0 and self.vae is not None:
            if _gpu_index(self._decode_device) == _gpu_index(torch.device(device)):
                raise ValueError(
                    f"pipeline_config.vae_decode_device is {self._decode_device}, the GPU the "
                    "rollout runs on; name another GPU or leave the key out."
                )
            logger.info(f"Cosmos3 Sim: decoding on {self._decode_device}")
            self._decode_vae = self._load_vae(checkpoint_dir, self._decode_device)

    def _check_decode_device(self) -> None:
        """A decode GPU takes the chunks as they finish. Component offloading and
        parallel VAE decode leave the decode until after the rollout, so the serving
        config may not ask for both. Every rank reads the same config and refuses."""
        if self._decode_device is None:
            return
        if (
            self.offloader.requested_components()
            or self.pipeline_config.parallel.parallel_vae_size > 1
        ):
            raise ValueError(
                "pipeline_config.vae_decode_device needs the chunks decoded during the "
                "rollout, which component offloading and parallel VAE decode turn off."
            )

    def _resolve_request(
        self,
        *,
        action_mode: Optional[str],
        domain_name: str | list[str] | None,
        domain_id: int | list[int] | None,
        raw_action_dim: Optional[int],
        **kwargs: Any,
    ):
        """The checkpoint, not the generic embodiment table, says which domain id and
        action width an embodiment has on this model."""
        mode = normalize_action_mode(action_mode)
        if mode not in (None, ACTION_MODE_FORWARD_DYNAMICS):
            raise ValueError(
                f"Cosmos3 Sim runs action_mode={ACTION_MODE_FORWARD_DYNAMICS!r} only, got "
                f"{action_mode!r}."
            )
        if kwargs.get("transfer_config") is not None or kwargs.get("video") is not None:
            raise ValueError("Cosmos3 Sim takes no transfer controls and no reference video.")
        if mode is None and kwargs.get("image") is not None:
            raise ValueError(
                "Cosmos3 Sim takes a first frame together with an action trajectory: set "
                f"action_mode={ACTION_MODE_FORWARD_DYNAMICS!r}."
            )
        domain_id, row_ids = split_row_domain_ids(domain_id)
        if isinstance(domain_name, (list, tuple)):
            if row_ids is not None:
                raise ValueError(
                    "Cosmos3 Sim takes the embodiment of each action row as domain_name or "
                    "as domain_id, not both."
                )
            row_ids = self.sim.row_domain_ids(domain_name)
            domain_name, domain_id = domain_name[0], row_ids[0]
        # Every request has an embodiment: without an action stream its null action
        # rows are still projected with that embodiment's weights.
        domain_name, domain_id, action_dim = self.sim.resolve_embodiment(
            domain_name, domain_id, raw_action_dim
        )
        if mode is not None:
            raw_action_dim = action_dim
            if row_ids is not None:
                for d in set(row_ids):
                    self.sim.resolve_embodiment(None, d, raw_action_dim)
        resolved = super()._resolve_request(
            action_mode=action_mode,
            domain_name=domain_name,
            domain_id=domain_id,
            raw_action_dim=raw_action_dim,
            **kwargs,
        )
        if row_ids is not None:
            if (
                resolved.action_chunk_size is not None
                and len(row_ids) != resolved.action_chunk_size
            ):
                raise ValueError(
                    f"Cosmos3 Sim got {len(row_ids)} domain ids for "
                    f"{resolved.action_chunk_size} action rows."
                )
            resolved = replace(resolved, action_row_domain_ids=row_ids)
        if mode is not None and self._latent_frames(resolved.num_frames) < 2:
            rows = self.sim.action_tokens_per_frame
            raise ValueError(
                f"Cosmos3 Sim generates one latent frame per {rows} action rows; "
                f"action_chunk_size={resolved.action_chunk_size} generates none."
            )
        content = (resolved.action_content_height, resolved.action_content_width)
        height, width = (resolved.height, resolved.width) if None in content else content
        self.check_envelope(height, width, resolved.num_frames)
        return resolved

    @nvtx_range("_tokenize_request_prompts", color="blue")
    def _tokenize_request_prompts(
        self,
        request,
        prompt: list[str],
        negative_prompt: Optional[str],
        *,
        use_duration_template: bool,
        use_resolution_template: bool,
    ):
        """``(prompt_ids, None, None, None)``: the prompt's ``L`` tokens ``[1, L]`` on the
        device, without padding, so the tensor's length is the prompt's length. A Sim
        checkpoint carries its guidance in the weights and the rollout runs one prompt
        per request: no negative prompt is tokenized and no mask is built."""
        del negative_prompt
        prompt, _, system_prompt = self._request_prompts(
            request,
            prompt,
            None,
            use_duration_template=use_duration_template,
            use_resolution_template=use_resolution_template,
        )
        token_ids = self._prompt_token_ids(
            prompt[0], request.max_sequence_length, request.use_system_prompt, system_prompt
        )
        return torch.tensor([token_ids], dtype=torch.long, device=self.device), None, None, None

    def _preprocess_action_first_frame(self, image, video, target_h, target_w):
        """The first frame as a clip of one frame, ``[1, 3, 1, H, W]``, which the
        request preparation takes as the clip to encode: the rollout conditions
        latent frame 0 only, and the VAE encodes that frame from pixel frame 0 alone."""
        frame = super()._preprocess_action_first_frame(image, video, target_h, target_w)
        return frame.unsqueeze(2)

    def _prepare_latents_action_video(
        self, video_tensor, mode, num_frames, generator, *, content_h, content_w
    ):
        """``(latents, velocity_mask, condition)`` of a rollout from its one-frame clip.
        ``latents`` ``[1, C, T, H, W]`` is an uninitialized buffer: the rollout writes
        its frames there when the video is decoded after it. ``velocity_mask``
        ``[1, 1, 1, 1, 1]`` (zero) and ``condition`` ``[1, C, 1, H, W]`` cover latent
        frame 0, the only one the request gives."""
        del mode, generator
        condition = crop_action_latent(
            self._encode_video_tensor(video_tensor),
            content_h,
            content_w,
            self.vae_scale_factor_spatial,
        )
        if condition.shape[2] != 1:
            raise ValueError(
                f"Cosmos3 Sim first-frame latent has {condition.shape[2]} frames, expected 1."
            )
        latents = torch.empty(
            (1, condition.shape[1], self._latent_frames(num_frames), *condition.shape[-2:]),
            device=self.device,
            dtype=self.dtype,
        )
        velocity_mask = torch.zeros((1, 1, 1, 1, 1), device=self.device, dtype=self.dtype)
        return latents, velocity_mask, condition

    def _prepare_request_latents(self, request, **kwargs):
        prepared = super()._prepare_request_latents(request, **kwargs)
        prepared.action_row_domain_ids = request.action_row_domain_ids
        if prepared.action_domain_id is None:
            # No action stream: the null action rows take the request's embodiment.
            prepared.action_domain_id = resolve_domain_id(
                domain_id=request.domain_id,
                domain_name=request.domain_name,
                require_explicit=True,
            )
        return prepared

    def check_envelope(self, height: int, width: int, num_frames: int) -> None:
        """Refuse, before any compute, a rollout the K/V cache cannot hold. ``height`` and
        ``width`` are the size of the frames the rollout generates: the picture's
        content, not the canvas that pads it. Refused: a configured window that does
        not fit the pool, or a full-history rollout longer than the pool affords.
        Nothing to check before the pool is sized."""
        if not self._pool_settled:
            return
        history = self._history_frames_for(self._tokens_per_frame(height, width))
        self._check_history_fits(self._latent_frames(num_frames), history)

    def _latent_frames(self, num_pixel_frames: int) -> int:
        return (num_pixel_frames - 1) // self.vae_scale_factor_temporal + 1

    def _check_history_fits(self, num_latent_frames: int, history_frames: int) -> None:
        """Full history (no window declared): the last chunk reads every frame before
        it, so the pool must hold that many. Longer rollouts are refused rather than
        run over a window nobody asked for."""
        if self.sim.history_frames is not None:
            return
        needed = self.sim.chunk_ranges(num_latent_frames)[-1][0]
        if needed <= history_frames:
            return
        requested = (num_latent_frames - 1) * self.vae_scale_factor_temporal + 1
        raise ValueError(
            f"Cosmos3 Sim runs full history: the last chunk of {requested} pixel frames reads "
            f"{needed} latent frames, and the K/V cache holds {history_frames} at this frame "
            "size. Raise pipeline_config.kv_cache_free_gpu_memory_fraction or shorten the "
            "request."
        )

    # ------------------------------------------------------------------ warmup and sizing

    @property
    def default_warmup_resolutions(self):
        return [self.sim.largest_frame()]

    @property
    def default_warmup_num_frames(self):
        """Pixel lengths whose rollouts run, between them, every chunk forward and every
        decode step a rollout of any length runs: the partition's chunks followed by a
        one-frame last chunk, then the first chunk followed by a last chunk of each
        size between one frame and a full chunk."""
        sim = self.sim
        latent = [sum(sim.chunk_partition) + 1]
        latent += [sim.chunk_partition[0] + tail for tail in range(2, sim.chunk_frames)]
        return [(n - 1) * self.vae_scale_factor_temporal + 1 for n in latent]

    def validate_resolution(self, height: int, width: int, num_frames: int) -> None:
        super().validate_resolution(height, width, num_frames)
        if self._latent_frames(num_frames) < 2:
            raise ValueError(
                "a Cosmos3 Sim rollout takes a first frame and generates at least one latent frame"
            )

    def resolve_warmup_plan(self):
        """``(shapes, steps)``: every first-frame size to warm, as
        ``compilation_config.resolutions`` lists them or ``default_warmup_resolutions``,
        with every length of ``default_warmup_num_frames`` and the lengths
        ``compilation_config.num_frames`` adds."""
        compilation = self.pipeline_config.compilation
        resolutions = compilation.resolutions or self.default_warmup_resolutions
        lengths = dict.fromkeys([*self.default_warmup_num_frames, *(compilation.num_frames or ())])
        shapes = []
        for (height, width), num_frames in itertools.product(resolutions, lengths):
            try:
                self.validate_resolution(height, width, num_frames)
            except ValueError as e:
                logger.warning(
                    f"Skipping invalid warmup shape ({height}x{width}, {num_frames} frames): {e}"
                )
                continue
            shapes.append((height, width, num_frames))
        return shapes, self.default_warmup_steps

    def request_warmup_cache_key(self, req) -> tuple:
        """Unresolved: a rollout's frame size follows from the request's first frame,
        which ``forward`` resolves. ``_denoise_request`` reports a size warmup did not
        run."""
        return (None,)

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        """One request as it is served: a first frame of ``height`` x ``width``,
        ``num_frames - 1`` null action rows of the checkpoint's default embodiment,
        forward dynamics, and a prompt that fills the text cache. The request path
        resolves the canvas, the length and the schedule."""
        sim = self.sim
        embodiment = sim.default_embodiment
        row_width = (
            sim.embodiments[embodiment].action_dim
            or resolve_raw_action_dim(domain_name=embodiment)
            or int(self.transformer.action_dim)
        )
        rows = num_frames - 1
        with torch.no_grad():
            self.forward(
                prompt=" ".join(["warmup"] * sim.text_cache_max_len),
                image=PIL.Image.new("RGB", (width, height)),
                max_sequence_length=sim.text_cache_max_len,
                use_guardrails=False,
                action_mode=ACTION_MODE_FORWARD_DYNAMICS,
                domain_name=embodiment,
                action=torch.zeros(rows, row_width),
                action_chunk_size=rows,
            )

    def _run_warmup_pass(self, shapes, steps) -> None:
        """One pass over the warmup shapes on the pool that serves. Until that pool
        exists the shapes first run twice over a pool just large enough for them: once
        to compile kernels and capture graphs, once measured, so the measured peak is
        that of requests with nothing left to compile. The serving pool then takes its
        share of what that peak leaves."""
        # With CUDA graphs on, a pass with capture off is the autotuner's: its forwards
        # run eagerly on tactics that are not final, and the pass after it measures.
        # Without graphs there is one pass, which measures.
        if not self._pool_settled and all(
            runner.allow_capture for runner in self._cuda_graph_runners.values()
        ):
            self._size_pool(shapes, steps)
        super()._run_warmup_pass(shapes, steps)

    def _size_pool(self, shapes, steps) -> None:
        fraction, device = self._kv_fraction, self.device
        try:
            builds = self._measuring_pool_builds
            super()._run_warmup_pass(shapes, steps)
            if self._measuring_pool_builds - builds > 1:
                # A later shape outgrew the pool and dropped the graphs captured before
                # it: capture them again, so the measured pass replays every shape.
                super()._run_warmup_pass(shapes, steps)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            super()._run_warmup_pass(shapes, steps)
            if self.world_size > 1:
                dist.barrier()  # every rank's warm requests are done, decode included
            torch.cuda.synchronize(device)
            stats = torch.cuda.memory_stats(device)
            free, total = torch.cuda.mem_get_info(device)
            measuring_pool = self._kv_cache.pool_bytes if self._kv_cache is not None else 0
            if self.world_size > 1:
                dist.barrier()  # every rank has read its GPU before any rank frees memory
        except BaseException:
            self._pool_tokens = None
            raise
        finally:
            self._shutdown_kv_cache()
            torch.cuda.empty_cache()
        # What torch took from the driver at its peak, plus what is on the device and is
        # neither torch's nor the measuring pool: every byte counted once.
        torch_peak = stats["reserved_bytes.all.peak"]
        outside_torch = max(
            0, (total - free) - stats["reserved_bytes.all.current"] - measuring_pool
        )
        peak = torch_peak + outside_torch
        available = int((total - peak) * fraction)
        self._settle_pool_tokens(self._pool_tokens_for_bytes(available))
        gib = 1 << 30
        pool_gib = self._pool_bytes_for_tokens(self._pool_tokens) / gib
        for tokens_per_frame in sorted(self._measured_tokens_per_frame):
            history = self._history_frames_for(tokens_per_frame)
            if self.sim.history_frames is None and history < self.sim.chunk_frames:
                raise RuntimeError(
                    f"Cosmos3 Sim cannot hold one chunk of history: warmup peak "
                    f"{peak / gib:.2f} GiB of {total / gib:.2f} GiB leaves "
                    f"{available / gib:.2f} GiB for the K/V cache at "
                    f"kv_cache_free_gpu_memory_fraction={fraction}; {history} frames of "
                    f"{tokens_per_frame} tokens fit, {self.sim.chunk_frames} are needed."
                )
            logger.info(
                f"Cosmos3 Sim K/V cache: {self._pool_tokens // KV_TOKENS_PER_PAGE} pages of "
                f"{KV_TOKENS_PER_PAGE} tokens ({pool_gib:.2f} GiB): {history} latent frames of "
                f"history at {tokens_per_frame} tokens per frame. Warmup peak "
                f"{peak / gib:.2f} GiB of {total / gib:.2f} GiB (torch reserved "
                f"{torch_peak / gib:.2f}, allocated "
                f"{stats['allocated_bytes.all.peak'] / gib:.2f}, other "
                f"{outside_torch / gib:.2f}), fraction {fraction}."
            )

    def _settle_pool_tokens(self, pool_tokens: int) -> None:
        """Every rank serves from the smallest pool any rank measured, so all ranks
        accept and refuse the same requests."""
        if self.world_size > 1:
            smallest = torch.tensor([pool_tokens], dtype=torch.int64, device=self.device)
            dist.all_reduce(smallest, op=dist.ReduceOp.MIN)
            pool_tokens = int(smallest.item())
        self._pool_tokens = pool_tokens
        self._pool_settled = True

    def _measuring_history(self, num_latent_frames: int) -> int:
        """History the measuring pool holds for a warmup rollout: the configured
        window, or with full history the whole rollout, and never below two chunks.
        The pool's own bytes are left out of the measured peak."""
        window = self.sim.history_frames
        return max(2 * self.sim.chunk_frames, num_latent_frames - 1 if window is None else window)

    def _fit_measuring_pool(self, tokens_per_frame: int, num_latent_frames: int) -> None:
        """Make the pool warmup measures over hold this rollout. A pool that is too
        small is rebuilt, which drops the graphs captured over it; the measured pass
        comes after every shape has run once, so it rebuilds nothing."""
        needed = self.pool_tokens_for_history(
            self._measuring_history(num_latent_frames), tokens_per_frame
        )
        self._measured_tokens_per_frame.add(tokens_per_frame)
        if self._pool_tokens is None or needed > self._pool_tokens:
            self._shutdown_kv_cache()
            self._pool_tokens = needed
            self._measuring_pool_builds += 1

    def _kv_geometry(self) -> dict:
        tf = self.transformer
        return dict(
            num_layers=len(tf.gen_layers),
            num_kv_heads=tf.cache_kv_heads,
            head_dim=tf.gen_layers[0].cross_attention.head_dim,
            dtype=self.dtype,
        )

    def _pool_bytes_for_tokens(self, pool_tokens: int) -> int:
        return BlockCausalKVCacheManager.bytes_for_pool_tokens(pool_tokens, **self._kv_geometry())

    def _pool_tokens_for_bytes(self, budget_bytes: int) -> int:
        """Pool tokens (whole pages) whose K/V fit ``budget_bytes``. The pool allocator
        maps whole physical chunks of at most 32 MiB, so the mapped size may exceed the
        budget by less than one chunk; a budget that is a fraction below 1 of the free
        memory leaves far more than that."""
        g = self._kv_geometry()
        bytes_per_token = (
            g["num_layers"]
            * 2
            * g["num_kv_heads"]
            * g["head_dim"]
            * torch.tensor([], dtype=g["dtype"]).element_size()
        )
        return budget_bytes // (KV_TOKENS_PER_PAGE * bytes_per_token) * KV_TOKENS_PER_PAGE

    def pool_tokens_for_history(self, history_frames: int, tokens_per_frame: int) -> int:
        """Pool tokens a rollout with this much history needs at this frame geometry."""
        return BlockCausalKVCacheManager.pool_tokens_for(
            self._rollout_spec(history_frames, tokens_per_frame),
            tokens_per_page=KV_TOKENS_PER_PAGE,
        )

    def _history_frames_for(self, tokens_per_frame: int) -> int:
        """Frames of history a rollout at this frame geometry gets from the pool: the
        configured window, which must fit, or for full history the most that fit."""
        window = self.sim.history_frames
        if window is not None:
            needed = self.pool_tokens_for_history(window, tokens_per_frame)
            if needed > self._pool_tokens:
                raise ValueError(
                    f"Cosmos3 Sim's window of {window} latent frames needs {needed} pool tokens "
                    f"at {tokens_per_frame} tokens per frame; the K/V cache holds "
                    f"{self._pool_tokens}. Raise "
                    "pipeline_config.kv_cache_free_gpu_memory_fraction or use a smaller frame."
                )
            return window
        # The pool holds (ceil((pins + window + staged) / page) + 1 + private) pages, where
        # the private pages of the causal blocks do not depend on the window: every page
        # left after them, the spare and the fixed tokens is window.
        tpp = KV_TOKENS_PER_PAGE
        spec = self._rollout_spec(1, tokens_per_frame)
        fixed = spec.max_pin_tokens + spec.max_staged_tokens
        private_tokens = (
            self.pool_tokens_for_history(1, tokens_per_frame)
            - (ceil_div(fixed + tokens_per_frame, tpp) + 1) * tpp
        )
        window_pages = (self._pool_tokens - private_tokens) // tpp - 1
        return max(0, (window_pages * tpp - fixed) // tokens_per_frame)

    def _rollout_spec(self, history_frames: int, tokens_per_frame: int) -> RolloutSpec:
        """The rollout's cache geometry: the window, one chunk staged per forward, and the
        causal blocks a forward cuts it into. The clean pass is one causal block per
        frame, up to ``chunk_frames`` of them. A denoising step is one causal block of
        its whole chunk, ``k`` frames for a chunk of ``k <= chunk_frames`` frames (the
        partition's first chunks and the last one may be shorter), so every size above
        one frame is declared once."""
        sim = self.sim
        frames = sim.chunk_frames
        causal_blocks = {k * tokens_per_frame: 1 for k in range(2, frames + 1)}
        causal_blocks[tokens_per_frame] = frames
        return RolloutSpec(
            window_tokens=max(history_frames, 1) * tokens_per_frame,
            max_staged_tokens=frames * tokens_per_frame,
            causal_blocks=causal_blocks,
            max_pin_tokens=sim.text_cache_max_len + sim.sink_frames * tokens_per_frame,
        )

    def _tokens_per_frame(self, height: int, width: int) -> int:
        s = self.vae_scale_factor_spatial
        Hp, Wp, _, _ = self.transformer._pad_to_patch_size(max(height // s, 1), max(width // s, 1))
        return self.sim.action_tokens_per_frame + Hp * Wp

    def _shutdown_kv_cache(self) -> None:
        if self._kv_cache is not None:
            # Graphs captured over this cache read its pages and tables by address.
            for runner in self._cuda_graph_runners.values():
                runner.clear()
            self._kv_cache.shutdown()
            self._kv_cache = None

    def cleanup(self):
        super().cleanup()
        self._shutdown_kv_cache()
        self._steps = None

    # ------------------------------------------------------------------ CUDA graphs

    def _setup_cuda_graphs(self) -> None:
        """Run the chunk forward under CUDA graphs. The bidirectional forward is not
        wrapped: the rollout never calls it. Keys: the tensor shapes, the packing,
        the pass and the forward plan's ``graph_key`` (cache, rollout spec, token
        count, causal block size). Commits rewrite the cache's device tensors in
        place, so one graph replays for every chunk of the same shape."""
        if not self.pipeline_config.cuda_graph.enable:
            return
        if self.offloader.stages():
            raise NotImplementedError(
                "CUDA graphs are not supported with visual generation offloading yet. "
                "Disable either cuda_graph_config.enable or cpu_offload_config.enable."
            )
        runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
        runner.register_extra_key_fn("clean_pass", lambda *a, **k: bool(k.get("clean_pass")))
        runner.register_extra_key_fn("packing", lambda *a, **k: k.get("packing"))
        runner.register_extra_key_fn("kv", lambda *a, kv=None, **k: kv.graph_key if kv else None)
        tf = self.transformer
        tf._forward_causal_core = runner.wrap(tf._forward_causal_core)
        self._cuda_graph_runners["transformer.forward_causal"] = runner
        logger.info("CUDA graph runner: wrapping transformer.forward_causal")

    # ------------------------------------------------------------------ the rollout

    def _kv_cache_manager(self) -> BlockCausalKVCacheManager:
        """The one resident page pool, built once and kept across requests of every
        frame geometry: ``close()``/``open()`` keep its device tensors, so graphs
        captured over it stay valid, and a request's spec takes the pages it needs."""
        if self._kv_cache is not None:
            return self._kv_cache
        if not self._pool_settled and not self._is_warmup:
            # Warmup was skipped, so nothing measured the peak: size the pool from what
            # is free right now, as the LLM executor does without estimation.
            free, _ = torch.cuda.mem_get_info(self.device)
            fraction = self._kv_fraction
            self._settle_pool_tokens(self._pool_tokens_for_bytes(int(free * fraction)))
            logger.warning(
                f"Cosmos3 Sim K/V cache sized without warmup: {self._pool_tokens} pool tokens "
                f"from {free / (1 << 30):.2f} GiB free at fraction {fraction}."
            )
        tf = self.transformer
        attention = tf.gen_layers[0].cross_attention.attn
        if not attention.support_kv_cache_page_size(KV_TOKENS_PER_PAGE, self.dtype):
            raise RuntimeError(
                f"the {tf.gen_layers[0].cross_attention.attn_backend} attention backend has no "
                f"kernel for {KV_TOKENS_PER_PAGE}-token K/V pages with head_dim "
                f"{attention.head_dim} in {self.dtype} on this GPU; use the CUDNN backend."
            )
        self._kv_cache = BlockCausalKVCacheManager(
            tokens_per_page=KV_TOKENS_PER_PAGE,
            pool_tokens=self._pool_tokens,
            **self._kv_geometry(),
        )
        return self._kv_cache

    def _chunk_generator(self, seed: int, first_frame: int) -> torch.Generator:
        """The random stream of one chunk, seeded by the request seed and the chunk's
        first latent frame: a chunk's noise does not depend on the chunks before it."""
        return make_noise_generator(int(seed) + first_frame, self.device)

    def _step_schedule(self, request) -> FixedStochasticSchedule:
        """The checkpoint's denoising schedule on the device, built for the request's
        resolved schedule and kept: requests of a distilled checkpoint resolve to the
        same one."""
        if self._steps is None or request.schedule != self._steps_schedule:
            self._steps = self.sampling.fixed_stochastic_schedule(
                self.scheduler, self.device, request.schedule
            )
            self._steps_schedule = request.schedule
        return self._steps

    def _rollout_begin(self, request, prepared, *, cond_ids, seed) -> "_SimRollout":
        """Open the rollout: the prompt's K/V once, then the cache is ready for chunks.
        ``cond_ids`` ``[1, L]`` is the prompt without padding."""
        if request.enable_audio:
            raise ValueError("Cosmos3 Sim does not generate audio.")
        sim, tf = self.sim, self.transformer
        _, C, _, H, W = prepared.video_latents.shape
        Hp, Wp, _, _ = tf._pad_to_patch_size(H, W)
        A = sim.action_tokens_per_frame
        tokens_per_frame = A + Hp * Wp
        if self._is_warmup and not self._pool_settled:
            self._fit_measuring_pool(tokens_per_frame, prepared.video_latents.shape[2])
        kv_cache = self._kv_cache_manager()
        history_frames = self._history_frames_for(tokens_per_frame)
        text_len = cond_ids.shape[1]
        if text_len > sim.text_cache_max_len:
            raise ValueError(
                f"prompt of {text_len} tokens exceeds the checkpoint's text cache of "
                f"{sim.text_cache_max_len} tokens"
            )
        given_frames: tuple[int, ...] = ()
        if prepared.velocity_mask is not None and prepared.condition_latents is not None:
            # The frames a request with a first frame gives, by its action mode.
            given_frames = tuple(
                vision_condition_indexes(
                    request.action_mode, request.num_frames, self.vae_scale_factor_temporal
                )
            )
        steps = self._step_schedule(request)
        offload = self.offloader.context_if_requested
        kv_cache.open(
            self._rollout_spec(history_frames, tokens_per_frame),
            pin_tokens=text_len + sim.sink_frames * tokens_per_frame,
        )
        try:
            # The text tower runs over the whole text cache for every prompt: one
            # shape, so nothing about it is set up inside a request.
            tf.write_prompt_kv(
                kv_cache, cond_ids, text_len, sim.text_cache_max_len, offload_context=offload
            )
            kv_cache.commit(text_len)
        except Exception:
            kv_cache.close()
            raise
        return _SimRollout(
            request=request,
            prepared=prepared,
            seed=int(seed),
            kv_cache=kv_cache,
            text_len=text_len,
            history_frames=history_frames,
            latent_shape=(C, H, W),
            given_frames=given_frames,
            steps=steps,
            offload=offload,
        )

    def _rollout_chunk(
        self,
        ro: "_SimRollout",
        f0: int,
        f1: int,
        action_hidden: torch.Tensor,
        *,
        commit: bool,
    ) -> torch.Tensor:
        """Denoise latent frames ``[f0, f1)`` against the committed history and, with
        ``commit``, run the clean pass that writes their K/V. ``action_hidden`` are the
        chunk's action tokens. A chunk of frames the request gives (the first frame)
        keeps their latents and is not denoised. Returns the chunk's clean latents
        ``[1, C, f1 - f0, H, W]``."""
        tf, kv_cache, request, steps = self.transformer, ro.kv_cache, ro.request, ro.steps
        k = f1 - f0
        chunk = dict(
            kv_cache=kv_cache,
            first_frame=f0,
            text_len=ro.text_len,
            action_hidden=action_hidden,
            fps=request.frame_rate,
            action_fps=request.action_fps,
            offload_context=ro.offload,
        )
        given = sum(f0 <= f < f1 for f in ro.given_frames)
        if given == k:
            x = ro.prepared.condition_latents[:, :, f0:f1]
        elif given:
            raise ValueError(
                f"Cosmos3 Sim: latent frames {f0}..{f1 - 1} hold given and generated frames; "
                "a given frame must fill its chunk."
            )
        else:
            C, H, W = ro.latent_shape
            generator = self._chunk_generator(ro.seed, f0)
            x = randn_tensor(
                (1, C, k, H, W), generator=generator, device=self.device, dtype=self.dtype
            )
            # Every chunk runs the whole schedule from its own noise.
            step_iter = (
                self._profile_denoise_steps(steps.timesteps)
                if self._profiler.enabled and not self._is_warmup
                else enumerate(steps.timesteps)
            )
            for i, timestep in step_iter:
                out = tf.forward_causal(x, timestep, **chunk)
                x = steps.step(i, out.video, x, generator)
        if commit:
            tf.forward_causal(x, None, clean_pass=True, **chunk)
            kv_cache.commit()
        return x

    def _rollout_end(self, ro: "_SimRollout") -> None:
        ro.kv_cache.close()

    def _denoise_request(
        self,
        request,
        prepared,
        *,
        cond_ids: torch.Tensor,
        cond_mask: None,
        uncond_ids: None,
        uncond_mask: None,
        generator: torch.Generator,
        timer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """The chunked causal rollout of one request: the prompt's K/V once, then per
        chunk the fixed denoising schedule over the committed history (skipped for a
        chunk of frames the request gives), the clean pass over the finished frames,
        and the commit of that pass's K/V. The last chunk gets no clean pass: nothing
        reads its K/V. ``cond_ids`` is the prompt without padding, as
        ``_tokenize_request_prompts`` returns it, and the other three prompt arguments
        are ``None``. The returned latents hold the rollout's frames only when the
        video is decoded after the rollout; chunks decoded as they finish go to the
        decoder and nowhere else."""
        del cond_mask, uncond_ids, uncond_mask
        sim = self.sim
        latents = prepared.video_latents
        if latents is None:
            raise RuntimeError("Cosmos3 Sim request reached the rollout without latents.")
        num_latent_frames = latents.shape[2]
        s = self.vae_scale_factor_spatial
        frame_size = (request.height, request.width, latents.shape[3] * s, latents.shape[4] * s)
        if (
            not self._is_warmup
            and self._warmed_frame_sizes
            and frame_size not in self._warmed_frame_sizes
            and self.rank == 0
        ):
            logger.warning(
                f"Cosmos3 Sim: {frame_size[3]}x{frame_size[2]} frames on a "
                f"{frame_size[1]}x{frame_size[0]} canvas were not warmed up. This request "
                "runs their first-use work (graph capture, kernel compilation); list the "
                "first-frame size under compilation_config.resolutions."
            )
        A = sim.action_tokens_per_frame
        self.transformer.reset_cache()
        action_hidden = self._sim_action_hidden(prepared, num_latent_frames, A)
        timer.mark_denoise_start()
        ro = self._rollout_begin(
            request, prepared, cond_ids=cond_ids, seed=generator.initial_seed()
        )
        decoder = None
        decode_error: Optional[Exception] = None
        try:
            self._check_history_fits(num_latent_frames, ro.history_frames)
            try:
                decoder = self._start_chunk_decode(latents)
            except Exception as error:
                decode_error = self._decode_failure(error)
            # Only a decode after the rollout reads the request's latents.
            decodes_afterwards = (
                decoder is None
                and decode_error is None
                and (self.rank == 0 or self._parallel_vae_enabled)
            )
            for f0, f1 in sim.chunk_ranges(num_latent_frames):
                x = self._rollout_chunk(
                    ro, f0, f1, action_hidden[:, f0 * A : f1 * A], commit=f1 < num_latent_frames
                )
                if decoder is not None:
                    if decode_error is None:
                        try:
                            decoder.submit(x)
                        except Exception as error:
                            decode_error = self._decode_failure(error)
                elif decodes_afterwards:
                    latents[:, :, f0:f1] = x
            if decode_error is not None:
                raise decode_error
            if decoder is not None:
                self._decoded_video = decoder.finish()
            self._warmed_frame_sizes.add(frame_size)
        finally:
            try:
                if decoder is not None:
                    decoder.close()
            finally:
                self._rollout_end(ro)
        return latents, None, prepared.action_latents, False

    # ------------------------------------------------------------------ chunk VAE decode

    def _decode_failure(self, error: Exception) -> Exception:
        """A chunk decode of this rank failed. Alone, the rank raises ``error`` at once.
        With other ranks it returns ``error`` for the caller to raise after the
        rollout: they are inside the rollout's collectives, which complete only if
        this rank keeps running them."""
        if self.world_size == 1:
            raise error
        logger.error(
            f"Cosmos3 Sim: chunk decode failed; the request fails after its rollout: {error}"
        )
        return error

    def _start_chunk_decode(self, latents: torch.Tensor) -> Optional["_ChunkDecoder"]:
        """Decode finished chunks during the rollout when this rank decodes alone and
        the VAE is the native streaming decoder; otherwise the usual decode after the
        rollout applies."""
        self._decoded_video = None
        if not self._streams_decode():
            return None
        if self._decode_device is not None and self._decode_stream is None:
            self._decode_stream = torch.cuda.Stream(device=self._decode_device)
        vae = self._decode_vae or self.vae
        if not getattr(vae, "latent_affine_folded", False):
            raise RuntimeError(
                "Cosmos3 Sim decodes chunks with the latent statistics folded into the VAE; "
                "this VAE is not folded."
            )
        num_pixel_frames = (latents.shape[2] - 1) * self.vae_scale_factor_temporal + 1
        return _ChunkDecoder(vae, self._decode_stream, num_pixel_frames)

    def _streams_decode(self) -> bool:
        """Whether this rank decodes finished chunks itself with the streaming VAE."""
        return (
            not self._parallel_vae_enabled
            and self.rank == 0
            and hasattr(self.vae, "decode_stream_step")
            and not self.offloader.stages()
        )

    def _decode_latents(self, latents):
        video = self._decoded_video
        if video is None:
            return super()._decode_latents(latents)
        self._decoded_video = None
        return video

    def _scale_latents_for_decode(self, latents):
        """Latents as ``self.vae.decode`` takes them: unchanged when the VAE applies the
        checkpoint's latent statistics itself, else with the statistics undone."""
        if getattr(self.vae, "latent_affine_folded", False):
            return latents
        return super()._scale_latents_for_decode(latents)

    def _sim_action_hidden(self, prepared, num_latent_frames: int, A: int) -> torch.Tensor:
        """``[1, T * A, hidden_size]`` action tokens, frame-major: frame ``f`` gets the
        ``A`` action rows that lead into it, each projected with the weights of its own
        embodiment. Frame 0 has nothing before it and gets null (zero) rows in the
        first real row's embodiment; without an action stream every frame gets null
        rows in the request's embodiment."""
        tf = self.transformer
        total = num_latent_frames * A
        rows = torch.zeros(1, total, tf.action_dim, device=self.device, dtype=self.dtype)
        runs = [(total, prepared.action_domain_id)]
        given = prepared.action_latents
        if given is not None:
            needed = (num_latent_frames - 1) * A
            if given.shape[1] < needed:
                raise ValueError(
                    f"{given.shape[1]} action rows for {num_latent_frames} latent frames; "
                    f"frames 1.. need {needed} ({A} per latent frame)"
                )
            rows[:, A : A + needed] = given[:, :needed]
            row_ids = prepared.action_row_domain_ids
            if row_ids is not None and needed:
                runs = domain_runs(row_ids[:needed], lead_rows=A)
        return tf.embed_action_rows(rows, runs)


__all__ = [
    "PIPELINE_CONFIG_DEFAULTS",
    "Cosmos3SimPipeline",
    "Cosmos3SimSettings",
    "domain_runs",
    "split_row_domain_ids",
]
