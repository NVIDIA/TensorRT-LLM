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
"""The MiniMax-H3 pipeline of the ``minimax_h3_nvfp4__sm_107__u1_u4`` scenario.

The family pipeline with the scenario's transformer (:mod:`.modeling`) and the work outside the denoise
loop as the recipe runs it: the VAE ViT decoder's Linears as FP8 per-tensor GEMMs with the tiled and
fused quantize path, the decoder's spatial tiles, the decoder call replayed from CUDA graphs per chunk
shape, and the text-only text-encoder prefill replayed from a CUDA graph per prompt length. All of it
is selected by :mod:`.config`; the family's own path (``visual_gen_scenario=off``) is untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
from diffusers.utils.torch_utils import randn_tensor

from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

from ..packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MiniMaxH3PackedSequence,
    RowTimestepPlanner,
)
from ..pipeline_minimax_h3 import MiniMaxH3Pipeline
from .config import SCENARIO
from .kernels.vae_decoder_fp8 import convert_decoder_linears_fp8, set_tiles
from .modeling import ScenarioTransformer


class MiniMaxH3Nvfp4Sm107U1U4Pipeline(MiniMaxH3Pipeline):
    """MiniMax-H3 on SM107: pre-quantized NVFP4 Linears, FP8 trtllm-gen attention, Ulysses 1 or 4."""

    @classmethod
    def resolve_variant(cls, config):  # the resolver has already chosen this class
        return cls

    def _init_transformer(self) -> None:
        self.transformer = ScenarioTransformer(
            self.pipeline_config.model_configs[PipelineComponent.TRANSFORMER]
        )

    def post_load_weights(self) -> None:
        # The family sets the schedulers' shifts and prepares the decoder (``_prepare_vae_decoder``);
        # the scenario adds its decoder tiles, the FP8 decoder Linears and the decoder graphs.
        if self.vae is not None and self.device.type == "cuda":
            tile = SCENARIO.vae_tile
            if tile:
                # Before the decoder calibration, which measures one tile of this geometry.
                self.vae.enable_tiling(
                    tile_sample_min_height=tile[0],
                    tile_sample_min_width=tile[1],
                    tile_sample_min_overlap_height=tile[2],
                    tile_sample_min_overlap_width=tile[2],
                )
                logger.info(
                    f"MiniMax-H3 VAE decoder tiles {tile[0]}x{tile[1]} px, min overlap {tile[2]}"
                )
            self.vae.decoder_graph_max_shapes = int(SCENARIO.vae_graph_max_shapes)
        super().post_load_weights()
        if (
            SCENARIO.preloop_device_plan
            or SCENARIO.preloop_async_copies
            or SCENARIO.preloop_noise_early
        ):
            logger.info(
                f"MiniMax-H3 pre-loop: device plan {SCENARIO.preloop_device_plan}, "
                f"non-blocking copies {SCENARIO.preloop_async_copies}, "
                f"early noise {SCENARIO.preloop_noise_early}"
            )
        if self.vae is not None and self.device.type == "cuda" and SCENARIO.vae_fp8:
            ulysses = self._ulysses_size()
            if SCENARIO.vae_fp8_tiled:
                set_tiles(mlp=SCENARIO.vae_fp8_tile, q=SCENARIO.resolved_vae_fp8_qtile(ulysses))
            convert_decoder_linears_fp8(self.vae.decoder)

    def _ulysses_size(self) -> int:
        vgm = self.pipeline_config.visual_gen_mapping
        return int(getattr(vgm, "ulysses_size", 1) or 1) if vgm is not None else 1

    # ------------------------------------------------------------------ text encoder CUDA graph
    def _text_prefill_fast(
        self, input_ids: torch.Tensor, vision_inputs: dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if not SCENARIO.text_encoder_graph or vision_inputs or self.device.type != "cuda":
            return None
        with self.offloader.context_if_requested("text_encoder"):
            return self._encode_text_tokens_graph(input_ids)

    def _text_prefill_layers(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The text-only Qwen3-VL prefill up to the layer MiniMax-H3 reads, as the modules run it.

        Replicates ``Qwen3VLModel.forward`` -> ``Qwen3VLTextModel.forward`` for a text-only prompt: the
        token embeddings, the 4-row position ids (text row + the three mRoPE rows, all ``arange``),
        the rotary tables, then the decoder layers with no attention mask (the HF call's all-ones mask
        is dropped by its ``_ignore_causal_mask_sdpa`` check, so the layers run with ``attention_mask=None``
        and SDPA's ``is_causal`` path either way). ``hidden_states[k]`` of the HF output is the output of
        layer ``k - 1``, so the first ``MINIMAX_H3_TEXT_ENCODER_LAYER`` layers are run and the later ones,
        which the HF call also executes, are skipped. No host synchronisation, so the call captures.
        """
        text_model = self.text_encoder.model.language_model
        hidden_states = text_model.embed_tokens(input_ids)
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        position_ids = position_ids.view(1, 1, -1).expand(4, hidden_states.shape[0], -1)
        text_position_ids, mrope_position_ids = position_ids[0], position_ids[1:]
        position_embeddings = text_model.rotary_emb(hidden_states, mrope_position_ids)
        for decoder_layer in text_model.layers[:MINIMAX_H3_TEXT_ENCODER_LAYER]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
        return hidden_states

    def _encode_text_tokens_graph(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``_text_prefill_layers`` replayed from a CUDA graph keyed by the prompt's token count.

        The first prompt of a length runs the layers eagerly once (the same kernels warm), then
        captures; later prompts of that length copy their ids into the static input and replay.
        The static output is cloned, since the next replay of this length overwrites it. At most
        ``text_encoder_graph_max`` lengths are kept (LRU).
        """
        seq_len = int(input_ids.shape[1])
        graphs = self.__dict__.setdefault("_te_graphs", {})
        entry = graphs.get(seq_len)
        if entry is None:
            if len(graphs) >= SCENARIO.text_encoder_graph_max:
                oldest = next(iter(graphs))
                del graphs[oldest]
            static_ids = input_ids.clone()
            self._text_prefill_layers(static_ids)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._text_prefill_layers(static_ids)
            entry = (graph, static_ids, static_out)
            logger.info(
                f"MiniMax-H3 text encoder: CUDA graph captured for {seq_len} tokens ({len(graphs) + 1} lengths)"
            )
        else:
            del graphs[seq_len]  # re-insert: dict order is the LRU order
        graphs[seq_len] = entry
        graph, static_ids, static_out = entry
        static_ids.copy_(input_ids)
        graph.replay()
        return static_out.clone().to(device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------ before the denoise loop
    def _staged(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """*tensor* on the device through a pinned buffer kept per (name, dtype, shape), not blocking.

        A pageable copy waits for the stream, which the text-encoder graph occupies. Each buffer is
        written once per request and a request ends synchronized, so no copy is in flight on reuse.
        """
        if not SCENARIO.preloop_async_copies or self.device.type != "cuda":
            return tensor.to(self.device)
        buffers = self.__dict__.setdefault("_preloop_pinned", {})
        key = (name, tensor.dtype, tuple(tensor.shape))
        buffer = buffers.pop(key, None)
        if buffer is None:
            if len(buffers) >= 32:
                del buffers[next(iter(buffers))]
            buffer = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=prefer_pinned())
        buffers[key] = buffer  # re-insert: dict order is the LRU order
        buffer.copy_(tensor)
        return buffer.to(self.device, non_blocking=True)

    def _layout_to_device(self, layout: MiniMaxH3PackedSequence) -> tuple[torch.Tensor, ...]:
        if not SCENARIO.preloop_async_copies:
            return super()._layout_to_device(layout)
        return (
            self._staged("position_ids", layout.position_ids),
            self._staged("token_tags", layout.token_tags),
            self._staged("video_indices", layout.video_indices),
            self._staged("audio_indices", layout.audio_indices),
            self._staged("text_indices", layout.text_indices),
        )

    def _text_encoding_hook(
        self,
        generator: torch.Generator,
        noise_shapes: tuple[tuple[int, ...], ...],
        keyframes: list,
    ) -> Optional[Callable[[], None]]:
        self.__dict__.pop("_preloop_noise", None)
        # With keyframes the condition noise is drawn first, so only a text-only request can draw early.
        if not SCENARIO.preloop_noise_early or keyframes or generator.device.type != "cpu":
            return None

        def draw() -> None:
            # The request's two draws, in the family's order, while the text encoder occupies the GPU.
            self._preloop_noise = [
                (shape, randn_tensor(shape, generator=generator, device="cpu", dtype=torch.float32))
                for shape in noise_shapes
            ]

        return draw

    def _draw_noise(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        pending = self.__dict__.get("_preloop_noise")
        if pending:
            drawn_shape, noise = pending.pop(0)
            if tuple(drawn_shape) != tuple(shape):
                raise RuntimeError(
                    f"MiniMax-H3 early noise {drawn_shape} does not match the draw {shape}"
                )
            return self._staged(f"noise{len(shape)}", noise)
        if not SCENARIO.preloop_async_copies or generator.device.type != "cpu":
            return super()._draw_noise(shape, generator)
        # The family draws on the generator's device (CPU) and copies; the same draw, staged.
        noise = randn_tensor(shape, generator=generator, device="cpu", dtype=torch.float32)
        return self._staged(f"noise{len(shape)}", noise)

    def _host_timesteps(self, num_inference_steps: int) -> tuple[list[float], list[float]]:
        if not SCENARIO.preloop_async_copies:
            return super()._host_timesteps(num_inference_steps)
        # set_timesteps builds its grid on the CPU from the step count and the shift alone, so the
        # host values are read back once per (step count, scheduler settings, shift) and kept.
        cache = self.__dict__.setdefault("_preloop_timesteps", {})
        key = (int(num_inference_steps),) + tuple(
            (repr(sorted(dict(getattr(s, "config", {})).items(), key=str)), float(s.shift))
            for s in (self.scheduler, self.audio_scheduler)
        )
        host = cache.get(key)
        if host is None:
            if len(cache) >= 8:
                del cache[next(iter(cache))]
            host = cache[key] = super()._host_timesteps(num_inference_steps)
        return host

    def _row_timestep_plan(
        self,
        layout: MiniMaxH3PackedSequence,
        video_timesteps: list[float],
        audio_timesteps: list[float],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not SCENARIO.preloop_device_plan:
            return super()._row_timestep_plan(layout, video_timesteps, audio_timesteps)
        # A step's row indices are its four class positions gathered over the layout's class map:
        # the same int64 values as the family's host gather per step, from one gather on the device.
        planner = RowTimestepPlanner(layout)
        cache = self.__dict__.setdefault("_preloop_class_plans", {})
        key = (tuple(video_timesteps), tuple(audio_timesteps), tuple(planner.present.tolist()))
        entry = cache.get(key)
        if entry is None:
            if len(cache) >= 8:
                del cache[next(iter(cache))]
            steps = [
                planner.class_positions(
                    float(video_timestep),
                    float(audio_timestep),
                    max(float(video_timestep), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                    1.0,
                )
                for video_timestep, audio_timestep in zip(video_timesteps, audio_timesteps)
            ]
            entry = cache[key] = (
                [int(unique.numel()) for unique, _ in steps],
                torch.cat([unique for unique, _ in steps]),
                torch.stack([positions for _, positions in steps]),
            )
        unique_sizes, unique_host, positions_host = entry
        unique_all = self._staged("plan_unique", unique_host)
        positions_all = self._staged("plan_positions", positions_host)
        indices_all = positions_all.index_select(1, self._staged("plan_classes", planner.classes))
        return [
            (unique, indices_all[index])
            for index, unique in enumerate(unique_all.split(unique_sizes))
        ]
