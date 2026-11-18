# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

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

import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.checkpoints import WeightLoader
from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema, RefSlotSpec, RoleSpec
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm._utils import mpi_disabled

from ..packing import resolve_canvas_size
from ..pipeline import H3PipelineBase
from .ref2va import load_references, prepare_references, validate_reference_order
from .text_encoder_tp import MiniMaxH3TensorParallelTextEncoder, _MiniMaxH3TextEncoderMapping


class MiniMaxH3Pipeline(H3PipelineBase):
    """MiniMax H3 functional pipeline for text and reference conditioning."""

    @property
    def ref_slot_specs(self) -> dict[str, RefSlotSpec]:
        if self.workflow == "ref2va":
            return {
                f"{kind}_reference": RefSlotSpec(
                    modality=kind, roles=[RoleSpec(role="reference", min=0, max=limit)]
                )
                for kind, limit in (("image", 9), ("video", 3), ("audio", 3))
            }
        return super().ref_slot_specs

    @property
    def extra_param_specs(self) -> dict[str, ExtraParamSchema]:
        if self.workflow != "ref2va":
            return {}
        return {
            "reference_order": ExtraParamSchema(
                type="list",
                default=None,
                validator=validate_reference_order,
                description="Ordered image:N/video:N/audio:N references (zero-based); "
                "defaults to images, videos, then audio, preserving each slot's order.",
            )
        }

    def load_transformer_weights(self, checkpoint_dir: str) -> dict:
        if self.workflow == "ref2va":
            return WeightLoader(components="transformer_ref").load_weights(
                checkpoint_dir, self.mapping
            )
        return super().load_transformer_weights(checkpoint_dir)

    def _setup_text_encoder_group(self, pipeline_config, world_size):
        self._text_encoder_tp_group = None
        if (
            self._uses_tensor_parallel_text_encoder
            and self.text_encoder_tp_size < world_size
            and not mpi_disabled()
        ):
            raise RuntimeError(
                "MiniMax-H3 subset text_encoder_tp_size requires TLLM_DISABLE_MPI=1 "
                "so Qwen text encoder TP uses a torch process group instead of MPI collectives."
            )
        if self._uses_tensor_parallel_text_encoder and dist.is_initialized():
            text_encoder_tp_ranks = list(range(self.text_encoder_tp_size))
            rank = int(getattr(pipeline_config.mapping, "rank", 0))
            if self.text_encoder_tp_size == world_size:
                group = dist.group.WORLD
            else:
                group = dist.new_group(
                    ranks=text_encoder_tp_ranks,
                    use_local_synchronization=False,
                )
            if rank in text_encoder_tp_ranks:
                self._text_encoder_tp_group = group

    def _load_tensor_parallel_text_encoder(
        self,
        checkpoint_dir: str,
        device: torch.device,
    ) -> MiniMaxH3TensorParallelTextEncoder:
        if not self._loads_text_encoder_on_this_rank:
            raise RuntimeError(
                "MiniMax-H3 tensor-parallel text encoder is loaded only on "
                f"ranks < {self.text_encoder_tp_size}; rank {self.rank} "
                "must receive broadcast prompt embeddings."
            )
        text_encoder_mapping = _MiniMaxH3TextEncoderMapping(
            world_size=self.text_encoder_tp_size,
            rank=self.rank,
            tp_size=self.text_encoder_tp_size,
            gpus_per_node=self.text_encoder_tp_size,
            tp_group_pg=self._text_encoder_tp_group,
        )
        text_encoder = MiniMaxH3TensorParallelTextEncoder.from_pretrained_config(
            checkpoint_dir, text_encoder_mapping, device
        )
        # Safetensors prefetch synchronizes on the default process group. Disable
        # it when only a subset of ranks loads the text encoder.
        prefetch = self.text_encoder_tp_size == self.world_size
        weights = WeightLoader(components=PipelineComponent.TEXT_ENCODER).load_weights(
            checkpoint_dir, text_encoder_mapping, prefetch=prefetch
        )
        text_encoder.load_weights(weights)
        return text_encoder.eval()

    def _encode_text_tokens(
        self, token_ids: list[int], vision_inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if not self._uses_tensor_parallel_text_encoder:
            return super()._encode_text_tokens(token_ids, vision_inputs)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        if vision_inputs:
            raise NotImplementedError(
                "MiniMax-H3 tensor-parallel text encoder currently supports "
                "text-only prompts; set text_encoder_tp_size=1 for requests "
                "with Qwen3-VL vision inputs."
            )
        with self.offloader.context_if_requested("text_encoder"):
            return self.text_encoder.encode(input_ids).to(device=self.device, dtype=self.dtype)

    def _encode_reference_text_tokens(
        self,
        token_ids: list[int],
        vision_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode Ref2VA presentation tokens on owner ranks and broadcast."""
        if self._uses_tensor_parallel_text_encoder and vision_inputs:
            raise NotImplementedError(
                "MiniMax-H3 tensor-parallel text encoder currently supports "
                "text-only prompts; set text_encoder_tp_size=1 for Ref2VA "
                "requests with image or video references."
            )
        if self.world_size == 1:
            return self._encode_text_tokens(token_ids, vision_inputs)

        prompt_embeds: torch.Tensor | None = None
        with self._coordinate_text_encoder_errors():
            if self._loads_text_encoder_on_this_rank:
                prompt_embeds = self._encode_text_tokens(token_ids, vision_inputs)
        return self._broadcast_tensor_from_rank0(prompt_embeds, dtype=self.dtype)

    def _load_text_encoder(self, checkpoint_dir, device, offloaded):
        if self._uses_tensor_parallel_text_encoder:
            return self._load_tensor_parallel_text_encoder(checkpoint_dir, device)
        return super()._load_text_encoder(checkpoint_dir, device, offloaded)

    def _configure_scheduler_shifts(self):
        if self.scheduler is not None:
            self.scheduler.set_shift(self.VIDEO_SCHEDULER_SHIFT)
        if self.audio_scheduler is not None:
            self.audio_scheduler.set_shift(self.AUDIO_SCHEDULER_SHIFT)

    def prepare_request(self, req):
        if self.workflow != "ref2va":
            return super().prepare_request(req)
        req.prepared_inputs["references"] = load_references(req.params)
        if (req.params.height is None) != (req.params.width is None):
            raise ValueError("MiniMax-H3 height and width must be set together.")
        if req.params.height is None:
            req.params.height, req.params.width = resolve_canvas_size(16, 9)

    def _request_conditions(self, req):
        if self.workflow != "ref2va":
            return super()._request_conditions(req)
        prepared = getattr(req, "prepared_inputs", {})
        refs = prepared["references"] if "references" in prepared else load_references(req.params)
        return dict(
            references=refs,
            keyframes=prepared.get("keyframes"),
            keyframe_anchors=prepared.get("keyframe_anchors"),
        )

    def _reference_conditioning(
        self, references, prompt, height, width, num_frames, keyframes, generator
    ):
        if self.workflow != "ref2va":
            return None
        if keyframes:
            raise ValueError("Ref2VA accepts references, not first/last keyframes.")
        return prepare_references(
            self, references or [], prompt, height, width, num_frames, generator
        )
