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

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from PIL import Image

from tensorrt_llm._torch.visual_gen.checkpoints import WeightLoader
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.pipeline import ExtraParamSchema, RefSlotSpec, RoleSpec
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.logger import logger

from ..components.linear import TensorParallelSubgroup
from ..components.transformer import (
    MiniMaxH3Transformer3DModel,
    takes_key_padding_mask,
    trunk_attention_for,
)
from ..packing import resolve_canvas_size
from ..pipeline import H3PipelineBase
from .ref2va import load_references, prepare_references, validate_reference_order

if TYPE_CHECKING:
    from .text_encoder_tp import MiniMaxH3TensorParallelTextEncoder


class MiniMaxH3Pipeline(H3PipelineBase):
    """MiniMax H3 functional pipeline for text and reference conditioning."""

    def __init__(self, pipeline_config: DiffusionPipelineConfig) -> None:
        self.text_encoder_tp_size = int(
            getattr(getattr(pipeline_config, "parallel", None), "text_encoder_tp_size", 1)
        )
        vgm = getattr(pipeline_config, "visual_gen_mapping", None)
        world_size = vgm.world_size if vgm is not None else pipeline_config.mapping.world_size
        if self.text_encoder_tp_size > world_size:
            raise ValueError(
                f"text_encoder_tp_size ({self.text_encoder_tp_size}) exceeds "
                f"world_size ({world_size})."
            )
        if world_size != 1:
            supports_topology = (
                vgm is not None
                and vgm.cfg_size == 1
                and vgm.cp_size == 1
                and vgm.ring_size == 1
                and vgm.attn2d_row_size == 1
                and vgm.attn2d_col_size == 1
                and vgm.ulysses_size * pipeline_config.mapping.tp_size == world_size
            )
            if not supports_topology:
                raise NotImplementedError(
                    "MiniMax-H3 supports tensor parallelism and Ulysses: "
                    "cfg_size=1, ring_size=1, attn2d_size=(1, 1), and "
                    "tp_size * ulysses_size=world_size."
                )
            if vgm.ulysses_size > 1 and not takes_key_padding_mask(pipeline_config.attention):
                raise NotImplementedError(
                    "MiniMax-H3 Ulysses requires VANILLA, FA4, TRTLLM, FLASHINFER, or CUTEDSL "
                    "with deco_attention. Other attention backends do not support the required "
                    "packed-row padding mask."
                )
        self._setup_text_encoder_group(world_size)

        self.workflow = pipeline_config.extra_attrs.get("workflow", "fl2va")
        super().__init__(pipeline_config)

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

    def _init_transformer(self) -> None:
        config = self.pipeline_config.primary_model_config
        self.transformer = MiniMaxH3Transformer3DModel(
            config, attention_cls=trunk_attention_for(config)
        )

    def load_transformer_weights(self, checkpoint_dir: str) -> dict:
        if self.workflow == "ref2va":
            return WeightLoader(components="transformer_ref").load_weights(
                checkpoint_dir, self.mapping
            )
        return super().load_transformer_weights(checkpoint_dir)

    def _setup_text_encoder_group(self, world_size):
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
            if self.text_encoder_tp_size == world_size:
                group = dist.group.WORLD
            else:
                group = dist.new_group(
                    ranks=text_encoder_tp_ranks,
                    use_local_synchronization=False,
                )
            if self.rank in text_encoder_tp_ranks:
                self._text_encoder_tp_group = group

    def _load_tensor_parallel_text_encoder(
        self,
        checkpoint_dir: str,
        device: torch.device,
    ) -> "MiniMaxH3TensorParallelTextEncoder":
        # Its Triton Gluon kernel is imported only when a TP text encoder is built.
        from .text_encoder_tp import MiniMaxH3TensorParallelTextEncoder

        if not self._loads_text_encoder_on_this_rank:
            raise RuntimeError(
                "MiniMax-H3 tensor-parallel text encoder is loaded only on "
                f"ranks < {self.text_encoder_tp_size}; rank {self.rank} "
                "must receive broadcast prompt embeddings."
            )
        text_encoder_mapping = TensorParallelSubgroup(
            self.text_encoder_tp_size, self.rank, self._text_encoder_tp_group
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

    def _encode_prompt(
        self,
        prompt: str,
        keyframes: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._uses_tensor_parallel_text_encoder and keyframes:
            raise NotImplementedError(
                "MiniMax-H3 tensor-parallel text encoder currently supports "
                "text-only prompts; set text_encoder_tp_size=1 for keyframe requests."
            )
        if self.world_size == 1:
            return self._encode_prompt_local(prompt, keyframes)

        prompt_embeds: torch.Tensor | None = None
        text_token_tags: torch.Tensor | None = None
        with self._coordinate_text_encoder_errors():
            if self._loads_text_encoder_on_this_rank:
                prompt_embeds, text_token_tags = self._encode_prompt_local(prompt, keyframes)

        prompt_embeds = self._broadcast_tensor_from_rank0(prompt_embeds, dtype=self.dtype)
        text_token_tags = self._broadcast_tensor_from_rank0(text_token_tags, dtype=torch.long)
        return prompt_embeds, text_token_tags.cpu()

    @property
    def _uses_tensor_parallel_text_encoder(self) -> bool:
        return self.text_encoder_tp_size > 1

    @property
    def _loads_text_encoder_on_this_rank(self) -> bool:
        return self.rank < self.text_encoder_tp_size

    @contextmanager
    def _coordinate_text_encoder_errors(self) -> Iterator[None]:
        """Propagate caught encoding failures before peers enter payload broadcasts."""
        if self.world_size == 1 or not dist.is_initialized():
            yield
            return

        # Allocate before encoding so an encoder OOM does not prevent reporting it.
        failed_rank = torch.zeros(1, dtype=torch.int32, device=self.device)
        local_error: Exception | None = None
        try:
            yield
        except Exception as error:
            local_error = error
            failed_rank.fill_(self.rank + 1)

        # This coordinates Python failures only; failed TP collectives still
        # require distributed worker teardown.
        dist.all_reduce(failed_rank, op=dist.ReduceOp.MAX)
        failed_rank_id = int(failed_rank.item()) - 1
        if failed_rank_id >= 0:
            message = f"MiniMax-H3 text encoding failed on rank {failed_rank_id}"
            if local_error is not None:
                message += f": {local_error}"
            raise RuntimeError(message) from local_error

    def _broadcast_tensor_from_rank0(
        self,
        tensor: torch.Tensor | None,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not dist.is_initialized():
            if tensor is None:
                raise ValueError("rank 0 tensor is required when distributed is not initialized.")
            return tensor.to(device=self.device, dtype=dtype)

        shape = torch.zeros(8, dtype=torch.long, device=self.device)
        if self.rank == 0:
            if tensor is None:
                raise ValueError("rank 0 tensor is required for MiniMax-H3 broadcast.")
            tensor = tensor.to(device=self.device, dtype=dtype).contiguous()
            shape[0] = tensor.ndim
            shape[1 : tensor.ndim + 1] = torch.tensor(tensor.shape, device=self.device)
        dist.broadcast(shape, src=0)
        ndim = int(shape[0].item())
        tensor_shape = tuple(int(value) for value in shape[1 : ndim + 1].tolist())
        if self.rank != 0:
            tensor = torch.empty(tensor_shape, dtype=dtype, device=self.device)
        dist.broadcast(tensor, src=0)
        return tensor

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
        if not self._loads_text_encoder_on_this_rank:
            logger.info("MiniMax-H3 rank %d receives prompt embeddings from rank 0", self.rank)
            return None
        if self._uses_tensor_parallel_text_encoder:
            return self._load_tensor_parallel_text_encoder(checkpoint_dir, device)
        return super()._load_text_encoder(checkpoint_dir, device, offloaded)

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        if self.workflow != "ref2va":
            super()._run_warmup(height, width, num_frames, steps)

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
            if references:
                raise ValueError(
                    "Reference conditioning requires pipeline_config workflow='ref2va'."
                )
            return None
        if keyframes:
            raise ValueError("Ref2VA accepts references, not first/last keyframes.")
        return prepare_references(
            self, references or [], prompt, height, width, num_frames, generator
        )
