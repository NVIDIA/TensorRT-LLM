# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-owned execution and lifecycle for GLM's embedded MTP draft layer."""

from typing import Generic, List, Optional

import torch

from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig, TConfig
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, TModel
from tensorrt_llm._torch.pyexecutor.guided_decoder import CapturableGuidedDecoder
from tensorrt_llm._torch.speculative import SpecMetadata, should_use_separate_draft_kv_cache
from tensorrt_llm.mapping import Mapping

from ._mtp import build_mtp_draft
from ._mtp_worker import RubinMTPEagleWorker


def _slice_spec_position_ids(
    position_ids: Optional[torch.Tensor], num_tokens: int
) -> Optional[torch.Tensor]:
    """Slice speculative position IDs along the token dimension."""
    if position_ids is None:
        return None
    return position_ids[..., :num_tokens]


class Glm52SpecDecOneEngineForCausalLM(
    DecoderModelForCausalLM[TModel, TConfig], Generic[TModel, TConfig]
):
    """Run the HIGH target and its checkpoint MTP layer in one engine."""

    spec_decode_fast_paths = False
    runtime_hooks = None

    @classmethod
    def supports_spec_decode_fast_paths(cls, model_config: ModelConfig) -> bool:
        return False

    def __init__(self, model: TModel, model_config: ModelConfig[TConfig], mapping: Mapping) -> None:
        super().__init__(
            model,
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )
        self.spec_config = model_config.spec_config
        self.draft_config = None
        self.draft_model = None
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        if self.spec_config is not None:
            if not (
                self.spec_config.spec_dec_mode.is_mtp_eagle_one_model()
                and not self.spec_config.uses_external_draft_model
                and not self.spec_config.uses_replacement_heads
            ):
                raise ValueError(
                    "The GLM high-throughput target requires the checkpoint's MTP layer"
                )
            self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(self.spec_config)
            self.draft_model = build_mtp_draft(model_config, None, self.lm_head, self.model)
            self.epilogue.append(self.draft_model)
            self.spec_worker = RubinMTPEagleWorker(
                self.spec_config,
                model_config=model_config,
                mapping=mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache,
            )
            self.spec_worker.set_draft_model(self.draft_model)
            self.spec_worker.set_target_model(self)
            self.epilogue.append(self.spec_worker)
            self.num_hidden_layers = self.config.num_hidden_layers
            self.model.layers.extend(self.draft_model.mtp_layers)
        self.layer_idx = -1

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        spec_config = self.spec_config
        if (
            self.spec_worker is not None
            and spec_config is not None
            and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
            and spec_config.draft_lm_head_dtype == "mxfp8"
        ):
            from tensorrt_llm._torch.modules.mxfp8_draft_head import build_mxfp8_draft_head

            # Rebuilt from the current BF16 weights (idempotent on reloads),
            # from the LM head the draft passes project with.
            draft_lm_head = getattr(self.draft_model, "lm_head", None)
            self.spec_worker.draft_head = None
            self.spec_worker.draft_head = build_mxfp8_draft_head(
                self.lm_head if draft_lm_head is None else draft_lm_head,
                self.model_config.mapping,
                self.spec_worker.key_exchange,
            )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        # A padded generation batch's (mrope) positions are a strided view; one
        # copy here instead of one per attention layer that needs them dense.
        # Not with context requests: a breakable prefill CUDA graph replays its
        # body against the capture-time position tensor.
        hoist_positions = (
            self.spec_decode_fast_paths
            and self.spec_worker is not None
            and position_ids is not None
            and attn_metadata.num_contexts == 0
        )
        if hoist_positions:
            position_ids = position_ids.contiguous()
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if getattr(attn_metadata, "csa2_remote_tail_mode", None) == "source":
            # The context worker intentionally stopped at the encoder/decoder
            # boundary. Produce a disposable token for the existing context-only
            # completion machinery, but do not run an embedded draft worker from
            # an encoder-boundary activation. The handoff strips this token.
            return self.logits_processor.forward(
                hidden_states,
                self.lm_head,
                attn_metadata,
                False,
            )
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states)
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[: attn_metadata.num_tokens]

        if self.spec_worker is not None:
            # The target logits, in the layout the worker's acceptance reads.
            logits = self.spec_worker.target_logits(
                hidden_states[spec_metadata.gather_ids],
                self.lm_head,
                self.logits_processor,
                attn_metadata,
                spec_metadata,
                self.draft_model,
            )

            # VLM wrappers (e.g. Qwen3VLModelBase) replace input_ids with
            # fused inputs_embeds; fall back to the pre-fusion token IDs
            # they forward via `orig_input_ids` so MTP / Eagle drafters
            # can still access the prompt tokens.
            spec_input_ids = input_ids if input_ids is not None else kwargs.get("orig_input_ids")
            spec_position_ids = position_ids
            if attn_metadata.padded_num_tokens is not None:
                if spec_input_ids is not None:
                    # Slice along the first dimension
                    spec_input_ids = spec_input_ids[: attn_metadata.num_tokens]
                if position_ids is not None:
                    spec_position_ids = _slice_spec_position_ids(
                        position_ids, attn_metadata.num_tokens
                    )
                    if hoist_positions:
                        # The padded batch's slice is strided again.
                        spec_position_ids = spec_position_ids.contiguous()

            # get accepted tokens and next draft tokens
            return self.spec_worker(
                input_ids=spec_input_ids,
                position_ids=spec_position_ids,
                hidden_states=hidden_states,
                logits=logits,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=self.draft_model,
                resource_manager=resource_manager,
            )
        else:
            logits = self.logits_processor.forward(
                hidden_states,
                self.lm_head,
                attn_metadata,
                return_context_logits,
            )

        return logits

    def mtp_head_module_names(self) -> List[str]:
        """Names of the MTP heads under every alias they are reachable by.

        One-model MTP registers the same head objects twice: under
        ``draft_model.mtp_layers.{h}`` and, after the target model extends its
        layer list, under ``model.layers.{num_hidden_layers + h}``. A load that
        wants to leave the heads untouched has to exclude both aliases.
        """
        mtp_layers = getattr(self.draft_model, "mtp_layers", None)
        if not mtp_layers:
            return []
        head_ids = {id(layer) for layer in mtp_layers}
        return [
            name
            for name, module in self.named_modules(remove_duplicate=False)
            if name and id(module) in head_ids
        ]

    def load_draft_weights(
        self, weights: dict[str, torch.Tensor], weight_mapper: BaseWeightMapper | None = None
    ) -> None:
        raise ValueError(
            "The GLM high-throughput target loads its MTP layer from the target checkpoint"
        )

    def set_guided_decoder(self, guided_decoder: CapturableGuidedDecoder) -> bool:
        if hasattr(self.spec_worker, "set_guided_decoder"):
            return self.spec_worker.set_guided_decoder(guided_decoder)
        return False
