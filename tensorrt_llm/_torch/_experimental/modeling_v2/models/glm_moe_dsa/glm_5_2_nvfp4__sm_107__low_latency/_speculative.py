# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-owned model execution with its embedded MTP draft layers."""

from typing import Dict, Generic, Iterator, List, Optional

import torch
from torch import nn

from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig, TConfig
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, TModel
from tensorrt_llm._torch.pyexecutor.guided_decoder import CapturableGuidedDecoder
from tensorrt_llm._torch.speculative import (
    SpecMetadata,
    get_spec_worker,
    should_use_separate_draft_kv_cache,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from ._mtp import build_mtp_draft
from ._mtp_config import _replacement_mtp_model_config


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
    """Target-owned forward and weight lifecycle for one-model MTP-Eagle."""

    spec_decode_fast_paths = False

    def __init__(
        self,
        model: TModel,
        model_config: ModelConfig[TConfig],
        hidden_size: int | None = None,
        vocab_size: int | None = None,
    ) -> None:
        if hidden_size is None:
            hidden_size = model_config.pretrained_config.hidden_size
        if vocab_size is None:
            vocab_size = model_config.pretrained_config.vocab_size
        super().__init__(model, config=model_config, hidden_size=hidden_size, vocab_size=vocab_size)
        self.draft_model = None
        self.draft_config = None
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        self.spec_config = model_config.spec_config
        spec_config = self.spec_config
        if spec_config is not None:
            if (
                not spec_config.spec_dec_mode.is_mtp_eagle_one_model()
                or spec_config.uses_external_draft_model
            ):
                raise ValueError(
                    "The GLM Rubin target supports embedded MTP-Eagle or no speculation"
                )
            if spec_config.uses_replacement_heads:
                self.draft_config = _replacement_mtp_model_config(model_config)
            self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(spec_config)
            self.draft_model = build_mtp_draft(
                model_config, self.draft_config, self.lm_head, self.model
            )
            if spec_config.uses_replacement_heads:
                self.draft_config = self.draft_model.model_config
            self.epilogue.append(self.draft_model)
            self.spec_worker = get_spec_worker(
                spec_config,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache,
            )
            if self.spec_worker is not None:
                self.spec_worker.set_draft_model(self.draft_model)
                self.epilogue.append(self.spec_worker)
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
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
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

            # Preserve token IDs when the caller provides precomputed embeddings.
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

    def _quantization_named_modules(self) -> Iterator[tuple[str, nn.Module]]:
        if not getattr(self.spec_config, "uses_replacement_heads", False):
            return super()._quantization_named_modules()
        owned_ids = {id(module) for _, module in self.draft_model._quantization_named_modules()}
        return (
            (name, module) for name, module in self.named_modules() if id(module) not in owned_ids
        )

    def __post_init__(self) -> None:
        if getattr(self.spec_config, "uses_replacement_heads", False):
            self.draft_model.apply_quant_config()
        super().__post_init__()

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

    def load_draft_weights(self, weights: Dict, weight_mapper: Optional[BaseWeightMapper] = None):
        from tensorrt_llm._torch.models.modeling_utils import _load_weights_impl_v2
        from tensorrt_llm._torch.speculative.utils import (
            remap_preprocessed_mtp_weights_for_draft_model,
            select_mtp_checkpoint_weights,
            skip_modules_for_separate_mtp_checkpoint,
            uses_mtp_head_checkpoint,
        )

        if uses_mtp_head_checkpoint(self.spec_config):
            # Require every non-shared MTP parameter to have a checkpoint tensor.
            n_total = len(weights)
            lm_head_weights = {}
            if getattr(self.draft_model, "owns_lm_head", False):
                lm_head_weights = {
                    name: value for name, value in weights.items() if name.startswith("lm_head.")
                }
                required_head_tensors = {"lm_head.weight"}
                head_quant_algo = self.draft_model.lm_head.quant_config.quant_algo
                if head_quant_algo in (QuantAlgo.FP8, QuantAlgo.W4A16_NVFP4):
                    required_head_tensors.add("lm_head.weight_scale")
                if head_quant_algo == QuantAlgo.W4A16_NVFP4:
                    required_head_tensors.add("lm_head.weight_scale_2")
                missing = sorted(required_head_tensors - lm_head_weights.keys())
                if missing:
                    raise ValueError(
                        "Quantized replacement MTP checkpoint is missing " + ", ".join(missing)
                    )
            weights = select_mtp_checkpoint_weights(weights)
            if not weights:
                raise ValueError(
                    "speculative_model was set for MTP but no 'mtp.*' weights "
                    f"were found in {self.spec_config.speculative_model!r}. "
                    "Expected keys like 'mtp.layers.0.*'."
                )
            n_dropped = n_total - len(weights) - len(lm_head_weights)
            if n_dropped:
                logger.warning(
                    "Ignoring %d unrelated tensors from speculative_model while "
                    "loading MTP heads (kept %d MTP and %d LM head tensors).",
                    n_dropped,
                    len(weights),
                    len(lm_head_weights),
                )
            if weight_mapper is None:
                raise ValueError("weight_mapper is required to load separate MTP heads")
            weights = weight_mapper.preprocess_weights(weights)
            num_hidden_layers = self.config.num_hidden_layers
            num_mtp_layers = len(self.draft_model.mtp_layers)
            weights = remap_preprocessed_mtp_weights_for_draft_model(
                weights,
                num_hidden_layers=num_hidden_layers,
                num_mtp_layers=num_mtp_layers,
            )
            weights.update(lm_head_weights)

            # Skip optional modules (e.g. shared_head) only when absent from
            # this checkpoint; architectures that ship those tensors still load
            # them under allow_partial_loading=False.
            _load_weights_impl_v2(
                self.draft_model,
                weights,
                weight_mapper,
                skip_modules=skip_modules_for_separate_mtp_checkpoint(weights),
                allow_partial_loading=False,
            )
            return

        raise ValueError(
            "The GLM Rubin target only loads separate draft weights for replacement MTP heads"
        )

    def set_guided_decoder(self, guided_decoder: CapturableGuidedDecoder) -> bool:
        if hasattr(self.spec_worker, "set_guided_decoder"):
            return self.spec_worker.set_guided_decoder(guided_decoder)
        return False
