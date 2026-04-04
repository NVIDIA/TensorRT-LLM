# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Factory definitions for building models related to Eagle in AutoDeploy.

This module provides:
- EagleDrafterFactory: Factory for building Eagle speculative decoding draft models.
- EagleOneModelFactory: Factory that composes target + draft factories for one-model
  Eagle speculative decoding.
"""

import types
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from torch._prims_common import DeviceLikeType
from torch.export import Dim
from torch.fx import GraphModule

from ....llmapi.llm_args import MTPDecodingConfig
from ..utils.logger import ad_logger
from .custom.modeling_eagle import (
    EagleConfig,
    EagleDrafterForCausalLM,
    EagleWrapper,
    EagleWrapperConfig,
)
from .factory import DynamicShape, ModelFactory, ModelFactoryRegistry, SubModuleExportInfo
from .hf import AutoModelForCausalLMFactory


@ModelFactoryRegistry.register("EagleDrafter")
class EagleDrafterFactory(AutoModelForCausalLMFactory):
    """Factory for building Eagle drafter models.

    The drafter builds its own model-specific layers internally based on
    config.model_type, allowing it to work with different base models
    (Llama, NemotronH, etc.) without the factory needing to know the details.

    The checkpoint config is expected to have the base model's model_type
    (e.g., "llama") along with Eagle-specific fields like draft_vocab_size.
    """

    def _build_model(self, device: DeviceLikeType) -> nn.Module:
        model_config, unused_kwargs = self._get_model_config()

        # Get model type for config
        model_type = model_config.model_type
        ad_logger.info(f"EagleDrafterFactory: building drafter for model_type='{model_type}'")

        # Convert base config to EagleConfig, preserving existing values
        # and applying model-specific defaults based on model_type
        model_config = EagleConfig(model_config, model_type)

        with (init_empty_weights if device == "meta" else nullcontext)():
            model = EagleDrafterForCausalLM._from_config(model_config, **unused_kwargs)

        if device == "meta":
            # post-init must be called explicitly for HF models with init_empty_weights
            if hasattr(model, "post_init"):
                model.post_init()
        else:
            model.to(device)

        # Store checkpoint conversion mapping if present
        self._checkpoint_conversion_mapping = getattr(model, "_checkpoint_conversion_mapping", None)

        model.eval()

        return model

    def build_and_load_model(self, _device: DeviceLikeType) -> nn.Module:
        raise NotImplementedError(
            "EagleDrafterFactory does not support build_and_load_model(). "
            "Use build_model() + load_or_random_init() instead."
        )


# ============================================================================ #
#  EagleOneModelFactory -- combined target + draft for one-model spec dec      #
# ============================================================================ #


class TargetModelExportInfo(SubModuleExportInfo):
    """Export info for the target model inside EagleWrapper."""

    def __init__(self, load_lm_head_from_target: bool):
        super().__init__("target_model")
        self.load_lm_head_from_target = load_lm_head_from_target

    def _init_dynamic_shape_lookup(self) -> Dict[str, DynamicShape]:
        batch_size_dyn = Dim.DYNAMIC
        seq_len_dyn = Dim.DYNAMIC
        return {
            "inputs_embeds": {0: batch_size_dyn, 1: seq_len_dyn},
            "position_ids": {0: batch_size_dyn, 1: seq_len_dyn},
        }

    def post_process(self, sub_mod: nn.Module, sub_gm: GraphModule):
        """Preserve embedding (always) and optionally lm_head on the exported GraphModule."""
        # --- Embedding: always needed (target embeds input_ids for both target and draft) ---
        embed_tokens = sub_mod.get_input_embeddings()
        # Find the submodule path for the embedding
        for embed_name, subsubmod in sub_mod.named_modules():
            if subsubmod is embed_tokens:
                break
        else:
            raise RuntimeError("Could not find embedding module in target model.")
        sub_gm.set_submodule(embed_name, embed_tokens)
        sub_gm.get_input_embeddings = types.MethodType(
            lambda self, _n=embed_name: self.get_submodule(_n), sub_gm
        )
        # Add impure node to prevent GC
        n_embed = sub_gm.graph.get_attr(f"{embed_name}.weight")
        sub_gm.graph.call_function(
            torch._assert, args=(n_embed, "Avoid embedding getting deleted from graph.")
        )

        # --- lm_head: only if draft model loads it from target ---
        if self.load_lm_head_from_target:
            lm_head = sub_mod.get_output_embeddings()
            for lm_head_name, subsubmod in sub_mod.named_modules():
                if subsubmod is lm_head:
                    break
            else:
                raise RuntimeError("Could not find lm_head module in target model.")
            sub_gm.set_submodule(lm_head_name, lm_head)
            sub_gm.get_output_embeddings = types.MethodType(
                lambda self, _n=lm_head_name: self.get_submodule(_n), sub_gm
            )
            n_lm_head = sub_gm.graph.get_attr(f"{lm_head_name}.weight")
            sub_gm.graph.call_function(
                torch._assert, args=(n_lm_head, "Avoid lm_head getting deleted from graph.")
            )

        # --- Final normalization: only if target model exposes it (e.g., NemotronH for MTP) ---
        if hasattr(sub_mod, "get_final_normalization"):
            norm_module = sub_mod.get_final_normalization()
            for norm_name, subsubmod in sub_mod.named_modules():
                if subsubmod is norm_module:
                    break
            else:
                raise RuntimeError("Could not find final normalization module in target model.")
            sub_gm.set_submodule(norm_name, norm_module)
            sub_gm.get_final_normalization = types.MethodType(
                lambda self, _n=norm_name: self.get_submodule(_n), sub_gm
            )
            n_norm = sub_gm.graph.get_attr(f"{norm_name}.weight")
            sub_gm.graph.call_function(
                torch._assert, args=(n_norm, "Avoid final norm getting deleted from graph.")
            )


class DraftModelExportInfo(SubModuleExportInfo):
    """Export info for the draft model inside EagleWrapper."""

    def __init__(self, load_embedding_from_target: bool, load_lm_head_from_target: bool):
        super().__init__("draft_model")
        self.load_embedding_from_target = load_embedding_from_target
        self.load_lm_head_from_target = load_lm_head_from_target

    def _init_dynamic_shape_lookup(self) -> Dict[str, DynamicShape]:
        batch_size_dyn = Dim.DYNAMIC
        seq_len_dyn = Dim.DYNAMIC
        return {
            "inputs_embeds": {0: batch_size_dyn, 1: seq_len_dyn},
            "position_ids": {0: batch_size_dyn, 1: seq_len_dyn},
            "hidden_states": {0: batch_size_dyn, 1: seq_len_dyn},
        }

    def post_process(self, sub_mod: nn.Module, sub_gm: GraphModule):
        """Preserve modules needed by EagleWrapper utility methods (fc, d2t, embed, lm_head)."""
        inner_model = sub_mod.model
        inner_gm = sub_gm.get_submodule("model")

        # mark this gm as the draft model gm
        sub_gm.is_draft = True

        # --- Embedding (only if draft model has its own) ---
        if not self.load_embedding_from_target:
            embed_tokens = sub_mod.get_input_embeddings()
            for embed_name, subsubmod in sub_mod.named_modules():
                if subsubmod is embed_tokens:
                    break
            else:
                raise RuntimeError("Could not find embedding module in draft model.")
            sub_gm.set_submodule(embed_name, embed_tokens)
            sub_gm.get_input_embeddings = types.MethodType(
                lambda self, _n=embed_name: self.get_submodule(_n), sub_gm
            )
            n_embed = sub_gm.graph.get_attr(f"{embed_name}.weight")
            sub_gm.graph.call_function(
                torch._assert, args=(n_embed, "Avoid draft embedding getting deleted.")
            )

        # --- lm_head (only if draft model has its own) ---
        if not self.load_lm_head_from_target:
            lm_head = sub_mod.get_output_embeddings()
            for lm_head_name, subsubmod in sub_mod.named_modules():
                if subsubmod is lm_head:
                    break
            else:
                raise RuntimeError("Could not find lm_head module in draft model.")
            sub_gm.set_submodule(lm_head_name, lm_head)
            sub_gm.get_output_embeddings = types.MethodType(
                lambda self, _n=lm_head_name: self.get_submodule(_n), sub_gm
            )
            n_lm_head = sub_gm.graph.get_attr(f"{lm_head_name}.weight")
            sub_gm.graph.call_function(
                torch._assert, args=(n_lm_head, "Avoid draft lm_head getting deleted.")
            )

        # --- fc module (fuses hidden states from multiple layers) ---
        fc_module = getattr(inner_model, "fc", None)
        if fc_module is not None:
            sub_gm.set_submodule("model.fc", fc_module)
            n_fc = sub_gm.graph.get_attr("model.fc.weight")
            sub_gm.graph.call_function(torch._assert, args=(n_fc, "Avoid fc getting deleted."))

        # --- d2t parameter (draft-to-target vocab mapping) ---
        d2t = getattr(inner_model, "d2t", None)
        if d2t is not None:
            inner_gm.register_parameter("d2t", d2t)
            n_d2t = sub_gm.graph.get_attr("model.d2t")
            sub_gm.graph.call_function(torch._assert, args=(n_d2t, "Avoid d2t getting deleted."))

        # --- model dtype (used by apply_eagle3_fc) ---
        model_dtype = getattr(inner_model, "dtype", None)
        if model_dtype is not None:
            inner_gm.dtype = model_dtype


@ModelFactoryRegistry.register("eagle_one_model")
class EagleOneModelFactory(ModelFactory):
    """Factory that composes target + draft factories for one-model Eagle speculative decoding.

    Creates its own ``AutoModelForCausalLMFactory`` (target) and ``EagleDrafterFactory`` (draft)
    internally from the kwargs passed via ``ModelFactoryRegistry``. This keeps ``llm_args.py``
    clean -- it just sets ``model_factory = "eagle_one_model"`` and passes all kwargs through.

    Extra kwargs consumed beyond ``ModelFactory.__init__``:
        speculative_config: The ``EagleDecodingConfig`` with ``speculative_model`` path.
        speculative_model_kwargs: Optional dict of kwargs for the draft model config.
    """

    def __init__(
        self,
        model: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        skip_loading_weights: bool = False,
        max_seq_len: int = 512,
        speculative_config: Any = None,
        speculative_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            model_kwargs=model_kwargs,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            skip_loading_weights=skip_loading_weights,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        if speculative_config is None:
            raise ValueError("speculative_config is required for EagleOneModelFactory.")

        self.speculative_config = speculative_config
        # For MTP, derive Eagle-pipeline fields from MTP-specific fields.
        if isinstance(speculative_config, MTPDecodingConfig):
            draft_model_path = speculative_config.speculative_model or model
        else:
            draft_model_path = speculative_config.speculative_model
        if draft_model_path is None:
            raise ValueError("speculative_config.speculative_model must be set.")

        # Create target factory (AutoModelForCausalLM)
        self.target_factory = AutoModelForCausalLMFactory(
            model=model,
            model_kwargs=model_kwargs,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            skip_loading_weights=skip_loading_weights,
            max_seq_len=max_seq_len,
        )

        # Create draft factory (EagleDrafter)
        self.draft_factory = EagleDrafterFactory(
            model=str(draft_model_path),
            model_kwargs=speculative_model_kwargs,
            tokenizer=tokenizer,
            skip_loading_weights=skip_loading_weights,
            max_seq_len=max_seq_len,
        )

    @property
    def max_seq_len(self) -> int:
        return self.draft_factory.max_seq_len

    @property
    def vocab_size_padded(self) -> Optional[int]:
        return self.target_factory.vocab_size_padded

    def _build_model(self, device: str) -> nn.Module:
        target_model = self.target_factory.build_model(device)
        draft_model = self.draft_factory.build_model(device)

        draft_config = draft_model.config
        wrapper_config = EagleWrapperConfig(
            max_draft_len=self.speculative_config.max_draft_len,
            load_embedding_from_target=getattr(draft_config, "load_embedding_from_target", True),
            load_lm_head_from_target=getattr(draft_config, "load_lm_head_from_target", True),
            normalize_target_hidden_state=getattr(
                draft_config, "normalize_target_hidden_state", False
            ),
        )

        return EagleWrapper(
            config=wrapper_config, target_model=target_model, draft_model=draft_model
        )

    def _load_checkpoint(
        self, model: nn.Module, device: DeviceLikeType, disable_preload: bool = False
    ):
        """Load checkpoints for both target and draft submodels."""
        assert isinstance(model, EagleWrapper), f"Expected EagleWrapper, got {type(model)}"
        self.target_factory._load_checkpoint(model.target_model, device, disable_preload)
        self.draft_factory._load_checkpoint(model.draft_model, device, disable_preload)

    def load_or_random_init(
        self, model: nn.Module, device: DeviceLikeType, disable_preload: bool = False
    ):
        """Load or random-init weights for both target and draft submodels."""
        assert isinstance(model, EagleWrapper), f"Expected EagleWrapper, got {type(model)}"
        self.target_factory.load_or_random_init(model.target_model, device, disable_preload)
        self.draft_factory.load_or_random_init(model.draft_model, device, disable_preload)

    def get_export_infos(self, model: nn.Module) -> List[SubModuleExportInfo]:
        draft_config = model.draft_model.config
        load_embedding_from_target = getattr(draft_config, "load_embedding_from_target", True)
        load_lm_head_from_target = getattr(draft_config, "load_lm_head_from_target", True)

        return [
            TargetModelExportInfo(load_lm_head_from_target),
            DraftModelExportInfo(load_embedding_from_target, load_lm_head_from_target),
        ]

    def get_sharding_config(self) -> Dict[str, Any]:
        return self.target_factory.get_sharding_config()

    # TODO(govind): It's possible that draft models have different quant configs than target models.
    # We need to address this possibility.
    def get_quant_config(self) -> Dict[str, Any]:
        return self.target_factory.get_quant_config()

    def get_cache_config_updates(self) -> Dict[str, Any]:
        return self.target_factory.get_cache_config_updates()

    def init_tokenizer(self) -> Optional[Any]:
        return self.target_factory.init_tokenizer()

    def init_processor(self) -> Optional[Any]:
        return self.target_factory.init_processor()

    def prefetch_checkpoint(self, force: bool = False, skip_loading_weights: Optional[bool] = None):
        self.target_factory.prefetch_checkpoint(force, skip_loading_weights)
        self.draft_factory.prefetch_checkpoint(force, skip_loading_weights)
        super().prefetch_checkpoint(force, skip_loading_weights)
