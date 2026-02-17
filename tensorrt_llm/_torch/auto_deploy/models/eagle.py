# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
- EagleDrafterFactory: A specialized factory for building Eagle speculative decoding draft models.
  It extends AutoModelForCausalLMFactory to handle the mapping from base model types (e.g., "llama")
  to their corresponding Eagle drafter implementations.
- EagleOneModelFactory: A factory that composes target and draft model factories to build a combined
  Eagle model that can be used for speculative decoding with a single engine/optimization pass.
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

from ..utils.logger import ad_logger
from .custom.modeling_eagle import (
    ADHiddenStateManager,
    Eagle3DrafterForCausalLM,
    EagleConfig,
    EagleWrapper,
    EagleWrapperConfig,
)
from .factory import DynamicShape, ModelFactory, ModelFactoryRegistry, SubModuleExportInfo
from .hf import AutoModelForCausalLMFactory


@ModelFactoryRegistry.register("EagleDrafter")
class EagleDrafterFactory(AutoModelForCausalLMFactory):
    """Factory for building Eagle drafter models.

    This factory handles the mapping from base model types (e.g., "llama") to
    their corresponding Eagle drafter model implementations. It overrides
    _build_model() to directly construct the appropriate drafter class based
    on the checkpoint's model_type.

    The checkpoint config is expected to have the base model's model_type
    (e.g., "llama") along with Eagle-specific fields like draft_vocab_size.
    """

    _drafter_classes: Dict[str, type] = {
        "llama": Eagle3DrafterForCausalLM,
    }

    def _build_model(self, device: DeviceLikeType) -> nn.Module:
        model_config, unused_kwargs = self._get_model_config()

        # Select the appropriate drafter class and config based on the base model type
        model_type = model_config.model_type
        if model_type not in self._drafter_classes:
            raise ValueError(
                f"Unsupported model_type '{model_type}' for Eagle drafter. "
                f"Supported types: {list(self._drafter_classes.keys())}"
            )
        drafter_cls = self._drafter_classes[model_type]
        ad_logger.info(
            f"EagleDrafterFactory: model_type='{model_type}' -> drafter_cls={drafter_cls.__name__}"
        )

        # Convert base config to EagleConfig, preserving existing values
        # and applying model-specific defaults based on model_type
        model_config = EagleConfig(model_config, model_type)

        # Build the model (same pattern as parent's _build_model)
        with (init_empty_weights if device == "meta" else nullcontext)():
            model = drafter_cls._from_config(model_config, **unused_kwargs)

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


# =============================================================================
# EagleOneModel classes for combined target+draft model support
# =============================================================================
@ModelFactoryRegistry.register("eagle_one_model")
class EagleOneModelFactory(ModelFactory):
    """A factory that composes target and draft model factories for Eagle speculative decoding.

    This factory enables building a combined target+draft model as a single unit,
    which can be optimized together through the AutoDeploy transform pipeline.

    Args:
        model: The model path (used as identifier, typically the target model path).
        target_factory: The ModelFactory for the target (main) model.
        draft_factory: The ModelFactory for the draft model.
        speculative_config: The speculative decoding configuration (e.g., EagleDecodingConfig).
        resource_manager: The resource manager for managing hidden states and other resources.
        model_kwargs: Optional kwargs passed to the base ModelFactory.
        tokenizer: Optional tokenizer path (defaults to target model's tokenizer).
        tokenizer_kwargs: Optional kwargs for tokenizer initialization.
        skip_loading_weights: Whether to skip loading weights.
        max_seq_len: Maximum sequence length.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        model: str,
        target_factory: ModelFactory,
        draft_factory: ModelFactory,
        speculative_config: Any,
        resource_manager: Any,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        skip_loading_weights: bool = False,
        max_seq_len: int = 512,
        **kwargs,
    ):
        # Use target model's tokenizer by default if not specified
        effective_tokenizer = tokenizer or target_factory.tokenizer

        super().__init__(
            model=model,
            model_kwargs=model_kwargs,
            tokenizer=effective_tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            skip_loading_weights=skip_loading_weights,
            max_seq_len=max_seq_len,
            **kwargs,
        )

        self.target_factory = target_factory
        self.draft_factory = draft_factory
        self.speculative_config = speculative_config
        self.resource_manager = resource_manager

        # Sync skip_loading_weights with child factories
        self.target_factory.skip_loading_weights = skip_loading_weights
        self.draft_factory.skip_loading_weights = skip_loading_weights

    @classmethod
    def build_from_target(
        cls,
        target_factory: ModelFactory,
        speculative_config: Any,
        max_batch_size: int,
        max_num_tokens: int,
        draft_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "EagleOneModelFactory":
        """Build an EagleOneModelFactory from a target factory and speculative config.

        This convenience method creates the draft factory and resource manager automatically
        from the speculative_config and target factory, then combines them.

        Args:
            target_factory: The ModelFactory for the target (main) model.
            speculative_config: The speculative decoding configuration (e.g., EagleDecodingConfig).
                Must have `speculative_model` attribute pointing to the draft model path.
            max_batch_size: Maximum batch size (used as max_num_requests for resource manager).
            max_num_tokens: Maximum number of tokens.
            draft_model_kwargs: Optional kwargs to override the draft model's config (e.g., to
                reduce model dimensions for testing). Passed as ``model_kwargs`` to
                ``EagleDrafterFactory``.

        Returns:
            An EagleOneModelFactory configured with both target and draft factories.

        Example:
            >>> target_factory = AutoModelForCausalLMFactory(
            ...     model="meta-llama/Llama-3.1-8B-Instruct"
            ... )
            >>> eagle_factory = EagleOneModelFactory.build_from_target(
            ...     target_factory=target_factory,
            ...     speculative_config=ad_config.speculative_config,
            ...     max_batch_size=ad_config.max_batch_size,
            ...     max_num_tokens=ad_config.max_num_tokens,
            ... )
        """
        # Get draft model path from speculative config
        draft_model_path = speculative_config.speculative_model
        if draft_model_path is None:
            raise ValueError(
                "speculative_config.speculative_model must be set to the draft model path"
            )

        # Create draft factory using EagleDrafterFactory
        # It maps from base model types (e.g., "llama") to Eagle drafter implementations
        draft_factory = EagleDrafterFactory(
            model=str(draft_model_path),
            model_kwargs=draft_model_kwargs,
            tokenizer=target_factory.tokenizer,  # Use target's tokenizer
            skip_loading_weights=target_factory.skip_loading_weights,
            max_seq_len=target_factory.max_seq_len,
        )

        # Build the resource manager from the target factory
        resource_manager = ADHiddenStateManager.build_from_target_factory(
            target_factory=target_factory,
            config=speculative_config,
            max_num_requests=max_batch_size,
            max_num_tokens=max_num_tokens,
        )

        # Create and return the combined factory
        return cls(
            model=target_factory.model,
            target_factory=target_factory,
            draft_factory=draft_factory,
            speculative_config=speculative_config,
            resource_manager=resource_manager,
            tokenizer=target_factory.tokenizer,
            tokenizer_kwargs=target_factory.tokenizer_kwargs,
            skip_loading_weights=target_factory.skip_loading_weights,
            max_seq_len=target_factory.max_seq_len,
        )

    @property
    def vocab_size_padded(self) -> Optional[int]:
        """Return the padded vocabulary size from the target model factory."""
        return self.target_factory.vocab_size_padded

    @property
    def hidden_size(self) -> Optional[int]:
        """Return the hidden size from the target model factory."""
        return self.target_factory.hidden_size

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Return the dtype from the target model factory."""
        return self.target_factory.dtype

    def _build_model(self, device: str) -> nn.Module:
        """Build both target and draft models and return a combined EagleWrapper module.

        Args:
            device: The device to build the models on.

        Returns:
            An EagleWrapper containing both target and draft models.
        """
        # Build target model
        target_model = self.target_factory.build_model(device)

        # Build draft model
        draft_model = self.draft_factory.build_model(device)

        # Create EagleWrapper config from speculative_config and draft model config
        draft_config = draft_model.config
        wrapper_config = EagleWrapperConfig(
            max_draft_len=self.speculative_config.max_draft_len,
            load_embedding_from_target=getattr(draft_config, "load_embedding_from_target", True),
            load_lm_head_from_target=getattr(draft_config, "load_lm_head_from_target", True),
        )

        # Create combined model using EagleWrapper
        return EagleWrapper(
            config=wrapper_config,
            target_model=target_model,
            draft_model=draft_model,
            resource_manager=self.resource_manager,
        )

    def _load_checkpoint(
        self, model: nn.Module, device: DeviceLikeType, disable_preload: bool = False
    ):
        """Load checkpoints for both target and draft models.

        Args:
            model: The combined EagleWrapper.
            device: The device to load the models on.
            disable_preload: If True, disable preloading weights (passed to sub-factories).
        """
        if not isinstance(model, EagleWrapper):
            raise ValueError(f"Expected EagleWrapper, got {type(model)}")

        # Load target model weights
        self.target_factory._load_checkpoint(model.target_model, device, disable_preload)

        # Load draft model weights
        self.draft_factory._load_checkpoint(model.draft_model, device, disable_preload)

    def get_quant_config(self) -> Dict:
        """Returns the quantization config from the target factory."""
        return self.target_factory.get_quant_config()

    def get_sharding_config(self) -> Dict:
        """Returns the sharding config from the target model factory."""
        # Use target model's sharding config as the primary config
        return self.target_factory.get_sharding_config()

    def get_cache_config_updates(self) -> Dict[str, Any]:
        """Return the cache configuration updates from the target model factory."""
        return self.target_factory.get_cache_config_updates()

    def init_tokenizer(self) -> Optional[Any]:
        """Initialize the tokenizer from the target model factory."""
        return self.target_factory.init_tokenizer()

    def init_processor(self) -> Optional[Any]:
        """Initialize the processor from the target model factory."""
        return self.target_factory.init_processor()

    def prefetch_checkpoint(self, force: bool = False, skip_loading_weights: Optional[bool] = None):
        """Prefetch checkpoints for both target and draft models.

        Args:
            force: Whether to force prefetching.
            skip_loading_weights: Whether to skip loading weights.
        """
        # Prefetch for both factories
        self.target_factory.prefetch_checkpoint(force, skip_loading_weights)
        self.draft_factory.prefetch_checkpoint(force, skip_loading_weights)

        # Also call parent's prefetch for any model-level prefetching
        super().prefetch_checkpoint(force, skip_loading_weights)

    def get_example_inputs(self) -> Dict[str, torch.Tensor]:
        """Return example inputs including num_previously_accepted for EagleWrapper.

        The EagleWrapper.forward() requires num_previously_accepted to be provided.
        This tensor is passed via set_example_sequence's extra_args mechanism.

        Returns:
            A dictionary containing num_previously_accepted with shape [batch_size].
        """
        # Use batch_size >= 2 for export to prevent torch.export from specializing
        # the batch dimension when max_batch_size=1
        batch_size = 2
        # num_previously_accepted must be >= 1 (at least context tokens are accepted)
        # and should be reasonable for the example sequence length (4 by default)
        num_previously_accepted = torch.ones(batch_size, dtype=torch.long)
        return {"num_previously_accepted": num_previously_accepted}

    def get_export_infos(self, model: nn.Module) -> List[SubModuleExportInfo]:
        """Return export configurations for the target and draft models separately.

        Args:
            model: The combined EagleWrapper.

        Returns:
            A list of export configurations for both target and draft models.
        """
        draft_config = model.draft_model.config
        load_embedding_from_target = getattr(draft_config, "load_embedding_from_target", True)
        load_lm_head_from_target = getattr(draft_config, "load_lm_head_from_target", True)

        return [
            TargetModelExportInfo(load_lm_head_from_target),
            DraftModelExportInfo(load_embedding_from_target, load_lm_head_from_target),
        ]


class TargetModelExportInfo(SubModuleExportInfo):
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
        # Always save the embedding module
        embed_tokens = sub_mod.get_input_embeddings()
        sub_gm.get_input_embeddings = types.MethodType(
            sub_mod.get_input_embeddings.__func__, sub_gm
        )

        # retrieve+replicate expected submodule hierarchy for where the embedding module is located
        for embed_name, subsubmod in sub_mod.named_modules():
            if subsubmod is embed_tokens:
                break
        else:
            raise RuntimeError(
                "Could not find embedding module in model. Expected embedding module to be a "
                "submodule of the target model."
            )
        sub_gm.set_submodule(embed_name, embed_tokens)

        # Add a dummy node to the graph for making the embedding module impure.
        # Impure nodes won't be deleted from the graph during cleanup, ensuring the
        # embedding module is not garbage collected from the GraphModule.
        n_embed_tokens = sub_gm.graph.get_attr(f"{embed_name}.weight")
        sub_gm.graph.call_function(
            torch._assert, args=(n_embed_tokens, "Avoid embedding getting deleted from graph.")
        )

        # Save lm_head only if the draft model loads it from target
        if self.load_lm_head_from_target:
            lm_head = sub_mod.get_output_embeddings()
            sub_gm.get_output_embeddings = types.MethodType(
                sub_mod.get_output_embeddings.__func__, sub_gm
            )
            for lm_head_name, subsubmod in sub_mod.named_modules():
                if subsubmod is lm_head:
                    break
            else:
                raise RuntimeError(
                    "Could not find lm_head module in model. Expected lm_head module to be a "
                    "submodule of the target model."
                )
            sub_gm.set_submodule(lm_head_name, lm_head)

            # Add dummy node to prevent lm_head from being garbage collected
            n_lm_head = sub_gm.graph.get_attr(f"{lm_head_name}.weight")
            sub_gm.graph.call_function(
                torch._assert, args=(n_lm_head, "Avoid lm_head getting deleted from graph.")
            )


class DraftModelExportInfo(SubModuleExportInfo):
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
        """Keep draft model modules available in the graph module for EagleWrapper utilities.

        This preserves modules/parameters needed by EagleWrapper's utility methods:
        - embedding (model.embed_tokens): for apply_token_embedding if draft has its own
        - lm_head: for apply_lm_head
        - fc (model.fc): for apply_eagle3_fc
        - d2t (model.d2t): for apply_d2t
        - model.dtype: for dtype conversion in apply_eagle3_fc

        We also add dummy graph nodes (torch._assert) to make these modules "impure",
        preventing them from being garbage collected during graph cleanup.
        """
        inner_model_mod = sub_mod.model

        # Save embedding only if draft model has its own (not loading from target)
        if not self.load_embedding_from_target:
            sub_gm.set_submodule("model.embed_tokens", inner_model_mod.embed_tokens)
            # Save get_input_embeddings method for EagleWrapper utilities
            sub_gm.get_input_embeddings = types.MethodType(
                sub_mod.get_input_embeddings.__func__, sub_gm
            )
            # Add dummy node to prevent embedding from being garbage collected
            n_embed = sub_gm.graph.get_attr("model.embed_tokens.weight")
            sub_gm.graph.call_function(
                torch._assert, args=(n_embed, "Avoid draft embedding getting deleted from graph.")
            )

        # Save lm_head only if draft model has its own (not loading from target)
        if not self.load_lm_head_from_target:
            sub_gm.set_submodule("lm_head", sub_mod.lm_head)
            # Save get_output_embeddings method for EagleWrapper utilities
            sub_gm.get_output_embeddings = types.MethodType(
                sub_mod.get_output_embeddings.__func__, sub_gm
            )
            # Add dummy node to prevent lm_head from being garbage collected
            n_lm_head = sub_gm.graph.get_attr("lm_head.weight")
            sub_gm.graph.call_function(
                torch._assert, args=(n_lm_head, "Avoid draft lm_head getting deleted from graph.")
            )

        inner_model_gm = sub_gm.get_submodule("model")

        # Save fc module if it exists (for fusing hidden states from multiple layers)
        fc_module = getattr(inner_model_mod, "fc", None)
        if fc_module is not None:
            sub_gm.set_submodule("model.fc", fc_module)
            # Add dummy node to prevent fc from being garbage collected
            n_fc = sub_gm.graph.get_attr("model.fc.weight")
            sub_gm.graph.call_function(
                torch._assert, args=(n_fc, "Avoid fc getting deleted from graph.")
            )

        # Save d2t parameter and model dtype if they exist
        # d2t is a Parameter (not a Module), so needs special handling
        d2t = getattr(inner_model_mod, "d2t", None)
        model_dtype = getattr(inner_model_mod, "dtype", None)

        if d2t is not None:
            inner_model_gm.register_parameter("d2t", d2t)
            # Add dummy node to prevent d2t from being garbage collected
            n_d2t = sub_gm.graph.get_attr("model.d2t")
            sub_gm.graph.call_function(
                torch._assert, args=(n_d2t, "Avoid d2t getting deleted from graph.")
            )

        if model_dtype is not None:
            inner_model_gm.dtype = model_dtype
