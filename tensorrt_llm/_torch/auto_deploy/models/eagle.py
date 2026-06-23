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

import re
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch.nn as nn
from accelerate import init_empty_weights
from torch._prims_common import DeviceLikeType
from torch.export import Dim
from torch.fx import GraphModule

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

from ..utils.logger import ad_logger
from .custom.modeling_eagle import (
    EagleConfig,
    EagleDrafterForCausalLM,
    EagleWrapper,
    EagleWrapperConfig,
)
from .factory import DynamicShape, ModelFactory, ModelFactoryRegistry, SubModuleExportInfo
from .hf import (
    AutoModelForCausalLMFactory,
    AutoModelForImageTextToTextFactory,
    expose_graph_module_accessor,
    insert_keepalive_sentinel,
)


def mapped_module_names(
    modules: List[str],
    renamed_modules: Optional[Dict[str, str]],
) -> List[str]:
    if not renamed_modules:
        return modules

    mapped_modules = []
    for module in modules:
        mapped_module = module
        for regex, replacement in renamed_modules.items():
            mapped_module = re.sub(regex, replacement, mapped_module)
        mapped_modules.append(mapped_module)
    return mapped_modules


@ModelFactoryRegistry.register("EagleDrafter")
class EagleDrafterFactory(AutoModelForCausalLMFactory):
    """Factory for building Eagle drafter models.

    The drafter builds its own model-specific layers internally based on
    config.model_type, allowing it to work with different base models
    (Llama, NemotronH, etc.) without the factory needing to know the details.

    The checkpoint config is expected to have the base model's model_type
    (e.g., "llama") along with Eagle-specific fields like draft_vocab_size.
    """

    def __init__(self, *args, use_inner_text_config: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_inner_text_config = use_inner_text_config

    def _get_model_config(self):
        """Return the drafter config, unwrapping VLM configs to their inner text config.

        AutoDeploy VLM configs expose the inner language-model config as ``text_config``.
        Eagle/MTP drafters run against that text model rather than the outer multimodal wrapper.
        """
        model_config, unused_kwargs = super()._get_model_config()
        text_config = getattr(model_config, "text_config", None)
        if self.use_inner_text_config and text_config is not None:
            ad_logger.info(
                f"EagleDrafterFactory: extracting text_config from multimodal config "
                f"(outer model_type='{model_config.model_type}')"
            )
            model_config = text_config
        return model_config, unused_kwargs

    def _build_model(self, device: DeviceLikeType) -> nn.Module:
        model_config, unused_kwargs = self._get_model_config()

        # Get model type for config
        model_type = model_config.model_type
        ad_logger.info(f"EagleDrafterFactory: building drafter for model_type='{model_type}'")

        # Convert base config to EagleConfig, preserving existing values
        # and applying model-specific defaults based on model_type
        model_config = EagleConfig.from_base_config(model_config, model_type)

        with (init_empty_weights if device == "meta" else nullcontext)():
            model = EagleDrafterForCausalLM._from_config(model_config, **unused_kwargs)

        if device == "meta":
            # post-init must be called explicitly for HF models with init_empty_weights
            if hasattr(model, "post_init"):
                model.post_init()
        else:
            model.to(device)

        # Store module-name conversion mappings if present.
        self._checkpoint_conversion_mapping = getattr(model, "_checkpoint_conversion_mapping", None)
        self._quant_exclude_conversion_mapping = getattr(
            model, "_quant_exclude_conversion_mapping", None
        )

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

    def __init__(
        self,
        load_lm_head_from_target: bool,
        target_export_info: SubModuleExportInfo,
        submodule_name: str = "target_model",
    ):
        super().__init__(submodule_name)
        self.load_lm_head_from_target = load_lm_head_from_target
        # Always populated -- the target factory exports exactly one submodule (a VLM text root, or
        # FullModelExportInfo with submodule_name "" for a full-model target). See
        # EagleOneModelFactory._single_target_export_info.
        self.target_export_info = target_export_info

    def _init_dynamic_shape_lookup(self) -> Dict[str, DynamicShape]:
        batch_size_dyn = Dim.DYNAMIC
        seq_len_dyn = Dim.DYNAMIC
        dynamic_shape_lookup = {
            "inputs_embeds": {0: batch_size_dyn, 1: seq_len_dyn},
            "position_ids": {0: batch_size_dyn, 1: seq_len_dyn},
        }
        # The Eagle wrapper forwards target inputs through a different module
        # path, but export still needs the target factory's tensor shape
        # contract, such as 3D mRoPE position_ids.
        for key, dynamic_shape in self.target_export_info.dynamic_shape_lookup.items():
            if key in dynamic_shape_lookup:
                dynamic_shape_lookup[key] = dynamic_shape
        return dynamic_shape_lookup

    def post_process(self, sub_mod: nn.Module, sub_gm: GraphModule):
        """Preserve target modules needed by the Eagle wrapper on the exported GraphModule."""
        # --- Embedding: always needed (target embeds input_ids for both target and draft) ---
        # Mirror the target factory's own export post-processing first. For VLMs, this is the same
        # hook used by TextModelExportInfo to keep language-model accessors available on the
        # exported text GraphModule. That hook already exposes get_input_embeddings, so the call
        # below re-exposes it for VLM targets -- intentional and harmless (only a redundant, no-op
        # keepalive sentinel). For a full-model target this is FullModelExportInfo.post_process, a
        # no-op.
        self.target_export_info.post_process(sub_mod, sub_gm)
        expose_graph_module_accessor(
            sub_mod,
            sub_gm,
            "get_input_embeddings",
            "Could not find embedding module in target model.",
        )

        # --- lm_head: only if draft model loads it from target ---
        if self.load_lm_head_from_target:
            expose_graph_module_accessor(
                sub_mod,
                sub_gm,
                "get_output_embeddings",
                "Could not find lm_head module in target model.",
            )

        # --- Final normalization: only if target model exposes it (e.g., NemotronH for MTP) ---
        if hasattr(sub_mod, "get_final_normalization"):
            expose_graph_module_accessor(
                sub_mod,
                sub_gm,
                "get_final_normalization",
                "Could not find final normalization module in target model.",
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
        inner_model = sub_mod.eagle_drafter
        inner_gm = sub_gm.get_submodule("eagle_drafter")

        # mark this gm as the draft model gm
        sub_gm.is_draft = True

        # --- Embedding (only if draft model has its own) ---
        if not self.load_embedding_from_target:
            expose_graph_module_accessor(
                sub_mod,
                sub_gm,
                "get_input_embeddings",
                "Could not find embedding module in draft model.",
            )

        # --- lm_head (only if draft model has its own) ---
        if not self.load_lm_head_from_target:
            expose_graph_module_accessor(
                sub_mod,
                sub_gm,
                "get_output_embeddings",
                "Could not find lm_head module in draft model.",
            )

        # --- fc module (fuses hidden states from multiple layers) ---
        fc_module = getattr(inner_model, "fc", None)
        if fc_module is not None:
            sub_gm.set_submodule("eagle_drafter.fc", fc_module)
            insert_keepalive_sentinel(sub_gm, "eagle_drafter.fc.weight")

        # --- d2t parameter (draft-to-target vocab mapping) ---
        d2t = getattr(inner_model, "d2t", None)
        if d2t is not None:
            inner_gm.register_parameter("d2t", d2t)
            insert_keepalive_sentinel(sub_gm, "eagle_drafter.d2t")

        # --- model dtype (used by apply_eagle3_fc) ---
        model_dtype = getattr(inner_model, "dtype", None)
        if model_dtype is not None:
            inner_gm.dtype = model_dtype


@ModelFactoryRegistry.register("eagle_one_model")
class EagleOneModelFactory(ModelFactory):
    """Factory that composes target + draft factories for one-model Eagle speculative decoding.

    Creates a target factory from ``target_model_factory`` and an ``EagleDrafterFactory``
    internally from the kwargs passed via ``ModelFactoryRegistry``.

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
        target_model_factory: str = "AutoModelForCausalLM",
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
        self.sync_before_hidden_state_capture = kwargs.get(
            "sync_before_hidden_state_capture", False
        )
        # For MTP, derive Eagle-pipeline fields from MTP-specific fields.
        if isinstance(speculative_config, MTPDecodingConfig):
            draft_model_path = speculative_config.speculative_model or model
        else:
            draft_model_path = speculative_config.speculative_model
        if draft_model_path is None:
            raise ValueError("speculative_config.speculative_model must be set.")

        # Create target factory using the configured factory class.
        target_factory_cls = ModelFactoryRegistry.get(target_model_factory)
        use_inner_text_config = issubclass(target_factory_cls, AutoModelForImageTextToTextFactory)
        self.target_factory = target_factory_cls(
            model=model,
            model_kwargs=model_kwargs,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            skip_loading_weights=skip_loading_weights,
            max_seq_len=max_seq_len,
        )

        # Export submodule path of the target text model (e.g. "model.language_model"),
        # populated in _build_model. Used by get_quant_config to translate checkpoint-namespace
        # exclude_modules into the exported (relative) graph namespace.
        self._target_export_submodule_name = ""

        # Create draft factory (EagleDrafter).
        self.draft_factory = EagleDrafterFactory(
            model=str(draft_model_path),
            model_kwargs=speculative_model_kwargs,
            tokenizer=tokenizer,
            skip_loading_weights=skip_loading_weights,
            max_seq_len=max_seq_len,
            use_inner_text_config=use_inner_text_config,
        )

    @property
    def max_seq_len(self) -> int:
        return self.draft_factory.max_seq_len

    @property
    def vocab_size_padded(self) -> Optional[int]:
        return self.target_factory.vocab_size_padded

    def _single_target_export_info(self, target_model: nn.Module) -> SubModuleExportInfo:
        """Return the target factory's single export info (never None).

        The Eagle one-model path assumes the target exports exactly one submodule -- a VLM text
        model rooted at e.g. "model.language_model", or a full-model export (FullModelExportInfo,
        submodule_name ""). Real target factories always return exactly one. Supporting multiple
        exported submodules would require threading several export roots through
        get_quant_config / get_export_infos; we deliberately don't (the goal is VLM targets, not
        arbitrary nesting) and assert loudly rather than silently using only the first. Asserting
        exactly 1 (vs <= 1) lets every caller treat the result as populated.
        """
        infos = self.target_factory.get_export_infos(target_model)
        assert len(infos) == 1, (
            "EagleOneModelFactory expects the target factory to export exactly one submodule "
            f"(a full-model export uses submodule_name ''), got {len(infos)}: "
            f"{[i.submodule_name for i in infos]}"
        )
        return infos[0]

    def _build_model(self, device: str) -> nn.Module:
        target_model = self.target_factory.build_model(device)
        # Record the target's export submodule path using the same get_export_infos call that
        # export_to_gm relies on, so the recorded prefix can never drift from the actual export
        # root. For a VLM target this is "model.language_model"; for a full-model export it is "".
        target_export_info = self._single_target_export_info(target_model)
        # Always populated: "model.language_model" for a VLM target, "" for a full-model export.
        self._target_export_submodule_name = target_export_info.submodule_name
        draft_model = self.draft_factory.build_model(device)

        draft_config = draft_model.config
        wrapper_config = EagleWrapperConfig(
            max_draft_len=self.speculative_config.max_draft_len,
            load_embedding_from_target=getattr(draft_config, "load_embedding_from_target", True),
            load_lm_head_from_target=getattr(draft_config, "load_lm_head_from_target", True),
            normalize_target_hidden_state=getattr(
                draft_config, "normalize_target_hidden_state", False
            ),
            sync_before_hidden_state_capture=self.sync_before_hidden_state_capture,
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
        target_export_info = self._single_target_export_info(model.target_model)
        target_submodule_name = "target_model"
        if target_export_info.submodule_name:
            # VLM target: export_to_gm uses this path to select and replace the exported submodule,
            # so preserve inner target paths under the wrapper. (Empty for a full-model target.)
            target_submodule_name = f"target_model.{target_export_info.submodule_name}"

        return [
            TargetModelExportInfo(
                load_lm_head_from_target,
                submodule_name=target_submodule_name,
                target_export_info=target_export_info,
            ),
            DraftModelExportInfo(load_embedding_from_target, load_lm_head_from_target),
        ]

    def get_sharding_config(self) -> Dict[str, Any]:
        return self.target_factory.get_sharding_config()

    def get_quant_config(self) -> Dict[str, Any]:
        """Return one-model quantization config for target and draft subgraphs.

        One-model Eagle/MTP checkpoints can carry one quant config for both the target model and
        the in-checkpoint draft head. The target graph may be exported from an inner VLM text module,
        so target exclude patterns need to be stripped from the checkpoint namespace into the
        exported target graph namespace. Draft exclude patterns may need the draft factory's rename
        mapping because the MTP head is loaded into the dedicated ``eagle_drafter.*`` namespace. The
        two rewrites are intentionally separate so a draft remap cannot corrupt target excludes, and
        so eager VLM wrappers still use the target factory's export root as the namespace boundary.
        """
        # Start from the target model's quant config (checkpoint-namespace exclude_modules).
        qcfg = dict(self.target_factory.get_quant_config())
        excluded = qcfg.get("exclude_modules")
        if excluded:
            prefix = self._target_export_submodule_name
            prefix_dot = prefix + "." if prefix else None
            draft_map = getattr(self.draft_factory, "_quant_exclude_conversion_mapping", None)

            # The draft model submodule name(s) the remap rewrites into, e.g. ("eagle_drafter.",).
            draft_namespaces = tuple(f"{v.split('.', 1)[0]}." for v in (draft_map or {}).values())

            new_excluded = []
            for p in excluded:
                if prefix_dot and p.startswith(prefix_dot):
                    stripped = p[len(prefix_dot) :]
                    # Core no-collision check. The draft's remapped excludes live under a dedicated
                    # namespace ("eagle_drafter.*") that no target model uses, which is the entire
                    # reason a draft exclude can never match a target graph node. The one way that
                    # breaks is a target whose own names, after stripping, land in the draft's
                    # reserved namespace -- guard against exactly that. Tripping this does not mean
                    # the model is wrong; it means the disjoint-namespace assumption no longer holds
                    # and this exclude-splitting logic must be revisited.
                    assert not stripped.startswith(draft_namespaces), (
                        f"target exclude {p!r} strips to {stripped!r}, which lands in the draft's "
                        f"reserved namespace {draft_namespaces}: target and draft exclude namespaces "
                        "must stay disjoint -- revisit EagleOneModelFactory.get_quant_config"
                    )
                    new_excluded.append(stripped)
                else:
                    new_excluded.extend(mapped_module_names([p], draft_map))
            qcfg["exclude_modules"] = new_excluded
        return qcfg

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
