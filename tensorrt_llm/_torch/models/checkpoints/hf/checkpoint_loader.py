# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
    BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    BaseWeightLoader, WeightBatchStream)
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.config_loader import \
    HfConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import \
    HfWeightLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader
from tensorrt_llm.mapping import Mapping


@register_checkpoint_loader("HF")
class HfCheckpointLoader(BaseCheckpointLoader):

    def __init__(self,
                 *,
                 weight_loader: Optional[BaseWeightLoader] = None,
                 weight_mapper: Optional[BaseWeightMapper] = None,
                 config_loader: Optional[BaseConfigLoader] = None):
        self._uses_custom_weight_mapper = weight_mapper is not None
        if weight_loader is None:
            self._weight_loader = self.get_default_weight_loader()
        else:
            self._weight_loader = weight_loader
        if config_loader is None:
            self._config_loader = self.get_default_config_loader()
        else:
            self._config_loader = config_loader
        self._weight_mapper = weight_mapper
        self._checkpoint_format = "HF"

    def _prepare_weight_loader_kwargs(self,
                                      kwargs: dict[str, Any]) -> dict[str, Any]:
        """Add private policy metadata only for the native HF loader."""
        kwargs = dict(kwargs)
        # Preserve compatibility with user-injected weight loaders whose
        # implementations predate BaseWeightLoader's ``**kwargs`` contract.
        if isinstance(self.weight_loader, HfWeightLoader):
            kwargs.setdefault("_checkpoint_format", self.checkpoint_format)
            kwargs.setdefault("_uses_custom_weight_mapper",
                              self._uses_custom_weight_mapper)
            if self.checkpoint_format == "HF":
                # Direct HF checkpoint-loader calls are the standard disk path.
                # ModelLoader supplies GMS explicitly before reaching here.
                kwargs.setdefault("_load_format", "AUTO")
        else:
            # Do not leak private HF policy hints into strict third-party
            # loaders that may not yet accept arbitrary keyword arguments.
            kwargs.pop("_checkpoint_format", None)
            kwargs.pop("_uses_custom_weight_mapper", None)
            kwargs.pop("_load_format", None)
            kwargs.pop("_weight_mapper", None)
            kwargs.pop("_model_supports_partial_loading", None)
        return kwargs

    def load_weights(self, checkpoint_dir: str, mapping: Mapping,
                     **kwargs) -> dict[str, Any]:
        """Load weights with source metadata used by optional HF policies."""
        kwargs = self._prepare_weight_loader_kwargs(kwargs)
        return super().load_weights(checkpoint_dir, mapping=mapping, **kwargs)

    @contextmanager
    def open_weight_session(
            self, checkpoint_dir: str, mapping: Mapping,
            **kwargs) -> Iterator[dict[str, Any]
                                  | WeightBatchStream]:
        """Keep native HF read-ahead active during model materialization."""
        if (self.checkpoint_format != "HF"
                or not isinstance(self.weight_loader, HfWeightLoader) or
                type(self).load_weights is not HfCheckpointLoader.load_weights
                or type(self.weight_loader).load_weights
                is not HfWeightLoader.load_weights):
            # MX and Mistral override load_weights(). The base implementation
            # deliberately calls that polymorphic method rather than bypassing
            # P2P transfer or format-specific preprocessing.
            with super().open_weight_session(checkpoint_dir,
                                             mapping=mapping,
                                             **kwargs) as weights:
                yield weights
            return

        kwargs = self._prepare_weight_loader_kwargs(kwargs)
        with self.weight_loader.open_weight_session(checkpoint_dir,
                                                    mapping=mapping,
                                                    **kwargs) as weights:
            yield weights

    def get_initialized_weight_mapper(self, model, config) -> BaseWeightMapper:
        auto_select_mapper = self.weight_mapper is None
        weight_mapper = super().get_initialized_weight_mapper(model, config)
        if auto_select_mapper:
            # BaseCheckpointLoader assigns the registry-selected mapper through
            # our property setter. It is a supported default, not user input.
            self._uses_custom_weight_mapper = False
        return weight_mapper

    def requires_initialized_mapper_for_session(self) -> bool:
        """Whether native HF policy preflight needs an initialized mapper."""
        return (self.checkpoint_format == "HF"
                and isinstance(self.weight_loader, HfWeightLoader)
                and type(self).load_weights is HfCheckpointLoader.load_weights
                and type(self.weight_loader).load_weights
                is HfWeightLoader.load_weights and
                self.weight_loader.requires_initialized_mapper_for_session())

    def cleanup(self) -> None:
        # Clean up weight mapper first as it may hold model references
        if self._weight_mapper is not None:
            self._weight_mapper.cleanup()
            self._weight_mapper = None

        if self._weight_loader is not None:
            self._weight_loader.cleanup()
            self._weight_loader = None

        if self._config_loader is not None:
            self._config_loader.cleanup()
            self._config_loader = None

    def get_default_weight_loader(self) -> HfWeightLoader:
        return HfWeightLoader()

    def get_default_config_loader(self) -> HfConfigLoader:
        return HfConfigLoader()

    @property
    def weight_loader(self) -> BaseWeightLoader:
        return self._weight_loader

    @property
    def weight_mapper(self) -> Optional[BaseWeightMapper]:
        return self._weight_mapper

    @weight_mapper.setter
    def weight_mapper(self, value: BaseWeightMapper) -> None:
        self._weight_mapper = value
        self._uses_custom_weight_mapper = value is not None

    @property
    def config_loader(self) -> Optional[BaseConfigLoader]:
        return self._config_loader

    @property
    def checkpoint_format(self) -> str:
        return self._checkpoint_format
