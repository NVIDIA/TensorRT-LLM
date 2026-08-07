# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
    BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    BaseWeightLoader
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

    @contextmanager
    def open_weight_session(self, checkpoint_dir: str, mapping: Mapping,
                            **kwargs) -> Iterator[dict[str, Any]]:
        """Keep native HF read-ahead alive during materialization.

        MX and Mistral subclass this loader and override ``load_weights``.
        Delegate those and custom weight loaders through the polymorphic base
        session instead of bypassing their format-specific behavior.
        """
        if (type(self) is not HfCheckpointLoader
                or type(self.weight_loader) is not HfWeightLoader):
            with super().open_weight_session(checkpoint_dir,
                                             mapping=mapping,
                                             **kwargs) as weights:
                yield weights
            return

        with self.weight_loader.open_weight_session(checkpoint_dir,
                                                    mapping=mapping,
                                                    **kwargs) as weights:
            yield weights

    def coordinate_checkpoint_io_request(self, mapping: Mapping) -> None:
        """Reject rank-divergent native-HF I/O policy requests."""
        if (type(self) is HfCheckpointLoader
                and type(self.weight_loader) is HfWeightLoader):
            self.weight_loader.coordinate_checkpoint_io_request(mapping)

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

    @property
    def config_loader(self) -> Optional[BaseConfigLoader]:
        return self._config_loader

    @property
    def checkpoint_format(self) -> str:
        return self._checkpoint_format
