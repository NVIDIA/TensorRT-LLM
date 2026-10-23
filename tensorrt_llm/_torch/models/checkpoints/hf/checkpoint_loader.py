# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
    BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.checkpoint_catalog import \
    CheckpointCatalog
from tensorrt_llm._torch.models.checkpoints.hf.config_loader import \
    HfConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import \
    HfWeightLoader
from tensorrt_llm._torch.models.modeling_utils import register_checkpoint_loader
from tensorrt_llm.checkpoint_format import CheckpointFormat
from tensorrt_llm.llmapi.llm_args import CheckpointIOPolicy
from tensorrt_llm.mapping import Mapping

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig


@register_checkpoint_loader(CheckpointFormat.HF)
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
        self._checkpoint_format = CheckpointFormat.HF

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

    def _uses_context_only_lazy_loading(self, model: Any) -> bool:
        return (type(self) is HfCheckpointLoader
                and type(self.weight_loader) is HfWeightLoader
                and getattr(getattr(model, "model", None),
                            "disagg_context_only", False) is True)

    def load_weights(self, checkpoint_dir: str, mapping: Mapping,
                     **kwargs) -> dict[str, Any]:
        """Keep omitted decoder tensors lazy on a context-only worker."""
        if self._uses_context_only_lazy_loading(kwargs.get("model")):
            # The model remapper drops unowned keys before touching their
            # slices. Eager loading or read-ahead would read those bytes anyway.
            loader = self.weight_loader
            loader._reset_checkpoint_io_status()
            status = loader._last_checkpoint_io_status
            status.selected = CheckpointIOPolicy.NATIVE
            if loader.checkpoint_io_policy != CheckpointIOPolicy.NATIVE:
                status.fallback_reason = (
                    "context-only model loading requires lazy safetensors "
                    "without full-checkpoint read-ahead")
            weights = loader._load_lazy_safetensors(
                checkpoint_dir, kwargs.get("use_consolidated", False))
            status.effective = CheckpointIOPolicy.NATIVE
            loader._log_checkpoint_io_status()
            return weights
        return super().load_weights(checkpoint_dir, mapping=mapping, **kwargs)

    def uses_rank_striped_read_ahead(self, config: "ModelConfig") -> bool:
        """Whether loading reads checkpoint bytes in the background.

        Only the built-in HF path with a non-native I/O policy reads ahead,
        and the answer comes from the config alone, so it is known before the
        model is built. Weights of such a load must be copied into the model
        inside `open_weight_session`.

        Args:
            config: validated model config of this load
        """
        # The weight loader holds the selected policy, so AUTO is already
        # resolved here.
        if not (type(self) is HfCheckpointLoader and type(self.weight_loader)
                is HfWeightLoader and self.weight_loader.checkpoint_io_policy
                == CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD):
            return False

        # A DeepSeek-V4.1 context-only disaggregated worker keeps the decoder
        # tensors it does not own lazy (see `_uses_context_only_lazy_loading`),
        # so reading the whole checkpoint ahead would read bytes it skips.
        # This repeats the test DeepseekV41Model uses to set
        # `disagg_context_only`. If more models like this appear, consider a
        # model-level attribute (readable before the model is built) as the
        # switch instead of per-model tests here.
        from tensorrt_llm._torch.configs.deepseek_v41 import \
            disagg_context_decoder_skipping_enabled
        architectures = config.pretrained_config.architectures or []
        if (architectures[:1] == ["DeepseekV41ForCausalLM"]
                and disagg_context_decoder_skipping_enabled(
                    config.extra_attrs.get("bounded_replay_on_generation",
                                           False))):
            return False
        return True

    @contextmanager
    def open_weight_session(self, checkpoint_dir: str, mapping: Mapping,
                            config: "ModelConfig",
                            **kwargs) -> Iterator[dict[str, Any]]:
        """Hand out checkpoint weights for one with-block, reading ahead if enabled.

        `uses_rank_striped_read_ahead` picks the pass: rank-striped read-ahead
        or a single native read. Both passes record how long the session took
        to start and to finish in ``weight_loader.last_checkpoint_io_status``.

        Args:
            checkpoint_dir: checkpoint to read
            mapping: parallel mapping of this rank
            config: validated model config of this load; the target config
                also for the separate draft checkpoint
            **kwargs: passed on to the read, e.g. ``model`` for the target
        """
        if self.uses_rank_striped_read_ahead(config):
            # The weight loader's session records its own timings.
            with self.weight_loader.open_weight_session(checkpoint_dir,
                                                        mapping=mapping,
                                                        **kwargs) as weights:
                yield weights
            return

        # native pass: read the checkpoint once. MX, Mistral, and custom
        # subclasses retain their polymorphic load_weights implementations.
        preparation_started = time.perf_counter()
        weights = self.load_weights(checkpoint_dir, mapping=mapping, **kwargs)
        preparation_seconds = time.perf_counter() - preparation_started
        # Record the timings where the read-ahead pass records them. MX P2P
        # does not reset this status, so also clear the finalization time.
        if isinstance(self.weight_loader, HfWeightLoader):
            io_status = self.weight_loader._last_checkpoint_io_status
            io_status.preparation_seconds = preparation_seconds
            io_status.finalization_seconds = 0.0
        yield weights

    def get_default_config_loader(self) -> HfConfigLoader:
        return HfConfigLoader()

    def build_checkpoint_catalog(self, checkpoint_dir: str,
                                 **kwargs) -> CheckpointCatalog | None:
        """Inspect metadata only for the exact built-in eager HF loader path."""
        if (type(self) is not HfCheckpointLoader
                or type(self.weight_loader) is not HfWeightLoader):
            return None
        return self.weight_loader.build_checkpoint_catalog(
            checkpoint_dir,
            use_consolidated=kwargs.get("use_consolidated", False),
        )

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
    def checkpoint_format(self) -> CheckpointFormat:
        return self._checkpoint_format
