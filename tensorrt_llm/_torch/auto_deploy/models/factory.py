"""The model factory interface used by auto-deploy to build custom models."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType

from ..utils.logger import ad_logger


class ModelFactory(ABC):
    """An interface to return and correctly initialize a model from a desired source.

    NOTE: we make the assumption that the model can be prompted with a set of input_ids of shape
    (batch_size, seq_len) and will return a tensor of shape (batch_size, seq_len, embedding_size).
    """

    def __init__(self, model: Optional[str] = None, skip_loading_weights: bool = False, **kwargs):
        self.model = model
        self._ckpt_path = None if skip_loading_weights else model
        self._prefetched_path: Optional[str] = None

    @property
    def ckpt_path(self):
        return self._prefetched_path or self._ckpt_path

    @abstractmethod
    def build_model(self, device: str) -> nn.Module:
        """Build the model on the desired device."""

    def get_quantization(self) -> Optional[Dict]:
        """Returns the quantization config for this model or None if not quantized."""
        return None

    def get_positional_encoding_config(self) -> Optional[Dict[str, Any]]:
        """Return the positional encoding configuration for the model.

        Returns:
            The positional encoding configuration for the model. If the model does not use positional
            encoding, then this method should return None.
        """
        return None

    def init_tokenizer(self) -> Optional[Any]:
        """Initialize the tokenizer for the model.

        Returns:
            The initialized tokenizer for the model. If the tokenizer is not available, then this
            method should return None.
        """
        return None

    def prefetch_checkpoint(self):
        """Try prefetching checkpoint."""
        pass

    def load_or_random_init(self, model: nn.Module, **kwargs):
        """Load the checkpoint into the model or randomly initialize the model.

        Args:
            model: The model to load the checkpoint into. Note that the model does not need to be
                the same model that is built above but it needs to have a state dict compatible with
                the model built above.
            **kwargs: Keyword arguments that will be passed to torch.load.
        """
        ad_logger.info("Loading and initializing weights.")
        if self.ckpt_path:
            self._load_checkpoint(model, **kwargs)
        else:
            self._load_random_init(model, **kwargs)

    @staticmethod
    def _to_maybe_empty(model: nn.Module, device: DeviceLikeType):
        """A mix of ``model.to(device)`` and ``model.to_empty(device)``.

        If a parameter is already initialized, then we will call `to()` on it. Otherwise, we will
        initialize it with an empty tensor on the given device.

        """
        model._apply(
            lambda t: torch.empty_like(t, device=device)
            if t.device == torch.device("meta")
            else t.to(device)
        )

    @classmethod
    def _load_random_init(cls, model: nn.Module, **kwargs):
        """Randomly initialize model."""
        cls._to_maybe_empty(model, kwargs.get("map_location"))
        state_dict = model.state_dict()
        for k in state_dict:
            state_dict[k] = torch.normal(
                0.0, 1.0, size=state_dict[k].shape, device=kwargs.get("map_location")
            ).to(state_dict[k].dtype)
        model.load_state_dict(state_dict, strict=True)

    @abstractmethod
    def _load_checkpoint(self, model: nn.Module, **kwargs):
        """Load the checkpoint into the model.

        Args:
            model: The model to load the checkpoint into. Note that the model does not need to be
                the same model that is built above but it needs to have a state dict compatible with
                the model built above.
            **kwargs: Keyword arguments that will be passed to torch.load.
        """


class ModelFactoryRegistry:
    _registry: Dict[str, Type[ModelFactory]] = {}

    @classmethod
    def register(
        cls: Type[ModelFactory], name: str
    ) -> Callable[[Type[ModelFactory]], Type[ModelFactory]]:
        def inner(fn: Type[ModelFactory]) -> Type[ModelFactory]:
            cls._registry[name] = fn
            return fn

        return inner

    @classmethod
    def get(cls, name: str) -> Type[ModelFactory]:
        assert cls.has(name), f"Model Factory {name} not found."
        return cls._registry[name]

    @classmethod
    def has(cls, model_factory_cls: str) -> bool:
        return model_factory_cls in cls._registry
