"""The model factory interface used by auto-deploy to build custom models."""

import copy
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType

from ..custom_ops.attention_interface import CacheConfig, DynamicShapeCallback
from ..utils.logger import ad_logger


class ShardingConfigSource(Enum):
    """Enum for factory source."""

    HUGGINGFACE = "huggingface"
    UNKNOWN = "unknown"


class ModelFactory(ABC):
    """An interface to return and correctly initialize a model from a desired source.

    NOTE: we make the assumption that the model can be prompted with a set of input_ids and
    position_ids of shape (batch_size, seq_len) and will return a tensor of shape
    (batch_size, seq_len, embedding_size) by default. Individual factories have the ability to
    define additional optional inputs and their (dynamic) shapes.
    """

    def __init__(
        self,
        model: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        skip_loading_weights: bool = False,
        max_seq_len: int = 512,
        **kwargs,
    ):
        self._model = model
        self.model_kwargs = copy.deepcopy(model_kwargs or {})
        self._tokenizer = tokenizer
        self.tokenizer_kwargs = copy.deepcopy(tokenizer_kwargs or {})
        self.skip_loading_weights = skip_loading_weights
        self.max_seq_len = max_seq_len
        self._prefetched_model_path: Optional[str] = None
        self._prefetched_tokenizer_path: Optional[str] = None
        self._sharding_config: Dict[str, Any] = {}
        self._sharding_config["source"] = ShardingConfigSource.UNKNOWN

    @property
    def model(self) -> Optional[str]:
        """The model+checkpoint path."""
        return self._prefetched_model_path or self._model

    @property
    def tokenizer(self) -> Optional[str]:
        """The tokenizer path."""
        return self._prefetched_tokenizer_path or self._tokenizer or self.model

    def build_model(self, device: str) -> nn.Module:
        """Build the model on the desired device.

        Args:
            device: The device to build the model on.

        Returns:
            The built model.


        Note that we assume that the model's forward function has the following signature:

        .. code-block:: python

            def forward(
                self, input_ids: torch.Tensor, position_ids: torch.Tensor, *extra_args: torch.Tensor
            ) -> Sequence[torch.Tensor]: ...

        ``logits`` are assumed to be the first output of the model, i.e.,
        ``model(input_ids, position_ids)[0]`` should return a ``logits`` tensor.

        Moreover, we assume the following tensor shapes:

        .. code-block:: python

            input_ids.shape == (batch_size, seq_len)
            position_ids.shape == (batch_size, seq_len)
            logits.shape == (batch_size, seq_len, vocab_size)

        We allow for additional arguments to be passed to the model's forward function as defined by
        the factory.
        """
        # make sure model architecture is pre-fetched (no weights needed at this point)
        skip_loading_weights = self.skip_loading_weights
        self.skip_loading_weights = True
        self.prefetch_checkpoint()
        self.skip_loading_weights = skip_loading_weights

        # build the model
        return self._build_model(device)

    @abstractmethod
    def _build_model(self, device: str) -> nn.Module:
        """Factory-specific model building logic."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_quant_config(self) -> Dict:
        """Returns the quantization config for this model or None if not quantized."""
        return {}

    def get_sharding_config(self) -> Dict:
        """Returns the sharding config for this model."""
        return self._sharding_config

    def get_cache_config(self) -> CacheConfig:
        """Return the cache configuration for the model.

        Returns:
            The cache configuration for the model.
        """
        return CacheConfig()

    def init_tokenizer(self) -> Optional[Any]:
        """Initialize the tokenizer for the model.

        Returns:
            The initialized tokenizer for the model. If the tokenizer is not available, then this
            method should return None.
        """
        return None

    def init_processor(self) -> Optional[Any]:
        """Initialize the (multi-modal) processor for the model.

        Returns:
            The initialized processor for the model. If the processor is not available, then this
            method should return None.
        """
        return None

    def prefetch_checkpoint(self, force: bool = False):
        """Try or skip prefetching the checkpoint for the model and tokenizer.

        Args:
            force: Whether to force prefetching the checkpoint.
        """
        if not self._prefetched_model_path or force:
            self._prefetched_model_path = self._prefetch_checkpoint(
                self._model, self.skip_loading_weights
            )
        if self._tokenizer and (not self._prefetched_tokenizer_path or force):
            self._prefetched_tokenizer_path = self._prefetch_checkpoint(self._tokenizer, True)

    def _prefetch_checkpoint(self, model_name_or_path: str, skip_prefetch_weights: bool) -> str:
        """Optionally prefetch the checkpoint if factory supports it.

        Args:
            model_name_or_path: The model or tokenizer name or path.
            skip_prefetch_weights: Whether to skip prefetching weights.

        Returns:
            The prefetched checkpoint path.
        """
        return model_name_or_path

    def load_or_random_init(self, model: nn.Module, device: DeviceLikeType):
        """Load the checkpoint into the model or randomly initialize the model.

        Args:
            model: The model to load the checkpoint into. Note that the model does not need to be
                the same model that is built above but it needs to have a state dict compatible with
                the model built above.
            device: The device to load the model on.
            load_factoy_model: If True, will load weights for the factory model in addition to main
                gm. This is useful for the transformers model.

        NOTE: we always call ``self._to_maybe_random(model, device)`` as a preprocessing step
        to ensure the model parameters already exist on the right device and have the desired dtype
        as set in the model architecture. Moreover, initializing weights will ensure that no
        ``assign=True`` logic is triggered during state dict loading. We want to avoid this since
        this can interfere with things like weight sharding loading hooks. Particularly,
        ``assign=True`` can cause OOM issues because although we shard the weight it may still
        reference the full weight tensor in memory and hence with ``assign=True`` memory equivalent
        to the full weight tensor and hence the full model may be reserved on each rank.

        NOTE: this function will roughly allocate the following amount of memory:

            * ``skip_loading_weights=True``:

                .. code-block:: python

                    sum(t.element_size() * t.numel() for t in model.state_dict().values())

            * ``skip_loading_weights=False``:

                .. code-block:: python

                    sum(t.element_size() * t.numel() for t in model.state_dict().values()) +
                    <SIZE_OF_LARGEST_CHECKPOINT_FILE>

        """
        ad_logger.info("Loading and initializing weights.")
        self._to_maybe_random(model, device)
        if not self.skip_loading_weights:
            self.prefetch_checkpoint(force=True)
            self._load_checkpoint(model, device)

    @staticmethod
    def _to_maybe_random(model: nn.Module, device: DeviceLikeType):
        """A mix of ``model.to(device)`` and random initialization of parameters.

        If a parameter is already initialized, then we will call `to()` on it. Otherwise, we will
        initialize it with a random tensor on the given device.

        NOTE: this utility is written in such a fashion that not more memory than what the model
        shard needs is reserved and/or allocated.
        """
        model._apply(
            # NOTE (lucaslie): torch.normal is not supported for all dtypes
            lambda t: torch.normal(0.0, 1.0, size=t.shape, device=device).to(t.dtype)
            if t.device == torch.device("meta")
            else t.to(device)
        )

    @abstractmethod
    def _load_checkpoint(self, model: nn.Module, device: DeviceLikeType):
        """Load the checkpoint into the model.

        Args:
            model: The model to load the checkpoint into. Note that the model does not need to be
                the same model that is built above but it needs to have a state dict compatible with
                the model built above.
            device: The device to load the model on.
        """

    def get_example_inputs(self) -> Dict[str, torch.Tensor]:
        """Return a dictionary of example inputs for the model.

        This function can be overwritten by a factory when it requires a specific example input to
        in order to run through export.

        Returns:
            A dictionary of example inputs for the model where the key corresponds to the argument
            name and the value corresponds to the example input.
        """
        return {}

    def get_extra_inputs(self) -> Dict[str, Tuple[torch.Tensor, Optional[DynamicShapeCallback]]]:
        """Return a dictionary of extra model inputs that behave like optional forward arguments.

        Returns:
            A dictionary of extra inputs for the model where the key corresponds to the argument
            name and the value corresponds to a tuple of (none_input, dynamic_shape_callback):
                - `none_input`: The none input value of the extra input indicating the tensor
                   value corresponding to the equivalent of the None input. `None` is not supported
                   as we require the input to be a tensor. Hence, this none_input acts as a
                   placeholder for the None input. We assume that the "optional" behavior of these
                   arguments can be represented via a placeholder tensor and and an appropriate
                   check within the forward function using ``torch.cond``.
                - `dynamic_shape_callback`: A function that returns the dynamic shape of the extra
                  input. Simply set to `None` if the extra input is not dynamic.
        """
        return {}


class ModelFactoryRegistry:
    _registry: Dict[str, Type[ModelFactory]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[ModelFactory]], Type[ModelFactory]]:
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
