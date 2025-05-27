from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, List, Union

from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig, TConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM


def guard_all_methods(cls):
    """
    This decorator is used to guard all methods of a class.
    It checks if the class is initialized before calling the method.
    If the class is not initialized, it raises a RuntimeError.
    It is needed because we wanted to allow 'shallow' initialization of the class,
    i.e. without calling .init() method, taking into account API design.
    """

    def wrap(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, "is_initialized",
                           False) and func.__name__ != "init":
                raise RuntimeError(
                    f"{self.__class__.__name__}.{func.__name__}() "
                    "requires .init() first")
            return func(self, *args, **kwargs)

        return wrapper

    for name, attr in list(vars(cls).items()):
        if callable(attr) and not name.startswith("__"):
            setattr(cls, name, wrap(attr))
    return cls


@guard_all_methods
class WeightMapperInterface(ABC):

    def __init__(self):
        self._callbacks: list[Callable] = []
        self._mapping: dict = {}
        self._skip_modules = []

        self.is_initialized = False

    def init(self, model: Union[nn.Module, DecoderModelForCausalLM],
             config: TConfig):
        self._model = model
        self._config = config

        if not hasattr(model, 'model_config') or not isinstance(
                model.model_config, ModelConfig):
            raise ValueError("model must have a model_config attribute")
        if not hasattr(model, 'config'):
            raise ValueError("model must have a config attribute")

        self._tp_size = 1 if model.model_config.mapping.enable_attention_dp else model.model_config.mapping.tp_size
        self._num_kv_heads = model.config.num_key_value_heads if hasattr(
            model.config, 'num_key_value_heads'
        ) and model.config.num_key_value_heads is not None else model.config.num_attention_heads

        self.is_initialized = True

    @abstractmethod
    def map_weights(self) -> None:
        """
        Maps weights from TRT-LLM to a source state dictionary (e.g., Hugging Face)
        """

    @abstractmethod
    def apply_callbacks(self, module_name: str,
                        module_names_breakdown: list[str],
                        weights: dict) -> list[dict]:
        """
        Applies a series of transformation functions to an internal representation
        of weights or to guide the mapping process. The exact behavior might depend
        on the implementation (e.g., storing callbacks to be applied later).

        Args:
            module_name: The specific module name (e.g., 'qkv_proj', 'gate_up_proj')
            module_names_breakdown: List of module path components for building full paths
            weights: The weights dictionary to process
        """

    def rename_by_params_map(self, params_map: dict[str, str],
                             weights: dict) -> dict:
        """
        Rename weight keys according to regex pattern matching.

        Args:
            pattern_mapping: A dictionary mapping regex patterns to replacement strings. The key is HF name pattern, and the value is corresponding TRT-LLM name pattern.
                The patterns will be used to match keys in the weights dict and replace
                them according to the replacement string, which can use regex backreferences.
                Example:
                HF name: vision_model.encoder.layers.1.self_attn.out_proj.{weight,bias}
                TRT-LLM name: vision_model.encoder.layers.1.self_attn.o_proj.{weight,bias}
                Then the pattern_mapping could be:
                pattern_mapping = {
                    r'(.*?)out_proj(.*)': r'\1o_proj\2'
                }
            weights: A dictionary of weights

        Returns:
            A dictionary of weights with renamed keys
        """
        import re

        # Create a new dictionary to store the renamed weights
        renamed_weights = {}

        # Keep track of keys that have been matched by a pattern
        matched_keys = set()

        # Process each key in the weights dictionary
        for key in list(weights.keys()):
            # Check each pattern for a match
            for pattern, replacement in params_map.items():
                if re.match(pattern, key):
                    # Create the new key by applying the regex replacement
                    new_key = re.sub(pattern, replacement, key)
                    # Store the weight with the new key
                    renamed_weights[new_key] = weights[key]
                    matched_keys.add(key)
                    break

            # If the key wasn't matched by any pattern, keep it as is
            if key not in matched_keys:
                renamed_weights[key] = weights[key]

        return renamed_weights

    def should_apply_to_module(self, module_name: str) -> bool:
        return module_name in self._mapping

    @property
    def skip_modules(self) -> List[str]:
        return self._skip_modules

    @skip_modules.setter
    def skip_modules(self, value: List[str]) -> None:
        self._skip_modules.extend(value)

    def should_skip_module(self, module_name: str) -> bool:
        return any(skip_module in module_name
                   for skip_module in self._skip_modules)

    def filter_weights(self, prefix: str, weights: dict) -> dict:
        result = {}
        for k, v in weights.items():
            if k.startswith(prefix):
                new_k = k[len(prefix) + 1:]
                result[new_k] = v
        return result

    @property
    def mapping(self) -> dict:
        return self._mapping
