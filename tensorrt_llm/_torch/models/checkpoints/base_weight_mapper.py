from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM


class BaseWeightMapper(ABC):
    """Helper for loading weights to each child module.

    A typical weight loader function walks `model.named_modules()` and loads
    weights for each child module. Although called `WeightMapper`, this class is
    called at multiple locations of the walk to do:
    - Weight dict preprocessing
    - Weight dict key renaming
    - Finding modules requiring special weight handling and calling weight processing
      hooks before passing them to the module
    - Finding really special modules and have complete custom methods to handle their
      weight loading.
    - Basic weight name prefix removing and weight copying onto PyTorch module
      Parameter.

    Subclasses of `BaseWeightMapper` can be selected by checkpoint format and model
    architecture through `AutoCheckpointMapper`.

    Abstract methods for subclasses to implement:
    - map_weights
    - apply_callbacks
    """

    def __init__(self):
        self._callbacks: list[Callable] = []
        # Mapping for modules that need special weight loading, like fusing
        # several weights.
        # It maps module names to the corresponding source names in the checkpoint, e.g.
        # 'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        # 'gate_up_proj': ['gate_proj', 'up_proj']
        # It will be initialized in `self.map_weights()` and queried in
        # `self.does_require_special_handling()`
        self._mapping: dict[str, list[str]] = {}
        self._skip_modules: list[str] = []
        self._model: nn.Module | DecoderModelForCausalLM | None = None
        self._config: ModelConfig | None = None

    def init_model_and_config(self, model: nn.Module | DecoderModelForCausalLM,
                              config: ModelConfig) -> None:
        """Bind this mapper to the LLM class instance and model config.

        Called once after the model is constructed and before weight loading
        begins. It validates the model has the attributes needed by the mapper,
        records the tensor parallel size, and calls `map_weights` so subclasses
        can populate fused-module mappings.

        Args
        - model: nn.Module | DecoderModelForCausalLM, LLM class instance whose
          child modules will be loaded.
        - config: ModelConfig, loaded model config used by mapper decisions.
        """
        self._model = model
        self._config = config

        if not hasattr(model, 'model_config') or not isinstance(
                model.model_config, ModelConfig):
            raise ValueError("model must have a model_config attribute")
        if not hasattr(model, 'config'):
            raise ValueError("model must have a config attribute")

        self._tp_size = 1 if model.model_config.mapping.enable_attention_dp else model.model_config.mapping.tp_size

        self.map_weights()

    def cleanup(self) -> None:
        self._model = None
        self._config = None

    @abstractmethod
    def map_weights(self) -> None:
        """Initialize mapping for modules that need special weight loading like weight fusion.

        This function is called inside `self.init_model_and_config()`. Derived classes implement
        this function to initialize `self.mapping`, which maps special module names to the
        corresponding source names in the checkpoint, e.g.
        - 'qkv_proj': ['q_proj', 'k_proj', 'v_proj']
        - 'gate_up_proj': ['gate_proj', 'up_proj']
        """

    @abstractmethod
    def apply_callbacks(
            self, module: nn.Module, module_name: str,
            module_names_breakdown: list[str],
            weights: Mapping[str,
                             torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Build processed weight dicts for a special child module.

        Used only when `self.does_require_special_handling()` is True, derived classes
        implement this function to process raw `weights` before passing them into
        the `module.load_weights()`.

        Example special module: qkv_proj that combines q_proj, k_proj and v_proj.

        Args
        - module: nn.Module, must have `module.load_weights()` to receive the
          returned weight dicts.
        - module_name: str, final component of the child module name, such as
          `qkv_proj` or `gate_up_proj`.
        - module_names_breakdown: list[str], parent module path split on `.`,
          needed for finding weights for this child module.
        - weights: Mapping[str, torch.Tensor], full checkpoint weight dict.

        Returns
        - module_weights: list[Mapping[str, torch.Tensor]], list of weight dicts
          to pass to `module.load_weights()`.
        """

    def rename_by_params_map(
            self, params_map: dict[str, str],
            weights: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """Rename checkpoint keys in `weights` with string rules defined in `params_map`.

        This should be called at beginning of a weight loading function to rename
        input weights.

        The basic implementation is regex replacement rule but derived classes of
        `BaseWeightMapper` may change that.

        Regex rule example: `r'(.*?)out_proj(.*)' -> r'\\1o_proj\\2'` maps
        `vision_model.encoder.layers.1.self_attn.out_proj.weight` to
        `vision_model.encoder.layers.1.self_attn.o_proj.weight`.

        Args
        - params_map: dict[str, str], regex pattern to replacement string.
          Replacement strings may use regex backreferences.
        - weights: Mapping[str, torch.Tensor], checkpoint/state-dict weight
          tensors keyed by checkpoint name.

        Returns
        - renamed_weights: Mapping[str, torch.Tensor], weight dict with renamed
          keys and unchanged tensor values. If the input `weights` is a
          `ConsumableWeightsDict`, the returned object preserved that type.
        """
        import re

        from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
            ConsumableWeightsDict

        # Check if input is a ConsumableWeightsDict to preserve the type
        is_consumable = isinstance(weights, ConsumableWeightsDict)

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

        # Preserve ConsumableWeightsDict type if that's what was passed in
        if is_consumable:
            return ConsumableWeightsDict(renamed_weights)
        return renamed_weights

    def preprocess_weights(
            self, weights: Mapping[str,
                                   torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """Rewrite a full checkpoint weight dict before module walking starts.

        If simple string renaming rules used in `rename_by_params_map` does not satisfy the
        need, call this function for a more custom rewrite of the checkpoint weight dict.

        Args
        - weights: Mapping[str, torch.Tensor], full checkpoint/state-dict weight
          tensors keyed by checkpoint name.

        Returns
        - weights: Mapping[str, torch.Tensor], preprocessed weight dict to pass to
          the child-module loader.
        """
        ...

    def handle_manual_copy(self,
                           module_name: str,
                           module_weights: Mapping[str, torch.Tensor],
                           n: str,
                           p: nn.Parameter,
                           allow_partial_loading: bool = False) -> None:
        """Copy one parameter for a module that has no `load_weights` method.

        Args
        - module_name: str, final component of the child module name.
        - module_weights: Mapping[str, torch.Tensor], tensors for this child module
          with the module prefix removed.
        - n: str, parameter name inside the child module, such as `weight` or
          `bias`.
        - p: nn.Parameter, destination parameter to update.
        - allow_partial_loading: bool, if `True`, skip missing parameters; if
          `False`, assert that `n` exists in `module_weights`.
        """
        if not allow_partial_loading:
            assert n in module_weights
        if n in module_weights:
            p.data.copy_(module_weights[n][:])

    def does_require_special_handling(self, module_name: str) -> bool:
        """
        Whether a module requires special weight loading like fusing weights.
        Examples include module 'qkv_proj' fuses weights 'q_proj', 'k_proj' and 'v_proj'.
        """
        return module_name in self.mapping

    def is_special_instance_module(self, module: nn.Module) -> bool:
        """If the module is special enough for a complete custom weight handling hook.

        If true, the weight loader function should call
        `self.handle_special_instance_module()` next to handle that special module.
        """
        return False

    def handle_special_instance_module(
            self,
            module: nn.Module,
            module_name: str,
            module_weights: Mapping[str, torch.Tensor],
            allow_partial_loading: bool = False) -> None:
        """Load weights for a special module that needs custom behavior.

        Only call this if `self.is_special_instance_module()` returns true.
        Subclasses opt into this path by overriding `is_special_instance_module` and
        `is_special_instance_module`. This hook is for special modules who cannot be
        handled by `does_require_special_handling()` and `apply_callbacks()`.

        Args
        - module: nn.Module, special module to load weights.
        - module_name: str, final component of the child module name.
        - module_weights: Mapping[str, torch.Tensor], tensors for this child module
          with the module prefix removed.
        - allow_partial_loading: bool, if `True`, the subclass should tolerate
          missing tensors when its custom loading logic supports that mode.
        """
        raise NotImplementedError()

    @property
    def skip_modules(self) -> list[str]:
        return self._skip_modules

    def add_skip_modules(self, value: list[str]) -> None:
        self._skip_modules.extend(value)

    def should_skip_module(self, module_name: str) -> bool:
        return any(skip_module in module_name
                   for skip_module in self._skip_modules)

    def filter_weights(
            self, prefix: str,
            weights: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Return only weights that start with the prefix (and with the prefix removed)
        """
        result = {}
        for k, v in weights.items():
            if k.startswith(prefix):
                new_k = k[len(prefix) + 1:]
                result[new_k] = v
        return result

    @property
    def mapping(self) -> dict[str, list[str]]:
        """Return the mapping for modules that need special weight loading.

        It maps module names to the corresponding source names in the checkpoint, e.g.
        'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        'gate_up_proj': ['gate_proj', 'up_proj']
        Those two modules fuse several GEMM weights together for a single GEMM call.

        Returns the mapping as dict[str, list[str]]
        """
        return self._mapping

    @property
    def config(self) -> ModelConfig:
        if self._config is None:
            raise RuntimeError("Weight mapper is not initialized")
        return self._config

    @property
    def model(self) -> nn.Module | DecoderModelForCausalLM:
        if self._model is None:
            raise RuntimeError("Weight mapper is not initialized")
        return self._model

    @property
    def _head_dim(self) -> int:
        model = self.model
        head_dim = model.config.head_dim if hasattr(
            model.config, 'head_dim'
        ) and model.config.head_dim is not None else model.config.hidden_size // model.config.num_attention_heads
        return head_dim
