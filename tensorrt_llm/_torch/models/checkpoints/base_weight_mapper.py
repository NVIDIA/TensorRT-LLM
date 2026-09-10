# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Callable, Mapping

import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.checkpoint_catalog import \
    CheckpointCatalog
from tensorrt_llm._torch.models.checkpoints.weight_load_plan import (
    WeightDemand, WeightLoadOrderConfidence, WeightLoadPlan,
    WeightLoadPlanCoverage)
from tensorrt_llm._torch.models.modeling_utils import (DecoderModelForCausalLM,
                                                       is_moe_weight_owner)


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

    def build_weight_load_plan(self,
                               catalog: CheckpointCatalog) -> WeightLoadPlan:
        """Build a conservative, source-neutral load plan for this rank.

        This method mirrors the native module walk without invoking any weight
        transformation or loading callback. It uses module structure, explicit
        fusion mappings, and preload hints to identify likely consumer groups.
        The inference is advisory: every source tensor is retained exactly once,
        unmatched tensors remain in physical catalog order at the tail, and the
        plan never permits selective I/O. Specialized mappers may override this
        method when they can provide stronger guarantees.
        """
        if self._model is None:
            raise ValueError(
                "weight mapper must be initialized before building a load plan")
        mapping = self._model.model_config.mapping

        # Catalog order is the source adapter's physical order. Preserve it
        # within each inferred group and in the unmatched tail.
        physical_names = tuple(tensor.name for tensor in catalog.tensors)
        physical_index = {
            name: index
            for index, name in enumerate(physical_names)
        }
        sorted_names = tuple(sorted(physical_names))

        named_modules = getattr(self._model, "named_modules", None)
        if named_modules is None:
            return self._build_opaque_weight_load_plan(catalog, physical_names)

        modules = tuple(named_modules(remove_duplicate=False))
        remaining_names = set(physical_names)
        demands: list[WeightDemand] = []

        def take_prefixes(prefixes: tuple[str, ...]) -> tuple[str, ...]:
            """Take unassigned tensor names below structural module prefixes."""
            matched_indexes: set[int] = set()
            for prefix in prefixes:
                if not prefix:
                    continue
                if prefix in remaining_names:
                    matched_indexes.add(physical_index[prefix])

                descendant_prefix = f"{prefix}."
                index = bisect_left(sorted_names, descendant_prefix)
                while (index < len(sorted_names)
                       and sorted_names[index].startswith(descendant_prefix)):
                    name = sorted_names[index]
                    if name in remaining_names:
                        matched_indexes.add(physical_index[name])
                    index += 1

            result = tuple(physical_names[index]
                           for index in sorted(matched_indexes))
            remaining_names.difference_update(result)
            return result

        def take_exact(names: tuple[str, ...]) -> tuple[str, ...]:
            matched_indexes = {
                physical_index[name]
                for name in names
                if name in remaining_names and name in physical_index
            }
            result = tuple(physical_names[index]
                           for index in sorted(matched_indexes))
            remaining_names.difference_update(result)
            return result

        def source_module_path(module_path: str, module: nn.Module) -> str:
            """Mirror native path normalization without touching payloads."""
            parts = module_path.split('.') if module_path else []
            if (parts and parts[-1] == "backend"
                    and is_moe_weight_owner(module)):
                return '.'.join(parts[:-1])
            return module_path

        def infer_source_names(module_path: str,
                               module: nn.Module) -> tuple[str, ...]:
            # Match the materializer's early exit: only modules that own at
            # least one parameter slot can be consumers in the native walk.
            if not module._parameters or self.should_skip_module(module_path):
                return ()

            normalized_path = source_module_path(module_path, module)
            module_parts = normalized_path.split('.') if normalized_path else []
            module_name = module_parts[-1] if module_parts else ""
            parent_parts = module_parts[:-1]

            # A fused destination consumes raw tensors from explicitly named
            # sibling subtrees, for example q_proj/k_proj/v_proj -> qkv_proj.
            if module_name and self.does_require_special_handling(module_name):
                prefixes = tuple('.'.join(parent_parts + [source_name])
                                 for source_name in self.mapping[module_name])
                return take_prefixes(prefixes)

            # Custom module loaders receive the full structurally matched
            # subtree. Do not treat the root as a subtree consumer: an empty
            # prefix contains the whole checkpoint and conveys no useful order.
            # Other model-specific path rewrites that are not expressed by the
            # mapper (for example a checkpoint-only namespace prefix)
            # intentionally miss here and remain in the safe, all-source
            # unmatched tail.
            if normalized_path and (self.is_special_instance_module(module)
                                    or hasattr(module, 'load_weights')):
                return take_prefixes((normalized_path, ))

            # The native fallback copies only direct parameters. Descendant
            # parameters belong to their own named-module consumers.
            parameter_names = tuple(
                f"{normalized_path}.{name}" if normalized_path else name
                for name, _ in module.named_parameters(recurse=False))
            return take_exact(parameter_names)

        preload_suffixes = tuple(
            getattr(self._model, "preload_weight_modules", None) or ())
        preload_modules: list[tuple[str, nn.Module]] = []
        preload_paths: set[str] = set()
        for suffix in preload_suffixes:
            for module_path, module in modules:
                if (source_module_path(module_path, module).endswith(suffix)
                        and module_path not in preload_paths):
                    preload_paths.add(module_path)
                    preload_modules.append((module_path, module))

        previous_preload_group: str | None = None
        for module_path, module in preload_modules:
            source_names = infer_source_names(module_path, module)
            if not source_names:
                continue
            group_id = f"preload:{len(demands):06d}:{module_path}"
            predecessors = ((previous_preload_group, )
                            if previous_preload_group is not None else ())
            demands.append(
                WeightDemand(
                    group_id=group_id,
                    source_names=source_names,
                    destination_ranks=(mapping.rank, ),
                    priority=len(demands),
                    predecessors=predecessors,
                ))
            previous_preload_group = group_id

        # Native materialization schedules non-preloaded modules concurrently,
        # so represent them with equal priority and no artificial dependency
        # chain. Tuple order remains a deterministic tie-breaker.
        nonpreload_priority = len(demands)
        for module_path, module in modules:
            if module_path in preload_paths:
                continue
            source_names = infer_source_names(module_path, module)
            if not source_names:
                continue
            demands.append(
                WeightDemand(
                    group_id=
                    f"module:{len(demands):06d}:{module_path or '<root>'}",
                    source_names=source_names,
                    destination_ranks=(mapping.rank, ),
                    priority=nonpreload_priority,
                ))

        # If structural inspection found nothing useful, expose no ordering
        # claim. Otherwise retain every unresolved tensor as a physical tail.
        if not demands:
            return self._build_opaque_weight_load_plan(catalog, physical_names)

        unmatched_names = tuple(name for name in physical_names
                                if name in remaining_names)
        if unmatched_names:
            demands.append(
                WeightDemand(
                    group_id="unmatched_checkpoint_tensors",
                    source_names=unmatched_names,
                    destination_ranks=(mapping.rank, ),
                    priority=nonpreload_priority + 1,
                ))

        plan = WeightLoadPlan(
            catalog_id=catalog.catalog_id,
            rank=mapping.rank,
            world_size=mapping.world_size,
            coverage=WeightLoadPlanCoverage.CONSERVATIVE,
            ordering=WeightLoadOrderConfidence.ADVISORY,
            demands=tuple(demands),
        )
        plan.validate_against(catalog)
        return plan

    def _build_opaque_weight_load_plan(
            self, catalog: CheckpointCatalog,
            physical_names: tuple[str, ...]) -> WeightLoadPlan:
        """Build an all-source plan when no useful ordering can be inferred."""
        mapping = self.model.model_config.mapping
        plan = WeightLoadPlan(
            catalog_id=catalog.catalog_id,
            rank=mapping.rank,
            world_size=mapping.world_size,
            coverage=WeightLoadPlanCoverage.CONSERVATIVE,
            ordering=WeightLoadOrderConfidence.OPAQUE,
            demands=(WeightDemand(
                group_id="all_checkpoint_tensors",
                source_names=physical_names,
                destination_ranks=(mapping.rank, ),
            ), ),
        )
        plan.validate_against(catalog)
        return plan

    def begin_update_weights(self) -> None:
        """Prepare mapper-owned state for an incremental weight update."""

    def finalize_update_weights(self) -> None:
        """Validate and release mapper-owned incremental update state."""

    def abort_update_weights(self) -> None:
        """Release mapper-owned state after a failed incremental update."""

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
          `ConsumableWeightsDict`, the returned object preserves that type and
          takes the tensors over from it -- the input is emptied, so **the
          caller must not use `weights` afterwards**. See
          `ConsumableWeightsDict.take_ownership` for why the transfer is what
          lets the loader release weights module by module.
        """
        import re

        from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
            ConsumableWeightsDict

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

        return ConsumableWeightsDict.take_ownership(weights, renamed_weights)

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
        `handle_special_instance_module`. This hook is for special modules who cannot be
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
        from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
            ConsumableWeightsDict

        # The loading loop calls this once per module, so on a large
        # checkpoint the scan below is quadratic; a ConsumableWeightsDict can
        # answer the same query from its key index instead.
        if prefix and isinstance(weights, ConsumableWeightsDict):
            return weights.filter_prefix(prefix)

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
