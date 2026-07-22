# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    WeightGroup
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._utils import is_device_integrated

from ..base_weight_mapper import BaseWeightMapper


@register_mapper("MX")
@register_mapper("HF")
class HfWeightMapper(BaseWeightMapper):

    _MODEL_INCREMENTAL_LOADING_CAPABILITY = \
        "_supports_generic_hf_incremental_loading"
    _SOURCE_TO_FUSED_MODULE = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
    }

    def __init__(self):
        super().__init__()
        self._callbacks = [
            self._duplicate_kv_weights,
        ]

    def _is_generic_incremental_loading_qualified(self) -> bool:
        """Return whether the exact model class opts into the generic contract.

        Falling back to ``HfWeightMapper`` only proves that source names use the
        common HF layout. It does not prove that a model wrapper can consume
        dependency groups independently or that it will not retain borrowed
        source tensors. Require an audited model class to declare the capability
        itself. Read the exact class namespace so a derived architecture cannot
        inherit qualification without a separate audit.
        """
        if type(self) is not HfWeightMapper or self._model is None:
            return False
        return vars(type(self._model)).get(
            self._MODEL_INCREMENTAL_LOADING_CAPABILITY, False) is True

    @property
    def borrowed_source_tensors_safe(self) -> bool:
        """Whether the audited generic-HF model can borrow source tensors."""
        return (self._is_generic_incremental_loading_qualified()
                and self._borrowed_source_tensor_config_is_safe())

    def _borrowed_source_tensor_config_is_safe(self) -> bool:
        """Whether this exact runtime profile may use transport-owned views.

        Several quantized Linear and MoE loaders retain source-backed temporary
        tensors until end-of-checkpoint finalization. Dynamic EPLB deliberately
        retains complete raw expert tensors for later remapping. Both lifetimes
        exceed a shared-stream slot lease, so keep those profiles on the
        rank-local staging path until they have a separately audited per-group
        finalization contract.
        """
        if self._config is None:
            return False
        if is_device_integrated():
            # Integrated devices may call mmap page-eviction hooks while a
            # module loads. Keep node-shared transport storage outside that
            # legacy path until it has an explicit stream-owned marker.
            return False

        quant_config = getattr(self._config, "quant_config", None)
        if getattr(quant_config, "quant_algo", None) is not None:
            return False
        if getattr(self._config, "force_dynamic_quantization", False):
            return False

        quant_config_dict = getattr(self._config, "quant_config_dict", None)
        if quant_config_dict:
            for entry in quant_config_dict.values():
                # Real entries are QuantConfig objects. Unknown non-None values
                # fail closed rather than guessing that their source lifetime is
                # compatible with a reusable transport slot.
                if getattr(entry, "quant_algo", entry) is not None:
                    return False

        load_balancer = getattr(self._config, "moe_load_balancer", None)
        if (load_balancer is not None
                and getattr(load_balancer, "layer_updates_per_iter", 0) > 0):
            return False
        return True

    @staticmethod
    def _canonical_source_weight_name(key: str) -> str:
        return key

    @classmethod
    def _source_dependency_root(cls, key: str) -> str:
        """Map one source tensor to its atomic destination dependency."""
        canonical_key = cls._canonical_source_weight_name(key)
        module_name, separator, _ = canonical_key.rpartition(".")
        if not separator:
            return canonical_key

        module_parts = module_name.split(".")
        if "experts" in module_parts:
            # Routed-MoE loaders select layouts and jointly consume every local
            # expert tensor. Keep that destination module as one dependency.
            experts_index = module_parts.index("experts")
            return ".".join(module_parts[:experts_index + 1])

        fused_module = cls._SOURCE_TO_FUSED_MODULE.get(module_parts[-1])
        if fused_module is not None:
            module_parts[-1] = fused_module
        return ".".join(module_parts)

    def _has_routed_experts(self) -> bool:
        if self._config is not None:
            pretrained_config = self._config.pretrained_config
        elif self._model is not None:
            pretrained_config = self.model.config
        else:
            return False
        text_config = getattr(pretrained_config, "text_config", None)
        configs = (pretrained_config, text_config)
        expert_fields = ("num_experts", "num_local_experts", "n_routed_experts")
        return any(
            bool(getattr(config, field, 0)) for config in configs
            if config is not None for field in expert_fields)

    def _incremental_dependency_root(self, key: str) -> str:
        canonical_key = self._canonical_source_weight_name(key)
        module_name, separator, _ = canonical_key.rpartition(".")
        if self._has_routed_experts() and separator:
            module_parts = module_name.split(".")
            for moe_component in ("mlp", "feed_forward", "block_sparse_moe",
                                  "moe"):
                if moe_component in module_parts:
                    component_index = module_parts.index(moe_component)
                    return ".".join(module_parts[:component_index + 1])
        return self._source_dependency_root(key)

    def _source_weight_group_id(self, key: str) -> str:
        if key.startswith("mtp."):
            return "hf.mtp"

        return f"hf.{self._incremental_dependency_root(key)}"

    def get_weight_groups(self,
                          keys: Iterable[str]) -> list[WeightGroup] | None:
        """Group destination dependencies for the unmodified generic mapper.

        The generic mapper's only cross-tensor transforms are q/k/v and
        gate/up fusion. Each fused source family is kept together, routed-MoE
        tensors stay atomic at their complete MLP/MoE module, and ordinary
        parameter families are independent. Subclasses must opt in explicitly
        because their preprocessing may add dependencies across these
        boundaries.
        """
        if not self._is_generic_incremental_loading_qualified():
            return None

        grouped_keys: dict[str, list[str]] = {}
        seen_keys = set()
        for key in keys:
            if key in seen_keys:
                raise ValueError(f"Duplicate checkpoint tensor name {key!r}")
            seen_keys.add(key)
            group_id = self._source_weight_group_id(key)
            grouped_keys.setdefault(group_id, []).append(key)

        return [
            WeightGroup(group_id=group_id, keys=tuple(group_keys))
            for group_id, group_keys in grouped_keys.items()
        ]

    def _get_hf_incremental_load_roots(self,
                                       keys: Iterable[str]) -> tuple[str, ...]:
        roots = []
        seen_roots = set()
        for key in keys:
            root = self._incremental_dependency_root(key)
            if root not in seen_roots:
                roots.append(root)
                seen_roots.add(root)
        return tuple(roots)

    def get_incremental_load_roots(
            self, keys: Iterable[str]) -> tuple[str, ...] | None:
        """Dispatch exact generic-HF groups to their destination subtrees."""
        if not self._is_generic_incremental_loading_qualified():
            return None
        return self._get_hf_incremental_load_roots(keys)

    def map_weights(self) -> None:
        self.mapping.update({
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
            'gate_up_proj': ['gate_proj', 'up_proj']
        })

    def apply_callbacks(self, module: nn.Module, module_name: str,
                        module_names_breakdown: list[str],
                        weights: dict) -> list[dict]:
        module_weights = []

        for new_name in self._mapping[module_name]:
            fw = self.filter_weights(
                '.'.join(module_names_breakdown + [new_name]), weights)
            for callback in self._callbacks:
                fw = callback(module, new_name, fw)
            module_weights.append(fw)

        return module_weights

    def should_skip_module(self, module_name: str) -> bool:
        if getattr(self.model.config, 'tie_word_embeddings',
                   False) and module_name.startswith("lm_head"):
            return True

        # Skip loading weights for embedding and lm_head if LoRA is enabled and has custom values
        if hasattr(self.model, "model") and hasattr(
                self.model.model, 'has_custom_embed_tokens'
        ) and self.model.model.has_custom_embed_tokens and module_name == "model.embed_tokens":
            return True
        if hasattr(
                self.model, 'has_custom_lm_head'
        ) and self.model.has_custom_lm_head and module_name == "lm_head":
            return True

        # WAR: better solution is that llama has its own load_weights function.
        if module_name.split('.')[-1] == 'next_layer_layernorm':
            return True

        return super().should_skip_module(module_name)

    @property
    def _num_kv_heads(self) -> int:
        config = self.model.config
        if hasattr(config, 'num_key_value_heads'
                   ) and config.num_key_value_heads is not None:
            return config.num_key_value_heads
        return config.num_attention_heads

    def _duplicate_kv_weights(self, module: nn.Module, new_name: str,
                              weights: dict):
        if new_name in ['k_proj', 'v_proj']:
            num_kv_heads = self._num_kv_heads

            duplicated_keys = ["weight", "bias"]
            if module.quant_config.quant_mode.has_nvfp4():
                duplicated_keys.append("weight_scale")

            processed_weights = {
                k:
                self._duplicate_kv(weight=v[:],
                                   num_kv_heads=num_kv_heads,
                                   tensor_parallel_size=self._tp_size)
                if k in duplicated_keys else v
                for k, v in weights.items()
            }
            return processed_weights

        return weights

    def _duplicate_kv(self, weight: torch.Tensor, num_kv_heads: int,
                      tensor_parallel_size: int):

        if num_kv_heads >= tensor_parallel_size:
            assert num_kv_heads % tensor_parallel_size == 0
            return weight

        assert tensor_parallel_size % num_kv_heads == 0
        reps = tensor_parallel_size // num_kv_heads

        # bias
        if weight.ndim == 1:
            return weight.repeat_interleave(reps)

        # weight and scale
        assert weight.shape[0] % num_kv_heads == 0
        size_per_kv_head = weight.shape[0] // num_kv_heads
        weight = weight.reshape(num_kv_heads, size_per_kv_head,
                                -1)[:,
                                    None, :, :].expand(num_kv_heads, reps,
                                                       size_per_kv_head,
                                                       weight.shape[1])
        return weight.reshape(num_kv_heads * reps * size_per_kv_head,
                              -1).clone().detach()
