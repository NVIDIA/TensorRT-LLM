"""
Dynamic weight quantization loader for Linear modules.

Wraps Linear.load_weights() to perform dynamic quantization before loading to device.
"""

from typing import Dict, List, Optional

import torch

from tensorrt_llm._torch.modules.linear import Linear, WeightMode
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.quantization.ops import (
    quantize_fp8_blockwise,
    quantize_fp8_per_tensor,
)
from tensorrt_llm.quantization.mode import QuantAlgo


class DynamicLinearWeightLoader:
    """
    Dynamic weight quantization loader for Linear modules.

    Wraps Linear.load_weights() to perform dynamic (load-time) quantization
    from BF16/FP16 to FP8 before loading weights to device.

    Example:
        params_map = {'qkv_proj': ['to_q', 'to_k', 'to_v']}
        loader = DynamicLinearWeightLoader(model_config, params_map=params_map)

        for name, module in model.named_modules():
            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)
                loader.load_linear_weights(module, name, weight_dicts)
    """

    def __init__(
        self,
        model_config: DiffusionModelConfig,
        params_map: Optional[Dict[str, List[str]]] = None,
    ):
        self.model_config = model_config
        self.quant_config = model_config.quant_config
        self.quant_config_dict = model_config.quant_config_dict
        self.dynamic_weight_quant = model_config.dynamic_weight_quant
        self.params_map = params_map or {}

    # =========================================================================
    # Weight gathering methods
    # =========================================================================

    def get_linear_weights(
        self,
        module: Linear,
        full_name: str,
        weights: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """Get weights for a Linear module, auto-detecting fused weights."""
        weights_config = getattr(module, "weights_loading_config", None)
        if weights_config is not None:
            weight_mode = getattr(weights_config, "weight_mode", None)
            if weight_mode == WeightMode.FUSED_QKV_LINEAR:
                fused_names = self._get_fused_names(full_name)
                return self._get_fused_weights(full_name, weights, fused_names)

        return self._get_vanilla_weights(full_name, weights)

    def filter_weights(
        self, prefix: str, weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Filter weights by prefix and strip the prefix.

        Example:
            prefix = 'blocks.0.attn1.to_q'
            weights = {'blocks.0.attn1.to_q.weight': ..., 'blocks.0.attn1.to_q.bias': ...}
            Returns: {'weight': ..., 'bias': ...}
        """
        result = {}
        prefix_dot = prefix + "."
        for k, v in weights.items():
            if k.startswith(prefix_dot):
                result[k[len(prefix_dot) :]] = v
        return result

    def _get_fused_names(self, full_name: str) -> List[str]:
        """Get checkpoint names for a fused module from params_map."""
        for suffix, names in self.params_map.items():
            if full_name.endswith(suffix):
                return names
        raise ValueError(
            f"No params_map entry for fused module '{full_name}'. "
            f"Add mapping like {{'qkv_proj': ['to_q', 'to_k', 'to_v']}} to params_map."
        )

    def _get_fused_weights(
        self,
        full_name: str,
        weights: Dict[str, torch.Tensor],
        fused_names: List[str],
    ) -> List[Dict[str, torch.Tensor]]:
        """Get weights for a fused module from checkpoint."""
        parent_path = ".".join(full_name.split(".")[:-1])
        module_weights = []
        for ckpt_name in fused_names:
            ckpt_path = f"{parent_path}.{ckpt_name}" if parent_path else ckpt_name
            filtered = self.filter_weights(ckpt_path, weights)
            module_weights.append(filtered)
        return module_weights

    def _get_vanilla_weights(
        self,
        full_name: str,
        weights: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """Get weights for a standard (non-fused) Linear module."""
        fw = self.filter_weights(full_name, weights)
        return [fw] if fw else []

    # =========================================================================
    # Quantization methods
    # =========================================================================

    def _get_quant_algo_for_layer(self, name: str) -> Optional[QuantAlgo]:
        """Get quantization algorithm for a specific layer."""
        if self.quant_config_dict is not None:
            layer_config = self.quant_config_dict.get(name)
            if layer_config is not None:
                return layer_config.quant_algo

        if self.quant_config is not None:
            return self.quant_config.quant_algo

        return None

    def _should_dynamic_quantize(
        self, weight_dict: Dict[str, torch.Tensor], quant_algo: Optional[QuantAlgo], name: str
    ) -> bool:
        """Decide if weight should be dynamically quantized at load time."""
        if not self.dynamic_weight_quant or quant_algo is None:
            return False

        # Check if module is excluded
        if self.quant_config is not None:
            if self.quant_config.is_module_excluded_from_quantization(name):
                return False

        weight = weight_dict.get("weight")
        if weight is None:
            return False

        # For FP8 algorithms: quantize if weight is high precision
        if quant_algo in (QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES):
            if weight.dtype == torch.float8_e4m3fn and "weight_scale" in weight_dict:
                return False  # Already quantized
            return weight.dtype in (torch.bfloat16, torch.float16, torch.float32)

        return False

    def _maybe_dynamic_quantize(
        self, weight_dict: Dict[str, torch.Tensor], quant_algo: Optional[QuantAlgo], name: str
    ) -> Dict[str, torch.Tensor]:
        """Conditionally quantize weight at load time on GPU."""
        if not self._should_dynamic_quantize(weight_dict, quant_algo, name):
            return weight_dict

        weight = weight_dict["weight"]

        # Move to GPU only if needed
        if weight.device.type != "cuda":
            weight = weight.cuda()

        if quant_algo == QuantAlgo.FP8:
            qweight, scale = quantize_fp8_per_tensor(weight)
        elif quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
            block_size = self.quant_config.group_size if self.quant_config else 128
            qweight, scale = quantize_fp8_blockwise(weight, block_size=block_size)
        else:
            return weight_dict

        return {**weight_dict, "weight": qweight, "weight_scale": scale}

    def load_linear_weights(
        self, module: Linear, name: str, weight_dicts: List[Dict[str, torch.Tensor]]
    ) -> None:
        """Load weights into Linear module with optional quantization."""
        module_quant_config = getattr(module, "quant_config", None)
        if module_quant_config is not None:
            quant_algo = module_quant_config.quant_algo
        else:
            quant_algo = self._get_quant_algo_for_layer(name)

        quantized_weight_dicts = [
            self._maybe_dynamic_quantize(wd, quant_algo, name) for wd in weight_dicts
        ]

        module.load_weights(quantized_weight_dicts)
