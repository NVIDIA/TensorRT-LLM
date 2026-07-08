import math
import re
from collections import defaultdict

import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.qwen3_next_weight_mapper import (
    Qwen3NextHfWeightMapper,
)
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.fused_moe.interface import MoE, MoEWeightLoadingMode
from tensorrt_llm.quantization import QuantAlgo

_FP8_2D_BLOCK_SIZE = 128


@register_mapper("HF", "Qwen3_5ForConditionalGeneration")
@register_mapper("HF", "QwenImageBenchForConditionalGeneration")
@register_mapper("HF", "Qwen3_5MoeForCausalLM")
@register_mapper("HF", "Qwen3_5MoeForConditionalGeneration")
@register_mapper("HF", "Qwen3_5ForCausalLM")
@register_mapper("HF", "Qwen3_5ForConditionalGeneration")
class Qwen3_5MoeHfWeightMapper(Qwen3NextHfWeightMapper):
    """Weight mapper for Qwen3.5 MoE text checkpoints.

    Qwen3.5 and Qwen3Next share the same model architecture (Qwen3NextModel),
    but their HF checkpoint layouts differ in three ways that require extra
    handling here:

    1. Weight namespace (handled in _normalize_weight_names):
       Qwen3.5 VLM checkpoints nest text weights under model.language_model.*
       and include model.visual.* tensors.  This mapper strips the
       language_model prefix and drops vision tensors so the shared
       Qwen3NextModel can load them.

    2. Linear-attention projections (handled in _pack_split_projections):
       Qwen3Next checkpoints store pre-packed in_proj_qkvz and in_proj_ba
       tensors.  Qwen3.5 checkpoints store them as separate in_proj_qkv + z
       (or fully split q/k/v/z) and b + a tensors.  This mapper packs them
       into the grouped-interleaved layout that TRT-LLM expects.
       For FP8 checkpoints, the packed qkvz tensor is then dequantized to
       bf16 as a temporary workaround for TP loading
       (handled in _dequantize_linear_attn_fp8_qkvz).

    3. MoE expert tensors (handled in handle_special_instance_module):
       Qwen3.5 BF16 checkpoints store fused gate_up_proj/down_proj per expert
       in transposed orientation [E, 2*I, H] vs TRT-LLM's [E, H, 2*I].
       Qwen3.5 FP8 checkpoints store vanilla gate_proj/up_proj/down_proj per
       expert.  This mapper detects which layout is present, transposes fused
       tensors, renames keys, and sets the matching MoEWeightLoadingMode.
    """

    _SPLIT_PROJ_PATTERN = re.compile(r"^(.*\.linear_attn)\.in_proj_(qkv|q|k|v|z|b|a)\.(.+)$")
    _DENSE_MLP_PATTERN = re.compile(
        r"^((?:model|mtp)\.layers\.\d+\.mlp)\.(gate_proj|up_proj|down_proj|gate_up_proj)(\..+)$"
    )

    def _normalize_weight_names(self, weights: dict) -> dict:
        normalized_weights = {}
        for key, tensor in weights.items():
            if key.startswith("model.visual."):
                continue
            if key.startswith("model.language_model."):
                key = "model." + key[len("model.language_model.") :]
            normalized_weights[key] = tensor
        return normalized_weights

    def _normalize_scale_names(self, weights: dict, quant_algo) -> tuple[dict, bool]:
        # Canonicalize FP8 weight_scale layout so the Linear loader sees one
        # shape per quant algo:
        #   - FP8_BLOCK_SCALES: modelopt fp8_pb_wo stores weight_scale shaped
        #     [blocks_out, 1, blocks_in, 1]; squeeze to [blocks_out, blocks_in]
        #     and rename to weight_scale_inv. Returns is_modelopt_pb_wo=True
        #     so the caller can keep modelopt's native FP8 path.
        #   - FP8_PER_CHANNEL_PER_TOKEN: compressed-tensors stores weight_scale
        #     shaped [out, 1]; squeeze to 1-D [out].

        is_modelopt_pb_wo = False
        if quant_algo not in (QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN):
            return weights, is_modelopt_pb_wo

        remapped_weights = {}
        for key, tensor in weights.items():
            if key.endswith(".weight_scale"):
                if quant_algo == QuantAlgo.FP8_BLOCK_SCALES and tensor.ndim == 4:
                    assert tensor.shape[1] == 1 and tensor.shape[-1] == 1, (
                        f"Expected scale shape [*, 1, *, 1] for {key}, got {tuple(tensor.shape)}"
                    )
                    tensor = tensor.squeeze(1).squeeze(-1)
                    # Rename so downstream packing and linear-attn dequant can
                    # detect 2D-block scales by suffix alone.
                    key = key[: -len("weight_scale")] + "weight_scale_inv"
                    is_modelopt_pb_wo = True
                elif (
                    quant_algo == QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
                    and tensor.ndim == 2
                    and tensor.shape[1] == 1
                ):
                    tensor = tensor.squeeze(-1)
            if key in remapped_weights:
                raise ValueError(f"Duplicate remapped key found: {key}")
            remapped_weights[key] = tensor
        return remapped_weights, is_modelopt_pb_wo

    def handle_special_instance_module(
        self,
        module: nn.Module,
        module_name: str,
        module_weights: dict,
        allow_partial_loading: bool = False,
    ) -> None:
        if isinstance(module, MoE):
            config = self.config.pretrained_config
            uses_fused_expert_tensors = "gate_up_proj" in module_weights
            updated_module_weights = {}
            for weight_name, weight_value in module_weights.items():
                if weight_name == "gate_up_proj" and weight_value.ndim == 3:
                    if weight_value.shape[-2] == 2 * config.moe_intermediate_size and (
                        weight_value.shape[-1] == config.hidden_size
                    ):
                        weight_value = weight_value.transpose(-1, -2).contiguous()
                elif weight_name == "down_proj" and weight_value.ndim == 3:
                    if weight_value.shape[-2] == config.hidden_size and (
                        weight_value.shape[-1] == config.moe_intermediate_size
                    ):
                        weight_value = weight_value.transpose(-1, -2).contiguous()
                if uses_fused_expert_tensors:
                    new_weight_name = weight_name.replace("scale_inv", "weight_scale")
                else:
                    new_weight_name = (
                        weight_name.replace("gate_proj", "w1")
                        .replace("up_proj", "w3")
                        .replace("down_proj", "w2")
                    )
                updated_module_weights[new_weight_name] = weight_value
            module.weight_loading_mode = (
                MoEWeightLoadingMode.FUSED_GATE_UP_PROJ
                if uses_fused_expert_tensors
                else MoEWeightLoadingMode.VANILLA
            )
            module.load_weights(
                weights=[updated_module_weights], allow_partial_loading=allow_partial_loading
            )
            return
        return super().handle_special_instance_module(
            module, module_name, module_weights, allow_partial_loading=allow_partial_loading
        )

    def _pack_projection_tensor(self, tensors: list[torch.Tensor], num_groups: int) -> torch.Tensor:
        reference_shape = tensors[0].shape[1:]
        for tensor in tensors:
            assert tensor.shape[1:] == reference_shape, (
                f"Expected matching trailing dims while packing projections, got "
                f"{tensors[0].shape} and {tensor.shape}"
            )
            assert tensor.shape[0] % num_groups == 0, (
                f"Projection with shape {tensor.shape} is not divisible by {num_groups} groups"
            )

        reshaped = [
            tensor.reshape(num_groups, tensor.shape[0] // num_groups, *reference_shape)
            for tensor in tensors
        ]
        return torch.cat(reshaped, dim=1).reshape(-1, *reference_shape).contiguous()

    def _split_qkv_tensor(
        self, tensor: torch.Tensor, expected_q: int, expected_v: int
    ) -> tuple[torch.Tensor, ...]:
        expected_total = expected_q * 2 + expected_v
        assert tensor.shape[0] == expected_total, (
            f"Expected packed qkv projection with leading dim {expected_total}, got {tensor.shape}"
        )
        return torch.split(tensor, [expected_q, expected_q, expected_v], dim=0)

    def _split_qkv_scale_tensor(
        self, tensor: torch.Tensor, expected_q: int, expected_v: int
    ) -> tuple[torch.Tensor, ...]:
        expected_q_blocks = math.ceil(expected_q / _FP8_2D_BLOCK_SIZE)
        expected_v_blocks = math.ceil(expected_v / _FP8_2D_BLOCK_SIZE)
        expected_total_blocks = expected_q_blocks * 2 + expected_v_blocks
        assert tensor.shape[0] == expected_total_blocks, (
            f"Expected packed qkv scale tensor with leading dim {expected_total_blocks}, "
            f"got {tensor.shape}"
        )
        return torch.split(tensor, [expected_q_blocks, expected_q_blocks, expected_v_blocks], dim=0)

    def _dequantize_fp8_block_scale_weight(
        self, weight: torch.Tensor, weight_scale_inv: torch.Tensor
    ) -> torch.Tensor:
        rows, cols = weight.shape
        expanded_scales = (
            weight_scale_inv.to(torch.float32)
            .repeat_interleave(_FP8_2D_BLOCK_SIZE, dim=0)
            .repeat_interleave(_FP8_2D_BLOCK_SIZE, dim=1)[:rows, :cols]
        )
        target_dtype = getattr(self.config.pretrained_config, "torch_dtype", torch.bfloat16)
        if target_dtype is None:
            target_dtype = torch.bfloat16
        return (weight.to(torch.float32) * expanded_scales).to(target_dtype).contiguous()

    def _dequantize_linear_attn_fp8_qkvz(self, weights: dict) -> dict:
        updated_weights = dict(weights)
        for name in list(weights):
            if not name.endswith(".linear_attn.in_proj_qkvz.weight"):
                continue
            scale_name = name.replace(".weight", ".weight_scale_inv")
            if scale_name not in weights:
                continue
            updated_weights[name] = self._dequantize_fp8_block_scale_weight(
                weights[name], weights[scale_name]
            )
            updated_weights.pop(scale_name, None)
        return updated_weights

    def _pack_split_projections(self, weights: dict) -> dict:
        config = self.config.pretrained_config
        num_k_groups = config.linear_num_key_heads
        num_v_heads = config.linear_num_value_heads
        assert num_v_heads % num_k_groups == 0, (
            f"linear_num_value_heads ({num_v_heads}) must be divisible by "
            f"linear_num_key_heads ({num_k_groups})"
        )

        grouped_weights = defaultdict(dict)
        packed_weights = {}

        for name, tensor in weights.items():
            match = self._SPLIT_PROJ_PATTERN.match(name)
            if match is None:
                packed_weights[name] = tensor
                continue
            prefix, projection_name, suffix = match.groups()
            grouped_weights[(prefix, suffix)][projection_name] = tensor

        expected_q = config.linear_key_head_dim * config.linear_num_key_heads
        expected_v = config.linear_value_head_dim * config.linear_num_value_heads
        expected_ba = config.linear_num_value_heads

        for (prefix, suffix), tensors in grouped_weights.items():
            # `weight_scale_inv` (loaded by FP8BlockScalesLinearMethod) is the
            # only suffix stored in 2D-block format [ceil(out/block_size), ceil(in/block_size)];
            # weight/bias/weight_scale all keep out_features as their leading dim.
            if suffix == "weight_scale_inv":
                row_q = math.ceil(expected_q / _FP8_2D_BLOCK_SIZE)
                row_v = math.ceil(expected_v / _FP8_2D_BLOCK_SIZE)
                row_ba = math.ceil(expected_ba / _FP8_2D_BLOCK_SIZE)
                split_packed_qkv = self._split_qkv_scale_tensor
            else:
                row_q, row_v, row_ba = expected_q, expected_v, expected_ba
                split_packed_qkv = self._split_qkv_tensor

            qkvz_keys = {"qkv", "q", "k", "v", "z"} & tensors.keys()
            if qkvz_keys:
                if "qkv" in tensors:
                    missing = {"qkv", "z"} - tensors.keys()
                    assert not missing, (
                        f"Missing split projections {sorted(missing)} for {prefix}.{suffix}"
                    )
                    q_tensor, k_tensor, v_tensor = split_packed_qkv(
                        tensors["qkv"], expected_q, expected_v
                    )
                else:
                    missing = {"q", "k", "v", "z"} - tensors.keys()
                    assert not missing, (
                        f"Missing split projections {sorted(missing)} for {prefix}.{suffix}"
                    )
                    q_tensor = tensors["q"]
                    k_tensor = tensors["k"]
                    v_tensor = tensors["v"]

                assert q_tensor.shape[0] == row_q
                assert k_tensor.shape[0] == row_q
                assert v_tensor.shape[0] == row_v
                assert tensors["z"].shape[0] == row_v

                packed_name = f"{prefix}.in_proj_qkvz.{suffix}"
                assert packed_name not in packed_weights, (
                    f"Packed projection {packed_name} already exists"
                )
                packed_weights[packed_name] = self._pack_projection_tensor(
                    [q_tensor, k_tensor, v_tensor, tensors["z"]], num_k_groups
                )

            ba_keys = {"b", "a"} & tensors.keys()
            if ba_keys:
                missing = {"b", "a"} - tensors.keys()
                assert not missing, (
                    f"Missing split projections {sorted(missing)} for {prefix}.{suffix}"
                )

                if suffix == "weight":
                    assert tensors["b"].shape[0] == row_ba
                    assert tensors["a"].shape[0] == row_ba

                packed_name = f"{prefix}.in_proj_ba.{suffix}"
                assert packed_name not in packed_weights, (
                    f"Packed projection {packed_name} already exists"
                )
                packed_weights[packed_name] = self._pack_projection_tensor(
                    [tensors["b"], tensors["a"]], num_k_groups
                )

        return packed_weights

    def _remap_dense_mlp_weights(self, weights: dict) -> dict:
        """Insert extra .mlp. into dense MLP weight keys.

        _DenseMlpAdapter wraps GatedMLP as self.mlp, so named_modules() reports
        model|mtp.layers.N.mlp.mlp.* while the checkpoint has model|mtp.layers.N.mlp.*.
        """
        remapped_weights = {}
        for name, tensor in weights.items():
            match = self._DENSE_MLP_PATTERN.match(name)
            if match:
                remapped_weights[f"{match.group(1)}.mlp.{match.group(2)}{match.group(3)}"] = tensor
            else:
                remapped_weights[name] = tensor
        return remapped_weights

    def preprocess_weights(self, weights: dict) -> dict:
        quant_algo = self.config.quant_config.quant_algo

        normalized_weights = self._normalize_weight_names(weights)
        normalized_weights, is_modelopt_pb_wo = self._normalize_scale_names(
            normalized_weights, quant_algo
        )

        packed_weights = self._pack_split_projections(normalized_weights)
        if quant_algo == QuantAlgo.FP8_BLOCK_SCALES and not is_modelopt_pb_wo:
            packed_weights = self._dequantize_linear_attn_fp8_qkvz(packed_weights)

        if not getattr(self.config.pretrained_config, "num_experts", 0):
            packed_weights = self._remap_dense_mlp_weights(packed_weights)

        return super().preprocess_weights(packed_weights)
