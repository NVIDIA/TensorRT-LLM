import re
from collections import defaultdict

import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.qwen2_moe_weight_mapper import \
    Qwen2MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.utils import split


@register_mapper("HF", "Qwen3NextForCausalLM")
class Qwen3NextHfWeightMapper(Qwen2MoeHfWeightMapper):

    _GDN_INPUT_PROJ_PATTERN = re.compile(
        r"^(.*\.linear_attn)\.in_proj_(qkvz|ba)\.(.+)$")

    def _combine_gdn_input_projections(self, weights: dict) -> dict:
        """Pack grouped QKVZ/BA checkpoint tensors in consumer order.

        Attention-DP uses TP=1 for GDN, so the combined Linear can emit
        contiguous feature groups in the order consumed by conv, recurrent
        GDN, and gated RMSNorm: [Q, K, V, Z, B, A].
        """
        config = self.config.pretrained_config
        num_k_heads = config.linear_num_key_heads
        num_v_heads = config.linear_num_value_heads
        heads_ratio = num_v_heads // num_k_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim
        qkvz_group_dim = head_k_dim * 2 + heads_ratio * head_v_dim * 2
        ba_group_dim = heads_ratio * 2
        expected_qkvz = num_k_heads * qkvz_group_dim
        expected_ba = num_k_heads * ba_group_dim

        grouped = defaultdict(dict)
        combined_weights = {}
        for name, tensor in weights.items():
            match = self._GDN_INPUT_PROJ_PATTERN.match(name)
            if match is None:
                combined_weights[name] = tensor
                continue
            prefix, projection, suffix = match.groups()
            grouped[(prefix, suffix)][projection] = tensor

        for (prefix, suffix), tensors in grouped.items():
            if tensors.keys() != {"qkvz", "ba"}:
                raise ValueError(
                    f"Expected both QKVZ and BA tensors for {prefix}.{suffix}, "
                    f"got {sorted(tensors)}")

            qkvz = tensors["qkvz"]
            ba = tensors["ba"]
            combined_name = f"{prefix}.in_proj_qkvzba.{suffix}"

            # Scalar/per-tensor metadata is shared by the two projections. It
            # cannot be row-reordered, so retain one copy after validating it.
            if (qkvz.ndim == 0 or ba.ndim == 0 or qkvz.shape[0] != expected_qkvz
                    or ba.shape[0] != expected_ba):
                if qkvz.shape != ba.shape or not torch.equal(qkvz, ba):
                    raise ValueError(
                        f"Cannot combine non-row GDN projection metadata "
                        f"{prefix}.{suffix}: QKVZ shape={tuple(qkvz.shape)}, "
                        f"BA shape={tuple(ba.shape)}")
                combined_weights[combined_name] = qkvz
                continue

            if qkvz.shape[1:] != ba.shape[1:]:
                raise ValueError(
                    f"GDN projection trailing shapes do not match for "
                    f"{prefix}.{suffix}: {tuple(qkvz.shape)} vs {tuple(ba.shape)}"
                )

            trailing_shape = qkvz.shape[1:]
            qkvz = qkvz.reshape(num_k_heads, qkvz_group_dim, *trailing_shape)
            ba = ba.reshape(num_k_heads, ba_group_dim, *trailing_shape)

            q_end = head_k_dim
            k_end = q_end + head_k_dim
            v_end = k_end + heads_ratio * head_v_dim
            z_end = v_end + heads_ratio * head_v_dim
            q = qkvz[:, :q_end].reshape(-1, *trailing_shape)
            k = qkvz[:, q_end:k_end].reshape(-1, *trailing_shape)
            v = qkvz[:, k_end:v_end].reshape(-1, *trailing_shape)
            z = qkvz[:, v_end:z_end].reshape(-1, *trailing_shape)
            b = ba[:, :heads_ratio].reshape(-1, *trailing_shape)
            a = ba[:, heads_ratio:].reshape(-1, *trailing_shape)
            combined_weights[combined_name] = torch.cat((q, k, v, z, b, a),
                                                        dim=0).contiguous()

        return combined_weights

    def should_skip_module(self, module_name: str) -> bool:
        if module_name.startswith("draft_model"):
            return True
        return super().should_skip_module(module_name)

    def _duplicate_kv_weights(self, module: nn.Module, new_name: str,
                              weights: dict):
        tensors_to_duplicate = ["weight", "bias"]
        if module.quant_config.quant_mode.has_nvfp4():
            tensors_to_duplicate.append("weight_scale")
        if module.quant_config.quant_mode.has_fp8_block_scales():
            tensors_to_duplicate.append("weight_scale_inv")

        if new_name in ['k_proj', 'v_proj']:
            num_kv_heads_list = [self._num_kv_heads
                                 ] * len(weights) if isinstance(
                                     self._num_kv_heads,
                                     int) else self._num_kv_heads
            processed_weights = {
                k:
                self._duplicate_kv(weight=v[:],
                                   num_kv_heads=num_kv_heads_list[i],
                                   tensor_parallel_size=self._tp_size)
                if k in tensors_to_duplicate else v
                for i, (k, v) in enumerate(weights.items())
            }
            return processed_weights

        return weights

    def preprocess_weights(self, weights: dict) -> dict:
        config = self.config.pretrained_config
        tp_size = self.config.mapping.tp_size
        tp_rank = self.config.mapping.tp_rank
        mtp_layer_offset = config.num_hidden_layers

        if self.config.mapping.enable_attention_dp:
            tp_size = 1
        # linear_num_value_heads = config.linear_num_value_heads
        # linear_num_key_heads = config.linear_num_key_heads
        # linear_key_head_dim = config.linear_key_head_dim
        # linear_value_head_dim = config.linear_value_head_dim
        linear_key_dim = config.linear_key_head_dim * config.linear_num_key_heads  # 16 * 128
        linear_value_dim = config.linear_value_head_dim * config.linear_num_value_heads  # 32 * 128

        mtp_mapping = {
            "mtp.fc": "fc",
            "mtp.norm": "shared_head.norm",
            "mtp.pre_fc_norm_embedding": "pre_fc_norm_embedding",
            "mtp.pre_fc_norm_hidden": "pre_fc_norm_hidden",
        }

        new_weights = {}
        for name, _ in weights.items():
            key = name

            if key.startswith("mtp.layers."):
                _, _, mtp_layer_idx, module_name = key.split(".", 3)
                key = (f"model.layers.{mtp_layer_offset + int(mtp_layer_idx)}."
                       f"{module_name}")
            elif key.startswith("mtp."):
                for mtp_prefix, trtllm_name in mtp_mapping.items():
                    if key.startswith(mtp_prefix):
                        suffix = key[len(mtp_prefix):]
                        key = f"model.layers.{mtp_layer_offset}.{trtllm_name}{suffix}"
                        break

            if "A_log" in key:
                w = split(weights[name], tp_size, tp_rank)
                w = w.to(torch.float32)
                new_weights[key] = w
            elif "dt_bias" in key:
                w = split(weights[name], tp_size, tp_rank)
                w = w.to(torch.float32)
                new_weights[key] = w
            elif "in_proj" in key:
                # Don't need to split in_proj weight based on the implementation of reference.
                # Need to know the reason.
                new_weights[key] = weights[name]
            elif "conv1d" in key:
                w = weights[name]
                # removing dim(1) because we are using Linear to store conv1d weights
                if "weight" in key:
                    w = w.squeeze(1)

                conv_q, conv_k, conv_v = torch.split(
                    w, [linear_key_dim, linear_key_dim, linear_value_dim],
                    dim=0)

                w = []
                for rank in range(tp_size):
                    conv_q_rank = split(conv_q, tp_size, rank)
                    conv_k_rank = split(conv_k, tp_size, rank)
                    conv_v_rank = split(conv_v, tp_size, rank)
                    y = torch.concat([conv_q_rank, conv_k_rank, conv_v_rank])
                    w.append(y)
                w = torch.concat(w).contiguous()
                new_weights[key] = w
            else:
                new_weights[key] = weights[name]

        if self.config.mapping.enable_attention_dp:
            new_weights = self._combine_gdn_input_projections(new_weights)

        return new_weights
