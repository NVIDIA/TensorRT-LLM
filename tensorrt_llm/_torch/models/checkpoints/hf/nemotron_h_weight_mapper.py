import re

import torch

import tensorrt_llm.logger as logger
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.utils import split
from tensorrt_llm.quantization.mode import QuantAlgo


@register_mapper("HF", "NemotronHPuzzleForCausalLM")
@register_mapper("HF", "NemotronHForCausalLM")
class NemotronHHfWeightMapper(HfWeightMapper):

    def preprocess_weights(self, weights: dict) -> dict:
        config = self.config.pretrained_config
        tp_size = 1 if self.config.mapping.enable_attention_dp else self.config.mapping.tp_size
        tp_rank = self.config.mapping.tp_rank
        d_inner = config.mamba_head_dim * config.mamba_num_heads

        def _split_mamba2_mixer_in_proj(w: torch.Tensor) -> torch.Tensor:
            # Special handling for Mamba2 mixer in_proj.weights and scales.
            in_proj_z, in_proj_x, in_proj_b, in_proj_c, in_proj_dt = torch.split(
                w, [
                    d_inner, d_inner, n_groups * d_state, n_groups * d_state,
                    nheads
                ],
                dim=0)
            w = []
            for rank in range(tp_size):
                in_proj_z_rank = split(in_proj_z, tp_size, rank)
                in_proj_x_rank = split(in_proj_x, tp_size, rank)
                in_proj_b_rank = split(in_proj_b, tp_size, rank)
                in_proj_c_rank = split(in_proj_c, tp_size, rank)
                in_proj_dt_rank = split(in_proj_dt, tp_size, rank)
                y = torch.concat([
                    in_proj_z_rank, in_proj_x_rank, in_proj_b_rank,
                    in_proj_c_rank, in_proj_dt_rank
                ])
                w.append(y)
            w = torch.concat(w).contiguous()
            return w

        quant_algo = getattr(self.config.quant_config, "quant_algo", None)
        is_nvfp4 = quant_algo in (QuantAlgo.NVFP4, QuantAlgo.W4A16_NVFP4,
                                  "NVFP4", "W4A16_NVFP4")
        n_groups = config.n_groups
        d_state = config.ssm_state_size
        nheads = config.mamba_num_heads

        def _invert_compressed_tensors_scale(value) -> torch.Tensor:
            value = value[...] if not isinstance(value, torch.Tensor) else value
            value = value.to(torch.float32)
            return torch.where(value > 0, value.reciprocal(),
                               torch.zeros_like(value)).contiguous()

        def _canonicalize_quant_weight(key: str, value):
            if key.endswith(".weight_packed"):
                return f"{key[:-len('.weight_packed')]}.weight", value
            if key.endswith(".weight_global_scale"):
                key = f"{key[:-len('.weight_global_scale')]}.weight_scale_2"
                return key, _invert_compressed_tensors_scale(value)
            if key.endswith(".input_global_scale"):
                key = f"{key[:-len('.input_global_scale')]}.input_scale"
                return key, _invert_compressed_tensors_scale(value)
            return key, value

        new_weights = {}
        for name, _ in weights.items():
            key = name
            value = weights[name]

            # change backbone root name to model
            if "backbone" in key:
                key = key.replace("backbone", "model")

            # change embedding layer to embed_token
            if "embeddings" in key:
                key = key.replace("embeddings", "embed_tokens")

            # MTP layers are stored as mtp.layers.0.xxx (sublayer 0, Attention) and mtp.layers.1.xxx (sublayer 1, MoE)
            if "mtp.layers." in key:
                match = re.match(r'mtp\.layers\.(\d+)\.(.*)', key)
                if match:
                    sublayer_idx, rest = match.groups()
                    key = f"model.layers.{config.num_hidden_layers}.layers.{sublayer_idx}.{rest}"
                else:
                    logger.error(f"Failed to match MTP pattern for: {name}")

            if "A_log" in key:
                key = key.replace("A_log", "A")

            key, value = _canonicalize_quant_weight(key, value)

            if ("mixer.in_proj" in key
                    or "mixer.out_proj" in key) and "_scale" in key:
                # Special handing for nvfp4 Mamba2 mixer in_proj.weight_scale.
                if is_nvfp4 and "in_proj.weight_scale_2" not in key and "in_proj.weight_scale" in key:
                    new_weights[key] = _split_mamba2_mixer_in_proj(value)
                else:
                    new_weights[key] = value
            elif "A" in key:
                w = split(value, tp_size, tp_rank)
                w = w.to(torch.float32)
                # Avoid extra temporaries: one fp32 cast, then in-place exp/neg.
                w.exp_()
                w.neg_()
                new_weights[key] = w
            elif "D" in key:
                w = split(value, tp_size, tp_rank)
                w = w.to(torch.float32)
                new_weights[key] = w
            elif "dt_bias" in key:
                w = split(value, tp_size, tp_rank)
                w = w.to(torch.float32)
                new_weights[key] = w
            elif "mixer.in_proj" in key:
                # Restrict the mamba2 in_proj split to the actual weight tensor.
                # NVFP4 checkpoints attach companion tensors (``input_scale``,
                # ``weight_scale``, ``weight_scale_2``, …) under ``mixer.in_proj.*``
                # — those are scalars / 1-D scales and must not go through the
                # Mamba2 split rearrangement.
                new_weights[key] = _split_mamba2_mixer_in_proj(value)
            elif "conv1d" in key:
                w = value
                # removing dim(1) because we are using Linear to store conv1d weights
                if "weight" in key:
                    w = w.squeeze(1)

                conv_x, conv_b, conv_c = torch.split(
                    w, [d_inner, n_groups * d_state, n_groups * d_state], dim=0)

                w = []
                for rank in range(tp_size):
                    conv_x_rank = split(conv_x, tp_size, rank)
                    conv_b_rank = split(conv_b, tp_size, rank)
                    conv_c_rank = split(conv_c, tp_size, rank)
                    y = torch.concat([conv_x_rank, conv_b_rank, conv_c_rank])
                    w.append(y)
                w = torch.concat(w).contiguous()
                new_weights[key] = w
            elif "mixer.norm.weight" in key:
                w = split(value, tp_size, tp_rank)
                new_weights[key] = w
            # Remap MoE expert weights.
            elif "mixer.experts." in key:
                if self.config.moe_backend == 'VANILLA':
                    new_weights[key] = value
                else:
                    # HF transformers 5.x exposes routed MoE experts as fused
                    # tensors stacked along dim 0 ([num_experts, ...]) under keys
                    # ``experts.up_proj`` and ``experts.down_proj`` (no per-expert
                    # index in the name). The on-disk safetensors checkpoint, by
                    # contrast, stores per-expert keys (``experts.{i}.up_proj``).
                    # The VANILLA FusedMoE loader expects per-expert keys, so
                    # unfuse the 3D HF format here before the standard rename.
                    val = value
                    m = re.match(r"(.*\.mixer\.experts)\.(up_proj|down_proj)$",
                                 key)
                    is_hf_fused = (m is not None
                                   and isinstance(val, torch.Tensor)
                                   and val.dim() == 3)
                    if is_hf_fused:
                        prefix, sub = m.group(1), m.group(2)
                        num_experts = val.shape[0]
                        if sub == "up_proj":
                            for i in range(num_experts):
                                w1_k = f"{prefix}.{i}.w1.weight"
                                w3_k = f"{prefix}.{i}.w3.weight"
                                # Nemotron-H MoE is non-gated; w3 (gate) is empty.
                                new_weights[w1_k] = val[i]
                                new_weights[w3_k] = val[i][:0]
                        else:  # down_proj
                            for i in range(num_experts):
                                w2_k = f"{prefix}.{i}.w2.weight"
                                new_weights[w2_k] = val[i]
                    elif "up_proj" in key:
                        w1_key = key.replace("up_proj", "w1")
                        w3_key = key.replace("up_proj", "w3")
                        # Per-tensor quantization parameters are shared by w1
                        # and the empty w3 projection.
                        if ("input_scale" in key or "weight_scale_2" in key
                                or "input_quantizer" in key
                                or "weight_quantizer" in key):
                            new_weights[w3_key] = value
                            new_weights[w1_key] = value
                        elif "weight_scale" in key:
                            # NVFP4 case.
                            if value.shape:
                                # w3 weight (gate_proj) scale should be empty for Nemotron-H MoE model.
                                # Use [:0] to keep the same input dimension as the other weights.
                                # The w3 weight_scale shape should be [0, input_dim].
                                new_weights[w3_key] = value[:0]
                                new_weights[w1_key] = value
                            # FP8 case.
                            else:
                                new_weights[w3_key] = value
                                new_weights[w1_key] = value
                        else:
                            # w3 weight (gate_proj) should be empty for Nemotron-H MoE model.
                            # Use [:0] to keep the same input dimension as the other weights.
                            # The w3 weight shape should be [0, input_dim].
                            new_weights[w3_key] = value[:0]
                            new_weights[w1_key] = value
                    elif "down_proj" in key:
                        key = key.replace("down_proj", "w2")
                        new_weights[key] = value
                    else:
                        raise ValueError(f"Unknown MoE weight: {key}")
            else:
                new_weights[key] = value

        return new_weights
