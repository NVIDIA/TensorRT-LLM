import torch

import tensorrt_llm.logger as logger
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.utils import split


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

        is_nvfp4 = self.config.quant_config.quant_algo == "NVFP4"
        n_groups = config.n_groups
        d_state = config.ssm_state_size
        nheads = config.mamba_num_heads

        new_weights = {}
        for name, _ in weights.items():
            key = name

            # change backbone root name to model
            if "backbone" in key:
                key = key.replace("backbone", "model")

            # change embedding layer to embed_token
            if "embeddings" in key:
                key = key.replace("embeddings", "embed_tokens")

            # MTP layers are stored as mtp.layers.0.xxx (sublayer 0, Attention) and mtp.layers.1.xxx (sublayer 1, MoE)
            if "mtp.layers." in key:
                import re
                match = re.match(r'mtp\.layers\.(\d+)\.(.*)', key)
                if match:
                    sublayer_idx, rest = match.groups()
                    key = f"model.layers.{config.num_hidden_layers}.layers.{sublayer_idx}.{rest}"
                else:
                    logger.error(f"Failed to match MTP pattern for: {name}")

            if "A_log" in key:
                key = key.replace("A_log", "A")

            if ("mixer.in_proj" in key
                    or "mixer.out_proj" in key) and "_scale" in key:
                # Special handing for nvfp4 Mamba2 mixer in_proj.weight_scale.
                if is_nvfp4 and "in_proj.weight_scale_2" not in key and "in_proj.weight_scale" in key:
                    new_weights[key] = _split_mamba2_mixer_in_proj(
                        weights[name])
                else:
                    new_weights[key] = weights[name]
            elif "A" in key:
                w = split(weights[name], tp_size, tp_rank)
                w = w.to(torch.float32)
                w = -torch.exp(w)
                new_weights[key] = w
            elif "D" in key:
                w = split(weights[name], tp_size, tp_rank)
                w = w.to(torch.float32)
                new_weights[key] = w
            elif "dt_bias" in key:
                w = split(weights[name], tp_size, tp_rank)
                w = w.to(torch.float32)
                new_weights[key] = w
            elif "mixer.in_proj" in key:
                new_weights[key] = _split_mamba2_mixer_in_proj(weights[name])
            elif "conv1d" in key:
                w = weights[name]
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
                w = split(weights[name], tp_size, tp_rank)
                new_weights[key] = w
            # Remap MoE expert weights.
            elif "mixer.experts." in key:
                if self.config.moe_backend == 'VANILLA':
                    new_weights[key] = weights[name]
                else:
                    if "up_proj" in key:
                        w1_key = key.replace("up_proj", "w1")
                        w3_key = key.replace("up_proj", "w3")
                        # Don't need to handle with input_scale and weight_scale_2 since they are scalar for fp8 and nvfp4 models.
                        if "input_scale" in key or "weight_scale_2" in key:
                            new_weights[w3_key] = weights[name]
                            new_weights[w1_key] = weights[name]
                        elif "weight_scale" in key:
                            # NVFP4 case.
                            if weights[name].shape:
                                # w3 weight (gate_proj) scale should be empty for Nemotron-H MoE model.
                                # Use [:0] to keep the same input dimension as the other weights.
                                # The w3 weight_scale shape should be [0, input_dim].
                                new_weights[w3_key] = weights[name][:0]
                                new_weights[w1_key] = weights[name]
                            # FP8 case.
                            else:
                                new_weights[w3_key] = weights[name]
                                new_weights[w1_key] = weights[name]
                        else:
                            # w3 weight (gate_proj) should be empty for Nemotron-H MoE model.
                            # Use [:0] to keep the same input dimension as the other weights.
                            # The w3 weight shape should be [0, input_dim].
                            new_weights[w3_key] = weights[name][:0]
                            new_weights[w1_key] = weights[name]
                    elif "down_proj" in key:
                        key = key.replace("down_proj", "w2")
                        new_weights[key] = weights[name]
                    else:
                        raise ValueError(f"Unknown MoE weight: {key}")
            else:
                new_weights[key] = weights[name]

        return new_weights
