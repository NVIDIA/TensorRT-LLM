import torch

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_nemotron_h import split
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "NemotronHForCausalLM")
class NemotronHHfWeightMapper(HfWeightMapper):

    def preprocess_weights(self, weights: dict) -> dict:
        config = self.config.pretrained_config
        tp_size = self.config.mapping.tp_size
        tp_rank = self.config.mapping.tp_rank
        d_inner = config.mamba_head_dim * config.mamba_num_heads

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

            if "A_log" in key:
                key = key.replace("A_log", "A")

            if "_scale" in key:
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
                w = weights[name]
                in_proj_z, in_proj_x, in_proj_b, in_proj_c, in_proj_dt = torch.split(
                    w, [
                        d_inner, d_inner, n_groups * d_state,
                        n_groups * d_state, nheads
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
                new_weights[key] = w
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
            else:
                new_weights[key] = weights[name]

        return new_weights
