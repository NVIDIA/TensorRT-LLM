import re

import torch

import tensorrt_llm.logger as logger
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.utils import split

# E2M1 (NVFP4) 4-bit code -> value lookup table.
_E2M1_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0,
    -4.0, -6.0
],
                            dtype=torch.float32)


def _dequant_nvfp4_linear(weight_uint8: torch.Tensor,
                          block_scale_fp8: torch.Tensor,
                          global_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a 2-D weight-only NVFP4 weight to bf16.

    ``weight_uint8`` is ``[M, K/2]`` (two e2m1 nibbles per byte), ``block_scale_fp8``
    is ``[M, K/16]`` (fp8_e4m3, un-swizzled linear layout as stored by modelopt),
    and ``global_scale`` is a scalar. Runs on GPU when available (the per-element
    LUT gather is far cheaper there than on CPU).
    """
    device = 'cuda' if torch.cuda.is_available() else weight_uint8.device
    w = weight_uint8.to(device)
    s1 = block_scale_fp8.to(device=device, dtype=torch.float32)
    s2 = global_scale.to(device=device, dtype=torch.float32).reshape([])
    lut = _E2M1_VALUES.to(device)

    K = w.shape[-1] * 2
    high = (w >> 4) & 0x0F
    low = w & 0x0F
    vals = torch.empty(*w.shape[:-1], K, dtype=torch.float32, device=device)
    vals[..., 0::2] = lut[low.long()]
    vals[..., 1::2] = lut[high.long()]

    scale = (s1 * s2).unsqueeze(-1)
    vals = vals.view(*w.shape[:-1], K // 16, 16) * scale
    return vals.view(*w.shape[:-1], K).to(torch.bfloat16).cpu()


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
                # Avoid extra temporaries: one fp32 cast, then in-place exp/neg.
                w.exp_()
                w.neg_()
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
                # Restrict the mamba2 in_proj split to the actual weight tensor.
                # NVFP4 checkpoints attach companion tensors (``input_scale``,
                # ``weight_scale``, ``weight_scale_2``, …) under ``mixer.in_proj.*``
                # — those are scalars / 1-D scales and must not go through the
                # Mamba2 split rearrangement.
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
                    # HF transformers 5.x exposes routed MoE experts as fused
                    # tensors stacked along dim 0 ([num_experts, ...]) under keys
                    # ``experts.up_proj`` and ``experts.down_proj`` (no per-expert
                    # index in the name). The on-disk safetensors checkpoint, by
                    # contrast, stores per-expert keys (``experts.{i}.up_proj``).
                    # The VANILLA FusedMoE loader expects per-expert keys, so
                    # unfuse the 3D HF format here before the standard rename.
                    val = weights[name]
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
                        # Don't need to handle with input_scale and weight_scale_2 since they are scalar for fp8 and nvfp4 models.
                        if "input_scale" in key or "weight_scale_2" in key or "input_quantizer" in key or "weight_quantizer" in key:
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
            # lm_head is kept in bf16 in TRT-LLM (LMHead is built unquantized,
            # see modeling_utils). If the checkpoint quantized it as weight-only
            # NVFP4, dequantize FP4 -> bf16 here (numerically equivalent to the
            # W4A16 dequant-then-GEMM path) and drop the standalone scale tensors.
            elif key.startswith("lm_head.") and "weight_scale" in key:
                continue
            elif key == "lm_head.weight" and "lm_head.weight_scale" in weights:
                new_weights[key] = _dequant_nvfp4_linear(
                    weights["lm_head.weight"], weights["lm_head.weight_scale"],
                    weights["lm_head.weight_scale_2"])
            else:
                new_weights[key] = weights[name]

        return new_weights
