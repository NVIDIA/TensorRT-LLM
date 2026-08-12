import math

import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.qwen2_moe_weight_mapper import \
    Qwen2MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.utils import split

# 2D block edge of FP8 block-scale tensors (weight_scale_inv), matching
# FP8BlockScalesLinearMethod.
_FP8_BLOCK_SIZE = 128


def grouped_to_dense_in_proj_qkvz_perm(num_k_heads: int, head_k_dim: int,
                                       num_v_heads: int, head_v_dim: int,
                                       tp_size: int) -> torch.Tensor:
    """Row permutation from grouped-interleaved in_proj_qkvz to dense [Q|K|V|Z].

    HF GDN checkpoints pack in_proj_qkvz rows per key-head group as
    [q_g | k_g | v_g | z_g]. The GDN mixer consumes the projection as plain
    column slices (mixed_qkv = [Q|K|V], then z), which requires the dense
    row order [all Q | all K | all V | all Z]. Rows are permuted per TP-rank
    chunk so the column-parallel contiguous row split still hands each rank
    its own heads.
    """
    assert num_k_heads % tp_size == 0 and num_v_heads % tp_size == 0
    ng = num_k_heads // tp_size
    ratio = num_v_heads // num_k_heads
    dk, rdv = head_k_dim, ratio * head_v_dim
    group_size = 2 * dk + 2 * rdv
    base = torch.arange(ng).unsqueeze(1) * group_size
    q = (base + torch.arange(dk)).flatten()
    k = (base + dk + torch.arange(dk)).flatten()
    v = (base + 2 * dk + torch.arange(rdv)).flatten()
    z = (base + 2 * dk + rdv + torch.arange(rdv)).flatten()
    rank_perm = torch.cat([q, k, v, z])
    rank_rows = ng * group_size
    return torch.cat([rank_perm + rank * rank_rows for rank in range(tp_size)])


def grouped_to_dense_in_proj_ba_perm(num_k_heads: int, num_v_heads: int,
                                     tp_size: int) -> torch.Tensor:
    """Row permutation from grouped-interleaved in_proj_ba to dense [b|a]."""
    assert num_k_heads % tp_size == 0 and num_v_heads % tp_size == 0
    ng = num_k_heads // tp_size
    ratio = num_v_heads // num_k_heads
    base = torch.arange(ng).unsqueeze(1) * (2 * ratio)
    b = (base + torch.arange(ratio)).flatten()
    a = (base + ratio + torch.arange(ratio)).flatten()
    rank_perm = torch.cat([b, a])
    rank_rows = ng * 2 * ratio
    return torch.cat([rank_perm + rank * rank_rows for rank in range(tp_size)])


def _rows_to_scale_block_perm(perm: torch.Tensor, block: int) -> torch.Tensor:
    """Derive the permutation of `block`-row scale blocks implied by a row
    permutation, requiring the row permutation to move whole aligned blocks."""
    if perm.numel() <= block:
        # A single scale block covers every row: any within-block row
        # permutation leaves the (one-row) scale tensor unchanged.
        return torch.zeros(1, dtype=torch.long)
    assert perm.numel() % block == 0, (
        f"row permutation of {perm.numel()} rows is not divisible by the "
        f"scale block size {block}")
    blocks = perm.view(-1, block)
    firsts = blocks[:, :1]
    assert torch.equal(blocks, firsts + torch.arange(block)) and \
        (firsts % block == 0).all(), (
        "in_proj row permutation must move whole aligned scale blocks to "
        "permute block-scale tensors; head dims must be multiples of "
        f"{block}")
    return firsts.squeeze(1) // block


def _permute_rows(tensor: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Reorder dim-0 rows of a (possibly lazy safetensors) tensor.

    Goes through a uint8 view so packed/quantized dtypes (float8, packed FP4
    uint8) reorder identically to plain floats.
    """
    t = tensor[...] if not isinstance(tensor, torch.Tensor) else tensor
    squeeze = t.dim() == 1
    if squeeze:
        t = t.unsqueeze(1)
    u8 = t.contiguous().view(torch.uint8)
    out = u8[perm].view(t.dtype)
    return out.squeeze(1) if squeeze else out


@register_mapper("HF", "Qwen3NextForCausalLM")
class Qwen3NextHfWeightMapper(Qwen2MoeHfWeightMapper):

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

    def _permute_in_proj_to_dense(self, key: str, tensor, tp_size: int):
        """Reorder one in_proj_qkvz / in_proj_ba tensor to the dense layout.

        Row tensors (weight of any dtype including packed FP4, per-row
        scales, bias) are permuted directly; FP8 2D-block ``weight_scale_inv``
        tensors are permuted at scale-block granularity; scalar per-tensor
        scales pass through unchanged.
        """
        config = self.config.pretrained_config
        if ".in_proj_qkvz." in key:
            perm = grouped_to_dense_in_proj_qkvz_perm(
                config.linear_num_key_heads, config.linear_key_head_dim,
                config.linear_num_value_heads, config.linear_value_head_dim,
                tp_size)
        elif ".in_proj_ba." in key:
            perm = grouped_to_dense_in_proj_ba_perm(
                config.linear_num_key_heads, config.linear_num_value_heads,
                tp_size)
        else:
            return tensor

        t = tensor[...] if not isinstance(tensor, torch.Tensor) else tensor
        if t.dim() == 0 or t.numel() == 1:
            return t
        rows = perm.numel()
        if t.shape[0] == rows:
            return _permute_rows(t, perm)
        if key.endswith("weight_scale_inv") and \
                t.shape[0] == math.ceil(rows / _FP8_BLOCK_SIZE):
            return _permute_rows(
                t, _rows_to_scale_block_perm(perm, _FP8_BLOCK_SIZE))
        raise ValueError(
            f"Cannot map {key} with shape {tuple(t.shape)} onto the dense "
            f"in_proj layout ({rows} rows expected)")

    def preprocess_weights(self,
                           weights: dict,
                           allow_partial_loading: bool = False) -> dict:
        _ = allow_partial_loading
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
                # in_proj stays unsplit here (the column-parallel Linear
                # splits contiguous row chunks itself); rows are reordered
                # from the checkpoint's grouped-interleaved layout to the
                # dense per-rank [Q|K|V|Z] / [b|a] layout the GDN mixer
                # slices at runtime.
                new_weights[key] = self._permute_in_proj_to_dense(
                    key, weights[name], tp_size)
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

        return new_weights
