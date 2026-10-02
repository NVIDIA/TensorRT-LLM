# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import Optional

import torch
from torch import nn

import tensorrt_llm.logger as logger
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.mamba.mamba2_tp import Mamba2TpShard
from tensorrt_llm._torch.utils import split


@register_mapper("HF", "NemotronHPuzzleForCausalLM")
@register_mapper("HF", "NemotronHForCausalLM")
class NemotronHHfWeightMapper(HfWeightMapper):

    def preprocess_weights(self, weights: dict) -> dict:
        config = self.config.pretrained_config
        tp_size = 1 if self.config.mapping.enable_attention_dp else self.config.mapping.tp_size
        tp_rank = self.config.mapping.tp_rank

        n_groups = config.n_groups
        d_state = config.ssm_state_size
        nheads = config.mamba_num_heads
        d_inner = config.mamba_head_dim * config.mamba_num_heads
        # Full in_proj out_features = concat([z, x, B, C, dt]). Only its
        # per-output-row block scale spans this dim 0 and takes the same
        # structured split as the weight; per-tensor scalars (weight_scale_2,
        # input_scale) do not and are left alone.
        d_in_proj = 2 * d_inner + 2 * n_groups * d_state + nheads
        # Per-rank row layout, including replicated B/C groups when
        # tp_size > n_groups (see mamba2_tp.py).
        shard = Mamba2TpShard(tp_size=tp_size,
                              nheads=nheads,
                              n_groups=n_groups,
                              head_dim=config.mamba_head_dim,
                              d_state=d_state)

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

            if "mixer.in_proj" in key and "_scale" in key:
                if self._num_rows(weights[name]) == d_in_proj:
                    new_weights[key] = shard.rearrange_in_proj_rows(
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
                new_weights[key] = shard.rearrange_in_proj_rows(weights[name])
            elif "conv1d" in key:
                w = weights[name]
                # removing dim(1) because we are using Linear to store conv1d weights
                if "weight" in key:
                    w = w.squeeze(1)
                new_weights[key] = shard.rearrange_conv1d_rows(w)
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
            else:
                new_weights[key] = weights[name]

        return new_weights

    @staticmethod
    def _num_rows(tensor) -> Optional[int]:
        """Size of dim 0 (the output-channel axis), or None for a scalar.

        Works for materialized tensors and lazy safetensors slices. A weight
        scale that shares this axis with the weight (NVFP4 / MXFP8 / FP8 block
        scale, FP8 rowwise) is per-output-channel and must undergo the same TP
        transform as the weight; a per-tensor scalar scale has no such axis.
        """
        shape = tensor.get_shape() if hasattr(tensor,
                                              "get_shape") else tensor.shape
        return shape[0] if shape else None

    def _duplicate_kv_weights(self, module: nn.Module, new_name: str,
                              weights: dict):
        # Override of the base NVFP4-only rule: NemotronH attention may be
        # FP8/MXFP8 (MIXED_PRECISION checkpoints), so duplicate ANY
        # per-output-channel weight_scale (one that shares dim 0 with the
        # weight) alongside the replicated kv weight, not just the NVFP4 case.
        if new_name not in ('k_proj', 'v_proj'):
            return weights

        num_kv_heads = self._num_kv_heads
        duplicated_keys = ["weight", "bias"]
        weight, scale = weights.get("weight"), weights.get("weight_scale")
        if (weight is not None and scale is not None
                and self._num_rows(scale) == self._num_rows(weight)):
            duplicated_keys.append("weight_scale")

        return {
            k:
            self._duplicate_kv(weight=v[:],
                               num_kv_heads=num_kv_heads,
                               tensor_parallel_size=self._tp_size)
            if k in duplicated_keys else v
            for k, v in weights.items()
        }
