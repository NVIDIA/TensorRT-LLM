# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from tensorrt_llm._torch.distributed.ops import AllReduce
from tensorrt_llm._torch.modules.linear import (
    Linear,
    TensorParallelMode,
    WeightMode,
    WeightsLoadingConfig,
)
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm.mapping import Mapping


class FluxJointAttnMLPProj(nn.Module):
    """Output projection that accepts split attn + mlp inputs.

    At TP=1, equivalent to a single Linear over cat([attn, mlp]).
    At TP>1, splits into two ROW-parallel Linears (one per input) so the
    preceding attention and MLP branches can stay sharded — avoiding an
    allgather before the projection.  Bias is added after the allreduce.

    Named ``proj_out`` on the parent block so that ``filter_weights`` finds
    the checkpoint keys ``proj_out.weight`` / ``proj_out.bias`` automatically.
    """

    def __init__(
        self,
        attn_dim: int,
        mlp_dim: int,
        out_dim: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        quant_config=None,
        skip_create_weights_in_init: bool = False,
        force_dynamic_quantization: bool = False,
        config: Optional[DiffusionModelConfig] = None,
    ):
        super().__init__()
        mapping = config.mapping if config else None
        self.tp_size = getattr(mapping, "tp_size", 1)
        self.tp_rank = getattr(mapping, "tp_rank", 0)
        self.attn_dim = attn_dim
        self.has_bias = bias

        if self.tp_size == 1:
            self.proj = Linear(
                attn_dim + mlp_dim,
                out_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=skip_create_weights_in_init,
                force_dynamic_quantization=force_dynamic_quantization,
                reduce_output=False,
            )
        else:
            self.attn_proj = Linear(
                attn_dim,
                out_dim,
                bias=False,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=skip_create_weights_in_init,
                force_dynamic_quantization=force_dynamic_quantization,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.ROW,
                reduce_output=False,
            )
            self.mlp_proj = Linear(
                mlp_dim,
                out_dim,
                bias=False,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=skip_create_weights_in_init,
                force_dynamic_quantization=force_dynamic_quantization,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.ROW,
                reduce_output=False,
            )
            self.allreduce = AllReduce(mapping, strategy=config.allreduce_strategy)
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_dim, dtype=dtype))

    def forward(self, attn_out: torch.Tensor, mlp_out: torch.Tensor) -> torch.Tensor:
        if self.tp_size <= 1:
            return self.proj(torch.cat([attn_out, mlp_out], dim=-1))
        out = self.allreduce(self.attn_proj(attn_out) + self.mlp_proj(mlp_out))
        if self.has_bias:
            out = out + self.bias
        return out

    def load_weights(self, weight_dict: Dict[str, torch.Tensor], loader: DynamicLinearWeightLoader):
        """Load from checkpoint proj_out.weight / proj_out.bias.

        At TP=1, passes through to the inner Linear.
        At TP>1, splits the weight along the input dim (columns) into
        attn and mlp portions, then loads each as a vanilla ROW Linear.
        """
        for sub in self.modules():
            if callable(getattr(sub, "create_weights", None)):
                sub.create_weights()

        if self.tp_size == 1:
            loader.load_linear_weights(self.proj, "proj_out", [weight_dict])
        else:
            W = weight_dict["weight"]  # [out_dim, attn_dim + mlp_dim]
            W_attn = W[:, : self.attn_dim]
            W_mlp = W[:, self.attn_dim :]

            for sub_module, sub_weight in [(self.attn_proj, W_attn), (self.mlp_proj, W_mlp)]:
                loader.load_linear_weights(sub_module, "proj_out", [{"weight": sub_weight}])

            if self.has_bias and "bias" in weight_dict:
                self.bias.data.copy_(weight_dict["bias"].to(self.bias.dtype))


class FluxJointQKVMLPProj(nn.Module):
    """Input projection producing QKV + MLP gate/up from a single input.

    At TP=1: single Linear, output split into [qkv, mlp_gate_up].
    At TP>1: two column-parallel Linears (qkv_proj + mlp_proj) so each
    can shard independently — QKV shards by heads, MLP shards by
    intermediate dim.

    HF checkpoint stores a single fused weight ``to_qkv_mlp_proj.weight``
    with layout [Q | K | V | gate | up] along the output dimension.

    All dimension arguments are FULL (pre-TP). Column-parallel Linears
    handle the TP sharding internally. The ``forward`` method returns
    tensors with local (post-TP) sizes.
    """

    def __init__(
        self,
        in_dim: int,
        q_dim: int,
        kv_dim: int,
        mlp_dim: int,
        bias: bool = False,
        dtype: torch.dtype = None,
        quant_config=None,
        skip_create_weights_in_init: bool = False,
        force_dynamic_quantization: bool = False,
        mapping: Optional[Mapping] = None,
    ):
        super().__init__()

        self.tp_size = mapping.tp_size if mapping else 1

        # Store full (pre-TP) dims for weight loading (splitting checkpoint weight)
        self.full_q_dim = q_dim
        self.full_kv_dim = kv_dim
        self.full_qkv_dim = q_dim + 2 * kv_dim
        self.full_mlp_dim = mlp_dim
        self.mlp_hidden_dim = mlp_dim // 2  # single gate or up dim

        if self.tp_size == 1:
            self.proj = Linear(
                in_dim,
                q_dim + 2 * kv_dim + mlp_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=skip_create_weights_in_init,
                force_dynamic_quantization=force_dynamic_quantization,
                reduce_output=False,
            )
            self.local_qkv_dim = q_dim + 2 * kv_dim
            self.local_mlp_dim = mlp_dim
        else:
            # QKV: column-parallel with fused Q/K/V sharding
            self.qkv_proj = Linear(
                in_dim,
                q_dim + 2 * kv_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=skip_create_weights_in_init,
                force_dynamic_quantization=force_dynamic_quantization,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_QKV_LINEAR,
                ),
                fused_weight_shard_indices_mapping={
                    "q": (0, q_dim),
                    "k": (q_dim, kv_dim),
                    "v": (q_dim + kv_dim, kv_dim),
                },
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                reduce_output=False,
            )
            # MLP gate+up: column-parallel with fused gate/up sharding
            self.mlp_proj = Linear(
                in_dim,
                mlp_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=skip_create_weights_in_init,
                force_dynamic_quantization=force_dynamic_quantization,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_GATE_UP_LINEAR,
                ),
                fused_weight_shard_indices_mapping={
                    "gate": (0, self.mlp_hidden_dim),
                    "up": (self.mlp_hidden_dim, self.mlp_hidden_dim),
                },
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                reduce_output=False,
            )
            self.local_qkv_dim = (q_dim + 2 * kv_dim) // self.tp_size
            self.local_mlp_dim = mlp_dim // self.tp_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (qkv, mlp_gate_up) with local (post-TP) sizes."""
        if self.tp_size == 1:
            out = self.proj(x)
            return out.split([self.local_qkv_dim, self.local_mlp_dim], dim=-1)
        return self.qkv_proj(x), self.mlp_proj(x)

    def load_weights(self, weight_dict: Dict[str, torch.Tensor], loader: DynamicLinearWeightLoader):
        """Load from checkpoint to_qkv_mlp_proj.weight (and optional .bias).

        The checkpoint stores a single fused weight with layout:
        [Q: q_dim | K: kv_dim | V: kv_dim | gate: mlp_hid | up: mlp_hid]

        At TP=1: loads directly into self.proj.
        At TP>1: splits into QKV and MLP portions, then loads into
        column-parallel sub-Linears which handle per-rank sharding.
        """
        for sub in self.modules():
            if callable(getattr(sub, "create_weights", None)):
                sub.create_weights()

        if self.tp_size == 1:
            loader.load_linear_weights(self.proj, "to_qkv_mlp_proj", [weight_dict])
            return

        W = weight_dict["weight"]  # [full_qkv_dim + full_mlp_dim, in_dim]
        W_qkv = W[: self.full_qkv_dim]
        W_mlp = W[self.full_qkv_dim :]

        # Split QKV into Q, K, V for FUSED_QKV_LINEAR loader
        W_q = W_qkv[: self.full_q_dim]
        W_k = W_qkv[self.full_q_dim : self.full_q_dim + self.full_kv_dim]
        W_v = W_qkv[self.full_q_dim + self.full_kv_dim :]

        # Split MLP into gate, up for FUSED_GATE_UP_LINEAR loader
        W_gate = W_mlp[: self.mlp_hidden_dim]
        W_up = W_mlp[self.mlp_hidden_dim :]

        # Build weight dict lists for fused loaders
        if "bias" in weight_dict:
            B = weight_dict["bias"]
            B_qkv = B[: self.full_qkv_dim]
            B_mlp = B[self.full_qkv_dim :]

            B_q = B_qkv[: self.full_q_dim]
            B_k = B_qkv[self.full_q_dim : self.full_q_dim + self.full_kv_dim]
            B_v = B_qkv[self.full_q_dim + self.full_kv_dim :]

            B_gate = B_mlp[: self.mlp_hidden_dim]
            B_up = B_mlp[self.mlp_hidden_dim :]

            qkv_dicts = [
                {"weight": W_q, "bias": B_q},
                {"weight": W_k, "bias": B_k},
                {"weight": W_v, "bias": B_v},
            ]
            mlp_dicts = [
                {"weight": W_gate, "bias": B_gate},
                {"weight": W_up, "bias": B_up},
            ]
        else:
            qkv_dicts = [{"weight": W_q}, {"weight": W_k}, {"weight": W_v}]
            mlp_dicts = [{"weight": W_gate}, {"weight": W_up}]

        loader.load_linear_weights(self.qkv_proj, "to_qkv_mlp_proj", qkv_dicts)
        loader.load_linear_weights(self.mlp_proj, "to_qkv_mlp_proj", mlp_dicts)
