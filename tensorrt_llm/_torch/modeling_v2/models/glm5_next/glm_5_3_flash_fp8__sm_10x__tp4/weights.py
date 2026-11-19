# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This rank's weights, read from the checkpoint the engine hands ``load_weights``.

Every tensor is read at this rank's share (``checkpoint.Checkpoint.read`` cuts the lazy
safetensors entry, so no rank reads another's rows), transformed into the layout the
target's operations read, and kept as a plain device tensor on the core. Under tensor
parallelism 4:

* **KDA** projections, convolution taps, gates and the output projection are split by this
  rank's 16 of 64 heads (``layers/kda.py``).
* **Sparse MLA**: the query heads (``q_b_proj``, ``kv_b_proj``) and the matching input
  columns of ``o_proj`` are split by 16 of 64 heads; the ``[q_a | kv_a]`` latent
  projections, the norms and the whole indexer are replicated, as are the latent and
  indexer cache rows they write. Sequence-parallel prefill attends a quarter of the rows
  with all 64 heads, so the layer also keeps full-head copies (``layers/mla.py``).
* **FFN**: the dense MLPs' and the experts' intermediate is split (the routed experts by
  moe_tp 4, all 288 on every rank); router and bias are replicated (``layers/ffn.py``).
* Everything here is replicated: the embedding, every norm, the mHC mixers, and the MTP
  layer's ``enorm`` / ``hnorm`` / ``shared_head.norm``. ``eh_proj`` is row-parallel (this
  rank's 2048 of its 8192 input columns). ``lm_head`` is the shell's vocabulary-parallel
  head: this rank's rows of the checkpoint tensor.

The checkpoint is block FP8 (128 x 128 blocks, FP32 ``weight_scale_inv``) for the MLA
low-rank and query / output projections and every MLP and expert projection, of the MTP
layer too; the KDA layers, ``kv_b_proj``, the indexer, the router, the norms, ``eh_proj``
and the mHC mixers are BF16 or FP32. ``activation_scheme`` is dynamic: there are no input
scales.

Loading contract: every checkpoint key is read exactly once, except the vision tower's,
and the MTP layer's when no speculative decoding builds it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

import torch

from .checkpoint import Checkpoint, cols, rows, share
from .dims import TEXT_PREFIX, Dims
from .layers.ffn import DenseMlpWeights, MoeWeights, load_dense_mlp, load_moe
from .layers.kda import KdaWeights, load_kda
from .layers.mla import MlaWeights, load_mla
from .schedule import DENSE_MLP, LINEAR_ATTENTION

#: Checkpoint key prefix of the vision tower, which this text-only target never loads.
VISION_PREFIX = "model.visual."


@dataclass
class HcWeights:
    """One mHC site's mixer: ``fn`` FP32 ``[mix, mult * hidden]`` (the checkpoint's BF16 cast
    to FP32), ``base`` FP32 ``[mix]`` and ``scale`` FP32 ``[3]``."""

    fn: torch.Tensor
    base: torch.Tensor
    scale: torch.Tensor


@dataclass
class DecoderLayerWeights:
    attention_kind: str
    mlp_kind: str
    input_norm: torch.Tensor  # [hidden] bf16
    post_norm: torch.Tensor  # [hidden] bf16
    hc_attn: HcWeights
    hc_ffn: HcWeights
    attention: Union[KdaWeights, MlaWeights]
    mlp: Union[DenseMlpWeights, MoeWeights]


@dataclass
class MtpLayerWeights:
    """The MTP layer (checkpoint layer ``num_hidden_layers``): a sparse-MLA + MoE layer
    without mHC, plus its front end and head norm."""

    enorm: torch.Tensor  # [hidden] bf16
    hnorm: torch.Tensor  # [hidden] bf16
    eh_proj: torch.Tensor  # [hidden, 2 * hidden / tp] bf16, this rank's input columns
    head_norm: torch.Tensor  # [hidden] bf16, shared_head.norm
    input_norm: torch.Tensor  # [hidden] bf16
    post_norm: torch.Tensor  # [hidden] bf16
    attention: MlaWeights
    mlp: MoeWeights


@dataclass
class ModelWeights:
    embed: torch.Tensor  # [vocab, hidden] bf16, replicated
    final_norm: torch.Tensor  # [hidden] bf16
    layers: list[DecoderLayerWeights]
    mtp: Optional[MtpLayerWeights]


def _hc(ck: Checkpoint, prefix: str, site: str) -> HcWeights:
    return HcWeights(
        fn=ck.read(f"{prefix}hc_{site}_fn", dtype=torch.float32),
        base=ck.read(f"{prefix}hc_{site}_base", dtype=torch.float32),
        scale=ck.read(f"{prefix}hc_{site}_scale", dtype=torch.float32),
    )


def _decoder_layer(ck: Checkpoint, dims: Dims, layer: int) -> DecoderLayerWeights:
    prefix = dims.layer_prefix(layer)
    attention_kind = dims.schedule.attention[layer]
    mlp_kind = dims.schedule.mlp[layer]
    attention = (
        load_kda(ck, dims, layer)
        if attention_kind == LINEAR_ATTENTION
        else load_mla(ck, dims, layer)
    )
    mlp = load_dense_mlp(ck, dims, layer) if mlp_kind == DENSE_MLP else load_moe(ck, dims, layer)
    return DecoderLayerWeights(
        attention_kind=attention_kind,
        mlp_kind=mlp_kind,
        input_norm=ck.read(f"{prefix}input_layernorm.weight"),
        post_norm=ck.read(f"{prefix}post_attention_layernorm.weight"),
        hc_attn=_hc(ck, prefix, "attn"),
        hc_ffn=_hc(ck, prefix, "ffn"),
        attention=attention,
        mlp=mlp,
    )


def _mtp_layer(ck: Checkpoint, dims: Dims) -> MtpLayerWeights:
    layer = dims.mtp_layer
    prefix = dims.layer_prefix(layer)
    start, end = share(2 * dims.hidden, dims.tp_size, dims.tp_rank)
    return MtpLayerWeights(
        enorm=ck.read(f"{prefix}enorm.weight"),
        hnorm=ck.read(f"{prefix}hnorm.weight"),
        eh_proj=ck.read(f"{prefix}eh_proj.weight", cut=cols(start, end)),
        head_norm=ck.read(f"{prefix}shared_head.norm.weight"),
        input_norm=ck.read(f"{prefix}input_layernorm.weight"),
        post_norm=ck.read(f"{prefix}post_attention_layernorm.weight"),
        attention=load_mla(ck, dims, layer),
        mlp=load_moe(ck, dims, layer),
    )


def load_model_weights(
    weights: Mapping[str, Any], dims: Dims, lm_head_weight: torch.Tensor, device: torch.device
) -> ModelWeights:
    """Read this rank's weights and fill the shell's vocabulary-parallel ``lm_head`` share."""
    ck = Checkpoint(weights, device)
    model = ModelWeights(
        embed=ck.read(f"{TEXT_PREFIX}embed_tokens.weight"),
        final_norm=ck.read(f"{TEXT_PREFIX}norm.weight"),
        layers=[_decoder_layer(ck, dims, i) for i in range(dims.num_layers)],
        mtp=_mtp_layer(ck, dims) if dims.mtp else None,
    )
    start = dims.tp_rank * dims.vocab_local
    end = min(start + dims.vocab_local, dims.vocab)
    head = ck.read("lm_head.weight", cut=rows(start, end))
    assert lm_head_weight.shape[1] == dims.hidden and lm_head_weight.shape[0] >= end - start, (
        f"lm_head share {tuple(lm_head_weight.shape)} for rows [{start}, {end})"
    )
    with torch.no_grad():
        lm_head_weight[: end - start].copy_(head)
        lm_head_weight[end - start :].zero_()
    skipped = {k for k in weights.keys() if k.startswith(VISION_PREFIX)}
    if not dims.mtp:
        mtp_prefix = dims.layer_prefix(dims.mtp_layer)
        skipped |= {k for k in weights.keys() if k.startswith(mtp_prefix)}
    ck.check_coverage(skipped)
    return model
