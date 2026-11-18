# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (SM107), Helix CP16 ULL generation: checkpoint mapping, loading and
sharding.

The model loads through two mixins, each the first base of the class it serves:

* ``KimiK3Helix16UllWeights`` (of ``ModelingV2KimiK3Nvfp4Sm107Helix16Moetp16ep1``): hands the
  ``language_model.*`` keys, prefix stripped, to the text model (the generation server builds no
  vision tower).
* ``KimiK3Helix16UllTextWeights`` (of ``KimiK3Helix16UllForCausalLM``): the checkpoint name plan
  and its key check, the trunk parameters of the modules the model runs itself (embedding, LM
  head, the MLA layers, layer 0's KDA o_proj), the checkpoint FP8 attention projections, layer
  0's dense MLP slice, and the chain's weights (``_chain_weights.load_chain_weights``: every KDA
  + MoE layer, every MoE and every layer's attention-residual and norm vectors, read once, into
  the chain's own containers; the decoder layers' ``attn_res`` are bound to them). Every rank
  keeps only its share: its attention heads (under Helix CP its chunk of the ``v_b_proj`` heads)
  and its slices.

The module-level helpers cut a rank's slice
of a tensor, and read FP8 block-scaled projections (dequantized for a BF16 parameter).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Set, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm.logger import logger

from ..common.parallel import TensorParallelMode
from ..common.weight_io import filter_weights, run_concurrently

_KIMI_K3_MLA_DERIVED_PARAM_SUFFIXES = (
    ".self_attn.mixer.k_b_proj_trans",
    ".self_attn.mixer.v_b_proj",
    ".self_attn.mixer.k_b_proj_trans_scale",
    ".self_attn.mixer.v_b_proj_scale",
    ".self_attn.mixer.k_b_proj_trans_dequant",
    ".self_attn.mixer.v_b_proj_dequant",
)


def _shard_dim(
    tensor: torch.Tensor, dim: int, tp_size: int, tp_rank: int, device: torch.device
) -> torch.Tensor:
    """Rank ``tp_rank``'s contiguous ``1 / tp_size`` of ``tensor`` along ``dim`` on ``device``
    (``layers.shard_range``); the whole tensor for one rank, a missing dim (a 1-D tensor's
    input columns) or a width-1 dim. The width must divide by ``tp_size``."""
    if tp_size <= 1 or dim >= tensor.dim() or tensor.shape[dim] == 1:
        return tensor.to(device)
    width = tensor.shape[dim]
    if width % tp_size:
        raise ValueError(f"dim {dim} of width {width} does not split over {tp_size} ranks")
    start, end = shard_range(width, tp_size, tp_rank)
    return tensor.narrow(dim, start, end - start).to(device)


def _shard_head_major_param(
    name: str,
    src: torch.Tensor,
    param: torch.nn.Parameter,
    *,
    kda_tp_size: int,
    kda_tp_rank: int,
    model_tp_rank: int,
) -> torch.Tensor:
    """TP-shard a KDA ``linear_attn`` or shared-expert / dense-MLP
    ``down_proj`` checkpoint tensor down to this rank's local slice.

    Returns ``src`` unchanged when it already matches ``param`` (replicated or
    single-rank tensors) and for every other name — the caller then routes
    MLA head-shards and the shape-match copy. Slices are contiguous
    (:func:`_shard_dim`).

    - ``.linear_attn.`` names are KDA head-major tensors (``linear_attn``
      exists only on KDA layers): ``o_proj`` shards its input columns (dim 1),
      every other projection its output rows (dim 0), by ``kda_tp_size``.
    - a shared-expert / dense-MLP ``down_proj`` is ROW-sharded on its input
      columns; the fused ``gate_up_proj`` is row-concatenated and returns
      before this helper, so ``down_proj`` is the only half that reaches here.
      The TP factor comes from the checkpoint-vs-param shapes; a subgroup
      smaller than model TP repeats, so the shard index is ``model_tp_rank``
      modulo the parameter's shard count.
    """
    if src.shape == param.shape:
        return src
    if ".linear_attn." in name:
        dim = 1 if name.endswith(".o_proj.weight") else 0
        return _shard_dim(src, dim, kda_tp_size, kda_tp_rank, param.device)
    if name.endswith(".down_proj.weight") and (".shared_experts." in name or ".mlp." in name):
        assert src.shape[1] % param.shape[1] == 0, (
            f"{name}: checkpoint input dim {src.shape[1]} is not "
            f"divisible by param input dim {param.shape[1]}"
        )
        tp = src.shape[1] // param.shape[1]
        return _shard_dim(src, 1, tp, model_tp_rank % tp, param.device)
    return src


def _helix_cp_v_b_shard(
    v_weight: torch.Tensor,
    *,
    num_heads_tp_cp: int,
    cp_rank: int,
) -> torch.Tensor:
    """Select this CP rank's ``v_b_proj`` head chunk from a tp-local KV-B split.

    Under Helix context-parallel (cp_size > 1) ``v_b_proj`` holds only this
    rank's 1/cp post-all-to-all head chunk, while ``kv_b_proj`` and
    ``k_b_proj_trans`` keep every tp-local head. When cp_size == 1
    (``num_heads_tp_cp`` equals the full tp-local head count) the tensor is
    returned unchanged. ``v_weight`` is ``[num_heads_tp, v_head_dim,
    kv_lora_rank]``; only its leading head axis is sliced.
    """
    if num_heads_tp_cp != v_weight.shape[0]:
        lo = cp_rank * num_heads_tp_cp
        v_weight = v_weight[lo : lo + num_heads_tp_cp]
    return v_weight


_FP8_BLOCK_SCALE_SUFFIX = "_scale"
_FP8_BLOCK = 128


def _fp8_block_scale_key(weight_key: str) -> str:
    return weight_key + _FP8_BLOCK_SCALE_SUFFIX


def _checkpoint_fp8_pair(ckpt_key: str, src: torch.Tensor, weights):
    """``(fp8 weight, fp32 128x128 block scale)`` if this tensor is FP8_PB_WO.

    ``None`` for an ordinary (BF16) checkpoint tensor. Raises when an FP8
    tensor has no companion scale rather than letting the caller fall through
    to a cast that would silently drop it.
    """
    if src.dtype != torch.float8_e4m3fn:
        return None
    scale_key = _fp8_block_scale_key(ckpt_key)
    if scale_key not in weights and scale_key + "_inv" in weights:
        scale_key += "_inv"
    if scale_key not in weights:
        raise KeyError(
            f"Kimi K3: {ckpt_key} is FP8 E4M3 but has no {scale_key} or {scale_key}_inv; refusing "
            "to load it as if it were unquantized."
        )
    scale = _materialize(weights[scale_key]).float()
    # The checkpoint stores the block scale 4-D as
    # [ceil(N/128), 1, ceil(K/128), 1]. Normalize here, once, so every
    # consumer sees the plain 2-D [n_blocks_m, n_blocks_k] that both the
    # dequantization below and deep_gemm's transform_sf_into_required_layout
    # expect -- the latter asserts on rank and gave an unhelpful
    # ``assert sf.dim() == ...`` when handed the raw 4-D tensor.
    if scale.dim() == 4:
        scale = scale.reshape(scale.shape[0], scale.shape[2])
    return src, scale


def _dequantize_fp8_block_scaled(ckpt_key: str, src: torch.Tensor, weights) -> torch.Tensor:
    """Apply checkpoint scales for projections explicitly configured as BF16.

    Checkpoint-native FP8 attention bypasses this loader. This handles an
    excluded projection or another unquantized trunk parameter whose source
    tensor is quantized; casting the codes alone would discard its scales.
    """
    pair = _checkpoint_fp8_pair(ckpt_key, src, weights)
    if pair is None:
        return src
    src, scale = pair
    out = src.to(torch.float32)
    if scale.numel() == 1:
        return (out * scale.reshape(())).to(torch.bfloat16)
    expanded = scale.repeat_interleave(_FP8_BLOCK, dim=0).repeat_interleave(_FP8_BLOCK, dim=1)
    if expanded.shape[0] < out.shape[0] or expanded.shape[1] < out.shape[1]:
        raise ValueError(
            f"Kimi K3: {_fp8_block_scale_key(ckpt_key)} covers "
            f"{tuple(expanded.shape)} but {ckpt_key} is {tuple(out.shape)}."
        )
    return (out * expanded[: out.shape[0], : out.shape[1]]).to(torch.bfloat16)


def _shards_itself(linear: nn.Module) -> bool:
    """A ``layers`` projection that takes the full checkpoint tensors and keeps its own TP shard."""
    return (
        isinstance(linear, Fp8BlockScaleLinear)
        and linear.tp_mode is not None
        and linear.tp_size > 1
    )


def _full_projection_shape(linear: nn.Module) -> Tuple[int, int]:
    """``(out, in)`` of the whole checkpoint projection a TP shard ``linear`` loads from."""
    if isinstance(linear, Fp8BlockScaleLinear):
        return (linear.full_out_features, linear.full_in_features)
    # Main's Linear (KDA / MLA projections not yet on core): the shape its tp_mode splits.
    shape = [linear.out_features, linear.in_features]
    if linear.tp_mode is not None:
        shape[0 if linear.tp_mode == TensorParallelMode.COLUMN else 1] *= linear.tp_size
    return tuple(shape)


def _load_fp8_projection(linear: nn.Module, weight: torch.Tensor, scale: torch.Tensor) -> None:
    """Copy checkpoint E4M3 codes and their FP32 128 x 128 scale grid into an FP8 projection
    (its own TP shard of them where it shards)."""
    if isinstance(linear, Fp8BlockScaleLinear):
        linear.load(weight, scale)
    else:
        linear.load_weights([{"weight": weight, "weight_scale": scale}])


def _projection_shard(
    linear: nn.Module, tensor: torch.Tensor, scale_span: Optional[int] = None
) -> torch.Tensor:
    """This rank's slice of a full checkpoint tensor (or with ``scale_span`` its block-scale
    grid) of a TP-sharded projection, on the projection's device."""
    if isinstance(linear, (Bf16Linear, Fp8BlockScaleLinear)):
        return linear.shard(tensor, scale_span=scale_span)
    kwargs = {} if scale_span is None else {"scale_span": scale_span}
    return linear.load_shard(tensor, device=linear.weight.device, **kwargs)


def _materialize(value) -> torch.Tensor:
    """Materialize a (possibly lazy safetensors slice) weight value."""
    if isinstance(value, torch.Tensor):
        return value
    # ``[:]`` is how a lazy slice is realized, but it is invalid on a 0-dim
    # entry. The NVFP4 checkpoint stores weight_scale_2 / input_scale as
    # scalars, so realize those entries with ``[()]`` instead.
    get_shape = getattr(value, "get_shape", None)
    if get_shape is not None and len(get_shape()) == 0:
        return value[()]
    return value[:]


_DENSE0_PREFIX = "model.layers.0.mlp."
_DENSE0_CKPT = "model.layers.0.mlp."
_DENSE0_PARTS = ("gate", "up", "down")


def _checkpoint_slice(key: str, weights, lo: int, hi: int, dim: int) -> torch.Tensor:
    """Rows (``dim`` 0) or columns (``dim`` 1) ``lo:hi`` of a 2-D checkpoint tensor, as BF16: an
    FP8 tensor is read on the 128-aligned span covering them and dequantized with its block
    scales (``_dequantize_fp8_block_scaled``) before the cut."""
    source = weights[key]
    shape = tuple(source.shape) if isinstance(source, torch.Tensor) else tuple(source.get_shape())
    if len(shape) != 2 or not 0 <= lo < hi <= shape[dim]:
        raise ValueError(f"{key}: [{lo}:{hi}] along dim {dim} of {shape}")
    start = lo // _FP8_BLOCK * _FP8_BLOCK
    end = min(-(-hi // _FP8_BLOCK) * _FP8_BLOCK, shape[dim])
    index = [slice(None), slice(None)]
    index[dim] = slice(start, end)
    part = _materialize(source[tuple(index)])
    pair = _checkpoint_fp8_pair(key, part, weights)
    if pair is not None:
        codes, scale = pair
        scale_index = [slice(None), slice(None)]
        scale_index[dim] = slice(start // _FP8_BLOCK, -(-end // _FP8_BLOCK))
        scale = scale[tuple(scale_index)]
        expanded = scale.repeat_interleave(_FP8_BLOCK, 0).repeat_interleave(_FP8_BLOCK, 1)
        part = (codes.float() * expanded[: codes.shape[0], : codes.shape[1]]).to(torch.bfloat16)
    index[dim] = slice(lo - start, hi - start)
    return part[tuple(index)].to(torch.bfloat16)


class KimiK3Helix16UllTextWeights:
    """Checkpoint loading of the text model: name plan, trunk, FP8 projections, layer 0's dense
    MLP slice, and the chain's weights.

    The first base of ``KimiK3Helix16UllForCausalLM``, so its ``load_weights`` resolves before
    ``SpecDecOneEngineForCausalLM``'s.
    """

    def _attention_fp8_linears(self) -> list[tuple[str, nn.Module, nn.Module]]:
        """Collect checkpoint-backed FP8 attention projections for weight loading.

        The loader uses this list to enumerate required weight/scale keys, exclude
        these parameters from ordinary trunk loading, and load FP8 codes and scales
        together: layer 0's KDA o_proj and the MLA layers' projections.

        Returns:
            Tuples of runtime module path, linear, and owning attention module.
        """
        projections = []
        for index, layer in enumerate(self.model.layers):
            if not _has_weights(layer):
                continue
            if hasattr(layer, "linear_attn"):
                attention = layer.linear_attn
                scope = f"model.layers.{index}.linear_attn"
            elif hasattr(layer, "self_attn"):
                attention = layer.self_attn.mixer
                scope = f"model.layers.{index}.self_attn.mixer"
            else:
                continue
            for attr, module in attention.named_children():
                if _is_fp8_block_projection(module):
                    projections.append((f"{scope}.{attr}", module, attention))
        return projections

    def _load_attention_fp8(self, weights: Dict[str, torch.Tensor], prefix: str) -> int:
        """Load checkpoint FP8 codes/scales without a BF16 projection intermediate."""
        projections = self._attention_fp8_linears()
        for name, linear, attention in projections:
            # Both attention implementations use self_attn keys in the checkpoint.
            key = (
                prefix
                + name.replace(".linear_attn.", ".self_attn.").replace(
                    ".self_attn.mixer.", ".self_attn."
                )
                + ".weight"
            )
            attr = name.rsplit(".", 1)[1]
            if attr == "kv_a_proj_with_mqa" and attention.fuse_qkv_a_proj:
                # The runtime fuses Q-A/KV-A, but the checkpoint stores them separately.
                # Construction checks that the fusion boundary aligns with FP8 blocks.
                q_key = key.replace("kv_a_proj_with_mqa.weight", "q_a_proj.weight")
                pairs = []
                for part_key, rows in (
                    (q_key, attention.q_lora_rank),
                    (key, attention.kv_lora_rank + attention.qk_rope_head_dim),
                ):
                    part = _materialize(weights[part_key])
                    pair = _checkpoint_fp8_pair(part_key, part, weights)
                    if pair is None:
                        raise ValueError(
                            f"{part_key}: quant config declares FP8 but tensor is not E4M3"
                        )
                    codes, scales = pair
                    expected = (rows, attention.hidden_size)
                    if tuple(codes.shape) != expected:
                        raise ValueError(
                            f"{part_key}: checkpoint shape {tuple(codes.shape)} != {expected}"
                        )
                    if tuple(scales.shape) != tuple(math.ceil(d / _FP8_BLOCK) for d in expected):
                        raise ValueError(f"{part_key}: scale shape does not cover {expected}")
                    pairs.append(pair)
                _load_fp8_projection(
                    linear,
                    torch.cat([pair[0] for pair in pairs], dim=0),
                    torch.cat([pair[1] for pair in pairs], dim=0),
                )
                continue
            source = weights[key]
            shape = (
                tuple(source.shape)
                if isinstance(source, torch.Tensor)
                else tuple(source.get_shape())
            )
            local_shape = (linear.out_features, linear.in_features)
            scale_slice = None
            if isinstance(attention, KdaOutputProjection) and not _shards_itself(linear):
                # KDA linears already have rank-local dimensions. Slice the lazy source
                # before materializing it, and apply the same shard to its block scales.
                split_dim = 1 if attr == "o_proj" else 0
                tp_size = attention._kda_tp_size
                tp_rank = attention._kda_tp_rank
                expected_shape = list(local_shape)
                expected_shape[split_dim] *= tp_size
                if shape != tuple(expected_shape):
                    raise ValueError(f"{key}: checkpoint shape {shape} != {tuple(expected_shape)}")
                if tp_size > 1:
                    width = local_shape[split_dim]
                    start, end = tp_rank * width, (tp_rank + 1) * width
                    if start % _FP8_BLOCK or end % _FP8_BLOCK:
                        raise ValueError(f"{key}: TP slice [{start}:{end}] is not block-aligned")
                    indices = [slice(None), slice(None)]
                    indices[split_dim] = slice(start, end)
                    source = source[tuple(indices)]
                    scale_indices = [slice(None), slice(None)]
                    scale_indices[split_dim] = slice(
                        start // _FP8_BLOCK, math.ceil(end / _FP8_BLOCK)
                    )
                    scale_slice = tuple(scale_indices)
            pair = _checkpoint_fp8_pair(key, _materialize(source), weights)
            if pair is None:
                raise ValueError(
                    f"{key}: quant config declares FP8 but checkpoint tensor is not E4M3"
                )
            weight, scale = pair
            full_scale_shape = tuple(math.ceil(dim / _FP8_BLOCK) for dim in shape)
            if tuple(scale.shape) != full_scale_shape:
                raise ValueError(f"{key}: scale shape {tuple(scale.shape)} != {full_scale_shape}")
            if scale_slice is not None:
                scale = scale[scale_slice]
            if isinstance(attention, KdaOutputProjection) and not _shards_itself(linear):
                if tuple(scale.shape) != tuple(math.ceil(dim / _FP8_BLOCK) for dim in local_shape):
                    raise ValueError(f"{key}: scale shard does not cover local shape {local_shape}")
                _load_fp8_projection(linear, weight, scale)
            elif attr == "kv_b_proj":
                # KV-B also supplies decode absorption weights; reorder codes and scales
                # together and retain raw FP8 copies for the absorption kernels.
                self._load_mla_fp8_kv_b(attention, weight, scale)
            else:
                # The projection shards the full checkpoint tensors itself.
                expected_shape = _full_projection_shape(linear)
                if shape != expected_shape:
                    raise ValueError(f"{key}: checkpoint shape {shape} != {expected_shape}")
                _load_fp8_projection(linear, weight, scale)
        return len(projections)

    @staticmethod
    def _load_mla_fp8_kv_b(attention: nn.Module, weight: torch.Tensor, scale: torch.Tensor) -> None:
        """Reorder checkpoint KV-B heads and their scale grid together."""
        linear = attention.kv_b_proj
        h, n, v, k = (
            attention.num_heads_tp,
            attention.qk_nope_head_dim,
            attention.v_head_dim,
            attention.kv_lora_rank,
        )
        if any(dim % _FP8_BLOCK for dim in (n, v, k)):
            raise ValueError(
                "Kimi MLA FP8 absorption requires block-aligned head and latent dimensions"
            )
        expected_shape = (h * linear.tp_size * (n + v), k)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(f"Kimi MLA KV-B shape {tuple(weight.shape)} != {expected_shape}")
        local = _projection_shard(linear, weight).reshape(h, n + v, k)
        local_scale = _projection_shard(linear, scale, scale_span=_FP8_BLOCK).reshape(
            h, (n + v) // _FP8_BLOCK, k // _FP8_BLOCK
        )
        key_weight, value_weight = local.split((n, v), dim=1)
        key_scale, value_scale = local_scale.split((n // _FP8_BLOCK, v // _FP8_BLOCK), dim=1)
        linear.weight.data.copy_(
            torch.cat((key_weight.reshape(h * n, k), value_weight.reshape(h * v, k)))
        )
        linear.weight_scale.data.copy_(
            torch.cat((key_scale.flatten(0, 1), value_scale.flatten(0, 1)))
        )
        attention.k_b_proj_trans.data.copy_(key_weight.transpose(1, 2))
        attention.k_b_proj_trans_scale.data.copy_(key_scale.transpose(1, 2))
        if attention.mapping.has_cp_helix():
            expanded = value_scale.repeat_interleave(_FP8_BLOCK, -2).repeat_interleave(
                _FP8_BLOCK, -1
            )
            # The MLA layer owns the V-up-before-exchange flag; bind through it (its
            # transform_weights rebuilds the same weight from kv_b_proj).
            attention._bind_helix_v_up_weight((value_weight.float() * expanded).to(torch.bfloat16))
        value_weight = _helix_cp_v_b_shard(
            value_weight,
            num_heads_tp_cp=attention.num_heads_tp_cp,
            cp_rank=attention.mapping.cp_rank,
        )
        value_scale = _helix_cp_v_b_shard(
            value_scale,
            num_heads_tp_cp=attention.num_heads_tp_cp,
            cp_rank=attention.mapping.cp_rank,
        )
        # Absorption must keep raw checkpoint codes independent of Linear's
        # backend-specific weight transformations.
        attention.v_b_proj = nn.Parameter(value_weight.clone().contiguous(), requires_grad=False)
        attention.v_b_proj_scale.data.copy_(value_scale)
        for target, codes, scales in (
            (
                attention.k_b_proj_trans_dequant,
                key_weight.transpose(1, 2),
                key_scale.transpose(1, 2),
            ),
            (attention.v_b_proj_dequant, value_weight, value_scale),
        ):
            if target is not None:
                expanded = scales.repeat_interleave(_FP8_BLOCK, -2).repeat_interleave(
                    _FP8_BLOCK, -1
                )
                target.data.copy_((codes.float() * expanded).to(target.dtype))

    def _trunk_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Named parameters of the trunk only. Spec-dec draft modules
        (e.g. the DFlash drafter attached by SpecDecOneEngineForCausalLM)
        live in a separate checkpoint loaded by
        ModelLoader.load_draft_weights, not in the target checkpoint. MLA
        K/V absorb Parameters are derived by the KV-B loader and are likewise
        excluded from checkpoint jobs, and layer 0's dense MLP slice loads on its own
        (:meth:`_load_dense0`)."""
        fp8_prefixes = tuple(name + "." for name, _, _ in self._attention_fp8_linears())
        return {
            name: param
            for name, param in self.named_parameters()
            if not name.startswith("draft_model.")
            and not name.startswith(fp8_prefixes)
            and not name.startswith(_DENSE0_PREFIX)
            and not name.endswith(_KIMI_K3_MLA_DERIVED_PARAM_SUFFIXES)
        }

    def checkpoint_name_plan(self, prefix: str) -> Tuple[Dict[str, str], Set[str]]:
        """Return ``(name_map, expected_keys)``.

        ``name_map`` maps every trunk parameter name to its checkpoint key; ``expected_keys``
        also covers the FP8 projections' weight / scale keys, the MLA layers' separate q_a
        projection and layer 0's dense MLP. The chain's keys are its loader's to check.
        Exposed separately so the weight-name mapping can be dry-run without touching any tensor
        data.
        """
        params = self._trunk_parameters()
        expected_keys = set()
        name_map: Dict[str, str] = {}
        for name in params:
            if name == "lm_head.weight":
                ckpt_key = prefix + "lm_head.weight"
            elif ".linear_attn." in name:
                ckpt_key = prefix + name.replace(".linear_attn.", ".self_attn.")
            else:
                # MLA retains its runtime/mixer hierarchy; the checkpoint has no intermediate
                # ``mixer`` scope.
                ckpt_key = prefix + name.replace(".self_attn.mixer.", ".self_attn.")
            name_map[name] = ckpt_key
            expected_keys.add(ckpt_key)
        for name, _, _ in self._attention_fp8_linears():
            key = (
                prefix
                + name.replace(".linear_attn.", ".self_attn.").replace(
                    ".self_attn.mixer.", ".self_attn."
                )
                + ".weight"
            )
            expected_keys.update((key, _fp8_block_scale_key(key)))
        for index, layer in enumerate(self.model.layers):
            if not hasattr(layer, "self_attn") or not _has_weights(layer):
                continue
            attention = layer.self_attn.mixer
            if getattr(attention, "fuse_qkv_a_proj", False):
                key = f"{prefix}model.layers.{index}.self_attn.q_a_proj.weight"
                expected_keys.add(key)
                if _is_fp8_block_projection(attention.kv_a_proj_with_mqa):
                    expected_keys.add(_fp8_block_scale_key(key))
        expected_keys.update(f"{prefix}{_DENSE0_CKPT}{part}_proj.weight" for part in _DENSE0_PARTS)
        return name_map, expected_keys

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        prefix = "language_model." if any(k.startswith("language_model.") for k in weights) else ""
        params = self._trunk_parameters()
        name_map, expected_keys = self.checkpoint_name_plan(prefix)

        self._validate_checkpoint_keys(weights, expected_keys, prefix)
        num_params = self._load_trunk_params(weights, params, name_map)
        num_params += self._load_attention_fp8(weights, prefix)
        self._load_dense0(weights, prefix)
        self._finalize_weight_load(num_params)
        device = next(self.parameters()).device
        # The chain's weights (KDA + MoE layers, every MoE, its AttnRes copies), this rank's.
        from ..ops.chain.k3_ull_chain import load_chain_weights

        chain_weights = load_chain_weights(weights, self.model_config, self.model.rank, device)
        for layer in self.model.layers:
            layer.attn_res = chain_weights.layers[layer.layer_idx].attn_res
        self.model.chain_weights = chain_weights
        if device.type == "cuda":
            # Lazy source mappings are load-scoped; finish nonblocking H2D
            # work before the caller can release the weights container.
            torch.cuda.synchronize(device)

    def _load_dense0(self, weights: Dict[str, torch.Tensor], prefix: str) -> None:
        """Layer 0's dense MLP: this rank's slice of the checkpoint gate / up rows and down
        columns (``layer0.Dense0``)."""
        mlp = self.model.layers[0].mlp
        lo, hi = mlp.index * mlp.inter, (mlp.index + 1) * mlp.inter
        base = f"{prefix}{_DENSE0_CKPT}"
        gate, up, down = (
            _checkpoint_slice(f"{base}{part}_proj.weight", weights, lo, hi, dim)
            for part, dim in zip(_DENSE0_PARTS, (0, 0, 1))
        )
        mlp.load(gate, up, down)

    def _validate_checkpoint_keys(
        self, weights: Dict[str, torch.Tensor], expected_keys: Set[str], prefix: str
    ) -> None:
        """Key-set validation (both directions): every expected key must be
        present; unmatched checkpoint keys (beyond the expected leftovers)
        only warn."""
        ckpt_keys = set(weights.keys())
        relevant_ckpt_keys = {
            k
            for k in ckpt_keys
            if not (k.startswith("vision_tower.") or k.startswith("mm_projector."))
        }
        missing = sorted(
            key
            for key in expected_keys - ckpt_keys
            if not (key.endswith(".weight_scale") and key + "_inv" in ckpt_keys)
        )
        if missing:
            raise KeyError(
                f"Kimi K3 load_weights: {len(missing)} expected checkpoint "
                f"keys are missing, e.g. {missing[:10]}"
            )
        unexpected = relevant_ckpt_keys - expected_keys
        # Non-local experts and (in layer-truncated debug mode) extra layers
        # are expected leftovers.
        surprising = sorted(
            k
            for k in unexpected
            if ".block_sparse_moe.experts." not in k and not k.startswith(f"{prefix}model.layers.")
        )
        if surprising:
            logger.warning(
                f"Kimi K3 load_weights: {len(surprising)} unmatched "
                f"checkpoint keys, e.g. {surprising[:10]}"
            )

    def _load_trunk_params(
        self,
        weights: Dict[str, torch.Tensor],
        params: Dict[str, torch.nn.Parameter],
        name_map: Dict[str, str],
    ) -> int:
        """Load every non-expert trunk parameter concurrently (with the
        per-parameter TP-shard / pad / fuse conversions) and return the
        number of parameters loaded."""
        # The checkpoint stores every MLA KV-B head as interleaved [K | V]
        # rows. Runtime keeps one DeepSeek-style [all K | all V] parameter
        # instead, so context can project directly into the FMHA layout and
        # absorbed decode can take zero-copy K/V views.
        mla_mixers = [
            layer.self_attn.mixer
            for layer in self.model.layers
            if hasattr(layer, "self_attn") and _has_weights(layer)
        ]
        mla_kv_b_mixers = {id(mixer.kv_b_proj.weight): mixer for mixer in mla_mixers}
        mla_fused_a_mixers = {
            id(mixer.kv_a_proj_with_mqa.weight): mixer
            for mixer in mla_mixers
            if getattr(mixer, "fuse_qkv_a_proj", False)
        }
        mla_head_shard_linears = {}
        for mixer in mla_mixers:
            mla_head_shard_linears[id(mixer.q_b_proj.weight)] = mixer.q_b_proj
            mla_head_shard_linears[id(mixer.o_proj.weight)] = mixer.o_proj
            g_proj = getattr(mixer, "g_proj", None)
            if g_proj is not None:
                mla_head_shard_linears[id(g_proj.weight)] = g_proj

        device = next(self.parameters()).device

        # MLP TP shard index. A dense MLP whose intermediate size does not
        # divide model TP uses a smaller repeated TP subgroup, so its local
        # shard rank is model tp_rank modulo the parameter's shard count.
        # Under helix the modules were sharded against the repurposed
        # mapping; the restored CP original's tp_rank is 0 on every rank.
        model_tp_rank = (
            self._repurposed_tp_mapping.tp_rank
            if self._repurposed_tp_mapping is not None
            else self.model_config.mapping.tp_rank
        )
        # KDA head-shard (attention-DP off): rank r loads head rows/cols
        # [r*local : (r+1)*local] of every head-major KDA tensor.
        kda_tp_size, kda_tp_rank = 1, 0
        for layer in self.model.layers:
            if hasattr(layer, "linear_attn"):
                kda_tp_size = layer.linear_attn._kda_tp_size
                kda_tp_rank = layer.linear_attn._kda_tp_rank
                break

        def load_param(name: str, param: torch.nn.Parameter) -> None:
            if device.type == "cuda":
                torch.cuda.set_device(device)
            src = _materialize(weights[name_map[name]])
            src = _dequantize_fp8_block_scaled(name_map[name], src, weights)
            fused_a = mla_fused_a_mixers.get(id(param))
            if fused_a is not None:
                q_key = name_map[name].replace("kv_a_proj_with_mqa.weight", "q_a_proj.weight")
                q_weight = _dequantize_fp8_block_scaled(
                    q_key, _materialize(weights[q_key]), weights
                )
                if q_weight.shape != (fused_a.q_lora_rank, fused_a.hidden_size) or src.shape != (
                    fused_a.kv_lora_rank + fused_a.qk_rope_head_dim,
                    fused_a.hidden_size,
                ):
                    raise ValueError(
                        f"{name}: checkpoint Q-A/KV-A shapes do not match MLA dimensions"
                    )
                src = torch.cat((q_weight, src), dim=0)
            if name == "lm_head.weight":
                # LMHead is vocab-sharded (TP column) + gathered; its
                # load_weights shards the full checkpoint tensor.
                self.lm_head.load_weights(weights=[{"weight": src}])
                return
            mla_mixer = mla_kv_b_mixers.get(id(param))
            if mla_mixer is not None:
                h = mla_mixer.num_heads_tp
                n = mla_mixer.qk_nope_head_dim
                v = mla_mixer.v_head_dim
                kv = mla_mixer.kv_lora_rank
                local = _projection_shard(mla_mixer.kv_b_proj, src).view(h, n + v, kv)
                k_weight, v_weight = local.split([n, v], dim=1)
                param.data.copy_(
                    torch.cat(
                        [
                            k_weight.reshape(h * n, kv),
                            v_weight.reshape(h * v, kv),
                        ],
                        dim=0,
                    ).to(param.dtype)
                )
                mla_mixer.k_b_proj_trans.data.copy_(k_weight.transpose(1, 2))
                # Share the loaded BF16 V rows with context GEMM instead of keeping a
                # separate copy. Wrap the view to retain v_b_proj's parameter registration.
                v_weight = _helix_cp_v_b_shard(
                    param[h * n :].view(h, v, kv),
                    num_heads_tp_cp=mla_mixer.num_heads_tp_cp,
                    cp_rank=mla_mixer.mapping.cp_rank,
                )
                mla_mixer.v_b_proj = nn.Parameter(v_weight, requires_grad=False)
                if mla_mixer.mapping.has_cp_helix():
                    mla_mixer._bind_helix_v_up_weight(param[h * n :].view(h, v, kv))
                return
            if name.endswith(".A_log") and src.numel() != param.numel():
                # The checkpoint pads A_log from [num_heads] to [head_dim]
                # (e.g. [96] -> [128]); the tail must be zeros. Under KDA
                # head-shard TP the param holds this rank's head range
                # instead of the full [num_heads].
                assert src.numel() > param.numel(), (name, src.shape)
                if kda_tp_size > 1:
                    lo = kda_tp_rank * param.numel()
                    src = src[lo : lo + param.numel()]
                else:
                    tail = src[param.numel() :]
                    if tail.abs().max().item() != 0.0:
                        raise ValueError(
                            f"{name}: expected zero padding beyond "
                            f"{param.numel()} entries, got nonzero tail"
                        )
                    src = src[: param.numel()]
            # KDA ``linear_attn`` (head-major) and shared-expert / dense-MLP
            # ``down_proj`` shards are shape-derived; replicated tensors and
            # every other name pass through unchanged. MLA head-shards and the
            # shape-match copy are handled by the block below.
            src = _shard_head_major_param(
                name,
                src,
                param,
                kda_tp_size=kda_tp_size,
                kda_tp_rank=kda_tp_rank,
                model_tp_rank=model_tp_rank,
            )

            if src.shape != param.shape:
                # MLA q_b/g/o head-shard: delegate to the same Linear modules
                # that own their COLUMN/ROW sharding policy (#17684 removed the
                # 96->128 head padding). KV-B is handled above; KDA and
                # shared-expert/MLP shards were resolved in the pre-block above.
                mla_sharded_linear = mla_head_shard_linears.get(id(param))
                if mla_sharded_linear is not None:
                    shard = _projection_shard(mla_sharded_linear, src)
                    if shard.shape != param.shape:
                        raise ValueError(
                            f"{name}: MLA shard shape {tuple(shard.shape)} does "
                            f"not match param shape {tuple(param.shape)}"
                        )
                    param.data.copy_(shard.to(param.dtype))
                    return
                raise ValueError(
                    f"{name}: shard/pad result {tuple(src.shape)} != param "
                    f"shape {tuple(param.shape)}"
                )
            param.data.copy_(src.to(param.dtype))

        param_jobs = [(name, params[name]) for name in name_map]
        run_concurrently(load_param, param_jobs, num_workers=8)

        logger.info(
            f"Kimi K3: loaded {len(mla_mixers)} MLA KV-B projections in grouped runtime layout"
        )
        return len(param_jobs)

    def _finalize_weight_load(self, num_params: int) -> None:
        """Swap the checkpoint FP8 projections (layer 0's KDA o_proj, the MLA layers') for the FP8
        weight-read GEMM."""
        for layer in self.model.layers:
            if not _has_weights(layer):
                continue
            if hasattr(layer, "linear_attn"):
                attention = layer.linear_attn
            elif hasattr(layer, "self_attn"):
                attention = layer.self_attn.mixer
            else:
                continue
            for name, linear in attention.named_children():
                if _is_fp8_block_projection(linear):
                    _swap_linear_to_fp8_weight_read(attention, name, (type(linear),))
        logger.info(f"Kimi K3 ULL: loaded {num_params} parameters")


class KimiK3Helix16UllWeights:
    """Checkpoint loading of the vision-language checkpoint's text model (the ``language_model.*``
    keys; the vision tower and projector keys are not read).

    The first base of ``ModelingV2KimiK3Nvfp4Sm107Helix16Moetp16ep1``.
    """

    def load_draft_weights(self, *args, **kwargs):
        return self.llm.load_draft_weights(*args, **kwargs)

    def load_weights(self, weights) -> None:
        """Load the text model's weights from the checkpoint."""
        if any(k.startswith(self._LANG_PREFIX) for k in weights):
            lm_weights = filter_weights("language_model", weights)
            lm_weights = ConsumableWeightsDict(lm_weights)
            checkpoint_dir = getattr(weights, "checkpoint_dir", None)
            if checkpoint_dir is not None:
                lm_weights.checkpoint_dir = checkpoint_dir
            lm_weights.checkpoint_prefix = self._LANG_PREFIX
        else:
            lm_weights = weights
        self.llm.load_weights(lm_weights)


# The model's blocks: modeling.py imports this module after defining them.
from .modeling import (  # noqa: E402
    Bf16Linear,
    Fp8BlockScaleLinear,
    KdaOutputProjection,
    _has_weights,
    _is_fp8_block_projection,
    _swap_linear_to_fp8_weight_read,
    shard_range,
)
