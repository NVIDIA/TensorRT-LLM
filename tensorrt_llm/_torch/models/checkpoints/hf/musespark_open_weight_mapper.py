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
"""Weight loading for the Muse Spark text decoder and its perception towers.

The checkpoint does not fit the generic module walk in three ways:

* Attention ships four separate projections (`q_proj`, `k_proj`,
  `v_proj`, `o_gate_proj`) that have to be fused into TRT-LLM's
  `qkv_proj` with the query and its output gate interleaved head by head.
* Experts are stored stacked per layer (`w13_weight [E, 2I, ...]`,
  `w2_weight [E, H, ...]`) with per-half global scales, while TRT-LLM's MoE
  loader wants one entry per expert and per projection.
* Shards are packed by tensor kind, not by layer, so a truncated checkpoint's
  linked shards expose tensors for layers the config does not have. Those keys
  are enumerated by the lazy loader and must be skipped without ever reading
  their payload.

Every tensor arrives as a lazy safetensors slice, so only the rank-local
expert range and the rank's attention shard are ever materialized.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ...modeling_musespark_open import PERCEPTION_KEY_PREFIXES, head_interleaved_q_gate
from .musespark_open_format import OPEN_FORMAT_SENTINEL_KEY, build_open_format_rename

_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.")

# Keys the checkpoint carries that this text model has no home for. Only KV
# quantization scales qualify: the released checkpoint has none, but a
# re-export with an FP8 KV cache would add them and the text path would still
# be correct without them.
_IGNORED_SUFFIXES = (
    ".self_attn.k_scale",
    ".self_attn.v_scale",
    ".self_attn.q_scale",
    ".self_attn.prob_scale",
)

# How each key under `PERCEPTION_KEY_PREFIXES` is handled: the towers consume
# part of that subtree, and every key they skip is recorded with its reason.
# Order matters: all four `modalities.*` prefixes below overlap, so they are
# matched longest-first.
#
# Counts on the released checkpoints: 808 canonical + 1 projection consumed,
# 658 + 50 skipped, 72 audio (consumed when an audio tower is built), 1479
# text, 3068 total.

#: The tower's own weights, and the only copy with attention biases.
CANONICAL_VISION_PREFIX = "vision_model."

#: The 8192x4096 projection into decoder hidden size. It sits under
#: `modalities.`, not under the tower, and it is the one tensor there the
#: model loads.
VISION_PROJECTION_KEY = "modalities.vision_perception.encoder.vision_projection.weight"

#: The same tensor under the bare spelling of a re-export that drops the
#: wrapper prefix.
VISION_PROJECTION_ALIAS = "vision_projection.weight"

#: 658 bit-identical copies of the tower. Skipped. Preferring this copy would
#: silently give the tower no attention biases -- the 150
#: `encoder.layers.{0..49}.attn.{q,k,v}_proj.bias` tensors exist ONLY under
#: :data:`CANONICAL_VISION_PREFIX`.
DUPLICATE_VISION_PREFIX = "modalities.vision_perception.encoder.vision_model."

#: 50 orphan `attn.wqkv.bias [4608]` tensors with no matching weight: a TP=8
#: fused column-parallel layout that nothing reads. Skipped.
STALE_FUSED_QKV_PREFIX = "modalities.vision_perception.encoder.vision_module."

#: 72 audio tensors. Consumed by the audio tower when the model builds one
#: (`load_audio=True`); otherwise skipped with a warning, see
#: :func:`_warn_unconsumed_audio`.
AUDIO_PREFIX = "modalities.soft_audio_perception."

#: Where the tower tensors actually live under :data:`AUDIO_PREFIX`. Verified
#: against the shipped checkpoint index: exactly 72 keys, all BF16, no
#: quantization scales -- the tower stays BF16 (see musespark_open_audio.py for
#: why the exemption cannot come from hf_quant_config.json's exclude_modules).
AUDIO_ENCODER_PREFIX = f"{AUDIO_PREFIX}interaction_encoder."

#: ln_1 (2) + ln_2 (2) + q/k/v proj (6) + out_proj (2) + mlp.c_fc (2) +
#: mlp.c_proj (2). 16 x 50 layers = 800, plus the 8 tensors outside the layer
#: stack, is the 808 the tower consumes.
_TENSORS_PER_VISION_LAYER = 16


class WeightAccounting:
    """Per-load tally of every checkpoint key: consumed, or skipped and why."""

    def __init__(self) -> None:
        self.consumed: set = set()
        self.ignored_out_of_range: set = set()
        self.ignored_other: set = set()
        # Perception subtrees, split by reason. A key lands in exactly one.
        self.ignored_vision_unused: set = set()
        self.ignored_duplicate_vision: set = set()
        self.ignored_stale_fused_qkv: set = set()
        self.ignored_audio: set = set()

    @property
    def ignored_perception(self) -> set:
        """Every perception key this load did not consume, whatever the reason.

        Derived from the per-reason buckets, for callers that only ask whether
        the towers were left alone.
        """
        return (
            self.ignored_vision_unused
            | self.ignored_duplicate_vision
            | self.ignored_stale_fused_qkv
            | self.ignored_audio
        )

    def summary(self) -> Dict[str, int]:
        return {
            "consumed": len(self.consumed),
            "ignored_out_of_range": len(self.ignored_out_of_range),
            "ignored_perception": len(self.ignored_perception),
            "ignored_other": len(self.ignored_other),
            "ignored_vision_unused": len(self.ignored_vision_unused),
            "ignored_duplicate_vision": len(self.ignored_duplicate_vision),
            "ignored_stale_fused_qkv": len(self.ignored_stale_fused_qkv),
            "ignored_audio": len(self.ignored_audio),
        }


def _classify_perception_key(
    key: str, accounting: WeightAccounting, load_vision: bool, load_audio: bool
) -> None:
    """Route one perception key to its named bucket.

    Longest prefix first: :data:`AUDIO_PREFIX` and the two
    `modalities.vision_perception.encoder.*` prefixes all sit under
    `modalities.`, so a shortest-first chain would swallow them all.
    """
    if key.startswith(DUPLICATE_VISION_PREFIX):
        accounting.ignored_duplicate_vision.add(key)
        return
    if key.startswith(STALE_FUSED_QKV_PREFIX):
        accounting.ignored_stale_fused_qkv.add(key)
        return
    if key.startswith(AUDIO_PREFIX):
        # Left unbucketed when an audio tower will consume it, exactly as the
        # canonical vision keys are for the vision walk.
        if not load_audio:
            accounting.ignored_audio.add(key)
        return
    if key in (VISION_PROJECTION_KEY, VISION_PROJECTION_ALIAS) or key.startswith(
        CANONICAL_VISION_PREFIX
    ):
        # Left unbucketed when a tower will consume it, exactly as in-range text
        # keys are left unbucketed for the decoder walk.
        if not load_vision:
            accounting.ignored_vision_unused.add(key)
        return
    if key.startswith("modalities."):
        # A checkpoint that grows a new modality must fail loudly rather than
        # load a model that silently ignores part of itself.
        raise ValueError(
            f"MuseSparkOpen checkpoint has an unknown 'modalities.' subtree: {key!r}. "
            "Handle it in musespark_open_weight_mapper.py rather than "
            "letting it be skipped."
        )
    # "audio_model." is in PERCEPTION_KEY_PREFIXES but matches nothing in the
    # released checkpoints (the audio prefix is AUDIO_PREFIX above). It is not
    # vision, so it goes to ignored_other.
    accounting.ignored_other.add(key)


def _warn_unconsumed_audio(accounting: WeightAccounting) -> None:
    """Announce the deliberately unconsumed audio tensors, once per load."""
    count = len(accounting.ignored_audio)
    if not count:
        return
    logger.warning(
        f"MuseSparkOpen: SKIPPING {count} audio-perception tensors under "
        f"'{AUDIO_ENCODER_PREFIX}'. This model instance built NO audio tower "
        "(text-only model, mm encoder disabled, or config "
        "audio_perception_modality_args.input_mode != 'wav'), so audio content "
        "in a request will NOT be perceived. The tensors are present in the "
        "checkpoint and are deliberately left unconsumed; this is not a load "
        "error."
    )


def classify_keys(
    keys, num_hidden_layers: int, load_vision: bool = False, load_audio: bool = False
) -> WeightAccounting:
    """Split checkpoint keys into in-range text keys and ignorable ones.

    In-range keys are left out of the accounting object; the loader adds them
    to `consumed` as it uses them, and anything left over is an error.

    Args:
        load_vision: when set, the canonical tower keys and the vision
            projection are left unbucketed for the tower walk to consume,
            instead of being recorded as skipped.
        load_audio: when set, the audio tensors are left unbucketed for the
            audio-tower walk instead of being skipped behind the guard.
    """
    accounting = WeightAccounting()
    for key in keys:
        if key.startswith(PERCEPTION_KEY_PREFIXES) or key == VISION_PROJECTION_ALIAS:
            _classify_perception_key(key, accounting, load_vision, load_audio)
            continue
        match = _LAYER_KEY.match(key)
        if match is not None and int(match.group(1)) >= num_hidden_layers:
            accounting.ignored_out_of_range.add(key)
            continue
        if key.endswith(_IGNORED_SUFFIXES):
            accounting.ignored_other.add(key)
    _warn_unconsumed_audio(accounting)
    return accounting


def rename_open_format_weights(weights) -> bool:
    """Rename open-export keys to the source spelling, in place.

    A no-op (returns False) on a source-format checkpoint, so both entry
    points can call it unconditionally and the multimodal loader's call makes
    the text loader's one idempotent. In place on purpose: the lazy weights dict
    (`_LazySafetensorsWeights`) owns the open file handles its slices read
    from, so a derived dict via `take_ownership` would clear them.

    The rename map is total and injective (`build_open_format_rename`
    raises otherwise), and no open-export key is the source spelling of a
    *different* tensor (the two namespaces share only `lm_head.weight`,
    which maps to itself), so the loop cannot overwrite a not-yet-renamed
    entry; the guard turns that invariant into an error rather than a
    corrupted load if a future re-export breaks it.
    """
    keys = list(weights.keys())
    if OPEN_FORMAT_SENTINEL_KEY not in set(keys):
        return False
    rename = build_open_format_rename(keys)
    for open_key, source_key in rename.items():
        if source_key == open_key:
            continue
        if source_key in weights:
            raise ValueError(
                f"Muse Spark Open format: renaming {open_key!r} would overwrite the "
                f"existing key {source_key!r}"
            )
        value = weights[open_key]
        del weights[open_key]
        weights[source_key] = value
    logger.info(
        f"MuseSparkOpen: open-format checkpoint detected; renamed {len(rename)} "
        "tensors to the source spelling."
    )
    return True


def _load_router(moe, weight: torch.Tensor, num_experts: int, hidden_size: int) -> None:
    """Load the router, whichever way this export oriented it.

    The source export stores `[hidden, experts]` (transposed into TRT-LLM's
    `Linear` `[out, in]` here); the open export already stores
    `[experts, hidden]`. The two shapes are distinguishable as long as the
    expert count differs from the hidden size, which the guard pins down --
    silently transposing a square router would be unfalsifiable at load time.
    """
    if num_experts == hidden_size:
        raise ValueError(
            "MuseSparkOpen: cannot infer router orientation when num_experts == "
            f"hidden_size == {num_experts}"
        )
    if weight.shape == (num_experts, hidden_size):
        _load_linear(moe.router, weight.contiguous())
    elif weight.shape == (hidden_size, num_experts):
        _load_linear(moe.router, weight.transpose(0, 1).contiguous())
    else:
        raise ValueError(
            f"MuseSparkOpen: router weight shape {tuple(weight.shape)} matches neither "
            f"[experts, hidden] nor [hidden, experts] for {num_experts} experts and "
            f"hidden size {hidden_size}"
        )


def interleave_rope_rows(tensor: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Convert q/k rows from the half rotary layout back to interleaved.

    The open export stores every `q_proj`/`k_proj` with the per-head rows
    in the half (NeoX) rotary order -- component `i` of a head's first half
    at row `i`, its partner at row `half + i` -- because that is the layout
    the composite HF model applies RoPE in. The runtime applies RoPE in the
    source interleaved order (pairs at rows `2i`/`2i + 1`), so the rows
    are put back: `out[2i] = in[i]`, `out[2i + 1] = in[half + i]`,
    per head.

    Applied uniformly to every layer, exactly as the exporter permuted them:
    on NoPE layers the permutation is semantically neutral (q and k are
    permuted identically, so their per-head dot products are unchanged), and
    only RoPE gives the row order meaning. Works for weights `[heads *
    head_dim, in]` and for per-head biases `[heads * head_dim]`.
    """
    if tensor.shape[0] % head_dim:
        raise ValueError(
            f"MuseSparkOpen: cannot re-interleave rows of shape {tuple(tensor.shape)} "
            f"with head_dim {head_dim}"
        )
    heads = tensor.shape[0] // head_dim
    halves = tensor.reshape(heads, 2, head_dim // 2, *tensor.shape[1:])
    return torch.stack((halves[:, 0], halves[:, 1]), dim=2).reshape(tensor.shape)


def _materialize(weights, key: str, accounting: WeightAccounting) -> torch.Tensor:
    """Read one lazy slice in full and mark it consumed."""
    accounting.consumed.add(key)
    value = weights[key]
    return value[:] if hasattr(value, "get_shape") else value


def _slice_experts(
    weights, key: str, expert_start: int, expert_end: int, accounting: WeightAccounting
) -> torch.Tensor:
    """Read only this rank's expert range out of a stacked expert tensor."""
    accounting.consumed.add(key)
    value = weights[key]
    return value[expert_start:expert_end]


def _load_linear(module, weight: torch.Tensor) -> None:
    module.load_weights([{"weight": weight}])


def _copy_param(param: torch.nn.Parameter, value: torch.Tensor) -> None:
    param.data.copy_(value.to(param.dtype))


def _local_expert_range(experts_module, num_experts: int) -> Tuple[int, int]:
    """Contiguous global expert ids owned by this rank."""
    expert_ids = getattr(experts_module, "initial_local_expert_ids", None)
    if not expert_ids:
        return 0, num_experts
    return int(min(expert_ids)), int(max(expert_ids)) + 1


def build_expert_weights(
    weights,
    prefix: str,
    expert_start: int,
    expert_end: int,
    num_experts: int,
    intermediate_size: int,
    is_quantized: bool,
    accounting: WeightAccounting,
) -> Dict[str, Any]:
    """Explode this layer's stacked expert tensors into per-expert entries.

    Checkpoint layout: `w13_weight[e]` holds the gate rows in `[0:I)` and
    the up rows in `[I:2I)` (packed FP4 halves the last dim). The global
    scales are per half: `w13_weight_scale_2[e, 0]` is gate, `[e, 1]` is
    up. Dropping, swapping or maxing those two columns silently rescales the
    up projection, so they are threaded through separately.
    """
    entries: Dict[str, Any] = {}
    w13 = _slice_experts(weights, f"{prefix}.w13_weight", expert_start, expert_end, accounting)
    w2 = _slice_experts(weights, f"{prefix}.w2_weight", expert_start, expert_end, accounting)
    for local_idx, expert_id in enumerate(range(expert_start, expert_end)):
        entries[f"{expert_id}.w1.weight"] = w13[local_idx, :intermediate_size]
        entries[f"{expert_id}.w3.weight"] = w13[local_idx, intermediate_size:]
        entries[f"{expert_id}.w2.weight"] = w2[local_idx]

    if not is_quantized:
        return entries

    w13_scale = _slice_experts(
        weights, f"{prefix}.w13_weight_scale", expert_start, expert_end, accounting
    )
    w2_scale = _slice_experts(
        weights, f"{prefix}.w2_weight_scale", expert_start, expert_end, accounting
    )
    for local_idx, expert_id in enumerate(range(expert_start, expert_end)):
        entries[f"{expert_id}.w1.weight_scale"] = w13_scale[local_idx, :intermediate_size]
        entries[f"{expert_id}.w3.weight_scale"] = w13_scale[local_idx, intermediate_size:]
        entries[f"{expert_id}.w2.weight_scale"] = w2_scale[local_idx]

    # Global scales are tiny; read all experts so the layer-level input-scale
    # maximum is identical on every rank.
    w13_scale_2 = _materialize(weights, f"{prefix}.w13_weight_scale_2", accounting)
    w2_scale_2 = _materialize(weights, f"{prefix}.w2_weight_scale_2", accounting)
    w13_input_scale = _materialize(weights, f"{prefix}.w13_input_scale", accounting)
    w2_input_scale = _materialize(weights, f"{prefix}.w2_input_scale", accounting)
    for expert_id in range(num_experts):
        entries[f"{expert_id}.w1.weight_scale_2"] = w13_scale_2[expert_id, 0]
        entries[f"{expert_id}.w3.weight_scale_2"] = w13_scale_2[expert_id, 1]
        entries[f"{expert_id}.w2.weight_scale_2"] = w2_scale_2[expert_id]
        entries[f"{expert_id}.w1.input_scale"] = w13_input_scale[expert_id, 0]
        entries[f"{expert_id}.w3.input_scale"] = w13_input_scale[expert_id, 1]
        entries[f"{expert_id}.w2.input_scale"] = w2_input_scale[expert_id]
    return entries


def load_layer(
    model,
    layer,
    layer_idx: int,
    weights,
    accounting: WeightAccounting,
    open_format: bool = False,
) -> None:
    config = model.config
    prefix = f"model.layers.{layer_idx}"
    attn_prefix = f"{prefix}.self_attn"
    ffn_prefix = f"{prefix}.feed_forward"

    _copy_param(
        layer.pre_qkv_proj_norm.weight,
        _materialize(weights, f"{attn_prefix}.pre_qkv_proj_norm.weight", accounting),
    )

    q_weight = _materialize(weights, f"{attn_prefix}.q_proj.weight", accounting)
    gate_weight = _materialize(weights, f"{attn_prefix}.o_gate_proj.weight", accounting)
    k_weight = _materialize(weights, f"{attn_prefix}.k_proj.weight", accounting)
    v_weight = _materialize(weights, f"{attn_prefix}.v_proj.weight", accounting)
    if open_format:
        # The open export stores q/k rows in the half rotary layout; the
        # runtime rotates interleaved pairs. Gate, v and o are untouched by
        # RoPE and keep their rows.
        q_weight = interleave_rope_rows(q_weight, config.head_dim)
        k_weight = interleave_rope_rows(k_weight, config.head_dim)
    q_gate = head_interleaved_q_gate(
        q_weight, gate_weight, config.num_attention_heads, config.head_dim
    )
    layer.self_attn.qkv_proj.load_weights(
        [
            {"weight": q_gate},
            {"weight": k_weight},
            {"weight": v_weight},
        ]
    )
    del q_weight, gate_weight, q_gate

    _load_linear(
        layer.self_attn.o_proj, _materialize(weights, f"{attn_prefix}.o_proj.weight", accounting)
    )

    _copy_param(layer.add_attn.gate, _materialize(weights, f"{prefix}.add_attn.gate", accounting))
    _copy_param(layer.add_ffn.gate, _materialize(weights, f"{prefix}.add_ffn.gate", accounting))
    _copy_param(
        layer.pre_ffn_norm.weight,
        _materialize(weights, f"{prefix}.pre_ffn_norm.weight", accounting),
    )
    _copy_param(
        layer.post_ffn_norm.weight,
        _materialize(weights, f"{prefix}.post_ffn_norm.weight", accounting),
    )

    moe = layer.feed_forward
    # Router orientation differs per export (source: [hidden, experts]; open:
    # [experts, hidden]); _load_router transposes into TRT-LLM's [out, in] as
    # needed.
    router_weight = _materialize(weights, f"{ffn_prefix}.router.weight", accounting)
    _load_router(moe, router_weight, config.num_experts, config.hidden_size)
    _copy_param(moe.expert_bias, _materialize(weights, f"{ffn_prefix}.expert_bias", accounting))
    _load_linear(moe.pooler, _materialize(weights, f"{ffn_prefix}.pooler.weight", accounting))
    _load_linear(moe.unpooler, _materialize(weights, f"{ffn_prefix}.unpooler.weight", accounting))
    _copy_param(
        moe.pre_expert_norm.weight,
        _materialize(weights, f"{ffn_prefix}.pre_expert_norm.weight", accounting),
    )
    _copy_param(
        moe.post_expert_norm_weight,
        _materialize(weights, f"{ffn_prefix}.post_expert_norm.weight", accounting),
    )

    experts_prefix = f"{ffn_prefix}.experts"
    is_quantized = f"{experts_prefix}.w13_weight_scale" in weights
    expert_start, expert_end = _local_expert_range(moe.experts, config.num_experts)
    expert_weights = build_expert_weights(
        weights,
        experts_prefix,
        expert_start,
        expert_end,
        config.num_experts,
        config.intermediate_size,
        is_quantized,
        accounting,
    )
    moe.experts.load_weights([expert_weights])
    expert_weights.clear()


def _load_linear_with_bias(module, weights, prefix: str, accounting: WeightAccounting) -> None:
    """Load a `Linear` whose checkpoint entry carries a bias."""
    module.load_weights(
        [
            {
                "weight": _materialize(weights, f"{prefix}.weight", accounting),
                "bias": _materialize(weights, f"{prefix}.bias", accounting),
            }
        ]
    )


def _load_layer_norm(module, weights, prefix: str, accounting: WeightAccounting) -> None:
    """Load a LayerNorm's weight and bias.

    The tower is LayerNorm throughout, unlike the decoder's RMSNorm: dropping
    the bias here would be a silent, plausible-looking wrong answer rather than
    a load error, which is why it goes through its own helper.
    """
    _copy_param(module.weight, _materialize(weights, f"{prefix}.weight", accounting))
    _copy_param(module.bias, _materialize(weights, f"{prefix}.bias", accounting))


def _submodule(layer, path: str):
    """Resolve a dotted attribute path against one encoder layer."""
    module = layer
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _load_vision_layer(
    layer,
    layer_idx: int,
    weights,
    accounting: WeightAccounting,
    open_format: bool = False,
    head_dim: int = 0,
) -> int:
    """Load one of the 50 encoder layers; returns the tensors it consumed.

    Checkpoint suffix to module attribute goes through the tower's own
    `CKPT_TO_MODULE`, not a second copy of the mapping here. Two of the six
    names differ because the repo module fixes them -- the tower composes
    `modules.mlp.MLP`, whose projections are `up_proj` / `down_proj`, and
    `attention.Attention`, whose output projection is `o_proj` -- and a
    private copy of that mapping in this file is exactly how they came to
    disagree in the first place.
    """
    from ...musespark_open_vision import CKPT_TO_MODULE

    prefix = f"{CANONICAL_VISION_PREFIX}encoder.layers.{layer_idx}"

    _load_layer_norm(layer.ln_1, weights, f"{prefix}.ln_1", accounting)
    _load_layer_norm(layer.ln_2, weights, f"{prefix}.ln_2", accounting)

    # Separate q/k/v in the checkpoint, fused qkv_proj in the module: the same
    # shape mismatch the decoder has, minus the output-gate interleave.
    attn_prefix = f"{prefix}.attn"
    qkv_entries = []
    for name in ("q", "k", "v"):
        weight = _materialize(weights, f"{attn_prefix}.{name}_proj.weight", accounting)
        bias = _materialize(weights, f"{attn_prefix}.{name}_proj.bias", accounting)
        if open_format and name in ("q", "k"):
            # Same half-to-interleaved row restore as the decoder; the tower's
            # axial 2D RoPE also rotates interleaved pairs, and v is untouched.
            weight = interleave_rope_rows(weight, head_dim)
            bias = interleave_rope_rows(bias, head_dim)
        qkv_entries.append({"weight": weight, "bias": bias})
    _submodule(layer, CKPT_TO_MODULE["attn.q_proj"]).load_weights(qkv_entries)
    _load_linear_with_bias(
        _submodule(layer, CKPT_TO_MODULE["attn.out_proj"]),
        weights,
        f"{attn_prefix}.out_proj",
        accounting,
    )

    for ckpt_name in ("mlp.c_fc", "mlp.c_proj"):
        _load_linear_with_bias(
            _submodule(layer, CKPT_TO_MODULE[ckpt_name]),
            weights,
            f"{prefix}.{ckpt_name}",
            accounting,
        )

    # 16 tensors per layer: 2 LayerNorms x2, 3 projections x2, out_proj x2,
    # 2 MLP projections x2. Asserting the count here is what catches a
    # checkpoint that renamed or dropped one of them.
    released = weights.mark_consumed(prefix)
    if released != _TENSORS_PER_VISION_LAYER:
        raise ValueError(
            f"MuseSparkOpen vision layer {layer_idx} released {released} tensors, "
            f"expected {_TENSORS_PER_VISION_LAYER}: the checkpoint layout for "
            "this layer is not what the tower was written against."
        )
    return released


def load_musespark_open_vision_weights(
    tower, weights, accounting: WeightAccounting, open_format: bool = False
) -> None:
    """Load the 808 canonical tower tensors plus the one projection.

    Consumes only :data:`CANONICAL_VISION_PREFIX` and the projection. The 658
    duplicates, the 50 stale fused-qkv biases and the 72 audio tensors were
    already bucketed by :func:`classify_keys` and are dropped here in one call
    so they never reach a module walk.
    """
    already_skipped = (
        accounting.ignored_duplicate_vision
        | accounting.ignored_stale_fused_qkv
        | accounting.ignored_audio
    )
    weights.mark_consumed_keys(sorted(already_skipped))

    _load_linear(
        tower.conv1_linear,
        _materialize(weights, f"{CANONICAL_VISION_PREFIX}conv1_linear.weight", accounting),
    )
    weights.mark_consumed(f"{CANONICAL_VISION_PREFIX}conv1_linear")

    # A bare tensor, not a submodule: mark_consumed() would look for
    # "...positional_embedding_vlm." and delete nothing, silently.
    pos_embed_key = f"{CANONICAL_VISION_PREFIX}positional_embedding_vlm"
    _copy_param(tower.pos_embed, _materialize(weights, pos_embed_key, accounting))
    weights.mark_consumed_keys([pos_embed_key])

    _load_layer_norm(tower.ln_pre, weights, f"{CANONICAL_VISION_PREFIX}ln_pre", accounting)
    weights.mark_consumed(f"{CANONICAL_VISION_PREFIX}ln_pre")

    for layer_idx, layer in enumerate(tower.encoder.layers):
        _load_vision_layer(
            layer,
            layer_idx,
            weights,
            accounting,
            open_format=open_format,
            head_dim=tower.config.head_dim,
        )

    _load_layer_norm(tower.ln_post, weights, f"{CANONICAL_VISION_PREFIX}ln_post", accounting)
    weights.mark_consumed(f"{CANONICAL_VISION_PREFIX}ln_post")

    # The adapter and the projection have no bias in the checkpoint.
    _load_linear(
        tower.adapter.c_fc,
        _materialize(weights, f"{CANONICAL_VISION_PREFIX}adapter.c_fc.weight", accounting),
    )
    _load_linear(
        tower.adapter.c_proj,
        _materialize(weights, f"{CANONICAL_VISION_PREFIX}adapter.c_proj.weight", accounting),
    )
    weights.mark_consumed(f"{CANONICAL_VISION_PREFIX}adapter")

    projection_key = (
        VISION_PROJECTION_KEY if VISION_PROJECTION_KEY in weights else VISION_PROJECTION_ALIAS
    )
    _load_linear(tower.vision_projection, _materialize(weights, projection_key, accounting))
    weights.mark_consumed_keys([projection_key])


def load_musespark_open_weights(
    model, weights, accounting=None, open_format: Optional[bool] = None
) -> Dict[str, int]:
    """Load the text decoder and return the key accounting summary.

    Args:
        accounting: an accounting object to extend instead of starting a fresh
            one. The multimodal model classifies once for both towers and
            passes the result through, so that the zero-unaccounted check at the end sees
            the whole checkpoint rather than the text half of it.
        open_format: whether the checkpoint uses the open-export layout (q/k
            rows in the half rotary order). None means "detect and rename
            here" -- the path text-only loads come through directly; the
            multimodal loader detects once for both towers and passes the
            result.
    """
    config = model.config
    if open_format is None:
        open_format = rename_open_format_weights(weights)
    if accounting is None:
        accounting = classify_keys(list(weights.keys()), config.num_hidden_layers)

    _load_linear(
        model.model.embed_tokens, _materialize(weights, "model.embed_tokens.weight", accounting)
    )
    _copy_param(model.model.norm.weight, _materialize(weights, "model.norm.weight", accounting))
    _load_linear(model.lm_head, _materialize(weights, "lm_head.weight", accounting))

    for layer_idx, layer in enumerate(model.model.layers):
        load_layer(model, layer, layer_idx, weights, accounting, open_format=open_format)

    unexpected = _unaccounted_keys(weights, accounting)
    if unexpected:
        raise ValueError(
            f"MuseSparkOpen checkpoint has {len(unexpected)} text tensors this "
            f"model never consumed, e.g. {sorted(unexpected)[:8]}"
        )

    # No cache-warming walk here on purpose. The frozen-constant caches in
    # MuseSparkOpenGatedResidual / MuseSparkOpenRMSNormWeightFirst are filled by the
    # loader's own post-load stages, which those modules implement directly.
    # A walk from here was actively harmful: it runs during loading, i.e. before
    # transform_weights, and calling a generic `cache_derived_state` on every
    # module reaches Linear -- whose implementation sets `_weights_transformed
    # = True` (modules/linear.py). transform_weights() then returns early, so
    # every quantized weight transform in the model is skipped, silently. The
    # loader's real walks guard on `_weights_removed` and run at the right
    # point; this one did neither.

    summary = accounting.summary()
    logger.info(f"MuseSparkOpen weight accounting: {summary}")
    return summary


def load_musespark_open_audio_weights(audio_encoder, weights, accounting: WeightAccounting) -> None:
    """Load the 72 audio-tower tensors under :data:`AUDIO_ENCODER_PREFIX`.

    The tower's module tree mirrors the checkpoint suffixes one for one, so
    the walk is the tower's own `state_dict` with the prefix prepended. All tensors are BF16
    in the checkpoint and the tower's parameters are BF16; `_copy_param`'s
    cast is a no-op. The tower is deliberately NOT quantized -- see the module
    docstring of `musespark_open_audio.py`.
    """
    from ...musespark_open_audio import audio_state_dict_to_checkpoint_keys

    state = dict(audio_encoder.named_parameters())
    consumed_keys: List[str] = []
    missing: List[str] = []
    for param_name, ckpt_key in audio_state_dict_to_checkpoint_keys(
        audio_encoder, AUDIO_ENCODER_PREFIX
    ):
        if ckpt_key not in weights:
            missing.append(ckpt_key)
            continue
        _copy_param(state[param_name], _materialize(weights, ckpt_key, accounting))
        consumed_keys.append(ckpt_key)
    if missing:
        raise ValueError(
            f"MuseSparkOpen audio tower: checkpoint is missing {len(missing)} tensors, "
            f"e.g. {missing[:4]}; refusing to serve a tower with zeroed weights"
        )
    extra = [
        key
        for key in weights.keys()
        if key.startswith(AUDIO_PREFIX) and key not in accounting.consumed
    ]
    if extra:
        raise ValueError(
            f"MuseSparkOpen audio tower: checkpoint carries {len(extra)} audio tensors "
            f"the tower has no home for, e.g. {sorted(extra)[:4]}; the checkpoint "
            "layout is not what the tower was written against"
        )
    weights.mark_consumed_keys(consumed_keys)


def load_musespark_open_omni_weights(model, weights) -> Dict[str, int]:
    """Load the audio tower, the vision tower and the text decoder from one checkpoint.

    The towers go first: they are the only consumers of the perception subtree,
    and dropping the skipped perception tensors before the decoder walk keeps
    peak host memory at the text model's footprint rather than the sum. The
    audio walk runs only when the model built a tower (`input_mode == "wav"`
    and the mm encoder enabled); otherwise the 72 tensors are skipped with a
    warning.
    """
    open_format = rename_open_format_weights(weights)
    audio_encoder = getattr(model, "audio_encoder", None)
    accounting = classify_keys(
        list(weights.keys()),
        model.config.num_hidden_layers,
        load_vision=True,
        load_audio=audio_encoder is not None,
    )
    if audio_encoder is not None:
        load_musespark_open_audio_weights(audio_encoder, weights, accounting)
    load_musespark_open_vision_weights(
        model.vision_tower, weights, accounting, open_format=open_format
    )
    return load_musespark_open_weights(model.llm, weights, accounting, open_format=open_format)


def missing_vision_keys(keys: List[str], num_vision_layers: int) -> List[str]:
    """Names of the tower tensors this checkpoint does not carry.

    The mirror of :func:`missing_in_range_keys` for the perception side: a
    checkpoint missing one attention bias must fail loudly rather than load a
    tower with a zeroed bias, which no benchmark would reliably catch.
    """
    present = set(keys)
    expected = [
        f"{CANONICAL_VISION_PREFIX}conv1_linear.weight",
        f"{CANONICAL_VISION_PREFIX}positional_embedding_vlm",
        f"{CANONICAL_VISION_PREFIX}ln_pre.weight",
        f"{CANONICAL_VISION_PREFIX}ln_pre.bias",
        f"{CANONICAL_VISION_PREFIX}ln_post.weight",
        f"{CANONICAL_VISION_PREFIX}ln_post.bias",
        f"{CANONICAL_VISION_PREFIX}adapter.c_fc.weight",
        f"{CANONICAL_VISION_PREFIX}adapter.c_proj.weight",
    ]
    for layer_idx in range(num_vision_layers):
        prefix = f"{CANONICAL_VISION_PREFIX}encoder.layers.{layer_idx}"
        for norm in ("ln_1", "ln_2"):
            expected += [f"{prefix}.{norm}.weight", f"{prefix}.{norm}.bias"]
        for name in ("q", "k", "v", "out"):
            expected += [
                f"{prefix}.attn.{name}_proj.weight",
                f"{prefix}.attn.{name}_proj.bias",
            ]
        for name in ("c_fc", "c_proj"):
            expected += [f"{prefix}.mlp.{name}.weight", f"{prefix}.mlp.{name}.bias"]
    missing = [name for name in expected if name not in present]
    if VISION_PROJECTION_KEY not in present and VISION_PROJECTION_ALIAS not in present:
        missing.append(VISION_PROJECTION_KEY)
    return missing


def _unaccounted_keys(weights, accounting: WeightAccounting) -> set:
    ignored = (
        accounting.ignored_out_of_range | accounting.ignored_perception | accounting.ignored_other
    )
    return set(weights.keys()) - accounting.consumed - ignored


def missing_in_range_keys(
    keys: List[str],
    num_hidden_layers: int,
    bf16_expert_layers: Optional[List[int]],
) -> List[str]:
    """Names of the in-range text tensors this checkpoint does not carry.

    A checkpoint that silently drops a projection or a scale must fail loudly
    rather than load a zero tensor.
    """
    present = set(keys)
    expected = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    bf16_layers = set(bf16_expert_layers or [])
    for layer_idx in range(num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        expected += [
            f"{prefix}.self_attn.pre_qkv_proj_norm.weight",
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_gate_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.add_attn.gate",
            f"{prefix}.add_ffn.gate",
            f"{prefix}.pre_ffn_norm.weight",
            f"{prefix}.post_ffn_norm.weight",
            f"{prefix}.feed_forward.router.weight",
            f"{prefix}.feed_forward.expert_bias",
            f"{prefix}.feed_forward.pooler.weight",
            f"{prefix}.feed_forward.unpooler.weight",
            f"{prefix}.feed_forward.pre_expert_norm.weight",
            f"{prefix}.feed_forward.post_expert_norm.weight",
            f"{prefix}.feed_forward.experts.w13_weight",
            f"{prefix}.feed_forward.experts.w2_weight",
        ]
        if layer_idx not in bf16_layers:
            expected += [
                f"{prefix}.feed_forward.experts.w13_weight_scale",
                f"{prefix}.feed_forward.experts.w2_weight_scale",
                f"{prefix}.feed_forward.experts.w13_weight_scale_2",
                f"{prefix}.feed_forward.experts.w2_weight_scale_2",
                f"{prefix}.feed_forward.experts.w13_input_scale",
                f"{prefix}.feed_forward.experts.w2_input_scale",
            ]
    return [name for name in expected if name not in present]
