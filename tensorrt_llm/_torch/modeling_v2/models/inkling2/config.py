# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Inkling 2 text decoder's shape, as the family's ops and targets read it.

`InklingDims` is built once from the checkpoint's `config.json` text config and its
`hf_quant_config.json`, and is the only place a target reads model geometry from. It accepts both
`tensorrt_llm._torch.configs.Inkling2TextConfig` and a native Transformers `Inkling2TextConfig`,
which name a few fields differently (`conv_kernel_size` for `sconv_kernel_size`,
`moe_intermediate_size` for `intermediate_size`) and mark local layers by `layer_types` value
`hybrid_sliding` instead of listing `local_layer_ids`.

Expert precision is per layer and comes from `hf_quant_config.json`, whose `parameter_groups` name
every expert tensor with the format it is stored in. The released checkpoint stores layers 0-39
in NVFP4 and layers 40-47 in MXFP8; nothing here assumes that split.
"""

from __future__ import annotations

import enum
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

#: The expert weight tensor `hf_quant_config.json` names for each MoE layer.
_EXPERT_W13 = re.compile(r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.w13_weight$")


class ExpertFormat(str, enum.Enum):
    """How a layer's routed experts are stored and how their activations are quantized."""

    #: E2M1 values in packed uint8, an FP8 E4M3 scale per 16 values along K, and a per-expert FP32
    #: global scale `amax / outer_scale_denominator`. Activations are quantized per token with the
    #: 4/6 block-scale rule.
    NVFP4 = "nvfp4"
    #: FP8 E4M3 values and a UE8M0 scale per 32 values along K; activations are MXFP8 too.
    MXFP8 = "mxfp8"
    #: BF16 experts, as in a dequantized or randomly initialized test checkpoint.
    BF16 = "bfloat16"


def _first(config: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def expert_formats_from_quant_config(
    hf_quant_config: Optional[Mapping[str, Any]], num_layers: int
) -> tuple[ExpertFormat, ...]:
    """Each layer's expert format, from an Inkling `hf_quant_config.json` payload.

    A layer that no parameter group names keeps BF16 experts; a layer that two groups name is a
    malformed file.
    """
    formats: dict[int, ExpertFormat] = {}
    groups = ((hf_quant_config or {}).get("model") or {}).get("parameter_groups") or []
    for group in groups:
        compute = str(group.get("compute_dtype", "")).lower()
        fmt = {"nvfp4": ExpertFormat.NVFP4, "mxfp8": ExpertFormat.MXFP8}.get(compute)
        if fmt is None:
            continue
        for tensor in group.get("tensors", []):
            match = _EXPERT_W13.match(tensor)
            if match is None:
                continue
            layer = int(match.group(1))
            if layer in formats and formats[layer] is not fmt:
                raise ValueError(f"hf_quant_config.json gives layer {layer} two expert formats")
            formats[layer] = fmt
    out_of_range = sorted(i for i in formats if not 0 <= i < num_layers)
    if out_of_range:
        raise ValueError(
            f"hf_quant_config.json names experts of layers {out_of_range}, "
            f"outside the {num_layers} decoder layers"
        )
    return tuple(formats.get(i, ExpertFormat.BF16) for i in range(num_layers))


def read_hf_quant_config(checkpoint_dir: Optional[str]) -> Optional[dict]:
    """The checkpoint's `hf_quant_config.json`, or None when it has none."""
    if not checkpoint_dir:
        return None
    path = os.path.join(checkpoint_dir, "hf_quant_config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


@dataclass(frozen=True)
class InklingDims:
    """Geometry of the Inkling 2 text decoder, whole-model (not per rank)."""

    hidden_size: int
    num_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    sliding_window: int
    global_layers: tuple[int, ...]
    conv_kernel_size: int
    attn_res_block_size: int
    num_experts: int
    top_k: int
    expert_intermediate_size: int
    route_scale: float
    gate_softcap: float
    up_softcap: float
    logit_softcap: float
    qk_clip_threshold: float
    rel_clip_threshold: float
    log_scaling_alpha: float
    log_scaling_n_floor: float
    rms_norm_eps: float
    expert_formats: tuple[ExpertFormat, ...]

    @classmethod
    def from_config(
        cls, text_config: Any, hf_quant_config: Optional[Mapping[str, Any]] = None
    ) -> "InklingDims":
        """Read the text config (either config class) and the quant config payload."""
        num_layers = int(text_config.num_hidden_layers)
        local = getattr(text_config, "local_layer_ids", None)
        if local is not None:
            local = set(int(i) for i in local)
            global_layers = tuple(i for i in range(num_layers) if i not in local)
        else:
            layer_types = list(text_config.layer_types)
            global_layers = tuple(
                i for i, kind in enumerate(layer_types) if "sliding" not in str(kind).lower()
            )
        swa = (
            _first(text_config, "swa_num_attention_heads"),
            _first(text_config, "swa_num_key_value_heads"),
            _first(text_config, "swa_head_dim"),
        )
        full = (
            text_config.num_attention_heads,
            text_config.num_key_value_heads,
            text_config.head_dim,
        )
        if any(s is not None and s != f for s, f in zip(swa, full)):
            raise ValueError(
                f"sliding-window layers with a different attention shape {swa} from "
                f"the global layers {full} are not supported"
            )
        return cls(
            hidden_size=int(text_config.hidden_size),
            num_layers=num_layers,
            vocab_size=int(text_config.vocab_size),
            unpadded_vocab_size=int(
                _first(text_config, "unpadded_vocab_size", default=text_config.vocab_size)
            ),
            num_heads=int(text_config.num_attention_heads),
            num_kv_heads=int(text_config.num_key_value_heads),
            head_dim=int(text_config.head_dim),
            d_rel=int(text_config.d_rel),
            rel_extent=int(text_config.rel_extent),
            sliding_window=int(_first(text_config, "sliding_window_size", "sliding_window")),
            global_layers=global_layers,
            conv_kernel_size=int(_first(text_config, "sconv_kernel_size", "conv_kernel_size")),
            attn_res_block_size=int(text_config.n_layers_per_attn_res_block),
            num_experts=int(text_config.n_routed_experts),
            top_k=int(text_config.num_experts_per_tok),
            expert_intermediate_size=int(
                _first(text_config, "moe_intermediate_size", "intermediate_size")
            ),
            route_scale=float(_first(text_config, "route_scale", "routed_scaling_factor")),
            gate_softcap=float(text_config.mlp_gate_softcap),
            up_softcap=float(text_config.mlp_up_softcap),
            logit_softcap=float(text_config.final_logit_softcapping),
            qk_clip_threshold=float(text_config.qk_l2_clip_threshold),
            rel_clip_threshold=float(text_config.rel_l2_clip_threshold),
            log_scaling_alpha=float(text_config.log_scaling_alpha),
            log_scaling_n_floor=float(text_config.log_scaling_n_floor),
            rms_norm_eps=float(text_config.rms_norm_eps),
            expert_formats=expert_formats_from_quant_config(hf_quant_config, num_layers),
        )

    def is_global(self, layer: int) -> bool:
        """Whether `layer` attends over the whole context rather than the sliding window."""
        return layer in self.global_layers

    def rel_extent_of(self, layer: int) -> int:
        """Relative distances the layer's position bias covers: the window in a sliding layer."""
        return self.rel_extent if self.is_global(layer) else self.sliding_window

    def is_block_boundary(self, layer: int) -> bool:
        """Whether `layer` starts a new attention-residual block."""
        return layer % self.attn_res_block_size == 0

    @property
    def num_block_states(self) -> int:
        """Completed blocks the final output mix reads: the embedding plus one per later boundary."""
        return len([i for i in range(self.num_layers) if self.is_block_boundary(i)])
