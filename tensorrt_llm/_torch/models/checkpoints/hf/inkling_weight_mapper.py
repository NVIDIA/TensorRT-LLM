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
"""HF -> TensorRT-LLM weight mapping for the Inkling text tower.

Two responsibilities:

1. **Accounting (authoritative, CPU-testable).** :func:`inkling_expected_text_keys`
   and :func:`inkling_account_checkpoint` derive the exact set of ``model.llm.*``
   checkpoint keys the text loader consumes, and classify every checkpoint key as
   consumed-text / intentionally-deferred (audio, vision, MTP) / unaccounted.
   This is a direct port of the primary-source-verified Stage-1 spec and is
   pinned by ``tests/unit/_torch/modeling/test_modeling_inkling.py`` against the
   real checkpoint index (no GPU). It guarantees no missing q/k-norm, rel-bias,
   short-conv, route/global-scale or unpadded-logit tensor can hide.

2. **Name/layout remapping (the load path).** :class:`InklingHfWeightMapper`
   renames the checkpoint's SGLang-style keys (``wq_du``, ``w13_weight`` …) to
   the TRT-LLM module tree, fuses q/k/v into the attention ``qkv_proj``, and
   unfuses the NVFP4 routed experts (``w13_weight`` -> per-expert ``w1``/``w3``
   with their block scales) into the layout the fused-MoE loader expects.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

import torch

from tensorrt_llm._torch.configs.inkling import InklingTextConfig
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper

# NVFP4 two-level-scale maxima: E2M1 element max (6.0) and E4M3 block-scale max
# (448.0). ModelOpt stores the per-tensor activation ``input_scale`` as
# ``amax / (E2M1_MAX * E4M3_MAX)``; Inkling's checkpoint instead ships the raw
# ``.input_amax``, so the mapper must apply this conversion (see ``_map_expert``).
_NVFP4_E2M1_MAX = 6.0
_NVFP4_E4M3_MAX = 448.0

# Prefixes intentionally unused for the text-only GSM8K/MMLU bring-up.
INKLING_DEFERRED_PREFIXES: Tuple[str, ...] = (
    "model.audio.",
    "model.visual.",
    "model.mtp.",
)

# Per-layer checkpoint keys (relative to ``model.llm.layers.N.``), present in
# every one of the 66 decoder layers.
_ATTN_AND_NORM_KEYS: Tuple[str, ...] = (
    "attn.wq_du.weight",
    "attn.wk_dv.weight",
    "attn.wv_dv.weight",
    "attn.wr_du.weight",
    "attn.wo_ud.weight",
    "attn.q_norm.weight",
    "attn.k_norm.weight",
    "attn.k_sconv.weight",
    "attn.v_sconv.weight",
    "attn.rel_logits_proj.proj",
    "attn_norm.weight",
    "mlp_norm.weight",
    "attn_sconv.weight",
    "mlp_sconv.weight",
)

# Dense MLP (layers 0, 1).
_DENSE_MLP_KEYS: Tuple[str, ...] = (
    "mlp.w13_dn.weight",
    "mlp.w2_md.weight",
    "mlp.global_scale",
)

# MoE common (all MoE layers, bf16 or NVFP4).
_MOE_COMMON_KEYS: Tuple[str, ...] = (
    "mlp.experts.w13_weight",
    "mlp.experts.w2_weight",
    "mlp.gate.weight",
    "mlp.gate.bias",
    "mlp.gate.global_scale",
    "mlp.shared_experts.shared_w13_weight",
    "mlp.shared_experts.shared_w2_weight",
)

# NVFP4 sidecars attached to each routed-expert weight tensor (layers 3..65).
_NVFP4_SIDECARS: Tuple[str, ...] = (".input_amax", ".original_shape", ".scale",
                                    ".scale2")
_NVFP4_QUANTIZED_EXPERT_TENSORS: Tuple[str, ...] = ("mlp.experts.w13_weight",
                                                    "mlp.experts.w2_weight")

_NON_LAYER_TEXT_KEYS: Tuple[str, ...] = (
    "model.llm.embed.weight",
    "model.llm.embed_norm.weight",
    "model.llm.norm.weight",
    "model.llm.unembed.weight",
)


def _experts_are_nvfp4(layer_idx: int, exclude_modules: Set[str]) -> bool:
    """Routed experts of an MoE layer are NVFP4 unless explicitly excluded."""
    return f"model.llm.layers.{layer_idx}.mlp.experts" not in exclude_modules


def inkling_expected_text_keys(config: InklingTextConfig,
                               exclude_modules: Set[str]) -> Set[str]:
    """Exact set of ``model.llm.*`` checkpoint keys the text loader consumes."""
    keys: Set[str] = set(_NON_LAYER_TEXT_KEYS)
    for n in range(config.num_hidden_layers):
        pfx = f"model.llm.layers.{n}."
        for k in _ATTN_AND_NORM_KEYS:
            keys.add(pfx + k)
        if config.is_dense_layer(n):
            for k in _DENSE_MLP_KEYS:
                keys.add(pfx + k)
        else:
            for k in _MOE_COMMON_KEYS:
                keys.add(pfx + k)
            if _experts_are_nvfp4(n, exclude_modules):
                for base in _NVFP4_QUANTIZED_EXPERT_TENSORS:
                    for side in _NVFP4_SIDECARS:
                        keys.add(pfx + base + side)
    return keys


def inkling_account_checkpoint(all_keys: Set[str], config: InklingTextConfig,
                               exclude_modules: Set[str]) -> Dict[str, Set[str]]:
    """Classify every checkpoint key into consumed-text / deferred / unaccounted.

    ``unaccounted`` and ``missing`` must both be empty for the text tower to be
    fully and exactly consumed.
    """
    expected = inkling_expected_text_keys(config, exclude_modules)
    consumed_text = all_keys & expected
    deferred = {
        k
        for k in all_keys if k.startswith(INKLING_DEFERRED_PREFIXES)
    }
    unaccounted = all_keys - consumed_text - deferred
    missing = expected - all_keys
    return {
        "consumed_text": consumed_text,
        "deferred": deferred,
        "unaccounted": unaccounted,
        "missing": missing,
    }


def inkling_nvfp4_expert_layers(config: InklingTextConfig,
                                exclude_modules: Set[str]) -> List[int]:
    """Layers whose routed experts are stored as NVFP4 (expected: 3..65)."""
    return [
        n for n in range(config.num_hidden_layers)
        if not config.is_dense_layer(n) and _experts_are_nvfp4(
            n, exclude_modules)
    ]


# ---------------------------------------------------------------------------
# Load path
# ---------------------------------------------------------------------------
# Simple 1:1 renames from the (``model.llm.`` stripped) checkpoint name to the
# TRT-LLM module tree.
_SIMPLE_RENAMES = {
    "embed.weight": "model.embed_tokens.weight",
    "embed_norm.weight": "model.embed_norm.weight",
    "norm.weight": "model.norm.weight",
    "unembed.weight": "lm_head.weight",
}

# Per-layer renames (regex on the ``layers.N.<rest>`` tail -> TRT name tail).
# q/k/v map to the standard separate HF names at the ``attn.`` level; the fused
# ``qkv_proj`` Linear's loader collects attn.q_proj/k_proj/v_proj via its
# special-handling callback and fuses them. Same for gate_up_proj <- gate_proj +
# up_proj (the dense w13_dn tensor is pre-fused and is split in _map_dense_w13).
_LAYER_RENAMES = {
    "attn.wq_du.weight": "attn.q_proj.weight",
    "attn.wk_dv.weight": "attn.k_proj.weight",
    "attn.wv_dv.weight": "attn.v_proj.weight",
    "attn.wo_ud.weight": "attn.o_proj.weight",
    "attn.wr_du.weight": "attn.r_proj.weight",
    "attn.q_norm.weight": "attn.q_norm.weight",
    "attn.k_norm.weight": "attn.k_norm.weight",
    "attn.k_sconv.weight": "attn.k_sconv.weight",
    "attn.v_sconv.weight": "attn.v_sconv.weight",
    "attn.rel_logits_proj.proj": "attn.rel_logits_proj",
    "attn_norm.weight": "attn_norm.weight",
    "mlp_norm.weight": "mlp_norm.weight",
    "attn_sconv.weight": "attn_sconv.weight",
    "mlp_sconv.weight": "mlp_sconv.weight",
    # dense (w13_dn is split in _map_dense_w13; w2_md -> down_proj)
    "mlp.w2_md.weight": "mlp.down_proj.weight",
    "mlp.global_scale": "mlp.global_scale",
    # moe (non-expert)
    "mlp.gate.weight": "mlp.gate.weight",
    "mlp.gate.bias": "mlp.gate.bias",
    "mlp.gate.global_scale": "mlp.gate.global_scale",
    "mlp.shared_experts.shared_w13_weight": "mlp.shared_experts.shared_w13",
    "mlp.shared_experts.shared_w2_weight": "mlp.shared_experts.shared_w2",
}

_EXPERT_RE = re.compile(
    r"layers\.(\d+)\.mlp\.experts\.(w13_weight|w2_weight)(\.\w+)?$")
_DENSE_W13_RE = re.compile(r"layers\.(\d+)\.mlp\.w13_dn\.weight$")


def _split_interleaved_gate_up(t: torch.Tensor,
                               dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split an Inkling gate/up-INTERLEAVED fused tensor into ``(gate, up)`` STRIDED
    VIEWS (no copy) along ``dim``: gate = even indices, up = odd indices.

    The Inkling checkpoint (SGLang ``inference_moe_w13_interleaved=True``, the
    default and the layout this NVFP4 checkpoint ships) stores every fused
    gate+up weight with the two projections INTERLEAVED along the output
    (``2*inter``) dim: ``[g0, u0, g1, u1, ...]``. SGLang's default SwiGLU reads it
    as ``silu(z[..., ::2]) * z[..., 1::2]``. TRT-LLM's fused gate_up / fused-MoE
    loaders instead want separate gate/up, and the old mapper split the fused
    tensor with a plain contiguous ``chunk(2)`` (``[first half | second half]``),
    which pairs the WRONG gate/up channels in every dense-MLP, routed-expert and
    shared-expert SwiGLU -> incoherent assembled text (invisible to isolated
    single-layer tests that made the same contiguous mis-read; reference-loop
    drift). Matches ``sglang .../inkling_common/util.py::deinterleave_gate_up``.

    Returns STRIDED VIEWS rather than a contiguous copy on purpose: the fused-MoE
    / gate_up loaders shard each rank's slice then call ``.contiguous()`` on that
    small shard (see quantization.py ``load_expert_w3_w1_weight``), so no
    full-tensor host copy is needed. A contiguous de-interleave here instead
    materialized a private per-rank copy of the ~hundreds-of-GiB fused w13,
    doubling host memory and OOM-killing the TP=4 load. Reorders whole output
    rows only -> valid for a packed NVFP4 weight and its per-block fp8 scale.
    """
    dim = dim % t.dim()
    if t.shape[dim] % 2 != 0:
        raise ValueError(
            f"cannot split odd gate/up dim {dim}: {tuple(t.shape)}")
    even = [slice(None)] * t.dim()
    odd = [slice(None)] * t.dim()
    even[dim] = slice(0, None, 2)
    odd[dim] = slice(1, None, 2)
    return t[tuple(even)], t[tuple(odd)]


@register_mapper("HF", "InklingForConditionalGeneration")
class InklingHfWeightMapper(HfWeightMapper):
    """Renames Inkling checkpoint keys to the TRT-LLM module tree.

    Runs after ``filter_weights("model.llm", ...)`` in the model's
    ``load_weights`` (so incoming keys start at ``layers.N.…`` / ``embed.weight``
    …). The NVFP4 routed experts are unfused from the checkpoint's stacked,
    gate+up-fused ``w13_weight [E, 2*inter, hidden/2]`` into the per-expert
    ``w1``/``w3`` layout (plus block ``weight_scale``, per-expert
    ``weight_scale_2`` and ``input_scale``) that the fused-MoE loader consumes.
    """

    def preprocess_weights(self, weights: Dict) -> Dict:
        new_weights: Dict[str, torch.Tensor] = {}
        unpadded_vocab = int(
            getattr(self.config.pretrained_config, "unpadded_vocab_size",
                    self.config.pretrained_config.vocab_size))
        for name, tensor in weights.items():
            if name in _SIMPLE_RENAMES:
                if name == "unembed.weight" and tensor.shape[0] > unpadded_vocab:
                    # The checkpoint LM-head matrix is padded to vocab_size
                    # (201024); the text tower emits logits only over the
                    # unpadded vocab (200058). Dropping the padding rows here is
                    # exactly the required "slice logits to unpadded" (logit[i]
                    # = h @ unembed[i]), and lets LMHead(num_embeddings=200058)
                    # load without a shape mismatch. embed_tokens keeps the full
                    # matrix (built at vocab_size), so input ids stay in range.
                    tensor = tensor[:unpadded_vocab]
                new_weights[_SIMPLE_RENAMES[name]] = tensor
                continue

            expert_match = _EXPERT_RE.search(name)
            if expert_match is not None:
                self._map_expert(name, tensor, expert_match, new_weights)
                continue

            dense_match = _DENSE_W13_RE.search(name)
            if dense_match is not None:
                # Dense w13_dn is gate/up-INTERLEAVED [g0,u0,...] along the output
                # (2*inter) dim; split into gate (even rows) / up (odd rows) for
                # the fused gate_up_proj loader (strided views, no copy).
                layer_idx = dense_match.group(1)
                gate, up = _split_interleaved_gate_up(tensor, dim=0)
                new_weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate
                new_weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up
                continue

            # shared_experts.shared_w13_weight loads RAW (interleaved) via
            # _LAYER_RENAMES; the gate/up interleave is undone by the strided split
            # in InklingSharedExperts.forward (zero-copy, param materialized once).

            m = re.match(r"layers\.(\d+)\.(.*)$", name)
            if m is not None:
                layer_idx, tail = m.group(1), m.group(2)
                trt_tail = _LAYER_RENAMES.get(tail, tail)
                new_weights[f"model.layers.{layer_idx}.{trt_tail}"] = tensor
                continue

            # Unknown key: keep as-is so any mismatch surfaces loudly at load.
            new_weights[name] = tensor
        return new_weights

    def _map_expert(self, name: str, tensor: torch.Tensor,
                    match: "re.Match", out: Dict) -> None:
        """Unfuse a stacked expert tensor into per-expert fused-MoE keys.

        ``w13_weight[e]`` is ``[2*inter, hidden]`` (gate rows first, up rows
        second, per HF ``InklingExperts``); split into ``w1`` (gate) and ``w3``
        (up). ``w2_weight[e]`` is the down projection. NVFP4 sidecars map to the
        fused-MoE scale names: ``.scale`` -> ``weight_scale`` (block),
        ``.scale2`` -> ``weight_scale_2`` (per-expert), ``.input_amax`` ->
        ``input_scale``. ``.original_shape`` is metadata and is dropped.
        """
        layer_idx, which, sidecar = match.group(1), match.group(2), match.group(3)
        prefix = f"model.layers.{layer_idx}.mlp.experts"

        scale_name = {
            None: "weight",
            ".scale": "weight_scale",
            ".scale2": "weight_scale_2",
            ".input_amax": "input_scale",
        }.get(sidecar)
        if scale_name is None:  # .original_shape -> drop (layout metadata)
            return

        if sidecar == ".input_amax":
            # Inkling's NVFP4 checkpoint stores the routed-expert activation
            # calibration as a RAW amax (``.input_amax``). The fused-MoE loader
            # (``NVFP4FusedMoEMethod.process_weights_after_loading`` ->
            # ``fc31_input_scale = 1 / max_e(input_scale)``) and
            # ``torch.ops.trtllm.fp4_quantize`` expect the ModelOpt per-tensor
            # activation ``input_scale = amax / (E2M1_MAX * E4M3_MAX)``, so the
            # activation global scale ``1 / max_e(input_scale)`` lands the e4m3
            # activation block scales in range. Without this conversion the global
            # scale is (E2M1_MAX*E4M3_MAX)=2688x too small, the activation fp4 block
            # scales underflow e4m3, and every routed expert loses ~0.62 rel_rms vs
            # the bf16 ground truth (7.6x SGLang) -- the baseline GSM8K gap.
            # Mirrors sglang inkling.py:1222 / inkling_common/dense_mlp.py:497
            # (``input_scale = input_amax / (6.0 * 448.0)``). Weight/block-scale/
            # scale2 layout is already correct (positional bisection: 24/24 L3
            # experts element-wise identical), so ONLY the activation input scale
            # needs this fix.
            tensor = tensor.to(torch.float32) / (_NVFP4_E2M1_MAX * _NVFP4_E4M3_MAX)

        n_experts = int(getattr(self.config.pretrained_config,
                                "n_routed_experts", tensor.shape[0]))
        projs = ("w1", "w3") if which == "w13_weight" else ("w2",)

        def _assign(e, vals):
            for proj, val in zip(projs, vals):
                out[f"{prefix}.{e}.{proj}.{scale_name}"] = val

        # Three sidecar shapes: per-expert multi-dim weight/block-scale (chunk
        # w13 into gate/up along the out dim), per-expert scalar weight_scale_2
        # (same value for gate and up), and a single global input_amax scalar
        # broadcast to every expert/proj.
        if tensor.dim() >= 2 and tensor.shape[0] == n_experts:
            for e in range(n_experts):
                if which == "w13_weight":
                    # w13 (packed fp4 weight AND its per-block fp8 scale) is
                    # gate/up-INTERLEAVED [g0,u0,...] along the per-expert output
                    # (2*inter) dim; split into w1 (gate = even rows) / w3 (up =
                    # odd rows) as strided views (no copy). Reorders whole rows, so
                    # it is correct for both the uint8 weight and the fp8 scale.
                    per = _split_interleaved_gate_up(tensor[e], dim=0)
                else:
                    per = (tensor[e], )
                _assign(e, per)
        elif tensor.dim() >= 1 and tensor.shape[0] == n_experts:
            for e in range(n_experts):
                _assign(e, (tensor[e], ) * len(projs))
        else:  # global scalar (input_amax [1]) -> broadcast to all experts
            val = tensor.reshape(-1)[0]
            for e in range(n_experts):
                _assign(e, (val, ) * len(projs))
