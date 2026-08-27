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

1. Accounting. :func:`inkling_expected_text_keys` and
   :func:`inkling_account_checkpoint` derive the exact set of ``model.llm.*`` keys
   the text loader consumes and classify every checkpoint key as consumed-text /
   deferred (audio, vision, MTP) / unaccounted.

2. Name and layout remapping. :class:`InklingHfWeightMapper` renames the
   checkpoint's keys to the TRT-LLM module tree, fuses q/k/v into ``qkv_proj``,
   and unfuses the NVFP4 routed experts into the fused-MoE loader's layout.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

import torch

from tensorrt_llm._torch.configs.inkling import InklingTextConfig
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper

# NVFP4 two-level-scale maxima. ModelOpt stores the per-tensor activation
# ``input_scale`` as ``amax / (E2M1_MAX * E4M3_MAX)``; Inkling's checkpoint ships
# the raw ``.input_amax``, so ``_map_expert`` applies the conversion.
_NVFP4_E2M1_MAX = 6.0
_NVFP4_E4M3_MAX = 448.0

# Prefixes the text loader does not consume: the vision / audio towers load
# themselves (see modeling_inkling_multimodal.py) and MTP is not implemented.
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
_NVFP4_SIDECARS: Tuple[str, ...] = (".input_amax", ".original_shape", ".scale", ".scale2")
_NVFP4_QUANTIZED_EXPERT_TENSORS: Tuple[str, ...] = (
    "mlp.experts.w13_weight",
    "mlp.experts.w2_weight",
)

_NON_LAYER_TEXT_KEYS: Tuple[str, ...] = (
    "model.llm.embed.weight",
    "model.llm.embed_norm.weight",
    "model.llm.norm.weight",
    "model.llm.unembed.weight",
)


def _experts_are_nvfp4(layer_idx: int, exclude_modules: Set[str], quantized: bool = True) -> bool:
    """Routed experts of an MoE layer are NVFP4 unless explicitly excluded.

    ``quantized=False`` for the BF16 release, where the exclusion list is empty
    because nothing is quantized rather than because everything is.
    """
    if not quantized:
        return False
    return f"model.llm.layers.{layer_idx}.mlp.experts" not in exclude_modules


# Per-depth MTP keys, relative to ``model.mtp.layers.N.``. The draft block is
# structurally a DENSE trunk layer -- same attention/norm keys, same dense-MLP
# keys, verified against both shipped checkpoints -- plus three tensors that
# fold the previous depth's hidden state into this one's embedding.
_MTP_PREFIX_KEYS: Tuple[str, ...] = (
    "embed_norm.weight",
    "hidden_norm.weight",
    "input_proj.weight",
)


def inkling_expected_mtp_keys(config: InklingTextConfig, num_depths: int) -> Set[str]:
    """Exact set of ``model.mtp.*`` keys the draft chain consumes."""
    keys: Set[str] = set()
    for d in range(num_depths):
        pfx = f"model.mtp.layers.{d}."
        for k in _MTP_PREFIX_KEYS:
            keys.add(pfx + k)
        block = pfx + "transformer_block."
        for k in _ATTN_AND_NORM_KEYS:
            keys.add(block + k)
        # Always dense: SGLang forces the dense MLP for every MTP depth, and
        # the checkpoints agree (one global_scale per depth, no expert tensors).
        for k in _DENSE_MLP_KEYS:
            keys.add(block + k)
    return keys


def inkling_expected_text_keys(
    config: InklingTextConfig, exclude_modules: Set[str], quantized: bool = True
) -> Set[str]:
    """Exact set of ``model.llm.*`` checkpoint keys the text loader consumes.

    ``quantized=False`` for the BF16 release, which ships no scale sidecars.
    """
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
            if _experts_are_nvfp4(n, exclude_modules, quantized):
                for base in _NVFP4_QUANTIZED_EXPERT_TENSORS:
                    for side in _NVFP4_SIDECARS:
                        keys.add(pfx + base + side)
    return keys


def inkling_account_checkpoint(
    all_keys: Set[str],
    config: InklingTextConfig,
    exclude_modules: Set[str],
    quantized: bool = True,
) -> Dict[str, Set[str]]:
    """Classify every checkpoint key into consumed-text / deferred / unaccounted.

    ``unaccounted`` and ``missing`` must both be empty for the checkpoint to be
    fully accounted for.
    """
    expected = inkling_expected_text_keys(config, exclude_modules, quantized)
    consumed_text = all_keys & expected
    deferred = {k for k in all_keys if k.startswith(INKLING_DEFERRED_PREFIXES)}
    missing = expected - all_keys

    unaccounted = all_keys - consumed_text - deferred
    return {
        "consumed_text": consumed_text,
        "deferred": deferred,
        "unaccounted": unaccounted,
        "missing": missing,
    }


def inkling_nvfp4_expert_layers(
    config: InklingTextConfig, exclude_modules: Set[str], quantized: bool = True
) -> List[int]:
    """Layers whose routed experts are stored as NVFP4 (expected: 3..65)."""
    return [
        n
        for n in range(config.num_hidden_layers)
        if not config.is_dense_layer(n) and _experts_are_nvfp4(n, exclude_modules, quantized)
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
# q/k/v map to the standard separate HF names; the fused ``qkv_proj`` loader
# collects and fuses them. Same for gate_up_proj <- gate_proj + up_proj.
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

_EXPERT_RE = re.compile(r"layers\.(\d+)\.mlp\.experts\.(w13_weight|w2_weight)(\.\w+)?$")
_DENSE_W13_RE = re.compile(r"layers\.(\d+)\.mlp\.w13_dn\.weight$")
# ``model.mtp.layers.<depth>.<tail>``: the draft chain. The tail below
# ``transformer_block.`` is an ordinary decoder layer and takes the same renames.
_MTP_RE = re.compile(r"^model\.mtp\.layers\.(\d+)\.(.*)$")


def _split_interleaved_gate_up(t: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a gate/up-interleaved fused tensor into ``(gate, up)`` strided views
    along ``dim``: gate = even indices, up = odd indices.

    The checkpoint interleaves the two projections along the output dim, so a
    contiguous ``chunk(2)`` would pair the wrong channels in every SwiGLU. The
    result is a view rather than a copy because the fused-MoE / gate_up loaders
    call ``.contiguous()`` on the small per-rank shard themselves; de-interleaving
    eagerly instead OOM-killed the TP=4 load. It reorders whole output rows, so it
    holds for a packed NVFP4 weight and its per-block scale alike.
    """
    dim = dim % t.dim()
    if t.shape[dim] % 2 != 0:
        raise ValueError(f"cannot split odd gate/up dim {dim}: {tuple(t.shape)}")
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

    @property
    def _text_config(self) -> InklingTextConfig:
        """The text sub-config the mapped weights actually describe.

        ``ModelLoader.load`` initializes the mapper with the top-level
        ``InklingConfig``, which carries the decoder geometry under
        ``text_config`` and has no ``vocab_size`` of its own.
        """
        cfg = self.config.pretrained_config
        return getattr(cfg, "text_config", cfg)

    def preprocess_weights(self, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        new_weights: dict[str, torch.Tensor] = {}
        text_config = self._text_config
        unpadded_vocab = int(
            getattr(
                text_config,
                "unpadded_vocab_size",
                text_config.vocab_size,
            )
        )
        for name, tensor in weights.items():
            if name in _SIMPLE_RENAMES:
                if name == "unembed.weight" and tensor.shape[0] > unpadded_vocab:
                    # The checkpoint LM head is padded to vocab_size while the
                    # tower emits logits only over the unpadded vocab.
                    # embed_tokens keeps the full matrix so input ids stay valid.
                    tensor = tensor[:unpadded_vocab]
                new_weights[_SIMPLE_RENAMES[name]] = tensor
                continue

            expert_match = _EXPERT_RE.search(name)
            if expert_match is not None:
                self._map_expert(name, tensor, expert_match, new_weights)
                continue

            dense_match = _DENSE_W13_RE.search(name)
            if dense_match is not None:
                # Dense w13_dn is gate/up-interleaved along the output dim; split
                # into gate (even rows) / up (odd rows) as strided views.
                layer_idx = dense_match.group(1)
                gate, up = _split_interleaved_gate_up(tensor, dim=0)
                new_weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate
                new_weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up
                continue

            # shared_experts.shared_w13_weight loads raw (interleaved); the
            # interleave is undone by the strided split in
            # ``InklingSharedExperts.forward``.

            mtp_match = _MTP_RE.match(name)
            if mtp_match is not None:
                self._map_mtp(mtp_match, tensor, new_weights)
                continue

            m = re.match(r"layers\.(\d+)\.(.*)$", name)
            if m is not None:
                layer_idx, tail = m.group(1), m.group(2)
                trt_tail = _LAYER_RENAMES.get(tail, tail)
                new_weights[f"model.layers.{layer_idx}.{trt_tail}"] = tensor
                continue

            # Unknown key: keep as-is so any mismatch surfaces loudly at load.
            new_weights[name] = tensor
        return new_weights

    def _map_mtp(self, match, tensor, new_weights: dict) -> None:
        """Rename one draft-chain tensor into the module tree the block builds.

        Emitted relative to the draft model (``mtp_layers.<depth>....``) because
        that is what the generic loader walks. The block's own submodules --
        ``embed_norm``, ``hidden_norm``, ``input_proj``, ``transformer_block`` --
        already carry the checkpoint's names, so only the decoder tail needs the
        same treatment the trunk gets: ``wq_du``/``wk_dv``/``wv_dv`` land as
        separate q/k/v that the loader fuses into ``qkv_proj``, and the
        gate/up-interleaved ``w13_dn`` is split first.

        Doing this here rather than in the model is the point: fusion, NVFP4
        scales and TP sharding are the loader's job, and a ``load_state_dict``
        that bypasses it can only fail (or, with strict off, quietly load
        nothing).
        """
        depth, tail = match.group(1), match.group(2)
        prefix = f"mtp_layers.{depth}"
        if tail.startswith("transformer_block."):
            inner = tail[len("transformer_block.") :]
            if inner == "mlp.w13_dn.weight":
                gate, up = _split_interleaved_gate_up(tensor, dim=0)
                new_weights[f"{prefix}.transformer_block.mlp.gate_proj.weight"] = gate
                new_weights[f"{prefix}.transformer_block.mlp.up_proj.weight"] = up
                return
            inner = _LAYER_RENAMES.get(inner, inner)
            new_weights[f"{prefix}.transformer_block.{inner}"] = tensor
            return
        new_weights[f"{prefix}.{tail}"] = tensor

    def _map_expert(
        self,
        name: str,
        tensor: torch.Tensor,
        match: re.Match[str],
        out: dict[str, torch.Tensor],
    ) -> None:
        """Unfuse a stacked expert tensor into per-expert fused-MoE keys.

        ``w13_weight[e]`` is gate/up-interleaved along the output dim, so ``w1``
        is the even rows and ``w3`` the odd ones (see
        :func:`_split_interleaved_gate_up`); ``w2_weight[e]`` is the down
        projection. NVFP4 sidecars map to the fused-MoE scale names, and
        ``.original_shape`` is dropped as layout metadata.
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
            # The checkpoint stores a raw amax, but the fused-MoE loader expects
            # ModelOpt's input_scale = amax / (E2M1_MAX * E4M3_MAX); without the
            # conversion the activation block scales underflow e4m3.
            tensor = tensor.to(torch.float32) / (_NVFP4_E2M1_MAX * _NVFP4_E4M3_MAX)

        n_experts = int(getattr(self._text_config, "n_routed_experts", tensor.shape[0]))
        projs = ("w1", "w3") if which == "w13_weight" else ("w2",)

        def _assign(e: int, vals: tuple[torch.Tensor, ...]) -> None:
            for proj, val in zip(projs, vals):
                out[f"{prefix}.{e}.{proj}.{scale_name}"] = val

        # Three sidecar shapes: per-expert multi-dim weight/block-scale, per-expert
        # scalar weight_scale_2 (shared by gate and up), and a single global
        # input_amax scalar broadcast to every expert/proj.
        if tensor.dim() >= 2 and tensor.shape[0] == n_experts:
            for e in range(n_experts):
                if which == "w13_weight":
                    # Reorders whole rows, so this holds for the packed fp4
                    # weight and its per-block fp8 scale alike.
                    per = _split_interleaved_gate_up(tensor[e], dim=0)
                else:
                    per = (tensor[e],)
                _assign(e, per)
        elif tensor.dim() >= 1 and tensor.shape[0] == n_experts:
            for e in range(n_experts):
                _assign(e, (tensor[e],) * len(projs))
        else:  # global scalar (input_amax [1]) -> broadcast to all experts
            val = tensor.reshape(-1)[0]
            for e in range(n_experts):
                _assign(e, (val,) * len(projs))
