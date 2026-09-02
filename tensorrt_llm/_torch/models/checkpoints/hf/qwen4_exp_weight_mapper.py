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
"""Text-core checkpoint weight mapper for Qwen3.8-Flash-Next.

Maps the composite Hugging Face checkpoint state dict onto the TensorRT-LLM
:class:`~tensorrt_llm._torch.models.modeling_qwen4_exp.Qwen4ExpForCausalLM` text
module tree. The checkpoint is the *composite multimodal*
checkpoint (arch ``Qwen4ExpForConditionalGeneration``), so every text tensor is
stored under a ``model.language_model.`` prefix; the vision tower lives under
``model.visual.`` and a 1-layer MTP head under ``mtp.``. This mapper is selected
for the flattened text arch ``Qwen4ExpForCausalLM`` (see
``config_utils.load_pretrained_config``).

What differs from the stock HF loader (why a bespoke mapper is required):

* **Namespace** — strip ``model.language_model.`` -> ``model.``; the text core is
  the model's ``model.*`` subtree with an untied top-level ``lm_head.weight``.
* **Optional modules** — ``model.visual.*`` is loaded by the composite wrapper.
  The checkpoint's single ``mtp.*`` layer is mapped onto the appended runtime
  decoder layer only when one-model MTP is enabled; otherwise it is skipped.
* **Gated-DeltaNet in-proj** — the checkpoint stores the linear-attention
  input projection as four *separate, head-first dense* tensors ``in_proj_qkv``
  ([q|k|v]=10240), ``in_proj_z`` (6144), ``in_proj_b``/``in_proj_a`` (48 each);
  the TRT-LLM ``Qwen3NextGatedDeltaNet`` mixer consumes a single fused
  ``in_proj_qkvz`` ([Q|K|V|Z]=16384) and ``in_proj_ba`` ([b|a]=96). Unlike the
  grouped-interleaved Qwen3Next checkpoint, Qwen4-Exp is already dense, so the
  fusion is a **plain concat** (``[qkv|z]`` / ``[b|a]``) with **no**
  grouped-to-dense permutation.
  For ``tp_size > 1`` each rank's contiguous column-parallel slice must carry its
  own q/k/v/z (resp. b/a) heads, so the fused rows are re-blocked per rank.
* **Gated-DeltaNet conv1d / A_log / dt_bias** — depthwise ``conv1d`` weight
  is squeezed + per-rank re-blocked ([q|k|v]); ``A_log`` / ``dt_bias`` are cast to
  fp32 (the SSM state dtype) and TP-split.
* **MoE** — the checkpoint stores fused, HF-transposed expert
  stacks ``experts.gate_up_proj`` ([E, 2*I, H]) / ``experts.down_proj``
  ([E, H, I]); they are transposed to TRT-LLM's ([E, H, 2*I] / [E, I, H]) and
  loaded through ``MoEWeightLoadingMode.FUSED_GATE_UP_PROJ``. The shared expert
  (``gate_proj``/``up_proj`` -> fused ``gate_up_proj``) and router ``gate`` use
  the stock fusion / direct-copy paths.
* **PLE n-gram embedding** — the n-gram table is stored as
  ``split_ngram_parts`` equal prime-tiled shards
  ``ple_embedding.ngram_embedding.shard_{i}.weight``; each rank streams only
  its overlapping row range into the local ``ngram_embedding`` shard, avoiding
  a full-table concat or replication. FP8 checkpoints additionally store the scalar
  ``ngram_embedding.weight_scale``; the table stays FP8 in device memory and
  only selected rows are dequantized after lookup. The three recurrent-hash
  metadata buffers (``layer_multipliers`` /
  ``ngram_heads_offsets`` / ``ngram_heads_vocab_sizes``) are copied into the
  module's registered buffers.
* **QSA full attention** — ``q_proj`` (output-gated, 2x q heads) / ``k_proj``
  / ``v_proj`` fuse to ``qkv_proj`` via the stock fusion; ``o_proj``, per-head
  ``q_norm``/``k_norm``, and the compressed indexer
  (``indexer.index_qk_proj``/``q_layernorm``/``k_layernorm``) map by direct name.
* **Hyper-Connection** — attention/MLP mixers pack the checkpoint's
  ``input_mix_weight_down`` and ``block_inject_weight`` into one 16-row-aligned
  runtime projection, eliminating a skinny GEMM launch. The final mix-only
  ``hyper_connection_mixer`` retains its direct checkpoint name. All HC weights
  are replicated across TP ranks.
"""

import re
from typing import Optional

import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.qwen2_moe_weight_mapper import Qwen2MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.qwen4_exp.ple import Qwen4ExpPinnedHostEmbedding
from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.weight_owner import is_moe_weight_owner
from tensorrt_llm._torch.utils import split

# Source namespace of the text core inside the composite multimodal checkpoint.
_LM_PREFIX = "model.language_model."
# The vision tower is loaded by the composite wrapper.
_SKIP_PREFIXES = ("model.visual.",)
_PER_EXPERT_PROJECTION_PATTERN = re.compile(r"^\d+\.(?:gate_proj|up_proj|down_proj)\.")


def _rank_block(components: list[torch.Tensor], tp_size: int) -> torch.Tensor:
    """Concat ``components`` (each ``[out_i, ...]``) into the per-rank column-
    parallel row order a contiguous ``TensorParallelMode.COLUMN`` split recovers.

    At ``tp_size == 1`` this is a plain ``[c0 | c1 | ...]`` concat. At
    ``tp_size > 1`` each component is split across ranks and the rank-``r`` slices
    are grouped together (``[c0_r0 c1_r0 ... | c0_r1 c1_r1 ... | ...]``), so a
    later contiguous per-rank row split hands each rank its own head slice of
    every component.
    """
    if not components:
        raise ValueError("rank blocking requires at least one projection")
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    trailing_shape = components[0].shape[1:]
    for component in components:
        if component.ndim == 0 or component.shape[1:] != trailing_shape:
            raise ValueError("rank-blocked projections must have matching trailing dimensions")
        if component.shape[0] % tp_size:
            raise ValueError(
                f"projection rows {component.shape[0]} are not divisible by tp_size={tp_size}"
            )
    if tp_size == 1:
        return torch.cat(components, dim=0).contiguous()
    rows: list[torch.Tensor] = []
    for rank in range(tp_size):
        rows.extend(split(c, tp_size, rank) for c in components)
    return torch.cat(rows, dim=0).contiguous()


def _normalize_moe_module_weights(
    module_weights: dict[str, torch.Tensor], config
) -> tuple[dict[str, torch.Tensor], MoEWeightLoadingMode]:
    """Normalize fused BF16 and ModelOpt per-expert MoE checkpoints."""
    has_per_expert_tensors = any(
        _PER_EXPERT_PROJECTION_PATTERN.match(name) for name in module_weights
    )
    updated: dict[str, torch.Tensor] = {}
    if has_per_expert_tensors:
        for name, value in module_weights.items():
            if name in ("gate_up_proj", "down_proj"):
                continue
            normalized_name = name.replace("gate_proj", "w1")
            normalized_name = normalized_name.replace("up_proj", "w3")
            normalized_name = normalized_name.replace("down_proj", "w2")
            updated[normalized_name] = value
        return updated, MoEWeightLoadingMode.VANILLA

    hidden = config.hidden_size
    inter = config.moe_intermediate_size
    num_experts = getattr(config, "num_experts", None)
    for name, value in module_weights.items():
        if name in ("gate_up_proj", "down_proj") and getattr(value, "ndim", None) != 3:
            raise ValueError(f"MoE {name} must be three-dimensional, got {tuple(value.shape)}")
        if name == "gate_up_proj":
            expected_source = (value.shape[0], 2 * inter, hidden)
            expected_runtime = (value.shape[0], hidden, 2 * inter)
            if tuple(value.shape) == expected_source:
                value = value.transpose(-1, -2).contiguous()
            elif tuple(value.shape) != expected_runtime:
                raise ValueError(
                    f"MoE gate_up_proj has shape {tuple(value.shape)}, expected "
                    f"{expected_source} or {expected_runtime}"
                )
        elif name == "down_proj":
            expected_source = (value.shape[0], hidden, inter)
            expected_runtime = (value.shape[0], inter, hidden)
            if tuple(value.shape) == expected_source:
                value = value.transpose(-1, -2).contiguous()
            elif tuple(value.shape) != expected_runtime:
                raise ValueError(
                    f"MoE down_proj has shape {tuple(value.shape)}, expected "
                    f"{expected_source} or {expected_runtime}"
                )
        if (
            num_experts is not None
            and name in ("gate_up_proj", "down_proj")
            and value.shape[0] != num_experts
        ):
            raise ValueError(f"MoE {name} has {value.shape[0]} experts, expected {num_experts}")
        updated[name] = value
    return updated, MoEWeightLoadingMode.FUSED_GATE_UP_PROJ


@register_mapper("HF", "Qwen4ExpForConditionalGeneration")
@register_mapper("HF", "Qwen4ExpForCausalLM")
class Qwen4ExpHfWeightMapper(Qwen2MoeHfWeightMapper):
    """Weight mapper for the Qwen4-Exp hybrid text core."""

    # Substring markers of the manually-streamed / skipped modules.
    _NGRAM_EMBED_MARKER = "ple_embedding.ngram_embedding"

    def should_skip_module(self, module_name: str) -> bool:
        # The recurrent layer is shared with ``model.layers`` and loaded through
        # that canonical path; skip its duplicate draft-model alias.
        if module_name.startswith("draft_model"):
            return True
        # The large, sharded n-gram table is streamed shard-by-shard in
        # ``preprocess_weights``; skip it in the generic per-module walk so the
        # driver does not assert on the (already-consumed) single-tensor key.
        if self._NGRAM_EMBED_MARKER in module_name:
            return True
        return super().should_skip_module(module_name)

    def handle_special_instance_module(
        self,
        module: nn.Module,
        module_name: str,
        module_weights: dict,
        allow_partial_loading: bool = False,
    ) -> None:
        """Load routed experts from fused or per-expert checkpoint tensors.

        The checkpoint stores HF-transposed fused stacks ``gate_up_proj``
        ([E, 2*I, H]) / ``down_proj`` ([E, H, I]); transpose to TRT-LLM's
        ([E, H, 2*I] / [E, I, H]) and load via ``FUSED_GATE_UP_PROJ``. Non-MoE
        special modules fall through to the base handler.
        """
        if is_moe_weight_owner(module):
            config = self.config.pretrained_config
            updated, loading_mode = _normalize_moe_module_weights(module_weights, config)
            module.weight_loading_mode = loading_mode
            module.load_weights(weights=[updated], allow_partial_loading=allow_partial_loading)
            return
        return super().handle_special_instance_module(
            module, module_name, module_weights, allow_partial_loading=allow_partial_loading
        )

    def preprocess_weights(self, weights: dict, allow_partial_loading: bool = False) -> dict:
        config = self.config.pretrained_config
        tp_size = 1 if self.config.mapping.enable_attention_dp else self.config.mapping.tp_size
        tp_rank = self.config.mapping.tp_rank
        spec_config = getattr(self.config, "spec_config", None)
        mtp_enabled = spec_config is not None and spec_config.spec_dec_mode.is_mtp_one_model()
        mtp_layer_offset = config.num_hidden_layers
        mtp_mapping = {
            "mtp.fc_embedding": "fc_embedding",
            "mtp.fc_hidden": "fc_hidden",
            "mtp.pre_fc_norm_embedding": "pre_fc_norm_embedding",
            "mtp.pre_fc_norm_hidden": "pre_fc_norm_hidden",
            "mtp.hyper_connection_mixer": "shared_head.hyper_connection_mixer",
        }
        owned_layer_ids = None
        mapping_has_pp = (
            self.config.mapping.has_pp()
            if hasattr(self.config.mapping, "has_pp")
            else getattr(self.config.mapping, "pp_size", 1) > 1
        )
        if mapping_has_pp:
            owned_layer_ids = set()
            layer_prefix = "model.layers."
            for module_name, module in self.model.named_modules():
                if not module_name.startswith(layer_prefix):
                    continue
                suffix = module_name[len(layer_prefix) :]
                if "." in suffix or not suffix.isdigit():
                    continue
                if not getattr(module, "_weights_removed", False):
                    owned_layer_ids.add(int(suffix))

        key_dim = config.linear_key_head_dim * config.linear_num_key_heads
        value_dim = config.linear_value_head_dim * config.linear_num_value_heads
        num_v_heads = config.linear_num_value_heads

        # --- pass 1: drop skips, strip the language_model namespace, and bucket
        #             the separate GDN in-proj tensors by layer for fusion. ---
        renamed: dict = {}
        # layer_prefix -> {"qkv"/"z"/"a"/"b": tensor}
        gdn_in_proj: dict = {}
        # layer_prefix -> {shard_i / buffer_name: tensor}
        ngram: dict = {}
        # HC module prefix -> {down/inject: tensor}. Mix-only final heads only
        # contain ``down`` and retain the checkpoint's direct parameter name.
        hc_down_inject: dict = {}

        for name, tensor in weights.items():
            if any(name.startswith(p) for p in _SKIP_PREFIXES):
                continue
            key = name
            if key.startswith("mtp."):
                if not mtp_enabled:
                    continue
                if key.startswith("mtp.layers."):
                    _, _, mtp_layer_idx, module_name = key.split(".", 3)
                    key = f"model.layers.{mtp_layer_offset + int(mtp_layer_idx)}.{module_name}"
                else:
                    for mtp_prefix, runtime_name in mtp_mapping.items():
                        if key.startswith(mtp_prefix):
                            suffix = key[len(mtp_prefix) :]
                            key = f"model.layers.{mtp_layer_offset}.{runtime_name}{suffix}"
                            break
            if key.startswith(_LM_PREFIX):
                key = "model." + key[len(_LM_PREFIX) :]

            if owned_layer_ids is not None and key.startswith("model.layers."):
                layer_index = int(key.split(".", 3)[2])
                if layer_index not in owned_layer_ids:
                    continue

            if ".linear_attn.in_proj_" in key and key.endswith(".weight"):
                prefix, proj = key.rsplit(".in_proj_", 1)
                proj = proj[: -len(".weight")]  # qkv | z | a | b
                gdn_in_proj.setdefault(prefix, {})[proj] = tensor
                continue
            if (
                self._NGRAM_EMBED_MARKER in key
                or ".ple.ple_embedding.layer_multipliers" in key
                or ".ple.ple_embedding.ngram_heads_offsets" in key
                or ".ple.ple_embedding.ngram_heads_vocab_sizes" in key
            ):
                # ``.ple_embedding.<buffer>`` and ``.ngram_embedding.shard_i``
                # both start under the ple_embedding prefix; bucket by the PLE
                # module prefix (``....ple``).
                ple_prefix = key.split(".ple_embedding.", 1)[0]
                leaf = key.split(".ple_embedding.", 1)[1]
                ngram.setdefault(ple_prefix, {})[leaf] = tensor
                continue
            if key.endswith(".input_mix_weight_down.weight"):
                prefix = key[: -len(".input_mix_weight_down.weight")]
                hc_down_inject.setdefault(prefix, {})["down"] = tensor
                continue
            if key.endswith(".block_inject_weight.weight"):
                prefix = key[: -len(".block_inject_weight.weight")]
                hc_down_inject.setdefault(prefix, {})["inject"] = tensor
                continue
            renamed[key] = tensor

        new_weights: dict = {}

        # --- pass 2: reshape the linear-attention (GDN) tensors. ---
        for key, tensor in renamed.items():
            if key.endswith(".linear_attn.A_log") or key.endswith(".linear_attn.dt_bias"):
                if tensor.ndim != 1 or tensor.shape[0] != num_v_heads:
                    raise ValueError(
                        f"{key} has shape {tuple(tensor.shape)}, expected {(num_v_heads,)}"
                    )
                if tensor.shape[0] % tp_size:
                    raise ValueError(
                        f"{key} rows {tensor.shape[0]} are not divisible by tp_size={tp_size}"
                    )
                new_weights[key] = split(tensor[:], tp_size, tp_rank).to(torch.float32)
            elif key.endswith(".linear_attn.conv1d.weight"):
                # [conv_dim, 1, k] depthwise conv -> [conv_dim, k], row-blocked
                # per rank as [q|k|v] so the column-parallel Linear split keeps
                # each rank's own channels.
                w = tensor[:]
                if w.dim() == 3:
                    if w.shape[1] != 1:
                        raise ValueError(
                            f"{key} must be depthwise with singleton group dim, "
                            f"got {tuple(w.shape)}"
                        )
                    w = w.squeeze(1)
                if w.ndim != 2 or w.shape[0] != 2 * key_dim + value_dim:
                    raise ValueError(
                        f"{key} has shape {tuple(w.shape)}, expected "
                        f"[{2 * key_dim + value_dim}, conv_kernel]"
                    )
                conv_kernel = getattr(config, "linear_conv_kernel_dim", None)
                if conv_kernel is not None and w.shape[1] != conv_kernel:
                    raise ValueError(
                        f"{key} convolution width is {w.shape[1]}, expected {conv_kernel}"
                    )
                conv_q, conv_k, conv_v = torch.split(w, [key_dim, key_dim, value_dim], dim=0)
                new_weights[key] = _rank_block([conv_q, conv_k, conv_v], tp_size)
            else:
                new_weights[key] = tensor

        # Fuse the separate GDN in-proj tensors -> dense [Q|K|V|Z] / [b|a].
        for prefix, parts in gdn_in_proj.items():
            missing = {"qkv", "z", "a", "b"} - parts.keys()
            if missing:
                raise ValueError(
                    f"Qwen4-Exp GDN layer {prefix} is missing in-proj projections {sorted(missing)}"
                )
            # Split the pre-fused in_proj_qkv [Q|K|V] into its Q/K/V sub-tensors
            # BEFORE rank-blocking. Passing the concatenated [Q|K|V] as one
            # component makes _rank_block cut it at contiguous (key+key+value)/tp
            # boundaries, which scrambles Q/K/V across ranks at tp>1 (rank r would
            # get [Q[..], K[..]] instead of [Q_r|K_r|V_r]) -> garbage GDN output.
            # Each of Q, K, V, and Z is sharded independently. Mirror the
            # convolution handling above.
            qkv_t = parts["qkv"][:]
            expected_qkv_shape = (2 * key_dim + value_dim, config.hidden_size)
            expected_z_shape = (value_dim, config.hidden_size)
            expected_ba_shape = (num_v_heads, config.hidden_size)
            if tuple(qkv_t.shape) != expected_qkv_shape:
                raise ValueError(
                    f"{prefix}.in_proj_qkv has shape {tuple(qkv_t.shape)}, "
                    f"expected {expected_qkv_shape}"
                )
            if tuple(parts["z"].shape) != expected_z_shape:
                raise ValueError(
                    f"{prefix}.in_proj_z has shape {tuple(parts['z'].shape)}, "
                    f"expected {expected_z_shape}"
                )
            for projection in ("a", "b"):
                if tuple(parts[projection].shape) != expected_ba_shape:
                    raise ValueError(
                        f"{prefix}.in_proj_{projection} has shape "
                        f"{tuple(parts[projection].shape)}, expected {expected_ba_shape}"
                    )
            q_t, k_t, v_t = torch.split(qkv_t, [key_dim, key_dim, value_dim], dim=0)
            qkvz = _rank_block([q_t, k_t, v_t, parts["z"][:]], tp_size)
            ba = _rank_block([parts["b"][:], parts["a"][:]], tp_size)
            expected_qkvz_rows = 2 * key_dim + 2 * value_dim
            if qkvz.shape[0] != expected_qkvz_rows:
                raise ValueError(
                    f"fused in_proj_qkvz for {prefix} has {qkvz.shape[0]} rows, "
                    f"expected {expected_qkvz_rows}"
                )
            expected_ba_rows = 2 * num_v_heads
            if ba.shape[0] != expected_ba_rows:
                raise ValueError(
                    f"fused in_proj_ba for {prefix} has {ba.shape[0]} rows, "
                    f"expected {expected_ba_rows}"
                )
            new_weights[f"{prefix}.in_proj_qkvz.weight"] = qkvz
            new_weights[f"{prefix}.in_proj_ba.weight"] = ba

        # Pack each layer's HC down and injection projections into a single
        # aligned GEMM. Final mix-only heads have no injection projection and
        # keep the original ``input_mix_weight_down`` parameter.
        for prefix, parts in hc_down_inject.items():
            down = parts.get("down")
            inject = parts.get("inject")
            try:
                target = self.model.get_submodule(prefix)
            except (AttributeError, KeyError) as error:
                raise ValueError(
                    f"No runtime Hyper-Connection module for checkpoint prefix {prefix!r}"
                ) from error
            packed_projection = getattr(target, "input_mix_weight_down_block_inject", None)
            direct_projection = getattr(target, "input_mix_weight_down", None)
            if packed_projection is None:
                if direct_projection is None:
                    raise ValueError(
                        f"Runtime Hyper-Connection {prefix} has no input-mix projection"
                    )
                if down is None or inject is not None:
                    raise ValueError(
                        f"Mix-only Hyper-Connection {prefix} requires only input_mix_weight_down"
                    )
                expected_shape = tuple(direct_projection.weight.shape)
                if tuple(down.shape) != expected_shape:
                    raise ValueError(
                        f"{prefix}.input_mix_weight_down has shape {tuple(down.shape)}, "
                        f"expected {expected_shape}"
                    )
                new_weights[f"{prefix}.input_mix_weight_down.weight"] = down
                continue
            if down is None or inject is None:
                raise ValueError(
                    f"Packed Hyper-Connection {prefix} requires both input-mix and injection "
                    "checkpoint projections"
                )
            expected_rows, expected_columns = packed_projection.weight.shape
            if down.ndim != 2 or inject.ndim != 2 or down.shape[1] != inject.shape[1]:
                raise ValueError(
                    f"Hyper-Connection {prefix} projections must be two-dimensional with "
                    "matching input widths"
                )
            if down.shape[1] != expected_columns or down.shape[0] + inject.shape[0] > expected_rows:
                raise ValueError(
                    f"Hyper-Connection {prefix} source shapes {tuple(down.shape)} and "
                    f"{tuple(inject.shape)} do not fit runtime shape "
                    f"{tuple(packed_projection.weight.shape)}"
                )
            components = [down, inject]
            padding = expected_rows - down.shape[0] - inject.shape[0]
            if padding:
                components.append(down.new_zeros((padding, down.shape[1])))
            packed = torch.cat(components, dim=0).contiguous()
            if tuple(packed.shape) != tuple(packed_projection.weight.shape):
                raise ValueError(
                    f"Packed Hyper-Connection {prefix} has shape {tuple(packed.shape)}, "
                    f"expected {tuple(packed_projection.weight.shape)}"
                )
            new_weights[f"{prefix}.input_mix_weight_down_block_inject.weight"] = packed

        # --- pass 3: stream the PLE n-gram table + copy its metadata buffers. ---
        if ngram:
            self._load_ngram_tables(ngram)

        return new_weights

    # ----- PLE n-gram embedding -------------------------------------------

    def _ngram_module_for_prefix(self, ple_prefix: str) -> Optional[nn.Module]:
        """Resolve the ``Qwen4ExpNGramEmbedding`` for a checkpoint PLE prefix.

        ``ple_prefix`` is e.g. ``model.layers.1.ple``; the target n-gram module
        is ``model.layers.<L>.ple.ple_embedding``.
        """
        target_name = ple_prefix + ".ple_embedding"
        for name, module in self.model.named_modules():
            if name == target_name:
                return module
        return None

    def _load_ngram_tables(self, ngram: dict) -> None:
        """Stream each PLE prefix's shards into its table + copy its buffers.

        Streaming shard-by-shard avoids materialising the full table a second
        time (a single ``torch.cat`` would double peak memory). Meta
        tensors (name/shape accounting) fall through the same slice ``copy_``,
        which is a shape-checked no-op there.
        """
        buffer_leaves = ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")
        for ple_prefix, leaves in ngram.items():
            module = self._ngram_module_for_prefix(ple_prefix)
            if module is None:
                raise ValueError(
                    f"No Qwen4ExpNGramEmbedding module for PLE prefix "
                    f"{ple_prefix!r}; cannot load its n-gram table"
                )
            if getattr(module, "_weights_removed", False):
                continue

            # Metadata buffers (recurrent-hash constants): load the checkpoint's
            # authoritative values into the module's registered buffers.
            for leaf in buffer_leaves:
                if leaf in leaves:
                    buf = getattr(module, leaf)
                    if tuple(leaves[leaf].shape) != tuple(buf.shape):
                        raise ValueError(
                            f"PLE metadata {ple_prefix}.{leaf} has shape "
                            f"{tuple(leaves[leaf].shape)}, expected {tuple(buf.shape)}"
                        )
                    buf.data.copy_(leaves[leaf][:].to(buf.dtype))

            # N-gram table shards: copy only the overlap with this rank's row
            # partition. This works for replicated, TP, and attention-DP tables
            # without materialising a global concatenation.
            vocab_start = int(getattr(module, "vocab_start_index", 0))
            vocab_end = int(getattr(module, "vocab_end_index", module.padded_vocab_size))
            shard_leaves = sorted(
                (leaf for leaf in leaves if leaf.startswith("ngram_embedding.shard_")),
                key=lambda s: int(s.split(".shard_")[1].split(".")[0]),
            )
            if not shard_leaves:
                raise ValueError(f"PLE n-gram table for {ple_prefix} has no table shards")
            shard_indices = [int(leaf.split(".shard_")[1].split(".")[0]) for leaf in shard_leaves]
            if shard_indices != list(range(len(shard_indices))):
                raise ValueError(
                    f"PLE n-gram shards for {ple_prefix} must be consecutively numbered from zero, "
                    f"got {shard_indices}"
                )
            pretrained_config = getattr(getattr(self, "_config", None), "pretrained_config", None)
            expected_shards = getattr(pretrained_config, "split_ngram_parts", None)
            if expected_shards is not None and len(shard_leaves) != expected_shards:
                raise ValueError(
                    f"PLE n-gram table for {ple_prefix} has {len(shard_leaves)} shards, "
                    f"expected split_ngram_parts={expected_shards}"
                )
            shard_dtypes = {leaves[leaf].dtype for leaf in shard_leaves}
            if len(shard_dtypes) > 1:
                raise ValueError(
                    f"PLE n-gram shards for {ple_prefix} have mixed dtypes: "
                    f"{sorted(map(str, shard_dtypes))}"
                )
            shard_dtype = next(iter(shard_dtypes))
            weight_scale = leaves.get("ngram_embedding.weight_scale")
            if weight_scale is not None:
                if weight_scale.numel() != 1:
                    raise ValueError(
                        f"PLE n-gram weight scale for {ple_prefix} must be scalar, "
                        f"got shape {tuple(weight_scale.shape)}"
                    )
                if shard_dtype != torch.float8_e4m3fn:
                    raise ValueError(
                        f"PLE n-gram weight scale for {ple_prefix} requires float8_e4m3fn "
                        f"shards, got {shard_dtype}"
                    )
                module.configure_fp8_weight_storage(
                    weight_scale,
                    shard_dtype,
                )
            elif shard_dtype == torch.float8_e4m3fn:
                raise ValueError(f"PLE FP8 n-gram shards for {ple_prefix} have no weight scale")

            if isinstance(module.ngram_embedding, Qwen4ExpPinnedHostEmbedding):
                table = module.ngram_embedding.materialize_pinned()
            else:
                table = module.ngram_embedding.weight
            for leaf in shard_leaves:
                shard = leaves[leaf]
                if shard.ndim != 2 or shard.shape[1] != table.shape[1] or shard.shape[0] <= 0:
                    raise ValueError(
                        f"PLE n-gram shard {leaf} has shape {tuple(shard.shape)}, expected "
                        f"[positive_rows, {table.shape[1]}]"
                    )
            if getattr(module, "host_offload", False) and table.dtype != shard_dtype:
                raise TypeError(
                    f"PLE host table for {ple_prefix} uses {table.dtype}, "
                    f"but checkpoint shards use {shard_dtype}"
                )
            table_ptr = table.data_ptr()
            row = 0
            copied_rows = 0
            for leaf in shard_leaves:
                shard = leaves[leaf]
                rows = shard.shape[0]
                shard_end = row + rows
                overlap_start = max(row, vocab_start)
                overlap_end = min(shard_end, vocab_end)
                if overlap_start < overlap_end:
                    source_start = overlap_start - row
                    target_start = overlap_start - vocab_start
                    overlap_rows = overlap_end - overlap_start
                    source = shard[source_start : source_start + overlap_rows]
                    target = table.data[target_start : target_start + overlap_rows]
                    target.copy_(source.to(table.dtype))
                    copied_rows += overlap_rows
                row += rows
            if shard_leaves and row != module.padded_vocab_size:
                raise ValueError(
                    f"PLE n-gram shards for {ple_prefix} tiled {row} rows, "
                    f"global table expects {module.padded_vocab_size}"
                )
            if shard_leaves and copied_rows != vocab_end - vocab_start:
                raise ValueError(
                    f"PLE n-gram shards for {ple_prefix} loaded {copied_rows} "
                    f"local rows, expected {vocab_end - vocab_start}"
                )
            local_rows = vocab_end - vocab_start
            if table.shape[0] > local_rows:
                table.data[local_rows:].zero_()
            if table.data_ptr() != table_ptr:
                raise RuntimeError("PLE n-gram table address changed while loading")
