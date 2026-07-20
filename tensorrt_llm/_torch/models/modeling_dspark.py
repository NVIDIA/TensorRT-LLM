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
#
# DSpark backbone ported from the DeepSeek-V4-Pro-DSpark reference
# (`inference/model.py`: DSparkBlock / Transformer.forward_spec).
"""DeepSeek-V4-Pro DSpark speculative-decoding draft backbone.

The DSpark draft is ``n_mtp_layers`` (3 for V4-Pro) **full DeepSeek-V4 blocks**
stored under the ``mtp.*`` checkpoint namespace — it reuses the V4 decoder block
(MLA attention + MoE + manifold Hyper-Connections) and adds:

  - **stage 0**: ``main_proj`` (Linear, fp8) + ``main_norm`` (RMSNorm) — projects
    the concatenation of captured target-layer hidden states ([58,59,60]) into the
    draft's cross-attention context (``main_x``); replaces vanilla-MTP's
    enorm/hnorm + e_proj/h_proj single-hidden mixing.
  - **last stage**: ``norm`` + ``markov_head`` + ``confidence_head`` +
    flat ``hc_head`` — the block-draft output head (see dspark_heads/dspark_draft).

The per-stage *backbone* forward (block attention whose K/V derive from ``main_x``,
+ MoE + mHC) is brought up and numerically validated against the real fp8 weights
separately; ``forward_embed`` (capture) and ``forward_head`` (block draft) below
are the reference-faithful, unit-validated I/O stages.
"""

import copy
import json
import os
import re
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm.logger import logger

from ..distributed import AllReduceParams
from ..modules.linear import Linear
from ..modules.mhc.hyper_connection import HCHead
from ..modules.rms_norm import RMSNorm
from ..utils import AuxStreamType
from .dspark.attention import (
    _rmsnorm,
    _rope_last_dims,
    _rope_last_dims_batched,
    dspark_attention_forward,
    dspark_attention_forward_batched,
    precompute_dspark_freqs_cis,
)
from .dspark.draft import build_draft_input_ids, dspark_propose
from .dspark.heads import DSparkConfidenceHead, build_markov_head
from .modeling_deepseekv4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4WeightLoader,
    _get_deepseek_v4_routed_moe_scale_name,
    _maybe_view_deepseek_v4_routed_moe_tensor,
    _rename_deepseek_v4_attn_subkey,
    _rename_deepseek_v4_ffn_subkey,
)

# Matches the draft namespace ``mtp.<stage>.<rest>`` in the V4-Pro-DSpark
# checkpoint. Each draft stage is a full DeepSeek-V4 block stored under this
# prefix; the main model's keys (``layers.*``, ``embed.weight``, ``head.weight``,
# top-level ``norm.weight`` / ``hc_head_*``) are loaded by the target model.
_DSPARK_MTP_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")


def count_dspark_stages(ckpt_dir: str) -> Optional[int]:
    """Count the DSpark draft stages (``mtp.{s}.*``) in a checkpoint index.

    The HF ``config.json`` does not expose ``n_mtp_layers`` (only the reference
    ``inference/config.json`` does), so the authoritative draft stage count is
    the number of distinct ``mtp.<stage>`` prefixes in the weight index. Returns
    ``None`` if the index is missing or has no ``mtp.*`` keys (caller falls back
    to the config-derived default).
    """
    index = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return None
    with open(index, encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})
    stages = {int(m.group(1)) for k in weight_map if (m := _DSPARK_MTP_RE.match(k))}
    return (max(stages) + 1) if stages else None


def _rename_dspark_stage_subkey(rest: str, routed_scale: str) -> str:
    """Map a per-stage checkpoint subkey to the ``DSparkBlock`` param subkey."""
    if rest == "attn_norm.weight":
        return "input_layernorm.weight"
    if rest == "ffn_norm.weight":
        return "post_attention_layernorm.weight"
    # Flat manifold-Hyper-Connections / draft-head weights are loaded via
    # ``load_flat_hc_weights`` (keyed by the parent module stem), so pass the
    # flat-underscore form through unchanged:
    #   hc_attn_* / hc_ffn_*  -> mHC on every block
    #   hc_head_*             -> HCHead on the last stage
    if rest.startswith(("hc_attn_", "hc_ffn_", "hc_head_")):
        return rest
    # DSpark capture projection (stage 0): fp8 Linear .scale -> .weight_scale_inv.
    if rest == "main_proj.scale":
        return "main_proj.weight_scale_inv"
    if rest.startswith("attn."):
        return f"self_attn.{_rename_deepseek_v4_attn_subkey(rest[len('attn.') :])}"
    if rest.startswith("ffn."):
        return f"mlp.{_rename_deepseek_v4_ffn_subkey(rest[len('ffn.') :], routed_scale)}"
    # main_proj.weight, main_norm.weight, norm.weight, markov_head.*,
    # confidence_head.* map 1:1 onto the DSparkBlock submodules.
    return rest


def remap_dspark_draft_keys(weights: Dict, num_stages: int) -> Dict:
    """Convert checkpoint ``mtp.{s}.*`` keys to ``mtp_layers.{s}.*`` model keys.

    Only the draft namespace is consumed (stages ``[0, num_stages)``); shared
    ``embed_tokens`` / ``lm_head`` and other top-level keys belong to the target
    model and are skipped here. The routed-expert scale suffix mirrors the V4
    loader: ``weight_scale`` for the packed MXFP4 layout, else ``weight_scale_inv``.
    """
    routed_scale = _get_deepseek_v4_routed_moe_scale_name(weights, "mtp.")
    out: Dict[str, torch.Tensor] = {}
    for k, v in weights.items():
        m = _DSPARK_MTP_RE.match(k)
        if not m:
            continue
        stage = int(m.group(1))
        if stage >= num_stages:
            continue
        sub = _rename_dspark_stage_subkey(m.group(2), routed_scale)
        model_key = f"mtp_layers.{stage}.{sub}"
        v = _maybe_view_deepseek_v4_routed_moe_tensor(model_key, v, routed_scale)
        out[model_key] = v
    return out


# The checkpoint stores
# ``mtp.{s}.attn.wo_a`` as fp8_e4m3 + a UE8M0 128x128 block scale (verified), and
# the reference (`inference/model.py`, ``self.wo_a`` is a bf16 ColumnParallelLinear
# loaded from the fp8 ckpt) uses the DEQUANTIZED bf16 ``wo_a`` (== ``wo_a_fp8 *
# scale`` ~ absmean 0.065). The bf16 captured-context path historically skipped
# this dequant (raw fp8-cast-to-bf16, ~993x too large); the correct behavior is to
# dequantize ``wo_a`` (cos 1.0 vs ``wo_a_fp8 * scale``). Always dequantize now.


class DSparkBlock(DeepseekV4DecoderLayer):
    """One DSpark draft stage = a DeepSeek-V4 decoder block + DSpark extras.

    ``stage_id`` in ``[0, num_stages)``; only stage 0 owns the capture projection
    and only the last stage owns the draft heads, matching the ``mtp.*`` schema.
    """

    def __init__(
        self,
        model_config,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        *,
        stage_id: int,
        num_stages: int,
        num_capture_layers: int,
    ):
        # The inherited attention uses a draft-local layer index, while the
        # decoder layer keeps its model-level index for weights and captures.
        super().__init__(
            model_config,
            layer_idx,
            aux_stream_dict,
            attention_layer_idx=stage_id,
            disable_post_moe_fusion=True,
        )
        config = model_config.pretrained_config
        spec_cfg = getattr(model_config, "spec_config", None)
        self.stage_id = int(stage_id)
        self.num_stages = int(num_stages)
        # mask_token_id is a user override on the speculative_config; None means
        # fall back to the draft checkpoint's dspark_noise_token_id.
        mask_token_id = getattr(spec_cfg, "mask_token_id", None)
        self.noise_token_id = int(
            mask_token_id
            if mask_token_id is not None
            else getattr(config, "dspark_noise_token_id", config.vocab_size)
        )
        self.markov_rank = int(getattr(config, "dspark_markov_rank", 0))
        self.hc_mult = config.hc_mult
        # markov_head_type is a user override on the speculative_config; None
        # means fall back to the draft checkpoint's dspark_markov_head_type.
        markov_head_type = getattr(spec_cfg, "markov_head_type", None)
        if markov_head_type is None:
            markov_head_type = getattr(config, "dspark_markov_head_type", "vanilla")
        self.markov_head_type = markov_head_type

        # Stage 0: capture projection of the concatenated target-layer hiddens.
        if self.has_capture:
            self.main_proj = Linear(
                config.hidden_size * num_capture_layers,
                config.hidden_size,
                bias=False,
                dtype=config.torch_dtype,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )
            self.main_norm = RMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
            )

        # Last stage: the block-draft output heads + mHC head + final norm.
        if self.has_heads:
            self.norm = RMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
            )
            self.hc_head = HCHead(config.hc_mult, config.hidden_size)
            self.markov_head = build_markov_head(
                markov_head_type=self.markov_head_type,
                vocab_size=config.vocab_size,
                markov_rank=self.markov_rank,
                hidden_size=config.hidden_size,
            )
            self.confidence_head = DSparkConfidenceHead(
                hidden_size=config.hidden_size,
                markov_rank=self.markov_rank,
                # Only concat the Markov prev-token embedding when a Markov head
                # actually exists (build_markov_head returns None for
                # markov_rank <= 0); otherwise dspark_propose passes no
                # prev_embeddings and DSparkConfidenceHead.forward would assert.
                with_markov=self.markov_rank > 0,
            )

    @property
    def has_capture(self) -> bool:
        return self.stage_id == 0

    @property
    def has_heads(self) -> bool:
        return self.stage_id == self.num_stages - 1


class DSparkDraftModel(nn.Module):
    """The ``n_mtp_layers``-stage DSpark draft stacked on a DeepSeek-V4 target.

    Shares ``embed_tokens`` / ``lm_head`` with the target model. ``forward_embed``
    builds the block input from the captured context; the per-stage backbone runs
    the 3 blocks; ``forward_head`` produces the block draft tokens + confidence.
    """

    def __init__(
        self,
        model_config,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        num_stages: Optional[int] = None,
        block_size: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.config = config
        # The DSpark stage count is NOT the HF ``num_nextn_predict_layers`` (=1).
        # It is ``n_mtp_layers`` (3 for V4-Pro), which lives in the draft
        # sub-checkpoint config (inference/config.json) and is reflected by the
        # ``mtp.{0..n-1}.*`` weight namespace. Resolve it from (in priority):
        # an explicit override, the spec config's ``num_draft_layers``, a
        # pretrained-config ``n_mtp_layers``, else fall back to nextn.
        spec_cfg = getattr(model_config, "spec_config", None)
        self.num_stages = int(
            num_stages
            if num_stages is not None
            else getattr(spec_cfg, "num_draft_layers", None)
            or getattr(config, "n_mtp_layers", None)
            or config.num_nextn_predict_layers
        )
        # Production passes the validated speculative-config value explicitly;
        # direct construction falls back to the checkpoint's trained block size.
        self.block_size = int(
            block_size if block_size is not None else getattr(config, "dspark_block_size", 5)
        )
        # mask_token_id is a user override on the speculative_config; None means
        # fall back to the draft checkpoint's dspark_noise_token_id.
        mask_token_id = getattr(spec_cfg, "mask_token_id", None)
        self.noise_token_id = int(
            mask_token_id
            if mask_token_id is not None
            else getattr(config, "dspark_noise_token_id", config.vocab_size)
        )
        self.hc_mult = config.hc_mult
        target_layer_ids = getattr(config, "dspark_target_layer_ids", [])
        self.num_capture_layers = len(target_layer_ids)
        base = config.num_hidden_layers
        # Derive a draft-only model_config (a shallow copy so the shared config
        # and the target model are untouched) carrying two draft-specific fixes:
        #
        #  1. compress_ratios SLICE — the draft runs as a separate engine, so the
        #     inherited DeepSeek-V4 block remaps each block's layer_idx to a
        #     draft-local index in [0, num_stages) (the 1-layer-style draft KV
        #     cache). Sparse-attention compress_ratios / RoPE are indexed by that
        #     draft-local id, so they must be the draft slice
        #     (compress_ratios[base : base + num_stages]); otherwise indices
        #     0..n-1 resolve to the first *main* layers' sparse ratios — building
        #     a compressor the DSpark draft lacks and selecting YaRN over the
        #     dense path. For V4-Pro the draft slice is [1, 1, 1] (dense).
        #
        #  2. quant_config_dict EXTENSION — the checkpoint's per-module quant map
        #     only enumerates the base layers, so the draft layers' routed
        #     experts fall back to the global fp8 config and build fp8-shaped
        #     buffers. The draft experts are physically MXFP4 (same as the main
        #     MoE layers), so copy a main MoE layer's experts quant onto the
        #     draft layer keys.
        draft_model_config = self._derive_draft_model_config(model_config, base, self.num_stages)
        self.mtp_layers = nn.ModuleList(
            [
                DSparkBlock(
                    draft_model_config,
                    base + s,
                    aux_stream_dict,
                    stage_id=s,
                    num_stages=self.num_stages,
                    num_capture_layers=self.num_capture_layers,
                )
                for s in range(self.num_stages)
            ]
        )
        # Shared with target; wired by the spec wrapper after construction.
        self.embed_tokens: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None

        # Scalar attention params for the captured-context draft attention. These
        # are the dense (compress_ratio == 0) DSparkAttention constants — see the
        # reference ``inference/model.py`` ``Attention.__init__``. ``head_dim`` is
        # the MLA latent (MQA) dim; ``softmax_scale = head_dim ** -0.5``; the dense
        # draft disables YaRN and uses the base ``rope_theta``.
        self._attn_params = dict(
            n_heads=int(config.num_attention_heads),
            head_dim=int(
                getattr(config, "head_dim", config.kv_lora_rank + config.qk_rope_head_dim)
            ),
            rope_head_dim=int(config.qk_rope_head_dim),
            n_groups=int(config.o_groups),
            o_lora_rank=int(config.o_lora_rank),
            window_size=int(getattr(config, "window_size", 128)),
            eps=float(config.rms_norm_eps),
        )
        self._attn_params["softmax_scale"] = self._attn_params["head_dim"] ** -0.5
        self._rope_theta = float(getattr(config, "rope_theta", 10000.0))
        # Fixed-cap plain-RoPE table shared by the eager and CUDA-graph-safe
        # batched paths. It is built once per device and gathered/sliced by the
        # runtime decode positions, so the cache does not grow with sequence
        # length and the batched consuming op's shape remains static.
        self._freqs_cap = (
            int(getattr(config, "max_position_embeddings", 163840)) + self.block_size + 2
        )
        self._freqs_table_cache: Dict = {}

    def post_load_weights(self) -> None:
        """Run the one-shot post-load transforms for the draft's quant linears.

        The fp8 UE8M0 linears we invoke as modules (``main_proj``, shared experts,
        the heads) need ``resmooth_to_fp8_e8m0`` + ``transform_sf_into_required_layout``
        before the first forward, or the kernel reads raw scales and emits NaNs.
        ``Linear.transform_weights`` is idempotent; the routed-expert MoE packs
        itself in its own ``load_weights``.

        The bf16 captured-context attention does NOT use the MLA module's forward —
        it runs ``dspark_attention_forward`` on dequantized bf16 weights cached via
        :meth:`cache_attn_weights_from_checkpoint` — so the MLA projection linears are
        skipped here (they would otherwise be transformed into the deep_gemm layout we
        don't consume).
        """
        attn_linear_ids = set()
        for stage in self.mtp_layers:
            for m in stage.self_attn.modules():
                if isinstance(m, Linear):
                    attn_linear_ids.add(id(m))

        for module in self.modules():
            if isinstance(module, Linear) and id(module) not in attn_linear_ids:
                module.transform_weights()

    @staticmethod
    def _block_dequant(w_fp8: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
        """DeepSeek ``block``×``block`` block-scale dequant → bf16: ``real = fp8 * scale``.

        ``scale`` (possibly UE8M0) is broadcast over each ``block``×``block`` tile.
        Pure-torch (matches the golden-validated reference dequant), robust to the
        e8m0 scale dtype.
        """
        wf = w_fp8.float()
        out, inn = wf.shape
        s = scale.float()
        s_full = s.repeat_interleave(block, 0)[:out].repeat_interleave(block, 1)[:, :inn]
        return (wf * s_full).to(torch.bfloat16)

    def _cache_attn_weights(self, src: Dict) -> None:
        """Populate each stage's ``_dspark_attn`` from a dict of raw ``mtp.{s}.attn.*``
        tensors (source-agnostic core shared by the two public entry points).

        The captured-context attention runs the validated ``dspark_attention_forward``
        free function on reference-layout bf16 weights dequantized here. Sourcing the
        separate ``wq_a``/``wkv`` (plain 128×128 block scale) sidesteps the TRT-LLM
        ``MLA`` module's fused + interleaved fp8 storage (``kv_a_proj_with_mqa`` fuses
        ``q_a``+``kv`` and stores the scale interleaved). This mirrors the
        golden-validated dequant exactly.
        """
        for s, stage in enumerate(self.mtp_layers):
            pref = f"mtp.{s}.attn."
            dev = stage.input_layernorm.weight.device

            def deq(name: str, fp8: bool) -> torch.Tensor:
                w = src[f"{pref}{name}.weight"].to(dev)
                if fp8:
                    return self._block_dequant(w, src[f"{pref}{name}.scale"].to(dev))
                return w.to(torch.bfloat16)

            stage._dspark_attn = dict(
                wq_a=deq("wq_a", True),
                q_norm_w=src[f"{pref}q_norm.weight"].to(dev).to(torch.bfloat16),
                wq_b=deq("wq_b", True),
                wkv=deq("wkv", True),
                kv_norm_w=src[f"{pref}kv_norm.weight"].to(dev).to(torch.bfloat16),
                # wo_a IS fp8+scale in the checkpoint (verified); always dequant.
                wo_a=deq("wo_a", True),
                wo_b=deq("wo_b", True),
                attn_sink=src[f"{pref}attn_sink"].to(dev).float(),
            )

    def cache_attn_weights_from_checkpoint(self, ckpt_dir: str, weight_map: Dict[str, str]) -> None:
        """Populate ``_dspark_attn`` by reading the ``mtp.{s}.attn.*`` tensors from the
        checkpoint shards on disk, then dequantizing via :meth:`_cache_attn_weights`.

        TODO(step 3): source these from the loaded ``MLA`` modules instead, once the
        fused/interleaved fp8 scale layout is decoded, to drop the checkpoint I/O.
        """
        from safetensors import safe_open

        prefixes = tuple(f"mtp.{s}.attn." for s in range(len(self.mtp_layers)))
        shards: Dict[str, list] = {}
        for k in weight_map:
            if k.startswith(prefixes):
                shards.setdefault(weight_map[k], []).append(k)
        raw: Dict[str, torch.Tensor] = {}
        for shard, ks in shards.items():
            with safe_open(os.path.join(ckpt_dir, shard), framework="pt", device="cpu") as f:
                for k in ks:
                    raw[k] = f.get_tensor(k)
        self._cache_attn_weights(raw)

    def cache_attn_weights_from_state_dict(self, weights: Dict) -> None:
        """Populate ``_dspark_attn`` from an already-loaded in-memory ``weights`` dict
        (no extra disk I/O); used on the one-engine load path
        (``DSparkForCausalLM.load_weights``). Delegates to :meth:`_cache_attn_weights`.
        """
        self._cache_attn_weights(weights)

    def _dspark_freqs_table(self, device: torch.device) -> torch.Tensor:
        """Return the fixed-size plain-RoPE table cached for ``device``."""
        key = str(device)
        cached = self._freqs_table_cache.get(key)
        if cached is None:
            cached = precompute_dspark_freqs_cis(
                self._attn_params["rope_head_dim"],
                self._freqs_cap,
                rope_theta=self._rope_theta,
                device=device,
            )
            self._freqs_table_cache[key] = cached
        return cached

    @classmethod
    def _derive_draft_model_config(cls, model_config, base: int, num_stages: int):
        """Return a draft-only ``model_config`` copy with draft-specific fixes.

        Applies (1) the ``compress_ratios`` draft slice and (2) the
        ``quant_config_dict`` MXFP4 extension for the draft layers' routed
        experts. A single shallow copy is made (and only when something needs to
        change) so the shared ``model_config`` and the target model are untouched.

        The draft MoE backend is **inherited** from the target's
        ``model_config.moe_backend`` (carried by the shallow copy) — not pinned —
        matching every other drafter (the MTP module reuses the V4 decoder layer,
        whose MoE is built with ``moe_backend=model_config.moe_backend``; separate
        Eagle3/DFlash drafts resolve it from their own config the same way). The
        draft ``mtp.*`` stages are full V4 blocks, so they share the target's
        MXFP4 ``n_routed_experts=384`` / ``n_group=8`` (= 48 experts/group) layout
        and therefore the same backend constraints: pick a backend that supports
        it (CUTLASS today, DeepGEMM megaMoE once available) on the target and the
        draft follows. Note the TRTLLM-Gen ``blockScaleMoe`` routing kernel asserts
        ``experts/group <= 32`` (warp size), so it is incompatible with this layout
        for both the target and the draft.
        """
        new_sa = cls._draft_sparse_config(model_config, base, num_stages)
        new_qcd = cls._draft_quant_config_dict(model_config, base, num_stages)
        if new_sa is None and new_qcd is None:
            return model_config
        draft_cfg = copy.copy(model_config)
        # ModelConfig is a frozen dataclass; bypass the guard for these fields.
        if new_sa is not None:
            object.__setattr__(draft_cfg, "sparse_attention_config", new_sa)
        if new_qcd is not None:
            object.__setattr__(draft_cfg, "quant_config_dict", new_qcd)
        return draft_cfg

    @staticmethod
    def _draft_sparse_config(model_config, base: int, num_stages: int):
        """Sparse-attention config sliced to the draft layers, or None if N/A.

        The inherited block remaps ``layer_idx`` to a draft-local index, so the
        sparse config must expose the draft layers' per-layer ratios at indices
        ``[0, num_stages)``.
        """
        sa = getattr(model_config, "sparse_attention_config", None)
        compress_ratios = getattr(sa, "compress_ratios", None) if sa is not None else None
        if not compress_ratios or len(compress_ratios) < base + num_stages:
            return None
        draft_ratios = list(compress_ratios)[base : base + num_stages]
        # Already draft-local (e.g. a draft-only checkpoint config); no slice.
        if (
            draft_ratios == list(compress_ratios)[:num_stages]
            and len(compress_ratios) == num_stages
        ):
            return None
        return sa.model_copy(update={"compress_ratios": draft_ratios})

    @staticmethod
    def _draft_quant_config_dict(model_config, base: int, num_stages: int):
        """quant_config_dict extended to cover draft-layer experts, or None.

        The checkpoint's per-module quant map only enumerates the base layers, so
        ``model.layers.{base+s}.mlp.experts`` would fall back to the global fp8
        config and build fp8-shaped expert buffers. The draft routed experts are
        physically MXFP4 (identical to the main MoE layers), so copy a
        representative main MoE layer's experts quant onto the draft layer keys.
        """
        qcd = getattr(model_config, "quant_config_dict", None)
        if not qcd:
            return None
        src = next(
            (
                qcd[f"model.layers.{li}.mlp.experts"]
                for li in range(base)
                if f"model.layers.{li}.mlp.experts" in qcd
            ),
            None,
        )
        if src is None:
            return None
        new_qcd = dict(qcd)
        changed = False
        for s in range(num_stages):
            key = f"model.layers.{base + s}.mlp.experts"
            if new_qcd.get(key) is not src:
                new_qcd[key] = src
                changed = True
        return new_qcd if changed else None

    @torch.inference_mode()
    def write_context_windows(
        self,
        main_hidden: torch.Tensor,
        positions: torch.Tensor,
        stage_windows: torch.Tensor,
    ) -> None:
        """Write captured-context ``main_kv`` into the rolling per-stage KV windows.

        Replicates exactly the per-position context write that
        :func:`dspark_attention_forward` performs each generation step
        (``main_kv = RoPE_pos(rmsnorm(wkv @ main_x))`` written at
        ``window[pos % window_size]``), but for an arbitrary set of
        ``(captured-hidden, position)`` pairs. Used to (a) seed a request's
        window from its prompt at prefill and (b) back-fill the intermediate
        accepted tokens of a multi-accept step — both of which the per-step
        generation path would otherwise leave as holes, starving the draft
        attention of context (acceptance-rate only; verified decoding keeps
        output correctness regardless).

        Args:
            main_hidden: ``[M, num_capture * hidden]`` captured target hiddens.
            positions: ``[M]`` absolute window positions (used for BOTH the RoPE
                phase and the slot ``pos % window_size``). By the generation-path
                convention this is ``committed_position + 1``. Must hold at most
                ``window_size`` entries with distinct slots (the caller passes a
                contiguous, deduplicated range) so the scatter is well defined.
            stage_windows: ``[num_stages, window_size, head_dim]`` window for one
                request's slot; updated in place.
        """
        if getattr(self.mtp_layers[0], "_dspark_attn", None) is None:
            return
        M = int(main_hidden.shape[0])
        if M == 0:
            return
        win = int(self._attn_params["window_size"])
        rd = int(self._attn_params["rope_head_dim"])
        eps = float(self._attn_params["eps"])
        positions = positions.to(main_hidden.device).long()
        # main_x is stage-invariant (stage 0's projection), matching forward_embed.
        stage0 = self.mtp_layers[0]
        main_x = stage0.main_norm(stage0.main_proj(main_hidden))  # [M, hidden]
        freqs = self._dspark_freqs_table(main_x.device)[positions]
        slots = positions % win  # [M]
        mx = main_x.unsqueeze(0)  # [1, M, hidden] for the per-position RoPE layout
        for s, stage in enumerate(self.mtp_layers):
            a = stage._dspark_attn
            kv = _rmsnorm(F.linear(mx, a["wkv"]), a["kv_norm_w"], eps)  # [1, M, head_dim]
            kv = _rope_last_dims(kv, rd, freqs)  # [1, M, head_dim]
            stage_windows[s, slots] = kv[0].to(stage_windows.dtype)

    def write_context_windows_batched(
        self,
        main_hidden: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        mask: torch.Tensor,
        kv_windows: torch.Tensor,
    ) -> None:
        """CUDA-graph-safe batched + masked variant of :meth:`write_context_windows`.

        Back-fills the intermediate accepted tokens of a multi-accept step into the
        rolling per-stage KV windows for ALL gen requests at once, with a fixed
        ``[G, M]`` shape (``M = max interim per request``) and a validity mask
        (invalid entries are no-ops via a read-modify-write), so nothing depends on
        the per-request accept count. Same per-position math as the scalar version
        (``RoPE_pos(rmsnorm(wkv @ main_x))`` written at ``window[pos % window]``),
        but indexed/scattered through ``slots`` into the shared persistent buffer.

        Args:
            main_hidden: ``[G, M, num_capture * hidden]`` captured target hiddens
                (rows beyond a request's interim count are masked).
            positions: ``[G, M]`` absolute window positions (RoPE phase + slot).
            slots: ``[G]`` row index of each request into ``kv_windows``.
            mask: ``[G, M]`` bool — which ``(g, m)`` entries are real interim writes.
            kv_windows: ``[N, num_stages, window_size, head_dim]`` persistent buffer;
                updated in place.
        """
        if getattr(self.mtp_layers[0], "_dspark_attn", None) is None:
            return
        G, M = positions.shape
        if G == 0 or M == 0:
            return
        win = int(self._attn_params["window_size"])
        rd = int(self._attn_params["rope_head_dim"])
        eps = float(self._attn_params["eps"])
        positions = positions.long()
        slots = slots.long()
        freqs = self._dspark_freqs_table(main_hidden.device)[positions]  # [G, M, rd//2]
        cols = positions % win  # [G, M]
        rows = slots[:, None].expand(-1, M)  # [G, M]
        mask3 = mask.unsqueeze(-1)  # [G, M, 1]
        stage0 = self.mtp_layers[0]
        main_x = stage0.main_norm(stage0.main_proj(main_hidden))  # [G, M, hidden]
        for s, stage in enumerate(self.mtp_layers):
            a = stage._dspark_attn
            kv = _rmsnorm(F.linear(main_x, a["wkv"]), a["kv_norm_w"], eps)  # [G, M, head_dim]
            kv = _rope_last_dims_batched(kv, rd, freqs)  # [G, M, head_dim]
            win_s = kv_windows[:, s]  # [N, win, head_dim] view onto the base buffer
            # Read-modify-write so masked-out (g, m) entries keep their current
            # value — a graph-safe masked scatter (no dynamic-shape compaction).
            cur = win_s[rows, cols]  # [G, M, head_dim]
            win_s[rows, cols] = torch.where(mask3, kv.to(win_s.dtype), cur)

    def forward_embed(
        self, main_hidden: torch.Tensor, bonus_token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the draft block input and the cross-attention context.

        Args:
            main_hidden: ``[num_tokens, num_capture * hidden]`` captured context.
            bonus_token_ids: ``[num_tokens]`` last accepted token per request.
        Returns:
            x: hc-expanded block embeddings ``[num_tokens, block_size, hc_mult, hidden]``
            main_x: projected context ``[num_tokens, hidden]``
            draft_ids: block input token ids ``[num_tokens, block_size]`` (the callers
                reuse these as the per-position MoE routing ids, so we return them
                rather than rebuild).
        """
        stage0 = self.mtp_layers[0]
        main_x = stage0.main_norm(stage0.main_proj(main_hidden))
        draft_ids = build_draft_input_ids(
            bonus_token_ids, block_size=self.block_size, noise_token_id=self.noise_token_id
        )
        x = self.embed_tokens(draft_ids)
        x = x.unsqueeze(-2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x, draft_ids

    def _forward_stage(
        self,
        stage: "DSparkBlock",
        h: torch.Tensor,
        main_x: torch.Tensor,
        start_pos,
        freqs_cis: torch.Tensor,
        moe_input_ids: torch.Tensor,
        stage_window: Optional[torch.Tensor] = None,
        slots: Optional[torch.Tensor] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """One DSpark stage = reference ``Block.forward`` with captured-context attn.

        ``h`` is the mHC residual stream ``[T, block, hc_mult, hidden]``. The mHC
        ``pre_mapping``/``post_mapping`` preserve the leading ``[T, block]`` dims;
        the captured-context attention and MoE run on the collapsed token axis.
        Mirrors the reference (unfused) mHC boundaries exactly:
        ``hc_pre → attn_norm → DSparkAttention → hc_post`` then
        ``hc_pre → ffn_norm → MoE → hc_post``.

        ``stage_window`` is this stage's persistent rolling captured-context KV
        window ``[T, window_size, head_dim]`` owned by the worker across decode
        steps; the attention writes the current ``main_kv`` into it in place. When
        ``None`` (golden / single-shot) a zero window is allocated per call.

        When ``slots`` (a ``[G]`` int tensor) is given, the CUDA-graph-safe batched
        attention (:func:`dspark_attention_forward_batched`) is used instead: it
        takes ``start_pos`` as a ``[G]`` tensor and writes/reads ``stage_window``
        (then shaped ``[N, window_size, head_dim]``) through the ``slots`` index.
        """
        T, block, _, hidden = h.shape

        # --- attention sub-block (captured-context, not paged-KV MLA) ---
        residual = h
        post_mix, comb_mix, layer_input = stage.hc_attn.pre_mapping(residual)
        layer_input = stage.input_layernorm(layer_input)  # [T, block, hidden]
        # Rolling-window cache: persist through the worker-owned ``stage_window``
        # for cross-step decode, else a fresh zero window for a single block.
        persist = stage_window is not None
        kv_cache = (
            stage_window
            if persist
            else torch.zeros(
                T,
                self._attn_params["window_size"],
                self._attn_params["head_dim"],
                dtype=torch.bfloat16,
                device=h.device,
            )
        )
        if slots is not None:
            # Batched, CUDA-graph-safe path (start_pos is a [G] tensor; window is
            # written/read through ``slots``).
            attn = dspark_attention_forward_batched(
                layer_input,
                main_x,
                start_pos,
                kv_cache,
                slots,
                freqs_cis=freqs_cis,
                persist=True,
                **stage._dspark_attn,
                **self._attn_params,
            )
        else:
            attn = dspark_attention_forward(
                layer_input,
                main_x,
                start_pos,
                kv_cache,
                freqs_cis=freqs_cis,
                persist=persist,
                **stage._dspark_attn,
                **self._attn_params,
            )
        if stage.enable_fused_hc:
            residual, post_mix, comb_mix, layer_input = stage.hc_ffn.fused_hc(
                x_prev=attn,
                residual_prev=residual,
                post_mix_prev=post_mix,
                comb_mix_prev=comb_mix,
                norm_weight=stage.post_attention_layernorm.weight,
                norm_eps=stage.post_attention_layernorm.variance_epsilon,
            )
        else:
            residual = stage.hc_attn.post_mapping(
                x=attn,
                residual=residual,
                post_layer_mix=post_mix,
                comb_res_mix=comb_mix,
            )
            post_mix, comb_mix, layer_input = stage.hc_ffn.pre_mapping(residual)
            layer_input = stage.post_attention_layernorm(layer_input)
        num_tokens = T * block
        # FUSED_COMM MoE backends (DeepGEMM MegaMoE) size their in-kernel
        # NVLink-barrier chunk loop from ``max(all_rank_num_tokens)`` and index
        # the local slice by ``moe_ep_rank``, so every EP rank must pass the same
        # globally-gathered per-rank list (here: gen tokens = num_gens * block per
        # rank). Passing only the local ``[num_tokens]`` desyncs the phase-flip
        # barrier across ranks (hang / "unspecified launch failure"). Fall back to
        # the local count for single-rank / non-ADP runs where no list is threaded.
        moe_all_rank_num_tokens = (
            all_rank_num_tokens if all_rank_num_tokens is not None else [num_tokens]
        )
        moe_out = stage.mlp(
            layer_input.reshape(num_tokens, hidden),
            input_ids=moe_input_ids,
            all_rank_num_tokens=moe_all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(enable_allreduce=False),
            do_finalize=True,
        ).reshape(T, block, hidden)
        h = stage.hc_ffn.post_mapping(
            x=moe_out, residual=residual, post_layer_mix=post_mix, comb_res_mix=comb_mix
        )
        return h

    def forward(
        self,
        main_hidden: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        start_pos: int,
        *,
        kv_windows: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        return_logits: bool = False,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> tuple:
        """Full block-draft forward: chain the ``num_stages`` DSpark stages.

        Mirrors the reference ``Transformer.forward_spec`` (generation path,
        ``start_pos > 0``): ``forward_embed`` builds the block input + captured
        context, each stage runs the captured-context backbone, and
        ``forward_head`` emits the block draft tokens + per-position confidence.

        Args:
            main_hidden: ``[T, num_capture * hidden]`` captured target context.
            bonus_token_ids: ``[T]`` last accepted token per request.
            start_pos: absolute decode position (must be > 0).
            kv_windows: optional persistent per-stage rolling captured-context KV
                windows ``[T, num_stages, window_size, head_dim]`` owned by the
                worker; updated in place each call. ``None`` allocates fresh zero
                windows (single-shot golden / test path).
        Returns:
            ``(draft_tokens [T, block], num_proposed [T])`` from ``forward_head``.
        """
        assert start_pos > 0, "DSpark draft runs at generation (start_pos > 0)"
        if getattr(self.mtp_layers[0], "_dspark_attn", None) is None:
            raise RuntimeError(
                "DSpark attention weights not cached; call "
                "cache_attn_weights_from_checkpoint(ckpt_dir, weight_map) after loading."
            )
        x, main_x, draft_ids = self.forward_embed(main_hidden, bonus_token_ids)
        main_x = main_x.unsqueeze(1)  # [T, 1, hidden] for the MQA K/V projection
        freqs_cis = self._dspark_freqs_table(x.device)
        moe_input_ids = draft_ids.reshape(-1)

        h = x
        for s, stage in enumerate(self.mtp_layers):
            stage_window = kv_windows[:, s] if kv_windows is not None else None
            h = self._forward_stage(
                stage,
                h,
                main_x,
                start_pos,
                freqs_cis,
                moe_input_ids,
                stage_window,
                all_rank_num_tokens=all_rank_num_tokens,
            )

        return self.forward_head(
            h,
            bonus_token_ids,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            return_logits=return_logits,
        )

    def forward_batched(
        self,
        main_hidden: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        kv_windows: torch.Tensor,
        slots: torch.Tensor,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        return_logits: bool = False,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> tuple:
        """CUDA-graph-safe batched block-draft forward (all gen requests at once).

        Same computation as :meth:`forward`, but every host-int / data-dependent
        operation is tensorized so the whole path can be captured into the target's
        CUDA graph (DSpark is a one-engine drafter — its worker runs inside the
        graph). ``start_pos`` is a ``[G]`` tensor (one absolute decode position per
        gen request); the rolling captured-context windows are written/read through
        ``slots`` into the worker-owned ``kv_windows`` buffer; RoPE phases are
        gathered from a fixed table. ``forward_head`` is run with
        ``confidence_threshold == 0`` (the worker proposes the full block), which is
        the graph-safe branch of :func:`dspark_propose`.

        Args:
            main_hidden: ``[G, num_capture * hidden]`` captured target context.
            bonus_token_ids: ``[G]`` last accepted token per gen request.
            start_pos: ``[G]`` int tensor of absolute decode positions (> 0).
            kv_windows: ``[N, num_stages, window_size, head_dim]`` persistent rolling
                windows; written in place through ``slots``.
            slots: ``[G]`` int tensor mapping each request to its ``kv_windows`` row.
        Returns:
            ``(draft_tokens [G, block], num_proposed [G])`` from ``forward_head``.
        """
        if getattr(self.mtp_layers[0], "_dspark_attn", None) is None:
            raise RuntimeError(
                "DSpark attention weights not cached; call "
                "cache_attn_weights_from_checkpoint(ckpt_dir, weight_map) after loading."
            )
        x, main_x, draft_ids = self.forward_embed(main_hidden, bonus_token_ids)
        main_x = main_x.unsqueeze(1)  # [G, 1, hidden] for the MQA K/V projection
        freqs_cis = self._dspark_freqs_table(x.device)
        moe_input_ids = draft_ids.reshape(-1)

        h = x
        for s, stage in enumerate(self.mtp_layers):
            stage_window = kv_windows[:, s]  # [N, window_size, head_dim]
            h = self._forward_stage(
                stage,
                h,
                main_x,
                start_pos,
                freqs_cis,
                moe_input_ids,
                stage_window,
                slots,
                all_rank_num_tokens=all_rank_num_tokens,
            )

        return self.forward_head(
            h,
            bonus_token_ids,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            return_logits=return_logits,
        )

    def run_moe_lockstep_noop(
        self, all_rank_num_tokens: Optional[List[int]], device: torch.device
    ) -> None:
        """Cross the FUSED_COMM MoE NVLink barrier the same number of times as
        gen-bearing ranks, for an EP rank whose local draft batch is empty.

        DeepGEMM MegaMoE (``scheduler_kind == FUSED_COMM``) synchronizes EP ranks
        with an in-kernel phase-flip NVLink barrier that flips on every kernel
        call, so every rank must invoke the MoE the same number of times or the
        barrier desyncs (hang / "unspecified launch failure"). In the DSpark
        draft only the MoE carries a cross-rank barrier (the captured-context
        attention and the markov/confidence heads are per-rank), so a rank with
        zero local generation requests replays just the per-stage MoE call with a
        single 1-row dummy (its entry in ``all_rank_num_tokens`` is ``1``). The
        scheduler runs its ``max``-derived chunk count, slicing this rank to the
        1 dummy row and zero-padding the remaining chunks, keeping the barrier
        lockstep. No-op when there is no cross-rank work (single-rank / non-ADP,
        or every rank is empty).
        """
        if all_rank_num_tokens is None or max(all_rank_num_tokens) == 0:
            return
        hidden = self.config.hidden_size
        # Use a 1-row dummy, NOT a 0-row tensor: DeepseekV4MoE's router /
        # shared-expert dense GEMMs reject a 0-row input (cuBLAS
        # CUBLAS_STATUS_INVALID_VALUE). The paired ``all_rank_num_tokens`` encodes
        # 1 for this rank, so the FUSED_COMM scheduler slices to this 1 dummy row
        # and still launches ``num_chunks`` cross-rank barrier crossings in
        # lockstep with the gen-bearing ranks.
        dummy_x = torch.zeros((1, hidden), dtype=torch.bfloat16, device=device)
        dummy_ids = torch.zeros((1,), dtype=torch.long, device=device)
        for stage in self.mtp_layers:
            stage.mlp(
                dummy_x,
                input_ids=dummy_ids,
                all_rank_num_tokens=all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(enable_allreduce=False),
                do_finalize=True,
            )

    def forward_head(
        self,
        block_hidden: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        *,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        return_logits: bool = False,
    ) -> tuple:
        """Block-draft head: hc_head + norm + lm_head -> markov refine + confidence.

        ``block_hidden`` is the last stage's mHC residual ``[*, block, hc_mult, hidden]``.
        Returns (draft_tokens [*, block], num_proposed [*]); with ``return_logits``
        also returns the per-position draft logits [*, block, vocab] (§7.9 1-TV).
        """
        last = self.mtp_layers[-1]
        h = last.hc_head(block_hidden)
        h = last.norm(h)
        base_logits = self.lm_head(h)
        return dspark_propose(
            base_logits,
            bonus_token_ids=bonus_token_ids,
            block_hidden=h,
            markov_head=last.markov_head,
            confidence_head=last.confidence_head,
            block_size=self.block_size,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            return_logits=return_logits,
        )


class DSparkForCausalLM(nn.Module):
    """One-engine draft wrapper for DSpark (mirrors ``DFlashForCausalLM``).

    Wraps :class:`DSparkDraftModel` (the ``n_mtp_layers``-stage ``mtp.*`` backbone)
    for the single-engine external-drafter flow: created by ``get_draft_model``,
    appended to the target's epilogue, and driven by ``DSparkWorker``.

    ``embed_tokens`` / ``lm_head`` are shared with the target model
    (:meth:`load_weights_from_target_model`). The draft weights live in the SAME
    checkpoint under ``mtp.*``; :meth:`load_weights` remaps them
    (``remap_dspark_draft_keys``), loads via ``DeepseekV4WeightLoader``, runs the
    fp8 ``post_load_weights`` transforms, and caches the bf16 captured-context
    attention weights from the in-memory state dict.
    """

    def __init__(self, draft_config, aux_stream_dict=None, num_stages=None, block_size=None):
        super().__init__()
        self.dspark_model = DSparkDraftModel(
            draft_config,
            aux_stream_dict,
            num_stages=num_stages,
            block_size=block_size,
        )
        # Generic handles expected by the loader / weight mappers.
        self.model = self.dspark_model
        self.model_config = draft_config
        self.config = draft_config.pretrained_config
        # Worker-facing interface (the worker receives this wrapper as
        # ``draft_model`` and calls forward()/reads these properties and scalars).
        self.num_stages = self.dspark_model.num_stages
        self._attn_params = self.dspark_model._attn_params
        self.lm_head = None  # shared from the target (load_weights_from_target_model)
        self.logits_processor = None  # set by the caller after construction

    @property
    def block_size(self):
        return self.dspark_model.block_size

    @property
    def embed_tokens(self):
        return self.dspark_model.embed_tokens

    def forward(self, main_hidden, bonus_token_ids, start_pos, **kwargs):
        return self.dspark_model.forward(main_hidden, bonus_token_ids, start_pos, **kwargs)

    def forward_batched(self, main_hidden, bonus_token_ids, start_pos, **kwargs):
        """CUDA-graph-safe batched draft forward (delegates to the draft model)."""
        return self.dspark_model.forward_batched(main_hidden, bonus_token_ids, start_pos, **kwargs)

    def run_moe_lockstep_noop(self, all_rank_num_tokens, device):
        """Empty-batch MoE barrier lockstep (delegates to the draft model)."""
        return self.dspark_model.run_moe_lockstep_noop(all_rank_num_tokens, device)

    def write_context_windows(self, main_hidden, positions, stage_windows):
        """Seed / back-fill the rolling KV windows (delegates to the draft model)."""
        return self.dspark_model.write_context_windows(main_hidden, positions, stage_windows)

    def write_context_windows_batched(self, main_hidden, positions, slots, mask, kv_windows):
        """Batched + masked window back-fill (delegates to the draft model)."""
        return self.dspark_model.write_context_windows_batched(
            main_hidden, positions, slots, mask, kv_windows
        )

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Load the ``mtp.*`` draft weights from the (full) checkpoint dict.

        ``weight_mapper`` is accepted for interface parity with the draft-weight
        loader but unused: DSpark does its own ``mtp.{s}.* -> mtp_layers.{s}.*``
        remap (``remap_dspark_draft_keys``) onto the shared V4 weight loader.
        """
        remapped = remap_dspark_draft_keys(weights, num_stages=self.num_stages)
        logger.info(
            f"[DSpark] loading {len(remapped)} draft params across {self.num_stages} stages"
        )
        DeepseekV4WeightLoader(self.dspark_model).load_weights(remapped)
        self.dspark_model.post_load_weights()
        # bf16 captured-context attention path: dequantize the raw mtp.{s}.attn.*
        # tensors for ``dspark_attention_forward``.
        self.dspark_model.cache_attn_weights_from_state_dict(weights)
        logger.info("[DSpark] draft weight load complete")

    def load_weights_from_target_model(self, target_model):
        """Share the target's embed_tokens / lm_head (DSpark has neither)."""
        if self.dspark_model.embed_tokens is None:
            self.dspark_model.embed_tokens = target_model.model.embed_tokens
        if self.lm_head is None:
            self.lm_head = target_model.lm_head
            self.dspark_model.lm_head = target_model.lm_head


__all__ = ["DSparkBlock", "DSparkDraftModel", "DSparkForCausalLM"]
