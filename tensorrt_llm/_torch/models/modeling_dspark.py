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
import os
import re
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm.logger import logger

from ..distributed import AllReduceParams
from ..modules.linear import Linear
from ..modules.mhc.hyper_connection import HCHead
from ..modules.rms_norm import RMSNorm
from ..speculative.dspark_attention import (
    _rmsnorm,
    _rope_last_dims,
    _rope_last_dims_batched,
    dspark_attention_forward,
    dspark_attention_forward_batched,
    dspark_sparse_attn,
    get_dspark_topk_idxs,
    precompute_dspark_freqs_cis,
)
from ..speculative.dspark_draft import build_draft_input_ids, dspark_propose
from ..speculative.dspark_heads import DSparkConfidenceHead, build_markov_head
from ..utils import AuxStreamType
from .modeling_deepseekv4 import (
    _ATTN_PARAM_RENAME,
    _SHARED_EXPERT_RENAME,
    DeepseekV4DecoderLayer,
    DeepseekV4WeightLoader,
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
    import json
    import os

    index = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return None
    weight_map = json.load(open(index)).get("weight_map", {})
    stages = {int(m.group(1)) for k in weight_map if (m := _DSPARK_MTP_RE.match(k))}
    return (max(stages) + 1) if stages else None


def _rename_dspark_attn_subkey(rest: str) -> str:
    """Rename a draft attention subkey (``mtp.{s}.attn.<rest>``).

    The DSpark draft uses *dense* MLA (no compressor/indexer), so only the base
    MLA projections appear. ``attn_sink`` (per-head fp32 Parameter) and ``wo_a``
    (the ``o_a_proj`` nn.Parameter) are special-cased exactly as the main
    DeepSeek-V4 loader handles them.
    """
    if rest == "attn_sink":
        return "attn_sink"
    if rest == "wo_a.weight":
        return "o_a_proj"
    if rest == "wo_a.scale":
        return "o_a_proj.weight_scale_inv"
    head, sep, tail = rest.partition(".")
    new_head = _ATTN_PARAM_RENAME.get(head, head)
    if tail == "scale":
        tail = "weight_scale_inv"
    return f"{new_head}.{tail}" if sep else new_head


def _rename_dspark_ffn_subkey(rest: str, routed_scale: str) -> str:
    """Rename a draft MoE subkey (``mtp.{s}.ffn.<rest>``)."""
    if rest == "gate.bias":
        return "gate.e_score_correction_bias"
    if rest.startswith("experts.") and rest.endswith(".scale"):
        return f"{rest[: -len('.scale')]}.{routed_scale}"
    rest = rest.replace(".scale", ".weight_scale_inv")
    if rest.startswith("shared_experts."):
        parts = rest.split(".")
        if len(parts) >= 2 and parts[1] in _SHARED_EXPERT_RENAME:
            parts[1] = _SHARED_EXPERT_RENAME[parts[1]]
        rest = ".".join(parts)
    return rest


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
        return f"self_attn.{_rename_dspark_attn_subkey(rest[len('attn.') :])}"
    if rest.startswith("ffn."):
        return f"mlp.{_rename_dspark_ffn_subkey(rest[len('ffn.') :], routed_scale)}"
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
    routed_scale = "weight_scale_inv"
    for key, value in weights.items():
        if (
            key.startswith("mtp.")
            and ".ffn.experts." in key
            and key.endswith(".weight")
            and getattr(value, "ndim", 0) == 2
            and getattr(value, "dtype", None) in (torch.int8, torch.uint8)
        ):
            routed_scale = "weight_scale"
            break
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
        # MXFP4 routed-expert tensors are stored as int8 but the fused-MoE
        # loader requires the packed uint8 view (mirrors the V4 main loader).
        if (
            routed_scale == "weight_scale"
            and ".mlp.experts." in model_key
            and (model_key.endswith(".weight") or model_key.endswith(".weight_scale"))
            and getattr(v, "dtype", None) is not None
            and v.dtype != torch.uint8
        ):
            v = v.view(torch.uint8)
        out[model_key] = v
    return out


# Increment A (DSPARK_DEV.md §7.2): run the captured-context draft attention
# through the REAL fp8 MLA projection modules the draft already loads
# (``kv_a_proj_with_mqa`` fused down-proj, ``q_a_layernorm``/``q_b_proj``,
# ``kv_a_layernorm``, ``o_a_proj``/``o_b_proj``) instead of the bf16-dequant
# ``dspark_attention_forward``. DeepSeek-V4 MLA is the *absorbed* latent form
# (``qk_head_dim == kv_lora_rank + qk_rope_head_dim == 512``, no ``kv_b_proj``),
# so the loaded projections produce exactly the reference DSpark attention's
# per-head 512-dim latent query + MQA latent K/V. RoPE stays the reference
# interleaved ``apply_dspark_rotary`` (applied to the projection outputs); only
# the linear projections move from bf16 to the real fp8 kernels. The windowed
# ``dspark_sparse_attn`` + worker-owned KV window are reused unchanged.
#
# Enabled by ``TLLM_DSPARK_REAL_MLA=1`` (default off): until benchmarked this is
# opt-in so the validated bf16 path and its acceptance regression guard stay
# the default (DSPARK_DEV.md Risk #1/#2).
DSPARK_REAL_MLA_ENV = "TLLM_DSPARK_REAL_MLA"

# DSPARK_DEV.md §7.2 root-cause (this session): the checkpoint stores
# ``mtp.{s}.attn.wo_a`` as fp8_e4m3 + a UE8M0 128x128 block scale (verified), and
# the reference (`inference/model.py`, ``self.wo_a`` is a bf16 ColumnParallelLinear
# loaded from the fp8 ckpt) uses the DEQUANTIZED bf16 ``wo_a`` (== ``wo_a_fp8 *
# scale`` ~ absmean 0.065). The bf16 captured-context path historically skipped
# this dequant (``deq("wo_a", False)`` -> raw fp8-cast-to-bf16, ~993x too large);
# the C4 golden masked it because the reference golden ALSO force-loaded wo_a raw
# (`scale=None`). The real-MLA path already uses the correctly-dequantized loader
# ``o_a_proj`` (cos 1.0 vs ``wo_a_fp8 * scale``). Gate the dequant fix so the prior
# (buggy) bf16 acceptance baseline stays reproducible for A/B until re-benchmarked.
DSPARK_FIX_WO_A_ENV = "DSPARK_FIX_WO_A"


def _dspark_mla_attention(
    self_attn,
    x: torch.Tensor,
    main_x: torch.Tensor,
    start_pos: int,
    kv_cache: torch.Tensor,
    *,
    window_size: int,
    eps: float,
    softmax_scale: float,
    freqs_cis: torch.Tensor,
    persist: bool = False,
) -> torch.Tensor:
    """Captured-context DSpark draft attention via the real fp8 MLA projections.

    Drop-in replacement for :func:`dspark_attention_forward` that sources the Q/KV
    down-projection, per-head query, MQA latent K/V and grouped O projection from
    the loaded :class:`DeepseekV4Attention` (``self_attn``) fp8 modules rather than
    cached bf16 weights. Shapes and semantics match the reference dense
    (``compress_ratio == 0``) ``DSparkAttention.forward``:

      * ``kv_a_proj_with_mqa(x)`` → ``[q_lora_rank | kv_lora_rank + qk_rope]``;
      * query = ``q_b_proj(q_a_layernorm(...))`` → per-head 512-dim latent, weightless
        per-head RMS, interleaved RoPE on the rope dims;
      * block/context latent K/V = ``kv_a_layernorm(...)`` (shared across heads, MQA),
        interleaved RoPE; context K/V from ``main_x`` written into the rolling window;
      * windowed attention-sink softmax (``mqa.attn_sink``); inverse RoPE on the output;
      * grouped O projection via ``o_a_proj`` (bf16 ``[g, o_lora, d]``) einsum + the
        fp8 ``o_b_proj`` Linear.

    ``kv_cache`` is ``[b, window_size, head_dim]`` (head_dim == ``qk_head_dim``);
    updated in place when ``persist`` (worker-owned cross-step window), else cloned.
    """
    assert start_pos > 0, "DSpark draft attention runs at generation (start_pos > 0)"
    b, block, hidden = x.shape
    n_heads = self_attn.num_heads_tp
    head_dim = self_attn.qk_head_dim
    rd = self_attn.qk_rope_head_dim
    n_groups = self_attn.n_local_groups
    qlr = self_attn.q_lora_rank
    main_freqs = freqs_cis[start_pos : start_pos + 1]
    blk_freqs = freqs_cis[start_pos + 1 : start_pos + 1 + block]

    # Fused down-projection on the block tokens: [q_lora | kv_lora + rope].
    qr_blk = self_attn.kv_a_proj_with_mqa(x.reshape(b * block, hidden))
    # Query: low-rank + per-head (weightless) RMS in the query dtype + RoPE.
    q = self_attn.q_a_layernorm(qr_blk[..., :qlr])
    q = self_attn.q_b_proj(q).view(b, block, n_heads, head_dim)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    q = _rope_last_dims(q, rd, blk_freqs)
    # Block latent K/V (MQA, shared across heads).
    kv = self_attn.kv_a_layernorm(qr_blk[..., qlr:]).view(b, block, head_dim)
    kv = _rope_last_dims(kv, rd, blk_freqs)
    # Captured-context latent K/V from main_x.
    qr_main = self_attn.kv_a_proj_with_mqa(main_x.reshape(b, hidden))
    main_kv = self_attn.kv_a_layernorm(qr_main[..., qlr:]).view(b, 1, head_dim)
    main_kv = _rope_last_dims(main_kv, rd, main_freqs)

    # Rolling-window write + windowed attention with the per-head sink.
    cache = kv_cache if persist else kv_cache.clone()
    cache[:, start_pos % window_size] = main_kv.squeeze(1).to(cache.dtype)
    kv_full = torch.cat([cache, kv], dim=1)
    topk = get_dspark_topk_idxs(window_size, b, block, start_pos, device=x.device)
    o = dspark_sparse_attn(q, kv_full, self_attn.mqa.attn_sink, topk, softmax_scale)
    o = _rope_last_dims(o, rd, blk_freqs, inverse=True)

    # Grouped low-rank O projection through the real o_a_proj (bf16) + o_b_proj (fp8).
    o = o.reshape(b, block, n_groups, -1).to(self_attn.o_a_proj.dtype)
    o = torch.einsum("bsgd,grd->bsgr", o, self_attn.o_a_proj)
    out = self_attn.o_b_proj(o.flatten(2).reshape(b * block, -1).contiguous())
    return out.view(b, block, hidden)


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
        # is_separate_draft_engine=True so the inherited attention uses a
        # draft-local layer index; disable_post_moe_fusion mirrors DeepseekV4MTP.
        super().__init__(
            model_config,
            layer_idx,
            aux_stream_dict,
            is_separate_draft_engine=True,
            disable_post_moe_fusion=True,
        )
        config = model_config.pretrained_config
        self.stage_id = int(stage_id)
        self.num_stages = int(num_stages)
        # DSPARK_BLOCK_SIZE_OVERRIDE lets a bench run the draft at a block_size
        # OTHER than the checkpoint's trained dspark_block_size (5) — block_size is
        # a runtime count of appended noise tokens, NOT a weight dimension, so the
        # backbone/markov/confidence run at any width (positions beyond the trained
        # 5 are off-distribution). Must equal the worker's max_draft_len. Default:
        # the trained config value (no behavior change).
        self.block_size = int(
            os.environ.get("DSPARK_BLOCK_SIZE_OVERRIDE", getattr(config, "dspark_block_size", 5))
        )
        self.noise_token_id = int(getattr(config, "dspark_noise_token_id", config.vocab_size))
        self.markov_rank = int(getattr(config, "dspark_markov_rank", 0))
        self.hc_mult = config.hc_mult

        # Stage 0: capture projection of the concatenated target-layer hiddens.
        if self.stage_id == 0:
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
        if self.stage_id == self.num_stages - 1:
            self.norm = RMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
            )
            self.hc_head = HCHead(config.hc_mult, config.hidden_size)
            self.markov_head = build_markov_head(
                markov_head_type=getattr(config, "dspark_markov_head_type", "vanilla"),
                vocab_size=config.vocab_size,
                markov_rank=self.markov_rank,
                hidden_size=config.hidden_size,
            )
            self.confidence_head = DSparkConfidenceHead(
                hidden_size=config.hidden_size,
                markov_rank=self.markov_rank,
                with_markov=True,
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
            or getattr(config, "num_nextn_predict_layers", 1)
        )
        # DSPARK_BLOCK_SIZE_OVERRIDE lets a bench run the draft at a block_size
        # OTHER than the checkpoint's trained dspark_block_size (5) — block_size is
        # a runtime count of appended noise tokens, NOT a weight dimension, so the
        # backbone/markov/confidence run at any width (positions beyond the trained
        # 5 are off-distribution). Must equal the worker's max_draft_len. Default:
        # the trained config value (no behavior change).
        self.block_size = int(
            os.environ.get("DSPARK_BLOCK_SIZE_OVERRIDE", getattr(config, "dspark_block_size", 5))
        )
        self.noise_token_id = int(getattr(config, "dspark_noise_token_id", config.vocab_size))
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
        # Lazily-built, device-keyed plain-RoPE table (sized to the decode pos).
        self._freqs_cache: Dict = {}
        # Fixed-cap plain-RoPE table for the CUDA-graph-safe batched path: a single
        # static buffer GATHERED by the per-request ``start_pos`` tensor, so its
        # shape never depends on the runtime decode position. Sized once to cover
        # any decode position (``max_position_embeddings``) plus the block
        # lookahead. The eager scalar path keeps using ``_dspark_freqs`` (which
        # slices a table sized to ``start_pos`` — a host int — each call).
        self._freqs_cap = (
            int(getattr(config, "max_position_embeddings", 163840)) + self.block_size + 2
        )
        self._freqs_table_cache: Dict = {}
        # Increment A (DSPARK_DEV.md §7.2): when enabled, the captured-context
        # draft attention runs through the loaded fp8 MLA projection modules
        # (``_dspark_mla_attention``) instead of the bf16-dequant
        # ``dspark_attention_forward``. Opt-in (default off) until benchmarked.
        self.use_real_mla = os.environ.get(DSPARK_REAL_MLA_ENV, "0") == "1"
        if self.use_real_mla:
            logger.info("[DSpark] real-fp8-MLA draft attention enabled (Increment A)")
        # Dequant the bf16-path wo_a (it IS fp8+scale in the checkpoint); see
        # DSPARK_FIX_WO_A_ENV. Default off to keep the prior baseline reproducible.
        self.fix_wo_a = os.environ.get(DSPARK_FIX_WO_A_ENV, "0") == "1"
        if self.fix_wo_a:
            logger.info("[DSpark] bf16-path wo_a fp8 dequant fix enabled")

    def post_load_weights(self) -> None:
        """Run the one-shot post-load transforms for the draft's quant linears.

        The fp8 UE8M0 linears we invoke as modules (``main_proj``, shared experts,
        the heads) need ``resmooth_to_fp8_e8m0`` + ``transform_sf_into_required_layout``
        before the first forward, or the kernel reads raw scales and emits NaNs.
        ``Linear.transform_weights`` is idempotent; the routed-expert MoE packs
        itself in its own ``load_weights``.

        The bf16 captured-context attention (``use_real_mla`` off) does NOT use the
        MLA module's forward — it runs ``dspark_attention_forward`` on dequantized
        bf16 weights cached via :meth:`cache_attn_weights_from_checkpoint` — so the
        MLA projection linears are skipped here (they would otherwise be transformed
        into the deep_gemm layout we don't consume). When ``use_real_mla`` is on
        (Increment A), the draft attention runs the loaded fp8 projections directly,
        so those linears MUST be transformed too (else deep_gemm reads raw scales).
        """
        attn_linear_ids = set()
        if not self.use_real_mla:
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
                # wo_a IS fp8+scale in the checkpoint (verified); dequant when the
                # fix is enabled, else keep the historical raw cast for the baseline.
                wo_a=deq("wo_a", self.fix_wo_a),
                wo_b=deq("wo_b", True),
                attn_sink=src[f"{pref}attn_sink"].to(dev).float(),
            )

    def cache_attn_weights_from_checkpoint(self, ckpt_dir: str, weight_map: Dict[str, str]) -> None:
        """Populate ``_dspark_attn`` by reading the ``mtp.{s}.attn.*`` tensors from the
        checkpoint shards on disk, then dequantizing via :meth:`_cache_attn_weights`.

        TODO(step 3): source these from the loaded ``MLA`` modules instead, once the
        fused/interleaved fp8 scale layout is decoded, to drop the checkpoint I/O.
        """
        import os

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

    def _dspark_freqs(self, seqlen: int, device: torch.device) -> torch.Tensor:
        """Cached plain-RoPE complex table sized to ``seqlen`` for ``device``."""
        key = (int(seqlen), str(device))
        cached = self._freqs_cache.get(key)
        if cached is None:
            cached = precompute_dspark_freqs_cis(
                self._attn_params["rope_head_dim"],
                int(seqlen),
                rope_theta=self._rope_theta,
                device=device,
            )
            self._freqs_cache[key] = cached
        return cached

    def _dspark_freqs_table(self, device: torch.device) -> torch.Tensor:
        """Fixed-size plain-RoPE table (cached per device) for the batched path.

        Unlike :meth:`_dspark_freqs` (sized to the host-int ``start_pos`` each
        call), this is built once to ``self._freqs_cap`` and GATHERED by the
        per-request ``start_pos`` tensor, so the consuming op's shape is constant
        — a prerequisite for CUDA-graph capture.
        """
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
        if not self.use_real_mla and getattr(self.mtp_layers[0], "_dspark_attn", None) is None:
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
        freqs = self._dspark_freqs(int(positions.max().item()) + 1, main_x.device)[positions]
        slots = positions % win  # [M]
        mx = main_x.unsqueeze(0)  # [1, M, hidden] for the per-position RoPE layout
        for s, stage in enumerate(self.mtp_layers):
            if self.use_real_mla:
                # Real fp8 MLA: latent K/V = kv_a_layernorm(kv_a_proj_with_mqa(main_x)[q_lora:]),
                # matching ``_dspark_mla_attention``'s context-write (same math as the
                # bf16 wkv/kv_norm path, via the loaded fp8 projection).
                qlr = stage.self_attn.q_lora_rank
                kv = stage.self_attn.kv_a_layernorm(
                    stage.self_attn.kv_a_proj_with_mqa(mx)[..., qlr:]
                )  # [1, M, head_dim]
            else:
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
        if not self.use_real_mla and getattr(self.mtp_layers[0], "_dspark_attn", None) is None:
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
            if self.use_real_mla:
                qlr = stage.self_attn.q_lora_rank
                kv = stage.self_attn.kv_a_layernorm(
                    stage.self_attn.kv_a_proj_with_mqa(main_x)[..., qlr:]
                )  # [G, M, head_dim]
            else:
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
            # written/read through ``slots``). Real-fp8-MLA (Increment A) is not
            # supported under graphs — guarded in ``forward_batched``.
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
        elif self.use_real_mla:
            attn = _dspark_mla_attention(
                stage.self_attn,
                layer_input,
                main_x,
                start_pos,
                kv_cache,
                window_size=self._attn_params["window_size"],
                eps=self._attn_params["eps"],
                softmax_scale=self._attn_params["softmax_scale"],
                freqs_cis=freqs_cis,
                persist=persist,
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
        h = stage.hc_attn.post_mapping(
            x=attn, residual=residual, post_layer_mix=post_mix, comb_res_mix=comb_mix
        )

        # --- MoE sub-block ---
        residual = h
        post_mix, comb_mix, layer_input = stage.hc_ffn.pre_mapping(residual)
        layer_input = stage.post_attention_layernorm(layer_input)
        num_tokens = T * block
        moe_out = stage.mlp(
            layer_input.reshape(num_tokens, hidden),
            input_ids=moe_input_ids,
            all_rank_num_tokens=[num_tokens],
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
        if not self.use_real_mla and getattr(self.mtp_layers[0], "_dspark_attn", None) is None:
            raise RuntimeError(
                "DSpark attention weights not cached; call "
                "cache_attn_weights_from_checkpoint(ckpt_dir, weight_map) after loading."
            )
        x, main_x, draft_ids = self.forward_embed(main_hidden, bonus_token_ids)
        main_x = main_x.unsqueeze(1)  # [T, 1, hidden] for the MQA K/V projection
        freqs_cis = self._dspark_freqs(start_pos + 1 + self.block_size, x.device)
        moe_input_ids = draft_ids.reshape(-1)

        h = x
        for s, stage in enumerate(self.mtp_layers):
            stage_window = kv_windows[:, s] if kv_windows is not None else None
            h = self._forward_stage(
                stage, h, main_x, start_pos, freqs_cis, moe_input_ids, stage_window
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
        if self.use_real_mla:
            raise NotImplementedError(
                "DSpark real-fp8-MLA draft attention (TLLM_DSPARK_REAL_MLA) is not "
                "supported on the CUDA-graph-safe batched draft path; the worker uses "
                "the eager per-request loop when real-fp8-MLA is enabled."
            )
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
                stage, h, main_x, start_pos, freqs_cis, moe_input_ids, stage_window, slots
            )

        return self.forward_head(
            h,
            bonus_token_ids,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
            return_logits=return_logits,
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

    def __init__(self, draft_config, aux_stream_dict=None, num_stages=None):
        super().__init__()
        self.dspark_model = DSparkDraftModel(draft_config, aux_stream_dict, num_stages=num_stages)
        # Generic handles expected by the loader / weight mappers.
        self.model = self.dspark_model
        self.model_config = draft_config
        self.config = draft_config.pretrained_config
        # Worker-facing interface (the worker receives this wrapper as
        # ``draft_model`` and calls forward()/reads these scalars).
        self.block_size = self.dspark_model.block_size
        self.num_stages = self.dspark_model.num_stages
        self._attn_params = self.dspark_model._attn_params
        # The batched, CUDA-graph-safe gen path does not support real-fp8-MLA
        # (Increment A); the worker falls back to the eager loop when this is set.
        self.use_real_mla = self.dspark_model.use_real_mla
        self.lm_head = None  # shared from the target (load_weights_from_target_model)
        self.logits_processor = None  # set by the caller after construction

    @property
    def embed_tokens(self):
        return self.dspark_model.embed_tokens

    def forward(self, main_hidden, bonus_token_ids, start_pos, **kwargs):
        return self.dspark_model.forward(main_hidden, bonus_token_ids, start_pos, **kwargs)

    def forward_batched(self, main_hidden, bonus_token_ids, start_pos, **kwargs):
        """CUDA-graph-safe batched draft forward (delegates to the draft model)."""
        return self.dspark_model.forward_batched(main_hidden, bonus_token_ids, start_pos, **kwargs)

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
        # tensors for ``dspark_attention_forward``. Skipped under real-fp8-MLA
        # (Increment A), which runs the loaded fp8 projections directly.
        if not self.dspark_model.use_real_mla:
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
