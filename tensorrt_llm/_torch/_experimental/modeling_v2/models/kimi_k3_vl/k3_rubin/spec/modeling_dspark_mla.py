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
# Port note carried over from tensorrt_llm/_torch/models/modeling_dspark.py, the file this drafter
# is sliced from (it applies to the parts copied from there, e.g. ``apply_dspark_rotary_batched``
# and ``_rmsnorm``):
#
# DSpark backbone ported from the DeepSeek-V4-Pro-DSpark reference
# (`inference/model.py`: DSparkBlock / Transformer.forward_spec).
#
# The captured-context attention primitives are ported from DeepSeek's DeepSpec
# reference ``inference/kernel.py`` (``sparse_attn``) and ``inference/model.py``
# (``get_dspark_topk_idxs``). The reference computes these with a TileLang
# kernel; this is a functional-first pure-PyTorch port with the same math
# (index-gather + online softmax + a learnable attention sink that contributes
# only to the softmax denominator).
#
# The draft I/O stages are ported from the same reference
# (`inference/model.py`: DSparkBlock.forward_embed / forward_head).
"""Kimi-K3 (Rubin) standalone MLA DSpark drafter, package-internal copy.

The MLA-backboned standalone slice of ``tensorrt_llm/_torch/models/modeling_dspark.py``
(``MLADSparkForCausalLM`` and its backbone), combined with TensorRT-LLM main and the K3 Rubin
drafter changes. Main's engine adaptations stay as main has them (``DraftBlockPlacement``, the
``_tp_fusions`` fc split and fused all-reduces, the cap-keyed YaRN table, the per-family backend
probe, the vocab-parallel Markov chain); the K3 drafter features are re-expressed on them behind the
K3 switches, default off:

* ``TLLM_DSPARK_DRAFT_FUSION``: q_a / kv_a as one GEMM, per-layer all-reduce + residual + RMSNorm
  over MNNVL, and the TP-sharded fc (replacing, never stacking on, main's fc split);
* ``TLLM_DSPARK_AR_PUSH``: the pushed all-reduce sites; ``TLLM_DSPARK_DRAFT_GEMM``: pinned GEMMs;
* ``TRTLLM_K3_DRAFT3_GLUE`` / ``TRTLLM_K3_DRAFT3_QHEAD``: side-stream glue and the folded query head;
* ``TLLM_SPEC_TAIL_FUSION`` / ``TLLM_DSPARK_TAIL_V2``: fused block stores and RoPE kernels;
* ``TRTLLM_K3_ULL2_DRAFT_A2A`` / ``_TAIL`` / ``_STORE``: the helix page-sharded latent cache;
* ``speculative_config.context_kv_only`` / ``context_kv_dtype``: disaggregated context servers and
  the FP8 latent cache.

No registry entries: main's registry keeps resolving its own drafter classes; the K3 shell builds
this drafter through :func:`build_k3_dspark_draft`.
"""

import contextlib
import copy
import inspect
import math
from dataclasses import replace
from functools import lru_cache
from typing import Callable, Dict, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM, resolve_dspark_head_config
from tensorrt_llm._torch.models.modeling_speculative import (
    SpecDecOneEngineForCausalLM,
    dspark_markov_chain,
    dspark_markov_step_bias,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModel
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.pyexecutor.config_utils import is_mla
from tensorrt_llm._torch.speculative.vocab_parallel import markov_chain_logits
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.functional import RopeEmbeddingUtils
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .dflash_attention import DRAFT_FP8_KV_SCALE, HelixDraftShard, to_draft_fp8
from .dspark_ar_sites import DraftArSites, draft_ar_push_enabled
from .dspark_draft3 import (
    draft3_glue_enabled,
    draft3_qhead_enabled,
    draft_qhead,
    fold_rmsnorm_weight,
)
from .dspark_draft_gemm import draft_gemm
from .dspark_ull2 import (
    draft_helix_a2a_enabled,
    draft_helix_store_enabled,
    draft_helix_tail,
    to_draft_fp8_1k,
)
from .spec_tail_kernels import (
    DraftBlockPlacement,
    draft_fusion_enabled,
    scatter_rows_to_pool,
    spec_tail_fusion_enabled,
    tail_v2_enabled,
)

dspark_mla_block_rope = None
dspark_mla_ctx_norm_rope = None
if IS_CUTLASS_DSL_AVAILABLE:
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        is_fused_dspark_rmsnorm_rope_supported,
    )

    # Main's DSpark RMSNorm + RoPE kernel compiled for row-strided input, and
    # TLLM_DSPARK_TAIL_V2's MLA RoPE kernels.
    from .dspark_mla_rope import (
        _compile_rmsnorm_rope_strided,
        dspark_mla_block_rope,
        dspark_mla_ctx_norm_rope,
        k3r_dspark_rmsnorm_rope,
    )

# ----------------------------------------------------------------------------
# Shared helpers of the block decode (the upper half of modeling_dspark.py).
# ----------------------------------------------------------------------------


def apply_dspark_rotary_batched(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Per-row (batched) DeepSpec ``apply_rotary_emb``.

    ``freqs_cis`` carries a leading batch axis so each row of ``x`` is rotated
    by its own per-request phases. ``x`` is the rope-dim slice only:
    ``[G, s, rd]`` (3D) or ``[G, s, h, rd]`` (4D), with ``freqs_cis`` of shape
    ``[G, s, rd // 2]``.
    """
    orig_dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    g, s, half = freqs_cis.shape
    if xc.ndim == 3:
        fc = freqs_cis.view(g, s, half)
    else:
        fc = freqs_cis.view(g, s, 1, half)
    out = torch.view_as_real(xc * fc).flatten(-2)
    return out.to(orig_dtype)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm matching the DeepSpec reference (fp32 reduce, then * weight)."""
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * xf).to(dtype)


def _rope_last_dims_batched(
    t: torch.Tensor, rope_head_dim: int, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply per-request RoPE to the last dimensions, preserving the non-RoPE prefix."""
    nope = t[..., :-rope_head_dim]
    rope = apply_dspark_rotary_batched(t[..., -rope_head_dim:], freqs_cis, inverse=inverse)
    return torch.cat([nope, rope], dim=-1)


def _plain(linear) -> bool:
    """A BF16 Linear without bias or quantization (its weight is the whole GEMM)."""
    return (
        getattr(linear, "bias", None) is None
        and not getattr(linear, "has_any_quant", True)
        and linear.weight.dtype == torch.bfloat16
    )


class _PendingDown(NamedTuple):
    """A drafter MLP whose down projection runs at the all-reduce site that consumes it."""

    h2: torch.Tensor
    mlp: nn.Module

    def partial(self, down_all_reduce_params) -> torch.Tensor:
        """The un-reduced down projection, as ``_draft_mlp`` computes it."""
        down = self.mlp.down_proj
        out = draft_gemm(self.h2, down.weight, "down")
        if out is None:
            out = down(
                self.h2, all_reduce_params=down_all_reduce_params, layer_idx=self.mlp.layer_idx
            )
        return out


def _draft_mlp(mlp, x: torch.Tensor, down_all_reduce_params, defer_down: bool = False):
    """``mlp(x, final_all_reduce_params=down_all_reduce_params)`` for the MLA
    drafter, with the gate_up / down GEMMs pinned by ``TLLM_DSPARK_DRAFT_GEMM``
    when it applies (same activation kernel in between). With ``defer_down`` a
    plain MLP returns a ``_PendingDown`` instead of running down_proj."""
    gate_up, down = getattr(mlp, "gate_up_proj", None), getattr(mlp, "down_proj", None)
    # Main's GatedMLP also has an opt-in BF16 gate_up + SwiGLU epilogue; where
    # its forward would take it, so does this one.
    fuse_bf16 = getattr(mlp, "_can_fuse_gate_up_swiglu_bf16", None)
    h1 = None
    plain = (
        gate_up is not None
        and down is not None
        and _plain(gate_up)
        and _plain(down)
        and not getattr(down_all_reduce_params, "enable_allreduce", True)
        and not mlp._can_fuse_gate_up_swiglu()
        and not mlp._can_fuse_gate_up_swiglu_fp4out()
        and not mlp._can_fuse_swiglu_fp8_quant()
        and not (fuse_bf16 is not None and fuse_bf16(x))
    )
    if plain:
        h1 = draft_gemm(x, gate_up.weight, "gate_up")
        if h1 is None and defer_down:
            h1 = gate_up(x)
    if h1 is None:
        return mlp(x, final_all_reduce_params=down_all_reduce_params)
    h2 = mlp._apply_activation(h1)
    pending = _PendingDown(h2, mlp)
    return pending if defer_down else pending.partial(down_all_reduce_params)


def _mla_rope_into(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    rope_head_dim: int,
    out_rope_offset: int,
) -> bool:
    """Rotate q's rope tail into ``out[..., out_rope_offset:]``; False if unavailable.

    Returns a bool rather than raising so the caller keeps the eager path for
    builds without the fused kernel, the same way _rmsnorm_rope_batched does.
    """
    if not (IS_CUTLASS_DSL_AVAILABLE and is_sm_100f()):
        return False
    freqs_real = torch.view_as_real(freqs_cis).reshape(-1, freqs_cis.shape[-1], 2)
    if not is_fused_dspark_rmsnorm_rope_supported(
        q, None, freqs_real, num_heads, rope_head_dim, q.shape[-1] - rope_head_dim
    ):
        return False
    torch.ops.trtllm.cute_dsl_dspark_rope_into(
        q, freqs_real, out, num_heads, rope_head_dim, out_rope_offset
    )
    return True


def _rmsnorm_rope_batched(
    t: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    rope_head_dim: int,
    freqs_cis: torch.Tensor,
    *,
    num_heads: int = 1,
    apply_weight: bool = True,
    apply_rmsnorm: bool = True,
    inverse_rope: bool = False,
    norm_dim: Optional[int] = None,
) -> torch.Tensor:
    """Fuse DSpark RMSNorm and last-dimension RoPE when supported.

    ``weight`` may be None when ``apply_weight`` is False -- no path reads it
    then, fused or eager.

    ``norm_dim`` bounds the RMSNorm (and the weight) to a prefix of the row.
    None means the whole row, which is DSpark's own convention. The other
    supported value is ``t.shape[-1] - rope_head_dim``: DeepSeek-style MLA
    normalizes the kv_lora_rank latent and leaves k_pe raw, and then ``weight``
    spans only that latent.
    """
    if IS_CUTLASS_DSL_AVAILABLE and is_sm_100f():
        freqs_real = torch.view_as_real(freqs_cis).reshape(-1, freqs_cis.shape[-1], 2)
        if is_fused_dspark_rmsnorm_rope_supported(
            t, weight, freqs_real, num_heads, rope_head_dim, norm_dim
        ):
            # Reads row-strided views (the fused [q_a | kv_a] GEMM's latent
            # columns) in place.
            return k3r_dspark_rmsnorm_rope(
                t,
                weight,
                freqs_real,
                num_heads,
                rope_head_dim,
                eps,
                apply_weight,
                apply_rmsnorm,
                inverse_rope,
                norm_dim,
            )

    split_norm = norm_dim is not None and norm_dim != t.shape[-1]
    head, tail = (t[..., :norm_dim], t[..., norm_dim:]) if split_norm else (t, None)
    if apply_rmsnorm:
        if apply_weight:
            head = _rmsnorm(head, weight, eps)
        else:
            head = head * torch.rsqrt(head.square().mean(-1, keepdim=True) + eps)
    elif apply_weight:
        head = (head.float() * weight.float()).to(head.dtype)
    t = head if tail is None else torch.cat([head, tail], dim=-1)
    if rope_head_dim > 0:
        t = _rope_last_dims_batched(t, rope_head_dim, freqs_cis, inverse=inverse_rope)
    return t


def _runtime_position_cap(model_config, pretrained_config, slack: int = 0) -> int:
    """Fallback RoPE table length: the served length, not the advertised one.

    The drafter sizes a position table from the checkpoint's
    max_position_embeddings. K3 advertises 1,048,576: ~256 MiB per rank as
    complex64, ~768 MiB transient via full-size fp32 cos/sin, while the context
    cache is bounded by the runtime max_seq_len. Only the table length moves;
    YaRN's correction range comes from original_max_position_embeddings.

    Shrinking the table this way is also what made its bound matter: sized from
    max_position_embeddings it could not be indexed past, sized from
    model_config.max_seq_len it can. Hence _runtime_position_ceiling, which
    the drafter prefers over this value whenever a worker has published it.
    """
    runtime = getattr(model_config, "max_seq_len", None)
    return int(runtime or getattr(pretrained_config, "max_position_embeddings", 163840)) + slack


def _dspark_markov_chain_logits(
    base_logits: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    argmax_fn=None,
) -> torch.Tensor:
    """Raw-tensor entry to ``dspark_markov_chain``, corrected logits only.

    The K3 ``modeling_speculative.dspark_markov_chain_logits``, which
    main does not have, rebuilt from main's chain and step bias. ``markov_w2``
    may already be sliced down to this rank's TP vocab shard, in which case
    ``argmax_fn`` must map a shard-local row back to a full-vocab token id.
    """

    def _step_bias(prev_tokens, step_hidden):
        del step_hidden
        return dspark_markov_step_bias(prev_tokens, markov_w1, markov_w2)

    _, corrected = dspark_markov_chain(
        base_logits,
        first_prev_tokens,
        _step_bias,
        next_token_fn=argmax_fn,
        cast_bias_to_logits=True,
    )
    return corrected


# ----------------------------------------------------------------------------
# MLA-shaped standalone drafter backbone (Inferact/Kimi-K3-DSpark).
#
# A weight container, not a servable model: nothing here registers with the
# attention backend or the KV cache manager, so the layers have no working
# ``forward``. ``MLADSparkForCausalLM`` runs its own block decode over them.
# Not a ``DeepseekV3`` subclass -- that carries MoE, a fused ``qkv_a_proj``
# switch, quantization and KV-manager registration, none of which apply.
# ----------------------------------------------------------------------------


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def build_dspark_mla_yarn_rope(
    *,
    dim: int,
    base: float,
    scaling_factor: float,
    original_max_position_embeddings: int,
    max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """YaRN cos/sin tables for an MLA drafter's ``qk_rope_head_dim`` slice.

    Repacks ``RopeEmbeddingUtils.create_sinusoidal_positions_yarn``, the same HF
    DeepSeek-V2 transcription the drafters are distilled under. NOT reused:
    ``RopeParams.create_rope_const_params()`` pins the fused MLA kernel's GPT-J
    packing and hard-codes ``device='cuda'``.

    ``duplicate_data=True`` though only the first half is read as a complex
    pair: the util's two modes disagree by 1 ulp on 6 of 8224 entries, and the
    duplicated one is what every measured drafter run used.

    Returns ``(cos, sin)``, both ``[max_position_embeddings, dim]`` fp32, with
    the frequency half duplicated so ``rotate_half`` applies.
    """
    # The util returns a flat numpy [pos][slot](cos, sin); split the pair.
    _, packed = RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
        num_pos=max_position_embeddings,
        dim=dim,
        base=base,
        scaling_factor=scaling_factor,
        original_max_position_embeddings=original_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        mscale=mscale,
        mscale_all_dim=mscale_all_dim,
        duplicate_data=True,
    )
    table = torch.from_numpy(packed).reshape(max_position_embeddings, dim, 2)
    return (
        table[..., 0].contiguous().to(device),
        table[..., 1].contiguous().to(device),
    )


def build_dspark_mla_yarn_freqs_cis(*, device: str = "cuda", **kwargs) -> torch.Tensor:
    """The same YaRN table as :func:`build_dspark_mla_yarn_rope`, adjacent-pair.

    Returns ``[max_position_embeddings, dim // 2]`` complex64: the convention the
    fused ``cute_dsl_dspark_rmsnorm_rope`` kernel consumes, where the last dim of
    the rotated slice is read as (re, im) pairs.

    HF DeepSeek de-interleaves and rotates halves instead, writing the same
    values to lanes i and i + dim/2 rather than 2i and 2i+1. That permutation
    cancels inside ``q_rope . k_rope``, so either convention works -- but every
    producer of a rope slice must use the SAME one, and mixing them only shows
    up as lower AL. ``mscale`` rides in the modulus, not the phase.
    """
    cos, sin = build_dspark_mla_yarn_rope(device=device, **kwargs)
    half = cos.shape[-1] // 2
    return torch.complex(cos[..., :half], sin[..., :half]).contiguous()


def apply_dspark_mla_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the trailing ``qk_rope_head_dim`` of ``x`` the HF DeepSeek way.

    ``x`` is ``[..., dim]`` and ``cos``/``sin`` are ``[..., dim]`` already
    gathered at the query positions. HF interleaves the rope slice, so it first
    reshapes ``(d//2, 2) -> transpose -> (d,)`` and only then applies the
    half-split ``rotate_half``; the net rotation is GPT-J style.

    Equal to ``RotaryEmbedding.apply_rotary_pos_emb(..., is_neox=False)`` up to
    an even/odd lane split of its output (checked exactly, max diff 0.0). Kept
    separate because that helper unsqueezes cos/sin itself, so routing through
    it would push a rank contract onto every caller for no numerical gain.
    """
    d = x.shape[-1]
    x = x.reshape(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(*x.shape[:-1], d)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class K3DsparkMLA(nn.Module):
    """MLA projections for one drafter layer. No attention, no KV registration.

    Head sharding follows :class:`~tensorrt_llm._torch.attention.mla.MLA`: the
    low-rank ``*_a`` projections are replicated (every head reads the same
    latent), ``q_b_proj`` / ``kv_b_proj`` are column-sharded by head and
    ``o_proj`` is row-sharded. Under attention-DP ``tp_size`` collapses to 1,
    which is also why the drafter's latent cache is unsharded.
    """

    def __init__(self, model_config, layer_idx: int, mapping_with_cp: Optional[Mapping] = None):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        dtype = config.torch_dtype

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        # What the drafter's KV cache stores per token: one MLA latent, no V.
        self.kv_cache_head_dim = self.kv_lora_rank + self.qk_rope_head_dim

        mapping = model_config.mapping
        if mapping.cp_size > 1:
            # MLA's own module folds cp_size into the head split and the o_proj
            # mapping; this drafter does not, and a silently wrong world_size
            # would mis-shard the head projections.
            raise NotImplementedError(
                f"The MLA DSpark drafter does not support context parallelism "
                f"(cp_size={mapping.cp_size})."
            )
        # Helix (mapping_with_cp): the latent cache is split by pages over the
        # CP ranks, so every rank needs every head's query (q_b / kv_b
        # replicated over the CP group); o_proj stays row-parallel over the
        # flattened tp x cp group (``mapping``, already repurposed).
        self.helix_mapping = (
            mapping_with_cp
            if mapping_with_cp is not None and mapping_with_cp.has_cp_helix()
            else None
        )
        o_mapping = mapping
        if self.helix_mapping is not None:
            mapping = self.helix_mapping
        tp_size = mapping.tp_size
        dp_size = 1
        if mapping.enable_attention_dp:
            dp_size = tp_size
            tp_size = 1
        elif self.helix_mapping is not None:
            # CP ranks hold identical head projections: fold them like DP.
            dp_size = mapping.cp_size
        if self.num_heads % tp_size != 0:
            raise ValueError(
                f"DSpark MLA drafter has {self.num_heads} heads, not divisible by tp_size {tp_size}."
            )
        self.num_heads_tp = self.num_heads // tp_size
        self.num_heads_tp_cp = self.num_heads_tp // mapping.cp_size
        attn_mapping = Mapping(
            world_size=mapping.pp_size * dp_size * tp_size,
            tp_size=tp_size,
            pp_size=mapping.pp_size * dp_size,
            rank=mapping.rank,
            gpus_per_node=mapping.gpus_per_node,
            enable_attention_dp=mapping.enable_attention_dp,
        )
        self.mapping = attn_mapping
        # Row-parallel o_proj reduces only when the heads are actually split.
        reduce_output = not o_mapping.enable_attention_dp and o_mapping.tp_size > 1

        skip_init = model_config.skip_create_weights_in_init
        self.q_a_proj = Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            dtype=dtype,
            skip_create_weights_in_init=skip_init,
        )
        self.q_a_layernorm = RMSNorm(
            hidden_size=self.q_lora_rank, eps=config.rms_norm_eps, dtype=dtype
        )
        self.q_b_proj = Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            dtype=dtype,
            mapping=attn_mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            skip_create_weights_in_init=skip_init,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        self.kv_a_proj_with_mqa = Linear(
            self.hidden_size,
            self.kv_cache_head_dim,
            bias=False,
            dtype=dtype,
            skip_create_weights_in_init=skip_init,
        )
        self.kv_a_layernorm = RMSNorm(
            hidden_size=self.kv_lora_rank, eps=config.rms_norm_eps, dtype=dtype
        )
        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            dtype=dtype,
            mapping=attn_mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            skip_create_weights_in_init=skip_init,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        self.o_proj = Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            dtype=dtype,
            mapping=attn_mapping if self.helix_mapping is None else o_mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=reduce_output,
            skip_create_weights_in_init=skip_init,
            allreduce_strategy=model_config.allreduce_strategy,
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "K3DsparkMLA holds the drafter's MLA weights for MLADSparkForCausalLM's "
            "block decode; it has no standalone attention path."
        )


class K3DsparkDecoderLayer(DecoderLayer):
    """One drafter layer: MLA projections + dense gated MLP + the two norms."""

    def __init__(self, model_config, layer_idx: int, mapping_with_cp: Optional[Mapping] = None):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.self_attn = K3DsparkMLA(model_config, layer_idx, mapping_with_cp)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
            dtype=config.torch_dtype,
            overridden_tp_size=1 if model_config.mapping.enable_attention_dp else None,
            config=model_config,
            layer_idx=layer_idx,
        )
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "K3DsparkDecoderLayer is driven by MLADSparkForCausalLM.dflash_forward, "
            "not by the generic decoder loop."
        )


class K3DsparkModel(DecoderModel):
    def __init__(self, model_config, mapping_with_cp: Optional[Mapping] = None):
        super().__init__(model_config)
        config = model_config.pretrained_config
        # The drafter ships its own trained embedding rather than sharing the
        # target's -- verified different weights on Inferact/Kimi-K3-DSpark --
        # so this is loaded from the drafter checkpoint, not overwritten.
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList(
            [
                K3DsparkDecoderLayer(model_config, layer_idx, mapping_with_cp)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "K3DsparkModel is a weight container for the DSpark block decode; "
            "see MLADSparkForCausalLM.dflash_forward."
        )


class K3DsparkForCausalLM(SpecDecOneEngineForCausalLM[K3DsparkModel, PretrainedConfig]):
    """The MLA DSpark drafter backbone with the helix page-sharded latent cache.

    Deliberately not registered. ``DFlashForCausalLM.__init__`` resolves the
    backbone through main's model registry, i.e. main's own
    ``K3DsparkForCausalLM``, which is identical to this one off helix (no
    ``mapping_with_cp``). ``MLADSparkForCausalLM`` builds this one only when the
    latent cache is page-sharded over helix CP
    (``MLADSparkForCausalLM._shard_context_kv_by_helix``).
    """

    def __init__(self, model_config):
        # Helix: only attention sees CP; build everything else on the
        # repurposed (CP folded into TP) mapping, then restore it.
        mapping_with_cp = None
        if model_config.mapping.has_cp_helix():
            mapping_with_cp = copy.deepcopy(model_config.mapping)
            frozen, model_config._frozen = model_config._frozen, False
            model_config.mapping = mapping_with_cp.repurpose_helix_cp_to_tp()
            model_config._frozen = frozen
        super().__init__(K3DsparkModel(model_config, mapping_with_cp), model_config)
        if mapping_with_cp is not None:
            frozen, model_config._frozen = model_config._frozen, False
            model_config.mapping = mapping_with_cp
            model_config._frozen = frozen


# ----------------------------------------------------------------------------
# Standalone DSpark drafter.
#
# The drafter ships as its own checkpoint instead of living in the target's
# ``mtp.*`` namespace, so it shares nothing with the target but the vocabulary
# and the captured hidden states. Its block decode is DFlash's -- DSpark is
# DFlash plus a Markov logit bias, a confidence head and the shift_label slot
# convention -- so it subclasses ``DFlashForCausalLM`` and adds exactly those
# three.
# ----------------------------------------------------------------------------

# Both published drafters (RadixArk/Kimi-K3-DSpark, Inferact/Kimi-K3-DSpark)
# name the head tensors after the submodules that own them: markov_head is a
# VanillaMarkov, confidence_head an AcceptRatePredictor whose linear is ``proj``.
# The bare spellings are kept for drafters exported without that nesting.
_DSPARK_HEAD_WEIGHT_ALIASES = {
    "markov_w1.weight": ("markov_head.markov_w1.weight", "markov_w1.weight"),
    "markov_w2.weight": ("markov_head.markov_w2.weight", "markov_w2.weight"),
    "confidence_proj.weight": ("confidence_head.proj.weight", "confidence_proj.weight"),
    "confidence_proj.bias": ("confidence_head.proj.bias", "confidence_proj.bias"),
}


_LN2 = math.log(2.0)
# flashinfer sizes its own scratch from this; the MLA decode path is the only
# consumer here and 256 MiB is what the DFlash TRTLLM buffers already reserve.
_MLA_DECODE_WORKSPACE_BYTES = 256 * 1024 * 1024
# Fixed cost of one split-KV piece, in KV tiles, in the per-request split plan
# of the cute-dsl block decode (launch_mla_split_kv_plan): 64 heads folded two
# tokens per M tile, fitted on R200 sweeps over 7.5K-200K tokens of context,
# batch 1-32, equal and mixed lengths. An FP8 KV tile takes half the time of a
# BF16 one, so the same fixed cost is twice the tiles.
_MLA_SPLIT_PLAN_PIECE_COST_TILES = 30
_MLA_SPLIT_PLAN_PIECE_COST_TILES_FP8 = 60


class _MLABlockFixup(NamedTuple):
    """Layer-invariant state for the MLA block-decode fixup.

    The block's own latents are written into the pool at ``ctx_len + j``. Those
    slots belong to this request -- the draft KV manager adds
    ``max(draft_len, max_total_draft_tokens)`` tokens per generation request each
    step (resource_manager.py:1060) -- and the accepted tokens overwrite them next
    step. Same thing the GQA drafter's TRTLLM backend does.

    In-block causality then leaves query j seeing block keys 0..j, so only the
    strict upper triangle needs fixing up. Every field depends on ctx_len, the
    page table and the block geometry alone, so it is built once per forward.
    """

    pages: torch.Tensor  # [B, block] page holding each block position
    offsets: torch.Tensor  # [B, block] offset of that position inside its page
    valid: torch.Tensor  # [1, block, 1, block] strict upper triangle
    page_tables_i32: torch.Tensor  # kernel operands, cast once
    seq_lens_i32: torch.Tensor  # ctx_len + block_size
    kv_bounds_i32: torch.Tensor  # [B * block] seq_lens repeated per query token
    # Helix: [B, block, 1] whether this rank's pages hold each block position;
    # pages/offsets and the lengths above are then rank-local.
    owned: Optional[torch.Tensor] = None


@lru_cache(maxsize=1)
def cute_dsl_mla_decode_unavailability_reason() -> Optional[str]:
    """Return why the cute-dsl MLA decode cannot run here, or None.

    Mirrors ``dflash_trtllm_gen_unavailability_reason``: the op itself only
    reports by raising, and ``attention_backend="CUTEDSL"`` has to fail at
    construction rather than five layers into the first draft step.

    The ``kv_bounds`` probe is not paranoia. The op exists upstream without
    per-token KV bounds, and this backend's whole shape -- one pass over
    context and block, no fixup -- is expressed through them. Without the probe
    the mismatch surfaces as a torch dispatch TypeError at the first draft
    step, naming an argument count rather than a missing kernel feature.
    """
    if (sm := get_sm_version()) not in (100, 103, 107):
        return f"requires SM100, SM103 or SM107, got SM{sm}"
    try:
        import cutlass  # noqa: F401

        from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
            CuteDSLNVMlaDecodeBlackwellRunner,  # noqa: F401
        )
    except ImportError as error:
        return f"the cute-dsl kernels are not importable: {error}"
    op = getattr(torch.ops.trtllm, "cute_dsl_mla_decode_fp16_blackwell", None)
    if op is None:
        return "trtllm::cute_dsl_mla_decode_fp16_blackwell is not registered"
    if not any(
        arg.name == "kv_bounds"
        for overload in op.overloads()
        for arg in getattr(op, overload)._schema.arguments
    ):
        return (
            "the registered cute_dsl_mla_decode_fp16_blackwell takes no kv_bounds, "
            "so per-token KV bounds are unavailable in this build"
        )
    return None


def _all_reduce_add_norm(
    all_reduce: AllReduce, partial: torch.Tensor, residual: torch.Tensor, norm: RMSNorm
) -> tuple[torch.Tensor, torch.Tensor]:
    """``norm(all_reduce(partial) + residual)`` and the new residual, fused."""
    return all_reduce(
        partial,
        all_reduce_params=AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=norm.weight,
            eps=norm.variance_epsilon,
        ),
    )


def _block_upper_triangle(block_size: int, device) -> torch.Tensor:
    """``[1, block, 1, block]``: query i sees block key j > i (the fixup's part)."""
    idx = torch.arange(block_size, device=device)
    return (idx.view(1, block_size) > idx.view(block_size, 1)).view(1, block_size, 1, block_size)


def _build_mla_block_fixup(
    ctx_len, page_tables, block_size, page_size, shard: Optional[HelixDraftShard] = None
) -> _MLABlockFixup:
    """Where the block's latents go, and the upper-triangle mask, for all layers.

    With a helix ``shard`` every position goes to its rank-local slot and the
    lengths count only this rank's positions.
    """
    device = ctx_len.device
    pos = ctx_len.view(-1, 1) + torch.arange(block_size, device=device)
    seq_lens = (ctx_len + block_size).to(torch.int32)
    owned = None
    if shard is not None:
        pos, owned = shard.local_slots(pos)
        owned = owned.unsqueeze(-1)
        seq_lens = shard.local_lengths(seq_lens)
    return _MLABlockFixup(
        owned=owned,
        pages=page_tables.gather(1, pos // page_size),
        offsets=pos % page_size,
        valid=_block_upper_triangle(block_size, device),
        page_tables_i32=page_tables.to(torch.int32),
        seq_lens_i32=seq_lens,
        # repeat_interleave, not repeat: the kernel indexes kv_bounds at
        # b * seq_len_q + q (mla_decode_fp16.py:400-407), so the batch must be
        # the slow axis. B == 1 cannot tell the two apart.
        kv_bounds_i32=seq_lens.repeat_interleave(block_size),
    )


def _resolve_dspark_mla_rope_params(pretrained_config, model_config=None, slack: int = 0) -> Dict:
    """RoPE settings for an MLA drafter, from either HF spelling.

    Transformers v5 checkpoints (TorchSpec) carry ``rope_parameters``; older
    ones carry ``rope_scaling``. Only YaRN is accepted: both published MLA
    drafters use it, and silently treating an unknown scaling as plain RoPE
    would cost acceptance without an error.
    """
    scaling = (
        getattr(pretrained_config, "rope_parameters", None)
        or getattr(pretrained_config, "rope_scaling", None)
        or {}
    )
    rope_type = scaling.get("rope_type") or scaling.get("type") or "default"
    if rope_type != "yarn":
        raise ValueError(
            f"MLA DSpark drafter declares rope_type={rope_type!r}; only 'yarn' is "
            "supported (the block decode builds its own YaRN table)."
        )
    theta = scaling.get("rope_theta") or getattr(pretrained_config, "rope_theta", 10000.0)
    return {
        "theta": float(theta),
        "scaling_factor": float(scaling["factor"]),
        "original_max_positions": int(scaling["original_max_position_embeddings"]),
        # Table length, not YaRN math: the correction range below uses
        # original_max_positions, so capping this at the served length is free.
        # K3 advertises 1,048,576, which would build a ~256 MiB complex64 table
        # per rank -- through full-size fp32 cos/sin, so ~768 MiB transient --
        # for a drafter whose point is 5760 B/token instead of 20480.
        "max_positions": _runtime_position_cap(model_config, pretrained_config, slack),
        "beta_fast": float(scaling.get("beta_fast", 32.0)),
        "beta_slow": float(scaling.get("beta_slow", 1.0)),
        "mscale": float(scaling.get("mscale", 1.0)),
        "mscale_all_dim": float(scaling.get("mscale_all_dim", 0.0)),
    }


@lru_cache(maxsize=None)
def _mla_split_plan_kwargs(piece_cost: int) -> Optional[tuple]:
    """``launch_mla_split_kv_plan`` keyword items that plan at ``piece_cost``, or None.

    The K3 planner takes the piece cost per call; main's compiles its own
    ``_SPLIT_PLAN_PIECE_COST_TILES`` in. On such a build only that cost can be
    planned, and any other cost keeps the uniform split (main's behaviour).
    """
    from tensorrt_llm._torch.attention.backends.fmha import cute_dsl_mla

    if "piece_cost" in inspect.signature(cute_dsl_mla.launch_mla_split_kv_plan).parameters:
        return (("piece_cost", piece_cost),)
    if piece_cost == getattr(cute_dsl_mla, "_SPLIT_PLAN_PIECE_COST_TILES", None):
        return ()
    return None


def _mla_split_plan_geometry(num_heads: int, seq_len_q: int) -> tuple[int, int]:
    """``(rows_per_request, wave_clusters)`` of the cute-dsl MLA decode's launch.

    The runner's own ``get_split_plan_geometry`` when it has one; otherwise the
    geometry main's target MLA planner hands the same kernel
    (``CuteDslMlaFmha``: packed 128-row query tiles, one 2-CTA cluster per
    split and tile).
    """
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        CuteDSLNVMlaDecodeBlackwellRunner as runner,
    )

    geometry = getattr(runner, "get_split_plan_geometry", None)
    if geometry is not None:
        return geometry(num_heads, seq_len_q)
    return (
        runner.get_num_q_tiles(num_heads, seq_len_q),
        max(runner._get_max_active_blocks() // runner._CLUSTER_SHAPE_MNK[0], 1),
    )


@lru_cache(maxsize=1)
def _helix_post_process_fn() -> tuple[Callable, bool]:
    """Main's ``_helix_post_process`` and whether it takes ``compile_nccl_reformat``."""
    from tensorrt_llm._torch.attention.attention import _helix_post_process

    takes_flag = "compile_nccl_reformat" in inspect.signature(_helix_post_process).parameters
    return _helix_post_process, takes_flag


def _draft_helix_post_process(
    partial_o: torch.Tensor,
    stats: torch.Tensor,
    mapping: Mapping,
    num_heads_tp_cp: int,
    value_dim: int,
    zero_kv_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """``_helix_post_process`` for the drafter: NCCL reformats eager where supported.

    The drafter meets each batch bucket's shape first in a warmup forward,
    after the target's layers: no compile between NCCL all-to-alls (it can
    block every rank). Main's ``_helix_post_process`` has no
    ``compile_nccl_reformat`` switch, so there its NCCL path keeps compiling.
    """
    fn, takes_flag = _helix_post_process_fn()
    kwargs = {"zero_kv_mask": zero_kv_mask}
    if takes_flag:
        kwargs["compile_nccl_reformat"] = False
    return fn(partial_o, stats, mapping, num_heads_tp_cp, value_dim, **kwargs)


class _DSparkHeadMixin:
    """The DSpark head set of a standalone drafter wrapper.

    DSpark is DFlash plus three things common to both backbone shapes: the
    vanilla Markov intra-block logit bias, the ``shift_label`` slot convention
    (block slot j predicts draft token j+1, so slot 0 holds the anchor) and the
    confidence head. The block decode has nothing in common between the shapes,
    which is why this is a mixin and not a base class.

    Confidence-scheduled verification is not implemented yet: ``confidence_proj``
    is loaded but unused, and drafting always proposes the full K tokens.
    """

    def _init_dspark_heads(self, cfg) -> None:
        # Defaults on, unlike the DFlash base: the shift_label slot layout is
        # part of what DSpark *is*, and both published drafters set
        # block_size == max_draft_len, where the DFlash layout (slots 1..K)
        # runs one slot past the block and reads the next request's anchor.
        # An explicit false still selects the legacy layout.
        shift_label = resolve_dspark_head_config(cfg, "shift_label")
        self._dspark_shift_label = True if shift_label is None else bool(shift_label)
        self._dspark_markov_rank = int(resolve_dspark_head_config(cfg, "markov_rank") or 0)
        self._dspark_markov_head_type = str(
            resolve_dspark_head_config(cfg, "markov_head_type") or "vanilla"
        ).lower()
        self._dspark_use_confidence_head = bool(
            resolve_dspark_head_config(cfg, "use_confidence_head") or False
        )
        # Plain None placeholders rather than nn.Parameter/buffer: the shapes
        # ([vocab, rank]) are checkpoint-dependent, so nothing is pre-allocated
        # and nothing is constructed in the module's default dtype. Using the
        # checkpoint tensor as-is is also what keeps the head in the checkpoint's
        # dtype instead of an nn.Module default. load_weights() fills them in;
        # consumers treat None as "head absent".
        self.markov_w1 = None  # [vocab, rank] (nn.Embedding weight layout)
        self.markov_w2 = None  # [vocab, rank] (nn.Linear(rank->vocab) weight)
        self.confidence_proj_weight = None  # loaded, unused (follow-up MR)
        self.confidence_proj_bias = None

        if self._dspark_markov_rank > 0 and self._dspark_markov_head_type != "vanilla":
            raise ValueError(
                f"DSpark drafter declares markov_head_type="
                f"'{self._dspark_markov_head_type}'; only 'vanilla' is "
                "supported (gated/rnn heads need per-step hidden features)."
            )
        # The block decode only supports the non-causal DSpark convention.
        # Legacy DFlash drafter configs (e.g. Laguna) also carry a causal field
        # and handle it in the legacy decode path, which is why this check lives
        # here rather than in the DFlash base.
        if resolve_dspark_head_config(cfg, "causal"):
            raise ValueError(
                "DSpark drafter sets causal=true; the block decode only "
                "supports the non-causal DSpark convention."
            )
        if self._dspark_use_confidence_head:
            logger.warning(
                "DSpark drafter declares use_confidence_head; "
                "confidence-scheduled verification is not implemented yet "
                "(confidence_proj weights are loaded but unused, drafting "
                "always proposes the full K tokens)."
            )

    @property
    def has_markov_head(self) -> bool:
        return self._dspark_markov_rank > 0 and self.markov_w1 is not None

    def apply_markov_chain_logits(
        self,
        base_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        tp_exchange=None,
        argmax_fn=None,
        vocab_slice: Optional[slice] = None,
    ) -> torch.Tensor:
        """Apply the vanilla-Markov intra-block bias to block logits.

        No-op (returns ``base_logits`` unchanged) when the checkpoint ships no
        Markov head. See :func:`dspark_markov_chain` for the semantics. Two
        calling conventions, one per worker:

        * main's ``DSparkWorker``: float32 ``base_logits`` of the full vocab, or
          this rank's vocab shard with the TP group's ``tp_exchange`` --
          :func:`markov_chain_logits` runs the chain;
        * the K3 ``DFlashWorker``: ``argmax_fn`` returning full-vocab token ids
          and, for a TP vocab shard, this rank's ``vocab_slice`` (to shard the
          markov_w2 rows identically) -- the eager chain runs.
        """
        if not self.has_markov_head:
            return base_logits
        if argmax_fn is None and vocab_slice is None:
            return markov_chain_logits(
                base_logits, first_prev_tokens, self.markov_w1, self.markov_w2, tp_exchange
            )
        if tp_exchange is not None:
            raise ValueError(
                "apply_markov_chain_logits takes either tp_exchange or argmax_fn / vocab_slice, "
                "not both."
            )
        markov_w2 = self.markov_w2 if vocab_slice is None else self.markov_w2[vocab_slice]
        return _dspark_markov_chain_logits(
            base_logits, first_prev_tokens, self.markov_w1, markov_w2, argmax_fn=argmax_fn
        )

    def _take_dspark_head_weights(self, weights: Dict) -> tuple[Dict, Dict]:
        """Pull the head tensors out of ``weights`` and load them onto self.

        The head keys are pulled out before the backbone remap: left in, they
        would pick up a ``model.`` prefix and be dropped by partial loading.
        Returns the remaining weights and the extracted head weights.
        """
        dspark_weights = {}
        consumed = set()
        for canonical, aliases in _DSPARK_HEAD_WEIGHT_ALIASES.items():
            for name in aliases:
                if name in weights:
                    dspark_weights[canonical] = weights[name]
                    consumed.add(name)
                    break
        if consumed:
            weights = {k: v for k, v in weights.items() if k not in consumed}
        # The inverse of the missing-weights check below. Without it, a config
        # whose head switches this build cannot resolve loads the drafter with
        # the heads silently dropped -- correct output, lower acceptance.
        if self._dspark_markov_rank <= 0 and "markov_w1.weight" in dspark_weights:
            raise ValueError(
                "DSpark drafter ships markov_w1/markov_w2 but markov_rank resolved to 0. "
                "The checkpoint's head switches were not found in dspark_config, "
                "dflash_config, or at the top level; loading it would drop the Markov "
                "head silently."
            )
        if self._dspark_markov_rank > 0:
            vocab = self.config.vocab_size
            rank = self._dspark_markov_rank
            for k in ("markov_w1.weight", "markov_w2.weight"):
                if k not in dspark_weights:
                    raise ValueError(
                        f"DSpark drafter declares markov_rank="
                        f"{self._dspark_markov_rank} but the checkpoint is "
                        f"missing {k}."
                    )
                if tuple(dspark_weights[k].shape) != (vocab, rank):
                    raise ValueError(
                        f"DSpark {k} has shape "
                        f"{tuple(dspark_weights[k].shape)}, expected "
                        f"[vocab, markov_rank] = ({vocab}, {rank})."
                    )
            self.markov_w1 = dspark_weights["markov_w1.weight"].to("cuda")
            self.markov_w2 = dspark_weights["markov_w2.weight"].to("cuda")
        if "confidence_proj.weight" in dspark_weights:
            self.confidence_proj_weight = dspark_weights["confidence_proj.weight"].to("cuda")
        if "confidence_proj.bias" in dspark_weights:
            self.confidence_proj_bias = dspark_weights["confidence_proj.bias"].to("cuda")
        return weights, dspark_weights


class _K3DFlashDrafterMixin:
    """The K3 ``DFlashForCausalLM`` additions, for the MLA drafter.

    Main's ``DFlashForCausalLM`` (``modeling_dflash.py``) stays main's; what the
    K3 changes add to it and the MLA drafter reaches lives here: the
    disaggregated context-server flag, the capture projection the worker
    accumulates layer by layer, and the TP-sharded ``fc`` of
    ``TLLM_DSPARK_DRAFT_FUSION``. That sharded fc replaces main's own fc split
    (``_tp_fusions``) rather than stacking on it: with the switch set at load
    time ``fc`` keeps its whole weight, and a fc main already split is never
    sharded again.

    The K3 GQA-only changes to ``DFlashForCausalLM.dflash_forward``
    (TRTLLM folded decode, FP8 cache, helix sharding of the GQA cache) are not
    here: the MLA drafter brings its own ``dflash_forward``.
    """

    # True for a drafter built for a disaggregated context server: it only
    # stores the context KV and holds no modules for the block decode.
    context_kv_only = False
    # The K3 DFlash worker binds to a unified draft cache: on KV cache manager
    # V2, KvCacheCreator appends this drafter's context KV to the target manager.
    supports_unified_draft_cache = True

    def _load_target_projection(self, weights: Dict) -> None:
        if not draft_fusion_enabled():
            super()._load_target_projection(weights)
            return
        # TLLM_DSPARK_DRAFT_FUSION: keep the whole fc; _resolve_fc_shard takes
        # this rank's columns from it.
        if "fc.weight" not in weights:
            return
        weight = weights.pop("fc.weight")
        self.fc = nn.Linear(
            weight.shape[1], weight.shape[0], bias=False, device="cuda", dtype=weight.dtype
        )
        self.fc.weight.data.copy_(weight)

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project captured target hidden states into the draft hidden space.

        ``TLLM_DSPARK_DRAFT_FUSION``: the TP-sharded fc when it applies.
        Otherwise main's ``_project_features`` (its own split fc under the TP
        fusions, else ``hidden_norm(fc(...))``).
        """
        hidden_states = hidden_states.to(self.fc.weight.dtype)
        if draft_fusion_enabled() and hidden_states.shape[0] > 0:
            projected = self._tp_sharded_project_target_hidden(hidden_states)
            if projected is not None:
                return projected
        return self._project_features(hidden_states)

    def capture_projection_blocks(self, num_captures: int) -> Optional[list[torch.Tensor]]:
        """``fc.weight`` split into one column block per captured target layer.

        ``fc`` over the concatenated captures equals the sum over captured
        layers of ``hidden_i @ blocks[i].T``, which lets the capture accumulate
        the projection layer by layer instead of keeping every layer's hidden
        states (see ``DFlashSpecMetadata.project_captures``). The blocks are
        views of ``fc.weight`` ``[out_features, in_features // num_captures]``.

        Returns None when ``project_target_hidden`` is not that plain bias-free
        ``fc`` followed by ``hidden_norm`` (a subclass override, main's split fc,
        or the TP-sharded projection), or the input width does not split evenly.
        """
        if type(self).project_target_hidden is not _K3DFlashDrafterMixin.project_target_hidden:
            return None
        if self.fc.bias is not None or self._fc_input_slice is not None:
            return None
        if draft_fusion_enabled() and self._resolve_fc_shard() is not None:
            return None
        weight = self.fc.weight
        width, remainder = divmod(weight.shape[1], num_captures)
        if remainder:
            return None
        return [weight[:, i * width : (i + 1) * width] for i in range(num_captures)]

    def finish_capture_projection(self, projected: torch.Tensor) -> torch.Tensor:
        """``project_target_hidden``'s result from the summed ``fc`` partials.

        Args:
            projected: ``[num_tokens, out_features]`` FP32 sum of the captured
                layers' ``hidden_i @ blocks[i].T`` (``capture_projection_blocks``).

        Returns:
            ``hidden_norm(fc(captures))`` in the fc weight dtype, rounded once
            from the FP32 sum like the single fc GEMM.
        """
        return self.hidden_norm(projected.to(self.fc.weight.dtype))

    def _resolve_fc_shard(self):
        """``(weight slice, begin, end, all_reduce)`` of the TP-sharded fc, or None.

        Resolved once: the shard exists when the drafter runs plain TP (no
        attention DP), the fc input width splits evenly over it, and main's TP
        fusions have not already split fc (``_fc_input_slice``).
        """
        if getattr(self, "_fc_shard", None) is None:
            self._fc_shard = False
            layers = getattr(self.model, "layers", None)
            o_proj = (
                getattr(getattr(layers[0], "self_attn", None), "o_proj", None) if layers else None
            )
            all_reduce = getattr(o_proj, "all_reduce", None)
            mapping = None if all_reduce is None else all_reduce.mapping
            in_features = self.fc.weight.shape[1]
            if (
                mapping is not None
                and self._fc_input_slice is None
                and mapping.tp_size > 1
                and not mapping.enable_attention_dp
                and in_features % mapping.tp_size == 0
                and isinstance(self.hidden_norm, nn.RMSNorm)
            ):
                width = in_features // mapping.tp_size
                begin = mapping.tp_rank * width
                weight = self.fc.weight.data[:, begin : begin + width].contiguous()
                self._fc_shard = (weight, begin, begin + width, all_reduce)
        return self._fc_shard or None

    def _tp_sharded_project_target_hidden(self, hidden_states: torch.Tensor):
        """fc split along its input features over the drafter's TP group.

        Each rank multiplies its slice of the captured features by its slice of
        ``fc.weight`` columns; one all-reduce sums the partials and, over MNNVL,
        applies hidden_norm in the same kernel. Every rank reads 1/tp of the fc
        weight instead of all of it. Returns None when the drafter is not
        TP-sharded or the feature width does not split evenly.

        The TP group is o_proj's all-reduce group: under helix the drafter's
        ``model_config.mapping`` carries the CP split of its context KV instead.
        """
        shard = self._resolve_fc_shard()
        if shard is None:
            return None
        weight, begin, end, all_reduce = shard
        partial = F.linear(hidden_states[:, begin:end], weight)
        eps = self.hidden_norm.eps
        if all_reduce.mnnvl_allreduce is not None and eps is not None:
            return all_reduce(
                partial,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RMS_NORM,
                    norm_weight=self.hidden_norm.weight,
                    eps=eps,
                ),
            )
        return self.hidden_norm(all_reduce(partial))


class MLADSparkForCausalLM(_DSparkHeadMixin, _K3DFlashDrafterMixin, DFlashForCausalLM):
    """DSpark drafter on an MLA-shaped backbone, from a standalone checkpoint.

    Separate from the GQA class because DFlash's block decode splits a fused
    ``qkv_proj`` by (num_heads, num_kv_heads, head_dim) and fuses per-head K/V
    across layers, neither of which MLA has. It keeps the DFlash contract --
    dual-source KV, non-causal block attention, one precomputed context cache.

    The cache stores one ``kv_lora_rank + qk_rope_head_dim`` latent per token
    per layer and no V, which is the point: 5 x 576 x 2B = 5760 B/token/rank
    against the GQA drafter's 20480 under attention-DP, where KV heads are
    unsharded.

    ``spec_config.attention_backend`` selects among this class's own three
    block-decode implementations, not among the shared DFlash worker op sets:

    ``CUTEDSL``  one cute-dsl pass with ``kv_bounds``, no fixup; paged only, AUTO's pick.
    ``TRTLLM``   trtllm-gen absorbed-MLA paged decode + upper-triangle fixup.
    ``VANILLA``  the eager torch reference; where the other two land unpaged.

    ``context_kv_dtype="fp8"`` stores the latents as FP8 E4M3 at a unit
    per-tensor scale (``to_draft_fp8``), 2880 instead of 5760 B/token/rank, and
    the CUTEDSL decode then reads them with an FP8 query through the FP8 MLA
    decode; the other two backends read BF16 only.

    Reference: arXiv 2607.05147; deepseek-ai/DeepSpec.
    """

    # Attention is _mla_paged_attention, so neither worker OP SET applies and
    # neither backend's shape checks bind -- the absorbed shape (64 heads : 1 KV
    # head, head_dim 576) would fail both. The backend NAME still selects, via
    # _mla_block_decode_variant below.
    _uses_worker_attention_backend = False
    # CUTEDSL by default: one cute-dsl pass over context and block through the
    # per-token kv_bounds of cute_dsl_mla_decode_fp16_blackwell, with no
    # upper-triangle fixup -- 35.7% off the drafter region (947.50 -> 608.95
    # us/iter) at unchanged acceptance and accuracy. It does not degrade (see
    # _auto_fallback_attention_backend): a build without the kernel raises in
    # __init__ with the reason from cute_dsl_mla_decode_unavailability_reason.
    _default_attention_backend = "CUTEDSL"
    _supported_attention_backends = ("VANILLA", "TRTLLM", "CUTEDSL")
    # No AUTO degrade. This drafter needs a 288 GB Blackwell part to hold the
    # target at all, so a build that cannot run its kernels cannot run the
    # model either; falling back to the eager reference would hide that behind
    # a path orders slower at identical output.
    _auto_fallback_attention_backend = None

    # The whole point of the MLA drafter: its context KV comes out of the
    # manager's pool (5760 B/token/rank) instead of an arena dense in
    # max_seq_len that free_gpu_memory_fraction never bounds. Independent of the
    # backend field, which is why either value of it works here.
    _paged_ctx_cache = True

    # Acceptance length is measured with the TP fusions: the split fc, and each
    # o_proj / down_proj all-reduce of the block decode below fused with the
    # residual add and the RMSNorm after it. TLLM_DSPARK_DRAFT_FUSION replaces
    # them with the K3 set (see _K3DFlashDrafterMixin and dflash_forward).
    _tp_fusions = True

    def __init__(
        self,
        draft_config,
        *,
        dflash_attention_backend: str = "AUTO",
        context_kv_only: bool = False,
        context_kv_dtype: str = "auto",
    ):
        if context_kv_dtype not in ("auto", "fp8"):
            raise ValueError(f"context_kv_dtype must be 'auto' or 'fp8', got {context_kv_dtype!r}")
        cfg = draft_config.pretrained_config
        # Pin the backbone instead of relying on the model_type-derived name
        # (the Laguna precedent in modeling_dflash.py): the checkpoint labels
        # itself "K3DSparkModel", which is registered nowhere.
        cfg.architectures = ["K3DsparkForCausalLM"]
        backend = dflash_attention_backend
        if context_kv_only:
            # A disaggregated context server never runs the block decode, so a
            # build that cannot run its kernels may still serve it. The base
            # probes whatever backend it is handed; it gets VANILLA, which loads
            # nothing for this family, and the name asked for is restored below
            # (the helix page sharding keys on it).
            backend = self._default_attention_backend if backend == "AUTO" else backend
            if backend not in self._supported_attention_backends:
                raise ValueError(
                    f"{type(self).__name__} attention backend must be one of "
                    f"{list(self._supported_attention_backends)}, got {backend!r}."
                )
        super().__init__(
            draft_config,
            dflash_attention_backend="VANILLA" if context_kv_only else backend,
        )
        if context_kv_only:
            self.dflash_attention_backend = backend
        self.context_kv_only = context_kv_only
        # The TP-sharded fc of TLLM_DSPARK_DRAFT_FUSION, resolved on first use.
        self._fc_shard = None
        # The helix page sharding of the latent cache, or None. Main's
        # DFlashForCausalLM.__init__ resolved the backbone through the registry
        # with the unsharded mapping; a sharded cache rebuilds it here.
        self._helix_mapping = self._shard_context_kv_by_helix(
            draft_config, self.dflash_attention_backend
        )
        if self._helix_mapping is not None:
            self._rebuild_backbone_for_helix(draft_config)
        self._init_dspark_heads(cfg)

        attn0 = self.model.layers[0].self_attn
        self.qk_nope_head_dim = attn0.qk_nope_head_dim
        self.qk_rope_head_dim = attn0.qk_rope_head_dim
        self.kv_lora_rank = attn0.kv_lora_rank
        self.v_head_dim = attn0.v_head_dim
        # The cache contract the worker sizes its arena from. Config-derived, so
        # it is published here rather than waiting for the lazy buffer build:
        # one MLA latent per token per layer, no V half.
        self._num_attn_layers = len(self.model.layers)
        self._num_heads = attn0.num_heads_tp
        self._num_kv_heads = 1
        self._head_dim = attn0.kv_cache_head_dim
        self._kv_factor = 1

        # No slack: the lookahead lives in dflash_position_ceiling, which the
        # worker publishes as _runtime_position_ceiling before any forward and
        # _mla_freqs_cis builds the table from. This config-derived cap is only
        # the fallback for direct construction, where no worker runs and the
        # positions asked for are whatever the caller passes.
        rope = _resolve_dspark_mla_rope_params(cfg, draft_config)
        self._mla_rope_params = rope
        self._mla_workspace = None
        self._mla_freqs = None
        self._mla_freqs_cap = None
        self._mla_fp8_scales = None
        self._cute_dsl_checked_shape = None
        # (fixup, fp8, plan) of the last _mla_split_kv_plan.
        self._mla_split_plan = None
        # HF DeepSeek folds YaRN's attention scaling into the softmax scale:
        # softmax_scale = mscale^2 / sqrt(qk_nope + qk_rope). NOT 1/sqrt(576) --
        # the absorbed product reproduces the 192-dim un-absorbed score.
        mscale = _yarn_get_mscale(rope["scaling_factor"], rope["mscale_all_dim"])
        self.softmax_scale = (mscale * mscale) / math.sqrt(
            self.qk_nope_head_dim + self.qk_rope_head_dim
        )
        # The pool's dtype is what the decode keys on; this flag is what sizes
        # the pool (KvCacheCreator) and what rejects a backend that reads BF16.
        self.context_kv_fp8 = context_kv_dtype == "fp8"
        if (
            self.context_kv_fp8
            and not context_kv_only
            and self.dflash_attention_backend != "CUTEDSL"
        ):
            raise ValueError(
                "speculative_config.context_kv_dtype='fp8' is read by the CUTEDSL block decode "
                f"only, but attention_backend resolved to {self.dflash_attention_backend!r}."
            )
        if context_kv_only:
            self._keep_context_kv_modules()

    def _rebuild_backbone_for_helix(self, draft_config) -> None:
        """Rebuild the backbone on the helix mapping ``draft_config`` now carries.

        The package-internal :class:`K3DsparkForCausalLM` keeps attention on the CP
        mapping and everything else on its repurposed TP form. Construction runs
        under meta init, so the discarded registry backbone never allocated
        device memory.
        """
        # Remove spec_config to prevent recursive spec-dec initialization.
        draft_config_no_spec = replace(draft_config, spec_config=None, lm_head_gather_output=False)
        # ModelConfig.extra_attrs is init=False, so dataclasses.replace() does
        # not preserve the shared custom-op registries.
        draft_config_no_spec.extra_attrs = draft_config.extra_attrs
        self.draft_model_full = K3DsparkForCausalLM(draft_config_no_spec)
        self.model = self.draft_model_full.model
        self.lm_head = self.draft_model_full.lm_head
        self.model_config = draft_config_no_spec
        self.config = draft_config_no_spec.pretrained_config
        mapping = draft_config_no_spec.mapping
        self._tp_split = (
            self._tp_fusions and mapping.tp_size > 1 and not mapping.enable_attention_dp
        )
        self._validate_gqa_shape()

    def _keep_context_kv_modules(self) -> None:
        """Drop every module the context KV build does not read.

        A disaggregated context server only stores the drafter's context KV:
        the target hidden states through ``fc`` / ``hidden_norm``, then each
        layer's ``kv_a_proj_with_mqa`` and ``kv_a_layernorm`` plus RoPE
        (``precompute_context_kv``). The embedding, the query / output
        projections, the MLPs, the LM head and the DSpark heads are never read
        there. Construction runs under meta init, so dropped modules never
        allocate device memory.
        """
        backbone = self.draft_model_full.model
        backbone.embed_tokens = None
        backbone.norm = None
        for layer in backbone.layers:
            layer.mlp = None
            layer.input_layernorm = None
            layer.post_attention_layernorm = None
            for name in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_b_proj", "o_proj"):
                setattr(layer.self_attn, name, None)
        self.draft_model_full.lm_head = None
        self.lm_head = None

    def _shard_context_kv_by_helix(self, draft_config, attention_backend: str):
        """Split the latent cache by pages over the whole repurposed helix group.

        One latent per token cannot be split by head, so attention runs at
        tp = 1 with every head on every rank; only the CUTEDSL decode takes the
        rank-local bounds this needs. Other backends keep the replicated cache.

        Uses ``Mapping.helix_mapping_for_kv_heads``; without it the cache stays
        replicated and this returns None. The sharded pages live in the unified
        draft cache, which sizes them from the returned ``_helix_mapping``.
        """
        helix_mapping_for_kv_heads = getattr(
            draft_config.mapping, "helix_mapping_for_kv_heads", None
        )
        if helix_mapping_for_kv_heads is None:
            return None
        # TRTLLM_K3_ULL2_DRAFT_A2A: exchange on the target's helix transport.
        mapping = helix_mapping_for_kv_heads(1, inherit_cp_config=draft_helix_a2a_enabled())
        if mapping is draft_config.mapping:
            return None
        if attention_backend != "CUTEDSL":
            logger.warning(
                f"MLA DSpark: attention_backend={attention_backend!r} keeps the latent "
                f"cache replicated on all {draft_config.mapping.tp_size} ranks; only "
                "CUTEDSL shards it over helix CP."
            )
            return None
        frozen, draft_config._frozen = draft_config._frozen, False
        draft_config.mapping = mapping
        draft_config._frozen = frozen
        transport = "NCCL" if mapping.cp_config.get("use_nccl_for_alltoall", True) else "MNNVL"
        logger.info(
            f"MLA DSpark: latent cache pages sharded over helix cp={mapping.cp_size} "
            f"({transport} all-to-all)"
        )
        return mapping

    # -- shape / buffers ---------------------------------------------------

    def _validate_gqa_shape(self):
        """Replace the base's GQA precondition with the MLA one."""
        layers = getattr(self.model, "layers", None)
        if not layers:
            return
        attn0 = layers[0].self_attn
        keys = (
            "num_heads_tp",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "kv_lora_rank",
            "v_head_dim",
        )
        base = tuple(getattr(attn0, k) for k in keys)
        for idx, layer in enumerate(layers[1:], start=1):
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "kv_a_proj_with_mqa"):
                raise ValueError(
                    f"MLA DSpark block decode needs self_attn.kv_a_proj_with_mqa on every "
                    f"draft layer, but layer {idx} of {type(self.config).__name__} has none."
                )
            if tuple(getattr(attn, k) for k in keys) != base:
                raise ValueError(
                    "MLA DSpark fuses the latent projection across layers and needs one "
                    f"uniform MLA shape, but layer {idx} differs from layer 0 "
                    f"({dict(zip(keys, base))})."
                )

    def _init_rope(self):
        """No shared RoPE cache: the MLA path rotates only the rope slice."""
        self._rope_initialized = True

    @staticmethod
    @lru_cache(maxsize=1)
    def _mla_decode_op():
        """flashinfer's absorbed-MLA paged decode, or None when unavailable.

        Loaded lazily so a build without flashinfer still runs the eager path.
        """
        try:
            import flashinfer

            return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla
        except (ImportError, AttributeError):
            return None

    @classmethod
    def _attention_backend_unavailability_reason(cls, backend: str) -> Optional[str]:
        """Probe THIS family's kernels, not the base's GQA op set.

        ``_uses_worker_attention_backend`` is False here, so neither worker op
        set is ever loaded and the base's trtllm-gen probe would answer about
        kernels this drafter does not call.
        """
        if backend == "TRTLLM":
            if cls._mla_decode_op() is None:
                return "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla is not importable"
            return None
        if backend == "CUTEDSL":
            return cute_dsl_mla_decode_unavailability_reason()
        return None

    def _mla_block_decode_variant(self, paged: bool) -> str:
        """Which of the three block-decode implementations this step runs.

        ``paged`` is False on the private arena, where there are no page tables
        for either kernel, so both kernel backends degrade to eager there.
        """
        if not paged or self.dflash_attention_backend == "VANILLA":
            return "eager"
        if self.dflash_attention_backend == "CUTEDSL":
            return "cute_dsl"
        return "trtllm_gen" if self._mla_decode_op() is not None else "eager"

    def _mla_decode_workspace(self, device, min_bytes: int = 0):
        want = max(_MLA_DECODE_WORKSPACE_BYTES, int(min_bytes))
        if self._mla_workspace is None or self._mla_workspace.numel() < want:
            self._mla_workspace = torch.zeros(want, dtype=torch.uint8, device=device)
        return self._mla_workspace

    def _mla_fp8_decode_scales(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """The FP8 decode's device scales, the target FP8 MLA's at unit q / kv scales.

        ``bmm1 = softmax_scale * dq_q * dq_kv`` and ``bmm2 = dq_kv`` (mlaKernels.cu),
        with both dequant scales ``DRAFT_FP8_KV_SCALE``. Built on first use, which
        the eager warmup reaches before any graph capture.
        """
        if self._mla_fp8_scales is None or self._mla_fp8_scales[0].device != device:
            dq = DRAFT_FP8_KV_SCALE
            self._mla_fp8_scales = (
                torch.full((1,), self.softmax_scale * dq * dq, dtype=torch.float32, device=device),
                torch.full((1,), dq, dtype=torch.float32, device=device),
            )
        return self._mla_fp8_scales

    def _assert_cute_dsl_can_implement(self, batch, block_size, page_size, dtype) -> None:
        """Reject a shape the kernel cannot serve, here, naming the reason.

        Without this the rejection is silent until it is not: ``get_valid_tactics``
        returns [] (cute_dsl_custom_ops.py:8508), the AutoTuner falls back to its -1
        sentinel, and ``forward`` compiles an unvalidated config -- so a draft pool
        with ``tokens_per_block=256`` (``128 % 256 != 0``, mla_decode_fp16.py:3798)
        surfaces as a CUTLASS error out of the fifth drafter layer.

        Page size is the only operand here that is not fixed by the checkpoint,
        and it comes from the draft KV pool, so this cannot move to __init__.
        ``dtype`` is the kernel input dtype; an FP8 one writes a BF16 output.
        """
        key = (batch, block_size, page_size, dtype)
        if self._cute_dsl_checked_shape == key:
            return
        import cutlass

        from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
            CuteDSLNVMlaDecodeBlackwellRunner,
        )

        if dtype == torch.float8_e4m3fn:
            in_dtype, out_dtype = cutlass.Float8E4M3FN, cutlass.BFloat16
        else:
            in_dtype = cutlass.Float16 if dtype == torch.float16 else cutlass.BFloat16
            out_dtype = in_dtype
        runner = CuteDSLNVMlaDecodeBlackwellRunner
        # Mirror the runner's own probe (cute_dsl_custom_ops.py:8542) through its
        # own constants, so this cannot drift from the tactic it will pick. Its
        # candidate tiler list has one entry; split_kv=1 is always in range.
        ok = runner._KERNEL_CLASS_BY_DTYPE[in_dtype].can_implement(
            batch,
            block_size,
            page_size,
            self._num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            in_dtype,
            out_dtype,
            cutlass.Float32,
            cutlass.Float32,
            (128, 128),
            (128, 256),
            1,
            False,
            runner._IS_VAR_SEQ,
            runner._IS_VAR_SPLIT_KV,
            page_size,
        )
        if not ok:
            raise ValueError(
                f"attention_backend='CUTEDSL' cannot serve this shape: batch={batch}, "
                f"block_size={block_size}, page_size={page_size}, "
                f"num_heads={self._num_heads}, dtype={dtype}. Use "
                f"attention_backend='TRTLLM'."
            )
        self._cute_dsl_checked_shape = key

    def _mla_split_kv_plan(
        self, fixup, block_size, fp8, stream: Optional[torch.cuda.Stream] = None
    ) -> Optional[torch.Tensor]:
        """Per-request split-KV counts for the block decode of ``fixup``'s step.

        The decode kernel's grid holds the uniform split of its tactic, sized
        from the batch alone; the plan lets each request use as many of those
        slots as its KV length is worth (cute_dsl_mla.launch_mla_split_kv_plan,
        the target MLA decode's planner, at this decode's piece cost). Every
        layer of a step decodes the same lengths, so the first layer launches
        the plan and the others reuse it; a new fixup -- the next step, or a
        CUDA graph capture -- plans again. None (the uniform split) for a
        batch beyond the planner's request capacity, or a piece cost the
        build's planner cannot take (``_mla_split_plan_kwargs``). ``stream``:
        launch the plan there (the output is allocated on the current stream;
        the caller joins ``stream`` before a decode reads it).
        """
        cached = self._mla_split_plan
        if cached is not None and cached[0] is fixup and cached[1] == fp8:
            return cached[2]
        from tensorrt_llm._torch.attention.backends.fmha.cute_dsl_mla import (
            _SPLIT_PLAN_MAX_REQUESTS,
            launch_mla_split_kv_plan,
        )

        seq_lens = fixup.seq_lens_i32
        batch = seq_lens.shape[0]
        plan_kwargs = _mla_split_plan_kwargs(
            _MLA_SPLIT_PLAN_PIECE_COST_TILES_FP8 if fp8 else _MLA_SPLIT_PLAN_PIECE_COST_TILES
        )
        plan = None
        if plan_kwargs is not None and batch <= _SPLIT_PLAN_MAX_REQUESTS:
            rows, wave_clusters = _mla_split_plan_geometry(self._num_heads, block_size)
            plan = torch.empty(batch, dtype=torch.int32, device=seq_lens.device)
            with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                launch_mla_split_kv_plan(
                    seq_lens,
                    plan,
                    num_requests=batch,
                    rows_per_request=rows,
                    wave_clusters=wave_clusters,
                    **dict(plan_kwargs),
                )
        self._mla_split_plan = (fixup, fp8, plan)
        return plan

    def _cute_dsl_block_decode(self, query, layer_cache, fixup, block_size, kernel_query=None):
        """One cute-dsl MLA decode over context AND block, no fixup.

        The trtllm-gen path splits the work: its q_len semantics are causal, so
        query i cannot see block positions j > i, and the strict upper triangle is
        recovered eagerly and merged by LSE. The cute-dsl kernel takes ``kv_bounds``,
        a per-query-token KV length REPLACING the implicit causal bound
        (mla_decode_fp16.py:400-407), so one pass sees all of it and the fixup goes.

        No fallback: what the kernel cannot serve raises. Build-level
        availability is checked at construction through
        ``cute_dsl_mla_decode_unavailability_reason``; the shape depends on the
        draft pool's page size, which is not known until here, so it is checked
        on the first call.

        An FP8 pool (``context_kv_dtype="fp8"``) is read by the FP8 decode with
        the query quantized the same way (``to_draft_fp8``); the output stays in
        the query's dtype.

        Each request splits its KV as the step's plan says
        (``_mla_split_kv_plan``); for a given split the result is the uniform
        launch's bit for bit.

        ``kernel_query``: the FP8 query of an FP8 pool when the caller built it
        (``draft_qhead``); ``query`` then only gives the shapes.
        """
        import cutlass

        from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
            CuteDSLNVMlaDecodeBlackwellRunner,
        )

        batch, _, num_heads, _ = query.shape
        device = query.device
        pool = layer_cache[:, 0, 0]  # [pages, page_size, head_dim]
        page_size = pool.shape[1]
        d_latent = self.kv_lora_rank
        fp8 = pool.dtype == torch.float8_e4m3fn
        self._assert_cute_dsl_can_implement(
            batch, block_size, page_size, pool.dtype if fp8 else query.dtype
        )

        # stride[1] == 1 is the kernel's only layout demand (mla_decode_fp16.py:465-470),
        # kept by a 576-wide row sliced at 512. detach() because from_dlpack refuses
        # grad-requiring tensors, which the pool write adds outside inference_mode.
        if kernel_query is not None:
            assert fp8 and kernel_query.dtype == torch.float8_e4m3fn
        elif not fp8:
            kernel_query = query.detach()
        elif draft_helix_store_enabled() and query.is_contiguous():
            kernel_query = to_draft_fp8_1k(query.detach())
        else:
            kernel_query = to_draft_fp8(query.detach())
        q_latent = kernel_query[..., :d_latent].permute(2, 3, 1, 0)
        q_rope = kernel_query[..., d_latent:].permute(2, 3, 1, 0)
        c_latent = pool.detach()[..., :d_latent].permute(1, 2, 0)
        c_rope = pool.detach()[..., d_latent:].permute(1, 2, 0)

        out = torch.empty(batch, block_size, num_heads, d_latent, dtype=query.dtype, device=device)
        # Helix: (max, sum) per query token and head for _helix_post_process.
        stats = (
            None
            if fixup.owned is None
            else torch.empty(batch * block_size, num_heads, 2, dtype=torch.float32, device=device)
        )
        workspace_numel = CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
            num_heads, block_size, d_latent, batch, cutlass.Float32
        )
        workspace = self._mla_decode_workspace(device, workspace_numel)
        if fp8:
            decode = torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell
            softmax_scale, output_scale = self._mla_fp8_decode_scales(device)
        else:
            decode = torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell
            softmax_scale, output_scale = self.softmax_scale, 1.0
        decode(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            # .t() WITHOUT .contiguous(): the kernel wants (max_blocks, B) with dim 0
            # stride-1. contiguous() gives stride (B, 1) and it raises "Expected
            # strides[leading_dim] == 1, but got B" for B > 1.
            fixup.page_tables_i32.t(),
            fixup.seq_lens_i32,
            out.permute(2, 3, 1, 0),
            workspace,
            num_heads,
            block_size,
            page_size,
            softmax_scale,
            output_scale,
            batch,
            stats,
            # Every block token sees the whole block, not just its causal prefix.
            fixup.kv_bounds_i32,
            self._mla_split_kv_plan(fixup, block_size, fp8),
        )
        return out if stats is None else (out, stats)

    def _mla_paged_attention(self, query, blk_kv, layer_cache, fixup, block_size, max_kv_len):
        """Absorbed-MLA block attention: paged kernel + a block-local fixup.

        The kernel is causal across the query block and takes no mask (measured,
        not assumed: it matches a causal reference to 1.6e-4 against 1.5e-2 for
        full visibility). Its LSE is base 2 (also measured: exact against
        ``log2(sum exp)``, 1.84 off against the natural log).

        The block's own latents go into the pool first, at ``ctx_len + j`` (see
        _MLABlockFixup). With ``seq_lens = ctx_len + block_size`` the kernel covers
        context AND block, and in-block causality leaves only the strict upper
        triangle -- 8 keys, attended eagerly against ``blk_kv`` and merged by
        log-sum-exp. ``dflash_allocated_ctx_limit`` reserves the room.

        Returns the latent-space output ``[B, block, heads, kv_lora_rank]``.
        """
        decode = self._mla_decode_op()
        batch, _, num_heads, _ = query.shape
        device = query.device

        # --- the block's own latents into the pool, then one causal pass ---
        # Must precede BOTH decode paths: each reads the block back out of the
        # pool, so writing it afterwards would attend uninitialised slots.
        pool = layer_cache[:, 0, 0]  # [pages, page_size, head_dim]
        pages, offsets = fixup.pages.reshape(-1), fixup.offsets.reshape(-1)
        blk_rows = blk_kv.reshape(-1, blk_kv.shape[-1])
        fused_store = fixup.owned is None and spec_tail_fusion_enabled() and blk_kv.is_contiguous()
        if pool.dtype == torch.float8_e4m3fn and not fused_store:
            # Stored like the context latents; byte views because index_put
            # support for FP8 varies across builds.
            pool, blk_rows = pool.view(torch.uint8), to_draft_fp8(blk_rows).view(torch.uint8)
        if fixup.owned is not None:
            # Helix: positions on other ranks' pages rewrite their local slot.
            pool[pages, offsets] = torch.where(
                fixup.owned.reshape(-1, 1), blk_rows, pool[pages, offsets]
            )
        elif fused_store:
            scatter_rows_to_pool(blk_kv.view(-1, blk_kv.shape[-1]), pool, pages, offsets)
        else:
            pool[pages, offsets] = blk_rows

        if self._mla_block_decode_variant(paged=True) == "cute_dsl":
            return self._cute_dsl_block_decode(query, layer_cache, fixup, block_size)
        if fixup.owned is not None:
            raise NotImplementedError("Helix-sharded MLA DSpark context needs the CUTEDSL decode.")
        if layer_cache.dtype == torch.float8_e4m3fn:
            raise NotImplementedError("An FP8 MLA DSpark context needs the CUTEDSL decode.")
        out_lower = torch.empty(
            batch, block_size, num_heads, self.kv_lora_rank, dtype=query.dtype, device=device
        )
        lse_lower = torch.empty(batch * block_size, num_heads, dtype=torch.float32, device=device)
        decode(
            query,
            layer_cache[:, 0],  # [pages, 1, page, head_dim]
            self._mla_decode_workspace(device),
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            fixup.page_tables_i32,
            fixup.seq_lens_i32,
            max_kv_len,
            0,
            out_lower,
            self.softmax_scale,
            1.0,
            None,
            None,
            None,
            backend="trtllm-gen",
            is_var_seq=True,
            uses_shared_paged_kv_idx=True,
            lse=lse_lower,
            return_lse=True,
        )

        # --- fixup: the block's strict upper triangle, the only thing the
        # kernel's in-block causality hides once the block is in the pool ---
        scores = torch.einsum("bshc,btc->bsht", query, blk_kv).float() * self.softmax_scale
        scores = scores.masked_fill(~fixup.valid, float("-inf"))
        lse_upper = scores.logsumexp(-1)
        probs = torch.softmax(scores, dim=-1).nan_to_num_().to(blk_kv.dtype)
        out_upper = torch.einsum("bsht,btc->bshc", probs, blk_kv[..., : self.kv_lora_rank])

        # --- log-sum-exp merge (flash-decoding combine), kernel LSE is base 2 ---
        # w_u / (w_l + w_u) == sigmoid(lse_u - lse_l): one sigmoid and one lerp, no
        # explicit max needed. The last query's triangle is empty, so sigmoid is 0.
        lse_l = lse_lower.view(batch, block_size, num_heads).float() * _LN2
        weight_upper = torch.sigmoid(lse_upper - lse_l).unsqueeze(-1)
        merged = torch.lerp(out_lower.float(), out_upper.float(), weight_upper)
        return merged.to(query.dtype)

    def _mla_freqs_cis(self, device):
        """Adjacent-pair YaRN table, cached. See build_dspark_mla_yarn_freqs_cis.

        Sized from the ceiling the worker publishes once it knows the runtime one
        (_runtime_position_ceiling, set in DFlashWorker._lazy_init_ctx_buffers before
        any forward), because that is what ctx_len is clamped to and it exceeds
        model_config.max_seq_len whenever spec decoding is on. The config cap is the
        fallback for direct construction in tests.
        """
        rope = dict(self._mla_rope_params)
        runtime_cap = self._runtime_position_ceiling
        if runtime_cap is not None:
            # Outright, not max(): the published ceiling is already
            # min(runtime, max_position_embeddings) + lookahead, so it is
            # both sufficient and the only thing that keeps the table small
            # when max_seq_len is unset and the config cap is the
            # checkpoint's advertised 1,048,576.
            rope["max_positions"] = int(runtime_cap)
        # Keyed on the cap, not just built once: _lazy_init_ctx_buffers
        # re-publishes the ceiling when the KV-estimation probe manager is
        # swapped for the real one, and drafter forwards do run during
        # estimation. A table built short for a later, larger cap would index
        # out of range on freqs[positions].
        if self._mla_freqs is None or self._mla_freqs_cap != rope["max_positions"]:
            self._mla_freqs = build_dspark_mla_yarn_freqs_cis(
                dim=self.qk_rope_head_dim,
                base=rope["theta"],
                scaling_factor=rope["scaling_factor"],
                original_max_position_embeddings=rope["original_max_positions"],
                max_position_embeddings=rope["max_positions"],
                beta_fast=rope["beta_fast"],
                beta_slow=rope["beta_slow"],
                mscale=rope["mscale"],
                mscale_all_dim=rope["mscale_all_dim"],
                device=device,
            )
            self._mla_freqs_cap = rope["max_positions"]
        return self._mla_freqs

    def _mla_freqs_real(self, device) -> torch.Tensor:
        """``_mla_freqs_cis`` as a ``[positions, rope_dim // 2, 2]`` FP32 (cos, sin) view."""
        freqs = self._mla_freqs_cis(device)
        return torch.view_as_real(freqs).reshape(freqs.shape[0], freqs.shape[-1], 2)

    def _mla_tail_v2_supported(self, device) -> bool:
        """Whether the shapes fit the fused MLA RoPE kernels of ``TLLM_DSPARK_TAIL_V2``.

        False without the CuTe DSL (no ``dspark_mla_block_rope`` /
        ``dspark_mla_ctx_norm_rope``).
        """
        return (
            dspark_mla_block_rope is not None
            and dspark_mla_ctx_norm_rope is not None
            and IS_CUTLASS_DSL_AVAILABLE
            and is_sm_100f()
            and self.kv_lora_rank % 32 == 0
            and (self.qk_rope_head_dim // 2) % 32 == 0
            and getattr(self, "_kv_a_norm_stacked", None) is not None
            and self._kv_a_norm_stacked.dtype == torch.bfloat16
        )

    def _build_fused_kv_buffers(self) -> None:
        """Stack the per-layer latent projection and its RMSNorm weight.

        The MLA analogue of the base's fused K/V GEMM: one ``[L*576, hidden]``
        weight, plus the ``[L, kv_lora_rank]`` layernorm scales applied after.
        """
        if self._fused_kv_weight is not None:
            return
        layers_attn = [layer.self_attn for layer in self.model.layers]
        attn0 = layers_attn[0]
        self._fused_kv_weight = torch.cat(
            [a.kv_a_proj_with_mqa.weight for a in layers_attn], dim=0
        ).contiguous()
        self._fused_kv_bias = None
        self._kv_a_norm_stacked = torch.stack([a.kv_a_layernorm.weight.data for a in layers_attn])
        self._kv_a_norm_eps = attn0.kv_a_layernorm.variance_epsilon
        self._input_ln_eps = None
        self._has_qk_norm = False
        self._use_fused_qk_norm_rope = False

        # Split kv_b_proj into the absorption operands once. K absorbs into the
        # query, V un-absorbs the attention output. Block decode only.
        nh = attn0.num_heads_tp
        for a in layers_attn if not self.context_kv_only else ():
            w = a.kv_b_proj.weight.view(nh, self.qk_nope_head_dim + self.v_head_dim, -1)
            a._k_b_proj = w[:, : self.qk_nope_head_dim].contiguous()
            a._v_b_proj = w[:, self.qk_nope_head_dim :].contiguous()

        logger.debug(
            f"MLA DSpark: fused latent projection built for {self._num_attn_layers} layers "
            f"(weight={tuple(self._fused_kv_weight.shape)}, head_dim={self._head_dim})"
        )

    # -- context cache -----------------------------------------------------

    def precompute_context_kv(self, projected_hidden, positions):
        """Post-norm / post-RoPE MLA latent for ALL drafter layers in one GEMM.

        Returns ``(latent, None)``: ``[N, L, 1, kv_lora_rank + qk_rope_head_dim]``
        and no V half, which is what ``_kv_factor == 1`` means to the worker.
        """
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        n = projected_hidden.shape[0]
        num_layers = self._num_attn_layers
        weight_dtype = self._fused_kv_weight.dtype
        if projected_hidden.dtype != weight_dtype:
            projected_hidden = projected_hidden.to(weight_dtype)

        latent = F.linear(projected_hidden, self._fused_kv_weight)
        latent = latent.view(n, num_layers, self._head_dim)

        # One fused RMSNorm+RoPE per layer, not one eager chain for all of them:
        # the kernel takes a single norm weight per call and the layers do not
        # share one. Same op the block path uses, so the two halves of an
        # attention stay numerically consistent -- which is the point here, not
        # speed (this runs per request, not per decode step).
        freqs = self._mla_freqs_cis(latent.device)[positions.view(-1).long()]
        out = torch.empty_like(latent)
        for layer_idx in range(num_layers):
            # 3D throughout: the eager fallback's rotary indexes [G, s, rd].
            out[:, layer_idx] = _rmsnorm_rope_batched(
                latent[:, layer_idx].unsqueeze(0).contiguous(),
                self._kv_a_norm_stacked[layer_idx],
                self._kv_a_norm_eps,
                self.qk_rope_head_dim,
                freqs.unsqueeze(0),
                norm_dim=self.kv_lora_rank,
            ).squeeze(0)
        return out.unsqueeze(2).contiguous(), None

    def precompute_context_kv_layer_major(self, projected_hidden, positions):
        """``precompute_context_kv``'s latents laid out ``[L, N, head_dim]``.

        Same GEMM, same frequency gather and the same fused RMSNorm+RoPE kernel
        per layer, but the kernel reads its layer's slice of the GEMM output in
        place and writes straight into a layer-major buffer, so no per-layer
        staging copies run. Returns None when the fused kernel is unavailable
        (the caller then uses ``precompute_context_kv``).
        """
        if not (IS_CUTLASS_DSL_AVAILABLE and is_sm_100f()):
            return None
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        n = projected_hidden.shape[0]
        num_layers = self._num_attn_layers
        weight_dtype = self._fused_kv_weight.dtype
        if projected_hidden.dtype != weight_dtype:
            projected_hidden = projected_hidden.to(weight_dtype)
        latent = draft_gemm(projected_hidden, self._fused_kv_weight, "ctx_latent")
        if latent is None:
            latent = F.linear(projected_hidden, self._fused_kv_weight)
        latent = latent.view(n, num_layers, self._head_dim)
        if tail_v2_enabled() and self._mla_tail_v2_supported(latent.device):
            # Every layer in one launch, the RoPE table read by position in the
            # kernel; bitwise the per-layer launches below.
            return dspark_mla_ctx_norm_rope(
                latent,
                self._kv_a_norm_stacked,
                positions.view(-1).long(),
                self._mla_freqs_real(latent.device),
                self._kv_a_norm_eps,
            )
        freqs = self._mla_freqs_cis(latent.device)[positions.view(-1).long()]
        freqs_real = torch.view_as_real(freqs).reshape(-1, freqs.shape[-1], 2)
        if not is_fused_dspark_rmsnorm_rope_supported(
            latent[:, 0],
            self._kv_a_norm_stacked[0],
            freqs_real,
            1,
            self.qk_rope_head_dim,
            self.kv_lora_rank,
        ):
            return None
        compiled = _compile_rmsnorm_rope_strided(
            self._head_dim,
            self.qk_rope_head_dim,
            1,
            self._kv_a_norm_eps,
            True,
            True,
            False,
            self.kv_lora_rank,
        )
        out = torch.empty((num_layers, n, self._head_dim), dtype=latent.dtype, device=latent.device)
        for layer_idx in range(num_layers):
            # Each layer's slice of the GEMM output is read in place.
            compiled(
                latent[:, layer_idx],
                self._kv_a_norm_stacked[layer_idx],
                freqs_real,
                out[layer_idx],
            )
        return out

    # -- block decode ------------------------------------------------------

    def dflash_forward(
        self,
        noise_embedding,
        query_positions,
        num_ctx_per_req,
        ctx_k_cache,
        ctx_v_cache,
        ctx_cache_batch_idx,
        ctx_kv_cache=None,
        ctx_page_table=None,
        block_placement: Optional[DraftBlockPlacement] = None,
        block_fixup: Optional[_MLABlockFixup] = None,
    ):
        """Eager absorbed-MLA block decode over the drafter's latent cache.

        Args mirror the base: ``ctx_k_cache`` is
        ``[pool_slots, L, max_ctx+block_size, 1, kv_cache_head_dim]`` and
        ``ctx_v_cache`` is unused (``None``). Paged context only, either worker
        may hand over the block's placement: ``block_placement`` (main's worker)
        is where the worker already placed the block in the pages;
        ``block_fixup`` (the K3 worker's fused prepare) is the kernel-path fixup
        operands outright, and wins when both are given.
        """
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()

        batch, block_size = noise_embedding.shape[0], noise_embedding.shape[1]
        device = noise_embedding.device
        num_heads = self._num_heads
        latent_dim = self.kv_lora_rank
        # TRTLLM_K3_DRAFT3_GLUE: per-step glue off the critical path (same values).
        glue = draft3_glue_enabled()
        qhead = draft3_qhead_enabled()

        paged = ctx_page_table is not None
        # attention_backend picks the implementation; both kernel backends need
        # page tables, so the eager gather also covers the private arena.
        variant = self._mla_block_decode_variant(paged)
        use_kernel = variant != "eager"
        # The kernel path with a caller-built fixup never reads the slots.
        slots = (
            None if glue and use_kernel and block_fixup is not None else ctx_cache_batch_idx.long()
        )
        # TLLM_DSPARK_TAIL_V2: per layer, one kernel rotates the query's rope
        # half, normalizes and rotates the block latents and stores them in the
        # pool (reading the RoPE table by position), and the cute-dsl decode
        # follows directly. Same values as the separate launches. Its store
        # writes every block position, so a helix-sharded cache, where other
        # ranks' positions must keep their slot's value, takes the separate
        # launches, unless TRTLLM_K3_ULL2_DRAFT_STORE stores the rows through
        # scratch with the rank's ownership mask.
        fused_block_rope = (
            tail_v2_enabled()
            and variant == "cute_dsl"
            and (self._helix_mapping is None or draft_helix_store_enabled())
            and self._mla_tail_v2_supported(device)
        )
        if fused_block_rope:
            blk_freqs = None
            freqs_real = self._mla_freqs_real(device)
            flat_positions = query_positions.reshape(-1).long()
        else:
            # [B, blk, rope/2] complex: one phase per draft position, shared by
            # the query and the block latent of every layer.
            blk_freqs = self._mla_freqs_cis(device)[query_positions.long()]
        # Context pages for this batch, resolved once: rows are batch positions
        # when bound to the manager's per-request block table and slots on the
        # private arena, which is exactly what ctx_cache_batch_idx already is.
        # A caller-built fixup already carries what the kernels read.
        page_tables = (
            None
            if not paged or (use_kernel and block_fixup is not None)
            else ctx_page_table.index_select(0, slots)
        )

        # Valid-key mask, shared by every layer: context slots below this
        # request's length, then the whole block (DSpark is non-causal). Pages
        # past a request's allocation hold another request's data, so the mask
        # is what keeps them out -- not the block table.
        page_size = None if not paged else ctx_kv_cache[0].shape[-2]
        capacity = ctx_k_cache.shape[2] if not paged else ctx_page_table.shape[1] * page_size
        # Kernel path: index and mask operands are layer-invariant, so build
        # them once here rather than five times inside the decode loop.
        helix = self._helix_mapping
        if helix is not None and (variant != "cute_dsl" or block_size > page_size):
            raise NotImplementedError(
                f"Helix-sharded MLA DSpark context needs the CUTEDSL paged decode and "
                f"block_size <= page size; got {variant!r}, {block_size} > {page_size}."
            )
        fp8_pool = paged and ctx_kv_cache[0].dtype == torch.float8_e4m3fn
        if fp8_pool and variant != "cute_dsl":
            raise NotImplementedError(
                f"An FP8 MLA DSpark context needs the CUTEDSL paged decode; got {variant!r}."
            )
        # The fused RoPE kernel stores BF16 rows into every position: for an
        # FP8 pool or a helix-sharded one it writes the block latents to
        # scratch rows, and one scatter stores them (as FP8; under helix only
        # the positions this rank's pages hold).
        block_scratch = fused_block_rope and (fp8_pool or helix is not None)
        if block_scratch:
            num_block_tokens = batch * block_size
            blk_scratch = torch.empty(
                (num_block_tokens, 1, self._head_dim), dtype=torch.bfloat16, device=device
            )
            scratch_rows, scratch_offsets = self._block_scratch_index(
                num_block_tokens, device, cached=glue
            )
        if not use_kernel:
            fixup = None
        elif block_fixup is not None:
            if helix is not None and block_fixup.owned is None:
                raise ValueError("Helix-sharded MLA DSpark needs a rank-local block fixup.")
            fixup = block_fixup
        elif block_placement is not None and helix is None:
            # Main's worker placed the block already; under helix its global
            # placement is not this rank's, so the fixup is rebuilt below.
            fixup = _MLABlockFixup(
                pages=block_placement.pages,
                offsets=block_placement.offsets,
                valid=_block_upper_triangle(block_size, device),
                page_tables_i32=page_tables,
                seq_lens_i32=block_placement.seq_lens,
                kv_bounds_i32=block_placement.kv_bounds,
            )
        else:
            fixup = _build_mla_block_fixup(
                num_ctx_per_req[:batch],
                page_tables.long(),
                block_size,
                page_size,
                None if helix is None else HelixDraftShard(helix.cp_size, helix.cp_rank, page_size),
            )
        logger.info_once(
            "MLA DSpark block decode: "
            + {
                "trtllm_gen": "trtllm-gen paged kernel, block in pool + upper-triangle fixup",
                "cute_dsl": "one cute-dsl pass over context and block, no fixup",
                "eager": f"eager (backend={self.dflash_attention_backend!r}, "
                f"paged={paged}, "
                f"flashinfer={self._mla_decode_op() is not None})",
            }[variant],
            key="mla_dspark_block_decode_variant",
        )
        # Eager path only: dense key mask over the gathered context.
        if not use_kernel:
            if paged:
                page_tables = page_tables.long()
            ctx_valid = torch.arange(capacity, device=device).unsqueeze(0) < num_ctx_per_req[
                :batch
            ].view(-1, 1)
            key_valid = torch.cat(
                [ctx_valid, torch.ones(batch, block_size, dtype=torch.bool, device=device)], dim=1
            )
        # Helix: the tokens this rank holds no key of, layer-invariant. Built
        # once here when a TRTLLM_K3_ULL2_DRAFT_* switch is on, else per layer.
        helix_zero_kv = (
            fixup.kv_bounds_i32 == 0
            if helix is not None
            and fixup is not None
            and (draft_helix_a2a_enabled() or draft_helix_store_enabled())
            else None
        )
        # GLUE: the first decode's split-KV plan on the drafter's side stream, beside
        # layer 0's projections, behind the context store the worker may have put
        # there (draft3_side_launch); joined before layer 0's first pool write.
        plan_stream = self._draft3_side_stream() if self._draft3_side_keep else None
        if glue and use_kernel and variant == "cute_dsl" and fixup is not None:
            plan_stream = self._draft3_side_stream()
            plan_stream.wait_stream(torch.cuda.current_stream())
            self._mla_split_kv_plan(fixup, block_size, fp8_pool, stream=plan_stream)
        hidden_states = noise_embedding
        residual = None
        # TLLM_DSPARK_DRAFT_FUSION: o_proj and down_proj return their un-reduced
        # partials and each reduction is fused with the residual add and the
        # following RMSNorm (post_attention_layernorm, the next layer's
        # input_layernorm, or the final norm), over MNNVL.
        fused_ar = self._draft_fused_all_reduce()
        # TLLM_DSPARK_AR_PUSH: o_proj and down_proj push their partials into the
        # all-reduce and one site kernel finishes it with the residual and norm.
        sites = self._draft_ar_sites(fused_ar)
        # Main's TP fusions (_tp_fusions) unless TLLM_DSPARK_DRAFT_FUSION selects
        # the set above instead: the same reductions fused through o_proj's and
        # down_proj's own all-reduces.
        tp_split = self._tp_split and not draft_fusion_enabled()
        # The previous layer's un-reduced MLP output and, for main's TP fusions,
        # the all-reduce that sums it.
        partial, partial_all_reduce = None, None
        for layer_idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            if partial is None:
                hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            if residual is None:
                # GLUE: the noise embedding is this forward's own buffer, read only
                # here; the pushed sites write new residual tensors.
                residual = hidden_states if glue and sites is not None else hidden_states.clone()
                hs_normed = layer.input_layernorm(hs_flat)
            elif partial is not None:
                res_flat = residual.reshape(-1, residual.shape[-1])
                hs_normed, res_flat = self._reduce_add_norm(
                    fused_ar, sites, partial, partial_all_reduce, res_flat, layer.input_layernorm
                )
                residual = res_flat.reshape(batch, block_size, -1)
            else:
                res_flat = residual.reshape(-1, residual.shape[-1])
                hs_normed, res_flat = layer.input_layernorm(hs_flat, res_flat)
                residual = res_flat.reshape(batch, block_size, -1)

            fused_a = self._draft_fused_a_proj(layer_idx) if fused_ar is not None else None
            if fused_a is not None:
                # q_a and kv_a read the same normed input: one GEMM, split after.
                # Both halves are consumed as row-strided views (q_a_layernorm
                # and the fused RMSNorm+RoPE read strided rows), so no copies.
                q_kv_a = draft_gemm(hs_normed, fused_a.weight, "q_kv_a")
                if q_kv_a is None:
                    q_kv_a = fused_a(hs_normed)
                q_a = q_kv_a[:, : attn.q_lora_rank]
                blk_latent_rows = q_kv_a[:, attn.q_lora_rank :]
            else:
                q_a = attn.q_a_proj(hs_normed)
                blk_latent_rows = None
            # TRTLLM_K3_DRAFT3_QHEAD: q_b of the un-normalized q_a with the norm weight
            # folded in (no q_a_layernorm launch, the GEMM follows the [q_a | kv_a] GEMM
            # directly); draft_qhead applies the inverse RMS and does the rest.
            q_raw = None
            if (
                qhead
                and block_scratch
                and fp8_pool
                and _plain(attn.q_b_proj)
                and type(attn.q_a_layernorm) is RMSNorm
                and not attn.q_a_layernorm.use_gemma
            ):
                folded = self._draft3_folded_q_b(layer_idx)
                if folded is not None:
                    q_raw = draft_gemm(q_a, folded, "q_b", strided=True)
            if q_raw is not None:
                q = q_raw
            else:
                q_a_normed = attn.q_a_layernorm(q_a)
                q = (
                    draft_gemm(q_a_normed, attn.q_b_proj.weight, "q_b")
                    if _plain(attn.q_b_proj)
                    else None
                )
                if q is None:
                    q = attn.q_b_proj(q_a_normed)
            q = q.view(batch, block_size, num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
            if fused_block_rope and q_raw is not None:
                query = torch.empty(
                    (batch, block_size, num_heads, self.kv_lora_rank + self.qk_rope_head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
                if blk_latent_rows is None:
                    blk_latent_rows = attn.kv_a_proj_with_mqa(hs_normed)
                num_tokens = batch * block_size
                if plan_stream is not None:
                    plan_stream = self._draft3_join_side()
                kernel_query = draft_qhead(
                    q_a,
                    blk_latent_rows,
                    q.view(num_tokens, num_heads, q.shape[-1]),
                    attn.q_a_layernorm.variance_epsilon,
                    attn._k_b_proj.to(q.dtype),
                    attn.kv_a_layernorm.weight,
                    attn.kv_a_layernorm.variance_epsilon,
                    flat_positions,
                    freqs_real,
                    ctx_kv_cache[layer_idx][:, 0, 0],
                    fixup.pages.reshape(-1),
                    fixup.offsets.reshape(-1),
                    rows_mask=fixup.owned,
                ).view(query.shape)
                ctx_out = self._cute_dsl_block_decode(
                    query, ctx_kv_cache[layer_idx], fixup, block_size, kernel_query=kernel_query
                )
            elif fused_block_rope:
                query = torch.empty(
                    (batch, block_size, num_heads, self.kv_lora_rank + self.qk_rope_head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
                if blk_latent_rows is None:
                    blk_latent_rows = attn.kv_a_proj_with_mqa(hs_normed)
                num_tokens = batch * block_size
                kv_pool = ctx_kv_cache[layer_idx][:, 0, 0]
                if plan_stream is not None and not block_scratch:
                    plan_stream = self._draft3_join_side()
                dspark_mla_block_rope(
                    q.view(num_tokens, num_heads, q.shape[-1]),
                    query.view(num_tokens, num_heads, query.shape[-1]),
                    blk_latent_rows,
                    attn.kv_a_layernorm.weight,
                    flat_positions,
                    freqs_real,
                    blk_scratch if block_scratch else kv_pool,
                    scratch_rows if block_scratch else fixup.pages.reshape(-1),
                    scratch_offsets if block_scratch else fixup.offsets.reshape(-1),
                    attn.kv_a_layernorm.variance_epsilon,
                )
                if plan_stream is not None:
                    plan_stream = self._draft3_join_side()
                if block_scratch:
                    scatter_rows_to_pool(
                        blk_scratch.view(num_tokens, -1),
                        kv_pool,
                        fixup.pages.reshape(-1),
                        fixup.offsets.reshape(-1),
                        rows_mask=fixup.owned,
                    )
                torch.ops.trtllm.bmm_out(
                    q[..., : self.qk_nope_head_dim]
                    .view(num_tokens, num_heads, self.qk_nope_head_dim)
                    .transpose(0, 1),
                    attn._k_b_proj.to(q.dtype),
                    query[..., : self.kv_lora_rank]
                    .view(num_tokens, num_heads, self.kv_lora_rank)
                    .transpose(0, 1),
                )
                ctx_out = self._cute_dsl_block_decode(
                    query, ctx_kv_cache[layer_idx], fixup, block_size
                )
            else:
                # One buffer rather than cat([absorbed, rope]): q's rope half is a 64-slice
                # of a 192-wide row, so cat fell off CatArrayBatchedCopy onto the generic
                # kernel -- 17.3 us vs 1.6 us, five launches/iter (measured).
                query = torch.empty(
                    (batch, block_size, num_heads, self.kv_lora_rank + self.qk_rope_head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
                # Rotate the trailing rope slice of every head straight into the
                # destination's rope columns. No norm and no weight here --
                # q_a_layernorm already ran. Writing the rotation into `query`
                # instead of back into `q` removes the strided
                # `query[..., kv_lora_rank:] = q_rope` copy that replaced the cat.
                q_rope_into = _mla_rope_into(
                    q, blk_freqs, query, num_heads, self.qk_rope_head_dim, self.kv_lora_rank
                )
                if not q_rope_into:
                    q = _rmsnorm_rope_batched(
                        q,
                        None,
                        0.0,
                        self.qk_rope_head_dim,
                        blk_freqs,
                        num_heads=num_heads,
                        apply_weight=False,
                        apply_rmsnorm=False,
                    )
                    query[..., self.kv_lora_rank :] = q[..., self.qk_nope_head_dim :]
                q_nope = q[..., : self.qk_nope_head_dim]
                # Absorb the nope half into latent space so the block attends the stored
                # latent directly (MQA) instead of expanding per head. view(), never
                # reshape(): both are strided views and reshape would stage a silent copy.
                torch.ops.trtllm.bmm_out(
                    q_nope.view(batch * block_size, num_heads, self.qk_nope_head_dim).transpose(
                        0, 1
                    ),
                    attn._k_b_proj.to(q.dtype),
                    query[..., : self.kv_lora_rank]
                    .view(batch * block_size, num_heads, self.kv_lora_rank)
                    .transpose(0, 1),
                )

                # The block's own latent comes from the draft hidden; the context's
                # from the projected target hidden (precompute_context_kv). That
                # dual-source split is the DFlash contract, not an MLA detail.
                if blk_latent_rows is None:
                    blk_latent_rows = attn.kv_a_proj_with_mqa(hs_normed)
                blk_latent = blk_latent_rows.view(batch, block_size, -1)
                # RMSNorm over the latent, RoPE over k_pe, one kernel. norm_dim keeps
                # k_pe out of the reduction, which is what DeepSeek-style MLA wants
                # and what DSpark's own convention (whole row) does not do.
                blk_kv = _rmsnorm_rope_batched(
                    blk_latent,
                    attn.kv_a_layernorm.weight,
                    attn.kv_a_layernorm.variance_epsilon,
                    self.qk_rope_head_dim,
                    blk_freqs,
                    norm_dim=latent_dim,
                )
                if plan_stream is not None:
                    plan_stream = self._draft3_join_side()
                if use_kernel:
                    ctx_out = self._mla_paged_attention(
                        query,
                        blk_kv,
                        ctx_kv_cache[layer_idx],
                        fixup,
                        block_size,
                        capacity,
                    )
                else:
                    if not paged:
                        ctx = ctx_k_cache[slots, layer_idx, :, 0, :]
                    else:
                        # [pages, kv_factor=1, nkv=1, page, hd] -> gather this
                        # batch's pages and flatten back to a per-request context.
                        ctx = ctx_kv_cache[layer_idx][:, 0, 0][page_tables].reshape(
                            batch, capacity, -1
                        )
                    kv_full = torch.cat([ctx, blk_kv], dim=1)

                    scores = (
                        torch.einsum("bshc,btc->bsht", query, kv_full).float() * self.softmax_scale
                    )
                    scores = scores.masked_fill(~key_valid.view(batch, 1, 1, -1), float("-inf"))
                    probs = torch.softmax(scores, dim=-1).to(kv_full.dtype)
                    ctx_out = torch.einsum("bsht,btc->bshc", probs, kv_full[..., :latent_dim])
            stats = None
            merged = None
            if helix is not None:
                ctx_out, stats = ctx_out
                zero_kv = helix_zero_kv if helix_zero_kv is not None else fixup.kv_bounds_i32 == 0
                # TRTLLM_K3_ULL2_DRAFT_TAIL: V up-projection, exchange and
                # combine in one launch.
                merged = draft_helix_tail(helix, ctx_out, stats, attn._v_b_proj, zero_kv)
            if merged is not None:
                attn_out = merged
            elif spec_tail_fusion_enabled():
                # Same batched GEMM as the einsum, written straight into the
                # [tokens, heads, v] rows o_proj reads, so no reshape copy.
                attn_out = torch.empty(
                    (batch * block_size, num_heads, self.v_head_dim),
                    dtype=ctx_out.dtype,
                    device=ctx_out.device,
                )
                torch.ops.trtllm.bmm_out(
                    ctx_out.reshape(batch * block_size, num_heads, -1).transpose(0, 1),
                    attn._v_b_proj.to(ctx_out.dtype).transpose(1, 2),
                    attn_out.transpose(0, 1),
                )
            else:
                attn_out = torch.einsum("bshc,hvc->bshv", ctx_out, attn._v_b_proj.to(ctx_out.dtype))
            if helix is not None and merged is None:
                # V is linear, so the per-rank partials merge after it: the
                # all-to-all moves v_head_dim (128), not kv_lora_rank (512).
                attn_out = _draft_helix_post_process(
                    attn_out.reshape(batch * block_size, -1),
                    stats,
                    helix,
                    attn.num_heads_tp_cp,
                    self.v_head_dim,
                    zero_kv,
                )

            res_flat = residual.reshape(-1, residual.shape[-1])
            if fused_ar is not None:
                no_reduce = AllReduceParams(enable_allreduce=False)
                site = None
                if sites is not None and _plain(attn.o_proj):
                    site = sites.linear_ar_norm(
                        attn_out.reshape(batch * block_size, -1),
                        attn.o_proj.weight,
                        "o_proj",
                        res_flat,
                        layer.post_attention_layernorm,
                    )
                if site is not None:
                    hidden_out, res_flat = site
                else:
                    hidden_out = attn.o_proj(
                        attn_out.reshape(batch * block_size, -1), all_reduce_params=no_reduce
                    )
                    hidden_out, res_flat = self._fused_ar_residual_norm(
                        fused_ar, hidden_out, res_flat, layer.post_attention_layernorm
                    )
                partial = _draft_mlp(layer.mlp, hidden_out, no_reduce, defer_down=sites is not None)
                hidden_states = partial
            elif tp_split:
                # Main's TP fusions: o_proj and down_proj reduce their outputs;
                # fuse each all-reduce with the residual add and the RMSNorm after it.
                hidden_out = attn.o_proj(
                    attn_out.reshape(batch * block_size, -1),
                    all_reduce_params=AllReduceParams(enable_allreduce=False),
                )
                hidden_out, res_flat = _all_reduce_add_norm(
                    attn.o_proj.all_reduce, hidden_out, res_flat, layer.post_attention_layernorm
                )
                partial = layer.mlp(
                    hidden_out, final_all_reduce_params=AllReduceParams(enable_allreduce=False)
                )
                partial_all_reduce = layer.mlp.down_proj.all_reduce
                hidden_states = partial
            else:
                hidden_out = attn.o_proj(attn_out.reshape(batch * block_size, -1))
                hidden_out, res_flat = layer.post_attention_layernorm(hidden_out, res_flat)
                hidden_out = layer.mlp(hidden_out)
                hidden_states = hidden_out.reshape(batch, block_size, -1)
            residual = res_flat.reshape(batch, block_size, -1)

        res_flat = residual.reshape(-1, residual.shape[-1])
        if partial is not None:
            hidden_states_out, _ = self._reduce_add_norm(
                fused_ar, sites, partial, partial_all_reduce, res_flat, self.model.norm
            )
        else:
            hidden_states_out, _ = self.model.norm(
                hidden_states.reshape(-1, hidden_states.shape[-1]), res_flat
            )
        if plan_stream is not None:
            self._draft3_join_side()
        return hidden_states_out

    def _block_scratch_index(self, num_tokens: int, device, cached: bool):
        """Pages ``arange(num_tokens)`` and offsets ``0`` of the block scratch rows (a
        ``[num_tokens, 1, D]`` pool view). ``cached``: built once per shape outside graph
        capture and reused (they depend on the shape only)."""
        cache = self.__dict__.setdefault("_draft3_scratch_index", {})
        key = (num_tokens, device)
        if cached and key in cache:
            return cache[key]
        rows = torch.arange(num_tokens, device=device)
        index = (rows, torch.zeros_like(rows))
        if cached and not torch.cuda.is_current_stream_capturing():
            cache[key] = index
        return index

    def _draft3_folded_q_b(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Layer ``layer_idx``'s q_b weight with q_a_layernorm's weight folded in
        (``fold_rmsnorm_weight``), built once outside graph capture; None until then."""
        cache = self.__dict__.setdefault("_draft3_qb_folded", {})
        folded = cache.get(layer_idx)
        if folded is None and not torch.cuda.is_current_stream_capturing():
            attn = self.model.layers[layer_idx].self_attn
            folded = cache[layer_idx] = fold_rmsnorm_weight(
                attn.q_b_proj.weight.detach(), attn.q_a_layernorm.weight.detach()
            )
        return folded

    def _draft3_side_stream(self) -> torch.cuda.Stream:
        stream = self.__dict__.get("_draft3_stream")
        if stream is None:
            stream = self.__dict__["_draft3_stream"] = torch.cuda.Stream()
        return stream

    @property
    def _draft3_side_keep(self) -> list:
        return self.__dict__.setdefault("_draft3_keep", [])

    def draft3_side_launch(self, launch: Callable[[], None], keep=()) -> None:
        """Run ``launch()`` on the drafter's side stream after the current stream's work
        (``TRTLLM_K3_DRAFT3_GLUE``). The next ``dflash_forward`` joins the side stream before
        its first draft-pool write; ``keep`` (the tensors the launch reads) stays referenced
        until then, so the caching allocator does not hand their memory to the current stream
        while the side stream may still read it."""
        stream = self._draft3_side_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            launch()
        self._draft3_side_keep.append(tuple(keep))

    def _draft3_join_side(self) -> None:
        """The current stream waits for the side stream; the kept tensors are released."""
        torch.cuda.current_stream().wait_stream(self._draft3_side_stream())
        self._draft3_side_keep.clear()
        return None

    def _draft_fused_all_reduce(self):
        """The all-reduce the block decode fuses with residual + RMSNorm, or None.

        Enabled by ``TLLM_DSPARK_DRAFT_FUSION=1`` for a TP-sharded drafter
        whose o_proj reduces over MNNVL (the fusion the MNNVL all-reduce
        implements).
        """
        if not draft_fusion_enabled():
            return None
        layers = self.model.layers
        all_reduce = getattr(layers[0].self_attn.o_proj, "all_reduce", None) if layers else None
        if all_reduce is None or all_reduce.mnnvl_allreduce is None:
            return None
        return all_reduce

    def _draft_ar_sites(self, fused_ar) -> Optional[DraftArSites]:
        """The pushed all-reduce sites (``TLLM_DSPARK_AR_PUSH=1``) on the fused
        all-reduce, or None.

        The sites push into the Lamport buffers of the all-reduce's MNNVL
        workspace (``comm.lamport.lamport_push_target``); a call a site does not
        take runs the fused all-reduce.
        """
        if fused_ar is None or not draft_ar_push_enabled():
            return None
        sites = self.__dict__.get("_ar_sites")
        if sites is None or sites.all_reduce is not fused_ar:
            sites = self.__dict__["_ar_sites"] = DraftArSites(fused_ar)
        return sites

    def _reduce_partial(self, fused_ar, sites, partial, residual, norm):
        """``norm(all_reduce(partial) + residual)`` and the new residual, where
        ``partial`` is a down projection's output or a ``_PendingDown`` that a
        pushed site projects and reduces itself."""
        if isinstance(partial, _PendingDown):
            site = None
            if sites is not None:
                site = sites.linear_ar_norm(
                    partial.h2, partial.mlp.down_proj.weight, "down", residual, norm
                )
            if site is not None:
                return site
            partial = partial.partial(AllReduceParams(enable_allreduce=False))
        return self._fused_ar_residual_norm(fused_ar, partial, residual, norm)

    def _reduce_add_norm(self, fused_ar, sites, partial, partial_all_reduce, residual, norm):
        """Finish the previous layer's un-reduced MLP output into ``norm``.

        Through the ``TLLM_DSPARK_DRAFT_FUSION`` all-reduce (and its pushed
        sites) when that set runs, else through main's TP fusions, whose
        down_proj all-reduce is ``partial_all_reduce``.
        """
        if fused_ar is not None:
            return self._reduce_partial(fused_ar, sites, partial, residual, norm)
        return _all_reduce_add_norm(partial_all_reduce, partial, residual, norm)

    def _draft_fused_a_proj(self, layer_idx: int):
        """One ``Linear`` computing ``[q_a_proj; kv_a_proj_with_mqa]`` of a layer.

        Both replicated projections read the same normed input, so one GEMM
        replaces two. The separate modules' weights become row views of the
        fused weight, so no memory is added. None when either module is
        quantized or biased.
        """
        cache = self.__dict__.setdefault("_fused_a_projs", {})
        if layer_idx in cache:
            return cache[layer_idx]
        attn = self.model.layers[layer_idx].self_attn
        q_a, kv_a = attn.q_a_proj, attn.kv_a_proj_with_mqa
        fused = None
        if (
            not q_a.has_any_quant
            and not kv_a.has_any_quant
            and q_a.bias is None
            and kv_a.bias is None
            and q_a.weight.dtype == kv_a.weight.dtype
            and not torch.cuda.is_current_stream_capturing()
        ):
            weight = torch.cat([q_a.weight.data, kv_a.weight.data], dim=0).contiguous()
            fused = Linear(
                q_a.in_features,
                weight.shape[0],
                bias=False,
                dtype=weight.dtype,
            ).to(weight.device)
            fused.weight.data = weight
            q_a.weight.data = weight[: q_a.out_features]
            kv_a.weight.data = weight[q_a.out_features :]
        cache[layer_idx] = fused
        return fused

    @staticmethod
    def _fused_ar_residual_norm(all_reduce, partial, residual, norm):
        """``norm(all_reduce(partial) + residual)`` and the new residual, in one kernel."""
        return _all_reduce_add_norm(all_reduce, partial, residual, norm)

    # -- weights -----------------------------------------------------------

    #: Only the head is shared. This drafter ships its own embed_tokens whose
    #: values differ from the target's -- see load_weights_from_target_model --
    #: so a checkpoint missing it must fail rather than run on random weights.
    WEIGHTS_SHARED_WITH_TARGET = ("lm_head",)

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Take the DSpark heads, rename the TorchSpec spellings, then load.

        The TorchSpec drafters name the capture projection ``context_proj`` /
        ``context_norm`` and the final norm ``final_norm``, where the SpecForge
        ones use ``fc`` / ``hidden_norm`` / ``norm``. Everything below that is
        already HF-standard MLA.
        """
        renames = {
            "context_proj.weight": "fc.weight",
            "context_norm.weight": "hidden_norm.weight",
            "final_norm.weight": "norm.weight",
        }
        if self.context_kv_only:
            # Only the tensors _keep_context_kv_modules left a home for.
            weights = {renames.get(k, k): v for k, v in weights.items()}
            weights = {
                k: v
                for k, v in weights.items()
                if k in self.WRAPPER_OWNED_WEIGHTS
                or ".self_attn.kv_a_proj_with_mqa." in k
                or ".self_attn.kv_a_layernorm." in k
            }
            return super().load_weights(weights, weight_mapper=weight_mapper, **kwargs)
        weights, _ = self._take_dspark_head_weights(weights)
        weights = {renames.get(k, k): v for k, v in weights.items()}
        return super().load_weights(weights, weight_mapper=weight_mapper, **kwargs)

    def load_weights_from_target_model(self, target_model: torch.nn.Module) -> None:
        """Share only ``lm_head``; the drafter has its own trained embedding.

        Inferact/Kimi-K3-DSpark ships ``embed_tokens`` whose values differ from
        the target's, so adopting the target's would feed the block decode
        embeddings it was not distilled against -- silently, at some acceptance
        cost. The GQA drafters ship no embedding and keep the base behavior.
        A context-KV-only drafter has no LM head.
        """
        if self.context_kv_only:
            return
        self.draft_model_full.lm_head = target_model.lm_head
        self.lm_head = target_model.lm_head


def build_k3_dspark_draft(model_config, draft_config, lm_head, model) -> nn.Module:
    """Build the Kimi-K3 standalone MLA DSpark drafter.

    The K3 shell routes only the standalone MLA drafter here; every other
    DSpark drafter (embedded DeepSeek-V4, standalone GQA) stays with main's
    registry builder.

    Args:
        model_config: the target engine's ``ModelConfig``.
        draft_config: the drafter's own ``ModelConfig``.
        lm_head: unused; the drafter shares the target's head at weight-load time.
        model: unused; kept for the draft-builder signature.

    Returns:
        The :class:`MLADSparkForCausalLM` drafter.

    Raises:
        ValueError: the draft is embedded in the target checkpoint, or its
            backbone is not MLA-shaped.
    """
    del lm_head, model
    spec_config = model_config.spec_config
    if getattr(spec_config, "draft_is_embedded_in_target", False):
        raise ValueError(
            "build_k3_dspark_draft builds the standalone MLA DSpark drafter only; this draft "
            "is embedded in the target checkpoint (main's DSpark builder serves it)."
        )
    if not is_mla(draft_config.pretrained_config):
        raise ValueError(
            "build_k3_dspark_draft builds the standalone MLA DSpark drafter only; this "
            "drafter's backbone is not MLA-shaped (main's DSpark builder serves it)."
        )
    # DSparkDecodingConfig fields; other speculative configs default to off / "auto".
    context_kv_only = bool(getattr(spec_config, "context_kv_only", False))
    context_kv_dtype = getattr(spec_config, "context_kv_dtype", "auto") or "auto"
    return MLADSparkForCausalLM(
        draft_config,
        dflash_attention_backend=spec_config.attention_backend,
        context_kv_only=context_kv_only,
        context_kv_dtype=context_kv_dtype,
    )


__all__ = [
    "MLADSparkForCausalLM",
    "build_k3_dspark_draft",
    # MLA drafter backbone (a weight container for the block decode).
    "K3DsparkForCausalLM",
    "K3DsparkModel",
    "build_dspark_mla_yarn_rope",
    "build_dspark_mla_yarn_freqs_cis",
    "apply_dspark_mla_rope",
]
