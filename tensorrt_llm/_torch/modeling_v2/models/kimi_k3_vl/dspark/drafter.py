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
"""The Kimi K3 MLA DSpark drafter of the SM107 targets: its weights and its kernel chain.

:class:`K3DSparkDrafter` is a plain ``nn.Module``. The engine and the worker
(:class:`~.worker.K3DSparkWorker`) read its declarations by attribute: the cache contract
(``_num_attn_layers``, ``_num_kv_heads``, ``_head_dim``, ``_kv_factor``, ``_helix_mapping``,
``reads_managed_ctx_pages``, ``reads_helix_sharded_ctx_pages``), the draft KV dtype
(``draft_kv_dtype``, FP8 E4M3) and, on a context server, ``context_kv_only``.

The weights live in ``layers.py`` containers (``Bf16Linear``, ``RMSNormWeight``, ``GatedMlp``,
``EmbeddingTable``, ``TpAllReduce``), loaded by :meth:`K3DSparkDrafter.load_weights`. Each draft
step is the kernel chain :meth:`K3DSparkDrafter.dflash_forward` spells out:

    noise embedding                                     (worker) vocab-sharded lookup + all-reduce
    per layer:
      input RMSNorm (+ residual)                        FlashInfer; under TP the previous layer's
                                                        down all-reduce + residual + RMSNorm
      [q_a | kv_a] GEMM (or q_a, kv_a)                  cuBLAS
      q_a RMSNorm, q_b GEMM                             FlashInfer, cuBLAS
      query RoPE, q_nope @ W_UK -> latent query         cute_dsl_dspark_rope_into, bmm_out
      block latent RMSNorm + RoPE                       k3_dspark_rmsnorm_rope
      block latents into the pool (FP8 E4M3 pool:       scatter (k3_scatter_rows_to_pool with
        saturated at a unit scale)                      spec_tail_fusion), or index_put
      split-KV plan (first layer of a step)             the MLA decode's planner kernel
      latent attention over context + block             cute_dsl_mla_decode_fp8_blackwell
                                                        (FP16 / BF16 pool: ..._fp16_blackwell)
      helix: merge of the rank partials                 NCCL all-to-all + helix_post_process
      V up-projection                                   bmm_out (or einsum)
      o_proj, all-reduce + residual + RMSNorm           pushed site, or cuBLAS + MNNVL
      gate_up GEMM, SwiGLU, down GEMM                   cuBLAS, silu_and_mul
    final RMSNorm (+ the last down all-reduce)

and the context latents of the target hidden states are fc (+ all-reduce with the RMSNorm fused)
-> one GEMM over the stacked ``kv_a`` weights of every layer -> RMSNorm + RoPE
(``k3_dspark_mla_ctx_norm_rope`` with ``tail_v2``, else one ``k3_dspark_rmsnorm_rope_into`` per
layer).

The target chooses the step's kernels with :class:`K3DSparkPath` (its constants, no switches):
``spec_tail_fusion`` (the fused block store and V up-projection, the fused accept / select of the
worker) and ``tail_v2`` (the one-launch context RoPE, the worker's fused rewind). A context
server's drafter (``context_kv_only``) builds only what the context KV reads: ``fc``,
``hidden_norm`` and each layer's ``kv_a_proj_with_mqa`` / ``kv_a_layernorm``.
"""

import dataclasses
import math
from functools import lru_cache
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed import alltoall_helix
from tensorrt_llm._torch.pyexecutor.config_utils import is_mla
from tensorrt_llm._torch.speculative.vocab_parallel import markov_chain_logits
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.functional import (
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
    RopeEmbeddingUtils,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType, Mapping

from ..common.parallel import TensorParallelMode
from ..kernels.k3r_spec.spec_tail_kernels import DraftBlockPlacement
from ..ops.activation.sm107_mlp import GatedMlp, gated_activation, subgroup_mapping
from ..ops.comm.sm107_allreduce import TpAllReduce
from ..ops.embedding.sm107_embedding import EmbeddingTable
from ..ops.gemm.sm107_linear import Bf16Linear, bf16_linear
from ..ops.norm.sm107_norm import RMSNormWeight
from ..ops.spec.k3_scatter_rows_to_pool import k3_scatter_rows_to_pool
from .primitives import DRAFT_FP8_KV_SCALE, DraftArSites, HelixDraftShard, to_draft_fp8


@dataclasses.dataclass(frozen=True)
class K3DSparkPath:
    """The kernels a target's drafter and worker run: constants of the target, not switches.

    The reference recipes: the context server (tp1) runs neither; dep16 runs ``spec_tail_fusion``;
    helix8 runs both.
    """

    # The fused block-latent store and V up-projection of the drafter, and the worker's fused
    # accept, select, block table and generation prepare.
    spec_tail_fusion: bool = False
    # The one-launch context-latent RMSNorm + RoPE of every layer, and the worker's fused rewind
    # and vocab-sharded tail (needs ``spec_tail_fusion``).
    tail_v2: bool = False

    def __post_init__(self):
        if self.tail_v2 and not self.spec_tail_fusion:
            raise ValueError("K3DSparkPath.tail_v2 runs on spec_tail_fusion.")


# Published DSpark drafters spell the head switches four different ways, and a
# reader that knows only one of them degrades silently: the heads are skipped,
# their weights are dropped, and nothing raises. Resolution order matches
# ``TorchLlmArgs.validate_speculative_config`` so the model and the user-visible
# spec_config can never disagree about whether a head is on.
_DSPARK_HEAD_KEY_ALIASES = {
    # RadixArk/Kimi-K3-DSpark ships ``enable_confidence_head`` top-level.
    "use_confidence_head": ("use_confidence_head", "enable_confidence_head"),
}


def resolve_dspark_head_config(pretrained_config, key):
    """Resolve one DSpark head switch across every spelling in the wild.

    Looks in ``dspark_config``, then ``dflash_config``, then the top level as
    ``dspark_<key>`` and as ``<key>``, for each accepted alias of ``key``.
    Returns ``None`` when the drafter declares the switch nowhere.
    """
    dspark_cfg = getattr(pretrained_config, "dspark_config", None) or {}
    dflash_cfg = getattr(pretrained_config, "dflash_config", None) or {}
    for name in _DSPARK_HEAD_KEY_ALIASES.get(key, (key,)):
        for value in (
            dspark_cfg.get(name),
            dflash_cfg.get(name),
            getattr(pretrained_config, f"dspark_{name}", None),
            getattr(pretrained_config, name, None),
        ):
            if value is not None:
                return value
    return None


def get_masked_input_and_mask(
    input_: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vocab_mask = (input_ >= vocab_start_index) & (input_ < vocab_end_index)
    valid_offset = vocab_start_index * vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask.unsqueeze(-1)


def markov_prev_embeddings(prev_tokens: torch.Tensor, markov_w1: torch.Tensor) -> torch.Tensor:
    """``markov_w1[prev_tokens]`` with out-of-vocab anchors masked to zero.

    The anchor is the last accepted token, which on the one-model rejection path
    comes from ``trtllm::chain_speculative_sampling``: it pads non-accepted
    positions with ``-1``. Mask like modules/embedding.py does, so such a
    position contributes no bias instead of tripping a device assert.

    Args:
        prev_tokens: previous token ids (draft vocab), any shape.
        markov_w1: [vocab, rank].
    Returns:
        ``prev_tokens.shape + (rank,)`` in ``markov_w1``'s dtype.
    """
    prev_tokens, invalid = get_masked_input_and_mask(prev_tokens.long(), 0, markov_w1.shape[0])
    return F.embedding(prev_tokens, markov_w1).masked_fill(invalid, 0)


def dspark_markov_step_bias(
    prev_tokens: torch.Tensor, markov_w1: torch.Tensor, markov_w2: torch.Tensor
) -> torch.Tensor:
    """Vanilla Markov head logit bias for one intra-block draft step.

    Reference: DeepSpec ``VanillaMarkov`` (deepspec/modeling/dspark/
    markov_head.py): ``bias = markov_w2(markov_w1(prev_token))`` where
    markov_w1 is nn.Embedding(vocab, rank) and markov_w2 is
    nn.Linear(rank, vocab, bias=False). With both weights stored
    [vocab, rank] this is ``markov_w1[prev] @ markov_w2.T``.

    Args:
        prev_tokens: [B] long, previous token per request (draft vocab).
        markov_w1: [vocab, rank].
        markov_w2: [vocab_or_shard, rank] (rows may be a TP vocab shard).
    Returns:
        [B, vocab_or_shard] bias in the markov weights' dtype.
    """
    return F.linear(markov_prev_embeddings(prev_tokens, markov_w1), markov_w2)


def dspark_markov_chain(
    base_logits: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    step_bias_fn,
    *,
    hidden_states: Optional[torch.Tensor] = None,
    next_token_fn=None,
    cast_bias_to_logits: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The intra-block Markov refinement loop, shared by every DSpark head.

    Reference: DeepSpec ``VanillaMarkov.sample_block_tokens``. For step i,
    ``logits_i += bias(prev_i)`` with ``prev_0`` = the anchor token (the last
    accepted token, block slot 0) and ``prev_{i>0}`` = the token drawn from
    step i-1's *biased* logits. llama.cpp PR #25173 implements the same chain.

    The chain being sequential is load-bearing and is NOT derivable from a
    checkpoint: published DSpark drafters ship only the training-time
    ``apply_block_logits``, which biases the whole block in one teacher-forced
    pass. SGLang ``srt/models/dspark.py:34-64 run_markov_block`` settles it --
    it steps the same way, feeding back the token drawn from the biased logits.

    Args:
        base_logits: [B, K, vocab_or_shard] shared-lm_head logits.
        first_prev_tokens: [B] long, anchor token ids (draft vocab).
        step_bias_fn: ``(prev_tokens [B], step_hidden or None) -> bias``. A
            closure, so a stateful head (RNN) can carry its recurrent state
            across positions without a second loop.
        hidden_states: [B, K, d] fed to ``step_bias_fn`` one position at a
            time; None for the memoryless heads.
        next_token_fn: ``([B, vocab_or_shard]) -> [B]`` token ids in the FULL
            draft vocab; defaults to a plain argmax. Drafters whose draft
            logits are TP vocab-sharded pass a shard-aware argmax here.
        cast_bias_to_logits: cast the bias down to ``base_logits.dtype``
            before adding. The DFlash drafter does; the V4-Pro drafter does
            not (its Markov weights and its logits already agree), and adding
            the cast there would silently narrow its accumulation dtype.
    Returns:
        sampled_tokens [B, K], corrected_logits [B, K, vocab_or_shard].
        Greedy per-position argmax of the corrected logits reproduces the
        reference sampled chain exactly.
    """
    batch_size, block_size = base_logits.shape[:2]
    if block_size == 0:
        empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
        return empty, base_logits
    sampled, corrected = [], []
    prev = first_prev_tokens.long()
    for k in range(block_size):
        step_hidden = None if hidden_states is None else hidden_states[:, k]
        bias = step_bias_fn(prev, step_hidden)
        if cast_bias_to_logits:
            bias = bias.to(base_logits.dtype)
        step_logits = base_logits[:, k] + bias
        corrected.append(step_logits.unsqueeze(1))
        if next_token_fn is None:
            prev = torch.argmax(step_logits, dim=-1)
        else:
            prev = next_token_fn(step_logits).long()
        sampled.append(prev)
    return torch.stack(sampled, dim=1), torch.cat(corrected, dim=1)


k3_dspark_mla_ctx_norm_rope = None
if IS_CUTLASS_DSL_AVAILABLE:
    # Registers torch.ops.trtllm.cute_dsl_dspark_rope_into, the block decode's query RoPE.
    import tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op  # noqa: F401

    # The DSpark RMSNorm + RoPE kernel compiled for row-strided input, and the one-launch
    # context-latent RMSNorm + RoPE of ``tail_v2``.
    from ..ops.spec.k3_dspark_mla_ctx_norm_rope import k3_dspark_mla_ctx_norm_rope
    from ..ops.spec.k3_dspark_rmsnorm_rope import k3_dspark_rmsnorm_rope
    from ..ops.spec.k3_dspark_rmsnorm_rope_into import (
        build_rmsnorm_rope_into,
        k3_dspark_rmsnorm_rope_into,
    )
    from .kernels.dspark_rmsnorm_rope import is_fused_dspark_rmsnorm_rope_supported

# ----------------------------------------------------------------------------
# Shared helpers of the block decode.
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


class _PendingDown(NamedTuple):
    """A drafter MLP whose down projection runs at the all-reduce site that consumes it."""

    h2: torch.Tensor
    mlp: GatedMlp

    def partial(self) -> torch.Tensor:
        """The un-reduced down projection (cuBLAS)."""
        return bf16_linear(self.h2, self.mlp.down_proj.weight)


def _draft_mlp(
    mlp: GatedMlp,
    x: torch.Tensor,
    down_all_reduce_params: Optional[AllReduceParams],
    defer_down: bool = False,
):
    """The drafter MLP on ``x``: ``mlp(x, final_all_reduce_params=down_all_reduce_params)``.

    With ``defer_down`` (the caller reduces the down projection itself at a pushed all-reduce
    site, ``enable_allreduce=False``) the gate_up GEMM and ``silu_and_mul`` run here and the down
    projection is returned unrun, as a :class:`_PendingDown`, for the site that consumes it.
    """
    if (
        defer_down
        and down_all_reduce_params is not None
        and not down_all_reduce_params.enable_allreduce
    ):
        h1 = bf16_linear(x, mlp.gate_up_proj.weight)
        return _PendingDown(gated_activation(h1, "silu"), mlp)
    return mlp(x, final_all_reduce_params=down_all_reduce_params)


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
            return k3_dspark_rmsnorm_rope(
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

    The chain of ``dspark_markov`` driven by its vanilla step bias. ``markov_w2``
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
# Weights only (layers.py containers): nothing here registers with the
# attention backend or the KV cache manager, and the modules have no
# ``forward``. ``K3DSparkDrafter.dflash_forward`` runs the block decode
# over them.
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


def _draft_mappings(mapping: Mapping) -> tuple[Mapping, Optional[Mapping]]:
    """``(dense, helix)`` of the drafter's mapping.

    ``helix`` is the helix CP mapping attention keeps (the latent cache split by pages over the CP
    ranks), None off helix; ``dense`` is the mapping of everything else, the helix CP ranks
    repurposed as TP ranks.
    """
    if mapping.has_cp_helix():
        return mapping.repurpose_helix_cp_to_tp(), mapping
    if mapping.cp_size > 1:
        raise NotImplementedError(
            f"The MLA DSpark drafter does not support context parallelism "
            f"(cp_size={mapping.cp_size})."
        )
    return mapping, None


class K3MLADSparkAttention(nn.Module):
    """The MLA weights of one drafter layer.

    The low-rank ``*_a`` projections are replicated (every head reads the same latent);
    ``q_b_proj`` / ``kv_b_proj`` are column-sharded by head over the attention TP group and
    ``o_proj`` is row-sharded. Under attention DP the attention TP group is one rank, which is
    also why the drafter's latent cache is unsharded. Under helix the latent cache is split by
    pages over the CP ranks, so every rank holds every head (``q_b`` / ``kv_b`` replicated over
    the CP group) and ``o_proj`` is row-parallel over the repurposed (CP folded into TP) group.
    """

    def __init__(
        self,
        config,
        dense: Mapping,
        helix: Optional[Mapping],
        allreduce_strategy: AllReduceStrategy,
    ):
        super().__init__()
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

        mapping = dense if helix is None else helix
        tp_size, dp_size = mapping.tp_size, 1
        if mapping.enable_attention_dp:
            tp_size, dp_size = 1, mapping.tp_size
        elif helix is not None:
            # CP ranks hold identical head projections: fold them like DP.
            dp_size = mapping.cp_size
        if self.num_heads % tp_size != 0:
            raise ValueError(
                f"DSpark MLA drafter has {self.num_heads} heads, "
                f"not divisible by tp_size {tp_size}."
            )
        self.num_heads_tp = self.num_heads // tp_size
        self.num_heads_tp_cp = self.num_heads_tp // mapping.cp_size
        self.mapping = Mapping(
            world_size=mapping.pp_size * dp_size * tp_size,
            tp_size=tp_size,
            pp_size=mapping.pp_size * dp_size,
            rank=mapping.rank,
            gpus_per_node=mapping.gpus_per_node,
            enable_attention_dp=mapping.enable_attention_dp,
        )
        o_mapping = self.mapping if helix is None else dense
        # Row-parallel o_proj reduces only when the heads are actually split.
        reduce_output = not dense.enable_attention_dp and dense.tp_size > 1
        eps = config.rms_norm_eps
        by_head = dict(
            dtype=dtype,
            tp_size=self.mapping.tp_size,
            tp_rank=self.mapping.tp_rank,
            tp_mode=TensorParallelMode.COLUMN,
        )

        self.q_a_proj = Bf16Linear(self.hidden_size, self.q_lora_rank, dtype=dtype)
        self.q_a_layernorm = RMSNormWeight(self.q_lora_rank, eps, dtype)
        self.q_b_proj = Bf16Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, **by_head)
        self.kv_a_proj_with_mqa = Bf16Linear(self.hidden_size, self.kv_cache_head_dim, dtype=dtype)
        self.kv_a_layernorm = RMSNormWeight(self.kv_lora_rank, eps, dtype)
        self.kv_b_proj = Bf16Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            **by_head,
        )
        self.o_proj = Bf16Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            dtype=dtype,
            tp_size=o_mapping.tp_size,
            tp_rank=o_mapping.tp_rank,
            tp_mode=TensorParallelMode.ROW,
            all_reduce=(
                TpAllReduce(o_mapping, dtype, allreduce_strategy) if reduce_output else None
            ),
        )


class K3MLADSparkDecoderLayer(nn.Module):
    """One drafter layer: the MLA weights, the dense SwiGLU MLP and the two RMSNorms."""

    def __init__(
        self,
        config,
        layer_idx: int,
        dense: Mapping,
        helix: Optional[Mapping],
        allreduce_strategy: AllReduceStrategy,
    ):
        super().__init__()
        dtype, hidden, eps = config.torch_dtype, config.hidden_size, config.rms_norm_eps
        self.layer_idx = layer_idx
        self.self_attn = K3MLADSparkAttention(config, dense, helix, allreduce_strategy)
        # Under attention DP every rank runs the whole MLP: TP groups of one rank.
        self.mlp = GatedMlp(
            hidden,
            config.intermediate_size,
            activation="silu",
            dtype=dtype,
            mapping=subgroup_mapping(dense, 1) if dense.enable_attention_dp else dense,
            allreduce_strategy=allreduce_strategy,
        )
        self.input_layernorm = RMSNormWeight(hidden, eps, dtype)
        self.post_attention_layernorm = RMSNormWeight(hidden, eps, dtype)


class K3MLADSparkModel(nn.Module):
    """The drafter's embedding, layers and final norm."""

    def __init__(self, model_config):
        super().__init__()
        config = model_config.pretrained_config
        dense, helix = _draft_mappings(model_config.mapping)
        strategy = model_config.allreduce_strategy
        # The drafter ships its own trained embedding rather than sharing the target's (verified
        # different weights on Inferact/Kimi-K3-DSpark), so it is loaded from the drafter
        # checkpoint. Vocab-parallel over the dense TP group, the lookup all-reduced at the
        # default strategy.
        self.embed_tokens = EmbeddingTable(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=dense,
            vocab_parallel=True,
        )
        self.layers = nn.ModuleList(
            [
                K3MLADSparkDecoderLayer(config, layer_idx, dense, helix, strategy)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNormWeight(config.hidden_size, config.rms_norm_eps, config.torch_dtype)


# What the drafter config names: the draft weight loader picks its checkpoint mapper by it (the
# checkpoint itself says "K3DSparkModel", registered nowhere).
_CHECKPOINT_ARCH = "K3DsparkForCausalLM"


class K3MLADSparkBackbone(nn.Module):
    """The drafter backbone (``K3DSparkDrafter.draft_model_full``): weights only.

    ``K3DSparkDrafter.dflash_forward`` runs the block decode over ``model``. No ``lm_head``:
    the drafter shares the target's (``K3DSparkDrafter.load_weights_from_target_model``).
    """

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self.model = K3MLADSparkModel(model_config)
        self.lm_head = None


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


# Lower bound of the cute-dsl MLA decode's workspace, grown to the runner's padded size.
_MLA_DECODE_WORKSPACE_BYTES = 256 * 1024 * 1024
# Fixed cost of one split-KV piece, in KV tiles, in the per-request split plan
# of the cute-dsl block decode (_launch_mla_split_plan): 64 heads folded two
# tokens per M tile, fitted on R200 sweeps over 7.5K-200K tokens of context,
# batch 1-32, equal and mixed lengths. An FP8 KV tile takes half the time of a
# BF16 one, so the same fixed cost is twice the tiles.
_MLA_SPLIT_PLAN_PIECE_COST_TILES = 30
_MLA_SPLIT_PLAN_PIECE_COST_TILES_FP8 = 60
# The planner kernel is one program over the batch, padded to a power of two of at least one
# warp: each of its candidate cuts is a reduction over the batch.
_MLA_SPLIT_PLAN_MIN_BLOCK = 32
_MLA_SPLIT_PLAN_LANES_PER_WARP = 256


class _MLABlockFixup(NamedTuple):
    """Layer-invariant operands of the MLA block decode.

    The block's own latents are written into the pool at ``ctx_len + j``. Those
    slots belong to this request -- the draft KV manager adds
    ``max(draft_len, max_total_draft_tokens)`` tokens per generation request each
    step (resource_manager.py:1060) -- and the accepted tokens overwrite them next
    step. The decode then reads context and block in one pass, every block token
    bounded at ``ctx_len + block_size`` (``kv_bounds_i32``). Every field depends on
    ctx_len, the page table and the block geometry alone, so it is built once per
    forward.
    """

    pages: torch.Tensor  # [B, block] page holding each block position
    offsets: torch.Tensor  # [B, block] offset of that position inside its page
    valid: torch.Tensor  # [1, block, 1, block] strict upper triangle (unread by the decode)
    page_tables_i32: torch.Tensor  # kernel operands, cast once
    seq_lens_i32: torch.Tensor  # ctx_len + block_size
    kv_bounds_i32: torch.Tensor  # [B * block] seq_lens repeated per query token
    # Helix: [B, block, 1] whether this rank's pages hold each block position;
    # pages/offsets and the lengths above are then rank-local.
    owned: Optional[torch.Tensor] = None


@lru_cache(maxsize=1)
def cute_dsl_mla_decode_unavailability_reason() -> Optional[str]:
    """Return why the cute-dsl MLA decode cannot run here, or None.

    The op itself only reports by raising, and ``attention_backend="CUTEDSL"``
    has to fail at construction rather than five layers into the first draft
    step.

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
    try:
        op = torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell
    except AttributeError:
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
    all_reduce: TpAllReduce, partial: torch.Tensor, residual: torch.Tensor, norm: RMSNormWeight
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


def _launch_mla_split_plan(
    seq_lens: torch.Tensor,
    plan: torch.Tensor,
    *,
    num_heads: int,
    seq_len_q: int,
    piece_cost: int,
) -> None:
    """Write the per-request split counts of the cute-dsl MLA decode at ``piece_cost`` into
    ``plan`` (one launch).

    The CuTe DSL MLA decode's planner kernel (``mla_split_plan._mla_split_kv_plan_kernel``) on
    the decode runner's geometry (:func:`_mla_split_plan_geometry`), launched with a block of the
    batch rounded up to a power of two, at this decode's piece cost.
    """
    import triton

    from ..kernels.k3r_spec import mla_split_plan as mla

    num_requests = seq_lens.shape[0]
    if not 0 < num_requests <= mla._SPLIT_PLAN_MAX_REQUESTS:
        raise ValueError(f"split plan supports 1..{mla._SPLIT_PLAN_MAX_REQUESTS} requests")
    if piece_cost < 0:
        raise ValueError(f"split plan piece cost must be >= 0, got {piece_cost}")
    for name, tensor in (("seq_lens", seq_lens), ("plan", plan)):
        if (
            tensor.dtype != torch.int32
            or tensor.ndim != 1
            or tensor.stride(0) != 1
            or tensor.numel() < num_requests
            or not tensor.is_cuda
        ):
            raise ValueError(f"{name} must be a contiguous CUDA int32 vector of >= {num_requests}")
    rows, wave_clusters = _mla_split_plan_geometry(num_heads, seq_len_q)
    block = max(_MLA_SPLIT_PLAN_MIN_BLOCK, triton.next_power_of_2(num_requests))
    mla._mla_split_kv_plan_kernel[(1,)](
        seq_lens,
        plan,
        num_requests,
        max(rows, 1),
        max(wave_clusters, 1),
        mla._SPLIT_PLAN_MAX_SPLIT,
        KV_TILE=mla._SPLIT_PLAN_KV_TILE,
        MIN_TILES=mla._SPLIT_PLAN_MIN_TILES,
        PIECE_COST=int(piece_cost),
        MAX_PIECES=mla._SPLIT_PLAN_MAX_SPLIT,
        BLOCK=block,
        num_warps=max(1, min(4, block // _MLA_SPLIT_PLAN_LANES_PER_WARP)),
    )


def _mla_split_plan_geometry(num_heads: int, seq_len_q: int) -> tuple[int, int]:
    """``(rows_per_request, wave_clusters)`` of the CuTe DSL MLA decode, for the planner."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    runner = CuteDSLNVMlaDecodeBlackwellRunner
    rows_per_request = runner.get_num_q_tiles(num_heads, seq_len_q)
    wave_clusters = max(runner._get_max_active_blocks() // runner._CLUSTER_SHAPE_MNK[0], 1)
    return rows_per_request, wave_clusters


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
        self.shift_label = True if shift_label is None else bool(shift_label)
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
        calling conventions:

        * float32 ``base_logits`` of the full vocab (the worker's gathered path), or
          this rank's vocab shard with the TP group's ``tp_exchange`` --
          :func:`markov_chain_logits` runs the chain;
        * ``argmax_fn`` returning full-vocab token ids and, for a TP vocab shard, this
          rank's ``vocab_slice`` (to shard the markov_w2 rows identically) -- the eager
          chain runs.
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
    """The capture projection of the K3 drafter: ``hidden_norm(fc(target hidden))``.

    ``fc`` and ``hidden_norm`` are built from the checkpoint by ``load_weights``, ``fc`` with its
    whole weight. Under plain TP (no attention DP) whose group divides the fc input width, every
    rank multiplies its slice of the captured features by its slice of the ``fc`` columns and one
    all-reduce sums the partials, with ``hidden_norm`` fused in over MNNVL.
    """

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project captured target hidden states into the draft hidden space."""
        hidden_states = hidden_states.to(self.fc.weight.dtype)
        if hidden_states.shape[0] > 0:
            projected = self._tp_sharded_project_target_hidden(hidden_states)
            if projected is not None:
                return projected
        norm = self.hidden_norm
        return F.rms_norm(
            bf16_linear(hidden_states, self.fc.weight),
            (norm.weight.shape[0],),
            norm.weight,
            norm.eps,
        )

    def _resolve_fc_shard(self):
        """``(weight slice, begin, end, all_reduce)`` of the TP-sharded fc, or None.

        Resolved once: the shard exists when the drafter runs plain TP (no attention DP) and the
        fc input width splits evenly over it. The TP group is o_proj's all-reduce group: under
        helix the drafter's ``model_config.mapping`` carries the CP split of its context KV
        instead.
        """
        if getattr(self, "_fc_shard", None) is None:
            self._fc_shard = False
            o_proj = self.model.layers[0].self_attn.o_proj
            all_reduce = self._fc_all_reduce if o_proj is None else o_proj.all_reduce
            mapping = None if all_reduce is None else all_reduce.mapping
            in_features = self.fc.weight.shape[1]
            if (
                mapping is not None
                and mapping.tp_size > 1
                and not mapping.enable_attention_dp
                and in_features % mapping.tp_size == 0
            ):
                width = in_features // mapping.tp_size
                begin = mapping.tp_rank * width
                weight = self.fc.weight.data[:, begin : begin + width].contiguous()
                self._fc_shard = (weight, begin, begin + width, all_reduce)
        return self._fc_shard or None

    def _tp_sharded_project_target_hidden(self, hidden_states: torch.Tensor):
        """The TP-sharded fc (``_resolve_fc_shard``) and its all-reduce with ``hidden_norm``, or
        None when the fc is not sharded."""
        shard = self._resolve_fc_shard()
        if shard is None:
            return None
        weight, begin, end, all_reduce = shard
        partial = bf16_linear(hidden_states[:, begin:end], weight)
        norm = self.hidden_norm
        if all_reduce.mnnvl_allreduce is not None:
            return all_reduce(
                partial,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RMS_NORM, norm_weight=norm.weight, eps=norm.eps
                ),
            )
        return F.rms_norm(all_reduce(partial), (norm.weight.shape[0],), norm.weight, norm.eps)


def _latent_cache_helix_mapping(
    mapping: Mapping, helix_mapping: Optional[Mapping]
) -> Optional[Mapping]:
    """The mapping that splits a one-latent-per-token cache by pages over a helix group.

    ``mapping`` is the drafter's: the target's helix group with its CP ranks repurposed
    as TP ranks. ``helix_mapping`` is the target's helix CP mapping, or None. The result
    keeps every head on every rank (tp = 1) and shards the sequence over cp =
    ``mapping.tp_size`` consecutive ranks, the target's CP groups. Returns None where
    that does not apply: no helix, attention DP, pipeline parallelism, a single rank, or
    a draft TP group that does not divide the target's CP (target tp > 1).

    The result's cp_config is ``{cp_type: HELIX}`` alone: the drafter's partials are exchanged
    by the NCCL all-to-all.
    """
    if (
        helix_mapping is None
        or not helix_mapping.has_cp_helix()
        or mapping.enable_attention_dp
        or mapping.pp_size != 1
        or mapping.tp_size <= 1
        or helix_mapping.cp_size % mapping.tp_size != 0
    ):
        return None
    return Mapping(
        world_size=mapping.world_size,
        rank=mapping.rank,
        gpus_per_node=mapping.gpus_per_node,
        cp_size=mapping.tp_size,
        cp_config={"cp_type": CpType.HELIX},
        tp_size=1,
        pp_size=1,
        moe_tp_size=mapping.moe_tp_size,
        moe_ep_size=mapping.moe_ep_size,
        moe_cluster_size=mapping.moe_cluster_size,
        enable_lm_head_tp_in_adp=mapping.enable_lm_head_tp_in_adp,
    )


def _shard_context_kv_by_helix(draft_config, helix_mapping: Optional[Mapping]) -> Optional[Mapping]:
    """Split the latent cache by pages over the whole repurposed helix group.

    One latent per token cannot be split by head, so attention runs at tp = 1 with every head
    on every rank, and the CuTe DSL decode takes the rank-local bounds this needs.
    ``helix_mapping`` is the target's helix CP mapping (None outside Helix); the sharding is
    :func:`_latent_cache_helix_mapping` of it, which replaces ``draft_config.mapping``. In a
    unified draft cache the pages follow the target manager's helix ledger, which the worker
    requires to be this mapping's. Returns the sharding, or None.
    """
    mapping = _latent_cache_helix_mapping(draft_config.mapping, helix_mapping)
    if mapping is None:
        return None
    frozen, draft_config._frozen = draft_config._frozen, False
    draft_config.mapping = mapping
    draft_config._frozen = frozen
    logger.info(
        f"MLA DSpark: latent cache pages sharded over helix cp={mapping.cp_size} (NCCL all-to-all)"
    )
    return mapping


class K3DSparkDrafter(_DSparkHeadMixin, _K3DFlashDrafterMixin, nn.Module):
    """The Kimi K3 DSpark drafter: an MLA backbone from a standalone checkpoint.

    DFlash's contract with the worker -- dual-source KV, non-causal block attention, one
    precomputed context cache -- on its own weights (:class:`K3MLADSparkBackbone`) and its own
    block decode.

    The cache stores one ``kv_lora_rank + qk_rope_head_dim`` latent per token per layer and no V.
    It lives in the draft KV manager's paged pool, whose dtype the drafter declares
    (``draft_kv_dtype``): FP8 E4M3 at a unit per-tensor scale (``to_draft_fp8``), 5 x 576 x 1 B
    = 2880 B/token/rank. The decode keys on the pool it is handed, so a BF16 pool (5760
    B/token/rank) is read by the FP16 / BF16 decode.

    The block decode is one cute-dsl pass over context and block with per-token ``kv_bounds``
    (``attention_backend="CUTEDSL"``, the only one).

    Reference: arXiv 2607.05147; deepseek-ai/DeepSpec.

    Args:
        draft_config: The drafter's ``ModelConfig``. Under helix its mapping is replaced by the
            page sharding of the latent cache (:func:`_shard_context_kv_by_helix`).
        path: The target's :class:`K3DSparkPath`.
        dflash_attention_backend: ``speculative_config.attention_backend``: ``AUTO`` or
            ``CUTEDSL``.
        helix_mapping: The target's helix CP mapping, or None.
        context_kv_only: A context server's drafter: it only computes the context KV
            (``fc``, ``hidden_norm`` and each layer's ``kv_a_proj_with_mqa`` /
            ``kv_a_layernorm``) and never drafts.
    """

    # The block decode is this class's own, so neither shared DFlash op set applies.
    _uses_worker_attention_backend = False
    # One cute-dsl pass over context and block through the per-token kv_bounds of the cute-dsl
    # MLA decode, no fixup. No AUTO degrade: a build without the kernel raises in __init__ with
    # the reason from cute_dsl_mla_decode_unavailability_reason.
    _default_attention_backend = "CUTEDSL"
    _supported_attention_backends = ("CUTEDSL",)
    # The context KV comes out of the manager's pool instead of an arena dense in max_seq_len
    # that free_gpu_memory_fraction never bounds.
    _paged_ctx_cache = True
    _uses_lilicorr = False
    # Positions the worker serves, published by K3DSparkWorker._lazy_init_ctx_buffers before
    # the first forward; None (direct construction) keeps the config-derived cap.
    _runtime_position_ceiling = None

    #: The dtype of the drafter's context KV pool, which the engine allocates the draft layers
    #: with: FP8 E4M3 latents at a unit per-tensor scale.
    draft_kv_dtype = torch.float8_e4m3fn

    #: Tensors the wrapper itself owns, built from the checkpoint in load_weights, and required.
    WRAPPER_OWNED_WEIGHTS = ("fc.weight", "hidden_norm.weight")
    #: Only the head is shared. This drafter ships its own embed_tokens whose values differ from
    #: the target's -- see load_weights_from_target_model -- so a checkpoint missing it must fail
    #: rather than run on random weights.
    WEIGHTS_SHARED_WITH_TARGET = ("lm_head",)

    def __init__(
        self,
        draft_config,
        *,
        path: K3DSparkPath,
        dflash_attention_backend: str = "AUTO",
        helix_mapping: Optional[Mapping] = None,
        context_kv_only: bool = False,
    ):
        super().__init__()
        cfg = draft_config.pretrained_config
        self.path = path
        self.context_kv_only = bool(context_kv_only)
        if dflash_attention_backend == "AUTO":
            dflash_attention_backend = self._default_attention_backend
        if dflash_attention_backend not in self._supported_attention_backends:
            raise ValueError(
                f"{type(self).__name__} attention backend must be one of "
                f"{list(self._supported_attention_backends)}, got {dflash_attention_backend!r}."
            )
        if not self.context_kv_only:
            # A context server never runs the block decode, so a build without its kernel may
            # still serve it.
            reason = cute_dsl_mla_decode_unavailability_reason()
            if reason is not None:
                raise ValueError(
                    f"attention_backend={dflash_attention_backend!r} is unavailable for "
                    f"{type(self).__name__}: {reason}."
                )
        self.dflash_attention_backend = dflash_attention_backend
        # Before the backbone is built: attention keeps the sharding's CP mapping.
        self._helix_mapping = _shard_context_kv_by_helix(draft_config, helix_mapping)
        # The draft weight loader picks its checkpoint mapper by this name (the checkpoint
        # itself says "K3DSparkModel", registered nowhere).
        cfg.architectures = [_CHECKPOINT_ARCH]

        # The drafter's own config: no speculative decoding, no gathered LM head output.
        model_config = dataclasses.replace(
            draft_config, spec_config=None, lm_head_gather_output=False
        )
        # ModelConfig.extra_attrs is init=False, so dataclasses.replace() does not keep it.
        model_config.extra_attrs = draft_config.extra_attrs
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self.draft_model_full = K3MLADSparkBackbone(model_config)
        self.model = self.draft_model_full.model
        self.lm_head = None
        # Set by the target's shell (the target's logits processor).
        self.logits_processor = None

        dflash_config = getattr(cfg, "dflash_config", None) or {}
        self.mask_token_id = dflash_config.get(
            "mask_token_id", getattr(cfg, "mask_token_id", cfg.vocab_size)
        )
        self.target_layer_ids = dflash_config.get("target_layer_ids", None)
        self.block_size = dflash_config.get("block_size", getattr(cfg, "block_size", None))
        # No DFlash 2 convolutions or candidate selector, no LiLiCorr.
        self.is_dflash2 = False
        self.has_block_conv = False
        self.has_candidate_selector = False
        self.lilicorr = None

        # ``fc`` and ``hidden_norm`` are built by load_weights from the checkpoint; the worker
        # tells a loaded drafter by their presence.
        # Lazy-built after weights load (see _build_fused_kv_buffers).
        self._fused_kv_weight = None
        self._fused_kv_bias = None
        self._input_ln_eps = None
        # The TP-sharded fc, resolved on first use.
        self._fc_shard = None
        self._fc_all_reduce = None
        self._init_dspark_heads(cfg)

        self._validate_mla_shape()
        attn0 = self.model.layers[0].self_attn
        self.qk_nope_head_dim = attn0.qk_nope_head_dim
        self.qk_rope_head_dim = attn0.qk_rope_head_dim
        self.kv_lora_rank = attn0.kv_lora_rank
        self.v_head_dim = attn0.v_head_dim
        # The cache contract the worker sizes its arena from. Config-derived, so it is published
        # here rather than waiting for the lazy buffer build: one MLA latent per token per layer,
        # no V half.
        self._num_attn_layers = len(self.model.layers)
        self._num_heads = attn0.num_heads_tp
        self._num_kv_heads = 1
        self._head_dim = attn0.kv_cache_head_dim
        self._kv_factor = 1

        # No slack: the lookahead lives in dflash_position_ceiling, which the worker publishes
        # as _runtime_position_ceiling before any forward and _mla_freqs_cis builds the table
        # from. This config-derived cap is only the fallback for direct construction, where no
        # worker runs and the positions asked for are whatever the caller passes.
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
        if self.context_kv_only:
            self._keep_context_kv_modules()
        logger.info(
            f"K3 DSpark drafter: mask_token_id={self.mask_token_id}, block_size={self.block_size}, "
            f"attention_backend={self.dflash_attention_backend}, "
            f"draft_kv_dtype={self.draft_kv_dtype}, context_kv_only={self.context_kv_only}, "
            f"helix={self._helix_mapping is not None}, {self.path}"
        )

    @property
    def context_kv_fp8(self) -> bool:
        """Whether the context KV is held in FP8 E4M3 (the worker's name for ``draft_kv_dtype``)."""
        return self.draft_kv_dtype == torch.float8_e4m3fn

    def _keep_context_kv_modules(self) -> None:
        """Drop every module the context KV build does not read.

        A context server only stores the drafter's context KV: the target hidden states through
        ``fc`` / ``hidden_norm``, then each layer's ``kv_a_proj_with_mqa`` and ``kv_a_layernorm``
        plus RoPE (``precompute_context_kv``). The embedding, the query / output projections,
        the MLPs, the final norm and the DSpark heads are never read there. Construction runs
        under meta init, so dropped modules never allocate device memory.
        """
        backbone = self.draft_model_full.model
        # The TP group of the sharded fc (``_resolve_fc_shard``) outlives o_proj.
        self._fc_all_reduce = backbone.layers[0].self_attn.o_proj.all_reduce
        backbone.embed_tokens = None
        backbone.norm = None
        for layer in backbone.layers:
            layer.mlp = None
            layer.input_layernorm = None
            layer.post_attention_layernorm = None
            for name in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_b_proj", "o_proj"):
                setattr(layer.self_attn, name, None)

    # -- shape / buffers ---------------------------------------------------

    def _validate_mla_shape(self):
        """One uniform MLA shape over the layers: the latent projection is fused across them."""
        layers = getattr(self.model, "layers", None)
        if not layers:
            raise ValueError("The MLA DSpark drafter needs at least one layer.")
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

    @property
    def reads_helix_sharded_ctx_pages(self) -> bool:
        """The CuTe DSL decode reads rank-local pages and returns softmax stats;
        ``_shard_context_kv_by_helix`` shards the cache for it."""
        return self._helix_mapping is not None

    def reads_managed_ctx_pages(self, tokens_per_block: int) -> bool:
        """The context is paged (``_paged_ctx_cache``); the CuTe DSL decode takes a page size
        that divides 128, above 1."""
        return tokens_per_block > 1 and 128 % tokens_per_block == 0

    def _mla_decode_workspace(self, device, min_bytes: int = 0):
        want = max(_MLA_DECODE_WORKSPACE_BYTES, int(min_bytes))
        if self._mla_workspace is None or self._mla_workspace.numel() < want:
            self._mla_workspace = torch.zeros(want, dtype=torch.uint8, device=device)
        return self._mla_workspace

    def _mla_fp8_decode_scales(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """The FP8 decode's device scales, the target FP8 MLA's at unit q / kv scales.

        ``bmm1 = softmax_scale * dq_q * dq_kv`` and ``bmm2 = dq_kv`` (mlaKernels.cu), with both
        dequant scales ``DRAFT_FP8_KV_SCALE``. Built on first use, which the eager warmup reaches
        before any graph capture.
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
                f"num_heads={self._num_heads}, dtype={dtype}."
            )
        self._cute_dsl_checked_shape = key

    def _mla_split_kv_plan(self, fixup, block_size, fp8: bool) -> Optional[torch.Tensor]:
        """Per-request split-KV counts for the block decode of ``fixup``'s step.

        The decode kernel's grid holds the uniform split of its tactic, sized
        from the batch alone; the plan lets each request use as many of those
        slots as its KV length is worth (:func:`_launch_mla_split_plan`, the
        decode's planner kernel at this decode's piece cost: an FP8 KV tile costs
        half a BF16 one). Every layer of a step decodes the same lengths, so the
        first layer launches the plan and the others reuse it; a new fixup -- the
        next step, or a CUDA graph capture -- plans again. None (the uniform
        split) for a batch beyond the planner's request capacity.
        """
        cached = self._mla_split_plan
        if cached is not None and cached[0] is fixup and cached[1] == fp8:
            return cached[2]
        from ..kernels.k3r_spec.mla_split_plan import _SPLIT_PLAN_MAX_REQUESTS

        seq_lens = fixup.seq_lens_i32
        batch = seq_lens.shape[0]
        plan = None
        if batch <= _SPLIT_PLAN_MAX_REQUESTS:
            plan = torch.empty(batch, dtype=torch.int32, device=seq_lens.device)
            _launch_mla_split_plan(
                seq_lens,
                plan,
                num_heads=self._num_heads,
                seq_len_q=block_size,
                piece_cost=(
                    _MLA_SPLIT_PLAN_PIECE_COST_TILES_FP8
                    if fp8
                    else _MLA_SPLIT_PLAN_PIECE_COST_TILES
                ),
            )
        self._mla_split_plan = (fixup, fp8, plan)
        return plan

    def _cute_dsl_block_decode(self, query, layer_cache, fixup, block_size):
        """One cute-dsl MLA decode over context AND block.

        The kernel takes ``kv_bounds``, a per-query-token KV length REPLACING the
        implicit causal bound (mla_decode_fp16.py:400-407), so one pass sees the
        whole block from every block token.

        No fallback: what the kernel cannot serve raises. Build-level
        availability is checked at construction through
        ``cute_dsl_mla_decode_unavailability_reason``; the shape depends on the
        draft pool's page size, which is not known until here, so it is checked
        on the first call.

        An FP8 pool is read by the FP8 decode with the query quantized the same way
        (``to_draft_fp8``); the output stays in the query's dtype.

        Each request splits its KV as the step's plan says
        (``_mla_split_kv_plan``); for a given split the result is the uniform
        launch's bit for bit. Under helix also returns the softmax stats.
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
        kernel_query = to_draft_fp8(query.detach()) if fp8 else query.detach()
        q_latent = kernel_query[..., :d_latent].permute(2, 3, 1, 0)
        q_rope = kernel_query[..., d_latent:].permute(2, 3, 1, 0)
        c_latent = pool.detach()[..., :d_latent].permute(1, 2, 0)
        c_rope = pool.detach()[..., d_latent:].permute(1, 2, 0)

        out = torch.empty(batch, block_size, num_heads, d_latent, dtype=query.dtype, device=device)
        # Helix: (max, sum) per query token and head for the merge of the rank partials.
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

    def _mla_paged_attention(self, query, blk_kv, layer_cache, fixup, block_size):
        """Store the block's latents into the pool at ``ctx_len + j`` (see _MLABlockFixup),
        then decode context and block in one pass (``_cute_dsl_block_decode``).

        The store precedes the decode, which reads the block back out of the pool. An FP8
        pool stores the latents as the context's are stored (``to_draft_fp8``).
        Returns the latent-space output ``[B, block, heads, kv_lora_rank]`` (and the
        softmax stats under helix).
        """
        pool = layer_cache[:, 0, 0]  # [pages, page_size, head_dim]
        pages, offsets = fixup.pages.reshape(-1), fixup.offsets.reshape(-1)
        blk_rows = blk_kv.reshape(-1, blk_kv.shape[-1])
        fused_store = fixup.owned is None and self.path.spec_tail_fusion and blk_kv.is_contiguous()
        if pool.dtype == torch.float8_e4m3fn and not fused_store:
            # Byte views: index_put support for FP8 varies across builds.
            pool, blk_rows = pool.view(torch.uint8), to_draft_fp8(blk_rows).view(torch.uint8)
        if fixup.owned is not None:
            # Helix: positions on other ranks' pages rewrite their local slot.
            pool[pages, offsets] = torch.where(
                fixup.owned.reshape(-1, 1), blk_rows, pool[pages, offsets]
            )
        elif fused_store:
            # Quantizes into an FP8 pool (to_draft_fp8).
            k3_scatter_rows_to_pool(blk_kv.view(-1, blk_kv.shape[-1]), pool, pages, offsets)
        else:
            pool[pages, offsets] = blk_rows
        return self._cute_dsl_block_decode(query, layer_cache, fixup, block_size)

    def _mla_freqs_cis(self, device):
        """Adjacent-pair YaRN table, cached. See build_dspark_mla_yarn_freqs_cis.

        Sized from the ceiling the worker publishes once it knows the runtime one
        (_runtime_position_ceiling, set in K3DSparkWorker._lazy_init_ctx_buffers before
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
        """Whether the shapes fit the fused MLA RoPE kernels of ``path.tail_v2``.

        False without the CuTe DSL (no ``k3_dspark_mla_ctx_norm_rope``).
        """
        return (
            k3_dspark_mla_ctx_norm_rope is not None
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
        latent = F.linear(projected_hidden, self._fused_kv_weight)
        latent = latent.view(n, num_layers, self._head_dim)
        if self.path.tail_v2 and self._mla_tail_v2_supported(latent.device):
            # Every layer in one launch, the RoPE table read by position in the
            # kernel; bitwise the per-layer launches below.
            return k3_dspark_mla_ctx_norm_rope(
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
        compiled = build_rmsnorm_rope_into(
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
            k3_dspark_rmsnorm_rope_into(
                latent[:, layer_idx],
                self._kv_a_norm_stacked[layer_idx],
                freqs_real,
                out[layer_idx],
                compiled,
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
        """Absorbed-MLA block decode of every drafter layer over the paged latent cache.

        ``ctx_kv_cache`` holds each layer's ``[pages, 1, 1, page, 576]`` latent pool (FP8 E4M3,
        or the noise embedding's dtype) and ``ctx_page_table`` the batch's pages
        (``ctx_k_cache`` / ``ctx_v_cache`` are unused). Either worker may hand over the block's
        placement: ``block_placement`` is where the worker already placed the block in the
        pages; ``block_fixup`` (the K3 worker's fused prepare) is the decode operands outright,
        and wins when both are given.

        Per layer: input RMSNorm (fused with the previous layer's MLP all-reduce under TP),
        ``[q_a | kv_a]`` GEMM (or the two), q_a RMSNorm, q_b GEMM, query RoPE and the
        ``W_UK`` absorption, block latent RMSNorm + RoPE and its store into the pool, the cute-dsl
        decode, the helix merge, the ``W_UV`` up-projection, o_proj with its all-reduce + residual
        + RMSNorm, and the MLP. Then the final RMSNorm.
        """
        del ctx_k_cache, ctx_v_cache
        if self.context_kv_only:
            raise RuntimeError("A context-KV-only drafter does not draft.")
        if ctx_page_table is None:
            raise NotImplementedError(
                "The MLA DSpark block decode reads the draft KV manager's paged latent cache "
                "(_paged_ctx_cache); this context has no page table."
            )
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()

        batch, block_size = noise_embedding.shape[0], noise_embedding.shape[1]
        device = noise_embedding.device
        num_heads = self._num_heads
        latent_dim = self.kv_lora_rank
        slots = ctx_cache_batch_idx.long()
        helix = self._helix_mapping
        # [B, blk, rope/2] complex: one phase per draft position, shared by the query and the
        # block latent of every layer.
        blk_freqs = self._mla_freqs_cis(device)[query_positions.long()]
        # Context pages for this batch, resolved once: rows are batch positions of the
        # manager's per-request block table, which is exactly what ctx_cache_batch_idx is.
        page_tables = None if block_fixup is not None else ctx_page_table.index_select(0, slots)
        page_size = ctx_kv_cache[0].shape[-2]
        if helix is not None and block_size > page_size:
            raise NotImplementedError(
                f"Helix-sharded MLA DSpark context needs block_size <= page size; "
                f"got {block_size} > {page_size}."
            )
        if ctx_kv_cache[0].dtype not in (torch.float8_e4m3fn, noise_embedding.dtype):
            raise ValueError(
                f"The MLA DSpark block decode reads an FP8 E4M3 or {noise_embedding.dtype} "
                f"latent pool, got {ctx_kv_cache[0].dtype}."
            )
        # The decode operands are layer-invariant: built once here.
        if block_fixup is not None:
            if helix is not None and block_fixup.owned is None:
                raise ValueError("Helix-sharded MLA DSpark needs a rank-local block fixup.")
            fixup = block_fixup
        elif block_placement is not None and helix is None:
            # The worker placed the block already; under helix its global placement is not
            # this rank's, so the fixup is rebuilt below.
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
            "MLA DSpark block decode: one cute-dsl pass over context and block, no fixup",
            key="mla_dspark_block_decode_variant",
        )
        hidden_states = noise_embedding
        residual = None
        # A TP-sharded drafter over MNNVL: o_proj and down_proj return their
        # un-reduced partials and each reduction is fused with the residual add
        # and the following RMSNorm (post_attention_layernorm, the next layer's
        # input_layernorm, or the final norm).
        fused_ar = self._draft_fused_all_reduce()
        # o_proj and down_proj push their partials into that all-reduce and one
        # site kernel finishes it with the residual and norm, where a site applies.
        sites = self._draft_ar_sites(fused_ar)
        # The previous layer's un-reduced MLP output.
        partial = None
        for layer_idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            if partial is None:
                hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            if residual is None:
                residual = hidden_states.clone()
                hs_normed = layer.input_layernorm(hs_flat)
            elif partial is not None:
                res_flat = residual.reshape(-1, residual.shape[-1])
                hs_normed, res_flat = self._reduce_partial(
                    fused_ar, sites, partial, res_flat, layer.input_layernorm
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
                q_kv_a = bf16_linear(hs_normed, fused_a)
                q_a = q_kv_a[:, : attn.q_lora_rank]
                blk_latent_rows = q_kv_a[:, attn.q_lora_rank :]
            else:
                q_a = attn.q_a_proj(hs_normed)
                blk_latent_rows = None
            q_a_normed = attn.q_a_layernorm(q_a)
            q = attn.q_b_proj(q_a_normed)
            q = q.view(batch, block_size, num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
            # One buffer rather than cat([absorbed, rope]): q's rope half is a 64-slice
            # of a 192-wide row, so cat fell off CatArrayBatchedCopy onto the generic
            # kernel -- 17.3 us vs 1.6 us, five launches/iter.
            query = torch.empty(
                (batch, block_size, num_heads, self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
            # Rotate the trailing rope slice of every head straight into the
            # destination's rope columns. No norm and no weight here --
            # q_a_layernorm already ran.
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
            bmm_out(
                q_nope.view(batch * block_size, num_heads, self.qk_nope_head_dim).transpose(0, 1),
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
            ctx_out = self._mla_paged_attention(
                query, blk_kv, ctx_kv_cache[layer_idx], fixup, block_size
            )
            stats = None
            if helix is not None:
                ctx_out, stats = ctx_out
            if self.path.spec_tail_fusion:
                # Same batched GEMM as the einsum, written straight into the
                # [tokens, heads, v] rows o_proj reads, so no reshape copy.
                attn_out = torch.empty(
                    (batch * block_size, num_heads, self.v_head_dim),
                    dtype=ctx_out.dtype,
                    device=ctx_out.device,
                )
                bmm_out(
                    ctx_out.reshape(batch * block_size, num_heads, -1).transpose(0, 1),
                    attn._v_b_proj.to(ctx_out.dtype).transpose(1, 2),
                    attn_out.transpose(0, 1),
                )
            else:
                attn_out = torch.einsum("bshc,hvc->bshv", ctx_out, attn._v_b_proj.to(ctx_out.dtype))
            if helix is not None:
                # V is linear, so the per-rank partials merge after it: the
                # all-to-all moves v_head_dim (128), not kv_lora_rank (512).
                attn_out = _helix_post_process(
                    attn_out.reshape(batch * block_size, -1),
                    stats,
                    helix,
                    zero_kv_mask=fixup.kv_bounds_i32 == 0,
                )

            res_flat = residual.reshape(-1, residual.shape[-1])
            if fused_ar is not None:
                no_reduce = AllReduceParams(enable_allreduce=False)
                site = None
                if sites is not None:
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
            else:
                hidden_out = attn.o_proj(attn_out.reshape(batch * block_size, -1))
                hidden_out, res_flat = layer.post_attention_layernorm(hidden_out, res_flat)
                hidden_out = layer.mlp(hidden_out)
                hidden_states = hidden_out.reshape(batch, block_size, -1)
            residual = res_flat.reshape(batch, block_size, -1)

        res_flat = residual.reshape(-1, residual.shape[-1])
        if partial is not None:
            hidden_states_out, _ = self._reduce_partial(
                fused_ar, sites, partial, res_flat, self.model.norm
            )
        else:
            hidden_states_out, _ = self.model.norm(
                hidden_states.reshape(-1, hidden_states.shape[-1]), res_flat
            )
        return hidden_states_out

    def _draft_fused_all_reduce(self) -> Optional[TpAllReduce]:
        """The all-reduce the block decode fuses with residual + RMSNorm, or None.

        For a TP-sharded drafter whose o_proj reduces over MNNVL (the fusion the
        MNNVL all-reduce implements).
        """
        all_reduce = self.model.layers[0].self_attn.o_proj.all_reduce
        if all_reduce is None or all_reduce.mnnvl_allreduce is None:
            return None
        return all_reduce

    def _draft_ar_sites(self, fused_ar) -> Optional[DraftArSites]:
        """The pushed all-reduce sites on the fused all-reduce, or None.

        The sites push into the Lamport buffers of the all-reduce's MNNVL
        workspace (``layers.lamport_push_target``); a call a site does not
        take runs the fused all-reduce.
        """
        if fused_ar is None:
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
            partial = partial.partial()
        return self._fused_ar_residual_norm(fused_ar, partial, residual, norm)

    def _draft_fused_a_proj(self, layer_idx: int) -> Optional[torch.Tensor]:
        """The ``[q_a_proj; kv_a_proj_with_mqa]`` weight of a layer, for one GEMM.

        Both replicated projections read the same normed input, so one GEMM
        replaces two. The separate projections' weights become row views of the
        fused weight, so no memory is added. Built once, outside graph capture
        (None if the first call is under capture).
        """
        cache = self.__dict__.setdefault("_fused_a_weights", {})
        if layer_idx in cache:
            return cache[layer_idx]
        attn = self.model.layers[layer_idx].self_attn
        q_a, kv_a = attn.q_a_proj, attn.kv_a_proj_with_mqa
        fused = None
        if q_a.weight.dtype == kv_a.weight.dtype and not torch.cuda.is_current_stream_capturing():
            fused = torch.cat([q_a.weight.data, kv_a.weight.data], dim=0).contiguous()
            q_a.weight.data = fused[: q_a.out_features]
            kv_a.weight.data = fused[q_a.out_features :]
        cache[layer_idx] = fused
        return fused

    @staticmethod
    def _fused_ar_residual_norm(all_reduce, partial, residual, norm):
        """``norm(all_reduce(partial) + residual)`` and the new residual, in one kernel."""
        return _all_reduce_add_norm(all_reduce, partial, residual, norm)

    # -- weights -----------------------------------------------------------

    #: The checkpoint's TorchSpec spellings of the capture projection and the final norm
    #: (the SpecForge drafters name them ``fc`` / ``hidden_norm`` / ``norm``).
    _TORCHSPEC_RENAMES = {
        "context_proj.weight": "fc.weight",
        "context_norm.weight": "hidden_norm.weight",
        "final_norm.weight": "norm.weight",
    }

    def _backbone_tensors(self) -> Iterator[tuple[str, Callable[..., None], tuple[str, ...]]]:
        """``(target, load, checkpoint names)`` of every backbone tensor, ``load`` taking the
        named full checkpoint tensors in order. A context-KV-only drafter has only each layer's
        ``kv_a_proj_with_mqa`` and ``kv_a_layernorm``."""

        def copy_into(param):
            return lambda tensor: param.data.copy_(tensor)

        backbone = self.model
        if self.context_kv_only:
            for idx, layer in enumerate(backbone.layers):
                prefix = f"layers.{idx}.self_attn."
                attn = layer.self_attn
                yield (
                    f"{prefix}kv_a_proj_with_mqa",
                    attn.kv_a_proj_with_mqa.load,
                    (f"{prefix}kv_a_proj_with_mqa.weight",),
                )
                yield (
                    f"{prefix}kv_a_layernorm",
                    copy_into(attn.kv_a_layernorm.weight),
                    (f"{prefix}kv_a_layernorm.weight",),
                )
            return
        yield "embed_tokens", backbone.embed_tokens.load, ("embed_tokens.weight",)
        yield "norm", copy_into(backbone.norm.weight), ("norm.weight",)
        for idx, layer in enumerate(backbone.layers):
            prefix = f"layers.{idx}."
            attn = layer.self_attn
            for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
                yield (
                    f"{prefix}self_attn.{name}",
                    getattr(attn, name).load,
                    (f"{prefix}self_attn.{name}.weight",),
                )
            for name in ("q_a_layernorm", "kv_a_layernorm"):
                yield (
                    f"{prefix}self_attn.{name}",
                    copy_into(getattr(attn, name).weight),
                    (f"{prefix}self_attn.{name}.weight",),
                )
            yield (
                f"{prefix}mlp",
                layer.mlp.load,
                tuple(f"{prefix}mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")),
            )
            for name in ("input_layernorm", "post_attention_layernorm"):
                yield (
                    f"{prefix}{name}",
                    copy_into(getattr(layer, name).weight),
                    (f"{prefix}{name}.weight",),
                )

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Load the drafter checkpoint into the K3-owned weights, each tensor by its name.

        The DSpark heads go to :meth:`_take_dspark_head_weights`; ``fc`` / ``hidden_norm`` (or
        their TorchSpec names ``context_proj`` / ``context_norm``) build the capture projection;
        every backbone tensor (HF MLA names, with or without the ``model.`` prefix; the final
        norm also as ``final_norm``) is copied into this rank's shard of its weight. A checkpoint
        missing any of them raises. A context-KV-only drafter takes ``fc``, ``hidden_norm`` and
        each layer's ``kv_a_proj_with_mqa`` / ``kv_a_layernorm`` and leaves the rest of the
        checkpoint. ``weight_mapper`` is not used.
        """
        del weight_mapper, kwargs
        if not self.context_kv_only:
            weights, _ = self._take_dspark_head_weights(weights)
        weights = {self._TORCHSPEC_RENAMES.get(k, k): v for k, v in weights.items()}
        weights = {
            k[len("model.") :] if k.startswith("model.") else k: v for k, v in weights.items()
        }

        # Without fc the drafter has no capture projection and drafts from an empty
        # context forever.
        wrapper_missing = [k for k in self.WRAPPER_OWNED_WEIGHTS if k not in weights]
        if wrapper_missing:
            raise ValueError(
                f"{type(self).__name__}: checkpoint is missing {wrapper_missing}, "
                "which this wrapper owns and builds from the checkpoint."
            )
        tensors = list(self._backbone_tensors())
        missing = sorted(
            target for target, _, names in tensors if any(n not in weights for n in names)
        )
        if missing:
            raise ValueError(
                f"{type(self).__name__}: checkpoint provides no weights for "
                f"{missing[:8]}{' ...' if len(missing) > 8 else ''}. These are "
                f"not in WEIGHTS_SHARED_WITH_TARGET "
                f"({', '.join(self.WEIGHTS_SHARED_WITH_TARGET)}), so loading "
                "would leave them uninitialized."
            )

        fc = weights["fc.weight"]
        self.fc = Bf16Linear(fc.shape[1], fc.shape[0], dtype=fc.dtype, device="cuda")
        self.fc.load(fc)
        norm = weights["hidden_norm.weight"]
        self.hidden_norm = RMSNormWeight(
            norm.shape[0], getattr(self.config, "rms_norm_eps", 1e-6), norm.dtype, device="cuda"
        )
        self.hidden_norm.weight.data.copy_(norm)
        used = set(self.WRAPPER_OWNED_WEIGHTS)
        for _, load, names in tensors:
            load(*(weights[n] for n in names))
            used.update(names)
        unused = sorted(set(weights) - used)
        if unused and not self.context_kv_only:
            logger.warning(
                f"{type(self).__name__}: checkpoint tensors not loaded: "
                f"{unused[:8]}{' ...' if len(unused) > 8 else ''}"
            )

    def load_weights_from_target_model(self, target_model: torch.nn.Module) -> None:
        """Share only ``lm_head``; the drafter has its own trained embedding.

        Inferact/Kimi-K3-DSpark ships ``embed_tokens`` whose values differ from
        the target's, so adopting the target's would feed the block decode
        embeddings it was not distilled against -- silently, at some acceptance
        cost. A context-KV-only drafter has no LM head.
        """
        if self.context_kv_only:
            return
        self.draft_model_full.lm_head = target_model.lm_head
        self.lm_head = target_model.lm_head


def _helix_sanitize_empty_kv(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    zero_kv_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Force zero-local-KV rows to a no-op contribution for the Helix combine.

    A CP rank that owns no KV blocks for a token attends to zero keys, so the
    attention kernel normalizes by a zero softmax sum and yields NaN. The combine
    weights each rank by sum * exp(max - global_max), so such rows must contribute
    softmax_stats of (-inf, 0) and a zeroed partial_o to act as a no-op. The rows
    are selected by zero_kv_mask, which is robust regardless of what the kernel
    wrote. Passing None disables sanitization.

    Args:
        partial_o: Partial attention output, shape [num_tokens, ...].
        softmax_stats: Per (token, head) (max, sum), shape [num_tokens, num_heads, 2].
        zero_kv_mask: Bool tensor of shape [num_tokens], True where this rank has
            zero local KV.
    """
    if zero_kv_mask is None:
        return partial_o, softmax_stats
    num_tokens = partial_o.shape[0]
    mask = zero_kv_mask[:num_tokens].view(-1, 1)
    # masked_fill overwrites masked rows regardless of their current (possibly NaN)
    # value and is CUDA-graph safe due to static shapes.
    partial_o = partial_o.masked_fill(mask, 0.0)
    sm_max = softmax_stats[..., 0].masked_fill(mask, float("-inf"))
    sm_sum = softmax_stats[..., 1].masked_fill(mask, 0.0)
    softmax_stats = torch.stack([sm_max, sm_sum], dim=-1)
    return partial_o, softmax_stats


# Distinct input specializations the NCCL reformat helpers below have been
# asked to compile. ``dynamic=False`` specializes on the input shapes and
# ``cp_size``, so this set tracks the same thing dynamo's per-frame cache does.
_HELIX_NCCL_SHAPES: set = set()
_HELIX_NCCL_LIMIT_REPORTED = False


def _helix_note_nccl_specialization(
    partial_o: torch.Tensor, softmax_stats: torch.Tensor, cp_size: int
) -> None:
    """Warn once if the Helix NCCL reformat is past dynamo's recompile budget.

    Past ``cache_size_limit`` dynamo stops compiling and runs the frame eagerly
    without raising, which silently gives back the fusion these helpers exist
    for. Report it once per process so the regression is visible in the log
    instead of only in a kernel trace.
    """
    global _HELIX_NCCL_LIMIT_REPORTED
    if _HELIX_NCCL_LIMIT_REPORTED:
        return
    key = (tuple(partial_o.shape), tuple(softmax_stats.shape), cp_size)
    if key in _HELIX_NCCL_SHAPES:
        return
    _HELIX_NCCL_SHAPES.add(key)
    try:
        import torch._dynamo as _dynamo

        limit = getattr(_dynamo.config, "cache_size_limit", None)
    except ImportError:
        limit = None
    if limit is None or len(_HELIX_NCCL_SHAPES) <= limit:
        return
    _HELIX_NCCL_LIMIT_REPORTED = True
    logger.warning(
        "Helix NCCL all-to-all reformat has seen %d distinct input "
        "specializations, above torch._dynamo.config.cache_size_limit=%d. "
        "Dynamo stops recompiling past that limit and runs these helpers "
        "eagerly without raising, losing the sanitize/transpose fusion. "
        "Reduce the number of CUDA-graph batch buckets or raise "
        "cache_size_limit.",
        len(_HELIX_NCCL_SHAPES),
        limit,
    )


# NCCL needs each all-to-all chunk to be contiguous. Inductor can pad the
# transposed buffers that the chunks are split from, even with
# ``pad_outputs=False``, so turn padding off for this helper. This keeps the
# fused kernels and avoids a copy of each chunk after the compiled call.
@torch.compile(dynamic=False, options={"comprehensive_padding": False})
def _helix_nccl_pre_alltoall(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    zero_kv_mask: Optional[torch.Tensor],
    cp_size: int,
) -> List[torch.Tensor]:
    """Sanitize zero-local-KV rows and reformat into the alltoall send layout.

    ``_helix_sanitize_empty_kv`` and the transpose and split into the send
    layout live in one compiled region so inductor folds the fill into the
    transposed store instead of writing ``partial_o`` and reading it straight
    back. Dynamo inlines the call, so the sanitize has exactly one definition
    and the fusion is unaffected.

    ``dynamic=False`` is deliberate. With dynamic shapes inductor emits a much
    slower transposed store, and picks different grids on different ranks for
    the same shape, which cancels the win at the collective.

    The cost is one specialization per CUDA-graph batch bucket. Exceeding
    dynamo's ``cache_size_limit`` falls back to eager SILENTLY -- the symptom is
    ``triton_poi_fused_*`` disappearing from the trace, not an error, so
    ``_helix_note_nccl_specialization`` counts the distinct shapes at the call
    site and says so once.
    """
    partial_o, softmax_stats = _helix_sanitize_empty_kv(partial_o, softmax_stats, zero_kv_mask)
    chunks = []
    for t in (partial_o, softmax_stats):
        t = t.transpose(1, 0).contiguous()
        chunks.extend(torch.split(t, t.shape[0] // cp_size))
    return chunks


@torch.compile(dynamic=False)
def _helix_nccl_post_alltoall(gathered: List[torch.Tensor]) -> List[torch.Tensor]:
    """Reformat the gathered partials into the helix_post_process layout."""
    return [t.transpose(1, 2).contiguous() for t in gathered]


def _helix_post_process(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    mapping: Mapping,
    zero_kv_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The drafter's helix CP post-processing: the NCCL all-to-all exchange of the partial
    attention outputs and their combine.

    ``partial_o`` is ``[tokens, heads * v_head_dim]`` (the V up-projection runs before the
    exchange); ``zero_kv_mask`` marks the tokens this CP rank owns no KV for, whose rows are
    forced to a no-op contribution in :func:`_helix_nccl_pre_alltoall`, fused with the reformat.
    The reformat helpers are compiled; the drafter meets each batch bucket's shape first in a
    warmup forward.
    """
    if not mapping.cp_config.get("use_nccl_for_alltoall", True):
        raise NotImplementedError("The DSpark drafter exchanges its helix partials over NCCL.")
    _helix_note_nccl_specialization(partial_o, softmax_stats, mapping.cp_size)
    chunks = _helix_nccl_pre_alltoall(partial_o, softmax_stats, zero_kv_mask, mapping.cp_size)
    gathered = alltoall_helix(chunks, mapping.cp_group)
    gathered = _helix_nccl_post_alltoall(gathered)
    return torch.ops.trtllm.helix_post_process(gathered[0], gathered[1], 1.0)


def build_k3_dspark_drafter(
    draft_config,
    spec_config,
    *,
    path: K3DSparkPath,
    helix_mapping: Optional[Mapping] = None,
    context_kv_only: bool = False,
) -> K3DSparkDrafter:
    """The Kimi K3 MLA DSpark drafter of a standalone DSpark checkpoint.

    Args:
        draft_config: The drafter's own ``ModelConfig``.
        spec_config: The target's DSpark ``speculative_config``.
        path: The target's :class:`K3DSparkPath`.
        helix_mapping: The target's helix CP mapping, or None.
        context_kv_only: The target is a context server (see :class:`K3DSparkDrafter`).

    Raises:
        ValueError: the draft is embedded in the target checkpoint, or its backbone is not
            MLA-shaped.
    """
    if getattr(spec_config, "draft_is_embedded_in_target", False):
        raise ValueError(
            "The Kimi K3 DSpark drafter is a standalone checkpoint; this draft is embedded in "
            "the target checkpoint."
        )
    if not is_mla(draft_config.pretrained_config):
        raise ValueError("The Kimi K3 DSpark drafter's backbone is MLA-shaped (kv_lora_rank).")
    return K3DSparkDrafter(
        draft_config,
        path=path,
        dflash_attention_backend=spec_config.attention_backend,
        helix_mapping=helix_mapping,
        context_kv_only=context_kv_only,
    )
