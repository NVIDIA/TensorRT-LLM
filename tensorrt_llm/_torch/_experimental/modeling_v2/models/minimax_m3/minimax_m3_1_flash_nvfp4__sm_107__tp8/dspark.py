# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-model DSpark on this target (phase W0): the in-tree DSpark drafter and worker, unchanged.

The target keeps its own forward; the shell hands the step to ``M3DSparkWorker``, which is the in-tree
``DSparkWorker`` plus the five calls the shell makes on any worker (``capture_hidden_states``,
``arm_capture_norm``, ``tail_eligible``, ``prebuild``, ``__call__``). In W0 none of them takes a fused path: the
captures are the in-tree DFlash captures, the target logits go through the logits processor and the in-tree
strict accept verifies them, so the speculation is the in-tree one by construction.

The drafter is the in-tree standalone DSpark drafter for the checkpoint's architecture, built through the draft
model registry. Its context K/V lives in the separate draft KV cache, which must keep the drafter's projection
dtype (BF16): a pool of another dtype is not bound and the worker falls back to a private arena sized by the
served sequence length. So the drafter never takes the target's KV quantization: neither the FP8 draft-KV
override (an Eagle3 need) nor the target's own KV algo (NVFP4 under ``kv_cache_config.dtype="nvfp4"``).
"""

from __future__ import annotations

import functools
import time
from types import SimpleNamespace
from typing import Tuple

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import m31_capture_fork
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import native_swiglu
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.comm import (
    allgather_cast as _allgather_cast,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    strict_accept as _strict_accept,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_block_tables import (
    enable_parallel_refresh,
    refresh_block_tables,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_ctx_advance import (
    advance_ctx_and_read_view,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_ctx_store import (
    fold_k_norm_weight,
    fused_ctx_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_drafter_inputs import (
    drafter_inputs as _drafter_inputs,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_drafter_inputs import (
    drafter_inputs_and_refresh as _drafter_inputs_and_refresh,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_forward_indices import (
    forward_indices,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_qk_norm_rope import (
    qk_norm_rope,
    qk_norm_rope_append,
)
from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_speculative import (
    external_drafter_config_kwargs,
    get_draft_model,
)
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import (
    _RNG_SLOT_SPAN,
    should_use_separate_draft_kv_cache,
)
from tensorrt_llm.logger import logger

from ..m31_tail_bookkeep import tail_bookkeep_draft_len, tail_bookkeep_log_once

# TP8_FLAGFREE_GATHER (../ops/flagfree_gather.py, a family switch): the flag-free gather for this worker's two per-step
# all-gathers, the verify logits (N3 A1) and the drafter fc output (FOLD_F7); set up in prebuild right after FOLD_F7's
# set-up. Bitwise the stock gathers. Default ON; kill switch TRTLLM_M31_DISABLE_TP8_FLAGFREE_GATHER=1, read at import.
from ..ops import flagfree_gather as _ffg  # noqa: E402
from . import dspark_ctxlen
from .drafter_down_splitk import route_down_splitk, route_oproj_splitk
from .drafter_mxfp8 import route_gu_swiglu, route_his_layout, route_mx_linears
from .dspark_markov_worker import MAX_FAST_BATCH, TAILBK_SPEC, S4WorkerMixin
from .kernels import m3_mc_allreduce
from .m31_switches import m31_disabled, m31_enabled
from .ops.embedding.m3_embedding import m3_embedding
from .ops.gemm.m31_fc_shard import MAX_ROWS as FC_SHARD_MAX_ROWS
from .ops.gemm.m31_fc_shard import FcShard
from .ops.sampling import m31_chain_spec_mc as _chain_spec_mc

# DRAFT_AR_NORM_CLUSTER (opt-in, TRTLLM_M31_ENABLE_DRAFT_AR_NORM_CLUSTER=1): the drafter's fused all-reduces + norms
# (DRAFT_AR_NORM_FUSED) on the target's multicast all-reduce + norm (its 8-token bucket, the drafter-norm form: the
# unrounded fp32 sum normalized, opts bit 16) instead of the stock fused all-reduce; not bitwise, as DRAFT_AR_NORM_FUSED
# (a different summation order). The draft-length-7 batch-1 drafter block (8 rows) takes it; other row counts keep the
# stock fused path. Off by default. Read at import.
S9B_MC = m31_enabled("DRAFT_AR_NORM_CLUSTER")
S9B_MC_MAX_ROWS = 8  # capacity: fixed: the drafter-norm form exists for the 8-token bucket only


# DRAFT_APPEND_FUSE: the drafter's per-layer flashinfer
# append_paged_kv_cache folded into the fused q / k norm + RoPE launch (EX11) (the same K bits, V copied; same pages /
# slots). Exact. Default ON; kill switch TRTLLM_M31_DISABLE_DRAFT_APPEND_FUSE=1, read at import. Taken only where the
# drafter appends through flashinfer's paged HND cache (append_supported); otherwise the per-layer append runs as
# before.
DRAFT_APPEND_FUSE = not m31_disabled("DRAFT_APPEND_FUSE")
# FC_SHARD: the drafter's fc (replicated on every rank) as this rank's output-column shard + an all-gather in
# project_target_hidden, bit for bit the full fc (the K + 1 verify rows at batch 1). Kill switch
# TRTLLM_M31_DISABLE_FC_SHARD, read at import.
FC_SHARD = not m31_disabled("FC_SHARD")
# FRONT_CAPFORK on this worker's capture: the DSpark target runs M3DSparkWorker, whose
# capture_hidden_states issues the in-tree DFlash capture copy (inplace_slice_copy, a memcpy node in the decode graph)
# between a capture layer's boundary all-reduce and the next QKV GEMV. That copy runs on the capture stream instead
# (m31_capture_fork.fork_capture), which the target forward joins right after its decoder stack, before the
# final-hidden capture, the head and this worker read the capture buffer. Bit for bit the in-line copy. Same switch as
# modeling.py's FRONT_CAPFORK: TRTLLM_M31_DISABLE_FRONT_CAPFORK=1 (the in-line copy), read at import.
FRONT_CAPFORK = not m31_disabled("FRONT_CAPFORK")
# modeling.FINAL_HIDDEN_LAYER_ID (the Eagle3 id of the final hidden state; modeling imports this module, so the value
# is restated here and the capfork UT asserts the two agree). That capture runs after the join and stays in line.
_FINAL_HIDDEN_LAYER_ID = -1  # capacity: fixed: an Eagle3 layer id, not a size
# FOLD_F8: on a pure-decode step the drafter's block-table refresh (DRAFTER_BLOCK_TABLE_REFRESH_FUSED) is decoded
# inside that step's input launch (DRAFTER_INPUTS_FUSED): nothing reads the tables between the two on such a step, and
# every value is the same integer. Needs both. Default ON; kill switch TRTLLM_M31_DISABLE_FOLD_BLOCK_TABLE_IN_INPUTS=1,
# read at import.
FOLD_F8 = not m31_disabled("FOLD_BLOCK_TABLE_IN_INPUTS")
_f8_logged = False
# FOLD_F2: on a decode-only step whose forward tail stores in the forward (modeling.M3SpecStepFold), N3 A2's
# next_new_tokens is written by the tail's launch (the same index arithmetic) instead of a launch of its own.
# Default ON; kill switch TRTLLM_M31_DISABLE_FOLD_NEXT_TOKENS_IN_TAIL=1, read at import.
FOLD_F2 = not m31_disabled("FOLD_NEXT_TOKENS_IN_TAIL")
# FOLD_F3: R1's Philox window (offsets[:k + 1] + first * SPAN) is computed by the accept's rejection launch (REJ_MC,
# ops/sampling/m31_chain_spec_mc), which also writes it for the draft block, instead of an add of its own; taken only
# while REJ_MC is the sampler's override (every accept offset reaches it, and it writes the window before declining).
# Default ON; kill switch TRTLLM_M31_DISABLE_FOLD_PHILOX_IN_REJECTION=1, read at import.
FOLD_F3 = not m31_disabled("FOLD_PHILOX_IN_REJECTION")
# FOLD_F7: FC_SHARD's output gather (the NCCL all-gather and the concat copy after it) as one multicast launch on its
# own workspace: the bf16 rows viewed as fp32 words, which allgather_f32 copies as 16-byte vectors, so the gathered
# bytes are the all-gather's. Collective at prebuild (every rank compiles, or every rank keeps NCCL). Default ON;
# kill switch TRTLLM_M31_DISABLE_FOLD_FC_SHARD_GATHER_MC=1, read at import.
FOLD_F7 = not m31_disabled("FOLD_FC_SHARD_GATHER_MC")
# FOLD_F6: the generation rows' accept counts go to the fused context store (DRAFTER_CTX_KV_STORE_FUSED) and advance
# (DRAFTER_CTX_ADVANCE_FUSED) as the int32 tensor they arrive in: both read int32 and int64 counts alike (the same
# integers), so the int64 copy before them is one launch fewer. Needs both. Default ON; kill switch
# TRTLLM_M31_DISABLE_FOLD_ACCEPT_COUNTS_INT32=1, read at import.
FOLD_F6 = not m31_disabled("FOLD_ACCEPT_COUNTS_INT32")
# TP8 CAUSAL_DRAFT (not bitwise: changes the draft distribution): the DSpark drafter's windowed block decode runs
# causal in-block (each block query attends the block keys up to itself) with the same left window, instead of the
# non-causal in-block window the drafter was trained with. Drops each windowed layer's log-sum-exp output and the merge
# of the later block keys (two launches per drafter layer). Default ON for this target; kill switch
# TRTLLM_M31_DISABLE_TP8_CAUSAL_DRAFT=1 (the non-causal decode). Read at import.
TP8_CAUSAL_DRAFT = not m31_disabled("TP8_CAUSAL_DRAFT")
_APPEND_FUSE_TAKEN = [0]


def _qk_norm_rope_append_logged(*args):
    """catalog qk_norm_rope_append, counting taken calls and logging the first (host calls: eager or captured)."""
    out = qk_norm_rope_append(*args)
    if out is not None:
        _APPEND_FUSE_TAKEN[0] += 1
        if _APPEND_FUSE_TAKEN[0] == 1:
            logger.info(
                "[M3 dispatch] DRAFT_APPEND_FUSE taken: drafter K / V append in the fused q / k norm + RoPE launch"
            )
    return out


def _s9b_mc_op(mapping):
    """The fused AR + norm op DRAFT_AR_NORM_CLUSTER installs: (normed, residual + reduced) from the multicast
    all-reduce + norm, or None (the stock fused all-reduce runs) for operands it does not take. Shapes and the
    collectively created workspace decide, so every TP rank agrees."""

    def op(partial, residual, norm_weight, eps):
        if partial.dim() != 2 or not 1 <= partial.shape[0] <= S9B_MC_MAX_ROWS:
            return None
        workspace = m3_mc_allreduce.get_workspace(mapping, partial.shape[-1])
        if workspace is None:
            return None
        return m3_mc_allreduce.m3_mc_allreduce_norm(
            partial.contiguous(),
            residual.contiguous(),
            norm_weight,
            eps,
            workspace,
            use_pdl=get_env_enable_pdl(),
            drafter_norm=True,
        )

    return op


# DRAFT_WINDOW_CAPPED_DECODE (opt-in, TRTLLM_M31_ENABLE_DRAFT_WINDOW_CAPPED_DECODE=1): the largest decode batch the
# window-capped max_seq_len applies to, by draft length; 0 = off. Per draft length: the largest batch at which the
# window-capped drafter attention is not slower than the stock call at every batch up to it (400K-token context).
S5_MAX_BATCH_BY_K = {3: 0, 7: 2}

# Lean drafter glue (DRAFTER_LEAN_GLUE) applies at every draft length: token-identical to the eager glue at draft
# lengths 3 and 7 with the AllReduce autotune off (as the target sets it). TRTLLM_M31_DISABLE_DRAFTER_LEAN_GLUE=1 turns
# it off.
S10_MAX_DRAFT_LEN = 63


def s10_applies(num_draft_tokens: int) -> bool:
    return num_draft_tokens <= S10_MAX_DRAFT_LEN


def install_draft_swiglu(draft_model) -> int:
    """DRAFT_SWIGLU (exact): the family NATIVE_SWIGLU hook (``minimax_m3/ops/native_swiglu``) on the drafter's
    GatedMLPs, so their SwiGLU-OAI (7 / 1.702 / 1) runs ``mv2_minimax_m3::minimax_m31_swiglu_oai_vec``'s row grid with
    PDL, bit for bit the Triton ``silu_and_mul`` the stock GatedMLP launches, instead of that Triton kernel. Kill switch
    TRTLLM_M31_DISABLE_NATIVE_SWIGLU=1 (a family switch, read at import): the torch SwiGLU-OAI (fp32, not bitwise),
    never Triton. Returns how many MLPs it wraps (5 on the M3.1 drafter)."""
    layers = getattr(getattr(draft_model, "model", None), "layers", None) or []
    return native_swiglu.install(getattr(layer, "mlp", None) for layer in layers)


def install_causal_draft(draft_model) -> int:
    """TP8 CAUSAL_DRAFT: resolve every non-causal windowed drafter layer's block decode as causal with the same left
    window (no later-key merge), on this draft model instance only. Returns how many layers it turns causal."""
    stock = getattr(draft_model, "_resolve_block_attention", None)
    layers = getattr(getattr(draft_model, "model", None), "layers", None)
    if stock is None or layers is None:
        return 0

    def resolve(layer_idx: int, _stock=stock):
        causal, (left, right) = _stock(layer_idx)
        if not causal and left >= 0:
            return True, (left, 0)
        return causal, (left, right)

    turned = sum(1 for i in range(len(layers)) if resolve(i) != stock(i))
    if turned:
        draft_model._resolve_block_attention = resolve
        logger.info(
            f"[M3 dispatch sm_107 tp8] TP8 CAUSAL_DRAFT taken: {turned} windowed drafter layers decode causal "
            f"in-block (no log-sum-exp, no later-key merge)"
        )
    return turned


def s5_max_batch_for(num_draft_tokens: int) -> int:
    """Draft length <= 3 takes the 3 entry, >= 7 the 7 entry, and in between the smaller of the two."""
    if num_draft_tokens <= 3:
        return S5_MAX_BATCH_BY_K[3]
    if num_draft_tokens >= 7:
        return S5_MAX_BATCH_BY_K[7]
    return min(S5_MAX_BATCH_BY_K.values())


def is_standalone_dspark(spec_config) -> bool:
    """Whether the run speculates with a standalone DSpark drafter (its own checkpoint, ``decoding_type: DSpark``)."""
    return bool(
        spec_config is not None
        and spec_config.spec_dec_mode.use_one_engine()
        and spec_config.spec_dec_mode.is_dspark()
        and not getattr(spec_config, "draft_is_embedded_in_target", False)
    )


class M3EmbedTokens(nn.Module):
    """The target's token embedding as the module the in-tree drafter borrows (``load_weights_from_target_model``).

    The drafter embeds the bonus and mask tokens through ``embed_tokens(ids)``; the target keeps its replicated
    table as a raw tensor and gathers it with ``m3_embedding``, so this module calls the same gather on the same
    table (an exact gather, so the drafter sees the rows the target embeds).
    """

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self._core = [core]  # not a submodule: the table belongs to the target

    @property
    def weight(self) -> torch.Tensor:
        return self._core[0].w["embed"]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return m3_embedding(ids, self.weight)


def s10c_c_drafter_inputs(
    embed_tokens,
    accepted,
    num_accepted,
    slots,
    ctx_len,
    max_ctx,
    mask_token_id,
    trained_mask_embedding,
    block_size,
    tokens_per_request,
):
    """Fused drafter inputs (DRAFTER_INPUTS_FUSED): the drafter's step inputs in one launch
    (ops/dspark/dspark_drafter_inputs), for this target's embedding only: ``M3EmbedTokens`` is an exact gather of the
    replicated table, so the noise block is table rows (the trained mask row if the drafter keeps one). None for any
    other embedding: the stock path runs."""
    if not isinstance(embed_tokens, M3EmbedTokens):
        return None
    table = embed_tokens.weight
    mask_row = (
        trained_mask_embedding if trained_mask_embedding is not None else table[int(mask_token_id)]
    )
    return _drafter_inputs(
        accepted,
        num_accepted,
        slots,
        ctx_len,
        max_ctx,
        table,
        mask_row,
        block_size,
        tokens_per_request,
    )


def s10c_c_drafter_inputs_and_refresh(
    embed_tokens,
    accepted,
    num_accepted,
    slots,
    ctx_len,
    max_ctx,
    mask_token_id,
    trained_mask_embedding,
    block_size,
    tokens_per_request,
    refresh,
):
    """FOLD_F8: s10c_c_drafter_inputs with the step's block-table refresh ``refresh`` (the fused refresh's arguments)
    decoded in the same launch; None when either half declines (the caller then runs both on their own)."""
    if not isinstance(embed_tokens, M3EmbedTokens):
        return None
    table = embed_tokens.weight
    mask_row = (
        trained_mask_embedding if trained_mask_embedding is not None else table[int(mask_token_id)]
    )
    return _drafter_inputs_and_refresh(
        accepted,
        num_accepted,
        slots,
        ctx_len,
        max_ctx,
        table,
        mask_row,
        block_size,
        tokens_per_request,
        *refresh,
    )


# Fused strict accept (N3 A2, switch TRTLLM_M31_DISABLE_STRICT_ACCEPT_FUSED): the target's strict acceptance and
# next-step input tokens in one launch each (ops/dspark/strict_accept), bitwise to the in-tree torch path. Read once, at
# import.
N3A_DISABLED = m31_disabled("STRICT_ACCEPT_FUSED")
# Fused draft KV rewind (N3A-KV, switch TRTLLM_M31_DISABLE_DRAFT_KV_REWIND_FUSED, and off with the fused strict accept):
# DFlash's draft-forward KV bookkeeping (the
# rewind amount 1 - num_accepted and kv_lens += 1) in one launch where the in-tree lines run, after the
# forced-acceptance override, bitwise (integer arithmetic). Read once, at import.
N3A_KV_DISABLED = N3A_DISABLED or m31_disabled("DRAFT_KV_REWIND_FUSED")
# Fused verify-logits copy (N3 P, switch TRTLLM_M31_DISABLE_VERIFY_LOGITS_COPY_FUSED): the verify-row target logits
# without the identity gather_ids copy and with the vocab concat and the fp32 cast in one strided copy, bitwise to the
# logits processor. Read at import.
N3P_DISABLED = m31_disabled("VERIFY_LOGITS_COPY_FUSED")
# Multicast verify-logits all-gather (N3 A1, switch TRTLLM_M31_DISABLE_VERIFY_LOGITS_ALLGATHER_MC): the verify logits'
# vocab all-gather, concat and fp32 cast in one launch over a multicast NVLink workspace (ops/comm/allgather_cast),
# bitwise to the NCCL all-gather + strided copy below. The verify rows (B x (K + 1), up to 64 rows: draft length 3 at
# batches 1..16, draft length 7 at batches 1..8). Read once, at import. Default on; the switch turns it off.
A1_DISABLED = m31_disabled("VERIFY_LOGITS_ALLGATHER_MC")
A1_MAX_ROWS = 64  # capacity: fixed: allgather_cast.MAX_ROWS (K3 at batch 16, K7 at batch 8)
# A1 at draft lengths other than 3 (switch TRTLLM_M31_DISABLE_VERIFY_LOGITS_ALLGATHER_MC_ANY_DRAFT_LEN; draft length 3
# keeps A1).
A1_K7_DISABLED = m31_disabled("VERIFY_LOGITS_ALLGATHER_MC_ANY_DRAFT_LEN")
# DRAFT_LOGITS_ALLGATHER_MC (opt-in, TRTLLM_M31_ENABLE_DRAFT_LOGITS_ALLGATHER_MC=1): the rejection sampler's gather of
# the vocab-sharded fp32 draft logits (stock: the NCCL all-gather + a concat copy) as one launch on the
# VERIFY_LOGITS_ALLGATHER_MC multicast workspace (allgather_cast.allgather_f32: copied as is, bit for bit). Taken for
# the B * K draft rows up to the fp32 capacity while that workspace is set up; other shapes keep the stock
# all-gather. Off by default: with it on, two runs of the same requests sampled different tokens on a few long
# requests, which the stock all-gather does not. Read at import.
MCAG = m31_enabled("DRAFT_LOGITS_ALLGATHER_MC")
_n3a_logged = False


def _n3a_log_once() -> None:
    global _n3a_logged
    if not _n3a_logged:
        _n3a_logged = True
        logger.info(
            "[M3 dispatch] n3_strict_accept taken: fused strict acceptance + next-step input tokens"
        )


_n3p_logged = False
_n3a_kv_logged = False


def _n3a_kv_log_once() -> None:
    global _n3a_kv_logged
    if not _n3a_kv_logged:
        _n3a_kv_logged = True
        logger.info(
            "[M3 dispatch] n3a_kv_rewind taken: draft-forward KV rewind + kv_lens bump in one launch"
        )


_a1_logged = set()


def _f3_defers(offsets) -> bool:
    """FOLD_F3's precondition: the accept offset reaches REJ_MC (the installed override, with its window forms)."""
    from tensorrt_llm._torch.speculative import interface

    return (
        FOLD_F3
        and getattr(interface, "_CHAIN_SPEC_OVERRIDE", None) is _chain_spec_mc.chain_spec_mc
        and _chain_spec_mc.window_ops_built()
        and isinstance(offsets, torch.Tensor)
        and offsets.dtype == torch.int64
        and offsets.is_cuda
    )


def _a1_log_once(key, message: str) -> None:
    if key not in _a1_logged:
        _a1_logged.add(key)
        logger.info(message)


def _n3p_log_once() -> None:
    global _n3p_logged
    if not _n3p_logged:
        _n3p_logged = True
        logger.info(
            "[M3 dispatch] n3p_target_logits taken: verify-row logits without the gather_ids copy and the concat"
        )


# The DSpark worker's drafter hooks that prebuild arms (engine side, DFlashWorker.enable_*). On a TensorRT-LLM
# without them prebuild sets up nothing: every item it owns stays off and the in-tree DSpark worker paths run.
_WORKER_HOOKS = (
    "enable_lean_glue",
    "enable_window_capped_decode",
    "enable_full_slot_arena",
    "enable_fc_tp_split",
    "enable_fused_ar_norm",
    "enable_fused_ar_norm_op",
    "enable_fused_ctx_store",
    "enable_fused_drafter_inputs",
    "enable_fused_ctx_advance",
    "enable_fused_block_refresh",
    "enable_int32_accept_counts",
    "enable_fused_forward_indices",
    "enable_fused_qk_norm_rope",
    "enable_fused_qk_norm_rope_append",
    "enable_attention_pdl",
    "enable_draft_logits_gather_op",
    "enable_fused_kv_rewind",
)
# TP8_ANCESTRAL_DRAFT (opt-in, TRTLLM_M31_ENABLE_TP8_ANCESTRAL_DRAFT=1): the in-tree DSpark ancestral draft
# (DSparkWorker._samples_block_ancestrally: a Markov-head drafter under advanced draft sampling) returns the head's
# logits from _try_fused_draft_head_and_refine, so _refine_block_logits -- where this target's Markov chain and
# DRAFT_PROB1 run -- is never called, and the block is drawn by the per-position loop (or an installed ancestral block
# override). Off by default: the target drafts through the Markov chain / DRAFT_PROB1 route. Read at import.
TP8_ANCESTRAL = m31_enabled("TP8_ANCESTRAL_DRAFT")
_TP8_ANCESTRAL_LOGGED: list = []


class M3DSparkWorker(S4WorkerMixin, DSparkWorker):
    """The in-tree ``DSparkWorker`` with the calls this target's shell makes on its worker. W1: the one-launch Markov
    chain (``S4WorkerMixin``, switch ``TRTLLM_M31_DISABLE_DRAFT_S4``). N3 A2: the strict acceptance and the next-step
    input tokens in one launch each (switch ``TRTLLM_M31_DISABLE_STRICT_ACCEPT_FUSED``). N3 P / A1: the verify logits
    (switches ``TRTLLM_M31_DISABLE_VERIFY_LOGITS_COPY_FUSED`` / ``TRTLLM_M31_DISABLE_VERIFY_LOGITS_ALLGATHER_MC``).
    TAILBK R1 / Z1: the one-request Philox rows from the per-token window and no padding-row zeroing in captured batch-1
    graphs, bitwise (spec_metadata's tail_bookkeep; switch ``TRTLLM_M31_DISABLE_TAIL_BOOKKEEP``). Everything else is the
    in-tree path."""

    _ctxlen_on = (
        False  # TP8 CTXLEN_FUSE: the resume update in one launch (prebuild compiled its kernels)
    )
    _f8_on = False
    _f8_pending = (
        None  # FOLD_F8: this step's deferred refresh arguments (the fused refresh's), or None
    )
    _f2_tail = None  # FOLD_F2: the M3SpecStepFold whose tail writes this forward's next_new_tokens, or None

    def _samples_block_ancestrally(self, draft_model, spec_metadata) -> bool:
        """TP8_ANCESTRAL_DRAFT (off by default): the in-tree ancestral draft only when opted in; otherwise the Markov
        chain / DRAFT_PROB1 route through _refine_block_logits."""
        if not _TP8_ANCESTRAL_LOGGED:
            _TP8_ANCESTRAL_LOGGED.append(True)
            route = (
                "on (the in-tree ancestral draft)"
                if TP8_ANCESTRAL
                else "off (the Markov chain / DRAFT_PROB1 route)"
            )
            logger.info(f"M3.1 DSpark TP8_ANCESTRAL_DRAFT: {route}")
        return TP8_ANCESTRAL and super()._samples_block_ancestrally(draft_model, spec_metadata)

    def prepare_1st_drafter_inputs(self, *args, **kwargs):
        """TP8 CTXLEN_FUSE: the in-tree first drafter inputs with the stored-context resume update in one launch
        (``dspark_ctxlen``); the in-tree method when the item is off on this rank."""
        if self._ctxlen_on:
            return dspark_ctxlen.prepare_1st_drafter_inputs_ctxlen_fused(self, *args, **kwargs)
        return super().prepare_1st_drafter_inputs(*args, **kwargs)

    def arm_next_new_tokens_tail(self, spec_fold) -> None:
        """Set by the shell before each forward: the fold whose forward tail runs after it (a decode-only step), else
        None."""
        self._f2_tail = spec_fold if FOLD_F2 else None

    def _refresh_ctx_block_tables(self, attn_metadata, num_seqs: int) -> bool:
        """FOLD_F8: on a pure-decode step the refresh is handed to the step-inputs launch in
        prepare_1st_drafter_inputs' generation branch, which runs before any reader of the tables on such a step;
        every other step refreshes here as before."""
        self._f8_pending = None
        src = getattr(attn_metadata, "draft_kv_cache_block_offsets", None)
        if (
            self._f8_on
            and getattr(attn_metadata, "num_contexts", None) == 0
            and num_seqs > 0
            and self._ctx_block_tables is not None
            and src is not None
        ):
            self._ctx_table_width = int(src.shape[-1])
            self._f8_pending = (
                src,
                self._ctx_pool_idx,
                num_seqs,
                self._ctx_block_divisor,
                self._ctx_block_tables,
                self._ctx_block_counts,
            )
            return True
        return super()._refresh_ctx_block_tables(attn_metadata, num_seqs)

    def _f8_drafter_inputs(self, *args):
        """FOLD_F8's step-inputs hook: the combined launch when a refresh is pending and both halves take it; else the
        pending refresh on its own first, then the fused drafter inputs."""
        global _f8_logged
        pending, self._f8_pending = self._f8_pending, None
        if pending is not None:
            out = s10c_c_drafter_inputs_and_refresh(*args, pending)
            if out is not None:
                if not _f8_logged:
                    _f8_logged = True
                    logger.info(
                        "[M3 dispatch sm_107 tp8] FOLD_F8 taken: block-table refresh in the step-inputs launch"
                    )
                return out
            super()._refresh_ctx_block_tables(
                SimpleNamespace(draft_kv_cache_block_offsets=pending[0]), pending[2]
            )
        return self._f8_inputs(*args)

    def capture_hidden_states(
        self, spec_metadata, layer_id: int, hidden_states, residual=None
    ) -> None:
        """The in-tree DFlash capture: layer output (hidden + residual) into the static capture buffer. A decode
        capture layer's plain copy (no residual addend, not the final hidden state) runs on the capture stream
        (FRONT_CAPFORK, joined by the target forward after its decoder stack)."""
        if (
            FRONT_CAPFORK
            and residual is None
            and layer_id != _FINAL_HIDDEN_LAYER_ID
            and hidden_states.is_cuda
        ):
            m31_capture_fork.fork_capture(spec_metadata, layer_id, hidden_states, "sm107_tp8")
            return
        spec_metadata.maybe_capture_hidden_states(layer_id, hidden_states, residual)

    def arm_capture_norm(
        self, attn_metadata, spec_metadata, lm_head: nn.Module, draft_model
    ) -> None:
        """No capture norm: DSpark's fc reads the raw captured layers."""
        return None

    def tail_eligible(self, attn_metadata, spec_metadata, lm_head: nn.Module) -> bool:
        """Never: the target logits always go through the logits processor and the in-tree strict accept."""
        return False

    def prebuild(self, lm_head: nn.Module, draft_model) -> None:
        """The one-launch Markov chain's shards, buffers, exchange workspace and compiles (outside capture, every
        rank); no fused verify, no coded target head, no fused draft loop."""
        rank = getattr(getattr(self, "mapping", None), "tp_rank", None)
        missing = [name for name in _WORKER_HOOKS if not hasattr(self, name)]
        if missing:
            logger.info(
                f"M3.1 DSpark: prebuild skipped on rank {rank}: the DSpark worker has no {missing[0]} hook "
                f"({len(missing)} of {len(_WORKER_HOOKS)} missing); the in-tree DSpark worker paths run"
            )
            return None
        # Lean drafter glue: the drafter's strided-view glue (bitwise; kill switch TRTLLM_M31_DISABLE_DRAFTER_LEAN_GLUE,
        # read at import).
        k = getattr(getattr(self, "spec_config", None), "max_draft_len", 0) or 0
        lean = not m31_disabled("DRAFTER_LEAN_GLUE") and s10_applies(k)
        self.enable_lean_glue(draft_model, lean)
        # DRAFT_WINDOW_CAPPED_DECODE (opt-in): window-capped max_seq_len for the drafter's sliding-window decode (not
        # bitwise: the TRTLLM-Gen tile / KV split changes, not the keys read).
        s5_max_batch = s5_max_batch_for(k) if m31_enabled("DRAFT_WINDOW_CAPPED_DECODE") else 0
        self.enable_window_capped_decode(draft_model, s5_max_batch)
        # The context arena spans every seq slot (CUDA graph metadata narrows max_num_requests to its bucket).
        self.enable_full_slot_arena()
        # DRAFT_FC_TP_SPLIT (opt-in): the drafter's replicated fc split row-parallel with one fp32 all-reduce (not
        # bitwise: the partial-sum order changes). Collective.
        s2 = self.enable_fc_tp_split(draft_model, m31_enabled("DRAFT_FC_TP_SPLIT"))
        split = getattr(draft_model, "_dflash_fc_split", None)
        ar = split[3] if split is not None else None
        ar_path = (
            "MNNVL"
            if getattr(ar, "mnnvl_allreduce", None) is not None
            else getattr(ar, "strategy", None)
        )
        # FC_SHARD (bitwise; collective: every rank builds or every rank declines). DRAFT_FC_TP_SPLIT, when on, wins.
        if hasattr(draft_model, "_dflash_fc_shard"):
            draft_model._dflash_fc_shard = (
                FcShard.build(draft_model.fc, self.mapping)
                if FC_SHARD and not s2 and getattr(draft_model, "fc", None) is not None
                else None
            )
        logger.info(f"M3.1 DSpark DRAFT_FC_TP_SPLIT: fc TP split {'on' if s2 else 'off'} on rank {rank} (all-reduce "
                    f"{ar_path})")  # fmt: skip
        logger.info(f"M3.1 DSpark DRAFT_WINDOW_CAPPED_DECODE: window-capped draft decode for batches <= "
                    f"{s5_max_batch} on rank {rank}")  # fmt: skip
        logger.info(f"M3.1 DSpark S10: lean drafter glue {'on' if lean else 'off'} on rank {rank}")
        # DRAFT_AR_NORM_FUSED: each drafter layer's all-reduce fused with the add + RMSNorm after it (not bitwise: the
        # sum is not rounded before the add). Kill switch TRTLLM_M31_DISABLE_DRAFT_AR_NORM_FUSED, read at import.
        s9b = self.enable_fused_ar_norm(draft_model, not m31_disabled("DRAFT_AR_NORM_FUSED"))
        logger.info(
            f"M3.1 DSpark S9b: fused drafter all-reduce + norm {'on' if s9b else 'off'} on rank {rank}"
        )
        # DRAFT_AR_NORM_CLUSTER (opt-in): the multicast workspace is the target's (created collectively at its first
        # eager boundary, the warmup forward, before any capture); until it exists the op declines on every rank alike.
        s9b_mc = self.enable_fused_ar_norm_op(
            draft_model, _s9b_mc_op(self.mapping) if s9b and S9B_MC else None
        )
        logger.info(
            f"M3.1 DSpark DRAFT_AR_NORM_CLUSTER: drafter all-reduce + norm on the multicast kernel "
            f"(<= {S9B_MC_MAX_ROWS} rows) {'on' if s9b_mc else 'off'} on rank {rank}"
        )
        # Fused context store: the drafter's context K / V epilogue + accept mask + paged store in one launch after the
        # K / V GEMM (bitwise). Kill switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all fused drafter glue) /
        # TRTLLM_M31_DISABLE_DRAFTER_CTX_KV_STORE_FUSED, read at import.
        s10c_a = self.enable_fused_ctx_store(
            None
            if m31_disabled("DRAFTER_FUSED_GLUE") or m31_disabled("DRAFTER_CTX_KV_STORE_FUSED")
            else fused_ctx_store
        )
        # TAILBK_SPEC F: the store kernel also runs the k_norm weight multiply (bitwise torch's bf16 multiply; one
        # launch fewer per step). Kill switch TRTLLM_M31_DISABLE_TAIL_BOOKKEEP_SPEC, read at import.
        tailbk_f = fold_k_norm_weight(bool(s10c_a) and TAILBK_SPEC)
        logger.info(
            f"M3.1 DSpark DRAFTER_CTX_KV_STORE_FUSED: {'on' if s10c_a else 'off'}; TAILBK_SPEC F k_norm weight "
            f"in the store kernel {'on' if tailbk_f else 'off'} on rank {rank}"
        )
        # Fused drafter inputs: the drafter's step inputs (bonus, positions, noise block) in one launch (bitwise:
        # integers and row copies). Kill switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all fused drafter glue) /
        # TRTLLM_M31_DISABLE_DRAFTER_INPUTS_FUSED, read at import.
        s10c_c = self.enable_fused_drafter_inputs(
            None
            if m31_disabled("DRAFTER_FUSED_GLUE") or m31_disabled("DRAFTER_INPUTS_FUSED")
            else s10c_c_drafter_inputs
        )
        logger.info(
            f"M3.1 DSpark DRAFTER_INPUTS_FUSED: fused drafter step inputs {'on' if s10c_c else 'off'} on rank {rank}"
        )
        # Fused context advance: the drafter's context advance + read view in one launch (bitwise, integer), with
        # ADV_PAR's parallel read-view copy. Kill switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all fused drafter
        # glue) / TRTLLM_M31_DISABLE_DRAFTER_CTX_ADVANCE_FUSED / TRTLLM_M31_DISABLE_ADV_PAR (the one-launch copy), read
        # at import.
        s10c_d = self.enable_fused_ctx_advance(
            None
            if m31_disabled("DRAFTER_FUSED_GLUE") or m31_disabled("DRAFTER_CTX_ADVANCE_FUSED")
            else functools.partial(advance_ctx_and_read_view, parallel=not m31_disabled("ADV_PAR"))
        )
        logger.info(
            f"M3.1 DSpark DRAFTER_CTX_ADVANCE_FUSED: {'on' if s10c_d else 'off'} on rank {rank}"
        )
        # Fused block-table refresh: the drafter's block-table refresh (decode + placeholder count) in one launch
        # (bitwise, integer). Kill switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all fused drafter glue) /
        # TRTLLM_M31_DISABLE_DRAFTER_BLOCK_TABLE_REFRESH_FUSED, read at import.
        s10c_b = self.enable_fused_block_refresh(
            None
            if m31_disabled("DRAFTER_FUSED_GLUE")
            or m31_disabled("DRAFTER_BLOCK_TABLE_REFRESH_FUSED")
            else refresh_block_tables
        )
        # Grid refresh: that refresh on a (row, column-chunk) grid (bitwise; the serial kernel's time grows with the
        # blocks per sequence, the grid's does not). Kill switch
        # TRTLLM_M31_DISABLE_DRAFTER_BLOCK_TABLE_REFRESH_GRID, read at import.
        s10c_b2 = enable_parallel_refresh(
            s10c_b and not m31_disabled("DRAFTER_BLOCK_TABLE_REFRESH_GRID")
        )
        logger.info(
            f"M3.1 DSpark DRAFTER_BLOCK_TABLE_REFRESH_FUSED: {'on' if s10c_b else 'off'} on rank {rank}"
            f" (2-D grid {'on' if s10c_b2 else 'off'})"
        )
        self._f8_on = bool(FOLD_F8 and s10c_b and s10c_c)
        self._f6_on = self.enable_int32_accept_counts(
            "[M3 dispatch sm_107 tp8] FOLD_F6 taken:" if FOLD_F6 and s10c_a and s10c_d else None
        )
        logger.info(
            f"M3.1 DSpark FOLD_F6: int32 accept counts to the fused ctx store + advance "
            f"{'on' if self._f6_on else 'off'} on rank {rank}"
        )
        if self._f8_on:
            self._f8_inputs = self._fused_drafter_inputs
            self._fused_drafter_inputs = self._f8_drafter_inputs
        logger.info(
            f"M3.1 DSpark FOLD_F8: block-table refresh in the step-inputs launch on pure-decode steps "
            f"{'on' if self._f8_on else 'off'} on rank {rank}"
        )
        # Fused forward indices: the draft model's paged block-decode index prelude in one launch (bitwise, integer).
        # Kill switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all fused drafter glue) /
        # TRTLLM_M31_DISABLE_DRAFTER_FORWARD_INDICES_FUSED, read at import.
        s10c_e = self.enable_fused_forward_indices(
            draft_model,
            None
            if m31_disabled("DRAFTER_FUSED_GLUE") or m31_disabled("DRAFTER_FORWARD_INDICES_FUSED")
            else forward_indices,
        )
        logger.info(
            f"M3.1 DSpark DRAFTER_FORWARD_INDICES_FUSED: {'on' if s10c_e else 'off'} on rank {rank}"
        )
        # EX11: the drafter block's q / k RMSNorm + partial RoPE in one launch (bitwise to flashinfer's three). Kill
        # switch TRTLLM_M31_DISABLE_EX11, read at import.
        ex11 = self.enable_fused_qk_norm_rope(
            draft_model, None if m31_disabled("EX11") else qk_norm_rope
        )
        logger.info(
            f"M3.1 DSpark EX11: fused drafter q/k norm + RoPE {'on' if ex11 else 'off'} on rank {rank}"
        )
        # DRAFT_APPEND_FUSE: on top of EX11, the same launch also writes the block's K / V into the layer's paged
        # cache (bitwise flashinfer's append), and the per-layer append launch is skipped. Kill switch
        # TRTLLM_M31_DISABLE_DRAFT_APPEND_FUSE, read at import.
        append_fuse = self.enable_fused_qk_norm_rope_append(
            draft_model, _qk_norm_rope_append_logged if ex11 and DRAFT_APPEND_FUSE else None
        )
        logger.info(
            f"M3.1 DSpark DRAFT_APPEND_FUSE: drafter K / V append in the EX11 launch {'on' if append_fuse else 'off'} "
            f"on rank {rank}"
        )
        attn_pdl = self.enable_attention_pdl(draft_model, not m31_disabled("DRAFT_ATTN_PDL"))
        logger.info(
            f"M3.1 DSpark DRAFT_ATTN_PDL: drafter attention PDL {'on' if attn_pdl else 'off'} on rank {rank}"
        )
        # DRAFT_SWIGLU (exact): the drafter MLPs' SwiGLU on the family C++ op, not the Triton silu_and_mul.
        # Before DRAFT_GU_SWIGLU_EPI, whose wrapper then passes its epilogue rows through and hands every other row
        # (rows it does not serve, DRAFTER_MXFP8 / the epilogue switched off) to this op.
        swiglu = install_draft_swiglu(draft_model)
        logger.info(f"M3.1 DSpark DRAFT_SWIGLU: {native_swiglu.engagement(swiglu)} on rank {rank}")
        his = (0, 0)
        if not m31_disabled("DRAFTER_HISLAYOUT") and k > 0 and draft_model is not None:
            his = route_his_layout(draft_model, [b * k for b in range(1, MAX_FAST_BATCH + 1)])
        logger.info(
            f"[M31_DRAFTER_HISLAYOUT] TP8 registered {his[0]} projections + {his[1]} fused MLPs on rank {rank}"
        )
        # Drafter MXFP8 projections: the drafter's qkv / o / gate_up / down GEMMs on this target's MXFP8 GEMV for the
        # decode passes' rows (B x K, B <= MAX_FAST_BATCH, up to the GEMV's own row bound), from the exact MXFP8 form of
        # the shipped weights (drafter-only precision: draft tokens change, the target's distribution does not). Kill
        # switch TRTLLM_M31_DISABLE_DRAFTER_MXFP8, read at import.
        mx = (
            0
            if m31_disabled("DRAFTER_MXFP8") or k <= 0
            else route_mx_linears(draft_model, [b * k for b in range(1, MAX_FAST_BATCH + 1)])
        )
        logger.info(
            f"M3.1 DSpark DRAFTER_MXFP8: MXFP8 drafter GEMV on {mx} projections on rank {rank}"
        )
        # DRAFT_GU_SWIGLU_EPI (exact): the drafter MLP's SwiGLU in its gate_up GEMV's epilogue (one launch fewer per
        # drafter MLP), on DRAFTER_MXFP8's rows. Kill switch TRTLLM_M31_DISABLE_DRAFT_GU_SWIGLU_EPI.
        gus = (
            0
            if his[1]
            or m31_disabled("DRAFTER_MXFP8")
            or m31_disabled("DRAFT_GU_SWIGLU_EPI")
            or k <= 0
            else route_gu_swiglu(draft_model, [b * k for b in range(1, MAX_FAST_BATCH + 1)])
        )
        logger.info(
            f"M3.1 DSpark DRAFT_GU_SWIGLU_EPI: gate_up SwiGLU epilogue on {gus} drafter MLPs on rank {rank}"
        )
        # DRAFTER_DOWN_SPLITK / DRAFTER_OPROJ_SPLITK: the drafter's down / o_proj projections on FlashInfer's BF16
        # split-K GEMM at a pinned tactic, on the 7-token draft passes whose all-reduce the fused drafter boundary
        # (DRAFT_AR_NORM_FUSED) defers; a projection the layout routing (DRAFTER_HISLAYOUT) took keeps that route.
        # Not with DRAFT_AR_NORM_CLUSTER: its multicast all-reduce + norm takes the stock projection output.
        down = route_down_splitk(
            draft_model,
            enabled=not his[1]
            and k == 7
            and s9b
            and not S9B_MC
            and not m31_disabled("DRAFTER_DOWN_SPLITK"),
        )
        logger.info(f"M3.1 DRAFTER_DOWN_SPLITK: armed {down}/5 projections on rank {rank}")
        oproj = route_oproj_splitk(
            draft_model,
            enabled=not his[0]
            and k == 7
            and s9b
            and not S9B_MC
            and not m31_disabled("DRAFTER_OPROJ_SPLITK"),
        )
        logger.info(f"M3.1 DRAFTER_OPROJ_SPLITK: armed {oproj}/5 projections on rank {rank}")
        self.prebuild_s4(lm_head, draft_model)
        # N3 A2 compiles (rank-local), then the N3 A1 workspace and compiles (collective), outside capture.
        if not N3A_DISABLED and lm_head is not None:
            _strict_accept.precompile((self.max_draft_len,), lm_head.weight.device)
        self._a1 = self._prebuild_a1(lm_head) if lm_head is not None else None
        self._mcag_cols = self._prebuild_mcag(lm_head) if self._a1 is not None and MCAG else 0
        mcag = self.enable_draft_logits_gather_op(self._mcag_gather if self._mcag_cols else None)
        logger.info(
            f"M3.1 DRAFT_LOGITS_ALLGATHER_MC: draft-logits all-gather on the multicast kernel "
            f"{'on' if mcag else 'off'} on rank {rank}"
        )
        fc_shard = getattr(draft_model, "_dflash_fc_shard", None)
        self._f7_ws = (
            self._prebuild_f7(fc_shard)
            if FOLD_F7 and fc_shard is not None and self._a1 is not None
            else None
        )
        if fc_shard is not None:
            fc_shard.set_fast_gather(self._f7_gather if self._f7_ws is not None else None)
        logger.info(
            f"M3.1 DSpark FOLD_F7: drafter fc gather on the multicast kernel "
            f"{'on' if self._f7_ws is not None else 'off'} on rank {rank}"
        )
        # TP8_FLAGFREE_GATHER: the flag-free gather of the verify logits and the fc output (collective: every rank,
        # right after both stock gathers are set up; None = off, the stock gathers serve every step).
        self._ffg = _ffg.attach(self, lm_head) if lm_head is not None else None
        n3a_kv = False
        if not N3A_KV_DISABLED and lm_head is not None:
            _strict_accept.precompile_kv_rewind(lm_head.weight.device)
            self._n3a_kv_buf = torch.empty(
                _strict_accept.max_requests(), dtype=torch.int32, device=lm_head.weight.device
            )
            n3a_kv = self.enable_fused_kv_rewind(self._n3a_kv_rewind)
        logger.info(
            f"M3.1 DRAFT_KV_REWIND_FUSED: one-launch draft-forward KV bookkeeping {'on' if n3a_kv else 'off'} on rank "
            f"{getattr(getattr(self, 'mapping', None), 'tp_rank', None)}"
        )
        # TP8 CTXLEN_FUSE (bitwise, integer): the drafter's stored-context resume update in one CUDA C++ launch, its
        # NVRTC compile here (rank-local, outside capture). Kill switch TRTLLM_M31_DISABLE_TP8_CTXLEN_FUSE.
        self._ctxlen_on = dspark_ctxlen.prepare(torch.cuda.current_device(), self)
        logger.info(
            f"M3.1 DSpark TP8 CTXLEN_FUSE: one-launch drafter ctx_len resume {'on' if self._ctxlen_on else 'off'} "
            f"on rank {rank}"
        )
        causal = install_causal_draft(draft_model) if TP8_CAUSAL_DRAFT else 0
        logger.info(
            f"M3.1 DSpark TP8 CAUSAL_DRAFT: causal in-block drafter decode on {causal} layers "
            f"({'on' if TP8_CAUSAL_DRAFT else 'killed'}) on rank {rank}"
        )

    _n3a_kv_buf = None  # the persistent rewind amounts of the fused KV rewind (int32, MAX_REQUESTS)

    def _n3a_kv_rewind(self, num_accepted, kv_lens, nc: int, bs: int):
        """The fused draft KV rewind for the DFlash hook: ``1 - num_accepted[nc:bs]`` (a view of the persistent buffer)
        with ``kv_lens[nc:bs] += 1`` in one launch, or None (the stock lines) for operands or a variant it does not
        take."""
        n = bs - nc
        buf = self._n3a_kv_buf
        if buf is None or not _strict_accept.supports_kv_rewind(num_accepted, kv_lens, buf, nc, n):
            return None
        if not _strict_accept.is_compiled_kv_rewind(num_accepted, kv_lens):
            return None
        _n3a_kv_log_once()
        return _strict_accept.spec_kv_rewind(num_accepted, kv_lens, buf, nc, n)

    _a1 = None  # the N3 A1 workspace (allgather_cast's arguments) once set up
    _ffg = None  # TP8_FLAGFREE_GATHER once set up (None = the stock gathers)

    def _a1_all_ok(self, local: bool) -> bool:
        """Whether ``local`` holds on every rank of the TP group: A1 is set up on every rank or on none (a rank that
        is not eligible, or whose compile failed, turns it off everywhere)."""
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _mnnvl_workspace_all_succeeded,
        )

        return _mnnvl_workspace_all_succeeded(_get_mnnvl_workspace_comm(self.mapping), local)

    def _prebuild_a1(self, lm_head: nn.Module):
        """The N3 A1 workspace and its K3 verify-row compiles, or None (every rank alike): off by its switch, at
        another draft length, without a column-parallel bf16 head over a TP group, or when a compile fails on any
        rank."""
        from tensorrt_llm._torch.modules.linear import TensorParallelMode

        mapping = self.mapping
        if mapping.tp_size <= 1 or mapping.enable_attention_dp:
            return None
        cols = int(getattr(lm_head, "out_features", 0))
        sms = torch.cuda.get_device_properties(lm_head.weight.device).multi_processor_count
        # Why this rank would not set A1 up (empty: it would). The grid is a function of rows and the SM count:
        # requiring every SM up to the workspace's MAX_GRID makes it a function of rows alone, the same on every rank.
        off = [
            why
            for why, bad in (
                (
                    "TRTLLM_M31_DISABLE_VERIFY_LOGITS_ALLGATHER_MC set",
                    A1_DISABLED,
                ),
                (
                    f"draft length {int(self.max_draft_len)} with "
                    "TRTLLM_M31_DISABLE_VERIFY_LOGITS_ALLGATHER_MC_ANY_DRAFT_LEN set",
                    int(self.max_draft_len) != 3 and A1_K7_DISABLED,
                ),
                (
                    f"draft length {int(self.max_draft_len)}: more than {A1_MAX_ROWS} verify rows at batch 1",
                    int(self.max_draft_len) + 1 > A1_MAX_ROWS,
                ),
                (
                    "not a column-parallel, gathered, unpadded, bias-free head",
                    getattr(lm_head, "tp_mode", None) != TensorParallelMode.COLUMN
                    or not getattr(lm_head, "gather_output", False)
                    or getattr(lm_head, "padding_size", 0) != 0
                    or getattr(lm_head, "bias", None) is not None,
                ),
                (f"head dtype {lm_head.weight.dtype}", lm_head.weight.dtype != torch.bfloat16),
                (
                    f"{cols} shard columns (whole 16-byte vectors needed)",
                    cols <= 0 or cols % 8 != 0,
                ),
                (
                    f"{sms} SMs (fewer than {_allgather_cast.MAX_GRID})",
                    sms < _allgather_cast.MAX_GRID,
                ),
            )
            if bad
        ]
        if not self._a1_all_ok(not off):
            logger.info(
                f"M3.1 N3 A1: off on rank {mapping.tp_rank} ({'; '.join(off) or 'off on another rank'}); "
                "the verify logits use the NCCL all-gather"
            )
            return None
        k1 = int(self.max_draft_len) + 1
        rows = tuple(b * k1 for b in range(1, A1_MAX_ROWS // k1 + 1))
        start = time.perf_counter()
        if not _allgather_cast.precompile(
            lm_head.mapping, cols, rows, lm_head.weight.device, agree=self._a1_all_ok
        ):
            logger.warning(
                "M3.1 N3 A1: a compile failed on some rank; every rank keeps the NCCL all-gather"
            )
            return None
        ws = _allgather_cast.workspace_args(lm_head.mapping, cols)
        # The engagement line: one per rank, at server start.
        logger.info(
            f"M3.1 N3 A1: one-launch verify all-gather set up on rank {mapping.tp_rank}/{mapping.tp_size} "
            f"(shard {cols} columns, rows {rows}, workspace {ws['ws_uc'].numel() * 4} bytes/rank, "
            f"set-up {time.perf_counter() - start:.2f} s)"
        )
        return ws

    def _prebuild_mcag(self, lm_head: nn.Module) -> int:
        """Compile the fp32 draft-logits gather for B * K rows up to its capacity, on every rank (A1's workspace and
        agreement); its shard columns, or 0 (every rank alike) when a compile fails anywhere."""
        cols = int(getattr(lm_head, "out_features", 0))
        k = int(self.max_draft_len)
        if k <= 0 or cols % 4 != 0:
            return 0
        rows = tuple(b * k for b in range(1, _allgather_cast.MAX_ROWS_FP32 // k + 1))
        if not _allgather_cast.precompile_f32(
            lm_head.mapping, cols, rows, lm_head.weight.device, agree=self._a1_all_ok
        ):
            logger.warning(
                "M3.1 DRAFT_LOGITS_ALLGATHER_MC: a compile failed on some rank; every rank keeps the NCCL all-gather"
            )
            return 0
        return cols

    _f7_ws = None  # FOLD_F7: (fp32 columns per rank, the gather's workspace arguments) once set up

    def _prebuild_f7(self, fc_shard):
        """Compile FOLD_F7's gather of FC_SHARD's [rows, width] bf16 output (as fp32 [rows, width / 2]) for every row
        count FC_SHARD takes, on its own workspace (A1's agreement); None (every rank alike) when it does not apply."""
        width = int(fc_shard.w_shard.shape[0])
        rows = tuple(range(1, min(FC_SHARD_MAX_ROWS, _allgather_cast.MAX_ROWS_FP32) + 1))
        if width <= 0 or width % 8 != 0:
            return None
        cols = width // 2
        if not _allgather_cast.precompile_f32(
            fc_shard.mapping, cols, rows, fc_shard.w_shard.device, agree=self._a1_all_ok
        ):
            logger.warning(
                "M3.1 DSpark FOLD_F7: a compile failed on some rank; every rank keeps the NCCL all-gather"
            )
            return None
        return cols, _allgather_cast.workspace_args(fc_shard.mapping, cols)

    def _f7_gather(self, y: torch.Tensor):
        """FC_SHARD's output gathered in one launch (its bf16 rows as fp32 words), or None (the NCCL all-gather) for a
        layout or shape it does not take; shapes decide, so every rank declines alike."""
        cols, ws = self._f7_ws
        if (
            y.dtype != torch.bfloat16
            or y.dim() != 2
            or not y.is_contiguous()
            or int(y.shape[1]) != 2 * cols
            or y.data_ptr() % 16 != 0
            or not _allgather_cast.is_compiled(
                int(y.shape[0]), cols, ws["world"], device=y.device, fp32=True
            )
        ):
            return None
        if self._ffg is not None:  # TP8_FLAGFREE_GATHER: the same gather, flag-free
            _ffg_out = self._ffg.fc_output(y.view(torch.float32))
            if _ffg_out is not None:
                _a1_log_once(
                    "f7",
                    "[M3 dispatch sm_107 tp8] FOLD_F7 taken: drafter fc gathered in one multicast launch",
                )
                return _ffg_out.view(torch.bfloat16)
        out = _allgather_cast.gather_last_dim_f32(y.view(torch.float32), **ws)
        if out is None:
            return None
        _a1_log_once(
            "f7",
            "[M3 dispatch sm_107 tp8] FOLD_F7 taken: drafter fc gathered in one multicast launch",
        )
        return out.view(torch.bfloat16)

    def _mcag_gather(self, logits: torch.Tensor):
        """The draft logits gathered in one launch on the multicast workspace, or None (the stock all-gather) for a
        shape that is not compiled; the decision depends on the shape only, the same on every rank."""
        rows = logits.numel() // max(int(logits.shape[-1]), 1)
        if (
            logits.dtype != torch.float32
            or int(logits.shape[-1]) != self._mcag_cols
            or not _allgather_cast.is_compiled(
                rows, self._mcag_cols, self._a1["world"], device=logits.device, fp32=True
            )
        ):
            _a1_log_once(
                ("mcag", rows, logits.dtype),
                f"[M3 dispatch] DRAFT_LOGITS_ALLGATHER_MC not taken: {rows} rows {logits.dtype}; NCCL all-gather",
            )
            return None
        _a1_log_once(
            "mcag",
            "[M3 dispatch] DRAFT_LOGITS_ALLGATHER_MC taken: one launch on the multicast workspace",
        )
        return _allgather_cast.gather_last_dim_f32(logits.contiguous(), **self._a1)

    def _n3a_eligible(
        self, logits, draft_tokens, num_contexts: int, batch_size: int, spec_metadata
    ) -> bool:
        """Whether this step's strict acceptance takes the fused op: a decode-only, non-ragged step whose logits hold
        exactly batch_size x (D + 1) verify rows and whose drafts the op takes. Host state and shapes only, so it is
        fixed per captured graph and the same on every rank. Decided before the target tokens are sampled, so the
        sampler runs exactly once either way."""
        if N3A_DISABLED or spec_metadata is None or spec_metadata.is_ragged_verify:
            return False
        if num_contexts != 0 or batch_size <= 0 or draft_tokens.dim() != 2:
            return False
        n_req, draft_len = draft_tokens.shape
        if n_req != batch_size or draft_len <= 0 or n_req > _strict_accept.max_requests():
            return False
        rows = logits.shape[0] if logits.dim() > 1 else 1
        return rows == batch_size * (draft_len + 1) and draft_tokens.is_contiguous()

    def _sample_and_accept_draft_tokens_base(
        self, logits, draft_tokens, num_contexts, batch_size, spec_metadata
    ):
        """The in-tree strict acceptance with its epilogue in one launch (N3 A2): the same target tokens (the in-tree
        sampler, called once), the same accepted tokens and counts, then the in-tree forced-acceptance override (its
        RNG pool and counters advance as on an in-tree step)."""
        if not self._n3a_eligible(logits, draft_tokens, num_contexts, batch_size, spec_metadata):
            return super()._sample_and_accept_draft_tokens_base(
                logits, draft_tokens, num_contexts, batch_size, spec_metadata
            )
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        target_tokens = self._sample_tokens_for_batch(
            logits, spec_metadata, num_contexts, batch_size
        )
        if _strict_accept.supports_strict_accept(
            target_tokens, draft_tokens
        ) and _strict_accept.is_compiled_strict_accept(target_tokens, draft_tokens):
            accepted, num_accepted = _strict_accept.spec_strict_accept(target_tokens, draft_tokens)
            _n3a_log_once()
        else:
            # The sampler already ran: finish with the in-tree epilogue (the op's reference is that code).
            accepted, num_accepted = _strict_accept.spec_strict_accept.reference(
                target_tokens, draft_tokens
            )
        num_accepted = self._apply_force_accepted_tokens(
            num_accepted, num_contexts, draft_tokens.shape[-1], spec_metadata=spec_metadata
        )
        return accepted, num_accepted

    def target_logits(self, hidden_states, attn_metadata, spec_metadata, lm_head: nn.Module):
        """The verify rows' gathered fp32 target logits (N3 P-gid + P-cat), or None for the logits processor.

        On a decode-only, non-ragged step every row is a verify row and ``gather_ids`` is ``arange(num_tokens)`` (each
        generation request extends it with the next ``1 + D`` row indices), so the rows are a slice, not a gather.
        The processor's head is the column-parallel BF16 GEMM, an all-gather along the vocab (NCCL, then a concat)
        and ``.float()``; here the same all-gather runs along dim 0 (no concat) and one strided copy writes the
        rank-major shards into the fp32 output: the same bytes, an exact widening. Shapes and host state only, so
        the decision is fixed per captured graph and the same on every rank."""
        from tensorrt_llm._torch.modules.linear import TensorParallelMode

        if (
            N3P_DISABLED
            or spec_metadata is None
            or getattr(spec_metadata, "is_ragged_verify", False)
        ):
            return None
        if attn_metadata.num_contexts != 0 or spec_metadata.gather_ids is None:
            return None
        rows = hidden_states.shape[0]
        if spec_metadata.gather_ids.shape[0] != rows or rows == 0:
            return None
        if (
            getattr(lm_head, "tp_mode", None) != TensorParallelMode.COLUMN
            or not getattr(lm_head, "gather_output", False)
            or getattr(lm_head, "padding_size", 0) != 0
            or getattr(lm_head, "bias", None) is not None
        ):
            return None
        shard = lm_head.apply_linear(hidden_states, None)
        _n3p_log_once()
        tp = lm_head.tp_size
        if tp == 1:
            return shard.float()
        # N3 A1: one launch for the all-gather, the concat and the cast. The decision depends on the rows and the
        # set-up only (the same on every rank, fixed per captured graph).
        if self._a1 is not None:
            if rows <= A1_MAX_ROWS and _allgather_cast.is_compiled(
                rows, shard.shape[-1], tp, 0, shard.device
            ):
                _a1_log_once(
                    "taken",
                    "[M3 dispatch] n3_a1 allgather_cast taken: verify logits gathered in one launch",
                )
                if self._ffg is not None:  # TP8_FLAGFREE_GATHER: the same gather, flag-free
                    _ffg_out = self._ffg.verify_logits(shard)
                    if _ffg_out is not None:
                        return _ffg_out
                return _allgather_cast.allgather_cast(shard, **self._a1)
            _a1_log_once(
                rows,
                f"[M3 dispatch] n3_a1 allgather_cast not taken: {rows} verify rows (compiled: verify rows up to "
                f"{A1_MAX_ROWS}); NCCL all-gather",
            )
        gathered = allgather(shard, lm_head.mapping, dim=0)
        vocab_shard = shard.shape[-1]
        out = torch.empty((rows, tp * vocab_shard), dtype=torch.float32, device=shard.device)
        out.view(rows, tp, vocab_shard).copy_(gathered.view(tp, rows, vocab_shard).transpose(0, 1))
        return out

    def _prepare_next_new_tokens(
        self,
        accepted_tokens,
        next_draft_tokens,
        batch_indices_cuda,
        batch_size,
        num_accepted_tokens,
    ):
        """The next step's input tokens in one launch (N3 A2), else the in-tree index + concat. FOLD_F2: with int32
        operands on a step whose forward tail runs, N3 A2's launch is left to the tail (an unwritten buffer here)."""
        tail, self._f2_tail = self._f2_tail, None
        batch_idx = batch_indices_cuda[:batch_size]
        if (
            not N3A_DISABLED
            and next_draft_tokens.dim() == 2
            and next_draft_tokens.shape[0] == batch_size
            and _strict_accept.supports_next_new_tokens(
                accepted_tokens, num_accepted_tokens, batch_idx, next_draft_tokens
            )
            and _strict_accept.is_compiled_next_new_tokens(
                accepted_tokens, num_accepted_tokens, batch_idx, next_draft_tokens
            )
        ):
            operands = (accepted_tokens, num_accepted_tokens, batch_idx, next_draft_tokens)
            if tail is not None and all(t.dtype == torch.int32 for t in operands):
                out = torch.empty(
                    (batch_size, next_draft_tokens.shape[1] + 1),
                    dtype=torch.int32,
                    device=next_draft_tokens.device,
                )
                tail.defer_next_new_tokens(
                    out, operands, lambda: _strict_accept.spec_next_new_tokens(*operands)
                )
                return out
            return _strict_accept.spec_next_new_tokens(*operands)
        return super()._prepare_next_new_tokens(
            accepted_tokens, next_draft_tokens, batch_indices_cuda, batch_size, num_accepted_tokens
        )

    # TAILBK (RuntimeHooks.spec_tail_bookkeep, armed by modeling.py's TAILBK; spec_metadata.tail_bookkeep): R1 and Z1,
    # this worker's part of the sampler tail bookkeeping (B1 / S1 / A1 are the shared branches in interface.py).
    _tailbk_win = None  # R1: (spec_metadata, offsets window) the accept call left for this forward's draft block

    def _forward_impl(self, *args, **kwargs):
        """The S4 / DFlash step; R1's offsets window lives for one step only (set by its accept call, read by its
        draft block), so a step whose accept did not stash one recomputes it."""
        self._tailbk_win = None
        try:
            return super()._forward_impl(*args, **kwargs)
        finally:
            self._tailbk_win = None

    def _rng_state_per_request(self, spec_metadata, start=0, end=None, repeat=1, *, slot):
        """TAILBK R1: at one decode request, the accept and draft-block Philox (seed, offset) rows read from the
        per-token buffers (one int64 add per step instead of eight launches), else the in-tree expansion.

        ``_populate_request_rng_state`` writes row r (0..K) of the request's per-token window as
        ``seeds[r] = seed`` and ``offsets[r] = o + r * SPAN`` (o = ``request_offsets[0]``; the host sum fits int64,
        or its upload would have raised), uploaded with ``request_seeds`` / ``request_offsets`` on every sampled step.
        The in-tree rows are ``(o + slot * SPAN) + k * SPAN``: draft row k at slot K + 1 is ``offsets[k] + (K + 1) *
        SPAN`` and the accept row at slot 2K + 1 is ``offsets[K] + (K + 1) * SPAN``. Integer addition is associative
        and commutative mod 2^64, so the window ``offsets[0:K + 1] + (K + 1) * SPAN`` holds exactly the in-tree
        offsets, even where they wrap, and the seeds are the request's seed by construction. The returned tensors are
        contiguous int64 views with one entry per row, as the in-tree ones. Preconditions are host state and shapes
        (``tail_bookkeep_draft_len``), so the choice is fixed per captured graph."""
        state = self._tailbk_rng_window(spec_metadata, start, end, repeat, slot)
        if state is not None:
            return state
        return super()._rng_state_per_request(spec_metadata, start, end, repeat, slot=slot)

    def _tailbk_rng_window(self, spec_metadata, start, end, repeat: int, slot: int):
        """R1's (seed, offset) rows, or None (the in-tree expansion) outside its preconditions."""
        k = tail_bookkeep_draft_len(spec_metadata, start, end)
        if k == 0:
            return None
        first = spec_metadata.rng_draft_slot(0)
        if first != k + 1 or spec_metadata.rng_accept_slot != first + k:
            return None
        accept = repeat == 1 and slot == spec_metadata.rng_accept_slot
        block = repeat == k and slot == first
        if not (accept or block):
            return None
        stash = self._tailbk_win
        if block and stash is not None and stash[0] is spec_metadata:
            win = stash[1]
            _chain_spec_mc.flush_window(win)  # FOLD_F3: unless the accept's launch already wrote it
        elif accept and _f3_defers(spec_metadata.offsets):
            win = torch.empty(k + 1, dtype=torch.int64, device=spec_metadata.offsets.device)
            _chain_spec_mc.defer_window(win, spec_metadata.offsets[: k + 1], first * _RNG_SLOT_SPAN)
        else:
            win = spec_metadata.offsets[: k + 1] + first * _RNG_SLOT_SPAN
        tail_bookkeep_log_once(
            "R1", "Philox seeds / offsets from the per-token window (one add per step)"
        )
        if accept:
            self._tailbk_win = (spec_metadata, win)
            return spec_metadata.request_seeds[0:1], win[k : k + 1]
        self._tailbk_win = None
        return spec_metadata.seeds[:k], win[:k]

    def _zero_padding_rows(
        self,
        logits,
        spec_metadata,
        num_contexts: int,
        batch_size: int,
        rows_per_request: int,
        *,
        packed_verify: bool = False,
    ):
        """TAILBK Z1: no padding-row zeroing while a batch-1 decode graph is captured, else the in-tree zeroing.

        Why the in-tree result is a bitwise copy of ``logits`` on every replay of such a graph:
        (i) a batch-1 graph replays exactly one real request. ``CUDAGraphRunner._get_padded_batch`` only appends
            dummies after the batch's own requests, up to the graph's size, and the executor forwards no empty batch
            without attention DP (``_can_queue``: batch_size > 0), so a size-1 graph never holds a dummy. Under
            attention DP an empty rank does run a dummy, so attention DP is excluded.
        (ii) a real request's slot is its ``py_seq_slot`` in ``[0, slot_capacity)`` and the padding marker is
            ``dummy_slot_row == slot_capacity`` (``prepare_rejection_sampling_buffers``), so the padding mask is
            all-false and the in-tree ``masked_fill`` copies ``logits`` unchanged.
        (iii) every consumer is out-of-place (the non-fused probs: the softmax / top-k mask ops and the m31_softmax_mc
            override allocate their outputs, top-p reads probs), so returning ``logits`` itself instead of that copy
            exposes no write to the caller's verify logits. The fused modes are not audited and keep the copy.
        (iv) the consumers see the layout the copy would have: the skip needs contiguous logits at a 256-byte aligned
            address (the copy is a fresh contiguous allocation, which the caching allocator aligns at least that much),
            since the m31_softmax_mc override takes contiguous, 16-byte aligned inputs only.
        Eager steps (the graph warmup's dummy included) keep the in-tree zeroing; so do padded graphs (2..8), which
        can hold dummies, and ragged verification (its layout check). Host state only, fixed per captured graph."""
        if self._tailbk_skips_padding_rows(logits, spec_metadata, num_contexts, batch_size):
            tail_bookkeep_log_once("Z1", "padding-row zeroing skipped in captured batch-1 graphs")
            return logits
        return super()._zero_padding_rows(
            logits,
            spec_metadata,
            num_contexts,
            batch_size,
            rows_per_request,
            packed_verify=packed_verify,
        )

    def _tailbk_skips_padding_rows(
        self, logits, spec_metadata, num_contexts: int, batch_size: int
    ) -> bool:
        """Z1's exact condition: tail bookkeeping on, one generation request and no context rows (num_gens == 1),
        no ragged verification, a non-fused sampling mode, no attention DP, contiguous 256-byte aligned logits and a
        capturing stream."""
        if not getattr(spec_metadata, "tail_bookkeep", False):
            return False
        if not logits.is_contiguous() or logits.data_ptr() % 256 != 0:
            return False
        if num_contexts != 0 or batch_size != 1 or getattr(spec_metadata, "is_ragged_verify", True):
            return False
        mode = getattr(spec_metadata, "advanced_sampling_mode", None)
        if mode is None or mode.is_fused:
            return False
        mapping = getattr(self, "mapping", None)
        if mapping is None or getattr(mapping, "enable_attention_dp", True):
            return False
        return torch.cuda.is_current_stream_capturing()


def build_dspark(
    shell, model_config: ModelConfig
) -> Tuple[ModelConfig, nn.Module, M3DSparkWorker, bool]:
    """The drafter config, the drafter and the worker of one-model standalone DSpark (mirrors
    ``SpecDecOneEngineForCausalLM`` for an external drafter)."""
    spec = model_config.spec_config
    if not is_standalone_dspark(spec):
        raise ValueError("build_dspark serves the standalone DSpark drafter only")
    draft_config = ModelConfig.from_pretrained(
        spec.speculative_model, **external_drafter_config_kwargs(model_config, spec)
    )
    # The drop is safe only because draft_config owns its QuantConfig: from_pretrained reads the drafter checkpoint
    # and takes no quant_config from the target. Never pass the target's quant_config in here.
    keep_drafter_kv_dtype(draft_config)
    draft_config.extra_attrs = model_config.extra_attrs
    use_separate = should_use_separate_draft_kv_cache(spec)
    draft = get_draft_model(model_config, draft_config, shell.lm_head, shell.model)
    # A parallel drafter (DSpark) projects its block rows through the target's logits processor, as the one-engine
    # construction sets it.
    draft.logits_processor = shell.logits_processor
    worker = M3DSparkWorker(spec, model_config.mapping, use_separate_draft_kv_cache=use_separate)
    worker.set_draft_model(draft)
    shell.epilogue.append(draft)
    shell.epilogue.append(worker)
    logger.info(
        f"M3.1 target DSpark: {type(worker).__name__} over {type(draft).__name__}, K={spec.max_draft_len}, "
        f"separate draft KV cache={use_separate}"
    )
    return draft_config, draft, worker, use_separate


def load_dspark_draft_weights(shell, weights, weight_mapper=None) -> None:
    """The drafter's own checkpoint, then the target's embedding and head (mirrors
    ``SpecDecOneEngineForCausalLM.load_draft_weights`` for DSpark).

    The drafter borrows the target's embedding and head, and ``M3EmbedTokens.weight`` IS the target's table, so a
    drafter checkpoint that ships either would load into a module the target owns or be shadowed by it: refused.
    """
    borrowed = sorted(
        k for k in weights.keys() if "embed_tokens." in k or k.split(".")[-2:-1] == ["lm_head"]
    )
    if borrowed:
        raise ValueError(
            f"the DSpark drafter checkpoint ships tensors this target lends it (embedding / head): {borrowed[:4]}"
        )
    shell.draft_model.load_weights(weights=weights, weight_mapper=weight_mapper)
    shell.draft_model.load_weights_from_target_model(shell)
    if shell.draft_model.lm_head is not shell.lm_head or (
        shell.draft_model.draft_model_full.model.embed_tokens is not shell.model.embed_tokens
    ):
        raise RuntimeError("the DSpark drafter does not borrow this target's embedding and head")
    own = sorted(
        n for n, _ in shell.named_parameters() if not n.startswith(("model.w.", "draft_model."))
    )
    logger.info(f"M3.1 target DSpark: shell parameters outside the core and the drafter: {own}")


def keep_drafter_kv_dtype(draft_config: ModelConfig) -> None:
    """Drop any KV quantization from the drafter's config, so its context pool is allocated in its projection dtype.

    The one-engine construction copies the target's KV algo onto the drafter; on this target that is NVFP4, which
    allocates a packed int8 pool the worker refuses to bind (it falls back to its private arena, and a reused
    prefix's drafter K/V is lost). The cached quant modes are dropped with it, as the draft KV budget does.
    """
    for quant in [draft_config.quant_config, *(draft_config.quant_config_dict or {}).values()]:
        quant.kv_cache_quant_algo = None
        quant.__dict__.pop("quant_mode", None)
        quant.__dict__.pop("layer_quant_mode", None)


def draft_kv_override_applies(spec_config) -> bool:
    """The target's FP8 draft-KV override is an Eagle3 need; a DSpark drafter keeps its BF16 context pool."""
    return not is_standalone_dspark(spec_config)


__all__ = [
    "M3DSparkWorker",
    "M3EmbedTokens",
    "build_dspark",
    "draft_kv_override_applies",
    "is_standalone_dspark",
    "keep_drafter_kv_dtype",
    "load_dspark_draft_weights",
]
