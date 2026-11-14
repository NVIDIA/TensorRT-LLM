# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-model DSpark on this target (phase W0): the in-tree DSpark drafter and worker, unchanged but for S9b-VR-a
(the drafter's row-parallel all-reduces take the one-shot all-reduce at decode rows; ``DRAFT_AR_ONESHOT_MAX_ROWS``)
and the bitwise tail items (TAIL_FORK, TAIL_DEFER_CAPTURE, TAIL_FC_GATHER, TAIL_FORK_REFRESH, TAIL_FORCEVEC,
TAIL_LEAN_WORKER below; ``dspark_tailfork``).

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
from typing import Optional, Set, Tuple

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import m31_capture_fork
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import native_swiglu
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.comm import (
    allgather_cast as _allgather_cast,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_ctx_store as _ctx_store,
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
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_forward_indices import (
    forward_indices,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_qk_norm_rope import (
    qk_norm_rope,
    qk_norm_rope_append,
)
from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy, allgather
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_speculative import (
    external_drafter_config_kwargs,
    get_draft_model,
)
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
    _RNG_SLOT_SPAN,
    should_use_separate_draft_kv_cache,
)
from tensorrt_llm.logger import logger

from ..m31_tail_bookkeep import tail_bookkeep_draft_len, tail_bookkeep_log_once
from . import dspark_tailfold as _tfold
from .drafter_mxfp8 import route_his_layout, route_mx_linears
from .dspark_markov_worker import MAX_FAST_BATCH, S4WorkerMixin
from .dspark_t6 import routes_line
from .dspark_tailfork import copy_captures, ctx_store_launch, ctx_store_prep, view_key
from .kernels import m3_nnt_fold
from .m31_switches import m31_disabled, m31_enabled
from .ops.embedding.m3_embedding import m3_embedding
from .ops.gemm.m31_fc_shard import FcShard
from .ops.spec.m3_force_vec import m3_force_vec

# S9BVR: the drafter's row-parallel all-reduces (each layer's o_proj and MLP down projection) of up to this many
# rows run the one-shot all-reduce instead of AUTO. On this target AUTO resolves to NCCL at every row count: the AR
# tuner's profiling of the custom tactics fails here ("Total failed profiling tactics: 14"), so the
# decode-step drafter ARs ran as the NCCL ring. SM107 TP4 [rows, 6144] BF16, graph of 10 calls: 4 rows 16.46 ->
# 5.64 us per call, 32 rows 16.66 -> 6.46; error vs the FP32 sum 2.0e-3 against NCCL's 3.9e-3 (a different
# reduction order: not bitwise against NCCL). Wider calls (the drafter's context passes) keep AUTO. Never the two-shot
# all-reduce: 430 us at 4 rows here. TRTLLM_M31_DISABLE_S9BVR=1 restores AUTO at every row count.
DRAFT_AR_ONESHOT_MAX_ROWS = (
    0 if m31_disabled("S9BVR") else 32
)  # capacity: fixed: the K3 c8 verify rows


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


class DraftOneshotAllReduce(nn.Module):
    """A row-parallel linear's all-reduce: the one-shot all-reduce up to ``max_rows`` rows, its original (AUTO)
    all-reduce above. The output buffer choice stays the original module's, so wider calls are unchanged."""

    def __init__(self, auto: AllReduce, dtype: torch.dtype, max_rows: int, name: str):
        super().__init__()
        self.auto = auto
        self.oneshot = AllReduce(
            mapping=auto.mapping, strategy=AllReduceStrategy.ONESHOT, dtype=dtype
        )
        self.max_rows = max_rows
        self.name = name
        self._logged: Set[int] = set()

    def uses_nccl_symmetric_memory_window(self) -> bool:
        return self.auto.uses_nccl_symmetric_memory_window()

    def forward(self, input: torch.Tensor, *args, **kwargs):
        rows = input.shape[0] if input.dim() > 1 else 1
        if rows > self.max_rows:
            return self.auto(input, *args, **kwargs)
        if rows not in self._logged:
            self._logged.add(rows)
            logger.info(
                f"[M3 dispatch] rank {self.auto.mapping.tp_rank}: S9b-VR-a drafter all-reduce ONESHOT taken at "
                f"{rows} rows ({self.name})"
            )
        return self.oneshot(input, *args, **kwargs)


def install_draft_oneshot_allreduce(draft: nn.Module, max_rows: int) -> int:
    """Wrap every row-parallel linear's AUTO all-reduce in the drafter (S9b-VR-a); returns how many were wrapped."""
    if max_rows <= 0:
        return 0
    n = 0
    for name, mod in draft.named_modules():
        ar = getattr(mod, "all_reduce", None)
        if (
            isinstance(mod, Linear)
            and mod.tp_mode == TensorParallelMode.ROW
            and isinstance(ar, AllReduce)
            and ar.strategy == AllReduceStrategy.AUTO
            and ar.mapping.tp_size > 1
        ):
            mod.all_reduce = DraftOneshotAllReduce(ar, mod.dtype, max_rows, name)
            n += 1
    return n


# Drafter items' defaults: S2 (prebuild below) stays off by default, because the draft length 3 set {S9b, S5, S2,
# FUSED_CAPTURE} lowered the K7 natural acceptance length measurably (mean -0.032, 95% CI [-0.059, -0.005]); S9b and
# S5 passed their own acceptance checks (below). S5 is on by default at this target (batch 1 at K7, see
# S5_MAX_BATCH_BY_K); S9b stays opt-in (drafter-only precision, not bitwise vs the default path). Opt in with
# TRTLLM_M31_ENABLE_<ID>=1; TRTLLM_M31_DISABLE_<ID>=1 still wins. FUSED_CAPTURE does not exist in this target.

# S9b (fused drafter all-reduce + norm): opt-in at this TP (TRTLLM_M31_ENABLE_DRAFT_AR_NORM_FUSED=1;
# TRTLLM_M31_DISABLE_DRAFT_AR_NORM_FUSED wins). Its fused all-reduce runs the one-shot path (S9b-VR-a's
# DraftOneshotAllReduce at the decode rows) instead of the untuned AUTO path. Its acceptance length matches the default
# path (mean -0.0012 of 3.604, 95% CI [-0.0047, +0.0024]). It stays opt-in because its decode-step gain on SM107 TP4 K7
# depends on the batch and context: faster at batch 1 and 128K, slower at batch 1 and 512K and at batch 2.
S9B_DEFAULT_ON = False

# S5 (window-capped drafter attention max_seq_len, prebuild below): on by default at this TP; kill switch
# TRTLLM_M31_DISABLE_S5=1 (TRTLLM_M31_ENABLE_S5=1 is accepted and changes nothing). The cap is against the served
# max_seq_len, so its decode-step saving at SM107 TP4 K7 batch 1 does not depend on the context; its acceptance length
# matches the default path (mean -0.0012 of 3.604, 95% CI [-0.0042, +0.0019]). At batch 2 it is slower, so the K7 bound
# is batch 1 (S5_MAX_BATCH_BY_K).
S5_DEFAULT_ON = True


# S5: the largest decode batch the window-capped max_seq_len applies to, by draft length; 0 = off. Per (TP, K) on
# VR: the largest batch at which the window-capped drafter attention is faster than the stock call at every batch up
# to it, in two replicates of the per-call microbench (L = 400K). TP4: K3 c1 / c2 / c4
# win -2.7 / -2.6 / -2.0 us per call, c8 loses +2.0; K7 c1 / c2 / c4 win per call, but in-model at K7 (the served
# draft length) batch 2 loses +39 / +49 us per step at 128K / 512K, so K7 stops at batch 1. K3 keeps its microbench
# bound (no in-model read at K3).
S5_MAX_BATCH_BY_K = {3: 4, 7: 1}

# S10 (lean drafter glue) is on only up to this draft length by default: bitwise at K3 on SM107; at K7 it diverged from
# request 5 before AR-DET (K7 OFF / OFF identical); K4..6 are unmeasured.
# S10_K7 (default ON) lifts the bound to every draft length: re-gated at K7 under AR-DET (SM107 TP4 K7 b1:
# outputs identical on 5 of 5 requests, decode step -6.3 us on the bitwise tail items, -3.6 us with TAILFOLD
# on top; set by the serving GEN env). Kill switch TRTLLM_M31_DISABLE_S10_K7=1 (S10 back to K <= 3); never set
# TRTLLM_M31_DISABLE_DRAFTER_LEAN_GLUE=1 for that (it also turns off TAIL_LEAN_WORKER). Read once, at import.
S10_K7 = not m31_disabled("S10_K7")
S10_MAX_DRAFT_LEN = 63 if S10_K7 else 3


# MXDRAFT is default OFF on TP4: the TP4 K7 c1 step was slower with it on in two c1 kill A/Bs (4,009.1 ->
# 4,040.7 us; 4,012.3 -> 4,036.4 us) while TP8 gains (-1.3%).
# TRTLLM_M31_ENABLE_DRAFTER_MXFP8=1 opts in; TRTLLM_M31_DISABLE_DRAFTER_MXFP8=1 still wins. Read once, at import.
MXDRAFT = m31_enabled("DRAFTER_MXFP8") and not m31_disabled("DRAFTER_MXFP8")
# DHEAD_MX4 (dspark_draft_head): the drafter head's S4 shard GEMM
# ([B x K, 6144] x [50016, 6144]^T) on a separate MXFP8 copy of the shard (FlashInfer CuTe DSL quantize + CuTe DSL
# MXFP8 GEMM at a pinned tactic, overridden by e.g. TRTLLM_M31_DHEAD_MX4_TACTIC=128x32-2x1-swap-nopf-k1, a malformed or
# unrunnable value failing the set-up) written straight into the S4 padded buffer at the logical N (its -inf tail never
# written), replacing the BF16 head GEMM (47.1-47.3 us) and the buffer copy (3.2 us) at the K7 batch-1 rows; any call
# it declines (another row count, another head) keeps the BF16 head: PADFOLD (dspark_markov_worker.PADFOLD) when on,
# else the GEMM + copy. One path writes per call. Not bitwise (drafter-only precision); adds 317.1 MB per GPU at
# prebuild (before the KV cache is sized). Natural acceptance length is unchanged within noise. Default ON at TP4; kill
# switch TRTLLM_M31_DISABLE_DHEAD_MX4=1 (TRTLLM_M31_ENABLE_DHEAD_MX4=1 is accepted and changes nothing). Read at import.
DHEAD_MX4_DEFAULT_ON = True
DHEAD_MX4 = (DHEAD_MX4_DEFAULT_ON or m31_enabled("DHEAD_MX4")) and not m31_disabled("DHEAD_MX4")
# HISLAYOUT (GV_C; drafter_mxfp8.route_his_layout): the reference drafter GEMV layout on this target's MXFP8 GEMV for
# the decode rows <= drafter_mxfp8.HIS_MAX_ROWS (qkv / o_proj / gate_up at the reference configs; SwiGLU + down fused).
# Default ON (drafter-only
# precision, gated on natural acceptance length); it takes precedence over MXDRAFT on the same
# projections. Kill switch TRTLLM_M31_DISABLE_DRAFTER_HISLAYOUT=1, read once at import.
HISLAYOUT = not m31_disabled("DRAFTER_HISLAYOUT")


# TAIL_FORK (dspark_tailfork): at a decode-only step whose fc takes FC_SHARD (the K7 verify rows at batch 1), the
# drafter's acceptance-independent input work -- the capture copies, the fc + all-gather + hidden norm and
# (TAIL_FORK_KV) the context K / V GEMM + k_norm of the DRAFTER_CTX_KV_STORE_FUSED store -- runs on an aux stream forked
# right after the verify head GEMM, beside the logits gather and the verify sampling, and joins where the stock path
# first reads the projected rows. TAIL_DEFER_CAPTURE: the capture layers' boundary residuals are kept by reference and
# copied into the capture buffer inside the fork, so no graph memcpy node breaks the PDL chain into the next layer's QKV
# GEMV. The same ops on the same operands, only on another stream and earlier: bitwise. TAIL_FORK / TAIL_FORK_KV default
# ON; kill switches TRTLLM_M31_DISABLE_TAIL_FORK (all three) / TRTLLM_M31_DISABLE_TAIL_FORK_KV, read at import.
# TAIL_DEFER_CAPTURE is opt-in on this target (TRTLLM_M31_ENABLE_TAIL_DEFER_CAPTURE=1;
# TRTLLM_M31_DISABLE_TAIL_DEFER_CAPTURE=1 still wins): FRONT_CAPFORK already takes the capture copies off the PDL chain
# onto the capture stream, so the default keeps that path.
TAIL_FORK = not m31_disabled("TAIL_FORK")
TAIL_FORK_KV = TAIL_FORK and not m31_disabled("TAIL_FORK_KV")
TAIL_DEFER_CAPTURE = (
    TAIL_FORK and m31_enabled("TAIL_DEFER_CAPTURE") and not m31_disabled("TAIL_DEFER_CAPTURE")
)
# TAIL_FC_GATHER (T14, inside the fork): the fc shards' all-gather as one multicast launch (catalog allgather_f32 on the
# bf16 [rows, 1536] shard viewed as fp32 [rows, 768]: the int32 words copied as is, in rank order along the last dim, so
# the bf16 view of the result is the NCCL all-gather's [rows, 6144] bit for bit) on its own workspace (not A1's, which
# runs at the same time on the main stream), instead of the NCCL ring all-gather + concat copy. Rows other than the
# compiled K7 c1 count keep the NCCL all-gather. Kill switch TRTLLM_M31_DISABLE_TAIL_FC_GATHER=1.
TAIL_FC_GATHER = TAIL_FORK and not m31_disabled("TAIL_FC_GATHER")
# TAIL_FORK_REFRESH (inside the fork): the drafter's block-table refresh (DRAFTER_BLOCK_TABLE_REFRESH_FUSED; reads the
# draft pool's block offsets uploaded before the replay, writes the drafter's tables and page counts, which nothing
# reads before the join) runs on the fork instead of on the critical path before the verify sampling. Kill switch
# TRTLLM_M31_DISABLE_TAIL_FORK_REFRESH=1.
TAIL_FORK_REFRESH = TAIL_FORK and not m31_disabled("TAIL_FORK_REFRESH")
# TAIL_FORCEVEC: a fractional forced acceptance (TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS, the benchmark knob) draws
# the step's forced counts with the one-launch m3_force_vec (the Eagle3 path's op) instead of the in-tree
# _apply_force_accepted_tokens' ~11 launches: the same counter bump, (counter, slot) hash, pool read, fp32 compare and
# clamp, written into the accepted counts in place (bitwise). Default ON; kill switch
# TRTLLM_M31_DISABLE_TAIL_FORCEVEC=1.
TAIL_FORCEVEC = not m31_disabled("TAIL_FORCEVEC")
# TAIL_LEAN_WORKER: at the draft lengths S10 (lean glue) does not cover (K7), the worker-side half of it only: the
# cached read-only arange tensors of the drafter inputs and the identity gather of the block outputs before the draft
# head (shift_label with block width K: the slot ids are arange(n * K), so the gather is a copy of the same rows). The
# draft model keeps its own copies (its lean-glue flag stays off). The same values in the same layouts: bitwise. Kill
# switch TRTLLM_M31_DISABLE_TAIL_LEAN_WORKER=1 (or TRTLLM_M31_DISABLE_DRAFTER_LEAN_GLUE=1).
TAIL_LEAN_WORKER = not m31_disabled("TAIL_LEAN_WORKER")
# TAILBK_F (TP2 / TP8's TAILBK_SPEC F): the DRAFTER_CTX_KV_STORE_FUSED store kernel also runs the drafter's k_norm
# weight multiply (catalog dspark_ctx_store.fold_k_norm_weight: the fp32 product rounded to bf16, bitwise torch's bf16
# multiply), so the store's prep -- TAIL_FORK_KV's ctx_store_prep on the fork, or the whole fused_ctx_store -- drops
# that launch (one elementwise kernel per step on the fork's chain). Bitwise the same cache bytes. Default ON; kill
# switch TRTLLM_M31_DISABLE_TAILBK_F=1 (TRTLLM_M31_DISABLE_TAIL_BOOKKEEP_SPEC=1, TP2 / TP8's switch for it, also turns
# it off), read at import.
TAILBK_F = not m31_disabled("TAILBK_F") and not m31_disabled("TAIL_BOOKKEEP_SPEC")


def _whole_store_folded(draft_model, projected: torch.Tensor) -> bool:
    """Whether the fused_ctx_store call that just stored ``projected`` ran the k_norm weight multiply in its kernel:
    the catalog store's own fold condition on the operands it used (its K is the rms_norm of the K / V GEMM's output,
    in the fused K / V weight's dtype, on ``projected``'s device)."""
    weight = getattr(draft_model, "_fused_kv_weight", None)
    w = getattr(draft_model, "_k_norm_stacked", None)
    if not _ctx_store._fold_w[0] or weight is None or w is None or projected.shape[0] == 0:
        return False
    return (
        weight.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and w.is_contiguous()
        and w.numel() == draft_model._num_attn_layers * draft_model._head_dim
        and w.device == projected.device
    )


def _stock_projection(draft) -> bool:
    """Whether ``draft`` projects with the in-tree DFlash project_target_hidden (fc or its FC_SHARD, then hidden_norm),
    the one _tf_project reproduces with the multicast gather."""
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    return type(draft).project_target_hidden is DFlashForCausalLM.project_target_hidden


class _TailStep:
    """One step's TAIL_FORK state (armed in arm_capture_norm, cleared when the worker's forward ends)."""

    __slots__ = (
        "spec",
        "draft",
        "attn",
        "rows",
        "total",
        "kept",
        "copied",
        "forked",
        "joined",
        "key",
        "proj",
        "prep",
        "used",
        "refreshed",
    )

    def __init__(self, spec, draft, attn, rows: int, total: int) -> None:
        self.spec = spec
        self.draft = draft
        self.attn = attn
        self.rows = rows  # the projected rows: generation requests x (K + 1)
        self.total = total  # the step's target tokens (get_hidden_states' bound)
        # [(capture index, boundary residual)] kept by reference; held until the step ends (after the join), so no
        # later allocation reuses their memory while the copy on the aux stream may still read it.
        self.kept = []
        self.copied = False  # the kept captures were copied into the buffer (in the fork or on the current stream)
        self.forked = False
        self.joined = False
        self.key = None  # view_key of the rows the fork projected
        self.proj = None  # the fork's project_target_hidden output
        self.prep = None  # the fork's ctx_store_prep
        self.used = False
        self.refreshed = (
            None  # (tables, num_seqs, attn_metadata) refreshed on the fork (TAIL_FORK_REFRESH)
        )


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


def s5_max_batch_for(num_draft_tokens: int) -> int:
    """K <= 3 takes the K3 entry, K >= 7 the K7 entry, and K in between the smaller of the two."""
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


# N3 A2 (switch TRTLLM_M31_DISABLE_STRICT_ACCEPT_FUSED): the target's strict acceptance and next-step input tokens in
# one launch each (ops/dspark/strict_accept), bitwise to the in-tree torch path. Read once, at import.
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
# TAILGLUE_B: on a decode-only step whose forward tail stores the step (TAIL_DSFOLD_FWD's
# M3DSparkStepFoldFwd), the next step's input tokens (N3 A2's next_new_tokens) and that tail's store + next-step
# staging (m3_fold_tail, this target's C++ foldTail) run as one CuTe DSL launch (kernels/m3_nnt_fold) right after the
# store wait the tail took: one launch and the gap between the two fewer per step. Integer copies of the same values
# into the same buffers: bitwise. The launch stands in for the fold tail whichever implementation the target picks
# (modeling.STEP_FOLD_TAIL_CPP: the C++ op or its Triton twin, the same writes). Default ON; kill switch
# TRTLLM_M31_DISABLE_TAILGLUE_B=1 (the two launches), read at import. Off with N3A
# (TRTLLM_M31_DISABLE_STRICT_ACCEPT_FUSED=1 restores the in-tree next_new_tokens on every step).
TAILGLUE_B = not m31_disabled("TAILGLUE_B") and not m31_disabled("STRICT_ACCEPT_FUSED")
# TAILFOLD (dspark_tailfold; each bitwise, read at import; only on the steps TAIL_FORK forks, which FC_SHARD's
# 8 rows bound to batch 1 at K7; TAILGLUE_A and MEMCPY86K default ON, MEMCPY4B default OFF):
# - TAILGLUE_A (the C++ op minimax_m31_tail_glue_a): on a forked step, the DRAFTER_CTX_KV_STORE_FUSED store, the
# DRAFTER_CTX_ADVANCE_FUSED advance +
#   read view and the DRAFTER_FORWARD_INDICES_FUSED forward indices in one launch after the join (kill switch
# TRTLLM_M31_DISABLE_TAILGLUE_A=1;
#   needs TAIL_FORK_KV and DRAFTER_CTX_KV_STORE_FUSED / C / D / E);
# - MEMCPY86K: on a forked step, the drafter's layer-0 residual stream starts in the DRAFTER_INPUTS_FUSED noise block
# instead of a
#   copy of it (kill switch TRTLLM_M31_DISABLE_MEMCPY86K=1; needs TAIL_FORK and DRAFTER_INPUTS_FUSED);
# - MEMCPY4B: on a forked step, the attention metadata's seq_lens copy is taken on the fork's stream instead of on the
#   sampling chain (needs TAIL_FORK). Off by default: its event edge into the PDL KV rewind costs more than the copy
#   it moves. Opt in with TRTLLM_M31_ENABLE_MEMCPY4B=1; TRTLLM_M31_DISABLE_MEMCPY4B=1 still wins.
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


# TP4_ANCESTRAL: the in-tree DSpark ancestral draft (DSparkWorker._samples_block_ancestrally: a Markov-head drafter
# under advanced draft sampling) returns the head's logits from _try_fused_draft_head_and_refine, so
# _refine_block_logits -- where this target's S4 chain (k3_markov + dsparkPrevEmbeddings + widen_add, the S4L tail)
# runs -- is never called, and the block is drawn by the per-position loop. OFF by default here, so this target keeps
# the S4 route; opt in with TRTLLM_M31_ENABLE_TP4_ANCESTRAL=1 (TRTLLM_M31_DISABLE_TP4_ANCESTRAL=1 still wins).
# Read at import.
TP4_ANCESTRAL = m31_enabled("TP4_ANCESTRAL") and not m31_disabled("TP4_ANCESTRAL")
# TP4_GUMBEL_DRAFT (TP2 stack-v2 / DEP DEP_GUMBEL_DRAFT): the family CUDA C++ Gumbel-max ancestral block sampler
# (ops/gumbel_draft, one fused draw per position) as the DSpark ancestral block override. Not additive with the guard:
# while it is installed the ancestral route is taken (the S4 route is not), and the block is drawn by it instead of the
# per-position loop. Distribution-exact vs the stock loop, not bitwise (another Philox stream; DEP natural AL +2.8 %).
# Unmeasured at TP4, so OFF until its solo arm beats S4 at b1 128K: opt in with TRTLLM_M31_ENABLE_TP4_GUMBEL_DRAFT=1;
# TRTLLM_M31_DISABLE_TP4_GUMBEL_DRAFT=1 wins. Flip TP4_GUMBEL_DEFAULT_ON after the read. Read at import.
TP4_GUMBEL_DEFAULT_ON = False
TP4_GUMBEL_DRAFT = (TP4_GUMBEL_DEFAULT_ON or m31_enabled("TP4_GUMBEL_DRAFT")) and not m31_disabled(
    "TP4_GUMBEL_DRAFT"
)
# Whether install_tp4_gumbel put the block sampler in (set once per process, before any draft).
_TP4_GUMBEL_INSTALLED: list = [False]
_TP4_ANCESTRAL_LOGGED: list = []


def install_tp4_gumbel() -> bool:
    """Once per process: install the family Gumbel block sampler as the ancestral block override when
    TP4_GUMBEL_DRAFT is on and its ops are built (logs its engagement line). Returns whether it is installed."""
    if not TP4_GUMBEL_DRAFT:
        logger.info(
            "[M3.1 TP4] TP4_GUMBEL_DRAFT: off (opt in with TRTLLM_M31_ENABLE_TP4_GUMBEL_DRAFT=1); the S4 route"
        )
        return False
    from ..ops import gumbel_draft as _gumbel

    _TP4_GUMBEL_INSTALLED[0] = bool(_gumbel.install("[M3.1 TP4]", "TP4_GUMBEL_DRAFT", True))
    return _TP4_GUMBEL_INSTALLED[0]


def tp4_ancestral_route() -> bool:
    """Whether this target takes the in-tree ancestral route: opted in (TP4_ANCESTRAL), or the Gumbel block sampler
    is installed (it only serves that route)."""
    return TP4_ANCESTRAL or _TP4_GUMBEL_INSTALLED[0]


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


class M3DSparkWorker(S4WorkerMixin, DSparkWorker):
    """The in-tree ``DSparkWorker`` with the calls this target's shell makes on its worker. W1: the one-launch Markov
    chain (``S4WorkerMixin``, switch ``TRTLLM_M31_DISABLE_DRAFT_S4``). N3 A2: the strict acceptance and the next-step
    input tokens in one launch each (switch ``TRTLLM_M31_DISABLE_STRICT_ACCEPT_FUSED``). N3 P / A1: the verify logits
    (switches ``TRTLLM_M31_DISABLE_VERIFY_LOGITS_COPY_FUSED`` / ``TRTLLM_M31_DISABLE_VERIFY_LOGITS_ALLGATHER_MC``).
    TAILBK R1 / Z1: the one-request Philox rows from the per-token window and no padding-row zeroing in captured batch-1
    graphs, bitwise (spec_metadata's tail_bookkeep; switch ``TRTLLM_M31_DISABLE_TAIL_BOOKKEEP``). Everything else is the
    in-tree path."""

    def _samples_block_ancestrally(self, draft_model, spec_metadata) -> bool:
        """TP4_ANCESTRAL (off by default): the in-tree ancestral draft only when opted in or with the Gumbel block
        sampler installed; otherwise the S4 route through _refine_block_logits, as measured on the original TP4
        route."""
        ancestral = tp4_ancestral_route()
        if not _TP4_ANCESTRAL_LOGGED:
            _TP4_ANCESTRAL_LOGGED.append(True)
            if not ancestral:
                route = "off (the S4 route through _refine_block_logits, as measured)"
            elif _TP4_GUMBEL_INSTALLED[0]:
                route = "on (the in-tree ancestral draft, block drawn by TP4_GUMBEL_DRAFT)"
            else:
                route = "on (the in-tree ancestral draft, per-position loop)"
            logger.info(f"M3.1 DSpark TP4_ANCESTRAL: {route}")
        return ancestral and super()._samples_block_ancestrally(draft_model, spec_metadata)

    def capture_hidden_states(
        self, spec_metadata, layer_id: int, hidden_states, residual=None
    ) -> None:
        """The in-tree DFlash capture: layer output (hidden + residual) into the static capture buffer. Under
        TAIL_DEFER_CAPTURE (opt-in on this target), on a step armed for the fork, a capture layer's boundary residual is
        kept instead and copied (the same bytes) inside the fork, or before the drafter reads the buffer when no fork
        runs. Otherwise a decode capture layer's plain copy (no residual addend, not the final hidden state) runs on the
        capture stream (FRONT_CAPFORK, joined by the target forward after its decoder stack)."""
        st = self._tf
        if (
            st is not None
            and TAIL_DEFER_CAPTURE
            and residual is None
            and spec_metadata is st.spec
            and not st.forked
            and not st.copied
            and self._tf_keepable(spec_metadata, layer_id, hidden_states)
        ):
            st.kept.append((spec_metadata._layer_to_idx[layer_id], hidden_states))
            return
        if (
            FRONT_CAPFORK
            and residual is None
            and layer_id != _FINAL_HIDDEN_LAYER_ID
            and hidden_states.is_cuda
        ):
            m31_capture_fork.fork_capture(spec_metadata, layer_id, hidden_states, "sm107_tp4")
            return
        spec_metadata.maybe_capture_hidden_states(layer_id, hidden_states, residual)

    def arm_capture_norm(
        self, attn_metadata, spec_metadata, lm_head: nn.Module, draft_model
    ) -> None:
        """No capture norm: DSpark's fc reads the raw captured layers. Arms (or clears) this step's TAIL_FORK."""
        self._tf = None
        if self._tf_aux is not None:
            self._tf = self._tf_arm(attn_metadata, spec_metadata, draft_model)
        return None

    # ------------------------------------------------------------------------------------------------ TAIL_FORK

    _tf = None  # this step's _TailStep, or None
    _tf_aux = None  # the fork's stream (prebuild, TAIL_FORK)
    _tf_capture_fast = (
        False  # the target's capture layers end at their fused boundary (fresh residual tensors)
    )
    _tf_logged = False
    _tf_gather = (
        None  # TAIL_FC_GATHER: (allgather_f32 workspace args, rows, fp32 shard columns) (prebuild)
    )
    _tfv_warm = False  # TAIL_FORCEVEC: the draw kernel compiled (prebuild)
    _tfv_logged = False

    def _tf_arm(self, attn_metadata, spec_metadata, draft_model):
        """This step's fork state when the step is one TAIL_FORK serves, else None. Host state and shapes only (the
        same on every rank, fixed per captured graph): a decode-only, non-ragged step of n requests x (K + 1) rows that
        FC_SHARD takes, a DFlash capture buffer, the context bound to the draft pool with the fused store installed."""
        if draft_model is None or spec_metadata is None or attn_metadata is None:
            return None
        if getattr(spec_metadata, "is_ragged_verify", False):
            return None
        if attn_metadata.num_contexts != 0:
            return None
        num_seqs = int(attn_metadata.num_seqs)
        if num_seqs <= 0 or int(getattr(spec_metadata, "runtime_draft_len", 0) or 0) <= 0:
            return None
        rows = num_seqs * int(self._draft_tokens_per_req)
        total = int(attn_metadata.num_tokens)
        buf = getattr(spec_metadata, "captured_hidden_states", None)
        if buf is None or not getattr(spec_metadata, "layers_to_capture", None) or rows != total:
            return None
        if total > buf.shape[0] or getattr(draft_model, "lilicorr", None) is not None:
            return None
        if getattr(draft_model, "_dflash_fc_split", None) is not None:
            return None
        shard = getattr(draft_model, "_dflash_fc_shard", None)
        fc = getattr(draft_model, "fc", None)
        if shard is None or fc is None or not shard.applies(fc, rows):
            return None
        if not (
            getattr(self, "_ctx_buf_inited", False)
            and getattr(self, "_ctx_paged", False)
            and getattr(self, "_ctx_block_tables", None) is not None
        ):
            return None
        return _TailStep(spec_metadata, draft_model, attn_metadata, rows, total)

    def _tf_keepable(self, spec_metadata, layer_id: int, hidden_states) -> bool:
        """Whether this capture can be kept by reference: a capture layer of the fast decode path (its boundary
        all-reduce returns a fresh residual nothing writes afterwards), the buffer's dtype and width, every row fits."""
        if not self._tf_capture_fast or not isinstance(hidden_states, torch.Tensor):
            return False
        idx = getattr(spec_metadata, "_layer_to_idx", {}).get(layer_id)
        buf = spec_metadata.captured_hidden_states
        return (
            idx is not None
            and hidden_states.dim() == 2
            and hidden_states.is_contiguous()
            and hidden_states.dtype == buf.dtype
            and hidden_states.device == buf.device
            and hidden_states.shape[1] == spec_metadata.hidden_size
            and hidden_states.shape[0] <= buf.shape[0]
        )

    def _tf_launch(self) -> None:
        """Fork this step's drafter input work onto the aux stream (after everything queued on the current stream)."""
        st = self._tf
        if st is None or st.forked:
            return
        spec, draft = st.spec, st.draft
        aux = self._tf_aux
        aux.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(aux):
            # MEMCPY4B: the seq_lens copy of this step's prepare_for_spec_dec, first on the fork.
            self._tf4b = (
                (st, _tfold.take_seq_lens_snapshot(st.attn, spec)) if self._tf4b_on else None
            )
            buf = spec.captured_hidden_states
            if st.kept and not st.copied:
                copy_captures(
                    buf,
                    st.kept[0][1].shape[0],
                    spec.hidden_size,
                    spec.num_capture_layers,
                    st.kept,
                )
                st.copied = True
            hs = spec.get_hidden_states(st.total)
            gen = hs[0 : st.rows]
            gen = gen.reshape(-1, gen.shape[-1])
            proj = self._tf_project(draft, gen)
            prep = None
            kv_buf = getattr(self, "_ctx_kv_buf", None)
            if TAIL_FORK_KV and kv_buf is not None:
                first = kv_buf if isinstance(kv_buf, torch.Tensor) else kv_buf[0]
                if proj.dtype == first.dtype:
                    prep = ctx_store_prep(draft, proj)
            attn = st.attn
            tables = getattr(self, "_ctx_block_tables", None)
            if TAIL_FORK_REFRESH and attn is not None and tables is not None:
                num_seqs = int(attn.num_seqs)
                if super()._refresh_ctx_block_tables(attn, num_seqs):
                    st.refreshed = (tables, num_seqs, attn)
        st.forked = True
        st.key = view_key(gen)
        st.proj = proj
        st.prep = prep
        if not self._tf_logged:
            self._tf_logged = True
            logger.info(
                f"[M3 dispatch] TAIL_FORK taken: drafter fc{' + ctx K/V prep' if prep is not None else ''}"
                f"{' + block-table refresh' if st.refreshed is not None else ''} on the aux stream after the verify "
                f"head ({st.rows} rows, captures deferred: {TAIL_DEFER_CAPTURE}, fc gather: "
                f"{'multicast' if self._tf_gather_takes(st.rows) else 'NCCL'}) on rank "
                f"{getattr(self.mapping, 'tp_rank', None)}"
            )

    def _tf_gather_takes(self, rows: int) -> bool:
        g = self._tf_gather
        return g is not None and rows == g[1]

    def _tf_project(self, draft, hidden_states):
        """``draft.project_target_hidden(hidden_states)``; under TAIL_FC_GATHER its FC_SHARD path with the multicast
        all-gather (the same partial GEMM, the same gathered bytes, the same hidden norm)."""
        shard = getattr(draft, "_dflash_fc_shard", None)
        rows = hidden_states.shape[0]
        if (
            self._tf_gather_takes(rows)
            and shard is not None
            and getattr(draft, "_dflash_fc_split", None) is None
            and shard.applies(draft.fc, rows)
            and _stock_projection(draft)
        ):
            ws, _, cols = self._tf_gather
            x = hidden_states.to(draft.fc.weight.dtype)  # project_target_hidden's first line
            part = shard.partial(x)
            if (
                part.dtype == torch.bfloat16
                and part.is_contiguous()
                and part.shape == (rows, 2 * cols)
                and part.data_ptr() % 16 == 0
                and _allgather_cast.is_compiled(
                    rows, cols, ws["world"], device=part.device, fp32=True
                )
            ):
                full = _allgather_cast.allgather_f32(part.view(torch.float32), **ws)
                return draft.hidden_norm(full.view(torch.bfloat16))
            return draft.hidden_norm(shard.gather(part))
        return draft.project_target_hidden(hidden_states)

    def _refresh_ctx_block_tables(self, attn_metadata, num_seqs: int) -> bool:
        """The in-tree refresh, unless this step's fork already refreshed the same tables for the same step."""
        st = self._tf
        if st is not None and st.refreshed is not None:
            tables, n, attn = st.refreshed
            if (
                attn is attn_metadata
                and n == int(num_seqs)
                and tables is getattr(self, "_ctx_block_tables", None)
            ):
                return True
        return super()._refresh_ctx_block_tables(attn_metadata, num_seqs)

    def _tf_join(self, st: _TailStep) -> None:
        """The current stream waits for the fork (once). Outside capture, the fork's outputs read on the current stream
        are recorded on it, so their blocks are not handed to later aux work while those reads are queued."""
        if not st.forked or st.joined:
            return
        main = torch.cuda.current_stream()
        main.wait_stream(self._tf_aux)
        st.joined = True
        if not torch.cuda.is_current_stream_capturing():
            for t in (
                st.proj,
                *(() if st.prep is None else (st.prep.kv_flat, st.prep.k)),
            ):
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    t.record_stream(main)

    def _tf_flush(self, st: _TailStep) -> None:
        """Write kept captures into the buffer on the current stream (no fork ran): the stock copies, later."""
        if st.kept and not st.copied:
            spec = st.spec
            copy_captures(
                spec.captured_hidden_states,
                st.kept[0][1].shape[0],
                spec.hidden_size,
                spec.num_capture_layers,
                st.kept,
            )
            st.copied = True

    def prepare_1st_drafter_inputs(self, *args, **kwargs):
        """``_tf_prepare_1st_drafter_inputs``; a store TAILGLUE_A deferred is launched before the inputs return (the
        drafter reads the cache), and on a forked step MEMCPY86K names the step's DRAFTER_INPUTS_FUSED noise block to
        the drafter."""
        try:
            inputs = self._tf_prepare_1st_drafter_inputs(*args, **kwargs)
        finally:
            if self._tg is not None:
                self._tg.flush()
        st = self._tf
        if self._tf86_on and st is not None and st.forked:
            draft = kwargs.get("draft_model", args[7] if len(args) > 7 else None)
            self._tf86_named = _tfold.own_noise(draft, inputs, self._tfold_inputs)
            self._tf86_step = _tfold.capturing()
            self._tf86_capt = self._tf86_named and self._tf86_step
        return inputs

    def _tf_prepare_1st_drafter_inputs(self, *args, **kwargs):
        """The in-tree first drafter inputs; on a forked step the drafter's project_target_hidden returns the fork's
        rows (after the join) for exactly the rows the fork projected."""
        st = self._tf
        if st is None:
            return super().prepare_1st_drafter_inputs(*args, **kwargs)
        if not st.forked:
            self._tf_flush(st)
            return super().prepare_1st_drafter_inputs(*args, **kwargs)
        draft = kwargs.get("draft_model", args[7] if len(args) > 7 else None)
        if draft is not st.draft:
            self._tf_join(st)
            return super().prepare_1st_drafter_inputs(*args, **kwargs)
        stock = type(draft).project_target_hidden

        def project(hidden_states, _st=st, _draft=draft, _stock=stock):
            self._tf_join(_st)
            if not _st.used and view_key(hidden_states) == _st.key:
                _st.used = True
                return _st.proj
            return _stock(_draft, hidden_states)

        draft.project_target_hidden = project
        try:
            return super().prepare_1st_drafter_inputs(*args, **kwargs)
        finally:
            del draft.project_target_hidden
            self._tf_join(st)

    def _tf_ctx_store(
        self,
        draft_model,
        projected,
        positions,
        num_accepted,
        request_rows,
        cols,
        block_table,
        kv_buf,
        tokens_per_request,
        *,
        capacity_counts=None,
        max_col=None,
    ):
        """DRAFTER_CTX_KV_STORE_FUSED's fused store; on a forked step whose fork prepared these projected rows, only its
        store kernel."""
        st = self._tf
        if st is not None and st.prep is not None and st.joined:
            # TAILGLUE_A: the same store, its kernel deferred into the advance's launch.
            done = (self._tg.store if self._tg is not None else ctx_store_launch)(
                st.prep,
                draft_model,
                projected,
                positions,
                num_accepted,
                request_rows,
                cols,
                block_table,
                kv_buf,
                tokens_per_request,
                capacity_counts=capacity_counts,
                max_col=max_col,
            )
            if done is not None:
                if done and st.prep.fold:
                    self._tailbk_f_log("fork prep", projected.shape[0])
                return done
        done = fused_ctx_store(
            draft_model,
            projected,
            positions,
            num_accepted,
            request_rows,
            cols,
            block_table,
            kv_buf,
            tokens_per_request,
            capacity_counts=capacity_counts,
            max_col=max_col,
        )
        if done and _whole_store_folded(draft_model, projected):
            self._tailbk_f_log("whole store", projected.shape[0])
        return done

    # The store kinds ("fork prep", "whole store") whose TAILBK_F line this rank logged.
    _tailbk_f_logged: frozenset = frozenset()

    def _tailbk_f_log(self, where: str, rows: int) -> None:
        """The TAILBK_F engagement line: once per rank and store kind (``where``: the forked step's prep, or the whole
        store of a step the fork does not serve, e.g. the first decode step), on the first store of that kind that
        left the k_norm weight multiply to the store kernel (host only; a captured step logs it while it is
        captured)."""
        if where in self._tailbk_f_logged:
            return
        self._tailbk_f_logged = self._tailbk_f_logged | {where}
        logger.info(
            f"[M3 dispatch] TAILBK_F taken: k_norm weight multiply in the ctx store kernel ({where}, {rows} rows) on "
            f"rank {getattr(getattr(self, 'mapping', None), 'tp_rank', None)}"
        )

    # ------------------------------------------------------------------------------------------------- TAILFOLD

    # TAILGLUE_A: the dspark_tailfold.TailGlue whose hooks DRAFTER_CTX_ADVANCE_FUSED / DRAFTER_FORWARD_INDICES_FUSED
    # run (prebuild), or None.
    _tg = None
    # This step's DRAFTER_INPUTS_FUSED outputs (TAILGLUE_A's query positions, MEMCPY86K's noise block).
    _tfold_inputs = None
    _tf86_on = False  # MEMCPY86K set up (prebuild)
    _tf86_named = False  # MEMCPY86K: this step named its noise block to the drafter
    _tf86_capt = False  # MEMCPY86K: ... while a CUDA graph is captured (the engagement line reports that path)
    _tf86_step = (
        None  # MEMCPY86K: this step is forked (whether a CUDA graph is captured), else None
    )
    _tf86_logged = False
    _tf86_logged_eager = False  # the first forked step run eagerly after the captured take (logged)
    _tf86_logged_stock = False  # the first forked step captured with the copy (logged)
    _tf4b_on = False  # MEMCPY4B set up (prebuild)
    _tf4b = None  # MEMCPY4B: (this step's _TailStep, its fork copy of seq_lens: dspark_tailfold.SeqLensSnapshot)
    _tf4b_logged = False

    def _tfold_hook_inputs(self, inputs_fn):
        """The DRAFTER_INPUTS_FUSED hook, recording its outputs per step when TAILGLUE_A or MEMCPY86K uses them."""
        if not (_tfold.TAILGLUE_A or _tfold.MEMCPY86K):
            return inputs_fn
        return _tfold.recording_inputs(self, inputs_fn)

    def _tfold_hook_advance(self, advance, store_forked: bool, indices_on: bool, inputs_on: bool):
        """The DRAFTER_CTX_ADVANCE_FUSED hook: TAILGLUE_A's (its fused launch, else ``advance``) where it can serve the
        forked store, else ``advance`` itself."""
        self._tg = None
        if _tfold.TAILGLUE_A and store_forked and indices_on and inputs_on:
            self._tg = _tfold.TailGlue(
                advance, forward_indices, getattr(getattr(self, "mapping", None), "tp_rank", None)
            )
            return self._tg.advance
        return advance

    def _prepare_attn_metadata_for_dflash(self, attn_metadata, spec_metadata):
        """MEMCPY4B: on a forked step, the stock prepare with the fork's seq_lens copy swapped in; else the stock."""
        taken, self._tf4b = self._tf4b, None
        snap = (
            taken[1]
            if taken is not None and taken[0] is self._tf and self._tf is not None
            else None
        )
        if _tfold.install_seq_lens_snapshot(snap, attn_metadata, spec_metadata):
            if not self._tf4b_logged and _tfold.capturing():
                self._tf4b_logged = True
                logger.info(
                    f"[M3 dispatch] MEMCPY4B taken: seq_lens copy for the draft forward taken on the fork's stream "
                    f"({attn_metadata._seq_lens_cuda.numel()} requests) on rank "
                    f"{getattr(getattr(self, 'mapping', None), 'tp_rank', None)}"
                )
            return
        super()._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)

    def _tfold_end_step(self, draft_model) -> None:
        """Step end: TAILGLUE_A's per-step state cleared (a deferred store launched), MEMCPY86K's engagement read from
        the drafter (and its name of the block dropped), MEMCPY4B's copy released."""
        if self._tg is not None:
            self._tg.end_step()
        self._tfold_inputs = None
        self._tf4b = None
        step, self._tf86_step = self._tf86_step, None
        taken = bool(
            self._tf86_named
            and draft_model is not None
            and getattr(draft_model, "_dflash_owned_noise_taken", False)
        )
        if self._tf86_named and draft_model is not None:
            if taken and self._tf86_capt and not self._tf86_logged:
                self._tf86_logged = True
                logger.info(
                    "[M3 dispatch] MEMCPY86K taken: drafter residual stream starts in the DRAFTER_INPUTS_FUSED "
                    f"noise block, no copy on rank {getattr(getattr(self, 'mapping', None), 'tp_rank', None)}"
                )
            draft_model._dflash_owned_noise = None
            draft_model._dflash_owned_noise_taken = False
        if step is not None:
            self._tf86_note(
                step,
                None
                if taken
                else (
                    "the drafter forward did not take the named block"
                    if self._tf86_named
                    else "the step's noise block is not the DRAFTER_INPUTS_FUSED one"
                ),
            )
        self._tf86_named = False
        self._tf86_capt = False

    def _tf86_note(self, capt: bool, why: Optional[str]) -> None:
        """MEMCPY86K's two lines beside the captured take's (once per rank each) for a forked step: the first one run
        eagerly after the captured take, and the first one captured with the copy. ``why`` is the reason the drafter's
        residual stream did not start in the noise block, None when it did."""
        rank = getattr(getattr(self, "mapping", None), "tp_rank", None)
        if capt and why is not None and not self._tf86_logged_stock:
            self._tf86_logged_stock = True
            logger.info(
                f"[M3 dispatch] MEMCPY86K stock path while a CUDA graph is captured ({why}) on rank {rank}"
            )
        elif not capt and self._tf86_logged and not self._tf86_logged_eager:
            self._tf86_logged_eager = True
            what = "taken" if why is None else f"stock path ({why})"
            logger.info(
                f"[M3 dispatch] MEMCPY86K at an eager step after the capture: {what} on rank {rank}"
            )

    def _tf_prebuild(self, lm_head, draft_model) -> None:
        """TAIL_FORK's stream and TAIL_FORCEVEC's draw kernel, outside capture, every rank (rank-local)."""
        rank = getattr(getattr(self, "mapping", None), "tp_rank", None)
        if TAIL_FORK and draft_model is not None and torch.cuda.is_available():
            from . import modeling as _modeling  # imported here: modeling imports this module

            # Kept captures must be the fast decode layer's boundary residuals (fresh tensors nothing writes later).
            self._tf_capture_fast = bool(
                getattr(_modeling, "CAPTURE_LAYERS_FAST", False)
                and getattr(_modeling, "CAPTURE_FUSED_BOUNDARY", False)
            )
            self._tf_aux = torch.cuda.Stream()
            self._tf_gather = self._tf_prebuild_gather(draft_model) if TAIL_FC_GATHER else None
        logger.info(
            f"M3.1 DSpark TAIL_FORK: {'on' if self._tf_aux is not None else 'off'} (KV prep "
            f"{'on' if TAIL_FORK_KV and self._tf_aux is not None else 'off'}, deferred captures "
            f"{'on' if TAIL_DEFER_CAPTURE and self._tf_capture_fast else 'off'}, refresh "
            f"{'on' if TAIL_FORK_REFRESH and self._tf_aux is not None else 'off'}, fc gather "
            f"{'multicast' if self._tf_gather is not None else 'NCCL'}) on rank {rank}"
        )
        warm = (
            self._tfv_prebuild(lm_head.weight.device)
            if lm_head is not None and getattr(lm_head, "weight", None) is not None
            else False
        )
        logger.info(
            f"M3.1 DSpark TAIL_FORCEVEC: one-launch forced counts {'on' if warm else 'off'} on rank {rank}"
        )

    def _tf_prebuild_gather(self, draft_model):
        """TAIL_FC_GATHER's workspace and compile (collective: every rank of the TP group runs it at the same point of
        prebuild, and every rank keeps it only if every rank compiled): (workspace args, rows, fp32 columns) or None."""
        mapping = getattr(self, "mapping", None)
        shard = getattr(draft_model, "_dflash_fc_shard", None)
        if mapping is None or mapping.tp_size <= 1 or mapping.enable_attention_dp or shard is None:
            return None
        k = int(getattr(getattr(self, "spec_config", None), "max_draft_len", 0) or 0)
        rows = k + 1  # the K + 1 verify rows at batch 1 (FC_SHARD's MAX_ROWS bounds them)
        w = shard.w_shard
        if k <= 0 or w.dtype != torch.bfloat16 or w.shape[0] % 8 != 0:
            return None
        cols = w.shape[0] // 2
        if not _allgather_cast.precompile_f32(
            mapping, cols, (rows,), w.device, agree=self._a1_all_ok
        ):
            logger.warning(
                "M3.1 DSpark TAIL_FC_GATHER: a compile failed on some rank; the fc keeps NCCL"
            )
            return None
        return (_allgather_cast.workspace_args(mapping, cols), rows, cols)

    _nnt_fold_target = None  # TAILGLUE_B: the model's M3DSparkStepFoldFwd (bind_tail_fold), or None
    _nnt_fold_step = False  # TAILGLUE_B: this step is decode-only (its forward tail stores it)
    _nnt_fold_logged = False

    def bind_tail_fold(self, fold) -> None:
        """TAILGLUE_B: the target's in-forward step fold, whose tail store the step's next_new_tokens launch can do."""
        self._nnt_fold_target = fold

    def _forward_impl(self, *args, **kwargs):
        """The S4 / DFlash step; a forked step's aux stream is always joined (and kept captures always written)
        before the step returns, and its state does not outlive the step. TAILBK R1's offsets window also lives for
        one step only (set by its accept call, read by its draft block), so a step whose accept did not stash one
        recomputes it. This is the worker's only _forward_impl override: both step-scoped states are handled here."""
        attn = kwargs.get("attn_metadata", args[4] if len(args) > 4 else None)
        self._nnt_fold_step = attn is not None and int(getattr(attn, "num_contexts", 1)) == 0
        self._tailbk_win = None
        try:
            return super()._forward_impl(*args, **kwargs)
        finally:
            self._tailbk_win = None
            self._nnt_fold_step = False
            st = self._tf
            if st is not None:
                if st.forked:
                    self._tf_join(st)
                else:
                    self._tf_flush(st)
            self._tf = None
            self._tfold_end_step(kwargs.get("draft_model", args[6] if len(args) > 6 else None))

    # ------------------------------------------------------------------------------------------- TAIL_FORCEVEC

    def _apply_force_accepted_tokens(
        self, num_accepted_tokens, num_contexts, runtime_draft_len: int, spec_metadata=None
    ):
        """TAIL_FORCEVEC: the in-tree fractional forced acceptance in one launch (m3_force_vec), else the in-tree
        lines."""
        if TAIL_FORCEVEC and self._tfv_warm and self.force_num_accepted_tokens != 0.0:
            done = self._tfv_draw(
                num_accepted_tokens, num_contexts, runtime_draft_len, spec_metadata
            )
            if done is not None:
                return done
        return super()._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, runtime_draft_len, spec_metadata=spec_metadata
        )

    def _tfv_draw(self, num_accepted_tokens, num_contexts, runtime_draft_len: int, spec_metadata):
        """The draw kernel's call where it reproduces the in-tree draw path (fractional part > 0 and a base count below
        the maximum: one counter bump, then every generation request's count), else None."""
        if spec_metadata is not None and (
            spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        ):
            return None  # the warmup: the in-tree method returns unchanged counts
        force = self.force_num_accepted_tokens
        int_part = int(force)
        frac = force - int_part
        max_total = runtime_draft_len + 1
        base_total = min(int_part + 1, max_total)
        if not (frac > 0.0 and base_total < max_total):
            return None
        pool = getattr(self, "_force_accept_rng_pool", None)
        counter = getattr(self, "_force_accept_rng_counter", None)
        if pool is None or counter is None:
            return None
        num_gens = num_accepted_tokens.shape[0] - num_contexts
        if (
            num_accepted_tokens.dim() != 1
            or num_accepted_tokens.dtype != torch.int32
            or not num_accepted_tokens.is_contiguous()
            or num_gens < 1
            or num_gens > 64
        ):
            return None
        m3_force_vec(
            pool,
            counter,
            num_accepted_tokens[num_contexts:],
            num_gens,
            base_total,
            max_total,
            float(frac),
            _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
            _FORCE_ACCEPT_RNG_SLOT_STRIDE,
            _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
            max(16, 1 << (num_gens - 1).bit_length()),
        )
        if not self._tfv_logged:
            self._tfv_logged = True
            logger.info(
                f"[M3 dispatch] TAIL_FORCEVEC taken: forced accepted counts in one launch ({num_gens} requests) on "
                f"rank {getattr(self.mapping, 'tp_rank', None)}"
            )
        return num_accepted_tokens

    def _tfv_prebuild(self, device) -> bool:
        """Build the forced-acceptance pool (the in-tree state, counter at zero) and compile the draw kernel for every
        block width, outside capture; a scratch counter keeps the real draw sequence untouched."""
        force = getattr(self, "force_num_accepted_tokens", 0.0)
        if not TAIL_FORCEVEC or force - int(force) <= 0.0:
            return False
        self._ensure_force_accept_rng_state(device)
        scratch = torch.zeros(1, dtype=torch.int64, device=device)
        dummy = torch.empty(64, dtype=torch.int32, device=device)
        for block in (16, 32, 64):
            m3_force_vec(
                self._force_accept_rng_pool,
                scratch,
                dummy,
                block,
                1,
                2,
                0.5,
                _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
                _FORCE_ACCEPT_RNG_SLOT_STRIDE,
                _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
                block,
            )
        torch.cuda.synchronize(device)
        self._tfv_warm = True
        return True

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
        # TAIL_LEAN_WORKER: the worker-side half of S10 where S10 is off by draft length (the draft model's flag stays
        # off).
        if (
            not lean
            and TAIL_LEAN_WORKER
            and not m31_disabled("DRAFTER_LEAN_GLUE")
            and draft_model is not None
        ):
            self._lean_glue = True
            logger.info(
                f"M3.1 DSpark TAIL_LEAN_WORKER: worker-side lean glue on (K{k}) on rank {rank}"
            )
        # S5: window-capped max_seq_len for the drafter's sliding-window decode (not bitwise: the TRTLLM-Gen tile / KV
        # split changes, not the keys read). On by default where S5_DEFAULT_ON (its own gate passed, see above);
        # TRTLLM_M31_ENABLE_S5=1 opts in elsewhere; TRTLLM_M31_DISABLE_S5 still wins.
        s5_on = (S5_DEFAULT_ON or m31_enabled("S5")) and not m31_disabled("S5")
        s5_max_batch = s5_max_batch_for(k) if s5_on else 0
        self.enable_window_capped_decode(draft_model, s5_max_batch)
        # The context arena spans every seq slot (CUDA graph metadata narrows max_num_requests to its bucket).
        self.enable_full_slot_arena()
        # S2: the drafter's replicated fc split row-parallel with one fp32 all-reduce (not bitwise: the partial-sum
        # order changes). Collective. OFF by default : opt in with TRTLLM_M31_ENABLE_S2=1;
        # TRTLLM_M31_DISABLE_S2 still wins.
        s2 = self.enable_fc_tp_split(draft_model, m31_enabled("S2") and not m31_disabled("S2"))
        split = getattr(draft_model, "_dflash_fc_split", None)
        ar = split[3] if split is not None else None
        ar_path = (
            "MNNVL"
            if getattr(ar, "mnnvl_allreduce", None) is not None
            else getattr(ar, "strategy", None)
        )
        # FC_SHARD (bitwise; collective: every rank builds or every rank declines). S2's split, when on, wins. No
        # drafter (a shell worker): nothing to set.
        if draft_model is not None:
            draft_model._dflash_fc_shard = (
                FcShard.build(draft_model.fc, self.mapping)
                if FC_SHARD and not s2 and getattr(draft_model, "fc", None) is not None
                else None
            )
        logger.info(
            f"M3.1 DSpark S2: fc TP split {'on' if s2 else 'off'} on rank {rank} (all-reduce {ar_path})"
        )
        logger.info(
            f"M3.1 DSpark S5: window-capped draft decode for batches <= {s5_max_batch} on rank {rank}"
        )
        logger.info(f"M3.1 DSpark S10: lean drafter glue {'on' if lean else 'off'} on rank {rank}")
        # DRAFT_AR_NORM_FUSED: each drafter layer's all-reduce fused with the add + RMSNorm after it (not bitwise: the
        # sum is not rounded before the add). Kill switch TRTLLM_M31_DISABLE_DRAFT_AR_NORM_FUSED, read at import.
        s9b = self.enable_fused_ar_norm(
            draft_model,
            (S9B_DEFAULT_ON or m31_enabled("DRAFT_AR_NORM_FUSED"))
            and not m31_disabled("DRAFT_AR_NORM_FUSED"),
        )
        logger.info(
            f"M3.1 DSpark S9b: fused drafter all-reduce + norm {'on' if s9b else 'off'} on rank {rank}"
        )
        # DRAFTER_CTX_KV_STORE_FUSED: the drafter's context K / V epilogue + accept mask + paged store in one launch
        # after the K / V GEMM (bitwise). Kill switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all fused drafter glue) /
        # TRTLLM_M31_DISABLE_DRAFTER_CTX_KV_STORE_FUSED, read at import.
        s10c_a = self.enable_fused_ctx_store(
            None
            if m31_disabled("DRAFTER_FUSED_GLUE") or m31_disabled("DRAFTER_CTX_KV_STORE_FUSED")
            else (self._tf_ctx_store if TAIL_FORK_KV else fused_ctx_store)
        )
        logger.info(
            f"M3.1 DSpark DRAFTER_CTX_KV_STORE_FUSED: fused drafter ctx store {'on' if s10c_a else 'off'} "
            f"on rank {rank}"
        )
        # TAILBK_F: the k_norm weight multiply in that store kernel (process-wide, set the same on every rank).
        tailbk_f = fold_k_norm_weight(bool(s10c_a) and TAILBK_F)
        logger.info(
            f"M3.1 DSpark TAILBK_F: k_norm weight multiply in the ctx store kernel {'on' if tailbk_f else 'off'} "
            f"on rank {rank}"
        )
        # Fused drafter inputs: the drafter's step inputs (bonus, positions, noise block) in one launch (bitwise:
        # integers and row copies). Kill switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all fused drafter glue) /
        # TRTLLM_M31_DISABLE_DRAFTER_INPUTS_FUSED, read at import.
        s10c_c = self.enable_fused_drafter_inputs(
            None
            if m31_disabled("DRAFTER_FUSED_GLUE") or m31_disabled("DRAFTER_INPUTS_FUSED")
            else self._tfold_hook_inputs(s10c_c_drafter_inputs)
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
            else self._tfold_hook_advance(
                functools.partial(advance_ctx_and_read_view, parallel=not m31_disabled("ADV_PAR")),
                bool(s10c_a) and TAIL_FORK_KV,
                not (
                    m31_disabled("DRAFTER_FUSED_GLUE")
                    or m31_disabled("DRAFTER_FORWARD_INDICES_FUSED")
                ),
                bool(s10c_c),
            )
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
        # DRAFTER_FORWARD_INDICES_FUSED: the draft model's paged block-decode index prelude in one launch (bitwise,
        # integer). Kill switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all fused drafter glue) /
        # TRTLLM_M31_DISABLE_DRAFTER_FORWARD_INDICES_FUSED, read at import.
        s10c_e = self.enable_fused_forward_indices(
            draft_model,
            None
            if m31_disabled("DRAFTER_FUSED_GLUE") or m31_disabled("DRAFTER_FORWARD_INDICES_FUSED")
            else (self._tg.indices if self._tg is not None else forward_indices),
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
        # DRAFT_SWIGLU (exact): the drafter MLPs' SwiGLU on the family C++ op, not the Triton silu_and_mul.
        # Before MXDRAFT, so a later activation wrapper wraps this one.
        swiglu = install_draft_swiglu(draft_model)
        logger.info(f"M3.1 DSpark DRAFT_SWIGLU: {native_swiglu.engagement(swiglu)} on rank {rank}")
        # HISLAYOUT (GV_C): before MXDRAFT, which skips projections already routed. After DRAFT_SWIGLU, so the
        # fused MLP's fallback runs the native SwiGLU.
        hl = (0, 0)
        if HISLAYOUT and k > 0 and draft_model is not None:
            hl = route_his_layout(draft_model, [b * k for b in range(1, MAX_FAST_BATCH + 1)])
        if hl[0] or hl[1]:
            logger.info(
                f"[M31_DRAFTER_HISLAYOUT] engaged: {hl[0]} projections + {hl[1]} fused SwiGLU-down MLPs "
                f"on the MXFP8 GEMV (reference configs), rows <= 8, rank {rank}"
            )
        else:
            logger.info(
                "M3.1 DSpark HISLAYOUT: "
                f"{'off (TRTLLM_M31_DISABLE_DRAFTER_HISLAYOUT=1)' if not HISLAYOUT else 'not taken'} "
                f"on rank {rank}"
            )
        # MXDRAFT: the drafter's qkv / o / gate_up / down GEMMs on this target's MXFP8 GEMV for the decode passes'
        # rows (B x K, B <= MAX_FAST_BATCH, up to the GEMV's own row bound), from the exact MXFP8 form of the shipped
        # weights (drafter-only precision, gated on natural AL). Default off on TP4 (``MXDRAFT``).
        mx = (
            0
            if not MXDRAFT or k <= 0
            else route_mx_linears(draft_model, [b * k for b in range(1, MAX_FAST_BATCH + 1)])
        )
        logger.info(f"M3.1 DSpark MXDRAFT: MXFP8 drafter GEMV on {mx} projections on rank {rank}")
        if k > 0:
            logger.info(routes_line(draft_model, (k, 2 * k), rank))
        self._tf_prebuild(lm_head, draft_model)
        self.prebuild_s4(lm_head, draft_model)
        # DHEAD_MX4: the drafter head's S4 shard GEMM on an MXFP8 copy (collective agreement when on).
        self._prebuild_dhead4(lm_head)
        # N3 A2 compiles (rank-local), then the N3 A1 workspace and compiles (collective), outside capture.
        if not N3A_DISABLED and lm_head is not None:
            _strict_accept.precompile((self.max_draft_len,), lm_head.weight.device)
        self._a1 = self._prebuild_a1(lm_head) if lm_head is not None else None
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
        # TAILGLUE_B: the one-launch next_new_tokens + fold tail (rank-local compiles, outside capture).
        nnt_fold = TAILGLUE_B and self._nnt_fold_target is not None and lm_head is not None
        if nnt_fold:
            m3_nnt_fold.precompile(lm_head.weight.device, (torch.int32, torch.int64))
        logger.info(
            f"M3.1 DSpark TAILGLUE_B: next_new_tokens + forward-tail fold in one launch {'on' if nnt_fold else 'off'} "
            f"on rank {getattr(getattr(self, 'mapping', None), 'tp_rank', None)}"
        )
        # TAILFOLD: each item serves only where its parts are set up (the fork's stream exists from _tf_prebuild).
        if self._tg is not None and not (s10c_e and self._tf_aux is not None):
            self._tg = None  # (its hooks then pass every call to the stock ops)
        self._tf86_on = (
            _tfold.MEMCPY86K and bool(s10c_c) and _tfold.drafter_owns_noise_hook(draft_model)
        )
        self._tf4b_on = _tfold.MEMCPY4B and self._tf_aux is not None
        for line in _tfold.tailfold_setup_line(
            self, self._tg is not None and self._tg.broken is None, self._tf86_on, self._tf4b_on
        ):
            logger.info(line)

    def _prebuild_dhead4(self, lm_head: nn.Module) -> bool:
        """Set up DHEAD_MX4 on the S4 state (every rank or none); False, with one line naming why, when it is off.
        Off by its switch, or without a head (a shell worker; the target builds its head on every rank): no
        collective (the environment and the model are the same on every rank). On: every rank builds the copy (or
        names why not) and the ranks agree before any of them uses it. Raises (on every rank: the environment and the
        shapes are the same) when TRTLLM_M31_DHEAD_MX4_TACTIC is set to a malformed tactic or to one the GEMM cannot
        run at the served shape; raises on every rank, after a first agreement round, when a set-up failure left the
        CUDA context faulted on any rank (the agreement is a host-side MPI allreduce, usable after a CUDA fault)."""
        from . import dspark_draft_head as _dhead4

        mapping = getattr(self, "mapping", None)
        rank = getattr(mapping, "tp_rank", None)
        if not DHEAD_MX4 or lm_head is None:
            if lm_head is None and DHEAD_MX4:
                why = "no head: a shell worker"
            elif m31_disabled("DHEAD_MX4"):
                why = "TRTLLM_M31_DISABLE_DHEAD_MX4 set"
            else:
                why = "default off: opt in with TRTLLM_M31_ENABLE_DHEAD_MX4=1"
            logger.info(
                f"M3.1 DSpark DHEAD_MX4: MXFP8 draft head off on rank {rank} ({why}); BF16 draft head"
            )
            return False
        state = getattr(self, "_s4", None)
        k = int(getattr(getattr(self, "spec_config", None), "max_draft_len", 0) or 0)
        rows = [b * k for b in _dhead4.BATCHES]
        head4 = None
        # An explicitly set tactic fails loudly: a malformed value raises here, an unrunnable one below (a decline
        # would serve the BF16 head under a DHEAD label).
        tactic, source = _dhead4.pinned_tactic()
        why = _dhead4.unfit(lm_head, getattr(state, "shard", None), k, getattr(state, "rows", None))
        if why is None and source != "default":
            bad = _dhead4.tactic_invalid(tactic, state.shard, rows, int(lm_head.weight.shape[1]))
            if bad is not None:
                raise ValueError(f"{_dhead4.TACTIC_ENV}={_dhead4.tactic_name(tactic)}: {bad}")
        fault = None
        if why is None:
            try:
                head4 = _dhead4.DraftHeadMx4(lm_head, rows, state.rows, rank=rank, tactic=tactic)
            except Exception as e:  # noqa: BLE001 - any set-up failure (a compile, the self-check, memory) declines
                # on every rank through the agreement below instead of leaving the ranks on different heads; one
                # that left the CUDA context faulted is raised on every rank (the next CUDA call would fail anyway)
                try:
                    _dhead4.raise_if_cuda_fault(e)
                except RuntimeError as f:
                    fault = f
                why = f"set-up failed: {type(e).__name__}: {e}"
        # First agreement: a faulted context on any rank fails the boot on every rank now, not as a hang of the healthy
        # ranks in the next collective.
        if not self._a1_all_ok(fault is None):
            if fault is not None:
                raise fault
            raise RuntimeError(
                f"DHEAD_MX4 set-up left the CUDA context faulted on another rank (rank {rank} is healthy)"
            )
        # Second agreement: the copy is used on every rank or on none.
        if not self._a1_all_ok(why is None):
            logger.info(
                f"M3.1 DSpark DHEAD_MX4: MXFP8 draft head off on rank {rank} ({why or 'off on another rank'}); "
                "BF16 draft head"
            )
            return False
        state.head4 = head4
        logger.info(
            f"M3.1 DSpark DHEAD_MX4: MXFP8 draft head set up on rank {rank}/{getattr(mapping, 'tp_size', None)} "
            f"(shard {head4.n} rows, MXFP8 copy {head4.np} rows, K {head4.k}, rows {head4.rows}; GEMM at logical N "
            f"{head4.n} into the S4 buffer, row stride {head4.row_stride}, -inf tail kept; tactic {head4.tactic} = "
            f"{_dhead4.tactic_name(head4.tactic)} ({source}); PDL {'on' if head4.pdl else 'off'}; quantize "
            f"{head4.quantizer}; copy {head4.bytes / 1e6:.1f} MB; self-check {head4.check}; set-up "
            f"{head4.setup_s:.2f} s)"
        )
        return True

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
        # TAIL_FORK: the drafter's input work forks here, after the head GEMM.
        self._tf_launch()
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
        """The next step's input tokens in one launch (N3 A2), else the in-tree index + concat; on a decode-only step
        whose forward tail stores the step, with that tail's store + staging in the same launch (TAILGLUE_B)."""
        batch_idx = batch_indices_cuda[:batch_size]
        fold = self._nnt_fold_target
        if (
            TAILGLUE_B
            and fold is not None
            and self._nnt_fold_step
            and next_draft_tokens.dim() == 2
            and next_draft_tokens.shape[0] == batch_size
            and accepted_tokens.dim() == 2
            and accepted_tokens.shape[0] == batch_size
        ):
            out = fold.fold_next_new_tokens(
                accepted_tokens, num_accepted_tokens, batch_idx, next_draft_tokens
            )
            if out is not None:
                if not self._nnt_fold_logged:
                    self._nnt_fold_logged = True
                    logger.info(
                        f"[M3 dispatch] TAILGLUE_B taken: next_new_tokens + forward-tail fold in one launch "
                        f"({batch_size} requests) on rank {getattr(getattr(self, 'mapping', None), 'tp_rank', None)}"
                    )
                return out
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
            return _strict_accept.spec_next_new_tokens(
                accepted_tokens, num_accepted_tokens, batch_idx, next_draft_tokens
            )
        return super()._prepare_next_new_tokens(
            accepted_tokens, next_draft_tokens, batch_indices_cuda, batch_size, num_accepted_tokens
        )

    # TAILBK (RuntimeHooks.spec_tail_bookkeep, armed by modeling.py's TAILBK; spec_metadata.tail_bookkeep): R1 and Z1,
    # this worker's part of the sampler tail bookkeeping (B1 / S1 / A1 are the shared branches in interface.py). The
    # sm_107 TP8 target's FOLD_F3 (R1's window written by the REJ_MC rejection launch) needs REJ_MC, which this target
    # does not install: R1 here always computes the window with its own add.
    _tailbk_win = None  # R1: (spec_metadata, offsets window) the accept call left for this forward's draft block

    # R1's window is reset at both ends of every step by the one _forward_impl override above (with TAIL_FORK's join):
    # a second override here would replace that one and leave a forked step's aux stream unjoined.

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
        (iii) every consumer is out-of-place (the non-fused probs: the softmax / top-k mask ops allocate their outputs,
            top-p reads probs; this target installs no softmax override), so returning ``logits`` itself instead of
            that copy exposes no write to the caller's verify logits. The fused modes are not audited and keep the
            copy.
        (iv) the consumers see the layout the copy would have: the skip needs contiguous logits at a 256-byte aligned
            address (the copy is a fresh contiguous allocation, which the caching allocator aligns at least that much;
            the verify logits here are the A1 all-gather's or the NCCL path's fresh output), so a consumer that
            dispatches on layout takes the same path.
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
    wrapped = install_draft_oneshot_allreduce(draft, DRAFT_AR_ONESHOT_MAX_ROWS)
    logger.info(
        f"[M3 prebuild] S9b-VR-a: {wrapped} drafter row-parallel all-reduce(s) take ONESHOT up to "
        f"{DRAFT_AR_ONESHOT_MAX_ROWS} rows"
        + ("" if DRAFT_AR_ONESHOT_MAX_ROWS else " (off: TRTLLM_M31_DISABLE_S9BVR)")
    )
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
