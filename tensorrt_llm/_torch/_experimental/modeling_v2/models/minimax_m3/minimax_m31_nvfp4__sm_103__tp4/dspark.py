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

import os
import time
from typing import Tuple

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
    allgather_cast as _allgather_cast,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode import (
    strict_accept as _strict_accept,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode.dspark_block_tables import (
    enable_parallel_refresh,
    refresh_block_tables,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode.dspark_ctx_advance import (
    advance_ctx_and_read_view,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode.dspark_ctx_store import (
    fused_ctx_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode.dspark_drafter_inputs import (
    drafter_inputs as _drafter_inputs,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode.dspark_forward_indices import (
    forward_indices,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode.dspark_qk_norm_rope import (
    qk_norm_rope,
)
from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_speculative import (
    external_drafter_config_kwargs,
    get_draft_model,
)
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm.logger import logger

from .dspark_s4 import MAX_FAST_BATCH, S4WorkerMixin
from .m31_switches import m31_disabled
from .mxdraft import route_mx_linears
from .ops.embedding.m3_embedding import m3_embedding

# MXDRAFT (MXFP8 drafter GEMV): ON by default. With the K3 bundle off (the ship config), its powered K7 gate c
# (n 1,776) passes: mean d +0.009 AL, CI [-0.019, +0.037]. Kill switch TRTLLM_M31_DISABLE_MXDRAFT=1, read at import.
MXDRAFT = not m31_disabled("MXDRAFT")


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
    """S10c-C: the drafter's step inputs in one launch (catalog/spec_decode/dspark_drafter_inputs), for this target's
    embedding only: ``M3EmbedTokens`` is an exact gather of the replicated table, so the noise block is table rows (the
    trained mask row if the drafter keeps one). None for any other embedding: the stock path runs."""
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


# N3 A2 (switch TRTLLM_M31_DISABLE_N3A): the target's strict acceptance and next-step input tokens in one launch each
# (catalog/spec_decode/strict_accept), bitwise to the in-tree torch path. Read once, at import.
N3A_DISABLED = m31_disabled("N3A")
# N3A-KV (switch TRTLLM_M31_DISABLE_N3A_KV, and off with N3A): DFlash's draft-forward KV bookkeeping (the rewind amount
# 1 - num_accepted and kv_lens += 1) in one launch where the in-tree lines run, after the forced-acceptance override,
# bitwise (integer arithmetic). Read once, at import.
N3A_KV_DISABLED = N3A_DISABLED or m31_disabled("N3A_KV")
# Gate-only cross-check (TRTLLM_M31_CHECK_N3A_KV=1; never in a perf arm): every N3A-KV call is compared with the stock
# lines on copies (graph-safe device counters; reported on eager steps).
N3A_KV_CHECK = os.environ.get("TRTLLM_M31_CHECK_N3A_KV", "").strip() not in ("", "0")
# N3 P-gid + P-cat (switch TRTLLM_M31_DISABLE_N3P): the verify-row target logits without the identity gather_ids copy
# and with the vocab concat and the fp32 cast in one strided copy, bitwise to the logits processor. Read at import.
N3P_DISABLED = m31_disabled("N3P")
# N3 A1 (switch TRTLLM_M31_DISABLE_A1): the verify logits' vocab all-gather, concat and fp32 cast in one launch over a
# multicast NVLink workspace (catalog/comm/allgather_cast), bitwise to the NCCL all-gather + strided copy below. K3
# verify rows (B x (K + 1), up to 64 rows: K3 batches 1..16, K7 batches 1..8). Read once, at import.
A1_DISABLED = m31_disabled("A1")
A1_MAX_ROWS = 64  # capacity: fixed: allgather_cast.MAX_ROWS (K3 at batch 16, K7 at batch 8)
# A1 at draft lengths other than 3 (switch TRTLLM_M31_DISABLE_A1_K7, K3 keeps A1).
A1_K7_DISABLED = m31_disabled("A1_K7")
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


class M3DSparkWorker(S4WorkerMixin, DSparkWorker):
    """The in-tree ``DSparkWorker`` with the calls this target's shell makes on its worker. W1: the one-launch Markov
    chain (``S4WorkerMixin``, switch ``TRTLLM_M31_DRAFT_DISABLE_S4``). N3 A2: the strict acceptance and the next-step
    input tokens in one launch each (switch ``TRTLLM_M31_DISABLE_N3A``). Everything else is the in-tree path."""

    def capture_hidden_states(
        self, spec_metadata, layer_id: int, hidden_states, residual=None
    ) -> None:
        """The in-tree DFlash capture: layer output (hidden + residual) into the static capture buffer."""
        spec_metadata.maybe_capture_hidden_states(layer_id, hidden_states, residual)

    def arm_capture_norm(
        self, attn_metadata, spec_metadata, lm_head: nn.Module, draft_model
    ) -> None:
        """No capture norm: DSpark's fc reads the raw captured layers."""
        return None

    def tail_eligible(self, attn_metadata, spec_metadata, lm_head: nn.Module) -> bool:
        """Never: the target logits always go through the logits processor and the in-tree strict accept."""
        return False

    _a1 = None  # the N3 A1 workspace (allgather_cast's arguments) once set up

    def prebuild(self, lm_head: nn.Module, draft_model) -> None:
        """The one-launch Markov chain's shards, buffers, exchange workspace and compiles, then the N3 A2 compiles
        (rank-local) and the N3 A1 workspace and compiles (collective), outside capture, every rank, in this order;
        no fused verify, no coded target head, no fused draft loop."""
        rank = getattr(getattr(self, "mapping", None), "tp_rank", None)
        # S10: the drafter's strided-view glue (bitwise; kill switch TRTLLM_M31_DISABLE_S10, read at import).
        lean = not m31_disabled("S10")
        self.enable_lean_glue(draft_model, lean)
        k = getattr(getattr(self, "spec_config", None), "max_draft_len", 0) or 0
        logger.info(f"M3.1 DSpark S10: lean drafter glue {'on' if lean else 'off'} on rank {rank}")
        # S10c-A: the drafter's context K / V epilogue + accept mask + paged store in one launch after the K / V GEMM
        # (bitwise). Kill switches TRTLLM_M31_DISABLE_S10C (all of S10c) / TRTLLM_M31_DISABLE_S10C_A, read at import.
        s10c_a = self.enable_fused_ctx_store(
            None if m31_disabled("S10C") or m31_disabled("S10C_A") else fused_ctx_store
        )
        logger.info(
            f"M3.1 DSpark S10c-A: fused drafter ctx store {'on' if s10c_a else 'off'} on rank {rank}"
        )
        # S10c-C: the drafter's step inputs (bonus, positions, noise block) in one launch (bitwise: integers and row
        # copies). Kill switches TRTLLM_M31_DISABLE_S10C (all of S10c) / TRTLLM_M31_DISABLE_S10C_C, read at import.
        s10c_c = self.enable_fused_drafter_inputs(
            None if m31_disabled("S10C") or m31_disabled("S10C_C") else s10c_c_drafter_inputs
        )
        logger.info(
            f"M3.1 DSpark S10c-C: fused drafter step inputs {'on' if s10c_c else 'off'} on rank {rank}"
        )
        # S10c-D: the drafter's context advance + read view in one launch (bitwise, integer). Kill switches
        # TRTLLM_M31_DISABLE_S10C (all of S10c) / TRTLLM_M31_DISABLE_S10C_D, read at import.
        s10c_d = self.enable_fused_ctx_advance(
            None if m31_disabled("S10C") or m31_disabled("S10C_D") else advance_ctx_and_read_view
        )
        logger.info(
            f"M3.1 DSpark S10c-D: fused drafter context advance {'on' if s10c_d else 'off'} on rank {rank}"
        )
        # S10c-B: the drafter's block-table refresh (decode + placeholder count) in one launch (bitwise, integer).
        # Kill switches TRTLLM_M31_DISABLE_S10C (all of S10c) / TRTLLM_M31_DISABLE_S10C_B, read at import.
        s10c_b = self.enable_fused_block_refresh(
            None if m31_disabled("S10C") or m31_disabled("S10C_B") else refresh_block_tables
        )
        # S10c-B2: that refresh on a (row, column-chunk) grid (bitwise; flat at short tables, removes the serial
        # kernel's growth past ~11K blocks per sequence). Kill switch TRTLLM_M31_DISABLE_S10C_B2, read at import.
        s10c_b2 = enable_parallel_refresh(s10c_b and not m31_disabled("S10C_B2"))
        logger.info(
            f"M3.1 DSpark S10c-B: fused drafter block-table refresh {'on' if s10c_b else 'off'} on rank {rank}"
            f" (S10c-B2 2-D grid {'on' if s10c_b2 else 'off'})"
        )
        # S10c-E: the draft model's paged block-decode index prelude in one launch (bitwise, integer). Kill switches
        # TRTLLM_M31_DISABLE_S10C (all of S10c) / TRTLLM_M31_DISABLE_S10C_E, read at import.
        s10c_e = self.enable_fused_forward_indices(
            draft_model, None if m31_disabled("S10C") or m31_disabled("S10C_E") else forward_indices
        )
        logger.info(
            f"M3.1 DSpark S10c-E: fused drafter forward indices {'on' if s10c_e else 'off'} on rank {rank}"
        )
        # EX-11: the drafter block's q / k RMSNorm + partial RoPE in one launch (bitwise to flashinfer's three). Kill
        # switch TRTLLM_M31_DISABLE_EX11, read at import.
        ex11 = self.enable_fused_qk_norm_rope(
            draft_model, None if m31_disabled("EX11") else qk_norm_rope
        )
        logger.info(
            f"M3.1 DSpark EX-11: fused drafter q/k norm + RoPE {'on' if ex11 else 'off'} on rank {rank}"
        )
        # MXDRAFT: the drafter's qkv / o / gate_up / down GEMMs on this target's MXFP8 GEMV for the decode passes'
        # rows (B x K, B <= MAX_FAST_BATCH), from the exact MXFP8 form of the shipped weights (drafter-only precision).
        # ON by default (see MXDRAFT); kill switch TRTLLM_M31_DISABLE_MXDRAFT.
        mx = (
            0
            if not MXDRAFT or k <= 0
            else route_mx_linears(draft_model, [b * k for b in range(1, MAX_FAST_BATCH + 1)])
        )
        logger.info(f"M3.1 DSpark MXDRAFT: MXFP8 drafter GEMV on {mx} projections on rank {rank}")
        self.prebuild_s4(lm_head, draft_model)
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
            f"M3.1 N3A-KV: one-launch draft-forward KV bookkeeping {'on' if n3a_kv else 'off'} on rank "
            f"{getattr(getattr(self, 'mapping', None), 'tp_rank', None)}"
        )

    _n3a_kv_buf = None  # the persistent rewind amounts of N3A-KV (int32, MAX_REQUESTS)
    _n3a_kv_counts = None  # gate-only cross-check counters (calls, mismatches)

    def _n3a_kv_rewind(self, num_accepted, kv_lens, nc: int, bs: int):
        """N3A-KV for the DFlash hook: ``1 - num_accepted[nc:bs]`` (a view of the persistent buffer) with
        ``kv_lens[nc:bs] += 1`` in one launch, or None (the stock lines) for operands or a variant it does not take."""
        n = bs - nc
        buf = self._n3a_kv_buf
        if buf is None or not _strict_accept.supports_kv_rewind(num_accepted, kv_lens, buf, nc, n):
            return None
        if not _strict_accept.is_compiled_kv_rewind(num_accepted, kv_lens):
            return None
        _n3a_kv_log_once()
        if not N3A_KV_CHECK:
            return _strict_accept.spec_kv_rewind(num_accepted, kv_lens, buf, nc, n)
        want_rewind = 1 - num_accepted[nc:bs]
        want_kv = kv_lens[nc:bs] + 1
        out = _strict_accept.spec_kv_rewind(num_accepted, kv_lens, buf, nc, n)
        if self._n3a_kv_counts is None:
            self._n3a_kv_counts = torch.zeros(2, dtype=torch.int64, device=kv_lens.device)
        bad = (out != want_rewind).any() | (kv_lens[nc:bs] != want_kv).any()
        self._n3a_kv_counts[0] += 1
        self._n3a_kv_counts[1] += bad.to(torch.int64)
        if not torch.cuda.is_current_stream_capturing():
            calls, mism = self._n3a_kv_counts.tolist()
            logger.info(f"M3.1 N3A-KV check: {calls} calls, {mism} mismatches")
        return out

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
                ("TRTLLM_M31_DISABLE_A1 set", A1_DISABLED),
                (
                    f"draft length {int(self.max_draft_len)} with TRTLLM_M31_DISABLE_A1_K7 set",
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
        """The next step's input tokens in one launch (N3 A2), else the in-tree index + concat."""
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
            return _strict_accept.spec_next_new_tokens(
                accepted_tokens, num_accepted_tokens, batch_idx, next_draft_tokens
            )
        return super()._prepare_next_new_tokens(
            accepted_tokens, next_draft_tokens, batch_indices_cuda, batch_size, num_accepted_tokens
        )


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
