# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet speculative-verify recurrence under the engine's cached-replay
state update: a checkpoint state per slot plus a compact history of the tokens
accepted since it, instead of a per-step state buffer.

Wraps the in-tree module callable
``tensorrt_llm._torch.modules.fla.cached_replay.fused_recurrent_gated_delta_rule_cached_replay_update``
(tier-2 sourcing: the recurrence has no op boundary). One launch of the vendored
Triton kernel ``_cached_replay_kernel`` -- plain ``triton.jit``, no autotuner;
the Python body picks the launch configuration from the shapes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.fla.cached_replay import (
    fused_recurrent_gated_delta_rule_cached_replay_update as _cached_replay_update,
)

#: This checkpoint's GDN geometry at tp1.
NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM = 16, 48, 128, 128
#: `[q | k | v]` row width of the post-conv activations a verify step packs.
PACKED_QKV_WIDTH = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM  # 10240
#: The cache manager's `replay_history_size`: max(MIN_REPLAY_HISTORY_SIZE=16, T).
HISTORY_SIZE = 16
#: From this batch size on the engine runs the partitioned replay (work items, no
#: in-kernel commit, one all-layer commit after every layer has run):
#: `CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE` in `modules/fla/cached_replay.py`.
ALL_LAYER_COMMIT_MIN_BATCH = 16
#: The recurrent pool's slot count of a full-size engine run.
ENGINE_SLOTS = 461
#: Layers per slot at which the last of `ENGINE_SLOTS` slots starts past 2^31 elements of
#: the pool (`460 * L * 48 * 128 * 128 > 2**31` from L = 6).
PAST_INT32_LAYERS = 8


def _cell_spec(path, n, t, **spec) -> dict:
    return dict(path=path, n=n, t=t, **spec)


def _bf16_cell(why, path, n, t, **spec) -> Cell:
    return Cell(
        why=f"bf16 state {why}", spec=_cell_spec(path, n, t, state_dtype=torch.bfloat16, **spec)
    )


class FusedRecurrentGatedDeltaRuleCachedReplayUpdate(OpWrapper):
    """One verify step of `N` sequences x `T` tokens through one GDN layer, for
    the cached-replay state layout. Per live row (`slot = state_indices[n] >= 0`),
    with `p = prev_num_accepted_tokens[slot]`, `b = cache_buf_idx[slot]`:

    1. the state the step starts from is rebuilt from the checkpoint
       `h0 = ssm_states[slot]` (`[HV, V, K]`, K innermost) and the `p` pending
       history entries in half `b` -- update vectors `u_j = old_u[slot, b, j]`,
       normalized keys `k_j = old_k[slot, b, j]`, cumulative log-decays
       `G_j = old_G[slot, b, :, j]` measured from the checkpoint:
       `h = exp(G_{p-1}) h0 + sum_j exp(G_{p-1} - G_j) u_j k_j^T`;
    2. the `T` new tokens run the gated delta rule in order (q/k head
       `hv // (HV // H)`, so a v-head group shares one q/k head):
       `h *= exp(g_t); u_t = beta_t (v_t - h k_t); h += u_t k_t^T; o_t = h (scale q_t)`,
       q and k L2-normalized first (`x / sqrt(sum(x^2) + 1e-6)`) when
       `use_qk_l2norm_in_kernel`, `scale=None` meaning `K ** -0.5`; with
       `A_log` given, `g = -exp(A_log) * softplus(g + dt_bias)` and
       `beta = sigmoid(beta)` are applied to the raw projections inside;
    3. the new tokens' `(u_t, k_t, G_t)` are appended to the history. When
       `p + T <= history_size` they go to half `b` at `p .. p+T-1` with `G`
       still cumulative from the checkpoint. Otherwise (a *wrap*) they go to
       half `1 - b` at `0 .. T-1` with `G` restarted from zero, and the
       checkpoint must advance to the state after the `p` pending tokens:
       without work items the kernel writes that state into `ssm_states[slot]`
       itself; with `replay_work_items` (`use_all_layer_commit=True`) it does
       not, and the caller owes one `commit_gdn_cached_replay_history_layers`
       over every layer before the next step.

    Nothing about acceptance happens here: the caller advances `p`/`b` after
    sampling (`p <- accepted` and `b <- 1 - b` on a wrap, else `p += accepted`,
    `accepted` counting the bonus token), so rejected tokens' history entries
    are simply never read.

    Fusion boundary: L2 norms, scale, the gating of the raw `a`/`b`
    projections, the history rebuild, the recurrence, the history append and
    (small batches) the checkpoint commit are inside. The conv, the projections,
    the all-layer commit and the `p`/`b` bookkeeping are not.

    Returns `[N, T, HV, V]` in `output`'s dtype, else `q`'s; it *is* `output`
    when one is given. `ssm_states`, `old_u`, `old_k`, `old_G` are written in
    place and not returned. Rows with a negative slot are skipped entirely:
    nothing is read or written for them, their output rows included.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: path ("small": no work items, the kernel commits a wrap itself |
    # "all_layer": work items + use_all_layer_commit, the test then runs the
    # manager's all-layer commit), n sequences x t tokens per verify step,
    # steps (consecutive verify steps, each followed by an acceptance per row),
    # accept ("random": 1..t per row, the default | "full": t every step, so from
    # a fresh slot at t=8 the second step appends at `pnat + t == 16` without
    # wrapping and every second step after wraps),
    # state_dtype (the checkpoint pool's dtype: fp32 | bf16, as
    # `mamba_ssm_cache_dtype` allocates it; fp32 by default),
    # layers (each with its own activations, gates
    # and history; one V2-style pool interleaving the layers per slot), slots
    # (the pool's slot count), start ("warm": random pending histories in both
    # halves, a wrap and an append forced at step 0 | "fresh": pnat 0 and
    # zeroed history, as the manager leaves a slot after prefill), edge_slots
    # (rows 0 and n-1 on the pool's last and first slot), dup (rows given the
    # SAME slot, as CUDA-graph padding repeats one dummy slot; only the other
    # rows and slots are held to the reference). Every cell passes `packed_qkv`,
    # separate raw `a`/`b` projections, no `output` and `launch_with_pdl=False`.
    #
    # Four cells, one over the three-cell budget: the forward reaches two commit
    # paths (in-kernel below 16 requests, all-layer from 16) at two checkpoint
    # dtypes, and two edges no main-shape cell can carry -- CUDA-graph padding
    # rows sharing one slot, and slot offsets past 2^31 elements at the engine's
    # pool.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="all_layer n=32 T=8 full accept: commit every second step",
            spec=_cell_spec("all_layer", 32, 8, start="fresh", accept="full", steps=8),
        ),
        _bf16_cell(
            "small n=4 T=8 full accept: pnat+T==16 appends, then wraps",
            "small",
            4,
            8,
            start="fresh",
            accept="full",
            steps=8,
        ),
        Cell(
            why="small n=8 T=3 dup: rows 5..7 share one padding slot",
            spec=_cell_spec("small", 8, 3, dup=(5, 6, 7), steps=6),
        ),
        Cell(
            why="all_layer n=16 T=3 L=8 S=461: slot 460's offset is past 2^31 elements",
            spec=_cell_spec(
                "all_layer",
                16,
                3,
                layers=PAST_INT32_LAYERS,
                slots=ENGINE_SLOTS,
                edge_slots=True,
                steps=3,
            ),
        ),
    )

    note = """
    Call sites. The engine's GDN mixer (`modules/mamba/gdn_mixer.py`) is the
    reference caller: the target-verify step of a generation-only batch, and the
    decode rows of a mixed context + verify batch -- `N = num_decodes`, `T =
    runtime_draft_len + 1` (2..8 here: max_draft_len 1..7). A plain decode step never reaches it. q/k/v
    are the `[N, T, 16|48, 128]` views of the post-conv `[N*T, 10240]` rows, plus
    `packed_qkv` = those rows for a generation-only batch (a mixed batch passes the
    views alone and the op copies them contiguous); `g=a`, `beta=b` are the raw
    bf16 `[N, T, 48]` projections, with the layer's fp32 `[48]` `A_log` and
    `dt_bias`; `ssm_states` is the layer's pool view `[S, 48, 128, 128]`, fp32 or bf16 as
    `mamba_ssm_cache_dtype` allocated it (V2 interleaves the layers inside each slot, so
    the slot stride is a multiple of one state); `state_indices` the decode rows' int32 slots; `old_u, old_k, old_G,
    old_beta` the layer's `old_x, old_B, old_dt, old_dA_cumsum` from
    `mamba_layer_cache(layer)`; `cache_buf_idx`, `prev_num_accepted_tokens`,
    `history_size` from `get_replay_state_update_metadata()` (16 here);
    `use_qk_l2norm_in_kernel=True`; and exactly when `N >= 16` and the manager's
    `use_gdn_cached_replay_all_layer_commit` holds, `replay_work_items =
    mamba_metadata.replay_work_items[:N]`, `n_writes = mamba_metadata.replay_n_writes`,
    `use_all_layer_commit=True`. That condition has to be the manager's own:
    `update_mamba_states` runs the all-layer commit iff it holds, so work items
    passed when it does not leave every wrap uncommitted, and work items withheld
    when it does commit twice -- both silent at the op and far off the exact
    recurrence one step later.

    Metadata consumed: `state_indices` (one slot per row), and per slot
    `prev_num_accepted_tokens` and `cache_buf_idx`. `replay_work_items` /
    `n_writes` are only checked (shape, dtype); the kernel never reads them, their
    presence switches the in-kernel commit off. `old_beta` is never read or written.

    Preconditions not checkable from metadata:
    * rows on distinct slots, except padding: two rows on one slot both append to the
      same history positions and the appended u is a per-element mixture of the two
      rows, so that slot's state and those rows' outputs are unspecified. The rows of
      CUDA-graph padding repeat one shared dummy slot (the manager's `_padding_slot`);
      the rows on unique slots are exact whatever the padding does, and the dummy slot
      is nobody's state;
    * `prev_num_accepted_tokens` in `[0, history_size]`, `cache_buf_idx` in `{0, 1}`,
      and the pending entries written by earlier calls of this op. A slot fresh from
      prefill starts at 0 pending (the manager zeroes the count, the half and the
      history);
    * after every step, before the next: with work items,
      `commit_gdn_cached_replay_history_layers` over every local GDN layer; then the
      advance (`_advance_replay_state` in `modules/fla/cache_manager.py`)
      with the accepted counts including the bonus token. Advancing as if every
      draft were accepted is silent at the op and wrong from the next step on;
    * the conv state is not this op's: under replay the conv still writes its
      per-step window and the manager promotes it.

    The launch is one V tile, `num_warps=4`, `num_stages=1`, grid `(1, N * 48)`: the
    tuned mappings need HV = 2H or 4H and never engage at HV/H = 3. The checkpoint's dtype
    is the kernel's `STATE_FP32` specialization and nothing else below N = 128: an fp32
    checkpoint enters the history rebuild's tensor-core dots as a bf16 hi/lo pair, a bf16
    one as stored, and every commit is computed in fp32 and rounded once to the pool's
    dtype. From N = 128 a bf16 state alone also turns on the L2-streaming mapping
    (`main_num_stages=3`, a 5-stage commit), which no cell drives. `FLA_USE_FAST_OPS=1` at
    import switches the kernel's `exp` to the approximate one; the cells run with it unset.

    CUDA-graph replay on new data is bit for bit the eager call (the test captures
    N = 32 at T = 8 over an fp32 checkpoint, with the all-layer commit). The two commit
    paths, `packed_qkv` against separate q/k/v, and `launch_with_pdl` on and off are not
    certified against each other; each form the target calls is certified against the
    reference by the cells.

    Certified: the all-layer commit over an fp32 checkpoint at N = 32, T = 8 and at
    N = 16, T = 3 with 8 interleaved layers over the engine's 461-slot pool (the last
    slot's offset past 2^31 elements); the in-kernel commit over a bf16 checkpoint at
    N = 4, T = 8 and over an fp32 one at N = 8, T = 3 with three CUDA-graph padding rows
    on one shared slot. T = 5..8 share one compiled token tile (`BT = next_pow2(T) = 8`);
    the cells drive the BT = 4 and BT = 8 tiles, not BT = 2 (T = 2). At T = 8 and history
    16 a slot whose every draft is accepted appends rows 8..15 at `pnat + T == 16` without
    wrapping and wraps on the next step, so it commits every second step; the
    `accept="full"` cells drive that boundary (in-kernel and all-layer commit).

    Loud refusals: work items without `use_all_layer_commit` (ValueError) and the
    reverse (ValueError); one of `replay_work_items` / `n_writes` alone
    (ValueError); a strided `output`; a strided head axis of `g`; `history_size <
    T` (a bare assert); and `history_size <= 8`, which does not compile -- the
    history tile is `next_pow2(history_size)` wide and feeds a `tl.dot` whose
    reduction must be 16, so only 9..16 work (the engine's is 16).

    `is_valid` guards what is accepted and silently wrong: `state_indices` shorter
    than N (the trailing rows read past it and append to a slot nobody named);
    history buffers whose shape disagrees with `history_size` or the head geometry
    (the kernel addresses them by `history_size`, so 16-deep buffers called at 12
    read other slots' entries); history buffers, `prev_num_accepted_tokens` or
    `cache_buf_idx` covering fewer slots than `ssm_states` (a slot of the pool past
    their end is read and written out of bounds); a strided `prev_num_accepted_tokens`
    or `cache_buf_idx` (read at element `slot`). A strided `state_indices` is
    harmless: the entry's input guard makes it contiguous.

    Not certified: pre-gated `g`/`beta` (`A_log=None`); `use_qk_l2norm_in_kernel=
    False`; an `ssm_states` other than fp32 or bf16, or a bf16 one at N >= 128; HV/H other
    than 3, K or V other than 128; T = 2, or T outside 2..8 (T = 9 raises: the kernel's
    bare `assert T <= 8`), at either `ssm_states` dtype; `history_size` other than 16; `old_u` /
    `old_k` other than bf16, `old_G` other than fp32; a `scale`; any launch-knob
    override (`block_v`, `num_warps`, `use_l2_*`, `main_num_stages`); an output
    dtype other than bf16; int64 `state_indices`; N above 32.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_states: torch.Tensor,
        state_indices: torch.Tensor,
        old_u: torch.Tensor,
        old_k: torch.Tensor,
        old_G: torch.Tensor,
        old_beta: torch.Tensor,
        cache_buf_idx: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        history_size: int,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        launch_with_pdl: bool = False,
        replay_work_items: Optional[torch.Tensor] = None,
        n_writes: Optional[torch.Tensor] = None,
        block_v: Optional[int] = None,
        num_warps: Optional[int] = None,
        use_l2_state_cache: Optional[bool] = None,
        use_l2_streaming_inputs: Optional[bool] = None,
        use_l2_shared_inputs: Optional[bool] = None,
        main_num_stages: Optional[int] = None,
        packed_qkv: Optional[torch.Tensor] = None,
        use_all_layer_commit: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _cached_replay_update(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            state_indices=state_indices,
            old_u=old_u,
            old_k=old_k,
            old_G=old_G,
            old_beta=old_beta,
            cache_buf_idx=cache_buf_idx,
            prev_num_accepted_tokens=prev_num_accepted_tokens,
            history_size=history_size,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            launch_with_pdl=launch_with_pdl,
            replay_work_items=replay_work_items,
            n_writes=n_writes,
            block_v=block_v,
            num_warps=num_warps,
            use_l2_state_cache=use_l2_state_cache,
            use_l2_streaming_inputs=use_l2_streaming_inputs,
            use_l2_shared_inputs=use_l2_shared_inputs,
            main_num_stages=main_num_stages,
            packed_qkv=packed_qkv,
            use_all_layer_commit=use_all_layer_commit,
            output=output,
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_states: torch.Tensor,
        state_indices: torch.Tensor,
        old_u: torch.Tensor,
        old_k: torch.Tensor,
        old_G: torch.Tensor,
        old_beta: torch.Tensor,
        cache_buf_idx: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        history_size: int,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        launch_with_pdl: bool = False,
        replay_work_items: Optional[torch.Tensor] = None,
        n_writes: Optional[torch.Tensor] = None,
        block_v: Optional[int] = None,
        num_warps: Optional[int] = None,
        use_l2_state_cache: Optional[bool] = None,
        use_l2_streaming_inputs: Optional[bool] = None,
        use_l2_shared_inputs: Optional[bool] = None,
        main_num_stages: Optional[int] = None,
        packed_qkv: Optional[torch.Tensor] = None,
        use_all_layer_commit: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(out, ssm_states, old_u, old_k, old_G)` as the call leaves them;
        mutates nothing. The state is rebuilt from the checkpoint and history
        in fp32 and the new tokens are stepped one at a time in fp32 -- no
        matmul, so no TF32 -- and every store is rounded once to its buffer's
        dtype. The history is read at whatever dtype and depth its buffers
        have, so the same function driven on fp32 history deeper than any run
        ever fills is the exact token-by-token recurrence.

        Rows with a negative slot keep `output`'s prior content, or are NaN
        (unspecified) when no `output` is given. `old_beta`, `launch_with_pdl`,
        `n_writes` and the launch-tuning knobs are ignored; `replay_work_items`
        only decides whether a wrap commits the checkpoint here."""
        N, T, H, K = k.shape
        HV, V = v.shape[2], v.shape[3]
        rep = HV // H
        dev = v.device
        scale = K**-0.5 if scale is None else scale
        if packed_qkv is not None:
            rows = packed_qkv.view(N, T, -1)
            q = rows[..., : H * K].view(N, T, H, K)
            k = rows[..., H * K : 2 * H * K].view(N, T, H, K)
            v = rows[..., 2 * H * K :].view(N, T, HV, V)
        qf, kf, vf = q.float(), k.float(), v.float()
        if use_qk_l2norm_in_kernel:
            qf = qf / torch.sqrt(qf.square().sum(-1, keepdim=True) + 1e-6)
            kf = kf / torch.sqrt(kf.square().sum(-1, keepdim=True) + 1e-6)
        qf = (qf * scale).repeat_interleave(rep, dim=2)  # [N, T, HV, K]
        kv = kf.repeat_interleave(rep, dim=2)  # [N, T, HV, K]
        if A_log is not None:
            gf = -torch.exp(A_log.float()) * F.softplus(g.float() + dt_bias.float())
            bf = torch.sigmoid(beta.float())
        else:
            gf, bf = g.float(), beta.float()

        out_dtype = q.dtype if output is None else output.dtype
        out = (
            torch.full((N, T, HV, V), float("nan"), device=dev)
            if output is None
            else output.float().clone()
        )
        pool, u_buf, k_buf, G_buf = (x.clone() for x in (ssm_states, old_u, old_k, old_G))
        live = (state_indices[:N] >= 0).nonzero().flatten()
        if live.numel() == 0:
            return out.to(out_dtype), pool, u_buf, k_buf, G_buf
        slot = state_indices[:N][live].long()
        p = prev_num_accepted_tokens[slot].long()
        b = cache_buf_idx[slot].long()
        M, depth = live.numel(), old_u.shape[2]

        # 1. the pending state: checkpoint plus the p history entries of half b
        pending = torch.arange(depth, device=dev)[None, :] < p[:, None]  # [M, depth]
        Gh = torch.where(pending[:, None, :], old_G[slot, b].float(), 0.0)  # [M, HV, depth]
        Uh = torch.where(pending[:, :, None, None], old_u[slot, b].float(), 0.0)
        Kh = torch.where(pending[:, :, None, None], old_k[slot, b].float(), 0.0)
        Kh = Kh.repeat_interleave(rep, dim=2)  # [M, depth, HV, K]
        last = (p - 1).clamp(min=0)[:, None, None].expand(M, HV, 1)
        g_start = torch.where(p[:, None] > 0, Gh.gather(2, last).squeeze(2), 0.0)  # [M, HV]
        weight = torch.where(pending[:, None, :], torch.exp(g_start[:, :, None] - Gh), 0.0)
        h = torch.exp(g_start)[:, :, None, None] * ssm_states[slot].float()  # [M, HV, V, K]
        for j in range(depth):
            h = h + weight[:, :, j, None, None] * Uh[:, j, :, :, None] * Kh[:, j, :, None, :]
        checkpoint = h

        # 2. the new tokens, in order
        gl, bl = gf[live], bf[live]  # [M, T, HV]
        ql, kl, vl = qf[live], kv[live], vf[live]
        G_local = torch.zeros(M, HV, device=dev)
        u_new, G_new = [], []
        for t in range(T):
            h = h * torch.exp(gl[:, t])[:, :, None, None]
            u = bl[:, t, :, None] * (vl[:, t] - (h * kl[:, t, :, None, :]).sum(-1))  # [M, HV, V]
            h = h + u[:, :, :, None] * kl[:, t, :, None, :]
            out[live, t] = (h * ql[:, t, :, None, :]).sum(-1)
            G_local = G_local + gl[:, t]
            u_new.append(u)
            G_new.append(G_local.clone())

        # 3. the history append, and the checkpoint commit on a wrap
        wrap = p + T > history_size
        w_buf = torch.where(wrap, 1 - b, b)
        w_off = torch.where(wrap, 0, p)
        for t in range(T):
            pos = w_off + t
            u_buf[slot, w_buf, pos] = u_new[t].to(u_buf.dtype)
            k_buf[slot, w_buf, pos] = kf[live, t].to(k_buf.dtype)
            G_store = torch.where(wrap[:, None], G_new[t], g_start + G_new[t])
            G_buf[slot, w_buf, :, pos] = G_store.to(G_buf.dtype)
        if replay_work_items is None and bool(wrap.any()):
            pool[slot[wrap]] = checkpoint[wrap].to(pool.dtype)
        return out.to(out_dtype), pool, u_buf, k_buf, G_buf

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_states: torch.Tensor,
        state_indices: torch.Tensor,
        old_u: torch.Tensor,
        old_k: torch.Tensor,
        old_G: torch.Tensor,
        old_beta: torch.Tensor,
        cache_buf_idx: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        history_size: int,
        scale: Optional[float] = None,
        use_qk_l2norm_in_kernel: bool = False,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        launch_with_pdl: bool = False,
        replay_work_items: Optional[torch.Tensor] = None,
        n_writes: Optional[torch.Tensor] = None,
        block_v: Optional[int] = None,
        num_warps: Optional[int] = None,
        use_l2_state_cache: Optional[bool] = None,
        use_l2_streaming_inputs: Optional[bool] = None,
        use_l2_shared_inputs: Optional[bool] = None,
        main_num_stages: Optional[int] = None,
        packed_qkv: Optional[torch.Tensor] = None,
        use_all_layer_commit: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> None:
        N, _, H, K = k.shape
        HV, V = v.shape[2], v.shape[3]
        assert state_indices.dim() == 1 and state_indices.shape[0] >= N, (
            f"state_indices has shape {tuple(state_indices.shape)} for {N} sequences -- the "
            "kernel reads one slot per sequence with no bound, so a shorter tensor hands the "
            "trailing rows whatever follows it in memory, and they write that slot's history"
        )
        pool_slots, slots = ssm_states.shape[0], old_u.shape[0]
        assert slots >= pool_slots, (
            f"the history buffers have {slots} slots for a {pool_slots}-slot pool -- a slot "
            "past the history's end is read and written out of bounds"
        )
        for name, tensor in (
            ("prev_num_accepted_tokens", prev_num_accepted_tokens),
            ("cache_buf_idx", cache_buf_idx),
        ):
            assert tensor.dim() == 1 and tensor.is_contiguous() and tensor.shape[0] >= pool_slots, (
                f"{name} has shape {tuple(tensor.shape)} and stride {tensor.stride()}; the kernel "
                f"reads one element per slot of the {pool_slots}-slot pool, at the slot's index"
            )
        expected = {
            "old_u": (old_u, (slots, 2, history_size, HV, V)),
            "old_k": (old_k, (slots, 2, history_size, H, K)),
            "old_G": (old_G, (slots, 2, HV, history_size)),
        }
        for name, (tensor, shape) in expected.items():
            assert tuple(tensor.shape) == shape, (
                f"{name} has shape {tuple(tensor.shape)}; history_size={history_size} and this "
                f"geometry need {shape} -- the kernel addresses the history by history_size and "
                "the head geometry, never by the buffer's own shape"
            )


fused_recurrent_gated_delta_rule_cached_replay_update = (
    FusedRecurrentGatedDeltaRuleCachedReplayUpdate()
)
