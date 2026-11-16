# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The workspace the TP4 Lamport collective entries share.

`cute_dsl_allreduce`, `cute_dsl_allreduce_add_rmsnorm` and
`cute_dsl_allreduce_add_rmsnorm_quant_nvfp4` are one transport with three
epilogues; `cute_dsl_allgather_widen` (the lm_head's vocabulary gather) is a
fourth. They read and write the same symmetric-memory peer buffers and
rotate the same device slot counter, so every collective of a forward goes
through one `LamportWorkspace`, built once at load by `build_workspace`.

The kernel reads its slot from the counter in device memory and advances it
itself, which is what lets one CUDA graph capture every collective of a
forward and still rotate slots on replay. The peer buffers are three slots x
world x the token budget, so every row count up to the budget fits.

Not an entry: it holds what the entries are handed, and no entry owns it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cutlass
import torch

from tensorrt_llm.logger import logger

from ...kernels.comm import tp4_allgather_lamport as _ag
from ...kernels.comm import tp4_allreduce_fused_rms as _ar
from ...kernels.common.cuda_stream import raw_stream as _raw_stream

_PING_PONG = 3
_SIGNAL_STATE = 6
_BF16_VEC = 8
# In-band readiness: bitwise bf16 -0.0 marks a slot not yet arrived, as TRT-LLM's own
# Lamport backend does. Separate ready flags double the peer traffic.
_NEG_ZERO_BF16 = -32768
# PDL release points. The module default, `strict`, hangs here: it waits at entry for a
# producer release the flashinfer and torch kernels ahead of it never issue. `entry` on the
# fused ops gives the dependent kv_a projection the allreduce's NVLink wait to prefetch its
# weight slice into L2.
_PDL_POLICY = "output_safe"
_FUSED_PDL_POLICY = "entry"
_FUSED_QUANT_PDL_POLICY = "entry"
_FUSED_OPTIONS = dict(
    all_to_all_cluster_reduction=True,
    last_cta_signal_advance=True,
    elide_self_rank=True,
    rotate_peer_order=True,
    rotated_poll_order=True,
)
#: Above this many rows a workspace built `wide` (an engine drafting more than one token: MTP3 /
#: MTP5 decode steps of 128 / 192 rows) runs the add+RMSNorm+NVFP4 entry on
#: `fused_quant_twoshot`: the same kernel forming the sum two-shot. Ahead of an MoE layer every
#: rank arrives at once (attention is balanced) and NVLink ingress is the bound: the one-shot
#: takes in the three peers' whole partials, the two-shot a quarter of each (the slices it owns,
#: which it sums and sends back as the bf16 residual) plus three quarters of the reduced rows,
#: half the bytes for one more hop. Alone on 4x B300 (16 calls per graph, 60 alternating
#: blocks), us per call one-shot unicast -> two-shot: 65 rows 10.2 -> 8.5, 128 rows 13.9 ->
#: 10.9, 192 rows 17.7 -> 12.6, 256 rows 24.0 -> 18.3. The owner sums in rank order in fp32 and
#: rounds once, as the one-shot: the outputs are bit for bit the one-shot kernel's. At 1..64 rows
#: (MTP1, the 32-row draft steps) the one-shot multicast build stays.
QUANT_TWOSHOT_ABOVE = 64
#: Above this many rows the add+RMSNorm entry runs two-shot: one-shot has every rank pull
#: the whole message from every peer. The two-shot is the push variant (each rank writes its
#: partial rows straight into the owning rank's slot, readers clear what they consumed): it
#: beats the one-shot from 512 rows up (4x B300: 30.7 vs 42.5 us at 512, 81.5 vs 133.1 at
#: 2048, 109.9 vs 192.8 at 3072; 258 us at 8192 against 429 for the flagged two-shot it
#: replaces). The choice is a function of the row count alone, and every rank derives the
#: row count from the same batch, so all ranks take the same side.
TWOSHOT_ABOVE = 512
#: MTP3 / MTP5 wide decode levers, wide workspaces only, rows above `QUANT_TWOSHOT_ABOVE`:
#: the pre-MoE two-shot (`fused_quant_twoshot`) and the AR + norm one-shot at 65..`TWOSHOT_ABOVE`
#: rows (`fused_wide`: the post-MoE, dense-layer and MTP-layer collectives of a 128 / 192-row
#: step) are built with
#:   * the CTA head (`TRTLLM_GLM52_WIDE_AR_CTA_HEAD`, default 1): thread 0 of each CTA loads the
#:     slot signal and the three slot counts for the CTA, and the input row loads ahead of them.
#:     Every thread loading them put one L2 line under 147k loads per launch at 192 rows (768
#:     CTAs x 192 threads): per-CTA stamps on 4x B300, 1.8 us from griddepcontrol.wait to the
#:     slot, 0.9 us with the CTA head;
#:   * self-cleaning slots (`TRTLLM_GLM52_WIDE_AR_SELF_CLEAN`, default 1): every reader puts the
#:     not-ready marker back on what it consumed and the launch records a clear count of 0, so
#:     the launch two calls on clears nothing. Each rank's slots are its own, so a self-cleaning
#:     launch interleaves with the clearing builds (MTP1 workspaces, 1..64 rows, the pure AR).
#:     A recorded count of 0 promises a clean slot, and the clearing builds trust it: any read
#:     or write path added to a self-cleaning build must have every vector the peers write into
#:     the slot read and reset by its reader in the same launch, or the launch that next uses
#:     the slot takes the stale vector as ready (a wrong sum, no hang). The compile functions
#:     reject the poll modes that have no reset (ready flags, batch / ready-mask polls).
#: 4-GPU decode-step chain (trunk dense + 2 MoE layers + the MTP layer, real all-reduce and
#: routing, 80 alternating blocks), us per chain step: CTA head -4.3 at 192 rows /
#: -1.2 at 128; with self-cleaning slots -12.8 / -8.9. Same bytes as the head builds on every
#: rank. At 32 rows (the 1-token draft steps) the CTA head measured +0.4 us per AR pair alone
#: (same stamps), so 1..64 rows keep `fused` / `fused_quant`.
_WIDE_AR_ENV = {
    "cta_head": "TRTLLM_GLM52_WIDE_AR_CTA_HEAD",
    "self_clean": "TRTLLM_GLM52_WIDE_AR_SELF_CLEAN",
}


def wide_ar_levers() -> dict:
    """The wide AR switches as read at build (each `0` restores the head build's behaviour)."""
    return {k: os.environ.get(v, "1") == "1" for k, v in _WIDE_AR_ENV.items()}


def ensure_process_group(rank: int, world: int) -> None:
    """Bring up `torch.distributed` for this rank if nothing has yet.

    `symm_mem.rendezvous` needs a process group. The engine's ranks are
    MPI-spawned and TRT-LLM's collectives use their own transport, so there may
    be none. Three cases, in order:

    * A group is already up (the engine brought one up, or a test launcher
      did): it has to be this rank in a group of `world`, and is used as is.
    * The rendezvous address is in the environment (`MASTER_ADDR`,
      `MASTER_PORT`, `RANK`, `WORLD_SIZE`): the group comes up from it.
      MPI is not touched.
    * Otherwise (the engine under MPI): rank 0 draws a free port and the
      address is broadcast over `MPI.COMM_WORLD`, as TRT-LLM's MegaMoE path
      does. Only this case needs MPI, so only it checks that `COMM_WORLD` is
      the TP group.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        assert dist.get_world_size() == world and dist.get_rank() == rank, (
            f"the process group is rank {dist.get_rank()} of {dist.get_world_size()}; "
            f"the collective was built for rank {rank} of {world}"
        )
        return
    if not all(os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        assert comm.Get_size() == world and comm.Get_rank() == rank, (comm.Get_size(), world)
        addr = None
        if rank == 0:
            import socket

            with socket.socket() as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            # The bare hostname does not resolve on every node; its address does.
            addr = (socket.gethostbyname(socket.gethostname()), port)
        host, port = comm.bcast(addr, root=0)
        os.environ["MASTER_ADDR"] = host
        os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )


@dataclass(frozen=True)
class LamportWorkspace:
    """One set of peer buffers and compiled kernels serving every collective in the forward.

    Every compiled kernel is symbolic in the row count, so one compile each covers every
    row count up to `max_rows`; the kernel reads the active rows off its input's shape.
    `state` is the device slot counter: every launch of any of the four kernels reads and
    advances it on the device, so the workspace's tensors are shared state even though the
    dataclass is frozen. One workspace per rank per process.
    """

    hidden: int
    eps: float
    max_rows: int
    rank: int
    world: int
    #: This rank's `[3, world, max_rows * hidden]` bf16 symmetric buffer and its
    #: `[3, world, max_rows * hidden / 8]` int32 flags.
    buffer: torch.Tensor
    flags: torch.Tensor
    #: Every rank's buffer and flags, mapped into this rank, in rank order.
    peer_buffers: list
    peer_flags: list
    #: `[6]` int32: entry counter, current slot, per-slot vector counts, release epoch.
    state: torch.Tensor
    #: The multicast address of `buffer`, or 0 when any rank has none.
    multicast_base: int
    pure: object
    fused: object
    fused_quant: object
    fused_twoshot: object
    #: The lm_head's vocabulary all-gather (bf16 `[M, 38720]` shards to fp32 `[M, 154880]`)
    #: on the same buffers and slot state, or None when the group did not build it.
    gather: object = None
    #: The fused two-shot that also writes the normalized rows as an MXF8 operand (the next
    #: layer's qkva GEMM input), or None when the group did not build it.
    fused_twoshot_x8: object = None
    #: `fused_quant` forming the sum two-shot, for rows above `QUANT_TWOSHOT_ABOVE`; None
    #: unless the workspace was built `wide`.
    fused_quant_twoshot: object = None
    #: `fused` with the wide-decode levers (CTA head, self-cleaning slots) for rows above
    #: `QUANT_TWOSHOT_ABOVE` up to `TWOSHOT_ABOVE`; None unless built `wide` with a lever on.
    fused_wide: object = None


def build_workspace(
    hidden: int, eps: float, max_rows: int, rank: int, world: int, wide: bool = False
) -> LamportWorkspace:
    """Bring up the group, pin the peer buffers, agree multicast, compile the kernels.

    Collective: every rank of the group calls it, with the same `hidden`, `eps`,
    `max_rows` and `wide`. Allocates `3 * world * max_rows * hidden` bf16 plus a quarter
    of that in int32 flags of symmetric memory per rank, and compiles six CuTe DSL kernels
    (pure, fused, fused+quant, fused two-shot, its MXF8-out variant, the lm_head
    all-gather) for this rank on the current device, plus, `wide` (an engine whose decode
    steps run above 64 rows: max_draft_len > 1), the two-shot fused+quant
    `fused_quant_twoshot` and the 65..512-row `fused_wide`, both with the wide-decode levers
    (`wide_ar_levers`). Launches nothing beyond the buffer fills and the one-element
    multicast and all-gather agreements.

    The multicast decision is agreed across the group: a rank compiling the multicast
    variant while a peer did not would wait on a store that never lands.
    """
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    ensure_process_group(rank, world)
    dev = torch.device("cuda", torch.cuda.current_device())
    numel = max_rows * hidden
    buffer = symm_mem.empty((_PING_PONG, world, numel), dtype=torch.bfloat16, device=dev)
    # Every slot starts not-ready, or the first exchange reads uninitialized memory as data.
    buffer.view(torch.int16).fill_(_NEG_ZERO_BF16)
    flags = symm_mem.empty(
        (_PING_PONG, world, numel // _BF16_VEC), dtype=torch.int32, device=dev
    ).zero_()
    group = dist.group.WORLD
    name = getattr(group, "group_name", group)
    bh = symm_mem.rendezvous(buffer, group=name)
    fh = symm_mem.rendezvous(flags, group=name)
    peer_buffers = [bh.get_buffer(peer, buffer.shape, buffer.dtype) for peer in range(world)]
    peer_flags = [fh.get_buffer(peer, flags.shape, flags.dtype) for peer in range(world)]
    state = torch.zeros(_SIGNAL_STATE, dtype=torch.int32, device=dev)
    try:
        mc = int(getattr(bh, "multicast_ptr", 0) or 0)
    except Exception:  # noqa: BLE001 -- no multicast on this fabric is an answer
        mc = 0
    # Agreed over the process group that `rendezvous` just used, so it needs no MPI:
    # the minimum of every rank's "has multicast" flag.
    agreed = torch.tensor([1 if mc != 0 else 0], dtype=torch.int32, device=dev)
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=group)
    use_mc = bool(agreed.item())

    common = dict(use_ready_flags=False, use_pdl=True)
    pure = _ar.compile_pure_for_rank(
        rank, world, dev.index, hidden, pdl_release_policy=_PDL_POLICY, **common
    )
    fused = _ar.compile_fused_for_rank(
        rank,
        world,
        dev.index,
        hidden,
        eps=eps,
        pdl_release_policy=_FUSED_PDL_POLICY,
        prefetch_fused_inputs=True,
        multicast_publish=use_mc,
        **_FUSED_OPTIONS,
        **common,
    )
    # Same transport, slots and signal state as the fused op; only the epilogue differs,
    # so the two interleave.
    fused_quant = _ar.compile_fused_out_quant_for_rank(
        rank,
        world,
        dev.index,
        hidden,
        eps=eps,
        pdl_release_policy=_FUSED_QUANT_PDL_POLICY,
        prefetch_fused_inputs=True,
        multicast_publish=use_mc,
        **_FUSED_OPTIONS,
        **common,
    )
    # Every rank derives `wide` from the same engine config, so the whole group builds -- and
    # at the same row counts selects -- the two-shot kernel. It publishes by unicast stores
    # whether or not the fabric has multicast.
    fused_quant_twoshot = None
    fused_wide = None
    if wide:
        lv = wide_ar_levers()
        # The two levers change the slot protocol, so the ranks must agree on them: a rank
        # self-cleaning while a peer is not reads a stale vector as ready (a wrong sum).
        # Agreed like multicast, the minimum over the group.
        agreed_lv = torch.tensor([int(lv[k]) for k in _WIDE_AR_ENV], dtype=torch.int32, device=dev)
        dist.all_reduce(agreed_lv, op=dist.ReduceOp.MIN, group=group)
        lv = dict(zip(_WIDE_AR_ENV, (bool(v) for v in agreed_lv.tolist())))
        head = dict(cta_head=True, early_input_load=True) if lv["cta_head"] else {}
        fused_quant_twoshot = _ar.compile_fused_out_quant_for_rank(
            rank,
            world,
            dev.index,
            hidden,
            eps=eps,
            pdl_release_policy=_FUSED_QUANT_PDL_POLICY,
            prefetch_fused_inputs=True,
            multicast_publish=False,
            two_shot=True,
            self_clean=lv["self_clean"],
            **head,
            **_FUSED_OPTIONS,
            **common,
        )
        if head or lv["self_clean"]:
            fused_wide = _ar.compile_fused_for_rank(
                rank,
                world,
                dev.index,
                hidden,
                eps=eps,
                pdl_release_policy=_FUSED_PDL_POLICY,
                prefetch_fused_inputs=True,
                multicast_publish=use_mc,
                self_clean=lv["self_clean"],
                **head,
                **_FUSED_OPTIONS,
                **common,
            )
    fused_twoshot = _ar.compile_fused_twoshot_for_rank(
        rank,
        world,
        dev.index,
        hidden,
        eps=eps,
        use_pdl=True,
        pdl_release_policy=_PDL_POLICY,
    )
    # The MXF8-out two-shot, agreed like the gather: only the next layer's qkva reads its
    # extra output, so a group without it keeps that GEMM's own activation quant.
    try:
        fused_twoshot_x8 = _ar.compile_fused_out_mxf8_twoshot_for_rank(
            rank,
            world,
            dev.index,
            hidden,
            eps=eps,
            use_pdl=True,
            pdl_release_policy=_PDL_POLICY,
        )
    except Exception as e:  # noqa: BLE001 -- that group's qkva keeps its activation quant
        logger.warning(f"modeling_v2 GLM-5.2: MXF8-out two-shot all-reduce build failed ({e})")
        fused_twoshot_x8 = None
    built = torch.tensor([0 if fused_twoshot_x8 is None else 1], dtype=torch.int32, device=dev)
    dist.all_reduce(built, op=dist.ReduceOp.MIN, group=group)
    if not bool(built.item()):
        fused_twoshot_x8 = None
    # The all-gather is separate on purpose: a group that cannot build it keeps its
    # logits on upstream's gather, never its all-reduces. Agreed like multicast: a rank
    # launching the gather while a peer did not would wait on stores that never land.
    try:
        gather = _ag.compile_for_rank(
            rank,
            world,
            dev.index,
            _ag.LM_HEAD_SHARD_N,
            out_fp32=True,
            use_pdl=True,
            pdl_release_policy=_PDL_POLICY,
        )
    except Exception as e:  # noqa: BLE001 -- that group's logits take upstream's gather
        logger.warning(f"modeling_v2 GLM-5.2: logits all-gather build failed ({e})")
        gather = None
    built = torch.tensor([0 if gather is None else 1], dtype=torch.int32, device=dev)
    dist.all_reduce(built, op=dist.ReduceOp.MIN, group=group)
    if not bool(built.item()):
        gather = None
    return LamportWorkspace(
        hidden=hidden,
        eps=eps,
        max_rows=max_rows,
        rank=rank,
        world=world,
        buffer=buffer,
        flags=flags,
        peer_buffers=peer_buffers,
        peer_flags=peer_flags,
        state=state,
        multicast_base=mc if use_mc else 0,
        pure=pure,
        fused=fused,
        fused_quant=fused_quant,
        fused_twoshot=fused_twoshot,
        gather=gather,
        fused_twoshot_x8=fused_twoshot_x8,
        fused_quant_twoshot=fused_quant_twoshot,
        fused_wide=fused_wide,
    )


def tail(workspace: LamportWorkspace, fused: bool) -> list:
    """The workspace operands every launch ends with. `signal` is dead with device signal
    state on, so it is 0; only the fused signatures carry `multicast_base`."""
    out = [workspace.peer_buffers, workspace.peer_flags, workspace.state]
    if fused:
        out.append(cutlass.Int64(workspace.multicast_base))
    out.append(cutlass.Int64(0))
    out.append(_raw_stream())
    return out


def check_rows(rows: int, workspace: LamportWorkspace) -> None:
    """The kernel takes the row count from its input, so a call above the token budget the
    peer buffers were sized for writes past every peer's buffer."""
    assert rows <= workspace.max_rows, (
        f"{rows} rows exceed the collective's budget of {workspace.max_rows}"
    )


def check_rows_by_numel(numel: int, workspace: LamportWorkspace) -> None:
    """The same budget for a collective whose rows are not hidden-wide (the all-gather's
    `[M, 38720]` shard): its elements must fit one peer region, `max_rows * hidden`."""
    budget = workspace.max_rows * workspace.hidden
    assert numel <= budget, f"{numel} elements exceed the collective's budget of {budget}"
