# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification of the MoE engine's all-reduce push (``m3_moe_engine`` with ``arp``), 2 ranks (SM107).

The decode forward's MoE boundary (MOE_PUSH rows): the engine pushes its bf16 sum (routed + shared) into set A of
the boundary's multicast exchange while it writes it, and the boundary all-reduce runs with ``prepushed`` 2, reading
the pushed words instead of loading and pushing the rows itself. Certified against the unpushed boundary the target
runs otherwise: the same engine call without ``arp``, then ``m3_mc_allreduce_norm`` with ``prepushed`` 0. Every
output must be bit for bit the same: the engine's own output tensor, the updated residual, the normed rows and, with
``quant``, the MXFP8 hand-off it publishes. Both read the same bf16 words, so the reduction and the norm see equal
inputs; any difference is a push that stored the wrong word, slot or parity.

Each rank holds its own expert shard (operand seed per rank) and the same hidden rows, residual and norm weight, as
the TP2 decode step does. Eager calls alternate pushed and unpushed in both orders so both call parities of the
exchange are exercised; a CUDA graph of the pushed engine + boundary is replayed on fresh rows against eager
unpushed references. The engine's early PDL release (EARLY_AR, the target's default) is checked against the release
after the push: scheduling only, the same bits.

Its own launcher, as ``_m3_mc_allreduce_norm_op_matrix.py`` is: run plainly it spawns one rank process per device
in CUDA_VISIBLE_DEVICES and kills them all on the first failure or on the deadline (a rank whose peer never pushes
polls forever). The collected entry point is ``test_modeling_v2_m3_moe_push_op_matrix.py``.

    CUDA_VISIBLE_DEVICES=0,1 python _m3_moe_push_op_matrix.py
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "m3_moe_engine requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 1200

HIDDEN = 6144
EPS = 1e-6
CLUSTER = 8
MAX_TOKENS = 8
EMPTY_WORD = -2147483648
RESERVED_SMS = 24
OPERAND_SEED = 20261013

# Bound inside the rank body: importing the entries pulls in tensorrt_llm.
ar = None
sf_written: Any = None
engine_op: Any = None
ar_push_desc: Any = None
moe_state: Any = None
operands: Any = None
prequant: Any = None
RANK = 0
WORLD = 1
WS: Any = None


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _residual(seed: int, m: int) -> torch.Tensor:
    return (torch.randn(m, HIDDEN, generator=_gen(seed + 7), device="cuda") * 4.0).to(
        torch.bfloat16
    )


def _norm_w(seed: int) -> torch.Tensor:
    g = _gen(seed + 11)
    return (1.0 + 0.25 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)


def _make_workspace():
    """The 8-token bucket's exchange, as ``McAllReduceWorkspace.create`` builds it, over the torch.distributed
    group."""
    from tensorrt_llm._torch.distributed.ops import _make_mnnvl_mcast_buffer
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=RANK)
    words = 4 * MAX_TOKENS * WORLD * (HIDDEN // 2)
    group = dist.group.WORLD
    handle = _make_mnnvl_mcast_buffer(group, words * 4, mapping, False)
    uc = handle.get_uc_buffer(RANK, (words,), torch.int32, 0)
    mc = handle.get_mc_buffer((words,), torch.int32, 0)
    uc.fill_(EMPTY_WORD)
    torch.cuda.synchronize()
    dist.barrier()
    ws = ar.McAllReduceWorkspace(
        uc=uc,
        mc=mc,
        flags={},
        rank=RANK,
        world_size=WORLD,
        hidden=HIDDEN,
        handle=handle,
        comm=group,
        max_tokens=MAX_TOKENS,
    )
    ws.flags_for(CLUSTER)
    return ws


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    assert out.dtype == ref.dtype and out.shape == ref.shape, (
        where,
        out.dtype,
        ref.dtype,
        out.shape,
    )
    a, b = out.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
    bad = (a != b).sum().item()
    assert bad == 0, f"{where}: {bad} of {a.numel()} bytes differ"


_STATES = {}
_LAYERS = {}


def _layer(m: int, case: str, early_release: bool = True):
    """This rank's engine layer for ``m`` tokens: its own expert shard; ``concentrated`` routes every token to the
    same four experts. ``early_release``: the state's EARLY_AR build (the target's default) or the release after the
    push."""
    if (m, early_release) not in _STATES:
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        _STATES[m, early_release] = moe_state(
            torch.device("cuda"),
            num_tokens=m,
            num_ctas=sms - RESERVED_SMS,
            shared=True,
            early_release=early_release,
        )
    key = (m, case, early_release)
    if key not in _LAYERS:
        ops = operands.experts(OPERAND_SEED + RANK)
        bias = ops.bias.clone()
        if case == "concentrated":
            bias[list(operands.CONCENTRATED)] += 1.0
        _LAYERS[key] = _STATES[m, early_release].layer(
            *ops.tensors,
            bias,
            router_weight=ops.router_weight,
            fc31_input_scale=ops.fc31_input_scale,
            shared_weights=ops.shared,
        )
    return _LAYERS[key]


def _desc():
    flags = WS.flags_for(CLUSTER)
    return ar_push_desc(
        WS.mc.data_ptr(), flags.data_ptr(), RANK, WORLD, WS.max_tokens, flags.device
    )


def _boundary(x, residual, norm_w, quant, prepushed):
    prequant.SLOT.clear()
    outs = ar.m3_mc_allreduce_norm(
        x, residual, norm_w, EPS, WS, use_pdl=True, quant=quant, prepushed=prepushed
    )
    assert outs is not None, "the boundary all-reduce declined its operands"
    pub = prequant.SLOT.pending
    prequant.SLOT.clear()
    return outs, (None if pub is None else (pub[3].clone(), pub[4].clone()))


def _unpushed(hidden, layer, residual, norm_w, quant):
    out = engine_op(hidden, layer)
    (normed, res), pub = _boundary(out, residual.clone(), norm_w, quant, 0)
    return out.clone(), normed.clone(), res.clone(), pub


def _pushed(hidden, layer, residual, norm_w, quant):
    out = engine_op(hidden, layer, arp=_desc())
    (normed, res), pub = _boundary(out, residual.clone(), norm_w, quant, 2)
    return out.clone(), normed.clone(), res.clone(), pub


def _compare(got, ref, quant: bool, where: str) -> None:
    _assert_bitwise(got[0], ref[0], f"{where} engine output")
    _assert_bitwise(got[2], ref[2], f"{where} updated residual")
    _assert_bitwise(got[1], ref[1], f"{where} normed")
    if quant:
        assert got[3] is not None and ref[3] is not None, f"{where}: no MXFP8 hand-off published"
        _assert_bitwise(got[3][0], ref[3][0], f"{where} e4m3 rows")
        # Only the scale atoms of the m rows are written; the swizzled buffer's padding is not.
        written = sf_written(got[0].shape[0], HIDDEN // 32, got[3][1].device)
        _assert_bitwise(got[3][1][written], ref[3][1][written], f"{where} UE8M0 scales")


# ---------------------------------------------------------------- checks


def check_pushed_boundary_is_the_unpushed_one() -> None:
    """Every token count, both routing cases, with and without the MXFP8 hand-off, both call orders."""
    for m in range(1, MAX_TOKENS + 1):
        for case in ("random", "concentrated"):
            layer = _layer(m, case)
            for quant in (False, True):
                for seed in (1, 2):
                    s = (
                        1000 * m
                        + 10 * seed
                        + (5 if quant else 0)
                        + (500 if case == "concentrated" else 0)
                    )
                    hidden = operands.hidden_states(m, s)
                    residual, norm_w = _residual(s, m), _norm_w(s)
                    where = f"m={m} {case} quant={quant} seed={seed}"
                    if seed == 1:
                        ref = _unpushed(hidden, layer, residual, norm_w, quant)
                        got = _pushed(hidden, layer, residual, norm_w, quant)
                    else:
                        got = _pushed(hidden, layer, residual, norm_w, quant)
                        ref = _unpushed(hidden, layer, residual, norm_w, quant)
                    torch.cuda.synchronize()
                    _compare(got, ref, quant, where)


def check_graph_replays() -> None:
    """A captured pushed engine + prepushed boundary, replayed on fresh rows, equals the eager unpushed path."""
    for m in (1, 4, 8):
        layer = _layer(m, "random")
        hidden = operands.hidden_states(m, 90000 + m).clone()
        residual = _residual(90000 + m, m).clone()
        norm_w = _norm_w(90000 + m)
        desc = _desc()
        for _ in range(
            2
        ):  # warm both parities eagerly; every push is consumed by its prepushed boundary
            _pushed(hidden, layer, residual, norm_w, False)
            _unpushed(hidden, layer, residual, norm_w, False)
        torch.cuda.synchronize()
        dist.barrier()
        res_in = residual.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            g_out = engine_op(hidden, layer, arp=desc)
            g_normed, g_res = ar.m3_mc_allreduce_norm(
                g_out, res_in, norm_w, EPS, WS, use_pdl=True, prepushed=2
            )
        torch.cuda.synchronize()
        dist.barrier()
        for rep in range(6):
            s = 91000 + 100 * m + rep
            fresh_h, fresh_r = operands.hidden_states(m, s), _residual(s, m)
            hidden.copy_(fresh_h)
            res_in.copy_(fresh_r)
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            got = (g_out.clone(), g_normed.clone(), g_res.clone(), None)
            ref = _unpushed(fresh_h, layer, fresh_r, norm_w, False)
            torch.cuda.synchronize()
            _compare(got, ref, False, f"graph m={m} replay {rep}")
        dist.barrier()


def check_early_release_is_scheduling_only() -> None:
    """The pushed engine + boundary with the early PDL release and with the release after the push: the same bits."""
    for m in range(1, MAX_TOKENS + 1):
        for quant in (False, True):
            s = 50000 + 10 * m + (5 if quant else 0)
            hidden = operands.hidden_states(m, s)
            residual, norm_w = _residual(s, m), _norm_w(s)
            early = _pushed(hidden, _layer(m, "random", True), residual, norm_w, quant)
            late = _pushed(hidden, _layer(m, "random", False), residual, norm_w, quant)
            torch.cuda.synchronize()
            _compare(early, late, quant, f"m={m} quant={quant} early vs late release")


CHECKS = [
    check_pushed_boundary_is_the_unpushed_one,
    check_early_release_is_scheduling_only,
    check_graph_replays,
]


def _run_one_rank() -> int:
    """Body of one spawned rank: run every check; a failing rank exits nonzero and the launcher kills its
    siblings."""
    global RANK, WORLD, ar, sf_written, engine_op, ar_push_desc, moe_state, operands, prequant, WS
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", init_method="env://", device_id=torch.device("cuda", rank)
    )
    RANK, WORLD = rank, world
    assert WORLD == 2, f"the push is certified at 2 ranks, got {WORLD}"

    import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.m3_mc_allreduce as _ar  # noqa: E501
    import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.m3_prequant as _pq  # noqa: E501
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.comm.m3_mc_allreduce_norm import (  # noqa: E501
        sf_written as _sf_written,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.moe.m3_moe_engine import (  # noqa: E501
        M3MoeState,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.moe.m3_moe_engine import (  # noqa: E501
        ar_push_desc as _desc_fn,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.moe.m3_moe_engine import (  # noqa: E501
        m3_moe_engine as _op,
    )

    ar, prequant, moe_state, ar_push_desc, engine_op = _ar, _pq, M3MoeState, _desc_fn, _op
    sf_written = _sf_written
    # Spawned by file path, not collected by pytest: the sibling moe/ operands the plain way.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moe"))
    import _m3_moe_operands as _operands

    operands = _operands
    WS = _make_workspace()

    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback  # only the rank subprocess's failure path needs it

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # A rank that leaves early wedges its peer's polls; the launcher kills them on this exit.
            os._exit(13)
        print(f"[rank {RANK}] {check.__name__} passed", flush=True)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    """One worker process per claimed device, each in its own session; on the first nonzero exit or on the
    deadline every session is killed."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the 2 devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size == 2, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); the matrix runs at 2"
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        TLLM_DISABLE_MPI="1",
    )
    # Drop a launching srun step's PMIx / MPI namespace: children registered in it get the whole step SIGKILLed
    # when the first one exits hard. These ranks rendezvous through torch.distributed's env://.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    print(f"[launcher] spawning {world_size} ranks, rendezvous on :{port}", flush=True)
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
            start_new_session=True,
        )
        for r in range(world_size)
    ]

    def _kill_all() -> None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()

    deadline = time.monotonic() + DEADLINE_S
    live = set(range(world_size))
    failure: Optional[str] = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {DEADLINE_S}s (wedged)"
            break
        for r in sorted(live):
            code = children[r].poll()
            if code is None:
                continue
            live.discard(r)
            if code != 0:
                failure = f"rank {r} exited {code}"
                break
        time.sleep(0.5)
    if failure is not None:
        _kill_all()
        raise AssertionError(f"the {world_size}-rank run failed: {failure}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")
