# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One real MoE layer's attention-DP round trip: the dep4 target's `_moe`, on four GPUs of one
node and on eight ranks over two nodes of four (dep8).

Under attention DP the ranks hold different tokens, in different numbers, and may run
different targets in the same step -- one prefilling, another decoding, another holding only
the engine's one-token dummy. Each MoE layer pads its rows to the largest rank's count, gathers
every rank's rows before the router, runs this rank's expert window over them (32 experts at
four ranks, 16 at eight) in calls of at most `_EXPERT_CALL_TOKENS[world]` rows (131072 at four,
32768 at eight; on an engine with chunked prefill, as the 1M config at max_seq_len 1048576 runs,
`_EXPERT_CALL_TOKENS_CHUNKED[world]`, 32768 at both), and reduce-scatters the partial sums
back. A padding, chunking or ordering defect there raises nothing: the collectives pair by
position and move equal byte counts, and every rank simply gets other rows.

So, on layer 1 of Mistral-Large-4-1T-A52B-NVFP4 itself (its router, its shared expert, each
rank's own window of its routed experts, laid out by the dep4 weight table's own manifest rows),
every rank runs the target's own `PrefillTarget._moe` or `DecodeTarget._moe` on its own rows,
and every rank's result is compared with the same layer computed on one rank over all ranks'
tokens concatenated: the shared expert plus every window's partial sum, added in fp32.

At four ranks, in each table's expert calls (the 32K config's and the 1M config's):

* unequal ranks: 1003 / 1 / 16 / 257 rows, ranks 0 and 3 prefilling, rank 1 decoding the
  one-row dummy and rank 2 sixteen rows, both padded to 1003 beside the prefilling peer;
* the gathered maximum: 32768 / 1 / 16 / 9001 rows, 131072 gathered rows in one expert call
  (in four of 32768 at the 1M config), the decoding ranks' few rows among them.

At eight ranks, where both tables call 32768 rows:

* unequal ranks: 1003 / 1 / 16 / 257 / 1 / 40 / 700 / 3 rows, ranks 0, 3, 5 and 6 prefilling
  (rank 5's 40 rows padded to 1003), ranks 1 and 4 decoding the one-row dummy and ranks 2 and
  7 decoding beside them; 8024 gathered rows, one expert call;
* past one call: 5000 / 1 / 16 / 4100 / 1 / 33 / 2048 / 7 rows, 40000 gathered rows in two
  expert calls (32768 + 7232), the decoding ranks' few rows among them;
* the gathered maximum: 32768 / 1 / 16 / 9001 / 1 / 2 / 300 / 32768 rows, 262144 gathered
  rows in eight expert calls.

Each rank marks the gathered padding rows as the targets' forwards do (`_padding_rows`): they
take expert id -1 after the routing and reach no rank's experts.

At both, CUDA graphs: every rank decoding the same captured batch, 1 and 16 rows (4 and 64
gathered at four ranks, 8 and 128 at eight), captured once and replayed on new tokens, as the
engine replays a decode graph only when every rank is generation-only at the same padded batch
(every count equal, so no padding rows to mark).

The gate is the runner entry's, against the row's scale: 16 bf16 ulp per element, 2 ulp
relative RMS (the reduce-scatter sums bf16 partials, the reference sums them in fp32). Tokens
whose 4th and 5th router logits are within `TIE` of each other are not compared: the router
GEMM's rounding depends on its row count, and the concatenated and gathered calls differ in it.
Two defects are shown to land far outside the gate: padding placed ahead of the rows instead of
behind them, and the reduce-scatter handing each rank its neighbour's rows. Each case's expert
calls are counted against its gathered rows cut at the call size.

The four ranks are processes this file starts itself, once per table, on the first four
visible devices, rendezvousing through torch.distributed with MPI disabled; the target's
collectives then run on the process group's communicator, the flavor the engine's own
collectives take without MPI. Rank 0 computes the reference and hands it to the others.

The eight ranks are already in place: pytest on each of eight MPI ranks started by the
session's launcher over two nodes of four (`mn_mpirun.sh 4`), each rank on its node-local
device, running the catalog's allgather and reduce-scatter on the MPI-session communicator over
both nodes (process_group None: the symbol a dep8 engine launched under MPI calls); every rank
computes the reference itself. It is gated on the launched MPI world (`get_mpi_world_size()`),
not on the devices a node shows: each node sees four.

Each world runs only in a launch of its own size: the four-rank test unlaunched (or in a launch
of four), the eight-rank test in a launch of eight. Skipped where the checkpoint is absent
(looked up as `Mistral-Large-4-1T-A52B-NVFP4` or `mistral-large-4-nvfp4` under
`LLM_MODELS_ROOT`), below four GPUs or eight launched ranks, or off sm_103.
"""

from __future__ import annotations

import collections
import faulthandler
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

#: Four ranks on one node: processes this file starts, over torch.distributed.
SPAWNED_WORLD = 4
#: Eight ranks over two nodes of four: MPI ranks a launcher starts.
MPI_WORLD = 8
#: The whole job's budget; a broken collective hangs rather than raising. A rank wedged in a
#: collective a sibling never joined dumps its stack and exits by then.
DEADLINE_S = 2400
_WORKER_FLAG = "--rank-worker"
CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")
SHARD_INDEX = "consolidated.safetensors.index.json"

#: The MoE layer: the checkpoint's largest FC2 global scale.
LAYER = 1
#: Per world: (name, rows per rank, target per rank).
CASES = {
    4: (
        ("unequal", (1003, 1, 16, 257), ("prefill", "decode", "decode", "prefill")),
        ("gathered maximum", (32768, 1, 16, 9001), ("prefill", "decode", "decode", "prefill")),
    ),
    8: (
        (
            "unequal",
            (1003, 1, 16, 257, 1, 40, 700, 3),
            ("prefill", "decode", "decode", "prefill", "decode", "prefill", "prefill", "decode"),
        ),
        (
            "past one call",
            (5000, 1, 16, 4100, 1, 33, 2048, 7),
            ("prefill", "decode", "decode", "prefill", "decode", "prefill", "prefill", "decode"),
        ),
        (
            "gathered maximum",
            (32768, 1, 16, 9001, 1, 2, 300, 32768),
            ("prefill", "decode", "decode", "prefill", "decode", "prefill", "prefill", "prefill"),
        ),
    ),
}
#: The four-rank round trip's expert calls by engine, the modeling's table `_derive_first_step`
#: picks there by the engine's chunked prefill flag alone (block reuse does not change it):
#: `_EXPERT_CALL_TOKENS` without it (the 32K serving config: max_seq_len = max_num_tokens =
#: 32768), `_EXPERT_CALL_TOKENS_CHUNKED` with it (the 1M config: max_seq_len 1048576).
EXPERT_CALLS = {"32k": "_EXPERT_CALL_TOKENS", "1m": "_EXPERT_CALL_TOKENS_CHUNKED"}
#: Decode batches captured on every rank.
GRAPH_BATCHES = (1, 16)
#: One bf16 ulp, relative: 2^-8 of the binade top.
ULP = 2.0**-8
GATE_ELT, GATE_RMS = 16.0, 2.0
#: A routing the row count's rounding may flip: the 4th and 5th logits closer than this
#: fraction of the row's largest logit (the router GEMM moves a logit by up to 133 fp32 ulp of
#: the row's largest, about 1.6e-5 of it).
TIE = 1e-4
#: The largest e2m1 magnitude and the largest finite e4m3 one.
E2M1_MAX, E4M3_MAX = 6.0, 448.0
#: The layer's parameters, in the weight table's names.
PARAMS = (
    "router",
    "mlp_gu",
    "mlp_gu_s",
    "mlp_dn",
    "mlp_dn_s",
    "fc1_w",
    "fc1_s",
    "fc2_w",
    "fc2_s",
    "isc1",
    "isc2",
    "ws2_13",
    "ws2_2",
)
WINDOW_PARAMS = ("fc1_w", "fc1_s", "fc2_w", "fc2_s", "ws2_13", "ws2_2")


def _checkpoint_dir() -> Path | None:
    from test_common.llm_data import llm_models_root

    root = llm_models_root()
    if root is None:
        return None
    for name in CHECKPOINTS:
        path = Path(root) / name
        if (path / "params.json").is_file() and (path / SHARD_INDEX).is_file():
            return path
    return None


def _modules():
    """The dep4 target's modeling and weight table. Imported where a rank runs, after its
    communicator is up."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4 import (  # noqa: E501
        modeling as M,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4 import (  # noqa: E501
        weights as W,
    )

    return M, W


def _config(ck: Path):
    """The configuration the engine adapts from the checkpoint's params.json."""
    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader

    return MistralConfigLoader().load_pretrained_config(str(ck))


# ---------------------------------------------------------------------------- one rank's body


def _core(config, rank: int, world: int) -> SimpleNamespace:
    """The surface the weight table reads off the core, for rank `rank` of `world`. The mapping
    is stated rather than built: with MPI disabled a `Mapping` takes the calling process's own
    rank from torch.distributed whatever `rank` it is given, and one rank loads the other ranks'
    windows for the reference."""
    mapping = SimpleNamespace(
        world_size=world,
        rank=rank,
        tp_size=world,
        tp_rank=rank,
        moe_tp_size=1,
        moe_ep_size=world,
        moe_ep_rank=rank,
        enable_attention_dp=True,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=config,
            mapping=mapping,
            torch_dtype=torch.bfloat16,
        )
    )


def _load(W, ck: Path, config, weight_map: dict, rank: int, world: int, names) -> tuple:
    """Layer `LAYER`'s parameters `names` as rank `rank` of `world`'s `load` leaves them: every
    manifest source read off the shard files and copied to its index through its transform,
    then each row's `layout` on every expert slot. Also the rank's dimension bundle."""
    from safetensors import safe_open

    core = _core(config, rank, world)
    d = W.MODEL_WEIGHTS.dims(core)
    manifest = W.MODEL_WEIGHTS.manifest(core)
    rows = {
        r.name: r
        for r in W.MODEL_WEIGHTS.WEIGHTS
        if r.name in names and LAYER in (r.layer_indices(d) or [])
    }
    keys = {k for n in names for ckpt, *_ in manifest[f"l{LAYER}_{n}"] for k in W._keys(ckpt)}
    by_file = collections.defaultdict(list)
    for key in keys:
        by_file[weight_map[key]].append(key)
    stored = {}
    for name, file_keys in by_file.items():
        with safe_open(str(ck / name), framework="pt", device="cpu") as handle:
            for key in file_keys:
                stored[key] = handle.get_tensor(key)
    out = {}
    for n in names:
        row = rows[n]
        param = torch.empty(row.shape(d), dtype=row.dtype or d.dtype, device="cuda")
        for ckpt, index, transform in manifest[f"l{LAYER}_{n}"]:
            dst = param if index is None else param[index]
            if isinstance(ckpt, tuple):
                src = transform(tuple(stored[k].cuda() for k in ckpt), core)
            else:
                src = stored[ckpt].cuda()
                if transform is not None:
                    src = transform(src.contiguous(), core)
            dst.copy_(src)
        if row.layout is not None:
            for slot in range(param.shape[0]):
                param[slot].copy_(row.layout(param[slot]))
        out[n] = param
    return out, d


def _layer(M, p: dict, window: dict) -> SimpleNamespace:
    """The `L` `_moe` reads, as `_layer_views` takes it: GEMM operands and FP8 scale storage
    transposed, the window's stacks and the runner's call scalars."""
    return SimpleNamespace(
        mlp_gu=p["mlp_gu"],
        mlp_gu_s=p["mlp_gu_s"].t(),
        mlp_dn=p["mlp_dn"],
        mlp_dn_s=p["mlp_dn_s"].t(),
        moe=SimpleNamespace(
            router=p["router"].t(),
            fc1_w=window["fc1_w"],
            fc1_s=window["fc1_s"],
            fc2_w=window["fc2_w"],
            fc2_s=window["fc2_s"],
            **vars(M._moe_scalars(p["isc1"], p["isc2"], window["ws2_13"], window["ws2_2"])),
        ),
    )


def _target(M, kind: str, d, rank: int, world: int, pg, side, call: int) -> object:
    """The target class `kind` runs over the surface `_moe` reads off the core: the four ranks
    on the process group's communicator, or the eight on the MPI-session communicator (`pg`
    None, as the core holds it under MPI), in expert calls of `call` rows."""
    target = object.__new__(M.PrefillTarget if kind == "prefill" else M.DecodeTarget)
    # The core's values at max_num_tokens 32768, both `call`: one call's tokens and the
    # autotuner's largest bucket -- all 131072 gathered at four ranks, 32768 of up to 262144
    # gathered at eight; with chunked prefill 32768 at both.
    target.core = SimpleNamespace(
        topk=4,
        num_experts=d.num_experts,
        moe_inter=d.moe_inter,
        expert_offset=d.expert_offset,
        local_experts=d.local_experts,
        expert_call_tokens=call,
        expert_tune_tokens=call,
        dp_group=list(range(world)),
        dp_rank=rank,
        dp_pg=pg,
        _side_stream=side,
    )
    return target


def _tokens(total: int, seed: int, scale: float) -> torch.Tensor:
    """Every rank's tokens concatenated in rank order, the same on every rank."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(total, 6144, generator=g, device="cuda") * scale).to(torch.bfloat16)


def _reference(M, x, L, windows: dict, offsets: dict, shared_of, call: int) -> tuple:
    """The layer on one rank over `x`, the concatenated tokens: the shared expert plus every
    window's partials, summed in fp32, in expert calls of `call` rows, the target's size. Also
    each token's router near-tie flag."""
    acc = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    tie = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for lo in range(0, x.shape[0], call):
        hi = min(lo + call, x.shape[0])
        chunk = x[lo:hi]
        logits = M.cublas_mm(chunk, L.moe.router, None, torch.float32)
        top = logits.topk(5, dim=1).values
        tie[lo:hi] = (top[:, 3] - top[:, 4]) < TIE * top.abs().amax(dim=1)
        ids, weights = M.renorm_moe_routing_op(logits, 4, torch.bfloat16)
        xq, xs = M.fp4_quantize(chunk, L.moe.gscale, 16, False, False)
        for w, m in windows.items():
            part = M.fp4_block_scale_moe_runner(
                None, None, xq, xs.view(torch.float8_e4m3fn), m.fc1_w, m.fc1_s, None, None,
                None, None, m.fc2_w, m.fc2_s, None, m.out1, m.out1_gate, m.out2, 128, 4, None,
                None, 6144, offsets[w], m.fc1_w.shape[0], None, 1, True, 0,
                topk_weights=weights, topk_ids=ids, tune_max_num_tokens=call,
            )[0]  # fmt: skip
            acc[lo:hi] += part.float()
    shared = shared_of(x)
    rms = lambda t: t.float().pow(2).mean().sqrt().item()  # noqa: E731
    print(
        f"[reference] {x.shape[0]} tokens: routed rms {rms(acc):.4g}, shared rms "
        f"{rms(shared):.4g}, input rms {rms(x):.4g}",
        flush=True,
    )
    return (acc.to(torch.bfloat16).float() + shared.float()).to(torch.bfloat16), tie


def _deviation(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(worst element deviation against its row's scale, relative RMS deviation), in ulp."""
    o, r = out.float(), ref.float()
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    elt = ((o - r).abs() / row).max().item() / ULP
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-30)).item() / ULP
    return elt, rms


# ------------------------------------------------------- four ranks: spawned, torch.distributed


def _run_spawned_rank() -> None:
    import torch.distributed as dist

    world = SPAWNED_WORLD
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="env://", device_id=torch.device("cuda", rank))
    pg = dist.distributed_c10d._get_default_group().boxed()

    M, W = _modules()
    # The expert call table these four ranks run (`_spawn_ranks`), by its engine.
    engine = os.environ["ML4_ENGINE"]
    call = getattr(M, EXPERT_CALLS[engine])[world]
    print(f"[rank {rank}] {engine}: expert calls of {call} rows", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    ck = Path(os.environ["ML4_CHECKPOINT"])
    config = _config(ck)
    weight_map = json.loads((ck / SHARD_INDEX).read_text())["weight_map"]

    # Every rank: the replicated part and its own window. Rank 0, for the reference: all four.
    p, d = _load(W, ck, config, weight_map, rank, world, PARAMS)
    windows, offsets = {rank: p}, {rank: d.expert_offset}
    if rank == 0:
        for w in range(1, world):
            windows[w], dw = _load(W, ck, config, weight_map, w, world, WINDOW_PARAMS)
            offsets[w] = dw.expert_offset
    assert offsets == {w: w * d.local_experts for w in windows}, offsets
    L = _layer(M, p, p)
    ref_windows = {w: _layer(M, p, windows[w]).moe for w in windows}
    side = torch.cuda.Stream()
    shared_of = _target(M, "prefill", d, rank, world, pg, side, call)._dense
    # Activations at the amax the layer's static input scale was calibrated for.
    scale = E4M3_MAX * E2M1_MAX * float(p["isc1"]) / 5
    failures = []

    def check(name: str, counts, out: torch.Tensor, x: torch.Tensor, gate: bool = True):
        """Rank 0 computes the layer over the concatenated tokens and hands every rank the
        reference; each rank compares its own rows. Returns (elt, rms, compared rows)."""
        if rank == 0:
            ref, tie = _reference(M, x, L, ref_windows, offsets, lambda t: shared_of(t, L), call)
        else:
            ref = torch.empty(x.shape, dtype=torch.bfloat16, device="cuda")
            tie = torch.empty(x.shape[0], dtype=torch.bool, device="cuda")
        dist.broadcast(ref, 0)
        dist.broadcast(tie, 0)
        lo = sum(counts[:rank])
        mine = slice(lo, lo + counts[rank])
        keep = ~tie[mine]
        elt, rms = _deviation(out[keep], ref[mine][keep])
        n = int(keep.sum())
        print(
            f"[rank {rank}] {name}: rows {counts} -> {n}/{counts[rank]} compared, "
            f"{counts[rank] - n} near-tie; {elt:.2f} elt / {rms:.3f} rms ulp",
            flush=True,
        )
        if gate and not (n > 0 and elt <= GATE_ELT and rms <= GATE_RMS):
            failures.append(f"{name}: {elt:.2f} elt / {rms:.3f} rms ulp over {n} rows")
        return elt, rms, n

    def round_trip(counts, kinds, x):
        """This rank's own rows through its target's `_moe`; also the expert calls it made."""
        lo = sum(counts[:rank])
        own = x[lo : lo + counts[rank]].contiguous()
        target = _target(M, kinds[rank], d, rank, world, pg, side, call)
        padded = M._padded_rows(target.core, {"all_rank_num_tokens": list(counts)}, counts[rank])
        runner = M.fp4_block_scale_moe_runner
        calls = []

        def counted(*args, **kwargs):
            calls.append(args[2].shape[0])
            return runner(*args, **kwargs)

        M.fp4_block_scale_moe_runner = counted
        try:
            out = target._moe(own, L, padded, M._padding_rows(list(counts), padded, own.device))
        finally:
            M.fp4_block_scale_moe_runner = runner
        torch.cuda.synchronize()
        return out, calls

    for seed, (name, counts, kinds) in enumerate(CASES[world]):
        x = _tokens(sum(counts), seed, scale)
        out, calls = round_trip(counts, kinds, x)
        gathered = world * max(counts)
        expected = [min(call, gathered - lo) for lo in range(0, gathered, call)]
        print(f"[rank {rank}] {name}: {gathered} gathered rows in expert calls {calls}", flush=True)
        if calls != expected:
            failures.append(f"{name}: expert calls {calls}, expected {expected}")
        check(name, counts, out, x)
        del x, out

    # Two defects the comparison must see: the padding ahead of the rows, and each rank handed
    # its neighbour's rows by the reduce-scatter.
    name, counts, kinds = CASES[world][0]
    x = _tokens(sum(counts), 0, scale)
    real_pad = torch.nn.functional.pad

    def front_pad(t, pad, *args, **kwargs):
        if len(pad) == 4:
            pad = [pad[0], pad[1], pad[3], pad[2]]
        return real_pad(t, pad, *args, **kwargs)

    torch.nn.functional.pad = front_pad
    try:
        out, _ = round_trip(counts, kinds, x)
    finally:
        torch.nn.functional.pad = real_pad
    elt, rms, _ = check("defect: padding ahead of the rows", counts, out, x, gate=False)
    if counts[rank] < max(counts) and rms <= 5 * GATE_RMS:
        failures.append(f"a front-padded rank passes: {rms:.3f} rms ulp")
    real_rs = M.reducescatter

    def neighbours_rows(t, sizes, group, process_group=None):
        return real_rs(torch.roll(t, -(t.shape[0] // world), 0), sizes, group, process_group)

    M.reducescatter = neighbours_rows
    try:
        out, _ = round_trip(counts, kinds, x)
    finally:
        M.reducescatter = real_rs
    elt, rms, _ = check("defect: the neighbour's rows", counts, out, x, gate=False)
    if rms <= 5 * GATE_RMS:
        failures.append(f"the neighbour's rows pass: {rms:.3f} rms ulp")

    # CUDA graphs: every rank decoding the same batch, captured once, replayed on new tokens.
    for b in GRAPH_BATCHES:
        counts = (b,) * world
        target = _target(M, "decode", d, rank, world, pg, side, call)
        static = torch.zeros(b, 6144, dtype=torch.bfloat16, device="cuda")
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(2):
                target._moe(static, L, b)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = target._moe(static, L, b)
        for replay in range(2):
            x = _tokens(world * b, 100 + 10 * b + replay, scale)
            static.copy_(x[rank * b : (rank + 1) * b])
            graph.replay()
            torch.cuda.synchronize()
            check(f"graph {b} rows, replay {replay}", counts, out.clone(), x)

    dist.barrier()
    if failures:
        raise AssertionError(f"rank {rank}: " + "; ".join(failures))
    print(f"[rank {rank}] all checks passed", flush=True)


def _worker() -> None:
    try:
        _run_spawned_rank()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # A rank that leaves a collective early wedges its siblings; the launcher kills them on
        # this exit.
        os._exit(13)
    sys.stdout.flush()
    # No process-group teardown: a communicator that captured collectives can wedge NCCL's
    # watchdog at destruction, and the launcher owns these processes.
    os._exit(0)


def _spawn_ranks(checkpoint: Path, engine: str) -> None:
    """One fresh process per device, with its own session, each running `engine`'s expert
    calls; on the first nonzero exit or on the deadline every session is killed, so a wedged
    collective cannot outlive the run."""
    world = SPAWNED_WORLD
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [v for v in visible.split(",") if v.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    repo_root = Path(__file__).resolve().parents[7]
    assert (repo_root / "tensorrt_llm").is_dir(), repo_root
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    # The launching step's PMIx/PMI/OMPI/SLURM namespace stays out of the ranks: they
    # rendezvous through torch.distributed, and a hard exit registered with that step would
    # take it down whole.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))
    }
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(devices[:world]),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world),
        TLLM_DISABLE_MPI="1",
        ML4_CHECKPOINT=str(checkpoint),
        ML4_ENGINE=engine,
        PYTHONPATH=os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p),
    )
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r)),
            start_new_session=True,
        )
        for r in range(world)
    ]

    def kill_all() -> None:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for child in children:
            child.wait()

    deadline = time.monotonic() + DEADLINE_S
    live = set(range(world))
    failure = None
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
        kill_all()
        raise AssertionError(f"the {world}-rank MoE round trip ({engine}) failed: {failure}")


# ------------------------------------------------------- eight ranks: launched, MPI


def _run_mpi_rank(ck: Path) -> list[str]:
    """One rank's body. Returns the rank's gate failures; raises on anything else."""
    from tensorrt_llm._utils import mpi_comm, mpi_rank

    world = MPI_WORLD
    M, W = _modules()
    # The expert call at eight ranks, with or without chunked prefill.
    call = M._EXPERT_CALL_TOKENS[world]
    assert call == M._EXPERT_CALL_TOKENS_CHUNKED[world] == 32768, call
    rank = mpi_rank()
    comm = mpi_comm()
    hosts = comm.allgather(socket.gethostname())
    print(
        f"[rank {rank}] host {hosts[rank]}, device {torch.cuda.current_device()} of "
        f"{torch.cuda.device_count()} visible; {len(set(hosts))} node(s)",
        flush=True,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    config = _config(ck)
    weight_map = json.loads((ck / SHARD_INDEX).read_text())["weight_map"]

    # Every rank: the replicated part and its own window, then the seven other windows for the
    # reference it computes itself.
    p, d = _load(W, ck, config, weight_map, rank, world, PARAMS)
    windows, offsets = {rank: p}, {rank: d.expert_offset}
    for w in range(world):
        if w != rank:
            windows[w], dw = _load(W, ck, config, weight_map, w, world, WINDOW_PARAMS)
            offsets[w] = dw.expert_offset
    assert offsets == {w: 16 * w for w in range(world)}, offsets
    assert d.local_experts == 16, d.local_experts
    L = _layer(M, p, p)
    ref_windows = {w: _layer(M, p, windows[w]).moe for w in sorted(windows)}
    side = torch.cuda.Stream()
    shared_of = _target(M, "prefill", d, rank, world, None, side, call)._dense
    # Activations at the amax the layer's static input scale was calibrated for.
    scale = E4M3_MAX * E2M1_MAX * float(p["isc1"]) / 5
    failures = []

    def check(name: str, counts, out: torch.Tensor, x: torch.Tensor, gate: bool = True):
        """Every rank computes the layer over the concatenated tokens and compares its own
        rows. Returns (elt, rms, compared rows)."""
        ref, tie = _reference(M, x, L, ref_windows, offsets, lambda t: shared_of(t, L), call)
        lo = sum(counts[:rank])
        mine = slice(lo, lo + counts[rank])
        keep = ~tie[mine]
        elt, rms = _deviation(out[keep], ref[mine][keep])
        n = int(keep.sum())
        print(
            f"[rank {rank}] {name}: rows {counts} -> {n}/{counts[rank]} compared, "
            f"{counts[rank] - n} near-tie; {elt:.2f} elt / {rms:.3f} rms ulp",
            flush=True,
        )
        if gate and not (n > 0 and elt <= GATE_ELT and rms <= GATE_RMS):
            failures.append(f"{name}: {elt:.2f} elt / {rms:.3f} rms ulp over {n} rows")
        return elt, rms, n

    def round_trip(counts, kinds, x):
        """This rank's own rows through its target's `_moe`; also the expert calls it made."""
        lo = sum(counts[:rank])
        own = x[lo : lo + counts[rank]].contiguous()
        target = _target(M, kinds[rank], d, rank, world, None, side, call)
        padded = M._padded_rows(target.core, {"all_rank_num_tokens": list(counts)}, counts[rank])
        runner = M.fp4_block_scale_moe_runner
        calls = []

        def counted(*args, **kwargs):
            calls.append(args[2].shape[0])
            return runner(*args, **kwargs)

        M.fp4_block_scale_moe_runner = counted
        try:
            out = target._moe(own, L, padded, M._padding_rows(list(counts), padded, own.device))
        finally:
            M.fp4_block_scale_moe_runner = runner
        torch.cuda.synchronize()
        return out, calls

    for seed, (name, counts, kinds) in enumerate(CASES[world]):
        x = _tokens(sum(counts), seed, scale)
        out, calls = round_trip(counts, kinds, x)
        gathered = world * max(counts)
        expected = [min(call, gathered - lo) for lo in range(0, gathered, call)]
        print(f"[rank {rank}] {name}: {gathered} gathered rows in expert calls {calls}", flush=True)
        if calls != expected:
            failures.append(f"{name}: expert calls {calls}, expected {expected}")
        check(name, counts, out, x)
        del x, out

    # Two defects the comparison must see: the padding ahead of the rows, and each rank handed
    # its neighbour's rows by the reduce-scatter.
    name, counts, kinds = CASES[world][0]
    x = _tokens(sum(counts), 0, scale)
    real_pad = torch.nn.functional.pad

    def front_pad(t, pad, *args, **kwargs):
        if len(pad) == 4:
            pad = [pad[0], pad[1], pad[3], pad[2]]
        return real_pad(t, pad, *args, **kwargs)

    torch.nn.functional.pad = front_pad
    try:
        out, _ = round_trip(counts, kinds, x)
    finally:
        torch.nn.functional.pad = real_pad
    elt, rms, _ = check("defect: padding ahead of the rows", counts, out, x, gate=False)
    if counts[rank] < max(counts) and rms <= 5 * GATE_RMS:
        failures.append(f"a front-padded rank passes: {rms:.3f} rms ulp")
    real_rs = M.reducescatter

    def neighbours_rows(t, sizes, group, process_group=None):
        return real_rs(torch.roll(t, -(t.shape[0] // world), 0), sizes, group, process_group)

    M.reducescatter = neighbours_rows
    try:
        out, _ = round_trip(counts, kinds, x)
    finally:
        M.reducescatter = real_rs
    elt, rms, _ = check("defect: the neighbour's rows", counts, out, x, gate=False)
    if rms <= 5 * GATE_RMS:
        failures.append(f"the neighbour's rows pass: {rms:.3f} rms ulp")

    # CUDA graphs: every rank decoding the same batch, captured once, replayed on new tokens.
    for b in GRAPH_BATCHES:
        counts = (b,) * world
        target = _target(M, "decode", d, rank, world, None, side, call)
        static = torch.zeros(b, 6144, dtype=torch.bfloat16, device="cuda")
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(2):
                target._moe(static, L, b)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = target._moe(static, L, b)
        for replay in range(2):
            x = _tokens(world * b, 100 + 10 * b + replay, scale)
            static.copy_(x[rank * b : (rank + 1) * b])
            graph.replay()
            torch.cuda.synchronize()
            check(f"graph {b} rows, replay {replay}", counts, out.clone(), x)
        del graph

    print(
        f"[rank {rank}] peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
        flush=True,
    )
    return failures


def _launched_world() -> int:
    """The MPI world this process is a rank of, when a launcher started it as one; 1 otherwise
    (including rank 0 of trtllm-llmapi-launch, whose other ranks serve the LLM API, not this
    body)."""
    from tensorrt_llm._utils import mpi_world_size
    from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size

    world = get_mpi_world_size()
    return world if world == mpi_world_size() else 1


# ---------------------------------------------------------------------------- the tests


@pytest.mark.no_xdist
@pytest.mark.parametrize("engine", list(EXPERT_CALLS))
def test_moe_round_trip_on_four_ranks(engine):
    checkpoint = _checkpoint_dir()
    if checkpoint is None:
        pytest.skip(f"checkpoint {CHECKPOINTS} is not under LLM_MODELS_ROOT")
    if torch.cuda.device_count() < SPAWNED_WORLD:
        pytest.skip(f"needs {SPAWNED_WORLD} GPUs")
    launched = _launched_world()
    if launched not in (1, SPAWNED_WORLD):
        pytest.skip(f"the {SPAWNED_WORLD}-rank round trip does not start in {launched} MPI ranks")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("the mistral_large4 modeling_v2 targets are certified on sm_103 only")
    _spawn_ranks(checkpoint, engine)


@pytest.mark.no_xdist
@pytest.mark.threadleak(enabled=False)
def test_moe_round_trip_on_eight_ranks():
    from tensorrt_llm._utils import (
        get_sm_version,
        local_mpi_rank,
        mpi_comm,
        mpi_rank,
        mpi_world_size,
    )
    from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size

    checkpoint = _checkpoint_dir()
    if checkpoint is None:
        pytest.skip(f"checkpoint {CHECKPOINTS} is not under LLM_MODELS_ROOT")
    if _launched_world() != MPI_WORLD:
        pytest.skip(
            f"needs {MPI_WORLD} launched MPI ranks (this world: get_mpi_world_size() "
            f"{get_mpi_world_size()}, MPI world {mpi_world_size()})"
        )
    if get_sm_version() != 103:
        pytest.skip("the mistral_large4 modeling_v2 targets are certified on sm_103 only")
    torch.cuda.set_device(local_mpi_rank())
    faulthandler.dump_traceback_later(DEADLINE_S, exit=True)
    rank = mpi_rank()
    try:
        failures = _run_mpi_rank(checkpoint)
    except BaseException:
        import traceback

        print(f"[rank {rank}] raised", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # A rank that leaves the collective sequence wedges its siblings; exiting without
        # finalizing makes the launcher end them.
        os._exit(13)
    # Every rank sees every rank's failures, so all eight pass or fail together.
    every = mpi_comm().allgather(failures)
    faulthandler.cancel_dump_traceback_later()
    bad = [f"rank {r}: {f}" for r, fs in enumerate(every) for f in fs]
    assert not bad, "; ".join(bad)
    print(f"[rank {rank}] all checks passed on every rank", flush=True)


if __name__ == "__main__" and _WORKER_FLAG in sys.argv:
    _worker()
