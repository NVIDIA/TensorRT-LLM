# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the Nemotron-3 Ultra BF16 tep8 target's MNNVL fused all-reduce entry.

Every rank is one MPI process, which is how the engine runs MNNVL: the workspace is built from an MPI
communicator and the ranks exchange fabric handles through it. Run it directly:

    mpirun -n 4 python _mnnvl_fusion_allreduce_op_matrix.py     # one node: the ranks map to local GPUs
    srun --mpi=pmix -N2 --ntasks-per-node=4 python _mnnvl_fusion_allreduce_op_matrix.py   # two NVL72 nodes

On one node it sets the engine's single-node test switches (`TLLM_TEST_MNNVL`, `TRTLLM_FORCE_MNNVL_AR`);
on more devices than a node holds it runs the production configuration, a group spanning nodes.

The collected entry point is `test_modeling_v2_mnnvl_fusion_allreduce_op_matrix.py` beside this file. Not a
pytest module, despite the `check_*` bodies: each is one step of a single job in which every rank reads
module-global state that only `run_rank` binds, and a collective that breaks hangs rather than raising.
"""

import os
import sys
import zlib
from pathlib import Path
from typing import Any, Tuple

import torch

HIDDEN = 8192
EPS = 1e-5
#: Layers whose all-reduces one decode graph holds: 48 mamba + 48 MoE + 12 attention blocks.
GRAPH_SITES = 108

# Bound inside `run_rank`, never at import: the entry pulls in tensorrt_llm, which initializes MPI.
entry: Any = None
validating: Any = None
mapping: Any = None
dist: Any = None
RANK = 0
WORLD = 1
MULTI_NODE = False


def _payload(rank: int, rows: int, seed: int, bound: int = 15) -> torch.Tensor:
    """One rank's contribution, exact: multiples of 1/8 with |x| <= bound/8.

    bf16 holds k/8 exactly for |k| <= 255. With `bound` 15, a sum of up to 8 ranks and a residual
    of at most 31/8 reaches |k| <= 151, so every partial sum is exact in every summation order --
    which is what lets the sums and the residual output be compared bitwise whatever order the
    kernel adds in. Every rank can regenerate every other rank's payload (cuRAND is deterministic
    for a given seed and shape), so the reference is arithmetic rather than a second collective.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    raw = torch.randint(-bound, bound + 1, (rows, HIDDEN), generator=gen, device="cuda")
    return (raw.float() / 8.0).to(torch.bfloat16)


def _residual(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed * 331 + 7)
    raw = torch.randint(-31, 32, (rows, HIDDEN), generator=gen, device="cuda")
    return (raw.float() / 8.0).to(torch.bfloat16)


def _rand(rank: int, rows: int, seed: int) -> torch.Tensor:
    """A contribution with no exactness property: standard normal in bf16."""
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    return torch.randn((rows, HIDDEN), generator=gen, device="cuda").to(torch.bfloat16)


def _gamma(seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed * 53 + 3)
    return (1.0 + 0.25 * torch.randn(HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)


def _buffers(rows: int):
    """The workspace for a call over `rows` rows, grown first if it is too small, as the target does."""
    from tensorrt_llm._torch.distributed.ops import (
        MNNVLAllReduce,
        get_or_scale_allreduce_mnnvl_workspace,
    )

    need = MNNVLAllReduce.get_required_workspace_size(rows, HIDDEN, WORLD, torch.bfloat16)
    workspace = get_or_scale_allreduce_mnnvl_workspace(
        mapping, torch.bfloat16, buffer_size_bytes=need
    )
    return workspace["uc_buffer"].view(torch.bfloat16).view(3, -1), workspace["buffer_flags"]


def _call(x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor):
    """One certified call, exactly as the target makes it, through the entry's guard."""
    comm_buffer, buffer_flags = _buffers(x.shape[0])
    with validating(entry.mnnvl_fusion_allreduce):
        norm_out, residual_out = entry.mnnvl_fusion_allreduce(
            x,
            gamma,
            residual,
            EPS,
            comm_buffer,
            buffer_flags,
            True,
            None,
            entry.FUSION_RESIDUAL_RMS_NORM,
        )
    return norm_out, residual_out


def _reference(rows: int, seed: int, exact: bool, gamma: torch.Tensor):
    make = _payload if exact else _rand
    parts = [make(r, rows, seed) for r in range(WORLD)]
    residual = _residual(rows, seed)
    return parts, residual, entry.mnnvl_fusion_allreduce.reference(parts, residual, gamma, EPS)


def _digest(*tensors: torch.Tensor) -> int:
    crc = 0
    for t in tensors:
        crc = zlib.crc32(t.contiguous().view(torch.int16).cpu().numpy().tobytes(), crc)
    return crc


def _assert_same_on_every_rank(where: str, *tensors: torch.Tensor) -> None:
    """Bit-identical on every rank: TP-replicated activations stay replicated."""
    from mpi4py import MPI

    digests = MPI.COMM_WORLD.allgather(_digest(*tensors))
    assert len(set(digests)) == 1, f"{where}: ranks disagree: {digests}"


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    assert out.dtype == ref.dtype and out.shape == ref.shape, (out.dtype, ref.dtype, out.shape)
    if not torch.equal(out, ref):
        bad = (out != ref).sum().item()
        raise AssertionError(f"{where}: {bad} of {out.numel()} elements differ from the reference")


def _kernel_ran(rows: int) -> str:
    """Which kernel the last call over `rows` rows ran, read off the workspace's protocol state rather
    than from the entry's own copy of the size threshold: every call ends by recording how many Lamport
    stages it used in `buffer_flags[3]` -- one for the one-shot kernel, two (scatter, broadcast) for the
    two-shot ones. Synchronizes, so never inside a capture."""
    torch.cuda.synchronize()
    _, buffer_flags = _buffers(rows)
    stages = buffer_flags.cpu().view(torch.int32).tolist()[3]
    assert stages in (1, 2), f"buffer_flags[3] = {stages}: not a stage count"
    return "oneshot" if stages == 1 else "twoshot"


def _assert_kernel(rows: int, where: str) -> str:
    """The kernel that ran must be the one the entry says this size selects."""
    ran = _kernel_ran(rows)
    expected = "oneshot" if entry.one_shot(rows, HIDDEN, WORLD) else "twoshot"
    assert ran == expected, f"{where}: ran the {ran} kernel, the entry's threshold says {expected}"
    return ran


#: Band for `norm_out` against `reference`, in units of 2^-8 of each row's largest magnitude
#: element-wise and of the RMS relative to the reference's RMS in aggregate (see `_band`).
#:
#: Why the two kernels and `reference` are not bit-identical: all three round at the same places (the
#: group sum to bf16, the residual sum to bf16, the squares to bf16), but the last step differs in
#: its fp32 association -- the one-shot kernel computes `(x * rcp) * gamma`, the two-shot one
#: `(gamma * x) * rcp`, `reference` `gamma * x * rcp` -- so a value within a few fp32 ulps of a bf16
#: rounding boundary can land on the other side. That is one bf16 flip of one element, at most 2
#: units in this metric (an element in the top binade of a row whose maximum sits at the bottom of
#: it); it is the element bound. The flips are rare: the fp32 ulp is 2^-16 of a bf16 one, so about
#: k / 2^16 of the elements for k fp32 ulps of disagreement, which as an RMS is about
#: sqrt(k / 2^16) bf16 ulps (k of 4: 0.008). Measured on GB200 (random inputs up to 2048 rows):
#: worst element 0.93 at 4 ranks and 0.92 at 8, worst RMS 0.007 at 4 ranks and 0.003 at 8, every
#: exact-payload cell and every `residual_out` at 0. The RMS gate leaves a factor of 2.5 over the worst.
NORM_BAND = dict(element_ulp=2.0, rms_ulp=0.02)


def _band(out: torch.Tensor, ref: torch.Tensor, where: str, **band) -> Tuple[float, float]:
    """Gate `out` against `ref` as `assert_within_ulp` does, returning the measured (element, rms)."""
    band = band or NORM_BAND
    ulp = 2.0**-8
    o, r = out.float(), ref.float()
    row_scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = ((o - r).abs() / row_scale).max().item() / ulp
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item() / ulp
    if os.environ.get("MNNVL_MATRIX_VERBOSE") == "1" and RANK == 0:
        print(f"  [band] {where}: {element:.3f} ulp element-wise / {rms:.3f} ulp RMS", flush=True)
    assert element <= band["element_ulp"] and rms <= band["rms_ulp"], (
        f"{where}: {element:.3f} ulp element-wise / {rms:.3f} ulp RMS outside "
        f"{band['element_ulp']} / {band['rms_ulp']}"
    )
    return element, rms


# --- the checks -----------------------------------------------------------------------------------


def check_certified_cells() -> None:
    """Every cell on exact payloads: `residual_out` bitwise, `norm_out` in the band, both identical on
    every rank, and the kernel that ran the one the call's size selects (read back, then named in the
    log)."""
    gamma = _gamma(11)
    for i, cell in enumerate(entry.mnnvl_fusion_allreduce.CELLS):
        rows = cell.spec["rows"]
        seed = 1000 + i
        parts, residual, (ref_norm, ref_res) = _reference(rows, seed, True, gamma)
        norm_out, residual_out = _call(parts[RANK], residual, gamma)
        kernel = _assert_kernel(rows, f"cell {rows} rows")
        _assert_bitwise(residual_out, ref_res, f"cell {rows} rows residual_out")
        element, rms = _band(norm_out, ref_norm, f"cell {rows} rows norm_out")
        _assert_same_on_every_rank(f"cell {rows} rows", norm_out, residual_out)
        if RANK == 0:
            print(
                f"[cell {rows:5d} rows] {kernel}: norm_out {element:.3f} / {rms:.3f} ulp",
                flush=True,
            )


def check_random_inputs() -> None:
    """Inputs with no exactness property: both outputs inside the band to the fp32 reference."""
    gamma = _gamma(23)
    for rows in (1, 9, 32, 300, 2048):
        parts, residual, (ref_norm, ref_res) = _reference(rows, 2000 + rows, False, gamma)
        norm_out, residual_out = _call(parts[RANK], residual, gamma)
        _band(residual_out, ref_res, f"random {rows} rows residual_out")
        _band(norm_out, ref_norm, f"random {rows} rows norm_out")
        _assert_same_on_every_rank(f"random {rows} rows", norm_out, residual_out)


def check_operands_untouched() -> None:
    """What the entry's docstring promises of its operands: `input`, `residual` and `gamma` come back
    unchanged, and both outputs are fresh tensors of `input`'s shape and dtype, aliasing none of them
    nor each other."""
    gamma = _gamma(61)
    for rows in (1, 300):
        parts, residual, _ = _reference(rows, 6000 + rows, True, gamma)
        x, res, g = parts[RANK].clone(), residual.clone(), gamma.clone()
        norm_out, residual_out = _call(x, res, g)
        torch.cuda.synchronize()
        assert torch.equal(x, parts[RANK]), f"{rows} rows: the call wrote to its input"
        assert torch.equal(res, residual), f"{rows} rows: the call wrote to its residual"
        assert torch.equal(g, gamma), f"{rows} rows: the call wrote to its gamma"
        operands = {x.data_ptr(), res.data_ptr(), g.data_ptr()}
        for name, out in (("norm_out", norm_out), ("residual_out", residual_out)):
            assert out.shape == x.shape and out.dtype == x.dtype, (name, out.shape, out.dtype)
            assert out.data_ptr() not in operands, f"{rows} rows: {name} aliases an operand"
        assert norm_out.data_ptr() != residual_out.data_ptr(), f"{rows} rows: outputs alias"


def check_call_sequence() -> None:
    """Many calls on one workspace in a mixed order: the Lamport buffers rotate, the flags advance and
    the one-shot / two-shot switch is crossed in both directions. Every call must be exact and must
    run the kernel its size selects."""
    gamma = _gamma(31)
    order = (1, 9, 32, 8, 17, 2048, 16, 300, 1, 8192, 32, 8, 9, 1) * 6
    for i, rows in enumerate(order):
        seed = 3000 + i
        parts, residual, (_, ref_res) = _reference(rows, seed, True, gamma)
        norm_out, residual_out = _call(parts[RANK], residual, gamma)
        _assert_kernel(rows, f"call {i} ({rows} rows)")
        _assert_bitwise(residual_out, ref_res, f"call {i} ({rows} rows) residual_out")
        _assert_same_on_every_rank(f"call {i} ({rows} rows)", norm_out, residual_out)


def _decode_chain(rows: int, layers: int, seed: int, residual: torch.Tensor, gamma: torch.Tensor):
    """The decode graph's shape: `layers` all-reduces chained through the residual, a fresh partial
    per layer. Returns the last normalized output and the final residual."""
    norm_out = None
    for layer in range(layers):
        part = _payload(RANK, rows, seed * 131 + layer)
        norm_out, residual = _call(part, residual, gamma)
    return norm_out, residual


def _graph_replay(rows: int, seed: int, gamma: torch.Tensor) -> None:
    residual0 = _residual(rows, seed)
    eager_norm, eager_res = _decode_chain(rows, GRAPH_SITES, seed, residual0.clone(), gamma)
    torch.cuda.synchronize()

    # Static inputs for the capture: the chain reads them in place at every replay.
    parts = [_payload(RANK, rows, seed * 131 + layer) for layer in range(GRAPH_SITES)]
    static_residual = residual0.clone()
    _buffers(rows)  # the workspace is sized before the capture, as the target does
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        residual = static_residual
        norm_out = None
        for layer in range(GRAPH_SITES):
            norm_out, residual = _call(parts[layer], residual, gamma)
        final_norm, final_res = norm_out, residual
    for replay in range(3):
        graph.replay()
        torch.cuda.synchronize()
        where = f"{rows}-row graph, replay {replay}"
        _assert_bitwise(final_res, eager_res, f"{where} residual")
        _assert_bitwise(final_norm, eager_norm, f"{where} norm_out")
        _assert_same_on_every_rank(where, final_norm, final_res)
    _assert_kernel(rows, f"{rows}-row graph")


def check_graph_replay() -> None:
    """A CUDA graph of 108 chained decode-sized calls replays bit for bit what the same calls produce
    eagerly, three times over, with the workspace's flags advancing between replays: once on a size
    that takes the one-shot kernel (a decode batch of one row) and once on the two-shot kernels (the
    32-row ceiling of the captured batches)."""
    gamma = _gamma(41)
    for rows, seed in ((1, 4001), (32, 4002)):
        _graph_replay(rows, seed, gamma)


def check_graph_survives_growth() -> None:
    """A graph captured on the small workspace keeps replaying correctly after an eager call grew the
    workspace past it -- the order a served engine goes through when a long prefill follows the
    captured decode batches.

    Runs first (see CHECKS), on the workspace as the target creates it: at the engine's default size,
    which the 8192-row call does not fit. It asserts the growth and that the workspace the graph was
    captured on is kept alive, so a reordering that left the workspace already large would fail here
    instead of passing without growing anything."""
    from tensorrt_llm._torch.distributed.ops import (
        MNNVLAllReduce,
        get_or_scale_allreduce_mnnvl_workspace,
    )

    gamma = _gamma(53)
    rows, big_rows = 32, 8192
    small = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
    big_need = MNNVLAllReduce.get_required_workspace_size(big_rows, HIDDEN, WORLD, torch.bfloat16)
    assert small["buffer_size_bytes"] < big_need, (
        f"the workspace ({small['buffer_size_bytes']} bytes per buffer) already holds the "
        f"{big_rows}-row call ({big_need}): this check has to run before anything grows it"
    )
    residual0 = _residual(rows, 5001)
    parts = [_payload(RANK, rows, 5001 * 131 + layer) for layer in range(4)]
    graph = torch.cuda.CUDAGraph()
    static_residual = residual0.clone()
    with torch.cuda.graph(graph):
        residual = static_residual
        for layer in range(4):
            norm_out, residual = _call(parts[layer], residual, gamma)
        final_norm, final_res = norm_out, residual
    graph.replay()
    torch.cuda.synchronize()
    ref_chain = (final_norm.clone(), final_res.clone())

    # Grow past whatever the workspace held: the two-shot footprint of a full prefill step.
    big_parts, big_residual, (_, big_ref_res) = _reference(big_rows, 5002, True, gamma)
    _, big_res = _call(big_parts[RANK], big_residual, gamma)
    _assert_bitwise(big_res, big_ref_res, f"{big_rows}-row call after capture")
    grown = MNNVLAllReduce.allreduce_mnnvl_workspaces[mapping]
    assert grown is not small and grown["buffer_size_bytes"] >= big_need, (
        "the call did not grow the workspace"
    )
    retired = MNNVLAllReduce.allreduce_mnnvl_retired_workspaces.get(mapping, [])
    assert any(w is small for w in retired), (
        "the workspace the graph was captured on was not kept alive"
    )

    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(final_res, ref_chain[1], "replay after growth residual")
    _assert_bitwise(final_norm, ref_chain[0], "replay after growth norm_out")
    _assert_same_on_every_rank("replay after growth", final_norm, final_res)


CHECKS = (
    # First: it needs the workspace as the target creates it, small, so that the call it makes later
    # has something to grow.
    check_graph_survives_growth,
    check_certified_cells,
    check_random_inputs,
    check_operands_untouched,
    check_call_sequence,
    check_graph_replay,
)


def run_rank() -> bool:
    """One rank's whole job: bind the module state, run every check, fail loudly."""
    global entry, validating, mapping, dist, RANK, WORLD, MULTI_NODE
    from mpi4py import MPI

    rank, world = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    devices = torch.cuda.device_count()
    torch.cuda.set_device(rank % devices)
    MULTI_NODE = world > devices
    if not MULTI_NODE:
        os.environ["TLLM_TEST_MNNVL"] = "1"
        os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (  # noqa: E501
        mnnvl_fusion_allreduce as entry_module,
    )
    from tensorrt_llm.mapping import Mapping

    entry = entry_module
    RANK, WORLD = rank, world
    assert WORLD >= 2, f"a collective needs at least 2 ranks, got {WORLD}"
    mapping = Mapping(world_size=world, tp_size=world, rank=rank, gpus_per_node=devices)

    # A spawned worker, not collected by pytest, so `__extra_import_path__` is not in play.
    # comm -> <target> -> nemotron_h -> models -> modeling_v2, where _validating lives.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _validating import validating as _validating_cm

    validating = _validating_cm

    MPI.COMM_WORLD.Barrier()
    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # Exit hard rather than return: a rank that leaves a collective early wedges every
            # sibling in it, and the launcher (mpirun / srun) kills the rest on a nonzero exit.
            os._exit(13)
        MPI.COMM_WORLD.Barrier()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed ({WORLD} ranks)", flush=True)
    sys.stdout.flush()
    return True


if __name__ == "__main__":
    ok = run_rank()
    # The workspace's multicast mappings are released by the engine's finalizers; skip them so a
    # throwaway job exits as soon as its checks have.
    os._exit(0 if ok else 1)
