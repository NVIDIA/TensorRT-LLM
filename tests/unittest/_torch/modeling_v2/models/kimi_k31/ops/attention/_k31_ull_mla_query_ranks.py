# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rank job of the dsa_mla_query test, started by file path in a fresh interpreter:
``python _k31_ull_mla_query_ranks.py <results.json>``.

Two processes on two GPUs (the q mailbox's multicast object needs at least two devices), one NCCL
group. For each cell, process 0 is the cell's Helix rank and process 1 rank ``(rank + 1) % 16``;
each builds the real ``dsa_front.FlatMlaQuery`` (its mailbox in symmetric memory, the multicast
address of both devices) and publishes. Each checks its own head block and local rows against the
FP32 query within the E4M3 band, the peer's head block in its mailbox against the rows the peer
stored locally (the multicast store reached the other GPU), every other head block and the other
parity untouched. Process 0 writes ``{cell.why: "ok" | error}`` and the is_valid refusals to the
results file."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from types import SimpleNamespace

WORLD = 2
_TARGET = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
LAYERS = (3, 7)  # an even count of DSA layers (the parity alternates)
IS_VALID = "is_valid"


def _owners(torch, layers, seed):
    """The DSA layers' weights FlatMlaQuery reads (the same on both processes)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    owners = {}
    for layer in LAYERS:
        weights = SimpleNamespace(
            q_norm=layers.NormWeight(
                (1 + 0.1 * torch.randn(1536, generator=g, device="cuda")).bfloat16(), 1e-6
            ),
            kv_norm=layers.NormWeight(torch.ones(512, device="cuda").bfloat16(), 1e-6),
            query=layers.Bf16Weight(
                (torch.randn(22528, 1536, generator=g, device="cuda") * 0.03).bfloat16()
            ),
            k_b=(torch.randn(96, 128, 512, generator=g, device="cuda") * 0.09).bfloat16(),
        )
        owners[layer] = SimpleNamespace(hidden_pitch=3040, weights=weights)
    return owners


def _q_abs(torch, query, layer, hidden, rows, local):
    """|q| [rows, 6, 192] of the reference's BF16 Q_B rows (the absorption's operand magnitudes)."""
    q_norm, _, q_b, _ = query.weights[layer]
    x = hidden[:rows, :1536].float()
    normed = x * torch.rsqrt(
        x.pow(2).mean(-1, keepdim=True) + float(getattr(query.eps, "value", query.eps))
    )
    normed = (normed * q_norm.float()).bfloat16().float()
    return (normed @ q_b.float().t()).abs().view(rows, local, 192)


def _check_cell(torch, dist, op, local_heads, validating, query, cell, peer_rank):
    import zlib

    rows = cell["rows"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell["why"].encode()) % 2**31)
    hidden = torch.randn(rows, 3040, generator=g, device="cuda").bfloat16()
    if dist.get_rank():  # the peer's own rows
        hidden = torch.randn(rows, 3040, generator=g, device="cuda").bfloat16()
    layer = LAYERS[rows % 2]
    parity = query.parity[layer]
    query.mailbox.fill_(0xFF)
    torch.cuda.synchronize()
    dist.barrier()
    with validating(op):
        op(query, rows, parity, layer, hidden)
    torch.cuda.synchronize()
    dist.barrier()  # both publishes landed
    ref = op.reference(query, rows, parity, layer, hidden)
    heads = slice(local_heads * query.rank, local_heads * (query.rank + 1))
    box = query.mailbox[parity, :rows, heads]
    assert torch.equal(box, query.local[:rows]), "the published and local rows differ"
    got = box.view(torch.float8_e4m3fn).float()
    # E4M3 at scale 1: one code step is 2^-3 relative (half of it from the rounding, the other
    # half for an operand the kernel rounded to BF16 on the other side of a midpoint), the
    # subnormal step 2^-9 absolute; the BF16 rounding of Q_B moves the latent by at most 2^-8 of
    # the magnitudes the absorption sums.
    k_b = query.weights[layer][3]
    q_abs = _q_abs(torch, query, layer, hidden, rows, local_heads)
    mag = torch.cat(
        (torch.einsum("rjn,jnl->rjl", q_abs[..., :128], k_b.float().abs()), q_abs[..., 128:]),
        dim=-1,
    )
    band = 2.0**-3 * ref.abs() + 2.0**-8 * mag + 2.0**-9
    err = (got - ref).abs()
    assert bool((err <= band).all()), f"max err {float(err.max()):.3e}"
    # The peer's multicast store landed in this GPU's mailbox, bitwise its local rows.
    mine = query.local[:rows].contiguous()
    both = [torch.empty_like(mine) for _ in range(WORLD)]
    dist.all_gather(both, mine)
    peer = slice(local_heads * peer_rank, local_heads * (peer_rank + 1))
    assert torch.equal(query.mailbox[parity, :rows, peer], both[1 - dist.get_rank()]), (
        "the peer's head block is not its published rows"
    )
    others = torch.ones(96, dtype=torch.bool, device="cuda")
    others[heads] = False
    others[peer] = False
    assert bool((query.mailbox[parity, :rows][:, others] == 0xFF).all()), "wrote another head block"
    assert bool((query.mailbox[1 - parity] == 0xFF).all()), "wrote the other parity"


def _is_valid_refusals(torch, op, query):
    hidden = torch.zeros(8, 3040, device="cuda").bfloat16()
    refused = []
    for args, match in (
        ((query, 16, 0, LAYERS[0], hidden), "rows"),
        ((query, 8, 1 - query.parity[LAYERS[0]], LAYERS[0], hidden), "parity"),
    ):
        try:
            op.is_valid(*args)
        except ValueError as error:
            if match in str(error):
                refused.append(match)
    return refused


def _rank(index: int, store_path: str, results_path: str, cells: list, test_dir: str) -> None:
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    import dataclasses
    import importlib

    import torch
    import torch.distributed as dist

    # The served profile, before any other module of the package (as the tests' conftest).
    features = importlib.import_module(
        _TARGET.rpartition(".")[0] + ".ops.common.rubin_ull_features"
    )
    if features.selected() is None:
        features.select(features.ULL16_GENERATION)
    # The query kernel alone: no DSA core over its mailbox (a test-settable field, not env).
    features._ACTIVE = dataclasses.replace(features.active(), dsa_core="fmha")
    torch.cuda.set_device(index)
    device = torch.device("cuda", index)
    dist.init_process_group(
        "nccl", store=dist.FileStore(store_path, WORLD), rank=index, world_size=WORLD,
        device_id=device,
    )  # fmt: skip
    sys.path[:0] = [test_dir, os.path.join(test_dir, "../../../..")]
    from _validating import validating

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_mla_query import (  # noqa: E501
        LOCAL,
        dsa_mla_query,
    )

    front = importlib.import_module(_TARGET + ".modeling")
    layers = importlib.import_module(_TARGET + ".modeling")
    owners = _owners(torch, layers, seed=22528)
    results = {}
    first = None
    for cell in cells:
        rank = (cell["rank"] + index) % 16
        peer = (cell["rank"] + 1 - index) % 16
        try:
            query = front.FlatMlaQuery(dist.group.WORLD, rank, owners, [cell["rows"]], device)
            if first is None:
                first = query
            _check_cell(torch, dist, dsa_mla_query, LOCAL, validating, query, cell, peer)
            outcome = "ok"
        except Exception:  # noqa: BLE001 -- reported per cell to the test
            outcome = traceback.format_exc(limit=6)
        gathered = [None] * WORLD
        dist.all_gather_object(gathered, outcome)
        bad = [f"process {i}: {o}" for i, o in enumerate(gathered) if o != "ok"]
        results[cell["why"]] = "ok" if not bad else "\n".join(bad)
    if first is not None:
        results[IS_VALID] = _is_valid_refusals(torch, dsa_mla_query, first)
    torch.cuda.synchronize()
    dist.barrier()
    if index == 0:
        with open(results_path, "w") as f:
            json.dump(results, f)
    dist.destroy_process_group()


def main(results_path: str) -> None:
    import torch.multiprocessing as mp

    with open(results_path) as f:
        cells = json.load(f)["cells"]
    test_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(results_path))) as tmp:
        store = os.path.join(tmp, "store")
        mp.spawn(_rank, args=(store, results_path, cells, test_dir), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main(sys.argv[1])
