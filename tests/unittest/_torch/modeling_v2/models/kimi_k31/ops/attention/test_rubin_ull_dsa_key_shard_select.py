# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's dsa_key_shard_select entry, the group emulated on one GPU (every
rank's workspace on it, every rank's call on its own stream, as the served call: LL exchange, the
scorer's vector maxima, KV pool rows).

- every cell against the reference: each rank keeps exactly its part of the global Top-K;
- the epoch contract of the merge's early read: over consecutive calls (parity flips included) the
  epoch the merge read before its ``griddepcontrol.wait`` -- the one its last CTA publishes into
  ``state[0]`` -- equals the epoch the select grid of the same call tagged its pushes with, and no
  poll timed out; a stale epoch planted on one rank is caught by the same check (kernels built
  without a trap record) and, with the served trap record, ends in a trap whose record names a
  key-shard wait (in a child process: the trap ends the CUDA context).
"""

import json
import os
import subprocess
import sys
import textwrap
import zlib
from pathlib import Path
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_key_shard_select import (  # noqa: E501
    PAGE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_key_shard_select import (  # noqa: E501
    dsa_key_shard_select as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="the key-shard select/merge kernels need an SM10x GPU",
    ),
    # spin_trap.record_address() starts the trap-record watcher, a daemon thread that lives as
    # long as the process (kernels/common/rubin_ull_spin_trap.py).
    pytest.mark.threadleak(enabled=False),
]

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

TOP_K = 2048
#: Poll deadline (us) of the emulated group: long enough for 16 ranks' grids on one GPU.
SPIN_US = 1 << 22
#: The emulated workspaces' width (local keys per row).
WIDTH = 16384


def _dsa_index():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention import (
        rubin_ull_dsa_index as dsa_index,  # noqa: E501
    )

    return dsa_index


class Group:
    """``world`` ranks' key-shard workspaces on this GPU, a Helix step's row lengths and KV rows."""

    def __init__(self, spec, seed, spin=SPIN_US):
        from tensorrt_llm._torch.attention.backends.sparse.dsa.key_shard import (
            KeyShardTopKMerge,
            key_shard_local_k,
        )

        di = _dsa_index()
        self.g = torch.Generator(device="cuda").manual_seed(seed)
        w, requests, next_n = spec["world"], spec["requests"], spec["next_n"]
        contexts = spec.get("contexts") or (spec["context"],) * requests
        self.world, self.rows, self.hot = w, requests * next_n, spec["hot"]
        local_pages = -(-(-(-max(contexts) // PAGE)) // w)
        # The served workspace is wider than any row (62464 keys): wide enough that the select
        # takes the scorer's maxima (rows of at least ``tops_from`` keys) and the split grids.
        width = max(local_pages * PAGE, WIDTH)
        local_k = key_shard_local_k(w, TOP_K)
        need = KeyShardTopKMerge.bytes_needed(self.rows, TOP_K, w, local_k, width)
        bufs = [torch.zeros(need + 256 * 4 * self.rows, dtype=torch.uint8, device="cuda")
                for _ in range(w)]  # fmt: skip
        self.merges = [
            KeyShardTopKMerge(bufs, r, self.rows, TOP_K, width, local_k=local_k, max_spin=spin)
            for r in range(w)
        ]
        self.runners = [di.select_pack_for(m) for m in self.merges]
        self.row_lens = []
        for r in range(w):
            lens = []
            for ctx in contexts:
                pos = torch.arange(ctx)
                mine = ((pos // PAGE) % w) == r
                lens += [int((mine & (pos < ctx - (next_n - 1 - i))).sum()) for i in range(next_n)]
            self.row_lens.append(torch.tensor(lens, dtype=torch.int32, device="cuda"))
        self.kv = [None] * w
        if spec["kv_rows"]:
            req_idx = (requests - 1 - torch.arange(self.rows, device="cuda") // next_n).int()
            for r in range(w):
                pages = torch.randperm(50_000, generator=self.g, device="cuda")
                table = pages[: (requests + 1) * (local_pages + 2)].int()
                table = table.view(requests + 1, local_pages + 2).contiguous()
                self.kv[r] = di.KvRows(table, req_idx, 24 * PAGE, 5 * PAGE)
        self.bounds = spec["bounds"]
        self.streams = [torch.cuda.Stream() for _ in range(w)]

    def fill(self):
        """A step's logits (each rank's shard; the columns past a row's keys hold stale values)
        and the scorer's vector maxima of the rows that select from them."""
        for r, (m, run) in enumerate(zip(self.merges, self.runners)):
            logits = m.local_logits[: self.rows]
            logits.copy_(torch.randn(logits.shape, generator=self.g, device="cuda"))
            if self.hot and r == 0:
                logits.add_(3.0)
            if run.tops is None:
                continue
            for t in range(self.rows):
                n = int(self.row_lens[r][t])
                if n < run.tops_from:
                    continue
                vectors = -(-n // 8)
                img = torch.zeros(vectors * 8, dtype=torch.long, device="cuda")
                img[:n] = _ordered(logits[t, :n])
                tops = img.view(vectors, 8).amax(1)
                run.tops[t, :vectors] = torch.where(tops >= 2**31, tops - 2**32, tops).int()

    def outputs(self):
        rows = self.rows
        return [
            (
                torch.full((rows, TOP_K), -7, dtype=torch.int32, device="cuda"),
                torch.full((rows,), -3, dtype=torch.int32, device="cuda"),
                self.row_lens[r].clone() if self.bounds else None,
                torch.zeros(rows, dtype=torch.bool, device="cuda") if self.bounds else None,
            )
            for r in range(self.world)
        ]

    def call(self, outs):
        """One served call on every rank, each on its own stream; joined before returning (the
        Helix exchange after the attention orders a rank's next call after every peer's merge)."""
        cur = torch.cuda.current_stream()
        for r, run in enumerate(self.runners):
            self.streams[r].wait_stream(cur)
            with torch.cuda.stream(self.streams[r]):
                op(run, self.row_lens[r], self.rows, *outs[r], exchange=True,
                   tops=run.tops is not None, kv_rows=self.kv[r])  # fmt: skip
        for s in self.streams:
            cur.wait_stream(s)

    def group(self):
        return [(m.local_logits, lens) for m, lens in zip(self.merges, self.row_lens)]


def _ordered(x: torch.Tensor) -> torch.Tensor:
    """``ptx.ordered_bits`` of FP32 values, as int64 in [0, 2**32)."""
    bits = x.view(torch.int32).long() & 0xFFFFFFFF
    neg = bits >= 0x80000000
    return torch.where(neg, bits ^ 0xFFFFFFFF, bits ^ 0x80000000)


def _check(grp: Group, outs) -> None:
    for r in range(grp.world):
        out, counts, bounds, mask = outs[r]
        kept, sizes, want_mask = op.reference(
            grp.runners[r], grp.row_lens[r], grp.rows, out, counts, bounds, mask, True,
            grp.runners[r].tops is not None, grp.kv[r], group=grp.group(),
        )  # fmt: skip
        assert torch.equal(counts, sizes), f"rank {r}: counts {counts.tolist()} {sizes.tolist()}"
        for t in range(grp.rows):
            n = int(sizes[t])
            got = out[t, :n].long().sort().values
            assert torch.equal(got, kept[t].sort().values), f"rank {r} row {t}: kept keys differ"
            assert bool((out[t, n:] == -1).all()), f"rank {r} row {t}: tail not -1"
        if mask is not None:
            assert torch.equal(mask, want_mask), f"rank {r}: combine mask"


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    grp = Group(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    grp.fill()
    outs = grp.outputs()
    torch.cuda.synchronize()
    with validating(op):
        grp.call(outs)
    torch.cuda.synchronize()
    assert all(int(m.state[3]) == 0 for m in grp.merges), "an LL poll timed out"
    _check(grp, outs)
    if cell.spec["hot"]:
        # The hot rank holds more of the Top-K than it pushed: its rows were re-selected exactly.
        assert sum(int(m.state[2]) for m in grp.merges) > 0


def _epochs(grp: Group):
    return [int(m.state[0]) for m in grp.merges]


def _contract_violations(grp: Group, before: list[int], timeouts: list[int]) -> list[str]:
    """The merge's pre-wait epoch (published into ``state[0]`` by its last CTA) against the epoch
    the select grids tagged this call's pushes with (every candidate and header pair of the
    current parity block, in every rank's LL region), and the timeout counter."""
    bad = []
    w, rows = grp.world, grp.rows
    for r, m in enumerate(grp.merges):
        merged = int(m.state[0])
        if merged != before[r] + 1:
            bad.append(f"rank {r}: merge epoch {merged}, epoch before the call {before[r]}")
        if int(m.state[3]) != timeouts[r]:
            bad.append(f"rank {r}: {int(m.state[3]) - timeouts[r]} LL poll timeouts")
        ll = m.ll.view(torch.int32).view(2, m.rows_max, m.ll_block, 4)
        block = ll[merged & 1, :rows, : w * (m.local_k + 2)]
        tags = torch.cat((block[..., 1], block[..., 3]))
        wrong = int((tags != merged).sum())
        if wrong:
            seen = sorted(set(tags.unique().tolist()))
            bad.append(f"rank {r}: {wrong} pushed words tagged {seen}, merge epoch {merged}")
    return bad


def test_merge_epoch_matches_the_select_epoch_across_calls() -> None:
    """The local_topk merge reads its epoch before ``griddepcontrol.wait`` (launched early by PDL,
    while its select grid runs). Four consecutive calls (two parity flips), fresh logits each:
    every call's merge epoch is the select's, one past the previous call's, with no timeout and
    exact rows (a stale epoch would have merged the other parity block's keys from two calls back)."""
    spec = dict(world=16, requests=1, next_n=4, context=225_000, kv_rows=True, bounds=True,
                hot=False)  # fmt: skip
    grp = Group(spec, seed=31)
    parities = set()
    for _ in range(4):
        grp.fill()
        outs = grp.outputs()
        torch.cuda.synchronize()
        before, timeouts = _epochs(grp), [int(m.state[3]) for m in grp.merges]
        assert len(set(before)) == 1, f"ranks out of step: {before}"
        grp.call(outs)
        torch.cuda.synchronize()
        assert _contract_violations(grp, before, timeouts) == []
        _check(grp, outs)
        parities.add(_epochs(grp)[0] & 1)
    assert parities == {0, 1}


def _stale_epoch_call():
    """Four ranks after two healthy calls, rank 0's epoch rewound by two (the other parity block,
    whose pairs from two calls back carry the matching tag), then one more call: the peers' waits
    for rank 0's pushes time out (65 ms deadlines). Returns (group, epochs and timeout counts
    before the call); the caller synchronizes."""
    spec = dict(world=4, requests=1, next_n=4, context=30_000, kv_rows=False, bounds=False,
                hot=False)  # fmt: skip
    grp = Group(spec, seed=7, spin=1 << 16)
    for _ in range(2):
        grp.fill()
        grp.call(grp.outputs())
    torch.cuda.synchronize()
    grp.merges[0].state[0] -= 2
    grp.fill()
    outs = grp.outputs()
    torch.cuda.synchronize()
    before, timeouts = _epochs(grp), [int(m.state[3]) for m in grp.merges]
    grp.call(outs)
    return grp, before, timeouts


def test_a_stale_epoch_is_caught() -> None:
    """The check above is not vacuous: rank 0's stale epoch is reported. The kernels are built
    without a trap record (``record_address`` patched to 0: a timed-out wait is only counted in
    ``state[3]``), so the check sees the call through; the served record traps (next test)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.common import (
        rubin_ull_spin_trap as spin_trap,
    )

    with mock.patch.object(spin_trap, "record_address", return_value=0):
        grp, before, timeouts = _stale_epoch_call()
        torch.cuda.synchronize()
    assert _contract_violations(grp, before, timeouts)
    assert spin_trap.recorded() is None


_CHILD = textwrap.dedent(
    """
    import json, os, sys
    sys.path[:0] = sys.argv[1:3]
    import torch
    import test_rubin_ull_dsa_key_shard_select as t
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.common import (
        rubin_ull_spin_trap as spin_trap,
    )

    error = None
    try:
        t._stale_epoch_call()
        torch.cuda.synchronize()
    except Exception as e:  # the trap: a CUDA error on the next synchronize
        error = type(e).__name__
    spin_trap.log_trap()
    print("RECORD " + json.dumps(dict(error=error, record=spin_trap.recorded())), flush=True)
    os._exit(0)  # the CUDA context is dead: skip the teardown
    """
)


#: The child is no rank of the caller's job step: without the launcher's PMI / PMIx / Slurm
#: variables the MPI it initializes on importing tensorrt_llm (mpi4py) is a singleton, so its exit
#: after the trap (no MPI finalize) cannot make Slurm abort the caller's step.
_LAUNCHER_PREFIXES = ("PMIX_", "PMI_", "SLURM_")


def _child_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_LAUNCHER_PREFIXES)}
    env["OMPI_MCA_ess_singleton_isolated"] = "1"
    return env


def test_a_stale_epoch_traps_with_the_served_record() -> None:
    """With the served trap record (``spin_trap.record_address()``) the same call ends in a trap:
    the first peer whose wait for rank 0's pushes timed out records a key-shard site (171 flag
    merge, 172 own LL slots, 173 exchange merge, 174 rescue; a / b / c = row / thread / epoch)."""
    here = Path(__file__).resolve().parent
    run = subprocess.run(
        [sys.executable, "-c", _CHILD, str(here), str(here.parents[3])],
        capture_output=True,
        text=True,
        timeout=900,
        env=_child_env(),
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith("RECORD ")]
    assert run.returncode == 0 and lines, f"child failed ({run.returncode}):\n{run.stderr[-4000:]}"
    result = json.loads(lines[-1][len("RECORD ") :])
    record = result["record"]
    assert result["error"] is not None, "the stale epoch did not trap"
    assert record is not None and record["site"] in (171, 172, 173, 174), record
    assert 0 <= record["a"] < 4 and record["c"] > 0, record
    assert f"K31_POLL_TIMEOUT site={record['site']} " in run.stderr


def test_is_valid_refuses_what_the_kernel_would_read_wrongly() -> None:
    spec = dict(world=2, requests=1, next_n=1, context=5_000, kv_rows=True, bounds=False,
                hot=False)  # fmt: skip
    grp = Group(spec, seed=3)
    run = grp.runners[0]
    long = grp.row_lens[0].clone()
    long[0] = grp.merges[0].local_width + 1
    with pytest.raises(ValueError, match="past the workspace"):
        op.is_valid(run, long, 1)
    with pytest.raises(ValueError, match="request indices"):
        op.is_valid(run, grp.row_lens[0], 2, kv_rows=grp.kv[0])
