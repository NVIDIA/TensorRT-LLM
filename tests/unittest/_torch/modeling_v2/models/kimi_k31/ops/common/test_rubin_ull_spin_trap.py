# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's spin_trap_wait entry: the device wait bound of
``kernels/common/rubin_ull_spin_trap.py``.

A published flag returns without writing the flag or the trap record; a flag published by another
stream while the kernel spins returns only after it lands. A wait that runs out traps: those cells
run in a child process (the trap ends the CUDA context) with the bound lowered to about 67 ms, and
check the record the host reads after the CUDA error and the ``K31_POLL_TIMEOUT`` line. The child
then ends with ``os._exit`` (no atexit hook, as a rank killed after a fatal error), so the line
can only come from the record's watcher thread, and it must come once."""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.common.rubin_ull_spin_trap import (  # noqa: E501
    spin_trap_wait as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="the bounded waits run where the K3.1 ULL target does (SM10x)",
    ),
    # spin_trap.record_address() starts the trap-record watcher, a daemon thread that lives as
    # long as the process (kernels/common/rubin_ull_spin_trap.py).
    pytest.mark.threadleak(enabled=False),
]

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
#: The child's bound: short enough for a test, far above a spin round.
CHILD_BOUND_NS = 1 << 26

_CHILD = textwrap.dedent(
    f"""
    import json, os, sys, time
    import torch
    from {_FAMILY}.kernels.common import rubin_ull_spin_trap as spin_trap
    from {_FAMILY}.ops.common.rubin_ull_spin_trap import spin_trap_wait

    spin_trap.BOUND_NS = {CHILD_BOUND_NS}  # read when the wait is compiled
    mode, threads, value = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    flag = torch.full((1,), value, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    error = None
    try:
        spin_trap_wait(flag, mode, threads)
        torch.cuda.synchronize()
    except Exception as e:  # the trap: a CUDA error on the next synchronize
        error = type(e).__name__
    time.sleep(2 * spin_trap.WATCH_PERIOD_S + 1)  # the watcher logs the record; no log_trap() here
    print("RECORD " + json.dumps(dict(error=error, record=spin_trap.recorded())), flush=True)
    os._exit(0)  # the CUDA context is dead: skip the teardown (and the atexit hook)
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


def _child(mode: str, threads: int, value: int) -> tuple[dict, str]:
    run = subprocess.run(
        [sys.executable, "-c", _CHILD, mode, str(threads), str(value)],
        capture_output=True,
        text=True,
        timeout=900,
        env=_child_env(),
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith("RECORD ")]
    assert run.returncode == 0 and lines, f"child failed ({run.returncode}):\n{run.stderr[-4000:]}"
    return json.loads(lines[-1][len("RECORD ") :]), run.stderr


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.common import (
        rubin_ull_spin_trap as spin_trap,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.common import (
        rubin_ull_spin_trap_wait as wait,
    )

    mode, publish, threads = cell.spec["mode"], cell.spec["publish"], cell.spec["threads"]
    if publish == "never":
        value = {"tick": 0, "expired": 1, "expired_at": 2}[mode]
        result, stderr = _child(mode, threads, value)
        record = result["record"]
        assert result["error"] is not None, "the wait did not trap"
        assert record is not None, "the trap left no record"
        site = {
            "tick": wait.SITE_TICK,
            "expired": wait.SITE_EXPIRED,
            "expired_at": wait.SITE_EXPIRED_AT + value - 1,
        }[mode]
        assert (record["site"], record["block"]) == (site, 0)
        assert 0 <= record["thread"] < threads and record["a"] == record["thread"]
        assert (record["b"], record["c"]) == (value, 0)
        if mode == "tick":
            # Lane 0 of a warp trips at the bound, the other lanes at twice it.
            assert record["thread"] % 32 == 0
            assert record["waited_ns"] > CHILD_BOUND_NS and record["spins"] > 0
        else:
            assert (record["spins"], record["waited_ns"]) == (-1, 0)
        assert record["gtimer"] > 0
        assert f"K31_POLL_TIMEOUT site={site} block=0 thread={record['thread']}" in stderr
        assert stderr.count("K31_POLL_TIMEOUT") == 1, "the record was logged more than once"
        return

    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    if mode == "tick":
        # Compile and load the wait first: no module may load after the spinner is resident.
        flag.fill_(1)
        op(flag, mode, threads)
        flag.clone()  # the copy the late cell reads the flag back with
        torch.cuda.synchronize()
        flag.zero_()
    if publish == "before":
        if mode == "tick":
            flag.fill_(1)
        with validating(op):
            op(flag, mode, threads)
        torch.cuda.synchronize()
    else:
        main = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(main)
        # The publish is enqueued first: no launch may follow the spinning kernel on the host (a
        # lazily loaded module, e.g. _sleep's, would wait for the resident spinner: a deadlock).
        with torch.cuda.stream(side):
            torch.cuda._sleep(20_000_000)  # about 10 ms: the kernel spins well before it lands
            flag.fill_(7)
        with validating(op):
            op(flag, mode, threads)
        seen = flag.clone()  # what the next op on the stream reads: 7 only if the wait waited
        torch.cuda.synchronize()
        assert int(seen) == 7, "the stream went on before the flag was published"
    assert op.reference(flag, mode, threads) is None
    assert spin_trap.recorded() is None, "a wait that returned wrote the trap record"
    assert int(flag) == (0 if mode != "tick" else 1 if publish == "before" else 7)


def test_is_valid_refuses_what_the_kernel_would_misread() -> None:
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    op.is_valid(flag, "tick", 32)
    with pytest.raises(ValueError, match="int32 CUDA word"):
        op.is_valid(flag.long(), "tick", 32)
    with pytest.raises(ValueError, match="mode"):
        op.is_valid(flag, "poll", 32)
    with pytest.raises(ValueError, match="whole warps"):
        op.is_valid(flag, "tick", 48)
