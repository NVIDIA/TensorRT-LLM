# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-process contracts of the finalize+all-reduce module: imports, input checks, release order.

These need no peers. Argument validation runs on a workspace whose launcher is
replaced by a recorder, so a rejected call is proven to launch nothing. Release
runs on a workspace holding fake buffers, with the TP group and the CUDA calls
replaced by recorders, so the order of every step can be checked exactly.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.models import musespark_open_finalize_ar
from tensorrt_llm._torch.models.musespark_open_finalize_ar import FinalizeIpcWorkspace

_BLOCKED_DSL = """
import sys
for name in ("cutlass", "cutlass.cute", "cutlass.utils", "cutlass._mlir", "cutlass.cutlass_dsl"):
    sys.modules[name] = None
from tensorrt_llm._torch.models import musespark_open_boundary, musespark_open_finalize_ar
from tensorrt_llm._torch.models.modeling_musespark_open import _build_finalize_allreduce_workspace
print(musespark_open_finalize_ar.HAS_CUTEDSL, musespark_open_boundary._HAS_CUTE_DSL,
      _build_finalize_allreduce_workspace(object()))
"""


def test_modules_import_without_cute_dsl() -> None:
    """Both kernels report unavailable and the model keeps the unfused path."""
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKED_DSL], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split()[-3:] == ["False", "False", "None"]


@pytest.fixture
def recorded():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    workspace = FinalizeIpcWorkspace.__new__(FinalizeIpcWorkspace)
    workspace._device = torch.device("cuda", torch.cuda.current_device())
    workspace._ranks, workspace._rank = 4, 0
    workspace._export, workspace._imports = 0, []
    workspace._kernel = object()
    workspace._closed = False
    launches = []
    workspace._launch = lambda *args, **kwargs: launches.append((args, kwargs))
    return workspace, launches


def _valid(tokens=4, rows=64, width=4096):
    gemm = torch.zeros(rows, width, device="cuda", dtype=torch.bfloat16)
    indices = torch.full((tokens, 8), -1, device="cuda", dtype=torch.int32)
    weights = torch.zeros(tokens, 8, device="cuda")
    norm = torch.zeros(4096, device="cuda", dtype=torch.bfloat16)
    return [gemm, indices, weights, norm, 1e-8, tokens]


def _malformed():
    """Each entry replaces one argument of a valid call."""
    gemm = torch.zeros(64, 4096, device="cuda", dtype=torch.bfloat16)
    return {
        "fp32 rows": (0, gemm.float()),
        "narrow rows": (0, gemm[:, :2048]),
        "transposed rows": (0, torch.zeros(4096, 64, device="cuda", dtype=torch.bfloat16).t()),
        "misaligned rows": (
            0,
            torch.zeros(64 * 4096 + 8, device="cuda", dtype=torch.bfloat16)[1 : 1 + 64 * 4096].view(
                64, 4096
            ),
        ),
        "row stride not a multiple of 8": (
            0,
            torch.zeros(64, 4100, device="cuda", dtype=torch.bfloat16)[:, :4096],
        ),
        "cpu rows": (0, gemm.cpu()),
        "int64 indices": (1, torch.full((4, 8), -1, device="cuda", dtype=torch.int64)),
        "short indices": (1, torch.full((4, 4), -1, device="cuda", dtype=torch.int32)),
        "strided indices": (1, torch.full((4, 16), -1, device="cuda", dtype=torch.int32)[:, ::2]),
        "fp16 weights": (2, torch.zeros(4, 8, device="cuda", dtype=torch.float16)),
        "short weights": (2, torch.zeros(4, 4, device="cuda")),
        "short norm": (3, torch.zeros(2048, device="cuda", dtype=torch.bfloat16)),
        "fp32 norm": (3, torch.zeros(4096, device="cuda")),
        "misaligned norm": (3, torch.zeros(4104, device="cuda", dtype=torch.bfloat16)[1:4097]),
        "zero eps": (4, 0.0),
        "negative eps": (4, -1e-6),
        "nan eps": (4, float("nan")),
        "inf eps": (4, float("inf")),
        "no tokens": (5, 0),
        "too many tokens": (5, 129),
        "float token count": (5, 4.0),
    }


_MALFORMED_CASES = [
    "fp32 rows",
    "narrow rows",
    "transposed rows",
    "misaligned rows",
    "row stride not a multiple of 8",
    "cpu rows",
    "int64 indices",
    "short indices",
    "strided indices",
    "fp16 weights",
    "short weights",
    "short norm",
    "fp32 norm",
    "misaligned norm",
    "zero eps",
    "negative eps",
    "nan eps",
    "inf eps",
    "no tokens",
    "too many tokens",
    "float token count",
]


@pytest.mark.parametrize("case", _MALFORMED_CASES)
def test_malformed_call_raises_before_launch(recorded, case: str) -> None:
    workspace, launches = recorded
    position, value = _malformed()[case]
    arguments = _valid()
    arguments[position] = value
    with pytest.raises(ValueError):
        workspace.finalize(*arguments)
    assert launches == []


@pytest.mark.parametrize(
    "rows, width, stride",
    [(0, 4096, None), (1, 4096, None), (1, 4100, 4100), (64, 4096, None), (64, 4160, 4160)],
)
def test_legal_per_rank_differences_launch(recorded, rows: int, width: int, stride) -> None:
    """Row counts and padded or arbitrary strides may differ between ranks."""
    workspace, launches = recorded
    arguments = _valid(rows=rows, width=width)
    if stride is not None:
        arguments[0] = arguments[0][:, :4096]
    output = workspace.finalize(*arguments)
    assert len(launches) == 1
    assert output.shape == (4, 4096) and output.dtype == torch.bfloat16
    args, _ = launches[0]
    gemm_pointer, row_count = args[0], args[5]
    assert row_count == rows
    assert gemm_pointer == (0 if rows == 0 else arguments[0].data_ptr())


def test_construction_stage_reports_any_exception_as_failure() -> None:
    """A rank must reach the agreement whatever its stage raised, or its peers wait forever."""

    def broken_stage(argument):
        raise KeyError(argument)

    status, detail = FinalizeIpcWorkspace._guarded(broken_stage, "injected")
    assert status == musespark_open_finalize_ar._FAILED
    assert detail.startswith("KeyError")


def test_closed_workspace_rejects_calls(recorded) -> None:
    workspace, launches = recorded
    workspace._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        workspace.finalize(*_valid())
    assert launches == []


class _FakeGroup:
    """Four-rank TP group whose peers echo this rank, except one peer that can fail one step."""

    def __init__(self, events):
        self.events = events
        self.failing_step = None
        self.steps = 0

    def tp_allgather_int64(self, values):
        self.steps += 1
        self.events.append("agree")
        table = np.array([list(values)] * 4, dtype=np.int64)
        if self.steps == self.failing_step:
            table[2, 0] = 0
        return table


_UNMAPS = ["unmap 0x1000", "unmap 0x2000", "unmap 0x3000"]
_ORDERED_RELEASE = ["sync", "agree", *_UNMAPS, "agree", "free 0xe000", "agree"]


@pytest.fixture
def releasable(monkeypatch: pytest.MonkeyPatch):
    """A workspace holding a fake IPC buffer and three fake peer mappings."""
    if not torch.cuda.is_available() or not hasattr(musespark_open_finalize_ar, "cudart"):
        pytest.skip("CUDA and cuda-python required")
    runtime = musespark_open_finalize_ar.cudart
    events = []

    def unmap(pointer):
        events.append(f"unmap {pointer:#x}")
        return (runtime.cudaError_t.cudaSuccess,)

    def free(pointer):
        events.append(f"free {pointer:#x}")
        return (runtime.cudaError_t.cudaSuccess,)

    def sync():
        events.append("sync")
        return True

    monkeypatch.setattr(runtime, "cudaIpcCloseMemHandle", unmap)
    monkeypatch.setattr(runtime, "cudaFree", free)
    workspace = FinalizeIpcWorkspace.__new__(FinalizeIpcWorkspace)
    workspace._device = torch.device("cuda", torch.cuda.current_device())
    workspace._export, workspace._imports = 0xE000, [0x1000, 0x2000, 0x3000]
    workspace._dist = _FakeGroup(events)
    workspace._sync = sync
    workspace._closed = False
    yield workspace, events
    # The fake pointers must never reach CUDA once the recorders are gone.
    workspace._closed = True


def test_close_frees_only_after_the_group_unmapped(releasable) -> None:
    workspace, events = releasable
    workspace.close()
    assert events == _ORDERED_RELEASE
    workspace.close()
    assert events == _ORDERED_RELEASE, "a second close must do nothing"


def test_failed_construction_releases_in_the_same_order(releasable) -> None:
    workspace, events = releasable
    with pytest.raises(ValueError, match="injected"):
        workspace._abandon(ValueError, "injected")
    assert events == _ORDERED_RELEASE


@pytest.mark.parametrize(
    "failing_step, released",
    [(1, ["sync", "agree"]), (2, ["sync", "agree", *_UNMAPS, "agree"])],
    ids=["peer_sync_fails", "peer_unmap_fails"],
)
def test_peer_failure_frees_nothing(releasable, failing_step: int, released) -> None:
    workspace, events = releasable
    workspace._dist.failing_step = failing_step
    with pytest.raises(RuntimeError, match=r"rank\(s\) \[2\]"):
        workspace.close()
    assert events == released


def test_local_unmap_failure_frees_nothing(releasable, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, events = releasable
    runtime = musespark_open_finalize_ar.cudart
    monkeypatch.setattr(
        runtime,
        "cudaIpcCloseMemHandle",
        lambda pointer: (runtime.cudaError_t.cudaErrorInvalidValue,),
    )
    with pytest.raises(RuntimeError, match="unmapping"):
        workspace.close()
    assert "free 0xe000" not in events


def test_local_sync_failure_releases_nothing(releasable) -> None:
    workspace, events = releasable
    workspace._sync = lambda: False
    with pytest.raises(RuntimeError, match="synchronization"):
        workspace.close()
    assert events == ["agree"]


def test_drop_unmaps_peers_but_frees_nothing(releasable) -> None:
    """Without close() the destructor involves no other rank and keeps the shared buffer."""
    workspace, events = releasable
    workspace.__del__()
    assert events == ["sync", *_UNMAPS]


def test_drop_after_failed_sync_releases_nothing(releasable) -> None:
    workspace, events = releasable
    workspace._sync = lambda: False
    workspace.__del__()
    assert events == []
