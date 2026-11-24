# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 strict-verification entry (CuTe DSL).

Every certified cell is gated bitwise against the entry's reference (the family's torch
``dflash2_greedy_verify``); the family's own strict-acceptance test is rerun with the kernel in
place of the torch function.
"""

import importlib.util
import pathlib
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec import (
    dflash2_greedy_verify as verify_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_greedy_verify import (  # noqa: E501
    dflash2_greedy_verify as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x",
)

_DRAWS = 4


def _build(spec, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n, g, vocab = spec["batch"], spec["drafts"], spec["vocab"]
    logits = 4.0 * torch.randn(n, g + 1, vocab, device="cuda", generator=gen)
    if spec["ties"]:
        # A repeated maximum in every row: the lower index must win.
        top = logits.argmax(-1, keepdim=True)
        twin = (
            top + 1 + torch.randint(0, vocab - 1, top.shape, device="cuda", generator=gen)
        ) % vocab
        logits.scatter_(-1, twin, logits.gather(-1, top))
    logits = logits.to(spec["dtype"])
    for b in range(spec["nan_rows"]):
        logits[b, b % (g + 1), (7919 * (b + 1)) % vocab] = float("nan")
        if b == 0:
            logits[b, 0, 11] = float("nan")  # two NaNs in one row: the first wins
    best = logits.argmax(-1)
    agree = torch.rand(n, g, device="cuda", generator=gen) < spec["agree"]
    other = torch.randint(0, vocab, (n, g), device="cuda", generator=gen)
    drafts = torch.where(agree, best[:, :g], other).to(spec["draft_dtype"])
    return logits, drafts


def _assert_equal(got, want, what="") -> None:
    for name, a, b in zip(("tau", "bonus"), got, want):
        assert a.shape == b.shape and a.dtype == b.dtype, (name, a.shape, b.shape, a.dtype, b.dtype)
        bad = (a != b).nonzero().flatten().tolist()
        assert not bad, f"{what} {name}: requests {bad[:8]} differ"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    base = zlib.crc32(cell.why.encode()) % 2**31
    for draw in range(_DRAWS):
        args = _build(cell.spec, seed=base + draw)
        _assert_equal(op(*args), op.reference(*args), f"draw {draw}")


def test_one_launch_per_call() -> None:
    args = _build(op.CELLS[4].spec, seed=1)
    op(*args), op.reference(*args)
    torch.cuda.synchronize()
    counts = {}
    for name, fn in (("op", op), ("reference", op.reference)):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(*args)
            torch.cuda.synchronize()
        counts[name] = sum(
            1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        )
    print(f"device launches per call: {counts}")
    assert counts["op"] == 1, counts


def test_cuda_graph_replay() -> None:
    spec = op.CELLS[3].spec
    logits, drafts = _build(spec, seed=2)
    op(logits, drafts)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(logits, drafts)
    for seed in (3, 4):
        fresh = _build(spec, seed=seed)
        logits.copy_(fresh[0])
        drafts.copy_(fresh[1])
        graph.replay()
        torch.cuda.synchronize()
        _assert_equal(out, op.reference(logits, drafts), f"replay {seed}")


def test_workspace_bounded_over_batch_sizes() -> None:
    """Every batch size slices the workspace of its power-of-two request capacity, so the cache stays bounded
    however the batch varies; each call keeps the reference's bits, and a graph captured at a batch size whose
    workspace a larger eager call made replays correctly."""
    spec = op.CELLS[2].spec
    before = set(verify_kernel._WORKSPACES)
    for n in (7, 3, 8, 1, 5, 2, 6):
        args = _build(dict(spec, batch=n), seed=40 + n)
        _assert_equal(op(*args), op.reference(*args), f"batch {n}")
    added = set(verify_kernel._WORKSPACES) - before
    assert {key[1] for key in added} <= {1, 2, 4, 8}, added
    assert all(key[1] & (key[1] - 1) == 0 for key in verify_kernel._WORKSPACES)
    g = spec["drafts"]
    device = torch.device("cuda", torch.cuda.current_device())
    shared = [verify_kernel._workspace(device, n, g) for n in (5, 8)]
    assert shared[0][0].data_ptr() == shared[1][0].data_ptr()
    assert shared[0][0].numel() == 5 * (g + 1) * verify_kernel.MAX_SLICES
    assert shared[0][2].numel() == 5

    logits, drafts = _build(dict(spec, batch=5), seed=50)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(logits, drafts)
    fresh = _build(dict(spec, batch=5), seed=51)
    logits.copy_(fresh[0])
    drafts.copy_(fresh[1])
    graph.replay()
    torch.cuda.synchronize()
    _assert_equal(out, op.reference(logits, drafts), "replay at 5 requests")


def test_guard_refuses_int_logits() -> None:
    logits, drafts = _build(op.CELLS[2].spec, seed=5)
    with validating(op), pytest.raises(AssertionError, match="logits"):
        op(logits.to(torch.int32), drafts)


_FAMILY_TEST = pathlib.Path(__file__).resolve().with_name("test_dflash2_verify.py")


def _family_tests():
    spec = importlib.util.spec_from_file_location("_k31_family_dflash2_verify_g", _FAMILY_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_family_strict_rule_on_kernel(monkeypatch) -> None:
    family = _family_tests()
    monkeypatch.setattr(family, "dflash2_greedy_verify", lambda lg, d: op(lg, d))
    family.test_greedy_verify_is_strict()
