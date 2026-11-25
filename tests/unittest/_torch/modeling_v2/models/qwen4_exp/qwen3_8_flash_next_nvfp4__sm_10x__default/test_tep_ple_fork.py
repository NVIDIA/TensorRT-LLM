# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tep4 rank's PLE front on its own stream (`DecodeTarget._ple_fork`), CPU only.

* **When** (`_ple_forks`): a tep4 rank with its fast forms, layers before the PLE layer, and no
  TP sum in the front; never at another layout.
* **Fork and join**: the PLE stream waits on the forward's stream before the front, an eager
  step marks the key and value for the forward's stream (a captured one does not), and the PLE
  layer waits on the PLE stream before the tail reads them.

Bitwise by construction (the same launches on another stream); the tep4 accuracy cases run it.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


def _target(tep_fast=True, ple_layer=1, stream=True, needs_sum=False, monkeypatch=None):
    monkeypatch.setattr(M, "_ple_needs_sum", lambda core: needs_sum)
    target = M.DecodeTarget.__new__(M.DecodeTarget)
    target.core = SimpleNamespace(
        tep_fast=tep_fast, ple_layer=ple_layer, _ple_stream=object() if stream else None
    )
    return target


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "kw, expected",
    [
        (dict(), True),
        (dict(tep_fast=False), False),
        (dict(ple_layer=0), False),  # nothing before the PLE layer to overlap
        (dict(stream=False), False),
        (dict(needs_sum=True), False),  # a TP sum on a second stream
    ],
)
def test_when(monkeypatch, kw, expected):
    assert _target(monkeypatch=monkeypatch, **kw)._ple_forks() is expected


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(f"{self.name} waits {other.name}")


@pytest.mark.cpu_only
@pytest.mark.parametrize("capturing", [False, True])
def test_fork_and_join(monkeypatch, capturing):
    log = []
    main, fork = _Stream("main", log), _Stream("ple", log)
    current = [main]

    @contextlib.contextmanager
    def stream(s):
        current.append(s)
        try:
            yield
        finally:
            current.pop()

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current[-1])
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    target = _target(monkeypatch=monkeypatch)
    target.core._ple_stream = fork

    def out(name):
        return SimpleNamespace(record_stream=lambda s: log.append(f"{name} marked for {s.name}"))

    def front(ids, state, stage):
        log.append(f"front on {current[-1].name}")
        return out("key"), out("value")

    target._ple_front = front
    key_value = target._ple_fork(None, None, True)
    assert log == ["ple waits main", "front on ple"] + (
        [] if capturing else ["key marked for main", "value marked for main"]
    )
    log.clear()
    monkeypatch.setattr(M, "ple_gate_conv_inject_norm", lambda hc, k, v, *a, **kw: (k, v))
    target._stack = SimpleNamespace(ple_nk_w=None, ple_nq_w=None, ple_nc_w=None, ple_conv_w=None)
    target.core.eps, target.core.hc_count, target.core.ple_dilation = 1e-6, 4, 2
    got = target._ple_block(
        torch.zeros(4, 8), None, (None, None, None, None, 4), None, False, key_value
    )
    assert log == ["main waits ple"] and got == key_value


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tep_fast, ids, expected",
    [
        (True, torch.tensor([3, 0, 5]), "embed_rows"),
        (True, torch.tensor([3, 0, 5], dtype=torch.int32), "embed_rows"),
        (True, torch.tensor([[3, 0, 5]]).t()[:, 0], "embed_rows"),  # a contiguous view
        (True, torch.tensor([3, 9, 0, 9, 5])[::2], "embedding"),  # strided ids
        (False, torch.tensor([3, 0, 5]), "embedding"),
    ],
)
def test_head_rows(monkeypatch, tep_fast, ids, expected):
    """The verify head's row gather (the rest of the measured TEP_GLUE)."""
    calls = []
    monkeypatch.setattr(
        M,
        "embed_rows",
        lambda i, t, copies, clamp: calls.append("embed_rows") or torch.index_select(t, 0, i),
    )
    final = torch.arange(24.0).reshape(6, 4)
    got = M._head_rows(SimpleNamespace(tep_fast=tep_fast), ids, final)
    assert calls == ([] if expected == "embedding" else ["embed_rows"])
    assert torch.equal(got, torch.nn.functional.embedding(ids.long(), final))
