# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 DSpark worker's target-head rows (``K3DFlashWorker.target_head_rows``).

On the ``tail_v2`` path (the helix8 targets) a generation-only batch hands the target head the hidden
states themselves: its gather ids are 0..num_tokens-1, so the index copy is an identity. Every other
batch, and every other target, gathers. The spec shell calls the hook for the rows the head reads.
"""

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"


def _worker(path):
    worker_cls = importlib.import_module(f"{_FAMILY}.dspark.worker").K3DSparkWorker
    worker = worker_cls.__new__(worker_cls)
    nn.Module.__init__(worker)
    worker._path = path
    return worker


def _path(target):
    drafter = importlib.import_module(f"{_FAMILY}.dspark.drafter")
    constants = importlib.import_module(
        f"{_FAMILY}.kimi_k3_nvfp4__sm_107__{target}.modeling"
    ).CONSTANTS
    return drafter.K3DSparkPath(
        spec_tail_fusion=constants.spec_tail_fusion, tail_v2=constants.dspark_tail_v2
    )


@pytest.mark.parametrize(
    "target,skips",
    [("helix8_moeep8", True), ("helix8_moetp2ep4", True), ("dep16", False), ("tp1", False)],
)
@pytest.mark.parametrize("num_contexts", [0, 1])
def test_generation_only_batch_skips_the_gather_on_tail_v2(target, skips, num_contexts):
    worker = _worker(_path(target))
    hidden = torch.randn(6, 4)
    gather_ids = torch.arange(6) if num_contexts == 0 else torch.tensor([2, 5])
    rows = worker.target_head_rows(
        hidden, SimpleNamespace(num_contexts=num_contexts), SimpleNamespace(gather_ids=gather_ids)
    )
    if skips and num_contexts == 0:
        assert rows is hidden
    else:
        assert rows is not hidden
        assert torch.equal(rows, hidden[gather_ids])


def test_spec_shell_reads_the_rows_from_the_worker():
    shell = importlib.import_module(f"{_FAMILY}.dspark.shell").KimiK3SpecForCausalLM
    source = inspect.getsource(shell.forward)
    assert (
        "self.spec_worker.target_head_rows(hidden_states, attn_metadata, spec_metadata)" in source
    )
    assert "hidden_states[spec_metadata.gather_ids]" not in source
