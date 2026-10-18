# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``drafter_fc_shard``: the DFlash ``fc`` column-sharded over TP8.

One GPU: the bound ``project_target_hidden`` is driven for each of the eight
ranks with a stand-in gather that computes every rank's slice the way that
rank would, so the rank's row range, the gather's column order and the
composition with ``hidden_norm`` are checked against the built-in projection
(within the GEMM band) and bitwise against the same composition done by hand;
a drafter weight reload (a new ``fc``, or new storage under the old one) is
followed on the next call.

Eight MPI ranks on two NVLink-connected four-GPU GB300 nodes
(``test_sharded_fc_tp8_matches_builtin``; skipped otherwise): the real
projection-gather workspace, eager and under CUDA-graph replay.
"""

import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    spec_worker as drafter_worker,
)
from tensorrt_llm._torch.models import musespark_open_projection_gather
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp  # noqa: E402

assert torch.cuda.is_available(), "the sharded fc requires a CUDA device"

HIDDEN, TAPPED = 8192, 5
#: In units of 2^-8 x the row's largest magnitude (``assert_within_ulp``).
#: The sharded and built-in projections differ only in the fc GEMM's K order
#: (both fp32-accumulated, rounded once): at most one bf16 rounding step of the
#: row scale before the norm, i.e. 2 units. RMSNorm rescales each row by one
#: shared factor (carrying that band through) and rounds its output once on
#: each side: 2 more.
ELEMENT_ULP, RMS_ULP = 4.0, 0.5


def _drafter(seed=0, device="cuda"):
    drafter = DFlashForCausalLM.__new__(DFlashForCausalLM)
    nn.Module.__init__(drafter)
    g = torch.Generator(device=device).manual_seed(seed)
    drafter.fc = nn.Linear(TAPPED * HIDDEN, HIDDEN, bias=False, device=device, dtype=torch.bfloat16)
    drafter.fc.weight.data.copy_(
        (0.01 * torch.randn(HIDDEN, TAPPED * HIDDEN, generator=g, device=device)).to(torch.bfloat16)
    )
    drafter.hidden_norm = nn.RMSNorm(HIDDEN, eps=1e-5, device=device, dtype=torch.bfloat16)
    drafter.hidden_norm.weight.data.copy_(
        (1.0 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).to(torch.bfloat16)
    )
    return drafter


class _EightRankGather:
    """Stands in for the fabric gather on one GPU: every rank's slice, as that rank computes it.

    Reads the drafter's ``fc`` when it gathers, so it follows a weight reload.
    """

    def __init__(self, x, drafter, rank, enable_pdl):
        self.x, self.drafter, self.rank, self.pdl = x, drafter, rank, enable_pdl
        self.calls = 0

    def gather(self, local):
        weight = self.drafter.fc.weight.detach()
        rows = weight.shape[0] // 8
        slices = [
            drafter_worker._gemm(self.x, weight[r * rows : (r + 1) * rows], self.pdl)
            for r in range(8)
        ]
        assert torch.equal(slices[self.rank], local), "the rank's own slice differs"
        self.calls += 1
        return torch.cat(slices, dim=1)


@pytest.mark.parametrize("rows", [8, 1, 4])
@torch.inference_mode()
def test_each_rank_projects_its_rows_and_the_gather_restores_order(rows):
    drafter = _drafter()
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = (2.0 * torch.randn(rows, TAPPED * HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    builtin = DFlashForCausalLM.project_target_hidden(drafter, x)
    for rank in range(8):
        mapping = SimpleNamespace(tp_size=8, tp_rank=rank, enable_attention_dp=False, cp_size=1)
        gather = _EightRankGather(x, drafter, rank, True)
        drafter.__dict__.pop("project_target_hidden", None)
        assert drafter_worker.install_sharded_fc(drafter, mapping, True, gather=gather) is None
        out = drafter.project_target_hidden(x)
        assert gather.calls == 1
        by_hand = drafter.hidden_norm(
            gather.gather(
                drafter_worker._gemm(x, drafter.fc.weight[rank * 1024 : (rank + 1) * 1024], True)
            )
        )
        assert torch.equal(out, by_hand)
        assert_within_ulp(out, builtin, ELEMENT_ULP, RMS_ULP)


@torch.inference_mode()
def test_prefill_sized_calls_run_the_builtin():
    drafter = _drafter()
    mapping = SimpleNamespace(tp_size=8, tp_rank=3, enable_attention_dp=False, cp_size=1)
    x9 = torch.randn(9, TAPPED * HIDDEN, device="cuda").to(torch.bfloat16)
    gather = _EightRankGather(x9, drafter, 3, True)
    assert drafter_worker.install_sharded_fc(drafter, mapping, True, gather=gather) is None
    out = drafter.project_target_hidden(x9)
    assert gather.calls == 0
    assert torch.equal(out, DFlashForCausalLM.project_target_hidden(drafter, x9))


@torch.inference_mode()
def test_a_drafter_weight_reload_is_followed(monkeypatch):
    drafter = _drafter(seed=1)
    rank = 5
    mapping = SimpleNamespace(tp_size=8, tp_rank=rank, enable_attention_dp=False, cp_size=1)
    g = torch.Generator(device="cuda").manual_seed(21)
    x = (2.0 * torch.randn(8, TAPPED * HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    gather = _EightRankGather(x, drafter, rank, True)
    assert drafter_worker.install_sharded_fc(drafter, mapping, True, gather=gather) is None
    first = drafter.project_target_hidden(x)

    # The DFlash loader builds a new fc and hidden_norm on every load; the shell
    # installs again after it, which keeps the bound method and the workspace.
    def no_new_workspace(mapping):
        raise AssertionError("a repeated install must not build another gather workspace")

    monkeypatch.setattr(
        musespark_open_projection_gather, "ProjectionGatherWorkspace", no_new_workspace
    )
    reloaded = _drafter(seed=2)
    drafter.fc, drafter.hidden_norm = reloaded.fc, reloaded.hidden_norm
    del reloaded
    assert drafter_worker.install_sharded_fc(drafter, mapping, True) is None
    assert drafter._fc_shard.gather is gather
    out = drafter.project_target_hidden(x)  # the gather checks the rank's slice of the new fc
    assert gather.calls == 2
    assert_within_ulp(
        out, DFlashForCausalLM.project_target_hidden(drafter, x), ELEMENT_ULP, RMS_ULP
    )
    assert not torch.equal(out, first)

    # A reload that rebinds the parameter's storage is followed without an install.
    drafter.fc.weight.data = _drafter(seed=3).fc.weight.data
    out = drafter.project_target_hidden(x)
    assert gather.calls == 3
    assert_within_ulp(
        out, DFlashForCausalLM.project_target_hidden(drafter, x), ELEMENT_ULP, RMS_ULP
    )

    # An fc the sharded path does not cover runs the built-in, gather untouched.
    drafter.fc.bias = nn.Parameter(torch.ones(HIDDEN, dtype=torch.bfloat16, device="cuda"))
    out = drafter.project_target_hidden(x)
    assert gather.calls == 3
    assert torch.equal(out, DFlashForCausalLM.project_target_hidden(drafter, x))


def test_install_declines_what_it_does_not_cover():
    drafter = _drafter(device="cpu")
    ok = SimpleNamespace(tp_size=8, tp_rank=0, enable_attention_dp=False, cp_size=1)
    assert "TP size" in drafter_worker.install_sharded_fc(
        drafter, SimpleNamespace(tp_size=4, tp_rank=0), True, gather=object()
    )
    assert "attention DP" in drafter_worker.install_sharded_fc(
        drafter, SimpleNamespace(**{**vars(ok), "enable_attention_dp": True}), True, gather=object()
    )

    class Laguna(DFlashForCausalLM):
        def project_target_hidden(self, hidden_states):
            return hidden_states

    other = _drafter(device="cpu")
    other.__class__ = Laguna
    assert "overrides" in drafter_worker.install_sharded_fc(other, ok, True, gather=object())
    assert "project_target_hidden" not in drafter.__dict__
    del other

    # Another feature already replaced the method on this drafter: left as it is.
    foreign = types.MethodType(lambda self, hidden_states: hidden_states, drafter)
    drafter.project_target_hidden = foreign
    assert drafter_worker.install_sharded_fc(drafter, ok, True, gather=object()) == (
        "project_target_hidden is already replaced on this drafter by "
        f"{__name__}.test_install_declines_what_it_does_not_cover.<locals>.<lambda>"
    )
    assert drafter.__dict__["project_target_hidden"] is foreign
    del drafter.project_target_hidden

    # A reload that leaves an fc this path does not cover puts the built-in back.
    assert drafter_worker.install_sharded_fc(drafter, ok, True, gather=object()) is None
    assert "project_target_hidden" in drafter.__dict__
    drafter.fc = nn.Linear(16, 16, bias=True, dtype=torch.bfloat16)
    assert "bias-free" in drafter_worker.install_sharded_fc(drafter, ok, True, gather=object())
    assert "project_target_hidden" not in drafter.__dict__
    assert "_fc_shard" not in drafter.__dict__


# --------------------------------------------------------------------------
# Eight ranks, real fabric gather
# --------------------------------------------------------------------------
@torch.inference_mode()
def test_sharded_fc_tp8_matches_builtin():
    from mpi4py import MPI

    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    if comm.Get_size() != 8:
        pytest.skip("Requires eight MPI ranks on two NVLink-connected four-GPU nodes")
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Requires GB300")
    mapping = Mapping(world_size=8, tp_size=8, rank=rank, gpus_per_node=4)
    drafter = _drafter(seed=11)  # identical weights on every rank
    assert drafter_worker.install_sharded_fc(drafter, mapping, True) is None
    g = torch.Generator(device="cuda").manual_seed(5)
    for rows in (8, 1, 8, 3):
        x = (2.0 * torch.randn(rows, TAPPED * HIDDEN, generator=g, device="cuda")).to(
            torch.bfloat16
        )
        builtin = DFlashForCausalLM.project_target_hidden(drafter, x)
        out = drafter.project_target_hidden(x)
        assert_within_ulp(out, builtin, ELEMENT_ULP, RMS_ULP)
        gathered = comm.allgather(out.view(torch.int16).cpu().numpy())
        assert all((peer == gathered[0]).all() for peer in gathered), "ranks disagree"
    x = (2.0 * torch.randn(8, TAPPED * HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    eager = drafter.project_target_hidden(x)
    torch.cuda.synchronize()
    comm.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = drafter.project_target_hidden(x)
    for step in range(6):
        x.copy_((torch.randn(x.shape, generator=g, device="cuda") * (step + 1)).to(x.dtype))
        torch.cuda.synchronize()
        comm.Barrier()
        graph.replay()
        torch.cuda.synchronize()
        expected = drafter_worker.F.linear(x, drafter.fc.weight)
        assert_within_ulp(out, drafter.hidden_norm(expected), ELEMENT_ULP, RMS_ULP)
    assert eager.shape == out.shape
    del graph
    comm.Barrier()
    # A drafter weight reload: the loader builds a new fc (the same on every
    # rank) and the shell installs again; the next calls use the new weight.
    reloaded = _drafter(seed=12)
    drafter.fc, drafter.hidden_norm = reloaded.fc, reloaded.hidden_norm
    del reloaded
    assert drafter_worker.install_sharded_fc(drafter, mapping, True) is None
    for rows in (8, 2):
        x = (2.0 * torch.randn(rows, TAPPED * HIDDEN, generator=g, device="cuda")).to(
            torch.bfloat16
        )
        builtin = DFlashForCausalLM.project_target_hidden(drafter, x)
        out = drafter.project_target_hidden(x)
        assert_within_ulp(out, builtin, ELEMENT_ULP, RMS_ULP)
        gathered = comm.allgather(out.view(torch.int16).cpu().numpy())
        assert all((peer == gathered[0]).all() for peer in gathered), "ranks disagree"
    comm.Barrier()


def test_gemm_helper_uses_the_builtin_for_unlisted_shapes():
    x = torch.randn(8, 256, device="cuda").to(torch.bfloat16)
    w = torch.randn(64, 256, device="cuda").to(torch.bfloat16)
    assert torch.equal(drafter_worker._gemm(x, w, True), F.linear(x, w))
