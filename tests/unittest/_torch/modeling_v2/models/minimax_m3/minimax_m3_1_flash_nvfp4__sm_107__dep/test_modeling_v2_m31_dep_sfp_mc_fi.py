# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_SFP_MC_FI's switch line and contract: ``sfp_mc_fi.sampling_from_probs_mc_fi`` picks REJ_MC's ctas_per_row rule,
passes inputs inside the kernel's contract to the op and declines (and logs once) the others. The install into the
sampler is ``test_modeling_v2_m31_dep_sampler_overrides.py``'s. CPU only: the op is a recording stand-in."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        sfp_mc_fi,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


class _RecordingOp:
    def __init__(self, result="tokens"):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


@pytest.fixture
def log_lines(monkeypatch):
    lines = []
    monkeypatch.setattr(sfp_mc_fi.logger, "info", lines.append)
    monkeypatch.setattr(sfp_mc_fi, "_DECLINED", set())
    return lines


def _not_taken(lines):
    return [line for line in lines if "SFP_MC_FI not taken" in line]


@pytest.fixture
def fake_op(monkeypatch, log_lines):
    op = _RecordingOp()
    monkeypatch.setattr(sfp_mc_fi, "_STATE", {"op": op, "ws_bytes": 528})
    monkeypatch.setattr(sfp_mc_fi, "prebuild", lambda device=None: torch.zeros(1))
    monkeypatch.setattr(sfp_mc_fi, "_sm_count", lambda device: 212)
    monkeypatch.setattr(sfp_mc_fi, "_TAKEN", set())
    return op


def test_the_engagement_line_names_the_switch(monkeypatch):
    monkeypatch.setattr(sfp_mc_fi, "DEP_SFP_MC_FI", False)
    assert "TRTLLM_M31_DISABLE_DEP_SFP_MC_FI=1" in sfp_mc_fi.engagement()
    monkeypatch.setattr(sfp_mc_fi, "DEP_SFP_MC_FI", True)
    monkeypatch.setattr(sfp_mc_fi, "_STATE", {"op": None, "ws_bytes": 0})
    assert sfp_mc_fi.engagement().startswith("not taken")
    monkeypatch.setattr(sfp_mc_fi, "_STATE", {"op": object(), "ws_bytes": 528})
    assert sfp_mc_fi.engagement().startswith("engaged")


@pytest.mark.parametrize(
    "rows, ctas", [(1, 64), (8, 16), (12, 16), (16, 8), (24, 8), (48, 4), (212, 1), (300, 1)]
)
def test_ctas_per_row_is_rej_mcs_rule(rows, ctas):
    assert sfp_mc_fi.ctas_per_row(rows, 212) == ctas


def _inputs(rows=12, vocab=4096):
    return (
        torch.zeros(rows, vocab),
        torch.zeros(rows, dtype=torch.int64),
        torch.zeros(rows, dtype=torch.int64),
    )


def test_cpu_inputs_are_declined(fake_op, log_lines):
    assert sfp_mc_fi.sampling_from_probs_mc_fi(*_inputs()) is None
    assert fake_op.calls == []
    assert len(_not_taken(log_lines)) == 1
    assert "probs not on a CUDA device (first: cpu)" in log_lines[0]


def _as_cuda_tensors(monkeypatch):
    # The contract checks read only metadata; present CPU tensors as CUDA ones so the checks run past is_cuda.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))


def test_an_input_inside_the_contract_reaches_the_op(fake_op, monkeypatch, log_lines):
    _as_cuda_tensors(monkeypatch)
    probs, seed, offset = _inputs()
    assert sfp_mc_fi.sampling_from_probs_mc_fi(probs, seed, offset) == "tokens"
    assert len(fake_op.calls) == 1
    args = fake_op.calls[0][0]
    assert args[0] is probs and args[1] is seed and args[2] is offset and args[-1] == 16
    assert _not_taken(log_lines) == []


def test_a_one_entry_seed_and_offset_reach_the_op(fake_op, monkeypatch, log_lines):
    # FlashInfer's sampling_from_probs broadcasts a one-entry seed / offset tensor to every row.
    _as_cuda_tensors(monkeypatch)
    probs, seed, offset = _inputs()
    assert sfp_mc_fi.sampling_from_probs_mc_fi(probs, seed[:1], offset[:1]) == "tokens"
    assert len(fake_op.calls) == 1
    assert _not_taken(log_lines) == []


def test_a_row_strided_view_reaches_the_op(fake_op, monkeypatch, log_lines):
    _as_cuda_tensors(monkeypatch)
    probs = torch.zeros(12, 4100)[:, :4096]
    _, seed, offset = _inputs()
    assert sfp_mc_fi.sampling_from_probs_mc_fi(probs, seed, offset) == "tokens"
    assert _not_taken(log_lines) == []


@pytest.mark.parametrize(
    "change, reason",
    [
        ("probs_fp16", "probs not float32 (first: torch.float16)"),
        ("probs_3d", "probs not 2-D (first: (12, 1, 4096))"),
        ("probs_transposed", "probs strides outside the contract (first: (1, 12))"),
        ("row_stride_odd", "probs strides outside the contract (first: (4098, 1))"),
        ("vocab_odd", "vocab not a multiple of 4 in 4..262144 (first: vocab 4098)"),
        ("rows_over_cap", "rows outside 1..48 (first: 49 rows)"),
        ("seed_int32", "seed not int64 (first: torch.int32)"),
        ("seed_empty", "seed not one entry or one per row (first: (0,))"),
        ("seed_short", "seed not one entry or one per row (first: (5,))"),
        ("offset_2d", "offset not one entry or one per row (first: (12, 1))"),
        ("offset_strided", "offset not contiguous (first: (2,))"),
    ],
)
def test_inputs_outside_the_contract_are_declined(fake_op, monkeypatch, log_lines, change, reason):
    _as_cuda_tensors(monkeypatch)
    rows = sfp_mc_fi.MAX_ROWS + 1 if change == "rows_over_cap" else 12
    vocab = 4098 if change == "vocab_odd" else 4096
    probs, seed, offset = _inputs(rows=rows, vocab=vocab)
    if change == "probs_fp16":
        probs = probs.half()
    elif change == "probs_3d":
        probs = probs.view(rows, 1, vocab)
    elif change == "probs_transposed":
        probs = torch.zeros(vocab, rows).t()
    elif change == "row_stride_odd":
        probs = torch.zeros(rows, vocab + 2)[:, :vocab]
    elif change == "vocab_odd":
        # A row stride inside the contract, so the vocab check is the one that declines.
        probs = torch.zeros(rows, vocab + 2)[:, :vocab]
    elif change == "seed_int32":
        seed = seed.int()
    elif change == "seed_empty":
        seed = seed[:0]
    elif change == "seed_short":
        seed = seed[:5]
    elif change == "offset_2d":
        offset = offset.view(rows, 1)
    elif change == "offset_strided":
        offset = torch.zeros(2 * rows, dtype=torch.int64)[::2]
    assert sfp_mc_fi.sampling_from_probs_mc_fi(probs, seed, offset) is None
    assert fake_op.calls == []
    assert log_lines == [
        f"[M3.1 DEP] SFP_MC_FI not taken: {reason}; stock FlashInfer sampling_from_probs"
    ]


def test_each_decline_reason_logs_once(fake_op, monkeypatch, log_lines):
    _as_cuda_tensors(monkeypatch)
    for rows in (49, 64, 49):
        assert sfp_mc_fi.sampling_from_probs_mc_fi(*_inputs(rows=rows)) is None
    probs, seed, offset = _inputs()
    assert sfp_mc_fi.sampling_from_probs_mc_fi(probs.half(), seed, offset) is None
    assert sfp_mc_fi.sampling_from_probs_mc_fi(probs, seed, offset) == "tokens"
    declined = _not_taken(log_lines)
    assert len(declined) == 2
    assert "rows outside 1..48 (first: 49 rows)" in declined[0]
    assert "probs not float32" in declined[1]


def test_a_non_tensor_seed_is_declined_and_named(fake_op, monkeypatch, log_lines):
    _as_cuda_tensors(monkeypatch)
    probs, _, offset = _inputs()
    assert sfp_mc_fi.sampling_from_probs_mc_fi(probs, 7, offset) is None
    assert fake_op.calls == []
    assert len(log_lines) == 1
    assert "not all tensors (first: seed is int)" in log_lines[0]


def test_a_missing_workspace_is_declined_and_logged(fake_op, monkeypatch, log_lines):
    _as_cuda_tensors(monkeypatch)
    monkeypatch.setattr(sfp_mc_fi, "prebuild", lambda device=None: None)
    for _ in range(2):
        assert sfp_mc_fi.sampling_from_probs_mc_fi(*_inputs()) is None
    assert fake_op.calls == []
    assert log_lines == [
        "[M3.1 DEP] SFP_MC_FI not taken: no workspace (first: cpu, first needed inside a graph capture); "
        "stock FlashInfer sampling_from_probs"
    ]


def test_a_build_without_the_op_is_declined_and_logged(monkeypatch, log_lines):
    _as_cuda_tensors(monkeypatch)
    monkeypatch.setattr(sfp_mc_fi, "_STATE", {"op": None, "ws_bytes": 0})
    assert sfp_mc_fi.sampling_from_probs_mc_fi(*_inputs()) is None
    assert len(log_lines) == 1
    assert "not taken: the op is not built" in log_lines[0]
