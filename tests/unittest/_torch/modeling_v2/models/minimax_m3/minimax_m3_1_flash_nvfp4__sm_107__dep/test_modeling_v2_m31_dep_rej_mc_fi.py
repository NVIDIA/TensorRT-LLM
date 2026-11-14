# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_REJ_MC_FI's switch line and contract: ``rej_mc_fi`` picks REJ_MC's ctas_per_row rule and declines inputs
outside the kernel's contract (the stock gather + FlashInfer call then run). The install into the rejection sampler is
``test_modeling_v2_m31_dep_sampler_overrides.py``'s. CPU only: the op is a recording stand-in."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        rej_mc_fi,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


class _RecordingOp:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return ("tokens", "num_tokens")


@pytest.fixture
def fake_ops(monkeypatch):
    op, slots = _RecordingOp(), _RecordingOp()
    monkeypatch.setattr(rej_mc_fi, "_STATE", {"op": op, "slots": slots, "ws_bytes": 512})
    monkeypatch.setattr(rej_mc_fi, "prebuild", lambda device=None: torch.zeros(1))
    monkeypatch.setattr(rej_mc_fi, "_sm_count", lambda device: 212)
    monkeypatch.setattr(rej_mc_fi, "_TAKEN", set())
    return op, slots


def test_the_engagement_line_names_the_switch(monkeypatch):
    monkeypatch.setattr(rej_mc_fi, "DEP_REJ_MC_FI", False)
    assert "TRTLLM_M31_DISABLE_DEP_REJ_MC_FI=1" in rej_mc_fi.engagement()
    monkeypatch.setattr(rej_mc_fi, "DEP_REJ_MC_FI", True)
    monkeypatch.setattr(rej_mc_fi, "_STATE", {"op": None, "slots": None, "ws_bytes": 0})
    assert rej_mc_fi.engagement().startswith("not taken")
    monkeypatch.setattr(rej_mc_fi, "_STATE", {"op": object(), "slots": object(), "ws_bytes": 512})
    assert rej_mc_fi.engagement().startswith("engaged")


@pytest.mark.parametrize(
    "batch, ctas",
    [(1, 32), (6, 32), (8, 16), (12, 16), (16, 8), (24, 8), (48, 4), (212, 1), (300, 1)],
)
def test_ctas_per_row_is_rej_mcs_rule(batch, ctas):
    assert rej_mc_fi.ctas_per_row(batch, 212) == ctas


def _inputs(batch=12, k=7, vocab=4096):
    return dict(
        draft_probs=torch.zeros(batch, k, vocab),
        draft_token_ids=torch.zeros(batch, k, dtype=torch.int32),
        target_probs=torch.zeros(batch, k + 1, vocab),
        seed=torch.zeros(batch, dtype=torch.int64),
        offset=torch.zeros(batch, dtype=torch.int64),
    )


def test_cpu_inputs_are_declined():
    assert rej_mc_fi.chain_spec_mc_fi(**_inputs()) is None
    x = _inputs()
    assert (
        rej_mc_fi.chain_spec_mc_fi_slots(
            torch.zeros(16, 7, 4096),
            torch.arange(12),
            7,
            x["draft_token_ids"],
            x["target_probs"],
            x["seed"],
            x["offset"],
        )
        is None
    )


def _as_cuda_tensors(monkeypatch):
    # The contract checks read only metadata; present CPU tensors as CUDA ones so the checks run past is_cuda.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))


def test_an_input_inside_the_contract_reaches_the_op(fake_ops, monkeypatch):
    op, slots = fake_ops
    _as_cuda_tensors(monkeypatch)
    x = _inputs()
    assert rej_mc_fi.chain_spec_mc_fi(**x) == ("tokens", "num_tokens")
    assert len(op.calls) == 1 and op.calls[0][-1] == 16
    buf = torch.zeros(16, 7, 4096)
    slot_ids = torch.arange(12)
    out = rej_mc_fi.chain_spec_mc_fi_slots(
        buf, slot_ids, 7, x["draft_token_ids"], x["target_probs"], x["seed"], x["offset"]
    )
    assert out == ("tokens", "num_tokens")
    assert slots.calls[0][0] is buf and slots.calls[0][1] is slot_ids and slots.calls[0][2] == 7
    assert slots.calls[0][-1] == 16


@pytest.mark.parametrize(
    "change",
    [
        "draft_fp16",
        "tokens_int64",
        "seed_int32",
        "seed_short",
        "vocab_odd",
        "batch_over_cap",
        "target_shape",
    ],
)
def test_inputs_outside_the_contract_are_declined(fake_ops, monkeypatch, change):
    op, _ = fake_ops
    _as_cuda_tensors(monkeypatch)
    batch = rej_mc_fi.MAX_BATCH + 1 if change == "batch_over_cap" else 12
    vocab = 4098 if change == "vocab_odd" else 4096
    x = _inputs(batch=batch, vocab=vocab)
    if change == "draft_fp16":
        x["draft_probs"] = x["draft_probs"].half()
    elif change == "tokens_int64":
        x["draft_token_ids"] = x["draft_token_ids"].long()
    elif change == "seed_int32":
        x["seed"] = x["seed"].int()
    elif change == "seed_short":
        x["seed"] = x["seed"][:-1]
    elif change == "target_shape":
        x["target_probs"] = x["target_probs"][:, :-1]
    assert rej_mc_fi.chain_spec_mc_fi(**x) is None
    assert op.calls == []
