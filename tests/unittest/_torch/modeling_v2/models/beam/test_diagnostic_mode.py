# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic validation mode of the Beam checkpoint-layout components (CPU; no weights).

Ordinary execution must not hash, copy back or agree on the MoE routing across ranks; the
explicit diagnostic mode (``TRTLLM_MODELING_V2_VALIDATE``) must, and every rank has to run the
same mode because the agreement is a collective. These checks stub the router, the expert
dispatch and the collectives, so they pin which work runs in which mode and in what order; the
real-checkpoint TP4 refactor-integrity runs exercise the collectives themselves.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
    component as nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
    component_bf16 as bf16,
)

pytestmark = pytest.mark.cpu_only

BF16_CLS = bf16.BeamCheckpointLayoutBf16Sm103Tp4
NVFP4_CLS = nvfp4.BeamCheckpointLayoutNvfp4Sm103Tp4
T, HIDDEN, EXPERTS, LAYER = 3, 8, 4, 3


def _routing() -> dict:
    keys = (
        "logits",
        "scores",
        "selection_scores",
        "group_scores",
        "groups",
        "masked_selection_scores",
    )
    routing = {key: torch.zeros(T, EXPERTS) for key in keys}
    routing["ids"] = torch.tensor([[0, 1]] * T)
    routing["weights"] = torch.full((T, 2), 0.5)
    return routing


def _component(diagnostics: bool, agreements: list) -> SimpleNamespace:
    component = SimpleNamespace(diagnostics=diagnostics, device=torch.device("cpu"))
    component.route = lambda layer, h: _routing()
    component.all_reduce = lambda t: t
    component._bank = lambda layer, which: torch.zeros(EXPERTS, HIDDEN, HIDDEN)
    component.expert_bank = lambda layer: {}
    component._shared_fc1 = {LAYER: [True] * EXPERTS}

    def digests_agree(tensors):
        agreements.append(len(tensors))
        return ["digest"] * len(tensors)

    component._digests_agree = digests_agree
    return component


@pytest.mark.parametrize("cls", [BF16_CLS, NVFP4_CLS], ids=["bf16", "nvfp4"])
@pytest.mark.parametrize("diagnostics", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("captured", [False, True], ids=["no-capture", "capture"])
def test_moe_router_agreement_runs_only_in_diagnostic_mode(monkeypatch, cls, diagnostics, captured):
    agreements = []
    component = _component(diagnostics, agreements)

    def dispatch(h, ids, weights, provider):
        if cls is BF16_CLS:
            provider(0, h[:1])  # one expert call, so a capture has rows to record
        return torch.ones_like(h)

    monkeypatch.setattr(bf16.dispatch_combine, "beam_dispatch_combine", dispatch)
    monkeypatch.setattr(bf16, "cublas_mm", lambda a, b: a.float() @ b.float())
    monkeypatch.setattr(bf16, "swiglu_fp32", lambda g, u: {"silu": g, "product": g, "act": g})
    monkeypatch.setattr(bf16, "_rec", lambda capture, items: None)
    monkeypatch.setattr(nvfp4, "_rec", lambda capture, items: None)
    monkeypatch.setattr(nvfp4, "_moe_capture", lambda *args: {})
    capture = {} if captured else None
    h = torch.zeros(T, HIDDEN, dtype=torch.bfloat16)

    out = cls.moe(component, LAYER, h, capture)

    assert torch.equal(out, torch.ones_like(h))
    # The router input, ids and weights are agreed on in diagnostic mode only, capture or not.
    assert agreements == ([3] if diagnostics else [])
    if captured:
        assert ("moe.agreement_sha256" in capture) is diagnostics


class _Reduce:
    """A stand-in for the four-rank all-reduce: returns the sum the peers would produce."""

    def __init__(self, total: int):
        self.total = total
        self.calls = 0

    def __call__(self, flag: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.tensor([float(self.total)])


@pytest.mark.parametrize(
    "mine,ranks_on,ok",
    [(False, 0, True), (True, 4, True), (True, 3, False), (False, 1, False), (False, 2, False)],
)
def test_every_rank_runs_the_same_diagnostic_mode(mine, ranks_on, ok):
    reduce = _Reduce(ranks_on)
    component = SimpleNamespace(
        diagnostics=mine, device=torch.device("cpu"), rank=0, _allreduce=reduce
    )
    if ok:
        BF16_CLS._check_diagnostics_agree(component)
    else:
        with pytest.raises(bf16.LoadError, match=f"on for {ranks_on} of 4 ranks"):
            BF16_CLS._check_diagnostics_agree(component)
    assert reduce.calls == 1


@pytest.mark.parametrize("cls", [BF16_CLS, NVFP4_CLS], ids=["bf16", "nvfp4"])
def test_load_settles_the_mode_before_any_other_collective(cls, tmp_path):
    order = []
    component = SimpleNamespace()

    def mode_mismatch():
        order.append("mode")
        raise bf16.LoadError("rank 0: diagnostic validation is on for 1 of 4 ranks")

    component._check_diagnostics_agree = mode_mismatch
    component._agree = lambda ok: order.append("agree") or ok
    with pytest.raises(bf16.LoadError, match="diagnostic validation"):
        cls.load(component, str(tmp_path))
    assert order == ["mode"]


def test_router_disagreement_is_detected(monkeypatch):
    component = SimpleNamespace(device=torch.device("cpu"), mapping=None)
    tensors = (
        torch.ones(2, HIDDEN, dtype=torch.bfloat16),
        torch.tensor([[0, 1], [2, 3]]),
        torch.full((2, 2), 0.5),
    )

    def peers(differing_rank):
        def allgather(local, mapping, dim=0):
            rows = [local.clone() for _ in range(bf16.WORLD_SIZE)]
            if differing_rank is not None:
                rows[differing_rank][0] ^= 1
            return torch.cat(rows)

        return allgather

    monkeypatch.setattr(bf16, "allgather", peers(None))
    assert len(BF16_CLS._digests_agree(component, tensors)) == 3
    monkeypatch.setattr(bf16, "allgather", peers(2))
    with pytest.raises(RuntimeError, match="replicated MoE routing differs"):
        BF16_CLS._digests_agree(component, tensors)


class _Stop(Exception):
    pass


@pytest.mark.parametrize(
    "call_checked,diagnostics,rechecks",
    [(False, False, 1), (True, False, 0), (True, True, 1), (False, True, 1)],
    ids=["standalone", "full-model-call", "full-model-call-diagnostic", "standalone-diagnostic"],
)
def test_attention_rechecks_the_step_per_layer_only_when_needed(
    monkeypatch, call_checked, diagnostics, rechecks
):
    calls = []

    def stop(md, layer):
        raise _Stop()

    monkeypatch.setattr(bf16._attention, "check_batch_metadata", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(bf16._attention, "step_args", stop)
    component = SimpleNamespace(
        diagnostics=diagnostics,
        hidden=HIDDEN,
        device=torch.device("cpu"),
        max_tokens=16,
        local_kv_heads=2,
        head_dim=4,
        kv_cache_dtype=None,
        _check_usable=lambda layer=None: None,
    )
    h = torch.zeros(T, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(_Stop):
        BF16_CLS.attention(component, LAYER, h, torch.arange(T), None, call_checked=call_checked)
    assert len(calls) == rechecks


def test_full_model_validates_once_and_marks_every_block_call_checked(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (  # noqa: E501
        model as full,
    )

    blocks = []
    checks = []
    model = SimpleNamespace(
        schedule=SimpleNamespace(is_sliding=lambda layer: True, is_sparse=lambda layer: True),
        window=512,
        _check_usable=lambda layer=None: None,
        _select_rows=lambda logit_rows, spans, num_contexts: [T - 1],
        embed=lambda ids: torch.zeros(T, HIDDEN),
        final_norm=lambda x: x,
        head=lambda x: x,
    )

    def check_call(input_ids, position_ids, md, **kwargs):
        checks.append(kwargs)
        return input_ids, position_ids, [(0, T)]

    def block(layer, x, positions, md, capture, **kwargs):
        blocks.append((layer, kwargs.get("call_checked")))
        return x

    model.check_call = check_call
    model.block = block
    md = SimpleNamespace(num_contexts=1)
    full.BeamFullModelNvfp4Sm103Tp4.forward(
        model, torch.zeros(T, dtype=torch.int64), torch.arange(T), md
    )
    assert len(checks) == 1
    assert blocks == [(layer, True) for layer in range(full.NUM_LAYERS)]


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True), ("on", True)])
def test_diagnostic_mode_is_read_once_from_the_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(MODELING_V2_VALIDATE_ENV, raising=False)
    else:
        monkeypatch.setenv(MODELING_V2_VALIDATE_ENV, value)
    component = SimpleNamespace()
    BF16_CLS._init_diagnostics(component)
    assert component.diagnostics is expected
    monkeypatch.setenv(MODELING_V2_VALIDATE_ENV, "0" if expected else "1")
    assert component.diagnostics is expected
