# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The heads under attention TP with the TP group's key exchange (as the FP8 target runs them).

* **Draft head** (`MTPLayer.shared_head`): a greedy draft pass keeps this rank's vocabulary
  shard of the logits (the worker takes the argmax through the exchange); advanced draft
  sampling, guided decoding, more rows than the exchange takes, a padded vocabulary or no
  exchange gather them through the logits processor.
* **Verify head** (the shell's `forward`): with the exchange the spec worker's `target_logits`
  forms the verified rows' logits (the shard for a greedy or temperature-only batch), and the
  draft layer sees the step's `spec_metadata`; without it the logits processor gathers them.

Stand-ins throughout; no kernel runs.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only


class _ShardHead:
    """A COLUMN-parallel `lm_head` stand-in recording each call's `gather_output`."""

    def __init__(self, padding_size: int = 0) -> None:
        self.tp_mode = M.TensorParallelMode.COLUMN
        self.gather_output = True
        self.padding_size = padding_size
        self.calls = []

    def __call__(self, x, gather_output=True):
        self.calls.append(gather_output)
        return "shard"


@pytest.mark.parametrize(
    "case",
    ["greedy", "rejection_sampling", "no_exchange", "guided", "rows_past_exchange", "padded_vocab"],
)
def test_draft_head_keeps_the_shard(monkeypatch, case):
    monkeypatch.setattr(M, "qwen4_exp_hc_inject_norm", lambda *a: (None, None))
    monkeypatch.setattr(M, "_hc_mixer", lambda *a: (torch.zeros(3, 8), None))
    core = SimpleNamespace(
        _mtp=SimpleNamespace(fhc_norm=None, fhc_first=None, fhc_up=None),
        eps=1e-6,
        hc_count=4,
        draft_head=None,
    )
    head = _ShardHead(padding_size=16 if case == "padded_vocab" else 0)
    worker = SimpleNamespace(key_exchange=SimpleNamespace(MAX_ROWS=8), guided_decoder=None)
    if case == "no_exchange":
        worker.key_exchange = None
    if case == "guided":
        worker.guided_decoder = object()
    if case == "rows_past_exchange":
        worker.key_exchange.MAX_ROWS = 2
    layer = M.MTPLayer(core, logits_processor=SimpleNamespace(forward=lambda *a: "gathered"))
    layer.worker = worker
    layer.spec_metadata = SimpleNamespace(
        wants_advanced_draft_sampling=case == "rejection_sampling"
    )
    sharded = case == "greedy"
    assert layer.shared_head(torch.zeros(3, 32), head, None, True) == (
        "shard" if sharded else "gathered"
    )
    assert head.calls == ([False] if sharded else [])


def test_draft_head_without_a_worker_gathers():
    """Before the shell hands the layer a worker (one rank, attention DP, no exchange)."""
    layer = M.MTPLayer(None, logits_processor=None)
    assert not layer._sharded_head(_ShardHead(), 1)


class _Core:
    """The core's step surface: `begin_step`, the forward and its output split, and no Lamport
    launch-index registry (`_lamport_steps`, which the shell's forward brackets the step with)."""

    _lamport_steps = None

    def __init__(self, final):
        self.final = final

    def begin_step(self, *args, **kwargs) -> None:
        pass

    def __call__(self, **kwargs):
        return None

    def split_spec_output(self, out):
        return self.final, self.final


class _Worker(torch.nn.Module):
    """The spec worker: `target_logits` records its rows; the step returns the logits it got."""

    def __init__(self, exchange: bool, calls: list):
        super().__init__()
        self.key_exchange = object() if exchange else None
        self.calls = calls

    def target_logits(self, rows, lm_head, logits_processor, md, spec_md, draft_model):
        self.calls.append(("worker", rows.clone()))
        return "shard"

    def forward(self, **kwargs):
        return kwargs["logits"]


@pytest.mark.parametrize("exchange", [True, False])
def test_verify_logits_go_through_the_worker_with_the_exchange(monkeypatch, exchange):
    calls = []
    cls = M.ModelingV2Qwen38FlashNextNvfp4Sm10xDefault
    shell = cls.__new__(cls)
    torch.nn.Module.__init__(shell)
    shell.model = _Core(torch.arange(12.0).reshape(3, 4))
    shell.spec_worker = _Worker(exchange, calls)
    shell.one_rank = False
    shell.lm_head = "head"
    shell.logits_processor = SimpleNamespace(
        forward=lambda rows, head, md, ctx: calls.append(("gather", rows.clone())) or "gathered"
    )
    layer = SimpleNamespace(spec_metadata=None)
    shell.draft_model = SimpleNamespace(mtp_layers=[layer])
    shell.vision_tower = None
    monkeypatch.setattr(M._vision, "multimodal_embeds", lambda tower, kwargs: None)
    spec_md = SimpleNamespace(gather_ids=torch.tensor([0, 2]))
    md = SimpleNamespace(padded_num_tokens=None)
    out = shell.forward(md, torch.zeros(3, dtype=torch.int32), None, spec_metadata=spec_md)
    assert out == ("shard" if exchange else "gathered")
    # The receiver gets the verified rows themselves (rows 0 and 2), not the first two.
    [(who, rows)] = calls
    assert who == ("worker" if exchange else "gather")
    assert torch.equal(rows, shell.model.final[spec_md.gather_ids])
    assert layer.spec_metadata is spec_md
