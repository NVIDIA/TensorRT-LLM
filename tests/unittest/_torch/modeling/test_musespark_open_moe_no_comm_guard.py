# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MuseSparkOpenMoE.forward`` may only skip its own finalize when a comm ran.

The model asks its backend for the unfinalized ``(gemm2_output,
expert_weights, expanded_idx_to_permuted_idx)`` triple with
``do_finalize=False`` so it can apply a per-``(token, expert)`` RMSNorm before
the top-k sum. Two different things can come back as a plain ``Tensor``:

1. the wide-EP path, where ``ExternalCommMoEScheduler`` ran the model's own
   finalize on the dispatched rows and then combined across EP ranks -- already
   correct, nothing left to do; and
2. a backend that ignored ``do_finalize=False`` and finalized internally --
   the backend's plain weighted sum, **without** the per-expert norm.

Only the first is acceptable, and the only thing that tells them apart is
whether this layer had a communication strategy. Accepting the second silently
substitutes a different model that still produces fluent text, which is exactly
the failure the ``else`` branch's refusal exists to catch.

These run on CPU with a stubbed module built by ``__new__``: the seam is
host-side control flow -- which branch runs, on what evidence -- and needs no
device and no 62-layer decoder.
"""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE

NUM_TOKENS = 2
HIDDEN = 3


class _Nvtx:
    """``self.nvtx(name)`` is used as a context manager in ``forward``."""

    def __call__(self, _name):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _RoutingMethod:
    def __init__(self):
        self.cleared = 0

    def clear_routing_memo(self):
        self.cleared += 1

    def apply(self, _logits):
        raise AssertionError("routing_method.apply must not run on the scheduler-finalized path")


def _moe(*, comm, expert_output):
    """A ``MuseSparkOpenMoE`` with only what ``forward`` touches before the branch."""
    mod = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    routing = _RoutingMethod()

    # A plain SimpleNamespace will not do: ``forward`` CALLS ``self.experts``.
    class _Experts:
        def __init__(self):
            self.comm = comm
            self.routing_method = routing

        def __call__(self, *_args, **_kwargs):
            return expert_output

    mod.experts = _Experts()
    mod.nvtx = _Nvtx()
    mod.router_logits = lambda h: torch.zeros(h.shape[0], 4)
    mod.pooler = lambda h: h
    mod.pre_expert_norm = lambda h: h
    mod.finalize = lambda *a, **k: pytest.fail(
        "self.finalize must not run when the scheduler already finalized"
    )
    return mod, routing


def test_a_dense_result_with_no_comm_is_refused():
    """The defect: an unguarded ``isinstance(outputs, Tensor)`` accepted this.

    A backend that finalizes internally returns a Tensor and there is no comm
    anywhere. Taking the scheduler-finalized branch here would drop the
    per-expert RMSNorm and return the backend's plain weighted sum instead --
    a different model, and a fluent one, so nothing downstream would notice.
    """
    dense = torch.full((NUM_TOKENS, HIDDEN), 6.0)
    mod, _routing = _moe(comm=None, expert_output=dense)

    with pytest.raises(RuntimeError) as excinfo:
        mod.forward(torch.zeros(NUM_TOKENS, HIDDEN), attn_metadata=None)

    # It must fail as the "backend finalized internally" refusal, naming the
    # triple -- not as an AttributeError or a shape error further down.
    message = str(excinfo.value)
    assert "unfinalized" in message
    assert "Tensor" in message


def test_a_dense_result_with_a_comm_takes_the_scheduler_finalized_branch():
    """The legitimate wide-EP case still works, and is told apart by ``comm``.

    Proven through the row-count guard rather than by reaching the end of
    ``forward``: a Tensor whose row count is not the local token count can only
    raise that error if the scheduler-finalized branch was entered.
    """
    wrong_rows = torch.zeros(NUM_TOKENS + 5, HIDDEN)
    mod, routing = _moe(comm=object(), expert_output=wrong_rows)

    with pytest.raises(RuntimeError) as excinfo:
        mod.forward(torch.zeros(NUM_TOKENS, HIDDEN), attn_metadata=None)

    message = str(excinfo.value)
    assert "combine returned" in message
    assert str(NUM_TOKENS + 5) in message
    # The branch clears the memo before checking the row count.
    assert routing.cleared == 1


def test_the_triple_is_still_accepted_when_there_is_no_comm():
    """The ordinary aggregated path is untouched by the guard.

    ``comm is None`` plus a 3-tuple is the TP4/EP4 path every non-wide-EP run
    takes; it must reach the model's own finalize, not the refusal.
    """
    triple = (
        torch.zeros(NUM_TOKENS * 4, HIDDEN),
        torch.zeros(NUM_TOKENS, 4),
        torch.zeros(NUM_TOKENS, 4, dtype=torch.int32),
    )
    mod, routing = _moe(comm=None, expert_output=triple)
    routing.apply = lambda _logits: (None, torch.ones(NUM_TOKENS, 4))
    reached = {}

    def _finalize(gemm2_output, idx_map, scales, num_tokens):
        reached["called"] = (num_tokens, gemm2_output.shape)
        return torch.zeros(num_tokens, HIDDEN)

    mod.finalize = _finalize
    # Stop after the branch: everything past it is the cast/reduce/unpooler
    # tail, which this test is not about.
    with pytest.raises((AttributeError, TypeError, RuntimeError)):
        mod.forward(torch.zeros(NUM_TOKENS, HIDDEN), attn_metadata=None)

    assert reached["called"] == (NUM_TOKENS, torch.Size([NUM_TOKENS * 4, HIDDEN]))
