# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model-owned finalize that ``ExternalCommMoEScheduler`` runs before the combine.

``do_finalize=False`` asks the backend for the unfinalized ``(gemm2_output,
expert_weights, expanded_idx_to_permuted_idx)`` triple so the model can apply a
per-``(token, expert)`` transform to each expert output before the top-k sum.
Every ``Communication.combine`` in the tree is typed for a dense per-token
tensor, so the triple cannot cross one -- it reaches ``.dim()`` on a ``list``
and dies. When a comm strategy is active the model's finalize therefore has to
run on the dispatched rows instead, and the combine reduces already-finalized
partials.

These tests run on CPU with a stubbed ``moe``: the seam is host-side control
flow, and the point is which object is handed to whom, in what dtype, with what
row count -- none of which needs a device. The four refusals matter as much as
the happy path, because each one guards a combination that would otherwise
produce a plausible-looking wrong answer rather than an error.
"""

import types
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.communication import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import ExternalCommMoEScheduler

HIDDEN = 5
TOP_K = 4
DISPATCHED_ROWS = 12


class _Backend:
    """Named only so the refusal messages have something to report."""


class _Comm:
    pass


def _scheduler(*, finalize_fn, register=True):
    moe = types.SimpleNamespace(backend=_Backend(), comm=_Comm())
    if register:
        moe.unfinalized_combine_fn = finalize_fn
    return ExternalCommMoEScheduler(moe)


def _triple(*, rows: int = DISPATCHED_ROWS, flattened: bool = False) -> list[torch.Tensor]:
    """What ``run_moe`` returns for ``do_finalize=False``.

    ``expanded_idx_to_permuted_idx`` covers ``num_tokens * top_k`` slots.
    FlashInfer returns it flattened; neither its leading dimension nor the
    expert-major ``gemm2_output`` height is necessarily the token count.
    """
    return [
        torch.randn(rows * TOP_K, HIDDEN),
        torch.empty(rows, TOP_K),  # the kernel leaves this buffer uninitialized
        torch.zeros((rows * TOP_K,) if flattened else (rows, TOP_K), dtype=torch.int32),
    ]


@pytest.mark.parametrize("flattened", [False, True], ids=["matrix", "flat-flashinfer"])
@pytest.mark.parametrize("rows", [0, 8, DISPATCHED_ROWS])
def test_the_finalize_runs_on_the_dispatched_rows(flattened: bool, rows: int) -> None:
    seen = {}

    def finalize(*, gemm2_output, expanded_idx_to_permuted_idx, routing_weights, num_tokens):
        seen.update(
            gemm2=gemm2_output,
            index=expanded_idx_to_permuted_idx,
            weights=routing_weights,
            num_tokens=num_tokens,
        )
        return torch.ones(num_tokens, HIDDEN, dtype=torch.float32)

    triple = _triple(rows=rows, flattened=flattened)
    scales = torch.rand(rows, TOP_K, dtype=torch.bfloat16)

    out = _scheduler(finalize_fn=finalize)._finalize_before_combine(
        triple, token_final_scales=scales, output_dtype=torch.bfloat16
    )

    assert seen["gemm2"] is triple[0]
    assert seen["index"] is triple[2]
    # The POST-dispatch weights, not the model's own: under attention DP this
    # rank holds rows for tokens it does not own, and only the dispatched copy
    # covers them.
    assert seen["weights"] is scales
    assert seen["num_tokens"] == rows
    assert out.shape == (rows, HIDDEN)


def test_the_result_is_cast_down_before_the_combine_not_after():
    """The one-sided combine region is sized from ``(hidden_size, act_dtype)``.

    The finalize accumulates in FP32; handing that to ``combine`` would need
    twice the symmetric-memory region the workspace reserved.
    """
    scheduler = _scheduler(
        finalize_fn=lambda **kwargs: torch.ones(kwargs["num_tokens"], HIDDEN, dtype=torch.float32)
    )
    out = scheduler._finalize_before_combine(
        _triple(),
        token_final_scales=torch.rand(DISPATCHED_ROWS, TOP_K),
        output_dtype=torch.bfloat16,
    )
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("register", [True, False], ids=["set-to-None", "never-declared"])
def test_a_model_that_registered_no_finalize_is_refused_by_name(register):
    """Not an ``AttributeError`` four frames down inside the comm layer.

    Every other model in the tree keeps ``do_finalize=False`` and a comm
    strategy apart, so this combination is unreachable for them today. If one
    ever reaches it, the message has to say which layer and which strategy.
    """
    scheduler = _scheduler(finalize_fn=None, register=register)
    with pytest.raises(NotImplementedError, match="unfinalized_combine_fn"):
        scheduler._finalize_before_combine(
            _triple(),
            token_final_scales=torch.rand(DISPATCHED_ROWS, TOP_K),
            output_dtype=torch.bfloat16,
        )


def test_a_backend_that_ignored_do_finalize_is_refused():
    """A dense tensor here means the backend already ran its own finalize.

    That finalize is the weighted sum *without* the model's per-expert
    transform. Running the model's finalize on top of it would apply the
    transform to the combined result, which is a different function.
    """
    scheduler = _scheduler(finalize_fn=lambda **kwargs: None)
    with pytest.raises(NotImplementedError, match="triple"):
        scheduler._finalize_before_combine(
            torch.randn(DISPATCHED_ROWS, HIDDEN),
            token_final_scales=torch.rand(DISPATCHED_ROWS, TOP_K),
            output_dtype=torch.bfloat16,
        )


def test_router_weights_folded_into_the_activations_are_refused():
    """``apply_router_weight_on_input`` leaves ``token_final_scales`` as ``None``.

    The weights are then already inside the expert inputs, so the finalize
    cannot apply them a second time and cannot recover them either.
    """
    scheduler = _scheduler(finalize_fn=lambda **kwargs: None)
    with pytest.raises(NotImplementedError, match="token_final_scales"):
        scheduler._finalize_before_combine(
            _triple(), token_final_scales=None, output_dtype=torch.bfloat16
        )


@pytest.mark.parametrize("flattened", [False, True], ids=["matrix", "flat-flashinfer"])
@pytest.mark.parametrize("shape", [(DISPATCHED_ROWS - 1, TOP_K), (DISPATCHED_ROWS, TOP_K - 1)])
def test_the_weights_and_the_index_map_have_to_cover_the_same_rows(
    flattened: bool, shape: tuple[int, int]
) -> None:
    """The two come from different places and must agree.

    ``token_final_scales`` comes back from ``comm.dispatch``; the index map
    comes out of the MoE kernel. A mismatch means one of them is the
    pre-dispatch copy, which would weight each token's experts with another
    token's routing weights.
    """
    scheduler = _scheduler(finalize_fn=lambda **kwargs: None)
    with pytest.raises(AssertionError, match="index map"):
        scheduler._finalize_before_combine(
            _triple(flattened=flattened),
            token_final_scales=torch.rand(shape),
            output_dtype=torch.bfloat16,
        )


def _onesided_scheduler(*, workspace_tensor=None):
    comm = MagicMock(spec=NVLinkOneSided)
    comm.get_combine_payload_tensor_in_workspace.return_value = workspace_tensor
    backend = MagicMock()
    backend.supports_moe_output_in_alltoall_workspace.return_value = True
    backend.input_requirement.onesided_workspace_dtype = None
    moe = types.SimpleNamespace(comm=comm, backend=backend, hidden_size=HIDDEN)
    return ExternalCommMoEScheduler(moe), comm


def test_the_combine_payload_is_not_claimed_to_be_in_the_workspace():
    """``do_finalize=False`` means the backend writes no combine payload.

    ``combine()`` passes ``payload_in_workspace`` to ``moe_a2a_combine`` as a
    flag, independent of the tensor argument, so a stale ``True`` makes the
    kernel reduce an unwritten workspace region and return garbage without
    raising. The finalize that runs before the combine produces a fresh
    tensor, which is staged like any other caller-owned payload.
    """
    scheduler, comm = _onesided_scheduler()
    moe_output, payload_in_workspace = scheduler._plan_onesided_workspace(
        all_rank_num_tokens=[4, 4], output_dtype=torch.bfloat16, do_finalize=False
    )
    assert moe_output is None
    assert payload_in_workspace is False
    assert not comm.get_combine_payload_tensor_in_workspace.called


def test_a_backend_that_does_finalize_still_gets_the_workspace_payload():
    """The opposite case, so the guard above cannot quietly disable the fast path."""
    sentinel = torch.zeros(8, HIDDEN)
    scheduler, _ = _onesided_scheduler(workspace_tensor=sentinel)
    moe_output, payload_in_workspace = scheduler._plan_onesided_workspace(
        all_rank_num_tokens=[4, 4], output_dtype=torch.bfloat16, do_finalize=True
    )
    assert moe_output is sentinel
    assert payload_in_workspace is True
