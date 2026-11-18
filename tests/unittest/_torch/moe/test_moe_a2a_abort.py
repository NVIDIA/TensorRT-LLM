# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run-time rank mask and abort of the fence MoE All-to-All ops.

Every logical peer lives in one allocation on one GPU, and the tests issue the ops of each rank in
turn, so they check masking, polling and the order of the ranks' steps only.
"""

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from _torch.moe.moe_a2a_test_utils import CAPACITY, HIDDEN, a2a_combine, a2a_dispatch, a2a_workspace

import tensorrt_llm  # noqa: F401
from tensorrt_llm.bindings import internal

# The abort state holds ``requested`` and ``status``, then the flag value awaited from each peer.
_ABORT_WORDS = int(internal.thop.MOE_A2A_ABORT_STATE_WORDS)


def _offset(meta: torch.Tensor, name: str) -> int:
    return int(meta[int(getattr(internal.thop, f"MOE_A2A_{name}_OFFSET_INDEX"))])


def _word(workspace: torch.Tensor, offset: int, row: int = 0) -> torch.Tensor:
    return workspace[row, offset : offset + 4].view(torch.int32)


def _mask_words(mask: int) -> torch.Tensor:
    """Return the rank mask as the kernels' four little-endian uint64 words."""
    return torch.tensor([(mask >> (64 * w)) & (2**64 - 1) for w in range(4)], dtype=torch.uint64)


def _rank_mask_controls(mask: int) -> tuple[dict, torch.Tensor]:
    """Return the rank-mask controls of a dispatch or combine and their pinned abort state."""
    abort = torch.zeros(_ABORT_WORDS, dtype=torch.int32).pin_memory()
    words = _mask_words(mask).cuda()
    return {"enable_rank_mask": True, "active_rank_mask": words, "abort_state": abort}, abort


def _next_parity(workspace: torch.Tensor, meta: torch.Tensor, rank: int = 0) -> tuple[int, int]:
    """Return the flag value and receive-count parity of ``rank``'s next round."""
    expected_flag = _word(workspace, _offset(meta, "FLAG_VAL"), row=rank).item() + 1
    return expected_flag, ((expected_flag - 1) >> 1) & 1


def _finish(abort: torch.Tensor, failure: str, timeout_s: float = 10) -> None:
    """Wait for the queued work. At the deadline, release it through the abort state and fail."""
    done = torch.cuda.Event()
    done.record()
    deadline = time.monotonic() + timeout_s
    while not done.query():
        if time.monotonic() > deadline:
            abort[0] = 1
            torch.cuda.synchronize()
            pytest.fail(failure)
        time.sleep(0.01)


def test_cancel_and_exclusion_on_replay() -> None:
    """One captured graph observes the abort state and the mask written before each replay."""
    workspace, meta, experts, payload = a2a_workspace(2)
    controls, abort = _rank_mask_controls(3)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        received, offset, _ = a2a_dispatch(workspace, meta, experts, controls, payload)
        output = a2a_combine(workspace, meta, received[0], offset, controls)

    # Peer 1 never signals. A pending abort request releases the dispatch's wait, and the combine of
    # the incomplete round takes only this rank's own rows.
    abort[0] = 1
    graph.replay()
    _finish(abort, "the abort request did not release the dispatch's wait")
    assert abort[:2].tolist() == [1, 1]
    assert _word(workspace, _offset(meta, "INCOMPLETE_ROUND")).item() == 1
    torch.testing.assert_close(output, payload, rtol=0, atol=0)

    # Until the workspace is reset, rank 0 publishes round flags only to itself, so peer 1 never
    # combines outputs of the incomplete round. Withdrawing the abort request does not change that.
    def peer_flags() -> list[int]:
        """Return rank 0's dispatch and combine flags in peer 1's row."""
        regions = ("DISPATCH_COMPLETION_FLAGS", "COMBINE_COMPLETION_FLAGS")
        return [_word(workspace, _offset(meta, region), row=1).item() for region in regions]

    published = peer_flags()
    assert published == [1, 0]
    # Peer 1 takes part in the next round with no rows for rank 0.
    expected_flag, parity = _next_parity(workspace, meta)
    _word(workspace, _offset(meta, "RECV_COUNTERS") + 4 * (parity * 2 + 1)).fill_(0)
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(expected_flag)
    _word(workspace, _offset(meta, "COMBINE_COMPLETION_FLAGS") + 4).fill_(expected_flag + 1)
    abort.zero_()
    graph.replay()
    _finish(abort, "the round after the incomplete one did not complete with peer 1's flags")
    assert peer_flags() == published

    # Rearmed with peer 1 excluded, the same graph completes without it.
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 0, 2, barrier=False)
    abort.zero_()
    controls["active_rank_mask"].copy_(_mask_words(1))
    for _ in range(2):
        graph.replay()
        _finish(abort, "the replay with peer 1 excluded waited for it")
        torch.testing.assert_close(output, payload, rtol=0, atol=0)
        assert torch.all(received[1][0] == 0)
    assert abort[:2].tolist() == [0, 0]


@pytest.mark.parametrize("ep_size, rank", [(2, 0), (72, 70)], ids=["first_word", "second_word"])
def test_no_peer_writes_after_incomplete_round(ep_size: int, rank: int) -> None:
    """Until the workspace reset, a dispatch writes nothing into rows a peer may still read."""
    workspace, meta, experts, payload = a2a_workspace(ep_size)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, ep_size, barrier=False)
    controls, abort = _rank_mask_controls((1 << ep_size) - 1)
    abort[0] = 1  # No peer signals, so the round stays incomplete.
    a2a_dispatch(workspace, meta, experts, controls, payload, rank=rank)
    _finish(abort, "the abort request did not release the dispatch's wait")
    assert _word(workspace, _offset(meta, "INCOMPLETE_ROUND"), row=rank).item() == 1

    abort.zero_()
    peers = torch.cat([workspace[:rank], workspace[rank + 1 :]])
    # Route the tokens to peers of both mask words.
    experts.copy_(
        torch.tensor([0, 65, 71, 64, 3, 70, 69, 1], dtype=torch.int32).view(-1, 1) % ep_size
    )
    a2a_dispatch(workspace, meta, experts, controls, payload, rank=rank)
    _finish(abort, "the dispatch after the incomplete round waited for a peer")
    assert torch.equal(torch.cat([workspace[:rank], workspace[rank + 1 :]]), peers)


@pytest.mark.parametrize(
    "ep_size, rank, peer", [(2, 0, 1), (72, 34, 2)], ids=["two_ranks", "rank_34_of_72"]
)
def test_aborted_dispatch_uses_no_rows_of_unconfirmed_peers(
    ep_size: int, rank: int, peer: int
) -> None:
    """An aborted round uses no rows of a peer whose completion flag it did not see.

    The peer writes its receive count but never its flag, and again after the abort. Rank 34 of 72
    waits on peer 2 and then on itself in the same lane, and keeps its own rows.
    """
    workspace, meta, experts, payload = a2a_workspace(ep_size)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, rank, ep_size, barrier=False)
    experts.fill_(rank)
    controls, abort = _rank_mask_controls((1 << ep_size) - 1)
    _, parity = _next_parity(workspace, meta, rank)
    counts = _offset(meta, "RECV_COUNTERS") + 4 * parity * ep_size
    count = _word(workspace, counts + 4 * peer, row=rank)
    count.fill_(CAPACITY)
    abort[0] = 1
    received, offset, _ = a2a_dispatch(workspace, meta, experts, controls, payload, rank=rank)
    _finish(abort, "the abort request did not release the dispatch's wait")
    assert count.item() == 0
    assert _word(workspace, counts + 4 * rank, row=rank).item() == CAPACITY

    count.fill_(CAPACITY)
    expert_ids = received[1]
    expert_ids[peer].fill_(12345)  # an expert id that no rank owns
    torch.ops.trtllm.moe_a2a_sanitize_expert_ids(expert_ids, workspace, meta, rank, -1)
    assert torch.all(expert_ids[peer] == -1)
    assert torch.equal(expert_ids[rank], experts)

    # The rows of the peer in this rank's combine input region, which the peer reads.
    stride = HIDDEN * payload.element_size()
    first = offset + peer * CAPACITY * stride
    peer_rows = workspace[rank, first : first + CAPACITY * stride]
    peer_rows.fill_(0x7F)
    output = a2a_combine(workspace, meta, received[0], offset, controls, rank=rank)
    _finish(abort, "the combine of the incomplete round waited for the peer")
    assert torch.all(peer_rows == 0x7F)
    torch.testing.assert_close(output, payload, rtol=0, atol=0)


def _check_incomplete_round(workspace, meta, abort, rank=0) -> None:
    """Run the incomplete-round check of ``rank`` that follows a forward."""
    torch.ops.trtllm.moe_a2a_check_incomplete_round(workspace, meta, rank, abort)
    _finish(abort, f"rank {rank}'s incomplete-round check did not finish")


@pytest.mark.parametrize("peer_aborts", [False, True], ids=["healthy", "peer_aborts"])
def test_incomplete_round_check_cancels_on_overwritten_peer_rows(peer_aborts: bool) -> None:
    """The check after a combine that read rows a peer overwrote after an aborted dispatch records a cancel.

    Rank 0's tokens go to rank 1, whose MoE output is its combine input region, as with zero-copy
    combine. Rank 1 completes its round before rank 0 combines it. With ``peer_aborts``, rank 1
    then aborts its next dispatch and overwrites that region like its next MoE. The combine
    records nothing; the check that follows the forward records the cancel.
    """
    workspace, meta, experts, payload = a2a_workspace(2)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 1, 2, barrier=False)
    controls, abort = _rank_mask_controls(3)
    peer_controls, peer_abort = _rank_mask_controls(3)
    experts.fill_(1)
    # Rank 1's dispatch flag for rank 0, and rank 0's combine flag for rank 1.
    dispatch_flag, _ = _next_parity(workspace, meta)
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(dispatch_flag)
    _word(workspace, _offset(meta, "COMBINE_COMPLETION_FLAGS"), row=1).fill_(dispatch_flag + 1)
    received, offset, _ = a2a_dispatch(workspace, meta, experts, controls, payload)
    peer_received, _, _ = a2a_dispatch(workspace, meta, experts, peer_controls, payload, rank=1)
    moe_output = torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
        workspace, meta, 1, 2, CAPACITY, payload.dtype, HIDDEN
    ).view(2, CAPACITY, HIDDEN)
    moe_output.copy_(peer_received[0])
    a2a_combine(workspace, meta, moe_output, offset, peer_controls, rank=1)
    if peer_aborts:
        peer_abort[0] = 1
        a2a_dispatch(workspace, meta, experts, peer_controls, payload, rank=1)
        moe_output.fill_(-1)
    _finish(peer_abort, "rank 1's round waited for rank 0")
    output = a2a_combine(workspace, meta, received[0], offset, controls)
    _finish(abort, "rank 0's combine waited for rank 1")
    expected = torch.full_like(payload, -1) if peer_aborts else payload
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert abort[:2].tolist() == [0, 0]
    _check_incomplete_round(workspace, meta, abort)
    assert abort[:2].tolist() == [0, int(peer_aborts)]


def test_own_round_value_keeps_peer_rows() -> None:
    """A word holding the round's own combine value keeps peer rows valid in the sanitize and the prepare.

    A peer whose combine of this round aborted stores that value with its next dispatch. The combine then
    exchanges with every peer, so the rows it publishes must hold the peers' tokens, and the check
    after it records a cancel.
    """
    workspace, meta, experts, payload = a2a_workspace(2)
    torch.ops.trtllm.moe_a2a_initialize(workspace, meta, 1, 2, barrier=False)
    controls, _ = _rank_mask_controls(3)
    peer_controls, peer_abort = _rank_mask_controls(3)
    experts.fill_(1)
    # Rank 1's dispatch flag for rank 0, and rank 0's combine flag for rank 1.
    dispatch_flag, _ = _next_parity(workspace, meta, rank=1)
    _word(workspace, _offset(meta, "DISPATCH_COMPLETION_FLAGS") + 4).fill_(dispatch_flag)
    _word(workspace, _offset(meta, "COMBINE_COMPLETION_FLAGS"), row=1).fill_(dispatch_flag + 1)
    a2a_dispatch(workspace, meta, experts, controls, payload)
    peer_received, offset, _ = a2a_dispatch(
        workspace, meta, experts, peer_controls, payload, rank=1
    )
    # Rank 0's combine of this round aborted, and its next dispatch marked rank 1.
    _word(workspace, _offset(meta, "INCOMPLETE_ROUND"), row=1).fill_(dispatch_flag + 2)
    expert_ids = peer_received[1].clone()
    torch.ops.trtllm.moe_a2a_sanitize_expert_ids(expert_ids, workspace, meta, 1, -1)
    torch.testing.assert_close(expert_ids, peer_received[1], rtol=0, atol=0)
    a2a_combine(workspace, meta, peer_received[0].clone(), offset, peer_controls, rank=1)
    _finish(peer_abort, "rank 1's combine waited for rank 0")
    rows = torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
        workspace, meta, 1, 2, CAPACITY, payload.dtype, HIDDEN
    ).view(2, CAPACITY, HIDDEN)
    torch.testing.assert_close(rows[0], peer_received[0][0], rtol=0, atol=0)
    # The combine published its readiness to rank 0, so it exchanged with every peer.
    ready = _word(workspace, _offset(meta, "COMBINE_COMPLETION_FLAGS") + 4)
    assert ready.item() == dispatch_flag + 1
    assert peer_abort[:2].tolist() == [0, 0]
    _check_incomplete_round(workspace, meta, peer_abort, rank=1)
    assert peer_abort[:2].tolist() == [0, 1]


def test_abort_state_must_be_pinned() -> None:
    """The kernels read the abort state at its host address, so the ops refuse pageable memory."""
    workspace, meta, experts, payload = a2a_workspace(2)
    controls, _ = _rank_mask_controls(3)
    controls["abort_state"] = torch.zeros(_ABORT_WORDS, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="pinned"):
        a2a_dispatch(workspace, meta, experts, controls, payload)


def test_incomplete_dispatch_marks_active_peers() -> None:
    """A dispatch whose round is incomplete marks the rounds of the active peers incomplete."""
    workspace, meta, experts, payload = a2a_workspace(3)
    controls, abort = _rank_mask_controls(0b011)  # rank 2 is inactive
    incomplete_round = _offset(meta, "INCOMPLETE_ROUND")
    # The incomplete-round value of a peer's aborted dispatch with flag value 5.
    _word(workspace, incomplete_round).fill_(5)
    a2a_dispatch(workspace, meta, experts, controls, payload)
    _finish(abort, "the dispatch of an incomplete round waited for a peer")
    assert [_word(workspace, incomplete_round, row=r).item() for r in range(3)] == [5, 5, 0]


def _fence_timeout() -> None:
    """Check that a fence wait on a silent peer still fails at the timeout."""
    torch.ops.trtllm.moe_a2a_set_timeout(1)
    workspace, meta, experts, payload = a2a_workspace(2)
    controls, _ = _rank_mask_controls(3)
    a2a_dispatch(workspace, meta, experts, controls, payload)
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        # Exit at once, flushing the device printf C stdio holds on a pipe.
        print("fence timeout trapped", flush=True)
        ctypes.CDLL(None).fflush(None)
        os._exit(0)
    raise AssertionError("dispatch completed without its peer")


def test_fence_timeout_still_traps() -> None:
    # The child traps on purpose, in a fresh interpreter without the launcher's environment.
    launcher_prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher_prefixes)}
    # The child imports the shared helpers of the unit tests, as pytest makes them importable.
    unittest_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (unittest_root, env.get("PYTHONPATH")) if p)
    result = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # The kernel's own report, not just any device error.
    assert "dispatch: rank 0 timed out waiting for rank 1 flag=" in result.stdout, result.stdout
    assert "fence timeout trapped" in result.stdout, result.stdout


if __name__ == "__main__":
    _fence_timeout()
