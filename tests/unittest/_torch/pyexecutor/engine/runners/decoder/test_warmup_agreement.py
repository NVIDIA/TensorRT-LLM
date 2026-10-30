# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from _torch.pyexecutor.engine_test_utils import _agreement


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "world_size,dwdp_size,has_batch,expected",
    [
        (1, 0, False, False),
        # Every rank in the forward group lacks a batch. The allgather proves
        # nobody is left inside the collective, so skipping strands no one --
        # this is the shape a max_batch_size=1 config simply cannot build.
        (2, 0, False, False),
        # DWDP peers live outside `self.dist`, so unanimity cannot be observed
        # and the shape stays fatal.
        (1, 4, False, None),
        (2, 0, True, True),
    ],
)
def test_warmup_batch_policy(
    world_size: int, dwdp_size: int, has_batch: bool, expected: bool | None
) -> None:
    agreement = _agreement(world_size=world_size, dwdp_size=dwdp_size)
    batch = object() if has_batch else None
    if expected is None:
        with pytest.raises(RuntimeError, match="cannot skip the shape"):
            agreement.should_run_warmup_batch(batch, 128, "general")
    else:
        assert agreement.should_run_warmup_batch(batch, 128, "general") is expected


@pytest.mark.cpu_only
def test_symmetric_absence_skips_instead_of_failing_the_world() -> None:
    """The regression this fix exists for.

    `max_batch_size=1` cannot build the mixed context+generation attention
    shape, on any rank, at any KV cache size. Every rank reaches the same
    verdict from configuration alone, so the group agrees and skips. Failing
    here took down ten multi-GPU tests that had always skipped this shape.
    """
    agreement = _agreement(world_size=2, tp_size=2, peer_values=(0,))

    assert (
        agreement.should_run_warmup_batch(None, 2, "attention, num_tokens=2, num_gen_requests=1")
        is False
    )


@pytest.mark.cpu_only
def test_group_disagreement_still_fails_with_the_ranks_named() -> None:
    """Asymmetric capacity is the deadlock the agreement exists to catch."""
    agreement = _agreement(world_size=3, peer_values=(1, 0))

    with pytest.raises(RuntimeError) as excinfo:
        agreement.should_run_warmup_batch(None, 128, "general")

    message = str(excinfo.value)
    assert "rank(s) [0, 2]" in message
    assert "would deadlock" in message


@pytest.mark.cpu_only
def test_pipeline_only_world_agrees_without_a_tp_group() -> None:
    """TP=1 with pipeline peers is where the TP-only check gathered nothing.

    Three of the ten failures were disaggregated context servers at
    `tensor_parallel_size: 1, pipeline_parallel_size: 2`.
    """
    agreement = _agreement(world_size=2, tp_size=1, peer_values=(0,))

    assert agreement.should_run_warmup_batch(None, 128, "general") is False


@pytest.mark.cpu_only
def test_agreement_group_is_absent_for_dwdp() -> None:
    agreement = _agreement(world_size=4, dwdp_size=4)

    assert agreement.allgather() is None


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "world_size,peer_values,flag,expected",
    [
        (1, None, True, True),
        (2, (1,), True, True),
        # One pipeline stage owns the guided decoder and opts out; the group
        # follows it rather than letting the others enter alone.
        (2, (0,), True, False),
        (2, (1,), False, False),
    ],
)
def test_phase_entry_flag_is_agreed(
    world_size: int, peer_values: tuple[int, ...] | None, flag: bool, expected: bool
) -> None:
    agreement = _agreement(world_size=world_size, peer_values=peer_values)

    assert agreement.agree_flag(flag) is expected


@pytest.mark.cpu_only
def test_shape_list_narrows_to_what_every_rank_proposed() -> None:
    """Attention-DP gives ranks different capacities and so different shapes."""
    agreement = _agreement(world_size=2)
    agreement.dist.allgather = lambda value: [value, [[1, 0], [2, 0]]]

    assert agreement.agree_shapes([(1, 0), (2, 0), (4096, 0)]) == [(1, 0), (2, 0)]


@pytest.mark.cpu_only
def test_shape_list_is_untouched_without_a_group() -> None:
    agreement = _agreement(world_size=1)
    configs = [(1, 0), (128, 0)]

    assert agreement.agree_shapes(configs) == configs


@pytest.mark.cpu_only
def test_warmup_batch_policy_without_communicator() -> None:
    """An engine built without a communicator cannot strand anyone: skip, don't raise."""
    agreement = _agreement(with_dist=False)

    assert agreement.should_run_warmup_batch(None, 128, "general") is False


@pytest.mark.cpu_only
def test_tp_agreement_needs_a_communicator_not_just_a_mapping() -> None:
    """A mapping can claim TP peers that no communicator exists to reach.

    The agreement check gates on `tp_size`, so it has to answer for that
    state rather than walking into a collective on `None`.
    """
    agreement = _agreement(tp_size=2, with_dist=False)

    assert agreement.should_run_warmup_batch(object(), 128, "general") is True
    assert agreement.should_run_warmup_batch(None, 128, "general") is False


@pytest.mark.cpu_only
def test_tp_disagreement_reports_the_ranks_that_lost_their_batch() -> None:
    """Asymmetric attention-DP capacity is the deadlock this check exists for.

    The rank that still has a batch must not walk into a forward its peer
    will never join, and the failure has to name the peer so the KV cache
    fraction can be raised.

    DWDP is what still reaches the TP-only check: its peers are outside
    ``self.dist``, so no group agreement is available and the narrower TP
    check remains the only thing standing between an asymmetric world and a
    deadlock.
    """
    agreement = _agreement(world_size=2, dwdp_size=4, tp_size=2, tp_peer_values=(0,))

    with pytest.raises(RuntimeError) as excinfo:
        agreement.should_run_warmup_batch(object(), 128, "general")

    message = str(excinfo.value)
    assert "TP rank(s) [1]" in message
    assert "[128, 0]" in message


@pytest.mark.cpu_only
def test_tp_agreement_lets_a_symmetric_world_run() -> None:
    agreement = _agreement(world_size=2, tp_size=2, tp_peer_values=(1,))

    assert agreement.should_run_warmup_batch(object(), 128, "general") is True
