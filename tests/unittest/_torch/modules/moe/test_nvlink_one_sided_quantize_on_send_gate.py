# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which MoE dispatches quantize their BF16 input to NVFP4 while they send it (NVLinkOneSided).

``quantize_on_send_enabled`` is the one place that decides (switch ``TRTLLM_MOE_A2A_QUANTIZE_ON_SEND``: off unless it is
set to ``1`` or ``auto``; the fused-round kernels, no CFT counted writes, top_k 16, SM10x, the shape the launch
accepts); ``NVLinkOneSided.can_quantize_on_send``
adds the checks on the tensors, and the scheduler asks it. The dispatch kernel itself is checked bit for bit by
``multi_gpu/test_moe_a2a.py``: ``test_dispatch_quantize_on_send`` on the op,
``test_nvlink_one_sided_quantize_on_send_ab`` through ``NVLinkOneSided.dispatch``, and
``test_quantize_on_send_launch_boundaries`` for the agreement of the shape checks below with the real launch.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.communication import nvlink_one_sided
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_one_sided import (
    QUANTIZE_ON_SEND_ENV,
    NVLinkOneSided,
    _dispatch_block_size,
    quantize_on_send_enabled,
    quantize_on_send_launchable,
)
from tensorrt_llm._torch.modules.fused_moe.moe_scheduler import ExternalCommMoEScheduler
from tensorrt_llm._torch.utils import Fp4QuantizedTensor

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

BLOCK_SIZE_ENV = "TLLM_MOE_A2A_DISPATCH_BLOCK_SIZE"
# The payload rows sent as they are in the Kimi K3 decode dispatch: int32 expert ids [16] and float32 final scales [16].
K3_OTHER_ROW_BYTES = 2 * 4 * 16


@pytest.fixture
def platform(monkeypatch):
    """The SM version and the environment the gate reads: Rubin (SM107) with the K3 decode recipe's variables and the
    switch on ``auto`` (``switch=None``: unset, the default)."""

    def set_platform(sm=107, fused_round="1", quant_kf="1", switch="auto", block_size=None):
        monkeypatch.setattr(nvlink_one_sided, "get_sm_version", lambda: sm)
        for name, value in (
            ("TLLM_MOE_A2A_FUSED_ROUND", fused_round),
            ("TRTLLM_NVFP4_QUANT_KF", quant_kf),
            (QUANTIZE_ON_SEND_ENV, switch),
            (BLOCK_SIZE_ENV, block_size),
        ):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)

    set_platform()
    return set_platform


@pytest.mark.parametrize("rows", [1, 4, 48, 192, 256, 384, 512])
def test_auto_quantizes_the_k3_decode_dispatch_on_send(platform, rows):
    assert quantize_on_send_enabled(rows=rows, hidden=3584, top_k=16, use_cft=False)


# Opt-in: with the switch unset or 0 nothing is selected, on any platform, at any shape.
@pytest.mark.parametrize("switch", [None, "", "0"], ids=["unset", "empty", "off"])
@pytest.mark.parametrize("sm", [100, 103, 107])
@pytest.mark.parametrize("rows", [1, 48, 384, 512, 1024])
def test_nothing_is_selected_unless_the_switch_is_set(platform, switch, sm, rows):
    platform(sm=sm, switch=switch)
    assert not quantize_on_send_enabled(rows, 3584, 16, False)


def test_the_gate_of_the_comm_is_false_by_default_without_looking_at_anything(
    platform, monkeypatch
):
    platform(switch=None)
    comm = _comm(monkeypatch)
    # nothing that the check would read: not even a rows list
    assert comm.can_quantize_on_send(object(), object(), object(), object(), []) is False


# Switch auto: only what was measured: SM107, the K3 latent, at most 512 rows per rank. "1" forces every shape
# the kernel can launch on any SM10x.
@pytest.mark.parametrize("sm", [100, 103])
def test_auto_selection_needs_sm107(platform, sm):
    platform(sm=sm)
    assert not quantize_on_send_enabled(48, 3584, 16, False)
    platform(sm=sm, switch="1")
    assert quantize_on_send_enabled(48, 3584, 16, False)


@pytest.mark.parametrize("rows", [513, 1024, 1025])
def test_auto_selection_stops_at_512_rows_and_a_forced_one_does_not(platform, rows):
    assert not quantize_on_send_enabled(rows, 3584, 16, False)
    platform(switch="1")
    assert quantize_on_send_enabled(rows, 3584, 16, False)


@pytest.mark.parametrize("switch", ["auto", "1"], ids=["auto", "forced"])
@pytest.mark.parametrize("sm", [90, 120])
def test_only_sm10x_runs_the_kernel(platform, sm, switch):
    platform(sm=sm, switch=switch)
    assert not quantize_on_send_enabled(48, 3584, 16, False)


@pytest.mark.parametrize(
    "rows,hidden,top_k,use_cft",
    [
        (0, 3584, 16, False),
        (48, 3584, 8, False),
        (48, 3584, 16, True),
        (48, 3592, 16, False),  # not a multiple of 16
        (
            48,
            3600,
            16,
            False,
        ),  # a multiple of 16, but the scale rows (225 bytes) are not 16-byte multiples
    ],
    ids=["no_rows", "top_k_8", "cft", "hidden_not_16_multiple", "hidden_not_256_multiple"],
)
@pytest.mark.parametrize("switch", ["auto", "1"], ids=["auto", "forced"])
def test_what_the_kernel_cannot_run_is_never_selected(
    platform, switch, rows, hidden, top_k, use_cft
):
    platform(switch=switch)
    assert not quantize_on_send_enabled(rows, hidden, top_k, use_cft)


def test_without_the_fused_round_kernels_nothing_is_selected(platform):
    for switch in ("auto", "1"):
        platform(fused_round=None, switch=switch)
        assert not quantize_on_send_enabled(48, 3584, 16, False)
    platform(fused_round="0", switch="1")
    assert not quantize_on_send_enabled(48, 3584, 16, False)


# Switch auto: only what TRTLLM_NVFP4_QUANT_KF quantizes in a kernel of its own (the 3584-wide latent, up to 512
# rows); "1" forces every shape the kernel can run.
@pytest.mark.parametrize(
    "rows,hidden,quant_kf",
    [(513, 3584, "1"), (48, 7168, "1"), (48, 3584, None)],
    ids=["above_measured_rows", "not_the_k3_latent", "quant_kf_off"],
)
def test_auto_switch_follows_the_kf_quantizer_shapes(platform, rows, hidden, quant_kf):
    platform(quant_kf=quant_kf)
    assert not quantize_on_send_enabled(rows, hidden, 16, False)
    platform(quant_kf=quant_kf, switch="1")
    assert quantize_on_send_enabled(rows, hidden, 16, False)


# The shape checks of the launch (moe_a2a_dispatch_launch): hidden a multiple of 256, the staged rows [codes | scales |
# the other payloads] within 4 vectors of 16 bytes per thread, a block size of at least 64, and 128 + the staged bytes
# in 48 KB of shared memory. Staged bytes = hidden / 2 + hidden / 16 + 128.
@pytest.mark.parametrize(
    "hidden,block_size,other_row_bytes,launchable",
    [
        (3584, 256, K3_OTHER_ROW_BYTES, True),  # Kimi K3 latent: 134 vectors
        (7168, 256, K3_OTHER_ROW_BYTES, True),
        (1024, 256, K3_OTHER_ROW_BYTES, True),
        (512, 256, K3_OTHER_ROW_BYTES, True),
        (256, 256, K3_OTHER_ROW_BYTES, True),
        (3584, 256, 64, True),  # no final scales
        (3584, 256, 96, True),  # BF16 final scales
        (3584, 256, 256, True),  # 16 vectors outside the codes and scales: the most that fit
        (3584, 256, 272, False),  # 17 do not
        (3584, 256, 120, False),  # not 16-byte multiples
        (3600, 256, K3_OTHER_ROW_BYTES, False),  # hidden not a multiple of 256
        (3592, 256, K3_OTHER_ROW_BYTES, False),
        (128, 256, K3_OTHER_ROW_BYTES, False),
        (0, 256, K3_OTHER_ROW_BYTES, False),
        # 4 vectors per thread: 1024 vectors at block size 256 (hidden 28672 is 1016, 28928 is 1025)
        (28672, 256, K3_OTHER_ROW_BYTES, True),
        (28928, 256, K3_OTHER_ROW_BYTES, False),
        # block size 64: 256 vectors (hidden 6912 is 251, 7168 is 260)
        (6912, 64, K3_OTHER_ROW_BYTES, True),
        (7168, 64, K3_OTHER_ROW_BYTES, False),
        (3584, 32, K3_OTHER_ROW_BYTES, False),  # below the minimum block size
        (14336, 128, K3_OTHER_ROW_BYTES, True),
        (14592, 128, K3_OTHER_ROW_BYTES, False),
        (57856, 512, K3_OTHER_ROW_BYTES, True),
        (58112, 512, K3_OTHER_ROW_BYTES, False),
        # block size 1024: the shared-memory bound (128 of routing + 9 / 16 hidden + 128) is the tighter one
        (86784, 1024, K3_OTHER_ROW_BYTES, True),
        (87040, 1024, K3_OTHER_ROW_BYTES, False),
    ],
)
def test_launchable_mirrors_the_launch_checks(hidden, block_size, other_row_bytes, launchable):
    assert quantize_on_send_launchable(hidden, 16, other_row_bytes, block_size) is launchable


@pytest.mark.parametrize(
    "value,block_size",
    [
        (None, 256),
        ("", 256),
        ("abc", 256),  # std::stoi throws: the default
        ("128", 128),
        (" 128", 128),
        ("+64", 64),
        ("128abc", 128),  # stoi stops at the first non-digit
        ("-5", 256),
        ("0", 256),
        ("100", 128),  # rounded up to a multiple of 32
        ("33", 64),
        ("1024", 1024),
        ("1025", 1024),  # clamped
        ("5000", 1024),
        ("99999999999", 256),  # out of range for stoi: the default
    ],
)
def test_block_size_is_read_as_the_launch_reads_it(platform, value, block_size):
    platform(block_size=value)
    assert _dispatch_block_size() == block_size


def test_forced_selection_declines_shapes_the_launch_rejects_instead_of_raising(platform):
    platform(switch="1")
    assert quantize_on_send_enabled(48, 3584, 16, False)
    assert not quantize_on_send_enabled(48, 3600, 16, False)
    assert not quantize_on_send_enabled(48, 28928, 16, False)
    assert not quantize_on_send_enabled(48, 3584, 16, False, other_row_bytes=272)
    platform(switch="1", block_size="64")
    assert quantize_on_send_enabled(48, 3584, 16, False)
    assert not quantize_on_send_enabled(48, 7168, 16, False)
    platform(switch="1", block_size="32")
    assert not quantize_on_send_enabled(48, 3584, 16, False)


def test_switch_off_wins(platform):
    platform(switch="0")
    assert not quantize_on_send_enabled(48, 3584, 16, False)


@pytest.mark.parametrize("value", ["2", "yes", "-1", "true", "Auto"])
def test_bad_switch_value_is_rejected(platform, value):
    platform(switch=value)
    with pytest.raises(ValueError, match=QUANTIZE_ON_SEND_ENV):
        quantize_on_send_enabled(48, 3584, 16, False)


@pytest.mark.parametrize("value", ["2", "yes", "-1", "true", "Auto"])
def test_bad_switch_value_is_rejected_when_the_comm_is_built(platform, monkeypatch, value):
    """At the start of the server, on every rank, whether or not the scheduler would ever ask the gate: a rank that
    raised in a step while its peers went on into the dispatch would hang them."""
    platform(switch=value)
    monkeypatch.setattr(NVLinkOneSided, "is_platform_supported", staticmethod(lambda: True))
    mapping = types.SimpleNamespace(moe_ep_size=1, moe_ep_rank=0, world_size=1)
    with pytest.raises(ValueError, match=QUANTIZE_ON_SEND_ENV):
        NVLinkOneSided(mapping, num_slots=32, top_k=16, max_num_tokens_per_rank=8)


def _comm(monkeypatch, use_cft=False, enable_eplb=False, rank_mask=False):
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    comm.top_k = 16
    comm.enable_eplb = enable_eplb
    comm._rank_mask_enabled = rank_mask
    comm._fake_use_cft = use_cft
    monkeypatch.setattr(
        NVLinkOneSided, "use_cft_for_dispatch", lambda self, rows: self._fake_use_cft
    )
    return comm


@requires_cuda
def test_tensor_checks(platform, monkeypatch):
    comm = _comm(monkeypatch)
    x = torch.zeros(48, 3584, dtype=torch.bfloat16, device="cuda")
    gs = torch.ones(1, dtype=torch.float32, device="cuda")
    slots = torch.zeros(48, 16, dtype=torch.int32, device="cuda")
    scales = torch.zeros(48, 16, dtype=torch.float32, device="cuda")
    ranks = [48, 48, 48, 48]

    def can(x=x, gs=gs, slots=slots, scales=scales, comm=comm):
        return comm.can_quantize_on_send(x, gs, slots, scales, ranks)

    assert can()
    assert can(scales=None)  # no final scales
    assert can(scales=scales.bfloat16())
    assert not can(gs=None)
    assert not can(x=x.float())
    assert not can(x=x.view(-1))  # one-dimensional
    assert not can(x=x[:, :3568])  # not contiguous
    assert not can(gs=gs.double())
    assert not can(gs=gs.cpu())
    assert not can(gs=gs[:0])  # no scale
    shifted = torch.zeros(48 * 3584 + 16, dtype=torch.bfloat16, device="cuda")[8 : 8 + 48 * 3584]
    assert not can(x=shifted.view(48, 3584))  # 16 B aligned, not 32 B
    assert not can(slots=slots.long())  # the op takes int32 expert ids
    assert not can(slots=slots.view(-1))
    assert not can(scales=scales.view(-1))
    assert not can(comm=_comm(monkeypatch, use_cft=True))
    assert not can(comm=_comm(monkeypatch, enable_eplb=True))  # the EPLB-statistics kernel variant
    assert not can(comm=_comm(monkeypatch, rank_mask=True))  # the rank-mask kernel variant
    # The rows of the routing payloads enter the staging: 16 vectors outside the codes and scales is the most that fit.
    assert can(scales=torch.zeros(48, 48, dtype=torch.float32, device="cuda"))  # 64 + 192 bytes
    assert not can(scales=torch.zeros(48, 52, dtype=torch.float32, device="cuda"))  # 64 + 208 bytes


@requires_cuda
@pytest.mark.parametrize(
    "all_rank_num_tokens,expected",
    [
        ([300, 500], True),
        ([300, 520], False),  # above the 512 rows that ``auto`` was measured at, on every rank
        ([520, 300], False),
        ([0, 48, 0, 7], True),  # ranks without rows follow the others
        ([0, 0], False),  # nothing to send
    ],
    ids=[
        "below_cap",
        "above_cap_on_the_last_rank",
        "above_cap_on_the_first_rank",
        "idle_ranks",
        "no_rows",
    ],
)
def test_every_rank_reaches_the_same_decision(platform, monkeypatch, all_rank_num_tokens, expected):
    """The payloads of a quantizing and of a non-quantizing dispatch differ, so the ranks of a dispatch must agree on
    which one it is: ``auto`` caps the rows of the dispatch (the largest count of any rank), not the rows that
    one rank happens to hold, which differ from rank to rank unless DP padding makes them equal."""
    comm = _comm(monkeypatch)
    gs = torch.ones(1, dtype=torch.float32, device="cuda")
    decisions = set()
    for local_rows in sorted(set(all_rank_num_tokens)):
        x = torch.zeros(local_rows, 3584, dtype=torch.bfloat16, device="cuda")
        slots = torch.zeros(local_rows, 16, dtype=torch.int32, device="cuda")
        scales = torch.zeros(local_rows, 16, dtype=torch.float32, device="cuda")
        decisions.add(comm.can_quantize_on_send(x, gs, slots, scales, all_rank_num_tokens))
    assert decisions == {expected}


class _DispatchOps:
    """Stands in for ``torch.ops.trtllm``: ``moe_a2a_dispatch`` records its call and returns the receive buffers."""

    def __init__(self, recv_buffers):
        self.recv_buffers = recv_buffers
        self.calls = []

    def moe_a2a_dispatch(self, *args):
        self.calls.append(args)
        return self.recv_buffers, 0, torch.empty(0, dtype=torch.int32)


class _TorchWithOps:
    """The ``torch`` of nvlink_one_sided.py with the ``trtllm`` ops replaced by ``ops``."""

    def __init__(self, ops):
        self.ops = types.SimpleNamespace(trtllm=ops)

    def __getattr__(self, name):
        return getattr(torch, name)


EP_SIZE = 2
RUNTIME_MAX = 5
ROWS = 3


def _dispatch_comm(monkeypatch, recv_buffers, use_cft=False):
    comm = _comm(monkeypatch, use_cft=use_cft)
    comm._dispatch_state = {"phase": "idle"}
    comm._watchdog_coordinator = types.SimpleNamespace(
        capture_active_rank_mask=lambda mask: types.SimpleNamespace(active_rank_mask=mask),
        watch_collective=lambda *args: None,
    )
    comm._alltoall_watchdog = None
    comm.workspace = comm.moe_a2a_metainfo = None
    comm.ep_rank, comm.ep_size, comm.num_experts = 0, EP_SIZE, 64
    comm.invalid_token_expert_id = -1
    ops = _DispatchOps(recv_buffers)
    monkeypatch.setattr(nvlink_one_sided, "torch", _TorchWithOps(ops))
    return comm, ops


def _recv(width, dtype, fill):
    return torch.full((EP_SIZE, RUNTIME_MAX, width), fill, dtype=dtype)


def _routing(final_scales=True):
    slots = torch.zeros(ROWS, 16, dtype=torch.int32)
    return slots, torch.zeros(ROWS, 16, dtype=torch.float32) if final_scales else None


def _dispatch(comm, hidden_states, hidden_states_sf, slots, scales, **kwargs):
    return comm.dispatch(
        hidden_states=hidden_states,
        hidden_states_sf=hidden_states_sf,
        token_selected_slots=slots,
        token_final_scales=scales,
        all_rank_num_tokens=[ROWS, RUNTIME_MAX],
        **kwargs,
    )


# The index of the received tensors that dispatch returns for each input payload. With quantize-on-send the op sends the
# NVFP4 codes and scales of the BF16 input as its first two payloads (one more than it was given).
@pytest.mark.parametrize("final_scales", [True, False], ids=["final_scales", "no_final_scales"])
def test_dispatch_with_quantize_on_send_returns_codes_scales_ids_and_final_scales(
    monkeypatch, final_scales
):
    recv = [_recv(1792, torch.uint8, 1), _recv(224, torch.uint8, 2), _recv(16, torch.int32, 3)]
    if final_scales:
        recv.append(_recv(16, torch.float32, 4.0))
    comm, ops = _dispatch_comm(monkeypatch, recv)
    slots, scales = _routing(final_scales)
    x = torch.zeros(ROWS, 3584, dtype=torch.bfloat16)
    global_scale = torch.ones(1)
    codes, sf, ids, final = _dispatch(
        comm, x, None, slots, scales, quantize_on_send_scale=global_scale
    )
    assert codes.shape == (EP_SIZE * RUNTIME_MAX, 1792) and codes.dtype == torch.uint8
    assert sf.shape == (EP_SIZE * RUNTIME_MAX, 224) and sf.dtype == torch.uint8
    assert ids.shape == (EP_SIZE * RUNTIME_MAX, 16) and ids.dtype == torch.int32
    assert (codes == 1).all() and (sf == 2).all() and (ids == 3).all()
    if final_scales:
        assert final.shape == (EP_SIZE * RUNTIME_MAX, 16) and (final == 4.0).all()
    else:
        assert final is None
    (call,) = ops.calls
    payloads = call[1]
    assert payloads[0] is x and payloads[1] is slots  # the BF16 input goes in as it is
    assert len(payloads) == (3 if final_scales else 2)
    assert call[11] == 1  # the expert-id payload index counts the input payloads
    assert call[10] is False  # the fence dispatch
    assert call[-1] is global_scale
    assert comm._dispatch_state["phase"] == "dispatched"


@pytest.mark.parametrize("final_scales", [True, False], ids=["final_scales", "no_final_scales"])
@pytest.mark.parametrize("with_sf", [True, False], ids=["post_quant", "dispatch_then_quantize"])
def test_dispatch_without_quantize_on_send_keeps_its_return_mapping(
    monkeypatch, with_sf, final_scales
):
    recv = [_recv(1792 if with_sf else 3584, torch.uint8 if with_sf else torch.bfloat16, 1)]
    if with_sf:
        recv.append(_recv(224, torch.uint8, 2))
    recv.append(_recv(16, torch.int32, 3))
    if final_scales:
        recv.append(_recv(16, torch.float32, 4.0))
    comm, ops = _dispatch_comm(monkeypatch, recv)
    slots, scales = _routing(final_scales)
    x = torch.zeros(
        ROWS, 1792 if with_sf else 3584, dtype=torch.uint8 if with_sf else torch.bfloat16
    )
    sf = torch.zeros(ROWS, 224, dtype=torch.uint8) if with_sf else None
    hidden, hidden_sf, ids, final = _dispatch(comm, x, sf, slots, scales)
    assert (hidden == 1).all()
    assert (hidden_sf is not None) == with_sf and (not with_sf or (hidden_sf == 2).all())
    assert (ids == 3).all()
    assert (final is not None) == final_scales and (not final_scales or (final == 4.0).all())
    assert ops.calls[0][-1] is None  # no quantize-on-send scale


def test_dispatch_logs_the_first_quantize_on_send_of_a_shape_once(monkeypatch):
    lines = []
    logger = nvlink_one_sided.tllm_logger
    monkeypatch.setattr(logger, "log", lambda severity, *msg: lines.append(" ".join(msg)))
    key = f"nvlink_one_sided_quantize_on_send_{ROWS}_3584"
    logger._appeared_keys.discard(key)
    recv = [_recv(1792, torch.uint8, 1), _recv(224, torch.uint8, 2), _recv(16, torch.int32, 3)]
    comm, _ = _dispatch_comm(monkeypatch, recv)
    slots, _ = _routing(final_scales=False)
    x = torch.zeros(ROWS, 3584, dtype=torch.bfloat16)
    for _ in range(2):
        _dispatch(comm, x, None, slots, None, quantize_on_send_scale=torch.ones(1))
        comm._dispatch_state = {"phase": "idle"}
    assert len([line for line in lines if "quantize-on-send engaged" in line]) == 1
    assert f"rows={ROWS}, hidden=3584" in lines[0] and QUANTIZE_ON_SEND_ENV in lines[0]


def test_dispatch_rejects_what_quantize_on_send_cannot_take_with_exceptions_not_asserts(
    monkeypatch,
):
    recv = [_recv(1792, torch.uint8, 1), _recv(224, torch.uint8, 2), _recv(16, torch.int32, 3)]
    comm, ops = _dispatch_comm(monkeypatch, recv)
    slots, _ = _routing(final_scales=False)
    x = torch.zeros(ROWS, 3584, dtype=torch.bfloat16)
    scale = torch.ones(1)
    with pytest.raises(
        ValueError, match="BF16 hidden states"
    ):  # scaling factors are the quantization's job
        _dispatch(
            comm,
            x,
            torch.zeros(ROWS, 224, dtype=torch.uint8),
            slots,
            None,
            quantize_on_send_scale=scale,
        )
    with pytest.raises(ValueError, match="BF16 hidden states"):
        _dispatch(comm, x.float(), None, slots, None, quantize_on_send_scale=scale)
    comm, ops = _dispatch_comm(monkeypatch, recv, use_cft=True)
    with pytest.raises(RuntimeError, match="fence dispatch"):
        _dispatch(comm, x, None, slots, None, quantize_on_send_scale=scale)
    assert not ops.calls


class _Backend:
    """A MoE backend whose plain NVFP4 quantization has the global scale ``scale`` (None: it has none)."""

    def __init__(self, scale):
        self.scale = scale

    def quantize_on_send_global_scale(self, x):
        return self.scale


def _scheduler(comm, backend):
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = types.SimpleNamespace(comm=comm, backend=backend)
    return scheduler


def test_scheduler_quantizes_on_send_when_the_comm_can(monkeypatch):
    seen = []

    def can_quantize_on_send(self, x, global_scale, slots, final_scales, all_rank_num_tokens):
        seen.append((x, global_scale, slots, final_scales, all_rank_num_tokens))
        return True

    monkeypatch.setattr(NVLinkOneSided, "can_quantize_on_send", can_quantize_on_send)
    scale = torch.ones(1)
    x = torch.zeros(4, 3584, dtype=torch.bfloat16)
    slots, final_scales = torch.zeros(4, 16, dtype=torch.int32), torch.zeros(4, 16)
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    got = _scheduler(comm, _Backend(scale))._quantize_on_send_scale(x, slots, final_scales, [4, 4])
    assert got is scale
    assert seen == [(x, scale, slots, final_scales, [4, 4])]


@pytest.mark.parametrize(
    "what", ["comm_cannot", "no_scale", "other_comm", "already_quantized", "no_method"]
)
def test_scheduler_falls_back_to_quantize_input(monkeypatch, what):
    monkeypatch.setattr(NVLinkOneSided, "can_quantize_on_send", lambda *args: what != "comm_cannot")
    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    backend = _Backend(None if what == "no_scale" else torch.ones(1))
    x = torch.zeros(4, 3584, dtype=torch.bfloat16)
    if what == "other_comm":
        comm = object()
    elif what == "already_quantized":
        x = Fp4QuantizedTensor(
            torch.zeros(4, 1792, dtype=torch.uint8), torch.zeros(4, 224, dtype=torch.uint8)
        )
    elif what == "no_method":
        backend = object()
    slots, final_scales = torch.zeros(4, 16, dtype=torch.int32), torch.zeros(4, 16)
    assert _scheduler(comm, backend)._quantize_on_send_scale(x, slots, final_scales, [4, 4]) is None
