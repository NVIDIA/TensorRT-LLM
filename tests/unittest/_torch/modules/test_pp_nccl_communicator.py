# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest

from tensorrt_llm import _mnnvl_utils as mnnvl_module
from tensorrt_llm._torch.distributed import communicator as communicator_module


@pytest.fixture(autouse=True)
def _reset_pp_comm_lifecycle(monkeypatch):
    previous_helix_comm = mnnvl_module.HelixCpMnnvlMemory.comm
    previous_helix_topology = mnnvl_module.HelixCpMnnvlMemory.comm_topology
    communicator_module._pp_comm = None
    communicator_module._pp_comm_refcount = 0
    communicator_module._pp_comm_control_refcount = 0
    communicator_module._pp_comm_final_release_pending = False
    monkeypatch.setattr(communicator_module, "NCCL_FAULT_TOLERANCE_ENABLED", True)
    mnnvl_module.HelixCpMnnvlMemory.comm = None
    mnnvl_module.HelixCpMnnvlMemory.comm_topology = None
    yield
    communicator_module._pp_comm = None
    communicator_module._pp_comm_refcount = 0
    communicator_module._pp_comm_control_refcount = 0
    communicator_module._pp_comm_final_release_pending = False
    mnnvl_module.HelixCpMnnvlMemory.comm = previous_helix_comm
    mnnvl_module.HelixCpMnnvlMemory.comm_topology = previous_helix_topology


class _FakeNcclCommunicatorOp:
    def __init__(self):
        self.aborted = False
        self.reinitialized_with = None
        self.reinit_calls = []
        self.rendezvous_ids = []
        self.active_ranks = [0, 1, 2, 3]
        self.async_error = "NCCL error: communicator was aborted after injected async failure"
        self.received_from = None

    def abort(self):
        self.aborted = True

    def abort_and_reinit(self, active_ranks, rendezvous_id):
        self.reinitialized_with = list(active_ranks)
        self.reinit_calls.append(list(active_ranks))
        self.rendezvous_ids.append(rendezvous_id)
        self.active_ranks = list(active_ranks)
        self.async_error = ""

    def get_async_error(self):
        return self.async_error

    def get_active_ranks(self):
        return list(self.active_ranks)

    def recv(self, tensor, src):
        self.received_from = src

    def send(self, tensor, dest):
        self.sent_to = dest


class _FakeMapping:
    world_size = 4
    rank = 2
    pp_group = [0, 1, 2, 3]
    cp_config = {}

    def has_cp_helix(self):
        return False

    def prev_pp_rank(self):
        return 1

    def next_pp_rank(self):
        return 3


class _CompatibleMapping(_FakeMapping):
    pass


class _HelixMapping(_FakeMapping):
    cp_config = {"use_nccl_for_alltoall": False}
    cp_group = [0, 2]
    cp_rank = 1
    pp_rank = 0
    tp_size = 2
    tp_rank = 0

    def has_cp_helix(self):
        return True


class _OtherHelixMapping(_HelixMapping):
    cp_group = [1, 2]


def _make_pp_comm(mapping=None):
    pp_comm = object.__new__(communicator_module.PPCommNCCL)
    pp_comm.mapping = mapping or _FakeMapping()
    pp_comm._topology = pp_comm._mapping_topology(pp_comm.mapping)
    pp_comm._active_ranks = tuple(range(pp_comm.mapping.world_size))
    pp_comm._reconfigure_lock = threading.Lock()
    pp_comm._reconfigure_generation = 0
    pp_comm._completed_recovery = None
    pp_comm.nccl_comm = _FakeNcclCommunicatorOp()
    return pp_comm


def test_pp_wrapper_initializes_membership_from_persistent_native_comm(monkeypatch):
    native_comm = _FakeNcclCommunicatorOp()
    native_comm.active_ranks = [0, 2, 3]
    monkeypatch.setattr(
        communicator_module.torch.classes.trtllm,
        "NcclCommunicatorOp",
        lambda world_size, rank: native_comm,
        raising=False,
    )
    monkeypatch.setattr(communicator_module.torch.cuda, "Event", object)
    monkeypatch.setattr(communicator_module.torch.cuda, "Stream", object)

    pp_comm = communicator_module.PPCommNCCL(_FakeMapping())

    assert pp_comm.get_active_ranks() == [0, 2, 3]


def test_default_off_wrapper_does_not_query_native_membership(monkeypatch):
    native_comm = _FakeNcclCommunicatorOp()
    native_comm.get_active_ranks = lambda: (_ for _ in ()).throw(
        AssertionError("default-off PP constructor queried native membership")
    )
    monkeypatch.setattr(communicator_module, "NCCL_FAULT_TOLERANCE_ENABLED", False)
    monkeypatch.setattr(
        communicator_module.torch.classes.trtllm,
        "NcclCommunicatorOp",
        lambda world_size, rank: native_comm,
        raising=False,
    )
    monkeypatch.setattr(communicator_module.torch.cuda, "Event", object)
    monkeypatch.setattr(communicator_module.torch.cuda, "Stream", object)

    pp_comm = communicator_module.PPCommNCCL(_FakeMapping())

    assert pp_comm._active_ranks == (0, 1, 2, 3)


def test_pp_nccl_fault_tolerance_methods_forward_to_custom_class():
    pp_comm = _make_pp_comm()

    assert "NCCL error" in pp_comm.get_async_error()
    pp_comm.abort_and_reinit((0, 2, 3))
    assert pp_comm.nccl_comm.reinitialized_with == [0, 2, 3]
    assert pp_comm.nccl_comm.rendezvous_ids == [1]
    assert pp_comm.get_async_error() == ""
    assert pp_comm.get_active_ranks() == [0, 2, 3]

    pp_comm.abort()
    assert pp_comm.nccl_comm.aborted


def test_module_level_pp_nccl_fault_tolerance_api(monkeypatch):
    pp_comm = _make_pp_comm()
    monkeypatch.setattr(communicator_module, "_pp_comm", pp_comm)
    monkeypatch.setattr(communicator_module, "_pp_comm_refcount", 1)

    assert "NCCL error" in communicator_module.pp_comm_get_async_error()
    communicator_module.pp_comm_abort_and_reinit([0, 2, 3], generation=3)
    assert pp_comm.nccl_comm.reinitialized_with == [0, 2, 3]
    assert pp_comm.nccl_comm.rendezvous_ids == [5]
    assert communicator_module.pp_comm_get_async_error() == ""
    assert communicator_module.pp_comm_get_active_ranks() == [0, 2, 3]

    communicator_module.pp_comm_abort()
    assert pp_comm.nccl_comm.aborted


def test_module_level_pp_nccl_api_requires_nccl_backend(monkeypatch):
    monkeypatch.setattr(communicator_module, "_pp_comm", object())

    with pytest.raises(RuntimeError, match="NCCL error: PP communicator is not initialized"):
        communicator_module.pp_comm_get_async_error()


def test_reinit_rejects_stale_default_peer_and_accepts_explicit_remap():
    pp_comm = _make_pp_comm()
    pp_comm.abort_and_reinit([0, 2, 3])

    # Static Mapping.prev_pp_rank() names failed rank 1. Silently skipping it
    # would bypass that stage's model layers, so default routing must fail.
    with pytest.raises(RuntimeError, match="peer world rank 1 is not active"):
        pp_comm.recv(object())

    # A higher-level layer/topology reconstruction may provide an explicit
    # replacement peer after it has reassigned the failed stage.
    pp_comm.recv(object(), src=0)
    assert pp_comm.nccl_comm.received_from == 0
    assert pp_comm._required_pp_peer(next_peer=True) == 3


def test_explicit_peer_must_belong_to_the_callers_pp_lane():
    class LaneMapping:
        world_size = 8
        rank = 2
        pp_group = [2, 6]
        cp_config = {}

        def has_cp_helix(self):
            return False

        def prev_pp_rank(self):
            return 6

        def next_pp_rank(self):
            return 6

    pp_comm = _make_pp_comm(LaneMapping())

    with pytest.raises(RuntimeError, match="not in this rank's PP group"):
        pp_comm.recv(object(), src=3)

    pp_comm.recv(object(), src=6)
    assert pp_comm.nccl_comm.received_from == 6


def test_default_off_preserves_legacy_peer_routing(monkeypatch):
    pp_comm = _make_pp_comm()
    monkeypatch.setattr(communicator_module, "NCCL_FAULT_TOLERANCE_ENABLED", False)
    monkeypatch.setattr(
        communicator_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )

    # Before FT wiring, explicit peers were forwarded to the native wrapper
    # without a Python PP-lane scan. Preserve that behavior and cost when the
    # startup mode is disabled.
    tensor = object()
    pp_comm.send(tensor, dest=7)
    pp_comm.recv(tensor, src=7)

    assert pp_comm.nccl_comm.sent_to == 7
    assert pp_comm.nccl_comm.received_from == 7


def test_reinit_rejects_reactivating_removed_pp_rank():
    pp_comm = _make_pp_comm()
    pp_comm.abort_and_reinit([0, 2, 3])

    with pytest.raises(ValueError, match="reactivate"):
        pp_comm.abort_and_reinit([0, 1, 2, 3])


def test_duplicate_reinit_target_is_idempotent():
    pp_comm = _make_pp_comm()

    pp_comm.abort_and_reinit([0, 2, 3], generation=1)
    pp_comm.abort_and_reinit([3, 2, 0], generation=1)

    assert pp_comm.nccl_comm.reinit_calls == [[0, 2, 3]]
    assert pp_comm.nccl_comm.rendezvous_ids == [3]
    assert pp_comm.get_active_ranks() == [0, 2, 3]


def test_shared_generation_forces_same_membership_rebuild():
    pp_comm = _make_pp_comm()

    pp_comm.abort_and_reinit([0, 1, 2, 3], generation=1)
    pp_comm.abort_and_reinit([0, 1, 2, 3], generation=1)
    pp_comm.abort_and_reinit([0, 1, 2, 3], generation=2)

    assert pp_comm.nccl_comm.reinit_calls == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert pp_comm.nccl_comm.rendezvous_ids == [3, 4]
    assert pp_comm.get_async_error() == ""


def test_same_membership_recovery_requires_shared_generation():
    pp_comm = _make_pp_comm()

    with pytest.raises(ValueError, match="generation is required"):
        pp_comm.abort_and_reinit([0, 1, 2, 3])

    assert pp_comm.nccl_comm.reinit_calls == []


@pytest.mark.parametrize("generation", [-1, 1.5, (1 << 63) - 2])
def test_recovery_generation_must_fit_reserved_torch_int_range(generation):
    pp_comm = _make_pp_comm()

    with pytest.raises(ValueError, match="generation must be a nonnegative integer"):
        pp_comm.abort_and_reinit([0, 2, 3], generation=generation)

    assert pp_comm.nccl_comm.reinit_calls == []


def test_generation_rejects_conflicting_pp_recovery_targets():
    pp_comm = _make_pp_comm()
    pp_comm.abort_and_reinit([0, 2, 3], generation=7)

    with pytest.raises(RuntimeError, match="conflicting.*generation 7"):
        pp_comm.abort_and_reinit([0, 2], generation=7)

    assert pp_comm.nccl_comm.reinit_calls == [[0, 2, 3]]


def test_concurrent_nested_reinit_requests_commit_newest_membership():
    pp_comm = _make_pp_comm()
    first_entered_native = threading.Event()
    second_entered_native = threading.Event()
    release_first = threading.Event()
    calls = []

    def blocking_reinit(active_ranks, rendezvous_id):
        active_ranks = list(active_ranks)
        calls.append((active_ranks, rendezvous_id))
        pp_comm.nccl_comm.active_ranks = active_ranks
        if active_ranks == [0, 2, 3]:
            first_entered_native.set()
            assert release_first.wait(timeout=5)
        else:
            second_entered_native.set()

    pp_comm.nccl_comm.abort_and_reinit = blocking_reinit
    errors = []

    def reconfigure(active_ranks):
        try:
            pp_comm.abort_and_reinit(active_ranks)
        except Exception as error:  # pragma: no cover - assertion reports details
            errors.append(error)

    first = threading.Thread(target=reconfigure, args=([0, 2, 3],))
    second = threading.Thread(target=reconfigure, args=([0, 2],))
    first.start()
    assert first_entered_native.wait(timeout=5)
    second.start()

    # The second callback must not enter the native rendezvous while the first
    # callback is between native completion and Python membership publication.
    serialized = not second_entered_native.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert serialized
    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert calls == [([0, 2, 3], 1), ([0, 2], 1)]
    assert pp_comm.get_active_ranks() == [0, 2]


def test_abort_waiting_for_reinit_does_not_abort_replacement():
    pp_comm = _make_pp_comm()
    abort_generation = pp_comm._reconfigure_generation
    entered_reinit = threading.Event()
    release_reinit = threading.Event()
    abort_started = threading.Event()

    def blocking_reinit(active_ranks, rendezvous_id):
        assert rendezvous_id == 1
        entered_reinit.set()
        assert release_reinit.wait(timeout=5)
        pp_comm.nccl_comm.active_ranks = list(active_ranks)
        pp_comm.nccl_comm.async_error = ""

    pp_comm.nccl_comm.abort_and_reinit = blocking_reinit
    errors = []

    def reconfigure():
        try:
            pp_comm.abort_and_reinit([0, 2, 3])
        except Exception as error:  # pragma: no cover - assertion reports details
            errors.append(error)

    def abort():
        abort_started.set()
        try:
            pp_comm.abort(abort_generation)
        except Exception as error:  # pragma: no cover - assertion reports details
            errors.append(error)

    rebuild_thread = threading.Thread(target=reconfigure)
    abort_thread = threading.Thread(target=abort)
    rebuild_thread.start()
    assert entered_reinit.wait(timeout=5)
    abort_thread.start()
    assert abort_started.wait(timeout=5)
    release_reinit.set()
    rebuild_thread.join(timeout=5)
    abort_thread.join(timeout=5)

    assert not rebuild_thread.is_alive()
    assert not abort_thread.is_alive()
    assert errors == []
    assert not pp_comm.nccl_comm.aborted
    assert pp_comm.get_active_ranks() == [0, 2, 3]

    # A genuinely new abort request after the rebuild must still take effect.
    pp_comm.abort()
    assert pp_comm.nccl_comm.aborted


def test_module_abort_waiting_for_reinit_does_not_abort_replacement(monkeypatch):
    pp_comm = _make_pp_comm()
    entered_reinit = threading.Event()
    abort_captured_generation = threading.Event()
    release_reinit = threading.Event()
    errors = []

    def blocking_reinit(active_ranks, rendezvous_id):
        assert rendezvous_id == 1
        entered_reinit.set()
        assert release_reinit.wait(timeout=5)
        pp_comm.nccl_comm.active_ranks = list(active_ranks)
        pp_comm.nccl_comm.async_error = ""

    pp_comm.nccl_comm.abort_and_reinit = blocking_reinit
    original_abort = pp_comm.abort

    def observed_abort(generation=None):
        # The module wrapper has already captured the epoch before it invokes
        # the class method. Do not let the rebuild finish until that point.
        abort_captured_generation.set()
        return original_abort(generation)

    pp_comm.abort = observed_abort
    monkeypatch.setattr(communicator_module, "_pp_comm", pp_comm)
    monkeypatch.setattr(communicator_module, "_pp_comm_refcount", 1)

    def reconfigure():
        try:
            communicator_module.pp_comm_abort_and_reinit([0, 2, 3])
        except Exception as error:  # pragma: no cover - assertion reports details
            errors.append(error)

    def abort():
        try:
            communicator_module.pp_comm_abort()
        except Exception as error:  # pragma: no cover - assertion reports details
            errors.append(error)

    rebuild_thread = threading.Thread(target=reconfigure)
    abort_thread = threading.Thread(target=abort)
    rebuild_thread.start()
    assert entered_reinit.wait(timeout=5)
    abort_thread.start()
    assert abort_captured_generation.wait(timeout=5)
    release_reinit.set()
    rebuild_thread.join(timeout=5)
    abort_thread.join(timeout=5)

    assert not rebuild_thread.is_alive()
    assert not abort_thread.is_alive()
    assert errors == []
    assert not pp_comm.nccl_comm.aborted
    assert pp_comm.get_active_ranks() == [0, 2, 3]

    communicator_module.pp_comm_abort()
    assert pp_comm.nccl_comm.aborted


def test_module_control_call_blocks_final_release(monkeypatch):
    pp_comm = _make_pp_comm()
    entered_query = threading.Event()
    release_query = threading.Event()
    final_release_completed = threading.Event()
    errors = []

    def blocking_get_async_error():
        entered_query.set()
        assert release_query.wait(timeout=5)
        return ""

    pp_comm.nccl_comm.get_async_error = blocking_get_async_error
    monkeypatch.setattr(communicator_module, "_pp_comm", pp_comm)
    monkeypatch.setattr(communicator_module, "_pp_comm_refcount", 1)

    def query():
        try:
            communicator_module.pp_comm_get_async_error()
        except Exception as error:  # pragma: no cover - assertion reports details
            errors.append(error)

    def final_release():
        try:
            communicator_module.release_pp_comm()
            final_release_completed.set()
        except Exception as error:  # pragma: no cover - assertion reports details
            errors.append(error)

    query_thread = threading.Thread(target=query)
    release_thread = threading.Thread(target=final_release)
    query_thread.start()
    assert entered_query.wait(timeout=5)
    release_thread.start()

    # The control call owns a transient reference under _pp_comm_lock, so a
    # final engine release cannot complete its ownership transition yet.
    assert not final_release_completed.wait(timeout=0.1)
    release_query.set()
    query_thread.join(timeout=5)
    release_thread.join(timeout=5)

    assert not query_thread.is_alive()
    assert not release_thread.is_alive()
    assert errors == []
    assert final_release_completed.is_set()
    assert communicator_module._pp_comm is pp_comm
    assert communicator_module._pp_comm_refcount == 0


def test_init_pp_comm_reuses_compatible_nccl_communicator(monkeypatch):
    created = []

    class FakePPComm:
        def __init__(self, mapping):
            self.topology = (mapping.world_size, mapping.rank, tuple(mapping.pp_group))
            created.append(self)

        def is_compatible(self, mapping):
            return self.topology == (mapping.world_size, mapping.rank, tuple(mapping.pp_group))

    monkeypatch.setattr(communicator_module, "PPCommNCCL", FakePPComm)
    monkeypatch.setattr(communicator_module, "_pp_comm", None)
    monkeypatch.setattr(communicator_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(communicator_module, "init_helix_cp_comm", lambda mapping: None)

    communicator_module.init_pp_comm(_FakeMapping())
    first = communicator_module._pp_comm
    # A target and draft engine construct equivalent Mapping objects. They
    # must share the sole process-local native PP communicator.
    communicator_module.init_pp_comm(_CompatibleMapping())

    assert communicator_module._pp_comm is first
    assert len(created) == 1
    assert communicator_module._pp_comm_refcount == 2

    communicator_module.release_pp_comm()
    assert communicator_module._pp_comm is first
    communicator_module.release_pp_comm()
    assert communicator_module._pp_comm is first
    assert communicator_module._pp_comm_refcount == 0


def test_ft_wrapper_churn_preserves_completed_recovery_generation(monkeypatch):
    pp_comm = _make_pp_comm()
    monkeypatch.setattr(communicator_module, "_pp_comm", pp_comm)
    monkeypatch.setattr(communicator_module, "_pp_comm_refcount", 1)
    monkeypatch.setattr(communicator_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(communicator_module, "init_helix_cp_comm", lambda mapping: None)

    communicator_module.pp_comm_abort_and_reinit([0, 2, 3], generation=7)
    communicator_module.release_pp_comm()
    assert communicator_module._pp_comm is pp_comm
    assert communicator_module._pp_comm_refcount == 0

    communicator_module.init_pp_comm(_CompatibleMapping())
    communicator_module.pp_comm_abort_and_reinit([0, 2, 3], generation=7)

    assert communicator_module._pp_comm is pp_comm
    assert pp_comm.nccl_comm.reinit_calls == [[0, 2, 3]]
    assert pp_comm.nccl_comm.rendezvous_ids == [9]


def test_init_pp_comm_rejects_incompatible_live_topology(monkeypatch):
    existing = _make_pp_comm()
    monkeypatch.setattr(communicator_module, "_pp_comm", existing)
    monkeypatch.setattr(communicator_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(communicator_module, "init_helix_cp_comm", lambda mapping: None)

    class OtherMapping(_FakeMapping):
        pp_group = [0, 2]

    with pytest.raises(RuntimeError, match="different topology"):
        communicator_module.init_pp_comm(OtherMapping())


def test_init_pp_comm_rejects_incompatible_live_helix_topology(monkeypatch):
    existing = _make_pp_comm(_HelixMapping())
    monkeypatch.setattr(communicator_module, "_pp_comm", existing)
    monkeypatch.setattr(communicator_module, "mpi_disabled", lambda: False)

    with pytest.raises(RuntimeError, match="Helix MNNVL CP topology"):
        communicator_module.init_pp_comm(_OtherHelixMapping())


def test_helix_mnnvl_comm_rejects_incompatible_cached_topology(monkeypatch):
    split_calls = []
    comm = object()

    class FakeMpiComm:
        def Split(self, color, key):
            split_calls.append((color, key))
            return comm

    monkeypatch.setattr(mnnvl_module, "mpi_comm", lambda: FakeMpiComm())

    assert mnnvl_module.HelixCpMnnvlMemory.get_comm(_HelixMapping()) is comm
    assert mnnvl_module.HelixCpMnnvlMemory.get_comm(_HelixMapping()) is comm
    with pytest.raises(RuntimeError, match="cannot reuse.*incompatible topology"):
        mnnvl_module.HelixCpMnnvlMemory.get_comm(_OtherHelixMapping())

    assert split_calls == [(0, 1)]


def test_final_release_preserves_helix_comm_topology(monkeypatch):
    original_pp_comm_class = communicator_module.PPCommNCCL
    split_calls = []

    class FakeMpiComm:
        def Split(self, color, key):
            split_calls.append((color, key))
            return object()

    class FakePPComm:
        def __init__(self, mapping):
            self.topology = original_pp_comm_class._mapping_topology(mapping)

        def is_compatible(self, mapping):
            return self.topology == original_pp_comm_class._mapping_topology(mapping)

    monkeypatch.setattr(mnnvl_module, "mpi_comm", lambda: FakeMpiComm())
    monkeypatch.setattr(communicator_module, "PPCommNCCL", FakePPComm)
    monkeypatch.setattr(communicator_module, "NCCL_FAULT_TOLERANCE_ENABLED", False)
    monkeypatch.setattr(communicator_module, "mpi_disabled", lambda: False)

    # Reacquiring the ordinary PP wrapper after its final release is valid when
    # the persistent Helix MPI communicator has the same CP rank topology.
    communicator_module.init_pp_comm(_HelixMapping())
    communicator_module.release_pp_comm()
    communicator_module.init_pp_comm(_HelixMapping())
    communicator_module.release_pp_comm()

    # A later engine may create a fresh PP wrapper, but it cannot reinterpret
    # the process-lifetime Helix MPI communicator with a different CP group.
    with pytest.raises(RuntimeError, match="cannot reuse.*incompatible topology"):
        communicator_module.init_pp_comm(_OtherHelixMapping())

    assert split_calls == [(0, 1)]
    assert communicator_module._pp_comm is None
    assert communicator_module._pp_comm_refcount == 0


def test_default_off_final_release_allows_a_later_incompatible_topology(monkeypatch):
    created = []

    class FakePPComm:
        def __init__(self, mapping):
            self.topology = (mapping.world_size, mapping.rank, tuple(mapping.pp_group))
            created.append(self.topology)

        def is_compatible(self, mapping):
            return self.topology == (mapping.world_size, mapping.rank, tuple(mapping.pp_group))

    class OtherMapping(_FakeMapping):
        pp_group = [0, 2]

    monkeypatch.setattr(communicator_module, "PPCommNCCL", FakePPComm)
    monkeypatch.setattr(communicator_module, "NCCL_FAULT_TOLERANCE_ENABLED", False)
    monkeypatch.setattr(communicator_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(communicator_module, "init_helix_cp_comm", lambda mapping: None)

    communicator_module.init_pp_comm(_FakeMapping())
    communicator_module.release_pp_comm()
    assert communicator_module._pp_comm is None
    communicator_module.init_pp_comm(OtherMapping())

    assert len(created) == 2
    assert created[0] != created[1]


def test_model_engine_cleanup_releases_pp_before_reporting_ub_failure(monkeypatch):
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor import model_engine

    engine = object.__new__(model_engine.PyTorchModelEngine)
    engine._cleanup_done = False
    engine._pp_comm_acquired = True
    engine.model_loader = None
    engine.model = object()
    engine.input_processor = object()
    engine.input_processor_with_hash = object()
    engine.ub_buffers = [SimpleNamespace(addr=123)]
    engine._release_cuda_graphs = lambda: None

    released = []
    gc_calls = []
    monkeypatch.setattr(model_engine, "release_pp_comm", lambda: released.append(True))
    monkeypatch.setattr(model_engine, "release_gc", lambda: gc_calls.append(True))
    monkeypatch.setattr(
        model_engine.ub,
        "ub_deallocate",
        lambda addr: (_ for _ in ()).throw(RuntimeError("injected UB failure")),
    )

    with pytest.raises(RuntimeError, match="userbuffers"):
        engine.cleanup()

    assert released == [True]
    assert gc_calls == [True]
    assert not engine._pp_comm_acquired
    assert not engine._cleanup_done
    assert [buffer.addr for buffer in engine.ub_buffers] == [123]

    monkeypatch.setattr(model_engine.ub, "ub_deallocate", lambda addr: None)
    engine.cleanup()
    assert released == [True]
    assert engine._cleanup_done
    assert engine.ub_buffers is None


def test_ft_mode_uses_engine_local_eager_graph_configuration(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import model_engine

    class FakeTorchCompileConfig:
        def __init__(self, enable_piecewise_cuda_graph):
            self.enable_piecewise_cuda_graph = enable_piecewise_cuda_graph

        def model_copy(self, *, update):
            copied = FakeTorchCompileConfig(self.enable_piecewise_cuda_graph)
            for name, value in update.items():
                setattr(copied, name, value)
            return copied

    class FakeLlmArgs:
        def __init__(self, cuda_graph_config, torch_compile_config):
            self.cuda_graph_config = cuda_graph_config
            self.torch_compile_config = torch_compile_config

        def model_copy(self, *, update):
            copied = FakeLlmArgs(self.cuda_graph_config, self.torch_compile_config)
            for name, value in update.items():
                setattr(copied, name, value)
            return copied

    original = FakeLlmArgs(object(), FakeTorchCompileConfig(True))
    monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", "1")

    effective = model_engine._force_eager_mode_for_nccl_fault_tolerance(original)

    assert effective is not original
    assert effective.cuda_graph_config is None
    assert not effective.torch_compile_config.enable_piecewise_cuda_graph
    # The caller-owned arguments remain reusable by a non-FT engine.
    assert original.cuda_graph_config is not None
    assert original.torch_compile_config.enable_piecewise_cuda_graph


def test_ft_mode_uses_eager_allreduce_autotuning(monkeypatch):
    from tensorrt_llm._torch.custom_ops import torch_custom_ops

    monkeypatch.delenv("TLLM_FAULT_TOLERANCE_MODE", raising=False)
    assert torch_custom_ops._use_cuda_graph_for_allreduce_tuning()

    monkeypatch.setenv("TLLM_FAULT_TOLERANCE_MODE", "1")
    assert not torch_custom_ops._use_cuda_graph_for_allreduce_tuning()


def test_partially_constructed_engine_cleanup_releases_pp(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import model_engine

    # Simulate a constructor failure immediately after init_pp_comm(), before
    # model/model_loader/input-processor attributes have been populated.
    engine = object.__new__(model_engine.PyTorchModelEngine)
    engine._cleanup_done = False
    engine._pp_comm_acquired = True

    released = []
    gc_calls = []
    monkeypatch.setattr(model_engine, "release_pp_comm", lambda: released.append(True))
    monkeypatch.setattr(model_engine, "release_gc", lambda: gc_calls.append(True))

    engine.cleanup()

    assert released == [True]
    assert gc_calls == [True]
    assert not engine._pp_comm_acquired
    assert engine._cleanup_done
