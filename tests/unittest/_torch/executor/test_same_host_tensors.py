# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request transport tests for same-host tensor sharing.

Multi-rank cases run on four MPI pool workers through ``mpi_pool_executor``.
Rank layouts cover TP, TP with CP, attention-DP-style world broadcasts, and
pipeline stages, and a patched hostname puts ranks on one or two hosts. CPU
cases are marked ``cpu_only``. CUDA cases need a visible GPU and run on GPU
stages.
"""

import contextlib
import faulthandler
import gc
import os
import pickle
import sys
import threading
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import patch

# An importable namedtuple, so peers can unpickle it.
from urllib.parse import DefragResult as Metadata

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.distributed.communicator import MPIDist
from tensorrt_llm._torch.pyexecutor import same_host_tensors
from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
from tensorrt_llm._torch.pyexecutor.llm_request import ExecutorRequest
from tensorrt_llm._torch.pyexecutor.request_utils import (
    RequestBroadcaster,
    attach_py_objects_to_requests,
)
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.bindings import executor as tle
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD_SIZE = 4
WORKER_TIMEOUT_S = 300
ON_FOUR_WORKERS = pytest.mark.parametrize("mpi_pool_executor", [WORLD_SIZE], indirect=True)
LARGE = same_host_tensors._MIN_SHARED_TENSOR_BYTES

LAYOUTS: Dict[str, Dict[str, int]] = {
    "tp4": dict(tp_size=4),
    "tp2cp2": dict(tp_size=2, cp_size=2),
    "pp2tp2": dict(tp_size=2, pp_size=2),
    "pp2cp2": dict(cp_size=2, pp_size=2),
    "pp4": dict(pp_size=4),
}
HOSTS: Dict[str, Callable[[int], str]] = {
    "one_host": lambda rank: "host-0",
    # Pipeline stage 0 on one host and stage 1 on another (for pp2*).
    "by_half": lambda rank: "host-" + str(rank // 2),
    # Every group spans both hosts; pipeline hops of pp2* stay on one host.
    "interleaved": lambda rank: "host-" + str(rank % 2),
}


class Pause:
    pause = staticmethod(contextlib.nullcontext)


class Context:
    """Per-worker MPI state, built inside the worker and never at import time."""

    def __init__(self, device: str = "cpu", layout: str = "tp4", hosts: str = "one_host") -> None:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.device = device
        size = self.comm.Get_size()
        self.mapping = Mapping(world_size=size, rank=self.rank, **LAYOUTS[layout])
        self.dist = MPIDist(self.mapping)
        self.hosts = [HOSTS[hosts](rank) for rank in range(size)]
        self.pids = self.comm.allgather(os.getpid())

    def broadcaster(self, handles: bool = True) -> RequestBroadcaster:
        broadcaster = RequestBroadcaster(self.dist, Pause())
        assert broadcaster.same_host_tensors is not None
        if not handles:
            broadcaster.same_host_tensors = None
        return broadcaster

    def hostname(self) -> Any:
        return patch("socket.gethostname", return_value=self.hosts[self.rank])

    def broadcast(self, broadcaster: RequestBroadcaster, requests: List) -> Any:
        with self.hostname():
            return broadcaster.broadcast(requests)

    def holds_shared_tensors(self) -> bool:
        """Whether this rank's request tensors live in shared memory.

        They do when the hop that delivered them mapped handles, or when this
        rank exported them to the next pipeline stage on its own host.
        """
        mapping, hosts, rank = self.mapping, self.hosts, self.rank
        if rank == 0:
            return False
        stage_size = mapping.tp_size * mapping.cp_size
        if mapping.pp_rank + 1 < mapping.pp_size and hosts[rank + stage_size] == hosts[rank]:
            return True
        if mapping.pp_size == 1:
            return len(set(hosts)) == 1
        if mapping.pp_rank == 0:
            group = mapping.cp_group if mapping.cp_size > 1 else mapping.tp_group
            return all(hosts[member] == hosts[group[0]] for member in group)
        return hosts[rank - stage_size] == hosts[rank]

    def sync(self) -> None:
        gc.collect()
        self.comm.Barrier()
        if self.device != "cpu":
            torch.cuda.synchronize(0)
            torch.cuda.ipc_collect()
            self.comm.Barrier()

    def assert_clean(self) -> None:
        """No rank owns leftover shared-memory files once every rank drops its views."""
        self.sync()
        deadline = time.monotonic() + 15
        owned = []
        if self.rank == 0:
            while time.monotonic() < deadline:
                owned = [
                    path
                    for pid in self.pids
                    for path in Path("/dev/shm").glob("torch_" + str(pid) + "_*")
                ]
                if not owned:
                    break
                gc.collect()
                time.sleep(0.05)
        owned = self.comm.bcast([str(path) for path in owned], root=0)
        assert not owned, owned


def _guarded(worker: Callable[..., None], *args: Any) -> None:
    """Run one rank's part; exit the worker if peers never finish a collective.

    A failed assertion on one rank leaves the others waiting in a collective.
    The watchdog turns that hang into a broken pool, so the test fails fast.
    """

    def expire() -> None:
        faulthandler.dump_traceback(all_threads=True)
        os._exit(1)

    watchdog = threading.Timer(WORKER_TIMEOUT_S, expire)
    watchdog.daemon = True
    watchdog.start()
    try:
        worker(*args)
    except BaseException:
        # Peers may hang in a collective, so report the failure right away.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        watchdog.cancel()


def _run(executor: Any, worker: Callable[..., None], *args: Any) -> None:
    results = executor.starmap(_guarded, [(worker, *args)] * WORLD_SIZE)
    assert list(results) == [None] * WORLD_SIZE


def _large(dtype: torch.dtype, device: str, seed: int = 0) -> torch.Tensor:
    """A tensor whose storage is exactly the sharing threshold."""
    numel = LARGE // torch.empty((), dtype=dtype).element_size()
    return (torch.arange(seed, seed + numel) % 251).to(dtype=dtype, device=device)


def _large_tensors(device: str) -> List[torch.Tensor]:
    dtypes = (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    tensors = [_large(dtype, device, seed) for seed, dtype in enumerate(dtypes)]
    # A strided view is shared with its whole storage.
    tensors.append(_large(torch.float32, device, 20).reshape(256, -1)[1::2, 3::5])
    # NaN payloads, signed zero and infinity keep their bits.
    special = (
        torch.tensor([0x7FC00123, 0x80000000, 0x7F800000], dtype=torch.int64)
        .to(torch.int32)
        .view(torch.float32)
    )
    tensors.append(special.repeat(LARGE // special.nbytes + 1).to(device))
    return tensors


def _multimodal_data(device: str) -> dict:
    values = _large_tensors(device)
    return dict(
        image=dict(
            pixel_values=values[0],
            image_grid_thw=torch.tensor([[1, 4, 4]], device=device),
        ),
        audio=dict(input_features=values[1], empty=torch.empty(0, device=device)),
        video=values[2:],
        mrope_config=dict(
            mrope_position_ids=torch.arange(3 * 64, device=device).reshape(3, 64),
        ),
        # The same tensor twice is sent once and maps to one view.
        alias=values[0],
        # Large tensors that are not shared are pickled as before.
        unshared=dict(
            complex=torch.zeros(LARGE // 8, dtype=torch.complex64, device=device),
            negative=torch._neg_view(_large(torch.float32, device, 30)),
        ),
        metadata=[1, None, "keep"],
    )


def _request(device: str) -> List[RequestQueueItem]:
    """One request with large tensors in every reachable place."""
    request = ExecutorRequest(
        input_token_ids=[1, 2],
        max_tokens=4,
        embedding_bias=_large(torch.float32, "cpu", 40),
        prompt_tuning_config=tle.PromptTuningConfig(
            _large(torch.float16, "cpu", 41).reshape(-1, 64)
        ),
        # A LoRA config is a few bytes, so it stays inline.
        lora_config=tle.LoraConfig(task_id=1, config=torch.ones(8, 3, dtype=torch.int32)),
    )
    request.py_multimodal_data = _multimodal_data(device)
    request.py_conversation_params = None
    request.py_disaggregated_params = DisaggregatedParams(
        first_gen_logits=[_large(torch.float32, device, 42)]
    )
    request.py_logits_post_processors = [torch.add]
    return [RequestQueueItem(id=41, request=request)]


# Distinct tensors that ``_request`` sends as handles.
SHARED_PER_REQUEST = len(_large_tensors("cpu")) + 3
# Unsupported dtype and negative view stay inline.
INLINE_LARGE_PER_REQUEST = 2


def _plain_request(data: Any) -> List[RequestQueueItem]:
    request = ExecutorRequest(input_token_ids=[1, 2], max_tokens=4)
    request.py_multimodal_data = data
    request.py_conversation_params = None
    return [RequestQueueItem(id=41, request=request)]


def _bits(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach()
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)
    return tensor.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _same(left: torch.Tensor, right: torch.Tensor) -> None:
    layout = (left.dtype, left.shape, left.stride())
    assert layout == (right.dtype, right.shape, right.stride()), (layout, right.stride())
    assert _bits(left) == _bits(right), layout


def _check(ctx: Context, requests: List, objects: Any, handles: bool) -> None:
    """Compare received requests with rank 0's and check what was mapped."""
    attach_py_objects_to_requests(requests, objects)
    expected = _request(ctx.device)[0].request
    request = requests[0].request
    data, reference = request.py_multimodal_data, expected.py_multimodal_data
    shared = [
        data["image"]["pixel_values"],
        data["audio"]["input_features"],
        *data["video"],
        request.py_disaggregated_params.first_gen_logits[0],
    ]
    expected_shared = [
        reference["image"]["pixel_values"],
        reference["audio"]["input_features"],
        *reference["video"],
        expected.py_disaggregated_params.first_gen_logits[0],
    ]
    inline = [
        data["image"]["image_grid_thw"],
        data["audio"]["empty"],
        data["mrope_config"]["mrope_position_ids"],
        *data["unshared"].values(),
    ]
    expected_inline = [
        reference["image"]["image_grid_thw"],
        reference["audio"]["empty"],
        reference["mrope_config"]["mrope_position_ids"],
        *reference["unshared"].values(),
    ]
    for left, right in zip(expected_shared + expected_inline, shared + inline):
        _same(left, right)
        if right.is_cuda:
            # Consumers may use a different GPU from the exporting rank.
            assert _bits(right.to(torch.cuda.current_device())) == _bits(left)
    assert data["alias"] is data["image"]["pixel_values"]
    assert data["metadata"] == [1, None, "keep"]
    _same(expected.embedding_bias, request.embedding_bias)
    _same(
        expected.prompt_tuning_config.embedding_table,
        request.prompt_tuning_config.embedding_table,
    )
    _same(expected.lora_config.config, request.lora_config.config)
    if ctx.rank == 0:
        return
    # CUDA tensors always report shared storage, so check CPU tensors only.
    if ctx.device == "cpu":
        flags = [tensor.is_shared() for tensor in shared]
        assert flags == [handles] * len(shared), (ctx.rank, handles, flags)
    assert not any(tensor.is_shared() for tensor in inline if not tensor.is_cuda)


def _layout_worker(layout: str, hosts: str, device: str = "cpu") -> None:
    ctx = Context(device, layout, hosts)
    broadcaster = ctx.broadcaster()
    # The second batch reuses the hop decisions of the first one.
    for _ in range(2):
        requests = _request(device) if ctx.rank == 0 else []
        requests, objects = ctx.broadcast(broadcaster, requests)
        _check(ctx, requests, objects, ctx.holds_shared_tensors())
        del requests, objects
        ctx.assert_clean()


def _bytes_path_worker() -> None:
    ctx = Context()
    requests = _request("cpu") if ctx.rank == 0 else []
    requests, objects = ctx.broadcast(ctx.broadcaster(handles=False), requests)
    _check(ctx, requests, objects, handles=False)
    del requests, objects
    ctx.assert_clean()


def _wire(ctx: Context, handles: bool = True) -> List[Any]:
    """Return what rank 0 broadcasts for one request, and its pickled size."""
    broadcaster = ctx.broadcaster(handles)
    requests = _request("cpu") if ctx.rank == 0 else []
    payloads = []
    original = ctx.dist.broadcast

    def capture(value: Any, **kwargs: Any) -> Any:
        if ctx.rank == 0:
            size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            payloads.append((value[0] is requests, size))
        return original(value, **kwargs)

    with patch.object(ctx.dist, "broadcast", capture):
        received, objects = ctx.broadcast(broadcaster, requests)
    _check(ctx, received, objects, handles and ctx.holds_shared_tensors())
    return payloads


def _wire_worker() -> None:
    ctx = Context()
    exports = 0
    original = SharedTensorContainer.from_tensor

    def count(*args: Any, **kwargs: Any) -> Any:
        nonlocal exports
        exports += 1
        return original(*args, **kwargs)

    with patch.object(SharedTensorContainer, "from_tensor", count):
        default = _wire(ctx)
    byte_transport = _wire(ctx, handles=False)
    # With a receiver on another host, the world broadcast sends rank 0's
    # objects to the byte transport unchanged.
    remote = _wire(Context(hosts="by_half"))
    if ctx.rank == 0:
        # Every distinct large tensor is exported once per receiving rank.
        assert exports == SHARED_PER_REQUEST * (WORLD_SIZE - 1), exports
        # Only the large tensors that cannot be shared stay in the payload.
        inline = LARGE * (INLINE_LARGE_PER_REQUEST + 1)
        assert default[0][1] < inline < LARGE * SHARED_PER_REQUEST < byte_transport[0][1]
        assert [passthrough for passthrough, _ in default] == [False]
        assert [passthrough for passthrough, _ in byte_transport + remote] == [True, True]
        assert remote[0][1] == byte_transport[0][1]
    else:
        assert exports == 0
    ctx.assert_clean()


def _metadata_worker() -> None:
    ctx = Context()
    for handles in (False, True):
        broadcaster = ctx.broadcaster(handles)
        requests = []
        if ctx.rank == 0:
            requests = _plain_request(
                dict(
                    image=_large(torch.float32, "cpu"),
                    metadata=Metadata("image", torch.Size((2, 3))),
                )
            )
            requests[0].request.py_scheduling_params = Metadata("schedule", torch.Size((4,)))
        requests, objects = ctx.broadcast(broadcaster, requests)
        attach_py_objects_to_requests(requests, objects)
        metadata = requests[0].request.py_multimodal_data["metadata"]
        other = requests[0].request.py_scheduling_params
        for value, label, size in ((metadata, "image", (2, 3)), (other, "schedule", (4,))):
            assert isinstance(value, Metadata) and isinstance(value[1], torch.Size), (
                type(value),
                type(value[1]),
            )
            assert value[0] == label and value[1] == size
        del requests, objects, metadata, other
        ctx.assert_clean()


def _last_consumer_worker() -> None:
    ctx = Context()
    requests = _request("cpu") if ctx.rank == 0 else []
    requests, objects = ctx.broadcast(ctx.broadcaster(), requests)
    if ctx.rank == WORLD_SIZE - 1:
        # Other ranks discard their request while the final consumer keeps reading.
        time.sleep(0.2)
        _check(ctx, requests, objects, True)
    del requests, objects
    ctx.assert_clean()


def _failure_falls_back(
    ctx: Context, method: str, failing_rank: int, hop: Any, other_hops: List[Any] = ()
) -> None:
    """A failed export or mapping sends that hop as bytes and disables it."""
    broadcaster = ctx.broadcaster()
    calls = 0
    original = getattr(SharedTensorContainer, method)

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2 and ctx.rank == failing_rank:
            raise RuntimeError("Deliberate failure")
        return original(*args, **kwargs)

    requests = _request(ctx.device) if ctx.rank == 0 else []
    with patch.object(SharedTensorContainer, method, fail_once):
        requests, objects = ctx.broadcast(broadcaster, requests)
    attach_py_objects_to_requests(requests, objects)
    expected = _request(ctx.device)[0].request.py_multimodal_data
    actual = requests[0].request.py_multimodal_data
    assert _bits(actual["image"]["pixel_values"]) == _bits(expected["image"]["pixel_values"])
    assert [_bits(tensor) for tensor in actual["video"]] == [
        _bits(tensor) for tensor in expected["video"]
    ]
    enabled = broadcaster.same_host_tensors.enabled
    if hop in enabled:
        assert enabled[hop] is False, enabled
    for other in other_hops:
        if other in enabled:
            assert enabled[other] is True, enabled
    if not other_hops:
        # Later batches use the byte path without exporting anything.
        requests = _request(ctx.device) if ctx.rank == 0 else []
        with patch.object(SharedTensorContainer, "from_tensor", side_effect=AssertionError):
            ctx.broadcast(broadcaster, requests)
    del requests, objects, actual
    ctx.assert_clean()


def _failure_worker(method: str) -> None:
    ctx = Context()
    failing_rank = 0 if method == "dump_to_dict" else 2
    _failure_falls_back(ctx, method, failing_rank, ("world", 0))


def _pipeline_failure_worker(method: str) -> None:
    # Rank 1 forwards the views it mapped from rank 0 to rank 3.
    ctx = Context(layout="pp2tp2")
    failing_rank = 1 if method == "dump_to_dict" else 3
    hop = ("send", 3) if ctx.rank == 1 else ("recv", 1)
    _failure_falls_back(ctx, method, failing_rank, hop, [("tp", 0), ("send", 2), ("recv", 0)])


def _serialization_abort(ctx: Context) -> None:
    broadcaster = ctx.broadcaster()
    requests = _plain_request(dict(image=_large(torch.float32, ctx.device)))
    if ctx.rank == 0:
        requests[0].request.py_logits_post_processors = lambda value: value
    else:
        requests = []
    failed = False
    try:
        ctx.broadcast(broadcaster, requests)
    except RuntimeError:
        failed = True
    assert failed, "Serialization failure must reach every rank"
    requests.clear()
    ctx.assert_clean()


def _serialization_abort_worker() -> None:
    _serialization_abort(Context())


def _retirement_cycles_worker(layout: str) -> None:
    ctx = Context(layout=layout)
    broadcaster = ctx.broadcaster()
    fds = []
    for cycle in range(20):
        requests = _request("cpu") if ctx.rank == 0 else []
        requests, objects = ctx.broadcast(broadcaster, requests)
        if cycle % 5 == 0:
            _check(ctx, requests, objects, ctx.holds_shared_tensors())
        del requests, objects
        ctx.assert_clean()
        fds.append(len(list(Path("/proc/self/fd").iterdir())))
    assert max(fds[2:]) <= fds[1], fds


def _no_tensor_payload_worker(data: dict | None) -> None:
    ctx = Context()
    requests = _plain_request(data) if ctx.rank == 0 else []
    with patch.object(SharedTensorContainer, "from_tensor", side_effect=AssertionError):
        requests, objects = ctx.broadcast(ctx.broadcaster(), requests)
    attach_py_objects_to_requests(requests, objects)
    if data is None:
        assert "py_multimodal_data" not in dict(objects)
    else:
        actual = requests[0].request.py_multimodal_data
        assert actual.keys() == data.keys()
        assert actual["metadata"] == data["metadata"]
        assert torch.equal(actual["small"], data["small"])


def _view_failure_worker() -> None:
    # CPU from_dict has consumed the exported reference before view creation.
    # A failing view must not make rank 0 release that reference a second time.
    ctx = Context()
    released = 0

    def record_release(_container: SharedTensorContainer) -> None:
        nonlocal released
        released += 1

    with patch.object(SharedTensorContainer, "release_unconsumed", record_release):
        _failure_falls_back(ctx, "get_local_view", 2, ("world", 0))
    released = ctx.comm.bcast(released if ctx.rank == 0 else None, root=0)
    assert released == 0, "An opened CPU storage reference was released twice"


def _async_pipeline_worker() -> None:
    ctx = Context(layout="pp2tp2")
    broadcaster = ctx.broadcaster()
    requests = _request("cpu") if ctx.rank == 0 else []
    with patch.dict(os.environ, {"TRTLLM_PP_REQ_SEND_ASYNC": "1"}):
        requests, objects = ctx.broadcast(broadcaster, requests)
    # Asynchronous pipeline sends keep the byte transport; stage 0 still maps.
    _check(ctx, requests, objects, ctx.rank == 1)
    if broadcaster.send_requests_handler is not None:
        broadcaster.send_requests_handler.wait()
    del requests, objects
    ctx.assert_clean()


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ["partial", "bound", "builtin_bound", "parameter", "attributes"])
def test_private_state_preserves_aliases(kind: str) -> None:
    tensor = _large(torch.float32, "cpu")
    if kind == "partial":
        state = partial(torch.add, tensor)
    elif kind == "bound":
        state = tensor.add
    elif kind == "builtin_bound":
        state = [tensor].append
    elif kind == "parameter":
        state = torch.nn.Parameter(tensor)
    else:
        tensor.metadata = {"value": "keep"}
        state = tensor
    payload = [tensor, state, state]
    exporter, message = same_host_tensors._dump(payload, 1)
    assert exporter is None
    assert message == (None, payload, [], None)
    restored = pickle.loads(pickle.dumps(message[1]))
    assert restored[1] is restored[2]
    if kind == "partial":
        assert restored[1].args[0] is restored[0]
    elif kind == "bound":
        assert restored[1].__self__ is restored[0]
    elif kind == "builtin_bound":
        assert restored[1].__self__[0] is restored[0]
    elif kind == "attributes":
        assert restored[1].metadata == {"value": "keep"}
    else:
        assert isinstance(restored[1], torch.nn.Parameter)
    assert not restored[0].is_shared()


@pytest.mark.cpu_only
@pytest.mark.skipif(not hasattr(torch.Tensor, "refine_names"), reason="Named tensors unavailable")
def test_named_tensor_keeps_pickle_error() -> None:
    tensor = _large(torch.float32, "cpu").refine_names("features")
    exporter, message = same_host_tensors._dump([tensor], 1)
    assert exporter is None
    assert message == (None, [tensor], [], None)
    with pytest.raises(RuntimeError, match="[Nn]amed tensors"):
        pickle.dumps(message[1])


def _private_payload(same_processor: bool) -> dict:
    tensor = _large(torch.float32, "cpu")
    state = {"calls": iter([1, 2]), "tensor": tensor}
    first = partial(next, state["calls"])
    second = first if same_processor else partial(next, state["calls"])
    return {"image": tensor, "processors": [first, second], "metadata": state}


def _private_state_worker(same_processor: bool) -> None:
    ctx = Context()
    broadcaster = ctx.broadcaster()
    requests = _plain_request(_private_payload(same_processor)) if ctx.rank == 0 else []
    requests, objects = ctx.broadcast(broadcaster, requests)
    attach_py_objects_to_requests(requests, objects)
    data = requests[0].request.py_multimodal_data
    first, second = data["processors"]
    assert (first is second) == same_processor
    assert first.args[0] is second.args[0] is data["metadata"]["calls"]
    assert data["metadata"]["tensor"] is data["image"]
    assert [first(), second()] == [1, 2]
    if ctx.rank > 0:
        assert not data["image"].is_shared()
    assert broadcaster.same_host_tensors.enabled[("world", 0)] is True
    del requests, objects, data, first, second
    ctx.assert_clean()

    # Normal private-state fallback does not turn off sharing on the hop.
    requests = _plain_request({"image": _large(torch.float32, "cpu")}) if ctx.rank == 0 else []
    requests, objects = ctx.broadcast(broadcaster, requests)
    attach_py_objects_to_requests(requests, objects)
    data = requests[0].request.py_multimodal_data
    if ctx.rank > 0:
        assert data["image"].is_shared()
    assert broadcaster.same_host_tensors.enabled[("world", 0)] is True
    del requests, objects, data
    ctx.assert_clean()


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
@pytest.mark.parametrize("same_processor", [True, False], ids=["repeated", "shared_state"])
def test_private_state_fallback_keeps_hop_enabled(
    mpi_pool_executor: Any, same_processor: bool
) -> None:
    _run(mpi_pool_executor, _private_state_worker, same_processor)


@pytest.mark.cpu_only
def test_activation() -> None:
    class Unsupported(MPIDist):
        """The gate reads the capability, not the transport class."""

        supports_local_tensor_handles = False

    for tp, pp, cp in [(1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 1, 2), (1, 2, 2), (4, 1, 1)]:
        mapping = Mapping(world_size=tp * pp * cp, rank=0, tp_size=tp, pp_size=pp, cp_size=cp)
        enabled = tp * pp * cp > 1
        broadcaster = RequestBroadcaster(MPIDist(mapping), Pause())
        assert (broadcaster.same_host_tensors is not None) == enabled
        if enabled and pp == 1:
            # The single-round path would bypass the hop dispatch.
            assert not broadcaster._single_round(False)
        assert RequestBroadcaster(Unsupported(mapping), Pause()).same_host_tensors is None


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
@pytest.mark.parametrize("hosts", list(HOSTS))
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_layouts(mpi_pool_executor: Any, layout: str, hosts: str) -> None:
    _run(mpi_pool_executor, _layout_worker, layout, hosts)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_bytes_path(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _bytes_path_worker)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_wire_and_remote_fallback(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _wire_worker)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_metadata_types(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _metadata_worker)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_last_consumer_lifetime(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _last_consumer_worker)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
@pytest.mark.parametrize("method", ["dump_to_dict", "from_dict"], ids=["export", "mapping"])
def test_failure_falls_back_to_bytes(mpi_pool_executor: Any, method: str) -> None:
    _run(mpi_pool_executor, _failure_worker, method)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
@pytest.mark.parametrize("method", ["dump_to_dict", "from_dict"], ids=["export", "mapping"])
def test_pipeline_failure_falls_back_to_bytes(mpi_pool_executor: Any, method: str) -> None:
    _run(mpi_pool_executor, _pipeline_failure_worker, method)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_serialization_abort(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _serialization_abort_worker)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
@pytest.mark.parametrize("layout", ["tp4", "pp2tp2"])
def test_retirement_cycles(mpi_pool_executor: Any, layout: str) -> None:
    _run(mpi_pool_executor, _retirement_cycles_worker, layout)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
@pytest.mark.parametrize(
    "data",
    [None, {"metadata": [1, "image", None], "small": torch.arange(LARGE // 16)}],
    ids=["none", "small_tensor"],
)
def test_requests_without_large_tensors(mpi_pool_executor: Any, data: dict | None) -> None:
    _run(mpi_pool_executor, _no_tensor_payload_worker, data)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_view_failure_after_storage_open(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _view_failure_worker)


@pytest.mark.cpu_only
@ON_FOUR_WORKERS
def test_async_pipeline_send_uses_bytes(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _async_pipeline_worker)


def _imported_cuda_roundtrip(ctx: Context) -> None:
    """Rank 0 re-shares a CUDA tensor it imported from another process."""
    source = _large(torch.int64, "cuda:0")
    handle = None
    if ctx.rank == 1:
        handle = SharedTensorContainer.from_tensor(source).dump_to_dict()
    handle = ctx.comm.bcast(handle, root=1)
    requests = []
    if ctx.rank == 0:
        imported = SharedTensorContainer.from_dict(handle).get_local_view()
        requests = _plain_request(dict(image=imported))
    requests, objects = ctx.broadcast(ctx.broadcaster(), requests)
    attach_py_objects_to_requests(requests, objects)
    assert torch.equal(requests[0].request.py_multimodal_data["image"].to("cuda:0"), source)
    del requests, objects
    ctx.comm.Barrier()


def _cuda_worker() -> None:
    torch.cuda.set_device(MPI.COMM_WORLD.Get_rank() % torch.cuda.device_count())
    ctx = Context("cuda:0")
    for handles in (False, True):
        broadcaster = ctx.broadcaster(handles)
        requests = _request(ctx.device) if ctx.rank == 0 else []
        requests, objects = ctx.broadcast(broadcaster, requests)
        if ctx.rank == WORLD_SIZE - 1:
            time.sleep(0.2)
        _check(ctx, requests, objects, handles)
        del requests, objects
        ctx.assert_clean()
    expanded = _large(torch.int64, "cuda:0")[:4].expand(4, 4)
    requests = _plain_request(dict(image=expanded)) if ctx.rank == 0 else []
    requests, objects = ctx.broadcast(ctx.broadcaster(), requests)
    assert torch.equal(dict(objects)["py_multimodal_data"][41]["image"], expanded)
    # The exported block keeps its CUDA IPC reference counter file until freed.
    del requests, objects, expanded
    _imported_cuda_roundtrip(ctx)
    _failure_falls_back(ctx, "dump_to_dict", 0, ("world", 0))
    _failure_falls_back(ctx, "from_dict", 2, ("world", 0))
    _serialization_abort(ctx)
    broadcaster = ctx.broadcaster()
    allocations, fds = [], []
    for cycle in range(20):
        requests = _request(ctx.device) if ctx.rank == 0 else []
        requests, objects = ctx.broadcast(broadcaster, requests)
        if cycle % 5 == 0:
            _check(ctx, requests, objects, ctx.rank > 0)
        del requests, objects
        ctx.assert_clean()
        # ipc_collect frees rank 0 blocks that peers released. Without it, the
        # caching allocator frees them on its next out-of-memory retry.
        allocations.append(torch.cuda.memory_allocated(0))
        fds.append(len(list(Path("/proc/self/fd").iterdir())))
    assert max(allocations[2:]) <= allocations[1], allocations
    assert max(fds[2:]) <= fds[1], fds


def _cuda_layout_worker(layout: str, hosts: str) -> None:
    torch.cuda.set_device(MPI.COMM_WORLD.Get_rank() % torch.cuda.device_count())
    # Pipeline forwarders cannot export imported CUDA views and send copies.
    _layout_worker(layout, hosts, "cuda:0")


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires a visible CUDA device")
@ON_FOUR_WORKERS
def test_cuda_transport(mpi_pool_executor: Any) -> None:
    _run(mpi_pool_executor, _cuda_worker)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires a visible CUDA device")
@ON_FOUR_WORKERS
@pytest.mark.parametrize(
    "layout,hosts", [("pp2tp2", "one_host"), ("pp4", "one_host"), ("pp2tp2", "interleaved")]
)
def test_cuda_layouts(mpi_pool_executor: Any, layout: str, hosts: str) -> None:
    _run(mpi_pool_executor, _cuda_layout_worker, layout, hosts)
