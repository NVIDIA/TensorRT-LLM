import atexit

import torch

from ..bindings.internal import runtime as bindings

HOSTFUNC_USER_DATA_HANDLES = set()

_low_latency_dispatch: bool = False

# Event recorded after the last replay of a CUDA graph that contains host
# functions; None once a drain has waited for it.
_replay_event: torch.cuda.Event | None = None


def set_low_latency_dispatch(enabled: bool) -> None:
    global _low_latency_dispatch
    _low_latency_dispatch = enabled


def launch_hostfunc(hostfunc, *args, **kwargs):
    stream = torch.cuda.current_stream()
    is_capturing = torch.cuda.is_current_stream_capturing()
    handle = bindings.launch_hostfunc(stream.cuda_stream, not is_capturing,
                                      _low_latency_dispatch, hostfunc, *args,
                                      **kwargs)
    if is_capturing:
        HOSTFUNC_USER_DATA_HANDLES.add(handle)
    else:
        assert handle is None
    return handle


def record_hostfunc_graph_replay() -> None:
    """Record an event after replaying a graph that contains host functions."""
    global _replay_event
    _replay_event = torch.cuda.Event()
    _replay_event.record()


def hostfuncs_pending() -> bool:
    """Return True if a graph with host functions has replayed since the last drain.

    Only host-side state is read, not device progress, so ranks that issue the
    same replays and drains reach the same answer.
    """
    return _replay_event is not None


def drain_captured_hostfuncs() -> None:
    """Wait until the host functions of the last replayed graph have run.

    A host function takes the GIL when it runs. A call that holds the GIL while
    it waits on the device, such as a multimodal encoder that builds a cuDNN
    plan, would otherwise deadlock with a host function that has not run yet.
    The wait is skipped during stream capture, where host synchronization is
    not allowed.
    """
    global _replay_event
    if _replay_event is None or torch.cuda.is_current_stream_capturing():
        return
    # Event.synchronize releases the GIL, which lets the host functions run.
    _replay_event.synchronize()
    _replay_event = None


def hostfunc(hostfunc):

    def wrapper(*args, **kwargs):
        return launch_hostfunc(hostfunc, *args, **kwargs)

    return wrapper


def free_hostfunc_user_data(handle: int):
    if handle not in HOSTFUNC_USER_DATA_HANDLES:
        raise ValueError(f"Hostfunc user data handle {handle} not found.")
    bindings.free_hostfunc_user_data(handle)
    HOSTFUNC_USER_DATA_HANDLES.remove(handle)


def free_all_hostfunc_user_data():
    for handle in HOSTFUNC_USER_DATA_HANDLES:
        bindings.free_hostfunc_user_data(handle)
    HOSTFUNC_USER_DATA_HANDLES.clear()


atexit.register(free_all_hostfunc_user_data)
