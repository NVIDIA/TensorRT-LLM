import atexit

import torch

from ..bindings.internal import runtime as bindings

HOSTFUNC_USER_DATA_HANDLES = set()


def launch_hostfunc(hostfunc, *args, **kwargs):
    stream = torch.cuda.current_stream()
    is_capturing = torch.cuda.is_current_stream_capturing()
    handle = bindings.launch_hostfunc(stream.cuda_stream, not is_capturing,
                                      hostfunc, *args, **kwargs)
    if is_capturing:
        HOSTFUNC_USER_DATA_HANDLES.add(handle)
    else:
        assert handle is None
    return handle


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
