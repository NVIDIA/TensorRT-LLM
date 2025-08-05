import atexit

import torch

from ..bindings.internal import runtime as bindings

HOSTFUNC_HANDLES = set()


def launch_hostfunc(hostfunc, *args, **kwargs):
    stream = torch.cuda.current_stream()
    handle = bindings.launch_hostfunc(stream.cuda_stream, hostfunc, *args,
                                      **kwargs)
    HOSTFUNC_HANDLES.add(handle)
    return handle


def hostfunc(hostfunc):

    def wrapper(*args, **kwargs):
        return launch_hostfunc(hostfunc, *args, **kwargs)

    return wrapper


def free_hostfunc_user_data(handle: int):
    if handle not in HOSTFUNC_HANDLES:
        raise ValueError(f"Host function handle {handle} not found.")
    bindings.free_hostfunc_user_data(handle)
    HOSTFUNC_HANDLES.remove(handle)


def free_all_hostfunc_user_data():
    for handle in HOSTFUNC_HANDLES:
        print(f"Freeing host function handle {handle}")
        bindings.free_hostfunc_user_data(handle)
    HOSTFUNC_HANDLES.clear()


atexit.register(free_all_hostfunc_user_data)
