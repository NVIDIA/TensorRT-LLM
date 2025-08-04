import ctypes

import cuda.bindings.runtime as cudart
import torch


# Simple wrapper - you need this because CUDA needs a C structure
class Struct(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_int),
    ]


def hostfunc(userData):
    data = Struct.from_address(userData)
    print(f"Hello, World! {data.a}")
    return 0


HostFn_t = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


def main():
    data = Struct(a=1)

    # ctypes is managing the pointer value for us
    c_hostfunc = HostFn_t(hostfunc)
    cuda_hostfunc = cudart.cudaHostFn_t(_ptr=ctypes.addressof(c_hostfunc))

    # Run
    stream = torch.cuda.Stream()
    cudart_stream = cudart.cudaStream_t(stream.cuda_stream)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        (err, ) = cudart.cudaLaunchHostFunc(cudart_stream, cuda_hostfunc,
                                            ctypes.addressof(data))
        assert err == cudart.cudaError_t.cudaSuccess
    torch.cuda.synchronize()
    print("Graph captured", flush=True)

    with torch.cuda.stream(stream):
        for i in range(2):
            print(f"Replay {i}", flush=True)
            g.replay()
            torch.cuda.synchronize()

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
