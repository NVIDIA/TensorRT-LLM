import time

import cupy as cp
import torch


def device_func(x, y):
    torch.matmul(x, x, out=y)


def my_callback(data):
    print("Host callback says:", data)


def main():
    x = torch.randn(4096, 4096, device='cuda')
    y = torch.empty_like(x)
    device_func(x, y)

    data = "Hello Host Node"
    stream = cp.cuda.Stream(non_blocking=True)
    stream_pytorch = torch.cuda.ExternalStream(stream.ptr)

    with stream, stream_pytorch:
        stream.begin_capture()
        device_func(x, y)
        stream.launch_host_func(my_callback, data)
        g = stream.end_capture()

    torch.cuda.synchronize()
    print("Graph captured", flush=True)

    for i in range(5):
        with stream, stream_pytorch:
            x.copy_(torch.ones_like(x))
            y.copy_(torch.ones_like(y))
            g.launch(stream=stream)
        torch.cuda.synchronize()
        print(y)

    time.sleep(5)


if __name__ == "__main__":
    main()
