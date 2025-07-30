import cupy
import torch

x = torch.randn(4096, 4096, device='cuda')
y = torch.empty_like(x)


def device_func(x, y):
    torch.matmul(x, x, out=y)


def host_callback(data):
    print("Host callback says:", data)


# Warm up
device_func(x, y)

# stream = torch.cuda.current_stream()
stream = torch.cuda.Stream()
stream_cupy = cupy.cuda.ExternalStream(stream.cuda_stream)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=stream):
    device_func(x, y)
    with stream_cupy:
        stream_cupy.launch_host_func(host_callback,
                                     "PyTorch CUDA graph completed!")

torch.cuda.synchronize()
print("Graph captured", flush=True)

for i in range(5):
    with torch.cuda.stream(stream), stream_cupy:
        x.copy_(torch.ones_like(x))
        y.copy_(torch.ones_like(y))
        g.replay()
    torch.cuda.synchronize()
    print(y)
