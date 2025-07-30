import torch


def device_func(x, y):
    torch.matmul(x, x, out=y)


x = torch.randn(4096, 4096, device='cuda')
y1 = torch.empty_like(x)
y2 = torch.empty_like(x)
y3 = torch.empty_like(x)
device_func(x, y1)
torch.cuda.synchronize()

# Create two separate streams on the same CUDA device
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
g = torch.cuda.CUDAGraph()

with torch.cuda.graph(g):
    event = torch.cuda.Event()
    s1.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s1):
        device_func(x, y1)
        event.record()
        device_func(x, y3)

    with torch.cuda.stream(s2):
        s2.wait_event(event)
        device_func(x, y2)
    torch.cuda.current_stream().wait_stream(s1)
    torch.cuda.current_stream().wait_stream(s2)

torch.cuda.synchronize()
print("Graph captured", flush=True)

s3 = torch.cuda.Stream()
with torch.cuda.stream(s3):
    g.replay()

s4 = torch.cuda.Stream()
s4.wait_stream(s3)
with torch.cuda.stream(s4):
    g.replay()

torch.cuda.synchronize()
