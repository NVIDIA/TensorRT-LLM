import torch

from tensorrt_llm._torch.hostfunc import HOSTFUNC_USER_DATA_HANDLES, hostfunc


def test_hostfunc():

    @hostfunc
    def increase(x: torch.Tensor):
        x.add_(1)

    x = torch.zeros(10, dtype=torch.int32)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(5):
            increase(x)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        increase(x)
        increase(x)
    torch.cuda.synchronize()

    with torch.cuda.stream(stream):
        for _ in range(10):
            g.replay()

    torch.cuda.synchronize()
    assert (x == 25).all().item()
    assert len(HOSTFUNC_USER_DATA_HANDLES) == 2
