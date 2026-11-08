import torch

from tensorrt_llm._torch.hostfunc import (HOSTFUNC_USER_DATA_HANDLES, hostfunc,
                                          set_low_latency_dispatch)


def test_hostfunc():

    @hostfunc
    def increase(x: torch.Tensor):
        x.add_(1)

    x = torch.zeros(10, dtype=torch.int32)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(5):
            increase(x)
    torch.cuda.synchronize()

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


def test_low_latency_dispatch():
    """set_low_latency_dispatch toggles the module-level flag.

    Hostfuncs fire correctly in both modes (cudaLaunchHostFunc v1 and v2 spin-wait).
    """
    import tensorrt_llm._torch.hostfunc as hf_mod

    for enabled in (False, True):
        set_low_latency_dispatch(enabled)
        assert hf_mod._low_latency_dispatch is enabled

        results = []

        @hostfunc
        def record(val):
            results.append(val)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            record(enabled)
        torch.cuda.synchronize()

        assert results == [enabled
                           ], f"hostfunc failed with low_latency={enabled}"

    # Restore default
    set_low_latency_dispatch(False)
