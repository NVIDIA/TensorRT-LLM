# Regression tests for MetaInitMode's handling of the value-irrelevant in-place
# initializers that stock torch modules run in __init__
# (nn.LayerNorm.reset_parameters -> init.ones_ -> aten.fill_.Scalar,
# nn.init.zeros_ -> aten.zero_.default).
#
# Without the deterministic_init_ops pass-through, these ops raise
# MetaInitException, which aborts meta init for the WHOLE model: the model
# loader's broad-except fallback then constructs the full model on CPU memory.
# That is a load-time slowdown on every platform, and on aarch64 builds (where
# torch's CPU allocator is mimalloc) the freed weight-shard-sized arena is
# retained for process lifetime — measured 142 GiB/rank for a ~554 GB
# checkpoint at TP4 on GB300, starving pinned-host budgets such as
# KvCacheConfig.host_cache_size. These tests pin the pass-through so it cannot
# silently regress.
import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.models.modeling_utils import MetaInitException, MetaInitMode


def test_layernorm_constructs_on_meta():
    # nn.LayerNorm runs init.ones_ (aten.fill_.Scalar) in reset_parameters.
    # Fails with MetaInitException on any tree without the
    # deterministic_init_ops pass-through.
    with MetaInitMode():
        m = nn.LayerNorm(64)
    assert m.weight.device == torch.device("meta")
    assert m.bias.device == torch.device("meta")


def test_zeros_init_constructs_on_meta():
    class ZeroBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.empty(16))
            nn.init.zeros_(self.bias)

    with MetaInitMode():
        m = ZeroBias()
    assert m.bias.device == torch.device("meta")


def test_meta_params_materialize_after_init():
    with MetaInitMode():
        m = nn.LayerNorm(64)
    m._apply(lambda t: torch.empty_like(t, device="cpu") if t.device == torch.device("meta") else t)
    assert m.weight.device == torch.device("cpu")
    assert m.weight.shape == (64,)


def test_unsupported_op_still_raises():
    # The conservative contract is load-bearing: ops whose results are
    # actually consumed (e.g. computing non-persistent buffers) must still
    # abort meta init so the caller can fall back to regular construction.
    with MetaInitMode():
        t = torch.empty(4, 4)  # redirected to the meta device
        with pytest.raises(MetaInitException):
            torch.mm(t, t)
