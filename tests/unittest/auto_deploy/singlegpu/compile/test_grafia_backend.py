from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.compile import CompileBackendRegistry
from tensorrt_llm._torch.auto_deploy.compile.backends.grafia import (
    GrafiaCompileError,
    GrafiaCompiler,
    GrafiaUnsupportedError,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm import *  # noqa: F403
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.library.compile_model import CompileModel
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


@dataclass
class _TensorMeta:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    strides: tuple[int, ...]

    def stride(self, dim):
        return self.strides[dim]

    def is_contiguous(self):
        expected = []
        stride = 1
        for size in reversed(self.shape):
            expected.append(stride)
            stride *= size
        return self.strides == tuple(reversed(expected))


def _make_torch_rmsnorm_gm(
    *,
    input_meta: _TensorMeta | None = None,
    weight_meta: _TensorMeta | None = None,
    eps: float = 1e-5,
) -> GraphModule:
    input_meta = input_meta or _TensorMeta(
        (107, 2880), torch.bfloat16, torch.device("cuda:0"), (2880, 1)
    )
    weight_meta = weight_meta or _TensorMeta(
        (2880,), torch.bfloat16, torch.device("cuda:0"), (1,)
    )

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    x.meta["val"] = input_meta
    weight.meta["val"] = weight_meta
    rms = graph.call_function(
        torch.ops.auto_deploy.torch_rmsnorm.default,
        args=(x, weight, eps),
    )
    rms.meta["val"] = input_meta
    graph.output(rms)
    return GraphModule(torch.nn.Module(), graph)


def _make_unsupported_gm() -> GraphModule:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = _TensorMeta(
        (107, 2880), torch.bfloat16, torch.device("cuda:0"), (2880, 1)
    )
    neg = graph.call_function(torch.ops.aten.neg.default, args=(x,))
    neg.meta["val"] = x.meta["val"]
    graph.output(neg)
    return GraphModule(torch.nn.Module(), graph)


def _make_grafia_rmsnorm_gm() -> GraphModule:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    meta = _TensorMeta((107, 2880), torch.bfloat16, torch.device("cuda:0"), (2880, 1))
    weight_meta = _TensorMeta((2880,), torch.bfloat16, torch.device("cuda:0"), (1,))
    x.meta["val"] = meta
    weight.meta["val"] = weight_meta
    rms = graph.call_function(
        torch.ops.auto_deploy.grafia_rms_norm.default,
        args=(x, weight, 1e-5),
    )
    rms.meta["val"] = meta
    graph.output(rms)
    return GraphModule(torch.nn.Module(), graph)


def _make_rmsnorm_with_passthrough_output_gm(output_kind: str) -> GraphModule:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.get_attr("weight")
    meta = _TensorMeta((107, 2880), torch.bfloat16, torch.device("cuda:0"), (2880, 1))
    weight_meta = _TensorMeta((2880,), torch.bfloat16, torch.device("cuda:0"), (1,))
    x.meta["val"] = meta
    weight.meta["val"] = weight_meta
    rms = graph.call_function(
        torch.ops.auto_deploy.torch_rmsnorm.default,
        args=(x, weight, 1e-5),
    )
    rms.meta["val"] = meta
    graph.output(x if output_kind == "placeholder" else weight)

    root = torch.nn.Module()
    root.register_parameter(
        "weight",
        torch.nn.Parameter(torch.ones(2880, dtype=torch.bfloat16)),
    )
    return GraphModule(root, graph)


def _runtime_available() -> bool:
    try:
        import grafia_runtime  # noqa: F401
        from backends.ctm.factories.rmsnorm_rts import _default_cubin_path

        if not torch.cuda.is_available() or _default_cubin_path() is None:
            return False
        major, _minor = torch.cuda.get_device_capability()
        return major >= 10
    except Exception:
        return False


def _rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    x32 = x.to(torch.float32)
    w32 = weight.to(torch.float32)
    return (x32 * torch.rsqrt((x32 * x32).mean(dim=-1, keepdim=True) + eps) * w32).to(
        x.dtype
    )


def test_grafia_backend_registered_and_config_accepts_backend():
    assert CompileBackendRegistry.has("grafia")
    assert CompileBackendRegistry.get("grafia") is GrafiaCompiler
    transform = CompileModel.from_kwargs(stage="compile", backend="grafia")
    assert transform.config.backend == "grafia"


def test_grafia_backend_strictly_rejects_unsupported_op():
    gm = _make_unsupported_gm()

    with pytest.raises(GrafiaUnsupportedError, match="no fallback path.*Unsupported FX"):
        GrafiaCompiler(gm).compile()


def test_grafia_backend_rejects_old_grafia_rms_norm_custom_op():
    with pytest.raises(GrafiaUnsupportedError, match="old per-op custom op path"):
        GrafiaCompiler(_make_grafia_rmsnorm_gm()).compile()


@pytest.mark.parametrize("output_kind", ["placeholder", "get_attr"])
def test_grafia_backend_rejects_passthrough_graph_outputs(output_kind):
    gm = _make_rmsnorm_with_passthrough_output_gm(output_kind)

    with pytest.raises(GrafiaUnsupportedError, match="every graph output"):
        GrafiaCompiler(gm).compile()


def test_grafia_backend_guard_error_for_unsupported_rmsnorm_shape():
    gm = _make_torch_rmsnorm_gm(
        input_meta=_TensorMeta(
            (107, 1024), torch.bfloat16, torch.device("cuda:0"), (1024, 1)
        ),
        weight_meta=_TensorMeta((1024,), torch.bfloat16, torch.device("cuda:0"), (1,)),
    )

    with pytest.raises(GrafiaUnsupportedError, match="hidden size 2880"):
        GrafiaCompiler(gm).compile()


def test_grafia_backend_missing_runtime_env_error(monkeypatch):
    from tensorrt_llm._torch.auto_deploy.compile.backends import grafia as grafia_backend

    real_import_module = grafia_backend.importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "grafia_runtime":
            raise ImportError("blocked for test")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(grafia_backend.importlib, "import_module", fake_import_module)

    with pytest.raises(GrafiaCompileError, match="requires Grafia runtime"):
        grafia_backend._import_grafia_runtime_deps()


@pytest.mark.skipif(not _runtime_available(), reason="Grafia runtime/CUDA sm100 unavailable")
def test_grafia_backend_rmsnorm_runtime_matches_torch_without_custom_op_path():
    gm = _make_torch_rmsnorm_gm()
    assert any(is_op(n, torch.ops.auto_deploy.torch_rmsnorm) for n in gm.graph.nodes)
    assert not any(is_op(n, torch.ops.auto_deploy.grafia_rms_norm) for n in gm.graph.nodes)

    torch.manual_seed(0)
    x = torch.randn(107, 2880, dtype=torch.bfloat16, device="cuda").contiguous()
    weight = torch.randn(2880, dtype=torch.bfloat16, device="cuda").contiguous()

    compiled = GrafiaCompiler(gm, args=(x, weight)).compile()

    assert compiled.lowered_op_kinds == ["grafia.fast_low_latency_rms_norm"]
    assert "grafia_rms_norm" not in " ".join(compiled.lowered_op_kinds)

    y = compiled(x, weight)
    expected = _rmsnorm_reference(x, weight)
    torch.testing.assert_close(y, expected, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not _runtime_available(), reason="Grafia runtime/CUDA sm100 unavailable")
def test_grafia_backend_exported_get_attr_weight_matches_torch_without_custom_op_path():
    class RMSNormModule(torch.nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight)

        def forward(self, x: torch.Tensor):
            return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, 1e-5)

    torch.manual_seed(1)
    x = torch.randn(107, 2880, dtype=torch.bfloat16, device="cuda").contiguous()
    weight = torch.randn(2880, dtype=torch.bfloat16, device="cuda").contiguous()
    model = RMSNormModule(weight).to("cuda").eval()
    expected = _rmsnorm_reference(x, model.weight.detach())

    gm = torch_export_to_gm(model, args=(x,), dynamic_shapes=None)

    assert any(
        n.op == "call_function" and n.target == torch.ops.auto_deploy.torch_rmsnorm.default
        for n in gm.graph.nodes
    )
    assert any(n.op == "get_attr" and n.target == "weight" for n in gm.graph.nodes)
    assert not any(is_op(n, torch.ops.auto_deploy.grafia_rms_norm) for n in gm.graph.nodes)

    compiled = GrafiaCompiler(gm, args=(x,)).compile()

    assert compiled.input_names == ["x"]
    assert compiled.lowered_op_kinds == ["grafia.fast_low_latency_rms_norm"]
    assert "grafia_rms_norm" not in " ".join(compiled.lowered_op_kinds)

    (y,) = compiled(x)
    torch.testing.assert_close(y, expected, atol=5e-2, rtol=5e-2)
