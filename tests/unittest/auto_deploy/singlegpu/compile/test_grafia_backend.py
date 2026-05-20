from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest
import torch
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.compile import CompileBackendRegistry
from tensorrt_llm._torch.auto_deploy.compile.backends.grafia import (
    RMSNORM_OP_KIND,
    GrafiaBackendContext,
    GrafiaBackendPlugin,
    GrafiaCompileError,
    GrafiaCompiler,
    GrafiaExecutor,
    GrafiaRegionModule,
)
from tensorrt_llm._torch.auto_deploy.compile.backends.grafia import plugin as grafia_plugin_module
from tensorrt_llm._torch.auto_deploy.compile.backends.grafia.ops import rmsnorm as grafia_rmsnorm
from tensorrt_llm._torch.auto_deploy.compile.lowering import (
    LOWERINGS,
    ModeContext,
    OpArgumentResolver,
    PlanDispatcher,
    ProgramData,
    SupportDecision,
    ValueType,
    analyze_region_boundaries,
    lower_rms_norm,
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


class _FakeDType(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT64 = "int64"
    INT32 = "int32"
    INT8 = "int8"
    UINT8 = "uint8"
    BOOL = "bool"


@dataclass(frozen=True)
class _FakeTensorSpec:
    shape: tuple[int, ...]
    dtype: _FakeDType
    storage_id: int


class _FakeTensorSpecFactory:
    @staticmethod
    def contiguous(shape, dtype, storage_id):
        return _FakeTensorSpec(tuple(shape), dtype, storage_id)


class _FakeTypesModule:
    DType = _FakeDType
    TensorSpec = _FakeTensorSpecFactory


@dataclass(frozen=True)
class _FakeCTMTensorSpec:
    spec: _FakeTensorSpec
    name: str
    producer_id: int | None = None
    producer_idx: int = 0


@dataclass
class _FakeCTMOpSpec:
    op_kind: str
    id: int
    inputs: list[_FakeCTMTensorSpec]
    outputs: list[_FakeCTMTensorSpec]
    attrs: dict[str, object]


@dataclass
class _FakeCTMGraphSpec:
    name: str
    ops: list[_FakeCTMOpSpec]
    inputs: list[_FakeCTMTensorSpec]
    outputs: list[_FakeCTMTensorSpec]
    constant_data: dict[_FakeCTMTensorSpec, torch.Tensor]


class _FakeSpecModule:
    CTMTensorSpec = _FakeCTMTensorSpec
    CTMOpSpec = _FakeCTMOpSpec
    CTMGraphSpec = _FakeCTMGraphSpec


class _RecordingRMSNormAdapter:
    def __init__(self) -> None:
        self.calls = []

    def emit_rms_norm(self, x, weight, *, eps, result_meta, loc=None):
        self.calls.append(("emit_rms_norm", x, weight, eps, result_meta, loc))
        return "%rms"


class _RecordingLoweringContext:
    def __init__(self, adapter: _RecordingRMSNormAdapter) -> None:
        self.args = OpArgumentResolver()
        self.b = adapter
        self.adapter = adapter

    def resolve(self, value):
        if isinstance(value, torch.fx.Node):
            return f"%{value.name}"
        return value

    def result_type(self, node):
        return ValueType.from_node(node)

    def loc(self, node):
        return node.name


def _patch_fake_ctm_compile(monkeypatch) -> None:
    monkeypatch.setattr(
        GrafiaBackendContext,
        "spec_deps",
        lambda self: (_FakeSpecModule, _FakeTypesModule),
    )
    monkeypatch.setattr(
        GrafiaBackendContext,
        "compile_spec",
        lambda self, spec, spec_cache_key, **kwargs: (None, None),
    )


def _make_torch_rmsnorm_gm(
    *,
    input_meta: _TensorMeta | None = None,
    weight_meta: _TensorMeta | None = None,
    eps: float = 1e-5,
) -> GraphModule:
    input_meta = input_meta or _TensorMeta(
        (107, 2880), torch.bfloat16, torch.device("cuda:0"), (2880, 1)
    )
    weight_meta = weight_meta or _TensorMeta((2880,), torch.bfloat16, torch.device("cuda:0"), (1,))

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
    x.meta["val"] = _TensorMeta((107, 2880), torch.bfloat16, torch.device("cuda:0"), (2880, 1))
    neg = graph.call_function(torch.ops.aten.neg.default, args=(x,), name="neg")
    neg.meta["val"] = x.meta["val"]
    graph.output(neg)
    return GraphModule(torch.nn.Module(), graph)


def _make_rmsnorm_then_unsupported_gm() -> GraphModule:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    meta = _TensorMeta((107, 2880), torch.bfloat16, torch.device("cuda:0"), (2880, 1))
    weight_meta = _TensorMeta((2880,), torch.bfloat16, torch.device("cuda:0"), (1,))
    x.meta["val"] = meta
    weight.meta["val"] = weight_meta
    rms = graph.call_function(
        torch.ops.auto_deploy.torch_rmsnorm.default,
        args=(x, weight, 1e-5),
    )
    rms.meta["val"] = meta
    neg = graph.call_function(torch.ops.aten.neg.default, args=(rms,), name="neg")
    neg.meta["val"] = meta
    graph.output(neg)
    return GraphModule(torch.nn.Module(), graph)


def _make_torch_rmsnorm_get_attr_gm() -> GraphModule:
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
    graph.output(rms)

    root = torch.nn.Module()
    root.register_parameter(
        "weight",
        torch.nn.Parameter(torch.ones(2880, dtype=torch.bfloat16)),
    )
    return GraphModule(root, graph)


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
    return (x32 * torch.rsqrt((x32 * x32).mean(dim=-1, keepdim=True) + eps) * w32).to(x.dtype)


def _has_target_name(gm: GraphModule, target_name: str) -> bool:
    return any(
        getattr(getattr(node.target, "overloadpacket", node.target), "__name__", "") == target_name
        for node in gm.graph.nodes
    )


def test_grafia_backend_registered_and_config_accepts_backend():
    assert CompileBackendRegistry.has("grafia")
    assert CompileBackendRegistry.get("grafia") is GrafiaCompiler
    transform = CompileModel.from_kwargs(stage="compile", backend="grafia")
    assert transform.config.backend == "grafia"


def test_grafia_rmsnorm_support_and_shared_lowering_dispatch():
    gm = _make_torch_rmsnorm_gm()
    rms = next(n for n in gm.graph.nodes if is_op(n, torch.ops.auto_deploy.torch_rmsnorm))

    decision = grafia_rmsnorm.classify_node(
        rms,
        ModeContext(name="rmsnorm", phase="rmsnorm"),
        ProgramData(gm),
        OpArgumentResolver(),
    )

    assert isinstance(decision, SupportDecision)
    assert decision.is_supported

    adapter = _RecordingRMSNormAdapter()
    result = lower_rms_norm(_RecordingLoweringContext(adapter), rms)

    assert result == "%rms"
    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert call[:4] == ("emit_rms_norm", "%x", "%weight", 1e-5)
    assert call[4] == ValueType.from_node(rms)
    assert call[5] == rms.name


def test_grafia_backend_plugin_uses_global_lowering_mapping(monkeypatch):
    _patch_fake_ctm_compile(monkeypatch)
    gm = _make_torch_rmsnorm_gm()
    program = ProgramData(gm)
    rms = next(n for n in gm.graph.nodes if is_op(n, torch.ops.auto_deploy.torch_rmsnorm))
    region = analyze_region_boundaries(
        gm,
        [rms],
        mode=ModeContext(name="decode", phase="decode"),
        region_id="r0",
    )
    captured = {}

    def fake_lower_region(
        program_arg,
        region_arg,
        adapter_arg,
        args_arg,
        *,
        lowerings=None,
    ):
        captured["program"] = program_arg
        captured["region"] = region_arg
        captured["args"] = args_arg
        captured["adapter"] = adapter_arg
        captured["lowerings"] = lowerings
        return "artifact"

    monkeypatch.setattr(
        grafia_plugin_module,
        "lower_region",
        fake_lower_region,
    )
    plugin = GrafiaBackendPlugin(compile_artifacts=False)
    args = OpArgumentResolver()

    artifact = plugin.lower_region(region, program, args)

    assert artifact == "artifact"
    assert captured["program"] is program
    assert captured["region"] is region
    assert captured["args"] is args
    assert captured["lowerings"] is LOWERINGS


def test_grafia_backend_routes_unsupported_op_eager_with_zero_regions():
    gm = _make_unsupported_gm()

    compiled = GrafiaCompiler(gm, grafia_dispatch_policy="eager").compile()

    assert isinstance(compiled, PlanDispatcher)
    assert isinstance(compiled, GrafiaExecutor)
    assert [plan.mode.name for plan in compiled.plans] == ["rmsnorm"]
    assert all(plan.regions == () for plan in compiled.plans)
    assert all(plan.coverage.eager_nodes == ("neg",) for plan in compiled.plans)

    x = torch.ones(2, 3)
    torch.testing.assert_close(compiled(x), gm(x))


@pytest.mark.parametrize("output_kind", ["placeholder", "get_attr"])
def test_grafia_backend_routes_passthrough_outputs_eager(output_kind):
    gm = _make_rmsnorm_with_passthrough_output_gm(output_kind)

    compiled = GrafiaCompiler(gm, grafia_dispatch_policy="eager").compile()

    assert all(plan.regions == () for plan in compiled.plans)
    assert all(plan.coverage.eager_nodes for plan in compiled.plans)


def test_grafia_backend_covers_rmsnorm_with_single_default_plan(monkeypatch):
    _patch_fake_ctm_compile(monkeypatch)
    gm = _make_rmsnorm_then_unsupported_gm()

    compiled = GrafiaCompiler(gm).compile()

    assert isinstance(compiled, GraphModule)
    assert not isinstance(compiled, PlanDispatcher)
    assert [plan.mode.name for plan in compiled.grafia_plans] == ["rmsnorm"]
    (plan,) = compiled.grafia_plans
    assert len(plan.regions) == 1
    (region,) = plan.regions
    assert region.source_node_names == ("torch_rmsnorm_default",)
    assert plan.coverage.eager_nodes == ("neg",)
    assert plan.coverage.region_for_node("torch_rmsnorm_default") == region.region_id
    assert any(node.op == "call_module" for node in compiled.graph.nodes)
    module = plan.region_modules[region.region_id]
    assert isinstance(module, GrafiaRegionModule)
    assert module.mode_name == plan.mode.name
    assert module.lowered_op_kinds == [RMSNORM_OP_KIND]


def test_grafia_backend_get_attr_weight_is_constant_not_runtime_input(monkeypatch):
    _patch_fake_ctm_compile(monkeypatch)
    gm = _make_torch_rmsnorm_get_attr_gm()

    compiled = GrafiaCompiler(gm).compile()
    plan = compiled.grafia_plans[0]
    (region,) = plan.regions
    lowered = plan.artifacts[region.region_id]
    module = plan.region_modules[region.region_id]

    assert isinstance(module, GrafiaRegionModule)
    assert module.input_names == ["x"]
    assert module.constant_names == ["weight"]
    assert [tensor.name for tensor in lowered.spec.inputs] == ["x"]
    assert [tensor.name for tensor in lowered.spec.constant_data] == ["weight"]


def test_grafia_backend_routes_unsupported_rmsnorm_shape_eager():
    gm = _make_torch_rmsnorm_gm(
        input_meta=_TensorMeta((107, 1024), torch.bfloat16, torch.device("cuda:0"), (1024, 1)),
        weight_meta=_TensorMeta((1024,), torch.bfloat16, torch.device("cuda:0"), (1,)),
    )

    compiled = GrafiaCompiler(gm, grafia_dispatch_policy="eager").compile()

    assert all(plan.regions == () for plan in compiled.plans)
    assert all(plan.coverage.eager_nodes == ("torch_rmsnorm_default",) for plan in compiled.plans)


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
    assert not _has_target_name(gm, "grafia_rms_norm")

    torch.manual_seed(0)
    x = torch.randn(107, 2880, dtype=torch.bfloat16, device="cuda").contiguous()
    weight = torch.randn(2880, dtype=torch.bfloat16, device="cuda").contiguous()

    compiled = GrafiaCompiler(gm, args=(x, weight), grafia_dispatch_policy="first").compile()

    assert isinstance(compiled, PlanDispatcher)
    assert [plan.mode.name for plan in compiled.plans] == ["rmsnorm"]
    first_region = next(iter(compiled.grafia_region_modules.values()))
    assert isinstance(first_region, GrafiaRegionModule)
    assert first_region.lowered_op_kinds == [RMSNORM_OP_KIND]
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
    assert not _has_target_name(gm, "grafia_rms_norm")

    compiled = GrafiaCompiler(gm, args=(x,), grafia_dispatch_policy="first").compile()

    assert compiled.input_names == ["x"]
    first_region = next(iter(compiled.grafia_region_modules.values()))
    assert isinstance(first_region, GrafiaRegionModule)
    assert first_region.input_names == ["x"]
    assert first_region.lowered_op_kinds == [RMSNORM_OP_KIND]
    assert "grafia_rms_norm" not in " ".join(compiled.lowered_op_kinds)

    (y,) = compiled(x)
    torch.testing.assert_close(y, expected, atol=5e-2, rtol=5e-2)
