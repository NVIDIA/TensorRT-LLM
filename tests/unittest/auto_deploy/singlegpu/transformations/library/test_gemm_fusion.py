"""Tests for basic GEMM fusion."""

import operator
from abc import abstractmethod
from typing import Callable, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _graph_test_helpers import count_buffers, run_test_transformed_gm
from _model_test_utils import FakeFP8Linear
from _torch_test_utils import all_close, fp8_compatible, reset_parameters

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 (registers torch_attention op)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op, is_op
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

torch.manual_seed(0)


class TestModel(nn.Module):
    @abstractmethod
    def get_input(self, **kwargs) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def keys_to_pop(self) -> Tuple[str, str]:
        pass

    @property
    @abstractmethod
    def num_gemms_after_fusion(self) -> int:
        pass


class FusableModel(TestModel):
    def __init__(self, batch_size=4, in_features=16, out_features=32):
        super().__init__()
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features

    def get_input(self, **kwargs):
        return torch.ones(self.batch_size, self.in_features, **kwargs)

    @property
    def keys_to_pop(self) -> Tuple[str, str]:
        return ("fc1.weight", "fc2.weight")


class FusableModel1(FusableModel):
    """The simplest fusable model with two consecutive GEMMs + concat."""

    def __init__(self, cls=nn.Linear, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = cls(self.in_features, self.out_features, bias=False)
        self.fc2 = cls(self.in_features, self.out_features, bias=False)

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        return torch.cat([y1, y2], dim=-1)

    @property
    def num_gemms_after_fusion(self) -> int:
        return 1


class FusableModel1_M(FusableModel1):
    def __init__(self, **kwargs):
        super().__init__(**{"in_features": 2048, "out_features": 128, **kwargs})


class FusableModel1_M_FP8(FusableModel1_M):
    def __init__(self, **kwargs):
        super().__init__(**{"cls": FakeFP8Linear, **kwargs})


class FusableModel1_L(FusableModel1):
    def __init__(self, **kwargs):
        super().__init__(**{"in_features": 2048, "out_features": 129, **kwargs})


class FusableModel1_XL(FusableModel1):
    def __init__(self, **kwargs):
        super().__init__(**{"in_features": 2048, "out_features": 144, **kwargs})


class FusableModel1_XL_FP8(FusableModel1_XL):
    def __init__(self, **kwargs):
        super().__init__(**{"cls": FakeFP8Linear, **kwargs})


class FusableModel2(FusableModel):
    def __init__(self, cls=nn.Linear, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = cls(self.in_features, self.out_features, bias=False)
        self.fc2 = cls(self.in_features, self.out_features, bias=False)
        self.fc3 = cls(self.in_features, self.out_features, bias=False)

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        return y1 * y2 + y3

    @property
    def num_gemms_after_fusion(self) -> int:
        return 1


class FusableModel2_FP8(FusableModel2):
    def __init__(self, **kwargs):
        super().__init__(**{"cls": FakeFP8Linear, **kwargs})


class FusableModel3(FusableModel):
    """Same as FusableModel1 except one GEMM is not fusable due to missing bias support."""

    def __init__(self, cls=nn.Linear, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = cls(self.in_features, self.out_features, bias=False)
        self.fc2 = cls(self.in_features, self.out_features, bias=False)
        self.fc3 = cls(self.in_features, self.out_features, bias=True)  # no bias support yet

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        return y1 * y2 + y3

    @property
    def num_gemms_after_fusion(self) -> int:
        return 2


class FusableModel3_FP8(FusableModel3):
    def __init__(self, **kwargs):
        super().__init__(**{"cls": FakeFP8Linear, **kwargs})


class FusableModel4(FusableModel):
    """Fusable but non-consecutive.

    In this model the fusable layers are not consecutive and we have a bunch of other layers
    operations in between.
    """

    def __init__(self, cls=nn.Linear, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = cls(self.in_features, self.out_features, bias=False)
        self.fc2 = cls(self.in_features, self.out_features, bias=False)
        self.fc3 = cls(self.in_features, self.out_features, bias=False)
        self.fc4 = cls(self.out_features, self.out_features, bias=False)

    def forward(self, x):
        y1 = self.fc1(x)
        y1 = self.fc4(F.relu(y1))
        y2 = y1 * (self.fc2(x) + y1)
        y3 = self.fc3(x)
        return F.relu(y2 + y3) * y1

    @property
    def num_gemms_after_fusion(self) -> int:
        return 2


class FusableModel4_FP8(FusableModel4):
    def __init__(self, **kwargs):
        super().__init__(**{"cls": FakeFP8Linear, **kwargs})


# TODO: consider adding test cases for classic GQA and MLP layers
@pytest.mark.parametrize(
    "get_model,dtype",
    [
        (FusableModel1, "float16"),
        (FusableModel1_M, "float16"),
        (FusableModel1_L, "float16"),
        (FusableModel1_XL, "float16"),
        (FusableModel2, "float16"),
        (FusableModel3, "float16"),
        (FusableModel4, "float16"),
        (FusableModel1, "bfloat16"),
        pytest.param(
            FusableModel1_M,
            "bfloat16",
            marks=pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5095416"),
        ),
        pytest.param(
            FusableModel1_L,
            "bfloat16",
            marks=pytest.mark.xfail(reason="Inconsistent bf16 GEMMs, shape not compatible for FP8"),
        ),
        pytest.param(
            FusableModel1_XL,
            "bfloat16",
            marks=pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5095416"),
        ),
        (FusableModel2, "bfloat16"),
        (FusableModel3, "bfloat16"),
        (FusableModel4, "bfloat16"),
        pytest.param(
            FusableModel1_M_FP8,
            "fp8",
            marks=pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support"),
        ),
        pytest.param(
            FusableModel1_XL_FP8,
            "fp8",
            marks=pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support"),
        ),
        pytest.param(
            FusableModel2_FP8,
            "fp8",
            marks=[
                pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support"),
                pytest.mark.xfail(reason="Inconsistent fp8 GEMMs"),
            ],
        ),
        pytest.param(
            FusableModel3_FP8,
            "fp8",
            marks=[
                pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support"),
                pytest.mark.xfail(reason="Inconsistent fp8 GEMMs"),
            ],
        ),
        pytest.param(
            FusableModel4_FP8,
            "fp8",
            marks=[
                pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support"),
                pytest.mark.xfail(reason="Inconsistent fp8 GEMMs"),
            ],
        ),
    ],
)
@torch.inference_mode()
def test_fusion(get_model: Callable[[], TestModel], dtype: str):
    # some basic config stuff
    model = get_model().to(device="cuda")
    if dtype != "fp8":
        torch_dtype = getattr(torch, dtype)
        model = model.to(torch_dtype)
        x = model.get_input(device="cuda", dtype=torch_dtype)
    else:
        x = model.get_input(device="cuda", dtype=torch.half)

    # run the main test
    y_model = model(x)

    tol = 5e-3 if dtype == "fp8" else 1e-3

    buffer_size_before = count_buffers(model)

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_gemms": {
                "stage": "post_load_fusion",
            },
            "fuse_fp8_gemms": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        lambda gm: sum(
            (is_linear_op(n) or is_op(n, torch.ops.auto_deploy.torch_fake_quant_fp8_linear))
            for n in gm.graph.nodes
        )
        == model.num_gemms_after_fusion,
        lambda num_p_og: num_p_og,  # unchanged since fusing doesn't change param count
        atol=tol,
        rtol=tol,
        test_load_hook=False,
    )

    buffer_size_after = count_buffers(gm_transformed)

    # Fusion should reduce buffer sizes (e.g. 3 qkv scales separately to 1 qkv scale).
    assert buffer_size_after <= buffer_size_before, (
        f"buffers after fusion {buffer_size_after} > buffers before fusion {buffer_size_before}"
    )

    reset_parameters(gm_transformed)
    y_random = gm_transformed(x)
    assert not all_close(y_model, y_random)


# ===========================================================================
# Tests for fuse_gemms_mixed_children (relaxed fusion with narrow / zero-copy views)
# ===========================================================================


class GdnLikeFusableModel(TestModel):
    """Mimics GatedDeltaNet projection pattern with 4 linears sharing the same input.

    Includes a non-linear user (shape access) on that input. Standard
    fuse_gemms will skip this because check_same_children fails.
    """

    def __init__(self, batch_size=4, seq_len=8, hidden=64, qkv_dim=48, v_dim=32, heads=4):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden = hidden
        self.v_dim = v_dim
        self.heads = heads

        self.in_proj_qkv = nn.Linear(hidden, qkv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, v_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, heads, bias=False)

    def get_input(self, **kwargs) -> torch.Tensor:
        return torch.randn(self.batch_size, self.seq_len, self.hidden, **kwargs)

    @property
    def keys_to_pop(self):
        return ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight")

    @property
    def num_gemms_after_fusion(self) -> int:
        return 1

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        z = z.reshape(batch_size, seq_len, -1)
        return qkv.sum(-1, keepdim=True) + z.sum(-1, keepdim=True) + b + a


class Qwen35GdnLikeFusableModel(TestModel):
    """Mirrors the actual Qwen3.5 GatedDeltaNet graph pattern more closely.

    Uses proportional dimensions from the real graph and includes downstream ops
    (reshape, sigmoid, element-wise mul) that match the actual model structure.
    """

    def __init__(
        self,
        batch_size=2,
        seq_len=8,
        hidden=256,
        qkv_dim=96,
        z_dim=64,
        num_heads=8,
        head_dim=8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.in_proj_qkv = nn.Linear(hidden, qkv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, z_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, num_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, num_heads, bias=False)

    def get_input(self, **kwargs) -> torch.Tensor:
        return torch.randn(self.batch_size, self.seq_len, self.hidden, **kwargs)

    @property
    def keys_to_pop(self):
        return ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight")

    @property
    def num_gemms_after_fusion(self) -> int:
        return 1

    def forward(self, x):
        batch_size, seq_len, _ = x.shape  # non-linear shape user
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        b = torch.sigmoid(self.in_proj_b(x))
        a = self.in_proj_a(x) * b  # element-wise gating
        return qkv.sum(-1, keepdim=True) + z.sum(-1).sum(-1, keepdim=True) + a.sum(-1, keepdim=True)


class Qwen35GdnMixedQuantModel(TestModel):
    """Qwen3.5-like GDN model with mixed quantization: qkv and z are FP8, a and b are bf16.

    After GDN fusion, qkv+z should be fused into one FP8 linear and a+b into
    one bf16 linear, producing 2 fused ops and 4 narrow nodes total.
    """

    def __init__(
        self,
        batch_size=2,
        seq_len=8,
        hidden=256,
        qkv_dim=96,
        z_dim=64,
        num_heads=8,
        head_dim=8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim

        # FP8 quantized projections (handle dtype internally via custom op)
        self.in_proj_qkv = FakeFP8Linear(hidden, qkv_dim, bias=False)
        self.in_proj_z = FakeFP8Linear(hidden, z_dim, bias=False)
        # bf16 projections (cast to half to match FP8 input dtype)
        self.in_proj_b = nn.Linear(hidden, num_heads, bias=False).half()
        self.in_proj_a = nn.Linear(hidden, num_heads, bias=False).half()

    def get_input(self, **kwargs) -> torch.Tensor:
        return torch.randn(self.batch_size, self.seq_len, self.hidden, **kwargs)

    @property
    def keys_to_pop(self):
        return ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight")

    @property
    def num_gemms_after_fusion(self) -> int:
        # 1 fused FP8 (qkv+z) + 1 fused bf16 (a+b)
        return 2

    def forward(self, x):
        batch_size, seq_len, _ = x.shape  # non-linear shape user
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        b = torch.sigmoid(self.in_proj_b(x))
        a = self.in_proj_a(x) * b  # element-wise gating
        return qkv.sum(-1, keepdim=True) + z.sum(-1).sum(-1, keepdim=True) + a.sum(-1, keepdim=True)


def _count_split_output_views(gm) -> int:
    """Count output split views produced by fusion (torch.narrow or split_output+getitem)."""
    count = 0
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target is torch.narrow:
            count += 1
        elif (
            n.op == "call_function"
            and n.target is operator.getitem
            and len(n.args) >= 1
            and hasattr(n.args[0], "target")
            and getattr(n.args[0].target, "__name__", "") == "split_output"
        ):
            count += 1
    return count


@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
@torch.inference_mode()
def test_fuse_gemms_mixed_children_mixed_quant():
    model = Qwen35GdnMixedQuantModel().to(device="cuda")
    x = model.get_input(device="cuda", dtype=torch.half)

    y_model = model(x)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # Verify 2 FP8 linears + 2 bf16 linears before fusion
    num_fp8_before = sum(
        is_op(n, torch.ops.auto_deploy.torch_fake_quant_fp8_linear) for n in gm.graph.nodes
    )
    num_bf16_before = sum(is_linear_op(n) for n in gm.graph.nodes)
    assert num_fp8_before == 2, f"Expected 2 FP8 linears before fusion, got {num_fp8_before}"
    assert num_bf16_before == 2, f"Expected 2 bf16 linears before fusion, got {num_bf16_before}"

    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_gemms_mixed_children": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    # After fusion: 1 fused FP8 + 1 fused bf16 = 2 total
    num_fp8_after = sum(
        is_op(n, torch.ops.auto_deploy.torch_fake_quant_fp8_linear)
        for n in gm_transformed.graph.nodes
    )
    num_bf16_after = sum(is_linear_op(n) for n in gm_transformed.graph.nodes)
    assert num_fp8_after == 1, f"Expected 1 fused FP8 linear, got {num_fp8_after}"
    assert num_bf16_after == 1, f"Expected 1 fused bf16 linear, got {num_bf16_after}"

    # Verify 4 split output views (2 from FP8 group + 2 from bf16 group)
    split_view_count = _count_split_output_views(gm_transformed)
    assert split_view_count == 4, f"Expected 4 split output nodes, got {split_view_count}"

    # Numerical correctness
    y_transformed = gm_transformed(x)
    assert all_close(y_model, y_transformed, atol=5e-3, rtol=5e-3)

    reset_parameters(gm_transformed)
    y_random = gm_transformed(x)
    assert not all_close(y_model, y_random)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@torch.inference_mode()
def test_fuse_gemms_mixed_children_qwen35_like(dtype: str):
    torch_dtype = getattr(torch, dtype)
    model = Qwen35GdnLikeFusableModel().to(device="cuda", dtype=torch_dtype)
    x = model.get_input(device="cuda", dtype=torch_dtype)

    y_model = model(x)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # Verify 4 linears before fusion
    num_linears_before = sum(is_linear_op(n) for n in gm.graph.nodes)
    assert num_linears_before == 4, f"Expected 4 linears before fusion, got {num_linears_before}"

    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_gemms_mixed_children": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        lambda gm: sum(is_linear_op(n) for n in gm.graph.nodes) == model.num_gemms_after_fusion,
        lambda num_p_og: num_p_og,
        atol=1e-3,
        rtol=1e-3,
        test_load_hook=False,
    )

    # Verify split output views exist (contiguous splitting)
    split_view_count = _count_split_output_views(gm_transformed)
    assert split_view_count == 4, f"Expected 4 split output nodes, got {split_view_count}"

    reset_parameters(gm_transformed)
    y_random = gm_transformed(x)
    assert not all_close(y_model, y_random)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@torch.inference_mode()
def test_fuse_gemms_mixed_children(dtype: str):
    torch_dtype = getattr(torch, dtype)
    model = GdnLikeFusableModel().to(device="cuda", dtype=torch_dtype)
    x = model.get_input(device="cuda", dtype=torch_dtype)

    y_model = model(x)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # Verify fuse_gemms does NOT fuse (shape user blocks it)
    num_linears_before = sum(is_linear_op(n) for n in gm.graph.nodes)
    assert num_linears_before == 4, f"Expected 4 linears before fusion, got {num_linears_before}"

    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_gemms_mixed_children": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        lambda gm: sum(is_linear_op(n) for n in gm.graph.nodes) == model.num_gemms_after_fusion,
        lambda num_p_og: num_p_og,
        atol=1e-3,
        rtol=1e-3,
        test_load_hook=False,
    )

    # Verify split output views exist (contiguous splitting)
    split_view_count = _count_split_output_views(gm_transformed)
    assert split_view_count == 4, f"Expected 4 split output nodes, got {split_view_count}"

    reset_parameters(gm_transformed)
    y_random = gm_transformed(x)
    assert not all_close(y_model, y_random)


# ===========================================================================
# Tests for QKV fusion with TRT-LLM attention
# ===========================================================================


def _count_narrow_nodes(gm):
    return sum(1 for n in gm.graph.nodes if n.op == "call_function" and n.target is torch.narrow)


def _get_narrow_nodes(gm):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target is torch.narrow]


def _get_linear_nodes(gm):
    return [n for n in gm.graph.nodes if is_linear_op(n)]


class QKVAttentionModel(TestModel):
    """Model with separate Q, K, V projections feeding into torch_attention.

    Mimics the attention pattern in transformer models where Q, K, V are
    projected from the same input. fuse_gemms_mixed_children should fuse the
    3 projections into one GEMM with 3 narrow views.
    """

    def __init__(
        self,
        batch_size=2,
        seq_len=8,
        hidden_size=64,
        num_heads=4,
        num_kv_heads=None,
    ):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def get_input(self, **kwargs):
        return torch.randn(self.batch_size, self.seq_len, self.hidden_size, **kwargs)

    @property
    def keys_to_pop(self):
        return ("q_proj.weight", "k_proj.weight", "v_proj.weight")

    @property
    def num_gemms_after_fusion(self) -> int:
        return 2  # 1 fused QKV + 1 o_proj

    @property
    def expected_narrow_count(self) -> int:
        return 3  # Q, K, V slices

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim)

        attn = torch.ops.auto_deploy.torch_attention.default(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
            sinks=None,
            sliding_window=None,
            logit_cap=None,
            layout="bsnd",
        )
        out = attn.reshape(b, s, self.hidden_size)
        return self.o_proj(out)


class SwiGLUModel(TestModel):
    """MLP with gate + up projections sharing the same input (SwiGLU pattern).

    fuse_gemms_mixed_children should fuse gate+up into one GEMM with 2 narrow
    views.
    """

    def __init__(self, batch_size=2, seq_len=8, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def get_input(self, **kwargs):
        return torch.randn(self.batch_size, self.seq_len, self.hidden_size, **kwargs)

    @property
    def keys_to_pop(self):
        return ("gate_proj.weight", "up_proj.weight")

    @property
    def num_gemms_after_fusion(self) -> int:
        return 2  # 1 fused gate+up + 1 down_proj

    @property
    def expected_narrow_count(self) -> int:
        return 2  # gate, up slices

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "model_cls,model_kwargs",
    [
        (QKVAttentionModel, {}),
        (QKVAttentionModel, {"num_kv_heads": 2}),
        (SwiGLUModel, {}),
    ],
    ids=["qkv_mha", "qkv_gqa", "swiglu"],
)
@torch.inference_mode()
def test_fuse_qkv_and_mlp_projections(model_cls, model_kwargs, dtype: str):
    """Verify QKV and gate/up fusion: graph structure, narrow nodes, numerical correctness."""
    torch_dtype = getattr(torch, dtype)
    model = model_cls(**model_kwargs).to(device="cuda", dtype=torch_dtype)
    x = model.get_input(device="cuda", dtype=torch_dtype)
    y_model = model(x)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    gm_transformed = InferenceOptimizer(
        None,
        {"fuse_gemms_mixed_children": {"stage": "post_load_fusion"}},
    )(None, gm)

    num_linears = sum(is_linear_op(n) for n in gm_transformed.graph.nodes)
    assert num_linears == model.num_gemms_after_fusion, (
        f"Expected {model.num_gemms_after_fusion} linears after fusion, got {num_linears}"
    )

    narrow_count = _count_narrow_nodes(gm_transformed)
    assert narrow_count == model.expected_narrow_count, (
        f"Expected {model.expected_narrow_count} narrow nodes, got {narrow_count}"
    )

    y_transformed = gm_transformed(x)
    assert all_close(y_model, y_transformed, atol=1e-3, rtol=1e-3)

    reset_parameters(gm_transformed)
    y_random = gm_transformed(x)
    assert not all_close(y_model, y_random)


@pytest.mark.parametrize(
    "model_cls,model_kwargs",
    [
        (QKVAttentionModel, {}),
        (QKVAttentionModel, {"num_kv_heads": 2}),
        (SwiGLUModel, {}),
    ],
    ids=["qkv_mha", "qkv_gqa", "swiglu"],
)
@torch.inference_mode()
def test_fuse_meta_val_propagation(model_cls, model_kwargs):
    """Verify meta['val'] shapes are correct on fused linear and narrow nodes.

    Without meta['val'] propagation, downstream transforms like sharding that
    read node.meta['val'].shape would see None and crash.
    """
    model = model_cls(**model_kwargs).to(device="cuda", dtype=torch.float16)
    x = model.get_input(device="cuda", dtype=torch.float16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    gm_transformed = InferenceOptimizer(
        None,
        {"fuse_gemms_mixed_children": {"stage": "post_load_fusion"}},
    )(None, gm)

    for n in gm_transformed.graph.nodes:
        if is_linear_op(n):
            val = n.meta.get("val")
            if val is not None:
                assert val.shape[0] > 0, "Fused linear meta['val'] has invalid batch dim"
                assert val.shape[-1] > 0, "Fused linear meta['val'] has invalid output dim"

    narrow_nodes = _get_narrow_nodes(gm_transformed)
    for narrow_node in narrow_nodes:
        val = narrow_node.meta.get("val")
        assert val is not None, "Narrow node is missing meta['val']"
        expected_size = narrow_node.args[3]  # torch.narrow(tensor, dim, start, length)
        assert val.shape[-1] == expected_size, (
            f"Narrow node meta['val'] last dim {val.shape[-1]} != expected {expected_size}"
        )

    linear_nodes = _get_linear_nodes(gm_transformed)
    for linear_node in linear_nodes:
        narrow_users = [
            u for u in linear_node.users if u.op == "call_function" and u.target is torch.narrow
        ]
        if not narrow_users:
            continue
        total_narrow_size = sum(u.args[3] for u in narrow_users)
        fused_val = linear_node.meta.get("val")
        assert fused_val is not None, "Fused linear is missing meta['val']"
        assert fused_val.shape[-1] == total_narrow_size, (
            f"Fused linear output dim {fused_val.shape[-1]} != "
            f"sum of narrow sizes {total_narrow_size}"
        )


@torch.inference_mode()
def test_fuse_qkv_with_trtllm_cache_insertion():
    """Chain QKV fusion → TRT-LLM cache insertion and verify the pipeline works.

    This tests that fuse_gemms_mixed_children produces correct meta['val']
    shapes that the insert_cached_attention transform can consume when using
    the TRT-LLM attention backend.
    """
    model = QKVAttentionModel(hidden_size=64, num_heads=4).to(device="cuda", dtype=torch.float16)
    x = model.get_input(device="cuda", dtype=torch.float16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    gm = InferenceOptimizer(
        None,
        {"fuse_gemms_mixed_children": {"stage": "post_load_fusion"}},
    )(None, gm)

    assert _count_narrow_nodes(gm) == 3
    assert sum(is_linear_op(n) for n in gm.graph.nodes) == 2

    kv_cache_config = KvCacheConfig(
        tokens_per_block=32,
        max_tokens=128,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=64,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    gm = InferenceOptimizer(
        None,
        {"insert_cached_attention": {"stage": "cache_init", "backend": "trtllm"}},
    )(cm, gm)

    cached_attn_nodes = [
        n
        for n in gm.graph.nodes
        if is_op(n, torch.ops.auto_deploy.trtllm_attention_mha_with_cache.default)
    ]
    assert len(cached_attn_nodes) == 1, (
        f"Expected 1 trtllm_attention_mha_with_cache node, got {len(cached_attn_nodes)}"
    )

    prep_meta_nodes = [
        n
        for n in gm.graph.nodes
        if is_op(n, torch.ops.auto_deploy.trtllm_attention_prepare_metadata.default)
    ]
    assert len(prep_meta_nodes) == 1, (
        f"Expected 1 prepare_metadata node, got {len(prep_meta_nodes)}"
    )

    assert _count_narrow_nodes(gm) == 3


@torch.inference_mode()
def test_fuse_qkv_gqa_with_trtllm_cache_insertion():
    """Same pipeline but with GQA (num_kv_heads < num_heads).

    Verifies that asymmetric Q/KV projection sizes work through the full
    fusion → cache insertion pipeline.
    """
    model = QKVAttentionModel(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
    ).to(device="cuda", dtype=torch.float16)
    x = model.get_input(device="cuda", dtype=torch.float16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    gm = InferenceOptimizer(
        None,
        {"fuse_gemms_mixed_children": {"stage": "post_load_fusion"}},
    )(None, gm)

    assert _count_narrow_nodes(gm) == 3

    narrow_sizes = sorted([n.args[3] for n in _get_narrow_nodes(gm)])
    assert narrow_sizes == [32, 32, 64], f"Unexpected narrow sizes for GQA: {narrow_sizes}"

    kv_cache_config = KvCacheConfig(
        tokens_per_block=32,
        max_tokens=128,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=64,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    gm = InferenceOptimizer(
        None,
        {"insert_cached_attention": {"stage": "cache_init", "backend": "trtllm"}},
    )(cm, gm)

    cached_attn_nodes = [
        n
        for n in gm.graph.nodes
        if is_op(n, torch.ops.auto_deploy.trtllm_attention_mha_with_cache.default)
    ]
    assert len(cached_attn_nodes) == 1
