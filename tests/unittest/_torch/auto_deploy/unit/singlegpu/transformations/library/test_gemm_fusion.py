"""Tests for basic GEMM fusion."""

from abc import abstractmethod
from typing import Callable, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _graph_test_helpers import count_buffers, run_test_transformed_gm
from _model_test_utils import FakeFP8Linear
from _torch_test_utils import all_close, fp8_compatible, reset_parameters

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op, is_op

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
