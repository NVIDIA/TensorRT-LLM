"""Tests for basic bf16 GEMMs."""

import pytest
import torch
import torch.nn as nn


class GemmCat(nn.Module):
    """Two GEMMs + concat."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(self.in_features, self.out_features, bias=False)
        self.fc2 = nn.Linear(self.in_features, self.out_features, bias=False)

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        return torch.cat([y1, y2], dim=-1)


class GemmBig(nn.Module):
    """One big GEMM."""

    def __init__(self, model1: GemmCat):
        super().__init__()
        # initialize from model
        self.fc_big = nn.Linear(model1.in_features, 2 * model1.out_features, bias=False)
        self.fc_big.load_state_dict(
            {"weight": torch.cat(list(model1.state_dict().values()), dim=0)}
        )

    def forward(self, x):
        return self.fc_big(x)

    @property
    def num_gemms_after_fusion(self) -> int:
        return 1


@pytest.mark.parametrize(
    "dtype",
    (
        "float32",
        "float16",
        pytest.param("bfloat16", marks=pytest.mark.xfail(reason="Inconsistent bf16 GEMM kernels")),
    ),
)
@pytest.mark.parametrize(
    "batch_size, in_features, out_features",
    [
        (4, 10, 10),
        (4, 2048, 128),
        (4, 2048, 135),
        (2, 2048, 135),
        (4, 2048, 136),
    ],
)
@torch.inference_mode()
def test_gemm_cat(dtype: str, batch_size: int, in_features: int, out_features: int):
    # some basic config stuff
    torch_dtype = getattr(torch, dtype)
    model1 = GemmCat(in_features, out_features).to(device="cuda", dtype=torch_dtype)
    model2 = GemmBig(model1).to(device="cuda", dtype=torch_dtype)

    x = torch.ones(batch_size, in_features, device="cuda", dtype=torch_dtype)

    y1 = model1(x)
    y2 = model2(x)
    assert torch.allclose(y1, y2, atol=1e-3, rtol=1e-3)
