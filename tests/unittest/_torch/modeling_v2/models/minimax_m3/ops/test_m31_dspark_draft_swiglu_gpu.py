# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_SWIGLU: the sm_107 TP2 / TP8 standalone DSpark drafter's GatedMLP SwiGLU-OAI on the family
NATIVE_SWIGLU hook (``minimax_m3/ops/native_swiglu``) instead of the Triton ``silu_and_mul``.

- The row grid (nv 0) of the family op ``minimax_m31_swiglu_oai_vec`` is bit for bit the Triton kernel at the
  drafter's widths (d 6144 per rank at TP2, 1536 at TP8) and rows (the 7-token block: 7, and B x 7 up to 56; 1 / 2 /
  8), PDL on and off, with clamped, signed-zero, subnormal and non-finite values, eagerly and inside a CUDA graph.
- Each target's ``install_draft_swiglu`` wraps the drafter's 5 MLPs once, and an installed MLP never reaches the
  Triton op (the stock ``swiglu`` is replaced by one that raises): on the op path, and on the kill-switch path
  (TRTLLM_M31_DISABLE_NATIVE_SWIGLU=1, the torch SwiGLU-OAI).
- Each target installs it in ``M3DSparkWorker.prebuild`` before the MXFP8 drafter routes its projections (TP8: before
  DRAFT_GU_SWIGLU_EPI wraps the same activation, so its epilogue rows still pass through).
- No Triton import or ``silu_and_mul`` call in the family module or the targets' install.
"""

import ast
import importlib
import inspect
import os
import subprocess
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import native_swiglu
from tensorrt_llm._torch.modules import gated_mlp
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._utils import get_sm_version

PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"
TARGETS = {
    "tp2": f"{PKG}.minimax_m3_1_flash_nvfp4__sm_107__tp2.dspark",
    "tp8": f"{PKG}.minimax_m3_1_flash_nvfp4__sm_107__tp8.dspark",
}
WIDTH = {"tp2": 6144, "tp8": 1536}
DRAFTER_LAYERS = 5

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
needs_op = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the native SwiGLU-OAI op runs on sm_107 (Rubin) only",
)


def _triton(x):
    return torch.ops.trtllm.silu_and_mul(x, swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0)


def _gu(rows, d, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, 2 * d, generator=gen, device="cuda") * 4.0).to(torch.bfloat16)
    gate = [
        7.0,
        7.0625,
        9.0,
        float("inf"),
        float("-inf"),
        float("nan"),
        -0.0,
        1e-40,
        -1e-39,
        -60.0,
        3e38,
    ]
    up = [
        7.0,
        -7.0,
        11.0,
        -11.0,
        float("inf"),
        float("-inf"),
        float("nan"),
        -0.0,
        1e-40,
        -1e-39,
        3e38,
    ]
    for k, v in enumerate(gate):
        x[k % rows, (7 * k + 1) % d] = v
    for k, v in enumerate(up):
        x[(k + 1) % rows, d + (5 * k + 3) % d] = v
    return x


def _assert_bitwise(got, want):
    """Bit for bit, NaN positions included (a NaN may differ only in its payload)."""
    assert got.shape == want.shape and got.dtype == want.dtype
    nan_got, nan_want = torch.isnan(got), torch.isnan(want)
    assert torch.equal(nan_got, nan_want)
    keep = ~nan_want
    assert torch.equal(got.view(torch.int16)[keep], want.view(torch.int16)[keep])


@pytest.mark.parametrize("topo", ["tp2", "tp8"])
@pytest.mark.parametrize("rows", [1, 2, 7, 8, 14, 21, 28, 56])
@pytest.mark.parametrize("pdl", [True, False])
@needs_op
def test_the_row_grid_is_bitwise_the_triton_kernel_at_the_drafter_shapes(topo, rows, pdl):
    x = _gu(rows, WIDTH[topo], rows * 10 + int(pdl))
    _assert_bitwise(native_swiglu.swiglu_op()(x, pdl, 0), _triton(x))


@pytest.mark.parametrize("topo", ["tp2", "tp8"])
@needs_op
def test_the_row_grid_replays_inside_a_cuda_graph(topo):
    d = WIDTH[topo]
    x = _gu(7, d, 3)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        native_swiglu.swiglu_oai(x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = native_swiglu.swiglu_oai(x)
    x.copy_(_gu(7, d, 4))
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(out, _triton(x))


def _mlp():
    """A GatedMLP shell carrying what ``native_swiglu.serves`` and the stock ``_apply_activation`` read: the MiniMax
    DSpark drafter's dense SwiGLU-OAI MLP (modeling_dspark.MiniMaxDSparkDecoderLayer), bf16 down projection."""
    mlp = GatedMLP.__new__(GatedMLP)
    torch.nn.Module.__init__(mlp)
    mlp.activation = F.silu
    mlp.split_gate_up = False
    mlp.swiglu_alpha, mlp.swiglu_beta, mlp.swiglu_limit = 1.702, 1.0, 7.0
    mlp.layer_idx = 0
    mlp.down_proj = types.SimpleNamespace(has_fp8_qdq=False, has_w4a8_nvfp4_fp8=False)
    return mlp


def _drafter():
    layers = [types.SimpleNamespace(mlp=_mlp()) for _ in range(DRAFTER_LAYERS)]
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))


@pytest.fixture
def no_triton(monkeypatch):
    """The stock GatedMLP activation's Triton entry raises."""

    def boom(*args, **kwargs):
        raise AssertionError("the Triton silu_and_mul was called")

    monkeypatch.setattr(gated_mlp, "swiglu", boom)


@pytest.mark.parametrize("topo", ["tp2", "tp8"])
@needs_op
def test_the_target_installs_the_hook_on_the_drafter_mlps_once(topo, no_triton):
    dspark = importlib.import_module(TARGETS[topo])
    draft = _drafter()
    x = _gu(7, WIDTH[topo], 11)
    want = _triton(x)
    assert dspark.install_draft_swiglu(draft) == DRAFTER_LAYERS
    assert dspark.install_draft_swiglu(draft) == 0  # once per instance
    assert (
        dspark.install_draft_swiglu(types.SimpleNamespace()) == 0
    )  # a shell worker without a drafter
    for layer in draft.model.layers:
        _assert_bitwise(layer.mlp._apply_activation(x), want)
        _assert_bitwise(layer.mlp._apply_activation(x, has_lora=False), want)


@pytest.mark.parametrize("topo", ["tp2", "tp8"])
@needs_gpu
def test_the_kill_switch_runs_the_torch_swiglu_not_triton(topo, no_triton, monkeypatch):
    dspark = importlib.import_module(TARGETS[topo])
    monkeypatch.setattr(native_swiglu, "NATIVE_SWIGLU", False)
    draft = _drafter()
    assert dspark.install_draft_swiglu(draft) == DRAFTER_LAYERS
    x = _gu(7, WIDTH[topo], 13)
    got = draft.model.layers[0].mlp._apply_activation(x)
    torch.testing.assert_close(
        got, native_swiglu.swiglu_oai_torch(x), rtol=0, atol=0, equal_nan=True
    )
    finite = (torch.randn(7, 2 * WIDTH[topo], device="cuda") * 4.0).to(torch.bfloat16)
    torch.testing.assert_close(
        native_swiglu.swiglu_oai_torch(finite).float(),
        _triton(finite).float(),
        atol=2e-2,
        rtol=2e-2,
    )


def test_the_kill_switch_is_read_at_import():
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRTLLM_M31_DISABLE_")}
    code = f"from {PKG}.ops import native_swiglu\nprint(native_swiglu.NATIVE_SWIGLU)"
    for value, on in (("", True), ("1", False), ("0", True)):
        env["TRTLLM_M31_DISABLE_NATIVE_SWIGLU"] = value
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        assert out.stdout.split()[-1] == str(on)


@pytest.mark.parametrize(
    "topo, later",
    [
        ("tp2", "route_mx_linears(draft_model"),
        ("tp8", "route_gu_swiglu(draft_model"),
    ],
)
def test_the_install_runs_in_prebuild_before_the_mxfp8_drafter(topo, later):
    dspark = importlib.import_module(TARGETS[topo])
    src = inspect.getsource(dspark.M3DSparkWorker.prebuild)
    assert src.count("install_draft_swiglu(draft_model)") == 1
    assert src.index("install_draft_swiglu(draft_model)") < src.index(later)


def _calls_and_imports(source):
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


@pytest.mark.parametrize("topo", ["tp2", "tp8"])
def test_no_triton_on_the_hook(topo):
    dspark = importlib.import_module(TARGETS[topo])
    for source in (
        inspect.getsource(native_swiglu),
        inspect.getsource(dspark.install_draft_swiglu),
    ):
        names = _calls_and_imports(source)
        assert not {"silu_and_mul", "swiglu", "triton"} & names
        assert not any(n.startswith("triton") for n in names)
