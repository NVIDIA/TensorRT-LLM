# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP8 drafter down / o_proj split-K routes' kernel on the installed FlashInfer (``drafter_down_splitk``).

The installed FlashInfer must be admitted by behavior (the split-K entry points exist and accept both pinned tactics at
every admitted row count), and its output at those tactics must be the BF16 product of the projection within one BF16
rounding of an fp32 reference, run to run identical (a fixed tactic: the same split order on every call).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
        drafter_down_splitk as dispatch,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)

_ROUTES = (
    ("down", dispatch.TACTIC, dispatch.DOWN_SHAPE),
    ("oproj", dispatch.OPROJ_TACTIC, dispatch.OPROJ_SHAPE),
)


@pytest.fixture(scope="module")
def provider():
    module = dispatch._load_provider()
    assert module is not None, (
        "the installed FlashInfer lacks the split-K entry points the route calls"
    )
    return module


@pytest.mark.parametrize("name,tactic_args,shape", _ROUTES, ids=[r[0] for r in _ROUTES])
def test_installed_flashinfer_admits_the_pinned_tactic(provider, name, tactic_args, shape):
    assert dispatch._tactic_admitted(provider, tactic_args, shape), name


@pytest.mark.parametrize("name,tactic_args,shape", _ROUTES, ids=[r[0] for r in _ROUTES])
@pytest.mark.parametrize("rows", list(dispatch.ROWS))
def test_split_k_product_matches_the_fp32_reference(provider, name, tactic_args, shape, rows):
    out_features, in_features = shape
    g = torch.Generator(device="cuda").manual_seed(1000 * rows + out_features + in_features)
    x = torch.randn(rows, in_features, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(out_features, in_features, device="cuda", generator=g) * 0.02).to(
        torch.bfloat16
    )
    tactic = provider.SplitKTactic(*tactic_args)
    out = torch.empty((rows, out_features), dtype=torch.bfloat16, device="cuda")
    again = torch.empty_like(out)
    provider.run_splitk_dense(x, w.t(), None, out, True, tactic)
    provider.run_splitk_dense(x, w.t(), None, again, True, tactic)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), again.view(torch.int16)), (
        f"{name}: two calls at one tactic differ"
    )
    ref = x.float() @ w.float().t()
    # One BF16 rounding of the fp32 sum plus the fp32 accumulation-order difference.
    tol = ref.abs() * 2.0**-8 + 1e-3
    bad = int(((out.float() - ref).abs() > tol).sum().item())
    assert bad == 0, f"{name} rows={rows}: {bad} elements outside one BF16 rounding of fp32"
