# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (SM103): the lane-split softmax of `qsa_paged_sparse_gqa_fp8_verify`
(`TRTLLM_QWEN4_EXP_DP_GQA_SOFTMAX_SPLIT`, `softmax_split=True`) is bitwise the default kernel (same
S, reference max, P, O and row-sum order), and inside the op's band of the fp32 reference, over
one and several splits, the two-wave and tail plans, gated and ungated, a partial last group, two
rows a request, an invalid request, the wide table, and eager against graph replay."""

import importlib.util
import os

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention import (  # noqa: E501
    qsa_paged_sparse_gqa_fp8_verify as _verify_module,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_fp8_verify import (  # noqa: E501
    qsa_paged_sparse_gqa_fp8_verify as op,
)

assert torch.cuda.is_available(), "qsa_paged_sparse_gqa_fp8_verify requires a CUDA device"
if torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the bitmap-union e4m3 verify kernel serves SM103", allow_module_level=True)

_spec = importlib.util.spec_from_file_location(
    "_fp8_verify_cases",
    os.path.join(os.path.dirname(__file__), "test_modeling_v2_qsa_paged_sparse_gqa_fp8_verify.py"),
)
_cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cases)
build, within_ulp, _segments, DEV = _cases.build, _cases.within_ulp, _cases._segments, _cases.DEV

grouped = _verify_module.fp8_sparse_gqa_verify_grouped


def _gate(args, seed):
    rows = args["q"].shape[0]
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(rows, 24 * 256, generator=g, device=DEV).to(torch.bfloat16)


def _pair(args, **kw):
    base = grouped(**args, softmax_split=False, **kw)
    s2 = grouped(**args, softmax_split=True, **kw)
    return base, s2


def _check(args, base, s2, rpr=4):
    assert s2.dtype == torch.bfloat16 and not s2.isnan().any()
    ref = op.reference(**args, rows_per_request=rpr)
    el, rms, ok = within_ulp(s2, ref)
    same = (s2 == base).float().mean().item()
    print(f"[s2] split vs ref {el:.2f}/{rms:.2f} ulp; {100 * same:.3f} % bitwise the default")
    assert ok, f"split vs reference: {el:.2f} ulp element, {rms:.2f} ulp RMS"
    assert torch.equal(s2, base), "the split kernel is bitwise the default kernel"


@pytest.mark.parametrize("gated", [False, True], ids=["ungated", "gated"])
@pytest.mark.parametrize(
    "requests, splits",
    [(8, None), (12, None), (24, None), (24, 1), (40, None), (90, None), (12, 2)],
    ids=str,
)
def test_split_matches_default(requests, splits, gated) -> None:
    args = build(_segments(requests, first=3000, step=1301), seed=900 + requests, pages=8190)
    if gated:
        args["output_gate"] = _gate(args, requests)
    base, s2 = _pair(args, rows_per_request=4, splits=splits, min_groups=2)
    _check(args, base, s2)


def test_partial_group_and_invalid_request() -> None:
    # 9 requests, the last with 3 rows (a partial group), request 4 invalid
    segs = _segments(8, first=5000, step=211) + ((9000, 3),)
    args = build(segs, seed=4242, pages=516)
    args["request_indices"][17] = -1
    base, s2 = _pair(args, rows_per_request=4, min_groups=2)
    assert not s2[17].any()
    assert torch.equal(s2, base)


def test_two_rows_a_request() -> None:
    args = build(_segments(16, rows_each=2, first=7000, step=503), seed=31337)
    base, s2 = _pair(args, rows_per_request=2, min_groups=2)
    _check(args, base, s2, rpr=2)


def test_split_is_deterministic_and_replays() -> None:
    segments = tuple((100 + (499 * g) % 8000, 4) for g in range(12))
    args = build(segments, seed=77)
    fresh = build(segments, seed=127)
    a = grouped(**args, rows_per_request=4, min_groups=2, softmax_split=True)
    assert torch.equal(a, grouped(**args, rows_per_request=4, min_groups=2, softmax_split=True))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = grouped(**args, rows_per_request=4, min_groups=2, softmax_split=True)
    for name in ("q", "k_cache", "v_cache", "block_table", "selected_tokens", "request_indices"):
        args[name].copy_(fresh[name])
    graph.replay()
    torch.cuda.synchronize()
    again = grouped(**fresh, rows_per_request=4, min_groups=2, softmax_split=True)
    assert torch.equal(captured, again)


def test_switch_default_off() -> None:
    assert _verify_module.SOFTMAX_SPLIT_ENV == "TRTLLM_QWEN4_EXP_DP_GQA_SOFTMAX_SPLIT"
    if os.environ.get(_verify_module.SOFTMAX_SPLIT_ENV) is None:
        assert not _verify_module.SOFTMAX_SPLIT
    assert not _verify_module._softmax_split_fits(5, 12)
    assert not _verify_module._softmax_split_fits(4, 17)
