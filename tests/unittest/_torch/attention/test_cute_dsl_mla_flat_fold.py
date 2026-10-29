# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTe DSL MLA decode with Kimi K3's 96 heads.

FP8: the flat (q_tok, head) fold and the balanced split. FP8 and FP16/BF16: a
large uniform split keeps its LSE partials inside the workspace.
"""

import math

import pytest
import torch

_HAVE_DEPS = True
_DEP_ERR = None
try:
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner
except ImportError as e:
    _HAVE_DEPS = False
    _DEP_ERR = str(e)


def _is_supported_sm():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) in ((10, 0), (10, 3), (10, 7))


pytestmark = [
    pytest.mark.skipif(not _is_supported_sm(), reason="needs sm100/sm103/sm107"),
    pytest.mark.skipif(not _HAVE_DEPS, reason=f"deps: {_DEP_ERR}"),
]

D, R, PAGE, H = 512, 64, 64, 96
_MAX_BATCH = 64
# the compiled-kernel cache is keyed by shape and tactic, not by the fold
_KERNEL_CACHES = {"0": {}, "1": {}}


def _inputs(lens, S, helix, seed):
    torch.manual_seed(seed)
    pages = [math.ceil(L / PAGE) for L in lens]
    perm = torch.randperm(sum(pages) + 8, device="cuda").to(torch.int32)
    pt = torch.zeros(len(lens), max(pages) + 1, dtype=torch.int32, device="cuda")
    off = 0
    for b, n in enumerate(pages):
        pt[b, :n] = perm[off : off + n]
        off += n
    kv = (torch.randn(perm.numel(), PAGE, D + R, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    q = (torch.randn(len(lens), S, H, D + R, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    bounds = None
    if helix:
        # rank-local verify bounds: each token sees between K - S_q and K entries
        bounds = [int(torch.randint(max(0, L - S), L + 1, (1,))) for L in lens for _ in range(S)]
        bounds = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    return dict(lens=lens, q=q, kv=kv, pt=pt, bounds=bounds)


def _run(x, S, split_kv, flat_fold, monkeypatch):
    """(O, LSE, Helix stats); NaN guard rows check nothing is stored past the batch."""
    flag = "1" if flat_fold else "0"
    monkeypatch.setenv("TRTLLM_MLA_FLAT_FOLD", flag)
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "kernel_cache", _KERNEL_CACHES[flag])
    B, helix = len(x["lens"]), x["bounds"] is not None
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN,
        num_heads=H,
        seq_len_q=S,
        page_size=PAGE,
        max_batch_size=_MAX_BATCH,
        emit_softmax_stats=helix,
    )
    n_o = B * S * H * D
    o_buf = torch.full((n_o + S * H * D,), float("nan"), dtype=torch.bfloat16, device="cuda")
    lse_off, lse_size, _, _, ws_size = runner.get_workspace_layout(
        H, S, D, _MAX_BATCH, cutlass.Float32
    )
    ws = torch.full((ws_size,), 0xFF, dtype=torch.uint8, device="cuda")
    stats = torch.full(((B + 1) * S, H, 2), float("nan"), device="cuda") if helix else None
    q, kv = x["q"], x["kv"]
    inputs = [
        q[..., :D].permute(2, 3, 1, 0),
        q[..., D:].permute(2, 3, 1, 0),
        kv[..., :D].permute(1, 2, 0),
        kv[..., D:].permute(1, 2, 0),
        x["pt"].t(),
        torch.tensor(x["lens"], dtype=torch.int32, device="cuda"),
        o_buf[:n_o].view(B, S, H, D).permute(2, 3, 1, 0),
        ws,
        stats[: B * S] if helix else None,
        x["bounds"],
        torch.tensor([1.0 / math.sqrt(192)], device="cuda"),
        torch.tensor([1.0], device="cuda"),
    ]
    runner.forward(inputs, ((128, 128), (128, 256), split_kv, False))
    torch.cuda.synchronize()
    lse = ws[lse_off : lse_off + lse_size].view(torch.float32).view(_MAX_BATCH, S, H)
    assert torch.isnan(o_buf[n_o:].float()).all()
    assert torch.isnan(lse[B:]).all()
    if helix:
        assert torch.isnan(stats[B * S :]).all()
        stats = stats[: B * S].clone()
    return o_buf[:n_o].view(B, S, H, D).clone(), lse[:B].clone(), stats


def _reference(x, S):
    """fp32 MTP-causal (or Helix-bounded) attention."""
    q, kv = x["q"].float(), x["kv"].float()
    out = torch.zeros(len(x["lens"]), S, H, D, device="cuda")
    for b, L in enumerate(x["lens"]):
        c = kv[x["pt"][b, : math.ceil(L / PAGE)].long()].reshape(-1, D + R)[:L]
        for s in range(S):
            bound = L - (S - 1) + s if x["bounds"] is None else int(x["bounds"][b * S + s])
            if bound > 0:
                p = torch.softmax(q[b, s] @ c[:bound].T / math.sqrt(192), dim=-1)
                out[b, s] = p @ c[:bound, :D]
    return out


@pytest.mark.parametrize("helix", [False, True])
@pytest.mark.parametrize("split_kv", [1, 4])
@pytest.mark.parametrize("S", [4, 7, 8])  # 4 and 8 fill whole M tiles, 7 pads the last one
def test_flat_fold_is_bitwise_identical(S, split_kv, helix, monkeypatch):
    x = _inputs([3000, 70, 1500, 8191, 64, 513, 2048, 4500], S, helix, seed=11)
    ref = _run(x, S, split_kv, False, monkeypatch)
    new = _run(x, S, split_kv, True, monkeypatch)
    for a, b in zip(new, ref):
        if a is not None:
            assert torch.equal(a, b)
    gold = _reference(x, S)
    assert (new[0].float() - gold).abs().max() <= 0.05 * gold.abs().max()


@pytest.mark.parametrize("lookahead", [True, False])
@pytest.mark.parametrize("helix", [False, True])
@pytest.mark.parametrize("S", [4, 7])
def test_balanced_split(S, helix, lookahead, monkeypatch):
    # forced on: unset, the balanced split and its lookahead skip Helix
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_BALANCED_SPLIT", True)
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_MMA_LOOKAHEAD", lookahead)
    lens = [600, 0, 61000, 2048, 9000, 0, 130, 15000, 7300, 64]  # skewed, two without KV
    x = _inputs(lens, S, helix, seed=13)
    ref_o, ref_lse, _ = _run(x, S, 1, True, monkeypatch)
    o, lse, stats = _run(
        x, S, CuteDSLNVMlaDecodeBlackwellRunner.BALANCED_SPLIT_TACTIC, True, monkeypatch
    )
    kv = torch.tensor([L > 0 for L in lens], device="cuda")
    gold = _reference(x, S)
    tol = 0.05 * gold.abs().max()
    # tiles cut across clusters round P to fp8 per piece
    assert (o[kv].float() - ref_o[kv].float()).abs().max() <= tol
    assert (o[kv].float() - gold[kv]).abs().max() <= tol
    assert (lse[kv] - ref_lse[kv]).abs().max() <= 1e-4
    assert (o[~kv] == 0).all() and torch.isinf(lse[~kv]).all()
    if helix:
        st = stats.view(len(lens), S, H, 2)[~kv]
        assert torch.isinf(st[..., 0]).all() and (st[..., 1] == 0).all()


# fp32 split-KV partials whose bytes-first acc_lse offset (cosize * 4) no longer fits Int32
_LSE_OFFSET_WRAP_ELEMENTS = 1 << 29


def _clear_split_kv_sizing():
    # Cached per shape and occupancy, not per _SPLIT_KV_WAVES.
    CuteDSLNVMlaDecodeBlackwellRunner._max_split_kv_rows.cache_clear()
    CuteDSLNVMlaDecodeBlackwellRunner._max_split_kv_acc_rows.cache_clear()


@pytest.fixture
def uniform_split_kv_32(monkeypatch):
    """Size the split-KV workspace for split_kv 32 at every batch bucket and keep the
    uniform split, whose initialize_workspace places acc_lse behind the O partials."""
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_KV_WAVES", 1 << 16)
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_BALANCED_SPLIT", False)
    _clear_split_kv_sizing()
    yield
    _clear_split_kv_sizing()


@pytest.mark.parametrize("in_dtype", ["fp8", "bf16"])
def test_large_split_partials_keep_lse_in_workspace(in_dtype, uniform_split_kv_32, monkeypatch):
    """B * split_kv * rows * D >= 2^29 fp32 partials: the acc_lse byte offset wrapped in
    Int32 and put acc_lse before the workspace (NaN outputs or an illegal address)."""
    monkeypatch.setenv("TRTLLM_MLA_FLAT_FOLD", "1")
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "kernel_cache", _KERNEL_CACHES["1"])
    fp8 = in_dtype == "fp8"
    S, B, max_batch = 4, 96, 128
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN if fp8 else cutlass.BFloat16,
        num_heads=H,
        seq_len_q=S,
        page_size=PAGE,
        max_batch_size=max_batch,
    )
    assert not runner._uses_balanced_split(B)
    split_kv = min(32, runner.get_fitting_split_kv(B, max_batch))
    rows = runner._split_kv_rows_per_request()
    assert split_kv == 32, split_kv
    assert B * split_kv * rows * D >= _LSE_OFFSET_WRAP_ELEMENTS, rows
    lens = [4096 + 37 * i for i in range(B)]
    x = _inputs(lens, S, False, seed=5)
    q, kv = x["q"], x["kv"]
    if not fp8:
        # fp8 values are exact in bf16, so the fp32 reference is shared
        q, kv = q.to(torch.bfloat16), kv.to(torch.bfloat16)
    size = runner.get_workspace_layout(H, S, D, max_batch, cutlass.Float32)[4]
    canary = 1 << 20
    ws = torch.full((size + canary,), 0x5A, dtype=torch.uint8, device="cuda")
    o = torch.full((B, S, H, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    inputs = [
        q[..., :D].permute(2, 3, 1, 0),
        q[..., D:].permute(2, 3, 1, 0),
        kv[..., :D].permute(1, 2, 0),
        kv[..., D:].permute(1, 2, 0),
        x["pt"].t(),
        torch.tensor(lens, dtype=torch.int32, device="cuda"),
        o.permute(2, 3, 1, 0),
        ws[:size],
        None,
        None,
    ]
    tactic = ((128, 128), (128, 256), split_kv, False)
    scale = 1.0 / math.sqrt(192)
    if fp8:
        inputs += [torch.tensor([scale], device="cuda"), torch.tensor([1.0], device="cuda")]
        runner.forward(inputs, tactic)
    else:
        runner.forward(inputs, tactic, softmax_scale=scale, output_scale=1.0)
    torch.cuda.synchronize()
    assert bool((ws[size:] == 0x5A).all()), "split-KV partials overflowed the workspace"
    out = o.float()
    assert torch.isfinite(out).all()
    gold = _reference(x, S)
    err = (out - gold).abs().max().item()
    assert err <= 0.05 * gold.abs().max().item(), err
