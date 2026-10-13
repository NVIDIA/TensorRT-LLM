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

"""CuTe DSL FP8 MLA decode: flat (q_tok, head) fold vs the unfolded kernel.

K3 has 96 MLA heads and verifies 1 + 3 draft tokens per request. 96 heads fill
only 3/4 of the 128-row M tile and no integer token fold fits, so the kernel
used to run S_q = 4 tiles of 96 valid rows per request. The flat fold packs the
384 (q_tok, head) rows into 3 full M tiles; each row's math is unchanged, so the
output must be BITWISE identical to the unfolded kernel (TRTLLM_MLA_FLAT_FOLD=0)
at any split_kv, and agree with an fp32 reference of the MTP-causal attention.

Also covers the opt-in split-KV policies (multi-wave heuristic and per-batch
table) and that their split stays inside the runner's workspace.

Requires 1 SM100/SM103/SM107 GPU and nvidia-cutlass-dsl.
"""

import math

import pytest
import torch

_HAVE_DEPS = True
_DEP_ERR = None
try:
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        CuteDSLNVMlaDecodeBlackwellRunner,
        _parse_mla_split_kv_table,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.attention.mla.mla_decode_fp8 import (
        BlackwellMultiHeadLatentAttentionForwardFP8,
    )
except ImportError as e:
    _HAVE_DEPS = False
    _DEP_ERR = str(e)


def _is_sm100_family():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) in ((10, 0), (10, 3), (10, 7))


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _is_sm100_family(), reason="needs sm100/sm103/sm107"),
    pytest.mark.skipif(not _HAVE_DEPS, reason=f"deps: {_DEP_ERR}"),
]

D, R, PAGE = 512, 64, 64


def _make_inputs(lens, S, H, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    B = len(lens)
    pages_per = [math.ceil(L / PAGE) for L in lens]
    max_blocks = max(pages_per) + 1
    num_pages = sum(pages_per) + 8
    perm = torch.randperm(num_pages, device=dev).to(torch.int32)
    pt = torch.zeros(B, max_blocks, dtype=torch.int32, device=dev)
    off = 0
    for b, n in enumerate(pages_per):
        pt[b, :n] = perm[off : off + n]
        off += n
    kv = (torch.randn(num_pages, 1, PAGE, 1, D + R, device=dev) * 0.5).to(torch.float8_e4m3fn)
    kv_pages = kv[:, 0, :, 0, :]
    q = (torch.randn(B, S, H, D + R, device=dev) * 0.5).to(torch.float8_e4m3fn)
    o = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=dev)
    return dict(
        q=q,
        kv_pages=kv_pages,
        pt=pt,
        o=o,
        inputs=[
            q[..., :D].permute(2, 3, 1, 0),
            q[..., D:].permute(2, 3, 1, 0),
            kv_pages[..., :D].permute(1, 2, 0),
            kv_pages[..., D:].permute(1, 2, 0),
            pt.transpose(0, 1),
            torch.tensor(lens, dtype=torch.int32, device=dev),
            o.permute(2, 3, 1, 0),
            torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=dev),
            None,
            None,
            torch.tensor([1.0 / math.sqrt(192)], dtype=torch.float32, device=dev),
            torch.tensor([1.0], dtype=torch.float32, device=dev),
        ],
    )


def _run(x, H, S, split_kv, flat_fold, monkeypatch):
    monkeypatch.setenv("TRTLLM_MLA_FLAT_FOLD", "1" if flat_fold else "0")
    # the compiled-kernel cache is keyed by shape/tactic only
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "kernel_cache", {})
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN, num_heads=H, seq_len_q=S, page_size=PAGE, max_batch_size=64
    )
    runner.forward(x["inputs"], ((128, 128), (128, 256), split_kv, False))
    torch.cuda.synchronize()
    return x["o"].clone()


def _reference(x, lens, S, H):
    q = x["q"].float()
    kv = x["kv_pages"].float()
    out = torch.empty(len(lens), S, H, D, dtype=torch.float32, device="cuda")
    scale = 1.0 / math.sqrt(192)
    for b, L in enumerate(lens):
        pages = x["pt"][b, : math.ceil(L / PAGE)].long()
        c = kv[pages].reshape(-1, D + R)[:L]
        for s in range(S):
            bound = L - (S - 1) + s
            logits = (q[b, s] @ c[:bound].T) * scale  # [H, bound]
            p = torch.softmax(logits, dim=-1)
            out[b, s] = p @ c[:bound, :D]
    return out


def test_k3_shape_takes_the_flat_fold(monkeypatch):
    monkeypatch.setenv("TRTLLM_MLA_FLAT_FOLD", "1")
    k = BlackwellMultiHeadLatentAttentionForwardFP8(
        cutlass.Float32,
        cutlass.Float32,
        (128, 128),
        (128, 256),
        76,
        PAGE,
        0.0,
        False,
        True,
        False,
        num_heads=96,
        seq_len_q=4,
        fold_sq=True,
    )
    assert k.fold_rows == 128 and k.flat_fold
    # integer folds are unchanged
    k32 = BlackwellMultiHeadLatentAttentionForwardFP8(
        cutlass.Float32,
        cutlass.Float32,
        (128, 128),
        (128, 256),
        76,
        PAGE,
        0.0,
        False,
        True,
        False,
        num_heads=32,
        seq_len_q=4,
        fold_sq=True,
    )
    assert k32.fold_rows == 128 and not k32.flat_fold


@pytest.mark.parametrize("split_kv", [1, 2, 4])
@pytest.mark.parametrize("H,S", [(96, 4), (48, 8)])
def test_flat_fold_is_bitwise_identical(H, S, split_kv, monkeypatch):
    lens = [3000, 70, 1500, 8191, 64, 513, 2048, 4500]
    x = _make_inputs(lens, S, H, seed=7)
    ref = _run(x, H, S, split_kv, flat_fold=False, monkeypatch=monkeypatch)
    new = _run(x, H, S, split_kv, flat_fold=True, monkeypatch=monkeypatch)
    assert torch.equal(new, ref)
    gold = _reference(x, lens, S, H)
    err = (new.float() - gold).abs().max().item()
    assert err <= 0.05 * gold.abs().max().item(), err


def test_multi_wave_split_heuristic(monkeypatch):
    R = CuteDSLNVMlaDecodeBlackwellRunner
    monkeypatch.setattr(R, "_cute_dsl_max_active_blocks", 152, raising=False)
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_MIN_WAVES", 0)
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_KV_TABLE", ())
    # default (off): at most one wave -> split 1 once the batch fills the GPU
    assert R.get_split_kv_for_batch(32, 96, 4, 152) == 1
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_MIN_WAVES", 4)
    # K3: 32 requests x 3 flat-folded tiles = 96 clusters, 76 cluster slots
    assert R.get_seq_len_q_eff(96, 4) == 3
    assert R.get_split_kv_for_batch(32, 96, 4, 152) == 4
    assert R.get_split_kv_for_batch(64, 96, 4, 152) == 2
    assert R.get_split_kv_for_batch(1024, 96, 4, 152) == 1
    # never below the default heuristic, never above 32
    assert R.get_split_kv_for_batch(1, 128, 1, 152) == 32
    # the workspace covers the largest batch of every bucket
    ws = R.get_multi_wave_split_kv_workspace_size(96, 4, 512, 64, cutlass.Float32)
    assert ws >= 63 * 4 * 96 * 4 * 513 * 4


def test_split_kv_table(monkeypatch):
    R = CuteDSLNVMlaDecodeBlackwellRunner
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_MIN_WAVES", 0)
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_KV_TABLE", ())
    assert not R.uses_split_policy()
    monkeypatch.setattr(R, "_SPLIT_KV_TABLE", _parse_mla_split_kv_table("64:6, 16:8,32:6,128:4"))
    assert R.uses_split_policy()
    assert R.get_split_kv_table() == ((16, 8), (32, 6), (64, 6), (128, 4))
    # below the smallest key: unchanged default (1 once the batch fills the GPU)
    assert R.get_split_kv_for_batch(8, 96, 4, 212) == R.get_default_split_kv(8, 4, 212)
    assert R.get_split_kv_for_batch(16, 96, 4, 212) == 8
    assert R.get_split_kv_for_batch(40, 96, 4, 212) == 6
    assert R.get_split_kv_for_batch(96, 96, 4, 212) == 6
    assert R.get_split_kv_for_batch(128, 96, 4, 212) == 4
    assert R.get_split_kv_for_batch(4096, 96, 4, 212) == 4
    # the table overrides the multi-wave heuristic where it has an entry
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_MIN_WAVES", 4)
    assert R.get_split_kv_for_batch(64, 96, 4, 212) == 6
    for bad in ("64", "64:0", "0:4", "64:33", "x:2"):
        with pytest.raises(ValueError):
            _parse_mla_split_kv_table(bad)


@pytest.mark.parametrize(
    "env",
    [
        ("TRTLLM_CUTE_DSL_MLA_SPLIT_MIN_WAVES", "4"),
        ("TRTLLM_CUTE_DSL_MLA_SPLIT_KV_TABLE", "16:8,32:6,64:6,128:4"),
    ],
)
def test_split_policy_stays_inside_workspace(env, monkeypatch):
    """Run the policy's split at the largest batch of its bucket with the
    exact workspace size from get_workspace_layout and a canary behind it."""
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_MIN_WAVES", 0)
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_KV_TABLE", ())
    if env[0].endswith("MIN_WAVES"):
        monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_MIN_WAVES", int(env[1]))
    else:
        monkeypatch.setattr(
            CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_KV_TABLE", _parse_mla_split_kv_table(env[1])
        )
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "kernel_cache", {})
    H, S, max_batch = 96, 4, 63
    lens = [64 + (997 * i) % 4000 for i in range(max_batch)]
    x = _make_inputs(lens, S, H, seed=3)
    R = CuteDSLNVMlaDecodeBlackwellRunner
    mab = R._get_max_active_blocks()
    split_kv = R.get_split_kv_for_batch(32, H, S, mab)  # bucket of B = 63
    assert split_kv > 1
    size = R.get_workspace_layout(H, S, D, max_batch, cutlass.Float32)[4]
    canary = 1 << 20
    ws = torch.full((size + canary,), 0x5A, dtype=torch.uint8, device="cuda")
    x["inputs"][7] = ws[:size]
    runner = R(
        in_dtype=cutlass.Float8E4M3FN,
        num_heads=H,
        seq_len_q=S,
        page_size=PAGE,
        max_batch_size=max_batch,
    )
    runner.forward(x["inputs"], ((128, 128), (128, 256), split_kv, False))
    torch.cuda.synchronize()
    assert bool((ws[size:] == 0x5A).all()), "split-KV partials overflowed the workspace"
    gold = _reference(x, lens, S, H)
    err = (x["o"].float() - gold).abs().max().item()
    assert err <= 0.05 * gold.abs().max().item(), err


def test_large_split_partials_keep_lse_in_workspace(monkeypatch):
    """B * split_kv * 384 rows * 512 >= 2^26 fp32 partials: the acc_lse byte
    offset used to overflow int32 (NaN outputs or an illegal address)."""
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_KV_TABLE", ())
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_SPLIT_MIN_WAVES", 64)
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "kernel_cache", {})
    H, S, max_batch = 96, 4, 15
    R = CuteDSLNVMlaDecodeBlackwellRunner
    split_kv = R.get_split_kv_for_batch(8, H, S, R._get_max_active_blocks())
    assert max_batch * split_kv * H * S * D >= 1 << 26, split_kv
    lens = [4096 + 509 * i for i in range(max_batch)]
    x = _make_inputs(lens, S, H, seed=5)
    size = R.get_workspace_layout(H, S, D, max_batch, cutlass.Float32)[4]
    canary = 1 << 20
    ws = torch.full((size + canary,), 0x5A, dtype=torch.uint8, device="cuda")
    x["inputs"][7] = ws[:size]
    runner = R(
        in_dtype=cutlass.Float8E4M3FN,
        num_heads=H,
        seq_len_q=S,
        page_size=PAGE,
        max_batch_size=max_batch,
    )
    runner.forward(x["inputs"], ((128, 128), (128, 256), split_kv, False))
    torch.cuda.synchronize()
    assert bool((ws[size:] == 0x5A).all()), "split-KV partials overflowed the workspace"
    gold = _reference(x, lens, S, H)
    out = x["o"].float()
    assert torch.isfinite(out).all()
    err = (out - gold).abs().max().item()
    assert err <= 0.05 * gold.abs().max().item(), err


def test_multi_wave_policy_ignores_later_environment_changes(monkeypatch):
    R = CuteDSLNVMlaDecodeBlackwellRunner
    monkeypatch.setattr(R, "_cute_dsl_max_active_blocks", 152, raising=False)
    monkeypatch.setattr(R, "_SPLIT_MIN_WAVES", 4)
    before = R.get_workspace_layout(96, 4, 512, 64, cutlass.Float32)
    monkeypatch.setenv("TRTLLM_CUTE_DSL_MLA_SPLIT_MIN_WAVES", "0")
    assert R.get_split_min_waves() == 4
    assert R.get_split_kv_for_batch(32, 96, 4, 152) == 4
    assert R.get_workspace_layout(96, 4, 512, 64, cutlass.Float32) == before


def test_split_table_ignores_later_environment_changes(monkeypatch):
    R = CuteDSLNVMlaDecodeBlackwellRunner
    table = _parse_mla_split_kv_table("64:6,16:8,32:6,128:4")
    monkeypatch.setattr(R, "_SPLIT_KV_TABLE", table)
    monkeypatch.setattr(R, "_SPLIT_MIN_WAVES", 0)
    monkeypatch.setattr(R, "_cute_dsl_max_active_blocks", 152, raising=False)
    before = R.get_workspace_layout(96, 4, 512, 128, cutlass.Float32)
    monkeypatch.setenv("TRTLLM_CUTE_DSL_MLA_SPLIT_KV_TABLE", "malformed")
    assert R.get_split_kv_table() is table
    assert R.get_split_kv_for_batch(64, 96, 4, 152) == 6
    assert R.get_workspace_layout(96, 4, 512, 128, cutlass.Float32) == before
    assert _parse_mla_split_kv_table("  ") == ()
