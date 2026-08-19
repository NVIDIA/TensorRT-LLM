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
"""Fused DSv4 decode indexer + top-K (no logits materialization) vs a pure
PyTorch reference.

Covers the failure modes found while auditing the kernel against real
captures: SIGNED per-head weights (negative scores must order correctly),
the all-negative adversarial case (K-th value == 0.0 with a massive
zero-tie storm), scattered page tables, and per-row ragged context
lengths. Correctness is value-set exactness of the selected scores under
(atol=1e-2, rtol=1e-3) — the kernel emits fp16-rounded values — plus
index uniqueness/range checks.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import fused_indexer_topk_nospill
from tensorrt_llm._utils import get_sm_version

skip_not_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="requires Blackwell sm_100/sm_103",
)

PAGE = 32
FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _quant_fp4(x: torch.Tensor, gran: int = 32):
    """[M, 128] f32 -> packed [M, 64] u8 + scale words [M] i32 (4x ue8m0)."""
    m, n = x.shape
    xv = x.view(m, n // gran, gran)
    amax = xv.abs().amax(dim=2).clamp_min(1e-4)
    sf = amax / 6.0
    bits = sf.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exp = exp.clamp(1, 254)
    sf = (exp << 23).view(torch.float32)
    xs = xv / sf.unsqueeze(2)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    code = torch.bucketize(xs.abs().clamp_max(6.0), bounds).to(torch.uint8)
    code = (code | (((xs < 0) & (code != 0)).to(torch.uint8) << 3)).view(m, n)
    packed = (code[:, 0::2] & 0x0F) | ((code[:, 1::2] & 0x0F) << 4)
    sf_words = exp.to(torch.uint8).contiguous().view(torch.int32).reshape(-1)
    return packed.contiguous(), sf_words


def _dequant_fp4(packed: torch.Tensor, sf_words: torch.Tensor, gran: int = 32):
    lut = torch.tensor(FP4_LUT, device=packed.device, dtype=torch.float32)
    lo = (packed & 0x0F).to(torch.long)
    hi = ((packed >> 4) & 0x0F).to(torch.long)
    codes = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    v = lut[codes & 0x07]
    v = torch.where((codes & 0x08) != 0, -v, v)
    exp = sf_words.view(torch.int32).reshape(-1, 1).view(torch.uint8)
    sf = (exp.to(torch.int32) << 23).view(torch.float32)
    g = torch.arange(v.shape[-1], device=packed.device) // gran
    return v * sf[:, g]


def _build_inputs(batch, n_comp, k_top, weight_mode, seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    maxb = n_comp // PAGE
    nb_total = batch * maxb
    q = torch.randn((batch * 64, 128), generator=g, device=device)
    q_packed, q_sf = _quant_fp4(q)
    kv = torch.randn((nb_total * PAGE, 128), generator=g, device=device)
    kv_packed, kv_sf = _quant_fp4(kv)
    # planar production page: 2048 B data plane then 128 B scale plane
    flat = torch.empty((nb_total, PAGE * 68), device=device, dtype=torch.uint8)
    flat[:, : PAGE * 64] = kv_packed.view(nb_total, PAGE * 64)
    flat[:, PAGE * 64 :] = (
        kv_sf.reshape(nb_total, PAGE, 1)
        .view(torch.int32)
        .view(torch.uint8)
        .reshape(nb_total, PAGE * 4)
    )
    kv_cache = flat.view(nb_total, PAGE, 1, 68)
    lens = torch.randint(
        max((3 * n_comp) // 4, k_top),
        n_comp + 1,
        (batch,),
        generator=g,
        device=device,
        dtype=torch.int32,
    )
    block_table = (
        torch.randperm(nb_total, generator=g, device=device)
        .to(torch.int32)
        .view(batch, maxb)
        .contiguous()
    )
    weights = torch.randn((batch, 64), generator=g, device=device)
    if weight_mode == "nonneg":
        weights = weights.abs()
    elif weight_mode == "allneg":
        weights = -weights.abs()
    return {
        "q_fp4": q_packed.view(batch, 1, 64, 64),
        "sf_q": q_sf.view(batch, 1, 64),
        "kv_cache": kv_cache,
        "weights": weights,
        "context_lens": lens,
        "block_table": block_table,
    }


def _reference(inp, k_top):
    kvf = inp["kv_cache"].reshape(inp["kv_cache"].shape[0], -1)
    kvp = kvf[:, : PAGE * 64].reshape(-1, 64)
    kvs = kvf[:, PAGE * 64 :].contiguous().view(torch.int32).reshape(-1)
    k = _dequant_fp4(kvp, kvs).reshape(-1, PAGE, 128)
    batch = inp["q_fp4"].shape[0]
    q = _dequant_fp4(
        inp["q_fp4"][:, 0].reshape(batch * 64, 64), inp["sf_q"][:, 0].reshape(batch * 64)
    ).view(batch, 64, 128)
    vals = []
    for i in range(batch):
        length = int(inp["context_lens"][i])
        nb = (length + PAGE - 1) // PAGE
        kx = k[inp["block_table"][i, :nb].long()].reshape(nb * PAGE, 128)
        s = torch.relu(q[i] @ kx.t())
        s = (s * inp["weights"][i].unsqueeze(1)).sum(dim=0)
        s[length:] = float("-inf")
        vals.append(torch.topk(s, k_top).values)
    return torch.stack(vals)


@skip_not_sm100
@pytest.mark.parametrize("batch", [2, 16])
@pytest.mark.parametrize("n_comp", [8192, 16384])
@pytest.mark.parametrize("k_top", [512, 1024])
@pytest.mark.parametrize("weight_mode", ["signed", "nonneg", "allneg"])
def test_fused_indexer_topk_nospill(batch, n_comp, k_top, weight_mode):
    device = torch.device("cuda")
    inp = _build_inputs(batch, n_comp, k_top, weight_mode, seed=1234, device=device)
    indices = torch.full((batch, k_top), -3, dtype=torch.int32, device=device)
    values = torch.full((batch, k_top), float("nan"), dtype=torch.float32, device=device)
    fused_indexer_topk_nospill.run(
        inp["q_fp4"],
        inp["sf_q"],
        inp["kv_cache"],
        inp["weights"],
        inp["context_lens"],
        inp["block_table"],
        None,
        indices,
        values,
    )
    torch.cuda.synchronize()
    ref_vals = _reference(inp, k_top)
    for i in range(batch):
        row = indices[i]
        assert int(row.min()) >= 0
        assert int(row.max()) < int(inp["context_lens"][i])
        assert row.unique().numel() == k_top, "duplicate indices"
        got, _ = torch.sort(values[i], descending=True)
        want, _ = torch.sort(ref_vals[i], descending=True)
        dv = (got - want).abs()
        assert bool((dv <= 1e-2 + 1e-3 * want.abs()).all()), (
            f"row {i}: max value err {float(dv.max()):.4f}"
        )
