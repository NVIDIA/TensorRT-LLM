# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 dsa_fp8_decode_prologue entry (fused indexer Q / K quantization,
weight scaling and index-key cache write).

The inputs are built the way the K3.1 indexer builds them: q the wq_b output [T, 32 * 128]
viewed [T, 32, 128], k a contiguous [T, 128] (the k LayerNorm output), the weights a [T, 32]
column view of a [T, 160] fused wk | weights_proj output; the cache is random bytes and the step's
keys go to distinct random slots. The op and its reference (the chain it replaces) each run on a
copy of the whole backing cache buffer, page-stride padding included, and every output and both
copies must be byte-identical. With the latent copy, compressed_kv is a contiguous [T, 512] (the
kv_a LayerNorm output) and k_pe a [T, 64] column view of the [T, 1536 + 512 + 64] qkv_a output.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_fp8_decode_prologue import (  # noqa: E501
    dsa_fp8_decode_prologue as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: The K3.1 indexer's weight scale, softmax_scale * n_heads**-0.5 (head dim 128, 32 heads).
WEIGHT_SCALE = (128**-0.5) * (32**-0.5)


def _values(shape, data: str, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(*shape, generator=gen)
    rows = x.view(-1, x.shape[-1])
    if data == "zeros_row":
        rows[::3] = 0
    elif data == "wide":
        x = x * torch.pow(10.0, torch.randint(-6, 7, shape[:-1], generator=gen).float())[..., None]
    return x.to(torch.bfloat16)


def _build(spec, seed: int):
    gen = torch.Generator().manual_seed(seed)
    n, heads, page = spec["tokens"], spec["heads"], spec["tokens_per_block"]
    rows = spec["rows"] or n
    q = _values((rows, heads * inputs.HEAD_DIM), spec["data"], gen).cuda()
    k = _values((rows, inputs.HEAD_DIM), spec["data"], gen).cuda()
    fused = _values((rows, inputs.HEAD_DIM + heads), spec["data"], gen).cuda()
    weights_bf16 = fused[:, inputs.HEAD_DIM :]
    num_pages = max(spec["num_pages"], -(-rows // page) + 1)
    base, view = inputs.raw_index_cache(num_pages, page, gen, spec["page_stride_pad"])
    fp8_slots, scale_slots = inputs.slot_mappings(rows, num_pages, page, gen)
    if spec["negative_slots"]:
        fp8_slots[1] = -1  # key slot only
        scale_slots[4] = -1  # scale slot only
        fp8_slots[9] = -1
        scale_slots[9] = -1
    compressed_kv = k_pe = None
    if spec["latent"]:
        compressed_kv = _values((rows, _KV_LORA_RANK), spec["data"], gen).cuda()
        qkv_a = _values((rows, _Q_LORA_RANK + _KV_LORA_RANK + _ROPE_DIM), spec["data"], gen).cuda()
        k_pe = qkv_a[:, _Q_LORA_RANK + _KV_LORA_RANK :]
    args = dict(
        q=q.view(rows, heads, inputs.HEAD_DIM),
        k=k,
        weights_bf16=weights_bf16,
        weight_scale=WEIGHT_SCALE,
        use_ue8m0=spec["use_ue8m0"],
        k_cache=view,
        slot_mapping_fp8=fp8_slots,
        slot_mapping_scale=scale_slots,
        num_tokens=n,
        use_pdl=True,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
    )
    return base, args


def _rebase(base: torch.Tensor, args: dict) -> tuple[torch.Tensor, dict]:
    """A copy of the backing buffer and the same args pointing into it."""
    copy = base.clone()
    page_bytes = args["k_cache"].shape[1] * args["k_cache"].shape[3]
    view = copy[:, :page_bytes].view(args["k_cache"].shape)
    return copy, dict(args, k_cache=view)


_NAMES = ("q_fp8", "q_scale", "weights", "k_fp8", "k_scale", "latent_cache")
#: The MLA dims around the indexer: q_lora_rank, kv_lora_rank, rope dim.
_Q_LORA_RANK, _KV_LORA_RANK, _ROPE_DIM = 1536, 512, 64


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    base, args = _build(cell.spec, inputs.seed_of(cell))
    got_cache, op_args = _rebase(base, args)
    want_cache, ref_args = _rebase(base, args)
    got = op(**op_args)
    want = op.reference(**ref_args)
    torch.cuda.synchronize()
    n, heads = args["num_tokens"], args["q"].shape[1]
    shapes = ((n * heads, 128), (n * heads, 1), (n, heads), (n, 128), (n, 1))
    shapes += ((n, _KV_LORA_RANK + _ROPE_DIM),) if cell.spec["latent"] else (None,)
    assert len(got) == len(want) == len(_NAMES)
    for name, g, w, shape in zip(_NAMES, got, want, shapes):
        if shape is None:
            assert g is None and w is None, name
            continue
        assert g.shape == w.shape == shape, (name, tuple(g.shape), tuple(w.shape))
        assert g.dtype == w.dtype, (name, g.dtype, w.dtype)
        assert torch.equal(_bits(g), _bits(w)), (
            f"{name}: {int((_bits(g) != _bits(w)).sum())} bytes differ"
        )
    assert torch.equal(got_cache, want_cache), (
        f"{int((got_cache != want_cache).sum())} cache bytes differ"
    )
    assert not torch.equal(got_cache, base), "nothing was written"


def test_negative_slots_leave_the_cache_alone():
    """A token with a negative key or scale slot writes neither: its other slot keeps its bytes."""
    spec = next(c.spec for c in op.CELLS if c.spec["negative_slots"])
    base, args = _build(spec, seed=5)
    got_cache, op_args = _rebase(base, args)
    op(**op_args)
    torch.cuda.synchronize()
    flat_got = got_cache[:, : args["k_cache"][0].numel()].reshape(-1)
    flat_base = base[:, : args["k_cache"][0].numel()].reshape(-1)
    fp8_slots = args["slot_mapping_fp8"].tolist()
    scale_slots = args["slot_mapping_scale"].tolist()
    skipped = [t for t in range(args["num_tokens"]) if fp8_slots[t] < 0 or scale_slots[t] < 0]
    assert skipped == [1, 4, 9], skipped
    for t in skipped:
        for slot, width in ((fp8_slots[t], 128), (scale_slots[t], 4)):
            if slot >= 0:
                assert torch.equal(flat_got[slot : slot + width], flat_base[slot : slot + width])


def test_guard_refuses_a_slot_past_the_cache():
    base, args = _build(op.CELLS[0].spec, seed=9)
    fp8 = args["slot_mapping_fp8"].clone()
    fp8[0] = args["k_cache"].numel()
    with pytest.raises(AssertionError, match="past the cache"):
        with validating(op):
            op(**dict(args, slot_mapping_fp8=fp8))


def test_guard_refuses_two_tokens_on_one_slot():
    base, args = _build(op.CELLS[0].spec, seed=10)
    fp8 = args["slot_mapping_fp8"].clone()
    fp8[1] = fp8[0]
    with pytest.raises(AssertionError, match="share a key slot"):
        with validating(op):
            op(**dict(args, slot_mapping_fp8=fp8))


def test_rejects_a_cache_with_strided_tokens():
    """A page view whose tokens are not contiguous is refused, not written wrongly."""
    base, args = _build(op.CELLS[0].spec, seed=11)
    wide = torch.zeros(64, 64, 1, 2 * 132, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        op(**dict(args, k_cache=wide[..., :132]))
