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

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_fp8_decode_prologue import (  # noqa: E501
    dsa_fp8_decode_prologue as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="tokens")[:2]


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
        k_cache=view,
        slot_mapping_fp8=fp8_slots,
        slot_mapping_scale=scale_slots,
        num_tokens=n,
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


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    base, args = _build(cell.spec, inputs.seed_of(cell))
    got_cache, op_args = _rebase(base, args)
    want_cache, ref_args = _rebase(base, args)
    got = op(**op_args)
    want = op.reference(**ref_args)
    torch.cuda.synchronize()
    assert len(got) == len(want) == len(_NAMES)
    for name, g, w in zip(_NAMES, got, want, strict=True):
        assert (g is None and w is None) or torch.equal(_bits(g), _bits(w)), name
    assert torch.equal(got_cache, want_cache), "the cache bytes differ"
