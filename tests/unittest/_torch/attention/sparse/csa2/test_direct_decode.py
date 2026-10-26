# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct sparse decode attention (``csa2/direct_decode.py``) against an FP32 reference, and the
backend's choice between it and the staged FP8 path."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.direct_decode import direct_sparse_decode
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows, unpack_rows

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="SM100/SM103 required",
)

SCALE = 512**-0.5


def reference_sparse_decode(
    q, swa_pool, swa_indices, main_pool, main_indices, sink, scale, *, main_mapping=None
) -> torch.Tensor:
    """FP32 dense reference of ``direct_sparse_decode`` (exactly decoded rows, FP32 softmax)."""
    rows = q.shape[0]
    parts, masks = [], []
    sources = [(swa_pool, swa_indices, "swa")]
    if main_indices is not None and main_indices.shape[1] > 0:
        slots = main_indices.long()
        if main_mapping is not None:
            table, page_size, max_positions, requests, visible = main_mapping
            requests = requests.reshape(-1).long()
            visible = visible.reshape(-1).long()
            logical = slots
            page = logical.clamp(0, max_positions) // page_size
            request = requests[:, None].expand_as(logical)
            valid = (
                (logical >= 0)
                & (logical < max_positions)
                & (page < table.shape[1])
                & (request >= 0)
                & (request < table.shape[0])
                & (logical < visible[:, None])
            )
            physical = table.long()[
                request.clamp(0, table.shape[0] - 1), page.clamp(0, table.shape[1] - 1)
            ]
            valid &= physical >= 0
            slots = torch.where(valid, physical * page_size + logical.clamp(min=0) % page_size, -1)
        sources.append((main_pool, slots, "main"))
    for pool, indices, fmt in sources:
        indices = indices.long()
        valid = (indices >= 0) & (indices < pool.shape[0])
        values = unpack_rows(pool[indices.clamp(0, pool.shape[0] - 1)], 512, fmt)
        parts.append(torch.where(valid[..., None], values, 0))
        masks.append(valid)
    kv = torch.cat(parts, 1).float()
    logits = torch.einsum("qhd,qkd->qhk", q.float(), kv) * scale
    logits.masked_fill_(~torch.cat(masks, 1)[:, None, :], float("-inf"))
    probabilities = torch.softmax(
        torch.cat((logits, sink.float()[None, :, None].expand(rows, -1, -1)), -1), -1
    )[..., :-1]
    return torch.einsum("qhk,qkd->qhd", probabilities, kv).bfloat16()


def _pools(swa_slots=300, main_slots=2048, seed=0, main_stride=356):
    g = torch.Generator(device="cuda").manual_seed(seed)
    swa = pack_rows(torch.randn(swa_slots, 512, device="cuda", generator=g) * 2, "swa")
    # The main cache is a strided view of its page group (main and index records interleaved).
    owner = torch.empty(main_slots, main_stride, device="cuda", dtype=torch.uint8)
    main = owner[:, :288]
    main.copy_(pack_rows(torch.randn(main_slots, 512, device="cuda", generator=g) * 2, "main"))
    return swa, main


def _case(rows, heads, swa_width, main_width, *, mapped, dtype=torch.int64, seed=0, **pools):
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    swa, main = _pools(seed=seed, **pools)
    q = (torch.randn(rows, heads, 512, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    sink = torch.randn(heads, device="cuda", generator=g)
    si = torch.randint(0, swa.shape[0], (rows, swa_width), device="cuda", generator=g).to(dtype)
    if swa_width:
        si[0, ::7] = -1
        si[-1, -1] = swa.shape[0]
    mi, mapping = None, None
    if main_width:
        if mapped:
            page = 64
            pages = main.shape[0] // page
            table = torch.full((3, pages + 2), -1, device="cuda", dtype=torch.int32)
            table[1, :pages] = torch.randperm(pages, device="cuda", generator=g)
            table[1, 3] = -1
            positions = pages * page
            requests = torch.ones(rows, device="cuda", dtype=torch.int64)
            visible = torch.full((rows,), positions - 100, device="cuda", dtype=torch.int64)
            mi = torch.randint(0, positions, (rows, main_width), device="cuda", generator=g)
            mi = mi.to(dtype)
            mi[:, 0] = 3 * page + 5  # unbacked page
            mi[:, 1] = positions - 50  # beyond the visible length
            mi[:, 2] = -1
            mi[0, 3] = positions + 7  # beyond max_positions
            mapping = (table, page, positions, requests, visible)
        else:
            mi = torch.randint(0, main.shape[0], (rows, main_width), device="cuda", generator=g)
            mi = mi.to(dtype)
            mi[:, ::11] = -1
            mi[-1, 1] = main.shape[0]
    return q, swa, si, main, mi, sink, mapping


def _run(case, **kwargs):
    q, swa, si, main, mi, sink, mapping = case
    return direct_sparse_decode(q, swa, si, main, mi, sink, SCALE, main_mapping=mapping, **kwargs)


def _reference(case):
    q, swa, si, main, mi, sink, mapping = case
    return reference_sparse_decode(q, swa, si, main, mi, sink, SCALE, main_mapping=mapping)


def _assert_close_to_fp32(actual, expected):
    """BF16 probabilities in the second product: ~2^-9 relative per term."""
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=4e-3 * expected.float().abs().max().item(), rtol=2e-2
    )


@pytest.mark.parametrize("rows", [2, 16])
# 8 query heads per rank (TP8) run as a zero-padded 16-head MMA tile.
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("main_width", [0, 512])
@torch.inference_mode()
def test_matches_fp32_reference(rows, heads, main_width):
    case = _case(rows, heads, 128, main_width, mapped=main_width > 0)
    si, mi = case[2], case[4]
    # A row without valid keys attends only the sink: zeros.
    si[-1] = -1
    if mi is not None:
        mi[-1] = -1
    out = _run(case)
    _assert_close_to_fp32(out, _reference(case))
    assert torch.count_nonzero(out[-1]) == 0
    # Work moved before the dependent wait reads the same inputs.
    ready = _run(case, inputs_ready=True, main_ready=main_width > 0)
    assert torch.equal(_bits(ready), _bits(out))
    # The query RoPE on load equals mla_rope_inplace before the launch, bit for bit.
    g = torch.Generator(device="cuda").manual_seed(7)
    positions = torch.randint(0, 4096, (rows,), device="cuda", generator=g, dtype=torch.int32)
    angle = torch.rand(4096, 32, device="cuda", generator=g) * 6.2831853
    cos_sin = torch.cat((angle.cos(), angle.sin()), 1).view(4096, 2, 32)
    roped = case[0].clone()
    torch.ops.trtllm.mla_rope_inplace(roped, positions, cos_sin, heads, 448, 64, False, False)
    expected = _run((roped, *case[1:]))
    assert torch.equal(_bits(_run(case, query_rope=(positions, cos_sin))), _bits(expected))


@torch.inference_mode()
def test_graph_replay_follows_new_inputs():
    case = _case(6, 16, 128, 512, mapped=True)
    q, swa, si, main, mi, sink, mapping = case
    output = torch.empty_like(q)
    _run(case, output=output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(case, output=output)
    q.copy_(torch.randn_like(q))
    si[2, :40] = -1
    mi[3] = mi[3].flip(0)
    mapping[4][0] = 600
    graph.replay()
    torch.cuda.synchronize()
    _assert_close_to_fp32(output, _reference(case))


# ---------------------------------------------------------------------------- backend


def _backend(heads, layout=None, layer_idx=20):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Params
    from tensorrt_llm._torch.attention.backends.utils import create_attention

    return create_attention(
        "TRTLLM",
        layer_idx,
        heads,
        512,
        num_kv_heads=1,
        is_mla_enable=True,
        q_lora_rank=1280,
        kv_lora_rank=448,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        v_head_dim=512,
        rope_append=False,
        sparse_params=CSA2Params(layout=layout),
        kv_cache_dtype="fp8",
    )


def _generation_args(case, stage_group=None):
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs

    q, swa, si, main, mi, sink, mapping = case
    return AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        attention_sinks=sink,
        output=torch.empty(q.shape[0], q.shape[1] * 512, device="cuda", dtype=torch.bfloat16),
        sparse_backend_args=CSA2BackendForwardArgs(
            swa_pool=swa,
            swa_indices=si,
            main_pool=main,
            topk_indices=mi,
            main_mapping=mapping,
            stage_group=stage_group,
        ),
    )


class _Forwards:
    """Two layers of one staging group over persistent query tiles, one forward at a time."""

    def __init__(self):
        self.layers = [_backend(16, layer_idx=21), _backend(16, layer_idx=22)]
        self.tiles = {}

    def run(self, case, group):
        from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

        q = case[0]
        tile = self.tiles.get(q.shape[0])
        if tile is None:
            tile = self.tiles[q.shape[0]] = CSA2TrtllmMetadata.for_query_tile(
                q, case[4].shape[1], staging_dtype=torch.float8_e4m3fn
            )
        tile._csa2_stage_epoch = getattr(tile, "_csa2_stage_epoch", 0) + 1
        outputs = []
        for attn in self.layers:
            args = _generation_args(case, group)
            attn.forward(q.flatten(1), None, None, tile, forward_args=args)
            outputs.append(args.output.view_as(q).clone())
        return outputs


@torch.inference_mode()
def test_alternating_direct_and_staged_forwards_on_one_group():
    """Bypassed (<= 16 rows) forwards between staged (> 16 rows) ones on the same layers and
    staging group: the staged forwards equal a staging that keeps nothing, while the caches
    change from forward to forward."""
    if not _backend(16).native_fp8:
        pytest.skip("native FP8 staging unavailable")
    group = ("kv", "index", 0)
    small = _case(6, 16, 128, 512, mapped=True, seed=3, main_stride=288)
    large = _case(24, 16, 128, 512, mapped=True, seed=4, main_stride=288)
    forwards = _Forwards()
    for step in range(3):
        staged = forwards.run(large, group)
        fresh = _Forwards().run(large, None)
        for out, expected in zip(staged, fresh):
            assert torch.equal(out, expected), f"step {step}"
        for out in forwards.run(small, group):
            _assert_close_to_fp32(out, _reference(small))
        # The next forward sees other main rows and another SWA row exponent.
        for case in (small, large):
            main = case[3]
            main[:64].copy_(pack_rows(torch.randn(64, 512, device="cuda") * (step + 2), "main"))
            case[1][:, 512:] += 1


def _bits(t):
    return t.view(torch.int16)
