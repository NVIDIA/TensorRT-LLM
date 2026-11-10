# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct sparse decode attention (``csa2/direct_decode.py``) against an FP32 reference, and the
backend's choice between it and the staged FP8 path."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2 import direct_decode
from tensorrt_llm._torch.attention.backends.sparse.csa2.direct_decode import direct_sparse_decode
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows, unpack_rows

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="SM100/SM103/SM107 required",
)

SCALE = 512**-0.5


def reference_sparse_decode(
    q, swa_pool, swa_indices, main_pool, main_indices, sink, scale, *, main_mapping=None
) -> torch.Tensor:
    """FP32 dense reference of ``direct_sparse_decode`` (exactly decoded rows, FP32 softmax)."""
    rows = q.shape[0]
    parts, masks = [], []
    sources = [(swa_pool, swa_indices, "swa_fp8" if swa_pool.shape[1] == 512 else "swa")]
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


def _pools(swa_slots=300, main_slots=2048, seed=0, main_stride=356, swa_format="swa"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    swa = pack_rows(torch.randn(swa_slots, 512, device="cuda", generator=g) * 2, swa_format)
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


# 8 query heads per rank (TP8) run as a zero-padded 16-head MMA tile.
@pytest.mark.parametrize("main_width,heads,rows", [(512, 8, 16), (512, 64, 6), (0, 64, 2)])
# Scaled 528-byte SWA records and unit-scale 512-byte E4M3 records (``kv_cache_config.dtype="fp8"``).
@pytest.mark.parametrize("swa_format", ["swa", "swa_fp8"])
@torch.inference_mode()
def test_matches_fp32_reference(rows, heads, main_width, swa_format):
    case = _case(rows, heads, 128, main_width, mapped=main_width > 0, swa_format=swa_format)
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="SM100/SM103 small-M O-A",
)
@pytest.mark.parametrize("rows,heads", [(6, 8)])
@torch.inference_mode()
def test_output_rotated_back_gives_the_o_a_latent_bitwise(rows, heads):
    """The output rotated back in the launch gives the small-M O-A without inverse RoPE the latent
    of the O-A that rotates on load, bit for bit, eagerly and in graph replays."""
    from tensorrt_llm._torch.modules.low_m_gemm import fp8_grouped_lowm, lowm_tiles

    case = _case(rows, heads, 128, 512, mapped=True, seed=rows + heads)
    q = case[0]
    g = torch.Generator(device="cuda").manual_seed(rows * heads)
    positions = torch.randint(0, 4096, (rows,), device="cuda", generator=g, dtype=torch.int32)
    angle = torch.rand(4096, 32, device="cuda", generator=g) * 6.2831853
    cos_sin = torch.cat((angle.cos(), angle.sin()), 1).view(4096, 2, 32)
    groups, rank, depth = heads // 8, 1024, 8 * 512
    # Latent row r of a group passes one channel of head (r % 512) // 64: RoPE ones below 512.
    r = torch.arange(rank, device="cuda")
    head = (r % 512) // 64
    column = torch.where(r < 512, head * 512 + 448 + r % 64, head * 512 + (r % 64) * 7)
    weight = torch.zeros(groups * rank, depth, device="cuda")
    weight[torch.arange(groups * rank, device="cuda"), column.repeat(groups)] = 1.0
    weight = weight.to(torch.float8_e4m3fn)
    scale = torch.ones(groups * rank // 128, depth // 128, device="cuda")
    tiles = lowm_tiles(groups * rank, rows, rank)

    def project(rotated_back):
        out = _run(case, query_rope=(positions, cos_sin), output_rope=rotated_back)
        latent = fp8_grouped_lowm(
            out.view(rows, -1),
            weight,
            scale,
            rank,
            inverse_rope=None if rotated_back else (positions, cos_sin, 512),
            n_tiles=tiles,
        )
        return out, latent

    def check(rotated, plain):
        assert torch.equal(_bits(rotated[0][..., :448]), _bits(plain[0][..., :448]))
        assert torch.equal(_bits(rotated[1]), _bits(plain[1]))

    check(project(True), project(False))
    graphs, captured = [], []
    for rotated_back in (True, False):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured.append(project(rotated_back))
        graphs.append(graph)
    for trial in range(2):
        q.copy_((torch.randn_like(q, dtype=torch.float32) * (trial + 1)).to(torch.bfloat16))
        positions.random_(0, 4096, generator=g)
        for graph in graphs:
            graph.replay()
        check(*captured)


@pytest.mark.parametrize("rows", [12])
@pytest.mark.parametrize("heads", [8])
@torch.inference_mode()
def test_graph_replay_follows_new_inputs(rows: int, heads: int) -> None:
    case = _case(rows, heads, 128, 512, mapped=True)
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
        sparse_params=CSA2Params(layout=layout, compute_backend="trtllm"),
        kv_cache_dtype="fp8_ds_mla",
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


def _reference_slots(main_indices, mapping):
    """The slots CSA2's owner page-table rules map logical main selections to (-1 if invalid)."""
    table, page_size, max_positions, requests, visible = mapping
    logical = main_indices.long()
    page = logical.clamp(0, max_positions) // page_size
    request = requests.reshape(-1).long()[:, None].expand_as(logical)
    valid = (
        (logical >= 0)
        & (logical < max_positions)
        & (page < table.shape[1])
        & (request >= 0)
        & (request < table.shape[0])
        & (logical < visible.reshape(-1).long()[:, None])
    )
    physical = table.long()[request.clamp(0, table.shape[0] - 1), page.clamp(0, table.shape[1] - 1)]
    valid &= physical >= 0
    return torch.where(valid, physical * page_size + logical.clamp(min=0) % page_size, -1)


@pytest.mark.parametrize("heads,rows", [(8, 16), (64, 6)])
@torch.inference_mode()
def test_exported_slots_replace_the_mapping_bitwise(rows, heads):
    """A launch exporting its mapped main slots is unchanged, the slots follow the owner's
    page-table rules, and a launch over them equals the mapped launch."""
    case = _case(rows, heads, 128, 512, mapped=True, seed=rows + 30)
    q, swa, si, main, mi, sink, mapping = case
    slots = torch.full((rows, 512), -7, dtype=torch.int64, device="cuda")
    out = _bits(_run(case)).clone()
    assert torch.equal(_bits(_run(case, slots_out=slots)), out)
    assert torch.equal(slots, _reference_slots(mi, mapping))
    assert torch.equal(_bits(_run((q, swa, si, main, slots, sink, None))), out)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="SM100/SM103 keys-as-M attention",
)
@torch.inference_mode()
def test_exported_main_rows_replace_the_decode_bitwise():
    """An index source storing its decoded main rows is unchanged, and a Reuse launch loading them
    equals the one decoding them from the exported slots, eagerly and in graph replays."""
    rows = 6
    case = _case(rows, 8, 128, 512, mapped=True, seed=46)
    q, swa, si, main, mi, sink, mapping = case
    image = direct_decode.main_rows_bytes(8, 128, 512, q.device)
    slots = torch.empty(rows, 512, dtype=torch.int64, device="cuda")
    main_rows = torch.full((rows, image), 0xFF, dtype=torch.uint8, device="cuda")
    outputs = [torch.empty_like(q) for _ in range(4)]
    reuse = (q, swa, si, main, slots, sink, None)

    def forward():
        _run(case, output=outputs[0])
        _run(case, output=outputs[1], slots_out=slots, rows_out=main_rows)
        # main_ready: the launch before a Reuse decode starts only once the source completed.
        slots.add_(0)
        _run(reuse, output=outputs[2], main_ready=True)
        _run(reuse, output=outputs[3], main_ready=True, rows_in=main_rows)

    def check():
        torch.cuda.synchronize()
        assert torch.equal(_bits(outputs[1]), _bits(outputs[0]))
        assert torch.equal(_bits(outputs[3]), _bits(outputs[2]))
        _assert_close_to_fp32(outputs[3], _reference(case))

    forward()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for trial in range(2):
        q.copy_((torch.randn_like(q, dtype=torch.float32) * (trial + 1) * 0.5).to(torch.bfloat16))
        mi.copy_(mi.flip(1))
        graph.replay()
        check()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="SM100/SM103 keys-as-M attention",
)
@pytest.mark.parametrize("selection", ["mapped"])
@torch.inference_mode()
def test_keys_as_m_attention_matches_the_two_part_launch(monkeypatch, selection):
    """At 8 heads per rank the keys-as-M attention matches the FP32 reference as the two-part launch
    does, eagerly and in graph replays; a row without valid keys outputs zeros."""
    main_width = 0 if selection == "swa" else 512
    case = _case(6, 8, 128, main_width, mapped=selection == "mapped", seed=76)
    q, swa, si, main, mi, sink, mapping = case
    si[-1] = -1
    if mi is not None:
        mi[-1] = -1
    with monkeypatch.context() as patch:
        patch.setattr(direct_decode, "_v2_engaged", lambda *args: False)
        two_part = _run(case).clone()
    output = torch.empty_like(q)
    _run(case, output=output)
    _assert_close_to_fp32(output, _reference(case))
    _assert_close_to_fp32(output, two_part)
    assert torch.count_nonzero(output[-1]) == 0
    # A Reuse layer's launch (main rows promised before the dependent wait) gives the same bits.
    ready = torch.empty_like(q)
    if main_width:
        _run(case, output=ready, main_ready=True)
        assert torch.equal(_bits(ready), _bits(output))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(case, output=output)
        if main_width:
            _run(case, output=ready, main_ready=True)
    for trial in range(2):
        q.copy_((torch.randn_like(q, dtype=torch.float32) * (trial + 1) * 0.5).to(torch.bfloat16))
        si.copy_(si.roll(1, 0))
        if mi is not None:
            mi.copy_(mi.flip(1))
        graph.replay()
        torch.cuda.synchronize()
        _assert_close_to_fp32(output, _reference(case))
        if main_width:
            assert torch.equal(_bits(ready), _bits(output))
