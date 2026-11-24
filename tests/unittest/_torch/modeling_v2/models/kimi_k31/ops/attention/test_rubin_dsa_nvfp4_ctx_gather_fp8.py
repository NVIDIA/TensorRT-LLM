# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rubin_dsa_nvfp4_ctx_gather_fp8 entry (a context step's NVFP4 keys into an FP8
scratch)."""

import zlib
from types import SimpleNamespace

import pytest
import torch

# l0_gb300 collects this directory: any GPU but sm_107 skips the module before a family import.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("certified on sm_107 only", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_nvfp4_ctx_gather_fp8 import (  # noqa: E402,E501
    rubin_dsa_nvfp4_ctx_gather_fp8 as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_nvfp4_ctx_gather_fp8 import (  # noqa: E402,E501
    scratch_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_nvfp4_gather_fp8 import (  # noqa: E402,E501
    CODE_BYTES,
    LATENT_DIM,
    SCALE_BYTES,
    nvfp4_rows_to_e4m3,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_dsa_nvfp4_gather_fp8 import (  # noqa: E402,E501
    rubin_dsa_nvfp4_gather_fp8 as gen_op,
)

__extra_import_path__ = ["../../../.."]
if torch.cuda.is_available():  # the GPU cells' guard interposer: a CPU run needs neither
    from _validating import certifying, validating  # noqa: E402,F401 -- path above

cpu = pytest.mark.cpu_only
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="drives the kernel on sm_107")

AMAX100 = 100.0 / (448.0 * 6.0)


def _scale_bytes(n: int, wide: bool, gen: torch.Generator) -> torch.Tensor:
    """E4M3 block-scale bytes: every finite code of both signs, or positive 2^-6 .. 2^5."""
    if wide:
        b = torch.randint(0, 254, (n,), generator=gen, dtype=torch.int64)
        return torch.where(b >= 0x7F, b + 1, b).to(torch.uint8)
    exp = torch.randint(1, 13, (n,), generator=gen)
    return (exp * 8 + torch.randint(0, 8, (n,), generator=gen)).to(torch.uint8)


def _pool(tokens: int, wide: bool, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    codes = torch.randint(0, 256, (tokens, CODE_BYTES), generator=gen, dtype=torch.uint8)
    scales = _scale_bytes(tokens * SCALE_BYTES, wide, gen).view(tokens, SCALE_BYTES)
    return codes, scales.view(torch.float8_e4m3fn)


# ---- CPU ---------------------------------------------------------------------------------------


def _chunk_args(**overrides) -> dict:
    """A later chunk of two requests (5 and 3 keys, two keys per page): row 0 and 1 of request 0,
    row 2 of request 1; one unmapped page, one position past its request, -1 slots."""
    args = dict(
        local_topk_indices=torch.tensor(
            [[0, 3, 4, -1], [1, 2, 3, 99], [0, 2, 3, -1]], dtype=torch.int32
        ),
        query_req_indices=torch.tensor([0, 0, 1], dtype=torch.int32),
        block_table=torch.tensor([[2, 0, -1], [1, 3, -1]], dtype=torch.int32),
        cu_kv_lengths=torch.tensor([0, 5, 8], dtype=torch.int64),
        global_dequant_scale=torch.tensor([0.25]),
        layer_idx=0,
        total_kv_tokens=8,
        tokens_per_block=2,
        page_stride=2,
        layer_id=0,
        residual_dim=0,
        num_pool_tokens=10,
    )
    args.update(overrides)
    return args


@cpu
def test_reference_on_a_hand_checked_chunk():
    codes, scales = _pool(10, False, torch.Generator().manual_seed(1))
    scratch, compact = op.reference(codes, scales, **_chunk_args())
    # Request 0's keys 0-3 sit on pages 2 and 0 (pool rows 4, 5, 0, 1); its key 4 on an unmapped
    # page and the 99 past its keys select nothing. Request 1's keys 0 and 2: pool rows 2 and 6.
    assert compact.tolist() == [[0, 3, -1, -1], [1, 2, 3, -1], [4, 5, -1, -1]]
    assert scratch.shape == (8, 1, LATENT_DIM) and scratch.dtype == torch.float8_e4m3fn
    rows = nvfp4_rows_to_e4m3(codes, scales, torch.tensor([0.25])).view(torch.uint8)
    got = scratch.view(torch.uint8)[:, 0]
    assert torch.equal(got[:6], rows[[4, 5, 0, 1, 2, 6]])
    assert torch.all(got[6:] == 0)


def _walk(codes, scales, args) -> tuple[torch.Tensor, list]:
    """The kernel's steps one pair at a time: mark each selected key with its pool row, number the
    marked keys in packed order, write each once, remap every pair to its key's number."""
    topk, req, table = args["local_topk_indices"], args["query_req_indices"], args["block_table"]
    cu, total = args["cu_kv_lengths"].tolist(), args["total_kv_tokens"]
    tpb, stride, layer = args["tokens_per_block"], args["page_stride"], args["layer_id"]
    marked = {}
    for r in range(topk.shape[0]):
        q = int(req[r])
        if not 0 <= q < len(cu) - 1:
            continue
        for pos in topk[r].tolist():
            if pos < 0 or pos >= cu[q + 1] - cu[q] or cu[q] + pos >= total:
                continue
            if pos // tpb >= table.shape[1]:
                continue
            page = int(table[q, pos // tpb])
            row = page * stride + pos % tpb + layer * tpb
            if page >= 0 and 0 <= row < args["num_pool_tokens"]:
                marked[cu[q] + pos] = row
    number = {key: k for k, key in enumerate(sorted(marked))}
    rows = [marked[key] for key in sorted(marked)]
    compact = []
    for r in range(topk.shape[0]):
        q = int(req[r])
        line = []
        for pos in topk[r].tolist():
            ok = 0 <= q < len(cu) - 1 and 0 <= pos < cu[q + 1] - cu[q]
            line.append(number.get(cu[q] + pos, -1) if ok else -1)
        compact.append(line)
    written = nvfp4_rows_to_e4m3(codes[rows], scales[rows], args["global_dequant_scale"])
    return written.view(torch.uint8), compact


@cpu
def test_reference_against_a_walk_of_the_kernel_steps():
    gen = torch.Generator().manual_seed(5)
    lengths, tpb, layers = [13, 0, 40, 7], 4, 3
    pages = [-(-n // tpb) for n in lengths]
    table = torch.full((len(lengths) + 1, max(pages) + 1), -1, dtype=torch.int32)
    slots = torch.randperm(sum(pages) + 4, generator=gen)
    used = 0
    for q, count in enumerate(pages):
        table[q, :count] = slots[used : used + count].to(torch.int32)
        used += count
    table[2, 3] = -1  # an unmapped page inside request 2
    tokens = (sum(pages) + 4) * layers * tpb
    codes, scales = _pool(tokens, True, gen)
    rows = 30
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
    req = torch.randint(-1, len(lengths) + 1, (rows,), generator=gen, dtype=torch.int32)
    topk = torch.randint(-1, 45, (rows, 16), generator=gen, dtype=torch.int32)
    args = dict(
        local_topk_indices=topk,
        query_req_indices=req,
        block_table=table,
        cu_kv_lengths=cu,
        global_dequant_scale=torch.tensor([AMAX100]),
        layer_idx=0,
        total_kv_tokens=int(cu[-1]),
        tokens_per_block=tpb,
        page_stride=layers * tpb,
        layer_id=2,
        residual_dim=0,
        # Below the pool: the last slot's rows fall past the bound.
        num_pool_tokens=tokens - layers * tpb,
    )
    scratch, compact = op.reference(codes, scales, **args)
    written, walked = _walk(codes, scales, args)
    assert compact.tolist() == walked
    n = written.shape[0]
    assert 0 < n < scratch.shape[0]
    assert torch.equal(scratch.view(torch.uint8)[:n, 0], written)
    assert torch.all(scratch.view(torch.uint8)[n:] == 0)


@cpu
def test_scratch_rows():
    assert scratch_rows(300, 300, 2048) == 2048  # fewer keys than the Top-K: the floor
    assert scratch_rows(6256, 256, 2048) == 6256  # the keys bound the union
    assert scratch_rows(10_000, 2, 2048) == 4096  # the pairs bound the union


def _target_args(**overrides) -> dict:
    """The chunk as a target passes it: request-local positions inside their requests."""
    topk = torch.tensor([[0, 3, 4, -1], [1, 2, 3, -1], [0, 2, -1, -1]], dtype=torch.int32)
    return dict(
        _chunk_args(local_topk_indices=topk),
        host_pool_pointers=torch.zeros(1, 2, 2, dtype=torch.int64),
        host_pool_mapping=torch.zeros(1, 2, dtype=torch.int32),
        **overrides,
    )


@cpu
def test_guard_admits_a_target_call():
    op.is_valid(**_target_args())


@cpu
@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(residual_dim=64), "residual_dim must be 0"),
        (dict(total_kv_tokens=7), "packed keys"),
        (dict(query_req_indices=torch.tensor([0, 0, 2], dtype=torch.int32)), "no context request"),
        (
            dict(local_topk_indices=torch.tensor([[0, 5, -1, -1]] * 3, dtype=torch.int32)),
            "past its request",
        ),
    ],
    ids=["residual", "total", "request", "position"],
)
def test_guard_refuses(overrides, match):
    args = _target_args()
    args.update(overrides)
    with pytest.raises(AssertionError, match=match):
        op.is_valid(**args)


# ---- GPU (sm_107) ----------------------------------------------------------------------------


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _build(spec: dict, seed: int) -> tuple[dict, dict]:
    """Call arguments on the GPU, and the pool the reference takes in place of the host tables.

    Each new key's row selects up to Top-K of the keys it sees (its request's keys up to and
    including itself), distinct and ascending, -1 after them.
    """
    gen = torch.Generator().manual_seed(seed)
    tpb, topk = spec["tokens_per_block"], spec["topk"]
    lengths = [c + n for c, n in zip(spec["cached"], spec["new"], strict=True)]
    pages = [-(-n // tpb) for n in lengths]
    slots = sum(pages) + 5
    perm = torch.randperm(slots, generator=gen)
    table = torch.full((len(lengths), max(pages)), -1, dtype=torch.int32)
    used = 0
    for q, count in enumerate(pages):
        table[q, :count] = perm[used : used + count].to(torch.int32)
        used += count
    tokens = slots * spec["page_scale"] * tpb
    pools = []
    for _ in range(spec["pools"]):
        codes, scales = _pool(tokens, spec["scales"] == "wide", gen)
        pools.append((codes.cuda(), scales.cuda()))
    pointers = torch.zeros((len(pools), 2, 2), dtype=torch.int64)
    for p, (codes, scales) in enumerate(pools):
        pointers[p, 0, 0], pointers[p, 0, 1] = codes.data_ptr(), scales.data_ptr()
    layer = len(pools) - 1
    rows = sum(spec["new"])
    picks = torch.full((rows, topk), -1, dtype=torch.int32)
    req = torch.empty(rows, dtype=torch.int32)
    r = 0
    for q, (cached, new) in enumerate(zip(spec["cached"], spec["new"], strict=True)):
        for i in range(new):
            seen = cached + i + 1
            keys = torch.arange(seen) if seen <= topk else torch.randperm(seen, generator=gen)
            keys = keys[:topk].sort().values
            picks[r, : keys.numel()] = keys.to(torch.int32)
            req[r] = q
            r += 1
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
    args = dict(
        host_pool_pointers=pointers,
        host_pool_mapping=torch.tensor([[p, 0] for p in range(len(pools))], dtype=torch.int32),
        local_topk_indices=picks.cuda(),
        query_req_indices=req.cuda(),
        block_table=table.cuda(),
        cu_kv_lengths=cu.cuda(),
        global_dequant_scale=torch.tensor([spec["scale"]], dtype=torch.float32, device="cuda"),
        layer_idx=layer,
        total_kv_tokens=int(cu[-1]),
        tokens_per_block=tpb,
        page_stride=spec["page_scale"] * tpb,
        layer_id=spec["layer_offset"],
        residual_dim=0,
        num_pool_tokens=tokens,
    )
    return args, dict(data_pool=pools[layer][0], scale_pool=pools[layer][1])


def _reference(entry, args: dict, pool: dict):
    rest = {k: v for k, v in args.items() if k not in ("host_pool_pointers", "host_pool_mapping")}
    return entry.reference(**pool, **rest)


def _assert_same(scratch, compact, ref_scratch, ref_compact) -> int:
    """Compact indices equal; scratch rows equal up to the selected count. Returns that count."""
    assert scratch.shape == ref_scratch.shape and scratch.dtype == torch.float8_e4m3fn
    assert torch.equal(compact, ref_compact)
    n = int(ref_compact.max()) + 1
    assert torch.equal(scratch.view(torch.uint8)[:n], ref_scratch.view(torch.uint8)[:n])
    return n


@gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args, pool = _build(cell.spec, _seed(cell.why))
    scratch, compact = op(**args)
    ref_scratch, ref_compact = _reference(op, args, pool)
    torch.cuda.synchronize()
    _assert_same(scratch, compact, ref_scratch, ref_compact)


@gpu
def test_the_catalog_nvfp4_write_then_both_gathers_on_the_family_cache_manager():
    """The family cache manager in the NVFP4 + FP8-tail layout (global MLA layers 1, 3 NVFP4; 5, 7
    FP8): the catalog context write in the NVFP4 KV quant mode fills layers 1 and 3, the context
    gather reads them through the manager's own host tables and page index parameters, bit for bit
    against the reference over the manager's pool memory and close to the written rows; the
    generation gather refuses an FP8 layer."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.mla_rope_append_paged_kv_assign_q import (  # noqa: E501
        mla_rope_append_paged_kv_assign_q as write,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (  # noqa: E501
        RubinKdaDsaHybridCacheManager,
    )
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekSparseAttentionConfig,
        KvCacheConfig,
        MambaStateConfig,
    )
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    tpb, mla_layers, total, topk = 32, 4, 100, 64
    text = SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=2 * mla_layers,
        kv_lora_rank=32,
        qk_rope_head_dim=32,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": list(range(1, 2 * mla_layers, 2)),
            "full_attn_layers": list(range(2, 2 * mla_layers + 1, 2)),
        },
        dsa_config={"dsa_layers": list(range(2, 2 * mla_layers + 1, 2)), "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )
    manager = RubinKdaDsaHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=4,
        mamba_head_dim=8,
        mamba_num_layers=mla_layers,
        mamba_layer_mask=[True, False] * mla_layers,
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(
            max_tokens=512,
            enable_block_reuse=False,
            enable_partial_reuse=True,
            mamba_state_config=MambaStateConfig(
                periodic_snapshot_interval=0, additional_snapshot_offsets_from_end=[0]
            ),
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=mla_layers,
        num_kv_heads=1,
        head_dim=LATENT_DIM,
        tokens_per_block=tpb,
        max_seq_len=256,
        max_batch_size=2,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.NVFP4,
        layer_mask=[False, True] * mla_layers,
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=4, index_head_dim=128, index_topk=topk, indexer_k_dtype="fp8"
        ),
        pretrained_config=text,
    )
    try:
        assert manager.mla_kv_cache_residual_dim == 0
        assert [manager.get_kv_cache_dtype(layer) for layer in (1, 3, 5, 7)] == [
            DataType.NVFP4,
            DataType.NVFP4,
            DataType.FP8,
            DataType.FP8,
        ]
        manager.add_dummy_requests([42], token_nums=[total], is_gen=False)
        offsets = torch.empty(
            (manager.num_attention_op_pools, 1, 2, manager.max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        manager.copy_batch_block_offsets(offsets, [42], 1, 1, 1)
        slots = manager.get_pool_block_indices(1, request_ids=[42], num_contexts=1)[0].cuda()
        table = slots[None].to(torch.int32).contiguous()
        rope = torch.zeros((256, 128), dtype=torch.float32, device="cuda")
        rope[:, ::2] = 1  # identity rotations: the latent rows are stored as given
        g = torch.tensor([AMAX100], dtype=torch.float32, device="cuda")
        quant_mode = int(QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4).layer_quant_mode)
        cu = torch.tensor([0, total], dtype=torch.int64, device="cuda")
        gen = torch.Generator(device="cuda").manual_seed(3)
        rows = 40
        picks = torch.stack(
            [torch.randperm(total, generator=gen, device="cuda")[:topk] for _ in range(rows)]
        )
        picks = picks.sort(dim=1).values.to(torch.int32).contiguous()
        req = torch.zeros(rows, dtype=torch.int32, device="cuda")
        for layer in (1, 3):
            local = manager.layer_offsets[layer]
            latent = torch.randn((total, LATENT_DIM), generator=gen, device="cuda")
            latent = latent.to(torch.bfloat16)
            q = torch.zeros((total, 8 * 192), dtype=torch.bfloat16, device="cuda")
            with validating(write):
                write(
                    q=q,
                    latent_cache=latent,
                    num_contexts=1,
                    cu_ctx_cached_kv_lens=cu.new_zeros(2),
                    cu_seq_lens=cu,
                    max_input_uncached_seq_len=total,
                    cos_sin_cache=rope,
                    head_num=8,
                    nope_size=128,
                    rope_size=64,
                    lora_size=512,
                    kv_cache_block_offsets=offsets,
                    host_kv_cache_pool_pointers=manager.kv_cache_pool_pointers,
                    host_kv_cache_pool_mapping=manager.kv_cache_pool_mapping,
                    kv_scale_orig_quant=g.reciprocal(),
                    residual_dim=0,
                    layer_idx=local,
                    tokens_per_block=tpb,
                    attention_window_size=manager.max_seq_len,
                    beam_width=1,
                    quant_mode=quant_mode,
                )
            primary = manager.get_primary_pool(local)
            tokens = primary.shape[0] * primary.shape[1] * tpb
            pool_id = int(manager.kv_cache_pool_mapping[local, 0])
            scale_base = int(manager.kv_cache_pool_pointers[pool_id, 0, 1])
            pool = dict(
                data_pool=primary.view(torch.uint8).reshape(tokens, CODE_BYTES),
                scale_pool=convert_to_torch_tensor(
                    TensorWrapper(scale_base, torch.uint8, [tokens, SCALE_BYTES])
                ).view(torch.float8_e4m3fn),
            )
            page_scale, layer_offset = manager.get_primary_pool_page_index_params(local)
            tables = dict(
                host_pool_pointers=manager.kv_cache_pool_pointers,
                host_pool_mapping=manager.kv_cache_pool_mapping,
            )
            ctx_args = dict(
                tables,
                local_topk_indices=picks,
                query_req_indices=req,
                block_table=table,
                cu_kv_lengths=cu,
                global_dequant_scale=g,
                layer_idx=local,
                total_kv_tokens=total,
                tokens_per_block=tpb,
                page_stride=page_scale * tpb,
                layer_id=layer_offset,
                residual_dim=0,
                num_pool_tokens=tokens,
            )
            with validating(op):
                scratch, compact = op(**ctx_args)
            n = _assert_same(scratch, compact, *_reference(op, ctx_args, pool))
            keys = torch.unique(picks)
            assert n == keys.numel()
            _assert_close_to(scratch[:n, 0], latent[keys.long()])

        fp8_layer = manager.layer_offsets[5]
        index = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
        with pytest.raises(RuntimeError, match="non-null"):
            gen_op(
                manager.kv_cache_pool_pointers,
                manager.kv_cache_pool_mapping,
                index,
                torch.empty((1, 4, LATENT_DIM), dtype=torch.float8_e4m3fn, device="cuda"),
                torch.empty_like(index),
                g,
                fp8_layer,
                0,
                1,
            )
    finally:
        manager.shutdown()


def _assert_close_to(e4m3_rows: torch.Tensor, latent: torch.Tensor) -> None:
    """The gathered rows (the gathers fold the global scale in) are the written rows to NVFP4's
    precision.

    E2M1 codes under a block scale of amax / 6 leave a Gaussian row a relative RMS error near
    0.1 (half-steps of 0.25 to 1 block scale), and E4M3 on the way out adds about 0.04; the gate
    is 0.25, and the rows must not be zero.
    """
    got = e4m3_rows.float()
    want = latent.float()
    assert got.abs().max() > 0
    rel = ((got - want).pow(2).mean() / want.pow(2).mean()).sqrt().item()
    assert rel < 0.25, f"relative RMS {rel:.3f}"
