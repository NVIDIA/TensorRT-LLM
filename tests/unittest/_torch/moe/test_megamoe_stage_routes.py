# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``megamoe_stage_routes`` / ``megamoe_stage_inputs`` write a MegaMoE bucket's staging byte for
byte as torch copies, fills and ``fp4_quantize.into`` do: routes in rows ``[0, n)`` (only the live
rows of a live-row word), -1 and 0 after them."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe import live_rows
from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_cute_dsl import MegaMoECuteDsl

ROWS, TOP_K = 16, 6

_DEVICES = [
    pytest.param("cpu", marks=pytest.mark.cpu_only),
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    ),
]


def _garbage(shape, dtype, generator, device):
    """A buffer of random bytes, as a bucket holds after earlier launches."""
    nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    raw = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=generator)
    return raw.view(dtype).view(shape).to(device)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "id_dtype,out_dtype",
    [(torch.int32, torch.int64), (torch.int32, torch.int32), (torch.int64, torch.int64)],
)
@pytest.mark.parametrize("rows", [0, 1, 7, ROWS])
@pytest.mark.parametrize("hidden", [None, 7168, 2080])
def test_stage_routes_matches_torch_copies_and_fills(device, id_dtype, out_dtype, rows, hidden):
    if device == "cpu" and hidden is not None:
        pytest.skip("fused quantization requires CUDA")
    gen = torch.Generator().manual_seed(rows)
    info = torch.iinfo(torch.int32)
    ids = torch.randint(-1, 384, (rows, TOP_K), dtype=id_dtype, generator=gen)
    if rows:
        ids[0, :2] = torch.tensor([info.max, info.min])  # widening keeps the sign
    ids = ids.to(device)
    weights = torch.rand(rows, TOP_K, generator=gen).to(device)
    ids_out = _garbage((ROWS, TOP_K), out_dtype, gen, device)
    weights_out = _garbage((ROWS, TOP_K), torch.float32, gen, device)
    want_ids, want_weights = ids_out.clone(), weights_out.clone()
    want_ids[:rows].copy_(ids)
    want_weights[:rows].copy_(weights)
    want_ids[rows:].fill_(-1)
    want_weights[rows:].zero_()

    if hidden is None:
        torch.ops.trtllm.megamoe_stage_routes(ids, weights, ids_out, weights_out)
    else:
        x = (torch.randn(rows, hidden, generator=gen) * 3).to(torch.bfloat16).cuda()
        x[::3, 16:32] = 0
        scale = torch.tensor([448 * 6 / 7.5], device="cuda")
        fp4 = _garbage((x.shape[0], hidden // 2), torch.uint8, gen, device)
        sf = _garbage((x.shape[0], hidden // 16), torch.uint8, gen, device)
        want_fp4, want_sf = fp4.clone(), sf.clone()
        torch.ops.trtllm.fp4_quantize.into(x, scale, want_fp4, want_sf, False)
        torch.ops.trtllm.megamoe_stage_inputs(x, scale, fp4, sf, ids, weights, ids_out, weights_out)
        assert torch.equal(fp4, want_fp4) and torch.equal(sf, want_sf)

    assert torch.equal(ids_out.view(torch.uint8), want_ids.view(torch.uint8))
    assert torch.equal(weights_out.view(torch.uint8), want_weights.view(torch.uint8))


@pytest.mark.parametrize("device", _DEVICES)
def test_stage_routes_stages_only_the_live_rows_of_the_word(device):
    """The op keeps the host reference's live rows (``live_rows.live_row_count``)."""
    top = live_rows.MAX_REQUESTS
    words = [
        live_rows.NO_ROWS,
        live_rows.ALL_ROWS,
        live_rows.encode(0, 4),
        live_rows.encode(3, 4),  # 4 requests of 3 rows: 9 of 12 rows live
        live_rows.encode(4, 4),
        (4 << live_rows.REQUEST_BITS) | 5,  # more real requests than requests: every row
        live_rows.encode(1, 5),  # 5 requests do not divide 12 rows: every row
        live_rows.encode(1, 32768),  # the sign bit of the word set
        live_rows.encode(top - 1, top),
    ]
    gen = torch.Generator().manual_seed(0)
    for n in (0, 1, 12, ROWS):
        ids = torch.randint(0, 384, (n, TOP_K), dtype=torch.int32, generator=gen).to(device)
        weights = torch.rand(n, TOP_K, generator=gen).to(device)
        for value in words:
            word = torch.tensor([value], dtype=torch.int64).to(torch.int32).to(device)
            ids_out = _garbage((ROWS, TOP_K), torch.int64, gen, device)
            weights_out = _garbage((ROWS, TOP_K), torch.float32, gen, device)
            live = live_rows.live_row_count(value, n)
            want_ids = torch.full((ROWS, TOP_K), -1, dtype=torch.int64, device=device)
            want_weights = torch.zeros(ROWS, TOP_K, device=device)
            want_ids[:live].copy_(ids[:live])
            want_weights[:live].copy_(weights[:live])

            torch.ops.trtllm.megamoe_stage_routes(ids, weights, ids_out, weights_out, word)

            assert torch.equal(ids_out, want_ids), (n, hex(value))
            assert torch.equal(weights_out.view(torch.uint8), want_weights.view(torch.uint8))


@pytest.mark.parametrize("device", _DEVICES)
def test_stage_routes_in_two_parts_writes_what_one_launch_writes(device):
    """A replica plan stages weights and tails before its route wait, live ids after it: together
    the bytes of one launch."""
    gen = torch.Generator().manual_seed(1)
    n = 12
    ids = torch.randint(0, 384, (n, TOP_K), dtype=torch.int32, generator=gen).to(device)
    weights = torch.rand(n, TOP_K, generator=gen).to(device)
    for value in (live_rows.ALL_ROWS, live_rows.encode(3, 4)):
        word = torch.tensor([value], dtype=torch.int64).to(torch.int32).to(device)
        whole_ids = _garbage((ROWS, TOP_K), torch.int64, gen, device)
        whole_weights = _garbage((ROWS, TOP_K), torch.float32, gen, device)
        torch.ops.trtllm.megamoe_stage_routes(ids, weights, whole_ids, whole_weights, word)

        ids_out = _garbage((ROWS, TOP_K), torch.int64, gen, device)
        weights_out = _garbage((ROWS, TOP_K), torch.float32, gen, device)
        before_ids = ids_out.clone()
        torch.ops.trtllm.megamoe_stage_routes(
            ids, weights, ids_out, weights_out, word, live_ids=False
        )
        live = live_rows.live_row_count(value, n)
        assert torch.equal(ids_out[:live], before_ids[:live])  # the ids wait for the routes
        assert torch.equal(ids_out[live:], whole_ids[live:])
        assert torch.equal(weights_out.view(torch.uint8), whole_weights.view(torch.uint8))
        after_rest = weights_out.clone()
        torch.ops.trtllm.megamoe_stage_routes(ids, weights, ids_out, weights_out, word, rest=False)
        assert torch.equal(ids_out, whole_ids), hex(value)
        assert torch.equal(weights_out.view(torch.uint8), after_rest.view(torch.uint8))


@pytest.mark.cpu_only
def test_stage_routes_refuses_a_word_or_weights_it_would_misread():
    ids, weights = torch.zeros(2, TOP_K, dtype=torch.int64), torch.zeros(2, TOP_K)
    ids_out, weights_out = torch.empty(ROWS, TOP_K, dtype=torch.int64), torch.empty(ROWS, TOP_K)
    for word in (torch.zeros(1, dtype=torch.int64), torch.zeros(2, dtype=torch.int32)):
        with pytest.raises(RuntimeError, match="live_row_word"):
            torch.ops.trtllm.megamoe_stage_routes(ids, weights, ids_out, weights_out, word)
    for strided in (
        torch.empty(ROWS, TOP_K + 2)[:, :TOP_K],
        torch.empty(1, TOP_K).expand(ROWS, -1),
    ):
        with pytest.raises(RuntimeError, match="contiguous"):
            torch.ops.trtllm.megamoe_stage_routes(ids, weights, ids_out, strided)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("rows,bucket", [(0, 16), (5, 16), (16, 16), (37, 64)])
def test_bf16_inputs_stage_in_one_launch_as_in_two(rows, bucket, monkeypatch):
    """The fused prepare (``_stage_bf16_inputs``), replayed from a CUDA graph over new inputs, stages
    the bucket byte for byte as ``fp4_quantize.into`` + ``megamoe_stage_routes``."""
    gen = torch.Generator().manual_seed(rows)
    shapes = dict(activation=(3584, torch.uint8), activation_sf=(448, torch.uint8))
    shapes.update(topk_idx_local=(TOP_K, torch.int64), topk_weights=(TOP_K, torch.float32))
    fused = SimpleNamespace(
        **{k: _garbage((bucket, n), t, gen, "cuda") for k, (n, t) in shapes.items()}
    )
    separate = SimpleNamespace(**{k: v.clone() for k, v in vars(fused).items()})
    scale = torch.tensor([448 * 6 / 7.5], device="cuda")
    moe = SimpleNamespace(
        fc31_input_scale=scale,
        _staging_of_next_run=lambda n, device: fused,
        _padding_drop_buckets={bucket},
        _shared_expert=None,
        _slot_of_expert=None,
        _prestaged=None,
    )
    moe._activation_to_quantize = MegaMoECuteDsl._activation_to_quantize.__get__(moe)
    moe._shared_activation_target = MegaMoECuteDsl._shared_activation_target.__get__(moe)
    x = torch.zeros(rows, 7168, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(rows, TOP_K, dtype=torch.int32, device="cuda")
    weights = torch.zeros(rows, TOP_K, device="cuda")
    word = torch.tensor([live_rows.encode(0, 1)], dtype=torch.int32, device="cuda")
    monkeypatch.setattr(live_rows, "current_word", lambda device: word)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        MegaMoECuteDsl._stage_bf16_inputs(moe, x, ids, weights, None)
    for value in (live_rows.NO_ROWS, live_rows.ALL_ROWS):
        word.fill_(value)
        x.copy_(torch.randn(rows, 7168, generator=gen) * 3)
        ids.copy_(torch.randint(-1, 384, (rows, TOP_K), generator=gen))
        weights.copy_(torch.rand(rows, TOP_K, generator=gen))
        graph.replay()
        if rows:
            torch.ops.trtllm.fp4_quantize.into(
                x, scale, separate.activation[:rows], separate.activation_sf[:rows], False
            )
        torch.ops.trtllm.megamoe_stage_routes(
            ids, weights, separate.topk_idx_local, separate.topk_weights[:, :TOP_K], word
        )
        for name, want in vars(separate).items():
            assert torch.equal(getattr(fused, name).view(torch.uint8), want.view(torch.uint8)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_live_row_word_writes_are_stream_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cuda.bindings import driver

    device = torch.device("cuda", torch.cuda.current_device())
    word = live_rows.LiveRowWord()
    word.write(live_rows.encode(3, 8))  # before the word exists: it is allocated holding this
    tensor = word.tensor(device)
    seen = [tensor.clone()]
    expected = [live_rows.encode(3, 8)]
    for value in (live_rows.encode(5, 8), live_rows.ALL_ROWS, live_rows.NO_ROWS, live_rows.NO_ROWS):
        # Work enqueued before a write reads the old value even while the host runs ahead.
        torch.cuda._sleep(2_000_000)
        seen.append(tensor.clone())
        expected.append(expected[-1])
        word.write(value)
        seen.append(tensor.clone())
        expected.append(value)
    torch.cuda.synchronize()
    assert [int(t.item()) for t in seen] == expected

    # Without stream memory operations the word keeps every row.
    monkeypatch.setattr(
        driver, "cuStreamWriteValue32", lambda *_: (driver.CUresult.CUDA_ERROR_NOT_SUPPORTED,)
    )
    unsupported = live_rows.LiveRowWord()
    for value in (live_rows.encode(3, 8), live_rows.NO_ROWS):
        unsupported.write(value)
        assert int(unsupported.tensor(device).item()) == live_rows.ALL_ROWS


def _stage_inputs_grid(launch) -> int:
    """The grid of the one megamoe_stage_inputs kernel ``launch`` runs, from the profiler's trace."""
    import json
    import tempfile

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        launch()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as trace:
        prof.export_chrome_trace(trace.name)
        events = json.load(open(trace.name))["traceEvents"]
    grids = [
        e["args"]["grid"][0]
        for e in events
        if e.get("cat") == "kernel" and "megaMoeStageInputsKernel" in e.get("name", "")
    ]
    assert len(grids) == 1, grids
    return grids[0]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="needs a Blackwell CUDA device",
)
@pytest.mark.parametrize("bucket", [4096, 16384, 32768])
def test_stage_inputs_one_launch_against_two_large_buckets(bucket):
    """Past one grid wave megamoe_stage_inputs still writes the bytes of the two-launch form, and its
    quantization grid does not shrink as the bucket grows."""
    hidden, top_k = 7168, 8
    gen = torch.Generator(device="cuda").manual_seed(bucket)
    x = (torch.randn(bucket, hidden, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    scale = torch.tensor([448 * 6 / 7.5], device="cuda")
    ids = torch.randint(0, 384, (bucket, top_k), dtype=torch.int32, device="cuda", generator=gen)
    weights = torch.rand(bucket, top_k, device="cuda", generator=gen)

    def buffers():
        return (
            torch.empty(bucket, hidden // 2, dtype=torch.uint8, device="cuda"),
            torch.empty(bucket, hidden // 16, dtype=torch.uint8, device="cuda"),
            torch.empty(bucket, top_k, dtype=torch.int64, device="cuda"),
            torch.empty(bucket, top_k, device="cuda"),
        )

    one, two = buffers(), buffers()

    def one_launch():
        torch.ops.trtllm.megamoe_stage_inputs(
            x, scale, one[0], one[1], ids, weights, one[2], one[3]
        )

    def two_launches():
        torch.ops.trtllm.fp4_quantize.into(x, scale, two[0], two[1], False)
        torch.ops.trtllm.megamoe_stage_routes(ids, weights, two[2], two[3])

    one_launch()
    two_launches()
    for a, b in zip(one, two):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))

    # The route writer takes one 256-thread block per 256 route elements, at most 1024 blocks; every
    # bucket here has more quantization items (rows x 2) than blocks fit on the device at once.
    def writer_blocks(rows: int) -> int:
        return min(-(-rows * top_k // 256), 1024)

    small = 4096
    reference = buffers()
    quantize_blocks = _stage_inputs_grid(one_launch) - writer_blocks(bucket)
    reference_blocks = _stage_inputs_grid(
        lambda: torch.ops.trtllm.megamoe_stage_inputs(
            x[:small],
            scale,
            reference[0][:small],
            reference[1][:small],
            ids[:small],
            weights[:small],
            reference[2][:small],
            reference[3][:small],
        )
    ) - writer_blocks(small)
    assert (
        quantize_blocks
        == reference_blocks
        >= torch.cuda.get_device_properties(0).multi_processor_count
    )


# ---------------------------------------------------------------------------------------------------------------------
# The adaptive route of the one-round dispatch router (adaptive_eplb_router_reference.py): the router's own device code
# (_OneRoundRouter._adaptive_counted_rows / _adaptive_pick), launched one warp per token row as the router launches it,
# against the module's plain-integer definition, bit for bit; and the kernels a build without the route compiles.
# ---------------------------------------------------------------------------------------------------------------------

_requires_cute_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="needs a Blackwell GPU",
)
_AR_WORLD, _AR_SLOTS, _AR_EXPERTS, _AR_TOP_K = 32, 512, 384, 6


def _router_route_module():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent / "adaptive_eplb_router_reference.py"
    spec = importlib.util.spec_from_file_location("router_route", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _route_table(rng, zero_one):
    """A packed route table row of 384 experts in 512 slots of 32 GPUs: every expert at home, the 128
    spare slots holding replicas, a share vector with some 0/1 copies (policy.build_route_tables)."""
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
        ExpertGeometry,
        build_route_tables,
    )

    geometry = ExpertGeometry(_AR_EXPERTS, _AR_SLOTS, _AR_WORLD)
    layout = geometry.home_layout().tolist()
    free = [s for s, e in enumerate(layout) if e < 0]
    rng.shuffle(free)
    hot = rng.sample(range(_AR_EXPERTS), 40)
    for slot in free[:96]:
        layout[slot] = rng.choice(hot)
    shares = [rng.random() for _ in range(_AR_SLOTS)]
    by_expert = {}
    for slot, e in enumerate(layout):
        if e >= 0:
            by_expert.setdefault(e, []).append(slot)
    for slots in by_expert.values():
        if len(slots) > 1 and rng.random() < zero_one:
            keep = rng.randrange(len(slots))
            for k, slot in enumerate(slots):
                shares[slot] = 1.0 if k == keep else 0.0
    return build_route_tables(layout, shares, _AR_EXPERTS, _AR_SLOTS).packed()


def _compile_router_probe(rows, staged=False):
    """``staged``: the probe's CTA first stages the route table in shared memory, as the router's push does."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils
    from cutlass.cutlass_dsl import Int32, Int64

    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.api import ImplDesc, ProblemDesc
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.communication.nvlink_domain.one_round_router import (
        _OneRoundRouter,
    )

    router = _OneRoundRouter(
        ProblemDesc(
            {
                "world_size": _AR_WORLD,
                "expert_count": _AR_SLOTS,
                "topk": _AR_TOP_K,
                "max_tokens_per_rank": rows,
                "hidden_size": 7168,
                "quant_kind": "nvfp4",
                "apply_topk_at_fc1": True,
            }
        ),
        ImplDesc(
            {
                "token_padding_block": 64,
                "one_round_cta_count": 512 // _AR_WORLD,
                "adaptive_route_experts": _AR_EXPERTS,
            }
        ),
    )
    warps = 4

    class Probe:
        @cute.kernel
        def kernel(
            self,
            ids: cute.Tensor,
            out: cute.Tensor,
            tables: Int64,
            stat_rows: Int64,
            forwards: Int64,
            enabled: Int64,
            word: Int64,
            num_tokens: Int32,
            first: Int32,
            ep_rank: Int32,
        ):
            thread_idx, _, _ = cute.arch.thread_idx()
            cta_idx, _, _ = cute.arch.block_idx()
            warp = thread_idx // Int32(32)
            lane = thread_idx % Int32(32)
            adaptive = (tables, stat_rows, forwards, enabled, word, num_tokens, first)
            table = None
            if cutlass.const_expr(staged):
                workspace = router.router_smem_workspace
                storage = cutlass.utils.SmemAllocator().allocate(workspace.storage_class())
                table = workspace.tensor(
                    router.smem_adaptive_table_region, storage.buffer.data_ptr()
                )
                entries = router._adaptive_table_load(tables, thread_idx, warps * 32)
                router._adaptive_table_store(entries, table, thread_idx, warps * 32)
                cute.arch.sync_threads()
            counted = router._adaptive_counted_rows(adaptive, cta_idx, thread_idx)
            token = cta_idx * Int32(warps) + warp
            if token < Int32(rows):
                if lane < Int32(_AR_TOP_K):
                    raw = ids[token, lane]
                    slot = Int32(-1)
                    if cutlass.Uint64(raw) < cutlass.Uint64(_AR_EXPERTS):
                        slot = router._adaptive_pick(
                            adaptive, counted, Int32(raw), token, ep_rank, table
                        )
                    out[token, lane] = slot

        @cute.jit
        def __call__(
            self,
            ids: cute.Tensor,
            out: cute.Tensor,
            tables: Int64,
            stat_rows: Int64,
            forwards: Int64,
            enabled: Int64,
            word: Int64,
            num_tokens: Int32,
            first: Int32,
            ep_rank: Int32,
            stream,
        ):
            self.kernel(
                ids, out, tables, stat_rows, forwards, enabled, word, num_tokens, first, ep_rank
            ).launch(
                grid=[(rows + warps - 1) // warps, 1, 1], block=[warps * 32, 1, 1], stream=stream
            )

    fake = cute.runtime.make_fake_compact_tensor
    return cute.compile(
        Probe(),
        fake(cutlass.Int64, (rows, _AR_TOP_K), stride_order=(1, 0)),
        fake(cutlass.Int32, (rows, _AR_TOP_K), stride_order=(1, 0)),
        *(Int64(0),) * 5,
        Int32(0),
        Int32(1),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@_requires_cute_gpu
@pytest.mark.parametrize("staged", [False, True], ids=["global", "staged"])
@pytest.mark.parametrize("rows", [8, 32, 64, 256])
def test_router_adaptive_route_is_bitwise_its_definition(rows, staged):
    """Slots and statistics of the router's adaptive route equal the reference's, with replicated experts,
    0/1 shares, invalid ids, padding rows past the forward's rows, live-row words and the recording flag; with
    the table read from global memory and from the copy the router stages in shared memory."""
    import random

    from _torch.moe import adaptive_eplb_reference as reference

    rr = _router_route_module()
    rng = random.Random(rows)
    compiled = _compile_router_probe(rows, staged)
    device = torch.device("cuda")
    for case in range(12):
        packed = _route_table(rng, zero_one=0.4 if case % 2 else 0.0)
        table = torch.tensor(packed, dtype=torch.int32, device=device)
        num_tokens = rng.randint(1, rows)
        ids = [[rng.randrange(_AR_EXPERTS) for _ in range(_AR_TOP_K)] for _ in range(rows)]
        for t in range(rows):
            if t >= num_tokens:
                ids[t] = [-1] * _AR_TOP_K
            elif rng.random() < 0.1:
                ids[t][rng.randrange(_AR_TOP_K)] = rng.choice([-1, _AR_EXPERTS, _AR_SLOTS + 3])
        requests = rng.choice([0, 1, 2, 4, num_tokens])
        real = rng.randint(0, requests) if requests else 0
        word = None if case % 3 == 0 else (requests << 16) | real
        enabled = case % 4 != 3
        first = case % 5 != 4
        ep_rank = rng.randrange(_AR_WORLD)

        stat_rows = torch.zeros(_AR_EXPERTS, dtype=torch.float32, device=device)
        forwards = torch.zeros(1, dtype=torch.int64, device=device)
        flag = torch.tensor([1 if enabled else 0], dtype=torch.int32, device=device)
        word_t = torch.tensor(
            [0x7FFFFFFF if word is None else word], dtype=torch.int32, device=device
        )
        out = torch.full((rows, _AR_TOP_K), -7, dtype=torch.int32, device=device)
        ids_t = torch.tensor(ids, dtype=torch.int64, device=device)
        compiled(
            ids_t,
            out,
            table.data_ptr(),
            stat_rows.data_ptr(),
            forwards.data_ptr(),
            flag.data_ptr(),
            word_t.data_ptr(),
            num_tokens,
            1 if first else 0,
            ep_rank,
        )
        torch.cuda.synchronize()

        want_slots = rr.slots_of(ids, ep_rank, packed, _AR_EXPERTS, _AR_SLOTS)
        # The statistics are those the staged route records over the forward's rows (a word that
        # keeps every row reads as 0x7FFF real of 0x7FFF requests).
        want_rows, want_forwards = torch.zeros(_AR_EXPERTS), torch.zeros(1, dtype=torch.int64)
        reference.record(
            torch.tensor(ids[:num_tokens]),
            want_rows,
            want_forwards,
            torch.tensor([int(enabled)]),
            first,
            word_t.cpu(),
        )
        assert out.cpu().tolist() == want_slots, (rows, case)
        assert stat_rows.cpu().tolist() == want_rows.tolist(), (rows, case)
        assert forwards.item() == want_forwards.item(), (rows, case)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="needs a Blackwell CUDA device",
)
@pytest.mark.parametrize("rows", [16, 64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_route_quant_into_the_bucket_stages_what_the_copy_path_stages(rows, index_dtype):
    """DeepSeek-V4's fused route+quant into the bucket, under a live-row word, leaves every byte as
    the launch into ordinary tensors followed by the copy path does, padding rows included."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("needs the CuTe DSL")
    from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import dsv4_route_nvfp4_quant

    bucket, hidden, experts, scaling = 256, 7168, 384, 2.5
    gen = torch.Generator().manual_seed(rows)
    logits = (torch.randn(rows, experts, generator=gen) * 4).cuda()
    bias = (torch.randn(experts, generator=gen) * 0.5).cuda()
    x = (torch.randn(rows, hidden, generator=gen) * 3).to(torch.bfloat16).cuda()
    scale = torch.tensor([448.0 * 6.0 / 20.0], device="cuda")
    shapes = dict(
        topk_idx_local=(TOP_K, index_dtype),
        topk_weights=(TOP_K, torch.float32),
        activation=(hidden // 2, torch.uint8),
        activation_sf=(hidden // 16, torch.uint8),
    )
    moe = SimpleNamespace(_wait_rebalance_routes=lambda replica_plan=None: None)
    for value in (live_rows.ALL_ROWS, live_rows.encode(3, 4), live_rows.NO_ROWS):
        word = torch.tensor([value], dtype=torch.int64).to(torch.int32).cuda()
        fused = SimpleNamespace(
            **{k: _garbage((bucket, n), t, gen, "cuda") for k, (n, t) in shapes.items()}
        )
        copied = SimpleNamespace(**{k: v.clone() for k, v in vars(fused).items()})
        args = (logits, bias, x, scale, TOP_K, scaling, torch.float32)
        staged_ids, *_ = dsv4_route_nvfp4_quant(
            *args,
            staged=tuple(getattr(fused, name) for name in shapes),  # ids, weights, x, scales
            live_row_word=word,
        )
        ids, weights, x_fp4, x_sf, _ = dsv4_route_nvfp4_quant(*args)
        assert torch.equal(staged_ids, ids), hex(value)
        MegaMoECuteDsl._stage_inputs(
            moe,
            bufs=copied,
            x=x_fp4,
            x_sf=x_sf,
            topk_idx=ids,
            topk_weights=weights,
            num_tokens=rows,
            top_k=TOP_K,
            stage_activation=True,
            live_row_word=word,
        )
        for name, want in vars(copied).items():
            got = getattr(fused, name).view(torch.uint8)
            assert torch.equal(got, want.view(torch.uint8)), (name, hex(value))
