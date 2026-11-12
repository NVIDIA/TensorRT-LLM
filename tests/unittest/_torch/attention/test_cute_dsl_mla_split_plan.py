# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request split-KV plan of the CuTe DSL MLA decode: policy, kernel vs mirror, kernel output."""

import contextlib
import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_CUTLASS_DSL_AVAILABLE or not is_sm_100f(),
    reason="CuTe DSL MLA requires an SM100-family CUDA GPU",
)

# One Rubin wave: 212 SMs, one 2-CTA cluster per SM pair -> 106 clusters.
WAVE = 106


class _KernelCacheKeyLog(dict):
    """kernel_cache stand-in: records the keys the runner looks up, stores into the shared cache."""

    def __init__(self, cache):
        super().__init__()
        self.cache, self.keys = cache, []

    def __contains__(self, key):
        self.keys.append(key)
        return key in self.cache

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value


def _plan_mod():
    import tensorrt_llm._torch.attention_backend.fmha.cute_dsl_mla as mod

    return mod


@pytest.mark.parametrize(
    "kv_lens,rows,expected",
    [
        # helix16 bs8 MTP7 (8 rows per request): 64 clusters fill the wave at one task each.
        ([14208] * 8, 8, [1] * 8),
        # helix16 bs4: 32 rows -> three pieces per request (96 clusters, one wave).
        ([14208] * 4, 8, [3] * 4),
        # helix16 bs2: six pieces of 19 tiles.
        ([14208] * 2, 8, [6] * 2),
        # helix16 bs16: already over one wave, a second piece would cost a third wave.
        ([14208] * 16, 8, [1] * 16),
        # One long request alone (H128, single token): the whole grid.
        ([65536], 1, [32]),
        # 64K x 8 (64 rows): three pieces fill two waves better than one 512-tile pass.
        ([65536] * 8, 8, [3] * 8),
        # Short requests never get pieces below the minimum tile count.
        ([512] * 4, 8, [1] * 4),
        # DEP skew: the 256K request takes eight 256-tile pieces, the 32K ones stay whole.
        ([32768] * 15 + [262144], 8, [1] * 15 + [8]),
    ],
)
def test_plan_mla_split_kv_policy(kv_lens: list[int], rows: int, expected: list[int]) -> None:
    plan = _plan_mod().plan_mla_split_kv(kv_lens, rows_per_request=rows, wave_clusters=WAVE)
    assert plan == expected


@pytest.mark.parametrize("piece_cost", [None, 0, 30, 60])
def test_plan_kernel_matches_host_mirror(piece_cost) -> None:
    """Every piece cost (None: the target decode's)."""
    mod = _plan_mod()
    generator = torch.Generator().manual_seed(3)
    cost = {} if piece_cost is None else {"piece_cost": piece_cost}
    for rows in (1, 4, 8):
        for batch in (1, 3, 8, 16, 37, 128):
            kv = torch.randint(1, 300_000, (batch,), generator=generator)
            kv[0] = 262144  # a long one
            kv_lens = kv.tolist()
            expected = mod.plan_mla_split_kv(
                kv_lens, rows_per_request=rows, wave_clusters=WAVE, **cost
            )
            cache_seqs = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
            plan = torch.full((batch + 3,), -5, dtype=torch.int32, device="cuda")
            mod.launch_mla_split_kv_plan(
                cache_seqs,
                plan,
                num_requests=batch,
                rows_per_request=rows,
                wave_clusters=WAVE,
                **cost,
            )
            torch.cuda.synchronize()
            assert plan[:batch].tolist() == expected, (rows, batch, kv_lens[:4])
            assert (plan[batch:] == -5).all()


def test_plan_piece_cost_changes_the_plan() -> None:
    """The piece cost is a launch argument of the kernel, not a constant of it."""
    mod = _plan_mod()
    kv_lens = [14208] * 8
    plans = {
        piece_cost: mod.plan_mla_split_kv(
            kv_lens, rows_per_request=8, wave_clusters=WAVE, piece_cost=piece_cost
        )
        for piece_cost in (0, 400)
    }
    assert plans[0] == [28] * 8 and plans[400] == [1] * 8
    cache_seqs = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    plan = torch.zeros(8, dtype=torch.int32, device="cuda")
    for piece_cost, expected in plans.items():
        mod.launch_mla_split_kv_plan(
            cache_seqs,
            plan,
            num_requests=8,
            rows_per_request=8,
            wave_clusters=WAVE,
            piece_cost=piece_cost,
        )
        assert plan.tolist() == expected
    with pytest.raises(ValueError, match="piece cost"):
        mod.launch_mla_split_kv_plan(
            cache_seqs,
            plan,
            num_requests=8,
            rows_per_request=8,
            wave_clusters=WAVE,
            piece_cost=-1,
        )


def test_split_kv_plan_is_launched_once_per_step(monkeypatch) -> None:
    mod = _plan_mod()
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    monkeypatch.setattr(
        CuteDSLNVMlaDecodeBlackwellRunner, "_get_max_active_blocks", classmethod(lambda cls: 212)
    )
    monkeypatch.setattr(mod, "_COMPACT_PLAN", False)
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_PLAN_FILL_SPLIT", False)
    launches = []
    monkeypatch.setattr(
        mod, "launch_mla_split_kv_plan", lambda *args, **kwargs: launches.append(kwargs)
    )
    meta = SimpleNamespace(max_num_requests=16, _cute_dsl_mla_split_plan_key=None)
    cache_seqs = torch.full((8,), 14208, dtype=torch.int32, device="cuda")
    first = mod.CuteDslMlaFmha._split_kv_plan(meta, cache_seqs, 8, 8, 96)
    assert first.numel() == 8 and meta._cute_dsl_mla_split_plan.numel() == 16
    assert launches == [dict(num_requests=8, rows_per_request=8, wave_clusters=106)]
    # The other layers of the step reuse the plan.
    second = mod.CuteDslMlaFmha._split_kv_plan(meta, cache_seqs, 8, 8, 96)
    assert second.data_ptr() == first.data_ptr() and len(launches) == 1
    # A new step (metadata invalidation) plans again.
    meta._cute_dsl_mla_split_plan_key = None
    mod.CuteDslMlaFmha._split_kv_plan(meta, cache_seqs, 8, 8, 96)
    assert len(launches) == 2
    # Beyond the plan's request capacity: uniform split.
    assert mod.CuteDslMlaFmha._split_kv_plan(meta, cache_seqs, 2000, 8, 96) is None


@pytest.mark.parametrize("reduction", ["default", "prefetch", "prefetch-4band"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("empty_request", [False, True])
def test_runner_variable_split_matches_uniform_split(
    monkeypatch, input_dtype: torch.dtype, empty_request: bool, reduction: str
) -> None:
    """Per-request splits under a 16-slot grid reproduce the uniform-split output and Helix stats.

    Covers requests planned onto one split (stored directly by the split
    kernel), onto several, onto more splits than KV tiles, and a request
    without local KV (the reduction's canonical empty result), and the
    uniform split. The prefetching reduction (one or four D bands per row)
    is bitwise the default one.
    """
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    prefetch = reduction != "default"
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_REDUCTION_PREFETCH", prefetch)
    # No row cap here (it has its own tests): the shape below has 768 query rows.
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_REDUCTION_PREFETCH_MAX_ROWS", 0)
    # 4 * B * S * H rows always fit this many waves: four D bands.
    max_waves = 10**6 if reduction == "prefetch-4band" else 0
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_REDUCTION_D_BAND_MAX_WAVES", max_waves)
    key_log = _KernelCacheKeyLog(CuteDSLNVMlaDecodeBlackwellRunner.kernel_cache)
    monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "kernel_cache", key_log)
    torch.manual_seed(5)
    device = torch.device("cuda")
    batch_size, seq_len_q, num_heads = 4, 2, 96
    latent_dim, rope_dim, page_size = 512, 64, 64
    kv_lens = [4096, 0 if empty_request else 300, 9000, 1500]
    blocks_per_sequence = math.ceil(max(kv_lens) / page_size)
    num_pages = batch_size * blocks_per_sequence
    softmax_scale = 1.0 / math.sqrt(latent_dim + rope_dim)
    q_storage = (
        torch.randn(batch_size, seq_len_q, num_heads, latent_dim + rope_dim, device=device) * 0.1
    ).to(input_dtype)
    cache_storage = (
        torch.randn(num_pages, page_size, latent_dim + rope_dim, device=device) * 0.1
    ).to(input_dtype)
    q_latent = q_storage[..., :latent_dim].permute(2, 3, 1, 0)
    q_rope = q_storage[..., latent_dim:].permute(2, 3, 1, 0)
    c_latent = cache_storage[..., :latent_dim].permute(1, 2, 0)
    c_rope = cache_storage[..., latent_dim:].permute(1, 2, 0)
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device=device)
        .view(batch_size, blocks_per_sequence)
        .transpose(0, 1)
    )
    cache_seqs = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    kv_bounds = torch.tensor(
        [max(k - (seq_len_q - 1 - q), 0) for k in kv_lens for q in range(seq_len_q)],
        dtype=torch.int32,
        device=device,
    )
    cutlass_dtype = cutlass.Float8E4M3FN if input_dtype == torch.float8_e4m3fn else cutlass.BFloat16
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass_dtype,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        page_size=page_size,
        max_batch_size=batch_size,
        emit_softmax_stats=True,
    )
    workspace = torch.empty(
        runner.get_max_padded_workspace_size(
            num_heads, seq_len_q, latent_dim, batch_size, cutlass.Float32
        ),
        device=device,
        dtype=torch.uint8,
    )

    def run(block_split_kvs, split_kv):
        output_storage = torch.zeros(
            batch_size, seq_len_q, num_heads, latent_dim, device=device, dtype=torch.bfloat16
        )
        stats = torch.zeros(batch_size * seq_len_q, num_heads, 2, device=device)
        inputs = [
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_table,
            cache_seqs,
            output_storage.permute(2, 3, 1, 0),
            workspace,
            stats,
            kv_bounds,
        ]
        kwargs = {"softmax_scale": softmax_scale, "output_scale": 1.0}
        if input_dtype == torch.float8_e4m3fn:
            inputs.extend(
                [torch.tensor([softmax_scale], device=device), torch.tensor([1.0], device=device)]
            )
            kwargs = {}
        if block_split_kvs is not None:
            inputs.append(torch.tensor(block_split_kvs, dtype=torch.int32, device=device))
        runner.forward(inputs, tactic=((128, 128), (128, 256), split_kv, False), **kwargs)
        torch.cuda.synchronize()
        return output_storage.float(), stats

    ref_out, ref_stats = run(None, 1)
    for plan, split_kv in (
        ([3, 1, 7, 2], 16),
        ([16, 16, 16, 16], 16),
        ([40, 1, 2, 1], 8),
        ([1, 1, 1, 1], 16),
        (None, 16),
    ):
        out, stats = run(plan, split_kv)
        # The last two kernel-cache key fields are (reducer_d_tiles, reducer_prefetch_splits).
        bands = 4 if reduction == "prefetch-4band" else 1
        assert key_log.keys[-1][-2:] == ((bands, 32) if prefetch else (1, 0))
        if prefetch:
            monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_REDUCTION_PREFETCH", False)
            out_default, stats_default = run(plan, split_kv)
            monkeypatch.setattr(CuteDSLNVMlaDecodeBlackwellRunner, "_REDUCTION_PREFETCH", True)
            assert torch.equal(out, out_default) and torch.equal(stats, stats_default)
        tol = 2e-2 if input_dtype == torch.float8_e4m3fn else 1e-2
        torch.testing.assert_close(out, ref_out, rtol=tol, atol=tol)
        log_partition = stats[:, :, 0] + stats[:, :, 1].log()
        ref_log_partition = ref_stats[:, :, 0] + ref_stats[:, :, 1].log()
        finite = torch.isfinite(ref_log_partition)
        assert torch.equal(torch.isfinite(log_partition), finite)
        torch.testing.assert_close(
            log_partition[finite], ref_log_partition[finite], rtol=1e-4, atol=1e-4
        )


def test_reducer_d_tiles_rule(monkeypatch) -> None:
    """Four D bands only with the prefetching reduction and while 4 * rows fits the wave budget (no row cap)."""
    import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops as ops

    runner = ops.CuteDSLNVMlaDecodeBlackwellRunner
    monkeypatch.setattr(ops, "_get_num_sms", lambda: 212)
    monkeypatch.setattr(runner, "_REDUCTION_PREFETCH", False)
    assert runner.get_reducer_d_tiles(1) == 1
    monkeypatch.setattr(runner, "_REDUCTION_PREFETCH", True)
    # The row cap has its own tests (test_cute_dsl_mla_reduction_cap.py): off here.
    monkeypatch.setattr(runner, "_REDUCTION_PREFETCH_MAX_ROWS", 0)
    last = runner._REDUCTION_D_BAND_MAX_WAVES * 212 // 4
    assert runner.get_reducer_d_tiles(last) == 4
    assert runner.get_reducer_d_tiles(last + 1) == 1
    # Decode shapes with 12 heads per rank: B * S * 12 query rows. The 12-wave budget takes
    # four bands up to B = 8 at S = 6 and B = 7 at S = 7.
    assert runner.get_reducer_d_tiles(8 * 6 * 12) == 4
    assert runner.get_reducer_d_tiles(7 * 7 * 12) == 4
    assert runner.get_reducer_d_tiles(9 * 6 * 12) == 1
    assert runner.get_reducer_d_tiles(8 * 7 * 12) == 1


def _plan_launch(mod, kv_lens, rows, max_split, **launch_kwargs):
    batch = len(kv_lens)
    cache_seqs = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    plan = torch.full((batch + 3,), -5, dtype=torch.int32, device="cuda")
    mod.launch_mla_split_kv_plan(
        cache_seqs,
        plan,
        num_requests=batch,
        rows_per_request=rows,
        wave_clusters=WAVE,
        max_split=max_split,
        **launch_kwargs,
    )
    return plan


def _kv_cases(batch: int, generator: torch.Generator) -> dict[str, list[int]]:
    mixed = torch.randint(1, 300_000, (batch,), generator=generator).tolist()
    mixed[0] = 262144
    skewed = [2048] * batch
    skewed[-1] = 245760
    return {"equal": [81920] * batch, "mixed": mixed, "skewed": skewed}


@pytest.mark.parametrize("max_split", [32, 64])
def test_compact_plan_is_bitwise_the_default_launch(max_split: int) -> None:
    """Request lanes sized to the call plan every batch exactly as the 1024-lane launch does.

    Every batch size 1..33 (both sides of the 16 / 32 lane boundaries), equal, mixed and skewed
    KV lengths, one and eight rows per request. The default launch tries 32 piece counts at any
    cap, and so does the compact one.
    """
    mod = _plan_mod()
    generator = torch.Generator().manual_seed(21)
    plans_with_splits = 0
    for batch in range(1, 34):
        for name, kv_lens in _kv_cases(batch, generator).items():
            for rows in (1, 8):
                default = _plan_launch(mod, kv_lens, rows, max_split)
                compact = _plan_launch(mod, kv_lens, rows, max_split, compact=True)
                torch.cuda.synchronize()
                assert torch.equal(compact, default), (batch, name, rows, kv_lens[:4])
                assert (compact[batch:] == -5).all()
                plans_with_splits += int(compact[:batch].max().item() > 1)
    assert plans_with_splits > 100


class _KernelLaunchLog:
    """Stand-in for the plan kernel: records the constexpr launch arguments, then launches it."""

    def __init__(self, kernel):
        self.kernel, self.launches = kernel, []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append({k: kwargs[k] for k in ("BLOCK", "MAX_PIECES")})
            return self.kernel[grid](*args, **kwargs)

        return launch


def test_compact_plan_constexpr_variants(monkeypatch) -> None:
    """Lanes: 16 up to 16 requests, then the next power of two; 1024 without ``compact``."""
    mod = _plan_mod()
    log = _KernelLaunchLog(mod._mla_split_kv_plan_kernel)
    monkeypatch.setattr(mod, "_mla_split_kv_plan_kernel", log)
    expected = {1: 16, 2: 16, 15: 16, 16: 16, 17: 32, 32: 32, 33: 64, 64: 64, 65: 128, 1024: 1024}
    for batch, block in expected.items():
        _plan_launch(mod, [81920] * batch, 1, 32, compact=True)
        assert log.launches[-1] == dict(BLOCK=block, MAX_PIECES=32), batch
    for batch in (1, 17):
        _plan_launch(mod, [81920] * batch, 1, 32)
        assert log.launches[-1] == dict(BLOCK=1024, MAX_PIECES=32), batch
    # Another cap is one more variant per lane count, not a change of the default one.
    _plan_launch(mod, [81920], 1, 64, compact=True, candidates=64)
    assert log.launches[-1] == dict(BLOCK=16, MAX_PIECES=64)
    torch.cuda.synchronize()
    with pytest.raises(ValueError, match="piece candidate"):
        _plan_launch(mod, [81920], 1, 32, candidates=0)


def test_split_kv_plan_launch_arguments_with_compact_plan(monkeypatch) -> None:
    mod = _plan_mod()
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    monkeypatch.setattr(
        CuteDSLNVMlaDecodeBlackwellRunner, "_get_max_active_blocks", classmethod(lambda cls: 212)
    )
    launches = []
    monkeypatch.setattr(
        mod, "launch_mla_split_kv_plan", lambda *args, **kwargs: launches.append(kwargs)
    )
    cache_seqs = torch.full((8,), 14208, dtype=torch.int32, device="cuda")
    for compact, extra in ((False, {}), (True, dict(compact=True))):
        monkeypatch.setattr(mod, "_COMPACT_PLAN", compact)
        meta = SimpleNamespace(max_num_requests=16, _cute_dsl_mla_split_plan_key=None)
        mod.CuteDslMlaFmha._split_kv_plan(meta, cache_seqs, 8, 8, 96)
        assert launches[-1] == dict(
            num_requests=8, rows_per_request=8, wave_clusters=106, **extra
        ), compact


def _fp8_decode_runner(max_batch: int, num_heads: int = 12, seq_len_q: int = 6):
    """A FP8 decode runner and its workspace, and a factory of decode cases for it."""
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    latent_dim, rope_dim, page_size = 512, 64, 64
    runner = CuteDSLNVMlaDecodeBlackwellRunner(
        in_dtype=cutlass.Float8E4M3FN,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        page_size=page_size,
        max_batch_size=max_batch,
    )
    device = torch.device("cuda")
    workspace = torch.empty(
        runner.get_max_padded_workspace_size(
            num_heads, seq_len_q, latent_dim, max_batch, cutlass.Float32
        ),
        device=device,
        dtype=torch.uint8,
    )
    softmax_scale = 1.0 / math.sqrt(latent_dim + rope_dim)

    def make_case(kv_lens):
        batch = len(kv_lens)
        blocks = math.ceil(max(kv_lens) / page_size)
        num_pages = batch * blocks
        q = (
            torch.randn(batch, seq_len_q, num_heads, latent_dim + rope_dim, device=device) * 0.1
        ).to(torch.float8_e4m3fn)
        cache = (torch.randn(num_pages, page_size, latent_dim + rope_dim, device=device) * 0.1).to(
            torch.float8_e4m3fn
        )
        page_table = (
            torch.arange(num_pages, dtype=torch.int32, device=device)
            .view(batch, blocks)
            .transpose(0, 1)
        )
        cache_seqs = torch.tensor(kv_lens, dtype=torch.int32, device=device)
        scales = [torch.tensor([softmax_scale], device=device), torch.tensor([1.0], device=device)]

        def run(plan, split_kv, **kwargs):
            out = torch.zeros(
                batch, seq_len_q, num_heads, latent_dim, device=device, dtype=torch.bfloat16
            )
            inputs = [
                q[..., :latent_dim].permute(2, 3, 1, 0),
                q[..., latent_dim:].permute(2, 3, 1, 0),
                cache[..., :latent_dim].permute(1, 2, 0),
                cache[..., latent_dim:].permute(1, 2, 0),
                page_table,
                cache_seqs,
                out.permute(2, 3, 1, 0),
                workspace,
                None,
                None,
                *scales,
            ]
            if plan is not None:
                inputs.append(plan)
            runner.forward(inputs, tactic=((128, 128), (128, 256), split_kv, False), **kwargs)
            torch.cuda.synchronize()
            return out.float()

        return run

    return runner, make_case


def test_compact_plan_outputs_are_bitwise_the_default() -> None:
    """Decode outputs under the compact launch's plan equal the default launch's, bit for bit."""
    mod = _plan_mod()
    runner, make_case = _fp8_decode_runner(max_batch=32)
    generator = torch.Generator().manual_seed(4)
    torch.manual_seed(4)
    for batch in (1, 2, 3, 16, 17, 32):
        kv_lens = torch.randint(300, 30000, (batch,), generator=generator).tolist()
        kv_lens[0] = 40000
        rows = runner.get_folded_seq_len_q(12, 6)
        default = _plan_launch(mod, kv_lens, rows, 32)[:batch].contiguous()
        compact = _plan_launch(mod, kv_lens, rows, 32, compact=True)[:batch].contiguous()
        torch.cuda.synchronize()
        assert torch.equal(compact, default) and int(default.max()) > 1
        run = make_case(kv_lens)
        assert torch.equal(run(compact, 32), run(default, 32)), batch


@contextlib.contextmanager
def _split_workspace_sizing(monkeypatch, *, fill: bool, compact: bool = False):
    """The fill and compact switches set, with the split-KV workspace sizing caches clean.

    The caches key on the shape, not on the switch, so they are cleared on both sides.
    """
    import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops as ops

    runner = ops.CuteDSLNVMlaDecodeBlackwellRunner
    monkeypatch.setattr(_plan_mod(), "_COMPACT_PLAN", compact)
    monkeypatch.setattr(runner, "_PLAN_FILL_SPLIT", fill)
    try:
        runner._max_split_kv_rows.cache_clear()
        runner._max_split_kv_acc_rows.cache_clear()
        yield runner
    finally:
        runner._max_split_kv_rows.cache_clear()
        runner._max_split_kv_acc_rows.cache_clear()


def test_plan_fill_split_rule(monkeypatch) -> None:
    """The fill switch lifts the plan's split cap from 32 to 64 for at most two query rows."""
    with _split_workspace_sizing(monkeypatch, fill=False) as runner:
        assert all(runner.get_plan_fill_split(b, s) == 0 for b in (1, 2, 3, 16) for s in (1, 2, 6))
    with _split_workspace_sizing(monkeypatch, fill=True) as runner:
        assert [runner.get_plan_fill_split(b, s) for b, s in ((1, 1), (2, 1), (1, 2))] == [64] * 3
        assert [runner.get_plan_fill_split(b, s) for b, s in ((3, 1), (2, 2), (1, 3), (16, 1))] == [
            0
        ] * 4


@pytest.mark.parametrize("max_split", [32, 64, 128])
def test_plan_kernel_with_candidates_matches_host_mirror(max_split: int) -> None:
    """A launch that tries ``max_split`` piece counts plans like the host mirror, at any cap."""
    mod = _plan_mod()
    generator = torch.Generator().manual_seed(11)
    for rows in (1, 2, 8):
        for batch in (1, 2, 3, 8, 16, 17, 40):
            kv_lens = torch.randint(1, 300_000, (batch,), generator=generator).tolist()
            kv_lens[0] = 262144
            expected = mod.plan_mla_split_kv(
                kv_lens, rows_per_request=rows, wave_clusters=WAVE, max_split=max_split
            )
            plan = _plan_launch(mod, kv_lens, rows, max_split, compact=True, candidates=max_split)
            torch.cuda.synchronize()
            assert plan[:batch].tolist() == expected, (rows, batch, kv_lens[:4])
            assert (plan[batch:] == -5).all()
    # Alone on the GPU a long request takes the whole split cap, which 32 candidates cannot reach.
    kv_lens = [65536]
    assert mod.plan_mla_split_kv(kv_lens, rows_per_request=1, wave_clusters=WAVE, max_split=64) == [
        64
    ]
    default = _plan_launch(mod, kv_lens, 1, 64)
    wide = _plan_launch(mod, kv_lens, 1, 64, compact=True, candidates=64)
    torch.cuda.synchronize()
    assert default[:1].tolist() == [32] and wide[:1].tolist() == [64]


@pytest.mark.parametrize("compact", [False, True], ids=["fill", "fill+compact"])
def test_split_kv_plan_launch_arguments_with_fill(monkeypatch, compact: bool) -> None:
    mod = _plan_mod()
    launches = []
    monkeypatch.setattr(
        mod, "launch_mla_split_kv_plan", lambda *args, **kwargs: launches.append(kwargs)
    )
    meta = SimpleNamespace(max_num_requests=16, _cute_dsl_mla_split_plan_key=None)
    with _split_workspace_sizing(monkeypatch, fill=True, compact=compact) as runner:
        monkeypatch.setattr(runner, "_get_max_active_blocks", classmethod(lambda cls: 212))
        # 12 heads per rank, S = 6: one folded row per request. The plan with the wide cap is
        # compact (and tries as many piece counts as it may split); the others are as without
        # the switch.
        for batch, extra in (
            (1, dict(max_split=64, candidates=64, compact=True)),
            (2, dict(max_split=64, candidates=64, compact=True)),
            (3, dict(compact=True) if compact else {}),
            (8, dict(compact=True) if compact else {}),
        ):
            meta._cute_dsl_mla_split_plan_key = None
            cache_seqs = torch.full((batch,), 65536, dtype=torch.int32, device="cuda")
            mod.CuteDslMlaFmha._split_kv_plan(meta, cache_seqs, batch, 6, 12)
            assert launches[-1] == dict(
                num_requests=batch, rows_per_request=1, wave_clusters=106, **extra
            ), batch


class _SplitKvLog(_KernelCacheKeyLog):
    """kernel_cache stand-in that also records the split_kv each launch passes to the kernel."""

    def __init__(self, cache):
        super().__init__(cache)
        self.splits = []

    def __getitem__(self, key):
        kernel = self.cache[key]

        def launch(*args):
            # runtime args end: ..., split_workspace, split_kv, cache_seqs, kv_bounds,
            # block_split_kvs, softmax_scale, output_scale, stream
            self.splits.append(args[-7])
            return kernel(*args)

        return launch


def test_runner_fill_split_matches_unsplit(monkeypatch) -> None:
    """Plans of up to 64 splits (B <= 2, FP8 KV, 12 heads, S = 6) stay within the unsplit tolerance."""
    torch.manual_seed(9)
    num_heads, seq_len_q, max_batch = 12, 6, 4
    with _split_workspace_sizing(monkeypatch, fill=True) as runner_cls:
        runner, make_case = _fp8_decode_runner(max_batch, num_heads, seq_len_q)
        assert runner.get_fitting_split_kv(1, max_batch) >= 64
        assert runner.get_fitting_split_kv(2, max_batch) >= 64
        for kv_lens in ([40000], [40000, 9000]):
            batch = len(kv_lens)
            run = make_case(kv_lens)
            rows = runner.get_folded_seq_len_q(num_heads, seq_len_q)
            plan64 = _plan_mod().plan_mla_split_kv(
                kv_lens,
                rows_per_request=rows,
                wave_clusters=WAVE,
                max_split=runner_cls.get_plan_fill_split(batch, rows),
                piece_cost=60,
            )
            assert max(plan64) > 32, plan64
            reference = run(None, 1)
            # The tuned grid is the default 32 slots: the runner widens it for the plan.
            out64 = run(
                torch.tensor(plan64, dtype=torch.int32, device="cuda"), 32, plan_max_split=64
            )
            torch.testing.assert_close(out64, reference, rtol=2e-2, atol=2e-2)
            plan32 = torch.tensor([min(p, 32) for p in plan64], dtype=torch.int32, device="cuda")
            out32 = run(plan32, 32)
            torch.testing.assert_close(out32, reference, rtol=2e-2, atol=2e-2)
            # More splits merge in another order: the plan of up to 64 splits really ran.
            assert not torch.equal(out64, out32)


@pytest.mark.parametrize("compact", [False, True], ids=["fill", "fill+compact"])
def test_fill_grid_follows_the_cap_of_the_plan_passed(monkeypatch, compact: bool) -> None:
    """Only a plan that says it has up to 64 splits gets a 64-wide grid.

    A plan capped at 32 (the DSpark drafter's plan, ``TLLM_DSPARK_MLA_SPLIT_PLAN``) passes
    no cap: it keeps the tuned grid, whatever the switches are, and a plan without a plan
    keeps it too.
    """
    torch.manual_seed(3)
    kv_lens = [40000]
    with _split_workspace_sizing(monkeypatch, fill=True, compact=compact) as runner_cls:
        runner, make_case = _fp8_decode_runner(2)
        log = _SplitKvLog(runner_cls.kernel_cache)
        monkeypatch.setattr(runner_cls, "kernel_cache", log)
        run = make_case(kv_lens)
        plan = torch.tensor([20], dtype=torch.int32, device="cuda")
        run(plan, 32, plan_max_split=64)
        run(plan, 32)  # a plan capped at 32: no cap passed
        run(plan, 16)  # and the tuned grid is not widened to the cap of 32 either
        run(None, 32, plan_max_split=64)  # no plan: no widening
        assert log.splits == [64, 32, 16, 32]


@pytest.mark.parametrize(
    "fill,batch,expected",
    [
        (True, 1, {"plan_max_split": 64}),
        (True, 2, {"plan_max_split": 64}),
        (True, 3, {}),
        (False, 1, {}),
    ],
)
def test_decode_call_passes_the_cap_of_its_plan(
    monkeypatch, fill: bool, batch: int, expected
) -> None:
    """``_run_mla_decode`` hands the op the cap of the plan it passes, and nothing for a plan of 32."""
    mod = _plan_mod()
    calls = []

    def record(*args, **kwargs):
        calls.append((len(args), kwargs))

    monkeypatch.setattr(
        torch.ops,
        "trtllm",
        SimpleNamespace(
            cute_dsl_mla_decode_fp8_blackwell=record, cute_dsl_mla_decode_fp16_blackwell=record
        ),
    )
    device = torch.device("cuda")
    heads, seq_len_q, latent, rope, page = 12, 6, 512, 64, 64
    blocks, pages = 8, 8 * batch
    attn = SimpleNamespace(
        kv_lora_rank=latent,
        qk_rope_head_dim=rope,
        qk_nope_head_dim=128,
        num_heads=heads,
        q_scaling=1.0,
        layer_idx=0,
        get_local_layer_idx=lambda meta: 0,
    )
    meta = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            get_buffers=lambda layer: torch.zeros(
                pages, 1, page, 1, latent + rope, device=device, dtype=torch.float8_e4m3fn
            )
        ),
        kv_cache_block_offsets=torch.zeros(1, batch, 2, blocks, device=device, dtype=torch.int32),
        host_kv_cache_pool_mapping=torch.zeros(1, 2, dtype=torch.int32),
        num_contexts=0,
        max_num_requests=16,
        helix_position_offsets=None,
        _cute_dsl_mla_split_plan_key=None,
    )
    fwd = SimpleNamespace(
        mla_bmm1_scale=torch.ones(1, device=device),
        mla_bmm2_scale=torch.ones(1, device=device),
        softmax_stats_tensor=None,
        quant_q_buffer=None,
    )
    params = SimpleNamespace(
        attn=attn,
        meta=meta,
        fwd=fwd,
        num_requests=batch,
        tokens_per_block=page,
        sequence_lengths=torch.full((batch,), 40000, dtype=torch.int32, device=device),
        workspace=torch.empty(1 << 28, dtype=torch.uint8, device=device),
    )
    q = torch.zeros(
        batch * seq_len_q, heads * (latent + rope), device=device, dtype=torch.float8_e4m3fn
    )
    output = torch.zeros(batch * seq_len_q, heads * latent, device=device, dtype=torch.bfloat16)
    with _split_workspace_sizing(monkeypatch, fill=fill) as runner:
        monkeypatch.setattr(runner, "_get_max_active_blocks", classmethod(lambda cls: 212))
        fmha = mod.CuteDslMlaFmha.__new__(mod.CuteDslMlaFmha)
        fmha._run_mla_decode(q, output, params, torch.float8_e4m3fn)
    torch.cuda.synchronize()
    assert len(calls) == 1 and calls[0][1] == expected, calls


def test_fill_split_workspace_rows_are_the_bucket_formula(monkeypatch) -> None:
    """The split-KV workspace rows are the maximum of B * s * split over the buckets, to the row.

    The fill switch only raises the split of buckets of at most two query rows to 64.
    """

    def expected_rows(runner, heads, seq_len_q, max_batch, fill):
        rows = acc_rows = 0
        for s in range(1, seq_len_q + 1):
            folded = runner.get_folded_seq_len_q(heads, s)
            per_request = runner.get_split_kv_rows_per_request(heads, s)
            bucket = 1
            while bucket <= max(1, max_batch):
                split = runner.get_default_split_kv(bucket, folded, 212)
                if fill and bucket * folded <= 2:
                    split = max(split, 64)
                if split > 1:
                    rows = max(rows, bucket * s * split)
                    acc_rows = max(acc_rows, bucket * split * per_request)
                bucket *= 2
        return rows, acc_rows

    for fill in (False, True):
        with _split_workspace_sizing(monkeypatch, fill=fill) as runner:
            monkeypatch.setattr(runner, "_get_max_active_blocks", classmethod(lambda cls: 212))
            for heads, seq_len_q, max_batch in (
                (12, 6, 1),
                (12, 6, 16),
                (12, 7, 64),
                (128, 1, 64),
                (128, 7, 1024),
            ):
                assert (
                    runner.get_max_split_kv_rows(heads, seq_len_q, max_batch),
                    runner.get_max_split_kv_acc_rows(heads, seq_len_q, max_batch),
                ) == expected_rows(runner, heads, seq_len_q, max_batch, fill), (
                    fill,
                    heads,
                    seq_len_q,
                    max_batch,
                )


def test_fill_split_workspace_fits_the_wide_plan_for_one_request(monkeypatch) -> None:
    """A runner sized for one request still holds a 64-split plan once the switch is on."""
    import cutlass

    fits = {}
    for fill in (False, True):
        with _split_workspace_sizing(monkeypatch, fill=fill) as runner_cls:
            monkeypatch.setattr(runner_cls, "_get_max_active_blocks", classmethod(lambda cls: 212))
            runner = runner_cls(
                in_dtype=cutlass.Float8E4M3FN,
                num_heads=12,
                seq_len_q=6,
                page_size=64,
                max_batch_size=1,
            )
            fits[fill] = runner.get_fitting_split_kv(1, 1)
    assert fits[False] < 64 <= fits[True]
