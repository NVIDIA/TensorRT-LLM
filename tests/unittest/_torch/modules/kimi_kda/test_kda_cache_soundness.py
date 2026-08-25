# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-integration soundness tests for the optimized KDA prefill op.

Covers the two classes of bug that op-math parity tests cannot see:

1. STREAM soundness. The executor runs the model on a dedicated non-blocking
   ``torch.cuda.Stream`` (``py_executor.execution_stream``); the CuTe DSL
   kernels must launch on torch's current stream, not the DSL default stream.
   A default-stream launch races with the projections producing q/k/v/g/beta
   and with consumers of the output — silent, intermittent corruption e2e
   (GSM8K 68.99 vs 97.01 on the FLA control) while single-stream unit tests
   all pass. ``test_nondefault_stream_parity`` reproduces this
   deterministically by delaying the execution stream with ``_sleep`` so any
   kernel launched on the default stream reads not-yet-written inputs.

2. id()-keyed cache soundness under the runtime's fresh-tensor-per-call
   pattern. CPython recycles an object's address (= its id) as soon as it is
   freed; the ``_prune_on_gc`` weakref guard must pop the cache entry at
   dealloc, before a new tensor can alias the key. The recycled-id tests
   engineer exactly that aliasing: cache a verdict under id(t1), free t1,
   re-allocate until a new tensor lands on the same id, refill it with data
   for which the stale verdict would be WRONG, and parity-check against FLA.
   ``test_single_seq_same_object_repeated`` additionally covers the
   Phase 2.1 poisoning case (same cu_seqlens object, two calls).
"""

import gc

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import KDAKernelDispatch  # noqa: E402

NUM_HEADS = 96
HEAD_K_DIM = 128
LOWER_BOUND = -5.0


def _has_supported_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in {(10, 0), (10, 3)}


pytestmark = pytest.mark.skipif(
    not _has_supported_gpu(),
    reason="Kimi K3 is supported only on Blackwell (SM100/SM103)",
)


def _op_module():
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_custom_ops

    return cute_dsl_kimi_k3_custom_ops


@pytest.fixture(scope="module")
def dispatch_pair():
    optimized = KDAKernelDispatch(use_optimized_prefill=True, use_optimized_decode=False)
    assert optimized.prefill_kernel_path == "optimized"
    reference = KDAKernelDispatch(use_optimized_prefill=False, use_optimized_decode=False)
    assert reference.prefill_kernel_path == "fla"
    return optimized, reference


@pytest.fixture(scope="module")
def gate_params():
    torch.manual_seed(0)
    a_log = torch.randn(NUM_HEADS, dtype=torch.float32, device="cuda") * 0.5
    dt_bias = torch.randn(NUM_HEADS * HEAD_K_DIM, dtype=torch.float32, device="cuda") * 0.1
    return a_log, dt_bias


def _make_inputs(total_t: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h, k = NUM_HEADS, HEAD_K_DIM

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cuda").to(dtype)

    q = rnd(1, total_t, h, k)
    key = rnd(1, total_t, h, k)
    v = rnd(1, total_t, h, k)
    g = rnd(1, total_t, h, k)
    beta = rnd(1, total_t, h, dtype=torch.float32)
    return q, key, v, g, beta


def _run(dispatch, gate_params, q, k, v, g, beta, cu):
    """No .clone() on cu — these tests exercise object identity on purpose."""
    a_log, dt_bias = gate_params
    return dispatch.prefill_chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=a_log,
        dt_bias=dt_bias,
        scale=HEAD_K_DIM**-0.5,
        initial_state=None,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu,
    )


def _assert_close(name, actual, expected):
    actual, expected = actual.float(), expected.float()
    cos = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    rel = ((actual - expected).norm() / (expected.norm() + 1e-12)).item()
    assert cos > 0.999 and rel < 3e-2, f"{name}: cos={cos:.6f} rel_l2={rel:.3e}"


def _make_cu(lens):
    return torch.tensor(
        [0] + torch.cumsum(torch.tensor(lens), 0).tolist(), dtype=torch.long, device="cuda"
    )


def _flush_tensor_cache_pins():
    """Evict external pins on cu_seqlens tensors.

    Both the dispatcher (fla.ops.utils.index) and the op
    (tensorrt_llm._torch.modules.fla.index) run cu_seqlens through
    ``@tensor_cache`` helpers that keep the 4 most recent (args, result)
    tuples alive. Until those entries are evicted, a cu_seqlens object
    stays pinned after the call — which is SOUND (the id cannot be
    recycled while our id-keyed entry exists) but makes the recycled-id
    scenario unreachable. Churn the caches with fresh dummy tensors so the
    pins drop and the finalizer-prune path can be exercised.
    """
    from fla.ops.utils.index import prepare_chunk_indices as fla_pci
    from fla.ops.utils.index import prepare_chunk_offsets as fla_pco

    from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices as intree_pci

    for _ in range(5):
        dummy = torch.tensor([0, 64], dtype=torch.long, device="cuda")
        fla_pci(dummy, 64)
        fla_pco(dummy, 64)
        intree_pci(dummy, 64)


def _release_cu(mod, cu_holder):
    """Free the cu_seqlens tensor held in the one-element list ``cu_holder``
    and return (old_id, pruned). ``pruned`` False means something still pins
    the tensor — sound, but the recycled-id scenario is unreachable; the
    caller must then prove the id cannot be recycled and skip.
    """
    old_id = id(cu_holder[0])
    cu_holder.clear()
    _flush_tensor_cache_pins()
    gc.collect()
    return old_id, old_id not in mod._varlen_pure_cache


def _alloc_with_recycled_id(target_id, make, attempts=512):
    """Allocate via ``make()`` until an object lands on ``target_id``.

    Wrong-address candidates are kept alive so the allocator cannot hand the
    same wrong slot back; pymalloc free-lists normally return the freed
    address on the first attempt. Returns None if unattainable.
    """
    hold = []
    for _ in range(attempts):
        cand = make()
        if id(cand) == target_id:
            return cand
        hold.append(cand)
    return None


# ---------------------------------------------------------------------------
# 1. Stream soundness — deterministic repro of the e2e accuracy regression.
# ---------------------------------------------------------------------------


def _make_eqlen_inputs(batch, t, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h, k = NUM_HEADS, HEAD_K_DIM

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cuda").to(dtype)

    return (
        rnd(batch, t, h, k),
        rnd(batch, t, h, k),
        rnd(batch, t, h, k),
        rnd(batch, t, h, k),
        rnd(batch, t, h, dtype=torch.float32),
    )


@pytest.mark.parametrize("regime", ["eqlen", "varlen"])
@torch.no_grad()
def test_nondefault_stream_parity(dispatch_pair, gate_params, regime):
    """Run the optimized prefill on a fresh non-default stream (like the
    executor's ``execution_stream``) whose queue is held back by a GPU sleep,
    with the inputs produced BEHIND that sleep, and read the outputs on that
    same stream right after the call. A kernel the op launches on the default
    stream instead of the current stream reads inputs before they exist
    (eqlen regime — the op has no host sync there, so the repro is
    deterministic) and/or its output is consumed before it is written
    (varlen regime — the ``out * 1.0`` capture below races a stray
    default-stream K4)."""
    optimized, reference = dispatch_pair
    if regime == "eqlen":
        lens = None
        make = lambda seed: _make_eqlen_inputs(2, 256, seed)  # noqa: E731
    else:
        lens = [100, 257, 300]  # multi-seq masked path, like eval traffic
        make = lambda seed: _make_inputs(sum(lens), seed)  # noqa: E731

    q, k, v, g, beta = make(31337)
    cu = _make_cu(lens) if lens else None
    out_ref, state_ref = _run(reference, gate_params, q, k, v, g, beta, cu)
    # Warmup compile of the optimized variants on the default stream so the
    # streamed run below launches immediately (a ~100 s JIT inside the
    # streamed region would let the sleep expire and hide the race).
    _run(optimized, gate_params, q, k, v, g, beta, _make_cu(lens) if lens else None)
    torch.cuda.synchronize()

    exec_stream = torch.cuda.Stream()  # non-blocking, like the executor's
    with torch.cuda.stream(exec_stream):
        torch.cuda._sleep(1 << 28)  # hold the stream back ~100 ms
        # Produced on exec_stream, pending behind the sleep: a default-stream
        # launch would read these buffers before they are written.
        q2, k2, v2 = q * 1.0, k * 1.0, v * 1.0
        g2, beta2 = g * 1.0, beta * 1.0
        out_opt, state_opt = _run(
            optimized, gate_params, q2, k2, v2, g2, beta2, _make_cu(lens) if lens else None
        )
        # Consume on the execution stream immediately, exactly like the
        # runtime's output-gate matmul and ssm_pool.index_copy_ do.
        out_opt = out_opt * 1.0
        state_opt = state_opt * 1.0
    torch.cuda.synchronize()

    _assert_close(f"stream_{regime}/out", out_opt, out_ref)
    _assert_close(f"stream_{regime}/state", state_opt, state_ref)


# ---------------------------------------------------------------------------
# 2. id()-keyed cache soundness under recycled ids / reused objects.
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_recycled_id_varlen_pure_cache(dispatch_pair, gate_params):
    """Aligned lens cache VARLEN_PURE=True under id(cu). Free cu, land a new
    cu_seqlens on the recycled id with NON-aligned lens: a stale hit would
    run the mask-free compile variant on partial chunks."""
    optimized, reference = dispatch_pair
    mod = _op_module()

    aligned = [128, 256, 192]
    cu_holder = [_make_cu(aligned)]
    q, k, v, g, beta = _make_inputs(sum(aligned), seed=11)
    _run(optimized, gate_params, q, k, v, g, beta, cu_holder[0])
    torch.cuda.synchronize()
    assert mod._varlen_pure_cache.get(id(cu_holder[0])) is True, (
        "test setup: expected VARLEN_PURE=True cached under id(cu1)"
    )

    old_id, pruned = _release_cu(mod, cu_holder)
    nonaligned = [100, 257, 219]  # same n_seqs so the entry shape matches
    if not pruned:
        # Entry outlived our release: something still pins the tensor. That
        # is sound ONLY if no new tensor can land on its id.
        clash = _alloc_with_recycled_id(old_id, lambda: _make_cu(nonaligned), attempts=64)
        assert clash is None, (
            "UNSOUND: id recycled while a stale _varlen_pure_cache entry for it is still present"
        )
        pytest.skip("cu_seqlens still pinned externally; recycled-id scenario unreachable this run")

    cu2 = _alloc_with_recycled_id(old_id, lambda: _make_cu(nonaligned))
    if cu2 is None:
        pytest.skip("could not obtain a recycled id for cu_seqlens")
    q, k, v, g, beta = _make_inputs(sum(nonaligned), seed=12)
    out_opt, state_opt = _run(optimized, gate_params, q, k, v, g, beta, cu2)
    out_ref, state_ref = _run(reference, gate_params, q, k, v, g, beta, _make_cu(nonaligned))
    _assert_close("recycled_pure/out", out_opt, out_ref)
    _assert_close("recycled_pure/state", state_opt, state_ref)
    assert mod._varlen_pure_cache.get(id(cu2)) is False


@torch.no_grad()
def test_recycled_id_single_seq_caches(dispatch_pair, gate_params):
    """Phase 2.1 caches: a single ALIGNED seq caches (pure=True, seqlen=256)
    under id(cu). Recycle the id with a single NON-aligned seq: a stale hit
    would skip the g sentinel pad / use the wrong single-seq padding."""
    optimized, reference = dispatch_pair
    mod = _op_module()

    cu_holder = [_make_cu([256])]
    q, k, v, g, beta = _make_inputs(256, seed=21)
    _run(optimized, gate_params, q, k, v, g, beta, cu_holder[0])
    torch.cuda.synchronize()
    assert mod._varlen_pure_cache.get(id(cu_holder[0])) is True

    old_id, pruned = _release_cu(mod, cu_holder)
    if not pruned:
        clash = _alloc_with_recycled_id(old_id, lambda: _make_cu([300]), attempts=64)
        assert clash is None, (
            "UNSOUND: id recycled while stale Phase 2.1 cache entries for it are still present"
        )
        pytest.skip("cu_seqlens still pinned externally; recycled-id scenario unreachable this run")
    assert old_id not in mod._varlen_single_seqlen_cache

    cu2 = _alloc_with_recycled_id(old_id, lambda: _make_cu([300]))
    if cu2 is None:
        pytest.skip("could not obtain a recycled id for cu_seqlens")
    q, k, v, g, beta = _make_inputs(300, seed=22)
    out_opt, state_opt = _run(optimized, gate_params, q, k, v, g, beta, cu2)
    out_ref, state_ref = _run(reference, gate_params, q, k, v, g, beta, _make_cu([300]))
    _assert_close("recycled_single/out", out_opt, out_ref)
    _assert_close("recycled_single/state", state_opt, state_ref)


@torch.no_grad()
def test_single_seq_same_object_repeated(dispatch_pair, gate_params):
    """Two calls with the SAME cu_seqlens object and a non-aligned single
    seq. The Phase 2.1 path must sentinel-pad g on BOTH calls; poisoning
    _varlen_pure_cache with the per-call override made the second call skip
    the pad while still compiling the mask-free variant."""
    optimized, reference = dispatch_pair
    cu = _make_cu([300])
    for i in range(2):
        q, k, v, g, beta = _make_inputs(300, seed=40 + i)
        out_opt, state_opt = _run(optimized, gate_params, q, k, v, g, beta, cu)
        out_ref, state_ref = _run(reference, gate_params, q, k, v, g, beta, _make_cu([300]))
        _assert_close(f"sameobj_call{i}/out", out_opt, out_ref)
        _assert_close(f"sameobj_call{i}/state", state_opt, state_ref)


@torch.no_grad()
def test_recycled_id_input_wrap_cache():
    """_ct_cached wrappers are keyed by (id(tensor), etype); the wrapper pins
    the tensor's storage, so entries for live tensors are immortal by
    design. Soundness requires the entry to be popped when the keyed tensor
    dies before its id can be recycled — otherwise a new tensor on the
    recycled id would silently reuse a wrapper over freed storage."""
    mod = _op_module()
    import cutlass

    t1 = torch.zeros(64, 64, dtype=torch.bfloat16, device="cuda")
    w1 = mod._ct_cached(t1, cutlass.BFloat16)
    assert mod._ct_cached(t1, cutlass.BFloat16) is w1  # cache hit path
    old_id = id(t1)
    key = (old_id, cutlass.BFloat16)
    assert key in mod._input_wrap_cache
    del t1
    gc.collect()

    if key in mod._input_wrap_cache:
        # The wrapper (still referenced by the cache) pins the tensor object,
        # so the entry survives — sound only if the id can NOT be recycled
        # while the entry is present. Prove no new tensor lands on old_id.
        t2 = _alloc_with_recycled_id(
            old_id,
            lambda: torch.zeros(64, 64, dtype=torch.bfloat16, device="cuda"),
            attempts=64,
        )
        assert t2 is None, (
            "UNSOUND: id recycled while a stale _input_wrap_cache entry for "
            "it is still present — a new tensor would reuse a wrapper over "
            "another tensor's storage"
        )
        # Dropping the last wrapper ref must let the finalizer prune.
        del w1
        mod._input_wrap_cache.pop(key, None)
        gc.collect()
    else:
        # Finalizer fired at t1's dealloc — entry pruned before any recycle.
        del w1

    # Fresh tensor (recycled id or not) must get a fresh wrapper.
    t3 = torch.zeros(64, 64, dtype=torch.bfloat16, device="cuda")
    w3 = mod._ct_cached(t3, cutlass.BFloat16)
    assert mod._input_wrap_cache[(id(t3), cutlass.BFloat16)] is w3
