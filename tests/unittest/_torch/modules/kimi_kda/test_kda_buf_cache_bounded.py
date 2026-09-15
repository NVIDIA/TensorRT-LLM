# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA prefill scratch caches must be bounded, and eviction must free.

Why this file exists, and why it is separate from the parity/soundness tests:

``_get_buffers`` is keyed on the packed prefill token count ``T``. In the
executor ``T = attn_metadata.num_ctx_tokens`` -- the sum of this iteration's
context chunk sizes -- which the scheduler does not quantize, so nearly every
iteration presents a shape the cache has never seen. The cache is therefore
LRU-capped, and the ONLY thing making that cap meaningful is that nothing
outside the cache entry references the buffers or their CuTe wrappers.

That invariant silently broke: the six wrappers K123 consumes were cached in
a module-level ``id(tensor)``-keyed dict. A wrapper built by ``from_dlpack``
holds a strong reference to its torch tensor, so each entry was immortal --
weakref pruning can never fire when the cache entry is what keeps the key
object alive. Every distinct ``T`` stranded a full ~1 GiB buffer set at K3's
H=96, K=V=128, max_num_tokens=8192. A 16-rank Kimi-K3 context-only run grew
~1.5 GiB/min per rank and died in the MoE with
``torch.OutOfMemoryError: Tried to allocate 4.26 GiB``, ~40 min in, on a GPU
that startup accounting said had ~80 GiB free. Measured at the wall:
``_get_buffers`` held 53.79 of 61.95 GiB of non-weight memory, in 86 live
blocks per allocation site against a cache capped at 8 -- while the two
buffers in the same tuple that were NOT wrapper-cached (``O_flat``,
``S_out``) sat at exactly 8, the cap, working correctly.

Parity tests cannot see this: every shape produces correct numbers. What
fails is only the lifetime. So the assertions here are about
``torch.cuda.memory_allocated`` and nothing else.

TWO TIERS, and the difference matters
-------------------------------------
``test_prefill_scratch_plateaus_over_distinct_token_counts`` drives the real
op. It is the only test here that REPRODUCES the leak: the pinning happened in
``_launch_fused_k123_inv``, not in ``_get_buffers``, so nothing short of an
actual forward triggers it. Verified against the pre-fix code on Rubin
(job 445071) -- see the note on that test.

Everything else calls ``_get_buffers`` directly. Those are structural guards
for the post-fix layout: cheap, GPU-agnostic, and they pin down "the cache
entry is the sole owner". They deliberately do NOT gate on SM100/SM103/SM107,
because ``_get_buffers`` launches no kernel and so runs on any CUDA GPU with
the CuTe DSL available -- which is where a cheap guard is worth the most.
They do NOT, on their own, prove the leak is gone.
"""

import gc

import pytest
import torch

pytest.importorskip("cutlass")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# Deliberately not K3's H=96/T=8192: at those shapes one entry is ~1 GiB and
# the pre-fix failure mode would OOM the test host rather than fail an
# assertion. The leak is shape-independent, so small shapes prove it too.
H = 4
K_DIM = 128
V_DIM = 128
BT = 64
T0 = 1024
N_SHAPES = 32


def _op_module():
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_custom_ops

    if not cute_dsl_kimi_k3_custom_ops.IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("CuTe DSL not available")
    return cute_dsl_kimi_k3_custom_ops


def _set_bytes(t: int) -> int:
    """Bytes of the six large buffers one varlen ``_get_buffers`` entry owns.

    Mirrors the allocations at the top of ``_get_buffers``: three [B, T+BT,
    H, K] bf16 (k_scaled/kg/q_scaled), two [B, T+BT, H, BT] bf16
    (A_qk/A_kk), one [B, T+BT, H, V] bf16 (O_flat). gk_last_exp / S_out /
    cu_eqlen / co_eqlen are orders of magnitude smaller and are left out of
    the budget on purpose -- the slack below covers them.
    """
    rows = t + BT
    return rows * H * 2 * (3 * K_DIM + 2 * BT + V_DIM)


def _alloc(mod, dev, t):
    return mod._get_buffers(dev, torch.bfloat16, 1, t, H, K_DIM, V_DIM, t // BT, 1, BT, varlen=True)


def _settle():
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def clean_caches():
    mod = _op_module()
    mod._buf_cache.clear()
    mod._g_sentinel_cache.clear()
    mod._padded_input_cache.clear()
    _settle()
    yield mod
    mod._buf_cache.clear()
    mod._g_sentinel_cache.clear()
    mod._padded_input_cache.clear()
    _settle()


def test_buf_cache_eviction_actually_frees(clean_caches):
    """N distinct shapes must leave at most ``_BUF_CACHE_MAX_ENTRIES`` alive.

    Structural guard, NOT a reproduction: this calls ``_get_buffers`` directly,
    and the pre-fix pinning lived in ``_launch_fused_k123_inv``, so this test
    passes against the buggy code too (confirmed on Rubin, job 445071). It
    still earns its place -- it is the cheap, GPU-agnostic statement of the
    invariant, and it fails immediately if someone caches a wrapper inside
    ``_get_buffers`` itself. For the real thing see
    ``test_prefill_scratch_plateaus_over_distinct_token_counts``.
    """
    mod = clean_caches
    dev = torch.device("cuda:0")
    base = torch.cuda.memory_allocated(dev)

    for i in range(N_SHAPES):
        _alloc(mod, dev, T0 + BT * i)

    _settle()
    live = torch.cuda.memory_allocated(dev) - base

    cap = mod._BUF_CACHE_MAX_ENTRIES
    t_max = T0 + BT * (N_SHAPES - 1)
    # 25% slack for the small buffers excluded from _set_bytes plus allocator
    # rounding; the pre-fix failure overshoots by >10x, so the exact slack
    # does not matter.
    budget = int(1.25 * cap * _set_bytes(t_max)) + (8 << 20)

    assert len(mod._buf_cache) <= cap
    assert live <= budget, (
        f"_get_buffers scratch is not bounded by its LRU: {live / 2**20:.0f} MiB live "
        f"after {N_SHAPES} distinct shapes, budget {budget / 2**20:.0f} MiB for {cap} "
        f"entries. Something outside the cache entry is holding evicted buffers alive "
        f"-- check that no module-level dict stores a CuTe wrapper over this scratch."
    )


def test_clearing_buf_cache_frees_everything(clean_caches):
    """The cache must be the sole owner: clearing it returns all the memory.

    This is the invariant the leak violated, stated directly. It is stricter
    than the eviction test and independent of the cap's value.
    """
    mod = clean_caches
    dev = torch.device("cuda:0")
    base = torch.cuda.memory_allocated(dev)

    for i in range(4):
        _alloc(mod, dev, T0 + BT * i)
    assert torch.cuda.memory_allocated(dev) > base  # sanity: we did allocate

    mod._buf_cache.clear()
    _settle()
    leaked = torch.cuda.memory_allocated(dev) - base

    assert leaked < (1 << 20), (
        f"{leaked / 2**20:.1f} MiB survived clearing _buf_cache -- the cache entry is "
        f"not the sole owner of its buffers."
    )


def test_g_sentinel_cache_is_bounded(clean_caches):
    """``real_T`` is in the key, i.e. one entry per distinct prompt length."""
    mod = clean_caches
    dev = torch.device("cuda:0")
    base = torch.cuda.memory_allocated(dev)

    for i in range(N_SHAPES):
        t_padded = T0 + BT * i
        mod._get_g_sentinel_buffer(1, t_padded, H, K_DIM, torch.bfloat16, dev, t_padded - 7)

    _settle()
    live = torch.cuda.memory_allocated(dev) - base

    cap = mod._G_SENTINEL_CACHE_MAX_ENTRIES
    t_max = T0 + BT * (N_SHAPES - 1)
    budget = int(1.25 * cap * t_max * H * K_DIM * 2) + (8 << 20)

    assert len(mod._g_sentinel_cache) <= cap
    assert live <= budget, (
        f"_g_sentinel_cache is unbounded: {live / 2**20:.0f} MiB live after "
        f"{N_SHAPES} distinct (T_padded, real_T) pairs, budget {budget / 2**20:.0f} MiB."
    )


def test_k123_wrappers_live_in_the_cache_entry(clean_caches):
    """Structural guard: the wrappers K123 consumes belong to the entry.

    If a future change moves them back into a module-level dict, the memory
    tests above catch it -- but only on a machine with enough GPU memory to
    run them. This one fails immediately and points at the cause.
    """
    mod = clean_caches
    dev = torch.device("cuda:0")
    wrappers = _alloc(mod, dev, T0)[-1]
    for name in (
        "k123_ks_ct",
        "k123_kg_ct",
        "k123_qs_ct",
        "k123_gk_ct",
        "k123_aqk_ct",
        "k123_akk_ct",
    ):
        assert name in wrappers, f"{name} must be owned by the _get_buffers entry"


# ---------------------------------------------------------------------------
# Tier 2: the actual reproduction. Needs the KDA kernels, so it gates on arch.
# ---------------------------------------------------------------------------

OP_HIDDEN_SIZE = 7168
OP_NUM_HEADS = 96
OP_HEAD_DIM = 128
OP_CONV_KERNEL_SIZE = 4
# 24 shapes x ~160 MiB/set is ~3.8 GiB held by the pre-fix code and ~0.5 GiB
# after the fix. Production T reaches max_num_tokens=8192, where one set is
# ~1 GiB -- 24 of those would OOM a small CI GPU instead of failing an
# assertion, and the leak is independent of T anyway.
OP_T0 = 512
OP_N_SHAPES = 24


def _has_kda_gpu() -> bool:
    # Defer to the predicate rather than mirroring it: it accepts SM100/SM103
    # only, and on SM107 KDAKernelDispatch takes the FLA fallback, which does
    # not exercise the bounded allocation path this test measures.
    if not torch.cuda.is_available():
        return False
    from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import is_kda_optimized_supported

    return is_kda_optimized_supported()


def _op_set_bytes(t: int) -> int:
    """Bytes of the six large buffers, at the op's real head count."""
    rows = t + BT
    return rows * OP_NUM_HEADS * 2 * (3 * OP_HEAD_DIM + 2 * BT + OP_HEAD_DIM)


@pytest.mark.skipif(not _has_kda_gpu(), reason="needs Blackwell SM100/SM103 or Rubin SM107")
@torch.no_grad()
def test_prefill_scratch_plateaus_over_distinct_token_counts(clean_caches):
    """Drive the real op over many distinct token counts; memory must plateau.

    THIS is the regression test for the ctx-worker OOM. It reproduces the
    executor's actual pattern -- one varlen prefill per iteration at a token
    count that changes every time -- and asserts the only thing that was ever
    wrong: how much memory is still live afterwards.

    Pre-fix, each distinct ``T`` stranded a full buffer set, because
    ``_launch_fused_k123_inv`` cached the six scratch wrappers in a
    module-level ``id()``-keyed dict whose entries can never be pruned (the
    cached wrapper holds a strong reference to the very tensor the weakref is
    keyed on). Post-fix those wrappers live in the ``_get_buffers`` entry, so
    LRU eviction actually returns the memory.

    Note the warmup loop below is not politeness: the first call per process
    JITs the CuTe DSL kernels and populates the shape-independent compile
    caches, and that allocation must land before the baseline is taken or it
    shows up as a leak.
    """
    fla = pytest.importorskip("fla")  # noqa: F841  (the mixer imports it)
    from types import SimpleNamespace

    from kimi_kda_test_utils import get_production_prefill_kernel_path

    from tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

    mod = clean_caches
    dev = torch.device("cuda:0")
    torch.manual_seed(0)

    # The mixer takes a config object and a layer index, and reads its shapes
    # out of ``linear_attn_config`` -- same construction as
    # test_kda_prefill_op.py::_make_kda.
    cfg = SimpleNamespace(
        hidden_size=OP_HIDDEN_SIZE,
        rms_norm_eps=1e-5,
        linear_attn_config={
            "num_heads": OP_NUM_HEADS,
            "head_dim": OP_HEAD_DIM,
            "short_conv_kernel_size": OP_CONV_KERNEL_SIZE,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    mixer = KimiKDALinearAttention(cfg, layer_idx=0).to("cuda")
    with torch.no_grad():
        mixer.dt_bias.zero_()
    # Allocates the fused projection buffers. It has to happen here, before the
    # baseline below, or it reads as a leak.
    mixer.finalize_decode_weights()

    prefill_path = get_production_prefill_kernel_path(mixer)
    assert prefill_path == "optimized", f"this test is meaningless on the {prefill_path} path"

    # Pool tensors are shape-independent, so allocate them once: only the
    # per-T scratch should move the measurement.
    batch = 1
    projection_size = OP_NUM_HEADS * OP_HEAD_DIM
    conv_pool = torch.zeros(
        batch, 3 * projection_size, OP_CONV_KERNEL_SIZE - 1, dtype=torch.bfloat16, device=dev
    )
    ssm_pool = torch.zeros(
        batch, OP_NUM_HEADS, OP_HEAD_DIM, OP_HEAD_DIM, dtype=torch.float32, device=dev
    )
    slot_indices = torch.arange(batch, device=dev, dtype=torch.long)
    has_initial_states = torch.zeros(batch, device=dev, dtype=torch.bool)

    def run(total_t: int) -> None:
        cu = torch.tensor([0, total_t], dtype=torch.long, device=dev)
        # forward_prefill takes packed 2-D tokens, not [B, T, H].
        x2d = torch.randn(total_t, OP_HIDDEN_SIZE, dtype=torch.bfloat16, device=dev) * 0.05
        metadata = SimpleNamespace(
            use_initial_states=False,
            has_initial_states=has_initial_states,
            state_indices=slot_indices.to(torch.int32),
            query_start_loc=cu.to(torch.int32),
        )
        out = mixer.forward_prefill(
            x2d,
            cu,
            metadata,
            batch,
            conv_pool,
            ssm_pool,
            slot_indices,
        )
        del out, x2d, cu, metadata

    # Warm: JIT + compile caches + one buffer set, all before the baseline.
    for i in range(2):
        run(OP_T0 + BT * i)
    _settle()
    mod._buf_cache.clear()
    _settle()
    base = torch.cuda.memory_allocated(dev)

    for i in range(OP_N_SHAPES):
        run(OP_T0 + BT * i)
    _settle()
    live = torch.cuda.memory_allocated(dev) - base

    cap = mod._BUF_CACHE_MAX_ENTRIES
    t_max = OP_T0 + BT * (OP_N_SHAPES - 1)
    # Clamp the cap used for budgeting. Scaling the budget with the module's
    # own knob would let someone silence a genuine regression by raising
    # TLLM_KDA_BUF_CACHE_ENTRIES -- and it already cost separation: measured
    # pre-fix on Rubin (job 445326) this test held 3394 MiB, which is 3.1x the
    # cap=2 budget but only 1.34x the cap=8 one. Four sets is generous for a
    # cache whose useful reuse is intra-forward (all 69 KDA layers share one
    # shape); a deliberate larger cap should adjust this test consciously.
    budget = (min(cap, 4) + 2) * _op_set_bytes(t_max) + (128 << 20)

    assert len(mod._buf_cache) <= cap, (
        f"_buf_cache holds {len(mod._buf_cache)} entries, cap is {cap}"
    )
    assert live <= budget, (
        f"KDA prefill scratch grows with the token count instead of plateauing: "
        f"{live / 2**20:.0f} MiB live after {OP_N_SHAPES} distinct T "
        f"({OP_T0}..{t_max}), budget {budget / 2**20:.0f} MiB for {cap} cached "
        f"sets. This is the ctx-worker OOM: a module-level cache is pinning "
        f"buffer sets past their LRU eviction."
    )
