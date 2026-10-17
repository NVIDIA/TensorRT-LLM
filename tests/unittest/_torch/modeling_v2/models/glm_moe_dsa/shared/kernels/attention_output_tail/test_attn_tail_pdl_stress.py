# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PDL boundary stress of the default attention_output_tail engine (``cute_stream``).

The default attention_output_tail is two kernels with programmatic dependent launch (PDL) at
both package boundaries:

* ``_wuv_kernel`` is a PDL dependent of the latent's producer. It triggers its
  own dependent at entry and loads ``w_uv`` before ``griddepcontrol.wait``;
  it reads the latent and writes ``v`` only after the wait.
* ``stream_oproj`` is a PDL dependent of ``_wuv_kernel``. It copies ``w_o``
  rows into its own shared memory before its wait, reads ``v`` and writes the
  partial after it, and triggers its own PDL consumer before it has written
  the partial.

The boundary contract is therefore: before its wait an attention_output_tail kernel touches only
immutable weights and its own shared memory, and every consumer of the
partial waits before reading it. These tests exercise that contract with
synthetic neighbour kernels that stretch the overlap windows:

* **Delayed producer.** A predecessor triggers its dependents at entry, spins
  for ``delay`` µs, then writes the latent. The latent and ``v`` are poisoned
  with NaN before the producer runs. An attention_output_tail read of the latent or of ``v`` that
  is not ordered after the producer (or after ``_wuv_kernel``) shows up as
  NaN or as another iteration's values.
* **Delayed consumer with rapid buffer reuse.** Every kernel of the chain
  producer → attention_output_tail → consumer is PDL-launched and triggers its dependents at
  entry, so the next iteration's producer and attention_output_tail kernels may become resident
  while the current consumer still runs. The consumer waits, spins, then
  copies the partial. attention_output_tail's ``v`` and partial come from the CUDA-graph pool and
  are reused across iterations. An attention_output_tail global write issued before its wait would
  overwrite the partial the sleeping consumer has not read yet.

Every captured iteration has its own latent, and the partial the consumer
copies must equal, bit for bit, an eager attention_output_tail run on that latent
(attention_output_tail is deterministic). The graphs are replayed many times with new latents. The
consumer records ``%globaltimer`` at entry and after its wait, so the tests
can report (and ``test_pdl_overlap_witness`` asserts) that the consumer really
was resident before attention_output_tail finished.

Weights: real GLM-5.2 ``v_b_proj`` / ``o_proj`` TP8 shards when
``GLM52_NVFP4_CKPT`` is set, otherwise synthetic. Latents: real-weight,
synthetic-activation mixtures of real compressed-KV rows when the checkpoint
is available, otherwise random.

``GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_PDL_STRESS_QUICK=1`` shrinks iteration and replay counts
(compute-sanitizer runs); ``GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_PDL_STRESS_SANITIZER=1`` skips the timing
witness, because the sanitizer serializes kernels;
``GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_PDL_STRESS_SYNTHETIC=1`` uses synthetic weights and latents even when
the checkpoint is available.
"""

import os
import statistics
import sys

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.kernels import (  # noqa: E501
    DEFAULT_CONFIG,
    TailConfig,
    attn_tail,
    check_tail_shapes,
    pdl_enabled,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tail_fixtures  # noqa: E402
import tail_reference  # noqa: E402

TP_SIZE = 8
LOCAL_HEADS, V_DIM, C_DIM, HIDDEN = 8, 256, 512, 6144
QUICK = os.environ.get("GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_PDL_STRESS_QUICK") == "1"
UNDER_SANITIZER = os.environ.get("GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_PDL_STRESS_SANITIZER") == "1"
N_ITERS = 3 if QUICK else 16
N_REPLAYS = 2 if QUICK else 24
_CKPT = tail_fixtures.checkpoint_dir()
# GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_PDL_STRESS_SYNTHETIC=1 forces synthetic weights and latents built by
# elementwise kernels only (compute-sanitizer initcheck: the real-weight
# fixture runs a cuBLAS GEMM whose output initcheck cannot see as written).
_HAVE_CKPT = (
    bool(_CKPT and os.path.isdir(_CKPT))
    and os.environ.get("GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_PDL_STRESS_SYNTHETIC") != "1"
)

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="attention_output_tail kernels target SM100/SM103",
    ),
    pytest.mark.skipif(not pdl_enabled(), reason="the stress needs TRTLLM_ENABLE_PDL=1"),
]

# A fully serialized attention_output_tail: no PDL at entry, inside, or at exit.
SERIAL_CONFIG = TailConfig(pdl_entry=False, trigger_dependents=False)

# ---------------------------------------------------------------------------
# Synthetic neighbour kernels.
# ---------------------------------------------------------------------------


@triton.jit
def _clock_fenced():
    # The memory clobber keeps loads and stores from moving across the read.
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,~{memory}", [], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _gdc_wait():
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _gdc_launch_dependents():
    tl.inline_asm_elementwise(
        "griddepcontrol.launch_dependents; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _spin(delay_ns):
    t0 = _clock_fenced()
    t = t0
    while t - t0 < delay_ns:
        t = _clock_fenced()
    return t


@triton.jit
def _delayed_producer(
    src_ptr,
    dst_ptr,
    stamp_ptr,
    numel,
    delay_ns,
    BLOCK: tl.constexpr,
    TRIGGER: tl.constexpr,
    WAIT: tl.constexpr,
):
    """Copy ``src`` to ``dst`` after spinning ``delay_ns``.

    ``TRIGGER`` lets the PDL dependent (attention_output_tail's W_UV kernel) launch at entry;
    ``WAIT`` (when itself PDL-launched) waits for its predecessor before the
    spin. ``stamp[0]`` = time the copy starts.
    """
    pid = tl.program_id(0)
    if TRIGGER:
        _gdc_launch_dependents()
    if WAIT:
        _gdc_wait()
    t = _spin(delay_ns)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)
    if pid == 0:
        tl.store(stamp_ptr, t)


@triton.jit
def _delayed_consumer(
    src_ptr, dst_ptr, stamp_ptr, numel, delay_ns, BLOCK: tl.constexpr, TRIGGER: tl.constexpr
):
    """PDL consumer of the attention_output_tail partial: wait, spin ``delay_ns``, copy.

    ``stamp[0]`` = entry time (before the wait), ``stamp[1]`` = time the
    wait returned, ``stamp[2]`` = time the copy starts.
    """
    pid = tl.program_id(0)
    t_entry = _clock_fenced()
    if TRIGGER:
        _gdc_launch_dependents()
    _gdc_wait()
    t_wait = _clock_fenced()
    t_copy = _spin(delay_ns)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)
    if pid == 0:
        tl.store(stamp_ptr, t_entry)
        tl.store(stamp_ptr + 1, t_wait)
        tl.store(stamp_ptr + 2, t_copy)


_BLOCK = 1024


def launch_producer(src, dst, stamp, delay_us, trigger, wait, pdl):
    numel = dst.numel()
    _delayed_producer[(triton.cdiv(numel, _BLOCK),)](
        src,
        dst,
        stamp,
        numel,
        int(delay_us * 1000),
        BLOCK=_BLOCK,
        TRIGGER=trigger,
        WAIT=wait,
        num_warps=4,
        **({"launch_pdl": True} if pdl else {}),
    )


def launch_consumer(src, dst, stamp, delay_us, trigger):
    numel = src.numel()
    _delayed_consumer[(triton.cdiv(numel, _BLOCK),)](
        src,
        dst,
        stamp,
        numel,
        int(delay_us * 1000),
        BLOCK=_BLOCK,
        TRIGGER=trigger,
        num_warps=4,
        launch_pdl=True,
    )


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

_CACHE = {}


def tail_weights(layer=3, rank=0):
    key = ("w", layer, rank)
    if key not in _CACHE:
        if _HAVE_CKPT:
            w_uv, w_o = tail_fixtures.load_tail_weights(_CKPT, layer, [rank], TP_SIZE)[rank]
            _CACHE[key] = (w_uv.cuda(), w_o.cuda(), "real")
        else:
            gen = torch.Generator(device="cuda").manual_seed(layer * 8 + rank)
            w_uv = (
                torch.randn(LOCAL_HEADS, V_DIM, C_DIM, generator=gen, device="cuda") * 0.02
            ).bfloat16()
            w_o = (
                torch.randn(HIDDEN, LOCAL_HEADS * V_DIM, generator=gen, device="cuda") * 0.02
            ).bfloat16()
            _CACHE[key] = (w_uv, w_o, "synthetic")
    return _CACHE[key]


def make_latents(count, rows, seed, layer=3):
    """``[count, rows, 8, 512]`` BF16 latents, one per captured iteration."""
    if _HAVE_CKPT:
        key = ("kv", layer)
        if key not in _CACHE:
            _CACHE[key] = tail_fixtures.compressed_kv_rows(_CKPT, layer)
        return torch.stack(
            [
                tail_fixtures.mix_latent(
                    _CACHE[key], rows, num_local_heads=LOCAL_HEADS, seed=seed + i
                )
                for i in range(count)
            ]
        )
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((count, rows, LOCAL_HEADS, C_DIM), generator=gen, device="cuda").bfloat16()


def eager_expected(latents, w_uv, w_o, rows, out_rows, config):
    """attention_output_tail on each latent, one isolated call at a time."""
    outs = []
    for i in range(latents.shape[0]):
        lat = latents[i].clone()
        torch.cuda.synchronize()
        outs.append(attn_tail(lat, w_uv, w_o, rows, out_rows=out_rows, config=config).clone())
        torch.cuda.synchronize()
    return torch.stack(outs)


def check_against_native(latents, expected, w_uv, w_o, rows, label):
    """Frozen attention_output_tail budget (test_attn_tail_attn_tail) on the first latent."""
    native = tail_reference.native_attn_tail(latents[0], w_uv, w_o, rows)
    ref, _ = tail_reference.oracle_attn_tail(latents[0], w_uv, w_o, rows)
    e_attn_tail = tail_reference.error_stats(expected[0][:rows], ref)
    e_nat = tail_reference.error_stats(native, ref)
    assert e_attn_tail["rel_rms"] <= 1.25 * e_nat["rel_rms"] + 1e-6, (label, e_attn_tail, e_nat)
    assert e_attn_tail["max_abs"] <= (2 * e_nat["max_abs"] + 2**-8 * e_nat["ref_max"]), (
        label,
        e_attn_tail,
        e_nat,
    )


def assert_bitwise(caps, expected, label):
    assert not bool(torch.isnan(caps).any()), (label, "NaN in a partial")
    bad = [i for i in range(caps.shape[0]) if not torch.equal(caps[i], expected[i])]
    if bad:
        i = bad[0]
        diff = (caps[i].float() - expected[i].float()).abs()
        pytest.fail(
            f"{label}: {len(bad)}/{caps.shape[0]} iterations differ "
            f"from the eager attention_output_tail result (first {i}: "
            f"{int((diff > 0).sum())} elements, max |diff| "
            f"{diff.max().item():.3e})"
        )


def _median_us(values):
    return statistics.median(values) / 1000.0 if values else float("nan")


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows,out_rows", [(6, 6), (6, 8), (1, 1), (8, 8)])
def test_eager_exact_buffers(rows, out_rows):
    """Exactly sized latent / v / partial (memcheck with the caching
    allocator off sees every out-of-bounds access), default config vs the
    fully serialized config, bit for bit, and the frozen budget vs native."""
    w_uv, w_o, kind = tail_weights()
    lat = make_latents(1, rows, seed=11)[0].contiguous()
    assert check_tail_shapes(lat, w_uv, w_o, rows, out_rows) is None
    v = torch.empty((rows, LOCAL_HEADS * V_DIM), dtype=torch.bfloat16, device="cuda")
    out = attn_tail(lat, w_uv, w_o, rows, out_rows=out_rows, v_scratch=v)
    serial = attn_tail(lat, w_uv, w_o, rows, out_rows=out_rows, use_pdl=False, config=SERIAL_CONFIG)
    torch.cuda.synchronize()
    assert torch.equal(out, serial), "PDL and serialized attention_output_tail differ"
    assert bool((out[rows:] == 0).all())
    check_against_native(lat[None], out[None], w_uv, w_o, rows, ("eager", kind, rows, out_rows))
    _, v_nat = tail_reference.native_attn_tail(lat, w_uv, w_o, rows, return_v=True)
    print(
        f"[attention_output_tail-pdl-stress-eager] weights={kind} rows={rows} out_rows={out_rows} "
        f"pdl_equals_serial=True v_equals_native={torch.equal(v, v_nat)}"
    )


def _run_delayed_producer(rows, out_rows, delay_us, config):
    w_uv, w_o, kind = tail_weights()
    n = LOCAL_HEADS * V_DIM
    latent = torch.empty((rows, LOCAL_HEADS, C_DIM), dtype=torch.bfloat16, device="cuda")
    v = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
    srcs = make_latents(N_ITERS, rows, seed=1000)
    caps = torch.empty((N_ITERS, out_rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    stamps = torch.zeros((N_ITERS, 4), dtype=torch.int64, device="cuda")

    def body():
        for i in range(N_ITERS):
            # Poison (ordinary, fully serialized kernels).
            latent.fill_(float("nan"))
            v.fill_(float("nan"))
            # Not PDL-launched itself, but lets attention_output_tail's W_UV launch at its entry.
            launch_producer(srcs[i], latent, stamps[i], delay_us, True, False, False)
            out = attn_tail(latent, w_uv, w_o, rows, out_rows=out_rows, v_scratch=v, config=config)
            launch_consumer(out, caps[i], stamps[i, 1:], 0, False)

    body()  # eager pass: JIT compile + a first stressed run
    torch.cuda.synchronize()
    expected = eager_expected(srcs, w_uv, w_o, rows, out_rows, config)
    assert_bitwise(caps, expected, "delayed-producer eager")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    gaps = []
    for rep in range(N_REPLAYS):
        srcs.copy_(make_latents(N_ITERS, rows, seed=2000 + 100 * rep))
        caps.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected = eager_expected(srcs, w_uv, w_o, rows, out_rows, config)
        assert_bitwise(caps, expected, f"delayed-producer replay {rep}")
        st = stamps.cpu()
        # Producer copy start -> consumer wait return: attention_output_tail's critical path
        # after the latent becomes available (plus the copy itself).
        gaps += [int(st[i, 2] - st[i, 0]) for i in range(N_ITERS)]
    check_against_native(srcs, expected, w_uv, w_o, rows, ("delayed-producer", kind))
    del graph
    return kind, gaps


@pytest.mark.parametrize("rows,out_rows", [(6, 6), (6, 8), (1, 1), (8, 8)])
@pytest.mark.parametrize("delay_us", [0, 3, 20, 50])
@pytest.mark.parametrize("entry", ["pdl_entry", "no_entry"])
def test_delayed_producer(rows, out_rows, delay_us, entry):
    config = DEFAULT_CONFIG if entry == "pdl_entry" else TailConfig(pdl_entry=False)
    kind, gaps = _run_delayed_producer(rows, out_rows, delay_us, config)
    print(
        f"[attention_output_tail-pdl-stress-producer] weights={kind} rows={rows} "
        f"out_rows={out_rows} delay_us={delay_us} entry={entry} "
        f"iterations={len(gaps)} bitwise_ok=True "
        f"producer_copy_to_consumer_wait_us_median={_median_us(gaps):.2f}"
    )


def _run_delayed_consumer(rows, out_rows, producer_us, consumer_us):
    w_uv, w_o, kind = tail_weights()
    latent = torch.empty((rows, LOCAL_HEADS, C_DIM), dtype=torch.bfloat16, device="cuda")
    srcs = make_latents(N_ITERS, rows, seed=3000)
    caps = torch.empty((N_ITERS, out_rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    stamps = torch.zeros((N_ITERS, 4), dtype=torch.int64, device="cuda")

    def body():
        out_ptrs, v_ptrs = [], []
        for i in range(N_ITERS):
            # PDL-launched and triggering at entry: the chain may run ahead
            # into this iteration while the previous consumer still spins.
            launch_producer(srcs[i], latent, stamps[i], producer_us, True, True, True)
            v = torch.empty((rows, LOCAL_HEADS * V_DIM), dtype=torch.bfloat16, device="cuda")
            out = attn_tail(latent, w_uv, w_o, rows, out_rows=out_rows, v_scratch=v)
            out_ptrs.append(out.data_ptr())
            v_ptrs.append(v.data_ptr())
            launch_consumer(out, caps[i], stamps[i, 1:], consumer_us, True)
            del out, v  # back to the pool: the next iteration reuses them
        return out_ptrs, v_ptrs

    body()
    torch.cuda.synchronize()
    expected = eager_expected(srcs, w_uv, w_o, rows, out_rows, DEFAULT_CONFIG)
    assert_bitwise(caps, expected, "delayed-consumer eager")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_ptrs, v_ptrs = body()
    early = []
    for rep in range(N_REPLAYS):
        srcs.copy_(make_latents(N_ITERS, rows, seed=4000 + 100 * rep))
        caps.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected = eager_expected(srcs, w_uv, w_o, rows, out_rows, DEFAULT_CONFIG)
        assert_bitwise(caps, expected, f"delayed-consumer replay {rep}")
        st = stamps.cpu()
        # Consumer resident before attention_output_tail finished: time spent in its wait.
        early += [int(st[i, 2] - st[i, 1]) for i in range(N_ITERS)]
    check_against_native(srcs, expected, w_uv, w_o, rows, ("delayed-consumer", kind))
    del graph
    reuse = {
        "distinct_out_ptrs": len(set(out_ptrs)),
        "distinct_v_ptrs": len(set(v_ptrs)),
        "v_overlaps_a_partial": any(
            any(o <= p < o + out_rows * HIDDEN * 2 for o in out_ptrs) for p in v_ptrs
        ),
    }
    return kind, early, reuse


@pytest.mark.parametrize("rows,out_rows", [(6, 6), (6, 8), (1, 1)])
@pytest.mark.parametrize("producer_us,consumer_us", [(0, 0), (0, 20), (20, 20), (5, 50)])
def test_delayed_consumer_buffer_reuse(rows, out_rows, producer_us, consumer_us):
    kind, early, reuse = _run_delayed_consumer(rows, out_rows, producer_us, consumer_us)
    # The pool hands the same partial buffer to consecutive iterations, so
    # the stress covers a reused output buffer.
    assert reuse["distinct_out_ptrs"] < N_ITERS, reuse
    frac = sum(e > 1000 for e in early) / max(len(early), 1)
    print(
        f"[attention_output_tail-pdl-stress-consumer] weights={kind} rows={rows} "
        f"out_rows={out_rows} producer_us={producer_us} "
        f"consumer_us={consumer_us} iterations={len(early)} bitwise_ok=True "
        f"reuse={reuse} consumer_wait_us_median={_median_us(early):.2f} "
        f"fraction_waited_over_1us={frac:.2f}"
    )


@pytest.mark.skipif(UNDER_SANITIZER, reason="the sanitizer serializes kernel launches")
def test_pdl_overlap_witness():
    """The stress is not vacuous: the PDL consumer of attention_output_tail is resident and
    waiting before attention_output_tail has finished (attention_output_tail triggers its dependents
    early), and attention_output_tail's PDL entry shortens the producer → consumer path (W_UV
    launched under the producer)."""
    _, early, _ = _run_delayed_consumer(6, 6, 0, 20)
    frac = sum(e > 1000 for e in early) / len(early)
    _, gaps_entry = _run_delayed_producer(6, 6, 20, DEFAULT_CONFIG)
    _, gaps_noentry = _run_delayed_producer(6, 6, 20, TailConfig(pdl_entry=False))
    print(
        f"[attention_output_tail-pdl-stress-witness] consumer_wait_us_median={_median_us(early):.2f} "
        f"fraction_waited_over_1us={frac:.2f} "
        f"producer_to_consumer_us_median pdl_entry="
        f"{_median_us(gaps_entry):.2f} no_entry={_median_us(gaps_noentry):.2f}"
    )
    assert frac >= 0.5, frac
    assert _median_us(gaps_entry) < _median_us(gaps_noentry)
