# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness of the fifo-v2 Helix all-to-all with all cp ranks on one GPU.

The receiver uses generic-proxy shared-memory stores (protoUnpack) to restore the words that the
LL128 flags displaced. One lane then copies the entry out with cp.async.bulk, an async-proxy read of
the same shared memory. The two need a fence.proxy.async between them. Without it, the copy can pick
up a flag, i.e. the FIFO step, in place of a restored data word. Read as bf16, that step is Inf or
NaN for 1/256 of steps.

In serving the race is rare, about 2e-9 per received entry at cp=16 across NVL72 trays. Two things
make it frequent here, on one GPU:
- The op only needs a workspace that every rank can address, so the cp ranks run as cp concurrent
  streams of one GPU sharing an ordinary device tensor, and their kernels share every SM.
- A kernel resident beside them streams shared-memory stores and loads on every SM. That delays the
  receiver's stores relative to its bulk copy. Without the fence this gives about ten bad entries
  per second on GB300; without this kernel it gives about one per ten seconds.

Every received entry is compared bit for bit with what its sender sent. A mismatch whose differing
words sit at LL128 flag slots and equal the call count is this race.
"""

import ctypes
import os
import time

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import internal as _tllm_internal

try:
    from cuda.bindings import driver as drv
    from cuda.bindings import nvrtc
except ImportError:
    from cuda import cuda as drv
    from cuda import nvrtc

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="exercises the cp.async.bulk receiver of the Helix all-to-all on SM100/SM103",
)

# K3 MLA shapes per (token, peer) entry: 6 local heads x 512 bf16 partial outputs + 6 float2 stats.
HEADS, KV_LORA_RANK = 6, 512
CP_SIZE, NUM_TOKENS = 4, 512
# Long enough that the unfenced kernel fails with near certainty, short enough for a unit test.
DURATION_S = float(os.environ.get("TLLM_HELIX_A2A_STRESS_SECONDS", "10"))
CHECK_EVERY = 8

_PRESSURE_SRC = r"""
extern "C" __global__ void smemPressure(unsigned long long iters, unsigned int words, unsigned int* sink)
{
    extern __shared__ unsigned int buf[];
    volatile unsigned int* vbuf = buf;
    unsigned int acc = threadIdx.x;
    for (unsigned long long i = 0; i < iters; ++i)
    {
        unsigned int const idx = (threadIdx.x + static_cast<unsigned int>(i) * 37u) % words;
        vbuf[idx] = acc;
        acc += vbuf[(idx + 1056u) % words];
    }
    if (acc == 0xdeadbeefu)
    {
        sink[0] = acc;
    }
}
"""


def _check(result):
    err, *values = result if isinstance(result, tuple) else (result,)
    if isinstance(err, nvrtc.nvrtcResult) and err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC error {err}")
    if isinstance(err, drv.CUresult) and err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {err}")
    return values[0] if len(values) == 1 else values


class _SmemPressure:
    """One CTA per SM streaming conflict-free st.shared/ld.shared. It is small enough to stay resident
    beside the all-to-all CTAs of all cp ranks."""

    THREADS, SMEM_BYTES = 512, 32 * 1024

    def __init__(self, device: torch.device):
        major, minor = torch.cuda.get_device_capability(device)
        prog = _check(
            nvrtc.nvrtcCreateProgram(_PRESSURE_SRC.encode(), b"smem_pressure.cu", 0, [], [])
        )
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        if nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            log = b" " * _check(nvrtc.nvrtcGetProgramLogSize(prog))
            _check(nvrtc.nvrtcGetProgramLog(prog, log))
            raise RuntimeError(f"NVRTC compile failed: {log.decode(errors='replace')}")
        cubin = b" " * _check(nvrtc.nvrtcGetCUBINSize(prog))
        _check(nvrtc.nvrtcGetCUBIN(prog, cubin))
        self._module = _check(drv.cuModuleLoadData(cubin))
        self._kernel = _check(drv.cuModuleGetFunction(self._module, b"smemPressure"))
        self._blocks = torch.cuda.get_device_properties(device).multi_processor_count
        self._sink = torch.zeros(1, dtype=torch.int32, device=device)
        self.stream = torch.cuda.Stream(device=device)
        self._iters = 1 << 16
        start = time.monotonic()
        self.launch().synchronize()
        # About half a second per launch from here on.
        self._iters = max(1 << 12, int(self._iters * 0.5 / max(time.monotonic() - start, 1e-3)))

    def launch(self) -> torch.cuda.Event:
        args = (
            (self._iters, self.SMEM_BYTES // 4, self._sink.data_ptr()),
            (ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_void_p),
        )
        _check(
            drv.cuLaunchKernel(
                self._kernel,
                self._blocks,
                1,
                1,
                self.THREADS,
                1,
                1,
                self.SMEM_BYTES,
                drv.CUstream(self.stream.cuda_stream),
                args,
                0,
            )
        )
        done = torch.cuda.Event()
        done.record(self.stream)
        return done


def _entry_lines(o_bytes: int, s_bytes: int) -> int:
    """LL128 lines of one FIFO entry: both fields padded to 16 B, the sum padded to 128 B, plus one
    tail line per 15 data lines for the words the flags displace."""
    data = -(-(-(-o_bytes // 16) * 16 + -(-s_bytes // 16) * 16) // 128)
    return data + -(-data // 15)


def _describe(actual, expected, call_index, channels, entry_lines, limit=4):
    """The first bad entries of one call: their differing u64 words, each with whether it sits at its
    LL128 line's flag slot and whether it equals the call count (the FIFO step)."""
    lines = []
    for rank, (act, exp) in enumerate(zip(actual, expected)):
        diff = act != exp
        for token, src in torch.nonzero(diff.any(-1)).tolist()[:limit]:
            base = (token // channels) * entry_lines
            words = torch.nonzero(diff[token, src]).flatten().tolist()
            info = [
                (u, u % 16 == (base + u // 16) % 16, int(act[token, src, u]) == call_index)
                for u in words[:6]
            ]
            lines.append(
                f"call {call_index} rank {rank} <- src {src} token {token}: {len(words)} bad u64 "
                f"(word, at flag slot, == step): {info}"
            )
    return lines


def _wait(stream: torch.cuda.Stream, timeout_s: float = 60.0) -> None:
    """Wait for the stream's work. Fail instead of hanging if the cp kernels cannot all be resident
    at once, since each one spins until its peers have written."""
    done = torch.cuda.Event()
    done.record(stream)
    start = time.monotonic()
    while not done.query():
        if time.monotonic() - start > timeout_s:
            pytest.fail(f"the all-to-all made no progress for {timeout_s:.0f} s")
        time.sleep(0.005)


@pytest.mark.parametrize("zero_kv_rows", [False, True], ids=["dense", "zero_kv_rows"])
def test_fifo_v2_receives_bit_exact_entries(zero_kv_rows: bool) -> None:
    """With ``zero_kv_rows`` the sender also overwrites masked rows in shared memory with ordinary
    stores before packing them, and its next entry is loaded into the same shared memory with
    cp.async.bulk, which needs the same proxy fence on the sender side."""
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)
    cp, tokens = CP_SIZE, NUM_TOKENS
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    # computeHelixMaxChannelCount; the kernel runs this many channels when shared memory allows
    # min(cp, 8) groups per CTA, which K3 entries do.
    channels = max(sm_count // (2 * -(-cp // 8)), 1)
    per_rank_bytes = _tllm_internal.thop.get_helix_workspace_size_per_rank(cp)
    # The FIFO alone is 512 KiB per (peer, channel). A smaller size means the cached channel count of
    # another cp_size leaked into this one.
    assert per_rank_bytes >= 512 * 1024 * cp * channels
    workspace = torch.empty((cp, per_rank_bytes // 8), dtype=torch.uint64, device=device)
    for rank in range(cp):
        torch.ops.trtllm.initialize_helix_workspace(workspace, rank, cp)

    partial_o = [
        (torch.randn(tokens, cp, HEADS * KV_LORA_RANK, device=device) * 4).to(torch.bfloat16)
        for _ in range(cp)
    ]
    stats = [torch.randn(tokens, cp, HEADS * 2, device=device) * 8 for _ in range(cp)]
    # A sender replaces the rows its mask flags with the no-op contribution (zeros, (-inf, 0)).
    masks = [torch.rand(tokens, device=device) < 0.25 if zero_kv_rows else None for _ in range(cp)]
    sent_o = [o.clone() for o in partial_o]
    sent_stats = [s.clone() for s in stats]
    for o, s, mask in zip(sent_o, sent_stats, masks):
        if mask is not None:
            o[mask] = 0
            pairs = s.view(tokens, cp, HEADS, 2)
            pairs[mask, ..., 0] = float("-inf")
            pairs[mask, ..., 1] = 0
    # Rank r receives entry [t, r] of every sender's tensors, at [t, sender] of its output.
    expected = [
        torch.cat(
            [
                torch.stack([sent_o[src][:, rank] for src in range(cp)], dim=1).view(torch.int64),
                torch.stack([sent_stats[src][:, rank] for src in range(cp)], dim=1).view(
                    torch.int64
                ),
            ],
            dim=-1,
        )
        for rank in range(cp)
    ]
    entry_lines = _entry_lines(HEADS * KV_LORA_RANK * 2, HEADS * 2 * 4)

    pressure = _SmemPressure(device)
    pressure_done = None
    streams = [torch.cuda.Stream(device=device) for _ in range(cp)]
    main = torch.cuda.current_stream(device)
    torch.cuda.synchronize(device)

    calls = bad = 0
    report = []
    deadline = time.monotonic() + DURATION_S
    while time.monotonic() < deadline and not report:
        if pressure_done is None or pressure_done.query():
            pressure_done = pressure.launch()
        received, counts = [], []
        for _ in range(CHECK_EVERY):
            outputs = []
            for rank in range(cp):
                # Also orders this stream's next allocations after the main stream's comparisons.
                streams[rank].wait_stream(main)
                with torch.cuda.stream(streams[rank]):
                    outputs.append(
                        torch.ops.trtllm.alltoall_helix_native(
                            partial_o[rank], stats[rank], workspace, rank, cp, masks[rank]
                        )
                    )
            for rank in range(cp):
                main.wait_stream(streams[rank])
            actual = [
                torch.cat([o.view(torch.int64), s.view(torch.int64)], dim=-1) for o, s in outputs
            ]
            counts.append(
                torch.stack([(a != e).any(-1).sum() for a, e in zip(actual, expected)]).sum()
            )
            received.append(actual)
        _wait(main)
        for offset, count in enumerate(torch.stack(counts).tolist()):
            if count:
                bad += count
                report += _describe(
                    received[offset], expected, calls + offset, channels, entry_lines
                )
        calls += CHECK_EVERY
    torch.cuda.synchronize(device)

    assert bad == 0, (
        f"{bad} received all-to-all entries differ from what was sent within {calls} calls "
        f"(cp={cp}, {tokens} tokens, {channels} channels). Differing words at LL128 flag slots equal "
        "to the call count mean the receiver's cp.async.bulk store read shared memory before "
        "protoUnpack's stores were visible to the async proxy:\n" + "\n".join(report[:12])
    )
