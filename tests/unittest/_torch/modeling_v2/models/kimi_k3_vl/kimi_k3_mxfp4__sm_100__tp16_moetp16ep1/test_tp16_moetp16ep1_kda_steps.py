# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp16ep1 target's KDA layer on step sequences, on one GPU and a real KimiK3HybridCacheManager at one
TP16 rank's shapes (``_kimi_k3_kda_steps``).

* The manager this target runs with (no speculation): fresh prefill, chunk continuation mixed with a decode, a decode
  on the decode kernels (k3_kda_decode_attn) and one on the generic path (kda_decode). Every step run twice from the
  same pools gives the same bits. This target serves no speculative decoding, so no step verifies.
* The decode kernels and the generic decode on the same step and pools agree within the ops' bands.
* The generic decode of more than 8 requests, after a prefill batch with a long sequence.
* A mixed step inside a breakable CUDA graph (the core as one eager break), replayed, gives the eager step's bits.
"""

import weakref

import pytest
import torch

__extra_import_path__ = [".."]
import _kimi_k3_kda_steps as ks  # noqa: E402
from _kimi_k3_parity import assert_bitwise, assert_within  # noqa: E402

pytestmark = pytest.mark.skipif(not ks.sm100(), reason="the KDA kernels need SM100 or SM103")

TARGET = "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1"
#: The decode kernels against the generic decode: each is within 2e-2 (outputs) / 1e-3 (states) of the fp32
#: reference (the kda_decode and k3_kda_decode_attn entries' bands), so within twice that of each other.
TOL_OUT, TOL_STATE = 4e-2, 2e-3


@pytest.fixture(scope="module")
def buffers():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.k3_kda_buffers import (
        K3KdaBuffers,
    )

    return K3KdaBuffers.create(torch.device("cuda", torch.cuda.current_device()))


def _run(module, metadata, hidden, step_kind: str = "branch"):
    """How the model calls the layer on a step: ``forward`` with the step's ``decode_step`` classification on the
    decode branch, ``forward_generic`` where ``decode_step`` gives None or ``step_kind`` is "generic"."""
    step = ks.decode_step_of(TARGET, metadata)
    if step_kind == "generic" or step is None:
        return lambda: module.forward_generic(hidden, metadata)
    return lambda: module.forward(hidden, metadata, step)


def _step(mgr, module, metadata, seed: int, what: str, step_kind: str = "branch"):
    hidden = ks.hidden_for(metadata, seed)
    return ks.run_twice(mgr, _run(module, metadata, hidden, step_kind), what)


def test_step_sequence(buffers):
    mgr = ks.build_manager("plain")
    try:
        module = ks.build_module(TARGET, k3_buffers=buffers, aux_stream=torch.cuda.Stream())
        slots = ks.request_slots(mgr, 3, first_id=10)
        ks.fill_pools(mgr, seed=1)
        _step(
            mgr, module, ks.make_step(mgr, slots, [37, 80, 5], [False] * 3, []), 2, "fresh prefill"
        )
        ks.committed_window_follows(mgr, slots, "fresh prefill")
        _step(
            mgr,
            module,
            ks.make_step(mgr, slots[:2], [50, 20], [True, True], slots[2:]),
            3,
            "continuation + decode",
        )
        ks.committed_window_follows(mgr, slots, "continuation + decode")
        decode = ks.make_step(mgr, [], [], [], slots)
        assert ks.decode_step_of(TARGET, decode).tokens_per_request == 1
        _step(mgr, module, decode, 4, "decode (k3_kda_decode_attn)")
        ks.committed_window_follows(mgr, slots, "decode (k3_kda_decode_attn)")
        _step(mgr, module, decode, 5, "decode (kda_decode)", step_kind="generic")
        ks.committed_window_follows(mgr, slots, "decode (kda_decode)")
    finally:
        mgr.shutdown()


def test_decode_kernels_match_generic_decode(buffers):
    """One decode step of the same requests from the same pools on k3_kda_decode_attn (the decode branch) and on
    kda_decode (the generic path): outputs and the advanced states within the ops' bands, the conv windows (raw
    inputs of the same projection rows rounded to bf16) within one bf16 step."""
    mgr = ks.build_manager("plain")
    try:
        module = ks.build_module(TARGET, k3_buffers=buffers)
        slots = ks.request_slots(mgr, 4, first_id=60)
        ks.fill_pools(mgr, seed=21)
        _step(mgr, module, ks.make_step(mgr, slots, [30, 7, 65, 2], [False] * 4, []), 22, "prefill")
        decode = ks.make_step(mgr, [], [], [], slots)
        hidden = ks.hidden_for(decode, 23)
        rows = torch.tensor(slots, dtype=torch.long, device="cuda")
        before = ks.snapshot(mgr)
        with torch.inference_mode():
            fused = _run(module, decode, hidden)().clone()
        cache = mgr.mamba_layer_cache(0)
        fused_state, fused_conv = cache.temporal[rows].clone(), cache.conv[rows].clone()
        ks.restore(mgr, before)
        with torch.inference_mode():
            generic = _run(module, decode, hidden, "generic")().clone()
        torch.cuda.synchronize()
        assert_within(fused, generic, TOL_OUT, "decode output")
        assert_within(fused_state, cache.temporal[rows], TOL_STATE, "decode state")
        assert_within(fused_conv, cache.conv[rows], 2**-7, "decode conv window")
    finally:
        mgr.shutdown()


def test_generic_decode_of_more_than_eight_requests(buffers):
    mgr = ks.build_manager("plain", max_batch_size=16)
    try:
        module = ks.build_module(TARGET, k3_buffers=buffers, aux_stream=torch.cuda.Stream())
        slots = ks.request_slots(mgr, 12, first_id=100)
        ks.fill_pools(mgr, seed=6)
        lens = [3, 300, 1, 17, 64, 9, 33, 2, 128, 5, 40, 11]
        prefill = ks.make_step(mgr, slots, lens, [i % 3 == 0 for i in range(12)], [])
        _step(mgr, module, prefill, 7, "prefill x12")
        decode = ks.make_step(mgr, [], [], [], slots)
        assert ks.decode_step_of(TARGET, decode) is None  # more than 8 requests: the generic path
        for i in range(2):
            _step(mgr, module, decode, 8 + i, f"decode x12 #{i}")
    finally:
        mgr.shutdown()


def test_breakable_cuda_graph_step_matches_eager(buffers):
    """A mixed step inside a breakable CUDA graph: the core runs as one eager break reading the live metadata from
    the engine's model attributes; the replay gives the bits of the same step run eagerly."""
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
        BreakableCUDAGraph,
        BreakableCUDAGraphCapture,
        enable_breakable_cuda_graph,
    )
    from tensorrt_llm._torch.utils import model_extra_attrs

    mgr = ks.build_manager("plain")
    try:
        module = ks.build_module(TARGET, k3_buffers=buffers)
        slots = ks.request_slots(mgr, 3, first_id=40)
        ks.fill_pools(mgr, seed=11)
        metadata = ks.make_step(mgr, slots[:2], [40, 24], [False, True], slots[2:])
        hidden = ks.hidden_for(metadata, 12)
        before = ks.snapshot(mgr)
        with torch.inference_mode():
            want = module.forward_generic(hidden, metadata).clone()
        torch.cuda.synchronize()
        after_eager = ks.snapshot(mgr)
        ks.restore(mgr, before)
        graph = BreakableCUDAGraph()
        static = hidden.clone()
        attrs = {"attention_metadata": weakref.ref(metadata)}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), model_extra_attrs(attrs), enable_breakable_cuda_graph():
            assert not module.will_run_decode_branch(metadata, None)
            with torch.cuda.stream(stream):
                with BreakableCUDAGraphCapture(
                    graph, pool=torch.cuda.graph_pool_handle(), stream=stream
                ):
                    out = module.forward_generic(static, metadata)
            torch.cuda.current_stream().wait_stream(stream)
            assert graph.num_breaks == 1
            ks.restore(mgr, before)
            graph.replay()
        torch.cuda.synchronize()
        assert_bitwise(out, want, "breakable graph: output")
        for name, t in ks.pool_tensors(mgr).items():
            assert_bitwise(t, after_eager[name], f"breakable graph: pool {name}")
    finally:
        mgr.shutdown()
