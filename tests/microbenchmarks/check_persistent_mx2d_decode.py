# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Checks for the persistent and MXFP4-2D CuTeDSL FP4 MLA decodes (Rubin SM107).

``persistent``: on one NVFP4 cache written by the production context writer, the
persistent launch must be bitwise identical to the default launch (eager and over
repeated CUDA-graph replays, which also soak for hangs); timing of both by graph replay.

``mx2d``: the cache's latent prefix is overwritten with MXFP4-2D values (E2M1 codes on
one power-of-two scale per 32 tokens x 32 dims, scale bytes E4M3(s * 64) in the K-scale
plane, KV global scale 64), the V-pack and V scales are rebuilt from K, and the MXFP4-2D
decode is compared with an FP32 reference on the dequantized cache and with the NVFP4
MUFU16 decode on the same cache; the H85 mixed-precision row sum
(``mxfp4_2d_mixed_rowsum``) must be bitwise the MXFP4-2D decode, the H90 tensor-core row
sum (``mxfp4_2d_tc_rowsum``) is compared with the reference and with the MXFP4-2D decode.

``mx2d_e2e``: TRTLLM_FP4_MLA_MXFP4_2D=1 (set here before import): the cache is written by
the production MXFP4-2D context writer and decoded through fp4_mla's dispatch (which
passes ``mxfp4_2d=True``); compared with the FP32 reference, re-launched for bitwise
determinism, and timed by graph replay.

usage::

    python check_persistent_mx2d_decode.py persistent --batch 7 64 512 1536
    python check_persistent_mx2d_decode.py mx2d --batch 7 64 512 1536
    python check_persistent_mx2d_decode.py mx2d_e2e --batch 7 64 512 1536
    python check_persistent_mx2d_decode.py mx2d_split --batch 1 4 8 32 64

``mx2d_split``: as ``mx2d_e2e``, the production decode with and without
TRTLLM_FP4_MLA_MXFP4_2D_SPLITKV=1 (split-KV tasks + merge): both compared with the FP32
reference and with each other, the split decode re-run for determinism, and both captured
in CUDA graphs (with the LPT order and split plan re-planned in the graph), soaked and
timed (a steady-state layer: order and plan cached).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

if len(sys.argv) > 1 and sys.argv[1] in ("mx2d_e2e", "mx2d_split"):
    # fp4_mla reads the KV format once, at import.
    os.environ["TRTLLM_FP4_MLA_MXFP4_2D"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_dsr1_mla_decode as bench  # noqa: E402

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


TRACE_BUF = None  # host-mapped pinned int32 buffer for the debug kernel build (KERN_TRACE=1)


def _dump_trace():
    if TRACE_BUF is None:
        return
    vals = TRACE_BUF.tolist()
    roles = {1: "QK", 2: "V", 3: "ROWMETA", 4: "MMA", 5: "PROD"}
    for slot, v in enumerate(vals):
        if v:
            cluster, rem = divmod(slot, 32)
            cta, warp = divmod(rem, 16)
            role, rest = divmod(v, 10000000)
            point, rest = divmod(rest, 1000000)
            fetch, tile = divmod(rest, 1000)
            print(
                f"TRACE cluster={cluster} cta={cta} warp={warp:2d} role={roles.get(role, role)} "
                f"point={point} fetch={fetch} tile={tile}",
                flush=True,
            )


def _sync_or_die(label, timeout_s=None):
    """Synchronize with a watchdog: a hung kernel ends the process with a message."""
    timeout_s = (
        float(os.environ.get("CHECK_SYNC_TIMEOUT", "30")) if timeout_s is None else timeout_s
    )
    event = torch.cuda.Event()
    event.record()
    deadline = time.time() + timeout_s
    while not event.query():
        if time.time() > deadline:
            print(f"HANG {label}: no completion after {timeout_s:.0f}s", flush=True)
            _dump_trace()
            os._exit(3)
        time.sleep(0.01)


def _fp4():
    import tensorrt_llm._torch.attention_backend.fp4_mla as fp4

    return fp4


def _capture_launch(fp4):
    """Run the production decode once with a recording launcher; return (kernel, args, kwargs)."""
    kernel = fp4._fp4_mla_cutedsl_kernel_module()
    original = kernel.run_trtllm_fp4_mla_decode_page_native_from_raw
    captured = {}

    def record(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = dict(kwargs)
        return original(*args, **kwargs)

    kernel.run_trtllm_fp4_mla_decode_page_native_from_raw = record
    return kernel, original, captured


def _decode_inputs(metadata, q, fp4):
    batch = q.shape[0]
    metadata.fp4_mla_generation_kv_lens.copy_(metadata.kv_lens_cuda_runtime)
    metadata.fp4_mla_generation_append_lens.copy_(metadata.prompt_lens_cuda_runtime)
    metadata.fp4_mla_generation_lengths_num_tokens = batch
    metadata.fp4_mla_generation_lengths_num_seqs = batch
    metadata.fp4_mla_generation_lengths_num_contexts = 0
    metadata._fp4_mla_generation_lengths_capture_recorded = True
    q_fp4, q_sf = torch.ops.trtllm.fp4_quantize_with_residual(
        q.view(-1, q.shape[-1]),
        metadata._fp4_mla_q_global_scale,
        fp4.FP4_MLA_Q_RESIDUAL_DIM,
        is_act=True,
    )
    return q_fp4, q_sf.view(torch.float8_e4m3fn).reshape(-1)


def _run_default_and_capture(metadata, q, heads, fp4):
    q_fp4, q_sf = _decode_inputs(metadata, q, fp4)
    out = torch.full(
        (q.shape[0], heads, bench.KV_LORA_RANK), float("nan"), dtype=torch.bfloat16, device=q.device
    )
    kernel, original, captured = _capture_launch(fp4)
    try:
        fp4.run_fp4_mla_attention_decode(
            metadata,
            layer_idx=0,
            local_layer=0,
            q=q,
            output=out,
            sm_scale=bench.SM_SCALE,
            kv_lora_rank=bench.KV_LORA_RANK,
            qk_rope_head_dim=bench.QK_ROPE_HEAD_DIM,
            prequantized_q=q_fp4,
            prequantized_q_sf=q_sf,
            q_batch_capacity=q.shape[0],
        )
    finally:
        kernel.run_trtllm_fp4_mla_decode_page_native_from_raw = original
    torch.cuda.synchronize()
    return kernel, out, captured


def _launch(kernel, captured, **overrides):
    args = list(captured["args"])
    kwargs = dict(captured["kwargs"])
    if TRACE_BUF is not None and overrides.get("persistent"):
        TRACE_BUF.zero_()
        kwargs["_trace"] = TRACE_BUF
    out = args[10]
    out.fill_(float("nan"))
    kwargs.update(overrides)
    kernel.run_trtllm_fp4_mla_decode_page_native_from_raw(*args, **kwargs)
    return out


def _graph_time(fn, iters=50, check=None):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.synchronize()
    for _ in range(5):
        graph.replay()
    _sync_or_die("graph warm replays")
    if check is not None:
        check()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    _sync_or_die("graph timed replays", 120)
    if check is not None:
        check()
    return start.elapsed_time(end) / iters * 1e3, graph


def _bitwise(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def check_persistent(seq_lens, heads, args):
    fp4 = _fp4()
    manager, metadata, q = bench.build_fp4_case(seq_lens, heads, "cutedsl")
    try:
        kernel, ref, captured = _run_default_and_capture(metadata, q, heads, fp4)
        ref = ref.clone()
        line = f"PERSISTENT B={len(seq_lens)} kv[min/mean/max]={min(seq_lens)}/{int(np.mean(seq_lens))}/{max(seq_lens)}"
        lpt = "request_order" in captured["kwargs"]
        results = {}
        for name, overrides in (
            ("default", {}),
            ("persistent", {"persistent": True}),
            ("persistent_c8", {"persistent": True, "max_clusters": 8}),
        ):
            print(f"   launch {name} ...", flush=True)
            out = _launch(kernel, captured, **overrides)
            _sync_or_die(f"{name} B={len(seq_lens)}")
            results[name] = _bitwise(out, ref) and bool(torch.isfinite(out).all().item())
            if not results[name]:
                diff = (out.float() - ref.float()).abs()
                bad = (diff > 0) | ~torch.isfinite(out.float())
                rows = bad.any(dim=2).any(dim=1).nonzero().flatten().tolist()
                print(
                    f"   {name}: differing queries {rows[:16]} (n={len(rows)}) max_abs="
                    f"{diff[torch.isfinite(diff)].max().item() if torch.isfinite(diff).any() else 'nan'}",
                    flush=True,
                )
        line += " eager: " + " ".join(
            f"{k}={'BITWISE' if v else 'DIFF'}" for k, v in results.items()
        )
        line += f" lpt={lpt}"
        print(line, flush=True)
        # CUDA-graph soak + timing.
        times = {}
        for name, overrides in (("default", {}), ("persistent", {"persistent": True})):
            out_holder = {}

            def fn(overrides=overrides, out_holder=out_holder):
                out_holder["out"] = _launch(kernel, captured, **overrides)

            def check(out_holder=out_holder):
                if not _bitwise(out_holder["out"], ref):
                    raise RuntimeError("graph replay output differs from the default launch")

            t0 = time.time()
            us, graph = _graph_time(fn, iters=args.iters, check=check)
            for _ in range(args.soak):
                graph.replay()
            _sync_or_die(f"soak {name}", 300)
            check()
            times[name] = us
            del graph
            print(
                f"   graph {name}: {us:.1f} us/call soak={args.soak} ok ({time.time() - t0:.0f}s)",
                flush=True,
            )
        print(
            f"PERSISTENT_TIMING B={len(seq_lens)} mean_kv={int(np.mean(seq_lens))} "
            f"default_us={times['default']:.1f} persistent_us={times['persistent']:.1f} "
            f"speedup={times['default'] / times['persistent']:.3f}",
            flush=True,
        )
    finally:
        manager.shutdown()


# --------------------------------------------------------------------------- #
# MXFP4-2D
# --------------------------------------------------------------------------- #
def _swizzled_sf_offset(row, col, sf_cols):
    padded = ((sf_cols + 3) // 4) * 4
    return (
        col % 4
        + (col // 4) * (4 * 128)
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (row // 128) * (128 * padded)
    )


def _e2m1_codes(y):
    """The writers' E2M1 rule: nearest, ties up, saturate at 6; zero -> code 0 (never -0)."""
    m = y.abs().clamp(max=6.0)
    small = torch.floor(m * 2.0 + 0.5) * 0.5
    mid = torch.floor(m + 0.5)
    big = torch.where(m >= 5.0, torch.full_like(m, 6.0), torch.full_like(m, 4.0))
    r = torch.where(m < 2.0, small, torch.where(m < 4.0, mid, big))
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=y.device)
    mag = (r[..., None] == lut).float().argmax(dim=-1)
    neg = (y < 0) & (mag > 0)
    return (mag + 8 * neg.to(mag.dtype)).to(torch.uint8)


def _mx2d_block_scale(amax):
    """2^(floor(log2 amax) - 2), clamped to [2^-15, 4] (amax 0 -> 2^-15)."""
    bits = amax.float().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 2
    e = e.clamp(127 - 15, 127 + 2)
    return (e << 23).view(torch.float32)


def write_mx2d_prefix(metadata, latent, seq_lens):
    """Overwrite the latent prefix codes and K-scale bytes of every token with MXFP4-2D values."""
    kv_cache, sf_cache = metadata.kv_cache_manager.get_fp4_mla_cache_buffers(0)
    sf_u8 = sf_cache.view(torch.uint8)
    stride = metadata.fp4_mla_page_table_stride
    pages = metadata.paged_kv_indices.view(-1, stride)
    starts = np.concatenate(([0], np.cumsum(seq_lens)[:-1]))
    gs = float(bench_kv_global_scale(metadata))
    assert gs == 64.0, gs
    sf_cols = (kv_cache.shape[-1] * 2) // 16
    for i, kv_len in enumerate(seq_lens):
        x = latent[starts[i] : starts[i] + kv_len, : bench.KV_LORA_RANK].float()
        nblk = (kv_len + 31) // 32
        xp = torch.zeros(nblk * 32, bench.KV_LORA_RANK, device=x.device)
        xp[:kv_len] = x
        blocks = xp.view(nblk, 32, 16, 32)  # [block, token, group, dim]
        amax = blocks.abs().amax(dim=(1, 3))  # [block, group]
        s = _mx2d_block_scale(amax)  # [block, group]
        y = blocks / s[:, None, :, None]
        codes = _e2m1_codes(y).view(nblk * 32, bench.KV_LORA_RANK)[:kv_len]
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)  # [T, 256]
        sbyte = (s * gs).to(torch.float8_e4m3fn).view(torch.uint8)  # [block, group]
        tok = torch.arange(kv_len, device=x.device)
        page_rel = tok // 128
        row = tok % 128
        phys = pages[i, page_rel].long()
        kv_view = kv_cache.view(torch.uint8)[:, 0, :, 0, :]  # [pages, 128, 320]
        kv_view[phys, row, :256] = packed
        tok_s = sbyte[tok // 32]  # [T, group]
        for g in range(16):
            for half in range(2):
                col = 2 * g + half
                off = _swizzled_sf_offset(row, col, sf_cols)
                sf_u8.view(sf_u8.shape[0], -1)[phys, off] = tok_s[:, g]
    torch.cuda.synchronize()


def bench_kv_global_scale(metadata):
    return float(metadata._fp4_mla_kv_global_scale.item())


def build_mx2d_case(seq_lens, heads, fp4, seed=8):
    """build_fp4_case with KV global scale 64 (the MXFP4-2D scale), keeping the latent."""
    orig = fp4.FP4_MLA_KV_GLOBAL_SCALE
    fp4.FP4_MLA_KV_GLOBAL_SCALE = bench_mx2d_gs = 64.0
    try:
        manager, metadata, q = bench.build_fp4_case(
            seq_lens, heads, "cutedsl", seed=seed, keep_latent=True
        )
    finally:
        fp4.FP4_MLA_KV_GLOBAL_SCALE = orig
    assert bench_kv_global_scale(metadata) == bench_mx2d_gs
    return manager, metadata, q


def check_mx2d(seq_lens, heads, args):
    fp4 = _fp4()
    import check_dsr1_fp4_mla_decode_numerics as numerics

    manager, metadata, q = build_mx2d_case(seq_lens, heads, fp4)
    try:
        write_mx2d_prefix(metadata, metadata._bench_latent, seq_lens)
        # Rebuild V scales and the V-pack from the MXFP4-2D K pages for the NVFP4 decode.
        for rid, kv_len in enumerate(seq_lens):
            fp4.rebuild_fp4_mla_disagg_imported_cache(manager, rid, int(kv_len))
        torch.cuda.synchronize()
        kernel, nvfp4_out, captured = _run_default_and_capture(metadata, q, heads, fp4)
        nvfp4_out = nvfp4_out.clone()
        print("   launch mx2d ...", flush=True)
        mx_out = _launch(kernel, captured, mxfp4_2d=True)
        _sync_or_die(f"mx2d B={len(seq_lens)}")
        mx_out = mx_out.clone()
        print("   launch mx2d_h85 ...", flush=True)
        h85_out = _launch(kernel, captured, mxfp4_2d=True, mxfp4_2d_mixed_rowsum=True)
        _sync_or_die(f"mx2d_h85 B={len(seq_lens)}")
        h85 = "BITWISE" if _bitwise(h85_out, mx_out) else "DIFF"
        print("   launch mx2d_h90 ...", flush=True)
        h90_out = _launch(kernel, captured, mxfp4_2d=True, mxfp4_2d_tc_rowsum=True)
        _sync_or_die(f"mx2d_h90 B={len(seq_lens)}")
        h90_out = h90_out.clone()
        exact, ref = numerics._references(metadata, q, seq_lens, bench.SM_SCALE)
        label = (
            f"MX2D B={len(seq_lens)} kv[min/mean/max]={min(seq_lens)}/"
            f"{int(np.mean(seq_lens))}/{max(seq_lens)}"
        )
        for a, b, e in (
            ("mx2d", "exact", numerics._err(mx_out, exact)),
            ("nvfp4_mufu16", "exact", numerics._err(nvfp4_out, exact)),
            ("mx2d", "nvfp4_mufu16", numerics._err(mx_out, nvfp4_out)),
            ("mx2d_h90", "exact", numerics._err(h90_out, exact)),
            ("mx2d_h90", "mx2d", numerics._err(h90_out, mx_out)),
            ("mx2d", "bf16", numerics._err(mx_out, ref)),
            ("exact_fp32", "bf16", numerics._err(exact, ref)),
        ):
            fields = " ".join(
                f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()
            )
            print(f"{label} {a} vs {b}: {fields}", flush=True)
        worst = max(numerics._err(mx_out[i], exact[i])["rel_l2"] for i in range(len(seq_lens)))
        print(
            f"{label} mx2d vs exact worst_seq_rel_l2={worst:.5g} h85_mixed_rowsum={h85}", flush=True
        )
        if args.time:
            times = {}
            for name, overrides in (
                ("nvfp4_default", {}),
                ("nvfp4_persistent", {"persistent": True}),
                ("mx2d", {"mxfp4_2d": True}),
                ("mx2d_h85", {"mxfp4_2d": True, "mxfp4_2d_mixed_rowsum": True}),
                ("mx2d_h90", {"mxfp4_2d": True, "mxfp4_2d_tc_rowsum": True}),
            ):
                holder = {}

                def fn(overrides=overrides, holder=holder):
                    holder["out"] = _launch(kernel, captured, **overrides)

                us, graph = _graph_time(fn, iters=args.iters)
                for _ in range(args.soak):
                    graph.replay()
                torch.cuda.synchronize()
                times[name] = us
                del graph
            print(
                f"MX2D_TIMING B={len(seq_lens)} mean_kv={int(np.mean(seq_lens))} "
                + " ".join(f"{k}_us={v:.1f}" for k, v in times.items()),
                flush=True,
            )
    finally:
        manager.shutdown()


def check_mx2d_e2e(seq_lens, heads, args):
    fp4 = _fp4()
    import check_dsr1_fp4_mla_decode_numerics as numerics

    assert fp4._fp4_mla_mxfp4_2d_enabled() and fp4.FP4_MLA_KV_GLOBAL_SCALE == 64.0
    manager, metadata, q = bench.build_fp4_case(seq_lens, heads, "cutedsl", keep_latent=True)
    try:
        print("   production decode ...", flush=True)
        kernel, out, captured = _run_default_and_capture(metadata, q, heads, fp4)
        assert captured["kwargs"].get("mxfp4_2d") is True, sorted(captured["kwargs"])
        out = out.clone()
        again = _launch(kernel, captured)
        _sync_or_die(f"mx2d_e2e B={len(seq_lens)}")
        exact, ref = numerics._references(metadata, q, seq_lens, bench.SM_SCALE)
        label = (
            f"MX2D_E2E B={len(seq_lens)} kv[min/mean/max]={min(seq_lens)}/"
            f"{int(np.mean(seq_lens))}/{max(seq_lens)}"
        )
        for a, b, e in (
            ("mx2d", "exact", numerics._err(out, exact)),
            ("mx2d", "bf16", numerics._err(out, ref)),
            ("exact_fp32", "bf16", numerics._err(exact, ref)),
        ):
            fields = " ".join(
                f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()
            )
            print(f"{label} {a} vs {b}: {fields}", flush=True)
        worst = max(numerics._err(out[i], exact[i])["rel_l2"] for i in range(len(seq_lens)))
        relaunch = "BITWISE" if _bitwise(again, out) else "DIFF"
        print(f"{label} worst_seq_rel_l2={worst:.5g} relaunch={relaunch}", flush=True)
        if args.time:
            holder = {}

            def fn():
                holder["out"] = _launch(kernel, captured)

            def check():
                if not _bitwise(holder["out"], out):
                    raise RuntimeError("graph replay output differs from the production decode")

            us, graph = _graph_time(fn, iters=args.iters, check=check)
            for _ in range(args.soak):
                graph.replay()
            _sync_or_die(f"soak mx2d_e2e B={len(seq_lens)}", 300)
            check()
            del graph
            print(
                f"MX2D_E2E_TIMING B={len(seq_lens)} mean_kv={int(np.mean(seq_lens))} "
                f"mx2d_us={us:.1f} soak={args.soak} ok",
                flush=True,
            )
    finally:
        manager.shutdown()


def _production_decode(metadata, q, heads, fp4, out, q_fp4, q_sf):
    fp4.run_fp4_mla_attention_decode(
        metadata,
        layer_idx=0,
        local_layer=0,
        q=q,
        output=out,
        sm_scale=bench.SM_SCALE,
        kv_lora_rank=bench.KV_LORA_RANK,
        qk_rope_head_dim=bench.QK_ROPE_HEAD_DIM,
        prequantized_q=q_fp4,
        prequantized_q_sf=q_sf,
        q_batch_capacity=q.shape[0],
    )


def _set_split(on):
    os.environ["TRTLLM_FP4_MLA_MXFP4_2D_SPLITKV"] = "1" if on else "0"


def _replan_next_call(metadata):
    """What a new forward step does: the next decode re-sorts the LPT order (and re-plans)."""
    metadata._fp4_mla_lpt_order_valid = False
    metadata._fp4_mla_lpt_capture_recorded = False


def check_mx2d_split(seq_lens, heads, args):
    fp4 = _fp4()
    import check_dsr1_fp4_mla_decode_numerics as numerics

    assert fp4._fp4_mla_mxfp4_2d_enabled() and fp4.FP4_MLA_KV_GLOBAL_SCALE == 64.0
    manager, metadata, q = bench.build_fp4_case(seq_lens, heads, "cutedsl", keep_latent=True)
    label = (
        f"MX2D_SPLIT B={len(seq_lens)} kv[min/mean/max]={min(seq_lens)}/"
        f"{int(np.mean(seq_lens))}/{max(seq_lens)}"
    )
    try:
        q_fp4, q_sf = _decode_inputs(metadata, q, fp4)
        outs = {}
        for name, on in (("unsplit", False), ("split", True)):
            _set_split(on)
            _replan_next_call(metadata)
            kernel, original, captured = _capture_launch(fp4)
            out = torch.full(
                (q.shape[0], heads, bench.KV_LORA_RANK),
                float("nan"),
                dtype=torch.bfloat16,
                device=q.device,
            )
            try:
                print(f"   {name} decode ...", flush=True)
                _production_decode(metadata, q, heads, fp4, out, q_fp4, q_sf)
                _sync_or_die(f"{name} B={len(seq_lens)}")
            finally:
                kernel.run_trtllm_fp4_mla_decode_page_native_from_raw = original
            assert ("split_kv" in captured["kwargs"]) == on, sorted(captured["kwargs"])
            if on:
                task_map = captured["kwargs"]["split_kv"][0]
                windows = int((task_map[:, 2] > task_map[:, 1]).sum().item())
                print(f"   split tasks={task_map.shape[0]} windows={windows}", flush=True)
            outs[name] = out
        # Determinism: the split decode again, from the cached plan.
        again = torch.full_like(outs["split"], float("nan"))
        _production_decode(metadata, q, heads, fp4, again, q_fp4, q_sf)
        _sync_or_die(f"split again B={len(seq_lens)}")
        exact, ref = numerics._references(metadata, q, seq_lens, bench.SM_SCALE)
        for a, b, e in (
            ("split", "exact", numerics._err(outs["split"], exact)),
            ("unsplit", "exact", numerics._err(outs["unsplit"], exact)),
            ("split", "unsplit", numerics._err(outs["split"], outs["unsplit"])),
        ):
            fields = " ".join(
                f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()
            )
            print(f"{label} {a} vs {b}: {fields}", flush=True)
        worst = {
            name: max(
                numerics._err(outs[name][i], exact[i])["rel_l2"] for i in range(len(seq_lens))
            )
            for name in outs
        }
        rerun = "BITWISE" if _bitwise(again, outs["split"]) else "DIFF"
        print(
            f"{label} worst_seq_rel_l2 split={worst['split']:.5g} "
            f"unsplit={worst['unsplit']:.5g} rerun={rerun}",
            flush=True,
        )
        if args.time:
            times = {}
            for name, on in (("unsplit", False), ("split", True)):
                _set_split(on)
                graph_out = torch.full_like(outs[name], float("nan"))

                def fn(graph_out=graph_out):
                    _production_decode(metadata, q, heads, fp4, graph_out, q_fp4, q_sf)

                # A graph that re-sorts and re-plans (the first layer of a step), soaked.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    fn()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                _replan_next_call(metadata)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    fn()
                graph_out.fill_(float("nan"))
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.soak):
                    graph.replay()
                end.record()
                _sync_or_die(f"soak {name} B={len(seq_lens)}", 300)
                times[name + "_replan"] = start.elapsed_time(end) / max(args.soak, 1) * 1e3
                if not _bitwise(graph_out, outs[name]):
                    raise RuntimeError(f"{name}: graph replay output differs from the eager decode")
                if on:
                    # The next step's lengths (every query 1-200 tokens shorter): the replay
                    # re-sorts and re-plans on the device and must match an eager re-plan.
                    lens = (metadata.kv_lens_cuda_runtime, metadata.fp4_mla_generation_kv_lens)
                    saved = [t[: len(seq_lens)].clone() for t in lens]
                    shrink = torch.randint(
                        1, 201, saved[0].shape, device=q.device, dtype=saved[0].dtype
                    )
                    for t, old in zip(lens, saved):
                        t[: len(seq_lens)].copy_(torch.clamp(old - shrink, min=1))
                    graph.replay()
                    _sync_or_die(f"replay new lengths B={len(seq_lens)}")
                    eager = torch.full_like(graph_out, float("nan"))
                    _replan_next_call(metadata)
                    _production_decode(metadata, q, heads, fp4, eager, q_fp4, q_sf)
                    _set_split(False)
                    _replan_next_call(metadata)
                    unsplit = torch.full_like(graph_out, float("nan"))
                    _production_decode(metadata, q, heads, fp4, unsplit, q_fp4, q_sf)
                    _set_split(True)
                    _sync_or_die(f"eager new lengths B={len(seq_lens)}")
                    same = "BITWISE" if _bitwise(graph_out, eager) else "DIFF"
                    err = numerics._err(graph_out, unsplit)
                    print(
                        f"{label} new lengths: graph_replay vs eager_split={same} "
                        f"vs unsplit rel_l2={err['rel_l2']:.3g}",
                        flush=True,
                    )
                    for t, old in zip(lens, saved):
                        t[: len(seq_lens)].copy_(old)
                    if same != "BITWISE":
                        raise RuntimeError(
                            "split: graph replay with new lengths differs from eager"
                        )
                del graph
                # A steady-state layer: order and plan cached (re-planned eagerly for these
                # lengths; the capture records neither the sort nor the plan).
                _replan_next_call(metadata)
                fn()
                metadata._fp4_mla_lpt_capture_recorded = True
                us, graph = _graph_time(fn, iters=args.iters)
                if not _bitwise(graph_out, outs[name]):
                    raise RuntimeError(f"{name}: timed replay output differs from the eager decode")
                del graph
                times[name] = us
            print(
                f"MX2D_SPLIT_TIMING B={len(seq_lens)} mean_kv={int(np.mean(seq_lens))} "
                f"unsplit_us={times['unsplit']:.1f} split_us={times['split']:.1f} "
                f"speedup={times['unsplit'] / times['split']:.3f} "
                f"first_layer(sort+plan) unsplit_us={times['unsplit_replan']:.1f} "
                f"split_us={times['split_replan']:.1f} soak={args.soak} ok",
                flush=True,
            )
    finally:
        _set_split(False)
        manager.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["persistent", "mx2d", "mx2d_e2e", "mx2d_split"])
    p.add_argument("--batch", type=int, nargs="+", default=[7, 64, 512, 1536])
    p.add_argument("--dist", default="dataset")
    p.add_argument("--isl-osl", default=None)
    p.add_argument("--max-len", type=int, default=23200)
    p.add_argument("--heads", type=int, default=128)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--soak", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--time", action="store_true")
    p.add_argument(
        "--trace", action="store_true", help="KERN_TRACE=1 kernel build: progress dump on hang"
    )
    args = p.parse_args()
    global TRACE_BUF
    if args.trace:
        TRACE_BUF = torch.zeros(16384, dtype=torch.int32, pin_memory=True)
    isl_osl = bench.load_isl_osl(args.isl_osl) if args.isl_osl else None
    for batch in args.batch:
        dist = (
            args.dist
            if (isl_osl is not None or not args.dist.startswith("dataset"))
            else "uniform:1-4000"
        )
        seq_lens = bench.sample_kv_lens(dist, batch, args.seed, isl_osl, args.max_len)
        try:
            if args.mode == "persistent":
                check_persistent(seq_lens, args.heads, args)
            elif args.mode == "mx2d":
                check_mx2d(seq_lens, args.heads, args)
            elif args.mode == "mx2d_e2e":
                check_mx2d_e2e(seq_lens, args.heads, args)
            else:
                check_mx2d_split(seq_lens, args.heads, args)
        except Exception as exc:  # report and continue with the next batch
            import traceback

            traceback.print_exc()
            print(f"CHECK_ERROR B={batch}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            torch.cuda.synchronize()
    print("CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
