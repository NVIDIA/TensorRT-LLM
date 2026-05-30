#!/usr/bin/env python3
"""Extract per-layer GPU kernel sequences from an nsys trace.

Uses graphNodeId to extract a single CUDA graph replay, then groups kernels
by stream. Each transformer layer uses a dedicated set of streams (main +
aux), so stream grouping naturally segments layers.

Layer boundaries are detected via fused_kernel MLIR hashes which act as
anchors between the FX graph dump and the GPU trace.

Usage:
    python extract_trace_kernels.py <trace.sqlite> [--layer N] [--output kernels.json]
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _shorten(name: str) -> str:
    """Produce a human-readable short kernel name."""
    if "fused_kernel_" in name:
        return name.split("(")[0] if "(" in name else name
    if "nvjet" in name:
        return name.split("(")[0].strip() if "(" in name else name.strip()
    if "rms_norm_ker" in name or "rms_norm_reduce" in name:
        return "rms_norm_reduce_fusion"
    if "fp8_1x128_quantize" in name:
        return "fp8_quantize"
    if "pack_fp32_into_ue8m0" in name:
        return "pack_fp32_to_ue8m0"
    if "sm100_fp8_gemm" in name:
        return "deep_gemm_fp8"
    if "act_and_mul" in name:
        return "act_and_mul(silu)"
    if "ncclSymkDevKernel_AllReduce" in name:
        return "nccl_allreduce_symk"
    if "ncclDevKernel_AllGather" in name:
        return "nccl_allgather"
    if "ncclDevKernel_AllReduce" in name:
        return "nccl_allreduce"
    if "multimem_all_gather" in name:
        return "symm_mem_allgather"
    if "multimem_all_reduce" in name:
        return "symm_mem_allreduce"
    if "applyMLARope" in name:
        return "mla_rope_assign_qkv"
    if "fmhaSm100" in name or "fmhaS" in name:
        return name.split("(")[0].strip() if "(" in name else name.strip()
    if "deepseek_v3_topk" in name:
        return "deepseek_v3_topk"
    if "routingIndicesCluster" in name or "routingRenormalize" in name:
        return "moe_routing_renormalize"
    if "activationDeepSeek" in name:
        return "moe_activation_deepseek"
    if "finalizeKernelVecLoad" in name:
        return "moe_finalize"
    if "bmm_E4m3" in name or "bmm_Bfloat16" in name:
        m = re.match(r"(bmm_\w+?)_t\d", name)
        return m.group(1) if m else name[:60]
    if "splitKreduce" in name:
        return "splitKreduce"
    if "scale_1x128_kernel" in name:
        return "fp8_blockscale"
    if "FillFunctor" in name:
        return "fill_zero"
    if "CatArrayBatched" in name:
        return "cat_batched"
    if "bfloat16_copy" in name:
        return "bf16_copy"
    if "BinaryFunctor" in name:
        return "binary_op"
    if "AUnaryFunctor" in name or "BUnaryFunctor" in name:
        return "unary_op"
    if "CUDAFunctor_add" in name or "CUDAFunctorOnSelf_add" in name:
        return "elementwise_add"
    if "unrolled_elementwise" in name and "direct_copy" in name:
        return "unrolled_copy"
    if "direct_copy" in name:
        return "direct_copy"
    if "vectorized_gather" in name:
        return "vectorized_gather"
    if "ragged_to_block" in name:
        return "ragged_to_block"
    if "gather_scatter" in name:
        return "gather_scatter"
    return name[:80]


def ensure_sqlite(trace_path: str) -> str:
    """Convert .nsys-rep to .sqlite if needed, return sqlite path."""
    p = Path(trace_path)
    if p.suffix == ".sqlite":
        return trace_path
    sqlite_path = p.with_suffix(".sqlite")
    if sqlite_path.exists():
        return str(sqlite_path)
    print(f"Exporting {p.name} to sqlite...", file=sys.stderr)
    subprocess.run(
        ["nsys", "export", "--type=sqlite", "--output", str(sqlite_path), trace_path],
        check=True,
    )
    return str(sqlite_path)


def _extract_single_replay(conn, graph_id: int) -> List[dict]:
    """Extract one complete replay using graphNodeId deduplication.

    Each unique graphNodeId represents a distinct kernel node in the CUDA
    graph topology. Taking the first execution (MIN start) of each gives
    us exactly one replay.
    """
    rows = conn.execute(
        """
        SELECT gn.graphNodeId, k.start, k.end, s.value, k.streamId
        FROM (
            SELECT graphNodeId, MIN(start) as first_start
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE graphId = ?
            GROUP BY graphNodeId
        ) gn
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k
            ON k.graphNodeId = gn.graphNodeId AND k.start = gn.first_start
        JOIN StringIds s ON k.demangledName = s.id
        WHERE k.graphId = ?
        ORDER BY k.start
    """,
        (graph_id, graph_id),
    ).fetchall()

    return [
        {
            "graph_node_id": r[0],
            "start": r[1],
            "end": r[2],
            "name": r[3],
            "stream_id": r[4],
        }
        for r in rows
    ]


def _detect_layer_streams(kernels: List[dict]) -> List[Tuple[int, List[int]]]:
    """Detect which streams belong to which layer.

    Strategy: fused_kernel MLIR hashes are anchor points. The residual-add +
    input_layernorm fused kernel (fused_kernel_a5fe... pattern) marks the
    transition from one layer's MoE/MLP output to the next layer's attention
    input. By tracking which streams contain these anchors and the attention
    fused kernels (fused_kernel_1984... and _2e06...), we can assign streams
    to layers.

    Returns a list of (layer_num, [stream_ids]) sorted by first kernel time.
    """
    stream_kernels: Dict[int, List[dict]] = defaultdict(list)
    for k in kernels:
        stream_kernels[k["stream_id"]].append(k)

    # Classify streams by their fused_kernel content
    stream_hashes: Dict[int, List[str]] = {}
    for sid, sks in stream_kernels.items():
        hashes = []
        for k in sks:
            if "fused_kernel_" in k["name"]:
                h = k["name"].split("(")[0] if "(" in k["name"] else k["name"]
                hashes.append(h)
        if hashes:
            stream_hashes[sid] = hashes

    # Count hash frequencies across all streams
    hash_freq: Dict[str, int] = defaultdict(int)
    for hashes in stream_hashes.values():
        for h in hashes:
            hash_freq[h] += 1

    # Group streams into layers by temporal adjacency and fused_kernel anchors.
    # Each layer's streams start executing after the previous layer's streams.
    # We order streams by their first kernel's start time.
    stream_first_time = {}
    for sid, sks in stream_kernels.items():
        stream_first_time[sid] = sks[0]["start"]

    ordered_streams = sorted(stream_kernels.keys(), key=lambda s: stream_first_time[s])

    # Build layers by grouping consecutive streams that form a layer.
    # A new layer starts when we see a stream containing the residual+norm
    # fused kernel (a5fe pattern) — this stream spans the boundary between
    # the current MoE/MLP output and the next layer's attention start.
    #
    # Alternative approach: use the attention fused kernels (1984, 2e06) as
    # layer markers. Each main attention stream has exactly these two.

    # Find which hash is the attention-block marker (appears on streams with
    # the most kernels that also have MLA rope, fmha, etc.)
    attn_streams = set()
    for sid, sks in stream_kernels.items():
        has_fmha = any("fmha" in k["name"].lower() or "applyMLARope" in k["name"] for k in sks)
        if has_fmha:
            attn_streams.add(sid)

    # MoE streams contain topk, bmm, moe_finalize
    moe_streams = set()
    for sid, sks in stream_kernels.items():
        has_moe = any(
            "deepseek_v3_topk" in k["name"] or "finalizeKernelVecLoad" in k["name"] for k in sks
        )
        if has_moe:
            moe_streams.add(sid)

    # Shared-expert / aux streams contain the residual+norm fused kernel
    # and shared expert deep_gemm, but NOT fmha
    aux_streams = set()
    for sid, hashes in stream_hashes.items():
        if sid not in attn_streams:
            aux_streams.add(sid)

    # Layer 0 is special: its main stream is the first stream and uses a
    # unique fused_kernel hash
    first_stream = ordered_streams[0]

    # Group streams into layers. Each layer consists of:
    # - One main attention stream (has fmha)
    # - Zero or one MoE stream (has topk/moe_finalize) — absent for dense layers
    # - Zero or more aux streams (shared experts, additional comms)
    #
    # The key insight: streams are dedicated per-layer, and we can pair them
    # by finding which streams overlap temporally.

    # For each stream, compute time range
    stream_ranges = {}
    for sid, sks in stream_kernels.items():
        stream_ranges[sid] = (sks[0]["start"], sks[-1]["end"])

    # Group streams by temporal overlap into layers
    # Start with attention streams as anchors (one per layer)
    attn_stream_list = sorted(attn_streams, key=lambda s: stream_first_time[s])

    # For layer 0, the first stream might not have fmha (dense layers 0-2
    # might share streams differently). Let's handle this by looking at all
    # streams that start before the first attention stream.

    # Detect layer 0 streams (everything before the 2nd attention stream's
    # first kernel, that isn't assigned to layer 1+)
    # Actually, simpler: each attention stream IS one layer's main stream.
    # Non-attention streams get assigned to the layer whose attention stream
    # they temporally overlap with.

    # But some streams span two layers (e.g., stream 2551 has 39 kernels
    # covering layer 1's MoE and layer 2's attention start). This happens
    # because the aux stream does: [layer N MoE] -> [residual+norm] ->
    # [layer N+1 attention projection start].
    #
    # For simplicity, assign each stream to the layer whose attention stream
    # it most overlaps with, or to the layer of its fused_kernel hash.

    # SIMPLER APPROACH: use the global kernel ordering.
    # Segment the globally-ordered kernel list by detecting layer boundaries.
    # The fused_kernel_a5fe... (residual+norm) marks the END of one layer
    # and START of the next. But layer 0 uses a different hash.
    #
    # Better: find attention blocks. Each fmha kernel marks one layer.
    # Count fmha kernels = num_layers. Kernels between consecutive fmha
    # blocks (with some offset) belong to the same layer.

    # Let's use the simplest reliable approach: per-stream assignment.
    # Assign each stream entirely to one layer based on which attention
    # stream it's closest to temporally.

    if not attn_stream_list:
        # No attention streams found — fall back to linear segmentation
        return [(0, list(stream_kernels.keys()))]

    # Build layer groups: one attention stream per layer
    layer_groups: List[Tuple[int, set]] = []  # (layer_num, {stream_ids})

    # Layer 0 might not have an fmha (if it's a different architecture).
    # Check if first stream is an attention stream:
    layer_num = 0
    for attn_sid in attn_stream_list:
        layer_groups.append((layer_num, {attn_sid}))
        layer_num += 1

    # Assign non-attention streams to the closest layer
    assigned = set(attn_streams)
    unassigned = [s for s in ordered_streams if s not in assigned]

    for sid in unassigned:
        sr_start, sr_end = stream_ranges[sid]
        mid = (sr_start + sr_end) // 2

        best_layer = 0
        best_dist = float("inf")
        for li, (lnum, sids) in enumerate(layer_groups):
            for lsid in sids:
                lr_start, lr_end = stream_ranges[lsid]
                lr_mid = (lr_start + lr_end) // 2
                dist = abs(mid - lr_mid)
                if dist < best_dist:
                    best_dist = dist
                    best_layer = li

        layer_groups[best_layer][1].add(sid)

    # Check if layer 0 has no attention stream — prepend it
    if first_stream not in attn_streams:
        # The first stream belongs to layer 0 but has no fmha.
        # Find which layer it got assigned to and fix up.
        for li, (lnum, sids) in enumerate(layer_groups):
            if first_stream in sids:
                if lnum != 0:
                    sids.discard(first_stream)
                    layer_groups.insert(0, (0, {first_stream}))
                    # Renumber
                    layer_groups = [(i, s) for i, (_, s) in enumerate(layer_groups)]
                break

    return [(lnum, sorted(sids)) for lnum, sids in layer_groups]


def extract_kernels(sqlite_path: str, target_layer: Optional[int] = None) -> dict:
    conn = sqlite3.connect(sqlite_path)

    # Find the CUDA graph with the most kernels
    graph_rows = conn.execute("""
        SELECT graphId, COUNT(*) as cnt
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE graphId != 0
        GROUP BY graphId
        ORDER BY cnt DESC
    """).fetchall()

    if not graph_rows:
        print("Error: no CUDA graphs found in trace.", file=sys.stderr)
        sys.exit(1)

    graph_id = graph_rows[0][0]
    total_in_graph = graph_rows[0][1]

    unique_nodes = conn.execute(
        """
        SELECT COUNT(DISTINCT graphNodeId)
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE graphId = ?
    """,
        (graph_id,),
    ).fetchone()[0]

    num_replays = total_in_graph // unique_nodes if unique_nodes else 0

    print(
        f"Graph {graph_id}: {unique_nodes} unique nodes, "
        f"{total_in_graph} total kernels, ~{num_replays} replays",
        file=sys.stderr,
    )

    # Extract one complete replay
    replay_kernels = _extract_single_replay(conn, graph_id)
    conn.close()

    if not replay_kernels:
        print("Error: no kernels found in replay.", file=sys.stderr)
        sys.exit(1)

    # Detect layer-to-stream mapping
    layer_streams = _detect_layer_streams(replay_kernels)

    # Build stream-to-kernel index
    stream_kernel_map: Dict[int, List[dict]] = defaultdict(list)
    for k in replay_kernels:
        stream_kernel_map[k["stream_id"]].append(k)

    # Build per-layer kernel lists
    layers = {}
    for lnum, stream_ids in layer_streams:
        if target_layer is not None and lnum != target_layer:
            continue

        layer_kerns = []
        for sid in stream_ids:
            for k in stream_kernel_map[sid]:
                layer_kerns.append(k)

        # Sort by start time
        layer_kerns.sort(key=lambda k: k["start"])

        kernels_out = []
        for ki, k in enumerate(layer_kerns):
            dur_ns = k["end"] - k["start"]
            kernels_out.append(
                {
                    "index": ki,
                    "kernel": _shorten(k["name"]),
                    "full_name": k["name"],
                    "duration_ns": dur_ns,
                    "duration_us": round(dur_ns / 1000, 1),
                    "stream_id": k["stream_id"],
                }
            )

        total_dur = sum(k["duration_ns"] for k in kernels_out)
        layers[str(lnum)] = {
            "layer": lnum,
            "num_streams": len(stream_ids),
            "stream_ids": stream_ids,
            "kernel_count": len(kernels_out),
            "total_duration_ns": total_dur,
            "total_duration_us": round(total_dur / 1000, 1),
            "kernels": kernels_out,
        }

    return {
        "source": os.path.basename(sqlite_path),
        "graph_id": graph_id,
        "unique_graph_nodes": unique_nodes,
        "num_replays": num_replays,
        "num_layers_detected": len(layer_streams),
        "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-layer kernel sequences from nsys trace."
    )
    parser.add_argument("trace", help="Path to .nsys-rep or .sqlite trace file")
    parser.add_argument("--layer", "-l", type=int, default=None, help="Extract only this layer")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    args = parser.parse_args()

    sqlite_path = ensure_sqlite(args.trace)
    result = extract_kernels(sqlite_path, args.layer)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
