import argparse
import bisect
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from parser_utils import (
    kernel_short_name,
    lazy_convert_sqlite,
    shortest_common_supersequence,
    warned_names,
)


def comma_separated_ints(s):
    return [int(x) for x in s.split(",")]


parser = argparse.ArgumentParser()
parser.add_argument("--eager-trace", type=str, required=True)
parser.add_argument("--graph-trace", type=str)
parser.add_argument("--target-ctx-reqs", type=int, default=0)
parser.add_argument("--target-gen-reqs", type=int)
parser.add_argument("--layer-indices", type=comma_separated_ints, required=True)
parser.add_argument("--warmup-times", type=int, default=5)
group = parser.add_mutually_exclusive_group()
group.add_argument("--error-on-unknown-kernel", action="store_true", dest="error_on_unknown_kernel")
group.add_argument(
    "--no-error-on-unknown-kernel", action="store_false", dest="error_on_unknown_kernel"
)
parser.set_defaults(error_on_unknown_kernel=False)
parser.add_argument("--output", "-o", type=str)
args = parser.parse_args()
if not args.eager_trace.endswith(".nsys-rep"):
    parser.error("Please provide a .nsys-rep file for the --eager-trace option.")
if args.graph_trace is not None and not args.graph_trace.endswith(".nsys-rep"):
    parser.error("Please provide a .nsys-rep file for the --graph-trace option.")
print(args)


def is_gemm(name):
    return "nvjet" in name or "gemm" in name.lower()


eager_nsys_rep_file_path = Path(args.eager_trace)
# For CTX phase which does not use CUDA Graphs, analysis the eager trace instead.
# Here we do not change the identifier name "graph_*" for convenience.
graph_nsys_rep_file_path = Path(args.graph_trace or args.eager_trace)
eager_sqlite_file_path = eager_nsys_rep_file_path.parent / (
    eager_nsys_rep_file_path.name[: -len(".nsys-rep")] + ".sqlite"
)
graph_sqlite_file_path = graph_nsys_rep_file_path.parent / (
    graph_nsys_rep_file_path.name[: -len(".nsys-rep")] + ".sqlite"
)
lazy_convert_sqlite(eager_nsys_rep_file_path, eager_sqlite_file_path)
lazy_convert_sqlite(graph_nsys_rep_file_path, graph_sqlite_file_path)
eager_conn = sqlite3.connect(f"file:{eager_sqlite_file_path}?mode=ro", uri=True)
graph_conn = sqlite3.connect(f"file:{graph_sqlite_file_path}?mode=ro", uri=True)

query = "SELECT * FROM ENUM_NSYS_EVENT_TYPE"
df = pd.read_sql_query(query, eager_conn)
eager_event_id_NvtxPushPopRange = df[df["name"] == "NvtxPushPopRange"].iloc[0]["id"].tolist()
df = pd.read_sql_query(query, graph_conn)
graph_event_id_NvtxPushPopRange = df[df["name"] == "NvtxPushPopRange"].iloc[0]["id"].tolist()

query = """SELECT T1.start, T1.end, T2.value AS text
    FROM NVTX_EVENTS AS T1
    JOIN StringIds AS T2 ON T1.textId = T2.id
    WHERE eventType = ?"""
df = pd.read_sql_query(query, eager_conn, params=(eager_event_id_NvtxPushPopRange,))
target_ctx_reqs = args.target_ctx_reqs
target_gen_reqs = args.target_gen_reqs
if target_gen_reqs is None:
    if target_ctx_reqs == 0:
        for _, _, text in df.itertuples(index=False):
            if m := re.match(
                r"^\[Executor\] _forward_step (\d+): (\d+) ctx reqs, (\d+) gen reqs", text
            ):
                ctx_reqs = int(m.group(2))
                gen_reqs = int(m.group(3))
                if ctx_reqs == target_ctx_reqs:
                    target_gen_reqs = gen_reqs
                    break
        else:
            raise ValueError("Cannot determine target_gen_reqs")
    else:
        target_gen_reqs = 0
print(f"{target_ctx_reqs=} {target_gen_reqs=}")
eager_iters = []
for start, end, text in df.itertuples(index=False):
    if m := re.match(r"^\[Executor\] _forward_step (\d+): (\d+) ctx reqs, (\d+) gen reqs", text):
        it = int(m.group(1))
        ctx_reqs = int(m.group(2))
        gen_reqs = int(m.group(3))
        if ctx_reqs == target_ctx_reqs and gen_reqs == target_gen_reqs:
            eager_iters.append((start, end, it))
eager_iters = sorted(eager_iters)[args.warmup_times :]
iter_list = [t[2] for t in eager_iters]
print("Iters (eager)", *iter_list)
per_iter_eager_layers = [[] for _ in iter_list]
for start, end, text in df.itertuples(index=False):
    if m := re.match(r"^layer_wise_benchmarks layer_idx (\d+)$", text):
        layer_idx = int(m.group(1))
        it_idx = bisect.bisect(eager_iters, (start,)) - 1
        if it_idx < 0 or end > eager_iters[it_idx][1]:
            continue
        assert end <= eager_iters[it_idx][1], "Not belong to any iter"
        per_iter_eager_layers[it_idx].append((start, end, it_idx, layer_idx))
layer_list = [t[3] for t in per_iter_eager_layers[0]]
print("Layers (eager)", *layer_list)
for eager_layers in per_iter_eager_layers:
    assert [t[3] for t in eager_layers] == layer_list, "inconsistent layer idx"
df = pd.read_sql_query(query, graph_conn, params=(graph_event_id_NvtxPushPopRange,))
graph_iters = []
for start, end, text in df.itertuples(index=False):
    if m := re.match(r"^\[Executor\] _forward_step (\d+): (\d+) ctx reqs, (\d+) gen reqs", text):
        it = int(m.group(1))
        ctx_reqs = int(m.group(2))
        gen_reqs = int(m.group(3))
        if ctx_reqs == target_ctx_reqs and gen_reqs == target_gen_reqs:
            graph_iters.append((start, end, it))
graph_iters = sorted(graph_iters)[args.warmup_times :]
graph_iter_list = [t[2] for t in graph_iters]
print("Iters (graph)", *graph_iter_list)
if iter_list != graph_iter_list:
    raise ValueError("The ID of iterations do not match")


def query_kernels(conn, iters):
    query = """SELECT name FROM sqlite_master WHERE type = ?"""
    df = pd.read_sql_query(query, conn, params=("table",))
    tables = df["name"].tolist()
    unified_subquery = """SELECT T1.start, T1.end, T1.demangledName, T1.correlationId, T1.graphNodeId
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS T1"""
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        unified_subquery += """ UNION ALL
            SELECT T2.start, T2.end, -2 AS demangledName, T2.correlationId, T2.graphNodeId
            FROM CUPTI_ACTIVITY_KIND_MEMCPY AS T2"""
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        unified_subquery += """ UNION ALL
            SELECT T3.start, T3.end, -3 AS demangledName, T3.correlationId, T3.graphNodeId
            FROM CUPTI_ACTIVITY_KIND_MEMSET AS T3"""
    query = f"""SELECT unified.start, unified.end, unified.graphNodeId, unified.demangledName,
        R.start AS runtime_start, R.end AS runtime_end
    FROM ({unified_subquery}) AS unified
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS R ON unified.correlationId = R.correlationId"""
    df = pd.read_sql_query(query, conn)
    per_iter_kernels = [[] for _ in iters]
    for start, end, graphNodeId, demangledName, runtime_start, runtime_end in df.itertuples(
        index=False
    ):
        it_idx = bisect.bisect(iters, (runtime_start,)) - 1
        if it_idx < 0 or runtime_end > iters[it_idx][1]:
            continue
        per_iter_kernels[it_idx].append((runtime_start, graphNodeId, start, end, demangledName))
    for kernels in per_iter_kernels:
        kernels.sort()
    return per_iter_kernels


eager_per_iter_kernels = query_kernels(eager_conn, eager_iters)
graph_per_iter_kernels = query_kernels(graph_conn, graph_iters)
print("#Kernels (eager)", *[len(kernels) for kernels in eager_per_iter_kernels])
print("#Kernels (graph)", *[len(kernels) for kernels in graph_per_iter_kernels])
for eager_kernels, graph_kernels in zip(eager_per_iter_kernels, graph_per_iter_kernels):
    assert all(a[4] == eager_per_iter_kernels[0][i][4] for i, a in enumerate(eager_kernels)), (
        "eager kernels change across iterations"
    )
    assert all(a[4] == graph_per_iter_kernels[0][i][4] for i, a in enumerate(graph_kernels)), (
        "graph kernels change across iterations"
    )

query = "SELECT * FROM StringIds"
df = pd.read_sql_query(query, eager_conn)
eager_string_ids = dict(zip(df["id"], df["value"]))
eager_string_ids.update({-2: "Memcpy", -3: "Memset"})
df = pd.read_sql_query(query, graph_conn)
graph_string_ids = dict(zip(df["id"], df["value"]))
graph_string_ids.update({-2: "Memcpy", -3: "Memset"})

eager_conn.close()
graph_conn.close()

eager_kernel_names = [eager_string_ids[kernel[4]] for kernel in eager_per_iter_kernels[0]]
graph_kernel_names = [graph_string_ids[kernel[4]] for kernel in graph_per_iter_kernels[0]]
super_kernel_names = shortest_common_supersequence(eager_kernel_names, graph_kernel_names)
print(f"#Kernels (supersequence) {len(super_kernel_names)}")
eager_per_layer_kernels = [[] for _ in layer_list]
for i, eager_kernel in enumerate(eager_per_iter_kernels[0]):
    eager_layers_idx = bisect.bisect(per_iter_eager_layers[0], (eager_kernel[0],)) - 1
    if eager_layers_idx < 0 or eager_kernel[0] > per_iter_eager_layers[0][eager_layers_idx][1]:
        continue
    eager_per_layer_kernels[eager_layers_idx].append(i)
eager2super = []
j = 0
for i, eager_kernel_name in enumerate(eager_kernel_names):
    while eager_kernel_name != super_kernel_names[j]:
        j += 1
    eager2super.append(j)
    j += 1
super_per_layer_starts = [eager2super[a[0]] for a in eager_per_layer_kernels]
super_per_layer_ends = [eager2super[a[-1]] for a in eager_per_layer_kernels]
graph_per_layer_kernels = [[] for _ in layer_list]
j = 0
for i, graph_kernel_name in enumerate(graph_kernel_names):
    while graph_kernel_name != super_kernel_names[j]:
        j += 1
    layer_idx = bisect.bisect(super_per_layer_starts, j) - 1
    if layer_idx >= 0 and j <= super_per_layer_ends[layer_idx]:
        graph_per_layer_kernels[layer_idx].append(i)
    j += 1
timeline = []
first_kernel_idx = min(graph_per_layer_kernels[layer_idx][0] for layer_idx in args.layer_indices)
for layer_idx in args.layer_indices:
    for kernel_idx in graph_per_layer_kernels[layer_idx]:
        duration_list = []
        end_list = []
        for it_idx in range(len(graph_per_iter_kernels)):
            layer_start_time = graph_per_iter_kernels[it_idx][first_kernel_idx][2]
            kernel_start_time = graph_per_iter_kernels[it_idx][kernel_idx][2]
            kernel_end_time = graph_per_iter_kernels[it_idx][kernel_idx][3]
            duration_list.append(kernel_end_time - kernel_start_time)
            end_list.append(kernel_end_time - layer_start_time)
        timeline.append(
            {
                "name": graph_kernel_names[kernel_idx],
                "duration": np.mean(duration_list).tolist(),
                "end": np.mean(end_list).tolist(),
            }
        )
print(f"{'Kernel':40s} {'Duration':>8s} {'End':>8s}")
print("-" * (40 + 1 + 8 + 1 + 8))
for o in timeline:
    print(
        f"{kernel_short_name(o['name'])[:40]:40s} {o['duration'] / 1000.0:-8.1f} {o['end'] / 1000.0:-8.1f}"
    )
if args.error_on_unknown_kernel and warned_names:
    raise ValueError("Unknown kernel names encountered")

if args.output:
    if not args.output.endswith(".json"):
        raise ValueError("Output file name must be *.json")
    with open(args.output, "w") as f:
        json.dump([{"name": "parse_e2e", "timeline": timeline}], f)
