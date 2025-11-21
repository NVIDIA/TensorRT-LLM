import argparse
import bisect
import json
import re
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# Parse cmdline
parser = argparse.ArgumentParser()
parser.add_argument("--profile-dir", type=str, default="profiles")
parser.add_argument("--world-size", type=int)
parser.add_argument("--rank", type=int, default=0)
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument("--unknown-kernel-as-error", action="store_true", dest="unknown_kernel_as_error")
group.add_argument(
    "--no-unknown-kernel-as-error", action="store_false", dest="unknown_kernel_as_error"
)
args = parser.parse_args()


def lazy_convert_sqlite(nsys_rep_file_path, sqlite_file_path):
    if (
        not sqlite_file_path.is_file()
        or nsys_rep_file_path.stat().st_mtime > sqlite_file_path.stat().st_mtime
    ):
        subprocess.check_call(
            [
                "nsys",
                "export",
                "--type",
                "sqlite",
                "-o",
                sqlite_file_path,
                "--force-overwrite=true",
                nsys_rep_file_path,
            ]
        )


def shortest_common_supersequence(a, b):
    # Merge two lists into their shortest common supersequence,
    # so that both `a` and `b` are subsequences of the result.
    # Uses dynamic programming to compute the shortest common supersequence, then reconstructs it.
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    # Backtrack to build the merged sequence
    res = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            res.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] < dp[i][j - 1]:
            res.append(a[i - 1])
            i -= 1
        else:
            res.append(b[j - 1])
            j -= 1
    while i > 0:
        res.append(a[i - 1])
        i -= 1
    while j > 0:
        res.append(b[j - 1])
        j -= 1
    res.reverse()
    return res


nsys_rep_file_path = Path(args.profile_dir) / f"report_np{args.world_size}_rank{args.rank}.nsys-rep"
sqlite_file_path = Path(args.profile_dir) / f"report_np{args.world_size}_rank{args.rank}.sqlite"
lazy_convert_sqlite(nsys_rep_file_path, sqlite_file_path)

conn = sqlite3.connect(f"file:{sqlite_file_path}?mode=ro", uri=True)

query = "SELECT * FROM ENUM_NSYS_EVENT_TYPE"
df = pd.read_sql_query(query, conn)
event_id_NvtxDomainCreate = df[df["name"] == "NvtxDomainCreate"].iloc[0]["id"].tolist()
event_id_NvtxPushPopRange = df[df["name"] == "NvtxPushPopRange"].iloc[0]["id"].tolist()

query = "SELECT domainId FROM NVTX_EVENTS WHERE eventType = ? AND text = ?"
df = pd.read_sql_query(query, conn, params=(event_id_NvtxDomainCreate, "NCCL"))
nccl_domain_id = -1 if df.empty else df.iloc[0]["domainId"].tolist()

query = """SELECT T1.start, T2.value AS text
    FROM NVTX_EVENTS AS T1
    JOIN StringIds AS T2 ON T1.textId = T2.id
    WHERE eventType = ? AND T2.value LIKE ?"""
df = pd.read_sql_query(query, conn, params=(event_id_NvtxPushPopRange, "layer_wise_benchmarks %"))
problem_start = []
problem_set = []
for start, text in df.itertuples(index=False):
    if text.startswith("layer_wise_benchmarks args {"):
        run_args = json.loads(text[len("layer_wise_benchmarks args") :])
    elif text.startswith("layer_wise_benchmarks problem_spec {"):
        problem_start.append(start)
        problem_set.append(
            {
                "spec": json.loads(text[len("layer_wise_benchmarks problem_spec") :]),
                "text": "",
                "runs": [],
                "runs_end": [],
                "ranges": [],
            }
        )

query = """SELECT T1.start, T1.end, T2.value AS text
    FROM NVTX_EVENTS AS T1
    JOIN StringIds AS T2 ON T1.textId = T2.id
    WHERE eventType = ? AND T2.value NOT LIKE ? AND domainId != ?"""
df = pd.read_sql_query(
    query, conn, params=(event_id_NvtxPushPopRange, "layer_wise_benchmarks %", nccl_domain_id)
)
for start, end, text in df.itertuples(index=False):
    problem_id = bisect.bisect(problem_start, start) - 1
    if re.match(r"b=\d+ s=\d+ ", text):
        problem_set[problem_id]["text"] = text
        problem_set[problem_id]["runs"].append(start)
        problem_set[problem_id]["runs_end"].append(end)
    else:
        problem_set[problem_id]["ranges"].append((start, end, text))

query = """SELECT unified.start, unified.end, unified.demangledName,
       R.start AS runtime_start, R.end AS runtime_end,
       CGE2.start AS capture_start, CGE2.end AS capture_end
FROM (
    SELECT T1.start, T1.end, T1.demangledName, T1.correlationId, T1.graphNodeId
    FROM CUPTI_ACTIVITY_KIND_KERNEL AS T1
    UNION ALL
    SELECT T2.start, T2.end, -2 AS demangledName, T2.correlationId, T2.graphNodeId
    FROM CUPTI_ACTIVITY_KIND_MEMCPY AS T2
    UNION ALL
    SELECT T3.start, T3.end, -3 AS demangledName, T3.correlationId, T3.graphNodeId
    FROM CUPTI_ACTIVITY_KIND_MEMSET AS T3
) AS unified
JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS R ON unified.correlationId = R.correlationId
LEFT JOIN CUDA_GRAPH_NODE_EVENTS AS CGE1 ON unified.graphNodeId = CGE1.graphNodeId AND
                                            CGE1.originalGraphNodeId IS NOT NULL
LEFT JOIN CUDA_GRAPH_NODE_EVENTS AS CGE2 ON CGE1.originalGraphNodeId = CGE2.graphNodeId"""
df = pd.read_sql_query(query, conn)
kernel_list = []
for (
    start,
    end,
    demangledName,
    runtime_start,
    runtime_end,
    capture_start,
    capture_end,
) in df.itertuples(index=False):
    problem_id = bisect.bisect(problem_start, start) - 1
    run_id = bisect.bisect(problem_set[problem_id]["runs"], runtime_start) - 1
    if (
        run_id == -1
        or run_id == len(problem_set[problem_id]["runs"])
        or runtime_start >= problem_set[problem_id]["runs_end"][run_id]
    ):
        run_id = -1
    ranges = [
        text
        for range_start, range_end, text in problem_set[problem_id]["ranges"]
        if capture_start >= range_start and capture_end <= range_end
    ]
    kernel_list.append(
        (
            problem_id,
            run_id,
            ranges,
            start,
            end,
            demangledName,
            runtime_start,
            runtime_end,
            capture_start,
            capture_end,
        )
    )

query = "SELECT * FROM StringIds"
df = pd.read_sql_query(query, conn)
string_ids = dict(zip(df["id"], df["value"]))

conn.close()

kernel_list.sort(key=lambda t: (t[6], t[8]))
kernels = [[[] for _ in problem["runs"]] for problem in problem_set]
for (
    problem_id,
    run_id,
    ranges,
    start,
    end,
    demangledName,
    runtime_start,
    runtime_end,
    capture_start,
    capture_end,
) in kernel_list:
    if run_id != -1:
        kernels[problem_id][run_id].append((demangledName, start, end, ranges))
for problem_id in range(len(kernels)):
    required_seq = [demangledName for demangledName, _, _, _ in kernels[problem_id][0]]
    for run_id in range(len(kernels[problem_id])):
        seq = [demangledName for demangledName, _, _, _ in kernels[problem_id][run_id]]
        assert seq == required_seq


parser_keywords = [
    ("Gemm", "nvjet"),
    ("RMSNorm", "RMSNormKernel"),
    ("Cat", "CatArrayBatchedCopy"),
    ("RoPE", "applyMLARope"),
    ("fmha", "fmhaSm100fKernel_Qkv"),
    ("fmhaReduction", "fmhaReductionKernel"),
    ("quant", "quantize_with_block_size"),
    ("expandInput", "expandInputRowsKernel"),
    ("computeStrides", "computeStridesTmaWarpSpecializedKernel"),
    ("GroupGemm", "cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal"),
    ("doActivation", "doActivationKernel"),
    ("Gemm", "GemmUniversal"),
    ("splitKreduce", "splitKreduce_kernel"),
    ("topk", "deepseek_v3_topk_kernel"),
    ("AllGather", "ncclDevKernel_AllGather_"),
    ("ReduceScatter", "ncclDevKernel_ReduceScatter_"),
    ("CountAndIndice", "computeCountAndIndiceDevice"),
    ("Cumsum", "computeCumsumDevice"),
    ("moveIndice", "moveIndiceDevice"),
    ("AllToAll", "moeAllToAllKernel"),
    ("memsetExpertIds", "memsetExpertIdsDevice"),
    ("blockSum", "blockExpertPrefixSumKernel"),
    ("globalSum", "globalExpertPrefixSumKernel"),
    ("mergePrefix", "mergeExpertPrefixSumKernel"),
    ("fusedBuildExpertMaps", "fusedBuildExpertMapsSortFirstTokenKernel"),
    ("swiglu", "silu_and_mul_kernel"),
    ("add", "CUDAFunctor_add"),
    ("Fill", "at::native::FillFunctor"),
    ("red_fused_add_sum", "triton_red_fused_add_sum_0"),
]
warned_names = set()


def parse_kernel_name(demangledName):
    if demangledName == -2:
        return "Memcpy"
    if demangledName == -3:
        return "Memset"
    name = string_ids[demangledName]
    for dst, src in parser_keywords:
        if src in name:
            return dst
    if name not in warned_names:
        print(f"Unknown kernel name: {name}")
        warned_names.add(name)
        if args.unknown_kernel_as_error:
            raise NotImplementedError(f"Unknown kernel name: {name}")
    return name[:20]


converted_seqs = []
for runs in kernels:
    converted_seq = []
    for i, (demangledName, _, _, ranges) in enumerate(runs[0]):
        # TODO: Group by ranges
        name = parse_kernel_name(demangledName)
        if ranges:
            name = f"{ranges[0]}.{name}"
        time_list = [run[i][2] - run[i][1] for run in runs]
        time_list = time_list[run_args["warmup_times"] :]
        time = np.mean(time_list).tolist()
        converted_seq.append((name, time))
    converted_seqs.append(converted_seq)

merged_title = []
for converted_seq in converted_seqs:
    title = [name for name, _ in converted_seq]
    merged_title = shortest_common_supersequence(merged_title, title)
print(merged_title)

print("Problem set:")
for problem in problem_set:
    print(
        f'- "{problem["text"]}"    {len(problem["runs"])} runs'
        f"    Ranges: [{', '.join(text for _, _, text in problem['ranges'])}]"
    )
