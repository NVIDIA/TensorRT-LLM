import argparse
import bisect
import csv
import json
import re
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path

import jinja2
import numpy as np
import pandas as pd

# Parse cmdline
parser = argparse.ArgumentParser()
parser.add_argument("--profile-dir", type=str, default="profiles")
parser.add_argument("--world-size", "--np", type=int, required=True)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--warmup-times", type=int)
parser.add_argument("--module", type=str)
parser.add_argument("--query", type=str)
group = parser.add_mutually_exclusive_group()
group.add_argument("--error-on-unknown-kernel", action="store_true", dest="error_on_unknown_kernel")
group.add_argument(
    "--no-error-on-unknown-kernel", action="store_false", dest="error_on_unknown_kernel"
)
parser.set_defaults(error_on_unknown_kernel=False)
args = parser.parse_args()
print(args)


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


profile_dir = Path(args.profile_dir)
nsys_rep_file_path = profile_dir / f"report_np{args.world_size}_rank{args.rank}.nsys-rep"
sqlite_file_path = profile_dir / f"report_np{args.world_size}_rank{args.rank}.sqlite"
csv_file_path = profile_dir / f"report_np{args.world_size}_rank{args.rank}.csv"
html_file_path = profile_dir / f"report_np{args.world_size}_rank{args.rank}.html"
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
                "range_in_module": [],
            }
        )

query = """SELECT T1.start, T1.end, T2.value AS text
    FROM NVTX_EVENTS AS T1
    JOIN StringIds AS T2 ON T1.textId = T2.id
    WHERE eventType = ? AND T2.value NOT LIKE ? AND T2.value NOT LIKE ? AND domainId != ?"""
df = pd.read_sql_query(
    query,
    conn,
    params=(event_id_NvtxPushPopRange, "layer_wise_benchmarks %", "[DG]%", nccl_domain_id),
)
for start, end, text in df.itertuples(index=False):
    problem_id = bisect.bisect(problem_start, start) - 1
    assert problem_id != -1
    if re.match(r"b=\d+ s=\d+ ", text):
        problem_set[problem_id]["text"] = text
        problem_set[problem_id]["runs"].append(start)
        problem_set[problem_id]["runs_end"].append(end)
    else:
        problem_set[problem_id]["ranges"].append((start, end, text))

# Determine whether each range is the first range that matches `args.module`,
# and store the result in `problem["range_in_module"]`
for problem in problem_set:
    if args.module is not None:
        problem["range_in_module"] = [False] * len(problem["ranges"])
        run_ids = [bisect.bisect(problem["runs"], start) - 1 for start, _, _ in problem["ranges"]]
        run2ranges = defaultdict(list)
        for i, run_id in enumerate(run_ids):
            run2ranges[run_id].append(i)
        for run_id, ranges in run2ranges.items():
            ranges = sorted(ranges, key=lambda i: problem["ranges"][i][0])
            num_matches = 0
            for range_id in ranges:
                if problem["ranges"][range_id][2] == args.module:
                    problem["range_in_module"][range_id] = True
                    num_matches += 1
            if num_matches != 1:
                raise ValueError(
                    f'Module "{args.module}" appears {num_matches} times'
                    f' in "{problem["text"]}"\'s {run_id + 1}-th run'
                )

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
query = f"""SELECT unified.start, unified.end, unified.demangledName,
       R.start AS runtime_start, R.end AS runtime_end,
       R.start AS capture_start, R.end AS capture_end
FROM ({unified_subquery}) AS unified
JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS R ON unified.correlationId = R.correlationId
WHERE unified.graphNodeId IS NULL"""
if "CUDA_GRAPH_NODE_EVENTS" in tables:
    query += f""" UNION ALL
    SELECT unified.start, unified.end, unified.demangledName,
           R.start AS runtime_start, R.end AS runtime_end,
           CGE2.start AS capture_start, CGE2.end AS capture_end
    FROM ({unified_subquery}) AS unified
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
    problem = problem_set[problem_id]
    run_id = bisect.bisect(problem["runs"], runtime_start) - 1
    if (
        run_id == -1
        or run_id == len(problem["runs"])
        or runtime_start >= problem["runs_end"][run_id]
    ):
        run_id = -1
    ranges = [
        i
        for i, (range_start, range_end, text) in enumerate(problem["ranges"])
        if capture_start >= range_start and capture_end <= range_end
    ]
    if args.module is None or any(problem["range_in_module"][i] for i in ranges):
        range_names = [problem["ranges"][i][2] for i in ranges]
        kernel_list.append(
            (
                problem_id,
                run_id,
                range_names,
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
    ("cuBLASGemm", "nvjet"),
    ("splitKreduce", "splitKreduce_kernel"),
    ("fusedAGemm", "fused_a_gemm_kernel"),
    ("RMSNorm", "RMSNormKernel"),
    ("torchCat", "CatArrayBatchedCopy"),
    ("applyMLARope", "applyMLARope"),
    ("fmhaSm100f", "fmhaSm100fKernel_Qkv"),
    ("fmhaReduction", "fmhaReductionKernel"),
    ("quant", "quantize_with_block_size"),
    ("AllGather", "ncclDevKernel_AllGather_"),
    ("ReduceScatter", "ncclDevKernel_ReduceScatter_"),
    ("allreduce_oneshot", "allreduce_fusion_kernel_oneshot_lamport"),
    ("allreduce_twoshot", "allreduce_fusion_kernel_twoshot_sync"),
    ("expandInput", "expandInputRowsKernel"),
    ("computeStrides", "computeStridesTmaWarpSpecializedKernel"),
    ("cutlassGroupGemm", "cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal"),
    ("doActivation", "doActivationKernel"),
    ("cutlassGemm", "GemmUniversal"),
    ("deepseek_v3_topk", "deepseek_v3_topk_kernel"),
    ("CountAndIndice", "computeCountAndIndiceDevice"),
    ("Cumsum", "computeCumsumDevice"),
    ("moveIndice", "moveIndiceDevice"),
    ("moeAllToAll", "moeAllToAllKernel"),
    ("moeA2APrepareDispatch", "moe_comm::moeA2APrepareDispatchKernel"),
    ("moeA2ADispatch", "moe_comm::moeA2ADispatchKernel"),
    ("moeA2ASanitizeExpertIds", "moe_comm::moeA2ASanitizeExpertIdsKernel"),
    ("moeA2APrepareCombine", "moe_comm::moeA2APrepareCombineKernel"),
    ("moeA2ACombine", "moe_comm::moeA2ACombineKernel"),
    ("memsetExpertIds", "memsetExpertIdsDevice"),
    ("blockSum", "blockExpertPrefixSumKernel"),
    ("globalSum", "globalExpertPrefixSumKernel"),
    ("globalSumLarge", "globalExpertPrefixSumLargeKernel"),
    ("mergePrefix", "mergeExpertPrefixSumKernel"),
    ("fusedBuildExpertMaps", "fusedBuildExpertMapsSortFirstTokenKernel"),
    ("swiglu", "silu_and_mul_kernel"),
    ("torchAdd", "CUDAFunctor_add"),
    ("torchFill", "at::native::FillFunctor"),
    ("triton_fused_add_sum", "triton_red_fused_add_sum_0"),
    ("torchCopy", "at::native::bfloat16_copy_kernel_cuda"),
    ("torchDistribution", "distribution_elementwise_grid_stride_kernel"),
    ("torchArange", "at::native::arange_cuda_out"),
    ("torchDirectCopy", "at::native::direct_copy_kernel_cuda"),
    ("torchBitonicSort", "at::native::bitonicSortKVInPlace"),
    ("routingInitExpertCounts", "routingInitExpertCounts"),
    ("routingIndicesCluster", "routingIndicesClusterKernel"),
    ("routingIndicesCoop", "routingIndicesCoopKernel"),
    ("bmm_4_44_32", "bmm_E2m1_E2m1E2m1_Fp32_t"),
    ("finalize", "finalize::finalizeKernel"),
    ("bmm_16_44_32", "bmm_Bfloat16_E2m1E2m1_Fp32_"),
    ("deep_gemm_gemm", "deep_gemm::sm100_fp8_gemm_1d1d_impl<"),
    ("per_token_quant", "_per_token_quant_and_transform_kernel"),
    ("triton_fused_layer_norm", "triton_per_fused__to_copy_native_layer_norm_0"),
    ("flashinferRoPE", "flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<"),
    ("flashinferRoPE", "flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheKernel<"),
    ("fp8_blockscale_gemm", "tensorrt_llm::kernels::fp8_blockscale_gemm"),
    ("triton_fused_mul_squeeze", "triton_poi_fused_mul_squeeze_0"),
    ("indexerKCacheScatter", "tensorrt_llm::kernels::indexerKCacheScatterUnifiedKernel"),
    ("deep_gemm_mqa_logits", "deep_gemm::sm100_fp8_paged_mqa_logits<"),
    ("topKPerRowDecode", "tensorrt_llm::kernels::topKPerRowDecode<"),
    ("torchAdd<int>", "at::native::CUDAFunctorOnSelf_add"),
    ("convert_req_index", "_convert_req_index_to_global_index_kernel_with_stride_factor"),
    ("preprocess_after_permute", "_preprocess_after_permute_kernel"),
    ("masked_index_copy_quant", "_masked_index_copy_group_quant_fp8"),
    ("swiglu_quant", "_silu_and_mul_post_quant_kernel"),
    ("masked_index_gather", "masked_index_gather_kernel"),
    ("finalizeMoeRouting", "tensorrt_llm::kernels::cutlass_kernels::finalizeMoeRoutingKernel<"),
    ("fused_qkvzba_split", "fused_qkvzba_split_reshape_cat_kernel"),
    ("causal_conv1d_update", "tensorrt_llm::kernels::causal_conv1d::causal_conv1d_update_kernel<"),
    ("fused_delta_rule_update", "fused_sigmoid_gating_delta_rule_update_kernel"),
    ("layer_norm_fwd_1pass", "_layer_norm_fwd_1pass_kernel"),
    ("torchGatherTopK", "at::native::sbtopk::gatherTopK<"),
    ("softmax_warp_forward", "softmax_warp_forward<"),
    ("torchSigmoid", "at::native::sigmoid_kernel_cuda"),
    ("torchMul", "at::native::binary_internal::MulFunctor<"),
    ("computeSeqAndPaddingOffsets", "tensorrt_llm::kernels::computeSeqAndPaddingOffsets<"),
    ("applyBiasRopeUpdateKVCache", "tensorrt_llm::kernels::applyBiasRopeUpdateKVCacheV2<"),
    ("routingIndicesHistogramScores", "routingRenormalize::routingIndicesHistogramScoresKernel<"),
    ("routingIndicesHistogram", "routingIndicesHistogramKernel<"),
    ("routingIndicesOffsets", "routingIndicesOffsetsKernel<"),
    ("torchReduceSum", ["at::native::reduce_kernel<", "at::native::sum_functor<"]),
    ("CuteDSLMoePermute", "cute_dsl::moePermuteKernel"),
    (
        "CuteDSLGemm",
        ["cute_dsl_kernels", "blockscaled_gemm_persistent"],
    ),
    (
        "CuteDSLGroupedGemmSwiglu",
        ["cute_dsl_kernels", "blockscaled_contiguous_grouped_gemm_swiglu_fusion"],
    ),
    (
        "CuteDSLGroupedGemmFinalize",
        ["cute_dsl_kernels", "blockscaled_contiguous_grouped_gemm_finalize_fusion"],
    ),
]
warned_names = set()


def parse_kernel_name(demangledName):
    if demangledName == -2:
        return "Memcpy"
    if demangledName == -3:
        return "Memset"
    name = string_ids[demangledName]
    for dst, src in parser_keywords:
        if not isinstance(src, (tuple, list)):
            src = [src]
        if all(keyword in name for keyword in src):
            return dst
    if name not in warned_names:
        print(f"Unknown kernel name: {name}")
        warned_names.add(name)
        if args.error_on_unknown_kernel:
            raise NotImplementedError(f"Unknown kernel name: {name}")
    if "<" in name:
        name = name[: name.index("<")]
    if "(" in name:
        name = name[: name.index("(")]
    return name


converted_seqs = []
for runs in kernels:
    warmup_times = run_args["warmup_times"] if args.warmup_times is None else args.warmup_times
    converted_seq = []
    # Kernel time
    for i, (demangledName, _, _, ranges) in enumerate(runs[0]):
        name = parse_kernel_name(demangledName)
        category = (*ranges, name)
        time_list = [run[i][2] - run[i][1] for run in runs]
        t = np.mean(time_list[warmup_times:]).tolist()
        converted_seq.append((category, t))
    # Space and Overlap
    overlap_list = []
    space_list = []
    for run in runs:
        sorted_run = sorted(run, key=lambda op: op[1])
        last_end = sorted_run[0][1]
        overlap_time = 0
        space_time = 0
        for _, start, end, _ in sorted_run:
            if start > last_end:
                space_time += start - last_end
            else:
                overlap_time += min(last_end, end) - start
            last_end = max(last_end, end)
        overlap_list.append(-overlap_time)
        space_list.append(space_time)
    converted_seq.append((("Overlap",), np.mean(overlap_list[warmup_times:]).tolist()))
    converted_seq.append((("Space",), np.mean(space_list[warmup_times:]).tolist()))
    converted_seq.append((("Total",), sum(t for _, t in converted_seq)))
    converted_seqs.append(converted_seq)

merged_title = []
for converted_seq in converted_seqs:
    title = [name for name, _ in converted_seq]
    merged_title = shortest_common_supersequence(merged_title, title)

merged_data = [[0.0] * len(problem_set) for _ in merged_title]
for problem_id, converted_seq in enumerate(converted_seqs):
    cur = 0
    for category, t in converted_seq:
        cur = merged_title.index(category, cur)
        merged_data[cur][problem_id] = t
        cur += 1

print("Run args:")
print(run_args)

print("Problem set:")
for problem in problem_set:
    print(
        f'- "{problem["text"]}"    {len(problem["runs"])} runs'
        f"    Ranges: [{', '.join(text for _, _, text in problem['ranges'])}]"
    )

stack = []
csv_data = [["", *[problem["text"] for problem in problem_set]]]
js_data = []
js_stack = [js_data]
max_title_len = max((len(title) - 1) * 3 + len(title[-1][:40]) for title in merged_title)
for title, time_data in zip(merged_title, merged_data):
    while stack != list(title[: len(stack)]):
        level_title = stack[-1]
        stack.pop()
        js_stack[-2].append(
            {
                "name": level_title,
                "children": js_stack[-1],
            }
        )
        js_stack.pop()
    while len(stack) != len(title) - 1:
        level_title = title[len(stack)]
        stack.append(level_title)
        level = len(stack)
        print("|--" * (level - 1) + level_title)
        csv_data.append(["|--" * (level - 1) + level_title] + [""] * len(problem_set))
        js_stack.append([])
    level = len(stack) + 1
    print(
        "|--" * (level - 1)
        + title[-1][:40]
        + " " * (max_title_len - (level - 1) * 3 - len(title[-1][:40])),
        *[f"{x / 1000:-6.1f}" for x in time_data],
    )
    csv_data.append(["|--" * (level - 1) + title[-1], *[f"{x / 1000:.1f}" for x in time_data]])
    if title != ("Total",):
        js_stack[-1].append(
            {
                "name": title[-1],
                "time": [x / 1000 for x in time_data],
            }
        )
# TODO: Group repeated modules
with csv_file_path.open("w", newline="") as f:
    csv_writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    for row in csv_data:
        csv_writer.writerow(row)
js_header_config = [{"name": problem["text"]} for problem in problem_set]
loader = jinja2.FileSystemLoader(Path(__file__).parent)
template = jinja2.Environment(loader=loader).get_template("template.html")
with html_file_path.open("w") as f:
    configText = (
        "Run:\n" + json.dumps(run_args, indent=4) + "\n\nParse:\n" + json.dumps(args.__dict__)
    )
    f.write(template.render(headerConfig=js_header_config, rawData=js_data, configText=configText))

if args.query is not None:
    print("Query:")
    for query in args.query.split(","):
        query = query.strip()
        query_matched = [0.0] * len(problem_set)
        for title, time_data in zip(merged_title, merged_data):
            if query in ".".join(title):
                for i, x in enumerate(time_data):
                    query_matched[i] += x
        print(
            query + " " * (max_title_len - len(query)),
            *[f"{x / 1000:-6.1f}" for x in query_matched],
        )
