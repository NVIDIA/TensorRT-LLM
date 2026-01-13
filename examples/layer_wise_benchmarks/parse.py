import argparse
import bisect
import csv
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import jinja2
import numpy as np
import pandas as pd

# Parse cmdline
parser = argparse.ArgumentParser()
parser.add_argument("--file-path", type=str)
parser.add_argument("--profile-dir", type=str, default="profiles")
parser.add_argument("--world-size", "--np", type=int)
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
if (args.file_path is None) == (args.world_size is None):
    parser.error("Please specify exactly one of --file-path and --world-size.")
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


if args.file_path is not None:
    nsys_rep_file_path = Path(args.file_path)
    if not nsys_rep_file_path.name.endswith(".nsys-rep"):
        raise ValueError("Expect a .nsys-rep file")
else:
    profile_dir = Path(args.profile_dir)
    nsys_rep_file_path = profile_dir / f"report_np{args.world_size}_rank{args.rank}.nsys-rep"
sqlite_file_path = nsys_rep_file_path.parent / (
    nsys_rep_file_path.name[: -len(".nsys-rep")] + ".sqlite"
)
csv_file_path = nsys_rep_file_path.parent / (nsys_rep_file_path.name[: -len(".nsys-rep")] + ".csv")
html_file_path = nsys_rep_file_path.parent / (
    nsys_rep_file_path.name[: -len(".nsys-rep")] + ".html"
)
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
                "kernel_count_per_range": [],
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
        problem_set[problem_id]["kernel_count_per_range"].append(0)

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
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS R ON unified.graphNodeId IS NOT NULL AND
                                             unified.correlationId = R.correlationId
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
    if run_id == -1 or runtime_start >= problem["runs_end"][run_id]:
        continue
    ranges = [
        i
        for i, (range_start, range_end, text) in enumerate(problem["ranges"])
        if capture_start >= range_start and capture_end <= range_end
    ]
    for range_id in ranges:
        problem["kernel_count_per_range"][range_id] += 1
    range_names = [problem["ranges"][i][2] for i in ranges]
    if args.module is None or args.module in range_names:
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

# Check ambiguous modules
if args.module:
    for problem in problem_set:
        num_matches_per_run = [0] * (len(problem["runs"]) + 1)
        for (range_start, _, text), kernel_count in zip(
            problem["ranges"], problem["kernel_count_per_range"]
        ):
            if text == args.module and kernel_count > 0:
                num_matches_per_run[bisect.bisect(problem["runs"], range_start)] += 1
        for run_id_plus_one, num_matches in enumerate(num_matches_per_run):
            if num_matches > 1:
                raise ValueError(
                    f'Module is ambiguous: "{args.module}" appears {num_matches} times'
                    f' in "{problem["text"]}"\'s {run_id_plus_one}-th run'
                )

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
    kernels[problem_id][run_id].append((demangledName, start, end, ranges))
for problem_id in range(len(kernels)):
    required_seq = [demangledName for demangledName, _, _, _ in kernels[problem_id][0]]
    for run_id in range(len(kernels[problem_id])):
        seq = [demangledName for demangledName, _, _, _ in kernels[problem_id][run_id]]
        assert seq == required_seq


parser_keywords = [
    ("cuBLASGemm", "nvjet"),
    ("cutlassGroupGemm", "cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal"),
    ("cutlassGemm", "GemmUniversal"),
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
    ("torchAdd", "at::native::CUDAFunctorOnSelf_add"),
    ("torchAdd", "CUDAFunctor_add"),
    ("torchClamp", "at::native::<unnamed>::launch_clamp_scalar("),
    ("torchCompare", "at::native::<unnamed>::CompareFunctor<"),
    ("torchCopy", "at::native::bfloat16_copy_kernel_cuda"),
    ("torchCopy", "at::native::direct_copy_kernel_cuda("),
    ("torchFill", "at::native::FillFunctor"),
    ("torchIndexPut", "at::native::index_put_kernel_impl<"),
    ("torchMul", "at::native::binary_internal::MulFunctor<"),
    ("torchPow", "at::native::<unnamed>::pow_tensor_scalar_kernel_impl<"),
    ("torchReduceSum", ["at::native::reduce_kernel<", "at::native::sum_functor<"]),
    ("torchSigmoid", "at::native::sigmoid_kernel_cuda"),
    ("torchWhere", "at::native::<unnamed>::where_kernel_impl("),
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
    if re.search(r"at::native::.*elementwise_kernel<", name):
        if name not in warned_names:
            print(f"Not parsed torch kernel name: {name}", file=sys.stderr)
            warned_names.add(name)
    assert "!unnamed!" not in name
    name = name.replace("<unnamed>", "!unnamed!")
    if "<" in name:
        name = name[: name.index("<")]
    if "(" in name:
        name = name[: name.index("(")]
    if "::" in name:
        name = name[name.rindex("::") + 2 :]
    name = name.replace("!unnamed!", "<unnamed>")
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
if args.error_on_unknown_kernel and warned_names:
    raise ValueError("Unknown kernel names encountered")

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
        f"    Ranges: [{', '.join(text for _, end, text in problem['ranges'] if end <= problem['runs_end'][0])}]"
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
js_header_config = []
for problem in problem_set:
    innermost_children = js_header_config
    for k, msg_prefix in [
        ("batch_size", "b="),
        ("seq_len_q", "q="),
        ("seq_len_kv_cache", "past="),
    ]:
        if len(run_args[k + "_list"]) > 1:
            if len(innermost_children) == 0 or problem["spec"][k] != innermost_children[-1][k]:
                innermost_children.append(
                    {
                        "name": msg_prefix + str(problem["spec"][k]),
                        "children": [],
                        k: problem["spec"][k],
                    }
                )
            innermost_children = innermost_children[-1]["children"]
    innermost_children.append({"name": problem["text"]})
loader = jinja2.FileSystemLoader(Path(__file__).parent)
template = jinja2.Environment(loader=loader).get_template("template.html")
with html_file_path.open("w") as f:
    configText = (
        "Run:\n"
        + json.dumps(run_args, indent=4)
        + "\n\nParse:\n"
        + json.dumps(args.__dict__, indent=4)
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
