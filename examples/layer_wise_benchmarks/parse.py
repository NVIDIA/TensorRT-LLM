import argparse
import bisect
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import NamedTuple

import jinja2
import numpy as np
import pandas as pd
from parser_utils import (
    kernel_short_name,
    lazy_convert_sqlite,
    shortest_common_supersequence,
    warned_names,
)


class NvtxRange(NamedTuple):
    """Represents an NVTX range with start/end times and text label."""

    start: int
    end: int
    text: str


class KernelRecord(NamedTuple):
    """Represents a kernel record from the database query.

    Used for sorting and grouping kernels by runtime and capture time.
    """

    problem_id: int
    run_id: int
    range_names: tuple[str, ...]
    kernel_start: int
    kernel_end: int
    demangled_name: int  # String ID reference
    runtime_start: int
    capture_start: int


class KernelTiming(NamedTuple):
    """Represents a kernel's timing within a run.

    Used after sorting and grouping for per-run analysis.
    """

    demangled_name: int  # String ID reference
    kernel_start: int
    kernel_end: int
    range_names: tuple[str, ...]


class CategoryTime(NamedTuple):
    """Represents a category (hierarchical path) and its associated time."""

    category: tuple[str, ...]
    time_ns: float


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
json_file_path = nsys_rep_file_path.parent / (
    nsys_rep_file_path.name[: -len(".nsys-rep")] + ".json"
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
problem_start_times: list[int] = []
problem_set: list[dict] = []
for start, text in df.itertuples(index=False):
    if text.startswith("layer_wise_benchmarks args {"):
        run_args = json.loads(text[len("layer_wise_benchmarks args") :])
    elif text.startswith("layer_wise_benchmarks problem_spec {"):
        problem_start_times.append(start)
        problem_set.append(
            {
                "spec": json.loads(text[len("layer_wise_benchmarks problem_spec ") :]),
                "text": "",
                "run_starts": [],
                "run_ends": [],
                "ranges": [],
                "kernel_count_per_range": [],
            }
        )

query = """SELECT T1.start, T1.end, T2.value AS text
    FROM NVTX_EVENTS AS T1
    JOIN StringIds AS T2 ON T1.textId = T2.id
    WHERE eventType = ? AND T2.value NOT LIKE ? AND domainId != ?"""
df = pd.read_sql_query(
    query,
    conn,
    params=(event_id_NvtxPushPopRange, "[DG]%", nccl_domain_id),
)
for start, end, text in df.itertuples(index=False):
    problem_id = bisect.bisect(problem_start_times, start) - 1
    if text.startswith("layer_wise_benchmarks "):
        if text != "layer_wise_benchmarks ignore":
            continue
    else:
        assert problem_id != -1
    if re.match(r"b=\d+ s=\d+ ", text):
        problem_set[problem_id]["text"] = text
        problem_set[problem_id]["run_starts"].append(start)
        problem_set[problem_id]["run_ends"].append(end)
    else:
        problem_set[problem_id]["ranges"].append(NvtxRange(start, end, text))
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
       R.start AS runtime_start, R.start AS capture_start, R.end AS capture_end
FROM ({unified_subquery}) AS unified
JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS R ON unified.correlationId = R.correlationId
WHERE unified.graphNodeId IS NULL"""
if "CUDA_GRAPH_NODE_EVENTS" in tables:
    query += f""" UNION ALL
    SELECT unified.start, unified.end, unified.demangledName,
           R.start AS runtime_start, CGE2.start AS capture_start, CGE2.end AS capture_end
    FROM ({unified_subquery}) AS unified
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS R ON unified.graphNodeId IS NOT NULL AND
                                             unified.correlationId = R.correlationId
    LEFT JOIN CUDA_GRAPH_NODE_EVENTS AS CGE1 ON unified.graphNodeId = CGE1.graphNodeId AND
                                                CGE1.originalGraphNodeId IS NOT NULL
    LEFT JOIN CUDA_GRAPH_NODE_EVENTS AS CGE2 ON CGE1.originalGraphNodeId = CGE2.graphNodeId"""
df = pd.read_sql_query(query, conn)
kernel_records: list[KernelRecord] = []
for (
    kernel_start,
    kernel_end,
    demangled_name,
    runtime_start,
    capture_start,
    capture_end,
) in df.itertuples(index=False):
    problem_id = bisect.bisect(problem_start_times, kernel_start) - 1
    problem = problem_set[problem_id]
    run_id = bisect.bisect(problem["run_starts"], runtime_start) - 1
    if run_id == -1 or runtime_start >= problem["run_ends"][run_id]:
        continue
    matching_range_indices = [
        i
        for i, nvtx_range in enumerate(problem["ranges"])
        if capture_start >= nvtx_range.start and capture_end <= nvtx_range.end
    ]
    for range_idx in matching_range_indices:
        problem["kernel_count_per_range"][range_idx] += 1
    range_names = tuple(problem["ranges"][i].text for i in matching_range_indices)
    if (
        args.module is None or args.module in range_names
    ) and "layer_wise_benchmarks ignore" not in range_names:
        kernel_records.append(
            KernelRecord(
                problem_id=problem_id,
                run_id=run_id,
                range_names=range_names,
                kernel_start=kernel_start,
                kernel_end=kernel_end,
                demangled_name=demangled_name,
                runtime_start=runtime_start,
                capture_start=capture_start,
            )
        )

query = "SELECT * FROM StringIds"
df = pd.read_sql_query(query, conn)
string_ids = dict(zip(df["id"], df["value"]))
string_ids.update({-2: "Memcpy", -3: "Memset"})

conn.close()

# Check ambiguous modules
if args.module:
    for problem in problem_set:
        num_matches_per_run = [0] * (len(problem["run_starts"]) + 1)
        for nvtx_range, kernel_count in zip(problem["ranges"], problem["kernel_count_per_range"]):
            if nvtx_range.text == args.module and kernel_count > 0:
                num_matches_per_run[bisect.bisect(problem["run_starts"], nvtx_range.start)] += 1
        for run_id_plus_one, num_matches in enumerate(num_matches_per_run):
            if num_matches > 1:
                raise ValueError(
                    f'Module is ambiguous: "{args.module}" appears {num_matches} times'
                    f' in "{problem["text"]}"\'s {run_id_plus_one}-th run'
                )

kernel_records.sort(key=lambda rec: (rec.runtime_start, rec.capture_start))
kernels_per_problem: list[list[list[KernelTiming]]] = [
    [[] for _ in problem["run_starts"]] for problem in problem_set
]
for rec in kernel_records:
    kernels_per_problem[rec.problem_id][rec.run_id].append(
        KernelTiming(
            demangled_name=rec.demangled_name,
            kernel_start=rec.kernel_start,
            kernel_end=rec.kernel_end,
            range_names=rec.range_names,
        )
    )
for problem_id, runs in enumerate(kernels_per_problem):
    required_seq = [kernel.demangled_name for kernel in runs[0]]
    for run_id, run in enumerate(runs):
        seq = [kernel.demangled_name for kernel in run]
        assert seq == required_seq

converted_seqs: list[list[CategoryTime]] = []
warmup_times = run_args["warmup_times"] if args.warmup_times is None else args.warmup_times
for runs in kernels_per_problem:
    converted_seq: list[CategoryTime] = []
    # Kernel time
    for i, kernel in enumerate(runs[0]):
        name = kernel_short_name(string_ids[kernel.demangled_name])
        category = (*kernel.range_names, name)
        time_list = [run[i].kernel_end - run[i].kernel_start for run in runs]
        time_ns = np.mean(time_list[warmup_times:]).tolist()
        converted_seq.append(CategoryTime(category, time_ns))
    # Space and Overlap
    overlap_list = []
    space_list = []
    for run in runs:
        sorted_run = sorted(run, key=lambda k: k.kernel_start)
        last_end = sorted_run[0].kernel_start
        overlap_time = 0
        space_time = 0
        for kernel in sorted_run:
            if kernel.kernel_start > last_end:
                space_time += kernel.kernel_start - last_end
            else:
                overlap_time += min(last_end, kernel.kernel_end) - kernel.kernel_start
            last_end = max(last_end, kernel.kernel_end)
        overlap_list.append(-overlap_time)
        space_list.append(space_time)
    converted_seq.append(CategoryTime(("Overlap",), np.mean(overlap_list[warmup_times:]).tolist()))
    converted_seq.append(CategoryTime(("Space",), np.mean(space_list[warmup_times:]).tolist()))
    converted_seq.append(CategoryTime(("Total",), sum(ct.time_ns for ct in converted_seq)))
    converted_seqs.append(converted_seq)
if args.error_on_unknown_kernel and warned_names:
    raise ValueError("Unknown kernel names encountered")

merged_title: list[tuple[str, ...]] = []
for converted_seq in converted_seqs:
    title = [ct.category for ct in converted_seq]
    merged_title = shortest_common_supersequence(merged_title, title)

merged_data: list[list[float]] = [[0.0] * len(problem_set) for _ in merged_title]
for problem_id, converted_seq in enumerate(converted_seqs):
    cur = 0
    for ct in converted_seq:
        cur = merged_title.index(ct.category, cur)
        merged_data[cur][problem_id] = ct.time_ns
        cur += 1

print("Run args:")
print(run_args)

print("Problem set:")
for problem in problem_set:
    print(
        f'- "{problem["text"]}"    {len(problem["run_starts"])} runs'
        f"    Ranges: [{', '.join(r.text for r in problem['ranges'] if r.end <= problem['run_ends'][0])}]"
    )

stack: list[str] = []
csv_data: list[list[str]] = [["", *[problem["text"] for problem in problem_set]]]
js_data: list[dict] = []
js_stack: list[list[dict]] = [js_data]
max_title_len = max((len(title) - 1) * 3 + len(title[-1][:40]) for title in merged_title)
print("-" * (max_title_len + 1 + 6 * len(problem_set)))
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
js_header_config: list[dict] = []
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
template = jinja2.Environment(loader=loader).get_template("breakdown_template.html")
with html_file_path.open("w") as f:
    config_text = (
        "Run:\n"
        + json.dumps(run_args, indent=4)
        + "\n\nParse:\n"
        + json.dumps(args.__dict__, indent=4)
    )
    f.write(template.render(headerConfig=js_header_config, rawData=js_data, configText=config_text))

if args.query is not None:
    print("Query:")
    for query_str in args.query.split(","):
        query_str = query_str.strip()
        query_matched = [0.0] * len(problem_set)
        for title, time_data in zip(merged_title, merged_data):
            if query_str in ".".join(title):
                for i, x in enumerate(time_data):
                    query_matched[i] += x
        print(
            query_str + " " * (max_title_len - len(query_str)),
            *[f"{x / 1000:-6.1f}" for x in query_matched],
        )

correlation: list[dict] = []
for problem, runs in zip(problem_set, kernels_per_problem):
    timeline: list[dict] = []
    for i, kernel in enumerate(runs[0]):
        name = string_ids[kernel.demangled_name]
        duration_list = [run[i].kernel_end - run[i].kernel_start for run in runs]
        end_list = [run[i].kernel_end - run[0].kernel_start for run in runs]
        timeline.append(
            {
                "name": name,
                "duration": np.mean(duration_list[warmup_times:]).tolist(),
                "end": np.mean(end_list[warmup_times:]).tolist(),
            }
        )
    correlation.append(
        {
            "name": problem["text"],
            "timeline": timeline,
        }
    )
with json_file_path.open("w") as f:
    json.dump(correlation, f)
