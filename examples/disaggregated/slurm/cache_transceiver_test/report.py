# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Helper for the KV cache transceiver harness.

Three modes, all driven by `launch.slurm`:
  --emit-launch-vars CONFIG          print shell KEY=VAL used by launch.slurm
  --emit-ucx-env CONFIG --sweep K    print `export UCX_*=...` for sweep K
  --aggregate CONFIG --out FILE       parse per-rank CSVs + logs into results.json

RequestID encoding (shared with run_cache_transceiver_test.py) lets us attribute
every CSV row to a (combination, request_length) without a side map:

    rid = combination_idx * 1_000_000 + reqlen_idx * 10_000 + request_index
"""

import argparse
import csv
import datetime
import glob
import json
import math
import os
import re
import shlex
import statistics
import sys

import yaml

RID_COMBINATION_STRIDE = 1_000_000
RID_REQLEN_STRIDE = 10_000

# Transport tokens we surface from UCX_PROTO_INFO=y, ranked most→least
# interesting (network transports before intra-node fallbacks).
TRANSPORT_TOKENS = [
    "dc_mlx5",
    "rc_mlx5",
    "rc_verbs",
    "ud_mlx5",
    "ud_verbs",
    "srd",
    "gdr_copy",
    "cuda_ipc",
    "cuda_copy",
    "tcp",
    "self",
]


def load_config(path):
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def decode_rid(rid):
    """Return (case_idx, reqlen_idx, request_index) encoded in a request id."""
    case_idx = rid // RID_COMBINATION_STRIDE
    reqlen_idx = (rid // RID_REQLEN_STRIDE) % 100
    req_index = rid % RID_REQLEN_STRIDE
    return case_idx, reqlen_idx, req_index


def build_cases(cfg):
    """Flatten the test matrix into cases = combinations x cache-manager versions.

    Shared with the driver so case indices (encoded in RequestID) agree. Each
    case is one (backend, runtime, cache_manager) tuple tested at every request
    length.
    """
    combinations = cfg["test_matrix"].get("combinations") or cfg["test_matrix"]["combos"]
    versions = cfg["test_matrix"].get("cache_manager_versions", ["V1"])
    cases = []
    for combination in combinations:
        for ver in versions:
            # The V2 cache manager only supports the Python cache transceiver.
            if ver == "V2" and combination["runtime"] == "CPP":
                continue
            cases.append(
                {
                    "backend": combination["backend"],
                    "runtime": combination["runtime"],
                    "cache_manager": ver,
                    "label": f"{combination['backend']}/{combination['runtime']}/{ver}",
                }
            )
    return cases


# --------------------------------------------------------------------------- #
# Mode: emit-launch-vars
# --------------------------------------------------------------------------- #
def emit_launch_vars(cfg):
    env = cfg["environment"]
    n = int(cfg["hardware"]["gpus_per_node"])
    num_sweeps = len(cfg["ucx_env_sweep"])
    # Hard wall-clock cap for one full sweep. launch.slurm wraps each srun in
    # `timeout` with this value, so a hung sweep is killed and the loop advances
    # to the next UCX env set within at most this many seconds (default 300 =
    # 5 min). The per-cell watchdog (run_cache_transceiver_test.py) fires before
    # this to record TIMEOUT status; this is the final backstop.
    per_sweep_timeout = int(cfg["run"].get("max_sweep_s", 300))
    # UCX_PROTO_INFO value that launch.slurm exports (UCX then prints protocol-
    # selection info that report.py parses into selected_transport). Selected by
    # run.capture_proto_info:
    #   true  -> "used"  (default): compact, keeps the per-rank logs small.
    #            Requires UCX >= 1.21 to emit selected transport details.
    #   false -> "" (disabled, UCX_PROTO_INFO not exported).
    #   <str> -> that literal value, e.g. "y" to emit the full per-size tables
    #            (fills selected_transport, but is very verbose: ~50x log size).
    cap = cfg["run"].get("capture_proto_info", True)
    if cap is True:
        proto_info_value = "used"
    elif cap in (False, None, ""):
        proto_info_value = ""
    else:
        proto_info_value = str(cap)
    out = {
        "N": n,
        "IMAGE": env["container_image"],
        "MOUNTS": env.get("container_mount", ""),
        "WORK": env["work_dir"],
        "REPO": env.get("trtllm_repo", ""),
        "WHEEL_PATH": env.get("trtllm_wheel_path", ""),
        "BUILD_WHEEL": str(env.get("build_wheel", False)).lower(),
        "CUDA_ARCH": env.get("cuda_architectures", ""),
        "NUM_SWEEPS": num_sweeps,
        "PER_SWEEP_TIMEOUT": per_sweep_timeout,
        "PROTO_INFO_VALUE": proto_info_value,
    }
    for k, v in out.items():
        print(f"{k}={shlex.quote(str(v))}")


# --------------------------------------------------------------------------- #
# Mode: emit-ucx-env
# --------------------------------------------------------------------------- #
def emit_ucx_env(cfg, sweep_idx):
    sweeps = cfg["ucx_env_sweep"]
    sweep = sweeps[sweep_idx]
    # Union of every env var ANY sweep sets, in first-seen order. We `unset`
    # all of them before exporting this sweep's, so a variable set only by an
    # earlier sweep (e.g. UCX_RNDV_FRAG_MEM_TYPES from "mem_type_host") cannot
    # leak into a later sweep that does not set it. launch.slurm eval's this
    # block twice -- in the batch shell (which then feeds `srun --export=ALL`)
    # and inside the container prelude (after the image entrypoint/profile and
    # ALL-propagation have run) -- so both leak paths are cleared each sweep.
    # UCX_TLS is ALWAYS managed: a sweep that doesn't set it (including a sweep
    # with no env at all) must still start from a clean UCX_TLS, so a value from
    # an earlier sweep, the container profile, or the submitting shell can't leak
    # in. It is unset here and only re-exported below if this sweep sets it.
    managed = ["UCX_TLS"]
    seen = {"UCX_TLS"}
    for s in sweeps:
        for k in s.get("env") or {}:
            if k not in seen:
                seen.add(k)
                managed.append(k)
    print(f"unset {' '.join(managed)}")
    # Always echo the sweep name so logs are easy to correlate.
    print(f"export CTT_SWEEP_NAME={shlex.quote(sweep.get('name', str(sweep_idx)))}")
    for k, v in (sweep.get("env") or {}).items():
        print(f"export {k}={shlex.quote(str(v))}")


# --------------------------------------------------------------------------- #
# Mode: aggregate
# --------------------------------------------------------------------------- #
def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _parse_cpp_recv_csvs(csv_dir):
    """Return {rid: [per_rank_GBps, ...]} from C++ rank_*_recv.csv files.

    Each rank writes one row per request with a repeating Bandwidth(Gbps) column
    per transmission; we take the mean transmission bandwidth as that rank's
    per-GPU bandwidth. Gbps (bits) -> GB/s (bytes) by dividing by 8. We return
    the per-rank list (NOT a sum) so the aggregator can report per-GPU bandwidth
    (median across ranks) as the headline and derive a labeled aggregate; summing
    per-rank rates with unequal durations would overstate the real throughput.
    """
    per_rid = {}  # rid -> list[per-rank mean bw]
    # The driver renames each combination's rank_*_recv.csv to rank_*_recv__c<ci>l<li>.csv
    # (and the un-renamed name may exist for the last iteration). Match both.
    for path in glob.glob(os.path.join(csv_dir, "rank_*_recv*.csv")):
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                continue
            bw_cols = [i for i, name in enumerate(header) if name.strip() == "Bandwidth(Gbps)"]
            rid_col = header.index("RequestID") if "RequestID" in header else 0
            for row in reader:
                if not row or len(row) <= rid_col:
                    continue
                try:
                    rid = int(float(row[rid_col]))
                except ValueError:
                    continue
                bws = []
                for c in bw_cols:
                    if c < len(row):
                        try:
                            val = float(row[c])
                            if val > 0:
                                bws.append(val / 8.0)  # Gbps -> GB/s
                        except ValueError:
                            pass
                if bws:
                    per_rid.setdefault(rid, []).append(_mean(bws))
    return per_rid  # {rid: [per-rank GB/s]}


def _parse_python_csvs(csv_dir):
    """Return {rid: [per_rank_GBps, ...]} from perf_logger py_*_*.csv files.

    `throughput_mbs` is recorded for send tasks in MiB/s (perf_logger divides by
    1024*1024). We convert MiB/s -> GB/s (`* 1024^2 / 1e9`) so it matches the
    (converted) C++ bandwidth, and return the per-rank list (NOT a sum) -- see
    `_parse_cpp_recv_csvs`.
    """
    per_rid = {}  # rid -> list[per-rank throughput GB/s]
    for path in glob.glob(os.path.join(csv_dir, "py_*_*.csv")):
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "unique_rid" not in row or "throughput_mbs" not in row:
                    continue
                if "Send" not in row.get("task_type", ""):
                    continue  # throughput is meaningful only for send tasks
                try:
                    rid = int(float(row["unique_rid"]))
                    mbs = float(row["throughput_mbs"])
                except (ValueError, TypeError):
                    continue
                if mbs > 0:
                    gbytes = mbs * 1024.0 * 1024.0 / 1e9  # MiB/s -> GB/s
                    per_rid.setdefault(rid, []).append(gbytes)
    return per_rid  # {rid: [per-rank GB/s]}


# Protocol selection in the last column of a UCX_PROTO_INFO=y table row, e.g.
#   "... | rendezvous zero-copy read from remote | cuda_ipc/cuda |"
# Multi-lane configs may contain weighted entries such as
# `50% on rc_mlx5/mlx5_0:1 and 50% on rc_mlx5/mlx5_1:1`.
_PROTO_LAST_COL = re.compile(r"\|\s*([^|]+?)\s*\|\s*$")
_TRANSPORT_TOKEN = re.compile(
    rf"(?<![a-z0-9_])({'|'.join(map(re.escape, TRANSPORT_TOKENS))})(?![a-z0-9_])"
)
# A UCX_PROTO_INFO config header line (one per operation/peer/size-class).
_CONFIG_HEADER = re.compile(r"cfg#\d")
# Case boundary the driver prints at the start of each (sweep, combination) case.
_CASE_BEGIN = re.compile(r"\[CTT_CASE_BEGIN\]\s+ci=(\d+)")
# Data-path ops that actually move the (large) KV buffers, as opposed to
# control traffic (tagged/active messages, wireup).
_KV_DATA_OPS = ("ucp_put", "ucp_get", "remote memory", "rendezvous", "fetch")
_CONTROL_OPS = ("tagged message", "active message")


def _is_kv_data_header(line):
    """True if this config header describes a GPU (cuda) bulk-data transfer.

    The KV cache movement -- rather than control traffic. We attribute transport
    only from such configs so an incidental cuda_ipc control row never masks the
    real data path (e.g. a NIXL put that fell back to tcp 'software emulation').
    """
    if "cuda" not in line:
        return False
    if any(c in line for c in _CONTROL_OPS):
        return False
    return any(op in line for op in _KV_DATA_OPS)


def _proto_row_transports(line):
    """Return known transports from the final column of a protocol-table row."""
    match = _PROTO_LAST_COL.search(line)
    if not match:
        return []
    found = set(_TRANSPORT_TOKEN.findall(match.group(1)))
    return [tok for tok in TRANSPORT_TOKENS if tok in found]


class _TransportAcc:
    """Scans a UCX_PROTO_INFO=y table, tracking the current config header.

    Transport is taken from the GPU bulk-data (KV) configs. `kv` holds the
    data-path transports; `any` is a fallback for cases that never reached a
    data transfer (e.g. setup hang).
    """

    def __init__(self):
        self.kv = set()  # transports on GPU bulk-data config rows
        self.any = set()  # transport on any proto-table row (fallback)
        self.sw_emul = False  # a KV data row used 'software emulation' (slow)
        self._in_kv = False

    def feed(self, line):
        if _CONFIG_HEADER.search(line):
            self._in_kv = _is_kv_data_header(line)
            return
        transports = _proto_row_transports(line)
        if not transports:
            return
        self.any.update(transports)
        if self._in_kv:
            self.kv.update(transports)
            if "software emulation" in line:
                self.sw_emul = True

    def ranked(self, kv_only=False):
        chosen = self.kv if kv_only else self.kv or self.any
        out = [t for t in TRANSPORT_TOKENS if t in chosen]
        # Flag host-staged fallback so a tcp KV path isn't read as "fine".
        if self.sw_emul and "tcp" in out:
            out = [t if t != "tcp" else "tcp(sw-emul)" for t in out]
        return out


def _parse_proto_info(log_glob, kv_only=False):
    """Sweep-level transport(s) UCX selected for the KV transfer.

    UCX_PROTO_INFO=y prints a protocol-selection table; the transport that
    carries each (operation, size-range) is the LAST '|'-delimited column
    (e.g. `cuda_ipc/cuda`, `rc_mlx5/cuda`, `tcp/eth0`). We parse that column
    instead of grepping token substrings anywhere in the log -- the latter also
    matches the `UCX_TLS=...` echo line and small-message/control protocols.
    The KV blocks are large CUDA buffers, so CUDA-memory bulk rows win. Falls
    back to a substring scan only if no parseable table is present (older logs),
    unless `kv_only` requires evidence from a CUDA KV-data table.

    This is sweep-granular; prefer `_parse_proto_info_by_case` when the per-case
    CTT_CASE_BEGIN markers are present.
    """
    acc = _TransportAcc()
    for path in glob.glob(log_glob):
        try:
            with open(path, errors="ignore") as f:
                lines = f.readlines()
        except OSError:
            continue
        for line in lines:
            acc.feed(line)
    ranked = acc.ranked(kv_only=kv_only)
    if ranked:
        return ranked
    if kv_only:
        return []
    # Fallback: no parseable proto table (e.g. older logs) -- substring scan.
    chosen = set()
    for path in glob.glob(log_glob):
        try:
            with open(path, errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        for tok in TRANSPORT_TOKENS:
            if tok in text:
                chosen.add(tok)
    return [t for t in TRANSPORT_TOKENS if t in chosen]


def _parse_proto_info_by_case(log_glob, kv_only=False):
    """Return {case_idx: [transports]} attributed per (sweep, combination).

    Splits each rank log on the driver's `[CTT_CASE_BEGIN] ci=N` markers.
    Transport is constant across a case's request lengths (one transceiver per
    case), so this is the right granularity. Returns {} if no markers are
    present (older logs), so the caller can fall back to sweep-level
    `_parse_proto_info`. When `kv_only` is true, control-table fallbacks are
    excluded from each case.
    """
    accs = {}  # ci -> _TransportAcc
    saw_marker = False
    for path in glob.glob(log_glob):
        try:
            with open(path, errors="ignore") as f:
                lines = f.readlines()
        except OSError:
            continue
        cur = None
        for line in lines:
            mb = _CASE_BEGIN.search(line)
            if mb:
                cur = int(mb.group(1))
                saw_marker = True
                accs.setdefault(cur, _TransportAcc())
                continue
            if cur is not None:
                accs[cur].feed(line)
    if not saw_marker:
        return {}
    return {ci: acc.ranked(kv_only=kv_only) for ci, acc in accs.items()}


# Driver per-request log timestamp, e.g. "[06/03/2026-06:59:17] ... rid=12 ...
# transfer START (send)". UCX proto lines start with an epoch ts "[1780495192.0..]".
_DRIVER_TS = re.compile(r"\[(\d\d)/(\d\d)/(\d{4})-(\d\d):(\d\d):(\d\d)\]")
_TRANSFER_RID = re.compile(r"rid=(\d+).*transfer (START|DONE)")
_UCX_TS = re.compile(r"^\s*\[(\d{6,}\.\d+)\]")


def _case_start_times(log_glob):
    """Return {ci: earliest transfer-START epoch} from per-request logs.

    Used to attribute UCX proto configs to cases by time (robust to UCX's
    buffered / out-of-order log writes, which break line-position attribution).
    """
    starts = {}
    for path in glob.glob(log_glob):
        try:
            f = open(path, errors="ignore")
        except OSError:
            continue
        with f:
            for line in f:
                mt = _TRANSFER_RID.search(line)
                if not mt or mt.group(2) != "START":
                    continue
                md = _DRIVER_TS.search(line)
                if not md:
                    continue
                mo, dd, yy, hh, mm, ss = map(int, md.groups())
                try:
                    ts = datetime.datetime(yy, mo, dd, hh, mm, ss).timestamp()
                except ValueError:
                    continue
                ci = int(mt.group(1)) // RID_COMBINATION_STRIDE
                if ci not in starts or ts < starts[ci]:
                    starts[ci] = ts
    return starts


def _parse_proto_info_by_case_ts(log_glob, case_starts):
    """Return {ci: [transports]} by timestamp-correlating proto configs.

    Attributes each KV-data proto config to the case whose transfer window it
    falls in, matched by UCX timestamp vs per-case start times. A config with
    timestamp T belongs to the latest case that had started by T. Robust to UCX
    writing its (earlier-timestamped) proto table out of order relative to the
    driver's CTT_CASE_BEGIN markers. Returns {} if there are no driver start
    times (older logs without per-request logging).
    """
    if not case_starts:
        return {}
    ordered = sorted(case_starts.items(), key=lambda kv: kv[1])  # (ci, start)

    def _ci_for(ts):
        ci = None
        for c, st in ordered:
            if st <= ts:
                ci = c
            else:
                break
        return ci

    kv, sw = {}, {}
    for path in glob.glob(log_glob):
        try:
            f = open(path, errors="ignore")
        except OSError:
            continue
        with f:
            in_kv = False
            for line in f:
                if _CONFIG_HEADER.search(line):
                    in_kv = _is_kv_data_header(line)
                    continue
                if not in_kv:
                    continue
                mt = _UCX_TS.search(line)
                transports = _proto_row_transports(line)
                if not mt or not transports:
                    continue
                ci = _ci_for(float(mt.group(1)))
                if ci is None:
                    continue
                kv.setdefault(ci, set()).update(transports)
                if "software emulation" in line:
                    sw[ci] = True
    out = {}
    for ci, toks in kv.items():
        ranked = [t for t in TRANSPORT_TOKENS if t in toks]
        if sw.get(ci) and "tcp" in ranked:
            ranked = [t if t != "tcp" else "tcp(sw-emul)" for t in ranked]
        out[ci] = ranked
    return out


def _read_status(work_dir, sweep_idx):
    """Merge ctx/gen status jsonl into {(combination_idx, reqlen_idx): worst_record}."""
    # Distinct severities so the merge is deterministic regardless of which
    # role's file is read first (TIMEOUT and TRANSFER_ERROR must not tie).
    severity = {"PASS": 0, "TRANSFER_ERROR": 1, "TIMEOUT": 2, "MISMATCH": 3}
    merged = {}
    for role in ("ctx", "gen"):
        path = os.path.join(work_dir, "status", f"sweep{sweep_idx}_{role}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping malformed status line in {path}", file=sys.stderr)
                    continue
                key = (rec.get("combination_idx", rec.get("combo_idx")), rec["reqlen_idx"])
                cur = merged.get(key)
                if cur is None or severity.get(rec["status"], 1) > severity.get(cur["status"], 1):
                    merged[key] = rec
    return merged


def aggregate(cfg, out_path, require_kv_transport=False):
    work_dir = cfg["environment"]["work_dir"]
    cases = build_cases(cfg)
    req_lens = cfg["test_matrix"]["request_lengths"]
    sweeps = cfg["ucx_env_sweep"]

    # Results are organized PER COMBINATION (the tuning axis: for each combination, compare
    # the UCX sweeps). Per (sweep, combination) we keep only the LONGEST request
    # length's stats -- the peak/most representative bandwidth (small requests
    # are latency-bound). longest req_len index:
    li_rep = max(range(len(req_lens)), key=lambda i: req_lens[i])
    rep_req_len = req_lens[li_rep]
    combination_entries = {case["label"]: [] for case in cases}  # combination -> [sweep stats]
    for sweep_idx, sweep in enumerate(sweeps):
        sweep_name = sweep.get("name", str(sweep_idx))
        ucx_env = sweep.get("env") or {}
        gen_csv_dir = os.path.join(work_dir, "csv", str(sweep_idx), "gen")
        ctx_csv_dir = os.path.join(work_dir, "csv", str(sweep_idx), "ctx")
        log_glob = os.path.join(work_dir, "logs", f"sweep{sweep_idx}_*_rank*.log")
        # Transport is constant across a case's request lengths, so attribute it
        # per (sweep, combination). Primary: timestamp-correlate UCX proto configs with
        # the driver's per-request transfer windows (robust to UCX's buffered,
        # out-of-order log writes). Fall back to CTT_CASE_BEGIN line position,
        # then sweep-level, for logs lacking per-request timestamps.
        per_case_transport = _parse_proto_info_by_case_ts(log_glob, _case_start_times(log_glob))
        if not per_case_transport:
            per_case_transport = _parse_proto_info_by_case(log_glob, kv_only=require_kv_transport)
        sweep_transport = _parse_proto_info(log_glob, kv_only=require_kv_transport)
        status_map = _read_status(work_dir, sweep_idx)

        cpp_bw = _parse_cpp_recv_csvs(gen_csv_dir)
        # Python throughput is logged on the send (ctx) side.
        py_bw = _parse_python_csvs(ctx_csv_dir)

        warmup = cfg["test_matrix"]["warmup_requests"]
        # Bucket PER-RANK bandwidths by (combination_idx, reqlen_idx), skipping warmup.
        # Each rid maps to a list of per-rank (per-GPU) GB/s; we collect them all
        # so the headline can be per-GPU median (not a sum across ranks, which
        # would overstate throughput when ranks finish at different times).
        buckets = {}  # (ci, li) -> list[per-rank GB/s across all timed requests]
        nranks = {}  # (ci, li) -> number of ranks (GPUs) observed
        for rid, vals in list(cpp_bw.items()) + list(py_bw.items()):
            ci, li, r = decode_rid(rid)
            if r < warmup:
                continue
            buckets.setdefault((ci, li), []).extend(vals)
            nranks[(ci, li)] = max(nranks.get((ci, li), 0), len(vals))

        for ci, case in enumerate(cases):
            # Per (sweep, combination) transport (constant across this case's req_lens).
            transport = per_case_transport.get(ci, sweep_transport)
            key = (ci, li_rep)  # longest request length only
            bws = sorted(buckets.get(key, []))  # per-rank (per-GPU) samples
            ng = nranks.get(key, 0)
            rec = status_map.get(key)
            status = rec["status"] if rec else ("PASS" if bws else "NO_DATA")
            detail = rec.get("reason", "") if rec else ""
            # Headline = per-GPU bandwidth (median across ranks & requests).
            per_gpu = round(statistics.median(bws), 2) if bws else None
            combination_entries[case["label"]].append(
                {
                    "sweep": sweep_name,
                    # Full env set this sweep configured (every var, not just
                    # UCX_TLS/UCX_NET_DEVICES), so the report shows exactly what was
                    # tuned -- e.g. UCX_RNDV_FRAG_MEM_TYPES, UCX_RNDV_SCHEME.
                    "env": dict(ucx_env),
                    "selected_transport": ",".join(transport),
                    "status": status,
                    "per_gpu_BW_GBps": per_gpu,
                    # Nearest-rank p90 (>= median even for tiny sample counts).
                    "p90_BW_GBps": round(bws[max(0, math.ceil(0.9 * len(bws)) - 1)], 2)
                    if bws
                    else None,
                    # All GPUs combined, as per-GPU-median x #GPUs. Labeled so it is
                    # not mistaken for per-GPU; avoids summing unequal-duration rates.
                    "aggregate_BW_GBps": round(per_gpu * ng, 2) if per_gpu is not None else None,
                    "num_gpus": ng,
                    "num_samples": len(bws),
                    "error_detail": detail,
                }
            )

    by_combination = [
        {"combination": case["label"], "sweeps": combination_entries[case["label"]]}
        for case in cases
    ]
    best = _rank_best_per_combination(by_combination)
    results = {
        "req_len": rep_req_len,  # stats are for this (longest) length
        "by_combination": by_combination,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    # The best-UCX-per-combination summary (the deliverable) goes to its own file,
    # "<results>.best.json", next to the full results.
    base, ext = os.path.splitext(out_path)
    best_path = f"{base}.best{ext or '.json'}"
    with open(best_path, "w") as f:
        json.dump({"req_len": rep_req_len, "best_per_combination": best}, f, indent=2)
    _print_table(by_combination, rep_req_len, best)
    print(f"\nBest-per-combination summary: {best_path}", file=sys.stderr)
    return results


def _rank_best_per_combination(by_combination):
    """Pick the sweep with the best per-GPU bandwidth for each combination.

    Uses the representative/longest request length.
    """
    out = []
    for c in by_combination:
        best = None
        for s in c["sweeps"]:
            if s["status"] != "PASS" or s["per_gpu_BW_GBps"] is None:
                continue
            if best is None or s["per_gpu_BW_GBps"] > best["per_gpu_BW_GBps"]:
                best = s
        if best is not None:
            out.append(
                {
                    "combination": c["combination"],
                    "best_sweep": best["sweep"],
                    "per_gpu_BW_GBps": best["per_gpu_BW_GBps"],
                    "aggregate_BW_GBps": best["aggregate_BW_GBps"],
                    "env": best["env"],
                    "selected_transport": best["selected_transport"],
                }
            )
    return out


def _print_table(by_combination, rep_req_len, best):
    print(f"\n=== KV cache transceiver results (per combination, req_len={rep_req_len}) ===")
    print(
        "# Bandwidth is PER-GPU, GB/s (bytes, /1e9): median across ranks & "
        "requests at the longest request length. C++ Bandwidth(Gbps)/8; Python "
        "throughput_mbs (MiB/s) x 1024^2 / 1e9. aggr = perGPU x nGPU (all GPUs "
        "combined). transport = GPU<->GPU data path UCX selected."
    )
    hdr = [
        "combination",
        "sweep",
        "transport",
        "perGPU_GB/s",
        "p90_GB/s",
        "aggr_GB/s",
        "nGPU",
        "status",
    ]
    print("\t".join(hdr))
    for c in by_combination:
        for s in c["sweeps"]:
            print(
                "\t".join(
                    str(x)
                    for x in [
                        c["combination"],
                        s["sweep"],
                        s["selected_transport"],
                        s["per_gpu_BW_GBps"],
                        s["p90_BW_GBps"],
                        s["aggregate_BW_GBps"],
                        s["num_gpus"],
                        s["status"],
                    ]
                )
            )
    print(
        f"\n=== Best UCX env set per combination by per-GPU bandwidth (req_len={rep_req_len}) ==="
    )
    for b in best:
        env_str = " ".join(f"{k}={v}" for k, v in (b.get("env") or {}).items()) or "(none)"
        print(
            f"  {b['combination']:18s} -> {b['best_sweep']:18s} "
            f"{b['per_gpu_BW_GBps']:.2f} GB/s/GPU  "
            f"[{b['selected_transport']}]  "
            f"{env_str}"
        )
    if not best:
        print("  (no successful transfers)")


def main():
    ap = argparse.ArgumentParser(description="KV cache transceiver harness helper")
    ap.add_argument("config")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--emit-launch-vars", action="store_true")
    g.add_argument("--emit-ucx-env", action="store_true")
    g.add_argument("--aggregate", action="store_true")
    ap.add_argument("--sweep", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--require-kv-transport",
        action="store_true",
        help="accept transport only from CUDA KV-data protocol tables",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.emit_launch_vars:
        emit_launch_vars(cfg)
    elif args.emit_ucx_env:
        emit_ucx_env(cfg, args.sweep)
    elif args.aggregate:
        out = args.out or os.path.join(cfg["environment"]["work_dir"], "results.json")
        aggregate(cfg, out, require_kv_transport=args.require_kv_transport)
        print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
