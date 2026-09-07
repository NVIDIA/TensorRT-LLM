"""Shared prose blocks for the perf-analyze role prompts.

The exact ``trtllm-serve`` / ``benchmark_serving.py`` / ``nsys`` command
knowledge lives here (not in optional agent-toolkit skills) so the
workflow is self-contained. The benchmarker and analyzer prompts import
the blocks they need; the ``EXECUTION_SLURM_BOOTSTRAP`` block is appended
only when ``task.yaml`` carries a ``slurm-environment`` section, and the
``SOL_ANALYZER_CONTEXT`` / ``SOL_REPORTER_GUIDANCE`` blocks are
appended (to the analyzer / reporter) only when the projector stage is
enabled — the default, unless ``task.yaml`` sets ``sol.enabled: false``.

Three blocks point at agent-toolkit skills: ``CASEBOOK_CONSULTATION``
tells both serving roles to load ``perf-optimization-casebook`` as
read-only reference so their analysis is grounded in known TRT-LLM
performance precedents; ``PROFILING_RUNS_REFERENCE``'s Run C has the
analyzer load ``perf-nsight-compute-analysis`` as the methodology for
capturing and interpreting the ncu per-kernel deep dive; and the
projector's own prompt (in ``projector.py``) builds on
``internal-perf-sol-analysis`` as its projection methodology (the
analyzer loads the same skill for the measured↔SOL correlation when the
projector stage is enabled). All are written to degrade gracefully when
the skill is not installed, so they do not turn the workflow into a
hard dependency on the toolkit. ``internal-perf-sol-analysis`` carries
the ``internal-`` prefix, so open-source builds of the toolkit strip it
while keeping ``perf-analysis``; which of the two this session has is
resolved in Python before the stage runs (see
``sol_methodology.resolve_sol_methodology``), and only the fallback case
appends ``SOL_METHODOLOGY_FALLBACK`` to the projector's prompt.

The role-neutral blocks shared with perf-optimize live here rather than
inline in the role modules so both workflows compose the same
single-sourced text: the projector methodology
(``SOL_PROJECTOR_METHODOLOGY`` / ``SOL_PROJECTOR_INTERNAL_KNOWLEDGE``),
the analyzer's findings contract (``PROFILE_FINDINGS_CONTRACT``), and
the measured↔SOL correlation recipe (``SOL_CORRELATION_METHOD``) —
perf-optimize's analyzer is this workflow's analyzer plus the roadmap
machinery, and its prompts compose these same fragments.
"""

# --------------------------------------------------------------------------- #
# Server lifecycle (shared by benchmarker + analyzer)
# --------------------------------------------------------------------------- #

SERVER_LIFECYCLE = """\
## Running `trtllm-serve` (local default)

You drive the whole server lifecycle yourself with `Bash`. One server at
a time on the GPU(s) — the previous stage is *expected* to have torn its
server down before you start, but never assume it did: step 1 verifies
it, because an interrupted run leaves a detached server behind and the
port is fixed.

**You get a single turn — finish the work in it.** Launch, readiness
poll, benchmark/profile, teardown, and writing your output `.md` file
all happen in this one turn. Wait for slow steps (a checkpoint can take
many minutes to load) with **foreground** blocking shell loops; if one
poll window is not long enough, issue another blocking poll and stay in
your turn. **Do not end your turn to wait for a background poll to wake
you** — nothing re-invokes you, so the stage would advance with your
output file still empty and the whole run is wasted.

1. **Assert port 8000 is free — before launching anything.** The port is
   fixed, and a `trtllm-serve` from an earlier stage or an interrupted run
   is `setsid`-detached, so it *survives* a Ctrl-C and keeps answering on
   :8000. If you skip this check, your own server dies with "address
   already in use" while the **stale** one — serving an older config, or a
   different checkpoint entirely — answers every health poll, and the whole
   stage silently measures the wrong server:
   ```bash
   ss -ltn 'sport = :8000' | grep -q LISTEN && {
     echo "FATAL: port 8000 already in use — a stale server is still up"
     ss -ltnp 'sport = :8000'   # shows the owning pid/cmd
     exit 1
   }
   ```
   Do **not** work around a busy port by picking another one — the
   benchmark and profiling commands all target :8000. Reap the stale
   server by its PID with the teardown recipe in step 4 (identify it from
   the `ss -ltnp` output above), confirm the port is free, then launch.
   If `ss` is unavailable, `lsof -i :8000` or
   `curl -fsS http://127.0.0.1:8000/health` works the same way — a health
   response *before* you have launched anything is a stale server, not a
   ready one.

2. **Launch in the background, fully detached**, redirecting logs to a
   file in the workspace, and capture the PID. `setsid` + `< /dev/null`
   puts the server in its own session/process group so it survives across
   your separate `Bash` calls (a plain `&` job can be reaped when the
   launching shell exits):
   ```bash
   cd <trtllm_repo_path>
   setsid trtllm-serve <checkpoint_path> \\
       --backend pytorch \\
       --host 127.0.0.1 --port 8000 \\
       [--extra_llm_api_options <extra_llm_api_options>] \\
       > <workspace>/serve.log 2>&1 < /dev/null &
   echo $! > <workspace>/serve.pid
   ```
   Pass `--extra_llm_api_options <path>` **only when** `task.yaml` sets the
   top-level `extra_llm_api_options` key; omit the flag otherwise. The
   `--backend pytorch` and `--host 127.0.0.1 --port 8000` settings are
   fixed — all other server tuning lives inside the
   `extra_llm_api_options` YAML.

3. **Poll readiness in the foreground** before sending any load. Large
   checkpoints can take many minutes to load; poll on an interval with a
   generous timeout, bail out early if the process dies, and `tail`
   `serve.log` if it stalls. Check **liveness first and ownership before
   declaring READY** — a `/health` response only proves *some* server is on
   :8000, never that it is yours:
   ```bash
   PID=$(cat <workspace>/serve.pid)
   # setsid made the server a process-group leader, so its PGID == its PID
   # and every worker it forks inherits that PGID: the listener on :8000 is
   # ours only if its process group is $PID.
   owns_port() {
     local owner
     owner=$(ss -ltnp 'sport = :8000' 2>/dev/null \\
             | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
     [ -n "$owner" ] || return 2          # could not resolve — do NOT assume ours
     [ "$(ps -o pgid= -p "$owner" 2>/dev/null | tr -d ' ')" = "$PID" ]
   }
   for i in $(seq 1 360); do
     kill -0 "$PID" 2>/dev/null || { echo "server exited"; tail -40 <workspace>/serve.log; break; }
     if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
       owns_port \\
         && { echo READY; break; } \\
         || { echo "FATAL: :8000 answered but is not owned by PID $PID"; \\
              ss -ltnp 'sport = :8000'; break; }
     fi
     sleep 5
   done
   ```
   If the server never becomes ready, read `serve.log`, fix the cause
   (OOM → lower `kv_cache_config.free_gpu_memory_fraction` or
   `max_batch_size` in the `extra_llm_api_options` YAML; bad option →
   correct it), and retry. Do not proceed to benchmarking against a server
   that never reported ready.

   ⚠️ **A `FATAL: … not owned by PID` result is never something to work
   around.** It means a foreign server holds :8000, so every number you
   would produce belongs to a different config or checkpoint. Reap it
   (step 4), confirm :8000 is free, relaunch, and only then benchmark.
   The same applies when `owns_port` cannot resolve an owner at all: treat
   "unverified" as "not ours" and resolve it by hand (`ss -ltnp` /
   `lsof -i :8000` may need more privilege than the current shell has)
   rather than benchmarking on the assumption it is yours.

4. **Tear the server down — always**, even when a step failed, so the
   GPU is free for the next stage. Kill by the **recorded PID and its
   process group** (never by a name pattern), escalating to SIGKILL:
   ```bash
   PID=$(cat <workspace>/serve.pid)
   # Negative PID signals the whole process group (setsid made the server
   # a group leader), reaping its MPI workers / child procs with it.
   kill -TERM -"$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
   for i in $(seq 1 24); do            # up to ~120s for a graceful exit
     kill -0 "$PID" 2>/dev/null || break
     sleep 5
   done
   kill -0 "$PID" 2>/dev/null && { kill -KILL -"$PID" 2>/dev/null; kill -KILL "$PID" 2>/dev/null; } || true
   ```
   Then confirm the GPU memory actually freed with `nvidia-smi`, **and
   that :8000 is free again** (`ss -ltn 'sport = :8000'` prints no
   `LISTEN` row), before launching another server. If an MPI daemon
   (`orted`) or other straggler survives, find it with a read-only
   `ps`/`pgrep` and reap it by its **exact PID**.

   ⚠️ **Never `pkill -f 'trtllm-serve'` (or any bare `trtllm-serve`
   pattern).** The string `trtllm-serve` appears in *your own* agent
   process and in the very shell running the teardown, so a name-based
   kill can terminate this agent. Always target the recorded PID /
   process group; if you must pattern-match a straggler you cannot reach
   by PID, use a precise pattern that includes the checkpoint path and
   confirm it with read-only `pgrep` before killing.

`nvidia-smi` (read-only) is a useful sanity check for how many GPUs are
visible and whether memory is free between runs.
"""


SERVE_FLAGS_REFERENCE = """\
### Server configuration

`trtllm-serve` always runs with `--backend pytorch` on `127.0.0.1:8000`.
The model checkpoint to serve is the top-level `checkpoint_path` (the
positional `model` argument).

**All** other server tuning lives in one optional YAML, passed verbatim as
`--extra_llm_api_options <path>` when `task.yaml` sets the top-level
`extra_llm_api_options` key (omit the flag when it is absent). Those keys
map to LLM API fields, e.g.:

| `extra_llm_api_options` key                  | tunes                          |
| -------------------------------------------- | ------------------------------ |
| `tensor_parallel_size`                       | tensor parallelism (TP)        |
| `pipeline_parallel_size`                     | pipeline parallelism (PP)      |
| `moe_expert_parallel_size`                   | expert parallelism (EP)        |
| `max_batch_size`                             | max batch size                 |
| `max_num_tokens`                             | max num tokens                 |
| `kv_cache_config.free_gpu_memory_fraction`   | KV-cache memory fraction       |

Read the `extra_llm_api_options` YAML (when present) to learn the parallel
sizes and other knobs actually in effect. Check `trtllm-serve --help` and
the LLM API reference in `<trtllm_repo_path>` if unsure of a field name.
"""


BENCHMARK_FLAGS_REFERENCE = """\
## Running the benchmark — `benchmark_serving.py`

The load generator lives in the TensorRT-LLM checkout at
`tensorrt_llm/serve/scripts/benchmark_serving.py`. Run it as a module
(deps are importable wherever `trtllm-serve` works).

**Start from this canonical command — do not improvise the flags.** Fill
the `<...>` placeholders from `task.yaml` (`checkpoint_path` and the
`benchmark` block) and the workspace path; keep every other flag exactly
as shown:

```bash
cd <trtllm_repo_path>
python -m tensorrt_llm.serve.scripts.benchmark_serving \\
    --model <checkpoint_path> \\
    --tokenizer <checkpoint_path> \\
    --trust-remote-code \\
    --backend openai \\
    --host 127.0.0.1 --port 8000 \\
    --dataset-name <benchmark.dataset_name> \\
    --random-input-len <benchmark.random_input_len> \\
    --random-output-len <benchmark.random_output_len> \\
    --random-ids \\
    --tokenize-on-client \\
    --num-prompts <the point's num_prompts — see below> \\
    --max-concurrency <one value from benchmark.concurrency> \\
    [--request-rate <benchmark.request_rate>] \\
    [--dataset-path <benchmark.dataset_path>] \\
    --ignore-eos \\
    --no-test-input \\
    --percentile-metrics ttft,tpot,itl,e2el \\
    --metric-percentiles 90,99 \\
    --save-result --save-detailed --result-dir <workspace>
```

- **One run per concurrency point.** `benchmark.concurrency` is a single
  integer (one operating point) or a list of integers (Pareto-curve
  mode; the resolved spec is sorted ascending). With a single integer,
  run the command once with that value. With a list, launch the server
  **once**, run the command once per point **sequentially in ascending
  order** against the same server, and tear down after the last point —
  never relaunch the server between points, never run points in
  parallel, and never add, drop, or resize points beyond the configured
  list. ISL / OSL stay fixed across points.
- **Per-point `--num-prompts`:** `benchmark.num_prompts` is a single
  integer (use it at every point) **or a list paired index-by-index with
  the `benchmark.concurrency` list** (the resolved spec keeps both
  sorted by concurrency, so low-concurrency points can run far fewer
  prompts). For the run at point `concurrency[i]`, pass
  `--num-prompts <num_prompts[i]>`. Never swap in any other value — the
  per-point pairing is part of the operating point, and a measurement
  with a different `num_prompts` at the same concurrency is not
  comparable.
- **Curve-mode result dirs:** with a concurrency list, pass
  `--result-dir <base result dir>/concurrency_<c>` for the run at point
  `<c>` (benchmark_serving.py names result files by timestamp; the
  per-point subdirectory makes each point's JSON unambiguous for later
  stages). With a single integer, keep `--result-dir <base result dir>`
  as shown.
- **Create every result dir before the run:** `benchmark_serving.py`
  does **not** create `--result-dir` — with a missing directory the run
  completes normally but the result JSON is silently never written, and
  the whole measurement has to be redone. `mkdir -p` each result
  directory (curve mode: every per-point `concurrency_<c>` directory)
  before starting the sweep, and treat a finished run whose result JSON
  is absent as a failed run, never as skippable.
- `--model` and `--tokenizer` both point at `checkpoint_path`, and
  `--trust-remote-code` lets the tokenizer load model-specific code — keep
  all three so client tokenization matches the served model.
- `--ignore-eos` forces every request to emit exactly
  `--random-output-len` tokens instead of stopping early on an EOS, so the
  decode length (and thus OSL / TPOT / throughput) is deterministic and
  comparable across runs. Keep it on.
- `--random-input-len` / `--random-output-len` apply to the default
  `random` dataset. **By default `random` samples real text from
  ShareGPT** (it needs a local download / `--dataset-path`); the
  **`--random-ids`** flag above synthesizes random token IDs instead —
  self-contained, no download, and deterministic (fixed seed) so the
  Analyzer can replay the identical load. Drop `--random-ids` only for a
  non-`random` dataset.
- Keep `--random-ids` paired with **`--tokenize-on-client`** so prompts
  are sent as raw token-ID lists. This pins the effective ISL to
  `--random-input-len` **exactly**: without it the client decodes the IDs
  to text, the server re-tokenizes it, and the ISL can drift **above**
  the server's `max_input_len`, getting requests rejected — a real risk
  when the operating point sets ISL equal to `max_input_len`. Verify
  afterward that the result JSON's `total_input_tokens` equals the
  point's `num_prompts × ISL`.
- `--no-test-input` skips benchmark_serving.py's warmup/test prompt. Keep
  it: it makes the Benchmarker and Analyzer drive the **same** load, and
  it is mandatory for the Analyzer's profiling replays (the test prompt
  would advance the server's iteration counter and shift the profiling
  window — see the profiling run steps).
- For `sharegpt`/`hf` datasets pass `--dataset-path` instead of
  `--random-ids` and follow `benchmark_serving.py --help` for the
  dataset's required flags. Keep **`--tokenize-on-client`** for the same
  ISL-pinning reason as above — it matters most when the operating point
  sets ISL equal to the server's `max_input_len`.
- `--save-result` writes a JSON result into `<workspace>`; `--save-detailed`
  adds per-request rows. Note the exact JSON filename it prints.
- Key reported metrics: request / output / total-token throughput;
  **TTFT** (time to first token), **TPOT** (time per output token),
  **ITL** (inter-token latency), **E2EL** (end-to-end latency) — each
  with mean / median / p90 / p99.
"""


DERIVED_METRICS_REFERENCE = """\
## Derived per-user / per-GPU metrics (the Pareto axes)

Two derived metrics accompany every benchmark measurement, computed from
the result JSON plus the serving world size:

- `tok/s/user = 1000 / mean_tpot_ms` — per-user decode speed
  (interactivity), from that run's `mean_tpot_ms`.
- `tok/s/gpu = output_throughput / num_gpus` — per-GPU output
  throughput.
- `num_gpus` is the serving world size: the product of the parallel
  sizes actually in effect (`tensor_parallel_size` ×
  `pipeline_parallel_size` × `moe_expert_parallel_size` where they
  multiply the GPU count, each defaulting to 1) read from the
  `extra_llm_api_options` YAML in effect, cross-checked against
  `nvidia-smi` (the GPUs actually holding server memory). Record the
  value **and how you determined it** next to the metrics.

In Pareto-curve mode (`benchmark.concurrency` is a list) every measuring
report must include this **curve summary table**, one row per
concurrency point in ascending order (add the workflow's target metric
as an extra column when your instructions name one other than
`output_throughput`):

| concurrency | output_throughput (tok/s) | tok/s/user | tok/s/gpu | mean TPOT (ms) | mean TTFT (ms) |
|---|---|---|---|---|---|

The Pareto curve plots **x = tok/s/user, y = tok/s/gpu**, one point per
concurrency. `tok/s/user` and `tok/s/gpu` are **reporting / Pareto
axes** — any acceptance gate or target-metric comparison runs on the
result-JSON metric named in your instructions (e.g.
`output_throughput`), never on these derived values.
"""


# --------------------------------------------------------------------------- #
# Profiling references (shared by both workflows' analyzers)
# --------------------------------------------------------------------------- #

PROFILING_KNOB_VERIFICATION = """\
## Verify the profiling knobs first (verify before asserting)

Profiling env-var names differ across TensorRT-LLM versions. **Before
relying on them**, confirm what *this* checkout in `trtllm_repo_path`
actually supports:

- With `grep -rn`/`rg` via `Bash`, search
  `tensorrt_llm/_torch/pyexecutor/py_executor.py` for **both**
  server-side profiler gates: `TLLM_PROFILE_START_STOP` (the iteration
  window, and its exact format, e.g. `"100-150"`) and the **torch-trace
  env var** (e.g. `TLLM_TORCH_PROFILE_TRACE`, or a `*_TORCH_PROFILE_*` /
  `*_TORCH_PROFILER_DIR` name). Use the name you find, not a guessed one,
  and note whether its value is a file path or a directory.
- With `grep -rn`/`rg` via `Bash`, search
  `tensorrt_llm/serve/openai_server.py` for a `/start_profile`
  endpoint. Many checkouts **do not have one** — both profilers are then
  driven entirely server-side by the env vars above, and the benchmark
  client's `--profile` flag is a no-op. Do not assume the endpoint exists.
- If a knob you need does not exist in this checkout, note that in
  `profile_findings.md` and skip that profiler gracefully rather than
  fabricating a trace.
"""


PROFILING_RUNS_REFERENCE = """\
## Run A — Nsight Systems (GPU timeline)

1. Relaunch `trtllm-serve` (same flags as the Benchmarker) wrapped in
   nsys, gating capture to a steady-state iteration window. **Start from
   this canonical `nsys profile` invocation — do not improvise the nsys
   flags** (fill the `<...>` placeholders; keep the rest as shown):
   ```bash
   cd <trtllm_repo_path>
   setsid env TLLM_PROFILE_START_STOP="<profile.nsys_iter_range>" \\
   nsys profile \\
       -o <workspace>/server_nsys -f true \\
       -t 'cuda,nvtx,python-gil' \\
       -c cudaProfilerApi --capture-range-end=stop \\
       --cuda-graph-trace node \\
       -e TLLM_NVTX_DEBUG=1 \\
       --trace-fork-before-exec=true \\
       trtllm-serve <checkpoint_path> ...same trtllm-serve flags... \\
       > <workspace>/serve.log 2>&1 < /dev/null &
   echo $! > <workspace>/serve.pid
   ```
   - `TLLM_PROFILE_START_STOP` is an **iteration** window (e.g. `100-150`)
     counted in server forward steps: at `<start>` the server calls
     `cudaProfilerStart()` and at `<stop>` `cudaProfilerStop()`, which is
     the range `-c cudaProfilerApi` arms nsys to capture.
   - **`--capture-range-end=stop`** is required. The template omits it, but
     this automated run must add it: it makes nsys stop *collecting* at the
     window's end while leaving the server running. The nsys default for
     `-c` is `stop-shutdown`, which SIGTERMs the engine mid-load, crashes
     it (`RuntimeError: threads can only be started once`), and produces no
     report.
   - `-e TLLM_NVTX_DEBUG=1` turns on TensorRT-LLM's extra NVTX ranges so
     the `nvtx` trace is richer; `-t 'cuda,nvtx,python-gil'` adds
     Python-GIL tracing and `--cuda-graph-trace node` expands CUDA-graph
     nodes — together they attribute host vs GPU time. `-f true`
     force-overwrites a prior report.
   - **`--cuda-graph-trace node` can hang a graph-heavy server, and when it
     does you get NO report at all.** Observed on GB300 with this model: the
     profiled server never finished, the job hit its wall clock, and the
     `.nsys-rep` was never written — the run then spent five more attempts
     and over an hour of 4-GPU time trying variations, because a missing
     report looks identical to "profiling did not run yet".

     Node granularity expands every node of every replayed CUDA graph, and
     this configuration replays a lot of them (`cuda_graph_config` with
     `max_batch_size` in the hundreds). If the first profiling attempt does
     not produce a `.nsys-rep` within its window, DROP TO
     `--cuda-graph-trace graph` — coarser, still attributes graph time as a
     whole, and completes. Do not simply retry the same flags: the second
     identical attempt costs another allocation and fails the same way.

     Prefer a report at graph granularity over no report at all. An analysis
     with no profile is an opinion, and the rounds that follow spend GPU
     hours on it.
   - `--trace-fork-before-exec=true` follows child processes forked before
     `exec` (needed to trace trtllm-serve's workers).
   - If your installed `nsys` is old enough to reject one of these flags,
     drop **only** that flag, note it in `profile_findings.md`, and keep
     going — do not fabricate a trace.
2. Poll readiness, then replay the **same** `benchmark_serving.py` load.
   The canonical benchmark command already carries **`--no-test-input`**,
   which matters doubly here: the default warmup/test prompt runs through
   the server first and advances the iteration counter, and without
   `--no-test-input` it can push the counter into your window before the
   real concurrency load arrives, so the capture lands on a single-request
   decode instead of steady state. Make sure the load runs for more than
   `<stop>` server iterations to reach the window (raise `num_prompts` or
   lower the window if not), and confirm `serve.log` logs `Profiling
   started at iteration <start>` then `... stopped at iteration <stop>`.
3. Tear the server down (the PID / process-group teardown below) so nsys
   flushes `server_nsys.nsys-rep`. If the report did not finalize, send
   `SIGINT` to the recorded PID and wait for nsys to write the report
   before killing harder.
4. Summarize the trace without opening the GUI:
   ```bash
   nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_trace \\
       <workspace>/server_nsys.nsys-rep > <workspace>/nsys_stats.txt
   ```
   Extract: top CUDA kernels by total time and their share, the GEMM /
   attention / tensor-core vs memory/elementwise kernel mix, GPU busy vs
   idle, inter-kernel gaps (host/launch exposure), and any NCCL/collective
   time for multi-GPU runs.

For multi-GPU runs (tensor/pipeline/expert parallelism set in
`extra_llm_api_options`), nsys wrapping only traces the launcher rank;
consult the repo's profiling guidance (e.g. `TLLM_PROFILE_LOG_RANKS`,
`examples/.../nsys` patterns) and at minimum profile rank 0, noting the
limitation.

## Run B — PyTorch profiler (op-level)

In current checkouts the torch profiler is **server-side and
iteration-gated** by the same `TLLM_PROFILE_START_STOP` window — you turn
it on by pointing the torch-trace env var you confirmed above (e.g.
`TLLM_TORCH_PROFILE_TRACE`) at a path under `<workspace>/torch_trace/`,
**not** with the benchmark client's `--profile` flag (which POSTs to a
`/start_profile` endpoint that may not exist in this build — see *Verify
the profiling knobs first*).

1. Relaunch `trtllm-serve` (same flags) with **both** env vars set, so the
   server writes a torch trace over the same window:
   ```bash
   cd <trtllm_repo_path>
   mkdir -p <workspace>/torch_trace
   setsid env TLLM_PROFILE_START_STOP="<profile.nsys_iter_range>" \\
       <TORCH_TRACE_ENV_VAR>=<workspace>/torch_trace/trace.json \\
       trtllm-serve <checkpoint_path> ...same trtllm-serve flags... \\
       > <workspace>/serve.log 2>&1 < /dev/null &
   echo $! > <workspace>/serve.pid
   ```
   Use the exact env-var name and value shape (a **file path**, not a
   directory) you confirmed from `py_executor.py`; some versions append
   `-rank-<N>` to the base name.
2. Poll readiness, then replay the **same** load (the canonical benchmark
   command already includes `--no-test-input`, as in Run A). Add the
   client-side `--profile` flag **only if** you
   confirmed `/start_profile` exists in this checkout; otherwise omit it —
   it is a no-op here and the server-side env vars do the work.
3. Also pass **`--save-request-time-breakdown`**, which fetches
   `/perf_metrics` (a per-request prefill-vs-decode breakdown) — keep the
   resulting `*perf_metrics*.json`. That endpoint only returns data when
   the server was launched with `return_perf_metrics` enabled (it defaults
   off); set `return_perf_metrics: true` in an `extra_llm_api_options` YAML
   for this run if you need the breakdown, else drop the flag and note it.
4. Tear the server down. Inspect the torch trace(s) under `torch_trace/`
   for the top operators by self/CUDA time and any host-side stalls.

## Run C — Nsight Compute (ncu, per-kernel deep dive)

nsys tells you **where** GPU time goes; ncu tells you **why** those
kernels are slow (per-kernel SOL%, occupancy, warp stalls). Run C
therefore runs **last** and targets the top kernels Run A surfaced —
never profile every kernel blindly.

1. **Verify the tool and load the methodology first.** Check
   `ncu --version` via `Bash`; if `ncu` is absent or lacks profiling
   permission (`ERR_NVGPUCTRPERM` in its output), skip this run, note it
   under *Caveats*, and keep the findings' `ncu kernel analysis` section
   with a one-line `ncu unavailable: <reason>` — never fabricate
   metrics. Then **load the `perf-nsight-compute-analysis` skill** (via
   the `Skill` tool; fully-qualified
   `trtllm-agent-toolkit:perf-nsight-compute-analysis` if the bare name
   is not found) — it is the methodology for everything ncu: the
   SOL%-based bottleneck classification thresholds, the section
   escalation table, and the interpretation vocabulary. If the skill is
   unavailable, note that in one line and fall back to the command below
   plus the classification table you know from it.
2. **Pick the targets from Run A.** From `nsys_stats.txt`'s
   `cuda_gpu_kern_sum`, take the top kernels by total GPU time (typically
   3–6 distinctive name stems covering the majority share) and build one
   `--kernel-name "regex:<stem1|stem2|...>"` filter. Record the mapping
   stem → full kernel name in your findings. (If nsys was not run —
   not in `profile.methods` or its knob was missing — take the stems
   from the torch trace's top CUDA kernels instead, or as a last resort
   drop the `--kernel-name` filter and let `--launch-count` alone bound
   the capture, noting the untargeted sample in *Caveats*.)
3. Relaunch `trtllm-serve` (same flags) wrapped in ncu, gated to the
   same steady-state window. **Start from this canonical invocation —
   do not improvise the ncu flags** (fill the `<...>` placeholders;
   keep the rest as shown):
   ```bash
   cd <trtllm_repo_path>
   setsid env TLLM_PROFILE_START_STOP="<profile.nsys_iter_range>" \\
   ncu --target-processes all \\
       --profile-from-start off \\
       -o <workspace>/server_ncu -f \\
       --section SpeedOfLight --section LaunchStats --section Occupancy \\
       --section WarpStateStats --section MemoryWorkloadAnalysis \\
       --section ComputeWorkloadAnalysis \\
       --kernel-name "regex:<top-kernel stems from Run A>" \\
       --launch-count 40 \\
       trtllm-serve <checkpoint_path> ...same trtllm-serve flags... \\
       > <workspace>/serve.log 2>&1 < /dev/null &
   echo $! > <workspace>/serve.pid
   ```
   - `--profile-from-start off` arms ncu on the `cudaProfilerStart()` /
     `cudaProfilerStop()` calls that `TLLM_PROFILE_START_STOP` already
     drives, so the capture lands in the same iteration window as Runs
     A and B.
   - `--launch-count 40` caps the total profiled launches — ncu replays
     each profiled kernel, running it 10–100× slower than native, so an
     uncapped capture would hang the load. Once the cap is hit the rest
     of the run proceeds at near-native speed.
   - **Multi-rank serves: expect the hang watchdog to end the capture
     early — that is a partial capture, not a failed run.** Replaying a
     CUDA-graph kernel under MPI costs on the order of a minute per
     launch (graph-node replay serialized + cross-rank sync), and
     TensorRT-LLM's executor hang watchdog
     (`tensorrt_llm/_torch/pyexecutor/hang_detector.py`, fixed 300 s —
     `hang_detection_timeout` has no config or env knob) kills the
     server after only a handful of profiled launches. ncu writes the
     report incrementally, so the launches already captured survive:
     import what is there, and only if dominant stems are still missing
     run another *small targeted pass* (one kernel family per pass) as
     walltime allows. State the achieved share of GPU time your
     captures cover under *Caveats* — never burn server relaunches
     chasing full coverage. Keep cross-rank collective stems (allreduce
     / allgather) **out** of the `--kernel-name` filter: replaying a
     collective under ncu deadlocks the ranks; take their share from
     nsys instead.
   - The `--section` list is the one-shot adaptation of the skill's
     escalation ladder (SpeedOfLight to classify; the rest are the
     sections its table escalates to per class). One combined capture
     is deliberate: each server relaunch costs a full checkpoint load,
     so the escalation sections are collected up front instead of over
     multiple runs.
   - `--target-processes all` follows trtllm-serve's forked workers.
     If your installed `ncu` rejects a flag, drop **only** that flag,
     note it, and keep going.
4. Poll readiness, then replay the **same** benchmark load (canonical
   command, `--no-test-input`). Expect the profiled window to take much
   longer wall-clock than Runs A/B — poll patiently rather than
   declaring a hang; the client-side numbers from this replay are
   **not measurements** (kernel replay serializes the GPU) and must
   never be reported as performance results.
5. Tear the server down so the report finalizes at
   `<workspace>/server_ncu.ncu-rep`, then summarize without the GUI:
   ```bash
   ncu --import <workspace>/server_ncu.ncu-rep --page details \\
       > <workspace>/ncu_details.txt
   ncu --import <workspace>/server_ncu.ncu-rep --page raw --csv \\
       > <workspace>/ncu_raw.csv
   ```
6. **Analyze per the skill** you loaded in step 1: for every profiled
   kernel extract duration, `Compute (SM) Throughput`, `Memory
   Throughput`, and achieved occupancy; classify each with the skill's
   thresholds (compute-bound / memory-bound / latency-bound /
   balanced); and for the dominant kernels pull the escalation evidence
   (dominant warp-stall reason, launch/occupancy limiters). This
   per-kernel *why* is the `ncu kernel analysis` section of your
   findings and a required pillar of your ranked hypotheses.

Tear every server down when done (see *Running `trtllm-serve`* below).
"""


# --------------------------------------------------------------------------- #
# The findings contract (both workflows' analyzers): the required structure
# of profile_findings.md. Path-neutral — perf-analyze writes it at the
# workspace root, perf-optimize in the round's analysis/ directory; each
# workflow's instructions name the exact path.
# --------------------------------------------------------------------------- #

PROFILE_FINDINGS_CONTRACT = """\
## Required findings structure (`profile_findings.md`)

`Write` your findings to the `profile_findings.md` path named in your
instructions, using this structure. Section headers must match.

```
# Profiling Findings: <model name>

## Profiling setup
- nsys: command + iteration window (TLLM_PROFILE_START_STOP) + trace file
- torch profiler: env var used + trace dir
- ncu: command + kernels targeted (stem → full name) + launch count +
  report file
- Operating point replayed: <ISL/OSL/concurrency> (matches the
  benchmark report)
- Profiled concurrency point: <c> (curve mode: the largest of
  `benchmark.concurrency`; scalar mode: the configured value)

## nsys timeline
- Top kernels (name, % of GPU time) — table
- GPU busy vs idle, inter-kernel gaps
- Kernel mix (compute/tensor-core vs memory/elementwise; NCCL if multi-GPU)

## Torch profiler
- Top operators (name, self/CUDA time) — table
- Host-side stalls / sync points, if any
- Per-request prefill vs decode split (from perf_metrics, if available)

## ncu kernel analysis
- Per-kernel table: kernel, duration, Compute (SM) SOL%, Memory SOL%,
  achieved occupancy, bound class (per the perf-nsight-compute-analysis
  skill's thresholds), dominant warp-stall reason
- Which top-timeline kernels (from the nsys table) were profiled, and
  the share of GPU time they cover
- Interpretation: why each dominant kernel is slow — the bound class
  plus the escalation evidence behind it (occupancy limiter, stall
  breakdown, launch stats)
- When ncu did not run (not in `profile.methods`, tool missing, no
  permission): keep the section with one line — `ncu unavailable:
  <reason>` — and record it in *Caveats*.

## SOL correlation (measured vs ceiling)
<Only when the task enables the SOL projector stage — your SOL-gated
instructions then define the section's content. Omit the section
entirely otherwise.>

## Ranked bottleneck hypotheses
1. <hypothesis> — its bottleneck-taxonomy category; supporting evidence
   (cite the trace file + numbers); the casebook *bottleneck signal →
   candidate pattern* row it matches, if any
2. ...

## Caveats
<Any profiler that failed or was unavailable, multi-GPU tracing limits,
windows that didn't capture, etc.>
```

Every signal must cite the trace file and the numbers it came from. For
each ranked hypothesis, name the taxonomy category it belongs to and
match the trace signal against the **`perf-optimization-casebook`**
index you loaded (e.g. "many small dependent kernels, SM≥90 → launch",
"AllReduce a large share at TP/EP>1 → communication") so the downstream
stages inherit a known precedent. Naming the precedent is not the same
as applying it — no fix is applied at this stage.

**Synthesize the analyses — one pillar is not enough.** The ranked
hypotheses (and every optimization suggestion built from them
downstream) rest on three evidence pillars: the **nsys timeline** (where
GPU time goes, what the host exposes), the **ncu kernel analysis** (why
the hot kernels are slow — per-kernel SOL% and bound class), and the
**SOL correlation** (how far each region sits from its analytical
ceiling — when the task enables it). Each ranked hypothesis must say
which pillar(s) support it and whether the others corroborate,
contradict, or are silent on it; a pillar that did not run is named as
missing, never silently skipped. A hypothesis all available pillars
agree on outranks one resting on a single pillar at comparable impact.
"""


# --------------------------------------------------------------------------- #
# Optimization casebook consultation (shared by benchmarker + analyzer)
# --------------------------------------------------------------------------- #

CASEBOOK_CONSULTATION = """\
## Ground your analysis in the optimization casebook (load it early)

Before you reason about this TensorRT-LLM run's performance, **load the
`perf-optimization-casebook` skill** with the `Skill` tool and keep it open
as reference. It ships with the `trtllm-agent-toolkit` plugin, so invoke it
as `perf-optimization-casebook` — or the fully-qualified
`trtllm-agent-toolkit:perf-optimization-casebook` if the bare name is not
found. Do this **early in your turn, right after you read `task.yaml`**, so
every judgement you record afterward is informed by it.

Its **bottleneck signal → candidate pattern** index is what anchors your
findings to known TRT-LLM precedents instead of reasoning from scratch;
the skill itself describes what it holds and how to search it.

Where this stage departs from the skill's own loop: treat it as
**read-only reference material only.** Loading it costs one
`Skill` call plus reading a family index / case file. Do **not** apply
optimizations, edit configs, or run extra experiments from it — building
the evidence is this stage's only job; acting on a precedent is out of
scope. If the skill is not available in this environment, note that in one
line and proceed — never block the run on it.
"""


# --------------------------------------------------------------------------- #
# Slurm execution (appended only when task.yaml has a slurm-environment block)
# --------------------------------------------------------------------------- #

EXECUTION_SLURM_BOOTSTRAP = """\
## Slurm execution (this task has a `slurm-environment` block)

`task.yaml` contains a `slurm-environment` section, so the server and the
benchmark must run **inside a Slurm-launched container**, not on the
login node. Read these fields and use them verbatim:

1. `slurm_partition` — the partition to submit GPU jobs to.
2. `docker_image` — the enroot/pyxis container image (typically a `.sqsh`).
3. Top-level `trtllm_repo_path` and `checkpoint_path` — bind-mount both
   into the container at the **same absolute path** they have on the host.

Do not invent a different partition, image, or path, and do not silently
fall back to a local non-Slurm run when `slurm-environment` is present. If
a value is unusable, stop and report it as a blocker.

Run the server, the readiness poll, the benchmark, and the profilers
**within one allocation** (so the load generator can reach the server),
e.g. an interactive `salloc`/`srun` shell or a single `sbatch` script:

```bash
# bind trtllm_repo_path, checkpoint_path, and workspace at identical
# host:container paths (comma-separated in --container-mounts).
srun --partition=<slurm_partition> \\
     --container-image=<docker_image> \\
     --container-mounts=<repo>:<repo>,<ckpt>:<ckpt>,<workspace>:<workspace> \\
     --gres=gpu:<num_gpus> --pty bash
# then, inside the container, follow the local launch / benchmark / nsys
# / torch-profiler steps exactly as described, writing all artifacts to
# <workspace>.
```

The local launch, readiness-poll, teardown, and profiling steps are
otherwise identical — they just run inside the container. All artifacts
(`serve.log`, result JSON, `*.nsys-rep`, `torch_trace/`, the `.md`
outputs) must land in `<workspace>` so later stages and the user can read
them.

### Under Slurm, `trtllm-serve` needs `trtllm-llmapi-launch`

A bare `trtllm-serve` works from a login shell and **fails inside a Slurm
step**, so this is a rule you cannot discover by testing the command
locally. The LLM API creates its workers with `MPI.COMM_SELF.Spawn`, and
dynamic MPI process spawning is not permitted inside an `srun` step. The
symptom is not a clear rejection — the server simply never becomes ready
while `serve.log` repeats:

```
mpi4py.MPI.Exception: MPI_ERR_SPAWN: could not spawn processes
```

and your readiness poll eventually times out, which reads as "the model is
slow to load" rather than "this can never work".

Three things are required together — one alone is not enough:

```bash
srun --partition=<slurm_partition> \\
     --mpi=pmix \\                      # 1. PMIx bootstrap for the ranks
     --ntasks-per-node=<world_size> \\   # 2. one task PER RANK, not 1
     --container-image=<docker_image> \\
     --container-mounts=<repo>:<repo>,<ckpt>:<ckpt>,<workspace>:<workspace> \\
     --container-workdir=<repo> \\
     --gres=gpu:<num_gpus> \\
  bash -c 'trtllm-llmapi-launch trtllm-serve <ckpt> ...'
#          ^ 3. adopts the ranks srun already created instead of spawning
```

`<world_size>` is the product of the parallel sizes in the tuning config
(`tensor_parallel_size` × `pipeline_parallel_size` × …; `moe_expert_parallel_size`
reuses the TP ranks and does **not** multiply it). It is a property of the
config, so recompute it whenever you change one of those values — leaving
`--ntasks-per-node=1` under a `tensor_parallel_size: 2` config is the
common version of this mistake.

Use `trtllm-llmapi-launch` even at world size 1: the restriction is on
spawning inside a step, not on the number of ranks.

`--gres=gpu:<num_gpus>` may exceed `<world_size>` when the cluster's QOS
enforces a floor (this one requires `--gres=gpu:4`). Allocating more GPUs
than ranks is allowed and simply leaves the extras idle; do NOT raise the
parallel sizes just to consume them, because that changes the configuration
under measurement.

### One STEP, not merely one allocation — `127.0.0.1` is per-step

The server and whatever talks to it (readiness poll, benchmark client) must
run in the **same `srun` step**. One allocation is not enough. Under pyxis
each `srun` gets its own container with its own network namespace, so
`127.0.0.1` inside the client's step is NOT the loopback the server bound —
the connection is refused even though both steps are on the same node and
the server is perfectly healthy.

This failure is a liar. The server log shows `Application startup complete`
and `200 OK` for its own polls, `/v1/models` lists the model, the GPUs show
memory in use — and the client still reports the server as unreachable, so
the natural conclusion is "the server did not come up" when it plainly did.

    sbatch script
      └── srun (ONE step)
            server &        # background, binds 127.0.0.1
            poll /health    # same namespace — this works
            benchmark       # same namespace — this works
            teardown

If you genuinely need a separate step, then the server must bind `0.0.0.0`
and the client must address it by node name (`$SLURMD_NODENAME`), and that
step needs `--overlap` to share the allocation. Prefer the single step: it
has fewer ways to go wrong and needs no extra flags.

### Use a batch script, not an interactive allocation

Prefer the single `sbatch` script form. Do **not** use `srun --pty bash`
or `salloc` and then issue further commands expecting the allocation to
still be there.

The reason is mechanical, not stylistic: each of your `Bash` calls is a
separate process. An interactive allocation belongs to the call that
created it and is gone when that call returns, so a server started in one
call is not running in the next, `serve.pid` names a process that no
longer exists, and the readiness poll on `127.0.0.1:8000` reaches a
different machine's loopback. The failure looks like a server that would
not start.

One script that runs the whole stage — start the server, poll it, run the
benchmark, capture the profiles, tear down — keeps all of that inside one
allocation on one node, which is what the "within one allocation" rule
above actually requires.

Two consequences worth stating:

- **Every path in the script is a CLUSTER path.** `<workspace>`,
  `<trtllm_repo_path>` and `<checkpoint_path>` are already cluster paths,
  so write them verbatim; do not try to translate them.
- **Send the job's own output somewhere shared.** `sbatch -o /tmp/x.out`
  writes to the compute node's local /tmp and disappears with the
  allocation. Point `-o` inside `<workspace>`.
"""


# --------------------------------------------------------------------------- #
# SOL projection derivation (the projector role's methodology blocks — shared
# with perf-optimize, whose projector prompt composes the same fragments)
# --------------------------------------------------------------------------- #

SOL_PROJECTOR_METHODOLOGY = """\
## The methodology: the `internal-perf-sol-analysis` skill

The skill is the single source of truth for SOL modeling — its α-β-u
model, its per-op recipes, its peaks calculator and
`measure_channels.py`, and the ground rules that come with them. Load
it and follow it: the arithmetic you write down instantiates *its*
formulas, not your own, and your report speaks its vocabulary — **% of
SOL** as the headline (latencies: SOL ÷ measured; throughput: measured
÷ SOL — both ≤ 100%), **MFU** / **MBU** as secondary utilizations,
**gap-to-SOL**, and **bound** ∈ compute / memory / launch (plus comm on
multi-GPU).

What the skill cannot know is this stage's contract:

- **`task.yaml`'s `sol.gpu` (when set) is the part-name hint** for
  `sol_calc.py peaks --part`; the `Skill` load announces the base
  directory its scripts live in.
- **A GPU may not be reachable from here.** On local runs the GPUs sit
  idle between stages (confirm with `nvidia-smi`), so measure the
  latency constants and merge them into the peaks file. Under the
  workflow's Slurm mode you run on a login node — do **not** guess α:
  derive the β/u terms and record the launch-α term as unmeasured in
  *Caveats*.
- **Nothing measured exists yet.** `sol_calc.py analyze` correlates
  *measured* per-op times with their ceilings, and no profiling stage
  has run — do not invent `measured_ms` rows, and never fabricate an
  input to force a script run. The **Analyzer** runs `analyze` after
  you, against its fresh profile and the peaks file you persist. Your
  job is the predictive end-to-end ceiling: instantiate the skill's
  formulas and per-op recipes yourself, with **every formula's actual
  numbers written down** — a projection whose arithmetic cannot be
  re-checked from the report is worthless.
- **Persist the machine-readable peaks file for the Analyzer** — the
  peaks-calculator output, with the measured latency constants merged
  in whenever you measured them — to
  `<workspace>/sol_work/peaks.json`, and record that path in
  *Projection setup*. The Analyzer's measured↔SOL correlation joins
  against this exact file; a projection whose peaks live only in prose
  starves that stage.
- **The structural quantities are this deployment's, and the skill's
  recipes do not supply them:** read them off the checkpoint's
  `config.json` (a misread config silently corrupts every downstream
  number — read it carefully) at the serving precision and the
  parallel mapping (tp/pp/ep, from the config named under *Workspace*)
  — weight bytes per GPU (count only active experts per token for
  MoE), KV-cache bytes read per decoded token at the mean context
  length, FLOPs per token, and on multi-GPU the per-layer collective
  term.
- **If the skill is not available in this environment** (neither name
  resolves), say so in one line, ground what you can from `config.json`
  + internal knowledge into a clearly marked coarse ceiling — and if
  nothing defensible can be grounded, write the unavailable form.
  Never fabricate.

One limit the skill does not state and your report must: the ceiling
models **kernel execution plus per-launch latency only** — no
serving-stack scheduler/host prep, no request queueing, no
dynamic-batching effects. A measured result far below even this α-aware
ceiling therefore points at host/scheduling costs the model does not
price; say so explicitly, it is a valuable signal for the downstream
stages.
"""


# Appended to the projector's prompt only when the workflow resolved
# ``perf-analysis`` instead of the skill above (``sol_methodology``). The
# methodology's own last bullet already says what to do without the
# calculator; this names the skill that replaces it and the one artifact
# that stops being writable.
SOL_METHODOLOGY_FALLBACK = """\
## Fallback: `internal-perf-sol-analysis` is not installed here

This session does not have the skill above, so load the `perf-analysis`
skill your driving message names instead and take its
bottleneck-classification table as the methodology. There is no peaks
calculator and no `measure_channels.py`, so follow the last bullet of
*The methodology*: ground what you can from `config.json` + internal
knowledge into a clearly marked coarse ceiling, and say in one line
that the peaks are not calculator-resolved. Skip
`sol_work/peaks.json` — `sol_calc.py` ships with the missing skill, so
nothing downstream reads it. If nothing defensible can be grounded,
write the *Projection unavailable* form. Never fabricate.
"""


SOL_PROJECTOR_INTERNAL_KNOWLEDGE = """\
## Consulting internal knowledge (reference only)

When the mapping is uncertain or a spec is missing — e.g. to
characterize a model architecture or GPU part the skill's references
leave uncertain — use the `internal-glean-search` skill or the
`internal-glean-specialist` subagent for detailed internal knowledge
(if that skill/subagent exists). **It is consultative** — every
projected number in your report must be reproducible from the
arithmetic you wrote down over named sources (the skill's calculator
output, `config.json`, what you retrieved); never copy a number you
cannot derive, and don't burn the turn searching when the derivation
already stands on cited sources.
"""


# --------------------------------------------------------------------------- #
# SOL projection consumption (appended only when the projector stage is enabled)
# --------------------------------------------------------------------------- #

# Role-neutral correlation recipe shared by both workflows' analyzers —
# each workflow's SOL analyzer extension composes it and names the
# artifact directory the regions/sol JSONs land in.
SOL_CORRELATION_METHOD = """\
### Correlate the fresh profile against the ceiling (`sol_calc.py analyze`)

The projection alone is a predictive end-to-end ceiling; your profile
just produced the measured per-op times the Projector did not have.
Join the two with the skill's calculator — the correlation turns "the
workload is at N% of SOL" into a per-op table naming *where* the gap
physically sits:

1. **Load the `internal-perf-sol-analysis` skill** (via the `Skill`
   tool; fully-qualified
   `trtllm-agent-toolkit:internal-perf-sol-analysis` if the bare name
   is not found). The load announces the skill's base directory — the
   calculator is `<skill_dir>/scripts/sol_calc.py`. Its correlation
   contract and `regions.json` schema are the method; what follows is
   only what the skill cannot know about this stage.
2. **Recover the Projector's peaks file** at
   `<workspace>/sol_work/peaks.json` (its path is recorded in
   `sol_projection.md`'s *Projection setup*). When it carries no
   measured `latencies`/`sms` — the Projector ran without GPU reach —
   run the skill's `measure_channels.py --launch … --merge-into <that
   peaks.json>` yourself: unlike the Projector's stage, a GPU is
   reachable here by construction (you just profiled on it).
3. **Build `regions.json` from your traces — structural facts only.**
   The rows come from the nsys per-kernel sums
   (`cuda_gpu_kern_sum`, NVTX ranges, torch-trace op attribution),
   rolled up into the skill's region keys and schema. The shapes come
   from `sol_projection.md`'s *Arithmetic* (the Projector already
   derived them from `config.json`) — reuse them rather than
   re-deriving. A region whose params you cannot ground stays in
   `other` with a note — **never invent params or `measured_ms` rows**.
4. **Run the calculator** (never hand-compute a SOL number):
   ```bash
   python <skill_dir>/scripts/sol_calc.py analyze \\
       --regions <artifact dir>/regions.json \\
       --peaks <workspace>/sol_work/peaks.json \\
       [--recipes-dir <artifact dir>/sol_recipes] \\
       --out <artifact dir>/sol.json
   ```
   An op family the built-in recipes do not cover is either given a
   recipe under `sol_recipes/` (the skill's `check-recipe` route) or
   left in `other` — those are the only two options.
5. **Transcribe `sol.json` into the `## SOL correlation (measured vs
   ceiling)` section** of `profile_findings.md`: the joined per-op
   table verbatim (region, calls, measured ms, SOL ms, % of SOL,
   MFU %, MBU %, gap ms, bound), the workload-level % of SOL line, and
   one sentence naming the largest-`gap ms` regions — that is where
   the headroom physically sits, and it is the sharpest signal the
   downstream stages get from you.

Degrade honestly: when a precondition fails (projection unavailable,
peaks file missing and the constants unmeasurable, nsys produced no
usable per-kernel table), keep the section with a one-line
`Correlation unavailable: <reason>`, record it in *Caveats*, and move
on — a fabricated correlation is worse than none.
"""


SOL_ANALYZER_CONTEXT = (
    """\
## SOL projection as context (the projector stage ran)

The Projector stage ran before you and left `sol_projection.md` — an
analytical speed-of-light (SOL) ceiling for this model/hardware/
operating point, derived with the `internal-perf-sol-analysis` skill,
with a measured-vs-SOL gap analysis. `Read` it (or call
`read_latest_progress` with `agent: "projector"`) after
`benchmark_results.md` and use it as **context, not evidence**:

- Let the projected headroom (% of SOL) and the bound mix (compute /
  memory / launch) inform which hypotheses you probe hardest and how
  you rank them — e.g. a low % of SOL with a memory bound raises the
  prior on memory-bandwidth causes; a gap far beyond what the ceiling
  can explain points at host/scheduling overhead (the ceiling models
  kernel execution plus per-launch latency only, so serving-stack
  scheduler and queueing costs are invisible to it).
- Measured trace evidence always outranks the projection: when they
  disagree, trust the trace and note the disagreement.
- In `profile_findings.md`, say where the profile **confirms or
  contradicts** the projection (a sentence per ranked hypothesis is
  enough).
- Projected numbers are not measurements — never present a SOL number
  as a measured one. If `sol_projection.md` declares itself
  unavailable, ignore it for ranking, skip the correlation below, and
  record that in *Caveats*.

"""
    + SOL_CORRELATION_METHOD
    + """
Artifact placement in this workflow: `regions.json`, `sol.json`, and
any `sol_recipes/` go under `<workspace>/sol_work/`, next to the
Projector's `peaks.json`.
"""
)


SOL_REPORTER_GUIDANCE = """\
## Projection vs Measured (the projector stage ran)

The Projector left `sol_projection.md` — an analytical speed-of-light
(SOL) ceiling derived with the `internal-perf-sol-analysis` skill, with
% of SOL / MFU / MBU numbers and a measured-vs-SOL gap analysis. `Read`
it with the other inputs and add one section to `performance_report.md`,
placed **between "Profiling Findings" and "Main Bottleneck"** (the HTML
companion mirrors it like every other section):

```
## Projection vs Measured

<The measured-vs-SOL table lifted from sol_projection.md (throughput,
TTFT, TPOT with % of SOL, plus measured MFU/MBU), the projected bound
mix (compute / memory / launch), and what the gaps mean: % of SOL sizes
the theoretical headroom, and the bound names which side it is on.
When profile_findings.md carries a **SOL correlation (measured vs
ceiling)** section, also lift its joined per-op table (region / calls /
measured ms / SOL ms / % of SOL / gap ms / bound) — it localizes the
same headroom per op — and name the largest-gap regions; when the
correlation was unavailable, say so in one line rather than
substituting. State explicitly how this projection moves (or does not
move) the verdict.>
```

Weighing rules:
- **Weigh the projection when deciding the Main Bottleneck and when
  ranking Recommendations**: the SOL headroom sizes the win (a fix
  cannot recover more than the ceiling says is available on that side),
  and the projected bound mix corroborates or challenges the Analyzer's
  ranked hypotheses — the per-op correlation table, when present, is
  the sharpest tie-breaker (measured rows against their own ceilings).
  State in the Main Bottleneck section how the projection was weighed.
- The projection is a model, not a measurement — when it conflicts with
  trace evidence, measured evidence wins, and the conflict is worth a
  sentence.
- The ceiling models kernel execution plus per-launch latency only — a
  measured result far below it often indicates serving-stack
  scheduler/queueing costs the model does not price; treat that as
  supporting evidence for host-side bottleneck categories, not as a
  contradiction.
- If `sol_projection.md` is missing or declares itself unavailable,
  the section must honestly say **"Projection unavailable (<reason>)"**
  and the verdict falls back to measured evidence alone — never
  fabricate projected numbers.
"""


# --------------------------------------------------------------------------- #
# Bottleneck taxonomy + HTML companion (reporter)
# --------------------------------------------------------------------------- #

BOTTLENECK_TAXONOMY = """\
## Bottleneck taxonomy

Classify the **single dominant** bottleneck into exactly one primary
category (note secondary factors separately). Tie the verdict to concrete
evidence rows from the benchmark + profile findings — never assert a
category without the signal that supports it.

- **Compute-bound** — GPU math units saturated. Signal: high GPU busy %,
  GEMM/attention/tensor-core kernels dominate kernel time, near-roofline
  FLOPs, throughput scales with batch up to a compute ceiling. Common in
  prefill / large-batch decode.
- **Memory-bandwidth-bound** — HBM bandwidth saturated. Signal: memory-bound
  kernels dominate (elementwise, norms, KV gather/scatter, dequant),
  high DRAM throughput at modest FLOPs, decode-phase TPOT dominated by
  weight/KV reads. Common in low-batch decode.
- **KV-cache-capacity-bound** — serving throughput limited by how many
  requests fit in KV cache. Signal: low KV-cache free-block headroom /
  high utilization, requests queued / preempted, concurrency capped below
  the requested level, throughput rises if `kv_cache_free_gpu_memory_fraction`
  or quantization increases.
- **Kernel-launch / host-overhead-bound** — GPU starved by the host. Two
  distinct sub-causes share this bucket; identify **which one** dominates
  before prescribing a fix, because the fixes differ:
  - *Kernel-launch overhead* — launching/dispatching the model forward
    dominates. Signal: many tiny kernels, launch calls dominate CUDA-API
    time, eager (non-CUDA-graph) execution, idle made of short per-launch
    gaps. This is what **CUDA graphs / overlap scheduler** collapse: graph
    replay wraps the model forward, removing the per-kernel launch cost
    inside it.
  - *Host-prep / scheduler exposed* — a host phase (input preparation,
    block-table/index math, host-device `.item()` syncs, request
    scheduling) runs on the timeline and is not hidden by GPU work.
    Signal: a named host phase (e.g. `_prepare_inputs`) whose wall time
    rivals or exceeds the GPU forward, high `.item()` /
    `cudaStreamSynchronize` counts, long (>100 µs) idle gaps. **CUDA graphs
    do not remove this** — the host prep runs before/around the replayed
    forward, not inside it; the fix is cutting host work and removing
    host-device syncs from the hot path (and when it also blocks graph
    capture, fix it first). Low GPU busy % at low batch is common to both.
- **Communication-bound (multi-GPU)** — collectives dominate. Signal:
  NCCL/all-reduce/all-gather kernels are a large share of time, GPUs wait
  on communication, scaling efficiency drops with TP/PP/EP size.

If two categories are close, say so and rank them; the Executive Summary
still names one headline bottleneck.
"""


HTML_COMPANION = """\
## HTML companion (`performance_report.html`)

Produce a **single self-contained** HTML file alongside the markdown — all
CSS/JS inline, **no external CDN, font, or asset URLs** so it opens
offline. It presents the *same content* as `performance_report.md` (same
sections, same numbers, same verdict) in a clean, interactive form.

**Required structure (top-down):**

1. `<!DOCTYPE html>` with `<html lang="en">`, a `<title>` matching the
   report's H1, and `<meta name="viewport">`.
2. Inline `<style>`: clean readable font stack, generous line-height,
   ~800–900 px max content width, and light/dark mode via
   `@media (prefers-color-scheme: dark)`.
3. A **sticky table-of-contents nav** listing every H2, each linking to
   the section's slugified anchor id (`#executive-summary`, etc.).
4. The main `<article>` body, sections in the same order as the markdown
   (Executive Summary, Configuration, Benchmark Results, Pareto Curve —
   Pareto-curve mode only, Profiling Findings, Main Bottleneck,
   Recommendations), each heading carrying a stable id.
5. Metric tables are real HTML `<table>`s (same columns/values as the
   markdown). The **Main Bottleneck** verdict is visually prominent
   (e.g. a callout box).
6. Inline `<script>` at the end of `<body>`.

**Required charts (self-contained — no chart library, no CDN):** embed
each chart's data as a JSON array in the inline script and render it to
inline SVG with your own small renderer. Style via CSS variables so both
color schemes stay readable, and never plot a value that differs from
the section's table — the table is the source of truth.

- **Top-kernel share bars** — at the top of *Profiling Findings*: one
  horizontal bar per row of the top-kernels table (GPU-time share,
  sorted descending), each labeled with the kernel name — abbreviate
  template-heavy names to a distinctive stem — and its share, with a
  hover tooltip (an SVG `<title>` is enough) carrying the full name and
  exact value. Render it only when the findings carry a top-kernels
  table (nsys ran); with no table, omit the chart rather than plotting
  invented numbers. Further charts (e.g. top operators) are welcome
  under the same self-contained rules.
- **Pareto curve** — at the top of *Pareto Curve*, only in Pareto-curve
  mode (`benchmark.concurrency` is a list): **x = tok/s/user,
  y = tok/s/gpu**, the measured curve as one polyline with a marked
  point per concurrency, each labeled `c=<n>` and carrying an SVG
  `<title>` tooltip with the exact x/y values. Pad both axis domains
  around the data (do not force zero) and put the axis names + units on
  both axes. When the report also carries per-point SOL-projected
  values (the projector ran in curve mode), overlay the projected curve
  as a second polyline distinguished by more than hue alone (e.g.
  dashed) plus a legend. In scalar mode, or when the curve summary
  table is absent, omit the chart and the section.

**Required interactivity:**

- **TOC scroll-spy** — the entry for the section in view gets an `active`
  class as the reader scrolls.
- **Collapsible H2 sections** — clicking a heading toggles a `collapsed`
  class on its body; default expanded.
- **Print-friendly** — hide the TOC and force-expand all sections in
  `@media print`.

**Faithfulness rule:** the HTML is not a remix — same sections, same
tables, same bottleneck verdict and evidence as the markdown, and charts
that plot exactly the numbers in the tables they sit above. If you
revise the markdown, revise the HTML in the same turn.
"""


# --------------------------------------------------------------------------- #
# Shared rigor / progress rules
# --------------------------------------------------------------------------- #

EVIDENCE_DISCIPLINE = """\
## Evidence discipline

- **Never fabricate numbers.** Every metric, kernel name, or percentage
  you report must come from a file you actually produced (the benchmark
  JSON, `nsys stats` output, the torch trace, server logs). If a run
  failed or a tool was unavailable, say so plainly — do not invent
  plausible-looking results.
- **Record exact commands.** Anyone reading the workspace must be able to
  reproduce your run from the commands you wrote down.
- **No conversational filler.** Jump straight into the work.
"""
