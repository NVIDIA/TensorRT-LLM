# Pareto trace-replay against `trtllm-serve`

Tools for sweeping throughput/latency Pareto frontiers by replaying a compact
execution trace against a running `trtllm-serve` instance.

Two scripts, two responsibilities:

- `trace_replay_client.py`: pure OpenAI client. Runs **one** ladder point —
  opens an `openai.AsyncOpenAI` to `--base_url`, wraps it in a `TRTOpenaiWorker`,
  submits `total_sessions` `ReplayEngine.launch_trace(trace)` tasks with at
  most `concurrency` of them in flight at once (an `asyncio.Semaphore` gates
  admission), records throughput / per-session latency statistics, and writes
  a single **step JSON** for that `(max_batch_size, total_sessions, concurrency)`
  point.
- `trace_replay_pareto_aggregate.py`: pure offline aggregator. Reads the N
  step JSONs produced by the client (one per ladder step), assembles them
  into a `trace_replay_pareto_frontier.v4` combined report, then delegates
  to the plotting helpers in `../tracing/` to write the two Pareto PNGs.

The server lifecycle (start `trtllm-serve` with a step-specific
`max_batch_size`, poll `/health`, stop it, wait for port release) lives in
the Slurm driver script, not in Python — signals, SLURM steps and port
ownership belong in shell. That driver is also where the three load knobs
(`max_batch_size` / `total_sessions` / `concurrency`) get pinned or left to
follow the ladder on a per-step basis. See `run_trace_pareto_server_tep4.sh`
at the rysun workspace root for the reference driver.

Companion package: `../benchmarks/` — the shape of these scripts mirrors
`benchmarks/{agent,chat}_benchmark.py` (external server + `AsyncOpenAI` +
`TRTOpenaiWorker`). The older in-process `LLM` API variant at
`../tracing/run_trace_replay_pareto_frontier.py` remains untouched and is
independent of this package.
