# Disagg Perf-Sanity Cache-Transceiver Precheck

A fast go/no-go network check that runs **inside every disaggregated
perf-sanity CI stage, before the real test**. It brings up one MPI instance
per ctx/gen server with the *same* topology, node placement, UCX environment,
and `cache_transceiver_config` as the real test, transfers deterministic KV
data ctx → gen through the cache transceiver, and verifies the received
bytes. If the transfer hangs, errors, or mismatches, the stage fails
immediately with a specific verdict — the (expensive) model bring-up never
starts.

## Parity with the real test — by construction

| Requirement | How it is guaranteed |
|---|---|
| Same UCX env vars (incl. the `unset UCX_TLS` cases) | `jenkins/scripts/perf/submit.py` builds the precheck commands from the **same** `ucx_tls_cmd` + `$CTX/GEN_WORKER_ENV_VARS` strings as the worker steps; `slurm_precheck_run.sh` sources the same `slurm_env_setup.sh` (the `UCX_TLS=tcp` fixup) as `slurm_run.sh`. |
| Same instance count / parallelism | One precheck `srun` per ctx/gen server with the same `-N/--ntasks/--ntasks-per-node/--mpi=pmix` and the same node slices (`-w`) as the real server steps (`slurm_launch_draft.sh`). TP/PP/CP/attention-DP come from the same `worker_config`. |
| Same transceiver config | `CacheTransceiverConfig(**yaml["worker_config"][role]["cache_transceiver_config"])` — the yaml block is passed through verbatim (backend, `max_tokens_in_buffer`, timeouts, ...). |
| Same KV cache manager version + transceiver runtime | The launch generator forwards the real test's `LLM_MODELS_ROOT` into every precheck process. Explicit per-side `kv_cache_config.use_kv_cache_manager_v2` wins; absent means "auto" and requires a registered model class, then resolves via `get_preferred_kv_cache_manager_version()`. `transceiver_runtime: auto` resolves via `get_preferred_transceiver_runtime()` (NIXL-gated) — both through the same llm_utils code serving uses. An unresolved manager-version `auto` setting fails with INIT_ERROR instead of silently assuming V1. V2 requires the Python transceiver (the C++ one only supports V1); a V2+CPP combination also fails fast. |

Asymmetric layouts (ctx dep4 → gen dep16, ctx pp8 → gen tp32, ...) are
supported: data is seeded per (request, **global** layer) and constant along
the KV-head axis, so the receiver regenerates its expected slice locally under
any TP/PP resharding. KV shape (layers/heads/head_dim, MLA vs GQA, MTP nextn
layers) is read from the real model's `config.json` under `$LLM_MODELS_ROOT`.

## Enabling / disabling

**On by default** for every disaggregated perf-sanity test. To opt out:

- per test yaml:

  ```yaml
  cache_transceiver_precheck:
    enabled: false
    # optional overrides (defaults in precheck_config.PRECHECK_DEFAULTS):
    # request_lengths: [1024, 8192]
    # num_requests: 2
    # wave_timeout_s: 180
    # wireup_timeout_s: 600      # first-rep NIXL agent wire-up allowance
    # step_timeout_s: 1200       # default; yaml overrides are capped at 1800
  ```

- or globally at launch-script generation time: `TRTLLM_DISAGG_CT_PRECHECK=0`
  (kill switch; `=1` force-enables). The env var, when set, overrides the yaml
  either way.

## Timeouts

The first rep of the schedule (the warmup rep) additionally budgets
`wireup_timeout_s` (default `min(600, 150 * max world size)`). "Wire-up" is
the one-time exchange of agent and registered-memory metadata and creation of
the peer endpoints before payload transfer. The C++ NIXL
path pays a one-time serialized `fetchRemoteMD` metadata exchange per
(receiver rank, ctx rank) agent pair, and cold cross-rack fetches were
measured at 100-170s each — real serving absorbs this as slow first requests,
so the precheck does too. The complete precheck has a topology-independent
20-minute default external limit; exceptional yaml overrides may extend it to
at most 30 minutes. Exceeding the configured limit is a network/environment
failure, even with multiple peers. Later reps run under the tight
`wave_timeout_s`, which is what actually catches hangs. Set `PRECHECK_DEBUG=1`
in the worker env to raise the C++/Python transceiver log levels when debugging
a stall.

The Python sender's `kv_transfer_timeout_ms` is also a real request deadline.
If block-all returns without every expected request completed—or any rank
reports failure, cancellation, or unsynchronized receive work—the precheck
retains the KV pages, persists an ownership-fatal verdict, and hard-aborts the
instance without running transceiver/cache-manager finalizers. Pages are freed
only after exact completion on every rank (plus CUDA synchronization on gen).

## Failure output

The sbatch log gets a summary block: per-instance verdicts
(`status/*.status`), the tail of each failing step log, and UCX red-flag lines
(`sw-emul` host-staged tcp fallback, UCX ERROR/WARN). Full artifacts under
`<testOutputDir>/cache_transceiver_precheck/`:

```
status/{ctx,gen}_<i>.json     # per-case detail: PASS/TIMEOUT/TRANSFER_ERROR/
                              # MISMATCH/INIT_ERROR + reason + UCX env snapshot
status/{ctx,gen}_<i>.status   # one-line verdict (parsed by the launch script)
logs/{ctx,gen}_<i>.log        # merged per-rank logs (UCX_PROTO_INFO=used table)
csv/gen_<i>/<uuid>_<rank>_recv.csv  # C++ transceiver per-request bandwidth
csv/ctx_<i>/<uuid>_<rank>.csv       # Python transceiver per-task perf
                                    # (KVSendTask throughput on the ctx side)
```

## Manual runs

```bash
# Inspect what a yaml resolves to (no GPU needed):
LLM_MODELS_ROOT='<models>' python3 run_precheck.py --role gen --server-idx 0 --dry-run \
    --config ../disaggregated/<test>.yaml --work-dir /tmp/ct --llm-src <repo>

# On a SLURM allocation: one srun per instance, e.g. ctx dep4 + gen dep8:
LLM_MODELS_ROOT='<models>' srun -N1 --ntasks=4 --mpi=pmix python3 run_precheck.py \
    --role ctx --server-idx 0 --config <yaml> --work-dir <shared-dir> --llm-src <repo> &
LLM_MODELS_ROOT='<models>' srun -N2 --ntasks=8 --mpi=pmix python3 run_precheck.py --role gen --server-idx 0 \
    --config <yaml> --work-dir <shared-dir> --llm-src <repo> &
wait
```

Tests:

- Config resolution (CPU-only):
  `pytest tests/unittest/others/test_cache_transceiver_precheck_config.py`
- Driver logic + internal-API contract (the precheck drives unstable
  TRT-LLM internals via `run_precheck.load_internal_apis()`; the contract
  tests catch upstream renames in pre-merge CI):
  `pytest tests/unittest/others/test_cache_transceiver_precheck_run.py`
- End-to-end on one GPU (one `mpirun` world per ctx/gen instance, real
  NIXL/PYTHON transfer + verdicts, incl. multi-instance and asymmetric TP):
  `pytest tests/unittest/disaggregated/test_cache_transceiver_precheck_e2e.py`

Lineage: adapted from `examples/disaggregated/slurm/cache_transceiver_test`
(the UCX tuning harness), reduced to a single sweep and extended to
asymmetric parallelism, attention DP, and multi-instance pairing.
