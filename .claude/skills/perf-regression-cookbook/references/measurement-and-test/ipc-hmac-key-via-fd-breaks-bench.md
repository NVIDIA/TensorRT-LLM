---
id: case-ipc-hmac-key-via-fd-breaks-bench
type: regression-case
family: measurement-and-test
module: measurement-and-test
maturity: full
regression_class: [measurement-artifact]
signals: [perf-ci-bar-failure, throughput-drop, startup-time-increase]
subsystems: [runtime-python, perf-test-config]
introduced_via: [new-feature]
phase: [any-phase]
patterns: [pattern-startup-handshake-fd-not-inherited]
nvbugs: ["6388787"]
commits: ["a422420db98c", "0d97e9c76fd3"]
success_prs: [14782, 15961]
failed_prs: []
---

# IPC HMAC key passed by file descriptor deadlocks the benchmark launcher

> Part of the [Measurement & test regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6388787` · commit `0d97e9c76fd3` · PR #15961 —
  "[https://nvbugs/6388787][fix] Revert Pass IPC HMAC key through file
  descriptor (#15654)"; related: nvbug `6244695` · commit `a422420db98c` ·
  PR #14782 — "[https://nvbugs/6244695][fix] Revert Pass IPC HMAC key through
  file descriptor" (the **first** round of the identical defect).
  **This feature has landed and been reverted twice**, with all four PRs
  carrying the same title: #14378 (`50ca49f8c53d`, 2026-05-28) reverted by
  #14782 (2026-06-01), then #15654 (`48fc7537baf0`, 2026-07-03) reverted by
  #15961 (2026-07-06). Both landings were security hardening (`5972776` and
  `6208457`), so neither is in `nvbugs:`: they motivated the culprit, they
  are not the regression. Round 1's own observable, `6244695` (the
  post-merge perf test failing with `BlockingIOError`), is a functional bug
  and likewise stays out of the frontmatter; `6388787` is the only
  performance bug, and it is the one #15961's title cites.
- **Symptom:** two shapes, and the second is the dangerous one.
  **Round 1 failed loudly** — `BlockingIOError: [Errno 11] Resource
  temporarily unavailable` out of
  `executor/utils.py:_read_spawn_proxy_process_ipc_hmac_key_fd`, so the perf
  test errored out and was diagnosed in days. **Round 2 failed silently**: the
  benchmark hung for 30–90 min with no output after
  `[llmapi] start MpiSession with <N> workers`, all N workers alive at 0% GPU
  utilization and ~4 MiB VRAM, until the harness stall-timeout killed it.
  Reported as a *perf regression* from the deepseek_v3.2_fp4 GB300 QA sweep
  (`6388787`, a Total_Token_Throughput regression against 1.3.0rc17) — the
  bar tripped and a percentage was quoted, because a truncated or absent run
  reaches the comparison as a degraded number, not as an error. The same
  window also surfaced as a functional timeout: `6435109`, a functional bug
  in which Disagg-PerfSanity GB200 stages time out because the ctx/gen
  `trtllm-serve` never becomes ready — it names the mechanism outright.
  A fourth shape is quoted in #15961's own description: the
  `nemotron_3_ultra_550b_nvfp4-serve` `/health` endpoint "did not become ready
  within 3600s" on **both** baseline and candidate wheels on 2026-07-05 — the
  tell that this is not a candidate-vs-baseline delta at all.
  **6388787 is a two-root-cause bug and the two must not be folded:** its other
  root cause is a DeepGEMM warmup-bucket hole fixed by PR #16178, recorded
  separately in the instability cookbook
  (`references/warmup-and-jit/deepgemm-paged-mqa-logits-prewarm.md`). #15961 and
  #16178 share no files; each is a fix for one of the two mechanisms, and
  neither is a failed attempt at the other.
- **Root cause:** the HMAC key was advertised to the child through an inherited
  file descriptor (env var `TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY_FD`), but the
  perf harness spawns the server via
  `subprocess.Popen(server_cmd, ...)` in `tests/integration/defs/perf/`
  **without `pass_fds`**, and `Popen` defaults to `close_fds=True`. The fd
  number therefore survives in the environment while the fd itself does not
  survive the fork+exec into the launcher, so the child's
  `os.set_blocking(fd, True); os.read(fd, 4096)` waits forever on a descriptor
  that is either closed or aliased to something that never writes. Blocking
  mode is what converts round 1's immediate `EAGAIN` into round 2's permanent
  deadlock: the same missing-fd condition raises in one and hangs in the other.
- **How introduced:** `new-feature` — security hardening replaced env-var/argv
  transport of the HMAC key with fd transport. The hardening is correct in
  itself; what it did not survive is an
  intermediate process in the launch chain that closes inherited descriptors.
- **Fix mechanism:** both times, a **full revert** of the fd transport
  (`commands/serve.py`, `executor/ipc.py`, `executor/utils.py`,
  `executor/worker.py`, `llmapi/trtllm-llmapi-launch`), restoring the previous
  key transport. Pure Python in both directions — **no rebuild is needed** to
  get out of a bad window, which matters when a bisect lands inside one.
  #15961 also deleted **70 lines** from
  `tests/integration/test_lists/waives.txt`: that un-waiving is the honest
  measure of the blast radius, and a good reason to check `waives.txt` churn
  when a "perf regression" spans many unrelated cases at once.
- **Detection signal:** the run produces no output after
  `[llmapi] start MpiSession with <N> workers` while N worker processes sit at
  0% GPU — a startup deadlock, not a slow model. Check whether your commit is
  inside either broken window before believing any number from it:
  `git merge-base --is-ancestor 48fc7537ba HEAD && ! git merge-base --is-ancestor 0d97e9c76f HEAD && echo "IN BROKEN WINDOW (round 2)"`
  (round 1 is the same test with `50ca49f8c5` / `a422420db98c`). Confirm the
  mechanism with
  `grep -rn 'IPC_HMAC_KEY_FD\|_read_spawn_proxy_process_ipc_hmac_key_fd' tensorrt_llm/executor/`
  and check the harness side for the missing `pass_fds` with
  `grep -n 'Popen(' tests/integration/defs/perf/test_perf_sanity.py`.
- **Prevention/guard:** the gap is a test-topology gap, not a missing test.
  #15654 **added** unit tests (`tests/unittest/executor/test_ipc.py`,
  `tests/unittest/executor/test_launcher_envs.py`) and they passed while the
  real path deadlocked, because they exercised the fd handshake in-process and
  never crossed the harness's `bash` → `pytest` → `Popen(close_fds=True)` →
  `mpirun` chain. Any change to how a secret/handle reaches a worker needs one
  end-to-end test through the *launcher actually used in CI*; asserting the
  handshake in isolation cannot see an fd that a middle process dropped. As a
  review checklist item: an fd passed by number through an env var is only
  valid if every intervening `Popen`/`exec` in the chain preserves it — audit
  for `pass_fds=` at each hop, and prefer a transport that fails loudly
  (non-blocking read, or a timeout) over one that blocks forever.
- **Generalizes to:** `pattern-startup-handshake-fd-not-inherited` — a
  *non-compute* product change breaks process bring-up, so a perf bar trips
  with no compute commit in range. Carries to: any handle/secret passed by fd
  or by inherited socket through a multi-hop launcher (`trtllm-llmapi-launch`,
  `mpirun`, `srun`, pyxis/enroot); blocking `os.read` on any resource a parent
  may not have provided; and QA "regression" reports whose real content is a
  timeout — where the first question is whether the server ever reached ready,
  not which kernel got slower. Sibling read: three other cases in these
  cookbooks had to exclude this exact confounder from their own windows
  (`kernel-and-fusion/mamba2-flashinfer-head-group-ratio-gate.md`,
  `allreduce-host-overhead-small-model-tp.md`, and the instability cookbook's
  `deepgemm-paged-mqa-logits-prewarm.md`) — a defect that pollutes three
  unrelated investigations is worth recognizing on sight.
