---
id: case-cc-blocking-d2h-copies-worker-thread
type: regression-case
family: execution-and-graph
module: sampler
maturity: full
regression_class: [sync-introduced]
signals: [gpu-idle-between-steps, throughput-drop, itl-increase]
subsystems: [sampler, runtime-python]
introduced_via: [pre-existing-gap]
phase: [decode]
patterns: [pattern-per-step-sync-added]
nvbugs: ["5508301"]
commits: ["2d33ae94d532"]
success_prs: [8463]
failed_prs: []
---

# Under confidential compute every D2H copy is blocking, so the sampler stalled the main thread each step

> Part of the [Sampler regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5508301` · commit `2d33ae94d532` · PR #8463 —
  "[https://nvbugs/5508301][feat] Move D->H copies to a worker thread when
  confidential compute is active". (The title is stored truncated in some views as
  "…Move D->H copies to a worker thread when confidential…" + "…compute is
  active"; that is a display artifact, not two PRs.)
- **Symptom:** throughput loss and GPU idle between decode steps on
  **confidential-compute (CC) enabled** systems only; the same build on a non-CC
  box is unaffected. The PR states no percentage, model or SKU —
  the discriminator is the platform mode, and that is the number-free fact worth
  carrying.
- **Root cause:** with CC active, device→host copies must route through a bounce
  buffer and **cannot be asynchronous** — a `non_blocking=True` copy that is free
  on a normal system becomes a blocking copy under CC. The sampler issues such
  copies every step (token ids, finish reasons, log probs), so on CC the main
  Python thread waits for each one inline, and the wait lands exactly where the
  overlap scheduler expects to be issuing the next step's work. Nothing in the
  sampler code changed; the *platform* changed the cost of an idiom the code
  already used, which is why this reads as "CC is slow" rather than as a code
  defect.
- **How introduced:** `pre-existing-gap`. No culprit commit — the copies were
  always there and were always free on the platforms they were written for.
- **Fix mechanism:** move the copies off the critical thread. A new
  `AsyncWorkerMixin` owns a **single-worker** `ThreadPoolExecutor` and its own CUDA
  stream; `_copy_to_host` clones on the worker stream, records a `cuda.Event`, and
  performs the blocking `dest.copy_` there. `SamplerEvent` grows from a bare CUDA
  event to `{cuda_event, worker_futures}` so consumers can wait on both. The whole
  path is gated: `confidential_compute_enabled() or llm_args.sampler_force_async_worker`
  — i.e. **off by default on normal systems**, with an explicit override for
  testing. That gate is the design decision to notice: the worker thread is a
  latency win only when the copy is forced blocking, and a pure cost otherwise.
- **Detection signal:** the platform check itself —
  `python -c "from tensorrt_llm._utils import confidential_compute_enabled as f; print(f())"`
  — decides whether this case can apply at all. On a CC box, a profile shows GPU
  idle in the decode loop whose duration scales with the *number of D2H copies per
  step* and not with token count, with the main thread inside a copy. Force the
  comparison with `sampler_force_async_worker=True/False` on the same hardware.
- **Prevention/guard:** guards were **added**, which is unusual for this class and
  worth imitating: the new `sampler_force_async_worker` flag is pinned in
  `tests/unittest/api_stability/references/llm.yaml`, and five accuracy tests are
  parametrized over `sampler_async_worker ∈ {True, False}` so the async path is
  exercised on ordinary CI hardware; shutdown uses `shutdown(wait=True)`. Two
  reviewer notes went unaddressed and remain true of the code: there is **no lock**
  around worker lifecycle (start/shutdown races are possible), and the correctness
  asserts vanish under `python -O`.
- **Generalizes to:** `pattern-per-step-sync-added`; carries to any per-step host
  read on a platform that redefines copy semantics (CC/NVLE bounce buffering,
  IOMMU-restricted paths, virtualized GPUs), and to the general rule that a
  *platform mode* can convert an existing async idiom into a per-step sync with no
  code change — so bisecting a CC-only regression against non-CC baselines finds
  nothing. See `case-nvle-only-treated-as-cc` for the follow-on defect in the same
  platform-detection helper this PR introduced, and
  `case-sampler-stop-words-implicit-d2h-sync` for the same overlap-collapse
  mechanism caused by code rather than platform.
