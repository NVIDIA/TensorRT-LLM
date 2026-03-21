# Review: SJF Implementation Plan

Reviewed against:
- Design doc: `dev_docs/sjf-waiting-queue/designs/2026-03-21-sjf-waiting-queue.md`
- Plan: `dev_docs/sjf-waiting-queue/plans/2026-03-21-implementation.md`
- Existing code cross-referenced: `waiting_queue.py`, `llm_args.py`, `_util.py`, `py_executor.py`, `test_waiting_queue.py`

---

## Summary

The plan is well-structured and architecturally sound. It correctly identifies all files that need modification, the dependency ordering between task groups is valid, and the design-to-plan alignment is close to 1:1. Several issues were found during cross-referencing with the actual codebase — two of which are Must-Fix because they will cause runtime errors or incorrect behavior, and one of which is a Must-Fix due to a coding-guidelines violation that will likely trigger a pre-commit hook failure. The remaining issues are Should-Fix or suggestions.

---

## Findings

### Must-Fix

**M1: `SjfConfig` must not define `__init__` — use `model_post_init` for internal state setup**

The design doc's `SJFWaitingQueue.__init__` directly constructs `self._config = sjf_config or SjfConfig()`. This is fine for the queue class itself since `SJFWaitingQueue` is a plain Python class, not a Pydantic model. However, the design doc shows `SjfConfig` as a `StrictBaseModel` and the plan inherits that. `CODING_GUIDELINES.md` (line 414) explicitly prohibits defining `__init__` on Pydantic models: "Do not define `__init__` methods — this bypasses Pydantic's validation and type coercion". The plan does not note this restriction. `SjfConfig` itself is straightforward and needs no post-init, so this is a non-issue for `SjfConfig` directly — but the plan should explicitly state that `SjfConfig` is a pure field-declaration class with no `__init__`, and that `SJFWaitingQueue` (non-Pydantic) is allowed to define `__init__`.

Action required: add a clarifying note in the plan that `SJFWaitingQueue` is NOT a Pydantic model (it inherits only from `WaitingQueue` ABC), and its `__init__` is therefore valid Python. The plan currently uses identical formatting for both, which could mislead an implementer into accidentally subclassing `StrictBaseModel`.

**M2: `Field(gt=0)` / `Field(ge=0)` inline constraints are NOT the right pattern for this codebase; use Pydantic type aliases instead**

The plan specifies:
```
length_median (int, default=32768, gt=0), time_median (float, default=5.0, gt=0),
length_weight (float, default=0.5, ge=0), time_weight (float, default=0.5, ge=0)
```

`CODING_GUIDELINES.md` line 427 states: "Prefer `PositiveInt`, `NonNegativeInt`, `NonNegativeFloat`, `PositiveFloat`, `Field(gt=0)`, `Field(ge=0)`, etc. for numeric constraints." Inline `Field(gt=0)` is permitted, so this is technically allowed. However, the codebase already imports `PositiveInt`, `NonNegativeFloat`, `PositiveFloat` from pydantic (confirmed at `llm_args.py:20`) and uses them pervasively (e.g., `max_concurrency: Optional[PositiveInt]`, `sa_spec_threshold: PositiveInt`). Using `PositiveInt` for `length_median` and `PositiveFloat` for `time_median` is the preferred idiomatic form in this codebase.

The plan's notation `gt=0` / `ge=0` is ambiguous about whether these are `Field()` kwargs or type annotations. If a developer reads `gt=0` and writes `Field(default=32768, gt=0)` in Pydantic v2, that will silently be ignored because `Field` does not accept `gt` directly in Pydantic v2 — the correct form is `Annotated[int, Field(gt=0)]`. This is a runtime correctness issue.

Action required: update the plan to specify:
- `length_median: PositiveInt = Field(default=32768, description="...")`
- `time_median: PositiveFloat = Field(default=5.0, description="...")`
- `length_weight: NonNegativeFloat = Field(default=0.5, description="...")`
- `time_weight: NonNegativeFloat = Field(default=0.5, description="...")`

**M3: Task 3.2 description is incorrect — `PyExecutor.__init__` does not need a new `sjf_config` parameter**

The plan says:

> Task 3.2: Update `PyExecutor.__init__` to accept and pass `sjf_config`

But looking at the actual call chain:

1. `_util.py:1327–1354` extracts `waiting_queue_policy` and passes it as a kwarg to `PyExecutor.__init__`
2. `py_executor.py:499–500` calls `create_waiting_queue(waiting_queue_policy)` internally

`create_waiting_queue` is called inside `PyExecutor.__init__`, which means `sjf_config` would need to be threaded through as a separate parameter. However, there is a cleaner alternative that avoids widening `PyExecutor.__init__`'s signature: pass the full `scheduler_config` object to `PyExecutor` instead (or pass the resolved `SjfConfig` alongside `waiting_queue_policy`).

The plan does not analyze the actual `PyExecutor.__init__` signature (line 258–289), which already has a large number of parameters. Adding `sjf_config: Optional[SjfConfig] = None` is mechanically correct but the plan should acknowledge this and confirm the approach is intentional, not accidental (versus, e.g., passing `scheduler_config` directly). The plan as written implies `sjf_config` is a standalone parameter, which diverges from the existing pattern where `waiting_queue_policy` is extracted in `_util.py` and passed as a plain value.

Action required: the plan should explicitly document whether:
- Option A (current plan): add `sjf_config: Optional[SjfConfig] = None` to `PyExecutor.__init__` — feasible but adds a parameter;
- Option B: extract `sjf_config` in `_util.py`, pass both `waiting_queue_policy` and `sjf_config` to `create_waiting_queue` directly inside `_util.py` before constructing `PyExecutor`, and instead pass the constructed `WaitingQueue` object to `PyExecutor`.

Option B is architecturally cleaner: `PyExecutor` already calls `create_waiting_queue` at line 499, so the simplest refactor is to extract `sjf_config` in `_util.py` and pass it alongside `waiting_queue_policy` to `PyExecutor`, which then forwards both to `create_waiting_queue`. Either option works — the plan needs to commit to one.

---

### Should-Fix

**S1: `prepend_requests` ordering semantics are under-specified in the plan**

The plan says:

> `prepend_requests`: extend front of `_prepended` (preserve input order)

The design doc provides the critical context (section 4.4 bottom): `request_utils.py:155` calls `waiting_queue.prepend_requests(reversed(pending_requests))`, and `FCFSWaitingQueue.prepend_requests` uses `deque.extendleft`, which reverses the iterable again, restoring original order. For `SJFWaitingQueue`, the plan says "preserve input order" but does not spell out what "input order" means for the caller.

Cross-referencing `request_utils.py:155`: `reversed(pending_requests)` is passed in. If `SJFWaitingQueue.prepend_requests` does `list(requests) + self._prepended`, the resulting front-of-queue order will be `reversed(pending_requests)`, NOT `pending_requests`. This is the opposite of what `FCFSWaitingQueue` produces.

The design doc (section 4.4) addresses this correctly but using the phrase "保持传入顺序" (preserve input order as received). The plan should spell out explicitly: "the implementation receives `reversed(pending_requests)` from the caller; to maintain the same front-of-queue semantics as FCFS, `prepend_requests` must re-reverse by prepending in `list(requests) + self._prepended` order." This is a subtle correctness point that must be stated clearly in the implementation plan before a developer touches it.

**S2: Test case "wait time aging" must be verifiable without real wall-clock sleep**

The plan lists "Wait time aging: long request eventually gets priority" as a test case, but provides no guidance on how to make this deterministic in a unit test. Real `time.sleep()` calls slow down CI. The `_get_arrival_time` method in the design uses `getattr(item.request, 'py_arrival_time', None)` — a Mock-able attribute — which means test code can inject a synthetic `py_arrival_time` value that simulates a request having waited a long time, without any actual sleep.

The plan should note that aging tests will inject `py_arrival_time` values directly on mock requests to simulate elapsed wait time, rather than using `time.sleep()`.

**S3: `_to_pybind` in `SchedulerConfig` does not include `waiting_queue_policy` and will not need updating, but this needs to be verified**

Looking at `SchedulerConfig._to_pybind` (lines 1914–1921), the method constructs a `_SchedulerConfig` C++ binding but does NOT include `waiting_queue_policy` — it only passes `capacity_scheduler_policy`, `context_chunking_policy`, and `dynamic_batch_config`. This is expected because `waiting_queue_policy` is a Python-side-only concept for the PyTorch backend. Adding `sjf_config` (also Python-side-only) to `SchedulerConfig` requires no changes to `_to_pybind`. The plan correctly omits this, but should add an explicit note confirming it — without this note an implementer may waste time trying to add `sjf_config` to the C++ binding.

**S4: `api_stability` tests may flag `SchedulerConfig` as a changed API surface**

`AGENTS.md` warns: "Protected APIs exist — changes to LLM API signatures will fail `tests/unittest/api_stability` tests." Adding `sjf_config` to `SchedulerConfig` extends the public API. The plan does not include a step to update the API stability snapshot. This should be a named task (e.g., "Task 4.2: Update API stability snapshot") so it is not forgotten.

---

### Nice-to-Have

**N1: `create_waiting_queue` factory signature change should be backward-compatible**

The plan adds `sjf_config` as a new parameter to `create_waiting_queue`. Since `sjf_config` would default to `None`, the existing call sites that pass only `policy` will continue to work. The plan should make this explicit to confirm backward compatibility is maintained.

**N2: Verification plan concurrency levels could be more precise**

The verification plan lists concurrency levels `8, 32, 64, 128`. For the MLPerf DeepSeek-R1 dataset, the typical QPS where SJF shows benefit is at moderate-to-high queue depth. The plan does not specify whether "concurrency" means inflight request count or QPS target. Clarifying the unit (and whether aiperf's `--max-concurrency` or `--qps` flag is used) will make the A/B test reproducible.

**N3: No mention of `add_requests` delegation pattern**

The plan describes `add_requests` as "call `add_request` for each". This is consistent with the FCFS pattern and is correct, but it should note that `_sorted` will be set to `False` only once per individual `add_request` call — since setting it to `False` repeatedly is idempotent, this is not a bug, but the plan should confirm that `add_requests` does NOT call `_ensure_sorted()` internally, which would be wasteful mid-batch.

---

## Verdict

PASS_WITH_CONDITIONS

The plan is implementable as described. The task grouping, dependency ordering, and design alignment are correct. Three Must-Fix conditions must be addressed before implementation begins:

1. Clarify that `SJFWaitingQueue` is NOT a Pydantic model (plain Python class inheriting from `WaitingQueue` ABC), to prevent accidental misuse of `StrictBaseModel`.
2. Use Pydantic type aliases (`PositiveInt`, `PositiveFloat`, `NonNegativeFloat`) instead of inline `gt=`/`ge=` notation in `SjfConfig`, and verify that any numeric constraint syntax used is valid for Pydantic v2.
3. Commit to a concrete plumbing strategy for Task 3.2: either add `sjf_config` as a parameter to `PyExecutor.__init__`, or resolve it fully in `_util.py` before `PyExecutor` construction and pass it alongside `waiting_queue_policy` — document the choice explicitly.

The two Should-Fix items (prepend ordering semantics clarification and API stability snapshot task) should be resolved as plan amendments before implementation starts, not deferred.
