# Invalid-case patterns

Patterns from earlier TensorRT-LLM test-list cleanups. Use them as a search
guide. Each pattern still needs its own evidence in the current tree.

## A. Claim mismatch

| Pattern | What to look for | Evidence to cite | Typical action |
|---|---|---|---|
| Parameter not on the function | A `name=value` token in a list id that the function does not parametrize | The function signature and decorators | Drop the entry |
| Parameter accepted, then ignored | The value is never passed to `LLM(...)` or the config object the test builds | The body line that builds the object | Wire it through, or unlist |
| Parameter overwritten | The body resets the caller's value to a default (for example `num_experts`/`top_k` replaced by defaults) | The assignment that overwrites it | Stop overwriting; keep the selector only where it now means something |
| Flag dropped by a selector | A strategy selector returns an implementation that ignores the flag for this combination (for example `low_precision_combine=True` with `moe_tp_size != 1` selects AllGatherReduceScatter, which drops it) | The selector branch in source | Move the case to the combination that reaches the feature (for example EP), delete the unreachable twin, and fail fast on the unreachable combination |
| Value cannot take effect | A threshold or schedule sits above a cap the test also sets (for example Eagle3 `max_concurrency=100` and `draft_len_schedule` `{50,200,350}` above `max_batch_size=32`, so draft length never changes) | The cap and the threshold in the same test | Move the thresholds inside the cap and assert the per-batch value |
| Harness decides the claim | A tester or fixture picks the thing the name says the test configures (for example a worker tester that picks the ctx or gen server itself, with `conditional_disagg_config` never set) | The config the test loads and the tester code | Unlist, keep the body with a NOTE, point to the test that covers the real path |
| Claim only in a comment | A list or code comment names a behavior no assertion checks | The comment and the missing assertion | Add the assertion, or unlist |
| Same leaf path as another test | A new case, differing only by name or an inactive parameter, reaches the same code and checks the same result | Both bodies and the shared leaf | Remove the newer entry; name the covering test |
| Parent entry already selects it | A function entry beside a file or directory entry that selects the same id | The parent entry, the stage markers, the waives | Remove the function entry |

Quick check for any claim: find the line that shows the feature ran (a path
flag, a call count, sharding state, a nonzero feature delta). If there is none,
the case is a claim mismatch until proven otherwise.

## B. Dead entry

| Pattern | What to look for | Evidence to cite | Typical action |
|---|---|---|---|
| Hardware marker vs stage GPU | `skip_pre_blackwell`, `skip_pre_hopper`, `skip_no_hopper`, `skip_post_*` on a test listed under a GPU the marker excludes | The marker line, the stage GPU in `L0_Test.groovy` | Remove from that list if it runs elsewhere; move it if it does not |
| Device-count marker | `skip_less_device(8)` on a list with `system_gpu_count: 4`, or a list block for a GPU count no stage provides | The marker, the stage `gpu_count` | Move to a live stage or remove |
| Unscheduled block | A list block whose `stage` has no matching stage in `L0_Test.groovy` (for example an 8-GPU post-merge block with no 8-GPU post-merge stage) | `scripts/test_to_stage_mapping.py` output and the groovy stage map | Remove the block; keep the ids where other stages run them |
| CPU/GPU selection | A GPU-list test marked `cpu_only`, or a CPU-list test without it | The marker and the stage selection rule | Fix the marker or the list |
| Flag-gated test | Skips unless a `--run-*` flag, which CI never passes | The skip condition and the stage command | Remove from lists, or make CI pass the flag |
| Environment skip | `pytest.skip` for a missing checkpoint or dataset | The skip call in the fixture or body | `pytest.fail` with the missing path |
| Exit-0 skip | `GTEST_SKIP` or `exit 0` recorded as PASSED | The gtest XML handling in the wrapper | Make zero tests, a missing report, or any skip a failure |
| Source gate | A support check in source (SM, dtype, backend table) now returns false on the scheduled GPU | The check and the stage GPU | Remove from that list, or move |
| Renamed or re-id'd test | The list id no longer collects | `check_test_list.py --validate` | Update or remove the entry |
| Waive on a live entry | `full:<platform>/` or `full:sm<N>/` waive over a scheduled entry | `waives.txt` line | Report; remove only if stale |

## What does not count as invalid

- A skip that fires only on a platform the entry is not scheduled on.
- A waive with a tracking bug, for a test you intend to fix.
- A test whose id looks redundant but runs a different backend, lifecycle, or
  boundary. Compare executed inputs and assertions before calling it a duplicate.
- QA-only or local-only tests that are deliberately off the CI lists.
