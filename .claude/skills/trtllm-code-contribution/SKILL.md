---
name: trtllm-code-contribution
tags: [tensorrt-llm, workflow, development]
description: >
  Best practices for contributing code to TensorRT-LLM. Covers the official
  contribution process (issue tracking, fork workflow, DCO signing), coding
  guidelines, implementation workflow, common mistakes, testing strategy, commit
  hygiene, and review readiness. Incorporates rules from CONTRIBUTING.md and
  CODING_GUIDELINES.md plus lessons distilled from real PR retrospectives.
  Use when implementing new features, optimizations, or bug fixes in the
  TensorRT-LLM codebase.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# TensorRT-LLM Code Contribution Best Practices

## Contribution Process

### 1. Developer Workflow

1. **Commit** the changes. Never commit using NVIDIA internal email (`<user>@nvidia.com`)!
2. **Push** changes to a branch on the personal fork:
   ```bash
   git push -u <user> <local-branch>:<remote-branch>
   ```
3. **Create a PR** from the fork branch into upstream (typically targeting `main`).

### 2. Coding Guidelines

TRT-LLM coding style is defined in `CODING_GUIDELINES.md`. Key highlights:

**C++:** Allman brace style, 4-space indent, 120 char line limit, camelCase for variables/methods, PascalCase for types, `m` prefix for member variables, `k` prefix for constants, Doxygen for API docs, smart pointers over raw, `static_cast` over `reinterpret_cast`, no C-style casts.

**Python:** snake_case for files/functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants, 4-space indent, Google-style docstrings, narrow `except` clauses, Pydantic `StrictBaseModel` for user-facing config classes (no custom `__init__`).

### 3. Pre-commit Setup

```bash
pip install pre-commit
pre-commit install
```

Pre-commit runs automatically on every `git commit`. Hooks include: isort, yapf, autoflake, clang-format, cmake-format, codespell, ruff, ruff-format, mdformat, and others. If hooks modify files, stage and commit them again.

### 4. DCO Sign-off (Required)

All commits must be signed off to certify the contribution under the [Developer Certificate of Origin](https://developercertificate.org/):

```bash
git commit -s -m "Add cool feature."
```

This appends `Signed-off-by: Your Name <your@email.com>` to the commit message. PRs containing unsigned commits will not be accepted.

**IMPORTANT**: Never sign off commits using NVIDIA internal email (`<user>@nvidia.com`)!

## Pre-Implementation Checklist

Before writing any code, complete these steps:

### 1. Survey Existing Infrastructure

**Search before building.** TRT-LLM is a large codebase with many reusable components. Before implementing something from scratch, search for existing utilities:

```
# Before writing a new attention computation
grep -r "TrtllmAttention\|create_attention\|scaled_dot_product" tensorrt_llm/_torch/

# Before writing a new compiled helper
grep -r "maybe_compile\|maybe_compiled_" tensorrt_llm/_torch/utils.py

# Before writing a custom RoPE
grep -r "RotaryEmbedding\|rotary_emb\|rope" tensorrt_llm/_torch/modules/

# Before writing a new cache management pattern
grep -r "mla_rope_append_paged_kv\|append_paged_kv" tensorrt_llm/_torch/
```

**Trace existing forward methods.** Before writing a new `forward_*` method, read all existing forward methods in the class and understand what each one does. Often an existing method already implements the computation you need, and you just need to set up the right state (e.g., create an attribute, adjust a guard) to dispatch to it.

```
# Find all forward methods in a class
grep -n "def forward" tensorrt_llm/_torch/modules/attention.py
# Then READ each one to understand what it does
```

**Lesson learned:** On the short-seq MHA branch (30 commits, ~250 lines written then deleted), the attention computation went through **4 rewrites**: per-sequence SDPA loop → batched SDPA with pad_sequence → custom TrtllmAttention backend → deletion in favor of the *already-existing* `forward_context_default()`. The final approach was +10 lines: a guard check + dispatch to an existing method. Similarly, `maybe_compiled_cat` was discovered only after a standalone `@maybe_compile` wrapper was written and then removed.

**Anti-pattern: Parallel reimplementation.** Before writing a new `forward_*` method, trace what existing forward methods do. The new method may already be implemented. In the MLA case, `forward_context_short_mha` reimplemented `forward_context_default` nearly line-for-line before being deleted.

### 2. Check Parallelism Dimensions

When adding a new code path, verify correctness under ALL parallelism modes:

| Dimension | Guard | Why |
|-----------|-------|-----|
| Tensor Parallelism (TP) | `mapping.tp_size` | Head counts are sharded |
| Pipeline Parallelism (PP) | `mapping.pp_size` | Layers may be on different ranks |
| Context Parallelism (CP) | `mapping.cp_size` | Sequence is split across ranks — tokens are not all local |
| Expert Parallelism (EP) | `mapping.ep_size` | MoE experts distributed |

**Lesson learned:** The short-seq MHA path assumed all tokens were local, which breaks under Context Parallelism. The `cp_size == 1` guard was added as a fix in a later commit instead of being part of the initial design.

### 3. Think About Threshold/Guard Semantics

When gating a code path with a threshold:
- **What does the threshold measure?** (per-sequence metric? total batch metric?)
- **What does the cost of the path scale with?** (per-sequence? total tokens? quadratic in something?)
- **Do these match?** If cost scales with total tokens, the threshold should check total tokens, not per-sequence max.

**Lesson learned:** The initial implementation checked `max_ctx_seq_len` (longest single sequence) against the threshold, but the cost of the short-seq path scales with total packed tokens. A batch of 100 short sequences could incorrectly trigger the path.

### 4. Check RoPE State

When adding attention code paths:
- Is `apply_rotary_emb` True (caller handles RoPE) or False (rope_fusion, backend handles RoPE)?
- Does your path apply RoPE? Will that cause double-application?
- Do you need to handle both RoPE states or can you gate to one?

### 5. Trace Method Limitations

**Understand what a method does NOT handle.** When reusing an existing method, fully trace the dispatch chain above it. A method may be correct for one scenario but miss edge cases handled by a higher-level dispatcher.

**Example:** `forward_context_default()` handles fresh prefill with no cached KV tokens. But when there are cached KV tokens (chunked context), it silently ignores them — causing a correctness bug. The fix was to call `forward_context()` instead, which dispatches to:
- `forward_context_with_chunked_prefill` (SM100+, chunked context)
- `forward_context_with_cached_kv` (SM90 fallback, or cached context)
- `forward_context_default` (fresh prefill, no cached tokens)

**Checklist for reusing a method:**
1. What does this method handle?
2. What does it NOT handle? (cached tokens? chunked prefill? specific hardware?)
3. Is there a higher-level dispatcher that routes to this method for the right cases?
4. Should I call the dispatcher instead of the method directly?

### 6. Check Hardware-Specific Behavior

The same algorithm can have different numerical properties across SM versions. FMHA kernels may use different internal implementations (e.g., online softmax merge on SM90 vs single-pass on SM100+) that produce different accuracy characteristics.

**Lesson learned:** The SM90 (Hopper) FMHA kernel's online softmax merge for chunked prefill diverged from the single-pass reference by ~0.45 max diff — unacceptable for a correctness-critical path. The fix was to gate chunked prefill behind `get_sm_version() >= 100` (Blackwell+) and fall back to `forward_context_with_cached_kv` on SM90.

**When to check:**
- Any new attention code path that uses fused kernels
- Any path that changes how attention is split/chunked (chunked prefill, context parallelism)
- When accuracy tolerances are tight and the path crosses hardware generations

## Implementation Workflow

### Use the Right Abstraction Level

Choose backends from this priority list:

1. **Existing forward method** (e.g., `forward_context_default`) — may already implement what you need; just set up state and dispatch
2. **Existing fused backend** (e.g., `TrtllmAttention`, `FlashInferAttention`) — handles packed sequences, variable lengths, KV cache natively
3. **PyTorch fused ops** (e.g., `F.scaled_dot_product_attention`) — good for prototyping but requires manual batching/padding
4. **Manual implementation** — last resort, only when no existing backend fits

### Use the Right Dispatch Abstraction Level

When dispatching to an existing method, use the **highest-level dispatch point** that provides the right abstraction. Don't bypass dispatch layers — you'll miss edge cases.

| Abstraction Level | Example | Handles |
|-------------------|---------|---------|
| Top-level dispatcher | `forward_context()` | Chunked prefill, cached KV, fresh prefill, SM-version gating |
| Specific handler | `forward_context_default()` | Fresh prefill only |
| Backend directly | `self.mha.forward(...)` | Nothing beyond raw attention |

**Lesson learned:** The initial short-seq MHA implementation called `forward_context_default()` directly. This worked for fresh prefill but silently dropped cached KV tokens during chunked context. Switching to `forward_context()` (which dispatches to `forward_context_with_cached_kv` or `forward_context_with_chunked_prefill` as appropriate) fixed the bug with a 1-line change.

### Prefer Reusing Existing Attributes Over Creating New Ones

When adding a new code path, check if an existing attribute can serve double duty:

```python
# BAD: parallel attribute alongside existing one
self._short_seq_mha = create_attention(...)  # separate from self.mha
# Then need special handling everywhere self.mha is referenced

# GOOD: reuse existing attribute with conditional initialization
if should_use_dense_mha:
    self.mha = create_attention(...)  # replaces None for DSA models
# Existing code paths that check self.mha just work
```

**Lesson learned:** The short-seq MHA initially used `self._short_seq_mha` as a separate attribute to "preserve the assertion that `self.mha is None`". Later, it was realized the assertion itself should change (`self.mqa is not None`) and `self.mha` could be reused.

### Run Pre-Commit Before Every Commit

**Always run `pre-commit run --all-files` before committing.** The short-seq MHA branch had a 377-line formatting-only commit (commit 15/19) that existed solely because pre-commit wasn't run on earlier commits. This is wasted reviewer attention and pollutes `git blame`.

```bash
# Before every commit:
pre-commit run --all-files
git add -u  # stage any auto-formatted files
git commit -s -m "..."
```

### Apply torch.compile Judiciously

| Pattern | Use `@maybe_compile`? | Why |
|---------|----------------------|-----|
| Fused math (RoPE rotation, GELU) | Yes | Fuses multiple element-wise ops into one kernel |
| `torch.cat` of computed tensors | Use `maybe_compiled_cat` | Already exists as a utility |
| Pure metadata ops (split, view, expand, reshape) | No | These are zero-cost; compile adds overhead |
| Mixed metadata + compute | Extract the compute part | Compile only what benefits from fusion |

### Extract Shared Logic Immediately

When a condition appears in more than one place, extract it into a helper method **in the same commit**. Don't wait for a later refactoring commit.

```python
# BAD: same 5-condition check in two places
if (threshold > 0 and not apply_rotary and cp_size == 1 and ...):  # site 1
    ...
if (threshold > 0 and not apply_rotary and cp_size == 1 and ...):  # site 2
    ...

# GOOD: extract immediately
def _should_use_short_mha(self, ...):
    return (threshold > 0 and not apply_rotary and cp_size == 1 and ...)
```

### Feature Flags for Complex Optimizations

Complex optimizations with multiple guards, edge cases, and hardware-specific behavior should ship **disabled by default**. Let users opt-in via environment variable after testing.

```python
# Pattern: disabled by default (threshold=0), opt-in via env var
_threshold_str = os.environ.get('TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD', '0')
self.short_seq_mha_threshold = int(_threshold_str)
```

**Lesson learned:** The short-seq MHA optimization was initially enabled by default (threshold=10240) at commit 8 but had 18 more correctness fixes over the next 22 commits before being disabled by default at commit 26. Complex optimizations accumulate edge cases (chunked context, SM90 accuracy, threshold semantics) that may not be discovered until broad testing.

**When to disable by default:**
- The optimization has 3+ guard conditions
- It touches attention/correctness-critical paths
- It has hardware-specific behavior (different SM versions)
- It hasn't been tested in full CI across all configurations

### Update All References When Changing Semantics

When changing what a variable/threshold means, grep for ALL references:
```bash
# After changing threshold from max_seq_len to total_packed_tokens:
grep -rn "max_seq_len\|max_ctx_seq_len\|short.*seq.*threshold" tests/ tensorrt_llm/
```
Update comments, docstrings, test descriptions, and variable names in the **same commit**.

## Testing Strategy

### When to Write Tests

| Phase | What to test | Why |
|-------|-------------|-----|
| After implementation stabilizes | Full correctness suite | Avoid rewriting tests with each iteration |
| During prototyping | Minimal smoke test only | Validates basic plumbing without coupling to implementation details |
| After optimization changes | Add regression tests for the specific optimization | Catches if the optimization breaks something |

**Lesson learned:** Tests were written before the attention backend was settled, then required 5 separate fix/update commits as the implementation evolved through 4 rewrites. The 770-line test file needed immediate fixing (device placement, weight layout bugs) because it was never run before committing.

### Common Test Gotchas in TRT-LLM

1. **Non-Module children aren't moved by `.to(device)`**: If a module has attributes that aren't `nn.Module` subclasses (e.g., `DSATrtllmAttention.indexer`), `model.to(device)` won't move their parameters. Move them explicitly.

2. **Weight layout differs from HuggingFace**: Model loading transforms weights. Initialize test weights in the **loaded layout** (check `modeling_*.py` for load functions), not the HuggingFace checkpoint layout.

3. **Background threads from cache managers**: `DSACacheManager` and similar create `ThreadPoolExecutor` threads that outlive tests. Add `pytestmark = pytest.mark.threadleak(enabled=False)` at the module level.

4. **`named_parameters()` misses non-Module attributes**: When copying weights for A/B comparison tests, explicitly copy parameters from non-Module children (like indexer weights).

5. **Attention metadata construction**: Use the test fixtures/helpers already in the codebase (check `tests/unittest/_torch/attention/` for patterns) rather than building `AttentionMetadata` from scratch.

### Test Consolidation

After implementation stabilizes, aggressively prune tests to a minimal set where each parametrized case exercises a **distinct code path**.

**Pattern:**
1. During development, write comprehensive tests (many parametrized cases covering all combinations)
2. After implementation stabilizes, identify which code paths each test case exercises
3. Merge cases that exercise the same code path; remove redundant cases
4. Extract shared test helpers (`_make_inputs`, `_make_metadata`, `_run_forward`) to reduce duplication

**Lesson learned:** The short-seq MHA test file peaked at 1394 lines with 21 parametrized cases, then was consolidated to 665 lines with 10 cases covering the same 6 code paths. Three separate cleanup commits were needed because consolidation wasn't done in one pass. Do consolidation as a single deliberate pass.

### Test on Multiple Hardware Targets

When testing attention kernels or fused operations, verify on multiple SM versions. The same kernel can have different numerical properties across hardware generations.

- SM90 (Hopper): Online softmax merge in FMHA — can diverge from reference
- SM100+ (Blackwell): Single-pass FMHA — tighter numerical accuracy
- Use `get_sm_version()` guards to skip or adjust tests per hardware

## Commit Hygiene

### During Development

Commit freely — small, frequent commits help track progress and enable bisection.

### Before PR Submission

Squash fix-on-fix chains using interactive rebase:

```bash
# Fold fix commits into the commits they fix
git rebase -i $(git merge-base HEAD main)
```

Target commit structure for a PR:
1. **Core implementation** — the new feature with all guards and edge cases
2. **Additional optimizations** — one commit per distinct optimization
3. **Tests** — comprehensive test suite
4. **Refactoring** (optional) — cleanup that's separate from the feature

### Anti-patterns to Avoid

| Anti-pattern | What happens | Prevention |
|-------------|-------------|------------|
| Fix-on-fix chains (A → fix A → fix fix A) | Noisy history, hard to review | Squash before PR |
| Add-then-revert (add X → revert X) | Wasted reviewer attention | Survey existing utilities first |
| Modify shared utility then revert (edit rotary_embedding.py → revert) | Pollutes unrelated files | Check if existing code paths handle it |
| Create compiled helper then inline it (add @maybe_compile → remove) | Churn | Profile first; only compile proven bottlenecks |
| Semantic change + behavior change in one commit | Hard to bisect regressions | Separate bug fixes from feature changes |
| Stale comment fix as separate commit | Shows the comment wasn't updated with the code change | Update comments in the same commit as the code |

### PR Title Format (Conventional Commits)

PR titles follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
type: description
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`, `None`

For breaking API changes, use `BREAKING CHANGE:` as the type to alert reviewers.

For NVIDIA developers, prefix with JIRA number or NVBUG ID:
```
[TRTLLM-5516] perf: description
[nvbug/5334370] fix: description
```

Examples:
- `feat: Add support for starcoder-v2 FP8 base + FP16/BF16 LoRA`
- `BREAKING CHANGE: Set default max batch size to 2048`
- `chore: Remove version from plugins .so`
- `None: Stringized enums for better error msgs`
- `fix https://github.com/NVIDIA/TensorRT-LLM/issues/700: a Memory leak issue in C++ runtime`
- `[TRTLLM-5516] perf: Replicate dummy request for cuda graph padding`

### PR Description

Address these points in the PR description:

1. **Background/motivation**: Why is the change necessary?
2. **Summary**: Summarize the changes in one paragraph.
3. **Size justification**: If the PR is large, explain why it cannot be broken into multiple PRs.
4. **Impact assessment**: Potential performance or functional impacts. Flag risks for reviewers.
5. **Related PRs**: Link to any related PRs.

### PR Conciseness

- Avoid committing commented-out code.
- Each PR should address a **single concern**. If there are several unrelated fixes, open separate PRs and indicate dependencies in the descriptions.

### API Stability Tests

Some APIs are protected by the [API stability testsuite](tests/api_stability). If your PR breaks a protected API, the stability tests will fail with `API stability validation failed`. In this case, request review from the API code owners.

## Quantified Impact of Common Mistakes

From the short-seq MHA branch (30 commits → net 2 files changed):

| Mistake | Commits wasted | Lines written & deleted | Root cause |
|---------|---------------|------------------------|------------|
| Reimplementing existing forward method | 4 (commits 1,5,6,17) | ~150 lines | Didn't read `forward_context_default` |
| Custom RoPE handling | 5 (commits 1,13,16,17,18) | ~100 lines | Didn't trace how fused kernel handles RoPE |
| Tests before stable implementation | 5 (commits 3,4,8,11,15) | ~200 lines of churn | Tests coupled to implementation details |
| Compiled helpers created then removed | 4 (commits 10,12,13,18) | ~60 lines | Premature optimization without profiling |
| Style-only commit | 1 (commit 15) | 377 lines reformatted | Pre-commit not run on earlier commits |
| Stale comment fixes | 2 (commits 11,18) | ~15 lines | Comments not updated with code changes |
| Calling method directly instead of dispatcher | 3 (commits 21,23,30) | ~20 lines | Didn't trace `forward_context()` dispatch chain |
| Not testing on SM90 | 1 (commit 30) | ~10 lines | Assumed uniform numerical behavior across SM versions |
| Enabled by default too early | 2 (commits 8,26) | ~5 lines | Shipped threshold=10240 before edge cases were found |
| Threshold semantics drift in chunked context | 1 (commit 28) | ~10 lines | `num_ctx_tokens` doesn't account for cached tokens |
| Redundant test parametrizations | 3 (commits 24,25,27) | ~730 lines pruned | Tests written incrementally without path-coverage analysis |

**Total waste**: ~24 of 30 commits were fixes/reverts/cleanups of earlier work on the same branch. The final net change is ~200 lines in attention.py and ~665 lines in tests — achievable in ~4-5 clean commits.

## Review Readiness Checklist

Before marking a PR ready for review:

- [ ] GitHub issue created and approved
- [ ] All parallelism modes checked (TP, PP, CP, EP)
- [ ] RoPE state handled correctly (no double-application)
- [ ] Threshold/guard semantics match the cost model
- [ ] Existing infrastructure surveyed and used where possible
- [ ] Shared logic extracted (no duplicated conditions)
- [ ] Comments/docstrings updated with any semantic changes
- [ ] Tests pass and cover key scenarios (including API stability tests if applicable)
- [ ] Commits squashed (no fix-on-fix chains)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] DCO sign-off on all commits (`git commit -s`)
- [ ] Dispatch calls use the right abstraction level (dispatcher, not specific handler)
- [ ] Method limitations understood (what the reused method does NOT handle)
- [ ] Hardware-specific behavior tested (SM90, SM100+) or gated appropriately
- [ ] Complex optimizations disabled by default with env var opt-in
- [ ] Test cases exercise distinct code paths (no redundant parametrizations)
- [ ] PR title follows Conventional Commits format
- [ ] PR description addresses background, summary, and impact
