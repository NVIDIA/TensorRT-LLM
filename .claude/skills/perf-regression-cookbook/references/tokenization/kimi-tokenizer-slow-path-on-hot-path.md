---
id: case-kimi-tokenizer-slow-path-on-hot-path
type: regression-case
family: execution-and-graph
module: tokenization
maturity: full
regression_class: [host-work-added]
signals: [itl-increase, throughput-drop, host-time-increase]
subsystems: [tokenizer, model-definition]
introduced_via: [prior-fix-side-effect]
phase: [any-phase]
patterns: [pattern-host-work-on-hot-path]
nvbugs: ["6248987"]
commits: ["e5b8094de2c7"]
success_prs: [14846]
failed_prs: []
---

# Pure-Python slow tokenizer swapped onto the Kimi K2.5 text-only hot path

> Part of the [Tokenization & detokenization regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6248987` · commit `e5b8094de2c7` · PR #14846 — Made
  the slow-tokenizer swap lazy and idempotent.
- **Symptom:** TPOT 9.8x higher and output token throughput down ~92% on the
  text-only `k25_thinking_fp4` perf path (ISL=8K, concurrency=2);
  `_fetch_new_requests` / `broadcast_requests` inflated ~34%. GPU kernel
  inventory and runtimes unchanged — pure host overhead (numbers from the PR
  description).
- **Root cause:** `KimiK25InputProcessor.__init__` unconditionally called
  `_ensure_k25_slow_tokenizer()`, replacing the fast Rust tokenizer with the
  pure-Python `TikTokenTokenizer` for every request — including pure-text
  prompts that never contain the K2.5 special tokens the swap exists to
  protect. Every request was then tokenized on the orchestrator's GIL
  (~100 ms per step at 8K-token prompts, per the fix diff's docstring).
- **How introduced:** PR #14392 (commit `546a5b09`), the correctness fix for
  NVBug 6182617: transformers 5.5.x's fast tokenizer BPE-splits K2.5 special
  tokens like `<|media_pad|>`, so the fix forced the slow tokenizer — but
  applied it to all inputs instead of only the ones that need it.
- **Fix mechanism:** Make the swap lazy and idempotent. `__init__` only sets
  `_slow_tokenizer_active = False`; `_ensure_k25_slow_tokenizer()`
  early-returns once active and is invoked only when the input actually needs
  canonical special-token mapping — multimodal data present, the disagg
  `get_prompt_token_ids` path, or a prompt containing a marker from
  `_K25_SPECIAL_TOKEN_MARKERS` (via `_input_needs_slow_tokenizer`). Pure-text
  prompts keep the fast Rust tokenizer; the NVBug 6182617 accuracy path still
  gets the slow `tokens_trie`. **Superseded at HEAD:** PR #14741 (merged
  2026-06-09, commit `28845ddf99a3`; filed under NVBug 6227203, a
  functional bug (a crash), so that id is deliberately not in `nvbugs:`)
  deleted the whole shim — #14392's swap *and* #14846's lazy machinery —
  because PR #14456 (transformers 5.5.4) restored correct `AutoTokenizer`
  routing for K2.5, making the forced load byte-identical to the default. So
  the perf property now holds structurally: no Python-side swap ever fires.
- **Detection signal:** ITL/TPOT collapses while the GPU kernel profile is
  unchanged; host spans around request intake (`_fetch_new_requests`,
  `broadcast_requests`) grow. Check whether the slow tokenizer was activated
  on a text-only run:
  `grep "swapping in slow TikTokenTokenizer" <serve.log>` (post-#14846 log
  line; pre-fix it reads "forcing slow TikTokenTokenizer"), and inspect
  `tensorrt_llm/_torch/models/modeling_kimi_k25.py` for unconditional
  `_ensure_k25_slow_tokenizer()` calls in `__init__`. Both greps go silent
  after #14741 removed the shim, so on a recent tree they are a check on
  whether the shim was reintroduced, not on whether it fired.
- **Prevention/guard:** The fix gates the swap behind
  `_input_needs_slow_tokenizer` and an idempotence flag, and documents both
  bugs at the call site. Gap: no perf test asserts the fast tokenizer stays
  active on the text-only path — the regression was caught by the
  `k25_thinking_fp4` perf run, after landing.
- **Generalizes to:** pattern-host-work-on-hot-path — a correctness fix for a
  rare input class imposes its slow path on all inputs. Recurs whenever a
  multimodal/special-token handler is installed unconditionally in an input
  processor `__init__`, when a Python fallback for a native (Rust/C++)
  component is enabled globally instead of per-request, and when per-request
  preprocessing (tokenize/encode/validate) runs on the orchestrator thread
  under the GIL instead of being deferred or scoped to inputs that need it.
