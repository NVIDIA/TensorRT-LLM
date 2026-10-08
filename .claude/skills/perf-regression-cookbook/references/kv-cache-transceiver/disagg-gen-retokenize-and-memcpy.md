---
id: case-disagg-gen-retokenize-and-memcpy
type: regression-case
family: execution-and-graph
module: kv-cache-transceiver
maturity: full
regression_class: [host-work-added]
signals: [ttft-increase, host-time-increase]
subsystems: [serve-endpoint, kv-cache]
introduced_via: [pre-existing-gap]
phase: [prefill]
patterns: [pattern-host-work-on-hot-path]
nvbugs: ["6196391"]
commits: ["54259ed5d232"]
success_prs: [14499, 14506, 14719]
failed_prs: []
---

# Disagg gen server retokenizes prompts and pays O(N^2) memcpy in KV reuse

> Part of the [KV-cache transceiver regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6196391`. **Three PRs, two branches — cite #14499
  first.** The fixes were authored against the `feat/bench_x` feature branch
  and only reached `main` as a carryover, so the PR you want depends on the
  question:
  - PR #14499 · commit `8fff9c50fe8f` · base `feat/bench_x`, merged
    2026-05-26 — `[https://nvbugs/6196391][fix] Eliminate O(N^2) memcpy in
    findBlocksInReuseTreeByBlockKey` (`kvCacheManager.cpp`, the only file).
    **This is the nvbug's fix PR** — the one carrying the actual C++ fix, and
    the one to cite when discussing this defect. Its commit `8fff9c50fe8f` is
    on `feat/bench_x` only and appears nowhere in `main`'s history.
  - PR #14506 · commit `91142da9b45c` · base `feat/bench_x`, merged
    2026-05-27 — `[None][fix] avoid duplicate harmony tokenization & populate
    perf_metrics` (`openai_server.py`). Titled `[None]`, so **no bug id links
    it at all**; it is discoverable only through #14719's description.
  - PR #14719 · commit `54259ed5d232` · base `main` — "Carryover disagg TTFT
    improvements", which is what actually put both of the above on `main`.
  Read as: a fix on a feature branch is not on `main`, and the branch pair
  splits the provenance in two. Here `gh pr list --search 6196391` returns
  #14719 and #14499 but **not** #14506, whose `[None]` title drops the id
  entirely; and the hash a bug→PR lookup hands you (#14499's
  `8fff9c50fe8f`) is not in `main`, while the hash in `main` (#14719's
  `54259ed5d232`) has a different author and a wider diff boundary than the
  fix. All three numbers are needed to reconstruct the change.
- **Symptom:** high TTFT in disaggregated serving on the Harmony GPT-OSS
  chat path; two independent host-side costs sat before the first token
  on the generation server.
- **Root cause:** two defects. (1) In `openai_server.py`, the gen server
  always re-ran `harmony_adapter.openai_to_harmony_tokens()` on the request
  messages even when the ctx server had already forwarded
  `request.prompt_token_ids` — full retokenization repeated per request.
  (2) In `kvCacheManager.cpp`, `WindowBlockManager::findBlocksInReuseTreeByBlockKeys`
  built its per-block key list with `blockKeys.push_back(blockKey)`, copying
  the *entire* prompt's `uniqueTokens` vector once per block before
  overwriting it — an O(N^2) memcpy in prompt length on the disagg reuse
  path.
- **How introduced:** unknown — not stated in the PR; both are
  pre-existing gaps in the disagg codepath (the PR is a carryover of
  improvements, not a revert of a named regressing change).
- **Fix mechanism:** (1) reuse `request.prompt_token_ids` when present and
  only tokenize when it is `None`; (2) stop copying the whole-prompt key per
  block — the two branches do this in different shapes, so match the shape to
  the branch you are reading: on `main` (#14719) `blockKeys.reserve(...)` plus
  an `emplace_back` of a `BlockKey` built directly from that block's token
  slice; on `feat/bench_x` (#14499) the `std::vector<BlockKey>` is dropped
  altogether in favour of one reusable `BlockKey lookupKey` whose
  `uniqueTokens` is rebound each iteration. Either way a key never copies more
  than its own block's tokens. Also stamps first-token time in the
  streaming generator and enables `return_perf_metrics` when
  `TRTLLM_KVCACHE_TIME_OUTPUT_PATH` is set (the `/perf_metrics` deque
  stays empty on this path otherwise).
- **Detection signal:** gen-server host time before the first forward step
  grows with prompt length (tokenizer + block-key construction spans in a
  py-spy/nsys host profile); verify the guard exists with
  `grep -n "prompt_token_ids is not None" tensorrt_llm/serve/openai_server.py`
  and compare ctx-vs-gen TTFT via the server's `/perf_metrics` endpoint
  (requires `TRTLLM_KVCACHE_TIME_OUTPUT_PATH` to be set on this path).
- **Prevention/guard:** the PR adds
  `test_prompt_token_ids_skips_retokenization` in
  `tests/unittest/llmapi/apps/_test_openai_chat_harmony.py` plus a
  `_test_openai_chat_harmony_perf_metrics.py` suite; no guard exists for
  the C++ copy cost — accidental whole-struct copies inside per-block
  loops still need review attention.
- **Generalizes to:** `pattern-host-work-on-hot-path`; carries to any
  disagg pipeline stage that redoes work already done upstream
  (detokenization, chat-template rendering), per-request Python work ahead
  of the first token in serve endpoints, and copying a whole
  container-holding struct inside a per-block/per-chunk loop, turning a
  linear pass quadratic.
