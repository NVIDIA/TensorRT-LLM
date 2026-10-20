---
id: case-dsa-mtp-stale-token-to-request-map
type: regression-case
family: execution-and-graph
module: spec-decode
maturity: full
regression_class: [stale-cached-metadata]
signals: [acceptance-length-drop, throughput-drop]
subsystems: [spec-decode, attention-kernel]
introduced_via: [pre-existing-gap]
phase: [decode]
patterns: [pattern-stale-metadata-across-layout-change]
nvbugs: ["6513132", "6513093"]
commits: ["5a47974235bf"]
success_prs: [16925]
failed_prs: []
---

# DSA sparse attention reused a token→request map built for the verify layout, so every MTP draft token was attributed to request 0

> Part of the [Speculative decoding regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbugs `6513132`, `6513093` · commit `5a47974235bf` ·
  PR #16925 — rebuilds `req_idx_per_token` for the MTP draft layout.
- **Symptom:** acceptance length far below what the draft depth should deliver,
  and the deficit **grows with batch size** — the tell that separates this from an
  ordinary quality gap. Measured acceptance length (AL), greedy:
  GLM-5.2 NVFP4, 1×GB200, tp4/ep4 + ADP, MTP k=7 — conc 8 / batch 2:
  **3.500 → 3.943 (+12.6 %)**; conc 64 / batch 16: **2.789 → 3.909 (+40.2 %)**.
  DeepSeek-V3.2 NVFP4 — c8/b2 **2.5397 → 2.6379 (+3.9 %)**, c64/b16
  **2.3994 → 2.6587 (+10.8 %)**, c128/b32 **2.3795 → 2.6769 (+12.5 %)**; the PR
  summarises the shape as "Slope c8 → c128: **−6.3 % → +1.5 %** (flat)" — i.e. the
  fix removes a batch-size-dependent decay, which is the real signature. Against a
  same-session vLLM reference the residual gap closes with depth: k=1
  1.8429→1.8504, k=4 3.0132→3.2547, k=7 3.3178→3.7032, gap **0.471 → 0.086**.
  End-to-end, GPQA-Diamond (198 questions, GB300, 1p1d disagg, c128, k=7):
  symbolic_correct 85.354 % → 85.859 %, no_answer 5.556 % → 3.030 %, generation
  wall-clock **5,117 s → 3,322 s (−35 %)**.
- **Root cause:** DSA builds `req_idx_per_token` — the map from each token in the
  flattened batch back to its request — **once, in `prepare()`**, from the *verify*
  layout of `max_draft_len + 1` tokens per request. The MTP draft loop then runs
  with **one token per request** and reuses that map unchanged. With a stride of
  `max_draft_len + 1 > 1`, the first `batch_size` entries of the old map all point
  at request 0, so a 3-request draft step is indexed `[0, 0, 0]` instead of
  `[0, 1, 2]`: every request's sparse-attention indices are computed against
  request 0's sequence. The draft tokens are therefore wrong-but-plausible, the
  verify step rejects them, and the loss shows up as *acceptance length* rather
  than as an error — which is why PR #16925 lists it as two problems (an AL
  regression at large batch, and an AL gap vs vLLM).
  The batch-size dependence follows directly: with batch 1 the stale map is
  accidentally correct.
- **How introduced:** `pre-existing-gap`. No culprit commit; the map was built for
  the layout that existed when it was written, and the MTP draft loop's different
  layout was never reflected. The general defect is **cached metadata that encodes a
  layout, reused across a layout change**.
- **Fix mechanism:** derive the map from the actual sequence lengths, every time.
  A new capture-safe helper
  `build_req_idx_per_token(seq_lens, num_tokens)` computes
  `torch.searchsorted(torch.cumsum(seq_lens, dtype=torch.int32), torch.arange(num_tokens), right=True)`
  — no data-dependent shapes, no host sync, so it is legal inside a CUDA-graph
  capture — and it is rebuilt **unconditionally** in `on_update_kv_lens()`, the hook
  that already fires whenever the layout changes. `sparse/dsa.py` +21/−2,
  `deepseek_v4.py` +7/−7. Rebuilding unconditionally rather than "when the layout
  differs" is the correct call: the predicate for *when* it differs is the thing
  that was wrong.
- **Detection signal:** `grep -n "req_idx_per_token" tensorrt_llm/_torch/modules/sparse/dsa.py`
  — pre-fix the only assignment is inside `prepare()`; post-fix
  `build_req_idx_per_token` is also called from `on_update_kv_lens`. Behaviourally,
  the discriminator is a **batch-size sweep of acceptance length at fixed draft
  depth**: AL that falls as batch grows (c8 → c128) with unchanged accuracy is this
  class. AL at batch 1 is normal, so a single-request smoke test proves nothing.
- **Prevention/guard:** a real unit test was added — `test_req_idx_per_token.py`
  (+100) — pinning the helper against the expected mapping for uneven sequence
  lengths. Affected models: `DeepseekV32ForCausalLM` and `GlmMoeDsaForCausalLM`,
  and per the PR the bug bites "yes, when `max_draft_len ≥ 2`". Rule to carry:
  any tensor that encodes *how tokens are packed* must be rebuilt at every
  packing change, and the AL metric is the only place a wrong index map surfaces —
  correctness tests pass, because rejected drafts are still correct output.
- **Generalizes to:** `pattern-stale-metadata-across-layout-change`; carries to
  every cached index/offset/cumsum buffer shared between a verify pass and a draft
  pass (spec-decode of any flavour, chunked prefill vs decode, ADP padding), and to
  the broader diagnostic: **a spec-decode acceptance deficit that scales with batch
  size is an indexing bug, not a model-quality issue.**
