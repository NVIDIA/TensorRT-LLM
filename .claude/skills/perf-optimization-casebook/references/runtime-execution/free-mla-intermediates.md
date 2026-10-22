---
id: case-free-mla-intermediates
type: case
family: runtime-execution
maturity: full
bottleneck: [memory]
signals: [memory-capacity-bound, high-concurrency, long-context]
architectures: [any-sm]
model_scope: [model-agnostic, mla, deepseek-v3r1]
phase: [any-phase]
patterns: [pattern-reuse-computed-state]
accuracy_risk: lossless
apply_via_kind: [code-change]
knobs: []
specialists: [perf-host-optimization, perf-analysis, perf-sweep-workflow]
commits: ['fbcf954d9c3a']
measured: []
---

# Free large MLA intermediate tensors right after use to cut peak activation memory

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `fbcf954d9c3a` [MLA] Deallocate tensors after use (#3286).
- **Applies when:** memory-bound — peak GPU memory limits max batch size or KV-cache token budget (OOM at higher concurrency / longer sequences), on an MLA path that materializes several large transient activations per forward (down-proj outputs `q`/`compressed_kv`/`k_pe`, `fused_q`, context/gen attention outputs, FP8 scale tensors).
- **Mechanism:** MLA forward holds several big intermediates simultaneously; the commit sets each to `None` immediately after its last consumer (`q = compressed_kv = k_pe = None` after `q_b_proj`; `fused_q = None`; `attn_output_context = attn_output_gen = q_nope_scales = None`; `attn_out_latent_scales = None`), dropping the last Python reference so the caching allocator can reuse that memory within the same forward → lower peak activation memory → headroom the KV estimator converts into more KV tokens or a larger max batch.
- **Generalizes to:** "drop references to large intermediates at their last use to lower peak memory and trade it for batch/KV capacity"; carries to any attention/MLP/MoE forward with chained large temporaries, and FP8 paths where extra scale tensors add up; adapt by finding the true last-use point of each big tensor and nulling it (or reusing buffers) without breaking autograd (inference forward has no backward dependence).
- **Apply via:** code-level change in `_torch/attention/mla.py` (MLA forward) — set spent intermediates to `None`. Not a config knob; payoff realized through the existing KV-cache auto-estimation (more `max_num_tokens`/KV tokens at the same memory). Delegate memory profiling to **perf-host-optimization** / **perf-analysis**; confirm the larger feasible batch with **perf-sweep-workflow**.
- **Expected effect:** lower peak GPU memory in MLA forward → enables larger max batch or more KV-cache tokens at fixed memory (indirect throughput gain at higher concurrency); measured Δ (peak mem, achievable batch/KV) to be recorded from run.
- **Accuracy risk:** lossless (frees only no-longer-referenced tensors; computation and outputs unchanged).
- **Verify:** peak memory (`torch.cuda.max_memory_allocated`) for the MLA forward before/after; confirm a higher max batch / KV-token count is feasible without OOM; outputs identical.
- **Rollback:** remove the `= None` deallocations. Trigger: only if a freed tensor is later needed (surfaces as an error, not silent wrongness) — none expected for the inference forward.
- **Prior art:** PR #3286. Files: `_torch/attention/mla.py` (MLA `forward`: nulling of `q`/`compressed_kv`/`k_pe`, `fused_q`, attention outputs, scale tensors). Detection: **perf-analysis** (memory).
