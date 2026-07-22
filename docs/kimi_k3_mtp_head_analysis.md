# Kimi K3 MTP-head input convention & SA/rejection-sampling constraint

Status: pre-weights analysis (2026-07-21). Companion to
`kimi_k3_suffix_automaton_design.md` §"Pre-weights MTP prep" (items 2 and 5).
Everything below is readable today; no draft-head checkpoint exists
(`config.json: num_nextn_predict_layers = 0`).

## Item 2 — which hidden state feeds the MTP head?

### Evidence surveyed

**(a) HF reference in the checkpoint — does NOT settle it.**
`configuration_kimi_k3.py:49,103` declares `num_nextn_predict_layers`, but
`modeling_kimi.py` (and `modeling_kimi_k3.py`) contain **zero** MTP/nextn
code — no head module, no capture, no draft loop. The only relevant fact is
the lm-head input convention: the trunk output is
`norm(_apply_output_attn_res(hidden, block_residual))`
(`modeling_kimi.py:1259-1263`, `_apply_output_attn_res:1270-1277`) — i.e.
attn_res output mixing happens *before* the final RMSNorm, and the lm_head
consumes the post-norm result.

**(b) Moonshot `KDA_decode_mtp` kernel drop — layer-level only.**
The kernel is the KDA conv+recurrent decode with `M=2` spec tokens
(`README.md`: t_max modes `1+NUM_SPEC` / `2*NUM_SPEC+1`). It tells us the
MTP draft path *contains full KDA block(s)* — a DeepSeek-style deep draft
layer, not a thin linear head — but is silent on the embedding/hidden
mixing at the head's input (no `enorm`/`eh_proj` analog appears anywhere in
the drop). Correction to earlier notes: the drop IS self-contained for
validation — `run_cpu_vs_cute.py` uses the included pure-torch CPU
reference; only `reference.py`'s `main()` needs the internal Triton import
(`kimi/kda_recurrent_kernel_specification`).

**(c) In-tree precedents.**
DeepSeek-V3 MTP (`modeling_deepseekv3.py:1668-1758`):
`eh_proj(cat(enorm(embed(next_token)), hnorm(trunk_hidden)))` → one full
decoder block → `shared_head.norm` → shared lm_head (`:1804-1809`).
Critically, `DeepseekV3Model.forward` returns **without** applying a final
norm (`:1875-1887` — layers' fused-residual output returned directly), so
the MTP head consumes **pre-final-norm** trunk hidden and re-normalizes via
`hnorm`. Qwen3-Next is eagle-style recurrence over the same idea. K3's
sibling K2.5 literally *is* DeepSeek-V3 in-tree — `modeling_kimi_k25.py`
wraps `DeepseekV3ForCausalLM` as its LLM (`:75,1579`) — so Moonshot's
prior MTP shipped with exactly the DeepSeek convention.

**(d) K3 candidate tap points** (TRT-LLM `modeling_kimi_linear.py`; HF
equivalents in parens):

| # | Tap | Evidence for | Evidence against |
|---|---|---|---|
| 1 | post-layers, **pre-attn_res-mix** (`prefix_sum` before `_apply_attn_res`) | none | attn_res mix is part of trunk output computation everywhere in the HF reference; skipping it starves the head of the snapshot-mixed signal |
| 2 | **post-attn_res-mix, pre-final-norm** | direct analog of DeepSeek's pre-norm trunk output (the lineage K3 descends from); `hnorm` re-norm makes a post-norm input redundant/distorting | none found |
| 3 | **post-final-norm** (what `KimiLinearModel.forward` returns today, `:1065`) | it's what `SpecDecOneEngineForCausalLM.forward` hands the spec worker with zero new plumbing | double normalization once the head applies `hnorm`; no precedent |

### Verdict

**Candidate 2 (post-attn_res-mix, pre-final-norm), MEDIUM confidence.**
Basis is lineage precedent (DeepSeek/K2.5 convention), not direct artifact
evidence — nothing shipped for K3 encodes the answer. Integration
consequence if confirmed: K3 needs a pre-norm capture point
(`KimiLinearModel.forward` applies `self.norm` before returning, so the
natural change is capturing/returning the post-mix pre-norm tensor for
`spec_metadata`), which is cheap but touches the same forward the SA work
owns — sequence it after SA-2 lands.

One K3-specific wrinkle with no precedent anywhere: **does the MTP draft
layer participate in attn_res?** If the head is a full K3 block trained
with snapshot mixing, `block_residual` must extend into the draft layer
(and per-step during multi-token drafting); if it was trained without
attn_res, it must not. This is invisible until weights/reference code drop.

### Moonshot asks (exact questions)

1. MTP head input: is it DeepSeek-style
   `eh_proj([enorm(embed(tok)); hnorm(h)])`? Which `h` — post-attn_res-mix
   pre-norm, or something else?
2. Does the MTP layer receive/extend the attn_res `block_residual` chain,
   or start fresh?
3. Checkpoint key schema for the head (layer index 93? `mtp.*` prefix?) —
   needed for the streaming-loader name plan (prep item 3).

## Item 5 — `sa_config` disqualifies rejection sampling: options

Verified at this ToT (`llmapi/llm_args.py:5114-5162`):
`rs_sa_active = speculative_config.sa_config is not None` → rejection
sampling is auto-disabled, and a **ValueError** is raised if
`use_rejection_sampling` was explicitly requested, reason string
"SA (sa_config) is active". Root cause is fundamental, not incidental: the
automaton emits tokens with no proposal distribution q(x), and RS
correctness requires one for every draft token; SA-overridden rows would
need q=δ(token), which the current sampler doesn't model.

| Option | Description | Cost/risk |
|---|---|---|
| A. Greedy + SA enhancer | accept the constraint; MTP+SA runs greedy acceptance (lossless, parity-checkable — same regime as all K3 spec work to date) | possible acceptance/quality gap vs RS at temperature>0; unmeasured |
| B. RS, no enhancer | neural-only drafts under rejection sampling | loses SA's cross-request/global-pool wins on repetitive workloads |
| C. Upstream feature | mixed acceptance: RS for neural rows, deterministic accept-test for SA-substituted rows | real sampler-correctness design work; upstream ownership unclear |

**Recommendation: A for bringup and measurement, B as the comparison arm.**
SA-3b's measurement phase should produce the A-vs-B numbers (tokens/step,
acceptance, e2e) on K3 workloads; only if both the SA override *and* RS
prove individually valuable does C justify an upstream ask. This also keeps
the K3 validation story uniform — greedy acceptance is what every parity
gate in the SA plan certifies.
