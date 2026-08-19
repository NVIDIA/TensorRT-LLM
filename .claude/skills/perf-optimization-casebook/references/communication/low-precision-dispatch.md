---
id: case-low-precision-dispatch
type: case
family: communication
maturity: full
bottleneck: [communication]
signals: [alltoall-dominates, full-precision-on-wire]
architectures: [sm90, sm100]
model_scope: [moe, deepseek-v3r1, qwen3-moe]
phase: [any-phase]
patterns: [pattern-send-activations-post-quant]
accuracy_risk: lossy
apply_via_kind: [env-var]
knobs: [TRTLLM_MOE_POST_QUANT_ALLTOALLV, TRTLLM_MOE_USE_LOW_PRECISION_COMBINE, TRTLLM_FORCE_ALLTOALL_METHOD]
specialists: [perf-sweep-challenger, trtllm-moe-develop]
commits: ['854655f2f7b3', 'f172face98cf', '336c2ef5408d']
eligibility:
  - "quantized checkpoint required: has_fp8_qdq / has_nvfp4 / has_w4afp8"
  - "hidden size divisible by 32 for nvfp4/fp8 low-latency dispatch (as of #7927)"
interactions:
  - {case: case-deepep, relation: depends-on, note: "rides on the DeepEP all-to-all path; requires DeepEP enabled"}
measured: []
---

# Communicate MoE tokens in low precision (FP4/FP8 post-quant dispatch & combine)

> Part of the [Communication casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `854655f2f7b3` deepEP fp4 post quant all2all dispatch (#5881); related: `f172face98cf` DeepEP LL dispatch FP4 (#6296), `336c2ef5408d` DeepEP LL fp8 dispatch/combine (#7927).
- **Applies when:** communication-bound + signals: MoE already on a DeepEP all-to-all (see the [DeepEP case](deepep.md)) and the model is FP8-QDQ / NVFP4 / W4A8 quantized; dispatch/combine bf16 byte volume dominates; large EP world size where comm bytes scale with tokens×hidden.
- **Mechanism:** quantize tokens *before* the all-to-all and ship the packed low-precision payload instead of bf16 — ~½ (FP8) or ~¼ (FP4) the dispatched bytes. FP8 dispatches `float8_e4m3fn` directly. FP4 (#5881) packs NVFP4 data + `uint8` scale factors into a bf16-typed buffer of `packed_hidden_size=2560` (LL kernel accepts bf16 hidden sizes 2560/4096/5120/7168), unpacks + re-`swizzle_sf` on receive; #6296 adds native `low_latency_dispatch_fp4`; #7927 generalizes LL fp8/nvfp4/w4afp8 dispatch and adds `low_latency_combine_low_precision(precision="fp8", ...)` so the combine leg is also low-precision.
- **Generalizes to:** "send activations post-quant, not pre-quant" — any quantized collective: EP combine, TP allgather/reducescatter of quantized activations, KV transfer; carries to FP8 and FP4 dispatch+combine; adapt by packing scale factors with the payload, re-swizzling on the receiver, honoring per-format hidden-size/divisibility (`%32` for nvfp4/fp8 LL).
- **Apply via:** env `TRTLLM_MOE_POST_QUANT_ALLTOALLV=1` (post-quant dispatch), `TRTLLM_MOE_USE_LOW_PRECISION_COMBINE=1` (low-precision combine), `TRTLLM_FORCE_ALLTOALL_METHOD` to pin method; requires DeepEP + a quantized checkpoint (`has_fp8_qdq`/`has_nvfp4`/`has_w4afp8`). Delegate accuracy gating to **perf-sweep-challenger**.
- **Expected effect:** lower dispatch/combine latency + higher EP throughput from reduced comm bytes; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossy — tokens cross the wire in FP4/FP8 (and combine may be low-precision); partial sums/activations lose precision vs bf16 transport. Needs accuracy record + rollback criterion; FP4 combine is the highest-risk leg.
- **Verify:** task-accuracy/perplexity parity vs the bf16-dispatch DeepEP baseline (and vs low-precision-combine off); confirm throughput gain; check FP4 pack/unpack round-trip (SF swizzle) correctness.
- **Rollback:** `TRTLLM_MOE_POST_QUANT_ALLTOALLV=0` and `TRTLLM_MOE_USE_LOW_PRECISION_COMBINE=0` (revert to bf16 dispatch/combine). Trigger: accuracy drop beyond recorded threshold.
- **Prior art:** PRs #5881, #6296, #7927. Files: `_torch/moe/fused_moe/fused_moe_wide_ep.py` (`alltoall_postquant_dispatch`, `low_latency_dispatch_fp4`, `low_latency_combine_low_precision`), `deep_ep_utils.py`, `thop/moeOp.cpp`. Owning skill: **trtllm-moe-develop**; gate with **perf-sweep-challenger**.
