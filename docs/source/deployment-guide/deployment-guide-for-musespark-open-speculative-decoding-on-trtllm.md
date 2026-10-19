# Deployment Guide for MuseSparkOpen Speculative Decoding on TensorRT LLM - GB300

## Introduction

This guide records how suffix-automaton (SA) speculative decoding behaves on MuseSparkOpen (NVFP4, TP4/EP4, FlashInfer
attention) on a GB300 tray, what the serving recipe must state, and how to re-run the accuracy gates. The
measurements come from the baseline aggregated serving recipe re-measured on the same build, so every ratio below is against the
configuration that actually serves the model.

SA drafts by matching the request's own token history (no draft model, no extra weights), verifies the drafts in the
target's forward and runs inside CUDA graphs. It pays when the output repeats earlier text (structured output, code
edits, tool-call loops) and costs a verification of `1 + max_draft_len` positions per step when it does not.

## Prerequisites

- The MuseSparkOpen NVFP4 checkpoint and its tokenizer (`custom_tokenizer: musespark_open`).
- A build containing the speculative-decoding fixes for this model: the FlashInfer one-engine gate, the draft-KV-cache
  rule for SA, the decoder admission (`assert_supported_spec_config`) and the guided-decoding rollback
  (`CapturableGuidedDecoder.rollback_rejected_batch`). The preview container pinned in the main guide's
  [Run Docker Container](deployment-guide-for-musespark-open-on-trtllm.md#run-docker-container) section contains all of these.
- 4 GPUs (one tray). The tests below assume `LLM_MODELS_ROOT`-style access to the checkpoint through `MUSESPARK_REFERENCE_CKPT`.

## Recipe

Start from the baseline aggregated serving YAML and add:

```yaml
attn_backend: FLASHINFER      # the speculative recipe is validated on FlashInfer (the main guide's aggregated recipe ships TRTLLM)
speculative_config:
  decoding_type: SA
  max_draft_len: 2            # 2 measured best on unmatched text; 4 gives the highest acceptance on real text
  max_matching_ngram_size: -1 # longest match
disable_overlap_scheduler: true   # required: SA + overlap scheduler corrupts generation on FlashInfer (see Limits)
enable_iter_perf_stats: true      # exposes acceptance counters per iteration
```

and start the server with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in its environment when
`max_draft_len >= 3` serves 256 concurrent requests (see Limits); this keeps the baseline KV fraction of 0.8.

Everything else stays as in the main guide's baseline recipe: `cuda_graph_config.max_batch_size: 256`,
chunked prefill, `moe_config.backend: TRTLLM`, `allreduce_strategy: NCCL`, `guided_decoding_backend: xgrammar`. Note the attention backend differs: this speculative recipe is validated with `attn_backend: FLASHINFER`, while the main guide's aggregated recipe ships `attn_backend: TRTLLM`.

## Limits the recipe must state

| limit | effect | source |
|---|---|---|
| Overlap scheduler must be off | with overlap on, SA generations diverge from the non-speculative engine on 14 of 20 prompts (12 tokens outside the reference top-5, margins up to 9.2 nats); the FlashInfer KV-length correction for one-engine drafters needs the trtllm-gen decode kernels, which do not exist for head_dim 64 on this SM | accuracy validation |
| `min_tokens` > 0 is rejected | every request with a length floor fails with `min_length is not supported with one-model speculative decoding`; clients must use `ignore_eos` or prompting | serving validation |
| `max_draft_len >= 3` at 256 requests runs out of memory with the default allocator | the forward fails with 0.3-0.7 GiB free while about 18 GiB is reserved but unallocated (fragmentation); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the server removes it at the baseline KV fraction (d=3: 1347 tok/s, d=4: 1377 tok/s at c=256); lowering the KV fraction to 0.6 also works but costs cache (d=4: 1272 tok/s) | serving validation |
| Guided decoding needs the rollback fix | without it SA emits grammar-forbidden tokens (10 of 20 JSON prompts); with it 0 of 20 | guided-decoding validation |
| Overlap-off carry cost | the deployed recipe owes 39 % of its c=1 throughput and 15 % of its c=256 throughput to the overlap scheduler; SA pays that before any speculation gain | serving validation |

## Measurements (shape A: 8,192 input / 1,024 output tokens, aiperf 0.12.0, streaming chat, `ignore_eos`)

Output tokens per second; the ratio is against the mean of two baseline runs (overlap on). Spread between the two runs:
0.4 % at c=1, 7.5 % at c=32, 0 % at c=256.

| recipe | c=1 | c=32 | c=256 |
|---|---|---|---|
| baseline AGG (overlap on), run 1 / run 2 | 74.7 / 75.0 | 1065 / 988 | 2432 / 2432 |
| baseline AGG with overlap off | 45.3 (0.61) | 986 (0.96) | 2071 (0.85) |
| SA d=2, overlap off | 62.3 (0.83) | 853 (0.83) | 1609 (0.66) |
| SA d=3, overlap off | 57.5 (0.77) | 745 (0.73) | 1347 (0.55) with expandable segments |
| SA d=4, overlap off | 61.6 (0.82) | 843 (0.82) | 1377 (0.57) with expandable segments; 1272 (0.52) at kv 0.6 |

Random-token prompts give the drafter nothing to match (acceptance rate about 2 %), so these rows are the cost floor.
The forced-acceptance sweep gives the cost curve, i.e. the throughput when exactly n drafts are accepted every step:

| forced | c=1 | c=32 | c=256 |
|---|---|---|---|
| d=2, 1 of 2 accepted (AL 2) | 1.22 | 1.32 | 1.00 |
| d=2, 2 of 2 accepted (AL 3) | 1.80 | 1.70 | 1.14 |
| d=4, 1 of 4 accepted (AL 2) | 1.23 | 1.07 | 0.94 |
| d=4, 2 of 4 accepted (AL 3) | 2.04 | 1.45 | 1.10 |
| d=4, 3 of 4 accepted (AL 4) | 2.64 | 1.93 | 1.18 |
| d=4, 4 of 4 accepted (AL 5) | 3.25 | 2.17 | 1.21 |

Break-even acceptance length is therefore about 1.44 tokens per step at c=1, 1.35 at c=32 and 2.0 at c=256 for d=2, and about 1.9, 2.7 and 3.5 for d=4.

Acceptance measured on real text (full checkpoint, greedy, one request at a time):

| workload | d | acceptance rate | mean acceptance length |
|---|---|---|---|
| mixed prompts (parity set) | 4 | 0.144 | 1.80 tokens/iteration |
| mixed prompts | 2 | 0.268 | 1.62 |
| chat / tool-call refset (15 turns) | 4 | 0.165 | 1.64 |
| GSM8K raw 5-shot, n=100 | 4 | 0.113 | 1.50 |
| copy-heavy prompts (acceptance-floor set) | 4 | — | 1.50 |
| guided JSON (xgrammar) | 4 | 0.154 | 1.40 |

## Decision guidance

Speculation pays at a concurrency when the acceptance length exceeds the break-even above and the gain exceeds the
run-to-run spread. With acceptance lengths of 1.5-1.8 on real text, SA d=2 is at or slightly above break-even at c=1
and c=32 and below it at c=256; on unmatched text it is 17-34 % slower than the deployed recipe. Recommendation:
ship the day-0 recipe without speculation; offer SA d=2 (overlap off) as an opt-in for low-concurrency or copy-heavy
serving; revisit when the overlap scheduler works with one-engine drafters on FlashInfer.

## Correctness tests

Accuracy under speculation is gated by the shipping per-model suite
(`tests/integration/defs/accuracy/test_musespark_open.py::TestMuseSparkOpen`): the SA entry in
`references/gsm8k.yaml` requires the speculative engine to reproduce the target model's GSM8K floor.

For guided decoding, replay the output tokens through a fresh xgrammar matcher: the near-tie rule cannot see a forbidden
token when the masked distribution is flat.

## Credit

The suffix-automaton drafter and its GPU kernels are Baseten's `sa_spec` contribution to TensorRT LLM.
