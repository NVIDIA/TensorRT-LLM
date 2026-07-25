# Kimi K3 DFlash/DSpark: scaffold

Status: **scaffold + real drafter schema** — the trained K3 drafter is in
progress; its final config and tensor schema are now known from the
training team's dummy-weight checkpoint (`dummy-dspark0724`, 2026-07-24).
This branch (`brnguyen/kimi-k3-dflash-scaffold`) wires the target side and
provides synthetic-weight tooling matching that schema so the full path
runs end-to-end before real weights drop. Follows the
`brnguyen/kimi-k3-mtp-scaffold` precedent.

## The real drafter is DSpark, not plain DFlash

The K3 drafter is a **DSpark** drafter — DeepSeek's DFlash follow-up
(arXiv 2607.05147; reference impl in deepseek-ai/DeepSpec; NeMo AutoModel
ships a training recipe). Architecture per the real config:

- **Backbone**: dense Qwen3-style parallel block drafter, 6 layers,
  32 Q heads / 8 KV heads / head_dim 128, intermediate 12288, q/k-norm,
  plain RoPE (theta 1e4, no scaling), `block_size` 8 — same structural
  family as nvidia/Kimi-K2.7-Code-DFlash, and loadable by the generic
  `DFlashForCausalLM` via the `model_type: "qwen3"` fallback.
- **DFlash projection**: `fc` `[7168, 43008]` + `hidden_norm`, reading 6
  captured target layers; no embed_tokens/lm_head (shared with target).
- **`target_layer_ids: [1, 19, 37, 54, 72, 90]`** over K3's 93 layers —
  exactly the even-spacing convention this scaffold predicted.
- **`mask_token_id: 163606`** (NOT vocab-2).
- **DSpark extras** (`dflash_config`): `projector_type: "dspark"`,
  `causal: false`, `use_swa: true` / `swa_window_size: 1024`
  (`layer_types: sliding_attention`), `shift_label: true`, low-rank
  **Markov head** (`markov_rank: 256`, `markov_head_type: "vanilla"`;
  tensors `markov_w1/w2 [163840, 256]` — token-conditioned intra-block
  logit bias), and a **confidence head**
  (`confidence_proj [1, 7424 = hidden + markov_rank]` + bias — per-position
  acceptance prediction for confidence-scheduled verification).

Checkpoint = 73 tensors (K2.7's 69-tensor DFlash layout + the 4 dspark
head tensors), verified against the dummy checkpoint's safetensors header.

### Runtime gap: dspark semantics not implemented

The generic `DFlashForCausalLM` loads and runs this checkpoint as **plain
DFlash**: the Markov and confidence weights are dropped at load
(`allow_partial_loading`), the intra-block Markov logit bias and
confidence-scheduled verification are skipped, and the draft block
attention runs non-causal full-window (no SWA) without the shift_label
convention. `DFlashDraftModel` logs a warning when these config fields are
present. Consequences:

- Wiring, capture, verify and Mamba-state rewind are fully exercisable.
- With real trained weights, acceptance will be **degraded** vs the
  drafter's potential (DSpark's reported gains over DFlash come from the
  Markov + confidence heads) — and possibly degraded vs plain DFlash if
  the backbone distribution depends on SWA/shift_label at context >1024.
- Implementing dspark semantics (markov bias in the block-decode sampling
  loop, optional confidence-based draft-length scheduling, SWA window in
  `flash_attn_with_kvcache`, shift_label alignment) is the follow-up work
  item; the DeepSpec reference and llama.cpp PR #25173 are the guides.

## What this branch changes

1. `modeling_kimi_linear.py`: the SA-only spec-dec assert now also admits
   `is_dflash()`.
2. `KimiLinearModel.forward`: explicit `spec_metadata` parameter (it was
   previously swallowed by `**kwargs`) + per-layer
   `maybe_capture_hidden_states(layer_idx, prefix_sum, None)` calls.
   Base `SpecMetadata` no-ops this, so SA and no-spec paths are
   unaffected.
3. `examples/kimi_k3/make_synthetic_dflash_drafter.py`: emits a
   random-weight drafter in the real dspark schema (73 tensors).
   `--config` adopts a real drafter config.json verbatim (authoritative
   mode); `--ckpt-dir` reads the real K3 target config for dims with
   drafter dims defaulting to dummy-dspark0724's; `--tiny` for unit tests.
4. `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dflash_scaffold.py`:
   schema round-trips (K2.7 reference + real dspark), decoding-config
   predicates, capture threading, and (GPU-gated) capture-buffer
   semantics.
5. `modeling_speculative.py`: warning when a drafter declares dspark
   features the runtime ignores.
6. `examples/kimi_k3/eval_extra_llm_options_dflash.yaml`: DEP16 eval
   config with a drafter-path placeholder.

## Remaining questions for the drafter training team

- **Capture tap point.** K3's attention-residual scheme means each layer
  returns a running *prefix sum* (residual already folded in). We capture
  that with `residual=None`. Qwen3/Llama capture `hidden + residual` at
  the analogous point, which for K3 is the same quantity — but the drafter
  must be *trained* against whatever the capture emits. Confirm the
  training-side hook matches the prefix-sum convention.
- **SWA + non-causal draft attention.** The config says `causal: false`
  with `swa_window_size: 1024`. Confirm how the window applies over the
  (projected-context + block) KV layout at draft time — needed before
  implementing SWA in the block decode.
- **shift_label.** Confirm the exact label-shift convention so drafter
  logits align with the right positions at inference.
- **DFlash x attention-DP parity.** SA x attention-DP was
  parity-certified; DFlash has not been. Rerun the parity harness
  (`brnguyen/k3-disagg-parity-harness`) once real weights exist.

## Exercising the wiring today (dummy weights)

The training team's dummy checkpoint works directly:

```bash
sbatch examples/kimi_k3/run_gsm8k_kimi_k3.sbatch \
    --model $K3_CKPT --image $SQSH --dflash ~/lustre/dummy-dspark0724
```

Or generate a synthetic checkpoint from a real drafter config:

```bash
python examples/kimi_k3/make_synthetic_dflash_drafter.py \
    --config <drafter_config.json> --out $SCRATCH/k3-dflash-synth
```

Outputs are gibberish by construction: acceptance ~0, no speedup. This
validates capture -> fc/hidden_norm projection -> drafter forward ->
verify plumbing only.
