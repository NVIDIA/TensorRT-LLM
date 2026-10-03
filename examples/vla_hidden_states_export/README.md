# Exporting Full Hidden States from TensorRT-LLM for VLA Models

## Background

VLA (Vision-Language-Action) models such as Orion, OpenVLA, and RT-2 use the
LLM's hidden states as input to downstream task heads (e.g. planning, trajectory
prediction) rather than for token generation:

- Standard LLM: input -> LLM -> logits -> token sampling -> text
- VLA model: images+text -> LLM -> hidden_states -> planning head -> trajectory

The downstream head needs the hidden state at a **specific token position**
(e.g. a "waypoint" token), which requires the complete hidden_states tensor,
not just the last token's logits.

See [issue #4414](https://github.com/NVIDIA/TensorRT-LLM/issues/4414) for
community demand (open since May 2025).

## Why Existing APIs Fall Short

| API | Limitation |
|-----|-----------|
| `gather_last_token_logits` | Compresses to last token only |
| `additional_model_outputs` (v1.1+) | Requires model `forward` to return `hidden_states` in its output dict; the standard `DecoderModelForCausalLM.forward` only returns logits |
| `SaveHiddenStatesDecodingConfig` | Offline only (EAGLE3 training), saves to disk |

## Solutions

### Solution A: TRT Backend (v0.7-v0.21)

For TRT network-based builds, insert `mark_output` before
`gather_last_token_logits` in `modeling_utils.py`:

```python
if self.config.mapping.is_last_pp_rank():
    # Export full hidden_states before compression
    hidden_states.mark_output('full_hidden_states', self.config.dtype)
    hidden_states = gather_last_token_logits(...)
    lm_logits = self.lm_head(hidden_states)
```

**Note on tensor shape**: `mark_output` exposes the tensor as-is without
reshaping. With `remove_input_padding` enabled, the shape may be packed
`[num_tokens, hidden_dim]` rather than `[batch, seq_len, hidden_dim]`.

Reading at inference:

```python
full_hs = model.session.debug_buffer["full_hidden_states"]
# Shape is [batch, seq_len, hidden_dim] or [num_tokens, hidden_dim] (packed)
ego_feature = full_hs[0, waypoint_idx, :]  # or full_hs[waypoint_idx, :]
```

### Solution B: PyTorch Backend (v1.x)

In v1.x, `DecoderModelForCausalLM.forward` returns logits only. To expose
hidden_states via `additional_model_outputs`, modify `forward` to return a dict
when requested. The key insight: `self.model()` returns the full tensor
before `LogitsProcessor` compresses it. See `patches/modeling_utils_v1x.patch`
for the approach, and refer to `handle_additional_outputs.py` for the
framework's dict-return contract.

**Note**: On the PyTorch backend with `remove_input_padding` (default), the
shape is packed `[num_tokens, hidden_dim]`.

## Verification

Solution A was tested on Orion VLA (ICCV 2025) with TRT-LLM v0.13.0:

- Engine output: `[1, 599, 4096]`
- Hidden_states CosSim vs PyTorch: 0.9994 (INT8)
- End-to-end plan_L2_1s: 0.686 (PyTorch: 0.690)

## Files

| File | Description |
|------|-------------|
| `patches/modeling_utils_v0x.patch` | Solution A (v0.7-v0.21, verified) |
| `patches/modeling_utils_v1x.patch` | Solution B (v1.x, conceptual) |
| `inference_python.py` | Python inference example |
| `tests/test_hidden_states.py` | Tests |
