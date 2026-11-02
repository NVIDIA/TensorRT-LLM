# MiniMax-H3 static FP8 descale table

`minimax_h3_fp8_qkv_descales_544x960x124.json` maps `str(layer_idx)` to `[descale_q, descale_k, descale_v]`
for the fused QK-norm + RoPE + FP8 cast (`fused_fp8_qk.py`) in static mode
(`H3_FP8_SCALE_FUSION=static H3_FP8_STATIC_SCALES=<this file>`). A descale is
`max(amax / 448, 1 / (448 * 512))` rounded to BF16, the value the dynamic mode computes per call.

The table was calibrated on 12 held-out prompts at 544x960x124 (seed 42, 28 steps) by running the dynamic
mode with `H3_FP8_AMAX_LOG=<json>` and keeping the element-wise maximum. `H3_FP8_STATIC_MARGIN` (default 1.0)
widens every scale uniformly. Static mode exists for multi-node Ulysses, where the dynamic mode's per-layer
amax all-reduce crosses the fabric; single-node runs use the dynamic mode.
