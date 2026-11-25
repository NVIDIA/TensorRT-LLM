# MiniMax-H3 static FP8 descale table format

When using `fp8_scale_fusion="static"`, set `fp8_static_scales` to a JSON file mapping `str(layer_idx)` to `[descale_q, descale_k, descale_v]` (either a bare layer map or wrapped under a top-level `"descale_max"` key):

```json
{
  "descale_max": {
    "0": [0.025, 0.025, 0.04],
    "1": [0.025, 0.025, 0.04]
  }
}
```

Use `attention_fmha="cudnn"` or `"trtllm-gen"` with `"static"`; FP16-softmax (`"trtllm-gen-fp16sm"`) requires `"qkstatic"` instead.

Each descale entry is `max(amax / 448, 1 / (448 * 512))` rounded to BF16, matching dynamic mode's per-call scale.
`fp8_static_margin` (default 1.0) multiplies each table entry before its BF16 rounding.
The default `qkstatic` mode uses fixed Q/K scales and dynamic V scales and does not require a scale table file.
