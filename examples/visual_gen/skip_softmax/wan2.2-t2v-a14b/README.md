# Wan 2.2 T2V-A14B Skip Softmax Example

This directory contains the component-specific ModelOpt calibration metadata needed to use
`target_sparsity` with Skip Softmax Attention on
[`Wan-AI/Wan2.2-T2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers).
It also provides a VisualGen configuration that combines NVFP4 GEMMs, SAGE attention, and Skip
Softmax Attention.

Wan 2.2 uses separate high-noise and low-noise transformers. Their calibration formulas differ,
so each component has its own overlay:

- `transformer_sparse_attention.json` applies to `transformer/config.json`.
- `transformer_2_sparse_attention.json` applies to `transformer_2/config.json`.

Each overlay also contains an `ignore` list. Matching attention layers remain dense; Skip Softmax
is applied only to the calibrated layers outside that list. The two transformers can have
different formulas and ignore lists, so do not combine the overlays or reuse them for another
model.

## Apply the calibration metadata

Set `MODEL_DIR` to a local copy of the public Diffusers checkpoint, then run:

```bash
python examples/visual_gen/skip_softmax/wan2.2-t2v-a14b/apply_calibration.py \
    --model-dir "$MODEL_DIR"
```

The helper validates the checkpoint variant and both overlays before modifying either component.
It adds only the top-level `sparse_attention_config` field and leaves every other checkpoint field
unchanged. Running it again is a no-op. If the checkpoint already contains different sparse
attention metadata, the helper stops instead of replacing it; use a fresh checkpoint copy or pass
`--force` when replacement is intentional.

## Run VisualGen

The included [`visual_gen.yaml`](visual_gen.yaml) selects the operating point used in the
accompanying video-generation optimization blog: `target_sparsity=0.65` and
`disabled_until_timestep=0.86`, together with dynamic NVFP4 GEMM quantization and INT8/FP8 SAGE
attention.

This dynamic quantization path is intended to reproduce the current characterization from a BF16
checkpoint. If a matching prequantized ModelOpt checkpoint is available, use a compatible
configuration and validate its latency and quality separately; dynamic and static quantization
results are not interchangeable.

```bash
trtllm-serve "$MODEL_DIR" \
    --visual_gen_args examples/visual_gen/skip_softmax/wan2.2-t2v-a14b/visual_gen.yaml
```

The overlay files contain a calibrated default target sparsity of 0.5. The value in
`visual_gen.yaml` overrides that default at runtime, and each transformer converts the requested
target to a threshold with its own calibration formula.
