### LTX-2 Specific Checkpoint Format

LTX-2 specific checkpoints pack all model components into a **single safetensors
file** with prefixed tensor keys. This document describes the layout using the
BF16 checkpoint as a reference.

#### File Overview

```
ltx-2-19b-dev.safetensors
  Total tensors : 6,404
  Metadata keys : license, encrypted_wandb_properties, _quantization_metadata, config
```

The `config` metadata key contains a JSON dict with per-component configuration
(e.g., `config["transformer"]`). The `_quantization_metadata` key holds the
ModelOpt quantization recipe (present only in quantized checkpoints).


#### Similarity to standard HF single safetensors:
  - Single .safetensors file containing all weights
  - Standard safetensors binary format

Key differences:

1. Embedded config in metadata — the safetensors header contains a "config" key with the full JSON config for all components (transformer, VAE, audio VAE,
vocoder). Standard HF models keep config in a separate config.json.
2. Non-standard weight key prefixes:
- Transformer: model.diffusion_model.* (not transformer.* or bare keys)
- Video VAE: vae.decoder.*
- Audio VAE: audio_vae.decoder.*
- Vocoder: vocoder.*
3. Multiple components in one file — the single checkpoint bundles the denoiser, video VAE, audio VAE, vocoder, and connectors together. Standard HF checkpoints
are typically one model per file.
4. Text encoder is separate — Gemma3 lives in its own directory and is loaded via the standard from_pretrained() path.

Detection Logic in the TRT-LLM codebase
1. No model_index.json present → not diffusers
2. Safetensors metadata "config" key contains both "transformer" and "vae" → LTX2Pipeline


#### Component Prefixes

Every tensor key is prefixed by its component name. The weight loader strips the
prefix when loading (e.g., `model.diffusion_model.proj_out.weight` becomes
`proj_out.weight` for the transformer).

| Prefix | Component | Tensors | Description |
|--------|-----------|---------|-------------|
| `model.diffusion_model.` | Transformer (DiT) | 5,920 | Video + audio denoising transformer |
| `vae.` | Video VAE | 187 | Video encoder/decoder |
| `audio_vae.` | Audio VAE | 102 | Audio encoder/decoder |
| `vocoder.` | Vocoder | 194 | Mel-spectrogram to waveform |
| `text_embedding_projection.` | Text projection | 1 | Aggregated text embedding projection |
