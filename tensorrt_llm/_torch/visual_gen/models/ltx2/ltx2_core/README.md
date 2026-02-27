# LTX-2 Core Components

Ported from [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)
(`packages/ltx-core/src/ltx_core/`). Minimal subset for one-stage and
two-stage inference — no training code.

## Architecture-specific components (ported from Lightricks)

| File | Purpose |
|------|---------|
| `rope.py` | 3D RoPE with interleaved/split variants and fractional positions |
| `adaln.py` | AdaLayerNormSingle — timestep → scale/shift/gate modulation |
| `timestep_embedding.py` | Sinusoidal timestep + PixArt-Alpha combined embeddings |
| `text_projection.py` | PixArtAlphaTextProjection — caption embedding projection |
| `modality.py` | `Modality` dataclass (latent, timesteps, positions, context) |
| `transformer_args.py` | `TransformerArgs` + preprocessors that wire patchify/AdaLN/RoPE |
| `normalization.py` | GroupNorm, PixelNorm, and factory for normalization layers |
| `attention.py` | Pure-PyTorch Attention + FeedForward used by the 1D connector |

## Diffusion pipeline components (ported from Lightricks)

| File | Purpose |
|------|---------|
| `schedulers.py` | LTX2Scheduler — token-count-dependent sigma shifting |
| `scheduler_adapter.py` | Wraps LTX2Scheduler + EulerDiffusionStep into diffusers-like API |
| `diffusion_steps.py` | EulerDiffusionStep for the Euler method |
| `guiders.py` | Multi-modal guidance (CFG + STG + modality) |
| `perturbations.py` | STG perturbation configs for attention masking |
| `protocols.py` | Protocol definitions (Patchifier, Scheduler, DiffusionStep) |
| `types.py` | VideoPixelShape, VideoLatentShape, AudioLatentShape, scale factors |
| `utils_ltx2.py` | `to_velocity`, `rms_norm` utilities |

## Text & latent connectors

| File | Purpose |
|------|---------|
| `connector.py` | Embeddings1DConnector, GemmaFeaturesExtractorProjLinear |
| `patchifier.py` | VideoLatentPatchifier, AudioPatchifier, pixel coord conversion |

## VAE decoders

| Subpackage | Purpose |
|------------|---------|
| `video_vae/` | Video decoder with spatial/temporal tiling support |
| `audio_vae/` | Audio decoder + Vocoder (mel → waveform) |

## Transformer (TRT-LLM native) — see `../transformer_ltx2.py`

The transformer itself (`LTXModel`, `BasicAVTransformerBlock`, `LTX2Attention`)
is **not** in this package. It lives in `transformer_ltx2.py` and is built
from TRT-LLM optimized primitives:

- `tensorrt_llm._torch.modules.linear.Linear` (quantization, TP)
- `tensorrt_llm._torch.modules.rms_norm.RMSNorm` (fused kernels)
- `tensorrt_llm._torch.modules.mlp.MLP` (GELU-tanh activation)
- `tensorrt_llm._torch.visual_gen.attention_backend` (TRT-LLM/SDPA backends)
