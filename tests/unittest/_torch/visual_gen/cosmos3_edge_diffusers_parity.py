# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-step Edge transformer parity: TRT-LLM vs diffusers main.

Not a pytest module — run as a subprocess by test_cosmos3_edge.py, because
diffusers main (>= 0.40, first with the Edge classes) cannot be imported into
a process that already imported the pinned diffusers.
Usage: DIFFUSERS_MAIN_PATH=/path/to/diffusers python cosmos3_edge_diffusers_parity.py <ckpt>

Runs the diffusers Cosmos3OmniPipeline for 2 steps (guidance off, fixed
latents), captures each transformer call's (latent, timestep, velocity),
then replays the same inputs through the TRT-LLM Edge transformer and
compares velocities.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.environ["DIFFUSERS_MAIN_PATH"], "src"))
os.environ.setdefault("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", "1")

CKPT = sys.argv[1]
DEV = "cuda"
H, W, FRAMES, STEPS = 192, 320, 9, 2
PROMPT = "A red cube sits on a wooden table."


def main():
    import diffusers
    import torch

    expected_root = os.path.realpath(os.environ["DIFFUSERS_MAIN_PATH"])
    assert os.path.realpath(diffusers.__file__).startswith(expected_root), diffusers.__file__
    print("diffusers:", diffusers.__version__)
    from diffusers import Cosmos3OmniPipeline

    pipe = Cosmos3OmniPipeline.from_pretrained(CKPT, torch_dtype=torch.bfloat16).to(DEV)

    records = []
    orig_forward = pipe.transformer.forward

    def spy(**kwargs):
        out = orig_forward(**kwargs)
        velocity = out[0][0] if isinstance(out, tuple) else out.sample[0]
        records.append(
            {
                "latent": kwargs["vision_tokens"][0].detach().clone(),
                "t": kwargs["vision_timesteps"].detach().reshape(-1)[0].clone(),
                "vel": velocity.detach().clone(),
            }
        )
        return out

    pipe.transformer.forward = spy

    latent_shape = (1, 48, (FRAMES - 1) // 4 + 1, H // 16, W // 16)
    init_latents = torch.randn(
        latent_shape, generator=torch.Generator().manual_seed(0), dtype=torch.float32
    ).to(DEV, torch.bfloat16)

    cond_ids, _ = pipe.tokenize_prompt(
        PROMPT, negative_prompt=None, num_frames=FRAMES, height=H, width=W, fps=24.0
    )

    pipe(
        prompt=PROMPT,
        num_frames=FRAMES,
        height=H,
        width=W,
        fps=24.0,
        num_inference_steps=STEPS,
        guidance_scale=1.0,
        generator=torch.Generator().manual_seed(0),
        latents=init_latents.clone(),
        output_type="latent",
    )
    print(f"captured {len(records)} transformer calls")

    del pipe.transformer
    del pipe
    torch.cuda.empty_cache()

    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineComponent, PipelineLoader
    from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

    args = VisualGenArgs(model=CKPT, torch_compile_config=TorchCompileConfig(enable=False))
    trt_pipe = PipelineLoader(args).load(
        skip_warmup=True,
        skip_components=[
            PipelineComponent.VAE,
            PipelineComponent.SCHEDULER,
            PipelineComponent.TOKENIZER,
            PipelineComponent.SOUND_TOKENIZER,
        ],
    )
    transformer = trt_pipe.transformer

    text_ids = torch.tensor([cond_ids], dtype=torch.long, device=DEV)
    text_mask = torch.ones_like(text_ids)
    latent_t, latent_h, latent_w = latent_shape[2:]

    transformer.reset_cache()
    for step, rec in enumerate(records):
        latent = rec["latent"].to(DEV, torch.bfloat16)
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)
        t = rec["t"].float().reshape(1).to(DEV)
        with torch.inference_mode():
            out = transformer(
                hidden_states=latent,
                timestep=t / 1000.0,
                raw_timestep=t,
                text_ids=text_ids,
                text_mask=text_mask,
                video_shape=(latent_t, latent_h, latent_w),
                fps=24.0,
            )
        ours = out.video[0].float().cpu()
        ref = rec["vel"].float().cpu()
        if ref.dim() == 5:
            ref = ref[0]
        diff = (ours - ref).abs()
        rel = diff.max() / ref.abs().max()
        print(
            f"step {step}: t={t.item():7.2f}  max|d|={diff.max():.5f}  "
            f"mean|d|={diff.mean():.6f}  rel={rel:.5f}  |ref|max={ref.abs().max():.3f}"
        )


if __name__ == "__main__":
    main()
