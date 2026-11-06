# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the real-trajectory DiT input fixture for the Wan 2.2 A14B L2 tests.

Runs the HF diffusers WanPipeline end to end (both experts, CFG, no VAE decode)
and records what each expert receives on every call. Embeddings are stored
without their zero padding and latents once per step (the CFG calls of a step
share them), so 16 DiT calls fit in about 1 MB. See README.md in this directory.

Usage (one GPU with room for both 14B experts):
    python generate_wan22_dit_inputs.py <Wan2.2-T2V-A14B-Diffusers dir> <output.safetensors>
"""

import json
import sys

import torch
from diffusers import WanPipeline
from safetensors.torch import save_file

PROMPTS = (
    ("A cat sitting on a sunny windowsill watching birds outside.", 17),
    ("Ocean waves rolling onto a sandy beach at sunset.", 29),
)
NEGATIVE_PROMPT = ""
HEIGHT, WIDTH, FRAMES, STEPS, GUIDANCE = 320, 320, 9, 4, 5.0
MAX_SEQUENCE_LENGTH = 512


def _valid_tokens(embedding: torch.Tensor) -> torch.Tensor:
    """Strip the zero padding WanPipeline appends after the real tokens."""
    (rows,) = embedding.abs().amax(dim=-1).nonzero(as_tuple=True)
    length = int(rows.max()) + 1
    if embedding[length:].any():
        raise RuntimeError("Padding is not all zero")
    return embedding[:length].contiguous()


def main(checkpoint: str, output: str) -> None:
    pipeline = WanPipeline.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    calls = []
    for expert in ("transformer", "transformer_2"):

        def hook(_module, _args, kwargs, _output, expert=expert):
            calls.append(
                {
                    "expert": expert,
                    "hidden_states": kwargs["hidden_states"].detach().cpu(),
                    "encoder_hidden_states": kwargs["encoder_hidden_states"].detach().cpu(),
                    "timestep": kwargs["timestep"].detach().cpu(),
                }
            )

        getattr(pipeline, expert).register_forward_hook(hook, with_kwargs=True)

    tensors, cases, texts = {}, [], {}
    with torch.inference_mode():
        for prompt_index, (prompt, seed) in enumerate(PROMPTS):
            start = len(calls)
            pipeline(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_frames=FRAMES,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                generator=torch.Generator().manual_seed(seed),
                output_type="latent",
            )
            prompt_calls = calls[start:]
            if len(prompt_calls) != 2 * STEPS:
                raise RuntimeError(f"Expected {2 * STEPS} DiT calls, saw {len(prompt_calls)}")
            for step in range(STEPS):
                cond, uncond = prompt_calls[2 * step : 2 * step + 2]
                if not torch.equal(cond["hidden_states"], uncond["hidden_states"]):
                    raise RuntimeError("CFG branches of one step must share the latent")
                latent = f"latent.p{prompt_index}.s{step}"
                tensors[latent] = cond["hidden_states"][0].contiguous()
                for branch, call, text_prompt in (
                    ("cond", cond, prompt),
                    ("uncond", uncond, NEGATIVE_PROMPT),
                ):
                    embedding = call["encoder_hidden_states"]
                    if embedding.shape != (1, MAX_SEQUENCE_LENGTH, 4096):
                        raise RuntimeError(f"Unexpected embedding {tuple(embedding.shape)}")
                    text = texts.setdefault(text_prompt, f"text.{len(texts)}")
                    valid = _valid_tokens(embedding[0])
                    if text in tensors:
                        if not torch.equal(tensors[text], valid):
                            raise RuntimeError(f"Embedding of {text_prompt!r} is not deterministic")
                    else:
                        tensors[text] = valid
                    cases.append(
                        {
                            "expert": call["expert"],
                            "prompt": prompt_index,
                            "step": step,
                            "branch": branch,
                            "latent": latent,
                            "text": text,
                            "timestep": float(call["timestep"].reshape(-1)[0]),
                        }
                    )
    metadata = {
        "cases": json.dumps(cases),
        "texts": json.dumps({key: text for text, key in texts.items()}),
        "prompts": json.dumps(PROMPTS),
        "generation": json.dumps(
            {
                "height": HEIGHT,
                "width": WIDTH,
                "num_frames": FRAMES,
                "num_inference_steps": STEPS,
                "guidance_scale": GUIDANCE,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "generator": "torch.Generator() (CPU), manual_seed per prompt",
            }
        ),
        "diffusers": __import__("diffusers").__version__,
        "torch": torch.__version__,
    }
    save_file(tensors, output, metadata=metadata)
    print(json.dumps(cases, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
