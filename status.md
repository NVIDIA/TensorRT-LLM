# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Nemotron Nano Omni Latest Status

## Multimodal Prompt Status

Latest verified prompt file:

- `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/TensorRT-LLM/multimodal_prompts.yaml`

Latest full-layer verification:

- `bash -ic "f8 && python examples/auto_deploy/build_and_run_ad.py --model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning --args.yaml-extra examples/auto_deploy/model_registry/configs/dashboard_default.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/world_size_4.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/nano_omni.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/nano_omni_multimodal_smoke.yaml --yaml-extra /lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/TensorRT-LLM/multimodal_prompts.yaml"` passed

Observed output:

```text
So, let's look at each image. First image: a Shiba Inu dog, close-up with part of its face and body, setting seems indoor with soft colors. Second image: a cat in snow, showing different parts like face, body, paws. Need to describe each in 10 words or less.

First image: Dog on furniture, indoor scene, detailed features. Let's count. "Shiba Inu dog on furniture, close-up with background items."
```

Interpretation:

- multimodal image grounding is working on the real AD path
- the first image is recognized as a Shiba Inu dog indoors/on furniture
- the second image is recognized as a cat in snow
- prompt-following is still imperfect because the model responds verbosely instead of giving two short captions

Related detailed running log:

- see [omni_status.md](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat8/TensorRT-LLM/omni_status.md)
