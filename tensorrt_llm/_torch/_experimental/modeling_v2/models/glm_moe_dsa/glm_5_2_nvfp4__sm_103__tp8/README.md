# `glm_5_2_nvfp4__sm_103__tp8`

GLM-5.2 NVFP4 on sm_103 for low-latency decode, with the attention heads and the experts
tensor-parallel over eight ranks and no attention data parallelism (tp8), and one-model
MTP-Eagle as a linear chain of five draft tokens. `../routing.py` routes only that layout here.

The fast paths were designed and tuned for this layout, and the target runs every one that has an
engine hook (`TP8_PATHS` in `modeling.py`). Their code lives in this directory: the plan
(`dispatch.py`), the attention adoption (`attention.py`, `attention_forward.py`), the layer
composition (`layers.py`), the speculative worker (`spec_worker.py`), the step-loop hooks
(`runtime.py`), and one kernel package per fast path under `kernels/`.

Other sm_103 layouts are served by other targets or by the built-in model: tep4 by
`../glm_5_2_nvfp4__sm_103__tep4`, and everything else by the built-in implementation.
