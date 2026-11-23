# `glm_5_2_nvfp4__sm_103__tp8`

GLM-5.2 NVFP4 on sm_103 for low-latency decode, with the attention heads and the experts
tensor-parallel over eight ranks and no attention data parallelism (tp8), and one-model
MTP-Eagle as a linear chain of five draft tokens. `../routing.py` routes only that layout here.

The fast paths were designed and tuned for this layout, and the target runs every one that has an
engine hook (`TP8_PATHS` in `modeling.py`). Their code lives in this directory: the plan
(`dispatch.py`), the attention adoption (`attention.py`, `attention_forward.py`), the layer
composition (`layers.py`), the speculative worker (`spec_worker.py`), the step-loop hooks
(`runtime.py`), and one kernel package per fast path under `kernels/`. The CUDA kernels and
their torch ops are in `csrc/`, built into `libth_modeling_v2_glm_moe_dsa.so`; `__init__.py`
loads them as `native_ops`, and `fake_ops.py` fakes every op.

GLM-5.3-NVFP4 routes here too. Its modelopt export is MIXED_PRECISION, but its MoE layers are
quantized as GLM-5.2-NVFP4's are (NVFP4 routed experts; attention, shared experts and the MTP layer
unquantized); only the dense MLPs of layers 0-2 differ (FP8), and the dense MLP fast path leaves
those to the built-in MLP. On such a checkpoint the built-in decoder layers set `is_nvfp4` false,
so the paths gated on it (`post_moe_hop` and its routed-input fusion) stay on the native path.

Other sm_103 layouts are served by other targets or by the built-in model: tep4 by
`../glm_5_2_nvfp4__sm_103__tep4`, and everything else by the built-in implementation.
