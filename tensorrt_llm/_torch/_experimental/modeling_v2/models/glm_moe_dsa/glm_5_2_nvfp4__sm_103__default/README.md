# `glm_5_2_nvfp4__sm_103__default`

GLM-5.2 NVFP4 on sm_103, at the layouts `../routing.py` admits. The target picks the fast
paths of the layout when it is built (`target_paths` in `modeling.py`).

**Only TP8 is optimized.** The family's fast paths were designed and tuned for low-latency
decode with the attention heads and the experts tensor-parallel over eight ranks, and TP8 runs
all of them, including the routed-MoE kernels under `kernels/`.

TEP and DEP are not separately optimized. They run the fast paths in `../shared/` whose
contracts also cover them:

* TEP (attention tensor-parallel over four or eight ranks, experts expert-parallel over the same
  ranks) runs the shared paths that take its tensor-parallel size, in front of the built-in
  expert-parallel MoE.
* DEP (attention data parallelism, experts expert-parallel over the same ranks) runs only the
  rank-local ones: the RoPE and KV append, the full-indexer layers' query front, the draft loop,
  the decode step's prologue and the host page retention policy. Its decoder layers, MoE modules
  and draft layer stay the built-in classes.

No fast path was designed or tuned for TEP or DEP batch shapes. They are admitted because those
shared paths also improve decode there. TEP and DEP should be revisited in the future with fast
paths of their own, in particular for the expert-parallel MoE and, under DEP, the attention-DP
communication.
