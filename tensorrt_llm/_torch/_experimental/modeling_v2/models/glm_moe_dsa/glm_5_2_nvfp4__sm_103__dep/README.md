# `glm_5_2_nvfp4__sm_103__dep`

GLM-5.2 NVFP4 with attention data parallelism over eight or sixteen ranks and the experts
expert-parallel over the same ranks.

**This target is not separately optimized.** The family's fast paths were designed and tuned
for low-latency decode on the TP8 target, `glm_5_2_nvfp4__sm_103__tp8`. Under attention data
parallelism only the rank-local ones apply, and this target runs those from `../shared/`: the
RoPE and KV append, the full-indexer layers' query front, the draft loop, the decode step's
prologue and the host page retention policy. Its decoder layers, MoE modules and draft layer
stay the built-in classes. No fast path was designed or tuned for DEP batch shapes.

It is in the family because those rank-local paths also improve decode at this topology. DEP
should be revisited in the future with fast paths of its own, in particular for the
attention-DP communication and the expert-parallel MoE.
