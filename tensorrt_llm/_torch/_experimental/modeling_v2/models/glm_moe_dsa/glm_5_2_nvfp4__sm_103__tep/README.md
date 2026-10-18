# `glm_5_2_nvfp4__sm_103__tep`

GLM-5.2 NVFP4 with the attention heads tensor-parallel over four or eight ranks and the
experts expert-parallel over the same ranks.

**This target is not separately optimized.** The family's fast paths were designed and tuned
for low-latency decode on the TP8 target, `glm_5_2_nvfp4__sm_103__tp8`. This target runs the
fast paths in `../shared/` whose contracts also cover its tensor-parallel size, in front of the
built-in expert-parallel MoE. No fast path was designed or tuned for TEP batch shapes, and the
routed experts run the built-in MoE.

It is in the family because those shared paths also improve decode at this topology. TEP should
be revisited in the future with fast paths of its own, in particular for the expert-parallel
MoE and the larger decode batches this topology serves.
