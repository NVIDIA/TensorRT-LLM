# Inferact DSpark fusions

These opt-in specializations target the stock MLA DSpark drafter on SM100,
TP16 with MoE TP4/EP4, based on revision adab54f2.
All flags default to 0 and accept only 0 or 1.

| Environment variable | Operation |
| --- | --- |
| `TRTLLM_K3_INFERACT_BOUNDARY` | Fuse integer metadata preparation for B1/K7. |
| `TRTLLM_K3_INFERACT_CACHE_WRITE` | Fuse normalization/YaRN and direct paged draft-cache writes for B1/K7. |
| `TRTLLM_K3_INFERACT_CONTEXT` | Fuse normalization/RoPE across five draft context layers, for 1–128 context rows. |

Flags are independent. Set all three to 1 to enable the combined candidate.
Unsupported inputs retain the stock route. Context rows are not scheduler batch sizes.
Warm eligible context shapes before CUDA graph capture; compilation during capture
is rejected. The context projection GEMM and frequency lookup remain separate.

This branch contains the original B1 metadata/cache implementation, not the later
B1–B8 Lyris extension. The context kernel is carried over from the combined
experiment; its runtime integration uses a flag instead of experiment audit hooks.
No Slurm paths, checkpoints, profiler hooks, or experiment output writers are needed.

Historical experiment results do not validate this repackaged integration.
GPU correctness, capture coverage, and performance must be rechecked before promotion.
