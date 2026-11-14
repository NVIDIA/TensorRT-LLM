# MiniMax-H3 NVFP4 scenario on SM107

The optimized pipeline for a pre-quantized NVFP4 checkpoint with per-tensor FP8 attention and Ulysses 1 or 4 on one SM107 node.
Select it through `pipeline_config: {visual_gen_scenario: require}`.
It also serves a ModelOpt `MIXED_PRECISION` checkpoint whose Linears are all static NVFP4 or static per-tensor FP8, at the same Ulysses sizes: the producers that write NVFP4 inputs are chosen per consumer Linear, and an FP8 Linear quantizes its own input.
Attention modules a checkpoint lists under `attention_precision` (see `../attention_precision.py`) run on BF16 operands through the family path with its BF16 Ulysses exchange.

See the [current setup and configuration guide](../../../../../../examples/visual_gen/minimax_h3_r200_stack.md) for the runtime environment, launch YAML, all 31 settings and tests.
This package owns the recipe's modeling, pipeline and Triton kernels; the family pipeline keeps its defaults when the scenario is `off`.

The first selected configuration is frozen for the process.
Use a separate process to compare settings.
