# Beam baseline serving

The 52-layer Beam target uses ModelingV2/StairCase with NVFP4 experts, attention TP4,
MoE TP4/EP1 and BF16 KV on four GB300 GPUs. Supply an authorized checkpoint separately.
Execution is eager, with overlap scheduling disabled.

Create `beam-baseline.yaml`:

```yaml
modeling_v2: require
checkpoint_format: beam_nvfp4
custom_tokenizer: tensorrt_llm._torch.models.checkpoints.beam.tokenizer.BeamFastTokenizer
reasoning_parser: beam_harmony_text
tensor_parallel_size: 4
max_batch_size: 1
max_num_tokens: 8192
cuda_graph_config: null
disable_overlap_scheduler: true
```

```bash
trtllm-serve /path/to/checkpoint --config beam-baseline.yaml
```

The same settings can be passed to `LLM()`; use `checkpoint_format="beam_nvfp4"`
and `modeling_v2="require"`. Use the concrete `BeamFastTokenizer` for prompts.
The public `examples/llm-api/eval_gsm8k.py` evaluator can score an existing compatible
endpoint; see its adjacent `gsm8k_eval/README.md` for arguments and resume behavior.

This baseline does not enable speculative decoding, FP8 KV, disaggregated serving,
Dynamo, CUDA graphs or overlap scheduling. Those configurations require separate qualification.
