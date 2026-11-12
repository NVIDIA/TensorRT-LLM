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

Set `max_num_tokens` to an integer from 1 through 8192; `null` is not supported.
Paged-context attention (cache reuse, chunked prefill)
requires `kv_cache_config.tokens_per_block: 32`.

This baseline does not enable speculative decoding, FP8 KV, disaggregated serving,
Dynamo, CUDA graphs or overlap scheduling. Those configurations require separate qualification.

## Baseline accuracy qualification

The named `TestModelingV2BeamFullModelNvfp4Sm103Tp4` gate starts the baseline with
`trtllm-serve` and `modeling_v2: require`, then runs the public evaluator's pinned
first 100 GSM8K questions. It requires at least 95 correct answers and no execution,
parsing or truncation failures. It uses BF16 KV, no drafter, and one request at a time.
The four-GB300 gate is listed in QA's scheduled functional suite,
`tests/integration/test_lists/qa/llm_function_core.txt`, with a 45-minute cap.
The evaluator has a 40-minute budget, reserving five minutes for cleanup and
result writing. QA infrastructure controls the daily/release schedule; this
test is not part of automatic L0 pre-merge or the nightly-release image build.
It never evaluates the full dataset in CI.

Stage an authorized target below `LLM_MODELS_ROOT`, set
`TRTLLM_TEST_TARGET_RELATIVE_PATH` to its relative directory, and stage the pinned
GSM8K files at `$LLM_MODELS_ROOT/datasets/openai/gsm8k` as described in the public
evaluator README. Missing inputs fail the selected run. With exactly four GB300s visible:

```bash
pytest tests/integration/defs/accuracy/modeling_v2/beam/full_model_nvfp4__sm_103__tp4/test_accuracy.py \
  -k test_gsm8k_first100 --timeout=2700
```

To qualify the same gate on a PR build without waiting for the QA schedule, use
the dedicated stage (subject to CI access and resource approval):

```text
/bot run --stage-list "GB300-4_GPUs-PyTorch-BeamAccuracy-OnDemand-1"
```

This stage selects only the named first-100 gate. Its test-db selector uses
`stage: pre_merge`, but the `OnDemand` stage is excluded from automatic CI
selection. The CI worker must have the staged inputs and environment variable
above; selecting the stage does not provision them. Require an actual passing
test result on the matching build, not a skipped stage or command acknowledgement.
Selective stage runs do not update the overall GitHub CI check status.

The test writes its server configuration, logs, per-question records and summary
under pytest's temporary directory. The timeout and incomplete runs are failures,
not accuracy passes. A gate definition alone does not qualify this baseline.
