# Step-5-Preview on Grace Blackwell

The experimental `modeling_v2` implementation loads the StepFun Step-5-Preview
vision-language checkpoint (`Step5ForConditionalGeneration`, `step5_vl`) with
ModelOpt NVFP4 routed-expert weights. Attention, shared/dense experts, the final
three routed-expert layers, and the vision tower retain their checkpoint dtypes.
The target is limited to SM103 (GB300/B300).

## Serving text

Set `STEP5_MODEL` to the local ModelOpt NVFP4 checkpoint directory. It must include
the text and vision configurations, tokenizer assets, and indexed safetensors.
No remote Python code is required.

```bash
trtllm-serve "$STEP5_MODEL" --tp_size 4 --ep_size 4 \
  --config examples/models/core/step5/adp4.yml
```

The configuration enables attention DP4 and expert parallelism across four GPUs,
chunked prefill, block reuse, padded CUDA graphs, and a 9216-token sequence budget
for an 8192-token input plus 1024-token output. `modeling_v2: require` prevents
silently falling back to another implementation. Choose the request's output
length separately, for example `max_tokens: 1024` in an OpenAI request.

For attention TP4 with EP4 (TEP4), set `enable_attention_dp: false`. For TP4 with
MoE tensor parallelism, also use `--ep_size 1`. All layouts require one NVLink
domain. The router additionally admits ADP2/EP2 instances.

`kv_cache_config.dtype` can be `auto` (BF16), `fp8`, or `nvfp4`. The V2 cache
manager owns both attention K/V and the sparse indexer's token and region state;
keep `use_kv_cache_manager_v2: true`. The implementation pads 192-wide attention
heads to 256 and compensates the softmax scale. This uses the existing native
attention geometry without requiring a new D192 native kernel.

## Images

The registered input processor accepts images through the standard OpenAI
`image_url` content format and LLM API `multi_modal_data={"image": ...}`. Use the
checkpoint's chat template so each image has an `<im_patch>` placeholder. The
processor creates the overview and crops, expands image spans, and the vision
tower replaces their token embeddings. Video and audio inputs are unsupported.

Image preprocessing and embedding insertion support chunked prefill. Text-only
TP4/TEP4 validation and multimodal ADP validation should be tracked separately.

## Disaggregated serving

Use the standard [disaggregated serving examples](../../../disaggregated/README.md)
with compatible Step-5 context and generation instances. Both roles require:

```yaml
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
kv_cache_config:
  use_kv_cache_manager_v2: true
```

The Python NIXL transceiver moves every V2 pool, including sliding-window K/V,
full-attention K/V, sparse indexer state, and NVFP4 block scales. Other transceiver
routes are rejected before allocation. Pool-transfer unit tests cover ADP2,
ADP4, TEP4-to-ADP4, and ADP2-to-ADP4 layouts; these tests do not substitute for
whole-model disaggregated generation or image smoke tests.

## Validation and limits

CPU CI covers configuration, weight sharding, image preprocessing, tokenizer
fidelity, request EOS handling, and deterministic MoE selection. GB300 unit tests
cover the operators and real cache managers. Whole-model GSM8K tests are in
`tests/integration/defs/accuracy/modeling_v2/test_step5.py`; the multi-GPU CI list
selects the core layouts, cache dtypes, and chunked-prefill cases.

For checkpoint-dependent unit tests, set `STEP5_PREVIEW_CHECKPOINT`, or stage the
checkpoint as `$LLM_MODELS_ROOT/Step-5-Preview-NVFP4`. Large cache-address and long
context stress tests use `pytest -m large` and are excluded from the regular
single-GPU list.

The integrated implementation still requires a fresh GB300 end-to-end validation
before release, including the complete ADP4 cache-dtype matrix, aggregate and
disaggregated image smoke tests, and 8K/1K performance measurement. Results from
independent development branches are not measurements of this integrated tree.
Speculative decoding/MTP, pipeline/context parallelism, guided decoding, and
GPUs other than SM103 are outside this target's support.
