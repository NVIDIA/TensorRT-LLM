# Granite

This document shows how to build and run a [Granite 3.0](https://huggingface.co/collections/ibm-granite/granite-30-language-models-66fdb59bbb54785c3512114f) model in TensorRT-LLM.

The TensorRT LLM Granite implementation is based on the LLaMA model, with Mixture of Experts (MoE) enabled. The implementation can be found in [`llama/model.py`](../../../../tensorrt_llm/models/llama/model.py). See the LLaMA example [`examples/models/core/llama`](../llama) for details.

- [Granite 3.0](#Granite)
  - [Download model checkpoints](#download-model-checkpoints)
  - [Convert weights from HF Transformers to TensorRT LLM format](#Convert-weights-from-HF-Transformers-to-TensorRT-LLM-format)
  - [Build TensorRT engine](#build-tensorrt-engine)
  - [Run Engine](#run-engine)

## Download model checkpoints

First, download the HuggingFace BF16 checkpoints of Granite 3.0 model.

```bash
HF_MODEL="granite-3.0-8b-instruct" # or granite-3.0-3b-a800m-instruct
# clone the model we want to build
git clone https://huggingface.co/ibm-granite/${HF_MODEL} tmp/hf_checkpoints/${HF_MODEL}
```

## Convert weights from HF Transformers to TensorRT LLM format
Set environment variables and necessary directory:

```bash
PREC_RAW="bfloat16"
TP=1
mkdir -p tmp/trt_engines
```

### BF16
Convert the weights using the `convert_checkpoint.py` script:

```bash
ENGINE="${HF_MODEL}_${PREC_RAW}_tp${TP}"
export TRTLLM_DISABLE_UNIFIED_CONVERTER=1  # The current checkpoint conversion code requires legacy path
python3 ../llama/convert_checkpoint.py --model_dir tmp/hf_checkpoints/${HF_MODEL} \
                                       --output_dir tmp/tllm_checkpoints/${ENGINE} \
                                       --dtype ${PREC_RAW} \
                                       --tp_size ${TP} \
                                       --use_embedding_sharing


```
### FP8 PTQ
Notes:
- Currently quantize.py does not support Expert Parallelism (EP) mode yet. User should use `../llama/convert_checkpoint.py` and specify `--moe_ep_size 1` instead, if needed.
- TensorRT LLM uses static quantization methods, which is expected to be faster at runtime as compared to dynamic quantization methods. This comes at a cost of an offline calibration step during quantization. `batch_size` and `calib_size` can be adjusted to shorten the calibration time. Please refer to `../../../quantization/README.md` for explanation.

```bash
PREC_QUANT="fp8"
ENGINE="${HF_MODEL}_${PREC_QUANT}_tp${TP}"
python ../../../quantization/quantize.py --model_dir tmp/hf_checkpoints/${HF_MODEL} \
                                   --dtype ${PREC_RAW} \
                                   --qformat ${PREC_QUANT} \
                                   --kv_cache_dtype ${PREC_QUANT} \
                                   --output_dir tmp/tllm_checkpoints/${ENGINE} \
                                   --batch_size 1 \
                                   --calib_size 128 \
                                   --tp_size ${TP}

```

## Build TensorRT engine
```bash
# Enable fp8 context fmha to get further acceleration by setting `--use_fp8_context_fmha enable`
# Use --workers to enable parallel build
trtllm-build --checkpoint_dir ./tmp/tllm_checkpoints/${ENGINE} \
             --output_dir ./tmp/trt_engines/${ENGINE} \
             --gpt_attention_plugin ${PREC_RAW} \
             --gemm_plugin ${PREC_RAW} \
             --workers ${TP}
```

## Run Engine
Test your engine with the [run.py](../../../run.py) script:

```bash
mpirun -n ${TP} --allow-run-as-root python ../../../run.py --engine_dir ./tmp/trt_engines/${ENGINE} --tokenizer_dir tmp/hf_checkpoints/${HF_MODEL} --max_output_len 20 --input_text "The future of AI is"
```

For more usage examples see [`examples/models/core/llama/README.md`](../llama/README.md)
