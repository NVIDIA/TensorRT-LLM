# Arctic

This document shows how to build and run a [Arctic](https://huggingface.co/Snowflake/snowflake-arctic-instruct) model in TensorRT-LLM.

The TensorRT LLM Arctic implementation is based on the LLaMA model, with Mixture of Experts (MoE) enabled. The implementation can
be found in [llama/model.py](../../../../tensorrt_llm/models/llama/model.py).
See the LLaMA example [`examples/models/core/llama`](../../../llama) for details.

- [Arctic](#arctic)
  - [Download model checkpoints](#download-model-checkpoints)
  - [TensorRT LLM workflow](#tensorrt-llm-workflow)
    - [Apply FP8 PTQ](#apply-fp8-ptq)
    - [Build TensorRT engine](#build-tensorrt-engine)
    - [Run Engine](#run-engine)
    - [OOTB](#ootb)

## Download model checkpoints

First, download the HuggingFace BF16 checkpoints of Arctic model.

**CAVEAT: this model is a pretty large Mixture-of-Experts (MoE) model, which has nearly 500B parameters and requires around 900GB disk space for storage. Please make sure you have enough space before proceeding.**

```bash
HF_MODEL="arctic"
git clone https://huggingface.co/Snowflake/snowflake-arctic-instruct tmp/hf_checkpoints/${HF_MODEL}

```

## TensorRT LLM workflow
Next, we use the general quantization script `quantize.py` to convert the checkpoints in FP8, and build the model with `trtllm-build` on multi-GPUs. In the example below, we use Tensor Parallelism (TP) across 8 GPUs.

**Note: for such large model, it is deemed necessary to apply Post-Training Quantization (PTQ) methods on the model weights to deploy it on a cluster node, e.g., 8xH100 GPUs. In this example, we demonstrate the FP8 quantization workflow, which is supported on Hopper-and-next GPU architectures. For instructions of other PTQ methods other than FP8, please refer to the LLaMA or Mixtral examples.**


Set environment variables and necessary directory:

```bash
PREC_RAW="bfloat16"
PREC_QUANT="fp8"
TP=8
ENGINE="${HF_MODEL}_${PREC_QUANT}_tp${TP}"

mkdir -p tmp/trt_engines
```

### Apply FP8 PTQ

Notes:
- currently quantize.py does not support for Expert Parallelism (EP) mode yet. User should use `../llama/convert_checkpoint.py` and specify `--moe_ep_size 1` instead, if needed.
- TensorRT LLM uses static quantization methods, which is expected to be faster at runtime as compared to dynamic quantization methods. This comes at a cost of an offline calibration step during quantization. `batch_size` and `calib_size` can be adjusted to shorten the calibration time. Please refer to ../quantization/README.md for explanation.
- **due to the large model size and the calibration step (which has to load the HuggingFace model and run forward passes), it is likely that you will need more number of GPUs during quantization step than the number of GPUs for engine building and final deployment. For example, using 16xH100 or 8xH200 for quantization & 8xH100 for deployment.**

```bash
python ../../../quantization/quantize.py --model_dir tmp/hf_checkpoints/${HF_MODEL} \
                                   --dtype ${PREC_RAW} \
                                   --qformat ${PREC_QUANT} \
                                   --kv_cache_dtype ${PREC_QUANT} \
                                   --output_dir tmp/tllm_checkpoints/${ENGINE} \
                                   --batch_size 1 \
                                   --calib_size 128 \
                                   --tp_size ${TP} |& tee tmp/trt_engines/${ENGINE}_quantize.log

```

### Build TensorRT engine
```bash
# Enable fp8 context fmha to get further acceleration by setting `--use_fp8_context_fmha enable`
# Use --workers to enable parallel build
trtllm-build --checkpoint_dir ./tmp/tllm_checkpoints/${ENGINE} \
             --output_dir ./tmp/trt_engines/${ENGINE} \
             --gpt_attention_plugin ${PREC_RAW} \
             --gemm_plugin ${PREC_RAW} \
             --workers ${TP} |& tee tmp/trt_engines/${ENGINE}_build.log
```

### Run Engine
Test your engine with the [run.py](../run.py) script:

```bash
mpirun -n ${TP} --allow-run-as-root python ../../../run.py --engine_dir ./tmp/trt_engines/${ENGINE} --tokenizer_dir tmp/hf_checkpoints/${HF_MODEL} --max_output_len 20 --input_text "The future of AI is" |& tee tmp/trt_engines/${ENGINE}_run.log
```

For more examples see [`examples/models/core/llama/README.md`](../../../llama/README.md)


### OOTB

Arctic supports OOTB operation without the plugin, however this comes at a significant performance cost. Users should prefer using the plugin path whenever possible.
