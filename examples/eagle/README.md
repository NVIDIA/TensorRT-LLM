# **MODEL IS NOT FULLY SUPPORTED YET! DO NOT USE IT.**

# EAGLE speculative Decoding

This document shows how to build and run a model using EAGLE decoding ([`Github`](https://github.com/SafeAILab/EAGLE/tree/main), [`BLOG`](https://sites.google.com/view/eagle-llm)) in TensorRT-LLM on a single node with one GPU or more.

## Overview
Different from other models, EAGLE decoding needs a base model and EAGLE model.

The TensorRT-LLM EAGLE Decoding implementation can be found in [tensorrt_llm/models/eagle/model.py](../../tensorrt_llm/models/eagle/model.py), which actually adds Eagle draft network to a base model.

<!---
For more info about EAGLE visit [speculative decoding documentation](../../docs/source/speculative_decoding.md).
-->

## Support Matrix
  * GPU Compute Capability >= 8.0 (Ampere or newer)
  * FP16
  * BF16
  * PAGED_KV_CACHE
  * Tensor Parallel

This example focuses on adding EAGLE to LLaMA base model. With some modifications EAGLE can be added to the other base models as well.

## Usage
The TensorRT-LLM EAGLE example code is located in [`examples/eagle`](./). There is one [`convert_checkpoint.py`](./convert_checkpoint.py) file to convert and build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run models with EAGLE decoding support.
In our example, we use the model from HuggingFace [`yuhuili/EAGLE-Vicuna-7B-v1.3`](https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3), which is a LLAMA based model.

### Build TensorRT engine(s)
Get the weights by downloading the base model [`vicuna-7b-v1.3`](https://huggingface.co/lmsys/vicuna-7b-v1.3) and the EAGLE draft model [`EAGLE-Vicuna-7B-v1.3`](https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3) from HF.

```
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3
```


Here is the example:
```bash
python convert_checkpoint.py --model_dir ./vicuna-7b-v1.3 \
                            --eagle_model_dir EAGLE-Vicuna-7B-v1.3 \
                            --output_dir ./tllm_checkpoint_1gpu_eagle \
                            --dtype float16 \
                            --max_draft_len 63 \
                            --num_eagle_layers 4

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_eagle \
             --output_dir ./tmp/eagle/7B/trt_engines/fp16/1-gpu/ \
             --gemm_plugin float16 \
             --speculative_decoding_mode eagle \
             --max_batch_size 4
```

### Run

### Summarization using EAGLE decoding
