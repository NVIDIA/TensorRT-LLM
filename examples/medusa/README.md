# Medusa Decoding

This document shows how to build and run a model using Medusa decoding([`Github`](https://github.com/FasterDecoding/Medusa), [`BLOG`](https://sites.google.com/view/medusa-llm)) in TensorRT LLM on single GPU, single node multiple GPU.

## Overview
Different from other models, Medusa decoding needs a base model and Medusa heads. The TensorRT LLM Medusa Decoding implementation can be found in [tensorrt_llm/models/medusa/model.py](../../tensorrt_llm/models/medusa/model.py). The implementation adds Medusa heads to a base model.

For more info about Medusa visit [speculative decoding documentation](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html).

## Support Matrix
  * GPU Compute Capability >= 8.0 (Ampere or newer)
  * FP16
  * BF16
  * FP8 (base model)
  * PAGED_KV_CACHE
  * Tensor Parallel

## Usage
The TensorRT LLM Medusa example code is located in [`examples/medusa`](./). There is one [`convert_checkpoint.py`](./convert_checkpoint.py) file to convert and build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run models with Medusa decoding support.
In this example, we demonstrate the usage of two models:
1. The Vucuna 7B model from Hugging Face [`FasterDecoding/medusa-vicuna-7b-v1.3`](https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3) with its Medusa heads [`medusa-vicuna-7b-v1.3`](https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3).
2. The quantized checkpoint [`nvidia/Llama-3.1-8B-Medusa-FP8`](https://huggingface.co/nvidia/Llama-3.1-8B-Medusa-FP8) on Hugging Face by [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (ModelOpt). This model is based on [Llama-3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) and enhanced with Medusa heads, with both the base model (except lm_head) and Medusa heads already quantized in FP8.

### Build TensorRT engine(s)
Get the weights by downloading base model [`vicuna-7b-v1.3`](https://huggingface.co/lmsys/vicuna-7b-v1.3) and Medusa Heads [`medusa-vicuna-7b-v1.3`](https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3) from HF.

```
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3
```

We use `convert_checkpoint.py` script to convert the model for Medusa decoding into TensorRT LLM checkpoint format.
We could use `--num_medusa_heads` to set the number of medusa heads that we want to use. If not, `num_medusa_heads` will be set according to the `medusa_num_heads` from medusa weights' `config.json`.

Here is the example:
```bash
# Convert and Build Medusa decoding support for vicuna-7b-v1.3
python convert_checkpoint.py --model_dir ./vicuna-7b-v1.3 \
                            --medusa_model_dir medusa-vicuna-7b-v1.3 \
                            --output_dir ./tllm_checkpoint_1gpu_medusa \
                            --dtype float16 \
                            --num_medusa_heads 4

# Note: Increasing the batch size may have a negative impact on performance
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_medusa \
             --output_dir ./tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
             --gemm_plugin float16 \
             --speculative_decoding_mode medusa \
             --max_batch_size 4

# Convert and Build Medusa decoding support for vicuna-13b-v1.3 with 4-way tensor parallelism.
python convert_checkpoint.py --model_dir ./vicuna-7b-v1.3 \
                            --medusa_model_dir medusa-vicuna-7b-v1.3 \
                            --output_dir ./tllm_checkpoint_1gpu_medusa \
                            --dtype float16 \
                            --num_medusa_heads 4 \
                            --tp_size 4 \
                            --workers 4

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_medusa \
             --output_dir ./tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
             --gemm_plugin float16 \
             --speculative_decoding_mode medusa \
             --max_batch_size 4

# Convert and Build Llama-3.1-8B-Medusa by ModelOpt
python convert_checkpoint.py --model_dir ./llama3.1-medusa-8b-hf_v0.1 \
                             --output_dir ./tllm_checkpoint_1gpu_modelopt_llama_medusa \
                             --dtype float16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_modelopt_llama_medusa \
             --output_dir ./tmp/modelopt/llama-8B-medusa/trt_engines/1-gpu/ \
             --gemm_plugin float16 \
             --speculative_decoding_mode medusa \
             --max_batch_size 4


# Convert and Build Llama-3.1-70B-Medusa by ModelOpt with 2-way tensor parallelism.
python convert_checkpoint.py --model_dir ./llama-3.1-70b-medusa_vfp8-fp8-fp8 \
                             --output_dir ./tllm_checkpoint_2gpu_modelopt_llama_medusa_70b \
                             --dtype float16
                             --tp_size 2
                             --workers 2

trtllm-build --checkpoint_dir ./tllm_checkpoint_2gpu_modelopt_llama_medusa_70b \
             --output_dir ./tmp/modelopt/llama-70B-medusa/trt_engines/2-gpu/ \
             --gemm_plugin float16 \
             --speculative_decoding_mode medusa \
             --max_batch_size 4


```

### FP8 Post-Training Quantization for Base Model
The example below quantizes the base model to FP8, while keeping the weight of the medusa head non-quantize.
```bash
# Quantize base model into FP8 and export trtllm checkpoint
python ../quantization/quantize.py --model_dir /path/to/base-model-hf/ \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ./tllm_checkpoint_1gpu_base_model_fp8_medusa_fp16 \
                                   --calib_size 512 \
                                   --tp_size 1 \
                                   --medusa_model_dir /path/to/medusa_head/ \
                                   --num_medusa_heads 4

# Build trtllm engines from the trtllm checkpoint
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_base_model_fp8_medusa_fp16 \
         --output_dir ./trt_engine_1gpu_base_model_fp8_medusa_fp16 \
         --gemm_plugin float16 \
         --gpt_attention_plugin float16 \
         --speculative_decoding_mode medusa \
         --max_batch_size 4
```

### Run
To run a TensorRT LLM model with Medusa decoding support, we can use `../run.py` script, with an additional argument `--medusa_choices`.
The `--medusa_choices` is of type `list[list[int]]`.

Medusa decoding is supported by Python runtime and C++ runtime with inflight-batching. C++ runtime is recommended for performance.
For Python runtime use `--use_py_session` flag to `run.py`.

Medusa decoding only supporting greedy decoding, indicated by `temperature=1.0` argument. The output is equivalent to the base model inference with `--temperature 0.0` (equivalent to `--temperature 1.0 --top-k 1`).

```bash
# Medusa decoding using vicuna-7b-v1.3 model with 1 GPU
python ../run.py --engine_dir ./tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
                 --tokenizer_dir ./vicuna-7b-v1.3/ \
                 --max_output_len=100 \
                 --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                 --temperature 1.0 \
                 --input_text "Once upon"

# Medusa decoding using vicuna-13b-v1.3 with 4 GPUs
mpirun -np 4 --allow-run-as-root --oversubscribe \
    python ../run.py --engine_dir ./tmp/medusa/13B/trt_engines/fp16/4-gpu/ \
                     --tokenizer_dir ./vicuna-13b-v1.3/ \
                     --max_output_len=100 \
                     --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                     --temperature 1.0 \
                     --input_text "Once upon"

# Medusa decoding using Llama-3.1-8B-Medusa by ModelOpt with 1 GPU
python ../run.py --engine_dir ./tmp/modelopt/llama-8B-medusa/trt_engines/1-gpu/ \
                 --tokenizer_dir ./llama3.1-medusa-8b-hf_v0.1 \
                 --max_output_len=100 \
                 --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [1, 6], [0, 7, 0]]" \
                 --temperature 1.0 \
                 --input_text "Once upon"

# Medusa decoding using Llama-3.1-70B-Medusa by ModelOpt with 2 GPUs
mpirun -np 2 --allow-run-as-root --oversubscribe \
    python ../run.py --engine_dir ./tmp/modelopt/llama-70B-medusa/trt_engines/2-gpu/ \
                     --tokenizer_dir ./llama-3.1-70b-medusa_vfp8-fp8-fp8 \
                     --max_output_len=100 \
                     --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                     --temperature 1.0 \
                     --input_text "Once upon"

```

And you will see output like this if run successfully:
```text
......
Input [Text 0]: "<s> Once upon"
Output [Text 0 Beam 0]: "a time, there was a young girl who loved to read. She would spend hours in the library, devouring books of all genres. She had a special love for fairy tales, and would often dream of living in a magical world where she could meet princes and princesses, and have adventures with talking animals.
One day, while she was reading a book, she came across a passage that spoke to her heart. It said, "You are the author of"
```

### Summarization using Medusa decoding

```bash
# Medusa decoding using vicuna-7b-v1.3 model with 1 GPU
python ../summarize.py --engine_dir ./tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
                       --hf_model_dir ./vicuna-7b-v1.3/ \
                       --tokenizer_dir ./vicuna-7b-v1.3/ \
                       --test_trt_llm \
                       --data_type fp16 \
                       --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                       --use_py_session \
                       --temperature 1.0 \
                       --batch_size 1

# Medusa decoding using vicuna-13b-v1.3 with 4 GPUs
mpirun -np 4 --allow-run-as-root --oversubscribe \
    python ../summarize.py --engine_dir ./tmp/medusa/13B/trt_engines/fp16/4-gpu/ \
                           --hf_model_dir ./vicuna-13b-v1.3/ \
                           --tokenizer_dir ./vicuna-13b-v1.3/ \
                           --test_trt_llm \
                           --data_type fp16 \
                           --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                           --use_py_session \
                           --temperature 1.0 \
                           --batch_size 1

# Medusa decoding using Llama-3.1-8B-Medusa by ModelOpt with 1 GPU
python ../summarize.py --engine_dir ./tmp/modelopt/llama-8B-medusa/trt_engines/1-gpu/ \
                       --hf_model_dir ./llama3.1-medusa-8b-hf_v0.1 \
                       --tokenizer_dir ./llama3.1-medusa-8b-hf_v0.1 \
                       --test_trt_llm \
                       --data_type fp16 \
                       --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [1, 6], [0, 7, 0]]" \
                       --use_py_session \
                       --temperature 1.0 \
                       --batch_size 1

# Medusa decoding using Llama-3.1-70B-Medusa by ModelOpt with 2 GPUs
mpirun -np 2 --allow-run-as-root --oversubscribe \
    python ../summarize.py --engine_dir ./tmp/modelopt/llama-70B-medusa/trt_engines/2-gpu/ \
                          --hf_model_dir ./llama-3.1-70b-medusa_vfp8-fp8-fp8 \
                          --tokenizer_dir ./llama-3.1-70b-medusa_vfp8-fp8-fp8 \
                          --test_trt_llm \
                          --data_type fp16 \
                          --medusa_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
                          --use_py_session \
                          --temperature 1.0 \
                          --batch_size 1
```

### Medusa with Qwen2

To use Medusa with Qwen2 models, specify `--model_type qwen2` to `convert_checkpoint.py`. You have to provide a Qwen2 model checkpoint and the medusa heads. After TRT-LLM checkpoint is generated, trllm-build and `../run.py` use the same arguments as for LLaMA models.
