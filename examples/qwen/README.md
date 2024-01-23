# Qwen

This document shows how to build and run a Qwen model in TensorRT-LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

## Overview

The TensorRT-LLM Qwen implementation can be found in [model.py](../../tensorrt_llm/models/qwen/model.py). The TensorRT-LLM Qwen example code is located in [`examples/qwen`](./). There is one main file:

* [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Qwen model.

In addition, there are two shared files in the parent folder [`examples`](../) for inference and evaluation:

* [`../run.py`](../run.py) to run the inference on an input text;
* [`../summarize.py`](../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset.

## Support Matrix
|    Model Name    | FP16  | FMHA  |  WO   |  AWQ   | GPTQ  |  SQ   |  TP   |  ST   | C++ Runtime | benchmark |  IFB  |   Arch  |
| :--------------: | :---: | :---: | :---: | :---:  | :---: | :---: | :---: |:---: | :---------: | :-------: | :---: |  :---:  |
|   Qwen-7B-Chat   |   Y   |   Y   |   Y   |   Y    |   Y   |   Y   |   Y   |   Y   |      Y      |     Y     |   Y   | Ampere+ |
|  Qwen-14B-Chat   |   Y   |   Y   |   Y   |   Y*   |   Y   |   Y   |   Y   |   Y   |      Y      |     Y     |   Y   | Ampere+ |
|  Qwen-72B-Chat   |   Y   |   Y   |   Y   |   -   |   Y   |   Y   |   Y   |   Y   |      Y      |     Y     |   Y   | Ampere+ |

*Please note that Qwen-14B-Chat model supports AWQ only with single GPU.
* Model Name: the name of the model, the same as the name on HuggingFace
* FMHA: Fused MultiHead Attention (see introduction below)
* WO: Weight Only Quantization (int8 / int4)
* AWQ: Activation Aware Weight Quantization (int4)
* GPTQ: Generative Pretrained Transformer Quantization (int4)
* SQ: Smooth Quantization
* TP: Tensor Parallel
* PP: Pipeline Parallel
* ST: Strongly Typed
* IFB: In-flight Batching (see introduction below)

## Usage

The TensorRT-LLM Qwen example code locates at [examples/qwen](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Need to prepare the HF Qwen checkpoint first by following the guides here [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) or [Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat)

Create a `tmp/Qwen` directory to store the weights downloaded from huaggingface.
```bash
mkdir -p ./tmp/Qwen
```

Store Qwen-7B-Chat or Qwen-14B-Chat separately.
- for Qwen-7B-Chat
```bash
mv Qwen-7B-Chat ./tmp/Qwen/7B
```
- for Qwen-14B-Chat
```
mv Qwen-14B-Chat ./tmp/Qwen/14B
```

TensorRT-LLM Qwen builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

Normally `build.py` only requires single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallelly building to make the engine building process faster by adding `--parallel_build` argument. Please note that currently `parallel_build` feature only supports single node.

Here're some examples:

```bash
# Build a single-GPU float16 engine from HF weights.
# use_gpt_attention_plugin is necessary in Qwen.
# Try use_gemm_plugin to prevent accuracy issue.
# It is recommend to use --remove_input_padding along with --use_gpt_attention_plugin for better performance

# Build the Qwen 7B model using a single GPU and FP16.
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen/7B/trt_engines/fp16/1-gpu/

# Build the Qwen 7B model using a single GPU and BF16.
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --use_gemm_plugin bfloat16 \
                --output_dir ./tmp/Qwen/7B/trt_engines/bf16/1-gpu/

# Build the Qwen 7B model using a single GPU and apply INT8 weight-only quantization.
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int8 \
                --output_dir ./tmp/Qwen/7B/trt_engines/int8_weight_only/1-gpu/

# Build the Qwen 7B model using a single GPU and apply INT4 weight-only quantization.
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4 \
                --output_dir ./tmp/Qwen/7B/trt_engines/int4_weight_only/1-gpu/

# Build Qwen 7B using 2-way tensor parallelism.
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen/7B/trt_engines/fp16/2-gpu/ \
                --world_size 2 \
                --tp_size 2

# Build Qwen 7B using 2-way tensor parallelism and 2-way pipeline parallelism.
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen/7B/trt_engines/fp16/2-gpu/ \
                --world_size 4 \
                --tp_size 2 \
                --pp_size 2

# Build Qwen 14B using 2-way tensor parallelism.
python build.py --hf_model_dir ./tmp/Qwen/14B \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen/14B/trt_engines/fp16/2-gpu/ \
                --world_size 2 \
                --tp_size 2

# Build Qwen 72B using 8-way tensor parallelism.
python build.py --hf_model_dir ./tmp/Qwen/72B \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen/72B/trt_engines/fp16/8-gpu/ \
                --world_size 8 \
                --tp_size 8
```
**Demo output of engine building:**
```python
python3 build.py --hf_model_dir /llm-models/Qwen-7B-Chat/ --output_dir /engine_qwen
```
```
[11/09/2023-00:57:06] [TRT-LLM] [I] Serially build TensorRT engines.
[11/09/2023-00:57:06] [TRT] [I] [MemUsageChange] Init CUDA: CPU +14, GPU +0, now: CPU 118, GPU 427 (MiB)
[11/09/2023-00:57:08] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1974, GPU +350, now: CPU 2227, GPU 777 (MiB)
[11/09/2023-00:57:08] [TRT-LLM] [W] Invalid timing cache, using freshly created one
[11/09/2023-00:57:14] [TRT-LLM] [I] Loading HF QWen ... from /llm-models/Qwen-7B-Chat/
......
[11/09/2023-01:01:34] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 47322 MiB
[11/09/2023-01:01:34] [TRT-LLM] [I] Total time of building qwen_float16_tp1_rank0.engine: 00:03:44
[11/09/2023-01:01:34] [TRT-LLM] [I] Config saved to /engine_qwen/config.json.
[11/09/2023-01:01:34] [TRT-LLM] [I] Serializing engine to /engine_qwen/qwen_float16_tp1_rank0.engine...
[11/09/2023-01:01:49] [TRT-LLM] [I] Engine serialized. Total time: 00:00:14
[11/09/2023-01:01:49] [TRT-LLM] [I] Timing cache serialized to /engine_qwen/model.cache
[11/09/2023-01:01:50] [TRT-LLM] [I] Total time of building all 1 engines: 00:04:43
```


#### INT8 weight only + INT8 KV cache
For INT8 KV cache, [`hf_qwen_convert.py`](./hf_qwen_convert.py) features a
`--calibrate-kv-cache, -kv` option. Setting `-kv` will calibrate the model,
and then export the scaling factors needed for INT8 KV cache inference.


Example:

```bash
python3 hf_qwen_convert.py \
    -i ./tmp/Qwen/7B/ \
    -o ./tmp/Qwen/7B/int8_kv_cache/ \
    --calibrate-kv-cache -t float16
```

[`build.py`](./build.py) add new options for the support of INT8 KV cache.

`--int8_kv_cache` is the command-line option to enable INT8 KV cache.

In addition, it could be combined with INT8 weight-only quantization, as follows:

Examples of INT8 weight-only quantization + INT8 KV cache

```bash
# Build model with both INT8 weight-only and INT8 KV cache enabled
python build.py --ft_dir_path ./tmp/Qwen/7B/int8_kv_cache/1-gpu/ \
                --dtype float16 \
                --hf_model_dir ./tmp/Qwen/7B \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen/7B/trt_engines/int8_kv_cache_weight_only/1-gpu \
                --int8_kv_cache \
                --use_weight_only
```

- run
```bash
python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/int8_kv_cache_weight_only/1-gpu
```

Test with `../summarize.py`:


- validate huggingface
```bash
python3 ../summarize.py --test_hf \
                        --tokenizer_dir ./tmp/Qwen/7B \
                        --hf_model_dir ./tmp/Qwen/7B \
                        --max_input_length 2048 \
                        --output_len 2048
```

- validate trt-llm
```bash
python3 ../summarize.py --test_trt_llm \
                        --tokenizer_dir ./tmp/Qwen/7B \
                        --engine_dir ./tmp/Qwen/7B/trt_engines/int8_kv_cache_weight_only/1-gpu \
                        --max_input_length 2048 \
                        --output_len 2048
```

#### SmoothQuant

The smoothquant supports both Qwen v1 and Qwen v2. Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT-LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

Example:
```bash
python3 hf_qwen_convert.py -i ./tmp/Qwen/7B -o ./tmp/Qwen/7B/sq0.5/ -sq 0.5 --tensor-parallelism 1 --storage-type float16
```

[`build.py`](./build.py) add new options for the support of INT8 inference of SmoothQuant models.

`--use_smooth_quant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:

```bash
# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
python3 build.py --ft_dir_path=./tmp/Qwen/7B/sq0.5/1-gpu/ \
                 --use_gpt_attention_plugin float16 \
                 --remove_input_padding \
                 --enable_context_fmha \
                 --use_smooth_quant \
                 --per_token \
                 --per_channel \
                 --hf_model_dir ./tmp/Qwen/7B \
                 --output_dir ./tmp/Qwen/7B/trt_engines/sq0.5/1-gpu/
```

- run
```bash
python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/sq0.5/1-gpu/
```

- summarize
```bash
python ../summarize.py --test_trt_llm \
                       --tokenizer_dir ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir=./tmp/Qwen/7B/trt_engines/sq0.5/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048
```
#### INT4-GPTQ
To run the GPTQ Qwen example, the following steps are required:
1. Install auto-gptq module:
```bash
pip install auto-gptq
```

2. Download quantized weights, for Qwen-7B-Chat, you can find it [here](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4):
```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen-7B-Chat-Int4
```

3. Build TRT-LLM engine:
```bash
python build.py --hf_model_dir Qwen-7B-Chat-Int4 \
                --quant_ckpt_path Qwen-7B-Chat-Int4 \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4_gptq \
                --per_group \
                --world_size 1 \
                --tp_size 1 \
                --output_dir ./tmp/Qwen/7B/trt_engines/int4-gptq/1-gpu
```

4. Run int4-gptq
```bash
python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir Qwen-7B-Chat-Int4 \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/int4-gptq/1-gpu
```
```
......
Input [Text 0]: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请问你叫什么？<|im_end|>
<|im_start|>assistant
"
Output [Text 0 Beam 0]: "你好，我是通义千问，由阿里云开发。"
```

5. Summarize
- validate huggingface
```bash
python3 ../summarize.py --test_hf \
                        --tokenizer_dir ./tmp/Qwen/7B \
                        --hf_model_dir ./tmp/Qwen/7B \
                        --max_input_length 2048 \
                        --output_len 2048
```

- validate trt-llm
```bash
python3 ../summarize.py --test_trt_llm \
                        --tokenizer_dir ./tmp/Qwen/7B \
                        --engine_dir ./tmp/Qwen/7B/trt_engines/int4-gptq/1-gpu \
                        --max_input_length 2048 \
                        --output_len 2048
```

#### INT4-AWQ
To run the AWQ Qwen example, the following steps are required:
1. Weight quantization

    NVIDIA AMMO toolkit is used for AWQ weight quantization. Please see [examples/quantization/README.md](/examples/quantization/README.md#preparation) for AMMO installation instructions.

```bash
python3 ../quantization/quantize.py --model_dir ./tmp/Qwen/7B \
                                    --dtype float16 \
                                    --qformat int4_awq \
                                    --export_path ./qwen_7b_4bit_gs128_awq.pt \
                                    --calib_size 32
```

2. TRT-LLM engine:
```bash
python build.py --hf_model_dir ./tmp/Qwen/7B \
                --quant_ckpt_path ./qwen_7b_4bit_gs128_awq.pt \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4_awq \
                --per_group \
                --world_size 1 \
                --tp_size 1 \
                --output_dir ./tmp/Qwen/7B/trt_engines/int4-awq/1-gpu
```
3. Run int4-awq
```bash
python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/int4-awq/1-gpu
```
```
......
Input [Text 0]: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请问你叫什么？<|im_end|>
<|im_start|>assistant
"
Output [Text 0 Beam 0]: "你好，我叫通义千问，是由阿里云开发的AI助手。有什么我可以帮助你的吗？"
```

4. Summarize
- validate huggingface
```bash
python3 ../summarize.py --test_hf \
                        --tokenizer_dir ./tmp/Qwen/7B \
                        --hf_model_dir ./tmp/Qwen/7B \
                        --max_input_length 2048 \
                        --output_len 2048
```

- validate trt-llm
```bash
python3 ../summarize.py --test_trt_llm \
                        --tokenizer_dir ./tmp/Qwen/7B \
                        --engine_dir ./tmp/Qwen/7B/trt_engines/int4-awq/1-gpu \
                        --max_input_length 2048 \
                        --output_len 2048
```

### Run

To run a TensorRT-LLM Qwen model using the engines generated by build.py

```bash
# With fp16 inference
python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/fp16/1-gpu/

# With bf16 inference
python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/bf16/1-gpu

# With int8 weight only inference
python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/int8_weight_only/1-gpu/

# With int4 weight only inference
python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir ./tmp/Qwen/7B/ \
                  --engine_dir=./tmp/Qwen/7B/trt_engines/int4_weight_only/1-gpu/

# Run 72B model with 8-gpu
mpirun -n 8 --allow-run-as-root \
    python ../run.py --input_text "What is your name?" \
                     --max_output_len=50 \
                     --tokenizer_dir ./tmp/Qwen/72B/ \
                     --engine_dir=./tmp/Qwen/72B/trt_engines/fp16/8-gpu/
```

**Demo output of run.py:**
```bash
python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir /llm-models/Qwen-7B-Chat/ \
                  --engine_dir /engine_qwen
```

```
Loading engine from /engine_qwen/qwen_float16_tp1_rank0.engine
Input: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请问你叫什么？<|im_end|>
<|im_start|>assistant
"
Output: "我是来自阿里云的大规模语言模型，我叫通义千问。"
```
```bash
mpirun -n 8 --allow-run-as-root \
    python ../run.py --input_text "What is your name?" \
                     --max_output_len=50 \
                     --tokenizer_dir ./tmp/Qwen/72B/ \
                     --engine_dir=./tmp/Qwen/72B/trt_engines/fp16/8-gpu/
```
```
Input [Text 0]: "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is your name?<|im_end|>
<|im_start|>assistant
"
Output [Text 0 Beam 0]: "I am QianWen, a large language model created by Alibaba Cloud."
```
### Summarization using the Qwen model

```bash
# Run summarization using the Qwen 7B model in FP16.
python ../summarize.py --test_trt_llm \
                       --tokenizer_dir ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/fp16/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048

# Run summarization using the Qwen 7B model in BF16.
python ../summarize.py --test_trt_llm \
                       --tokenizer_dir ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/bf16/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048

# Run summarization using the Qwen 7B model quantized to INT8.
python ../summarize.py --test_trt_llm \
                       --tokenizer_dir  ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/int8_weight_only/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048

# Run summarization using the Qwen 7B model quantized to INT4.
python ../summarize.py --test_trt_llm \
                       --tokenizer_dir  ./tmp/Qwen/7B/ \
                       --data_type fp16 \
                       --engine_dir ./tmp/Qwen/7B/trt_engines/int4_weight_only/1-gpu/ \
                       --max_input_length 2048 \
                       --output_len 2048

# Run summarization using the Qwen 7B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python ../summarize.py --test_trt_llm \
                           --tokenizer_dir  ./tmp/Qwen/7B/ \
                           --data_type fp16 \
                           --engine_dir ./tmp/Qwen/7B/trt_engines/fp16/2-gpu/ \
                           --max_input_length 2048 \
                           --output_len 2048

# Run summarization using the Qwen 14B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python ../summarize.py --test_trt_llm \
                           --tokenizer_dir  ./tmp/Qwen/14B/ \
                           --data_type fp16 \
                           --engine_dir ./tmp/Qwen/14B/trt_engines/fp16/2-gpu/ \
                           --max_input_length 2048 \
                           --output_len 2048
```
**Demo output of summarize.py:**
```python
python3 ../summarize.py --test_trt_llm --tokenizer_dir /llm-models/Qwen-7B-Chat/ --engine_dir /engine_qwen --max_input_length 2048 --output_len 2048
```
```
[11/09/2023-02:21:10] [TRT-LLM] [I] Load tokenizer takes: 0.4043385982513428 sec
Downloading builder script: 100%|███████████████████████████████████████████| 9.27k/9.27k [00:00<00:00, 35.4MB/s]
Downloading and preparing dataset cnn_dailymail/3.0.0 to /root/.cache/huggingface/datasets/ccdv___cnn_dailymail/3
......
[11/09/2023-02:23:33] [TRT-LLM] [I]
 Highlights : ['James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .']
[11/09/2023-02:23:33] [TRT-LLM] [I]
 Summary : [['Actor James Best, known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," has died at 88 after a brief illness. Best\'s career spanned decades in theater and Hollywood, but it was his role in "The Dukes of Hazzard" that made him a household name. The show ran for seven seasons from 1979 to 1985 and became a hit on TV, spawning TV movies, an animated series and video games. Best\'s portrayal of Rosco was beloved by fans for his childlike enthusiasm and goofy catchphrases. He is survived by friends and colleagues who paid tribute to him on social media.']]
[11/09/2023-02:23:33] [TRT-LLM] [I] ---------------------------------------------------------
load rouge ...
Downloading builder script: 5.60kB [00:00, 18.9MB/s]
load rouge done
[11/09/2023-02:24:06] [TRT-LLM] [I] TensorRT-LLM (total latency: 30.13867211341858 sec)
[11/09/2023-02:24:06] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[11/09/2023-02:24:06] [TRT-LLM] [I]   rouge1 : 26.35215119137573
[11/09/2023-02:24:06] [TRT-LLM] [I]   rouge2 : 9.507814774384485
[11/09/2023-02:24:06] [TRT-LLM] [I]   rougeL : 18.171982659482865
[11/09/2023-02:24:06] [TRT-LLM] [I]   rougeLsum : 21.10413175647868
```

## Credits
This Qwen model example exists thanks to Tlntin (TlntinDeng01@gmail.com) and zhaohb (zhaohbcloud@126.com).
