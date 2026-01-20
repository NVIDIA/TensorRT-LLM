# RecurrentGemma

This document shows how to build and run a [RecurrentGemma](https://github.com/google-deepmind/recurrentgemma) model in TensorRT-LLM.

## Overview

The TensorRT LLM RecurrentGemma implementation can be found in [`tensorrt_llm/models/recurrentgemma/model.py`](../../../../tensorrt_llm/models/recurrentgemma/model.py). The TensorRT LLM RecurrentGemma example code is located in [`examples/models/core/recurrentgemma`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert a checkpoint from the JAX format to the TensorRT LLM format.

In addition, there are two shared files in the parent folder [`examples`](../../../) for inference and evaluation:

* [`run.py`](../../../run.py) to run the inference on an input text;
* [`summarize.py`](../../../summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset.

## Support Matrix
|    Checkpoint type    | FP16  | BF16  | FP8 | INT8 SQ | INT4 AWQ | TP  |
| :-------------------: | :---: | :---: | :-: | :-----: | :------: | :-: |
|    Huggingface (HF)   |   Y   |   Y   |  Y  |    Y    |    Y     |  Y  |
|    Jax                |   Y   |   Y   |  N  |    N    |    N     |  Y  |

* TensorRT LLM can support different post-training quantization for the Huggingface checkpoints, including FP8, INT8 SmoothQuant, and INT4 AWQ.

## Usage

### 1. Prepare requirements and download weights

Please install required packages first and setup `git-lfs`:

```bash
pip install -r requirements.txt
git lfs install
```

Then use one of the following commands to fetch the checkpoint you are interested in. These models are public but users need to login and then they are able to clone the models.

```bash
# recurrentgemma-2b
git clone https://huggingface.co/google/recurrentgemma-2b ./recurrentgemma_model/recurrentgemma-2b

# recurrentgemma-2b-it
git clone https://huggingface.co/google/recurrentgemma-2b-it ./recurrentgemma_model/recurrentgemma-2b-it

# recurrentgemma-2b-flax
git clone https://huggingface.co/google/recurrentgemma-2b-flax ./recurrentgemma_model/recurrentgemma-2b-flax

# recurrentgemma-2b-it-flax
git clone https://huggingface.co/google/recurrentgemma-2b-it-flax ./recurrentgemma_model/recurrentgemma-2b-it-flax
```

### 2. Convert weights from JAX to TensorRT LLM format
The [`convert_checkpoint.py`](./convert_checkpoint.py) script converts HF/JAX weights to TensorRT LLM checkpoints. TensorRT LLM can support different post-training quantization methods. Here we use recurrentgemma-2b-it model as an example to show how to run quantized model.

```bash
# recurrentgemma-2b
CKPT_2B_PATH=./recurrentgemma_model/recurrentgemma-2b
UNIFIED_CKPT_2B_PATH=./recurrentgemma_model/recurrentgemma-2b/trt_ckpt/fp16/1-gpu/
python convert_checkpoint.py --model_dir ${CKPT_2B_PATH} \
                             --ckpt_type hf \
                             --dtype float16 \
                             --output_dir ${UNIFIED_CKPT_2B_PATH}

# recurrentgemma-2b-it FP8 with FP8 kv cache
CKPT_2B_IT_PATH=./recurrentgemma_model/recurrentgemma-2b-it
UNIFIED_CKPT_2B_IT_FP8_PATH=./recurrentgemma_model/recurrentgemma-2b-it/trt_ckpt/fp8/1-gpu/
python ../../../quantization/quantize.py --model_dir ${CKPT_2B_IT_PATH} \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ${UNIFIED_CKPT_2B_IT_FP8_PATH} \
                                   --calib_size 512 \
                                   --tp_size 1

# recurrentgemma-2b-it INT8 SmoothQuant with INT8 kv cache
UNIFIED_CKPT_2B_IT_INT8_SQ_PATH=./recurrentgemma_model/recurrentgemma-2b-it/trt_ckpt/int8_sq/1-gpu/
python ../../../quantization/quantize.py --model_dir ${CKPT_2B_IT_PATH} \
                                   --dtype float16 \
                                   --qformat int8_sq \
                                   --kv_cache_dtype int8 \
                                   --output_dir ${UNIFIED_CKPT_2B_IT_INT8_SQ_PATH} \
                                   --calib_size 512 \
                                   --tp_size 1

# recurrentgemma-2b-it INT4 AWQ with INT8 kv cache
UNIFIED_CKPT_2B_IT_INT4_AWQ_PATH=./recurrentgemma_model/recurrentgemma-2b-it/trt_ckpt/int4_awq/1-gpu/
python ../../../quantization/quantize.py --model_dir ${CKPT_2B_IT_PATH} \
                                   --dtype float16 \
                                   --qformat int4_awq \
                                   --kv_cache_dtype int8 \
                                   --output_dir ${UNIFIED_CKPT_2B_IT_INT4_AWQ_PATH} \
                                   --calib_size 512 \
                                   --tp_size 1

# recurrentgemma-2b-flax
CKPT_2B_FLAX_PATH=./recurrentgemma_model/recurrentgemma-2b-flax/2b
UNIFIED_CKPT_2B_FLAX_PATH=./recurrentgemma_model/recurrentgemma-2b-flax/trt_ckpt/fp16/1-gpu/
python convert_checkpoint.py --model_dir ${CKPT_2B_FLAX_PATH} \
                             --ckpt_type jax \
                             --dtype float16 \
                             --output_dir ${UNIFIED_CKPT_2B_FLAX_PATH}

# recurrentgemma-2b-it-flax
CKPT_2B_IT_FLAX_PATH=./recurrentgemma_model/recurrentgemma-2b-it-flax/2b-it
UNIFIED_CKPT_2B_IT_FLAX_PATH=./recurrentgemma_model/recurrentgemma-2b-it-flax/trt_ckpt/bf16/1-gpu/
python convert_checkpoint.py --model_dir ${CKPT_2B_IT_FLAX_PATH} \
                             --ckpt_type jax \
                             --dtype bfloat16 \
                             --output_dir ${UNIFIED_CKPT_2B_IT_FLAX_PATH}
```

### 3. Build TensorRT engine(s)
After getting checkpoint, we can use `trtllm-build` command to build TensorRT LLM engines from TensorRT LLM checkpoints.

```bash
# recurrentgemma-2b
ENGINE_2B_PATH=./recurrentgemma_model/recurrentgemma-2b/trt_engines/fp16/1-gpu/
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_2B_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_2B_PATH}

# recurrentgemma-2b-it FP8 with FP8 kv cache
ENGINE_2B_IT_FP8_PATH=./recurrentgemma_model/recurrentgemma-2b-it/trt_engines/fp8/1-gpu/
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_2B_IT_FP8_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_2B_IT_FP8_PATH}

# recurrentgemma-2b-it INT8 SmoothQuant with INT8 kv cache
ENGINE_2B_IT_INT8_SQ_PATH=./recurrentgemma_model/recurrentgemma-2b-it/trt_engines/int8_sq/1-gpu/
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_2B_IT_INT8_SQ_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_2B_IT_INT8_SQ_PATH}

# recurrentgemma-2b-it INT4 AWQ with INT8 kv cache
ENGINE_2B_IT_INT4_AWQ_PATH=./recurrentgemma_model/recurrentgemma-2b-it/trt_engines/int4_awq/1-gpu/
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_2B_IT_INT4_AWQ_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_2B_IT_INT4_AWQ_PATH}

# recurrentgemma-2b-flax
ENGINE_2B_FLAX_PATH=./recurrentgemma_model/recurrentgemma-2b-flax/trt_engines/fp16/1-gpu/
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_2B_FLAX_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_2B_FLAX_PATH}

# recurrentgemma-2b-it-flax
ENGINE_2B_IT_FLAX_PATH=./recurrentgemma_model/recurrentgemma-2b-it-flax/trt_engines/bf16/1-gpu/
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_2B_IT_FLAX_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_2B_IT_FLAX_PATH}
```

### 4. Run inference with the TensorRT engine(s)

We provide three examples to run inference `run.py`, `summarize.py` and `mmlu.py`. `run.py` only run inference with `input_text` and show the output.

`summarize.py` runs summarization on [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset and evaluate the model by [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.

`mmlu.py` runs MMLU to evaluate the model by accuracy.

Note that we need to download the dataset of MMLU first and the evaluation of MMLU requires more time.

* run.py

```bash
# recurrentgemma-2b
TOKENIZER_DIR_2B_PATH=./recurrentgemma_model/recurrentgemma-2b
python3 ../../../run.py --max_output_len=100 \
                  --use_py_session \
                  --max_attention_window_size 2048 \
                  --tokenizer_dir ${TOKENIZER_DIR_2B_PATH} \
                  --engine_dir ${ENGINE_2B_PATH}

# recurrentgemma-2b-it FP8 with FP8 kv cache
TOKENIZER_DIR_2B_IT_PATH=./recurrentgemma_model/recurrentgemma-2b-it
python3 ../../../run.py --max_output_len=100 \
                  --use_py_session \
                  --max_attention_window_size 2048 \
                  --tokenizer_dir ${TOKENIZER_DIR_2B_IT_PATH} \
                  --engine_dir ${ENGINE_2B_IT_FP8_PATH}

# recurrentgemma-2b-it INT8 SmoothQuant with INT8 kv cache
python3 ../../../run.py --max_output_len=100 \
                  --use_py_session \
                  --max_attention_window_size 2048 \
                  --tokenizer_dir ${TOKENIZER_DIR_2B_IT_PATH} \
                  --engine_dir ${ENGINE_2B_IT_INT8_SQ_PATH}

# recurrentgemma-2b-it INT4 AWQ with INT8 kv cache
python3 ../../../run.py --max_output_len=100 \
                  --use_py_session \
                  --max_attention_window_size 2048 \
                  --tokenizer_dir ${TOKENIZER_DIR_2B_IT_PATH} \
                  --engine_dir ${ENGINE_2B_IT_INT4_AWQ_PATH}

# recurrentgemma-2b-flax
VOCAB_FILE_2B_FLAX_PATH=./recurrentgemma_model/recurrentgemma-2b-flax/tokenizer.model
python3 ../../../run.py --max_output_len=100 \
                  --use_py_session \
                  --max_attention_window_size 2048 \
                  --vocab_file ${VOCAB_FILE_2B_FLAX_PATH} \
                  --engine_dir ${ENGINE_2B_FLAX_PATH}

# recurrentgemma-2b-it-flax
VOCAB_FILE_2B_IT_FLAX_PATH=./recurrentgemma_model/recurrentgemma-2b-it-flax/tokenizer.model
python3 ../../../run.py --max_output_len=100 \
                  --use_py_session \
                  --max_attention_window_size 2048 \
                  --vocab_file ${VOCAB_FILE_2B_IT_FLAX_PATH} \
                  --engine_dir ${ENGINE_2B_IT_FLAX_PATH}
```

* summarize.py

```bash
# recurrentgemma-2b
python3 ../../../summarize.py --test_trt_llm \
                        --use_py_session \
                        --engine_dir ${ENGINE_2B_PATH} \
                        --batch_size 8 \
                        --max_attention_window_size 2048 \
                        --tokenizer_dir ${TOKENIZER_DIR_2B_PATH}

# recurrentgemma-2b-it FP8 with FP8 kv cache
python3 ../../../summarize.py --test_trt_llm \
                        --use_py_session \
                        --engine_dir ${ENGINE_2B_IT_FP8_PATH} \
                        --batch_size 8 \
                        --max_attention_window_size 2048 \
                        --tokenizer_dir ${TOKENIZER_DIR_2B_IT_PATH}

# recurrentgemma-2b-it INT8 SmoothQuant with INT8 kv cache
python3 ../../../summarize.py --test_trt_llm \
                        --use_py_session \
                        --engine_dir ${ENGINE_2B_IT_INT8_SQ_PATH} \
                        --batch_size 8 \
                        --max_attention_window_size 2048 \
                        --tokenizer_dir ${TOKENIZER_DIR_2B_IT_PATH}

# recurrentgemma-2b-it INT4 AWQ with INT8 kv cache
python3 ../../../summarize.py --test_trt_llm \
                        --use_py_session \
                        --engine_dir ${ENGINE_2B_IT_INT4_AWQ_PATH} \
                        --batch_size 8 \
                        --max_attention_window_size 2048 \
                        --tokenizer_dir ${TOKENIZER_DIR_2B_IT_PATH}

# recurrentgemma-2b-flax
python3 ../../../summarize.py --test_trt_llm \
                        --use_py_session \
                        --engine_dir ${ENGINE_2B_FLAX_PATH} \
                        --batch_size 8 \
                        --max_attention_window_size 2048 \
                        --vocab_file ${VOCAB_FILE_2B_FLAX_PATH}

# recurrentgemma-2b-it-flax
python3 ../../../summarize.py --test_trt_llm \
                        --use_py_session \
                        --engine_dir ${ENGINE_2B_IT_FLAX_PATH} \
                        --batch_size 8 \
                        --max_attention_window_size 2048 \
                        --vocab_file ${VOCAB_FILE_2B_IT_FLAX_PATH}
```

* mmlu.py

Download the dataset first

```bash
mkdir data
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
tar -xf data/mmlu.tar -C data
mv data/data data/mmlu
```

Evaluate on MMLU dataset.

```bash
# recurrentgemma-2b
python3 ../../../mmlu.py --test_trt_llm \
                   --max_attention_window_size 2048 \
                   --tokenizer_dir ${TOKENIZER_DIR_2B_PATH} \
                   --engine_dir ${ENGINE_2B_PATH}

# recurrentgemma-2b-it FP8 with FP8 kv cache
python3 ../../../mmlu.py --test_trt_llm \
                   --max_attention_window_size 2048 \
                   --tokenizer_dir ${TOKENIZER_DIR_2B_IT_PATH} \
                   --engine_dir ${ENGINE_2B_IT_FP8_PATH}

# recurrentgemma-2b-it INT8 SmoothQuant with INT8 kv cache
python3 ../../../mmlu.py --test_trt_llm \
                   --max_attention_window_size 2048 \
                   --tokenizer_dir ${TOKENIZER_DIR_2B_IT_PATH} \
                   --engine_dir ${ENGINE_2B_IT_INT8_SQ_PATH}

# recurrentgemma-2b-it INT4 AWQ with INT8 kv cache
python3 ../../../mmlu.py --test_trt_llm \
                   --max_attention_window_size 2048 \
                   --tokenizer_dir ${TOKENIZER_DIR_2B_IT_PATH} \
                   --engine_dir ${ENGINE_2B_IT_INT4_AWQ_PATH}

# recurrentgemma-2b-flax
python3 ../../../mmlu.py --test_trt_llm \
                   --max_attention_window_size 2048 \
                   --vocab_file ${VOCAB_FILE_2B_FLAX_PATH} \
                   --engine_dir ${ENGINE_2B_FLAX_PATH}

# recurrentgemma-2b-it-flax
python3 ../../../mmlu.py --test_trt_llm \
                   --max_attention_window_size 2048 \
                   --vocab_file ${VOCAB_FILE_2B_IT_FLAX_PATH} \
                   --engine_dir ${ENGINE_2B_IT_FLAX_PATH}
```
