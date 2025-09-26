# Run Gemma on TensorRT-LLM

## Table Of Contents

- [Run Gemma on TensorRT-LLM](#run-gemma-on-tensorrt-llm)
  - [Table Of Contents](#table-of-contents)
  - [Support Matrix](#support-matrix)
  - [Common scripts](#common-scripts)
    - [Convert checkpoint](#convert-checkpoint)
    - [Build engine](#build-engine)
    - [Run inference](#run-inference)
  - [Specific commands](#specific-commands)
    - [Run Gemma 2B](#run-gemma-2b)
      - [Run inference under bfloat16 for HF checkpoint](#run-inference-under-bfloat16-for-hf-checkpoint)
      - [Run inference under FP8 for keras checkpoint](#run-inference-under-fp8-for-keras-checkpoint)
      - [Run 2B inference under SmoothQuant for jax checkpoint](#run-2b-inference-under-smoothquant-for-jax-checkpoint)
      - [Run inference under weight only for jax checkpoint](#run-inference-under-weight-only-for-jax-checkpoint)
      - [Run inference under INT8 KV caches for jax checkpoint](#run-inference-under-int8-kv-caches-for-jax-checkpoint)
    - [Run Gemma 7B](#run-gemma-7b)
      - [Run inference under bfloat16 for torch checkpoint](#run-inference-under-bfloat16-for-torch-checkpoint)
      - [Run inference under FP8 for jax checkpoint](#run-inference-under-fp8-for-jax-checkpoint)
      - [Run 7B inference under SmoothQuant for jax checkpoint](#run-7b-inference-under-smoothquant-for-jax-checkpoint)
      - [Run inference under weight only for keras checkpoint](#run-inference-under-weight-only-for-keras-checkpoint)
      - [Run inference under INT8 KV caches for keras checkpoint](#run-inference-under-int8-kv-caches-for-keras-checkpoint)
    - [Run Gemma 2](#run-gemma-2)
      - [Run inference under bfloat16 for torch checkpoint](#run-inference-under-bfloat16-for-torch-checkpoint-1)
    - [Run Gemma 3](#run-gemma-3)
      - [Run inference under bfloat16 for HF checkpoint](#run-inference-under-bfloat16-for-hf-checkpoint-1)
      - [Disaggregated Serving](#disaggregated-serving)
      - [Dynamo](#dynamo)
    - [Run Modelopt Quantization](#run-modelopt-quantization)
      - [Requirements](#requirements)
      - [Quantize Checkpoints](#quantize-checkpoints)
      - [Build Engines](#build-engines)
      - [Accuracy Results on MMLU](#accuracy-results-on-mmlu)

## Support Matrix
  * FP32/FP16/BF16/INT8 Weight-Only/INT4 AWQ/SmoothQuant/FP8
    * For SmoothQuant, TRT-LLM only supports FP16 higher precision now.
  * checkpoint type: Jax, Torch, Keras, Huggingface (HF)
  * STRONGLY TYPED
  * python runtime and triton backend

## Common scripts

### Convert checkpoint

Please install required packages first:

```bash
pip install -r requirements.txt
```

Users can use `convert_checkpoint.py` to convert the different source checkpoint to unified TensorRT LLM checkpoint format. Users could set `--dtype` to determine the inference data type, and set the quantization options like `--enable_fp8`, `--fp8_kv_cache` `--use_smooth_quant`, `--calibrate_kv_cache` (for INT8 kv cache) and `--use-weight-only-with-precision` (weight only). Users could also control the source checkpoint type by `--ckpt-type`. Currently, supported checkpoint types are `jax`, `torch` and `keras`.

```bash
CKPT_PATH=/tmp/models/gemma_nv/checkpoints/tmp_2b_it
UNIFIED_CKPT_PATH=/tmp/checkpoints/tmp_2b_it_tensorrt_llm/bf16/tp1/

python3 ./convert_checkpoint.py \
    --ckpt-type jax \
    --model-dir ${CKPT_PATH} \
    --dtype bfloat16 \
    --world-size 1 \
    --output-model-dir ${UNIFIED_CKPT_PATH}
```

### Build engine

After getting checkpoint, we can use `trtllm-build` command to build TensorRT LLM engines from TensorRT LLM checkpoints.

```bash
ENGINE_PATH=/tmp/gemma/2B/bf16/1-gpu/
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}
```

### Run inference

We provide three examples to run inference `run.py`, `summarize.py` and `mmlu.py`. `run.py` only run inference with `input_text` and show the output.

`summarize.py` runs summarization on [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset and evaluate the model by [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.

`mmlu.py` runs MMLU to evaluate the model by accuracy.

Note that we need to download the dataset of MMLU first and the evaluation of MMLU requires more time.

* run.py

```bash
VOCAB_FILE_PATH=/tmp/models/gemma_nv/checkpoints/tmp_vocab.model
python3 ../../../run.py --engine_dir ${ENGINE_PATH} \
                  --max_output_len 30 \
                  --vocab_file ${VOCAB_FILE_PATH}

[TensorRT-LLM] TensorRT LLM version: 0.9.0.dev2024020600Input [Text 0]: "<bos> Born in north-east France, Soyer trained as a"
Output [Text 0 Beam 0]: "chef in the renowned kitchens of Lyon. After honing his skills in various Michelin-starred establishments, he embarked on a solo venture, establishing his own restaurant"
```

* summarize.py

```bash
python3 ../../../summarize.py --test_trt_llm \
                        --engine_dir ${ENGINE_PATH} \
                        --batch_size 8 \
                        --max_ite 5 \
                        --vocab_file ${VOCAB_FILE_PATH}

[02/06/2024-10:08:54] [TRT-LLM] [I] TensorRT LLM (total latency: 3.2821836471557617 sec)
[02/06/2024-10:08:54] [TRT-LLM] [I] TensorRT LLM (total output tokens: 1989)
[02/06/2024-10:08:54] [TRT-LLM] [I] TensorRT LLM (tokens per second: 605.9989975648089)
[02/06/2024-10:08:54] [TRT-LLM] [I] TensorRT LLM beam 0 result
[02/06/2024-10:08:55] [TRT-LLM] [I]   rouge1 : 26.376388677070615
[02/06/2024-10:08:55] [TRT-LLM] [I]   rouge2 : 7.468157586877296
[02/06/2024-10:08:55] [TRT-LLM] [I]   rougeL : 17.953060795106556
[02/06/2024-10:08:55] [TRT-LLM] [I]   rougeLsum : 22.410938121151652
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
python3 ../../../mmlu.py --test_trt_llm \
                  --vocab_file ${VOCAB_FILE_PATH} \
                  --engine_dir ${ENGINE_PATH}

Average accuracy 0.358 - social sciences
Average accuracy 0.359 - other (business, health, misc.)
Average accuracy: 0.329
```

## Specific commands

In this section, we demonstrate the scripts to convert checkpoint, building engine and run inference on different settings. We will not demonstrate all combinations here because there are too many cases. We choose some important cases to demonstrate.

### Run Gemma 2B

#### Run inference under bfloat16 for HF checkpoint

```bash
git clone git@hf.co:google/gemma-2b
CKPT_PATH=gemma-2b/
UNIFIED_CKPT_PATH=/tmp/ckpt/hf/gemma/2b/1-gpu/
ENGINE_PATH=/tmp/engines/gemma/2B/bf16/1-gpu/
VOCAB_FILE_PATH=gemma-2b/

python3 ./convert_checkpoint.py \
    --ckpt-type hf \
    --model-dir ${CKPT_PATH} \
    --dtype bfloat16 \
    --world-size 1 \
    --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}

python3 ../../../summarize.py --test_trt_llm \
                      --tokenizer_dir ${VOCAB_FILE_PATH} \
                      --engine_dir ${ENGINE_PATH} \
                      --batch_size 8 \
                      --max_ite 5

[03/05/2024-02:24:39] [TRT-LLM] [I] TensorRT LLM (total latency: 3.0897433757781982 sec)
[03/05/2024-02:24:39] [TRT-LLM] [I] TensorRT LLM (total output tokens: 2141)
[03/05/2024-02:24:39] [TRT-LLM] [I] TensorRT LLM (tokens per second: 692.9378073221881)
[03/05/2024-02:24:39] [TRT-LLM] [I] TensorRT LLM beam 0 result
[03/05/2024-02:24:39] [TRT-LLM] [I]   rouge1 : 21.042873132085678
[03/05/2024-02:24:39] [TRT-LLM] [I]   rouge2 : 6.322669223228836
[03/05/2024-02:24:39] [TRT-LLM] [I]   rougeL : 16.450116567540338
[03/05/2024-02:24:39] [TRT-LLM] [I]   rougeLsum : 18.836567173262736
```

#### Run inference under FP8 for keras checkpoint

WARNING: This way of running FP8 will introduce noticeable accuracy drop. To avoid that, use Modelopt quantization mentioned in this readme.

In this example, we demonstrate how to run FP8 inference on Gemma. Note that `convert_checkpoint.py` only uses identity activation scales, so the accuracy might be little worse than higher precision in some cases, but it is still very good because we don't do any calibration. This also shows the stability of FP8 compared to INT8.

```bash
git clone git@hf.co:google/gemma-2b-it-keras
GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:google/gemma-2b-it-flax # clone tokenizer model
cd gemma-2b-it-flax
git lfs pull -I tokenizer.model

CKPT_PATH=gemma-2b-it-keras
UNIFIED_CKPT_PATH=/tmp/checkpoints/tmp_2b_en_tensorrt_llm/fp8/tp1/
ENGINE_PATH=/tmp/gemma/2B/fp8/1-gpu/
VOCAB_FILE_PATH=gemma-2b-it-flax/tokenizer.model

python3 ./convert_checkpoint.py \
    --ckpt-type keras \
    --model-dir ${CKPT_PATH} \
    --dtype bfloat16 \
    --world-size 1 \
    --enable_fp8 \
    --fp8_kv_cache \
    --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}

python3 ../../../summarize.py --test_trt_llm \
                      --vocab_file ${VOCAB_FILE_PATH} \
                      --engine_dir ${ENGINE_PATH} \
                      --batch_size 8 \
                      --max_ite 5

[02/08/2024-10:37:15] [TRT-LLM] [I] TensorRT LLM (total latency: 3.116227149963379 sec)
[02/08/2024-10:37:15] [TRT-LLM] [I] TensorRT LLM (total output tokens: 2419)
[02/08/2024-10:37:15] [TRT-LLM] [I] TensorRT LLM (tokens per second: 776.259201781368)
[02/08/2024-10:37:15] [TRT-LLM] [I] TensorRT LLM beam 0 result
[02/08/2024-10:37:15] [TRT-LLM] [I]   rouge1 : 20.206082692133098
[02/08/2024-10:37:15] [TRT-LLM] [I]   rouge2 : 5.902141189518428
[02/08/2024-10:37:15] [TRT-LLM] [I]   rougeL : 15.403458457907643
[02/08/2024-10:37:15] [TRT-LLM] [I]   rougeLsum : 17.44535527417846

python3 ../mmlu.py --test_trt_llm \
                  --vocab_file ${VOCAB_FILE_PATH} \
                  --engine_dir ${ENGINE_PATH}

Average accuracy 0.390 - social sciences
Average accuracy 0.405 - other (business, health, misc.)
Average accuracy: 0.356
```

#### Run 2B inference under SmoothQuant for jax checkpoint

```bash
git clone git@hf.co:google/gemma-2b-it-flax
CKPT_PATH=gemma-2b-it-flax/2b-it/
UNIFIED_CKPT_PATH=/tmp/checkpoints/tmp_2b_it_tensorrt_llm/sq/tp1
ENGINE_PATH=/tmp/gemma/2B/int8_sq/1-gpu/
VOCAB_FILE_PATH=gemma-2b-it-flax/tokenizer.model

python3 ./convert_checkpoint.py \
    --ckpt-type jax \
    --model-dir ${CKPT_PATH} \
    --dtype float16 \
    --use_smooth_quant_plugin 0.5 \
    --tokenizer_dir ${VOCAB_FILE_PATH} \
    --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}

python3 ../../../summarize.py --test_trt_llm \
                      --vocab_file ${VOCAB_FILE_PATH} \
                      --engine_dir ${ENGINE_PATH} \
                      --batch_size 8 \
                      --max_ite 5

[02/08/2024-04:42:06] [TRT-LLM] [I] TensorRT LLM (total latency: 3.460859775543213 sec)
[02/08/2024-04:42:06] [TRT-LLM] [I] TensorRT LLM (total output tokens: 1786)
[02/08/2024-04:42:06] [TRT-LLM] [I] TensorRT LLM (tokens per second: 516.0567361385428)
[02/08/2024-04:42:06] [TRT-LLM] [I] TensorRT LLM beam 0 result
[02/08/2024-04:42:06] [TRT-LLM] [I]   rouge1 : 22.534044843245525
[02/08/2024-04:42:06] [TRT-LLM] [I]   rouge2 : 5.940093176022924
[02/08/2024-04:42:06] [TRT-LLM] [I]   rougeL : 16.258991712579736
[02/08/2024-04:42:06] [TRT-LLM] [I]   rougeLsum : 19.60977626046262
```

#### Run inference under weight only for jax checkpoint

Available precisions: `int8` and `int4`

Note that `int4-weight-only` might not be able to keep the accuracies on all models. If users want to use int4 to run inference, we recommend using `int4_awq`.

* `int8`

```bash
git clone git@hf.co:google/gemma-2b-it-flax
CKPT_PATH=gemma-2b-it-flax/2b-it/
UNIFIED_CKPT_PATH=/tmp/checkpoints/tmp_2b_it_tensorrt_llm/w8_a16/tp1/
ENGINE_PATH=/tmp/gemma/2B/w8_a16/1-gpu/
VOCAB_FILE_PATH=gemma-2b-it-flax/tokenizer.model

python3 ./convert_checkpoint.py \
    --ckpt-type jax \
    --model-dir ${CKPT_PATH} \
    --use-weight-only-with-precision int8 \
    --dtype bfloat16 \
    --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
                 --gemm_plugin auto \
                 --max_batch_size 32 \
                 --max_input_len 3000 \
                 --max_seq_len 3100 \
                 --output_dir ${ENGINE_PATH}

python3 ../../../summarize.py --test_trt_llm \
                      --vocab_file ${VOCAB_FILE_PATH} \
                      --engine_dir ${ENGINE_PATH} \
                      --batch_size 8 \
                      --max_ite 5

[02/08/2024-04:44:54] [TRT-LLM] [I] TensorRT LLM (total latency: 3.5987987518310547 sec)
[02/08/2024-04:44:54] [TRT-LLM] [I] TensorRT LLM (total output tokens: 1797)
[02/08/2024-04:44:54] [TRT-LLM] [I] TensorRT LLM (tokens per second: 499.3332842203787)
[02/08/2024-04:44:54] [TRT-LLM] [I] TensorRT LLM beam 0 result
[02/08/2024-04:44:54] [TRT-LLM] [I]   rouge1 : 24.48521318679745
[02/08/2024-04:44:54] [TRT-LLM] [I]   rouge2 : 7.240543314565931
[02/08/2024-04:44:54] [TRT-LLM] [I]   rougeL : 17.857921729984078
[02/08/2024-04:44:54] [TRT-LLM] [I]   rougeLsum : 21.214162155642896
```

#### Run inference under INT8 KV caches for jax checkpoint

```bash
git clone git@hf.co:google/gemma-2b-it-flax
CKPT_PATH=gemma-2b-it-flax/2b-it/
UNIFIED_CKPT_PATH=/tmp/checkpoints/tmp_2b_it_tensorrt_llm/int8kv/tp1
ENGINE_PATH=/tmp/gemma/2B/int8kv/1-gpu/
VOCAB_FILE_PATH=gemma-2b-it-flax/tokenizer.model

python3 ./convert_checkpoint.py \
             --ckpt-type jax \
             --model-dir ${CKPT_PATH} \
             --world-size 1 \
             --dtype bfloat16 \
             --calibrate_kv_cache \
             --tokenizer_dir ${VOCAB_FILE_PATH} \
             --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --gemm_plugin auto \
             --max_batch_size 32 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}

python3 ../../../summarize.py --test_trt_llm \
                      --vocab_file ${VOCAB_FILE_PATH} \
                      --engine_dir ${ENGINE_PATH} \
                      --batch_size 8 \
                      --max_ite 5

[02/08/2024-04:52:22] [TRT-LLM] [I] TensorRT LLM (total latency: 3.5348474979400635 sec)
[02/08/2024-04:52:22] [TRT-LLM] [I] TensorRT LLM (total output tokens: 1819)
[02/08/2024-04:52:22] [TRT-LLM] [I] TensorRT LLM (tokens per second: 514.5907994786265)
[02/08/2024-04:52:22] [TRT-LLM] [I] TensorRT LLM beam 0 result
[02/08/2024-04:52:22] [TRT-LLM] [I]   rouge1 : 24.0397941580232
[02/08/2024-04:52:22] [TRT-LLM] [I]   rouge2 : 7.325311340360227
[02/08/2024-04:52:22] [TRT-LLM] [I]   rougeL : 17.54210044633271
[02/08/2024-04:52:22] [TRT-LLM] [I]   rougeLsum : 20.627861723682177
```

### Run Gemma 7B

#### Run inference under bfloat16 for torch checkpoint

Since torch model does not have model config, we need to add it manually in `CKPT_PATH` with file name `config.json`.

```bash
git clone git@hf.co:google/gemma-7b-pytorch

CKPT_PATH=gemma-7b-pytorch/
UNIFIED_CKPT_PATH=/tmp/checkpoints/tmp_7b_it_tensorrt_llm/bf16/tp1/
ENGINE_PATH=/tmp/gemma/7B/bf16/1-gpu/
VOCAB_FILE_PATH=gemma-7b-pytorch/tokenizer.model

python3 ./convert_checkpoint.py \
    --ckpt-type torch \
    --model-dir ${CKPT_PATH} \
    --dtype bfloat16 \
    --world-size 1 \
    --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}

python3 ../../../summarize.py --test_trt_llm \
                      --vocab_file ${VOCAB_FILE_PATH} \
                      --engine_dir ${ENGINE_PATH} \
                      --batch_size 8 \
                      --max_ite 5

python3 ../../../mmlu.py --test_trt_llm \
                 --vocab_file ${VOCAB_FILE_PATH} \
                 --engine_dir ${ENGINE_PATH}

Average accuracy 0.739 - social sciences
Average accuracy 0.697 - other (business, health, misc.)
Average accuracy: 0.630
```

#### Run inference under FP8 for jax checkpoint

WARNING: This way of running FP8 will introduce noticeable accuracy drop. To avoid that, use Modelopt quantization mentioned in this readme.

In this example, we demonstrate how to run FP8 inference on Gemma. Note that `convert_checkpoint.py` only uses identity activation scales, so the accuracy might be little worse than higher precision in some cases, but it is still very good because we don't do any calibration. This also shows the stability of FP8 compared to INT8.

```bash
CKPT_PATH=/tmp/models/gemma_nv/checkpoints/tmp_7b_it
UNIFIED_CKPT_PATH=/tmp/checkpoints/tmp_7b_it_tensorrt_llm/fp8/tp1/
ENGINE_PATH=/tmp/gemma/7B/fp8/1-gpu/
VOCAB_FILE_PATH=/tmp/models/gemma_nv/checkpoints/tmp_vocab.model

python3 ./convert_checkpoint.py \
    --ckpt-type jax \
    --model-dir ${CKPT_PATH} \
    --dtype bfloat16 \
    --world-size 1 \
    --enable_fp8 \
    --fp8_kv_cache \
    --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}

python3 ../../../summarize.py --test_trt_llm \
                      --vocab_file ${VOCAB_FILE_PATH} \
                      --engine_dir ${ENGINE_PATH} \
                      --batch_size 8 \
                      --max_ite 5

[02/08/2024-06:42:13] [TRT-LLM] [I] TensorRT LLM (total latency: 5.884302377700806 sec)
[02/08/2024-06:42:13] [TRT-LLM] [I] TensorRT LLM (total output tokens: 2694)
[02/08/2024-06:42:13] [TRT-LLM] [I] TensorRT LLM (tokens per second: 457.8282737830064)
[02/08/2024-06:42:13] [TRT-LLM] [I] TensorRT LLM beam 0 result
[02/08/2024-06:42:13] [TRT-LLM] [I]   rouge1 : 27.18633861010837
[02/08/2024-06:42:13] [TRT-LLM] [I]   rouge2 : 7.734928823230158
[02/08/2024-06:42:13] [TRT-LLM] [I]   rougeL : 19.32537431798716
[02/08/2024-06:42:13] [TRT-LLM] [I]   rougeLsum : 22.82522575944535
```

#### Run 7B inference under SmoothQuant for jax checkpoint

```bash
git clone git@hf.co:google/gemma-7b-it-flax
CKPT_PATH=gemma-7b-it-flax/7b-it/
UNIFIED_CKPT_PATH=/tmp/checkpoints/tmp_7b_it_tensorrt_llm/sq/tp1
ENGINE_PATH=/tmp/gemma/7B/int8_sq/1-gpu/
VOCAB_FILE_PATH=gemma-7b-it-flax/tokenizer.model

python3 ./convert_checkpoint.py \
    --ckpt-type jax \
    --model-dir ${CKPT_PATH} \
    --dtype float16 \
    --use_smooth_quant_plugin 0.5 \
    --tokenizer_dir ${VOCAB_FILE_PATH} \
    --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}

python3 ../../../summarize.py --test_trt_llm \
                        --vocab_file ${VOCAB_FILE_PATH} \
                        --engine_dir ${ENGINE_PATH} \
                        --batch_size 8 \
                        --max_ite 5

[02/19/2024-10:02:53] [TRT-LLM] [I] ---------------------------------------------------------
[02/19/2024-10:03:09] [TRT-LLM] [I] TensorRT LLM (total latency: 13.65670919418335 sec)
[02/19/2024-10:03:09] [TRT-LLM] [I] TensorRT LLM (total output tokens: 8351)
[02/19/2024-10:03:09] [TRT-LLM] [I] TensorRT LLM (tokens per second: 611.494312521266)
[02/19/2024-10:03:09] [TRT-LLM] [I] TensorRT LLM beam 0 result
[02/19/2024-10:03:09] [TRT-LLM] [I]   rouge1 : 28.8107815115074
[02/19/2024-10:03:09] [TRT-LLM] [I]   rouge2 : 8.623835512061866
[02/19/2024-10:03:09] [TRT-LLM] [I]   rougeL : 19.7277195532959
[02/19/2024-10:03:09] [TRT-LLM] [I]   rougeLsum : 23.434950511855114
```

#### Run inference under weight only for keras checkpoint

Available precisions: `int8` and `int4`

Note that `int4-weight-only` might not be able to keep the accuracies on all models. If users want to use int4 to run inference, we recommend using `int4_awq`.

* `int8`

```bash
git clone git@hf.co:google/gemma-7b-it-keras
GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:google/gemma-7b-it-flax # clone tokenizer model
cd gemma-7b-it-flax
git lfs pull -I tokenizer.model

CKPT_PATH=gemma-7b-it-keras
UNIFIED_CKPT_PATH=/tmp/checkpoints/tmp_7b_it_tensorrt_llm/w8_a16/tp1/
ENGINE_PATH=/tmp/gemma/7B/w8_a16/1-gpu/
VOCAB_FILE_PATH=gemma-7b-it-flax/tokenizer.model

python3 ./convert_checkpoint.py \
    --ckpt-type keras \
    --model-dir ${CKPT_PATH} \
    --use-weight-only-with-precision int8 \
    --dtype bfloat16 \
    --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
                 --gemm_plugin auto \
                 --max_batch_size 32 \
                 --max_input_len 3000 \
                 --max_seq_len 3100 \
                 --output_dir ${ENGINE_PATH}

python3 ../../../summarize.py --test_trt_llm \
                      --vocab_file ${VOCAB_FILE_PATH} \
                      --engine_dir ${ENGINE_PATH} \
                      --batch_size 8 \
                      --max_ite 5

[02/08/2024-07:38:15] [TRT-LLM] [I] TensorRT LLM (total latency: 8.49835753440857 sec)
[02/08/2024-07:38:15] [TRT-LLM] [I] TensorRT LLM (total output tokens: 2654)
[02/08/2024-07:38:15] [TRT-LLM] [I] TensorRT LLM (tokens per second: 312.2956393931832)
[02/08/2024-07:38:15] [TRT-LLM] [I] TensorRT LLM beam 0 result
[02/08/2024-07:38:16] [TRT-LLM] [I]   rouge1 : 20.396209981234687
[02/08/2024-07:38:16] [TRT-LLM] [I]   rouge2 : 5.73302850102211
[02/08/2024-07:38:16] [TRT-LLM] [I]   rougeL : 16.001683776127507
[02/08/2024-07:38:16] [TRT-LLM] [I]   rougeLsum : 18.36957526315223
```

#### Run inference under INT8 KV caches for keras checkpoint

```bash
CKPT_PATH=/tmp/models/gemma_keras/keras/gemma_7b_en/
UNIFIED_CKPT_PATH=/tmp/checkpoints/tmp_7b_it_tensorrt_llm/int8kv/tp1
ENGINE_PATH=/tmp/gemma/7B/int8kv/1-gpu/
VOCAB_FILE_PATH=/tmp/models/gemma_nv/checkpoints/tmp_vocab.model

python3 ./convert_checkpoint.py \
             --ckpt-type keras \
             --model-dir ${CKPT_PATH} \
             --world-size 1 \
             --dtype bfloat16 \
             --calibrate_kv_cache \
             --tokenizer_dir ${VOCAB_FILE_PATH} \
             --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --gemm_plugin auto \
             --max_batch_size 32 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}

python3 ../../../summarize.py --test_trt_llm \
                      --vocab_file ${VOCAB_FILE_PATH} \
                      --engine_dir ${ENGINE_PATH} \
                      --batch_size 8 \
                      --max_ite 5

[02/08/2024-07:51:11] [TRT-LLM] [I] TensorRT LLM (total latency: 8.73880124092102 sec)
[02/08/2024-07:51:11] [TRT-LLM] [I] TensorRT LLM (total output tokens: 2771)
[02/08/2024-07:51:11] [TRT-LLM] [I] TensorRT LLM (tokens per second: 317.09154649544956)
[02/08/2024-07:51:11] [TRT-LLM] [I] TensorRT LLM beam 0 result
[02/08/2024-07:51:11] [TRT-LLM] [I]   rouge1 : 20.934864626327627
[02/08/2024-07:51:11] [TRT-LLM] [I]   rouge2 : 4.954721611692932
[02/08/2024-07:51:11] [TRT-LLM] [I]   rougeL : 15.307592049634444
[02/08/2024-07:51:11] [TRT-LLM] [I]   rougeLsum : 17.94213019528988
```


### Run Gemma 2

Gemma 2 currently has following limitations:
 - Only HF style checkpoints are supported.
 - The maximum sequence length allowed is 4096.
#### Run inference under bfloat16 for torch checkpoint
```bash
variant=9b # 27b
git clone git@hf.co:google/gemma-2-$variant-it

CKPT_PATH=gemma-2-$variant-it/
UNIFIED_CKPT_PATH=/tmp/checkpoints/tmp_$variant_it_tensorrt_llm/bf16/tp1/
ENGINE_PATH=/tmp/gemma2/$variant/bf16/1-gpu/
VOCAB_FILE_PATH=gemma-2-$variant-it/tokenizer.model

python3 ./examples/models/core/gemma/convert_checkpoint.py \
    --ckpt-type hf \
    --model-dir ${CKPT_PATH} \
    --dtype bfloat16 \
    --world-size 1 \
    --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}

python3 ../../../summarize.py --test_trt_llm \
                      --vocab_file ${VOCAB_FILE_PATH} \
                      --engine_dir ${ENGINE_PATH} \
                      --batch_size 8 \
                      --max_ite 5

python3 ../../../mmlu.py --test_trt_llm \
                 --vocab_file ${VOCAB_FILE_PATH} \
                 --engine_dir ${ENGINE_PATH}

Average accuracy 0.739 - social sciences
Average accuracy 0.697 - other (business, health, misc.)
Average accuracy: 0.630
```

### Run Gemma 3

Gemma 3's text generation model from HuggingFace is supported. Gemma3 1B model interleaves 5 local layers between each global layer. While local layers use sliding window attention with a short span of 512 tokens, global layers attend to the long context. TRTLLM support layerwise sliding-window attention and the sliding window size for each layer could be passed in using the `--max_attention_window_size` parameter at runtime. If a subpattern is provided, TRTLLM can extrapolate the complete pattern and the extrapolation logic is printed to terminal.

#### Run inference under bfloat16 for HF checkpoint
```bash
git clone https://huggingface.co/google/gemma-3-1b-it

CKPT_PATH=gemma-3-1b-it/
UNIFIED_CKPT_PATH=/tmp/checkpoints/tmp_1b_it_tensorrt_llm/bf16/tp1/
ENGINE_PATH=/tmp/gemma3/1b/bf16/1-gpu/
VOCAB_FILE_PATH=gemma-3-1b-it/tokenizer.model

python3 ./convert_checkpoint.py \
    --ckpt-type hf \
    --model-dir ${CKPT_PATH} \
    --dtype bfloat16 \
    --world-size 1 \
    --output-model-dir ${UNIFIED_CKPT_PATH}

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}

python3 ../../../summarize.py --test_trt_llm \
                      --vocab_file ${VOCAB_FILE_PATH} \
                      --engine_dir ${ENGINE_PATH} \
                      --batch_size 1 \
                      --max_ite 5 \
                      --max_attention_window_size 512 512 512 512 512 3100

...
[TensorRT-LLM][INFO] TRTGptModel mMaxAttentionWindowSize: (512, 512, 512, 512, 512, 3100) * 4 + (512, 512)
...
[04/09/2025-18:28:26] [TRT-LLM] [I] TensorRT LLM (total latency: 1.6197962760925293 sec)
[04/09/2025-18:28:26] [TRT-LLM] [I] TensorRT LLM (total output tokens: 475)
[04/09/2025-18:28:26] [TRT-LLM] [I] TensorRT LLM (tokens per second: 293.2467539349165)
[04/09/2025-18:28:26] [TRT-LLM] [I] TensorRT LLM beam 0 result
[04/09/2025-18:28:26] [TRT-LLM] [I]   rouge1: 22.780314381954003
[04/09/2025-18:28:26] [TRT-LLM] [I]   rouge2: 4.331099231480823
[04/09/2025-18:28:26] [TRT-LLM] [I]   rougeL: 15.26751867562475
[04/09/2025-18:28:26] [TRT-LLM] [I]   rougeLsum: 20.14696930976001
```

#### Disaggregated Serving

To serve the model in disaggregated mode, you should launch context and generation servers using `trtllm-serve`.

For example, you can launch a single context server on port 8001 with:

```bash
export TRTLLM_USE_UCX_KVCACHE=1

cat >./ctx-extra-llm-api-config.yml <<EOF
print_iter_log: true
disable_overlap_scheduler: true
kv_cache_config:
  max_attention_window: [512, 512, 512, 512, 512, 32768]
  enable_block_reuse: false
EOF

trtllm-serve \
  google/gemma-3-1b-it \
  --host localhost \
  --port 8001 \
  --max_batch_size 8 \
  --tp_size 2 \
  --ep_size 2 \
  --pp_size 1 \
  --kv_cache_free_gpu_memory_fraction 0.95 \
  --extra_llm_api_options ./ctx-extra-llm-api-config.yml \
  &> output_ctx_8001 &
```

Then launch a single generation server on port 8002 with:

```bash
cat >./gen-extra-llm-api-config.yml <<EOF
print_iter_log: true
kv_cache_config:
  max_attention_window: [512, 512, 512, 512, 512, 32768]
  enable_block_reuse: false
EOF

trtllm-serve \
  google/gemma-3-1b-it \
  --host localhost \
  --port 8002 \
  --max_batch_size 8 \
  --tp_size 2 \
  --ep_size 2 \
  --pp_size 1 \
  --kv_cache_free_gpu_memory_fraction 0.95 \
  --extra_llm_api_options ./gen-extra-llm-api-config.yml \
  &> output_gen_8002 &
```

Finally, you can launch the disaggregated server which will accept requests from the client and do
the orchestration between the context and generation servers with:

```bash
cat >./disagg-config.yaml <<EOF
hostname: localhost
port: 8000
backend: pytorch
context_servers:
  num_instances: 1
  urls:
      - "localhost:8001"
generation_servers:
  num_instances: 1
  urls:
      - "localhost:8002"
EOF

trtllm-serve disaggregated -c disagg-config.yaml
```

To query the server, you can start with a `curl` command:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "google/gemma-3-1b-it",
      "prompt": "Where is New York?",
      "max_tokens": 16,
      "temperature": 0
  }'
```


#### Dynamo

NVIDIA Dynamo is a high-throughput low-latency inference framework designed for serving generative AI and reasoning models in multi-node distributed environments.
Dynamo supports TensorRT LLM as one of its inference engine. For details on how to use TensorRT LLM with Dynamo please refer to [LLM Deployment Examples using TensorRT-LLM](https://github.com/ai-dynamo/dynamo/blob/main/examples/tensorrt_llm/README.md)

### Run Modelopt Quantization

#### Requirements

Modelopt toolkit also provides quantization solutions. To enable it, have the latest modelopt and transformers Python package installed to support Gemma. Then run the following commands.

#### Quantize Checkpoints

```bash
python ../../../quantization/quantize.py --model_dir ${HF_GEMMA_PATH} \
            --dtype float16 \
            --qformat ${QUANT_TYPE} \
            --output_dir ${UNIFIED_CKPT_PATH} \
            --tp_size 1
```
HF_GEMMA_PATH can either be HF model card name or the downloaded model path. QUANT_TYPE can be chosen from fp8, int4_awq, and int8_sq.

#### Build Engines

For fp8, build engines with:
```bash
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}
```

For int4_awq and int8_sq, build engines with:

```bash
trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --gemm_plugin auto \
             --max_batch_size 8 \
             --max_input_len 3000 \
             --max_seq_len 3100 \
             --output_dir ${ENGINE_PATH}
```

#### Accuracy Results on MMLU

| Model         | fp8   | int4_awq | int8_sq (Modelopt) | int8_sq (Native per-channel) |
|---------------|-------|----------|----------------|------------------|
| 2B Pretrained | 0.407 | 0.378    |    0.338       |     0.338        |
| 7B Pretrained | 0.643 | 0.615    |    0.448       |     0.595        |
