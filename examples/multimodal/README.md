# Multi-Modal

This document shows how to run multimodal pipelines with TensorRT-LLM, e.g. from image+text input modalities to text output.

Multimodal models' LLM part has an additional parameter `--max_multimodal_len` compared to LLM-only build commands. Under the hood, `max_multimodal_len` and `max_prompt_embedding_table_size` are effectively the same concept, i.e., prepended/concatenated embeddings (either multimodal feature embeddings or prompt tuning embeddings) to the LLM input embeddings. The multimodal features from the visual encoder of shape `[batch_size, num_visual_features, visual_hidden_dim]` is flattened as `[batch_size * num_visual_features, visual_hidden_dim]` and passed like a prompt embedding table.

We first describe how to run each model on a single GPU. We then provide general guidelines on using tensor parallelism for LLM part of the pipeline.

## BLIP2-T5

1. Download Huggingface weights and convert original checkpoint to TRT-LLM checkpoint format
   following example in `examples/enc_dec/README.md`.

    ```bash
    export MODEL_NAME=flan-t5-xl
    git clone https://huggingface.co/google/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

    python ../enc_dec/t5/convert.py -i tmp/hf_models/${MODEL_NAME} \
        -o tmp/trt_models/${MODEL_NAME} --weight_data_type float32 \
        --inference_tensor_para_size 1
    ```

2. Build TRT-LLM engine from TRT-LLM checkpoint

    ```bash
    python ../enc_dec/build.py --model_type t5 \
        --weight_dir tmp/trt_models/${MODEL_NAME}/tp1 \
        --output_dir trt_engines/${MODEL_NAME}/1-gpu \
        --engine_name ${MODEL_NAME} \
        --remove_input_padding \
        --use_bert_attention_plugin \
        --use_gpt_attention_plugin \
        --use_gemm_plugin \
        --dtype bfloat16 \
        --max_beam_width 1 \
        --max_batch_size 8 \
        --max_encoder_input_len 924 \
        --max_output_len 100 \
        --max_multimodal_len 256 # 8 (max_batch_size) * 32 (num_visual_features)
    ```

    **NOTE**: `max_multimodal_len = max_batch_size * num_visual_features`, so if you change max_batch_size, max multimodal length **MUST** be changed accordingly.

    The built T5 engines are located in `./trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1`.

3.  Build TensorRT engines for visual components

    ```bash
    python build_visual_engine.py --model_name ${MODEL_NAME} --model_path tmp/hf_models/${MODEL_NAME} --max_batch_size 8
    ```

    The built engines are located in `./visual_engines/${MODEL_NAME}`.

    To run the BLIP2 pipeline with batch size > 1, change `--max_batch_size` argument to `build_visual_engine.py` accordingly.

4. Assemble everything into BLIP2 pipeline

    ```bash
    python run.py \
        --blip2_encoder \
        --max_new_tokens 30 \
        --input_text "Question: which city is this? Answer:" \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1
    ```

## BLIP2-OPT

OPT pipeline needs few minor changes from T5 pipeline

1. Convert Huggingface weights to TRT-LLM checkpoint format following `examples/opt/README.md`.

2. Use `trtllm-build` command to build TRT-LLM engine for OPT.

3. Add `--decoder-llm` argument to inference script, since OPT is a decoder-only LLM.

4. The full list of commands is as follows:

    ```bash
    export MODEL_NAME=opt-2.7b
    git clone https://huggingface.co/facebook/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

    python ../opt/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --dtype float16 \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --gemm_plugin float16 \
        --max_beam_width 1 \
        --max_batch_size 8 \
        --max_multimodal_len 256 \
        --max_input_len 924 \
        --max_output_len 100

    python build_visual_engine.py --model_name ${MODEL_NAME} --model_path tmp/hf_models/${MODEL_NAME}

    python run.py \
        --blip2_encoder \
        --max_new_tokens 30 \
        --input_text "Question: which city is this? Answer:" \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --decoder_llm
    ```

5. INT8/INT4 weight-only quantization for OPT can be enabled using commands as follows (take `INT4` as an example, while `INT8` is the default precision for weight-only quantization):
    ```bash
    python ../opt/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --dtype float16 \
        --output_dir tmp/trt_models/${MODEL_NAME}/int4_weightonly/1-gpu \
        --use_weight_only \
        --weight_only_precision int4

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/int4_weightonly/1-gpu \
        --output_dir trt_engines/${MODEL_NAME}/int4_weightonly/1-gpu \
        --gemm_plugin float16 \
        --max_beam_width 1 \
        --max_batch_size 8 \
        --max_multimodal_len 256 \
        --max_input_len 924 \
        --max_output_len 100
    ```

    The built OPT engines lie in `trt_engines/${MODEL_NAME}/int4_weightonly/1-gpu`.
    You should use this directory as `--llm_engine_dir` argument to `run.py`

    **NOTE:** INT8/INT4 option is not supported for BLIP2-T5, because quantization support has not been
          added for encoder-decoder models yet.

## LLaVA

1. Download Huggingface model weights. This model has both LLM and visual components
   unlike BLIP2 example which downloads only LLM components from Huggingface.

    ```bash
    export MODEL_NAME="llava-1.5-7b-hf"
    git clone https://huggingface.co/llava-hf/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

2. Generate TRT-LLM engine for LLaMA following example in `examples/llama/README.md`

    ```bash
    python ../llama/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --dtype float16

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --gemm_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_output_len 512 \
        --max_multimodal_len 576 # 1 (max_batch_size) * 576 (num_visual_features)
    ```

3.  Build TensorRT engines for visual components

    ```bash
    python build_visual_engine.py --model_name ${MODEL_NAME} --model_path tmp/hf_models/${MODEL_NAME}
    ```

4. Add `--decoder-llm` argument to inference script, since LLaMA is a decoder-only LLM.

    ```bash
    python run.py \
        --max_new_tokens 30 \
        --input_text "Question: which city is this? Answer:" \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --decoder_llm
    ```

5. INT8/INT4 weight-only quantization for LLaMA can be enabled as follows (take `INT4` as an example, while `INT8` is the default precision for weight-only quantization):
    ```bash
    python ../llama/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --dtype float16 \
        --output_dir tmp/trt_models/${MODEL_NAME}/int4_weightonly/1-gpu \
        --use_weight_only \
        --weight_only_precision int4

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/int4_weightonly/1-gpu \
        --output_dir trt_engines/${MODEL_NAME}/int4_weightonly/1-gpu \
        --gemm_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 924 \
        --max_output_len 100 \
        --max_multimodal_len 576
    ```

    The built engines lie in `trt_engines/${MODEL_NAME}/int4_weightonly/1-gpu`.
    You should use this directory as `--llm_engine_dir` argument to `run.py`

6. One can use LLaVA with other quantization options, like SmoothQuant and INT4 AWQ, that are supported by LLaMA.
   Instructions in [../llama/README.md](../llama/README.md) to enable SmoothQuant and INT4 AWQ can be re-used to generate
   quantized TRT engines for LLM component of LLaVA.

## Nougat

1. Download Huggingface weights

    ```bash
    export MODEL_NAME="nougat-base" # or nougat-small
    git clone https://huggingface.co/facebook/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

2. Convert Huggingface weights into TRT-LLM checkpoints and build TRT engines using scripts in `examples/enc_dec`

   Nougat uses mBART architecture but replaces the LLM encoder with a Swin Transformer encoder.
   To achieve this, we add an extra `--nougat` flag (over mBART example) to
   `bart/convert.py` and `build.py` in `examples/enc_dec`.

    ```bash
    python ../enc_dec/bart/convert.py -i tmp/hf_models/${MODEL_NAME} \
        -o tmp/trt_models/${MODEL_NAME} --weight_data_type float32 \
        --inference_tensor_para_size 1 --nougat

    python ../enc_dec/build.py \
        --model_type bart \
        --weight_dir tmp/trt_models/${MODEL_NAME}/tp1 \
        -o trt_engines/${MODEL_NAME}/1-gpu \
        --engine_name $MODEL_NAME \
        --bert_attention_plugin \
        --use_gpt_attention_plugin \
        --use_gemm_plugin \
        --dtype bfloat16 \
        --max_beam_width 1 \
        --max_batch_size 1 \
        --nougat \
        --max_output_len 100 \
        --max_multimodal_len 588 # 1 (max_batch_size) * 588 (num_visual_features)
    ```

3. Generate TensorRT engines for visual components and combine everything into final pipeline.

    ```bash
    python build_visual_engine.py --model_name ${MODEL_NAME} --model_path tmp/hf_models/${MODEL_NAME}

    python run.py \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1 \
        --nougat
    ```

## Enabling tensor parallelism for multi-GPU

The LLM part of the pipeline can be run on multiple GPUs using tensor parallelism.
The visual encoder will be replicated on each GPU and operate in a data parallel fashion.

To enable tensor parallelism, both weight conversion step (from Huggingface to FT format)
and engine building step should use additional arguments. Finally `run.py` should be prefixed
with `mpirun -n NUM_GPUS --allow-run-as-root`.

The full set of commands to enable 2-way tensor parallelism for LLaVA is:

    ```bash
    export MODEL_NAME="llava-1.5-7b-hf"

    python ../llama/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/2-gpu \
        --dtype float16 --tp_size 2

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/2-gpu \
        --output_dir trt_engines/${MODEL_NAME}/fp16/2-gpu \
        --gemm_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_output_len 512 \
        --max_multimodal_len 576

    python build_visual_engine.py --model_name ${MODEL_NAME} --model_path tmp/hf_models/${MODEL_NAME}

    mpirun -n 2 --allow-run-as-root \
        python run.py \
        --max_new_tokens 30 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir trt_engines/${MODEL_NAME}/fp16/2-gpu \
        --decoder_llm
    ```
