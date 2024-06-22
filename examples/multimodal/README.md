<!-- omit from toc -->
# Multi-Modal

This document shows how to run multimodal pipelines with TensorRT-LLM, e.g. from image+text input modalities to text output.

Multimodal models' LLM part has an additional parameter `--max_multimodal_len` compared to LLM-only build commands. Under the hood, `max_multimodal_len` and `max_prompt_embedding_table_size` are effectively the same concept, i.e., prepended/concatenated embeddings (either multimodal feature embeddings or prompt tuning embeddings) to the LLM input embeddings. The multimodal features from the visual encoder of shape `[batch_size, num_visual_features, visual_hidden_dim]` is flattened as `[batch_size * num_visual_features, visual_hidden_dim]` and passed like a prompt embedding table.

We first describe how to run each model on a single GPU. We then provide general guidelines on using tensor parallelism for the LLM part of the pipeline.

- [BLIP2-T5](#blip2-t5)
- [BLIP2-OPT](#blip2-opt)
- [CogVLM](#cogvlm)
- [Deplot](#deplot)
- [Fuyu](#fuyu)
- [Kosmos-2](#kosmos-2)
- [LLaVA and VILA](#llava-and-vila)
- [Neva](#neva)
- [Nougat](#nougat)
- [Enabling tensor parallelism for multi-GPU](#enabling-tensor-parallelism-for-multi-gpu)

## BLIP2-T5

1. Download Huggingface weights and convert original checkpoint to TRT-LLM checkpoint format
   following example in `examples/enc_dec/README.md`.

    ```bash
    export MODEL_NAME="flan-t5-xl" # also flan-t5-xxl
    git clone https://huggingface.co/google/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

    python ../enc_dec/convert_checkpoint.py --model_type t5 \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/bfloat16 \
        --tp_size 1 \
        --pp_size 1 \
        --weight_data_type float32 \
        --dtype bfloat16 \
        --max_multimodal_len 256 # 8 (max_batch_size) * 32 (num_visual_features)
    ```

2. Build TRT-LLM engine from TRT-LLM checkpoint

    ```bash
    trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/bfloat16/tp1/pp1/encoder \
        --output_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1/encoder \
        --paged_kv_cache disable \
        --moe_plugin disable \
        --enable_xqa disable \
        --use_custom_all_reduce disable \
        --gemm_plugin bfloat16 \
        --bert_attention_plugin bfloat16 \
        --gpt_attention_plugin bfloat16 \
        --remove_input_padding enable \
        --context_fmha disable \
        --max_beam_width 1 \
        --max_batch_size 8 \
        --max_output_len 100 \
        --max_input_len 924 \
        --max_multimodal_len 256 # 8 (max_batch_size) * 32 (num_visual_features)

    # Same command for decoder but don't set --max_multimodal_len
    trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/bfloat16/tp1/pp1/decoder \
        --output_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1/decoder \
        --paged_kv_cache disable \
        --moe_plugin disable \
        --enable_xqa disable \
        --use_custom_all_reduce disable \
        --gemm_plugin bfloat16 \
        --bert_attention_plugin bfloat16 \
        --gpt_attention_plugin bfloat16 \
        --remove_input_padding enable \
        --context_fmha disable \
        --max_beam_width 1 \
        --max_batch_size 8 \
        --max_output_len 100 \
        --max_encoder_input_len 924 \
        --max_input_len 1
    ```

    **NOTE**: `max_multimodal_len = max_batch_size * num_visual_features`, so if you change max_batch_size, max multimodal length **MUST** be changed accordingly.

    The built T5 engines are located in `./tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1`.

3. Build TensorRT engines for visual components

    ```bash
    python build_visual_engine.py --model_type ${MODEL_NAME} --model_path tmp/hf_models/${MODEL_NAME} --max_batch_size 8
    ```

    The built engines are located in `./visual_engines/${MODEL_NAME}`.

    To run the BLIP2 pipeline with batch size > 1, change `--max_batch_size` argument to `build_visual_engine.py` accordingly.

4. Assemble everything into BLIP2 pipeline

    ```bash
    python run.py \
        --max_new_tokens 30 \
        --input_text "Question: which city is this? Answer:" \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1
    ```

## BLIP2-OPT

OPT pipeline needs few minor changes from T5 pipeline

1. Convert Huggingface weights to TRT-LLM checkpoint format following `examples/opt/README.md`.

2. Use `trtllm-build` command to build TRT-LLM engine for OPT.

3. The full list of commands is as follows:

    ```bash
    export MODEL_NAME="opt-2.7b" # also opt-6.7b
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

    python build_visual_engine.py --model_type ${MODEL_NAME} --model_path tmp/hf_models/${MODEL_NAME}

    python run.py \
        --max_new_tokens 30 \
        --input_text "Question: which city is this? Answer:" \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
    ```

4. INT8/INT4 weight-only quantization for OPT can be enabled using commands as follows (take `INT4` as an example, while `INT8` is the default precision for weight-only quantization):
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

## CogVLM

Currently, CogVLM only support bfloat16 precision and doesn't support `remove_input_padding` feature.

1. Download Huggingface weights

    ```bash
    export MODEL_NAME="cogvlm-chat-hf"
    git clone https://huggingface.co/THUDM/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    export TOKENIZER_NAME="vicuna-7b-v1.5"
    git clone https://huggingface.co/lmsys/${TOKENIZER_NAME} tmp/hf_models/${TOKENIZER_NAME}
    ```

    Because currently onnx doesn't support `xops.memory_efficient_attention`, we need to modify some source code of the huggingface CogVLM.
    ```
    cd tmp/hf_models/${MODEL_NAME}
    sed -i '4s/.*//;40s/.*/        out = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()/;41s/.*//;42s/.*//' visual.py   # It will replace memory_efficient_attention with some basic ops
    ```

2. Convert Huggingface weights into TRT-LLM checkpoints and build TRT engines using scripts in `examples/cogvlm`

   CogVLM uses a Vit encoder as LLM encoder and a modified Llama as decoder.

    ```bash
    python ../cogvlm/convert_checkpoint.py --model_dir tmp/hf_models/${MODEL_NAME}  --output_dir ./tllm_checkpoint_1gpu_bf16 --dtype bfloat16 --use_prompt_tuning

    trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_bf16  \
    --output_dir ./tmp/cogvlm/trt_engines/bf16/1-gpu \
    --gemm_plugin bfloat16 \
    --gpt_attention_plugin bfloat16 \
    --context_fmha_fp32_acc enable \
    --remove_input_padding disable \
    --max_batch_size 48 \
    --max_input_len 2048 \
    --max_output_len 1024 \
    --paged_kv_cache disable \
    --use_custom_all_reduce disable \
    --enable_xqa disable \
    --bert_attention_plugin disable \
    --moe_plugin disable \
    --max_multimodal_len 61440 # 48 (max_batch_size) * 1280 (max_num_visual_features)
    ```

3. Generate TensorRT engines for visual components and combine everything into final pipeline.

    ```bash
    python build_visual_engine.py --model_type cogvlm --model_path tmp/hf_models/${MODEL_NAME} --max_batch_size 48

    python run.py \
    --max_new_tokens 1000 \
    --input_text " [INST] please describe this image in detail [/INST] " \
    --hf_model_dir tmp/hf_models/${TOKENIZER_NAME} \
     --visual_engine_dir visual_engines/${MODEL_NAME} \
     --llm_engine_dir tmp/cogvlm/trt_engines/bf16/1-gpu \
     --batch_size 1 \
     --top_p 0.4 \
     --top_k 1 \
     --temperature 0.2 \
     --repetition_penalty 1.2
    ```

## Deplot

1. Download Huggingface weights and convert original checkpoint to TRT-LLM checkpoint format
   following example in `examples/enc_dec/README.md`.

    ```bash
    export MODEL_NAME="deplot"
    git clone https://huggingface.co/google/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

    python ../enc_dec/convert_checkpoint.py --model_type pix2struct \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/float16 \
        --tp_size 1 \
        --pp_size 1 \
        --weight_data_type float32 \
        --dtype float16
    ```

2. Build TRT-LLM engine from TRT-LLM checkpoint

    ```bash
    trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/float16/tp1/pp1/decoder \
        --output_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/float16/tp1/decoder \
        --paged_kv_cache disable \
        --moe_plugin disable \
        --enable_xqa disable \
        --use_custom_all_reduce disable \
        --gemm_plugin float16 \
        --bert_attention_plugin float16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --context_fmha disable \
        --max_beam_width 1 \
        --max_batch_size 8 \
        --max_output_len 510 \
        --max_encoder_input_len 2048 \
        --max_input_len 1
    ```

    The built deplot engines are located in `./tmp/trt_engines/${MODEL_NAME}/1-gpu/float16/tp1`.

3. Build TensorRT engines for visual components

    ```bash
    python build_visual_engine.py --model_type pix2struct --model_path tmp/hf_models/${MODEL_NAME} --max_batch_size 8
    ```

    The built engines are located in `./visual_engines/${MODEL_NAME}`.

    To run the deplot pipeline with batch size > 1, change `--max_batch_size` argument to `build_visual_engine.py` accordingly.

4. Assemble everything into deplot pipeline

    ```bash
    python run.py \
        --max_new_tokens 100 \
        --input_text "" \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/float16/tp1
    ```

## Fuyu

1. Download Huggingface weights

    ```bash
    export MODEL_NAME="fuyu-8b"
    git clone https://huggingface.co/adept/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

2. Convert Huggingface weights into TRT-LLM checkpoints and build TRT engines using scripts in `examples/gpt`.
    The LLM portion of Fuyu uses a Persimmon model
    ```bash
    python ../gpt/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --dtype float16 \
        --gpt_variant persimmon

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --gemm_plugin float16 \
        --use_fused_mlp \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_output_len 512 \
        --max_multimodal_len 2048
    ```

3. Generate TensorRT engines for visual components and combine everything into final pipeline.

    ```bash
    python build_visual_engine.py --model_type fuyu --model_path tmp/hf_models/${MODEL_NAME}

    python run.py \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1
    ```

## Kosmos-2

1. Download Huggingface weights

    ```bash
    export MODEL_NAME="kosmos-2"
    git clone https://huggingface.co/microsoft/kosmos-2-patch14-224 tmp/hf_models/${MODEL_NAME}
    ```

2. Convert Huggingface weights into TRT-LLM checkpoints and build TRT engines using scripts in `examples/gpt`.
    ```bash
    python ../gpt/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --dtype float16 \
        --gpt_variant ${MODEL_NAME}

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 512 \
        --max_output_len 512 \
        --max_multimodal_len 64 # 1 (max_batch_size) * 64 (num_visual_features)
    ```

3. Generate TensorRT engines for visual components and combine everything into final pipeline.

    ```bash
    python build_visual_engine.py --model_type kosmos-2 --model_path tmp/hf_models/${MODEL_NAME}

    python run.py \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1
    ```

## LLaVA and VILA

[LLaVA](https://github.com/haotian-liu/LLaVA) and [VILA](https://github.com/Efficient-Large-Model/VILA) are both visual language models (VLM) that can be deployed in TensorRT-LLM with many quantization options.

1. Download Huggingface model weights. These models have both visual and LLM components
   unlike BLIP2 example which downloads only LLM components from Huggingface.

    For LLaVA,

    ```bash
        export MODEL_NAME="llava-1.5-7b-hf" # also llava-1.5-13b-hf
        git clone https://huggingface.co/llava-hf/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

    For VILA, we need a few more steps until it is added to HF model zoo

    ```bash
        # clone original VILA repo
        export VILA_PATH="tmp/hf_models/VILA"
        git clone https://github.com/Efficient-Large-Model/VILA.git ${VILA_PATH}

        # download VILA checkpoints
        export MODEL_NAME="vila-7B" # also vila-2.7B, vila-13B
        git clone https://huggingface.co/Efficient-Large-Model/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

        # turn off delay_load to allow model component access
        sed -i 's/delay_load=True/delay_load=False/g' ${VILA_PATH}/llava/model/llava_arch.py
        # line manipulation to enable AWQ. otherwise need to replace HF's llama implementation
        sed -i '/vision_tower = self.get_vision_tower()/a \        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)' ${VILA_PATH}/llava/model/llava_arch.py
        sed -i 's/seqlens_in_batch=sorted_seqlens_in_batch/#seqlens_in_batch=sorted_seqlens_in_batch/g' ${VILA_PATH}/llava/model/language_model/llava_llama.py
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
        --use_fused_mlp \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_output_len 512 \
        --max_multimodal_len 576 # 1 (max_batch_size) * 576 (num_visual_features)
    ```

    Note: do not use `--use_fused_mlp` flag in quantization mode.

3. Build TensorRT engines for visual components

    ```bash
    python build_visual_engine.py --model_path tmp/hf_models/${MODEL_NAME} --model_type llava # for LLaVA
    python build_visual_engine.py --model_path tmp/hf_models/${MODEL_NAME} --model_type vila --vila_path ${VILA_PATH} # for VILA
    ```

    ```bash
    python run.py \
        --max_new_tokens 30 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --input_text "Question: which city is this? Answer:" # or "Please describe the traffic condition." for VILA
    ```

    Note: use `--run_profiling` for performance measurement, use `--check_accuracy` for accuracy check.

4. (Optional) INT8/INT4 weight-only quantization for LLaMA can be enabled as follows (take `INT4` as an example, while `INT8` is the default precision for weight-only quantization):
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
        --max_input_len 1024 \
        --max_output_len 100 \
        --max_multimodal_len 576
    ```

    The built engines lie in `trt_engines/${MODEL_NAME}/int4_weightonly/1-gpu`.
    You should use this directory as `--llm_engine_dir` argument to `run.py`

5. (Optional) One can also use LLaVA/VILA with other quantization options, like SmoothQuant and INT4 AWQ, that are supported by LLaMA.
   Instructions in LLaMA [README](../llama/README.md) to enable SmoothQuant and INT4 AWQ can be re-used to generate
   quantized TRT engines for LLM component of LLaVA/VILA.

   For example,

   ```bash
   python ../quantization/quantize.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/int4_awq/1-gpu \
        --dtype float16 \
        --qformat int4_awq \
        --calib_size 32

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/int4_awq/1-gpu \
        --output_dir trt_engines/${MODEL_NAME}/int4_awq/1-gpu \
        --gemm_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 1024 \
        --max_output_len 100 \
        --max_multimodal_len 576
   ```

## NeVA

[NeVA](https://docs.nvidia.com/nemo-framework/user-guide/latest/multimodalmodels/neva/index.html) is a groundbreaking addition to the NeMo Multimodal ecosystem. This model seamlessly integrates large language-centric models with a vision encoder, that can be deployed in TensorRT-LLM.

1. Generate TRT-LLM engine for NVGPT following example in `examples/gpt/README.md`. To adhere to the NVGPT conventions of the conversion script, some layer keys have to be remapped using `--nemo_rename_key`.

    ```bash
    export MODEL_NAME="neva"
    python ../gpt/convert_checkpoint.py \
	--nemo_ckpt_path ./${MODEL_NAME}.nemo \
	--dtype bfloat16 \
	--output_dir tmp/trt_models/${MODEL_NAME} \
	--nemo_rename_key model:model.language_model \
        attention.linear_qkv.layer_norm_bias:input_layernorm.bias \
        attention.linear_qkv.layer_norm_weight:input_layernorm.weight \
        mlp.linear_fc1.layer_norm_bias:post_attention_layernorm.bias \
        mlp.linear_fc1.layer_norm_weight:post_attention_layernorm.weight \
        linear_qkv:query_key_value \
        linear_fc1:dense_h_to_4h \
        linear_fc2:dense_4h_to_h \
        linear_proj:dense \
        decoder:encoder

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME} \
        --output_dir trt_engines/${MODEL_NAME}/bf16/1-gpu \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_output_len 512 \
        --max_multimodal_len 729 # 1 (max_batch_size) * 729 (num_visual_features)
    ```

2. Build TensorRT engines for visual components

    ```bash
    python build_visual_engine.py --model_path ./${MODEL_NAME}.nemo --model_type neva
    ```

    ```bash
    python run.py \
        --max_new_tokens 30 \
        --hf_model_dir tmp/trt_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir trt_engines/${MODEL_NAME}/bf16/1-gpu \
        --input_text "Question: which city is this? Answer:"
    ```

    Note: use `--run_profiling` for performance measurement, use `--check_accuracy` for accuracy check.



## Nougat

1. Download Huggingface weights

    ```bash
    export MODEL_NAME="nougat-base" # also nougat-small
    git clone https://huggingface.co/facebook/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

2. Convert Huggingface weights into TRT-LLM checkpoints and build TRT engines using scripts in `examples/enc_dec`

   Nougat uses mBART architecture but replaces the LLM encoder with a Swin Transformer encoder.
   To achieve this, we add an extra `--nougat` flag (over mBART example) to
   `convert_checkpoint.py` in `examples/enc_dec` and `trtllm-build`.

    ```bash
    python ../enc_dec/convert_checkpoint.py --model_type bart \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/bfloat16 \
        --tp_size 1 \
        --pp_size 1 \
        --weight_data_type float32 \
        --dtype bfloat16 \
        --nougat

    trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/bfloat16/tp1/pp1/decoder \
        --output_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1/decoder \
        --paged_kv_cache disable \
        --moe_plugin disable \
        --enable_xqa disable \
        --use_custom_all_reduce disable \
        --gemm_plugin bfloat16 \
        --bert_attention_plugin bfloat16 \
        --gpt_attention_plugin bfloat16 \
        --remove_input_padding enable \
        --max_beam_width 1 \
        --max_batch_size 1 \
        --max_output_len 100 \
        --max_input_len 1 \
        --max_encoder_input_len 588 # 1 (max_batch_size) * 588 (num_visual_features)
    ```

3. Generate TensorRT engines for visual components and combine everything into final pipeline.

    ```bash
    python build_visual_engine.py --model_type nougat --model_path tmp/hf_models/${MODEL_NAME}

    python run.py \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1 \
    ```

    Note: Nougat models usually do not need a text prompt.


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

    python build_visual_engine.py --model_type llava --model_path tmp/hf_models/${MODEL_NAME}

    mpirun -n 2 --allow-run-as-root \
        python run.py \
        --max_new_tokens 30 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir visual_engines/${MODEL_NAME} \
        --llm_engine_dir trt_engines/${MODEL_NAME}/fp16/2-gpu \
    ```
