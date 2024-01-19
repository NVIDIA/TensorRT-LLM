# Multi-Modal

This document shows how to run multimodal pipelines with TensorRT-LLM, e.g. from image+text input modalities to text output.

## BLIP + T5

1. Download Huggingface weights and convert original checkpoint to TRT-LLM checkpoint format
   following example in `examples/enc_dec/README.md`.

    ```bash
    export MODEL_NAME=flan-t5-xl
    git clone https://huggingface.co/google/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

    python ../enc_dec/t5/convert.py -i tmp/hf_models/${MODEL_NAME} \
            -o tmp/hf_models/${MODEL_NAME} --weight_data_type float32 \
            --inference_tensor_para_size 1
    ```

2. Build TRT-LLM engine from TRT-LLM checkpoint

    Add an additional parameter `--max_prompt_embedding_table_size` compared to LLM build commands.
    `max_prompt_embedding_table_size = visual_feature_dim * max_batch_size`,
    so if you change max_batch_size, prompt table size must be reset accordingly.

    ```bash
    python ../enc_dec/build.py --model_type t5 \
                --weight_dir tmp/hf_models/${MODEL_NAME}/tp1 \
                --output_dir trt_engines/${MODEL_NAME}/1-gpu \
                --engine_name ${MODEL_NAME} \
                --remove_input_padding \
                --use_bert_attention_plugin \
                --use_gpt_attention_plugin \
                --use_gemm_plugin \
                --use_rmsnorm_plugin \
                --dtype bfloat16 \
                --max_beam_width 1 \
                --max_batch_size 8 \
                --max_prompt_embedding_table_size 256 # 32 (visual_feature_dim) * 8 (max_batch_size)
    ```

    The built T5 engines are located in `./trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1`.

3.  Build TensorRT engines for visual components

    ```bash
    python build_visual_engine.py --model_name ${MODEL_NAME} --model_path tmp/hf_models/${MODEL_NAME}
    ```

    The built engines are located in `./visual_engines/${MODEL_NAME}`.

4. Assemble everything into BLIP pipeline

    ```bash
    python run.py \
              --blip_encoder \
              --max_new_tokens 30 \
              --input_text "Question: which city is this? Answer:" \
              --hf_model_dir tmp/hf_models/${MODEL_NAME} \
              --visual_engine_dir visual_engines/${MODEL_NAME} \
              --llm_engine_dir trt_engines/${MODEL_NAME}/1-gpu/bfloat16/tp1
    ```

## BLIP + OPT

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
           --output_dir tmp/hf_models/${MODEL_NAME}/c-model/fp16/1-gpu

    trtllm-build \
             --checkpoint_dir tmp/hf_models/${MODEL_NAME}/c-model/fp16/1-gpu \
             --output_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
             --use_gpt_attention_plugin float16 \
             --use_gemm_plugin float16 \
             --max_input_len 924 \
             --max_output_len 100 \
             --max_beam_width 1 \
             --max_batch_size 8 \
             --max_prompt_embedding_table_size 256

    python build_visual_engine.py --model_name ${MODEL_NAME} --model_path tmp/hf_models/${MODEL_NAME}

    python run.py \
              --blip_encoder \
              --max_new_tokens 30 \
              --input_text "Question: which city is this? Answer:" \
              --hf_model_dir tmp/hf_models/${MODEL_NAME} \
              --visual_engine_dir visual_engines/${MODEL_NAME} \
              --llm_engine_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
              --decoder_llm
    ```

## LLaVA

1. Download and install LLaVA library. Rebuild TRT-LLM after installing LLaVA.

    ```bash
    git clone https://github.com/haotian-liu/LLaVA.git
    sudo pip install -e LLaVA
    ```

2. Download Huggingface model weights. This model has both LLM and visual components
   unlike BLIP example which downloads only LLM components from Huggingface.

    ```bash
    export MODEL_NAME="llava-v1.5-7b"
    git clone https://huggingface.co/liuhaotian/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

3. Generate TRT-LLM engine for LLaMA following example in `examples/llama/README.md`

    ```bash
    python ../llama/build.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir trt_engines/${MODEL_NAME}/fp16/1-gpu
        --dtype float16 \
        --remove_input_padding \
        --use_gpt_attention_plugin float16 \
        --enable_context_fmha \
        --use_gemm_plugin float16 \
        --max_batch_size 1 \
        --max_prompt_embedding_table_size 576 # 576 (visual_feature_dim) * 1 (max_batch_size)
    ```

4.  Build TensorRT engines for visual components

    ```bash
    python build_visual_engine.py --model_name ${MODEL_NAME} --model_path tmp/hf_models/${MODEL_NAME}
    ```

5. Add `--decoder-llm` argument to inference script, since LLaMA is a decoder-only LLM.

    ```bash
    python run.py \
              --max_new_tokens 30 \
              --input_text "Question: which city is this? Answer:" \
              --hf_model_dir tmp/hf_models/${MODEL_NAME} \
              --visual_engine_dir visual_engines/${MODEL_NAME} \
              --llm_engine_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
              --decoder_llm
    ```
