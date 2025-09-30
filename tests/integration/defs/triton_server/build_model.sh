#!/usr/bin/bash

install_requirements() {
    pip install -r requirements.txt
    #This is to WAR an issue with latest transformers package not being compatible with sentencepiece 0.1.99 apparently
    pip install sentencepiece --upgrade
}

MODEL=$1

GPT2=$LLM_MODELS_ROOT/gpt2
GPT2_MEDIUM=$LLM_MODELS_ROOT/gpt2-medium
GPT2_NEXT_PTUNING=$LLM_MODELS_ROOT/email_composition
OPT_125M=$LLM_MODELS_ROOT/opt-125m
LLAMA=$LLM_MODELS_ROOT/llama-models/llama-7b-hf
GPTJ=$LLM_MODELS_ROOT/gpt-j-6b
MISTRAL=$LLM_MODELS_ROOT/mistral-7b-v0.1
GPT_2B=$LLM_MODELS_ROOT/GPT-2B-001_bf16_tp1.nemo
GPT_2B_LORA=$LLM_MODELS_ROOT/lora/gpt-next-2b
VICUNA=$LLM_MODELS_ROOT/vicuna-7b-v1.3
MEDUSA_VICUNA=$LLM_MODELS_ROOT/medusa-vicuna-7b-v1.3/
EAGLE_VICUNA=$LLM_MODELS_ROOT/EAGLE-Vicuna-7B-v1.3/
BART=$LLM_MODELS_ROOT/bart-large-cnn/
T5=$LLM_MODELS_ROOT/t5-small/
BLIP2_OPT_2_7B=$LLM_MODELS_ROOT/blip2-opt-2.7b
LLAVA_7B=$LLM_MODELS_ROOT/llava-1.5-7b-hf
VILA1_5_3B=$LLM_MODELS_ROOT/vila/VILA1.5-3b
VILA_PATH=$LLM_MODELS_ROOT/vila/VILA
LLAMA_3_2_11B_VISION=$LLM_MODELS_ROOT/llama-3.2-models/Llama-3.2-11B-Vision-Instruct
WHISPER_LAREGE_V3=$LLM_MODELS_ROOT/whisper-models/large-v3
LLAVA_ONEVISION_7B=$LLM_MODELS_ROOT/llava-onevision-qwen2-7b-ov-hf
QWEN2_VL_7B=$LLM_MODELS_ROOT/Qwen2-VL-7B-Instruct
set -e

pkill -9 -f tritonserver || true

pushd $LLM_ROOT/

# install deps
pip3 install -r requirements-dev.txt

if [ "$MODEL" = "gpt" ] || [ "$MODEL" = "gpt-disaggregated-serving-bls" ]; then

    # GPT2
    pushd examples/models/core/gpt

    install_requirements

    echo "Convert GPT from HF"
    python3 convert_checkpoint.py --model_dir ${GPT2} --dtype float16 --output_dir ./c-model/gpt2/fp16

    echo "Build GPT: float16 | remove_input_padding"
    trtllm-build --checkpoint_dir ./c-model/gpt2/fp16 \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --context_fmha enable \
        --remove_input_padding enable \
        --max_batch_size 8 \
        --max_seq_len 1024 \
        --max_num_tokens 7392 \
        --output_dir trt_engine/gpt2/fp16/1-gpu/

    popd # examples/models/core/gpt

fi

if [ "$MODEL" = "opt" ]; then

    pushd examples/models/contrib/opt

    install_requirements

    echo "Convert OPT from HF"
    python3 convert_checkpoint.py --model_dir ${OPT_125M} --dtype float16 --output_dir ./c-model/opt-125m/fp16

    echo "OPT builder"
    trtllm-build --checkpoint_dir ./c-model/opt-125m/fp16  \
                --gemm_plugin float16 \
                --gpt_attention_plugin float16 \
                --context_fmha=enable \
                --max_batch_size 8 \
                --max_seq_len 1024 \
                --max_num_tokens 7392 \
                --output_dir trt_engine/opt-125m/fp16/1-gpu/


    popd # examples/models/contrib/opt

fi

if [ "$MODEL" = "llama" ]; then

    pushd examples/models/core/llama

    install_requirements

    echo "Convert LLaMA from HF"
    python3 convert_checkpoint.py --dtype float16 --n_layer 2 --output_dir ./c-model/llama-7b/fp16

    echo "Build LLaMA"
    trtllm-build --model_config ./c-model/llama-7b/fp16/config.json  \
        --context_fmha=enable \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --max_batch_size 8 \
        --output_dir llama_outputs

    popd # examples/models/core/llama

fi

if [ "$MODEL" = "mistral" ]; then

    pushd examples/models/core/llama

    install_requirements

    echo "Convert Mistral from HF"
    python3 convert_checkpoint.py --dtype float16 \
        --n_layer 2 --n_positions 32768 \
        --output_dir ./c-model/mistral-7b/fp16

    echo "Build Mistral"
    trtllm-build --model_config ./c-model/mistral-7b/fp16/config.json  \
        --context_fmha=enable \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --max_input_len 8192 \
        --max_batch_size 8 \
        --max_num_tokens 8192 \
        --output_dir mistral_7b_outputs

    popd # examples/models/core/llama

fi

if [ "$MODEL" = "mistral-ib" ]; then

    pushd examples/models/core/llama

    install_requirements

    echo "Convert Mistral from HF"
    python3 convert_checkpoint.py --dtype float16 \
        --model_dir ${MISTRAL} \
        --output_dir ./c-model/mistral-7b/fp16

    echo "Build Mistral with inflight batching"
    trtllm-build --checkpoint_dir ./c-model/mistral-7b/fp16/ \
        --context_fmha=enable \
        --remove_input_padding=enable \
        --kv_cache_type=paged \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --max_batch_size 1 \
        --max_seq_len 9216 \
        --use_paged_context_fmha disable \
        --max_beam_width 2 \
        --output_dir ib_mistral_7b_outputs

    popd # examples/models/core/llama

fi

if [ "$MODEL" = "gptj" ]; then

    pushd examples/models/contrib/gptj

    install_requirements

    echo "Convert GPT-J from HF"
    python3 convert_checkpoint.py --dtype float16 --n_layer 2 --output_dir ./c-model/gpt-j-6b/fp16

    echo "Build GPT-J"
    trtllm-build --model_config ./c-model/gpt-j-6b/fp16/config.json  \
        --context_fmha=enable \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --max_batch_size 8 \
        --output_dir gptj_outputs

    popd # examples/models/contrib/gptj

fi

if [ "$MODEL" = "gpt-ib" ]; then

    # GPT2
    pushd examples/models/core/gpt

    install_requirements

    echo "Convert GPT from HF"
    python3 convert_checkpoint.py --model_dir ${GPT2} --dtype float16 --output_dir ./c-model/gpt2/fp16

    echo "Build GPT: float16"
    trtllm-build --checkpoint_dir ./c-model/gpt2/fp16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --kv_cache_type paged \
        --use_paged_context_fmha enable \
        --gemm_plugin float16 \
        --max_batch_size 8 \
        --max_num_tokens 7392 \
        --gather_generation_logits \
        --output_dir trt_engine/gpt2-ib/fp16/1-gpu/

    popd # examples/models/core/gpt

fi

if [ "$MODEL" = "gpt-ib-lad" ]; then

    # GPT2
    pushd examples/models/core/gpt

    # install_requirements

    echo "Convert GPT from HF"
    python3 convert_checkpoint.py --model_dir ${GPT2} --dtype float16 --output_dir ./c-model/gpt2/fp16

    echo "Build GPT: float16"
    trtllm-build --checkpoint_dir ./c-model/gpt2/fp16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --kv_cache_type paged \
        --use_paged_context_fmha enable \
        --gemm_plugin float16 \
        --max_batch_size 8 \
        --max_num_tokens 7392 \
        --gather_generation_logits \
        --max_draft_len=83 \
        --speculative_decoding_mode=lookahead_decoding \
        --output_dir trt_engine/gpt2-ib-lad/fp16/1-gpu/

    popd # examples/models/core/gpt

fi

if [ "$MODEL" = "bart-ib" ] || [ "$MODEL" = "t5-ib" ]; then

    pushd examples/models/core/enc_dec

    if [ "$MODEL" = "bart-ib" ]; then
      MODEL_DIR=${BART}
      MODEL_TYPE="bart"
    elif [ "$MODEL" = "t5-ib" ]; then
      MODEL_DIR=${T5}
      MODEL_TYPE="t5"
    fi
    echo "Convert ${MODEL_TYPE} from HF"
    python3 convert_checkpoint.py --model_type ${MODEL_TYPE} --model_dir ${MODEL_DIR} --dtype float16 --output_dir ./c-model/${MODEL}/fp16

    echo "Build Encoder: "
    trtllm-build --checkpoint_dir ./c-model/${MODEL}/fp16/encoder \
    --output_dir trt_engine/${MODEL}/fp16/1-gpu/encoder \
    --kv_cache_type disabled --moe_plugin disable \
    --max_beam_width 1 \
    --max_batch_size 8 --max_input_len 512 --max_seq_len 512 \
    --gemm_plugin float16 \
    --bert_attention_plugin float16 \
    --gpt_attention_plugin float16 \
    --remove_input_padding enable --context_fmha disable


    echo "Build Decoder:"
    trtllm-build --checkpoint_dir ./c-model/${MODEL}/fp16/decoder \
    --output_dir trt_engine/${MODEL}/fp16/1-gpu/decoder \
    --moe_plugin disable \
    --max_beam_width 1 \
    --max_batch_size 8 --max_input_len 1 --max_seq_len 512 --max_encoder_input_len 512 \
    --gemm_plugin float16 \
    --bert_attention_plugin float16 \
    --gpt_attention_plugin float16 \
    --remove_input_padding enable --context_fmha disable

    popd # examples/models/core/enc_dec

fi


if [ "$MODEL" = "gpt-medium-ib" ]; then

    # GPT2
    pushd examples/models/core/gpt

    install_requirements

    echo "Convert GPT2 medium from HF"
    python3 convert_checkpoint.py --model_dir ${GPT2_MEDIUM} --dtype float16 --output_dir ./c-model/gpt2-medium/fp16

    echo "Build GPT2 medium control: float16"
    trtllm-build --checkpoint_dir ./c-model/gpt2-medium/fp16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --kv_cache_type paged \
        --gemm_plugin float16 \
        --context_fmha enable \
        --use_paged_context_fmha enable \
        --max_batch_size 8 \
        --max_num_tokens 7392 \
        --gather_generation_logits \
        --output_dir trt_engine/gpt2-medium-ib/fp16/1-gpu/

    echo "Build GPT2 medium target: float16"
    trtllm-build --checkpoint_dir ./c-model/gpt2-medium/fp16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --kv_cache_type paged \
        --gemm_plugin float16 \
        --context_fmha enable \
        --use_paged_context_fmha enable \
        --max_draft_len 5 \
        --speculative_decoding_mode draft_tokens_external \
        --max_batch_size 8 \
        --max_num_tokens 7392 \
        --gather_generation_logits \
        --output_dir trt_engine/gpt2-medium-ib-target/fp16/1-gpu/

    popd # examples/models/core/gpt

fi

if [ "$MODEL" = "gpt-ib-ptuning" ]; then

    # GPT2
    pushd examples/models/core/gpt

    install_requirements

    echo "Convert GPT from NeMo"
    python3 convert_checkpoint.py --nemo_ckpt_path ${GPT2_NEXT_PTUNING}/megatron_converted_8b_tp4_pp1.nemo --dtype float16 --output_dir ./c-model/email_composition/fp16

    echo "Convert ptuning table"
    python3 nemo_prompt_convert.py -i ${GPT2_NEXT_PTUNING}/email_composition.nemo -o email_composition.npy

    cp ${GPT2_NEXT_PTUNING}/input.csv ./

    echo "Build GPT: float16"
    trtllm-build --checkpoint_dir ./c-model/email_composition/fp16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --kv_cache_type paged \
        --gemm_plugin float16 \
        --context_fmha enable \
        --max_batch_size 4 --max_seq_len 256 --max_beam_width 1 \
        --max_num_tokens 512 \
        --output_dir trt_engine/email_composition/fp16/1-gpu/ \
        --max_prompt_embedding_table_size 300

    popd # examples/models/core/gpt

fi

if [ "$MODEL" = "gpt-2b-ib-lora" ]; then

    # GPT-2B
    pushd examples/models/core/gpt

    install_requirements

    echo "Convert GPT from NeMo"
    python3 convert_checkpoint.py --nemo_ckpt_path ${GPT_2B} --dtype float16 --output_dir ./c-model/gpt-2b-lora/fp16

    echo "Build GPT: float16"
    trtllm-build --checkpoint_dir ./c-model/gpt-2b-lora/fp16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --kv_cache_type paged \
        --gemm_plugin float16 \
        --lora_plugin float16 \
        --lora_dir ${GPT_2B_LORA}/gpt2b_lora-900.nemo \
        --lora_ckpt_source nemo \
        --lora_target_modules attn_qkv \
        --max_batch_size 8 \
        --max_num_tokens 7392 \
        --output_dir trt_engine/gpt-2b-lora-ib/fp16/1-gpu/

    python3 nemo_lora_convert.py -i ${GPT_2B_LORA}/gpt2b_lora-900.nemo \
        -o gpt-2b-lora-train-900 --write-cpp-runtime-tensors --storage-type float16
    python3 nemo_lora_convert.py -i ${GPT_2B_LORA}/gpt2b_lora-900.nemo \
        -o gpt-2b-lora-train-900-tllm --storage-type float16
    cp ${GPT_2B_LORA}/gpt2b_lora-900.nemo .

    cp ${GPT_2B_LORA}/input.csv .

    popd # examples/models/core/gpt
fi

if [ "$MODEL" = "gpt-gather-logits" ]; then

    # GPT2
    pushd examples/models/core/gpt

    install_requirements

    echo "Convert GPT from HF"
    python3 convert_checkpoint.py --model_dir ${GPT2} --dtype float16 --output_dir ./c-model/gpt2/fp16

    echo "Build GPT: float16 | gather_all_token_logits"
    trtllm-build --checkpoint_dir ./c-model/gpt2/fp16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --kv_cache_type paged \
        --gemm_plugin float16 \
        --context_fmha enable \
        --max_batch_size 128 --max_seq_len 600 \
        --gather_all_token_logits \
        --output_dir trt_engine/gpt2-gather-logits/fp16/1-gpu/ \
        --max_num_tokens 38400

    popd # examples/models/core/gpt

fi

if [ "$MODEL" = "medusa" ]; then

    # Medusa
    pushd examples/medusa

    install_requirements

    echo "Convert Medusa from HF"
    python convert_checkpoint.py --model_dir ${VICUNA} \
                            --medusa_model_dir ${MEDUSA_VICUNA} \
                            --output_dir ./tllm_checkpoint_1gpu_medusa \
                            --dtype float16 \
                            --num_medusa_heads 4

    echo "Build Medusa: float16"
    trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_medusa \
             --output_dir ./tmp/medusa/7B/trt_engines/fp16/1-gpu/ \
             --gemm_plugin float16 \
             --speculative_decoding_mode medusa \
             --max_batch_size 8 --max_seq_len 600 \
             --max_num_tokens 2400

    popd # examples/medusa

fi

if [ "$MODEL" = "eagle" ]; then

    # Eagle
    pushd examples/eagle

    install_requirements

    echo "Convert Eagle from HF"
    python convert_checkpoint.py --model_dir ${VICUNA} \
                            --eagle_model_dir ${EAGLE_VICUNA} \
                            --output_dir ./tllm_checkpoint_1gpu_eagle \
                            --dtype float16 \
                            --num_eagle_layers 4

    echo "Build Eagle: float16"
    trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_eagle \
             --output_dir ./tmp/eagle/7B/trt_engines/fp16/1-gpu/ \
             --gemm_plugin float16 \
             --speculative_decoding_mode eagle \
             --max_batch_size 8 --max_seq_len 600 \
             --max_num_tokens 2400

    popd # examples/eagle

fi

if [ "$MODEL" = "gpt-gather-generation-logits" ]; then

    # GPT2
    pushd examples/models/core/gpt

    install_requirements

    echo "Convert GPT from HF"
    python3 convert_checkpoint.py --model_dir ${GPT2} --dtype float16 --output_dir ./c-model/gpt2/fp16

    # draft model, only gather_generation_logits
    echo "Build GPT: float16 | gather_all_token_logits"
    trtllm-build --checkpoint_dir ./c-model/gpt2/fp16 \
        --max_batch_size 4 \
        --max_seq_len 640 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --kv_cache_type paged \
        --context_fmha enable \
        --max_num_tokens 38400 \
        --use_paged_context_fmha enable \
        --gather_generation_logits \
        --output_dir trt_engine/gpt2-draft-gather-generation-logits/fp16/1-gpu/

    popd # examples/models/core/gpt

fi

if [ "$MODEL" = "blip2-opt" ]; then

    pushd examples/models/core/multimodal

    echo "Convert OPT from HF"
    python3 ../../contrib/opt/convert_checkpoint.py --model_type blip2 --model_dir ${BLIP2_OPT_2_7B} --dtype float16 --output_dir ./c-model/opt-2.7b/fp16

    echo "OPT builder"
    trtllm-build --checkpoint_dir ./c-model/opt-2.7b/fp16 \
                --gemm_plugin float16 \
                --max_beam_width 1 \
                --max_batch_size 8 \
                --max_multimodal_len 256 \
                --max_input_len 924 \
                --max_seq_len 1024 \
                --use_paged_context_fmha enable \
                --output_dir trt_engines/opt-2.7b/fp16/1-gpu

    python build_multimodal_engine.py --model_type blip2 --model_path ${BLIP2_OPT_2_7B} --max_batch_size 8

    popd # examples/models/core/multimodal

fi

if [ "$MODEL" = "llava" ]; then

    pushd examples/models/core/multimodal

    echo "Convert LLaMA from HF"
    python3 ../llama/convert_checkpoint.py --model_dir ${LLAVA_7B} --dtype float16 --output_dir ./c-model/llava-1.5-7b-hf/fp16

    echo "LLAVA builder"
    trtllm-build --checkpoint_dir ./c-model/llava-1.5-7b-hf/fp16 \
                --gemm_plugin float16 \
                --max_batch_size 8 \
                --max_input_len 2048 \
                --max_seq_len 2560 \
                --max_multimodal_len 4608 \
                --output_dir trt_engines/llava-1.5-7b-hf/fp16/1-gpu

    python build_multimodal_engine.py --model_path ${LLAVA_7B} --model_type llava --max_batch_size 8

    popd # examples/models/core/multimodal

fi

if [ "$MODEL" = "llava_fp8" ]; then

    pushd examples/models/core/multimodal

    echo "Convert LLaVA from HF to FP8"
    python3 ../../../quantization/quantize.py \
                --model_dir ${LLAVA_7B} \
                --dtype float16 \
                --qformat fp8 \
                --kv_cache_dtype fp8 \
                --output_dir ./c-model/llava-1.5-7b-hf/fp8 \
                --calib_size 512

    echo "LLAVA builder for FP8"
    trtllm-build --checkpoint_dir ./c-model/llava-1.5-7b-hf/fp8 \
                --gemm_plugin auto \
                --max_batch_size 8 \
                --max_input_len 2048 \
                --max_seq_len 2560 \
                --max_multimodal_len 4608 \
                --output_dir trt_engines/llava-1.5-7b-hf/fp8/1-gpu

    python build_multimodal_engine.py --model_path ${LLAVA_7B} --model_type llava --max_batch_size 8

    popd # examples/models/core/multimodal

fi

if [ "$MODEL" = "vila" ]; then

    echo "Install vila requirements"
    pip install -r $LLM_BACKEND_ROOT/all_models/multimodal/requirements-vila.txt

    pushd examples/models/core/multimodal

    echo "Convert LLaMA from HF"
    python3 ../llama/convert_checkpoint.py --model_dir ${VILA1_5_3B} --dtype float16 --output_dir ./c-model/vila1.5-3b/fp16

    echo "VILA builder"
    trtllm-build --checkpoint_dir ./c-model/vila1.5-3b/fp16 \
                --gemm_plugin float16 \
                --max_batch_size 8 \
                --max_input_len 2048 \
                --max_seq_len 2560 \
                --max_multimodal_len 6272 \
                --output_dir trt_engines/vila1.5-3b/fp16/1-gpu

    python build_multimodal_engine.py --model_path ${VILA1_5_3B} --model_type vila --max_batch_size 32 --vila_path ${VILA_PATH}

    popd # examples/models/core/multimodal
fi

if [ "$MODEL" = "mllama" ]; then

    echo "Install mllama requirements"
    pip install -r $LLM_BACKEND_ROOT/all_models/multimodal/requirements-mllama.txt

    pushd examples/models/core/multimodal

    echo "Convert mllama from HF"
    python3 ../mllama/convert_checkpoint.py --model_dir ${LLAMA_3_2_11B_VISION} --dtype bfloat16 --output_dir ./c-model/Llama-3.2-11B-Vision-Instruct/bf16

    echo "mllama builder"
    trtllm-build --checkpoint_dir ./c-model/Llama-3.2-11B-Vision-Instruct/bf16 \
                --gemm_plugin auto \
                --max_batch_size 8 \
                --max_input_len 2048 \
                --max_seq_len 2560 \
                --max_encoder_input_len 8200 \
                --output_dir trt_engines/Llama-3.2-11B-Vision-Instruct/bf16/1-gpu

    python build_multimodal_engine.py --model_path ${LLAMA_3_2_11B_VISION} \
                                  --model_type mllama --max_batch_size 8 \
                                  --output_dir tmp/trt_engines/Llama-3.2-11B-Vision-Instruct/multimodal_encoder

    popd # examples/models/core/multimodal

fi

if [ "$MODEL" = "whisper" ]; then

    pushd examples/models/core/whisper

    echo "Convert OpenAI Whisper Checkpoint"
    python3 convert_checkpoint.py --model_dir ${WHISPER_LAREGE_V3} --output_dir ./c-model/${MODEL}/tllm_checkpoint

    echo "Build Whisper Encoder: "
    trtllm-build --checkpoint_dir ./c-model/${MODEL}/tllm_checkpoint/encoder \
                --output_dir trt_engine/${MODEL}/encoder \
                --moe_plugin disable \
                --max_batch_size 8 \
                --gemm_plugin disable \
                --bert_attention_plugin float16 \
                --max_input_len 3000 --max_seq_len=3000

    echo "Build Whisper Decoder: "
    trtllm-build  --checkpoint_dir ./c-model/${MODEL}/tllm_checkpoint/decoder \
                --output_dir trt_engine/${MODEL}/decoder \
                --moe_plugin disable \
                --max_beam_width 4 \
                --max_batch_size 8 \
                --max_seq_len 114 \
                --max_input_len 14 \
                --max_encoder_input_len 3000 \
                --gemm_plugin float16 \
                --bert_attention_plugin float16 \
                --gpt_attention_plugin float16
    popd # examples/models/core/whisper

fi

if [ "$MODEL" = "llava_onevision" ]; then

    echo "Install llava_onevision requirements"
    pip install -r $LLM_BACKEND_ROOT/all_models/multimodal/requirements-llava-onevision.txt

    pushd examples/models/core/multimodal

    echo "Convert Qwen from HF"
    python3 ../qwen/convert_checkpoint.py --model_dir ${LLAVA_ONEVISION_7B} --dtype float16 --output_dir ./c-model/llava-7b/fp16

    echo "Qwen builder"
    trtllm-build --checkpoint_dir ./c-model/llava-7b/fp16 \
                --gemm_plugin float16 \
                --max_batch_size 1 \
                --max_input_len 7500 \
                --max_seq_len 7600 \
                --max_multimodal_len 7300 \
                --output_dir trt_engines/llava-onevision-7b/fp16/1-gpu

    python build_multimodal_engine.py --model_path ${LLAVA_ONEVISION_7B} --model_type llava_onevision --max_batch_size 16 --output_dir tmp/trt_engines/llava-onevision-qwen2-7b-ov-hf/multimodal_encoder

    popd # examples/models/core/multimodal

fi

if [ "$MODEL" = "qwen2_vl" ]; then
    echo "Install llava_onevision requirements"
    pip install -r $LLM_BACKEND_ROOT/all_models/multimodal/requirements-qwen2vl.txt

    pushd examples/models/core/multimodal

    echo "Convert Qwen from HF"
    python3 ../qwen/convert_checkpoint.py --model_dir ${QWEN2_VL_7B} --dtype float16 --output_dir ./c-model/qwen2-vl-7b/fp16

    echo "Qwen builder"
    trtllm-build --checkpoint_dir ./c-model/qwen2-vl-7b/fp16 \
                --gemm_plugin float16 \
                --max_batch_size 4 \
                --max_input_len 2048 \
                --max_seq_len 76307200 \
                --max_multimodal_len 1296 \
                --output_dir trt_engines/qwen2-vl-7b/fp16/1-gpu

    python build_multimodal_engine.py --model_path ${QWEN2_VL_7B} --model_type qwen2_vl --output_dir tmp/trt_engines/Qwen2-VL-7B-Instruct/multimodal_encoder

    popd # examples/models/core/multimodal

fi

popd # $LLM_ROOT
