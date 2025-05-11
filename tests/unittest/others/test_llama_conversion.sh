set -ex

export PATH=~/.local/bin/:$PATH # trtllm-build is inside ~/.local/bin
export MODEL=/home/scratch.trt_llm_data/llm-models/llama-models/llama-7b-hf/

test_fake_config() {
    python3 convert_checkpoint.py --dtype float16 --n_layer 2 --output_dir ./c-model/llama-7b/fp16
    trtllm-build --model_config ./c-model/llama-7b/fp16/config.json  \
        --gemm_plugin float16 \
        --max_batch_size 8 \
        --output_dir ./llama_nlayer_2
    python3 ../run.py --max_output_len=1 --engine_dir ./llama_nlayer_2
}

test_meta() {
    python convert_checkpoint.py --meta_ckpt_dir /home/scratch.trt_llm_data/llm-models/llama-models-v2/7B/ --output_dir ./tllm_checkpoint/llama-v2-7b-ckpt-from-meta --tp_size 2
    trtllm-build --checkpoint_dir ./tllm_checkpoint/llama-v2-7b-ckpt-from-meta  --output_dir ./trt_engines/llama-v2-7b-engine-tp2-meta  --gemm_plugin float16
    mpirun -n 2 --allow-run-as-root \
    python ../summarize.py --test_trt_llm \
                           --tensorrt_llm_rouge1_threshold 18 \
                           --hf_model_dir /home/scratch.trt_llm_data/llm-models/llama-models-v2/llama-v2-7b-hf/ \
                           --data_type fp16 \
                           --engine_dir ./trt_engines/llama-v2-7b-engine-tp2-meta \
                           --test_hf
}


test_hf() {
    python convert_checkpoint.py --model_dir ${MODEL} --output_dir ./tllm_checkpoint/tp2_hf  --tp_size 2 --workers 2
    trtllm-build --checkpoint_dir ./tllm_checkpoint/tp2_hf --output_dir ./trt_engines/llama-v2-7b-engine-tp2  --gemm_plugin float16
    mpirun -n 2 --allow-run-as-root \
    python ../summarize.py --test_trt_llm \
                            --tensorrt_llm_rouge1_threshold 18 \
                            --hf_model_dir ${MODEL} \
                            --data_type fp16 \
                            --engine_dir ./trt_engines/llama-v2-7b-engine-tp2 \
                            --test_hf
}


test_hf_by_shard() {
    python convert_checkpoint.py --model_dir ${MODEL} --output_dir ./tllm_checkpoint/tp2_hf-by-shard  --tp_size 2 --workers 2 --load_by_shard
    trtllm-build --checkpoint_dir ./tllm_checkpoint/tp2_hf-by-shard --output_dir ./trt_engines/llama-v2-7b-engine-tp2-by-shard  --gemm_plugin float16
    mpirun -n 2 --allow-run-as-root \
    python ../summarize.py --test_trt_llm \
                            --tensorrt_llm_rouge1_threshold 18 \
                            --hf_model_dir ${MODEL} \
                            --data_type fp16 \
                            --engine_dir ./trt_engines/llama-v2-7b-engine-tp2-by-shard \
                            --test_hf
}


test_wo_int8() {
    python convert_checkpoint.py --model_dir ${MODEL} \
                              --output_dir ./tllm_checkpoint/1gpu_fp16_wq \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int8 \
                              --int8_kv_cache
    trtllm-build --checkpoint_dir ./tllm_checkpoint/1gpu_fp16_wq \
            --output_dir trt_engines/int8_kv_cache_weight_only/1-gpu \
            --gemm_plugin float16 \

    python ../summarize.py --test_trt_llm \
                       --hf_model_dir ${MODEL} \
                       --data_type fp16 \
                       --engine_dir trt_engines/int8_kv_cache_weight_only/1-gpu \
                       --test_hf
}

test_sq() {
    python3 convert_checkpoint.py --model_dir ${MODEL} --output_dir ./tllm_checkpoint/sq --dtype float16 --smoothquant 0.5 --int8_kv_cache
    trtllm-build --checkpoint_dir ./tllm_checkpoint/sq  --output_dir ./trt_engines/sq  --gemm_plugin float16
    python ../summarize.py --test_trt_llm --hf_model_dir ${MODEL}  --data_type fp16  --engine_dir trt_engines/sq  --test_hf
}


test_gptq() {
    python convert_checkpoint.py --model_dir ${MODEL} \
                                 --output_dir ./tllm_checkpoint/2gpu_gptq \
                                 --dtype float16 \
                                 --quant_ckpt_path /home/scratch.trt_llm_data/llm-models/int4-quantized-gptq-awq/llama-7b-4bit-gs128.safetensors \
                                 --use_weight_only \
                                 --weight_only_precision int4_gptq \
                                 --per_group \
                                 --tp_size 2 \
                                 --workers 2

    trtllm-build --checkpoint_dir ./tllm_checkpoint/2gpu_gptq \
             --output_dir ./trt_engines/gptq \
             --gemm_plugin float16

    mpirun -n 2 --allow-run-as-root \
    python ../summarize.py --test_trt_llm \
                       --hf_model_dir ${MODEL} \
                       --data_type fp16 \
                       --engine_dir trt_engines/gptq \
                       --test_hf
}

test_lora() {
    lora_dir=/home/scratch.trt_llm_data/llm-models/llama-models-v2/chinese-llama-2-lora-13b
    python convert_checkpoint.py --model_dir /home/scratch.trt_llm_data/llm-models/llama-models-v2/llama-v2-13b-hf \
                         --output_dir ./tllm_checkpoint/2gpu_lora \
                         --dtype float16 \
                         --tp_size 2

    trtllm-build --checkpoint_dir   ./tllm_checkpoint/2gpu_lora \
            --output_dir ./trt_engines/llama-v2-13b-with-lora \
            --gemm_plugin float16 \
            --lora_plugin float16 \
            --lora_dir ${lora_dir} \
            --max_batch_size 1 \
            --max_input_len 512 \
            --max_seq_len 562

    mpirun -n 2 --allow-run-as-root \
    python ../run.py --engine_dir ./trt_engines/llama-v2-13b-with-lora \
              --max_output_len 50 \
              --tokenizer_dir ${lora_dir} \
              --input_text "今天天气很好，我到公园的时候，" \
              --lora_task_uids 0 \
              --no_add_special_tokens \
              --use_py_session
}

test_mixtral() {
    python convert_checkpoint.py --model_dir /home/scratch.trt_llm_data/llm-models/Mixtral-8x7B-v0.1/ \
                                 --output_dir ./tllm_checkpoint/mixtral_2gpu \
                                 --dtype float16 \
                                 --pp_size 2 \

    trtllm-build --checkpoint_dir ./tllm_checkpoint/mixtral_2gpu \
                    --output_dir ./trt_engines/mixtral/pp2 \
                    --gemm_plugin float16
}

test_long_alpaca_rope_scaling() {
    python convert_checkpoint.py --model_dir /home/scratch.trt_llm_data/llm-models/LongAlpaca-7B/ \
                            --output_dir ./tllm_checkpoint/long_alpaca_tp2 \
                            --dtype float16 \
                            --tp_size 2
    trtllm-build --checkpoint_dir ./tllm_checkpoint/long_alpaca_tp2 \
                 --output_dir ./trt_engines/long_alpaca_tp2 \
                 --gemm_plugin float16 \
                --max_input_len 32768 \

    mpirun -n 2 --allow-run-as-root \
    python ../run.py \
        --max_output_len 128 \
        --max_input_length 32768 \
        --input_file ../../tests/integration/test_input_files/pg64317_sanitized.txt \
        --engine_dir ./trt_engines/long_alpaca_tp2  \
        --tokenizer_dir /home/scratch.trt_llm_data/llm-models/LongAlpaca-7B/
}

test_llava() {
    python ../llama/convert_checkpoint.py \
        --model_dir /home/scratch.trt_llm_data/llm-models/llava-1.5-7b-hf/ \
        --output_dir ./trt_checkpoint/llava-1gpu \
        --dtype float16

    trtllm-build \
        --checkpoint_dir ./trt_checkpoint/llava-1gpu \
        --output_dir ./trt_engines/llava/fp16/1-gpu \
        --gemm_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_seq_len 2560 \
        --max_multimodal_len 576 # 1 (max_batch_size) * 576 (num_visual_features)
}

test_bfloat16() {
    python convert_checkpoint.py --output_dir ./tllm_checkpoint/llama_v2-summarization/bfloat16/1-gpu --dtype=bfloat16 --tp_size=1 --pp_size=1 --model_dir /home/scratch.trt_llm_data/llm-models/llama-models-v2/llama-v2-7b-hf
}

test_all()
{
    test_fake_config
    test_meta
    test_hf
    test_wo_int8
    test_sq
    test_gptq
    test_lora
    test_mixtral
    test_long_alpaca_rope_scaling
    test_llava
}
