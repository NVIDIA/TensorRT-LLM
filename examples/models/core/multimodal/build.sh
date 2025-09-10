export MODEL_NAME="florence-2-large-ft"
export MODEL_TYPE="florence2"
export INFERENCE_PRECISION="float16"
export TP_SIZE=1
export PP_SIZE=1
export WORLD_SIZE=1
export BATCH_SIZE=32
export MAX_BEAM_WIDTH=1
export NUM_VISUAL_FEATURES=577

python ../enc_dec/convert_checkpoint.py --model_type ${MODEL_TYPE} \
    --model_dir /workspace/models/hf_models/${MODEL_NAME} \
    --output_dir /workspace/models/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION} \
    --tp_size ${TP_SIZE} \
    --pp_size ${PP_SIZE} \
    --dtype ${INFERENCE_PRECISION} \
    --workers 1

trtllm-build --checkpoint_dir /workspace/models/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION}/encoder \
    --output_dir /workspace/models/trt_engines/${MODEL_NAME}/${INFERENCE_PRECISION}/encoder \
    --kv_cache_type disabled \
    --moe_plugin disable \
    --max_beam_width ${MAX_BEAM_WIDTH} \
    --max_batch_size ${BATCH_SIZE} \
    --max_input_len 1024 \
    --max_prompt_embedding_table_size $((NUM_VISUAL_FEATURES * BATCH_SIZE)) \
    --gemm_plugin ${INFERENCE_PRECISION} \
    --bert_attention_plugin ${INFERENCE_PRECISION} \
    --bert_context_fmha_fp32_acc enable \
    --gpt_attention_plugin ${INFERENCE_PRECISION} \
    --remove_input_padding enable \
    --workers 1

trtllm-build --checkpoint_dir /workspace/models/trt_models/${MODEL_NAME}/${INFERENCE_PRECISION}/decoder \
    --output_dir /workspace/models/trt_engines/${MODEL_NAME}/${INFERENCE_PRECISION}/decoder \
    --kv_cache_type paged \
    --moe_plugin disable \
    --max_beam_width ${MAX_BEAM_WIDTH} \
    --max_batch_size ${BATCH_SIZE} \
    --max_input_len 1 \
    --max_seq_len 1024 \
    --max_encoder_input_len $((1024 * BATCH_SIZE)) \
    --gemm_plugin ${INFERENCE_PRECISION} \
    --bert_attention_plugin ${INFERENCE_PRECISION} \
    --gpt_attention_plugin ${INFERENCE_PRECISION} \
    --remove_input_padding enable \
    --workers 1

python build_multimodal_engine.py \
        --model_type ${MODEL_TYPE} \
        --model_path /workspace/models/hf_models/${MODEL_NAME} \
        --output_dir /workspace/models/trt_engines/${MODEL_NAME}/${INFERENCE_PRECISION}/vision \
        --max_batch_size ${BATCH_SIZE}