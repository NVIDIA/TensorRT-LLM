#!/bin/bash
set -e


MODEL_PATH="EleutherAI/gpt-j-6B"
MODEL_TYPE=gptj
QFORMAT=$1
if [ -z "$QFORMAT" ]; then
    # If ALGO is empty, set it to "fp8"
    QFORMAT="fp8"
fi


case $QFORMAT in
    fp8|int8_sq)
        ;;
    *)
        echo "Usage: \`gptj_example.sh <quantization_format>\`" >&2
        echo "Unknown quantization-format argument: Expected one of: [fp8, int8_sq]" >&2
        exit 1
esac

CALIB_NUM_BATCHES=512
# Please reduce the following parameters to save GPU memory if a GPU out-of-memory error occur.
BUILD_MAX_INPUT_LEN=2048
BUILD_MAX_OUTPUT_LEN=512
BUILD_MAX_BATCH_SIZE=8

script_dir="$(dirname "$(readlink -f "$0")")"

pushd $script_dir/..

SAVE_PATH=$(pwd)/saved_models_${MODEL_TYPE}_${QFORMAT}
MODEL_CONFIG=${SAVE_PATH}/${MODEL_TYPE}_tp1.json
ENGINE_DIR=${SAVE_PATH}/${MODEL_TYPE}_engine

if ! [ -f $MODEL_CONFIG ]; then
    echo "Quantizing original model..."
    python hf_ptq.py \
        --pyt_ckpt_path=$MODEL_PATH \
        --export_path=$SAVE_PATH \
        --qformat=$QFORMAT \
        --calib_size=$CALIB_NUM_BATCHES
else
    echo "Quantized model config $MODEL_CONFIG exists, skipping the quantization stage"
fi

echo "Building tensorrt_llm engine from AMMO-quantized model..."

python ammo_to_tensorrt_llm.py \
    --model_config=$MODEL_CONFIG \
    --engine_dir=$ENGINE_DIR \
    --max_input_len=$BUILD_MAX_INPUT_LEN \
    --max_output_len=$BUILD_MAX_OUTPUT_LEN \
    --max_batch_size=$BUILD_MAX_BATCH_SIZE

echo "Evaluating the built TRT engine..."

python summarize.py \
    --engine_dir=$ENGINE_DIR \
    --hf_model_location=$MODEL_PATH \
    --tokenizer=$MODEL_TYPE \
    --data_type=fp16 \
    --test_trt_llm \
    --tensorrt_llm_rouge1_threshold=13

echo "Evaluating the original model for comparison..."

python summarize.py \
    --hf_model_location=$MODEL_PATH \
    --tokenizer=$MODEL_TYPE \
    --data_type=fp16 \
    --test_hf \
    --tensorrt_llm_rouge1_threshold=13

popd
