set -e

# Usage:
# gptnext_example.sh <quantization algorithm>

QFORMAT=$1
if [ -z "$QFORMAT" ]; then
    # If ALGO is empty, set it to "fp8"
    QFORMAT="fp8"
fi

case $QFORMAT in
    fp8|int8_sq)
        ;;
    *)
        echo "Usage: \`gptnext_example.sh <quantization_format>\`" >&2
        echo "Unknown quantization-format argument: Expected one of: [fp8, int8_sq]" >&2
        exit 1
esac

# Please reduce the following parameters to save GPU memory if a GPU out-of-memory error occur.
BUILD_MAX_INPUT_LEN=2048
BUILD_MAX_OUTPUT_LEN=512
BUILD_MAX_BATCH_SIZE=8

script_dir="$(dirname "$(readlink -f "$0")")"

pushd $script_dir/..

# Auto download nemo checkpoint if not present
if [ ! -f "GPT-2B-001_bf16_tp1.nemo" ]; then
    wget -O GPT-2B-001_bf16_tp1.nemo https://huggingface.co/nvidia/GPT-2B-001/resolve/main/GPT-2B-001_bf16_tp1.nemo
fi

tar -xvf GPT-2B-001_bf16_tp1.nemo model_config.yaml

TOKENIZER_FILE=$(cat model_config.yaml | grep tokenizer_model | awk -F':' '{print $NF}' | awk '{$1=$1};1')
echo "Tokenzier file: $TOKENIZER_FILE"

tar -xvf GPT-2B-001_bf16_tp1.nemo $TOKENIZER_FILE

GPT_MODEL_FILE=GPT-2B-001_bf16_tp1.nemo
SAVE_PATH=$(pwd)/saved_models_gptnext_${QFORMAT}
MODEL_CONFIG_PTQ=${SAVE_PATH}/gptnext_tp1.json

if ! [ -f $MODEL_CONFIG_PTQ ]; then
    echo "Quantizing original model..."
    python nemo_ptq.py \
        gpt_model_file=$GPT_MODEL_FILE \
        model_save_path=$SAVE_PATH \
        trainer.devices=1 \
        trainer.num_nodes=1 \
        tensor_model_parallel_size=1 \
        pipeline_model_parallel_size=1 \
        quantization.algorithm=$QFORMAT
else
    echo "Quantized model config $MODEL_CONFIG_PTQ exists, skipping the quantization stage"
fi

python ammo_to_tensorrt_llm.py \
    --model_config $MODEL_CONFIG_PTQ \
    --engine_dir $SAVE_PATH/trt_llm_engines/ \
    --max_input_len=$BUILD_MAX_INPUT_LEN \
    --max_output_len=$BUILD_MAX_OUTPUT_LEN \
    --max_batch_size=$BUILD_MAX_BATCH_SIZE

python summarize.py \
    --engine_dir $SAVE_PATH/trt_llm_engines/ \
    --engine_name ammo_bfloat16_tp1_rank0.engine \
    --batch_size 1 \
    --test_trt_llm \
    --tokenizer gptnext \
    --vocab_file $TOKENIZER_FILE

popd
