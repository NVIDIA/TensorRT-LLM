#!/bin/bash

# This script runs genai-perf to profile a multimodal model.

# --- Configuration for genai-perf ---
MODEL_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
TOKENIZER_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
SERVICE_KIND="openai"
ENDPOINT_TYPE="vision"
INPUT_FILE="./mm_data_oai.json"
OUTPUT_TOKENS_MEAN=128
OUTPUT_TOKENS_STDDEV=0
REQUEST_COUNT=100
REQUEST_RATE=10
PROFILE_EXPORT_FILE="mm_profile_export.json"
SERVER_URL="localhost:8001"
RANDOM_SEED=123
# Set to true if your endpoint supports streaming and you want to test it
ADD_STREAMING_FLAG=true # or true

# --- Build the genai-perf command ---
CMD="genai-perf profile"
CMD="${CMD} -m \"${MODEL_NAME}\""
CMD="${CMD} --tokenizer \"${TOKENIZER_NAME}\""
#CMD="${CMD} --service-kind \"${SERVICE_KIND}\""
CMD="${CMD} --endpoint-type \"${ENDPOINT_TYPE}\""
#CMD="${CMD} --input-file \"${INPUT_FILE}\""
CMD="${CMD} --output-tokens-mean ${OUTPUT_TOKENS_MEAN}"
#CMD="${CMD} --output-tokens-stddev ${OUTPUT_TOKENS_STDDEV}"
CMD="${CMD} --request-count ${REQUEST_COUNT}"
CMD="${CMD} --request-rate ${REQUEST_RATE}"
CMD="${CMD} --profile-export-file \"${PROFILE_EXPORT_FILE}\""
CMD="${CMD} --url \"${SERVER_URL}\""
CMD="${CMD} --random-seed ${RANDOM_SEED}"

CMD="${CMD} --image-width-mean 512"
CMD="${CMD} --image-width-stddev 30"
CMD="${CMD} --image-height-mean 512"
CMD="${CMD} --image-height-stddev 30"
CMD="${CMD} --image-format png"
CMD="${CMD} --synthetic-input-tokens-mean 100"
CMD="${CMD} --synthetic-input-tokens-stddev 0"

if [ "${ADD_STREAMING_FLAG}" = true ] ; then
    CMD="${CMD} --streaming"
fi

# --- Execute the command ---
echo "Executing command:"
echo "${CMD}"
eval "${CMD}"

# Example of how the command would look with direct line continuation (for reference):
# genai-perf profile \
#         -m "llava-hf/llava-v1.6-mistral-7b-hf" \
#         --tokenizer "llava-hf/llava-v1.6-mistral-7b-hf" \
#         --service-kind "openai" \
#         --endpoint-type "chat" \
#         --input-filename "mm_data_oai.jsonl" \
#         --output-tokens-mean 64 \
#         --output-tokens-stddev 0 \
#         --request-count 1 \
#         --request-rate 1 \
#         --profile-export-file "mm_profile_export.json" \
#         --url "http://localhost:8003" \
#         --random-seed 123
#         # --streaming # Uncomment if needed