#!/bin/bash

# This script runs genai-perf to profile a multimodal model.
# Supports two modes: concurrency or request_rate

# --- Command Line Arguments Parsing ---
usage() {
    echo "Usage: $0 [--concurrency <value> | --request-rate <value>] --port <port>"
    echo ""
    echo "Options:"
    echo "  --concurrency <value>    Run in concurrency mode with specified concurrency level"
    echo "  --request-rate <value>   Run in request rate mode with specified rate (requests/sec)"
    echo "  --port <port>           Server port number (e.g., 8001, 8003)"
    echo ""
    echo "Examples:"
    echo "  $0 --concurrency 2 --port 8003"
    echo "  $0 --request-rate 15 --port 8001"
    echo "  $0 --concurrency 1 --port 9000"
    exit 1
}

# Initialize variables
MODE=""
CONCURRENCY=""
REQUEST_RATE=""
PORT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --concurrency)
            MODE="concurrency"
            CONCURRENCY="$2"
            shift 2
            ;;
        --request-rate)
            MODE="request_rate"
            REQUEST_RATE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate arguments
if [ -z "$MODE" ]; then
    echo "Error: Must specify either --concurrency or --request-rate"
    usage
fi

if [ -z "$PORT" ]; then
    echo "Error: Must specify --port"
    usage
fi

# Validate PORT
if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || [ "${PORT}" -lt 1 ] || [ "${PORT}" -gt 65535 ]; then
    echo "Error: PORT must be a valid port number (1-65535)"
    echo "You provided: '${PORT}'"
    exit 1
fi

# Validate and set mode-specific values
if [ "${MODE}" = "concurrency" ]; then
    if ! [[ "${CONCURRENCY}" =~ ^[0-9]+$ ]] || [ "${CONCURRENCY}" -lt 1 ]; then
        echo "Error: CONCURRENCY must be a positive integer"
        echo "You provided: '${CONCURRENCY}'"
        exit 1
    fi
    if [ "${CONCURRENCY}" -gt 1 ]; then
        REQUEST_COUNT=$((CONCURRENCY*5))
    else
        REQUEST_COUNT=$((CONCURRENCY*50))
    fi
    echo "Running in CONCURRENCY mode: CONCURRENCY=${CONCURRENCY}, REQUEST_COUNT=${REQUEST_COUNT}, PORT=${PORT}"
elif [ "${MODE}" = "request_rate" ]; then
    if ! [[ "${REQUEST_RATE}" =~ ^[0-9]+$ ]] || [ "${REQUEST_RATE}" -lt 1 ]; then
        echo "Error: REQUEST_RATE must be a positive integer"
        echo "You provided: '${REQUEST_RATE}'"
        exit 1
    fi
    REQUEST_COUNT=$((REQUEST_RATE*10))
    echo "Running in REQUEST_RATE mode: REQUEST_RATE=${REQUEST_RATE}, REQUEST_COUNT=${REQUEST_COUNT}, PORT=${PORT}"
fi

ISL=64
OSL=64

# --- Configuration for genai-perf ---
MODEL_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
TOKENIZER_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
SERVICE_KIND="openai"
ENDPOINT_TYPE="multimodal"
INPUT_FILE="./mm_data_oai.json"
SERVER_URL="localhost:${PORT}"

# Set append name based on port
if [ "${PORT}" = "8003" ]; then
    APPEND_NAME="disagg"
elif [ "${PORT}" = "8001" ]; then
    APPEND_NAME="agg"
else
    APPEND_NAME="port${PORT}"
fi

if [ "${MODE}" = "concurrency" ]; then
    PROFILE_EXPORT_FILE="ISL_${ISL}_OSL_${OSL}_CONCURRENCY_${CONCURRENCY}_${APPEND_NAME}.json"
else
    PROFILE_EXPORT_FILE="ISL_${ISL}_OSL_${OSL}_RATE_${REQUEST_RATE}_${APPEND_NAME}.json"
fi

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
CMD="${CMD} --output-tokens-mean ${OSL}"
#CMD="${CMD} --output-tokens-stddev ${OUTPUT_TOKENS_STDDEV}"
CMD="${CMD} --request-count ${REQUEST_COUNT}"
CMD="${CMD} --profile-export-file \"${PROFILE_EXPORT_FILE}\""
CMD="${CMD} --url \"${SERVER_URL}\""
CMD="${CMD} --random-seed ${RANDOM_SEED}"

# --- Mode-specific flags ---
if [ "${MODE}" = "concurrency" ]; then
    CMD="${CMD} --num-prompts ${CONCURRENCY}"
    CMD="${CMD} --concurrency ${CONCURRENCY}"
    echo "Added concurrency flags: --num-prompts ${CONCURRENCY} --concurrency ${CONCURRENCY}"
elif [ "${MODE}" = "request_rate" ]; then
    CMD="${CMD} --request-rate ${REQUEST_RATE}"
    echo "Added request rate flag: --request-rate ${REQUEST_RATE}"
fi

CMD="${CMD} --image-width-mean 512"
CMD="${CMD} --image-width-stddev 0"
CMD="${CMD} --image-height-mean 512"
CMD="${CMD} --image-height-stddev 0"
CMD="${CMD} --image-format png"
CMD="${CMD} --synthetic-input-tokens-mean ${ISL}"
CMD="${CMD} --synthetic-input-tokens-stddev 0"

if [ "${ADD_STREAMING_FLAG}" = true ] ; then
    CMD="${CMD} --streaming"
fi
CMD="${CMD} --extra-inputs \"max_tokens:${OSL}\""
CMD="${CMD} --extra-inputs \"min_tokens:${OSL}\""
CMD="${CMD} --extra-inputs \"ignore_eos:true\""
CMD="${CMD} -- -v"
CMD="${CMD} --max-threads 1"

# --- Execute the command ---
echo "Executing command:"
echo "${CMD}"
eval "${CMD}"

# Example usage:
# ./test_client_disag_mm.sh --concurrency 2 --port 8003
# ./test_client_disag_mm.sh --request-rate 15 --port 8001