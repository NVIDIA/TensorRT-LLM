#!/bin/bash
set -e

TRTLLM_ROOT=$(realpath ${TRTLLM_ROOT:-$(pwd)/../../..})
DATA_DIR=$TRTLLM_ROOT/benchmarks/cpp
WORKSPACE=$TRTLLM_ROOT/tests/llmapi/_perf_evaluator

echo "TRTLLM_ROOT: $TRTLLM_ROOT"
echo "WORKSPACE: $WORKSPACE"

HF_MODEL_DIR=""
TP_SIZE=1
ISL="128,2048"
OSL="128,2048"
RETURN_CONTEXT_LOGITS=0
RETURN_GENERATION_LOGITS=0
KVCACHE_MEM_FRACTION=0.95
MAX_BATCH_SIZE=2048
MAX_NUM_TOKENS=8192

function usage() {
  echo "Usage: $0 -m <hf_model_dir> -t <tp_size> -i <isl> -o <osl>"
  echo "  -m: Path to the model"
  echo "  -t: TP size"
  echo "  -i: Input sequence length, comma separated"
  echo "  -o: Output sequence length, comma separated"
  echo "  -b: Max batch size"
  echo "  -n: Max number of tokens"
  echo "  -c: Return context logits"
  echo "  -g: Return generation logits"
  echo "  -h: Display this help message"
  exit 1
}

PARSED_OPTIONS=$(getopt -n "$0" -o m:t:i:o:b:n:h:cg -l "model:,tp-size:,isl:,osl:,max-batch-size:,max-num-tokens:,help:,context-logit,generation-logit" -- "$@")

if [ $? -ne 0 ]; then
  usage
fi

function parse_args() {
  eval set -- "$PARSED_OPTIONS"

  while true; do
    case "$1" in
      -m|--model)
        HF_MODEL_DIR=$2
        shift 2
        ;;
      -t|--tp-size)
        TP_SIZE=$2
        shift 2
        ;;
      -i|--isl)
        ISL=$2
        shift 2
        ;;
      -o|--osl)
        OSL=$2
        shift 2
        ;;
      -b|--max-batch-size)
        MAX_BATCH_SIZE=$2
        shift 2
        ;;
      -n|--max-num-tokens)
        MAX_NUM_TOKENS=$2
        shift 2
        ;;
      -h|--help)
        usage
        ;;
      -c|--context-logit)
        RETURN_CONTEXT_LOGITS=1
        shift
        ;;
      -g|--generation-logit)
        RETURN_GENERATION_LOGITS=1
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Internal error!"
        exit 1
        ;;
    esac
  done
}


function assert_not_empty () {
  if [ -z "$1" ]; then
    echo "Empty value found"
    exit 1
  fi
}

parse_args

echo "HF_MODEL_DIR: $HF_MODEL_DIR"
echo "TP_SIZE: $TP_SIZE"
echo "ISL: $ISL"
echo "OSL: $OSL"
echo "RETURN_CONTEXT_LOGITS: $RETURN_CONTEXT_LOGITS"
echo "RETURN_GENERATION_LOGITS: $RETURN_GENERATION_LOGITS"

assert_not_empty $HF_MODEL_DIR
assert_not_empty $TP_SIZE
assert_not_empty $ISL
assert_not_empty $OSL

MODEL=$(basename $HF_MODEL_DIR)
SCRATCH_ROOT=${TRTLLM_SCRATCH_ROOT:-$PWD/_scratch-$MODEL}
JSON_REPORTS_DIR=$SCRATCH_ROOT/reports-$TP_SIZE
LOGS_DIR=$SCRATCH_ROOT/logs-$TP_SIZE
NUM_SAMPLES=2000

ENGINE_DIR="/tmp/engine-${MODEL}-tp${TP_SIZE}"

if [ -z "$MODEL" ]; then
  echo "MODEL is empty"
  exit 1
fi

rm -rf $LOGS_DIR
rm -rf $JSON_REPORTS_DIR

# Display the settings
echo "TRTLLM_ROOT: $TRTLLM_ROOT"
echo "TP_SIZE: $TP_SIZE"
echo "SCRATCH_ROOT: $SCRATCH_ROOT"

mkdir -p $JSON_REPORTS_DIR
mkdir -p $LOGS_DIR

function build_engine_using_cli() {
  cd $TRTLLM_ROOT

  local CKPT_DIR=/tmp/ckpt-${MODEL}-tp${TP_SIZE}

  if [ ! -e $CKPT_DIR ]; then
      python examples/llama/convert_checkpoint.py \
          --model_dir $HF_MODEL_DIR \
          --output_dir $CKPT_DIR \
          --tp_size $TP_SIZE \
          --workers $TP_SIZE
  fi

  if [ ! -e $ENGINE_DIR ]; then
      gather_context_logits=""
      if [ $RETURN_CONTEXT_LOGITS -eq 1 ]; then
          gather_context_logits="--gather_context_logits"
      fi

      gather_generation_logits=""
      if [ $RETURN_GENERATION_LOGITS -eq 1 ]; then
          gather_generation_logits="--gather_generation_logits"
      fi

      set -x
      python $TRTLLM_ROOT/tensorrt_llm/commands/build.py \
          --max_seq_len 4096 \
          --checkpoint_dir $CKPT_DIR \
          --output_dir $ENGINE_DIR \
          --workers $TP_SIZE \
          --max_input_len 2048 \
          $gather_context_logits \
          $gather_generation_logits \
          --max_batch_size $MAX_BATCH_SIZE \
          --max_num_tokens $MAX_NUM_TOKENS
      set +x
  fi
}

function generate_data() {
  local isl=$1
  local osl=$2
  local samples=$3
  cd $DATA_DIR

  local data_path=$DATA_DIR/data.$samples.$isl.$osl.json

  echo "TRTLLM_ROOT: $TRTLLM_ROOT"
  echo "Working on $PWD"

  local prepare_dataset="prepare_dataset.py"

  python3 $prepare_dataset \
    --output $data_path \
    --tokenizer $HF_MODEL_DIR \
    token-norm-dist \
    --num-requests $samples \
    --input-mean $isl \
    --output-mean $osl \
    --input-stdev 0 \
    --output-stdev 0
}

function run_perf() {
  local isl=$1
  local osl=$2
  local samples=$3
  local streaming=$4

  local data_path=$DATA_DIR/data.$samples.$isl.$osl.json
  local cpp_benchmark=$TRTLLM_ROOT/cpp/build/benchmarks/gptManagerBenchmark
  local log_path=$LOGS_DIR/$samples.$isl.$osl.log
  local report_path=$WORKSPACE/reports/$samples.$isl.$osl
  local evaluator_py=$WORKSPACE/llmapi_evaluator.py

  cd $TRTLLM_ROOT/tests/llmapi

  local streaming_suffix=""
  if [ "$streaming" = true ]; then
    streaming_suffix="--streaming"
  fi

  rm -f $log_path
  mkdir -p $report_path

  python $evaluator_py benchmark \
  --model-path $ENGINE_DIR \
  --tp-size $TP_SIZE \
  --samples-path $data_path \
  --num-samples $samples \
  --report-path-prefix $report_path \
  --warmup 100 \
  --return-context-logits $RETURN_CONTEXT_LOGITS \
  --return-generation-logits $RETURN_GENERATION_LOGITS \
   ${streaming_suffix} \
   --kv-cache-free-gpu-mem-fraction $KVCACHE_MEM_FRACTION \
   --cpp-executable $cpp_benchmark \
  2>&1 | tee -a $log_path

  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Python command failed. Exiting..."
    exit 1
  fi
}

function extractlog() {
  local log_path=$1
  local report_path=$2

  cd $WORKSPACE

  cat $log_path | ./extract_log.py > $report_path
}

# ANSI color codes
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
BLUE='\033[34m'
NC='\033[0m' # No Color

function run() {
  local isl=$1
  local osl=$2
  local samples=$3
  local streaming=$4

  local streaming_suffix=""
  if [ "$streaming" = true ]; then
    streaming_suffix="-streaming"
  fi

  local log_path=$LOGS_DIR/$samples.$isl.$osl.log

  echo -e "${GREEN}Running with isl: $isl, osl: $osl, samples: $samples, streaming: $streaming${NC}"

  generate_data $isl $osl $samples
  run_perf $isl $osl $samples $streaming
  extractlog $log_path $JSON_REPORTS_DIR/report-$samples.$isl.$osl$streaming_suffix.json
}

function parse_final_report {
  cd $WORKSPACE
  echo "Parsing CSV result"
  echo "--------------------------------------------------------"
  echo
  ./parse_json_report.py $JSON_REPORTS_DIR --output-file $LOGS_DIR/output.csv
  cat $LOGS_DIR/output.csv
  echo
  echo "--------------------------------------------------------"
}

# Function to split a comma-separated string into an array
split_into_array() {
    local string="$1"
    IFS=',' read -r -a array <<< "$string"
    echo "${array[@]}"
}

# main
cd $WORKSPACE
chmod +x *.py

# get summary of the GPUs
nvidia-smi --query-gpu=index,name,gpu_uuid,pci.bus_id,power.draw,temperature.gpu,utilization.gpu,memory.total,memory.free,memory.used --format=csv

build_engine_using_cli

ISL=($(split_into_array "$ISL"))
OSL=($(split_into_array "$OSL"))

for isl in "${ISL[@]}"; do
  for osl in "${OSL[@]}"; do
    echo "Running with isl: $isl, osl: $osl"
    run $isl $osl $NUM_SAMPLES true
    run $isl $osl $NUM_SAMPLES false
  done
done

# You can paste the generated CSV string to excel to visualize the results
parse_final_report
