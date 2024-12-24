#!/bin/bash
set -e
set -x

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
KVCACHE_MEM_FRACTION=0.98
MAX_BATCH_SIZE=2048
MAX_NUM_TOKENS=8192
FP8_QUANT=0
EXAMPLE_DIR="llama"
OUTPUT_FILE=$WORKSPACE/"output.txt"

function usage() {
  echo "Usage: $0 -m <hf_model_dir> -t <tp_size> -i <isl> -o <osl>"
  echo "  -m: Path to the model"
  echo "  -t: TP size"
  echo "  -i: Input sequence length, comma separated"
  echo "  -o: Output sequence length, comma separated"
  echo "  -b: Max batch size"
  echo "  -n: Max number of tokens"
  echo "  -r: Example directory"
  echo "  -c: Return context logits"
  echo "  -g: Return generation logits"
  echo "  -q: FP8 quantization"
  echo "  -h: Display this help message"
  echo ""
  echo "Example: ./run.sh -m /gptj -i "128" -o "128" -t 2 -b 2048 -n 2048 -r gptj"
  echo "Example: ./run.sh -m /llama -i "128" -o "128" -t 1 -b 2048 -n 2048 -r llama"
  exit 1
}

# ANSI color codes
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
BLUE='\033[34m'
NC='\033[0m' # No Color


PARSED_OPTIONS=$(getopt -n "$0" -o m:t:i:o:b:n:r:q:h:cg -l "model:,tp-size:,isl:,osl:,max-batch-size:,max-num-tokens:,example:,fp8:,help:,context-logit,generation-logit" -- "$@")

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
      -r|--example)
        EXAMPLE_DIR=$2
        shift 2
        ;;
      -q|--fp8)
        FP8_QUANT=$2
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

FP8_SUFFIX=""
if [ $FP8_QUANT -eq 1 ]; then
  FP8_SUFFIX="-fp8"
fi

MODEL=$(basename $HF_MODEL_DIR)
SCRATCH_ROOT=${TRTLLM_SCRATCH_ROOT:-$PWD/_scratch-${MODEL}${FP8_QUANT}-${ISL}-${OSL}}
JSON_REPORTS_DIR=$SCRATCH_ROOT/reports-$TP_SIZE
LOGS_DIR=$SCRATCH_ROOT/logs-$TP_SIZE
NUM_SAMPLES=2000
MAX_SEQ_LEN=4096

# if isl=128 and osl=128 then NUM_SAMPLES=30000
# if isl=128 and osl=2048 then NUM_SAMPLES=3000
if [ "$ISL" = "128" ] && [ "$OSL" = "128" ]; then
  NUM_SAMPLES=30000
  NUM_SAMPLES=3000 # DEBUG
  MAX_SEQ_LEN=256
fi
if [ "$ISL" = "128" ] && [ "$OSL" = "2048" ]; then
  NUM_SAMPLES=3000
  MAX_SEQ_LEN=2176
fi
if [ "$ISL" = "2048" ] && [ "$OSL" = "128" ]; then
  NUM_SAMPLES=3000
  MAX_SEQ_LEN=2176
fi
if [ "$ISL" = "2048" ] && [ "$OSL" = "2048" ]; then
  NUM_SAMPLES=1500
  MAX_SEQ_LEN=4096
fi

ENGINE_DIR="/tmp/engine-${MODEL}-tp${TP_SIZE}${FP8_SUFFIX}.${ISL}-${OSL}"

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

function get_data_path {
  local isl=$1
  local osl=$2
  local samples=$3
  echo $DATA_DIR/data.$samples.$isl.$osl.json
}

function build_engine_using_cli() {
  cd $TRTLLM_ROOT
  workspace=$ENGINE_DIR
  #local data_path=$(get_data_path $ISL $OSL $NUM_SAMPLES)
  local max_seq_len=$((ISL + OSL))

  generate_data_for_bench $ISL $OSL $NUM_SAMPLES

  local log_file=$workspace/engine_build.log
  if [ ! -d $workspace ]; then
    mkdir -p $workspace
    local cmd="trtllm-bench -m $HF_MODEL_DIR -w $workspace build -tp $TP_SIZE --dataset $(get_data_path $ISL $OSL $NUM_SAMPLES)"
    if [ $FP8_QUANT -eq 1 ]; then
      cmd="$cmd -q FP8"
    fi
    set -x
    # tee to a file
    $cmd 2>&1 | tee $log_file
    set +x
  fi
  ENGINE_DIR=$(grep 'ENGINE SAVED:' $log_file | sed 's/.*SAVED: //')
  echo "Engine saved to $ENGINE_DIR"
}

function generate_data() {
  local isl=$1
  local osl=$2
  local samples=$3
  cd $DATA_DIR

  local data_path=$(get_data_path $isl $osl $samples)

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

function generate_data_for_bench() {
  local isl=$1
   local osl=$2
   local samples=$3
   cd $DATA_DIR

   local data_path=$(get_data_path $isl $osl $samples)

   echo "TRTLLM_ROOT: $TRTLLM_ROOT"
   echo "Working on $PWD"

   local prepare_dataset="prepare_dataset.py"

   python $prepare_dataset \
    --tokenizer=$HF_MODEL_DIR \
    --stdout token-norm-dist \
    --num-requests=$samples \
    --input-mean=$isl \
    --output-mean=$osl --input-stdev=0 --output-stdev=0 > $data_path
}

function run_perf() {
  local isl=$1
  local osl=$2
  local samples=$3
  local streaming=$4

  local data_path=$DATA_DIR/data.$samples.$isl.$osl.json
  local cpp_benchmark=$TRTLLM_ROOT/cpp/build/benchmarks/gptManagerBenchmark
  local log_path=$LOGS_DIR/$samples.$isl.$osl.log
  local report_path=reports/$samples.$isl.$osl
  local evaluator_py=$WORKSPACE/llmapi_evaluator.py

  cd $TRTLLM_ROOT/tests/llmapi

  local streaming_suffix=""
  if [ "$streaming" = true ]; then
    streaming_suffix="--streaming"
  fi

  rm -f $log_path
  mkdir -p $report_path


  generate_data $ISL $OSL $NUM_SAMPLES

  set -x
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
  set +x

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
export NCCL_DEBUG=INFO


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

# Append the result to the result
echo "Appending result to $OUTPUT_FILE"
LOCK_FILE=$WORKSPACE/_lockfile
(
  flock -x 200

  echo "Date: $(date)" >> $OUTPUT_FILE
  echo $MODEL >> $OUTPUT_FILE
  echo "TP: ${TP_SIZE}" >> $OUTPUT_FILE
  echo "FP8: $FP8_QUANT" >> $OUTPUT_FILE
  cat $LOGS_DIR/output.csv >> $OUTPUT_FILE
  echo >> $OUTPUT_FILE
) 200>"$LOCK_FILE"
