
export PATH=$PATH:/home/dev_user/.local/bin/

# BASE_MODEL=/llm-models/llama-3.2-models/Llama-3.2-1B-FP8
# BASE_MODEL=/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct
BASE_MODEL=/scratch/weights/Llama-3.1-8B-Instruct-FP8
dataset_file=/scratch/data/test_bench.txt
isl=1024
osl=2048
num_prompts=10000
BS=64

LOGDIR=/scratch/data/log

mkdir -p $LOGDIR

python /scratch/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py \
            --stdout --tokenizer $BASE_MODEL \
            token-norm-dist \
            --input-mean $isl --output-mean $osl \
            --input-stdev 0 --output-stdev 0 \
            --num-requests $num_prompts > $dataset_file

trtllm-bench -m $BASE_MODEL \
            --model_path $BASE_MODEL \
            throughput \
            --backend pytorch \
            --max_batch_size $BS \
            --max_num_tokens 3072 \
            --dataset ${dataset_file} \
            --kv_cache_free_gpu_mem_fraction 0.8 \
            --warmup 1 \
            --streaming \
            --num_requests $((BS*5)) \
            --concurrency $BS \
            --tp 1 > $LOGDIR/out.log
