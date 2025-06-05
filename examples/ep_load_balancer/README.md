# Expert Parallelism Load Balancer (EPLB)

Effective load balancing is crucial when leveraging large-scale expert parallelism. As described in the [DeepSeek-V3 paper](https://arxiv.org/abs/2412.19437), redundant experts can be introduced to rebalance the workload across GPUs. This mechanism is known as the Expert Parallelism Load Balancer ([EPLB](https://github.com/deepseek-ai/EPLB)).

## Offline EP Load Balancer

### Step 1: Run Inference and Collect Statistics

To generate the necessary statistics for load rebalancing, run your model on a target dataset and count the routed expert IDs during inference. Once the counting is complete, the statistics will be saved for further processing. In this example, we use `deepseek-ai/DeepSeek-R1`.

Set up some environment variables:

```bash
export MODEL_NAME=deepseek-ai/DeepSeek-R1
export MODEL_PATH=<YOUR_MODEL_PATH>
# Set the expert statistic data path
export EXPERT_STATISTIC_PATH=./expert_statistic
# Enable counting of routed expert IDs from iteration 100 to iteration 200
export EXPERT_STATISTIC_ITER_RANGE=100-200
```

Prepare a dataset following the [benchmarking documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-benchmarking.md#preparing-a-dataset) and save it as `./dataset.json`.

Run 32-way expert parallelism inference on the prepared dataset. Please refer to the [LLM API MGMN example](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/llm_mgmn_trtllm_bench.sh) for details on running `trtllm-bench` on Slurm.

```bash
cat > ./extra_llm_api_options.yaml <<EOF
enable_attention_dp: true
use_cuda_graph: true
EOF

trtllm-llmapi-launch \
trtllm-bench --model ${MODEL_NAME} \
    --model_path ${MODEL_PATH} \
    throughput \
    --tp 32 \
    --ep 32 \
    --extra_llm_api_options ./extra_llm_api_options.yaml \
    --kv_cache_free_gpu_mem_fraction 0.75 \
    --backend pytorch \
    --dataset ./dataset.json \
    --warmup 0 \
    --eos_id -1
```

After inference, review the dumped statistic files in `$EXPERT_STATISTIC_PATH`. For each layer and iteration, the load imbalance can be measured using simple metrics such as the standard deviation or the imbalance ratio. Given the routed token counts for all ranks, the imbalance ratio is defined as $(max - mean) / mean$, which represents the excessive workload received by the hottest rank. A perfectly balanced load would have an imbalance ratio of 0. Run the [`report_load_statistics.py`](./report_load_statistics.py) script:

```bash
python report_load_statistics.py --expert_statistic_path $EXPERT_STATISTIC_PATH
```

The output would look like:

```txt
Load statistics:
           mean         std  imbalance-ratio
3        1024.0  187.955200         0.498043
4        1024.0  202.728516         0.537602
5        1024.0  209.339981         0.458676
...
58       1024.0  570.880676         2.461014
59       1024.0  341.339447         0.717498
60       1024.0  381.045471         1.119648
average  1024.0  491.651199         1.564272
```

The metrics are computed at each iteration and then averaged. The load imbalance is significant — on average, the hottest rank receives 1.56 times more routed tokens than the mean.

### Step 2: Generate the EPLB Configuration

Use the provided [`generate_eplb_config.py`](./generate_eplb_config.py) script to convert the collected statistics into an EPLB configuration file. Specify the target expert parallelism size (`--ep_size`) and the total number of slots (`--num_slots`) that will be used for deployment. One potential strategy is to maintain 8 expert slots per rank while increasing expert parallelism to 36 ways. This results in 32 redundant experts and 288 expert slots in total.

```bash
python generate_eplb_config.py \
    --ep_size 36 \
    --num_slots 288 \
    --expert_statistic_path $EXPERT_STATISTIC_PATH \
    --output_path ./moe_load_balancer.yaml
```

The `./moe_load_balancer.yaml` file would look like:

```yaml
initial_global_assignments:
  3: [138, 81, 60, ..., 69, 250, 77]
  4: [24, 243, 72, ..., 90, 251, 52]
  5: [120, 162, 246, ..., 14, 192, 171]
  ...
  58: [67, 70, 160, ..., 212, 103, 125]
  59: [45, 142, 152, ..., 99, 205, 49]
  60: [34, 162, 119, ..., 234, 26, 129]
num_slots: 288
layer_updates_per_iter: 0
```

### Step 3: Run Inference with the EPLB Configuration

Set up some environment variables:

```bash
# Set a new expert statistic data path
export EXPERT_STATISTIC_PATH=./expert_statistic_eplb
# Enable counting of routed expert IDs from iteration 100 to iteration 200
export EXPERT_STATISTIC_ITER_RANGE=100-200
```

Run 36-way expert parallelism inference with the EPLB configuration incorporated:

```bash
cat > ./extra_llm_api_options_eplb.yaml <<EOF
enable_attention_dp: true
use_cuda_graph: true
moe_load_balancer: ./moe_load_balancer.yaml
EOF

trtllm-llmapi-launch \
trtllm-bench --model ${MODEL_NAME} \
    --model_path ${MODEL_PATH} \
    throughput \
    --tp 36 \
    --ep 36 \
    --extra_llm_api_options ./extra_llm_api_options_eplb.yaml \
    --kv_cache_free_gpu_mem_fraction 0.75 \
    --backend pytorch \
    --dataset ./dataset.json \
    --warmup 0 \
    --eos_id -1
```

Run the [`report_load_statistics.py`](./report_load_statistics.py) script again:

```bash
python report_load_statistics.py --expert_statistic_path $EXPERT_STATISTIC_PATH
```

The output would look like:

```txt
Load statistics:
           mean        std  imbalance-ratio
3        1024.0  37.612328         0.081947
4        1024.0  42.367714         0.093256
5        1024.0  42.623219         0.092623
...
58       1024.0  49.167507         0.113420
59       1024.0  44.529514         0.092314
60       1024.0  48.408348         0.101029
average  1024.0  53.976442         0.115378
```

Clearly, the load is much more balanced now — on average, the hottest rank receives only about 0.11 times more routed tokens than the mean.

> **Note:** The expert ID counting could significantly hurt performance, so remember to disable it by unsetting `EXPERT_STATISTIC_ITER_RANGE` when running inference for benchmarking or production purposes.


## Online EP Load Balancer
Doc coming soon.
